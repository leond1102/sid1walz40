"""Ansteuerung des Greifarms (ESP32-C6) über MQTT.

Der ESP32 führt nur noch aus. Dieses Modul
  * prüft jeden Befehl gegen die Grenzen aus arm_config.py,
  * rechnet Zentimeter in Schritte um,
  * schickt die Bewegungsparameter als retained Konfiguration an den ESP32,
  * verfolgt Zustand, Ereignisse und Heartbeat des ESP32.

Verwendung in eigenen Skripten:
    arm = ArmController(client_id="mein_skript")
    arm.connect()
    arm.home(wait=True)            # wait=True: blockiert, bis der ESP32 fertig meldet
    arm.move_cm(12.5, wait=True)
    arm.gripper_close(wait=True)
    print(arm.has_part)            # True, wenn der Sensor ein Metallteil erkennt
    arm.disconnect()

Ohne wait kehren die Befehle sofort zurück (so arbeitet arm_console.py).
Über on_event kann ein Programm auf Meldungen des ESP32 reagieren, z. B. "part_lost".

Jeder Client braucht eine eigene client_id, sonst trennt der Broker die
Verbindungen gegenseitig.
"""

import importlib
import json
import threading
import time
from typing import Callable

import paho.mqtt.client as mqtt
from loguru import logger

import arm_config as cfg

TOPIC_CMD_AXIS = f"{cfg.TOPIC_BASE}/cmd/axis"
TOPIC_CMD_GRIPPER = f"{cfg.TOPIC_BASE}/cmd/gripper"
TOPIC_CMD_STOP = f"{cfg.TOPIC_BASE}/cmd/stop"
TOPIC_CMD_STATUS = f"{cfg.TOPIC_BASE}/cmd/status"
TOPIC_CMD_CONFIG = f"{cfg.TOPIC_BASE}/cmd/config"
TOPIC_STATE = f"{cfg.TOPIC_BASE}/state"
TOPIC_EVENT = f"{cfg.TOPIC_BASE}/event"
TOPIC_HEARTBEAT = f"{cfg.TOPIC_BASE}/heartbeat"

# Zustände und Ereignisse des ESP32 (siehe Arm_MQTT.ino)
BUSY_STATES = ("homing", "moving", "gripper")
# limit_reached: Achse ist beim Runterfahren auf den Limit-Schalter gefahren (= min-Position),
# der ESP32 setzt die Position auf 0 und steht -> gilt als normales Fahrtende, nicht als Fehler
DONE_EVENTS = ("move_done", "homed", "limit_reached")
FAIL_EVENTS = ("stopped",)  # dazu jedes Ereignis mit level "error"
PART_EVENTS = ("part_gripped", "no_part", "part_lost", "part_back")  # vom Teilesensor


def steps_to_cm(steps: int) -> float:
    """Rechnet Schritte in Zentimeter um."""
    return steps * cfg.CM_PER_REV / cfg.STEPS_PER_REV


def cm_to_steps(cm: float) -> int:
    """Rechnet Zentimeter in Schritte um (gerundet wie zuvor auf dem ESP32)."""
    return int(cm * cfg.STEPS_PER_REV / cfg.CM_PER_REV + 0.5)


def validate_config() -> None:
    """Prüft die Werte aus arm_config.py auf Plausibilität.

    Raises:
        ValueError: Wenn ein Wert fehlt, den falschen Typ hat oder unplausibel ist.
    """
    positive_ints = (
        "AXIS_MAX_STEPS", "HOMING_MAX_STEPS", "BACKOFF_STEPS", "STEP_HALF_US",
        "GRIP_CLOSED_US", "GRIP_OPEN_US", "SERVO_MIN_US", "SERVO_MAX_US",
        "SERVO_TRAVEL_MS", "SENSOR_LOSS_MS", "GRIP_SETTLE_MS", "GRIP_SAMPLES",
        "GRIP_SAMPLE_INTERVAL_MS", "STEPS_PER_REV",
    )
    for name in positive_ints:
        value = getattr(cfg, name, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} muss eine ganze Zahl > 0 sein (ist {value!r})")

    if not isinstance(cfg.AXIS_MIN_STEPS, int) or cfg.AXIS_MIN_STEPS < 0:
        raise ValueError("AXIS_MIN_STEPS muss eine ganze Zahl >= 0 sein")
    if cfg.AXIS_MIN_STEPS >= cfg.AXIS_MAX_STEPS:
        raise ValueError("AXIS_MIN_STEPS muss kleiner als AXIS_MAX_STEPS sein")
    if cfg.CM_PER_REV <= 0:
        raise ValueError("CM_PER_REV muss > 0 sein")
    if not cfg.SERVO_MIN_US <= cfg.GRIP_CLOSED_US < cfg.GRIP_OPEN_US <= cfg.SERVO_MAX_US:
        raise ValueError(
            "Es muss gelten: SERVO_MIN_US <= GRIP_CLOSED_US < GRIP_OPEN_US <= SERVO_MAX_US"
        )


def config_payload() -> str:
    """Baut die Konfigurationsnachricht für den ESP32.

    Die Schlüssel müssen zu CONFIG_KEYS in Arm_MQTT.ino passen.
    """
    values = {
        "step_half_us": cfg.STEP_HALF_US,
        "homing_max": cfg.HOMING_MAX_STEPS,
        "backoff": cfg.BACKOFF_STEPS,
        "servo_min_us": cfg.SERVO_MIN_US,
        "servo_max_us": cfg.SERVO_MAX_US,
        "servo_travel_ms": cfg.SERVO_TRAVEL_MS,
        "boot_grip_us": cfg.GRIP_OPEN_US if cfg.OPEN_GRIPPER_ON_BOOT else 0,
        "grip_settle_ms": cfg.GRIP_SETTLE_MS,
        "grip_samples": cfg.GRIP_SAMPLES,
        "grip_sample_ms": cfg.GRIP_SAMPLE_INTERVAL_MS,
        "sensor_loss_ms": cfg.SENSOR_LOSS_MS,
    }
    return " ".join(f"{key}={value}" for key, value in values.items())


class ArmController:
    """Verbindung zum Greifarm: prüft Grenzen, sendet Befehle, verfolgt den Zustand.

    Attributes:
        state: Letzter empfangener Zustand des ESP32 als Dictionary.
        last_heartbeat: Zeitpunkt des letzten Heartbeats (time.monotonic()).
    """

    def __init__(self, client_id: str = "arm_controller") -> None:
        """
        Args:
            client_id: Eindeutige MQTT-Client-ID dieses Skripts.

        Raises:
            ValueError: Wenn arm_config.py unplausible Werte enthält.
        """
        validate_config()
        self.state: dict = {}
        self.last_heartbeat: float = 0.0
        # Wird bei jedem Ereignis des ESP32 aufgerufen: on_event(level, code, info)
        self.on_event: Callable[[str, str, str], None] | None = None
        self._broker_connected = False
        # Für Befehle mit wait=True: wird gesetzt, sobald der ESP32 fertig meldet
        self._done = threading.Event()
        self._busy_seen = False
        self._failure: str | None = None
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    # ------------------------------------------------------------------
    #  Verbindung
    # ------------------------------------------------------------------
    @property
    def online(self) -> bool:
        """True, wenn Broker verbunden ist und der ESP32 aktuelle Lebenszeichen sendet."""
        if not self._broker_connected:
            return False
        if self.state.get("state") in (None, "offline"):
            return False
        return (time.monotonic() - self.last_heartbeat) < cfg.HEARTBEAT_TIMEOUT_S

    @property
    def has_part(self) -> bool:
        """True, wenn ein Teil im Greifer ist.

        Mit Sensor: letzte Meldung des ESP32. Ohne Sensor (USE_PART_SENSOR = False):
        auf Vertrauensbasis, sobald der Greifer geschlossen ist.
        """
        if not cfg.USE_PART_SENSOR:
            return self.state.get("gripper") == "closed"
        return bool(self.state.get("part"))

    def is_connected(self) -> bool:
        """Gleichbedeutend mit online (einheitlicher Name für alle Geräte)."""
        return self.online

    def connect(self, background: bool = False) -> None:
        """Verbindet mit dem Broker und startet die Netzwerkschleife.

        Args:
            background: True = nicht warten; die Verbindung wird im Hintergrund aufgebaut
                und nach Abbrüchen neu versucht (für main.py, das auch ohne Broker starten soll).

        Raises:
            OSError: Wenn der Broker nicht erreichbar ist (nur ohne background).
        """
        if background:
            self._client.reconnect_delay_set(min_delay=1, max_delay=10)
            self._client.connect_async(cfg.BROKER, cfg.PORT, keepalive=30)
        else:
            self._client.connect(cfg.BROKER, cfg.PORT, keepalive=30)
        self._client.loop_start()

    def disconnect(self) -> None:
        """Trennt die Verbindung sauber."""
        self._client.disconnect()
        self._client.loop_stop()

    # ------------------------------------------------------------------
    #  Konfiguration
    # ------------------------------------------------------------------
    def push_config(self) -> bool:
        """Schickt die Parameter aus arm_config.py retained an den ESP32.

        Returns:
            True, wenn die Nachricht an den Broker ging.
        """
        payload = config_payload()
        logger.info(f"-> config: {payload}")
        result = self._client.publish(TOPIC_CMD_CONFIG, payload, qos=1, retain=True)
        return result.rc == mqtt.MQTT_ERR_SUCCESS

    def reload_config(self) -> bool:
        """Lädt arm_config.py neu, prüft die Werte und schickt sie an den ESP32.

        Returns:
            True, wenn die neue Konfiguration gesendet wurde.
        """
        try:
            importlib.reload(cfg)
            validate_config()
        except Exception as e:  # auch Syntaxfehler in der Config abfangen
            logger.error(f"Config nicht übernommen: {e}")
            return False
        logger.info("arm_config.py neu geladen.")
        return self.push_config()

    # ------------------------------------------------------------------
    #  Senden
    # ------------------------------------------------------------------
    def send(self, topic: str, payload: str, force: bool = False) -> bool:
        """Sendet einen Befehl, sofern der ESP32 erreichbar ist.

        Args:
            topic: Ziel-Topic.
            payload: Nutzlast als Text.
            force: Auch senden, wenn der ESP32 als nicht erreichbar gilt.

        Returns:
            True, wenn der Befehl gesendet wurde.
        """
        if not self.online and not force:
            logger.error(
                f"ESP32 nicht erreichbar (Zustand: {self.state.get('state', 'unbekannt')}), "
                f"Befehl nicht gesendet."
            )
            return False
        logger.info(f"-> {topic.split('/')[-1]}: {payload}")
        self._client.publish(topic, payload)
        return True

    def _command(self, topic: str, payload: str, wait: bool) -> bool:
        """Sendet einen Bewegungsbefehl und wartet auf Wunsch, bis er fertig ist.

        Returns:
            Ohne wait: True, wenn der Befehl gesendet wurde.
            Mit wait: True, wenn der ESP32 den Befehl ohne Fehler beendet hat.
        """
        if wait:
            # Vor dem Senden zurücksetzen, damit keine Antwort verloren geht
            self._done.clear()
            self._busy_seen = False
            self._failure = None
        if not self.send(topic, payload):
            return False
        if not wait:
            return True

        if not self._done.wait(cfg.WAIT_TIMEOUT_S):
            logger.error(f"Keine Rückmeldung vom ESP32 nach {cfg.WAIT_TIMEOUT_S} s.")
            return False
        if self._failure:
            logger.error(f"Befehl fehlgeschlagen: {self._failure}")
            return False
        return True

    # ------------------------------------------------------------------
    #  Achse
    # ------------------------------------------------------------------
    def home(self, wait: bool = False) -> bool:
        """Referenzfahrt auf den Limit-Schalter (setzt den Nullpunkt)."""
        return self._command(TOPIC_CMD_AXIS, "home", wait)

    def move_steps(self, steps: int, wait: bool = False) -> bool:
        """Fährt auf eine absolute Position in Schritten, sofern sie innerhalb der Grenzen liegt."""
        if not cfg.AXIS_MIN_STEPS <= steps <= cfg.AXIS_MAX_STEPS:
            logger.error(
                f"Ziel {steps} Schritte ausserhalb {cfg.AXIS_MIN_STEPS}..{cfg.AXIS_MAX_STEPS}, "
                f"nicht gesendet."
            )
            return False
        return self._command(TOPIC_CMD_AXIS, str(steps), wait)

    def move_cm(self, cm: float, wait: bool = False) -> bool:
        """Fährt auf eine absolute Position in Zentimetern."""
        return self.move_steps(cm_to_steps(cm), wait)

    def move_min(self, wait: bool = False) -> bool:
        """Fährt auf die minimale Position (unten, am Limit-Schalter)."""
        return self.move_steps(cfg.AXIS_MIN_STEPS, wait)

    def move_max(self, wait: bool = False) -> bool:
        """Fährt auf die maximale Position."""
        return self.move_steps(cfg.AXIS_MAX_STEPS, wait)

    def stop(self) -> bool:
        """Sofortiger Halt der Achse. Wird immer gesendet (Not-Halt)."""
        return self.send(TOPIC_CMD_STOP, "stop", force=True)

    # ------------------------------------------------------------------
    #  Greifer
    # ------------------------------------------------------------------
    def gripper_us(self, us: int, wait: bool = False) -> bool:
        """Stellt den Greifer auf eine Servozeit, sofern sie innerhalb der Grenzen liegt.

        Unterhalb von GRIP_OPEN_US gilt der Greifer als geschlossen: der ESP32
        prüft dann, ob ein Teil gegriffen wurde, und überwacht es weiter.
        Ob ein Teil erkannt wurde, steht danach in has_part.
        """
        if not cfg.GRIP_CLOSED_US <= us <= cfg.GRIP_OPEN_US:
            logger.error(
                f"Greifer {us} us ausserhalb {cfg.GRIP_CLOSED_US}..{cfg.GRIP_OPEN_US}, "
                f"nicht gesendet."
            )
            return False
        check = us < cfg.GRIP_OPEN_US and cfg.USE_PART_SENSOR
        payload = f"{us} check" if check else str(us)
        return self._command(TOPIC_CMD_GRIPPER, payload, wait)

    def gripper_open(self, wait: bool = False) -> bool:
        """Öffnet den Greifer vollständig."""
        return self.gripper_us(cfg.GRIP_OPEN_US, wait)

    def gripper_close(self, wait: bool = False) -> bool:
        """Schließt den Greifer vollständig."""
        return self.gripper_us(cfg.GRIP_CLOSED_US, wait)

    # ------------------------------------------------------------------
    #  Zustand
    # ------------------------------------------------------------------
    def request_status(self) -> bool:
        """Fordert eine aktuelle Zustandsmeldung vom ESP32 an."""
        return self.send(TOPIC_CMD_STATUS, "status")

    def show_state(self) -> None:
        """Gibt den zuletzt empfangenen Zustand aus."""
        if not self.state:
            logger.warning("Noch kein Zustand empfangen.")
            return
        s = self.state
        steps = s.get("steps")
        cm = f"{steps_to_cm(steps):.2f}" if isinstance(steps, (int, float)) else "?"
        logger.info(
            f"Zustand: {s.get('state')} | konfiguriert: {s.get('configured')} | "
            f"referenziert: {s.get('homed')} | "
            f"Position: {steps} Schritte ({cm} cm) | "
            f"Greifer: {s.get('gripper')} ({s.get('gripper_us')} us) | "
            f"Teil erkannt: {s.get('part')} | erreichbar: {self.online}"
        )

    # ------------------------------------------------------------------
    #  MQTT-Callbacks
    # ------------------------------------------------------------------
    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code.is_failure:
            logger.error(f"Broker hat die Verbindung abgelehnt: {reason_code}")
            return
        self._broker_connected = True
        logger.info("Mit dem MQTT-Broker verbunden.")
        # Abo in on_connect, damit es nach einer Neuverbindung automatisch erneuert wird
        client.subscribe([(TOPIC_STATE, 1), (TOPIC_EVENT, 0), (TOPIC_HEARTBEAT, 0)])
        # Retained: der ESP32 bekommt die Werte auch nach jedem eigenen Neustart
        self.push_config()

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        self._broker_connected = False
        if reason_code.is_failure:
            logger.warning(f"Verbindung zum Broker verloren ({reason_code}), verbinde neu...")

    def _on_message(self, client, userdata, msg) -> None:
        text = msg.payload.decode(errors="replace").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Unlesbare Nachricht auf {msg.topic}: {text}")
            return

        if msg.topic == TOPIC_HEARTBEAT:
            self.last_heartbeat = time.monotonic()
        elif msg.topic == TOPIC_STATE:
            previous = self.state.get("state")
            self.state = data
            current = data.get("state")
            if current == "offline":
                logger.error("ESP32 ist offline (Last Will vom Broker).")
            elif current != previous:
                logger.info(f"Zustand: {current} bei {data.get('steps')} Schritten")

            # Befehlsende erkennen: erst beschäftigt, danach wieder "idle"
            if current in BUSY_STATES:
                self._busy_seen = True
            elif current == "idle" and self._busy_seen:
                self._done.set()
            elif current in ("error", "offline"):
                self._failure = self._failure or f"ESP32 im Zustand '{current}'"
                self._done.set()

            if current == "offline" and current != previous:
                self._notify("error", "offline", "ESP32 ist offline")
        elif msg.topic == TOPIC_EVENT:
            level = data.get("level", "info")
            code = data.get("code", "")
            info = data.get("info", "")
            line = f"ESP32: {code} {info}".strip()
            if code in PART_EVENTS and not cfg.USE_PART_SENSOR:
                logger.debug(f"{line} (Teilesensor deaktiviert, ignoriert)")
                return
            if code == "limit_reached":
                level = "info"  # min-Position = Limit-Schalter, das ist normal
            {"error": logger.error, "warn": logger.warning}.get(level, logger.info)(line)

            if level == "error" or code in FAIL_EVENTS:
                self._failure = f"{code} {info}".strip()
                self._done.set()
            elif code in DONE_EVENTS:
                self._done.set()
            self._notify(level, code, info)

    def _notify(self, level: str, code: str, info: str) -> None:
        """Gibt ein Ereignis an on_event weiter, ohne dass Fehler dort den MQTT-Thread stoppen."""
        if self.on_event is None:
            return
        try:
            self.on_event(level, code, info)
        except Exception:
            logger.exception("Fehler im on_event-Callback")
