"""Fließband-Controller (ESP32-C6) über MQTT.
Stellt die Klasse ConveyorRemote zur Verfügung, um das Fließband zu steuern.

Verwendung:
    conveyor = ConveyorRemote(client_id="mein_skript")
    conveyor.connect()
    conveyor.drive(5.0)             # losfahren, sofort zurück
    conveyor.drive(5.0, wait=True)  # losfahren und warten, bis das Band wieder steht
    conveyor.disconnect()

Jeder Client braucht eine eigene client_id, sonst trennt der Broker die
Verbindungen gegenseitig.

eine ganze Fahrt = 18s
"""

import threading

import paho.mqtt.client as mqtt
from loguru import logger

# --- MQTT Konfiguration ---
BROKER = "192.168.2.1"
PORT = 1883
CLIENT_ID = "conveyor_controller"

TOPIC_CMD = "station/conveyor"            # Fahrbefehle: Sekunden oder "stop"
TOPIC_STATUS = "station/conveyor/status"  # idle / running / offline (retained)
TOPIC_ERROR = "station/conveyor/error"    # vom ESP32 abgelehnte Befehle

# So lange wird bei drive(..., wait=True) über die Fahrdauer hinaus auf "idle" gewartet
WAIT_MARGIN_S = 5.0


class ConveyorRemote:
    """Sendet Befehle an das Fließband und verfolgt dessen Status.

    Attributes:
        state: Letzter vom ESP32 gemeldeter Zustand ("idle", "running",
            "offline") oder None, solange kein Status bekannt ist.
    """

    def __init__(self, client_id: str = CLIENT_ID) -> None:
        """
        Args:
            client_id: Eindeutige MQTT-Client-ID dieses Skripts.
        """
        self.state: str | None = None
        self._broker_connected = False
        # Für drive(..., wait=True): wird gesetzt, sobald die Fahrt vorbei ist
        self._done = threading.Event()
        self._running_seen = False
        self._failure: str | None = None

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    # ------------------------------------------------------------------
    #  Verbindung
    # ------------------------------------------------------------------
    def connect(self, background: bool = False) -> None:
        """Verbindet mit dem Broker und startet die MQTT-Netzwerkschleife.

        Args:
            background: True = nicht warten; die Verbindung wird im Hintergrund aufgebaut
                und nach Abbrüchen neu versucht (für main.py, das auch ohne Broker starten soll).

        Raises:
            OSError: Wenn der Broker nicht erreichbar ist (nur ohne background).
        """
        if background:
            self._client.reconnect_delay_set(min_delay=1, max_delay=10)
            self._client.connect_async(BROKER, PORT, keepalive=60)
        else:
            self._client.connect(BROKER, PORT, keepalive=60)
        self._client.loop_start()

    def disconnect(self) -> None:
        """Stoppt eine laufende Fahrt und trennt die Verbindung sauber."""
        if self.state == "running":
            try:
                self.stop().wait_for_publish(timeout=1.0)
            except (RuntimeError, ValueError):
                logger.warning("Stopp beim Beenden konnte nicht gesendet werden.")
        self._client.disconnect()
        self._client.loop_stop()

    def is_connected(self) -> bool:
        """True, wenn der Broker verbunden ist und das Fließband einen gültigen Status meldet."""
        return self._broker_connected and self.state in ("idle", "running")

    # ------------------------------------------------------------------
    #  Befehle
    # ------------------------------------------------------------------
    def drive(self, seconds: float, wait: bool = False) -> bool:
        """Sendet einen Fahrbefehl, sofern das Fließband bereit ist.

        Args:
            seconds: Fahrdauer in Sekunden.
            wait: Blockieren, bis das Band wieder steht.

        Returns:
            Ohne wait: True, wenn der Befehl gesendet wurde.
            Mit wait: True, wenn die Fahrt ohne Fehler zu Ende ging.
        """
        if seconds <= 0:
            logger.warning(f"Fahrdauer muss größer als 0 sein (ist {seconds}).")
            return False
        if self.state != "idle":
            logger.warning(
                f"Fahrbefehl nicht gesendet, Fließband ist nicht bereit "
                f"(Status: {self.state or 'unbekannt'})."
            )
            return False

        # Vor dem Senden zurücksetzen, damit keine Antwort verloren geht
        self._done.clear()
        self._running_seen = False
        self._failure = None

        logger.info(f"Fließband fährt für {seconds} Sekunden vorwärts.")
        self._client.publish(TOPIC_CMD, str(seconds))
        if not wait:
            return True

        timeout = seconds + WAIT_MARGIN_S
        if not self._done.wait(timeout):
            logger.error(f"Fließband meldet nach {timeout:.0f} s kein Fahrtende.")
            return False
        if self._failure:
            logger.error(f"Fahrt fehlgeschlagen: {self._failure}")
            return False
        return True

    def stop(self) -> mqtt.MQTTMessageInfo:
        """Sendet einen Stopp-Befehl, unabhängig vom aktuellen Status."""
        logger.info("Stopp gesendet.")
        return self._client.publish(TOPIC_CMD, "stop")

    # ------------------------------------------------------------------
    #  MQTT-Callbacks
    # ------------------------------------------------------------------
    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code.is_failure:
            logger.error(f"Broker hat die Verbindung abgelehnt: {reason_code}")
            return
        self._broker_connected = True
        logger.info("Fließband: mit dem MQTT-Broker verbunden.")
        client.subscribe([(TOPIC_STATUS, 1), (TOPIC_ERROR, 0)])

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        self._broker_connected = False
        self.state = None
        if reason_code.is_failure:
            logger.warning(f"Verbindung zum Broker verloren ({reason_code}), verbinde neu...")

    def _on_message(self, client, userdata, msg) -> None:
        text = msg.payload.decode(errors="replace").strip()
        if msg.topic == TOPIC_STATUS:
            self.state = text
            logger.info(f"Fließband-Status: {text}")
            # Fahrtende erkennen: erst "running", danach wieder "idle"
            if text == "running":
                self._running_seen = True
            elif text == "idle" and self._running_seen:
                self._done.set()
            elif text == "offline":
                self._failure = "Fließband ist offline"
                self._done.set()
        elif msg.topic == TOPIC_ERROR:
            logger.warning(f"ESP32 hat den Befehl abgelehnt: {text}")
            self._failure = f"abgelehnt: {text}"
            self._done.set()
