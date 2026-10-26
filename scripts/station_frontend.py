"""Bedienoberfläche der Station im Browser (Vollbild).

Der Server läuft im Prozess von main.py in eigenen Threads (nur Standardbibliothek) und zeigt:
    - zwei Kamerabilder (Standard: YOLO-Kamera mit Erkennungen, Arducam mit ArUco-Ortung)
    - den Ablauf: vorheriger / aktueller / nächster Schritt, Fortschritt, Zustand des Automaten
    - Fehlermeldungen (quittierbar) und die Meldungen der Station
    - Knöpfe Start / Pause / Stop / Quittieren / Not-Aus, alle mit einstellbaren Tastenkürzeln

Konfiguration:
    config/frontend.yaml   Server, Browser, Quellen der Kamerabilder
    config/hotkeys.yaml    Tastenkürzel (auch im Frontend über ⚙ änderbar)

HTTP-Schnittstelle (nur für die Oberfläche in scripts/frontend/):
    GET  /                    Oberfläche
    GET  /api/config          Titel der Kamerabilder, Tastenkürzel, Aktionen, Zustände
    GET  /api/events          Zustand als Server-Sent Events (SNAPSHOT_HZ pro Sekunde)
    POST /api/cmd             {"cmd": "start", "mode": "alles"} | pause | stop | ack | estop | quit
    PUT  /api/hotkeys         {"start": "F2", ...} -> config/hotkeys.yaml
    GET  /feed/<name>.mjpg    Kamerabild als MJPEG-Strom

Quittieren (ack) markiert alle Fehlermeldungen als gesehen und löst, wenn die Station in
error / estop steht, 'reset' aus.

Das Frontend startet vor allen Geräten und läuft auch dann weiter, wenn ein Gerät oder die
ganze Station nicht startet. Welche Systeme laufen, zeigt SystemBoard (Prüffunktionen
meldet main.py an).
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import threading
import time
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import yaml
from loguru import logger

from F_stilt_erkennung_YOLO import annotate
from station_display import say

WEB_DIR = Path(__file__).resolve().parent / "frontend"
STATIC_FILES = {"/": "index.html", "/index.html": "index.html", "/app.css": "app.css", "/app.js": "app.js",
                "/walz_logo.png": "walz_logo.png"}   # Kopie von docs/walz_4.0_logo-horizontal.png
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".png": "image/png"}

MQTT_CLIENT_ID = "station_frontend"
SNAPSHOT_HZ = 5               # so oft bekommt die Oberfläche den Zustand
FEED_FPS = 15                 # höchstens so viele Bilder pro Sekunde je Kamerabild
JPEG_QUALITY = 80
# Vorschaubilder, die ein Sender auf diesem Rechner zusätzlich als Datei ablegt (Topic -> Datei).
# Muss zu PREVIEW_FILE in aruco_tracker_calibrated.py passen.
_SHM = Path("/dev/shm" if Path("/dev/shm").is_dir() else "/tmp")
LOCAL_PREVIEW_FILES = {"station/tracker/preview": _SHM / "station_tracker_preview.jpg"}
LOCAL_PREVIEW_POLL_S = 0.05
LOCAL_PREVIEW_MAX_AGE_S = 2.0       # ältere Datei = Sender läuft nicht mehr
LOCAL_PREVIEW_MQTT_GRACE_S = 2.0    # so lange nach einem MQTT-Bild wird die Datei ignoriert
MAX_ALARMS = 100
MAX_LOG = 150

# Aktion -> Beschriftung im Einstellungsdialog (Reihenfolge = Anzeige)
ACTIONS = {
    "start": "Ablauf starten",
    "pause": "Pause / Weiter",
    "stop": "Stop",
    "ack": "Alarm quittieren",
    "estop": "NOT-AUS",
    "quit": "Programm beenden",
}
DEFAULT_HOTKEYS = {"start": "F2", "pause": "F3", "stop": "F4", "ack": "F8", "estop": "Space", "quit": "F10"}
HOTKEY_RE = re.compile(r"^(?:(?:Ctrl|Alt|Shift|Meta)\+)*\w+$")
HOTKEYS_HEADER = (
    "# Tastenkürzel des Frontends. Im Frontend über ⚙ änderbar – die Datei wird dann neu geschrieben.\n"
    "# Schreibweise: [Ctrl+][Alt+][Shift+]Taste, z. B. F2, Space, Enter, Escape, Ctrl+S, A, 1\n"
)
DEFAULT_CONFIG = {
    "server": {"host": "127.0.0.1", "port": 8080},
    "browser": {"open": True, "command": "auto"},
    "feeds": {
        "gripper": {"title": "Kamera Greifarm", "source": "yolo"},
        "aruco": {"title": "SID-Ortung (Arducam, ArUco)", "source": "mqtt:station/tracker/preview"},
    },
    "stale_after_s": 2.0,
}
CYCLE_MODES = ("alles", "liefern", "holen")


# ----------------------------------------------------------------------
#  Konfiguration
# ----------------------------------------------------------------------
def load_config(path: Path) -> dict:
    """Liest config/frontend.yaml; fehlende Einträge kommen aus DEFAULT_CONFIG."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # tiefe Kopie
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict) and key != "feeds":
                config[key].update(value)
            else:
                config[key] = value
    return config


def check_hotkeys(hotkeys: dict) -> dict[str, str]:
    """Prüft Tastenkürzel: nur bekannte Aktionen, gültige Schreibweise, keine Taste doppelt.

    Raises:
        ValueError: mit verständlicher Begründung.
    """
    result = {}
    for action, combo in hotkeys.items():
        if action not in ACTIONS:
            raise ValueError(f"Unbekannte Aktion '{action}'")
        combo = str(combo).strip()
        if not HOTKEY_RE.match(combo):
            raise ValueError(f"Ungültiges Tastenkürzel '{combo}' für {ACTIONS[action]}")
        if combo in result.values():
            raise ValueError(f"Taste '{combo}' ist doppelt belegt")
        result[action] = combo
    missing = [ACTIONS[a] for a in ACTIONS if a not in result]
    if missing:
        raise ValueError(f"Ohne Tastenkürzel: {', '.join(missing)}")
    return result


def load_hotkeys(path: Path) -> dict[str, str]:
    """Liest config/hotkeys.yaml; bei Fehlern gelten die Standardtasten."""
    hotkeys = dict(DEFAULT_HOTKEYS)
    if path.is_file():
        hotkeys.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    try:
        return check_hotkeys(hotkeys)
    except ValueError as e:
        say("error", f"{path.name}: {e} – es gelten die Standardtasten.")
        return dict(DEFAULT_HOTKEYS)


def save_hotkeys(path: Path, hotkeys: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{action}: {hotkeys[action]}\n" for action in ACTIONS)
    path.write_text(HOTKEYS_HEADER + body, encoding="utf-8")


# ----------------------------------------------------------------------
#  Fehlermeldungen und Stationsmeldungen
# ----------------------------------------------------------------------
class EventBook:
    """Sammelt aus dem loguru-Log die Fehlermeldungen (quittierbar) und die Stationsmeldungen (say)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._alarms: deque[dict] = deque(maxlen=MAX_ALARMS)
        self._log: deque[dict] = deque(maxlen=MAX_LOG)
        self._next_id = 1

    def sink(self, message) -> None:
        """loguru-Sink: say()-Meldungen ins Protokoll, Fehler und Warnungen der Geräte als Alarm."""
        record = message.record
        kind = record["extra"].get("kind")
        level = record["level"].no
        text = record["message"]
        stamp = record["time"].timestamp()
        with self._lock:
            if kind is not None:
                self._log.append({"t": stamp, "kind": kind, "text": text})
            if kind == "error" or (kind is None and level >= 30):
                severity = "error" if level >= 40 else "warning"
                last = self._alarms[-1] if self._alarms else None
                if last and not last["acked"] and last["text"] == text:
                    last["count"] += 1
                    last["t"] = stamp
                else:
                    self._alarms.append({"id": self._next_id, "t": stamp, "level": severity,
                                         "text": text, "count": 1, "acked": False})
                    self._next_id += 1

    @staticmethod
    def wants(record) -> bool:
        return "kind" in record["extra"] or record["level"].no >= 30

    def ack_all(self) -> int:
        """Markiert alle Meldungen als quittiert und gibt zurück, wie viele neu quittiert wurden."""
        with self._lock:
            open_alarms = [a for a in self._alarms if not a["acked"]]
            for alarm in open_alarms:
                alarm["acked"] = True
            return len(open_alarms)

    def snapshot(self) -> tuple[list[dict], list[dict]]:
        with self._lock:
            return [dict(a) for a in self._alarms], list(self._log)


# ----------------------------------------------------------------------
#  Systemübersicht
# ----------------------------------------------------------------------
SystemCheck = Callable[[], tuple[str, str]]


class SystemBoard:
    """Live-Übersicht, welche Teilsysteme laufen.

    Für jedes System wird eine Prüffunktion angemeldet, die (Stufe, Text) liefert:
        ok    läuft
        warn  läuft eingeschränkt (z. B. SID nicht im Bild der Ortung)
        bad   läuft nicht
        off   noch nicht gestartet / nicht benutzt
    Statt einer Funktion geht auch ein festes Paar, z. B. ("bad", "Start fehlgeschlagen: ...").

    Ein Thread prüft alle CHECK_INTERVAL_S Sekunden. Nach GRACE_S Sekunden Anlaufzeit meldet er
    einmal, was nicht bereit ist; danach jeden Ausfall (-> bad) und jede Wiederkehr als
    Stationsmeldung (Terminal, Frontend).
    """

    CHECK_INTERVAL_S = 0.5
    GRACE_S = 8.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._checks: dict[str, SystemCheck | tuple[str, str]] = {}
        self._results: dict[str, tuple[str, str]] = {}

    def set(self, name: str, check: SystemCheck | tuple[str, str]) -> None:
        """Meldet ein System an oder ersetzt seine Prüffunktion (Reihenfolge = erste Anmeldung)."""
        with self._lock:
            self._checks[name] = check
        self._evaluate()

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [{"name": name, "level": level, "text": text} for name, (level, text) in self._results.items()]

    def start(self, stop: threading.Event) -> None:
        threading.Thread(target=self._run, args=(stop,), daemon=True, name="systeme").start()

    def _evaluate(self) -> dict[str, tuple[str, str]]:
        with self._lock:
            checks = dict(self._checks)
        results = {}
        for name, check in checks.items():
            try:
                level, text = check() if callable(check) else check
            except Exception as e:  # eine kaputte Prüfung darf die Übersicht nicht stoppen
                level, text = "bad", f"Prüfung fehlgeschlagen: {e}"
            results[name] = (level, text)
        with self._lock:
            self._results = results
        return results

    def _run(self, stop: threading.Event) -> None:
        started = time.monotonic()
        baseline: dict[str, tuple[str, str]] | None = None
        while not stop.wait(self.CHECK_INTERVAL_S):
            results = self._evaluate()
            if baseline is None:
                if time.monotonic() - started < self.GRACE_S:
                    continue
                missing = [f"{name} ({text})" for name, (level, text) in results.items() if level == "bad"]
                if missing:
                    say("error", "Nicht bereit: " + "; ".join(missing))
                else:
                    say("ok", "Alle Systeme bereit.")
                baseline = results
                continue
            for name, (level, text) in results.items():
                old = baseline.get(name, ("off", ""))[0]
                if level == "bad" and old != "bad":
                    say("error", f"{name} ausgefallen: {text}")
                elif old == "bad" and level in ("ok", "warn"):
                    say("ok", f"{name} wieder da: {text}")
            baseline = results


# ----------------------------------------------------------------------
#  Kamerabilder
# ----------------------------------------------------------------------
class Feed:
    """Hält das letzte Bild einer Quelle als JPEG; die MJPEG-Ströme warten auf das nächste."""

    def __init__(self, name: str, title: str, source: str) -> None:
        self.name = name
        self.title = title
        self.source = source
        self.viewers = 0
        self.note = ""                 # Hinweis, warum es kein Bild gibt
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._stamp = 0.0

    def start(self, stop: threading.Event) -> None:
        """Startet ggf. einen Lesethread; stop beendet ihn."""

    def attach(self, delta: int) -> None:
        """Meldet einen Zuschauer an (+1) oder ab (-1)."""
        with self._cond:
            self.viewers += delta

    def put(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._stamp = time.time()
            self._cond.notify_all()

    def put_image(self, image: np.ndarray) -> None:
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            self.put(jpeg.tobytes())

    def wait_next(self, seq: int, timeout: float) -> tuple[int, bytes] | None:
        """Wartet auf ein Bild, das neuer ist als seq (None = Zeit abgelaufen)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != seq and self._jpeg is not None, timeout):
                return None
            return self._seq, self._jpeg

    def info(self, stale_after_s: float) -> dict:
        age = time.time() - self._stamp if self._stamp else None
        return {"ok": age is not None and age < stale_after_s, "age": age, "note": self.note}


class YoloFeed(Feed):
    """Bild der YOLO-Kamera mit den letzten Erkennungen (liest nur, solange jemand zuschaut)."""

    def __init__(self, name: str, title: str, source: str, detector=None) -> None:
        super().__init__(name, title, source)
        self.detector = detector   # kann nach dem Start gesetzt werden (StationFrontend.attach)

    def start(self, stop: threading.Event) -> None:
        threading.Thread(target=self._run, args=(stop,), daemon=True, name=f"feed-{self.name}").start()

    def _run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            if self.viewers == 0:
                time.sleep(0.2)
                continue
            detector = self.detector
            if detector is None or not detector.is_connected():
                self.note = "YOLO-Kamera nicht verbunden"
                time.sleep(0.5)
                continue
            started = time.monotonic()
            image = detector.read_image(fresh=False)
            if image is None:
                self.note = "YOLO-Kamera liefert kein Bild"
                time.sleep(0.2)
                continue
            self.note = ""
            self.put_image(annotate(image, detector.last_detections))
            time.sleep(max(0.0, 1.0 / FEED_FPS - (time.monotonic() - started)))


class CameraFeed(Feed):
    """Eigene USB-Kamera (camera:<Nummer> oder camera:<Teilstring des Gerätenamens>)."""

    def __init__(self, name: str, title: str, source: str, device: str) -> None:
        super().__init__(name, title, source)
        self._device = device

    def _find_index(self) -> int | None:
        if self._device.isdigit():
            return int(self._device)
        for entry in sorted(Path("/sys/class/video4linux").glob("video*")):
            try:
                if self._device.lower() in (entry / "name").read_text().lower():
                    return int(entry.name.removeprefix("video"))
            except (OSError, ValueError):
                continue
        return None

    def start(self, stop: threading.Event) -> None:
        threading.Thread(target=self._run, args=(stop,), daemon=True, name=f"feed-{self.name}").start()

    def _run(self, stop: threading.Event) -> None:
        cap = None
        while not stop.is_set():
            if cap is None:
                index = self._find_index()
                cap = cv2.VideoCapture(index, cv2.CAP_V4L2) if index is not None else None
                if cap is None or not cap.isOpened():
                    self.note = f"Kamera '{self._device}' nicht gefunden"
                    cap = None
                    stop.wait(2.0)
                    continue
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            started = time.monotonic()
            ok, frame = cap.read()
            if not ok or frame is None:
                self.note = f"Kamera '{self._device}' liefert kein Bild"
                cap.release()
                cap = None
                stop.wait(1.0)
                continue
            self.note = ""
            if self.viewers:
                self.put_image(frame)
            time.sleep(max(0.0, 1.0 / FEED_FPS - (time.monotonic() - started)))
        if cap is not None:
            cap.release()


class MqttFeed(Feed):
    """JPEG-Bilder, die per MQTT ankommen (z. B. Vorschau der SID-Ortung)."""

    def __init__(self, name: str, title: str, source: str, topic: str) -> None:
        super().__init__(name, title, source)
        self.topic = topic
        self.note = "SID-Ortung sendet kein Bild (läuft sie mit --preview?)"
        # Läuft der Sender auf diesem Rechner, kommt das Bild auch ohne Broker über eine Datei
        self.local_file = LOCAL_PREVIEW_FILES.get(topic)
        self._last_mqtt = 0.0

    def put_mqtt(self, jpeg: bytes) -> None:
        """Bild, das per MQTT angekommen ist."""
        self._last_mqtt = time.monotonic()
        self.note = ""
        self.put(jpeg)

    def start(self, stop: threading.Event) -> None:
        if self.local_file is not None:
            threading.Thread(target=self._watch_file, args=(stop,), daemon=True, name=f"feed-{self.name}").start()

    def _watch_file(self, stop: threading.Event) -> None:
        """Liest das lokale Vorschaubild, sobald es sich ändert (nur solange per MQTT nichts kommt)."""
        last_mtime = 0
        while not stop.wait(LOCAL_PREVIEW_POLL_S):
            if time.monotonic() - self._last_mqtt < LOCAL_PREVIEW_MQTT_GRACE_S:
                continue
            try:
                stat = self.local_file.stat()
                if stat.st_mtime_ns == last_mtime or time.time() - stat.st_mtime > LOCAL_PREVIEW_MAX_AGE_S:
                    continue
                last_mtime = stat.st_mtime_ns
                jpeg = self.local_file.read_bytes()
            except OSError:
                continue
            if jpeg:
                self.note = ""
                self.put(jpeg)


def make_feed(name: str, spec: dict) -> Feed:
    title = str(spec.get("title", name))
    source = str(spec.get("source", ""))
    kind, _, arg = source.partition(":")
    if kind == "yolo":
        return YoloFeed(name, title, source)
    if kind == "camera" and arg:
        return CameraFeed(name, title, source, arg)
    if kind == "mqtt" and arg:
        return MqttFeed(name, title, source, arg)
    raise ValueError(f"Kamerabild '{name}': unbekannte Quelle '{source}' (yolo | camera:<...> | mqtt:<Topic>)")


# ----------------------------------------------------------------------
#  Browserfenster
# ----------------------------------------------------------------------
FIREFOX_PREFS = """\
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.aboutwelcome.enabled", false);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("browser.translations.automaticallyPopup", false);
user_pref("datareporting.policy.dataSubmissionPolicyBypassNotification", true);
user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);
// Wird die Station kurz nach dem Start beendet, zählt Firefox das als „Absturz beim Start“ und
// fragt nach 3 solchen Läufen nach dem Fehlerbehebungsmodus statt das Frontend zu zeigen. Aus:
user_pref("toolkit.startup.max_resumed_crashes", -1);
user_pref("toolkit.startup.recent_crashes", 0);
"""


def _profile_dir(browser: str) -> Path:
    """Eigenes Browserprofil, damit ein neues Kiosk-Fenster aufgeht (Snap-Browser dürfen nur ~/snap/...)."""
    if Path(f"/snap/{browser}").exists():
        folder = Path.home() / "snap" / browser / "common" / "station-frontend"
    else:
        folder = Path.home() / ".cache" / "station-frontend" / browser
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def open_browser(url: str, command: str = "auto") -> subprocess.Popen | None:
    """Öffnet die Oberfläche im Vollbild (Kiosk). Gibt den Browserprozess zurück (None = nicht steuerbar)."""
    if command and command != "auto":
        return subprocess.Popen(shlex.split(command.format(url=url)))
    for name in ("chromium", "chromium-browser", "google-chrome"):
        exe = shutil.which(name)
        if exe:
            profile = _profile_dir(name.removesuffix("-browser"))
            return subprocess.Popen([exe, "--kiosk", f"--user-data-dir={profile}", "--no-first-run",
                                     "--noerrdialogs", "--disable-translate", f"--app={url}"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    exe = shutil.which("firefox")
    if exe:
        profile = _profile_dir("firefox")
        (profile / "user.js").write_text(FIREFOX_PREFS)
        # ohne --new-instance: läuft das Kiosk-Profil schon, öffnet Firefox die Seite dort (kein „läuft bereits“)
        return subprocess.Popen([exe, "--kiosk", "--profile", str(profile), url],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    webbrowser.open(url)
    return None


# ----------------------------------------------------------------------
#  Server
# ----------------------------------------------------------------------
class StationFrontend:
    """Webserver der Bedienoberfläche. Die Station wird nur über handle_command() und status() bedient.

    Läuft auch ohne Station (vor dem Start der Geräte oder wenn sie nicht startet);
    attach() hängt sie an, sobald es sie gibt.
    """

    def __init__(self, config_dir: Path, broker: str, port: int, systems: SystemBoard, states: list[str]) -> None:
        self.station = None
        self.systems = systems
        self.states = list(states)
        self.problem = ""             # warum es keine Station gibt (Anzeige statt Zustand)
        self.on_quit: Callable[[], None] | None = None   # beendet das Programm (setzt main.py)
        self.config_path = config_dir / "frontend.yaml"
        self.hotkeys_path = config_dir / "hotkeys.yaml"
        self.config = load_config(self.config_path)
        self.hotkeys = load_hotkeys(self.hotkeys_path)
        self.book = EventBook()
        self.feeds = {name: make_feed(name, spec) for name, spec in self.config["feeds"].items()}
        self.stopping = threading.Event()
        self.url = ""
        self._broker = (broker, port)
        self._server: ThreadingHTTPServer | None = None
        self._browser: subprocess.Popen | None = None
        self._mqtt: mqtt.Client | None = None
        self._sink_id = logger.add(self.book.sink, level=0, filter=EventBook.wants, format="{message}")

    # ---------- Start / Ende ----------
    def attach(self, station=None, detector=None) -> None:
        """Hängt Station und YOLO-Erkennung an, sobald es sie gibt."""
        if detector is not None:
            for feed in self.feeds.values():
                if isinstance(feed, YoloFeed):
                    feed.detector = detector
        if station is not None:
            self.station = station
            self.problem = ""

    def start(self) -> str:
        """Startet Server, Kamerabilder und MQTT-Empfang. Gibt die Adresse der Oberfläche zurück."""
        host, port = self.config["server"]["host"], int(self.config["server"]["port"])
        self._server = ThreadingHTTPServer((host, port), _make_handler(self))
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True, name="frontend").start()
        for feed in self.feeds.values():
            feed.start(self.stopping)
        self._start_mqtt()
        self.url = f"http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}/"
        say("ok", f"Frontend läuft: {self.url}")
        return self.url

    def open_window(self) -> None:
        if not self.config["browser"].get("open", True):
            return
        try:
            self._browser = open_browser(self.url, str(self.config["browser"].get("command", "auto")))
        except OSError as e:
            say("error", f"Browser konnte nicht gestartet werden ({e}). Frontend von Hand öffnen: {self.url}")

    def stop(self) -> None:
        self.stopping.set()
        logger.remove(self._sink_id)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._mqtt is not None:
            self._mqtt.disconnect()
            self._mqtt.loop_stop()
        if self._browser is not None and self._browser.poll() is None:
            self._browser.terminate()

    def _start_mqtt(self) -> None:
        topics = {feed.topic: feed for feed in self.feeds.values() if isinstance(feed, MqttFeed)}
        if not topics:
            return
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID)

        def on_connect(client, userdata, flags, reason_code, properties):
            if not reason_code.is_failure:
                client.subscribe([(topic, 0) for topic in topics])

        def on_message(client, userdata, msg):
            feed = topics.get(msg.topic)
            if feed is not None:
                feed.put_mqtt(bytes(msg.payload))

        client.on_connect = on_connect
        client.on_message = on_message
        client.reconnect_delay_set(min_delay=1, max_delay=10)
        client.connect_async(*self._broker, keepalive=30)
        client.loop_start()
        self._mqtt = client

    # ---------- Daten für die Oberfläche ----------
    def client_config(self) -> dict:
        return {
            "actions": ACTIONS,
            "hotkeys": self.hotkeys,
            "default_hotkeys": DEFAULT_HOTKEYS,
            "feeds": [{"name": f.name, "title": f.title, "source": f.source} for f in self.feeds.values()],
            "modes": CYCLE_MODES,
            "states": self.states,
        }

    def _no_station_status(self) -> dict:
        """Zustand für die Anzeige, solange es (noch) keine Station gibt."""
        return {
            "state": "failed" if self.problem else "init",
            "message": self.problem or "Station startet ...",
            "phase": "", "step": "", "band": "", "prev_state": "", "cycle": "", "holding": False,
            "step_index": 0, "step_total": 0, "prev_step": "", "next_step": "",
            "location": None, "pose": None, "tracker": "", "detections": 0, "devices_ok": {},
        }

    def snapshot(self) -> dict:
        st = self.station
        plan = st.plan if st is not None else []
        alarms, log = self.book.snapshot()
        stale = float(self.config.get("stale_after_s", 2.0))
        return {
            "station": st.status() if st is not None else self._no_station_status(),
            "systems": self.systems.snapshot(),
            "plan": [{"text": s.text, "section": s.section, "stilt": s.stilt} for s in plan],
            "alarms": alarms,
            "log": log,
            "feeds": {name: feed.info(stale) for name, feed in self.feeds.items()},
            "time": time.time(),
        }

    # ---------- Befehle ----------
    def command(self, cmd: str, mode: str | None = None) -> None:
        st = self.station
        if cmd == "quit":  # geht immer, auch ohne Station (main.py stoppt dann alle Geräte)
            if self.on_quit is None:
                raise ValueError("Beenden ist hier nicht möglich.")
            say("info", "Programm beenden vom Frontend.")
            self.on_quit()
            return
        if st is None:
            if cmd == "ack":
                self.book.ack_all()
                return
            raise ValueError("Die Station läuft nicht – siehe Systeme.")
        if cmd == "estop":  # zuerst handeln, dann melden
            st.handle_command("estop")
            say("info", "NOT-AUS vom Frontend.")
            return
        say("info", f"Befehl vom Frontend: {cmd}" + (f" {mode}" if cmd == "start" and mode else ""))
        if cmd == "start":
            st.handle_command(f"start {mode if mode in CYCLE_MODES else 'alles'}")
        elif cmd == "pause":
            st.handle_command("resume" if st.state == "paused" else "pause")
        elif cmd == "stop":
            st.handle_command("stop")
        elif cmd == "ack":
            count = self.book.ack_all()
            if st.state in ("error", "estop"):
                st.handle_command("reset")
            elif count:
                say("ok", f"{count} Meldung(en) quittiert.")
        else:
            raise ValueError(f"Unbekannter Befehl '{cmd}'")

    def set_hotkeys(self, hotkeys: dict) -> dict[str, str]:
        checked = check_hotkeys(hotkeys)
        save_hotkeys(self.hotkeys_path, checked)
        self.hotkeys = checked
        say("ok", "Tastenkürzel gespeichert: " + ", ".join(f"{ACTIONS[a]} = {k}" for a, k in checked.items()))
        return checked


def _make_handler(frontend: StationFrontend):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # kein Zugriffsprotokoll in der Konsole
            logger.trace("frontend: " + fmt % args)

        # ---------- Hilfen ----------
        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(data, default=str).encode(), "application/json")

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("JSON-Objekt erwartet")
            return data

        # ---------- GET ----------
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in STATIC_FILES:
                file = WEB_DIR / STATIC_FILES[path]
                self._send(HTTPStatus.OK, file.read_bytes(), CONTENT_TYPES[file.suffix])
            elif path == "/api/config":
                self._json(frontend.client_config())
            elif path == "/api/events":
                self._events()
            elif path.startswith("/feed/") and path.endswith(".mjpg"):
                feed = frontend.feeds.get(path[len("/feed/"):-len(".mjpg")])
                if feed is None:
                    self._json({"error": "unbekanntes Kamerabild"}, HTTPStatus.NOT_FOUND)
                else:
                    self._mjpeg(feed)
            else:
                self._json({"error": "nicht gefunden"}, HTTPStatus.NOT_FOUND)

        def _events(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                while not frontend.stopping.is_set():
                    data = json.dumps(frontend.snapshot(), default=str)
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                    frontend.stopping.wait(1.0 / SNAPSHOT_HZ)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _mjpeg(self, feed: Feed) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            feed.attach(+1)
            seq = -1
            try:
                while not frontend.stopping.is_set():
                    item = feed.wait_next(seq, timeout=1.0)
                    if item is None:
                        continue
                    seq, jpeg = item
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                feed.attach(-1)

        # ---------- POST / PUT ----------
        def do_POST(self):
            if self.path != "/api/cmd":
                self._json({"error": "nicht gefunden"}, HTTPStatus.NOT_FOUND)
                return
            try:
                data = self._read_json()
                frontend.command(str(data.get("cmd", "")), data.get("mode"))
            except ValueError as e:
                self._json({"ok": False, "error": str(e)}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"ok": True, "state": frontend.station.state})

        def do_PUT(self):
            if self.path != "/api/hotkeys":
                self._json({"error": "nicht gefunden"}, HTTPStatus.NOT_FOUND)
                return
            try:
                hotkeys = frontend.set_hotkeys(self._read_json())
            except ValueError as e:
                self._json({"ok": False, "error": str(e)}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"ok": True, "hotkeys": hotkeys})

    return Handler
