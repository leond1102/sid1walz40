"""Bewegungstraining: Webserver und Ablaufsteuerung für trainer.py (unabhängig von main.py).

Das Programm startet die vorhandenen Skripte als eigene Prozesse und liest ihre Ausgabe mit:
    Sammeln     scripts/collect_<policy>.py --samples N [--sim] [--groups ...]
    Trainieren  scripts/train_dynamics.py / train_rotation.py --csv ... --out models/kandidaten/...
Ein trainiertes Modell ist erst ein Kandidat; „Übernehmen“ legt das bisherige Modell in
models/archiv/ ab und kopiert den Kandidaten nach models/ (gilt ab dem nächsten Start der Station).

Sicherheit:
    - Immer nur ein Lauf gleichzeitig.
    - Mit echtem Roboter nur, wenn die Station (main.py) NICHT läuft (beide würden den SID steuern)
      und die Ortung den SID verfolgt.
    - STOPP schickt dem Lauf Strg+C (SIGINT): Die Skripte stoppen den Roboter sofort und schließen
      ihre Dateien. Reagiert ein Prozess nicht, wird er nach STOP_TIMEOUT_S beendet.

HTTP (nur für scripts/frontend/trainer.*):
    GET  /                  Oberfläche
    GET  /api/config        Policies, Modelle, Tasten
    GET  /api/events        Zustand als Server-Sent Events (neue Ausgabezeilen werden mitgeschickt)
    POST /api/cmd           collect | train | stop | activate | quit
    GET  /feed.mjpg         Bild der Ortung (MQTT station/tracker/preview)
    GET  /plots/<name>.png  Diagramme aus data/plots
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import paho.mqtt.client as mqtt
from loguru import logger

import paths
from station_display import say
from station_frontend import CONTENT_TYPES, WEB_DIR, MqttFeed, load_config

SCRIPTS = Path(__file__).resolve().parent
BROKER = "192.168.2.1"
PORT = 1883
MQTT_CLIENT_ID = "trainer_frontend"
TOPIC_PREVIEW = "station/tracker/preview"
TOPIC_POSE = "station/tracker/robot/pose"
TOPIC_TRACKER_STATUS = "station/tracker/status"

SNAPSHOT_HZ = 4
MAX_LINES = 3000              # so viele Ausgabezeilen werden gehalten
FIRST_LINES = 400             # so viele bekommt ein neu verbundenes Frontend
STOP_TIMEOUT_S = 6.0
STATION_CHECK_S = 2.0
STATIC_FILES = {"/": "trainer.html", "/trainer.html": "trainer.html", "/trainer.js": "trainer.js",
                "/trainer.css": "trainer.css", "/app.css": "app.css", "/walz_logo.png": "walz_logo.png"}
CONTENT_TYPES = {**CONTENT_TYPES, ".png": "image/png"}
GROUP_LABELS = {"yaw": "Drehen", "lateral": "seitlich", "sideyaw": "seitlich + drehen", "longitudinal": "vor / zurück"}
MODEL_LABELS = {"dynamics": "Laufmodell", "rotation": "Drehmodell"}


# ----------------------------------------------------------------------
#  Policies und Dateien
# ----------------------------------------------------------------------
def load_policies() -> dict[str, dict]:
    """Liest Aktionen, Sample-Zahlen und Radien direkt aus den Sammelskripten (eine Quelle der Wahrheit)."""
    import collect_common as cc
    import collect_fine
    import collect_linear
    import collect_mixed
    import collect_rotation

    return {
        "linear": dict(
            title="Geradlinig", script="collect_linear.py", samples=collect_linear.NUM_SAMPLES,
            text="vor / zurück / links / rechts, ohne Drehen. Basis des Laufmodells.",
            actions=[a for a, _ in collect_linear.ACTIONS], dataset="linear_dataset.csv",
            workspace=cc.WORKSPACE_RADIUS, limit=cc.MAX_DISTANCE_FROM_START),
        "mixed": dict(
            title="Gemischt", script="collect_mixed.py", samples=collect_mixed.NUM_SAMPLES,
            text="mehrere Achsen gleichzeitig: schräg laufen, Kurven, seitlich + drehen.",
            actions=[a for a, _ in collect_mixed.ACTIONS], dataset="mixed_dataset.csv",
            workspace=cc.WORKSPACE_RADIUS, limit=cc.MAX_DISTANCE_FROM_START),
        "rotation": dict(
            title="Drehungen", script="collect_rotation.py", samples=collect_rotation.NUM_SAMPLES,
            text="Einzeldrehungen und Flicks (hin und zurück). Für das Drehmodell.",
            actions=list(collect_rotation.KIND_WEIGHTS), dataset="rotation_dataset.csv",
            workspace=None, limit=collect_rotation.MAX_DISTANCE_FROM_START),
        "fine": dict(
            title="Feine Schritte", script="collect_fine.py", samples=collect_fine.NUM_SAMPLES,
            text="kurze Impulse, Bremsimpulse und Doppeltipper für präzise Korrekturen.",
            actions=list(collect_fine.PRIMITIVES), dataset="fine_dataset.csv",
            groups={g: GROUP_LABELS.get(g, g) for g in sorted({p["group"] for p in collect_fine.PRIMITIVES.values()})},
            workspace=cc.WORKSPACE_RADIUS, limit=cc.MAX_DISTANCE_FROM_START),
    }


def csv_kind(path: Path) -> str:
    """Format einer CSV: rotation (collect_rotation / nav_rotation_log), fine (collect_fine) oder motion."""
    with path.open(encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split(",")
    if "segments" in header:
        return "fine"
    if "t1" in header and "kind" in header:
        return "rotation"
    return "motion"


class DataFiles:
    """Übersicht der Datensätze (mit Zeilenzahl, zwischengespeichert bis sich die Datei ändert)."""

    def __init__(self) -> None:
        self._cache: dict[Path, tuple[float, int, dict]] = {}

    def list(self) -> list[dict]:
        files = sorted(paths.DATASETS.glob("*.csv")) + sorted(paths.NAV_LOGS.glob("*.csv"))
        result = []
        for path in files:
            st = path.stat()
            cached = self._cache.get(path)
            if cached is None or cached[:2] != (st.st_mtime, st.st_size):
                with path.open("rb") as f:
                    rows = max(0, sum(1 for _ in f) - 1)
                info = {"name": path.name, "folder": path.parent.name, "rows": rows, "mtime": st.st_mtime,
                        "kind": csv_kind(path), "sim": path.stem.endswith("_sim")}
                self._cache[path] = (st.st_mtime, st.st_size, info)
            result.append(self._cache[path][2])
        return result

    @staticmethod
    def resolve(name: str) -> Path:
        """Nur Dateien aus data/datasets und data/nav_logs (kein Pfad von außen)."""
        for folder in (paths.DATASETS, paths.NAV_LOGS):
            candidate = folder / Path(name).name
            if candidate.is_file():
                return candidate
        raise ValueError(f"Datensatz '{name}' nicht gefunden")


def model_info() -> dict:
    def info(path: Path) -> dict | None:
        if not path.is_file():
            return None
        st = path.stat()
        return {"name": path.name, "mtime": st.st_mtime, "size": st.st_size}

    candidates = []
    for path in sorted(paths.CANDIDATES.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True):
        model = "rotation" if path.name.startswith("rotation") else "dynamics"
        candidates.append({**info(path), "model": model, "label": MODEL_LABELS[model]})
    return {"active": {"dynamics": info(paths.DYNAMICS_MODEL), "rotation": info(paths.ROTATION_MODEL)},
            "candidates": candidates}


# ----------------------------------------------------------------------
#  Läufe (Sammeln / Trainieren)
# ----------------------------------------------------------------------
class Job:
    """Ein Skript als eigener Prozess; die Ausgabe wird mitgelesen und ausgewertet."""

    SAMPLE_RE = re.compile(r"^Sample (\d+)/(\d+):\s*(.*)$")
    RESULT_RE = re.compile(r"^\s*->\s*(.*)$")
    HOME_RE = re.compile(r"Startpunkt: x=([-\d.]+) y=([-\d.]+)")
    RETURN_RE = re.compile(r"^\s*\[Rückkehr (\d+)\]\s*(.*)$")
    MODEL_RE = re.compile(r"^Modell gespeichert: (.+)$")

    def __init__(self, kind: str, title: str, args: list[str], on_line: Callable[[str], None],
                 stdin_text: str = "", info: dict | None = None) -> None:
        self.kind = kind                  # collect | train
        self.title = title
        self.args = args
        self.info = info or {}
        self.state = "running"            # running | done | failed | stopped
        self.progress = (0, 0)
        self.current = ""
        self.last_result = ""
        self.home: tuple[float, float] | None = None
        self.error = ""
        self.output_file = ""
        self.started = time.time()
        self.ended: float | None = None
        self._on_line = on_line
        self._stop_requested = False
        self._finished_text = False
        self._last_lines: deque[str] = deque(maxlen=15)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
        # eigene Sitzung: Fenster zu / Strg+C im Terminal erreichen das Skript nicht direkt, sondern
        # trainer.py schickt gezielt STOPP (SIGINT) -> das Skript hält den Roboter an
        self.proc = subprocess.Popen([sys.executable, "-u", *args], cwd=paths.ROOT, env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, errors="replace", start_new_session=True)
        if stdin_text:
            self.proc.stdin.write(stdin_text)   # z. B. "START" für die Sicherheitsabfrage der Sammelskripte
            self.proc.stdin.flush()
        threading.Thread(target=self._read, daemon=True, name=f"job-{kind}").start()

    @property
    def running(self) -> bool:
        return self.state == "running"

    def stop(self) -> None:
        """Wie Strg+C: Die Skripte stoppen den Roboter und schließen ihre Dateien."""
        if not self.running or self.proc.poll() is not None:
            return
        self._stop_requested = True
        self.proc.send_signal(signal.SIGINT)
        threading.Thread(target=self._ensure_stopped, daemon=True).start()

    def _ensure_stopped(self) -> None:
        try:
            self.proc.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            say("error", f"{self.title}: reagiert nicht auf STOPP – Prozess wird beendet.")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _read(self) -> None:
        for raw in self.proc.stdout:
            line = raw.rstrip("\n")
            self._parse(line)
            self._last_lines.append(line)
            self._on_line(line)
        code = self.proc.wait()
        self.ended = time.time()
        if self._stop_requested or self.state == "stopped":
            self.state = "stopped"
        elif self.state == "running":
            if code == 0 and not self.error:
                self.state = "done"
            else:
                self.state = "failed"
                if not self.error:
                    tail = [l for l in self._last_lines if l.strip()]
                    self.error = tail[-1] if tail else f"Abbruch mit Code {code}"
        text = {"done": "fertig", "failed": f"FEHLER: {self.error}", "stopped": "gestoppt"}[self.state]
        say("ok" if self.state == "done" else "error", f"{self.title}: {text}")

    def _parse(self, line: str) -> None:
        if m := self.SAMPLE_RE.match(line):
            self.progress = (int(m.group(1)), int(m.group(2)))
            self.current = m.group(3)
        elif m := self.RESULT_RE.match(line):
            self.last_result = m.group(1)
        elif m := self.RETURN_RE.match(line):
            self.current = f"Rückkehr zum Start ({m.group(2)})"
        elif m := self.HOME_RE.search(line):
            self.home = (float(m.group(1)), float(m.group(2)))
        elif m := self.MODEL_RE.match(line):
            self.output_file = m.group(1).strip()
        elif line.startswith("Kritischer Fehler"):
            self.error = line.split("Grund:", 1)[-1].strip()
            self.state = "failed"
        elif line.startswith("Manuell abgebrochen"):
            self.state = "stopped"
        elif line.startswith("Fertig!"):
            self.current = "fertig"

    def snapshot(self) -> dict:
        return {
            "kind": self.kind, "title": self.title, "state": self.state, "progress": self.progress,
            "current": self.current, "last_result": self.last_result, "home": self.home, "error": self.error,
            "output_file": Path(self.output_file).name if self.output_file else "", "info": self.info,
            "started": self.started, "ended": self.ended,
        }


# ----------------------------------------------------------------------
#  Server
# ----------------------------------------------------------------------
class TrainerServer:
    """Bewegungstraining: Ortungsbild, Policy-Auswahl, Sammeln, Trainieren, Modelle übernehmen."""

    def __init__(self, port: int, on_quit: Callable[[], None]) -> None:
        self.port = port
        self.on_quit = on_quit
        self.policies = load_policies()
        self.data = DataFiles()
        self.feed = MqttFeed("aruco", "SID-Ortung (Arducam, ArUco)", f"mqtt:{TOPIC_PREVIEW}", TOPIC_PREVIEW)
        self.job: Job | None = None
        self.lines: deque[tuple[int, str]] = deque(maxlen=MAX_LINES)
        self.line_seq = 0
        self.tracker_status = ""
        self.pose: tuple[float, float, float] | None = None
        self.pose_time = 0.0
        self.station_running = False
        self.stopping = threading.Event()
        self.url = f"http://localhost:{port}/"
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID)
        self._station_url = self._station_frontend_url()

    # ---------- Start / Ende ----------
    def start(self) -> str:
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), _make_handler(self))
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True, name="trainer-http").start()
        self._start_mqtt()
        self.feed.start(self.stopping)   # liest das Ortungsbild auch lokal, falls kein Broker
        threading.Thread(target=self._watch_station, daemon=True, name="station-check").start()
        say("ok", f"Bewegungstraining läuft: {self.url}")
        return self.url

    def stop(self) -> None:
        if self.job is not None and self.job.running:
            say("step", "Laufender Lauf wird gestoppt ...")
            self.job.stop()
            try:
                self.job.proc.wait(timeout=STOP_TIMEOUT_S + 4)
            except subprocess.TimeoutExpired:
                pass
        self.stopping.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        self._mqtt.disconnect()
        self._mqtt.loop_stop()

    def _start_mqtt(self) -> None:
        def on_connect(client, userdata, flags, reason_code, properties):
            if not reason_code.is_failure:
                client.subscribe([(TOPIC_PREVIEW, 0), (TOPIC_POSE, 0), (TOPIC_TRACKER_STATUS, 1)])

        def on_message(client, userdata, msg):
            if msg.topic == TOPIC_PREVIEW:
                self.feed.put_mqtt(bytes(msg.payload))
            elif msg.topic == TOPIC_POSE:
                try:
                    d = json.loads(msg.payload)
                    self.pose = (float(d["x"]), float(d["y"]), float(d["theta"]))
                    self.pose_time = time.time()
                except (ValueError, KeyError):
                    pass
            elif msg.topic == TOPIC_TRACKER_STATUS:
                self.tracker_status = msg.payload.decode(errors="replace").strip()

        self._mqtt.on_connect = on_connect
        self._mqtt.on_message = on_message
        self._mqtt.reconnect_delay_set(min_delay=1, max_delay=10)
        self._mqtt.connect_async(BROKER, PORT, keepalive=30)
        self._mqtt.loop_start()

    @staticmethod
    def _station_frontend_url() -> str:
        port = load_config(paths.ROOT / "config" / "frontend.yaml")["server"]["port"]
        return f"http://localhost:{port}/api/config"

    def _watch_station(self) -> None:
        """Läuft die Station (main.py)? Dann darf hier kein echter Roboter bewegt werden."""
        while not self.stopping.wait(STATION_CHECK_S):
            try:
                with urllib.request.urlopen(self._station_url, timeout=0.8):
                    running = True
            except OSError:
                running = False
            if running != self.station_running:
                self.station_running = running
                say("error" if running else "ok", "Station (main.py) läuft – Sammeln mit dem echten Roboter gesperrt."
                    if running else "Station (main.py) läuft nicht mehr – Sammeln wieder möglich.")

    # ---------- Ausgabe ----------
    def add_line(self, text: str) -> None:
        with self._lock:
            self.line_seq += 1
            self.lines.append((self.line_seq, text))

    def lines_after(self, seq: int) -> list[tuple[int, str]]:
        with self._lock:
            if seq < 0:
                return list(self.lines)[-FIRST_LINES:]
            return [item for item in self.lines if item[0] > seq]

    # ---------- Befehle ----------
    def command(self, data: dict) -> str:
        cmd = str(data.get("cmd", ""))
        if cmd == "stop":
            if self.job is None or not self.job.running:
                return "Es läuft nichts."
            say("error", f"STOPP: {self.job.title}")
            self.job.stop()
            return "STOPP gesendet"
        if cmd == "quit":
            say("info", "Bewegungstraining beenden vom Frontend.")
            self.on_quit()
            return "Programm wird beendet"
        if cmd == "activate":
            return self._activate(str(data.get("candidate", "")))
        if self.job is not None and self.job.running:
            raise ValueError(f"Es läuft schon: {self.job.title}. Erst fertig werden lassen oder STOPP.")
        if cmd == "collect":
            return self._collect(data)
        if cmd == "train":
            return self._train(data)
        raise ValueError(f"Unbekannter Befehl '{cmd}'")

    def _collect(self, data: dict) -> str:
        key = str(data.get("policy", ""))
        policy = self.policies.get(key)
        if policy is None:
            raise ValueError(f"Unbekannte Policy '{key}'")
        sim = bool(data.get("sim"))
        samples = int(data.get("samples") or policy["samples"])
        if not 1 <= samples <= 5000:
            raise ValueError("Anzahl Samples: 1 … 5000")
        if not sim:
            if self.station_running:
                raise ValueError("Die Station (main.py) läuft – beide würden den SID steuern. Erst die Station beenden.")
            if self.tracker_status != "tracking" or time.time() - self.pose_time > 1.0:
                raise ValueError("Die Ortung verfolgt den SID nicht (keine aktuelle Pose). SID ins Kamerabild stellen.")
        args = [str(SCRIPTS / policy["script"]), "--samples", str(samples)]
        if sim:
            args.append("--sim")
        if key == "fine":
            groups = [g for g in data.get("groups") or [] if g in policy["groups"]]
            if not groups:
                raise ValueError("Mindestens eine Gruppe wählen.")
            args += ["--groups", ",".join(groups)]
        title = f"Sammeln: {policy['title']}" + (" (Simulation)" if sim else "")
        self.add_line(f"===== {title} – {samples} Samples – {time.strftime('%H:%M:%S')} =====")
        say("step", f"{title} startet ({samples} Samples).")
        self.job = Job("collect", title, args, self.add_line, stdin_text="" if sim else "START\n",
                       info={"policy": key, "sim": sim, "samples": samples, "limit": policy["limit"],
                             "workspace": policy["workspace"]})
        return "gestartet"

    def _train(self, data: dict) -> str:
        model = str(data.get("model", ""))
        if model not in MODEL_LABELS:
            raise ValueError(f"Unbekanntes Modell '{model}'")
        files = [self.data.resolve(str(name)) for name in data.get("files") or []]
        if not files:
            raise ValueError("Mindestens einen Datensatz wählen.")
        if model == "rotation" and any(csv_kind(f) != "rotation" for f in files):
            raise ValueError("Das Drehmodell braucht Dreh-Datensätze (rotation_dataset / nav_rotation_log).")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = paths.CANDIDATES / f"{model}_model_{stamp}.joblib"
        script = "train_dynamics.py" if model == "dynamics" else "train_rotation.py"
        args = [str(SCRIPTS / script), "--csv", *map(str, files), "--out", str(out)]
        if data.get("no_plots"):
            args.append("--no-plots")
        if data.get("no_cv"):
            args.append("--no-cv")
        title = f"Trainieren: {MODEL_LABELS[model]}"
        self.add_line(f"===== {title} – {len(files)} Datei(en) – {time.strftime('%H:%M:%S')} =====")
        say("step", f"{title} startet ({', '.join(f.name for f in files)}).")
        self.job = Job("train", title, args, self.add_line, info={"model": model, "files": [f.name for f in files]})
        return "gestartet"

    @staticmethod
    def _activate(name: str) -> str:
        candidate = paths.CANDIDATES / Path(name).name
        if not candidate.is_file() or candidate.suffix != ".joblib":
            raise ValueError(f"Kandidat '{name}' nicht gefunden")
        model = "rotation" if candidate.name.startswith("rotation") else "dynamics"
        target = paths.ROTATION_MODEL if model == "rotation" else paths.DYNAMICS_MODEL
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if target.is_file():
            backup = paths.MODEL_ARCHIVE / f"{target.stem}_bis_{stamp}.joblib"
            shutil.copy2(target, backup)
            say("info", f"Bisheriges {MODEL_LABELS[model]} gesichert: models/archiv/{backup.name}")
        shutil.copy2(candidate, target)
        say("ok", f"{MODEL_LABELS[model]} übernommen: {candidate.name} -> models/{target.name} "
                  f"(gilt ab dem nächsten Start der Station).")
        return "übernommen"

    # ---------- Zustand ----------
    def client_config(self) -> dict:
        return {"policies": self.policies, "models": MODEL_LABELS,
                "hotkeys": {"stop": "Space", "quit": "F10"}}

    def snapshot(self) -> dict:
        pose_age = time.time() - self.pose_time if self.pose_time else None
        plots = sorted(paths.PLOTS.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        return {
            "job": self.job.snapshot() if self.job is not None else None,
            "tracker": {"status": self.tracker_status, "pose": self.pose, "pose_age": pose_age},
            "broker": self._mqtt.is_connected(),
            "station_running": self.station_running,
            "feed": self.feed.info(2.0),
            "datasets": self.data.list(),
            "models": model_info(),
            "plots": [{"name": p.name, "mtime": p.stat().st_mtime} for p in plots[:60]],
            "time": time.time(),
        }


def _make_handler(server: TrainerServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            logger.trace("trainer: " + fmt % args)

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(data, default=str).encode(), "application/json")

        def _stream_headers(self, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in STATIC_FILES:
                file = WEB_DIR / STATIC_FILES[path]
                self._send(HTTPStatus.OK, file.read_bytes(), CONTENT_TYPES[file.suffix])
            elif path == "/api/config":
                self._json(server.client_config())
            elif path == "/api/events":
                self._events()
            elif path == "/feed.mjpg":
                self._mjpeg()
            elif path.startswith("/plots/") and re.fullmatch(r"/plots/[\w.\-]+\.png", path):
                file = paths.PLOTS / path[len("/plots/"):]
                if file.is_file():
                    self._send(HTTPStatus.OK, file.read_bytes(), "image/png")
                else:
                    self._json({"error": "nicht gefunden"}, HTTPStatus.NOT_FOUND)
            else:
                self._json({"error": "nicht gefunden"}, HTTPStatus.NOT_FOUND)

        def _events(self) -> None:
            self._stream_headers("text/event-stream")
            seq, first = -1, True
            try:
                while not server.stopping.is_set():
                    snap = server.snapshot()
                    new = server.lines_after(seq)
                    if new:
                        seq = new[-1][0]
                    elif seq == -1:
                        seq = 0
                    snap["lines"] = [text for _, text in new]
                    snap["first"] = first          # neue Verbindung: Ausgabe im Frontend neu füllen
                    first = False
                    self.wfile.write(f"data: {json.dumps(snap, default=str)}\n\n".encode())
                    self.wfile.flush()
                    server.stopping.wait(1.0 / SNAPSHOT_HZ)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _mjpeg(self) -> None:
            self._stream_headers("multipart/x-mixed-replace; boundary=frame")
            feed = server.feed
            feed.attach(+1)
            seq = -1
            try:
                while not server.stopping.is_set():
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

        def do_POST(self):
            if self.path != "/api/cmd":
                self._json({"error": "nicht gefunden"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                data = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("JSON-Objekt erwartet")
                message = server.command(data)
            except ValueError as e:
                self._json({"ok": False, "error": str(e)}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"ok": True, "message": message})

    return Handler
