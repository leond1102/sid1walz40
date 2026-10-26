"""
trainer.py – Bewegungstraining des SID mit eigenem Frontend (unabhängig von main.py).

Zeigt das Bild der SID-Ortung (Arducam, ArUco), lässt eine Trainings-Policy wählen und starten
(Sammelskripte in scripts/collect_*.py), trainiert danach Lauf- oder Drehmodell
(scripts/train_*.py) und übernimmt ein neues Modell auf Knopfdruck. Details: scripts/trainer_server.py.

Die Station (main.py) und dieses Programm dürfen den SID nicht gleichzeitig steuern: Läuft die
Station, sperrt das Training das Sammeln mit dem echten Roboter (Simulation geht immer).

Aufruf:
    python trainer.py               Frontend im Vollbild (http://localhost:8081/)
    python trainer.py --no-window   kein Fenster öffnen, nur die Adresse
    python trainer.py --port 8090   anderer Port

Tasten im Frontend: Leertaste = STOPP (laufenden Lauf sofort anhalten), F10 = Beenden.
"""

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

from station_display import say, setup_logging, start_tracker  # noqa: E402
from station_frontend import open_browser  # noqa: E402
from trainer_server import TrainerServer  # noqa: E402

DEFAULT_PORT = 8081


def main() -> None:
    ap = argparse.ArgumentParser(description="Bewegungstraining des SID")
    ap.add_argument("--no-window", action="store_true", help="kein Fenster öffnen")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port des Frontends (Standard {DEFAULT_PORT})")
    args = ap.parse_args()
    log_file = setup_logging(ROOT / "logs", prefix="trainer")

    say("phase", "BEWEGUNGSTRAINING")
    say("info", f"Alle Details stehen in {log_file.relative_to(ROOT)}")
    quit_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: quit_event.set())   # kill
    signal.signal(signal.SIGHUP, lambda *_: quit_event.set())    # Terminalfenster geschlossen

    server = TrainerServer(args.port, on_quit=quit_event.set)
    try:
        server.start()
    except OSError as e:
        say("error", f"Frontend nicht verfügbar ({e}). Läuft das Bewegungstraining schon?")
        return
    tracker = start_tracker(window=False, preview=True)   # Ortungsbild fürs Frontend (falls nicht schon läuft)
    browser = None if args.no_window else open_browser(server.url)
    say("info", f"Frontend: {server.url}  (Fenster zu? Einfach diese Adresse im Browser öffnen.)")
    try:
        while not quit_event.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        say("step", "Bewegungstraining endet ...")
        server.stop()
        if tracker is not None:
            tracker.terminate()
        if browser is not None and browser.poll() is None:
            browser.terminate()
        say("ok", "Beendet.")


if __name__ == "__main__":
    main()
