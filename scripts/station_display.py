"""Anzeige der Station: farbige Konsole, Ortungsfenster und Kamerafenster.

Konsole:
    setup_logging() leitet alle Details (Navigation, MQTT, ...) in eine Logdatei.
    Im Terminal erscheinen nur die leicht verständlichen Meldungen der Station
    (say) sowie Warnungen und Fehler der Geräte.

    say("step", "SID läuft zur Dockingstation")    ▶ blau    etwas beginnt
    say("ok",   "Stütze abgelegt")                  ✔ grün    etwas ist fertig
    say("wait", "Bitte Stütze aufs Band stellen")   ⏳ gelb    Station wartet
    say("phase","AUSLIEFERN")                       ═ lila    neuer Abschnitt
    say("error","Docken fehlgeschlagen")            ✖ rot     Fehler

Fenster:
    start_tracker()        startet aruco_tracker_calibrated.py (SID-Ortung), falls er nicht schon läuft
                           (für das Frontend ohne Fenster, mit Vorschaubild per MQTT)
    run_camera_window()    Live-Bild der YOLO-Kamera mit den letzten Erkennungen und dem Stationszustand
"""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from loguru import logger

from F_stilt_erkennung_YOLO import StiltDetector, annotate, window_open

SCRIPT_DIR = Path(__file__).resolve().parent
TRACKER_SCRIPT = SCRIPT_DIR / "aruco_tracker_calibrated.py"
CAMERA_WINDOW = "YOLO-Kamera"
UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"})

# Art der Meldung -> (Farbe, Symbol)
STYLES = {
    "phase": ("magenta", "═"),
    "step": ("light-blue", "▶"),
    "ok": ("green", "✔"),
    "wait": ("yellow", "⏳"),
    "info": ("white", "•"),
    "error": ("red", "✖"),
}


# ----------------------------------------------------------------------
#  Konsole
# ----------------------------------------------------------------------
class _ToLoguru(logging.Handler):
    """Leitet Meldungen des Standard-logging (navigate.py) an loguru weiter."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def _console_format(record) -> str:
    kind = record["extra"].get("kind")
    if kind is None:  # Warnung oder Fehler eines Geräts
        color = "red" if record["level"].no >= logging.ERROR else "yellow"
        return f"<dim>{{time:HH:mm:ss}}</dim>  <{color}>! {{message}}</{color}>\n{{exception}}"
    color, symbol = STYLES[kind]
    if kind == "phase":
        return f"\n<{color}><bold>══════ {{message}} ══════</bold></{color}>\n"
    return f"<dim>{{time:HH:mm:ss}}</dim>  <{color}>{symbol} {{message}}</{color}>\n{{exception}}"


def _console_filter(record) -> bool:
    return "kind" in record["extra"] or record["level"].no >= logging.WARNING


def setup_logging(log_dir: Path, prefix: str = "station") -> Path:
    """Richtet Konsole (nur Stationsmeldungen + Warnungen) und Logdatei (alles) ein.

    Args:
        log_dir: Ordner der Logdateien.
        prefix: Anfang des Dateinamens (station, trainer, ...).

    Returns:
        Pfad der Logdatei.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logger.remove()
    logger.add(sys.stderr, format=_console_format, filter=_console_filter, colorize=True)
    logger.add(log_file, level="DEBUG", encoding="utf-8")

    logging.basicConfig(level=logging.INFO, handlers=[_ToLoguru()], force=True)
    logging.getLogger("transitions").setLevel(logging.WARNING)
    return log_file


def say(kind: str, text: str) -> None:
    """Gibt eine leicht verständliche Stationsmeldung farbig aus (kind siehe STYLES)."""
    level = "ERROR" if kind == "error" else "INFO"
    logger.bind(kind=kind).log(level, text)


# ----------------------------------------------------------------------
#  Ortungsfenster (ArUco-Tracker)
# ----------------------------------------------------------------------
def tracker_running() -> bool:
    """True, wenn schon ein aruco_tracker-Prozess läuft (egal aus welchem Ordner)."""
    result = subprocess.run(["pgrep", "-f", "aruco_tracker"], capture_output=True, check=False)
    return result.returncode == 0


def start_tracker(window: bool = True, preview: bool = False) -> subprocess.Popen | None:
    """Startet die SID-Ortung, falls sie nicht schon läuft.

    Args:
        window: Eigenes Ortungsfenster öffnen.
        preview: Debugbild per MQTT senden (Anzeige im Frontend).

    Returns:
        Der gestartete Prozess (wird beim Beenden gestoppt) oder None, wenn er schon lief.
    """
    if tracker_running():
        say("info", "SID-Ortung läuft bereits." + (" Für das Bild im Frontend muss sie mit --preview laufen."
                                                    if preview else ""))
        return None
    args = [sys.executable, str(TRACKER_SCRIPT)]
    if not window:
        args.append("--headless")
    if preview:
        args.append("--preview")
    say("step", "Starte SID-Ortung" + (" (Fenster öffnet sich) ..." if window else " ..."))
    return subprocess.Popen(args, cwd=SCRIPT_DIR)


# ----------------------------------------------------------------------
#  Kamerafenster (YOLO)
# ----------------------------------------------------------------------
def _draw_status(image, lines: list[str]):
    """Schreibt die Statuszeilen mit dunklem Hintergrund oben ins Bild."""
    for number, text in enumerate(lines):
        text = text.translate(UMLAUTS)  # OpenCV kann keine Umlaute zeichnen
        y = 16 + number * 16
        cv2.putText(image, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def run_camera_window(detector: StiltDetector, status_lines: Callable[[], list[str]],
                      should_stop: Callable[[], bool], on_emergency: Callable[[], None]) -> None:
    """Zeigt das Live-Bild der YOLO-Kamera, bis should_stop() True liefert oder 'q' gedrückt wird.

    Muss im Hauptthread laufen (OpenCV-Fenster). Tasten im Fenster:
        LEERTASTE   NOT-AUS (on_emergency)
        q / ESC     Programm beenden

    Args:
        detector: Verbundene Erkennung (liefert Bild und letzte Erkennungen).
        status_lines: Liefert die Textzeilen, die oben ins Bild geschrieben werden.
        should_stop: Liefert True, wenn das Programm beendet werden soll.
        on_emergency: Wird bei LEERTASTE aufgerufen.
    """
    cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
    try:
        while not should_stop():
            image = detector.read_image(fresh=False) if detector.is_connected() else None
            if image is None:
                view = np.zeros((340, 340, 3), np.uint8)
                status = status_lines() + ["Kamera nicht verfuegbar"]
                time.sleep(0.1)
            else:
                view = annotate(image, detector.last_detections)
                status = status_lines()
            cv2.imshow(CAMERA_WINDOW, _draw_status(view, status))

            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                on_emergency()
            elif key in (ord("q"), 27):
                break
            if not window_open(CAMERA_WINDOW):
                break
    finally:
        cv2.destroyAllWindows()
