"""
collect_linear.py — geradlinige Bewegungen (ohne Drehbefehl)

  forward / backward / left / right   — alles im Slow-Modus (Werte wie in den alten Scripts:
  SPEED_X = 15000, SPEED_Y = 20000). Kleinere Werte ignoriert der Roboter teilweise.

Dauern bis 4 s (vor/zurück) — lange Läufe werden nur gewählt, wenn sie laut grober Schätzung
im Arbeitsbereich enden (WORKSPACE_RADIUS in collect_common.py).

Nutzung:
    python collect_linear.py
    python collect_linear.py --samples 300
    python collect_linear.py --sim

Ausgabe: linear_dataset.csv (+ linear_return_log.csv), Format wie motion_dataset.csv
"""
from collect_common import SPEED_X, SPEED_Y, run_collection

NUM_SAMPLES = 400    # ~100 pro Aktion

ACTIONS = [
    ("forward",        ( SPEED_X,        0,            0)),
    ("backward",       (-SPEED_X,        0,            0)),
    ("right",          ( 0,              SPEED_Y,      0)),
    ("left",           ( 0,             -SPEED_Y,      0)),
]

DURATIONS = {
    "forward": (0.3, 4.0), "backward": (0.3, 4.0),
    "right": (0.3, 3.0), "left": (0.3, 3.0),
}

if __name__ == "__main__":
    run_collection("Geradlinige Bewegungen", ACTIONS, DURATIONS,
                   "linear_dataset.csv", "linear_return_log.csv", NUM_SAMPLES)
