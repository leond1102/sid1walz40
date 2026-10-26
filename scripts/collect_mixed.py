"""
collect_mixed.py — gemischte Bewegungen (mehrere Achsen gleichzeitig)

  fwd_* / bwd_*      : vor- bzw. rückwärts + seitlich  (schräg laufen, Kurs ohne Drehen korrigieren)
  arc_fwd_* / arc_bwd_* : vor- bzw. rückwärts + drehen   (Kurvenlauf)
  side_arc_*         : seitlich + drehen

Die Namen fwd_*/bwd_* erkennt navigate.py automatisch als "gerade Familie" für lange Läufe.
Richtungskonvention wie in den anderen Scripts: "right" = +SPEED_Y, "yaw left" = +SPEED_YAW.
Alle Achsen im Slow-Modus mit den Werten aus den alten Scripts (kleinere Werte werden ignoriert).

Nutzung:
    python collect_mixed.py
    python collect_mixed.py --samples 600
    python collect_mixed.py --sim

Ausgabe: mixed_dataset.csv (+ mixed_return_log.csv), Format wie motion_dataset.csv
"""
from collect_common import SPEED_X, SPEED_Y, SPEED_YAW, run_collection

NUM_SAMPLES = 500    # ~50 pro Aktion

X, Y, W = SPEED_X, SPEED_Y, SPEED_YAW

ACTIONS = [
    # vorwärts + seitlich
    ("fwd_right",       ( X,  Y,       0)),
    ("fwd_left",        ( X, -Y,       0)),
    # rückwärts + seitlich
    ("bwd_right",       (-X,  Y,       0)),
    ("bwd_left",        (-X, -Y,       0)),
    # Kurven: vor/zurück + drehen
    ("arc_fwd_left",    ( X,  0,  W)),
    ("arc_fwd_right",   ( X,  0, -W)),
    ("arc_bwd_left",    (-X,  0,  W)),
    ("arc_bwd_right",   (-X,  0, -W)),
    # seitlich + drehen
    ("side_arc_right",  ( 0,  Y, -W)),
    ("side_arc_left",   ( 0, -Y,  W)),
]

DURATIONS = {name: ((0.3, 3.5) if cmd[0] != 0 else (0.3, 2.5)) for name, cmd in ACTIONS}

if __name__ == "__main__":
    run_collection("Gemischte Bewegungen", ACTIONS, DURATIONS,
                   "mixed_dataset.csv", "mixed_return_log.csv", NUM_SAMPLES)
