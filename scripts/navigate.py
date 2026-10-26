"""
navigate.py
Fährt den Lite3 mit gelernten Modellen (MPC mit Re-Planning) zu Zielkoordinaten.
Alle Koordinaten sind ABSOLUTE Kamerakoordinaten (m, Grad, gleiche Konvention wie der Tracker).

Modelle:
    dynamics_model.joblib  (train_dynamics.py)  -> alle Lauf-Primitive (vor/zurück/seitlich/Kombis/…)
    rotation_model.joblib  (train_rotation.py)  -> präzise Drehungen: single, flick, flick_pause
    Fehlt rotation_model.joblib (oder --no-rot), wird mit yaw_left/yaw_right gedreht.

Sicherheitszone:
    Alles mit x < SAFETY_ZONE_X (1.2 m) ist gesperrt.
    - Planer/Grobphase: kein Primitiv darf laut Modell in der Zone enden. Da die Zone eine Halbebene
      ist, kann ein gerader Lauf zwischen zwei Punkten außerhalb sie auch nicht schneiden.
    - Ziele in der Zone werden abgelehnt.
    - Live-Überwachung während jeder Bewegung: Pose in der Zone (mit Marge) -> sofortiger Stopp.
    - Einzige Ausnahme: dock() / undock().

Docking (siehe DOCK_*):
    1. Vorposition (1.2, -0.25, 180°) anfahren und so genau wie beim Anfahren ausrichten
    2. AB HIER IST DREHEN GESPERRT (Planer + harte Prüfung jedes gesendeten Befehls)
       EIN gerader Vorwärtslauf bis zur Dockmitte; Dauer aus der gemessenen Entfernung:
       T = Totzeit + Strecke / Geschwindigkeit (DOCK_SPEED/DOCK_DEADTIME). Keine Korrekturschritte.
    3. undock(): EIN gerader Rückwärtslauf bis x > UNDOCK_EXIT_X. Erst dann ist Drehen wieder erlaubt.
    Steht der Roboter beim Start in der Zone, ist nur undock() erlaubt.

Ablauf pro Schritt (Receding Horizon):
    1. Pose messen (gemittelt im Stand, gleiches Zeitfenster wie im Datensatz!)
    2. Befehl wählen:
       Grobphase (weiter als APPROACH_DIST): EINE Drehung, dann EIN langer gerader Lauf bis STOP_SHORT vors Ziel.
       Feinphase: Beam-Search über alle Primitive (Kosten = erwarteter quadratischer Zielfehler + Strafen).
       Endausrichtung (falls Zielwinkel): gezielte Drehung aus dem Drehmodell.
    3. Nur den ERSTEN Befehl ausführen, stoppen, zurück zu 1.

Einbindung in die State Machine:
    from navigate import NavigationController, NavStatus
    nav = NavigationController.create(sim=False)          # oder mit tracker=/commander= der Station
    nav.initialize()                                      # Startpose messen
    res = nav.go_to(2.0, -0.5, 90)                        # blockierend -> NavResult
    res = nav.dock();  res = nav.undock()
    res = nav.dock_pre()                                  # nur Dock-Vorposition anfahren (auf Übergabe warten)
    res = nav.approach_pre(2.0, 0.0, 0);  res = nav.approach_final(2.0, 0.0, 0)   # Anfahren in zwei Hälften
    res = nav.back_off(0.3)                               # gerade rückwärts (außerhalb der Zone)
    res = nav.enter_zone(1.0, 0.25);  res = nav.undock()  # gerade in die Zone (Abgabe) und wieder raus
    nav.run_async(nav.dock, callback=on_result)           # nicht-blockierend (eigener Thread)
    nav.abort()                                           # aus beliebigem Thread: Befehl abbrechen
    nav.emergency_stop()                                  # abbrechen + sofort stoppen
    nav.location                                          # "FREE" | "IN_ZONE" | "DOCKED"
    Ergebnis: NavResult(status=SUCCESS|FAILED|ABORTED|SAFETY_STOP|REJECTED|ERROR, message, pose)

CLI (zum Testen):
    python navigate.py --sim                         # Simulation, interaktive Befehle
    python navigate.py --sim --goals 2.0,0.3 d u     # Ziel, docken, undocken
    python navigate.py                               # echter Roboter
    Befehle:  x y [theta]                  Ziel anfahren
              a x y [winkel] [abstand]     Punkt aus fester Richtung anfahren (auch a:x,y,...)
              d | dock                     docken
              u | undock                   ausdocken
              s | status                   Pose / Zustand
              q                            beenden

Logs (für späteres Nachtrainieren):
    nav_log.csv           Lauf-Primitive, Format wie linear_dataset.csv   -> train_dynamics.py --csv ...
    nav_rotation_log.csv  Drehungen,      Format wie rotation_dataset.csv -> train_rotation.py --csv ...
"""
import argparse
import csv
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import joblib
import numpy as np

import paths
from dynamics_model import DynamicsModel, world_to_body, wrap_angle  # noqa: F401 (für joblib.load nötig)

try:
    import rotation_model  # noqa: F401 (für joblib.load des Drehmodells nötig)
except ImportError:
    rotation_model = None

log = logging.getLogger("navigate")

# ==========================================
# KONFIGURATION
# ==========================================
ROBOT_IP = "192.168.1.120"
MQTT_BROKER = "192.168.2.1"
TOPIC_POSE = "station/tracker/robot/pose"
MODEL_PATH = str(paths.DYNAMICS_MODEL)                  # models/ (siehe paths.py)
ROT_MODEL_PATH = str(paths.ROTATION_MODEL)
LOG_PATH = str(paths.NAV_LOGS / "nav_log.csv")         # data/nav_logs/
ROT_LOG_PATH = str(paths.NAV_LOGS / "nav_rotation_log.csv")

# Timing -- MUSS zum Datensatz passen, sonst stimmt das Modell nicht!
AVERAGING_START = 3.0
AVERAGING_END = 4.0
START_POSE_AVERAGE_WINDOW = 1.0
SEND_PERIOD = 0.05

# Ziel-Toleranzen
POS_TOLERANCE = 0.05          # m
THETA_TOLERANCE = 5.0         # Grad (nur wenn Ziel-Theta angegeben)
MAX_STEPS_PER_GOAL = 25

# Planer
HORIZON = 3                   # max. Anzahl Primitive pro Plan
BEAM_WIDTH = 60
DURATION_STEP = 0.05          # s, Auflösung der Dauer-Diskretisierung
THETA_WEIGHT = 0.01           # m pro Grad: 1° Fehler kostet so viel wie 1 cm
STEP_PENALTY = 0.0004         # m² pro Primitiv (~2 cm): verhindert sinnlose Mini-Korrekturen
TIME_PENALTY = 0.0001         # m² pro Sekunde Befehlsdauer
MIN_IMPROVEMENT = 0.0002      # m²: darunter lohnt sich kein weiterer Schritt

# Drehmodell
ROT_TARGET_STEP = 2.5         # Grad: Raster der Dreh-Primitive im Planer (-150° … +150°)
ROT_TARGET_MAX = 150.0        # Grad: größere Drehungen sind nicht eindeutig gemessen worden
ROT_SIGMA_FLOOR = 1.5         # Grad: Untergrenze für σ(dθ) (GP ist bei wenigen Samples zu optimistisch)
ROT_POS_SIGMA_FLOOR = 0.02    # m: Untergrenze für σ(dx), σ(dy) der Dreh-Primitive

# Schema "Drehen -> Geradeaus -> Feinanflug" (Grobphase)
APPROACH_DIST = 0.30          # m: ab hier übernimmt der MPC-Feinanflug (alle Primitive)
STOP_SHORT = 0.10             # m: Geradeauslauf endet so weit VOR dem Ziel
COARSE_YAW_MIN_DIST = 0.50    # m: näher dran wird in der Grobphase nicht mehr gedreht -> Feinanflug
MAX_STRAIGHT_DURATION = 5.0   # s: lange Geradeausläufe über die Trainingsdauer hinaus (Extrapolation)
EXTRAP_LAT_FRAC = 0.05        # zusätzliche Streuung pro extrapoliertem Meter (vorsichtig)
MISS_ABS = 0.10               # m: Geradeauslauf OK, wenn er laut Modell so nah am Zielpunkt endet ...
MISS_FRAC = 0.08              # ... oder innerhalb dieses Anteils der Distanz (2 m -> 16 cm)
FINE_ACTIONS = None           # Feinanflug: None = alle Primitive; sonst Liste von Namen ("rotation" = Drehmodell)
EXCLUDE_ACTIONS = ["arc_fwd_left", "arc_fwd_right", "arc_bwd_left", "arc_bwd_right"]  # zu ungenau (5–12 cm)

# Anfahren aus fester Richtung (Modus "a")
APPROACH_ANGLE_DEFAULT = 0.0
APPROACH_STANDOFF = 0.5
APPROACH_STANDOFF_BAND = (0.40, 0.65)
APPROACH_LAT_TOL = 0.04               # m: seitliche Toleranz an der Vorposition
APPROACH_THETA_TOL = 4                # Grad: Ausrichtungstoleranz an der Vorposition
APPROACH_THETA_WEIGHT = 0.02          # m pro Grad im Planer während der Ausrichtung
APPROACH_RELAXED = (0.12, 12.0)       # (m, Grad): gelockerte Toleranzen an der Vorposition (mit Warnung; vorher 0.08, 8)
APPROACH_FINAL_ACTIONS = ["forward", "fwd_left", "fwd_right"]
APPROACH_FINAL_TOL = 0.02
APPROACH_OVERSHOOT_WEIGHT = 4.0
APPROACH_MAX_FINAL_STEPS = 2
APPROACH_ACCEPT = (0.15, 0.10, 15.0)  # (längs m, seitlich m, Grad): Anfahren gilt als erfolgreich
                                      #  (vorher 0.04, 0.08, 8)

# ------------------------------------------
# SICHERHEITSZONE (absolute Kamerakoordinaten)
# ------------------------------------------
SAFETY_ZONE_X = 1.2           # m: alles mit x < SAFETY_ZONE_X ist gesperrt
ZONE_POSE_TOL = 0.03          # m: gemessene Pose gilt erst ab x < SAFETY_ZONE_X - ZONE_POSE_TOL als "in der Zone"
                              #    (Messrauschen; darunter werden auch KEINE Drehungen mehr ausgeführt)
LIVE_ZONE_MARGIN = 0.05       # m: Live-Überwachung im Lauf stoppt ab x < Zonengrenze - Marge (Tracker-Rauschen)

# ------------------------------------------
# DOCKING (absolute Kamerakoordinaten)
# ------------------------------------------
DOCK_PRE_POSE = (1.20, -0.25, 180.0)   # Vorposition (x, y, θ) — liegt genau auf der Zonengrenze
DOCK_POSE = (0.15, -0.25)              # Zielpunkt im Dock (größeres x = weniger tief hinein)
DOCK_ANGLE = 180.0                     # Einfahrrichtung = Blickrichtung im Dock
DOCK_PRE_ALONG_TOL = 0.25              # m: Toleranz entlang der Andocklinie an der Vorposition
DOCK_PRE_LAT_TOL = APPROACH_LAT_TOL    # m: seitlich (wie beim Anfahren)
DOCK_PRE_THETA_TOL = APPROACH_THETA_TOL  # Grad (wie beim Anfahren)
DOCK_PRE_ZONE_ALLOWANCE = 0.03         # m: beim Ausrichten an der Vorposition darf der Planer so weit
                                       #    in die Zone (Vorposition liegt auf der Grenze)
DOCK_PRE_AIM_OFFSET = 0.12             # m: Vorposition wird so weit AUSSERHALB der Zonengrenze angezielt,
                                       # damit der Planer dort Platz für seitliche Korrekturen hat
                                       # (erlaubt ist sie trotzdem im Bereich ±DOCK_PRE_ALONG_TOL um die Grenze)
DOCK_PRE_RETRY_BACKOFF = 0.15          # m: klappt die Vorposition nicht, so weit gerade rückwärts und neu versuchen
DOCK_ALLOW_RELAXED = False             # True: auch mit APPROACH_RELAXED-Toleranzen einfahren (nicht empfohlen)

# Ein-/Ausfahren: EIN gerader Lauf, Dauer T = Totzeit + Strecke / Geschwindigkeit (hartcodiert, kein Planer)
DOCK_RUN_ACTION = "forward"            # Befehl (Achswerte aus model.command) fürs Einfahren — KEIN Yaw
UNDOCK_RUN_ACTION = "backward"         # Befehl fürs Ausfahren — KEIN Yaw
DOCK_SPEED = None                      # m/s vorwärts; None = aus dem Dynamikmodell geschätzt (siehe Log)
DOCK_DEADTIME = None                   # s bis der Roboter läuft; None = aus dem Dynamikmodell geschätzt
UNDOCK_SPEED = None                    # m/s rückwärts; None = aus dem Dynamikmodell geschätzt
UNDOCK_DEADTIME = None                 # s; None = aus dem Dynamikmodell geschätzt
DOCK_DISTANCE_OFFSET = 0.0             # m: + = etwas weiter laufen (in den Puffer), - = kürzer
DOCK_RUN_T_MAX = 8.0                   # s: Obergrenze für einen Lauf (Plausibilitätsgrenze)

DOCK_ACCEPT = (0.25, 0.10)             # (zu kurz max. m, seitlich m): Andocken gilt als erfolgreich (vorher 0.04, 0.05)
                                       #  (zu weit ist OK – der Puffer fängt ab; Grenze siehe DOCK_LIVE_OVERSHOOT)
DOCK_THETA_ABORT = 20.0                # Grad: Winkelabweichung im Dock -> Sicherheitsstopp (gemessen im Stand; vorher 10)
DOCK_THETA_LIVE_ABORT = 30.0           # Grad: dito live während der Bewegung (Tracker im Lauf unruhiger; vorher 15)
DOCK_LAT_ABORT = 0.20                  # m: seitliche Abweichung von der Andocklinie -> Sicherheitsstopp (vorher 0.10)
DOCK_LIVE_OVERSHOOT = 0.08             # m: live so weit über die Dockmitte -> Lauf wird regulär beendet
                                       #    (kein Fehler, nur früher gestoppt); None = aus
UNDOCK_EXIT_X = SAFETY_ZONE_X + 0.10   # m: ausgedockt (Drehen wieder erlaubt) erst ab x >= diesem Wert
UNDOCK_TARGET_OVERRUN = 0.05           # m: Rückwärtslauf zielt so weit hinter UNDOCK_EXIT_X
UNDOCK_LIVE_BACKSLIDE = 0.10           # m: läuft er live so weit tiefer in die Zone -> Sicherheitsstopp (vorher 0.05)
UNDOCK_MAX_STEPS = 3                   # max. Anzahl gerader Rückwärtsläufe (normal: 1)

# Sicherheit
MAX_DISTANCE_FROM_START = 4.0  # m, harter Radius um die Startpose (Planer + Not-Aus)
MAX_POSE_AGE = 1.5             # s

# Simulation
SIM_NOISE_SCALE = 1.0
SIM_MEAS_NOISE = (0.002, 0.002, 0.4)


# ==========================================
# ERGEBNISSE / AUSNAHMEN (Schnittstelle zur State Machine)
# ==========================================
class NavStatus(str, Enum):
    SUCCESS = "SUCCESS"          # Ziel erreicht
    FAILED = "FAILED"            # Ziel nicht erreicht (kein Schritt verbessert mehr, Toleranzen verfehlt, …)
    ABORTED = "ABORTED"          # von außen abgebrochen (abort()/emergency_stop())
    SAFETY_STOP = "SAFETY_STOP"  # Sicherheitsverletzung, Roboter wurde gestoppt
    REJECTED = "REJECTED"        # Befehl nicht zulässig (Ziel in Zone, gedockt, beschäftigt, …)
    ERROR = "ERROR"              # unerwarteter Fehler (Tracker weg, Kommunikation, …)


@dataclass
class NavResult:
    status: NavStatus
    message: str = ""
    pose: Optional[tuple] = None
    command: str = ""

    @property
    def ok(self):
        return self.status == NavStatus.SUCCESS


class NavAborted(Exception):
    """Abbruch von außen."""


class SafetyStop(Exception):
    """Sicherheitsverletzung -> Roboter gestoppt."""


class NavRejected(Exception):
    """Befehl im aktuellen Zustand nicht zulässig."""


class GuardStop(str):
    """Rückgabe eines Guards: Lauf regulär (früher) beenden – KEIN Sicherheitsstopp."""


# ==========================================
# HILFSFUNKTIONEN
# ==========================================
def pose_stats(poses):
    n = len(poses)
    if n == 0:
        return None
    xs, ys, ths = zip(*poses)
    mean_th = math.degrees(math.atan2(sum(math.sin(math.radians(t)) for t in ths),
                                      sum(math.cos(math.radians(t)) for t in ths)))
    mx, my = sum(xs) / n, sum(ys) / n
    return {
        "mean_x": mx, "mean_y": my, "mean_theta": mean_th,
        "std_x": math.sqrt(sum((x - mx) ** 2 for x in xs) / n),
        "std_y": math.sqrt(sum((y - my) ** 2 for y in ys) / n),
        "std_theta": math.sqrt(sum(wrap_angle(t - mean_th) ** 2 for t in ths) / n),
        "count": n,
    }


def end_positions(pose, dx_body, dy_body):
    """Endposition(en) in Weltkoordinaten nach Bewegung (dx_body, dy_body) aus pose (skalar oder Arrays)."""
    x, y, th = pose
    t = math.radians(th)
    c, s = math.cos(t), math.sin(t)
    return x + c * dx_body - s * dy_body, y + s * dx_body + c * dy_body


def line_frame(px, py, target, angle):
    """Fehler im Anfahr-KS: along > 0 = über den Punkt hinaus, lat = seitlich (+ links der Linie)."""
    u = (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
    ex, ey = px - target[0], py - target[1]
    return ex * u[0] + ey * u[1], -ex * u[1] + ey * u[0]


def weighted_line_cost(al, la, var=0.0, overshoot_weight=APPROACH_OVERSHOOT_WEIGHT):
    w = np.where(al > 0, overshoot_weight, 1.0)
    return (w * al) ** 2 + la ** 2 + var


def describe(ex):
    """Lesbare Beschreibung eines Befehls."""
    if ex["type"] == "dyn":
        return f"{ex['action']} {ex['T']:.2f}s"
    c = ex["cmd"]
    txt = f"{c['kind']} {c['direction']} {c['t1']:.2f}s"
    if not c["kind"].startswith("single"):
        txt += f"/{c['t2']:.2f}s"
    return txt + f" (erw. {c['dtheta']:+.1f}°)"


def _sleep(seconds, abort_event=None):
    """Schläft; mit abort_event unterbrechbar (-> NavAborted)."""
    if abort_event is None:
        time.sleep(seconds)
    elif abort_event.wait(seconds):
        raise NavAborted("Abbruch angefordert")


# ==========================================
# PLANER
# ==========================================
class PrimitivePlanner:
    """
    Beam-Search über Primitiv-Folgen mit vorberechneter Lookup-Tabelle.
    Ein Primitiv ist entweder ein Lauf-Befehl aus dem Dynamikmodell ({"type": "dyn", action, T})
    oder eine Drehung aus dem Drehmodell ({"type": "rot", cmd}).

    Sicherheitszone: zone_x (None = aus). Kein Primitiv darf laut Modell bei x < zone_x enden.
    allow_yaw=False sperrt alle Primitive mit Yaw-Anteil.
    """

    def __init__(self, model, rot_model=None):
        self.model = model
        self.rm = rot_model
        family, T_total, means, variances, execs, has_yaw = [], [], [], [], [], []

        # --- Lauf-Primitive (Dynamikmodell) ---
        use_dyn_yaw = self.rm is None
        for a in model.actions:
            if a in EXCLUDE_ACTIONS:
                continue
            if a.startswith("yaw_") and not use_dyn_yaw:
                continue  # Drehungen kommen aus dem Drehmodell
            lo, hi = model.duration_range[a]
            T = np.round(np.arange(lo, hi + 1e-9, DURATION_STEP), 3)
            m, s = model.predict(a, T)
            yaw = abs(model.command[a][2]) > 1e-9
            for k in range(len(T)):
                family.append(a)
                T_total.append(T[k])
                means.append(m[k])
                variances.append(s[k] ** 2)
                execs.append({"type": "dyn", "action": a, "T": float(T[k])})
                has_yaw.append(yaw)

        # --- Dreh-Primitive (Drehmodell) ---
        n_rot = 0
        if self.rm is not None:
            seen = set()
            for target in np.arange(-ROT_TARGET_MAX, ROT_TARGET_MAX + 1e-9, ROT_TARGET_STEP):
                if abs(target) < 1e-9:
                    continue
                c = self.rm.best_command(float(target))
                key = (c["kind"], c["direction"], round(c["t1"], 3), round(c["t2"], 3))
                if key in seen:
                    continue
                seen.add(key)
                family.append("rotation")
                gap = self.rm.groups[(c["kind"], c["direction"])].get("gap", 0.0)
                T_total.append(c["t1"] + (0.0 if c["kind"].startswith("single") else c["t2"] + gap))
                means.append(np.array([c["dx_body"], c["dy_body"], c["dtheta"]]))
                mean, std = self.rm.predict(c["kind"], c["direction"], c["t1"], c["t2"])
                variances.append(np.array([max(std[0, 0], ROT_POS_SIGMA_FLOOR) ** 2,
                                           max(std[0, 1], ROT_POS_SIGMA_FLOOR) ** 2,
                                           max(c["sigma_theta"], ROT_SIGMA_FLOOR) ** 2]))
                execs.append({"type": "rot", "cmd": c})
                has_yaw.append(True)
                n_rot += 1

        self.p_family = np.array(family)
        self.p_T = np.array(T_total, dtype=float)
        self.p_mean = np.vstack(means)          # (P, 3): dx_body, dy_body, dtheta
        self.p_var = np.vstack(variances)
        self.p_exec = execs
        self.p_yaw = np.array(has_yaw, bool)
        self.theta_weight = THETA_WEIGHT        # beim Anfahren/Docken vorübergehend erhöht
        self.zone_x = SAFETY_ZONE_X             # beim Docken vorübergehend geändert
        self.allow_yaw = True
        log.info(f"Planer: {len(self.p_T)} Primitive "
                 f"({len(self.p_T) - n_rot} Lauf-Primitive, {n_rot} Dreh-Primitive"
                 f"{'' if self.rm is not None else ' – ohne Drehmodell'})")

    def yaw_allowed_at(self, x):
        """Drehen ist nur erlaubt, wenn freigegeben UND die Pose nicht in der Sicherheitszone liegt."""
        return self.allow_yaw and x >= SAFETY_ZONE_X - ZONE_POSE_TOL

    def _cost(self, x, y, th, var_pos, var_th, n_steps, t_total, goal):
        gx, gy, gth = goal
        c = (x - gx) ** 2 + (y - gy) ** 2 + var_pos
        if gth is not None:
            c = c + self.theta_weight ** 2 * (wrap_angle(th - gth) ** 2 + var_th)
        return c + n_steps * STEP_PENALTY + t_total * TIME_PENALTY

    def plan(self, pose, goal, center, allowed=None):
        """
        pose: (x, y, theta) aktuell; goal: (x, y, theta|None); center: Mittelpunkt des Sicherheitsradius.
        allowed: optionale Liste erlaubter Primitiv-Familien (Aktionsnamen bzw. "rotation").
        Rückgabe: (Plan als Liste von Primitiv-Indizes, erwartete Kosten, Kosten ohne Handeln)
        """
        mask = np.ones(len(self.p_T), bool) if allowed is None else np.isin(self.p_family, allowed)
        if not self.yaw_allowed_at(pose[0]):
            mask &= ~self.p_yaw
        x0, y0, th0 = pose
        cost_stay = self._cost(x0, y0, th0, 0.0, 0.0, 0, 0.0, goal)

        bx = np.array([x0]); by = np.array([y0]); bth = np.array([th0])
        bvp = np.zeros(1); bvt = np.zeros(1); bt = np.zeros(1)
        bseq = [[]]
        best_cost, best_seq = cost_stay, []

        P = len(self.p_T)
        for depth in range(1, HORIZON + 1):
            B = len(bx)
            th_r = np.radians(bth)[:, None]
            c, s = np.cos(th_r), np.sin(th_r)
            dxb, dyb, dth = self.p_mean[None, :, 0], self.p_mean[None, :, 1], self.p_mean[None, :, 2]
            nx = bx[:, None] + c * dxb - s * dyb
            ny = by[:, None] + s * dxb + c * dyb
            nth = wrap_angle(bth[:, None] + dth)
            dist = np.hypot(dxb, dyb)
            nvt = bvt[:, None] + self.p_var[None, :, 2]
            nvp = (bvp[:, None] + self.p_var[None, :, 0] + self.p_var[None, :, 1]
                   + (dist * np.radians(np.sqrt(bvt[:, None]))) ** 2)
            nt = bt[:, None] + self.p_T[None, :]
            cost = self._cost(nx, ny, nth, nvp, nvt, depth, nt, goal)

            out = np.hypot(nx - center[0], ny - center[1]) > MAX_DISTANCE_FROM_START
            if self.zone_x is not None:
                out |= nx < self.zone_x                      # Sicherheitszone
            cost = np.where(out | ~mask[None, :], np.inf, cost)

            flat = cost.ravel()
            k = min(BEAM_WIDTH, int(np.isfinite(flat).sum()))
            if k == 0:
                break
            idx = np.argpartition(flat, k - 1)[:k]
            bi, pi = np.unravel_index(idx, (B, P))

            if flat[idx].min() < best_cost:
                j = idx[np.argmin(flat[idx])]
                b, p = np.unravel_index(j, (B, P))
                best_cost = float(flat[j])
                best_seq = bseq[b] + [int(p)]

            bx, by, bth = nx[bi, pi], ny[bi, pi], nth[bi, pi]
            bvp, bvt, bt = nvp[bi, pi], nvt[bi, pi], nt[bi, pi]
            bseq = [bseq[b] + [int(p)] for b, p in zip(bi, pi)]

        return best_seq, best_cost, cost_stay

    # ---------- Drehen ----------
    def best_yaw(self, pose, heading_target):
        """
        Drehbefehl, dessen vorhergesagtes Ergebnis dem Soll-Kurs am nächsten kommt.
        None, wenn keine Drehung den Fehler verkleinert, Drehen gesperrt ist oder die Drehung
        laut Modell in der Sicherheitszone enden würde.
        Rückgabe: (exec, erwartete Drehung, erwarteter Restfehler)
        """
        x, y, th = pose
        if not self.yaw_allowed_at(x):
            return None
        err_now = abs(wrap_angle(heading_target - th))
        if self.rm is not None:
            target = wrap_angle(heading_target - th)
            if abs(target) > ROT_TARGET_MAX:
                target = math.copysign(ROT_TARGET_MAX, target)  # in zwei Schritten drehen
            c = self.rm.best_command(float(target))
            rest = abs(wrap_angle(th + c["dtheta"] - heading_target)) + max(c["sigma_theta"], ROT_SIGMA_FLOOR)
            if rest >= err_now:
                return None
            ex_, _ = end_positions(pose, c["dx_body"], c["dy_body"])
            if self.zone_x is not None and ex_ < self.zone_x:
                return None
            return {"type": "rot", "cmd": c}, c["dtheta"], rest

        yi = np.concatenate([np.where(self.p_family == "yaw_left")[0], np.where(self.p_family == "yaw_right")[0]])
        if len(yi) == 0:
            return None
        pred = np.abs(wrap_angle(th + self.p_mean[yi, 2] - heading_target)) + np.sqrt(self.p_var[yi, 2])
        if self.zone_x is not None:
            ex_, _ = end_positions(pose, self.p_mean[yi, 0], self.p_mean[yi, 1])
            pred = np.where(ex_ < self.zone_x, np.inf, pred)
        k = int(np.argmin(pred))
        if pred[k] >= err_now:
            return None
        j = yi[k]
        return self.p_exec[j], float(self.p_mean[j, 2]), float(pred[k])

    # ---------- gerade Läufe (Grobphase + Docking) ----------
    def line_lut(self, actions, max_T=MAX_STRAIGHT_DURATION, yaw_free=False):
        """
        Lookup-Tabelle für gerade Befehle mit Dauern bis max_T (darüber extrapoliert, mit Zusatzstreuung).
        yaw_free=True: Befehle mit Yaw-Anteil werden verworfen (Docking).
        Rückgabe: dict(action, T, mean, var, hi) oder None.
        """
        A, T, M, V, HI = [], [], [], [], {}
        for a in actions:
            if a not in self.model.actions:
                continue
            if yaw_free and abs(self.model.command[a][2]) > 1e-9:
                log.warning(f"'{a}' hat einen Yaw-Anteil und wird für die Docking-Bewegung NICHT verwendet.")
                continue
            lo, hi = self.model.duration_range[a]
            Ta = np.round(np.arange(lo, max(hi, max_T) + 1e-9, DURATION_STEP), 3)
            m, sd = self.model.predict(a, Ta)
            var = np.array(sd, dtype=float) ** 2
            m_hi, _ = self.model.predict(a, hi)
            extra = np.clip(np.hypot(m[:, 0], m[:, 1]) - np.hypot(*m_hi[0, :2]), 0, None)
            extra = np.where(Ta > hi, extra, 0.0)
            var[:, 0] += (EXTRAP_LAT_FRAC * extra) ** 2
            var[:, 1] += (EXTRAP_LAT_FRAC * extra) ** 2
            A += [a] * len(Ta); T.append(Ta); M.append(m); V.append(var); HI[a] = hi
        if not A:
            return None
        return {"action": np.array(A), "T": np.concatenate(T), "mean": np.vstack(M),
                "var": np.vstack(V), "hi": HI}

    @staticmethod
    def lut_exec(lut, k):
        return {"type": "dyn", "action": str(lut["action"][k]), "T": float(lut["T"][k])}

    def _build_straight_lut(self):
        acts = [a for a in self.model.actions
                if a in ("forward", "backward") or a.startswith("fwd_") or a.startswith("bwd_")]
        self.s = self.line_lut(acts)

    def _best_straight(self, pose, aim, mask=None):
        """Bester gerader Befehl von pose zum Punkt aim. Rückgabe: (index, Fehlabstand, Kosten) oder None."""
        ex, ey = end_positions(pose, self.s["mean"][:, 0], self.s["mean"][:, 1])
        miss = np.hypot(ex - aim[0], ey - aim[1])
        cost = miss ** 2 + self.s["var"][:, 0] + self.s["var"][:, 1] + self.s["T"] * TIME_PENALTY
        if mask is not None:
            cost = np.where(mask, cost, np.inf)
        if self.zone_x is not None:
            cost = np.where(ex < self.zone_x, np.inf, cost)   # Sicherheitszone
        k = int(np.argmin(cost))
        if not np.isfinite(cost[k]):
            return None
        return k, float(miss[k]), float(cost[k])

    def coarse_step(self, pose, goal):
        """
        Grobphase in höchstens ZWEI Etappen:
          1) (falls nötig) EINE gezielte Drehung, so dass danach vorwärts ODER rückwärts aufs Ziel zeigt
          2) EIN langer gerader Lauf bis STOP_SHORT vor das Ziel
        Rückgabe: (exec, Beschreibung) oder None (-> Feinanflug)
        """
        if not hasattr(self, "s"):
            self._build_straight_lut()
        if self.s is None:
            return None
        x, y, th = pose
        gx, gy, _ = goal
        dist = math.hypot(gx - x, gy - y)
        ux, uy = (gx - x) / dist, (gy - y) / dist
        aim = (gx - STOP_SHORT * ux, gy - STOP_SHORT * uy)
        tol = max(MISS_ABS, MISS_FRAC * dist)

        # 1) Geht es ohne Drehen mit einem geraden Lauf?
        best = self._best_straight(pose, aim)
        if best is not None and best[1] <= tol:
            k, miss, _ = best
            ex = self.lut_exec(self.s, k)
            ext = "  (extrapoliert)" if ex["T"] > self.s["hi"][ex["action"]] + 1e-9 else ""
            return ex, f"Gerader Lauf: {dist:.2f} m zum Ziel, erw. {miss * 100:.0f} cm neben dem Zielpunkt{ext}"

        if dist < COARSE_YAW_MIN_DIST:
            return None

        # 2) Drehen: für vorwärts UND rückwärts den nötigen Kurs bestimmen, kleinere Drehung gewinnt
        bearing = math.degrees(math.atan2(uy, ux))
        options = []
        for base in ("forward", "backward"):
            idx = np.where(self.s["action"] == base)[0]
            if len(idx) == 0:
                continue
            lens = np.hypot(self.s["mean"][idx, 0], self.s["mean"][idx, 1])
            j = idx[np.argmin(np.abs(lens - (dist - STOP_SHORT)))]
            move_dir_body = math.degrees(math.atan2(self.s["mean"][j, 1], self.s["mean"][j, 0]))
            heading_needed = wrap_angle(bearing - move_dir_body)
            err_now = abs(wrap_angle(heading_needed - th))
            yaw = self.best_yaw(pose, heading_needed)
            if yaw is not None:
                options.append((abs(yaw[1]) + 10 * yaw[2], base, heading_needed, yaw))
            else:
                options.append((10 * err_now, base, heading_needed, None))
        if not options:
            if best is None:
                return None
            k, miss, _ = best
            return self.lut_exec(self.s, k), f"Gerader Lauf (keine bessere Drehung möglich), erw. {miss * 100:.0f} cm daneben"

        _, base, heading_needed, yaw = min(options, key=lambda o: o[0])
        if yaw is None:
            fam = np.array([a == base or a.startswith("fwd_" if base == "forward" else "bwd_")
                            for a in self.s["action"]])
            b2 = self._best_straight(pose, aim, mask=fam)
            if b2 is None:
                return None
            return (self.lut_exec(self.s, b2[0]),
                    f"Gerader Lauf {base} (Kurs {wrap_angle(heading_needed - th):+.1f}° daneben, "
                    f"keine Drehung wäre genauer)")
        ex, dth, rest = yaw
        err = wrap_angle(heading_needed - th)
        return ex, f"Ausrichten für {base}: Kursfehler {err:+.1f}° -> erw. {dth:+.1f}° (Rest ≈ {rest:.1f}°)"


# ==========================================
# ROBOTER-SCHNITTSTELLEN
# ==========================================
class PoseTracker:
    """Empfängt die Posen der Ortung. Verbindet im Hintergrund und nach Abbrüchen neu,
    ein fehlender Broker blockiert also den Start nicht (es kommen dann nur keine Posen)."""

    def __init__(self, broker, topic):
        import paho.mqtt.client as mqtt
        self.topic = topic
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.lock = threading.Lock()
        self.buf = []
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.connect_async(broker, 1883, 60)
        self.client.loop_start()

    @property
    def connected(self):
        return self.client.is_connected()

    def _on_connect(self, client, *args):
        client.subscribe(self.topic)   # auch nach jedem Neuverbinden

    def _on_message(self, client, userdata, msg):
        try:
            d = json.loads(msg.payload.decode())
            if all(k in d for k in ("x", "y", "theta")):
                with self.lock:
                    self.buf.append((time.time(), d["x"], d["y"], d["theta"]))
                    if len(self.buf) > 1000:
                        self.buf.pop(0)
        except Exception:
            pass

    def latest(self):
        with self.lock:
            return self.buf[-1] if self.buf else None

    def window(self, t0, t1):
        with self.lock:
            return [(x, y, th) for (t, x, y, th) in self.buf if t0 <= t <= t1]


class RealRobot:
    """
    Echter Lite3. tracker/commander können von der Station übergeben werden (geteilte Verbindungen);
    sonst werden eigene angelegt (und beim shutdown wieder geschlossen).
    """

    def __init__(self, tracker=None, commander=None):
        self.tracker = tracker or PoseTracker(MQTT_BROKER, TOPIC_POSE)
        self._own_cmd = commander is None
        if commander is None:
            from lite3_commander import Lite3UdpCommander, Lite3UdpConfig
            commander = Lite3UdpCommander(Lite3UdpConfig(robot_ip=ROBOT_IP, verbose=False))
            commander.start()
            time.sleep(1.0)
        self.cmd = commander
        self.cmd.set_control_mode_manual()
        self.cmd.set_motion_mode("move")

    def latest_pose(self):
        latest = self.tracker.latest()
        return None if latest is None else latest[1:]

    def check_alive(self, center):
        latest = self.tracker.latest()
        if not latest:
            raise SafetyStop("Keine Pose empfangen!")
        t, x, y, _ = latest
        if time.time() - t > MAX_POSE_AGE:
            raise SafetyStop(f"Pose zu alt ({time.time() - t:.2f}s)")
        if center and math.hypot(x - center[0], y - center[1]) > MAX_DISTANCE_FROM_START:
            raise SafetyStop("Roboter außerhalb des Sicherheitsradius!")

    def measure(self, settle=0.0):
        """Pose im Stand mitteln (nach settle Sekunden Wartezeit)."""
        time.sleep(settle + START_POSE_AVERAGE_WINDOW + 0.1)
        t = time.time()
        return pose_stats(self.tracker.window(t - START_POSE_AVERAGE_WINDOW, t))

    def stop(self):
        try:
            self.cmd.stop_robot(buffer_s=0.0)
        except Exception as e:
            log.error(f"Stoppbefehl fehlgeschlagen: {e}")

    def execute(self, segments, info=None, guard=None, abort_event=None):
        """
        segments: [(cmd_xyz, dauer), ...] direkt hintereinander, Stopp erst am Ende.
        guard(pose) -> str|GuardStop|None: wird bei jedem Sendezyklus mit der Live-Pose geprüft;
                       str = Sicherheitsstopp, GuardStop = Lauf regulär beenden (dann normal messen).
        abort_event: gesetzt -> sofort stoppen, NavAborted.
        Messung im selben Fenster wie beim Training.
        """
        t_cmd = time.time()
        try:
            early = None
            for cmd_xyz, duration in segments:
                t_end = time.time() + duration
                while time.time() < t_end:
                    if abort_event is not None and abort_event.is_set():
                        raise NavAborted("Abbruch angefordert")
                    latest = self.tracker.latest()
                    if latest is None or time.time() - latest[0] > MAX_POSE_AGE:
                        raise SafetyStop("Tracker liefert während der Bewegung keine aktuelle Pose!")
                    if guard is not None:
                        reason = guard(latest[1:])
                        if isinstance(reason, GuardStop):
                            early = reason
                            break
                        if reason:
                            raise SafetyStop(reason)
                    self.cmd.send_axes(*cmd_xyz)
                    time.sleep(SEND_PERIOD)
                if early:
                    log.info(f"    Lauf vorzeitig beendet: {early}")
                    break
        except BaseException:
            self.stop()
            raise
        t_stop = time.time()
        self.stop()
        _sleep(AVERAGING_END + 0.1, abort_event)
        stats = pose_stats(self.tracker.window(t_stop + AVERAGING_START, t_stop + AVERAGING_END))
        if not stats:
            raise SafetyStop("Keine Posen im Messfenster! Tracker ausgefallen?")
        return stats, t_cmd, t_stop

    def shutdown(self):
        try:
            self.cmd.stop_robot(buffer_s=0.5)
            if self._own_cmd:
                self.cmd.shutdown()
        except Exception:
            pass


class SimRobot:
    """Simuliert den Roboter mit den Modellen selbst (Mittelwert + gesampeltes Rauschen)."""

    def __init__(self, model, rot_model=None, start=(2.0, -0.25, 0.0), seed=None):
        self.model = model
        self.rm = rot_model
        self.pose = list(start)
        self.rng = np.random.default_rng(seed)

    def latest_pose(self):
        return tuple(self.pose)

    def check_alive(self, center):
        pass

    def _measure(self):
        nx, ny, nt = SIM_MEAS_NOISE
        x, y, th = self.pose
        return {"mean_x": x + self.rng.normal(0, nx), "mean_y": y + self.rng.normal(0, ny),
                "mean_theta": wrap_angle(th + self.rng.normal(0, nt)),
                "std_x": nx, "std_y": ny, "std_theta": nt, "count": 8}

    def measure(self, settle=0.0):
        return self._measure()

    def stop(self):
        pass

    def execute(self, segments, info, guard=None, abort_event=None):
        if abort_event is not None and abort_event.is_set():
            raise NavAborted("Abbruch angefordert")
        if info["type"] == "dyn":
            mean, std = self.model.predict(info["action"], info["T"])
        else:
            c = info["cmd"]
            mean, std = self.rm.predict(c["kind"], c["direction"], c["t1"], c["t2"])
            std = np.maximum(std, [ROT_POS_SIGMA_FLOOR, ROT_POS_SIGMA_FLOOR, ROT_SIGMA_FLOOR])
        d = mean[0] + self.rng.normal(0, 1, 3) * std[0] * SIM_NOISE_SCALE
        x, y, th = self.pose
        t = math.radians(th)
        self.pose = [x + math.cos(t) * d[0] - math.sin(t) * d[1],
                     y + math.sin(t) * d[0] + math.cos(t) * d[1],
                     wrap_angle(th + d[2])]
        if guard is not None:  # Simulation: nur Endpose prüfbar
            reason = guard(tuple(self.pose))
            if reason and not isinstance(reason, GuardStop):
                raise SafetyStop(reason)
        now = time.time()
        return self._measure(), now, now + sum(T for _, T in segments)

    def shutdown(self):
        pass


# ==========================================
# NAVIGATION
# ==========================================
LOG_FIELDS = [
    "sample_id", "start_timestamp", "command_start_timestamp", "stop_timestamp", "end_timestamp",
    "start_x", "start_y", "start_theta", "end_x", "end_y", "end_theta",
    "delta_x", "delta_y", "delta_theta", "command_x", "command_y", "command_yaw",
    "command_name", "command_duration", "mean_x", "mean_y", "mean_theta",
    "std_x", "std_y", "std_theta", "num_pose_samples",
]
ROT_LOG_FIELDS = [
    "sample_id", "kind", "direction", "t1", "t2", "gap", "cmd_yaw_1", "cmd_yaw_2",
    "command_start_timestamp", "stop_timestamp",
    "start_x", "start_y", "start_theta", "end_x", "end_y", "end_theta",
    "delta_x", "delta_y", "delta_theta", "dx_body", "dy_body",
    "std_x", "std_y", "std_theta", "num_pose_samples",
]


class Navigator:
    """
    Blockierende Navigationslogik. Wirft NavRejected / NavAborted / SafetyStop;
    für die State Machine besser NavigationController verwenden (fängt das ab, liefert NavResult).
    """

    def __init__(self, robot, model, rot_model=None, log_path=LOG_PATH, rot_log_path=ROT_LOG_PATH,
                 mode="schema", abort_event=None, on_event=None):
        self.mode = mode  # "schema" (drehen -> geradeaus -> Feinanflug) oder "mpc" (nur Beam-Search)
        self.robot = robot
        self.model = model
        self.rm = rot_model
        self.planner = PrimitivePlanner(model, rot_model)
        self.log_path = log_path
        self.rot_log_path = rot_log_path
        self.abort_event = abort_event or threading.Event()
        self.on_event = on_event
        self.sample_id = 0
        self.stats = None
        self.center = None
        self.last_message = ""
        self.docked = False
        self.lane = (DOCK_POSE, DOCK_ANGLE)   # aktuelle Einfahrbahn in die Zone: Dock oder Abgabepunkt
        self._rotation_locked = False

    # ---------- Drehsperre (Dock) ----------
    @property
    def rotation_locked(self):
        return self._rotation_locked

    @rotation_locked.setter
    def rotation_locked(self, value):
        if value != self._rotation_locked:
            log.info("🔒 Drehen GESPERRT (Dock/Andockbahn)." if value else "🔓 Drehen wieder erlaubt.")
        self._rotation_locked = value
        self.planner.allow_yaw = not value

    def _emit(self, event, **data):
        if self.on_event is not None:
            try:
                self.on_event(event, data)
            except Exception as e:
                log.warning(f"on_event-Callback fehlgeschlagen: {e}")

    def init_pose(self):
        self.stats = self.robot.measure()
        if not self.stats:
            raise SafetyStop("Keine initiale Pose empfangen!")
        self.center = (self.stats["mean_x"], self.stats["mean_y"])
        return self.stats

    def resync_pose(self, settle=AVERAGING_START):
        """Pose nach Abbruch/Stopp neu messen (der Roboter steht)."""
        stats = self.robot.measure(settle)
        if stats:
            self.stats = stats
        return self.stats

    def pose(self):
        s = self.stats
        return s["mean_x"], s["mean_y"], s["mean_theta"]

    def in_zone(self, pose=None):
        x = (pose or self.pose())[0]
        return x < SAFETY_ZONE_X - ZONE_POSE_TOL

    # ---------- Ausführen + Loggen ----------
    def _segments(self, ex):
        if ex["type"] == "dyn":
            return [(self.model.command[ex["action"]], ex["T"])]
        return self.rm.segments(ex["cmd"])

    def _zone_guard(self, p):
        zx = self.planner.zone_x
        if zx is not None and p[0] < zx - LIVE_ZONE_MARGIN:
            return f"Sicherheitszone verletzt: x={p[0]:.3f} m < {zx - LIVE_ZONE_MARGIN:.2f} m"
        return None

    def run(self, ex, guard=None):
        """Einen Befehl ausführen. Letzte Verteidigungslinie: Drehbefehle im Dock/in der Zone -> SafetyStop."""
        if self.abort_event.is_set():
            raise NavAborted("Abbruch angefordert")
        self.robot.check_alive(self.center)
        segs = self._segments(ex)
        has_yaw = ex["type"] == "rot" or any(abs(cmd[2]) > 1e-9 for cmd, _ in segs)
        if has_yaw and (self.rotation_locked or self.in_zone()):
            raise SafetyStop(f"Drehbefehl '{describe(ex)}' im Dock/in der Sicherheitszone verweigert!")
        start = self.stats
        self._emit("command", command=describe(ex), pose=self.pose())
        end, t_cmd, t_stop = self.robot.execute(segs, ex, guard=guard or self._zone_guard,
                                                abort_event=self.abort_event)
        self._log(ex, segs, start, end, t_cmd, t_stop)
        self.stats = end

    @staticmethod
    def _append(path, fields, row):
        new = not os.path.isfile(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)

    def _log(self, ex, segs, start, end, t_cmd, t_stop):
        self.sample_id += 1
        dx, dy = end["mean_x"] - start["mean_x"], end["mean_y"] - start["mean_y"]
        dth = wrap_angle(end["mean_theta"] - start["mean_theta"])
        common = {
            "sample_id": self.sample_id, "command_start_timestamp": t_cmd, "stop_timestamp": t_stop,
            "start_x": start["mean_x"], "start_y": start["mean_y"], "start_theta": start["mean_theta"],
            "end_x": end["mean_x"], "end_y": end["mean_y"], "end_theta": end["mean_theta"],
            "delta_x": dx, "delta_y": dy, "delta_theta": dth,
            "std_x": end["std_x"], "std_y": end["std_y"], "std_theta": end["std_theta"],
            "num_pose_samples": end["count"],
        }
        if ex["type"] == "dyn":
            cmd = segs[0][0]
            self._append(self.log_path, LOG_FIELDS, {
                **common, "start_timestamp": t_cmd, "end_timestamp": time.time(),
                "command_x": cmd[0], "command_y": cmd[1], "command_yaw": cmd[2],
                "command_name": ex["action"], "command_duration": ex["T"],
                "mean_x": end["mean_x"], "mean_y": end["mean_y"], "mean_theta": end["mean_theta"],
            })
        else:
            c = ex["cmd"]
            dxb, dyb = world_to_body(dx, dy, start["mean_theta"])
            yaw_segs = [s for s in segs if s[0][2] != 0]
            gap = sum(T for cmd, T in segs if tuple(cmd) == (0, 0, 0))
            self._append(self.rot_log_path, ROT_LOG_FIELDS, {
                **common, "kind": c["kind"], "direction": c["direction"], "t1": c["t1"],
                "t2": c["t2"], "gap": gap,
                "cmd_yaw_1": yaw_segs[0][0][2], "cmd_yaw_2": yaw_segs[-1][0][2] if len(yaw_segs) > 1 else 0,
                "dx_body": float(dxb), "dy_body": float(dyb),
            })

    # ---------- Planung ----------
    @staticmethod
    def _errors(pose, goal):
        x, y, th = pose
        pos_err = math.hypot(goal[0] - x, goal[1] - y)
        th_err = abs(wrap_angle(th - goal[2])) if goal[2] is not None else 0.0
        return pos_err, th_err

    def _mpc_step(self, pose, goal, allowed, label):
        t0 = time.time()
        plan, cost_plan, cost_stay = self.planner.plan(pose, goal, self.center, allowed)
        dt = (time.time() - t0) * 1000
        if not plan or cost_plan > cost_stay - MIN_IMPROVEMENT:
            return None
        plan_txt = " → ".join(describe(self.planner.p_exec[p]) for p in plan)
        log.info(f"    [{label}] Plan ({dt:.0f} ms): {plan_txt}   "
                 f"[erw. Restfehler ≈ {math.sqrt(max(cost_plan, 0)) * 100:.1f} cm-äquiv.]")
        return self.planner.p_exec[plan[0]]

    def go_to(self, goal, done_fn=None, title=None):
        """
        Fährt zu goal = (x, y, theta|None) (absolute Kamerakoordinaten).
        done_fn(pose) -> bool: optional eigenes Erfolgskriterium.
        Rückgabe: True (erreicht) / False (nicht erreicht, Grund in last_message).
        """
        gx, gy, gth = goal
        zx = self.planner.zone_x
        if self.rotation_locked:
            raise NavRejected("Roboter steht im Dock / in der Andockbahn – zuerst undock().")
        if zx is not None and gx < zx:
            raise NavRejected(f"Ziel x={gx:.3f} m liegt in der Sicherheitszone (x < {zx:.2f} m).")
        if math.hypot(gx - self.center[0], gy - self.center[1]) > MAX_DISTANCE_FROM_START:
            raise NavRejected(f"Ziel liegt außerhalb des Sicherheitsradius ({MAX_DISTANCE_FROM_START} m).")

        th_txt = f", θ={gth:.1f}°" if gth is not None else ""
        log.info(f"\n=== {title or 'Ziel'}: x={gx:.3f}, y={gy:.3f}{th_txt} ===")
        for step in range(1, MAX_STEPS_PER_GOAL + 1):
            self.robot.check_alive(self.center)
            pose = self.pose()
            if zx is not None and pose[0] < zx - ZONE_POSE_TOL:
                msg = f"Roboter steht in der Sicherheitszone (x={pose[0]:.3f} m)"
                if step == 1:
                    raise NavRejected(msg + " – zuerst undock().")
                raise SafetyStop(msg + "!")
            pos_err, th_err = self._errors(pose, goal)
            log.info(f"\n[{step}] Pose x={pose[0]:.3f} y={pose[1]:.3f} θ={pose[2]:.1f}°  "
                     f"| Fehler {pos_err * 100:.1f} cm" + (f", {th_err:.1f}°" if gth is not None else ""))
            self._emit("step", step=step, pose=pose, pos_err=pos_err, th_err=th_err)

            done = done_fn(pose) if done_fn else (pos_err <= POS_TOLERANCE and th_err <= THETA_TOLERANCE)
            if done:
                log.info("✓ Ziel erreicht.")
                self.last_message = "Ziel erreicht."
                return True

            ex = None
            if self.mode == "schema" and pos_err > APPROACH_DIST:
                coarse = self.planner.coarse_step(pose, goal)
                if coarse is not None:
                    ex, why = coarse
                    log.info(f"    [Grob] {why}  -> {describe(ex)}")

            if ex is None:
                if self.mode == "schema":
                    if pos_err > POS_TOLERANCE:
                        ex = self._mpc_step(pose, goal, FINE_ACTIONS, "Fein")
                    if ex is None and gth is not None and th_err > THETA_TOLERANCE:
                        yaw = self.planner.best_yaw(pose, gth)
                        if yaw is not None:
                            ex = yaw[0]
                            log.info(f"    [Ausrichten] Winkelfehler {wrap_angle(gth - pose[2]):+.1f}° "
                                     f"-> {describe(ex)}")
                    if ex is None and gth is not None:
                        ex = self._mpc_step(pose, goal, FINE_ACTIONS, "Nachkorrektur")
                else:
                    ex = self._mpc_step(pose, goal, None, "MPC")

            if ex is None:
                self.last_message = (f"Kein Schritt verbessert die Pose mehr — bestmöglich erreicht: "
                                     f"{pos_err * 100:.1f} cm" + (f", {th_err:.1f}°" if gth is not None else ""))
                log.info(self.last_message)
                return False

            self.run(ex)

        self.last_message = f"Maximale Schrittzahl ({MAX_STEPS_PER_GOAL}) erreicht."
        log.info(self.last_message)
        return False

    # ---------- Anfahren aus fester Richtung ----------
    @staticmethod
    def _approach_errors(pose, target, angle):
        """along < 0 = noch davor (kurz), > 0 = darüber hinaus; lat = seitlich; dth = Winkelfehler."""
        along, lat = line_frame(pose[0], pose[1], target, angle)
        return along, lat, wrap_angle(pose[2] - angle)

    def approach(self, target, angle=APPROACH_ANGLE_DEFAULT, standoff=APPROACH_STANDOFF):
        """
        Fährt den Punkt target=(x, y) präzise aus Richtung `angle` an:
          1. Vorposition `standoff` vor dem Punkt anfahren und genau ausrichten   (approach_pre)
          2. letzte Bewegung nur vorwärts (lieber zu kurz als zu weit)            (approach_final)
        """
        if not self.approach_pre(target, angle, standoff):
            return False
        return self.approach_final(target, angle)

    def approach_pre(self, target, angle=APPROACH_ANGLE_DEFAULT, standoff=APPROACH_STANDOFF):
        """Teil 1 von approach(): Vorposition `standoff` vor target anfahren und genau ausrichten."""
        tx, ty = target
        rad = math.radians(angle)
        pre = (tx - standoff * math.cos(rad), ty - standoff * math.sin(rad), angle)
        zx = self.planner.zone_x
        if zx is not None and (tx < zx or pre[0] < zx):
            raise NavRejected(f"Anfahrpunkt/Vorposition liegt in der Sicherheitszone (x < {zx:.2f} m).")
        lo, hi = APPROACH_STANDOFF_BAND
        log.info(f"\n##### Anfahren: Punkt x={tx:.3f} y={ty:.3f}, Richtung {angle:.1f}°, "
                 f"Vorposition {standoff * 100:.0f} cm davor #####")

        def pre_ok(pose, lat_tol=APPROACH_LAT_TOL, th_tol=APPROACH_THETA_TOL):
            along, lat, dth = self._approach_errors(pose, (tx, ty), angle)
            return (-hi <= along <= -lo) and abs(lat) <= lat_tol and abs(dth) <= th_tol

        # --- 1. Vorposition + Ausrichtung ---
        self.planner.theta_weight = APPROACH_THETA_WEIGHT
        try:
            reached = self.go_to(pre, done_fn=pre_ok, title="Vorposition")
        finally:
            self.planner.theta_weight = THETA_WEIGHT

        along, lat, dth = self._approach_errors(self.pose(), (tx, ty), angle)
        log.info(f"\nVorposition: Abstand {-along * 100:.1f} cm, seitlich {lat * 100:+.1f} cm, Winkel {dth:+.1f}°")
        if not reached:
            if pre_ok(self.pose(), *APPROACH_RELAXED):
                log.warning(f"⚠ Strenge Toleranzen nicht erreicht, aber innerhalb der gelockerten "
                            f"({APPROACH_RELAXED[0] * 100:.0f} cm / {APPROACH_RELAXED[1]:.0f}°) -> fahre trotzdem an.")
            else:
                self.last_message = "Vorposition nicht genau genug erreicht -> Anfahren abgebrochen."
                log.info("✗ " + self.last_message)
                return False
        self.last_message = "Vorposition erreicht."
        return True

    def approach_final(self, target, angle=APPROACH_ANGLE_DEFAULT):
        """Teil 2 von approach(): von der Vorposition aus nur vorwärts bis target."""
        tx, ty = target
        zx = self.planner.zone_x
        if zx is not None and tx < zx:
            raise NavRejected(f"Anfahrpunkt liegt in der Sicherheitszone (x < {zx:.2f} m).")

        # --- 2. Letzte Bewegung: nur vorwärts ---
        pl = self.planner
        cand = np.where(np.isin(pl.p_family, APPROACH_FINAL_ACTIONS))[0]
        if len(cand) == 0:
            self.last_message = "Keine Vorwärts-Primitive im Modell -> Anfahren abgebrochen."
            log.info("✗ " + self.last_message)
            return False

        for k in range(1, APPROACH_MAX_FINAL_STEPS + 1):
            pose = self.pose()
            along, lat, dth = self._approach_errors(pose, (tx, ty), angle)
            if -APPROACH_FINAL_TOL <= along <= APPROACH_FINAL_TOL and abs(lat) <= APPROACH_FINAL_TOL:
                break
            if along > APPROACH_FINAL_TOL:
                log.warning(f"⚠ Bereits {along * 100:.1f} cm über den Punkt hinaus – kein weiterer Vorwärtsschritt.")
                break
            ex, ey = end_positions(pose, pl.p_mean[cand, 0], pl.p_mean[cand, 1])
            al, la = line_frame(ex, ey, (tx, ty), angle)
            cost = weighted_line_cost(al, la, pl.p_var[cand, 0] + pl.p_var[cand, 1])
            if zx is not None:
                cost = np.where(ex < zx, np.inf, cost)
            j = int(np.argmin(cost))
            if not np.isfinite(cost[j]) or cost[j] >= weighted_line_cost(np.array(along), np.array(lat)) - MIN_IMPROVEMENT:
                log.info(f"Kein Vorwärtsschritt verbessert die Lage (Rest {-along * 100:.1f} cm) – fertig.")
                break
            ex_cmd = pl.p_exec[cand[j]]
            log.info(f"\n[Endanflug {k}] Rest {-along * 100:.1f} cm, seitlich {lat * 100:+.1f} cm "
                     f"-> {describe(ex_cmd)} (erw. Rest {-al[j] * 100:+.1f} cm, seitlich {la[j] * 100:+.1f} cm)")
            self.run(ex_cmd)

        along, lat, dth = self._approach_errors(self.pose(), (tx, ty), angle)
        ok = abs(along) <= APPROACH_ACCEPT[0] and abs(lat) <= APPROACH_ACCEPT[1] and abs(dth) <= APPROACH_ACCEPT[2]
        self.last_message = (f"Anfahren beendet: Längsfehler {along * 100:+.1f} cm "
                             f"({'zu weit' if along > 0 else 'zu kurz'}), seitlich {lat * 100:+.1f} cm, Winkel {dth:+.1f}°")
        log.info(f"\n{'✓' if ok else '⚠'} " + self.last_message)
        return ok

    # ---------- Docking ----------
    def _dock_errors(self, pose):
        return self._approach_errors(pose, *self.lane)

    def _check_dock_pose(self, pose):
        """Gemessene Pose (im Stand) in der Andockbahn prüfen -> SafetyStop bei Verletzung."""
        along, lat, dth = self._dock_errors(pose)
        if abs(dth) > DOCK_THETA_ABORT:
            raise SafetyStop(f"Winkelabweichung in der Andockbahn {dth:+.1f}° (> {DOCK_THETA_ABORT}°) – "
                             f"Drehen ist hier verboten, bitte manuell eingreifen.")
        if abs(lat) > DOCK_LAT_ABORT:
            raise SafetyStop(f"Seitliche Abweichung von der Andocklinie {lat * 100:+.1f} cm "
                             f"(> {DOCK_LAT_ABORT * 100:.0f} cm) – bitte manuell eingreifen.")
        return along, lat, dth

    def _dock_guard(self, p):
        along, lat, dth = self._dock_errors(p)
        if abs(dth) > DOCK_THETA_LIVE_ABORT:
            return f"Dock: Winkelabweichung {dth:+.1f}° während der Bewegung"
        if abs(lat) > DOCK_LAT_ABORT + LIVE_ZONE_MARGIN:
            return f"Dock: seitliche Abweichung {lat * 100:+.1f} cm während der Bewegung"
        if DOCK_LIVE_OVERSHOOT is not None and along > DOCK_LIVE_OVERSHOOT:
            return GuardStop(f"{along * 100:.1f} cm über die Dockmitte")   # regulär beenden, kein Fehler
        return None

    def run_speed(self, action, speed, deadtime):
        """
        (Geschwindigkeit m/s, Totzeit s) für einen geraden Lauf. Nicht gesetzte Werte werden aus dem
        Dynamikmodell geschätzt: lineare Regression Strecke(T) über den Trainingsbereich.
        """
        if speed is not None and deadtime is not None:
            return float(speed), float(deadtime)
        lo, hi = self.model.duration_range[action]
        T = np.linspace(lo, hi, 20)
        m, _ = self.model.predict(action, T)
        d = np.hypot(m[:, 0], m[:, 1])
        slope, icpt = np.polyfit(T, d, 1)
        if slope <= 1e-3:
            raise SafetyStop(f"Dynamikmodell liefert für '{action}' keine sinnvolle Geschwindigkeit.")
        v = float(speed) if speed is not None else float(slope)
        t0 = float(deadtime) if deadtime is not None else max(0.0, float(-icpt / slope))
        return v, t0

    def _straight_run(self, action, distance, speed, deadtime, label, guard):
        """EIN gerader Lauf über `distance` m: T = Totzeit + distance / v. Nur yaw-freie Befehle."""
        if action not in self.model.command:
            raise SafetyStop(f"Befehl '{action}' nicht im Modell.")
        if abs(self.model.command[action][2]) > 1e-9:
            raise SafetyStop(f"'{action}' hat einen Yaw-Anteil – im Dock verboten!")
        v, t0 = self.run_speed(action, speed, deadtime)
        T = t0 + max(distance, 0.0) / v
        if T > DOCK_RUN_T_MAX:
            raise SafetyStop(f"{label}: berechnete Laufzeit {T:.2f}s > {DOCK_RUN_T_MAX}s – Strecke {distance:.2f} m "
                             f"unplausibel, bitte prüfen.")
        ex = {"type": "dyn", "action": action, "T": round(T, 2)}
        log.info(f"\n[{label}] {distance * 100:.1f} cm geradeaus -> {describe(ex)} "
                 f"(v={v:.3f} m/s, Totzeit {t0:.2f}s)")
        self.run(ex, guard=guard)

    def _reject_if_in_zone(self):
        if self.rotation_locked or self.in_zone():
            raise NavRejected("Roboter ist bereits in der Sicherheitszone – zuerst undock().")

    def _zone_pre(self, pre_pose, title):
        """Vorposition vor einer Einfahrbahn anfahren und ausrichten (Planer darf minimal über die Zonengrenze)."""
        px, py, pth = pre_pose

        def pre_ok(pose, along_tol=DOCK_PRE_ALONG_TOL, lat_tol=DOCK_PRE_LAT_TOL, th_tol=DOCK_PRE_THETA_TOL):
            along, lat, dth = self._approach_errors(pose, (px, py), pth)
            return abs(along) <= along_tol and abs(lat) <= lat_tol and abs(dth) <= th_tol

        # Ziel etwas außerhalb der Zonengrenze: dort kann der Planer seitlich korrigieren, ohne dass
        # jeder Schritt in die Zone führen könnte. Erfolg zählt trotzdem nach pre_ok (Toleranzen um pre_pose).
        rad = math.radians(pth)
        aim = (px - DOCK_PRE_AIM_OFFSET * math.cos(rad), py - DOCK_PRE_AIM_OFFSET * math.sin(rad), pth)

        reached = False
        for attempt in (1, 2):
            self.planner.theta_weight = APPROACH_THETA_WEIGHT
            self.planner.zone_x = SAFETY_ZONE_X - DOCK_PRE_ZONE_ALLOWANCE
            try:
                reached = self.go_to(aim, done_fn=pre_ok, title=title if attempt == 1 else f"{title} (2. Versuch)")
            finally:
                self.planner.theta_weight = THETA_WEIGHT
                self.planner.zone_x = SAFETY_ZONE_X
            if reached or attempt == 2 or not self._retry_backoff(pth):
                break

        along, lat, dth = self._approach_errors(self.pose(), (px, py), pth)
        log.info(f"\n{title}: längs {along * 100:+.1f} cm, seitlich {lat * 100:+.1f} cm, Winkel {dth:+.1f}°")
        if not reached:
            if DOCK_ALLOW_RELAXED and pre_ok(self.pose(), APPROACH_RELAXED[0], APPROACH_RELAXED[0], APPROACH_RELAXED[1]):
                log.warning("⚠ Strenge Toleranzen verfehlt, gelockerte erfüllt -> fahre trotzdem ein.")
            else:
                self.last_message = (f"{title} nicht genau genug erreicht (längs {along * 100:+.1f} cm, "
                                     f"seitlich {lat * 100:+.1f} cm, {dth:+.1f}°) – Einfahren abgebrochen.")
                log.info("✗ " + self.last_message)
                return False
        self.last_message = f"{title} erreicht."
        return True

    def _retry_backoff(self, lane_angle):
        """Vor dem 2. Versuch gerade rückwärts aus dem Grenzbereich (nur wenn er grob in Einfahrrichtung schaut)."""
        x, y, th = self.pose()
        if abs(wrap_angle(th - lane_angle)) > 30.0:
            return True   # steht quer: einfach neu planen, der Planer darf hier noch drehen
        log.info(f"\nVorposition verfehlt -> {DOCK_PRE_RETRY_BACKOFF * 100:.0f} cm gerade zurück und neu versuchen.")
        self._straight_run(UNDOCK_RUN_ACTION, DOCK_PRE_RETRY_BACKOFF, UNDOCK_SPEED, UNDOCK_DEADTIME,
                           "Zurück für 2. Versuch", self._zone_guard)
        return True

    def _zone_enter(self, label):
        """EIN gerader Vorwärtslauf entlang self.lane bis zum Zielpunkt, KEIN Drehen. Rückgabe: Restfehler."""
        along, lat, dth = self._check_dock_pose(self.pose())
        self.rotation_locked = True       # bleibt gesetzt bis undock() die Zone verlassen hat
        self._straight_run(DOCK_RUN_ACTION, -along + DOCK_DISTANCE_OFFSET, DOCK_SPEED, DOCK_DEADTIME,
                           label, self._dock_guard)
        return self._check_dock_pose(self.pose())

    def dock_pre(self):
        """Nur die Dock-Vorposition anfahren und ausrichten, z. B. um dort auf die Übergabe zu warten."""
        self._reject_if_in_zone()
        self.lane = (DOCK_POSE, DOCK_ANGLE)
        return self._zone_pre(DOCK_PRE_POSE, "Dock-Vorposition")

    def dock(self):
        """
        1. Vorposition DOCK_PRE_POSE anfahren und ausrichten (wie beim Anfahren)
        2. Drehsperre an; EIN gerader Vorwärtslauf über die gemessene Entfernung bis zur Dockmitte.
        """
        if self.docked:
            self.last_message = "Bereits gedockt."
            return True
        self._reject_if_in_zone()

        px, py, pth = DOCK_PRE_POSE
        log.info(f"\n##### Docken: Vorposition x={px:.2f} y={py:.2f} θ={pth:.0f}° -> "
                 f"Dock x={DOCK_POSE[0]:.2f} y={DOCK_POSE[1]:.2f} #####")
        self.lane = (DOCK_POSE, DOCK_ANGLE)
        if not self._zone_pre(DOCK_PRE_POSE, "Dock-Vorposition"):
            return False

        along, lat, dth = self._zone_enter("Einfahren")
        self.docked = along >= -DOCK_ACCEPT[0] and abs(lat) <= DOCK_ACCEPT[1]
        self.last_message = (f"Docken {'erfolgreich' if self.docked else 'NICHT genau genug'}: "
                             f"längs {along * 100:+.1f} cm ({'zu weit' if along > 0 else 'zu kurz'}), "
                             f"seitlich {lat * 100:+.1f} cm, Winkel {dth:+.1f}°")
        log.info(f"\n{'✓' if self.docked else '⚠'} " + self.last_message)
        return self.docked

    def enter_zone(self, target):
        """
        Wie dock(), aber zu einem beliebigen Punkt target=(x, y) in der Sicherheitszone (z. B. Abgabe).
        Einfahrt wie beim Docken mit Blick DOCK_ANGLE: Vorposition auf der Zonengrenze, dann EIN gerader
        Vorwärtslauf ohne Drehen. Wieder hinaus geht es mit undock().
        """
        tx, ty = target
        if tx >= SAFETY_ZONE_X:
            raise NavRejected(f"Punkt x={tx:.3f} m liegt nicht in der Sicherheitszone – dafür approach() verwenden.")
        self._reject_if_in_zone()

        pre_pose = (SAFETY_ZONE_X, ty, DOCK_ANGLE)
        log.info(f"\n##### In die Zone: Vorposition x={pre_pose[0]:.2f} y={ty:.2f} θ={DOCK_ANGLE:.0f}° -> "
                 f"Punkt x={tx:.2f} y={ty:.2f} #####")
        self.lane = ((tx, ty), DOCK_ANGLE)
        if not self._zone_pre(pre_pose, "Zonen-Vorposition"):
            return False

        along, lat, dth = self._zone_enter("Einfahren")
        ok = along >= -DOCK_ACCEPT[0] and abs(lat) <= DOCK_ACCEPT[1]
        self.last_message = (f"Punkt in der Zone {'erreicht' if ok else 'NICHT genau genug erreicht'}: "
                             f"längs {along * 100:+.1f} cm, seitlich {lat * 100:+.1f} cm, Winkel {dth:+.1f}°")
        log.info(f"\n{'✓' if ok else '⚠'} " + self.last_message)
        return ok

    def back_off(self, distance):
        """EIN gerader Rückwärtslauf um `distance` m (z. B. nach dem Ablegen). Nur außerhalb der Zone."""
        self._reject_if_in_zone()
        x, y, th = self.pose()
        end_x = x - math.cos(math.radians(th)) * distance
        if end_x < SAFETY_ZONE_X:
            raise NavRejected(f"Rückwärtslauf würde in der Sicherheitszone enden (x={end_x:.2f} m).")
        self._straight_run(UNDOCK_RUN_ACTION, distance, UNDOCK_SPEED, UNDOCK_DEADTIME,
                           "Zurücksetzen", self._zone_guard)
        x, y, th = self.pose()
        self.last_message = f"Zurückgesetzt: x={x:.3f} m, y={y:.3f} m, θ={th:.1f}°"
        log.info("✓ " + self.last_message)
        return True

    def undock(self):
        """EIN gerader Rückwärtslauf bis x >= UNDOCK_EXIT_X. Erst danach wird Drehen wieder freigegeben."""
        pose = self.pose()
        if not self.rotation_locked and not self.in_zone(pose):
            self.docked = False
            self.last_message = "Nicht in der Sicherheitszone – nichts zu tun."
            return True

        self.rotation_locked = True
        self.docked = False
        target_x = UNDOCK_EXIT_X + UNDOCK_TARGET_OVERRUN
        log.info(f"\n##### Ausdocken: rückwärts bis x ≥ {UNDOCK_EXIT_X:.2f} m #####")

        for k in range(1, UNDOCK_MAX_STEPS + 1):
            x, y, th = self.pose()
            if x >= UNDOCK_EXIT_X:
                break
            lane_y, lane_angle = self.lane[0][1], self.lane[1]
            in_lane = abs(y - lane_y) <= DOCK_LAT_ABORT
            if in_lane:
                self._check_dock_pose((x, y, th))
            # rückwärts = Blickrichtung + 180°; Anteil in +x muss deutlich positiv sein
            back_dx = -math.cos(math.radians(th))
            if back_dx < 0.5:
                raise SafetyStop(f"Rückwärts führt nicht aus der Zone (θ={th:.1f}°) – bitte manuell eingreifen.")
            dist = (target_x - x) / back_dx

            def guard(p, x_start=x, lane=in_lane, lane_angle=lane_angle):
                if p[0] < x_start - UNDOCK_LIVE_BACKSLIDE:
                    return f"Ausdocken: läuft tiefer in die Zone (x={p[0]:.3f} m)"
                if lane and p[0] < SAFETY_ZONE_X and abs(wrap_angle(p[2] - lane_angle)) > DOCK_THETA_LIVE_ABORT:
                    return f"Ausdocken: Winkelabweichung {wrap_angle(p[2] - lane_angle):+.1f}° während der Bewegung"
                return None

            self._straight_run(UNDOCK_RUN_ACTION, dist, UNDOCK_SPEED, UNDOCK_DEADTIME,
                               f"Ausfahren{'' if k == 1 else f' {k}'}", guard)

        x, y, th = self.pose()
        if x >= UNDOCK_EXIT_X:
            self.rotation_locked = False
            self.last_message = f"Ausgedockt: x={x:.3f} m, y={y:.3f} m, θ={th:.1f}°"
            log.info("✓ " + self.last_message)
            return True
        self.last_message = f"Ausdocken nach {UNDOCK_MAX_STEPS} Läufen nicht abgeschlossen (x={x:.3f} m)."
        log.info("✗ " + self.last_message)
        return False


# ==========================================
# CONTROLLER (Schnittstelle für die State Machine)
# ==========================================
class NavigationController:
    """
    Thread-sichere Fassade um Navigator:
      - immer nur EIN Befehl gleichzeitig (sonst REJECTED)
      - alle Befehle liefern NavResult statt Ausnahmen
      - abort()/emergency_stop() aus beliebigem Thread
      - on_event(event, data): optionaler Callback für Statusmeldungen
        events: "step", "command", "result"
    """

    FREE, IN_ZONE, DOCKED = "FREE", "IN_ZONE", "DOCKED"

    def __init__(self, robot, model, rot_model=None, log_path=LOG_PATH, rot_log_path=ROT_LOG_PATH,
                 mode="schema", on_event=None):
        self.robot = robot
        self.abort_event = threading.Event()
        self.on_event = on_event
        self.nav = Navigator(robot, model, rot_model, log_path=log_path, rot_log_path=rot_log_path,
                             mode=mode, abort_event=self.abort_event, on_event=on_event)
        self._lock = threading.Lock()
        self._current = ""
        self.initialized = False

    @classmethod
    def create(cls, sim=False, model_path=MODEL_PATH, rot_model_path=ROT_MODEL_PATH, no_rot=False,
               mode="schema", on_event=None, sim_start=(2.0, -0.25, 0.0), seed=None,
               tracker=None, commander=None):
        """Modelle laden und Roboter (echt oder simuliert) anlegen."""
        model = joblib.load(model_path)
        rm = load_rotation_model(rot_model_path, no_rot)
        if sim:
            robot = SimRobot(model, rm, start=sim_start, seed=seed)
            logs = (str(paths.NAV_LOGS / "nav_log_sim.csv"), str(paths.NAV_LOGS / "nav_rotation_log_sim.csv"))
        else:
            robot = RealRobot(tracker=tracker, commander=commander)
            logs = (LOG_PATH, ROT_LOG_PATH)
        return cls(robot, model, rm, log_path=logs[0], rot_log_path=logs[1], mode=mode, on_event=on_event)

    # ---------- Zustand ----------
    @property
    def busy(self):
        return self._lock.locked()

    @property
    def current_command(self):
        return self._current

    @property
    def pose(self):
        return self.nav.pose() if self.nav.stats else None

    @property
    def location(self):
        """FREE (außerhalb der Zone), IN_ZONE (Zone/Andockbahn, nur undock), DOCKED."""
        if self.nav.docked:
            return self.DOCKED
        if self.nav.rotation_locked or (self.nav.stats and self.nav.in_zone()):
            return self.IN_ZONE
        return self.FREE

    def status(self):
        return {"initialized": self.initialized, "busy": self.busy, "command": self._current,
                "location": self.location, "pose": self.pose, "rotation_locked": self.nav.rotation_locked}

    # ---------- Befehle ----------
    def initialize(self):
        """Startpose messen. Steht der Roboter in der Zone, ist danach nur undock() erlaubt."""
        def _init():
            self.nav.init_pose()
            self.initialized = True
            pose = self.nav.pose()
            if self.nav.in_zone(pose):
                self.nav.rotation_locked = True
                along, lat, _ = self.nav._dock_errors(pose)
                self.nav.docked = along >= -2 * DOCK_ACCEPT[0] and abs(lat) <= 2 * DOCK_ACCEPT[1]
                self.nav.last_message = ("Start im Dock" if self.nav.docked else "Start in der Sicherheitszone") + \
                                        " – nur undock() erlaubt."
                log.warning("⚠ " + self.nav.last_message)
            else:
                self.nav.last_message = "Startpose gemessen."
            return True
        return self._run("initialize", _init, needs_init=False)

    def go_to(self, x, y, theta=None):
        return self._run(f"go_to({x:.3f}, {y:.3f}, {theta})", self.nav.go_to, (x, y, theta))

    def approach(self, x, y, angle=APPROACH_ANGLE_DEFAULT, standoff=APPROACH_STANDOFF):
        return self._run(f"approach({x:.3f}, {y:.3f}, {angle}, {standoff})", self.nav.approach, (x, y), angle, standoff)

    def approach_pre(self, x, y, angle=APPROACH_ANGLE_DEFAULT, standoff=APPROACH_STANDOFF):
        return self._run(f"approach_pre({x:.3f}, {y:.3f}, {angle}, {standoff})", self.nav.approach_pre,
                         (x, y), angle, standoff)

    def approach_final(self, x, y, angle=APPROACH_ANGLE_DEFAULT):
        return self._run(f"approach_final({x:.3f}, {y:.3f}, {angle})", self.nav.approach_final, (x, y), angle)

    def back_off(self, distance):
        return self._run(f"back_off({distance:.2f})", self.nav.back_off, distance)

    def dock_pre(self):
        return self._run("dock_pre", self.nav.dock_pre)

    def dock(self):
        return self._run("dock", self.nav.dock)

    def enter_zone(self, x, y):
        return self._run(f"enter_zone({x:.3f}, {y:.3f})", self.nav.enter_zone, (x, y))

    def undock(self):
        return self._run("undock", self.nav.undock)

    def abort(self):
        """Laufenden Befehl abbrechen (Roboter stoppt innerhalb eines Sendezyklus)."""
        if self.busy:
            log.warning(f"Abbruch angefordert ({self._current}).")
            self.abort_event.set()

    def emergency_stop(self):
        self.abort_event.set()
        self.robot.stop()

    def run_async(self, fn, *args, callback: Optional[Callable[[NavResult], None]] = None, **kwargs):
        """Befehl in eigenem Thread ausführen; callback(NavResult) am Ende."""
        def worker():
            res = fn(*args, **kwargs)
            if callback is not None:
                callback(res)
        th = threading.Thread(target=worker, daemon=True, name=f"nav-{getattr(fn, '__name__', 'cmd')}")
        th.start()
        return th

    def shutdown(self):
        self.abort_event.set()
        self.robot.shutdown()

    # ---------- intern ----------
    def _run(self, name, fn, *args, needs_init=True):
        if not self._lock.acquire(blocking=False):
            return self._result(NavResult(NavStatus.REJECTED, f"Beschäftigt mit {self._current}", self.pose, name))
        self._current = name
        self.abort_event.clear()
        try:
            if needs_init and not self.initialized:
                raise NavRejected("Nicht initialisiert – zuerst initialize().")
            ok = fn(*args)
            res = NavResult(NavStatus.SUCCESS if ok else NavStatus.FAILED, self.nav.last_message)
        except NavRejected as e:
            res = NavResult(NavStatus.REJECTED, str(e))
        except NavAborted as e:
            self._safe_resync()
            res = NavResult(NavStatus.ABORTED, str(e))
        except SafetyStop as e:
            self.robot.stop()
            log.error(f"🛑 SICHERHEITSSTOPP: {e}")
            self._safe_resync()
            res = NavResult(NavStatus.SAFETY_STOP, str(e))
        except Exception as e:
            self.robot.stop()
            log.exception("Unerwarteter Fehler")
            res = NavResult(NavStatus.ERROR, f"{type(e).__name__}: {e}")
        finally:
            self._current = ""
            self._lock.release()
        res.pose, res.command = self.pose, name
        return self._result(res)

    def _safe_resync(self):
        try:
            self.nav.resync_pose()
        except Exception as e:
            log.error(f"Pose nach Stopp nicht messbar: {e}")

    def _result(self, res):
        log.info(f"[{res.command}] -> {res.status.value}: {res.message}")
        if self.on_event is not None:
            try:
                self.on_event("result", {"result": res})
            except Exception as e:
                log.warning(f"on_event-Callback fehlgeschlagen: {e}")
        return res


# ==========================================
# MAIN (CLI zum Testen)
# ==========================================
def parse_goal(txt):
    parts = [float(v) for v in txt.replace(",", " ").split()]
    if len(parts) not in (2, 3):
        raise ValueError("Format: x y oder x y theta (auch mit Kommas)")
    return parts[0], parts[1], (parts[2] if len(parts) == 3 else None)


def parse_command(txt):
    """
    'x y [theta]'                -> ("goto", (x, y, theta|None))
    'a x y [winkel] [abstand]'   -> ("approach", (x, y, winkel, abstand))   (auch 'a:x,y,...')
    'd' | 'dock'                 -> ("dock", None)
    'u' | 'undock'               -> ("undock", None)
    's' | 'status'               -> ("status", None)
    """
    t = txt.strip()
    low = t.lower()
    if low in ("d", "dock"):
        return "dock", None
    if low in ("u", "undock"):
        return "undock", None
    if low in ("s", "status"):
        return "status", None
    if low[:1] == "a" and (len(t) == 1 or t[1] in " :"):
        parts = [float(v) for v in t[1:].lstrip(":").replace(",", " ").split()]
        if len(parts) not in (2, 3, 4):
            raise ValueError("Format Anfahren: a x y [winkel] [abstand]")
        angle = parts[2] if len(parts) >= 3 else APPROACH_ANGLE_DEFAULT
        standoff = parts[3] if len(parts) == 4 else APPROACH_STANDOFF
        return "approach", (parts[0], parts[1], angle, standoff)
    return "goto", parse_goal(t)


def load_rotation_model(path, disabled):
    if disabled:
        log.info("Drehmodell deaktiviert (--no-rot) -> Drehungen mit yaw_left/yaw_right.")
        return None
    if rotation_model is None or not os.path.isfile(path):
        log.info(f"Kein Drehmodell ({path}) gefunden -> Drehungen mit yaw_left/yaw_right.")
        return None
    rm = joblib.load(path)
    log.info(f"Drehmodell geladen: {path} ({', '.join(f'{k}/{d}' for k, d in sorted(rm.groups))})")
    return rm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="Simulation statt echtem Roboter")
    ap.add_argument("--goals", nargs="*", default=None,
                    help="Befehle nacheinander: x,y[,theta] | a:x,y[,winkel[,abstand]] | d | u")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--rot-model", default=ROT_MODEL_PATH)
    ap.add_argument("--no-rot", action="store_true", help="Drehmodell nicht verwenden")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mode", choices=["schema", "mpc"], default="schema",
                    help="schema: drehen -> geradeaus -> Feinanflug (Standard); mpc: nur Beam-Search")
    ap.add_argument("--start", default="2.0,-0.25,0", help="Startpose in der Simulation: x,y,theta (absolut)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ctrl = NavigationController.create(sim=args.sim, model_path=args.model, rot_model_path=args.rot_model,
                                       no_rot=args.no_rot, mode=args.mode, seed=args.seed,
                                       sim_start=tuple(float(v) for v in args.start.split(",")))
    try:
        if not args.sim:
            print("\n" + "!" * 50 + "\n! ACHTUNG: ROBOTER BEWEGT SICH AUTOMATISCH !\n" + "!" * 50)
            if input("Tippe 'START' zum Beginnen: ").strip() != "START":
                return
        res = ctrl.initialize()
        if not res.ok:
            return
        x0, y0, th0 = ctrl.pose
        print(f"Startpose: x={x0:.3f} y={y0:.3f} θ={th0:.1f}°  (absolute Kamerakoordinaten, Zone x < {SAFETY_ZONE_X} m)"
              f"  -> {ctrl.location}")

        def do(txt):
            kind, val = parse_command(txt)
            if kind == "status":
                print(ctrl.status())
            elif kind == "dock":
                ctrl.dock()
            elif kind == "undock":
                ctrl.undock()
            elif kind == "approach":
                ctrl.approach(*val)
            else:
                ctrl.go_to(*val)

        if args.goals:
            for g in args.goals:
                do(g)
        else:
            while True:
                txt = input("\nZiel (x y [theta]) | a x y [winkel] [abstand] | d | u | s | q: ").strip()
                if txt.lower() in ("q", "quit", "exit", ""):
                    break
                try:
                    do(txt)
                except ValueError as e:
                    print(f"Ungültige Eingabe: {e}")
    except KeyboardInterrupt:
        print("\nManuell abgebrochen (Ctrl+C).")
    finally:
        print("Stoppe Roboter...")
        ctrl.shutdown()


if __name__ == "__main__":
    main()