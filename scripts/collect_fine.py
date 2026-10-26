"""
collect_fine.py — kleine, präzise Bewegungsprimitive (Fokus: Drehen + seitlich links/rechts)
Braucht KEIN gelerntes Modell. Baut auf collect_common.py auf (Tracking, Not-Aus, Rückkehr, Messfenster).

Da der Roboter kleinere Geschwindigkeitswerte ignoriert, entsteht "klein & präzise" hier NICHT über
die Geschwindigkeit, sondern über die ZEIT und über Gegenimpulse:

  pulse   : ein kurzer Impuls einer Achse                       (0.10 … ~1 s)
            yaw_left / yaw_right / left / right / forward / backward
            -> Namen identisch zu den bestehenden Aktionen, direkt fürs Dynamikmodell nutzbar
  brake   : Impuls t1, SOFORT kurzer Gegenimpuls t2 (<t1), dann Stopp
            -> bremst Nachlauf/Überschwingen ab, liefert sehr kleine Netto-Bewegungen
            yaw_left_brake / yaw_right_brake / left_brake / right_brake
  2tap    : zwei kurze Impulse gleicher Richtung mit Pause dazwischen
            -> Wiederholgenauigkeit von "Schritten", kleine Winkel in zwei Stufen
            yaw_left_2tap / yaw_right_2tap / left_2tap / right_2tap
  sideyaw : kurzer Seitschritt + gleichzeitige Drehung (Feinkorrektur Lage + Winkel)
            sideyaw_right / sideyaw_left   (Richtungen wie side_arc_* in collect_mixed.py)

Ausgewogene Abdeckung: Statt rein zufällig zu ziehen, bekommt jedes Primitiv seinen Anteil
(WEIGHTS), und die Dauern werden pro Primitiv stratifiziert über den ganzen Bereich verteilt
(kein Klumpen, auch die ganz kurzen Impulse / Totzone werden sicher abgedeckt). Reihenfolge gemischt.

Nutzung:
    python collect_fine.py                       # alle Gruppen, NUM_SAMPLES
    python collect_fine.py --samples 800
    python collect_fine.py --groups yaw,lateral  # nur Dreh- und Seitwärts-Primitive
    python collect_fine.py --sim                 # Logik testen ohne Roboter

Ausgabe: fine_dataset.csv (+ fine_return_log.csv)
  Spalten wie motion_dataset.csv, plus:
    kind            pulse | brake | 2tap | sideyaw
    segments        JSON-Liste [[cmd_x, cmd_y, cmd_yaw, dauer], ...] (so wie gesendet)
    actual_duration tatsächliche Befehlszeit (Sendeschleife rundet auf SEND_PERIOD)
    dx_body, dy_body  Verschiebung im Körpersystem der Startpose
  Bei pulse-Zeilen sind command_x/y/yaw/duration exakt der Einzelbefehl (-> train_dynamics.py).
  Bei zusammengesetzten Zeilen (brake/2tap) steht in command_* der ERSTE Impuls und in
  command_duration die Summe der aktiven Zeiten; für Einzelbefehl-Training nach kind == "pulse"
  filtern, die anderen als Sequenz-Primitive (segments) verwenden.
"""
import argparse
import csv
import json
import math
import os
import random
import time

import collect_common as cc
import paths
from collect_common import (SPEED_X, SPEED_Y, SPEED_YAW, Robot, SimRobot, body_to_world,
                            make_row, pose_stats, return_home, rough_body_delta, world_to_body)

# ==========================================
# KONFIGURATION
# ==========================================
NUM_SAMPLES = 600

X, Y, W = SPEED_X, SPEED_Y, SPEED_YAW

# Primitive: name -> dict(group, kind, cmd, Parameterbereiche)
#   pulse  : T
#   brake  : T1, FRAC (t2 = FRAC * t1, Gegenrichtung)
#   2tap   : T (je Impuls), GAP (Pause)
PRIMITIVES = {
    # --- Drehen ---
    "yaw_left":        dict(group="yaw", kind="pulse", cmd=(0, 0, W), T=(0.10, 1.00)),
    "yaw_right":       dict(group="yaw", kind="pulse", cmd=(0, 0, -W), T=(0.10, 1.00)),
    "yaw_left_brake":  dict(group="yaw", kind="brake", cmd=(0, 0, W), T1=(0.20, 1.00), FRAC=(0.10, 0.70)),
    "yaw_right_brake": dict(group="yaw", kind="brake", cmd=(0, 0, -W), T1=(0.20, 1.00), FRAC=(0.10, 0.70)),
    "yaw_left_2tap":   dict(group="yaw", kind="2tap", cmd=(0, 0, W), T=(0.10, 0.45), GAP=(0.15, 0.60)),
    "yaw_right_2tap":  dict(group="yaw", kind="2tap", cmd=(0, 0, -W), T=(0.10, 0.45), GAP=(0.15, 0.60)),
    # --- seitlich ---
    "right":           dict(group="lateral", kind="pulse", cmd=(0, Y, 0), T=(0.10, 1.20)),
    "left":            dict(group="lateral", kind="pulse", cmd=(0, -Y, 0), T=(0.10, 1.20)),
    "right_brake":     dict(group="lateral", kind="brake", cmd=(0, Y, 0), T1=(0.25, 1.20), FRAC=(0.10, 0.70)),
    "left_brake":      dict(group="lateral", kind="brake", cmd=(0, -Y, 0), T1=(0.25, 1.20), FRAC=(0.10, 0.70)),
    "right_2tap":      dict(group="lateral", kind="2tap", cmd=(0, Y, 0), T=(0.10, 0.50), GAP=(0.15, 0.60)),
    "left_2tap":       dict(group="lateral", kind="2tap", cmd=(0, -Y, 0), T=(0.10, 0.50), GAP=(0.15, 0.60)),
    # --- seitlich + drehen (Feinkorrektur) ---
    "sideyaw_right":   dict(group="sideyaw", kind="sideyaw", cmd=(0, Y, -W), T=(0.10, 0.80)),
    "sideyaw_left":    dict(group="sideyaw", kind="sideyaw", cmd=(0, -Y, W), T=(0.10, 0.80)),
    # --- kurze Vor/Zurück-Tipper (Ergänzung, damit das Modell kleine x-Schritte kennt) ---
    "forward":         dict(group="longitudinal", kind="pulse", cmd=(X, 0, 0), T=(0.10, 0.60)),
    "backward":        dict(group="longitudinal", kind="pulse", cmd=(-X, 0, 0), T=(0.10, 0.60)),
}

# relative Anteile (Fokus auf Drehen und seitlich)
WEIGHTS = {
    "yaw_left": 3, "yaw_right": 3, "yaw_left_brake": 2, "yaw_right_brake": 2,
    "yaw_left_2tap": 1, "yaw_right_2tap": 1,
    "right": 3, "left": 3, "right_brake": 2, "left_brake": 2, "right_2tap": 1, "left_2tap": 1,
    "sideyaw_right": 1, "sideyaw_left": 1,
    "forward": 1, "backward": 1,
}

CSV_PATH = "fine_dataset.csv"
RETURN_LOG = "fine_return_log.csv"
EXTRA_FIELDS = ["kind", "segments", "actual_duration", "dx_body", "dy_body"]
FIELDS = cc.FIELDS + EXTRA_FIELDS
MAX_DEFER = 3          # so oft darf ein Primitiv wegen Arbeitsbereich nach hinten verschoben werden


# ==========================================
# PLANUNG: ausgewogen + stratifizierte Dauern
# ==========================================
def stratified(lo, hi, n, rng):
    """n Werte, je einer pro gleich breitem Teilintervall, gemischt."""
    vals = [lo + (k + rng.random()) / n * (hi - lo) for k in range(n)]
    rng.shuffle(vals)
    return vals


def build_schedule(names, total, rng):
    wsum = sum(WEIGHTS[n] for n in names)
    counts = {n: max(1, round(total * WEIGHTS[n] / wsum)) for n in names}
    schedule = []
    for name in names:
        p, n = PRIMITIVES[name], counts[name]
        if p["kind"] == "brake":
            t1s, fracs = stratified(*p["T1"], n, rng), stratified(*p["FRAC"], n, rng)
            params = [dict(t1=a, frac=b) for a, b in zip(t1s, fracs)]
        elif p["kind"] == "2tap":
            ts, gaps = stratified(*p["T"], n, rng), stratified(*p["GAP"], n, rng)
            params = [dict(t=a, gap=b) for a, b in zip(ts, gaps)]
        else:
            params = [dict(t=a) for a in stratified(*p["T"], n, rng)]
        schedule += [(name, prm) for prm in params]
    rng.shuffle(schedule)
    return schedule


def segments_for(name, prm):
    p = PRIMITIVES[name]
    cmd = p["cmd"]
    neg = tuple(-c for c in cmd)
    if p["kind"] == "brake":
        t1 = prm["t1"]
        t2 = max(0.05, prm["frac"] * t1)
        return [(cmd, t1), (neg, t2)]
    if p["kind"] == "2tap":
        return [(cmd, prm["t"]), ((0, 0, 0), prm["gap"]), (cmd, prm["t"])]
    return [(cmd, prm["t"])]


def rough_end_ok(segments, stats, home):
    """Grobe Schätzung des Endpunkts (nur Sicherheit): liegt er im Arbeitsbereich?"""
    dxb = dyb = 0.0
    for cmd, T in segments:
        a, b = rough_body_delta(cmd, T)
        dxb, dyb = dxb + a, dyb + b
    dx, dy = body_to_world(dxb, dyb, stats["mean_theta"])
    return math.hypot(stats["mean_x"] + dx - home[0], stats["mean_y"] + dy - home[1]) <= cc.WORKSPACE_RADIUS


# ==========================================
# ROBOTER: Segmentfolgen ausführen
# ==========================================
class FineRobot(Robot):
    def run_segments(self, segments, home=None):
        """segments direkt hintereinander, Stopp erst am Ende; Not-Aus-Überwachung läuft mit."""
        t_cmd = time.time()
        try:
            for cmd, duration in segments:
                t_end = time.time() + duration
                while time.time() < t_end:
                    self.cmd.send_axes(*cmd)
                    self.watch(home)
                    time.sleep(cc.SEND_PERIOD)
        finally:
            t_stop = time.time()
            self.cmd.stop_robot(buffer_s=0.0)
        time.sleep(cc.AVERAGING_END + 0.1)
        stats = pose_stats(self.tracker.window(t_stop + cc.AVERAGING_START, t_stop + cc.AVERAGING_END))
        if not stats or stats["count"] < cc.MIN_POSE_SAMPLES:
            raise RuntimeError("Zu wenige Posen im Messfenster! Tracker ausgefallen?")
        return stats, t_cmd, t_stop


class FineSimRobot(SimRobot):
    """Fake-Physik NUR zum Testen der Script-Logik (mit Totzeit -> kurze Impulse ~wirkungslos)."""

    def run_segments(self, segments, home=None):
        t0 = time.time()
        for cmd, T in segments:
            self.run(cmd, T, home)
        return self._meas(), t0, t0 + sum(T for _, T in segments)


# ==========================================
# MAIN
# ==========================================
def open_csv(path, fields):
    new = not os.path.isfile(path)
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    if new:
        w.writeheader()
    return f, w


def main():
    groups_all = sorted({p["group"] for p in PRIMITIVES.values()})
    ap = argparse.ArgumentParser(description="Kleine präzise Bewegungsprimitive")
    ap.add_argument("--sim", action="store_true", help="Logik testen ohne Roboter")
    ap.add_argument("--samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--groups", default="all", help=f"Komma-Liste aus {groups_all} oder 'all'")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    groups = groups_all if args.groups == "all" else [g.strip() for g in args.groups.split(",")]
    names = [n for n, p in PRIMITIVES.items() if p["group"] in groups]
    if not names:
        raise SystemExit(f"Keine Primitive für --groups {args.groups}. Erlaubt: {groups_all}")

    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)
    schedule = build_schedule(names, args.samples, rng)

    csv_path, ret_path = CSV_PATH, RETURN_LOG
    if args.sim:
        csv_path, ret_path = csv_path.replace(".csv", "_sim.csv"), ret_path.replace(".csv", "_sim.csv")
    csv_path, ret_path = paths.dataset(csv_path), paths.dataset(ret_path)   # data/datasets/

    meta = {"title": "Feine Primitive", "seed": seed, "groups": groups, "primitives": PRIMITIVES,
            "weights": WEIGHTS, "planned_samples": len(schedule),
            **{k: v for k, v in vars(cc).items() if k.isupper() and isinstance(v, (int, float, str))}}
    with open(paths.meta_file(csv_path, int(time.time())), "w") as fm:
        json.dump(meta, fm, indent=2, default=str)

    robot = FineSimRobot() if args.sim else FineRobot()
    f, writer = open_csv(csv_path, FIELDS)
    fr, rwriter = open_csv(ret_path, cc.FIELDS)
    counter = [0]
    done = 0
    try:
        if not args.sim:
            print("\n" + "!" * 50 + "\n! ACHTUNG: ROBOTER BEWEGT SICH AUTOMATISCH !\n" + "!" * 50)
            if input("Tippe 'START' zum Beginnen: ").strip() != "START":
                return
        stats = robot.initial_pose()
        if not stats:
            raise RuntimeError("Keine initiale Pose empfangen!")
        home = (stats["mean_x"], stats["mean_y"])
        print(f"Feine Primitive ({len(schedule)} geplant, Seed {seed}, Gruppen {groups})")
        print(f"Startpunkt: x={home[0]:.3f} y={home[1]:.3f} θ={stats['mean_theta']:.1f}°")

        queue = [(name, prm, 0) for name, prm in schedule]
        while queue:
            name, prm, deferred = queue.pop(0)
            robot.check(home)
            if math.hypot(stats["mean_x"] - home[0], stats["mean_y"] - home[1]) > cc.RETURN_RADIUS:
                stats = return_home(robot, stats, home, rwriter, fr, counter)

            segs = segments_for(name, prm)
            if not rough_end_ok(segs, stats, home):
                if deferred < MAX_DEFER:
                    queue.append((name, prm, deferred + 1))
                stats = return_home(robot, stats, home, rwriter, fr, counter)
                continue

            done += 1
            desc = " -> ".join(f"[{c[0]},{c[1]},{c[2]}] {T:.2f}s" for c, T in segs)
            print(f"\nSample {done}/{len(schedule)}: {name}   {desc}")
            start = stats
            end, t_cmd, t_stop = robot.run_segments(segs, home)

            active = sum(T for c, T in segs if any(c))
            row = make_row(done, name, segs[0][0], active, start, end, t_cmd, t_stop)
            dxb, dyb = world_to_body(row["delta_x"], row["delta_y"], start["mean_theta"])
            row.update({
                "kind": PRIMITIVES[name]["kind"],
                "segments": json.dumps([[c[0], c[1], c[2], round(T, 4)] for c, T in segs]),
                "actual_duration": t_stop - t_cmd, "dx_body": dxb, "dy_body": dyb,
            })
            print(f"  -> dx_body={dxb * 100:+.1f} cm  dy_body={dyb * 100:+.1f} cm  "
                  f"dθ={row['delta_theta']:+.1f}°  ({end['count']} Posen)")
            writer.writerow(row)
            f.flush()
            stats = end
        print(f"\nFertig! {done} Samples in {csv_path}, Rückkehr-Läufe in {ret_path}")
    except KeyboardInterrupt:
        print(f"\nManuell abgebrochen (Ctrl+C) nach {done} Samples.")
    except Exception as e:
        print(f"\nKritischer Fehler, Notstopp! Grund: {e}")
    finally:
        print("Stoppe Roboter...")
        robot.shutdown()
        f.close()
        fr.close()


if __name__ == "__main__":
    main()
