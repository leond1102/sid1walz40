"""
collect_rotation.py  —  eigenständig, braucht KEIN gelerntes Modell
Sammelt Drehdaten für den Lite3 in allen Varianten:

  single       : eine Drehung mit voller Drehgeschwindigkeit           (~25° … 220°)
  flick        : t1 drehen, SOFORT t2 in Gegenrichtung, dann Stopp      (kleine Winkel, auch
                 Gegendrehung länger als die erste -> Nulldurchgang / Richtungswechsel)
  flick_pause  : wie flick, aber mit kurzer Pause (FLICK_PAUSE_GAP) zwischen den Drehungen

Pro Sample: Startpose (gemittelt im Stand) -> Befehlsfolge -> Stopp -> Endpose im Fenster
2–3 s nach dem Stopp (wie im restlichen Datensatz) -> CSV.

Rückkehr zum Startpunkt (ohne Modell): Ist der Roboter weiter als RETURN_RADIUS vom Start weg,
läuft er OHNE zu drehen in die Richtung (vor/zurück/links/rechts), die am besten zum Start zeigt,
misst nach und wiederholt, bis er innerhalb RETURN_TARGET ist. Diese Läufe werden in
rotation_return_log.csv geloggt (Spalten wie motion_dataset.csv -> nutzbar fürs Dynamikmodell).

Nutzung:
    python collect_rotation.py              # echter Roboter
    python collect_rotation.py --sim        # Logik testen ohne Roboter (Fake-Physik)
    python collect_rotation.py --samples 200
"""
import argparse
import csv
import json
import math
import os
import random
import threading
import time

import paths

# ==========================================
# KONFIGURATION
# ==========================================
ROBOT_IP = "192.168.1.120"
MQTT_BROKER = "192.168.2.1"
TOPIC_POSE = "station/tracker/robot/pose"

SPEED_X = 15000
SPEED_Y = 20000
SPEED_YAW = 20000          # Slow-Modus (Werte aus den alten Scripts); kleinere Werte ignoriert der Roboter

NUM_SAMPLES = 500

# Anteile der Befehlsarten (werden normiert)
KIND_WEIGHTS = {"single": 0.45, "flick": 0.35, "flick_pause": 0.20}

SINGLE_T_RANGE = (0.3, 2.6)          # s, volle Drehgeschwindigkeit -> bis ~220°
FLICK_T1_RANGE = (0.3, 1.5)          # s, erste Drehung
FLICK_T2_FRAC_RANGE = (0.2, 1.6)     # Gegendrehung = Anteil von t1 (>1: länger als die erste!)
FLICK_T2_MIN = 0.1
FLICK_PAUSE_GAP = 0.3                # s Pause (Stillstand-Befehl) bei flick_pause

# Timing — muss zum restlichen Datensatz passenSTART

SEND_PERIOD = 0.05
AVERAGING_START = 3.0
AVERAGING_END = 4.0
START_POSE_AVERAGE_WINDOW = 1.0

# Rückkehr zum Startpunkt (ohne Modell)
RETURN_RADIUS = 0.50                 # m: weiter weg -> zurücklaufen
RETURN_TARGET = 0.30                 # m: so nah soll er zurückkommen
RETURN_MAX_STEPS = 6
RETURN_SPEED = {"forward": 0.55, "backward": 0.50, "left": 0.12, "right": 0.12}  # m/s grob
RETURN_DELAY = 0.35                  # s grobe Totzeit
RETURN_MAX_DURATION = 3.0            # s pro Laufschritt

# Sicherheit
MAX_DISTANCE_FROM_START = 1.5        # m: harter Not-Aus-Radius
MAX_POSE_AGE = 1.5                   # s
MIN_POSE_SAMPLES = 3

CSV_PATH = "rotation_dataset.csv"
RETURN_LOG = "rotation_return_log.csv"

FIELDS = [
    "sample_id", "kind", "direction", "t1", "t2", "gap", "cmd_yaw_1", "cmd_yaw_2",
    "command_start_timestamp", "stop_timestamp",
    "start_x", "start_y", "start_theta", "end_x", "end_y", "end_theta",
    "delta_x", "delta_y", "delta_theta", "dx_body", "dy_body",
    "std_x", "std_y", "std_theta", "num_pose_samples",
]
RETURN_FIELDS = [
    "sample_id", "start_timestamp", "command_start_timestamp", "stop_timestamp", "end_timestamp",
    "start_x", "start_y", "start_theta", "end_x", "end_y", "end_theta",
    "delta_x", "delta_y", "delta_theta", "command_x", "command_y", "command_yaw",
    "command_name", "command_duration", "mean_x", "mean_y", "mean_theta",
    "std_x", "std_y", "std_theta", "num_pose_samples",
]
# Laufrichtungen im Körpersystem (x vorne, y = Richtung, in die "right" läuft) und UDP-Befehle
MOVES = {
    "forward":  ((1.0, 0.0), (SPEED_X, 0, 0)),
    "backward": ((-1.0, 0.0), (-SPEED_X, 0, 0)),
    "right":    ((0.0, 1.0), (0, SPEED_Y, 0)),
    "left":     ((0.0, -1.0), (0, -SPEED_Y, 0)),
}


# ==========================================
# HILFSFUNKTIONEN
# ==========================================
def wrap_angle(a):
    return (a + 180.0) % 360.0 - 180.0


def world_to_body(dx, dy, theta_deg):
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return c * dx + s * dy, -s * dx + c * dy


def pose_stats(poses):
    n = len(poses)
    if n == 0:
        return None
    xs, ys, ths = zip(*poses)
    mth = math.degrees(math.atan2(sum(math.sin(math.radians(t)) for t in ths),
                                  sum(math.cos(math.radians(t)) for t in ths)))
    mx, my = sum(xs) / n, sum(ys) / n
    return {"mean_x": mx, "mean_y": my, "mean_theta": mth,
            "std_x": math.sqrt(sum((x - mx) ** 2 for x in xs) / n),
            "std_y": math.sqrt(sum((y - my) ** 2 for y in ys) / n),
            "std_theta": math.sqrt(sum(wrap_angle(t - mth) ** 2 for t in ths) / n),
            "count": n}


# ==========================================
# ROBOTER
# ==========================================
class PoseTracker:
    """Posen der Ortung. Verbindet im Hintergrund (ohne Broker: keine Posen -> klare Fehlermeldung)."""

    def __init__(self):
        import paho.mqtt.client as mqtt
        self.client = mqtt.Client()
        self.client.on_connect = lambda client, *args: client.subscribe(TOPIC_POSE)   # auch nach Neuverbinden
        self.client.on_message = self._on_message
        self.lock = threading.Lock()
        self.buf = []
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.connect_async(MQTT_BROKER, 1883, 60)
        self.client.loop_start()

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


class Robot:
    def __init__(self):
        from lite3_commander import Lite3UdpCommander, Lite3UdpConfig
        self.tracker = PoseTracker()
        self.cmd = Lite3UdpCommander(Lite3UdpConfig(robot_ip=ROBOT_IP, verbose=False))
        self.cmd.start()
        time.sleep(1.0)
        self.cmd.set_control_mode_manual()
        self.cmd.set_motion_mode("move")

    def check(self, home):
        latest = self.tracker.latest()
        if not latest:
            raise RuntimeError("Sicherheitsabbruch: Keine Pose empfangen!")
        t, x, y, _ = latest
        if time.time() - t > MAX_POSE_AGE:
            raise RuntimeError(f"Sicherheitsabbruch: Pose zu alt ({time.time() - t:.2f}s)")
        if home and math.hypot(x - home[0], y - home[1]) > MAX_DISTANCE_FROM_START:
            raise RuntimeError("Sicherheitsabbruch: Roboter außerhalb des Sicherheitsradius!")

    def initial_pose(self):
        time.sleep(START_POSE_AVERAGE_WINDOW + 0.1)
        t = time.time()
        return pose_stats(self.tracker.window(t - START_POSE_AVERAGE_WINDOW, t))

    def run(self, segments):
        """segments: [(cmd_xyz, dauer), ...] direkt hintereinander, Stopp erst am Ende."""
        t_cmd = time.time()
        for cmd_xyz, duration in segments:
            t_end = time.time() + duration
            while time.time() < t_end:
                self.cmd.send_axes(*cmd_xyz)
                time.sleep(SEND_PERIOD)
        t_stop = time.time()
        self.cmd.stop_robot(buffer_s=0.0)
        time.sleep(AVERAGING_END + 0.1)
        stats = pose_stats(self.tracker.window(t_stop + AVERAGING_START, t_stop + AVERAGING_END))
        if not stats or stats["count"] < MIN_POSE_SAMPLES:
            raise RuntimeError("Zu wenige Posen im Messfenster! Tracker ausgefallen?")
        return stats, t_cmd, t_stop

    def shutdown(self):
        try:
            self.cmd.stop_robot(buffer_s=0.5)
            self.cmd.shutdown()
        except Exception:
            pass


class SimRobot:
    """Fake-Physik NUR zum Testen der Script-Logik."""

    def __init__(self):
        self.pose = [0.0, 0.0, 0.0]
        self.rng = random.Random(0)

    def check(self, home):
        if home and math.hypot(self.pose[0] - home[0], self.pose[1] - home[1]) > MAX_DISTANCE_FROM_START:
            raise RuntimeError("Sicherheitsabbruch (Sim)")

    def _meas(self):
        x, y, th = self.pose
        return {"mean_x": x, "mean_y": y, "mean_theta": th, "std_x": 0.002, "std_y": 0.002,
                "std_theta": 0.4, "count": 15}

    def initial_pose(self):
        return self._meas()

    def run(self, segments):
        dth, dxb, dyb = 0.0, 0.0, 0.0
        for (cx, cy, cyaw), T in segments:
            eff = max(0.0, T - 0.3)
            dth += -cyaw / 20000 * 56 * eff
            dxb += cx / 15000 * 0.5 * eff
            dyb += cy / 20000 * 0.12 * eff
        if any(c[2] for c, _ in segments):
            dxb -= 0.10
        dth += self.rng.gauss(0, 2)
        x, y, th = self.pose
        t = math.radians(th)
        self.pose = [x + math.cos(t) * dxb - math.sin(t) * dyb + self.rng.gauss(0, 0.02),
                     y + math.sin(t) * dxb + math.cos(t) * dyb + self.rng.gauss(0, 0.02),
                     wrap_angle(th + dth)]
        now = time.time()
        return self._meas(), now, now

    def shutdown(self):
        pass


# ==========================================
# BEFEHLE
# ==========================================
def draw_command():
    """Rückgabe: kind, direction, t1, t2, gap, segments"""
    kinds, weights = zip(*KIND_WEIGHTS.items())
    kind = random.choices(kinds, weights=weights)[0]
    direction = random.choice(["left", "right"])
    yaw1 = SPEED_YAW if direction == "left" else -SPEED_YAW   # wie yaw_left/yaw_right im Datensatz

    if kind == "single":
        t1 = random.uniform(*SINGLE_T_RANGE)
        return kind, direction, t1, 0.0, 0.0, [((0, 0, yaw1), t1)]

    t1 = random.uniform(*FLICK_T1_RANGE)
    t2 = max(FLICK_T2_MIN, random.uniform(*FLICK_T2_FRAC_RANGE) * t1)
    gap = FLICK_PAUSE_GAP if kind == "flick_pause" else 0.0
    segs = [((0, 0, yaw1), t1)]
    if gap > 0:
        segs.append(((0, 0, 0), gap))
    segs.append(((0, 0, -yaw1), t2))
    return kind, direction, t1, t2, gap, segs


# ==========================================
# RÜCKKEHR ZUM START (ohne Modell, ohne Drehen)
# ==========================================
def return_home(robot, stats, home, log_writer, log_file, counter):
    for step in range(1, RETURN_MAX_STEPS + 1):
        dx, dy = home[0] - stats["mean_x"], home[1] - stats["mean_y"]
        dist = math.hypot(dx, dy)
        if dist <= RETURN_TARGET:
            break
        bx, by = world_to_body(dx, dy, stats["mean_theta"])
        # Richtung mit dem kleinsten Winkel zum Start (max. 45° daneben)
        name = max(MOVES, key=lambda n: (MOVES[n][0][0] * bx + MOVES[n][0][1] * by))
        (ux, uy), cmd = MOVES[name]
        along = ux * bx + uy * by                           # Strecke in Laufrichtung
        T = min(RETURN_MAX_DURATION, max(0.3, along / RETURN_SPEED[name] + RETURN_DELAY))
        print(f"  [Rückkehr {step}] {dist * 100:.0f} cm vom Start -> {name} {T:.2f}s")
        end, t_cmd, t_stop = robot.run([(cmd, T)])
        counter[0] += 1
        log_writer.writerow({
            "sample_id": counter[0], "start_timestamp": t_cmd, "command_start_timestamp": t_cmd,
            "stop_timestamp": t_stop, "end_timestamp": time.time(),
            "start_x": stats["mean_x"], "start_y": stats["mean_y"], "start_theta": stats["mean_theta"],
            "end_x": end["mean_x"], "end_y": end["mean_y"], "end_theta": end["mean_theta"],
            "delta_x": end["mean_x"] - stats["mean_x"], "delta_y": end["mean_y"] - stats["mean_y"],
            "delta_theta": wrap_angle(end["mean_theta"] - stats["mean_theta"]),
            "command_x": cmd[0], "command_y": cmd[1], "command_yaw": cmd[2],
            "command_name": name, "command_duration": T,
            "mean_x": end["mean_x"], "mean_y": end["mean_y"], "mean_theta": end["mean_theta"],
            "std_x": end["std_x"], "std_y": end["std_y"], "std_theta": end["std_theta"],
            "num_pose_samples": end["count"],
        })
        log_file.flush()
        stats = end
    return stats


# ==========================================
# MAIN
# ==========================================
def open_csv(path, fields):
    new = not os.path.isfile(path)
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    return f, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--samples", type=int, default=NUM_SAMPLES)
    args = ap.parse_args()

    csv_path, ret_path = (("rotation_dataset_sim.csv", "rotation_return_log_sim.csv") if args.sim
                          else (CSV_PATH, RETURN_LOG))
    csv_path, ret_path = paths.dataset(csv_path), paths.dataset(ret_path)   # data/datasets/
    meta = {k: v for k, v in globals().items()
            if k.isupper() and isinstance(v, (int, float, str, tuple, dict))}
    with open(paths.meta_file(csv_path, int(time.time())), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    robot = SimRobot() if args.sim else Robot()
    fcsv, writer = open_csv(csv_path, FIELDS)
    fret, ret_writer = open_csv(ret_path, RETURN_FIELDS)
    ret_counter = [0]
    try:
        if not args.sim:
            print("\n" + "!" * 50 + "\n! ACHTUNG: ROBOTER BEWEGT SICH AUTOMATISCH !\n" + "!" * 50)
            if input("Tippe 'START' zum Beginnen: ").strip() != "START":
                return
        stats = robot.initial_pose()
        if not stats:
            raise RuntimeError("Keine initiale Pose empfangen!")
        home = (stats["mean_x"], stats["mean_y"])
        print(f"Startpunkt: x={home[0]:.3f} y={home[1]:.3f} θ={stats['mean_theta']:.1f}°")

        for i in range(1, args.samples + 1):
            robot.check(home)
            if math.hypot(stats["mean_x"] - home[0], stats["mean_y"] - home[1]) > RETURN_RADIUS:
                stats = return_home(robot, stats, home, ret_writer, fret, ret_counter)

            kind, direction, t1, t2, gap, segs = draw_command()
            desc = " -> ".join(f"[{c[0]},{c[1]},{c[2]}] {T:.2f}s" for c, T in segs)
            print(f"\nSample {i}/{args.samples}: {kind} {direction}   {desc}")

            start = stats
            end, t_cmd, t_stop = robot.run(segs)
            dx, dy = end["mean_x"] - start["mean_x"], end["mean_y"] - start["mean_y"]
            dth = wrap_angle(end["mean_theta"] - start["mean_theta"])
            dxb, dyb = world_to_body(dx, dy, start["mean_theta"])
            print(f"  -> dθ = {dth:+.1f}°   dx_body = {dxb * 100:+.1f} cm   dy_body = {dyb * 100:+.1f} cm"
                  f"   ({end['count']} Posen)")

            writer.writerow({
                "sample_id": i, "kind": kind, "direction": direction, "t1": t1, "t2": t2, "gap": gap,
                "cmd_yaw_1": segs[0][0][2], "cmd_yaw_2": segs[-1][0][2] if len(segs) > 1 else 0,
                "command_start_timestamp": t_cmd, "stop_timestamp": t_stop,
                "start_x": start["mean_x"], "start_y": start["mean_y"], "start_theta": start["mean_theta"],
                "end_x": end["mean_x"], "end_y": end["mean_y"], "end_theta": end["mean_theta"],
                "delta_x": dx, "delta_y": dy, "delta_theta": dth, "dx_body": dxb, "dy_body": dyb,
                "std_x": end["std_x"], "std_y": end["std_y"], "std_theta": end["std_theta"],
                "num_pose_samples": end["count"],
            })
            fcsv.flush()
            stats = end

        print(f"\nFertig! Daten in {csv_path}, Rückkehr-Läufe in {ret_path}")
    except KeyboardInterrupt:
        print("\nManuell abgebrochen (Ctrl+C).")
    except Exception as e:
        print(f"\nKritischer Fehler, Notstopp! Grund: {e}")
    finally:
        print("Stoppe Roboter...")
        robot.shutdown()
        fcsv.close()
        fret.close()


if __name__ == "__main__":
    main()
