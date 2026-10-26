"""
collect_common.py
Gemeinsamer Code für collect_linear.py und collect_mixed.py (braucht KEIN gelerntes Modell).

- Pose-Tracking über MQTT, Roboter-Steuerung über Lite3UdpCommander
- Messung wie im bisherigen Datensatz: Endpose gemittelt 2–3 s nach dem Stopp
- Sichere Befehlswahl: Zufallsbefehle, deren grob geschätzter Endpunkt außerhalb des
  Arbeitsbereichs läge, werden neu gezogen -> auch lange Läufe möglich (Richtung Mitte)
- Rückkehr zum Startpunkt ohne Modell (vor/zurück/seitlich, ohne Drehen)
- CSV im Format von motion_dataset.csv (-> train_dynamics.py)
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
# KONFIGURATION (gemeinsam)
# ==========================================
ROBOT_IP = "192.168.1.120"
MQTT_BROKER = "192.168.2.1"
TOPIC_POSE = "station/tracker/robot/pose"

SPEED_X = 15000
SPEED_Y = 20000
SPEED_YAW = 20000

SEND_PERIOD = 0.05
AVERAGING_START = 3.0
AVERAGING_END = 4.0
START_POSE_AVERAGE_WINDOW = 1.0
MIN_POSE_SAMPLES = 3

# Arbeitsbereich um den Startpunkt
WORKSPACE_RADIUS = 1.0           # m: geschätzter Endpunkt muss hier drin liegen, sonst neu ziehen
MAX_DISTANCE_FROM_START = 1.6    # m: harter Not-Aus
MAX_POSE_AGE = 1.5
MAX_RESAMPLE = 200

# Grobe Schätzwerte (nur für Sicherheit/Rückkehr, NICHT fürs Modell)
ROUGH_SPEED_X = 0.6             # m/s bei SPEED_X
ROUGH_SPEED_Y = 0.12             # m/s bei SPEED_Y
ROUGH_DELAY = 0.35               # s

# Rückkehr
RETURN_RADIUS = 0.5              # m: weiter weg -> zurück Richtung Start
RETURN_TARGET = 0.3             # m
RETURN_MAX_STEPS = 6
RETURN_MAX_DURATION = 3.0

FIELDS = [
    "sample_id", "start_timestamp", "command_start_timestamp", "stop_timestamp", "end_timestamp",
    "start_x", "start_y", "start_theta", "end_x", "end_y", "end_theta",
    "delta_x", "delta_y", "delta_theta", "command_x", "command_y", "command_yaw",
    "command_name", "command_duration", "mean_x", "mean_y", "mean_theta",
    "std_x", "std_y", "std_theta", "num_pose_samples",
]
RETURN_MOVES = {
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


def body_to_world(dxb, dyb, theta_deg):
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return c * dxb - s * dyb, s * dxb + c * dyb


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


def rough_body_delta(cmd, T):
    """Sehr grobe Schätzung der Verschiebung im Körpersystem (nur für die Sicherheit)."""
    eff = max(0.0, T - ROUGH_DELAY)
    return (cmd[0] / SPEED_X * ROUGH_SPEED_X * eff, cmd[1] / SPEED_Y * ROUGH_SPEED_Y * eff)


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

    def watch(self, home):
        """Während der Fahrt: Not-Aus bei Verlassen des Radius."""
        latest = self.tracker.latest()
        if latest and home and math.hypot(latest[1] - home[0], latest[2] - home[1]) > MAX_DISTANCE_FROM_START:
            raise RuntimeError("Sicherheitsabbruch während der Fahrt: Radius verlassen!")

    def initial_pose(self):
        time.sleep(START_POSE_AVERAGE_WINDOW + 0.1)
        t = time.time()
        return pose_stats(self.tracker.window(t - START_POSE_AVERAGE_WINDOW, t))

    def run(self, cmd, duration, home=None):
        t_cmd = time.time()
        t_end = t_cmd + duration
        try:
            while time.time() < t_end:
                self.cmd.send_axes(*cmd)
                self.watch(home)
                time.sleep(SEND_PERIOD)
        finally:
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
        pass

    def _meas(self):
        x, y, th = self.pose
        return {"mean_x": x, "mean_y": y, "mean_theta": th, "std_x": 0.002, "std_y": 0.002,
                "std_theta": 0.4, "count": 15}

    def initial_pose(self):
        return self._meas()

    def run(self, cmd, T, home=None):
        dxb, dyb = rough_body_delta(cmd, T)
        dth = -cmd[2] / SPEED_YAW * 56 * max(0.0, T - ROUGH_DELAY) + self.rng.gauss(0, 2)
        dx, dy = body_to_world(dxb + self.rng.gauss(0, 0.03), dyb + self.rng.gauss(0, 0.03), self.pose[2])
        self.pose = [self.pose[0] + dx, self.pose[1] + dy, wrap_angle(self.pose[2] + dth)]
        now = time.time()
        return self._meas(), now, now + T

    def shutdown(self):
        pass


# ==========================================
# ABLAUF
# ==========================================
def make_row(sample_id, name, cmd, T, start, end, t_cmd, t_stop):
    return {
        "sample_id": sample_id, "start_timestamp": t_cmd, "command_start_timestamp": t_cmd,
        "stop_timestamp": t_stop, "end_timestamp": time.time(),
        "start_x": start["mean_x"], "start_y": start["mean_y"], "start_theta": start["mean_theta"],
        "end_x": end["mean_x"], "end_y": end["mean_y"], "end_theta": end["mean_theta"],
        "delta_x": end["mean_x"] - start["mean_x"], "delta_y": end["mean_y"] - start["mean_y"],
        "delta_theta": wrap_angle(end["mean_theta"] - start["mean_theta"]),
        "command_x": cmd[0], "command_y": cmd[1], "command_yaw": cmd[2],
        "command_name": name, "command_duration": T,
        "mean_x": end["mean_x"], "mean_y": end["mean_y"], "mean_theta": end["mean_theta"],
        "std_x": end["std_x"], "std_y": end["std_y"], "std_theta": end["std_theta"],
        "num_pose_samples": end["count"],
    }


def open_csv(path):
    new = not os.path.isfile(path)
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    return f, w


def return_home(robot, stats, home, writer, f, counter):
    for step in range(1, RETURN_MAX_STEPS + 1):
        dx, dy = home[0] - stats["mean_x"], home[1] - stats["mean_y"]
        dist = math.hypot(dx, dy)
        if dist <= RETURN_TARGET:
            break
        bx, by = world_to_body(dx, dy, stats["mean_theta"])
        name = max(RETURN_MOVES, key=lambda n: RETURN_MOVES[n][0][0] * bx + RETURN_MOVES[n][0][1] * by)
        (ux, uy), cmd = RETURN_MOVES[name]
        speed = ROUGH_SPEED_X if ux else ROUGH_SPEED_Y
        T = min(RETURN_MAX_DURATION, max(0.3, (ux * bx + uy * by) / speed + ROUGH_DELAY))
        print(f"  [Rückkehr {step}] {dist * 100:.0f} cm vom Start -> {name} {T:.2f}s")
        end, t_cmd, t_stop = robot.run(cmd, T, home)
        counter[0] += 1
        writer.writerow(make_row(counter[0], name, cmd, T, stats, end, t_cmd, t_stop))
        f.flush()
        stats = end
    return stats


def draw_safe(actions, duration_ranges, stats, home):
    """Zufälliger Befehl, dessen grob geschätzter Endpunkt im Arbeitsbereich liegt."""
    for _ in range(MAX_RESAMPLE):
        name, cmd = random.choice(actions)
        T = random.uniform(*duration_ranges[name])
        dxb, dyb = rough_body_delta(cmd, T)
        dx, dy = body_to_world(dxb, dyb, stats["mean_theta"])
        if math.hypot(stats["mean_x"] + dx - home[0], stats["mean_y"] + dy - home[1]) <= WORKSPACE_RADIUS:
            return name, cmd, T
    return None


def run_collection(title, actions, duration_ranges, csv_path, return_log, default_samples):
    """
    actions: Liste (name, (cmd_x, cmd_y, cmd_yaw)); duration_ranges: name -> (min, max) in s.
    """
    ap = argparse.ArgumentParser(description=title)
    ap.add_argument("--sim", action="store_true", help="Logik testen ohne Roboter")
    ap.add_argument("--samples", type=int, default=default_samples)
    args = ap.parse_args()
    if args.sim:
        csv_path = csv_path.replace(".csv", "_sim.csv")
        return_log = return_log.replace(".csv", "_sim.csv")
    csv_path, return_log = paths.dataset(csv_path), paths.dataset(return_log)   # data/datasets/

    meta = {"title": title, "actions": actions, "duration_ranges": duration_ranges,
            **{k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str))}}
    with open(paths.meta_file(csv_path, int(time.time())), "w") as fm:
        json.dump(meta, fm, indent=2)

    robot = SimRobot() if args.sim else Robot()
    f, writer = open_csv(csv_path)
    fr, rwriter = open_csv(return_log)
    counter = [0]
    try:
        if not args.sim:
            print("\n" + "!" * 50 + "\n! ACHTUNG: ROBOTER BEWEGT SICH AUTOMATISCH !\n" + "!" * 50)
            if input("Tippe 'START' zum Beginnen: ").strip() != "START":
                return
        stats = robot.initial_pose()
        if not stats:
            raise RuntimeError("Keine initiale Pose empfangen!")
        home = (stats["mean_x"], stats["mean_y"])
        print(f"{title}\nStartpunkt: x={home[0]:.3f} y={home[1]:.3f} θ={stats['mean_theta']:.1f}°")

        for i in range(1, args.samples + 1):
            robot.check(home)
            if math.hypot(stats["mean_x"] - home[0], stats["mean_y"] - home[1]) > RETURN_RADIUS:
                stats = return_home(robot, stats, home, rwriter, fr, counter)
            pick = draw_safe(actions, duration_ranges, stats, home)
            if pick is None:
                stats = return_home(robot, stats, home, rwriter, fr, counter)
                continue
            name, cmd, T = pick
            print(f"\nSample {i}/{args.samples}: {name} UDP={list(cmd)} {T:.2f}s")
            start = stats
            end, t_cmd, t_stop = robot.run(cmd, T, home)
            row = make_row(i, name, cmd, T, start, end, t_cmd, t_stop)
            dxb, dyb = world_to_body(row["delta_x"], row["delta_y"], start["mean_theta"])
            print(f"  -> dx_body={dxb * 100:+.1f} cm  dy_body={dyb * 100:+.1f} cm  dθ={row['delta_theta']:+.1f}°"
                  f"  ({end['count']} Posen)")
            writer.writerow(row)
            f.flush()
            stats = end
        print(f"\nFertig! Daten in {csv_path}, Rückkehr-Läufe in {return_log}")
    except KeyboardInterrupt:
        print("\nManuell abgebrochen (Ctrl+C).")
    except Exception as e:
        print(f"\nKritischer Fehler, Notstopp! Grund: {e}")
    finally:
        print("Stoppe Roboter...")
        robot.shutdown()
        f.close()
        fr.close()
