"""
dynamics_model.py
Gelerntes Primitiv-Dynamikmodell für den Lite3:
    (Aktion, Dauer T)  ->  (dx_body, dy_body, dtheta)  inkl. Unsicherheit

Aufbau pro Aktion und Zielgröße:
    1) Physik-Baseline:  v * softplus(T - t_delay) + b
       (Geschwindigkeit, Totzeit, konstanter Versatz z.B. Start/Stopp-Schritt)
    2) Gaussian Process auf den Residuen der Baseline (liefert auch die Streuung)

Konvention (wie im Kamerasystem):
    x_body = vorne, y_body = +90° gegenüber x_body im Kamerasystem
    (bei Bildkoordinaten mit y nach unten: +y_body = RECHTS, +theta = Rechtsdrehung)
    Alle Winkel in Grad, Längen in Metern.

Dieses Modul wird von train_dynamics.py (Training) und später vom MPC (Nutzung) importiert.
"""
import math
import numpy as np

TARGETS = ["dx_body", "dy_body", "dtheta"]
SOFTPLUS_K = 25.0  # Schärfe des "Knicks" bei der Totzeit (1/s)


def wrap_angle(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


def world_to_body(dx, dy, theta_deg):
    """Welt-Delta -> Körper-Delta (bezogen auf Start-Orientierung)."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return c * dx + s * dy, -s * dx + c * dy


def body_to_world(dxb, dyb, theta_deg):
    """Körper-Delta -> Welt-Delta."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return c * dxb - s * dyb, s * dxb + c * dyb


def baseline_fn(T, v, t_delay, b):
    """Glatte Version von v * max(0, T - t_delay) + b."""
    z = SOFTPLUS_K * (np.asarray(T, dtype=float) - t_delay)
    sp = np.where(z > 30, z, np.log1p(np.exp(np.minimum(z, 30)))) / SOFTPLUS_K
    return v * sp + b


class DynamicsModel:
    """Container für die gefitteten Parameter + GPs. Wird per joblib gespeichert."""

    def __init__(self):
        self.baseline = {}   # (action, target) -> (v, t_delay, b)
        self.gps = {}        # (action, target) -> sklearn GP oder None
        self.resid_std = {}  # (action, target) -> Streuung, falls kein GP
        self.duration_range = {}  # action -> (min, max) der Trainingsdaten
        self.command = {}    # action -> (cmd_x, cmd_y, cmd_yaw)
        self.meta = {}

    @property
    def actions(self):
        return sorted(self.command.keys())

    def predict(self, action, duration):
        """
        Vorhersage für eine oder mehrere Dauern.
        Rückgabe: mean, std  (je Array der Form (N, 3): dx_body, dy_body, dtheta)
        """
        T = np.atleast_1d(np.asarray(duration, dtype=float))
        mean = np.zeros((len(T), 3))
        std = np.zeros((len(T), 3))
        for j, tgt in enumerate(TARGETS):
            v, td, b = self.baseline[(action, tgt)]
            mean[:, j] = baseline_fn(T, v, td, b)
            gp = self.gps.get((action, tgt))
            if gp is not None:
                m, s = gp.predict(T.reshape(-1, 1), return_std=True)
                mean[:, j] += m
                std[:, j] = s
            else:
                std[:, j] = self.resid_std[(action, tgt)]
        return mean, std

    def predict_world(self, pose, action, duration):
        """
        pose = (x, y, theta_deg) im Kamerasystem.
        Rückgabe: neue Pose (x, y, theta) und std im Körpersystem (3,)
        """
        x, y, th = pose
        mean, std = self.predict(action, duration)
        dxb, dyb, dth = mean[0]
        dxw, dyw = body_to_world(dxb, dyb, th)
        return (x + dxw, y + dyw, wrap_angle(th + dth)), std[0]