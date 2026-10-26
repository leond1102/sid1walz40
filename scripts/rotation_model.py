"""
rotation_model.py
Gelerntes Drehmodell für den Lite3: Befehl -> (dx_body, dy_body, dtheta) inkl. Unsicherheit.

Befehlsarten (wie in collect_rotation.py):
  single / single_slow (direction, t1) : eine Drehung für t1 Sekunden (volle / reduzierte Geschwindigkeit)
  flick  (direction, t1, t2)           : t1 in "direction", sofort danach t2 in Gegenrichtung
  flick_pause (direction, t1, t2)      : wie flick, mit fester Pause dazwischen

Pro (Art, Richtung) und Zielgröße: einfache Baseline + Gaussian Process auf den Residuen.
Wird von train_rotation.py erzeugt und kann im Planer genutzt werden:
    rm = joblib.load("models/rotation_model.joblib")
    cmd = rm.best_command(+12.0)   # -> dict mit kind, direction, t1, t2, erwarteter Drehung, σ
"""
import numpy as np

from dynamics_model import baseline_fn

TARGETS = ["dx_body", "dy_body", "dtheta"]
GRID_STEP = 0.02  # s, Auflösung für best_command
SIGMA_FLOOR_DEG = 1.5


class SoftplusBaseline:
    """single: v * max(0, t1 - t_delay) + b   (wie im Dynamikmodell)"""

    def __init__(self, v, t_delay, b):
        self.p = (v, t_delay, b)

    def __call__(self, X):
        return baseline_fn(X[:, 0], *self.p)


class LinearBaseline:
    """flick: c0 + c1 * t1 + c2 * t2"""

    def __init__(self, coef):
        self.coef = np.asarray(coef, float)

    def __call__(self, X):
        return self.coef[0] + X[:, 0] * self.coef[1] + X[:, 1] * self.coef[2]


class RotationModel:
    def __init__(self):
        self.groups = {}      # (kind, direction) -> dict(baseline=..., gps=[..3..], ranges=..., cmd=...)
        self.meta = {}
        self._table = None

    # ---------- Vorhersage ----------
    @staticmethod
    def is_single(kind):
        return kind.startswith("single")

    @staticmethod
    def _features(kind, t1, t2):
        t1 = np.atleast_1d(np.asarray(t1, float))
        t2 = np.zeros_like(t1) if kind.startswith("single") else np.broadcast_to(np.asarray(t2, float), t1.shape)
        return np.column_stack([t1, t2])

    def predict(self, kind, direction, t1, t2=0.0):
        g = self.groups[(kind, direction)]
        X = self._features(kind, t1, t2)
        mean = np.zeros((len(X), 3))
        std = np.zeros((len(X), 3))
        for j in range(3):
            mean[:, j] = g["baseline"][j](X)
            gp = g["gps"][j]
            if gp is not None:
                m, s = gp.predict(X[:, :g["dims"]], return_std=True)
                mean[:, j] += m
                std[:, j] = s
            else:
                std[:, j] = g["resid_std"][j]
        return mean, std

    # ---------- Befehlsauswahl ----------
    def candidate_table(self):
        """Alle Befehle im Datenbereich auf einem Raster, mit Vorhersage. Wird gecacht."""
        if self._table is not None:
            return self._table
        rows = []
        for (kind, direction), g in self.groups.items():
            lo1, hi1 = g["ranges"]["t1"]
            t1s = np.arange(lo1, hi1 + 1e-9, GRID_STEP)
            if self.is_single(kind):
                T1, T2 = t1s, np.zeros_like(t1s)
            else:
                lo2, hi2 = g["ranges"]["t2"]
                a, b = np.meshgrid(t1s, np.arange(lo2, hi2 + 1e-9, GRID_STEP))
                rlo, rhi = g["ranges"]["ratio"]   # nur Verhältnisse t2/t1, die in den Daten vorkamen
                ok = (b.ravel() <= rhi * a.ravel()) & (b.ravel() >= rlo * a.ravel())
                T1, T2 = a.ravel()[ok], b.ravel()[ok]
            m, s = self.predict(kind, direction, T1, T2)
            for k in range(len(T1)):
                rows.append((kind, direction, T1[k], T2[k], *m[k], *s[k]))
        dt = np.dtype([("kind", "U12"), ("direction", "U5"), ("t1", float), ("t2", float),
                       ("dx", float), ("dy", float), ("dth", float),
                       ("sx", float), ("sy", float), ("sth", float)])
        self._table = np.array(rows, dtype=dt)
        return self._table

    def best_command(self, target_deg, pos_weight=20.0, kinds=None):
        """
        Befehl, dessen vorhergesagte Drehung target_deg am besten trifft.
        Kosten = |Fehler| + σ_θ + pos_weight * (Positionsversatz in m)   [alles in Grad]
        """
        tab = self.candidate_table()
        if kinds is not None:
            tab = tab[np.isin(tab["kind"], kinds)]
        err = np.abs(tab["dth"] - target_deg)
        # Untergrenze: GP-σ ist bei wenigen Samples zu optimistisch (echter Fehler laut CV ~1.5–2.5°)
        cost = err + np.maximum(tab["sth"], SIGMA_FLOOR_DEG) + pos_weight * np.hypot(tab["dx"], tab["dy"])
        k = int(np.argmin(cost))
        r = tab[k]
        return {"kind": str(r["kind"]), "direction": str(r["direction"]),
                "t1": float(r["t1"]), "t2": float(r["t2"]),
                "dtheta": float(r["dth"]), "sigma_theta": float(r["sth"]),
                "dx_body": float(r["dx"]), "dy_body": float(r["dy"]), "error": float(err[k])}

    def segments(self, cmd):
        """Befehl -> UDP-Segmente [(cmd_xyz, dauer), ...] für execute_sequence."""
        g = self.groups[(cmd["kind"], cmd["direction"])]
        y1, y2 = g["cmd"]
        segs = [((0, 0, y1), cmd["t1"])]
        if not self.is_single(cmd["kind"]):
            if g.get("gap", 0.0) > 0:
                segs.append(((0, 0, 0), g["gap"]))
            segs.append(((0, 0, y2), cmd["t2"]))
        return segs
