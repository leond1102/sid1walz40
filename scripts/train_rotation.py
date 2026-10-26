"""
train_rotation.py
Trainiert das Drehmodell (rotation_model.py) auf rotation_dataset.csv.

Nutzung:
    python train_rotation.py
    python train_rotation.py --csv rotation_dataset.csv --no-plots

Ausgabe:
    models/rotation_model.joblib  -> Modell (--out für einen anderen Pfad)
    data/plots/rotation_single*.png    -> Einzeldrehungen (normal / langsam): Winkel über Dauer
    data/plots/rotation_flick*_*.png   -> Flicks (mit/ohne Pause): Winkel über (t1, t2)
    Konsole                       -> Kreuzvalidierung + Tabelle "welcher Winkel ist wie genau machbar"

Benötigt: numpy, pandas, scipy, scikit-learn, joblib, matplotlib, dynamics_model.py, rotation_model.py
"""
import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold

import paths
from dynamics_model import baseline_fn, wrap_angle
from rotation_model import TARGETS, LinearBaseline, RotationModel, SoftplusBaseline

MIN_POSE_SAMPLES = 3
MAX_STD_POS = 0.02
MAX_STD_THETA = 3.0
MIN_SAMPLES_FOR_GP = 15
N_FOLDS = 5
TABLE_TARGETS = [3, 5, 8, 10, 15, 20, 25, 30, 45, 60, 90, 120, 150, 180]
UNITS = {"dx_body": "m", "dy_body": "m", "dtheta": "°"}


# ==========================================
# DATEN
# ==========================================
def load(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    n0 = len(df)
        # Einzeldrehungen > ~160° sind nicht eindeutig messbar (±180°-Mehrdeutigkeit) -> verwerfen
    df = df[~((df["kind"] == "single") & (df["t1"] > 2.6))]
    df = df[(df["num_pose_samples"] >= MIN_POSE_SAMPLES) & (df["std_x"] <= MAX_STD_POS)
            & (df["std_y"] <= MAX_STD_POS) & (df["std_theta"] <= MAX_STD_THETA)].copy()

    if "gap" not in df.columns:
        df["gap"] = 0.0
    # Entfalten: Drehungen > 180° erscheinen umgewickelt (z.B. +200° -> -160°). Pro Gruppe wird aus
    # den unproblematischen Samples (|dθ| < 150°) eine grobe Ebene dθ(t1, t2) gefittet und jede
    # Messung auf die 360°-Variante gelegt, die ihr am nächsten ist (unabhängig von der
    # Vorzeichen-Konvention der Kamera).
    df["dtheta"] = wrap_angle(df["delta_theta"].values)
    for _, g in df.groupby(["kind", "direction"]):
        ok = g["dtheta"].abs() < 150
        if ok.sum() < 4:
            continue
        A = np.column_stack([np.ones(ok.sum()), g.loc[ok, "t1"], g.loc[ok, "t2"]])
        c = np.linalg.lstsq(A, g.loc[ok, "dtheta"], rcond=None)[0]
        rough = c[0] + c[1] * g["t1"] + c[2] * g["t2"]
        df.loc[g.index, "dtheta"] = g["dtheta"] + 360.0 * np.round((rough - g["dtheta"]) / 360.0)

    print(f"Samples: {n0}, nach Qualitätsfilter: {len(df)}")
    for (k, d), g in df.groupby(["kind", "direction"]):
        print(f"  {k:6s} {d:5s} {len(g):4d}   dθ {g['dtheta'].min():+6.1f}° … {g['dtheta'].max():+6.1f}°")
    return df


# ==========================================
# FITTING
# ==========================================
def fit_gp(X, y):
    dims = X.shape[1]
    kernel = (ConstantKernel(1.0, (1e-4, 1e3))
              * RBF(length_scale=[0.5] * dims, length_scale_bounds=(0.05, 10.0))
              + WhiteKernel(noise_level=np.var(y) + 1e-8, noise_level_bounds=(1e-8, 1e3)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)
    return gp


def fit_group(kind, g, use_gp=True):
    single = kind.startswith("single")
    X = np.column_stack([g["t1"].values, np.zeros(len(g)) if single else g["t2"].values])
    dims = 1 if single else 2
    baselines, gps, resid_std = [], [], []
    for tgt in TARGETS:
        y = g[tgt].values
        if single:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p, _ = curve_fit(baseline_fn, X[:, 0], y, p0=[np.polyfit(X[:, 0], y, 1)[0], 0.2, 0.0],
                                     bounds=([-np.inf, 0, -np.inf], [np.inf, 1.5, np.inf]), maxfev=20000)
            except Exception:
                k, b = np.polyfit(X[:, 0], y, 1)
                p = (k, 0.0, b)
            base = SoftplusBaseline(*p)
        else:
            A = np.column_stack([np.ones(len(X)), X])
            base = LinearBaseline(np.linalg.lstsq(A, y, rcond=None)[0])
        r = y - base(X)
        baselines.append(base)
        resid_std.append(float(np.std(r)))
        gps.append(fit_gp(X[:, :dims], r) if use_gp and len(y) >= MIN_SAMPLES_FOR_GP else None)
    y1 = int(g["cmd_yaw_1"].iloc[0])
    ratio = (g["t2"] / g["t1"]).values
    return {"baseline": baselines, "gps": gps, "resid_std": resid_std, "dims": dims,
            "ranges": {"t1": (float(g["t1"].min()), float(g["t1"].max())),
                       "t2": (float(g["t2"].min()), float(g["t2"].max())),
                       "ratio": (float(ratio.min()), float(ratio.max()))},
            "gap": float(g["gap"].median()),
            "cmd": (y1, -y1)}


def train(df, use_gp=True):
    m = RotationModel()
    for (kind, direction), g in df.groupby(["kind", "direction"]):
        m.groups[(kind, direction)] = fit_group(kind, g, use_gp)
    return m


# ==========================================
# EVALUATION
# ==========================================
def cross_validate(df):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    rows = []
    for tr, te in kf.split(df):
        m = train(df.iloc[tr])
        for (kind, direction), g in df.iloc[te].groupby(["kind", "direction"]):
            if (kind, direction) not in m.groups:
                continue
            mean, std = m.predict(kind, direction, g["t1"].values, g["t2"].values)
            err = g[TARGETS].values - mean
            rows.append(pd.DataFrame({
                "kind": kind, "direction": direction,
                **{f"rmse_{t}": err[:, j] ** 2 for j, t in enumerate(TARGETS)},
                "in2sigma": (np.abs(err[:, 2]) <= 2 * std[:, 2]).astype(float)}))
    res = pd.concat(rows).groupby(["kind", "direction"]).mean()
    for t in TARGETS:
        res[f"rmse_{t}"] = np.sqrt(res[f"rmse_{t}"])
    print("\n" + "=" * 70 + f"\nKREUZVALIDIERUNG ({N_FOLDS}-fach)\n" + "=" * 70)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(res.rename(columns={"rmse_dx_body": "dx_body [m]", "rmse_dy_body": "dy_body [m]",
                                  "rmse_dtheta": "dθ [°]", "in2sigma": "dθ in ±2σ"}).to_string())


def achievability_table(m):
    print("\n" + "=" * 70)
    print("MACHBARKEIT: bester Befehl je Zielwinkel (nur im Datenbereich)")
    print("=" * 70)
    print(f"{'Ziel':>6}  {'Befehl':<34} {'erw. dθ':>8} {'Fehler':>7} {'σ':>6} {'Versatz':>8}")
    for tgt in TABLE_TARGETS:
        for s in (+1, -1):
            c = m.best_command(s * tgt)
            desc = f"{c['kind']} {c['direction']} {c['t1']:.2f}s"
            if not c["kind"].startswith("single"):
                desc += f" / {c['t2']:.2f}s"
            print(f"{s * tgt:+6d}° {desc:<34} {c['dtheta']:+7.1f}° {c['error']:6.1f}° {c['sigma_theta']:5.1f}° "
                  f"{np.hypot(c['dx_body'], c['dy_body']) * 100:6.1f} cm")
    # Vergleich: wie gut wären kleine Winkel NUR mit Einzeldrehungen?
    singles = tuple(k for (k, _) in m.groups if k.startswith("single"))
    if not singles:
        return
    print("\nKleine Winkel nur mit Einzeldrehungen (zum Vergleich):")
    for tgt in (3, 5, 10, 15, 20):
        for s in (+1, -1):
            c = m.best_command(s * tgt, kinds=singles)
            print(f"{s * tgt:+6d}°  {c['kind']} {c['direction']} {c['t1']:.2f}s  -> erw. {c['dtheta']:+.1f}° "
                  f"(Fehler {c['error']:.1f}°, σ {c['sigma_theta']:.1f}°)")


# ==========================================
# PLOTS
# ==========================================
def make_plots(df, m, outdir=str(paths.PLOTS)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    for skind in ("single", "single_slow"):
      singles = df[df["kind"] == skind]
      if len(singles):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for d, col in (("left", "tab:blue"), ("right", "tab:orange")):
            g = singles[singles["direction"] == d]
            if (skind, d) not in m.groups or not len(g):
                continue
            lo, hi = m.groups[(skind, d)]["ranges"]["t1"]
            T = np.linspace(lo, hi, 200)
            mean, std = m.predict(skind, d, T)
            for j, tgt in enumerate(TARGETS):
                ax = axes[j]
                ax.fill_between(T, mean[:, j] - 2 * std[:, j], mean[:, j] + 2 * std[:, j], color=col, alpha=0.15)
                ax.plot(T, mean[:, j], color=col, lw=2, label=f"yaw {d}")
                ax.scatter(g["t1"], g[tgt], s=10, color=col, alpha=0.5)
        for j, tgt in enumerate(TARGETS):
            axes[j].set_xlabel("Dauer t1 [s]")
            axes[j].set_ylabel(f"{tgt} [{UNITS[tgt]}]")
            axes[j].grid(alpha=0.3)
            axes[j].axhline(0, color="k", lw=0.5)
        axes[2].legend()
        fig.suptitle(f"Einzeldrehungen {skind} (±2σ)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"rotation_{skind}.png"), dpi=120)
        plt.close(fig)

    for fkind, d in [(k, d) for k in ("flick", "flick_pause") for d in ("left", "right")]:
        g = df[(df["kind"] == fkind) & (df["direction"] == d)]
        if (fkind, d) not in m.groups or not len(g):
            continue
        r = m.groups[(fkind, d)]["ranges"]
        a, b = np.meshgrid(np.linspace(*r["t1"], 60), np.linspace(*r["t2"], 60))
        mean, std = m.predict(fkind, d, a.ravel(), b.ravel())
        mask = (b <= r["ratio"][1] * a) & (b >= r["ratio"][0] * a)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for ax, vals, title in ((axes[0], mean[:, 2], "erw. dθ [°]"), (axes[1], std[:, 2], "σ dθ [°]")):
            Z = np.where(mask, vals.reshape(a.shape), np.nan)
            cs = ax.contourf(a, b, Z, levels=20, cmap="RdBu_r" if "erw" in title else "viridis")
            fig.colorbar(cs, ax=ax)
            if "erw" in title:
                ax.contour(a, b, Z, levels=[-30, -15, -5, 0, 5, 15, 30], colors="k", linewidths=0.6)
                ax.scatter(g["t1"], g["t2"], c=g["dtheta"], cmap="RdBu_r", edgecolors="k", s=18,
                           vmin=np.nanmin(Z), vmax=np.nanmax(Z))
            ax.set_xlabel("t1: erste Drehung [s]")
            ax.set_ylabel("t2: Gegendrehung [s]")
            ax.set_title(title)
        fig.suptitle(f"{fkind} {d}: erst {d}, dann zurück", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"rotation_{fkind}_{d}.png"), dpi=120)
        plt.close(fig)
    print(f"\nPlots gespeichert in ./{outdir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=["rotation_dataset.csv"])
    ap.add_argument("--out", default=str(paths.ROTATION_MODEL))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-cv", action="store_true")
    args = ap.parse_args()
    args.csv = [paths.find_data(c) for c in args.csv]   # nur der Name reicht (data/datasets, data/nav_logs)

    df = load(args.csv)
    if not args.no_cv:
        cross_validate(df)
    m = train(df)
    m.meta = {"csv": [os.path.abspath(c) for c in args.csv], "n_samples": len(df)}
    achievability_table(m)
    joblib.dump(m, args.out)
    print(f"\nModell gespeichert: {args.out}")
    if not args.no_plots:
        make_plots(df, m)


if __name__ == "__main__":
    main()
