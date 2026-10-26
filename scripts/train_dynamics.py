"""
train_dynamics.py
Trainiert das Primitiv-Dynamikmodell auf motion_dataset.csv.

Nutzung:
    python train_dynamics.py                          # Standard: motion_dataset.csv
    python train_dynamics.py --csv motion_dataset.csv nav_log.csv   # mehrere Dateien
    python train_dynamics.py --csv linear_dataset.csv mixed_dataset.csv rotation_dataset.csv \
                                   linear_return_log.csv mixed_return_log.csv rotation_return_log.csv
    (rotation_dataset.csv: Einzeldrehungen werden übernommen, Flicks gehören ins Drehmodell)
    python train_dynamics.py --csv daten.csv --no-plots
    python train_dynamics.py --no-gp                  # nur Physik-Baseline

Ausgabe:
    models/dynamics_model.joblib -> gespeichertes Modell (laden mit joblib.load)
    data/plots/<aktion>.png   -> Daten + Baseline + GP (±2σ) je Aktion
    Konsole                   -> Parameter (Geschwindigkeit, Totzeit, Versatz)
                                 und Kreuzvalidierung: Mittelwert vs. Baseline vs. Baseline+GP

Benötigt: numpy, pandas, scipy, scikit-learn, joblib, matplotlib
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
from dynamics_model import (TARGETS, DynamicsModel, baseline_fn,
                            world_to_body, wrap_angle)

# ==========================================
# KONFIGURATION
# ==========================================
MIN_POSE_SAMPLES = 3      # weniger Posen im Mittelungsfenster -> Sample verwerfen
MAX_STD_POS = 0.02        # m, Tracker-Streuung im Stand
MAX_STD_THETA = 3.0       # Grad
MIN_SAMPLES_FOR_GP = 15   # darunter nur Baseline + konstante Streuung
N_FOLDS = 5
UNITS = {"dx_body": "m", "dy_body": "m", "dtheta": "°"}


# ==========================================
# DATEN
# ==========================================
def rotation_csv_to_motion(r):
    """rotation_dataset.csv -> Format von motion_dataset.csv (nur Einzeldrehungen; Flicks -> Drehmodell)."""
    r = r[r["kind"].isin(["single", "single_slow"]) & (r["t1"] <= 2.6)].copy()
    suffix = np.where(r["kind"] == "single_slow", "_slow", "")
    r["command_name"] = "yaw_" + r["direction"].astype(str) + suffix
    r["command_x"] = 0
    r["command_y"] = 0
    r["command_yaw"] = r["cmd_yaw_1"]
    r["command_duration"] = r["t1"]
    return r


def unwrap_yaw(df):
    """
    Drehungen > 180° erscheinen umgewickelt (+200° -> -160°). Pro Aktion mit Drehanteil wird aus den
    unproblematischen Samples (|dθ| < 150°) eine grobe Gerade dθ(T) gefittet und jede Messung auf
    die 360°-Variante gelegt, die dieser Erwartung am nächsten ist. Unabhängig von der Vorzeichen-
    Konvention der (neu kalibrierten) Kamera.
    """
    for a, g in df.groupby("command_name"):
        if (g["command_yaw"] == 0).all():
            continue
        ok = g["dtheta"].abs() < 150
        if ok.sum() < 3:
            continue
        k, b = np.polyfit(g.loc[ok, "T"], g.loc[ok, "dtheta"], 1)
        rough = k * g["T"] + b
        df.loc[g.index, "dtheta"] = g["dtheta"] + 360.0 * np.round((rough - g["dtheta"]) / 360.0)
    return df


def load_dataset(paths):
    if isinstance(paths, str):
        paths = [paths]
    parts = []
    for p in paths:
        d = pd.read_csv(p)
        if "segments" in d.columns:      # Datei aus collect_fine.py: nur Einzelimpulse sind Einzelbefehle
            d = d[d["kind"] == "pulse"].copy()
        elif "kind" in d.columns:        # Datei aus collect_rotation.py
            d = rotation_csv_to_motion(d)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)
    

    # Tracker-Qualität
    ok = (
        (df["num_pose_samples"] >= MIN_POSE_SAMPLES)
        & (df["std_x"] <= MAX_STD_POS)
        & (df["std_y"] <= MAX_STD_POS)
        & (df["std_theta"] <= MAX_STD_THETA)
    )
    df = df[ok].copy()

    # Welt-Deltas -> Körpersystem (bezogen auf die Start-Orientierung)
    dxb, dyb = world_to_body(df["delta_x"].values, df["delta_y"].values,
                             df["start_theta"].values)
    df["dx_body"] = dxb
    df["dy_body"] = dyb
    df["dtheta"] = wrap_angle(df["delta_theta"].values)
    df["T"] = df["command_duration"].astype(float)
    df = unwrap_yaw(df)

    print(f"Samples geladen: {n0}, nach Qualitätsfilter: {len(df)} "
          f"({n0 - len(df)} verworfen)")
    print("\nSamples pro Aktion:")
    for a, g in df.groupby("command_name"):
        print(f"  {a:10s} {len(g):4d}   Dauer {g['T'].min():.2f}–{g['T'].max():.2f} s")
    return df


# ==========================================
# FITTING
# ==========================================
MIN_SAMPLES_FOR_SOFTPLUS = 15   # darunter nur Gerade (Totzeit-Fit wäre instabil)


def fit_baseline(T, y):
    """
    Fit v * softplus(T - t_delay) + b. Fallback auf eine Gerade, wenn zu wenige Samples,
    der Fit fehlschlägt oder unplausibel ist (Totzeit am Rand / hinter den meisten Daten).
    """
    if len(T) < 3:
        return (0.0, 0.0, float(np.mean(y)) if len(y) else 0.0)
    k, b = np.polyfit(T, y, 1)
    linear = (float(k), 0.0, float(b))
    if len(T) < MIN_SAMPLES_FOR_SOFTPLUS:
        return linear
    bounds = ([-np.inf, 0.0, -np.inf], [np.inf, 1.5, np.inf])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(baseline_fn, T, y, p0=[k, 0.2, 0.0], bounds=bounds, maxfev=20000)
    except Exception:
        return linear
    v, td, bb = (float(x) for x in p)
    sse_sp = np.sum((y - baseline_fn(T, v, td, bb)) ** 2)
    sse_lin = np.sum((y - (k * T + b)) ** 2)
    implausible = td > 1.45 or td > np.percentile(T, 70) or abs(v) > 5 * abs(k) + 1.0
    if implausible or sse_sp > sse_lin:
        return linear
    return (v, td, bb)


def fit_gp(T, resid):
    kernel = (ConstantKernel(1.0, (1e-4, 1e3)) * RBF(length_scale=0.5, length_scale_bounds=(0.1, 10.0))
              + WhiteKernel(noise_level=np.var(resid) + 1e-8, noise_level_bounds=(1e-8, 1e3)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
                                  random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(T.reshape(-1, 1), resid)
    return gp


def train(df, use_gp=True):
    model = DynamicsModel()
    for action, g in df.groupby("command_name"):
        T = g["T"].values
        model.duration_range[action] = (float(T.min()), float(T.max()))
        model.command[action] = tuple(int(v) for v in g[["command_x", "command_y", "command_yaw"]].iloc[0])
        for tgt in TARGETS:
            y = g[tgt].values
            p = fit_baseline(T, y)
            model.baseline[(action, tgt)] = p
            resid = y - baseline_fn(T, *p)
            model.resid_std[(action, tgt)] = float(np.std(resid)) if len(y) > 1 else 0.0
            if use_gp and len(y) >= MIN_SAMPLES_FOR_GP:
                model.gps[(action, tgt)] = fit_gp(T, resid)
            else:
                model.gps[(action, tgt)] = None
    return model


# ==========================================
# EVALUATION
# ==========================================
def cross_validate(df, use_gp=True):
    """Vergleicht: Mittelwert pro Aktion | Baseline | Baseline+GP (RMSE, 2σ-Abdeckung)."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    variants = ["mean", "baseline"] + (["baseline+gp"] if use_gp else [])
    errs = {v: [] for v in variants}
    cover = []  # Anteil Testpunkte innerhalb ±2σ (Baseline+GP)
    rows = []

    for train_idx, test_idx in kf.split(df):
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        m_base = train(tr, use_gp=False)
        m_gp = train(tr, use_gp=True) if use_gp else None
        for action, g in te.groupby("command_name"):
            if action not in m_base.command:
                continue
            y = g[TARGETS].values
            T = g["T"].values
            pred_mean = np.tile(tr[tr["command_name"] == action][TARGETS].mean().values, (len(g), 1))
            pred_base, _ = m_base.predict(action, T)
            e = {"mean": y - pred_mean, "baseline": y - pred_base}
            if use_gp:
                pred_gp, std_gp = m_gp.predict(action, T)
                e["baseline+gp"] = y - pred_gp
                cover.append(np.abs(y - pred_gp) <= 2 * std_gp)
            for v in variants:
                rows.append(pd.DataFrame(e[v], columns=TARGETS).assign(action=action, variant=v))

    res = pd.concat(rows)
    print("\n" + "=" * 70)
    print(f"KREUZVALIDIERUNG ({N_FOLDS}-fach) — RMSE im Körpersystem")
    print("=" * 70)
    rmse = res.groupby(["action", "variant"])[TARGETS].apply(lambda d: np.sqrt((d ** 2).mean()))
    rmse = rmse.unstack("variant")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        for tgt in TARGETS:
            print(f"\n{tgt} [{UNITS[tgt]}]")
            print(rmse[tgt][variants].to_string())
        total = res.groupby("variant")[TARGETS].apply(lambda d: np.sqrt((d ** 2).mean()))
        print("\nGesamt:")
        print(total.loc[variants].to_string())
    if cover:
        c = np.vstack(cover).mean(axis=0)
        print("\nAnteil Testpunkte innerhalb ±2σ (ideal ≈ 0.95): "
              + ", ".join(f"{t}={v:.2f}" for t, v in zip(TARGETS, c)))


def print_parameters(model):
    print("\n" + "=" * 70)
    print("BASELINE-PARAMETER  (Δ = v · max(0, T − t_delay) + b)")
    print("=" * 70)
    for action in model.actions:
        print(f"\n{action}  (UDP {model.command[action]})")
        for tgt in TARGETS:
            v, td, b = model.baseline[(action, tgt)]
            u = UNITS[tgt]
            print(f"  {tgt:8s}  v = {v:+8.3f} {u}/s   t_delay = {td:.2f} s   "
                  f"b = {b:+7.3f} {u}   Residuum-σ = {model.resid_std[(action, tgt)]:.3f} {u}")


# ==========================================
# PLOTS
# ==========================================
def make_plots(df, model, outdir=str(paths.PLOTS)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    for action, g in df.groupby("command_name"):
        lo, hi = model.duration_range[action]
        Tq = np.linspace(max(0.0, lo - 0.2), hi + 0.2, 200)
        mean, std = model.predict(action, Tq)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for j, tgt in enumerate(TARGETS):
            ax = axes[j]
            v, td, b = model.baseline[(action, tgt)]
            ax.fill_between(Tq, mean[:, j] - 2 * std[:, j], mean[:, j] + 2 * std[:, j],
                            color="tab:blue", alpha=0.18, label="Modell ±2σ")
            ax.plot(Tq, baseline_fn(Tq, v, td, b), "--", color="gray", lw=1.2, label="Baseline")
            ax.plot(Tq, mean[:, j], color="tab:blue", lw=2, label="Baseline + GP")
            ax.scatter(g["T"], g[tgt], s=14, color="black", alpha=0.6, label="Daten")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("Befehlsdauer T [s]")
            ax.set_ylabel(f"{tgt} [{UNITS[tgt]}]")
            ax.set_title(f"{tgt}   (t_delay={td:.2f}s, b={b:+.3f})", fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8, loc="best")
        fig.suptitle(f"{action}  —  {len(g)} Samples", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{action}.png"), dpi=120)
        plt.close(fig)
    print(f"\nPlots gespeichert in ./{outdir}/")


# ==========================================
# MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=["linear_dataset.csv"],
                    help="eine oder mehrere CSVs, z.B. linear_dataset.csv nav_log.csv "
                         "(nur der Name reicht: wird in data/datasets und data/nav_logs gesucht)")
    ap.add_argument("--out", default=str(paths.DYNAMICS_MODEL))
    ap.add_argument("--no-gp", action="store_true", help="nur Physik-Baseline")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-cv", action="store_true", help="Kreuzvalidierung überspringen")
    args = ap.parse_args()
    args.csv = [paths.find_data(c) for c in args.csv]
    use_gp = not args.no_gp

    df = load_dataset(args.csv)
    if not args.no_cv:
        cross_validate(df, use_gp=use_gp)

    model = train(df, use_gp=use_gp)
    model.meta = {"csv": [os.path.abspath(c) for c in args.csv], "n_samples": len(df), "use_gp": use_gp}
    print_parameters(model)

    joblib.dump(model, args.out)
    print(f"\nModell gespeichert: {args.out}")

    if not args.no_plots:
        make_plots(df, model)

    # Kleines Beispiel zur Nutzung
    a = model.actions[0]
    mean, std = model.predict(a, 1.0)
    print(f"\nBeispiel: {a} für 1.0 s -> "
          f"dx_body={mean[0,0]:+.3f}±{std[0,0]:.3f} m, "
          f"dy_body={mean[0,1]:+.3f}±{std[0,1]:.3f} m, "
          f"dθ={mean[0,2]:+.1f}±{std[0,2]:.1f}°")


if __name__ == "__main__":
    main()