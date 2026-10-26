"""
check_heading.py — prüft, ob der gemessene Winkel θ zur tatsächlichen Laufrichtung passt.

Idee: Bei reinem Vorwärtslauf muss sich der Roboter (fast) genau in Richtung θ bewegen,
bei Rückwärtslauf in θ+180°, bei "right"/"left" in θ±90°. Die Abweichung zwischen
Bewegungsrichtung (aus x/y) und gemessenem θ verrät, was mit der Kalibrierung los ist:

  - konstanter Versatz          -> Marker verdreht aufgeklebt / Winkel-Offset (harmlos, korrigierbar)
  - Muster mit 2 Perioden pro 360° über θ
                                -> x- und y-Achse unterschiedlich skaliert oder nicht rechtwinklig
                                   (Kalibrierung / Homographie von x,y passt nicht zu θ)
  - Muster mit 1 Periode        -> θ wird anders berechnet als x,y (z.B. im Bild statt am Boden)
  - abhängig vom Ort (x, y)     -> Verzerrung im Bild (Linsenverzerrung, Perspektive)

Nutzung:
    python check_heading.py linear_dataset.csv [weitere CSVs im motion_dataset-Format]
Ausgabe: Konsole + data/plots/heading_check.png (CSV-Namen werden in data/datasets gesucht)
"""
import sys

import numpy as np
import pandas as pd

MIN_DIST = 0.25       # m: nur deutliche Läufe
MAX_DTHETA = 6.0      # °: nur Läufe ohne nennenswerte Drehung

OFFSETS = {"forward": 0.0, "backward": 180.0}   # Soll-Richtung relativ zu θ
                                                 # (left/right siehe unten, Vorzeichen aus Daten)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def main(files):
    import paths
    df = pd.concat([pd.read_csv(paths.find_data(p)) for p in files], ignore_index=True)
    dist = np.hypot(df["delta_x"], df["delta_y"])
    dth = wrap(df["delta_theta"])
    ok = (dist > MIN_DIST) & (dth.abs() < MAX_DTHETA)
    d = df[ok & df["command_name"].isin(["forward", "backward", "left", "right"])].copy()
    d["move_dir"] = np.degrees(np.arctan2(d["delta_y"], d["delta_x"]))
    d["theta_mid"] = wrap(d["start_theta"] + wrap(d["delta_theta"]) / 2)

    # Seitlich: Vorzeichen (+90 oder -90) aus den Daten bestimmen
    for name in ("right", "left"):
        g = d[d["command_name"] == name]
        if len(g):
            cand = [90.0, -90.0]
            OFFSETS[name] = min(cand, key=lambda o: np.median(np.abs(wrap(g["move_dir"] - g["theta_mid"] - o))))

    d["err"] = wrap(d["move_dir"] - d["theta_mid"] - d["command_name"].map(OFFSETS))
    print(f"{len(d)} geeignete Läufe (> {MIN_DIST} m, < {MAX_DTHETA}° Drehung)\n")
    print("Abweichung Bewegungsrichtung vs. gemessenes θ:")
    for name, g in d.groupby("command_name"):
        print(f"  {name:9s} n={len(g):3d}   Mittel {g['err'].mean():+6.1f}°   Streuung {g['err'].std():5.1f}°"
              f"   (Soll-Offset {OFFSETS[name]:+.0f}°)")

    # Harmonische Analyse über θ: err ≈ c + a1 sin θ + b1 cos θ + a2 sin 2θ + b2 cos 2θ
    t = np.radians(d["theta_mid"].values)
    A = np.column_stack([np.ones_like(t), np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)])
    c = np.linalg.lstsq(A, d["err"].values, rcond=None)[0]
    resid = d["err"].values - A @ c
    print(f"\nZerlegung über θ:")
    print(f"  konstanter Versatz        : {c[0]:+.1f}°")
    print(f"  1 Periode pro 360°        : Amplitude {np.hypot(c[1], c[2]):.1f}°")
    print(f"  2 Perioden pro 360°       : Amplitude {np.hypot(c[3], c[4]):.1f}°")
    print(f"  Rest-Streuung             : {resid.std():.1f}°  (Gangrauschen + Messrauschen)")

    # Ortsabhängigkeit
    A2 = np.column_stack([np.ones(len(d)), d["start_x"], d["start_y"]])
    c2 = np.linalg.lstsq(A2, resid, rcond=None)[0]
    print(f"\nOrtsabhängigkeit des Rests: {c2[1]:+.1f}°/m in x, {c2[2]:+.1f}°/m in y")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import paths
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        for name, g in d.groupby("command_name"):
            ax[0].scatter(g["theta_mid"], g["err"], s=14, label=name, alpha=0.7)
        tt = np.linspace(-180, 180, 361)
        r = np.radians(tt)
        ax[0].plot(tt, c[0] + c[1] * np.sin(r) + c[2] * np.cos(r) + c[3] * np.sin(2 * r) + c[4] * np.cos(2 * r),
                   "k", lw=2, label="Fit")
        ax[0].axhline(0, color="k", lw=0.5)
        ax[0].set_xlabel("gemessenes θ [°]")
        ax[0].set_ylabel("Bewegungsrichtung − θ − Soll [°]")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        sc = ax[1].scatter(d["start_x"], d["start_y"], c=d["err"], cmap="RdBu_r", s=20, vmin=-25, vmax=25)
        fig.colorbar(sc, ax=ax[1], label="Abweichung [°]")
        ax[1].set_xlabel("x [m]")
        ax[1].set_ylabel("y [m]")
        ax[1].set_aspect("equal")
        ax[1].set_title("Abweichung nach Ort")
        fig.tight_layout()
        out = paths.PLOTS / "heading_check.png"
        fig.savefig(out, dpi=120)
        print(f"\nPlot: {out}")
    except Exception as e:
        print(f"(kein Plot: {e})")


if __name__ == "__main__":
    main(sys.argv[1:] or ["linear_dataset.csv"])
