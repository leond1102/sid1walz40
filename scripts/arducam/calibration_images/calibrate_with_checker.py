import glob
import os
import sys

import cv2
import numpy as np

# Schachbrett-Dimensionen (Anzahl der INNEREN Ecken)
CHECKERBOARD = (15, 10)

# Ordner mit den Kalibrierbildern (aus dem Screenshot-Script), per Argument überschreibbar:
#   python calibrate.py calibration_images_4656x3496
IMAGE_DIR = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("calibration_images_*"))[-1]
DETECT_MAX_WIDTH = 1600  # Ecken zuerst auf verkleinertem Bild suchen (schnell), dann voll verfeinern

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")) + glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
print(f"Ordner: {IMAGE_DIR}, {len(images)} Bilder")

objpoints, imgpoints, used = [], [], []
image_size = None

for fname in images:
    gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"  {os.path.basename(fname)}: nicht lesbar")
        continue
    size = gray.shape[::-1]  # (Breite, Höhe)
    if image_size is None:
        image_size = size
    elif size != image_size:
        print(f"  {os.path.basename(fname)}: andere Größe {size} statt {image_size} -> übersprungen")
        continue

    # 1) Ecken auf verkleinertem Bild suchen
    scale = min(1.0, DETECT_MAX_WIDTH / size[0])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    ret, corners = cv2.findChessboardCorners(
        small, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ret:
        print(f"  {os.path.basename(fname)}: kein Schachbrett gefunden")
        continue

    # 2) Auf volle Auflösung hochrechnen und dort verfeinern.
    #    Suchfenster ~ 1/4 des Eckenabstands (skaliert mit der Auflösung)
    corners = corners / scale
    spacing = np.median(np.linalg.norm(np.diff(corners.reshape(-1, 2), axis=0), axis=1))
    win = int(max(3, min(25, spacing / 4)))
    corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (win, win), (-1, -1), subpix_criteria)

    objpoints.append(objp)
    imgpoints.append(corners.reshape(1, -1, 2).astype(np.float64))
    used.append(fname)
    print(f"  {os.path.basename(fname)}: ok (Suchfenster {win}px)")

if len(objpoints) < 10:
    sys.exit(f"Nur {len(objpoints)} brauchbare Bilder – mindestens 10, besser 20+ aufnehmen.")

calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND
                     + cv2.fisheye.CALIB_FIX_SKEW)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

# Schlechte Bilder (CALIB_CHECK_COND schlägt fehl) automatisch entfernen und neu rechnen
while True:
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, image_size, K, D, None, None, calibration_flags, criteria)
        break
    except cv2.error as e:
        msg = str(e)
        # OpenCV nennt den Index des problematischen Bildes: "... input array N"
        idx = None
        for token in msg.replace(".", " ").split():
            if token.isdigit():
                idx = int(token)
        if idx is None or idx >= len(objpoints) or len(objpoints) <= 10:
            raise
        print(f"  Bild verworfen (schlechte Kondition): {os.path.basename(used[idx])}")
        del objpoints[idx], imgpoints[idx], used[idx]

print(f"\n{len(objpoints)} Bilder verwendet, Auflösung {image_size[0]}x{image_size[1]}")
print(f"Reprojektionsfehler (RMS): {rms:.3f} px   (gut: < 0.5 px, akzeptabel: < 1 px)")

print("\n# ---- ins Ortungsscript (aruco_tracker.py) übernehmen ----")
print(f"REQUESTED_RESOLUTION = (10000, 10000)")
print(f"CALIB_RESOLUTION = ({image_size[0]}, {image_size[1]})")
print("CALIB_K = np.array(\n    [")
for row in K:
    print(f"        [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}],")
print("    ]\n)")
print(f"CALIB_D = np.array([{D[0,0]:.8f}, {D[1,0]:.8f}, {D[2,0]:.8f}, {D[3,0]:.8f}]).reshape(4, 1)")