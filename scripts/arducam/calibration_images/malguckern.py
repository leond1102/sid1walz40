import cv2
import numpy as np

# 1. Deine berechneten Werte eintragen
K = np.array([[476.44468693,   0.        , 346.85183889],
              [  0.        , 477.19751487, 256.06817105],
              [  0.        ,   0.        ,   1.        ]])

D = np.array([[-0.06578177],
              [-0.02171111],
              [ 0.06853957],
              [-0.06098964]])

# 2. Kamera starten
cap = cv2.VideoCapture(0) # Ggf. wieder auf 1 ändern, falls nötig

if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

# Wir lesen ein erstes Bild, um die Auflösung (Höhe, Breite) zu bekommen
ret, frame = cap.read()
if not ret:
    exit()
h, w = frame.shape[:2]

# 3. Entzerrungs-Map vorbereiten (Das macht die Echtzeit-Verarbeitung super schnell)
# Wir nutzen K auch als neue Kameramatrix, dadurch wird das Bild optimal zugeschnitten.
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)

print("Starte entzerrten Live-Feed... (Beenden mit 'q')")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Entzerrung auf den aktuellen Frame anwenden
    undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Wir packen Original und Entzerrt nebeneinander, damit du den Unterschied siehst!
    combined = np.hstack((frame, undistorted_img))
    
    cv2.imshow("Arducam: Original (links) vs. Entzerrt (rechts)", combined)

    # Beenden mit 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
