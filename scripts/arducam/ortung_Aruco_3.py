#!/usr/bin/env python3
"""ArUco-Tracking des Roboters mit der Arducam, Pose-Ausgabe über MQTT.

Bestimmt Position und Orientierung des Roboters (ID 0) in einem 2D-
Koordinatensystem, das von drei Referenzmarkern aufgespannt wird:

    ID 1: Ursprung
    ID 2: liegt auf der +X-Achse
    ID 3: liegt auf der -Y-Achse

Ablauf pro Bild:
    1. Ein Lesethread holt fortlaufend das neueste MJPG-Bild der Kamera.
       Ältere Bilder werden verworfen, ohne sie zu dekodieren.
    2. Das Bild wird direkt als Graubild in voller Auflösung dekodiert.
    3. ArUco-Erkennung auf dem Originalbild. Solange der Roboter verfolgt
       wird, nur in kleinen Suchfenstern um die zuletzt gesehenen Marker,
       sonst im ganzen Bild.
    4. Nur die gefundenen Markerecken werden entzerrt (Fisheye-Kalibrierung),
       nicht das ganze Bild.
    5. Referenz-KS einlernen bzw. auf Verschiebung prüfen, Roboterpose
       berechnen und per MQTT senden.

MQTT (Broker siehe Stations-Doku):
    station/tracker/robot/pose  JSON, nicht retained, pro Bild mit gültiger Pose:
                                {"seq": 812, "t": 1758612345.123,
                                 "x": 0.4123, "y": -0.1034, "theta": 12.34}
                                x, y in m, theta in Grad (0 = +X, positiv in
                                Richtung +Y), t = Aufnahmezeit (Unix-Zeit in s)
    station/tracker/status      retained: teaching, tracking, lost, offline
    station/tracker/cmd         "reteach": Referenz-KS neu einlernen

Sicherheit:
    Eine Pose wird nur gesendet, wenn das Referenz-KS gültig ist und der
    Roboter im aktuellen Bild erkannt wurde. Alte Posen werden nie wiederholt.
    Verschieben sich die Referenzmarker (Kamera bewegt), wird das KS sofort
    verworfen und neu eingelernt. Der Empfänger muss den Roboter trotzdem
    selbst anhalten, wenn für kurze Zeit (z. B. 0,5 s) keine Pose ankommt.

Aufruf:
    python aruco_tracker.py              mit Debug-Fenster ('q'/ESC Ende, 'r' neu einlernen)
    python aruco_tracker.py --headless   ohne Fenster, z. B. als Dienst

Voraussetzungen: OpenCV >= 4.7, paho-mqtt >= 2.0.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from loguru import logger

# =============================================================================
# Konfiguration
# =============================================================================

# --- MQTT ---
MQTT_HOST = "192.168.2.1"
MQTT_PORT = 1883
MQTT_CLIENT_ID = "station_tracker"  # muss am Broker eindeutig sein
MQTT_KEEPALIVE_S = 5  # Last Will "offline" greift nach ca. 1,5 x Keepalive
TOPIC_POSE = "station/tracker/robot/pose"
TOPIC_STATUS = "station/tracker/status"
TOPIC_CMD = "station/tracker/cmd"

# --- Kamera ---
CAMERA_NAME_KEYWORD = "arducam"  # Teilstring im USB-Gerätenamen (Groß-/Kleinschreibung egal)
REQUESTED_RESOLUTION = (10000, 10000)  # absichtlich zu groß -> Kamera wählt ihr Maximum
CAMERA_FRAME_TIMEOUT_S = 2.0  # so lange kein neues Bild -> Abbruch
CAMERA_MAX_READ_FAILURES = 10  # so viele Lesefehler in Folge -> Abbruch

# --- Kamerakalibrierung (Fisheye-Modell, cv2.fisheye) ---
# K wurde bei CALIB_RESOLUTION bestimmt und wird für die tatsächliche Auflösung
# linear skaliert. D gilt in normierten Koordinaten und bleibt unverändert.
# Das ist nur korrekt, wenn die Kamera bei beiden Auflösungen denselben
# Bildausschnitt liefert (kein Crop). Am sichersten: bei der Tracking-Auflösung
# kalibrieren und diese hier eintragen.
CALIB_RESOLUTION = (640, 480)  # (Breite, Höhe) der Kalibrierbilder -> ggf. anpassen!
CALIB_K = np.array(
    [
        [476.44468693, 0.0, 346.85183889],
        [0.0, 477.19751487, 256.06817105],
        [0.0, 0.0, 1.0],
    ]
)
CALIB_D = np.array([-0.06578177, -0.02171111, 0.06853957, -0.06098964]).reshape(4, 1)

# --- ArUco ---
ARUCO_DICT = cv2.aruco.DICT_4X4_50  # muss zum gedruckten Marker-Dictionary passen!
MARKER_LENGTH_M = 0.10  # Kantenlänge des schwarzen Quadrats (ohne weißen Rand)

ID_ROBOT = 0
ID_ORIGIN = 1
ID_POS_X = 2
ID_NEG_Y = 3
REFERENCE_IDS = (ID_ORIGIN, ID_POS_X, ID_NEG_Y)
TRACKED_IDS = (ID_ROBOT, *REFERENCE_IDS)  # alle anderen IDs werden ignoriert

# Kleinste Markerkante in px, die bei der Suche im ganzen Bild noch gefunden
# wird. Der OpenCV-Standard liegt bei ca. 0,75 % der Bildbreite (bei 4000 px
# also 30 px) und verwirft weit entfernte Marker trotz hoher Auflösung.
# Kleiner = mehr Reichweite, aber langsamere Suche im ganzen Bild.
MIN_MARKER_SIDE_PX = 15

# --- Suchfenster ---
ROI_MARGIN_FACTOR = 1.5  # Rand um den zuletzt gesehenen Marker, in Markergrößen
ROI_MIN_MARGIN_PX = 64
FULL_SCAN_INTERVAL_S = 2.0  # fehlt ein Referenzmarker, spätestens so oft im ganzen Bild suchen

# --- Referenz-Koordinatensystem ---
REFERENCE_TEACH_FRAMES = 30  # Bilder, über die beim Einlernen gemittelt wird
REFERENCE_MAX_RMS_PX = 5.0  # Ebene mit größerem Restfehler wird nicht übernommen
REFERENCE_DRIFT_PX = 8.0  # mittlere Eckenabweichung in px (volle Auflösung) ...
REFERENCE_DRIFT_FRAMES = 5  # ... so viele Bilder in Folge -> KS verwerfen, neu einlernen

# Optional: gemessene Abstände Markermitte ID1 -> ID2 und ID1 -> ID3 in m.
# None = aus dem Bild schätzen. Eingetragene Werte machen die Ebene deutlich genauer.
REFERENCE_DISTANCE_X_M: float | None = None
REFERENCE_DISTANCE_Y_M: float | None = None

# Brennweite aus der Referenzebene schätzen. Mit kalibrierter Kamera nicht nötig
# (die kalibrierte Brennweite ist genauer), daher aus.
ESTIMATE_FOCAL_LENGTH = False
FOCAL_MIN_TILT_DEG = 3.0

# --- Roboter ---
# Höhe der Markermitte über der Referenzebene in m. None = automatisch schätzen.
ROBOT_MARKER_HEIGHT_M: float | None = None
ROBOT_HEIGHT_SMOOTHING_ALPHA = 0.05

# Welche Marker-Achse zeigt am Roboter nach vorn?
#   0.0  -> Marker-X-Achse (von linker zu rechter Markerkante)
#   90.0 -> obere Markerkante zeigt nach vorn
ROBOT_HEADING_OFFSET_DEG = 0.0

ROBOT_SMOOTHING_ALPHA = 0.5  # 0 < alpha <= 1, 1 = keine Glättung
ROBOT_RESET_DISTANCE_M = 0.10  # größerer Sprung -> Pose direkt übernehmen
ROBOT_LOST_TIMEOUT_S = 0.5  # so lange nicht erkannt -> Status "lost", Glättung zurücksetzen

# --- Anzeige und Log ---
WINDOW_NAME = "Arducam ArUco Tracking"
DISPLAY_MAX_WIDTH = 1600  # px, Erkennung läuft immer auf voller Auflösung
DISPLAY_MAX_FPS = 15.0  # Anzeige kostet Rechenzeit, daher begrenzt
ROBOT_ARROW_LENGTH_M = 0.15
STATS_INTERVAL_S = 10.0  # so oft eine Statuszeile ins Log

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_X = (0, 0, 255)  # BGR
COLOR_Y = (0, 200, 0)
COLOR_MARKER = (255, 200, 0)
COLOR_ROBOT = (255, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_WARN = (0, 200, 255)

_last_warning: dict[str, float] = {}


def warn_throttled(message: str, interval_s: float = 5.0) -> None:
    """Loggt eine Warnung höchstens alle `interval_s` Sekunden."""
    now = time.monotonic()
    if now - _last_warning.get(message, -math.inf) >= interval_s:
        _last_warning[message] = now
        logger.warning(message)


# =============================================================================
# Kamera
# =============================================================================


@dataclass(frozen=True)
class VideoDevice:
    """Ein V4L2-Videogerät unter /dev/videoN.

    Attributes:
        index: N in /dev/videoN.
        name: Gerätename laut Treiber.
        is_capture_node: False für Metadaten-Knoten (liefern keine Bilder).
    """

    index: int
    name: str
    is_capture_node: bool


def list_video_devices() -> list[VideoDevice]:
    """Listet alle V4L2-Videogeräte mit ihrem Namen aus sysfs, sortiert nach Index."""
    devices = []
    for sys_dir in Path("/sys/class/video4linux").glob("video*"):
        try:
            index = int(sys_dir.name.removeprefix("video"))
            name = (sys_dir / "name").read_text().strip()
            node_file = sys_dir / "index"
            node = int(node_file.read_text()) if node_file.exists() else 0
        except (OSError, ValueError):
            continue
        devices.append(VideoDevice(index, name, node == 0))
    return sorted(devices, key=lambda d: d.index)


def find_camera_index(keyword: str = CAMERA_NAME_KEYWORD) -> int | None:
    """Sucht die Kamera, deren Gerätename `keyword` enthält.

    Zuerst über die stabilen Symlinks unter /dev/v4l/by-id, danach über die
    Gerätenamen in sysfs. Unabhängig von der Einsteckreihenfolge.

    Args:
        keyword: Gesuchter Teilstring, Groß-/Kleinschreibung egal.

    Returns:
        Index N von /dev/videoN oder None.
    """
    keyword = keyword.lower()
    by_id = Path("/dev/v4l/by-id")
    if by_id.is_dir():
        for link in sorted(by_id.iterdir()):
            if keyword in link.name.lower() and link.name.endswith("video-index0"):
                target = link.resolve().name
                logger.info(f"Kamera gefunden: {link.name} -> /dev/{target}")
                return int(target.removeprefix("video"))
    for dev in list_video_devices():
        if dev.is_capture_node and keyword in dev.name.lower():
            logger.info(f"Kamera gefunden: '{dev.name}' -> /dev/video{dev.index}")
            return dev.index
    return None


def open_camera(index: int) -> cv2.VideoCapture:
    """Öffnet /dev/video{index} mit maximaler Auflösung im MJPG-Format.

    Liefert die Kamera MJPG, wird die Umwandlung nach BGR abgeschaltet. Das
    Bild kommt dann als komprimierte JPEG-Daten und wird später direkt als
    Graubild dekodiert (spart Farbumrechnung und Kopien).

    Args:
        index: Index N von /dev/videoN.

    Returns:
        Geöffnetes VideoCapture-Objekt.

    Raises:
        RuntimeError: Wenn die Kamera nicht geöffnet werden kann.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera /dev/video{index} konnte nicht geöffnet werden.")

    # MJPG vor der Auflösung setzen: unkomprimiert schafft USB bei hoher Auflösung kaum FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("ascii", "replace")
    if fourcc == "MJPG":
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    else:
        logger.warning(f"Kamera liefert {fourcc} statt MJPG, bei hoher Auflösung evtl. nur wenige FPS.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Kamera /dev/video{index} geöffnet: {width} x {height}, {fourcc}, {fps:.0f} FPS")
    return cap


def decode_gray(frame: np.ndarray) -> np.ndarray | None:
    """Wandelt ein Kamerabild in ein Graubild um.

    Args:
        frame: JPEG-Rohdaten (Form (1, N)), BGR-Bild oder Graubild.

    Returns:
        Graubild oder None, falls das Bild defekt ist.
    """
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 1 or 1 in frame.shape:
        return cv2.imdecode(frame.reshape(-1), cv2.IMREAD_GRAYSCALE)
    return frame


class CameraReader:
    """Liest die Kamera in einem eigenen Thread und hält nur das neueste Bild.

    Verarbeitung und USB-Übertragung laufen dadurch parallel, und die
    Verarbeitung bekommt immer das aktuellste Bild (geringe Latenz).
    """

    def __init__(self, index: int) -> None:
        """Öffnet die Kamera und startet den Lesethread.

        Args:
            index: Index N von /dev/videoN.
        """
        self._cap = open_camera(index)
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        failures = 0
        while self._running:
            ok, frame = self._cap.read()
            stamp = time.time()
            if not ok or frame is None or frame.size == 0:
                failures += 1
                if failures >= CAMERA_MAX_READ_FAILURES:
                    logger.error(f"{failures} Lesefehler in Folge, Kamera wird aufgegeben.")
                    break
                time.sleep(0.05)
                continue
            failures = 0
            with self._cond:
                self._frame, self._stamp = frame, stamp
                self._cond.notify()
        with self._cond:
            self._running = False
            self._cond.notify()

    def read(self, timeout: float) -> tuple[np.ndarray, float] | None:
        """Wartet auf ein neues Bild. Jedes Bild wird nur einmal ausgegeben.

        Args:
            timeout: Maximale Wartezeit in s.

        Returns:
            (Rohbild, Aufnahmezeit als Unix-Zeit) oder None bei Timeout oder
            wenn die Kamera ausgefallen ist.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._frame is not None or not self._running, timeout)
            frame, stamp = self._frame, self._stamp
            self._frame = None
        return None if frame is None else (frame, stamp)

    def close(self) -> None:
        """Beendet den Lesethread und gibt die Kamera frei."""
        self._running = False
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logger.warning("Kamerathread reagiert nicht, Kamera wird trotzdem freigegeben.")
        self._cap.release()


def build_camera_matrix(width: int, height: int) -> np.ndarray:
    """Kalibrierte Kameramatrix, skaliert auf die tatsächliche Bildgröße.

    Die entzerrten Bildpunkte verwenden dieselbe Matrix (wie im
    Kalibrierscript: neue Kameramatrix = K).

    Args:
        width: Bildbreite in px.
        height: Bildhöhe in px.

    Returns:
        3x3-Kameramatrix.
    """
    calib_w, calib_h = CALIB_RESOLUTION
    sx, sy = width / calib_w, height / calib_h
    if abs(sx - sy) > 0.01 * max(sx, sy):
        logger.warning(
            f"Seitenverhältnis {width}x{height} passt nicht zur Kalibrierung {calib_w}x{calib_h}. "
            "Vermutlich anderer Bildausschnitt -> Kalibrierung ungültig, bitte bei dieser Auflösung kalibrieren."
        )
    elif (width, height) != (calib_w, calib_h):
        logger.info(f"Kalibrierung {calib_w}x{calib_h} wird auf {width}x{height} skaliert (Faktor {sx:.3f}).")
    k = CALIB_K.astype(np.float64).copy()
    k[0, 0] *= sx
    k[1, 1] *= sy
    # Pixelmitten-Konvention: (c + 0.5) skaliert, dann wieder - 0.5
    k[0, 2] = (k[0, 2] + 0.5) * sx - 0.5
    k[1, 2] = (k[1, 2] + 0.5) * sy - 0.5
    return k


_UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)


def undistort_points(points: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """Entzerrt Bildpunkte mit dem Fisheye-Modell.

    Ergebnis in Pixeln des entzerrten Bildes mit derselben Kameramatrix.

    Args:
        points: Punkte im Originalbild, Form (N, 2).
        camera_matrix: Kameramatrix aus build_camera_matrix.

    Returns:
        Entzerrte Punkte, Form (N, 2).
    """
    result = cv2.fisheye.undistortPoints(
        points.reshape(-1, 1, 2).astype(np.float64),
        camera_matrix,
        CALIB_D,
        R=np.eye(3),
        P=camera_matrix,
        criteria=_UNDISTORT_CRITERIA,
    )
    return result.reshape(-1, 2).astype(np.float64)


# =============================================================================
# ArUco-Erkennung
# =============================================================================

_HALF = MARKER_LENGTH_M / 2
# Eckenreihenfolge wie von SOLVEPNP_IPPE_SQUARE gefordert (= ArUco-Reihenfolge:
# oben links, oben rechts, unten rechts, unten links)
MARKER_OBJECT_POINTS = np.array(
    [[-_HALF, _HALF, 0.0], [_HALF, _HALF, 0.0], [_HALF, -_HALF, 0.0], [-_HALF, -_HALF, 0.0]]
)

Roi = tuple[int, int, int, int]  # x0, y0, x1, y1 in px


def _detector_parameters(min_perimeter_rate: float | None) -> cv2.aruco.DetectorParameters:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if min_perimeter_rate is not None:
        params.minMarkerPerimeterRate = min_perimeter_rate
    return params


class MarkerDetector:
    """ArUco-Erkennung im ganzen Bild oder in Suchfenstern."""

    def __init__(self, image_size: tuple[int, int]) -> None:
        """Erstellt die Detektoren.

        Args:
            image_size: (Breite, Höhe) des Kamerabildes in px.
        """
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        full_rate = 4 * MIN_MARKER_SIDE_PX / max(image_size)
        self._full = cv2.aruco.ArucoDetector(dictionary, _detector_parameters(full_rate))
        # In den kleinen Suchfenstern passt der Standard (Mindestgröße relativ zum Fenster)
        self._roi = cv2.aruco.ArucoDetector(dictionary, _detector_parameters(None))

    def detect(self, gray: np.ndarray, rois: list[Roi] | None = None) -> dict[int, np.ndarray]:
        """Sucht die Marker aus TRACKED_IDS.

        Args:
            gray: Graubild in voller Auflösung.
            rois: Suchfenster oder None für das ganze Bild.

        Returns:
            Ecken je Marker-ID im Originalbild, Form (4, 2).
        """
        if rois is None:
            return self._detect(self._full, gray, 0, 0)
        found: dict[int, np.ndarray] = {}
        for x0, y0, x1, y1 in rois:
            found.update(self._detect(self._roi, gray[y0:y1, x0:x1], x0, y0))
        return found

    @staticmethod
    def _detect(detector: cv2.aruco.ArucoDetector, gray: np.ndarray, x0: int, y0: int) -> dict[int, np.ndarray]:
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            return {}
        ids = ids.ravel()
        found = {}
        for marker_corners, marker_id in zip(corners, ids):
            marker_id = int(marker_id)
            if marker_id not in TRACKED_IDS:
                continue
            if np.count_nonzero(ids == marker_id) > 1:
                warn_throttled(f"Marker-ID {marker_id} mehrfach im Bild, wird ignoriert.")
                continue
            found[marker_id] = marker_corners.reshape(4, 2).astype(np.float64) + (x0, y0)
        return found


def roi_around(corners: np.ndarray, image_size: tuple[int, int]) -> Roi:
    """Suchfenster um einen zuletzt gesehenen Marker.

    Args:
        corners: Markerecken im Originalbild, Form (4, 2).
        image_size: (Breite, Höhe) des Bildes.

    Returns:
        (x0, y0, x1, y1) in px, auf das Bild begrenzt.
    """
    (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
    margin = max(ROI_MIN_MARGIN_PX, ROI_MARGIN_FACTOR * max(x1 - x0, y1 - y0))
    w, h = image_size
    return (
        max(0, int(x0 - margin)),
        max(0, int(y0 - margin)),
        min(w, int(x1 + margin) + 1),
        min(h, int(y1 + margin) + 1),
    )


def marker_center_camera(corners: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray | None:
    """Schätzt die Markermitte im Kamera-KS (m) aus einem einzelnen Marker.

    Args:
        corners: Ecken im entzerrten Bild, Form (4, 2).
        camera_matrix: Kameramatrix des entzerrten Bildes.

    Returns:
        Markermitte, Form (3,), oder None.
    """
    try:
        ok, _, tvec = cv2.solvePnP(
            MARKER_OBJECT_POINTS, corners, camera_matrix, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
    except cv2.error:
        return None
    return tvec.reshape(3) if ok else None


# =============================================================================
# Arbeitsraum-Koordinatensystem
# =============================================================================


@dataclass
class WorkspaceFrame:
    """2D-Arbeitsraum-KS, aufgespannt von den Referenzmarkern ID 1, 2 und 3.

    Die X-Achse zeigt exakt von ID 1 nach ID 2, die Y-Achse steht senkrecht
    darauf und zeigt von ID 3 weg. Die Referenzebene ist z = 0.

    Attributes:
        origin: Ursprung (Mitte von ID 1) im Kamera-KS in m.
        rotation: 3x3-Matrix, Spalten = Achsen x, y, z im Kamera-KS.
        reference_points: Mittelpunkte der Referenzmarker in Weltkoordinaten (m).
        axis_angle_deg: Winkel zwischen ID1->ID2 und ID3->ID1, ideal 90°.
        camera_matrix: Kameramatrix des entzerrten Bildes zu dieser Schätzung.
        focal_estimated: True, wenn die Brennweite aus der Ebene geschätzt wurde.
        reprojection_rms_px: Mittlerer Fehler der 12 Referenzecken in px.
    """

    origin: np.ndarray
    rotation: np.ndarray
    reference_points: dict[int, np.ndarray]
    axis_angle_deg: float
    camera_matrix: np.ndarray
    focal_estimated: bool
    reprojection_rms_px: float
    camera_position: np.ndarray = field(init=False, repr=False)
    _ray_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.camera_position = -self.rotation.T @ self.origin
        self._ray_matrix = np.linalg.inv(self.camera_matrix).T @ self.rotation

    def point_to_world(self, p_cam: np.ndarray) -> np.ndarray:
        """Rechnet einen Punkt vom Kamera-KS ins Arbeitsraum-KS um."""
        return self.rotation.T @ (p_cam - self.origin)

    def intersect_pixels(self, pixels: np.ndarray, z_plane: float) -> np.ndarray | None:
        """Schneidet die Sehstrahlen durch Bildpunkte mit der Ebene z = z_plane.

        Args:
            pixels: Punkte im entzerrten Bild (volle Auflösung), Form (N, 2).
            z_plane: Höhe der Schnittebene im Arbeitsraum-KS in m.

        Returns:
            Schnittpunkte im Arbeitsraum-KS, Form (N, 3), oder None, falls ein
            Strahl die Ebene nicht vor der Kamera trifft.
        """
        rays = np.column_stack([pixels, np.ones(len(pixels))]) @ self._ray_matrix
        if np.any(np.abs(rays[:, 2]) < 1e-9):
            return None
        s = (z_plane - self.camera_position[2]) / rays[:, 2]
        if np.any(s <= 0):
            return None
        return self.camera_position + s[:, None] * rays

    def project(self, points_world, camera_matrix: np.ndarray) -> np.ndarray:
        """Projiziert Punkte aus dem Arbeitsraum-KS ins Bild.

        Args:
            points_world: Punkte (x, y, z) in m.
            camera_matrix: Kameramatrix des Zielbildes.

        Returns:
            Bildkoordinaten, Form (N, 2). NaN für Punkte hinter der Kamera.
        """
        pts_cam = np.asarray(points_world, dtype=np.float64).reshape(-1, 3) @ self.rotation.T + self.origin
        uvw = pts_cam @ camera_matrix.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uvw[:, :2] / uvw[:, 2:3]
        uv[pts_cam[:, 2] <= 0] = np.nan
        return uv


# --- Referenzebene aus den 12 Markerecken schätzen ---

_SQUARE = MARKER_OBJECT_POINTS[:, :2]  # Markerecken im Marker-KS (m), Mittelpunkt = 0


def _least_squares(fun: Callable[[np.ndarray], np.ndarray], p0: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Minimiert |fun(p)|^2 mit Levenberg-Marquardt (numerische Jacobi-Matrix)."""
    p = p0.astype(np.float64).copy()
    r = fun(p)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(max_iter):
        jac = np.empty((r.size, p.size))
        for j in range(p.size):
            dp = 1e-7 * max(1.0, abs(p[j]))
            p_step = p.copy()
            p_step[j] += dp
            jac[:, j] = (fun(p_step) - r) / dp
        jtj = jac.T @ jac
        grad = jac.T @ r
        damping = np.diag(np.diag(jtj)) + 1e-12 * np.eye(p.size)
        while lam < 1e10:
            step = np.linalg.lstsq(jtj + lam * damping, -grad, rcond=None)[0]
            r_new = fun(p + step)
            cost_new = float(r_new @ r_new)
            if np.isfinite(cost_new) and cost_new < cost:
                break
            lam *= 10.0
        else:
            break  # keine Verbesserung mehr möglich
        p += step
        converged = cost - cost_new < 1e-12 * cost
        r, cost = r_new, cost_new
        lam = max(lam / 10.0, 1e-9)
        if converged:
            break
    return p


def _plane_residuals(
    params: np.ndarray,
    observed: list[np.ndarray],
    distances: tuple[float | None, float | None],
    distance_weight: float,
) -> np.ndarray:
    """Residuen des Modells "drei gleich ausgerichtete Quadrate in einer Ebene".

    params = 8 Homographie-Einträge (h33 = 1), Mitte ID2 (2), Mitte ID3 (2),
    jeweils im Marker-KS von ID1, abgebildet in normierte Bildkoordinaten.
    """
    homography = np.append(params[:8], 1.0).reshape(3, 3)
    centers = (np.zeros(2), params[8:10], params[10:12])
    residuals = []
    for center, obs in zip(centers, observed):
        projected = np.column_stack([_SQUARE + center, np.ones(4)]) @ homography.T
        residuals.append((projected[:, :2] / projected[:, 2:3] - obs).ravel())
    for center, distance in zip(centers[1:], distances):
        if distance is not None:
            residuals.append(np.array([distance_weight * (np.linalg.norm(center) - distance)]))
    return np.concatenate(residuals)


def _estimate_focal_length(homography_px: np.ndarray, cx: float, cy: float) -> float | None:
    """Schätzt die Brennweite aus einer Ebenen-Homographie (Hauptpunkt bekannt).

    Nutzt, dass die beiden Ebenenachsen im Raum senkrecht und gleich lang sind.
    """
    g = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]]) @ homography_px
    h1, h2 = g[:, 0], g[:, 1]
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    inv_f_squared = -(a @ b) / (a @ a)
    return float(1.0 / math.sqrt(inv_f_squared)) if inv_f_squared > 0 else None


def _plane_tilt_deg(homography_px: np.ndarray, camera_matrix: np.ndarray) -> float:
    """Winkel zwischen Blickrichtung der Kamera und Ebenennormale."""
    k_inv = np.linalg.inv(camera_matrix)
    normal = np.cross(k_inv @ homography_px[:, 0], k_inv @ homography_px[:, 1])
    return math.degrees(math.acos(min(1.0, abs(normal[2]) / np.linalg.norm(normal))))


def estimate_workspace_frame(corners: dict[int, np.ndarray], guess_matrix: np.ndarray) -> WorkspaceFrame | None:
    """Schätzt das Arbeitsraum-KS aus den (gemittelten) Ecken von ID 1-3.

    1. Homographie Ebene -> Bild aus allen 12 Ecken, mit den Bedingungen
       "10-cm-Quadrate, gleich ausgerichtet, eine Ebene".
    2. Brennweite aus der Homographie (nur falls ESTIMATE_FOCAL_LENGTH).
    3. Kamerapose relativ zur Ebene aus den 12 Ecken (solvePnP).

    Args:
        corners: Ecken je Marker-ID im entzerrten Bild, müssen ID 1-3 enthalten.
        guess_matrix: Kameramatrix aus build_camera_matrix.

    Returns:
        Das Koordinatensystem oder None bei unbrauchbarer Markeranordnung.
    """
    cx, cy = float(guess_matrix[0, 2]), float(guess_matrix[1, 2])
    scale = cx  # Bildkoordinaten auf ca. [-1, 1] normieren (numerisch stabiler)
    normalize = np.array([[1 / scale, 0.0, -cx / scale], [0.0, 1 / scale, -cy / scale], [0.0, 0.0, 1.0]])
    observed = [(corners[i] - (cx, cy)) / scale for i in REFERENCE_IDS]

    # Startwert: affine Näherung (gemeinsame 2x2-Matrix für alle drei Quadrate)
    means = [obs.mean(axis=0) for obs in observed]
    centered = np.vstack([obs - mean for obs, mean in zip(observed, means)])
    affine = np.linalg.lstsq(np.vstack([_SQUARE] * 3), centered, rcond=None)[0].T
    if abs(np.linalg.det(affine)) < 1e-12:
        return None
    c2 = np.linalg.solve(affine, means[1] - means[0])
    c3 = np.linalg.solve(affine, means[2] - means[0])
    p0 = np.array([affine[0, 0], affine[0, 1], means[0][0], affine[1, 0], affine[1, 1], means[0][1],
                   0.0, 0.0, *c2, *c3])

    distances = (REFERENCE_DISTANCE_X_M, REFERENCE_DISTANCE_Y_M)
    distance_weight = (1.0 / scale) / 0.001  # 1 mm Abstandsfehler ~ 1 px Eckenfehler
    params = _least_squares(lambda p: _plane_residuals(p, observed, distances, distance_weight), p0)
    homography_marker = np.linalg.inv(normalize) @ np.append(params[:8], 1.0).reshape(3, 3)
    c2, c3 = params[8:10], params[10:12]

    # Welt-Achsen im Marker-KS von ID1: X Richtung ID2, Y senkrecht und weg von ID3
    if np.linalg.norm(c2) < 1e-6 or np.linalg.norm(c3) < 1e-6:
        return None
    x_axis = c2 / np.linalg.norm(c2)
    y_axis = np.array([-x_axis[1], x_axis[0]])
    if c3 @ y_axis > 0:
        y_axis = -y_axis
    axes = np.column_stack([x_axis, y_axis])  # Weltkoordinaten -> Marker-KS
    axis_angle = math.degrees(math.acos(float(np.clip(x_axis @ (-c3 / np.linalg.norm(c3)), -1.0, 1.0))))
    if not 20.0 < axis_angle < 160.0:
        return None  # ID2 und ID3 (fast) auf einer Linie

    homography_world = homography_marker @ np.block([[axes, np.zeros((2, 1))], [np.zeros((1, 2)), np.ones((1, 1))]])

    # Brennweite
    camera_matrix = guess_matrix.astype(np.float64).copy()
    focal_estimated = False
    if ESTIMATE_FOCAL_LENGTH:
        f = _estimate_focal_length(homography_world, cx, cy)
        width = 2 * cx
        if f is not None and 0.2 * width < f < 5.0 * width:
            candidate = camera_matrix.copy()
            candidate[0, 0] = candidate[1, 1] = f
            if _plane_tilt_deg(homography_world, candidate) >= FOCAL_MIN_TILT_DEG:
                camera_matrix, focal_estimated = candidate, True

    # Kamerapose aus allen 12 Ecken
    world_corners = np.vstack([(_SQUARE + c) @ axes for c in (np.zeros(2), c2, c3)])  # Marker-KS -> Welt
    object_points = np.column_stack([world_corners, np.zeros(12)])
    image_points = np.vstack([corners[i] for i in REFERENCE_IDS]).astype(np.float64)
    no_distortion = np.zeros(5)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, no_distortion, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, camera_matrix, no_distortion, rvec, tvec)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, no_distortion)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))))

    reference_points = {
        marker_id: np.array([*(axes.T @ c), 0.0]) for marker_id, c in zip(REFERENCE_IDS, (np.zeros(2), c2, c3))
    }
    return WorkspaceFrame(
        origin=tvec.reshape(3),
        rotation=cv2.Rodrigues(rvec)[0],
        reference_points=reference_points,
        axis_angle_deg=axis_angle,
        camera_matrix=camera_matrix,
        focal_estimated=focal_estimated,
        reprojection_rms_px=rms,
    )


class ReferenceFrameEstimator:
    """Lernt das Arbeitsraum-KS aus den Referenzmarkern ein und fixiert es.

    Geht davon aus, dass Kamera und Referenzmarker fest montiert sind. Während
    des (Neu-)Einlernens gibt es kein gültiges KS, damit nie mit einem
    veralteten KS gerechnet wird.
    """

    def __init__(self) -> None:
        """Initialisiert den Schätzer im Zustand "einlernen"."""
        self._samples: list[dict[int, np.ndarray]] = []
        self._locked_corners: dict[int, np.ndarray] | None = None
        self._drift_count = 0
        self.frame: WorkspaceFrame | None = None

    @property
    def is_teaching(self) -> bool:
        """True, solange eingelernt wird."""
        return self._locked_corners is None

    @property
    def progress(self) -> tuple[int, int]:
        """Fortschritt beim Einlernen als (gesammelt, benötigt)."""
        return len(self._samples), REFERENCE_TEACH_FRAMES

    def restart(self, reason: str) -> None:
        """Verwirft das KS und startet das Einlernen neu.

        Args:
            reason: Grund für das Log.
        """
        self._samples.clear()
        self._locked_corners = None
        self._drift_count = 0
        self.frame = None
        logger.info(f"Referenz-KS wird neu eingelernt ({reason}).")

    def update(self, marker_corners: dict[int, np.ndarray], guess_matrix: np.ndarray) -> WorkspaceFrame | None:
        """Verarbeitet die Marker des aktuellen Bildes.

        Args:
            marker_corners: Ecken je Marker-ID im entzerrten Bild.
            guess_matrix: Kameramatrix aus build_camera_matrix.

        Returns:
            Das fixierte KS oder None, solange keins gültig ist.
        """
        if self._locked_corners is not None:
            self._check_drift(marker_corners)
        if self._locked_corners is None:
            self._collect(marker_corners, guess_matrix)
        return self.frame

    def _check_drift(self, marker_corners: dict[int, np.ndarray]) -> None:
        deviations = [
            float(np.linalg.norm(marker_corners[i] - self._locked_corners[i], axis=1).mean())
            for i in REFERENCE_IDS
            if i in marker_corners
        ]
        if deviations and max(deviations) > REFERENCE_DRIFT_PX:
            self._drift_count += 1
            if self._drift_count >= REFERENCE_DRIFT_FRAMES:
                logger.warning("Referenzmarker haben sich im Bild verschoben (Kamera bewegt?).")
                self.restart("Verschiebung erkannt")
        else:
            self._drift_count = 0

    def _collect(self, marker_corners: dict[int, np.ndarray], guess_matrix: np.ndarray) -> None:
        if not all(i in marker_corners for i in REFERENCE_IDS):
            return
        self._samples.append({i: marker_corners[i] for i in REFERENCE_IDS})
        if len(self._samples) < REFERENCE_TEACH_FRAMES:
            return

        # Median über alle Bilder: robust gegen einzelne Fehldetektionen
        median = {i: np.median([s[i] for s in self._samples], axis=0) for i in REFERENCE_IDS}
        self._samples.clear()
        try:
            frame = estimate_workspace_frame(median, guess_matrix)
        except (np.linalg.LinAlgError, cv2.error) as exc:
            logger.error(f"Referenzebene nicht berechenbar: {exc}")
            return
        if frame is None:
            logger.error("Referenzmarker ungünstig angeordnet (ID2 und ID3 in gleicher Richtung?).")
            return
        if frame.reprojection_rms_px > REFERENCE_MAX_RMS_PX:
            logger.error(
                f"Referenzebene verworfen: Restfehler {frame.reprojection_rms_px:.2f} px > "
                f"{REFERENCE_MAX_RMS_PX} px (Kalibrierung/Auflösung oder Marker prüfen). Neuer Versuch ..."
            )
            return

        self.frame = frame
        self._locked_corners = median
        logger.info(
            f"Koordinatensystem fixiert: ID2 bei x={frame.reference_points[ID_POS_X][0]:.3f} m, "
            f"ID3 bei y={frame.reference_points[ID_NEG_Y][1]:.3f} m, "
            f"Achswinkel {frame.axis_angle_deg:.1f}°, "
            f"f={frame.camera_matrix[0, 0]:.0f} px ({'geschätzt' if frame.focal_estimated else 'kalibriert'}), "
            f"Restfehler {frame.reprojection_rms_px:.2f} px"
        )


# =============================================================================
# Roboterpose
# =============================================================================


@dataclass
class RobotPose:
    """Roboterpose im Arbeitsraum-KS.

    Attributes:
        x: X-Position in m (Lot von der Markermitte auf die Referenzebene).
        y: Y-Position in m.
        theta_deg: Orientierung in Grad, 0° = +X, positiv Richtung +Y, Bereich (-180, 180].
        z: Höhe der Markermitte über der Referenzebene in m (Vorzeichen je nach Kameraseite).
    """

    x: float
    y: float
    theta_deg: float
    z: float


class RobotTracker:
    """Bestimmt und glättet die Roboterpose aus den Ecken des Robotermarkers.

    Die Sehstrahlen durch die vier Markerecken werden mit der Ebene in
    Markerhöhe geschnitten. Das nutzt nur die genauen Pixelpositionen statt
    der verrauschten Entfernungsschätzung eines einzelnen Markers.
    """

    def __init__(self) -> None:
        """Initialisiert den Tracker ohne Pose und ohne Höhenschätzung."""
        self._height_estimate: float | None = None
        self._state: np.ndarray | None = None  # x, y, cos(theta), sin(theta)

    @property
    def height_is_estimated(self) -> bool:
        """True, wenn die Markerhöhe automatisch geschätzt wird."""
        return ROBOT_MARKER_HEIGHT_M is None

    def reset_pose(self) -> None:
        """Verwirft die geglättete Pose (z. B. nachdem der Roboter verloren wurde)."""
        self._state = None

    def reset(self) -> None:
        """Verwirft Pose und Höhenschätzung (z. B. bei neuem KS)."""
        self._state = None
        self._height_estimate = None

    def update(self, frame: WorkspaceFrame, corners: np.ndarray) -> RobotPose | None:
        """Berechnet die geglättete Roboterpose.

        Args:
            frame: Arbeitsraum-KS.
            corners: Ecken des Robotermarkers im entzerrten Bild, Form (4, 2).

        Returns:
            Roboterpose oder None, falls sie nicht bestimmt werden kann.
        """
        z_plane = self._marker_plane_height(frame, corners)
        if z_plane is None:
            return None
        world = frame.intersect_pixels(corners, z_plane)
        if world is None:
            return None

        center = world.mean(axis=0)
        x_dir = (world[1] - world[0]) + (world[2] - world[3])  # Marker-X-Achse (links -> rechts)
        theta = math.atan2(x_dir[1], x_dir[0]) + math.radians(ROBOT_HEADING_OFFSET_DEG)
        raw = np.array([center[0], center[1], math.cos(theta), math.sin(theta)])

        if self._state is None or np.linalg.norm(raw[:2] - self._state[:2]) > ROBOT_RESET_DISTANCE_M:
            self._state = raw
        else:
            self._state = self._state + ROBOT_SMOOTHING_ALPHA * (raw - self._state)

        x, y, c, s = self._state
        return RobotPose(float(x), float(y), math.degrees(math.atan2(s, c)), z_plane)

    def _marker_plane_height(self, frame: WorkspaceFrame, corners: np.ndarray) -> float | None:
        if ROBOT_MARKER_HEIGHT_M is not None:
            # Markerebene liegt auf der Kameraseite der Referenzebene
            return abs(ROBOT_MARKER_HEIGHT_M) * float(np.sign(frame.camera_position[2]))

        center_cam = marker_center_camera(corners, frame.camera_matrix)
        if center_cam is None:
            return self._height_estimate
        z = float(frame.point_to_world(center_cam)[2])
        if self._height_estimate is None:
            self._height_estimate = z
        else:
            self._height_estimate += ROBOT_HEIGHT_SMOOTHING_ALPHA * (z - self._height_estimate)
        return self._height_estimate


# =============================================================================
# MQTT
# =============================================================================


class TrackerMqtt:
    """MQTT-Anbindung: sendet Pose und Status, empfängt Befehle.

    paho-mqtt läuft in einem eigenen Netzwerkthread und verbindet sich nach
    einem Abbruch selbst neu. Ohne Broker läuft das Tracking weiter, Posen
    werden dann verworfen und nie nachgeschickt.
    """

    def __init__(self, host: str, port: int = MQTT_PORT) -> None:
        """Startet den Verbindungsaufbau im Hintergrund.

        Args:
            host: Adresse des Brokers.
            port: Port des Brokers.
        """
        self.reteach_requested = threading.Event()
        self._status = "teaching"
        self._seq = 0
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID)
        self._client.will_set(TOPIC_STATUS, "offline", qos=1, retain=True)
        self._client.reconnect_delay_set(min_delay=1, max_delay=10)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        logger.info(f"MQTT: verbinde mit {host}:{port} als '{MQTT_CLIENT_ID}' ...")
        self._client.connect_async(host, port, keepalive=MQTT_KEEPALIVE_S)
        self._client.loop_start()

    @property
    def connected(self) -> bool:
        """True, wenn eine Verbindung zum Broker besteht."""
        return self._client.is_connected()

    def publish_pose(self, pose: RobotPose, stamp: float) -> None:
        """Sendet eine Pose (QoS 0, nicht retained, bei fehlender Verbindung verworfen).

        Args:
            pose: Roboterpose.
            stamp: Aufnahmezeit des Bildes als Unix-Zeit in s.
        """
        self._seq += 1
        if not self._client.is_connected():
            return
        payload = json.dumps(
            {
                "seq": self._seq,
                "t": round(stamp, 3),
                "x": round(pose.x, 4),
                "y": round(pose.y, 4),
                "theta": round(pose.theta_deg, 2),
            },
            separators=(",", ":"),
        )
        self._client.publish(TOPIC_POSE, payload, qos=0, retain=False)

    def set_status(self, status: str) -> None:
        """Sendet den Status (retained), aber nur bei Änderung.

        Args:
            status: teaching, tracking oder lost.
        """
        if status == self._status:
            return
        logger.info(f"Status: {self._status} -> {status}")
        self._status = status
        if self._client.is_connected():
            self._client.publish(TOPIC_STATUS, status, qos=1, retain=True)

    def close(self) -> None:
        """Meldet "offline" und trennt die Verbindung."""
        try:
            if self._client.is_connected():
                self._client.publish(TOPIC_STATUS, "offline", qos=1, retain=True).wait_for_publish(timeout=1.0)
        except (RuntimeError, ValueError) as exc:
            logger.warning(f"MQTT: 'offline' konnte nicht gesendet werden: {exc}")
        finally:
            self._client.disconnect()
            self._client.loop_stop()

    def _on_connect(self, client: mqtt.Client, userdata, flags, reason_code, properties) -> None:
        if reason_code.is_failure:
            logger.error(f"MQTT: Verbindung abgelehnt: {reason_code}")
            return
        logger.info("MQTT: verbunden.")
        client.subscribe(TOPIC_CMD, qos=1)
        # Nach einem Abbruch steht am Broker noch der Last Will "offline"
        client.publish(TOPIC_STATUS, self._status, qos=1, retain=True)

    def _on_disconnect(self, client: mqtt.Client, userdata, flags, reason_code, properties) -> None:
        if reason_code != 0:
            logger.warning(f"MQTT: Verbindung verloren ({reason_code}), verbinde neu ...")

    def _on_message(self, client: mqtt.Client, userdata, message: mqtt.MQTTMessage) -> None:
        if message.retain:
            return  # alte, gespeicherte Befehle nie ausführen
        command = message.payload.decode("utf-8", "replace").strip().lower()
        if command == "reteach":
            self.reteach_requested.set()
        else:
            logger.warning(f"MQTT: unbekannter Befehl auf {message.topic}: {command[:50]!r}")


# =============================================================================
# Debug-Anzeige
# =============================================================================


def _drawable(p: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(p)) and np.all(np.abs(p) < 1e5))


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def _text(img: np.ndarray, text: str, org: tuple[int, int], color=COLOR_TEXT, scale: float = 0.6) -> None:
    """Text mit dunklem Rand, auf jedem Hintergrund lesbar (nur ASCII)."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def _line(img: np.ndarray, a: np.ndarray, b: np.ndarray, color, thickness: int, label: str = "", arrow: bool = False) -> None:
    if not (_drawable(a) and _drawable(b)):
        return
    if arrow:
        length = max(float(np.linalg.norm(b - a)), 1.0)
        cv2.arrowedLine(img, _pt(a), _pt(b), color, thickness, cv2.LINE_AA, tipLength=min(0.5, 20 / length))
    else:
        cv2.line(img, _pt(a), _pt(b), color, thickness, cv2.LINE_AA)
    if label:
        _text(img, label, (_pt(b)[0] + 10, _pt(b)[1] - 10), color, 0.8)


def draw_workspace(img: np.ndarray, frame: WorkspaceFrame, camera_matrix: np.ndarray) -> None:
    """Zeichnet Achsen und Referenzpunkte."""
    ref_x, ref_y = frame.reference_points[ID_POS_X], frame.reference_points[ID_NEG_Y]
    origin, p2, p3, y_neg, y_pos = frame.project(
        [(0, 0, 0), ref_x, ref_y, (0, ref_y[1], 0), (0, -ref_y[1], 0)], camera_matrix
    )
    _line(img, origin, p2, COLOR_X, 3, "+X", arrow=True)
    _line(img, origin, y_neg, COLOR_Y, 2, "-Y")
    _line(img, origin, y_pos, COLOR_Y, 3, "+Y", arrow=True)
    for p, label, color in ((origin, "O (ID1)", COLOR_TEXT), (p2, "ID2", COLOR_X), (p3, "ID3", COLOR_Y)):
        if _drawable(p):
            cv2.circle(img, _pt(p), 7, color, -1, cv2.LINE_AA)
            _text(img, label, (_pt(p)[0] + 10, _pt(p)[1] + 24), color)


def draw_robot(img: np.ndarray, frame: WorkspaceFrame, robot: RobotPose, camera_matrix: np.ndarray) -> None:
    """Zeichnet Position und Blickrichtung des Roboters."""
    th = math.radians(robot.theta_deg)
    tip = (robot.x + ROBOT_ARROW_LENGTH_M * math.cos(th), robot.y + ROBOT_ARROW_LENGTH_M * math.sin(th), robot.z)
    center, tip_px = frame.project([(robot.x, robot.y, robot.z), tip], camera_matrix)
    if not _drawable(center):
        return
    cv2.circle(img, _pt(center), 9, COLOR_ROBOT, 2, cv2.LINE_AA)
    _line(img, center, tip_px, COLOR_ROBOT, 3, arrow=True)
    _text(img, f"x={robot.x:+.3f} y={robot.y:+.3f} th={robot.theta_deg:+.1f}",
          (_pt(center)[0] + 14, _pt(center)[1] - 14), COLOR_ROBOT)


def draw_info_panel(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Halbtransparenter Infokasten oben links."""
    pad, line_h, scale = 10, 24, 0.6
    width = max(cv2.getTextSize(text, FONT, scale, 1)[0][0] for text, _ in lines) + 2 * pad
    height = 2 * pad + line_h * len(lines) - 6
    roi = img[:height, :width]
    roi[:] = (roi * 0.35).astype(np.uint8)
    for i, (text, color) in enumerate(lines):
        _text(img, text, (pad, pad + line_h * (i + 1) - 8), color, scale)


class DebugView:
    """Verkleinertes, entzerrtes Kamerabild mit Overlay, mit begrenzter Bildrate."""

    def __init__(self, image_size: tuple[int, int], camera_matrix: np.ndarray) -> None:
        """Bereitet die Fisheye-Entzerrung in Anzeigegröße vor und öffnet das Fenster.

        Args:
            image_size: (Breite, Höhe) des Kamerabildes.
            camera_matrix: Kameramatrix aus build_camera_matrix.
        """
        w, h = image_size
        self.scale = min(1.0, DISPLAY_MAX_WIDTH / w)
        self._size = (round(w * self.scale), round(h * self.scale))
        small_matrix = camera_matrix.copy()
        small_matrix[:2] *= self.scale
        # Entzerrt wird nur das kleine Anzeigebild, das ist sehr viel billiger
        self._maps = cv2.fisheye.initUndistortRectifyMap(
            small_matrix, CALIB_D, np.eye(3), small_matrix, self._size, cv2.CV_16SC2
        )
        self._next_show = 0.0
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def due(self, now: float) -> bool:
        """True, wenn das nächste Anzeigebild fällig ist."""
        return now >= self._next_show

    def show(
        self,
        now: float,
        gray: np.ndarray,
        corners: dict[int, np.ndarray],
        workspace: WorkspaceFrame | None,
        robot: RobotPose | None,
        info_lines: list[tuple[str, tuple[int, int, int]]],
    ) -> int:
        """Zeichnet und zeigt das Anzeigebild.

        Returns:
            Gedrückte Taste (-1 = keine), ord('q') wenn das Fenster geschlossen wurde.
        """
        self._next_show = now + 1.0 / DISPLAY_MAX_FPS
        small = cv2.resize(gray, self._size, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(
            cv2.remap(small, *self._maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),
            cv2.COLOR_GRAY2BGR,
        )

        for marker_id, c in corners.items():
            pts = c * self.scale
            if _drawable(pts):
                cv2.polylines(img, [np.round(pts).astype(np.int32)], True, COLOR_MARKER, 2, cv2.LINE_AA)
                if workspace is None:  # sonst beschriften Achsen und Roboterpfeil die Marker
                    _text(img, f"ID{marker_id}", _pt(pts[0]), COLOR_MARKER, 0.5)
        if workspace is not None:
            display_matrix = workspace.camera_matrix.copy()
            display_matrix[:2] *= self.scale
            draw_workspace(img, workspace, display_matrix)
            if robot is not None:
                draw_robot(img, workspace, robot, display_matrix)
        draw_info_panel(img, info_lines)

        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return ord("q")
        return key


def build_info_lines(
    image_size: tuple[int, int],
    fps: float,
    latency_ms: float,
    full_scan: bool,
    mqtt_connected: bool,
    corners: dict[int, np.ndarray],
    references: ReferenceFrameEstimator,
    robot: RobotPose | None,
    robot_tracker: RobotTracker,
) -> list[tuple[str, tuple[int, int, int]]]:
    """Stellt die Textzeilen für den Infokasten zusammen (nur ASCII)."""
    w, h = image_size
    lines = [(
        f"{w}x{h}  {fps:4.1f} FPS  Latenz {latency_ms:3.0f} ms  {'Vollbild' if full_scan else 'Suchfenster'}  "
        f"MQTT {'ok' if mqtt_connected else 'GETRENNT'}   [r] neu einlernen  [q] Ende",
        COLOR_TEXT if mqtt_connected else COLOR_WARN,
    )]

    visible = "  ".join(f"ID{i} {'ok' if i in corners else '--'}" for i in REFERENCE_IDS)
    if references.is_teaching:
        done, total = references.progress
        lines.append((f"Referenz: einlernen {done}/{total}  ({visible})", COLOR_WARN))
    else:
        frame = references.frame
        lines.append((
            f"Referenz: fixiert ({visible})  Restfehler {frame.reprojection_rms_px:.2f} px  "
            f"f={frame.camera_matrix[0, 0]:.0f} px ({'geschaetzt' if frame.focal_estimated else 'kalibriert'})  "
            f"Achswinkel {frame.axis_angle_deg:.1f} deg",
            COLOR_TEXT if frame.reprojection_rms_px < 2.0 else COLOR_WARN,
        ))

    if robot is not None:
        height_source = "geschaetzt" if robot_tracker.height_is_estimated else "fest"
        lines.append((
            f"Roboter: x={robot.x:+.3f} m  y={robot.y:+.3f} m  theta={robot.theta_deg:+.1f} deg  "
            f"h={abs(robot.z):.3f} m ({height_source})",
            COLOR_ROBOT,
        ))
    else:
        lines.append(("Roboter (ID0): keine Pose", COLOR_WARN))
    return lines


# =============================================================================
# Hauptprogramm
# =============================================================================


def track(camera: CameraReader, link: TrackerMqtt, headless: bool, stop: threading.Event) -> None:
    """Hauptschleife: Bild holen, Marker finden, Pose berechnen und senden.

    Args:
        camera: Laufender Kameraleser.
        link: MQTT-Anbindung.
        headless: True = ohne Anzeigefenster.
        stop: Wird gesetzt, um die Schleife zu beenden.
    """
    references = ReferenceFrameEstimator()
    robot_tracker = RobotTracker()
    image_size: tuple[int, int] | None = None
    camera_matrix = np.eye(3)
    detector: MarkerDetector | None = None
    view: DebugView | None = None
    active_frame: WorkspaceFrame | None = None
    last_raw: dict[int, np.ndarray] = {}
    last_full_scan = last_robot_seen = -math.inf
    fps = latency_ms = 0.0
    t_prev = t_stats = time.monotonic()
    n_frames = n_full = 0

    while not stop.is_set():
        item = camera.read(CAMERA_FRAME_TIMEOUT_S)
        if item is None:
            logger.error("Kein Bild von der Kamera erhalten, Abbruch.")
            return
        raw_frame, stamp = item
        gray = decode_gray(raw_frame)
        if gray is None:
            warn_throttled("Defektes Kamerabild übersprungen.")
            continue

        h, w = gray.shape[:2]
        if image_size is None:
            image_size = (w, h)
            camera_matrix = build_camera_matrix(w, h)
            detector = MarkerDetector(image_size)
            view = None if headless else DebugView(image_size, camera_matrix)
            if raw_frame.ndim == 3:
                logger.warning("Kamera liefert BGR statt JPEG-Rohdaten, Dekodierung ist langsamer.")
            else:
                logger.info("Kamera liefert JPEG-Rohdaten, Dekodierung direkt als Graubild.")
            logger.info(f"Läuft ({w} x {h}). Referenzmarker ID1-3 frei sichtbar halten, bis das KS fixiert ist.")
        elif image_size != (w, h):
            logger.error(f"Bildgröße hat sich geändert ({image_size} -> {(w, h)}), Abbruch.")
            return

        if link.reteach_requested.is_set():
            link.reteach_requested.clear()
            references.restart("Befehl über MQTT")

        # --- Erkennung: Suchfenster, solange der Roboter verfolgt wird ---
        now = time.monotonic()
        references_missing = any(i not in last_raw for i in REFERENCE_IDS)
        full_scan = (
            references.frame is None
            or ID_ROBOT not in last_raw
            or (references_missing and now - last_full_scan >= FULL_SCAN_INTERVAL_S)
        )
        if full_scan:
            last_raw = detector.detect(gray)
            last_full_scan = now
            n_full += 1
        else:
            last_raw = detector.detect(gray, [roi_around(c, image_size) for c in last_raw.values()])
        corners = {i: undistort_points(c, camera_matrix) for i, c in last_raw.items()}

        # --- Koordinatensystem und Roboterpose ---
        workspace = references.update(corners, camera_matrix)
        if workspace is not active_frame:
            robot_tracker.reset()  # Höhenschätzung gehört zum alten KS
            active_frame = workspace

        robot = None
        if workspace is not None and ID_ROBOT in corners:
            robot = robot_tracker.update(workspace, corners[ID_ROBOT])
        if robot is not None:
            link.publish_pose(robot, stamp)
            latency_ms = (time.time() - stamp) * 1000.0
            last_robot_seen = now
        elif now - last_robot_seen > ROBOT_LOST_TIMEOUT_S:
            robot_tracker.reset_pose()

        if workspace is None:
            status = "teaching"
        elif now - last_robot_seen <= ROBOT_LOST_TIMEOUT_S:
            status = "tracking"
        else:
            status = "lost"
        link.set_status(status)

        # --- Statistik ---
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        n_frames += 1
        if now - t_stats >= STATS_INTERVAL_S:
            logger.info(
                f"{n_frames / (now - t_stats):.1f} FPS ({n_full} davon Vollbildsuche), "
                f"Status {status}, MQTT {'verbunden' if link.connected else 'GETRENNT'}"
            )
            t_stats, n_frames, n_full = now, 0, 0

        # --- Anzeige ---
        if view is not None and view.due(now):
            lines = build_info_lines(image_size, fps, latency_ms, full_scan, link.connected,
                                     corners, references, robot, robot_tracker)
            key = view.show(now, gray, corners, workspace, robot, lines)
            if key in (ord("q"), 27):  # q oder ESC
                return
            if key == ord("r"):
                references.restart("Taste r")


def run(headless: bool, broker: str) -> None:
    """Startet Kamera und MQTT, führt das Tracking aus und räumt sicher auf.

    Args:
        headless: True = ohne Anzeigefenster.
        broker: Adresse des MQTT-Brokers.
    """
    index = find_camera_index()
    if index is None:
        logger.error(f"Keine Kamera mit '{CAMERA_NAME_KEYWORD}' im Namen gefunden. Vorhandene Geräte:")
        for dev in list_video_devices():
            suffix = "" if dev.is_capture_node else "  (Metadaten-Knoten)"
            logger.error(f"  /dev/video{dev.index}: {dev.name}{suffix}")
        return

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    camera = CameraReader(index)
    try:
        link = TrackerMqtt(broker)
        try:
            track(camera, link, headless, stop)
        finally:
            link.close()
    finally:
        camera.close()
        if not headless:
            cv2.destroyAllWindows()
    logger.info("Beendet.")


def main() -> None:
    """Liest die Kommandozeile und startet das Tracking."""
    parser = argparse.ArgumentParser(description="ArUco-Tracking des Roboters, Pose über MQTT.")
    parser.add_argument("--headless", action="store_true", help="ohne Anzeigefenster")
    parser.add_argument("--broker", default=MQTT_HOST, help=f"Adresse des MQTT-Brokers (Standard: {MQTT_HOST})")
    args = parser.parse_args()
    run(args.headless, args.broker)


if __name__ == "__main__":
    main()