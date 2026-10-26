#!/usr/bin/env python3
"""Detect the metal supports in the IR camera image with the trained YOLO model.

The class StiltDetector bundles camera, model and MQTT so the station state
machine can use it like the other devices:

    detector = StiltDetector(client_id="station_vision")
    detector.connect()                 # load model, open camera, connect to the broker
    detections = detector.detect()     # grab a fresh frame, detect, save, publish
    detector.disconnect()

Every result is published as JSON on TOPIC_DETECTIONS; TOPIC_STATUS holds
"online" / "offline" (retained, "offline" also as last will). Without a
broker the detection still works, only nothing is published.

Run directly, the script opens a live view of the IR camera; pressing space
grabs the current frame, runs the detection and prints the coordinates of
every support found. The raw frame and an annotated copy are stored on the
desktop.

The camera node is detected automatically: the script probes the /dev/videoN
nodes and takes the first one that actually delivers YUYV frames.

Usage:
    python scripts/F_stilt_erkennung_YOLO.py

Keys in the viewer window:
    SPACE     grab a frame, detect the supports, print their coordinates
    c         clear the boxes of the last detection
    n         toggle contrast normalisation of the live preview only
    q / ESC   quit
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# The Quadro K620 (sm_50) is not covered by current torch wheels - hide the
# GPU so ultralytics stays on the CPU. Must happen before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from loguru import logger

# --- MQTT -------------------------------------------------------------------
BROKER = "192.168.2.1"
PORT = 1883
CLIENT_ID = "stilt_detector"
TOPIC_DETECTIONS = "station/vision/detections"  # JSON with every detection result
TOPIC_STATUS = "station/vision/status"          # online / offline (retained)

# --- camera -----------------------------------------------------------------
FOURCC = "YUYV"
WIDTH = 340
HEIGHT = 340
# Optional: substring of the camera name (see /sys/class/video4linux/videoN/name)
# to pin detection to one camera if several YUYV cameras are connected.
NAME_HINT = ""

# --- model ------------------------------------------------------------------
# "best.pt" (PyTorch) or "best.onnx" (needs onnxruntime).
MODEL_FILE = "best.pt"
# Must match the imgsz used during training / ONNX export.
IMGSZ = 640
CONF_THRESHOLD = 0.5
DEVICE = "cpu"

OUTPUT_DIR_NAME = "ir_detections"
IMAGE_RE = re.compile(r"ir_(\d+)\.png")
SYSFS_V4L = Path("/sys/class/video4linux")
NO_FRAME_TIMEOUT_S = 5.0
PROBE_READS = 5
# Frames thrown away before a detection, so the image is fresh and not
# an old one still waiting in the camera buffer.
FLUSH_FRAMES = 3


def list_video_nodes() -> list[int]:
    """Return the indices of all /dev/videoN nodes, sorted ascending.

    Returns:
        List of video node indices, e.g. ``[0, 1, 2, 3]``.
    """
    indices = []
    for entry in SYSFS_V4L.glob("video*"):
        match = re.fullmatch(r"video(\d+)", entry.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def node_name(index: int) -> str:
    """Read the device name the driver reports for a video node.

    Args:
        index: Video node index (N in /dev/videoN).

    Returns:
        The device name, or ``"?"`` if it cannot be read.
    """
    try:
        return (SYSFS_V4L / f"video{index}" / "name").read_text().strip()
    except OSError:
        return "?"


def node_formats(index: int) -> list[str]:
    """Query the pixel formats a video node offers, using v4l2-ctl.

    Args:
        index: Video node index (N in /dev/videoN).

    Returns:
        FourCC codes such as ``["YUYV", "MJPG"]``. Empty if the node offers
        no capture formats (e.g. UVC metadata nodes) or v4l2-ctl is missing.
    """
    if shutil.which("v4l2-ctl") is None:
        return []
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{index}", "--list-formats"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return []
    return [fourcc.strip() for fourcc in re.findall(r"'(.{4})'", result.stdout)]


def open_node(index: int) -> cv2.VideoCapture | None:
    """Open a video node with the configured format and check that it streams.

    Args:
        index: Video node index (N in /dev/videoN).

    Returns:
        The opened capture, or ``None`` if the node cannot be opened or
        delivers no frames.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    for _ in range(PROBE_READS):
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        time.sleep(0.1)

    cap.release()
    return None


def find_camera() -> tuple[int, cv2.VideoCapture] | None:
    """Find and open the first video node that streams in the configured format.

    Returns:
        Tuple of (node index, opened capture), or ``None`` if no node works.
    """
    have_v4l2_ctl = shutil.which("v4l2-ctl") is not None
    if not have_v4l2_ctl:
        logger.warning("v4l2-ctl not found - probing every node (sudo apt install v4l-utils)")

    for index in list_video_nodes():
        name = node_name(index)
        if NAME_HINT and NAME_HINT.lower() not in name.lower():
            continue
        if have_v4l2_ctl and FOURCC not in node_formats(index):
            continue
        cap = open_node(index)
        if cap is not None:
            logger.info(f"Camera detected: /dev/video{index} ({name})")
            return index, cap
        logger.debug(f"/dev/video{index} ({name}) offers {FOURCC} but delivers no frames")
    return None


def desktop_dir() -> Path:
    """Locate the desktop folder, whatever its localised name is.

    Returns:
        Path to the desktop folder (may not exist yet).
    """
    home = Path.home()
    if shutil.which("xdg-user-dir") is not None:
        try:
            result = subprocess.run(
                ["xdg-user-dir", "DESKTOP"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            path = Path(result.stdout.strip())
            if result.stdout.strip() and path != home and path.is_dir():
                return path
        except subprocess.TimeoutExpired:
            pass
    for name in ("Desktop", "Schreibtisch"):
        candidate = home / name
        if candidate.is_dir():
            return candidate
    return home / "Desktop"


def output_dir() -> Path:
    """Create (if needed) and return the folder the captures go into.

    Returns:
        Path to the output folder on the desktop.
    """
    folder = desktop_dir() / OUTPUT_DIR_NAME
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def next_image_path(folder: Path) -> Path:
    """Build the path for the next capture, continuing the existing numbering.

    Args:
        folder: Folder the captures are stored in.

    Returns:
        Path such as ``<folder>/ir_0007.png``.
    """
    numbers = []
    for path in folder.glob("ir_*.png"):
        match = IMAGE_RE.fullmatch(path.name)
        if match:
            numbers.append(int(match.group(1)))
    return folder / f"ir_{max(numbers, default=0) + 1:04d}.png"


def find_weights() -> Path | None:
    """Look for the model weights in the usual places.

    Returns:
        Path to the weights file, or ``None`` if it was not found.
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / MODEL_FILE,
        Path.cwd() / "models" / MODEL_FILE,
        Path.cwd() / "weights" / MODEL_FILE,
        script_dir / MODEL_FILE,
        script_dir.parent / "models" / MODEL_FILE,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    logger.error(f"{MODEL_FILE} not found. Looked in:")
    for candidate in candidates:
        logger.error(f"  {candidate}")
    return None


def load_model(weights: Path):
    """Load the trained YOLO model.

    Args:
        weights: Path to best.pt or best.onnx.

    Returns:
        The loaded ``ultralytics.YOLO`` model, or ``None`` on failure.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is missing - install it with: pip install ultralytics")
        return None
    try:
        model = YOLO(str(weights), task="detect")
    except Exception as error:  # onnxruntime missing, corrupt weights, ...
        logger.error(f"Could not load {weights}: {error}")
        return None
    logger.info(f"Model loaded: {weights} (classes: {', '.join(model.names.values())})")
    return model


def detect(model, image: np.ndarray) -> list[dict]:
    """Run the model on one greyscale frame and collect the detections.

    Args:
        model: Loaded YOLO model.
        image: Single-channel uint8 image, as it comes from the camera.

    Returns:
        List of detections, numbered from left to right. Each entry holds the
        keys ``id``, ``label``, ``confidence``, ``x1``, ``y1``, ``x2``, ``y2``,
        ``cx``, ``cy``, ``width`` and ``height``, all pixel values.
    """
    # The model was trained on the saved greyscale PNGs, so feed it the same
    # image content - just as the 3-channel array that ultralytics expects.
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    results = model.predict(bgr, imgsz=IMGSZ, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
    boxes = results[0].boxes

    detections = []
    for index in range(len(boxes)):
        x1, y1, x2, y2 = (float(value) for value in boxes.xyxy[index].tolist())
        detections.append(
            {
                "label": model.names[int(boxes.cls[index])],
                "confidence": float(boxes.conf[index]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1,
            }
        )

    detections.sort(key=lambda item: item["cx"])
    for number, detection in enumerate(detections, start=1):
        detection["id"] = number
    return detections


def print_detections(detections: list[dict]) -> None:
    """Print the detected supports as a table of pixel coordinates.

    Args:
        detections: Detections as returned by :func:`detect`.
    """
    if not detections:
        print("no supports detected")
        return
    print(f"{'#':>2}  {'label':<12} {'conf':>5}  {'cx':>7} {'cy':>7}  {'w':>6} {'h':>6}  box (x1,y1,x2,y2)")
    for item in detections:
        print(
            f"{item['id']:>2}  {item['label']:<12} {item['confidence']:>5.2f}  "
            f"{item['cx']:>7.1f} {item['cy']:>7.1f}  {item['width']:>6.1f} {item['height']:>6.1f}  "
            f"({item['x1']:.0f}, {item['y1']:.0f}, {item['x2']:.0f}, {item['y2']:.0f})"
        )


def annotate(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw the detections onto a copy of the frame.

    Args:
        image: Single-channel uint8 image.
        detections: Detections as returned by :func:`detect`.

    Returns:
        Three-channel BGR image with boxes, centre points and labels.
    """
    view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for item in detections:
        top_left = (int(item["x1"]), int(item["y1"]))
        bottom_right = (int(item["x2"]), int(item["y2"]))
        cv2.rectangle(view, top_left, bottom_right, (0, 255, 0), 1)
        cv2.circle(view, (int(item["cx"]), int(item["cy"])), 3, (0, 0, 255), -1)
        cv2.putText(
            view,
            f"{item['id']} {item['confidence']:.2f}",
            (top_left[0], max(top_left[1] - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return view


def to_display(frame: np.ndarray, normalize: bool) -> np.ndarray:
    """Convert a captured frame into an 8-bit greyscale image.

    Args:
        frame: Frame as returned by ``cv2.VideoCapture.read()``.
        normalize: Stretch the contrast to the full 0-255 range.

    Returns:
        Single-channel uint8 image.
    """
    if frame.ndim == 3:
        # OpenCV hands the IR stream out as 3-channel BGR by default;
        # for these formats all three channels are identical.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if normalize or frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return frame


def window_open(window: str) -> bool:
    """Check whether the viewer window is still there.

    Args:
        window: Window name.

    Returns:
        ``True`` while the window is visible. The Qt backend raises instead of
        returning 0 once its GUI thread is gone, which counts as closed here.
    """
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


class StiltDetector:
    """Camera + YOLO model + MQTT, with the same interface as the other devices.

    Attributes:
        last_detections: Result of the last detection (empty before the first).
    """

    def __init__(self, client_id: str = CLIENT_ID, save_images: bool = True) -> None:
        """
        Args:
            client_id: Unique MQTT client id of this script.
            save_images: Store every analysed frame (raw + annotated) on the desktop.
        """
        self.save_images = save_images
        # False: Bilder ohne Treffer weder speichern noch ausgeben (wichtig bei Dauersuche)
        self.keep_empty = True
        self.last_detections: list[dict] = []
        self.camera_index: int | None = None
        self._model = None
        self._cap: cv2.VideoCapture | None = None
        self._folder: Path | None = None
        self._camera_lock = threading.Lock()
        self._broker_connected = False

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._client.will_set(TOPIC_STATUS, "offline", qos=1, retain=True)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    # ------------------------------------------------------------------
    #  Connection
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Load the model, open the camera and connect to the broker.

        A missing broker is only logged: detection works without it.

        Raises:
            RuntimeError: If the model or the camera is not available.
        """
        weights = find_weights()
        self._model = load_model(weights) if weights is not None else None
        if self._model is None:
            raise RuntimeError("YOLO model could not be loaded")

        found = find_camera()
        if found is None:
            raise RuntimeError(f"No video node delivering {FOURCC} frames found - is the camera plugged in?")
        self.camera_index, self._cap = found
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.save_images:
            self._folder = output_dir()
            logger.info(f"Captures go to {self._folder}")

        # In the background: no waiting when the broker is down, reconnects on its own later
        self._client.reconnect_delay_set(min_delay=1, max_delay=10)
        self._client.connect_async(BROKER, PORT, keepalive=60)
        self._client.loop_start()

    def disconnect(self) -> None:
        """Release the camera and leave the broker cleanly."""
        with self._camera_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        if self._broker_connected:
            self._client.publish(TOPIC_STATUS, "offline", qos=1, retain=True).wait_for_publish(timeout=1.0)
        self._client.disconnect()
        self._client.loop_stop()

    def is_connected(self) -> bool:
        """True if camera and model are ready (MQTT is optional)."""
        return self._cap is not None and self._model is not None

    # ------------------------------------------------------------------
    #  Detection
    # ------------------------------------------------------------------
    def read_image(self, fresh: bool = True) -> np.ndarray | None:
        """Read one frame from the camera as an unnormalised greyscale image.

        Args:
            fresh: Throw away buffered frames first, so the image is current.

        Returns:
            Single-channel uint8 image, or ``None`` if the camera delivers nothing.
        """
        with self._camera_lock:
            if self._cap is None:
                return None
            if fresh:
                for _ in range(FLUSH_FRAMES):
                    self._cap.grab()
            ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return to_display(frame, normalize=False)

    def detect(self) -> list[dict] | None:
        """Grab a fresh frame and detect the supports in it.

        Returns:
            The detections (see :func:`detect`), or ``None`` if no frame could be read.
        """
        image = self.read_image(fresh=True)
        if image is None:
            logger.error("No frame received from the camera")
            return None
        return self.analyse(image)

    def analyse(self, image: np.ndarray) -> list[dict]:
        """Run the detection on one frame, print, save and publish the result.

        Args:
            image: Single-channel uint8 image to analyse.

        Returns:
            The detections, as returned by :func:`detect`.
        """
        started = time.perf_counter()
        detections = detect(self._model, image)
        elapsed_ms = (time.perf_counter() - started) * 1000

        image_name = ""
        keep = bool(detections) or self.keep_empty
        if self._folder is not None and keep:
            raw_path = next_image_path(self._folder)
            cv2.imwrite(str(raw_path), image)
            cv2.imwrite(str(raw_path.with_name(f"{raw_path.stem}_det.png")), annotate(image, detections))
            image_name = raw_path.name

        logger.info(f"{image_name or 'frame'}: {len(detections)} support(s) in {elapsed_ms:.0f} ms")
        if keep:
            print_detections(detections)
        self.last_detections = detections
        self._publish(detections, image_name)
        return detections

    def _publish(self, detections: list[dict], image_name: str) -> None:
        if not self._broker_connected:
            return
        payload = {
            "time": time.time(),
            "count": len(detections),
            "image": image_name,
            "detections": detections,
        }
        self._client.publish(TOPIC_DETECTIONS, json.dumps(payload), qos=1)

    # ------------------------------------------------------------------
    #  MQTT callbacks
    # ------------------------------------------------------------------
    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code.is_failure:
            logger.error(f"Broker refused the connection: {reason_code}")
            return
        self._broker_connected = True
        logger.info("Vision: connected to the MQTT broker.")
        client.publish(TOPIC_STATUS, "online", qos=1, retain=True)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        self._broker_connected = False
        if reason_code.is_failure:
            logger.warning(f"Lost the broker ({reason_code}), reconnecting...")


def run_viewer(detector: StiltDetector) -> int:
    """Show the camera frames and detect on demand until the user quits.

    Args:
        detector: Connected detector.

    Returns:
        Process exit code (0 on success).
    """
    logger.info("Keys: SPACE detect | c clear boxes | n normalise preview | q/ESC quit")

    window = f"IR - /dev/video{detector.camera_index}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    normalize = False
    detections: list[dict] = []
    last_frame = time.perf_counter()

    try:
        while True:
            # Detection always runs on the unnormalised image, so it sees the
            # same pixel values the training data was saved with.
            image = detector.read_image(fresh=False)
            now = time.perf_counter()
            if image is None:
                if now - last_frame > NO_FRAME_TIMEOUT_S:
                    logger.error("No frames received - camera unplugged?")
                    return 1
                continue
            last_frame = now

            preview = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX) if normalize else image
            # The boxes of the last capture stay on screen until the next one.
            cv2.imshow(window, annotate(preview, detections))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                detections = detector.analyse(image)
            elif key == ord("c"):
                detections = []
            elif key == ord("n"):
                normalize = not normalize
                logger.info(f"Preview normalisation {'on' if normalize else 'off'}")
            if not window_open(window):
                break
    finally:
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    """Entry point.

    Returns:
        Process exit code.
    """
    detector = StiltDetector(client_id="stilt_detector_viewer")
    try:
        detector.connect()
    except RuntimeError as error:
        logger.error(error)
        return 1
    try:
        return run_viewer(detector)
    finally:
        detector.disconnect()


if __name__ == "__main__":
    sys.exit(main())
