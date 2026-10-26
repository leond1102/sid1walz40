"""Eigenständiger Test für die Arducam: zeigt den Videofeed und meldet wiederholte Bilder.

Aufruf:
    python scripts/arducam/arducam_test.py                  # Arducam automatisch suchen, MJPG 1280x720
    python scripts/arducam/arducam_test.py --device 2       # /dev/video2
    python scripts/arducam/arducam_test.py --fourcc YUYV --width 640 --height 480
    python scripts/arducam/arducam_test.py --headless       # ohne Fenster, nur Statistik

Tasten im Fenster: q / Esc = beenden, s = Screenshot speichern.
"""

import argparse
import hashlib
import time
from pathlib import Path

import cv2


def find_arducam() -> int | None:
    """Erster /dev/videoN, dessen Name 'arducam' enthält (das ist der Bild-Knoten)."""
    for entry in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:])):
        try:
            if "arducam" in (entry / "name").read_text().lower():
                return int(entry.name[5:])
        except (OSError, ValueError):
            continue
    return None


def frame_hash(frame) -> str:
    return hashlib.md5(frame.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, help="Nummer von /dev/videoN (Standard: Arducam suchen)")
    parser.add_argument("--fourcc", default="MJPG", help="MJPG oder YUYV (Standard: MJPG)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--headless", action="store_true", help="kein Fenster, nur Ausgabe im Terminal")
    parser.add_argument("--seconds", type=float, default=0, help="nach N Sekunden beenden (0 = endlos)")
    args = parser.parse_args()

    index = args.device if args.device is not None else find_arducam()
    if index is None:
        raise SystemExit("Keine Arducam gefunden (v4l2-ctl --list-devices prüfen) – mit --device N angeben.")

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"/dev/video{index} lässt sich nicht öffnen (belegt? fuser -v /dev/video{index})")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("ascii", "replace")
    print(f"/dev/video{index}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"{fourcc} @ {cap.get(cv2.CAP_PROP_FPS):.0f} FPS (Backend {cap.getBackendName()})")

    seen: dict[str, int] = {}  # Hash -> Bildnummer, um Wiederholungen zu erkennen
    count = repeats = failures = 0
    started = last_report = time.monotonic()
    fps_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            now = time.monotonic()
            if not ok or frame is None:
                failures += 1
                print(f"read() fehlgeschlagen ({failures}x)")
                if failures >= 10:
                    raise SystemExit("Kamera liefert keine Bilder mehr.")
                time.sleep(0.1)
                continue
            failures = 0
            count += 1
            fps_frames += 1

            digest = frame_hash(frame)
            if digest in seen:
                repeats += 1
                if repeats <= 20 or repeats % 100 == 0:
                    print(f"WIEDERHOLUNG: Bild #{count} ist identisch mit Bild #{seen[digest]} "
                          f"(vor {count - seen[digest]} Bildern) – insgesamt {repeats}")
            seen[digest] = count
            if len(seen) > 2000:
                seen.clear()

            if now - last_report >= 2.0:
                print(f"{count} Bilder, {fps_frames / (now - last_report):.1f} FPS, {repeats} Wiederholungen")
                last_report, fps_frames = now, 0

            if not args.headless:
                cv2.putText(frame, f"#{count}  {time.strftime('%H:%M:%S')}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow(f"Arducam /dev/video{index}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    name = f"arducam_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(name, frame)
                    print(f"Gespeichert: {name}")

            if args.seconds and now - started >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        elapsed = time.monotonic() - started
        print(f"Fertig: {count} Bilder in {elapsed:.1f} s ({count / max(elapsed, 1e-6):.1f} FPS), "
              f"{repeats} Wiederholungen")


if __name__ == "__main__":
    main()
