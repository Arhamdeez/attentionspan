"""
Gaze Attention Detector — reveal an image when you look away from the screen.

Usage:
    python main.py [--image path/to/reveal.png] [--frames 30]

- Camera runs; when you look at the camera = "attention".
- When you look away for N consecutive frames, the image is shown in a separate window.
- As soon as you look back, the window closes.
- Press 'q' to quit.
"""
# Reduce noisy logs from MediaPipe / TensorFlow Lite / protobuf
import os
import warnings

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", message=".*GetPrototype.*", category=UserWarning)

import argparse
import pathlib
import sys

import cv2
import numpy as np

from gaze_detector import GazeDetector, GazeState


# Default image to reveal when looking away (create a simple placeholder if missing)
DEFAULT_IMAGE_PATH = pathlib.Path(__file__).parent / "assets" / "reveal.png"
LOOK_AWAY_FRAMES_TO_TRIGGER = 6  # ~0.2 sec at 30fps; use --frames to tune
WINDOW_MAIN = "Gaze Attention — look away to reveal"


def make_placeholder_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a placeholder image if no file is provided."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 40, 60)
    cv2.putText(
        img,
        "You looked away!",
        (width // 2 - 150, height // 2 - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        "Put your image at assets/reveal.png",
        (width // 2 - 200, height // 2 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 200),
        1,
    )
    return img


def load_reveal_image(path: pathlib.Path) -> np.ndarray:
    if path.exists():
        img = cv2.imread(str(path))
        if img is not None:
            return img
    return make_placeholder_image()


def run(
    reveal_image_path: pathlib.Path = DEFAULT_IMAGE_PATH,
    look_away_frames: int = LOOK_AWAY_FRAMES_TO_TRIGGER,
    camera_id: int = 0,
    yaw_threshold_deg: float = 25.0,
    pitch_threshold_deg: float = 25.0,
) -> None:
    reveal_image = load_reveal_image(reveal_image_path)
    detector = GazeDetector(
        look_away_threshold=0.25,
        yaw_threshold_deg=yaw_threshold_deg,
        pitch_threshold_deg=pitch_threshold_deg,
        smoothing_frames=5,
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open webcam.", file=sys.stderr)
        sys.exit(1)

    consecutive_look_away = 0
    reveal_visible = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gaze = detector.process_frame(frame)

            # Draw status on camera frame
            if gaze.face_detected:
                status = "LOOKING AWAY" if not gaze.looking_at_screen else "ATTENTION"
                color = (0, 0, 255) if not gaze.looking_at_screen else (0, 255, 0)
                cv2.putText(
                    frame,
                    status,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"yaw={getattr(gaze, 'yaw_deg', 0):.0f} pitch={getattr(gaze, 'pitch_deg', 0):.0f}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )
            else:
                cv2.putText(
                    frame,
                    "No face",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (100, 100, 100),
                    2,
                )

            # Trigger: only count stable look-away (smoothed) so the image doesn’t flash randomly.
            # Hide as soon as current frame says you’re looking back.
            if gaze.face_detected and gaze.looking_away_stable:
                consecutive_look_away += 1
                if consecutive_look_away >= look_away_frames:
                    reveal_visible = True
            else:
                consecutive_look_away = 0
                if gaze.looking_at_screen or not gaze.face_detected:
                    reveal_visible = False

            # Single window: show reveal image or camera (avoids two dock icons on macOS)
            if reveal_visible:
                cv2.imshow(WINDOW_MAIN, reveal_image)
            else:
                cv2.imshow(WINDOW_MAIN, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reveal an image when you look away from the camera."
    )
    parser.add_argument(
        "--image",
        type=pathlib.Path,
        default=DEFAULT_IMAGE_PATH,
        help="Path to image to show when looking away",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=LOOK_AWAY_FRAMES_TO_TRIGGER,
        help="Consecutive look-away frames needed to trigger reveal",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=25.0,
        metavar="DEG",
        help="Head turn (left/right) threshold in degrees; below = attention (default 25)",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=25.0,
        metavar="DEG",
        help="Head tilt (up/down) threshold in degrees; below = attention (default 25)",
    )
    args = parser.parse_args()
    run(
        reveal_image_path=args.image,
        look_away_frames=args.frames,
        camera_id=args.camera,
        yaw_threshold_deg=args.yaw,
        pitch_threshold_deg=args.pitch,
    )


if __name__ == "__main__":
    main()
