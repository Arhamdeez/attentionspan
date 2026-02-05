"""
Gaze / attention detection using MediaPipe Face Mesh.
Detects when the user is "looking at screen" vs "looking away" using head pose (yaw/pitch).
"""
import math
from dataclasses import dataclass
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


# MediaPipe Face Mesh landmark indices (468 landmarks)
# Used for head pose (solvePnP) and fallback nose-offset
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_LEFT = 33
RIGHT_EYE_RIGHT = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
LEFT_FACE = 234
RIGHT_FACE = 454
FOREHEAD = 10

# Generic 3D face model for head pose (same order as landmark indices below)
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),           # nose tip
    (0.0, -330.0, -65.0),      # chin
    (-225.0, 170.0, -135.0),   # left eye left
    (225.0, 170.0, -135.0),    # right eye right
    (-150.0, -150.0, -125.0),  # left mouth
    (150.0, -150.0, -125.0),   # right mouth
], dtype=np.float64)

# Landmark indices matching FACE_3D_MODEL order
POSE_LANDMARK_INDICES = [NOSE_TIP, CHIN, LEFT_EYE_LEFT, RIGHT_EYE_RIGHT, LEFT_MOUTH, RIGHT_MOUTH]


@dataclass
class GazeState:
    """Result of gaze analysis for one frame."""
    looking_at_screen: bool   # current frame: for hiding reveal instantly
    looking_away_stable: bool # smoothed: for triggering reveal (avoids random flashes)
    nose_offset_x: float
    nose_offset_y: float
    face_detected: bool
    yaw_deg: float = 0.0   # left-right head turn (negative = left, positive = right)
    pitch_deg: float = 0.0 # up-down (negative = up, positive = down)


class GazeDetector:
    """
    Uses head pose (yaw/pitch) from face landmarks to detect "looking at camera" vs "looking away".
    Falls back to nose-offset heuristic if pose estimation fails.
    """

    def __init__(
        self,
        look_away_threshold: float = 0.25,
        yaw_threshold_deg: float = 25.0,
        pitch_threshold_deg: float = 25.0,
        smoothing_frames: int = 5,
        min_confidence: float = 0.5,
    ):
        self.look_away_threshold = look_away_threshold
        self.yaw_threshold_deg = yaw_threshold_deg
        self.pitch_threshold_deg = pitch_threshold_deg
        self.smoothing_frames = smoothing_frames
        self.min_confidence = min_confidence
        self._offset_history: deque[tuple[float, float]] = deque(maxlen=smoothing_frames)
        self._yaw_history: deque[float] = deque(maxlen=smoothing_frames)
        self._pitch_history: deque[float] = deque(maxlen=smoothing_frames)

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )

    def _get_landmark(self, landmarks, idx: int) -> tuple[float, float]:
        lm = landmarks.landmark[idx]
        return (lm.x, lm.y)

    def _get_landmark_xy_px(self, landmarks, idx: int, w: int, h: int) -> tuple[float, float]:
        lm = landmarks.landmark[idx]
        return (lm.x * w, lm.y * h)

    def _rotation_matrix_to_euler_deg(self, R: np.ndarray) -> tuple[float, float, float]:
        """Extract yaw (z), pitch (y), roll (x) in degrees from rotation matrix."""
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            yaw = math.atan2(R[1, 0], R[0, 0])
            pitch = math.atan2(-R[2, 0], sy)
            roll = math.atan2(R[2, 1], R[2, 2])
        else:
            yaw = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            roll = 0.0
        return (
            math.degrees(yaw),
            math.degrees(pitch),
            math.degrees(roll),
        )

    def _estimate_head_pose(self, face_landmarks, w: int, h: int) -> tuple[float, float] | None:
        """Estimate yaw and pitch in degrees. Returns (yaw_deg, pitch_deg) or None if failed."""
        image_points = np.array([
            self._get_landmark_xy_px(face_landmarks, i, w, h)
            for i in POSE_LANDMARK_INDICES
        ], dtype=np.float64)

        # Camera matrix (approximate; focal length ~ width for typical webcam FOV)
        fx = w
        fy = w
        cx = w / 2.0
        cy = h / 2.0
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        R, _ = cv2.Rodrigues(rvec)
        yaw_deg, pitch_deg, _ = self._rotation_matrix_to_euler_deg(R)
        return (yaw_deg, pitch_deg)

    def process_frame(self, frame: np.ndarray) -> GazeState:
        """
        Analyze one BGR frame. Returns gaze state.
        Looking at screen = small yaw and pitch (face toward camera). Looking away = head turned.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._offset_history.clear()
            self._yaw_history.clear()
            self._pitch_history.clear()
            return GazeState(
                looking_at_screen=True,
                looking_away_stable=False,
                nose_offset_x=0.0,
                nose_offset_y=0.0,
                face_detected=False,
            )

        face_landmarks = results.multi_face_landmarks[0]

        # 1) Head pose (yaw/pitch) — primary signal
        pose = self._estimate_head_pose(face_landmarks, w, h)

        if pose is not None:
            yaw_deg, pitch_deg = pose
            self._yaw_history.append(yaw_deg)
            self._pitch_history.append(pitch_deg)
            avg_yaw = sum(self._yaw_history) / len(self._yaw_history)
            avg_pitch = sum(self._pitch_history) / len(self._pitch_history)

            # Current frame (no smoothing): for hiding reveal as soon as you look back
            looking_at_screen = (
                abs(yaw_deg) < self.yaw_threshold_deg
                and abs(pitch_deg) < self.pitch_threshold_deg
            )
            # Smoothed: for triggering reveal only after sustained look-away (stops random flashes)
            looking_away_stable = not (
                abs(avg_yaw) < self.yaw_threshold_deg
                and abs(avg_pitch) < self.pitch_threshold_deg
            )

            # Nose offset for display (use smoothed for stable on-screen numbers)
            nose_offset_x = -avg_yaw / 90.0
            nose_offset_y = -avg_pitch / 90.0
            self._offset_history.append((nose_offset_x, nose_offset_y))

            return GazeState(
                looking_at_screen=looking_at_screen,
                looking_away_stable=looking_away_stable,
                nose_offset_x=nose_offset_x,
                nose_offset_y=nose_offset_y,
                face_detected=True,
                yaw_deg=avg_yaw,
                pitch_deg=avg_pitch,
            )

        # 2) Fallback: nose offset from face center (original heuristic)
        nose = self._get_landmark(face_landmarks, NOSE_TIP)
        left_face = self._get_landmark(face_landmarks, LEFT_FACE)
        right_face = self._get_landmark(face_landmarks, RIGHT_FACE)
        forehead = self._get_landmark(face_landmarks, FOREHEAD)

        face_center_x = (left_face[0] + right_face[0]) / 2
        face_center_y = (nose[1] + forehead[1]) / 2

        nose_offset_x = (nose[0] - face_center_x) * 2
        nose_offset_y = (nose[1] - face_center_y) * 2

        self._offset_history.append((nose_offset_x, nose_offset_y))
        avg_x = sum(o[0] for o in self._offset_history) / len(self._offset_history)
        avg_y = sum(o[1] for o in self._offset_history) / len(self._offset_history)

        # Current frame: for instant hide when looking back
        magnitude_now = math.hypot(nose_offset_x, nose_offset_y)
        looking_at_screen = magnitude_now < self.look_away_threshold
        # Smoothed: for trigger (avoid random flashes)
        magnitude_avg = math.hypot(avg_x, avg_y)
        looking_away_stable = magnitude_avg >= self.look_away_threshold

        return GazeState(
            looking_at_screen=looking_at_screen,
            looking_away_stable=looking_away_stable,
            nose_offset_x=avg_x,
            nose_offset_y=avg_y,
            face_detected=True,
        )

    def close(self) -> None:
        self._face_mesh.close()
