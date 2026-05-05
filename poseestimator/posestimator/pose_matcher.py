from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
POSE_VISIBILITY_THRESHOLD = 0.45
NON_POSE_VISIBILITY_THRESHOLD = 0.01
MIN_MATCH_WEIGHT = 0.40
MIN_ANGLE_MATCHES = 3
POSE_ANGLE_BLEND = 0.48
FACE_EXPRESSION_WEIGHT = 0.30
MOUTH_SHAPE_WEIGHT = 0.24
LIVE_PROCESS_MAX_EDGE = 640
LIVE_FACE_REFRESH_INTERVAL = 4
LIVE_HAND_CROP_INTERVAL = 3
LIVE_FACE_SCORE_THRESHOLD = 135.0
LIVE_HAND_SCORE_THRESHOLD = 96.0
FONT = cv2.FONT_HERSHEY_SIMPLEX

mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def holistic_landmark_value(*names: str) -> int:
    for name in names:
        landmark = getattr(mp_holistic.PoseLandmark, name, None)
        if landmark is not None:
            return landmark.value

    raise AttributeError(f"Pose landmark not found: {', '.join(names)}")


POSE_LEFT_RIGHT_PAIRS = [
    (holistic_landmark_value("LEFT_EYE_INNER"), holistic_landmark_value("RIGHT_EYE_INNER")),
    (holistic_landmark_value("LEFT_EYE"), holistic_landmark_value("RIGHT_EYE")),
    (holistic_landmark_value("LEFT_EYE_OUTER"), holistic_landmark_value("RIGHT_EYE_OUTER")),
    (holistic_landmark_value("LEFT_EAR"), holistic_landmark_value("RIGHT_EAR")),
    (
        holistic_landmark_value("LEFT_MOUTH", "MOUTH_LEFT"),
        holistic_landmark_value("RIGHT_MOUTH", "MOUTH_RIGHT"),
    ),
    (holistic_landmark_value("LEFT_SHOULDER"), holistic_landmark_value("RIGHT_SHOULDER")),
    (holistic_landmark_value("LEFT_ELBOW"), holistic_landmark_value("RIGHT_ELBOW")),
    (holistic_landmark_value("LEFT_WRIST"), holistic_landmark_value("RIGHT_WRIST")),
    (holistic_landmark_value("LEFT_PINKY"), holistic_landmark_value("RIGHT_PINKY")),
    (holistic_landmark_value("LEFT_INDEX"), holistic_landmark_value("RIGHT_INDEX")),
    (holistic_landmark_value("LEFT_THUMB"), holistic_landmark_value("RIGHT_THUMB")),
    (holistic_landmark_value("LEFT_HIP"), holistic_landmark_value("RIGHT_HIP")),
    (holistic_landmark_value("LEFT_KNEE"), holistic_landmark_value("RIGHT_KNEE")),
    (holistic_landmark_value("LEFT_ANKLE"), holistic_landmark_value("RIGHT_ANKLE")),
    (holistic_landmark_value("LEFT_HEEL"), holistic_landmark_value("RIGHT_HEEL")),
    (holistic_landmark_value("LEFT_FOOT_INDEX"), holistic_landmark_value("RIGHT_FOOT_INDEX")),
]

FACE_KEYPOINT_INDEXES = sorted(
    {index for connection in mp_face_mesh.FACEMESH_CONTOURS for index in connection}
)
MOUTH_KEYPOINT_INDEXES = sorted(
    {index for connection in mp_face_mesh.FACEMESH_LIPS for index in connection}
)
LEFT_EYE_INDEXES = sorted(
    {index for connection in mp_face_mesh.FACEMESH_LEFT_EYE for index in connection}
)
RIGHT_EYE_INDEXES = sorted(
    {index for connection in mp_face_mesh.FACEMESH_RIGHT_EYE for index in connection}
)
LEFT_EYEBROW_INDEXES = sorted(
    {index for connection in mp_face_mesh.FACEMESH_LEFT_EYEBROW for index in connection}
)
RIGHT_EYEBROW_INDEXES = sorted(
    {index for connection in mp_face_mesh.FACEMESH_RIGHT_EYEBROW for index in connection}
)


@dataclass(frozen=True)
class GroupConfig:
    label: str
    weight: float
    min_points: int
    local_ratio: float
    visibility_threshold: float


FACE_GROUP = GroupConfig(
    label="Yuz",
    weight=0.16,
    min_points=36,
    local_ratio=0.72,
    visibility_threshold=NON_POSE_VISIBILITY_THRESHOLD,
)
LEFT_HAND_GROUP = GroupConfig(
    label="Sol el",
    weight=0.27,
    min_points=8,
    local_ratio=0.70,
    visibility_threshold=NON_POSE_VISIBILITY_THRESHOLD,
)
RIGHT_HAND_GROUP = GroupConfig(
    label="Sag el",
    weight=0.27,
    min_points=8,
    local_ratio=0.70,
    visibility_threshold=NON_POSE_VISIBILITY_THRESHOLD,
)
POSE_GROUP = GroupConfig(
    label="Poz",
    weight=0.30,
    min_points=12,
    local_ratio=0.0,
    visibility_threshold=POSE_VISIBILITY_THRESHOLD,
)

FACE_EXPRESSION_INDEXES = {
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_lip": 13,
    "lower_lip": 14,
    "chin": 152,
    "nose_tip": 1,
}

POSE_ANGLE_TRIPLETS = [
    (
        "Sol omuz",
        mp_holistic.PoseLandmark.LEFT_ELBOW.value,
        mp_holistic.PoseLandmark.LEFT_SHOULDER.value,
        mp_holistic.PoseLandmark.LEFT_HIP.value,
    ),
    (
        "Sag omuz",
        mp_holistic.PoseLandmark.RIGHT_ELBOW.value,
        mp_holistic.PoseLandmark.RIGHT_SHOULDER.value,
        mp_holistic.PoseLandmark.RIGHT_HIP.value,
    ),
    (
        "Sol dirsek",
        mp_holistic.PoseLandmark.LEFT_SHOULDER.value,
        mp_holistic.PoseLandmark.LEFT_ELBOW.value,
        mp_holistic.PoseLandmark.LEFT_WRIST.value,
    ),
    (
        "Sag dirsek",
        mp_holistic.PoseLandmark.RIGHT_SHOULDER.value,
        mp_holistic.PoseLandmark.RIGHT_ELBOW.value,
        mp_holistic.PoseLandmark.RIGHT_WRIST.value,
    ),
    (
        "Sol kalca",
        mp_holistic.PoseLandmark.LEFT_SHOULDER.value,
        mp_holistic.PoseLandmark.LEFT_HIP.value,
        mp_holistic.PoseLandmark.LEFT_KNEE.value,
    ),
    (
        "Sag kalca",
        mp_holistic.PoseLandmark.RIGHT_SHOULDER.value,
        mp_holistic.PoseLandmark.RIGHT_HIP.value,
        mp_holistic.PoseLandmark.RIGHT_KNEE.value,
    ),
    (
        "Sol diz",
        mp_holistic.PoseLandmark.LEFT_HIP.value,
        mp_holistic.PoseLandmark.LEFT_KNEE.value,
        mp_holistic.PoseLandmark.LEFT_ANKLE.value,
    ),
    (
        "Sag diz",
        mp_holistic.PoseLandmark.RIGHT_HIP.value,
        mp_holistic.PoseLandmark.RIGHT_KNEE.value,
        mp_holistic.PoseLandmark.RIGHT_ANKLE.value,
    ),
    (
        "Sol bilek",
        mp_holistic.PoseLandmark.LEFT_ELBOW.value,
        mp_holistic.PoseLandmark.LEFT_WRIST.value,
        mp_holistic.PoseLandmark.LEFT_INDEX.value,
    ),
    (
        "Sag bilek",
        mp_holistic.PoseLandmark.RIGHT_ELBOW.value,
        mp_holistic.PoseLandmark.RIGHT_WRIST.value,
        mp_holistic.PoseLandmark.RIGHT_INDEX.value,
    ),
]

HAND_STATE_BLEND = 0.82
HAND_BINARY_WEIGHT = 0.52
HAND_COUNT_WEIGHT = 0.26
HAND_FEATURE_MIN_MATCHES = 3
HAND_STATE_OPEN_THRESHOLD = 0.60
HAND_MISSING_PENALTY = 0.42
HAND_ACTIVE_WEIGHT_MULTIPLIER = 2.35
HAND_PRIMARY_WEIGHT_MULTIPLIER = 4.20
HAND_PRIMARY_MISSING_PENALTY = 0.86
HAND_PRIMARY_POSE_SCALE = 0.74
HAND_PRIMARY_FACE_SCALE = 0.18
HAND_PRIMARY_EXPRESSION_SCALE = 0.12
HAND_PRIMARY_MOUTH_SCALE = 0.10
HAND_COUNT_PRIORITY_WEIGHT = 0.95
FACE_FALLBACK_POSE_SCALE = 0.88
FACE_FALLBACK_FACE_SCALE = 1.20
FACE_FALLBACK_EXPRESSION_SCALE = 1.28
FACE_FALLBACK_MOUTH_SCALE = 1.40
HAND_PALM_ANCHORS = [5, 9, 13, 17]
REFERENCE_VARIANT_MAX_EDGE = 2200
HAND_FINGER_TRIPLETS = [
    ("Basparmak", 1, 2, 4, 2, 4),
    ("Isaret", 5, 6, 8, 5, 8),
    ("Orta", 9, 10, 12, 9, 12),
    ("Yuzuk", 13, 14, 16, 13, 16),
    ("Serce", 17, 18, 20, 17, 20),
]


@dataclass(frozen=True)
class LandmarkGroup:
    config: GroupConfig
    raw: np.ndarray
    global_normalized: np.ndarray
    local_normalized: np.ndarray
    valid: np.ndarray
    gesture_features: np.ndarray | None = None
    gesture_valid: np.ndarray | None = None


@dataclass(frozen=True)
class FeatureSet:
    pose: LandmarkGroup | None
    face: LandmarkGroup | None
    left_hand: LandmarkGroup | None
    right_hand: LandmarkGroup | None
    pose_angles: np.ndarray | None
    pose_angle_valid: np.ndarray | None
    face_mesh_points: np.ndarray | None
    face_expression_features: np.ndarray | None
    face_expression_valid: np.ndarray | None
    mouth_shape_features: np.ndarray | None
    mouth_shape_valid: np.ndarray | None

    def has_pose(self) -> bool:
        return self.pose is not None


@dataclass(frozen=True)
class ReferencePose:
    name: str
    path: Path
    image: np.ndarray
    features: FeatureSet
    preview_image: np.ndarray
    skeleton_image: np.ndarray


@dataclass(frozen=True)
class Match:
    reference: ReferencePose
    distance: float
    mirrored: bool
    component_scores: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kameradaki pozu, elleri ve yuz seklini poses klasorundeki gorsellerle eslestirir."
    )
    parser.add_argument(
        "--poses",
        type=Path,
        default=Path("poses"),
        help="Referans fotograflarin bulundugu klasor.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Kamera indeksi.")
    parser.add_argument("--width", type=int, default=960, help="Kamera genisligi.")
    parser.add_argument("--height", type=int, default=540, help="Kamera yuksekligi.")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Kamera goruntusunu ayna gibi cevirmeyi kapatir.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.34,
        help="Bu degerin ustundeki eslesmeler zayif olarak isaretlenir.",
    )
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None

    if data.size == 0:
        return None

    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def iter_pose_images(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def landmarks_to_array(
    landmarks,
    indexes: list[int] | None = None,
    use_visibility: bool = False,
) -> np.ndarray | None:
    if landmarks is None:
        return None

    source = landmarks.landmark
    selected = source if indexes is None else [source[index] for index in indexes]
    rows = []
    for landmark in selected:
        visibility = float(getattr(landmark, "visibility", 1.0)) if use_visibility else 1.0
        rows.append([landmark.x, landmark.y, landmark.z, float(visibility)])
    return np.asarray(rows, dtype=np.float32)


def center_point(landmarks: np.ndarray, left: int, right: int) -> np.ndarray:
    return (landmarks[left, :3] + landmarks[right, :3]) * 0.5


def compute_body_anchor(pose_landmarks: np.ndarray | None) -> tuple[np.ndarray, float]:
    if pose_landmarks is None:
        return np.zeros(3, dtype=np.float32), 1.0

    valid = pose_landmarks[:, 3] >= POSE_VISIBILITY_THRESHOLD
    left_hip = mp_holistic.PoseLandmark.LEFT_HIP.value
    right_hip = mp_holistic.PoseLandmark.RIGHT_HIP.value
    left_shoulder = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
    right_shoulder = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value

    hip_center = center_point(pose_landmarks, left_hip, right_hip)
    shoulder_center = center_point(pose_landmarks, left_shoulder, right_shoulder)

    anchor_points = [hip_center, shoulder_center]
    visible_points = pose_landmarks[valid, :3]
    if visible_points.size:
        anchor_points.append(np.mean(visible_points, axis=0))

    center = np.mean(np.asarray(anchor_points), axis=0)

    shoulder_width = np.linalg.norm(
        pose_landmarks[left_shoulder, :3] - pose_landmarks[right_shoulder, :3]
    )
    hip_width = np.linalg.norm(pose_landmarks[left_hip, :3] - pose_landmarks[right_hip, :3])
    torso_height = np.linalg.norm(shoulder_center - hip_center)

    if visible_points.size:
        span = np.max(visible_points, axis=0) - np.min(visible_points, axis=0)
        body_span = float(np.linalg.norm(span[:2]))
    else:
        body_span = 0.0

    scale = max(float(shoulder_width), float(hip_width), float(torso_height), body_span, 1e-6)
    return center.astype(np.float32), scale


def compute_local_anchor(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    visible_points = points[valid, :3]
    if not visible_points.size:
        center = np.mean(points[:, :3], axis=0)
        return center.astype(np.float32), 1.0

    center = np.mean(visible_points, axis=0)
    span = np.max(visible_points, axis=0) - np.min(visible_points, axis=0)
    radial = np.linalg.norm(visible_points[:, :2] - center[:2], axis=1)
    scale = max(float(np.linalg.norm(span[:2])), float(np.max(radial) * 2.0), 1e-6)
    return center.astype(np.float32), scale


def normalize_points(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    normalized = points.copy()
    normalized[:, :3] = (normalized[:, :3] - center) / scale
    return normalized


def angle_between_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = a - b
    cb = c - b
    denom = np.linalg.norm(ab) * np.linalg.norm(cb)
    if denom <= 1e-6:
        return 0.0

    cosine = float(np.dot(ab, cb) / denom)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.arccos(cosine))


def compute_pose_angles(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angles: list[float] = []
    angle_valid: list[bool] = []

    for _, first, center, third in POSE_ANGLE_TRIPLETS:
        triplet_valid = bool(valid[first] and valid[center] and valid[third])
        angle_valid.append(triplet_valid)
        if triplet_valid:
            angles.append(angle_between_points(points[first, :3], points[center, :3], points[third, :3]))
        else:
            angles.append(0.0)

    return np.asarray(angles, dtype=np.float32), np.asarray(angle_valid, dtype=bool)


def mean_point_from_indexes(
    points: np.ndarray,
    valid: np.ndarray,
    indexes: list[int],
) -> np.ndarray | None:
    selected = [index for index in indexes if index < len(points) and valid[index]]
    if not selected:
        return None
    return np.mean(points[selected, :2], axis=0).astype(np.float32)


def vertical_span_from_indexes(
    points: np.ndarray,
    valid: np.ndarray,
    indexes: list[int],
) -> float | None:
    selected = [index for index in indexes if index < len(points) and valid[index]]
    if not selected:
        return None
    y_values = points[selected, 1]
    return float(np.max(y_values) - np.min(y_values))


def extreme_point_from_indexes(
    points: np.ndarray,
    valid: np.ndarray,
    indexes: list[int],
    axis: int,
    pick_max: bool,
) -> np.ndarray | None:
    selected = [index for index in indexes if index < len(points) and valid[index]]
    if not selected:
        return None

    selected_points = points[selected, :2]
    values = selected_points[:, axis]
    choice = int(np.argmax(values) if pick_max else np.argmin(values))
    return selected_points[choice].astype(np.float32)


def compute_face_expression_features(
    points: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    indexes = FACE_EXPRESSION_INDEXES
    needed = list(indexes.values())
    if not all(index < len(points) and valid[index] for index in needed):
        return None, None

    left_eye_outer = points[indexes["left_eye_outer"], :2]
    right_eye_outer = points[indexes["right_eye_outer"], :2]
    mouth_left = points[indexes["mouth_left"], :2]
    mouth_right = points[indexes["mouth_right"], :2]
    upper_lip = points[indexes["upper_lip"], :2]
    lower_lip = points[indexes["lower_lip"], :2]
    chin = points[indexes["chin"], :2]
    nose_tip = points[indexes["nose_tip"], :2]
    left_eye_center = mean_point_from_indexes(points, valid, LEFT_EYE_INDEXES)
    right_eye_center = mean_point_from_indexes(points, valid, RIGHT_EYE_INDEXES)
    left_brow_center = mean_point_from_indexes(points, valid, LEFT_EYEBROW_INDEXES)
    right_brow_center = mean_point_from_indexes(points, valid, RIGHT_EYEBROW_INDEXES)
    left_brow_inner = extreme_point_from_indexes(points, valid, LEFT_EYEBROW_INDEXES, axis=0, pick_max=True)
    right_brow_inner = extreme_point_from_indexes(
        points,
        valid,
        RIGHT_EYEBROW_INDEXES,
        axis=0,
        pick_max=False,
    )
    left_eye_open = vertical_span_from_indexes(points, valid, LEFT_EYE_INDEXES)
    right_eye_open = vertical_span_from_indexes(points, valid, RIGHT_EYE_INDEXES)

    if any(
        value is None
        for value in (
            left_eye_center,
            right_eye_center,
            left_brow_center,
            right_brow_center,
            left_brow_inner,
            right_brow_inner,
            left_eye_open,
            right_eye_open,
        )
    ):
        return None, None

    eye_width = max(float(np.linalg.norm(left_eye_outer - right_eye_outer)), 1e-6)
    eye_center = (left_eye_center + right_eye_center) * 0.5
    face_height = max(float(np.linalg.norm(eye_center - chin)), 1e-6)
    mouth_width = float(np.linalg.norm(mouth_left - mouth_right))
    mouth_open = float(np.linalg.norm(upper_lip - lower_lip))
    mouth_center = (upper_lip + lower_lip) * 0.5
    lip_center_y = float(mouth_center[1])
    corner_center_y = float((mouth_left[1] + mouth_right[1]) * 0.5)
    eye_open = float((left_eye_open + right_eye_open) * 0.5)
    brow_raise = float(
        ((left_eye_center[1] - left_brow_center[1]) + (right_eye_center[1] - right_brow_center[1]))
        * 0.5
    )
    brow_gap = float(np.linalg.norm(left_brow_inner - right_brow_inner))
    mouth_drop = float(np.linalg.norm(mouth_center - nose_tip))

    features = np.asarray(
        [
            np.clip(mouth_open / (eye_width * 0.34), 0.0, 1.0),
            np.clip(mouth_width / (eye_width * 1.12), 0.0, 1.0),
            np.clip(0.5 + (lip_center_y - corner_center_y) / (face_height * 0.16), 0.0, 1.0),
            np.clip(eye_open / (eye_width * 0.15), 0.0, 1.0),
            np.clip(brow_raise / (face_height * 0.18), 0.0, 1.0),
            np.clip(brow_gap / (eye_width * 0.72), 0.0, 1.0),
            np.clip(mouth_drop / (face_height * 0.42), 0.0, 1.0),
            np.clip(
                0.72 * np.clip(mouth_open / (eye_width * 0.34), 0.0, 1.0)
                + 0.28 * np.clip(mouth_drop / (face_height * 0.42), 0.0, 1.0),
                0.0,
                1.0,
            ),
            np.clip(
                0.62 * np.clip(0.5 + (lip_center_y - corner_center_y) / (face_height * 0.16), 0.0, 1.0)
                + 0.38 * np.clip(mouth_width / (eye_width * 1.12), 0.0, 1.0),
                0.0,
                1.0,
            ),
            np.clip(
                0.78 * (1.0 - np.clip(0.5 + (lip_center_y - corner_center_y) / (face_height * 0.16), 0.0, 1.0))
                + 0.22 * (1.0 - np.clip(mouth_width / (eye_width * 1.12), 0.0, 1.0)),
                0.0,
                1.0,
            ),
            np.clip(
                0.80 * np.clip(mouth_open / (eye_width * 0.34), 0.0, 1.0)
                - 0.45 * np.clip(mouth_width / (eye_width * 1.12), 0.0, 1.0)
                + 0.35,
                0.0,
                1.0,
            ),
        ],
        dtype=np.float32,
    )
    return features, np.ones(features.shape, dtype=bool)


def compute_mouth_shape_features(
    points: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    indexes = FACE_EXPRESSION_INDEXES
    needed = [indexes["mouth_left"], indexes["mouth_right"], indexes["upper_lip"], indexes["lower_lip"]]
    if not all(index < len(points) and valid[index] for index in needed):
        return None, None

    mouth_indexes = [index for index in MOUTH_KEYPOINT_INDEXES if index < len(points) and valid[index]]
    if len(mouth_indexes) < 16:
        return None, None

    mouth_left = points[indexes["mouth_left"], :2]
    mouth_right = points[indexes["mouth_right"], :2]
    upper_lip = points[indexes["upper_lip"], :2]
    lower_lip = points[indexes["lower_lip"], :2]
    mouth_center = np.mean(np.asarray([mouth_left, mouth_right, upper_lip, lower_lip]), axis=0)
    mouth_width = max(float(np.linalg.norm(mouth_right - mouth_left)), 1e-6)

    selected = points[mouth_indexes, :2].copy()
    selected[:, 0] = np.abs(selected[:, 0] - mouth_center[0]) / mouth_width
    selected[:, 1] = (selected[:, 1] - mouth_center[1]) / mouth_width
    selected[:, 0] *= 0.72
    selected[:, 1] *= 1.35

    features = selected.reshape(-1).astype(np.float32)
    return features, np.ones(features.shape, dtype=bool)


def compute_hand_gesture_features(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wrist = points[0, :3]
    palm_lengths = [
        float(np.linalg.norm(points[index, :3] - wrist))
        for index in HAND_PALM_ANCHORS
        if valid[index]
    ]
    palm_scale = max(float(np.mean(palm_lengths)) if palm_lengths else 0.0, 1e-6)

    features: list[float] = []
    feature_valid: list[bool] = []
    for _, first, center, third, base, tip in HAND_FINGER_TRIPLETS:
        is_valid = bool(valid[first] and valid[center] and valid[third] and valid[base] and valid[tip])
        feature_valid.append(is_valid)
        if not is_valid:
            features.append(0.0)
            continue

        straightness = angle_between_points(points[first, :3], points[center, :3], points[third, :3]) / np.pi
        extension = np.linalg.norm(points[tip, :3] - points[base, :3]) / palm_scale
        extension = float(np.clip(extension / 1.35, 0.0, 1.0))
        features.append(0.68 * straightness + 0.32 * extension)

    return np.asarray(features, dtype=np.float32), np.asarray(feature_valid, dtype=bool)


def hand_gesture_distance(
    current_features: np.ndarray | None,
    current_valid: np.ndarray | None,
    reference_features: np.ndarray | None,
    reference_valid: np.ndarray | None,
) -> float | None:
    if (
        current_features is None
        or current_valid is None
        or reference_features is None
        or reference_valid is None
    ):
        return None

    common = current_valid & reference_valid
    if int(np.count_nonzero(common)) < HAND_FEATURE_MIN_MATCHES:
        return None

    current_values = current_features[common]
    reference_values = reference_features[common]
    state_diff = float(np.mean(np.abs(current_values - reference_values)))
    binary_diff = float(
        np.mean(
            (current_values >= HAND_STATE_OPEN_THRESHOLD)
            != (reference_values >= HAND_STATE_OPEN_THRESHOLD)
        )
    )

    current_count = int(np.count_nonzero(current_values >= HAND_STATE_OPEN_THRESHOLD))
    reference_count = int(np.count_nonzero(reference_values >= HAND_STATE_OPEN_THRESHOLD))
    count_diff = min(1.0, abs(current_count - reference_count) / 2.0)

    return (
        (1.0 - HAND_BINARY_WEIGHT - HAND_COUNT_WEIGHT) * state_diff
        + HAND_BINARY_WEIGHT * binary_diff
        + HAND_COUNT_WEIGHT * count_diff
    )


def feature_vector_distance(
    current_features: np.ndarray | None,
    current_valid: np.ndarray | None,
    reference_features: np.ndarray | None,
    reference_valid: np.ndarray | None,
) -> float | None:
    if (
        current_features is None
        or current_valid is None
        or reference_features is None
        or reference_valid is None
    ):
        return None

    common = current_valid & reference_valid
    if int(np.count_nonzero(common)) == 0:
        return None

    return float(np.mean(np.abs(current_features[common] - reference_features[common])))


def build_group(
    points: np.ndarray | None,
    config: GroupConfig,
    body_center: np.ndarray,
    body_scale: float,
) -> LandmarkGroup | None:
    if points is None:
        return None

    valid = points[:, 3] >= config.visibility_threshold
    if int(np.count_nonzero(valid)) < config.min_points:
        return None

    local_center, local_scale = compute_local_anchor(points, valid)
    gesture_features, gesture_valid = (None, None)
    if points.shape[0] == 21:
        gesture_features, gesture_valid = compute_hand_gesture_features(points, valid)

    return LandmarkGroup(
        config=config,
        raw=points,
        global_normalized=normalize_points(points, body_center, body_scale),
        local_normalized=normalize_points(points, local_center, local_scale),
        valid=valid,
        gesture_features=gesture_features,
        gesture_valid=gesture_valid,
    )


def build_feature_set(
    pose_points: np.ndarray | None,
    face_points: np.ndarray | None,
    face_expression_points: np.ndarray | None,
    left_hand_points: np.ndarray | None,
    right_hand_points: np.ndarray | None,
) -> FeatureSet:
    body_center, body_scale = compute_body_anchor(pose_points)
    pose_group = build_group(pose_points, POSE_GROUP, body_center, body_scale)
    face_group = build_group(face_points, FACE_GROUP, body_center, body_scale)
    pose_angles, pose_angle_valid = (None, None)
    face_expression_features, face_expression_valid = (None, None)
    mouth_shape_features, mouth_shape_valid = (None, None)
    if pose_group is not None:
        pose_angles, pose_angle_valid = compute_pose_angles(pose_group.raw, pose_group.valid)
    if face_expression_points is not None:
        face_expression_features, face_expression_valid = compute_face_expression_features(
            face_expression_points,
            np.ones(len(face_expression_points), dtype=bool),
        )
        mouth_shape_features, mouth_shape_valid = compute_mouth_shape_features(
            face_expression_points,
            np.ones(len(face_expression_points), dtype=bool),
        )

    return FeatureSet(
        pose=pose_group,
        face=face_group,
        left_hand=build_group(left_hand_points, LEFT_HAND_GROUP, body_center, body_scale),
        right_hand=build_group(right_hand_points, RIGHT_HAND_GROUP, body_center, body_scale),
        pose_angles=pose_angles,
        pose_angle_valid=pose_angle_valid,
        face_mesh_points=face_expression_points,
        face_expression_features=face_expression_features,
        face_expression_valid=face_expression_valid,
        mouth_shape_features=mouth_shape_features,
        mouth_shape_valid=mouth_shape_valid,
    )


def assign_detected_hands(
    detected_hands: list[np.ndarray],
    pose_points: np.ndarray | None,
    handedness_labels: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not detected_hands:
        return None, None

    left_wrist_index = mp_holistic.PoseLandmark.LEFT_WRIST.value
    right_wrist_index = mp_holistic.PoseLandmark.RIGHT_WRIST.value
    wrist_targets: dict[str, np.ndarray] = {}
    if pose_points is not None:
        if pose_points[left_wrist_index, 3] >= NON_POSE_VISIBILITY_THRESHOLD:
            wrist_targets["left"] = pose_points[left_wrist_index, :2]
        if pose_points[right_wrist_index, 3] >= NON_POSE_VISIBILITY_THRESHOLD:
            wrist_targets["right"] = pose_points[right_wrist_index, :2]

    if len(detected_hands) == 1:
        hand = detected_hands[0]
        if wrist_targets:
            distances = {
                side: float(np.linalg.norm(hand[0, :2] - target))
                for side, target in wrist_targets.items()
            }
            side = min(distances, key=distances.get)
            return (hand, None) if side == "left" else (None, hand)

        label = handedness_labels[0] if handedness_labels else "Left"
        return (hand, None) if label == "Left" else (None, hand)

    if len(detected_hands) >= 2 and len(wrist_targets) == 2:
        assignments = []
        for order in itertools.permutations(range(len(detected_hands)), 2):
            left_distance = float(np.linalg.norm(detected_hands[order[0]][0, :2] - wrist_targets["left"]))
            right_distance = float(np.linalg.norm(detected_hands[order[1]][0, :2] - wrist_targets["right"]))
            assignments.append((left_distance + right_distance, order))

        _, best_order = min(assignments, key=lambda item: item[0])
        return detected_hands[best_order[0]], detected_hands[best_order[1]]

    left_hand = None
    right_hand = None
    for hand, label in zip(detected_hands, handedness_labels, strict=False):
        if label == "Left" and left_hand is None:
            left_hand = hand
        elif label == "Right" and right_hand is None:
            right_hand = hand

    if left_hand is None and detected_hands:
        left_hand = detected_hands[0]
    if right_hand is None and len(detected_hands) > 1:
        right_hand = detected_hands[1]
    return left_hand, right_hand


def extract_hand_points(hands_result, pose_points: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if hands_result.multi_hand_landmarks is None:
        return None, None

    detected_hands = [landmarks_to_array(hand_landmarks) for hand_landmarks in hands_result.multi_hand_landmarks]
    handedness_labels = []
    if hands_result.multi_handedness is not None:
        for handedness in hands_result.multi_handedness:
            handedness_labels.append(handedness.classification[0].label)
    else:
        handedness_labels = ["Left"] * len(detected_hands)

    return assign_detected_hands(detected_hands, pose_points, handedness_labels)


def extract_holistic_hand_points(holistic_result) -> tuple[np.ndarray | None, np.ndarray | None]:
    left_hand = landmarks_to_array(holistic_result.left_hand_landmarks)
    right_hand = landmarks_to_array(holistic_result.right_hand_landmarks)
    return left_hand, right_hand


def resize_with_limit(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    limited_scale = min(scale, REFERENCE_VARIANT_MAX_EDGE / max(height, width, 1))
    if abs(limited_scale - 1.0) < 1e-3:
        return image.copy()

    new_width = max(1, int(round(width * limited_scale)))
    new_height = max(1, int(round(height * limited_scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def apply_clahe_bgr(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    channel_l, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    channel_l = clahe.apply(channel_l)
    enhanced = cv2.merge((channel_l, channel_a, channel_b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def sharpen_bgr(image: np.ndarray) -> np.ndarray:
    kernel = np.asarray([[0.0, -1.0, 0.0], [-1.0, 5.2, -1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def generate_reference_variants(image_bgr: np.ndarray) -> list[np.ndarray]:
    base = image_bgr
    clahe = apply_clahe_bgr(base)
    sharp = sharpen_bgr(base)
    clahe_sharp = sharpen_bgr(clahe)

    return [
        base,
        resize_with_limit(base, 1.55),
        resize_with_limit(base, 1.95),
        clahe,
        sharp,
        resize_with_limit(clahe_sharp, 1.75),
    ]


def detect_hands_with_fallback(
    hands_detector: mp_hands.Hands,
    image_bgr: np.ndarray,
    use_retry_variants: bool,
):
    def process(image_variant: np.ndarray):
        rgb_variant = cv2.cvtColor(image_variant, cv2.COLOR_BGR2RGB)
        rgb_variant.flags.writeable = False
        return hands_detector.process(rgb_variant)

    best_result = process(image_bgr)
    best_count = 0 if best_result.multi_hand_landmarks is None else len(best_result.multi_hand_landmarks)

    if not use_retry_variants or best_count >= 2:
        return best_result

    for scale in (1.5, 2.0):
        variant = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        candidate = process(variant)
        candidate_count = 0 if candidate.multi_hand_landmarks is None else len(candidate.multi_hand_landmarks)
        if candidate_count > best_count:
            best_result = candidate
            best_count = candidate_count
        if best_count >= 2:
            break

    return best_result


def face_landmarks_from_results(face_result, holistic_result):
    if face_result.multi_face_landmarks:
        return face_result.multi_face_landmarks[0]
    if holistic_result.face_landmarks is not None:
        return holistic_result.face_landmarks
    return None


def pose_points_score(pose_points: np.ndarray | None) -> float:
    if pose_points is None:
        return -1.0

    valid = pose_points[:, 3] >= POSE_VISIBILITY_THRESHOLD
    score = float(np.count_nonzero(valid))
    important = [
        mp_holistic.PoseLandmark.LEFT_WRIST.value,
        mp_holistic.PoseLandmark.RIGHT_WRIST.value,
        mp_holistic.PoseLandmark.LEFT_ELBOW.value,
        mp_holistic.PoseLandmark.RIGHT_ELBOW.value,
        mp_holistic.PoseLandmark.LEFT_SHOULDER.value,
        mp_holistic.PoseLandmark.RIGHT_SHOULDER.value,
    ]
    score += 3.5 * float(sum(valid[index] for index in important))

    visible_points = pose_points[valid, :2]
    if visible_points.size:
        span = np.max(visible_points, axis=0) - np.min(visible_points, axis=0)
        score += float(np.linalg.norm(span) * 25.0)

    return score


def face_points_score(face_points: np.ndarray | None) -> float:
    if face_points is None or len(face_points) <= max(FACE_EXPRESSION_INDEXES.values()):
        return -1.0

    valid = np.ones(len(face_points), dtype=bool)
    features, feature_valid = compute_face_expression_features(face_points, valid)
    mouth_indexes = [FACE_EXPRESSION_INDEXES["mouth_left"], FACE_EXPRESSION_INDEXES["mouth_right"]]
    mouth_span = float(np.linalg.norm(face_points[mouth_indexes[0], :2] - face_points[mouth_indexes[1], :2]))
    face_span = np.max(face_points[:, :2], axis=0) - np.min(face_points[:, :2], axis=0)
    score = float(np.linalg.norm(face_span) * 160.0 + mouth_span * 120.0)
    if features is not None and feature_valid is not None:
        score += 80.0 + 8.0 * float(np.count_nonzero(feature_valid))
    return score


def hand_points_score(hand_points: np.ndarray | None) -> float:
    if hand_points is None:
        return -1.0

    valid = np.ones(len(hand_points), dtype=bool)
    gesture_features, gesture_valid = compute_hand_gesture_features(hand_points, valid)
    hand_span = np.max(hand_points[:, :2], axis=0) - np.min(hand_points[:, :2], axis=0)
    score = float(np.linalg.norm(hand_span) * 140.0)
    if gesture_valid is not None:
        score += 16.0 * float(np.count_nonzero(gesture_valid))
    if gesture_features is not None:
        score += float(np.mean(np.abs(gesture_features - 0.5)) * 30.0)
    return score


def hand_alignment_score(
    hand_points: np.ndarray | None,
    pose_points: np.ndarray | None,
    side: str,
) -> float:
    if hand_points is None or pose_points is None:
        return 0.0

    wrist_index = getattr(mp_holistic.PoseLandmark, f"{side}_WRIST").value
    elbow_index = getattr(mp_holistic.PoseLandmark, f"{side}_ELBOW").value
    wrist = pose_points[wrist_index]
    if wrist[3] < NON_POSE_VISIBILITY_THRESHOLD:
        return 0.0

    wrist_distance = float(np.linalg.norm(hand_points[0, :2] - wrist[:2]))
    wrist_score = max(0.0, 1.0 - wrist_distance / 0.18)

    arm_score = 0.0
    elbow = pose_points[elbow_index]
    if elbow[3] >= NON_POSE_VISIBILITY_THRESHOLD:
        pose_arm = wrist[:2] - elbow[:2]
        hand_arm = hand_points[0, :2] - hand_points[9, :2]
        pose_norm = float(np.linalg.norm(pose_arm))
        hand_norm = float(np.linalg.norm(hand_arm))
        if pose_norm > 1e-6 and hand_norm > 1e-6:
            similarity = float(np.dot(pose_arm, hand_arm) / (pose_norm * hand_norm))
            arm_score = max(0.0, (similarity + 1.0) * 0.5)

    return 42.0 * wrist_score + 18.0 * arm_score


def score_hand_candidate(
    hand_points: np.ndarray | None,
    pose_points: np.ndarray | None,
    side: str,
) -> float:
    return hand_points_score(hand_points) + hand_alignment_score(hand_points, pose_points, side)


def select_best_hand_points(
    side: str,
    pose_points: np.ndarray | None,
    *candidates: np.ndarray | None,
) -> tuple[np.ndarray | None, float]:
    best_points = None
    best_score = -1.0
    for candidate in candidates:
        current_score = score_hand_candidate(candidate, pose_points, side)
        if current_score > best_score:
            best_score = current_score
            best_points = candidate
    return best_points, best_score


def hand_points_primary_score(hand_points: np.ndarray | None) -> float:
    if hand_points is None:
        return 0.0

    valid = np.ones(len(hand_points), dtype=bool)
    gesture_features, gesture_valid = compute_hand_gesture_features(hand_points, valid)
    if gesture_features is None or gesture_valid is None:
        return 0.0

    valid_features = gesture_features[gesture_valid]
    if valid_features.size < HAND_FEATURE_MIN_MATCHES:
        return 0.0

    open_count = int(np.count_nonzero(valid_features >= HAND_STATE_OPEN_THRESHOLD))
    confidence = float(np.mean(np.abs(valid_features - HAND_STATE_OPEN_THRESHOLD)))
    spread = float(np.max(valid_features) - np.min(valid_features))

    if 1 <= open_count <= 4:
        shape_bonus = 1.0
    elif open_count in {0, 5}:
        shape_bonus = 0.18
    else:
        shape_bonus = 0.35

    return 0.52 * confidence + 0.24 * spread + 0.24 * shape_bonus


def resize_for_live_processing(image: np.ndarray, max_edge: int = LIVE_PROCESS_MAX_EDGE) -> np.ndarray:
    height, width = image.shape[:2]
    current_edge = max(height, width)
    if current_edge <= max_edge:
        return image

    scale = max_edge / max(current_edge, 1)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def slice_face_contour_points(face_points: np.ndarray | None) -> np.ndarray | None:
    if face_points is None or len(face_points) <= FACE_KEYPOINT_INDEXES[-1]:
        return None
    return face_points[FACE_KEYPOINT_INDEXES].copy()


def remap_points_from_crop(
    points: np.ndarray,
    crop_x: int,
    crop_y: int,
    crop_width: int,
    crop_height: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    remapped = points.copy()
    remapped[:, 0] = (crop_x + points[:, 0] * crop_width) / max(image_width, 1)
    remapped[:, 1] = (crop_y + points[:, 1] * crop_height) / max(image_height, 1)
    remapped[:, 2] = points[:, 2] * max(crop_width / max(image_width, 1), crop_height / max(image_height, 1))
    return remapped


def detect_hand_from_wrist_crop(
    hands_detector: mp_hands.Hands,
    image_bgr: np.ndarray,
    pose_points: np.ndarray | None,
    side: str,
) -> tuple[np.ndarray | None, float]:
    if pose_points is None:
        return None, -1.0

    image_height, image_width = image_bgr.shape[:2]
    wrist_index = getattr(mp_holistic.PoseLandmark, f"{side}_WRIST").value
    elbow_index = getattr(mp_holistic.PoseLandmark, f"{side}_ELBOW").value
    shoulder_index = getattr(mp_holistic.PoseLandmark, f"{side}_SHOULDER").value

    wrist = pose_points[wrist_index]
    if wrist[3] < NON_POSE_VISIBILITY_THRESHOLD:
        return None, -1.0

    anchor = None
    if pose_points[elbow_index, 3] >= NON_POSE_VISIBILITY_THRESHOLD:
        anchor = pose_points[elbow_index, :2]
    elif pose_points[shoulder_index, 3] >= NON_POSE_VISIBILITY_THRESHOLD:
        anchor = pose_points[shoulder_index, :2]

    wrist_xy = wrist[:2]
    if anchor is not None:
        direction = wrist_xy - anchor
        length = float(np.linalg.norm(direction))
        if length > 1e-6:
            direction = direction / length
        else:
            direction = np.zeros(2, dtype=np.float32)
        base_half_size = max(length * 1.35, 0.11)
        crop_candidates = [
            (wrist_xy + direction * (base_half_size * 0.28), base_half_size),
            (wrist_xy, base_half_size * 1.08),
            (wrist_xy + direction * (base_half_size * 0.50), base_half_size * 1.18),
        ]
    else:
        crop_candidates = [(wrist_xy, 0.14), (wrist_xy, 0.18)]

    best_points = None
    best_score = -1.0

    for center, half_size in crop_candidates:
        x0 = max(0, int((center[0] - half_size) * image_width))
        x1 = min(image_width, int((center[0] + half_size) * image_width))
        y0 = max(0, int((center[1] - half_size) * image_height))
        y1 = min(image_height, int((center[1] + half_size) * image_height))

        if x1 - x0 < 36 or y1 - y0 < 36:
            continue
        crop = image_bgr[y0:y1, x0:x1]
        crop_variants = [
            crop,
            resize_with_limit(crop, 1.8),
            sharpen_bgr(crop),
            resize_with_limit(apply_clahe_bgr(crop), 1.6),
        ]

        for variant in crop_variants:
            rgb_variant = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
            rgb_variant.flags.writeable = False
            hands_result = hands_detector.process(rgb_variant)
            if hands_result.multi_hand_landmarks is None:
                continue

            for hand_landmarks in hands_result.multi_hand_landmarks:
                local_points = landmarks_to_array(hand_landmarks)
                current_score = hand_points_score(local_points)
                if current_score <= best_score:
                    continue

                best_score = current_score
                best_points = remap_points_from_crop(
                    local_points,
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                    image_width,
                    image_height,
                )

    return best_points, best_score


def estimate_face_crop_bounds(
    image_width: int,
    image_height: int,
    pose_points: np.ndarray | None,
    face_points: np.ndarray | None,
) -> tuple[int, int, int, int] | None:
    candidate_points: list[np.ndarray] = []

    if face_points is not None and len(face_points):
        candidate_points.append(face_points[:, :2])

    if pose_points is not None:
        pose_valid = pose_points[:, 3] >= NON_POSE_VISIBILITY_THRESHOLD
        head_indexes = [
            mp_holistic.PoseLandmark.NOSE.value,
            mp_holistic.PoseLandmark.LEFT_EYE_INNER.value,
            mp_holistic.PoseLandmark.RIGHT_EYE_INNER.value,
            mp_holistic.PoseLandmark.LEFT_EYE.value,
            mp_holistic.PoseLandmark.RIGHT_EYE.value,
            mp_holistic.PoseLandmark.LEFT_EYE_OUTER.value,
            mp_holistic.PoseLandmark.RIGHT_EYE_OUTER.value,
            mp_holistic.PoseLandmark.LEFT_EAR.value,
            mp_holistic.PoseLandmark.RIGHT_EAR.value,
            holistic_landmark_value("LEFT_MOUTH", "MOUTH_LEFT"),
            holistic_landmark_value("RIGHT_MOUTH", "MOUTH_RIGHT"),
        ]
        selected = [pose_points[index, :2] for index in head_indexes if pose_valid[index]]
        if selected:
            candidate_points.append(np.asarray(selected, dtype=np.float32))

    if not candidate_points:
        return None

    stacked = np.vstack(candidate_points)
    min_xy = np.min(stacked, axis=0)
    max_xy = np.max(stacked, axis=0)
    center = (min_xy + max_xy) * 0.5
    face_span = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1], 0.06))

    shoulder_width = 0.0
    if pose_points is not None:
        left_shoulder = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
        right_shoulder = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value
        if (
            pose_points[left_shoulder, 3] >= NON_POSE_VISIBILITY_THRESHOLD
            and pose_points[right_shoulder, 3] >= NON_POSE_VISIBILITY_THRESHOLD
        ):
            shoulder_width = float(np.linalg.norm(pose_points[left_shoulder, :2] - pose_points[right_shoulder, :2]))

    half_size = max(face_span * 0.92, shoulder_width * 0.42, 0.12)
    center[1] -= half_size * 0.10

    x0 = max(0, int((center[0] - half_size) * image_width))
    x1 = min(image_width, int((center[0] + half_size) * image_width))
    y0 = max(0, int((center[1] - half_size * 1.05) * image_height))
    y1 = min(image_height, int((center[1] + half_size * 0.95) * image_height))

    if x1 - x0 < 48 or y1 - y0 < 48:
        return None

    return x0, y0, x1, y1


def detect_face_from_crop(
    face_detector: mp_face_mesh.FaceMesh,
    image_bgr: np.ndarray,
    pose_points: np.ndarray | None,
    face_points: np.ndarray | None,
) -> tuple[np.ndarray | None, float]:
    image_height, image_width = image_bgr.shape[:2]
    bounds = estimate_face_crop_bounds(image_width, image_height, pose_points, face_points)
    if bounds is None:
        return None, -1.0

    x0, y0, x1, y1 = bounds
    crop = image_bgr[y0:y1, x0:x1]
    crop_variants = [
        crop,
        resize_with_limit(crop, 1.7),
        resize_with_limit(apply_clahe_bgr(crop), 1.7),
        resize_with_limit(sharpen_bgr(crop), 1.5),
        resize_with_limit(sharpen_bgr(apply_clahe_bgr(crop)), 1.9),
    ]

    best_points = None
    best_score = -1.0
    for variant in crop_variants:
        rgb_variant = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
        rgb_variant.flags.writeable = False
        face_result = face_detector.process(rgb_variant)
        if not face_result.multi_face_landmarks:
            continue

        for face_landmarks in face_result.multi_face_landmarks:
            local_points = landmarks_to_array(face_landmarks)
            current_score = face_points_score(local_points)
            if current_score <= best_score:
                continue

            best_score = current_score
            best_points = remap_points_from_crop(
                local_points,
                x0,
                y0,
                x1 - x0,
                y1 - y0,
                image_width,
                image_height,
            )

    return best_points, best_score


def detect_reference_features(
    holistic_detector: mp_holistic.Holistic,
    hands_detector: mp_hands.Hands,
    face_detector: mp_face_mesh.FaceMesh,
    image_bgr: np.ndarray,
) -> FeatureSet:
    best_pose_points = None
    best_face_points = None
    best_left_hand = None
    best_right_hand = None
    best_pose_score = -1.0
    best_face_score = -1.0
    best_left_hand_score = -1.0
    best_right_hand_score = -1.0

    for variant in generate_reference_variants(image_bgr):
        image_rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        holistic_result = holistic_detector.process(image_rgb)
        face_result = face_detector.process(image_rgb)
        hands_result = hands_detector.process(image_rgb)

        pose_points = landmarks_to_array(holistic_result.pose_landmarks, use_visibility=True)
        face_landmarks = face_landmarks_from_results(face_result, holistic_result)
        face_points = landmarks_to_array(face_landmarks)
        holistic_left_hand, holistic_right_hand = extract_holistic_hand_points(holistic_result)
        detected_left_hand, detected_right_hand = extract_hand_points(hands_result, pose_points)
        left_hand_points, current_left_hand_score = select_best_hand_points(
            "LEFT",
            pose_points,
            holistic_left_hand,
            detected_left_hand,
        )
        right_hand_points, current_right_hand_score = select_best_hand_points(
            "RIGHT",
            pose_points,
            holistic_right_hand,
            detected_right_hand,
        )

        current_pose_score = pose_points_score(pose_points)
        if current_pose_score > best_pose_score:
            best_pose_score = current_pose_score
            best_pose_points = pose_points

        current_face_score = face_points_score(face_points)
        if current_face_score > best_face_score:
            best_face_score = current_face_score
            best_face_points = face_points

        if current_left_hand_score > best_left_hand_score:
            best_left_hand_score = current_left_hand_score
            best_left_hand = left_hand_points

        if current_right_hand_score > best_right_hand_score:
            best_right_hand_score = current_right_hand_score
            best_right_hand = right_hand_points

    cropped_face_points, cropped_face_score = detect_face_from_crop(
        face_detector,
        image_bgr,
        best_pose_points,
        best_face_points,
    )
    if cropped_face_score > best_face_score:
        best_face_score = cropped_face_score
        best_face_points = cropped_face_points

    cropped_left_hand, cropped_left_hand_score = detect_hand_from_wrist_crop(
        hands_detector,
        image_bgr,
        best_pose_points,
        "LEFT",
    )
    _, candidate_left_score = select_best_hand_points(
        "LEFT",
        best_pose_points,
        best_left_hand,
        cropped_left_hand,
    )
    if candidate_left_score > best_left_hand_score:
        best_left_hand_score = candidate_left_score
        best_left_hand = cropped_left_hand

    cropped_right_hand, cropped_right_hand_score = detect_hand_from_wrist_crop(
        hands_detector,
        image_bgr,
        best_pose_points,
        "RIGHT",
    )
    _, candidate_right_score = select_best_hand_points(
        "RIGHT",
        best_pose_points,
        best_right_hand,
        cropped_right_hand,
    )
    if candidate_right_score > best_right_hand_score:
        best_right_hand_score = candidate_right_score
        best_right_hand = cropped_right_hand

    return build_feature_set(
        pose_points=best_pose_points,
        face_points=slice_face_contour_points(best_face_points),
        face_expression_points=best_face_points,
        left_hand_points=best_left_hand,
        right_hand_points=best_right_hand,
    )


def detect_features(
    holistic_detector: mp_holistic.Holistic,
    hands_detector: mp_hands.Hands,
    face_detector: mp_face_mesh.FaceMesh,
    image_bgr: np.ndarray,
    use_retry_variants: bool = False,
    frame_index: int = 0,
    prefer_speed: bool = False,
):
    process_bgr = resize_for_live_processing(image_bgr) if prefer_speed else image_bgr
    fallback_bgr = image_bgr if prefer_speed else process_bgr
    image_rgb = cv2.cvtColor(process_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    holistic_result = holistic_detector.process(image_rgb)
    hands_result = detect_hands_with_fallback(hands_detector, process_bgr, use_retry_variants)

    pose_points = landmarks_to_array(holistic_result.pose_landmarks, use_visibility=True)
    face_landmarks = holistic_result.face_landmarks
    face_expression_points = landmarks_to_array(face_landmarks)
    face_score = face_points_score(face_expression_points)
    cropped_face_points, cropped_face_score = (None, -1.0)
    holistic_left_hand, holistic_right_hand = extract_holistic_hand_points(holistic_result)
    detected_left_hand, detected_right_hand = extract_hand_points(hands_result, pose_points)
    left_hand_points, left_hand_score = select_best_hand_points(
        "LEFT",
        pose_points,
        holistic_left_hand,
        detected_left_hand,
    )
    right_hand_points, right_hand_score = select_best_hand_points(
        "RIGHT",
        pose_points,
        holistic_right_hand,
        detected_right_hand,
    )
    has_strong_hand_shape = max(
        hand_points_primary_score(left_hand_points),
        hand_points_primary_score(right_hand_points),
    ) >= 0.24

    should_refresh_face = (frame_index % LIVE_FACE_REFRESH_INTERVAL) == 0
    if face_expression_points is None or (
        should_refresh_face
        and not has_strong_hand_shape
        and face_score < LIVE_FACE_SCORE_THRESHOLD
    ):
        cropped_face_points, cropped_face_score = detect_face_from_crop(
            face_detector,
            fallback_bgr,
            pose_points,
            face_expression_points,
        )
        if cropped_face_score > face_score:
            face_expression_points = cropped_face_points

    should_refresh_hand_crop = (frame_index % LIVE_HAND_CROP_INTERVAL) == 0
    if should_refresh_hand_crop and (
        left_hand_points is None or left_hand_score < LIVE_HAND_SCORE_THRESHOLD
    ):
        cropped_left_hand, _ = detect_hand_from_wrist_crop(
            hands_detector,
            fallback_bgr,
            pose_points,
            "LEFT",
        )
        left_hand_points, left_hand_score = select_best_hand_points(
            "LEFT",
            pose_points,
            left_hand_points,
            cropped_left_hand,
        )
    if should_refresh_hand_crop and (
        right_hand_points is None or right_hand_score < LIVE_HAND_SCORE_THRESHOLD
    ):
        cropped_right_hand, _ = detect_hand_from_wrist_crop(
            hands_detector,
            fallback_bgr,
            pose_points,
            "RIGHT",
        )
        right_hand_points, right_hand_score = select_best_hand_points(
            "RIGHT",
            pose_points,
            right_hand_points,
            cropped_right_hand,
        )

    face_contour_points = landmarks_to_array(face_landmarks, FACE_KEYPOINT_INDEXES)
    if face_expression_points is not None:
        cropped_contours = slice_face_contour_points(face_expression_points)
        if cropped_contours is not None and (face_contour_points is None or cropped_face_score > face_score):
            face_contour_points = cropped_contours

    features = build_feature_set(
        pose_points=pose_points,
        face_points=face_contour_points,
        face_expression_points=face_expression_points,
        left_hand_points=left_hand_points,
        right_hand_points=right_hand_points,
    )
    return features, holistic_result, hands_result, face_landmarks


def mirror_points(points: np.ndarray) -> np.ndarray:
    mirrored = points.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    return mirrored


def mirror_feature_set(features: FeatureSet) -> FeatureSet:
    pose_points = mirror_points(features.pose.raw) if features.pose is not None else None
    if pose_points is not None:
        for left, right in POSE_LEFT_RIGHT_PAIRS:
            pose_points[[left, right]] = pose_points[[right, left]]

    face_points = mirror_points(features.face.raw) if features.face is not None else None
    face_mesh_points = mirror_points(features.face_mesh_points) if features.face_mesh_points is not None else None
    left_hand_points = mirror_points(features.right_hand.raw) if features.right_hand is not None else None
    right_hand_points = mirror_points(features.left_hand.raw) if features.left_hand is not None else None

    mirrored = build_feature_set(
        pose_points,
        face_points,
        face_mesh_points,
        left_hand_points,
        right_hand_points,
    )
    return FeatureSet(
        pose=mirrored.pose,
        face=mirrored.face,
        left_hand=mirrored.left_hand,
        right_hand=mirrored.right_hand,
        pose_angles=mirrored.pose_angles,
        pose_angle_valid=mirrored.pose_angle_valid,
        face_mesh_points=features.face_mesh_points,
        face_expression_features=features.face_expression_features,
        face_expression_valid=features.face_expression_valid,
        mouth_shape_features=features.mouth_shape_features,
        mouth_shape_valid=features.mouth_shape_valid,
    )


def average_distance(
    left_points: np.ndarray,
    right_points: np.ndarray,
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    valid: np.ndarray,
) -> float:
    weights = left_raw[valid, 3].clip(0.05, 1.0) * right_raw[valid, 3].clip(0.05, 1.0)
    distances = np.linalg.norm(left_points[valid, :3] - right_points[valid, :3], axis=1)
    return float(np.average(distances, weights=weights))


def angle_distance(
    current_angles: np.ndarray | None,
    current_valid: np.ndarray | None,
    reference_angles: np.ndarray | None,
    reference_valid: np.ndarray | None,
) -> float | None:
    if (
        current_angles is None
        or current_valid is None
        or reference_angles is None
        or reference_valid is None
    ):
        return None

    common = current_valid & reference_valid
    if int(np.count_nonzero(common)) < MIN_ANGLE_MATCHES:
        return None

    delta = np.abs(current_angles[common] - reference_angles[common])
    return float(np.mean(delta / np.pi))


def group_distance(current: LandmarkGroup, reference: LandmarkGroup) -> float | None:
    common = current.valid & reference.valid
    if int(np.count_nonzero(common)) < current.config.min_points:
        return None

    global_score = average_distance(
        current.global_normalized,
        reference.global_normalized,
        current.raw,
        reference.raw,
        common,
    )

    if current.config.local_ratio <= 0.0:
        score = global_score
    else:
        local_score = average_distance(
            current.local_normalized,
            reference.local_normalized,
            current.raw,
            reference.raw,
            common,
        )
        score = (1.0 - current.config.local_ratio) * global_score + current.config.local_ratio * local_score

    gesture_score = hand_gesture_distance(
        current.gesture_features,
        current.gesture_valid,
        reference.gesture_features,
        reference.gesture_valid,
    )
    if gesture_score is not None:
        score = (1.0 - HAND_STATE_BLEND) * score + HAND_STATE_BLEND * gesture_score

    return score


def detected_hand_count(features: FeatureSet) -> int:
    return int(features.left_hand is not None) + int(features.right_hand is not None)


def hand_shape_priority_score(group: LandmarkGroup | None) -> float:
    if group is None or group.gesture_features is None or group.gesture_valid is None:
        return 0.0

    valid_features = group.gesture_features[group.gesture_valid]
    if valid_features.size < HAND_FEATURE_MIN_MATCHES:
        return 0.0

    open_count = int(np.count_nonzero(valid_features >= HAND_STATE_OPEN_THRESHOLD))
    confidence = float(np.mean(np.abs(valid_features - HAND_STATE_OPEN_THRESHOLD)))
    spread = float(np.max(valid_features) - np.min(valid_features))

    if 1 <= open_count <= 4:
        shape_bonus = 1.0
    elif open_count in {0, 5}:
        shape_bonus = 0.18
    else:
        shape_bonus = 0.35

    return 0.52 * confidence + 0.24 * spread + 0.24 * shape_bonus


def has_primary_hand_gesture(features: FeatureSet) -> bool:
    if detected_hand_count(features) == 0:
        return False

    best_score = max(
        hand_shape_priority_score(features.left_hand),
        hand_shape_priority_score(features.right_hand),
    )
    return best_score >= 0.24


def resolve_match_weight_profile(current: FeatureSet) -> dict[str, float]:
    current_hand_count = detected_hand_count(current)
    if current_hand_count > 0 and has_primary_hand_gesture(current):
        return {
            "pose": current.pose.config.weight * HAND_PRIMARY_POSE_SCALE if current.pose is not None else 0.0,
            "face_scale": HAND_PRIMARY_FACE_SCALE,
            "expression": FACE_EXPRESSION_WEIGHT * HAND_PRIMARY_EXPRESSION_SCALE,
            "mouth": MOUTH_SHAPE_WEIGHT * HAND_PRIMARY_MOUTH_SCALE,
            "hand_multiplier": HAND_ACTIVE_WEIGHT_MULTIPLIER * HAND_PRIMARY_WEIGHT_MULTIPLIER,
            "missing_penalty": HAND_PRIMARY_MISSING_PENALTY,
            "count_weight": HAND_COUNT_PRIORITY_WEIGHT,
        }

    return {
        "pose": current.pose.config.weight * FACE_FALLBACK_POSE_SCALE if current.pose is not None else 0.0,
        "face_scale": FACE_FALLBACK_FACE_SCALE,
        "expression": FACE_EXPRESSION_WEIGHT * FACE_FALLBACK_EXPRESSION_SCALE,
        "mouth": MOUTH_SHAPE_WEIGHT * FACE_FALLBACK_MOUTH_SCALE,
        "hand_multiplier": HAND_ACTIVE_WEIGHT_MULTIPLIER,
        "missing_penalty": HAND_MISSING_PENALTY,
        "count_weight": 0.0,
    }


def feature_set_distance(
    current: FeatureSet,
    reference: FeatureSet,
) -> tuple[float | None, dict[str, float]]:
    if current.pose is None or reference.pose is None:
        return None, {}

    total_score = 0.0
    total_weight = 0.0
    component_scores: dict[str, float] = {}

    pose_score = group_distance(current.pose, reference.pose)
    if pose_score is None:
        return None, {}

    pose_angle_score = angle_distance(
        current.pose_angles,
        current.pose_angle_valid,
        reference.pose_angles,
        reference.pose_angle_valid,
    )
    if pose_angle_score is not None:
        pose_score = (1.0 - POSE_ANGLE_BLEND) * pose_score + POSE_ANGLE_BLEND * pose_angle_score
        component_scores["Eklem"] = pose_angle_score

    weight_profile = resolve_match_weight_profile(current)

    component_scores[current.pose.config.label] = pose_score
    total_score += pose_score * weight_profile["pose"]
    total_weight += weight_profile["pose"]

    for current_group, reference_group in (
        (current.face, reference.face),
        (current.left_hand, reference.left_hand),
        (current.right_hand, reference.right_hand),
    ):
        if current_group is None:
            continue

        group_weight = current_group.config.weight
        if current_group.config.label == "Yuz":
            group_weight *= weight_profile["face_scale"]
        if current_group.config.label in {"Sol el", "Sag el"}:
            group_weight *= weight_profile["hand_multiplier"]

        if reference_group is None:
            if current_group.config.label in {"Sol el", "Sag el"}:
                component_scores[current_group.config.label] = weight_profile["missing_penalty"]
                total_score += weight_profile["missing_penalty"] * group_weight
                total_weight += group_weight
            continue

        score = group_distance(current_group, reference_group)
        if score is None:
            continue

        component_scores[current_group.config.label] = score
        total_score += score * group_weight
        total_weight += group_weight

    expression_score = feature_vector_distance(
        current.face_expression_features,
        current.face_expression_valid,
        reference.face_expression_features,
        reference.face_expression_valid,
    )
    if expression_score is not None:
        component_scores["Mimik"] = expression_score
        total_score += expression_score * weight_profile["expression"]
        total_weight += weight_profile["expression"]

    mouth_shape_score = feature_vector_distance(
        current.mouth_shape_features,
        current.mouth_shape_valid,
        reference.mouth_shape_features,
        reference.mouth_shape_valid,
    )
    if mouth_shape_score is not None:
        component_scores["Agiz"] = mouth_shape_score
        total_score += mouth_shape_score * weight_profile["mouth"]
        total_weight += weight_profile["mouth"]

    if weight_profile["count_weight"] > 0.0:
        current_hand_count = detected_hand_count(current)
        reference_hand_count = detected_hand_count(reference)
        hand_count_score = min(1.0, abs(current_hand_count - reference_hand_count) / max(current_hand_count, 1))
        component_scores["El sayisi"] = hand_count_score
        total_score += hand_count_score * weight_profile["count_weight"]
        total_weight += weight_profile["count_weight"]

    if total_weight < MIN_MATCH_WEIGHT:
        return None, component_scores

    return total_score / total_weight, component_scores


def find_best_match(features: FeatureSet, references: list[ReferencePose]) -> Match | None:
    mirrored_features = mirror_feature_set(features)

    best: Match | None = None
    for reference in references:
        for candidate_features, mirrored in ((features, False), (mirrored_features, True)):
            distance, component_scores = feature_set_distance(candidate_features, reference.features)
            if distance is None:
                continue

            if best is None or distance < best.distance:
                best = Match(
                    reference=reference,
                    distance=distance,
                    mirrored=mirrored,
                    component_scores=component_scores,
                )

    return best


def load_references(folder: Path) -> list[ReferencePose]:
    references: list[ReferencePose] = []

    with (
        mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            smooth_landmarks=False,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        ) as holistic_detector,
        mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        ) as face_detector,
        mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        ) as hands_detector,
    ):
        for image_path in iter_pose_images(folder):
            image = read_image(image_path)
            if image is None:
                print(f"Atlandi, okunamadi: {image_path.name}")
                continue

            features = detect_reference_features(
                holistic_detector,
                hands_detector,
                face_detector,
                image,
            )
            if not features.has_pose():
                print(f"Atlandi, govde pozu bulunamadi: {image_path.name}")
                continue

            references.append(
                ReferencePose(
                    name=image_path.name,
                    path=image_path,
                    image=image,
                    features=features,
                    preview_image=render_reference_preview(image, features),
                    skeleton_image=render_reference_skeleton(features),
                )
            )

    return references


def fit_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((max_height, max_width, 3), dtype=np.uint8)

    scale = min(max_width / width, max_height / height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    canvas = np.full((max_height, max_width, 3), 18, dtype=np.uint8)
    x = (max_width - new_width) // 2
    y = (max_height - new_height) // 2
    canvas[y : y + new_height, x : x + new_width] = resized
    return canvas


def cover_image(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    scale = max(target_width / width, target_height / height)
    new_width = max(1, int(np.ceil(width * scale)))
    new_height = max(1, int(np.ceil(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    x = max(0, (new_width - target_width) // 2)
    y = max(0, (new_height - target_height) // 2)
    return resized[y : y + target_height, x : x + target_width].copy()


def get_screen_size(default_width: int = 1920, default_height: int = 1080) -> tuple[int, int]:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass

    return default_width, default_height


def point_to_pixel(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(point[0], 0.0, 1.0) * max(width - 1, 1))
    y = int(np.clip(point[1], 0.0, 1.0) * max(height - 1, 1))
    return x, y


def draw_connections_from_array(
    image: np.ndarray,
    points: np.ndarray | None,
    connections,
    color: tuple[int, int, int],
    thickness: int,
    radius: int,
    valid: np.ndarray | None = None,
) -> None:
    if points is None:
        return

    height, width = image.shape[:2]
    for first, second in connections:
        if first >= len(points) or second >= len(points):
            continue
        if valid is not None and not (valid[first] and valid[second]):
            continue
        cv2.line(
            image,
            point_to_pixel(points[first], width, height),
            point_to_pixel(points[second], width, height),
            color,
            thickness,
            cv2.LINE_AA,
        )

    draw_indexes = np.flatnonzero(valid) if valid is not None else np.arange(len(points))
    for index in draw_indexes:
        cv2.circle(
            image,
            point_to_pixel(points[index], width, height),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )


def hand_open_count(group: LandmarkGroup | None) -> int | None:
    if group is None or group.gesture_features is None:
        return None
    return int(np.count_nonzero(group.gesture_features >= HAND_STATE_OPEN_THRESHOLD))


def hand_summary_text(features: FeatureSet) -> str:
    labels: list[str] = []
    left_count = hand_open_count(features.left_hand)
    right_count = hand_open_count(features.right_hand)
    if left_count is not None:
        labels.append(f"L:{left_count}")
    if right_count is not None:
        labels.append(f"R:{right_count}")
    return " ".join(labels) if labels else "El:-"


def render_reference_preview(image: np.ndarray, features: FeatureSet) -> np.ndarray:
    preview = cv2.addWeighted(image, 0.45, np.full_like(image, 16), 0.55, 0.0)
    if features.face_mesh_points is not None:
        draw_connections_from_array(
            preview,
            features.face_mesh_points,
            mp_face_mesh.FACEMESH_LIPS,
            (115, 130, 255),
            2,
            1,
        )
        draw_connections_from_array(
            preview,
            features.face_mesh_points,
            mp_face_mesh.FACEMESH_LEFT_EYE,
            (105, 230, 255),
            1,
            1,
        )
        draw_connections_from_array(
            preview,
            features.face_mesh_points,
            mp_face_mesh.FACEMESH_RIGHT_EYE,
            (105, 230, 255),
            1,
            1,
        )
        draw_connections_from_array(
            preview,
            features.face_mesh_points,
            mp_face_mesh.FACEMESH_LEFT_EYEBROW,
            (150, 255, 170),
            1,
            1,
        )
        draw_connections_from_array(
            preview,
            features.face_mesh_points,
            mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
            (150, 255, 170),
            1,
            1,
        )
    if features.pose is not None:
        draw_connections_from_array(
            preview,
            features.pose.raw,
            mp_holistic.POSE_CONNECTIONS,
            (90, 220, 255),
            2,
            2,
            features.pose.valid,
        )
    for group, color in (
        (features.left_hand, (110, 255, 150)),
        (features.right_hand, (255, 180, 80)),
    ):
        if group is None:
            continue
        draw_connections_from_array(
            preview,
            group.raw,
            mp_hands.HAND_CONNECTIONS,
            color,
            2,
            2,
            group.valid,
        )
    return preview


def render_reference_skeleton(features: FeatureSet, width: int = 420, height: int = 220) -> np.ndarray:
    skeleton = np.full((height, width, 3), (14, 16, 20), dtype=np.uint8)
    if features.face_mesh_points is not None:
        draw_connections_from_array(
            skeleton,
            features.face_mesh_points,
            mp_face_mesh.FACEMESH_FACE_OVAL,
            (85, 105, 140),
            1,
            1,
        )
        draw_connections_from_array(
            skeleton,
            features.face_mesh_points,
            mp_face_mesh.FACEMESH_LIPS,
            (115, 130, 255),
            3,
            2,
        )
        draw_connections_from_array(
            skeleton,
            features.face_mesh_points,
            mp_face_mesh.FACEMESH_LEFT_EYE,
            (105, 230, 255),
            2,
            1,
        )
        draw_connections_from_array(
            skeleton,
            features.face_mesh_points,
            mp_face_mesh.FACEMESH_RIGHT_EYE,
            (105, 230, 255),
            2,
            1,
        )
        draw_connections_from_array(
            skeleton,
            features.face_mesh_points,
            mp_face_mesh.FACEMESH_LEFT_EYEBROW,
            (150, 255, 170),
            2,
            1,
        )
        draw_connections_from_array(
            skeleton,
            features.face_mesh_points,
            mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
            (150, 255, 170),
            2,
            1,
        )
    if features.pose is not None:
        draw_connections_from_array(
            skeleton,
            features.pose.raw,
            mp_holistic.POSE_CONNECTIONS,
            (95, 220, 255),
            2,
            3,
            features.pose.valid,
        )
    for group, color in (
        (features.left_hand, (110, 255, 150)),
        (features.right_hand, (255, 190, 95)),
    ):
        if group is None:
            continue
        draw_connections_from_array(
            skeleton,
            group.raw,
            mp_hands.HAND_CONNECTIONS,
            color,
            2,
            3,
            group.valid,
        )
    return skeleton


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 2,
) -> None:
    x, y = origin
    cv2.putText(image, text, (x + 2, y + 2), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_panel(
    frame_height: int,
    panel_width: int,
    match: Match | None,
    references: list[ReferencePose],
    threshold: float,
    camera_has_pose: bool,
    current_features: FeatureSet | None,
) -> np.ndarray:
    panel = np.full((frame_height, panel_width, 3), (26, 28, 32), dtype=np.uint8)
    padding = max(18, int(min(frame_height, panel_width) * 0.022))
    gap = max(12, int(frame_height * 0.015))
    usable_width = max(1, panel_width - padding * 2)
    usable_height = max(1, frame_height - padding * 2)
    skeleton_height = max(170, int(usable_height * 0.27))
    skeleton_height = min(skeleton_height, max(180, usable_height - 240))
    preview_height = max(180, usable_height - skeleton_height - gap)

    if match is not None:
        skeleton_preview = render_reference_skeleton(
            match.reference.features,
            width=usable_width,
            height=skeleton_height,
        )
        panel[padding : padding + skeleton_height, padding : padding + usable_width] = skeleton_preview

        preview_top = padding + skeleton_height + gap
        preview = cover_image(match.reference.image, usable_width, preview_height)
        panel[preview_top : preview_top + preview_height, padding : padding + usable_width] = preview

        badge_width = min(300, max(180, usable_width // 2))
        badge_height = 52
        badge_x = padding + 18
        badge_y = preview_top + preview_height - badge_height - 18
        badge = panel[badge_y : badge_y + badge_height, badge_x : badge_x + badge_width]
        badge_fill = np.full_like(badge, (18, 18, 18))
        panel[badge_y : badge_y + badge_height, badge_x : badge_x + badge_width] = cv2.addWeighted(
            badge,
            0.25,
            badge_fill,
            0.75,
            0.0,
        )
        draw_text(
            panel,
            f"Eslesme skoru: {match.distance:.3f}",
            (badge_x + 14, badge_y + 34),
            0.70,
            (235, 235, 235),
            2,
        )
    else:
        center_y = frame_height // 2
        draw_text(panel, "Referans bekleniyor", (padding, center_y - 22), 0.78, (80, 170, 255))
        if not references:
            draw_text(panel, "poses klasorunde okunabilen", (padding, center_y + 18), 0.62)
            draw_text(panel, "referans gorsel yok.", (padding, center_y + 54), 0.62)
        elif not camera_has_pose:
            draw_text(panel, "Kamerada govde pozu", (padding, center_y + 18), 0.62)
            draw_text(panel, "bulunamadi.", (padding, center_y + 54), 0.62)
    return panel


def draw_landmarks(annotated: np.ndarray, holistic_result, hands_result, face_landmarks) -> None:
    if face_landmarks is not None:
        mp_drawing.draw_landmarks(
            annotated,
            face_landmarks,
            mp_face_mesh.FACEMESH_FACE_OVAL,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(90, 180, 255), thickness=1),
        )
        for connections, color, thickness in (
            (mp_face_mesh.FACEMESH_LIPS, (120, 135, 255), 2),
            (mp_face_mesh.FACEMESH_LEFT_EYE, (110, 230, 255), 1),
            (mp_face_mesh.FACEMESH_RIGHT_EYE, (110, 230, 255), 1),
            (mp_face_mesh.FACEMESH_LEFT_EYEBROW, (150, 255, 170), 1),
            (mp_face_mesh.FACEMESH_RIGHT_EYEBROW, (150, 255, 170), 1),
        ):
            mp_drawing.draw_landmarks(
                annotated,
                face_landmarks,
                connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=thickness),
            )

    if hands_result.multi_hand_landmarks is not None:
        for hand_landmarks in hands_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
            )

    if holistic_result.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            annotated,
            holistic_result.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    backends: list[int | None]
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    else:
        backends = [None]

    for backend in backends:
        if backend is None:
            capture = cv2.VideoCapture(camera_index)
        else:
            capture = cv2.VideoCapture(camera_index, backend)

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if capture.isOpened():
            return capture

        capture.release()

    capture = cv2.VideoCapture()
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def run() -> int:
    args = parse_args()
    references = load_references(args.poses)
    print(f"{len(references)} referans poz yuklendi.")

    screen_width, screen_height = get_screen_size()
    requested_width = max(640, args.width)
    requested_height = max(360, args.height)

    capture = open_camera(args.camera, requested_width, requested_height)
    if not capture.isOpened():
        print("Kamera acilamadi. Farkli bir indeks deneyin: --camera 1")
        if sys.platform == "darwin":
            print(
                "macOS kullaniyorsaniz Terminal veya Python icin "
                "System Settings > Privacy & Security > Camera iznini kontrol edin."
            )
        return 1

    window_name = "Pose Matcher"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass

    try:
        with (
            mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                refine_face_landmarks=True,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            ) as holistic_detector,
            mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45,
            ) as face_detector,
            mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45,
            ) as hands_detector,
        ):
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    print("Kameradan goruntu alinamadi.")
                    break

                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                features, holistic_result, hands_result, face_landmarks = detect_features(
                    holistic_detector,
                    hands_detector,
                    face_detector,
                    frame,
                    frame_index=frame_index,
                    prefer_speed=True,
                )
                match = find_best_match(features, references) if features.has_pose() else None

                annotated = frame.copy()
                draw_landmarks(annotated, holistic_result, hands_result, face_landmarks)
                pane_height = screen_height
                pane_width = screen_width // 2
                camera_pane = cover_image(annotated, pane_width, pane_height)
                panel = draw_panel(
                    frame_height=pane_height,
                    panel_width=screen_width - pane_width,
                    match=match,
                    references=references,
                    threshold=args.threshold,
                    camera_has_pose=features.has_pose(),
                    current_features=features,
                )
                output = np.hstack((camera_pane, panel))
                cv2.imshow(window_name, output)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    references = load_references(args.poses)
                    print(f"{len(references)} referans poz yeniden yuklendi.")
                frame_index += 1
    except KeyboardInterrupt:
        print("\nUygulama kapatildi.")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
