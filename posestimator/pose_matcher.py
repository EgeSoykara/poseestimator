from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VISIBILITY_THRESHOLD = 0.45
MIN_COMMON_POINTS = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def landmark_value(*names: str) -> int:
    for name in names:
        landmark = getattr(mp_pose.PoseLandmark, name, None)
        if landmark is not None:
            return landmark.value

    raise AttributeError(f"Pose landmark not found: {', '.join(names)}")


LEFT_RIGHT_PAIRS = [
    (landmark_value("LEFT_EYE_INNER"), landmark_value("RIGHT_EYE_INNER")),
    (landmark_value("LEFT_EYE"), landmark_value("RIGHT_EYE")),
    (landmark_value("LEFT_EYE_OUTER"), landmark_value("RIGHT_EYE_OUTER")),
    (landmark_value("LEFT_EAR"), landmark_value("RIGHT_EAR")),
    (landmark_value("LEFT_MOUTH", "MOUTH_LEFT"), landmark_value("RIGHT_MOUTH", "MOUTH_RIGHT")),
    (landmark_value("LEFT_SHOULDER"), landmark_value("RIGHT_SHOULDER")),
    (landmark_value("LEFT_ELBOW"), landmark_value("RIGHT_ELBOW")),
    (landmark_value("LEFT_WRIST"), landmark_value("RIGHT_WRIST")),
    (landmark_value("LEFT_PINKY"), landmark_value("RIGHT_PINKY")),
    (landmark_value("LEFT_INDEX"), landmark_value("RIGHT_INDEX")),
    (landmark_value("LEFT_THUMB"), landmark_value("RIGHT_THUMB")),
    (landmark_value("LEFT_HIP"), landmark_value("RIGHT_HIP")),
    (landmark_value("LEFT_KNEE"), landmark_value("RIGHT_KNEE")),
    (landmark_value("LEFT_ANKLE"), landmark_value("RIGHT_ANKLE")),
    (landmark_value("LEFT_HEEL"), landmark_value("RIGHT_HEEL")),
    (landmark_value("LEFT_FOOT_INDEX"), landmark_value("RIGHT_FOOT_INDEX")),
]


@dataclass(frozen=True)
class ReferencePose:
    name: str
    path: Path
    image: np.ndarray
    landmarks: np.ndarray
    normalized: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class Match:
    reference: ReferencePose
    distance: float
    mirrored: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kameradaki pozu poses klasorundeki fotograflarla eslestirir."
    )
    parser.add_argument(
        "--poses",
        type=Path,
        default=Path("poses"),
        help="Referans fotograflarin bulundugu klasor.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Kamera indeksi.")
    parser.add_argument("--width", type=int, default=1280, help="Kamera genisligi.")
    parser.add_argument("--height", type=int, default=720, help="Kamera yuksekligi.")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Kamera goruntusunu ayna gibi cevirmeyi kapatir.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.15,
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

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def iter_pose_images(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def landmarks_to_array(pose_landmarks) -> np.ndarray | None:
    if pose_landmarks is None:
        return None

    rows = [
        [landmark.x, landmark.y, landmark.z, landmark.visibility]
        for landmark in pose_landmarks.landmark
    ]
    return np.asarray(rows, dtype=np.float32)


def center_point(landmarks: np.ndarray, left: int, right: int) -> np.ndarray:
    return (landmarks[left, :3] + landmarks[right, :3]) * 0.5


def normalize_landmarks(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = landmarks[:, 3] >= VISIBILITY_THRESHOLD

    left_hip = mp_pose.PoseLandmark.LEFT_HIP.value
    right_hip = mp_pose.PoseLandmark.RIGHT_HIP.value
    left_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    right_shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER.value

    hip_center = center_point(landmarks, left_hip, right_hip)
    shoulder_center = center_point(landmarks, left_shoulder, right_shoulder)

    anchor_points = [hip_center, shoulder_center]
    visible_points = landmarks[valid, :3]
    if visible_points.size:
        anchor_points.append(np.mean(visible_points, axis=0))

    center = np.mean(np.asarray(anchor_points), axis=0)

    shoulder_width = np.linalg.norm(
        landmarks[left_shoulder, :3] - landmarks[right_shoulder, :3]
    )
    hip_width = np.linalg.norm(landmarks[left_hip, :3] - landmarks[right_hip, :3])
    torso_height = np.linalg.norm(shoulder_center - hip_center)

    if visible_points.size:
        span = np.max(visible_points, axis=0) - np.min(visible_points, axis=0)
        body_span = float(np.linalg.norm(span[:2]))
    else:
        body_span = 0.0

    scale = max(float(shoulder_width), float(hip_width), float(torso_height), body_span, 1e-6)
    normalized = landmarks.copy()
    normalized[:, :3] = (normalized[:, :3] - center) / scale
    return normalized, valid


def mirror_landmarks(landmarks: np.ndarray) -> np.ndarray:
    mirrored = landmarks.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]

    for left, right in LEFT_RIGHT_PAIRS:
        mirrored[[left, right]] = mirrored[[right, left]]

    return mirrored


def detect_pose(detector: mp_pose.Pose, image_bgr: np.ndarray) -> np.ndarray | None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    result = detector.process(image_rgb)
    return landmarks_to_array(result.pose_landmarks)


def load_references(folder: Path) -> list[ReferencePose]:
    references: list[ReferencePose] = []

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as detector:
        for image_path in iter_pose_images(folder):
            image = read_image(image_path)
            if image is None:
                print(f"Atlandi, okunamadi: {image_path.name}")
                continue

            landmarks = detect_pose(detector, image)
            if landmarks is None:
                print(f"Atlandi, poz bulunamadi: {image_path.name}")
                continue

            normalized, valid = normalize_landmarks(landmarks)
            references.append(
                ReferencePose(
                    name=image_path.name,
                    path=image_path,
                    image=image,
                    landmarks=landmarks,
                    normalized=normalized,
                    valid=valid,
                )
            )

    return references


def pose_distance(
    current_normalized: np.ndarray,
    current_valid: np.ndarray,
    reference: ReferencePose,
) -> float | None:
    common = current_valid & reference.valid
    if int(np.count_nonzero(common)) < MIN_COMMON_POINTS:
        return None

    current_points = current_normalized[common, :3]
    reference_points = reference.normalized[common, :3]

    visibility_weight = (
        current_normalized[common, 3].clip(0.05, 1.0)
        * reference.normalized[common, 3].clip(0.05, 1.0)
    )
    distances = np.linalg.norm(current_points - reference_points, axis=1)
    return float(np.average(distances, weights=visibility_weight))


def find_best_match(landmarks: np.ndarray, references: list[ReferencePose]) -> Match | None:
    normalized, valid = normalize_landmarks(landmarks)
    mirrored_normalized, mirrored_valid = normalize_landmarks(mirror_landmarks(landmarks))

    best: Match | None = None
    for reference in references:
        distance = pose_distance(normalized, valid, reference)
        if distance is not None and (best is None or distance < best.distance):
            best = Match(reference=reference, distance=distance, mirrored=False)

        mirrored_distance = pose_distance(mirrored_normalized, mirrored_valid, reference)
        if mirrored_distance is not None and (
            best is None or mirrored_distance < best.distance
        ):
            best = Match(reference=reference, distance=mirrored_distance, mirrored=True)

    return best


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
) -> np.ndarray:
    panel = np.full((frame_height, panel_width, 3), (26, 28, 32), dtype=np.uint8)
    padding = 18
    image_height = max(180, frame_height - 170)
    preview = None

    if match is not None:
        preview = fit_image(match.reference.image, panel_width - padding * 2, image_height)
        panel[padding : padding + image_height, padding : panel_width - padding] = preview
        label = match.reference.name
        if len(label) > 34:
            label = label[:31] + "..."

        status = "Zayif eslesme" if match.distance > threshold else "Eslesme"
        status_color = (70, 210, 255) if match.distance <= threshold else (80, 170, 255)
        draw_text(panel, status, (padding, frame_height - 120), 0.7, status_color)
        draw_text(panel, label, (padding, frame_height - 82), 0.55)
        draw_text(panel, f"Skor: {match.distance:.3f}", (padding, frame_height - 50), 0.55)
    else:
        draw_text(panel, "Referans bekleniyor", (padding, 58), 0.7, (80, 170, 255))
        if not references:
            draw_text(panel, "poses klasorunde", (padding, 102), 0.58)
            draw_text(panel, "poz okunabilen", (padding, 134), 0.58)
            draw_text(panel, "fotograf yok.", (padding, 166), 0.58)
        elif not camera_has_pose:
            draw_text(panel, "Kamerada poz", (padding, 102), 0.58)
            draw_text(panel, "bulunamadi.", (padding, 134), 0.58)

    draw_text(panel, "Q/ESC: cikis  R: yenile", (padding, frame_height - 18), 0.48, (190, 190, 190), 1)
    return panel


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

    capture = open_camera(args.camera, args.width, args.height)
    if not capture.isOpened():
        print("Kamera acilamadi. Farkli bir indeks deneyin: --camera 1")
        return 1

    window_name = "Pose Matcher"

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as detector:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Kameradan goruntu alinamadi.")
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            result = detector.process(image_rgb)

            camera_landmarks = landmarks_to_array(result.pose_landmarks)
            camera_has_pose = camera_landmarks is not None
            match = find_best_match(camera_landmarks, references) if camera_has_pose else None

            annotated = frame.copy()
            if result.pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    annotated,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            panel_width = max(360, annotated.shape[1] // 3)
            panel = draw_panel(
                frame_height=annotated.shape[0],
                panel_width=panel_width,
                match=match,
                references=references,
                threshold=args.threshold,
                camera_has_pose=camera_has_pose,
            )
            output = np.hstack((annotated, panel))
            cv2.imshow(window_name, output)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                references = load_references(args.poses)
                print(f"{len(references)} referans poz yeniden yuklendi.")

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
