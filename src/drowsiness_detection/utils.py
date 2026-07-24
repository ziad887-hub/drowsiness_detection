"""
Utility functions for the drowsiness detection project.
Contains helpers for image processing, geometry, and data I/O.
"""
import cv2
import numpy as np
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional, Any, Dict

# Configure module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# --------------------------------------------------------------------------- #
# Geometry / landmark helpers
# --------------------------------------------------------------------------- #
def eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) given six eye landmarks.

    Args:
        eye: (6, 2) array of (x, y) points ordered as:
             [0] outer corner, [1] upper inner, [2] lower inner,
             [3] inner corner, [4] lower outer, [5] upper outer.

    Returns:
        EAR value.
    """
    if eye.shape != (6, 2):
        raise ValueError(f"Expected shape (6, 2), got {eye.shape}")
    # vertical distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # horizontal distance
    C = np.linalg.norm(eye[0] - eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    """
    Compute Mouth Aspect Ratio (MAR) for yawn detection.

    Args:
        mouth: (12, 2) array of mouth landmarks (MediaPipe indices 61–308).

    Returns:
        MAR value.
    """
    if mouth.shape[0] < 6:
        return 0.0
    # vertical distances (inner)
    A = np.linalg.norm(mouth[2] - mouth[10])   # top–bottom inner
    B = np.linalg.norm(mouth[4] - mouth[8])    # top–bottom outer
    # horizontal distance (outer corners)
    C = np.linalg.norm(mouth[0] - mouth[6])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def normalize_landmarks(landmarks: np.ndarray,
                        image_w: int,
                        image_h: int) -> np.ndarray:
    """
    Normalise landmark coordinates to [0, 1] range.

    Args:
        landmarks: (N, 2) pixel coordinates.
        image_w:   Frame width.
        image_h:   Frame height.

    Returns:
        Normalised landmarks in [0, 1].
    """
    norm = landmarks.astype(np.float32)
    norm[:, 0] /= image_w
    norm[:, 1] /= image_h
    return norm


def denormalize_landmarks(landmarks: np.ndarray,
                          image_w: int,
                          image_h: int) -> np.ndarray:
    """
    Convert normalised [0, 1] landmarks back to pixel coordinates.
    """
    denorm = landmarks.copy()
    denorm[:, 0] *= image_w
    denorm[:, 1] *= image_h
    return denorm.astype(int)


# --------------------------------------------------------------------------- #
# Image I/O helpers
# --------------------------------------------------------------------------- #
def read_image(path: str, grayscale: bool = False) -> Optional[np.ndarray]:
    """
    Safe image read that logs a warning if the file cannot be loaded.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if img is None:
        logger.warning(f"Failed to read image: {path}")
    return img


def write_image(path: str, image: np.ndarray) -> bool:
    """
    Safe image write with directory creation.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        cv2.imwrite(path, image)
        return True
    except Exception as e:
        logger.error(f"Could not write image to {path}: {e}")
        return False


def list_images(root: str, exts=('.jpg', '.jpeg', '.png', '.bmp')) -> List[str]:
    """
    Recursively list image files under root.
    """
    root = Path(root)
    return [str(p) for p in root.rglob('*') if p.suffix.lower() in exts]


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #
def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# --------------------------------------------------------------------------- #
# Session / logging helpers
# --------------------------------------------------------------------------- #
def init_session_dir(base: str = "runs") -> Path:
    """
    Create a timestamped run directory and return its path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "frames").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    logger.info(f"Session directory created at {run_dir}")
    return run_dir


def log_metrics(run_dir: Path, metrics: Dict, step: int) -> None:
    """
    Append a line of metrics to a CSV log file.
    """
    log_file = run_dir / "metrics.csv"
    header = not log_file.exists()
    with open(log_file, 'a') as f:
        if header:
            f.write("step," + ",".join(metrics.keys()) + "\n")
        f.write(str(step) + "," + ",".join(str(v) for v in metrics.values()) + "\n")


# --------------------------------------------------------------------------- #
# MediaPipe landmark index constants (for quick access)
# --------------------------------------------------------------------------- #
# MediaPipe Face Mesh (468 points) – eye and mouth subsets
LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_IDX = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]


def extract_eye_mouth_landmarks(face_landmarks,
                                image_w: int,
                                image_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract left eye, right eye, and mouth landmarks from a MediaPipe face mesh.

    Returns:
        left_eye  (6, 2) pixel coords
        right_eye (6, 2) pixel coords
        mouth    (12, 2) pixel coords
    """
    def pick(idxs):
        return np.array([
            [int(face_landmarks.landmark[i].x * image_w),
             int(face_landmarks.landmark[i].y * image_h)]
            for i in idxs
        ], dtype=int)

    left = pick(LEFT_EYE_IDX)
    right = pick(RIGHT_EYE_IDX)
    mouth = pick(MOUTH_IDX)
    return left, right, mouth


# --------------------------------------------------------------------------- #
# Simple drawing utilities (used for debugging / dashboard)
# --------------------------------------------------------------------------- #
def draw_landmarks(img: np.ndarray, points: np.ndarray,
                   color: Tuple[int, int, int] = (0, 255, 0),
                   radius: int = 1, thickness: int = -1) -> None:
    for pt in points:
        cv2.circle(img, tuple(pt), radius, color, thickness)


def draw_polygon(img: np.ndarray, points: np.ndarray,
                 color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 1) -> None:
    if len(points) >= 3:
        cv2.polylines(img, [points.astype(np.int32)], isClosed=True,
                      color=color, thickness=thickness)


# --------------------------------------------------------------------------- #
# Quick self‑test when executed directly
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Simple sanity checks
    dummy_eye = np.array([[0, 0], [0, 1], [1, 1], [2, 0], [2, 1], [1, 0]], dtype=float)
    ear = eye_aspect_ratio(dummy_eye)
    print(f"Dummy EAR: {ear:.3f}")

    dummy_mouth = np.zeros((12, 2))
    dummy_mouth[2] = [0, 5]
    dummy_mouth[10] = [0, 0]
    dummy_mouth[4] = [2, 5]
    dummy_mouth[8] = [2, 0]
    dummy_mouth[0] = [0, 2]
    dummy_mouth[6] = [2, 2]
    mar = mouth_aspect_ratio(dummy_mouth)
    print(f"Dummy MAR: {mar:.3f}")

    print("utils module self‑test passed.")