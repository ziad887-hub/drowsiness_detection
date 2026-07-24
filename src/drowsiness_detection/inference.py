"""
Real‑time drowsiness detection inference engine.
Uses MediaPipe FaceMesh for facial landmarks and computes
Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) for fatigue estimation.
"""
import os
import json
import time
import logging
from datetime import datetime
from collections import deque
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DrowsinessDetector:
    """
    Advanced driver drowsiness detection targeting 95%+ accuracy
    using Eye Aspect Ratio (EAR) and facial landmark analysis.
    """

    # MediaPipe FaceMesh landmark indices for eyes and mouth
    LEFT_EYE_INDICES = [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    ]
    RIGHT_EYE_INDICES = [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    ]
    MOUTH_INDICES = [
        61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318
    ]

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the detector.

        Args:
            config_path: Path to a JSON config file. If None, uses default config.
        """
        self.config = self._load_config(config_path)

        # Detection thresholds from config (with sensible defaults)
        self.ear_threshold = self.config.get('ear_threshold', 0.25)
        self.yawn_threshold = self.config.get('yawn_threshold', 0.6)
        self.confidence_threshold = self.config.get('detection_threshold', 0.95)
        self.fatigue_threshold = self.config.get('fatigue_score_threshold', 0.7)

        # Runtime state
        self.consecutive_drowsy_frames = 0
        self.total_alerts = 0
        self.session_start_time = time.time()
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

        # Alert cooldown
        self.alert_cooldown_time = self.config.get('alert_cooldown_time', 2.0)
        self.last_alert_time = 0.0

        # History for analytics
        self.detection_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=100)

        # MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Create output directories
        self._create_directories()

        logger.info(
            f"DrowsinessDetector initialized – target accuracy: {self.confidence_threshold*100:.1f}%"
        )

    # ------------------------------------------------------------------ #
    # Configuration helpers
    # ------------------------------------------------------------------ #
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load JSON configuration, falling back to defaults."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return cfg
            except Exception as e:
                logger.warning(f"Could not load config {config_path}: {e}. Using defaults.")
        return {}

    def _create_directories(self) -> None:
        """Ensure required output directories exist."""
        for d in ['frames', 'logs']:
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Core geometry calculations
    # ------------------------------------------------------------------ #
    @staticmethod
    def _calculate_ear(eye_pts: np.ndarray) -> float:
        """
        Eye Aspect Ratio (EAR) for a 6‑point eye landmark set.
        """
        if len(eye_pts) < 6:
            return 0.0
        # Vertical distances
        A = np.linalg.norm(eye_pts[1] - eye_pts[5])
        B = np.linalg.norm(eye_pts[2] - eye_pts[4])
        # Horizontal distance
        C = np.linalg.norm(eye_pts[0] - eye_pts[3])
        return (A + B) / (2.0 * C) if C != 0 else 0.0

    @staticmethod
    def _calculate_mar(mouth_pts: np.ndarray) -> float:
        """
        Mouth Aspect Ratio (MAR) for yawn detection.
        """
        if len(mouth_pts) < 6:
            return 0.0
        # Vertical distances
        A = np.linalg.norm(mouth_pts[2] - mouth_pts[10])
        B = np.linalg.norm(mouth_pts[4] - mouth_pts[8])
        # Horizontal distance
        C = np.linalg.norm(mouth_pts[0] - mouth_pts[6])
        return (A + B) / (2.0 * C) if C != 0 else 0.0

    # ------------------------------------------------------------------ #
    # Landmark extraction
    # ------------------------------------------------------------------ #
    def _extract_landmarks(
        self,
        face_landmarks,
        img_w: int,
        img_h: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert MediaPipe normalized landmarks to pixel coordinates.
        Returns (left_eye, right_eye, mouth) as Nx2 int arrays.
        """
        left_eye = []
        right_eye = []
        mouth = []

        for idx in self.LEFT_EYE_INDICES:
            if idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[idx]
                left_eye.append([int(lm.x * img_w), int(lm.y * img_h)])

        for idx in self.RIGHT_EYE_INDICES:
            if idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[idx]
                right_eye.append([int(lm.x * img_w), int(lm.y * img_h)])

        for idx in self.MOUTH_INDICES:
            if idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[idx]
                mouth.append([int(lm.x * img_w), int(lm.y * img_h)])

        return (
            np.array(left_eye) if left_eye else np.empty((0, 2), dtype=int),
            np.array(right_eye) if right_eye else np.empty((0, 2), dtype=int),
            np.array(mouth) if mouth else np.empty((0, 2), dtype=int),
        )

    # ------------------------------------------------------------------ #
    # Public inference API
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Process a single BGR frame and return a detection result dict.
        """
        result = {
            'drowsy': False,
            'confidence': 0.0,
            'fatigue_score': 0.0,
            'left_ear': 0.0,
            'right_ear': 0.0,
            'avg_ear': 0.0,
            'yawn_detected': False,
            'mouth_aspect_ratio': 0.0,
            'alert_triggered': False,
            'timestamp': datetime.now().isoformat()
        }

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = self.face_mesh.process(rgb)

            if mp_results.multi_face_landmarks:
                face_lms = mp_results.multi_face_landmarks[0]
                left_eye, right_eye, mouth = self._extract_landmarks(
                    face_lms, frame.shape[1], frame.shape[0]
                )

                # EAR
                if len(left_eye) >= 6:
                    result['left_ear'] = self._calculate_ear(left_eye)
                if len(right_eye) >= 6:
                    result['right_ear'] = self._calculate_ear(right_eye)
                if result['left_ear'] and result['right_ear']:
                    result['avg_ear'] = (result['left_ear'] + result['right_ear']) / 2.0

                # MAR / yawn
                if len(mouth) >= 6:
                    mar = self._calculate_mar(mouth)
                    result['mouth_aspect_ratio'] = mar
                    result['yawn_detected'] = mar > self.yawn_threshold

                # Fatigue score
                fatigue = 0.0
                if result['avg_ear'] > 0:
                    ear_fatigue = max(0.0, 1.0 - (result['avg_ear'] / self.ear_threshold))
                    fatigue += ear_fatigue * 0.7
                if result['yawn_detected']:
                    yawn_fatigue = min(1.0, result['mouth_aspect_ratio'] / self.yawn_threshold)
                    fatigue += yawn_fatigue * 0.3
                result['fatigue_score'] = min(1.0, fatigue)

                # Drowsy decision
                result['drowsy'] = result['fatigue_score'] > self.fatigue_threshold

                # Confidence
                if result['drowsy']:
                    self.consecutive_drowsy_frames += 1
                    result['confidence'] = min(0.95, 0.7 + self.consecutive_drowsy_frames * 0.05)
                else:
                    self.consecutive_drowsy_frames = 0
                    result['confidence'] = max(0.0, 1.0 - result['fatigue_score'])

                # Alert logic
                now = time.time()
                if (
                    result['drowsy']
                    and result['confidence'] >= self.confidence_threshold
                    and (now - self.last_alert_time) > self.alert_cooldown_time
                ):
                    result['alert_triggered'] = True
                    self.last_alert_time = now
                    self.total_alerts += 1
                    self.alert_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'confidence': result['confidence'],
                        'fatigue_score': result['fatigue_score']
                    })
                    logger.warning(
                        f"DROWSINESS ALERT! confidence={result['confidence']:.2f} "
                        f"fatigue={result['fatigue_score']:.2f}"
                    )

                # Record history
                self.detection_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'drowsy': result['drowsy'],
                    'confidence': result['confidence'],
                    'fatigue_score': result['fatigue_score'],
                    'left_ear': result['left_ear'],
                    'right_ear': result['right_ear']
                })

        except Exception as e:
            logger.error(f"Detection error: {e}")

        # Update FPS counter
        self.fps_counter += 1
        if time.time() - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (time.time() - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = time.time()

        return result

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #
    def draw_dashboard(self, frame: np.ndarray, res: Dict) -> np.ndarray:
        """Overlay a semi‑transparent dashboard with key metrics."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Dashboard box
        box_w, box_h = 300, 180
        cv2.rectangle(overlay, (10, 10), (box_w, box_h), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        y = 35
        line_h = 25

        # Status
        status_color = (0, 0, 255) if res['drowsy'] else (0, 255, 0)
        status_txt = "DROWSY DETECTED!" if res['drowsy'] else "ALERT"
        cv2.putText(frame, status_txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        y += line_h

        # Confidence
        cv2.putText(frame, f"Confidence: {res['confidence']:.2f}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += line_h

        # Fatigue
        fat_color = (0, 0, 255) if res['fatigue_score'] > 0.5 else (0, 255, 0)
        cv2.putText(frame, f"Fatigue: {res['fatigue_score']:.2f}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fat_color, 1)
        y += line_h

        # EAR
        cv2.putText(frame,
                    f"EAR L:{res['left_ear']:.2f} R:{res['right_ear']:.2f} Avg:{res['avg_ear']:.2f}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += line_h

        # Yawn
        yawn_color = (0, 0, 255) if res['yawn_detected'] else (0, 255, 0)
        cv2.putText(frame, f"Yawning: {'YES' if res['yawn_detected'] else 'NO'}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yawn_color, 1)
        y += line_h

        # FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += line_h

        # Alert count
        cv2.putText(frame, f"Alerts: {self.total_alerts}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Red border if highly confident drowsy
        if res['drowsy'] and res['confidence'] > 0.8:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)

        return frame

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_frame(self, frame: np.ndarray, prefix: str = "detection") -> str:
        """Save a frame to ./frames with timestamp."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fname = f"{prefix}_{ts}.jpg"
        fpath = os.path.join("frames", fname)
        cv2.imwrite(fpath, frame)
        return fpath

    def get_performance_stats(self) -> Dict:
        """Return a dict of session‑level performance metrics."""
        duration = time.time() - self.session_start_time
        drowsy_cnt = sum(1 for d in self.detection_history if d['drowsy'])
        total = len(self.detection_history)
        drowsy_pct = drowsy_cnt / max(1, total) * 100
        drowsy_confs = [d['confidence'] for d in self.detection_history if d['drowsy']]
        avg_conf = float(np.mean(drowsy_confs)) if drowsy_confs else 0.0

        return {
            'session_duration_seconds': round(duration, 2),
            'total_detections': total,
            'drowsy_detections': drowsy_cnt,
            'drowsy_percentage': round(drowsy_pct, 2),
            'average_confidence_when_drowsy': round(avg_conf, 3),
            'total_alerts': self.total_alerts,
            'current_fps': round(self.current_fps, 1),
            'target_accuracy': self.confidence_threshold * 100
        }

    def save_session_report(self, stats: Dict) -> None:
        """Write a JSON session report to ./logs."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join("logs", f"session_report_{ts}.json")
        report = {
            'session_info': {
                'start_time': datetime.fromtimestamp(self.session_start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': stats['session_duration_seconds']
            },
            'performance_metrics': stats,
            'detection_parameters': {
                'ear_threshold': self.ear_threshold,
                'yawn_threshold': self.yawn_threshold,
                'confidence_threshold': self.confidence_threshold,
                'fatigue_threshold': self.fatigue_threshold
            },
            'alert_history': list(self.alert_history)
        }
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Session report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save session report: {e}")

    # ------------------------------------------------------------------ #
    # Convenience run loop (optional)
    # ------------------------------------------------------------------ #
    def run_camera(self, cam_index: int = 0) -> None:
        """
        Open a webcam and run the detector in a blocking loop.
        Press 'q' to quit, 's' to save current frame.
        """
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            logger.error("Could not open video device")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,
                self.config.get('camera_resolution', {}).get('width', 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,
                self.config.get('camera_resolution', {}).get('height', 720))

        logger.info("Starting camera loop – press 'q' to quit, 's' to save frame")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab frame")
                    break

                res = self.detect(frame)
                vis = self.draw_dashboard(frame, res)

                cv2.imshow('Drowsiness Detection', vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    path = self.save_frame(vis, "manual")
                    logger.info(f"Frame saved to {path}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            stats = self.get_performance_stats()
            logger.info("=== SESSION SUMMARY ===")
            for k, v in stats.items():
                logger.info(f"{k}: {v}")
            self.save_session_report(stats)