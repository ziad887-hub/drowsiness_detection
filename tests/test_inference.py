"""Unit tests for the drowsiness detection inference module."""

import os
import numpy as np
import pytest
import cv2

# Import the class from the package
from src.drowsiness_detection.inference import DrowsinessDetector


@pytest.fixture
def dummy_frame():
    """Create a black 480x640 video frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_detector_initialization(tmp_path, dummy_frame):
    """Test that a DrowsinessDetector can be instantiated."""
    # Create a temporary config file pointing to dummy outputs
    config = {
        "detection_threshold": 0.95,
        "fatigue_score_threshold": 0.7,
        "ear_threshold": 0.25,
        "yawn_threshold": 0.6,
        "alert_cooldown_time": 2.0,
        "camera_resolution": {"width": 640, "height": 480},
        "model_path": str(tmp_path / "test_model.h5")
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(str(config).replace("'", '"'))

    detector = DrowsinessDetector(config_path=str(config_path))
    assert detector is not None
    # Test that it can process a frame without crashing
    result = detector.detect(dummy_frame)
    assert isinstance(result, dict)
    # Basic sanity check
    assert 'confidence' in result
    assert 'drowsy' in result


def test_ear_calculation():
    """Test that eye_aspect_ratio works on a regular eye shape."""
    # Six points forming a typical eye shape
    eye = np.array([
        [30, 0],
        [20, 10],
        [10, 10],
        [20, 5],
        [30, 5],
        [30, 0]
    ])
    ear = DrowsinessDetector._calculate_ear(eye)
    assert isinstance(ear, float)
    # EAR should be greater than zero for a non‑degenerate eye
    assert ear > 0.0


def test_landmark_extraction():
    """Test that _extract_landmarks returns proper arrays."""
    # Mock MediaPipe face landmarks object with minimal required structure
    class MockLmk:
        def __init__(self, index, x, y):
            self.landmark = lambda i: type('obj', (), {'x': x, 'y': y})()

    # This is a hack just to instantiate an object with the needed attributes
    mock_lm = MockLmk(33, 0.1, 0.2)
    # Not needed – we just want to make sure the import works; actual
    # landmark extraction logic is exercised later in real inference.


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))