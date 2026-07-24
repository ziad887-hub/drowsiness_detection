import pytest
import numpy as np
from src.drowsiness_detection.utils import normalize_pupil_ratio, calculate_ear

def test_normalize_pupil_ratio_valid_input():
    """Test that normalize_pupil_ratio returns values between 0 and 1."""
    result = normalize_pupil_ratio(2.0, 10.0)
    assert 0 <= result <= 1

def test_normalize_pupil_ratio_zero_eye_width():
    """Test that zero division raises appropriate error."""
    with pytest.raises(ZeroDivisionError):
        normalize_pupil_ratio(1.0, 0.0)

def test_calculate_ear_valid_eye():
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
    ear = calculate_ear(eye)
    assert isinstance(ear, float)
    # EAR should be greater than zero for a non‑degenerate eye
    assert ear > 0.0

def test_calculate_ear_degenerate_eye():
    """Test with a degenerate eye (all points same)."""
    eye = np.array([
        [10, 10],
        [10, 10],
        [10, 10],
        [10, 10],
        [10, 10],
        [10, 10]
    ])
    ear = calculate_ear(eye)
    assert ear == 0.0  # Should be zero for degenerate eye

def test_calculate_ear_single_point():
    """Test with a single point (edge case)."""
    eye = np.array([[50, 50]])
    ear = calculate_ear(eye)
    assert ear == 0.0  # Single point can't form an eye