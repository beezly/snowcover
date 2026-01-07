"""Image processing utilities including IR/night mode detection."""

import cv2
import numpy as np


def is_ir_mode(frame: np.ndarray, saturation_threshold: float = 15.0) -> bool:
    """Detect if the camera is in IR/night mode.

    IR mode typically produces grayscale images with very low color saturation.
    This function checks if the frame has minimal color information.

    Args:
        frame: BGR frame
        saturation_threshold: Maximum average saturation to consider IR mode

    Returns:
        True if the frame appears to be from IR/night mode
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get saturation channel
    saturation = hsv[:, :, 1]

    # Calculate average saturation
    avg_saturation = np.mean(saturation)

    # IR mode typically has very low saturation (nearly grayscale)
    return avg_saturation < saturation_threshold


def get_frame_brightness(frame: np.ndarray) -> float:
    """Calculate the average brightness of a frame.

    Args:
        frame: BGR frame

    Returns:
        Average brightness (0-255)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def enhance_ir_frame(frame: np.ndarray) -> np.ndarray:
    """Enhance an IR frame for better detection.

    Applies contrast enhancement to help with particle detection
    in low-contrast IR scenes.

    Args:
        frame: BGR frame (from IR camera)

    Returns:
        Enhanced frame
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Convert back to BGR for consistency
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def detect_snow_brightness_ir(
    frame: np.ndarray,
    ground_mask: np.ndarray | None = None,
    brightness_threshold: int = 200,
) -> tuple[float, int]:
    """Detect snow coverage using brightness in IR mode.

    In IR mode, snow appears as bright areas. This function counts
    bright pixels as potential snow.

    Args:
        frame: BGR frame (IR mode)
        ground_mask: Optional mask for ground region
        brightness_threshold: Minimum brightness to count as snow

    Returns:
        Tuple of (coverage_percent, snow_pixel_count)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply ground mask if provided
    if ground_mask is not None:
        gray = cv2.bitwise_and(gray, ground_mask)
        total_pixels = cv2.countNonZero(ground_mask)
    else:
        total_pixels = gray.shape[0] * gray.shape[1]

    # Threshold for bright pixels (snow)
    _, bright_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

    snow_pixels = cv2.countNonZero(bright_mask)

    if total_pixels > 0:
        coverage = (snow_pixels / total_pixels) * 100
    else:
        coverage = 0.0

    return coverage, snow_pixels
