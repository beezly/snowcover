"""Falling snow detection using frame differencing and particle analysis."""

from dataclasses import dataclass

import cv2
import numpy as np
import structlog

from snowcover.config import FallingSnowConfig

logger = structlog.get_logger()


@dataclass
class FallingSnowResult:
    """Result of falling snow detection."""

    is_snowing: bool
    particle_count: int
    confidence: float
    avg_brightness: float
    is_ir_mode: bool = False


# Threshold for detecting IR mode based on color saturation
IR_SATURATION_THRESHOLD = 15.0


class FallingSnowDetector:
    """Detect falling snow using temporal frame differencing.

    This detector works by:
    1. Computing the absolute difference between consecutive frames
    2. Finding contours (potential particles) in the difference image
    3. Filtering particles by size and brightness to identify snow
    4. Counting particles and comparing to threshold
    """

    def __init__(self, config: FallingSnowConfig):
        """Initialize the falling snow detector.

        Args:
            config: Detection configuration
        """
        self.config = config
        self._log = logger.bind(component="falling_snow")

    def _is_ir_mode(self, frame: np.ndarray) -> bool:
        """Detect if the camera is in IR/night mode.

        Args:
            frame: BGR frame

        Returns:
            True if the frame appears to be from IR/night mode
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation)
        return avg_saturation < IR_SATURATION_THRESHOLD

    def detect(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray,
    ) -> FallingSnowResult:
        """Detect falling snow between two consecutive frames.

        Automatically detects IR/night mode and adjusts thresholds accordingly.

        Args:
            current_frame: Current frame (BGR)
            previous_frame: Previous frame (BGR)

        Returns:
            Detection result with snow presence and particle count
        """
        # Detect IR mode and select appropriate thresholds
        is_ir = self._is_ir_mode(current_frame)
        if is_ir:
            motion_threshold = self.config.ir_motion_threshold
            brightness_threshold = self.config.ir_brightness_threshold
            self._log.debug("Using IR mode thresholds")
        else:
            motion_threshold = self.config.motion_threshold
            brightness_threshold = self.config.brightness_threshold

        # Convert to grayscale
        gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray_curr = cv2.GaussianBlur(gray_curr, (5, 5), 0)
        gray_prev = cv2.GaussianBlur(gray_prev, (5, 5), 0)

        # Compute absolute difference
        diff = cv2.absdiff(gray_curr, gray_prev)

        # Threshold to binary
        _, thresh = cv2.threshold(
            diff,
            motion_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        # Find contours (potential particles)
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Filter particles by size and brightness
        snow_particles = []
        total_brightness = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by size
            if not (self.config.min_particle_size <= area <= self.config.max_particle_size):
                continue

            # Get bounding rectangle to check brightness in original frame
            x, y, w, h = cv2.boundingRect(contour)

            # Ensure bounds are within frame
            x = max(0, x)
            y = max(0, y)
            x2 = min(current_frame.shape[1], x + w)
            y2 = min(current_frame.shape[0], y + h)

            # Extract region and check brightness
            region = gray_curr[y:y2, x:x2]
            if region.size == 0:
                continue

            avg_brightness = np.mean(region)

            # Filter by brightness (snow is bright/white, reflects IR)
            if avg_brightness >= brightness_threshold:
                snow_particles.append({
                    "contour": contour,
                    "area": area,
                    "brightness": avg_brightness,
                    "center": (x + w // 2, y + h // 2),
                })
                total_brightness += avg_brightness

        particle_count = len(snow_particles)
        is_snowing = particle_count >= self.config.min_particle_count

        # Calculate confidence based on how far above threshold we are
        if particle_count == 0:
            confidence = 0.0
        else:
            # Confidence scales from 0.5 at threshold to 1.0 at 2x threshold
            ratio = particle_count / self.config.min_particle_count
            confidence = min(1.0, 0.5 + (ratio - 1.0) * 0.5) if ratio >= 1.0 else ratio * 0.5

        avg_brightness = total_brightness / particle_count if particle_count > 0 else 0.0

        return FallingSnowResult(
            is_snowing=is_snowing,
            particle_count=particle_count,
            confidence=confidence,
            avg_brightness=avg_brightness,
            is_ir_mode=is_ir,
        )

    def detect_with_intensity(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray,
    ) -> tuple[FallingSnowResult, int]:
        """Detect falling snow and estimate intensity from particle density.

        Args:
            current_frame: Current frame (BGR)
            previous_frame: Previous frame (BGR)

        Returns:
            Tuple of (detection result, intensity percentage 0-100)
        """
        result = self.detect(current_frame, previous_frame)

        # Estimate intensity based on particle count
        # Scale: min_particle_count = light snow, 5x = heavy snow
        if result.particle_count == 0:
            intensity = 0
        else:
            # Map particle count to 0-100 scale
            # Using a logarithmic scale for better sensitivity at lower counts
            min_count = self.config.min_particle_count
            max_count = min_count * 5  # 5x threshold = 100% intensity

            if result.particle_count < min_count:
                intensity = int((result.particle_count / min_count) * 25)
            else:
                # Scale from 25-100 for counts at or above threshold
                ratio = (result.particle_count - min_count) / (max_count - min_count)
                intensity = int(25 + min(1.0, ratio) * 75)

        return result, min(100, intensity)
