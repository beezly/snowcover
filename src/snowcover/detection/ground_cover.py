"""Ground snow cover detection using HSV color analysis."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import structlog

from snowcover.config import GroundCoverConfig

logger = structlog.get_logger()


@dataclass
class GroundCoverResult:
    """Result of ground cover detection."""

    has_snow: bool
    coverage_percent: float
    ground_pixels: int
    snow_pixels: int
    avg_brightness: float
    confidence: float = 0.0
    is_ir_mode: bool = False


class GroundCoverDetector:
    """Detect snow cover on the ground using HSV color analysis.

    This detector works by:
    1. Creating a mask for the ground region of the frame (auto-detected or manual)
    2. Converting the frame to HSV color space
    3. Detecting white/snow-colored pixels (high Value, low Saturation)
    4. Calculating the percentage of ground covered by snow
    """

    def __init__(self, config: GroundCoverConfig):
        """Initialize the ground cover detector.

        Args:
            config: Detection configuration
        """
        self.config = config
        self._log = logger.bind(component="ground_cover")
        self._ground_mask: np.ndarray | None = None
        self._mask_shape: tuple | None = None
        self._auto_detected: bool = False
        self._segmenter = None
        self._last_ir_mode: bool | None = None  # Track IR mode transitions

        # Initialize auto-detection if enabled
        if config.auto_detect:
            self._init_segmenter()

    def _init_segmenter(self) -> None:
        """Initialize the ground segmentation model."""
        try:
            from snowcover.detection.ground_segmentation import GroundSegmenter

            model_path = Path(self.config.segmentation_model)
            self._segmenter = GroundSegmenter(model_path)

            if self._segmenter.is_available:
                self._log.info("Ground auto-detection enabled")
            else:
                self._log.warning(
                    "Ground segmentation model not found, using manual region",
                    model_path=str(model_path),
                )
                self._segmenter = None
        except Exception as e:
            self._log.error("Failed to initialize ground segmenter", error=str(e))
            self._segmenter = None

    @property
    def is_auto_detected(self) -> bool:
        """Check if ground region was auto-detected."""
        return self._auto_detected

    @property
    def ground_mask(self) -> np.ndarray | None:
        """Get the current ground mask (for visualization)."""
        return self._ground_mask

    def _create_ground_mask(self, frame_shape: tuple) -> np.ndarray:
        """Create a mask for the ground region.

        Args:
            frame_shape: Shape of the frame (height, width, channels)

        Returns:
            Binary mask where ground region is white (255)
        """
        height, width = frame_shape[:2]

        # Convert percentage-based polygon to pixel coordinates
        polygon_points = []
        for point in self.config.ground_region:
            x = int(point[0] * width)
            y = int(point[1] * height)
            polygon_points.append([x, y])

        polygon = np.array(polygon_points, dtype=np.int32)

        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        return mask

    def _get_ground_mask(self, frame: np.ndarray, is_ir_mode: bool = False) -> np.ndarray:
        """Get or create the ground mask for the given frame size.

        Uses auto-detection if available and in color mode.
        If already auto-detected during daytime, keeps using that mask at night.
        Falls back to manual polygon only if auto-detection has never succeeded.

        Args:
            frame: Input frame
            is_ir_mode: Whether the camera is in IR/night mode

        Returns:
            Ground region mask
        """
        frame_shape = frame.shape[:2]

        # Check if frame size changed - need to recalculate mask
        if self._mask_shape != frame_shape:
            self._ground_mask = None
            self._mask_shape = None
            self._auto_detected = False

        # Return cached mask if available
        # This allows daytime-detected masks to be used at night
        if self._ground_mask is not None:
            return self._ground_mask

        # Try auto-detection, but only in color/daytime mode
        # The ML model was trained on RGB images and doesn't work well with IR/monochrome
        if self._segmenter is not None and not is_ir_mode:
            auto_mask = self._segmenter.detect_ground(frame, use_cache=True)
            if auto_mask is not None:
                # Validate that auto-detection found a reasonable ground region
                ground_percent = (np.sum(auto_mask > 0) / auto_mask.size) * 100
                if ground_percent >= 5.0:  # At least 5% of frame should be ground
                    self._ground_mask = auto_mask
                    self._mask_shape = frame_shape
                    self._auto_detected = True
                    self._log.info("Using auto-detected ground region", coverage_pct=round(ground_percent, 1))
                    return self._ground_mask
                else:
                    self._log.warning(
                        "Auto-detection found too little ground, falling back to manual",
                        detected_pct=round(ground_percent, 1),
                    )
        elif is_ir_mode and self._segmenter is not None:
            self._log.info(
                "In IR/night mode without cached ground mask - using manual region. "
                "Ground will be auto-detected when camera switches to daytime mode."
            )

        # Fall back to manual polygon
        self._ground_mask = self._create_ground_mask(frame.shape)
        self._mask_shape = frame_shape
        self._auto_detected = False
        self._log.info("Using manual ground region polygon")

        return self._ground_mask

    def _calculate_confidence(self, coverage_percent: float) -> float:
        """Calculate confidence based on distance from threshold.

        Confidence is higher when coverage is clearly above or below threshold,
        and lower when it's near the threshold (ambiguous).

        Args:
            coverage_percent: Detected coverage percentage

        Returns:
            Confidence value 0.0-1.0
        """
        threshold = self.config.coverage_threshold

        if coverage_percent >= threshold:
            # Snow detected - confidence increases as coverage exceeds threshold
            # At threshold: 0.5, at 2x threshold: ~0.9, at 100%: 1.0
            excess = coverage_percent - threshold
            max_excess = 100 - threshold
            if max_excess > 0:
                confidence = 0.5 + 0.5 * min(1.0, excess / max_excess)
            else:
                confidence = 1.0
        else:
            # No snow detected - confidence increases as coverage approaches 0
            # At threshold: 0.5, at 0%: 1.0
            if threshold > 0:
                confidence = 0.5 + 0.5 * (1.0 - coverage_percent / threshold)
            else:
                confidence = 1.0

        return round(confidence, 3)

    def _is_ir_mode(self, frame: np.ndarray) -> bool:
        """Detect if the camera is in IR/night mode.

        IR mode produces grayscale images with very low color saturation.

        Args:
            frame: BGR frame

        Returns:
            True if the frame appears to be from IR/night mode
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation)
        return avg_saturation < self.config.ir_saturation_threshold

    def _detect_ir_mode(self, frame: np.ndarray, ground_mask: np.ndarray) -> GroundCoverResult:
        """Detect snow cover using brightness-based method for IR mode.

        In IR mode, we can't use color information, so we rely on
        brightness thresholding instead.

        Args:
            frame: BGR frame (IR mode)
            ground_mask: Ground region mask

        Returns:
            Detection result
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply ground mask
        ground_region = cv2.bitwise_and(gray, ground_mask)

        # Threshold for bright pixels (snow reflects IR strongly)
        _, snow_mask = cv2.threshold(
            ground_region,
            self.config.ir_brightness_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        # Count pixels
        ground_pixels = cv2.countNonZero(ground_mask)
        snow_pixels = cv2.countNonZero(snow_mask)

        # Calculate coverage
        if ground_pixels > 0:
            coverage_percent = (snow_pixels / ground_pixels) * 100
            avg_brightness = float(np.sum(ground_region) / ground_pixels)
        else:
            coverage_percent = 0.0
            avg_brightness = 0.0

        has_snow = coverage_percent >= self.config.coverage_threshold

        # Calculate confidence based on distance from threshold
        confidence = self._calculate_confidence(coverage_percent)

        return GroundCoverResult(
            has_snow=has_snow,
            coverage_percent=round(coverage_percent, 1),
            ground_pixels=ground_pixels,
            snow_pixels=snow_pixels,
            avg_brightness=avg_brightness,
            confidence=confidence,
            is_ir_mode=True,
        )

    def _detect_color_mode(self, frame: np.ndarray, ground_mask: np.ndarray) -> GroundCoverResult:
        """Detect snow cover using HSV color analysis for daytime mode.

        Args:
            frame: BGR frame
            ground_mask: Ground region mask

        Returns:
            Detection result
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create snow color mask
        # Snow characteristics: Any hue, low saturation, high value (bright)
        lower_snow = np.array(self.config.snow_hsv_lower, dtype=np.uint8)
        upper_snow = np.array(self.config.snow_hsv_upper, dtype=np.uint8)
        snow_mask = cv2.inRange(hsv, lower_snow, upper_snow)

        # Apply ground region mask
        snow_in_ground = cv2.bitwise_and(snow_mask, ground_mask)

        # Count pixels
        ground_pixels = cv2.countNonZero(ground_mask)
        snow_pixels = cv2.countNonZero(snow_in_ground)

        # Calculate coverage percentage
        if ground_pixels > 0:
            coverage_percent = (snow_pixels / ground_pixels) * 100
        else:
            coverage_percent = 0.0

        # Calculate average brightness in ground region
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ground_region = cv2.bitwise_and(gray, ground_mask)
        if ground_pixels > 0:
            avg_brightness = float(np.sum(ground_region) / ground_pixels)
        else:
            avg_brightness = 0.0

        has_snow = coverage_percent >= self.config.coverage_threshold

        # Calculate confidence based on distance from threshold
        confidence = self._calculate_confidence(coverage_percent)

        return GroundCoverResult(
            has_snow=has_snow,
            coverage_percent=round(coverage_percent, 1),
            ground_pixels=ground_pixels,
            snow_pixels=snow_pixels,
            avg_brightness=avg_brightness,
            confidence=confidence,
            is_ir_mode=False,
        )

    def detect(self, frame: np.ndarray) -> GroundCoverResult:
        """Detect snow cover on the ground.

        Automatically detects IR/night mode and uses the appropriate
        algorithm (brightness-based for IR, HSV color-based for day).

        Args:
            frame: BGR frame

        Returns:
            Detection result with coverage percentage
        """
        # Check IR mode first - this affects ground mask detection
        is_ir = self._is_ir_mode(frame)

        # Get ground mask, passing IR mode so we don't run ML on monochrome images
        ground_mask = self._get_ground_mask(frame, is_ir_mode=is_ir)

        # Use appropriate detection method based on camera mode
        if is_ir:
            self._log.debug("Using IR mode detection")
            return self._detect_ir_mode(frame, ground_mask)
        else:
            return self._detect_color_mode(frame, ground_mask)

    def detect_with_baseline(
        self,
        frame: np.ndarray,
        baseline_brightness: float | None = None,
    ) -> GroundCoverResult:
        """Detect snow cover with baseline comparison.

        If a baseline brightness is provided, the detection can be
        adjusted based on how much brighter the ground is compared
        to the no-snow baseline.

        Args:
            frame: BGR frame
            baseline_brightness: Optional baseline brightness for comparison

        Returns:
            Detection result
        """
        result = self.detect(frame)

        # If we have a baseline, adjust confidence based on brightness change
        if baseline_brightness is not None and baseline_brightness > 0:
            brightness_ratio = result.avg_brightness / baseline_brightness

            # If ground is significantly brighter than baseline, increase confidence
            if brightness_ratio > 1.2:  # 20% brighter
                # Adjust coverage slightly upward for bright ground
                adjusted_coverage = min(100, result.coverage_percent * 1.1)
                result = GroundCoverResult(
                    has_snow=adjusted_coverage >= self.config.coverage_threshold,
                    coverage_percent=round(adjusted_coverage, 1),
                    ground_pixels=result.ground_pixels,
                    snow_pixels=result.snow_pixels,
                    avg_brightness=result.avg_brightness,
                )

        return result

    def calibrate_baseline(self, frame: np.ndarray) -> float:
        """Capture baseline brightness for a no-snow frame.

        Call this with a frame that has no snow to establish
        a baseline for comparison.

        Args:
            frame: BGR frame with no snow

        Returns:
            Baseline brightness value
        """
        ground_mask = self._get_ground_mask(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ground_region = cv2.bitwise_and(gray, ground_mask)

        ground_pixels = cv2.countNonZero(ground_mask)
        if ground_pixels > 0:
            baseline = float(np.sum(ground_region) / ground_pixels)
        else:
            baseline = 128.0  # Default mid-gray

        self._log.info("Calibrated baseline brightness", baseline=baseline)
        return baseline

    def get_overlay_image(
        self,
        frame: np.ndarray,
        alpha: float = 0.3,
        color: tuple[int, int, int] = (0, 255, 0),  # Green BGR
    ) -> np.ndarray:
        """Create an overlay image showing the detected ground region.

        Args:
            frame: Original BGR frame
            alpha: Overlay transparency (0-1)
            color: BGR color for ground overlay

        Returns:
            Frame with ground region highlighted
        """
        mask = self._get_ground_mask(frame)

        if mask is None:
            return frame

        # Create colored overlay
        overlay = frame.copy()
        overlay[mask > 0] = color

        # Blend with original
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

        # Draw contour outline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)

        # Add label showing if auto-detected or manual
        label = "AUTO" if self._auto_detected else "MANUAL"
        cv2.putText(
            result,
            f"Ground: {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return result
