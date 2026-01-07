"""Snow detection orchestrator combining all detection modules."""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from snowcover.config import DetectionConfig, SmoothingConfig
from snowcover.detection.falling_snow import FallingSnowDetector, FallingSnowResult
from snowcover.detection.ground_cover import GroundCoverDetector, GroundCoverResult
from snowcover.detection.intensity import IntensityClassifier, IntensityResult

logger = structlog.get_logger()


@dataclass
class SnowDetectionResult:
    """Combined result from all detection modules."""

    # Falling snow
    is_snowing: bool
    particle_count: int
    snow_confidence: float

    # Intensity
    intensity_percent: int
    intensity_category: str

    # Ground cover
    has_ground_snow: bool
    ground_coverage_percent: float
    ground_confidence: float
    ground_brightness: float

    # Metadata
    timestamp: float
    confidence: float  # Overall confidence (max of snow and ground)
    model_available: bool

    # Raw results for debugging
    raw_falling_snow: FallingSnowResult | None = None
    raw_intensity: IntensityResult | None = None
    raw_ground_cover: GroundCoverResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_snowing": self.is_snowing,
            "particle_count": self.particle_count,
            "snow_confidence": self.snow_confidence,
            "intensity_percent": self.intensity_percent,
            "intensity_category": self.intensity_category,
            "has_ground_snow": self.has_ground_snow,
            "ground_coverage_percent": self.ground_coverage_percent,
            "ground_confidence": self.ground_confidence,
            "ground_brightness": self.ground_brightness,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "model_available": self.model_available,
        }


@dataclass
class SmoothedState:
    """Smoothed state with debouncing."""

    is_snowing: bool = False
    intensity_percent: int = 0
    intensity_category: str = "none"
    has_ground_snow: bool = False
    ground_coverage_percent: float = 0.0

    last_snowing_change: float = 0.0
    last_ground_snow_change: float = 0.0


class SnowDetector:
    """Main snow detection orchestrator.

    Combines falling snow detection, intensity classification,
    and ground cover detection into a unified interface with
    smoothing and debouncing.
    """

    def __init__(
        self,
        detection_config: DetectionConfig,
        smoothing_config: SmoothingConfig,
    ):
        """Initialize the snow detector.

        Args:
            detection_config: Detection module configurations
            smoothing_config: Smoothing and debouncing configuration
        """
        self.detection_config = detection_config
        self.smoothing_config = smoothing_config
        self._log = logger.bind(component="detector")

        # Initialize detection modules
        self._falling_snow: FallingSnowDetector | None = None
        self._intensity: IntensityClassifier | None = None
        self._ground_cover: GroundCoverDetector | None = None

        if detection_config.falling_snow.enabled:
            self._falling_snow = FallingSnowDetector(detection_config.falling_snow)

        if detection_config.intensity.enabled:
            self._intensity = IntensityClassifier(detection_config.intensity)

        if detection_config.ground_cover.enabled:
            self._ground_cover = GroundCoverDetector(detection_config.ground_cover)

        # Smoothing state
        self._history: deque[SnowDetectionResult] = deque(
            maxlen=smoothing_config.window_size
        )
        self._smoothed_state = SmoothedState()

    @property
    def ground_cover_detector(self) -> GroundCoverDetector | None:
        """Get the ground cover detector instance (for overlay rendering)."""
        return self._ground_cover

    def detect(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray | None = None,
    ) -> SnowDetectionResult:
        """Run all detection modules on the given frame(s).

        Args:
            current_frame: Current frame (BGR)
            previous_frame: Previous frame for temporal analysis (optional)

        Returns:
            Combined detection result
        """
        timestamp = time.time()

        # Initialize results
        falling_snow_result: FallingSnowResult | None = None
        intensity_result: IntensityResult | None = None
        ground_cover_result: GroundCoverResult | None = None

        is_snowing = False
        particle_count = 0
        snow_confidence = 0.0
        intensity_percent = 0
        intensity_category = "none"
        has_ground_snow = False
        ground_coverage_percent = 0.0
        ground_confidence = 0.0
        ground_brightness = 0.0
        confidence = 0.0
        model_available = False

        # Run falling snow detection (requires two frames)
        if self._falling_snow is not None and previous_frame is not None:
            try:
                falling_snow_result, particle_intensity = (
                    self._falling_snow.detect_with_intensity(current_frame, previous_frame)
                )
                is_snowing = falling_snow_result.is_snowing
                particle_count = falling_snow_result.particle_count
                snow_confidence = falling_snow_result.confidence
                confidence = max(confidence, falling_snow_result.confidence)

                # Use particle intensity as fallback
                if self._intensity is None:
                    intensity_percent = particle_intensity
                    intensity_category = self._intensity_to_category(particle_intensity)
            except Exception as e:
                self._log.error("Falling snow detection failed", error=str(e))

        # Run intensity classification
        if self._intensity is not None:
            try:
                # Pass particle intensity as fallback
                fallback_intensity = intensity_percent if falling_snow_result else None
                intensity_result = self._intensity.classify(
                    current_frame,
                    particle_intensity=fallback_intensity,
                )
                intensity_percent = intensity_result.intensity_percent
                intensity_category = intensity_result.intensity_category
                model_available = intensity_result.model_available
                confidence = max(confidence, intensity_result.confidence)

                # If intensity model says it's snowing but falling snow didn't detect it,
                # trust the model for is_snowing if confidence is high
                if intensity_percent > 0 and intensity_result.confidence > 0.7:
                    if not is_snowing and intensity_percent >= 25:
                        is_snowing = True
            except Exception as e:
                self._log.error("Intensity classification failed", error=str(e))

        # Run ground cover detection
        if self._ground_cover is not None:
            try:
                ground_cover_result = self._ground_cover.detect(current_frame)
                has_ground_snow = ground_cover_result.has_snow
                ground_coverage_percent = ground_cover_result.coverage_percent
                ground_confidence = ground_cover_result.confidence
                ground_brightness = ground_cover_result.avg_brightness
                confidence = max(confidence, ground_cover_result.confidence)
            except Exception as e:
                self._log.error("Ground cover detection failed", error=str(e))

        result = SnowDetectionResult(
            is_snowing=is_snowing,
            particle_count=particle_count,
            snow_confidence=snow_confidence,
            intensity_percent=intensity_percent,
            intensity_category=intensity_category,
            has_ground_snow=has_ground_snow,
            ground_coverage_percent=ground_coverage_percent,
            ground_confidence=ground_confidence,
            ground_brightness=ground_brightness,
            timestamp=timestamp,
            confidence=confidence,
            model_available=model_available,
            raw_falling_snow=falling_snow_result,
            raw_intensity=intensity_result,
            raw_ground_cover=ground_cover_result,
        )

        # Add to history for smoothing
        self._history.append(result)

        return result

    def _intensity_to_category(self, percent: int) -> str:
        """Convert intensity percentage to category."""
        cfg = self.detection_config.intensity
        if percent < cfg.light_threshold:
            return "none"
        elif percent < cfg.moderate_threshold:
            return "light"
        elif percent < cfg.heavy_threshold:
            return "moderate"
        else:
            return "heavy"

    def get_smoothed_result(self) -> SnowDetectionResult | None:
        """Get a smoothed result based on recent detections.

        Returns:
            Smoothed detection result, or None if no data available
        """
        if not self._history:
            return None

        # Average numeric values
        avg_intensity = int(
            sum(r.intensity_percent for r in self._history) / len(self._history)
        )
        avg_ground_coverage = sum(r.ground_coverage_percent for r in self._history) / len(
            self._history
        )
        avg_particle_count = int(
            sum(r.particle_count for r in self._history) / len(self._history)
        )
        avg_confidence = sum(r.confidence for r in self._history) / len(self._history)
        avg_snow_confidence = sum(r.snow_confidence for r in self._history) / len(self._history)
        avg_ground_confidence = sum(r.ground_confidence for r in self._history) / len(
            self._history
        )
        avg_ground_brightness = sum(r.ground_brightness for r in self._history) / len(
            self._history
        )

        # Majority vote for boolean values
        snowing_votes = sum(1 for r in self._history if r.is_snowing)
        ground_snow_votes = sum(1 for r in self._history if r.has_ground_snow)
        is_snowing = snowing_votes > len(self._history) / 2
        has_ground_snow = ground_snow_votes > len(self._history) / 2

        # Derive intensity category from averaged percent
        intensity_category = self._intensity_to_category(avg_intensity)

        # Use most recent for metadata
        latest = self._history[-1]

        return SnowDetectionResult(
            is_snowing=is_snowing,
            particle_count=avg_particle_count,
            snow_confidence=round(avg_snow_confidence, 3),
            intensity_percent=avg_intensity,
            intensity_category=intensity_category,
            has_ground_snow=has_ground_snow,
            ground_coverage_percent=round(avg_ground_coverage, 1),
            ground_confidence=round(avg_ground_confidence, 3),
            ground_brightness=round(avg_ground_brightness, 1),
            timestamp=latest.timestamp,
            confidence=round(avg_confidence, 3),
            model_available=latest.model_available,
        )

    def get_state_changes(
        self,
        current_result: SnowDetectionResult,
    ) -> dict[str, Any]:
        """Determine which states have changed enough to publish.

        Applies debouncing and change thresholds.

        Args:
            current_result: Current detection result

        Returns:
            Dictionary of changed states to publish
        """
        now = time.time()
        changes: dict[str, Any] = {}
        cfg = self.smoothing_config

        # Check is_snowing change with debouncing
        if current_result.is_snowing != self._smoothed_state.is_snowing:
            if now - self._smoothed_state.last_snowing_change >= cfg.debounce_seconds:
                changes["is_snowing"] = current_result.is_snowing
                self._smoothed_state.is_snowing = current_result.is_snowing
                self._smoothed_state.last_snowing_change = now

        # Check intensity change (threshold-based)
        intensity_diff = abs(
            current_result.intensity_percent - self._smoothed_state.intensity_percent
        )
        if intensity_diff >= cfg.change_threshold:
            changes["intensity_percent"] = current_result.intensity_percent
            changes["intensity_category"] = current_result.intensity_category
            self._smoothed_state.intensity_percent = current_result.intensity_percent
            self._smoothed_state.intensity_category = current_result.intensity_category

        # Check ground snow change with debouncing
        if current_result.has_ground_snow != self._smoothed_state.has_ground_snow:
            if now - self._smoothed_state.last_ground_snow_change >= cfg.debounce_seconds:
                changes["has_ground_snow"] = current_result.has_ground_snow
                self._smoothed_state.has_ground_snow = current_result.has_ground_snow
                self._smoothed_state.last_ground_snow_change = now

        # Check ground coverage change (threshold-based)
        coverage_diff = abs(
            current_result.ground_coverage_percent
            - self._smoothed_state.ground_coverage_percent
        )
        if coverage_diff >= cfg.change_threshold:
            changes["ground_coverage_percent"] = current_result.ground_coverage_percent
            self._smoothed_state.ground_coverage_percent = (
                current_result.ground_coverage_percent
            )

        return changes

    def reset(self) -> None:
        """Reset detection history and smoothed state."""
        self._history.clear()
        self._smoothed_state = SmoothedState()
