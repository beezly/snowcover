"""Snow intensity classification using ONNX models."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import structlog

from snowcover.config import IntensityConfig

logger = structlog.get_logger()

# Try to import ONNX runtime, but make it optional
try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None


@dataclass
class IntensityResult:
    """Result of intensity classification."""

    intensity_percent: int  # 0-100
    intensity_category: str  # none, light, moderate, heavy
    confidence: float
    model_available: bool


class IntensityClassifier:
    """Classify snow intensity using a pre-trained ONNX model.

    Falls back to particle-based intensity estimation if no model is available.
    """

    # Standard weather classification classes (for pre-trained models)
    WEATHER_CLASSES = [
        "clear",
        "cloudy",
        "fog",
        "rain",
        "snow",
        "hail",
        "thunderstorm",
    ]

    def __init__(self, config: IntensityConfig):
        """Initialize the intensity classifier.

        Args:
            config: Classification configuration
        """
        self.config = config
        self._log = logger.bind(component="intensity")
        self._session: "ort.InferenceSession | None" = None
        self._input_name: str | None = None
        self._input_shape: tuple | None = None

        # Try to load the model
        if config.enabled:
            self._load_model()

    def _load_model(self) -> None:
        """Load the ONNX model if available."""
        if not ONNX_AVAILABLE:
            self._log.warning(
                "ONNX Runtime not installed - using fallback intensity estimation"
            )
            return

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            self._log.warning(
                "Model file not found - using fallback intensity estimation",
                model_path=str(model_path),
            )
            return

        try:
            # Create inference session with CPU provider
            self._session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )

            # Get input details
            input_info = self._session.get_inputs()[0]
            self._input_name = input_info.name
            self._input_shape = input_info.shape

            self._log.info(
                "Loaded ONNX model",
                model_path=str(model_path),
                input_shape=self._input_shape,
            )
        except Exception as e:
            self._log.error("Failed to load ONNX model", error=str(e))
            self._session = None

    @property
    def model_available(self) -> bool:
        """Check if the ML model is available."""
        return self._session is not None

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input.

        Args:
            frame: BGR frame

        Returns:
            Preprocessed tensor ready for model input
        """
        if self._input_shape is None:
            raise RuntimeError("Model not loaded")

        # Get expected input size (assuming NCHW format)
        _, _, height, width = self._input_shape

        # Resize
        resized = cv2.resize(frame, (width, height))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and convert to float32
        normalized = rgb.astype(np.float32) / 255.0

        # ImageNet normalization (common for pre-trained models)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std

        # Transpose to NCHW format: (H, W, C) -> (1, C, H, W)
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor

    def _classify_with_model(self, frame: np.ndarray) -> IntensityResult:
        """Classify intensity using the ONNX model.

        Args:
            frame: BGR frame

        Returns:
            Classification result
        """
        if self._session is None or self._input_name is None:
            raise RuntimeError("Model not loaded")

        # Preprocess
        input_tensor = self._preprocess_frame(frame)

        # Run inference
        outputs = self._session.run(None, {self._input_name: input_tensor})
        logits = outputs[0][0]

        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Find snow class probability (if available)
        snow_idx = None
        for i, cls in enumerate(self.WEATHER_CLASSES):
            if cls == "snow" and i < len(probs):
                snow_idx = i
                break

        if snow_idx is not None:
            snow_prob = probs[snow_idx]
            confidence = float(snow_prob)

            # Map snow probability to intensity
            intensity_percent = int(min(100, snow_prob * 100))
        else:
            # Fallback: use highest probability class
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])

            # Simple mapping: non-snow = 0%, snow detected = scale by confidence
            if pred_class < len(self.WEATHER_CLASSES):
                if self.WEATHER_CLASSES[pred_class] in ["snow", "hail"]:
                    intensity_percent = int(confidence * 100)
                else:
                    intensity_percent = 0
            else:
                intensity_percent = 0

        # Determine category
        category = self._percent_to_category(intensity_percent)

        return IntensityResult(
            intensity_percent=intensity_percent,
            intensity_category=category,
            confidence=confidence,
            model_available=True,
        )

    def _percent_to_category(self, percent: int) -> str:
        """Convert intensity percentage to category.

        Args:
            percent: Intensity percentage (0-100)

        Returns:
            Category string
        """
        if percent < self.config.light_threshold:
            return "none"
        elif percent < self.config.moderate_threshold:
            return "light"
        elif percent < self.config.heavy_threshold:
            return "moderate"
        else:
            return "heavy"

    def classify(
        self,
        frame: np.ndarray,
        particle_intensity: int | None = None,
    ) -> IntensityResult:
        """Classify snow intensity.

        Args:
            frame: BGR frame
            particle_intensity: Optional fallback intensity from particle analysis

        Returns:
            Classification result
        """
        # Try ML model first
        if self._session is not None:
            try:
                return self._classify_with_model(frame)
            except Exception as e:
                self._log.error("Model inference failed", error=str(e))

        # Fallback to particle-based intensity
        if particle_intensity is not None:
            category = self._percent_to_category(particle_intensity)
            return IntensityResult(
                intensity_percent=particle_intensity,
                intensity_category=category,
                confidence=0.5,  # Lower confidence for fallback
                model_available=False,
            )

        # No data available
        return IntensityResult(
            intensity_percent=0,
            intensity_category="none",
            confidence=0.0,
            model_available=False,
        )

    def classify_from_brightness(self, frame: np.ndarray) -> IntensityResult:
        """Estimate intensity from overall frame brightness.

        This is a simple fallback method that assumes brighter frames
        indicate more snow (due to snow's reflectivity).

        Args:
            frame: BGR frame

        Returns:
            Classification result
        """
        # Convert to grayscale and calculate mean brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        # Also calculate the variance (snowy scenes tend to have lower contrast)
        variance = np.var(gray)

        # Heuristic: high brightness + low variance = likely snow
        # Scale brightness from 0-255 to 0-100
        brightness_factor = min(100, (mean_brightness / 255) * 100)

        # Reduce intensity if variance is high (unlikely to be uniform snow)
        variance_penalty = min(30, variance / 1000)
        intensity = max(0, int(brightness_factor - variance_penalty))

        category = self._percent_to_category(intensity)

        return IntensityResult(
            intensity_percent=intensity,
            intensity_category=category,
            confidence=0.3,  # Low confidence for heuristic method
            model_available=False,
        )
