"""Automatic ground region detection using semantic segmentation."""

import threading
from pathlib import Path

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()

# ADE20K class indices that represent ground-like surfaces
# These are the classes from ADE20K dataset that we consider "ground"
# Note: ADE20K uses 0-indexed classes, but some models use 1-indexed (with 0 as background)
# We handle both cases in the detection code
GROUND_CLASSES = {
    3: "floor",
    6: "road",
    9: "grass",
    11: "sidewalk",
    13: "earth",
    29: "path",
    46: "dirt",
    52: "platform",
    53: "field",
    59: "runway",
    91: "land",
    94: "dirt_track",
    96: "gravel",
}

# Also check offset-by-1 versions (some models use 0 as background)
GROUND_CLASSES_OFFSET = {k + 1: v for k, v in GROUND_CLASSES.items()}

# Combined set of ground class indices for quick lookup
ALL_GROUND_INDICES = set(GROUND_CLASSES.keys()) | set(GROUND_CLASSES_OFFSET.keys())

# ADE20K class names (first 150 from standard ADE20K)
ADE20K_CLASSES = {
    0: "wall", 1: "building", 2: "sky", 3: "floor", 4: "tree", 5: "ceiling",
    6: "road", 7: "bed", 8: "windowpane", 9: "grass", 10: "cabinet",
    11: "sidewalk", 12: "person", 13: "earth", 14: "door", 15: "table",
    16: "mountain", 17: "plant", 18: "curtain", 19: "chair", 20: "car",
    21: "water", 22: "painting", 23: "sofa", 24: "shelf", 25: "house",
    26: "sea", 27: "mirror", 28: "rug", 29: "field", 30: "armchair",
    31: "seat", 32: "fence", 33: "desk", 34: "rock", 35: "wardrobe",
    36: "lamp", 37: "bathtub", 38: "railing", 39: "cushion", 40: "base",
    41: "box", 42: "column", 43: "signboard", 44: "chest", 45: "counter",
    46: "sand", 47: "sink", 48: "skyscraper", 49: "fireplace", 50: "refrigerator",
    51: "grandstand", 52: "path", 53: "stairs", 54: "runway", 55: "case",
    56: "pool_table", 57: "pillow", 58: "screen_door", 59: "stairway", 60: "river",
    61: "bridge", 62: "bookcase", 63: "blind", 64: "coffee_table", 65: "toilet",
    66: "flower", 67: "book", 68: "hill", 69: "bench", 70: "countertop",
    71: "stove", 72: "palm", 73: "kitchen_island", 74: "computer", 75: "swivel_chair",
    76: "boat", 77: "bar", 78: "arcade_machine", 79: "hovel", 80: "bus",
    81: "towel", 82: "light", 83: "truck", 84: "tower", 85: "chandelier",
    86: "awning", 87: "streetlight", 88: "booth", 89: "television", 90: "airplane",
    91: "dirt_track", 92: "apparel", 93: "pole", 94: "land", 95: "bannister",
    96: "escalator", 97: "ottoman", 98: "bottle", 99: "buffet", 100: "poster",
    101: "stage", 102: "van", 103: "ship", 104: "fountain", 105: "conveyor",
    106: "canopy", 107: "washer", 108: "plaything", 109: "pool", 110: "stool",
    111: "barrel", 112: "basket", 113: "waterfall", 114: "tent", 115: "bag",
    116: "minibike", 117: "cradle", 118: "oven", 119: "ball", 120: "food",
    121: "step", 122: "tank", 123: "trade_name", 124: "microwave", 125: "pot",
    126: "animal", 127: "bicycle", 128: "lake", 129: "dishwasher", 130: "screen",
    131: "blanket", 132: "sculpture", 133: "hood", 134: "sconce", 135: "vase",
    136: "traffic_light", 137: "tray", 138: "ashcan", 139: "fan", 140: "pier",
    141: "screen", 142: "plate", 143: "monitor", 144: "bulletin_board", 145: "shower",
    146: "radiator", 147: "glass", 148: "clock", 149: "flag",
}


class GroundSegmenter:
    """Detects ground regions in camera frames using semantic segmentation.

    Uses a lightweight DeepLabV3 model with MobileNetV3 backbone to identify
    ground surfaces like roads, grass, pavement, and dirt.
    """

    MODEL_INPUT_SIZE = (512, 512)

    def __init__(self, model_path: str | Path | None = None):
        """Initialize the ground segmenter.

        Args:
            model_path: Path to ONNX segmentation model. If None, uses default path.
        """
        self._log = logger.bind(component="ground_segmentation")
        self._model = None
        self._session = None
        self._lock = threading.Lock()

        # Cached ground mask
        self._cached_mask: np.ndarray | None = None
        self._cached_frame_shape: tuple | None = None

        # Default model path
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent.parent / "models" / "ground_segmentation.onnx"
        self._model_path = Path(model_path)

        self._load_model()

    def _load_model(self) -> None:
        """Load the ONNX segmentation model."""
        if not self._model_path.exists():
            self._log.warning(
                "Ground segmentation model not found - auto-detection disabled",
                model_path=str(self._model_path),
            )
            return

        try:
            import onnxruntime as ort

            # Use CPU provider for compatibility
            self._session = ort.InferenceSession(
                str(self._model_path),
                providers=['CPUExecutionProvider']
            )

            self._log.info(
                "Ground segmentation model loaded",
                model_path=str(self._model_path),
            )
        except Exception as e:
            self._log.error("Failed to load ground segmentation model", error=str(e))
            self._session = None

    @property
    def is_available(self) -> bool:
        """Check if the segmentation model is loaded and ready."""
        return self._session is not None

    def detect_ground(self, frame: np.ndarray, use_cache: bool = True) -> np.ndarray | None:
        """Detect ground regions in a frame.

        Args:
            frame: BGR image from camera
            use_cache: If True, return cached mask if frame size matches

        Returns:
            Binary mask where 255 = ground, 0 = not ground.
            None if model not available.
        """
        if not self.is_available:
            return None

        # Check cache
        if use_cache and self._cached_mask is not None:
            if self._cached_frame_shape == frame.shape[:2]:
                return self._cached_mask.copy()

        with self._lock:
            mask = self._segment_frame(frame)

            if mask is not None:
                self._cached_mask = mask
                self._cached_frame_shape = frame.shape[:2]

            return mask

    def _segment_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """Run segmentation on a frame.

        Args:
            frame: BGR image

        Returns:
            Binary ground mask
        """
        try:
            original_h, original_w = frame.shape[:2]

            # Preprocess: resize and normalize
            input_image = cv2.resize(frame, self.MODEL_INPUT_SIZE)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1] and then apply ImageNet normalization
            input_image = input_image.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            input_image = (input_image - mean) / std

            # Convert to NCHW format
            input_tensor = np.transpose(input_image, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)

            # Run inference
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: input_tensor})

            # Get class map from model outputs
            class_map = self._process_model_output(outputs)

            # Create ground mask from ground classes
            ground_mask = np.zeros(class_map.shape, dtype=np.uint8)
            for class_idx in ALL_GROUND_INDICES:
                ground_mask[class_map == class_idx] = 255

            # Resize mask back to original frame size
            ground_mask = cv2.resize(
                ground_mask,
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST
            )

            # Clean up mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, kernel)
            ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_OPEN, kernel)

            # Fill small holes and keep only significant regions
            contours, _ = cv2.findContours(ground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Keep only large contours (at least 5% of image area)
                min_area = (original_w * original_h) * 0.05
                large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
                if large_contours:
                    ground_mask = np.zeros_like(ground_mask)
                    cv2.drawContours(ground_mask, large_contours, -1, 255, -1)

            ground_pixels = np.sum(ground_mask > 0)
            total_pixels = ground_mask.size
            coverage = (ground_pixels / total_pixels) * 100

            self._log.info(
                "Ground segmentation complete",
                ground_coverage_pct=round(coverage, 1),
                detected_area_pixels=int(ground_pixels),
            )

            return ground_mask

        except Exception as e:
            self._log.error("Ground segmentation failed", error=str(e))
            import traceback
            self._log.debug("Segmentation traceback", tb=traceback.format_exc())
            return None

    def _process_model_output(self, outputs: list) -> np.ndarray:
        """Process model outputs to get per-pixel class map.

        Handles different model output formats:
        - Per-pixel classification (SegFormer, DeepLab): (1, num_classes, H, W)
        - Mask-based (MaskFormer): class_queries_logits + masks_queries_logits

        Args:
            outputs: Raw model outputs

        Returns:
            2D array of class indices per pixel
        """
        # Check if this is MaskFormer output (two outputs: class queries + mask queries)
        if len(outputs) == 2:
            class_queries = outputs[0]  # (batch, num_queries, num_classes)
            mask_queries = outputs[1]   # (batch, num_queries, H, W)

            # MaskFormer processing
            return self._process_maskformer_output(class_queries, mask_queries)

        # Standard per-pixel classification model
        segmentation = outputs[0]

        if len(segmentation.shape) == 4:
            # Output shape: (1, num_classes, H, W)
            return np.argmax(segmentation[0], axis=0)
        elif len(segmentation.shape) == 3:
            # Output shape: (1, H, W) - already class indices
            return segmentation[0].astype(np.int32)
        else:
            # Output shape: (H, W)
            return segmentation.astype(np.int32)

    def _process_maskformer_output(
        self,
        class_queries: np.ndarray,
        mask_queries: np.ndarray,
    ) -> np.ndarray:
        """Process MaskFormer-style model output.

        Args:
            class_queries: (batch, num_queries, num_classes) class logits per query
            mask_queries: (batch, num_queries, H, W) mask logits per query

        Returns:
            2D array of class indices per pixel
        """
        # Remove batch dimension
        class_logits = class_queries[0]  # (num_queries, num_classes)
        mask_logits = mask_queries[0]    # (num_queries, H, W)

        num_queries, h, w = mask_logits.shape
        num_classes = class_logits.shape[1]

        # Apply softmax to class logits to get class probabilities
        # Subtract max for numerical stability
        class_logits = class_logits - np.max(class_logits, axis=1, keepdims=True)
        class_probs = np.exp(class_logits) / np.sum(np.exp(class_logits), axis=1, keepdims=True)

        # Apply sigmoid to mask logits to get per-pixel mask probabilities
        mask_probs = 1 / (1 + np.exp(-mask_logits))  # (num_queries, H, W)

        # For each pixel, compute weighted sum of class probabilities
        # weighted by the mask probability for each query
        # Result shape: (num_classes, H, W)
        # Limit to first 150 classes (ADE20K semantic classes)
        semantic_classes = min(num_classes, 150)

        # Compute per-pixel class scores
        # For efficiency, we compute: score[c, h, w] = sum_q(mask_probs[q,h,w] * class_probs[q,c])
        pixel_class_scores = np.zeros((semantic_classes, h, w), dtype=np.float32)

        for q in range(num_queries):
            mask_prob = mask_probs[q]  # (H, W)
            for c in range(semantic_classes):
                pixel_class_scores[c] += mask_prob * class_probs[q, c]

        # Get class with highest score per pixel
        class_map = np.argmax(pixel_class_scores, axis=0)

        return class_map.astype(np.int32)

    def get_overlay_image(
        self,
        frame: np.ndarray,
        mask: np.ndarray | None = None,
        alpha: float = 0.3,
        color: tuple[int, int, int] = (0, 255, 0),  # Green
    ) -> np.ndarray:
        """Create an overlay image showing detected ground regions.

        Args:
            frame: Original BGR frame
            mask: Ground mask (if None, will detect)
            alpha: Overlay transparency (0-1)
            color: BGR color for ground overlay

        Returns:
            Frame with ground regions highlighted
        """
        if mask is None:
            mask = self.detect_ground(frame)

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

        return result

    def get_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Create a debug overlay showing all detected classes with labels.

        Args:
            frame: Original BGR frame

        Returns:
            Frame with segmentation classes visualized and labeled
        """
        if not self.is_available:
            return frame

        try:
            original_h, original_w = frame.shape[:2]

            # Preprocess
            input_image = cv2.resize(frame, self.MODEL_INPUT_SIZE)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = input_image.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            input_image = (input_image - mean) / std
            input_tensor = np.transpose(input_image, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)

            # Run inference
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: input_tensor})

            # Get class map using shared processing logic
            class_map = self._process_model_output(outputs)

            # Resize to original
            class_map = cv2.resize(
                class_map.astype(np.float32),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

            # Create color visualization
            result = frame.copy()
            unique_classes = np.unique(class_map)

            # Color map for visualization
            np.random.seed(42)
            colors = {i: tuple(map(int, np.random.randint(0, 255, 3))) for i in range(256)}
            # Make ground classes green-ish
            for class_idx in ALL_GROUND_INDICES:
                colors[class_idx] = (0, 255, 0)

            # Apply colors
            overlay = np.zeros_like(result)
            for class_idx in unique_classes:
                mask = class_map == class_idx
                overlay[mask] = colors.get(int(class_idx), (128, 128, 128))

            result = cv2.addWeighted(result, 0.5, overlay, 0.5, 0)

            # Add labels for detected classes
            class_info = []
            for class_idx in unique_classes:
                class_idx = int(class_idx)
                # Check if it's a ground class
                name = GROUND_CLASSES.get(class_idx) or GROUND_CLASSES_OFFSET.get(class_idx)
                is_ground = name is not None
                if name is None:
                    # Try to get name from ADE20K class list
                    name = ADE20K_CLASSES.get(class_idx, f"class_{class_idx}")
                # Count pixels
                pixel_count = np.sum(class_map == class_idx)
                pct = (pixel_count / class_map.size) * 100
                if pct > 1:  # Only show classes with >1% coverage
                    class_info.append((name, pct, is_ground))

            # Sort by coverage
            class_info.sort(key=lambda x: x[1], reverse=True)

            # Draw info panel
            y_offset = 30
            cv2.rectangle(result, (5, 5), (250, 25 + len(class_info[:10]) * 20), (0, 0, 0), -1)
            cv2.putText(result, "Detected Classes:", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for name, pct, is_ground in class_info[:10]:
                color = (0, 255, 0) if is_ground else (200, 200, 200)
                marker = "[GROUND]" if is_ground else ""
                text = f"{name}: {pct:.1f}% {marker}"
                cv2.putText(result, text, (10, y_offset + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 18

            return result

        except Exception as e:
            self._log.error("Debug overlay failed", error=str(e))
            return frame

    def clear_cache(self) -> None:
        """Clear the cached ground mask."""
        with self._lock:
            self._cached_mask = None
            self._cached_frame_shape = None
