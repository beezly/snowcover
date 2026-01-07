# IR Ground Detection Training Plan

## Overview

Train a ground segmentation model that works well with IR/night vision images by using paired RGB and IR data from the same camera position.

## The Problem

The current MaskFormer model was trained on daytime RGB images (ADE20K dataset) and produces poor results on monochrome IR night vision footage. Surfaces that look distinct in color (grass vs driveway vs pavement) all become similar shades of gray under IR illumination.

## The Solution

Use domain adaptation: train a model to recognize ground regions in IR images by learning the correspondence between RGB and IR views of the same scene.

---

## Phase 1: Data Collection

### Equipment Setup
- Camera with controllable IR-cut filter (most security cameras have this)
- Ability to toggle between day mode (RGB) and night mode (IR) via API or settings

### Capture Process

1. **Choose good conditions**: Daylight, clear weather, no snow on ground
2. **Capture paired frames**:
   - Take RGB frame (IR filter ON)
   - Immediately toggle to IR mode (IR filter OFF)
   - Take IR frame
   - Repeat for 50-100 pairs
3. **Vary conditions**:
   - Different times of day (morning, noon, afternoon)
   - Different weather (sunny, cloudy, overcast)
   - Different seasons if possible

### Data Organization

```
data/
  paired/
    001_rgb.jpg
    001_ir.jpg
    002_rgb.jpg
    002_ir.jpg
    ...
```

### Annotation

Option A: **Use RGB model to auto-label**
- Run MaskFormer on RGB images to get ground masks
- These become training labels for IR images
- Quick but inherits any RGB model errors

Option B: **Manual annotation**
- Use labelme or similar to draw ground polygons
- More accurate but time-consuming
- Only need to annotate RGB (use same mask for paired IR)

---

## Phase 2: Model Training

### Approach 1: Fine-tune Existing Model

Fine-tune the MaskFormer/SegFormer on IR images using RGB-derived labels.

```python
# Pseudo-code
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

# Convert IR images to 3-channel (duplicate grayscale)
# Train with ground masks from RGB model
# Use low learning rate to preserve pre-trained features
```

### Approach 2: Domain Adaptation Network

Train a model that explicitly learns RGB→IR mapping.

References:
- DANNet: https://github.com/W-zx-Y/DANNet
- MGCDA: https://github.com/sakaridis/MGCDA

### Approach 3: Simple Transfer Learning

1. Take pre-trained encoder (ResNet, MobileNet)
2. Train decoder on IR images with RGB-derived labels
3. Smaller dataset required

### Training Tips

- Start with frozen encoder, train decoder only
- Use data augmentation: brightness, contrast, noise
- Validate on held-out IR images
- Target metrics: IoU for ground class > 0.7

---

## Phase 3: Export and Integration

### Export to ONNX

```python
import torch

model.eval()
dummy_input = torch.randn(1, 3, 512, 512)

torch.onnx.export(
    model,
    dummy_input,
    "ground_segmentation_ir.onnx",
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch"}},
    opset_version=14,
)
```

### Integration Options

**Option A: Replace existing model**
- Use IR-trained model for both day and night
- Simpler, single model

**Option B: Dual model approach**
- Use RGB model during day, IR model at night
- Better accuracy, more complexity
- Update `ground_segmentation.py` to load both models

```python
class GroundSegmenter:
    def __init__(self):
        self._rgb_session = load_model("ground_segmentation_rgb.onnx")
        self._ir_session = load_model("ground_segmentation_ir.onnx")

    def detect_ground(self, frame, is_ir_mode=False):
        session = self._ir_session if is_ir_mode else self._rgb_session
        # ... run inference
```

---

## Estimated Effort

| Phase | Time | Notes |
|-------|------|-------|
| Data collection | 1-2 hours | Capturing paired frames |
| Annotation | 0-2 hours | Zero if using auto-labels |
| Training setup | 2-4 hours | Environment, dependencies |
| Training | 1-4 hours | GPU time, depends on approach |
| Export & integration | 1 hour | ONNX export, code updates |
| **Total** | **5-13 hours** | |

---

## Quick Start Commands

```bash
# Create virtual environment for training
python -m venv train-env
source train-env/bin/activate

# Install dependencies
pip install torch transformers datasets albumentations

# Capture script (implement based on camera API)
python scripts/capture_paired_frames.py --count 100

# Auto-label IR images using RGB model
python scripts/auto_label_ir.py --input data/paired --output data/labels

# Fine-tune model
python scripts/train_ir_model.py --data data/paired --labels data/labels

# Export to ONNX
python scripts/export_onnx.py --checkpoint best_model.pt --output models/ground_segmentation_ir.onnx
```

---

## Resources

- Dark Zurich dataset (night driving): https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/
- DANNet paper: https://arxiv.org/abs/2104.10834
- Segmentation Models PyTorch: https://github.com/qubvel-org/segmentation_models.pytorch
- HuggingFace Transformers ONNX export: https://huggingface.co/docs/transformers/serialization
