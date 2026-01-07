#!/usr/bin/env python3
"""Download and convert a weather classification model to ONNX format.

This script downloads a pretrained weather classification model from HuggingFace
and exports it to ONNX format for use with SnowCover's intensity detection.

The model classifies images into 5 weather categories:
- cloudy/overcast
- foggy/hazy
- rain/storm
- snow/frosty
- sun/clear

Usage:
    pip install transformers torch onnx onnxruntime
    python scripts/download_weather_model.py

The model will be saved to models/weather_classifier.onnx
"""

import sys
from pathlib import Path


def main():
    print("SnowCover Weather Classification Model Setup")
    print("=" * 50)

    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    output_path = models_dir / "weather_classifier.onnx"

    if output_path.exists():
        print(f"\nModel already exists at {output_path}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    model_name = "prithivMLmods/Weather-Image-Classification"

    print(f"\n1. Downloading weather classification model...")
    print(f"   Model: {model_name}")
    print("   Classes: cloudy/overcast, foggy/hazy, rain/storm, snow/frosty, sun/clear")
    print("   Accuracy: 85.89%")

    try:
        import torch
        from transformers import AutoImageProcessor, SiglipForImageClassification

        print("\n2. Loading model from HuggingFace...")
        model = SiglipForImageClassification.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)

        model.eval()

        print("\n3. Exporting to ONNX format...")
        # Create dummy input matching the expected image size
        # SigLIP uses 224x224 images by default
        dummy_input = torch.randn(1, 3, 224, 224)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )

        # Get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   Exported to {output_path} ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"   Error during export: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n4. Verifying ONNX model...")
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("   ONNX model is valid!")

        # Print input/output info
        print(f"   Input: {onnx_model.graph.input[0].name}")
        print(f"   Output: {onnx_model.graph.output[0].name}")

        # Get output shape to determine number of classes
        output_shape = [d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]
        print(f"   Output shape: {output_shape}")

    except Exception as e:
        print(f"   Warning during verification: {e}")

    print("\n5. Testing inference...")
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        # Run test inference
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {input_name: test_input})
        logits = outputs[0][0]

        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        print(f"   Test inference successful!")
        print(f"   Output classes: {len(probs)}")
        print(f"   Sample probabilities: {[f'{p:.3f}' for p in probs]}")

    except Exception as e:
        print(f"   Warning during test: {e}")

    print("\n" + "=" * 50)
    print("Setup complete!")
    print(f"\nModel saved to: {output_path}")
    print("\nWeather classes detected by this model:")
    print("  - cloudy/overcast (index 0)")
    print("  - foggy/hazy (index 1)")
    print("  - rain/storm (index 2)")
    print("  - snow/frosty (index 3)")
    print("  - sun/clear (index 4)")
    print("\nRestart SnowCover to use ML-based intensity classification.")


if __name__ == "__main__":
    main()
