#!/usr/bin/env python3
"""Download and convert a semantic segmentation model to ONNX format.

This script downloads a pretrained SegFormer model trained on ADE20K
(150 classes including road, grass, earth, sidewalk, etc.) and exports
it to ONNX format for use with SnowCover's automatic ground detection.

Usage:
    pip install transformers optimum onnx onnxruntime
    python scripts/download_segmentation_model.py

The model will be saved to models/ground_segmentation.onnx
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check and install required dependencies."""
    required = ["transformers", "optimum", "onnx", "onnxruntime"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Installing: pip install {' '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)


def main():
    print("SnowCover Ground Segmentation Model Setup")
    print("=" * 50)

    # Check dependencies
    print("\n1. Checking dependencies...")
    check_dependencies()
    print("   All dependencies installed!")

    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    output_path = models_dir / "ground_segmentation.onnx"
    temp_dir = models_dir / "segformer_temp"

    if output_path.exists():
        print(f"\nModel already exists at {output_path}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    print("\n2. Downloading SegFormer-B0 model trained on ADE20K...")
    print("   Model: nvidia/segformer-b0-finetuned-ade-512-512")
    print("   This model recognizes 150 classes including:")
    print("   - road, sidewalk, earth, grass, path, dirt")
    print("   - buildings, sky, trees, cars, people, etc.")

    # Export using optimum
    from optimum.onnxruntime import ORTModelForSemanticSegmentation

    try:
        # This downloads and converts the model
        model = ORTModelForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            export=True,
        )

        # Save to our models directory
        model.save_pretrained(str(temp_dir))

        # Move the ONNX file to the expected location
        onnx_file = temp_dir / "model.onnx"
        if onnx_file.exists():
            import shutil

            shutil.move(str(onnx_file), str(output_path))
            # Clean up temp directory
            shutil.rmtree(str(temp_dir), ignore_errors=True)
        else:
            print(f"   Error: ONNX file not found at {onnx_file}")
            sys.exit(1)

    except Exception as e:
        print(f"   Error during export: {e}")
        print("\n   Trying alternative method with optimum-cli...")

        # Try using optimum-cli as fallback
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "optimum.exporters.onnx",
                    "--model",
                    "nvidia/segformer-b0-finetuned-ade-512-512",
                    "--task",
                    "semantic-segmentation",
                    str(temp_dir),
                ]
            )

            # Move the ONNX file
            onnx_file = temp_dir / "model.onnx"
            if onnx_file.exists():
                import shutil

                shutil.move(str(onnx_file), str(output_path))
                shutil.rmtree(str(temp_dir), ignore_errors=True)
        except Exception as e2:
            print(f"   Alternative method also failed: {e2}")
            sys.exit(1)

    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n   Exported to {output_path} ({size_mb:.1f} MB)")

    print("\n3. Verifying ONNX model...")
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("   ONNX model is valid!")

        # Print input/output info
        print(f"   Input: {onnx_model.graph.input[0].name}")
        print(f"   Output: {onnx_model.graph.output[0].name}")
    except Exception as e:
        print(f"   Warning during verification: {e}")

    print("\n" + "=" * 50)
    print("Setup complete!")
    print(f"\nModel saved to: {output_path}")
    print("\nGround classes detected by this model include:")
    print("  - floor (3), road (6), grass (9), sidewalk (11)")
    print("  - earth (13), path (29), dirt (46), field (53)")
    print("\nRestart SnowCover to use automatic ground detection.")
    print("The web UI will show 'Auto-detected' for Ground Region.")


if __name__ == "__main__":
    main()
