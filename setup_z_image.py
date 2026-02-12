#!/usr/bin/env python
"""Z-Image Local Setup and Test Script.

This script helps you set up and test the Z-Image local image generation provider.

Usage:
    python setup_z_image.py check     # Check system requirements
    python setup_z_image.py install   # Install required packages
    python setup_z_image.py download  # Download the Z-Image model
    python setup_z_image.py test      # Generate a test image
    python setup_z_image.py all       # Do everything

Requirements:
    - NVIDIA GPU with CUDA support (8GB+ VRAM recommended, 16GB+ ideal)
    - Python 3.10+
    - ~10GB disk space for the model
"""

import sys
import os
import subprocess
import argparse


def check_python_version():
    """Check Python version is 3.10+."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(f"❌ Python 3.10+ required, you have {major}.{minor}")
        return False
    print(f"✓ Python {major}.{minor}")
    return True


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ CUDA available: {device_name} ({vram:.1f}GB VRAM)")

            if vram < 8:
                print("  ⚠ Warning: Less than 8GB VRAM. Enable Z_IMAGE_OFFLOAD=1")
            elif vram < 16:
                print("  ℹ Tip: 8-16GB VRAM. Consider Z_IMAGE_OFFLOAD=1 for stability")

            return True
        else:
            print("❌ CUDA not available. Z-Image requires an NVIDIA GPU.")
            print("   You can still test on CPU (very slow) with Z_IMAGE_DEVICE=cpu")
            return False
    except ImportError:
        print("❌ PyTorch not installed. Run: pip install torch torchvision")
        return False


def check_diffusers():
    """Check diffusers installation."""
    try:
        # ZImagePipeline is new in diffusers, type stubs may not have it
        from diffusers import ZImagePipeline as _Pipeline  # type: ignore[attr-defined]

        _ = _Pipeline  # Verify import succeeded
        print("✓ diffusers installed with ZImagePipeline support")
        return True
    except ImportError as e:
        if "ZImagePipeline" in str(e):
            print("❌ diffusers installed but missing ZImagePipeline")
            print("   Run: pip install git+https://github.com/huggingface/diffusers")
        else:
            print("❌ diffusers not installed")
            print("   Run: pip install git+https://github.com/huggingface/diffusers")
        return False


def check_other_deps():
    """Check other required dependencies."""
    deps = ["transformers", "accelerate", "sentencepiece"]
    missing = []

    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"❌ {dep} not installed")
            missing.append(dep)

    if missing:
        print(f"   Run: pip install {' '.join(missing)}")
        return False
    return True


def check_requirements():
    """Run all requirement checks."""
    print("\n🔍 Checking Z-Image Requirements\n" + "=" * 40)

    checks = [
        ("Python version", check_python_version),
        ("CUDA/GPU", check_cuda),
        ("diffusers", check_diffusers),
        ("Other dependencies", check_other_deps),
    ]

    all_passed = True
    for name, check_fn in checks:
        print(f"\n{name}:")
        if not check_fn():
            all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("✅ All requirements met! You can use Z-Image.")
    else:
        print("❌ Some requirements missing. See above for fixes.")

    return all_passed


def install_packages():
    """Install required packages."""
    print("\n📦 Installing Z-Image Dependencies\n" + "=" * 40)

    packages = [
        # Install diffusers from git for latest ZImagePipeline
        ("diffusers (from git)", "git+https://github.com/huggingface/diffusers"),
        # Core dependencies
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("sentencepiece", "sentencepiece"),
        # Optional but recommended
        ("xformers", "xformers"),  # Memory-efficient attention
    ]

    for name, package in packages:
        print(f"\nInstalling {name}...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-U", package],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"  ✓ {name} installed")
            else:
                print(f"  ⚠ {name} installation had warnings")
                if "xformers" in package:
                    print("    (xformers is optional, continuing...)")
        except Exception as e:
            print(f"  ❌ Failed to install {name}: {e}")

    print("\n✅ Package installation complete!")


def download_model():
    """Download the Z-Image model."""
    print("\n📥 Downloading Z-Image Model\n" + "=" * 40)
    print("This will download ~10GB and may take a while...\n")

    try:
        # Use huggingface_hub for efficient download
        from huggingface_hub import snapshot_download

        # Set for high-performance download if available
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

        cache_dir = os.getenv("Z_IMAGE_CACHE_DIR", None)

        print("Downloading Tongyi-MAI/Z-Image...")
        path = snapshot_download(
            "Tongyi-MAI/Z-Image",
            cache_dir=cache_dir,
        )
        print(f"\n✅ Model downloaded to: {path}")

    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

    return True


def test_generation():
    """Generate a test image."""
    print("\n🎨 Testing Z-Image Generation\n" + "=" * 40)

    # Set environment for test
    os.environ.setdefault("IMAGE_PROVIDER", "z_image")

    try:
        from image_providers import get_image_provider

        print("Loading Z-Image model (first load takes a few minutes)...")
        provider = get_image_provider("z_image")

        if not provider.is_configured:
            print("❌ Z-Image not configured properly. Run 'check' command.")
            return False

        prompt = (
            "A professional business woman in a modern office, "
            "natural lighting, photorealistic, high quality"
        )

        print("\nGenerating test image...")
        print(f"Prompt: {prompt}\n")

        image_bytes = provider.generate(prompt)

        # Save the test image
        output_dir = "generated_images"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "z_image_test.png")

        with open(output_path, "wb") as f:
            f.write(image_bytes)

        print(f"\n✅ Test image saved to: {output_path}")
        print(f"   Image size: {len(image_bytes) // 1024}KB")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def show_usage():
    """Show usage instructions."""
    print("""
🎨 Z-Image Local Provider Setup Complete!

To use Z-Image for image generation, set in your .env file:
    IMAGE_PROVIDER=z_image

Optional configuration:
    Z_IMAGE_STEPS=28          # Inference steps (28-50, higher=better but slower)
    Z_IMAGE_GUIDANCE=4.0      # CFG scale (3.0-5.0)
    Z_IMAGE_DEVICE=cuda       # Device ('cuda' or 'cpu')
    Z_IMAGE_OFFLOAD=0         # Set to '1' for low VRAM (8-12GB)
    Z_IMAGE_CACHE_DIR=        # Custom model cache directory
    Z_IMAGE_NEGATIVE_PROMPT=  # Default negative prompt

Example usage in code:
    from image_providers import get_image_provider

    provider = get_image_provider("z_image")
    image_bytes = provider.generate("A sunset over mountains, photorealistic")

    with open("output.png", "wb") as f:
        f.write(image_bytes)

Tips:
    - First generation loads the model (~2-3 minutes)
    - Subsequent generations are much faster (~10-30 seconds)
    - Use Z_IMAGE_OFFLOAD=1 if you have less than 16GB VRAM
    - Negative prompts help avoid artifacts and unwanted content
""")


def main():
    parser = argparse.ArgumentParser(
        description="Z-Image Local Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        choices=["check", "install", "download", "test", "all"],
        help="Command to run",
    )

    args = parser.parse_args()

    if args.command == "check":
        check_requirements()

    elif args.command == "install":
        install_packages()
        print("\nRun 'python setup_z_image.py check' to verify installation.")

    elif args.command == "download":
        if check_diffusers():
            download_model()
        else:
            print(
                "\n❌ Please install diffusers first: python setup_z_image.py install"
            )

    elif args.command == "test":
        if check_requirements():
            test_generation()
            show_usage()

    elif args.command == "all":
        print("🚀 Full Z-Image Setup\n")
        install_packages()
        if check_requirements():
            download_model()
            test_generation()
            show_usage()


if __name__ == "__main__":
    main()
