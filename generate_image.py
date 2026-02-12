#!/usr/bin/env python
"""Standalone Z-Image generation script.

Generate images from text prompts using the local Z-Image model.

Usage:
    python generate_image.py                          # Interactive mode
    python generate_image.py "your prompt here"       # Direct prompt
    python generate_image.py -p "prompt" -n "negative prompt"
    python generate_image.py -p "prompt" -o output.png -s 512x512
    python generate_image.py --fast "quick test prompt"  # Fast mode for low VRAM

Examples:
    python generate_image.py "A sunset over mountains, photorealistic"
    python generate_image.py -p "Professional headshot" -n "blurry, low quality"
    python generate_image.py -p "Anime girl" -s 768x1024 --steps 35
    python generate_image.py --fast "A cat" -s 512x512   # ~3-4x faster
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class HardwareSettings:
    """Hardware-optimized settings for image generation."""

    dtype: str = "bfloat16"
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    max_size: int = 1024
    recommended_steps: int = 28
    recommended_guidance: float = 4.0
    vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    compute_capability: tuple[int, int] = (0, 0)
    is_ampere_or_newer: bool = False
    enable_tf32: bool = False  # TF32 matmul on Ampere+ (free ~10-15% speedup)


# Cache hardware settings to avoid repeated detection
_hardware_settings_cache: HardwareSettings | None = None


def get_optimal_settings(
    force_refresh: bool = False, args: argparse.Namespace | None = None
) -> HardwareSettings:
    """Detect hardware and return optimal generation settings.

    Args:
        force_refresh: Force re-detection of hardware (default: use cache)
        args: Optional command-line arguments to override settings.

    Returns:
        Hardware settings optimized for the detected GPU
    """
    global _hardware_settings_cache

    # Return cached settings unless refresh is forced
    if _hardware_settings_cache is not None and not force_refresh:
        # If args are provided, apply overrides to the cached settings
        if args:
            return _apply_cli_overrides(_hardware_settings_cache, args)
        return _hardware_settings_cache

    settings = HardwareSettings()

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_vram = props.total_memory / (1024**3)

            # Get compute capability for architecture-specific optimizations
            settings.compute_capability = torch.cuda.get_device_capability(0)
            settings.is_ampere_or_newer = settings.compute_capability[0] >= 8

            # Check free VRAM
            free_vram = (props.total_memory - torch.cuda.memory_allocated(0)) / (
                1024**3
            )

            settings.vram_gb = total_vram
            settings.free_vram_gb = free_vram

            # Optimize based on TOTAL VRAM (not free) for consistent behavior
            if total_vram <= 3:
                settings.enable_sequential_cpu_offload = True
                settings.enable_vae_tiling = True
                settings.max_size = 512
                settings.recommended_steps = 12
                settings.recommended_guidance = 3.0
            elif total_vram <= 4.5:
                # 4-4.5GB VRAM (e.g. RTX 3050 Ti): sequential offload is essential
                # (transformer=11.5GB, text_encoder=7.5GB — neither fits in VRAM)
                settings.enable_sequential_cpu_offload = True
                # VAE is only 0.16GB — no need for tiling/slicing
                settings.enable_vae_tiling = False
                settings.enable_vae_slicing = False
                # Attention slicing reduces peak VRAM per layer, which avoids
                # memory thrashing under sequential offload (tested 37% faster)
                settings.enable_attention_slicing = True
                settings.max_size = 768
                settings.recommended_steps = 28
                settings.recommended_guidance = 4.0
            elif total_vram <= 6:
                settings.enable_model_cpu_offload = True
                settings.max_size = 768
                settings.recommended_steps = 20
                settings.recommended_guidance = 4.0
            elif total_vram <= 8:
                settings.max_size = 896
                settings.recommended_steps = 25
            else:
                settings.enable_attention_slicing = False
                settings.enable_vae_slicing = False
                settings.max_size = 1280
                settings.recommended_steps = 28

            # Ampere+ (compute 8.0+) optimizations
            if settings.is_ampere_or_newer:
                settings.enable_tf32 = True
                # TF32 gives ~10-15% speedup on matmul/conv with negligible precision loss
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # Enable cuDNN benchmark for consistent input sizes (speeds up conv operations)
            torch.backends.cudnn.benchmark = True

    except ImportError:
        pass

    # Apply overrides from command line if provided
    if args:
        settings = _apply_cli_overrides(settings, args)

    # Cache the settings
    _hardware_settings_cache = settings
    return settings


def _apply_cli_overrides(
    settings: HardwareSettings, args: argparse.Namespace
) -> HardwareSettings:
    """Apply command-line overrides to hardware settings."""
    if args.dtype:
        settings.dtype = args.dtype
    if args.cpu_offload:
        if args.cpu_offload == "sequential":
            settings.enable_sequential_cpu_offload = True
            settings.enable_model_cpu_offload = False
        elif args.cpu_offload == "model":
            settings.enable_model_cpu_offload = True
            settings.enable_sequential_cpu_offload = False
        else:
            settings.enable_sequential_cpu_offload = False
            settings.enable_model_cpu_offload = False
    if args.no_attention_slicing:
        settings.enable_attention_slicing = False
    if args.no_vae_slicing:
        settings.enable_vae_slicing = False
    if args.no_tf32:
        settings.enable_tf32 = False
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    return settings



def clear_memory(aggressive: bool = False) -> None:
    """Clear GPU and system memory.

    Args:
        aggressive: If True, also runs garbage collection (slower but more thorough)
    """
    if aggressive:
        gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()
    except ImportError:
        pass


def check_requirements() -> None:
    """Check if Z-Image requirements are met."""
    try:
        import torch

        if not torch.cuda.is_available():
            print("⚠ Warning: CUDA not available. Generation will be very slow on CPU.")
            response = input("Continue anyway? [y/N]: ").strip().lower()
            if response != "y":
                sys.exit(0)
        else:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / (1024**3)

            torch.cuda.empty_cache()
            free_vram = (props.total_memory - torch.cuda.memory_allocated(0)) / (
                1024**3
            )

            print(f"✓ GPU: {gpu_name}")
            print(f"  VRAM: {vram:.1f}GB total, {free_vram:.1f}GB free")

            settings = get_optimal_settings()
            if (
                settings.enable_model_cpu_offload
                or settings.enable_sequential_cpu_offload
            ):
                print("  💡 Auto-optimizations enabled for low VRAM")
            if vram < 6:
                print("  💡 Tip: Use --fast mode for faster generation")
                print(
                    f"  💡 Recommended max size: {settings.max_size}x{settings.max_size}"
                )

    except ImportError as e:
        print(f"❌ PyTorch not installed: {e}")
        print("   Run: pip install torch torchvision")
        sys.exit(1)

    try:
        # Check if diffusers is available (FluxPipeline or any pipeline will do)
        import diffusers

        _ = diffusers  # Verify import succeeded
    except ImportError as e:
        print(f"❌ diffusers not installed: {e}")
        print("   Run: pip install diffusers")
        sys.exit(1)


def optimize_size_for_hardware(
    width: int, height: int, settings: HardwareSettings
) -> tuple[int, int]:
    """Adjust image size to fit available VRAM."""
    max_size = settings.max_size

    if width > max_size or height > max_size:
        scale = max_size / max(width, height)
        width = int(width * scale)
        height = int(height * scale)

    width = (width // 64) * 64
    height = (height // 64) * 64

    width = max(width, 256)
    height = max(height, 256)

    return width, height


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    output_path: str | None = None,
    size: str = "1024x1024",
    steps: int = 28,
    guidance: float = 4.0,
    seed: int | None = None,
    fast_mode: bool = False,
    args: argparse.Namespace | None = None,
) -> str:
    """Generate an image using Z-Image.

    The provider handles all GPU optimizations (offloading, attention slicing,
    VAE tiling, etc.) internally — this function only adjusts high-level
    parameters (steps, guidance, size) based on detected hardware.
    """
    hw_settings = get_optimal_settings(args=args)

    try:
        width, height = map(int, size.lower().split("x"))
    except ValueError:
        print(f"Invalid size format: {size}. Using 1024x1024")
        width, height = 1024, 1024

    orig_width, orig_height = width, height
    width, height = optimize_size_for_hardware(width, height, hw_settings)

    if (width, height) != (orig_width, orig_height):
        print(
            f"📐 Size adjusted: {orig_width}x{orig_height} → {width}x{height} (VRAM optimization)"
        )

    if fast_mode:
        steps = min(steps, hw_settings.recommended_steps)
        guidance = min(guidance, hw_settings.recommended_guidance)
        print(f"⚡ Fast mode: {steps} steps, guidance {guidance}")
    elif hw_settings.vram_gb <= 6 and hw_settings.vram_gb > 0:
        if steps > hw_settings.recommended_steps:
            steps = hw_settings.recommended_steps
            print(f"⚡ Auto-optimized: {steps} steps (low VRAM detected)")

    from image_providers import get_image_provider
    from image_providers.z_image import ZImageProvider

    provider = get_image_provider(provider_name="z_image", size=f"{width}x{height}")

    # Configure provider — pipeline optimizations are handled inside the provider
    if isinstance(provider, ZImageProvider):
        provider.num_inference_steps = steps
        provider.guidance_scale = guidance
        if negative_prompt:
            provider.default_negative_prompt = negative_prompt

    print("\n🎨 Generating image...")
    print(f"   Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    if negative_prompt:
        print(
            f"   Negative: {negative_prompt[:50]}{'...' if len(negative_prompt) > 50 else ''}"
        )
    print(f"   Size: {width}x{height}")
    print(f"   Steps: {steps}, Guidance: {guidance}")
    if hw_settings.vram_gb > 0:
        print(
            f"   VRAM: {hw_settings.vram_gb:.1f}GB ({hw_settings.free_vram_gb:.1f}GB free)"
        )
    print()

    start_time = time.time()
    image_bytes: bytes

    try:
        if isinstance(provider, ZImageProvider):
            # Single code path — seed is passed to provider.generate()
            image_bytes = provider.generate(prompt, negative_prompt, seed=seed)
        else:
            image_bytes = provider.generate(prompt)
    finally:
        # Light cache clear after generation (no GC — it's slow and unnecessary here)
        clear_memory(aggressive=False)

    elapsed = time.time() - start_time

    final_output_path: str
    if not output_path:
        output_dir = Path("generated_images")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(
            c if c.isalnum() or c in " -_" else "" for c in prompt[:30]
        )
        safe_prompt = safe_prompt.strip().replace(" ", "_")
        final_output_path = str(output_dir / f"z_image_{timestamp}_{safe_prompt}.png")
    else:
        final_output_path = output_path

    with open(final_output_path, "wb") as f:
        f.write(image_bytes)

    print(f"\n✅ Image saved to: {final_output_path}")
    print(f"   Size: {len(image_bytes) // 1024}KB")
    print(f"   Time: {elapsed:.1f}s ({elapsed / steps:.2f}s/step)")

    return final_output_path


def interactive_mode() -> None:
    """Run interactive prompt mode."""
    print("\n" + "=" * 50)
    print("  Z-Image Interactive Generator")
    print("=" * 50)

    hw_settings = get_optimal_settings()
    if hw_settings.vram_gb > 0:
        print(
            f"\n  Hardware: {hw_settings.vram_gb:.1f}GB VRAM ({hw_settings.free_vram_gb:.1f}GB free)"
        )
        print(
            f"  Recommended: {hw_settings.max_size}x{hw_settings.max_size}, {hw_settings.recommended_steps} steps"
        )

    print("\nEnter prompts to generate images. Type 'quit' to exit.")
    print("Type 'help' for advanced options.\n")

    negative_prompt = (
        "text, watermark, logo, blurry, low quality, artifacts, "
        "deformed, disfigured, extra limbs, bad anatomy, jpeg artifacts, "
        "oversaturated, underexposed, noise, grain"
    )
    size = f"{hw_settings.max_size}x{hw_settings.max_size}"
    steps: int = hw_settings.recommended_steps
    guidance: float = hw_settings.recommended_guidance

    while True:
        try:
            user_input = input("\n🎨 Prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "quit":
            print("Goodbye!")
            break

        if cmd == "help":
            print("""
Commands:
  quit              - Exit the program
  help              - Show this help
  settings          - Show current settings
  hardware          - Show hardware info and recommendations
  negative <text>   - Set negative prompt
  size <WxH>        - Set image size (e.g., 512x512, 1280x720)
  steps <n>         - Set inference steps (28-50)
  guidance <n>      - Set CFG scale (3.0-5.0)
  reset             - Reset to default settings
  clearmem          - Clear GPU memory

Just type a prompt to generate an image!
""")
            continue

        if cmd == "settings":
            print(f"\n  Negative: {negative_prompt}")
            print(f"  Size: {size}")
            print(f"  Steps: {steps}")
            print(f"  Guidance: {guidance}")
            continue

        if cmd == "hardware":
            hw = get_optimal_settings()
            print(f"\n  VRAM: {hw.vram_gb:.1f}GB total, {hw.free_vram_gb:.1f}GB free")
            print(f"  Recommended size: {hw.max_size}x{hw.max_size}")
            print(f"  Recommended steps: {hw.recommended_steps}")
            if hw.enable_sequential_cpu_offload:
                offload = "sequential"
            elif hw.enable_model_cpu_offload:
                offload = "model"
            else:
                offload = "off"
            print(f"  CPU offload: {offload}")
            print(f"  VAE tiling: {'on' if hw.enable_vae_tiling else 'off'}")
            continue

        if cmd == "clearmem":
            clear_memory(aggressive=True)
            # Force refresh of hardware settings to get updated VRAM
            hw = get_optimal_settings(force_refresh=True)
            print(f"  ✓ Memory cleared. Free VRAM: {hw.free_vram_gb:.1f}GB")
            continue

        if cmd.startswith("negative "):
            negative_prompt = user_input[9:].strip()
            print(f"  ✓ Negative prompt set to: {negative_prompt}")
            continue

        if cmd.startswith("size "):
            size = user_input[5:].strip()
            print(f"  ✓ Size set to: {size}")
            continue

        if cmd.startswith("steps "):
            try:
                steps = int(user_input[6:].strip())
                print(f"  ✓ Steps set to: {steps}")
            except ValueError:
                print("  ❌ Invalid number")
            continue

        if cmd.startswith("guidance "):
            try:
                guidance = float(user_input[9:].strip())
                print(f"  ✓ Guidance set to: {guidance}")
            except ValueError:
                print("  ❌ Invalid number")
            continue

        if cmd == "reset":
            hw = get_optimal_settings()
            negative_prompt = "text, watermark, logo, blurry, low quality, artifacts"
            size = f"{hw.max_size}x{hw.max_size}"
            steps = hw.recommended_steps
            guidance = hw.recommended_guidance
            print("  ✓ Settings reset to hardware-optimized defaults")
            continue

        # Generate image with the user's prompt
        try:
            generated_path = generate_image(
                prompt=user_input,
                negative_prompt=negative_prompt,
                size=size,
                steps=steps,
                guidance=guidance,
            )

            try:
                open_it = input("\nOpen image? [Y/n]: ").strip().lower()
                if open_it != "n":
                    if sys.platform == "win32":
                        os.startfile(generated_path)  # type: ignore[attr-defined]
                    elif sys.platform == "darwin":
                        subprocess.run(["open", generated_path], check=False)
                    else:
                        subprocess.run(["xdg-open", generated_path], check=False)
            except OSError:
                pass

        except RuntimeError as e:
            print(f"\n❌ Error: {e}")
            # Aggressive cleanup on error
            clear_memory(aggressive=True)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate images using Z-Image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Read defaults from .env (Z_IMAGE_* vars), falling back to sensible defaults
    default_size = os.getenv("Z_IMAGE_SIZE", "1024x1024")
    default_steps = int(os.getenv("Z_IMAGE_STEPS", "28"))
    default_guidance = float(os.getenv("Z_IMAGE_GUIDANCE", "4.0"))
    default_dtype = os.getenv("Z_IMAGE_DTYPE", None)
    default_offload = os.getenv("Z_IMAGE_OFFLOAD", None)
    # Normalize offload value for CLI default
    if default_offload in ("1", "sequential"):
        default_offload = "sequential"
    elif default_offload == "model":
        default_offload = "model"
    else:
        default_offload = None
    default_negative = os.getenv(
        "Z_IMAGE_NEGATIVE_PROMPT",
        "text, watermark, logo, blurry, low quality, artifacts, "
        "deformed, disfigured, extra limbs, bad anatomy, jpeg artifacts, "
        "oversaturated, underexposed, noise, grain",
    )

    parser.add_argument("prompt", nargs="?", help="Text prompt for image generation")
    parser.add_argument(
        "-p", "--prompt", dest="prompt_flag", help="Text prompt (alternative)"
    )
    parser.add_argument(
        "-n",
        "--negative",
        default=default_negative,
        help="Negative prompt",
    )

    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument(
        "-s", "--size", default=default_size, help=f"Image size as WIDTHxHEIGHT (default: {default_size})"
    )
    parser.add_argument(
        "--steps", type=int, default=default_steps, help=f"Inference steps (default: {default_steps})"
    )
    parser.add_argument(
        "--guidance", type=float, default=default_guidance, help=f"CFG guidance scale (default: {default_guidance})"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "-f", "--fast", action="store_true", help="Fast mode for quicker generation"
    )

    # Hardware-specific overrides (defaults read from .env)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=default_dtype,
        help="Override data type (e.g., float16 for performance)",
    )
    parser.add_argument(
        "--cpu-offload",
        choices=["none", "model", "sequential"],
        default=default_offload,
        help="CPU offloading strategy",
    )
    parser.add_argument(
        "--no-attention-slicing", action="store_true", help="Disable attention slicing"
    )
    parser.add_argument(
        "--no-vae-slicing", action="store_true", help="Disable VAE slicing"
    )
    parser.add_argument(
        "--no-tf32", action="store_true", help="Disable TF32 for Ampere+ GPUs"
    )

    args = parser.parse_args()
    check_requirements()

    prompt_text = args.prompt_flag or args.prompt

    if args.interactive or not prompt_text:
        interactive_mode()
    else:
        try:
            result_path = generate_image(
                prompt=prompt_text,
                negative_prompt=args.negative,
                output_path=args.output,
                size=args.size,
                steps=args.steps,
                guidance=args.guidance,
                seed=args.seed,
                fast_mode=args.fast,
                args=args,
            )

            if sys.platform == "win32":
                try:
                    os.startfile(result_path)  # type: ignore[attr-defined]
                except OSError:
                    pass

        except KeyboardInterrupt:
            print("\n\nCancelled.")
        except RuntimeError as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()


# ============================================================================
# Module Tests
# ============================================================================


def _create_module_tests():
    """Create and run tests for generate_image module."""
    from test_framework import TestSuite

    suite = TestSuite("Generate Image", "generate_image.py")

    def test_hardware_settings_defaults():
        """Test HardwareSettings has correct defaults."""
        settings = HardwareSettings()
        assert settings.dtype == "bfloat16"
        assert settings.enable_attention_slicing is True
        assert settings.enable_vae_slicing is True
        assert settings.enable_vae_tiling is False
        assert settings.enable_model_cpu_offload is False
        assert settings.max_size == 1024
        assert settings.recommended_steps == 28

    def test_hardware_settings_custom():
        """Test HardwareSettings with custom values."""
        settings = HardwareSettings(max_size=512, recommended_steps=12, vram_gb=4.0)
        assert settings.max_size == 512
        assert settings.recommended_steps == 12
        assert settings.vram_gb == 4.0

    def test_optimize_size_within_limits():
        """Test optimize_size_for_hardware with size within limits."""
        settings = HardwareSettings(max_size=1024)
        w, h = optimize_size_for_hardware(768, 768, settings)
        assert w == 768
        assert h == 768

    def test_optimize_size_exceeds_limits():
        """Test optimize_size_for_hardware clamps to max_size."""
        settings = HardwareSettings(max_size=512)
        w, h = optimize_size_for_hardware(1024, 1024, settings)
        assert w <= 512
        assert h <= 512

    def test_optimize_size_alignment():
        """Test optimize_size_for_hardware aligns to 64-pixel grid."""
        settings = HardwareSettings(max_size=1024)
        w, h = optimize_size_for_hardware(500, 500, settings)
        assert w % 64 == 0
        assert h % 64 == 0

    def test_optimize_size_minimum():
        """Test optimize_size_for_hardware enforces minimum 256."""
        settings = HardwareSettings(max_size=256)
        w, h = optimize_size_for_hardware(100, 100, settings)
        assert w >= 256
        assert h >= 256

    def test_optimize_size_rectangular():
        """Test optimize_size_for_hardware preserves aspect ratio."""
        settings = HardwareSettings(max_size=512)
        w, h = optimize_size_for_hardware(1024, 512, settings)
        assert w <= 512
        assert h <= 512
        # Width should be larger than or equal to height
        assert w >= h

    def test_clear_memory_non_aggressive():
        """Test clear_memory in non-aggressive mode."""
        clear_memory(aggressive=False)  # Should not raise

    def test_clear_memory_aggressive():
        """Test clear_memory in aggressive mode."""
        clear_memory(aggressive=True)  # Should not raise (handles missing torch)

    suite.run_test(
        test_name="HardwareSettings defaults",
        test_func=test_hardware_settings_defaults,
        test_summary="Tests HardwareSettings default values",
        method_description="Creates HardwareSettings with defaults",
        expected_outcome="All defaults correct",
    )
    suite.run_test(
        test_name="HardwareSettings custom",
        test_func=test_hardware_settings_custom,
        test_summary="Tests HardwareSettings with custom values",
        method_description="Creates HardwareSettings with overrides",
        expected_outcome="Custom values stored correctly",
    )
    suite.run_test(
        test_name="Optimize size within limits",
        test_func=test_optimize_size_within_limits,
        test_summary="Tests size optimization with small input",
        method_description="Passes size within max_size",
        expected_outcome="Size unchanged",
    )
    suite.run_test(
        test_name="Optimize size exceeds limits",
        test_func=test_optimize_size_exceeds_limits,
        test_summary="Tests size optimization with large input",
        method_description="Passes size exceeding max_size",
        expected_outcome="Size clamped to max",
    )
    suite.run_test(
        test_name="Optimize size alignment",
        test_func=test_optimize_size_alignment,
        test_summary="Tests 64-pixel grid alignment",
        method_description="Passes unaligned size",
        expected_outcome="Output aligned to 64-pixel grid",
    )
    suite.run_test(
        test_name="Optimize size minimum",
        test_func=test_optimize_size_minimum,
        test_summary="Tests minimum size enforcement",
        method_description="Passes very small size",
        expected_outcome="Output at least 256x256",
    )
    suite.run_test(
        test_name="Optimize size rectangular",
        test_func=test_optimize_size_rectangular,
        test_summary="Tests aspect ratio handling",
        method_description="Passes rectangular size",
        expected_outcome="Aspect ratio approximately preserved",
    )
    suite.run_test(
        test_name="Clear memory non-aggressive",
        test_func=test_clear_memory_non_aggressive,
        test_summary="Tests non-aggressive memory clear",
        method_description="Calls clear_memory(aggressive=False)",
        expected_outcome="No exception raised",
    )
    suite.run_test(
        test_name="Clear memory aggressive",
        test_func=test_clear_memory_aggressive,
        test_summary="Tests aggressive memory clear",
        method_description="Calls clear_memory(aggressive=True)",
        expected_outcome="No exception raised",
    )

    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
