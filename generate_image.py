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

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def check_requirements():
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
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ GPU: {gpu_name} ({vram:.1f}GB VRAM)")
            if vram < 8:
                print(
                    "  💡 Tip: Use --fast mode for faster generation on low VRAM GPUs"
                )
    except ImportError:
        print("❌ PyTorch not installed. Run: pip install torch torchvision")
        sys.exit(1)

    try:
        from diffusers import ZImagePipeline  # type: ignore[attr-defined]  # noqa: F401
    except ImportError:
        print("❌ diffusers not installed or missing ZImagePipeline.")
        print("   Run: pip install git+https://github.com/huggingface/diffusers")
        sys.exit(1)


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    output_path: str | None = None,
    size: str = "1024x1024",
    steps: int = 28,
    guidance: float = 4.0,
    seed: int | None = None,
    fast_mode: bool = False,
) -> str:
    """Generate an image using Z-Image.

    Args:
        prompt: Text description of the image
        negative_prompt: Things to avoid in the image
        output_path: Where to save the image (auto-generated if not provided)
        size: Image size as 'WIDTHxHEIGHT'
        steps: Number of inference steps (28-50)
        guidance: CFG guidance scale (3.0-5.0)
        seed: Random seed for reproducibility
        fast_mode: Use optimized settings for faster generation

    Returns:
        Path to the saved image
    """
    from image_providers import get_image_provider
    from image_providers.z_image import ZImageProvider

    # Fast mode overrides for low VRAM GPUs
    if fast_mode:
        # Reduce steps and use smaller default size
        steps = min(steps, 20)
        if size == "1024x1024":
            size = "768x768"  # Smaller default for fast mode
        print("⚡ Fast mode enabled (reduced steps and optimizations)")

    # Parse size
    try:
        width, height = map(int, size.lower().split("x"))
    except ValueError:
        print(f"Invalid size format: {size}. Using 1024x1024")
        width, height = 1024, 1024

    # Get or create provider
    provider = get_image_provider(
        provider_name="z_image",
        size=size,
    )

    # Override settings if provider is ZImageProvider
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
    print()

    start_time = time.time()

    # Generate
    if isinstance(provider, ZImageProvider):
        # Use seed if provided
        if seed is not None:
            import torch
            import io

            pipe = provider._get_pipeline()
            result = pipe(  # type: ignore[operator]
                prompt=prompt,
                negative_prompt=negative_prompt or provider.default_negative_prompt,
                height=height,
                width=width,
                cfg_normalization=False,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=torch.Generator(provider.device).manual_seed(seed),
            )
            image = result.images[0]  # type: ignore[union-attr]
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        else:
            image_bytes = provider.generate(prompt, negative_prompt)
    else:
        image_bytes = provider.generate(prompt)

    elapsed = time.time() - start_time

    # Determine output path
    if not output_path:
        output_dir = Path("generated_images")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a safe filename from prompt
        safe_prompt = "".join(
            c if c.isalnum() or c in " -_" else "" for c in prompt[:30]
        )
        safe_prompt = safe_prompt.strip().replace(" ", "_")
        output_path = str(output_dir / f"z_image_{timestamp}_{safe_prompt}.png")

    # Save image
    with open(output_path, "wb") as f:
        f.write(image_bytes)

    print(f"\n✅ Image saved to: {output_path}")
    print(f"   Size: {len(image_bytes) // 1024}KB")
    print(f"   Time: {elapsed:.1f}s")

    return output_path


def interactive_mode():
    """Run interactive prompt mode."""
    print("\n" + "=" * 50)
    print("  Z-Image Interactive Generator")
    print("=" * 50)
    print("\nEnter prompts to generate images. Type 'quit' to exit.")
    print("Type 'help' for advanced options.\n")

    # Default settings
    negative_prompt = "text, watermark, logo, blurry, low quality, artifacts"
    size = "1024x1024"
    steps = 28
    guidance = 4.0

    while True:
        try:
            prompt = input("\n🎨 Prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() == "quit":
            print("Goodbye!")
            break

        if prompt.lower() == "help":
            print("""
Commands:
  quit              - Exit the program
  help              - Show this help
  settings          - Show current settings
  negative <text>   - Set negative prompt
  size <WxH>        - Set image size (e.g., 512x512, 1280x720)
  steps <n>         - Set inference steps (28-50)
  guidance <n>      - Set CFG scale (3.0-5.0)
  reset             - Reset to default settings

Just type a prompt to generate an image!
""")
            continue

        if prompt.lower() == "settings":
            print(f"\n  Negative: {negative_prompt}")
            print(f"  Size: {size}")
            print(f"  Steps: {steps}")
            print(f"  Guidance: {guidance}")
            continue

        if prompt.lower().startswith("negative "):
            negative_prompt = prompt[9:].strip()
            print(f"  ✓ Negative prompt set to: {negative_prompt}")
            continue

        if prompt.lower().startswith("size "):
            size = prompt[5:].strip()
            print(f"  ✓ Size set to: {size}")
            continue

        if prompt.lower().startswith("steps "):
            try:
                steps = int(prompt[6:].strip())
                print(f"  ✓ Steps set to: {steps}")
            except ValueError:
                print("  ❌ Invalid number")
            continue

        if prompt.lower().startswith("guidance "):
            try:
                guidance = float(prompt[9:].strip())
                print(f"  ✓ Guidance set to: {guidance}")
            except ValueError:
                print("  ❌ Invalid number")
            continue

        if prompt.lower() == "reset":
            negative_prompt = "text, watermark, logo, blurry, low quality, artifacts"
            size = "1024x1024"
            steps = 28
            guidance = 4.0
            print("  ✓ Settings reset to defaults")
            continue

        # Generate image
        try:
            output_path = generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                size=size,
                steps=steps,
                guidance=guidance,
            )

            # Offer to open the image
            try:
                open_it = input("\nOpen image? [Y/n]: ").strip().lower()
                if open_it != "n":
                    import subprocess

                    if sys.platform == "win32":
                        os.startfile(output_path)  # type: ignore[attr-defined]
                    elif sys.platform == "darwin":
                        subprocess.run(["open", output_path])
                    else:
                        subprocess.run(["xdg-open", output_path])
            except Exception:
                pass

        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Z-Image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Text prompt for image generation (omit for interactive mode)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        dest="prompt_flag",
        help="Text prompt (alternative to positional argument)",
    )
    parser.add_argument(
        "-n",
        "--negative",
        default="text, watermark, logo, blurry, low quality, artifacts",
        help="Negative prompt (things to avoid)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (auto-generated if not specified)",
    )
    parser.add_argument(
        "-s",
        "--size",
        default="1024x1024",
        help="Image size as WIDTHxHEIGHT (default: 1024x1024)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Inference steps (default: 28, range: 20-50)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=4.0,
        help="CFG guidance scale (default: 4.0, range: 3.0-5.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="Fast mode: reduces steps and size for ~3-4x faster generation",
    )

    args = parser.parse_args()

    # Check requirements first
    check_requirements()

    # Determine prompt
    prompt = args.prompt_flag or args.prompt

    if args.interactive or not prompt:
        interactive_mode()
    else:
        try:
            output_path = generate_image(
                prompt=prompt,
                negative_prompt=args.negative,
                output_path=args.output,
                size=args.size,
                steps=args.steps,
                guidance=args.guidance,
                seed=args.seed,
                fast_mode=args.fast,
            )

            # Open the image on Windows
            if sys.platform == "win32":
                try:
                    os.startfile(output_path)  # type: ignore[attr-defined]
                except Exception:
                    pass

        except KeyboardInterrupt:
            print("\n\nCancelled.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
