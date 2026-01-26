#!/usr/bin/env python3
"""
Test Image Providers - Comprehensive Comparison Script

This script generates the same image across all available providers and models
to enable visual comparison of output quality.

Usage:
    python test_image_providers.py [--prompt "custom prompt"] [--output-dir path]

Generated images are saved to ./generated_images/provider_comparison/ by default.
"""

import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from image_providers import get_image_provider, list_available_providers
from image_providers.base import ImageProviderError

# Load environment variables from .env
load_dotenv()

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ============================================================================
# PROVIDER MODELS REGISTRY
# ============================================================================
# Define all available models for each provider to test

PROVIDER_MODELS = {
    "pollinations": {
        "description": "Free unlimited, no API key required",
        "models": [
            ("flux", "FLUX standard - balanced quality/speed"),
            ("flux-realism", "FLUX Realism - photorealistic focus"),
            ("flux-anime", "FLUX Anime - anime/illustration style"),
            ("flux-3d", "FLUX 3D - 3D rendered style"),
            ("turbo", "Turbo - fastest generation"),
        ],
    },
    "cloudflare": {
        "description": "Cloudflare Workers AI - free tier ~10k/day",
        "models": [
            (
                "@cf/stabilityai/stable-diffusion-xl-base-1.0",
                "SDXL Base - highest quality Stability AI model",
            ),
            (
                "@cf/bytedance/stable-diffusion-xl-lightning",
                "SDXL Lightning - faster, optimised for speed",
            ),
            (
                "@cf/lykon/dreamshaper-8-lcm",
                "Dreamshaper 8 LCM - artistic/creative style",
            ),
        ],
    },
    "ai_horde": {
        "description": "Community-run distributed network - free, queue-based",
        "models": [
            ("SDXL 1.0", "Stable Diffusion XL - high quality"),
            ("stable_diffusion_2.1", "Stable Diffusion 2.1 - reliable classic"),
            ("Deliberate", "Deliberate - photorealistic focus"),
        ],
    },
    "huggingface": {
        "description": "HuggingFace Inference API - free tier",
        "models": [
            (
                "black-forest-labs/FLUX.1-schnell",
                "FLUX Schnell - fast, free, no token needed",
            ),
            (
                "stabilityai/stable-diffusion-xl-base-1.0",
                "SDXL Base - may require token",
            ),
        ],
    },
}


# ============================================================================
# DEFAULT TEST PROMPT
# ============================================================================
# Comprehensive industrial/technical prompt for testing

DEFAULT_PROMPT = """
Photorealistic image of a professional woman working with industrial electrolysis
equipment to produce green hydrogen. She is a chemical engineer, wearing safety
goggles, a hard hat, and a white lab coat, standing next to a large PEM
(Proton Exchange Membrane) electrolyzer stack with visible hydrogen and oxygen
output pipes. The facility is a modern, clean industrial plant with stainless
steel equipment, blue LED status indicators, pressure gauges, and digital
control panels. Soft industrial lighting creates a professional atmosphere.
The engineer is analyzing data on a tablet showing hydrogen production metrics.
In the background, solar panels and wind turbines are visible through large
windows, emphasizing the renewable energy source for the hydrogen production.
High detail, professional photography, 8K quality, sharp focus on the equipment
and engineer, depth of field creating a sense of scale.
""".strip()


def create_output_directory(base_dir: Optional[str] = None) -> Path:
    """Create timestamped output directory for comparison images."""
    if base_dir:
        output_base = Path(base_dir)
    else:
        output_base = Path(__file__).parent / "generated_images" / "provider_comparison"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    return output_dir


def sanitize_filename(name: str) -> str:
    """Convert model name to safe filename."""
    # Replace problematic characters
    safe = name.replace("/", "_").replace("@", "").replace(".", "_")
    safe = safe.replace(" ", "_").replace("-", "_")
    # Remove double underscores
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_").lower()


def save_prompt_file(output_dir: Path, prompt: str) -> None:
    """Save the test prompt to a file for reference."""
    prompt_file = output_dir / "prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    logger.info(f"Prompt saved to: {prompt_file}")


def test_provider_model(
    provider_name: str,
    model_name: str,
    model_desc: str,
    prompt: str,
    output_dir: Path,
    size: str = "1024x1024",
    timeout: int = 120,
) -> dict:
    """Test a single provider/model combination.

    Returns:
        dict with keys: success, provider, model, time_seconds, error, file_path
    """
    result = {
        "provider": provider_name,
        "model": model_name,
        "description": model_desc,
        "success": False,
        "time_seconds": 0.0,
        "error": None,
        "file_path": None,
    }

    safe_model = sanitize_filename(model_name)
    filename = f"{provider_name}_{safe_model}.png"
    output_path = output_dir / filename

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing: {provider_name} / {model_name}")
    logger.info(f"  Description: {model_desc}")
    logger.info(f"  Output: {filename}")

    start_time = time.time()

    try:
        provider = get_image_provider(
            provider_name=provider_name,
            model=model_name,
            size=size,
            timeout=timeout,
        )

        logger.info("  Generating image...")
        image_bytes = provider.generate(prompt)

        # Save the image
        output_path.write_bytes(image_bytes)

        elapsed = time.time() - start_time
        result["success"] = True
        result["time_seconds"] = elapsed
        result["file_path"] = str(output_path)

        logger.info(f"  ✓ Success in {elapsed:.1f}s - {len(image_bytes):,} bytes")

    except ImageProviderError as e:
        elapsed = time.time() - start_time
        result["time_seconds"] = elapsed
        result["error"] = str(e)
        logger.error(f"  ✗ Provider error: {e}")

    except Exception as e:
        elapsed = time.time() - start_time
        result["time_seconds"] = elapsed
        result["error"] = f"{type(e).__name__}: {e}"
        logger.error(f"  ✗ Unexpected error: {type(e).__name__}: {e}")

    return result


def print_summary(results: list[dict], output_dir: Path) -> None:
    """Print a summary of all test results."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    # Group by provider
    by_provider = {}
    for r in results:
        prov = r["provider"]
        if prov not in by_provider:
            by_provider[prov] = []
        by_provider[prov].append(r)

    total_success = 0
    total_failed = 0

    for provider, prov_results in by_provider.items():
        print(f"\n{provider.upper()}")
        print("-" * 40)

        for r in prov_results:
            status = "✓" if r["success"] else "✗"
            time_str = f"{r['time_seconds']:.1f}s"

            if r["success"]:
                total_success += 1
                print(f"  {status} {r['model'][:40]:<40} {time_str}")
            else:
                total_failed += 1
                error_short = (r["error"] or "Unknown error")[:50]
                print(f"  {status} {r['model'][:40]:<40} {time_str} - {error_short}")

    print("\n" + "=" * 70)
    print(f"TOTAL: {total_success} succeeded, {total_failed} failed")
    print(f"Images saved to: {output_dir}")
    print("=" * 70)


def write_results_csv(results: list[dict], output_dir: Path) -> None:
    """Write results to CSV for further analysis."""
    csv_path = output_dir / "results.csv"

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("provider,model,description,success,time_seconds,error,file_path\n")
        for r in results:
            error = (r["error"] or "").replace('"', '""')
            f.write(
                f'"{r["provider"]}","{r["model"]}","{r["description"]}",'
                f'{r["success"]},{r["time_seconds"]:.2f},"{error}","{r["file_path"] or ""}"\n'
            )

    logger.info(f"Results CSV saved to: {csv_path}")


def check_provider_availability() -> dict:
    """Check which providers are configured and available."""
    available = list_available_providers()

    print("\n" + "=" * 70)
    print("PROVIDER AVAILABILITY CHECK")
    print("=" * 70)

    for name, info in available.items():
        status = "✓ Configured" if info["configured"] else "✗ Not configured"
        requires = ", ".join(info.get("requires", [])) or "none"
        default_model = info.get("default_model", "N/A")[:30]
        print(f"  {name:<15} {status:<20} (requires: {requires})")
        print(f"                 Default model: {default_model}")

    print("=" * 70 + "\n")

    return available


def run_comprehensive_test(
    prompt: str,
    output_dir: Path,
    providers: Optional[list[str]] = None,
    size: str = "1024x1024",
    timeout: int = 120,
) -> list[dict]:
    """Run tests across all providers and models.

    Args:
        prompt: The image generation prompt
        output_dir: Directory to save images
        providers: List of provider names to test (None = all configured)
        size: Image size
        timeout: Timeout per generation

    Returns:
        List of result dictionaries
    """
    # Check availability
    available = check_provider_availability()

    # Filter to configured providers
    if providers:
        providers_to_test = [p for p in providers if p in available]
    else:
        # Test all configured providers
        providers_to_test = [p for p, info in available.items() if info["configured"]]

    if not providers_to_test:
        logger.error("No configured providers available for testing!")
        return []

    logger.info(f"Testing providers: {', '.join(providers_to_test)}")

    # Save the prompt for reference
    save_prompt_file(output_dir, prompt)

    # Run all tests
    results = []

    for provider_name in providers_to_test:
        if provider_name not in PROVIDER_MODELS:
            logger.warning(f"No model list defined for {provider_name}, skipping")
            continue

        provider_info = PROVIDER_MODELS[provider_name]
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# Provider: {provider_name.upper()}")
        logger.info(f"# {provider_info['description']}")
        logger.info(f"{'#' * 60}")

        for model_name, model_desc in provider_info["models"]:
            result = test_provider_model(
                provider_name=provider_name,
                model_name=model_name,
                model_desc=model_desc,
                prompt=prompt,
                output_dir=output_dir,
                size=size,
                timeout=timeout,
            )
            results.append(result)

            # Brief pause between requests to be polite to APIs
            time.sleep(1)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test all image providers and models with a comprehensive prompt"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Custom prompt (default: industrial hydrogen equipment scene)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ./generated_images/provider_comparison/TIMESTAMP)",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=None,
        help="Specific providers to test (default: all configured)",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1024x1024",
        help="Image size WIDTHxHEIGHT (default: 1024x1024)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per generation in seconds (default: 120)",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="Just list available providers and exit",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("IMAGE PROVIDER COMPARISON TEST")
    print("=" * 70)

    if args.list_providers:
        check_provider_availability()
        print("\nAvailable models per provider:")
        for prov, info in PROVIDER_MODELS.items():
            print(f"\n{prov.upper()} - {info['description']}")
            for model, desc in info["models"]:
                print(f"  • {model}: {desc}")
        return

    # Create output directory
    output_dir = create_output_directory(args.output_dir)

    # Display the prompt
    print(f"\nTest Prompt:\n{'-' * 40}")
    print(args.prompt[:500] + ("..." if len(args.prompt) > 500 else ""))
    print(f"{'-' * 40}\n")

    # Run tests
    results = run_comprehensive_test(
        prompt=args.prompt,
        output_dir=output_dir,
        providers=args.providers,
        size=args.size,
        timeout=args.timeout,
    )

    if results:
        # Write CSV results
        write_results_csv(results, output_dir)

        # Print summary
        print_summary(results, output_dir)
    else:
        logger.error("No tests were run!")


if __name__ == "__main__":
    main()
