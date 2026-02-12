#!/usr/bin/env python
"""Test script to find optimal Z-Image generation settings."""

import argparse
import gc
import itertools
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from generate_image import get_optimal_settings
from image_providers import get_image_provider
from image_providers.z_image import ZImageProvider
from tqdm import tqdm


@dataclass
class TestResult:
    """Stores the results of a single test run."""
    settings: dict
    time_taken: float
    vram_used: float
    error: str | None = None


def clear_memory(aggressive: bool = True):
    """Clear GPU and system memory."""
    if aggressive:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            torch.cuda.synchronize()


def run_test(
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
) -> TestResult:
    """Run a single generation test with specific settings."""
    start_time = time.time()
    error_message = None
    vram_used = 0

    try:
        # Get a new provider instance for each test to ensure isolation
        provider = get_image_provider(provider_name="z_image", size=f"{width}x{height}")
        if not isinstance(provider, ZImageProvider):
            raise RuntimeError("Could not get ZImageProvider")

        # Apply settings to the provider
        provider.num_inference_steps = steps
        provider.guidance_scale = guidance

        # Manually trigger model loading to measure VRAM
        provider._get_pipeline()
        vram_before = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0

        provider.generate(prompt)

        vram_after = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
        vram_used = (vram_after - vram_before) / (1024**3)

    except Exception as e:
        error_message = str(e)
    finally:
        # Clear memory after each run to isolate tests
        clear_memory()

    time_taken = time.time() - start_time

    return TestResult(
        settings={
            "size": f"{width}x{height}",
            "steps": steps,
            "guidance": guidance,
        },
        time_taken=time_taken,
        vram_used=vram_used,
        error=error_message,
    )


def main():
    """Run the optimization test suite."""
    parser = argparse.ArgumentParser(description="Z-Image Optimizer")
    parser.add_argument(
        "-p",
        "--prompt",
        default="A beautiful landscape painting, detailed, 4k",
        help="Test prompt",
    )
    args = parser.parse_args()

    print("🔬 Starting Z-Image Optimization Test...")
    hw_settings = get_optimal_settings(force_refresh=True)
    print(f"  Hardware: {hw_settings.vram_gb:.1f}GB VRAM, Compute {hw_settings.compute_capability}")

    # Define permutations to test
    sizes = [(512, 512), (768, 512), (512, 768)]
    if hw_settings.max_size >= 768:
        sizes.append((768, 768))

    steps_list = [12, 20, 28]
    guidance_list = [3.0, 4.0]

    # Filter sizes that are too large for the hardware
    sizes = [s for s in sizes if s[0] <= hw_settings.max_size and s[1] <= hw_settings.max_size]

    permutations = list(itertools.product(sizes, steps_list, guidance_list))
    print(f"  Testing {len(permutations)} permutations...")

    results = []

    progress_bar = tqdm(permutations, desc="Testing Settings")
    for size, steps, guidance in progress_bar:
        width, height = size
        progress_bar.set_postfix_str(f"{width}x{height}, {steps} steps, {guidance} CFG")
        
        result = run_test(args.prompt, width, height, steps, guidance)
        results.append(result)

        if result.error:
            tqdm.write(f"\n❌ Error at {result.settings}: {result.error}")

    # Analyze results
    successful_runs = [r for r in results if r.error is None and r.time_taken > 0]
    if not successful_runs:
        print("\nNo successful runs. Cannot determine optimal settings.")
        return

    # Find the best setting (fastest time for a decent quality)
    # Prioritize larger sizes and reasonable steps
    # Simple heuristic: time / (width * height * steps)
    best_run = min(
        successful_runs,
        key=lambda r: r.time_taken
        / (int(r.settings["size"].split("x")[0]) * int(r.settings["size"].split("x")[1]) * r.settings["steps"]),
    )

    print("\n🏆 Optimal Settings Found:")
    print(f"  - Size: {best_run.settings['size']}")
    print(f"  - Steps: {best_run.settings['steps']}")
    print(f"  - Guidance: {best_run.settings['guidance']}")
    print(f"  - Time: {best_run.time_taken:.2f}s")
    print(f"  - VRAM Used: {best_run.vram_used:.2f}GB")

    # Save results to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"optimization_results_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\nFull results saved to {output_file}")


if __name__ == "__main__":
    main()
