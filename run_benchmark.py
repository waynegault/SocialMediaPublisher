"""
Automated benchmark script for Z-Image generation.

This script systematically tests different generation settings by launching 
the generate_image.py script as a separate process for each permutation. 
This ensures test isolation and prevents memory-related issues from affecting
subsequent runs.

Usage:
    python run_benchmark.py
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# Get the path to the python executable in the current virtual environment
PYTHON_EXECUTABLE = sys.executable


@dataclass
class BenchmarkResult:
    """Stores the results of a single benchmark run."""

    settings: dict
    time_taken: float
    error: str | None = None
    output_file: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None


def run_single_benchmark(
    prompt: str,
    size: str,
    steps: int,
    guidance: float,
    cpu_offload: str,
    dtype: str,
) -> BenchmarkResult:
    """
    Runs a single benchmark instance of generate_image.py in a subprocess.
    """
    output_dir = Path("benchmark_images")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sanitize settings for filename
    sanitized_settings = f"{size}_s{steps}_g{guidance}_co-{cpu_offload}_dt-{dtype}"
    output_file = output_dir / f"bench_{timestamp}_{sanitized_settings}.png"

    command = [
        PYTHON_EXECUTABLE,
        "generate_image.py",
        prompt,
        f"--size={size}",
        f"--steps={steps}",
        f"--guidance={guidance}",
        f"--cpu-offload={cpu_offload}",
        f"--dtype={dtype}",
        f"--output={output_file}",
    ]

    start_time = time.time()
    error_message = None

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit code
            encoding="utf-8",
        )
        if process.returncode != 0:
            # Capture stderr or stdout if stderr is empty
            error_output = process.stderr.strip() or process.stdout.strip()
            # Get the last few lines of the error
            error_message = "\n".join(error_output.splitlines()[-5:])

    except Exception as e:
        error_message = str(e)

    time_taken = time.time() - start_time

    return BenchmarkResult(
        settings={
            "size": size,
            "steps": steps,
            "guidance": guidance,
            "cpu_offload": cpu_offload,
            "dtype": dtype,
        },
        time_taken=time_taken,
        error=error_message,
        output_file=str(output_file) if error_message is None else None,
    )


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="Z-Image Benchmark")
    parser.add_argument(
        "--prompt",
        default="A photorealistic portrait of a tabby cat",
        help="The prompt to use for all benchmark runs.",
    )
    args = parser.parse_args()

    print("🚀 Starting Z-Image Benchmark...")

    # --- Define settings permutations to test ---
    # Your RTX 3050 Ti has 4GB VRAM, so we'll focus on settings for that.
    sizes = ["512x512", "768x512"]
    steps_list = [12, 20]  # Faster steps for benchmarking
    guidance_list = [4.0]
    cpu_offload_list = ["sequential", "model"]
    dtype_list = ["float16", "bfloat16"]

    permutations = list(
        itertools.product(
            sizes, steps_list, guidance_list, cpu_offload_list, dtype_list
        )
    )
    print(f"  Will test {len(permutations)} different configurations.")

    results = []
    progress_bar = tqdm(permutations, desc="Benchmarking", unit="run")

    for size, steps, guidance, cpu_offload, dtype in progress_bar:
        progress_bar.set_postfix_str(
            f"{size}, {steps} steps, {cpu_offload}, {dtype}"
        )
        result = run_single_benchmark(
            args.prompt, size, steps, guidance, cpu_offload, dtype
        )
        results.append(result)
        if result.error:
            tqdm.write(f"  ❌ FAILED: {result.settings}\n      Error: {result.error}\n")
        else:
            tqdm.write(f"  ✅ SUCCESS: {result.settings} in {result.time_taken:.2f}s")

    # --- Analyze results ---
    successful_runs = [r for r in results if r.is_success]

    if not successful_runs:
        print("\n😭 All benchmark runs failed. Cannot determine optimal settings.")
        return

    # Find the best run. Heuristic: fastest time for the largest image area.
    best_run = min(
        successful_runs,
        key=lambda r: r.time_taken / (
            int(r.settings["size"].split("x")[0]) * int(r.settings["size"].split("x")[1])
        ),
    )

    print("\n" + "=" * 50)
    print("🏆 Benchmark Complete! 🏆")
    print("\nOptimal settings for speed vs. size:")
    print(f"  - Size: {best_run.settings['size']}")
    print(f"  - Steps: {best_run.settings['steps']}")
    print(f"  - Guidance: {best_run.settings['guidance']}")
    print(f"  - CPU Offload: {best_run.settings['cpu_offload']}")
    print(f"  - DType: {best_run.settings['dtype']}")
    print(f"  - Generation Time: {best_run.time_taken:.2f}s")
    print(f"\n  See generated image: {best_run.output_file}")
    print("=" * 50)

    # Save full results to a JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"benchmark_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=4)

    print(f"\nFull results saved to {results_file}")


if __name__ == "__main__":
    main()
