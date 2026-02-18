"""
Image provider benchmark — measure generation speed across providers.

Runs each configured provider with one or more prompts, records timing,
and reports a ranked comparison.  Provider-specific settings (Z-Image
steps/guidance, DALL-E quality, etc.) come from the environment or
sensible defaults — this script is provider-agnostic.

Usage:
    python run_benchmark.py                          # All configured providers
    python run_benchmark.py --providers pollinations huggingface
    python run_benchmark.py --providers z_image --sizes 512x512 768x768
    python run_benchmark.py --list                   # Show available providers
    python run_benchmark.py --runs 3                 # 3 runs per combo for averaging
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from image_providers import get_image_provider, list_available_providers  # noqa: E402
from image_providers.base import ImageProviderError  # noqa: E402

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = "A photorealistic portrait of a tabby cat sitting on a windowsill"

DEFAULT_SIZES = ["1024x1024"]

# Providers that are typically slow or queue-based get a longer timeout.
_SLOW_PROVIDERS = {"ai_horde", "z_image"}


@dataclass
class BenchmarkResult:
    """Stores the results of a single benchmark run."""

    provider: str
    model: str
    size: str
    run_index: int
    time_seconds: float
    image_bytes_len: int = 0
    error: str | None = None
    output_file: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None


@dataclass
class ProviderSummary:
    """Aggregated stats for one provider + size combination."""

    provider: str
    model: str
    size: str
    runs: int = 0
    successes: int = 0
    failures: int = 0
    times: list[float] = field(default_factory=list)

    @property
    def mean_time(self) -> float:
        return statistics.mean(self.times) if self.times else 0.0

    @property
    def median_time(self) -> float:
        return statistics.median(self.times) if self.times else 0.0

    @property
    def min_time(self) -> float:
        return min(self.times) if self.times else 0.0

    @property
    def max_time(self) -> float:
        return max(self.times) if self.times else 0.0


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------


def _run_single(
    provider_name: str,
    prompt: str,
    size: str,
    run_index: int,
    timeout: int,
    save_images: bool,
    output_dir: Path,
) -> BenchmarkResult:
    """Run a single generation and return the result."""
    error_message: str | None = None
    image_bytes_len = 0
    output_file: str | None = None

    start = time.time()
    try:
        provider = get_image_provider(
            provider_name=provider_name, size=size, timeout=timeout,
        )
        image_bytes = provider.generate(prompt)
        image_bytes_len = len(image_bytes)

        if save_images:
            safe_name = provider_name.replace("/", "_")
            fname = f"{safe_name}_{size}_{run_index}.png"
            path = output_dir / fname
            path.write_bytes(image_bytes)
            output_file = str(path)

    except ImageProviderError as exc:
        error_message = str(exc)
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"

    elapsed = time.time() - start

    return BenchmarkResult(
        provider=provider_name,
        model="",  # filled later
        size=size,
        run_index=run_index,
        time_seconds=elapsed,
        image_bytes_len=image_bytes_len,
        error=error_message,
        output_file=output_file,
    )


def _list_providers() -> None:
    """Print configured providers and exit."""
    available = list_available_providers()
    print("\nAvailable image providers:")
    print("-" * 55)
    for name, info in sorted(available.items()):
        status = "✓ ready" if info["configured"] else "✗ not configured"
        model = info.get("default_model", "")[:35]
        reqs = ", ".join(info.get("requires", [])) or "none"
        print(f"  {name:<18} {status:<20} model: {model}")
        if not info["configured"]:
            print(f"  {'':<18} requires: {reqs}")
    print()


def _print_summary(summaries: list[ProviderSummary]) -> None:
    """Print a ranked summary table."""
    # Sort by median time (fastest first), failures last
    ranked = sorted(summaries, key=lambda s: (s.successes == 0, s.median_time))

    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS  (ranked by median time)")
    print("=" * 70)
    print(
        f"  {'Provider':<18} {'Size':<12} {'Runs':>4} {'OK':>3} "
        f"{'Median':>8} {'Mean':>8} {'Min':>8} {'Max':>8}"
    )
    print("-" * 70)

    for s in ranked:
        if s.successes == 0:
            print(
                f"  {s.provider:<18} {s.size:<12} {s.runs:>4} {s.successes:>3} "
                f"{'— all failed —':>36}"
            )
        else:
            print(
                f"  {s.provider:<18} {s.size:<12} {s.runs:>4} {s.successes:>3} "
                f"{s.median_time:>7.1f}s {s.mean_time:>7.1f}s "
                f"{s.min_time:>7.1f}s {s.max_time:>7.1f}s"
            )

    # Fastest provider
    successful = [s for s in ranked if s.successes > 0]
    if successful:
        best = successful[0]
        print(f"\n  🏆 Fastest: {best.provider} @ {best.size} — {best.median_time:.1f}s median")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark image generation providers",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help="Prompt to use (default: tabby cat portrait)",
    )
    parser.add_argument(
        "--providers", nargs="+", default=None,
        help="Providers to benchmark (default: all configured)",
    )
    parser.add_argument(
        "--sizes", nargs="+", default=DEFAULT_SIZES,
        help="Image sizes to test (default: 1024x1024)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Runs per provider/size combo for averaging (default: 1)",
    )
    parser.add_argument(
        "--timeout", type=int, default=180,
        help="Timeout per generation in seconds (default: 180)",
    )
    parser.add_argument(
        "--save-images", action="store_true",
        help="Save generated images to benchmark_images/",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_providers",
        help="List available providers and exit",
    )
    args = parser.parse_args()

    if args.list_providers:
        _list_providers()
        return

    # Determine which providers to run
    available = list_available_providers()
    if args.providers:
        providers = [p for p in args.providers if p in available]
        missing = set(args.providers) - set(providers)
        if missing:
            print(f"⚠️  Unknown providers skipped: {', '.join(sorted(missing))}")
    else:
        providers = [p for p, info in available.items() if info["configured"]]

    if not providers:
        print("❌ No configured providers available. Run with --list to see status.")
        sys.exit(1)

    output_dir = Path("benchmark_images")
    if args.save_images:
        output_dir.mkdir(exist_ok=True)

    total_combos = len(providers) * len(args.sizes) * args.runs
    print(f"\n🚀 Image Provider Benchmark")
    print(f"   Providers : {', '.join(providers)}")
    print(f"   Sizes     : {', '.join(args.sizes)}")
    print(f"   Runs/combo: {args.runs}")
    print(f"   Total runs: {total_combos}")
    print(f"   Prompt    : {args.prompt[:80]}{'…' if len(args.prompt) > 80 else ''}\n")

    all_results: list[BenchmarkResult] = []
    summaries: list[ProviderSummary] = []

    for provider_name in providers:
        default_model = available[provider_name].get("default_model", "")
        timeout = max(args.timeout, 300) if provider_name in _SLOW_PROVIDERS else args.timeout

        for size in args.sizes:
            summary = ProviderSummary(
                provider=provider_name, model=default_model, size=size,
            )
            print(f"  ▶ {provider_name} @ {size} ", end="", flush=True)

            for i in range(args.runs):
                result = _run_single(
                    provider_name, args.prompt, size, i + 1,
                    timeout, args.save_images, output_dir,
                )
                result.model = default_model
                all_results.append(result)
                summary.runs += 1

                if result.is_success:
                    summary.successes += 1
                    summary.times.append(result.time_seconds)
                    print(f" {result.time_seconds:.1f}s", end="", flush=True)
                else:
                    summary.failures += 1
                    print(f" ✗", end="", flush=True)

                # Brief pause between runs to be polite to APIs
                if i < args.runs - 1:
                    time.sleep(1)

            # Line summary
            if summary.successes > 0:
                print(f"  → median {summary.median_time:.1f}s")
            else:
                print(f"  → all failed")

            summaries.append(summary)

    _print_summary(summaries)

    # Save raw results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"benchmark_results_{timestamp}.json")
    results_file.write_text(
        json.dumps([asdict(r) for r in all_results], indent=2),
        encoding="utf-8",
    )
    print(f"Raw results saved to {results_file}")


if __name__ == "__main__":
    main()
