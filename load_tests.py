"""Load Testing Suite for Social Media Publisher.

This module provides load testing capabilities to ensure the system
scales under load and identify performance bottlenecks.

Features:
- Test rate limiter under load
- Test database performance
- Test concurrent story processing
- Identify bottlenecks

TASK 7.3: Load Testing

Note: For production load testing, consider using Locust (pip install locust).
This module provides lightweight in-process load testing for development.
"""

import concurrent.futures
import logging
import random
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from database import Database, Story
from rate_limiter import AdaptiveRateLimiter

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LoadTestResult:
    """Result from a load test run."""

    test_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    requests_per_second: float = 0.0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate(),
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "requests_per_second": self.requests_per_second,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors[:10],  # First 10 errors
        }


@dataclass
class LoadTestConfig:
    """Configuration for a load test."""

    name: str
    num_requests: int = 100
    concurrency: int = 10
    warmup_requests: int = 5
    think_time_ms: float = 0.0  # Delay between requests per user
    timeout_seconds: float = 30.0


# =============================================================================
# Load Test Runner
# =============================================================================


class LoadTestRunner:
    """Runs load tests with configurable concurrency."""

    def __init__(self, config: LoadTestConfig) -> None:
        """Initialize the load test runner.

        Args:
            config: Load test configuration
        """
        self.config = config
        self.latencies: list[float] = []
        self.errors: list[str] = []
        self.lock = threading.Lock()
        self.successful = 0
        self.failed = 0

    def run(self, task: Callable[[], None]) -> LoadTestResult:
        """Run the load test.

        Args:
            task: Callable to execute for each request

        Returns:
            LoadTestResult with metrics
        """
        self.latencies = []
        self.errors = []
        self.successful = 0
        self.failed = 0

        # Warmup phase
        logger.info(f"Warmup: running {self.config.warmup_requests} requests...")
        for _ in range(self.config.warmup_requests):
            try:
                task()
            except Exception:
                pass

        # Main test phase
        logger.info(
            f"Starting load test: {self.config.num_requests} requests, "
            f"{self.config.concurrency} concurrent"
        )

        start_time = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.concurrency
        ) as executor:
            futures = []

            for _ in range(self.config.num_requests):
                future = executor.submit(self._execute_task, task)
                futures.append(future)

            # Wait for all to complete
            concurrent.futures.wait(futures, timeout=self.config.timeout_seconds)

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Calculate results
        return self._calculate_results(duration)

    def _execute_task(self, task: Callable[[], None]) -> None:
        """Execute a single task and record metrics."""
        # Think time (simulates user delay)
        if self.config.think_time_ms > 0:
            time.sleep(self.config.think_time_ms / 1000)

        start = time.perf_counter()
        try:
            task()
            latency = (time.perf_counter() - start) * 1000  # Convert to ms

            with self.lock:
                self.latencies.append(latency)
                self.successful += 1

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000

            with self.lock:
                self.latencies.append(latency)
                self.failed += 1
                self.errors.append(str(e)[:100])

    def _calculate_results(self, duration: float) -> LoadTestResult:
        """Calculate test results from collected metrics."""
        result = LoadTestResult(
            test_name=self.config.name,
            total_requests=self.successful + self.failed,
            successful_requests=self.successful,
            failed_requests=self.failed,
            duration_seconds=duration,
            errors=self.errors,
        )

        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            result.min_latency_ms = min(sorted_latencies)
            result.max_latency_ms = max(sorted_latencies)
            result.avg_latency_ms = statistics.mean(sorted_latencies)

            # Percentiles
            n = len(sorted_latencies)
            result.p50_latency_ms = sorted_latencies[int(n * 0.50)]
            result.p95_latency_ms = sorted_latencies[int(n * 0.95)]
            result.p99_latency_ms = sorted_latencies[min(int(n * 0.99), n - 1)]

        if duration > 0:
            result.requests_per_second = result.total_requests / duration

        return result


# =============================================================================
# Load Tests
# =============================================================================


class LoadTests:
    """Collection of load tests for the Social Media Publisher."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize load tests.

        Args:
            db_path: Path to test database (uses :memory: by default)
        """
        self.db_path = db_path or ":memory:"
        self.results: list[LoadTestResult] = []

    def run_all(self) -> list[LoadTestResult]:
        """Run all load tests."""
        self.results = []

        tests = [
            self.test_rate_limiter_under_load,
            self.test_database_read_performance,
            self.test_database_write_performance,
            self.test_concurrent_story_queries,
            self.test_database_connection_pool,
        ]

        for test in tests:
            try:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Running: {test.__name__}")
                logger.info("=" * 60)

                result = test()
                self.results.append(result)

                self._print_result(result)

            except Exception as e:
                logger.error(f"Test {test.__name__} failed: {e}")

        return self.results

    def _print_result(self, result: LoadTestResult) -> None:
        """Print a formatted test result."""
        print(f"\n📊 {result.test_name}")
        print(
            f"   Requests: {result.total_requests} ({result.success_rate():.1f}% success)"
        )
        print(f"   Throughput: {result.requests_per_second:.1f} req/s")
        print(
            f"   Latency (ms): min={result.min_latency_ms:.1f}, avg={result.avg_latency_ms:.1f}, max={result.max_latency_ms:.1f}"
        )
        print(
            f"   Percentiles: p50={result.p50_latency_ms:.1f}, p95={result.p95_latency_ms:.1f}, p99={result.p99_latency_ms:.1f}"
        )

        if result.failed_requests > 0:
            print(f"   ⚠️  Failures: {result.failed_requests}")
            if result.errors:
                print(f"   First error: {result.errors[0]}")

    def test_rate_limiter_under_load(self) -> LoadTestResult:
        """Test rate limiter performance under concurrent load."""
        rate_limiter = AdaptiveRateLimiter(
            initial_fill_rate=100.0,  # High rate for testing
            min_fill_rate=10.0,
            max_fill_rate=1000.0,
        )

        def task() -> None:
            # Simulate the rate limiting workflow
            rate_limiter.wait(endpoint="test_endpoint")
            # Small computation to simulate work
            _ = sum(range(1000))
            # Record success
            rate_limiter.on_success(endpoint="test_endpoint")

        config = LoadTestConfig(
            name="Rate Limiter Concurrent Access",
            num_requests=1000,
            concurrency=50,
            warmup_requests=10,
        )

        runner = LoadTestRunner(config)
        return runner.run(task)

    def test_database_read_performance(self) -> LoadTestResult:
        """Test database read performance under load."""
        # Setup test database with sample data
        db = Database(self.db_path)
        self._populate_test_data(db, num_stories=100)

        def task() -> None:
            # Random read operations
            operation = random.choice(
                ["get_verified", "get_needing_images", "get_by_id"]
            )

            if operation == "get_verified":
                stories = db.get_stories_needing_verification()
                assert len(stories) >= 0
            elif operation == "get_needing_images":
                stories = db.get_stories_needing_images(min_quality=70)
                assert isinstance(stories, list)
            else:
                story_id = random.randint(1, 100)
                _ = db.get_story(story_id)
                # May be None if ID doesn't exist

        config = LoadTestConfig(
            name="Database Read Performance",
            num_requests=500,
            concurrency=20,
            warmup_requests=10,
        )

        runner = LoadTestRunner(config)
        return runner.run(task)

    def test_database_write_performance(self) -> LoadTestResult:
        """Test database write performance under load."""
        db = Database(self.db_path)

        counter = [0]
        counter_lock = threading.Lock()

        def task() -> None:
            with counter_lock:
                counter[0] += 1
                idx = counter[0]

            # Create a new story
            story = Story(
                title=f"Load Test Story {idx}",
                summary=f"Summary for load test story {idx}",
                source_links=[f"https://example.com/story{idx}"],
                quality_score=random.randint(60, 100),
                category="Technology",
            )

            story_id = db.add_story(story)
            assert story_id > 0

            # Update the story
            story.id = story_id
            story.quality_score = random.randint(70, 100)
            db.update_story(story)

        config = LoadTestConfig(
            name="Database Write Performance",
            num_requests=200,
            concurrency=10,
            warmup_requests=5,
        )

        runner = LoadTestRunner(config)
        return runner.run(task)

    def test_concurrent_story_queries(self) -> LoadTestResult:
        """Test concurrent story query patterns."""
        db = Database(self.db_path)
        self._populate_test_data(db, num_stories=200)

        def task() -> None:
            # Simulate typical workflow queries
            queries = [
                lambda: db.get_stories_needing_images(min_quality=70),
                lambda: db.get_stories_needing_verification(),
                lambda: db.get_published_stories(),
                lambda: db.get_stories_needing_enrichment(),
            ]

            query = random.choice(queries)
            result = query()
            assert isinstance(result, list)

        config = LoadTestConfig(
            name="Concurrent Story Queries",
            num_requests=300,
            concurrency=30,
            warmup_requests=5,
        )

        runner = LoadTestRunner(config)
        return runner.run(task)

    def test_database_connection_pool(self) -> LoadTestResult:
        """Test database connection handling under many concurrent connections."""
        db_path = self.db_path

        def task() -> None:
            # Each task creates its own connection (simulating connection pool stress)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Simple query
            cursor.execute("SELECT COUNT(*) FROM stories")
            _ = cursor.fetchone()[0]  # Verify query works

            conn.close()

        # First ensure the stories table exists
        db = Database(db_path)
        self._populate_test_data(db, num_stories=10)
        # Database doesn't have close(), it auto-closes

        config = LoadTestConfig(
            name="Database Connection Pool Stress",
            num_requests=500,
            concurrency=100,  # High concurrency
            warmup_requests=10,
        )

        runner = LoadTestRunner(config)
        return runner.run(task)

    def _populate_test_data(self, db: Database, num_stories: int) -> None:
        """Populate database with test data."""
        for i in range(num_stories):
            story = Story(
                title=f"Test Story {i}",
                summary=f"This is test story {i} for load testing purposes.",
                source_links=[f"https://example.com/story{i}"],
                quality_score=random.randint(50, 100),
                category=random.choice(["Technology", "Research", "Industry", "AI"]),
                verification_status=random.choice(["pending", "approved"]),
                publish_status=random.choice(["unpublished", "published"]),
            )
            db.add_story(story)


# =============================================================================
# Benchmark Functions
# =============================================================================


def benchmark_function(
    func: Callable,
    iterations: int = 100,
    warmup: int = 5,
) -> dict:
    """Benchmark a function's performance.

    Args:
        func: Function to benchmark
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with benchmark results
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    sorted_times = sorted(times)
    n = len(sorted_times)

    return {
        "iterations": iterations,
        "min_ms": min(times),
        "max_ms": max(times),
        "avg_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if n > 1 else 0,
        "p50_ms": sorted_times[int(n * 0.50)],
        "p95_ms": sorted_times[int(n * 0.95)],
        "p99_ms": sorted_times[min(int(n * 0.99), n - 1)],
    }


def run_load_tests() -> None:
    """Run all load tests and print summary."""
    print("\n" + "=" * 60)
    print("  Social Media Publisher - Load Tests")
    print("=" * 60 + "\n")

    tests = LoadTests()
    results = tests.run_all()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.success_rate() >= 95.0)
    failed = len(results) - passed

    print(f"\n✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    # Identify bottlenecks
    bottlenecks = []
    for r in results:
        if r.p95_latency_ms > 100:
            bottlenecks.append(f"{r.test_name}: p95={r.p95_latency_ms:.1f}ms")
        if r.success_rate() < 99.0:
            bottlenecks.append(f"{r.test_name}: success_rate={r.success_rate():.1f}%")

    if bottlenecks:
        print("\n⚠️  Potential Bottlenecks:")
        for b in bottlenecks:
            print(f"   - {b}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_load_tests()
