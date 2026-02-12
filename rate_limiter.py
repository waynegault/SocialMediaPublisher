#!/usr/bin/env python3

"""
Unified Adaptive Rate Limiting System.

Provides a simplified, unified rate limiting system that uses
an adaptive token bucket algorithm. Adjusts the fill rate based on
API feedback (429 errors and success streaks).

Based on waynes_framework rate_limiter.py.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterMetrics:
    """Metrics for monitoring rate limiter performance."""

    total_requests: int = 0
    total_wait_time: float = 0.0
    rate_decreases: int = 0
    rate_increases: int = 0
    error_429_count: int = 0
    current_fill_rate: float = 0.0
    success_count: int = 0
    tokens_available: float = 0.0
    avg_wait_time: float = 0.0
    endpoint_rate_cap: Optional[float] = None


@dataclass
class _EndpointState:
    """Mutable adaptive state for a specific endpoint's rate limiting."""

    current_rate: float
    min_rate: float
    max_rate: float
    success_count: int = 0
    last_call_time: float = 0.0
    penalty_until: float = 0.0
    total_requests: int = 0
    total_429s: int = 0
    rate_increases: int = 0
    rate_decreases: int = 0


class AdaptiveRateLimiter:
    """
    Unified adaptive rate limiter using token bucket algorithm.

    Features:
    - Per-endpoint rate tracking
    - Adaptive rate adjustment based on 429 errors and success streaks
    - Thread-safe operation
    """

    def __init__(
        self,
        initial_fill_rate: float = 2.0,
        capacity: float = 20.0,
        min_fill_rate: float = 0.1,
        max_fill_rate: float = 5.0,
        success_threshold: int = 5,
        rate_limiter_429_backoff: float = 0.80,
        rate_limiter_success_factor: float = 1.05,
    ):
        if initial_fill_rate <= 0:
            raise ValueError(f"initial_fill_rate must be > 0, got {initial_fill_rate}")
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if min_fill_rate <= 0:
            raise ValueError(f"min_fill_rate must be > 0, got {min_fill_rate}")
        if max_fill_rate < min_fill_rate:
            raise ValueError(
                f"max_fill_rate ({max_fill_rate}) must be >= min_fill_rate ({min_fill_rate})"
            )
        if success_threshold < 1:
            raise ValueError(f"success_threshold must be >= 1, got {success_threshold}")

        self.capacity = capacity
        self.fill_rate = initial_fill_rate
        self.tokens = capacity
        self.last_refill_time = time.monotonic()

        self.min_fill_rate = min_fill_rate
        self.max_fill_rate = max_fill_rate

        self.success_count = 0
        self.success_threshold = success_threshold
        self.rate_limiter_429_backoff = rate_limiter_429_backoff
        self.rate_limiter_success_factor = rate_limiter_success_factor

        self._lock = threading.Lock()

        self._metrics: dict[str, float | int] = {
            "total_requests": 0,
            "total_wait_time": 0.0,
            "rate_decreases": 0,
            "rate_increases": 0,
            "error_429_count": 0,
        }

        self._endpoint_states: dict[str, _EndpointState] = {}
        self._default_endpoint = "_default_"

    def wait(self, endpoint: Optional[str] = None) -> float:
        """Wait according to per-endpoint adaptive rate limiting."""
        with self._lock:
            effective_endpoint = endpoint or self._default_endpoint
            state = self._get_or_create_endpoint_state(effective_endpoint)

            now = time.monotonic()
            wait_time = 0.0

            if state.penalty_until > now:
                penalty_wait = state.penalty_until - now
                time.sleep(penalty_wait)
                now = time.monotonic()
                wait_time += penalty_wait

            min_interval = 1.0 / state.current_rate if state.current_rate > 0 else 1.0
            elapsed = now - state.last_call_time

            if elapsed < min_interval:
                rate_wait = min_interval - elapsed
                time.sleep(rate_wait)
                wait_time += rate_wait

            state.last_call_time = time.monotonic()
            state.total_requests += 1

            self._metrics["total_requests"] += 1
            self._metrics["total_wait_time"] += wait_time

            return wait_time

    def _get_or_create_endpoint_state(self, endpoint: str) -> _EndpointState:
        if endpoint in self._endpoint_states:
            return self._endpoint_states[endpoint]

        initial_rate = (
            self.min_fill_rate + (self.max_fill_rate - self.min_fill_rate) * 0.5
        )
        state = _EndpointState(
            current_rate=initial_rate,
            min_rate=self.min_fill_rate,
            max_rate=self.max_fill_rate,
        )
        self._endpoint_states[endpoint] = state
        return state

    def on_429_error(
        self, endpoint: Optional[str] = None, retry_after: Optional[float] = None
    ) -> None:
        """Handle 429 rate limit error by backing off the rate."""
        with self._lock:
            effective_endpoint = endpoint or self._default_endpoint
            state = self._get_or_create_endpoint_state(effective_endpoint)

            old_rate = state.current_rate
            state.current_rate = max(
                state.current_rate * self.rate_limiter_429_backoff,
                state.min_rate,
            )
            state.success_count = 0
            state.total_429s += 1
            state.rate_decreases += 1

            self._metrics["error_429_count"] += 1
            self._metrics["rate_decreases"] += 1

            cooldown = retry_after or 0.0
            if cooldown > 0:
                state.penalty_until = time.monotonic() + cooldown

            logger.warning(
                f"429 on '{effective_endpoint}': rate {old_rate:.3f} -> "
                f"{state.current_rate:.3f} req/s"
            )

    def on_success(self, endpoint: Optional[str] = None) -> None:
        """Handle successful API call by potentially increasing rate."""
        with self._lock:
            effective_endpoint = endpoint or self._default_endpoint
            state = self._get_or_create_endpoint_state(effective_endpoint)

            state.success_count += 1
            if state.success_count >= self.success_threshold:
                old_rate = state.current_rate
                state.current_rate = min(
                    state.current_rate * self.rate_limiter_success_factor,
                    state.max_rate,
                )
                if state.current_rate > old_rate:
                    state.rate_increases += 1
                    self._metrics["rate_increases"] += 1
                    logger.debug(
                        f"Increased rate for '{effective_endpoint}': "
                        f"{old_rate:.3f} -> {state.current_rate:.3f} req/s"
                    )
                state.success_count = 0

    def get_metrics(self) -> RateLimiterMetrics:
        """Return current metrics."""
        with self._lock:
            avg_wait = 0.0
            total_reqs = int(self._metrics["total_requests"])
            if total_reqs > 0:
                avg_wait = float(self._metrics["total_wait_time"]) / total_reqs

            return RateLimiterMetrics(
                total_requests=total_reqs,
                total_wait_time=float(self._metrics["total_wait_time"]),
                rate_decreases=int(self._metrics["rate_decreases"]),
                rate_increases=int(self._metrics["rate_increases"]),
                error_429_count=int(self._metrics["error_429_count"]),
                current_fill_rate=self.fill_rate,
                success_count=self.success_count,
                tokens_available=self.tokens,
                avg_wait_time=avg_wait,
            )

    def reset(self, endpoint: Optional[str] = None) -> None:
        """Reset rate limiter state for an endpoint or all endpoints."""
        with self._lock:
            if endpoint:
                if endpoint in self._endpoint_states:
                    del self._endpoint_states[endpoint]
            else:
                self._endpoint_states.clear()
                self._metrics = {
                    "total_requests": 0,
                    "total_wait_time": 0.0,
                    "rate_decreases": 0,
                    "rate_increases": 0,
                    "error_429_count": 0,
                }


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests() -> bool:
    """Create unit tests for rate_limiter module."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Rate Limiter Tests", "rate_limiter.py")
    suite.start_suite()

    def test_rate_limiter_creation():
        rl = AdaptiveRateLimiter(initial_fill_rate=2.0, capacity=10.0)
        assert rl.capacity == 10.0
        assert rl.fill_rate == 2.0

    def test_rate_limiter_wait():
        rl = AdaptiveRateLimiter(initial_fill_rate=10.0)
        wait_time = rl.wait(endpoint="test")
        assert wait_time >= 0

    def test_rate_limiter_429_backoff():
        rl = AdaptiveRateLimiter(initial_fill_rate=2.0, rate_limiter_429_backoff=0.5)
        rl.wait(endpoint="test")
        with suppress_logging():
            rl.on_429_error(endpoint="test")
        metrics = rl.get_metrics()
        assert metrics.error_429_count == 1
        assert metrics.rate_decreases == 1

    def test_rate_limiter_success_increase():
        rl = AdaptiveRateLimiter(success_threshold=2, rate_limiter_success_factor=1.5)
        rl.wait(endpoint="test")
        rl.on_success(endpoint="test")
        rl.on_success(endpoint="test")
        metrics = rl.get_metrics()
        assert metrics.rate_increases >= 1

    def test_rate_limiter_metrics():
        rl = AdaptiveRateLimiter()
        rl.wait()
        metrics = rl.get_metrics()
        assert isinstance(metrics, RateLimiterMetrics)
        assert metrics.total_requests == 1

    def test_rate_limiter_reset():
        rl = AdaptiveRateLimiter()
        rl.wait()
        rl.reset()
        metrics = rl.get_metrics()
        assert metrics.total_requests == 0

    def test_rate_limiter_per_endpoint():
        rl = AdaptiveRateLimiter()
        rl.wait(endpoint="endpoint1")
        rl.wait(endpoint="endpoint2")
        metrics = rl.get_metrics()
        assert metrics.total_requests == 2

    suite.run_test(
        test_name="Rate limiter creation",
        test_func=test_rate_limiter_creation,
        test_summary="Tests Rate limiter creation functionality",
        method_description="Calls AdaptiveRateLimiter and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Rate limiter wait",
        test_func=test_rate_limiter_wait,
        test_summary="Tests Rate limiter wait functionality",
        method_description="Calls AdaptiveRateLimiter and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Rate limiter 429 backoff",
        test_func=test_rate_limiter_429_backoff,
        test_summary="Tests Rate limiter 429 backoff functionality",
        method_description="Calls AdaptiveRateLimiter and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Rate limiter success increase",
        test_func=test_rate_limiter_success_increase,
        test_summary="Tests Rate limiter success increase functionality",
        method_description="Calls AdaptiveRateLimiter and verifies the result",
        expected_outcome="Function returns the expected successful result",
    )
    suite.run_test(
        test_name="Rate limiter metrics",
        test_func=test_rate_limiter_metrics,
        test_summary="Tests Rate limiter metrics functionality",
        method_description="Calls AdaptiveRateLimiter and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Rate limiter reset",
        test_func=test_rate_limiter_reset,
        test_summary="Tests Rate limiter reset functionality",
        method_description="Calls AdaptiveRateLimiter and verifies the result",
        expected_outcome="Function correctly updates the target",
    )
    suite.run_test(
        test_name="Rate limiter per endpoint",
        test_func=test_rate_limiter_per_endpoint,
        test_summary="Tests Rate limiter per endpoint functionality",
        method_description="Calls AdaptiveRateLimiter and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )

    return suite.finish_suite()