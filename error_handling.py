#!/usr/bin/env python3

"""
Standardized Error Handling Framework.

Provides consistent error handling patterns with proper logging,
recovery strategies, and user-friendly messages.

Based on waynes_framework error_handling.py.
"""

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

# Type variables for decorators
P = ParamSpec("P")
R = TypeVar("R")


# === EXCEPTION HIERARCHY ===


class FrameworkError(Exception):
    """Base exception class for all framework errors."""

    pass


class RetryableError(FrameworkError):
    """Exception that indicates the operation can be retried."""

    def __init__(
        self, message: str = "Operation can be retried", **kwargs: Any
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after: Optional[float] = kwargs.get("retry_after")
        self.max_retries: Optional[int] = kwargs.get("max_retries")
        self.context: dict[str, Any] = kwargs.get("context", {})
        self.recovery_hint: Optional[str] = kwargs.get("recovery_hint")


class FatalError(FrameworkError):
    """Exception that indicates the operation should not be retried."""

    def __init__(self, message: str = "Fatal error occurred", **kwargs: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = kwargs.get("context", {})
        self.recovery_hint: Optional[str] = kwargs.get("recovery_hint")


class APIRateLimitError(RetryableError):
    """Exception for API rate limit errors."""

    def __init__(self, message: str = "API rate limit exceeded", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = kwargs.get("retry_after", 60)


class NetworkTimeoutError(RetryableError):
    """Exception for network timeout errors."""

    def __init__(
        self, message: str = "Network timeout occurred", **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.timeout_duration: Optional[float] = kwargs.get("timeout_duration")


class DatabaseConnectionError(RetryableError):
    """Exception for database connection errors."""

    def __init__(
        self, message: str = "Database connection failed", **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.connection_string: Optional[str] = kwargs.get("connection_string")
        self.error_code: str = kwargs.get("error_code", "DB_CONNECTION_FAILED")


class DataValidationError(FatalError):
    """Exception for data validation errors."""

    def __init__(self, message: str = "Data validation failed", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.validation_errors: list[str] = kwargs.get("validation_errors", [])


class ConfigurationError(FatalError):
    """Exception for configuration errors."""

    def __init__(self, message: str = "Configuration error occurred", **kwargs: Any):
        super().__init__(message, **kwargs)
        self.config_section: Optional[str] = kwargs.get("config_section")


class AuthenticationExpiredError(RetryableError):
    """Exception for expired authentication errors."""

    def __init__(
        self, message: str = "Authentication has expired", **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.expired_at: Optional[datetime] = kwargs.get("expired_at")


class ImageGenerationError(RetryableError):
    """Exception for image generation failures."""

    def __init__(self, message: str = "Image generation failed", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.story_id: Optional[int] = kwargs.get("story_id")


class PublishingError(RetryableError):
    """Exception for social media publishing failures."""

    def __init__(self, message: str = "Publishing failed", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.platform: str = kwargs.get("platform", "unknown")
        self.story_id: Optional[int] = kwargs.get("story_id")


# === CIRCUIT BREAKER ===


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, calls rejected
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: int = 60  # Seconds before attempting recovery
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: int = 30  # Request timeout in seconds


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation for fault tolerance.
    Opens the circuit after a threshold of failures and closes after a timeout.
    """

    def __init__(
        self, name: str = "default", config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()

    def _handle_success_locked(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        else:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0

    def _handle_failure_locked(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.success_count = 0

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if (
                    self.last_failure_time
                    and time.time() - self.last_failure_time
                    > self.config.recovery_timeout
                ):
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} is OPEN"
                    )

        try:
            result = func(*args, **kwargs)
            with self._lock:
                self._handle_success_locked()
            return result
        except Exception as e:
            with self._lock:
                self._handle_failure_locked()
            raise e

    def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
        }


# === RECOVERY CONTEXT ===


@dataclass
class RecoveryContext:
    """Context container shared across enhanced recovery attempts."""

    operation_name: str
    attempt_number: int = 1
    max_attempts: int = 3
    last_error: Optional[Exception] = None
    error_history: list[Exception] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    def add_error(self, error: Exception) -> None:
        self.last_error = error
        self.error_history.append(error)

    def should_retry(self) -> bool:
        return self.attempt_number < self.max_attempts

    def get_backoff_delay(
        self, base_delay: float = 1.0, max_delay: float = 60.0
    ) -> float:
        delay = min(base_delay * (2 ** max(self.attempt_number - 1, 0)), max_delay)
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter


# === DECORATORS ===


def with_enhanced_recovery(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that provides automatic retry with exponential backoff."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            operation_name = f"{func.__module__}.{func.__name__}"
            context = RecoveryContext(
                operation_name=operation_name,
                max_attempts=max_attempts,
            )

            last_exception: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                context.attempt_number = attempt
                try:
                    logger.debug(
                        "Attempting %s (%d/%d)", operation_name, attempt, max_attempts
                    )
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(
                            "✅ %s succeeded after %d attempts", operation_name, attempt
                        )
                    return result
                except Exception as exc:
                    last_exception = exc
                    context.add_error(exc)

                    if not isinstance(exc, retryable_exceptions):
                        logger.error(
                            "❌ Non-retryable error in %s: %s", operation_name, exc
                        )
                        raise

                    logger.warning(
                        "⚠️ %s failed (%d/%d): %s",
                        operation_name,
                        attempt,
                        max_attempts,
                        exc,
                    )

                    if not context.should_retry():
                        logger.error(
                            "❌ %s failed after %d attempts",
                            operation_name,
                            max_attempts,
                        )
                        raise

                    delay = context.get_backoff_delay(base_delay, max_delay)
                    logger.debug("Retrying %s in %.1fs", operation_name, delay)
                    time.sleep(delay)

            if last_exception is not None:
                raise last_exception
            raise RuntimeError(f"Unknown error in {operation_name}")

        return wrapper

    return decorator


def graceful_degradation(
    fallback_value: Any = None, fallback_func: Optional[Callable[..., Any]] = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for graceful degradation when service fails."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed: {e}, using fallback")
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                return fallback_value

        return wrapper

    return decorator


def timeout_protection(timeout: int = 30) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for timeout protection (uses threading on Windows)."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result_container: list[R] = []
            exception_container: list[Optional[Exception]] = [None]

            def target() -> None:
                try:
                    result_container.append(func(*args, **kwargs))
                except Exception as e:
                    exception_container[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                raise TimeoutError(
                    f"Function {func.__name__} timed out after {timeout} seconds"
                )
            if exception_container[0]:
                raise exception_container[0]
            return result_container[0]

        return wrapper

    return decorator


def safe_execute(
    default_return: Any = None,
    log_errors: bool = True,
    error_message: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to safely execute a function with error handling."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    msg = error_message or f"Error in {func.__name__}: {e}"
                    logger.warning(msg)
                return default_return

        return wrapper

    return decorator


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():
    """Create unit tests for error_handling module."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Error Handling Tests")

    def test_retryable_error_creation():
        err = RetryableError("Test error", retry_after=5.0)
        assert err.retry_after == 5.0
        assert "Test error" in str(err)

    def test_fatal_error_creation():
        err = FatalError("Fatal test", context={"key": "value"})
        assert err.context == {"key": "value"}
        assert "Fatal test" in str(err)

    def test_api_rate_limit_error():
        err = APIRateLimitError("Rate limited", retry_after=30)
        assert err.retry_after == 30

    def test_network_timeout_error():
        err = NetworkTimeoutError("Timeout", timeout_duration=10.0)
        assert err.timeout_duration == 10.0

    def test_circuit_breaker_closed_state():
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_opens_after_failures():
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        cb = CircuitBreaker(name="test", config=config)
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
            except ValueError:
                pass
        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_reset():
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(name="test", config=config)
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_stats():
        cb = CircuitBreaker(name="test_stats")
        stats = cb.get_stats()
        assert stats["name"] == "test_stats"
        assert "state" in stats

    def test_recovery_context():
        ctx = RecoveryContext(operation_name="test_op", max_attempts=3)
        assert ctx.should_retry() is True
        ctx.attempt_number = 3
        assert ctx.should_retry() is False

    def test_recovery_context_backoff():
        ctx = RecoveryContext(operation_name="test", attempt_number=1)
        delay = ctx.get_backoff_delay(base_delay=1.0)
        assert delay >= 1.0  # At least base delay

    def test_with_enhanced_recovery_decorator():
        call_count = 0

        @with_enhanced_recovery(max_attempts=3, base_delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "success"

        with suppress_logging():
            result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_graceful_degradation_decorator():
        @graceful_degradation(fallback_value="fallback")
        def failing_function() -> str:
            raise ValueError("Error")

        with suppress_logging():
            result = failing_function()
        assert result == "fallback"  # type: ignore[unreachable]

    def test_safe_execute_decorator():
        @safe_execute(default_return=None, log_errors=False)
        def failing_function():
            raise ValueError("Error")

        result = failing_function()
        assert result is None

    suite.add_test("RetryableError creation", test_retryable_error_creation)
    suite.add_test("FatalError creation", test_fatal_error_creation)
    suite.add_test("APIRateLimitError creation", test_api_rate_limit_error)
    suite.add_test("NetworkTimeoutError creation", test_network_timeout_error)
    suite.add_test("CircuitBreaker closed state", test_circuit_breaker_closed_state)
    suite.add_test(
        "CircuitBreaker opens after failures", test_circuit_breaker_opens_after_failures
    )
    suite.add_test("CircuitBreaker reset", test_circuit_breaker_reset)
    suite.add_test("CircuitBreaker stats", test_circuit_breaker_stats)
    suite.add_test("RecoveryContext basics", test_recovery_context)
    suite.add_test("RecoveryContext backoff", test_recovery_context_backoff)
    suite.add_test(
        "with_enhanced_recovery decorator", test_with_enhanced_recovery_decorator
    )
    suite.add_test(
        "graceful_degradation decorator", test_graceful_degradation_decorator
    )
    suite.add_test("safe_execute decorator", test_safe_execute_decorator)

    return suite
