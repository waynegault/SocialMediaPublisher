#!/usr/bin/env python3
"""
Unit Tests for Social Media Publisher.

Comprehensive tests for all modules using the test_framework.
Run with: python unit_tests.py
"""

import os
import tempfile

from test_framework import TestSuite, run_all_suites, suppress_logging, MockLogger


# ============================================================================
# Error Handling Tests
# ============================================================================
def create_error_handling_tests() -> TestSuite:
    """Create tests for error_handling module."""
    suite = TestSuite("Error Handling Tests")

    def test_retryable_error_creation():
        from error_handling import RetryableError

        err = RetryableError("Test error", retry_after=5.0)
        assert err.retry_after == 5.0
        assert "Test error" in str(err)

    def test_fatal_error_creation():
        from error_handling import FatalError

        err = FatalError("Fatal test", context={"key": "value"})
        assert err.context == {"key": "value"}
        assert "Fatal test" in str(err)

    def test_circuit_breaker_closed_state():
        from error_handling import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_opens_after_failures():
        from error_handling import CircuitBreaker, CircuitBreakerConfig, CircuitState

        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        cb = CircuitBreaker(name="test", config=config)
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
            except ValueError:
                pass
        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_reset():
        from error_handling import CircuitBreaker, CircuitBreakerConfig, CircuitState

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
        from error_handling import CircuitBreaker

        cb = CircuitBreaker(name="test_stats")
        stats = cb.get_stats()
        assert stats["name"] == "test_stats"
        assert "state" in stats

    def test_with_enhanced_recovery_decorator():
        from error_handling import with_enhanced_recovery

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

    suite.add_test("RetryableError creation", test_retryable_error_creation)
    suite.add_test("FatalError creation", test_fatal_error_creation)
    suite.add_test("CircuitBreaker closed state", test_circuit_breaker_closed_state)
    suite.add_test(
        "CircuitBreaker opens after failures", test_circuit_breaker_opens_after_failures
    )
    suite.add_test("CircuitBreaker reset", test_circuit_breaker_reset)
    suite.add_test("CircuitBreaker stats", test_circuit_breaker_stats)
    suite.add_test(
        "with_enhanced_recovery decorator", test_with_enhanced_recovery_decorator
    )

    return suite


# ============================================================================
# Rate Limiter Tests
# ============================================================================
def create_rate_limiter_tests() -> TestSuite:
    """Create tests for rate_limiter module."""
    suite = TestSuite("Rate Limiter Tests")

    def test_rate_limiter_creation():
        from rate_limiter import AdaptiveRateLimiter

        rl = AdaptiveRateLimiter(initial_fill_rate=2.0, capacity=10.0)
        assert rl.capacity == 10.0
        assert rl.fill_rate == 2.0

    def test_rate_limiter_wait():
        from rate_limiter import AdaptiveRateLimiter

        rl = AdaptiveRateLimiter(initial_fill_rate=10.0)
        wait_time = rl.wait(endpoint="test")
        assert wait_time >= 0

    def test_rate_limiter_429_backoff():
        from rate_limiter import AdaptiveRateLimiter

        rl = AdaptiveRateLimiter(initial_fill_rate=2.0, rate_limiter_429_backoff=0.5)
        rl.wait(endpoint="test")
        with suppress_logging():
            rl.on_429_error(endpoint="test")
        metrics = rl.get_metrics()
        assert metrics.error_429_count == 1
        assert metrics.rate_decreases == 1

    def test_rate_limiter_success_increase():
        from rate_limiter import AdaptiveRateLimiter

        rl = AdaptiveRateLimiter(success_threshold=2, rate_limiter_success_factor=1.5)
        rl.wait(endpoint="test")
        rl.on_success(endpoint="test")
        rl.on_success(endpoint="test")
        metrics = rl.get_metrics()
        assert metrics.rate_increases >= 1

    def test_rate_limiter_metrics():
        from rate_limiter import AdaptiveRateLimiter, RateLimiterMetrics

        rl = AdaptiveRateLimiter()
        rl.wait()
        metrics = rl.get_metrics()
        assert isinstance(metrics, RateLimiterMetrics)
        assert metrics.total_requests == 1

    def test_rate_limiter_reset():
        from rate_limiter import AdaptiveRateLimiter

        rl = AdaptiveRateLimiter()
        rl.wait()
        rl.reset()
        metrics = rl.get_metrics()
        assert metrics.total_requests == 0

    suite.add_test("Rate limiter creation", test_rate_limiter_creation)
    suite.add_test("Rate limiter wait", test_rate_limiter_wait)
    suite.add_test("Rate limiter 429 backoff", test_rate_limiter_429_backoff)
    suite.add_test("Rate limiter success increase", test_rate_limiter_success_increase)
    suite.add_test("Rate limiter metrics", test_rate_limiter_metrics)
    suite.add_test("Rate limiter reset", test_rate_limiter_reset)

    return suite


# ============================================================================
# Database Tests
# ============================================================================
def create_database_tests() -> TestSuite:
    """Create tests for database module."""
    suite = TestSuite("Database Tests")

    def test_story_dataclass():
        from database import Story

        story = Story(title="Test", summary="Summary", quality_score=8)
        assert story.title == "Test"
        assert story.quality_score == 8
        assert story.category == "Other"

    def test_story_to_dict():
        from database import Story

        story = Story(
            title="Test", summary="Summary", source_links=["http://example.com"]
        )
        d = story.to_dict()
        assert d["title"] == "Test"
        assert "http://example.com" in d["source_links"]

    def test_database_init():
        from database import Database

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            _db = Database(db_path)  # Constructor calls _init_db
            assert os.path.exists(db_path)
        finally:
            os.unlink(db_path)

    def test_database_add_story():
        from database import Database, Story

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            story = Story(title="Test Story", summary="Test summary", quality_score=7)
            story_id = db.add_story(story)
            assert story_id is not None
            assert story_id > 0
        finally:
            os.unlink(db_path)

    def test_database_get_story():
        from database import Database, Story

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            story = Story(title="Fetch Test", summary="Summary", quality_score=8)
            story_id = db.add_story(story)
            fetched = db.get_story(story_id)
            assert fetched is not None
            assert fetched.title == "Fetch Test"
        finally:
            os.unlink(db_path)

    def test_database_get_story_by_title():
        from database import Database, Story

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            story = Story(title="Unique Title", summary="Summary", quality_score=7)
            db.add_story(story)
            found = db.get_story_by_title("Unique Title")
            assert found is not None
            assert found.title == "Unique Title"
            not_found = db.get_story_by_title("Nonexistent")
            assert not_found is None
        finally:
            os.unlink(db_path)

    suite.add_test("Story dataclass", test_story_dataclass)
    suite.add_test("Story to_dict", test_story_to_dict)
    suite.add_test("Database init", test_database_init)
    suite.add_test("Database add story", test_database_add_story)
    suite.add_test("Database get story", test_database_get_story)
    suite.add_test("Database get story by title", test_database_get_story_by_title)

    return suite


# ============================================================================
# Config Tests
# ============================================================================
def create_config_tests() -> TestSuite:
    """Create tests for config module."""
    suite = TestSuite("Config Tests")

    def test_config_defaults():
        from config import Config

        assert Config.SUMMARY_WORD_COUNT >= 50
        assert Config.STORIES_PER_CYCLE >= 1
        assert Config.MAX_STORIES_PER_SEARCH >= 1

    def test_config_validate():
        from config import Config

        errors = Config.validate()
        assert isinstance(errors, list)

    def test_config_ensure_directories():
        from config import Config

        Config.ensure_directories()
        assert os.path.isdir(Config.IMAGE_DIR)

    suite.add_test("Config defaults", test_config_defaults)
    suite.add_test("Config validate", test_config_validate)
    suite.add_test("Config ensure directories", test_config_ensure_directories)

    return suite


# ============================================================================
# Test Framework Self-Tests
# ============================================================================
def create_test_framework_tests() -> TestSuite:
    """Create tests for the test framework itself."""
    suite = TestSuite("Test Framework Tests")

    def test_mock_logger():
        logger = MockLogger()
        logger.info("Test message")
        logger.warning("Warning message")
        assert len(logger.messages) == 2
        assert logger.has_message_containing("Test")
        assert logger.has_message_containing("Warning", level="warning")

    def test_mock_logger_clear():
        logger = MockLogger()
        logger.error("Error")
        logger.clear()
        assert len(logger.messages) == 0

    def test_suppress_logging():
        import logging

        with suppress_logging():
            logging.getLogger().info("This should be suppressed")
        # If we get here without error, the test passes

    suite.add_test("MockLogger basic", test_mock_logger)
    suite.add_test("MockLogger clear", test_mock_logger_clear)
    suite.add_test("suppress_logging context", test_suppress_logging)

    return suite


# ============================================================================
# Main Entry Point
# ============================================================================
def run_tests() -> bool:
    """Run all unit tests and return success status."""
    suites = [
        create_test_framework_tests(),
        create_error_handling_tests(),
        create_rate_limiter_tests(),
        create_config_tests(),
        create_database_tests(),
    ]
    return run_all_suites(suites, verbose=True)


if __name__ == "__main__":
    import sys

    success = run_tests()
    sys.exit(0 if success else 1)
