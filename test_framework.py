#!/usr/bin/env python3
"""Testing Framework and Utilities."""

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional


@dataclass
class TestResult:
    """Result of a single test execution."""

    name: str
    passed: bool
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    exception: Optional[Exception] = None


@dataclass
class SuiteResult:
    """Result of a test suite execution."""

    suite_name: str
    results: list[TestResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    total_duration_ms: float = 0.0

    def add_result(self, result: TestResult) -> None:
        self.results.append(result)
        self.passed += 1 if result.passed else 0
        self.failed += 0 if result.passed else 1
        self.total_duration_ms += result.duration_ms


class TestSuite:
    """Structured test suite with formatted output."""

    def __init__(self, name: str = "Test Suite"):
        self.name = name
        self.tests: list[tuple[str, Callable[[], None]]] = []
        self._setup: Optional[Callable[[], None]] = None
        self._teardown: Optional[Callable[[], None]] = None

    def add_test(self, name: str, test_func: Callable[[], None]) -> None:
        self.tests.append((name, test_func))

    def set_setup(self, setup_func: Callable[[], None]) -> None:
        self._setup = setup_func

    def set_teardown(self, teardown_func: Callable[[], None]) -> None:
        self._teardown = teardown_func

    def run(self, verbose: bool = True) -> SuiteResult:
        result = SuiteResult(suite_name=self.name)
        if verbose:
            print(f"\n{'=' * 60}\n  {self.name}\n{'=' * 60}\n")
        for test_name, test_func in self.tests:
            test_result = self._run_single_test(test_name, test_func, verbose)
            result.add_result(test_result)
        if verbose:
            self._print_summary(result)
        return result

    def _run_single_test(
        self, name: str, test_func: Callable[[], None], verbose: bool
    ) -> TestResult:
        start_time = time.perf_counter()
        try:
            if self._setup:
                self._setup()
            test_func()
            if self._teardown:
                self._teardown()
            duration_ms = (time.perf_counter() - start_time) * 1000
            if verbose:
                print(f"  ✅ {name} ({duration_ms:.1f}ms)")
            return TestResult(name=name, passed=True, duration_ms=duration_ms)
        except (AssertionError, Exception) as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_msg = str(e) if str(e) else f"{type(e).__name__}"
            if verbose:
                print(f"  ❌ {name} ({duration_ms:.1f}ms)\n     └─ {error_msg}")
            return TestResult(
                name=name,
                passed=False,
                duration_ms=duration_ms,
                error_message=error_msg,
                exception=e,
            )

    def _print_summary(self, result: SuiteResult) -> None:
        total = result.passed + result.failed
        status = "✅ PASSED" if result.failed == 0 else "❌ FAILED"
        print(f"\n{'-' * 60}\n  {status}: {result.passed}/{total} tests\n{'=' * 60}\n")


@contextlib.contextmanager
def suppress_logging(level: int = logging.CRITICAL) -> Generator[None, None, None]:
    """Context manager to suppress logging during tests."""
    root_logger = logging.getLogger()
    original_level = root_logger.level
    try:
        root_logger.setLevel(level + 1)
        yield
    finally:
        root_logger.setLevel(original_level)


class MockLogger:
    """Mock logger for capturing log messages in tests."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def _log(self, level: str, msg: str, *args: Any, **kwargs: Any) -> None:
        del kwargs  # Unused but accepted for compatibility
        formatted = msg % args if args else msg
        self.messages.append({"level": level, "message": formatted})

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log("debug", msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log("info", msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log("warning", msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log("error", msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log("critical", msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log("error", msg, *args, **kwargs)

    def clear(self) -> None:
        self.messages.clear()

    def has_message_containing(self, text: str, level: Optional[str] = None) -> bool:
        if level:
            return any(
                text in m["message"] for m in self.messages if m["level"] == level
            )
        return any(text in m["message"] for m in self.messages)


def run_all_suites(
    suites: list[TestSuite], verbose: bool = True
) -> tuple[bool, list[tuple[str, str]]]:
    """Run multiple test suites and return (success, failed_tests).

    Returns:
        Tuple of (all_passed, list of (suite_name, test_name) for failures)
    """
    all_passed = True
    total_passed = total_failed = 0
    failed_tests: list[tuple[str, str]] = []

    for suite in suites:
        result = suite.run(verbose=verbose)
        total_passed += result.passed
        total_failed += result.failed
        if result.failed > 0:
            all_passed = False
            # Collect failed test names
            for test_result in result.results:
                if not test_result.passed:
                    failed_tests.append((result.suite_name, test_result.name))

    if verbose and len(suites) > 1:
        print(
            f"\n{'=' * 60}\n  OVERALL: {total_passed}/{total_passed + total_failed}\n{'=' * 60}\n"
        )
    return all_passed, failed_tests
