#!/usr/bin/env python3
"""
Comprehensive Testing Infrastructure & Quality Assurance Framework

Standardized test execution platform providing systematic validation of the
Social Media Publisher system through comprehensive test suite orchestration,
intelligent quality assessment, and detailed performance analytics with
automated test discovery and professional reporting.

Test Execution Framework:
• Standardized test suite execution with comprehensive reporting and analytics
• Advanced test orchestration with dependency management and error handling
• Intelligent test discovery with automatic test registration and categorization
• Comprehensive assertion utilities with detailed validation and error reporting
• Integration with run_all_tests.py for automated testing workflows

Quality Assurance:
• Comprehensive validation utilities with business rule enforcement
• Advanced performance monitoring with timing analysis
• Intelligent test result aggregation with quality scoring
• Automated regression detection with failure pattern analysis

Usage:
    Tests in each module follow the pattern:
        def _test_feature():
            ...assert statements...

        def _create_module_tests() -> bool:
            suite = TestSuite("Suite Name", "module.py")
            suite.start_suite()
            suite.run_test(test_name=..., test_func=..., ...)
            return suite.finish_suite()
"""

import contextlib
import logging
import time
import traceback
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

__all__ = [
    "Colors",
    "Icons",
    "MagicMock",
    "MockLogger",
    "TestSuite",
    "create_standard_test_runner",
    "patch",
    "suppress_logging",
]


# =============================================================================
# ANSI Color Utilities
# =============================================================================

class Colors:
    """ANSI color codes for terminal output with formatting utilities."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
    RESET = "\033[0m"

    @staticmethod
    def colorize(text: str, color_code: str) -> str:
        """Apply color formatting to text."""
        return f"{color_code}{text}{Colors.END}"

    @staticmethod
    def green(text: str) -> str:
        return Colors.colorize(text, Colors.GREEN)

    @staticmethod
    def red(text: str) -> str:
        return Colors.colorize(text, Colors.RED)

    @staticmethod
    def yellow(text: str) -> str:
        return Colors.colorize(text, Colors.YELLOW)

    @staticmethod
    def blue(text: str) -> str:
        return Colors.colorize(text, Colors.BLUE)

    @staticmethod
    def cyan(text: str) -> str:
        return Colors.colorize(text, Colors.CYAN)

    @staticmethod
    def gray(text: str) -> str:
        return Colors.colorize(text, Colors.GRAY)

    @staticmethod
    def bold(text: str) -> str:
        return Colors.colorize(text, Colors.BOLD)


class Icons:
    """Consistent visual indicators for test output."""

    PASS = "✅"
    FAIL = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    GEAR = "⚙️"
    ROCKET = "🚀"
    CLOCK = "⏰"
    MAGNIFY = "🔍"


# =============================================================================
# Test Suite
# =============================================================================

class TestSuite:
    """Standardized test suite with consistent formatting and reporting.

    Usage:
        suite = TestSuite("Feature Tests", "module.py")
        suite.start_suite()
        suite.run_test(
            test_name="Feature X validation",
            test_func=_test_feature_x,
            test_summary="Verify feature X works correctly",
            functions_tested="feature_x(), helper_y()",
            method_description="Call feature_x with known inputs and assert outputs",
            expected_outcome="Returns correct values for all test cases",
        )
        return suite.finish_suite()
    """

    __test__ = False  # Prevent pytest from collecting this helper class

    def __init__(self, suite_name: str, module_name: str) -> None:
        self.suite_name = suite_name
        self.module_name = module_name
        self.start_time: float | None = None
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.warnings = 0
        self.test_results: list[dict[str, Any]] = []

    def start_suite(self) -> None:
        """Initialize the test suite with formatted header."""
        self.start_time = time.time()
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{Icons.ROCKET} Testing: {self.suite_name}{Colors.RESET}")
        print(f"{Colors.GRAY}Module: {self.module_name}{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")

    def run_test(
        self,
        test_name: str,
        test_func: Callable[[], Any],
        test_summary: str = "",
        functions_tested: str = "",
        method_description: str = "",
        expected_outcome: str = "",
    ) -> bool:
        """Run a single test with standardized output and error handling.

        Args:
            test_name: Name/title of the test
            test_func: Test function to execute
            test_summary: Summary of what is being tested
            functions_tested: Names of the functions being tested
            method_description: How the functions are being tested
            expected_outcome: Expected outcome if functions work correctly

        Returns:
            True if test passed, False if failed
        """
        self.tests_run += 1
        test_start = time.time()

        print(f"{Colors.BLUE}{Icons.GEAR} Test {self.tests_run}: {test_name}{Colors.RESET}")
        if test_summary:
            print(f"Test: {test_summary}")
        if functions_tested:
            print(f"Functions tested: {functions_tested}")
        if method_description:
            print(f"Method: {method_description}")
        if expected_outcome:
            print(f"Expected outcome: {expected_outcome}")

        try:
            test_func()
            duration = time.time() - test_start
            actual_outcome = "Test executed successfully with all assertions passing"
            print(f"Actual outcome: {actual_outcome}")
            print(f"Duration: {duration:.3f}s")
            print(f"Conclusion: {Colors.GREEN}{Icons.PASS} PASSED{Colors.RESET}")
            print()

            self.tests_passed += 1
            self.test_results.append({
                "name": test_name,
                "status": "PASSED",
                "duration": duration,
                "expected": expected_outcome,
                "outcome": actual_outcome,
            })
            return True

        except AssertionError as e:
            duration = time.time() - test_start
            actual_outcome = f"Assertion failed: {e!s}"
            print(f"Actual outcome: {actual_outcome}")
            traceback.print_exc()
            print(f"Duration: {duration:.3f}s")
            print(f"Conclusion: {Colors.RED}{Icons.FAIL} FAILED{Colors.RESET}")
            print()

            self.tests_failed += 1
            self.test_results.append({
                "name": test_name,
                "status": "FAILED",
                "duration": duration,
                "error": str(e),
                "expected": expected_outcome,
                "outcome": actual_outcome,
            })
            return False

        except Exception as e:
            duration = time.time() - test_start
            actual_outcome = f"Exception occurred: {type(e).__name__}: {e!s}"
            print(f"Actual outcome: {actual_outcome}")
            print(f"Duration: {duration:.3f}s")
            print(f"Conclusion: {Colors.RED}{Icons.FAIL} FAILED{Colors.RESET}")
            print()

            self.tests_failed += 1
            self.test_results.append({
                "name": test_name,
                "status": "ERROR",
                "duration": duration,
                "error": f"{type(e).__name__}: {e!s}",
                "expected": expected_outcome,
                "outcome": actual_outcome,
            })
            return False

    def add_warning(self, message: str) -> None:
        """Add a warning message to the test output."""
        self.warnings += 1
        print(f"  {Colors.YELLOW}{Icons.WARNING} WARNING: {message}{Colors.RESET}")

    def finish_suite(self) -> bool:
        """Complete the test suite and print summary. Returns True if all passed."""
        total_duration = time.time() - self.start_time if self.start_time else 0

        print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{Icons.MAGNIFY} Test Summary: {self.suite_name}{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")

        if self.tests_failed == 0:
            status_color = Colors.GREEN
            status_icon = Icons.PASS
            status_text = "ALL TESTS PASSED"
        else:
            status_color = Colors.RED
            status_icon = Icons.FAIL
            status_text = "SOME TESTS FAILED"

        print(f"{status_color}{Icons.CLOCK} Duration: {total_duration:.3f}s{Colors.RESET}")
        print(f"{status_color}{status_icon} Status: {status_text}{Colors.RESET}")
        print(f"{Colors.GREEN}{Icons.PASS} Passed: {self.tests_passed}{Colors.RESET}")
        print(f"{Colors.RED}{Icons.FAIL} Failed: {self.tests_failed}{Colors.RESET}")
        if self.warnings > 0:
            print(f"{Colors.YELLOW}{Icons.WARNING} Warnings: {self.warnings}{Colors.RESET}")

        failed_tests = [r for r in self.test_results if r["status"] in {"FAILED", "ERROR"}]
        if failed_tests:
            print(f"\n{Colors.YELLOW}{Icons.INFO} Failed Test Details:{Colors.RESET}")
            for test in failed_tests:
                print(f"  {Colors.RED}• {test['name']}{Colors.RESET}")
                if "error" in test:
                    print(f"    {Colors.GRAY}{test['error']}{Colors.RESET}")
                if test.get("expected"):
                    print(f"    {Colors.GRAY}Expected: {test['expected']}{Colors.RESET}")

        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")

        return self.tests_failed == 0


# =============================================================================
# Logging Utilities
# =============================================================================

@contextlib.contextmanager
def suppress_logging(level: int = logging.CRITICAL) -> Iterator[None]:
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
        del kwargs
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

    def has_message_containing(self, text: str, level: str | None = None) -> bool:
        if level:
            return any(
                text in m["message"] for m in self.messages if m["level"] == level
            )
        return any(text in m["message"] for m in self.messages)


def create_standard_test_runner(
    module_test_function: Callable[[], bool],
) -> Callable[[], bool]:
    """Create a standardized test runner that wraps a module's test function.

    Consolidates the ``_create_module_tests`` / ``run_comprehensive_tests``
    pattern so every module can expose a discoverable entry-point with a
    single line::

        run_comprehensive_tests = create_standard_test_runner(my_tests)

    The wrapper prints a minimal pass/fail summary that ``run_tests.py`` can
    parse, and returns the boolean result of *module_test_function*.
    """

    def run_comprehensive_tests() -> bool:
        try:
            result = module_test_function()
            if result:
                print("\u2705 Passed: 1")
                print("\u274c Failed: 0")
            else:
                print("\u2705 Passed: 0")
                print("\u274c Failed: 1")
            return result
        except Exception as e:
            print(f"\u274c Test execution failed: {e}")
            print("\u2705 Passed: 0")
            print("\u274c Failed: 1")
            return False

    return run_comprehensive_tests
