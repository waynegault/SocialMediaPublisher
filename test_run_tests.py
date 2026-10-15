"""Tests for the repository test orchestrator."""

import contextlib
import io
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import run_tests
from test_framework import TestSuite, create_standard_test_runner


def _create_module_tests() -> bool:
    suite = TestSuite("Test Runner", "run_tests.py")
    suite.start_suite()

    def test_missing_entrypoint_fails():
        legacy_module = SimpleNamespace(_create_module_tests=lambda: True)
        with patch.object(
            run_tests.importlib, "import_module", return_value=legacy_module
        ):
            success, _ = run_tests._run_module_tests("legacy", "Legacy")
        assert success is False

    def test_missing_ruff_fails_lint_gate():
        with (
            patch.object(run_tests, "_ruff_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            assert run_tests._run_linter() is False

    def test_linter_timeout_fails_lint_gate():
        timeout = subprocess.TimeoutExpired(cmd="ruff", timeout=60)
        with (
            patch.object(run_tests, "_ruff_available", return_value=True),
            patch.object(run_tests.subprocess, "run", side_effect=timeout),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            assert run_tests._run_linter() is False

    def test_lint_failure_fails_process():
        with (
            patch.object(run_tests, "TEST_MODULES", []),
            patch.object(run_tests, "_run_linter", return_value=False),
            patch.object(run_tests.sys, "argv", ["run_tests.py"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            assert run_tests.main() == 1

    for name, test in [
        ("Missing test entrypoint fails", test_missing_entrypoint_fails),
        ("Missing Ruff fails lint gate", test_missing_ruff_fails_lint_gate),
        ("Linter timeout fails lint gate", test_linter_timeout_fails_lint_gate),
        ("Lint failure fails process", test_lint_failure_fails_process),
    ]:
        suite.run_test(
            test_name=name,
            test_func=test,
            test_summary=f"Verify {name.lower()}",
            expected_outcome="The quality gate reports failure",
        )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(_create_module_tests)
