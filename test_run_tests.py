"""Tests for the repository test orchestrator."""

import contextlib
import io
import subprocess
import sys
from pathlib import Path
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
            patch.object(run_tests, "_is_repository_venv", return_value=True),
            patch.object(run_tests, "_run_linter", return_value=False),
            patch.object(run_tests.sys, "argv", ["run_tests.py"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            assert run_tests.main() == 1

    def test_repository_venv_detection():
        expected_venv = run_tests._repository_venv_path()
        with patch.object(run_tests.sys, "prefix", str(expected_venv)):
            assert run_tests._is_repository_venv()
        with patch.object(run_tests.sys, "prefix", str(expected_venv.parent)):
            assert not run_tests._is_repository_venv()

    def test_runner_rejects_non_repository_venv():
        output = io.StringIO()
        with (
            patch.object(run_tests, "_is_repository_venv", return_value=False),
            patch.object(run_tests, "_run_linter") as run_linter,
            contextlib.redirect_stdout(output),
        ):
            assert run_tests.main() == 2
        run_linter.assert_not_called()
        assert "repository virtual environment" in output.getvalue()

    def test_unit_suite_does_not_read_stdin():
        """A unit-mode test that waits on stdin would wedge the CI step for hours."""
        script = (
            "import linkedin_profile_lookup as m; "
            "raise SystemExit(0 if m.run_comprehensive_tests() else 1)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parent,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert completed.returncode == 0, completed.stdout[-2000:]

    def test_skipped_tests_are_reported_and_pass():
        """A gated live test must be named in the summary, not silently absent."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            suite = TestSuite("Skip probe", "probe.py")
            suite.start_suite()
            suite.skip("live probe", "needs --integration")
            result = suite.finish_suite()
        assert result is True
        assert "live probe" in output.getvalue()
        assert "needs --integration" in output.getvalue()

    for name, test in [
        ("Missing test entrypoint fails", test_missing_entrypoint_fails),
        ("Missing Ruff fails lint gate", test_missing_ruff_fails_lint_gate),
        ("Linter timeout fails lint gate", test_linter_timeout_fails_lint_gate),
        ("Repository venv detection", test_repository_venv_detection),
        ("Rejects non-repository venv", test_runner_rejects_non_repository_venv),
        ("Lint failure fails process", test_lint_failure_fails_process),
    ]:
        suite.run_test(
            test_name=name,
            test_func=test,
            test_summary=f"Verify {name.lower()}",
            expected_outcome="The quality gate reports failure",
        )

    suite.run_test(
        test_name="Unit suite does not read stdin",
        test_func=test_unit_suite_does_not_read_stdin,
        test_summary="Verify the unit suite completes with stdin closed",
        expected_outcome="The suite finishes without waiting for input",
    )

    suite.run_test(
        test_name="Skipped tests are reported",
        test_func=test_skipped_tests_are_reported_and_pass,
        test_summary="Verify a skipped test is named and does not fail the suite",
        expected_outcome="The suite passes and the skip is reported with its reason",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(_create_module_tests)
