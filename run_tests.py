#!/usr/bin/env python3
"""
Comprehensive Test Orchestration & Quality Assurance Engine

Advanced test execution platform providing systematic validation of the entire
Social Media Publisher system through comprehensive test suite orchestration,
intelligent quality assessment, and detailed performance analytics with
automated test discovery and professional reporting for reliable system validation.

Usage:
    python run_tests.py                # Run all tests with detailed reporting
    python run_tests.py --integration  # Run integration tests with live API access
    python run_tests.py --skip-linter  # Skip linter checks

Modes:
    Default: Unit tests only, SKIP_LIVE_API_TESTS=true
    --integration: Enables live browser/API tests with real authenticated sessions
"""

import atexit
import importlib
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Final

from test_framework import Colors, Icons, suppress_logging

SEPARATOR_LINE: Final[str] = "=" * 70


def _cleanup_asyncio() -> None:
    """Close the shared browser if its module was loaded during the test run."""
    module = sys.modules.get("linkedin_profile_lookup")
    if module is not None:
        lookup = getattr(module, "LinkedInCompanyLookup", None)
        if lookup is not None:
            lookup.close_shared_browser()


atexit.register(_cleanup_asyncio)


# Module registry: (module_name, display_name)
TEST_MODULES: list[tuple[str, str]] = [
    ("test_run_tests", "Test Runner"),
    ("test_story_update_api", "Story Update API"),
    ("config", "Configuration"),
    ("domain_credibility", "Domain Credibility"),
    ("url_utils", "URL Utilities"),
    ("text_utils", "Text Utilities"),
    ("error_handling", "Error Handling"),
    ("rate_limiter", "Rate Limiter"),
    ("database", "Database"),
    ("scheduler", "Scheduler"),
    ("verifier", "Content Verifier"),
    ("image_generator", "Image Generator"),
    ("linkedin_publisher", "LinkedIn Publisher"),
    ("linkedin_profile_lookup", "LinkedIn Profile Lookup"),
    ("searcher", "Story Searcher"),
    ("company_mention_enricher", "Company Mention Enricher"),
    ("opportunity_messages", "Opportunity Messages"),
    ("validation_server", "Validation Server"),
    ("find_indirect_people", "Indirect People"),
    ("originality_checker", "Originality Checker"),
    ("source_verifier", "Source Verifier"),
    ("linkedin_optimizer", "LinkedIn Optimizer"),
    ("monitoring", "Monitoring"),
    ("trend_detector", "Trend Detector"),
    ("ab_testing", "A/B Testing"),
    ("cache", "Cache"),
    ("image_quality", "Image Quality"),
    ("image_style", "Image Style"),
    ("notifications", "Notifications"),
    ("property_tests", "Property Tests"),
    ("cli", "CLI"),
    ("integration_tests", "Integration Tests"),
    ("analytics_db", "Analytics Data Warehouse"),
    ("migrations", "Database Migrations"),
    ("events", "Event System"),
    ("di", "Dependency Injection"),
    ("async_utils", "Async Utilities"),
    ("dashboard", "Dashboard"),
    ("analytics_engine", "Analytics Engine"),
    ("api_client", "API Client"),
    ("intent_classifier", "Intent Classifier"),
    ("linkedin_engagement", "LinkedIn Engagement"),
    ("linkedin_networking", "LinkedIn Networking"),
    ("ner_engine", "NER Engine"),
    ("rag_engine", "RAG Engine"),
    ("content_validation", "Content Validation"),
    ("entity_constants", "Entity Constants"),
    ("organization_aliases", "Organization Aliases"),
    ("profile_matcher", "Profile Matcher"),
    ("models", "Models"),
    ("browser_backends", "Browser Backends"),
    ("web_server", "Web Server"),
    ("browser", "Browser"),
    ("publish_daemon", "Publish Daemon"),
    ("linkedin_rapidapi_client", "LinkedIn RapidAPI Client"),
    ("linkedin_voyager_client", "LinkedIn Voyager Client"),
    ("generate_image", "Generate Image"),
]


def _ruff_available() -> bool:
    """Return True if Ruff CLI is available."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _run_linter() -> bool:
    """Run Ruff linter and return True if no issues found."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{SEPARATOR_LINE}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}🔍 Running Linter (Ruff){Colors.RESET}")
    print(f"{Colors.CYAN}{SEPARATOR_LINE}{Colors.RESET}\n")

    if not _ruff_available():
        print(f"{Colors.RED}❌ Ruff is required but not available in this environment{Colors.RESET}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "."],
            check=False, capture_output=True, text=True,
            cwd=Path.cwd(), timeout=60,
        )
        if result.returncode == 0:
            print(f"{Colors.GREEN}✅ Linter: No issues found{Colors.RESET}")
            return True
        else:
            output = "\n".join(
                part for part in (result.stdout, result.stderr) if part
            )
            lines = [ln for ln in output.splitlines() if ln.strip()][-30:]
            for ln in lines:
                print(ln)
            print(f"\n{Colors.RED}❌ Linter: Issues found{Colors.RESET}")
            return False
    except subprocess.TimeoutExpired as e:
        print(f"{Colors.RED}❌ Linter timed out after {e.timeout} seconds{Colors.RESET}")
        return False


def _run_module_tests(module_name: str, display_name: str) -> tuple[bool, float]:
    """Import a module and run its test function.

    Requires the single supported ``run_comprehensive_tests`` entry point.

    Returns:
        Tuple of (success, duration_seconds)
    """
    start = time.time()
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "run_comprehensive_tests"):
            success = module.run_comprehensive_tests()
            return success, time.time() - start
        else:
            print(
                f"{Colors.RED}❌ {display_name}: "
                "required run_comprehensive_tests entry point not found"
                f"{Colors.RESET}"
            )
            return False, time.time() - start
    except Exception as e:  # noqa: BLE001 - isolate failures and continue other suites
        duration = time.time() - start
        traceback.print_exc()
        print(f"{Colors.RED}❌ {display_name}: Failed to load - {e}{Colors.RESET}")
        return False, duration


def main() -> int:
    """Run all tests and return exit code."""
    overall_start = time.time()
    skip_linter = "--skip-linter" in sys.argv
    integration_mode = "--integration" in sys.argv

    if integration_mode:
        os.environ["SKIP_LIVE_API_TESTS"] = "false"
    else:
        os.environ.setdefault("SKIP_LIVE_API_TESTS", "true")

    print(f"\n{Colors.BOLD}{Colors.CYAN}{SEPARATOR_LINE}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{Icons.ROCKET} Social Media Publisher - Test Suite{Colors.RESET}")
    print(f"{Colors.GRAY}Mode: {'Integration' if integration_mode else 'Unit Tests'}{Colors.RESET}")
    print(f"{Colors.CYAN}{SEPARATOR_LINE}{Colors.RESET}")

    # --- Linter ---
    linter_ok = True
    if not skip_linter:
        linter_ok = _run_linter()

    # --- Module Tests ---
    passed_modules = 0
    failed_modules = 0
    failed_names: list[str] = []

    original_sigint_handler = signal.getsignal(signal.SIGINT)
    try:
        with suppress_logging(logging.ERROR):
            for module_name, display_name in TEST_MODULES:
                success, _ = _run_module_tests(module_name, display_name)
                if success:
                    passed_modules += 1
                else:
                    failed_modules += 1
                    failed_names.append(display_name)
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)

    # Remove any rogue logging handlers added by imported modules
    # (e.g. publish_daemon.py's logging.basicConfig adds FileHandlers)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)

    # --- Final Summary ---
    total_duration = time.time() - overall_start
    total_modules = passed_modules + failed_modules

    print(f"\n{Colors.BOLD}{Colors.CYAN}{SEPARATOR_LINE}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{Icons.MAGNIFY} Final Test Report{Colors.RESET}")
    print(f"{Colors.CYAN}{SEPARATOR_LINE}{Colors.RESET}")
    print(f"{Icons.CLOCK} Total Duration: {total_duration:.1f}s")
    print(f"{Colors.GREEN}{Icons.PASS} Modules Passed: {passed_modules}/{total_modules}{Colors.RESET}")

    if failed_modules > 0:
        print(f"{Colors.RED}{Icons.FAIL} Modules Failed: {failed_modules}/{total_modules}{Colors.RESET}")
        for name in failed_names:
            print(f"  {Colors.RED}• {name}{Colors.RESET}")

    if not linter_ok:
        print(f"{Colors.YELLOW}⚠️ Linter issues detected{Colors.RESET}")

    if failed_modules == 0 and linter_ok:
        print(f"\n{Colors.GREEN}{Colors.BOLD}{Icons.PASS} ALL TESTS PASSED{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}{Icons.FAIL} QUALITY GATES FAILED{Colors.RESET}")

    print(f"{Colors.CYAN}{SEPARATOR_LINE}{Colors.RESET}\n")

    return 0 if failed_modules == 0 and linter_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
