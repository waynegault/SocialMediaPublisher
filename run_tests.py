#!/usr/bin/env python3
"""Test runner for all module tests.

This is the ONLY test runner for the project. All unit tests are defined
within their respective module files using _create_module_tests() functions.
This runner discovers and executes all module tests.
"""

import asyncio
import atexit
import io
import logging
import os
import sys
import warnings
from test_framework import run_all_suites, suppress_logging

# Suppress asyncio warnings about pending tasks (nodriver/websockets cleanup issue)
warnings.filterwarnings("ignore", message=".*Task was destroyed.*")
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited.*")


def _suppress_unraisable_hook(unraisable):
    """Suppress nodriver/websockets cleanup errors at exit."""
    # These are expected during process exit when nodriver cleanup races with gc
    err_str = str(unraisable.exc_value) if unraisable.exc_value else ""
    if any(x in err_str for x in ["closed pipe", "Event loop is closed"]):
        return  # Suppress these known harmless errors
    # For other unraisable exceptions, use default behavior
    sys.__unraisablehook__(unraisable)


# Install custom hook to suppress known harmless cleanup errors
sys.unraisablehook = _suppress_unraisable_hook


def _cleanup_asyncio():
    """Clean up asyncio resources at exit to prevent warnings."""
    try:
        # Close any shared browser sessions
        from linkedin_profile_lookup import LinkedInCompanyLookup

        LinkedInCompanyLookup.close_shared_browser()
    except Exception:
        pass


# Register cleanup to run at exit
atexit.register(_cleanup_asyncio)


# List of modules that have _create_module_tests() functions
TEST_MODULES = [
    ("config", "Config"),
    ("domain_credibility", "Domain Credibility"),
    ("url_utils", "URL Utilities"),
    ("error_handling", "Error Handling"),
    ("rate_limiter", "Rate Limiter"),
    ("database", "Database"),
    ("scheduler", "Scheduler"),
    ("verifier", "Verifier"),
    ("image_generator", "Image Generator"),
    ("linkedin_publisher", "LinkedIn Publisher"),
    ("linkedin_profile_lookup", "LinkedIn Profile Lookup"),
    ("searcher", "Searcher"),
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
]


def main_with_failures() -> tuple[int, list[tuple[str, str]]]:
    """Collect and run all module tests.

    Returns:
        Tuple of (exit_code, list of (suite_name, test_name) for failures)
    """
    # Note: venv check is handled by config.py at import time
    suites = []

    # Suppress logging during test imports and execution (up to ERROR level)
    with suppress_logging(logging.ERROR):
        # Import and collect test suites from each module
        for module_name, display_name in TEST_MODULES:
            try:
                module = __import__(module_name)
                if hasattr(module, "_create_module_tests"):
                    suites.append(module._create_module_tests())
                else:
                    print(f"Warning: {module_name} has no _create_module_tests()")
            except Exception as e:
                print(f"Failed to load {display_name} tests: {e}")

        if not suites:
            print("No test suites found!")
            return 1, []

        print(f"\nRunning {len(suites)} test suites...\n")
        success, failed_tests = run_all_suites(suites, verbose=True)

    # Browser cleanup is handled by atexit handler
    return (0, []) if success else (1, failed_tests)


def main() -> int:
    """Entry point that returns exit code only."""
    exit_code, _ = main_with_failures()

    # Suppress stderr to hide nodriver/websockets cleanup warnings during gc
    # These are harmless and happen after all tests complete successfully
    sys.stderr = io.StringIO()

    return exit_code


if __name__ == "__main__":
    result = main()
    # os._exit bypasses normal cleanup and avoids asyncio task warnings
    # All tests have completed and results have been reported
    os._exit(result)
