#!/usr/bin/env python3
"""Test runner for all module tests.

This is the ONLY test runner for the project. All unit tests are defined
within their respective module files using _create_module_tests() functions.
This runner discovers and executes all module tests.
"""

import logging
import sys
from test_framework import run_all_suites, suppress_logging


# List of modules that have _create_module_tests() functions
TEST_MODULES = [
    ("config", "Config"),
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
    ("find_leadership", "Find Leadership"),
]


def main():
    """Collect and run all module tests."""
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
            return 1

        print(f"\nRunning {len(suites)} test suites...\n")
        success = run_all_suites(suites, verbose=True)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
