#!/usr/bin/env python3
"""Test runner for all module tests."""

import sys
from test_framework import run_all_suites


def main():
    """Collect and run all module tests."""
    suites = []

    # Import and collect test suites from each module
    try:
        from error_handling import _create_module_tests as error_tests

        suites.append(error_tests())
    except Exception as e:
        print(f"Failed to load error_handling tests: {e}")

    try:
        from rate_limiter import _create_module_tests as rate_tests

        suites.append(rate_tests())
    except Exception as e:
        print(f"Failed to load rate_limiter tests: {e}")

    try:
        from database import _create_module_tests as db_tests

        suites.append(db_tests())
    except Exception as e:
        print(f"Failed to load database tests: {e}")

    try:
        from scheduler import _create_module_tests as scheduler_tests

        suites.append(scheduler_tests())
    except Exception as e:
        print(f"Failed to load scheduler tests: {e}")

    try:
        from verifier import _create_module_tests as verifier_tests

        suites.append(verifier_tests())
    except Exception as e:
        print(f"Failed to load verifier tests: {e}")

    try:
        from image_generator import _create_module_tests as image_tests

        suites.append(image_tests())
    except Exception as e:
        print(f"Failed to load image_generator tests: {e}")

    try:
        from linkedin_publisher import _create_module_tests as linkedin_tests

        suites.append(linkedin_tests())
    except Exception as e:
        print(f"Failed to load linkedin_publisher tests: {e}")

    try:
        from searcher import _create_module_tests as searcher_tests

        suites.append(searcher_tests())
    except Exception as e:
        print(f"Failed to load searcher tests: {e}")

    if not suites:
        print("No test suites found!")
        return 1

    print(f"\nRunning {len(suites)} test suites...\n")
    success = run_all_suites(suites, verbose=True)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

