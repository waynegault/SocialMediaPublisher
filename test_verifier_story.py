#!/usr/bin/env python3
"""
Test Story Verification - Run tests specifically for the ContentVerifier.

This script tests the story verification functionality to ensure it:
1. Correctly parses AI verification responses
2. Builds verification prompts correctly
3. Handles approved/rejected/unclear responses
4. Retrieves verification statistics
"""

import sys
from verifier import _create_module_tests
from test_framework import run_all_suites

def main():
    """Run all verifier tests."""
    print("=" * 70)
    print("Testing Story Verification Functionality")
    print("=" * 70)
    print()
    
    # Get the verifier test suite
    suite = _create_module_tests()
    
    # Run the suite
    result = suite.run(verbose=True)
    
    # Print summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Suite: {result.suite_name}")
    print(f"Passed: {result.passed}")
    print(f"Failed: {result.failed}")
    print(f"Total Duration: {result.total_duration_ms:.2f}ms")
    print()
    
    if result.failed > 0:
        print("FAILED TESTS:")
        for test_result in result.results:
            if not test_result.passed:
                print(f"  ✗ {test_result.name}")
                if test_result.error_message:
                    print(f"    Error: {test_result.error_message}")
                if test_result.exception:
                    print(f"    Exception: {test_result.exception}")
        print()
        return 1
    else:
        print("✓ All tests passed!")
        print()
        return 0

if __name__ == "__main__":
    sys.exit(main())
