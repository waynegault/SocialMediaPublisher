"""Property-based testing using Hypothesis.

This module provides property-based tests to find edge cases that
unit tests might miss. Uses the Hypothesis library for generating
test data automatically.

Features:
- URL validation edge cases
- JSON parsing robustness
- Date parsing edge cases
- Similarity calculation properties
- Text processing properties

Example:
    Run with: python property_tests.py
    Or: pytest property_tests.py -v
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from test_framework import TestSuite


# =============================================================================
# Strategy Definitions (for when Hypothesis is available)
# =============================================================================


@dataclass
class PropertyTestResult:
    """Result of a property test run."""

    name: str
    passed: bool
    examples_tested: int
    failing_example: Any | None = None
    error: str | None = None


# =============================================================================
# Test Properties (Pure Python fallback)
# =============================================================================


def is_valid_url(url: str) -> bool:
    """Check if a URL is valid (simplified)."""
    if not url:
        return False
    pattern = r"^https?://[^\s<>\"{}|\\^`\[\]]+$"
    return bool(re.match(pattern, url))


def normalize_url(url: str) -> str:
    """Normalize a URL for consistency."""
    url = url.strip()
    url = url.rstrip("/")
    return url


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate word-based similarity between two texts."""
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def parse_json_safely(text: str) -> tuple[bool, Any]:
    """Parse JSON safely, returning (success, result)."""
    try:
        result = json.loads(text)
        return True, result
    except (json.JSONDecodeError, ValueError):
        return False, None


def extract_date(text: str) -> datetime | None:
    """Try to extract a date from text."""
    patterns = [
        r"(\d{4})-(\d{2})-(\d{2})",  # ISO format
        r"(\d{2})/(\d{2})/(\d{4})",  # US format
        r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})",
    ]

    for pattern in patterns[:2]:  # Only ISO and US format for simplicity
        match = re.search(pattern, text)
        if match:
            try:
                if "-" in pattern:
                    return datetime(
                        int(match.group(1)),
                        int(match.group(2)),
                        int(match.group(3)),
                        tzinfo=timezone.utc,
                    )
                else:
                    return datetime(
                        int(match.group(3)),
                        int(match.group(1)),
                        int(match.group(2)),
                        tzinfo=timezone.utc,
                    )
            except ValueError:
                continue
    return None


def sanitize_text(text: str) -> str:
    """Sanitize text for safe display."""
    # Remove control characters
    result = "".join(c for c in text if ord(c) >= 32 or c in "\n\t")
    return result.strip()


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    if max_length < 4:
        return text[:max_length]
    return text[: max_length - 3] + "..."


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Replace multiple spaces with single space
    result = re.sub(r"\s+", " ", text)
    return result.strip()


# =============================================================================
# Property Test Functions
# =============================================================================


def test_similarity_identity(n_examples: int = 100) -> PropertyTestResult:
    """Property: similarity(x, x) == 1.0 for non-empty x."""
    import random
    import string

    failures = []

    for _ in range(n_examples):
        # Generate random text
        length = random.randint(1, 100)
        text = "".join(random.choices(string.ascii_letters + " ", k=length))
        text = " ".join(text.split())  # Normalize whitespace

        if text.strip():
            similarity = calculate_similarity(text, text)
            if similarity != 1.0:
                failures.append((text, similarity))

    if failures:
        return PropertyTestResult(
            name="similarity_identity",
            passed=False,
            examples_tested=n_examples,
            failing_example=failures[0],
            error=f"Expected 1.0, got {failures[0][1]}",
        )

    return PropertyTestResult(
        name="similarity_identity",
        passed=True,
        examples_tested=n_examples,
    )


def test_similarity_symmetry(n_examples: int = 100) -> PropertyTestResult:
    """Property: similarity(a, b) == similarity(b, a)."""
    import random
    import string

    failures = []

    for _ in range(n_examples):
        text1 = "".join(
            random.choices(string.ascii_letters + " ", k=random.randint(1, 50))
        )
        text2 = "".join(
            random.choices(string.ascii_letters + " ", k=random.randint(1, 50))
        )

        sim1 = calculate_similarity(text1, text2)
        sim2 = calculate_similarity(text2, text1)

        if abs(sim1 - sim2) > 0.0001:
            failures.append((text1, text2, sim1, sim2))

    if failures:
        return PropertyTestResult(
            name="similarity_symmetry",
            passed=False,
            examples_tested=n_examples,
            failing_example=failures[0],
            error=f"sim(a,b)={failures[0][2]}, sim(b,a)={failures[0][3]}",
        )

    return PropertyTestResult(
        name="similarity_symmetry",
        passed=True,
        examples_tested=n_examples,
    )


def test_similarity_bounds(n_examples: int = 100) -> PropertyTestResult:
    """Property: 0 <= similarity(a, b) <= 1."""
    import random
    import string

    failures = []

    for _ in range(n_examples):
        text1 = "".join(
            random.choices(string.ascii_letters + " ", k=random.randint(0, 50))
        )
        text2 = "".join(
            random.choices(string.ascii_letters + " ", k=random.randint(0, 50))
        )

        similarity = calculate_similarity(text1, text2)

        if not (0 <= similarity <= 1):
            failures.append((text1, text2, similarity))

    if failures:
        return PropertyTestResult(
            name="similarity_bounds",
            passed=False,
            examples_tested=n_examples,
            failing_example=failures[0],
            error=f"Similarity {failures[0][2]} out of bounds",
        )

    return PropertyTestResult(
        name="similarity_bounds",
        passed=True,
        examples_tested=n_examples,
    )


def test_json_roundtrip(n_examples: int = 100) -> PropertyTestResult:
    """Property: json.loads(json.dumps(x)) == x for valid x."""
    failures = []

    for _ in range(n_examples):
        # Generate random JSON-compatible data
        data = _generate_random_json_data()

        try:
            json_str = json.dumps(data)
            success, parsed = parse_json_safely(json_str)

            if not success or parsed != data:
                failures.append((data, parsed))
        except Exception as e:
            failures.append((data, str(e)))

    if failures:
        return PropertyTestResult(
            name="json_roundtrip",
            passed=False,
            examples_tested=n_examples,
            failing_example=failures[0],
            error="Roundtrip failed",
        )

    return PropertyTestResult(
        name="json_roundtrip",
        passed=True,
        examples_tested=n_examples,
    )


def _generate_random_json_data(depth: int = 0) -> Any:
    """Generate random JSON-compatible data."""
    import random
    import string

    if depth > 3:
        return random.choice([None, True, False, random.randint(-1000, 1000)])

    choice = random.randint(0, 6)

    if choice == 0:
        return None
    elif choice == 1:
        return random.choice([True, False])
    elif choice == 2:
        return random.randint(-10000, 10000)
    elif choice == 3:
        return random.random() * 1000 - 500
    elif choice == 4:
        length = random.randint(0, 20)
        return "".join(random.choices(string.ascii_letters + " ", k=length))
    elif choice == 5:
        length = random.randint(0, 5)
        return [_generate_random_json_data(depth + 1) for _ in range(length)]
    else:
        count = random.randint(0, 5)
        keys = [
            "".join(random.choices(string.ascii_lowercase, k=5)) for _ in range(count)
        ]
        return {k: _generate_random_json_data(depth + 1) for k in keys}


def test_truncate_length(n_examples: int = 100) -> PropertyTestResult:
    """Property: len(truncate(text, n)) <= n."""
    import random
    import string

    failures = []

    for _ in range(n_examples):
        text = "".join(random.choices(string.printable, k=random.randint(0, 200)))
        max_len = random.randint(1, 100)

        result = truncate_text(text, max_len)

        if len(result) > max_len:
            failures.append((text, max_len, result, len(result)))

    if failures:
        return PropertyTestResult(
            name="truncate_length",
            passed=False,
            examples_tested=n_examples,
            failing_example=failures[0],
            error=f"Result length {failures[0][3]} > max {failures[0][1]}",
        )

    return PropertyTestResult(
        name="truncate_length",
        passed=True,
        examples_tested=n_examples,
    )


def test_truncate_preserves_short(n_examples: int = 100) -> PropertyTestResult:
    """Property: truncate(text, n) == text when len(text) <= n."""
    import random
    import string

    failures = []

    for _ in range(n_examples):
        length = random.randint(0, 50)
        text = "".join(random.choices(string.ascii_letters, k=length))
        max_len = length + random.randint(0, 50)  # Always >= len(text)

        result = truncate_text(text, max_len)

        if result != text:
            failures.append((text, max_len, result))

    if failures:
        return PropertyTestResult(
            name="truncate_preserves_short",
            passed=False,
            examples_tested=n_examples,
            failing_example=failures[0],
            error="Short text was modified",
        )

    return PropertyTestResult(
        name="truncate_preserves_short",
        passed=True,
        examples_tested=n_examples,
    )


def test_normalize_whitespace_idempotent(n_examples: int = 100) -> PropertyTestResult:
    """Property: normalize(normalize(text)) == normalize(text)."""
    import random
    import string

    failures = []

    for _ in range(n_examples):
        # Generate text with random whitespace
        chars = string.ascii_letters + "     \t\n"
        text = "".join(random.choices(chars, k=random.randint(0, 100)))

        normalized1 = normalize_whitespace(text)
        normalized2 = normalize_whitespace(normalized1)

        if normalized1 != normalized2:
            failures.append((text, normalized1, normalized2))

    if failures:
        return PropertyTestResult(
            name="normalize_whitespace_idempotent",
            passed=False,
            examples_tested=n_examples,
            failing_example=failures[0],
            error="Normalization not idempotent",
        )

    return PropertyTestResult(
        name="normalize_whitespace_idempotent",
        passed=True,
        examples_tested=n_examples,
    )


def test_sanitize_removes_control(n_examples: int = 100) -> PropertyTestResult:
    """Property: sanitize removes all control characters."""
    import random

    failures = []

    for _ in range(n_examples):
        # Generate text with some control characters
        chars = [chr(i) for i in range(128)]  # ASCII range
        text = "".join(random.choices(chars, k=random.randint(0, 100)))

        result = sanitize_text(text)

        # Check no control chars except \n, \t
        for c in result:
            if ord(c) < 32 and c not in "\n\t":
                failures.append((text, result, c, ord(c)))
                break

    if failures:
        return PropertyTestResult(
            name="sanitize_removes_control",
            passed=False,
            examples_tested=n_examples,
            failing_example=failures[0],
            error=f"Control char {failures[0][3]} not removed",
        )

    return PropertyTestResult(
        name="sanitize_removes_control",
        passed=True,
        examples_tested=n_examples,
    )


def test_url_normalize_idempotent(n_examples: int = 100) -> PropertyTestResult:
    """Property: normalize_url(normalize_url(url)) == normalize_url(url)."""
    import random
    import string

    failures = []

    for _ in range(n_examples):
        # Generate URL-like strings
        protocol = random.choice(["http://", "https://"])
        domain = "".join(
            random.choices(string.ascii_lowercase, k=random.randint(3, 15))
        )
        path = "/" + "/".join(
            "".join(random.choices(string.ascii_lowercase, k=random.randint(1, 10)))
            for _ in range(random.randint(0, 3))
        )
        trailing = random.choice(["", "/", "//", "   "])

        url = protocol + domain + ".com" + path + trailing

        normalized1 = normalize_url(url)
        normalized2 = normalize_url(normalized1)

        if normalized1 != normalized2:
            failures.append((url, normalized1, normalized2))

    if failures:
        return PropertyTestResult(
            name="url_normalize_idempotent",
            passed=False,
            examples_tested=n_examples,
            failing_example=failures[0],
            error="URL normalization not idempotent",
        )

    return PropertyTestResult(
        name="url_normalize_idempotent",
        passed=True,
        examples_tested=n_examples,
    )


def test_date_extraction_valid_iso(n_examples: int = 100) -> PropertyTestResult:
    """Property: ISO dates can be extracted correctly."""
    import random

    failures = []

    for _ in range(n_examples):
        year = random.randint(1900, 2100)
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # Safe for all months

        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        text = f"Published on {date_str} by author"

        result = extract_date(text)

        if result is None:
            failures.append((date_str, text, None))
        elif (result.year, result.month, result.day) != (year, month, day):
            failures.append((date_str, text, result))

    if failures:
        return PropertyTestResult(
            name="date_extraction_valid_iso",
            passed=False,
            examples_tested=n_examples,
            failing_example=failures[0],
            error="ISO date not extracted correctly",
        )

    return PropertyTestResult(
        name="date_extraction_valid_iso",
        passed=True,
        examples_tested=n_examples,
    )


# =============================================================================
# Test Runner
# =============================================================================


ALL_PROPERTY_TESTS = [
    test_similarity_identity,
    test_similarity_symmetry,
    test_similarity_bounds,
    test_json_roundtrip,
    test_truncate_length,
    test_truncate_preserves_short,
    test_normalize_whitespace_idempotent,
    test_sanitize_removes_control,
    test_url_normalize_idempotent,
    test_date_extraction_valid_iso,
]


def run_all_property_tests(n_examples: int = 50) -> list[PropertyTestResult]:
    """Run all property tests."""
    results = []

    for test_func in ALL_PROPERTY_TESTS:
        try:
            result = test_func(n_examples)
            results.append(result)
        except Exception as e:
            results.append(
                PropertyTestResult(
                    name=test_func.__name__,
                    passed=False,
                    examples_tested=0,
                    error=str(e),
                )
            )

    return results


def format_results(results: list[PropertyTestResult]) -> str:
    """Format test results for display."""
    lines = [
        "Property-Based Test Results",
        "=" * 50,
    ]

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for result in results:
        status = "✅" if result.passed else "❌"
        line = f"  {status} {result.name} ({result.examples_tested} examples)"
        if result.error:
            line += f"\n       Error: {result.error}"
        lines.append(line)

    lines.extend(
        [
            "",
            "-" * 50,
            f"  {'✅' if passed == total else '❌'} PASSED: {passed}/{total} tests",
            "=" * 50,
        ]
    )

    return "\n".join(lines)


# =============================================================================
# Unit Tests (TestSuite integration)
# =============================================================================


def _create_module_tests() -> "TestSuite":
    """Create unit tests for this module."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_framework import TestSuite

    suite = TestSuite("Property Tests")

    def test_similarity_identity_property():
        result = test_similarity_identity(20)
        assert result.passed, result.error

    def test_similarity_symmetry_property():
        result = test_similarity_symmetry(20)
        assert result.passed, result.error

    def test_similarity_bounds_property():
        result = test_similarity_bounds(20)
        assert result.passed, result.error

    def test_json_roundtrip_property():
        result = test_json_roundtrip(20)
        assert result.passed, result.error

    def test_truncate_length_property():
        result = test_truncate_length(20)
        assert result.passed, result.error

    def test_truncate_preserves_short_property():
        result = test_truncate_preserves_short(20)
        assert result.passed, result.error

    def test_normalize_idempotent_property():
        result = test_normalize_whitespace_idempotent(20)
        assert result.passed, result.error

    def test_sanitize_control_property():
        result = test_sanitize_removes_control(20)
        assert result.passed, result.error

    def test_url_normalize_property():
        result = test_url_normalize_idempotent(20)
        assert result.passed, result.error

    def test_date_extraction_property():
        result = test_date_extraction_valid_iso(20)
        assert result.passed, result.error

    def test_run_all():
        results = run_all_property_tests(10)
        assert len(results) == 10
        passed = sum(1 for r in results if r.passed)
        assert passed == 10

    def test_format_results():
        results = [
            PropertyTestResult("test1", True, 50),
            PropertyTestResult("test2", False, 50, error="Test error"),
        ]
        formatted = format_results(results)
        assert "test1" in formatted
        assert "test2" in formatted
        assert "1/2" in formatted

    def test_calculate_similarity():
        assert calculate_similarity("hello world", "hello world") == 1.0
        assert calculate_similarity("hello", "goodbye") == 0.0
        assert calculate_similarity("", "") == 0.0

    def test_truncate_text():
        assert truncate_text("hello", 10) == "hello"
        assert truncate_text("hello world", 5) == "he..."
        assert len(truncate_text("hello world", 8)) <= 8

    def test_normalize_whitespace():
        assert normalize_whitespace("  hello   world  ") == "hello world"
        assert normalize_whitespace("a\n\nb") == "a b"

    suite.add_test("Similarity identity property", test_similarity_identity_property)
    suite.add_test("Similarity symmetry property", test_similarity_symmetry_property)
    suite.add_test("Similarity bounds property", test_similarity_bounds_property)
    suite.add_test("JSON roundtrip property", test_json_roundtrip_property)
    suite.add_test("Truncate length property", test_truncate_length_property)
    suite.add_test(
        "Truncate preserves short property", test_truncate_preserves_short_property
    )
    suite.add_test("Normalize idempotent property", test_normalize_idempotent_property)
    suite.add_test("Sanitize control property", test_sanitize_control_property)
    suite.add_test("URL normalize property", test_url_normalize_property)
    suite.add_test("Date extraction property", test_date_extraction_property)
    suite.add_test("Run all property tests", test_run_all)
    suite.add_test("Format results", test_format_results)
    suite.add_test("Calculate similarity", test_calculate_similarity)
    suite.add_test("Truncate text", test_truncate_text)
    suite.add_test("Normalize whitespace", test_normalize_whitespace)

    return suite


if __name__ == "__main__":
    # Run as standalone property test suite
    print("\n" + "=" * 60)
    print("  Running Property-Based Tests")
    print("=" * 60 + "\n")

    results = run_all_property_tests(n_examples=100)
    print(format_results(results))

    # Exit with appropriate code
    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)
