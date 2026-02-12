"""
Unified Content Validation Module.

Provides centralized spam detection patterns and validation helpers
for LinkedIn content optimization and engagement filtering.

This module consolidates spam patterns that were previously duplicated
in linkedin_optimizer.py and linkedin_engagement.py.
"""

import re
from typing import List

# =============================================================================
# Spam Pattern Categories
# =============================================================================

# Patterns for post content optimization (LinkedIn algorithm may penalize these)
POST_SPAM_PATTERNS = [
    r"(?i)like if you agree",
    r"(?i)share this post",
    r"(?i)comment \d+ for",
    r"(?i)drop an emoji",
    r"(?i)follow me for",
    r"(?i)link in (bio|comments)",
    r"(?i)dm me for",
    r"(?i)🔥{3,}",  # Excessive fire emojis
    r"(?i)👇{3,}",  # Excessive pointing emojis
]

# Patterns for engagement/comment spam filtering
COMMENT_SPAM_PATTERNS = [
    r"check out my",
    r"visit my profile",
    r"click (the )?link",
    r"buy now",
    r"limited offer",
    r"free.*download",
    r"guaranteed results",
    r"make money",
    r"dm me",
    r"inbox me",
    r"check inbox",
    r"promo code",
    r"discount",
    r"subscribe to",
    r"follow me",
    r"like my",
]

# Combined patterns for general spam detection
ALL_SPAM_PATTERNS = POST_SPAM_PATTERNS + [
    p for p in COMMENT_SPAM_PATTERNS if p not in POST_SPAM_PATTERNS
]


# =============================================================================
# Validation Functions
# =============================================================================


def detect_post_spam(content: str) -> List[str]:
    """
    Detect spam patterns in post content.

    Args:
        content: Post content to analyze

    Returns:
        List of detected spam pattern descriptions
    """
    found = []
    for pattern in POST_SPAM_PATTERNS:
        if re.search(pattern, content):
            # Extract readable description from pattern
            clean_pattern = pattern.replace(r"(?i)", "").replace("\\", "")
            found.append(clean_pattern)
    return found


def detect_comment_spam(content: str) -> List[str]:
    """
    Detect spam patterns in comment content.

    Args:
        content: Comment content to analyze

    Returns:
        List of detected spam pattern descriptions
    """
    found = []
    content_lower = content.lower()
    for pattern in COMMENT_SPAM_PATTERNS:
        if re.search(pattern, content_lower, re.IGNORECASE):
            found.append(pattern.replace("\\", ""))
    return found


def is_spam_content(content: str) -> bool:
    """
    Check if content contains any spam patterns.

    Args:
        content: Content to check

    Returns:
        True if spam detected, False otherwise
    """
    content_lower = content.lower()
    for pattern in ALL_SPAM_PATTERNS:
        if re.search(pattern, content_lower, re.IGNORECASE):
            return True
    return False


def remove_spam_patterns(content: str) -> str:
    """
    Remove spam patterns from content.

    Args:
        content: Content to clean

    Returns:
        Content with spam patterns removed
    """
    cleaned = content
    for pattern in POST_SPAM_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    # Clean up extra whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.strip()


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# For linkedin_engagement.py compatibility
SPAM_PATTERNS = COMMENT_SPAM_PATTERNS

# Re-export for convenience
__all__ = [
    "POST_SPAM_PATTERNS",
    "COMMENT_SPAM_PATTERNS",
    "ALL_SPAM_PATTERNS",
    "SPAM_PATTERNS",
    "detect_post_spam",
    "detect_comment_spam",
    "is_spam_content",
    "remove_spam_patterns",
]


# =============================================================================
# Module Tests
# =============================================================================


def _create_module_tests() -> bool:
    """Create unit tests for content_validation module."""
    from test_framework import TestSuite

    suite = TestSuite("Content Validation", "content_validation.py")
    suite.start_suite()

    def test_detect_post_spam_clean():
        result = detect_post_spam("Exciting developments in quantum computing research.")
        assert result == []

    def test_detect_post_spam_positive():
        result = detect_post_spam("Follow me for more amazing content! #followback")
        assert len(result) > 0

    def test_detect_comment_spam_clean():
        result = detect_comment_spam("Great insights, thanks for sharing.")
        assert result == []

    def test_detect_comment_spam_positive():
        result = detect_comment_spam("Check out my profile! DM for collab opportunities")
        assert len(result) > 0

    def test_is_spam_content_false():
        assert is_spam_content("A well-written professional article.") is False

    def test_is_spam_content_true():
        assert is_spam_content("Follow me for more! Click the link in my bio!") is True

    def test_remove_spam_patterns():
        cleaned = remove_spam_patterns("Great article. Follow me for more content!")
        assert "follow me" not in cleaned.lower()

    def test_remove_spam_patterns_clean():
        text = "A professional engineering update about BASF."
        assert remove_spam_patterns(text) == text

    def test_spam_pattern_constants_exist():
        assert isinstance(POST_SPAM_PATTERNS, list)
        assert isinstance(COMMENT_SPAM_PATTERNS, list)
        assert len(POST_SPAM_PATTERNS) > 0
        assert len(COMMENT_SPAM_PATTERNS) > 0

    def test_all_spam_patterns_combined():
        assert isinstance(ALL_SPAM_PATTERNS, list)
        assert len(ALL_SPAM_PATTERNS) >= len(POST_SPAM_PATTERNS)

    suite.run_test(

        test_name="detect_post_spam - clean",

        test_func=test_detect_post_spam_clean,

        test_summary="detect_post_spam behavior with clean input",

        method_description="Testing detect_post_spam with clean input using equality assertions and membership verification",

        expected_outcome="detect_post_spam returns the expected value",

    )
    suite.run_test(
        test_name="detect_post_spam - spam",
        test_func=test_detect_post_spam_positive,
        test_summary="detect_post_spam behavior with spam input",
        method_description="Testing detect_post_spam with spam input using size validation",
        expected_outcome="Result has the expected size; Result falls within expected bounds",
    )
    suite.run_test(
        test_name="detect_comment_spam - clean",
        test_func=test_detect_comment_spam_clean,
        test_summary="detect_comment_spam behavior with clean input",
        method_description="Testing detect_comment_spam with clean input using equality assertions",
        expected_outcome="detect_comment_spam returns the expected value",
    )
    suite.run_test(
        test_name="detect_comment_spam - spam",
        test_func=test_detect_comment_spam_positive,
        test_summary="detect_comment_spam behavior with spam input",
        method_description="Testing detect_comment_spam with spam input using size validation",
        expected_outcome="Result has the expected size; Result falls within expected bounds",
    )
    suite.run_test(
        test_name="is_spam_content - false",
        test_func=test_is_spam_content_false,
        test_summary="is_spam_content behavior with false input",
        method_description="Testing is_spam_content with false input using boolean return verification",
        expected_outcome="Function returns False for false input",
    )
    suite.run_test(
        test_name="is_spam_content - true",
        test_func=test_is_spam_content_true,
        test_summary="is_spam_content behavior with true input",
        method_description="Testing is_spam_content with true input using boolean return verification and membership verification",
        expected_outcome="Function returns True for true input",
    )
    suite.run_test(
        test_name="remove_spam_patterns - spam",
        test_func=test_remove_spam_patterns,
        test_summary="remove_spam_patterns behavior with spam input",
        method_description="Testing remove_spam_patterns with spam input using membership verification",
        expected_outcome="Result contains expected elements",
    )
    suite.run_test(
        test_name="remove_spam_patterns - clean",
        test_func=test_remove_spam_patterns_clean,
        test_summary="remove_spam_patterns behavior with clean input",
        method_description="Testing remove_spam_patterns with clean input using equality assertions",
        expected_outcome="remove_spam_patterns returns the expected value",
    )
    suite.run_test(
        test_name="Pattern constants exist",
        test_func=test_spam_pattern_constants_exist,
        test_summary="Verify Pattern constants exist produces correct results",
        method_description="Testing Pattern constants exist using type checking and size validation",
        expected_outcome="Pattern constants exist returns the correct type; Result falls within expected bounds",
    )
    suite.run_test(
        test_name="ALL_SPAM_PATTERNS combined",
        test_func=test_all_spam_patterns_combined,
        test_summary="Verify ALL_SPAM_PATTERNS combined produces correct results",
        method_description="Testing ALL_SPAM_PATTERNS combined using type checking and size validation",
        expected_outcome="ALL_SPAM_PATTERNS combined returns the correct type; Result falls within expected bounds",
    )
    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
