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
