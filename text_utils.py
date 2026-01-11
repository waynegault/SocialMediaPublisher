"""Text utilities for Social Media Publisher.

Shared text processing functions used across the codebase.
"""

import re


def strip_markdown_code_block(text: str) -> str:
    """Strip markdown code block delimiters from AI response text.

    Handles various formats:
    - ```json ... ```
    - ```python ... ```
    - ``` ... ```

    Args:
        text: Raw text that may contain markdown code block delimiters.

    Returns:
        Text with code block delimiters removed.
    """
    if not text:
        return text

    text = text.strip()

    if text.startswith("```"):
        # Remove opening delimiter (e.g., ```json, ```python, ```)
        text = re.sub(r"^```\w*\n?", "", text)
        # Remove closing delimiter
        text = re.sub(r"\n?```$", "", text)

    return text.strip()


def normalize_name(name: str) -> str:
    """Normalize a name for comparison.

    Handles:
    - Case normalization (lowercase)
    - Multiple spaces
    - Common prefixes (Dr., Prof., etc.)
    - Suffixes (Jr., Sr., PhD, etc.)

    Args:
        name: A person's name.

    Returns:
        Normalized name for comparison.
    """
    if not name:
        return ""

    name = name.lower().strip()

    # Remove common prefixes
    prefixes = [
        r"^dr\.?\s+",
        r"^prof\.?\s+",
        r"^professor\s+",
        r"^mr\.?\s+",
        r"^mrs\.?\s+",
        r"^ms\.?\s+",
        r"^miss\s+",
    ]
    for prefix in prefixes:
        name = re.sub(prefix, "", name, flags=re.IGNORECASE)

    # Remove common suffixes
    suffixes = [
        r",?\s+jr\.?$",
        r",?\s+sr\.?$",
        r",?\s+phd\.?$",
        r",?\s+md\.?$",
        r",?\s+esq\.?$",
        r",?\s+ii+$",
        r",?\s+iii+$",
    ]
    for suffix in suffixes:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE)

    # Normalize whitespace
    name = re.sub(r"\s+", " ", name)

    return name.strip()


def extract_first_last_name(full_name: str) -> tuple[str, str]:
    """Extract first and last name from a full name.

    Args:
        full_name: A person's full name.

    Returns:
        Tuple of (first_name, last_name). Last name may be empty for single names.
    """
    normalized = normalize_name(full_name)
    parts = normalized.split()

    if not parts:
        return "", ""
    elif len(parts) == 1:
        return parts[0], ""
    else:
        # First word is first name, last word is last name
        return parts[0], parts[-1]


def name_matches(name1: str, name2: str, strict: bool = False) -> bool:
    """Check if two names refer to the same person.

    Args:
        name1: First name to compare.
        name2: Second name to compare.
        strict: If True, require exact match. If False, allow partial matches.

    Returns:
        True if names match according to the matching rules.
    """
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)

    if not n1 or not n2:
        return False

    if strict:
        return n1 == n2

    # Check if one contains the other
    if n1 in n2 or n2 in n1:
        return True

    # Check first/last name match
    first1, last1 = extract_first_last_name(name1)
    first2, last2 = extract_first_last_name(name2)

    # Both first and last must match (if present)
    if last1 and last2:
        return first1 == first2 and last1 == last2

    # Single name - just check first name
    return first1 == first2


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to a maximum length with suffix.

    Args:
        text: Text to truncate.
        max_length: Maximum length including suffix.
        suffix: Suffix to append when truncating.

    Returns:
        Truncated text.
    """
    if not text or len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix
