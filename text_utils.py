"""Text utilities for Social Media Publisher.

Shared text processing functions used across the codebase.
"""

import re
from typing import Optional

# Common first/last names that need extra matching signals when searching LinkedIn
# These generic Western names could match many people, so require more context.
# Note: Ethnic surnames (Wang, Zhang, Kim, Patel, etc.) are NOT included because
# when combined with first names they're usually distinctive enough.
COMMON_FIRST_NAMES: frozenset[str] = frozenset(
    {
        "john",
        "james",
        "michael",
        "david",
        "robert",
        "mary",
        "jennifer",
        "sarah",
        # Common surnames that are also first names
        "smith",
        "johnson",
        "williams",
        "brown",
        "jones",
    }
)

# Common stopwords to remove from context keyword matching
CONTEXT_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "center",
        "centre",
        "department",
        "division",
        "school",
        "institute",
        "research",
        "university",
        "college",
    }
)


def is_common_name(first_name: str, last_name: str = "") -> bool:
    """Check if a name is common and needs extra matching signals.

    Args:
        first_name: Person's first name.
        last_name: Person's last name (optional).

    Returns:
        True if either name part is in the common names list.
    """
    first_lower = first_name.lower() if first_name else ""
    last_lower = last_name.lower() if last_name else ""
    return first_lower in COMMON_FIRST_NAMES or last_lower in COMMON_FIRST_NAMES


def build_context_keywords(
    company: Optional[str] = None,
    department: Optional[str] = None,
    position: Optional[str] = None,
    research_area: Optional[str] = None,
    industry: Optional[str] = None,
) -> set[str]:
    """Build context keywords for profile matching from available metadata.

    Args:
        company: Company/organization name.
        department: Department name.
        position: Job title/position.
        research_area: Research field (for academics).
        industry: Industry sector.

    Returns:
        Set of lowercase keywords for matching, with stopwords removed.
    """
    keywords: set[str] = set()

    if company:
        keywords.update(w.lower() for w in company.split() if len(w) > 2)
    if department:
        keywords.update(w.lower() for w in department.split() if len(w) > 3)
    if position:
        keywords.update(w.lower() for w in position.split() if len(w) > 4)
    if research_area:
        keywords.update(w.lower() for w in research_area.split() if len(w) > 3)
    if industry:
        keywords.update(w.lower() for w in industry.split() if len(w) > 3)

    # Remove common stopwords
    keywords -= CONTEXT_STOPWORDS

    return keywords


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
