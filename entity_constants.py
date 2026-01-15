"""Entity validation constants for Social Media Publisher.

This module provides shared constants for entity (person, organization) validation
that are used across multiple modules without requiring optional dependencies like spaCy.

Constants:
- INVALID_ORG_NAMES: Terms that appear as "affiliations" but aren't real organizations
- INVALID_ORG_PATTERNS: Regex patterns that indicate headlines/descriptions, not org names
- INVALID_PERSON_NAMES: Generic placeholders that aren't actual person names
- VALID_SINGLE_WORD_ORGS: Single-word org abbreviations that ARE valid

Consumers:
- ner_engine.py
- linkedin_profile_lookup.py
- company_mention_enricher.py
"""

# =============================================================================
# Invalid Organization Names
# =============================================================================
# These are terms that appear as "affiliations" or "names" but aren't real entities.

INVALID_ORG_NAMES: set[str] = {
    # Generic terms / research topics (not organizations)
    "molecular",
    "chemistry",
    "physics",
    "biology",
    "research",
    "science",
    "engineering",
    "technology",
    "materials",
    "nanotechnology",
    "computational",
    "theoretical",
    "applied",
    "advanced",
    "fundamental",
    # Materials and topics (not organizations)
    "mxenes",
    "graphene",
    "nanoparticles",
    "nanomaterials",
    "quantum",
    "polymers",
    "catalysis",
    "electrochemistry",
    "spectroscopy",
    "synthesis",
    # Government entities (too generic)
    "government",
    "the government",
    "uk government",
    "us government",
    "federal government",
    "state government",
    "local government",
    "ministry",
    "department",
    "agency",
    # Retail (often false positives from articles)
    "aldi",
    "asda",
    "tesco",
    "sainsburys",
    "morrisons",
    "lidl",
    "waitrose",
    "co-op",
    "walmart",
    "target",
    "costco",
    "supermarket",
    "retail",
    "store",
    "shop",
    # Common Asian surnames (false positives from parsing)
    "jiang",
    "wang",
    "zhang",
    "chen",
    "liu",
    "li",
    "yang",
    "huang",
    "zhou",
    "wu",
    # Other invalid patterns
    "n/a",
    "none",
    "unknown",
    "various",
    "multiple",
    "other",
    "others",
    "tba",
    "independent",
    "freelance",
    "self-employed",
    "retired",
}

# =============================================================================
# Invalid Organization Patterns
# =============================================================================
# Patterns that indicate a headline/description rather than an org name

INVALID_ORG_PATTERNS: list[str] = [
    r"^new\s+",  # Headlines often start with "New ..."
    r"smash",  # "Smashes", "Smashing" - headline verbs
    r"breakthrough",
    r"discover",
    r"announc",  # "Announces", "Announced"
    r"reveal",
    r"launch",
    r"unveil",
    r"develop",
    r"creat",  # "Creates", "Created"
    r"powered",  # "Gold-Powered" etc.
    r"benchmark",
    r"record",
    r"-old\b",  # "decade-old", "year-old"
]

# =============================================================================
# Invalid Person Names
# =============================================================================
# Generic placeholders that aren't actual person names

INVALID_PERSON_NAMES: set[str] = {
    "individual researcher",
    "researcher",
    "professor",
    "scientist",
    "engineer",
    "author",
    "contributor",
    "correspondent",
    "editor",
    "staff",
    "staff writer",
    "team",
    "anonymous",
    "unknown",
}

# =============================================================================
# Valid Single-Word Organizations
# =============================================================================
# Single-word org abbreviations that ARE valid (exceptions to the "too short" rule)

VALID_SINGLE_WORD_ORGS: set[str] = {
    "mit",
    "nasa",
    "ibm",
    "gsk",
    "rspca",
    "bva",
    "basf",
    "dow",
    "shell",
    "bp",
    "sabic",
    "ineos",
    "linde",
}


def is_invalid_org_name(name: str) -> bool:
    """
    Check if a name is an invalid organization name.

    Args:
        name: Organization name to check

    Returns:
        True if the name is invalid (should be rejected)
    """
    import re

    if not name:
        return True

    normalized = name.lower().strip()

    # Check against invalid names set
    if normalized in INVALID_ORG_NAMES:
        return True

    # Check against patterns
    for pattern in INVALID_ORG_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return True

    return False


def is_invalid_person_name(name: str) -> bool:
    """
    Check if a name is an invalid person name.

    Args:
        name: Person name to check

    Returns:
        True if the name is invalid (should be rejected)
    """
    if not name:
        return True

    normalized = name.lower().strip()
    return normalized in INVALID_PERSON_NAMES


def is_valid_single_word_org(name: str) -> bool:
    """
    Check if a single-word organization abbreviation is valid.

    Args:
        name: Organization name to check

    Returns:
        True if it's a known valid single-word org
    """
    if not name:
        return False

    normalized = name.lower().strip()
    return normalized in VALID_SINGLE_WORD_ORGS
