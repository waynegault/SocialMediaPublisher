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
    r"\bcrack",  # "Cracks", "Cracking" - headline verbs (e.g., "China Cracks")
    r"\bsolv",  # "Solves", "Solving"
    r"\bwin",  # "Wins"
    r"\bbeat",  # "Beats"
    r"\bcode\b",  # "The Code" - often part of headlines
    r"\bsecret\b",  # "Secret" - headline word
    r"manufactur",  # "Manufacturing" as headline action
    r"\bhigh-performance\b",  # Technical descriptor, not org
    r"^how\s+",  # Headlines: "How [org]..."
    r"^why\s+",  # Headlines: "Why [org]..."
    r"^what\s+",  # Headlines: "What [org]..."
    r"^china\s+",  # Country + verb headlines (e.g., "China Cracks", "China Develops")
    r"^usa?\s+",  # "US Develops", "USA Announces"
    r"^uk\s+",  # "UK Unveils"
    r"^india\s+",  # "India Launches"
    r"^japan\s+",  # "Japan Creates"
    r"\bnot\s+applicable\b",  # AI explanation text
    r"^not\s+",  # "Not applicable", "Not specified"
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
    "n/a",
    "none",
    "not applicable",
    "not specified",
    "tba",
    "tbd",
    "placeholder",
}

# Patterns that indicate AI explanation text rather than a real person name
# These are checked as substrings (case-insensitive)
INVALID_PERSON_PATTERNS: list[str] = [
    "not applicable",
    "this is not",
    "no organization",
    "none mentioned",
    "not specified",
    "generalized",
    "headline",
    "actual research",
    "the organization",
    "the company",
    "the university",
    "not a person",
    "not an individual",
    " - ",  # Dash with spaces often indicates explanation
]

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

    # Check against invalid names set
    if normalized in INVALID_PERSON_NAMES:
        return True

    # Check against AI explanation patterns (substring match)
    for pattern in INVALID_PERSON_PATTERNS:
        if pattern in normalized:
            return True

    # Reject names that are too long (likely AI explanations)
    if len(name) > 60:
        return True

    # Reject names with too many words (real names are typically 2-4 words)
    word_count = len(normalized.split())
    if word_count > 6:
        return True

    return False


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
