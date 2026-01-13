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


# Nickname mappings for fuzzy name matching
# Maps formal first names to common nicknames and vice versa
NICKNAME_MAP: dict[str, list[str]] = {
    "william": ["will", "bill", "billy", "willy", "liam"],
    "jennifer": ["jen", "jenny", "jenn"],
    "michael": ["mike", "mikey", "mick"],
    "robert": ["rob", "bob", "bobby", "robbie", "bert"],
    "elizabeth": ["liz", "beth", "lizzy", "betty", "eliza", "ellie"],
    "richard": ["rick", "rich", "dick", "ricky"],
    "james": ["jim", "jimmy", "jamie"],
    "margaret": ["meg", "maggie", "peggy", "marge", "margo"],
    "katherine": ["kate", "kathy", "katie", "kat", "cathy", "kay"],
    "kathryn": ["kate", "kathy", "katie", "kat", "cathy", "kay"],
    "christopher": ["chris", "kit", "topher"],
    "anthony": ["tony", "ant"],
    "nicholas": ["nick", "nicky", "nico"],
    "alexander": ["alex", "xander", "alec", "lex"],
    "benjamin": ["ben", "benny", "benji"],
    "daniel": ["dan", "danny"],
    "matthew": ["matt", "matty"],
    "jonathan": ["jon", "jonny", "nathan"],
    "joseph": ["joe", "joey"],
    "timothy": ["tim", "timmy"],
    "edward": ["ed", "ted", "eddie", "teddy", "ned"],
    "thomas": ["tom", "tommy"],
    "david": ["dave", "davy"],
    "stephen": ["steve", "stevie"],
    "steven": ["steve", "stevie"],
    "andrew": ["andy", "drew"],
    "charles": ["charlie", "chuck", "chas"],
    "raymond": ["ray"],
    "gerald": ["gerry", "jerry"],
    "lawrence": ["larry", "laurie"],
    "patricia": ["pat", "patty", "trish", "tricia"],
    "barbara": ["barb", "barbie"],
    "susan": ["sue", "susie", "suzy"],
    "rebecca": ["becky", "becca"],
    "dorothy": ["dot", "dottie", "dorothea"],
    "deborah": ["deb", "debbie"],
    "jessica": ["jess", "jessie"],
    "samantha": ["sam", "sammy"],
    "alexandra": ["alex", "lexi", "sandra"],
    "victoria": ["vicky", "vic", "tori"],
    "natalie": ["nat", "natty"],
    "jacqueline": ["jackie", "jacqui"],
    "caroline": ["carrie", "carol"],
    "catherine": ["cathy", "kate", "katie", "cat"],
    "phillip": ["phil"],
    "philip": ["phil"],
    "frederick": ["fred", "freddy", "rick"],
    "gregory": ["greg"],
    "peter": ["pete"],
    "donald": ["don", "donny"],
    "ronald": ["ron", "ronny"],
    "kenneth": ["ken", "kenny"],
    "eugene": ["gene"],
    "harold": ["harry", "hal"],
    "henry": ["hank", "harry"],
    "walter": ["walt", "wally"],
    "arthur": ["art", "artie"],
    "albert": ["al", "bert", "bertie"],
    "leonard": ["leo", "lenny", "len"],
    "theodore": ["ted", "teddy", "theo"],
    "frank": ["frankie"],
    "francis": ["frank", "fran"],
    "francesca": ["fran", "frankie", "frannie"],
}

# Build reverse mapping for quick lookup
_REVERSE_NICKNAME_MAP: dict[str, str] = {}
for formal, nicknames in NICKNAME_MAP.items():
    for nick in nicknames:
        _REVERSE_NICKNAME_MAP[nick] = formal


def get_name_variants(name: str) -> set[str]:
    """Get all known variants (nicknames) of a given first name.

    Args:
        name: A first name.

    Returns:
        Set of all known variants including the original name.
    """
    name_lower = name.lower()
    variants = {name_lower}

    # Check if it's a formal name with nicknames
    if name_lower in NICKNAME_MAP:
        variants.update(NICKNAME_MAP[name_lower])

    # Check if it's a nickname with a formal name
    if name_lower in _REVERSE_NICKNAME_MAP:
        formal = _REVERSE_NICKNAME_MAP[name_lower]
        variants.add(formal)
        # Also add all other nicknames for this formal name
        variants.update(NICKNAME_MAP[formal])

    return variants


def is_nickname_of(name1: str, name2: str) -> bool:
    """Check if one name is a nickname variant of another.

    Args:
        name1: First name to compare.
        name2: Second name to compare.

    Returns:
        True if the names are variants of each other.
    """
    variants1 = get_name_variants(name1)
    variants2 = get_name_variants(name2)
    return bool(variants1 & variants2)


def names_could_match(name1: str, name2: str, min_prefix: int = 2) -> bool:
    """Check if two first names could be the same person using multiple heuristics.

    This is a general approach that handles:
    - Known nicknames (via NICKNAME_MAP)
    - Prefix matching (e.g., 'Kam' and 'Kathryn' both start with 'Ka')
    - Phonetic similarity for longer shared prefixes

    Args:
        name1: First name from search/article.
        name2: First name from LinkedIn profile.
        min_prefix: Minimum shared prefix length to consider a match (default 2).

    Returns:
        True if the names could plausibly be the same person.
    """
    if not name1 or not name2:
        return False

    n1_lower = name1.lower().strip()
    n2_lower = name2.lower().strip()

    # Exact match
    if n1_lower == n2_lower:
        return True

    # Check known nickname variants
    if is_nickname_of(n1_lower, n2_lower):
        return True

    # Check prefix matching - handles cases like "Kam" / "Kathryn" (both start with "Ka")
    # Use the shorter name's length (minus 1) as the prefix length, with minimum of min_prefix
    shorter_len = min(len(n1_lower), len(n2_lower))
    prefix_len = max(min_prefix, shorter_len - 1)

    # Don't match on very short prefixes unless names are very short
    if shorter_len < min_prefix:
        prefix_len = shorter_len

    if prefix_len > 0 and n1_lower[:prefix_len] == n2_lower[:prefix_len]:
        return True

    return False


# Comprehensive title/prefix patterns to strip from names
TITLE_PREFIXES: list[str] = [
    r"^(?:the\s+)?rt\.?\s*hon(?:ourable)?\.?\s+",  # Right Honourable
    r"^(?:the\s+)?hon(?:ourable)?\.?\s+",  # Honourable
    r"^sir\s+",
    r"^dame\s+",
    r"^lord\s+",
    r"^lady\s+",
    r"^baron(?:ess)?\s+",
    r"^count(?:ess)?\s+",
    r"^duke\s+",
    r"^duchess\s+",
    r"^prince(?:ss)?\s+",
    r"^dr\.?\s+",
    r"^prof(?:essor)?\.?\s+",
    r"^pres(?:ident)?\.?\s+",
    r"^dean\s+",
    r"^provost\s+",
    r"^chancellor\s+",
    r"^vice[\s\-]?chancellor\s+",
    r"^rector\s+",
    r"^principal\s+",
    r"^emeritus\s+",
    r"^distinguished\s+",
    r"^associate\s+",
    r"^assistant\s+",
    r"^adjunct\s+",
    r"^visiting\s+",
    r"^research\s+",
    r"^senior\s+",
    r"^executive\s+",
    r"^mr\.?\s+",
    r"^mrs\.?\s+",
    r"^ms\.?\s+",
    r"^miss\s+",
    r"^mx\.?\s+",
    r"^rev(?:erend)?\.?\s+",
    r"^rabbi\s+",
    r"^imam\s+",
    r"^fr\.?\s+",  # Father
    r"^brother\s+",
    r"^sister\s+",
    r"^captain\s+",
    r"^capt\.?\s+",
    r"^colonel\s+",
    r"^col\.?\s+",
    r"^general\s+",
    r"^gen\.?\s+",
    r"^major\s+",
    r"^maj\.?\s+",
    r"^admiral\s+",
    r"^adm\.?\s+",
]


def strip_titles(name: str) -> str:
    """Strip all titles and honorifics from a name.

    Handles academic titles (Prof., Dr.), honorary titles (Sir, Dame, Lord),
    military ranks, religious titles, and common honorifics.

    Args:
        name: A name that may include titles.

    Returns:
        Name with all title prefixes removed.
    """
    if not name:
        return ""

    result = name.strip()

    # Apply all title patterns repeatedly until no more matches
    # (handles cases like "Professor Emeritus Dr. John Smith")
    changed = True
    while changed:
        changed = False
        for pattern in TITLE_PREFIXES:
            new_result = re.sub(pattern, "", result, flags=re.IGNORECASE)
            if new_result != result:
                result = new_result.strip()
                changed = True

    return result.strip()


def normalize_name(name: str) -> str:
    """Normalize a name for comparison.

    Handles:
    - Case normalization (lowercase)
    - Multiple spaces
    - Common prefixes (Dr., Prof., etc.) - now using comprehensive strip_titles
    - Suffixes (Jr., Sr., PhD, etc.)

    Args:
        name: A person's name.

    Returns:
        Normalized name for comparison.
    """
    if not name:
        return ""

    # First strip all titles using comprehensive function
    name = strip_titles(name).lower()

    # Remove common suffixes
    suffixes = [
        r",?\s+jr\.?$",
        r",?\s+sr\.?$",
        r",?\s+phd\.?$",
        r",?\s+ph\.?d\.?$",
        r",?\s+md\.?$",
        r",?\s+m\.?d\.?$",
        r",?\s+esq\.?$",
        r",?\s+ii+$",
        r",?\s+iii+$",
        r",?\s+iv$",
        r",?\s+dphil\.?$",
        r",?\s+dsc\.?$",
        r",?\s+frs\.?$",  # Fellow of Royal Society
        r",?\s+freng\.?$",  # Fellow of Royal Academy of Engineering
        r",?\s+cbe\.?$",
        r",?\s+obe\.?$",
        r",?\s+mbe\.?$",
        r",?\s+knighted$",
    ]
    for suffix in suffixes:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE)

    # Remove standalone middle initials (e.g., "B." or "B")
    name = re.sub(r"\b[a-z]\.\b", " ", name)
    name = re.sub(r"\b[a-z]\b", " ", name)

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
