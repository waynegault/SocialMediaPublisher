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

    # Remove standalone middle initials (e.g., "B." or "B" or "g.")
    # Pattern: single letter followed by optional period, surrounded by spaces or at word boundaries
    # First handle initials with periods (e.g., "wayne g. gault" -> "wayne gault")
    name = re.sub(r"\s+[a-z]\.\s+", " ", name)
    # Then handle initials without periods (e.g., "wayne g gault" -> "wayne gault")
    name = re.sub(r"\s+[a-z]\s+", " ", name)
    # Handle trailing initials with period
    name = re.sub(r"\s+[a-z]\.$", "", name)
    # Handle trailing initials without period
    name = re.sub(r"\s+[a-z]$", "", name)

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


# =============================================================================
# Text Similarity
# =============================================================================

# Stopwords for similarity calculations
SIMILARITY_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        # Pronouns and determiners
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
        "them",
        "their",
        # Question words
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        # Quantifiers and others
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
    }
)


def calculate_similarity(
    text1: str, text2: str, remove_stopwords: bool = True
) -> float:
    """
    Calculate similarity between two texts using the best of Jaccard and
    SequenceMatcher scores.

    Jaccard (word-set overlap) catches topically identical titles with different
    word order.  SequenceMatcher (character-level) catches titles that share a
    long common prefix/suffix but differ in a few words — e.g.
    "Advancements in Water Treatment Chemicals and Global Market Trends" vs
    "Advancements in Water Treatment Chemicals: A Growing Market" (Jaccard 0.63,
    SeqMatch 0.85).

    Args:
        text1: First text to compare.
        text2: Second text to compare.
        remove_stopwords: Whether to remove common stopwords before comparison.

    Returns:
        A value between 0.0 (no similarity) and 1.0 (identical).
    """
    if not text1 or not text2:
        return 0.0

    # --- Jaccard similarity (word-set) ---
    words1 = set(re.sub(r"[^\w\s]", "", text1.lower()).split())
    words2 = set(re.sub(r"[^\w\s]", "", text2.lower()).split())

    if remove_stopwords:
        words1 -= SIMILARITY_STOPWORDS
        words2 -= SIMILARITY_STOPWORDS

    jaccard = 0.0
    if words1 and words2:
        intersection = words1 & words2
        union = words1 | words2
        jaccard = len(intersection) / len(union) if union else 0.0

    # --- SequenceMatcher similarity (character-level) ---
    # Only use SequenceMatcher when Jaccard shows at least some word overlap.
    # Without this gate, SequenceMatcher gives noisy non-zero scores for
    # completely unrelated texts (e.g. "hello" vs "goodbye" → 0.17).
    # Average both directions to guarantee symmetry (SequenceMatcher's
    # internal junk heuristic can make ratio(a,b) != ratio(b,a)).
    seq_ratio = 0.0
    if jaccard > 0:
        from difflib import SequenceMatcher

        t1, t2 = text1.lower(), text2.lower()
        seq_ratio = (
            SequenceMatcher(None, t1, t2).ratio()
            + SequenceMatcher(None, t2, t1).ratio()
        ) / 2.0

    return max(jaccard, seq_ratio)


# =============================================================================
# Module Tests
# =============================================================================


def _create_module_tests() -> bool:
    """Create unit tests for text_utils module."""
    from test_framework import TestSuite

    suite = TestSuite("Text Utilities", "text_utils.py")
    suite.start_suite()

    def test_is_common_name_true():
        assert is_common_name("John") is True
        assert is_common_name("james") is True

    def test_is_common_name_false():
        assert is_common_name("Xiaoying") is False

    def test_strip_markdown_code_block():
        raw = '```json\n{"key": "value"}\n```'
        result = strip_markdown_code_block(raw)
        assert result.strip() == '{"key": "value"}'

    def test_strip_markdown_no_fences():
        assert strip_markdown_code_block("plain text") == "plain text"

    def test_get_name_variants():
        variants = get_name_variants("Robert")
        assert "robert" in variants
        assert "bob" in variants or "rob" in variants

    def test_is_nickname_of():
        assert is_nickname_of("Bob", "Robert") is True
        assert is_nickname_of("Alice", "Robert") is False

    def test_names_could_match_exact():
        assert names_could_match("John", "John") is True

    def test_names_could_match_prefix():
        assert names_could_match("Rob", "Robert") is True

    def test_strip_titles():
        assert strip_titles("Dr. John Smith") == "John Smith"
        assert strip_titles("Prof. Jane Doe") == "Jane Doe"

    def test_normalize_name():
        result = normalize_name("  Dr.  John   SMITH  ")
        assert result == "john smith"

    def test_extract_first_last_name():
        first, last = extract_first_last_name("John Smith")
        assert first == "john"
        assert last == "smith"

    def test_extract_first_last_single():
        first, last = extract_first_last_name("Madonna")
        assert first == "madonna"
        assert last == ""

    def test_name_matches_exact():
        assert name_matches("John Smith", "John Smith") is True

    def test_name_matches_different():
        assert name_matches("John Smith", "Jane Doe") is False

    def test_truncate_text():
        result = truncate_text("Hello World", max_length=8)
        assert len(result) <= 8
        assert result.endswith("...")

    def test_truncate_text_short():
        assert truncate_text("Hi", max_length=100) == "Hi"

    def test_calculate_similarity_identical():
        score = calculate_similarity(
            "quantum physics research", "quantum physics research"
        )
        assert score == 1.0

    def test_calculate_similarity_different():
        score = calculate_similarity("quantum physics", "banana smoothie")
        assert score == 0.0

    def test_calculate_similarity_empty():
        assert calculate_similarity("", "text") == 0.0
        assert calculate_similarity("text", "") == 0.0

    def test_build_context_keywords():
        kw = build_context_keywords(company="Acme Corp", department="R&D")
        assert "acme" in kw or "corp" in kw

    suite.run_test(

        test_name="is_common_name - common",

        test_func=test_is_common_name_true,

        test_summary="is_common_name behavior with common input",

        method_description="Testing is_common_name with common input using boolean return verification",

        expected_outcome="Function returns True for common input",

    )
    suite.run_test(
        test_name="is_common_name - uncommon",
        test_func=test_is_common_name_false,
        test_summary="is_common_name behavior with uncommon input",
        method_description="Testing is_common_name with uncommon input using boolean return verification",
        expected_outcome="Function returns False for uncommon input",
    )
    suite.run_test(
        test_name="strip_markdown_code_block",
        test_func=test_strip_markdown_code_block,
        test_summary="Verify strip_markdown_code_block produces correct results",
        method_description="Testing strip_markdown_code_block using equality assertions",
        expected_outcome="strip_markdown_code_block returns the expected value",
    )
    suite.run_test(
        test_name="strip_markdown - no fences",
        test_func=test_strip_markdown_no_fences,
        test_summary="strip_markdown behavior with no fences input",
        method_description="Testing strip_markdown with no fences input using equality assertions",
        expected_outcome="strip_markdown returns the expected value",
    )
    suite.run_test(
        test_name="get_name_variants",
        test_func=test_get_name_variants,
        test_summary="Verify get_name_variants produces correct results",
        method_description="Testing get_name_variants using membership verification",
        expected_outcome="Result contains expected elements",
    )
    suite.run_test(
        test_name="is_nickname_of",
        test_func=test_is_nickname_of,
        test_summary="Verify is_nickname_of produces correct results",
        method_description="Testing is_nickname_of using boolean return verification",
        expected_outcome="is_nickname_of returns True for matching input; is_nickname_of returns False for non-matching input",
    )
    suite.run_test(
        test_name="names_could_match - exact",
        test_func=test_names_could_match_exact,
        test_summary="names_could_match behavior with exact input",
        method_description="Testing names_could_match with exact input using boolean return verification",
        expected_outcome="Function returns True for exact input",
    )
    suite.run_test(
        test_name="names_could_match - prefix",
        test_func=test_names_could_match_prefix,
        test_summary="names_could_match behavior with prefix input",
        method_description="Testing names_could_match with prefix input using boolean return verification",
        expected_outcome="Function returns True for prefix input",
    )
    suite.run_test(
        test_name="strip_titles",
        test_func=test_strip_titles,
        test_summary="Verify strip_titles produces correct results",
        method_description="Testing strip_titles using equality assertions",
        expected_outcome="strip_titles returns the expected value",
    )
    suite.run_test(
        test_name="normalize_name",
        test_func=test_normalize_name,
        test_summary="Verify normalize_name produces correct results",
        method_description="Testing normalize_name using equality assertions",
        expected_outcome="normalize_name returns the expected value",
    )
    suite.run_test(
        test_name="extract_first_last_name",
        test_func=test_extract_first_last_name,
        test_summary="Verify extract_first_last_name produces correct results",
        method_description="Testing extract_first_last_name using equality assertions",
        expected_outcome="extract_first_last_name returns the expected value",
    )
    suite.run_test(
        test_name="extract_first_last - single",
        test_func=test_extract_first_last_single,
        test_summary="extract_first_last behavior with single input",
        method_description="Testing extract_first_last with single input using equality assertions",
        expected_outcome="extract_first_last returns the expected value",
    )
    suite.run_test(
        test_name="name_matches - exact",
        test_func=test_name_matches_exact,
        test_summary="name_matches behavior with exact input",
        method_description="Testing name_matches with exact input using boolean return verification",
        expected_outcome="Function returns True for exact input",
    )
    suite.run_test(
        test_name="name_matches - different",
        test_func=test_name_matches_different,
        test_summary="name_matches behavior with different input",
        method_description="Testing name_matches with different input using boolean return verification",
        expected_outcome="Function returns False for different input",
    )
    suite.run_test(
        test_name="truncate_text",
        test_func=test_truncate_text,
        test_summary="Verify truncate_text produces correct results",
        method_description="Testing truncate_text using size validation",
        expected_outcome="Result has the expected size; Result falls within expected bounds",
    )
    suite.run_test(
        test_name="truncate_text - short",
        test_func=test_truncate_text_short,
        test_summary="truncate_text behavior with short input",
        method_description="Testing truncate_text with short input using equality assertions",
        expected_outcome="truncate_text returns the expected value",
    )
    suite.run_test(
        test_name="calculate_similarity - identical",
        test_func=test_calculate_similarity_identical,
        test_summary="calculate_similarity behavior with identical input",
        method_description="Testing calculate_similarity with identical input using equality assertions",
        expected_outcome="calculate_similarity returns the expected value",
    )
    suite.run_test(
        test_name="calculate_similarity - different",
        test_func=test_calculate_similarity_different,
        test_summary="calculate_similarity behavior with different input",
        method_description="Testing calculate_similarity with different input using equality assertions",
        expected_outcome="calculate_similarity returns the expected value",
    )
    suite.run_test(
        test_name="calculate_similarity - empty",
        test_func=test_calculate_similarity_empty,
        test_summary="calculate_similarity behavior with empty input",
        method_description="Testing calculate_similarity with empty input using equality assertions",
        expected_outcome="calculate_similarity returns the expected value",
    )
    suite.run_test(
        test_name="build_context_keywords",
        test_func=test_build_context_keywords,
        test_summary="Verify build_context_keywords produces correct results",
        method_description="Testing build_context_keywords using membership verification",
        expected_outcome="Result contains expected elements",
    )
    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
