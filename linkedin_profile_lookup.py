"""LinkedIn company and person profile lookup using Gemini with Google Search grounding and undetected-chromedriver."""

import base64
import concurrent.futures
import json
import logging
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, TypedDict, cast

import httpx
from google import genai  # type: ignore

from api_client import api_client
from config import Config
from error_handling import with_enhanced_recovery, NetworkTimeoutError
from organization_aliases import ORG_ALIASES as _ORG_ALIASES
from rate_limiter import AdaptiveRateLimiter

# Import shared constants from entity_constants (no optional dependencies)
from entity_constants import (
    INVALID_ORG_NAMES,
    INVALID_ORG_PATTERNS,
    INVALID_PERSON_NAMES,
    VALID_SINGLE_WORD_ORGS,
)

logger = logging.getLogger(__name__)

# Import undetected-chromedriver for CAPTCHA-resistant browser automation
# Type stubs for optional dependencies
uc: Any = None
UC_AVAILABLE = False

try:
    import undetected_chromedriver as _uc

    uc = _uc
    UC_AVAILABLE = True
except ImportError:
    logger.warning(
        "undetected-chromedriver not installed - pip install undetected-chromedriver"
    )


def _suppress_uc_cleanup_errors() -> None:
    """Suppress Windows handle errors from UC Chrome cleanup."""
    if sys.platform == "win32":
        # Monkey-patch time.sleep in the UC module to suppress cleanup errors
        original_sleep = time.sleep

        def patched_sleep(seconds: float) -> None:
            try:
                original_sleep(seconds)
            except OSError:
                pass  # Suppress WinError 6: handle is invalid

        # Apply to UC module if loaded
        if UC_AVAILABLE:
            import undetected_chromedriver

            undetected_chromedriver.time = type(sys)("time")
            undetected_chromedriver.time.sleep = patched_sleep


# Apply the patch on module load
_suppress_uc_cleanup_errors()


# === TypedDict definitions for structured return types ===


class CacheCountStats(TypedDict):
    """Statistics for a single cache type."""

    total: int
    found: int
    not_found: int


class AllCacheStats(TypedDict):
    """Statistics for all cache types."""

    person: CacheCountStats
    company: CacheCountStats
    department: CacheCountStats


class TimingStats(TypedDict):
    """Timing statistics for an operation type."""

    avg: float
    min: float
    max: float
    total: float
    count: int


class GeminiStats(TypedDict):
    """Statistics for Gemini API usage."""

    attempts: int
    successes: int
    success_rate: float
    disabled: bool


def _clean_optional_string(value: Optional[str]) -> Optional[str]:
    """Clean optional string value, returning None if empty or literal 'None'."""
    if value is None:
        return None
    stripped = str(value).strip()
    # Filter out empty strings and literal "None" string
    if not stripped or stripped.lower() == "none":
        return None
    return stripped


@dataclass
class PersonSearchContext:
    """Context for person LinkedIn profile search."""

    first_name: str
    last_name: str
    company: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    location: Optional[str] = None
    role_type: Optional[str] = None
    research_area: Optional[str] = None
    require_org_match: bool = True
    is_common_name: bool = False
    context_keywords: set[str] = field(default_factory=set)


# Import shared constants from text_utils
from text_utils import (
    COMMON_FIRST_NAMES,
    CONTEXT_STOPWORDS,
    build_context_keywords,
    is_common_name,
    normalize_name,
    strip_titles,
    get_name_variants,
    is_nickname_of,
    names_could_match,
)


def _build_context_keywords(ctx: PersonSearchContext) -> set[str]:
    """Build context keywords from search context for matching."""
    return build_context_keywords(
        company=ctx.company,
        department=ctx.department,
        position=ctx.position,
        research_area=ctx.research_area,
    )


def _calculate_contradiction_penalty(
    result_text: str, ctx: PersonSearchContext
) -> tuple[int, list[str]]:
    """Calculate penalty score for contradicting signals in search result.

    Returns:
        Tuple of (penalty_score, list_of_reasons)
    """
    penalty = 0
    reasons: list[str] = []

    # Wrong field of work - unrelated professions
    if ctx.role_type == "academic":
        wrong_fields = [
            "real estate agent",
            "realtor",
            "sales manager",
            "marketing manager",
            "insurance agent",
            "financial advisor",
            "fitness trainer",
            "hair stylist",
            "photographer",
            "life coach",
            "attorney",
            "lawyer",
            ", esq",
            "law firm",
            "legal counsel",
        ]
        for wrong_field in wrong_fields:
            if wrong_field in result_text:
                penalty += 3
                reasons.append(f"wrong field: {wrong_field}")

    elif ctx.role_type == "executive" and ctx.company:
        # Executive at company X should not be an attorney/lawyer
        lawyer_indicators = [
            "attorney",
            "lawyer",
            ", esq",
            "law firm",
            "legal counsel",
            "partner at law",
        ]
        for indicator in lawyer_indicators:
            if indicator in result_text:
                # Strong penalty - lawyers are rarely CEOs of non-law companies
                penalty += 4
                reasons.append(f"wrong profession: {indicator}")
                break

        # Executive at company X should not show as professor at different org
        if "professor" in result_text:
            company_lower = ctx.company.lower()
            if not any(
                word in result_text for word in company_lower.split() if len(word) > 2
            ):
                penalty += 2
                reasons.append("professor at different org")

    # Wrong industry indicators
    if ctx.department or ctx.research_area:
        expected_terms: set[str] = set()
        if ctx.department:
            expected_terms.update(
                w.lower() for w in ctx.department.split() if len(w) > 4
            )
        if ctx.research_area:
            expected_terms.update(
                w.lower() for w in ctx.research_area.split() if len(w) > 4
            )

        science_expected = any(
            term in expected_terms
            for term in [
                "chemical",
                "engineering",
                "biology",
                "chemistry",
                "physics",
                "research",
            ]
        )
        if science_expected:
            unrelated = [
                "real estate",
                "retail sales",
                "hospitality",
                "food service",
                "beauty salon",
            ]
            for industry in unrelated:
                if industry in result_text:
                    penalty += 2
                    reasons.append(f"unrelated industry: {industry}")

    # Location contradiction - different country
    if ctx.location:
        location_parts = [p.strip() for p in ctx.location.lower().split(",")]
        expected_country = location_parts[-1] if location_parts else ""

        country_conflicts = {
            "usa": ["india", "china", "brazil", "indonesia", "nigeria", "pakistan"],
            "uk": ["india", "china", "brazil", "indonesia", "nigeria", "pakistan"],
            "canada": ["india", "china", "brazil", "indonesia", "nigeria"],
            "australia": ["india", "china", "brazil", "indonesia"],
            "germany": ["india", "china", "brazil", "indonesia"],
        }

        for expected, conflicts in country_conflicts.items():
            if expected in expected_country:
                for conflict_country in conflicts:
                    if conflict_country in result_text:
                        # Don't penalize if expected location also appears
                        if not any(
                            part in result_text
                            for part in location_parts
                            if len(part) > 2
                        ):
                            penalty += 2
                            reasons.append(f"location mismatch: {conflict_country}")
                            break

    return penalty, reasons


def _score_linkedin_candidate(
    linkedin_url: str,
    result_text: str,
    ctx: PersonSearchContext,
) -> tuple[int, list[str], bool]:
    """Score a LinkedIn profile candidate based on matching signals.

    Args:
        linkedin_url: The LinkedIn profile URL
        result_text: Text from the search result
        ctx: Search context with person details

    Returns:
        Tuple of (score, matched_keywords, name_in_url)
    """
    url_slug = linkedin_url.split("/in/")[-1] if "/in/" in linkedin_url else ""
    url_text = url_slug.replace("-", " ").lower()
    # Also keep the raw slug for checking concatenated names like "kellybenkert"
    url_slug_lower = url_slug.lower()
    result_lower = result_text.lower()

    # Check if name is in URL (including nickname variants and prefix matching)
    # Handle both separated names (kelly-benkert) and concatenated names (kellybenkert)
    name_in_url = True
    first_name_in_url = False
    first_name_exact_match = False  # Track if first name is exact vs fuzzy match
    if ctx.first_name and ctx.last_name:
        first_name_lower = ctx.first_name.lower()
        last_name_lower = ctx.last_name.lower()

        # Check with word boundaries first (for dash-separated names)
        first_pattern = r"\b" + re.escape(first_name_lower) + r"\b"
        last_pattern = r"\b" + re.escape(last_name_lower) + r"\b"
        first_in_url = bool(re.search(first_pattern, url_text))
        last_in_url = bool(re.search(last_pattern, url_text))

        # Track if this is an exact match (word boundary match)
        first_name_exact_match = first_in_url

        # Also check for concatenated names (e.g., "kellybenkert" contains "kelly" and "benkert")
        if not first_in_url:
            first_in_url = first_name_lower in url_slug_lower
            if first_in_url:
                first_name_exact_match = (
                    True  # Substring match is still considered exact
                )
        if not last_in_url:
            last_in_url = last_name_lower in url_slug_lower

        # General name matching: check if URL contains a name that could match
        # This handles nicknames AND prefix matching (e.g., "Kam" matches "Kathryn" via "Ka" prefix)
        if not first_in_url:
            # Extract potential first name from URL (first part before last name or first word)
            url_parts = url_text.split()
            # For concatenated slugs, try to extract first name by removing last name
            if last_name_lower in url_slug_lower:
                # Find where the last name starts and extract what's before it
                last_name_pos = url_slug_lower.find(last_name_lower)
                if last_name_pos > 0:
                    potential_first = url_slug_lower[:last_name_pos].strip("-_ ")
                    if potential_first and names_could_match(
                        first_name_lower, potential_first
                    ):
                        first_in_url = True
                        # This is a fuzzy match, not exact
                        first_name_exact_match = False
            # For dash-separated slugs, check each part
            if not first_in_url and url_parts:
                for part in url_parts:
                    if names_could_match(first_name_lower, part):
                        first_in_url = True
                        first_name_exact_match = False  # Fuzzy match
                        break

        first_name_in_url = first_in_url
        name_in_url = first_in_url and last_in_url

    score = 0
    matched: list[str] = []

    # Context keyword matches
    for keyword in ctx.context_keywords:
        if keyword in result_lower:
            score += 1
            matched.append(keyword)

    # Strong boost for name in URL - exact matches get higher score
    if name_in_url:
        if first_name_exact_match:
            score += 4  # Exact first name match gets +4 (was +3)
            matched.append("name_in_url_exact")
        else:
            score += 2  # Fuzzy first name match (prefix/nickname) gets +2
            matched.append("name_in_url_fuzzy")
    elif ctx.last_name and ctx.last_name.lower() in url_text:
        # Partial boost for last name only in URL (useful for single-name searches)
        score += 2
        matched.append("last_name_in_url")

    # Org match
    if ctx.company:
        company_lower = ctx.company.lower()
        company_words = [w for w in company_lower.split() if len(w) > 3]
        org_matched = any(word in result_lower for word in company_words)
        org_matched = org_matched or company_lower in result_lower
        if org_matched:
            score += 2
        elif ctx.require_org_match:
            return -100, [], name_in_url  # Disqualify

    # Location match
    if ctx.location:
        location_parts = [p.strip() for p in ctx.location.lower().split(",")]
        if any(part in result_lower for part in location_parts if len(part) > 2):
            score += 1

    # Role-type specific matches
    if ctx.role_type == "academic":
        academic_indicators = ["professor", "researcher", "phd", "dr.", "university"]
        if any(ind in result_lower for ind in academic_indicators):
            score += 1
    elif ctx.role_type == "executive":
        exec_indicators = ["ceo", "cto", "cfo", "vp", "president", "director", "chief"]
        if any(ind in result_lower for ind in exec_indicators):
            score += 1

    # Apply contradiction penalties
    penalty, reasons = _calculate_contradiction_penalty(result_lower, ctx)
    if penalty > 0:
        score -= penalty
        logger.debug(
            f"Contradiction for '{linkedin_url}': {reasons}, penalty={penalty}"
        )

    return score, matched, name_in_url


class LinkedInCompanyLookup:
    """Look up LinkedIn company pages using Gemini with Google Search grounding."""

    # === CLASS-LEVEL SHARED STATE ===
    # Chrome driver is shared across ALL instances to prevent multiple browser windows
    _shared_uc_driver: Optional[Any] = None
    _shared_driver_search_count: int = 0
    _shared_linkedin_login_verified: bool = False
    _shared_driver_lock = None  # Will be initialized on first use (threading.Lock)

    # === ORGANIZATION NAME ALIASES (imported from organization_aliases module) ===
    # See module-level import: from organization_aliases import ORG_ALIASES as _ORG_ALIASES

    # === INVALID ENTITY NAMES (imported from entity_constants) ===
    # Use module-level imports: INVALID_ORG_NAMES, INVALID_ORG_PATTERNS,
    # INVALID_PERSON_NAMES, VALID_SINGLE_WORD_ORGS

    # === CLASS-LEVEL CACHES (shared across all instances and steps) ===
    # Cache person search results - Key: "name@canonical_company" -> URL or None
    _shared_person_cache: dict[str, Optional[str]] = {}
    # Cache found profiles by name only - Key: normalized_name -> profile_url
    # This allows finding the same person even with different org variations
    _shared_found_profiles_by_name: dict[str, str] = {}
    # Cache LinkedIn company URL to canonical name - Key: linkedin_url -> canonical_name
    _shared_company_url_to_name: dict[str, str] = {}
    # Cache failed lookups with timestamp to avoid re-searching - Key: name -> timestamp
    _shared_failed_lookups: dict[str, float] = {}
    # Failed lookup TTL in seconds (don't re-search failed lookups within this window)
    _FAILED_LOOKUP_TTL: float = 86400.0  # 24 hours
    # Cache company search results - Key: company name -> (url, slug, urn) or None
    _shared_company_cache: dict[
        str, Optional[tuple[str, Optional[str], Optional[str]]]
    ] = {}

    def __init__(self, genai_client: Optional[genai.Client] = None) -> None:
        """Initialize the LinkedIn company lookup service.

        Args:
            genai_client: Optional Gemini client. If not provided, creates one using Config.
        """
        if genai_client:
            self.client = genai_client
        elif Config.GEMINI_API_KEY:
            self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        else:
            self.client = None
            logger.warning(
                "GEMINI_API_KEY not configured - LinkedIn company lookup disabled"
            )

        # HTTP client for LinkedIn API calls
        self._http_client: Optional[httpx.Client] = None

        # Use class-level shared driver (singleton pattern)
        # This prevents multiple browser instances from being created

        # Track if we've verified LinkedIn login this session (use class-level)
        # Instance reference for backward compatibility

        # === INSTANCE PROPERTIES THAT REDIRECT TO CLASS-LEVEL CACHES ===
        # (for backward compatibility - all instances share the same caches)

        # Cache of profile URLs already validated and rejected for a given normalized name
        # Key: normalized name -> set of rejected profile URLs
        self._rejected_profile_cache: dict[str, set[str]] = {}

        # Cache department search results across multiple stories
        # Key: "dept@company" -> (url, slug, urn) or None
        self._department_cache: dict[
            str, tuple[Optional[str], Optional[str], Optional[str]]
        ] = {}

        # Track Gemini fallback success rate - disable after too many failures
        self._gemini_attempts = 0
        self._gemini_successes = 0
        self._gemini_disabled = False

        # Rate limiting for search engines to avoid CAPTCHA
        self._last_search_time: float = 0.0
        self._min_search_interval: float = (
            8.0  # Minimum seconds between searches (increased to avoid CAPTCHA)
        )
        self._consecutive_searches: int = 0
        self._captcha_cooldown_until: float = 0.0  # Timestamp when cooldown ends
        # NOTE: _driver_search_count is now class-level (_shared_driver_search_count)
        self._max_searches_per_driver: int = (
            8  # Recreate driver after this many searches
        )

        # Timing metrics for search operations
        self._timing_stats: dict[str, list[float]] = {
            "person_search": [],
            "company_search": [],
            "department_search": [],
        }

        # Cache persistence file path
        self._cache_dir = Path(
            os.path.expandvars(r"%LOCALAPPDATA%\SocialMediaPublisher")
        )
        self._cache_file = self._cache_dir / "linkedin_cache.json"

        # Load persisted cache on startup
        self._load_cache_from_disk()

    # === Properties to access class-level shared state ===
    @property
    def _uc_driver(self) -> Optional[Any]:
        """Get the shared Chrome driver (class-level singleton)."""
        return LinkedInCompanyLookup._shared_uc_driver

    @_uc_driver.setter
    def _uc_driver(self, value: Optional[Any]) -> None:
        """Set the shared Chrome driver (class-level singleton)."""
        LinkedInCompanyLookup._shared_uc_driver = value

    @property
    def _driver_search_count(self) -> int:
        """Get the shared driver search count."""
        return LinkedInCompanyLookup._shared_driver_search_count

    @_driver_search_count.setter
    def _driver_search_count(self, value: int) -> None:
        """Set the shared driver search count."""
        LinkedInCompanyLookup._shared_driver_search_count = value

    @property
    def _linkedin_login_verified(self) -> bool:
        """Get the shared login verification status."""
        return LinkedInCompanyLookup._shared_linkedin_login_verified

    @_linkedin_login_verified.setter
    def _linkedin_login_verified(self, value: bool) -> None:
        """Set the shared login verification status."""
        LinkedInCompanyLookup._shared_linkedin_login_verified = value

    @property
    def _person_cache(self) -> dict[str, Optional[str]]:
        """Get the shared person cache (class-level singleton)."""
        return LinkedInCompanyLookup._shared_person_cache

    @_person_cache.setter
    def _person_cache(self, value: dict[str, Optional[str]]) -> None:
        """Set the shared person cache (class-level singleton)."""
        LinkedInCompanyLookup._shared_person_cache = value

    @property
    def _company_cache(
        self,
    ) -> dict[str, Optional[tuple[str, Optional[str], Optional[str]]]]:
        """Get the shared company cache (class-level singleton)."""
        return LinkedInCompanyLookup._shared_company_cache

    @_company_cache.setter
    def _company_cache(
        self, value: dict[str, Optional[tuple[str, Optional[str], Optional[str]]]]
    ) -> None:
        """Set the shared company cache (class-level singleton)."""
        LinkedInCompanyLookup._shared_company_cache = value

    def get_cache_stats(self) -> AllCacheStats:
        """Get statistics about all search caches.

        Returns:
            Dict with stats for 'person', 'company', 'department' caches
        """
        person_total = len(self._person_cache)
        person_found = sum(1 for url in self._person_cache.values() if url is not None)

        company_total = len(self._company_cache)
        company_found = sum(1 for v in self._company_cache.values() if v is not None)

        dept_total = len(self._department_cache)
        dept_found = sum(1 for v in self._department_cache.values() if v[0] is not None)

        return {
            "person": {
                "total": person_total,
                "found": person_found,
                "not_found": person_total - person_found,
            },
            "company": {
                "total": company_total,
                "found": company_found,
                "not_found": company_total - company_found,
            },
            "department": {
                "total": dept_total,
                "found": dept_found,
                "not_found": dept_total - dept_found,
            },
        }

    def get_timing_stats(self) -> dict[str, dict[str, float]]:
        """Get timing statistics for search operations.

        Returns:
            Dict with avg/min/max/total times in seconds for each operation type
        """
        result = {}
        for op_type, times in self._timing_stats.items():
            if times:
                result[op_type] = {
                    "count": len(times),
                    "total": sum(times),
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }
            else:
                result[op_type] = {"count": 0, "total": 0, "avg": 0, "min": 0, "max": 0}
        return result

    def get_gemini_stats(self) -> GeminiStats:
        """Get Gemini fallback statistics.

        Returns:
            Dict with attempts, successes, success_rate, and disabled status
        """
        rate = (
            self._gemini_successes / self._gemini_attempts * 100
            if self._gemini_attempts > 0
            else 0
        )
        return {
            "attempts": self._gemini_attempts,
            "successes": self._gemini_successes,
            "success_rate": round(rate, 1),
            "disabled": self._gemini_disabled,
        }

    def _normalize_org_name(self, org_name: str) -> str:
        """Normalize organization name to canonical form.

        This prevents duplicate searches for the same organization with
        different name variations (e.g., "UChicago" vs "University of Chicago").

        Args:
            org_name: Raw organization name from story/person data

        Returns:
            Normalized canonical organization name (lowercase)
        """
        if not org_name:
            return ""

        # Lowercase and strip whitespace
        norm = org_name.lower().strip()

        # Strip leading "the " - common variation (e.g., "the University of Chicago")
        if norm.startswith("the "):
            norm = norm[4:].strip()

        # Check if it's an alias we know about
        if norm in _ORG_ALIASES:
            canonical = _ORG_ALIASES[norm]
            logger.debug(f"Normalized org '{org_name}' -> '{canonical}'")
            return canonical

        # Also check without common suffixes for partial matches
        # e.g., "University of Chicago" should match even if stored as "university of chicago"
        suffixes_to_strip = [
            " university",
            " college",
            " institute",
            " corporation",
            " inc",
            " inc.",
            " llc",
            " ltd",
            " co",
            " co.",
            " & co",
            " & company",
            " company",
            " group",
        ]

        stripped = norm
        for suffix in suffixes_to_strip:
            if stripped.endswith(suffix):
                stripped = stripped[: -len(suffix)].strip()
                break

        # Check stripped version in aliases
        if stripped in _ORG_ALIASES:
            canonical = _ORG_ALIASES[stripped]
            logger.debug(f"Normalized org '{org_name}' (stripped) -> '{canonical}'")
            return canonical

        # No alias found - return the normalized name
        return norm

    def _expand_org_for_search(self, org_name: str) -> str:
        """Expand organization abbreviation to full name for search queries.

        Unlike _normalize_org_name (which returns lowercase for cache keys),
        this returns a properly cased name suitable for search engines.

        Args:
            org_name: Raw organization name (e.g., "UChicago", "MIT")

        Returns:
            Expanded organization name with proper casing (e.g., "University of Chicago")
        """
        if not org_name:
            return ""

        # Check lowercase version against aliases
        norm = org_name.lower().strip()

        if norm in _ORG_ALIASES:
            # Return title-cased expanded name
            expanded = _ORG_ALIASES[norm]
            # Title case but preserve common words
            result = " ".join(
                word
                if word.lower() in ("of", "and", "the", "for", "at", "in", "on")
                else word.capitalize()
                for word in expanded.split()
            )
            # Capitalize first word always
            words = result.split()
            if words:
                words[0] = words[0].capitalize()
                result = " ".join(words)
            logger.debug(f"Expanded org '{org_name}' -> '{result}' for search")
            return result

        # Also check without common suffixes
        suffixes_to_strip = [
            " university",
            " college",
            " institute",
            " corporation",
            " inc",
            " inc.",
            " llc",
            " ltd",
            " co",
            " co.",
            " & co",
            " & company",
            " company",
            " group",
        ]

        stripped = norm
        for suffix in suffixes_to_strip:
            if stripped.endswith(suffix):
                stripped = stripped[: -len(suffix)].strip()
                break

        if stripped in _ORG_ALIASES:
            expanded = _ORG_ALIASES[stripped]
            result = " ".join(
                word
                if word.lower() in ("of", "and", "the", "for", "at", "in", "on")
                else word.capitalize()
                for word in expanded.split()
            )
            words = result.split()
            if words:
                words[0] = words[0].capitalize()
                result = " ".join(words)
            logger.debug(
                f"Expanded org '{org_name}' (stripped) -> '{result}' for search"
            )
            return result

        # No alias found - return original name unchanged
        return org_name.strip()

    def _is_valid_org_name(self, org_name: str) -> bool:
        """Check if an organization name is valid for searching.

        Filters out:
        - Single generic words that aren't real organizations
        - Research topics or materials (e.g., "MXenes", "Molecular")
        - Person names that were incorrectly parsed as orgs
        - Very short names (likely parsing errors)
        - Very long names (likely full sentences or descriptions)

        Args:
            org_name: Organization name to validate

        Returns:
            True if the org name appears valid for searching
        """
        if not org_name:
            return False

        # Normalize for checking
        norm = org_name.lower().strip()

        # Strip "the " prefix for checking
        if norm.startswith("the "):
            norm = norm[4:].strip()

        # Check against known invalid names
        if norm in INVALID_ORG_NAMES:
            logger.debug(f"Skipping invalid org name: '{org_name}' (in blocklist)")
            return False

        # Single words are usually not valid orgs unless they're known aliases
        words = norm.split()
        if len(words) == 1:
            # Check if it's a known alias
            if norm not in _ORG_ALIASES:
                # Single word that's not a known alias - likely invalid
                # Exceptions: very short common abbrevs handled in aliases
                logger.debug(
                    f"Skipping single-word org: '{org_name}' (not in known aliases)"
                )
                return False

        # Very long names are likely descriptions, not org names
        if len(org_name) > 120:
            logger.debug(
                f"Skipping too-long org name: '{org_name}' ({len(org_name)} chars)"
            )
            return False

        # Very short names (1-2 chars) are likely errors unless they're aliases
        if len(norm) <= 2 and norm not in _ORG_ALIASES:
            logger.debug(f"Skipping too-short org name: '{org_name}'")
            return False

        # Check against regex patterns (headlines, action verbs, etc.)
        for pattern in INVALID_ORG_PATTERNS:
            if re.search(pattern, norm, re.IGNORECASE):
                logger.debug(
                    f"Skipping org matching invalid pattern: '{org_name}' (pattern: {pattern})"
                )
                return False

        return True

    def _is_valid_person_name(self, name: str) -> bool:
        """Check if a person name is valid for searching.

        Filters out:
        - Generic placeholders like "Individual Researcher", "Staff Writer"
        - Role descriptions that aren't actual names
        - Very short names (likely parsing errors)

        Args:
            name: Person name to validate

        Returns:
            True if the name appears to be a real person's name
        """
        if not name:
            return False

        # Normalize for checking
        norm = name.lower().strip()

        # Check against known invalid person names
        if norm in INVALID_PERSON_NAMES:
            logger.debug(f"Skipping invalid person name: '{name}' (in blocklist)")
            return False

        # Very short names (< 3 chars) are likely errors
        if len(norm) < 3:
            logger.debug(f"Skipping too-short person name: '{name}'")
            return False

        # Names with only one word that's a common role/title are invalid
        words = norm.split()
        if len(words) == 1:
            role_words = {
                "researcher",
                "professor",
                "scientist",
                "engineer",
                "doctor",
                "author",
                "editor",
                "correspondent",
                "writer",
                "contributor",
                "analyst",
                "manager",
                "director",
                "head",
                "lead",
                "chief",
                "staff",
                "team",
                "group",
                "anonymous",
                "unknown",
            }
            if norm in role_words:
                logger.debug(f"Skipping single-word role as name: '{name}'")
                return False

        return True

    def _load_cache_from_disk(self) -> None:
        """Load persisted cache from disk if available.

        Implements TTL-based expiry for negative cache entries:
        - Positive results (URLs found) are kept forever
        - Negative results (None) expire after NEGATIVE_CACHE_TTL_DAYS
        """
        if not self._cache_file.exists():
            return

        NEGATIVE_CACHE_TTL_DAYS = 7  # Retry failed searches after 7 days
        now = time.time()
        ttl_seconds = NEGATIVE_CACHE_TTL_DAYS * 24 * 3600

        try:
            with open(self._cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Restore person cache with TTL for negative entries
            expired_count = 0
            person_data = data.get("person", {})
            for key, val in person_data.items():
                if isinstance(val, dict):
                    # New format with timestamp
                    url = val.get("url")
                    ts = val.get("ts", 0)
                    if url is not None:
                        # Positive result - keep forever
                        self._person_cache[key] = url
                    elif now - ts < ttl_seconds:
                        # Recent negative result - keep it
                        self._person_cache[key] = None
                    else:
                        # Expired negative result - don't add (will be re-searched)
                        expired_count += 1
                else:
                    # Old format without timestamp - treat as needing re-search if negative
                    if val is not None:
                        self._person_cache[key] = val
                    # Skip old None entries to force re-search

            if expired_count > 0:
                logger.info(
                    f"Expired {expired_count} old negative cache entries (will be retried)"
                )

            # Restore company cache (convert lists back to tuples)
            for key, val in data.get("company", {}).items():
                if val is None:
                    self._company_cache[key] = None
                else:
                    self._company_cache[key] = tuple(val)  # type: ignore

            # Restore department cache (convert lists back to tuples)
            for key, val in data.get("department", {}).items():
                self._department_cache[key] = tuple(val)  # type: ignore

            total = (
                len(self._person_cache)
                + len(self._company_cache)
                + len(self._department_cache)
            )
            logger.info(f"Loaded {total} cached entries from {self._cache_file}")

        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted cache file, starting fresh: {e}")
        except OSError as e:
            logger.warning(f"Failed to load cache from disk: {e}")

    def save_cache_to_disk(self) -> None:
        """Persist cache to disk for future runs.

        Saves person cache with timestamps for TTL-based expiry of negative entries.
        """
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # Save person cache with timestamps for negative entries
            person_data = {}
            now = time.time()
            for key, url in self._person_cache.items():
                if url is not None:
                    # Positive result - store with URL
                    person_data[key] = {"url": url, "ts": now}
                else:
                    # Negative result - store with timestamp for TTL
                    person_data[key] = {"url": None, "ts": now}

            data = {
                "person": person_data,
                "company": {
                    k: list(v) if v else None for k, v in self._company_cache.items()
                },
                "department": {k: list(v) for k, v in self._department_cache.items()},
            }

            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            total = (
                len(self._person_cache)
                + len(self._company_cache)
                + len(self._department_cache)
            )
            logger.info(f"Saved {total} cache entries to {self._cache_file}")

        except (OSError, TypeError) as e:
            logger.warning(f"Failed to save cache to disk: {e}")

    def get_person_cache_stats(self) -> CacheCountStats:
        """Get statistics about the person search cache.

        Returns:
            Dict with 'total', 'found' (with URLs), 'not_found' (None values)
        """
        return self.get_cache_stats()["person"]

    def clear_all_caches(self) -> dict[str, int]:
        """Clear all search caches.

        Returns:
            Dict with count of entries cleared per cache type
        """
        counts = {
            "person": len(self._person_cache),
            "company": len(self._company_cache),
            "department": len(self._department_cache),
            "found_profiles_by_name": len(
                LinkedInCompanyLookup._shared_found_profiles_by_name
            ),
            "company_url_to_name": len(
                LinkedInCompanyLookup._shared_company_url_to_name
            ),
        }
        self._person_cache.clear()
        self._company_cache.clear()
        self._department_cache.clear()
        LinkedInCompanyLookup._shared_found_profiles_by_name.clear()
        LinkedInCompanyLookup._shared_company_url_to_name.clear()
        logger.info(f"Cleared all caches: {counts}")
        return counts

    def clear_person_cache(self) -> int:
        """Clear the person search cache.

        Returns:
            Number of entries cleared
        """
        count = len(self._person_cache)
        count += len(LinkedInCompanyLookup._shared_found_profiles_by_name)
        self._person_cache.clear()
        LinkedInCompanyLookup._shared_found_profiles_by_name.clear()
        logger.info(f"Cleared person cache ({count} entries)")
        return count

    def __enter__(self) -> "LinkedInCompanyLookup":
        """Context manager entry - returns self."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Context manager exit - closes browser session."""
        self.close_browser()
        return False

    def _reset_chrome_preferences(self, profile_path: str) -> None:
        """Reset Chrome preferences file to avoid detection flags.

        Based on ancestry project's approach to defeat bot detection.
        """
        preferences_file = Path(profile_path) / "Default" / "Preferences"
        try:
            preferences_file.parent.mkdir(parents=True, exist_ok=True)
            minimal_preferences = {
                "profile": {"exit_type": "Normal", "exited_cleanly": True},
                "browser": {
                    "has_seen_welcome_page": True,
                    "window_placement": {
                        "bottom": 1,
                        "left": 0,
                        "maximized": False,
                        "right": 1,
                        "top": 0,
                        "work_area_bottom": 1,
                        "work_area_left": 0,
                        "work_area_right": 1,
                        "work_area_top": 0,
                    },
                },
                "privacy_sandbox": {
                    "m1": {
                        "ad_measurement_enabled": False,
                        "consent_decision_made": True,
                        "eea_notice_acknowledged": True,
                        "fledge_enabled": False,
                        "topics_enabled": False,
                    }
                },
                "sync": {"allowed": False},
                "extensions": {"alerts": {"initialized": True}},
                "session": {"restore_on_startup": 4, "startup_urls": []},
            }
            with open(preferences_file, "w", encoding="utf-8") as f:
                json.dump(minimal_preferences, f, indent=2)
            logger.debug(f"Reset Chrome preferences at {preferences_file}")
        except OSError as e:
            logger.warning(f"Could not reset Chrome preferences: {e}")

    def _get_uc_driver(self) -> Optional[Any]:
        """Get or create a shared UC Chrome driver instance.

        Uses enhanced anti-detection approach based on ancestry project:
        - Additional stealth flags to appear as normal browser
        - Experimental options to disable automation indicators
        - Preferences reset to clean state
        - Version pinning for ChromeDriver compatibility

        The driver is reused across multiple searches for efficiency.
        Call close_browser() when done with all searches.

        Returns:
            UC Chrome driver instance, or None if UC not available
        """
        if not UC_AVAILABLE:
            logger.warning(
                "undetected-chromedriver not installed - pip install undetected-chromedriver"
            )
            return None

        # Return existing driver if still valid and not over usage limit
        if self._uc_driver is not None:
            # Check if we should recreate driver due to search count
            if self._driver_search_count >= self._max_searches_per_driver:
                logger.debug(
                    f"Resetting search count after {self._driver_search_count} searches (keeping same driver)"
                )
                # Instead of recreating, just reset count and add a longer delay
                # Recreating the driver causes Chrome profile lock issues
                self._driver_search_count = 0
                time.sleep(5 + random.random() * 3)  # Longer pause to avoid detection
                try:
                    # Clear cookies and refresh to reset LinkedIn's tracking
                    self._uc_driver.delete_all_cookies()
                    self._uc_driver.get("https://www.linkedin.com")
                    time.sleep(2)
                    return self._uc_driver
                except Exception:
                    # Driver died, will recreate below
                    try:
                        self._uc_driver.quit()
                    except Exception:
                        pass
                    self._uc_driver = None
            else:
                try:
                    # Check if driver is still alive
                    _ = self._uc_driver.current_url
                    return self._uc_driver
                except Exception:
                    # Driver is dead, clean up and create new one
                    try:
                        self._uc_driver.quit()
                    except Exception:
                        pass
                    self._uc_driver = None
                    self._driver_search_count = 0
                    time.sleep(2)  # Allow Chrome to fully terminate before restart

        # Create new driver with enhanced anti-detection settings
        try:
            from selenium.webdriver.common.by import By

            options = uc.ChromeOptions()

            # === CORE STABILITY OPTIONS ===
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-software-rasterizer")
            options.add_argument("--no-first-run")
            options.add_argument("--no-default-browser-check")

            # === ANTI-DETECTION: Disable automation flags (ancestry approach) ===
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-popup-blocking")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-plugins-discovery")
            options.add_argument("--disable-automation")

            # === ANTI-DETECTION: Experimental options to hide automation ===
            # NOTE: excludeSwitches and useAutomationExtension are deprecated in newer ChromeDriver
            # The --disable-blink-features=AutomationControlled flag above handles anti-detection
            # options.add_experimental_option(
            #     "excludeSwitches", ["enable-automation", "enable-logging"]
            # )
            # options.add_experimental_option("useAutomationExtension", False)

            # === ANTI-DETECTION: Disable password manager and notifications ===
            prefs = {
                "credentials_enable_service": False,
                "profile.password_manager_enabled": False,
                "profile.default_content_setting_values.notifications": 2,
            }
            options.add_experimental_option("prefs", prefs)

            # Use a dedicated profile directory for automation
            # This persists LinkedIn login between runs without conflicting with main Chrome
            automation_profile = os.path.expandvars(
                r"%LOCALAPPDATA%\SocialMediaPublisher\ChromeProfile"
            )
            os.makedirs(automation_profile, exist_ok=True)
            options.add_argument(f"--user-data-dir={automation_profile}")
            options.add_argument("--profile-directory=Default")
            logger.debug(f"Using automation profile at {automation_profile}")

            # Reset preferences to clean state (ancestry approach)
            self._reset_chrome_preferences(automation_profile)

            # Consistent user agent (random user agents are red flags for bot detection)
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
            options.add_argument(f"--user-agent={user_agent}")

            # Randomize window size slightly (exact same size each time is suspicious)
            width = 1920 + random.randint(-100, 100)
            height = 1080 + random.randint(-50, 50)
            options.add_argument(f"--window-size={width},{height}")

            # Create driver with version pinning for compatibility
            # use_subprocess=True with headless=False prevents port conflicts
            # no_sandbox=True is already set via args
            self._uc_driver = uc.Chrome(
                options=options,
                version_main=142,  # Pin to Chrome 142 for stability
                use_subprocess=True,  # Separate process avoids port conflicts on restart
                suppress_welcome=True,
                headless=False,  # Must be False for LinkedIn (detects headless)
            )
            if self._uc_driver is not None:
                self._uc_driver.set_page_load_timeout(30)
            logger.debug(
                "Created new UC Chrome driver session with enhanced anti-detection"
            )
            return self._uc_driver

        except Exception as e:
            logger.error(f"Failed to create UC Chrome driver: {e}")
            return None

    def _ensure_linkedin_login(self, driver) -> bool:
        """Ensure user is logged in to LinkedIn with the correct account.

        Verifies the logged-in account matches the configured LINKEDIN_USERNAME.
        If a different account is logged in, logs out and re-authenticates.

        Returns:
            True if logged in with correct account, False if login failed
        """
        if self._linkedin_login_verified:
            return True

        # Navigate to LinkedIn to check login status
        try:
            driver.get("https://www.linkedin.com/feed/")
            time.sleep(3)

            current_url = driver.current_url

            # If redirected to login page, need to log in
            if (
                "/login" in current_url
                or "/authwall" in current_url
                or "/checkpoint" in current_url
            ):
                return self._perform_login(driver)

            # We appear to be logged in - verify it's the correct account
            if Config.LINKEDIN_USERNAME:
                logged_in_email = self._get_logged_in_email(driver)
                if logged_in_email:
                    expected_email = Config.LINKEDIN_USERNAME.lower().strip()
                    actual_email = logged_in_email.lower().strip()
                    if actual_email != expected_email:
                        logger.info(
                            f"Wrong LinkedIn account logged in. Expected: {expected_email}, Got: {actual_email}"
                        )
                        print(f"\n⚠️  Wrong LinkedIn account detected: {actual_email}")
                        print(f"   Expected account: {expected_email}")
                        print("   Logging out and switching accounts...")

                        # Reset login flag since we're switching accounts
                        self._linkedin_login_verified = False

                        # Log out and re-login with correct credentials
                        if self._logout_linkedin(driver):
                            time.sleep(2)  # Allow logout to fully complete
                            return self._perform_login(driver)
                        else:
                            logger.error("Failed to log out of wrong account")
                            return False
                    else:
                        logger.info(
                            f"Verified correct LinkedIn account: {actual_email}"
                        )

            self._linkedin_login_verified = True
            logger.info("LinkedIn login verified")
            return True

        except Exception as e:
            logger.error(f"Error checking LinkedIn login: {e}")
            return False

    def _get_logged_in_email(self, driver) -> Optional[str]:
        """Get the email of the currently logged-in LinkedIn user.

        Returns:
            Email address if found, None otherwise
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            # Navigate to settings page which shows email
            driver.get("https://www.linkedin.com/psettings/email")
            time.sleep(2)

            # Look for email in the settings page
            try:
                # Primary email is usually in a specific element
                email_elements = driver.find_elements(
                    By.CSS_SELECTOR, "[data-email], .email-address, [href^='mailto:']"
                )
                for elem in email_elements:
                    text = (
                        elem.get_attribute("data-email")
                        or elem.get_attribute("href")
                        or elem.text
                    )
                    if text and "@" in text:
                        email = text.replace("mailto:", "").strip()
                        logger.debug(f"Found logged-in email: {email}")
                        return email

                # Alternative: look in page text
                page_source = driver.page_source
                import re

                # Find email pattern near "Primary email" or similar
                email_match = re.search(
                    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", page_source
                )
                if email_match:
                    email = email_match.group(0)
                    # Filter out common non-user emails
                    if not any(
                        x in email.lower()
                        for x in ["linkedin.com", "example.com", "email.com"]
                    ):
                        logger.debug(f"Found email from page source: {email}")
                        return email

            except Exception as e:
                logger.debug(f"Error finding email element: {e}")

            # Fallback: Try the account settings page
            driver.get("https://www.linkedin.com/mypreferences/d/sign-in-and-security")
            time.sleep(2)

            import re

            email_match = re.search(
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", driver.page_source
            )
            if email_match:
                email = email_match.group(0)
                if not any(
                    x in email.lower()
                    for x in ["linkedin.com", "example.com", "email.com"]
                ):
                    return email

            return None

        except Exception as e:
            logger.error(f"Error getting logged-in email: {e}")
            return None

    def _logout_linkedin(self, driver) -> bool:
        """Log out of LinkedIn.

        Returns:
            True if logout successful, False otherwise
        """
        try:
            # Direct navigation to logout endpoint
            driver.get("https://www.linkedin.com/m/logout/")
            time.sleep(3)

            # Check if we're back at login page
            current_url = driver.current_url
            if (
                "/login" in current_url
                or "/authwall" in current_url
                or "linkedin.com/home" in current_url
            ):
                logger.info("Successfully logged out of LinkedIn")
                return True

            # Alternative: try the settings logout
            driver.get("https://www.linkedin.com/logout")
            time.sleep(3)

            return "/login" in driver.current_url or "/authwall" in driver.current_url

        except Exception as e:
            logger.error(f"Error logging out of LinkedIn: {e}")
            return False

    def _perform_login(self, driver) -> bool:
        """Perform LinkedIn login with configured credentials or prompt for manual login.

        Returns:
            True if login successful, False otherwise
        """
        # Try automatic login if credentials are configured
        if Config.LINKEDIN_USERNAME and Config.LINKEDIN_PASSWORD:
            logger.info(
                f"Attempting automatic LinkedIn login as {Config.LINKEDIN_USERNAME}..."
            )
            if self._auto_login(driver):
                self._linkedin_login_verified = True
                logger.info("Automatic LinkedIn login successful")
                return True
            else:
                logger.warning("Automatic login failed, falling back to manual")

        # Fall back to manual login
        print("\n" + "=" * 60)
        print("LinkedIn Login Required")
        print("=" * 60)
        print("A browser window has opened. Please log in to LinkedIn.")
        print("This is required to extract @mention URNs from profiles.")
        print("Your session will be saved for future runs.")
        print(
            "\nTip: Add linkedin_username and linkedin_password to .env for auto-login."
        )
        print("\nAfter logging in, press Enter to continue...")
        input()

        # Check if login succeeded
        time.sleep(2)
        current_url = driver.current_url
        if "/login" in current_url or "/authwall" in current_url:
            print("Login not detected. Please try again.")
            return False

        self._linkedin_login_verified = True
        return True

    def _auto_login(self, driver) -> bool:
        """Attempt automatic login using credentials from config.

        Returns:
            True if login successful, False otherwise
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.common.keys import Keys

            # Clear cookies first to ensure clean login state
            try:
                driver.delete_all_cookies()
            except Exception:
                pass

            # Navigate to login page with a fresh start
            driver.get("https://www.linkedin.com/login")
            time.sleep(3)  # Give page more time to fully load

            # Wait for page to be ready
            WebDriverWait(driver, 15).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            # Wait for and fill username field
            username_field = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable((By.ID, "username"))
            )
            username_field.click()
            time.sleep(0.5)
            username_field.clear()

            # Type slowly to avoid detection
            for char in Config.LINKEDIN_USERNAME:
                username_field.send_keys(char)
                time.sleep(0.05 + random.random() * 0.05)

            # Fill password field
            password_field = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "password"))
            )
            password_field.click()
            time.sleep(0.5)
            password_field.clear()

            # Type password slowly
            for char in Config.LINKEDIN_PASSWORD:
                password_field.send_keys(char)
                time.sleep(0.05 + random.random() * 0.05)

            time.sleep(1)  # Brief pause before clicking

            # Click sign in button
            sign_in_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))
            )
            sign_in_button.click()

            # Wait for redirect (successful login redirects away from /login)
            time.sleep(5)

            current_url = driver.current_url

            # Check for security checkpoint (2FA, CAPTCHA, etc.)
            if "/checkpoint" in current_url:
                print("\n" + "=" * 60)
                print("LinkedIn Security Checkpoint Detected")
                print("=" * 60)
                print("Please complete the security verification in the browser.")
                print("\nPress Enter after completing verification...")
                input()
                time.sleep(2)
                current_url = driver.current_url

            # Success if we're no longer on login/authwall page
            if "/login" not in current_url and "/authwall" not in current_url:
                return True

            return False

        except Exception as e:
            logger.error(f"Auto-login failed: {e}")
            return False

    def close_browser(self) -> None:
        """Close the shared browser session.

        Note: Since the driver is shared across all instances, this affects
        all LinkedInCompanyLookup instances. Only call when completely done
        with all LinkedIn searches.
        """
        if LinkedInCompanyLookup._shared_uc_driver is not None:
            try:
                LinkedInCompanyLookup._shared_uc_driver.quit()
            except Exception:
                pass
            LinkedInCompanyLookup._shared_uc_driver = None
            LinkedInCompanyLookup._shared_driver_search_count = 0
            LinkedInCompanyLookup._shared_linkedin_login_verified = False
            logger.debug("Closed shared UC Chrome driver session")

    @classmethod
    def close_shared_browser(cls) -> None:
        """Class method to close the shared browser session.

        Can be called without an instance: LinkedInCompanyLookup.close_shared_browser()
        """
        if cls._shared_uc_driver is not None:
            try:
                cls._shared_uc_driver.quit()
            except Exception:
                pass
            cls._shared_uc_driver = None
            cls._shared_driver_search_count = 0
            cls._shared_linkedin_login_verified = False
            logger.debug("Closed shared UC Chrome driver session")

    def send_connection_via_browser(
        self,
        profile_url: str,
        message: str | None = None,
    ) -> tuple[bool, str]:
        """
        Send a LinkedIn connection request by visiting the profile and clicking Connect.

        Uses undetected-chromedriver to automate the browser, avoiding API restrictions.

        Args:
            profile_url: LinkedIn profile URL (e.g., 'https://www.linkedin.com/in/username')
            message: Optional custom message (max 300 chars) - requires LinkedIn Premium for custom messages

        Returns:
            Tuple of (success: bool, message: str describing result)
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver = self._get_uc_driver()
        if not driver:
            return (
                False,
                "Browser automation not available - install undetected-chromedriver",
            )

        # Ensure logged in
        if not self._ensure_linkedin_login(driver):
            return (False, "LinkedIn login required")

        try:
            # Navigate to the profile
            logger.debug(f"Visiting profile: {profile_url}")
            driver.get(profile_url)
            time.sleep(3 + random.random() * 2)  # Wait for page load

            # Check if profile exists
            if "Page not found" in driver.page_source or "/404" in driver.current_url:
                return (False, "Profile not found")

            # Look for Connect button - LinkedIn has multiple possible button configurations
            connect_selectors = [
                # Primary Connect button on profile page
                'button[aria-label*="Invite"][aria-label*="connect"]',
                'button[aria-label*="Connect with"]',
                # Connect in the action buttons area
                'div.pvs-profile-actions button:has-text("Connect")',
                'button.artdeco-button--primary:has-text("Connect")',
                # More general selectors
                '//button[contains(., "Connect")]',
                '//button[contains(@aria-label, "connect")]',
            ]

            connect_button = None

            # Try CSS selectors first
            for selector in connect_selectors[:4]:
                try:
                    buttons = driver.find_elements(By.CSS_SELECTOR, selector)
                    for btn in buttons:
                        btn_text = btn.text.strip().lower()
                        if "connect" in btn_text and btn.is_displayed():
                            connect_button = btn
                            break
                    if connect_button:
                        break
                except Exception:
                    continue

            # Try XPath selectors
            if not connect_button:
                for selector in connect_selectors[4:]:
                    try:
                        buttons = driver.find_elements(By.XPATH, selector)
                        for btn in buttons:
                            if btn.is_displayed():
                                connect_button = btn
                                break
                        if connect_button:
                            break
                    except Exception:
                        continue

            # Final fallback: find any button with "Connect" text
            if not connect_button:
                try:
                    all_buttons = driver.find_elements(By.TAG_NAME, "button")
                    for btn in all_buttons:
                        try:
                            btn_text = btn.text.strip().lower()
                            aria_label = (btn.get_attribute("aria-label") or "").lower()
                            if (
                                btn.is_displayed()
                                and ("connect" in btn_text or "connect" in aria_label)
                                and "disconnect" not in btn_text
                                and "message" not in btn_text
                            ):
                                connect_button = btn
                                break
                        except Exception:
                            continue
                except Exception:
                    pass

            if not connect_button:
                # Check if already connected or pending
                page_text = driver.page_source.lower()
                if "pending" in page_text and "invitation" in page_text:
                    return (False, "Invitation already pending")
                if "message" in page_text and "connect" not in page_text:
                    return (False, "Already connected (Message button visible)")
                return (False, "Connect button not found - may already be connected")

            # Scroll the button into view
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});", connect_button
            )
            time.sleep(0.5 + random.random() * 0.5)

            # Click the Connect button
            logger.debug("Clicking Connect button...")
            connect_button.click()
            time.sleep(2 + random.random())

            # Handle the connection modal (if it appears)
            # LinkedIn may show "Add a note" or "Send without a note" options
            try:
                # If custom message provided, click "Add a note" and fill in the message
                if message:
                    # Look for "Add a note" button
                    add_note_buttons = driver.find_elements(
                        By.XPATH,
                        '//button[contains(., "Add a note")]',
                    )
                    for btn in add_note_buttons:
                        if btn.is_displayed():
                            btn.click()
                            time.sleep(1 + random.random())
                            break

                    # Find the textarea for the message and fill it
                    try:
                        textarea = WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located(
                                (
                                    By.CSS_SELECTOR,
                                    'textarea[name="message"], textarea#custom-message',
                                )
                            )
                        )
                        textarea.clear()
                        # Type the message slowly to appear human
                        for char in message[:300]:  # LinkedIn limits to 300 chars
                            textarea.send_keys(char)
                            time.sleep(0.02 + random.random() * 0.02)
                        time.sleep(0.5)
                        logger.debug(f"Added custom message: {message[:50]}...")
                    except Exception as e:
                        logger.debug(f"Could not find message textarea: {e}")
                        # Continue anyway - will try to send without custom message

                    # Click Send button
                    send_buttons = driver.find_elements(
                        By.XPATH,
                        '//button[contains(@aria-label, "Send") or contains(., "Send invitation") or contains(., "Send")]',
                    )
                    for btn in send_buttons:
                        btn_text = btn.text.strip().lower()
                        if (
                            btn.is_displayed()
                            and "send" in btn_text
                            and "without" not in btn_text
                        ):
                            btn.click()
                            time.sleep(1)
                            logger.info(
                                f"Connection request sent with message to {profile_url}"
                            )
                            return (True, "Connection request sent with message")

                # Look for "Send without a note" button if no custom message
                if not message:
                    send_buttons = driver.find_elements(
                        By.XPATH,
                        '//button[contains(., "Send without a note") or contains(., "Send")]',
                    )
                    for btn in send_buttons:
                        if btn.is_displayed() and "without" in btn.text.lower():
                            btn.click()
                            time.sleep(1)
                            logger.info(f"Connection request sent to {profile_url}")
                            return (True, "Connection request sent")

                # If custom message provided or no "without note" option, look for send button
                send_buttons = driver.find_elements(
                    By.XPATH,
                    '//button[contains(@aria-label, "Send") or contains(., "Send invitation")]',
                )
                for btn in send_buttons:
                    if btn.is_displayed():
                        btn.click()
                        time.sleep(1)
                        logger.info(f"Connection request sent to {profile_url}")
                        return (True, "Connection request sent")

                # If modal has a send button
                send_buttons = driver.find_elements(
                    By.CSS_SELECTOR, 'button[aria-label*="Send"]'
                )
                for btn in send_buttons:
                    if btn.is_displayed():
                        btn.click()
                        time.sleep(1)
                        logger.info(f"Connection request sent to {profile_url}")
                        return (True, "Connection request sent")

            except Exception as e:
                logger.debug(f"Modal handling: {e}")

            # If we clicked Connect and no modal appeared, it might have just sent
            return (True, "Connection request sent (or modal appeared)")

        except Exception as e:
            logger.error(f"Browser connection error: {e}")
            return (False, f"Browser error: {str(e)}")

    def send_bulk_connections_via_browser(
        self,
        profile_urls: list[str],
        delay_seconds: float = 5.0,
    ) -> tuple[int, int, list[str]]:
        """
        Send connection requests to multiple profiles using browser automation.

        Args:
            profile_urls: List of LinkedIn profile URLs
            delay_seconds: Delay between requests (default 5s for safety)

        Returns:
            Tuple of (success_count, failure_count, list of error messages)
        """
        success_count = 0
        failure_count = 0
        errors: list[str] = []

        for i, url in enumerate(profile_urls):
            success, result_msg = self.send_connection_via_browser(url)

            if success:
                success_count += 1
                logger.info(f"[{i + 1}/{len(profile_urls)}] Connected: {url}")
            else:
                failure_count += 1
                errors.append(f"{url}: {result_msg}")
                logger.warning(
                    f"[{i + 1}/{len(profile_urls)}] Failed: {url} - {result_msg}"
                )

            # Rate limiting delay between requests
            if i < len(profile_urls) - 1:
                # Add some randomness to appear more human
                actual_delay = delay_seconds + random.random() * 3
                time.sleep(actual_delay)

        return (success_count, failure_count, errors)

    def _get_http_client(self) -> httpx.Client:
        """Get or create HTTP client for LinkedIn API calls."""
        if self._http_client is None:
            self._http_client = httpx.Client(
                timeout=10.0,
                headers={
                    "Authorization": f"Bearer {Config.LINKEDIN_ACCESS_TOKEN}",
                    "X-Restli-Protocol-Version": "2.0.0",
                    "LinkedIn-Version": "202401",
                },
            )
        return self._http_client

    def lookup_organization_by_vanity_name(
        self, vanity_name: str
    ) -> Optional[tuple[str, int]]:
        """
        Look up organization URN using LinkedIn's official API.

        Uses the endpoint: https://api.linkedin.com/v2/organizations?q=vanityName&vanityName=xxx

        Args:
            vanity_name: The vanity name (slug) from the LinkedIn URL, e.g., "stanford-university"

        Returns:
            Tuple of (urn, organization_id) if found, None otherwise
            e.g., ("urn:li:organization:12345", 12345)
        """
        if not Config.LINKEDIN_ACCESS_TOKEN:
            logger.warning("LINKEDIN_ACCESS_TOKEN not configured - cannot lookup URN")
            return None

        if not vanity_name:
            return None

        # Clean the vanity name (remove any URL parts if present)
        vanity_name = vanity_name.strip().lower()
        if "/" in vanity_name:
            # Extract just the slug if a URL was passed
            match = re.search(
                r"linkedin\.com/(?:company|school)/([\w\-]+)", vanity_name
            )
            if match:
                vanity_name = match.group(1)

        logger.info(f"Looking up LinkedIn organization by vanity name: {vanity_name}")

        try:
            client = self._get_http_client()
            response = client.get(
                "https://api.linkedin.com/v2/organizations",
                params={"q": "vanityName", "vanityName": vanity_name},
            )

            if response.status_code == 200:
                data = response.json()
                elements = data.get("elements", [])

                if elements:
                    org = elements[0]
                    org_id = org.get("id")
                    if org_id:
                        urn = f"urn:li:organization:{org_id}"
                        logger.info(
                            f"Found organization via API: {vanity_name} -> {urn}"
                        )
                        return (urn, org_id)

                logger.info(f"No organization found for vanity name: {vanity_name}")
                return None

            elif response.status_code == 401:
                logger.error("LinkedIn API authentication failed - check access token")
                return None
            elif response.status_code == 403:
                logger.error(
                    "LinkedIn API access denied - token may lack required permissions"
                )
                return None
            else:
                logger.warning(
                    f"LinkedIn API returned {response.status_code} for {vanity_name}"
                )
                return None

        except httpx.TimeoutException:
            logger.warning(f"Timeout calling LinkedIn API for: {vanity_name}")
            return None
        except Exception as e:
            logger.error(f"Error calling LinkedIn API for {vanity_name}: {e}")
            return None

    def validate_linkedin_org_url(self, url: str) -> tuple[bool, Optional[str]]:
        """
        Validate that a LinkedIn company/school URL is well-formed and extract the slug.

        Note: LinkedIn blocks direct HTTP requests with 999 status, so we cannot
        validate by fetching the page. Instead we validate the URL format and
        extract the organization slug which can be used for further lookups.

        Args:
            url: LinkedIn company or school URL

        Returns:
            Tuple of (is_valid, organization_slug)
            organization_slug is the path component like "stanford-university" or "basf"
        """
        if not url:
            return (False, None)

        # Validate URL format
        pattern = r"https?://(?:www\.)?linkedin\.com/(company|school)/([\w\-]+)"
        match = re.match(pattern, url)

        if not match:
            logger.debug(f"Invalid LinkedIn URL format: {url}")
            return (False, None)

        org_type = match.group(1)  # "company" or "school"
        org_slug = match.group(2)  # e.g., "stanford-university"

        logger.debug(f"Valid LinkedIn {org_type} URL, slug: {org_slug}")
        return (True, org_slug)

    def lookup_organization_urn(self, url: str) -> Optional[str]:
        """
        Find the numeric organization ID for a LinkedIn company/school URL.

        First tries LinkedIn's official API (requires valid access token).
        Falls back to Gemini with Google Search if API fails.

        Args:
            url: LinkedIn company or school URL

        Returns:
            URN string like "urn:li:organization:12345" if found, None otherwise
        """
        if not url:
            return None

        # Extract the slug from the URL
        pattern = r"linkedin\.com/(company|school)/([\w\-]+)"
        match = re.search(pattern, url)
        if not match:
            return None

        org_slug = match.group(2)

        # Try 1: Use LinkedIn's official API
        result = self.lookup_organization_by_vanity_name(org_slug)
        if result:
            urn, _ = result
            return urn

        # Try 2: Fall back to Gemini search (usually doesn't work but worth trying)
        return self._lookup_urn_via_gemini(url, org_slug)

    def _lookup_urn_via_gemini(self, url: str, org_slug: str) -> Optional[str]:
        """Fall back to Gemini search for organization URN (rarely works)."""
        if not self.client:
            return None

        org_type = "company" if "/company/" in url else "school"

        prompt = f"""Find the LinkedIn organization ID for: {url}

I need the numeric organization ID (also called company ID) for this LinkedIn page.
The slug is: {org_slug}

Search for information about this LinkedIn {org_type} page's numeric ID.
The ID is typically a number like 1234567 or similar.

RESPONSE FORMAT:
If you find the organization ID, respond with just the number:
12345678

If you cannot find it, respond with:
NOT_FOUND"""

        try:
            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_TEXT,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "max_output_tokens": 64,
                },
                endpoint="org_id_lookup",
            )

            if not response.text:
                return None

            result = response.text.strip()

            if "NOT_FOUND" in result.upper():
                return None

            # Extract numeric ID from response
            id_match = re.search(r"\b(\d{5,15})\b", result)
            if id_match:
                org_id = id_match.group(1)
                urn = f"urn:li:organization:{org_id}"
                logger.info(f"Found organization URN via Gemini for {org_slug}: {urn}")
                return urn

            return None

        except Exception as e:
            logger.error(f"Error looking up organization URN for {url}: {e}")
            return None

    def search_company(
        self, company_name: str, validate: bool = True
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Search for a LinkedIn company page by company name.

        Uses Gemini with Google Search grounding to find the LinkedIn company URL.
        Tries multiple search strategies if initial search fails.

        Args:
            company_name: Company/organization name (required)
            validate: Whether to validate the URL format (default True)

        Returns:
            Tuple of (linkedin_url, organization_slug) if found, (None, None) otherwise
        """
        if not self.client:
            logger.warning("Cannot search - Gemini client not initialized")
            return (None, None)

        if not company_name or not company_name.strip():
            logger.warning("Cannot search - company name is required")
            return (None, None)

        # === Validate org name before searching ===
        # Skip invalid org names (generic terms, materials, person names, etc.)
        if not self._is_valid_org_name(company_name):
            logger.info(f"Skipping invalid organization name: '{company_name}'")
            return (None, None)

        start_time = time.time()
        company_name = company_name.strip()
        logger.info(f"Searching LinkedIn for company: {company_name}")

        # Track URLs we validate in Playwright fallback to avoid duplicate work
        seen_company_urls: set[str] = set()

        # Try multiple search strategies
        search_strategies = [
            # Strategy 1: Direct company name search
            f'"{company_name}" site:linkedin.com/company OR site:linkedin.com/school',
            # Strategy 2: Simplified name (remove common suffixes)
            self._get_simplified_search(company_name),
            # Strategy 3: Acronym if applicable
            self._get_acronym_search(company_name),
            # Strategy 4: Try with common suffixes added (Inc, LLC, etc.)
            self._get_suffix_search(company_name),
        ]

        # Remove None strategies
        search_strategies = [s for s in search_strategies if s]

        for i, search_query in enumerate(search_strategies):
            if i > 0:
                logger.info(
                    f"Trying alternative search strategy {i + 1} for: {company_name}"
                )

            url, urn = self._search_with_query(company_name, search_query, validate)
            if url:
                elapsed = time.time() - start_time
                self._timing_stats["company_search"].append(elapsed)
                return (url, urn)

            # Small delay between retries
            if i < len(search_strategies) - 1:
                time.sleep(0.5)

        # Fallback: Try Playwright/Bing search (most reliable)
        try:
            logger.info(f"Trying Playwright/Bing search for: {company_name}")
            url = self._search_company_playwright(company_name, seen_company_urls)
        except Exception:
            logger.exception("Playwright company search failed for %s", company_name)
            url = None

        if url:
            is_valid, slug = self.validate_linkedin_org_url(url)
            if is_valid:
                elapsed = time.time() - start_time
                self._timing_stats["company_search"].append(elapsed)
                return (url, slug)

        # Try Playwright search with acronym if we can generate one
        acronym = self._generate_acronym(company_name)
        if acronym and acronym.upper() != company_name.upper():
            try:
                logger.info(f"Trying Playwright/Bing search with acronym: {acronym}")
                url = self._search_company_playwright(acronym, seen_company_urls)
            except Exception:
                logger.exception(
                    "Playwright company search failed for acronym %s", acronym
                )
                url = None

            if url:
                is_valid, slug = self.validate_linkedin_org_url(url)
                if is_valid:
                    elapsed = time.time() - start_time
                    self._timing_stats["company_search"].append(elapsed)
                    return (url, slug)

        elapsed = time.time() - start_time
        self._timing_stats["company_search"].append(elapsed)
        logger.info(
            f"No LinkedIn company page found for: {company_name} ({elapsed:.1f}s)"
        )
        return (None, None)

    def _generate_acronym(self, company_name: str) -> Optional[str]:
        """Generate an acronym from a multi-word company name."""
        # Known acronyms
        known_acronyms = {
            "Massachusetts Institute of Technology": "MIT",
            "University of California, Santa Barbara": "UCSB",
            "University of California, Los Angeles": "UCLA",
            "University of California, Berkeley": "UC Berkeley",
            "California Institute of Technology": "Caltech",
            "Georgia Institute of Technology": "Georgia Tech",
            "International Union of Pure and Applied Chemistry": "IUPAC",
        }

        if company_name in known_acronyms:
            return known_acronyms[company_name]

        # Generate acronym from multi-word names
        words = company_name.split()
        skip_words = {"of", "the", "and", "for", "in", "on", "at", "to", "a", "an"}
        significant_words = [w for w in words if w.lower() not in skip_words]

        if len(significant_words) >= 3:
            acronym = "".join(w[0].upper() for w in significant_words if w)
            if len(acronym) >= 3:
                return acronym

        return None

    def _get_simplified_search(self, company_name: str) -> Optional[str]:
        """Generate a simplified search query by removing common suffixes."""
        # Common suffixes to remove for cleaner searching
        suffixes_to_remove = [
            " Corporation",
            " Corp",
            " Inc",
            " LLC",
            " Ltd",
            " Limited",
            " GmbH",
            " AG",
            " SE",
            " PLC",
            " of Technology",
            " University",
            " School of Engineering",
            " Samueli School of Engineering",
            " School of",
        ]

        simplified = company_name
        for suffix in suffixes_to_remove:
            if simplified.lower().endswith(suffix.lower()):
                simplified = simplified[: -len(suffix)].strip()

        # Also try removing "University of" prefix
        if simplified.lower().startswith("university of "):
            simplified = simplified[14:].strip()

        if simplified != company_name:
            return f'"{simplified}" linkedin company OR school'
        return None

    def _get_suffix_search(self, company_name: str) -> Optional[str]:
        """Generate search queries by adding common company suffixes.

        Many companies have LinkedIn pages with suffixes like Inc, LLC, etc.
        that don't appear in how they're commonly referred to.
        """
        company_lower = company_name.lower()

        # Check if name already has a suffix
        existing_suffixes = [
            "inc",
            "llc",
            "ltd",
            "corp",
            "corporation",
            "limited",
            "gmbh",
            "plc",
        ]
        has_suffix = any(
            company_lower.endswith(f" {s}") or company_lower.endswith(f", {s}")
            for s in existing_suffixes
        )

        if has_suffix:
            return None

        # Try adding common suffixes - Inc is most common
        return (
            f'"{company_name} Inc" OR "{company_name}, Inc" site:linkedin.com/company'
        )

    def _get_acronym_search(self, company_name: str) -> Optional[str]:
        """Generate an acronym-based search if company name is multi-word."""
        acronym = self._generate_acronym(company_name)
        if acronym:
            return f'"{acronym}" site:linkedin.com/company OR site:linkedin.com/school'
        return None

    def _search_with_query(
        self, company_name: str, search_query: str, validate: bool
    ) -> tuple[Optional[str], Optional[str]]:
        """Execute a search with a specific query."""
        # Note: caller (search_company) guarantees self.client is not None
        assert self.client is not None

        prompt = f"""Find the official LinkedIn page for: {company_name}

Search for: {search_query}

TASK: Find the MAIN LinkedIn company or school page URL.

IMPORTANT RULES:
1. Find the MAIN/OFFICIAL page for the whole organization, NOT a department or subsidiary
2. For universities, look for the main university page, not individual departments
3. The URL must be linkedin.com/company/xxx or linkedin.com/school/xxx
4. Return the most official/verified page if multiple exist
5. If the organization has multiple entities, return the main/parent one

EXAMPLES of what to return:
- For "Stanford University" → https://www.linkedin.com/school/stanford-university
- For "Google" → https://www.linkedin.com/company/google
- For "MIT" → https://www.linkedin.com/school/massachusetts-institute-of-technology

RESPONSE FORMAT:
If found, respond with ONLY the LinkedIn URL on a single line:
https://www.linkedin.com/company/company-name
OR
https://www.linkedin.com/school/school-name

If NOT found or only department pages exist, respond with exactly:
NOT_FOUND"""

        try:
            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_TEXT,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "max_output_tokens": 256,
                },
                endpoint="company_search",
            )

            if not response.text:
                return (None, None)

            result = response.text.strip()

            # Check for NOT_FOUND response
            if "NOT_FOUND" in result.upper():
                return (None, None)

            # Extract LinkedIn company URL from response
            linkedin_url = self._extract_company_url(result)

            if not linkedin_url:
                return (None, None)

            # Validate the URL if requested
            if validate:
                is_valid, org_urn = self.validate_linkedin_org_url(linkedin_url)
                if is_valid:
                    logger.info(
                        f"Found and validated LinkedIn page for {company_name}: {linkedin_url}"
                    )
                    return (linkedin_url, org_urn)
                else:
                    logger.info(
                        f"LinkedIn URL failed validation for {company_name}: {linkedin_url}"
                    )
                    return (None, None)
            else:
                logger.info(
                    f"Found LinkedIn page for {company_name}: {linkedin_url} (not validated)"
                )
                return (linkedin_url, None)

        except Exception as e:
            logger.error(f"Error searching LinkedIn for {company_name}: {e}")
            return (None, None)

    def _extract_company_url(self, text: str) -> Optional[str]:
        """Extract a LinkedIn company/school URL from text."""
        # Pattern for LinkedIn company or school URLs
        pattern = r"https?://(?:www\.)?linkedin\.com/(?:company|school)/[\w\-]+"

        match = re.search(pattern, text)
        if match:
            url = match.group(0)
            # Normalize to https://www.linkedin.com format
            if not url.startswith("https://www."):
                url = url.replace("http://", "https://")
                url = url.replace("https://linkedin.com", "https://www.linkedin.com")
            return url
        return None

    def _extract_person_url(self, text: str) -> Optional[str]:
        """Extract a LinkedIn personal profile URL from text."""
        # Pattern for LinkedIn personal profile URLs (linkedin.com/in/username)
        pattern = r"https?://(?:www\.)?linkedin\.com/in/([\w\-]+)"

        match = re.search(pattern, text)
        if match:
            url = match.group(0)
            # Normalize to https://www.linkedin.com format
            if not url.startswith("https://www."):
                url = url.replace("http://", "https://")
                url = url.replace("https://linkedin.com", "https://www.linkedin.com")
            return url
        return None

    def search_person(
        self,
        name: str,
        company: str,
        position: Optional[str] = None,
        department: Optional[str] = None,
        location: Optional[str] = None,
        role_type: Optional[str] = None,
        research_area: Optional[str] = None,
    ) -> Optional[str]:
        """
        Search for an individual person's LinkedIn profile.

        Uses Playwright with Bing search as the primary method (most reliable),
        with a two-pass strategy: first requiring org match, then without.

        Results are cached to avoid redundant searches for the same person.

        Args:
            name: Person's name (e.g., "Suzanne Farid")
            company: Company or organization they work at (e.g., "UCL")
            position: Optional job title to help identify the right person
            department: Optional department name (e.g., "Chemical Engineering")
            location: Optional location (e.g., "Cambridge, MA, USA")
            role_type: Optional role type (academic, executive, researcher, etc.)
            research_area: Optional research field for academics

        Returns:
            LinkedIn profile URL if found, None otherwise
        """
        if not name or not company:
            return None

        # === Validate person name before searching ===
        # Skip invalid names like "Individual Researcher", "Staff Writer", etc.
        if not self._is_valid_person_name(name):
            logger.info(f"Skipping invalid person name: '{name}'")
            return None

        # Clean optional parameters - filter out empty strings and literal "None"
        position = _clean_optional_string(position)
        department = _clean_optional_string(department)
        location = _clean_optional_string(location)
        role_type = _clean_optional_string(role_type)
        research_area = _clean_optional_string(research_area)

        # Normalize name and company for cache lookups
        # Use normalize_name() to strip titles like "Professor", "Dr.", handle middle initials
        # This ensures "Professor Dmitri Talapin" matches "Dmitri Talapin" in cache
        name_norm = normalize_name(name)
        company_norm = self._normalize_org_name(company)

        # === IMPROVEMENT 3: Check if we already found this person by name ===
        # If we found a profile for this person before (regardless of org), reuse it
        if name_norm in LinkedInCompanyLookup._shared_found_profiles_by_name:
            cached_url = LinkedInCompanyLookup._shared_found_profiles_by_name[name_norm]
            logger.info(f"Person found in name-only cache: {name} -> {cached_url}")
            return cached_url

        # === IMPROVEMENT 2: Use company LinkedIn URL as cache key if available ===
        # This handles "UChicago" vs "University of Chicago" pointing to the same LinkedIn page
        company_cache_key = company_norm
        if company_norm in self._company_cache:
            cached_company = self._company_cache[company_norm]
            if cached_company and cached_company[0]:  # Has URL
                # Use LinkedIn URL as canonical key instead of name
                linkedin_url = cached_company[0]
                if linkedin_url in LinkedInCompanyLookup._shared_company_url_to_name:
                    # Use the canonical name from first lookup
                    company_cache_key = (
                        LinkedInCompanyLookup._shared_company_url_to_name[linkedin_url]
                    )
                else:
                    # First time seeing this company URL - register it
                    LinkedInCompanyLookup._shared_company_url_to_name[linkedin_url] = (
                        company_norm
                    )
                    company_cache_key = company_norm

        # Use simple cache key (name + canonical company) to avoid re-searching same person
        simple_cache_key = f"{name_norm}@{company_cache_key}"

        # Check simple cache first - this catches the same person across multiple stories
        if simple_cache_key in self._person_cache:
            cached_url = self._person_cache[simple_cache_key]
            if cached_url:
                logger.info(f"Person cache hit: {name} at {company} -> {cached_url}")
            else:
                logger.info(
                    f"Person cache hit (not found previously): {name} at {company} - skipping re-search"
                )
            return cached_url

        start_time = time.time()

        # Expand organization abbreviation for search (e.g., "UChicago" -> "University of Chicago")
        search_company = self._expand_org_for_search(company)
        if search_company != company:
            logger.info(f"Expanded org for search: '{company}' -> '{search_company}'")

        # Build enhanced context for logging
        context_parts = [f"{name} at {search_company}"]
        if position:
            context_parts.append(f"({position})")
        if department:
            context_parts.append(f"in {department}")
        if location:
            context_parts.append(f"from {location}")
        logger.info(f"Searching for person LinkedIn profile: {' '.join(context_parts)}")

        # Track URLs we have already validated to avoid reloading the same profile across fallbacks
        seen_urls: set[str] = set()

        # Primary method: Playwright/Bing search with org matching
        profile_url = self._search_person_playwright(
            name,
            location=location,
            company=search_company,
            position=position,
            department=department,
            role_type=role_type,
            research_area=research_area,
            require_org_match=True,
            seen_urls=seen_urls,
        )

        # === REDUCED RETRIES TO AVOID CAPTCHA ===
        # Only do 1 retry instead of 3 to reduce search engine load

        # Fallback 1: Retry without org matching if needed (combines previous retries 1 & 2)
        if not profile_url:
            logger.info(f"Retrying search for {name} without org matching...")
            # Use just name + company (skip department/location that might be wrong)
            profile_url = self._search_person_playwright(
                name,
                company=search_company,
                require_org_match=False,
                seen_urls=seen_urls,
            )

        # NOTE: Removed "last name only" retry - too broad and causes false positives
        # The cache will prevent re-searching for failed lookups

        if profile_url:
            elapsed = time.time() - start_time
            self._timing_stats["person_search"].append(elapsed)
            logger.info(
                f"Found person profile: {name} -> {profile_url} ({elapsed:.1f}s)"
            )
            # Cache the successful result using simple key
            self._person_cache[simple_cache_key] = profile_url
            # Also cache by name only so we can find this person with different org variations
            LinkedInCompanyLookup._shared_found_profiles_by_name[name_norm] = (
                profile_url
            )
            return profile_url

        # Fallback to Gemini search (skip if disabled due to low success rate)
        if self._gemini_disabled:
            elapsed = time.time() - start_time
            self._timing_stats["person_search"].append(elapsed)
            logger.debug(f"Skipping Gemini fallback (disabled) for {name}")
            # Cache the negative result to avoid re-searching
            self._person_cache[simple_cache_key] = None
            return None

        result = self._search_person_gemini(
            name, company, position, department, location, role_type, research_area
        )

        elapsed = time.time() - start_time
        self._timing_stats["person_search"].append(elapsed)

        # Cache the result (found or not found) using simple key
        self._person_cache[simple_cache_key] = result
        # If found, also cache by name only for future lookups with different org variations
        if result:
            LinkedInCompanyLookup._shared_found_profiles_by_name[name_norm] = result
        return result

    def _search_person_gemini(
        self,
        name: str,
        company: str,
        position: Optional[str] = None,
        department: Optional[str] = None,
        location: Optional[str] = None,
        role_type: Optional[str] = None,
        research_area: Optional[str] = None,
    ) -> Optional[str]:
        """
        Fallback: Search for a person's LinkedIn profile using Gemini with Google Search.

        Note: This method is less reliable than Playwright/Bing search.
        Automatically disables itself after 10+ attempts with <10% success rate.
        """
        self._gemini_attempts += 1

        if not self.client:
            return None

        # Build rich context for better matching
        # Only include non-empty values to avoid "None" appearing in prompts
        context_lines = []
        if name and name.strip():
            context_lines.append(f"Name: {name.strip()}")
        if company and company.strip():
            context_lines.append(f"Company/Organization: {company.strip()}")

        if position and position.strip():
            context_lines.append(f"Position/Title: {position.strip()}")
        if department and department.strip():
            context_lines.append(f"Department: {department.strip()}")
        if location and location.strip():
            context_lines.append(f"Location: {location.strip()}")
        if role_type and role_type.strip():
            context_lines.append(f"Role Type: {role_type.strip()}")
        if research_area and research_area.strip():
            context_lines.append(f"Research Area/Field: {research_area.strip()}")

        person_context = "\n".join(context_lines)

        # Build role-specific matching guidance
        matching_guidance = ""
        if role_type == "academic":
            matching_guidance = """
MATCHING TIPS FOR ACADEMIC PROFILE:
- Look for university affiliations and department matches
- Research area/field should align with their expertise
- Academic titles: Professor, Dr., Researcher, Postdoc, PhD
- May list publications or research interests"""
        elif role_type == "executive":
            matching_guidance = """
MATCHING TIPS FOR EXECUTIVE PROFILE:
- Look for C-suite titles or VP/Director positions
- Industry experience should match the company sector
- Senior leadership roles at the mentioned organization"""
        elif role_type == "researcher":
            matching_guidance = """
MATCHING TIPS FOR RESEARCHER PROFILE:
- Look for research-focused titles and affiliations
- Research area should match their published work
- May be affiliated with universities, labs, or R&D divisions"""

        prompt = f"""Find the LinkedIn profile for this specific person:

{person_context}

TASK: Search for the LinkedIn personal profile page for this individual.

CRITICAL MATCHING RULES:
1. Find a linkedin.com/in/username profile URL for THIS SPECIFIC PERSON
2. The person MUST work at or be affiliated with {company}
3. ALL available context (position, department, location) should match
4. Be VERY careful with common names - require multiple matching attributes
5. Do NOT return company pages (linkedin.com/company/...)
6. Do NOT return school pages (linkedin.com/school/...)
7. Only return a profile if you're HIGHLY CONFIDENT it's the right person
{matching_guidance}

CONTRADICTION DETECTION - REJECT if ANY of these are true:
- Profile shows a completely DIFFERENT field of work (e.g., real estate agent when expecting researcher)
- Profile shows a conflicting location (e.g., India when expecting USA, unless recent move indicated)
- Profile shows unrelated industry (e.g., hospitality, retail when expecting engineering/science)
- Profile title/role is fundamentally incompatible (e.g., "Marketing Manager" when expecting "Professor")
- Same name but clearly different person (different photo context, different career entirely)

VERIFICATION CHECKLIST:
- Name matches (including possible variations like Dr., Prof.)
- Organization/company affiliation matches
- Job title/position is consistent with the context
- Location is consistent (if provided)
- Department or field aligns (if provided)
- NO contradictory information present

RESPONSE FORMAT:
If you find their LinkedIn profile with HIGH CONFIDENCE, respond with ONLY the URL like:
https://www.linkedin.com/in/username

If you cannot find their personal LinkedIn profile OR are uncertain OR found contradictory information, respond exactly:
NOT_FOUND"""

        try:
            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_TEXT,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "max_output_tokens": 256,
                },
                endpoint="person_search",
            )

            if not response.text:
                return None

            result = response.text.strip()

            if "NOT_FOUND" in result.upper():
                logger.info(f"Person profile not found: {name} at {company}")
                self._check_gemini_disable()
                return None

            # Extract LinkedIn personal profile URL
            linkedin_url = self._extract_person_url(result)
            if linkedin_url:
                self._gemini_successes += 1
                logger.info(
                    f"Found person profile via Gemini: {name} -> {linkedin_url}"
                )
                return linkedin_url

            self._check_gemini_disable()
            return None

        except Exception as e:
            logger.error(f"Error searching for person {name}: {e}")
            self._check_gemini_disable()
            return None

    def _check_gemini_disable(self) -> None:
        """Check if Gemini fallback should be disabled due to low success rate."""
        if self._gemini_disabled:
            return
        # Only evaluate after 10+ attempts
        if self._gemini_attempts >= 10:
            success_rate = self._gemini_successes / self._gemini_attempts
            if success_rate < 0.10:  # Less than 10% success
                self._gemini_disabled = True
                logger.info(
                    f"Disabling Gemini fallback: {self._gemini_successes}/{self._gemini_attempts} "
                    f"({success_rate * 100:.0f}%) success rate"
                )

    def _search_company_playwright(
        self, company_name: str, seen_urls: Optional[set[str]] = None
    ) -> Optional[str]:
        """
        Search for LinkedIn company page using undetected-chromedriver to render Bing search results.

        Uses shared browser session for efficiency across multiple searches.

        Args:
            company_name: Company name to search for

        Returns:
            LinkedIn company/school URL if found, None otherwise
        """
        try:
            driver = self._get_uc_driver()
            if driver is None:
                return None

            from selenium.webdriver.common.by import By

            # Track already validated URLs to avoid repeated page loads across attempts
            validated_urls: set[str] = set(seen_urls or [])

            # Build search query
            search_query = f"{company_name} LinkedIn company"
            logger.debug(f"UC Chrome Bing company search: {search_query}")

            url = f"https://www.bing.com/search?q={search_query.replace(' ', '+')}"
            driver.get(url)
            time.sleep(2)  # Wait for page to load

            # Prepare company name parts for matching
            company_lower = company_name.lower()
            company_words = [w for w in company_lower.split() if len(w) > 2]

            result_items = driver.find_elements(By.CSS_SELECTOR, ".b_algo")
            for item in result_items:
                try:
                    heading = item.find_element(By.CSS_SELECTOR, "h2")
                    title = heading.text.lower()
                    link = heading.find_element(By.CSS_SELECTOR, "a")
                    href = link.get_attribute("href") or ""

                    # Decode Bing redirect URL
                    u_match = re.search(r"[&?]u=a1([^&]+)", href)
                    if not u_match:
                        continue

                    try:
                        encoded = u_match.group(1)
                        padding = 4 - len(encoded) % 4
                        if padding != 4:
                            encoded += "=" * padding
                        decoded_url = base64.urlsafe_b64decode(encoded).decode("utf-8")
                    except Exception:
                        continue

                    # Check if it's a LinkedIn company or school page
                    if (
                        "linkedin.com/company/" not in decoded_url
                        and "linkedin.com/school/" not in decoded_url
                    ):
                        continue

                    # Verify company name appears in title or URL
                    url_lower = decoded_url.lower()
                    matches = sum(
                        1
                        for word in company_words
                        if word in title or word in url_lower
                    )
                    if matches < len(company_words) * 0.5:
                        logger.debug(f"Skipping '{title[:40]}' - name mismatch")
                        continue

                    # Extract and return the URL
                    match = re.search(
                        r"linkedin\.com/(company|school)/([\w\-]+)", decoded_url
                    )
                    if match:
                        page_type = match.group(1)
                        slug = match.group(2)
                        result_url = f"https://www.linkedin.com/{page_type}/{slug}"
                        if result_url in validated_urls:
                            logger.debug(
                                f"Skipping duplicate company URL already validated: {result_url}"
                            )
                            continue
                        validated_urls.add(result_url)
                        logger.info(f"Found company via UC Chrome: {result_url}")
                        return result_url

                except Exception as e:
                    logger.debug(f"Error processing result item: {e}")
                    continue

            logger.debug("No matching LinkedIn company found via UC Chrome")

        except Exception as e:
            logger.exception(
                "UC Chrome company search error for %s: %s", company_name, e
            )
            # If browser crashed, reset it
            self._uc_driver = None

        return None

    def _validate_profile_name(
        self,
        driver: Any,
        profile_url: str,
        first_name: str,
        last_name: str,
        company: Optional[str] = None,
        first_name_variants: Optional[set[str]] = None,
    ) -> bool:
        """Visit a LinkedIn profile page and validate that the name matches.

        This is the most reliable way to verify we have the correct person,
        as it reads the actual profile name from the page.

        Args:
            driver: Selenium WebDriver instance
            profile_url: LinkedIn profile URL to validate
            first_name: Expected first name (lowercase)
            last_name: Expected last name (lowercase)
            company: Expected company/org (for additional validation)
            first_name_variants: Optional set of nickname variants for first name

        Returns:
            True if the profile name matches, False otherwise
        """
        from selenium.webdriver.common.by import By

        # Use provided variants or generate them
        if first_name_variants is None and first_name:
            first_name_variants = get_name_variants(first_name)

        try:
            driver.get(profile_url)
            time.sleep(1.5)  # Wait for page to load

            # Extract the profile name from the page
            # LinkedIn profile titles are typically "FirstName LastName - Title | LinkedIn"
            page_title = driver.title.lower() if driver.title else ""

            # Also try to get the main profile name heading
            profile_name = ""
            for selector in [
                "h1",
                ".text-heading-xlarge",
                "[data-generated-suggestion-target]",
            ]:
                try:
                    name_elem = driver.find_element(By.CSS_SELECTOR, selector)
                    if name_elem and name_elem.text:
                        profile_name = name_elem.text.lower().strip()
                        break
                except Exception:
                    continue

            # Use profile name if found, otherwise fall back to page title
            name_text = profile_name if profile_name else page_title

            if not name_text:
                logger.debug(f"Could not extract name from profile page: {profile_url}")
                return False

            # Get full page text for context matching (for single-name searches)
            page_text = ""
            try:
                body = driver.find_element(By.TAG_NAME, "body")
                page_text = body.text.lower() if body else ""
            except Exception:
                page_text = page_title

            # Validate: BOTH first and last name must appear in the profile name
            # Use word boundary matching, and allow nickname/prefix variants for first name
            if first_name and last_name:
                # Check last name (should be exact)
                last_pattern = rf"\b{re.escape(last_name)}\b"
                last_match = bool(re.search(last_pattern, name_text))

                # Check first name - use general names_could_match for flexibility
                first_match = bool(
                    re.search(rf"\b{re.escape(first_name)}\b", name_text)
                )
                if not first_match:
                    # Try nickname variants from NICKNAME_MAP
                    if first_name_variants:
                        for variant in first_name_variants:
                            if re.search(rf"\b{re.escape(variant)}\b", name_text):
                                first_match = True
                                logger.debug(
                                    f"First name matched via variant: {first_name} -> {variant}"
                                )
                                break

                    # Try general prefix matching (e.g., "Kam" matches "Kathryn" via shared "Ka" prefix)
                    if not first_match:
                        # Extract first name from profile name text
                        name_parts = name_text.split()
                        if name_parts:
                            profile_first = name_parts[0]
                            if names_could_match(first_name, profile_first):
                                first_match = True
                                logger.debug(
                                    f"First name matched via prefix: {first_name} ~ {profile_first}"
                                )

                if first_match and last_match:
                    logger.debug(
                        f"Profile name validated: '{name_text[:50]}' matches '{first_name} {last_name}'"
                    )
                    # If company validation requested, also check it
                    if company:
                        # Include short company names (like AGC) - use len > 2 instead of > 3
                        company_words = [
                            w.lower()
                            for w in company.split()
                            if len(w) > 2
                            and w.lower()
                            not in ("inc", "llc", "ltd", "corp", "plc", "the")
                        ]
                        # Check both page text AND experience section for company match
                        # This handles people who previously worked at the company
                        company_match = any(word in page_text for word in company_words)

                        # Also explicitly check Experience section for past employment
                        if not company_match:
                            experience_text = self._extract_experience_section(driver)
                            if experience_text:
                                company_match = any(
                                    word in experience_text.lower()
                                    for word in company_words
                                )
                                if company_match:
                                    logger.debug(
                                        f"Company '{company}' found in Experience section (past employment)"
                                    )

                        if not company_match:
                            logger.debug(
                                f"Profile company mismatch: expected '{company}' (words: {company_words}) not found on profile"
                            )
                            return False
                    return True
                else:
                    logger.debug(
                        f"Profile name mismatch: expected '{first_name} {last_name}', found '{name_text[:50]}'"
                    )
                    return False
            elif first_name:
                # Single-name search - check with variants, and require company
                first_match = bool(
                    re.search(rf"\b{re.escape(first_name)}\b", name_text)
                )
                if not first_match and first_name_variants:
                    for variant in first_name_variants:
                        if re.search(rf"\b{re.escape(variant)}\b", name_text):
                            first_match = True
                            break

                if not first_match:
                    logger.debug(
                        f"Profile name mismatch: expected '{first_name}', found '{name_text[:50]}'"
                    )
                    return False

                # For single-name searches, require the company to appear on the profile
                if company:
                    company_words = [w.lower() for w in company.split() if len(w) > 3]
                    company_match = any(word in page_text for word in company_words)
                    # Also check Experience section for past employment
                    if not company_match:
                        experience_text = self._extract_experience_section(driver)
                        if experience_text:
                            company_match = any(
                                word in experience_text.lower()
                                for word in company_words
                            )
                    if not company_match:
                        logger.debug(
                            f"Profile company mismatch: expected '{company}' not found on profile"
                        )
                        return False
                    logger.debug(
                        f"Profile validated (single-name): '{name_text[:30]}' with company '{company}'"
                    )
                return True
            else:
                # No name parts to validate - accept
                return True

        except Exception as e:
            logger.debug(f"Error validating profile name for {profile_url}: {e}")
            return False

    def _extract_experience_section(self, driver: Any) -> str:
        """Extract text from the Experience section of a LinkedIn profile.

        This is used to find past employers that may not appear in the headline.

        Args:
            driver: Selenium WebDriver instance (already on profile page)

        Returns:
            Text content of the Experience section, or empty string if not found
        """
        from selenium.webdriver.common.by import By

        try:
            experience_text = ""

            # Try multiple selectors for the Experience section
            # LinkedIn uses different structures on different pages
            experience_selectors = [
                # Main experience section
                "#experience",
                "[data-section='experience']",
                "section.experience-section",
                # Experience list items
                ".experience-section li",
                "[id*='experience'] li",
                # Alternative: look for "Experience" heading and get sibling content
            ]

            for selector in experience_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for elem in elements:
                        if elem.text:
                            experience_text += " " + elem.text
                except Exception:
                    continue

            # If no experience section found via selectors, try finding by text pattern
            if not experience_text:
                try:
                    # Look for text that starts with "Experience" section marker
                    body = driver.find_element(By.TAG_NAME, "body")
                    full_text = body.text if body else ""

                    # Find the Experience section in the page text
                    # LinkedIn profiles typically have sections in order
                    exp_start = full_text.lower().find("experience")
                    if exp_start != -1:
                        # Get text after "Experience" heading (next ~2000 chars should cover it)
                        experience_text = full_text[exp_start : exp_start + 2000]
                except Exception:
                    pass

            return experience_text.strip()

        except Exception as e:
            logger.debug(f"Error extracting experience section: {e}")
            return ""

    def _search_person_linkedin_direct(
        self,
        driver: Any,
        name: str,
        company: Optional[str] = None,
        position: Optional[str] = None,
        first_name: str = "",
        last_name: str = "",
        first_name_variants: Optional[set] = None,
        original_first_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Search for LinkedIn profile directly on LinkedIn's people search.

        This is more accurate than Google/Bing as we're searching LinkedIn's own index.
        Requires being logged in to LinkedIn.

        Args:
            driver: UC Chrome driver instance
            name: Person's full name
            company: Optional company/organization
            position: Optional job title
            first_name: First name for validation
            last_name: Last name for validation
            first_name_variants: Nickname variants for first name
            original_first_name: For surname-only searches

        Returns:
            LinkedIn profile URL if found, None otherwise
        """
        from selenium.webdriver.common.by import By
        import urllib.parse

        # Build search keywords (normalized to drop suffixes/credentials)
        normalized_query = normalize_name(name)
        search_query = normalized_query if normalized_query else name

        keywords = [search_query]
        if company:
            keywords.append(company)

        search_query = " ".join(keywords)
        encoded_query = urllib.parse.quote(search_query)

        # LinkedIn people search URL
        linkedin_search_url = (
            f"https://www.linkedin.com/search/results/people/?keywords={encoded_query}"
        )

        logger.debug(f"LinkedIn direct search: {search_query}")

        try:
            driver.get(linkedin_search_url)
            time.sleep(3 + random.random() * 2)  # Wait for results to load

            # Check if we need to log in
            current_url = driver.current_url.lower()
            if "/login" in current_url or "/authwall" in current_url:
                logger.debug("LinkedIn login required for direct search")
                if not self._ensure_linkedin_login(driver):
                    return None
                # Retry the search after login
                driver.get(linkedin_search_url)
                time.sleep(3 + random.random() * 2)

            # Human-like behavior
            try:
                driver.execute_script("window.scrollBy(0, 200 + Math.random() * 300);")
                time.sleep(0.5 + random.random())
            except Exception:
                pass

            page_source = driver.page_source

            # Extract profile URLs from LinkedIn search results
            # LinkedIn search results have links like /in/username in the result cards
            profile_urls = []

            # Pattern 1: Direct profile links in search results
            # LinkedIn uses different formats, try multiple patterns
            patterns = [
                r'href="(/in/[\w\-]+)"',  # Relative URL
                r'href="(https://www\.linkedin\.com/in/[\w\-]+)"',  # Absolute URL
                r'href="(https://[a-z]{2}\.linkedin\.com/in/[\w\-]+)"',  # Country-specific
            ]

            for pattern in patterns:
                matches = re.findall(pattern, page_source)
                for match in matches:
                    if match.startswith("/in/"):
                        url = f"https://www.linkedin.com{match}"
                    else:
                        url = match
                    # Clean the URL
                    url = url.split("?")[0].rstrip("/")
                    if url not in profile_urls:
                        profile_urls.append(url)

            if not profile_urls:
                logger.debug(
                    f"LinkedIn direct search: no profile URLs found for '{name}'"
                )
                return None

            logger.debug(
                f"LinkedIn direct search: found {len(profile_urls)} profile URLs"
            )

            # For single-name searches, use original first name for validation
            validation_first = first_name
            validation_last = last_name
            if original_first_name:
                validation_first = normalize_name(original_first_name)
                validation_last = first_name  # The search term is the last name

            # Validate each profile until we find a match
            for profile_url in profile_urls[:5]:  # Check up to 5 results
                # Skip if URL doesn't look like a real profile
                if "/in/ANON" in profile_url or "/in/headless" in profile_url:
                    continue

                # Check if name appears in URL (quick pre-filter)
                url_slug = profile_url.split("/in/")[-1].lower()
                url_parts = url_slug.replace("-", " ").split()

                # Must have at least last name in URL
                if validation_last and validation_last not in url_parts:
                    # Also check if last name is a substring (for concatenated names)
                    if validation_last not in url_slug:
                        continue

                # Full validation on profile page
                validation_company = company if company else None
                if self._validate_profile_name(
                    driver,
                    profile_url,
                    validation_first,
                    validation_last,
                    validation_company,
                    first_name_variants,
                ):
                    logger.info(
                        f"LinkedIn direct search: found {name} -> {profile_url}"
                    )
                    return profile_url
                else:
                    logger.debug(
                        f"LinkedIn direct search: rejected {profile_url} (validation failed)"
                    )

            logger.debug(f"LinkedIn direct search: no valid profile found for '{name}'")
            return None

        except Exception as e:
            logger.debug(f"LinkedIn direct search error: {e}")
            return None

    def _search_person_playwright(
        self,
        name: str,
        location: Optional[str] = None,
        company: Optional[str] = None,
        position: Optional[str] = None,
        department: Optional[str] = None,
        role_type: Optional[str] = None,
        research_area: Optional[str] = None,
        require_org_match: bool = True,
        original_first_name: Optional[str] = None,
        seen_urls: Optional[set[str]] = None,
    ) -> Optional[str]:
        """
        Search for LinkedIn profile using undetected-chromedriver to render Bing search results.

        Uses shared browser session for efficiency across multiple searches.

        Args:
            name: Person's name (may be just last name for fallback searches)
            location: Optional location (city, state, country)
            company: Optional company/organisation
            position: Optional job title
            department: Optional department name
            role_type: Optional role type (academic, executive, researcher)
            research_area: Optional research field for academics
            require_org_match: If True, only return results that match the org.
                             If False, return first result matching the name.
            original_first_name: For surname-only searches, the original first name
                                 to validate against the profile page.

        Returns:
            LinkedIn profile URL if found, None otherwise
        """
        driver = self._get_uc_driver()
        if driver is None:
            return None

        from selenium.webdriver.common.by import By

        # Parse and normalize name early for validation and query construction
        normalized_name = normalize_name(name)
        name_parts = normalized_name.split()
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) >= 2 else ""

        # Name key for per-session rejection cache
        name_key = normalized_name or name.strip().lower()

        # Track already validated URLs to avoid repeated page loads across attempts
        seen_urls = seen_urls or set()

        # For single-name searches, use original_first_name for validation (normalized)
        is_single_name = len(name_parts) == 1
        validation_first_name = first_name
        validation_last_name = last_name
        if is_single_name and original_first_name:
            orig_clean = normalize_name(original_first_name)
            validation_first_name = orig_clean
            validation_last_name = (
                first_name  # The search term is actually the last name
            )

        # Get name variants for fuzzy matching
        first_name_variants = (
            get_name_variants(validation_first_name) if validation_first_name else set()
        )

        # === PRIMARY METHOD: Direct LinkedIn Search ===
        # Try LinkedIn's own search first - more accurate than Google/Bing
        # Only use for full name searches (not surname-only fallback)
        # Can be disabled via SKIP_LINKEDIN_DIRECT_SEARCH to avoid LinkedIn rate limits
        if not Config.SKIP_LINKEDIN_DIRECT_SEARCH and not is_single_name and company:
            logger.info(
                f"LinkedIn direct search for '{normalized_name or name}' at '{company}'"
            )
            linkedin_result = self._search_person_linkedin_direct(
                driver=driver,
                name=normalized_name or name,
                company=company,
                position=position,
                first_name=first_name,
                last_name=last_name,
                first_name_variants=first_name_variants,
                original_first_name=original_first_name,
            )
            if linkedin_result:
                return linkedin_result
            logger.debug(f"LinkedIn direct search failed, falling back to Google/Bing")

        # === FALLBACK: Google/Bing Search ===
        # Build search queries - try quoted first (more precise), then unquoted (more lenient)
        # This helps find profiles where Google doesn't honor exact phrase matching

        # Build quoted query (primary - more precise)
        parts_quoted = []
        # Use normalized name in search queries to avoid noise from suffixes/credentials
        query_name = normalized_name if normalized_name else name.strip()

        if query_name:
            parts_quoted.append(f'"{query_name}"')  # Quote person name for exact match
        if company and company.strip():
            parts_quoted.append(
                f'"{company.strip()}"'
            )  # Quote organization name for exact match
        if location and location.strip():
            location_parts = location.split(",")
            if location_parts and location_parts[0].strip():
                parts_quoted.append(location_parts[0].strip())
        parts_quoted.append("site:linkedin.com/in")
        quoted_query = " ".join(parts_quoted)

        # Build unquoted query (fallback - more lenient)
        parts_unquoted = []
        if query_name:
            parts_unquoted.append(query_name)  # No quotes - allows partial matches
        if company and company.strip():
            parts_unquoted.append(company.strip())  # No quotes
        if location and location.strip():
            location_parts = location.split(",")
            if location_parts and location_parts[0].strip():
                parts_unquoted.append(location_parts[0].strip())
        parts_unquoted.append("site:linkedin.com/in")
        unquoted_query = " ".join(parts_unquoted)

        # Try both query strategies
        search_queries = [
            ("quoted", quoted_query),
            ("unquoted", unquoted_query),
        ]

        try:
            # Rate limiting: respect minimum interval between searches
            import random
            import urllib.parse

            # Try both query strategies (quoted first, then unquoted if no results)
            for query_type, search_query in search_queries:
                logger.debug(f"UC Chrome search ({query_type}): {search_query}")

                current_time = time.time()

                # Check if we're in CAPTCHA cooldown; skip rather than blocking long sleeps
                if current_time < self._captcha_cooldown_until:
                    wait_time = self._captcha_cooldown_until - current_time
                    logger.warning(
                        f"In CAPTCHA cooldown (ends in {wait_time:.0f}s) - skipping browser search"
                    )
                    return None

                # Enforce minimum interval between searches
                time_since_last = current_time - self._last_search_time
                if time_since_last < self._min_search_interval:
                    wait_time = self._min_search_interval - time_since_last
                    # Add randomness to appear more human (3-7 seconds extra)
                    wait_time += 3 + random.random() * 4
                    logger.debug(
                        f"Rate limiting: waiting {wait_time:.1f}s before search"
                    )
                    time.sleep(wait_time)

                # Progressive slowdown: every 3 consecutive searches, add extra delay
                self._consecutive_searches += 1
                if (
                    self._consecutive_searches > 0
                    and self._consecutive_searches % 3 == 0
                ):
                    extra_delay = (
                        12 + random.random() * 8
                    )  # 12-20 seconds extra (was 8-15)
                    logger.info(
                        f"Taking {extra_delay:.0f}s break after {self._consecutive_searches} searches"
                    )
                    time.sleep(extra_delay)

                self._last_search_time = time.time()
                self._driver_search_count += 1  # Track searches for driver rotation

                encoded_query = urllib.parse.quote(search_query)

                # Try Google first (better LinkedIn results), fall back to Bing
                search_engines = [
                    ("Google", f"https://www.google.com/search?q={encoded_query}"),
                    ("Bing", f"https://www.bing.com/search?q={encoded_query}"),
                ]

                page_source = None
                captcha_count = 0
                for engine_name, url in search_engines:
                    driver.get(url)
                    # Random delay to appear more human (4-8 seconds)
                    time.sleep(4 + random.random() * 4)

                    # Human-like behavior: scroll down a bit to simulate reading
                    # Human-like behavior: scroll down randomly and move mouse
                    try:
                        # Scroll down to simulate reading
                        driver.execute_script(
                            "window.scrollBy(0, 200 + Math.random() * 300);"
                        )
                        time.sleep(0.3 + random.random() * 0.5)

                        # Simulate mouse movement by moving focus (UC Chrome handles this better)
                        driver.execute_script("""
                            // Simulate human-like interaction
                            document.dispatchEvent(new MouseEvent('mousemove', {
                                clientX: 100 + Math.random() * 500,
                                clientY: 100 + Math.random() * 300,
                                bubbles: true
                            }));
                        """)
                        time.sleep(0.2 + random.random() * 0.3)

                        # Scroll back up a bit (humans don't just scroll down)
                        driver.execute_script(
                            "window.scrollBy(0, -(50 + Math.random() * 100));"
                        )
                        time.sleep(0.3 + random.random() * 0.4)
                    except Exception:
                        pass

                    # Check for CAPTCHA
                    page_source = driver.page_source.lower()
                    current_url = driver.current_url.lower()

                    # Detect CAPTCHA patterns
                    captcha_indicators = [
                        "captcha" in page_source,
                        "i'm not a robot" in page_source,
                        "unusual traffic" in page_source,
                        "/sorry/" in current_url,
                        "challenge" in current_url and "recaptcha" in page_source,
                    ]

                    if any(captcha_indicators):
                        captcha_count += 1
                        logger.warning(
                            f"{engine_name} CAPTCHA detected, trying next search engine..."
                        )
                        continue

                    # Success - got search results, reset consecutive counter
                    self._consecutive_searches = 0
                    used_engine = engine_name
                    logger.debug(f"Using {engine_name} search results")
                    break
                else:
                    # All search engines blocked - enter cooldown
                    # Keep cooldown bounded so we fail fast instead of blocking the pipeline
                    cooldown_seconds = min(
                        60, 30 * captcha_count
                    )  # Increased cooldown (was 30)
                    self._captcha_cooldown_until = time.time() + cooldown_seconds
                    logger.error(
                        f"All search engines returned CAPTCHA - entering {cooldown_seconds}s cooldown"
                    )
                    return None

                # Variable to track which engine succeeded
                used_engine = locals().get("used_engine", "Unknown")

                # Name already parsed at start of function for LinkedIn direct search
                # Log single-name warning if applicable
                if is_single_name and original_first_name:
                    logger.debug(
                        f"Single-name search: will validate full name '{validation_first_name} {validation_last_name}'"
                    )

                # first_name_variants already computed at start of function

                if is_single_name:
                    logger.warning(
                        f"Single-name search '{name}' - will require strong org/context match"
                    )

                # Check if this is a very common Western name (requires slightly higher confidence)
                name_is_common = is_common_name(first_name, last_name)

                # Build context matching keywords from all available metadata
                context_keywords = build_context_keywords(
                    company=company,
                    department=department,
                    position=position,
                    research_area=research_area,
                )

                # Extract LinkedIn URLs directly from page source
                # Re-fetch page source (don't use the lowercase version from CAPTCHA check)
                page_source = driver.page_source

                # Log page source length and search engine for debugging
                logger.debug(
                    f"{used_engine}: Page source length = {len(page_source)} chars"
                )

                # Try multiple patterns to find LinkedIn URLs
                linkedin_urls = []

                # Pattern 1: Direct LinkedIn URLs (works for Google)
                direct_urls = re.findall(
                    r"https://(?:www\.)?linkedin\.com/in/[\w\-]+/?", page_source
                )
                linkedin_urls.extend(direct_urls)
                logger.debug(
                    f"{used_engine}: Pattern 1 (direct URLs) found {len(direct_urls)} URLs"
                )

                # Pattern 2: URL-encoded redirects (for Bing click-tracking)
                encoded_urls = re.findall(
                    r'href="[^"]*linkedin\.com(?:%2F|/)in(?:%2F|/)([\w\-]+)',
                    page_source,
                )
                for slug in encoded_urls:
                    url = f"https://www.linkedin.com/in/{slug}"
                    if url not in linkedin_urls:
                        linkedin_urls.append(url)
                logger.debug(
                    f"{used_engine}: Pattern 2 (encoded URLs) found {len(encoded_urls)} additional slugs"
                )

                # Pattern 3: Bing wraps URLs in their redirect format
                # e.g., href="https://www.bing.com/ck/a?...&u=a1aHR0cHM6Ly93d3cubGlua2VkaW4uY29tL2luL3VzZXJuYW1l..."
                bing_redirect_urls = re.findall(
                    r"a1aHR0c[A-Za-z0-9+/=]+",
                    page_source,  # Base64-encoded https://
                )
                for b64 in bing_redirect_urls:
                    try:
                        import base64

                        decoded = base64.b64decode(b64 + "==").decode(
                            "utf-8", errors="ignore"
                        )
                        if "linkedin.com/in/" in decoded:
                            # Extract the LinkedIn URL
                            match = re.search(
                                r"https://(?:www\.)?linkedin\.com/in/[\w\-]+", decoded
                            )
                            if match and match.group(0) not in linkedin_urls:
                                linkedin_urls.append(match.group(0))
                    except Exception:
                        pass
                logger.debug(
                    f"{used_engine}: Pattern 3 (Bing base64) found additional URLs, total now {len(linkedin_urls)}"
                )

                # Debug: log how many URLs found
                if not linkedin_urls:
                    logger.debug(
                        f"No LinkedIn URLs found in search results for '{name}'"
                    )
                    # Check if page has any content
                    if len(page_source) < 1000:
                        logger.warning(
                            f"Page source very short ({len(page_source)} chars) - possible load failure"
                        )
                    # Log a snippet of the page source for debugging
                    logger.debug(f"Page source snippet: {page_source[:500]}...")
                else:
                    logger.debug(
                        f"Found {len(linkedin_urls)} LinkedIn URLs in search results"
                    )

                # Deduplicate while preserving order
                seen = set()
                unique_urls = []
                for u in linkedin_urls:
                    u_clean = u.rstrip("/")
                    if u_clean not in seen:
                        seen.add(u_clean)
                        unique_urls.append(u_clean)

                # Track all candidates with their scores
                candidates: List[
                    Tuple[int, str, List[str]]
                ] = []  # (score, url, matched_keywords)

                # For each LinkedIn URL found, try to get context from the search result
                for linkedin_url in unique_urls[:10]:  # Check up to 10 results
                    try:
                        # Find the search result containing this URL
                        # Google results are in divs with class 'g' or we can search for the URL in any element
                        result_text = ""

                        # Try to find result elements containing this URL
                        try:
                            # Look for any element containing this URL
                            elements = driver.find_elements(
                                By.XPATH, f"//*[contains(@href, '{linkedin_url}')]"
                            )
                            for elem in elements:
                                # Get parent container text
                                parent = elem
                                for _ in range(5):  # Go up to 5 levels
                                    try:
                                        parent = parent.find_element(By.XPATH, "..")
                                        parent_text = parent.text
                                        if (
                                            len(parent_text) > len(result_text)
                                            and len(parent_text) < 1000
                                        ):
                                            result_text = parent_text
                                    except Exception:
                                        break
                        except Exception:
                            pass

                        result_text = result_text.lower() if result_text else ""

                        # If no context found from search results, we can still use the URL
                        # The URL slug often contains the person's name
                        url_slug = (
                            linkedin_url.split("/in/")[-1]
                            if "/in/" in linkedin_url
                            else ""
                        )
                        url_text = url_slug.replace("-", " ").lower()
                        url_parts = url_text.split()

                        # === IMPROVED NAME VALIDATION WITH FUZZY MATCHING ===
                        # Check if name appears in URL, but allow for nicknames and variations
                        name_in_url = True
                        name_match_score = 0  # Track how well the name matches
                        first_name_exact = False  # Track if first name match is exact

                        if first_name and last_name:
                            # Check for exact first name match (word boundary)
                            first_in_url = first_name in url_parts
                            first_name_exact = first_in_url  # Exact word match in URL

                            # If not exact, check for nickname/variant match
                            if not first_in_url and first_name_variants:
                                first_in_url = any(
                                    variant in url_parts
                                    for variant in first_name_variants
                                )
                                if first_in_url:
                                    logger.debug(
                                        f"Matched first name variant in URL: {first_name} -> {url_text}"
                                    )
                                    # Variant matches are still considered strong but not exact
                                    first_name_exact = False

                            # Check for last name match (should be exact word match)
                            last_in_url = last_name in url_parts

                            # Also check presence in result text using word boundaries
                            # Use regex to avoid substring matches like "lee" in "jiholee"
                            first_pattern = rf"\b{re.escape(first_name)}\b"
                            last_pattern = rf"\b{re.escape(last_name)}\b"
                            first_in_text = bool(re.search(first_pattern, result_text))
                            if not first_in_text and first_name_variants:
                                first_in_text = any(
                                    bool(re.search(rf"\b{re.escape(v)}\b", result_text))
                                    for v in first_name_variants
                                )
                            last_in_text = bool(re.search(last_pattern, result_text))

                            # Require last name somewhere (URL or text) as word boundary match
                            if not (last_in_url or last_in_text):
                                logger.debug(
                                    f"Skipping '{linkedin_url}' - last name '{last_name}' missing in URL/text"
                                )
                                continue

                            # Require first name (or variant) somewhere (URL or text)
                            if not (first_in_url or first_in_text):
                                logger.debug(
                                    f"Skipping '{linkedin_url}' - first name missing in URL/text"
                                )
                                continue

                            if first_in_url and last_in_url:
                                if first_name_exact:
                                    name_match_score = (
                                        4  # Full exact match - best score
                                    )
                                else:
                                    name_match_score = (
                                        3  # Full match with variant first name
                                    )
                            elif last_in_url:
                                # Last name matches - this is more reliable than first name
                                name_in_url = True  # Consider this acceptable
                                if first_name_exact:
                                    name_match_score = 2
                                else:
                                    name_match_score = (
                                        2  # Last name exact is what matters here
                                    )
                            elif first_in_url:
                                name_in_url = False
                                if first_name_exact:
                                    name_match_score = 1
                                else:
                                    name_match_score = 1
                            else:
                                # Only in text (both names present) - weakest signal
                                name_in_url = False
                                name_match_score = 1

                        elif first_name:
                            # Single word name - check with variants
                            first_in_url = first_name in url_parts
                            if not first_in_url and first_name_variants:
                                first_in_url = any(
                                    variant in url_parts
                                    for variant in first_name_variants
                                )
                            if not first_in_url:
                                # Check result text too
                                if first_name not in result_text:
                                    has_variant = any(
                                        v in result_text for v in first_name_variants
                                    )
                                    if not has_variant:
                                        logger.debug(
                                            f"Skipping '{linkedin_url}' - name not in URL slug: '{url_text}'"
                                        )
                                        continue
                                name_in_url = False

                        # Secondary check: name in search result text (less reliable but helpful)
                        text_to_check = f"{result_text} {url_text}"

                        if first_name and last_name:
                            # If name wasn't fully in URL, check result text more leniently
                            # Allow nickname variants in result text
                            if not name_in_url:
                                first_in_text = first_name in result_text or any(
                                    v in result_text for v in first_name_variants
                                )
                                last_in_text = last_name in result_text
                                if not (first_in_text or last_in_text):
                                    logger.debug(
                                        f"Skipping '{linkedin_url}' - partial URL match but name not in result text"
                                    )
                                    continue
                        elif first_name:
                            # Single word name - be more careful
                            if first_name not in text_to_check:
                                # Also check variants
                                if not any(
                                    v in text_to_check for v in first_name_variants
                                ):
                                    logger.debug(
                                        f"Skipping '{linkedin_url}' - name mismatch"
                                    )
                                    continue

                        # Calculate match score based on context keywords
                        match_score = 0
                        matched_keywords = []
                        for keyword in context_keywords:
                            if keyword in result_text:
                                match_score += 1
                                matched_keywords.append(keyword)

                        # Check org match FIRST (needed for name_in_text validation)
                        org_matched = False
                        if company:
                            company_lower = company.lower()
                            # Include short company names (like AGC) and exclude generic suffixes
                            generic_suffixes = {
                                "inc",
                                "llc",
                                "ltd",
                                "corp",
                                "plc",
                                "the",
                                "co",
                            }
                            company_words = [
                                w
                                for w in company_lower.split()
                                if len(w) > 2 and w not in generic_suffixes
                            ]
                            org_matched = any(
                                word in result_text for word in company_words
                            )
                            org_matched = org_matched or company_lower in result_text

                        # Use the name_match_score from URL validation
                        # (4 = exact full match, 3 = full match with variant first name,
                        #  2 = last name match, 1 = partial/text only)
                        # IMPORTANT: We require at least last_name in URL (score >= 2)
                        # because "name_in_text" often matches the search query itself, not the profile
                        if name_match_score >= 4:
                            match_score += 4  # Best: exact first + last name in URL
                            matched_keywords.append("exact_name_in_url")
                        elif name_match_score == 3:
                            match_score += 3  # Full match with variant first name
                            matched_keywords.append("name_in_url")
                        elif name_match_score == 2:
                            match_score += 2
                            matched_keywords.append("last_name_in_url")
                        elif name_match_score == 1:
                            # Name only in text, not URL - this is very weak evidence
                            # Skip this candidate unless we have strong org match
                            # Also require both first and last name to appear in the result text
                            if first_name and last_name:
                                first_in_text = first_name in result_text or any(
                                    v in result_text for v in first_name_variants
                                )
                                last_in_text = last_name in result_text
                                if not (first_in_text and last_in_text):
                                    logger.debug(
                                        f"Skipping '{linkedin_url}' - names only partially present in text"
                                    )
                                    continue
                            if not org_matched:
                                logger.debug(
                                    f"Skipping '{linkedin_url}' - name only in text (not URL), no org match"
                                )
                                continue
                            match_score += 1
                            matched_keywords.append("name_in_text")

                        if require_org_match and not org_matched:
                            logger.debug(f"Skipping '{linkedin_url}' - org mismatch")
                            continue

                        # Boost score for org match
                        if org_matched:
                            match_score += 2

                        # Boost score for location match
                        if location:
                            location_lower = location.lower()
                            location_parts = [
                                p.strip() for p in location_lower.split(",")
                            ]
                            if any(
                                part in result_text
                                for part in location_parts
                                if len(part) > 2
                            ):
                                match_score += 1

                        # Boost score for role-type specific matches
                        if role_type == "academic":
                            academic_indicators = [
                                "professor",
                                "researcher",
                                "phd",
                                "dr.",
                                "university",
                            ]
                            if any(ind in result_text for ind in academic_indicators):
                                match_score += 1
                        elif role_type == "executive":
                            exec_indicators = [
                                "ceo",
                                "cto",
                                "cfo",
                                "vp",
                                "president",
                                "director",
                                "chief",
                            ]
                            if any(ind in result_text for ind in exec_indicators):
                                match_score += 1

                        # === CONTRADICTION DETECTION ===
                        # Use helper function to calculate penalties for wrong-person indicators
                        search_ctx = PersonSearchContext(
                            first_name=first_name,
                            last_name=last_name,
                            company=company,
                            department=department,
                            position=position,
                            location=location,
                            role_type=role_type,
                            research_area=research_area,
                        )
                        contradiction_penalty, contradiction_reasons = (
                            _calculate_contradiction_penalty(result_text, search_ctx)
                        )

                        # Apply contradiction penalty
                        if contradiction_penalty > 0:
                            match_score -= contradiction_penalty
                            logger.debug(
                                f"Contradiction detected for '{linkedin_url}': {contradiction_reasons}, penalty={contradiction_penalty}"
                            )

                        # Skip if contradictions outweigh matches
                        if match_score < 0:
                            logger.debug(
                                f"Skipping '{linkedin_url}' - contradictions outweigh matches (score={match_score})"
                            )
                            continue

                        # Track all candidates with positive scores (not just best)
                        # We'll validate them in order of score if needed
                        vanity_match = re.search(
                            r"linkedin\.com/in/([\w\-]+)", linkedin_url
                        )
                        if vanity_match:
                            vanity = vanity_match.group(1)
                            candidate_url = f"https://www.linkedin.com/in/{vanity}"
                            rejected_for_name = self._rejected_profile_cache.get(
                                name_key, set()
                            )
                            if candidate_url in rejected_for_name:
                                logger.debug(
                                    f"Skipping rejected candidate for {name_key}: {candidate_url}"
                                )
                                continue
                            if candidate_url in seen_urls:
                                logger.debug(
                                    f"Skipping already validated candidate: {candidate_url}"
                                )
                                continue
                            candidates.append(
                                (match_score, candidate_url, matched_keywords)
                            )
                            logger.debug(
                                f"Candidate: {candidate_url} (score={match_score}, keywords={matched_keywords})"
                            )

                    except Exception as e:
                        logger.debug(f"Error processing result item: {e}")
                        continue

                # Sort candidates by score (highest first)
                candidates.sort(key=lambda x: x[0], reverse=True)

                # Log candidates - concise at INFO, details at DEBUG
                if candidates:
                    top_url = (
                        candidates[0][1].split("/in/")[-1].split("/")[0]
                        if "/in/" in candidates[0][1]
                        else candidates[0][1]
                    )
                    logger.debug(
                        f"Found {len(candidates)} candidates for '{name}' ({query_type}), top: {top_url} (score={candidates[0][0]})"
                    )
                    for i, (score, url, kws) in enumerate(
                        candidates[:5]
                    ):  # Top 5 at DEBUG
                        logger.debug(
                            f"  #{i + 1}: {url} (score={score}, keywords={kws})"
                        )

                # IMPROVED THRESHOLD: Lower thresholds but rely more on final page validation
                # Score breakdown:
                # - Full name in URL: +3 (strong signal)
                # - Last name in URL: +2 (good signal, allows nickname first names)
                # - Name in text only: +1 (weak signal)
                # - Org match: +2
                # - Location match: +1
                # - Role type match: +1
                # - Keyword matches: +1 each
                #
                # We've lowered thresholds because:
                # 1. Nickname matching means we might have +2 (last name) instead of +3
                # 2. Page validation will catch false positives
                # 3. Better to find more candidates and validate than miss profiles
                MIN_CONFIDENCE_SCORE = 2  # Last name + any other signal is acceptable
                MIN_CONFIDENCE_SCORE_COMMON_NAME = (
                    3  # Common names need stronger signals
                )
                MIN_CONFIDENCE_SCORE_SINGLE_NAME = (
                    3  # Single-name: last_name_in_url (+2) + org (+1) is acceptable
                )

                if is_single_name:
                    threshold = MIN_CONFIDENCE_SCORE_SINGLE_NAME
                elif name_is_common:
                    threshold = MIN_CONFIDENCE_SCORE_COMMON_NAME
                else:
                    threshold = MIN_CONFIDENCE_SCORE

                # Try candidates in order of score until we find a valid one
                for candidate_score, candidate_url, matched_kws in candidates:
                    if candidate_score < threshold:
                        logger.debug(
                            f"Remaining candidates below threshold (best: {candidate_url}, score={candidate_score}, threshold={threshold})"
                        )
                        break

                    # Ensure we do not validate the same URL twice across query variants
                    if candidate_url in seen_urls:
                        logger.debug(
                            f"Skipping already validated candidate during final check: {candidate_url}"
                        )
                        continue
                    seen_urls.add(candidate_url)

                    # FINAL VALIDATION: Visit the profile page and verify the name
                    # For lower-confidence matches, also verify company appears on profile
                    # This catches false positives from lowered thresholds
                    # For single-name searches, always require company validation
                    validation_company = (
                        company if is_single_name or candidate_score < 4 else None
                    )
                    # Use validation_first_name/validation_last_name which may differ from search terms
                    # e.g., for surname-only search "choi", we validate "kyoung-ho choi"
                    if self._validate_profile_name(
                        driver,
                        candidate_url,
                        validation_first_name,
                        validation_last_name,
                        validation_company,
                        first_name_variants,
                    ):
                        logger.info(
                            f"Found profile via UC Chrome ({query_type} query): {candidate_url} (score={candidate_score}, threshold={threshold})"
                        )
                        return candidate_url
                    else:
                        logger.debug(
                            f"Rejecting candidate: {candidate_url} (name/company mismatch)"
                        )
                        # Record rejected URL for this name to avoid re-validating this session
                        if name_key:
                            self._rejected_profile_cache.setdefault(
                                name_key, set()
                            ).add(candidate_url)

                if candidates:
                    logger.debug(
                        f"No valid candidate found from {len(candidates)} candidates ({query_type} query)"
                    )
                else:
                    logger.debug(
                        f"No matching LinkedIn profile found via UC Chrome ({query_type} query)"
                    )

                # If quoted query found no results, try unquoted (continue to next iteration)
                if query_type == "quoted" and not candidates:
                    logger.debug(
                        "Quoted search found no results, trying unquoted query..."
                    )
                    continue

        except Exception as e:
            logger.error(f"UC Chrome search error: {e}")
            # If browser crashed, reset it
            self._uc_driver = None

        return None

    def _extract_department_name(self, position: str) -> Optional[str]:
        """Extract department name from a position title if present."""
        if not position:
            return None

        position_lower = position.lower()

        # Common patterns for department positions
        department_patterns = [
            r"(?:head|director|chair|dean|professor)\s+(?:of\s+)?(?:the\s+)?(?:department\s+(?:of|for)\s+)?(.+?)(?:\s+department)?$",
            r"(?:department|dept\.?)\s+(?:of|for)\s+(.+?)(?:\s+(?:head|director|chair|dean))?$",
            r"(.+?)\s+department\s+head",
        ]

        for pattern in department_patterns:
            match = re.search(pattern, position_lower)
            if match:
                dept = match.group(1).strip()
                # Clean up common suffixes
                dept = re.sub(r"\s+at\s+.*$", "", dept)
                # Remove trailing title words
                dept = re.sub(r"\s+(?:head|director|chair|dean)$", "", dept)
                if len(dept) > 3:  # Avoid very short matches
                    return dept.title()

        return None

    def _generate_department_slug_candidates(
        self, department: str, parent_org: str, parent_slug: Optional[str] = None
    ) -> list[str]:
        """Generate likely LinkedIn slug patterns for a department."""
        candidates = []

        # Normalize names
        dept_lower = department.lower().strip()
        org_lower = parent_org.lower().strip()

        # Clean department name for slug (with hyphens)
        dept_slug = re.sub(r"[^a-z0-9\s]", "", dept_lower)
        dept_slug = re.sub(r"\s+", "-", dept_slug.strip())

        # Clean org name for slug (with hyphens)
        org_slug = re.sub(r"[^a-z0-9\s]", "", org_lower)
        org_slug = re.sub(r"\s+", "-", org_slug.strip())

        # No-hyphen versions (e.g., "uclaengineering" instead of "ucla-engineering")
        dept_nohyphen = dept_slug.replace("-", "")
        org_nohyphen = org_slug.replace("-", "")

        # Common patterns - try no-hyphen versions first (more likely for schools)
        # Pattern: orgdept (e.g., uclaengineering)
        candidates.append(f"{org_nohyphen}{dept_nohyphen}")

        # Pattern: org-dept (e.g., ucla-engineering)
        candidates.append(f"{org_slug}-{dept_slug}")

        # If parent_slug provided, try with it
        if parent_slug:
            parent_nohyphen = parent_slug.replace("-", "")
            candidates.append(f"{parent_nohyphen}{dept_nohyphen}")
            candidates.append(f"{parent_slug}-{dept_slug}")

        # Additional patterns
        candidates.append(f"{dept_slug}-{org_slug}")
        candidates.append(f"{org_slug}-{dept_nohyphen}")

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        return unique_candidates

    def search_department(
        self, department: str, parent_org: str, parent_slug: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Search for a department-specific LinkedIn page.

        First tries common URL patterns via LinkedIn API, then falls back to Gemini search.

        Args:
            department: Department name (e.g., "Biochemical Engineering")
            parent_org: Parent organization name (e.g., "UCL")
            parent_slug: Optional parent org slug to help narrow search

        Returns:
            Tuple of (linkedin_url, slug) if found, (None, None) otherwise
        """
        start_time = time.time()

        # Skip department lookup for generic academic departments that rarely have LinkedIn pages
        # This saves 4+ API calls per department that will almost certainly fail
        dept_lower = department.lower()
        skip_patterns = [
            "department of",  # Generic academic department naming
            "graduate school of",  # Graduate programs
            "school of",  # Generic school naming (but keep business schools)
            "faculty of",  # Faculty naming
            "institute of",  # Research institutes (too generic)
            "center for",  # Research centers
            "centre for",  # UK spelling
            "division of",  # Academic divisions
            "college of",  # Generic college naming
        ]

        # Don't skip business schools, engineering schools, etc. that often have LinkedIn pages
        keep_patterns = ["business", "management", "mba", "sloan", "gsb", "engineering"]

        should_skip = any(pattern in dept_lower for pattern in skip_patterns)
        should_keep = any(pattern in dept_lower for pattern in keep_patterns)

        if should_skip and not should_keep:
            logger.debug(
                f"Skipping department lookup (unlikely to have LinkedIn page): {department}"
            )
            return (None, None)

        logger.info(
            f"Searching for department LinkedIn page: {department} at {parent_org}"
        )

        # Strategy 1: Try common URL patterns directly via LinkedIn API
        # Limit to 2 most likely patterns to reduce API calls
        slug_candidates = self._generate_department_slug_candidates(
            department, parent_org, parent_slug
        )[:2]  # Only try top 2 candidates

        for candidate_slug in slug_candidates:
            result = self.lookup_organization_by_vanity_name(candidate_slug)
            if result:
                urn, org_id = result
                if urn:
                    url = f"https://www.linkedin.com/company/{candidate_slug}"
                    elapsed = time.time() - start_time
                    self._timing_stats["department_search"].append(elapsed)
                    logger.info(
                        f"Found department page via API: {department} at {parent_org} -> {url} ({elapsed:.1f}s)"
                    )
                    return (url, candidate_slug)

        # Strategy 2: Fall back to Gemini search
        if not self.client:
            return (None, None)

        prompt = f"""Find the LinkedIn company/school page for this specific department, faculty, or school:

Department/Faculty: {department}
Parent Organization: {parent_org}
{f"Parent LinkedIn slug: {parent_slug}" if parent_slug else ""}

TASK: Search for a LinkedIn page that is specifically for this department, faculty, or school unit - NOT the main parent organization.

SEARCH EXAMPLES:
- "UCL Biochemical Engineering" might have linkedin.com/school/ucl-biochemical-engineering
- "MIT Sloan" might have linkedin.com/school/mit-sloan
- "Stanford Graduate School of Business" might have linkedin.com/school/stanford-gsb

IMPORTANT:
- Many university departments have their own LinkedIn pages
- Look for pages with names like "{parent_org} {department}" or "{department} at {parent_org}"
- The URL format is linkedin.com/company/xxx or linkedin.com/school/xxx
- Do NOT return the main parent organization page (e.g., don't return UCL main page)
- The page name should specifically reference the department or faculty

RESPONSE:
If you find a department-specific LinkedIn page, respond with ONLY the full URL, like:
https://www.linkedin.com/school/ucl-biochemical-engineering

If you cannot find a department-specific page (only the parent org exists), respond with exactly:
NOT_FOUND"""

        try:
            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_TEXT,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "max_output_tokens": 256,
                },
                endpoint="department_search",
            )

            if not response.text:
                return (None, None)

            result = response.text.strip()

            if "NOT_FOUND" in result.upper():
                elapsed = time.time() - start_time
                self._timing_stats["department_search"].append(elapsed)
                return (None, None)

            # Extract LinkedIn URL
            linkedin_url = self._extract_company_url(result)
            if not linkedin_url:
                elapsed = time.time() - start_time
                self._timing_stats["department_search"].append(elapsed)
                return (None, None)

            # Validate the URL format
            is_valid, slug = self.validate_linkedin_org_url(linkedin_url)
            if is_valid:
                elapsed = time.time() - start_time
                self._timing_stats["department_search"].append(elapsed)
                logger.info(
                    f"Found department page via search: {department} at {parent_org} -> {linkedin_url} ({elapsed:.1f}s)"
                )
                return (linkedin_url, slug)

            elapsed = time.time() - start_time
            self._timing_stats["department_search"].append(elapsed)
            return (None, None)

        except Exception as e:
            elapsed = time.time() - start_time
            self._timing_stats["department_search"].append(elapsed)
            logger.error(f"Error searching for department {department}: {e}")
            return (None, None)

    def populate_company_profiles(
        self,
        people: list[dict],
        delay_between_requests: float = 1.0,
    ) -> tuple[int, int, dict[str, tuple[str, Optional[str]]]]:
        """
        Look up LinkedIn profiles for people using a hierarchical approach.

        For each person, tries to find profiles in this order:
        1. Personal LinkedIn profile (linkedin.com/in/username)
        2. Department-specific page (if position indicates a department)
        3. Parent organization page

        Args:
            people: List of person dicts with name, company, position, linkedin_profile
            delay_between_requests: Seconds to wait between API calls (rate limiting)

        Returns:
            Tuple of (companies_found, companies_not_found, company_data_dict)
            company_data_dict maps company names to (url, urn) tuples
        """
        if not self.client:
            logger.warning("Cannot populate profiles - Gemini client not initialized")
            return (0, 0, {})

        # First, look up parent organizations (needed as fallback)
        companies = set()
        for person in people:
            company = person.get("company", "").strip()
            if company:
                companies.add(company)

        if not companies:
            logger.info("No companies found in people list")
            return (0, 0, {})

        companies_found = 0
        companies_not_found = 0
        company_data: dict[str, tuple[str, Optional[str], Optional[str]]] = {}

        # Look up parent organizations first (as fallback), using instance cache
        for i, company in enumerate(companies):
            if company in self._company_cache:
                # Cache hit
                cached = self._company_cache[company]
                if cached:
                    logger.info(f"Company cache hit: {company} -> {cached[0]}")
                    company_data[company] = cached
                    companies_found += 1
                else:
                    logger.debug(f"Company cache hit (not found): {company}")
                    companies_not_found += 1
                continue

            # Cache miss - do the lookup
            linkedin_url, slug = self.search_company(company)

            if linkedin_url:
                # Look up the organization URN via LinkedIn API
                urn = self.lookup_organization_urn(linkedin_url)
                company_data[company] = (linkedin_url, slug, urn)
                self._company_cache[company] = (linkedin_url, slug, urn)
                # Register this company's LinkedIn URL for canonical name mapping
                company_norm = self._normalize_org_name(company)
                if (
                    linkedin_url
                    not in LinkedInCompanyLookup._shared_company_url_to_name
                ):
                    LinkedInCompanyLookup._shared_company_url_to_name[linkedin_url] = (
                        company_norm
                    )
                companies_found += 1
            else:
                self._company_cache[company] = None
                companies_not_found += 1

            # Rate limiting
            if i < len(companies) - 1:
                time.sleep(delay_between_requests)

        # Department cache for fallback searches
        department_cache = self._department_cache

        # Process each person with hierarchical lookup
        for person in people:
            # Skip if already has a profile (from previous run or earlier in this run)
            if person.get("linkedin_profile"):
                logger.debug(f"Skipping {person.get('name')} - already has profile")
                continue

            name = person.get("name", "").strip()
            company = person.get("company", "").strip()
            position = person.get("position", "").strip()
            # Extract enhanced person metadata for better matching
            department = person.get("department", "").strip()
            location = person.get("location", "").strip()
            role_type = person.get("role_type", "").strip()
            research_area = person.get("research_area", "").strip()

            if not company:
                continue

            # ============================================================
            # HIERARCHY LEVEL 1: Try to find the individual's personal profile
            # ============================================================
            if name:
                # search_person handles caching internally now
                # Check cache before logging to avoid misleading "searching" messages
                name_norm = name.lower().strip()
                company_norm = company.lower().strip()
                simple_cache_key = f"{name_norm}@{company_norm}"

                if simple_cache_key in self._person_cache:
                    cached_url = self._person_cache[simple_cache_key]
                    if cached_url:
                        logger.info(
                            f"Level 1: Cache hit for {name} at {company} -> {cached_url}"
                        )
                        person["linkedin_profile"] = cached_url
                        match = re.search(r"linkedin\.com/in/([\w\-]+)", cached_url)
                        if match:
                            person["linkedin_slug"] = match.group(1)
                        person["linkedin_urn"] = None
                        person["linkedin_profile_type"] = "personal"
                        person["match_confidence"] = "high"
                        continue
                    else:
                        logger.debug(
                            f"Level 1: Cache hit (not found previously) for {name} at {company}"
                        )
                        continue  # Skip - we already know this person can't be found

                logger.info(
                    f"Level 1: Searching for personal profile: {name} at {company}"
                )
                person_url = self.search_person(
                    name,
                    company,
                    position=position or None,
                    department=department or None,
                    location=location or None,
                    role_type=role_type or None,
                    research_area=research_area or None,
                )
                time.sleep(delay_between_requests)

                if person_url:
                    person["linkedin_profile"] = person_url
                    # Extract slug from personal profile URL
                    match = re.search(r"linkedin\.com/in/([\w\-]+)", person_url)
                    if match:
                        person["linkedin_slug"] = match.group(1)
                    # Personal profiles don't have organization URNs
                    person["linkedin_urn"] = None
                    person["linkedin_profile_type"] = "personal"
                    # Phase 1: Track match confidence for personal profiles
                    person["match_confidence"] = "high"
                    continue  # Found personal profile, skip to next person

            # ============================================================
            # HIERARCHY LEVEL 2: Try to find department-specific page
            # ============================================================
            # Use actual department field first; only parse from position as fallback
            dept_for_lookup = (
                department if department else self._extract_department_name(position)
            )

            if dept_for_lookup:
                dept_cache_key = f"{dept_for_lookup}@{company}"

                if dept_cache_key in department_cache:
                    # Cache hit
                    cached_dept = department_cache[dept_cache_key]
                    if cached_dept[0]:
                        logger.info(
                            f"Level 2: Cache hit for {dept_for_lookup}@{company} -> {cached_dept[0]}"
                        )
                    else:
                        logger.debug(
                            f"Level 2: Cache hit (not found) for {dept_for_lookup}@{company}"
                        )
                else:
                    # Cache miss - do the lookup
                    # Get parent org slug if available
                    parent_slug = None
                    if company in company_data:
                        parent_slug = company_data[company][1]

                    logger.info(
                        f"Level 2: Looking for department page: {dept_for_lookup} at {company}"
                    )
                    dept_url, dept_slug = self.search_department(
                        dept_for_lookup, company, parent_slug
                    )

                    if dept_url:
                        # Look up URN for department
                        dept_urn = self.lookup_organization_urn(dept_url)
                        department_cache[dept_cache_key] = (
                            dept_url,
                            dept_slug,
                            dept_urn,
                        )
                        time.sleep(delay_between_requests)
                    else:
                        department_cache[dept_cache_key] = (None, None, None)
                        time.sleep(delay_between_requests * 0.5)

                # Use department data if found
                dept_data = department_cache[dept_cache_key]
                if dept_data[0]:  # Department page found
                    person["linkedin_profile"] = dept_data[0]
                    if dept_data[1]:
                        person["linkedin_slug"] = dept_data[1]
                    if dept_data[2]:
                        person["linkedin_urn"] = dept_data[2]
                    person["linkedin_profile_type"] = "department"
                    # Phase 1: Department is medium confidence (not personal profile)
                    person["match_confidence"] = "medium"
                    continue  # Found department, skip to next person

            # ============================================================
            # HIERARCHY LEVEL 3: Fall back to parent organization
            # Phase 1: This is the org_fallback case from spec Step 3.7
            # ============================================================
            if company in company_data:
                url, slug, urn = company_data[company]
                logger.info(
                    f"Level 3: Using organization fallback for {name}: {company}"
                )
                if not person.get("linkedin_profile", "").strip():
                    person["linkedin_profile"] = url
                if slug:
                    person["linkedin_slug"] = slug
                if urn:
                    person["linkedin_urn"] = urn
                person["linkedin_profile_type"] = "organization"
                # Phase 1: Org fallback is explicit - person works here but no personal profile found
                person["match_confidence"] = "org_fallback"
                person["fallback_reason"] = (
                    f"Could not find personal LinkedIn profile for {name}"
                )
            else:
                # ============================================================
                # HIERARCHY LEVEL 4: Nothing found
                # ============================================================
                logger.warning(
                    f"Level 4: No LinkedIn profile found for {name} at {company}"
                )
                person["linkedin_profile"] = None
                person["linkedin_slug"] = None
                person["linkedin_urn"] = None
                person["linkedin_profile_type"] = None
                # Phase 1: Track rejected status for metrics
                person["match_confidence"] = "rejected"
                person["fallback_reason"] = (
                    f"No LinkedIn profile or org page found for {name} at {company}"
                )

        # Return company_data with just (url, slug) for backwards compatibility
        return_data = {k: (v[0], v[1]) for k, v in company_data.items()}
        return (companies_found, companies_not_found, return_data)

    def lookup_person_urn(self, profile_url: str) -> Optional[str]:
        """
        Look up a person's URN from their LinkedIn profile URL.

        LinkedIn embeds the member URN in the profile page source code.
        This method loads the profile page and extracts the URN.

        Uses retry logic with exponential backoff for reliability.

        Args:
            profile_url: LinkedIn profile URL (e.g., https://www.linkedin.com/in/username)

        Returns:
            Person URN (e.g., "urn:li:person:ABC123XYZ") if found, None otherwise
        """
        if not profile_url or "linkedin.com/in/" not in profile_url:
            return None

        # Retry the actual lookup with exponential backoff
        try:
            return self._lookup_person_urn_with_retry(profile_url)
        except Exception as e:
            logger.error(f"All retry attempts failed for {profile_url}: {e}")
            return None

    @with_enhanced_recovery(
        max_attempts=3,
        base_delay=5.0,
        retryable_exceptions=(Exception,),  # Broad retry for network/browser issues
    )
    def _lookup_person_urn_with_retry(self, profile_url: str) -> Optional[str]:
        """Internal method with retry logic for URN lookup."""
        driver = self._get_uc_driver()
        if driver is None:
            logger.warning("Cannot lookup person URN - browser driver not available")
            return None

        # Ensure user is logged in to LinkedIn (required to see URNs)
        if not self._ensure_linkedin_login(driver):
            logger.warning("LinkedIn login required for URN extraction")
            return None

        logger.debug(f"Looking up person URN for: {profile_url}")

        try:
            driver.get(profile_url)
            time.sleep(4)  # Wait for page to fully load

            # Extract the vanity name from URL to help identify the correct profile
            vanity_match = re.search(r"linkedin\.com/in/([\w\-]+)", profile_url)
            vanity_name = vanity_match.group(1) if vanity_match else None

            # LinkedIn embeds the member URN in several places in the page source
            # We need to find the URN that belongs to the PROFILE OWNER, not the viewer

            page_source = driver.page_source

            if vanity_name:
                # BEST PATTERN: Look for memberRelationship URN after the target's publicIdentifier
                # Format: "publicIdentifier":"username",...,"*memberRelationship":"urn:li:fsd_memberRelationship:ID"
                # Use [\s\S] instead of [^{}] to handle nested braces in JSON
                member_rel_pattern = rf'"publicIdentifier":"{re.escape(vanity_name)}"[\s\S]{{0,500}}?"\*memberRelationship":"urn:li:fsd_memberRelationship:([A-Za-z0-9_-]+)"'
                member_match = re.search(member_rel_pattern, page_source)
                if member_match:
                    profile_id = member_match.group(1)
                    urn = f"urn:li:person:{profile_id}"
                    logger.info(f"Found person URN (memberRelationship): {urn}")
                    return urn

                # Alternative: Look for fsd_profile after publicIdentifier
                profile_block_pattern = rf'"publicIdentifier":"{re.escape(vanity_name)}"[\s\S]{{0,500}}?"entityUrn":"urn:li:fsd_profile:([A-Za-z0-9_-]+)"'
                profile_match = re.search(profile_block_pattern, page_source)
                if profile_match:
                    profile_id = profile_match.group(1)
                    urn = f"urn:li:person:{profile_id}"
                    logger.info(f"Found person URN (matched publicIdentifier): {urn}")
                    return urn

                # Alternative: Look for miniProfile with matching publicIdentifier
                mini_pattern = rf'"publicIdentifier":"{re.escape(vanity_name)}"[\s\S]{{0,500}}?"urn":"urn:li:fs_miniProfile:([A-Za-z0-9_-]+)"'
                mini_match = re.search(mini_pattern, page_source)
                if mini_match:
                    profile_id = mini_match.group(1)
                    urn = f"urn:li:person:{profile_id}"
                    logger.info(f"Found person URN (miniProfile with publicId): {urn}")
                    return urn

            # Fallback: Look for the first fsd_profile in a profile context
            # Search specifically in profile-related JSON blocks
            profile_data_match = re.search(
                r'"Profile"[^}]*"entityUrn":"(urn:li:fsd_profile:[A-Za-z0-9_-]+)"',
                page_source,
            )
            if profile_data_match:
                profile_id = profile_data_match.group(1).split(":")[-1]
                urn = f"urn:li:person:{profile_id}"
                logger.info(f"Found person URN (Profile entity): {urn}")
                return urn

            logger.warning(f"Could not extract URN from profile: {profile_url}")
            return None

        except Exception as e:
            logger.error(f"Error looking up person URN for {profile_url}: {e}")
            # Reset browser if it crashed
            self._uc_driver = None
            return None

    def close(self) -> None:
        """Clean up HTTP client and browser resources, and save cache."""
        # Save cache before closing
        self.save_cache_to_disk()

        if self._http_client:
            self._http_client.close()
            self._http_client = None
        self.close_browser()


# Backwards compatibility alias
LinkedInProfileLookup = LinkedInCompanyLookup


# =============================================================================
# TASK 2.1: Robust LinkedIn Profile URN Resolution
# =============================================================================


@dataclass
class URNLookupResult:
    """Result of a URN lookup attempt."""

    urn: Optional[str] = None
    source: str = ""  # "cache", "api", "playwright", "uc_chrome", "google_search"
    success: bool = False
    error: Optional[str] = None
    lookup_time_ms: float = 0.0


@dataclass
class URNCache:
    """Simple in-memory cache for URN lookups."""

    _cache: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _max_size: int = 1000

    def get(self, url: str) -> Optional[str]:
        """Get URN from cache."""
        with self._lock:
            return self._cache.get(url)

    def set(self, url: str, urn: str) -> None:
        """Set URN in cache."""
        with self._lock:
            # Simple LRU: if full, remove oldest entry
            if len(self._cache) >= self._max_size:
                first_key = next(iter(self._cache))
                del self._cache[first_key]
            self._cache[url] = urn

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()


# Type hint for Playwright
if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page, Playwright as PlaywrightSync


# Global URN cache
_urn_cache = URNCache()


class RobustURNResolver:
    """
    Robust LinkedIn URN resolver with multiple fallback strategies.

    TASK 2.1: Implements fallback chain for reliable URN extraction:
    1. Cache lookup (fastest)
    2. API lookup (if LinkedIn API configured)
    3. Playwright browser automation (preferred for reliability)
    4. UC Chrome fallback (if Playwright unavailable)
    5. Google Search (last resort via Gemini)

    Features:
    - Rate-limited concurrent lookups
    - Browser cookie persistence
    - Automatic retry with exponential backoff
    """

    # Rate limiting for LinkedIn: 60 lookups per minute
    _rate_limiter = AdaptiveRateLimiter(
        initial_fill_rate=1.0,  # 1 lookup per second
        capacity=10.0,  # Allow burst of 10
        min_fill_rate=0.5,
        max_fill_rate=2.0,
    )

    def __init__(
        self,
        use_playwright: bool = True,
        use_uc_chrome: bool = True,
        use_gemini: bool = True,
        max_concurrent: int = 3,
    ) -> None:
        """Initialize the robust URN resolver.

        Args:
            use_playwright: Enable Playwright for browser automation
            use_uc_chrome: Enable UC Chrome as fallback
            use_gemini: Enable Gemini/Google Search as last resort
            max_concurrent: Maximum concurrent lookups
        """
        self.use_playwright = use_playwright
        self.use_uc_chrome = use_uc_chrome
        self.use_gemini = use_gemini
        self.max_concurrent = max_concurrent

        # Playwright resources (use Any to avoid type issues with optional Playwright)
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._playwright_lock = threading.Lock()

        # UC Chrome fallback
        self._uc_lookup: Optional[LinkedInCompanyLookup] = None

        # Semaphore for concurrent lookup limiting
        self._semaphore = threading.Semaphore(max_concurrent)

        # Thread pool for concurrent lookups
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def __enter__(self) -> "RobustURNResolver":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Context manager exit."""
        self.close()
        return False

    def _get_playwright_page(self) -> Optional["Page"]:
        """Get or create a Playwright page with persistent session.

        Returns:
            Playwright Page object, or None if Playwright unavailable
        """
        with self._playwright_lock:
            if self._page is not None:
                return self._page

            try:
                from playwright.sync_api import sync_playwright

                # Use persistent context to save cookies/session
                playwright_profile = os.path.expandvars(
                    r"%LOCALAPPDATA%\SocialMediaPublisher\PlaywrightProfile"
                )
                os.makedirs(playwright_profile, exist_ok=True)

                self._playwright = sync_playwright().start()
                self._browser = self._playwright.chromium.launch_persistent_context(
                    playwright_profile,
                    headless=False,  # Visible for LinkedIn auth
                    viewport={"width": 1280, "height": 800},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
                )
                self._page = self._browser.new_page()
                logger.info(
                    f"Created Playwright session with profile: {playwright_profile}"
                )
                return self._page

            except ImportError:
                logger.warning(
                    "Playwright not installed - pip install playwright && playwright install chromium"
                )
                return None
            except Exception as e:
                logger.error(f"Failed to create Playwright session: {e}")
                return None

    def _ensure_linkedin_login_playwright(self, page: "Page") -> bool:
        """Ensure LinkedIn login in Playwright browser."""
        try:
            page.goto("https://www.linkedin.com/feed/", timeout=30000)
            page.wait_for_load_state("networkidle", timeout=10000)

            current_url = page.url

            if "/login" in current_url or "/authwall" in current_url:
                # Try auto-login if credentials available
                if Config.LINKEDIN_USERNAME and Config.LINKEDIN_PASSWORD:
                    logger.info("Attempting Playwright auto-login...")
                    page.goto("https://www.linkedin.com/login", timeout=30000)
                    page.wait_for_load_state("networkidle")

                    page.fill("#username", Config.LINKEDIN_USERNAME)
                    page.fill("#password", Config.LINKEDIN_PASSWORD)
                    page.click('button[type="submit"]')

                    page.wait_for_load_state("networkidle", timeout=30000)

                    if "/feed" in page.url:
                        logger.info("Playwright auto-login successful")
                        return True

                # Manual login fallback
                print("\n" + "=" * 60)
                print("LinkedIn Login Required (Playwright)")
                print("=" * 60)
                print("Please log in to LinkedIn in the browser window.")
                input("\nPress Enter after logging in...")

                page.goto("https://www.linkedin.com/feed/", timeout=30000)
                return "/feed" in page.url

            return True

        except Exception as e:
            logger.error(f"Playwright LinkedIn login error: {e}")
            return False

    def _lookup_via_playwright(self, profile_url: str) -> URNLookupResult:
        """Look up URN using Playwright browser automation."""
        start_time = time.perf_counter()

        page = self._get_playwright_page()
        if page is None:
            return URNLookupResult(
                error="Playwright not available",
                lookup_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        if not self._ensure_linkedin_login_playwright(page):
            return URNLookupResult(
                error="LinkedIn login required",
                lookup_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        try:
            page.goto(profile_url, timeout=30000)
            page.wait_for_load_state("networkidle", timeout=10000)

            # Extract URN from page source
            content = page.content()

            # Extract vanity name for matching
            vanity_match = re.search(r"linkedin\.com/in/([\w\-]+)", profile_url)
            vanity_name = vanity_match.group(1) if vanity_match else None

            if vanity_name:
                # Look for memberRelationship URN
                pattern = rf'"publicIdentifier":"{re.escape(vanity_name)}"[\s\S]{{0,500}}?"\*memberRelationship":"urn:li:fsd_memberRelationship:([A-Za-z0-9_-]+)"'
                match = re.search(pattern, content)
                if match:
                    urn = f"urn:li:person:{match.group(1)}"
                    return URNLookupResult(
                        urn=urn,
                        source="playwright",
                        success=True,
                        lookup_time_ms=(time.perf_counter() - start_time) * 1000,
                    )

                # Fallback: fsd_profile
                pattern = rf'"publicIdentifier":"{re.escape(vanity_name)}"[\s\S]{{0,500}}?"entityUrn":"urn:li:fsd_profile:([A-Za-z0-9_-]+)"'
                match = re.search(pattern, content)
                if match:
                    urn = f"urn:li:person:{match.group(1)}"
                    return URNLookupResult(
                        urn=urn,
                        source="playwright",
                        success=True,
                        lookup_time_ms=(time.perf_counter() - start_time) * 1000,
                    )

            return URNLookupResult(
                error="URN not found in page",
                lookup_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        except Exception as e:
            return URNLookupResult(
                error=str(e),
                lookup_time_ms=(time.perf_counter() - start_time) * 1000,
            )

    def _lookup_via_uc_chrome(self, profile_url: str) -> URNLookupResult:
        """Look up URN using UC Chrome (fallback)."""
        start_time = time.perf_counter()

        if not UC_AVAILABLE:
            return URNLookupResult(
                error="UC Chrome not available",
                lookup_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        try:
            if self._uc_lookup is None:
                self._uc_lookup = LinkedInCompanyLookup(genai_client=None)

            urn = self._uc_lookup.lookup_person_urn(profile_url)

            if urn:
                return URNLookupResult(
                    urn=urn,
                    source="uc_chrome",
                    success=True,
                    lookup_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            else:
                return URNLookupResult(
                    error="URN not found via UC Chrome",
                    lookup_time_ms=(time.perf_counter() - start_time) * 1000,
                )

        except Exception as e:
            return URNLookupResult(
                error=str(e),
                lookup_time_ms=(time.perf_counter() - start_time) * 1000,
            )

    def resolve_person_urn(self, profile_url: str) -> URNLookupResult:
        """
        Resolve a person's URN using the fallback chain.

        Fallback order:
        1. Cache (instant)
        2. Playwright (preferred browser automation)
        3. UC Chrome (fallback)
        4. Returns error if all methods fail

        Args:
            profile_url: LinkedIn profile URL

        Returns:
            URNLookupResult with URN or error details
        """
        if not profile_url or "linkedin.com/in/" not in profile_url:
            return URNLookupResult(error="Invalid LinkedIn profile URL")

        # Normalize URL
        profile_url = profile_url.split("?")[0].rstrip("/")

        # 1. Check cache first
        cached_urn = _urn_cache.get(profile_url)
        if cached_urn:
            return URNLookupResult(
                urn=cached_urn,
                source="cache",
                success=True,
                lookup_time_ms=0.0,
            )

        # Rate limit before browser lookups
        self._rate_limiter.wait(endpoint="urn_lookup")

        with self._semaphore:
            # 2. Try Playwright first (if enabled)
            if self.use_playwright:
                result = self._lookup_via_playwright(profile_url)
                if result.success and result.urn:
                    _urn_cache.set(profile_url, result.urn)
                    self._rate_limiter.on_success(endpoint="urn_lookup")
                    return result

            # 3. Try UC Chrome fallback (if enabled)
            if self.use_uc_chrome:
                result = self._lookup_via_uc_chrome(profile_url)
                if result.success and result.urn:
                    _urn_cache.set(profile_url, result.urn)
                    self._rate_limiter.on_success(endpoint="urn_lookup")
                    return result

        # All methods failed
        return URNLookupResult(error="All lookup methods failed")

    def resolve_multiple_urns(
        self, profile_urls: list[str]
    ) -> dict[str, URNLookupResult]:
        """
        Resolve multiple URNs concurrently with rate limiting.

        Args:
            profile_urls: List of LinkedIn profile URLs

        Returns:
            Dictionary mapping URLs to their lookup results
        """
        results: dict[str, URNLookupResult] = {}

        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_concurrent
            )

        # Submit all lookups
        futures = {
            self._executor.submit(self.resolve_person_urn, url): url
            for url in profile_urls
        }

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                results[url] = future.result(timeout=60)
            except Exception as e:
                results[url] = URNLookupResult(error=str(e))

        return results

    def close(self) -> None:
        """Clean up all resources."""
        # Close Playwright
        with self._playwright_lock:
            if self._page:
                try:
                    self._page.close()
                except Exception:
                    pass
                self._page = None

            if self._browser:
                try:
                    self._browser.close()
                except Exception:
                    pass
                self._browser = None

            if self._playwright:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None

        # Close UC Chrome
        if self._uc_lookup:
            self._uc_lookup.close()
            self._uc_lookup = None

        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None


# Singleton instance for easy access
_robust_resolver: Optional[RobustURNResolver] = None


def get_robust_urn_resolver() -> RobustURNResolver:
    """Get or create the global RobustURNResolver instance."""
    global _robust_resolver
    if _robust_resolver is None:
        _robust_resolver = RobustURNResolver()
    return _robust_resolver


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for linkedin_profile_lookup module."""
    from test_framework import TestSuite

    suite = TestSuite("LinkedIn Profile Lookup Tests")

    def test_validate_linkedin_org_url_valid_company():
        lookup = LinkedInCompanyLookup(genai_client=None)
        valid, slug = lookup.validate_linkedin_org_url(
            "https://www.linkedin.com/company/google"
        )
        assert valid is True
        assert slug == "google"
        lookup.close()

    def test_validate_linkedin_org_url_valid_school():
        lookup = LinkedInCompanyLookup(genai_client=None)
        valid, slug = lookup.validate_linkedin_org_url(
            "https://www.linkedin.com/school/stanford-university"
        )
        assert valid is True
        assert slug == "stanford-university"
        lookup.close()

    def test_validate_linkedin_org_url_invalid():
        lookup = LinkedInCompanyLookup(genai_client=None)
        valid, slug = lookup.validate_linkedin_org_url(
            "https://example.com/not-linkedin"
        )
        assert valid is False
        assert slug is None
        lookup.close()

    def test_validate_linkedin_org_url_empty():
        lookup = LinkedInCompanyLookup(genai_client=None)
        valid, slug = lookup.validate_linkedin_org_url("")
        assert valid is False
        assert slug is None
        lookup.close()

    def test_generate_acronym_long():
        lookup = LinkedInCompanyLookup(genai_client=None)
        # Acronym for 3+ word phrases
        acronym = lookup._generate_acronym(
            "International Business Machines Corporation"
        )
        assert acronym == "IBMC"
        lookup.close()

    def test_generate_acronym_short():
        lookup = LinkedInCompanyLookup(genai_client=None)
        # Short names don't generate acronyms
        acronym = lookup._generate_acronym("MIT")
        assert acronym is None
        lookup.close()

    def test_extract_company_url():
        lookup = LinkedInCompanyLookup(genai_client=None)
        text = "Check out https://www.linkedin.com/company/acme-corp for more info"
        url = lookup._extract_company_url(text)
        assert url == "https://www.linkedin.com/company/acme-corp"
        lookup.close()

    def test_extract_person_url():
        lookup = LinkedInCompanyLookup(genai_client=None)
        text = "Visit https://www.linkedin.com/in/john-doe-test for profile"
        url = lookup._extract_person_url(text)
        assert url is not None
        assert "john-doe-test" in url
        lookup.close()

    def test_lookup_class_init():
        lookup = LinkedInCompanyLookup(genai_client=None)
        # When no genai_client passed and no API key, client is None
        assert lookup._uc_driver is None
        assert lookup._http_client is None
        lookup.close()

    def test_context_manager():
        with LinkedInCompanyLookup(genai_client=None) as lookup:
            assert lookup is not None
        # Should close without error

    def test_lookup_person_urn_invalid_url():
        """Test that invalid URLs return None without error."""
        lookup = LinkedInCompanyLookup(genai_client=None)
        # Test with invalid URL
        result = lookup.lookup_person_urn("https://example.com/not-linkedin")
        assert result is None
        # Test with empty URL
        result = lookup.lookup_person_urn("")
        assert result is None
        lookup.close()

    def test_lookup_person_urn_no_driver():
        """Test that lookup handles gracefully with or without browser driver."""
        lookup = LinkedInCompanyLookup(genai_client=None)
        # Test with valid LinkedIn profile URL (generic test user)
        result = lookup.lookup_person_urn("https://www.linkedin.com/in/john-doe-test")
        # Result could be None (no driver) or a URN string (driver available)
        assert result is None or (
            isinstance(result, str) and result.startswith("urn:li:")
        )
        lookup.close()

    def test_urn_cache():
        """Test URN cache operations."""
        cache = URNCache()
        cache.clear()

        # Test set and get
        cache.set("https://linkedin.com/in/test", "urn:li:person:ABC123")
        result = cache.get("https://linkedin.com/in/test")
        assert result == "urn:li:person:ABC123"

        # Test missing key
        result = cache.get("https://linkedin.com/in/nonexistent")
        assert result is None

    def test_robust_resolver_invalid_url():
        """Test robust resolver with invalid URL."""
        resolver = RobustURNResolver(use_playwright=False, use_uc_chrome=False)
        result = resolver.resolve_person_urn("https://example.com/not-linkedin")
        assert not result.success
        assert result.error == "Invalid LinkedIn profile URL"
        resolver.close()

    suite.add_test(
        "Validate LinkedIn URL - valid company",
        test_validate_linkedin_org_url_valid_company,
    )
    suite.add_test(
        "Validate LinkedIn URL - valid school",
        test_validate_linkedin_org_url_valid_school,
    )
    suite.add_test(
        "Validate LinkedIn URL - invalid", test_validate_linkedin_org_url_invalid
    )
    suite.add_test(
        "Validate LinkedIn URL - empty", test_validate_linkedin_org_url_empty
    )
    suite.add_test("Generate acronym - long phrase", test_generate_acronym_long)
    suite.add_test("Generate acronym - short name", test_generate_acronym_short)
    suite.add_test("Extract company URL from text", test_extract_company_url)
    suite.add_test("Extract person URL from text", test_extract_person_url)
    suite.add_test("Lookup class initialization", test_lookup_class_init)
    suite.add_test("Context manager works", test_context_manager)
    suite.add_test(
        "Lookup person URN - invalid URL", test_lookup_person_urn_invalid_url
    )
    suite.add_test("Lookup person URN - no driver", test_lookup_person_urn_no_driver)
    suite.add_test("URN cache operations", test_urn_cache)
    suite.add_test("Robust resolver - invalid URL", test_robust_resolver_invalid_url)

    return suite
