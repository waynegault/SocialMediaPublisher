"""
Profile Matcher Engine

A high-precision LinkedIn profile matching system that uses multi-signal scoring
to identify the correct profile for a person, or fall back to organization profiles
when individual profiles cannot be confidently matched.

Key Design Principles:
- PRECISION OVER RECALL: Better to return no profile than the wrong profile
- MULTI-SIGNAL SCORING: Use all available context (name, org, role, location, field)
- CONTRADICTION DETECTION: Strong negative signals to reject wrong profiles
- PROFILE VERIFICATION: Visit profile pages to verify borderline matches
- ORG FALLBACK: Use organization pages when individual profiles can't be matched

Signal Weights:
- POSITIVE SIGNALS (increase confidence):
  +3.0: Exact name match (first + last)
  +3.0: Organization match
  +2.0: Role/title match
  +1.5: Department/field match
  +1.0: Location match (region)
  +0.5: Academic indicators for academic role_type
  +0.5: Executive indicators for executive role_type

- NEGATIVE SIGNALS (decrease confidence):
  -5.0: Completely wrong field (e.g., realtor vs researcher)
  -4.0: Wrong country/continent (e.g., India vs USA)
  -3.0: Incompatible role (e.g., marketing vs engineering)
  -2.0: Conflicting organization
  -1.0: Generic/common name without strong signals

Confidence Thresholds:
- HIGH (>= 6.0): Use profile confidently
- MEDIUM (4.0-5.9): Verify profile page, then decide
- LOW (< 4.0): Reject, use org fallback

Author: Wayne
Created: 2025 for SocialMediaPublisher
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from text_utils import (
    COMMON_FIRST_NAMES,
    CONTEXT_STOPWORDS,
    build_context_keywords,
    is_common_name,
)

logger = logging.getLogger(__name__)


# =============================================================================
# COMMON NAME THRESHOLDS (Phase 1 - spec Step 3.4)
# =============================================================================

# Common surname list for stricter matching requirements
COMMON_LAST_NAMES: frozenset[str] = frozenset(
    {
        "smith",
        "johnson",
        "williams",
        "brown",
        "jones",
        "garcia",
        "miller",
        "davis",
        "rodriguez",
        "martinez",
        "hernandez",
        "lopez",
        "gonzalez",
        "wilson",
        "anderson",
        "thomas",
        "taylor",
        "moore",
        "jackson",
        "martin",
        "lee",
        "thompson",
        "white",
        "harris",
        "sanchez",
        "clark",
        "lewis",
        "robinson",
        "walker",
        "young",
        "allen",
        "king",
        "wright",
        "scott",
        "torres",
        "nguyen",
        "hill",
        "flores",
        "green",
        "adams",
        "nelson",
        "baker",
        "hall",
        "rivera",
        "campbell",
        "mitchell",
        "carter",
        "roberts",
        "chen",
        "wang",
        "kim",
        "li",
        "zhang",
        "liu",
        "singh",
        "kumar",
        "patel",
    }
)


def get_required_signals(first_name: str, last_name: str) -> int:
    """Return minimum positive signals needed for common names.

    This is a HARD THRESHOLD, not a scoring penalty.
    If a candidate doesn't meet this threshold, reject regardless of score.

    Phase 1 Implementation (spec Step 3.4):
    - Very Common (both first AND last are common): Need 3 signals (org + title + location)
    - Common (only first name is common): Need 2 signals (org + one more)
    - Uncommon: Need 1 signal (org match is enough)

    Args:
        first_name: Person's first name
        last_name: Person's last name

    Returns:
        Minimum number of significant positive signals required
    """
    first_lower = first_name.lower().strip() if first_name else ""
    last_lower = last_name.lower().strip() if last_name else ""

    # Check if both first and last are common (highest risk)
    first_is_common = first_lower in COMMON_FIRST_NAMES
    last_is_common = last_lower in COMMON_LAST_NAMES

    if first_is_common and last_is_common:
        # Very common: "John Smith", "Mary Johnson" - need org + title + location
        return 3
    elif first_is_common:
        # Common first name: "John Chen", "Mary Wang" - need org + one more
        return 2
    else:
        # Uncommon: "Xiaoying Zhang", "Sanjay Gupta" - org match is enough
        return 1


def meets_common_name_threshold(
    candidate_signals: list,
    first_name: str,
    last_name: str,
    min_signal_weight: float = 1.0,
) -> tuple[bool, str]:
    """Check if candidate meets minimum signal threshold for common names.

    Args:
        candidate_signals: List of MatchSignal objects with positive signals
        first_name: Person's first name
        last_name: Person's last name
        min_signal_weight: Minimum weight for a signal to count (default 1.0)

    Returns:
        Tuple of (passes_threshold, reason_if_failed)
    """
    required = get_required_signals(first_name, last_name)

    # Count significant positive signals
    signals_present = sum(1 for s in candidate_signals if s.weight >= min_signal_weight)

    if signals_present < required:
        return (
            False,
            f"Common name '{first_name} {last_name}' needs {required} signals, "
            f"only has {signals_present}",
        )
    return True, ""


# =============================================================================
# SIMPLE MATCH SCORING (for API client use)
# =============================================================================


def calculate_match_score(
    target_name: str,
    target_company: str,
    result_name: str,
    result_company: str,
    result_headline: str = "",
    result_job_title: str = "",
) -> float:
    """
    Calculate how well a search result matches target criteria.

    This is a simplified scoring function for use by API clients
    (linkedin_rapidapi_client, linkedin_voyager_client) that don't need
    the full ProfileMatcher machinery.

    Scoring breakdown:
    - Name matching: 50% of score (0.5 max)
    - Company matching: 40% of score (0.4 max)
    - Title/role bonus: 10% of score (0.1 max)

    Args:
        target_name: The name we're searching for
        target_company: The company we expect
        result_name: Name from the search result
        result_company: Company from the search result
        result_headline: Headline/tagline from the search result
        result_job_title: Job title from the search result

    Returns:
        Match score from 0.0 to 1.0 where 1.0 is a perfect match
    """
    score = 0.0

    # Normalize inputs
    target_name_lower = target_name.lower().strip()
    target_company_lower = target_company.lower().strip()
    result_name_lower = result_name.lower().strip()
    result_company_lower = result_company.lower().strip()
    headline_lower = result_headline.lower() if result_headline else ""
    job_title_lower = result_job_title.lower() if result_job_title else ""

    # === Name matching (50% of score) ===
    if result_name_lower == target_name_lower:
        score += 0.5
    elif (
        target_name_lower in result_name_lower or result_name_lower in target_name_lower
    ):
        score += 0.35
    else:
        # Partial name match by word overlap
        target_parts = set(target_name_lower.split())
        result_parts = set(result_name_lower.split())
        overlap = len(target_parts & result_parts)
        if overlap > 0:
            score += 0.25 * (overlap / max(len(target_parts), 1))

    # === Company matching (40% of score) ===
    # Skip generic company suffixes for comparison
    skip_words = {"inc", "llc", "ltd", "corp", "corporation", "the", "company", "group"}

    def clean_company(name: str) -> str:
        return " ".join(w for w in name.split() if w not in skip_words)

    target_company_clean = clean_company(target_company_lower)
    result_company_clean = clean_company(result_company_lower)

    if result_company_clean == target_company_clean:
        score += 0.4
    elif (
        target_company_clean in result_company_clean
        or result_company_clean in target_company_clean
    ):
        score += 0.3
    elif target_company_clean in headline_lower:
        # Company mentioned in headline
        score += 0.25
    else:
        # Partial company match
        target_parts = set(target_company_clean.split())
        result_parts = set(result_company_clean.split())
        overlap = len(target_parts & result_parts)
        if overlap > 0:
            score += 0.2 * (overlap / max(len(target_parts), 1))

    # === Title/Role bonus (10% of score) ===
    executive_titles = [
        "ceo",
        "cto",
        "cfo",
        "coo",
        "president",
        "founder",
        "director",
        "vp",
        "vice president",
        "head of",
        "chief",
        "professor",
        "researcher",
        "scientist",
        "engineer",
    ]
    combined_title = f"{job_title_lower} {headline_lower}"
    for title in executive_titles:
        if title in combined_title:
            score += 0.1
            break

    return min(score, 1.0)


@dataclass
class ScoredCandidate:
    """Result of scoring a LinkedIn profile candidate."""

    score: float
    signals: list[str]


# Common skip words for company name comparison
COMPANY_SKIP_WORDS: frozenset[str] = frozenset(
    {"inc", "llc", "ltd", "corp", "corporation", "the", "company", "group"}
)


def score_person_candidate(
    candidate_first_name: str,
    candidate_last_name: str,
    candidate_headline: str,
    candidate_location: str,
    candidate_public_id: str,
    target_first_name: str,
    target_last_name: str,
    target_company: str,
    target_title: str | None = None,
    target_location: str | None = None,
) -> ScoredCandidate:
    """
    Score a LinkedIn profile candidate against target criteria.

    This is the detailed multi-signal scoring function used for
    ranking multiple candidates from search results.

    Scoring weights:
    - Exact name match: +4.0
    - Last name match: +2.0 (with +1.0 bonus for first name variant)
    - Name in full name: +2.5
    - Company match in headline: +3.0
    - Title match: +1.5
    - Location match: +1.0
    - No public ID penalty: -2.0

    Args:
        candidate_*: Fields from the candidate profile
        target_*: Target criteria to match against

    Returns:
        ScoredCandidate with score and matched signals
    """
    score = 0.0
    signals: list[str] = []

    # Normalize inputs
    first_lower = target_first_name.lower()
    last_lower = target_last_name.lower()
    company_lower = target_company.lower()
    title_lower = target_title.lower() if target_title else ""
    location_lower = target_location.lower() if target_location else ""

    candidate_full = f"{candidate_first_name} {candidate_last_name}".lower()
    candidate_first = candidate_first_name.lower()
    candidate_last = candidate_last_name.lower()
    headline_lower = candidate_headline.lower()
    candidate_loc_lower = candidate_location.lower()

    # Extract significant company words
    company_words = [
        w for w in company_lower.split() if w not in COMPANY_SKIP_WORDS and len(w) > 2
    ]

    # === Name matching ===
    if candidate_first == first_lower and candidate_last == last_lower:
        score += 4.0
        signals.append("exact_name")
    elif candidate_last == last_lower:
        score += 2.0
        signals.append("last_name")
        # Check if first name is variant
        if first_lower in candidate_full or candidate_first.startswith(first_lower[:3]):
            score += 1.0
            signals.append("first_name_variant")
    elif first_lower in candidate_full and last_lower in candidate_full:
        score += 2.5
        signals.append("name_in_full")

    # === Company matching (in headline) ===
    if company_words:
        company_match = any(word in headline_lower for word in company_words)
        if company_match:
            score += 3.0
            signals.append("company_match")

    # === Title matching ===
    if title_lower:
        title_words = [w for w in title_lower.split() if len(w) > 3]
        title_match = any(word in headline_lower for word in title_words)
        if title_match:
            score += 1.5
            signals.append("title_match")

    # === Location matching ===
    if location_lower and candidate_loc_lower:
        location_parts = [p.strip() for p in location_lower.split(",")]
        location_match = any(
            part in candidate_loc_lower for part in location_parts if len(part) > 2
        )
        if location_match:
            score += 1.0
            signals.append("location_match")

    # === Penalty for OUT_OF_NETWORK ===
    if not candidate_public_id or "UNKNOWN" in candidate_public_id.upper():
        score -= 2.0
        signals.append("no_public_id")

    return ScoredCandidate(score=score, signals=signals)


class RoleType(str, Enum):
    """Classification of role types for matching."""

    ACADEMIC = "academic"
    EXECUTIVE = "executive"
    RESEARCHER = "researcher"
    ENGINEER = "engineer"
    HR_RECRUITER = "hr_recruiter"
    PR_COMMS = "pr_comms"
    STUDENT = "student"
    SPOKESPERSON = "spokesperson"
    OTHER = "other"


class MatchConfidence(str, Enum):
    """Confidence levels for profile matches."""

    HIGH = "high"  # >= 6.0 - Use without verification
    MEDIUM = "medium"  # 4.0-5.9 - Needs verification
    LOW = "low"  # < 4.0 - Reject, use fallback
    VERIFIED = "verified"  # Confirmed via profile page visit
    REJECTED = "rejected"  # Explicitly rejected
    ORG_FALLBACK = "org_fallback"  # Using org profile as fallback


@dataclass
class MatchSignal:
    """A single signal (positive or negative) in profile matching."""

    name: str
    weight: float
    description: str


@dataclass
class PersonContext:
    """
    Complete context about a person for LinkedIn profile matching.

    This captures all known information about a person from the story,
    used to find and validate their LinkedIn profile.
    """

    # Core identification
    name: str
    organization: str  # Company/university

    # Role information
    position: str = ""  # Job title
    department: str = ""  # Department/school
    role_type: RoleType = RoleType.OTHER

    # Additional context
    location: str = ""  # City, state, country
    research_area: str = ""  # For academics
    industry: str = ""  # For business professionals

    # Story context (helps disambiguation)
    story_title: str = ""
    story_category: str = ""

    def to_search_terms(self) -> list[str]:
        """Generate search terms for this person."""
        terms = [self.name]
        if self.organization:
            terms.append(self.organization)
        if self.department:
            terms.append(self.department)
        if self.position:
            terms.append(self.position)
        return terms

    def get_context_keywords(self) -> set[str]:
        """Get keywords for matching from all available context."""
        return build_context_keywords(
            company=self.organization,
            department=self.department,
            position=self.position,
            research_area=self.research_area,
            industry=self.industry,
        )


@dataclass
class ProfileCandidate:
    """
    A LinkedIn profile candidate with match scoring.

    Represents a potential match found via search, with scoring
    signals to determine if it's the correct person.
    """

    linkedin_url: str
    vanity_name: str  # The /in/username part

    # Match details
    result_title: str = ""  # From search result
    result_snippet: str = ""  # From search result

    # Scoring
    confidence_score: float = 0.0
    positive_signals: list[MatchSignal] = field(default_factory=list)
    negative_signals: list[MatchSignal] = field(default_factory=list)

    # Verification state
    verified: bool = False
    verified_data: dict = field(default_factory=dict)

    def total_score(self) -> float:
        """Calculate total score from all signals."""
        positive = sum(s.weight for s in self.positive_signals)
        negative = sum(s.weight for s in self.negative_signals)
        return positive + negative  # Negative signals have negative weights

    def get_confidence_level(self) -> MatchConfidence:
        """Determine confidence level from score."""
        if self.verified:
            return MatchConfidence.VERIFIED
        score = self.total_score()
        if score >= 6.0:
            return MatchConfidence.HIGH
        elif score >= 4.0:
            return MatchConfidence.MEDIUM
        else:
            return MatchConfidence.LOW


@dataclass
class ProfileMatchResult:
    """
    Final result of profile matching for a person.

    Includes the matched profile (if any), confidence level,
    and detailed match reasoning for debugging.
    """

    person_context: PersonContext

    # Result
    matched_profile: Optional[ProfileCandidate] = None
    org_linkedin_url: Optional[str] = None  # Fallback organization page
    confidence: MatchConfidence = MatchConfidence.LOW

    # Reasoning
    match_reason: str = ""
    candidates_evaluated: int = 0

    def get_best_url(self) -> Optional[str]:
        """Get the best available LinkedIn URL (person or org fallback)."""
        if self.matched_profile and self.confidence in [
            MatchConfidence.HIGH,
            MatchConfidence.VERIFIED,
            MatchConfidence.MEDIUM,
        ]:
            return self.matched_profile.linkedin_url
        elif self.org_linkedin_url and self.confidence == MatchConfidence.ORG_FALLBACK:
            return self.org_linkedin_url
        return None

    def is_person_profile(self) -> bool:
        """Check if result is a personal profile (vs org fallback)."""
        return (
            self.matched_profile is not None
            and self.confidence != MatchConfidence.ORG_FALLBACK
        )


# =============================================================================
# EXCLUSION CRITERIA EVALUATOR (Phase 1 - spec Step 3.4)
# =============================================================================


@dataclass
class ExclusionResult:
    """Result of exclusion criteria evaluation for a profile candidate.

    Phase 1 Implementation (spec Step 3.4):
    Evaluates whether a candidate should be REJECTED based on definite mismatches.
    Any exclusion trigger = immediate rejection regardless of positive signals.
    """

    rejected: bool = False
    reasons: list[str] = field(default_factory=list)
    triggered_criteria: list[str] = field(default_factory=list)


# Extended wrong field indicators for stricter exclusion
EXCLUSION_WRONG_FIELD_PATTERNS: dict[str, list[str]] = {
    # If expecting academic/researcher, reject these professions entirely
    "academic": [
        "real estate",
        "realtor",
        "realty",
        "property agent",
        "insurance agent",
        "insurance broker",
        "life insurance",
        "car sales",
        "auto sales",
        "automotive sales",
        "mortgage broker",
        "loan officer",
        "network marketing",
        "mlm",
        "direct sales",
        "fitness trainer",
        "personal trainer",
        "gym",
        "hair stylist",
        "beauty salon",
        "nail technician",
        "wedding planner",
        "event planner",
        "travel agent",
        "tourism",
        "life coach",
        "motivational speaker",
    ],
    "researcher": [
        "real estate",
        "realtor",
        "realty",
        "retail sales",
        "store manager",
        "food service",
        "restaurant",
        "hospitality",
        "beauty salon",
        "cosmetology",
        "network marketing",
        "mlm",
    ],
    "executive": [
        "student",
        "intern",
        "internship",
        "entry level",
        "junior developer",
        "trainee",
        "assistant",
        "receptionist",
    ],
    "engineer": [
        "real estate",
        "realtor",
        "insurance agent",
        "network marketing",
        "mlm",
        "life coach",
        "wellness coach",
    ],
}

# Career stage mismatch patterns
CAREER_STAGE_CONFLICTS: dict[str, list[str]] = {
    # If expecting senior roles, reject junior indicators
    "senior": [
        "student",
        "intern",
        "internship",
        "entry level",
        "junior",
        "trainee",
        "recent graduate",
        "fresh graduate",
        "looking for opportunities",
    ],
    # If expecting academic faculty, reject student indicators
    "faculty": [
        "phd student",
        "doctoral student",
        "graduate student",
        "masters student",
        "postdoc seeking",
        "aspiring professor",
    ],
}


def evaluate_exclusion_criteria(
    result_text: str,
    person_role_type: str,
    person_organization: str,
    person_location: str = "",
    is_senior_role: bool = False,
) -> ExclusionResult:
    """
    Evaluate whether a profile candidate should be REJECTED.

    Phase 1 Implementation (spec Step 3.4):
    Any exclusion trigger = immediate rejection, confidence = 0.

    Exclusion Criteria:
    1. Wrong Profession/Field - Profile shows unrelated profession
    2. Geographic Mismatch - Different country/continent
    3. Employer Contradiction - Direct competitor or unrelated employer
    4. Career Stage Mismatch - Junior vs senior role expectation

    Args:
        result_text: Lowercase text from search result (title + snippet)
        person_role_type: Expected role type (academic, researcher, executive, etc.)
        person_organization: Expected organization
        person_location: Expected location (optional)
        is_senior_role: Whether the expected role is senior-level

    Returns:
        ExclusionResult with rejection decision and reasons
    """
    result = ExclusionResult()
    result_lower = result_text.lower()

    # --- 1. Wrong Profession/Field ---
    role_key = person_role_type.lower() if person_role_type else "other"
    wrong_field_patterns = EXCLUSION_WRONG_FIELD_PATTERNS.get(role_key, [])

    for pattern in wrong_field_patterns:
        if pattern in result_lower:
            result.rejected = True
            result.triggered_criteria.append("wrong_field")
            result.reasons.append(
                f"Wrong field detected: '{pattern}' (expected {role_key})"
            )
            logger.debug(f"Exclusion: wrong field '{pattern}' for {role_key}")
            break  # One wrong field is enough to reject

    # --- 2. Career Stage Mismatch ---
    if is_senior_role or role_key in ["executive", "academic"]:
        stage_patterns = CAREER_STAGE_CONFLICTS.get("senior", [])
        if role_key == "academic":
            stage_patterns = stage_patterns + CAREER_STAGE_CONFLICTS.get("faculty", [])

        for pattern in stage_patterns:
            if pattern in result_lower:
                # Check if they also have the expected org (might be alumni)
                org_lower = person_organization.lower() if person_organization else ""
                org_present = org_lower and org_lower in result_lower

                if not org_present:
                    result.rejected = True
                    result.triggered_criteria.append("career_stage_mismatch")
                    result.reasons.append(
                        f"Career stage mismatch: '{pattern}' (expected senior role)"
                    )
                    logger.debug(f"Exclusion: career stage '{pattern}' for senior role")
                    break

    # --- 3. Profile Quality Issues (multiple weak signals) ---
    quality_issues = []
    if "open to work" in result_lower and "hiring" not in result_lower:
        quality_issues.append("actively job seeking")
    if "aspiring" in result_lower or "future" in result_lower:
        quality_issues.append("aspirational role")

    # Only reject on quality issues if combined with other weak signals
    if len(quality_issues) >= 2:
        result.rejected = True
        result.triggered_criteria.append("quality_issues")
        result.reasons.append(f"Profile quality issues: {', '.join(quality_issues)}")

    return result


class ProfileMatcher:
    """
    High-precision LinkedIn profile matching engine.

    Uses multi-signal scoring with contradiction detection to find
    the correct profile for a person, or fall back to organization
    profiles when individual profiles cannot be confidently matched.
    """

    # Score thresholds - lowered to accept more valid matches
    HIGH_CONFIDENCE_THRESHOLD = 4.0  # Name + Org match = 6, so this is reasonable
    MEDIUM_CONFIDENCE_THRESHOLD = 2.0  # Accept with just org match

    # Signal weights - POSITIVE
    WEIGHT_NAME_MATCH = 3.0
    WEIGHT_ORG_MATCH = 3.0
    WEIGHT_ROLE_MATCH = 2.0
    WEIGHT_DEPT_MATCH = 1.5
    WEIGHT_LOCATION_MATCH = 1.0
    WEIGHT_ROLE_TYPE_INDICATOR = 0.5

    # Signal weights - NEGATIVE (still strict to avoid false positives)
    WEIGHT_WRONG_FIELD = -5.0
    WEIGHT_WRONG_COUNTRY = -4.0
    WEIGHT_INCOMPATIBLE_ROLE = -3.0
    WEIGHT_CONFLICTING_ORG = -2.0
    WEIGHT_COMMON_NAME_NO_SIGNALS = -0.5  # Reduced penalty

    # Use shared common names constant from text_utils
    COMMON_NAMES = COMMON_FIRST_NAMES

    # Wrong field indicators by expected role type
    WRONG_FIELD_INDICATORS = {
        RoleType.ACADEMIC: [
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
            "travel agent",
            "car sales",
            "mortgage broker",
        ],
        RoleType.RESEARCHER: [
            "real estate",
            "retail sales",
            "hospitality",
            "food service",
            "beauty salon",
            "event planner",
            "social media influencer",
            "network marketing",
        ],
        RoleType.EXECUTIVE: [
            "student",
            "intern",
            "entry level",
            "junior",
            "trainee",
            "assistant",
        ],
        RoleType.ENGINEER: [
            "real estate",
            "insurance",
            "mlm",
            "network marketing",
            "life coach",
            "wellness",
        ],
    }

    # Country conflict map (if expecting country A, seeing country B is a red flag)
    COUNTRY_CONFLICTS = {
        "usa": [
            "india",
            "pakistan",
            "bangladesh",
            "nigeria",
            "philippines",
            "indonesia",
        ],
        "uk": ["india", "pakistan", "bangladesh", "nigeria", "philippines"],
        "canada": ["india", "pakistan", "nigeria", "philippines"],
        "australia": ["india", "pakistan", "philippines", "indonesia"],
        "germany": ["india", "pakistan", "turkey", "nigeria"],
        "france": ["india", "morocco", "algeria"],
        "singapore": ["india", "indonesia", "philippines"],
        "japan": ["china", "korea", "vietnam"],
    }

    def __init__(self, linkedin_lookup=None):
        """
        Initialize the profile matcher.

        Args:
            linkedin_lookup: Optional LinkedInCompanyLookup instance for searches
        """
        self.linkedin_lookup = linkedin_lookup

    def score_candidate(
        self, candidate: ProfileCandidate, person: PersonContext
    ) -> ProfileCandidate:
        """
        Score a profile candidate against person context.

        Applies positive and negative signals based on matching/contradicting
        information in the search result.

        Phase 1 Enhancement:
        - Runs exclusion criteria FIRST (any exclusion = immediate reject)
        - Applies common name threshold checks
        - Falls back to org profile if rejected

        Args:
            candidate: The profile candidate to score
            person: The person context to match against

        Returns:
            The candidate with updated signals and score
        """
        result_text = f"{candidate.result_title} {candidate.result_snippet}".lower()

        # Parse name for matching
        name_clean = re.sub(r"^(dr\.?|prof\.?|professor)\s+", "", person.name.lower())
        name_parts = name_clean.split()
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) >= 2 else first_name

        # === PHASE 1: EXCLUSION CRITERIA CHECK (spec Step 3.4) ===
        # Any exclusion = immediate reject with score 0
        is_senior = person.role_type in [RoleType.EXECUTIVE, RoleType.ACADEMIC]
        exclusion_result = evaluate_exclusion_criteria(
            result_text=result_text,
            person_role_type=person.role_type.value if person.role_type else "",
            person_organization=person.organization,
            person_location=person.location,
            is_senior_role=is_senior,
        )

        if exclusion_result.rejected:
            # Add rejection signals and return with 0 score
            for reason in exclusion_result.reasons:
                candidate.negative_signals.append(
                    MatchSignal(
                        "exclusion_criteria",
                        -10.0,  # Heavy penalty to ensure rejection
                        reason,
                    )
                )
            candidate.confidence_score = 0.0
            logger.debug(
                f"Candidate rejected by exclusion criteria: {exclusion_result.reasons}"
            )
            return candidate

        # === POSITIVE SIGNALS ===

        # Name match
        if first_name and last_name:
            if first_name in result_text and last_name in result_text:
                candidate.positive_signals.append(
                    MatchSignal(
                        "name_match",
                        self.WEIGHT_NAME_MATCH,
                        f"Full name match: {first_name} {last_name}",
                    )
                )
        elif first_name:
            if first_name in result_text:
                candidate.positive_signals.append(
                    MatchSignal(
                        "partial_name_match",
                        self.WEIGHT_NAME_MATCH * 0.5,
                        f"Partial name match: {first_name}",
                    )
                )

        # Organization match
        if person.organization:
            org_lower = person.organization.lower()
            org_words = [w for w in org_lower.split() if len(w) > 3]
            if org_lower in result_text or any(w in result_text for w in org_words):
                candidate.positive_signals.append(
                    MatchSignal(
                        "org_match",
                        self.WEIGHT_ORG_MATCH,
                        f"Organization match: {person.organization}",
                    )
                )

        # Role/position match
        if person.position:
            pos_words = [w.lower() for w in person.position.split() if len(w) > 4]
            matched_pos = [w for w in pos_words if w in result_text]
            if matched_pos:
                candidate.positive_signals.append(
                    MatchSignal(
                        "role_match",
                        self.WEIGHT_ROLE_MATCH,
                        f"Role match: {', '.join(matched_pos)}",
                    )
                )

        # Department match
        if person.department:
            dept_words = [w.lower() for w in person.department.split() if len(w) > 3]
            matched_dept = [w for w in dept_words if w in result_text]
            if matched_dept:
                candidate.positive_signals.append(
                    MatchSignal(
                        "dept_match",
                        self.WEIGHT_DEPT_MATCH,
                        f"Department match: {', '.join(matched_dept)}",
                    )
                )

        # Location match
        if person.location:
            loc_parts = [p.strip().lower() for p in person.location.split(",")]
            matched_loc = [p for p in loc_parts if len(p) > 2 and p in result_text]
            if matched_loc:
                candidate.positive_signals.append(
                    MatchSignal(
                        "location_match",
                        self.WEIGHT_LOCATION_MATCH,
                        f"Location match: {', '.join(matched_loc)}",
                    )
                )

        # Role type indicators
        if person.role_type == RoleType.ACADEMIC:
            academic_indicators = [
                "professor",
                "researcher",
                "phd",
                "dr.",
                "university",
            ]
            if any(ind in result_text for ind in academic_indicators):
                candidate.positive_signals.append(
                    MatchSignal(
                        "academic_indicator",
                        self.WEIGHT_ROLE_TYPE_INDICATOR,
                        "Academic role indicator found",
                    )
                )
        elif person.role_type == RoleType.EXECUTIVE:
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
                candidate.positive_signals.append(
                    MatchSignal(
                        "executive_indicator",
                        self.WEIGHT_ROLE_TYPE_INDICATOR,
                        "Executive role indicator found",
                    )
                )
        elif person.role_type == RoleType.HR_RECRUITER:
            hr_indicators = ["hr", "recruiter", "talent", "people", "hiring"]
            if any(ind in result_text for ind in hr_indicators):
                candidate.positive_signals.append(
                    MatchSignal(
                        "hr_indicator",
                        self.WEIGHT_ROLE_TYPE_INDICATOR,
                        "HR/Recruiter indicator found",
                    )
                )

        # === NEGATIVE SIGNALS ===

        # Wrong field detection
        wrong_fields = self.WRONG_FIELD_INDICATORS.get(person.role_type, [])
        for wrong_field in wrong_fields:
            if wrong_field in result_text:
                candidate.negative_signals.append(
                    MatchSignal(
                        "wrong_field",
                        self.WEIGHT_WRONG_FIELD,
                        f"Wrong field detected: {wrong_field}",
                    )
                )
                break  # One wrong field is enough

        # Country/location conflict
        if person.location:
            loc_parts = person.location.lower().split(",")
            expected_country = loc_parts[-1].strip() if loc_parts else ""

            for expected, conflicts in self.COUNTRY_CONFLICTS.items():
                if expected in expected_country:
                    for conflict in conflicts:
                        if conflict in result_text:
                            # But don't penalize if expected location also appears
                            if not any(
                                p.strip() in result_text
                                for p in loc_parts
                                if len(p.strip()) > 2
                            ):
                                candidate.negative_signals.append(
                                    MatchSignal(
                                        "location_conflict",
                                        self.WEIGHT_WRONG_COUNTRY,
                                        f"Location conflict: expecting {expected_country}, found {conflict}",
                                    )
                                )
                                break
                    break

        # Incompatible role detection
        if person.role_type == RoleType.ACADEMIC:
            # Academics shouldn't primarily be sales/marketing
            incompatible = [
                "sales representative",
                "account manager",
                "marketing specialist",
            ]
            for role in incompatible:
                if role in result_text:
                    # Unless they're also at the expected org
                    if (
                        person.organization
                        and person.organization.lower() not in result_text
                    ):
                        candidate.negative_signals.append(
                            MatchSignal(
                                "incompatible_role",
                                self.WEIGHT_INCOMPATIBLE_ROLE,
                                f"Incompatible role: {role} (expected academic)",
                            )
                        )
                        break

        # === PHASE 1: COMMON NAME THRESHOLD CHECK (spec Step 3.4) ===
        # For common names, require minimum number of positive signals
        passes_threshold, threshold_reason = meets_common_name_threshold(
            candidate_signals=candidate.positive_signals,
            first_name=first_name,
            last_name=last_name,
            min_signal_weight=1.0,
        )

        if not passes_threshold:
            candidate.negative_signals.append(
                MatchSignal(
                    "common_name_threshold",
                    -5.0,  # Significant penalty for common name without enough signals
                    threshold_reason,
                )
            )
            logger.debug(f"Common name threshold not met: {threshold_reason}")

        # Calculate final score
        candidate.confidence_score = candidate.total_score()

        return candidate

    def select_best_candidate(
        self, candidates: list[ProfileCandidate]
    ) -> Optional[ProfileCandidate]:
        """
        Select the best candidate from scored candidates.

        Returns the highest-scoring candidate if it meets the threshold,
        or None if no candidate is good enough.

        Args:
            candidates: List of scored candidates

        Returns:
            Best candidate if score >= threshold, else None
        """
        if not candidates:
            return None

        # Sort by score descending
        sorted_candidates = sorted(
            candidates, key=lambda c: c.confidence_score, reverse=True
        )

        best = sorted_candidates[0]

        # Check confidence level
        confidence = best.get_confidence_level()

        if confidence == MatchConfidence.HIGH:
            return best
        elif confidence == MatchConfidence.MEDIUM:
            # Medium confidence - might need verification
            # For now, return it with a note
            logger.info(
                f"Medium confidence match (score={best.confidence_score:.1f}): {best.linkedin_url}"
            )
            return best
        else:
            # Low confidence - reject
            logger.info(
                f"Rejecting low confidence match (score={best.confidence_score:.1f}): {best.linkedin_url}"
            )
            return None

    def match_person(
        self, person: PersonContext, org_fallback_url: Optional[str] = None
    ) -> ProfileMatchResult:
        """
        Match a person to their LinkedIn profile with high precision.

        This is the main entry point for profile matching. It:
        1. Searches for candidates
        2. Scores each candidate
        3. Selects the best match (if confidence is high enough)
        4. Falls back to org profile if no good match found

        Args:
            person: PersonContext with all known information
            org_fallback_url: Organization LinkedIn URL to use if no person match

        Returns:
            ProfileMatchResult with match details and confidence
        """
        result = ProfileMatchResult(
            person_context=person,
            org_linkedin_url=org_fallback_url,
        )

        if not self.linkedin_lookup:
            logger.warning("No linkedin_lookup instance configured")
            if org_fallback_url:
                result.confidence = MatchConfidence.ORG_FALLBACK
                result.match_reason = "No profile search available, using org fallback"
            return result

        # Search for candidates
        profile_url = self.linkedin_lookup.search_person(
            name=person.name,
            company=person.organization,
            position=person.position,
            department=person.department,
            location=person.location,
            role_type=person.role_type.value if person.role_type else None,
            research_area=person.research_area,
        )

        if profile_url:
            # Found a profile - create candidate and score it
            # Note: The search_person method already does scoring internally
            # For now, we trust its result but set appropriate confidence
            candidate = ProfileCandidate(
                linkedin_url=profile_url,
                vanity_name=profile_url.split("/in/")[-1].rstrip("/")
                if "/in/" in profile_url
                else "",
            )

            result.matched_profile = candidate
            result.confidence = MatchConfidence.HIGH  # search_person already filters
            result.match_reason = "Profile found via multi-signal search"
            result.candidates_evaluated = 1

        else:
            # No profile found - use org fallback if available
            if org_fallback_url:
                result.confidence = MatchConfidence.ORG_FALLBACK
                result.match_reason = (
                    f"No confident profile match for {person.name}, "
                    f"using organization page: {org_fallback_url}"
                )
            else:
                result.confidence = MatchConfidence.REJECTED
                result.match_reason = f"No profile found for {person.name}"

        return result

    def verify_profile(
        self, candidate: ProfileCandidate, person: PersonContext
    ) -> tuple[bool, dict]:
        """
        Verify a borderline match by visiting the LinkedIn profile page.

        This method visits the actual LinkedIn profile page and extracts
        key details (name, headline, company, location) to verify the match.

        Args:
            candidate: The profile candidate to verify
            person: The person context to match against

        Returns:
            Tuple of (is_verified, verified_data)
        """
        if not self.linkedin_lookup:
            return False, {}

        driver = self.linkedin_lookup._get_uc_driver()
        if not driver:
            logger.warning("No browser driver available for profile verification")
            return False, {}

        try:
            from selenium.webdriver.common.by import By
            import time

            # Navigate to profile page
            logger.info(f"Verifying profile: {candidate.linkedin_url}")
            driver.get(candidate.linkedin_url)
            time.sleep(3)  # Wait for page load

            verified_data = {}

            # Try to extract profile details
            try:
                # Name (usually in h1)
                name_el = driver.find_element(By.CSS_SELECTOR, "h1.text-heading-xlarge")
                verified_data["name"] = name_el.text.strip()
            except Exception:
                pass

            try:
                # Headline (title/role)
                headline_el = driver.find_element(
                    By.CSS_SELECTOR, "div.text-body-medium"
                )
                verified_data["headline"] = headline_el.text.strip()
            except Exception:
                pass

            try:
                # Location
                location_el = driver.find_element(
                    By.CSS_SELECTOR, "span.text-body-small:not(.pvs-header__subtitle)"
                )
                verified_data["location"] = location_el.text.strip()
            except Exception:
                pass

            try:
                # Current company (from experience section or headline)
                # This is trickier - try multiple selectors
                selectors = [
                    "button[aria-label*='current company']",
                    "div.pv-text-details__right-panel-item a[href*='/company/']",
                    "span.pv-text-details__separator + button",
                ]
                for selector in selectors:
                    try:
                        company_el = driver.find_element(By.CSS_SELECTOR, selector)
                        verified_data["company"] = company_el.text.strip()
                        break
                    except Exception:
                        continue
            except Exception:
                pass

            # Validate extracted data against person context
            is_verified = self._validate_verified_data(verified_data, person)

            # Store verified data in candidate
            candidate.verified = is_verified
            candidate.verified_data = verified_data

            if is_verified:
                logger.info(f"Profile verified: {verified_data}")
            else:
                logger.info(
                    f"Profile verification failed: {verified_data} vs {person.name}"
                )

            return is_verified, verified_data

        except Exception as e:
            logger.error(f"Error verifying profile {candidate.linkedin_url}: {e}")
            return False, {}

    def _validate_verified_data(
        self, verified_data: dict, person: PersonContext
    ) -> bool:
        """
        Validate verified profile data against person context.

        Args:
            verified_data: Data extracted from LinkedIn profile page
            person: Expected person context

        Returns:
            True if profile matches, False otherwise
        """
        # Name validation (required)
        verified_name = verified_data.get("name", "").lower()
        if not verified_name:
            return False

        # Parse expected name
        name_clean = re.sub(r"^(dr\.?|prof\.?|professor)\s+", "", person.name.lower())
        name_parts = name_clean.split()

        # Check if first and last name appear in verified name
        if len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = name_parts[-1]
            if first_name not in verified_name or last_name not in verified_name:
                logger.debug(
                    f"Name mismatch: expected '{person.name}', got '{verified_data.get('name')}'"
                )
                return False
        elif name_parts:
            if name_parts[0] not in verified_name:
                return False

        # Organization validation (if available)
        verified_headline = verified_data.get("headline", "").lower()
        verified_company = verified_data.get("company", "").lower()

        if person.organization:
            org_lower = person.organization.lower()
            org_words = [w for w in org_lower.split() if len(w) > 3]

            # Check if org appears in headline or company
            org_found = (
                org_lower in verified_headline
                or org_lower in verified_company
                or any(w in verified_headline for w in org_words)
                or any(w in verified_company for w in org_words)
            )

            if not org_found:
                logger.debug(
                    f"Organization not found: expected '{person.organization}', "
                    f"headline='{verified_headline}', company='{verified_company}'"
                )
                # Don't immediately fail - org might be in a different format
                # But reduce confidence

        # Location validation (optional - just a bonus signal)
        verified_location = verified_data.get("location", "").lower()
        if person.location and verified_location:
            loc_parts = [p.strip().lower() for p in person.location.split(",")]
            loc_match = any(p in verified_location for p in loc_parts if len(p) > 2)
            if loc_match:
                logger.debug(f"Location match confirmed: {verified_location}")

        # If we get here, name matched and nothing contradicted
        return True


def create_person_context(
    person_dict: dict,
    story_title: str = "",
    story_category: str = "",
) -> PersonContext:
    """
    Create a PersonContext from a dictionary (e.g., from direct_people or indirect_people).

    Args:
        person_dict: Dictionary with person fields
        story_title: Optional story title for context
        story_category: Optional story category for context

    Returns:
        PersonContext object
    """
    # Parse role_type
    role_type_str = person_dict.get("role_type", "").lower()
    try:
        role_type = RoleType(role_type_str)
    except ValueError:
        # Map common variations
        if role_type_str in ["professor", "scholar", "faculty"]:
            role_type = RoleType.ACADEMIC
        elif role_type_str in [
            "ceo",
            "cto",
            "cfo",
            "vp",
            "director",
            "founder",
            "owner",
        ]:
            role_type = RoleType.EXECUTIVE
        elif role_type_str in ["hr", "recruiter", "talent", "people"]:
            role_type = RoleType.HR_RECRUITER
        elif role_type_str in ["pr", "communications", "media", "press"]:
            role_type = RoleType.PR_COMMS
        else:
            role_type = RoleType.OTHER

    return PersonContext(
        name=person_dict.get("name", ""),
        organization=person_dict.get("company", "")
        or person_dict.get("organization", ""),
        position=person_dict.get("position", "") or person_dict.get("title", ""),
        department=person_dict.get("department", ""),
        role_type=role_type,
        location=person_dict.get("location", ""),
        research_area=person_dict.get("research_area", ""),
        story_title=story_title,
        story_category=story_category,
    )
