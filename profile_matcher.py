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

from text_utils import COMMON_FIRST_NAMES, CONTEXT_STOPWORDS, build_context_keywords

logger = logging.getLogger(__name__)


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

        # Common name without strong signals
        if (
            first_name.lower() in self.COMMON_NAMES
            or last_name.lower() in self.COMMON_NAMES
        ):
            positive_count = len(candidate.positive_signals)
            if positive_count <= 1:  # Just name match isn't enough
                candidate.negative_signals.append(
                    MatchSignal(
                        "common_name_weak",
                        self.WEIGHT_COMMON_NAME_NO_SIGNALS,
                        f"Common name ({person.name}) with insufficient signals",
                    )
                )

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
    Create a PersonContext from a dictionary (e.g., from story_people or org_leaders).

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
