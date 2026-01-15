"""Company mention enrichment using AI for LinkedIn posts.

This module identifies real companies explicitly mentioned in news sources and adds them
to posts as professional, analytical context. It is conservative by design - when in doubt,
it defaults to NO_COMPANY_MENTION.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from google import genai  # type: ignore
from openai import OpenAI
import requests
import re

from api_client import api_client
from text_utils import strip_markdown_code_block
from config import Config
from database import Database, Story

# Import shared entity validation constants from entity_constants (no optional dependencies)
from entity_constants import (
    INVALID_ORG_NAMES,
    INVALID_ORG_PATTERNS,
    VALID_SINGLE_WORD_ORGS,
    INVALID_PERSON_NAMES,
    INVALID_PERSON_PATTERNS,
    is_invalid_org_name,
    is_invalid_person_name,
)

# Patterns that look like organization names rather than person names
ORG_NAME_PATTERNS: list[str] = [
    "university",
    "college",
    "institute",
    "school",
    "academy",
    "laboratory",
    "lab",
    "center",
    "centre",
    "corporation",
    "company",
    "inc",
    "llc",
    "ltd",
    "group",
    "foundation",
    "society",
    "association",
    "department",
    "ministry",
    "government",
    "state",  # "Penn State", "State University"
]

if TYPE_CHECKING:
    from linkedin_profile_lookup import LinkedInCompanyLookup
    from linkedin_voyager_client import HybridLinkedInLookup

logger = logging.getLogger(__name__)

# Exact string that indicates no company mention should be added
NO_COMPANY_MENTION = "NO_COMPANY_MENTION"


# =============================================================================
# Phase 4: Enrichment Metrics and Data Structures
# =============================================================================


@dataclass
class EnrichmentMetrics:
    """Comprehensive enrichment pipeline metrics (Phase 4).

    Tracks throughput, quality, timing, and API usage for monitoring
    and optimization of the enrichment pipeline.
    """

    # Throughput metrics
    stories_processed: int = 0
    direct_people_found: int = 0
    indirect_people_found: int = 0
    linkedin_matches: int = 0

    # Quality distribution
    high_confidence_matches: int = 0
    medium_confidence_matches: int = 0
    low_confidence_matches: int = 0
    rejected_matches: int = 0
    org_fallback_matches: int = 0

    # Timing (seconds)
    total_processing_time: float = 0.0
    avg_story_enrichment_time: float = 0.0
    avg_validation_time: float = 0.0
    avg_linkedin_search_time: float = 0.0

    # API usage
    gemini_calls: int = 0
    google_searches: int = 0
    duckduckgo_searches: int = 0

    # Error tracking
    validation_errors: int = 0
    linkedin_search_failures: int = 0
    network_timeouts: int = 0

    def record_story(self, processing_time: float) -> None:
        """Record a story being processed."""
        self.stories_processed += 1
        self.total_processing_time += processing_time
        if self.stories_processed > 0:
            self.avg_story_enrichment_time = (
                self.total_processing_time / self.stories_processed
            )

    def record_match(self, confidence: str) -> None:
        """Record a LinkedIn match by confidence level."""
        self.linkedin_matches += 1
        conf_lower = confidence.lower()
        if conf_lower == "high":
            self.high_confidence_matches += 1
        elif conf_lower == "medium":
            self.medium_confidence_matches += 1
        elif conf_lower == "org_fallback":
            self.org_fallback_matches += 1
        else:
            self.low_confidence_matches += 1

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for logging."""
        return {
            "stories_processed": self.stories_processed,
            "direct_people_found": self.direct_people_found,
            "indirect_people_found": self.indirect_people_found,
            "linkedin_matches": self.linkedin_matches,
            "match_rate": (
                f"{self.linkedin_matches / (self.direct_people_found + self.indirect_people_found):.1%}"
                if (self.direct_people_found + self.indirect_people_found) > 0
                else "N/A"
            ),
            "high_confidence_rate": (
                f"{self.high_confidence_matches / self.linkedin_matches:.1%}"
                if self.linkedin_matches > 0
                else "N/A"
            ),
            "avg_enrichment_time_s": round(self.avg_story_enrichment_time, 2),
            "gemini_calls": self.gemini_calls,
            "total_searches": self.google_searches + self.duckduckgo_searches,
            "errors": self.validation_errors + self.linkedin_search_failures,
        }


@dataclass
class QACheckResult:
    """Results of quality assurance checks on enriched story (Phase 4)."""

    story_id: int
    checks_passed: int = 0
    checks_failed: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Per-check results
    linkedin_url_validity: bool = True
    employer_consistency: bool = True
    duplicate_detection: bool = True
    confidence_distribution_ok: bool = True
    required_fields_ok: bool = True

    @property
    def passed(self) -> bool:
        """Check if all critical validations passed."""
        return self.checks_failed == 0

    @property
    def status(self) -> str:
        """Overall status: passed, passed_with_warnings, or failed."""
        if self.checks_failed > 0:
            return "failed"
        if self.warnings:
            return "passed_with_warnings"
        return "passed"

    def to_dict(self) -> dict:
        """Convert to dictionary for storage in enrichment_log."""
        return {
            "story_id": self.story_id,
            "status": self.status,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def validate_enrichment_quality(story: Story) -> QACheckResult:
    """Run QA checks on an enriched story (Phase 4).

    Performs validation checks including:
    - LinkedIn URL validity
    - Required fields presence
    - Duplicate detection across direct/indirect
    - Confidence distribution analysis

    Args:
        story: The enriched story to validate

    Returns:
        QACheckResult with detailed check results
    """
    result = QACheckResult(story_id=story.id or 0)

    # Ensure we have list data
    direct_people = story.direct_people if isinstance(story.direct_people, list) else []
    indirect_people = (
        story.indirect_people if isinstance(story.indirect_people, list) else []
    )
    all_people = direct_people + indirect_people

    # Check 1: LinkedIn URL validity
    for person in all_people:
        linkedin_url = person.get("linkedin_profile") or person.get("linkedin_url", "")
        profile_type = person.get("linkedin_profile_type", "")
        confidence = person.get("match_confidence", "")
        if linkedin_url:
            if "linkedin.com/in/" in linkedin_url:
                # Personal profile URL - valid
                result.checks_passed += 1
            elif profile_type == "organization" or confidence == "org_fallback":
                # Org fallback is acceptable, not an error - just skip counting
                # Don't add to passed or failed, it's a known limitation
                pass
            elif (
                "linkedin.com/school/" in linkedin_url
                or "linkedin.com/company/" in linkedin_url
            ):
                # Organization URL explicitly assigned - also acceptable for org_fallback
                pass
            else:
                result.linkedin_url_validity = False
                result.checks_failed += 1
                result.errors.append(
                    f"Invalid LinkedIn URL for {person.get('name', 'Unknown')}: {linkedin_url}"
                )

    # Check 2: Required fields (name is always required)
    for person in all_people:
        if not person.get("name"):
            result.required_fields_ok = False
            result.checks_failed += 1
            result.errors.append("Person record missing required 'name' field")
        else:
            result.checks_passed += 1

    # Check 3: Duplicate detection across direct/indirect
    direct_names = {
        p.get("name", "").lower().strip() for p in direct_people if p.get("name")
    }
    indirect_names = {
        p.get("name", "").lower().strip() for p in indirect_people if p.get("name")
    }
    duplicates = direct_names & indirect_names
    if duplicates:
        result.duplicate_detection = False
        result.warnings.append(
            f"Duplicate names in direct/indirect: {', '.join(duplicates)}"
        )
    else:
        result.checks_passed += 1

    # Check 4: Confidence distribution - flag if >40% low confidence
    confidence_counts = {"high": 0, "medium": 0, "low": 0, "org_fallback": 0}
    for person in all_people:
        conf = (person.get("match_confidence") or "low").lower()
        if conf in confidence_counts:
            confidence_counts[conf] += 1
        else:
            confidence_counts["low"] += 1

    total_with_linkedin = sum(
        1 for p in all_people if p.get("linkedin_profile") or p.get("linkedin_url")
    )
    low_count = confidence_counts.get("low", 0)

    if total_with_linkedin > 0 and low_count / total_with_linkedin > 0.4:
        result.confidence_distribution_ok = False
        result.warnings.append(
            f"High proportion of low-confidence matches: {low_count}/{total_with_linkedin}"
        )
    else:
        result.checks_passed += 1

    return result


def validate_person_record(person: dict) -> tuple[bool, list[str]]:
    """Validate a single person record before storage (Phase 4).

    Args:
        person: Person dictionary to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Required field: name
    if not person.get("name"):
        errors.append("Missing required field: name")

    # URL validation
    linkedin_url = person.get("linkedin_profile") or person.get("linkedin_url", "")
    if linkedin_url:
        if not linkedin_url.startswith("http"):
            errors.append(f"Invalid LinkedIn URL (not HTTP/S): {linkedin_url}")
        elif "linkedin.com" not in linkedin_url:
            errors.append(f"LinkedIn URL not from linkedin.com: {linkedin_url}")

    # Confidence score validation (if present)
    match_score = person.get("match_score")
    if match_score is not None:
        try:
            score = float(match_score)
            if score < 0 or score > 10:
                errors.append(f"Match score out of range (0-10): {score}")
        except (ValueError, TypeError):
            errors.append(f"Invalid match score format: {match_score}")

    return len(errors) == 0, errors


# =============================================================================
# Step 5.4: Alerts and Notifications
# =============================================================================


class EnrichmentAlerts:
    """Alerting system for enrichment pipeline monitoring (Step 5.4).

    Provides proactive monitoring with three severity levels:
    - Critical: Immediate action required
    - Warning: Review within 24h
    - Info: Weekly review
    """

    # Thresholds for alerts
    CRITICAL_SUCCESS_RATE = 0.50  # Below 50% triggers critical
    WARNING_SUCCESS_RATE = 0.70  # Below 70% triggers warning
    WARNING_LOW_CONFIDENCE_RATE = 0.40  # >40% low confidence triggers warning
    WARNING_PROCESSING_TIME = 300.0  # >5 minutes per story

    def __init__(self) -> None:
        """Initialize the alerting system."""
        self._alerts: list[dict] = []

    def check_metrics(self, metrics: EnrichmentMetrics) -> list[dict]:
        """Check metrics and generate alerts as needed.

        Args:
            metrics: Current enrichment metrics

        Returns:
            List of alert dictionaries
        """
        self._alerts = []

        if metrics.stories_processed == 0:
            return self._alerts

        # Calculate rates
        total_people = metrics.direct_people_found + metrics.indirect_people_found
        success_rate = (
            metrics.linkedin_matches / total_people if total_people > 0 else 0
        )
        low_conf_rate = (
            metrics.low_confidence_matches / metrics.linkedin_matches
            if metrics.linkedin_matches > 0
            else 0
        )
        error_rate = (
            (metrics.validation_errors + metrics.linkedin_search_failures)
            / metrics.stories_processed
            if metrics.stories_processed > 0
            else 0
        )

        # Check critical conditions
        if success_rate < self.CRITICAL_SUCCESS_RATE:
            self._alert_critical(
                "Enrichment success rate critically low",
                {"success_rate": f"{success_rate:.1%}", "threshold": "50%"},
            )

        if error_rate > 0.20:  # >20% error rate
            self._alert_critical(
                "High error rate in enrichment pipeline",
                {"error_rate": f"{error_rate:.1%}", "threshold": "20%"},
            )

        # Check warning conditions
        if self.CRITICAL_SUCCESS_RATE <= success_rate < self.WARNING_SUCCESS_RATE:
            self._alert_warning(
                "Enrichment success rate below target",
                {"success_rate": f"{success_rate:.1%}", "target": "70%"},
            )

        if low_conf_rate > self.WARNING_LOW_CONFIDENCE_RATE:
            self._alert_warning(
                "High proportion of low-confidence matches",
                {"low_conf_rate": f"{low_conf_rate:.1%}", "threshold": "40%"},
            )

        if metrics.avg_story_enrichment_time > self.WARNING_PROCESSING_TIME:
            self._alert_warning(
                "Story enrichment time exceeds target",
                {
                    "avg_time": f"{metrics.avg_story_enrichment_time:.1f}s",
                    "threshold": "300s",
                },
            )

        # Info alerts for trends
        if metrics.high_confidence_matches > 0:
            high_conf_rate = metrics.high_confidence_matches / metrics.linkedin_matches
            if high_conf_rate > 0.80:
                self._alert_info(
                    "High confidence match rate is excellent",
                    {"high_conf_rate": f"{high_conf_rate:.1%}"},
                )

        return self._alerts

    def _alert_critical(self, message: str, context: dict) -> None:
        """Log critical alert requiring immediate action."""
        alert = {
            "severity": "critical",
            "message": message,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }
        self._alerts.append(alert)
        logger.critical(f"ALERT: {message} - {context}")

    def _alert_warning(self, message: str, context: dict) -> None:
        """Log warning alert requiring 24h review."""
        alert = {
            "severity": "warning",
            "message": message,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }
        self._alerts.append(alert)
        logger.warning(f"ALERT: {message} - {context}")

    def _alert_info(self, message: str, context: dict) -> None:
        """Log info alert for weekly review."""
        alert = {
            "severity": "info",
            "message": message,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }
        self._alerts.append(alert)
        logger.info(f"ALERT: {message} - {context}")

    def get_alerts(self) -> list[dict]:
        """Get all current alerts."""
        return self._alerts.copy()


# =============================================================================
# Step 6.2: Incremental Updates
# =============================================================================


def needs_refresh(
    story: Story,
    direct_refresh_days: int = 365,
    indirect_refresh_days: int = 90,
) -> bool:
    """Determine if a story needs re-enrichment (Step 6.2).

    Args:
        story: Story to check
        direct_refresh_days: Days before direct people need refresh
        indirect_refresh_days: Days before indirect people (org leaders) need refresh

    Returns:
        True if story needs re-enrichment
    """
    # Never enriched
    if story.enrichment_status in ("pending", "", None):
        return True

    # Error state - retry
    if story.enrichment_status == "error":
        return True

    # Check enrichment log for completion time
    log = story.enrichment_log
    if isinstance(log, str):
        try:
            log = json.loads(log) if log else {}
        except json.JSONDecodeError:
            log = {}

    completed_at = log.get("completed_at")
    if not completed_at:
        return True

    try:
        completed_date = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        days_old = (datetime.now() - completed_date.replace(tzinfo=None)).days
    except (ValueError, AttributeError):
        return True

    # Check quality - low quality should be refreshed sooner
    if story.enrichment_quality in ("low", "failed"):
        return days_old > 7  # Retry low quality after a week

    # Check if indirect people are stale (quarterly refresh)
    indirect_people = story.indirect_people
    if isinstance(indirect_people, str):
        try:
            indirect_people = json.loads(indirect_people) if indirect_people else []
        except json.JSONDecodeError:
            indirect_people = []

    if indirect_people and days_old > indirect_refresh_days:
        return True

    # Check if direct people need refresh (yearly)
    if days_old > direct_refresh_days:
        return True

    return False


def get_stories_needing_refresh(
    db: Database,
    limit: int = 50,
    direct_refresh_days: int = 365,
    indirect_refresh_days: int = 90,
) -> list[Story]:
    """Get stories that need re-enrichment (Step 6.2).

    Args:
        db: Database instance
        limit: Maximum stories to return
        direct_refresh_days: Days before direct people need refresh
        indirect_refresh_days: Days before indirect people need refresh

    Returns:
        List of stories needing refresh
    """
    # Get enriched stories
    enriched = db.get_published_stories() + db.get_scheduled_stories()

    needing_refresh = []
    for story in enriched:
        if needs_refresh(story, direct_refresh_days, indirect_refresh_days):
            needing_refresh.append(story)
            if len(needing_refresh) >= limit:
                break

    return needing_refresh


# =============================================================================
# Step 6.5: Export Options
# =============================================================================


def export_direct_people(story: Story, format: str = "json") -> str:
    """Export people data from a story in various formats (Step 6.5).

    Args:
        story: Story to export people from
        format: Output format - "json", "csv", or "markdown"

    Returns:
        Formatted string of people data
    """
    # Parse people data
    direct_people = story.direct_people
    if isinstance(direct_people, str):
        try:
            direct_people = json.loads(direct_people) if direct_people else []
        except json.JSONDecodeError:
            direct_people = []

    indirect_people = story.indirect_people
    if isinstance(indirect_people, str):
        try:
            indirect_people = json.loads(indirect_people) if indirect_people else []
        except json.JSONDecodeError:
            indirect_people = []

    people = {
        "story_id": story.id,
        "story_title": story.title,
        "direct_people": direct_people,
        "indirect_people": indirect_people,
        "enrichment_quality": story.enrichment_quality,
        "exported_at": datetime.now().isoformat(),
    }

    if format == "json":
        return json.dumps(people, indent=2, default=str)

    elif format == "csv":
        lines = ["category,name,title,employer,linkedin_url,confidence"]
        for p in direct_people:
            lines.append(
                f'direct,"{p.get("name", "")}","{p.get("title", p.get("position", ""))}",'
                f'"{p.get("company", p.get("affiliation", ""))}",'
                f'"{p.get("linkedin_profile", p.get("linkedin_url", ""))}",'
                f'"{p.get("match_confidence", "")}"'
            )
        for p in indirect_people:
            lines.append(
                f'indirect,"{p.get("name", "")}","{p.get("title", "")}",'
                f'"{p.get("organization", "")}",'
                f'"{p.get("linkedin_profile", p.get("linkedin_url", ""))}",'
                f'"{p.get("match_confidence", "")}"'
            )
        return "\n".join(lines)

    elif format == "markdown":
        md = [f"# People in Story: {story.title}", ""]

        if direct_people:
            md.append("## Direct People (mentioned in story)")
            md.append("")
            for p in direct_people:
                name = p.get("name", "Unknown")
                title = p.get("title", p.get("position", ""))
                org = p.get("company", p.get("affiliation", ""))
                linkedin = p.get("linkedin_profile", p.get("linkedin_url", ""))
                conf = p.get("match_confidence", "")

                md.append(f"### {name}")
                if title:
                    md.append(f"- **Title:** {title}")
                if org:
                    md.append(f"- **Organization:** {org}")
                if linkedin:
                    md.append(f"- **LinkedIn:** [{linkedin}]({linkedin})")
                if conf:
                    md.append(f"- **Match Confidence:** {conf}")
                md.append("")

        if indirect_people:
            md.append("## Indirect People (organization leadership)")
            md.append("")
            for p in indirect_people:
                name = p.get("name", "Unknown")
                title = p.get("title", "")
                org = p.get("organization", "")
                linkedin = p.get("linkedin_profile", p.get("linkedin_url", ""))
                conf = p.get("match_confidence", "")

                md.append(f"### {name}")
                if title:
                    md.append(f"- **Title:** {title}")
                if org:
                    md.append(f"- **Organization:** {org}")
                if linkedin:
                    md.append(f"- **LinkedIn:** [{linkedin}]({linkedin})")
                if conf:
                    md.append(f"- **Match Confidence:** {conf}")
                md.append("")

        return "\n".join(md)

    return ""


def export_all_people(db: Database, format: str = "json") -> str:
    """Export all people across all stories (Step 6.5).

    Args:
        db: Database instance
        format: Output format - "json" or "csv"

    Returns:
        Formatted string of all people data
    """
    stories = db.get_published_stories() + db.get_scheduled_stories()

    all_people: list[dict] = []
    seen_urns: set[str] = set()

    for story in stories:
        # Parse people
        direct_people = story.direct_people
        if isinstance(direct_people, str):
            try:
                direct_people = json.loads(direct_people) if direct_people else []
            except json.JSONDecodeError:
                direct_people = []

        indirect_people = story.indirect_people
        if isinstance(indirect_people, str):
            try:
                indirect_people = json.loads(indirect_people) if indirect_people else []
            except json.JSONDecodeError:
                indirect_people = []

        # Add direct people
        for p in direct_people:
            urn = p.get("linkedin_urn", "")
            if urn and urn in seen_urns:
                continue  # Deduplicate by URN
            if urn:
                seen_urns.add(urn)

            all_people.append(
                {
                    "name": p.get("name", ""),
                    "title": p.get("title", p.get("position", "")),
                    "employer": p.get("company", p.get("affiliation", "")),
                    "linkedin_url": p.get(
                        "linkedin_profile", p.get("linkedin_url", "")
                    ),
                    "linkedin_urn": urn,
                    "confidence": p.get("match_confidence", ""),
                    "category": "direct",
                    "story_id": story.id,
                }
            )

        # Add indirect people
        for p in indirect_people:
            urn = p.get("linkedin_urn", "")
            if urn and urn in seen_urns:
                continue
            if urn:
                seen_urns.add(urn)

            all_people.append(
                {
                    "name": p.get("name", ""),
                    "title": p.get("title", ""),
                    "employer": p.get("organization", ""),
                    "linkedin_url": p.get(
                        "linkedin_profile", p.get("linkedin_url", "")
                    ),
                    "linkedin_urn": urn,
                    "confidence": p.get("match_confidence", ""),
                    "category": "indirect",
                    "story_id": story.id,
                }
            )

    if format == "json":
        return json.dumps(
            {
                "total_people": len(all_people),
                "unique_by_urn": len(seen_urns),
                "exported_at": datetime.now().isoformat(),
                "people": all_people,
            },
            indent=2,
            default=str,
        )

    elif format == "csv":
        lines = [
            "name,title,employer,linkedin_url,linkedin_urn,confidence,category,story_id"
        ]
        for p in all_people:
            lines.append(
                f'"{p["name"]}","{p["title"]}","{p["employer"]}",'
                f'"{p["linkedin_url"]}","{p["linkedin_urn"]}",'
                f'"{p["confidence"]}","{p["category"]}",{p["story_id"]}'
            )
        return "\n".join(lines)

    return ""


# =============================================================================
# Global Enrichment Metrics Instance
# =============================================================================

_enrichment_metrics: EnrichmentMetrics | None = None


def get_enrichment_metrics() -> EnrichmentMetrics:
    """Get or create the global enrichment metrics instance."""
    global _enrichment_metrics
    if _enrichment_metrics is None:
        _enrichment_metrics = EnrichmentMetrics()
    return _enrichment_metrics


def reset_enrichment_metrics() -> None:
    """Reset the global enrichment metrics."""
    global _enrichment_metrics
    _enrichment_metrics = EnrichmentMetrics()


def validate_linkedin_profile_url(url: str, strict: bool = False) -> bool:
    """
    Validate that a LinkedIn personal profile URL appears to be valid.

    By default (strict=False), only validates URL format since LinkedIn blocks
    unauthenticated HEAD/GET requests. Full validation happens later when
    visiting the profile with an authenticated browser session.

    When strict=True, performs HTTP validation (may fail due to LinkedIn blocks).

    Args:
        url: The LinkedIn profile URL to validate
        strict: If True, perform HTTP request validation (may be blocked by LinkedIn)

    Returns:
        True if the URL appears to be a valid LinkedIn profile URL
    """
    if not url or "linkedin.com/in/" not in url:
        return False

    # Extract the username/slug from the URL
    import re

    match = re.search(r"linkedin\.com/in/([\w\-]+)", url)
    if not match:
        return False

    slug = match.group(1)

    # Basic format validation - reject obviously invalid slugs
    if len(slug) < 2 or len(slug) > 100:
        return False

    # Reject common error page slugs
    invalid_slugs = {"login", "authwall", "error", "404", "unavailable", "uas"}
    if slug.lower() in invalid_slugs:
        return False

    # If not strict mode, accept the URL based on format alone
    if not strict:
        return True

    # Strict mode: perform HTTP validation (may be blocked by LinkedIn)
    try:
        # Use headers that mimic a browser to avoid blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        response = api_client.http_request(
            method="HEAD",
            url=url,
            headers=headers,
            timeout=10,
            allow_redirects=True,
            endpoint="linkedin_profile_validate",
        )

        # Check if we got a successful response
        if response.status_code == 200:
            # Check if we weren't redirected to a login or error page
            final_url = response.url if hasattr(response, "url") else url
            if "/login" in final_url or "/authwall" in final_url:
                logger.debug(f"LinkedIn URL redirected to login: {url}")
                return False
            return True

        # 405 Method Not Allowed - try GET instead
        if response.status_code == 405:
            response = api_client.http_request(
                method="GET",
                url=url,
                headers=headers,
                timeout=10,
                allow_redirects=True,
                endpoint="linkedin_profile_validate",
            )
            if response.status_code == 200:
                # Check we're still on a profile page
                if "/in/" in response.url and "/login" not in response.url:
                    return True

        logger.debug(
            f"LinkedIn URL validation failed with status {response.status_code}: {url}"
        )
        return False

    except requests.exceptions.Timeout:
        # Timeout might mean the URL exists but is slow - accept cautiously
        logger.debug(f"LinkedIn URL timeout (accepting cautiously): {url}")
        return True
    except requests.exceptions.RequestException as e:
        logger.debug(f"LinkedIn URL validation error ({type(e).__name__}): {url}")
        return False


# =============================================================================
# Validation Cache (Phase 2 - spec Step 6.1)
# =============================================================================


class ValidationCache:
    """Cache for person validation results to reduce API costs.

    Phase 2 implementation per spec:
    - Deduplicate validation requests across stories
    - Cache organization leadership lookups
    - Track cache hit rates for monitoring
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize the validation cache.

        Args:
            max_size: Maximum entries before LRU eviction
        """
        self._person_cache: dict[str, dict] = {}
        self._indirect_people_cache: dict[str, list[dict]] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _make_person_key(self, name: str, org: str) -> str:
        """Create cache key for person validation."""
        return f"{name.lower().strip()}|{org.lower().strip()}"

    def get_person_validation(self, name: str, org: str) -> dict | None:
        """Get cached person validation result.

        Args:
            name: Person's name
            org: Organization name

        Returns:
            Cached validation dict or None if not cached
        """
        key = self._make_person_key(name, org)
        if key in self._person_cache:
            self._hits += 1
            return self._person_cache[key]
        self._misses += 1
        return None

    def set_person_validation(self, name: str, org: str, result: dict) -> None:
        """Cache a person validation result.

        Args:
            name: Person's name
            org: Organization name
            result: Validation result dict
        """
        # Simple LRU: if at max, remove oldest (first) entry
        if len(self._person_cache) >= self._max_size:
            oldest_key = next(iter(self._person_cache))
            del self._person_cache[oldest_key]

        key = self._make_person_key(name, org)
        self._person_cache[key] = result

    def get_indirect_people(self, org: str) -> list[dict] | None:
        """Get cached indirect people (org leadership).

        Args:
            org: Organization name

        Returns:
            Cached list of leaders or None if not cached
        """
        key = org.lower().strip()
        if key in self._indirect_people_cache:
            self._hits += 1
            return self._indirect_people_cache[key]
        self._misses += 1
        return None

    def set_indirect_people(self, org: str, leaders: list[dict]) -> None:
        """Cache indirect people (org leadership).

        Args:
            org: Organization name
            leaders: List of leader dicts
        """
        key = org.lower().strip()
        self._indirect_people_cache[key] = leaders

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "person_cache_size": len(self._person_cache),
            "indirect_people_cache_size": len(self._indirect_people_cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }

    def clear(self) -> None:
        """Clear all caches and reset stats."""
        self._person_cache.clear()
        self._indirect_people_cache.clear()
        self._hits = 0
        self._misses = 0


# Global validation cache instance
_validation_cache: ValidationCache | None = None


def get_validation_cache() -> ValidationCache:
    """Get or create the global validation cache."""
    global _validation_cache
    if _validation_cache is None:
        _validation_cache = ValidationCache()
    return _validation_cache


class CompanyMentionEnricher:
    """Enrich stories with company mentions extracted from sources."""

    def __init__(
        self,
        database: Database,
        client: genai.Client,
        local_client: OpenAI | None = None,
    ):
        """Initialize the company mention enricher."""
        self.db = database
        self.client = client
        self.local_client = local_client
        # Phase 2: Use validation cache for deduplication
        self._cache = get_validation_cache()
        # Phase 4: Track enrichment metrics
        self._metrics = get_enrichment_metrics()

    def enrich_story_atomic(
        self,
        story: Story,
        validate: bool = True,
    ) -> tuple[bool, QACheckResult | None]:
        """Atomically enrich and store a story with validation (Phase 4).

        This method wraps the enrichment update in a transaction and performs
        pre-storage validation to ensure data quality.

        Args:
            story: Story to persist with enrichment data
            validate: If True, run QA checks before storage

        Returns:
            Tuple of (success, qa_result). qa_result is None if validate=False
        """
        start_time = time.time()
        qa_result = None

        try:
            # Pre-storage validation
            if validate:
                qa_result = validate_enrichment_quality(story)

                # Validate individual person records
                direct_people = (
                    story.direct_people if isinstance(story.direct_people, list) else []
                )
                indirect_people = (
                    story.indirect_people
                    if isinstance(story.indirect_people, list)
                    else []
                )

                all_people = direct_people + indirect_people
                for person in all_people:
                    is_valid, errors = validate_person_record(person)
                    if not is_valid:
                        qa_result.checks_failed += len(errors)
                        qa_result.errors.extend(errors)

                # Log validation results
                if qa_result.errors:
                    logger.warning(
                        f"Story {story.id} validation errors: {qa_result.errors}"
                    )
                if qa_result.warnings:
                    logger.info(
                        f"Story {story.id} validation warnings: {qa_result.warnings}"
                    )

            # Build enrichment log
            log = story.enrichment_log if isinstance(story.enrichment_log, dict) else {}
            log["completed_at"] = datetime.now().isoformat()
            log["processing_time_seconds"] = round(time.time() - start_time, 2)
            direct_people = (
                story.direct_people if isinstance(story.direct_people, list) else []
            )
            indirect_people = (
                story.indirect_people if isinstance(story.indirect_people, list) else []
            )

            log["direct_people_count"] = len(direct_people)
            log["indirect_people_count"] = len(indirect_people)

            if validate and qa_result:
                log["validation"] = qa_result.to_dict()

            story.enrichment_log = log

            # Determine enrichment quality based on validation
            if validate and qa_result:
                if qa_result.status == "failed":
                    story.enrichment_quality = "low"
                elif qa_result.status == "passed_with_warnings":
                    story.enrichment_quality = "medium"
                else:
                    story.enrichment_quality = "high"

            # Atomic update
            story.enrichment_status = "enriched"
            success = self.db.update_story(story)

            # Save people to the people table for connection tracking
            if success and story.id:
                self._save_people_to_table(story, direct_people, indirect_people)

            # Track metrics
            processing_time = time.time() - start_time
            self._metrics.record_story(processing_time)
            self._metrics.direct_people_found += log.get("direct_people_count", 0)
            self._metrics.indirect_people_found += log.get("indirect_people_count", 0)

            return success, qa_result

        except Exception as e:
            logger.error(f"Atomic enrichment failed for story {story.id}: {e}")
            self._metrics.validation_errors += 1

            # Mark as error but don't block pipeline
            try:
                story.enrichment_status = "error"
                log = (
                    story.enrichment_log
                    if isinstance(story.enrichment_log, dict)
                    else {}
                )
                log["error"] = str(e)
                log["error_at"] = datetime.now().isoformat()
                story.enrichment_log = log
                self.db.update_story(story)
            except Exception:
                pass  # Don't fail on error logging

            return False, qa_result

    def _save_people_to_table(
        self,
        story: Story,
        direct_people: list[dict],
        indirect_people: list[dict],
    ) -> None:
        """Save people from enrichment to the people table for connection tracking.

        Args:
            story: The story these people are associated with
            direct_people: List of direct person dicts from enrichment
            indirect_people: List of indirect person dicts from enrichment
        """
        from database import Person

        try:
            # Save direct people
            for person_dict in direct_people:
                linkedin_profile = person_dict.get("linkedin_profile", "")
                # Only save if we have a valid LinkedIn profile
                if linkedin_profile and "linkedin.com/in/" in linkedin_profile:
                    person = Person(
                        name=person_dict.get("name", ""),
                        title=person_dict.get("job_title") or person_dict.get("title"),
                        organization=person_dict.get("affiliation")
                        or person_dict.get("organization")
                        or person_dict.get("company"),
                        location=person_dict.get("location"),
                        specialty=person_dict.get("specialty"),
                        department=person_dict.get("department"),
                        story_id=story.id,
                        relationship_type="direct",
                        linkedin_profile=linkedin_profile,
                        linkedin_urn=person_dict.get("linkedin_urn"),
                    )
                    self.db.add_person(person)

            # Save indirect people
            for person_dict in indirect_people:
                linkedin_profile = person_dict.get("linkedin_profile", "")
                # Only save if we have a valid LinkedIn profile
                if linkedin_profile and "linkedin.com/in/" in linkedin_profile:
                    person = Person(
                        name=person_dict.get("name", ""),
                        title=person_dict.get("job_title") or person_dict.get("title"),
                        organization=person_dict.get("affiliation")
                        or person_dict.get("organization")
                        or person_dict.get("company"),
                        location=person_dict.get("location"),
                        specialty=person_dict.get("specialty"),
                        department=person_dict.get("department"),
                        story_id=story.id,
                        relationship_type="indirect",
                        linkedin_profile=linkedin_profile,
                        linkedin_urn=person_dict.get("linkedin_urn"),
                    )
                    self.db.add_person(person)

            logger.debug(
                f"Saved {len(direct_people)} direct + {len(indirect_people)} indirect people for story {story.id}"
            )
        except Exception as e:
            logger.warning(f"Error saving people to table for story {story.id}: {e}")

    def get_metrics(self) -> EnrichmentMetrics:
        """Get current enrichment metrics."""
        return self._metrics

    def log_metrics_summary(self) -> None:
        """Log a summary of enrichment metrics."""
        metrics = self._metrics.to_dict()
        logger.info(f"Enrichment metrics: {json.dumps(metrics, indent=2)}")

    # ---------------------------------------------------------------------
    # Helpers for direct/indirect people extraction
    # ---------------------------------------------------------------------
    def _normalize_people_records(
        self, people: list[dict] | None, source: str
    ) -> list[dict]:
        """Normalize person dictionaries into a consistent schema with enhanced matching fields."""
        normalized: list[dict] = []
        for person in people or []:
            name = str(person.get("name", "")).strip()
            if not name:
                continue

            # Extract department separately for LinkedIn matching
            department = str(
                person.get("department")
                or person.get("specialty")
                or person.get("research_area")
                or ""
            ).strip()

            normalized.append(
                {
                    "name": name,
                    "job_title": str(
                        person.get("job_title")
                        or person.get("title")
                        or person.get("position")
                        or ""
                    ).strip(),
                    "employer": str(
                        person.get("employer")
                        or person.get("company")
                        or person.get("affiliation")
                        or person.get("organization")
                        or ""
                    ).strip(),
                    # Legacy-compatible aliases
                    "company": str(
                        person.get("employer")
                        or person.get("company")
                        or person.get("affiliation")
                        or person.get("organization")
                        or ""
                    ).strip(),
                    "organization": str(
                        person.get("employer")
                        or person.get("company")
                        or person.get("affiliation")
                        or person.get("organization")
                        or ""
                    ).strip(),
                    "position": str(
                        person.get("job_title")
                        or person.get("title")
                        or person.get("position")
                        or ""
                    ).strip(),
                    "department": department,
                    "location": str(person.get("location", "")).strip(),
                    "specialty": str(
                        person.get("specialty")
                        or person.get("research_area")
                        or department
                        or ""
                    ).strip(),
                    # New fields for enhanced LinkedIn matching
                    "credentials": str(person.get("credentials", "")).strip(),
                    "context_clues": str(person.get("context_clues", "")).strip(),
                    "role_in_story": str(person.get("role_in_story", ""))
                    .strip()
                    .lower(),
                    "role_type": str(person.get("role_type", source)).strip().lower(),
                    "linkedin_profile": person.get("linkedin_profile", ""),
                    "linkedin_urn": person.get("linkedin_urn", ""),
                    "match_confidence": person.get("match_confidence", ""),
                    "source": source,
                }
            )

        return normalized

    def _is_valid_person_name(self, name: str) -> bool:
        """Check if a name looks like a valid person name (not an org).

        Filters out:
        - Organization names (containing 'University', 'State', etc.)
        - Generic roles ('researcher', 'professor' alone)
        - Single-word names that look like surnames only or orgs

        Args:
            name: The name to validate

        Returns:
            True if this looks like a valid person name
        """
        if not name:
            return False

        name_lower = name.lower().strip()

        # Check against known invalid person names
        if name_lower in INVALID_PERSON_NAMES:
            return False

        # Check for organization-like patterns
        for pattern in ORG_NAME_PATTERNS:
            if pattern in name_lower:
                logger.debug(
                    f"Filtering org-like name: '{name}' (contains '{pattern}')"
                )
                return False

        # Single word names need special handling
        words = name.split()
        if len(words) == 1:
            # Single word - likely a surname only or org abbreviation
            # Allow if it has title case and doesn't match org patterns
            if name_lower in INVALID_ORG_NAMES:
                logger.debug(f"Filtering single-word org name: '{name}'")
                return False
            # Very short single words are suspicious
            if len(name_lower) < 3:
                logger.debug(f"Filtering too-short name: '{name}'")
                return False

        # Names with "Srinivasan Srinivasan" pattern (duplicate first/last)
        if len(words) == 2 and words[0].lower() == words[1].lower():
            logger.debug(f"Filtering duplicate-word name: '{name}'")
            return False

        return True

    def _filter_valid_people(self, people: list[dict]) -> list[dict]:
        """Filter people list to remove invalid/organization names.

        Args:
            people: List of person dictionaries

        Returns:
            Filtered list with only valid person names
        """
        valid = []
        for person in people:
            name = person.get("name", "")
            if self._is_valid_person_name(name):
                valid.append(person)
            else:
                logger.debug(f"Filtered invalid person name: '{name}'")
        if len(valid) < len(people):
            logger.info(
                f"Filtered people: {len(people)} -> {len(valid)} "
                f"(removed {len(people) - len(valid)} invalid names)"
            )
        return valid

    def _dedupe_people(self, people: list[dict]) -> list[dict]:
        """Deduplicate person records by normalized name.

        Uses name-only deduplication because the same person often appears with
        different employer name variations (e.g., 'DEFRA' vs 'Department for Environment,
        Food & Rural Affairs' vs 'Government'). When duplicates are found, we keep the
        record with the most complete LinkedIn profile information.

        Name normalization handles:
        - Titles: "Dr. John Smith" -> "john smith"
        - Middle initials: "Paula T. Hammond" -> "paula hammond"
        - Suffixes: "John Smith Jr." -> "john smith"
        """
        # Track best record for each normalized name
        best_by_name: dict[str, dict] = {}

        for person in people:
            # Normalize name: lowercase, strip whitespace, remove common prefixes
            name = person.get("name", "").lower().strip()
            # Remove common titles/prefixes for matching
            for prefix in [
                "dame ",
                "sir ",
                "lord ",
                "lady ",
                "dr ",
                "dr. ",
                "prof ",
                "prof. ",
                "professor ",
            ]:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break

            # Remove common suffixes
            for suffix in [
                " jr",
                " jr.",
                " sr",
                " sr.",
                " iii",
                " ii",
                " iv",
                " phd",
                " ph.d",
                " md",
                " m.d",
            ]:
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break

            # Remove middle initials (single letter followed by optional period and space)
            # "paula t. hammond" -> "paula hammond"
            # "john a b smith" -> "john smith"
            name = re.sub(r"\s+[a-z]\.?\s+", " ", name)  # Remove middle initials
            name = re.sub(r"\s+", " ", name).strip()  # Normalize whitespace

            if not name:
                continue

            existing = best_by_name.get(name)
            if existing is None:
                best_by_name[name] = person
            else:
                # Keep the record with better LinkedIn info
                new_has_profile = bool(
                    person.get("linkedin_profile") or person.get("linkedin_urn")
                )
                old_has_profile = bool(
                    existing.get("linkedin_profile") or existing.get("linkedin_urn")
                )

                if new_has_profile and not old_has_profile:
                    best_by_name[name] = person
                elif new_has_profile and old_has_profile:
                    # Both have profiles - prefer higher confidence
                    new_conf = person.get("match_confidence", "")
                    old_conf = existing.get("match_confidence", "")
                    confidence_rank = {
                        "high": 3,
                        "medium": 2,
                        "low": 1,
                        "org_fallback": 0,
                        "rejected": -1,
                    }
                    if confidence_rank.get(new_conf, 0) > confidence_rank.get(
                        old_conf, 0
                    ):
                        best_by_name[name] = person

        unique = list(best_by_name.values())
        if len(unique) < len(people):
            logger.info(
                f"Deduplicated people: {len(people)} -> {len(unique)} (removed {len(people) - len(unique)} duplicates)"
            )
        return unique

    def _extract_direct_people_via_ner(
        self, story: Story
    ) -> tuple[list[dict], set[str]]:
        """Use spaCy-based NER to seed direct people/org lists."""
        try:
            from ner_engine import extract_entities_from_story, get_ner_engine

            ner = get_ner_engine()
            result = extract_entities_from_story(
                story.title or "", story.summary or "", ner_engine=ner
            )
        except Exception as e:
            logger.debug(f"NER extraction unavailable: {e}")
            return [], set()

        people: list[dict] = []
        organizations: set[str] = set()

        for person in result.persons:
            metadata = getattr(person, "metadata", {}) or {}
            people.append(
                {
                    "name": person.text,
                    "job_title": metadata.get("title", getattr(person, "title", "")),
                    "employer": metadata.get(
                        "affiliation", getattr(person, "affiliation", "")
                    ),
                    "location": metadata.get("location", ""),
                    "specialty": metadata.get("field", metadata.get("department", "")),
                    "role_type": "direct",
                }
            )
            employer = metadata.get("affiliation", getattr(person, "affiliation", ""))
            if employer:
                organizations.add(employer)

        for org in result.organizations:
            org_name = getattr(org, "normalized", "") or getattr(org, "text", "")
            if org_name:
                organizations.add(org_name)

        normalized_people = self._normalize_people_records(people, source="direct")
        return normalized_people, organizations

    def _extract_direct_people_from_story(
        self, story: Story
    ) -> tuple[list[dict], list[str]]:
        """Extract direct people and orgs from story text using AI with enhanced details for LinkedIn matching."""

        sources_str = (
            ", ".join(story.source_links[:5]) if story.source_links else "Not provided"
        )
        prompt = f"""
You are extracting people explicitly mentioned in a news story. Your goal is to provide DETAILED information that enables accurate LinkedIn profile matching.

Story title: {story.title}
Story summary: {story.summary}
Story sources: {sources_str}

Return STRICT JSON with this shape:
{{
  "direct_people": [
    {{
      "name": "Full Name (First Last format)",
      "job_title": "Exact title as mentioned (e.g., 'Senior Research Scientist', 'CEO', 'Professor')",
      "employer": "Company/Institution name (official name, not abbreviation)",
      "department": "Department, Division, or Lab name if mentioned",
      "location": "City, State/Country if mentioned",
      "specialty": "Field of expertise, research area, or domain",
      "credentials": "PhD, Dr., Prof., etc. if mentioned",
      "role_in_story": "primary|quoted|mentioned",
      "context_clues": "Any other identifying details (e.g., 'co-author', 'led the study', 'announced at conference')"
    }}
  ],
  "organizations": ["Official Organization Name", ...]
}}

EXTRACTION RULES:
1. Include ONLY people explicitly named in the story content
2. Use FULL official organization names (e.g., "Massachusetts Institute of Technology" not "MIT")
3. Extract department/lab names when available (e.g., "Department of Chemical Engineering")
4. Include location details for disambiguation (multiple people may share names)
5. Capture credentials like "Dr.", "Professor", "PhD" when present
6. For "role_in_story":
   - "primary": Main subject of the story or lead researcher
   - "quoted": Directly quoted in the story
   - "mentioned": Referenced but not quoted
7. Keep strings concise; no markdown, no additional text beyond the JSON
8. If any field is not available, use empty string ""
9. NEVER use the same word for first AND last name (e.g., don't create "Srinivasan Srinivasan")
10. For single-name mentions, try to find the full name from context, or use the single name only
11. Do NOT include organization names (universities, companies, institutes) as person names
"""

        try:
            response_text = self._get_ai_response(prompt)
            if not response_text:
                return [], list(story.organizations)

            response_text = strip_markdown_code_block(response_text)
            data = json.loads(response_text)

            direct_people_raw = data.get("direct_people") or data.get("people") or []
            organizations = data.get("organizations", []) or []

            direct_people = self._normalize_people_records(
                direct_people_raw, source="direct"
            )
            return direct_people, organizations
        except Exception as e:
            logger.debug(f"Direct extraction fallback for story {story.id}: {e}")
            return [], list(story.organizations)

    def _filter_valid_organizations(self, organizations: set[str]) -> set[str]:
        """Pre-filter organizations to skip invalid/generic names before LinkedIn lookups.

        This avoids wasted API calls on names like 'Government', 'Aldi', 'Asda' that will
        fail validation in the LinkedIn lookup anyway.

        Args:
            organizations: Set of organization names to filter

        Returns:
            Filtered set containing only valid organization names
        """
        valid_orgs: set[str] = set()
        for org in organizations:
            if not org:
                continue

            # Use centralized validation (handles patterns, length, AI explanations)
            if is_invalid_org_name(org):
                logger.debug(
                    f"Pre-filtering invalid org (centralized check): '{org[:50]}'"
                )
                continue

            norm = org.lower().strip()

            # Skip if org name is too long (likely an AI explanation)
            if len(org) > 100:
                logger.debug(f"Pre-filtering overly long org: '{org[:50]}...'")
                continue

            # Skip if too many words (likely a sentence/explanation)
            word_count = len(norm.split())
            if word_count > 10:
                logger.debug(
                    f"Pre-filtering org with too many words ({word_count}): '{org[:50]}...'"
                )
                continue

            # Skip single generic words (unless they're known companies)
            words = norm.split()
            if len(words) == 1 and norm not in VALID_SINGLE_WORD_ORGS:
                logger.debug(f"Pre-filtering single-word org: '{org}'")
                continue

            # Skip patterns that indicate AI explanation rather than org name
            ai_explanation_patterns = [
                "not applicable",
                "this is not",
                "no organization",
                "none mentioned",
                "not specified",
                "generalized",
                "generic",
                "various companies",
                "multiple ",  # "multiple companies"
                "several ",  # "several organizations"
                "unspecified",
                "headline",
                "actual research",
                " - ",  # Dash with spaces often indicates explanation
            ]
            if any(pattern in norm for pattern in ai_explanation_patterns):
                logger.debug(f"Pre-filtering AI explanation org: '{org[:50]}'")
                continue

            valid_orgs.add(org)

        if len(valid_orgs) < len(organizations):
            logger.info(
                f"Pre-filtered orgs: {len(organizations)} -> {len(valid_orgs)} "
                f"(removed {len(organizations) - len(valid_orgs)} invalid)"
            )

        return valid_orgs

    def _build_indirect_people(
        self,
        organizations: set[str],
        story: Story,
        direct_people: list[dict],
    ) -> list[dict]:
        """Discover indirect people (org leaders) while avoiding direct duplicates."""

        direct_names = {
            p.get("name", "").lower().strip() for p in direct_people if p.get("name")
        }

        # Pre-filter organizations: skip invalid/generic org names to avoid wasted lookups
        valid_orgs = self._filter_valid_organizations(organizations)

        indirect_candidates: list[dict] = []
        for org in sorted(org for org in valid_orgs if org):
            leaders = self._get_indirect_people(
                org, story_category=story.category, story_title=story.title
            )
            for leader in leaders:
                name = str(leader.get("name", "")).strip()
                if not name or name.lower() in direct_names:
                    continue

                normalized = self._normalize_people_records(
                    [
                        {
                            "name": name,
                            "job_title": leader.get("title", ""),
                            "employer": leader.get("organization", org),
                            "location": leader.get("location", ""),
                            "specialty": leader.get("department", ""),
                            "role_type": leader.get("role_type", "indirect"),
                            "linkedin_profile": leader.get("linkedin_profile", ""),
                            "linkedin_urn": leader.get("linkedin_urn", ""),
                            "match_confidence": leader.get("match_confidence", ""),
                        }
                    ],
                    source="indirect",
                )

                indirect_candidates.extend(normalized)

        return self._dedupe_people(indirect_candidates)

    def _attach_linkedin_profiles(self, people: list[dict], story: Story) -> list[dict]:
        """Match people to LinkedIn profiles with validation + fallbacks."""

        if not people:
            return people

        def _match_people_to_linkedin(people_to_match: list[dict]) -> list[dict]:
            try:
                from linkedin_profile_lookup import LinkedInCompanyLookup
                from profile_matcher import (
                    MatchConfidence,
                    ProfileMatcher,
                    create_person_context,
                )
            except Exception as e:  # pragma: no cover - optional dependency
                logger.debug(f"Profile matcher unavailable, falling back: {e}")
                return []

            matched: list[dict] = []
            org_cache: dict[str, str] = {}

            with LinkedInCompanyLookup(genai_client=self.client) as lookup:
                matcher = ProfileMatcher(linkedin_lookup=lookup)

                for person in people_to_match:
                    updated = dict(person)
                    org_name = (
                        person.get("employer")
                        or person.get("company")
                        or person.get("organization")
                        or ""
                    )

                    fallback_url = None
                    if org_name:
                        if org_name in org_cache:
                            fallback_url = org_cache[org_name]
                        else:
                            url, _ = lookup.search_company(org_name)
                            if url:
                                org_cache[org_name] = url
                                fallback_url = url

                    person_dict = {
                        "name": person.get("name", ""),
                        "company": org_name,
                        "organization": org_name,
                        "position": person.get("job_title")
                        or person.get("position", ""),
                        "department": person.get("specialty", ""),
                        "location": person.get("location", ""),
                        "role_type": person.get("role_type", ""),
                        "research_area": person.get("specialty", ""),
                    }

                    ctx = create_person_context(
                        person_dict,
                        story_title=story.title,
                        story_category=story.category,
                    )

                    result = matcher.match_person(ctx, org_fallback_url=fallback_url)
                    linkedin_url = result.get_best_url() or ""

                    updated["linkedin_profile"] = linkedin_url
                    updated["linkedin_profile_type"] = (
                        "personal"
                        if result.is_person_profile()
                        else "organization"
                        if result.confidence == MatchConfidence.ORG_FALLBACK
                        else ""
                    )
                    updated["match_confidence"] = result.confidence.value
                    updated["match_reason"] = result.match_reason

                    # Immediately extract URN if we found a personal profile
                    # This uses the same browser session, saving time
                    if linkedin_url and "/in/" in linkedin_url:
                        try:
                            urn = lookup.lookup_person_urn(linkedin_url)
                            if urn:
                                updated["linkedin_urn"] = urn
                                logger.debug(
                                    f"  Extracted URN for {person.get('name')}: {urn}"
                                )
                        except Exception as e:
                            logger.debug(
                                f"  URN extraction failed for {person.get('name')}: {e}"
                            )

                    if (
                        linkedin_url
                        and result.confidence != MatchConfidence.ORG_FALLBACK
                    ):
                        self._metrics.record_match(result.confidence.value)

                    matched.append(updated)

            return matched

        # First attempt: high-precision matcher
        matched_people = _match_people_to_linkedin(people)
        if not matched_people:
            matched_people = [dict(p) for p in people]
        missing_profiles = [p for p in matched_people if not p.get("linkedin_profile")]

        # Fallback: browser search for any still missing
        if missing_profiles:
            fallback_people = [
                {
                    "name": p.get("name", ""),
                    "title": p.get("job_title", ""),
                    "affiliation": p.get("employer", ""),
                    "role_type": p.get("role_type", ""),
                    "department": p.get("specialty", ""),
                    "location": p.get("location", ""),
                }
                for p in missing_profiles
                if p.get("name")
            ]

            if fallback_people:
                fallback_profiles = self._find_linkedin_profiles_batch(
                    fallback_people,
                    story_title=story.title,
                    story_category=story.category,
                )

                fallback_map = {
                    fp.get("name", "").lower(): fp for fp in (fallback_profiles or [])
                }

                for person in matched_people:
                    if person.get("linkedin_profile"):
                        continue
                    key = person.get("name", "").lower()
                    if key in fallback_map:
                        fp = fallback_map[key]
                        person["linkedin_profile"] = fp.get("linkedin_url", "")
                        person["linkedin_profile_type"] = fp.get(
                            "profile_type", "personal"
                        )
                        person["match_confidence"] = fp.get(
                            "match_confidence", "medium"
                        )
                        if person["linkedin_profile"]:
                            self._metrics.record_match(
                                person.get("match_confidence", "medium")
                            )

        return matched_people

    def enrich_pending_stories(self) -> tuple[int, int]:
        """
        Enrich all pending stories with direct and indirect people + LinkedIn profiles.

        Flow:
        1) Normalize direct people from story data; if missing, extract from title/summary
        2) Build indirect people from organizations and employer leadership
        3) Match all people to LinkedIn profiles with validation
        4) Persist direct_people/indirect_people

        Returns (enriched_count, skipped_count).
        """
        stories = self.db.get_stories_needing_enrichment()

        if not stories:
            logger.info("No stories pending enrichment")
            return (0, 0)

        total = len(stories)
        logger.info(f"Enriching {total} stories...")

        enriched = 0
        skipped = 0

        for i, story in enumerate(stories, 1):
            try:
                logger.info(f"[{i}/{total}] Enriching: {story.title}")
                orgs: set[str] = set(story.organizations or [])

                # Seed with NER-based extraction (fast, no API cost)
                ner_people, ner_orgs = self._extract_direct_people_via_ner(story)
                orgs.update(ner_orgs)

                # 1) Direct people: normalize existing or extract fresh
                direct_people = self._normalize_people_records(
                    story.direct_people, source="direct"
                )

                if ner_people:
                    direct_people.extend(ner_people)

                needs_richer_context = not direct_people or any(
                    not (p.get("employer") and p.get("job_title"))
                    for p in direct_people
                )

                if needs_richer_context:
                    logger.info(
                        "  Extracting direct people from story text for enrichment..."
                    )
                    self._metrics.gemini_calls += 1
                    llm_people, extracted_orgs = self._extract_direct_people_from_story(
                        story
                    )
                    orgs.update(extracted_orgs)
                    direct_people.extend(llm_people)

                if not direct_people:
                    # Legacy fallback using existing extraction prompt
                    self._metrics.gemini_calls += 1
                    result = self._extract_orgs_and_people(story)
                    if result:
                        orgs.update(result.get("organizations", []))
                        direct_people.extend(
                            self._normalize_people_records(
                                result.get("direct_people"), source="direct"
                            )
                        )

                # Enrich org list with employers from direct people
                for person in direct_people:
                    employer = person.get("employer", "")
                    if employer:
                        orgs.add(employer)

                direct_people = self._dedupe_people(direct_people)

                # Filter out invalid person names (org names, duplicates, etc.)
                direct_people = self._filter_valid_people(direct_people)

                # 2) Indirect people from organizations/leaders
                indirect_people = self._build_indirect_people(
                    orgs, story, direct_people
                )

                # 3) Limit people count before LinkedIn matching to reduce API calls
                max_people = Config.MAX_PEOPLE_PER_STORY
                if len(direct_people) > max_people:
                    logger.info(
                        f"  Limiting direct_people: {len(direct_people)} -> {max_people}"
                    )
                    direct_people = direct_people[:max_people]
                if len(indirect_people) > max_people:
                    logger.info(
                        f"  Limiting indirect_people: {len(indirect_people)} -> {max_people}"
                    )
                    indirect_people = indirect_people[:max_people]

                # 4) LinkedIn matching
                direct_people = self._attach_linkedin_profiles(direct_people, story)
                indirect_people = self._attach_linkedin_profiles(indirect_people, story)

                # 5) Final deduplication after all enrichment (catch duplicates from different org name variations)
                all_people = direct_people + indirect_people
                all_people = self._dedupe_people(all_people)
                # Split back into direct and indirect
                direct_people = [p for p in all_people if p.get("source") == "direct"]
                indirect_people = [p for p in all_people if p.get("source") != "direct"]

                # 6) Persist + validate
                story.organizations = sorted(org for org in orgs if org)
                story.direct_people = direct_people
                story.indirect_people = indirect_people

                if not direct_people and not indirect_people:
                    logger.info("  ⏭ No people discovered; marking as low-confidence")
                    story.enrichment_status = "enriched"
                    story.enrichment_quality = "low"
                    log = (
                        story.enrichment_log
                        if isinstance(story.enrichment_log, dict)
                        else {}
                    )
                    log["note"] = "No direct or indirect people extracted"
                    log["completed_at"] = datetime.now().isoformat()
                    story.enrichment_log = log
                    self.db.update_story(story)
                    skipped += 1
                    continue

                success, qa_result = self.enrich_story_atomic(story, validate=True)
                if success:
                    enriched += 1
                    if qa_result and qa_result.warnings:
                        logger.info(f"  ⚠ Warnings: {len(qa_result.warnings)}")
                else:
                    skipped += 1

            except KeyboardInterrupt:
                logger.warning(f"\nEnrichment interrupted by user at story {i}/{total}")
                break
            except Exception as e:
                logger.error(f"  ! Error: {e}")
                story.enrichment_status = "error"
                # Phase 4: Track error in enrichment log
                log = (
                    story.enrichment_log
                    if isinstance(story.enrichment_log, dict)
                    else {}
                )
                log["error"] = str(e)
                log["error_at"] = datetime.now().isoformat()
                story.enrichment_log = log
                self.db.update_story(story)
                self._metrics.validation_errors += 1
                skipped += 1
                continue

        # Phase 4: Log metrics summary
        logger.info(f"Enrichment complete: {enriched} enriched, {skipped} skipped")
        self.log_metrics_summary()
        return (enriched, skipped)

    def _extract_orgs_and_people(self, story: Story) -> dict | None:
        """Extract organizations and people from a story using AI with Google Search grounding.

        Uses Google Search to fetch the actual source article content for more thorough
        extraction of people mentioned in the story.
        """
        sources_str = (
            ", ".join(story.source_links[:5])
            if story.source_links
            else "No sources provided"
        )

        # Build a prompt that instructs the AI to search the source URL for people
        search_prompt = f"""Find all people mentioned in this news story by searching the source URL.

STORY TITLE: {story.title}

STORY SUMMARY: {story.summary}

SOURCE URL TO SEARCH: {sources_str}

TASK: Search the source URL and extract:
1. All researchers, scientists, professors named in the article
2. Any executives, leaders, or spokespersons quoted
3. Authors of the study if it's research
4. Anyone receiving awards or honors

Look for:
- Names in quotes (people who said something)
- Names linked to profile pages
- Names in bylines or "about the author" sections
- Names in research paper citations
- "Senior author", "lead researcher", "principal investigator" mentions

Return a JSON object:
{{
  "organizations": ["Org Name 1", "Org Name 2"],
  "direct_people": [
    {{"name": "Full Name", "title": "Their Title/Role", "affiliation": "Their Institution"}}
  ]
}}

If nothing found, return: {{"organizations": [], "direct_people": []}}

Return ONLY valid JSON, no explanation."""

        try:
            # Use Gemini with Google Search grounding to fetch source content
            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_TEXT,
                contents=search_prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "max_output_tokens": 2000,
                },
                endpoint="source_extraction",
            )

            if not response.text:
                logger.warning("Empty response from source article extraction")
                # Fall back to non-grounded extraction
                return self._extract_orgs_and_people_fallback(story)

            response_text = strip_markdown_code_block(response.text)

            data = json.loads(response_text)
            orgs = data.get("organizations", [])
            people = data.get("direct_people", [])

            logger.info(
                f"  Extracted from source: {len(orgs)} orgs, {len(people)} people"
            )
            for person in people:
                logger.info(
                    f"    → {person.get('name', 'Unknown')} ({person.get('title', '')})"
                )

            return {
                "organizations": orgs,
                "direct_people": people,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse enrichment JSON: {e}")
            return self._extract_orgs_and_people_fallback(story)
        except Exception as e:
            logger.error(f"Error extracting orgs/people with search: {e}")
            return self._extract_orgs_and_people_fallback(story)

    def _extract_orgs_and_people_fallback(self, story: Story) -> dict | None:
        """Fallback extraction without Google Search (uses only summary text)."""
        sources_str = (
            ", ".join(story.source_links[:5])
            if story.source_links
            else "No sources provided"
        )
        prompt = Config.STORY_ENRICHMENT_PROMPT.format(
            story_title=story.title,
            story_summary=story.summary,
            story_sources=sources_str,
        )

        try:
            response_text = self._get_ai_response(prompt)
            if not response_text:
                return None

            # Clean up response - sometimes AI adds markdown code blocks
            response_text = strip_markdown_code_block(response_text)

            data = json.loads(response_text)
            return {
                "organizations": data.get("organizations", []),
                "direct_people": data.get("direct_people", []),
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse enrichment JSON (fallback): {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting orgs/people (fallback): {e}")
            return None

    def find_indirect_people(self) -> tuple[int, int]:
        """
        Find indirect people (org leadership) for organizations in enriched stories.
        Returns tuple of (enriched_count, skipped_count).
        """
        # Get stories that have organizations but no indirect_people yet
        stories = self._get_stories_needing_indirect_people()

        if not stories:
            logger.info("No stories need indirect people enrichment")
            return (0, 0)

        total = len(stories)
        logger.info(f"Finding indirect people for {total} stories...")

        enriched = 0
        skipped = 0

        for i, story in enumerate(stories, 1):
            try:
                logger.info(f"[{i}/{total}] Finding indirect people for: {story.title}")

                indirect_people: list[dict] = []
                for org in story.organizations:
                    # Pass story context for context-aware leader selection
                    people = self._get_indirect_people(
                        org,
                        story_category=story.category,
                        story_title=story.title,
                    )
                    if people:
                        indirect_people.extend(people)
                        logger.info(f"  ✓ {org}: {len(people)} indirect people found")
                    else:
                        logger.info(f"  ⏭ {org}: No indirect people found")

                # Remove duplicates: indirect_people who are already in direct_people
                if indirect_people and story.direct_people:
                    direct_names = {
                        p.get("name", "").lower().strip()
                        for p in story.direct_people
                        if p.get("name")
                    }

                    # Helper to check if an indirect person matches any direct person
                    def is_duplicate(person_name: str) -> bool:
                        person_lower = person_name.lower().strip()
                        if person_lower in direct_names:
                            return True

                        # Fuzzy match: check if names share first and last name parts
                        person_parts = person_lower.split()
                        if len(person_parts) >= 2:
                            person_first = person_parts[0]
                            person_last = person_parts[-1]
                            for direct_name in direct_names:
                                direct_parts = direct_name.split()
                                if len(direct_parts) >= 2:
                                    if (
                                        direct_parts[0] == person_first
                                        and direct_parts[-1] == person_last
                                    ):
                                        return True
                        return False

                    original_count = len(indirect_people)
                    indirect_people = [
                        person
                        for person in indirect_people
                        if not is_duplicate(person.get("name", ""))
                    ]
                    if len(indirect_people) < original_count:
                        logger.info(
                            f"  Removed {original_count - len(indirect_people)} duplicates already in direct_people"
                        )

                # Filter out invalid names (org/department names, placeholders)
                invalid_terms = {
                    "school",
                    "college",
                    "university",
                    "institute",
                    "department",
                    "division",
                    "center",
                    "centre",
                    "lab",
                    "laboratory",
                    "office",
                    "foundation",
                    "association",
                    "society",
                    "committee",
                    "board",
                    "faculty",
                    "tba",
                    "tbd",
                    "unknown",
                    "n/a",
                    "none",
                    "placeholder",
                }
                valid_indirect_people = []
                for person in indirect_people:
                    name = person.get("name", "").strip()
                    name_lower = name.lower()
                    if any(term in name_lower for term in invalid_terms):
                        logger.debug(f"  Skipping invalid indirect person name: {name}")
                        continue
                    if len(name) < 5 or " " not in name:
                        continue
                    valid_indirect_people.append(person)
                if len(valid_indirect_people) < len(indirect_people):
                    logger.info(
                        f"  Filtered {len(indirect_people) - len(valid_indirect_people)} invalid indirect people"
                    )
                indirect_people = valid_indirect_people

                # Look up LinkedIn profiles for indirect_people using enhanced matching
                if indirect_people:
                    people_for_lookup = [
                        {
                            "name": person.get("name", ""),
                            "title": person.get("title", ""),
                            "affiliation": person.get("organization", ""),
                            "role_type": person.get("role_type", ""),
                            "department": person.get("department", ""),
                            "location": person.get("location", ""),
                        }
                        for person in indirect_people
                        if person.get("name")
                    ]

                    if people_for_lookup:
                        linkedin_profiles = self._find_linkedin_profiles_batch(
                            people_for_lookup,
                            story_title=story.title,
                            story_category=story.category,
                        )

                        # Update indirect_people with found LinkedIn profiles
                        if linkedin_profiles:
                            profiles_by_name = {
                                p.get("name", "").lower(): p for p in linkedin_profiles
                            }
                            for person in indirect_people:
                                name_lower = person.get("name", "").lower()
                                if name_lower in profiles_by_name:
                                    profile = profiles_by_name[name_lower]
                                    url = profile.get("linkedin_url", "")
                                    p_type = profile.get("profile_type", "")

                                    if url and "/in/" in url:
                                        person["linkedin_profile"] = url
                                        person["linkedin_profile_type"] = "personal"
                                    else:
                                        person["linkedin_profile"] = ""
                                        person["linkedin_profile_type"] = p_type or ""
                            logger.info(
                                f"  ✓ Found {len(linkedin_profiles)} LinkedIn profiles for indirect people"
                            )

                story.indirect_people = indirect_people
                self.db.update_story(story)

                if indirect_people:
                    enriched += 1
                else:
                    skipped += 1

            except KeyboardInterrupt:
                logger.warning(
                    f"\nIndirect-people enrichment interrupted at story {i}/{total}"
                )
                break
            except Exception as e:
                logger.error(f"  ! Error: {e}")
                skipped += 1
                continue

        logger.info(
            f"Indirect-people enrichment complete: {enriched} enriched, {skipped} skipped"
        )
        return (enriched, skipped)

    def _get_indirect_people(
        self,
        organization_name: str,
        story_category: str = "",
        story_title: str = "",
    ) -> list[dict]:
        """Get indirect people (org leadership) using AI with story context.

        Args:
            organization_name: Name of the organization to find leaders for
            story_category: Category of the story (Research, Business, etc.)
            story_title: Title of the story for additional context

        Returns:
            List of indirect-person dictionaries with name, title, organization, role_type, etc.
        """
        # Format prompt with story context for context-aware leader selection
        prompt = Config.INDIRECT_PEOPLE_PROMPT.format(
            organization_name=organization_name,
            story_category=story_category or "General",
            story_title=story_title or "N/A",
        )

        try:
            response_text = self._get_ai_response(prompt)
            if not response_text:
                return []

            # Clean up response
            response_text = strip_markdown_code_block(response_text)

            data = json.loads(response_text)
            leaders = data.get("leaders", [])

            # Validate and filter each leader - filter out AI explanation text
            validated_leaders = []
            for leader in leaders:
                name = leader.get("name", "").strip()
                if not name:
                    continue

                # Use centralized validation to reject invalid names
                if is_invalid_person_name(name):
                    logger.debug(f"Filtering invalid indirect person: '{name[:50]}'")
                    continue

                # Set defaults and add to validated list
                leader.setdefault("role_type", "executive")
                leader.setdefault("department", "")
                leader.setdefault("location", "")
                leader.setdefault("linkedin_profile", "")
                validated_leaders.append(leader)

            if len(validated_leaders) < len(leaders):
                logger.debug(
                    f"Filtered indirect people: {len(leaders)} -> {len(validated_leaders)}"
                )

            return validated_leaders

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse leaders JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting indirect people: {e}")
            return []

    def _get_stories_needing_indirect_people(self) -> list[Story]:
        """Get stories with organizations but no indirect_people yet."""
        stories = []
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM stories
                WHERE enrichment_status = 'enriched'
                AND organizations IS NOT NULL
                AND organizations != '[]'
                AND (indirect_people IS NULL OR indirect_people = '[]')
                ORDER BY id DESC
            """)
            for row in cursor.fetchall():
                stories.append(Story.from_row(row))
        return stories

    def _find_linkedin_profiles_batch(
        self,
        people: list[dict],
        story_title: str = "",
        story_category: str = "",
    ) -> list[dict]:
        """
        Search for LinkedIn profiles for a batch of people using Playwright browser search.

        Uses LinkedInCompanyLookup.find_person_profile which performs real Google/Bing
        searches via a headless browser to find actual LinkedIn profile URLs.

        Phase 2: Uses validation cache to avoid redundant API calls.

        Args:
            people: List of person dictionaries with name, title, affiliation
            story_title: Optional story title for context
            story_category: Optional story category for context

        Returns:
            List of validated profile dictionaries
        """
        if not people:
            return []

        # Deduplicate people by normalized name to avoid searching the same person multiple times
        seen_names: set[str] = set()
        deduplicated_people: list[dict] = []
        for person in people:
            name = person.get("name", "").strip()
            if not name:
                continue
            # Normalize name for deduplication (lowercase, single spaces)
            normalized_name = " ".join(name.lower().split())
            if normalized_name in seen_names:
                logger.debug(f"Skipping duplicate person in batch: {name}")
                continue
            seen_names.add(normalized_name)
            deduplicated_people.append(person)

        if len(deduplicated_people) < len(people):
            logger.info(
                f"Deduplicated batch: {len(people)} -> {len(deduplicated_people)} people"
            )

        # Phase 2: Check cache first, separate people into cached and uncached
        cached_profiles: list[dict] = []
        uncached_people: list[dict] = []

        for person in deduplicated_people:
            name = person.get("name", "")
            org = person.get("affiliation", "")
            if not name:
                continue

            # Check cache
            cached = self._cache.get_person_validation(name, org)
            if cached is not None:
                logger.debug(f"Cache hit for {name} at {org}")
                cached_profiles.append(cached)
            else:
                uncached_people.append(person)

        # If all people were cached, return cached results
        if not uncached_people:
            logger.info(f"All {len(cached_profiles)} profiles found in cache")
            return cached_profiles

        # Try Voyager API first (faster, no CAPTCHA), then fall back to browser-based search
        from linkedin_voyager_client import HybridLinkedInLookup

        valid_profiles = []
        searched_in_session: set[str] = set()  # Track searches within this session
        with HybridLinkedInLookup() as lookup:
            for person in uncached_people:
                name = person.get("name", "")
                if not name:
                    continue

                # Skip if already searched in this session (covers case where same person
                # is in both direct_people and indirect_people with different affiliations)
                session_key = " ".join(name.lower().split())
                if session_key in searched_in_session:
                    logger.debug(f"Skipping already-searched person: {name}")
                    continue
                searched_in_session.add(session_key)

                title = person.get("title", "")
                affiliation = person.get("affiliation", "")
                role_type = person.get("role_type", "")
                department = person.get("department", "")
                location = person.get("location", "")

                try:
                    # Use HybridLinkedInLookup - Voyager API first, browser fallback
                    url, urn, confidence = lookup.find_person(
                        name=name,
                        company=affiliation,
                        title=title,
                        location=location,
                        department=department,
                        role_type=role_type,
                    )

                    if url and "linkedin.com/in/" in url:
                        # Triangulation validation: cross-check profile against expected data
                        is_valid, validation_score, validation_signals = (
                            lookup.validate_profile_match(
                                linkedin_url=url,
                                expected_name=name,
                                expected_company=affiliation,
                                expected_title=title,
                                expected_location=location,
                            )
                        )

                        if not is_valid and confidence != "high":
                            # Reject low-confidence match that failed validation
                            logger.warning(
                                f"Rejecting {name} match - validation failed "
                                f"(score={validation_score:.1f}, signals={validation_signals})"
                            )
                            # Cache as not found to avoid re-searching
                            self._cache.set_person_validation(
                                name,
                                affiliation,
                                {
                                    "name": name,
                                    "affiliation": affiliation,
                                    "linkedin_url": "",
                                },
                            )
                            continue

                        # Determine final confidence based on validation
                        final_confidence = confidence
                        if is_valid and validation_score >= 7.0:
                            final_confidence = "validated_high"
                        elif is_valid:
                            final_confidence = "validated_medium"

                        # Extract username from URL
                        username = (
                            url.split("linkedin.com/in/")[-1].rstrip("/").split("?")[0]
                        )
                        validated_profile = {
                            "name": name,
                            "title": title,
                            "affiliation": affiliation,
                            "handle": f"@{username}" if username else None,
                            "linkedin_url": url,
                            "linkedin_urn": urn,  # Store URN for @mentions
                            "profile_type": "person",
                            "match_confidence": final_confidence,
                            "validation_score": validation_score,
                            "validation_signals": validation_signals,
                        }
                        valid_profiles.append(validated_profile)

                        # Phase 2: Cache the result
                        self._cache.set_person_validation(
                            name,
                            affiliation,
                            validated_profile,
                        )

                        logger.info(
                            f"Found LinkedIn profile ({final_confidence}): {name} -> {url} "
                            f"(validation: {validation_score:.1f})"
                        )
                    else:
                        # No personal profile found - try department fallback for leaders
                        dept_url = None
                        if department and affiliation:
                            # Search for department/school LinkedIn page
                            dept_search = f"{affiliation} {department}"
                            dept_url_result, _ = lookup.find_company(dept_search)
                            if dept_url_result:
                                logger.info(
                                    f"  → Using department fallback for {name}: {dept_url_result}"
                                )
                                dept_url = dept_url_result

                        if dept_url:
                            # Use department LinkedIn page as fallback
                            validated_profile = {
                                "name": name,
                                "title": title,
                                "affiliation": affiliation,
                                "handle": None,
                                "linkedin_url": dept_url,
                                "profile_type": "dept_fallback",
                            }
                            valid_profiles.append(validated_profile)

                            # Cache with department fallback
                            self._cache.set_person_validation(
                                name,
                                affiliation,
                                validated_profile,
                            )
                        else:
                            # Cache negative result to avoid repeated searches
                            self._cache.set_person_validation(
                                name,
                                affiliation,
                                {
                                    "name": name,
                                    "affiliation": affiliation,
                                    "linkedin_url": "",
                                },
                            )
                            logger.debug(f"No LinkedIn profile found for: {name}")

                except Exception as e:
                    logger.warning(f"Error searching for {name}: {e}")

        # Combine cached and newly found profiles (filter out empty URLs from cache)
        all_profiles = [
            p for p in cached_profiles if p.get("linkedin_url")
        ] + valid_profiles
        return all_profiles

    def _find_profile_with_fallback(
        self,
        person: dict,
        org_linkedin_urls: dict[str, str],
        lookup: "LinkedInCompanyLookup",
    ) -> tuple[str, str, bool]:
        """
        Find LinkedIn profile for a person with department/organization fallback.

        This method implements the high-precision profile matching strategy:
        1. Search for person's LinkedIn profile
        2. If found with high confidence, use it
        3. If not found OR low confidence, fall back to department/school page if available
        4. If no department page, fall back to organization page

        Args:
            person: Person dictionary with name, company, position, etc.
            org_linkedin_urls: Cache of organization name -> LinkedIn URL
            lookup: LinkedInCompanyLookup instance

        Returns:
            Tuple of (linkedin_url, match_type, is_person_profile)
            - linkedin_url: The URL (person or org)
            - match_type: "person", "dept_fallback", "org_fallback", or "none"
            - is_person_profile: True if this is a personal profile
        """
        from profile_matcher import (
            ProfileMatcher,
            PersonContext,
            RoleType,
            create_person_context,
        )

        name = person.get("name", "")
        org_name = person.get("company", "") or person.get("organization", "")
        department = person.get("department", "")

        if not name:
            return ("", "none", False)

        # Create person context from dictionary
        person_context = create_person_context(person)

        # Get department LinkedIn URL first (more specific fallback)
        dept_url = None
        if department and org_name:
            # Create a cache key for department
            dept_key = f"{org_name} {department}"
            if dept_key in org_linkedin_urls:
                dept_url = org_linkedin_urls[dept_key]
            else:
                # Search for department page - combine org name and department
                # e.g., "Stanford University School of Engineering"
                dept_search = f"{org_name} {department}"
                url, slug = lookup.search_company(dept_search)
                if url:
                    org_linkedin_urls[dept_key] = url
                    dept_url = url
                    logger.debug(
                        f"Found department LinkedIn page: {dept_search} -> {url}"
                    )

        # Get org LinkedIn URL (cached or look up)
        org_url = None
        if org_name:
            if org_name in org_linkedin_urls:
                org_url = org_linkedin_urls[org_name]
            else:
                # Look up org LinkedIn page
                url, slug = lookup.search_company(org_name)
                if url:
                    org_linkedin_urls[org_name] = url
                    org_url = url
                    logger.debug(f"Found org LinkedIn page: {org_name} -> {url}")

        # Prefer department URL over org URL for fallback
        fallback_url = dept_url or org_url
        fallback_type = "dept_fallback" if dept_url else "org_fallback"

        # Use ProfileMatcher for high-precision matching
        matcher = ProfileMatcher(linkedin_lookup=lookup)
        result = matcher.match_person(person_context, org_fallback_url=fallback_url)

        if result.is_person_profile() and result.matched_profile:
            return (result.matched_profile.linkedin_url, "person", True)
        elif result.confidence.value == "org_fallback" and result.org_linkedin_url:
            logger.info(
                f"  → Using {fallback_type} for {name}: {result.org_linkedin_url}"
            )
            return (result.org_linkedin_url, fallback_type, False)
        else:
            return ("", "none", False)

    def find_profiles_with_fallback(self, story: Story) -> dict:
        """
        Find LinkedIn profiles for all people in a story with org fallback.

        This enhanced method uses multi-signal scoring and falls back to
        organization profiles when personal profiles can't be confidently matched.

        Args:
            story: Story object with direct_people, indirect_people, and organizations

        Returns:
            Dictionary with:
            - person_profiles: List of people with matched personal profiles
            - org_fallbacks: List of people using org profile fallback
            - org_urls: Mapping of org name -> LinkedIn URL
        """
        from linkedin_profile_lookup import LinkedInCompanyLookup

        result = {
            "person_profiles": [],
            "org_fallbacks": [],
            "org_urls": {},
            "total_matched": 0,
        }

        all_people = (story.direct_people or []) + (story.indirect_people or [])
        if not all_people:
            return result

        logger.info(f"Finding profiles with fallback for {len(all_people)} people")

        # Pre-fetch org LinkedIn pages
        org_urls = {}
        with LinkedInCompanyLookup(genai_client=self.client) as lookup:
            # Look up organization pages first (for fallback)
            for org in story.organizations or []:
                url, slug = lookup.search_company(org)
                if url:
                    org_urls[org] = url
                    logger.debug(f"  Org page: {org} -> {url}")

            # Now match each person
            for person in all_people:
                name = person.get("name", "")
                if not name:
                    continue

                linkedin_url, match_type, is_person = self._find_profile_with_fallback(
                    person, org_urls, lookup
                )

                if is_person:
                    result["person_profiles"].append(
                        {
                            **person,
                            "linkedin_profile": linkedin_url,
                            "profile_type": "person",
                        }
                    )
                elif match_type == "org_fallback":
                    result["org_fallbacks"].append(
                        {
                            **person,
                            "linkedin_profile": linkedin_url,
                            "profile_type": "org_fallback",
                        }
                    )

        result["org_urls"] = org_urls
        result["total_matched"] = len(result["person_profiles"]) + len(
            result["org_fallbacks"]
        )
        return result

    def populate_linkedin_mentions(self) -> tuple[int, int]:
        """
        Look up and store URNs directly in direct_people.linkedin_urn and indirect_people.linkedin_urn.

        This streamlined approach stores URNs within direct_people and indirect_people rather
        than duplicating data in a separate linkedin_mentions column.

        Returns tuple of (enriched_count, skipped_count).
        """
        from linkedin_profile_lookup import LinkedInCompanyLookup

        # Get stories that have profiles but no URNs
        stories = self._get_stories_needing_urns()

        if not stories:
            logger.info("No stories need LinkedIn URN lookup")
            return (0, 0)

        total = len(stories)
        logger.info(f"Looking up LinkedIn URNs for {total} stories...")

        enriched = 0
        skipped = 0

        # Use a single browser session for efficiency
        with LinkedInCompanyLookup(genai_client=self.client) as lookup:
            for i, story in enumerate(stories, 1):
                try:
                    logger.info(f"[{i}/{total}] Looking up URNs for: {story.title}")

                    # Combine direct_people and indirect_people for lookup
                    all_people = (story.direct_people or []) + (
                        story.indirect_people or []
                    )
                    if not all_people:
                        logger.info("  ⏭ No people to look up")
                        skipped += 1
                        continue

                    # Find people needing URN lookup:
                    # 1. Has personal profile URL but no URN
                    # 2. Has organization profile (wrong type - needs personal profile search)
                    people_needing_lookup = []
                    for p in all_people:
                        profile = p.get("linkedin_profile", "")
                        urn = p.get("linkedin_urn", "")
                        profile_type = p.get("linkedin_profile_type", "")

                        # Has personal profile but no URN
                        if profile and "/in/" in profile and not urn:
                            people_needing_lookup.append(
                                {"person": p, "needs_search": False}
                            )
                        else:
                            # Anything that isn't a personal profile (org fallback, school page, or missing)
                            # should trigger a fresh personal profile search.
                            if (
                                profile_type in {"organization", "org_fallback"}
                                or (profile and "/in/" not in profile)
                                or "urn:li:organization:" in str(urn)
                            ):
                                people_needing_lookup.append(
                                    {"person": p, "needs_search": True}
                                )

                    if not people_needing_lookup:
                        logger.info("  ⏭ All people already have personal URNs")
                        skipped += 1
                        continue

                    logger.info(
                        f"  Found {len(people_needing_lookup)} people to lookup"
                    )

                    urns_found = 0
                    for item in people_needing_lookup:
                        person = item["person"]
                        needs_search = item["needs_search"]
                        name = person.get("name", "Unknown")
                        company = person.get("company", "")
                        position = person.get("position", "")

                        import time

                        if needs_search:
                            # Search for personal profile using lookup.search_person
                            # This uses the shared person cache to avoid duplicate searches
                            logger.info(
                                f"    🔍 Searching for personal profile: {name}"
                            )
                            personal_url = lookup.search_person(
                                name, company, position=position
                            )
                            if personal_url:
                                # Update to personal profile
                                person["linkedin_profile"] = personal_url
                                person["linkedin_profile_type"] = "personal"
                                # Clear wrong org URN
                                person["linkedin_urn"] = None
                                person.pop("linkedin_slug", None)
                                logger.info(
                                    f"    ✓ Found personal profile: {personal_url}"
                                )
                                # Now look up the URN
                                urn = lookup.lookup_person_urn(personal_url)
                                if urn:
                                    person["linkedin_urn"] = urn
                                    urns_found += 1
                                    logger.info(f"    ✓ {name}: {urn}")
                                time.sleep(2.0)
                            else:
                                # Clear org/school fallback to avoid invalid URLs persisting
                                person["linkedin_profile"] = ""
                                person["linkedin_profile_type"] = ""
                                logger.info(f"    ✗ {name}: No personal profile found")
                        else:
                            # Already has personal profile URL, just look up URN
                            url = person.get("linkedin_profile", "")
                            urn = lookup.lookup_person_urn(url)
                            if urn:
                                person["linkedin_urn"] = urn
                                urns_found += 1
                                logger.info(f"    ✓ {name}: {urn}")
                            else:
                                # Profile URL might be invalid/removed - search for new one
                                logger.info(
                                    f"    ⚠ {name}: Existing profile invalid, searching..."
                                )
                                personal_url = lookup.search_person(
                                    name, company, position=position
                                )
                                if personal_url:
                                    person["linkedin_profile"] = personal_url
                                    logger.info(
                                        f"    ✓ Found new profile: {personal_url}"
                                    )
                                    urn = lookup.lookup_person_urn(personal_url)
                                    if urn:
                                        person["linkedin_urn"] = urn
                                        urns_found += 1
                                        logger.info(f"    ✓ {name}: {urn}")
                                    time.sleep(2.0)
                                else:
                                    logger.info(f"    ✗ {name}: No URN found")
                            time.sleep(2.0)

                    # Save updated direct_people and indirect_people
                    self.db.update_story(story)

                    if urns_found > 0:
                        logger.info(
                            f"  ✓ Updated {urns_found}/{len(people_needing_lookup)} URNs"
                        )
                        enriched += 1
                    else:
                        logger.info("  ⏭ No URNs found")
                        skipped += 1

                except KeyboardInterrupt:
                    logger.warning(f"\nURN lookup interrupted at story {i}/{total}")
                    break
                except Exception as e:
                    logger.error(f"  ! Error: {e}")
                    skipped += 1
                    continue

        logger.info(f"URN lookup complete: {enriched} enriched, {skipped} skipped")
        return (enriched, skipped)

    def _search_personal_linkedin(
        self, lookup, name: str, company: str, position: str
    ) -> str | None:
        """Search for a person's personal LinkedIn profile URL using ProfileMatcher.

        This method delegates to ProfileMatcher for high-precision profile matching,
        ensuring consistent search behavior across the codebase.

        Args:
            lookup: LinkedInCompanyLookup instance
            name: Person's name
            company: Company/organization name
            position: Job title/position

        Returns:
            LinkedIn profile URL if found with high confidence, None otherwise
        """
        from profile_matcher import (
            ProfileMatcher,
            PersonContext,
            RoleType,
            create_person_context,
        )

        # Build person dict for create_person_context
        person_dict = {
            "name": name,
            "company": company,
            "position": position,
        }
        person_context = create_person_context(person_dict)

        # Use ProfileMatcher for consistent high-precision matching
        matcher = ProfileMatcher(linkedin_lookup=lookup)
        result = matcher.match_person(person_context, org_fallback_url=None)

        # Only return personal profile URLs, not org fallbacks
        if result.is_person_profile() and result.matched_profile:
            logger.debug(
                f"ProfileMatcher found: {result.matched_profile.linkedin_url} "
                f"(confidence: {result.confidence.value})"
            )
            return result.matched_profile.linkedin_url

        logger.debug(f"ProfileMatcher: No confident match for {name}")
        return None

    def _get_stories_needing_urns(self) -> list[Story]:
        """Get stories with people needing personal LinkedIn URNs.

        Includes people who:
        1. Have linkedin_profile but no linkedin_urn
        2. Have organization URN instead of personal URN (wrong type)
        """
        stories = []
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            # Stories with direct_people or indirect_people containing linkedin_profile
            cursor.execute("""
                SELECT * FROM stories
                WHERE (direct_people LIKE '%linkedin_profile%')
                   OR (indirect_people LIKE '%linkedin_profile%')
                ORDER BY id DESC
            """)
            for row in cursor.fetchall():
                story = Story.from_row(row)
                all_people = (story.direct_people or []) + (story.indirect_people or [])
                if all_people:
                    needs_lookup = False
                    for p in all_people:
                        has_profile = bool(p.get("linkedin_profile"))
                        has_urn = bool(p.get("linkedin_urn"))
                        profile_type = p.get("linkedin_profile_type", "")
                        urn = p.get("linkedin_urn", "")

                        # Need lookup if: has personal profile URL but no URN
                        if (
                            has_profile
                            and "/in/" in p.get("linkedin_profile", "")
                            and not has_urn
                        ):
                            needs_lookup = True
                            break
                        # Or if: has organization URN (wrong - person needs personal URN)
                        if profile_type == "organization" or (
                            has_urn and "urn:li:organization:" in str(urn)
                        ):
                            needs_lookup = True
                            break
                    if needs_lookup:
                        stories.append(story)
        return stories

    def _get_ai_response(self, prompt: str) -> str | None:
        """Get response from AI (local LLM or Gemini)."""
        if self.local_client:
            try:
                response = self.local_client.chat.completions.create(
                    model=Config.LM_STUDIO_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content
                if content:
                    return content.strip()
            except Exception as e:
                logger.warning(f"Local AI failed: {e}. Falling back to Gemini.")

        # Fallback to Gemini
        response = api_client.gemini_generate(
            client=self.client,
            model=Config.MODEL_TEXT,
            contents=prompt,
            endpoint="ai_response",
        )
        return response.text.strip() if response.text else None

    def _enrich_story(self, story: Story) -> tuple[str, str]:
        """
        Enrich a single story with company mentions.
        Returns tuple of (mention_or_reason, status).
        Status can be: "completed", "skipped", or "error"
        """
        try:
            # Build enrichment prompt
            prompt = self._build_enrichment_prompt(story)

            # Get company mention from AI
            mention = self._get_company_mention(prompt)

            # Validate the response
            mention = self._validate_mention(mention)

            return (mention, "completed")

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.error(
                    f"Enrichment failed: API quota exceeded (429 RESOURCE_EXHAUSTED)"
                )
                return ("API quota exceeded", "error")
            logger.error(f"Enrichment error for story {story.id}: {e}")
            return (str(e), "error")

    def _get_company_mention(self, prompt: str) -> str:
        """Get company mention from AI model.
        Returns the mention text or NO_COMPANY_MENTION.
        """
        # Use local LLM if available
        if self.local_client:
            try:
                logger.info("Using local LLM for company mention enrichment...")
                response = self.local_client.chat.completions.create(
                    model=Config.LM_STUDIO_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content
                if content:
                    return content.strip()
            except Exception as e:
                logger.warning(f"Local enrichment failed: {e}. Falling back to Gemini.")

        # Fallback to Gemini
        response = api_client.gemini_generate(
            client=self.client,
            model=Config.MODEL_TEXT,
            contents=prompt,
            endpoint="company_mention",
        )
        if not response.text:
            logger.warning("Empty response from Gemini during enrichment")
            return NO_COMPANY_MENTION
        return response.text.strip()

    def _build_enrichment_prompt(self, story: Story) -> str:
        """Build the enrichment prompt for a story."""
        sources_str = (
            ", ".join(story.source_links[:5])
            if story.source_links
            else "No sources provided"
        )
        return Config.COMPANY_MENTION_PROMPT.format(
            story_title=story.title,
            story_summary=story.summary,
            story_sources=sources_str,
        )

    def _validate_mention(self, mention: str) -> str:
        """
        Validate mention output.
        Returns either a valid sentence or NO_COMPANY_MENTION.
        """
        mention = mention.strip()

        # If AI returned multiple lines, take the first non-empty line that looks like a sentence
        if "\n" in mention:
            lines = [line.strip() for line in mention.split("\n") if line.strip()]
            # Find the first line that looks like a valid sentence (ends with punctuation)
            for line in lines:
                if line.endswith((".", "!", "?")) and not line.startswith(
                    ("NO_COMPANY", "If", "Note:", "Respond")
                ):
                    mention = line
                    break
            else:
                # No valid sentence found in any line
                logger.warning(
                    f"Invalid mention (no valid sentence in multi-line response): {mention[:50]}"
                )
                return NO_COMPANY_MENTION

        # If it's the no-mention marker, return it as-is
        if mention == NO_COMPANY_MENTION:
            return NO_COMPANY_MENTION

        # Validate that it's a single sentence
        # A valid sentence should:
        # 1. Not be empty
        # 2. End with a period (or be very short and professional)
        # 3. Not contain newlines (single sentence rule)
        # 4. Be reasonable length (max 350 chars for longer org names)

        if not mention:
            return NO_COMPANY_MENTION

        if "\n" in mention:
            logger.warning(f"Invalid mention (contains newlines): {mention[:50]}")
            return NO_COMPANY_MENTION

        if len(mention) > 350:
            logger.warning(f"Invalid mention (too long): {mention[:50]}")
            return NO_COMPANY_MENTION

        # Ensure it ends with proper punctuation
        if not mention.endswith((".", "!", "?")):
            # If it doesn't end with punctuation, check if it looks like a sentence
            # Default to rejection if ambiguous
            logger.warning(f"Invalid mention (no ending punctuation): {mention}")
            return NO_COMPANY_MENTION

        # Check for multiple sentences - but allow abbreviations like Inc., Ltd., Corp., etc.
        # Remove common abbreviations before counting periods
        temp_mention = mention
        for abbrev in [
            "Inc.",
            "Ltd.",
            "Corp.",
            "Co.",
            "LLC.",
            "L.L.C.",
            "P.L.C.",
            "S.A.",
            "N.V.",
            "GmbH.",
            "AG.",
            "Dr.",
            "Prof.",
            "Mr.",
            "Ms.",
            "Mrs.",
        ]:
            temp_mention = temp_mention.replace(abbrev, "ABBREV")

        sentence_count = (
            temp_mention.count(".") + temp_mention.count("!") + temp_mention.count("?")
        )
        if sentence_count > 1:
            logger.warning(f"Invalid mention (multiple sentences): {mention[:50]}")
            return NO_COMPANY_MENTION

        # Check for list indicators (these shouldn't be in a single sentence)
        # But allow hyphens in company names
        if any(indicator in mention for indicator in ["1.", "2.", "•", "* "]):
            logger.warning(f"Invalid mention (contains list): {mention[:50]}")
            return NO_COMPANY_MENTION

        # Check for inappropriate elements
        if any(
            bad in mention.lower()
            for bad in ["@", "#", "no_company_mention", "hashtag", "tag"]
        ):
            logger.warning(f"Invalid mention (inappropriate elements): {mention[:50]}")
            return NO_COMPANY_MENTION

        # If all validation passes, it's acceptable
        return mention

    def _resolve_linkedin_urn(self, company_name: str) -> str | None:
        """Try to resolve a LinkedIn organization URN for the company name.

        Uses the LinkedIn Organizations endpoint with `q=vanityName` and a few
        heuristic attempts (original name and a slugified form).
        Returns a URN string like 'urn:li:organization:12345' on success or None.
        """
        if not Config.LINKEDIN_ACCESS_TOKEN:
            return None

        headers = {
            "Authorization": f"Bearer {Config.LINKEDIN_ACCESS_TOKEN}",
            "X-Restli-Protocol-Version": "2.0.0",
        }

        # Build simple slug candidate
        slug = re.sub(r"[^a-z0-9-]", "", company_name.lower().replace(" ", "-"))

        candidates = [company_name, slug]

        for candidate in candidates:
            if not candidate:
                continue
            params = {"q": "vanityName", "vanityName": candidate}
            try:
                resp = api_client.linkedin_request(
                    method="GET",
                    url="https://api.linkedin.com/v2/organizations",
                    headers=headers,
                    params=params,
                    timeout=10,
                    endpoint="org_lookup",
                )
            except requests.RequestException:
                continue

            if resp.status_code != 200:
                continue

            data = resp.json()
            elements = data.get("elements", [])
            if not elements:
                continue

            # Iterate returned elements and verify match heuristically
            for elem in elements:
                org_id = elem.get("id") or elem.get("organization", {}).get("id")
                if not org_id:
                    continue

                try:
                    if self._confirm_organization(company_name, elem):
                        return f"urn:li:organization:{org_id}"
                except Exception:
                    # If verification fails, still fall back to first id
                    return f"urn:li:organization:{org_id}"

        return None

    def _confirm_organization(self, company_name: str, org_elem: dict) -> bool:
        """Heuristic check whether org_elem matches company_name.

        Checks localizedName, vanityName and some token overlap. Conservative by design.
        """
        try:
            name = company_name.lower().strip()
            localized = "".join(org_elem.get("localizedName", "") or "").lower()
            vanity = "".join(org_elem.get("vanityName", "") or "").lower()

            # Exact match or substring match
            if localized and (
                name == localized or name in localized or localized in name
            ):
                return True
            if vanity and (name == vanity or name in vanity or vanity in name):
                return True

            # Token overlap (at least 2 tokens in common)
            def tokens(s: str):
                return {t for t in re.split(r"\W+", s) if t}

            name_tokens = tokens(name)
            if localized:
                loc_tokens = tokens(localized)
                if len(name_tokens & loc_tokens) >= 2:
                    return True
            if vanity:
                van_tokens = tokens(vanity)
                if len(name_tokens & van_tokens) >= 2:
                    return True

        except Exception as e:
            logger.debug(f"Error in _confirm_organization: {e}")
        return False

    def get_enrichment_stats(self) -> dict:
        """Get enrichment statistics."""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()

            # Count total enriched stories
            cursor.execute(
                "SELECT COUNT(*) FROM stories WHERE enrichment_status = 'enriched'"
            )
            total_enriched = cursor.fetchone()[0]

            # Count stories with organizations
            cursor.execute(
                """SELECT COUNT(*) FROM stories
                   WHERE enrichment_status = 'enriched'
                   AND organizations IS NOT NULL
                   AND organizations != '[]'"""
            )
            with_orgs = cursor.fetchone()[0]

            # Count stories with people (direct_people or indirect_people)
            cursor.execute(
                """SELECT COUNT(*) FROM stories
                   WHERE enrichment_status = 'enriched'
                   AND (
                       (direct_people IS NOT NULL AND direct_people != '[]')
                       OR (indirect_people IS NOT NULL AND indirect_people != '[]')
                   )"""
            )
            with_people = cursor.fetchone()[0]

            # Count stories with LinkedIn URNs in direct_people or indirect_people
            cursor.execute(
                """SELECT COUNT(*) FROM stories
                   WHERE enrichment_status = 'enriched'
                   AND (direct_people LIKE '%linkedin_urn%' OR indirect_people LIKE '%linkedin_urn%')"""
            )
            with_urns = cursor.fetchone()[0]

            # Count pending
            cursor.execute(
                "SELECT COUNT(*) FROM stories WHERE enrichment_status = 'pending'"
            )
            pending = cursor.fetchone()[0]

        return {
            "pending": pending,
            "total_enriched": total_enriched,
            "with_orgs": with_orgs,
            "with_people": with_people,
            "with_urns": with_urns,
        }


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for company_mention_enricher module."""
    import os
    import tempfile

    from test_framework import TestSuite

    suite = TestSuite("Company Mention Enricher Tests")

    def test_confirm_organization_exact_match():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            elem = {
                "localizedName": "SOLVE Chemistry",
                "vanityName": "solve-chemistry",
                "id": 123,
            }
            assert enricher._confirm_organization("SOLVE Chemistry", elem)
        finally:
            os.unlink(db_path)

    def test_confirm_organization_token_overlap():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            elem = {
                "localizedName": "Greentown Labs",
                "vanityName": "greentownlabs",
                "id": 456,
            }
            assert enricher._confirm_organization("Greentown Labs", elem)
        finally:
            os.unlink(db_path)

    suite.add_test("Confirm org - exact match", test_confirm_organization_exact_match)
    suite.add_test(
        "Confirm org - token overlap", test_confirm_organization_token_overlap
    )

    def test_validate_mention_valid_sentence():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            result = enricher._validate_mention(
                "This work integrates BASF's established catalysis technology."
            )
            assert (
                result
                == "This work integrates BASF's established catalysis technology."
            )
        finally:
            os.unlink(db_path)

    def test_validate_mention_no_company():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            result = enricher._validate_mention(NO_COMPANY_MENTION)
            assert result == NO_COMPANY_MENTION
        finally:
            os.unlink(db_path)

    def test_validate_mention_empty():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            result = enricher._validate_mention("")
            assert result == NO_COMPANY_MENTION
        finally:
            os.unlink(db_path)

    def test_validate_mention_with_newlines():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            result = enricher._validate_mention("Line 1.\nLine 2.")
            assert result == NO_COMPANY_MENTION
        finally:
            os.unlink(db_path)

    def test_validate_mention_with_hashtag():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            result = enricher._validate_mention(
                "This mentions Dow. #ChemicalEngineering"
            )
            assert result == NO_COMPANY_MENTION
        finally:
            os.unlink(db_path)

    def test_validate_mention_with_list():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            result = enricher._validate_mention("Companies: 1. Dow 2. BASF.")
            assert result == NO_COMPANY_MENTION
        finally:
            os.unlink(db_path)

    def test_validate_mention_missing_punctuation():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            result = enricher._validate_mention("This mentions Dow")
            assert result == NO_COMPANY_MENTION
        finally:
            os.unlink(db_path)

    def test_build_enrichment_prompt():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            story = Story()
            story.title = "BASF Innovation"
            story.summary = "Development by BASF team"
            story.source_links = ["https://example.com/basf"]
            prompt = enricher._build_enrichment_prompt(story)
            assert "BASF Innovation" in prompt
            assert "Development by BASF team" in prompt
            assert "https://example.com/basf" in prompt
        finally:
            os.unlink(db_path)

    def test_get_enrichment_stats():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            enricher = CompanyMentionEnricher(db, None, None)  # type: ignore
            stats = enricher.get_enrichment_stats()
            assert "pending" in stats
            assert "total_enriched" in stats
        finally:
            os.unlink(db_path)

    suite.add_test(
        "Validate mention - valid sentence", test_validate_mention_valid_sentence
    )
    suite.add_test(
        "Validate mention - NO_COMPANY_MENTION", test_validate_mention_no_company
    )
    suite.add_test("Validate mention - empty string", test_validate_mention_empty)
    suite.add_test(
        "Validate mention - with newlines", test_validate_mention_with_newlines
    )
    suite.add_test(
        "Validate mention - with hashtag", test_validate_mention_with_hashtag
    )
    suite.add_test("Validate mention - with list", test_validate_mention_with_list)
    suite.add_test(
        "Validate mention - missing punctuation",
        test_validate_mention_missing_punctuation,
    )
    suite.add_test("Build enrichment prompt", test_build_enrichment_prompt)
    suite.add_test("Get enrichment stats", test_get_enrichment_stats)

    return suite
