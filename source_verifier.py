"""Multi-source verification for story credibility.

This module implements source credibility scoring and multi-source verification
to increase content credibility and flag potential misinformation.
"""

import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse

from database import Story

logger = logging.getLogger(__name__)


# Credibility tiers for domains
TIER_1_DOMAINS = frozenset(
    {
        # Major scientific publishers
        "nature.com",
        "science.org",
        "sciencedirect.com",
        "ieee.org",
        "springer.com",
        "wiley.com",
        "acs.org",
        "rsc.org",
        "elsevier.com",
        "cell.com",
        # Major news organizations
        "nytimes.com",
        "washingtonpost.com",
        "bbc.com",
        "bbc.co.uk",
        "reuters.com",
        "apnews.com",
        "bloomberg.com",
        "wsj.com",
        "economist.com",
        "ft.com",
        "theguardian.com",
        # Top universities
        "mit.edu",
        "stanford.edu",
        "harvard.edu",
        "berkeley.edu",
        "caltech.edu",
        "princeton.edu",
        "yale.edu",
        "columbia.edu",
        "cam.ac.uk",
        "ox.ac.uk",
        "ethz.ch",
        "mpg.de",
        # Government / Research institutions
        "nih.gov",
        "nasa.gov",
        "doe.gov",
        "energy.gov",
        "noaa.gov",
        "nist.gov",
        "arxiv.org",
        "pubmed.gov",
    }
)

TIER_2_DOMAINS = frozenset(
    {
        # Quality tech publications
        "techcrunch.com",
        "wired.com",
        "arstechnica.com",
        "theverge.com",
        "engadget.com",
        "zdnet.com",
        "cnet.com",
        "venturebeat.com",
        # Industry publications
        "forbes.com",
        "businessinsider.com",
        "fortune.com",
        "cnbc.com",
        # Energy/sustainability specific
        "greentechmedia.com",
        "cleantechnica.com",
        "renewableenergyworld.com",
        "pv-magazine.com",
        "hydrogeninsight.com",
        "rechargenews.com",
        "spglobal.com",
        "woodmac.com",
        # Other reputable universities
        "cornell.edu",
        "uchicago.edu",
        "upenn.edu",
        "duke.edu",
        "northwestern.edu",
        "ucla.edu",
        "umich.edu",
        "gatech.edu",
        "cmu.edu",
        "utexas.edu",
    }
)

TIER_3_DOMAINS = frozenset(
    {
        # Company blogs/newsrooms (primary sources but biased)
        "newsroom.",
        "blog.",
        "press.",
        "media.",
        # Regional news
        "latimes.com",
        "chicagotribune.com",
        "bostonglobe.com",
        "sfchronicle.com",
        # International quality sources
        "lemonde.fr",
        "spiegel.de",
        "zeit.de",
        "nikkei.com",
        "scmp.com",
    }
)

# Domains that reduce credibility (tabloids, known misinformation, etc.)
LOW_CREDIBILITY_DOMAINS = frozenset(
    {
        "dailymail.co.uk",
        "thesun.co.uk",
        "nypost.com",
        "breitbart.com",
        "infowars.com",
        "naturalnews.com",
    }
)


@dataclass
class SourceCredibilityResult:
    """Result of source credibility analysis."""

    url: str
    domain: str
    credibility_score: float  # 0.0 = no credibility, 1.0 = maximum credibility
    tier: int  # 1 = highest, 2 = medium, 3 = lower, 0 = unknown
    is_academic: bool
    is_government: bool
    is_primary_source: bool  # Company newsroom, press release
    is_low_credibility: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class MultiSourceVerificationResult:
    """Result of multi-source verification."""

    is_verified: bool
    source_count: int
    unique_domain_count: int
    average_credibility: float
    min_credibility: float
    highest_tier: int  # Best tier among sources (1 is best)
    has_academic_source: bool
    has_government_source: bool
    has_primary_source: bool
    has_low_credibility: bool
    recommendation: str
    source_results: list[SourceCredibilityResult] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "✓ VERIFIED" if self.is_verified else "⚠ NEEDS MORE SOURCES"
        return (
            f"{status} ({self.source_count} sources, "
            f"avg credibility: {self.average_credibility:.1%}, "
            f"tier {self.highest_tier})"
        )


class SourceVerifier:
    """Verify story credibility through multi-source analysis."""

    # Default thresholds (can be overridden via Config)
    DEFAULT_MIN_SOURCES = 1  # Minimum sources required for verification
    DEFAULT_MIN_CREDIBILITY = 0.3  # Minimum average credibility
    DEFAULT_REQUIRE_TIER1_OR_2 = False  # Require at least one tier 1 or 2 source

    def __init__(
        self,
        min_sources: int | None = None,
        min_credibility: float | None = None,
        require_tier1_or_2: bool | None = None,
    ):
        """
        Initialize the source verifier.

        Args:
            min_sources: Minimum number of sources required
            min_credibility: Minimum average credibility score (0.0-1.0)
            require_tier1_or_2: Whether to require at least one tier 1 or 2 source
        """
        # Import Config here to allow override
        try:
            from config import Config

            self.min_sources = min_sources or getattr(
                Config, "MIN_SOURCES_REQUIRED", self.DEFAULT_MIN_SOURCES
            )
            self.min_credibility = min_credibility or getattr(
                Config, "MIN_SOURCE_CREDIBILITY", self.DEFAULT_MIN_CREDIBILITY
            )
            self.require_tier1_or_2 = (
                require_tier1_or_2
                if require_tier1_or_2 is not None
                else getattr(
                    Config, "REQUIRE_TIER1_OR_2_SOURCE", self.DEFAULT_REQUIRE_TIER1_OR_2
                )
            )
        except ImportError:
            self.min_sources = min_sources or self.DEFAULT_MIN_SOURCES
            self.min_credibility = min_credibility or self.DEFAULT_MIN_CREDIBILITY
            self.require_tier1_or_2 = (
                require_tier1_or_2
                if require_tier1_or_2 is not None
                else self.DEFAULT_REQUIRE_TIER1_OR_2
            )

    def check_source_credibility(self, url: str) -> SourceCredibilityResult:
        """
        Check the credibility of a single source URL.

        Args:
            url: The source URL to analyze

        Returns:
            SourceCredibilityResult with credibility details
        """
        if not url:
            return SourceCredibilityResult(
                url=url,
                domain="",
                credibility_score=0.0,
                tier=0,
                is_academic=False,
                is_government=False,
                is_primary_source=False,
                is_low_credibility=True,
                notes=["Empty URL"],
            )

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]

            notes: list[str] = []
            tier = 0
            score = 0.3  # Base score for any URL

            # Check if academic (.edu domains)
            is_academic = ".edu" in domain or domain in (
                "arxiv.org",
                "researchgate.net",
                "academia.edu",
                "cam.ac.uk",
                "ox.ac.uk",
            )

            # Check if government
            is_government = ".gov" in domain

            # Check if primary source (company newsroom, press release)
            # These patterns must be subdomains at the START of the domain
            is_primary_source = any(
                domain.startswith(prefix)
                for prefix in ("newsroom.", "blog.", "press.", "media.")
            ) or any(
                path_part in parsed.path.lower()
                for path_part in ("/press/", "/newsroom/", "/news/", "/media/")
            )

            # Check tier 1
            is_tier1 = self._is_domain_in_set(domain, TIER_1_DOMAINS)
            if is_tier1:
                tier = 1
                score = 1.0
                notes.append("Tier 1 source (highest credibility)")

            # Check tier 2
            is_tier2 = self._is_domain_in_set(domain, TIER_2_DOMAINS)
            if not is_tier1 and is_tier2:
                tier = 2
                score = 0.8
                notes.append("Tier 2 source (high credibility)")

            # Check tier 3
            is_tier3 = self._is_domain_in_set(domain, TIER_3_DOMAINS)
            if not is_tier1 and not is_tier2 and is_tier3:
                tier = 3
                score = 0.6
                notes.append("Tier 3 source (moderate credibility)")

            # Check low credibility
            is_low_credibility = self._is_domain_in_set(domain, LOW_CREDIBILITY_DOMAINS)
            if is_low_credibility:
                tier = 0
                score = 0.1
                notes.append("Low credibility source")

            # Apply bonuses
            if is_academic and tier == 0:
                tier = 2
                score = max(score, 0.75)
                notes.append("Academic source")

            if is_government and tier == 0:
                tier = 2
                score = max(score, 0.8)
                notes.append("Government source")

            if is_primary_source:
                notes.append("Primary source (company newsroom/press release)")
                if tier == 0:
                    tier = 3
                    score = max(score, 0.5)

            return SourceCredibilityResult(
                url=url,
                domain=domain,
                credibility_score=score,
                tier=tier or 4,  # Unknown sources get tier 4
                is_academic=is_academic,
                is_government=is_government,
                is_primary_source=is_primary_source,
                is_low_credibility=is_low_credibility,
                notes=notes,
            )

        except Exception as e:
            logger.warning(f"Error analyzing source URL {url}: {e}")
            return SourceCredibilityResult(
                url=url,
                domain="",
                credibility_score=0.2,
                tier=4,
                is_academic=False,
                is_government=False,
                is_primary_source=False,
                is_low_credibility=False,
                notes=[f"Parse error: {e}"],
            )

    def _is_domain_in_set(self, domain: str, domain_set: frozenset[str]) -> bool:
        """Check if domain or any parent domain is in the set."""
        # Direct match
        if domain in domain_set:
            return True

        # Check parent domains (e.g., news.mit.edu -> mit.edu)
        parts = domain.split(".")
        for i in range(1, len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in domain_set:
                return True

        # Check for subdomain prefix patterns (e.g., "newsroom." matches "newsroom.company.com")
        # These patterns must appear at the START of the domain
        for pattern in domain_set:
            if pattern.endswith(".") and domain.startswith(pattern):
                return True

        return False

    def verify_story_sources(self, story: Story) -> MultiSourceVerificationResult:
        """
        Verify a story's sources for credibility.

        Args:
            story: The Story object to verify

        Returns:
            MultiSourceVerificationResult with verification details
        """
        source_urls = story.source_links or []

        # Filter out empty/invalid URLs
        valid_urls = [url for url in source_urls if url and url.strip()]

        if not valid_urls:
            return MultiSourceVerificationResult(
                is_verified=False,
                source_count=0,
                unique_domain_count=0,
                average_credibility=0.0,
                min_credibility=0.0,
                highest_tier=0,
                has_academic_source=False,
                has_government_source=False,
                has_primary_source=False,
                has_low_credibility=False,
                recommendation="No sources provided - cannot verify story credibility",
                source_results=[],
                issues=["No source URLs provided"],
            )

        # Analyze each source
        source_results = [self.check_source_credibility(url) for url in valid_urls]

        # Calculate metrics
        unique_domains = set(r.domain for r in source_results if r.domain)
        credibility_scores = [r.credibility_score for r in source_results]

        avg_credibility = (
            sum(credibility_scores) / len(credibility_scores)
            if credibility_scores
            else 0.0
        )
        min_credibility = min(credibility_scores) if credibility_scores else 0.0
        highest_tier = (
            min(r.tier for r in source_results if r.tier > 0) if source_results else 0
        )

        has_academic = any(r.is_academic for r in source_results)
        has_government = any(r.is_government for r in source_results)
        has_primary = any(r.is_primary_source for r in source_results)
        has_low_cred = any(r.is_low_credibility for r in source_results)

        # Build issues list
        issues: list[str] = []

        if len(valid_urls) < self.min_sources:
            issues.append(
                f"Insufficient sources: {len(valid_urls)} provided, {self.min_sources} required"
            )

        if avg_credibility < self.min_credibility:
            issues.append(
                f"Low average credibility: {avg_credibility:.1%} (minimum: {self.min_credibility:.1%})"
            )

        if self.require_tier1_or_2 and highest_tier > 2:
            issues.append("No tier 1 or tier 2 source found")

        if has_low_cred:
            issues.append("Contains low-credibility source(s)")

        # Determine if verified
        is_verified = (
            len(valid_urls) >= self.min_sources
            and avg_credibility >= self.min_credibility
            and (not self.require_tier1_or_2 or highest_tier <= 2)
            and not has_low_cred
        )

        # Build recommendation
        if is_verified:
            recommendation = f"Story verified with {len(valid_urls)} source(s), "
            recommendation += f"average credibility {avg_credibility:.0%}"
            if highest_tier == 1:
                recommendation += " (includes tier 1 source)"
            elif highest_tier == 2:
                recommendation += " (includes tier 2 source)"
        else:
            recommendation = "Story needs improvement: " + "; ".join(issues)

        return MultiSourceVerificationResult(
            is_verified=is_verified,
            source_count=len(valid_urls),
            unique_domain_count=len(unique_domains),
            average_credibility=avg_credibility,
            min_credibility=min_credibility,
            highest_tier=highest_tier,
            has_academic_source=has_academic,
            has_government_source=has_government,
            has_primary_source=has_primary,
            has_low_credibility=has_low_cred,
            recommendation=recommendation,
            source_results=source_results,
            issues=issues,
        )

    def get_source_summary(self, story: Story) -> str:
        """
        Get a human-readable summary of a story's source credibility.

        Args:
            story: The Story object to summarize

        Returns:
            A formatted string summarizing source credibility
        """
        result = self.verify_story_sources(story)

        lines = [f"Source Verification: {result}"]
        lines.append(
            f"  Sources: {result.source_count} ({result.unique_domain_count} unique domains)"
        )
        lines.append(f"  Average credibility: {result.average_credibility:.1%}")
        lines.append(f"  Highest tier: {result.highest_tier}")

        if result.has_academic_source:
            lines.append("  ✓ Has academic source")
        if result.has_government_source:
            lines.append("  ✓ Has government source")
        if result.has_primary_source:
            lines.append("  ✓ Has primary source")
        if result.has_low_credibility:
            lines.append("  ⚠ Contains low-credibility source")

        if result.issues:
            lines.append("  Issues:")
            for issue in result.issues:
                lines.append(f"    - {issue}")

        return "\n".join(lines)


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for source_verifier module."""
    from test_framework import TestSuite

    suite = TestSuite("Source Verifier Tests")

    def test_tier1_nature():
        verifier = SourceVerifier()
        result = verifier.check_source_credibility(
            "https://www.nature.com/articles/123"
        )
        assert result.tier == 1, f"Expected tier 1, got {result.tier}"
        assert result.credibility_score == 1.0
        assert not result.is_low_credibility

    def test_tier1_mit():
        verifier = SourceVerifier()
        result = verifier.check_source_credibility("https://news.mit.edu/2024/story")
        assert result.tier == 1, f"Expected tier 1, got {result.tier}"
        assert result.is_academic

    def test_tier2_techcrunch():
        verifier = SourceVerifier()
        result = verifier.check_source_credibility("https://techcrunch.com/article")
        assert result.tier == 2, f"Expected tier 2, got {result.tier}"
        assert result.credibility_score == 0.8

    def test_government_source():
        verifier = SourceVerifier()
        result = verifier.check_source_credibility(
            "https://www.doe.gov/articles/energy"
        )
        assert result.tier == 1, f"Expected tier 1, got {result.tier}"
        assert result.is_government

    def test_low_credibility():
        verifier = SourceVerifier()
        result = verifier.check_source_credibility("https://dailymail.co.uk/article")
        assert result.is_low_credibility
        assert result.credibility_score == 0.1

    def test_unknown_domain():
        verifier = SourceVerifier()
        result = verifier.check_source_credibility("https://random-blog.xyz/post")
        assert result.tier == 4, f"Expected tier 4 (unknown), got {result.tier}"
        assert result.credibility_score == 0.3  # Base score

    def test_empty_url():
        verifier = SourceVerifier()
        result = verifier.check_source_credibility("")
        assert result.credibility_score == 0.0
        assert result.is_low_credibility

    def test_primary_source():
        verifier = SourceVerifier()
        result = verifier.check_source_credibility(
            "https://newsroom.toyota.com/release"
        )
        assert result.is_primary_source
        assert "Primary source" in str(result.notes)

    def test_story_multiple_sources():
        story = Story(
            title="Test Story",
            summary="Test summary",
            source_links=[
                "https://nature.com/article1",
                "https://techcrunch.com/article2",
            ],
        )
        verifier = SourceVerifier(min_sources=1)
        result = verifier.verify_story_sources(story)
        assert result.is_verified
        assert result.source_count == 2
        assert result.highest_tier == 1

    def test_story_no_sources():
        story = Story(title="Test Story", summary="Test summary", source_links=[])
        verifier = SourceVerifier(min_sources=1)
        result = verifier.verify_story_sources(story)
        assert not result.is_verified
        assert "No source URLs provided" in result.issues

    def test_story_low_credibility():
        story = Story(
            title="Test Story",
            summary="Test summary",
            source_links=["https://dailymail.co.uk/article"],
        )
        verifier = SourceVerifier()
        result = verifier.verify_story_sources(story)
        assert not result.is_verified
        assert result.has_low_credibility

    def test_average_credibility():
        story = Story(
            title="Test",
            summary="Test",
            source_links=[
                "https://nature.com/a",  # 1.0
                "https://random-blog.xyz/b",  # 0.3
            ],
        )
        verifier = SourceVerifier()
        result = verifier.verify_story_sources(story)
        expected_avg = (1.0 + 0.3) / 2
        assert abs(result.average_credibility - expected_avg) < 0.01

    def test_subdomain():
        verifier = SourceVerifier()
        result = verifier.check_source_credibility("https://blog.stanford.edu/post")
        assert result.tier == 1  # stanford.edu is tier 1
        assert result.is_academic

    def test_source_summary():
        story = Story(
            title="Test",
            summary="Test",
            source_links=["https://nature.com/article"],
        )
        verifier = SourceVerifier()
        summary = verifier.get_source_summary(story)
        assert "Source Verification" in summary
        assert "tier" in summary.lower() or "credibility" in summary.lower()

    def test_require_tier1_or_2():
        story = Story(
            title="Test",
            summary="Test",
            source_links=["https://random-blog.xyz/post"],
        )
        verifier = SourceVerifier(require_tier1_or_2=True)
        result = verifier.verify_story_sources(story)
        assert not result.is_verified
        assert any("tier 1 or tier 2" in issue.lower() for issue in result.issues)

    suite.add_test("Tier 1 domain - nature.com", test_tier1_nature)
    suite.add_test("Tier 1 domain - mit.edu", test_tier1_mit)
    suite.add_test("Tier 2 domain - techcrunch", test_tier2_techcrunch)
    suite.add_test("Government source - doe.gov", test_government_source)
    suite.add_test("Low credibility source", test_low_credibility)
    suite.add_test("Unknown domain - base score", test_unknown_domain)
    suite.add_test("Empty URL handling", test_empty_url)
    suite.add_test("Primary source - newsroom", test_primary_source)
    suite.add_test("Story verification - multiple sources", test_story_multiple_sources)
    suite.add_test("Story verification - no sources", test_story_no_sources)
    suite.add_test("Story verification - low credibility", test_story_low_credibility)
    suite.add_test("Average credibility calculation", test_average_credibility)
    suite.add_test("Subdomain handling", test_subdomain)
    suite.add_test("Source summary generation", test_source_summary)
    suite.add_test("Require tier 1 or 2 option", test_require_tier1_or_2)

    return suite
