"""
Content freshness and trending topic detection module.

This module provides:
- Trending topic detection using Google Trends
- Story freshness scoring based on publication date
- Topic relevance boosting for quality scores
- Breaking news detection

Implements TASK 3.2 from IMPROVEMENT_TASKS.md.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote_plus


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TrendingTopic:
    """Represents a trending topic with relevance score."""

    topic: str
    score: float  # 0.0 to 1.0, higher = more trending
    source: str  # Where the trend was detected
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    related_terms: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"TrendingTopic({self.topic!r}, score={self.score:.2f})"


@dataclass
class FreshnessScore:
    """Represents the freshness analysis of content."""

    age_hours: float
    freshness_score: float  # 0.0 to 1.0, higher = fresher
    is_breaking_news: bool
    trending_boost: float  # Additional boost from trending topics
    trending_topics_matched: list[str]

    @property
    def total_score(self) -> float:
        """Combined freshness and trending score."""
        return min(1.0, self.freshness_score + self.trending_boost)


# =============================================================================
# Trend Detector
# =============================================================================


class TrendDetector:
    """
    Detect trending topics and score content freshness.

    This implementation uses:
    1. Time-based decay for freshness scoring
    2. Keyword matching for trending topic detection
    3. Domain-specific trending topics for hydrogen/energy industry
    """

    # Industry-specific keywords that indicate breaking news
    BREAKING_NEWS_INDICATORS = [
        "breaking",
        "just announced",
        "just released",
        "breaking news",
        "developing story",
        "exclusive",
        "first look",
        "world first",
        "major breakthrough",
        "unprecedented",
        "historic",
        "record-breaking",
    ]

    # Domain-specific trending topics (hydrogen/clean energy focus)
    INDUSTRY_HOT_TOPICS = [
        "green hydrogen",
        "blue hydrogen",
        "pink hydrogen",
        "electrolysis",
        "electrolyzer",
        "fuel cell",
        "hydrogen storage",
        "hydrogen pipeline",
        "ammonia fuel",
        "hydrogen hub",
        "clean hydrogen",
        "hydrogen production",
        "renewable hydrogen",
        "hydrogen economy",
        "hydrogen infrastructure",
        "hydrogen refueling",
        "hydrogen truck",
        "hydrogen aircraft",
        "hydrogen ship",
        "CCUS",
        "carbon capture",
        "direct air capture",
        "net zero",
        "decarbonization",
        "energy transition",
        "IRA",  # Inflation Reduction Act
        "DOE",  # Department of Energy
        "ARPA-E",
    ]

    # Freshness decay parameters (exponential decay)
    FRESHNESS_HALF_LIFE_HOURS = 24.0  # Score halves every 24 hours
    BREAKING_NEWS_THRESHOLD_HOURS = 6.0  # News < 6 hours is "breaking"
    MAX_TRENDING_BOOST = 0.3  # Maximum boost from trending topics

    def __init__(self) -> None:
        """Initialize the trend detector."""
        self._cached_trends: list[TrendingTopic] = []
        self._cache_time: datetime | None = None
        self._cache_duration = timedelta(hours=1)

    def calculate_freshness_score(
        self,
        publication_date: datetime | None,
        reference_time: datetime | None = None,
    ) -> float:
        """
        Calculate freshness score based on publication date.

        Uses exponential decay with configurable half-life.

        Args:
            publication_date: When the content was published
            reference_time: Reference time for comparison (defaults to now)

        Returns:
            Freshness score between 0.0 and 1.0
        """
        if publication_date is None:
            return 0.5  # Unknown age gets neutral score

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        # Ensure both datetimes are timezone-aware
        if publication_date.tzinfo is None:
            publication_date = publication_date.replace(tzinfo=timezone.utc)
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)

        age = reference_time - publication_date
        age_hours = age.total_seconds() / 3600.0

        if age_hours < 0:
            # Future date (shouldn't happen, but handle gracefully)
            return 1.0

        # Exponential decay: score = 0.5 ^ (age / half_life)
        # This gives 1.0 at age=0, 0.5 at age=half_life, 0.25 at age=2*half_life
        decay_factor = 0.5 ** (age_hours / self.FRESHNESS_HALF_LIFE_HOURS)

        return max(0.0, min(1.0, decay_factor))

    def is_breaking_news(
        self,
        content: str,
        publication_date: datetime | None = None,
    ) -> bool:
        """
        Determine if content qualifies as breaking news.

        Args:
            content: The content text to analyze
            publication_date: When the content was published

        Returns:
            True if the content appears to be breaking news
        """
        content_lower = content.lower()

        # Check for breaking news indicators in text
        for indicator in self.BREAKING_NEWS_INDICATORS:
            if indicator in content_lower:
                return True

        # Check if publication date is very recent
        if publication_date:
            if publication_date.tzinfo is None:
                publication_date = publication_date.replace(tzinfo=timezone.utc)

            age = datetime.now(timezone.utc) - publication_date
            if age.total_seconds() / 3600.0 < self.BREAKING_NEWS_THRESHOLD_HOURS:
                return True

        return False

    def detect_trending_topics(self, content: str) -> list[TrendingTopic]:
        """
        Detect trending topics mentioned in content.

        Args:
            content: The content text to analyze

        Returns:
            List of trending topics found in the content
        """
        content_lower = content.lower()
        found_topics: list[TrendingTopic] = []

        for topic in self.INDUSTRY_HOT_TOPICS:
            if topic.lower() in content_lower:
                # Score based on how specific/hot the topic is
                score = self._calculate_topic_score(topic, content_lower)
                found_topics.append(
                    TrendingTopic(
                        topic=topic,
                        score=score,
                        source="industry_keywords",
                        related_terms=self._find_related_terms(topic, content),
                    )
                )

        # Sort by score descending
        found_topics.sort(key=lambda t: t.score, reverse=True)
        return found_topics

    def calculate_trending_boost(self, content: str) -> tuple[float, list[str]]:
        """
        Calculate quality score boost from trending topics.

        Args:
            content: The content text to analyze

        Returns:
            Tuple of (boost_value, list of matched topics)
        """
        topics = self.detect_trending_topics(content)

        if not topics:
            return 0.0, []

        # Sum scores with diminishing returns for multiple topics
        total_score = 0.0
        for i, topic in enumerate(topics[:5]):  # Max 5 topics contribute
            # Each subsequent topic contributes less
            weight = 1.0 / (i + 1)
            total_score += topic.score * weight

        # Normalize to max boost
        boost = min(self.MAX_TRENDING_BOOST, total_score * 0.1)
        matched = [t.topic for t in topics]

        return boost, matched

    def analyze_content_freshness(
        self,
        content: str,
        publication_date: datetime | None = None,
        title: str = "",
    ) -> FreshnessScore:
        """
        Perform complete freshness analysis on content.

        Args:
            content: The content text to analyze
            publication_date: When the content was published
            title: Optional title for additional analysis

        Returns:
            FreshnessScore with all metrics
        """
        full_text = f"{title} {content}"

        # Calculate age
        if publication_date:
            if publication_date.tzinfo is None:
                publication_date = publication_date.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - publication_date
            age_hours = age.total_seconds() / 3600.0
        else:
            age_hours = 48.0  # Assume ~2 days old if unknown

        # Calculate base freshness
        freshness = self.calculate_freshness_score(publication_date)

        # Check for breaking news
        is_breaking = self.is_breaking_news(full_text, publication_date)

        # Calculate trending boost
        trending_boost, matched_topics = self.calculate_trending_boost(full_text)

        return FreshnessScore(
            age_hours=age_hours,
            freshness_score=freshness,
            is_breaking_news=is_breaking,
            trending_boost=trending_boost,
            trending_topics_matched=matched_topics,
        )

    def get_google_trends_url(self, topic: str) -> str:
        """
        Generate a Google Trends URL for a topic.

        Args:
            topic: The topic to search

        Returns:
            Google Trends URL for the topic
        """
        encoded_topic = quote_plus(topic)
        return f"https://trends.google.com/trends/explore?q={encoded_topic}"

    def suggest_trending_angles(self, content: str) -> list[str]:
        """
        Suggest ways to tie content to trending topics.

        Args:
            content: The content text to analyze

        Returns:
            List of suggestions for trending angles
        """
        suggestions: list[str] = []
        matched_topics = self.detect_trending_topics(content)

        if not matched_topics:
            # Suggest adding trending topics
            suggestions.append(
                "Consider connecting this story to trending topics like "
                "'green hydrogen', 'hydrogen hubs', or 'IRA funding'"
            )
        else:
            # Suggest emphasizing matched topics
            top_topics = [t.topic for t in matched_topics[:3]]
            suggestions.append(
                f"This story mentions trending topics: {', '.join(top_topics)}. "
                "Consider emphasizing these in the headline."
            )

        # Check for timeliness
        if any(
            "2024" in content or "2025" in content or "2026" in content for _ in [1]
        ):
            suggestions.append(
                "Content mentions recent dates - emphasize timeliness in the post."
            )

        return suggestions

    def _calculate_topic_score(self, topic: str, content: str) -> float:
        """Calculate relevance score for a topic based on context."""
        score = 0.5  # Base score for any match

        # Boost for multiple mentions
        count = content.count(topic.lower())
        if count > 1:
            score += min(0.2, count * 0.05)

        # Boost for high-priority topics
        high_priority = ["green hydrogen", "hydrogen hub", "DOE", "IRA", "fuel cell"]
        if topic.lower() in [t.lower() for t in high_priority]:
            score += 0.2

        # Boost for appearing early in content (likely in title/lead)
        first_occurrence = content.find(topic.lower())
        if first_occurrence < 200:
            score += 0.1

        return min(1.0, score)

    def _find_related_terms(self, topic: str, content: str) -> list[str]:
        """Find terms related to a topic in the content."""
        related: list[str] = []

        # Topic-specific related terms
        relations = {
            "green hydrogen": ["renewable", "electrolysis", "solar", "wind"],
            "blue hydrogen": ["SMR", "CCS", "natural gas", "carbon capture"],
            "fuel cell": ["PEM", "SOFC", "automotive", "stationary power"],
            "electrolyzer": ["PEM", "alkaline", "SOEC", "capacity", "MW"],
            "hydrogen hub": ["DOE", "funding", "regional", "infrastructure"],
        }

        if topic.lower() in relations:
            for term in relations[topic.lower()]:
                if term.lower() in content.lower():
                    related.append(term)

        return related

    def get_freshness_summary(self, score: FreshnessScore) -> str:
        """
        Generate a human-readable freshness summary.

        Args:
            score: The freshness score to summarize

        Returns:
            Formatted summary string
        """
        lines = ["Content Freshness Analysis:"]

        # Age category
        if score.age_hours < 6:
            lines.append(f"  ⚡ Very Fresh ({score.age_hours:.1f} hours old)")
        elif score.age_hours < 24:
            lines.append(f"  ✓ Fresh ({score.age_hours:.1f} hours old)")
        elif score.age_hours < 72:
            lines.append(f"  → Recent ({score.age_hours / 24:.1f} days old)")
        else:
            lines.append(f"  ⚠ Aging ({score.age_hours / 24:.1f} days old)")

        lines.append(f"  Freshness Score: {score.freshness_score:.2f}")

        if score.is_breaking_news:
            lines.append("  🔴 BREAKING NEWS detected")

        if score.trending_topics_matched:
            lines.append(
                f"  🔥 Trending Topics: {', '.join(score.trending_topics_matched)}"
            )
            lines.append(f"  Trending Boost: +{score.trending_boost:.2f}")

        lines.append(f"  Total Score: {score.total_score:.2f}")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================


def extract_publication_date(content: str) -> datetime | None:
    """
    Attempt to extract a publication date from content.

    Args:
        content: The content text to parse

    Returns:
        Extracted datetime or None if not found
    """
    # Common date patterns
    patterns = [
        # ISO format: 2024-01-15
        r"(\d{4}-\d{2}-\d{2})",
        # US format: January 15, 2024 or Jan 15, 2024
        r"((?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December|"
        r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s+\d{1,2},?\s+\d{4})",
        # European format: 15 January 2024
        r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4})",
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            try:
                # Try parsing various formats
                for fmt in [
                    "%Y-%m-%d",
                    "%B %d, %Y",
                    "%B %d %Y",
                    "%b %d, %Y",
                    "%b %d %Y",
                    "%d %B %Y",
                ]:
                    try:
                        return datetime.strptime(date_str, fmt).replace(
                            tzinfo=timezone.utc
                        )
                    except ValueError:
                        continue
            except Exception:
                continue

    return None


def boost_quality_score(
    base_score: float,
    content: str,
    publication_date: datetime | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Apply freshness and trending boosts to a quality score.

    Args:
        base_score: The original quality score (0-100)
        content: The content text
        publication_date: When the content was published

    Returns:
        Tuple of (boosted_score, boost_details)
    """
    detector = TrendDetector()
    analysis = detector.analyze_content_freshness(content, publication_date)

    # Calculate boost (max 15 points)
    freshness_boost = analysis.freshness_score * 10  # Up to 10 points
    trending_boost_points = analysis.trending_boost * 50  # Up to 15 points

    # Breaking news gets extra boost
    breaking_boost = 5.0 if analysis.is_breaking_news else 0.0

    total_boost = freshness_boost + trending_boost_points + breaking_boost
    boosted_score = min(100.0, base_score + total_boost)

    details = {
        "original_score": base_score,
        "freshness_boost": freshness_boost,
        "trending_boost": trending_boost_points,
        "breaking_boost": breaking_boost,
        "total_boost": total_boost,
        "final_score": boosted_score,
        "age_hours": analysis.age_hours,
        "trending_topics": analysis.trending_topics_matched,
        "is_breaking": analysis.is_breaking_news,
    }

    return boosted_score, details


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests() -> bool:
    """Create unit tests for trend_detector module."""
    from test_framework import TestSuite

    suite = TestSuite("Trend Detector Tests", "trend_detector.py")
    suite.start_suite()

    def test_freshness_score_new():
        detector = TrendDetector()
        now = datetime.now(timezone.utc)
        score = detector.calculate_freshness_score(now)
        assert score > 0.99  # Just published should be ~1.0

    def test_freshness_score_old():
        detector = TrendDetector()
        old_date = datetime.now(timezone.utc) - timedelta(days=7)
        score = detector.calculate_freshness_score(old_date)
        assert score < 0.1  # 7 days old should be low

    def test_freshness_score_none():
        detector = TrendDetector()
        score = detector.calculate_freshness_score(None)
        assert score == 0.5  # Unknown gets neutral

    def test_freshness_half_life():
        detector = TrendDetector()
        reference = datetime.now(timezone.utc)
        half_life_ago = reference - timedelta(hours=detector.FRESHNESS_HALF_LIFE_HOURS)
        score = detector.calculate_freshness_score(half_life_ago, reference)
        assert 0.45 < score < 0.55  # Should be ~0.5

    def test_is_breaking_news_indicator():
        detector = TrendDetector()
        content = "BREAKING: Major hydrogen breakthrough announced today"
        assert detector.is_breaking_news(content) is True

    def test_is_breaking_news_recent():
        detector = TrendDetector()
        content = "New electrolyzer facility opens"
        recent = datetime.now(timezone.utc) - timedelta(hours=2)
        assert detector.is_breaking_news(content, recent) is True

    def test_is_breaking_news_old():
        detector = TrendDetector()
        content = "New electrolyzer facility opens"
        old = datetime.now(timezone.utc) - timedelta(days=2)
        assert detector.is_breaking_news(content, old) is False

    def test_detect_trending_topics():
        detector = TrendDetector()
        content = "The new green hydrogen project uses advanced electrolyzer technology"
        topics = detector.detect_trending_topics(content)
        assert len(topics) >= 2
        topic_names = [t.topic for t in topics]
        assert "green hydrogen" in topic_names
        assert "electrolyzer" in topic_names

    def test_detect_trending_topics_empty():
        detector = TrendDetector()
        content = "This is a generic article about nothing specific"
        topics = detector.detect_trending_topics(content)
        assert len(topics) == 0

    def test_trending_boost():
        detector = TrendDetector()
        content = "Green hydrogen and fuel cell technology advances"
        boost, matched = detector.calculate_trending_boost(content)
        assert boost > 0
        assert len(matched) >= 2

    def test_analyze_content_freshness():
        detector = TrendDetector()
        content = "Breaking news about green hydrogen hub funded by DOE"
        now = datetime.now(timezone.utc)
        analysis = detector.analyze_content_freshness(content, now)
        assert analysis.freshness_score > 0.9
        assert analysis.is_breaking_news is True
        assert len(analysis.trending_topics_matched) >= 2

    def test_freshness_score_dataclass():
        score = FreshnessScore(
            age_hours=12.0,
            freshness_score=0.7,
            is_breaking_news=False,
            trending_boost=0.1,
            trending_topics_matched=["hydrogen"],
        )
        assert 0.79 < score.total_score < 0.81  # Allow for floating-point

    def test_extract_publication_date_iso():
        content = "Published on 2024-06-15 by the research team"
        date = extract_publication_date(content)
        assert date is not None
        assert date.year == 2024
        assert date.month == 6
        assert date.day == 15

    def test_extract_publication_date_us():
        content = "January 15, 2024 - The company announced..."
        date = extract_publication_date(content)
        assert date is not None
        assert date.month == 1
        assert date.day == 15

    def test_boost_quality_score():
        content = "Green hydrogen breakthrough in fuel cell technology"
        boosted, details = boost_quality_score(70.0, content)
        assert boosted >= 70.0  # Should not decrease
        assert "trending_topics" in details
        assert len(details["trending_topics"]) >= 2

    def test_freshness_summary():
        detector = TrendDetector()
        score = FreshnessScore(
            age_hours=2.0,
            freshness_score=0.95,
            is_breaking_news=True,
            trending_boost=0.2,
            trending_topics_matched=["green hydrogen", "DOE"],
        )
        summary = detector.get_freshness_summary(score)
        assert "Very Fresh" in summary
        assert "BREAKING NEWS" in summary
        assert "green hydrogen" in summary

    suite.run_test(
        test_name="Freshness score - new",
        test_func=test_freshness_score_new,
        test_summary="Tests Freshness score with new scenario",
        method_description="Calls TrendDetector and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Freshness score - old",
        test_func=test_freshness_score_old,
        test_summary="Tests Freshness score with old scenario",
        method_description="Calls TrendDetector and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Freshness score - none",
        test_func=test_freshness_score_none,
        test_summary="Tests Freshness score with none scenario",
        method_description="Calls TrendDetector and verifies the result",
        expected_outcome="Function handles empty or missing input gracefully",
    )
    suite.run_test(
        test_name="Freshness half-life",
        test_func=test_freshness_half_life,
        test_summary="Tests Freshness half-life functionality",
        method_description="Calls TrendDetector and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Breaking news - indicator",
        test_func=test_is_breaking_news_indicator,
        test_summary="Tests Breaking news with indicator scenario",
        method_description="Calls TrendDetector and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Breaking news - recent",
        test_func=test_is_breaking_news_recent,
        test_summary="Tests Breaking news with recent scenario",
        method_description="Calls TrendDetector and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Breaking news - old",
        test_func=test_is_breaking_news_old,
        test_summary="Tests Breaking news with old scenario",
        method_description="Calls TrendDetector and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Detect trending topics",
        test_func=test_detect_trending_topics,
        test_summary="Tests Detect trending topics functionality",
        method_description="Calls TrendDetector and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Detect trending - empty",
        test_func=test_detect_trending_topics_empty,
        test_summary="Tests Detect trending with empty scenario",
        method_description="Calls TrendDetector and verifies the result",
        expected_outcome="Function handles empty or missing input gracefully",
    )
    suite.run_test(
        test_name="Trending boost",
        test_func=test_trending_boost,
        test_summary="Tests Trending boost functionality",
        method_description="Calls TrendDetector and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Analyze content freshness",
        test_func=test_analyze_content_freshness,
        test_summary="Tests Analyze content freshness functionality",
        method_description="Calls TrendDetector and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="FreshnessScore dataclass",
        test_func=test_freshness_score_dataclass,
        test_summary="Tests FreshnessScore dataclass functionality",
        method_description="Calls FreshnessScore and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Extract date - ISO",
        test_func=test_extract_publication_date_iso,
        test_summary="Tests Extract date with iso scenario",
        method_description="Calls extract publication date and verifies the result",
        expected_outcome="Function correctly parses and extracts the data",
    )
    suite.run_test(
        test_name="Extract date - US format",
        test_func=test_extract_publication_date_us,
        test_summary="Tests Extract date with us format scenario",
        method_description="Calls extract publication date and verifies the result",
        expected_outcome="Function produces correctly formatted output",
    )
    suite.run_test(
        test_name="Boost quality score",
        test_func=test_boost_quality_score,
        test_summary="Tests Boost quality score functionality",
        method_description="Calls boost quality score and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Freshness summary",
        test_func=test_freshness_summary,
        test_summary="Tests Freshness summary functionality",
        method_description="Calls TrendDetector and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )

    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
