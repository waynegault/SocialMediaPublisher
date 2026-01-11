"""
A/B Testing module for LinkedIn post format optimization.

This module provides:
- Variant creation for different post formats
- Random assignment of posts to variants
- Performance tracking and statistical analysis
- Winner determination based on engagement metrics

Implements TASK 3.3 from IMPROVEMENT_TASKS.md.
"""

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# =============================================================================
# Enums and Constants
# =============================================================================


class VariantType(Enum):
    """Types of A/B test variants."""

    HEADLINE_STYLE = "headline_style"  # Question vs statement
    CTA_FORMAT = "cta_format"  # Different CTA styles
    EMOJI_USAGE = "emoji_usage"  # With vs without emojis
    POST_LENGTH = "post_length"  # Short vs long
    IMAGE_STYLE = "image_style"  # Different image approaches


class EngagementMetric(Enum):
    """Metrics used to evaluate variant performance."""

    IMPRESSIONS = "impressions"
    LIKES = "likes"
    COMMENTS = "comments"
    SHARES = "shares"
    CLICKS = "clicks"
    ENGAGEMENT_RATE = "engagement_rate"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Variant:
    """Represents a single variant in an A/B test."""

    id: str
    name: str
    description: str
    variant_type: VariantType
    config: dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class VariantResult:
    """Tracks performance of a variant."""

    variant_id: str
    impressions: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    clicks: int = 0
    assignments: int = 0  # How many times this variant was assigned
    
    @property
    def total_engagement(self) -> int:
        """Total engagement actions."""
        return self.likes + self.comments + self.shares + self.clicks
    
    @property
    def engagement_rate(self) -> float:
        """Calculate engagement rate as percentage."""
        if self.impressions == 0:
            return 0.0
        return (self.total_engagement / self.impressions) * 100


@dataclass
class ABTest:
    """Represents an A/B test configuration."""

    id: str
    name: str
    description: str
    variant_type: VariantType
    variants: list[Variant]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    winner_id: str | None = None
    
    def __post_init__(self) -> None:
        if len(self.variants) < 2:
            raise ValueError("A/B test requires at least 2 variants")


@dataclass
class TestSummary:
    """Summary of A/B test results."""

    test_id: str
    test_name: str
    variant_results: dict[str, VariantResult]
    total_assignments: int
    best_variant_id: str | None
    confidence_score: float  # 0-100%
    is_statistically_significant: bool
    recommendation: str


# =============================================================================
# A/B Test Manager
# =============================================================================


class ABTestManager:
    """
    Manages A/B tests for post format optimization.

    Features:
    - Create and configure A/B tests
    - Assign content to variants deterministically or randomly
    - Track performance metrics
    - Determine winning variants
    """

    # Minimum sample size for statistical significance
    MIN_SAMPLE_SIZE = 30
    SIGNIFICANCE_THRESHOLD = 0.95  # 95% confidence

    def __init__(self, storage_path: Path | None = None) -> None:
        """
        Initialize the A/B test manager.

        Args:
            storage_path: Optional path for persisting test data
        """
        self._tests: dict[str, ABTest] = {}
        self._results: dict[str, dict[str, VariantResult]] = {}
        self._assignments: dict[str, str] = {}  # content_id -> variant_id
        self._storage_path = storage_path

        if storage_path and storage_path.exists():
            self._load_state()

    def create_test(
        self,
        name: str,
        description: str,
        variant_type: VariantType,
        variants: list[tuple[str, str, dict[str, Any]]],  # (name, desc, config)
    ) -> ABTest:
        """
        Create a new A/B test.

        Args:
            name: Test name
            description: Test description
            variant_type: Type of variant being tested
            variants: List of (name, description, config) tuples

        Returns:
            Created ABTest instance
        """
        test_id = self._generate_id(name)

        variant_objects = []
        for i, (v_name, v_desc, v_config) in enumerate(variants):
            variant_id = f"{test_id}_v{i}"
            variant_objects.append(
                Variant(
                    id=variant_id,
                    name=v_name,
                    description=v_desc,
                    variant_type=variant_type,
                    config=v_config,
                )
            )

        test = ABTest(
            id=test_id,
            name=name,
            description=description,
            variant_type=variant_type,
            variants=variant_objects,
        )

        self._tests[test_id] = test
        self._results[test_id] = {
            v.id: VariantResult(variant_id=v.id) for v in variant_objects
        }

        self._save_state()
        return test

    def get_variant_for_content(
        self,
        test_id: str,
        content_id: str,
        deterministic: bool = True,
    ) -> Variant | None:
        """
        Get the assigned variant for a piece of content.

        Args:
            test_id: The A/B test ID
            content_id: Unique identifier for the content
            deterministic: If True, same content always gets same variant

        Returns:
            Assigned Variant or None if test not found
        """
        test = self._tests.get(test_id)
        if not test or not test.is_active:
            return None

        # Check for existing assignment
        assignment_key = f"{test_id}:{content_id}"
        if assignment_key in self._assignments:
            variant_id = self._assignments[assignment_key]
            return next((v for v in test.variants if v.id == variant_id), None)

        # Assign variant
        if deterministic:
            # Hash-based deterministic assignment
            hash_input = f"{test_id}:{content_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            variant_index = hash_value % len(test.variants)
        else:
            # Random assignment
            variant_index = random.randint(0, len(test.variants) - 1)

        variant = test.variants[variant_index]
        self._assignments[assignment_key] = variant.id
        self._results[test_id][variant.id].assignments += 1

        self._save_state()
        return variant

    def record_metrics(
        self,
        test_id: str,
        variant_id: str,
        impressions: int = 0,
        likes: int = 0,
        comments: int = 0,
        shares: int = 0,
        clicks: int = 0,
    ) -> None:
        """
        Record performance metrics for a variant.

        Args:
            test_id: The A/B test ID
            variant_id: The variant ID
            impressions: Number of impressions
            likes: Number of likes
            comments: Number of comments
            shares: Number of shares
            clicks: Number of clicks
        """
        if test_id not in self._results:
            return

        if variant_id not in self._results[test_id]:
            return

        result = self._results[test_id][variant_id]
        result.impressions += impressions
        result.likes += likes
        result.comments += comments
        result.shares += shares
        result.clicks += clicks

        self._save_state()

    def get_test_summary(self, test_id: str) -> TestSummary | None:
        """
        Get summary of test results.

        Args:
            test_id: The A/B test ID

        Returns:
            TestSummary or None if test not found
        """
        test = self._tests.get(test_id)
        if not test:
            return None

        results = self._results.get(test_id, {})
        total_assignments = sum(r.assignments for r in results.values())

        # Find best performing variant
        best_variant_id = None
        best_engagement_rate = -1.0

        for variant_id, result in results.items():
            if result.engagement_rate > best_engagement_rate:
                best_engagement_rate = result.engagement_rate
                best_variant_id = variant_id

        # Calculate confidence
        confidence, is_significant = self._calculate_significance(results)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            test, results, best_variant_id, is_significant
        )

        return TestSummary(
            test_id=test_id,
            test_name=test.name,
            variant_results=results,
            total_assignments=total_assignments,
            best_variant_id=best_variant_id,
            confidence_score=confidence,
            is_statistically_significant=is_significant,
            recommendation=recommendation,
        )

    def declare_winner(self, test_id: str, variant_id: str) -> bool:
        """
        Declare a winner and deactivate the test.

        Args:
            test_id: The A/B test ID
            variant_id: The winning variant ID

        Returns:
            True if successful, False otherwise
        """
        test = self._tests.get(test_id)
        if not test:
            return False

        if not any(v.id == variant_id for v in test.variants):
            return False

        test.winner_id = variant_id
        test.is_active = False

        self._save_state()
        return True

    def get_active_tests(self) -> list[ABTest]:
        """Get all active A/B tests."""
        return [t for t in self._tests.values() if t.is_active]

    def get_all_tests(self) -> list[ABTest]:
        """Get all A/B tests."""
        return list(self._tests.values())

    def _calculate_significance(
        self,
        results: dict[str, VariantResult],
    ) -> tuple[float, bool]:
        """
        Calculate statistical significance using simplified z-test.

        Returns:
            Tuple of (confidence_score, is_significant)
        """
        if len(results) < 2:
            return 0.0, False

        # Get results as list
        result_list = list(results.values())

        # Check minimum sample size
        total_impressions = sum(r.impressions for r in result_list)
        if total_impressions < self.MIN_SAMPLE_SIZE * len(result_list):
            return 0.0, False

        # Calculate engagement rates
        rates = [r.engagement_rate for r in result_list]
        if max(rates) == 0:
            return 0.0, False

        # Simplified significance calculation
        # Based on relative difference between best and second-best
        sorted_rates = sorted(rates, reverse=True)
        if len(sorted_rates) < 2:
            return 0.0, False

        best_rate = sorted_rates[0]
        second_rate = sorted_rates[1]

        if second_rate == 0:
            relative_diff = 1.0
        else:
            relative_diff = (best_rate - second_rate) / second_rate

        # Map relative difference to confidence (simplified)
        # 20%+ difference with adequate sample = high confidence
        confidence = min(100.0, relative_diff * 200)

        # Need at least 10% difference and adequate sample for significance
        is_significant = (
            relative_diff >= 0.1
            and total_impressions >= self.MIN_SAMPLE_SIZE * len(result_list)
        )

        return confidence, is_significant

    def _generate_recommendation(
        self,
        test: ABTest,
        results: dict[str, VariantResult],
        best_variant_id: str | None,
        is_significant: bool,
    ) -> str:
        """Generate a recommendation based on test results."""
        if not best_variant_id:
            return "Insufficient data to make a recommendation."

        best_result = results.get(best_variant_id)
        best_variant = next(
            (v for v in test.variants if v.id == best_variant_id), None
        )

        if not best_result or not best_variant:
            return "Unable to determine best variant."

        total_assignments = sum(r.assignments for r in results.values())

        if total_assignments < self.MIN_SAMPLE_SIZE:
            return (
                f"Need more data ({total_assignments}/{self.MIN_SAMPLE_SIZE} "
                f"samples). Continue testing."
            )

        if is_significant:
            return (
                f"✓ WINNER: '{best_variant.name}' with "
                f"{best_result.engagement_rate:.2f}% engagement rate. "
                f"Consider adopting this variant."
            )
        else:
            return (
                f"Leading: '{best_variant.name}' with "
                f"{best_result.engagement_rate:.2f}% engagement rate, "
                f"but more data needed for significance."
            )

    @staticmethod
    def _generate_id(name: str) -> str:
        """Generate a unique ID from a name."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        name_slug = "".join(c if c.isalnum() else "_" for c in name.lower())[:20]
        return f"test_{name_slug}_{timestamp}"

    def _save_state(self) -> None:
        """Save state to disk if storage path is configured."""
        if not self._storage_path:
            return

        state = {
            "tests": {
                tid: {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "variant_type": t.variant_type.value,
                    "variants": [
                        {
                            "id": v.id,
                            "name": v.name,
                            "description": v.description,
                            "variant_type": v.variant_type.value,
                            "config": v.config,
                        }
                        for v in t.variants
                    ],
                    "created_at": t.created_at.isoformat(),
                    "is_active": t.is_active,
                    "winner_id": t.winner_id,
                }
                for tid, t in self._tests.items()
            },
            "results": {
                tid: {
                    vid: {
                        "variant_id": r.variant_id,
                        "impressions": r.impressions,
                        "likes": r.likes,
                        "comments": r.comments,
                        "shares": r.shares,
                        "clicks": r.clicks,
                        "assignments": r.assignments,
                    }
                    for vid, r in variants.items()
                }
                for tid, variants in self._results.items()
            },
            "assignments": self._assignments,
        }

        self._storage_path.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> None:
        """Load state from disk."""
        if not self._storage_path or not self._storage_path.exists():
            return

        try:
            state = json.loads(self._storage_path.read_text())

            # Restore tests
            for tid, tdata in state.get("tests", {}).items():
                variants = [
                    Variant(
                        id=v["id"],
                        name=v["name"],
                        description=v["description"],
                        variant_type=VariantType(v["variant_type"]),
                        config=v.get("config", {}),
                    )
                    for v in tdata["variants"]
                ]

                self._tests[tid] = ABTest(
                    id=tdata["id"],
                    name=tdata["name"],
                    description=tdata["description"],
                    variant_type=VariantType(tdata["variant_type"]),
                    variants=variants,
                    created_at=datetime.fromisoformat(tdata["created_at"]),
                    is_active=tdata["is_active"],
                    winner_id=tdata.get("winner_id"),
                )

            # Restore results
            for tid, variants in state.get("results", {}).items():
                self._results[tid] = {
                    vid: VariantResult(
                        variant_id=r["variant_id"],
                        impressions=r["impressions"],
                        likes=r["likes"],
                        comments=r["comments"],
                        shares=r["shares"],
                        clicks=r["clicks"],
                        assignments=r["assignments"],
                    )
                    for vid, r in variants.items()
                }

            # Restore assignments
            self._assignments = state.get("assignments", {})

        except (json.JSONDecodeError, KeyError):
            pass  # Start fresh on error


# =============================================================================
# Pre-configured Tests
# =============================================================================


def create_headline_test(manager: ABTestManager) -> ABTest:
    """Create a pre-configured headline style A/B test."""
    return manager.create_test(
        name="Headline Style Test",
        description="Test question headlines vs statement headlines",
        variant_type=VariantType.HEADLINE_STYLE,
        variants=[
            (
                "Question",
                "Use question-format headlines",
                {"format": "question", "example": "Why is hydrogen the future?"},
            ),
            (
                "Statement",
                "Use statement-format headlines",
                {"format": "statement", "example": "Hydrogen is the future."},
            ),
        ],
    )


def create_cta_test(manager: ABTestManager) -> ABTest:
    """Create a pre-configured CTA format A/B test."""
    return manager.create_test(
        name="CTA Format Test",
        description="Test different call-to-action formats",
        variant_type=VariantType.CTA_FORMAT,
        variants=[
            (
                "Question CTA",
                "End with a question",
                {"format": "question", "example": "What do you think?"},
            ),
            (
                "Share CTA",
                "Ask for shares/reactions",
                {"format": "share", "example": "Share if you agree!"},
            ),
            (
                "Comment CTA",
                "Encourage comments",
                {"format": "comment", "example": "Drop your thoughts below."},
            ),
        ],
    )


def create_emoji_test(manager: ABTestManager) -> ABTest:
    """Create a pre-configured emoji usage A/B test."""
    return manager.create_test(
        name="Emoji Usage Test",
        description="Test posts with vs without emojis",
        variant_type=VariantType.EMOJI_USAGE,
        variants=[
            (
                "With Emojis",
                "Include strategic emojis",
                {"use_emojis": True, "max_emojis": 5},
            ),
            (
                "No Emojis",
                "Plain text only",
                {"use_emojis": False, "max_emojis": 0},
            ),
        ],
    )


# =============================================================================
# Utility Functions
# =============================================================================


def format_test_results(summary: TestSummary) -> str:
    """
    Format test results for display.

    Args:
        summary: The test summary to format

    Returns:
        Formatted string
    """
    lines = [
        f"A/B Test Results: {summary.test_name}",
        "=" * 50,
        f"Total Assignments: {summary.total_assignments}",
        "",
        "Variant Performance:",
    ]

    for variant_id, result in summary.variant_results.items():
        lines.append(
            f"  • {variant_id}: "
            f"{result.engagement_rate:.2f}% engagement "
            f"({result.total_engagement}/{result.impressions} impressions)"
        )

    lines.extend([
        "",
        f"Best Performer: {summary.best_variant_id}",
        f"Confidence: {summary.confidence_score:.1f}%",
        f"Statistically Significant: {'Yes' if summary.is_statistically_significant else 'No'}",
        "",
        f"Recommendation: {summary.recommendation}",
    ])

    return "\n".join(lines)


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for ab_testing module."""
    from test_framework import TestSuite

    suite = TestSuite("A/B Testing Tests")

    def test_variant_creation():
        variant = Variant(
            id="v1",
            name="Test Variant",
            description="A test",
            variant_type=VariantType.HEADLINE_STYLE,
        )
        assert variant.id == "v1"
        assert variant.name == "Test Variant"

    def test_variant_result_engagement():
        result = VariantResult(
            variant_id="v1",
            impressions=100,
            likes=10,
            comments=5,
            shares=2,
            clicks=3,
        )
        assert result.total_engagement == 20
        assert result.engagement_rate == 20.0

    def test_variant_result_zero_impressions():
        result = VariantResult(variant_id="v1", impressions=0)
        assert result.engagement_rate == 0.0

    def test_ab_test_creation():
        manager = ABTestManager()
        test = manager.create_test(
            name="Test",
            description="Description",
            variant_type=VariantType.EMOJI_USAGE,
            variants=[
                ("A", "Variant A", {}),
                ("B", "Variant B", {}),
            ],
        )
        assert test.name == "Test"
        assert len(test.variants) == 2
        assert test.is_active is True

    def test_ab_test_min_variants():
        try:
            ABTest(
                id="t1",
                name="Test",
                description="Test",
                variant_type=VariantType.EMOJI_USAGE,
                variants=[Variant("v1", "V1", "Desc", VariantType.EMOJI_USAGE)],
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_get_variant_deterministic():
        manager = ABTestManager()
        manager.create_test(
            name="Det Test",
            description="Test",
            variant_type=VariantType.HEADLINE_STYLE,
            variants=[("A", "A", {}), ("B", "B", {})],
        )
        test_id = list(manager._tests.keys())[0]

        # Same content should get same variant
        v1 = manager.get_variant_for_content(test_id, "content_123")
        v2 = manager.get_variant_for_content(test_id, "content_123")
        assert v1 is not None
        assert v1.id == v2.id

    def test_get_variant_different_content():
        manager = ABTestManager()
        manager.create_test(
            name="Diff Test",
            description="Test",
            variant_type=VariantType.HEADLINE_STYLE,
            variants=[("A", "A", {}), ("B", "B", {})],
        )
        test_id = list(manager._tests.keys())[0]

        variants_assigned = set()
        for i in range(100):
            v = manager.get_variant_for_content(test_id, f"content_{i}")
            if v:
                variants_assigned.add(v.id)

        # With 100 samples, should have both variants
        assert len(variants_assigned) == 2

    def test_record_metrics():
        manager = ABTestManager()
        test = manager.create_test(
            name="Metrics Test",
            description="Test",
            variant_type=VariantType.CTA_FORMAT,
            variants=[("A", "A", {}), ("B", "B", {})],
        )
        variant_id = test.variants[0].id

        manager.record_metrics(
            test.id, variant_id, impressions=100, likes=10, comments=5
        )

        result = manager._results[test.id][variant_id]
        assert result.impressions == 100
        assert result.likes == 10
        assert result.comments == 5

    def test_get_test_summary():
        manager = ABTestManager()
        test = manager.create_test(
            name="Summary Test",
            description="Test",
            variant_type=VariantType.EMOJI_USAGE,
            variants=[("A", "A", {}), ("B", "B", {})],
        )

        # Add some metrics
        manager.record_metrics(
            test.id, test.variants[0].id, impressions=100, likes=20
        )
        manager.record_metrics(
            test.id, test.variants[1].id, impressions=100, likes=10
        )

        summary = manager.get_test_summary(test.id)
        assert summary is not None
        assert summary.test_name == "Summary Test"
        assert summary.best_variant_id == test.variants[0].id

    def test_declare_winner():
        manager = ABTestManager()
        test = manager.create_test(
            name="Winner Test",
            description="Test",
            variant_type=VariantType.POST_LENGTH,
            variants=[("Short", "Short", {}), ("Long", "Long", {})],
        )

        result = manager.declare_winner(test.id, test.variants[0].id)
        assert result is True
        assert test.winner_id == test.variants[0].id
        assert test.is_active is False

    def test_get_active_tests():
        manager = ABTestManager()
        test1 = manager.create_test(
            name="Active 1",
            description="Test",
            variant_type=VariantType.EMOJI_USAGE,
            variants=[("A", "A", {}), ("B", "B", {})],
        )
        test2 = manager.create_test(
            name="Active 2",
            description="Test",
            variant_type=VariantType.CTA_FORMAT,
            variants=[("A", "A", {}), ("B", "B", {})],
        )

        manager.declare_winner(test1.id, test1.variants[0].id)

        active = manager.get_active_tests()
        assert len(active) == 1
        assert active[0].id == test2.id

    def test_create_headline_test():
        manager = ABTestManager()
        test = create_headline_test(manager)
        assert test.variant_type == VariantType.HEADLINE_STYLE
        assert len(test.variants) == 2

    def test_create_cta_test():
        manager = ABTestManager()
        test = create_cta_test(manager)
        assert test.variant_type == VariantType.CTA_FORMAT
        assert len(test.variants) == 3

    def test_create_emoji_test():
        manager = ABTestManager()
        test = create_emoji_test(manager)
        assert test.variant_type == VariantType.EMOJI_USAGE
        assert len(test.variants) == 2

    def test_format_test_results():
        summary = TestSummary(
            test_id="test_1",
            test_name="Test 1",
            variant_results={
                "v1": VariantResult("v1", impressions=100, likes=20, assignments=50),
                "v2": VariantResult("v2", impressions=100, likes=10, assignments=50),
            },
            total_assignments=100,
            best_variant_id="v1",
            confidence_score=75.0,
            is_statistically_significant=False,
            recommendation="Continue testing",
        )
        formatted = format_test_results(summary)
        assert "Test 1" in formatted
        assert "v1" in formatted
        assert "75.0%" in formatted

    suite.add_test("Variant creation", test_variant_creation)
    suite.add_test("Variant result engagement", test_variant_result_engagement)
    suite.add_test("Variant result zero impressions", test_variant_result_zero_impressions)
    suite.add_test("AB test creation", test_ab_test_creation)
    suite.add_test("AB test min variants", test_ab_test_min_variants)
    suite.add_test("Get variant deterministic", test_get_variant_deterministic)
    suite.add_test("Get variant different content", test_get_variant_different_content)
    suite.add_test("Record metrics", test_record_metrics)
    suite.add_test("Get test summary", test_get_test_summary)
    suite.add_test("Declare winner", test_declare_winner)
    suite.add_test("Get active tests", test_get_active_tests)
    suite.add_test("Create headline test", test_create_headline_test)
    suite.add_test("Create CTA test", test_create_cta_test)
    suite.add_test("Create emoji test", test_create_emoji_test)
    suite.add_test("Format test results", test_format_test_results)

    return suite
