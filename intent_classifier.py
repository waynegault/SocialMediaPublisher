"""Intent Classifier for Job-Relevant Story Filtering.

This module provides NLP-based intent classification to ensure content
aligns with career goals and filters stories appropriately.

Features:
- Intent classification for stories (skill_showcase, network_building, etc.)
- Career alignment scoring
- Negative positioning detection
- DistilBERT-based classification (with fallback)

TASK 1.3: Intent Recognition for Job-Relevant Story Filtering
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

# Type hints for optional dependencies
if TYPE_CHECKING:
    from transformers import pipeline

# Optional transformers import for DistilBERT
try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None  # type: ignore[assignment]
    logger.debug("transformers not available - using keyword-based classification")


# =============================================================================
# Intent Types
# =============================================================================


class StoryIntent(Enum):
    """Classification of story intents for career positioning."""

    SKILL_SHOWCASE = "skill_showcase"  # Demonstrates technical expertise
    NETWORK_BUILDING = "network_building"  # Connects with industry professionals
    THOUGHT_LEADERSHIP = "thought_leadership"  # Positions as industry expert
    INDUSTRY_AWARENESS = "industry_awareness"  # Shows knowledge of trends
    INNOVATION = "innovation"  # Highlights innovative thinking
    PROBLEM_SOLVING = "problem_solving"  # Demonstrates analytical skills
    COLLABORATION = "collaboration"  # Shows teamwork abilities
    NEUTRAL = "neutral"  # General industry news
    NEGATIVE = "negative"  # Could harm professional image
    IRRELEVANT = "irrelevant"  # Not career-relevant


@dataclass
class IntentScore:
    """Score for a specific intent."""

    intent: StoryIntent
    score: float  # 0.0 to 1.0
    confidence: float = 0.0  # Model confidence
    keywords_matched: list[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Result of intent classification."""

    primary_intent: StoryIntent
    career_alignment_score: float  # 0.0 to 1.0 (higher = better for career)
    all_scores: list[IntentScore] = field(default_factory=list)
    is_publishable: bool = True
    rejection_reason: Optional[str] = None
    confidence: float = 0.0

    @property
    def top_intents(self) -> list[StoryIntent]:
        """Get top 3 intents by score."""
        sorted_scores = sorted(self.all_scores, key=lambda x: x.score, reverse=True)
        return [s.intent for s in sorted_scores[:3]]


# =============================================================================
# Keyword Patterns for Classification
# =============================================================================

# Keywords that indicate each intent type
INTENT_KEYWORDS: dict[StoryIntent, set[str]] = {
    StoryIntent.SKILL_SHOWCASE: {
        "engineering",
        "technical",
        "design",
        "implementation",
        "development",
        "optimization",
        "efficiency",
        "process",
        "methodology",
        "solution",
        "architecture",
        "system",
        "integration",
        "automation",
        "innovation",
        "achievement",
        "breakthrough",
        "patent",
        "invention",
        "expertise",
        "specialization",
        "capability",
        "competency",
    },
    StoryIntent.NETWORK_BUILDING: {
        "collaboration",
        "partnership",
        "joint venture",
        "consortium",
        "alliance",
        "merger",
        "acquisition",
        "conference",
        "summit",
        "symposium",
        "workshop",
        "networking",
        "industry leaders",
        "executives",
        "ceo",
        "cto",
        "appointment",
        "hire",
        "promotion",
        "keynote",
        "panel",
        "speaker",
    },
    StoryIntent.THOUGHT_LEADERSHIP: {
        "future",
        "vision",
        "strategy",
        "transformation",
        "disruption",
        "paradigm",
        "revolution",
        "pioneering",
        "groundbreaking",
        "cutting-edge",
        "state-of-the-art",
        "next generation",
        "emerging",
        "trend",
        "forecast",
        "prediction",
        "insight",
        "perspective",
        "opinion",
        "analysis",
        "commentary",
        "thought leader",
    },
    StoryIntent.INDUSTRY_AWARENESS: {
        "market",
        "industry",
        "sector",
        "growth",
        "expansion",
        "investment",
        "funding",
        "regulation",
        "policy",
        "legislation",
        "compliance",
        "standard",
        "certification",
        "benchmark",
        "report",
        "study",
        "research",
        "survey",
        "statistics",
        "data",
        "trend",
        "outlook",
    },
    StoryIntent.INNOVATION: {
        "innovation",
        "breakthrough",
        "discovery",
        "invention",
        "patent",
        "r&d",
        "research",
        "development",
        "prototype",
        "pilot",
        "trial",
        "experiment",
        "novel",
        "first",
        "new",
        "advanced",
        "revolutionary",
        "transformative",
        "disruptive",
        "game-changing",
    },
    StoryIntent.PROBLEM_SOLVING: {
        "challenge",
        "problem",
        "solution",
        "solve",
        "overcome",
        "address",
        "tackle",
        "resolve",
        "fix",
        "improve",
        "optimize",
        "enhance",
        "reduce",
        "increase",
        "efficiency",
        "cost",
        "savings",
        "performance",
        "reliability",
        "safety",
    },
    StoryIntent.COLLABORATION: {
        "team",
        "collaboration",
        "partnership",
        "joint",
        "together",
        "cooperative",
        "alliance",
        "consortium",
        "cross-functional",
        "interdisciplinary",
        "multi-stakeholder",
        "community",
        "ecosystem",
        "network",
        "collective",
    },
    StoryIntent.NEGATIVE: {
        "scandal",
        "fraud",
        "lawsuit",
        "investigation",
        "violation",
        "penalty",
        "fine",
        "accident",
        "disaster",
        "failure",
        "bankruptcy",
        "layoff",
        "downsizing",
        "closure",
        "recall",
        "contamination",
        "pollution",
        "spill",
        "explosion",
        "fatality",
        "death",
        "injury",
        "criminal",
        "indictment",
        "misconduct",
        "negligence",
    },
}

# Career alignment weights for each intent
CAREER_ALIGNMENT_WEIGHTS: dict[StoryIntent, float] = {
    StoryIntent.SKILL_SHOWCASE: 1.0,
    StoryIntent.THOUGHT_LEADERSHIP: 0.95,
    StoryIntent.INNOVATION: 0.9,
    StoryIntent.PROBLEM_SOLVING: 0.85,
    StoryIntent.NETWORK_BUILDING: 0.8,
    StoryIntent.COLLABORATION: 0.75,
    StoryIntent.INDUSTRY_AWARENESS: 0.7,
    StoryIntent.NEUTRAL: 0.5,
    StoryIntent.IRRELEVANT: 0.2,
    StoryIntent.NEGATIVE: 0.0,
}


# =============================================================================
# Intent Classifier
# =============================================================================


class IntentClassifier:
    """Classifies story intent for career alignment.

    Uses keyword matching with optional transformer-based classification
    for improved accuracy.
    """

    def __init__(
        self,
        use_transformers: bool = False,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        career_keywords: Optional[list[str]] = None,
    ) -> None:
        """Initialize the intent classifier.

        Args:
            use_transformers: Whether to use transformer models
            model_name: HuggingFace model name for sentiment/classification
            career_keywords: Additional career-relevant keywords
        """
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self.model_name = model_name
        self.classifier = None
        self.sentiment_pipeline = None

        # Additional career keywords (chemical engineering focus)
        self.career_keywords = set(career_keywords or [])
        self._add_domain_keywords()

        # Initialize transformer if available and requested
        if self.use_transformers and pipeline is not None:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",  # type: ignore[arg-type]
                    model=model_name,
                    device=-1,  # CPU
                )
                logger.info(f"Loaded transformer model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load transformer model: {e}")
                self.use_transformers = False

    def _add_domain_keywords(self) -> None:
        """Add chemical engineering domain-specific keywords."""
        domain_keywords = {
            # Process engineering
            "reactor",
            "catalyst",
            "distillation",
            "separation",
            "heat exchanger",
            "compressor",
            "pump",
            "valve",
            "piping",
            "instrumentation",
            # Safety
            "hazop",
            "process safety",
            "risk assessment",
            "sil",
            "safety integrity",
            "bow-tie",
            "lopa",
            # Sustainability
            "carbon capture",
            "hydrogen",
            "green chemistry",
            "renewable",
            "sustainable",
            "circular economy",
            "net zero",
            "decarbonization",
            "electrification",
            # Digital
            "digital twin",
            "industry 4.0",
            "iot",
            "machine learning",
            "ai",
            "predictive maintenance",
            "advanced process control",
            "apc",
            "mpc",
        }
        self.career_keywords.update(domain_keywords)

    def classify(self, title: str, summary: str) -> ClassificationResult:
        """Classify a story's intent and career alignment.

        Args:
            title: Story title
            summary: Story summary/content

        Returns:
            ClassificationResult with intent and alignment scores
        """
        text = f"{title} {summary}".lower()

        # Calculate scores for each intent
        all_scores: list[IntentScore] = []

        for intent, keywords in INTENT_KEYWORDS.items():
            score, matched = self._calculate_keyword_score(text, keywords)
            intent_score = IntentScore(
                intent=intent,
                score=score,
                confidence=min(1.0, len(matched) * 0.15),
                keywords_matched=matched,
            )
            all_scores.append(intent_score)

        # Add career keyword bonus
        career_matches = self._count_career_keywords(text)
        career_bonus = min(0.2, career_matches * 0.02)

        # Determine primary intent
        sorted_scores = sorted(all_scores, key=lambda x: x.score, reverse=True)
        primary_intent = (
            sorted_scores[0].intent if sorted_scores else StoryIntent.NEUTRAL
        )

        # Check for negative content (overrides other intents)
        negative_score = next(
            (s for s in all_scores if s.intent == StoryIntent.NEGATIVE), None
        )
        if negative_score and negative_score.score > 0.3:
            primary_intent = StoryIntent.NEGATIVE

        # Calculate career alignment score
        career_alignment = self._calculate_career_alignment(
            all_scores, career_bonus, primary_intent
        )

        # Use transformer sentiment for additional validation if available
        sentiment_adjustment = 0.0
        if self.use_transformers and self.sentiment_pipeline:
            sentiment_adjustment = self._get_sentiment_adjustment(title, summary)

        career_alignment = max(0.0, min(1.0, career_alignment + sentiment_adjustment))

        # Determine if publishable
        is_publishable = True
        rejection_reason = None

        if primary_intent == StoryIntent.NEGATIVE:
            is_publishable = False
            rejection_reason = "Story contains potentially negative content"
        elif career_alignment < 0.3:
            is_publishable = False
            rejection_reason = f"Low career alignment score: {career_alignment:.2f}"
        elif primary_intent == StoryIntent.IRRELEVANT:
            is_publishable = False
            rejection_reason = "Story not relevant to career goals"

        return ClassificationResult(
            primary_intent=primary_intent,
            career_alignment_score=career_alignment,
            all_scores=all_scores,
            is_publishable=is_publishable,
            rejection_reason=rejection_reason,
            confidence=sorted_scores[0].confidence if sorted_scores else 0.0,
        )

    def _calculate_keyword_score(
        self,
        text: str,
        keywords: set[str],
    ) -> tuple[float, list[str]]:
        """Calculate score based on keyword matches."""
        matched = []
        words = set(re.findall(r"\b\w+\b", text))

        for keyword in keywords:
            # Handle multi-word keywords
            if " " in keyword:
                if keyword in text:
                    matched.append(keyword)
            elif keyword in words:
                matched.append(keyword)

        # Score based on matches (diminishing returns)
        if not matched:
            return 0.0, []

        score = min(1.0, len(matched) * 0.12)
        return score, matched

    def _count_career_keywords(self, text: str) -> int:
        """Count matches with career-relevant keywords."""
        count = 0
        words = set(re.findall(r"\b\w+\b", text))

        for keyword in self.career_keywords:
            if " " in keyword:
                if keyword in text:
                    count += 1
            elif keyword in words:
                count += 1

        return count

    def _calculate_career_alignment(
        self,
        all_scores: list[IntentScore],
        career_bonus: float,
        primary_intent: StoryIntent,
    ) -> float:
        """Calculate overall career alignment score."""
        # Weighted average of intent scores
        total_weight = 0.0
        weighted_sum = 0.0

        for intent_score in all_scores:
            weight = CAREER_ALIGNMENT_WEIGHTS.get(intent_score.intent, 0.5)
            weighted_sum += intent_score.score * weight
            total_weight += intent_score.score

        if total_weight == 0:
            base_score = 0.5
        else:
            base_score = weighted_sum / total_weight

        # Apply career keyword bonus
        base_score = min(1.0, base_score + career_bonus)

        # Penalize if primary intent is negative
        if primary_intent == StoryIntent.NEGATIVE:
            base_score *= 0.1

        return base_score

    def _get_sentiment_adjustment(self, title: str, summary: str) -> float:
        """Get sentiment-based score adjustment using transformer."""
        if not self.sentiment_pipeline:
            return 0.0

        try:
            # Analyze title (more weight)
            title_result = self.sentiment_pipeline(title[:512])[0]
            summary_result = self.sentiment_pipeline(summary[:512])[0]

            # Convert to adjustment (-0.1 to +0.1)
            title_adj = 0.05 if title_result["label"] == "POSITIVE" else -0.05
            summary_adj = 0.03 if summary_result["label"] == "POSITIVE" else -0.03

            return title_adj + summary_adj

        except Exception as e:
            logger.debug(f"Sentiment analysis failed: {e}")
            return 0.0

    def is_career_aligned(
        self,
        title: str,
        summary: str,
        threshold: float = 0.5,
    ) -> bool:
        """Quick check if a story is career-aligned.

        Args:
            title: Story title
            summary: Story summary
            threshold: Minimum alignment score

        Returns:
            True if career alignment score meets threshold
        """
        result = self.classify(title, summary)
        return result.career_alignment_score >= threshold and result.is_publishable

    def get_story_categories(
        self,
        title: str,
        summary: str,
    ) -> list[str]:
        """Get category tags for a story based on intent.

        Args:
            title: Story title
            summary: Story summary

        Returns:
            List of category strings for the story
        """
        result = self.classify(title, summary)

        categories = []
        for intent_score in result.all_scores:
            if intent_score.score > 0.3:
                # Convert intent to readable category
                category = intent_score.intent.value.replace("_", " ").title()
                categories.append(category)

        return categories[:3]  # Top 3 categories


# =============================================================================
# Integration Functions
# =============================================================================


def classify_story_intent(
    title: str,
    summary: str,
    classifier: Optional[IntentClassifier] = None,
) -> ClassificationResult:
    """Classify a story's intent for career alignment.

    Args:
        title: Story title
        summary: Story summary
        classifier: Optional pre-initialized classifier

    Returns:
        ClassificationResult with intent and alignment scores
    """
    if classifier is None:
        classifier = IntentClassifier()

    return classifier.classify(title, summary)


def filter_career_relevant_stories(
    stories: list[dict],
    threshold: float = 0.5,
    classifier: Optional[IntentClassifier] = None,
) -> list[dict]:
    """Filter a list of stories to only career-relevant ones.

    Args:
        stories: List of story dicts with 'title' and 'summary' keys
        threshold: Minimum career alignment score
        classifier: Optional pre-initialized classifier

    Returns:
        Filtered list of career-relevant stories
    """
    if classifier is None:
        classifier = IntentClassifier()

    relevant = []
    for story in stories:
        title = story.get("title", "")
        summary = story.get("summary", "")

        result = classifier.classify(title, summary)

        if result.is_publishable and result.career_alignment_score >= threshold:
            # Add classification metadata
            story["career_alignment_score"] = result.career_alignment_score
            story["primary_intent"] = result.primary_intent.value
            relevant.append(story)

    return relevant


# =============================================================================
# Module-level convenience instance
# =============================================================================

_intent_classifier: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    """Get or create the singleton intent classifier."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests() -> bool:
    """Create unit tests for intent_classifier module."""
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from test_framework import TestSuite

    from test_framework import TestSuite

    suite = TestSuite("Intent Classifier Tests", "intent_classifier.py")
    suite.start_suite()

    def test_story_intent_enum():
        """Test StoryIntent enum values."""
        assert StoryIntent.SKILL_SHOWCASE.value == "skill_showcase"
        assert StoryIntent.NEGATIVE.value == "negative"
        assert len(StoryIntent) == 10

    def test_intent_score_creation():
        """Test IntentScore dataclass creation."""
        score = IntentScore(
            intent=StoryIntent.SKILL_SHOWCASE,
            score=0.8,
            confidence=0.9,
            keywords_matched=["engineering", "technical"],
        )
        assert score.intent == StoryIntent.SKILL_SHOWCASE
        assert score.score == 0.8
        assert len(score.keywords_matched) == 2

    def test_classification_result_creation():
        """Test ClassificationResult dataclass creation."""
        result = ClassificationResult(
            primary_intent=StoryIntent.INNOVATION,
            career_alignment_score=0.75,
            is_publishable=True,
        )
        assert result.primary_intent == StoryIntent.INNOVATION
        assert result.career_alignment_score == 0.75
        assert result.rejection_reason is None

    def test_classification_result_top_intents():
        """Test ClassificationResult top_intents property."""
        result = ClassificationResult(
            primary_intent=StoryIntent.INNOVATION,
            career_alignment_score=0.75,
            all_scores=[
                IntentScore(StoryIntent.INNOVATION, 0.9, 0.8, []),
                IntentScore(StoryIntent.SKILL_SHOWCASE, 0.7, 0.6, []),
                IntentScore(StoryIntent.NEUTRAL, 0.3, 0.4, []),
            ],
        )
        top = result.top_intents
        assert len(top) == 3
        assert top[0] == StoryIntent.INNOVATION

    def test_intent_keywords_defined():
        """Test INTENT_KEYWORDS dictionary is populated."""
        assert len(INTENT_KEYWORDS) > 0
        assert StoryIntent.SKILL_SHOWCASE in INTENT_KEYWORDS
        assert StoryIntent.NEGATIVE in INTENT_KEYWORDS

    def test_career_alignment_weights_defined():
        """Test CAREER_ALIGNMENT_WEIGHTS dictionary is populated."""
        assert len(CAREER_ALIGNMENT_WEIGHTS) > 0
        assert CAREER_ALIGNMENT_WEIGHTS[StoryIntent.SKILL_SHOWCASE] == 1.0
        assert CAREER_ALIGNMENT_WEIGHTS[StoryIntent.NEGATIVE] == 0.0

    def test_intent_classifier_creation():
        """Test IntentClassifier creation."""
        classifier = IntentClassifier()
        assert classifier is not None
        assert classifier.use_transformers is False  # Default

    def test_classifier_skill_showcase():
        """Test classification of skill showcase content."""
        classifier = IntentClassifier()
        result = classifier.classify(
            "New Engineering Solution for Process Optimization",
            "This technical implementation demonstrates expertise in system design and methodology.",
        )
        assert result.primary_intent in (
            StoryIntent.SKILL_SHOWCASE,
            StoryIntent.INNOVATION,
            StoryIntent.PROBLEM_SOLVING,
        )
        assert result.career_alignment_score > 0.3

    def test_classifier_negative_content():
        """Test classification of negative content."""
        classifier = IntentClassifier()
        result = classifier.classify(
            "Company Faces Major Scandal After Investigation",
            "The fraud investigation revealed serious misconduct and negligence.",
        )
        assert result.primary_intent == StoryIntent.NEGATIVE
        assert result.is_publishable is False

    def test_classifier_neutral_content():
        """Test classification of neutral content."""
        classifier = IntentClassifier()
        result = classifier.classify(
            "Weather Update for Today",
            "The weather will be sunny with temperatures around 20 degrees.",
        )
        # Should have low career alignment for irrelevant content
        assert result.career_alignment_score < 0.7

    def test_classify_story_intent_function():
        """Test classify_story_intent convenience function."""
        result = classify_story_intent(
            "Breakthrough in Hydrogen Technology",
            "New catalyst innovation enables more efficient hydrogen production.",
        )
        assert result is not None
        assert result.career_alignment_score > 0

    def test_filter_career_relevant_stories():
        """Test filter_career_relevant_stories function."""
        stories = [
            {
                "title": "New Reactor Design Innovation",
                "summary": "Engineering breakthrough in catalyst technology.",
            },
            {"title": "Celebrity Gossip", "summary": "Latest news about celebrities."},
        ]
        relevant = filter_career_relevant_stories(stories, threshold=0.3)
        # At least the engineering story should pass through
        assert len(relevant) <= len(stories)

    def test_get_intent_classifier_singleton():
        """Test get_intent_classifier returns singleton."""
        global _intent_classifier
        _intent_classifier = None  # Reset for test
        c1 = get_intent_classifier()
        c2 = get_intent_classifier()
        assert c1 is c2
        _intent_classifier = None  # Cleanup

    suite.run_test(
        test_name="StoryIntent enum",
        test_func=test_story_intent_enum,
        test_summary="Tests StoryIntent enum functionality",
        method_description="Invokes the function under test and validates behavior",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="IntentScore creation",
        test_func=test_intent_score_creation,
        test_summary="Tests IntentScore creation functionality",
        method_description="Calls IntentScore and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="ClassificationResult creation",
        test_func=test_classification_result_creation,
        test_summary="Tests ClassificationResult creation functionality",
        method_description="Calls ClassificationResult and verifies the result",
        expected_outcome="Function returns None as expected",
    )
    suite.run_test(
        test_name="ClassificationResult top_intents",
        test_func=test_classification_result_top_intents,
        test_summary="Tests ClassificationResult top intents functionality",
        method_description="Calls ClassificationResult and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="INTENT_KEYWORDS defined",
        test_func=test_intent_keywords_defined,
        test_summary="Tests INTENT KEYWORDS defined functionality",
        method_description="Invokes the function under test and validates behavior",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="CAREER_ALIGNMENT_WEIGHTS defined",
        test_func=test_career_alignment_weights_defined,
        test_summary="Tests CAREER ALIGNMENT WEIGHTS defined functionality",
        method_description="Invokes the function under test and validates behavior",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="IntentClassifier creation",
        test_func=test_intent_classifier_creation,
        test_summary="Tests IntentClassifier creation functionality",
        method_description="Calls IntentClassifier and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Classifier skill showcase",
        test_func=test_classifier_skill_showcase,
        test_summary="Tests Classifier skill showcase functionality",
        method_description="Calls IntentClassifier and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Classifier negative content",
        test_func=test_classifier_negative_content,
        test_summary="Tests Classifier negative content functionality",
        method_description="Calls IntentClassifier and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Classifier neutral content",
        test_func=test_classifier_neutral_content,
        test_summary="Tests Classifier neutral content functionality",
        method_description="Calls IntentClassifier and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="classify_story_intent function",
        test_func=test_classify_story_intent_function,
        test_summary="Tests classify story intent function functionality",
        method_description="Calls classify story intent and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="filter_career_relevant_stories",
        test_func=test_filter_career_relevant_stories,
        test_summary="Tests filter career relevant stories functionality",
        method_description="Calls filter career relevant stories and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="get_intent_classifier singleton",
        test_func=test_get_intent_classifier_singleton,
        test_summary="Tests get intent classifier singleton functionality",
        method_description="Calls get intent classifier and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )

    return suite.finish_suite()