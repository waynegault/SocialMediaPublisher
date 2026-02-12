"""Originality checking to ensure content is sufficiently paraphrased and unique.

This module provides:
- Text similarity analysis between summaries and source content
- N-gram overlap detection for phrase-level similarity
- LLM-based originality assessment
- Citation formatting recommendations
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

from google import genai  # type: ignore
from openai import OpenAI

from api_client import api_client
from config import Config
from database import Story

logger = logging.getLogger(__name__)


@dataclass
class OriginalityResult:
    """Result of an originality check."""

    is_original: bool
    similarity_score: float  # 0.0 = completely different, 1.0 = identical
    ngram_overlap_score: float  # Percentage of n-grams shared
    flagged_phrases: list[str]  # Phrases that appear too similar to source
    recommendation: str  # Suggested action or approval message
    details: dict[str, Any]  # Additional analysis details

    def __str__(self) -> str:
        status = "✓ ORIGINAL" if self.is_original else "⚠ NEEDS REVISION"
        return (
            f"{status} (similarity: {self.similarity_score:.1%}, "
            f"ngram overlap: {self.ngram_overlap_score:.1%})"
        )


class OriginalityChecker:
    """Check content originality to protect reputation and ensure unique insights."""

    # Default thresholds (overridden by Config if available)
    DEFAULT_MAX_SIMILARITY = 0.6  # Reject if word similarity > 60%
    DEFAULT_MAX_NGRAM_OVERLAP = 0.4  # Reject if n-gram overlap > 40%
    NGRAM_SIZE = 4  # Use 4-grams for phrase detection
    MIN_PHRASE_LENGTH = 5  # Minimum words for a flagged phrase

    def __init__(
        self,
        client: genai.Client | None = None,
        local_client: OpenAI | None = None,
    ):
        """
        Initialize the originality checker.

        Args:
            client: Gemini client for LLM-based assessment
            local_client: Local LLM client (LM Studio) as fallback
        """
        self.client = client
        self.local_client = local_client

        # Load thresholds from Config
        self.max_similarity = getattr(
            Config, "ORIGINALITY_MAX_SIMILARITY", self.DEFAULT_MAX_SIMILARITY
        )
        self.max_ngram_overlap = getattr(
            Config, "ORIGINALITY_MAX_NGRAM_OVERLAP", self.DEFAULT_MAX_NGRAM_OVERLAP
        )
        self.enabled = getattr(Config, "ORIGINALITY_CHECK_ENABLED", True)

    def check_originality(
        self,
        summary: str,
        source_texts: list[str],
        use_llm: bool = False,
    ) -> OriginalityResult:
        """
        Check if a summary is sufficiently original compared to source texts.

        Args:
            summary: The generated summary text to check
            source_texts: List of original source article texts
            use_llm: Whether to use LLM for deeper analysis (slower but more accurate)

        Returns:
            OriginalityResult with analysis details
        """
        if not summary or not source_texts:
            return OriginalityResult(
                is_original=True,
                similarity_score=0.0,
                ngram_overlap_score=0.0,
                flagged_phrases=[],
                recommendation="No source text available for comparison",
                details={"skipped": True},
            )

        # Combine all source texts for comparison
        combined_source = " ".join(source_texts)

        # Calculate word-level similarity (Jaccard)
        similarity_score = self._calculate_word_similarity(summary, combined_source)

        # Calculate n-gram overlap
        ngram_overlap, flagged_phrases = self._calculate_ngram_overlap(
            summary, combined_source
        )

        # Determine if content is original based on thresholds
        is_original = (
            similarity_score <= self.max_similarity
            and ngram_overlap <= self.max_ngram_overlap
        )

        # Build recommendation
        if is_original:
            recommendation = "Content is sufficiently paraphrased and original."
        else:
            issues = []
            if similarity_score > self.max_similarity:
                issues.append(
                    f"word similarity ({similarity_score:.0%}) exceeds {self.max_similarity:.0%} threshold"
                )
            if ngram_overlap > self.max_ngram_overlap:
                issues.append(
                    f"phrase overlap ({ngram_overlap:.0%}) exceeds {self.max_ngram_overlap:.0%} threshold"
                )
            recommendation = f"Needs revision: {'; '.join(issues)}."
            if flagged_phrases:
                recommendation += f" Rephrase: {flagged_phrases[:3]}"

        details = {
            "word_similarity": similarity_score,
            "ngram_overlap": ngram_overlap,
            "flagged_phrase_count": len(flagged_phrases),
            "source_word_count": len(combined_source.split()),
            "summary_word_count": len(summary.split()),
        }

        # Optional LLM assessment for borderline cases
        if use_llm and self.client and 0.4 <= similarity_score <= 0.7:
            llm_assessment = self._get_llm_assessment(summary, combined_source)
            details["llm_assessment"] = llm_assessment
            # LLM can override if it provides strong evidence
            if llm_assessment.get("override_decision"):
                is_original = llm_assessment.get("is_original", is_original)
                recommendation = llm_assessment.get("recommendation", recommendation)

        return OriginalityResult(
            is_original=is_original,
            similarity_score=similarity_score,
            ngram_overlap_score=ngram_overlap,
            flagged_phrases=flagged_phrases[:10],  # Limit to top 10
            recommendation=recommendation,
            details=details,
        )

    def check_story_originality(
        self,
        story: Story,
        source_content: str | None = None,
    ) -> OriginalityResult:
        """
        Convenience method to check originality for a Story object.

        Args:
            story: The Story object to check
            source_content: Optional source content (if not provided, uses source_links)

        Returns:
            OriginalityResult with analysis details
        """
        if not story.summary:
            return OriginalityResult(
                is_original=True,
                similarity_score=0.0,
                ngram_overlap_score=0.0,
                flagged_phrases=[],
                recommendation="No summary to check",
                details={"skipped": True, "reason": "empty_summary"},
            )

        source_texts = []
        if source_content:
            source_texts.append(source_content)

        # If no source content provided, we can only do basic checks
        if not source_texts:
            logger.debug(
                f"No source content for story '{story.title[:50]}...' - skipping deep check"
            )
            return OriginalityResult(
                is_original=True,
                similarity_score=0.0,
                ngram_overlap_score=0.0,
                flagged_phrases=[],
                recommendation="No source content available for comparison",
                details={"skipped": True, "reason": "no_source_content"},
            )

        return self.check_originality(story.summary, source_texts)

    def _calculate_word_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts using word sets.
        Returns a value between 0.0 (no similarity) and 1.0 (identical).
        """
        # Normalize: lowercase, remove punctuation, split into words
        words1 = set(re.sub(r"[^\w\s]", "", text1.lower()).split())
        words2 = set(re.sub(r"[^\w\s]", "", text2.lower()).split())

        # Remove common stopwords
        stopwords = {
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
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
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
        words1 -= stopwords
        words2 -= stopwords

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _calculate_ngram_overlap(
        self, summary: str, source: str
    ) -> tuple[float, list[str]]:
        """
        Calculate n-gram overlap between summary and source.

        Returns:
            Tuple of (overlap_ratio, list_of_matching_phrases)
        """
        # Generate n-grams from both texts
        summary_ngrams = self._get_ngrams(summary, self.NGRAM_SIZE)
        source_ngrams = self._get_ngrams(source, self.NGRAM_SIZE)

        if not summary_ngrams:
            return 0.0, []

        # Find overlapping n-grams
        summary_set = set(summary_ngrams)
        source_set = set(source_ngrams)
        overlap = summary_set & source_set

        # Calculate overlap ratio (what % of summary n-grams appear in source)
        overlap_ratio = len(overlap) / len(summary_set) if summary_set else 0.0

        # Extract the actual matching phrases
        matching_phrases = [" ".join(ngram) for ngram in sorted(overlap)[:20]]

        return overlap_ratio, matching_phrases

    def _get_ngrams(self, text: str, n: int) -> list[tuple[str, ...]]:
        """Extract n-grams from text."""
        # Normalize text
        words = re.sub(r"[^\w\s]", "", text.lower()).split()

        if len(words) < n:
            return []

        return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

    def _get_llm_assessment(self, summary: str, source: str) -> dict[str, Any]:
        """
        Use LLM for deeper originality assessment.

        Returns dict with:
            - is_original: bool
            - confidence: float (0-1)
            - issues: list of specific issues found
            - recommendation: str
            - override_decision: bool (whether to override threshold-based decision)
        """
        prompt = f"""Analyze the originality of this summary compared to the source text.

SUMMARY:
{summary[:1500]}

SOURCE EXCERPT:
{source[:2000]}

EVALUATE:
1. Is the summary sufficiently paraphrased (not just rearranged words)?
2. Does it add unique perspective or insight beyond the source?
3. Are there any phrases copied verbatim that should be quoted or rephrased?
4. Would this pass academic plagiarism standards?

Respond in this exact format:
ORIGINAL: YES or NO
CONFIDENCE: 0.0-1.0
ISSUES: [comma-separated list or "none"]
RECOMMENDATION: [one sentence]"""

        try:
            if self.local_client:
                content = api_client.local_llm_generate(
                    client=self.local_client,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    endpoint="originality_check",
                )
            elif self.client:
                response = api_client.gemini_generate(
                    client=self.client,
                    model=Config.MODEL_VERIFICATION,
                    contents=prompt,
                    endpoint="originality_check",
                )
                content = response.text or ""
            else:
                return {"error": "No LLM client available"}

            return self._parse_llm_assessment(content)

        except Exception as e:
            logger.warning(f"LLM originality assessment failed: {e}")
            return {"error": str(e)}

    def _parse_llm_assessment(self, response: str) -> dict[str, Any]:
        """Parse LLM response into structured assessment."""
        result: dict[str, Any] = {
            "is_original": True,
            "confidence": 0.5,
            "issues": [],
            "recommendation": "",
            "override_decision": False,
            "raw_response": response,
        }

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("ORIGINAL:"):
                value = line.replace("ORIGINAL:", "").strip().upper()
                result["is_original"] = value == "YES"
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(
                        line.replace("CONFIDENCE:", "").strip()
                    )
                except ValueError:
                    pass
            elif line.startswith("ISSUES:"):
                issues_str = line.replace("ISSUES:", "").strip()
                if issues_str.lower() != "none":
                    result["issues"] = [
                        i.strip() for i in issues_str.split(",") if i.strip()
                    ]
            elif line.startswith("RECOMMENDATION:"):
                result["recommendation"] = line.replace("RECOMMENDATION:", "").strip()

        # Override only if LLM is confident (>0.8)
        if result["confidence"] >= 0.8:
            result["override_decision"] = True

        return result

    def format_citation(self, source_url: str, source_title: str = "") -> str:
        """
        Format a source citation for inclusion in posts.

        Args:
            source_url: URL of the source
            source_title: Optional title of the source article

        Returns:
            Formatted citation string
        """
        if source_title:
            return f"Source: {source_title} ({source_url})"
        return f"Source: {source_url}"

    def suggest_citations(self, story: Story) -> list[str]:
        """
        Suggest citation formats for a story's sources.

        Returns list of formatted citation strings.
        """
        citations = []
        for url in story.source_links[:3]:  # Limit to 3 sources
            citations.append(self.format_citation(url))
        return citations


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests() -> bool:
    """Create unit tests for originality_checker module."""
    from test_framework import TestSuite

    suite = TestSuite("Originality Checker Tests", "originality_checker.py")
    suite.start_suite()

    def test_word_similarity_identical():
        checker = OriginalityChecker()
        result = checker._calculate_word_similarity(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
        )
        assert result > 0.9, (
            f"Identical texts should have high similarity, got {result}"
        )

    def test_word_similarity_different():
        checker = OriginalityChecker()
        result = checker._calculate_word_similarity(
            "Hydrogen fuel cells power electric vehicles",
            "Ocean waves generate renewable electricity",
        )
        assert result < 0.3, f"Different texts should have low similarity, got {result}"

    def test_word_similarity_partial():
        checker = OriginalityChecker()
        result = checker._calculate_word_similarity(
            "New catalyst improves hydrogen production efficiency",
            "Researchers develop catalyst for hydrogen production",
        )
        # Should have moderate similarity (shared key terms)
        assert 0.2 < result < 0.8, (
            f"Partial overlap should have moderate similarity, got {result}"
        )

    def test_ngram_overlap_identical():
        checker = OriginalityChecker()
        text = "The new hydrogen electrolyzer achieves record efficiency"
        overlap, phrases = checker._calculate_ngram_overlap(text, text)
        assert overlap > 0.9, (
            f"Identical texts should have high n-gram overlap, got {overlap}"
        )
        assert len(phrases) > 0, "Should have matching phrases"

    def test_ngram_overlap_different():
        checker = OriginalityChecker()
        overlap, phrases = checker._calculate_ngram_overlap(
            "Solar panels generate clean electricity efficiently",
            "Nuclear reactors produce steady baseload power",
        )
        assert overlap < 0.2, (
            f"Different texts should have low n-gram overlap, got {overlap}"
        )

    def test_ngram_extraction():
        checker = OriginalityChecker()
        ngrams = checker._get_ngrams("one two three four five six", 4)
        assert len(ngrams) == 3, f"Expected 3 4-grams, got {len(ngrams)}"
        assert ngrams[0] == ("one", "two", "three", "four")
        assert ngrams[2] == ("three", "four", "five", "six")

    def test_check_originality_original():
        checker = OriginalityChecker()
        result = checker.check_originality(
            summary="From my perspective as an engineer, the breakthrough in catalyst design "
            "represents a significant step forward for industrial hydrogen production.",
            source_texts=[
                "Scientists at MIT have developed a new catalyst that improves "
                "hydrogen production efficiency by 40%. The research was published "
                "in Nature Chemistry."
            ],
        )
        assert result.is_original, f"Should be original: {result.recommendation}"
        assert result.similarity_score < 0.6

    def test_check_originality_copied():
        checker = OriginalityChecker()
        source = (
            "Scientists at MIT have developed a new catalyst that improves "
            "hydrogen production efficiency by 40 percent. The breakthrough "
            "could accelerate the transition to clean energy."
        )
        result = checker.check_originality(
            summary=source,  # Exact copy
            source_texts=[source],
        )
        assert not result.is_original, f"Copied text should not be original: {result}"
        assert result.similarity_score > 0.8

    def test_check_originality_empty():
        checker = OriginalityChecker()
        result = checker.check_originality("", [])
        assert result.is_original  # Empty should pass (nothing to check)
        assert result.details.get("skipped") is True

    def test_check_story_originality():
        from database import Story

        checker = OriginalityChecker()
        story = Story(
            title="New Hydrogen Catalyst",
            summary="What stands out about this MIT research is the engineering challenge "
            "of scaling catalyst production while maintaining the reported 40% efficiency gains.",
        )
        result = checker.check_story_originality(
            story,
            source_content="MIT researchers develop new catalyst with 40% efficiency improvement.",
        )
        assert isinstance(result, OriginalityResult)
        assert result.similarity_score >= 0.0
        assert result.similarity_score <= 1.0

    def test_format_citation():
        checker = OriginalityChecker()
        citation = checker.format_citation(
            "https://example.com/article",
            "New Discovery in Hydrogen Tech",
        )
        assert "https://example.com/article" in citation
        assert "New Discovery" in citation

    def test_format_citation_no_title():
        checker = OriginalityChecker()
        citation = checker.format_citation("https://example.com/article")
        assert "https://example.com/article" in citation
        assert "Source:" in citation

    def test_suggest_citations():
        from database import Story

        checker = OriginalityChecker()
        story = Story(
            title="Test",
            summary="Test summary",
            source_links=[
                "https://a.com",
                "https://b.com",
                "https://c.com",
                "https://d.com",
            ],
        )
        citations = checker.suggest_citations(story)
        assert len(citations) == 3, "Should limit to 3 citations"

    def test_parse_llm_assessment():
        checker = OriginalityChecker()
        response = """ORIGINAL: YES
CONFIDENCE: 0.85
ISSUES: none
RECOMMENDATION: Content is well paraphrased with unique insights."""
        result = checker._parse_llm_assessment(response)
        assert result["is_original"] is True
        assert result["confidence"] == 0.85
        assert len(result["issues"]) == 0
        assert "paraphrased" in result["recommendation"]

    def test_parse_llm_assessment_with_issues():
        checker = OriginalityChecker()
        response = """ORIGINAL: NO
CONFIDENCE: 0.9
ISSUES: verbatim copying, lacks unique perspective
RECOMMENDATION: Rephrase the first paragraph."""
        result = checker._parse_llm_assessment(response)
        assert result["is_original"] is False
        assert result["confidence"] == 0.9
        assert len(result["issues"]) == 2
        assert "verbatim copying" in result["issues"]

    suite.run_test(
        test_name="Word similarity - identical",
        test_func=test_word_similarity_identical,
        test_summary="Tests Word similarity with identical scenario",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Word similarity - different",
        test_func=test_word_similarity_different,
        test_summary="Tests Word similarity with different scenario",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Word similarity - partial",
        test_func=test_word_similarity_partial,
        test_summary="Tests Word similarity with partial scenario",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="N-gram overlap - identical",
        test_func=test_ngram_overlap_identical,
        test_summary="Tests N-gram overlap with identical scenario",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="N-gram overlap - different",
        test_func=test_ngram_overlap_different,
        test_summary="Tests N-gram overlap with different scenario",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="N-gram extraction",
        test_func=test_ngram_extraction,
        test_summary="Tests N-gram extraction functionality",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function correctly parses and extracts the data",
    )
    suite.run_test(
        test_name="Check originality - original",
        test_func=test_check_originality_original,
        test_summary="Tests Check originality with original scenario",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Check originality - copied",
        test_func=test_check_originality_copied,
        test_summary="Tests Check originality with copied scenario",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function returns False or falsy value",
    )
    suite.run_test(
        test_name="Check originality - empty",
        test_func=test_check_originality_empty,
        test_summary="Tests Check originality with empty scenario",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function returns True",
    )
    suite.run_test(
        test_name="Check story originality",
        test_func=test_check_story_originality,
        test_summary="Tests Check story originality functionality",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Format citation",
        test_func=test_format_citation,
        test_summary="Tests Format citation functionality",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function produces correctly formatted output",
    )
    suite.run_test(
        test_name="Format citation - no title",
        test_func=test_format_citation_no_title,
        test_summary="Tests Format citation with no title scenario",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function handles empty or missing input gracefully",
    )
    suite.run_test(
        test_name="Suggest citations",
        test_func=test_suggest_citations,
        test_summary="Tests Suggest citations functionality",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Parse LLM assessment",
        test_func=test_parse_llm_assessment,
        test_summary="Tests Parse LLM assessment functionality",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function returns True",
    )
    suite.run_test(
        test_name="Parse LLM assessment - with issues",
        test_func=test_parse_llm_assessment_with_issues,
        test_summary="Tests Parse LLM assessment with with issues scenario",
        method_description="Calls OriginalityChecker and verifies the result",
        expected_outcome="Function correctly parses and extracts the data",
    )

    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
