"""LinkedIn post optimization for maximum algorithm favor and engagement.

This module implements best practices for LinkedIn content optimization including:
- Optimal post length (1200-1500 characters)
- Hook + Value + CTA structure
- Mobile-friendly formatting
- Engagement pattern analysis
- Dwell time optimization
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PostAnalysis:
    """Analysis results for a LinkedIn post."""

    original_length: int
    optimized_length: int
    paragraph_count: int
    has_hook: bool
    has_cta: bool
    readability_score: float  # 0.0 = poor, 1.0 = excellent
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "✓ OPTIMIZED" if self.readability_score >= 0.7 else "⚠ NEEDS WORK"
        return (
            f"{status} ({self.optimized_length} chars, "
            f"{self.paragraph_count} paragraphs, "
            f"readability: {self.readability_score:.0%})"
        )


class LinkedInOptimizer:
    """Optimize LinkedIn posts for maximum engagement and algorithm favor."""

    # Optimal post parameters based on LinkedIn best practices
    OPTIMAL_LENGTH_MIN = 1200  # Characters
    OPTIMAL_LENGTH_MAX = 1500  # Characters
    OPTIMAL_LENGTH_ABSOLUTE_MAX = 3000  # LinkedIn's limit is ~3000
    OPTIMAL_PARAGRAPHS = 4  # For mobile readability
    OPTIMAL_WORDS_PER_PARAGRAPH = 25  # Keep paragraphs short
    OPTIMAL_LINE_LENGTH = 80  # Characters per line for readability

    # Hook patterns that work well on LinkedIn
    HOOK_STARTERS = [
        "I just discovered",
        "Here's what most people miss about",
        "The surprising truth about",
        "This changed how I think about",
        "A breakthrough in",
        "Why",
        "How",
        "What if",
        "The future of",
        "Breaking:",
    ]

    # CTA patterns that drive engagement
    CTA_PATTERNS = [
        r"what do you think\??",
        r"thoughts\??",
        r"agree\??",
        r"share your",
        r"let me know",
        r"drop a comment",
        r"follow for more",
        r"connect with me",
        r"what's your take",
        r"have you experienced",
    ]

    # Patterns LinkedIn algorithm may penalize
    SPAM_PATTERNS = [
        r"(?i)like if you agree",
        r"(?i)share this post",
        r"(?i)comment \d+ for",
        r"(?i)drop an emoji",
        r"(?i)follow me for",
        r"(?i)link in (bio|comments)",
        r"(?i)dm me for",
        r"(?i)🔥{3,}",  # Excessive fire emojis
        r"(?i)👇{3,}",  # Excessive pointing emojis
    ]

    def __init__(self):
        """Initialize the LinkedIn optimizer."""
        # Load config if available
        try:
            from config import Config

            self.min_length = getattr(
                Config, "LINKEDIN_OPTIMAL_LENGTH_MIN", self.OPTIMAL_LENGTH_MIN
            )
            self.max_length = getattr(
                Config, "LINKEDIN_OPTIMAL_LENGTH_MAX", self.OPTIMAL_LENGTH_MAX
            )
        except ImportError:
            self.min_length = self.OPTIMAL_LENGTH_MIN
            self.max_length = self.OPTIMAL_LENGTH_MAX

    def analyze_post(self, content: str) -> PostAnalysis:
        """
        Analyze a post for LinkedIn optimization.

        Args:
            content: The post text to analyze

        Returns:
            PostAnalysis with metrics and suggestions
        """
        if not content:
            return PostAnalysis(
                original_length=0,
                optimized_length=0,
                paragraph_count=0,
                has_hook=False,
                has_cta=False,
                readability_score=0.0,
                warnings=["Empty content"],
                suggestions=["Add content to the post"],
            )

        warnings: list[str] = []
        suggestions: list[str] = []

        # Calculate metrics
        original_length = len(content)
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        paragraph_count = len(paragraphs)

        # Check for hook
        has_hook = self._has_strong_hook(content)
        if not has_hook:
            suggestions.append("Add a compelling hook in the first line")

        # Check for CTA
        has_cta = self._has_call_to_action(content)
        if not has_cta:
            suggestions.append("Add a call-to-action to encourage engagement")

        # Check length
        if original_length < self.min_length:
            suggestions.append(
                f"Consider expanding content (currently {original_length} chars, "
                f"optimal is {self.min_length}-{self.max_length})"
            )
        elif original_length > self.max_length:
            if original_length > self.OPTIMAL_LENGTH_ABSOLUTE_MAX:
                warnings.append(
                    f"Content exceeds LinkedIn limit ({original_length} chars)"
                )
            else:
                suggestions.append(
                    f"Consider condensing content (currently {original_length} chars, "
                    f"optimal is {self.min_length}-{self.max_length})"
                )

        # Check for spam patterns
        spam_found = self._detect_spam_patterns(content)
        if spam_found:
            warnings.extend(spam_found)

        # Check paragraph structure
        if paragraph_count < 2:
            suggestions.append("Break content into multiple paragraphs for readability")
        elif paragraph_count > 6:
            suggestions.append("Consider consolidating paragraphs (too many breaks)")

        # Calculate readability score
        readability_score = self._calculate_readability_score(
            content, has_hook, has_cta, paragraph_count, len(warnings)
        )

        return PostAnalysis(
            original_length=original_length,
            optimized_length=original_length,  # Will be updated by optimize_post
            paragraph_count=paragraph_count,
            has_hook=has_hook,
            has_cta=has_cta,
            readability_score=readability_score,
            warnings=warnings,
            suggestions=suggestions,
        )

    def optimize_post(self, content: str, add_hook: bool = False) -> tuple[str, PostAnalysis]:
        """
        Optimize a post for LinkedIn engagement.

        Args:
            content: The post text to optimize
            add_hook: Whether to add a hook if missing

        Returns:
            Tuple of (optimized_content, analysis)
        """
        if not content:
            return content, self.analyze_post(content)

        optimized = content

        # Format for mobile readability
        optimized = self._format_for_mobile(optimized)

        # Add strategic line breaks
        optimized = self._add_strategic_breaks(optimized)

        # Remove spam patterns
        optimized = self._remove_spam_patterns(optimized)

        # Analyze the optimized version
        analysis = self.analyze_post(optimized)
        analysis.optimized_length = len(optimized)

        return optimized, analysis

    def create_hook(self, topic: str) -> str:
        """
        Create a compelling hook for a given topic.

        Args:
            topic: The main topic of the post

        Returns:
            A hook sentence
        """
        # Simple hook templates
        hooks = [
            f"The future of {topic} is here — and it's not what you'd expect.",
            f"Here's what most engineers miss about {topic}:",
            f"I've been studying {topic} for years. This discovery changes everything.",
            f"Breaking: Major development in {topic} that could reshape the industry.",
            f"Why {topic} matters more than ever for the energy transition:",
        ]

        # Return first hook (in production, could use LLM to pick best)
        import random
        return random.choice(hooks)

    def create_cta(self, topic: str) -> str:
        """
        Create an engaging call-to-action.

        Args:
            topic: The main topic of the post

        Returns:
            A CTA sentence
        """
        ctas = [
            f"\n\nWhat's your take on {topic}? I'd love to hear your perspective.",
            f"\n\nHave you worked with {topic}? Share your experience below.",
            "\n\nThoughts? Let me know in the comments.",
            "\n\nWhat do you think the next breakthrough will be?",
            "\n\nAre you seeing similar trends in your work?",
        ]

        import random
        return random.choice(ctas)

    def _has_strong_hook(self, content: str) -> bool:
        """Check if content has a compelling opening hook."""
        if not content:
            return False

        first_line = content.split("\n")[0].strip().lower()

        # Check for hook starters
        for starter in self.HOOK_STARTERS:
            if first_line.startswith(starter.lower()):
                return True

        # Check if first line ends with a question or exclamation
        if first_line.endswith("?") or first_line.endswith("!"):
            return True

        # Check if first line is short and punchy (under 100 chars)
        if len(first_line) < 100 and len(first_line) > 20:
            return True

        return False

    def _has_call_to_action(self, content: str) -> bool:
        """Check if content has a call-to-action."""
        if not content:
            return False

        # Get last 200 characters (where CTA usually is)
        last_section = content[-200:].lower()

        for pattern in self.CTA_PATTERNS:
            if re.search(pattern, last_section):
                return True

        return False

    def _detect_spam_patterns(self, content: str) -> list[str]:
        """Detect patterns that LinkedIn may penalize."""
        warnings = []

        for pattern in self.SPAM_PATTERNS:
            if re.search(pattern, content):
                warnings.append(f"Detected potentially penalized pattern: {pattern}")

        # Check for excessive hashtags
        hashtags = re.findall(r"#\w+", content)
        if len(hashtags) > 5:
            warnings.append(f"Too many hashtags ({len(hashtags)}) - LinkedIn prefers 3-5")

        # Check for excessive emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "]+",
            flags=re.UNICODE,
        )
        emojis = emoji_pattern.findall(content)
        total_emojis = sum(len(e) for e in emojis)
        if total_emojis > 10:
            warnings.append(f"Excessive emojis ({total_emojis}) may reduce reach")

        return warnings

    def _remove_spam_patterns(self, content: str) -> str:
        """Remove or replace spam patterns."""
        result = content

        # Remove excessive emoji repetitions
        result = re.sub(r"([\U0001F600-\U0001F64F])\1{2,}", r"\1", result)
        result = re.sub(r"([\U0001F300-\U0001F5FF])\1{2,}", r"\1", result)

        return result

    def _format_for_mobile(self, content: str) -> str:
        """Format content for optimal mobile reading."""
        # Split into paragraphs
        paragraphs = content.split("\n\n")

        formatted_paragraphs = []
        for para in paragraphs:
            # Remove excessive whitespace within paragraphs
            para = " ".join(para.split())

            # Keep paragraphs reasonably short
            if len(para) > 300:
                # Try to split at sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 250:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            formatted_paragraphs.append(current_chunk.strip())
                        current_chunk = sentence + " "
                if current_chunk:
                    formatted_paragraphs.append(current_chunk.strip())
            else:
                if para.strip():
                    formatted_paragraphs.append(para.strip())

        return "\n\n".join(formatted_paragraphs)

    def _add_strategic_breaks(self, content: str) -> str:
        """Add line breaks at strategic points for readability."""
        # Add break before bullet points if missing
        result = re.sub(r"([^\n])(\n[•\-\*])", r"\1\n\2", content)

        # Ensure break before numbered lists
        result = re.sub(r"([^\n])(\n\d+\.)", r"\1\n\2", result)

        return result

    def _calculate_readability_score(
        self,
        content: str,
        has_hook: bool,
        has_cta: bool,
        paragraph_count: int,
        warning_count: int,
    ) -> float:
        """Calculate an overall readability score (0.0 - 1.0)."""
        score = 0.5  # Base score

        # Hook bonus
        if has_hook:
            score += 0.15

        # CTA bonus
        if has_cta:
            score += 0.15

        # Paragraph structure bonus
        if 2 <= paragraph_count <= 5:
            score += 0.1
        elif paragraph_count == 1:
            score -= 0.1

        # Length bonus
        length = len(content)
        if self.min_length <= length <= self.max_length:
            score += 0.1
        elif length < 500:
            score -= 0.1

        # Warning penalty
        score -= warning_count * 0.1

        return max(0.0, min(1.0, score))

    def get_optimization_summary(self, content: str) -> str:
        """
        Get a human-readable optimization summary.

        Args:
            content: The post to analyze

        Returns:
            Formatted summary string
        """
        analysis = self.analyze_post(content)

        lines = [f"LinkedIn Post Analysis: {analysis}"]
        lines.append(f"  Length: {analysis.original_length} characters")
        lines.append(f"  Paragraphs: {analysis.paragraph_count}")
        lines.append(f"  Hook: {'✓' if analysis.has_hook else '✗'}")
        lines.append(f"  CTA: {'✓' if analysis.has_cta else '✗'}")

        if analysis.warnings:
            lines.append("  Warnings:")
            for warning in analysis.warnings:
                lines.append(f"    ⚠ {warning}")

        if analysis.suggestions:
            lines.append("  Suggestions:")
            for suggestion in analysis.suggestions:
                lines.append(f"    → {suggestion}")

        return "\n".join(lines)


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for linkedin_optimizer module."""
    from test_framework import TestSuite

    suite = TestSuite("LinkedIn Optimizer Tests")

    def test_analyze_empty_post():
        optimizer = LinkedInOptimizer()
        analysis = optimizer.analyze_post("")
        assert analysis.original_length == 0
        assert analysis.readability_score == 0.0
        assert "Empty content" in analysis.warnings

    def test_analyze_short_post():
        optimizer = LinkedInOptimizer()
        analysis = optimizer.analyze_post("Hello world")
        assert analysis.original_length == 11
        assert len(analysis.suggestions) > 0  # Should suggest expanding

    def test_analyze_optimal_length_post():
        optimizer = LinkedInOptimizer()
        # Create a post in optimal range
        content = "This is a test. " * 80  # ~1280 characters
        analysis = optimizer.analyze_post(content)
        assert optimizer.min_length <= analysis.original_length <= optimizer.max_length

    def test_has_hook_question():
        optimizer = LinkedInOptimizer()
        content = "Why is hydrogen the future of energy?\n\nLet me explain..."
        analysis = optimizer.analyze_post(content)
        assert analysis.has_hook is True

    def test_has_hook_starter():
        optimizer = LinkedInOptimizer()
        content = "I just discovered something amazing about fuel cells.\n\nHere's what..."
        analysis = optimizer.analyze_post(content)
        assert analysis.has_hook is True

    def test_no_hook():
        optimizer = LinkedInOptimizer()
        # Use a very long first line (>100 chars) that doesn't match hook patterns
        content = (
            "The electrolyzer efficiency was measured at 85% during the testing "
            "phase and the results were documented in the laboratory records for analysis."
            "\n\nThis is the second paragraph."
        )
        analysis = optimizer.analyze_post(content)
        assert analysis.has_hook is False

    def test_has_cta():
        optimizer = LinkedInOptimizer()
        content = "Great news about hydrogen.\n\nWhat do you think?"
        analysis = optimizer.analyze_post(content)
        assert analysis.has_cta is True

    def test_no_cta():
        optimizer = LinkedInOptimizer()
        content = "Great news about hydrogen.\n\nThe end."
        analysis = optimizer.analyze_post(content)
        assert analysis.has_cta is False

    def test_detect_spam_hashtags():
        optimizer = LinkedInOptimizer()
        content = "Post #tag1 #tag2 #tag3 #tag4 #tag5 #tag6 #tag7"
        analysis = optimizer.analyze_post(content)
        assert any("hashtag" in w.lower() for w in analysis.warnings)

    def test_detect_spam_pattern():
        optimizer = LinkedInOptimizer()
        content = "Like if you agree! Share this post!"
        analysis = optimizer.analyze_post(content)
        assert len(analysis.warnings) > 0

    def test_optimize_post_formatting():
        optimizer = LinkedInOptimizer()
        content = "This is paragraph one.     This is still paragraph one.\n\nParagraph two."
        optimized, analysis = optimizer.optimize_post(content)
        # Should normalize whitespace
        assert "     " not in optimized

    def test_create_hook():
        optimizer = LinkedInOptimizer()
        hook = optimizer.create_hook("hydrogen")
        assert "hydrogen" in hook.lower()
        assert len(hook) > 10

    def test_create_cta():
        optimizer = LinkedInOptimizer()
        cta = optimizer.create_cta("fuel cells")
        assert "?" in cta or "." in cta
        assert len(cta) > 10

    def test_readability_score_bounds():
        optimizer = LinkedInOptimizer()
        # Score should always be between 0 and 1
        analysis = optimizer.analyze_post("Short post")
        assert 0.0 <= analysis.readability_score <= 1.0

        long_content = "Word " * 500
        analysis = optimizer.analyze_post(long_content)
        assert 0.0 <= analysis.readability_score <= 1.0

    def test_optimization_summary():
        optimizer = LinkedInOptimizer()
        content = "Why hydrogen matters.\n\nHere's the story.\n\nWhat do you think?"
        summary = optimizer.get_optimization_summary(content)
        assert "LinkedIn Post Analysis" in summary
        assert "Length" in summary
        assert "Hook" in summary

    suite.add_test("Analyze empty post", test_analyze_empty_post)
    suite.add_test("Analyze short post", test_analyze_short_post)
    suite.add_test("Analyze optimal length post", test_analyze_optimal_length_post)
    suite.add_test("Has hook - question", test_has_hook_question)
    suite.add_test("Has hook - starter", test_has_hook_starter)
    suite.add_test("No hook detection", test_no_hook)
    suite.add_test("Has CTA detection", test_has_cta)
    suite.add_test("No CTA detection", test_no_cta)
    suite.add_test("Detect spam hashtags", test_detect_spam_hashtags)
    suite.add_test("Detect spam pattern", test_detect_spam_pattern)
    suite.add_test("Optimize post formatting", test_optimize_post_formatting)
    suite.add_test("Create hook", test_create_hook)
    suite.add_test("Create CTA", test_create_cta)
    suite.add_test("Readability score bounds", test_readability_score_bounds)
    suite.add_test("Optimization summary", test_optimization_summary)

    return suite
