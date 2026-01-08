"""Company mention enrichment using AI for LinkedIn posts.

This module identifies real companies explicitly mentioned in news sources and adds them
to posts as professional, analytical context. It is conservative by design - when in doubt,
it defaults to NO_COMPANY_MENTION.
"""

import logging

from google import genai  # type: ignore
from openai import OpenAI

from config import Config
from database import Database, Story

logger = logging.getLogger(__name__)

# Exact string that indicates no company mention should be added
NO_COMPANY_MENTION = "NO_COMPANY_MENTION"


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

    def enrich_pending_stories(self) -> tuple[int, int]:
        """
        Enrich all pending stories with company mentions.
        Returns tuple of (enriched_count, skipped_count).
        """
        stories = self.db.get_stories_needing_enrichment()

        if not stories:
            logger.info("No stories pending enrichment")
            return (0, 0)

        total = len(stories)
        logger.info(f"Enriching {total} stories with company mentions...")

        enriched = 0
        skipped = 0

        for i, story in enumerate(stories, 1):
            try:
                logger.info(f"[{i}/{total}] Enriching: {story.title}")

                mention, status = self._enrich_story(story)

                if status == "completed":
                    story.company_mention_enrichment = (
                        mention if mention != NO_COMPANY_MENTION else None
                    )
                    story.enrichment_status = "completed"
                    if mention and mention != NO_COMPANY_MENTION:
                        logger.info(f"  ✓ ENRICHED: {mention}")
                        enriched += 1
                    else:
                        logger.info("  ⏭ SKIPPED: No companies found")
                        skipped += 1
                else:
                    story.enrichment_status = "skipped"
                    skipped += 1
                    logger.info(f"  ⏳ SKIPPED: {mention}")

                self.db.update_story(story)

            except KeyboardInterrupt:
                logger.warning(f"\nEnrichment interrupted by user at story {i}/{total}")
                break
            except Exception as e:
                logger.error(f"  ! Error: {e}")
                story.enrichment_status = "skipped"
                self.db.update_story(story)
                skipped += 1
                continue

        logger.info(f"Enrichment complete: {enriched} enriched, {skipped} skipped")
        return (enriched, skipped)

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
        response = self.client.models.generate_content(
            model=Config.MODEL_TEXT, contents=prompt
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

        # If it's the no-mention marker, return it as-is
        if mention == NO_COMPANY_MENTION:
            return NO_COMPANY_MENTION

        # Validate that it's a single sentence
        # A valid sentence should:
        # 1. Not be empty
        # 2. End with a period (or be very short and professional)
        # 3. Not contain newlines (single sentence rule)
        # 4. Be reasonable length (max 250 chars)

        if not mention:
            return NO_COMPANY_MENTION

        if "\n" in mention:
            logger.warning(f"Invalid mention (contains newlines): {mention[:50]}")
            return NO_COMPANY_MENTION

        if len(mention) > 250:
            logger.warning(f"Invalid mention (too long): {mention[:50]}")
            return NO_COMPANY_MENTION

        # Ensure it ends with proper punctuation
        if not mention.endswith((".", "!", "?")):
            # If it doesn't end with punctuation, check if it looks like a sentence
            # Default to rejection if ambiguous
            logger.warning(f"Invalid mention (no ending punctuation): {mention}")
            return NO_COMPANY_MENTION

        # Check for multiple sentences (more than one . ! or ?)
        # Count sentence-ending punctuation
        sentence_count = mention.count(".") + mention.count("!") + mention.count("?")
        if sentence_count > 1:
            logger.warning(f"Invalid mention (multiple sentences): {mention[:50]}")
            return NO_COMPANY_MENTION

        # Check for list indicators (these shouldn't be in a single sentence)
        if any(indicator in mention for indicator in ["1.", "2.", "-", "•", "* "]):
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

    def get_enrichment_stats(self) -> dict:
        """Get enrichment statistics."""
        stats = self.db.get_statistics()
        return {
            "pending": stats.get("pending_enrichment", 0),
            "enriched": stats.get("enriched_count", 0),
        }

    def force_set_mention(self, story_id: int, mention: str | None) -> bool:
        """
        Manually set company mention for a story.
        Pass None or empty string to clear mention.
        Returns True if successful.
        """
        story = self.db.get_story(story_id)
        if not story:
            return False

        # Validate the mention if provided
        if mention:
            mention = self._validate_mention(mention)
            if mention == NO_COMPANY_MENTION:
                mention = None

        story.company_mention_enrichment = mention
        story.enrichment_status = "completed"
        return self.db.update_story(story)


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for company_mention_enricher module."""
    import os
    import tempfile

    from test_framework import TestSuite

    suite = TestSuite("Company Mention Enricher Tests")

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
            story = Story(
                title="BASF Innovation",
                summary="Development by BASF team",
                source_links=["https://example.com/basf"],
            )
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
            assert "enriched" in stats
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
