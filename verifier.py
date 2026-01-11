"""Content verification using AI to ensure quality and appropriateness."""

import logging
import base64
from pathlib import Path
from typing import Callable

from google import genai  # type: ignore
from openai import OpenAI
from PIL import Image

from config import Config
from database import Database, Story
from api_client import api_client
from originality_checker import OriginalityChecker
from source_verifier import SourceVerifier

logger = logging.getLogger(__name__)


class ContentVerifier:
    """Verify content quality and appropriateness before publishing."""

    def __init__(
        self,
        database: Database,
        client: genai.Client,
        local_client: OpenAI | None = None,
    ):
        """Initialize the content verifier."""
        self.db = database
        self.client = client
        self.local_client = local_client
        self.originality_checker = OriginalityChecker(
            client=client,
            local_client=local_client,
        )
        self.source_verifier = SourceVerifier()

    def verify_pending_content(
        self,
        linkedin_lookup_callback: Callable[[list[Story]], None] | None = None,
    ) -> tuple[int, int]:
        """
        Verify all pending content.

        Args:
            linkedin_lookup_callback: Optional callback function to run LinkedIn profile
                lookup when coverage is insufficient. Takes a list of stories to process.

        Returns tuple of (approved_count, rejected_count).
        """
        stories = self.db.get_stories_needing_verification()

        if not stories:
            logger.info("No stories pending verification")
            return (0, 0)

        total = len(stories)
        logger.info(f"Verifying {total} stories...")

        approved = 0
        rejected = 0
        skipped = 0

        for i, story in enumerate(stories, 1):
            try:
                logger.info(f"[{i}/{total}] Verifying: {story.title}")

                # Check if story has no image due to low quality score
                if not story.image_path:
                    if story.quality_score < Config.MIN_QUALITY_SCORE:
                        reason = (
                            f"Quality score ({story.quality_score}) below minimum "
                            f"threshold ({Config.MIN_QUALITY_SCORE}) - no image generated"
                        )
                        story.verification_status = "rejected"
                        story.verification_reason = reason
                        rejected += 1
                        logger.warning(f"  ✗ REJECTED: {reason}")
                        self.db.update_story(story)
                        continue
                    else:
                        # Quality is good but image missing - skip for now
                        reason = "No image available - generate images first"
                        logger.info(f"  ⏳ SKIPPED: {reason}")
                        skipped += 1
                        continue

                # Log LinkedIn profile status
                linkedin_valid, linkedin_msg = self._validate_linkedin_profiles(story)
                logger.info(f"  LinkedIn status: {linkedin_msg}")

                # If LinkedIn coverage insufficient and callback available, try to fix it
                if not linkedin_valid and linkedin_lookup_callback is not None:
                    logger.info(f"  🔄 Auto-running LinkedIn profile lookup...")
                    try:
                        linkedin_lookup_callback([story])
                        # Refresh the story from database after lookup
                        if story.id is not None:
                            refreshed_story = self.db.get_story(story.id)
                            if refreshed_story:
                                story = refreshed_story
                        # Re-validate LinkedIn profiles after lookup
                        linkedin_valid, linkedin_msg = self._validate_linkedin_profiles(
                            story
                        )
                        logger.info(f"  LinkedIn status (after lookup): {linkedin_msg}")
                    except Exception as e:
                        logger.warning(f"  LinkedIn lookup failed: {e}")

                # Reject stories without sufficient LinkedIn profile coverage
                if not linkedin_valid:
                    story.verification_status = "rejected"
                    story.verification_reason = linkedin_msg
                    rejected += 1
                    logger.warning(f"  ✗ REJECTED: {linkedin_msg}")
                    self.db.update_story(story)
                    continue

                is_approved, reason = self._verify_story(story)

                if is_approved:
                    story.verification_status = "approved"
                    story.verification_reason = reason
                    approved += 1
                    logger.info(f"  ✓ APPROVED: {reason}" if reason else "  ✓ APPROVED")
                else:
                    story.verification_status = "rejected"
                    story.verification_reason = reason
                    rejected += 1
                    logger.warning(
                        f"  ✗ REJECTED: {reason}" if reason else "  ✗ REJECTED"
                    )

                self.db.update_story(story)

            except KeyboardInterrupt:
                logger.warning(
                    f"\nVerification interrupted by user at story {i}/{total}"
                )
                break
            except Exception as e:
                logger.error(f"  ! Error: {e}")
                skipped += 1
                continue

        logger.info(
            f"Verification complete: {approved} approved, {rejected} rejected, {skipped} skipped"
        )
        return (approved, rejected)

    def _verify_story(self, story: Story) -> tuple[bool, str]:
        """
        Verify a single story for publication suitability.
        Returns tuple of (is_approved, reason).
        """
        # Check source credibility first (if enabled)
        if Config.SOURCE_VERIFICATION_ENABLED:
            try:
                source_result = self.source_verifier.verify_story_sources(story)
                if not source_result.is_verified:
                    reason = (
                        f"Source verification failed: {source_result.recommendation}"
                    )
                    logger.warning(f"  Source verification failed: {reason}")
                    return (False, reason)
                logger.debug(
                    f"  Source verification passed ({source_result.source_count} sources, "
                    f"avg credibility: {source_result.average_credibility:.0%})"
                )
            except Exception as e:
                logger.warning(f"  Source verification error (continuing): {e}")

        # Check originality (if enabled)
        if self.originality_checker.enabled:
            try:
                originality_result = self.originality_checker.check_story_originality(
                    story=story,
                )
                if not originality_result.is_original:
                    reason = (
                        f"Originality check failed: {originality_result.recommendation}"
                    )
                    if originality_result.flagged_phrases:
                        reason += f" Flagged: {'; '.join(originality_result.flagged_phrases[:3])}"
                    logger.warning(f"  Originality check failed: {reason}")
                    return (False, reason)
                logger.debug(
                    f"  Originality check passed (score: {originality_result.similarity_score:.2%})"
                )
            except Exception as e:
                logger.warning(f"  Originality check error (continuing): {e}")

        # Build verification prompt
        prompt = self._build_verification_prompt(story)

        try:
            # If story has an image, include it in verification
            if story.image_path and Path(story.image_path).exists():
                return self._verify_with_image(story, prompt)
            else:
                return self._verify_text_only(prompt)

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.error(
                    f"Verification failed: API quota exceeded (429 RESOURCE_EXHAUSTED)"
                )
                raise RuntimeError(
                    "API quota exceeded during verification. Please try again later."
                ) from e
            logger.error(f"Verification error for story {story.id}: {e}")
            # Default to rejection on error for safety
            return (False, f"Error during verification: {e}")

    def _verify_with_image(self, story: Story, prompt: str) -> tuple[bool, str]:
        """Verify story content with its associated image.
        Returns tuple of (is_approved, reason).
        """
        try:
            if not story.image_path:
                return self._verify_text_only(prompt)

            image_path = Path(story.image_path)
            if not image_path.exists():
                return self._verify_text_only(prompt)

            # Use local LLM if available
            if self.local_client:
                try:
                    logger.info("Using local LLM for image verification...")
                    with open(image_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode(
                            "utf-8"
                        )

                    response = self.local_client.chat.completions.create(
                        model=Config.LM_STUDIO_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                    )
                    content = response.choices[0].message.content
                    if content:
                        return self._parse_verification_response(content)
                except Exception as e:
                    logger.warning(
                        f"Local image verification failed: {e}. Falling back to Gemini."
                    )

            # Fallback to Gemini
            image = Image.open(str(image_path))
            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_VERIFICATION,
                contents=[prompt, image],
                endpoint="image_verify",
            )
            if not response.text:
                logger.warning("Empty response from Gemini during image verification")
                return (False, "Empty response from API")
            return self._parse_verification_response(response.text)

        except Exception as e:
            logger.warning(f"Image verification failed, falling back to text: {e}")
            return self._verify_text_only(prompt)

    def _verify_text_only(self, prompt: str) -> tuple[bool, str]:
        """Verify story content without image.
        Returns tuple of (is_approved, reason).
        """
        # Use local LLM if available
        if self.local_client:
            try:
                logger.info("Using local LLM for text verification...")
                response = self.local_client.chat.completions.create(
                    model=Config.LM_STUDIO_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content
                if content:
                    return self._parse_verification_response(content)
            except Exception as e:
                logger.warning(
                    f"Local text verification failed: {e}. Falling back to Gemini."
                )

        # Fallback to Gemini
        response = api_client.gemini_generate(
            client=self.client,
            model=Config.MODEL_VERIFICATION,
            contents=prompt,
            endpoint="text_verify",
        )
        if not response.text:
            logger.warning("Empty response from Gemini during text verification")
            return (False, "Empty response from API")
        return self._parse_verification_response(response.text)

    def _build_verification_prompt(self, story: Story) -> str:
        """Build the verification prompt for a story."""
        # Count people from story_people and org_leaders
        all_people = (story.story_people or []) + (story.org_leaders or [])
        people_count = len(all_people)
        linkedin_profiles_found = 0

        # Count LinkedIn profiles in story_people and org_leaders
        if all_people:
            linkedin_profiles_found = sum(
                1
                for p in all_people
                if (p.get("linkedin_urn") or p.get("linkedin_profile"))
                and str(p.get("linkedin_urn") or p.get("linkedin_profile", "")).strip()
            )

        # Get promotion message (or placeholder if not set)
        promotion_message = (
            story.promotion if story.promotion else "(No promotion message assigned)"
        )

        return Config.VERIFICATION_PROMPT.format(
            search_prompt=Config.SEARCH_PROMPT,
            story_title=story.title,
            story_summary=story.summary,
            story_sources=", ".join(story.source_links[:3]),
            people_count=people_count,
            linkedin_profiles_found=linkedin_profiles_found,
            summary_word_limit=Config.SUMMARY_WORD_COUNT,
            promotion_message=promotion_message,
        )

    def _validate_linkedin_profiles(self, story: Story) -> tuple[bool, str]:
        """
        Validate that LinkedIn profiles have been identified for the story.
        Returns (is_valid, message).

        Stories require:
        - At least 1 relevant person identified
        - At least 1 LinkedIn profile found (50% coverage minimum)
        """
        all_people = (story.story_people or []) + (story.org_leaders or [])
        relevant_count = len(all_people)
        linkedin_count = 0

        if all_people:
            linkedin_count = sum(
                1
                for p in all_people
                if (p.get("linkedin_urn") or p.get("linkedin_profile"))
                and str(p.get("linkedin_urn") or p.get("linkedin_profile", "")).strip()
            )

        # Validation rules for sufficient people and LinkedIn profile coverage
        if relevant_count == 0:
            return (
                False,
                "No relevant people identified - story needs associated individuals for engagement",
            )
        elif linkedin_count == 0:
            return (
                False,
                f"{relevant_count} people identified but no LinkedIn profiles found - run enrichment first",
            )
        elif linkedin_count < (relevant_count * 0.5):
            return (
                False,
                f"Insufficient LinkedIn coverage: {linkedin_count}/{relevant_count} profiles "
                f"({int(linkedin_count / relevant_count * 100)}%) - need at least 50%",
            )
        else:
            return (True, f"{linkedin_count}/{relevant_count} LinkedIn profiles found")

    def _parse_verification_response(self, response_text: str) -> tuple[bool, str]:
        """Parse the verification response to determine approval status.
        Returns tuple of (is_approved, reason).
        """
        lines = response_text.strip().split("\n")
        first_line = lines[0].strip().upper()

        # Extract reason from second line onwards
        reason = ""
        if len(lines) > 1:
            reason = " ".join(line.strip() for line in lines[1:] if line.strip())

        if "APPROVED" in first_line:
            return (True, reason)
        elif "REJECTED" in first_line:
            return (False, reason)
        else:
            # If unclear, default to rejection for safety
            logger.warning(f"Unclear verification response: {first_line}")
            return (False, f"Unclear response: {first_line}")

    def force_verify_story(
        self, story_id: int, approved: bool, reason: str = "Manually set by user"
    ) -> bool:
        """
        Manually force verification status for a story.
        Returns True if successful.
        """
        story = self.db.get_story(story_id)
        if not story:
            return False

        story.verification_status = "approved" if approved else "rejected"
        story.verification_reason = reason
        return self.db.update_story(story)

    def get_verification_stats(self) -> dict:
        """Get verification statistics."""
        stats = self.db.get_statistics()
        return {
            "pending": stats.get("pending_verification", 0),
            "available": stats.get("available_count", 0),
        }


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for verifier module."""
    import os
    import tempfile

    from test_framework import TestSuite

    suite = TestSuite("Verifier Tests")

    def test_parse_verification_response_approved():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            # Create minimal verifier without real clients
            verifier = ContentVerifier(db, None, None)  # type: ignore
            result, reason = verifier._parse_verification_response(
                "APPROVED\nGood content"
            )
            assert result is True
            assert "Good content" in reason
        finally:
            os.unlink(db_path)

    def test_parse_verification_response_rejected():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            verifier = ContentVerifier(db, None, None)  # type: ignore
            result, reason = verifier._parse_verification_response(
                "REJECTED\nNot suitable"
            )
            assert result is False
            assert "Not suitable" in reason
        finally:
            os.unlink(db_path)

    def test_parse_verification_response_unclear():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            verifier = ContentVerifier(db, None, None)  # type: ignore
            result, reason = verifier._parse_verification_response("Maybe")
            assert result is False  # Default to rejection
            assert "Unclear" in reason
        finally:
            os.unlink(db_path)

    def test_build_verification_prompt():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            verifier = ContentVerifier(db, None, None)  # type: ignore
            story = Story(
                title="Test Story",
                summary="Test summary about AI",
                quality_score=8,
            )
            prompt = verifier._build_verification_prompt(story)
            assert "Test Story" in prompt
            assert "Test summary" in prompt
            assert "APPROVED" in prompt
            assert "REJECTED" in prompt
        finally:
            os.unlink(db_path)

    def test_get_verification_stats():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            verifier = ContentVerifier(db, None, None)  # type: ignore
            stats = verifier.get_verification_stats()
            assert "pending" in stats
            assert "available" in stats
        finally:
            os.unlink(db_path)

    suite.add_test(
        "Parse response - approved", test_parse_verification_response_approved
    )
    suite.add_test(
        "Parse response - rejected", test_parse_verification_response_rejected
    )
    suite.add_test("Parse response - unclear", test_parse_verification_response_unclear)
    suite.add_test("Build verification prompt", test_build_verification_prompt)
    suite.add_test("Get verification stats", test_get_verification_stats)

    return suite
