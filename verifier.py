"""Content verification using AI to ensure quality and appropriateness."""

import logging
import base64
from pathlib import Path

from google import genai  # type: ignore
from openai import OpenAI
from PIL import Image

from config import Config
from database import Database, Story

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

    def verify_pending_content(self) -> tuple[int, int]:
        """
        Verify all pending content.
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
            response = self.client.models.generate_content(
                model=Config.MODEL_VERIFICATION, contents=[prompt, image]
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
        response = self.client.models.generate_content(
            model=Config.MODEL_VERIFICATION, contents=prompt
        )
        if not response.text:
            logger.warning("Empty response from Gemini during text verification")
            return (False, "Empty response from API")
        return self._parse_verification_response(response.text)

    def _build_verification_prompt(self, story: Story) -> str:
        """Build the verification prompt for a story."""
        return f"""
You are a strict editorial standards board for a professional social media publication.

ORIGINAL SELECTION CRITERIA:
"{Config.SEARCH_PROMPT}"

STORY TO EVALUATE:
Title: {story.title}
Summary: {story.summary}
Sources: {", ".join(story.source_links[:3])}

EVALUATION CRITERIA:
1. RELEVANCE: Does this story genuinely match the original selection criteria?
2. PROFESSIONALISM: Is the content written in a professional, objective tone?
3. DECENCY: Is the content appropriate for all professional audiences?
4. ACCURACY: Does the summary appear factual and well-sourced?
5. QUALITY: Is this content worth sharing on a professional social media account?

If an image is provided, also evaluate:
6. IMAGE APPROPRIATENESS: Is the image professional and suitable for the story?
7. IMAGE RELEVANCE: Does the image relate to the story topic?

IMPORTANT IMAGE NOTES:
- All images are AI-generated, so "AI generated" watermarks/tags are ACCEPTABLE and should NOT cause rejection
- Focus on whether the image content is professional and relevant to the story

DECISION RULES:
- APPROVE only if ALL criteria are satisfactorily met
- REJECT if ANY criteria is not met
- When in doubt, REJECT

Respond with ONLY one of these exact words:
APPROVED
or
REJECTED

Then on a new line, provide a brief (one sentence) reason.
"""

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
def _create_module_tests():
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
