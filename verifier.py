"""Content verification using AI to ensure quality and appropriateness."""

import logging
from pathlib import Path

from google import genai  # type: ignore
from PIL import Image

from config import Config
from database import Database, Story

logger = logging.getLogger(__name__)


class ContentVerifier:
    """Verify content quality and appropriateness before publishing."""

    def __init__(self, database: Database, client: genai.Client):
        """Initialize the content verifier."""
        self.db = database
        self.client = client

    def verify_pending_content(self) -> tuple[int, int]:
        """
        Verify all pending content.
        Returns tuple of (approved_count, rejected_count).
        """
        stories = self.db.get_stories_needing_verification()

        if not stories:
            logger.info("No stories pending verification")
            return (0, 0)

        logger.info(f"Verifying {len(stories)} stories...")

        approved = 0
        rejected = 0

        for story in stories:
            try:
                is_approved = self._verify_story(story)

                if is_approved:
                    story.verification_status = "approved"
                    approved += 1
                    logger.info(f"Story {story.id} APPROVED: {story.title}")
                else:
                    story.verification_status = "rejected"
                    rejected += 1
                    logger.warning(f"Story {story.id} REJECTED: {story.title}")

                self.db.update_story(story)

            except Exception as e:
                logger.error(f"Verification failed for story {story.id}: {e}")
                continue

        logger.info(f"Verification complete: {approved} approved, {rejected} rejected")
        return (approved, rejected)

    def _verify_story(self, story: Story) -> bool:
        """
        Verify a single story for publication suitability.
        Returns True if approved, False if rejected.
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
            logger.error(f"Verification error for story {story.id}: {e}")
            # Default to rejection on error for safety
            return False

    def _verify_with_image(self, story: Story, prompt: str) -> bool:
        """Verify story content with its associated image."""
        try:
            image = Image.open(str(story.image_path))

            response = self.client.models.generate_content(
                model=Config.MODEL_VERIFICATION, contents=[prompt, image]
            )
            return self._parse_verification_response(response.text)

        except Exception as e:
            logger.warning(f"Image verification failed, falling back to text: {e}")
            return self._verify_text_only(prompt)

    def _verify_text_only(self, prompt: str) -> bool:
        """Verify story content without image."""
        response = self.client.models.generate_content(
            model=Config.MODEL_VERIFICATION, contents=prompt
        )
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

    def _parse_verification_response(self, response_text: str) -> bool:
        """Parse the verification response to determine approval status."""
        response = response_text.strip().upper()

        # Check first line for decision
        first_line = response.split("\n")[0].strip()

        if "APPROVED" in first_line:
            return True
        elif "REJECTED" in first_line:
            return False
        else:
            # If unclear, default to rejection for safety
            logger.warning(f"Unclear verification response: {first_line}")
            return False

    def force_verify_story(self, story_id: int, approved: bool) -> bool:
        """
        Manually force verification status for a story.
        Returns True if successful.
        """
        story = self.db.get_story(story_id)
        if not story:
            return False

        story.verification_status = "approved" if approved else "rejected"
        return self.db.update_story(story)

    def get_verification_stats(self) -> dict:
        """Get verification statistics."""
        stats = self.db.get_statistics()
        return {
            "pending": stats.get("pending_verification", 0),
            "available": stats.get("available_count", 0),
        }
