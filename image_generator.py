"""AI Image generation for story illustrations."""

import os
import time
import logging
from pathlib import Path

from google import genai
from google.genai import types

from config import Config
from database import Database, Story

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate images for stories using Google's Imagen model."""

    def __init__(self, database: Database, client: genai.Client):
        """Initialize the image generator."""
        self.db = database
        self.client = client
        self._ensure_image_directory()

    def _ensure_image_directory(self) -> None:
        """Ensure the image directory exists."""
        Path(Config.IMAGE_DIR).mkdir(parents=True, exist_ok=True)

    def generate_images_for_stories(self) -> int:
        """
        Generate images for stories that need them.
        Returns the number of images successfully generated.
        """
        # Get stories that need images
        stories = self.db.get_stories_needing_images(Config.MIN_QUALITY_SCORE)

        if not stories:
            logger.info("No stories need images")
            return 0

        logger.info(f"Generating images for {len(stories)} stories...")

        success_count = 0
        for story in stories:
            try:
                image_path = self._generate_image_for_story(story)
                if image_path:
                    story.image_path = image_path
                    self.db.update_story(story)
                    success_count += 1
                    logger.info(
                        f"Generated image for story ID {story.id}: {story.title}"
                    )
            except Exception as e:
                logger.error(f"Failed to generate image for story {story.id}: {e}")
                continue

        logger.info(f"Successfully generated {success_count}/{len(stories)} images")
        return success_count

    def _generate_image_for_story(self, story: Story) -> str | None:
        """Generate an image for a single story. Returns the file path if successful."""
        prompt = self._build_image_prompt(story)

        try:
            # Use Imagen model for image generation
            response = self.client.models.generate_images(
                model=Config.MODEL_IMAGE,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    safety_filter_level=types.SafetyFilterLevel.BLOCK_ONLY_HIGH,
                    person_generation=types.PersonGeneration.ALLOW_ADULT,
                    aspect_ratio="16:9",
                ),
            )

            if not response.generated_images or not response.generated_images[0].image:
                logger.warning(f"No image generated for story {story.id}")
                return None

            # Save the image
            filename = f"story_{story.id}_{int(time.time())}.png"
            filepath = os.path.join(Config.IMAGE_DIR, filename)

            image_data = response.generated_images[0].image.image_bytes
            if image_data:
                with open(filepath, "wb") as f:
                    f.write(image_data)
                logger.debug(f"Saved image to {filepath}")
                return filepath
            else:
                logger.warning(f"No image bytes for story {story.id}")
                return None

        except Exception as e:
            logger.error(f"Image generation error for story {story.id}: {e}")
            return None
            logger.error(f"Image generation error: {e}")
            return None

    def _generate_image_fallback(self, story: Story, prompt: str) -> str | None:
        """Fallback image generation using alternative API approach."""
        try:
            # Try using the Gemini multimodal model for image generation
            # This is a fallback if ImageGenerationModel is not available
            model = genai.GenerativeModel("gemini-2.0-flash-exp")  # type: ignore[attr-defined]

            response = model.generate_content(
                f"Generate a professional news illustration image for: {prompt}",
                generation_config={"response_mime_type": "image/png"},
            )

            if hasattr(response, "data") and response.data:  # type: ignore[union-attr]
                filename = f"story_{story.id}_{int(time.time())}.png"
                filepath = os.path.join(Config.IMAGE_DIR, filename)

                with open(filepath, "wb") as f:
                    f.write(response.data)  # type: ignore[union-attr]

                return filepath

            logger.warning("Fallback image generation did not return data")
            return None

        except Exception as e:
            logger.error(f"Fallback image generation failed: {e}")
            return None

    def _build_image_prompt(self, story: Story) -> str:
        """Build a prompt for image generation based on the story."""
        # Create a detailed prompt that will generate a relevant, professional image
        base_prompt = f"""
Create a photorealistic, professional news illustration for a story titled:
"{story.title}"

Context: {story.summary[:200]}...

Requirements:
- Modern, clean visual style suitable for professional social media
- No text or words in the image
- High quality, editorial photograph style
- Relevant to the story topic
- Appropriate lighting and composition
- Safe for work / professional audience
"""
        return base_prompt.strip()

    def get_stories_with_images_count(self) -> int:
        """Get count of stories that have images."""
        stories = self.db.get_stories_needing_verification()
        return len(stories)

    def cleanup_orphaned_images(self) -> int:
        """
        Remove images for stories that no longer exist.
        Returns count of files cleaned up.
        """
        image_dir = Path(Config.IMAGE_DIR)
        if not image_dir.exists():
            return 0

        cleaned = 0
        for image_file in image_dir.glob("story_*.png"):
            try:
                # Extract story ID from filename (format: story_{id}_{timestamp}.png)
                parts = image_file.stem.split("_")
                if len(parts) >= 2:
                    story_id = int(parts[1])
                    story = self.db.get_story(story_id)
                    if story is None:
                        image_file.unlink()
                        cleaned += 1
                        logger.debug(f"Cleaned up orphaned image: {image_file}")
            except (ValueError, IndexError):
                continue

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} orphaned images")
        return cleaned
