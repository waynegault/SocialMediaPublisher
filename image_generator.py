"""AI Image generation for story illustrations."""

import os
import time
import logging
from pathlib import Path

from google import genai
from google.genai import types
from openai import OpenAI
from huggingface_hub import InferenceClient

from config import Config
from database import Database, Story

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate images for stories using Google's Imagen model."""

    def __init__(
        self,
        database: Database,
        client: genai.Client,
        local_client: OpenAI | None = None,
    ):
        """Initialize the image generator."""
        self.db = database
        self.client = client
        self.local_client = local_client
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
        for i, story in enumerate(stories):
            try:
                # Small delay between generations to be respectful of API rate limits
                if i > 0:
                    time.sleep(1)

                logger.info(
                    f"[{i + 1}/{len(stories)}] Generating image for: {story.title[:60]}..."
                )
                image_path = self._generate_image_for_story(story)
                if image_path:
                    story.image_path = image_path
                    self.db.update_story(story)
                    success_count += 1
                    logger.info(f"✓ Saved: {image_path}")
            except Exception as e:
                logger.error(f"✗ Failed to generate image for story {story.id}: {e}")
                continue

        logger.info(f"Successfully generated {success_count}/{len(stories)} images")
        return success_count

    def _generate_image_for_story(self, story: Story) -> str | None:
        """Generate an image for a single story. Returns the file path if successful."""
        prompt = self._build_image_prompt(story)

        try:
            # Provider selection diagnostics
            logger.debug(
                f"HF token present={bool(Config.HUGGINGFACE_API_TOKEN)}, prefer_hf={Config.HF_PREFER_IF_CONFIGURED}"
            )
            # If a Hugging Face token is configured and preferred, try it first
            if Config.HUGGINGFACE_API_TOKEN and Config.HF_PREFER_IF_CONFIGURED:
                hf_path = self._generate_huggingface_image(story, prompt)
                if hf_path:
                    return hf_path
                # HuggingFace failed, fall back to Imagen
                logger.info("HuggingFace unavailable, falling back to Google Imagen...")

            # Use Imagen model for image generation
            logger.info(f"Using Imagen model: {Config.MODEL_IMAGE}")
            response = self.client.models.generate_images(
                model=Config.MODEL_IMAGE,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    # API requires "block_low_and_above"
                    safety_filter_level=types.SafetyFilterLevel.BLOCK_LOW_AND_ABOVE,
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
            error_msg = str(e)
            if "billed users" in error_msg.lower():
                print("\n" + "-" * 60)
                print("Notice: Google's Imagen API requires a billed account.")
                if Config.HUGGINGFACE_API_TOKEN:
                    print("I'll try Hugging Face Inference instead...")
                    hf_path = self._generate_huggingface_image(story, prompt)
                    if hf_path:
                        return hf_path
                print("I'll try to use your local LLM/Image generator instead...")
                print("-" * 60 + "\n")

                # Try local fallback
                return self._generate_local_image(story, prompt)

            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                logger.error(
                    f"Image generation failed: API quota exceeded (429 RESOURCE_EXHAUSTED)"
                )
                raise RuntimeError(
                    "API quota exceeded during image generation. Please try again later."
                ) from e
            logger.error(f"Image generation error for story {story.id}: {e}")
            return None

    def _generate_huggingface_image(self, story: Story, prompt: str) -> str | None:
        """Generate an image using Hugging Face InferenceClient (FREE with FLUX.1-schnell)."""
        # Use the configured model or default to the free FLUX.1-schnell
        model = Config.HF_TTI_MODEL or "black-forest-labs/FLUX.1-schnell"

        logger.info(f"Attempting Hugging Face image generation via model '{model}'")

        try:
            # Initialize InferenceClient - works with or without token for free models
            if Config.HUGGINGFACE_API_TOKEN:
                client = InferenceClient(token=Config.HUGGINGFACE_API_TOKEN)
            else:
                client = InferenceClient()

            # Generate image using text_to_image - this is FREE for FLUX.1-schnell
            image = client.text_to_image(
                prompt=prompt,
                model=model,
            )

            # Save the image (returns a PIL Image object)
            filename = f"story_{story.id}_{int(time.time())}_hf.png"
            filepath = os.path.join(Config.IMAGE_DIR, filename)
            image.save(filepath)

            logger.info(f"Successfully generated HF image: {filepath}")
            return filepath

        except Exception as e:
            logger.warning(f"Hugging Face image generation failed: {e}")
            return None

    def _generate_local_image(self, story: Story, prompt: str) -> str | None:
        """Attempt to generate an image using the local OpenAI-compatible client."""
        if not self.local_client:
            logger.warning("No local client available for image generation")
            return None

        logger.info(f"Attempting local image generation for story {story.id}...")
        try:
            # Some local servers (like LocalAI or custom OpenAI proxies) support this
            response = self.local_client.images.generate(
                model=Config.LM_STUDIO_MODEL,
                prompt=prompt,
                n=1,
                size="1024x1024",
                response_format="b64_json",
            )

            if response.data and response.data[0].b64_json:
                import base64

                image_data = base64.b64decode(response.data[0].b64_json)

                filename = f"story_{story.id}_{int(time.time())}_local.png"
                filepath = os.path.join(Config.IMAGE_DIR, filename)

                with open(filepath, "wb") as f:
                    f.write(image_data)

                logger.info(f"Successfully generated local image: {filepath}")
                return filepath

        except Exception as e:
            logger.warning(f"Local image generation failed: {e}")
            print(
                f"Local image generation is not supported by your current local model ({Config.LM_STUDIO_MODEL})."
            )
            print(
                "Most local LLMs only support text. For local images, you would need a service like Stable Diffusion with an OpenAI-compatible API."
            )

        return None

    def _build_image_prompt(self, story: Story) -> str:
        """Build a prompt for image generation using an LLM for refinement."""
        context = f"""
Story Title: {story.title}
Summary: {story.summary}
"""

        # Get the configurable image style
        image_style = Config.IMAGE_STYLE

        # Professional prompt for industrial/engineering trade publications
        refinement_prompt = f"""
You are creating an image for a professional chemical engineering trade publication (like Chemical Engineering Magazine or AIChE publications).

STORY CONTEXT:
{context}

YOUR TASK: Create an image prompt for a REALISTIC, PROFESSIONAL photograph that would appear in an engineering trade journal.

CRITICAL REQUIREMENTS - THE IMAGE MUST BE:
1. PHOTOREALISTIC - like a real photograph, NOT artistic, NOT fantasy, NOT stylized
2. PROFESSIONAL - suitable for a serious engineering publication
3. TECHNICALLY ACCURATE - showing real equipment, processes, or concepts correctly
4. CREDIBLE - something a chemical engineer would recognize as realistic

SUBJECT SELECTION (choose the most appropriate):
- Industrial equipment: reactors, distillation columns, heat exchangers, piping systems, control rooms
- Laboratory settings: analytical instruments, lab glassware, researchers in lab coats
- Manufacturing facilities: chemical plants, refineries, pharmaceutical production
- Process technology: flow diagrams visualized as real equipment, process units
- Materials and products: chemicals, polymers, catalysts, finished products
- Data/monitoring: control panels, SCADA screens, process monitoring (if story is about digitalization)

WHAT TO AVOID:
- Fantasy or sci-fi elements
- Artistic interpretations or abstract concepts
- Glowing/magical effects
- Futuristic imaginary technology
- Cartoonish or illustrated styles
- Anything that would look silly to a practicing engineer

STYLE REQUIREMENTS:
{image_style}

PHOTOGRAPHY SPECS:
- Professional industrial photography style
- Clean, well-lit scenes (industrial facility lighting or natural daylight)
- Sharp focus, high resolution
- Neutral, realistic colors
- Documentary/journalistic aesthetic

OUTPUT: Write ONLY the image prompt. No explanations. Maximum 100 words.
Format: "[Specific industrial subject], [realistic setting], professional industrial photograph, photorealistic, sharp focus, natural lighting"
"""

        try:
            if self.local_client:
                logger.info("Using local LLM to refine image prompt...")
                response = self.local_client.chat.completions.create(
                    model=Config.LM_STUDIO_MODEL,
                    messages=[{"role": "user", "content": refinement_prompt}],
                )
                refined = response.choices[0].message.content
                if refined:
                    return refined.strip()

            # Fallback to Gemini for refinement
            logger.info("Using Gemini to refine image prompt...")
            response = self.client.models.generate_content(
                model=Config.MODEL_TEXT, contents=refinement_prompt
            )
            if response.text:
                return response.text.strip()

        except Exception as e:
            logger.warning(f"Prompt refinement failed: {e}. Using base prompt.")

        # Ultimate fallback - professional industrial photography prompt
        return (
            f"Professional industrial photograph for chemical engineering publication: "
            f"{story.title[:60]}. Industrial facility or laboratory setting, "
            f"photorealistic, documentary style, natural lighting, sharp focus, "
            f"neutral colors, suitable for engineering trade journal"
        )

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
