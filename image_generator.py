"""AI Image generation for story illustrations."""

import os
import time
import logging
from pathlib import Path
import requests

from google import genai
from google.genai import types
from openai import OpenAI

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
            # Provider selection diagnostics
            logger.debug(
                f"HF token present={bool(Config.HUGGINGFACE_API_TOKEN)}, prefer_hf={Config.HF_PREFER_IF_CONFIGURED}"
            )
            # If a Hugging Face token is configured and preferred, try it first
            if Config.HUGGINGFACE_API_TOKEN and Config.HF_PREFER_IF_CONFIGURED:
                hf_path = self._generate_huggingface_image(story, prompt)
                if hf_path:
                    return hf_path

            # Use Imagen model for image generation
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
        """Generate an image using Hugging Face Inference API."""
        if not Config.HUGGINGFACE_API_TOKEN:
            return None

        logger.info(
            f"Attempting Hugging Face image generation via model '{Config.HF_TTI_MODEL}'"
        )

        headers = {
            "Authorization": f"Bearer {Config.HUGGINGFACE_API_TOKEN}",
        }

        # Choose endpoint: custom inference endpoint or public models route
        if Config.HF_INFERENCE_ENDPOINT:
            url = Config.HF_INFERENCE_ENDPOINT.rstrip("/")
        else:
            url = f"https://api-inference.huggingface.co/models/{Config.HF_TTI_MODEL}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": Config.HF_NEGATIVE_PROMPT,
                # Aim for 16:9 while keeping sizes reasonable for free tiers
                "width": 1024,
                "height": 576,
            },
            "options": {"wait_for_model": True},
        }

        try:
            # Some models stream or return raw bytes; set stream=False for simplicity
            response = requests.post(url, headers=headers, json=payload, timeout=120)

            # Model cold-start: 503 while loading. Try a short backoff/poll.
            retries = 0
            while response.status_code == 503 and retries < 3:
                time.sleep(5 * (retries + 1))
                response = requests.post(
                    url, headers=headers, json=payload, timeout=120
                )
                retries += 1

            if response.status_code == 200 and response.headers.get(
                "content-type", ""
            ).startswith("image/"):
                filename = f"story_{story.id}_{int(time.time())}_hf.png"
                filepath = os.path.join(Config.IMAGE_DIR, filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                logger.info(f"Successfully generated HF image: {filepath}")
                return filepath

            # Some models may return JSON with an image in base64
            if response.headers.get("content-type", "").startswith(
                "application/json"
            ):
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        # common error message format
                        error_msg = data.get("error") or data.get("message")
                        if error_msg:
                            logger.warning(
                                f"Hugging Face API responded with error: {error_msg}"
                            )
                        # handle b64 image if present
                        b64 = data.get("image") or data.get("data")
                        if b64:
                            import base64

                            image_bytes = base64.b64decode(b64)
                            filename = (
                                f"story_{story.id}_{int(time.time())}_hf_b64.png"
                            )
                            filepath = os.path.join(Config.IMAGE_DIR, filename)
                            with open(filepath, "wb") as f:
                                f.write(image_bytes)
                            logger.info(
                                f"Successfully generated HF image (b64): {filepath}"
                            )
                            return filepath
                except Exception:
                    pass

            logger.warning(
                f"Hugging Face generation failed (status {response.status_code})"
            )
            return None
        except requests.RequestException as e:
            logger.warning(f"Hugging Face request failed: {e}")
            return None

    def _generate_image_fallback(self, story: Story, prompt: str) -> str | None:
        """Fallback image generation using alternative API approach."""
        try:
            # Try using the Gemini multimodal model for image generation
            # This is a fallback if ImageGenerationModel is not available
            model = genai.GenerativeModel("gemini-2.0-flash")  # type: ignore[attr-defined]

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
        sources_text = (
            "\n".join(story.source_links)
            if story.source_links
            else "No sources provided"
        )
        context = f"""
Story Title: {story.title}
Summary: {story.summary}
Sources:
{sources_text}
"""

        refinement_prompt = f"""
You are an expert AI image prompt engineer.
Based on the news story and sources below, create a detailed, high-quality prompt for an image generation model (like Imagen or Stable Diffusion).

STORY:
{context}

REQUIREMENTS for the prompt you generate:
- Focus on a single, powerful visual metaphor or scene.
- Style: Professional, editorial photography, high resolution, 8k.
- NO TEXT or labels in the image.
- Avoid people's faces if possible, focus on objects, environments, or symbolic representations.
- Lighting: Cinematic, professional.
- Aspect ratio: 16:9.
- Use the story title, summary, and source links to identify key visual elements or specific locations/technologies mentioned.

Respond with ONLY the refined prompt text.
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

        # Ultimate fallback
        return (
            f"Professional news illustration for: {story.title}. {story.summary[:100]}"
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
