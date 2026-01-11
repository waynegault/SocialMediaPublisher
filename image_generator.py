"""AI Image generation for story illustrations."""

import os
import time
import logging
import random
from pathlib import Path

from google import genai
from google.genai import types
from openai import OpenAI
from huggingface_hub import InferenceClient  # type: ignore
from PIL import Image, ImageDraw, ImageFont

from api_client import api_client
from config import Config
from database import Database, Story
from rate_limiter import AdaptiveRateLimiter

# Appearance variations for generated images - ensures diverse women
HAIR_COLORS = [
    "platinum blonde",
    "honey blonde",
    "strawberry blonde",
    "chestnut brown",
    "dark brown",
    "jet black",
    "copper red",
    "fiery redhead",
]
HAIR_STYLES = [
    "long flowing hair",
    "shoulder-length wavy hair",
    "sleek ponytail",
    "elegant updo",
    "loose curls",
    "professional bob",
    "spikey on top and shaved at the back and side",
]
ETHNICITIES = [
    "Caucasian",
    "East Asian",
    "South Asian",
    "Latina",
    "Middle Eastern",
    "African American",
    "mixed heritage",
]
BODY_DESCRIPTORS = [
    "slim and curvaceous",
    "slender with an athletic figure",
    "petite and shapely",
    "large chested",
]

logger = logging.getLogger(__name__)


def get_random_appearance() -> str:
    """Generate a random appearance description for image variety."""
    hair_color = random.choice(HAIR_COLORS)
    hair_style = random.choice(HAIR_STYLES)
    ethnicity = random.choice(ETHNICITIES)
    body_type = random.choice(BODY_DESCRIPTORS)

    return (
        f"a gorgeous highly attractive {ethnicity} woman with {hair_color} {hair_style}, "
        f"{body_type}, with striking beautiful features and a confident radiant smile"
    )


def add_ai_watermark(image: Image.Image) -> Image.Image:
    """Add 'AI generated image' watermark to bottom right corner of image."""
    # Convert to RGBA to support transparency, then back to RGB for saving
    img = image.convert("RGBA")

    watermark_text = "AI generated image"

    # Try to use a reasonable font size based on image size
    font_size = max(14, min(img.width, img.height) // 35)

    try:
        # Try to use a system font (Windows)
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            # Fallback fonts for Linux
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except (OSError, IOError):
            # Ultimate fallback to default font (smaller but works)
            font = ImageFont.load_default()

    # Create a transparent overlay for the watermark
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Get text bounding box
    bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position at bottom right with padding
    padding = 15
    x = img.width - text_width - padding
    y = img.height - text_height - padding

    # Draw semi-transparent background rectangle on overlay
    bg_padding = 6
    draw.rectangle(
        [
            x - bg_padding,
            y - bg_padding,
            x + text_width + bg_padding,
            y + text_height + bg_padding,
        ],
        fill=(0, 0, 0, 160),  # Semi-transparent black
    )

    # Draw the text in white on overlay
    draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, 255))

    # Composite the overlay onto the image
    result = Image.alpha_composite(img, overlay)

    # Convert back to RGB for saving as PNG/JPEG
    return result.convert("RGB")


class ImageGenerator:
    """Generate images for stories using Google's Imagen model."""

    # Shared rate limiter for Imagen API (conservative limits for image generation)
    _rate_limiter = AdaptiveRateLimiter(
        initial_fill_rate=0.5,  # Start at 1 request per 2 seconds
        min_fill_rate=0.1,  # Minimum: 1 request per 10 seconds
        max_fill_rate=2.0,  # Maximum: 2 requests per second
        success_threshold=3,  # Increase rate after 3 successes
        rate_limiter_429_backoff=0.5,  # Aggressive backoff on 429: halve the rate
    )

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
                # Use adaptive rate limiter to manage request timing
                wait_time = self._rate_limiter.wait(endpoint="imagen")
                if wait_time > 0.5:
                    logger.info(f"Rate limiter: waited {wait_time:.1f}s before request")

                logger.info(
                    f"[{i + 1}/{len(stories)}] Generating image for: {story.title}"
                )
                result = self._generate_image_for_story_with_retry(story)
                if result:
                    image_path, alt_text = result
                    story.image_path = image_path
                    story.image_alt_text = alt_text
                    # Reset verification status so the new image gets verified
                    story.verification_status = "pending"
                    story.verification_reason = None
                    self.db.update_story(story)
                    success_count += 1
                    self._rate_limiter.on_success(endpoint="imagen")
                    logger.info(f"✓ Saved: {image_path}")
            except Exception as e:
                error_msg = str(e)
                if (
                    "429" in error_msg
                    or "RESOURCE_EXHAUSTED" in error_msg
                    or "quota" in error_msg.lower()
                ):
                    # Parse retry-after header if available (usually in the error message)
                    retry_after = self._parse_retry_after(error_msg)
                    self._rate_limiter.on_429_error(
                        endpoint="imagen", retry_after=retry_after
                    )
                logger.error(f"✗ Failed to generate image for story {story.id}: {e}")
                continue

        # Log rate limiter metrics
        metrics = self._rate_limiter.get_metrics()
        logger.info(
            f"Rate limiter stats: {metrics.total_requests} requests, "
            f"{metrics.error_429_count} 429s, current rate: {metrics.current_fill_rate:.2f} req/s"
        )

        logger.info(f"Successfully generated {success_count}/{len(stories)} images")
        return success_count

    def _parse_retry_after(self, error_msg: str) -> float:
        """Extract retry-after value from error message if present."""
        import re

        # Look for patterns like "retry after 30 seconds" or "retry-after: 30"
        match = re.search(r"retry[- ]?after[:\s]+(\d+)", error_msg.lower())
        if match:
            return float(match.group(1))
        # Default penalty for 429: 30 seconds
        return 30.0

    def _generate_image_for_story_with_retry(
        self, story: Story, max_retries: int = 3
    ) -> tuple[str, str] | None:
        """Generate an image with retry logic for rate limits."""
        for attempt in range(max_retries):
            try:
                result = self._generate_image_for_story(story)
                if result:
                    return result
                # If result is None but no exception, don't retry
                return None
            except RuntimeError as e:
                error_msg = str(e)
                if "quota" in error_msg.lower() or "429" in error_msg:
                    if attempt < max_retries - 1:
                        # Notify rate limiter and let it handle the backoff timing
                        retry_after = self._parse_retry_after(error_msg)
                        self._rate_limiter.on_429_error(
                            endpoint="imagen", retry_after=retry_after
                        )
                        # Wait according to rate limiter before retry
                        wait_time = self._rate_limiter.wait(endpoint="imagen")
                        logger.warning(
                            f"Rate limited, waited {wait_time:.1f}s for retry {attempt + 2}/{max_retries}..."
                        )
                        continue
                raise
            except Exception:
                # Don't retry other errors
                raise
        return None

    def _generate_image_for_story(self, story: Story) -> tuple[str, str] | None:
        """Generate an image for a single story. Returns (file_path, alt_text) if successful."""
        prompt = self._build_image_prompt(story)

        try:
            # Provider selection diagnostics
            logger.debug(
                f"HF token present={bool(Config.HUGGINGFACE_API_TOKEN)}, prefer_hf={Config.HF_PREFER_IF_CONFIGURED}"
            )
            # If a Hugging Face token is configured and preferred, try it first
            if Config.HUGGINGFACE_API_TOKEN and Config.HF_PREFER_IF_CONFIGURED:
                hf_result = self._generate_huggingface_image(story, prompt)
                if hf_result:
                    image_path, _ = hf_result
                    alt_text = self._generate_alt_text(story, prompt)
                    return (image_path, alt_text)
                # HuggingFace failed, fall back to Imagen
                logger.info("HuggingFace unavailable, falling back to Google Imagen...")

            # Use Imagen model for image generation
            logger.info(f"Using Imagen model: {Config.MODEL_IMAGE}")
            logger.debug(f"Prompt ({len(prompt.split())} words): {prompt[:200]}...")
            response = api_client.imagen_generate(
                client=self.client,
                model=Config.MODEL_IMAGE,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,  # pyright: ignore[reportCallIssue]
                    # API requires "block_low_and_above"
                    safety_filter_level=types.SafetyFilterLevel.BLOCK_LOW_AND_ABOVE,  # pyright: ignore[reportCallIssue]
                    person_generation=types.PersonGeneration.ALLOW_ADULT,  # pyright: ignore[reportCallIssue]
                    aspect_ratio=Config.IMAGE_ASPECT_RATIO,  # pyright: ignore[reportCallIssue]
                    image_size=Config.IMAGE_SIZE,  # pyright: ignore[reportCallIssue] - "2K" for higher quality
                    # Note: negative_prompt is NOT supported by Imagen API (only works with HuggingFace)
                ),
            )

            if not response.generated_images or not response.generated_images[0].image:
                logger.warning(f"No image generated for story {story.id}")
                return None

            # Get image data and apply watermark
            image_data = response.generated_images[0].image.image_bytes
            if image_data:
                from io import BytesIO

                # Load image from bytes, add watermark, and save
                image = Image.open(BytesIO(image_data))
                watermarked_image = add_ai_watermark(image)

                filename = f"story_{story.id}_{int(time.time())}.png"
                filepath = os.path.join(Config.IMAGE_DIR, filename)
                watermarked_image.save(filepath)

                # Verify the file was actually written
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    logger.info(f"Verified image saved: {filepath} ({file_size} bytes)")
                else:
                    logger.error(f"Image file NOT found after save: {filepath}")
                    return None

                logger.debug(f"Saved watermarked image to {filepath}")
                alt_text = self._generate_alt_text(story, prompt)
                return (filepath, alt_text)
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
                    hf_result = self._generate_huggingface_image(story, prompt)
                    if hf_result:
                        image_path, _ = hf_result
                        alt_text = self._generate_alt_text(story, prompt)
                        return (image_path, alt_text)
                print("I'll try to use your local LLM/Image generator instead...")
                print("-" * 60 + "\n")

                # Try local fallback
                local_result = self._generate_local_image(story, prompt)
                if local_result:
                    image_path, _ = local_result
                    alt_text = self._generate_alt_text(story, prompt)
                    return (image_path, alt_text)
                return None

            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                logger.error(
                    "Image generation failed: API quota exceeded (429 RESOURCE_EXHAUSTED)"
                )
                raise RuntimeError(
                    "API quota exceeded during image generation. Please try again later."
                ) from e
            logger.error(f"Image generation error for story {story.id}: {e}")
            return None

    def _generate_huggingface_image(
        self, story: Story, prompt: str
    ) -> tuple[str, str] | None:
        """Generate an image using Hugging Face InferenceClient (FREE with FLUX.1-schnell).
        Returns (file_path, prompt_used) if successful, or None.
        """
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
            # Note: negative_prompt is ignored by FLUX models but works with SD/SDXL
            # Build kwargs with proper typing to avoid Pylance errors
            if Config.HF_NEGATIVE_PROMPT and "flux" not in model.lower():
                logger.debug(
                    f"Using negative prompt: {Config.HF_NEGATIVE_PROMPT[:50]}..."
                )
                image = client.text_to_image(
                    prompt=prompt,
                    model=model,
                    negative_prompt=Config.HF_NEGATIVE_PROMPT,
                )
            else:
                image = client.text_to_image(prompt=prompt, model=model)

            # Add AI watermark to the image
            watermarked_image = add_ai_watermark(image)

            # Save the image (returns a PIL Image object)
            filename = f"story_{story.id}_{int(time.time())}_hf.png"
            filepath = os.path.join(Config.IMAGE_DIR, filename)
            watermarked_image.save(filepath)

            # Verify the file was actually written
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                logger.info(f"Verified HF image saved: {filepath} ({file_size} bytes)")
            else:
                logger.error(f"HF image file NOT found after save: {filepath}")
                return None

            logger.info(f"Successfully generated HF image: {filepath}")
            return (filepath, prompt)

        except Exception as e:
            logger.warning(f"Hugging Face image generation failed: {e}")
            return None

    def _generate_local_image(
        self, story: Story, prompt: str
    ) -> tuple[str, str] | None:
        """Attempt to generate an image using the local OpenAI-compatible client.
        Returns (file_path, prompt_used) if successful, or None.
        """
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
                from io import BytesIO

                image_data = base64.b64decode(response.data[0].b64_json)

                # Load image, add watermark, and save
                image = Image.open(BytesIO(image_data))
                watermarked_image = add_ai_watermark(image)

                filename = f"story_{story.id}_{int(time.time())}_local.png"
                filepath = os.path.join(Config.IMAGE_DIR, filename)
                watermarked_image.save(filepath)

                logger.info(f"Successfully generated local image: {filepath}")
                return (filepath, prompt)

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
        # Generate random appearance for this image to ensure variety
        random_appearance = get_random_appearance()

        # Build refinement prompt from config template with random appearance injected
        refinement_prompt = Config.IMAGE_REFINEMENT_PROMPT.format(
            story_title=story.title,
            story_summary=story.summary,
            image_style=Config.IMAGE_STYLE,
        )

        # Inject the specific appearance to use for this image
        appearance_instruction = f"""
MANDATORY APPEARANCE FOR THIS IMAGE (use exactly as specified):
The female engineer in this image must be: {random_appearance}.
Do NOT deviate from this appearance description. Include these exact physical traits in your prompt.
"""
        refinement_prompt = appearance_instruction + "\n" + refinement_prompt

        try:
            refined = None
            if self.local_client:
                logger.info(
                    f"Using local LLM to refine image prompt (appearance: {random_appearance[:50]}...)"
                )
                response = self.local_client.chat.completions.create(
                    model=Config.LM_STUDIO_MODEL,
                    messages=[{"role": "user", "content": refinement_prompt}],
                )
                refined = response.choices[0].message.content
            else:
                # Fallback to Gemini for refinement
                logger.info(
                    f"Using Gemini to refine image prompt (appearance: {random_appearance[:50]}...)"
                )
                response = api_client.gemini_generate(
                    client=self.client,
                    model=Config.MODEL_TEXT,
                    contents=refinement_prompt,
                    endpoint="prompt_refinement",
                )
                refined = response.text

            if refined:
                refined = self._clean_image_prompt(refined.strip())
                return refined

        except Exception as e:
            logger.warning(f"Prompt refinement failed: {e}. Using base prompt.")

        # Ultimate fallback - use configurable fallback template with random appearance
        fallback = Config.IMAGE_FALLBACK_PROMPT.format(story_title=story.title[:60])
        # Inject the random appearance into fallback
        fallback = fallback.replace(
            "a beautiful female chemical engineer",
            f"{random_appearance} as a chemical engineer",
        )
        return fallback

    def _clean_image_prompt(self, prompt: str) -> str:
        """Clean and validate the image prompt for optimal Imagen generation."""
        # Remove markdown formatting (quotes, backticks, etc.)
        prompt = prompt.strip().strip('"').strip("'").strip("`")
        # Remove markdown code blocks if present
        if prompt.startswith("```"):
            lines = prompt.split("\n")
            prompt = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        # Ensure prompt starts with "A photo of" for photorealistic output
        lower_prompt = prompt.lower()
        if not lower_prompt.startswith("a photo of"):
            # Try to fix common variations
            if lower_prompt.startswith("photo of"):
                prompt = "A " + prompt
            elif lower_prompt.startswith("photograph of"):
                prompt = "A p" + prompt[1:]  # Keep original casing but fix
            else:
                # Prepend "A photo of" if missing
                prompt = f"A photo of {prompt[0].lower()}{prompt[1:]}"

        logger.debug(f"Cleaned prompt: {prompt[:100]}...")
        return prompt

    def _generate_alt_text(self, story: Story, image_prompt: str) -> str:
        """Generate accessibility alt text for an image based on the story and image prompt.

        Creates a concise, descriptive alt text suitable for screen readers.
        """
        alt_text_prompt = f"""Generate a concise alt text description (1-2 sentences, max 150 characters) for an image.

The image was generated for this story:
Title: {story.title}
Summary: {story.summary[:200]}...

The image depicts: {image_prompt[:300]}

Write alt text that:
1. Describes what's shown in the image for visually impaired users
2. Is concise but informative (aim for 80-150 characters)
3. Focuses on the main subject and setting
4. Does NOT start with "Image of" or "Picture of"
5. Does NOT include hashtags or promotional content

Return ONLY the alt text, nothing else."""

        try:
            alt_text = None
            if self.local_client:
                logger.debug("Using local LLM to generate alt text")
                response = self.local_client.chat.completions.create(
                    model=Config.LM_STUDIO_MODEL,
                    messages=[{"role": "user", "content": alt_text_prompt}],
                )
                alt_text = response.choices[0].message.content
            else:
                logger.debug("Using Gemini to generate alt text")
                response = api_client.gemini_generate(
                    client=self.client,
                    model=Config.MODEL_TEXT,
                    contents=alt_text_prompt,
                    endpoint="alt_text",
                )
                alt_text = response.text

            if alt_text:
                # Clean up the alt text
                alt_text = alt_text.strip().strip('"').strip("'")
                # Truncate if too long
                if len(alt_text) > 200:
                    alt_text = alt_text[:197] + "..."
                logger.info(f"Generated alt text: {alt_text[:60]}...")
                return alt_text

        except Exception as e:
            logger.warning(f"Alt text generation failed: {e}")

        # Fallback alt text based on story title
        fallback = f"Illustration for: {story.title[:100]}"
        logger.info(f"Using fallback alt text: {fallback}")
        return fallback

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


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for image_generator module."""
    import os
    import tempfile

    from test_framework import TestSuite
    from PIL import Image

    suite = TestSuite("Image Generator Tests")

    def test_add_ai_watermark():
        # Create a simple test image
        img = Image.new("RGB", (200, 200), color="white")
        result = add_ai_watermark(img)
        assert result is not None
        assert result.size == (200, 200)
        assert result.mode == "RGB"

    def test_add_ai_watermark_small_image():
        # Test with very small image
        img = Image.new("RGB", (50, 50), color="blue")
        result = add_ai_watermark(img)
        assert result is not None
        assert result.size == (50, 50)

    def test_image_generator_init():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            gen = ImageGenerator(db, None, None)  # type: ignore
            assert gen.db is db
            assert os.path.isdir(Config.IMAGE_DIR)
        finally:
            os.unlink(db_path)

    def test_build_image_prompt_fallback():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            gen = ImageGenerator(db, None, None)  # type: ignore
            story = Story(
                title="AI in Healthcare",
                summary="AI is revolutionizing medical diagnostics",
                quality_score=8,
            )
            # Without a client, should fall back to simple prompt
            prompt = gen._build_image_prompt(story)
            assert "AI in Healthcare" in prompt
            assert len(prompt) > 0
        finally:
            os.unlink(db_path)

    def test_ensure_image_directory():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            gen = ImageGenerator(db, None, None)  # type: ignore
            gen._ensure_image_directory()
            assert os.path.isdir(Config.IMAGE_DIR)
        finally:
            os.unlink(db_path)

    def test_generate_huggingface_image_no_token():
        """Test HuggingFace image generation gracefully fails without token."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            gen = ImageGenerator(db, None, None)  # type: ignore
            story = Story(
                id=999,
                title="Test Story",
                summary="Test summary",
                quality_score=8,
            )
            # This should handle the case gracefully (return None or raise)
            # We're testing that it doesn't crash unexpectedly
            try:
                result = gen._generate_huggingface_image(story, "test prompt")
                # If it returns, it should be None (no token) or a tuple (path, prompt)
                # Note: FLUX.1-schnell is free and works without token
                assert result is None or isinstance(result, tuple)
            except Exception as e:
                # Expected if HuggingFace is not available
                assert (
                    "API" in str(e)
                    or "token" in str(e).lower()
                    or "connection" in str(e).lower()
                )
        finally:
            os.unlink(db_path)

    def test_generate_local_image_no_client():
        """Test local image generation gracefully fails without client."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            gen = ImageGenerator(db, None, None)  # type: ignore - no local_client
            story = Story(
                id=998,
                title="Test Story",
                summary="Test summary",
                quality_score=8,
            )
            result = gen._generate_local_image(story, "test prompt")
            # Should return None when no local client is available
            assert result is None
        finally:
            os.unlink(db_path)

    def test_cleanup_orphaned_images():
        """Test orphaned image cleanup."""
        # Use a temp directory to avoid deleting real images
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db_path = f.name
            try:
                db = Database(db_path)
                gen = ImageGenerator(db, None, None)  # type: ignore
                # Mock the image directory to use temp dir instead of real Config.IMAGE_DIR
                original_image_dir = Config.IMAGE_DIR
                Config.IMAGE_DIR = temp_dir
                try:
                    # Should not crash even with no orphaned images
                    cleaned = gen.cleanup_orphaned_images()
                    assert isinstance(cleaned, int)
                    assert cleaned >= 0
                finally:
                    Config.IMAGE_DIR = original_image_dir
            finally:
                os.unlink(db_path)

    def test_get_stories_with_images_count():
        """Test stories with images count."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            gen = ImageGenerator(db, None, None)  # type: ignore
            count = gen.get_stories_with_images_count()
            assert isinstance(count, int)
            assert count >= 0
        finally:
            os.unlink(db_path)

    def test_clean_image_prompt_already_starts_with_photo():
        """Test prompt cleaning when already starts with 'A photo of'."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            gen = ImageGenerator(db, None, None)  # type: ignore
            prompt = "A photo of an industrial plant with equipment"
            result = gen._clean_image_prompt(prompt)
            assert result.lower().startswith("a photo of")
            assert "industrial plant" in result
        finally:
            os.unlink(db_path)

    def test_clean_image_prompt_missing_prefix():
        """Test prompt cleaning when missing 'A photo of' prefix."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            gen = ImageGenerator(db, None, None)  # type: ignore
            prompt = "Industrial plant with equipment"
            result = gen._clean_image_prompt(prompt)
            assert result.lower().startswith("a photo of")
            assert "industrial plant" in result.lower()
        finally:
            os.unlink(db_path)

    def test_clean_image_prompt_removes_quotes():
        """Test prompt cleaning removes markdown quotes."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            gen = ImageGenerator(db, None, None)  # type: ignore
            prompt = '"A photo of an industrial plant"'
            result = gen._clean_image_prompt(prompt)
            assert result.lower().startswith("a photo of")
            assert not result.startswith('"')
            assert not result.endswith('"')
        finally:
            os.unlink(db_path)

    suite.add_test("Add AI watermark", test_add_ai_watermark)
    suite.add_test("Add AI watermark small image", test_add_ai_watermark_small_image)
    suite.add_test("Image generator init", test_image_generator_init)
    suite.add_test("Build image prompt fallback", test_build_image_prompt_fallback)
    suite.add_test("Ensure image directory", test_ensure_image_directory)
    suite.add_test(
        "HuggingFace image - no token", test_generate_huggingface_image_no_token
    )
    suite.add_test("Local image - no client", test_generate_local_image_no_client)
    suite.add_test("Cleanup orphaned images", test_cleanup_orphaned_images)
    suite.add_test("Get stories with images count", test_get_stories_with_images_count)
    suite.add_test(
        "Clean prompt - already has prefix",
        test_clean_image_prompt_already_starts_with_photo,
    )
    suite.add_test(
        "Clean prompt - missing prefix", test_clean_image_prompt_missing_prefix
    )
    suite.add_test(
        "Clean prompt - removes quotes", test_clean_image_prompt_removes_quotes
    )

    return suite
