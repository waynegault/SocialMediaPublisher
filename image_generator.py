"""AI Image generation for story illustrations.

TASK 6.3: Multi-Model Image Generation
- DALL-E 3 as alternative generator
- Model selection based on prompt type
- A/B testing support for image models
- Auto-fallback on model failures
"""

import os
import time
import logging
import random
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
from google import genai
from google.genai import types
from openai import OpenAI
from huggingface_hub import InferenceClient  # type: ignore
from PIL import Image, ImageDraw, ImageFont

from api_client import api_client
from config import Config
from database import Database, Story
from rate_limiter import AdaptiveRateLimiter


# =============================================================================
# Multi-Model Configuration (TASK 6.3)
# =============================================================================


class ImageModel(Enum):
    """Available image generation models."""

    GOOGLE_IMAGEN = "google_imagen"
    OPENAI_DALLE3 = "openai_dalle3"
    HUGGINGFACE_FLUX = "huggingface_flux"
    HUGGINGFACE_SDXL = "huggingface_sdxl"
    LOCAL = "local"


@dataclass
class ModelConfig:
    """Configuration for an image model."""

    model_id: ImageModel
    display_name: str
    priority: int  # Lower = higher priority
    enabled: bool = True
    requires_api_key: str = ""  # Config attribute name for API key
    best_for: Optional[list[str]] = None  # Content types this model excels at

    def __post_init__(self) -> None:
        if self.best_for is None:
            self.best_for = []


# Model configurations with priority and capabilities
MODEL_CONFIGS: dict[ImageModel, ModelConfig] = {
    ImageModel.GOOGLE_IMAGEN: ModelConfig(
        model_id=ImageModel.GOOGLE_IMAGEN,
        display_name="Google Imagen",
        priority=1,
        requires_api_key="GEMINI_API_KEY",
        best_for=["photorealistic", "people", "engineering", "industrial"],
    ),
    ImageModel.OPENAI_DALLE3: ModelConfig(
        model_id=ImageModel.OPENAI_DALLE3,
        display_name="DALL-E 3",
        priority=2,
        requires_api_key="OPENAI_API_KEY",
        best_for=["artistic", "creative", "abstract", "conceptual"],
    ),
    ImageModel.HUGGINGFACE_FLUX: ModelConfig(
        model_id=ImageModel.HUGGINGFACE_FLUX,
        display_name="HuggingFace FLUX",
        priority=3,
        requires_api_key="",  # Works without key for free models
        best_for=["general", "fast"],
    ),
    ImageModel.HUGGINGFACE_SDXL: ModelConfig(
        model_id=ImageModel.HUGGINGFACE_SDXL,
        display_name="HuggingFace SDXL",
        priority=4,
        requires_api_key="HUGGINGFACE_API_TOKEN",
        best_for=["artistic", "stylized"],
    ),
    ImageModel.LOCAL: ModelConfig(
        model_id=ImageModel.LOCAL,
        display_name="Local Model",
        priority=5,
        best_for=["testing", "offline"],
    ),
}


def select_model_for_prompt(prompt: str, story_category: str = "") -> ImageModel:
    """Select the best model based on prompt content and story category.

    Args:
        prompt: The image generation prompt
        story_category: Category of the story

    Returns:
        Best model for this prompt
    """
    prompt_lower = prompt.lower()
    category_lower = story_category.lower()

    # Keywords that suggest artistic/creative content
    artistic_keywords = {
        "abstract",
        "artistic",
        "creative",
        "conceptual",
        "symbolic",
        "metaphor",
    }

    # Keywords that suggest photorealistic/industrial content
    industrial_keywords = {
        "engineering",
        "industrial",
        "factory",
        "plant",
        "equipment",
        "reactor",
        "pipeline",
        "process",
        "laboratory",
        "research",
    }

    # Check prompt content
    has_artistic = any(kw in prompt_lower for kw in artistic_keywords)
    has_industrial = any(
        kw in prompt_lower or kw in category_lower for kw in industrial_keywords
    )

    # Determine best model based on content
    if has_artistic and not has_industrial:
        # Prefer DALL-E 3 for artistic content
        if _is_model_available(ImageModel.OPENAI_DALLE3):
            return ImageModel.OPENAI_DALLE3

    if has_industrial or "Technology" in story_category or "Research" in story_category:
        # Prefer Imagen for industrial/engineering content
        if _is_model_available(ImageModel.GOOGLE_IMAGEN):
            return ImageModel.GOOGLE_IMAGEN

    # Fall back to priority order
    return get_best_available_model()


def _is_model_available(model: ImageModel) -> bool:
    """Check if a model is available based on API key configuration."""
    config = MODEL_CONFIGS.get(model)
    if not config or not config.enabled:
        return False

    if config.requires_api_key:
        api_key = getattr(Config, config.requires_api_key, None)
        return bool(api_key)

    return True


def get_best_available_model() -> ImageModel:
    """Get the best available model based on priority and configuration."""
    available = [
        (model, config)
        for model, config in MODEL_CONFIGS.items()
        if config.enabled and _is_model_available(model)
    ]

    if not available:
        # Fallback to HuggingFace (works without API key)
        return ImageModel.HUGGINGFACE_FLUX

    # Sort by priority
    available.sort(key=lambda x: x[1].priority)
    return available[0][0]


def get_available_models() -> list[ImageModel]:
    """Get list of all available models."""
    return [model for model in ImageModel if _is_model_available(model)]


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
        f"a highly attractive {ethnicity} woman with model level good looks with {hair_color} {hair_style}, "
        f"{body_type}, with striking beautiful features, large shapely bust and a confident radiant smile"
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

            # Use Imagen model for image generation with retry on safety filter blocks
            logger.info(f"Using Imagen model: {Config.MODEL_IMAGE}")
            logger.debug(f"Prompt ({len(prompt.split())} words): {prompt[:200]}...")

            max_retries = 3
            response = None
            for attempt in range(max_retries):
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

                if response.generated_images and response.generated_images[0].image:
                    break  # Success, exit retry loop

                if attempt < max_retries - 1:
                    logger.warning(
                        f"No image generated for story {story.id} (attempt {attempt + 1}/{max_retries}) - "
                        f"Google's safety filters may have blocked. Retrying..."
                    )
                    time.sleep(1)  # Brief pause before retry
                else:
                    logger.warning(
                        f"No image generated for story {story.id} after {max_retries} attempts - "
                        f"Google's safety filters blocked the request. "
                        f"Prompt was: {prompt[:150]}..."
                    )
                    return None

            # Get image data and apply watermark
            if response is None or not response.generated_images:
                return None
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

    def _generate_dalle3_image(
        self, story: Story, prompt: str
    ) -> tuple[str, str] | None:
        """Generate an image using OpenAI's DALL-E 3.

        DALL-E 3 excels at artistic, creative, and conceptual imagery.
        Returns (file_path, prompt_used) if successful, or None.
        """
        # Check for OpenAI API key
        openai_key = getattr(Config, "OPENAI_API_KEY", None)
        if not openai_key:
            logger.debug("No OPENAI_API_KEY configured for DALL-E 3")
            return None

        logger.info(f"Attempting DALL-E 3 image generation for story {story.id}...")

        try:
            # Create OpenAI client for DALL-E
            dalle_client = OpenAI(api_key=openai_key)

            # DALL-E 3 API call
            response = dalle_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",  # DALL-E 3 supports 1024x1024, 1024x1792, 1792x1024
                quality="standard",  # "standard" or "hd"
                style="natural",  # "natural" or "vivid"
            )

            if response.data and response.data[0].url:
                import httpx
                from io import BytesIO

                # Download the image
                image_url = response.data[0].url
                image_response = httpx.get(image_url, timeout=30.0)
                image_data = image_response.content

                # Load image, add watermark, and save
                image = Image.open(BytesIO(image_data))
                watermarked_image = add_ai_watermark(image)

                filename = f"story_{story.id}_{int(time.time())}_dalle3.png"
                filepath = os.path.join(Config.IMAGE_DIR, filename)
                watermarked_image.save(filepath)

                # Verify the file was actually written
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    logger.info(
                        f"Verified DALL-E 3 image saved: {filepath} ({file_size} bytes)"
                    )
                else:
                    logger.error(
                        f"DALL-E 3 image file NOT found after save: {filepath}"
                    )
                    return None

                # DALL-E 3 may revise the prompt - log if different
                revised_prompt = response.data[0].revised_prompt
                if revised_prompt and revised_prompt != prompt:
                    logger.info(f"DALL-E 3 revised prompt: {revised_prompt[:100]}...")

                logger.info(f"Successfully generated DALL-E 3 image: {filepath}")
                return (filepath, prompt)

        except Exception as e:
            logger.warning(f"DALL-E 3 image generation failed: {e}")

        return None

    def generate_with_model(
        self,
        story: Story,
        model: ImageModel,
        prompt: Optional[str] = None,
    ) -> tuple[str, str] | None:
        """Generate an image using a specific model.

        Args:
            story: Story to generate image for
            model: Specific model to use
            prompt: Optional custom prompt (generates from story if not provided)

        Returns:
            Tuple of (file_path, alt_text) if successful
        """
        if prompt is None:
            prompt = self._build_image_prompt(story)

        result = None

        if model == ImageModel.GOOGLE_IMAGEN:
            result = self._generate_image_for_story(story)
        elif model == ImageModel.OPENAI_DALLE3:
            result = self._generate_dalle3_image(story, prompt)
            if result:
                image_path, _ = result
                alt_text = self._generate_alt_text(story, prompt)
                return (image_path, alt_text)
        elif model == ImageModel.HUGGINGFACE_FLUX:
            result = self._generate_huggingface_image(story, prompt)
            if result:
                image_path, _ = result
                alt_text = self._generate_alt_text(story, prompt)
                return (image_path, alt_text)
        elif model == ImageModel.HUGGINGFACE_SDXL:
            # Use SDXL model variant
            original_model = Config.HF_TTI_MODEL
            Config.HF_TTI_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
            result = self._generate_huggingface_image(story, prompt)
            Config.HF_TTI_MODEL = original_model
            if result:
                image_path, _ = result
                alt_text = self._generate_alt_text(story, prompt)
                return (image_path, alt_text)
        elif model == ImageModel.LOCAL:
            result = self._generate_local_image(story, prompt)
            if result:
                image_path, _ = result
                alt_text = self._generate_alt_text(story, prompt)
                return (image_path, alt_text)

        return result

    def generate_with_auto_fallback(
        self,
        story: Story,
        prompt: Optional[str] = None,
    ) -> tuple[str, str, ImageModel] | None:
        """Generate an image with automatic model selection and fallback.

        Tries models in priority order until one succeeds.

        Args:
            story: Story to generate image for
            prompt: Optional custom prompt

        Returns:
            Tuple of (file_path, alt_text, model_used) if successful
        """
        if prompt is None:
            prompt = self._build_image_prompt(story)

        # Select best model for this prompt
        preferred_model = select_model_for_prompt(prompt, story.category)
        models_to_try = [preferred_model]

        # Add other available models as fallbacks
        for model in get_available_models():
            if model not in models_to_try:
                models_to_try.append(model)

        for model in models_to_try:
            logger.info(
                f"Trying image generation with {MODEL_CONFIGS[model].display_name}..."
            )

            try:
                result = self.generate_with_model(story, model, prompt)
                if result:
                    image_path, alt_text = result
                    logger.info(
                        f"Successfully generated image with {MODEL_CONFIGS[model].display_name}"
                    )
                    return (image_path, alt_text, model)

            except Exception as e:
                logger.warning(f"{MODEL_CONFIGS[model].display_name} failed: {e}")
                continue

        logger.error("All image generation models failed")
        return None

    def _analyze_source_image(self, image_url: str) -> str:
        """Use multimodal AI to analyze a source article image and describe what it shows.

        This provides rich visual context for generating more accurate images.
        """
        try:
            # Download the image
            response = httpx.get(
                image_url,
                timeout=10.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )

            if response.status_code != 200:
                return ""

            # Check content type
            content_type = response.headers.get("content-type", "")
            if not any(
                img_type in content_type.lower()
                for img_type in ["image/jpeg", "image/png", "image/webp", "image/gif"]
            ):
                return ""

            # Check image size (skip tiny images)
            if len(response.content) < 5000:  # Skip images < 5KB (likely icons)
                return ""

            # Load image with PIL
            from io import BytesIO

            img = Image.open(BytesIO(response.content))

            # Skip small images (icons, buttons)
            if img.width < 200 or img.height < 200:
                return ""

            # Use Gemini to analyze the image
            analysis_prompt = """Analyze this image from a technical/engineering news article.
Describe in 2-3 sentences:
1. What specific equipment, technology, or facility is shown (be precise - e.g., "PEM electrolyzer stack", "distillation column", "CRISPR gene editing setup")
2. The setting/environment (laboratory, industrial plant, pilot facility, control room, etc.)
3. Any distinctive visual elements (colors, scale, materials, instrumentation)

Focus on technical accuracy. Be specific about equipment types, not generic.
If this appears to be a stock photo or generic image, say "GENERIC STOCK IMAGE".
Keep response under 100 words."""

            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_TEXT,
                contents=[analysis_prompt, img],
                endpoint="image_analysis",
            )

            if response and response.text:
                analysis = response.text.strip()
                # Skip if it's identified as generic
                if "GENERIC STOCK IMAGE" in analysis.upper():
                    return ""
                logger.info(f"Analyzed source image: {analysis[:100]}...")
                return analysis

        except Exception as e:
            logger.debug(f"Failed to analyze source image {image_url}: {e}")

        return ""

    def _extract_technical_terms(self, text: str) -> list[str]:
        """Extract specific technical/engineering terms from article text."""
        # Chemical engineering and industrial equipment patterns
        equipment_patterns = [
            r"\b(?:PEM|alkaline|SOEC)\s+electrolyz(?:er|is)",
            r"\b(?:CSTR|PFR|batch)\s+reactor\b",
            r"\b(?:packed|tray|bubble.cap)\s+(?:column|tower)\b",
            r"\b(?:membrane|RO|UF|NF)\s+(?:system|unit|module)\b",
            r"\b(?:heat|shell.and.tube|plate)\s+exchanger\b",
            r"\b(?:centrifugal|positive.displacement|peristaltic)\s+pump\b",
            r"\b(?:fluidized|fixed)\s+bed\s+(?:reactor|system)\b",
            r"\b(?:SCADA|DCS|PLC)\s+(?:system|control)\b",
            r"\b(?:GC|HPLC|mass.spec|NMR|IR)\s+(?:system|analysis)\b",
            r"\b(?:bioreactor|fermenter|bioprocessing)\b",
            r"\b(?:crystalliz|evaporat|distill|extract|absorb|adsorb)(?:er|or|ion)\b",
            r"\b(?:compressor|turbine|generator|motor)\b",
            r"\b(?:catalyst|catalytic)\s+(?:bed|system|reactor)\b",
            r"\b(?:storage|pressure|cryogenic)\s+(?:tank|vessel)\b",
            r"\b(?:pilot|demo|commercial)\s+(?:plant|facility|scale)\b",
            r"\b(?:control|monitor)(?:ing)?\s+(?:panel|room|system)\b",
            r"\b(?:safety|relief|check|control)\s+valve\b",
            r"\b(?:carbon|CO2)\s+capture\b",
            r"\b(?:hydrogen|ammonia|methanol)\s+(?:production|synthesis|plant)\b",
            r"\b(?:solar|wind|battery|fuel.cell)\s+(?:system|array|plant)\b",
        ]

        terms = []
        text_lower = text.lower()

        for pattern in equipment_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            terms.extend(matches)

        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            term_clean = term.strip().lower()
            if term_clean not in seen:
                seen.add(term_clean)
                unique_terms.append(term)

        return unique_terms[:5]  # Return top 5 terms

    def _fetch_source_content(self, story: Story) -> dict:
        """Fetch content and images from source URLs to inform image generation.

        Returns a dict with:
            - 'text': Extracted article text (first 2000 chars)
            - 'images': List of image URLs found in the article
            - 'image_descriptions': Descriptions from alt text or captions
            - 'ai_image_analysis': AI-generated descriptions of source images
            - 'technical_terms': Extracted equipment/technology terms
        """
        result = {
            "text": "",
            "images": [],
            "image_descriptions": [],
            "ai_image_analysis": [],
            "technical_terms": [],
        }

        if not story.source_links:
            return result

        # Try to fetch from first source link
        source_url = story.source_links[0] if story.source_links else None
        if not source_url:
            return result

        try:
            logger.debug(f"Fetching source content from: {source_url}")
            response = httpx.get(
                source_url,
                timeout=15.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )

            if response.status_code != 200:
                logger.debug(f"Source fetch failed with status {response.status_code}")
                return result

            html = response.text
            base_url = source_url.rsplit("/", 1)[0]  # For resolving relative URLs

            # Extract main text content (simple extraction without BeautifulSoup)
            # Remove script and style tags
            html_clean = re.sub(
                r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
            )
            html_clean = re.sub(
                r"<style[^>]*>.*?</style>",
                "",
                html_clean,
                flags=re.DOTALL | re.IGNORECASE,
            )

            # Extract text from paragraph tags
            paragraphs = re.findall(
                r"<p[^>]*>(.*?)</p>", html_clean, flags=re.DOTALL | re.IGNORECASE
            )
            text_content = " ".join(paragraphs)
            # Strip HTML tags from paragraphs
            text_content = re.sub(r"<[^>]+>", " ", text_content)
            # Clean up whitespace
            text_content = re.sub(r"\s+", " ", text_content).strip()
            result["text"] = text_content[:2000]  # Limit to 2000 chars

            # Extract technical terms from text
            result["technical_terms"] = self._extract_technical_terms(text_content)
            if result["technical_terms"]:
                logger.info(f"Extracted technical terms: {result['technical_terms']}")

            # Extract image URLs and their alt text
            img_pattern = r'<img[^>]+src=["\']([^"\'>]+)["\'][^>]*>'
            img_matches = re.findall(img_pattern, html, flags=re.IGNORECASE)

            # Filter for likely content images (not icons, logos, etc.)
            content_images = []
            for img_url in img_matches[:10]:  # Check first 10 images
                # Skip tiny images, icons, tracking pixels
                if any(
                    skip in img_url.lower()
                    for skip in [
                        "icon",
                        "logo",
                        "pixel",
                        "tracking",
                        "avatar",
                        "1x1",
                        "spacer",
                        "thumbnail",
                        "social",
                        "share",
                        "button",
                    ]
                ):
                    continue

                # Resolve relative URLs
                if img_url.startswith("//"):
                    img_url = "https:" + img_url
                elif img_url.startswith("/"):
                    # Extract domain from source_url
                    from urllib.parse import urlparse

                    parsed = urlparse(source_url)
                    img_url = f"{parsed.scheme}://{parsed.netloc}{img_url}"
                elif not img_url.startswith("http"):
                    img_url = f"{base_url}/{img_url}"

                content_images.append(img_url)
                if len(content_images) >= 3:  # Limit to 3 content images
                    break

            result["images"] = content_images

            # Extract alt text from images
            alt_pattern = r'<img[^>]+alt=["\']([^"\'>]+)["\'][^>]*>'
            alt_matches = re.findall(alt_pattern, html, flags=re.IGNORECASE)
            result["image_descriptions"] = [
                alt for alt in alt_matches if len(alt) > 10
            ][:3]

            # Use AI to analyze actual source images (limit to first 2 for speed)
            for img_url in content_images[:2]:
                analysis = self._analyze_source_image(img_url)
                if analysis:
                    result["ai_image_analysis"].append(analysis)

            logger.info(
                f"Extracted {len(result['text'])} chars, {len(result['images'])} images, "
                f"{len(result['ai_image_analysis'])} AI analyses from source"
            )

        except Exception as e:
            logger.debug(f"Failed to fetch source content: {e}")

        return result

    def _build_image_prompt(self, story: Story) -> str:
        """Build a prompt for image generation using an LLM for refinement."""
        # Generate random appearance for this image to ensure variety
        random_appearance = get_random_appearance()

        # Fetch source content for richer context
        source_context = self._fetch_source_content(story)

        # Build comprehensive source context section
        source_section = ""
        has_context = (
            source_context["text"]
            or source_context["image_descriptions"]
            or source_context["ai_image_analysis"]
            or source_context["technical_terms"]
        )

        if has_context:
            source_section = "\n\n=== CRITICAL: SOURCE ARTICLE VISUAL CONTEXT ===\n"
            source_section += "Use this information to create a SPECIFIC image that matches the actual story, NOT a generic industrial scene.\n\n"

            # AI analysis of actual source images (highest priority)
            if source_context["ai_image_analysis"]:
                source_section += (
                    "ACTUAL IMAGES FROM THE SOURCE ARTICLE (replicate these visuals):\n"
                )
                for i, analysis in enumerate(source_context["ai_image_analysis"], 1):
                    source_section += f"  Image {i}: {analysis}\n"
                source_section += "\n"

            # Extracted technical terms
            if source_context["technical_terms"]:
                source_section += f"SPECIFIC EQUIPMENT/TECHNOLOGY MENTIONED: {', '.join(source_context['technical_terms'])}\n"
                source_section += "Your image MUST show this specific equipment, not generic machinery.\n\n"

            # Alt text descriptions
            if source_context["image_descriptions"]:
                source_section += f"IMAGE CAPTIONS/ALT TEXT: {'; '.join(source_context['image_descriptions'])}\n\n"

            # Article text excerpt
            if source_context["text"]:
                source_section += (
                    f"ARTICLE CONTEXT: {source_context['text'][:1200]}\n\n"
                )

            source_section += "=== END SOURCE CONTEXT ===\n"
            source_section += "Your image prompt MUST incorporate the specific visual elements described above.\n"

        # Build refinement prompt from config template with random appearance injected
        refinement_prompt = Config.IMAGE_REFINEMENT_PROMPT.format(
            story_title=story.title,
            story_summary=story.summary,
            image_style=Config.IMAGE_STYLE,
            discipline=Config.DISCIPLINE,
        )

        # Inject the specific appearance to use for this image
        appearance_instruction = f"""
MANDATORY APPEARANCE FOR THIS IMAGE (use exactly as specified):
    The female {Config.DISCIPLINE} professional in this image must be: {random_appearance}.
Do NOT deviate from this appearance description. Include these exact physical traits in your prompt.
"""
        # Combine appearance instruction, source context, and refinement prompt
        refinement_prompt = (
            appearance_instruction + source_section + "\n" + refinement_prompt
        )

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
        fallback_template = Config.IMAGE_FALLBACK_PROMPT
        fallback = fallback_template.format(
            story_title=story.title[:60],
            discipline=Config.DISCIPLINE,
            appearance=random_appearance,
        )
        # If the template does not expose {appearance}, replace a generic phrase
        if "{appearance}" not in fallback_template:
            fallback = fallback.replace("a beautiful female", random_appearance, 1)
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

    def test_fetch_source_content_no_links():
        """Test fetch_source_content returns empty dict when no source links."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            gen = ImageGenerator(db, None, None)  # type: ignore
            story = Story(
                id=101,
                title="Test Story",
                summary="Test summary",
                quality_score=8,
                source_links=[],  # Empty source links
            )
            result = gen._fetch_source_content(story)
            assert isinstance(result, dict)
            assert result["text"] == ""
            assert result["images"] == []
            assert result["image_descriptions"] == []
        finally:
            os.unlink(db_path)

    def test_fetch_source_content_invalid_url():
        """Test fetch_source_content handles invalid URLs gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            gen = ImageGenerator(db, None, None)  # type: ignore
            story = Story(
                id=102,
                title="Test Story",
                summary="Test summary",
                quality_score=8,
                source_links=["https://invalid.nonexistent.url.fake/article"],
            )
            result = gen._fetch_source_content(story)
            # Should return empty dict on failure, not crash
            assert isinstance(result, dict)
            assert "text" in result
            assert "images" in result
            assert "image_descriptions" in result
        finally:
            os.unlink(db_path)

    def test_fetch_source_content_structure():
        """Test fetch_source_content returns correct structure."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            gen = ImageGenerator(db, None, None)  # type: ignore
            story = Story(
                id=103,
                title="Test Story",
                summary="Test summary",
                quality_score=8,
            )
            result = gen._fetch_source_content(story)
            # Verify structure even when no source links
            assert "text" in result
            assert "images" in result
            assert "image_descriptions" in result
            assert "ai_image_analysis" in result
            assert "technical_terms" in result
            assert isinstance(result["text"], str)
            assert isinstance(result["images"], list)
            assert isinstance(result["image_descriptions"], list)
            assert isinstance(result["ai_image_analysis"], list)
            assert isinstance(result["technical_terms"], list)
        finally:
            os.unlink(db_path)

    def test_extract_technical_terms():
        """Test extraction of technical engineering terms from text."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            gen = ImageGenerator(db, None, None)  # type: ignore

            # Test with chemical engineering text
            text = """The new PEM electrolyzer system achieves 95% efficiency.
            The hydrogen production facility uses a membrane reactor and
            includes carbon capture technology. The pilot plant features
            a fluidized bed reactor for catalyst regeneration."""

            terms = gen._extract_technical_terms(text)
            assert isinstance(terms, list)
            assert len(terms) > 0
            # Should find electrolyzer, reactor, carbon capture, etc.
            combined = " ".join(terms).lower()
            assert any(
                keyword in combined
                for keyword in ["electrolyz", "reactor", "carbon capture", "membrane"]
            )
        finally:
            os.unlink(db_path)

    def test_extract_technical_terms_empty():
        """Test extraction with no technical terms."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            gen = ImageGenerator(db, None, None)  # type: ignore

            text = "This is a simple sentence with no technical equipment."
            terms = gen._extract_technical_terms(text)
            assert isinstance(terms, list)
            # May be empty or have few terms
            assert len(terms) <= 5
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
    suite.add_test(
        "Fetch source content - no links", test_fetch_source_content_no_links
    )
    suite.add_test(
        "Fetch source content - invalid URL", test_fetch_source_content_invalid_url
    )
    suite.add_test(
        "Fetch source content - structure", test_fetch_source_content_structure
    )
    suite.add_test("Extract technical terms", test_extract_technical_terms)
    suite.add_test(
        "Extract technical terms - empty", test_extract_technical_terms_empty
    )

    return suite
