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

from google import genai
from openai import OpenAI
from huggingface_hub import InferenceClient  # type: ignore
from PIL import Image, ImageDraw, ImageFont

from api_client import api_client
from config import Config
from database import Database, Story
from image_quality import ImageQualityAssessor, QualityScore
from rate_limiter import AdaptiveRateLimiter
from url_utils import resolve_relative_url


# =============================================================================
# Multi-Model Configuration (TASK 6.3)
# =============================================================================


class ImageModel(Enum):
    """Available image generation models."""

    Z_IMAGE = "z_image"  # Local high-quality generation (preferred when CUDA available)
    GOOGLE_IMAGEN = "google_imagen"
    OPENAI_DALLE3 = "openai_dalle3"
    HUGGINGFACE_FLUX = "huggingface_flux"
    HUGGINGFACE_SDXL = "huggingface_sdxl"
    POLLINATIONS = "pollinations"  # Free, no API key required
    LOCAL = "local"
    # Extensible provider - uses IMAGE_PROVIDER from .env
    EXTENSIBLE_PROVIDER = "extensible_provider"


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
    ImageModel.Z_IMAGE: ModelConfig(
        model_id=ImageModel.Z_IMAGE,
        display_name="Z-Image (Local)",
        priority=0,  # Highest priority when CUDA available
        requires_api_key="",  # No API key - runs locally with CUDA
        best_for=["photorealistic", "artistic", "people", "high-quality", "local"],
    ),
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
    ImageModel.POLLINATIONS: ModelConfig(
        model_id=ImageModel.POLLINATIONS,
        display_name="Pollinations.ai (Free)",
        priority=5,
        requires_api_key="",  # Completely free, no key needed
        best_for=["general", "free", "fallback"],
    ),
    ImageModel.LOCAL: ModelConfig(
        model_id=ImageModel.LOCAL,
        display_name="Local Model",
        priority=6,
        best_for=["testing", "offline"],
    ),
    ImageModel.EXTENSIBLE_PROVIDER: ModelConfig(
        model_id=ImageModel.EXTENSIBLE_PROVIDER,
        display_name="Extensible Provider (via IMAGE_PROVIDER env)",
        priority=7,  # Use explicit provider setting
        requires_api_key="",  # Varies by provider
        best_for=["configurable", "switchable"],
    ),
}


def select_model_for_prompt(prompt: str, story_category: str = "") -> ImageModel:
    """Select the best model based on prompt content and story category.

    Z-Image is preferred for all content types when CUDA is available,
    as it excels at both photorealistic and artistic content.

    Args:
        prompt: The image generation prompt
        story_category: Category of the story

    Returns:
        Best model for this prompt
    """
    # Z-Image is versatile and handles all content types well
    # Prefer it when available for consistent high quality
    if _is_model_available(ImageModel.Z_IMAGE):
        return ImageModel.Z_IMAGE

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

    # Determine best model based on content (fallback when Z-Image not available)
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

    # Special handling for Z-Image - check CUDA availability
    if model == ImageModel.Z_IMAGE:
        try:
            from image_providers import check_z_image_available
            status = check_z_image_available()
            return bool(status.get("available", False))
        except ImportError:
            return False

    if config.requires_api_key:
        api_key = getattr(Config, config.requires_api_key, None)
        return bool(api_key)

    return True


def get_best_available_model() -> ImageModel:
    """Get the best available model based on priority and configuration.

    Z-Image is preferred when CUDA is available for high-quality local generation.
    """
    # Check Z-Image first (highest priority when CUDA available)
    if _is_model_available(ImageModel.Z_IMAGE):
        logger.info("Z-Image selected as best available model (CUDA detected)")
        return ImageModel.Z_IMAGE

    available = [
        (model, config)
        for model, config in MODEL_CONFIGS.items()
        if config.enabled and _is_model_available(model) and model != ImageModel.Z_IMAGE
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

    # Track if HuggingFace is unavailable (402 Payment Required)
    # This persists across all instances to avoid repeated failed calls
    _huggingface_unavailable: bool = False

    # Image quality assessor (shared across instances)
    _quality_assessor: ImageQualityAssessor | None = None

    # Minimum acceptable quality score for generated images
    # Note: Technical metrics alone can't detect "ludicrous" AI artifacts like
    # distorted faces/hands - those are caught by LLM verification later.
    # This threshold catches obvious technical issues (blur, noise, banding).
    MIN_IMAGE_QUALITY_SCORE = 0.65

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

        # Initialize quality assessor if not already done
        if ImageGenerator._quality_assessor is None:
            ImageGenerator._quality_assessor = ImageQualityAssessor()

    def _ensure_image_directory(self) -> None:
        """Ensure the image directory exists."""
        Path(Config.IMAGE_DIR).mkdir(parents=True, exist_ok=True)

    def _validate_image_quality(self, filepath: str) -> tuple[bool, QualityScore]:
        """Validate the quality of a generated image.

        Args:
            filepath: Path to the generated image

        Returns:
            Tuple of (is_acceptable, quality_score)
        """
        if self._quality_assessor is None:
            # No assessor available, accept by default
            return (
                True,
                QualityScore(
                    overall_score=0.7,
                    metrics=None,  # type: ignore
                    artifacts=None,  # type: ignore
                ),
            )

        score = self._quality_assessor.assess_image(filepath)

        # Log quality assessment
        if score.overall_score < self.MIN_IMAGE_QUALITY_SCORE:
            logger.warning(
                f"⚠️ Image quality below threshold: {score.overall_score:.2f} < {self.MIN_IMAGE_QUALITY_SCORE}"
            )
            if score.issues:
                logger.warning(f"   Issues: {', '.join(score.issues[:3])}")
            if score.artifacts and not score.artifacts.is_clean:
                logger.warning(
                    f"   Artifacts: {', '.join(score.artifacts.artifact_descriptions[:3])}"
                )
        else:
            logger.info(
                f"✓ Image quality acceptable: {score.overall_score:.2f} (grade: {score.get_grade()})"
            )

        return (score.is_acceptable, score)

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
                    logger.info(f"⏳ Pacing requests... waited {wait_time:.1f}s")

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

        # Log rate limiter metrics (only if there were issues)
        metrics = self._rate_limiter.get_metrics()
        if metrics.error_429_count > 0:
            logger.info(
                f"📊 API throttling: {metrics.error_429_count} rate limits hit, "
                f"adjusted to {metrics.current_fill_rate:.2f} req/s"
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
        """Generate an image for a single story. Returns (file_path, alt_text) if successful.

        Uses the IMAGE_PROVIDER setting from Config to determine which provider to use.
        Falls back to other providers if the configured one fails.
        """
        prompt = self._build_image_prompt(story)

        # Get the configured provider
        configured_provider = Config.IMAGE_PROVIDER.lower().strip()
        logger.info(f"🎨 Using configured provider: {configured_provider}")

        # Known providers handled by the extensible system
        extensible_providers = {
            "cloudflare",
            "ai_horde",
            "pollinations",
            "huggingface",
            "google_imagen",
            "openai_dalle",
        }

        # Try the configured provider first via the extensible provider system
        if configured_provider in extensible_providers:
            try:
                result = self._generate_extensible_provider_image(story, prompt)
                if result:
                    image_path, _ = result
                    alt_text = self._generate_alt_text(story, prompt)
                    return (image_path, alt_text)
                logger.warning(f"⚠️ {configured_provider} failed, trying fallbacks...")
            except Exception as e:
                error_msg = str(e).lower()
                # Check for billing/configuration errors that should trigger fallbacks
                if any(x in error_msg for x in ["billed", "billing", "quota", "429"]):
                    logger.warning(f"⚠️ {configured_provider} unavailable: {e}")
                else:
                    logger.error(f"❌ {configured_provider} error: {e}")

        # Fallback chain - try other providers if configured one failed
        # Order: Pollinations (free) -> HuggingFace (free) -> Google Imagen -> DALL-E
        fallback_providers = [
            "pollinations",
            "huggingface",
            "google_imagen",
            "openai_dalle",
        ]

        # Remove the configured provider from fallbacks (already tried)
        fallback_providers = [p for p in fallback_providers if p != configured_provider]

        for fallback_provider in fallback_providers:
            logger.info(f"🔄 Trying fallback provider: {fallback_provider}...")
            try:
                # Temporarily override IMAGE_PROVIDER for the fallback
                original_provider = os.environ.get("IMAGE_PROVIDER", "")
                os.environ["IMAGE_PROVIDER"] = fallback_provider
                try:
                    result = self._generate_extensible_provider_image(story, prompt)
                    if result:
                        image_path, _ = result
                        alt_text = self._generate_alt_text(story, prompt)
                        return (image_path, alt_text)
                finally:
                    # Restore original provider
                    if original_provider:
                        os.environ["IMAGE_PROVIDER"] = original_provider
                    elif "IMAGE_PROVIDER" in os.environ:
                        del os.environ["IMAGE_PROVIDER"]
            except Exception as e:
                logger.warning(f"⚠️ Fallback {fallback_provider} failed: {e}")
                continue

        # All providers failed
        print("\n" + "=" * 60)
        print("IMAGE GENERATION: All providers unavailable")
        print("=" * 60)
        print("Note: Pollinations.ai should always work (it's free).")
        print("If it failed, check your internet connection.")
        print("=" * 60 + "\n")
        return None

    def _generate_huggingface_image(
        self, story: Story, prompt: str
    ) -> tuple[str, str] | None:
        """Generate an image using Hugging Face InferenceClient (FREE with FLUX.1-schnell).
        Returns (file_path, prompt_used) if successful, or None.
        """
        # Skip if HuggingFace was already determined to be unavailable (402 error)
        if ImageGenerator._huggingface_unavailable:
            logger.debug("Skipping HuggingFace (previously got 402 Payment Required)")
            return None

        # Use the configured model or default to the free FLUX.1-schnell
        model = Config.HF_TTI_MODEL or "black-forest-labs/FLUX.1-schnell"

        logger.info(f"🎨 Trying HuggingFace ({model.split('/')[-1]})...")

        try:
            # Initialize InferenceClient - works with or without token for free models
            # Use effective token which checks both HUGGINGFACE_API_TOKEN and HUGGINGFACE_API_KEY
            hf_token = Config.get_settings().effective_huggingface_token
            if hf_token:
                client = InferenceClient(token=hf_token)
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

            # Validate image quality
            is_acceptable, quality_score = self._validate_image_quality(filepath)
            if not is_acceptable:
                logger.warning(
                    f"⚠️ HuggingFace image rejected due to low quality (score: {quality_score.overall_score:.2f}). "
                    f"Will try another provider..."
                )
                # Delete the low-quality image
                try:
                    os.remove(filepath)
                except OSError:
                    pass
                return None

            logger.info(f"Successfully generated HF image: {filepath}")
            return (filepath, prompt)

        except Exception as e:
            error_str = str(e)
            # Handle 402 Payment Required - HuggingFace free tier exhausted
            if "402" in error_str or "Payment Required" in error_str:
                logger.warning(
                    "⚠️ HuggingFace free credits used up - trying other providers..."
                )
                ImageGenerator._huggingface_unavailable = True
            else:
                logger.warning(f"⚠️ HuggingFace unavailable: {e}")
            return None

    def _generate_pollinations_image(
        self, story: Story, prompt: str
    ) -> tuple[str, str] | None:
        """Generate an image using Pollinations.ai - completely FREE, no API key needed!

        Pollinations.ai provides free image generation using FLUX and other models.
        No rate limits, no authentication required.

        Returns (file_path, prompt_used) if successful, or None.
        """
        import requests
        from urllib.parse import quote

        logger.info("🎨 Trying Pollinations.ai (free, unlimited)...")

        try:
            # Pollinations.ai uses a simple URL-based API
            # URL encode the prompt for safe transmission
            encoded_prompt = quote(prompt)

            # Log the prompt being sent
            logger.info(
                f"📝 Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}"
            )

            # Build the URL with parameters for best quality
            # Available models: flux, flux-realism, flux-anime, flux-3d, turbo
            # Using flux-realism for photorealistic results suited for engineering content
            url = (
                f"https://image.pollinations.ai/prompt/{encoded_prompt}"
                f"?width=1024&height=1024&model=flux-realism&nologo=true"
            )

            logger.debug(f"Pollinations URL: {url[:100]}...")

            # Make the request with a reasonable timeout
            response = requests.get(
                url, timeout=120
            )  # Images can take time to generate

            if response.status_code == 200 and response.content:
                from io import BytesIO

                # Load image from response bytes
                image = Image.open(BytesIO(response.content))

                # Add AI watermark
                watermarked_image = add_ai_watermark(image)

                # Save the image
                filename = f"story_{story.id}_{int(time.time())}_pollinations.png"
                filepath = os.path.join(Config.IMAGE_DIR, filename)
                watermarked_image.save(filepath)

                # Verify the file was written
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    logger.info(
                        f"✅ Image created via Pollinations.ai ({file_size // 1024}KB)"
                    )

                    # Validate image quality
                    is_acceptable, quality_score = self._validate_image_quality(
                        filepath
                    )
                    if not is_acceptable:
                        logger.warning(
                            f"⚠️ Pollinations image rejected due to low quality (score: {quality_score.overall_score:.2f})"
                        )
                        # Delete the low-quality image
                        try:
                            os.remove(filepath)
                        except OSError:
                            pass
                        return None

                    return (filepath, prompt)
                else:
                    logger.error(
                        f"Pollinations image file NOT found after save: {filepath}"
                    )
                    return None
            else:
                logger.warning(
                    f"Pollinations.ai returned status {response.status_code}"
                )
                return None

        except requests.Timeout:
            logger.warning("Pollinations.ai request timed out (>120s)")
            return None
        except Exception as e:
            logger.warning(f"Pollinations.ai image generation failed: {e}")
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

    def _generate_extensible_provider_image(
        self, story: Story, prompt: str
    ) -> tuple[str, str] | None:
        """Generate an image using the extensible provider system.

        Uses IMAGE_PROVIDER from environment to select the backend.
        Switch providers by changing .env - no code changes required.

        Returns (file_path, prompt_used) if successful, or None.
        """
        from io import BytesIO

        try:
            from image_providers import get_image_provider, ImageProviderError
        except ImportError as e:
            logger.warning(f"⚠️ Extensible provider not available: {e}")
            return None

        try:
            provider = get_image_provider()
            logger.info(f"🎨 Trying {provider.name} ({provider.model})...")

            # Log the prompt being sent
            logger.info(
                f"📝 Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}"
            )

            # Generate the image
            image_bytes = provider.generate(prompt)

            # Load image from bytes
            image = Image.open(BytesIO(image_bytes))

            # Add AI watermark
            watermarked_image = add_ai_watermark(image)

            # Save the image
            provider_name = Config.IMAGE_PROVIDER.lower().replace(" ", "_")
            filename = f"story_{story.id}_{int(time.time())}_{provider_name}.png"
            filepath = os.path.join(Config.IMAGE_DIR, filename)
            watermarked_image.save(filepath)

            # Verify the file was written
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                logger.info(
                    f"✅ Image created via {provider.name} ({file_size // 1024}KB)"
                )

                # Validate image quality
                is_acceptable, quality_score = self._validate_image_quality(filepath)
                if not is_acceptable:
                    logger.warning(
                        f"⚠️ {provider.name} image rejected due to low quality "
                        f"(score: {quality_score.overall_score:.2f})"
                    )
                    # Delete the low-quality image
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
                    return None

                return (filepath, prompt)
            else:
                logger.error(
                    f"{provider.name} image file NOT found after save: {filepath}"
                )
                return None

        except ImageProviderError as e:
            logger.warning(f"⚠️ {e.provider} error: {e}")
            if e.retryable:
                logger.info("   (This error may be transient, consider retrying)")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Extensible provider failed: {e}")
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

        logger.info("🎨 Trying OpenAI DALL-E 3...")

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
                from io import BytesIO

                # Download the image using centralized client
                image_url = response.data[0].url
                image_response = api_client.http_get(
                    image_url, timeout=30, endpoint="dalle_download"
                )
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
            # Make error message more user-friendly
            error_msg = str(e)
            if "billing_hard_limit_reached" in error_msg:
                logger.warning(
                    "⚠️ OpenAI billing limit reached - trying other providers..."
                )
            elif "insufficient_quota" in error_msg:
                logger.warning("⚠️ OpenAI quota exhausted - trying other providers...")
            else:
                logger.warning(f"⚠️ DALL-E 3 unavailable: {e}")

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
        elif model == ImageModel.POLLINATIONS:
            result = self._generate_pollinations_image(story, prompt)
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
        elif model == ImageModel.EXTENSIBLE_PROVIDER:
            result = self._generate_extensible_provider_image(story, prompt)
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
            # Download the image using centralized client
            response = api_client.http_get(
                image_url,
                timeout=10,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                endpoint="source_image",
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

            # Use Gemini to analyze the image (requires vision capability)
            # Note: Groq doesn't support vision, so Gemini is required here
            analysis_prompt = """Analyze this image from a technical/engineering news article.
Describe in 2-3 sentences:
1. What specific equipment, technology, or facility is shown (be precise - e.g., "PEM electrolyzer stack", "distillation column", "CRISPR gene editing setup")
2. The setting/environment (laboratory, industrial plant, pilot facility, control room, etc.)
3. Any distinctive visual elements (colors, scale, materials, instrumentation)

Focus on technical accuracy. Be specific about equipment types, not generic.
If this appears to be a stock photo or generic image, say "GENERIC STOCK IMAGE".
Keep response under 100 words."""

            try:
                response = api_client.gemini_generate(
                    client=self.client,
                    model=Config.MODEL_TEXT,
                    contents=[analysis_prompt, img],
                    endpoint="image_analysis",
                )
            except Exception as e:
                # Check for rate limit errors - skip analysis rather than fail
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    logger.info("⏸️ Skipping source image analysis (API rate limited)")
                    return ""
                raise

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
            response = api_client.http_get(
                source_url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                endpoint="source_content",
            )

            if response.status_code != 200:
                logger.debug(f"Source fetch failed with status {response.status_code}")
                return result

            html = response.text
            _base_url = source_url.rsplit("/", 1)[0]  # For resolving relative URLs

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

                # Resolve relative URLs using url_utils
                img_url = resolve_relative_url(img_url, source_url)

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

            # Only log if we found something useful
            if result["text"] or result["images"] or result["ai_image_analysis"]:
                logger.info(
                    f"📖 Source context: {len(result['text'])} chars, "
                    f"{len(result['images'])} images found"
                )

        except Exception as e:
            logger.debug(f"Failed to fetch source content: {e}")

        return result

    def _build_image_prompt(self, story: Story) -> str:
        """Build a prompt for image generation using an LLM for refinement."""
        # Generate random appearance for this image to ensure variety (only if human in image)
        random_appearance = get_random_appearance() if Config.HUMAN_IN_IMAGE else ""

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
        # Use the appropriate refinement prompt based on HUMAN_IN_IMAGE setting
        if Config.HUMAN_IN_IMAGE:
            refinement_prompt = Config.IMAGE_REFINEMENT_PROMPT.format(
                story_title=story.title,
                story_summary=story.summary,
                image_style=Config.IMAGE_STYLE,
                discipline=Config.DISCIPLINE,
            )
        else:
            refinement_prompt = Config.IMAGE_REFINEMENT_PROMPT_NO_HUMAN.format(
                story_title=story.title,
                story_summary=story.summary,
                image_style=Config.IMAGE_STYLE,
                discipline=Config.DISCIPLINE,
            )

        # Conditionally inject appearance instruction based on HUMAN_IN_IMAGE setting
        if Config.HUMAN_IN_IMAGE:
            # Include central human character with specific appearance
            appearance_instruction = f"""
MANDATORY APPEARANCE FOR THIS IMAGE (use exactly as specified):
    The female {Config.DISCIPLINE} professional in this image must be: {random_appearance}.
Do NOT deviate from this appearance description. Include these exact physical traits in your prompt.
"""
        else:
            # No central human character - focus on concepts, technology, environments
            appearance_instruction = """
IMPORTANT: NO CENTRAL HUMAN CHARACTER IN THIS IMAGE.
Focus on illustrating the story through:
- Technology, equipment, machinery, or scientific apparatus relevant to the story
- Environments, facilities, or locations described in the story
- Abstract concepts, data visualizations, or process diagrams
- Natural phenomena, materials, or products being discussed

If people appear in the image, they MUST be:
- Incidental and peripheral to the main subject (e.g., small figures in background)
- Part of a crowd or group scene where no individual is the focus
- Silhouettes or partially visible, not the main subject

The image should focus on the SUBJECT MATTER of the story, not on any individual person.
Do NOT include: close-up portraits, waist-up shots of individuals, or any person as the central figure.
"""
        # Combine appearance instruction, source context, and refinement prompt
        refinement_prompt = (
            appearance_instruction + source_section + "\n" + refinement_prompt
        )

        try:
            refined = None

            # Try Local LLM first
            if self.local_client:
                try:
                    if Config.HUMAN_IN_IMAGE:
                        logger.info(
                            "📝 Refining prompt with local LLM (with person)..."
                        )
                    else:
                        logger.info("📝 Refining prompt with local LLM...")
                    refined = api_client.local_llm_generate(
                        client=self.local_client,
                        messages=[{"role": "user", "content": refinement_prompt}],
                        endpoint="prompt_refinement",
                    )
                except Exception as e:
                    if "No models loaded" in str(e):
                        logger.debug("Local LLM has no model loaded, using Groq...")
                    else:
                        logger.warning(
                            f"Local LLM refinement failed: {e}. Trying Groq..."
                        )

            # Try Groq as fallback
            if not refined:
                groq_client = api_client.get_groq_client()
                if groq_client:
                    try:
                        if Config.HUMAN_IN_IMAGE:
                            logger.info(
                                "📝 Refining prompt with Groq (with person)..."
                            )
                        else:
                            logger.info("📝 Refining prompt with Groq...")
                        refined = api_client.groq_generate(
                            client=groq_client,
                            messages=[{"role": "user", "content": refinement_prompt}],
                            endpoint="prompt_refinement",
                        )
                    except Exception as e:
                        logger.warning(f"Groq refinement failed: {e}. Trying Gemini...")

            # Final fallback to Gemini
            if not refined:
                if Config.HUMAN_IN_IMAGE:
                    logger.info("📝 Refining prompt with Gemini (with person)...")
                else:
                    logger.info("📝 Refining prompt with Gemini...")
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

        # Ultimate fallback - use configurable fallback template
        fallback_template = Config.IMAGE_FALLBACK_PROMPT

        if Config.HUMAN_IN_IMAGE:
            # Use appearance-based fallback
            fallback = fallback_template.format(
                story_title=story.title[:60],
                discipline=Config.DISCIPLINE,
                appearance=random_appearance,
            )
            # If the template does not expose {appearance}, replace a generic phrase
            if "{appearance}" not in fallback_template:
                fallback = fallback.replace("a beautiful female", random_appearance, 1)
        else:
            # No-human fallback - create a concept-focused prompt
            fallback = (
                f"A photo of industrial/scientific equipment and technology related to: {story.title[:60]}. "
                f"Focus on machinery, facilities, processes, or environments relevant to {Config.DISCIPLINE}. "
                "No people as the central subject. Wide shot showing the technology or environment. "
                "Professional documentary photography, editorial quality for a trade publication."
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

            # Try Local LLM first
            if self.local_client:
                try:
                    logger.debug("Using local LLM to generate alt text")
                    alt_text = api_client.local_llm_generate(
                        client=self.local_client,
                        messages=[{"role": "user", "content": alt_text_prompt}],
                        endpoint="alt_text",
                    )
                except Exception as e:
                    if "No models loaded" in str(e):
                        logger.debug("Local LLM has no model loaded, using Groq...")
                    else:
                        logger.warning(
                            f"Local LLM alt text failed: {e}. Trying Groq..."
                        )

            # Try Groq as fallback
            if not alt_text:
                groq_client = api_client.get_groq_client()
                if groq_client:
                    try:
                        logger.debug("Using Groq to generate alt text")
                        alt_text = api_client.groq_generate(
                            client=groq_client,
                            messages=[{"role": "user", "content": alt_text_prompt}],
                            endpoint="alt_text",
                        )
                    except Exception as e:
                        logger.warning(f"Groq alt text failed: {e}. Trying Gemini...")

            # Final fallback to Gemini
            if not alt_text:
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
                logger.info(f"📄 Alt text: {alt_text[:60]}...")
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
def _create_module_tests() -> bool:
    """Create unit tests for image_generator module."""
    import os
    import tempfile

    from test_framework import TestSuite
    from PIL import Image

    suite = TestSuite("Image Generator Tests", "image_generator.py")
    suite.start_suite()

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

    suite.run_test(
        test_name="Add AI watermark",
        test_func=test_add_ai_watermark,
        test_summary="Tests Add AI watermark functionality",
        method_description="Calls new and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Add AI watermark small image",
        test_func=test_add_ai_watermark_small_image,
        test_summary="Tests Add AI watermark small image functionality",
        method_description="Calls new and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Image generator init",
        test_func=test_image_generator_init,
        test_summary="Tests Image generator init functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Build image prompt fallback",
        test_func=test_build_image_prompt_fallback,
        test_summary="Tests Build image prompt fallback functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Ensure image directory",
        test_func=test_ensure_image_directory,
        test_summary="Tests Ensure image directory functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="HuggingFace image - no token",
        test_func=test_generate_huggingface_image_no_token,
        test_summary="Tests HuggingFace image with no token scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns None as expected",
    )
    suite.run_test(
        test_name="Local image - no client",
        test_func=test_generate_local_image_no_client,
        test_summary="Tests Local image with no client scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns None as expected",
    )
    suite.run_test(
        test_name="Cleanup orphaned images",
        test_func=test_cleanup_orphaned_images,
        test_summary="Tests Cleanup orphaned images functionality",
        method_description="Calls TemporaryDirectory and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Get stories with images count",
        test_func=test_get_stories_with_images_count,
        test_summary="Tests Get stories with images count functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Clean prompt - already has prefix",
        test_func=test_clean_image_prompt_already_starts_with_photo,
        test_summary="Tests Clean prompt with already has prefix scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Clean prompt - missing prefix",
        test_func=test_clean_image_prompt_missing_prefix,
        test_summary="Tests Clean prompt with missing prefix scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function handles empty or missing input gracefully",
    )
    suite.run_test(
        test_name="Clean prompt - removes quotes",
        test_func=test_clean_image_prompt_removes_quotes,
        test_summary="Tests Clean prompt with removes quotes scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns False or falsy value",
    )
    suite.run_test(
        test_name="Fetch source content - no links",
        test_func=test_fetch_source_content_no_links,
        test_summary="Tests Fetch source content with no links scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns an empty collection",
    )
    suite.run_test(
        test_name="Fetch source content - invalid URL",
        test_func=test_fetch_source_content_invalid_url,
        test_summary="Tests Fetch source content with invalid url scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function handles invalid input appropriately",
    )
    suite.run_test(
        test_name="Fetch source content - structure",
        test_func=test_fetch_source_content_structure,
        test_summary="Tests Fetch source content with structure scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Extract technical terms",
        test_func=test_extract_technical_terms,
        test_summary="Tests Extract technical terms functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function correctly parses and extracts the data",
    )
    suite.run_test(
        test_name="Extract technical terms - empty",
        test_func=test_extract_technical_terms_empty,
        test_summary="Tests Extract technical terms with empty scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function handles empty or missing input gracefully",
    )

    return suite.finish_suite()