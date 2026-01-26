"""Google Imagen image provider.

Uses Google's Imagen model via the google.genai SDK.
Requires a Google Cloud project with Vertex AI API enabled and billing.

Environment variables:
    GOOGLE_API_KEY: Google AI API key
    MODEL_IMAGE: Imagen model name (default: imagen-4.0-generate-001)
    IMAGE_ASPECT_RATIO: Aspect ratio (default: 1:1)
"""

import os
import logging
from typing import Optional

from .base import ImageProvider, ImageProviderError

logger = logging.getLogger(__name__)


class GoogleImagenProvider(ImageProvider):
    """Image generation via Google Imagen.

    Supports high-quality image generation using Google's Imagen models.
    Requires a billed Google Cloud account.
    """

    DEFAULT_MODEL = "imagen-4.0-generate-001"

    def __init__(
        self,
        model: str = "",
        size: str = "1024x1024",
        timeout: int = 60,
        api_key: Optional[str] = None,
        aspect_ratio: str = "1:1",
    ):
        """Initialize Google Imagen provider.

        Args:
            model: Imagen model name (default: imagen-4.0-generate-001)
            size: Output size (e.g., '1024x1024', '2K')
            timeout: Request timeout in seconds
            api_key: Google API key (or from GOOGLE_API_KEY env)
            aspect_ratio: Image aspect ratio (e.g., '1:1', '16:9')
        """
        model = model or self.DEFAULT_MODEL
        super().__init__(model, size, timeout)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.aspect_ratio = aspect_ratio or os.getenv("IMAGE_ASPECT_RATIO", "1:1")
        self._client = None

    @property
    def name(self) -> str:
        return "Google Imagen"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        """Get or create the google.genai client."""
        if self._client is None:
            try:
                import google.genai as genai

                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImageProviderError(
                    "google-genai package not installed. Run: pip install google-genai",
                    provider=self.name,
                )
            except Exception as e:
                raise ImageProviderError(
                    f"Failed to initialize Google client: {e}",
                    provider=self.name,
                )
        return self._client

    def generate(self, prompt: str) -> bytes:
        """Generate an image using Google Imagen.

        Args:
            prompt: Text description of the image

        Returns:
            Raw image bytes (PNG)

        Raises:
            ImageProviderError: If generation fails
        """
        try:
            from google.genai import types
        except ImportError:
            raise ImageProviderError(
                "google-genai package not installed. Run: pip install google-genai",
                provider=self.name,
            )

        if not self.api_key:
            raise ImageProviderError(
                "Google Imagen requires GOOGLE_API_KEY to be set",
                provider=self.name,
            )

        client = self._get_client()
        logger.debug(f"Google Imagen request: model={self.model}, size={self.size}")

        max_retries = 3
        response = None
        last_error = None

        for attempt in range(max_retries):
            try:
                response = client.models.generate_images(
                    model=self.model,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        safety_filter_level=types.SafetyFilterLevel.BLOCK_LOW_AND_ABOVE,
                        person_generation=types.PersonGeneration.ALLOW_ADULT,
                        aspect_ratio=self.aspect_ratio,
                        image_size=self.size,
                    ),
                )

                if response.generated_images and response.generated_images[0].image:
                    image_bytes = response.generated_images[0].image.image_bytes
                    if image_bytes:
                        logger.info(
                            f"✓ Google Imagen generated {len(image_bytes) // 1024}KB image"
                        )
                        return image_bytes

                # No image generated - likely safety filter block
                if attempt < max_retries - 1:
                    logger.warning(
                        f"No image generated (attempt {attempt + 1}/{max_retries}) - "
                        "safety filter may have blocked. Retrying..."
                    )
                    import time

                    time.sleep(1)
                else:
                    raise ImageProviderError(
                        "Google Imagen safety filter blocked the request after retries",
                        provider=self.name,
                        retryable=False,
                    )

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()

                # Check for billing/quota errors
                if "billed" in error_msg:
                    raise ImageProviderError(
                        "Google Imagen requires a billed Google Cloud account",
                        provider=self.name,
                        retryable=False,
                    )

                if "429" in error_msg or "resource_exhausted" in error_msg:
                    raise ImageProviderError(
                        "Google Imagen quota exceeded (429)",
                        provider=self.name,
                        status_code=429,
                        retryable=True,
                    )

                if attempt < max_retries - 1:
                    logger.warning(
                        f"Google Imagen error (attempt {attempt + 1}): {e}. Retrying..."
                    )
                    import time

                    time.sleep(1)
                else:
                    raise ImageProviderError(
                        f"Google Imagen generation failed: {e}",
                        provider=self.name,
                    )

        # Should not reach here, but just in case
        raise ImageProviderError(
            f"Google Imagen failed after {max_retries} attempts: {last_error}",
            provider=self.name,
        )
