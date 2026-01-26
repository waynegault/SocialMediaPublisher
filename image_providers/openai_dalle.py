"""OpenAI DALL-E image provider.

Uses OpenAI's DALL-E 3 model for high-quality image generation.
Requires an OpenAI API key with available credits.

Environment variables:
    OPENAI_API_KEY: OpenAI API key
"""

import os
import logging
from typing import Optional

import requests

from .base import ImageProvider, ImageProviderError

logger = logging.getLogger(__name__)


class OpenAIDalleProvider(ImageProvider):
    """Image generation via OpenAI's DALL-E 3.

    DALL-E 3 excels at artistic, creative, and conceptual imagery.
    Requires an OpenAI API key with available credits.
    """

    DEFAULT_MODEL = "dall-e-3"

    def __init__(
        self,
        model: str = "",
        size: str = "1024x1024",
        timeout: int = 60,
        api_key: Optional[str] = None,
        quality: str = "standard",
        style: str = "natural",
    ):
        """Initialize OpenAI DALL-E provider.

        Args:
            model: Model name (default: dall-e-3)
            size: Output size - '1024x1024', '1024x1792', or '1792x1024'
            timeout: Request timeout in seconds
            api_key: OpenAI API key (or from OPENAI_API_KEY env)
            quality: 'standard' or 'hd'
            style: 'natural' or 'vivid'
        """
        model = model or self.DEFAULT_MODEL
        # DALL-E 3 only supports specific sizes
        if size not in ("1024x1024", "1024x1792", "1792x1024"):
            logger.warning(f"DALL-E 3 doesn't support size {size}, using 1024x1024")
            size = "1024x1024"
        super().__init__(model, size, timeout)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.quality = quality
        self.style = style
        self._client = None

    @property
    def name(self) -> str:
        return "OpenAI DALL-E 3"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImageProviderError(
                    "openai package not installed. Run: pip install openai",
                    provider=self.name,
                )
            except Exception as e:
                raise ImageProviderError(
                    f"Failed to initialize OpenAI client: {e}",
                    provider=self.name,
                )
        return self._client

    def generate(self, prompt: str) -> bytes:
        """Generate an image using DALL-E 3.

        Args:
            prompt: Text description of the image

        Returns:
            Raw image bytes (PNG)

        Raises:
            ImageProviderError: If generation fails
        """
        if not self.api_key:
            raise ImageProviderError(
                "DALL-E requires OPENAI_API_KEY to be set",
                provider=self.name,
            )

        client = self._get_client()
        logger.debug(
            f"DALL-E request: model={self.model}, size={self.size}, "
            f"quality={self.quality}, style={self.style}"
        )

        try:
            response = client.images.generate(
                model=self.model,
                prompt=prompt,
                n=1,
                size=self.size,  # type: ignore[arg-type]  # validated in __init__
                quality=self.quality,  # type: ignore[arg-type]  # validated param
                style=self.style,  # type: ignore[arg-type]  # validated param
            )

            if not response.data or not response.data[0].url:
                raise ImageProviderError(
                    "DALL-E returned no image",
                    provider=self.name,
                )

            # Download the image
            image_url = response.data[0].url

            # DALL-E may revise the prompt - log if different
            revised_prompt = response.data[0].revised_prompt
            if revised_prompt and revised_prompt != prompt:
                logger.debug(f"DALL-E revised prompt: {revised_prompt[:100]}...")

            try:
                image_response = requests.get(image_url, timeout=self.timeout)
                image_response.raise_for_status()
            except requests.Timeout:
                raise ImageProviderError(
                    f"DALL-E image download timed out after {self.timeout}s",
                    provider=self.name,
                    retryable=True,
                )
            except requests.RequestException as e:
                raise ImageProviderError(
                    f"DALL-E image download failed: {e}",
                    provider=self.name,
                    retryable=True,
                )

            image_bytes = image_response.content
            if len(image_bytes) < 1000:
                raise ImageProviderError(
                    "DALL-E returned invalid image (too small)",
                    provider=self.name,
                )

            logger.info(f"✓ DALL-E 3 generated {len(image_bytes) // 1024}KB image")
            return image_bytes

        except ImageProviderError:
            raise
        except Exception as e:
            error_msg = str(e).lower()

            # Check for billing/quota errors
            if "billing_hard_limit_reached" in error_msg:
                raise ImageProviderError(
                    "OpenAI billing limit reached",
                    provider=self.name,
                    retryable=False,
                )
            if "insufficient_quota" in error_msg:
                raise ImageProviderError(
                    "OpenAI quota exhausted",
                    provider=self.name,
                    retryable=False,
                )
            if "rate_limit" in error_msg or "429" in error_msg:
                raise ImageProviderError(
                    "OpenAI rate limit exceeded",
                    provider=self.name,
                    status_code=429,
                    retryable=True,
                )

            raise ImageProviderError(
                f"DALL-E generation failed: {e}",
                provider=self.name,
            )
