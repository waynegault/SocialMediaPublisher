"""Pollinations.ai image provider.

Completely FREE image generation - no API key required.
Uses FLUX and other models with no rate limits.

No environment variables required.
"""

import logging
from urllib.parse import quote

import requests

from .base import ImageProvider, ImageProviderError

logger = logging.getLogger(__name__)


class PollinationsProvider(ImageProvider):
    """Image generation via Pollinations.ai.

    Pollinations provides free, unlimited image generation.
    Supports multiple models:
    - flux (default)
    - flux-realism (photorealistic)
    - flux-anime
    - flux-3d
    - turbo (faster, lower quality)
    """

    BASE_URL = "https://image.pollinations.ai/prompt"

    def __init__(
        self,
        model: str = "flux-realism",
        size: str = "1024x1024",
        timeout: int = 120,
    ):
        """Initialise Pollinations provider.

        Args:
            model: Model name (flux, flux-realism, flux-anime, flux-3d, turbo)
            size: Output size as 'WIDTHxHEIGHT'
            timeout: Request timeout in seconds (default 120s as generation can be slow)
        """
        super().__init__(model, size, timeout)

    @property
    def name(self) -> str:
        return "Pollinations.ai"

    @property
    def is_configured(self) -> bool:
        # Always available - no API key required
        return True

    def generate(self, prompt: str) -> bytes:
        """Generate an image using Pollinations.ai.

        Args:
            prompt: Text description of the image

        Returns:
            Raw image bytes (PNG)

        Raises:
            ImageProviderError: If generation fails
        """
        # URL encode the prompt
        encoded_prompt = quote(prompt)

        # Build URL with parameters
        url = (
            f"{self.BASE_URL}/{encoded_prompt}"
            f"?width={self.width}&height={self.height}"
            f"&model={self.model}&nologo=true"
        )

        logger.debug(f"Pollinations request: {url[:100]}...")

        try:
            resp = requests.get(url, timeout=self.timeout)
        except requests.Timeout:
            raise ImageProviderError(
                f"Pollinations request timed out after {self.timeout}s",
                provider=self.name,
                retryable=True,
            )
        except requests.RequestException as e:
            raise ImageProviderError(
                f"Pollinations request failed: {e}",
                provider=self.name,
                retryable=True,
            )

        if not resp.ok:
            raise ImageProviderError(
                f"Pollinations error {resp.status_code}",
                provider=self.name,
                status_code=resp.status_code,
                retryable=resp.status_code in (429, 500, 502, 503, 504, 530),
            )

        if len(resp.content) < 1000:
            raise ImageProviderError(
                "Pollinations returned invalid image (too small)",
                provider=self.name,
            )

        logger.info(f"✓ Pollinations generated {len(resp.content) // 1024}KB image")
        return resp.content
