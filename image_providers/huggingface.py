"""HuggingFace image provider.

Uses HuggingFace Inference API with FLUX and Stable Diffusion models.
Free tier available for many models.

Optional environment variable:
    HUGGINGFACE_API_TOKEN: HuggingFace API token (optional for free models)
"""

import logging
from typing import Optional

from .base import ImageProvider, ImageProviderError

logger = logging.getLogger(__name__)


class HuggingFaceProvider(ImageProvider):
    """Image generation via HuggingFace Inference API.

    Supports various models including:
    - black-forest-labs/FLUX.1-schnell (FREE, no token required)
    - black-forest-labs/FLUX.1-dev
    - stabilityai/stable-diffusion-xl-base-1.0
    - runwayml/stable-diffusion-v1-5
    """

    def __init__(
        self,
        model: str = "black-forest-labs/FLUX.1-schnell",
        size: str = "1024x1024",
        api_token: Optional[str] = None,
        negative_prompt: str = "",
        timeout: int = 60,
    ):
        """Initialise HuggingFace provider.

        Args:
            model: Model ID (e.g., 'black-forest-labs/FLUX.1-schnell')
            size: Output size as 'WIDTHxHEIGHT' (note: many models ignore this)
            api_token: HuggingFace API token (optional for free models)
            negative_prompt: Things to avoid in the image (ignored by FLUX models)
            timeout: Request timeout in seconds
        """
        super().__init__(model, size, timeout)
        self.api_token = api_token
        self.negative_prompt = negative_prompt
        self._client = None
        self._unavailable = False

    @property
    def name(self) -> str:
        return "HuggingFace"

    @property
    def is_configured(self) -> bool:
        # Always available - free models work without token
        return True

    def _get_client(self):
        """Lazy-load the InferenceClient."""
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient

                if self.api_token:
                    self._client = InferenceClient(token=self.api_token)
                else:
                    self._client = InferenceClient()
            except ImportError:
                raise ImageProviderError(
                    "huggingface_hub not installed. Run: pip install huggingface_hub",
                    provider=self.name,
                )
        return self._client

    def generate(self, prompt: str) -> bytes:
        """Generate an image using HuggingFace Inference API.

        Args:
            prompt: Text description of the image

        Returns:
            Raw PNG image bytes

        Raises:
            ImageProviderError: If generation fails
        """
        if self._unavailable:
            raise ImageProviderError(
                "HuggingFace free tier exhausted (402 Payment Required)",
                provider=self.name,
                status_code=402,
            )

        client = self._get_client()

        logger.debug(f"HuggingFace request with model {self.model}")

        try:
            # FLUX models ignore negative_prompt, so only pass it for SD models
            if self.negative_prompt and "flux" not in self.model.lower():
                image = client.text_to_image(
                    prompt=prompt,
                    model=self.model,
                    negative_prompt=self.negative_prompt,
                )
            else:
                image = client.text_to_image(
                    prompt=prompt,
                    model=self.model,
                )

            # Convert PIL Image to bytes
            import io

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            logger.info(f"✓ HuggingFace generated {len(image_bytes) // 1024}KB image")
            return image_bytes

        except Exception as e:
            error_str = str(e)

            # Handle 402 Payment Required - free tier exhausted
            if "402" in error_str or "Payment Required" in error_str:
                self._unavailable = True
                raise ImageProviderError(
                    "HuggingFace free tier exhausted",
                    provider=self.name,
                    status_code=402,
                )

            # Handle rate limiting
            if "429" in error_str:
                raise ImageProviderError(
                    f"HuggingFace rate limited: {e}",
                    provider=self.name,
                    status_code=429,
                    retryable=True,
                )

            raise ImageProviderError(
                f"HuggingFace generation failed: {e}",
                provider=self.name,
                retryable=True,
            )
