"""Cloudflare Workers AI image provider.

Uses Cloudflare's AI inference API with Stable Diffusion models.
Generous free tier with predictable latency.

Required environment variables:
    CLOUDFLARE_ACCOUNT_ID: Your Cloudflare account ID
    CLOUDFLARE_API_TOKEN: API token with Workers AI permissions
"""

import logging
import requests

from .base import ImageProvider, ImageProviderError

logger = logging.getLogger(__name__)


class CloudflareProvider(ImageProvider):
    """Image generation via Cloudflare Workers AI.

    Supports various Stable Diffusion models including:
    - @cf/stabilityai/stable-diffusion-xl-base-1.0
    - @cf/bytedance/stable-diffusion-xl-lightning
    - @cf/lykon/dreamshaper-8-lcm
    """

    def __init__(
        self,
        model: str,
        size: str,
        account_id: str,
        api_token: str,
        timeout: int = 60,
    ):
        """Initialise Cloudflare provider.

        Args:
            model: Cloudflare model ID (e.g., '@cf/stabilityai/stable-diffusion-xl-base-1.0')
            size: Output size as 'WIDTHxHEIGHT'
            account_id: Cloudflare account ID
            api_token: API token with Workers AI permission
            timeout: Request timeout in seconds
        """
        super().__init__(model, size, timeout)
        self.account_id = account_id
        self.api_token = api_token

        # Build endpoint URL
        # Handle model names with or without @cf/ prefix
        model_path = model if model.startswith("@cf/") else f"@cf/{model}"
        self.endpoint = (
            f"https://api.cloudflare.com/client/v4/"
            f"accounts/{account_id}/ai/run/{model_path}"
        )

        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    @property
    def name(self) -> str:
        return "Cloudflare Workers AI"

    @property
    def is_configured(self) -> bool:
        return bool(self.account_id and self.api_token)

    def generate(self, prompt: str) -> bytes:
        """Generate an image using Cloudflare Workers AI.

        Args:
            prompt: Text description of the image

        Returns:
            Raw PNG image bytes

        Raises:
            ImageProviderError: If generation fails
        """
        if not self.is_configured:
            raise ImageProviderError(
                "Cloudflare not configured: missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN",
                provider=self.name,
            )

        payload = {
            "prompt": prompt,
            "width": self.width,
            "height": self.height,
        }

        logger.debug(f"Cloudflare request to {self.endpoint}")

        try:
            resp = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout:
            raise ImageProviderError(
                f"Cloudflare request timed out after {self.timeout}s",
                provider=self.name,
                retryable=True,
            )
        except requests.RequestException as e:
            raise ImageProviderError(
                f"Cloudflare request failed: {e}",
                provider=self.name,
                retryable=True,
            )

        if not resp.ok:
            # Parse error from response
            try:
                error_data = resp.json()
                error_msg = error_data.get("errors", [{}])[0].get("message", resp.text)
            except Exception:
                error_msg = resp.text

            raise ImageProviderError(
                f"Cloudflare error {resp.status_code}: {error_msg}",
                provider=self.name,
                status_code=resp.status_code,
                retryable=resp.status_code in (429, 500, 502, 503, 504),
            )

        # Response is raw image bytes for Workers AI
        if len(resp.content) < 1000:
            raise ImageProviderError(
                "Cloudflare returned invalid image (too small)",
                provider=self.name,
            )

        logger.info(f"✓ Cloudflare generated {len(resp.content) // 1024}KB image")
        return resp.content
