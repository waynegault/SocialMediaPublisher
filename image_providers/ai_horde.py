"""AI Horde image provider.

Community-run, distributed image generation network.
Queue-based and async - may be slower but truly free.

Optional environment variable:
    AI_HORDE_API_KEY: API key (defaults to 'anonymous' for free tier)
"""

import logging
import time
import requests

from .base import ImageProvider, ImageProviderError, GenerationResult

logger = logging.getLogger(__name__)


class AIHordeProvider(ImageProvider):
    """Image generation via AI Horde distributed network.

    AI Horde is a crowdsourced distributed cluster of image generation workers.
    Generation is queue-based and asynchronous.

    Supported models vary based on worker availability. Common options:
    - stable_diffusion
    - stable_diffusion_2.1
    - SDXL 1.0
    - Deliberate
    - Anything Diffusion
    """

    BASE_URL = "https://aihorde.net/api/v2"

    def __init__(
        self,
        model: str,
        size: str,
        api_key: str = "0000000000",
        timeout: int = 120,
    ):
        """Initialise AI Horde provider.

        Args:
            model: Model name (e.g., 'stable_diffusion', 'SDXL 1.0')
            size: Output size as 'WIDTHxHEIGHT'
            api_key: AI Horde API key (use '0000000000' for anonymous free tier)
            timeout: Maximum wait time for generation in seconds
        """
        super().__init__(model, size, timeout)
        self.api_key = api_key or "anonymous"

    @property
    def name(self) -> str:
        return "AI Horde"

    @property
    def is_configured(self) -> bool:
        # Always configured - anonymous access is available
        return True

    def generate(self, prompt: str) -> bytes:
        """Generate an image using AI Horde.

        This method submits a generation request and polls for completion.

        Args:
            prompt: Text description of the image

        Returns:
            Raw image bytes

        Raises:
            ImageProviderError: If generation fails or times out
        """
        # Submit generation request
        submit_url = f"{self.BASE_URL}/generate/async"

        payload = {
            "prompt": prompt,
            "models": [self.model],
            "params": {
                "width": self.width,
                "height": self.height,
                "steps": 30,  # Reasonable default
                "cfg_scale": 7.5,
            },
            "nsfw": False,
            "censor_nsfw": True,
        }

        headers = {"apikey": self.api_key}

        logger.debug(f"Submitting job to AI Horde with model {self.model}")

        try:
            submit_resp = requests.post(
                submit_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
        except requests.RequestException as e:
            raise ImageProviderError(
                f"AI Horde submission failed: {e}",
                provider=self.name,
                retryable=True,
            )

        if not submit_resp.ok:
            raise ImageProviderError(
                f"AI Horde submission error {submit_resp.status_code}: {submit_resp.text}",
                provider=self.name,
                status_code=submit_resp.status_code,
            )

        try:
            job_data = submit_resp.json()
            job_id = job_data["id"]
        except (KeyError, ValueError) as e:
            raise ImageProviderError(
                f"AI Horde returned invalid response: {e}",
                provider=self.name,
            )

        logger.info(f"AI Horde job submitted: {job_id}")

        # Poll for completion
        check_url = f"{self.BASE_URL}/generate/check/{job_id}"
        status_url = f"{self.BASE_URL}/generate/status/{job_id}"

        start_time = time.time()
        poll_interval = 2  # Start with 2 second polls

        while time.time() - start_time < self.timeout:
            try:
                check_resp = requests.get(check_url, timeout=10)
                if not check_resp.ok:
                    time.sleep(poll_interval)
                    continue

                status_data = check_resp.json()

                # Check if done
                if status_data.get("done"):
                    # Fetch the final result with image URL
                    result_resp = requests.get(status_url, timeout=30)
                    if not result_resp.ok:
                        raise ImageProviderError(
                            f"Failed to fetch AI Horde result: {result_resp.text}",
                            provider=self.name,
                        )

                    result_data = result_resp.json()
                    generations = result_data.get("generations", [])

                    if not generations:
                        raise ImageProviderError(
                            "AI Horde returned no generations",
                            provider=self.name,
                        )

                    img_url = generations[0].get("img")
                    if not img_url:
                        raise ImageProviderError(
                            "AI Horde generation missing image URL",
                            provider=self.name,
                        )

                    # Download the image
                    img_resp = requests.get(img_url, timeout=30)
                    if not img_resp.ok:
                        raise ImageProviderError(
                            f"Failed to download AI Horde image: {img_resp.status_code}",
                            provider=self.name,
                        )

                    logger.info(
                        f"✓ AI Horde generated {len(img_resp.content) // 1024}KB image"
                    )
                    return img_resp.content

                # Check for failure
                if status_data.get("faulted"):
                    raise ImageProviderError(
                        "AI Horde job faulted",
                        provider=self.name,
                    )

                # Log progress
                queue_position = status_data.get("queue_position", "?")
                wait_time = status_data.get("wait_time", "?")
                logger.debug(
                    f"AI Horde: queue position {queue_position}, ETA {wait_time}s"
                )

                # Increase poll interval gradually to avoid hammering the API
                poll_interval = min(poll_interval + 1, 10)
                time.sleep(poll_interval)

            except requests.RequestException:
                time.sleep(poll_interval)
                continue

        raise ImageProviderError(
            f"AI Horde generation timed out after {self.timeout}s",
            provider=self.name,
            retryable=True,
        )

    def generate_with_result(self, prompt: str) -> GenerationResult:
        """Generate image with additional AI Horde metadata."""
        import time

        start = time.time()
        image_bytes = self.generate(prompt)
        elapsed_ms = int((time.time() - start) * 1000)

        return GenerationResult(
            image_bytes=image_bytes,
            format="png",  # AI Horde typically returns WebP, but we store as PNG
            width=self.width,
            height=self.height,
            model_used=self.model,
            generation_time_ms=elapsed_ms,
            metadata={
                "provider": "ai_horde",
                "is_distributed": True,
            },
        )
