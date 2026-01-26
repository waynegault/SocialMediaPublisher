"""Base interface for image generation providers.

All providers must implement this interface, ensuring consistent behaviour
across different backends (Cloudflare, AI Horde, Pollinations, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ImageProviderError(RuntimeError):
    """Raised when an image provider encounters an error.

    Attributes:
        provider: Name of the provider that raised the error
        status_code: HTTP status code if applicable
        retryable: Whether the operation can be retried
    """

    def __init__(
        self,
        message: str,
        provider: str = "",
        status_code: Optional[int] = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class GenerationResult:
    """Result from an image generation request.

    Attributes:
        image_bytes: Raw image data (PNG/JPEG)
        format: Image format ('png' or 'jpeg')
        width: Image width in pixels
        height: Image height in pixels
        model_used: Model identifier used for generation
        generation_time_ms: Time taken to generate in milliseconds
        metadata: Additional provider-specific metadata
    """

    image_bytes: bytes
    format: str = "png"
    width: int = 1024
    height: int = 1024
    model_used: str = ""
    generation_time_ms: int = 0
    metadata: Optional[dict] = None


class ImageProvider(ABC):
    """Abstract base class for image generation providers.

    Implement this interface to add support for new providers.
    The provider is responsible for:
    - Validating its own configuration
    - Making API requests
    - Returning raw image bytes
    - Raising ImageProviderError on failure

    Example:
        class MyProvider(ImageProvider):
            def generate(self, prompt: str) -> bytes:
                # ... implementation
                return image_bytes
    """

    def __init__(
        self,
        model: str,
        size: str = "1024x1024",
        timeout: int = 60,
    ):
        """Initialise the provider.

        Args:
            model: Model identifier to use for generation
            size: Output image size as 'WIDTHxHEIGHT' (e.g., '1024x1024')
            timeout: Request timeout in seconds
        """
        self.model = model
        self.size = size
        self.timeout = timeout
        self._parse_size()

    def _parse_size(self) -> None:
        """Parse size string into width and height."""
        try:
            parts = self.size.lower().split("x")
            self.width = int(parts[0])
            self.height = int(parts[1])
        except (ValueError, IndexError):
            logger.warning(f"Invalid size '{self.size}', defaulting to 1024x1024")
            self.width = 1024
            self.height = 1024

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...

    @property
    def is_configured(self) -> bool:
        """Check if the provider has valid configuration.

        Override to add provider-specific validation.
        """
        return True

    @abstractmethod
    def generate(self, prompt: str) -> bytes:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate

        Returns:
            Raw image bytes (PNG or JPEG format)

        Raises:
            ImageProviderError: If generation fails
        """
        ...

    def generate_with_result(self, prompt: str) -> GenerationResult:
        """Generate an image and return detailed result.

        Override for providers that can return richer metadata.
        Default implementation wraps generate() with basic metadata.

        Args:
            prompt: Text description of the image to generate

        Returns:
            GenerationResult with image bytes and metadata

        Raises:
            ImageProviderError: If generation fails
        """
        import time

        start = time.time()
        image_bytes = self.generate(prompt)
        elapsed_ms = int((time.time() - start) * 1000)

        return GenerationResult(
            image_bytes=image_bytes,
            format="png",
            width=self.width,
            height=self.height,
            model_used=self.model,
            generation_time_ms=elapsed_ms,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, size={self.size!r})"
