"""Image provider package - extensible provider pattern for image generation.

Switch provider via IMAGE_PROVIDER environment variable.
No code changes required to swap providers.
"""

from .base import ImageProvider, ImageProviderError
from .factory import get_image_provider, list_available_providers

__all__ = [
    "ImageProvider",
    "ImageProviderError",
    "get_image_provider",
    "list_available_providers",
]
