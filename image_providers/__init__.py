"""Image provider package - extensible provider pattern for image generation.

Switch provider via IMAGE_PROVIDER environment variable.
Z-Image is auto-selected when CUDA GPU is available.
"""

from .base import ImageProvider, ImageProviderError, GenerationResult
from .factory import (
    get_image_provider,
    list_available_providers,
    get_configured_providers,
    check_z_image_available,
    offer_z_image_install,
    get_default_provider,
)

__all__ = [
    "ImageProvider",
    "ImageProviderError",
    "GenerationResult",
    "get_image_provider",
    "list_available_providers",
    "get_configured_providers",
    "check_z_image_available",
    "offer_z_image_install",
    "get_default_provider",
]
