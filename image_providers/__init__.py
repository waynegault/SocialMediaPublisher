"""Image provider package - extensible provider pattern for image generation.

Switch provider via IMAGE_PROVIDER environment variable.
Z-Image is auto-selected when CUDA GPU is available.
"""

from .base import GenerationResult, ImageProvider, ImageProviderError
from .factory import (
    check_z_image_available,
    get_configured_providers,
    get_default_provider,
    get_image_provider,
    list_available_providers,
    offer_z_image_install,
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
