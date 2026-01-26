"""Provider factory - single entry point for image generation.

Reads IMAGE_PROVIDER from environment and returns the appropriate provider.
Switch providers by changing .env - no code changes required.

Environment variables:
    IMAGE_PROVIDER: Provider name (cloudflare, ai_horde, pollinations, huggingface)
    IMAGE_MODEL: Model identifier for the selected provider
    IMAGE_SIZE: Output size as 'WIDTHxHEIGHT' (default: 1024x1024)
    IMAGE_TIMEOUT_SECONDS: Request timeout (default: 60)

Provider-specific variables - see individual provider modules.
"""

import os
import logging
from typing import Optional

from .base import ImageProvider

logger = logging.getLogger(__name__)

# Registry of available providers
PROVIDERS = {
    "cloudflare": "CloudflareProvider",
    "ai_horde": "AIHordeProvider",
    "pollinations": "PollinationsProvider",
    "huggingface": "HuggingFaceProvider",
}

# Default models for each provider
DEFAULT_MODELS = {
    "cloudflare": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    "ai_horde": "SDXL 1.0",
    "pollinations": "turbo",
    "huggingface": "black-forest-labs/FLUX.1-schnell",
}

# Model patterns that indicate compatibility with each provider


def _parse_size(size: str) -> tuple[int, int]:
    """Parse size string like '1024x1024' to tuple (width, height)."""
    try:
        parts = size.lower().split("x")
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (1024, 1024)


MODEL_PATTERNS = {
    "cloudflare": ["@cf/", "stabilityai", "bytedance", "lykon"],
    "ai_horde": ["SDXL", "stable_diffusion", "Deliberate", "Anything"],
    "pollinations": ["flux", "turbo"],
    "huggingface": ["/", "black-forest", "stabilityai", "runwayml"],
}


def _is_model_compatible(provider: str, model: str) -> bool:
    """Check if a model identifier looks compatible with a provider."""
    patterns = MODEL_PATTERNS.get(provider, [])
    if not patterns:
        # No patterns defined means accept anything
        return True
    model_lower = model.lower()
    return any(pattern.lower() in model_lower for pattern in patterns)


def get_image_provider(
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    size: Optional[str] = None,
    timeout: Optional[int] = None,
) -> ImageProvider:
    """Get an image provider instance based on configuration.

    Args:
        provider_name: Provider name (overrides IMAGE_PROVIDER env var)
        model: Model identifier (overrides IMAGE_MODEL env var)
        size: Image size (overrides IMAGE_SIZE env var)
        timeout: Timeout in seconds (overrides IMAGE_TIMEOUT_SECONDS env var)

    Returns:
        Configured ImageProvider instance

    Raises:
        RuntimeError: If provider is unknown or misconfigured

    Example:
        # Use environment configuration
        provider = get_image_provider()
        image_bytes = provider.generate("A sunset over mountains")

        # Override specific settings
        provider = get_image_provider(provider_name="pollinations", model="flux-anime")
    """
    # Read configuration with overrides
    provider = provider_name or os.getenv("IMAGE_PROVIDER", "pollinations")
    provider = provider.lower().strip()

    # Get model from parameter, env, or use provider's default
    env_model = os.getenv("IMAGE_MODEL", "")
    requested_model = model or env_model

    # Use default model if none specified or if model looks incompatible with provider
    if not requested_model or not _is_model_compatible(provider, requested_model):
        requested_model = DEFAULT_MODELS.get(provider, "")
        if env_model and not model:
            logger.debug(
                f"Model '{env_model}' not compatible with {provider}, "
                f"using default: {requested_model}"
            )

    model = requested_model
    size = size or os.getenv("IMAGE_SIZE", "1024x1024")
    timeout = timeout or int(os.getenv("IMAGE_TIMEOUT_SECONDS", "180"))

    if provider not in PROVIDERS:
        available = ", ".join(sorted(PROVIDERS.keys()))
        raise RuntimeError(
            f"Unknown IMAGE_PROVIDER: '{provider}'. Available providers: {available}"
        )

    logger.debug(f"Creating {provider} provider with model={model}, size={size}")

    # Lazy import and instantiate the provider
    if provider == "cloudflare":
        from .cloudflare import CloudflareProvider

        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        api_token = os.getenv("CLOUDFLARE_API_TOKEN", "")

        if not account_id or not api_token:
            raise RuntimeError(
                "Cloudflare provider requires CLOUDFLARE_ACCOUNT_ID and "
                "CLOUDFLARE_API_TOKEN environment variables"
            )

        return CloudflareProvider(
            model=model,
            size=size,
            account_id=account_id,
            api_token=api_token,
            timeout=timeout,
        )

    if provider == "ai_horde":
        from .ai_horde import AIHordeProvider

        api_key = os.getenv("AI_HORDE_API_KEY", "0000000000")  # Anonymous key
        # AI Horde is queue-based and can be slow, use longer timeout
        ai_horde_timeout = max(timeout, 180)  # At least 3 minutes

        return AIHordeProvider(
            model=model,
            size=size,
            api_key=api_key,
            timeout=ai_horde_timeout,
        )

    if provider == "pollinations":
        from .pollinations import PollinationsProvider

        # Turbo model can take 90+ seconds, ensure adequate timeout
        pollinations_timeout = max(timeout, 180) if model == "turbo" else timeout

        return PollinationsProvider(
            model=model,
            size=size,
            timeout=pollinations_timeout,
        )

    if provider == "huggingface":
        from .huggingface import HuggingFaceProvider

        # Check both token names for backward compatibility
        api_token = (
            os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or ""
        )
        negative_prompt = os.getenv(
            "HF_NEGATIVE_PROMPT",
            "text, watermark, logo, blurry, low quality, artifacts, jpeg artifacts, nsfw",
        )

        return HuggingFaceProvider(
            model=model,
            size=size,
            api_token=api_token or None,
            negative_prompt=negative_prompt,
            timeout=timeout,
        )

    # Should never reach here due to provider check above
    raise RuntimeError(f"Provider '{provider}' not implemented")


def list_available_providers() -> dict[str, dict]:
    """List all available providers and their configuration status.

    Returns:
        Dictionary mapping provider names to their status:
        {
            "cloudflare": {
                "configured": True/False,
                "requires": ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"],
                "default_model": "..."
            },
            ...
        }
    """
    status = {}

    for name in PROVIDERS:
        info = {
            "default_model": DEFAULT_MODELS.get(name, ""),
            "requires": [],
            "configured": False,
        }

        if name == "cloudflare":
            info["requires"] = ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"]
            info["configured"] = bool(
                os.getenv("CLOUDFLARE_ACCOUNT_ID") and os.getenv("CLOUDFLARE_API_TOKEN")
            )
        elif name == "ai_horde":
            info["requires"] = []  # Anonymous access available
            info["configured"] = True
        elif name == "pollinations":
            info["requires"] = []  # No API key needed
            info["configured"] = True
        elif name == "huggingface":
            info["requires"] = []  # Free models work without token
            info["configured"] = True

        status[name] = info

    return status


def get_configured_providers() -> list[str]:
    """Get list of providers that are properly configured.

    Returns:
        List of provider names that can be used immediately
    """
    return [
        name for name, info in list_available_providers().items() if info["configured"]
    ]
