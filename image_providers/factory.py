"""Provider factory - single entry point for image generation.

Reads IMAGE_PROVIDER from environment and returns the appropriate provider.
Switch providers by changing .env - no code changes required.

Z-Image Auto-Detection:
    When CUDA is available and Z-Image dependencies are installed,
    Z-Image will be used as the default provider for high-quality local generation.
    Set IMAGE_PROVIDER explicitly to override this behavior.

Environment variables:
    IMAGE_PROVIDER: Provider name (cloudflare, ai_horde, pollinations, huggingface, z_image)
    IMAGE_MODEL: Model identifier for the selected provider
    IMAGE_SIZE: Output size as 'WIDTHxHEIGHT' (default: 1024x1024)
    IMAGE_TIMEOUT_SECONDS: Request timeout (default: 60)

Provider-specific variables - see individual provider modules.
"""

import os
import logging
import subprocess
import sys
from typing import Optional

from .base import ImageProvider

logger = logging.getLogger(__name__)

# Registry of available providers
PROVIDERS = {
    "cloudflare": "CloudflareProvider",
    "ai_horde": "AIHordeProvider",
    "pollinations": "PollinationsProvider",
    "huggingface": "HuggingFaceProvider",
    "google_imagen": "GoogleImagenProvider",
    "openai_dalle": "OpenAIDalleProvider",
    "z_image": "ZImageProvider",
}

# Default models for each provider
DEFAULT_MODELS = {
    "cloudflare": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    "ai_horde": "SDXL 1.0",
    "pollinations": "turbo",
    "huggingface": "black-forest-labs/FLUX.1-schnell",
    "google_imagen": "imagen-4.0-generate-001",
    "openai_dalle": "dall-e-3",
    "z_image": "Tongyi-MAI/Z-Image",
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
    "google_imagen": ["imagen"],
    "openai_dalle": ["dall-e", "dalle"],
    "z_image": ["z-image", "Tongyi-MAI", "Z-Image"],
}

# Cache for Z-Image availability check
_z_image_status: dict[str, object] = {}


def check_z_image_available() -> dict[str, object]:
    """Check if Z-Image infrastructure is available.

    Returns:
        Dictionary with status information:
        {
            "available": bool,  # True if Z-Image can be used
            "cuda_available": bool,
            "diffusers_installed": bool,
            "z_image_pipeline_available": bool,
            "gpu_name": str or None,
            "gpu_vram_gb": float or None,
            "missing": list[str],  # What's missing
            "recommendation": str,  # Human-readable recommendation
        }
    """
    global _z_image_status
    if _z_image_status:
        return _z_image_status

    status: dict[str, object] = {
        "available": False,
        "cuda_available": False,
        "diffusers_installed": False,
        "z_image_pipeline_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "missing": [],
        "recommendation": "",
    }

    missing = []

    # Check PyTorch and CUDA
    try:
        import torch

        if torch.cuda.is_available():
            status["cuda_available"] = True
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            )
        else:
            missing.append("CUDA GPU (PyTorch installed but CUDA not available)")
    except ImportError:
        missing.append("PyTorch with CUDA support")

    # Check diffusers with ZImagePipeline
    try:
        from diffusers import ZImagePipeline as _ZImagePipeline  # type: ignore[attr-defined]

        status["diffusers_installed"] = True
        status["z_image_pipeline_available"] = _ZImagePipeline is not None
    except ImportError as e:
        status["diffusers_installed"] = False
        if "ZImagePipeline" in str(e) or "cannot import name" in str(e):
            status["diffusers_installed"] = True  # diffusers installed but old version
            missing.append("diffusers with ZImagePipeline (update required)")
        else:
            missing.append("diffusers library")

    # Check other dependencies
    for dep in ["accelerate", "sentencepiece"]:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)

    status["missing"] = missing
    status["available"] = len(missing) == 0 and status["cuda_available"]

    # Generate recommendation
    if status["available"]:
        vram = status["gpu_vram_gb"]
        if vram is not None and isinstance(vram, (int, float)) and vram < 8:
            status["recommendation"] = (
                f"Z-Image ready! GPU: {status['gpu_name']} ({vram}GB). "
                "CPU offload will be auto-enabled for stability."
            )
        else:
            status["recommendation"] = (
                f"Z-Image ready! GPU: {status['gpu_name']} ({vram}GB)"
            )
    else:
        status["recommendation"] = (
            f"Z-Image not available. Missing: {', '.join(missing)}"
        )

    _z_image_status = status
    return status


def offer_z_image_install(interactive: bool = True) -> bool:
    """Check Z-Image availability and offer to install if missing.

    Args:
        interactive: If True, prompt user for installation. If False, just return status.

    Returns:
        True if Z-Image is available (or was successfully installed), False otherwise.
    """
    status = check_z_image_available()

    if status["available"]:
        logger.info(f"✓ Z-Image available: {status['recommendation']}")
        return True

    missing = status["missing"]
    if not missing:
        return True

    print("\n" + "=" * 60)
    print("🎨 Z-Image Local Image Generation")
    print("=" * 60)
    print("\nZ-Image provides high-quality local image generation.")
    print(f"Status: {'✓ Available' if status['available'] else '✗ Not configured'}")

    if status["cuda_available"]:
        print(f"GPU: {status['gpu_name']} ({status['gpu_vram_gb']}GB VRAM)")
    else:
        print("GPU: No CUDA GPU detected")

    if missing and isinstance(missing, list):
        print("\nMissing components:")
        for item in missing:
            print(f"  • {item}")

    if not interactive:
        return False

    # Offer installation
    print("\nWould you like to install Z-Image dependencies?")
    print("This will run: python setup_z_image.py install")

    try:
        response = input("\nInstall Z-Image? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        response = "n"

    if response in ("y", "yes"):
        print("\nInstalling Z-Image dependencies...")
        try:
            # Run the setup script
            setup_script = os.path.join(
                os.path.dirname(__file__), "..", "setup_z_image.py"
            )
            if os.path.exists(setup_script):
                result = subprocess.run(
                    [sys.executable, setup_script, "install"],
                    capture_output=False,
                )
                if result.returncode == 0:
                    # Clear cache and re-check
                    global _z_image_status
                    _z_image_status = {}
                    new_status = check_z_image_available()
                    if new_status["available"]:
                        print("\n✅ Z-Image installed successfully!")
                        return True
            else:
                # Fallback: install packages directly
                print("Installing required packages...")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "diffusers>=0.36.0",
                        "accelerate",
                        "sentencepiece",
                    ]
                )
                _z_image_status = {}
                return check_z_image_available()["available"]  # type: ignore[return-value]
        except Exception as e:
            print(f"Installation failed: {e}")

    return False


def get_default_provider() -> str:
    """Get the default image provider, preferring Z-Image if available.

    Returns:
        Provider name string
    """
    # Check if user explicitly set IMAGE_PROVIDER
    explicit_provider = os.getenv("IMAGE_PROVIDER", "").strip()
    if explicit_provider:
        return explicit_provider.lower()

    # Auto-detect: prefer Z-Image if CUDA is available
    status = check_z_image_available()
    if status["available"]:
        logger.info(f"Auto-selected Z-Image: {status['recommendation']}")
        return "z_image"

    # Fall back to pollinations (free, no API key required)
    return "pollinations"


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
        # Use environment configuration (auto-detects Z-Image if CUDA available)
        provider = get_image_provider()
        image_bytes = provider.generate("A sunset over mountains")

        # Override specific settings
        provider = get_image_provider(provider_name="pollinations", model="flux-anime")
    """
    # Read configuration with overrides - use auto-detection for default
    provider = provider_name or get_default_provider()
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

    if provider == "google_imagen":
        from .google_imagen import GoogleImagenProvider

        api_key = os.getenv("GOOGLE_API_KEY", "")
        aspect_ratio = os.getenv("IMAGE_ASPECT_RATIO", "1:1")

        return GoogleImagenProvider(
            model=model,
            size=size,
            api_key=api_key,
            aspect_ratio=aspect_ratio,
            timeout=timeout,
        )

    if provider == "openai_dalle":
        from .openai_dalle import OpenAIDalleProvider

        api_key = os.getenv("OPENAI_API_KEY", "")
        quality = os.getenv("DALLE_QUALITY", "standard")
        style = os.getenv("DALLE_STYLE", "natural")

        return OpenAIDalleProvider(
            model=model,
            size=size,
            api_key=api_key,
            quality=quality,
            style=style,
            timeout=timeout,
        )

    if provider == "z_image":
        from .z_image import ZImageProvider

        num_steps = int(os.getenv("Z_IMAGE_STEPS", "28"))
        guidance = float(os.getenv("Z_IMAGE_GUIDANCE", "4.0"))
        device = os.getenv("Z_IMAGE_DEVICE", "cuda")
        # Support 'none', 'model', 'sequential' (legacy '1'/'0' still works)
        offload_env = os.getenv("Z_IMAGE_OFFLOAD", "0")
        if offload_env in ("1", "sequential"):
            enable_offload = True
            offload_mode = "sequential"
        elif offload_env == "model":
            enable_offload = True
            offload_mode = "model"
        else:
            enable_offload = False
            offload_mode = "none"
        dtype = os.getenv("Z_IMAGE_DTYPE", "bfloat16")
        cache_dir = os.getenv("Z_IMAGE_CACHE_DIR", None)
        negative_prompt = os.getenv(
            "Z_IMAGE_NEGATIVE_PROMPT",
            "text, watermark, logo, blurry, low quality, artifacts, jpeg artifacts",
        )
        # Override size with Z-Image-specific size if set
        z_image_size = os.getenv("Z_IMAGE_SIZE", "")
        if z_image_size:
            size = z_image_size
        # Local generation can take a while, use longer timeout
        z_image_timeout = max(timeout, 300)

        return ZImageProvider(
            model=model,
            size=size,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            negative_prompt=negative_prompt,
            device=device,
            enable_cpu_offload=enable_offload,
            offload_mode=offload_mode,
            dtype=dtype,
            cache_dir=cache_dir,
            timeout=z_image_timeout,
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
        elif name == "google_imagen":
            info["requires"] = ["GOOGLE_API_KEY"]
            info["configured"] = bool(os.getenv("GOOGLE_API_KEY"))
        elif name == "openai_dalle":
            info["requires"] = ["OPENAI_API_KEY"]
            info["configured"] = bool(os.getenv("OPENAI_API_KEY"))
        elif name == "z_image":
            info["requires"] = []  # No API key - runs locally
            info["notes"] = "Requires CUDA GPU with 8GB+ VRAM"
            # Check if torch and CUDA are available
            try:
                import torch

                info["configured"] = torch.cuda.is_available()
                if info["configured"]:
                    info["gpu"] = torch.cuda.get_device_name(0)
            except ImportError:
                info["configured"] = False
                info["notes"] = "Requires: pip install torch torchvision"

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
