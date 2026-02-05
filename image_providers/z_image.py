"""Z-Image local image generation provider.

Uses the Tongyi-MAI/Z-Image model locally via diffusers for high-quality
text-to-image generation without requiring external API calls.

This provider runs the model on your local GPU (CUDA required).

Requirements:
    pip install git+https://github.com/huggingface/diffusers
    pip install torch torchvision accelerate transformers sentencepiece

Environment variables:
    Z_IMAGE_STEPS: Number of inference steps (default: 28, recommended: 28-50)
    Z_IMAGE_GUIDANCE: Guidance scale (default: 4.0, recommended: 3.0-5.0)
    Z_IMAGE_DEVICE: Device to use (default: 'cuda', can be 'cpu' for testing)
    Z_IMAGE_OFFLOAD: Set to '1' to enable CPU offload for low VRAM (default: '0')
    Z_IMAGE_CACHE_DIR: Directory to cache the model (optional)

Model info:
    - Resolution: 512×512 to 2048×2048 (any aspect ratio within total pixel area)
    - Supports negative prompting for artifact suppression
    - Full CFG support for precise prompt adherence
    - High diversity in composition, faces, and lighting
"""

import logging
import os
import io
from typing import Optional

from .base import ImageProvider, ImageProviderError, GenerationResult

logger = logging.getLogger(__name__)

# Global pipeline cache to avoid reloading the model
_pipeline_cache: dict[str, object] = {}


class ZImageProvider(ImageProvider):
    """Local image generation using Tongyi-MAI/Z-Image model.

    Z-Image is a high-quality foundation model supporting:
    - Photorealistic photography
    - Cinematic digital art
    - Anime and stylized illustrations
    - Complex prompt engineering with CFG
    - Negative prompting for artifact control

    Requires CUDA-capable GPU with at least 8GB VRAM (16GB+ recommended).
    For lower VRAM, enable Z_IMAGE_OFFLOAD=1 for CPU offloading.
    """

    MODEL_ID = "Tongyi-MAI/Z-Image"

    def __init__(
        self,
        model: str = "Tongyi-MAI/Z-Image",
        size: str = "1024x1024",
        num_inference_steps: int = 28,
        guidance_scale: float = 4.0,
        negative_prompt: str = "",
        device: str = "cuda",
        enable_cpu_offload: bool = False,
        cache_dir: Optional[str] = None,
        timeout: int = 300,  # Local generation can take a while
    ):
        """Initialize Z-Image provider.

        Args:
            model: Model ID (always 'Tongyi-MAI/Z-Image')
            size: Output size as 'WIDTHxHEIGHT' (e.g., '1024x1024', '1280x720')
            num_inference_steps: Number of denoising steps (28-50 recommended)
            guidance_scale: CFG scale (3.0-5.0 recommended)
            negative_prompt: Default negative prompt for all generations
            device: Device to run on ('cuda' or 'cpu')
            enable_cpu_offload: Enable sequential CPU offload for low VRAM
            cache_dir: Directory to cache the downloaded model
            timeout: Timeout in seconds (mainly for download)
        """
        super().__init__(model, size, timeout)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.default_negative_prompt = negative_prompt
        self.device = device
        self.enable_cpu_offload = enable_cpu_offload
        self.cache_dir = cache_dir
        self._pipe = None

    @property
    def name(self) -> str:
        return "Z-Image (Local)"

    @property
    def is_configured(self) -> bool:
        """Check if CUDA is available for local generation."""
        try:
            import torch

            if self.device == "cuda":
                return torch.cuda.is_available()
            return True  # CPU always available (but slow)
        except ImportError:
            return False

    def _get_pipeline(self):
        """Load or retrieve cached pipeline."""
        global _pipeline_cache

        cache_key = f"{self.model}_{self.device}_{self.enable_cpu_offload}"

        if cache_key in _pipeline_cache:
            logger.debug("Using cached Z-Image pipeline")
            return _pipeline_cache[cache_key]

        logger.info(
            "Loading Z-Image model (this may take a few minutes on first run)..."
        )

        try:
            import torch

            # ZImagePipeline is new in diffusers, may not be in type stubs yet
            from diffusers import ZImagePipeline  # type: ignore[attr-defined]
        except ImportError as e:
            raise ImageProviderError(
                "Required packages not installed. Run:\n"
                "  pip install git+https://github.com/huggingface/diffusers\n"
                "  pip install torch torchvision accelerate transformers sentencepiece",
                provider=self.name,
            ) from e

        # Check CUDA availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU (will be slow)")
            self.device = "cpu"

        # Auto-detect low VRAM and adjust strategy
        vram_gb = 0.0
        if self.device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 6:
                # Very low VRAM (<6GB) - use aggressive optimizations
                logger.info(
                    f"Detected {vram_gb:.1f}GB VRAM - using aggressive memory optimizations"
                )
                self.enable_cpu_offload = True
                self._use_attention_slicing = True
                self._use_vae_slicing = True
            elif vram_gb < 12:
                # Low VRAM (6-12GB) - use moderate optimizations
                logger.info(
                    f"Detected {vram_gb:.1f}GB VRAM - enabling memory optimizations"
                )
                self.enable_cpu_offload = True
                self._use_attention_slicing = True
                self._use_vae_slicing = False
            else:
                self._use_attention_slicing = False
                self._use_vae_slicing = False

        try:
            # Use float16 instead of bfloat16 for better compatibility/speed on some GPUs
            if self.device == "cuda":
                # Check if GPU supports bfloat16 efficiently (Ampere+)
                compute_cap = torch.cuda.get_device_capability(0)
                if compute_cap[0] >= 8:  # Ampere or newer
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16  # Faster on older GPUs
                    logger.debug("Using float16 for better performance on this GPU")
            else:
                dtype = torch.float32  # CPU doesn't support fp16/bf16 well

            # Load pipeline with optimizations
            pipe = ZImagePipeline.from_pretrained(
                self.model,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                cache_dir=self.cache_dir,
            )

            # Apply memory optimizations based on VRAM
            if self.enable_cpu_offload and self.device == "cuda":
                if vram_gb < 6:
                    # Use model CPU offload (keeps more on GPU, faster than sequential)
                    pipe.enable_model_cpu_offload()
                    logger.info("Z-Image loaded with model CPU offload (optimized)")
                else:
                    # Use sequential offload for 6-12GB
                    pipe.enable_sequential_cpu_offload()
                    logger.info("Z-Image loaded with sequential CPU offload")
            else:
                pipe.to(self.device)
                logger.info(f"Z-Image loaded on {self.device}")

            # Enable attention slicing for low VRAM
            if getattr(self, "_use_attention_slicing", False):
                try:
                    pipe.enable_attention_slicing(slice_size="auto")
                    logger.debug("Attention slicing enabled")
                except Exception:
                    pass

            # Enable VAE slicing for very low VRAM
            if getattr(self, "_use_vae_slicing", False):
                try:
                    pipe.enable_vae_slicing()
                    logger.debug("VAE slicing enabled")
                except Exception:
                    pass

            # Enable memory-efficient attention (xformers or native)
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.debug("xformers memory-efficient attention enabled")
            except Exception:
                try:
                    # Try native scaled dot product attention (PyTorch 2.0+)
                    from diffusers.models.attention_processor import AttnProcessor2_0

                    pipe.transformer.set_attn_processor(AttnProcessor2_0())
                    logger.debug("Native SDPA attention enabled")
                except Exception:
                    pass  # Continue without optimized attention

            # Enable channels_last memory format for potential speedup
            if self.device == "cuda":
                try:
                    pipe.transformer = pipe.transformer.to(
                        memory_format=torch.channels_last
                    )
                    logger.debug("Channels-last memory format enabled")
                except Exception:
                    pass

            _pipeline_cache[cache_key] = pipe
            return pipe

        except Exception as e:
            error_msg = str(e)

            if "CUDA out of memory" in error_msg:
                raise ImageProviderError(
                    "Not enough GPU memory. Try:\n"
                    "  1. Set Z_IMAGE_OFFLOAD=1 for CPU offloading\n"
                    "  2. Use a smaller image size\n"
                    "  3. Close other GPU applications",
                    provider=self.name,
                ) from e

            raise ImageProviderError(
                f"Failed to load Z-Image model: {e}",
                provider=self.name,
            ) from e

    def generate(self, prompt: str, negative_prompt: Optional[str] = None) -> bytes:
        """Generate an image using Z-Image locally.

        Args:
            prompt: Text description of the image to generate
            negative_prompt: Optional negative prompt (uses default if not provided)

        Returns:
            Raw PNG image bytes

        Raises:
            ImageProviderError: If generation fails
        """
        pipe = self._get_pipeline()

        # Use provided negative prompt or fall back to default
        neg_prompt = (
            negative_prompt
            if negative_prompt is not None
            else self.default_negative_prompt
        )

        logger.debug(
            f"Z-Image generation: {self.width}x{self.height}, "
            f"steps={self.num_inference_steps}, guidance={self.guidance_scale}"
        )

        try:
            import torch

            # Generate image (pipe is dynamically typed from diffusers)
            result = pipe(  # type: ignore[operator]
                prompt=prompt,
                negative_prompt=neg_prompt or "",
                height=self.height,
                width=self.width,
                cfg_normalization=False,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=torch.Generator(self.device).manual_seed(
                    int.from_bytes(os.urandom(4), byteorder="big")  # Random seed
                ),
            )

            image = result.images[0]  # type: ignore[union-attr]

            # Convert PIL Image to bytes
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            logger.info(
                f"✓ Z-Image generated {len(image_bytes) // 1024}KB image "
                f"({self.width}x{self.height})"
            )

            return image_bytes

        except Exception as e:
            error_msg = str(e)

            if "CUDA out of memory" in error_msg:
                raise ImageProviderError(
                    "GPU out of memory during generation. Try:\n"
                    "  1. Set Z_IMAGE_OFFLOAD=1\n"
                    "  2. Reduce image size\n"
                    "  3. Reduce Z_IMAGE_STEPS",
                    provider=self.name,
                    retryable=False,
                ) from e

            raise ImageProviderError(
                f"Z-Image generation failed: {e}",
                provider=self.name,
                retryable=True,
            ) from e

    def generate_with_result(
        self, prompt: str, negative_prompt: Optional[str] = None
    ) -> GenerationResult:
        """Generate an image and return detailed result.

        Args:
            prompt: Text description of the image to generate
            negative_prompt: Optional negative prompt

        Returns:
            GenerationResult with image bytes and metadata
        """
        import time

        start = time.time()
        image_bytes = self.generate(prompt, negative_prompt)
        elapsed_ms = int((time.time() - start) * 1000)

        return GenerationResult(
            image_bytes=image_bytes,
            format="png",
            width=self.width,
            height=self.height,
            model_used=self.model,
            generation_time_ms=elapsed_ms,
            metadata={
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "device": self.device,
                "cpu_offload": self.enable_cpu_offload,
            },
        )


def clear_pipeline_cache():
    """Clear the cached pipeline to free GPU memory."""
    global _pipeline_cache
    _pipeline_cache.clear()

    # Force garbage collection
    import gc

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    logger.info("Z-Image pipeline cache cleared")
