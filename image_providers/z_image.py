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
    Z_IMAGE_COMPILE: Set to '0' to disable torch.compile (default: '1' for CUDA)
    Z_IMAGE_WARMUP: Set to '0' to disable warmup generation (default: '1' for <6GB VRAM)
    Z_IMAGE_QUANTIZE: Set to '1' to enable INT8 quantization (default: '0', experimental)

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
        offload_mode: str = "none",
        dtype: str = "bfloat16",
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
            enable_cpu_offload: Enable CPU offload for low VRAM
            offload_mode: CPU offload strategy ('none', 'model', 'sequential')
            dtype: Data type for model weights ('float16', 'bfloat16', 'float32')
            cache_dir: Directory to cache the downloaded model
            timeout: Timeout in seconds (mainly for download)
        """
        super().__init__(model, size, timeout)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.default_negative_prompt = negative_prompt
        self.device = device
        self.enable_cpu_offload = enable_cpu_offload
        self.offload_mode = offload_mode
        self.dtype = dtype
        self.cache_dir = cache_dir
        self._pipe = None

        # Auto-adjust settings based on available VRAM
        self._auto_adjust_for_vram()

    def _auto_adjust_for_vram(self):
        """Auto-adjust settings based on available VRAM."""
        try:
            import torch
            if self.device == "cuda" and torch.cuda.is_available():
                _vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # No auto-adjustment needed — use official recommended guidance_scale=4.0
        except Exception:
            pass  # Silently ignore if can't detect

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

    def _quantize_model(self, pipe, vram_gb: float):
        """Apply INT8 quantization for extreme low VRAM scenarios.

        This is experimental and should only be used on 4GB or less VRAM.
        Enable by setting Z_IMAGE_QUANTIZE=1 environment variable.

        Args:
            pipe: The pipeline to quantize
            vram_gb: Available VRAM in GB

        Returns:
            The pipeline (potentially quantized)
        """
        if vram_gb > 4:
            return pipe  # Skip quantization for >4GB

        try:
            # Try optimum-quanto for INT8 quantization
            from optimum.quanto import quantize, freeze  # type: ignore[import-unresolved]
            import torch

            logger.info("Applying INT8 quantization (experimental)...")

            # Quantize transformer to INT8
            quantize(pipe.transformer, weights=torch.int8, activations=None)
            freeze(pipe.transformer)

            logger.info("INT8 quantization applied to transformer")
            return pipe

        except ImportError:
            logger.debug("optimum-quanto not available, skipping quantization")
            logger.debug("Install with: pip install optimum-quanto")
            return pipe
        except Exception as e:
            logger.warning(f"Quantization failed: {e}, continuing without it")
            return pipe

    def _pin_offload_weights(self, pipe):
        """Pin offloaded weights in page-locked RAM for faster CPU→GPU DMA transfers.

        When using sequential CPU offload, model weights live in a CPU dict and
        get transferred to GPU layer-by-layer during each denoising step. By default
        these are in pageable RAM, requiring an extra CPU-side copy before DMA.
        Pinning puts them in page-locked (non-swappable) memory, allowing the GPU
        to DMA directly — measured ~20% per-step speedup on 4GB VRAM.

        One-time cost: ~8-10s for pinning ~11.7GB of transformer + text_encoder weights.
        """
        try:
            import torch
            from accelerate.hooks import AlignDevicesHook

            pinned_total = 0
            for comp_name in ("transformer", "text_encoder"):
                comp = getattr(pipe, comp_name, None)
                if comp is None:
                    continue

                # Find the shared root weight dict used by all offload hooks
                root_dict = None
                for _name, mod in comp.named_modules():
                    hook = getattr(mod, "_hf_hook", None)
                    if (
                        hook
                        and isinstance(hook, AlignDevicesHook)
                        and getattr(hook, "offload", False)
                        and hook.weights_map is not None
                    ):
                        wm = hook.weights_map
                        root_dict = wm.dataset if hasattr(wm, "dataset") else wm
                        break

                if root_dict is None:
                    continue

                count = 0
                for k in root_dict:
                    v = root_dict[k]
                    if isinstance(v, torch.Tensor) and not v.is_pinned():
                        root_dict[k] = v.pin_memory()
                        count += 1
                pinned_total += count
                logger.debug(f"Pinned {count} {comp_name} weight tensors in page-locked RAM")

            if pinned_total > 0:
                logger.info(f"Pinned {pinned_total} weight tensors for faster CPU→GPU transfers")
        except Exception as e:
            logger.debug(f"Weight pinning skipped: {e}")

    def _get_pipeline(self):
        """Load or retrieve cached pipeline."""
        global _pipeline_cache

        cache_key = f"{self.model}_{self.device}_{self.enable_cpu_offload}_{self.offload_mode}_{self.dtype}"

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
        compute_cap = (0, 0)
        is_ampere_or_newer = False  # Default value
        if self.device == "cuda":
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            compute_cap = torch.cuda.get_device_capability(0)
            is_ampere_or_newer = compute_cap[0] >= 8

            # Use TOTAL VRAM for configuration decisions (more consistent)
            if vram_gb <= 4.5:
                # Very low VRAM (<=4.5GB) - use SEQUENTIAL offload
                # Transformer=11.5GB, text_encoder=7.5GB — neither fits in 4GB
                # VAE=0.16GB — easily fits, no tiling/slicing needed
                logger.info(
                    f"Detected {vram_gb:.1f}GB VRAM - using SEQUENTIAL CPU offload"
                )
                self.enable_cpu_offload = True
                self._use_sequential_offload = True
                # Attention slicing HELPS under sequential offload — reduces peak
                # VRAM per layer, avoiding memory thrashing (tested 37% faster)
                self._use_attention_slicing = True
                self._use_vae_slicing = False
                self._use_vae_tiling = False
            elif vram_gb <= 6:
                # Low VRAM (4.5-6GB) - use moderate optimizations
                logger.info(
                    f"Detected {vram_gb:.1f}GB VRAM - enabling memory optimizations"
                )
                self.enable_cpu_offload = True
                self._use_attention_slicing = True
                self._use_vae_slicing = False
                self._use_vae_tiling = False
            elif vram_gb <= 12:
                # Medium VRAM (6-12GB) - light optimizations
                self._use_attention_slicing = True
                self._use_vae_slicing = False
                self._use_vae_tiling = False
            else:
                # High VRAM (>12GB) - no optimizations needed
                self._use_attention_slicing = False
                self._use_vae_slicing = False
                self._use_vae_tiling = False

        try:
            # Select dtype based on configuration.
            # float16 is fastest on most GPUs; bfloat16 has native Ampere+ support.
            # The dtype can be configured via Z_IMAGE_DTYPE env var.
            if self.device == "cuda":
                dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }
                dtype = dtype_map.get(self.dtype, torch.bfloat16)
                if is_ampere_or_newer:
                    # TF32 gives ~10-15% speedup on matmul/conv ops
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    logger.debug(f"Using {self.dtype} + TF32 on Ampere+ GPU (compute {compute_cap[0]}.{compute_cap[1]})")
                else:
                    logger.debug(f"Using {self.dtype} on pre-Ampere GPU")
            else:
                dtype = torch.float32  # CPU fallback

            # Load pipeline — match official HuggingFace example
            pipe = ZImagePipeline.from_pretrained(
                self.model,
                torch_dtype=dtype,
                cache_dir=self.cache_dir,
            )

            # Note: Z-Image uses a custom scheduler - do not modify it
            # Note: Do NOT call .eval() on pipeline modules — Z-Image's custom
            # architecture handles inference mode internally

            # Apply memory optimizations based on VRAM and configured offload mode
            if self.enable_cpu_offload and self.device == "cuda":
                # Use configured offload mode, or auto-detect from VRAM
                if self.offload_mode == "sequential" or getattr(self, "_use_sequential_offload", False):
                    pipe.enable_sequential_cpu_offload()
                    logger.info("Z-Image loaded with SEQUENTIAL CPU offload (max memory savings)")

                    # Pin offloaded weights in page-locked RAM for faster CPU→GPU DMA transfers.
                    self._pin_offload_weights(pipe)
                else:
                    # Model CPU offload for 6-12GB
                    pipe.enable_model_cpu_offload()
                    logger.info("Z-Image loaded with model CPU offload (optimized)")
            else:
                pipe.to(self.device)
                logger.info(f"Z-Image loaded on {self.device}")

            # Attention slicing: reduces peak VRAM per attention layer.
            # On 4GB with sequential offload, this avoids memory thrashing and is
            # measurably ~37% faster. On high-VRAM GPUs, slicing adds overhead.
            if getattr(self, "_use_attention_slicing", False):
                try:
                    if vram_gb <= 4.5:
                        # slice_size=1 is most aggressive but prevents VRAM thrashing
                        pipe.enable_attention_slicing(slice_size=1)
                    elif not is_ampere_or_newer:
                        pipe.enable_attention_slicing(slice_size="auto")
                    else:
                        pipe.enable_attention_slicing(slice_size="auto")
                    logger.debug("Attention slicing enabled")
                except Exception:
                    pass

            # Enable VAE slicing/tiling only when explicitly needed
            # (VAE is only 0.16GB, so these are unnecessary on any modern GPU)
            if getattr(self, "_use_vae_slicing", False):
                try:
                    if hasattr(pipe, 'enable_vae_slicing'):
                        pipe.enable_vae_slicing()
                        logger.debug("VAE slicing enabled")
                except Exception as e:
                    logger.debug(f"VAE slicing unavailable: {e}")

            if getattr(self, "_use_vae_tiling", False):
                try:
                    if hasattr(pipe, 'enable_vae_tiling'):
                        pipe.enable_vae_tiling()
                        logger.debug("VAE tiling enabled")
                except Exception as e:
                    logger.debug(f"VAE tiling unavailable: {e}")

            # Enable memory-efficient attention
            # For Ampere+ GPUs (compute 8.0+), use native SDPA (fastest)
            # For older GPUs, try xformers or fall back to standard
            if is_ampere_or_newer:
                try:
                    # Native scaled dot product attention (PyTorch 2.0+, fastest on Ampere)
                    from diffusers.models.attention_processor import AttnProcessor2_0
                    pipe.transformer.set_attn_processor(AttnProcessor2_0())
                    logger.debug("Native SDPA attention enabled (Ampere optimized)")
                except Exception:
                    pass  # Fall back to default
            else:
                try:
                    # Try xformers for older GPUs
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.debug("xformers memory-efficient attention enabled")
                except Exception:
                    pass  # Continue without optimized attention

            # cuDNN benchmarking — speeds up conv operations for consistent input sizes
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True

            # Apply INT8 quantization if requested (experimental, for 4GB VRAM)
            enable_quantize = os.getenv("Z_IMAGE_QUANTIZE", "0") == "1"
            if enable_quantize and vram_gb <= 4.5:
                pipe = self._quantize_model(pipe, vram_gb)

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

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> bytes:
        """Generate an image using Z-Image locally.

        Args:
            prompt: Text description of the image to generate
            negative_prompt: Optional negative prompt (uses default if not provided)
            seed: Optional seed for reproducible generation

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

            # Build generator — official example uses torch.Generator("cuda")
            # For sequential CPU offload, latents start on CPU so use "cpu"
            gen_seed = seed if seed is not None else int.from_bytes(os.urandom(4), byteorder="big")
            gen_device = "cpu" if getattr(self, '_use_sequential_offload', False) else self.device
            generator = torch.Generator(device=gen_device).manual_seed(gen_seed)

            # inference_mode is marginally faster than no_grad (skips version counter)
            with torch.inference_mode():
                result = pipe(  # type: ignore[operator]
                    prompt=prompt,
                    negative_prompt=neg_prompt or "",
                    height=self.height,
                    width=self.width,
                    cfg_normalization=False,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                )

            image = result.images[0]  # type: ignore[union-attr]

            # Convert PIL Image to PNG bytes with fast compression
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", compress_level=1)  # Level 1 = fast
            image_bytes = buffer.getvalue()

            # Free the result object immediately (can hold GPU references)
            del result, image

            logger.info(
                f"✓ Z-Image generated {len(image_bytes) // 1024}KB image "
                f"({self.width}x{self.height}), seed={gen_seed}"
            )

            # Quick cache clear for low VRAM — no sync, no GC
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return image_bytes

        except Exception as e:
            error_msg = str(e)

            if "CUDA out of memory" in error_msg:
                # Try to recover: clear cache and retry with fewer steps
                logger.warning("CUDA OOM during generation — clearing cache and retrying with fewer steps...")
                try:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                    reduced_steps = max(self.num_inference_steps // 2, 10)
                    logger.info(f"Retrying with {reduced_steps} steps (was {self.num_inference_steps})")

                    gen_seed2 = seed if seed is not None else int.from_bytes(os.urandom(4), byteorder="big")
                    gen_device2 = "cpu" if getattr(self, '_use_sequential_offload', False) else self.device
                    generator2 = torch.Generator(device=gen_device2).manual_seed(gen_seed2)

                    with torch.inference_mode():
                        result = pipe(  # type: ignore[operator]
                            prompt=prompt,
                            negative_prompt=neg_prompt or "",
                            height=self.height,
                            width=self.width,
                            cfg_normalization=False,
                            num_inference_steps=reduced_steps,
                            guidance_scale=self.guidance_scale,
                            generator=generator2,
                        )
                    image = result.images[0]  # type: ignore[union-attr]
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG", compress_level=1)
                    image_bytes = buffer.getvalue()
                    del result, image
                    logger.info(f"OOM recovery succeeded with {reduced_steps} steps")
                    torch.cuda.empty_cache()
                    return image_bytes
                except Exception:
                    pass  # Fall through to error

                raise ImageProviderError(
                    "GPU out of memory during generation. Try:\n"
                    "  1. Reduce image size\n"
                    "  2. Reduce inference steps\n"
                    "  3. Close other GPU applications",
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
