"""Configuration management for Social Media Publisher."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _get_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def _get_str(key: str, default: str = "") -> str:
    """Get string from environment variable."""
    return os.getenv(key, default)


class Config:
    """Application configuration loaded from environment variables."""

    # --- API Keys ---
    GEMINI_API_KEY: str = _get_str("GEMINI_API_KEY")
    # Support either HUGGINGFACE_API_TOKEN or HUGGINGFACE_API_KEY
    HUGGINGFACE_API_TOKEN: str = _get_str("HUGGINGFACE_API_TOKEN") or _get_str(
        "HUGGINGFACE_API_KEY"
    )
    LINKEDIN_ACCESS_TOKEN: str = _get_str("LINKEDIN_ACCESS_TOKEN")
    LINKEDIN_AUTHOR_URN: str = _get_str("LINKEDIN_AUTHOR_URN")

    # --- Local LLM (LM Studio) ---
    LM_STUDIO_BASE_URL: str = _get_str("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    LM_STUDIO_MODEL: str = _get_str("LM_STUDIO_MODEL", "local-model")
    PREFER_LOCAL_LLM: bool = _get_bool("PREFER_LOCAL_LLM", True)

    # --- AI Models ---
    MODEL_TEXT: str = _get_str("MODEL_TEXT", "gemini-2.0-flash")
    MODEL_VERIFICATION: str = _get_str("MODEL_VERIFICATION", "gemini-2.0-flash")
    MODEL_IMAGE: str = _get_str("MODEL_IMAGE", "imagen-4.0-generate-001")

    # --- Hugging Face Image Generation ---
    # Default to a broadly available SDXL model on the Inference API
    HF_TTI_MODEL: str = _get_str(
        "HF_TTI_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"
    )
    # Optional: use a dedicated Inference Endpoint URL instead of the public models route
    HF_INFERENCE_ENDPOINT: str = _get_str("HF_INFERENCE_ENDPOINT", "")
    # Optional negative prompt to reduce unwanted artifacts
    HF_NEGATIVE_PROMPT: str = _get_str(
        "HF_NEGATIVE_PROMPT",
        "text, watermark, logo, blurry, low quality, artifacts, jpeg artifacts, nsfw",
    )
    # Automatically prefer Hugging Face for images when a token is present
    HF_PREFER_IF_CONFIGURED: bool = _get_bool("HF_PREFER_IF_CONFIGURED", True)

    # --- Image Style Settings ---
    # Style directive for image generation prompts
    IMAGE_STYLE: str = _get_str(
        "IMAGE_STYLE",
        "photorealistic, 1960s Kodachrome film aesthetic, slightly desaturated warm "
        "tones, vintage cinema color grading, soft golden hour lighting, subtle film "
        "grain, rich shadows, muted pastel highlights",
    )

    # --- Search Settings ---
    SEARCH_PROMPT: str = _get_str(
        "SEARCH_PROMPT",
        "Find news stories about technology breakthroughs and innovations",
    )
    # Optional template for the full search instruction prompt.
    # Supports placeholders: {criteria}, {since_date}, {summary_words}
    SEARCH_PROMPT_TEMPLATE: str = _get_str("SEARCH_PROMPT_TEMPLATE", "")
    SEARCH_LOOKBACK_DAYS: int = _get_int("SEARCH_LOOKBACK_DAYS", 7)
    USE_LAST_CHECKED_DATE: bool = _get_bool("USE_LAST_CHECKED_DATE", True)
    # Maximum number of stories to find per search (default 5)
    MAX_STORIES_PER_SEARCH: int = _get_int("MAX_STORIES_PER_SEARCH", 5)
    # Similarity threshold for semantic deduplication (0.0-1.0, higher = stricter)
    DEDUP_SIMILARITY_THRESHOLD: float = float(
        _get_str("DEDUP_SIMILARITY_THRESHOLD", "0.7")
    )
    # Number of retries for transient API failures
    API_RETRY_COUNT: int = _get_int("API_RETRY_COUNT", 3)
    # Base delay for exponential backoff (seconds)
    API_RETRY_DELAY: float = float(_get_str("API_RETRY_DELAY", "2.0"))
    # Enable URL validation before saving sources
    VALIDATE_SOURCE_URLS: bool = _get_bool("VALIDATE_SOURCE_URLS", True)
    # Enable story preview mode in test search
    SEARCH_PREVIEW_MODE: bool = _get_bool("SEARCH_PREVIEW_MODE", True)

    # --- Content Settings ---
    SUMMARY_WORD_COUNT: int = _get_int("SUMMARY_WORD_COUNT", 250)
    MIN_QUALITY_SCORE: int = _get_int("MIN_QUALITY_SCORE", 7)

    # --- Publication Settings ---
    STORIES_PER_CYCLE: int = _get_int("STORIES_PER_CYCLE", 3)
    PUBLISH_WINDOW_HOURS: int = _get_int("PUBLISH_WINDOW_HOURS", 24)
    PUBLISH_START_HOUR: int = _get_int("PUBLISH_START_HOUR", 8)
    PUBLISH_END_HOUR: int = _get_int("PUBLISH_END_HOUR", 20)
    JITTER_MINUTES: int = _get_int("JITTER_MINUTES", 30)

    # --- Cleanup Settings ---
    EXCLUSION_PERIOD_DAYS: int = _get_int("EXCLUSION_PERIOD_DAYS", 30)

    # --- Signature Block ---
    SIGNATURE_BLOCK: str = _get_str("SIGNATURE_BLOCK", "\n\n#News #Update")

    # --- Paths ---
    DB_NAME: str = _get_str("DB_NAME", "content_engine.db")
    IMAGE_DIR: str = _get_str("IMAGE_DIR", "generated_images")

    # --- Cycle Timing ---
    SEARCH_CYCLE_HOURS: int = _get_int("SEARCH_CYCLE_HOURS", 24)
    PUBLISHER_CHECK_INTERVAL_SECONDS: int = _get_int(
        "PUBLISHER_CHECK_INTERVAL_SECONDS", 60
    )

    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate that required configuration is present.
        Returns a list of error messages (empty if valid).
        """
        errors = []

        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is required")

        if not cls.LINKEDIN_ACCESS_TOKEN:
            errors.append("LINKEDIN_ACCESS_TOKEN is required for publishing")

        if not cls.LINKEDIN_AUTHOR_URN:
            errors.append("LINKEDIN_AUTHOR_URN is required for publishing")

        if cls.PUBLISH_START_HOUR >= cls.PUBLISH_END_HOUR:
            errors.append("PUBLISH_START_HOUR must be less than PUBLISH_END_HOUR")

        if cls.STORIES_PER_CYCLE < 1:
            errors.append("STORIES_PER_CYCLE must be at least 1")

        if cls.SUMMARY_WORD_COUNT < 50:
            errors.append("SUMMARY_WORD_COUNT should be at least 50")

        return errors

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure required directories exist."""
        Path(cls.IMAGE_DIR).mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls) -> None:
        """Print current configuration (masking sensitive values)."""
        print("=== Social Media Publisher Configuration ===")
        print(f"  GEMINI_API_KEY: {'*' * 8 if cls.GEMINI_API_KEY else 'NOT SET'}")
        print(
            f"  HUGGINGFACE_API_TOKEN: {'*' * 8 if cls.HUGGINGFACE_API_TOKEN else 'NOT SET'}"
        )
        print(
            f"  LINKEDIN_ACCESS_TOKEN: {'*' * 8 if cls.LINKEDIN_ACCESS_TOKEN else 'NOT SET'}"
        )
        print(f"  LINKEDIN_AUTHOR_URN: {cls.LINKEDIN_AUTHOR_URN or 'NOT SET'}")
        print(f"  MODEL_TEXT: {cls.MODEL_TEXT}")
        print(f"  MODEL_IMAGE: {cls.MODEL_IMAGE}")
        if cls.HUGGINGFACE_API_TOKEN:
            print(
                f"  HF_TTI_MODEL: {cls.HF_TTI_MODEL} (prefer={cls.HF_PREFER_IF_CONFIGURED})"
            )
        print(f"  SEARCH_PROMPT: {cls.SEARCH_PROMPT[:50]}...")
        if cls.SEARCH_PROMPT_TEMPLATE:
            print("  SEARCH_PROMPT_TEMPLATE: custom (from .env)")
        print(f"  SEARCH_LOOKBACK_DAYS: {cls.SEARCH_LOOKBACK_DAYS}")
        print(f"  USE_LAST_CHECKED_DATE: {cls.USE_LAST_CHECKED_DATE}")
        print(f"  MAX_STORIES_PER_SEARCH: {cls.MAX_STORIES_PER_SEARCH}")
        print(f"  DEDUP_SIMILARITY_THRESHOLD: {cls.DEDUP_SIMILARITY_THRESHOLD}")
        print(f"  API_RETRY_COUNT: {cls.API_RETRY_COUNT}")
        print(f"  VALIDATE_SOURCE_URLS: {cls.VALIDATE_SOURCE_URLS}")
        print(f"  SEARCH_PREVIEW_MODE: {cls.SEARCH_PREVIEW_MODE}")
        print(f"  SUMMARY_WORD_COUNT: {cls.SUMMARY_WORD_COUNT}")
        print(f"  MIN_QUALITY_SCORE: {cls.MIN_QUALITY_SCORE}")
        print(f"  STORIES_PER_CYCLE: {cls.STORIES_PER_CYCLE}")
        print(f"  PUBLISH_WINDOW_HOURS: {cls.PUBLISH_WINDOW_HOURS}")
        print(f"  PUBLISH_START_HOUR: {cls.PUBLISH_START_HOUR}")
        print(f"  PUBLISH_END_HOUR: {cls.PUBLISH_END_HOUR}")
        print(f"  JITTER_MINUTES: {cls.JITTER_MINUTES}")
        print(f"  EXCLUSION_PERIOD_DAYS: {cls.EXCLUSION_PERIOD_DAYS}")
        print(f"  DB_NAME: {cls.DB_NAME}")
        print(f"  IMAGE_DIR: {cls.IMAGE_DIR}")
        print("=============================================")
