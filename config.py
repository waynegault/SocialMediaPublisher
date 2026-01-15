"""Configuration management for Social Media Publisher.

This module provides Pydantic-based configuration validation with:
- Type coercion and validation at startup
- Clear error messages for misconfiguration
- Environment variable loading from .env file
- Backward-compatible Config class interface
"""

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _check_venv() -> None:
    """Check if running in the correct virtual environment."""
    proj_root = Path(__file__).parent
    current_exe = Path(sys.executable).resolve()

    # Check for both common venv directory names
    venv_candidates = [(".venv", proj_root / ".venv"), ("venv", proj_root / "venv")]
    existing_venvs = [(name, path) for name, path in venv_candidates if path.exists()]

    if not existing_venvs:
        # No venv directory found, skip check
        return

    # Check if current interpreter is in ANY of the existing venvs
    for venv_name, venv_dir in existing_venvs:
        venv_scripts = venv_dir / ("Scripts" if os.name == "nt" else "bin")
        if str(current_exe).startswith(str(venv_scripts.resolve())):
            # Running in this venv - all good
            return

    # Not running in any of the existing venvs - show error
    # Use the first existing venv for the error message
    venv_name, venv_dir = existing_venvs[0]
    venv_scripts = venv_dir / ("Scripts" if os.name == "nt" else "bin")

    print("=" * 60)
    print(f"ERROR: Not running in the {venv_name} virtual environment!")
    print("=" * 60)
    print(f"\nCurrent Python: {current_exe}")
    print(f"Expected venv:  {venv_scripts}")
    print("\nTo fix, either:")
    print("  1. Activate the venv first:")
    if os.name == "nt":  # pragma: no cover
        print(f"     {venv_name}\\Scripts\\activate")
    else:  # pragma: no cover
        print(f"     source {venv_name}/bin/activate")
    print("  2. Or run directly with venv Python:")
    if os.name == "nt":  # pragma: no cover
        print(f"     {venv_name}\\Scripts\\python main.py")
    else:  # pragma: no cover
        print(f"     {venv_name}/bin/python main.py")
    print()
    sys.exit(1)


_check_venv()

load_dotenv()


# ============================================================================
# Pydantic Settings Model
# ============================================================================
class SettingsModel(BaseSettings):
    """
    Pydantic-based settings with validation.

    This model loads configuration from environment variables (with .env support)
    and validates all values at startup. Type coercion is automatic.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore unknown env vars
        validate_default=True,
    )

    # --- API Keys ---
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    huggingface_api_token: str = Field(default="", alias="HUGGINGFACE_API_TOKEN")
    huggingface_api_key: str = Field(
        default="", alias="HUGGINGFACE_API_KEY"
    )  # Alternative name
    linkedin_access_token: str = Field(default="", alias="LINKEDIN_ACCESS_TOKEN")
    linkedin_author_urn: str = Field(default="", alias="LINKEDIN_AUTHOR_URN")
    linkedin_organization_urn: str = Field(
        default="", alias="LINKEDIN_ORGANIZATION_URN"
    )
    linkedin_author_name: str = Field(default="", alias="LINKEDIN_AUTHOR_NAME")

    # --- Professional Discipline ---
    # The user's professional discipline (e.g., "chemical engineer", "software developer")
    discipline: str = Field(default="chemical engineer", alias="DISCIPLINE")

    linkedin_username: str = Field(default="", alias="LINKEDIN_USERNAME")
    linkedin_password: str = Field(default="", alias="LINKEDIN_PASSWORD")

    # --- LinkedIn Search Settings ---
    # Skip LinkedIn's direct search (subject to rate limits) and use Google/Bing instead
    skip_linkedin_direct_search: bool = Field(
        default=True, alias="SKIP_LINKEDIN_DIRECT_SEARCH"
    )

    # --- LinkedIn Voyager API (for reliable profile lookups) ---
    # These cookies can be extracted from browser dev tools after logging into LinkedIn
    linkedin_li_at: str = Field(default="", alias="LINKEDIN_LI_AT")
    linkedin_jsessionid: str = Field(default="", alias="LINKEDIN_JSESSIONID")

    # --- RapidAPI Fresh LinkedIn Data API (primary LinkedIn lookup method) ---
    # Get API key from: https://rapidapi.com/freshdata-freshdata-default/api/fresh-linkedin-profile-data
    rapidapi_key: str = Field(default="", alias="RAPIDAPI_KEY")

    # --- Local LLM (LM Studio) ---
    lm_studio_base_url: str = Field(
        default="http://localhost:1234/v1", alias="LM_STUDIO_BASE_URL"
    )
    lm_studio_model: str = Field(default="local-model", alias="LM_STUDIO_MODEL")
    prefer_local_llm: bool = Field(default=True, alias="PREFER_LOCAL_LLM")

    # --- AI Models ---
    model_text: str = Field(default="gemini-2.0-flash", alias="MODEL_TEXT")
    model_verification: str = Field(
        default="gemini-2.0-flash", alias="MODEL_VERIFICATION"
    )
    model_image: str = Field(default="imagen-4.0-generate-001", alias="MODEL_IMAGE")

    # --- Hugging Face Image Generation ---
    hf_tti_model: str = Field(
        default="black-forest-labs/FLUX.1-schnell", alias="HF_TTI_MODEL"
    )
    hf_inference_endpoint: str = Field(default="", alias="HF_INFERENCE_ENDPOINT")
    hf_negative_prompt: str = Field(
        default="text, watermark, logo, blurry, low quality, artifacts, jpeg artifacts, nsfw",
        alias="HF_NEGATIVE_PROMPT",
    )
    hf_prefer_if_configured: bool = Field(default=True, alias="HF_PREFER_IF_CONFIGURED")

    # --- Image Settings ---
    # Control whether generated images include a central human character
    # YES = Include central human character (default behavior)
    # NO = No central character; random humans only if incidental/peripheral to the scene
    human_in_image: bool = Field(default=True, alias="HUMAN_IN_IMAGE")

    image_style: str = Field(
        default=(
            "industrial engineering photography, technical documentation style, "
            "female engineer or scientist performing hands-on technical work, "
            "sharp focus on equipment and processes with worker in context, "
            "authentic PPE and workwear - hard hats, safety glasses, lab coats, coveralls, "
            "real industrial or laboratory environment with visible technical detail, "
            "natural workplace lighting supplemented by equipment glow, "
            "photorealistic, editorial quality for engineering trade publication"
        ),
        alias="IMAGE_STYLE",
    )
    image_aspect_ratio: str = Field(default="16:9", alias="IMAGE_ASPECT_RATIO")
    image_size: str = Field(default="2K", alias="IMAGE_SIZE")
    image_negative_prompt: str = Field(
        default=(
            "text, words, letters, numbers, labels, signs, writing, captions, titles, "
            "watermark, logo, blurry, low quality, artifacts, jpeg artifacts, "
            "cartoon, illustration, anime, drawing, painting, sketch, abstract, "
            "deformed, distorted, disfigured, bad anatomy, unrealistic proportions, "
            "stock photo watermark, grainy, out of focus, overexposed, underexposed"
        ),
        alias="IMAGE_NEGATIVE_PROMPT",
    )

    # --- Search Settings ---
    search_prompt: str = Field(
        default=(
            "I'm a professional {discipline}. I'm looking for the latest professional {discipline} "
            "stories I can summarise for publication on my LinkedIn profile"
        ),
        alias="SEARCH_PROMPT",
    )
    search_prompt_template: str = Field(default="", alias="SEARCH_PROMPT_TEMPLATE")
    search_lookback_days: int = Field(
        default=7, ge=1, le=365, alias="SEARCH_LOOKBACK_DAYS"
    )
    use_last_checked_date: bool = Field(default=True, alias="USE_LAST_CHECKED_DATE")
    max_stories_per_search: int = Field(
        default=5, ge=1, le=50, alias="MAX_STORIES_PER_SEARCH"
    )
    # Maximum number of people to search for LinkedIn profiles per story (reduces API calls)
    max_people_per_story: int = Field(
        default=3, ge=1, le=20, alias="MAX_PEOPLE_PER_STORY"
    )
    dedup_similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, alias="DEDUP_SIMILARITY_THRESHOLD"
    )
    dedup_all_stories_window_days: int = Field(
        default=90, ge=1, alias="DEDUP_ALL_STORIES_WINDOW_DAYS"
    )
    dedup_published_window_days: int = Field(
        default=30, ge=1, alias="DEDUP_PUBLISHED_WINDOW_DAYS"
    )
    api_retry_count: int = Field(default=3, ge=0, le=10, alias="API_RETRY_COUNT")
    api_retry_delay: float = Field(default=2.0, ge=0.0, alias="API_RETRY_DELAY")
    api_timeout_default: int = Field(default=30, ge=1, alias="API_TIMEOUT_DEFAULT")
    llm_local_timeout: int = Field(default=120, ge=1, alias="LLM_LOCAL_TIMEOUT")
    validate_source_urls: bool = Field(default=True, alias="VALIDATE_SOURCE_URLS")
    search_preview_mode: bool = Field(default=True, alias="SEARCH_PREVIEW_MODE")
    duckduckgo_max_results: int = Field(
        default=10, ge=1, le=100, alias="DUCKDUCKGO_MAX_RESULTS"
    )
    llm_max_output_tokens: int = Field(
        default=8192, ge=1, alias="LLM_MAX_OUTPUT_TOKENS"
    )

    # --- Content Settings ---
    summary_word_count: int = Field(
        default=250, ge=50, le=1000, alias="SUMMARY_WORD_COUNT"
    )
    min_quality_score: int = Field(default=7, ge=1, le=10, alias="MIN_QUALITY_SCORE")

    # --- Quality Score Calibration ---
    quality_weight_recency: float = Field(
        default=1.0, ge=0.0, alias="QUALITY_WEIGHT_RECENCY"
    )
    quality_weight_source: float = Field(
        default=1.0, ge=0.0, alias="QUALITY_WEIGHT_SOURCE"
    )
    quality_weight_relevance: float = Field(
        default=1.0, ge=0.0, alias="QUALITY_WEIGHT_RELEVANCE"
    )
    quality_weight_people_mentioned: float = Field(
        default=0.5, ge=0.0, alias="QUALITY_WEIGHT_PEOPLE_MENTIONED"
    )
    quality_weight_geographic: float = Field(
        default=0.5, ge=0.0, alias="QUALITY_WEIGHT_GEOGRAPHIC"
    )
    quality_max_calibration_bonus: int = Field(
        default=2, ge=0, alias="QUALITY_MAX_CALIBRATION_BONUS"
    )

    # --- Originality Checking ---
    originality_max_similarity: float = Field(
        default=0.6, ge=0.0, le=1.0, alias="ORIGINALITY_MAX_SIMILARITY"
    )
    originality_max_ngram_overlap: float = Field(
        default=0.4, ge=0.0, le=1.0, alias="ORIGINALITY_MAX_NGRAM_OVERLAP"
    )
    originality_check_enabled: bool = Field(
        default=True, alias="ORIGINALITY_CHECK_ENABLED"
    )

    # --- Source Verification ---
    min_sources_required: int = Field(
        default=1, ge=0, le=10, alias="MIN_SOURCES_REQUIRED"
    )
    min_source_credibility: float = Field(
        default=0.3, ge=0.0, le=1.0, alias="MIN_SOURCE_CREDIBILITY"
    )
    require_tier1_or_2_source: bool = Field(
        default=False, alias="REQUIRE_TIER1_OR_2_SOURCE"
    )
    source_verification_enabled: bool = Field(
        default=True, alias="SOURCE_VERIFICATION_ENABLED"
    )

    # --- URL Archiving ---
    archive_source_urls: bool = Field(default=False, alias="ARCHIVE_SOURCE_URLS")

    # --- Publication Settings ---
    max_stories_per_day: int = Field(
        default=4, ge=1, le=20, alias="MAX_STORIES_PER_DAY"
    )
    start_pub_time: str = Field(default="08:00", alias="START_PUB_TIME")
    end_pub_time: str = Field(default="20:00", alias="END_PUB_TIME")
    jitter_minutes: int = Field(default=30, ge=0, le=60, alias="JITTER_MINUTES")
    include_opportunity_message: bool = Field(
        default=True, alias="INCLUDE_OPPORTUNITY_MESSAGE"
    )

    # --- LinkedIn Post Thresholds ---
    linkedin_post_min_chars: int = Field(
        default=100, ge=0, alias="LINKEDIN_POST_MIN_CHARS"
    )
    linkedin_post_optimal_chars: int = Field(
        default=1500, ge=100, alias="LINKEDIN_POST_OPTIMAL_CHARS"
    )
    linkedin_post_max_chars: int = Field(
        default=3000, ge=100, alias="LINKEDIN_POST_MAX_CHARS"
    )
    linkedin_max_hashtags: int = Field(
        default=5, ge=0, le=30, alias="LINKEDIN_MAX_HASHTAGS"
    )

    # --- Cleanup Settings ---
    exclusion_period_days: int = Field(default=30, ge=1, alias="EXCLUSION_PERIOD_DAYS")
    image_regen_cutoff_days: int = Field(
        default=14, ge=1, alias="IMAGE_REGEN_CUTOFF_DAYS"
    )

    # --- Signature Details ---
    signature_block_detail: str = Field(default="", alias="SIGNATURE_BLOCK_DETAIL")

    # --- Paths ---
    db_name: str = Field(default="content_engine.db", alias="DB_NAME")
    image_dir: str = Field(default="generated_images", alias="IMAGE_DIR")

    # --- Cycle Timing ---
    search_cycle_hours: int = Field(default=24, ge=1, alias="SEARCH_CYCLE_HOURS")
    publisher_check_interval_seconds: int = Field(
        default=60, ge=1, alias="PUBLISHER_CHECK_INTERVAL_SECONDS"
    )

    # --- Prompt Templates (loaded separately for brevity) ---
    # These are loaded from env vars but validated as non-empty strings
    image_refinement_prompt: str = Field(default="", alias="IMAGE_REFINEMENT_PROMPT")
    image_fallback_prompt: str = Field(default="", alias="IMAGE_FALLBACK_PROMPT")
    search_instruction_prompt: str = Field(
        default="", alias="SEARCH_INSTRUCTION_PROMPT"
    )
    verification_prompt: str = Field(default="", alias="VERIFICATION_PROMPT")
    search_distill_prompt: str = Field(default="", alias="SEARCH_DISTILL_PROMPT")
    local_llm_search_prompt: str = Field(default="", alias="LOCAL_LLM_SEARCH_PROMPT")
    linkedin_mention_prompt: str = Field(default="", alias="LINKEDIN_MENTION_PROMPT")
    story_enrichment_prompt: str = Field(default="", alias="STORY_ENRICHMENT_PROMPT")
    linkedin_profile_search_prompt: str = Field(
        default="", alias="LINKEDIN_PROFILE_SEARCH_PROMPT"
    )
    indirect_people_prompt: str = Field(default="", alias="INDIRECT_PEOPLE_PROMPT")
    company_mention_prompt: str = Field(default="", alias="COMPANY_MENTION_PROMPT")
    individual_extraction_prompt: str = Field(
        default="", alias="INDIVIDUAL_EXTRACTION_PROMPT"
    )
    json_repair_prompt: str = Field(default="", alias="JSON_REPAIR_PROMPT")

    @field_validator("image_aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        """Validate aspect ratio is one of the supported values."""
        valid_ratios = {"1:1", "16:9", "9:16", "4:3", "3:4"}
        if v not in valid_ratios:
            raise ValueError(
                f"IMAGE_ASPECT_RATIO must be one of {valid_ratios}, got '{v}'"
            )
        return v

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, v: str) -> str:
        """Validate image size is 1K or 2K."""
        if v not in {"1K", "2K"}:
            raise ValueError(f"IMAGE_SIZE must be '1K' or '2K', got '{v}'")
        return v

    @field_validator("start_pub_time", "end_pub_time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Validate time format is HH:MM."""
        try:
            parts = v.split(":")
            if len(parts) != 2:
                raise ValueError("Invalid format")
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError("Out of range")
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Time must be in HH:MM format (00:00-23:59), got '{v}': {e}"
            ) from e
        return v

    @model_validator(mode="after")
    def validate_time_range(self) -> "SettingsModel":
        """Validate START_PUB_TIME is before END_PUB_TIME."""
        start_hour = int(self.start_pub_time.split(":")[0])
        end_hour = int(self.end_pub_time.split(":")[0])
        if start_hour >= end_hour:
            raise ValueError(
                f"START_PUB_TIME ({self.start_pub_time}) must be before "
                f"END_PUB_TIME ({self.end_pub_time})"
            )
        return self

    @model_validator(mode="after")
    def validate_linkedin_thresholds(self) -> "SettingsModel":
        """Validate LinkedIn post thresholds are in correct order."""
        if self.linkedin_post_min_chars >= self.linkedin_post_optimal_chars:
            raise ValueError(
                f"LINKEDIN_POST_MIN_CHARS ({self.linkedin_post_min_chars}) must be less than "
                f"LINKEDIN_POST_OPTIMAL_CHARS ({self.linkedin_post_optimal_chars})"
            )
        if self.linkedin_post_optimal_chars >= self.linkedin_post_max_chars:
            raise ValueError(
                f"LINKEDIN_POST_OPTIMAL_CHARS ({self.linkedin_post_optimal_chars}) must be "
                f"less than LINKEDIN_POST_MAX_CHARS ({self.linkedin_post_max_chars})"
            )
        return self

    @property
    def effective_huggingface_token(self) -> str:
        """Get the effective Hugging Face token (supporting both env var names)."""
        return self.huggingface_api_token or self.huggingface_api_key


# Create the settings instance (validates at import time)
try:
    _settings = SettingsModel()
except Exception as e:
    print("=" * 60)
    print("ERROR: Configuration validation failed!")
    print("=" * 60)
    print(f"\n{e}\n")
    print("Please check your .env file and fix the configuration.")
    print("=" * 60)
    sys.exit(1)


# ============================================================================
# Legacy Helper Functions (kept for backward compatibility)
# ============================================================================
def _get_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
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
    """
    Application configuration loaded from environment variables.

    NOTE: Configuration is now validated at import time using Pydantic.
    Invalid configuration will cause the application to exit immediately
    with clear error messages.

    This class provides backward-compatible access to configuration values
    via class attributes. The underlying validated settings are in `_settings`.
    """

    # Access to the validated Pydantic settings model
    _pydantic_settings: SettingsModel = _settings

    # --- API Keys ---
    GEMINI_API_KEY: str = _get_str("GEMINI_API_KEY")
    # Support either HUGGINGFACE_API_TOKEN or HUGGINGFACE_API_KEY
    HUGGINGFACE_API_TOKEN: str = _get_str("HUGGINGFACE_API_TOKEN") or _get_str(
        "HUGGINGFACE_API_KEY"
    )
    LINKEDIN_ACCESS_TOKEN: str = _get_str("LINKEDIN_ACCESS_TOKEN")
    LINKEDIN_AUTHOR_URN: str = _get_str("LINKEDIN_AUTHOR_URN")
    # Optional org URN to post as a company (e.g. urn:li:organization:123456)
    LINKEDIN_ORGANIZATION_URN: str = _get_str("LINKEDIN_ORGANIZATION_URN")
    # Author's display name for first-person story writing (e.g., "Wayne Gault")
    LINKEDIN_AUTHOR_NAME: str = _get_str("LINKEDIN_AUTHOR_NAME", "")

    # --- Professional Discipline ---
    # The user's professional discipline (e.g., "chemical engineer", "software developer")
    DISCIPLINE: str = _get_str("DISCIPLINE", "chemical engineer")

    # LinkedIn browser login credentials (for profile URN extraction)
    # Note: Uses lowercase keys to match .env file format
    LINKEDIN_USERNAME: str = _get_str("linkedin_username", "")
    LINKEDIN_PASSWORD: str = _get_str("linkedin_password", "")

    # --- LinkedIn Voyager API (internal API using browser cookies) ---
    # Extract these from browser dev tools after logging into LinkedIn
    LINKEDIN_LI_AT: str = _get_str("LINKEDIN_LI_AT", "")
    LINKEDIN_JSESSIONID: str = _get_str("LINKEDIN_JSESSIONID", "")

    # --- LinkedIn Search Settings ---
    # Skip LinkedIn's direct search (subject to rate limits) and use Google/Bing instead
    SKIP_LINKEDIN_DIRECT_SEARCH: bool = _get_bool("SKIP_LINKEDIN_DIRECT_SEARCH", True)

    # --- RapidAPI Fresh LinkedIn Data API ---
    # Primary method for reliable LinkedIn profile lookups (90%+ success rate)
    # Get API key from: https://rapidapi.com/freshdata-freshdata-default/api/fresh-linkedin-profile-data
    RAPIDAPI_KEY: str = _get_str("RAPIDAPI_KEY", "")

    # --- Local LLM (LM Studio) ---
    LM_STUDIO_BASE_URL: str = _get_str("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    LM_STUDIO_MODEL: str = _get_str("LM_STUDIO_MODEL", "local-model")
    PREFER_LOCAL_LLM: bool = _get_bool("PREFER_LOCAL_LLM", True)

    # --- AI Models ---
    MODEL_TEXT: str = _get_str("MODEL_TEXT", "gemini-2.0-flash")
    MODEL_VERIFICATION: str = _get_str("MODEL_VERIFICATION", "gemini-2.0-flash")
    MODEL_IMAGE: str = _get_str("MODEL_IMAGE", "imagen-4.0-generate-001")

    # --- Hugging Face Image Generation ---
    # Default to FLUX.1-schnell - a FREE, fast model via InferenceClient
    HF_TTI_MODEL: str = _get_str("HF_TTI_MODEL", "black-forest-labs/FLUX.1-schnell")
    # Optional: use a dedicated Inference Endpoint URL instead of the public models route
    HF_INFERENCE_ENDPOINT: str = _get_str("HF_INFERENCE_ENDPOINT", "")
    # Optional negative prompt to reduce unwanted artifacts (not used by FLUX models)
    HF_NEGATIVE_PROMPT: str = _get_str(
        "HF_NEGATIVE_PROMPT",
        "text, watermark, logo, blurry, low quality, artifacts, jpeg artifacts, nsfw",
    )
    # Automatically prefer Hugging Face for images when a token is present
    HF_PREFER_IF_CONFIGURED: bool = _get_bool("HF_PREFER_IF_CONFIGURED", True)

    # --- Image Style Settings ---
    # Style directive for image generation prompts - professional field photography
    IMAGE_STYLE: str = _get_str(
        "IMAGE_STYLE",
        "documentary-style professional photography, authentic real-world setting for the subject matter, "
        "female professional actively working, clear view of field-specific tools/subjects, "
        "accurate PPE or professional attire appropriate to the field, "
        "natural workplace lighting with contextual highlights, "
        "photorealistic, editorial quality suitable for a professional publication",
    )
    # Aspect ratio for generated images (options: 1:1, 16:9, 9:16, 4:3, 3:4)
    IMAGE_ASPECT_RATIO: str = _get_str("IMAGE_ASPECT_RATIO", "16:9")
    # Image resolution size ("1K" or "2K") - higher = better quality but slower
    IMAGE_SIZE: str = _get_str("IMAGE_SIZE", "2K")
    # Control whether generated images include a central human character
    # True = Include central human character (default behavior)
    # False = No central character; random humans only if incidental/peripheral
    HUMAN_IN_IMAGE: bool = _get_bool("HUMAN_IN_IMAGE", True)
    # Negative prompt - describes what to AVOID in generated images
    IMAGE_NEGATIVE_PROMPT: str = _get_str(
        "IMAGE_NEGATIVE_PROMPT",
        "text, words, letters, numbers, labels, signs, writing, captions, titles, "
        "watermark, logo, blurry, low quality, artifacts, jpeg artifacts, "
        "cartoon, illustration, anime, drawing, painting, sketch, abstract, "
        "deformed, distorted, disfigured, bad anatomy, unrealistic proportions, "
        "stock photo watermark, grainy, out of focus, overexposed, underexposed",
    )

    # --- Prompt Templates ---
    # Image refinement prompt - sent to LLM to generate image generation prompts
    # Placeholders: {story_title}, {story_summary}, {image_style}
    IMAGE_REFINEMENT_PROMPT: str = _get_str(
        "IMAGE_REFINEMENT_PROMPT",
        """You are creating an image for a professional {discipline} publication.

STORY TO ILLUSTRATE:
- Title: {story_title}
- Summary: {story_summary}

CRITICAL: YOUR IMAGE MUST DIRECTLY ILLUSTRATE THIS SPECIFIC STORY
Do NOT create a generic scene. Your image must show:
- The exact subject or activity described (technology, patient, animal, experiment, field site, instrument, product, facility)
- If the story mentions a specific organization, setting, device, species, or process — SHOW THAT
- Match the environment to the story (clinic, lab, field site, office, control room, classroom, facility, etc.)

READ THE SOURCE ARTICLE CONTEXT ABOVE (if provided) for additional visual details.
This ensures your image reflects the real story, not a generic illustration.

STEP 1 - EXTRACT THE KEY VISUAL ELEMENTS FROM SOURCE:
Identify the specific subjects, people, equipment, animals, patients, locations, materials, or processes mentioned.

STEP 2 - CREATE A SCENE THAT DIRECTLY DEPICTS THE STORY:
- Research breakthrough → show the experiment setup, samples, or instruments in use
- Clinical/animal care → show practitioner with patient/animal, relevant tools, and setting
- Field work → show on-site inspection/measurement with the environment visible
- Operations/manufacturing → show the relevant line, machinery, or workflow step being discussed
- If the source describes colors, equipment types, or context details, include them

STEP 3 - SUGGEST AUTHENTIC SETTING:
- Choose a setting appropriate to the field: clinic/hospital, laboratory, field site, office/control room, classroom, industrial or agricultural facility, etc.
- Avoid explicit logos/branding; keep the look credible for the discipline and story

STEP 4 - ADD A PROMINENT HUMAN ELEMENT (MINIMUM 40% of image):
- Include a beautiful female {discipline} professional as the PROMINENT SUBJECT occupying 40-50% of the frame
- Position her CENTER-LEFT or CENTER-RIGHT (not at the far edges)
- Frame waist-up or closer; show her actively working on the story’s subject (examining, operating, sampling, analyzing, reviewing data)
- Face clearly visible with confident, warm expression; vary hair color/features across images
- Authentic PPE or professional attire appropriate to the setting (scrubs, lab coat, safety gear, business attire, field gear, etc.)

TECHNICAL ACCURACY IS CRITICAL:
- Use specific details from the story: name the subject/equipment/species/process shown
- Match the environment to what the story implies
- Include relevant instruments, tools, or context objects; avoid invented or generic props

COMPOSITION RULE (STRICT - MINIMUM 40% HUMAN):
- Female professional: 40-50% of the frame, positioned center-left or center-right
- Subject matter shares the frame (equipment/patient/animal/data/setting) and is clearly visible
- Medium or medium close-up framing; show active interaction with the subject

AVOID:
- Generic backgrounds that could apply to any story
- Vague descriptions like "technical equipment" without specifics
- Putting the person tiny or at the far edge of the frame
- Full-body distant shots; backs of heads; hidden faces
- Settings or tools that do not match the story
- Explicit company logos or trademarks

CRITICAL - NO TEXT IN IMAGE:
- NEVER include any text, words, labels, signs, or writing in the image
- Describe controls/indicators by appearance, not by text labels

GOOD IMAGE PROMPT EXAMPLE (story-specific):
"A photo of a beautiful brunette female {discipline} professional, framed from waist up in center-right, gently examining a patient/subject relevant to the story, confident warm expression, field-appropriate tools visible behind her shoulder, she occupies 45% of the frame, shot with 85mm lens, natural workplace lighting, editorial quality"

STYLE: {image_style}

CRITICAL OUTPUT FORMAT:
- MUST start with "A photo of..." (this triggers photorealistic rendering)
- Write ONLY the image prompt. Maximum 80 words.
- NEVER include any text, labels, signs, or writing
- Describe visual elements only; no readable text
- End with photography/camera style keywords like "shot with professional camera, editorial quality"
- The prompt MUST describe the specific subject/process from the story, not a generic scene.""",
    )

    # Image refinement prompt for NO HUMAN mode - focus on technology/environments
    # Placeholders: {story_title}, {story_summary}, {image_style}, {discipline}
    IMAGE_REFINEMENT_PROMPT_NO_HUMAN: str = _get_str(
        "IMAGE_REFINEMENT_PROMPT_NO_HUMAN",
        """You are creating an image for a professional {discipline} publication.

STORY TO ILLUSTRATE:
- Title: {story_title}
- Summary: {story_summary}

CRITICAL: YOUR IMAGE MUST DIRECTLY ILLUSTRATE THIS SPECIFIC STORY
Do NOT create a generic scene. Your image must show:
- The exact subject or activity described (technology, equipment, experiment, facility, environment, product)
- If the story mentions a specific organization, setting, device, species, or process — SHOW THAT
- Match the environment to the story (lab, factory, field site, office, control room, facility, etc.)

READ THE SOURCE ARTICLE CONTEXT ABOVE (if provided) for additional visual details.

STEP 1 - EXTRACT THE KEY VISUAL ELEMENTS FROM SOURCE:
Identify the specific equipment, technology, environments, materials, or processes mentioned.

STEP 2 - CREATE A SCENE THAT DIRECTLY DEPICTS THE STORY:
- Research breakthrough → show the experiment setup, samples, or instruments
- Manufacturing/industry → show the machinery, production line, or products
- Field work → show the environment, site, or natural phenomena
- Technology → show the devices, systems, or infrastructure

STEP 3 - SUGGEST AUTHENTIC SETTING:
- Choose a setting appropriate to the field and story
- Avoid explicit logos/branding; keep the look credible

CRITICAL - NO PEOPLE AS CENTRAL SUBJECT:
- DO NOT include any person as the main subject of the image
- If people appear, they must be incidental background elements (small, distant, silhouettes)
- Focus entirely on EQUIPMENT, TECHNOLOGY, ENVIRONMENTS, or PROCESSES
- The subject matter should be a thing or place, NOT a person

TECHNICAL ACCURACY IS CRITICAL:
- Use specific details from the story
- Match the environment to what the story implies
- Include relevant equipment, machinery, or context objects

AVOID:
- Generic backgrounds
- Any person as the central or prominent figure
- Portrait-style compositions
- Close-up or waist-up shots of people
- Explicit company logos or trademarks

CRITICAL - NO TEXT IN IMAGE:
- NEVER include any text, words, labels, signs, or writing in the image

GOOD IMAGE PROMPT EXAMPLE:
"A photo of industrial laboratory equipment with precision instruments and sample containers on a clean workbench, modern research facility environment with soft ambient lighting, wide shot showing the full experimental setup, professional documentary photography style, editorial quality"

STYLE: {image_style}

CRITICAL OUTPUT FORMAT:
- MUST start with "A photo of..." (this triggers photorealistic rendering)
- Write ONLY the image prompt. Maximum 80 words.
- NEVER include any text, labels, signs, or writing
- NO PEOPLE as the main subject
- End with photography/camera style keywords like "shot with professional camera, editorial quality"
- The prompt MUST describe the specific subject/process from the story, not a generic scene.""",
    )

    # Fallback image prompt template when LLM refinement fails
    # Placeholders: {story_title}, {appearance}, {discipline}
    IMAGE_FALLBACK_PROMPT: str = _get_str(
        "IMAGE_FALLBACK_PROMPT",
        "A photo of {appearance} {discipline} professional, framed from waist up in center-right of image, "
        "actively working with the subject matter related to: {story_title}. "
        "She occupies 45% of the frame with face clearly visible, confident warm expression. "
        "Relevant tools, subjects, or setting visible behind her shoulder. Authentic workplace setting appropriate to {discipline} (clinic, lab, field, office, facility). "
        "Shot with 85mm lens, natural workplace lighting, editorial quality for a professional publication.",
    )

    # Search instruction prompt - the system prompt for story search
    # Placeholders: {max_stories}, {search_prompt}, {since_date}, {summary_words}, {author_name}
    SEARCH_INSTRUCTION_PROMPT: str = _get_str(
        "SEARCH_INSTRUCTION_PROMPT",
        """You are writing AS {author_name}, a {discipline} professional sharing insights on LinkedIn. Find EXACTLY {max_stories} recent news stories matching: "{search_prompt}"

CRITICAL - STORY COUNT:
- Return EXACTLY {max_stories} stories - no more, no fewer
- If you find more candidates, select only the {max_stories} highest quality ones
- Do NOT return extra stories "just in case"

REQUIREMENTS:
- Stories must be from after {since_date}
- Each story needs:
    * title: An informative, credible headline (avoid clickbait). Use FULL institution names, NOT abbreviations:
      - "University of California, Riverside" NOT "UC Riverside" or "UCR"
      - "University of California, Los Angeles" NOT "UCLA"
      - "University of Southern California" NOT "USC"
      - "University of Chicago" NOT "UChicago"
      - "Massachusetts Institute of Technology" NOT "MIT"
      - "California Institute of Technology" NOT "Caltech"
      - Apply this rule to ALL universities and institutions
    * sources: Array of REAL source URLs from your search results
    * summary: {summary_words} words max, written in FIRST PERSON as {author_name}
    * category: One of: Clinical/Practice, Research, Technology, Business, Policy/Regulation, Education, Science, Other
    * quality_score: 1-10 rating (see scoring rubric below)
    * quality_justification: Brief explanation of the score
    * hashtags: Array of 1-3 relevant hashtags (without # symbol)
    * direct_people: Array of people objects (see format below) - MAX 3 most relevant people per story

QUALITY SCORE RUBRIC:
- 10: Breakthrough with major implications for the field, from top-tier source
- 8-9: Significant development with clear relevance to the discipline, reputable source
- 6-7: Relevant professional news, solid source, moderate significance
- 4-5: Tangential relevance or routine announcement
- 1-3: Weak relevance, questionable source, or outdated

GEOGRAPHIC PRIORITY (add +1 to score for stories from these regions):
- English-speaking countries: USA, UK, Canada, Australia, New Zealand, Ireland
- European countries: Germany, France, Netherlands, Switzerland, Sweden, Denmark, Norway, Finland, Belgium, Austria, Italy, Spain
- Apply the +1 bonus AFTER calculating the base score (cap at 10)

CRITICAL - INCLUDE NAMES IN SUMMARY:
- The summary MUST mention specific ORGANIZATION names (e.g., "researchers at MIT", "RSPCA veterinarians", "BASF announced")
- The summary MUST mention at least ONE key individual by full name with their role
- If the story is academic: name the institution AND lead researcher(s)
- If the story is about a company/clinic/agency: name the organization AND any executives or practitioners mentioned

DIRECT PEOPLE - MANDATORY EXTRACTION:
For EVERY story, identify and include in direct_people (people explicitly mentioned in the story):
1. The FIRST AUTHOR of any research paper (usually listed first in author order)
2. The CORRESPONDING/SENIOR AUTHOR who supervised the research
3. Other co-authors if prominently mentioned
4. Executives, practitioners, or spokespersons quoted in the article
5. Look for: "first author", "senior author", "lead author", "principal investigator", "corresponding author", "supervised by"

CRITICAL - AUTHOR AFFILIATIONS:
- Use the author's ACTUAL INSTITUTIONAL AFFILIATION (their university/company/clinic/agency), NOT the journal name
- Journal names (e.g., Nature, Science, JAMA, Chemical Engineering Journal) are PUBLISHERS, not affiliations

Note: direct_people captures people explicitly mentioned for LinkedIn @mentions. Indirect leadership comes from indirect_people enrichment.
The summary should still name at least 1-2 key individuals for readability.

CRITICAL - NO PLACEHOLDERS:
- Extract REAL names from the article - never use "TBA", "Unknown", "N/A", or placeholder text
- If you cannot find real names, leave direct_people as an empty array []

DIRECT_PEOPLE FIELD REQUIREMENTS (for accurate LinkedIn matching):
- name: FULL name (first AND last name required)
- company: Organization name (university, company, clinic, agency, institution)
- position: Job title/role exactly as stated
- department: Department/service/school if mentioned
- location: City/country if mentioned
- role_type: One of: "academic", "executive", "researcher", "practitioner", "engineer", "student", "spokesperson", "other"
- research_area: Research/subject area if mentioned (leave "" if not stated)
- linkedin_profile: Leave empty ""

WRITING STYLE FOR SUMMARIES:
- Write in first person ("I", "what stands out to me")
- Sound like an expert {discipline} sharing professional insight
- Be concise, analytical, and reflective rather than promotional
- Include at least one field-relevant perspective (e.g., clinical impact, patient/animal welfare, operational feasibility, safety, policy/regulatory nuance, cost, adoption/scale, evidence strength, limitations)
- When mentioning "applications" or "potential uses", ALWAYS provide at least one SPECIFIC example (e.g., "applications in energy storage, such as next-generation batteries" NOT just "applications in energy storage")
- Avoid sounding like a news aggregator or influencer

HOOK AND CTA STRUCTURE (CRITICAL FOR LINKEDIN ENGAGEMENT):
- START with an attention-grabbing HOOK in the first 1-2 sentences:
  * Ask a provocative question
  * Share a surprising statistic or finding
  * Make a bold or counterintuitive statement
  * Create curiosity that makes readers want to continue
- END with a call-to-action (CTA) question to invite engagement:
  * "What's your take on this approach?"
  * "Have you seen similar results in your practice?"
  * "Would this work in your organization?"
  * "What implications do you see for [field]?"

HOOK EXAMPLES:
- "What if the solution to antibiotic resistance has been hiding in plain sight?"
- "I never expected a 40% reduction in processing time from such a simple change."
- "The data on this one stopped me in my tracks."
- "This challenges everything I thought I knew about [topic]."

BAD SUMMARY EXAMPLE (too promotional, no hook, no CTA):
"This breakthrough will change everything overnight! Truly revolutionary!"

GOOD SUMMARY EXAMPLE (hook + insight + CTA):
"What if the solution to crop yield optimization was hiding in our gut microbiome? Researchers at MIT just demonstrated a 30% improvement using this unexpected approach. What stands out to me is how quickly this could translate into practice — the workflow and safety profile look feasible. Have you seen similar cross-disciplinary approaches work in your field?"

HASHTAG GUIDELINES:
- Use 1-3 relevant, professional hashtags per story
- CamelCase for multi-word hashtags (e.g., VeterinaryMedicine, AnimalHealth, ProcessSafety, ClimatePolicy)
- Focus on discipline- or topic-specific tags; also include the story category as a tag

CRITICAL - SOURCE URLs:
- Stories originated from a URL. Only include URLs you found in the search results and used to create the story.
- Do NOT invent or guess URLs. Every story must have at least 1 URL used to create the story associated with it.
- URLs must be SPECIFIC ARTICLE pages, NOT generic category/index pages like "/news/" or "/articles/"
- BAD URL examples: "https://example.com/news/", "https://example.com/veterinary/", "https://example.com/"
- GOOD URL examples: "https://example.com/news/2026/01/laying-hen-welfare-plans", "https://example.com/articles/12345-study-reveals"

RESPOND WITH ONLY THIS JSON FORMAT:
{{
    "stories": [
        {{
            "title": "Story Title",
            "sources": ["https://real-url-from-search.com/article"],
            "summary": "I found this work compelling because... [first-person summary]",
            "category": "Research",
            "quality_score": 8,
            "quality_justification": "Highly relevant topic, reputable source, timely",
            "hashtags": ["VeterinaryMedicine", "AnimalHealth"],
            "organizations": ["University of Glasgow", "RSPCA"],
            "direct_people": [
                {{"name": "Dr. Jane Smith", "company": "University of Glasgow", "position": "Lead Researcher", "department": "Veterinary Medicine", "location": "Glasgow, UK", "role_type": "academic", "research_area": "zoonotic disease", "linkedin_profile": ""}},
                {{"name": "John Doe", "company": "RSPCA", "position": "Clinical Director", "department": "", "location": "London, UK", "role_type": "practitioner", "research_area": "", "linkedin_profile": ""}}
            ]
        }}
    ]
}}

ORGANIZATIONS - ALWAYS EXTRACT:
- ALWAYS include an "organizations" array listing ALL companies, universities, clinics, agencies, and institutions mentioned in the story
- PRIORITIZE the institutions where the named people WORK (their affiliations)
- Journal names are PUBLISHERS, not organizations to include
- Organizations enable later lookup of leadership profiles on LinkedIn

IMPORTANT: Return complete, valid JSON. Keep summaries concise. Use ONLY real URLs. Write ALL summaries in first person. ALWAYS populate direct_people with people from the story AND key leaders from mentioned organizations. ALWAYS populate organizations with ALL institutions mentioned.""",
    )

    # Verification prompt - used to verify story suitability for publication
    # Placeholders: {search_prompt}, {story_title}, {story_summary}, {story_sources}, {people_count}, {linkedin_profiles_found}, {summary_word_limit}, {promotion_message}
    VERIFICATION_PROMPT: str = _get_str(
        "VERIFICATION_PROMPT",
        """You are a strict editorial review board for a professional, discipline-focused LinkedIn publication.

ORIGINAL SELECTION CRITERIA:
"{search_prompt}"

STORY TO EVALUATE:
Title: {story_title}
Summary: {story_summary}
Sources: {story_sources}

PROMOTION MESSAGE (appended to post):
{promotion_message}

LINKEDIN PROFILE STATUS:
People identified: {people_count}
LinkedIn profiles found: {linkedin_profiles_found}

EVALUATION CRITERIA:
1. RELEVANCE: Does this story clearly and genuinely match the original selection criteria?
2. PROFESSIONALISM: Is the tone suitable for a professional audience on LinkedIn?
3. DECENCY: Is the content appropriate for all professional audiences?
4. CREDIBILITY: Does the summary appear factual, plausible, and supported by reputable sources? (Major publications, academic institutions, established industry sources)
5. FIELD VALUE: Does the post demonstrate discipline-specific insight, judgement, or practical relevance (e.g., clinical or operational implications, safety, policy/regulatory nuance, feasibility, limitations)?
6. DISTINCTIVENESS: Would this post make the author appear thoughtful rather than automated or generic?
7. LENGTH: Is the summary appropriately concise (under {summary_word_limit} words)?
8. HASHTAGS: Are hashtags professional and relevant (no promotional or generic tags like #news)?
9. LINKEDIN MENTIONS: Have LinkedIn profiles been identified for key people? (Not required for approval, but good to have)

PROMOTION MESSAGE EVALUATION:
10. PROMOTION ALIGNMENT: Does the promotion message connect authentically to the story's topic/technology/industry?
11. PROMOTION TONE: Is the promotion professional, dignified, and confident WITHOUT being:
    - Sycophantic (excessive flattery or fawning)
    - Begging or desperate-sounding
    - Self-demeaning or apologetic
    - Overly humble or submissive
    - Pushy, aggressive, or demanding
12. PROMOTION EFFECTIVENESS: Is it an effective call to action that:
    - Clearly signals availability for opportunities
    - Invites connection or conversation
    - Demonstrates genuine interest in the field/technology
    - Would encourage recruiters or hiring managers to reach out
13. PROMOTION QUALITY: Does it maintain professional gravitas while being approachable and personable?

BAD PROMOTION EXAMPLES (should REJECT):
- "I would be so grateful if anyone could help me find a job, I really need this opportunity!" (desperate, begging)
- "Your company is absolutely amazing and I'd do anything to work there!" (sycophantic)
- "I know I'm just a graduate but maybe someone might consider giving me a chance?" (self-demeaning)
- "HIRE ME! DM for my CV!" (pushy, unprofessional)
- "Passionate about sustainability." (vague, no call to action)

GOOD PROMOTION EXAMPLES (should APPROVE):
- "DVM exploring opportunities in advanced animal health. I'd welcome a conversation with teams working in this space."
- "Clinical researcher actively seeking roles in translational medicine. Open to connecting with hiring managers in trial operations or data science."
- "Data professional looking to contribute to impactful analytics projects. If your team is building in this area, I'd love to hear from you."
- "Fascinated by this approach. Currently seeking roles in this field — feel free to reach out or connect."

IMAGE EVALUATION (if an image is provided):
13. IMAGE PROFESSIONALISM: Is the image appropriate for a professional context?
14. IMAGE RELEVANCE: Does the image relate to the subject of the story?
15. IMAGE CREDIBILITY: Does the image depict realistic settings, tools, or subjects appropriate to the story?

IMPORTANT NOTES ON IMAGES - BE LENIENT:
- Images are AI-generated and intentionally feature attractive professionals
- An attractive or beautiful person in the image is EXPECTED and ACCEPTABLE - do NOT reject for this reason
- Professional appearance (well-groomed, confident, attractive) is a positive quality in business imagery
- DO NOT reject for minor clothing details like necklines, fitted clothing, or fashionable professional attire
- Lab coats, safety gear, and professional attire in various styles are ALL acceptable
- Only reject images showing explicit content, nudity, or clearly unprofessional scenarios (e.g., swimwear, lingerie)
- A "low-cut" top or fitted clothing is NOT grounds for rejection - this is normal professional attire
- Focus ONLY on: Is there safety gear where needed? Is the setting credible? Does equipment match the story?
- AI watermarks or generation artifacts are acceptable
- When in doubt about clothing or appearance, APPROVE the image

IMPORTANT NOTES:
- LinkedIn profiles are helpful but not mandatory for approval
- The promotion message MUST maintain professional dignity - reject if it sounds desperate or self-demeaning

BAD CONTENT EXAMPLE (should REJECT):
Title: "AMAZING Breakthrough Will Change Everything!"
Summary: "This incredible new technology is absolutely revolutionary and will transform the entire industry overnight! Everyone needs to know about this game-changing innovation!"
Reason: Promotional tone, lacks substance, clickbait headline, no professional perspective

GOOD CONTENT EXAMPLE (should APPROVE):
Title: "Clinical team reports improved outcomes with new protocol"
Summary: "What interests me is how this protocol could improve routine practice — the early safety signals and workflow fit look promising, but I'd like to see longer-term follow-up before broad adoption."
Reason: Professional tone, first-person perspective, discipline-relevant analysis, specific details, critical thinking

DECISION RULES:
- APPROVE only if ALL primary criteria (1-9) AND promotion criteria (10-13) are satisfied
- REJECT if ANY primary criterion is weak or unmet
- REJECT if promotion message sounds sycophantic, begging, self-demeaning, or lacks a clear call to action
- REJECT if promotion is vague without inviting engagement or connection
- When uncertain, REJECT

Respond with ONLY one of these exact words:
APPROVED
or
REJECTED

Then on a new line, provide a brief (one sentence) reason.""",
    )

    # Search query distillation prompt - converts long search requests into keywords
    # Used by local LLM to optimize DuckDuckGo queries
    SEARCH_DISTILL_PROMPT: str = _get_str(
        "SEARCH_DISTILL_PROMPT",
        """You are a search query optimizer for technical news. Convert the user's request into 3-5 keyword search terms optimized for DuckDuckGo news search.

Focus on:
- Technical terms and industry jargon
- Company or institution names if mentioned
- Specific technologies or processes

Do NOT include generic terms like "news", "recent", "latest", or "update".

BAD EXAMPLE:
Input: "Find recent news about chemical engineering innovations in sustainable processes"
Output: "recent news chemical engineering" (too generic, includes "news")

GOOD EXAMPLE:
Input: "Find recent news about chemical engineering innovations in sustainable processes"
Output: "chemical engineering sustainable process innovation catalyst" (specific, technical terms)

Return ONLY the 3-5 keywords, space-separated, no explanation.""",
    )

    # Local LLM story processing prompt - processes DuckDuckGo results into stories
    # Placeholders: {author_name}, {search_prompt}, {search_results}, {max_stories}, {summary_words}
    LOCAL_LLM_SEARCH_PROMPT: str = _get_str(
        "LOCAL_LLM_SEARCH_PROMPT",
        """You are writing AS {author_name}, a {discipline} professional sharing industry insights on LinkedIn. I have found the following search results for the query: "{search_prompt}"

SEARCH RESULTS:
{search_results}

TASK:
1. Select EXACTLY {max_stories} of the most relevant and interesting stories - no more, no fewer.
2. For each story, provide:
   - title: An informative, technical headline (avoid clickbait or sensationalism). Use FULL institution names, NOT abbreviations:
     - "University of California, Riverside" NOT "UC Riverside" or "UCR"
     - "University of California, Los Angeles" NOT "UCLA"
     - "University of Southern California" NOT "USC"
     - "University of Chicago" NOT "UChicago"
     - "Massachusetts Institute of Technology" NOT "MIT"
     - "California Institute of Technology" NOT "Caltech"
     - Apply this rule to ALL universities and institutions
   - summary: A {summary_words}-word summary written in FIRST PERSON as {author_name}
   - sources: A list containing the original link
   - category: One of: Medicine, Hydrogen, Research, Technology, Business, Science, AI, Other
   - quality_score: 1-10 rating (see scoring rubric below)
   - quality_justification: Brief explanation of the score
    - hashtags: Array of 1-3 relevant hashtags (without # symbol)
    - direct_people: Array of MAX 3 people objects - only the most relevant people mentioned in the story

QUALITY SCORE RUBRIC:
- 10: Breakthrough with major industry implications, from top-tier source
- 8-9: Significant news with clear engineering relevance, reputable source
- 6-7: Relevant industry news, solid source, moderate significance
- 4-5: Tangential relevance or routine announcement
- 1-3: Weak relevance, questionable source, or outdated

CRITICAL - INCLUDE NAMES IN SUMMARY:
- ALWAYS mention specific COMPANY NAMES involved (e.g., "BASF", "MIT", "ExxonMobil")
- ALWAYS mention KEY INDIVIDUALS by full name (researchers, CEOs, lead engineers)
- Include their role/title (e.g., "Dr. Jane Smith, lead researcher at MIT")
- If academic research, name the university AND lead researcher(s) by name
- If company development, name the company AND any executives mentioned
- Read the source article carefully to extract actual names

DIRECT PEOPLE - MANDATORY EXTRACTION:
For EVERY story, identify and include in direct_people (people explicitly mentioned in the story):
1. The FIRST AUTHOR of any research paper (usually listed first in author order)
2. The CORRESPONDING/SENIOR AUTHOR who supervised the research
3. Other co-authors if prominently mentioned
4. Executives or spokespersons quoted in the article
5. Look for: "first author", "senior author", "lead author", "principal investigator", "corresponding author", "supervised by"

CRITICAL - AUTHOR AFFILIATIONS:
- Use the author's ACTUAL INSTITUTIONAL AFFILIATION (their university/company), NOT the journal name
- Example: If a paper in "Chemical Engineering Journal" is by researchers at "National Taiwan University",
  use company: "National Taiwan University", NOT "Chemical Engineering Journal"
- Journal names (e.g., Nature, Science, Chemical Engineering Journal) are PUBLISHERS, not affiliations
- Look for phrases like "researchers at", "from", "affiliated with", "Department of X at Y University"

CRITICAL - NO PLACEHOLDERS:
- Extract REAL names from the article - never use "TBA", "Unknown", "N/A", or placeholder text
- If a person's name is explicitly mentioned in the article, include them
- If you cannot find real names, leave direct_people as an empty array []
- Academic stories usually mention researchers by name - look carefully

WRITING STYLE FOR SUMMARIES:
- Write in first person (use "I", "what stands out to me", "from an engineering perspective", etc.)
- Sound like an expert {discipline} professional sharing insights
- Be concise, technical, and reflective rather than promotional
- Each summary MUST include at least one engineering or industrial perspective
  (e.g. scalability, process efficiency, integration, cost, energy use, sustainability, environment, or limitations)
- Avoid sounding like a news aggregator or influencer

BAD SUMMARY EXAMPLE (too promotional):
"Dow Chemical just launched an AMAZING new process that's going to REVOLUTIONIZE polymer production! This is HUGE!"

GOOD SUMMARY EXAMPLE (thoughtful, technical):
"What interests me about Dow's new polyethylene process is the claimed 30% energy reduction — if that holds at commercial scale, it could reshape economics for downstream converters. The question I'd want answered is catalyst longevity under continuous operation."

APPLICATIONS - BE SPECIFIC:
- When mentioning "applications" or "potential uses", ALWAYS provide at least one SPECIFIC example
- BAD: "applications in energy storage and more"
- GOOD: "applications in energy storage, such as next-generation lithium-ion batteries and supercapacitors"

HASHTAG GUIDELINES:
- Use 1-3 relevant, professional hashtags per story
- CamelCase for multi-word hashtags (e.g., ChemicalEngineering, ProcessOptimization)
- Focus on industry, technology, or topic-specific tags
- Common tags: ChemicalEngineering, ProcessSafety, Sustainability, Innovation, Engineering, ClimateChange, Hydrogen
- Also use the story category as a Tag

Return the results as a JSON object with a "stories" key containing an array of story objects.
Example:
{{
  "stories": [
    {{
      "title": "Example Story",
      "summary": "I found this work at Dow Chemical fascinating... [first-person summary]",
      "sources": ["https://example.com"],
      "category": "Technology",
      "quality_score": 8,
      "quality_justification": "Highly relevant, reputable source",
      "hashtags": ["ChemicalEngineering", "Innovation"],
      "direct_people": [
        {{"name": "Dr. John Doe", "company": "Dow Chemical", "position": "Lead Researcher", "linkedin_profile": ""}},
        {{"name": "Jane Smith", "company": "Dow Chemical", "position": "CEO", "linkedin_profile": ""}}
      ]
    }}
  ]
}}

Return ONLY the JSON object. Write ALL summaries in first person. ALWAYS populate direct_people.

CRITICAL - SOURCE URLs:
- Sources must be the ACTUAL article URLs from the search results that you used to create each story
- Do NOT invent or guess URLs
- URLs must be SPECIFIC ARTICLE pages, NOT generic category/index pages like "/news/" or "/articles/"
- BAD URL examples: "https://example.com/news/", "https://example.com/veterinary/", "https://example.com/"
- GOOD URL examples: "https://example.com/news/2026/01/laying-hen-welfare-plans", "https://example.com/articles/12345-study-reveals"
- Every story must have at least 1 specific article URL""",
    )

    # LinkedIn mention search prompt - finds LinkedIn profiles for story entities
    # Placeholders: {title}, {summary}, {sources_text}
    LINKEDIN_MENTION_PROMPT: str = _get_str(
        "LINKEDIN_MENTION_PROMPT",
        """Analyze this news story and find LinkedIn profiles for key entities.

STORY TITLE: {title}

STORY SUMMARY: {summary}

SOURCES:
{sources_text}

TASK: Search for LinkedIn profiles/company pages for:
1. The main company/companies mentioned in the story
2. Key executives (CEO, CTO, Chief Engineer, Founder, Owner, etc.) of those companies
3. Researchers or authors named in the story
4. Any other notable individuals mentioned

For each entity found, provide:
- name: Full name or company name
- linkedin_url: The LinkedIn profile/company page URL (must be real, verified URL)
- type: "person" or "organization"
- role: Their role in the story context (e.g., "company", "ceo", "researcher", "author")

IMPORTANT: Only include entities where you can find a REAL LinkedIn profile/page.
Do NOT guess or make up LinkedIn URLs.

Return JSON format:
{{
  "mentions": [
    {{
      "name": "Company or Person Name",
      "linkedin_url": "https://www.linkedin.com/company/xxx or /in/xxx",
      "type": "organization",
      "role": "company"
    }}
  ]
}}

If no LinkedIn profiles can be found, return: {{"mentions": []}}""",
    )

    # Story enrichment prompt - extracts organizations and people from stories
    # Placeholders: {story_title}, {story_summary}, {story_sources}
    STORY_ENRICHMENT_PROMPT: str = _get_str(
        "STORY_ENRICHMENT_PROMPT",
        """Analyze this news story and extract all organizations and people mentioned.

STORY TITLE: {story_title}

STORY SUMMARY: {story_summary}

SOURCES: {story_sources}

TASK: Extract all organizations and people explicitly mentioned in this story.

WHAT COUNTS AS AN ORGANIZATION:
- Companies (e.g., BASF, Shell, ExxonMobil, Ecovyst, SABIC)
- Universities (e.g., MIT, UCLA, Chalmers University, Imperial College)
- University research groups or initiatives (e.g., MIT Energy Initiative, Stanford AI Lab)
- Research institutions (e.g., Max Planck Institute, CSIRO, Fraunhofer Institute)
- Professional bodies (e.g., IChemE, AIChE, RSC)
- Government agencies (e.g., EPA, DOE, NASA)
- Industry associations and consortia

WHAT COUNTS AS A PERSON:
- Researchers or scientists named in the story
- Executives or company leaders mentioned
- Professors or academics
- Anyone receiving an award or honor
- Spokespersons quoted in the story

RULES:
1. Only include organizations and people EXPLICITLY NAMED in the story
2. Do NOT guess or infer names
3. Include affiliation/title where stated
4. Extract SPECIALTY/RESEARCH AREA when mentioned or clearly implied by the story context
5. List ALL organizations mentioned, not just the primary one

SPECIALTY EXTRACTION GUIDANCE:
- Look for research areas mentioned (e.g., "carbon capture", "catalysis", "polymer science")
- Infer from department names (e.g., "Department of Electrochemistry" → "electrochemistry")
- Infer from project focus (e.g., working on "hydrogen electrolyzers" → "hydrogen/electrolysis")
- Use the story topic if person's specific role relates to it
- Common specialties: catalysis, process engineering, sustainability, materials science,
  electrochemistry, polymer chemistry, thermodynamics, separation processes, etc.

BAD EXTRACTION EXAMPLE:
Story mentions "a major oil company" → Do NOT add "ExxonMobil" or guess which company

GOOD EXTRACTION EXAMPLE:
Story mentions "researchers at MIT's Department of Chemical Engineering led by Prof. Chen, working on carbon capture technologies" →
Add organization: "MIT", "MIT Department of Chemical Engineering"
Add person: {{"name": "Prof. Chen", "title": "Professor", "affiliation": "MIT", "specialty": "carbon capture"}}

Return a JSON object:
{{
  "organizations": ["Organization Name 1", "Organization Name 2"],
  "direct_people": [
    {{"name": "Dr. Jane Smith", "title": "Lead Researcher", "affiliation": "MIT", "specialty": "catalysis"}},
    {{"name": "John Doe", "title": "CEO", "affiliation": "BASF", "specialty": ""}}
  ]
}}

If nothing found, return: {{"organizations": [], "direct_people": []}}

Return ONLY valid JSON, no explanation.""",
    )

    # LinkedIn profile search prompt - finds actual LinkedIn profile URLs
    # Placeholders: {people_list}
    LINKEDIN_PROFILE_SEARCH_PROMPT: str = _get_str(
        "LINKEDIN_PROFILE_SEARCH_PROMPT",
        """Search for the PERSONAL LinkedIn profile URLs of the following people.

PEOPLE TO FIND:
{people_list}

STORY CONTEXT (use to help verify correct profiles):
{story_context}

**CRITICAL: ONLY RETURN URLs FROM ACTUAL SEARCH RESULTS**
- You MUST search Google for each person's LinkedIn profile
- ONLY return a URL if you found it in actual search results
- DO NOT generate, guess, or construct LinkedIn URLs
- If search returns no LinkedIn profile for a person, do NOT include them in the output

**SEARCH STRATEGY:**
- Search: "FirstName LastName" site:linkedin.com/in
- Look for the actual LinkedIn URL in search results
- The URL format is: linkedin.com/in/username (username varies per person)

**VERIFICATION:**
For each URL you find in search results:
1. Verify the name on the profile matches the person you're searching for
2. Verify the organization/title is plausible (current OR past employer)
3. If the search result shows a snippet with wrong name/org, skip it

**DO NOT:**
- Invent URLs like linkedin.com/in/firstname-lastname-12345 (these are often wrong)
- Guess what someone's LinkedIn username might be
- Return URLs you haven't actually found in search results

**HANDLING NO RESULTS:**
- If Google search finds no LinkedIn profile for a person, simply omit them
- It's OK to return fewer profiles than people searched, or even an empty array
- Many people (especially academics) don't have LinkedIn profiles

Return a JSON array with ONLY profiles you found in search results:
[
  {{"name": "Person Name", "linkedin_url": "https://www.linkedin.com/in/actualusername", "title": "Their Title", "affiliation": "Their Organization", "confidence": "high|medium"}}
]

If no profiles found in search results, return: []

Return ONLY the JSON array, no explanation.""",
    )

    # Indirect people prompt - finds key executives for organizations
    # Placeholders: {organization_name}
    INDIRECT_PEOPLE_PROMPT: str = _get_str(
        "INDIRECT_PEOPLE_PROMPT",
        """Search for CURRENT key contacts at "{organization_name}" relevant to this story context.

STORY CONTEXT (use to prioritize leader types):
Story Category: {story_category}
Story Title: {story_title}

IMPORTANT: Leadership positions change frequently. Use web search to find CURRENT information.
Do NOT rely on potentially outdated training data.

ORGANIZATION TYPE DETECTION:
First, determine if this is a COMPANY or a UNIVERSITY/RESEARCH INSTITUTION.

LEADER TYPE PRIORITIES BY STORY CATEGORY:

FOR RESEARCH/SCIENCE/TECHNOLOGY STORIES:
- Principal Investigator / Lead Researcher / Lab Director (HIGHEST PRIORITY)
- Department Head / Chair of relevant department
- CTO / Chief Science Officer / VP R&D
- Dean of Engineering/Science (for universities)

FOR BUSINESS/JOBS/HIRING STORIES:
- Head of HR / Talent Acquisition Director / Chief People Officer (HIGHEST PRIORITY)
- Recruitment Manager / University Careers Director
- CEO / Managing Director (if directly quoted in story)

FOR PR/ANNOUNCEMENTS/NEWS STORIES:
- Head of Communications / PR Director / Media Relations (HIGHEST PRIORITY)
- Corporate Communications Manager / Press Office
- CEO / Founder (if announcement is about company direction)

FOR GENERAL STORIES:
- CEO / Managing Director / Founder (for companies)
- Department Head / Dean (for universities)
- CTO / VP Engineering (for tech-related)

EXPANDED ROLE TYPES TO SEARCH:

EXECUTIVE ROLES (C-suite):
- CEO, CTO, CFO, COO, CSO (Chief Science Officer)
- Managing Director, President, Founder, Owner
- VP/SVP Engineering, Research, Technology, Operations

HR/RECRUITMENT ROLES:
- Chief People Officer, Head of HR, HR Director
- Talent Acquisition Director/Manager, Recruiting Lead
- University Careers/Placement Director

PR/COMMUNICATIONS ROLES:
- Chief Communications Officer, VP Communications
- Director of Media Relations, Press Office Lead
- Corporate Communications Manager

ACADEMIC ROLES:
- Department Chair/Head, Dean, Associate Dean
- Principal Investigator, Lab Director
- Director of Research, Research Group Leader

SEARCH AND VERIFICATION RULES:
1. Search for current information: "{organization_name} [role] 2025" or 2026
2. For universities, search department-specific: "[university] [department] head"
3. Only include leaders found in RECENT search results
4. Verify the person is CURRENTLY in the role
5. Include LOCATION if found (city, country) for LinkedIn matching
6. When in doubt, OMIT — a missing leader is better than a wrong one
7. Maximum 4 leaders per organization
8. Include role_type for each leader: "executive", "academic", "hr_recruiter", "pr_comms", "researcher"
9. Extract SPECIALTY if evident from their role or the story context

SPECIALTY EXTRACTION:
- Infer from their department (e.g., "Head of Catalysis" → specialty: "catalysis")
- Infer from story topic if their role is directly related
- Common specialties: process engineering, sustainability, R&D, catalysis, polymers,
  separations, energy, materials, biotechnology, digitalization, etc.
- For HR/PR roles, specialty is usually not applicable - leave blank

Return a JSON object with enhanced fields:
{{
  "leaders": [
    {{
      "name": "Full Name",
      "title": "Exact Title",
      "organization": "{organization_name}",
      "role_type": "executive|academic|hr_recruiter|pr_comms|researcher",
      "department": "Department name if applicable",
      "location": "City, Country if found",
      "specialty": "Area of expertise if known"
    }}
  ]
}}

If no leaders can be VERIFIED with high confidence, return: {{"leaders": []}}

Return ONLY valid JSON, no explanation.""",
    )

    # DEPRECATED - kept for backward compatibility
    # Organization mention enrichment prompt - identifies organizations for professional mentions
    # Placeholders: {story_title}, {story_summary}, {story_sources}
    COMPANY_MENTION_PROMPT: str = _get_str(
        "COMPANY_MENTION_PROMPT",
        """Analyze this news story and identify if a specific organization is EXPLICITLY NAMED as the primary subject.

STORY TITLE: {story_title}

STORY SUMMARY: {story_summary}

SOURCES: {story_sources}

TASK: Determine if a specific, real organization is explicitly named and central to this story.

WHAT COUNTS AS AN ORGANIZATION:
- Companies (e.g., BASF, Shell, ExxonMobil)
- Universities (e.g., MIT, UCLA, Stanford, Oklahoma State University)
- Research institutions (e.g., Max Planck Institute, CSIRO)
- Professional bodies (e.g., IChemE, AIChE, RSC)
- Government agencies (e.g., EPA, DOE, NASA)
- Industry associations

RULES:
1. Only identify organizations that are EXPLICITLY NAMED in the story (not inferred)
2. The organization must be the PRIMARY subject of the story, not just mentioned
3. Do NOT guess or infer organization names from context
4. If multiple organizations are mentioned, choose the most prominent one
5. If no specific organization is clearly the primary subject, respond with: NO_COMPANY_MENTION

If an organization qualifies, respond with a single professional sentence like:
"This development from [Organization Name] demonstrates their commitment to [area]."

If no organization qualifies, respond with exactly: NO_COMPANY_MENTION

Respond with ONLY the sentence or NO_COMPANY_MENTION, nothing else.""",
    )

    # Individual extraction prompt - identifies key people from stories
    # Placeholders: {story_title}, {story_summary}, {company_context}
    INDIVIDUAL_EXTRACTION_PROMPT: str = _get_str(
        "INDIVIDUAL_EXTRACTION_PROMPT",
        """Analyze this news story to identify key individuals.

STORY TITLE: {story_title}

STORY SUMMARY: {story_summary}

{company_context}

TASK: Identify key individuals associated with this story.

LOOK FOR:
1. Anyone EXPLICITLY NAMED in the story (researchers, executives, spokespersons, scientists, engineers)
2. Authors or lead researchers on the work described
3. Company executives if a company is involved (CEO, CTO, President, Founder, Owner)
4. University professors or lab directors if academic research
5. Government officials if regulatory/policy related

RULES:
1. Only include people who are REAL and can be verified
2. Prioritize people mentioned in the story itself
3. Include their job title/affiliation if known
4. Maximum 5 individuals

Return a JSON object with an array of individuals:
{{
  "individuals": [
    {{"name": "Full Name", "title": "Job Title or Affiliation", "source": "mentioned_in_story" or "known_expert"}},
    ...
  ]
}}

If no individuals can be identified, return: {{"individuals": []}}

Return ONLY valid JSON, no explanation.""",
    )

    # JSON repair prompt - attempts to fix malformed JSON from LLM responses
    # Placeholder: {malformed_json}
    JSON_REPAIR_PROMPT: str = _get_str(
        "JSON_REPAIR_PROMPT",
        """The following JSON is malformed. Please fix it and return ONLY the corrected JSON:

{malformed_json}

Return ONLY valid JSON, no explanation.""",
    )

    # --- Search Settings ---
    SEARCH_PROMPT: str = _get_str(
        "SEARCH_PROMPT",
        (
            f"I'm a professional {DISCIPLINE}. I'm looking for the latest professional {DISCIPLINE} "
            "stories I can summarise for publication on my LinkedIn profile"
        ),
    )
    # Optional template for the full search instruction prompt.
    # Supports placeholders: {criteria}, {since_date}, {summary_words}
    SEARCH_PROMPT_TEMPLATE: str = _get_str("SEARCH_PROMPT_TEMPLATE", "")
    SEARCH_LOOKBACK_DAYS: int = _get_int("SEARCH_LOOKBACK_DAYS", 7)
    USE_LAST_CHECKED_DATE: bool = _get_bool("USE_LAST_CHECKED_DATE", True)
    # Search engine for LinkedIn profile lookups (google, bing, duckduckgo)
    SEARCH_ENGINE: str = _get_str("SEARCH_ENGINE", "google").lower()
    # Maximum number of stories to find per search (default 5)
    MAX_STORIES_PER_SEARCH: int = _get_int("MAX_STORIES_PER_SEARCH", 5)
    # Maximum number of people to search for LinkedIn profiles per story (reduces API calls)
    MAX_PEOPLE_PER_STORY: int = _get_int("MAX_PEOPLE_PER_STORY", 3)
    # Similarity threshold for semantic deduplication (0.0-1.0, higher = stricter)
    DEDUP_SIMILARITY_THRESHOLD: float = float(
        _get_str("DEDUP_SIMILARITY_THRESHOLD", "0.7")
    )
    # Days to look back for all story deduplication (title matching)
    DEDUP_ALL_STORIES_WINDOW_DAYS: int = _get_int("DEDUP_ALL_STORIES_WINDOW_DAYS", 90)
    # Days to look back for published story deduplication (prevents similar content)
    DEDUP_PUBLISHED_WINDOW_DAYS: int = _get_int("DEDUP_PUBLISHED_WINDOW_DAYS", 30)
    # Number of retries for transient API failures
    API_RETRY_COUNT: int = _get_int("API_RETRY_COUNT", 3)
    # Base delay for exponential backoff (seconds)
    API_RETRY_DELAY: float = float(_get_str("API_RETRY_DELAY", "2.0"))
    # Default timeout for API calls (seconds)
    API_TIMEOUT_DEFAULT: int = _get_int("API_TIMEOUT_DEFAULT", 30)
    # Timeout for local LLM calls (seconds) - often need longer than cloud APIs
    LLM_LOCAL_TIMEOUT: int = _get_int("LLM_LOCAL_TIMEOUT", 120)
    # Enable URL validation before saving sources
    VALIDATE_SOURCE_URLS: bool = _get_bool("VALIDATE_SOURCE_URLS", True)
    # Enable story preview mode in test search
    SEARCH_PREVIEW_MODE: bool = _get_bool("SEARCH_PREVIEW_MODE", True)
    # Max results to fetch from DuckDuckGo search
    DUCKDUCKGO_MAX_RESULTS: int = _get_int("DUCKDUCKGO_MAX_RESULTS", 10)
    # Max output tokens for LLM responses
    LLM_MAX_OUTPUT_TOKENS: int = _get_int("LLM_MAX_OUTPUT_TOKENS", 8192)

    # --- Content Settings ---
    SUMMARY_WORD_COUNT: int = _get_int("SUMMARY_WORD_COUNT", 250)
    MIN_QUALITY_SCORE: int = _get_int("MIN_QUALITY_SCORE", 7)

    # --- Quality Score Calibration ---
    # These weights allow fine-tuning of the quality scoring system.
    # Higher weights = more influence on final score. Set to 0 to disable a factor.
    # The AI provides a base score (1-10), then these weights adjust it.
    QUALITY_WEIGHT_RECENCY: float = _get_float(
        "QUALITY_WEIGHT_RECENCY", 1.0
    )  # Newer is better
    QUALITY_WEIGHT_SOURCE: float = _get_float(
        "QUALITY_WEIGHT_SOURCE", 1.0
    )  # Reputable sources
    QUALITY_WEIGHT_RELEVANCE: float = _get_float(
        "QUALITY_WEIGHT_RELEVANCE", 1.0
    )  # Topic match
    QUALITY_WEIGHT_PEOPLE_MENTIONED: float = _get_float(
        "QUALITY_WEIGHT_PEOPLE_MENTIONED", 0.5
    )  # Named sources
    QUALITY_WEIGHT_GEOGRAPHIC: float = _get_float(
        "QUALITY_WEIGHT_GEOGRAPHIC", 0.5
    )  # Priority regions
    # Maximum bonus points from calibration (prevents inflation)
    QUALITY_MAX_CALIBRATION_BONUS: int = _get_int("QUALITY_MAX_CALIBRATION_BONUS", 2)

    # --- Originality Checking ---
    # Maximum word similarity (0-1) before flagging as too similar to source
    ORIGINALITY_MAX_SIMILARITY: float = _get_float("ORIGINALITY_MAX_SIMILARITY", 0.6)
    # Maximum n-gram overlap (0-1) before flagging copied phrases
    ORIGINALITY_MAX_NGRAM_OVERLAP: float = _get_float(
        "ORIGINALITY_MAX_NGRAM_OVERLAP", 0.4
    )
    # Enable/disable originality checking during verification
    ORIGINALITY_CHECK_ENABLED: bool = _get_bool("ORIGINALITY_CHECK_ENABLED", True)

    # --- Source Verification ---
    # Minimum number of sources required for a story
    MIN_SOURCES_REQUIRED: int = _get_int("MIN_SOURCES_REQUIRED", 1)
    # Minimum average credibility score (0-1) for sources
    MIN_SOURCE_CREDIBILITY: float = _get_float("MIN_SOURCE_CREDIBILITY", 0.3)
    # Require at least one tier 1 or tier 2 source
    REQUIRE_TIER1_OR_2_SOURCE: bool = _get_bool("REQUIRE_TIER1_OR_2_SOURCE", False)
    # Enable/disable source verification during content verification
    SOURCE_VERIFICATION_ENABLED: bool = _get_bool("SOURCE_VERIFICATION_ENABLED", True)

    # --- URL Archiving ---
    # Automatically archive source URLs to Wayback Machine to prevent link rot
    ARCHIVE_SOURCE_URLS: bool = _get_bool("ARCHIVE_SOURCE_URLS", False)

    # --- Publication Settings ---
    MAX_STORIES_PER_DAY: int = _get_int("MAX_STORIES_PER_DAY", 4)
    START_PUB_TIME: str = _get_str("START_PUB_TIME", "08:00")
    END_PUB_TIME: str = _get_str("END_PUB_TIME", "20:00")
    JITTER_MINUTES: int = _get_int("JITTER_MINUTES", 30)
    # Include "open to opportunities" postscript in LinkedIn posts
    INCLUDE_OPPORTUNITY_MESSAGE: bool = _get_bool("INCLUDE_OPPORTUNITY_MESSAGE", True)

    # --- LinkedIn Post Thresholds ---
    # Character count thresholds for post warnings
    LINKEDIN_POST_MIN_CHARS: int = _get_int("LINKEDIN_POST_MIN_CHARS", 100)
    LINKEDIN_POST_OPTIMAL_CHARS: int = _get_int("LINKEDIN_POST_OPTIMAL_CHARS", 1500)
    LINKEDIN_POST_MAX_CHARS: int = _get_int("LINKEDIN_POST_MAX_CHARS", 3000)
    # Hashtag recommendations
    LINKEDIN_MAX_HASHTAGS: int = _get_int("LINKEDIN_MAX_HASHTAGS", 5)

    @classmethod
    def get_pub_start_hour(cls) -> int:
        """Parse START_PUB_TIME and return hour."""
        try:
            return int(cls.START_PUB_TIME.split(":")[0])
        except (ValueError, IndexError):
            return 8

    @classmethod
    def get_pub_end_hour(cls) -> int:
        """Parse END_PUB_TIME and return hour."""
        try:
            return int(cls.END_PUB_TIME.split(":")[0])
        except (ValueError, IndexError):
            return 20

    # --- Cleanup Settings ---
    EXCLUSION_PERIOD_DAYS: int = _get_int("EXCLUSION_PERIOD_DAYS", 30)
    # Days after which stories are too old for image regeneration
    IMAGE_REGEN_CUTOFF_DAYS: int = _get_int("IMAGE_REGEN_CUTOFF_DAYS", 14)

    # --- Signature Details ---
    SIGNATURE_BLOCK_DETAIL: str = _get_str("SIGNATURE_BLOCK_DETAIL", "")

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

        NOTE: Pydantic validation already runs at import time for type/range checks.
        This method provides additional runtime validation for required fields
        that may be empty strings (valid for Pydantic but not for actual use).

        Returns a list of error messages (empty if valid).
        """
        errors: list[str] = []

        # Check required API keys (may be empty strings which pass Pydantic)
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is required")

        if not cls.LINKEDIN_ACCESS_TOKEN:
            errors.append("LINKEDIN_ACCESS_TOKEN is required for publishing")

        if not cls.LINKEDIN_AUTHOR_URN:
            errors.append("LINKEDIN_AUTHOR_URN is required for publishing")

        # Time range validation (already validated by Pydantic, but kept for compatibility)
        if cls.get_pub_start_hour() >= cls.get_pub_end_hour():
            errors.append("START_PUB_TIME must be before END_PUB_TIME")

        # Range validation (already validated by Pydantic, but kept for compatibility)
        if cls.MAX_STORIES_PER_DAY < 1:
            errors.append("MAX_STORIES_PER_DAY must be at least 1")

        if cls.SUMMARY_WORD_COUNT < 50:
            errors.append("SUMMARY_WORD_COUNT should be at least 50")

        return errors

    @classmethod
    def get_settings(cls) -> SettingsModel:
        """Get the underlying Pydantic settings model for advanced access."""
        return cls._pydantic_settings

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
        print(f"  MAX_STORIES_PER_DAY: {cls.MAX_STORIES_PER_DAY}")
        print(f"  START_PUB_TIME: {cls.START_PUB_TIME}")
        print(f"  END_PUB_TIME: {cls.END_PUB_TIME}")
        print(f"  JITTER_MINUTES: {cls.JITTER_MINUTES}")
        print(f"  EXCLUSION_PERIOD_DAYS: {cls.EXCLUSION_PERIOD_DAYS}")
        print(f"  DB_NAME: {cls.DB_NAME}")
        print(f"  IMAGE_DIR: {cls.IMAGE_DIR}")
        print("=============================================")


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for config module."""
    from test_framework import TestSuite

    suite = TestSuite("Config Tests")

    def test_get_int_valid():
        result = _get_int("NONEXISTENT_VAR_12345", 42)
        assert result == 42, f"Expected 42, got {result}"

    def test_get_bool_default():
        result = _get_bool("NONEXISTENT_VAR_12345", True)
        assert result is True, f"Expected True, got {result}"

    def test_get_str_default():
        result = _get_str("NONEXISTENT_VAR_12345", "default")
        assert result == "default", f"Expected 'default', got {result}"

    def test_get_float_default():
        result = _get_float("NONEXISTENT_VAR_12345", 3.14)
        assert result == 3.14, f"Expected 3.14, got {result}"

    def test_config_defaults():
        assert Config.SUMMARY_WORD_COUNT >= 50
        assert Config.MAX_STORIES_PER_DAY >= 1
        assert Config.MAX_STORIES_PER_SEARCH >= 1

    def test_config_validate():
        errors = Config.validate()
        assert isinstance(errors, list)

    def test_config_ensure_directories():
        Config.ensure_directories()
        assert os.path.isdir(Config.IMAGE_DIR)

    def test_config_publish_hours_valid():
        start_hour = Config.get_pub_start_hour()
        end_hour = Config.get_pub_end_hour()
        assert 0 <= start_hour <= 23
        assert 0 <= end_hour <= 23
        assert start_hour < end_hour

    suite.add_test("_get_int with default", test_get_int_valid)
    suite.add_test("_get_bool with default", test_get_bool_default)
    suite.add_test("_get_str with default", test_get_str_default)
    suite.add_test("_get_float with default", test_get_float_default)
    suite.add_test("Config defaults", test_config_defaults)
    suite.add_test("Config validate", test_config_validate)
    suite.add_test("Config ensure directories", test_config_ensure_directories)
    suite.add_test("Config publish hours valid", test_config_publish_hours_valid)

    # ========================================================================
    # Pydantic Settings Tests
    # ========================================================================

    def test_pydantic_settings_exists():
        """Test that Pydantic settings model is accessible."""
        settings = Config.get_settings()
        assert settings is not None
        assert isinstance(settings, SettingsModel)

    def test_pydantic_settings_type_coercion():
        """Test that Pydantic correctly coerces types from env vars."""
        settings = Config.get_settings()
        # These should be properly typed (not strings)
        assert isinstance(settings.search_lookback_days, int)
        assert isinstance(settings.dedup_similarity_threshold, float)
        assert isinstance(settings.prefer_local_llm, bool)
        assert isinstance(settings.search_prompt, str)

    def test_pydantic_settings_constraints():
        """Test that Pydantic constraint validators are properly defined."""
        # Test aspect ratio validator directly
        from pydantic import ValidationError

        valid = SettingsModel.validate_aspect_ratio("16:9")
        assert valid == "16:9", "Valid aspect ratio should pass"

        try:
            SettingsModel.validate_aspect_ratio("invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "IMAGE_ASPECT_RATIO" in str(e) or "aspect_ratio" in str(e).lower()

        # Test image size validator directly
        valid = SettingsModel.validate_image_size("2K")
        assert valid == "2K", "Valid image size should pass"

        try:
            SettingsModel.validate_image_size("3K")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "IMAGE_SIZE" in str(e) or "image_size" in str(e).lower()

    def test_pydantic_settings_time_validation():
        """Test that time format validation works."""
        # Test time format validator directly
        valid = SettingsModel.validate_time_format("08:00")
        assert valid == "08:00", "Valid time should pass"

        try:
            SettingsModel.validate_time_format("8am")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "HH:MM" in str(e)

        try:
            SettingsModel.validate_time_format("25:00")
            assert False, "Should have raised ValueError for invalid hour"
        except ValueError as e:
            assert "HH:MM" in str(e)

    def test_pydantic_settings_range_validation():
        """Test that range constraints are enforced via Field definitions."""
        from annotated_types import Ge, Le

        # Range constraints (ge, le) are defined in Field() and stored in metadata
        # We verify the field definitions have proper constraints
        field_info = SettingsModel.model_fields["search_lookback_days"]
        ge_constraints = [m for m in field_info.metadata if isinstance(m, Ge)]
        le_constraints = [m for m in field_info.metadata if isinstance(m, Le)]
        assert len(ge_constraints) == 1, (
            "search_lookback_days should have ge constraint"
        )
        assert ge_constraints[0].ge == 1, "search_lookback_days ge should be 1"
        assert len(le_constraints) == 1, (
            "search_lookback_days should have le constraint"
        )
        assert le_constraints[0].le == 365, "search_lookback_days le should be 365"

        field_info = SettingsModel.model_fields["dedup_similarity_threshold"]
        ge_constraints = [m for m in field_info.metadata if isinstance(m, Ge)]
        le_constraints = [m for m in field_info.metadata if isinstance(m, Le)]
        assert len(ge_constraints) == 1, "dedup_similarity_threshold should have ge"
        assert ge_constraints[0].ge == 0.0, (
            "dedup_similarity_threshold ge should be 0.0"
        )
        assert len(le_constraints) == 1, "dedup_similarity_threshold should have le"
        assert le_constraints[0].le == 1.0, (
            "dedup_similarity_threshold le should be 1.0"
        )

    def test_pydantic_time_range_validation():
        """Test that end time must be after start time (model validator)."""
        # This is a model validator - test that current config passes
        settings = Config.get_settings()
        start_hour = int(settings.start_pub_time.split(":")[0])
        end_hour = int(settings.end_pub_time.split(":")[0])
        assert start_hour < end_hour, "START_PUB_TIME should be before END_PUB_TIME"

    def test_pydantic_linkedin_thresholds_validation():
        """Test that LinkedIn thresholds are validated in order (model validator)."""
        # This is a model validator - test that current config passes
        settings = Config.get_settings()
        assert settings.linkedin_post_min_chars < settings.linkedin_post_optimal_chars
        assert settings.linkedin_post_optimal_chars < settings.linkedin_post_max_chars

    def test_pydantic_huggingface_token_fallback():
        """Test effective_huggingface_token property."""
        settings = Config.get_settings()
        # Just verify the property is accessible
        token = settings.effective_huggingface_token
        assert isinstance(token, str)

    suite.add_test("Pydantic settings exists", test_pydantic_settings_exists)
    suite.add_test("Pydantic type coercion", test_pydantic_settings_type_coercion)
    suite.add_test("Pydantic constraints", test_pydantic_settings_constraints)
    suite.add_test("Pydantic time validation", test_pydantic_settings_time_validation)
    suite.add_test("Pydantic range validation", test_pydantic_settings_range_validation)
    suite.add_test(
        "Pydantic time range validation", test_pydantic_time_range_validation
    )
    suite.add_test(
        "Pydantic LinkedIn thresholds", test_pydantic_linkedin_thresholds_validation
    )
    suite.add_test(
        "Pydantic HuggingFace token", test_pydantic_huggingface_token_fallback
    )

    return suite
