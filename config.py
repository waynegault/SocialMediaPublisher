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
    openai_api_key: str = Field(
        default="", alias="OPENAI_API_KEY"
    )  # For RAG and embeddings
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

    # --- LinkedIn Search & Connection Master Switch ---
    # WARNING: LinkedIn has threatened account bans for automation detection.
    # Set to True only if you accept the risk of account restrictions.
    # When False, all LinkedIn profile searching and connection requests are disabled.
    linkedin_search_enabled: bool = Field(
        default=False, alias="LINKEDIN_SEARCH_ENABLED"
    )

    # --- LinkedIn Search Settings ---
    # Skip LinkedIn's direct search (subject to rate limits) and use Google/Bing instead
    skip_linkedin_direct_search: bool = Field(
        default=True, alias="SKIP_LINKEDIN_DIRECT_SEARCH"
    )

    # --- LinkedIn Voyager API (for reliable profile lookups) ---
    # These cookies can be extracted from browser dev tools after logging into LinkedIn
    linkedin_li_at: str = Field(default="", alias="LINKEDIN_LI_AT")
    linkedin_jsessionid: str = Field(default="", alias="LINKEDIN_JSESSIONID")

    # --- LinkedIn Engagement Limits ---
    # These limits help avoid triggering LinkedIn's anti-automation detection
    linkedin_daily_comment_limit: int = Field(
        default=25, ge=1, le=100, alias="LINKEDIN_DAILY_COMMENT_LIMIT"
    )
    linkedin_hourly_comment_limit: int = Field(
        default=5, ge=1, le=20, alias="LINKEDIN_HOURLY_COMMENT_LIMIT"
    )
    linkedin_min_comment_interval: int = Field(
        default=300,
        ge=60,
        alias="LINKEDIN_MIN_COMMENT_INTERVAL",  # seconds
    )
    linkedin_weekly_connection_limit: int = Field(
        default=100, ge=1, le=200, alias="LINKEDIN_WEEKLY_CONNECTION_LIMIT"
    )
    linkedin_daily_connection_limit: int = Field(
        default=20, ge=1, le=50, alias="LINKEDIN_DAILY_CONNECTION_LIMIT"
    )

    # --- RapidAPI Fresh LinkedIn Data API (primary LinkedIn lookup method) ---
    # Get API key from: https://rapidapi.com/freshdata-freshdata-default/api/fresh-linkedin-profile-data
    rapidapi_key: str = Field(default="", alias="RAPIDAPI_KEY")

    # --- Local LLM (LM Studio) ---
    lm_studio_base_url: str = Field(
        default="http://localhost:1234/v1", alias="LM_STUDIO_BASE_URL"
    )
    lm_studio_model: str = Field(default="local-model", alias="LM_STUDIO_MODEL")
    prefer_local_llm: bool = Field(default=True, alias="PREFER_LOCAL_LLM")

    # --- Groq API (free cloud LLM alternative) ---
    # Get free API key from: https://console.groq.com/keys
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    prefer_groq: bool = Field(default=False, alias="PREFER_GROQ")

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

    # --- Extensible Image Provider Settings ---
    # Switch provider via IMAGE_PROVIDER - no code changes required
    # Supported: cloudflare, ai_horde, pollinations, huggingface
    image_provider: str = Field(default="pollinations", alias="IMAGE_PROVIDER")
    image_model: str = Field(default="flux-realism", alias="IMAGE_MODEL")
    image_timeout_seconds: int = Field(
        default=120, ge=10, alias="IMAGE_TIMEOUT_SECONDS"
    )

    # Cloudflare Workers AI
    cloudflare_account_id: str = Field(default="", alias="CLOUDFLARE_ACCOUNT_ID")
    cloudflare_api_token: str = Field(default="", alias="CLOUDFLARE_API_TOKEN")

    # AI Horde (community-run, free with anonymous access)
    ai_horde_api_key: str = Field(default="anonymous", alias="AI_HORDE_API_KEY")

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

    # --- LinkedIn Profile Requirement ---
    # When True, stories require identified people with LinkedIn profiles to pass verification
    # When False, stories can pass verification without any identified people
    require_linkedin_profiles: bool = Field(
        default=False, alias="REQUIRE_LINKEDIN_PROFILES"
    )
    # Minimum number of LinkedIn profiles required (only applies when require_linkedin_profiles=True)
    min_linkedin_profiles: int = Field(
        default=1, ge=0, le=10, alias="MIN_LINKEDIN_PROFILES"
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
    image_refinement_prompt_no_human: str = Field(
        default="", alias="IMAGE_REFINEMENT_PROMPT_NO_HUMAN"
    )
    image_fallback_prompt: str = Field(
        default=(
            "A photo of {appearance} {discipline} professional, framed from waist up in center-right of image, "
            "actively working with the subject matter related to: {story_title}. "
            "She occupies 45% of the frame with face clearly visible, confident warm expression. "
            "Relevant tools, subjects, or setting visible behind her shoulder. "
            "Authentic workplace setting appropriate to {discipline} (clinic, lab, field, office, facility). "
            "Shot with 85mm lens, natural workplace lighting, editorial quality for a professional publication."
        ),
        alias="IMAGE_FALLBACK_PROMPT",
    )
    search_instruction_prompt: str = Field(
        default=(
            "You are an expert {discipline_title} news curator with HIGH editorial standards.\n\n"
            "TASK: Find up to {max_stories} groundbreaking stories matching: {search_prompt}\n\n"
            "STRICT REQUIREMENTS:\n"
            "1. Stories must be DIRECTLY relevant to {discipline} professional work\n"
            "2. Must be from reputable sources (major publications, research institutions, industry news)\n"
            "3. Must have verifiable facts and specific technical details\n"
            "4. No speculation, opinion pieces, or tangentially related content\n"
            "5. Must be published after {since_date}\n\n"
            "For EACH story, provide:\n"
            "- title: Clear, engaging headline\n"
            "- summary: EXACTLY {summary_words} words, first-person narrative starting with 'I', "
            "providing professional insight and analysis (NOT just restating the headline)\n"
            "- sources: Array of source article URLs\n"
            "- quality_score: 1-10 rating (be harsh: 7=decent, 8=good, 9=excellent, 10=exceptional)\n"
            "- quality_justification: 1-2 sentences explaining WHY this score and relevance (REQUIRED)\n"
            "- category: One of [Research, Industry, Career, Technology, Policy]\n"
            "- direct_people: Array of people mentioned with name, position, company\n\n"
            "QUALITY STANDARDS:\n"
            "- Only include stories you would score 7+\n"
            "- A 9/10 story must have exceptional relevance and depth\n"
            "- Reject tangentially related content (e.g., shipping for chemical engineering)\n\n"
            "Return ONLY valid JSON array, no markdown or explanation."
        ),
        alias="SEARCH_INSTRUCTION_PROMPT",
    )
    verification_prompt: str = Field(
        default=(
            "You are a strict quality gatekeeper verifying content for LinkedIn publication.\n\n"
            "STORY TO EVALUATE:\n"
            "Title: {story_title}\n"
            "Summary: {story_summary}\n"
            "Summary word count: {summary_word_count} (target: {summary_word_limit})\n"
            "Quality justification provided: {quality_justification}\n"
            "Sources: {story_sources}\n"
            "Discipline: {discipline}\n"
            "Promotion: {promotion_message}\n"
            "People identified: {people_count}\n"
            "LinkedIn profiles found: {linkedin_profiles_found}\n"
            "Search criteria: {search_prompt}\n\n"
            "STRICT REJECTION CRITERIA (reject if ANY apply):\n"
            "1. SUMMARY LENGTH: Summary must be 80%-130% of target word count ({min_summary_words}-{summary_word_limit}+ words). "
            "Too short (<80%) = lacks depth. Slightly over target (up to 130%) is ACCEPTABLE.\n"
            "2. RELEVANCE: Story must be DIRECTLY relevant to {discipline} work - not tangentially related. "
            "A story about shipbuilding is NOT relevant to chemical engineering even if ships carry chemicals.\n"
            "3. SUBSTANCE: Summary must provide insight, analysis, or professional value - not just restate the headline.\n"
            "4. QUALITY JUSTIFICATION: If no quality justification was provided by the search, this indicates low-quality curation.\n\n"
            "RESPONSE FORMAT:\n"
            "Line 1: APPROVED or REJECTED\n"
            "Line 2: Brief reason (MAX 20 words) citing which criteria failed or why approved.\n\n"
            "Example responses:\n"
            "APPROVED\n"
            "Strong technical content directly relevant to chemical engineering with good depth.\n\n"
            "REJECTED\n"
            "Summary only 30 words, needs 160+ words minimum.\n\n"
            "Be STRICT on relevance and substance - but accept summaries within 80-130% of target."
        ),
        alias="VERIFICATION_PROMPT",
    )
    search_distill_prompt: str = Field(
        default=(
            "You are a search query optimizer. Extract 3-6 concise search keywords from the user's conversational query. "
            "Return ONLY the keywords as a short phrase, no explanations or formatting. "
            "Example: 'chemical engineering breakthroughs 2024' NOT 'Here are the keywords:...'"
        ),
        alias="SEARCH_DISTILL_PROMPT",
    )
    local_llm_search_prompt: str = Field(
        default=(
            "You are a strict news curator for a {discipline} professional. Analyze these search results and extract ONLY highly relevant stories.\n\n"
            "SEARCH RESULTS:\n{search_results}\n\n"
            "TASK: Select up to {max_stories} stories that are DIRECTLY relevant to {discipline} professional work for LinkedIn publication.\n\n"
            "RELEVANCE CRITERIA (be strict):\n"
            "- Story must be about {discipline} processes, techniques, research, or industry developments\n"
            "- Tangentially related stories (e.g., logistics, shipping, business deals) should be scored LOW or excluded\n"
            "- Prefer stories with technical depth, research findings, or professional insights\n\n"
            "For each story, provide:\n"
            "- title: Clear, engaging headline\n"
            "- summary: EXACTLY {summary_words} words, first-person narrative starting with 'I' providing professional insight and analysis, NOT just restating the headline\n"
            "- source_links: Array of source URLs\n"
            "- quality_score: 1-10 rating (be harsh: 7-8 = good, 9-10 = exceptional only)\n"
            "- quality_justification: 1-2 sentences explaining WHY this score (REQUIRED)\n"
            "- category: One of [Research, Industry, Career, Technology, Policy]\n\n"
            "Return ONLY a valid JSON array, no markdown or explanation:\n"
            '[{{"title": "...", "summary": "...", "source_links": ["..."], "quality_score": 7, "quality_justification": "Directly relevant to {discipline} because...", "category": "Research"}}]'
        ),
        alias="LOCAL_LLM_SEARCH_PROMPT",
    )
    linkedin_mention_prompt: str = Field(default="", alias="LINKEDIN_MENTION_PROMPT")
    story_enrichment_prompt: str = Field(
        default=(
            "Extract organizations and people from this news story.\n\n"
            "Title: {story_title}\n"
            "Summary: {story_summary}\n"
            "Sources: {story_sources}\n\n"
            "Look for:\n"
            "- Company/organization names mentioned\n"
            "- People quoted or mentioned by name with their titles\n\n"
            "Return ONLY valid JSON (no markdown, no explanation):\n"
            '{{"organizations": ["Org Name"], "direct_people": [{{"name": "Full Name", "title": "Job Title", "affiliation": "Organization"}}]}}\n\n'
            'If nothing found, return: {"organizations": [], "direct_people": []}'
        ),
        alias="STORY_ENRICHMENT_PROMPT",
    )
    linkedin_profile_search_prompt: str = Field(
        default="", alias="LINKEDIN_PROFILE_SEARCH_PROMPT"
    )
    indirect_people_prompt: str = Field(
        default=(
            "Find 1-3 senior leaders at {organization_name} relevant to this story.\n"
            "Story category: {story_category}\n"
            "Story title: {story_title}\n\n"
            "Return ONLY valid JSON (no markdown, no explanation):\n"
            '{{"leaders": [{{"name": "Full Name", "title": "Job Title", "organization": "{organization_name}", "role_type": "executive|academic|pr_comms", "location": "City, Country"}}]}}\n\n'
            'If you cannot find specific leaders, return: {"leaders": []}'
        ),
        alias="INDIRECT_PEOPLE_PROMPT",
    )
    company_mention_prompt: str = Field(
        default=(
            "Given this story, generate a brief company mention sentence.\n\n"
            "Title: {story_title}\n"
            "Summary: {story_summary}\n"
            "Sources: {story_sources}\n\n"
            "If companies are mentioned, return a single sentence mentioning them.\n"
            "If no companies are mentioned, return: NO_COMPANY_MENTION"
        ),
        alias="COMPANY_MENTION_PROMPT",
    )
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
# Config Class - Provides uppercase access to Pydantic settings
# ============================================================================
class _ConfigMeta(type):
    """Metaclass that provides uppercase attribute access to _settings."""

    def __getattr__(cls, name: str):
        """Map UPPERCASE_NAME to _settings.lowercase_name."""
        # Convert UPPERCASE_NAME to lowercase_name
        lower_name = name.lower()
        if hasattr(_settings, lower_name):
            return getattr(_settings, lower_name)
        raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(cls, name: str, value):
        """Map UPPERCASE_NAME assignment to _settings.lowercase_name."""
        # Allow setting class-level attributes normally (like _settings)
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        # Convert UPPERCASE_NAME to lowercase_name and set on _settings
        lower_name = name.lower()
        if hasattr(_settings, lower_name):
            object.__setattr__(_settings, lower_name, value)
        else:
            raise AttributeError(f"Config has no attribute '{name}'")


class Config(metaclass=_ConfigMeta):
    """
    Application configuration loaded from environment variables.

    Configuration is validated at import time using Pydantic.
    Access settings using UPPERCASE names: Config.LINKEDIN_SEARCH_ENABLED

    The underlying validated settings are in _settings (SettingsModel instance).
    """

    # Direct access to the validated Pydantic settings model
    _settings: SettingsModel = _settings

    @classmethod
    def get_settings(cls) -> SettingsModel:
        """Get the underlying Pydantic settings model."""
        return _settings

    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration. Returns list of error messages (empty if valid)."""
        # Pydantic validates at load time, so if we got here, config is valid
        return []

    @classmethod
    def get_pub_start_hour(cls) -> int:
        """Get the publishing window start hour."""
        return int(_settings.start_pub_time.split(":")[0])

    @classmethod
    def get_pub_end_hour(cls) -> int:
        """Get the publishing window end hour."""
        return int(_settings.end_pub_time.split(":")[0])

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure required directories exist."""
        Path(_settings.image_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls) -> None:
        """Print current configuration (masking sensitive values)."""
        s = _settings
        print("=== Social Media Publisher Configuration ===")
        print(f"  GEMINI_API_KEY: {'*' * 8 if s.gemini_api_key else 'NOT SET'}")
        print(
            f"  HUGGINGFACE_API_TOKEN: {'*' * 8 if s.huggingface_api_token else 'NOT SET'}"
        )
        print(
            f"  LINKEDIN_ACCESS_TOKEN: {'*' * 8 if s.linkedin_access_token else 'NOT SET'}"
        )
        print(f"  LINKEDIN_AUTHOR_URN: {s.linkedin_author_urn or 'NOT SET'}")
        print(f"  LINKEDIN_SEARCH_ENABLED: {s.linkedin_search_enabled}")
        print(f"  MODEL_TEXT: {s.model_text}")
        print(f"  MODEL_IMAGE: {s.model_image}")
        if s.huggingface_api_token:
            print(
                f"  HF_TTI_MODEL: {s.hf_tti_model} (prefer={s.hf_prefer_if_configured})"
            )
        print(f"  SEARCH_PROMPT: {s.search_prompt[:50]}...")
        if s.search_prompt_template:
            print("  SEARCH_PROMPT_TEMPLATE: custom (from .env)")
        print(f"  SEARCH_LOOKBACK_DAYS: {s.search_lookback_days}")
        print(f"  USE_LAST_CHECKED_DATE: {s.use_last_checked_date}")
        print(f"  MAX_STORIES_PER_SEARCH: {s.max_stories_per_search}")
        print(f"  DEDUP_SIMILARITY_THRESHOLD: {s.dedup_similarity_threshold}")
        print(f"  API_RETRY_COUNT: {s.api_retry_count}")
        print(f"  VALIDATE_SOURCE_URLS: {s.validate_source_urls}")
        print(f"  SEARCH_PREVIEW_MODE: {s.search_preview_mode}")
        print(f"  SUMMARY_WORD_COUNT: {s.summary_word_count}")
        print(f"  MIN_QUALITY_SCORE: {s.min_quality_score}")
        print(f"  MAX_STORIES_PER_DAY: {s.max_stories_per_day}")
        print(f"  START_PUB_TIME: {s.start_pub_time}")
        print(f"  END_PUB_TIME: {s.end_pub_time}")
        print(f"  JITTER_MINUTES: {s.jitter_minutes}")
        print(f"  EXCLUSION_PERIOD_DAYS: {s.exclusion_period_days}")
        print(f"  DB_NAME: {s.db_name}")
        print(f"  IMAGE_DIR: {s.image_dir}")
        print("=============================================")


def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for config module."""
    from test_framework import TestSuite

    suite = TestSuite("Config Tests")

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
