"""Configuration management for Social Media Publisher."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
import sys


def _check_venv():
    """Check if running in the correct virtual environment."""
    proj_root = Path(__file__).parent
    venv_dir = proj_root / ".venv"

    if not venv_dir.exists():
        # No .venv directory, skip check
        return

    # Check if current interpreter is in the .venv
    current_exe = Path(sys.executable).resolve()
    venv_scripts = venv_dir / ("Scripts" if os.name == "nt" else "bin")

    if not str(current_exe).startswith(str(venv_scripts.resolve())):
        print("=" * 60)
        print("ERROR: Not running in the .venv virtual environment!")
        print("=" * 60)
        print(f"\nCurrent Python: {current_exe}")
        print(f"Expected venv:  {venv_scripts}")
        print("\nTo fix, either:")
        print("  1. Activate the venv first:")
        if os.name == "nt":  # pragma: no cover
            print("     .venv\\Scripts\\activate")
        else:  # pragma: no cover
            print("     source .venv/bin/activate")
        print("  2. Or run directly with venv Python:")
        if os.name == "nt":  # pragma: no cover
            print("     .venv\\Scripts\\python main.py")
        else:  # pragma: no cover
            print("     .venv/bin/python main.py")
        print()
        sys.exit(1)


_check_venv()

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
    # Optional org URN to post as a company (e.g. urn:li:organization:123456)
    LINKEDIN_ORGANIZATION_URN: str = _get_str("LINKEDIN_ORGANIZATION_URN")
    # Author's display name for first-person story writing (e.g., "Wayne Gault")
    LINKEDIN_AUTHOR_NAME: str = _get_str("LINKEDIN_AUTHOR_NAME", "")

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
    # Style directive for image generation prompts - professional industrial photography
    IMAGE_STYLE: str = _get_str(
        "IMAGE_STYLE",
        "professional industrial photography, photorealistic, documentary style, "
        "clean corporate aesthetic, neutral color palette, realistic lighting, "
        "sharp focus, high resolution, suitable for engineering trade publication, all humans shown are highly attractive women",
    )
    # Aspect ratio for generated images (options: 1:1, 16:9, 9:16, 4:3, 3:4)
    IMAGE_ASPECT_RATIO: str = _get_str("IMAGE_ASPECT_RATIO", "16:9")

    # --- Prompt Templates ---
    # Image refinement prompt - sent to LLM to generate image generation prompts
    # Placeholders: {story_title}, {story_summary}, {image_style}
    IMAGE_REFINEMENT_PROMPT: str = _get_str(
        "IMAGE_REFINEMENT_PROMPT",
        """You are creating an image for a professional chemical engineering publication (like Chemical Engineering Magazine or AIChE publications).

STORY CONTEXT:
Story Title: {story_title}
Summary: {story_summary}

YOUR TASK: Create an image prompt for a REALISTIC, PROFESSIONAL photograph that would appear in an engineering trade journal.

CRITICAL REQUIREMENTS - THE IMAGE MUST BE:
1. PHOTOREALISTIC - like a real photograph, NOT artistic, NOT fantasy, NOT stylized
2. PROFESSIONAL - suitable for a serious engineering publication
3. TECHNICALLY ACCURATE - showing real equipment, processes, or concepts correctly
4. CREDIBLE - something a chemical engineer would recognize as realistic

SUBJECT SELECTION (choose the most appropriate):
- Industrial equipment: reactors, distillation columns, heat exchangers, piping systems, control rooms
- Laboratory settings: analytical instruments, lab glassware, researchers in lab coats
- Manufacturing facilities: chemical plants, refineries, pharmaceutical production
- Process technology: flow diagrams visualized as real equipment, process units
- Materials and products: chemicals, polymers, catalysts, finished products
- Data/monitoring: control panels, SCADA screens, process monitoring (if story is about digitalization)
- People: engineers, researchers, technicians are all highly attractive women

WHAT TO AVOID:
- Fantasy or sci-fi elements
- Artistic interpretations or abstract concepts
- Glowing/magical effects
- Futuristic imaginary technology
- Cartoonish or illustrated styles
- Anything that would look silly to a practicing chemical engineer

STYLE REQUIREMENTS:
{image_style}

PHOTOGRAPHY SPECS:
- Professional industrial photography style
- Clean, well-lit scenes (industrial facility lighting or natural daylight)
- Sharp focus, high resolution
- Neutral, realistic colors
- Documentary/journalistic aesthetic
- Human subjects are all highly attractive women

OUTPUT: Write ONLY the image prompt. No explanations. Maximum 100 words.
Format: "[Specific industrial subject], [realistic setting], professional industrial photograph, photorealistic, sharp focus, natural lighting\"""",
    )

    # Fallback image prompt template when LLM refinement fails
    # Placeholders: {story_title}
    IMAGE_FALLBACK_PROMPT: str = _get_str(
        "IMAGE_FALLBACK_PROMPT",
        "Professional industrial photograph for chemical engineering publication: "
        "{story_title}. Industrial facility or laboratory setting, "
        "photorealistic, documentary style, natural lighting, sharp focus, "
        "neutral colors, suitable for engineering trade journal",
    )

    # Search instruction prompt - the system prompt for story search
    # Placeholders: {max_stories}, {search_prompt}, {since_date}, {summary_words}, {author_name}
    SEARCH_INSTRUCTION_PROMPT: str = _get_str(
        "SEARCH_INSTRUCTION_PROMPT",
        """You are a news curator writing for {author_name}'s LinkedIn profile. Find {max_stories} recent news stories matching: "{search_prompt}"

REQUIREMENTS:
- Stories must be from after {since_date}
- Each story needs:
  * title: A clear, engaging headline
  * sources: Array of REAL source URLs from your search results
  * summary: {summary_words} words max, written in FIRST PERSON as if {author_name} is sharing their perspective
  * category: One of: Technology, Business, Science, AI, Other
  * quality_score: 1-10 rating
  * quality_justification: Brief explanation of the score
  * hashtags: Array of 1-3 relevant hashtags (without # symbol, e.g., ["ChemicalEngineering", "Sustainability"])

WRITING STYLE FOR SUMMARIES:
- Write in first person (use "I", "my", "I've found", "I'm excited about", etc.)
- Sound like a professional sharing industry insights with their network
- Be conversational but authoritative
- Example: "I've been following this development closely, and I think it represents a significant shift in how we approach..."

HASHTAG GUIDELINES:
- Use 1-3 relevant, professional hashtags per story
- CamelCase for multi-word hashtags (e.g., ChemicalEngineering, ProcessOptimization)
- Focus on industry, technology, or topic-specific tags
- Common tags: ChemicalEngineering, ProcessSafety, Sustainability, Innovation, Engineering

CRITICAL: Only include URLs you found in your search results. Do NOT invent or guess URLs.
If you cannot find a real URL for a story, omit that story entirely.

RESPOND WITH ONLY THIS JSON FORMAT:
{{
  "stories": [
    {{
      "title": "Story Title",
      "sources": ["https://real-url-from-search.com/article"],
      "summary": "I found this fascinating development in... [first-person summary]",
      "category": "Technology",
      "quality_score": 8,
      "quality_justification": "Highly relevant topic, reputable source, timely",
      "hashtags": ["ChemicalEngineering", "Innovation"]
    }}
  ]
}}

IMPORTANT: Return complete, valid JSON. Keep summaries concise. Use ONLY real URLs. Write ALL summaries in first person.""",
    )

    # Verification prompt - used to verify story suitability for publication
    # Placeholders: {search_prompt}, {story_title}, {story_summary}, {story_sources}
    VERIFICATION_PROMPT: str = _get_str(
        "VERIFICATION_PROMPT",
        """You are a strict editorial standards board for a professional social media publication.

ORIGINAL SELECTION CRITERIA:
"{search_prompt}"

STORY TO EVALUATE:
Title: {story_title}
Summary: {story_summary}
Sources: {story_sources}

EVALUATION CRITERIA:
1. RELEVANCE: Does this story genuinely match the original selection criteria?
2. PROFESSIONALISM: Is the content written in a professional, objective tone?
3. DECENCY: Is the content appropriate for all professional audiences?
4. ACCURACY: Does the summary appear factual and well-sourced?
5. QUALITY: Is this content worth sharing on a professional social media account?

If an image is provided, also evaluate:
6. IMAGE APPROPRIATENESS: Is the image professional and suitable for the story?
7. IMAGE RELEVANCE: Does the image relate to the story topic?

IMPORTANT IMAGE NOTES:
- All images are AI-generated, so "AI generated" watermarks/tags are ACCEPTABLE and should NOT cause rejection
- Focus on whether the image content is professional and relevant to the story

DECISION RULES:
- APPROVE only if ALL criteria are satisfactorily met
- REJECT if ANY criteria is not met
- When in doubt, REJECT

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
        "You are a search query optimizer. Convert long requests into 3-5 keyword search terms.",
    )

    # Local LLM story processing prompt - processes DuckDuckGo results into stories
    # Placeholders: {author_name}, {search_prompt}, {search_results}, {max_stories}, {summary_words}
    LOCAL_LLM_SEARCH_PROMPT: str = _get_str(
        "LOCAL_LLM_SEARCH_PROMPT",
        """You are a news curator writing for {author_name}'s LinkedIn profile. I have found the following search results for the query: "{search_prompt}"

SEARCH RESULTS:
{search_results}

TASK:
1. Select up to {max_stories} of the most relevant and interesting stories.
2. For each story, provide:
   - title: A catchy headline
   - summary: A {summary_words}-word summary written in FIRST PERSON as if {author_name} is sharing their perspective
   - sources: A list containing the original link
   - category: One of: Technology, Business, Science, AI, Other
   - quality_score: A score from 1-10 based on relevance and significance
   - quality_justification: Brief explanation of the score
   - hashtags: Array of 1-3 relevant hashtags (without # symbol)

WRITING STYLE FOR SUMMARIES:
- Write in first person (use "I", "my", "I've found", "I'm excited about", etc.)
- Sound like a professional sharing industry insights with their network
- Be conversational but authoritative
- Example: "I've been following this development closely, and I think it represents..."

HASHTAG GUIDELINES:
- Use 1-3 relevant, professional hashtags per story
- CamelCase for multi-word hashtags (e.g., ChemicalEngineering, ProcessOptimization)

Return the results as a JSON object with a "stories" key containing an array of story objects.
Example:
{{
  "stories": [
    {{
      "title": "Example Story",
      "summary": "I found this fascinating development... [first-person summary]",
      "sources": ["https://example.com"],
      "category": "Technology",
      "quality_score": 8,
      "quality_justification": "Highly relevant, reputable source",
      "hashtags": ["ChemicalEngineering", "Innovation"]
    }}
  ]
}}

Return ONLY the JSON object. Write ALL summaries in first person.""",
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
2. Key executives (CEO, CTO, Chief Engineer, etc.) of those companies
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

    # --- Publication Settings ---
    STORIES_PER_CYCLE: int = _get_int("STORIES_PER_CYCLE", 3)
    PUBLISH_WINDOW_HOURS: int = _get_int("PUBLISH_WINDOW_HOURS", 24)
    PUBLISH_START_HOUR: int = _get_int("PUBLISH_START_HOUR", 8)
    PUBLISH_END_HOUR: int = _get_int("PUBLISH_END_HOUR", 20)
    JITTER_MINUTES: int = _get_int("JITTER_MINUTES", 30)
    # Include "open to opportunities" postscript in LinkedIn posts
    INCLUDE_OPPORTUNITY_MESSAGE: bool = _get_bool("INCLUDE_OPPORTUNITY_MESSAGE", True)

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

    def test_config_defaults():
        assert Config.SUMMARY_WORD_COUNT >= 50
        assert Config.STORIES_PER_CYCLE >= 1
        assert Config.MAX_STORIES_PER_SEARCH >= 1

    def test_config_validate():
        errors = Config.validate()
        assert isinstance(errors, list)

    def test_config_ensure_directories():
        Config.ensure_directories()
        assert os.path.isdir(Config.IMAGE_DIR)

    def test_config_publish_hours_valid():
        assert 0 <= Config.PUBLISH_START_HOUR <= 23
        assert 0 <= Config.PUBLISH_END_HOUR <= 23
        assert Config.PUBLISH_START_HOUR < Config.PUBLISH_END_HOUR

    suite.add_test("_get_int with default", test_get_int_valid)
    suite.add_test("_get_bool with default", test_get_bool_default)
    suite.add_test("_get_str with default", test_get_str_default)
    suite.add_test("Config defaults", test_config_defaults)
    suite.add_test("Config validate", test_config_validate)
    suite.add_test("Config ensure directories", test_config_ensure_directories)
    suite.add_test("Config publish hours valid", test_config_publish_hours_valid)

    return suite
