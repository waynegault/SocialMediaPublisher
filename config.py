"""Configuration management for Social Media Publisher."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
import sys


def _check_venv():
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
    # Style directive for image generation prompts - technical industrial photography
    IMAGE_STYLE: str = _get_str(
        "IMAGE_STYLE",
        "industrial engineering photography, technical documentation style, "
        "female engineer or scientist performing hands-on technical work, "
        "sharp focus on equipment and processes with worker in context, "
        "authentic PPE and workwear - hard hats, safety glasses, lab coats, coveralls, "
        "real industrial or laboratory environment with visible technical detail, "
        "natural workplace lighting supplemented by equipment glow, "
        "photorealistic, editorial quality for engineering trade publication",
    )
    # Aspect ratio for generated images (options: 1:1, 16:9, 9:16, 4:3, 3:4)
    IMAGE_ASPECT_RATIO: str = _get_str("IMAGE_ASPECT_RATIO", "16:9")

    # --- Prompt Templates ---
    # Image refinement prompt - sent to LLM to generate image generation prompts
    # Placeholders: {story_title}, {story_summary}, {image_style}
    IMAGE_REFINEMENT_PROMPT: str = _get_str(
        "IMAGE_REFINEMENT_PROMPT",
        """You are creating an image for a professional chemical engineering publication.

STORY TO ILLUSTRATE:
- Title: {story_title}
- Summary: {story_summary}

STEP 1 - EXTRACT THE KEY TECHNICAL ELEMENTS:
Read the story carefully and identify:
- The specific technology, process, or innovation mentioned
- The type of equipment or facility involved (reactor, distillation column, lab, refinery, etc.)
- The industry sector (petrochemical, pharmaceutical, renewable energy, etc.)
- Any specific materials, chemicals, or products discussed

STEP 2 - CREATE A SCENE THAT DIRECTLY DEPICTS THE STORY:
Your image MUST show the actual subject matter from the headline. Examples:
- Story about "new catalyst for hydrogen production" → Show hydrogen production equipment with catalyst handling
- Story about "CO2 capture technology" → Show carbon capture systems, absorption columns, or flue gas treatment
- Story about "battery recycling process" → Show battery materials, hydrometallurgical equipment, or sorting facilities
- Story about "biofuel breakthrough" → Show fermentation vessels, biomass handling, or biorefinery equipment

STEP 3 - SUGGEST AUTHENTIC INDUSTRIAL SETTING:
Create a setting that reflects the sector mentioned in the story:
- For pharmaceutical stories: sterile lab environments with appropriate equipment
- For petrochemical stories: refinery or plant control room settings
- For academic research: university laboratory with research equipment
- For renewable energy: appropriate generation or storage facilities
- Avoid explicit company logos or branding to prevent trademark issues
- Use authentic industrial or laboratory aesthetics that match the story context

STEP 4 - ADD A HUMAN ELEMENT (STRICTLY 40% of image composition):
- Include a female engineer or technician occupying MAXIMUM 40% of the image
- Position her to the far left or far right edge of the frame, never center
- She should be at a smaller scale relative to the equipment
- Her activity relates to the technology but is clearly secondary
- Authentic PPE appropriate for the specific work environment
- The human element provides scale and context, NOT the main subject

TECHNICAL ACCURACY IS CRITICAL:
- Name the specific type of equipment in your prompt (not just "industrial equipment")
- Reference the actual process or technology from the story
- Match the setting to what the story describes
- Include relevant instrumentation, gauges, control systems

COMPOSITION RULE (STRICT - 60% TECHNOLOGY / 40% HUMAN):
- Technology/equipment from the story: Dominates 60% of the frame, centered
- Female engineer: Maximum 40%, positioned to far left or right edge
- The technology IS the story; the human provides scale and context only
- Equipment and process details are the hero of the image
- Industrial/laboratory environment matching the story context

AVOID:
- Generic industrial backgrounds that could apply to any story
- Vague descriptions like "technical equipment" or "machinery"
- The engineer as the main focus - she is supporting context only
- People occupying more than 40% of the visual space
- Centering the person - they must be to one side
- Any technology or setting not mentioned in the story
- Explicit company logos or trademarks

BAD IMAGE PROMPT EXAMPLE (too generic):
"Industrial worker in factory with machinery and equipment in background"

GOOD IMAGE PROMPT EXAMPLE (specific to story):
"Hydrogen electrolysis facility with PEM electrolyzer stacks dominating center frame, female process engineer in hard hat and safety glasses reviewing control panel at far right edge, industrial piping and pressure vessels visible, technical documentation quality"

STYLE: {image_style}

OUTPUT: Write ONLY the image prompt. Maximum 80 words.
The prompt MUST specifically describe the technology/process from the story headline - not generic industrial imagery.""",
    )

    # Fallback image prompt template when LLM refinement fails
    # Placeholders: {story_title}
    IMAGE_FALLBACK_PROMPT: str = _get_str(
        "IMAGE_FALLBACK_PROMPT",
        "Technical photograph illustrating: {story_title}. "
        "The specific technology or process from the headline dominates 60% of the frame, positioned center. "
        "Female engineer in appropriate PPE positioned to far left or right edge, occupying maximum 40% of the image. "
        "Technology is the story (60%), engineer provides human context and scale only (40%). "
        "Authentic industrial or laboratory setting matching the story subject. "
        "Natural workplace lighting, photorealistic, engineering publication quality. "
        "Avoid generic industrial backgrounds, explicit logos, or centering the person.",
    )

    # Search instruction prompt - the system prompt for story search
    # Placeholders: {max_stories}, {search_prompt}, {since_date}, {summary_words}, {author_name}
    SEARCH_INSTRUCTION_PROMPT: str = _get_str(
        "SEARCH_INSTRUCTION_PROMPT",
        """You are writing AS {author_name}, a chemical engineering professional sharing industry insights on LinkedIn. Find {max_stories} recent news stories matching: "{search_prompt}"

REQUIREMENTS:
- Stories must be from after {since_date}
- Each story needs:
  * title: An informative, technical headline (avoid clickbait or sensationalism)
  * sources: Array of REAL source URLs from your search results
  * summary: {summary_words} words max, written in FIRST PERSON as {author_name}
  * category: One of: Medicine, Hydrogen, Research, Technology, Business, Science, AI, Other
  * quality_score: 1-10 rating (see scoring rubric below)
  * quality_justification: Brief explanation of the score
  * hashtags: Array of 1-3 relevant hashtags (without # symbol, e.g., ["ChemicalEngineering", "Sustainability"])
  * relevant_people: Array of people objects (see format below) - people mentioned in the story AND key leaders from organizations

QUALITY SCORE RUBRIC:
- 10: Breakthrough with major industry implications, from top-tier source
- 8-9: Significant news with clear engineering relevance, reputable source
- 6-7: Relevant industry news, solid source, moderate significance
- 4-5: Tangential relevance or routine announcement
- 1-3: Weak relevance, questionable source, or outdated

GEOGRAPHIC PRIORITY (add +1 to score for stories from these regions):
- English-speaking countries: USA, UK, Canada, Australia, New Zealand, Ireland
- European countries: Germany, France, Netherlands, Switzerland, Sweden, Denmark, Norway, Finland, Belgium, Austria, Italy, Spain
- Stories from these regions typically have better LinkedIn engagement for English-speaking professional audiences
- Apply the +1 bonus AFTER calculating the base score (cap at 10)

CRITICAL - INCLUDE NAMES IN SUMMARY:
- ALWAYS mention specific COMPANY NAMES involved in the story (e.g., "BASF", "MIT", "ExxonMobil")
- ALWAYS mention KEY INDIVIDUALS by full name when available (researchers, CEOs, lead engineers)
- Include their role/title (e.g., "Dr. Jane Smith, lead researcher at MIT")
- If the story is about academic research, name the university AND the lead researcher(s) by name
- If the story is about a company development, name the company AND any executives mentioned
- Read the source article carefully to extract actual names - they are usually mentioned

RELEVANT PEOPLE - MANDATORY EXTRACTION:
For EVERY story, identify and include in relevant_people:
1. People MENTIONED in the story (researchers, engineers, scientists, executives quoted)
2. Key leaders from ANY organizations/universities mentioned in the story:
   - CEO, President, Managing Director, Founder, Owner
   - Head/Director of Engineering, Head/Director of Research
   - Head/Director of Operations, Head/Director of HR
   - Principal Investigator, Head of Lab/School
   - University Principal, Chancellor, Dean
This reduces the need to mention all people in the summary itself.

CRITICAL - NO PLACEHOLDERS:
- Extract REAL names from the article - never use "TBA", "Unknown", "N/A", or placeholder text
- If a person's name is explicitly mentioned in the article, include them
- If you cannot find real names, leave relevant_people as an empty array []
- Academic stories usually mention researchers by name - look carefully
- Include the researcher's full name, their institution, and their role
- Example: If article mentions "Siddharth Deshpande, assistant professor" at "University of Rochester"
  → Include: {{"name": "Siddharth Deshpande", "company": "University of Rochester", "position": "Assistant Professor"}}

WRITING STYLE FOR SUMMARIES:
- Write in first person (use "I", "what stands out to me", "from an engineering perspective", etc.)
- Sound like an expert chemical/process engineer sharing professional insights
- Be concise, technical, and reflective rather than promotional
- Each summary MUST include at least one engineering or industrial perspective
  (e.g. scalability, process efficiency, integration, cost, energy use, sustainability, environment, or limitations)
- Avoid sounding like a news aggregator or influencer

BAD SUMMARY EXAMPLE (too promotional):
"BASF just launched an AMAZING new catalyst that's going to REVOLUTIONIZE the industry! This is HUGE news!"

GOOD SUMMARY EXAMPLE (thoughtful, technical):
"What stands out to me about BASF's new zeolite catalyst is the engineering challenge behind this — particularly how it could scale beyond lab conditions and integrate with existing process infrastructure. The selectivity improvements they're reporting are impressive, but I'll be watching to see how this performs at industrial scale."

HASHTAG GUIDELINES:
- Use 1-3 relevant, professional hashtags per story
- CamelCase for multi-word hashtags (e.g., ChemicalEngineering, ProcessOptimization)
- Focus on industry, technology, or topic-specific tags
- Common tags: ChemicalEngineering, ProcessSafety, Sustainability, Innovation, Engineering, ClimateChange, Hydrogen
- Also use the story category as a Tag

CRITICAL: Stories originated from a URL. Only include URLs you found in the search results and used to create the story. Do NOT invent or guess URLs. Every story must have at least 1 URL used to create the story associated with it.

RESPOND WITH ONLY THIS JSON FORMAT:
{{
  "stories": [
    {{
      "title": "Story Title",
      "sources": ["https://real-url-from-search.com/article"],
      "summary": "I found this work at MIT fascinating... [first-person summary]",
      "category": "Technology",
      "quality_score": 8,
      "quality_justification": "Highly relevant topic, reputable source, timely",
      "hashtags": ["ChemicalEngineering", "Innovation"],
      "relevant_people": [
        {{"name": "Dr. Jane Smith", "company": "MIT", "position": "Lead Researcher", "linkedin_profile": ""}},
        {{"name": "John Doe", "company": "BASF", "position": "CEO", "linkedin_profile": ""}}
      ]
    }}
  ]
}}

IMPORTANT: Return complete, valid JSON. Keep summaries concise. Use ONLY real URLs. Write ALL summaries in first person. ALWAYS populate relevant_people with people from the story AND key leaders from mentioned organizations.""",
    )

    # Verification prompt - used to verify story suitability for publication
    # Placeholders: {search_prompt}, {story_title}, {story_summary}, {story_sources}, {relevant_people_count}, {linkedin_profiles_found}, {summary_word_limit}
    VERIFICATION_PROMPT: str = _get_str(
        "VERIFICATION_PROMPT",
        """You are a strict editorial review board for a professional engineering-focused LinkedIn publication.

ORIGINAL SELECTION CRITERIA:
"{search_prompt}"

STORY TO EVALUATE:
Title: {story_title}
Summary: {story_summary}
Sources: {story_sources}

LINKEDIN PROFILE STATUS:
Relevant people identified: {relevant_people_count}
LinkedIn profiles found: {linkedin_profiles_found}

EVALUATION CRITERIA:
1. RELEVANCE: Does this story clearly and genuinely match the original selection criteria?
2. PROFESSIONALISM: Is the tone suitable for a professional engineering audience on LinkedIn?
3. DECENCY: Is the content appropriate for all professional audiences?
4. CREDIBILITY: Does the summary appear factual, technically plausible, and supported by reputable sources? (Major publications, academic institutions, established industry sources)
5. ENGINEERING VALUE: Does the post demonstrate technical insight, judgement, critical thinking or industrial relevance (e.g. scalability, process implications, limitations)?
6. DISTINCTIVENESS: Would this post make the author appear thoughtful rather than automated or generic?
7. LENGTH: Is the summary appropriately concise (under {summary_word_limit} words)?
8. HASHTAGS: Are hashtags professional and relevant (no promotional or generic tags like #news)?
9. LINKEDIN MENTIONS: Have LinkedIn profiles been identified for key people? (Not required for approval, but good to have)

IMAGE EVALUATION (if an image is provided):
10. IMAGE PROFESSIONALISM: Is the image credible and appropriate for a serious engineering context?
11. IMAGE RELEVANCE: Does the image clearly relate to the technical subject of the story?
12. IMAGE CREDIBILITY: Does the image avoid looking staged, promotional, or like generic AI stock imagery?

IMPORTANT NOTES:
- Images are AI-generated; AI watermarks or tags are acceptable
- Evaluate image credibility and relevance, not origin
- LinkedIn profiles are helpful but not mandatory for approval

BAD CONTENT EXAMPLE (should REJECT):
Title: "AMAZING Breakthrough Will Change Everything!"
Summary: "This incredible new technology is absolutely revolutionary and will transform the entire industry overnight! Everyone needs to know about this game-changing innovation!"
Reason: Promotional tone, lacks technical substance, clickbait headline, no engineering perspective

GOOD CONTENT EXAMPLE (should APPROVE):
Title: "MIT Researchers Demonstrate Improved Selectivity in CO2 Electroreduction"
Summary: "What interests me about Dr. Chen's work at MIT is the practical engineering angle — achieving 85% Faradaic efficiency at industrially relevant current densities addresses a key scale-up barrier. The question is whether this catalyst stability holds over extended operation."
Reason: Technical headline, first-person perspective, engineering analysis, specific details, critical thinking

DECISION RULES:
- APPROVE only if ALL primary criteria (1-8) are clearly satisfied
- REJECT if ANY primary criterion is weak or unmet
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
        """You are writing AS {author_name}, a chemical engineering professional sharing industry insights on LinkedIn. I have found the following search results for the query: "{search_prompt}"

SEARCH RESULTS:
{search_results}

TASK:
1. Select up to {max_stories} of the most relevant and interesting stories.
2. For each story, provide:
   - title: An informative, technical headline (avoid clickbait or sensationalism)
   - summary: A {summary_words}-word summary written in FIRST PERSON as {author_name}
   - sources: A list containing the original link
   - category: One of: Medicine, Hydrogen, Research, Technology, Business, Science, AI, Other
   - quality_score: 1-10 rating (see scoring rubric below)
   - quality_justification: Brief explanation of the score
   - hashtags: Array of 1-3 relevant hashtags (without # symbol)
   - relevant_people: Array of people objects (see format below) - people mentioned AND key org leaders

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

RELEVANT PEOPLE - MANDATORY EXTRACTION:
For EVERY story, identify and include in relevant_people:
1. People MENTIONED in the story (researchers, engineers, scientists, executives quoted)
2. Key leaders from ALL organizations/universities mentioned in the story:
   - CEO, President, Managing Director, Founder, Owner
   - Head/Director of Engineering, Head/Director of Research
   - Head/Director of Operations, Head/Director of HR
   - Principal Investigator, Head of Lab/School
   - University Principal, Chancellor, Dean

CRITICAL - NO PLACEHOLDERS:
- Extract REAL names from the article - never use "TBA", "Unknown", "N/A", or placeholder text
- If a person's name is explicitly mentioned in the article, include them
- If you cannot find real names, leave relevant_people as an empty array []
- Academic stories usually mention researchers by name - look carefully

WRITING STYLE FOR SUMMARIES:
- Write in first person (use "I", "what stands out to me", "from an engineering perspective", etc.)
- Sound like an expert chemical/process engineer sharing professional insights
- Be concise, technical, and reflective rather than promotional
- Each summary MUST include at least one engineering or industrial perspective
  (e.g. scalability, process efficiency, integration, cost, energy use, sustainability, environment, or limitations)
- Avoid sounding like a news aggregator or influencer

BAD SUMMARY EXAMPLE (too promotional):
"Dow Chemical just launched an AMAZING new process that's going to REVOLUTIONIZE polymer production! This is HUGE!"

GOOD SUMMARY EXAMPLE (thoughtful, technical):
"What interests me about Dow's new polyethylene process is the claimed 30% energy reduction — if that holds at commercial scale, it could reshape economics for downstream converters. The question I'd want answered is catalyst longevity under continuous operation."

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
      "relevant_people": [
        {{"name": "Dr. John Doe", "company": "Dow Chemical", "position": "Lead Researcher", "linkedin_profile": ""}},
        {{"name": "Jane Smith", "company": "Dow Chemical", "position": "CEO", "linkedin_profile": ""}}
      ]
    }}
  ]
}}

Return ONLY the JSON object. Write ALL summaries in first person. ALWAYS populate relevant_people.""",
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
4. List ALL organizations mentioned, not just the primary one

BAD EXTRACTION EXAMPLE:
Story mentions "a major oil company" → Do NOT add "ExxonMobil" or guess which company

GOOD EXTRACTION EXAMPLE:
Story mentions "researchers at MIT's Department of Chemical Engineering led by Prof. Chen" →
Add organization: "MIT", "MIT Department of Chemical Engineering"
Add person: {{"name": "Prof. Chen", "title": "Professor", "affiliation": "MIT"}}

Return a JSON object:
{{
  "organizations": ["Organization Name 1", "Organization Name 2"],
  "story_people": [
    {{"name": "Dr. Jane Smith", "title": "Lead Researcher", "affiliation": "MIT"}},
    {{"name": "John Doe", "title": "CEO", "affiliation": "BASF"}}
  ]
}}

If nothing found, return: {{"organizations": [], "story_people": []}}

Return ONLY valid JSON, no explanation.""",
    )

    # LinkedIn profile search prompt - finds actual LinkedIn profile URLs
    # Placeholders: {people_list}
    LINKEDIN_PROFILE_SEARCH_PROMPT: str = _get_str(
        "LINKEDIN_PROFILE_SEARCH_PROMPT",
        """Search for the LinkedIn profile URLs of the following people.

PEOPLE TO FIND:
{people_list}

For each person, search for their actual LinkedIn profile URL (format: linkedin.com/in/username).

VALIDATION RULES (CRITICAL):
1. Only include profiles you found through LIVE SEARCH RESULTS
2. The LinkedIn URL must appear in your search results, not be constructed or guessed
3. Verify the profile shows the SAME NAME AND AFFILIATION before including
4. When in doubt, EXCLUDE — a missing profile is better than a wrong one
5. Do NOT construct LinkedIn URLs from names (e.g., /in/john-smith)
6. Do NOT guess usernames based on common patterns

BAD EXAMPLE (should NOT do):
Person: "Dr. Jane Smith, MIT Professor"
Output: {{"name": "Dr. Jane Smith", "linkedin_url": "https://www.linkedin.com/in/jane-smith-mit"}}
Reason: URL was guessed/constructed, not found in search results

GOOD EXAMPLE (correct approach):
Person: "Dr. Jane Smith, MIT Professor"
Search finds: LinkedIn profile at linkedin.com/in/janesmith-chem showing "Jane Smith, Professor at MIT"
Output: {{"name": "Dr. Jane Smith", "linkedin_url": "https://www.linkedin.com/in/janesmith-chem", "title": "Professor", "affiliation": "MIT"}}
Reason: URL was found in search, name and affiliation match

Return a JSON array:
[
  {{"name": "Person Name", "linkedin_url": "https://www.linkedin.com/in/actualusername", "title": "Their Title", "affiliation": "Their Organization"}}
]

If no profiles can be VERIFIED through search, return: []

Return ONLY the JSON array, no explanation.""",
    )

    # Organization leaders prompt - finds key executives for organizations
    # Placeholders: {organization_name}
    ORG_LEADERS_PROMPT: str = _get_str(
        "ORG_LEADERS_PROMPT",
        """Search for the CURRENT key leadership of "{organization_name}".

IMPORTANT: Leadership positions change frequently. Use web search to find CURRENT information.
Do NOT rely on potentially outdated training data.

Find the following roles if they exist:
1. CEO / Chief Executive Officer / Managing Director / Founder / Owner
2. President
3. CTO / Chief Technology Officer / Chief Engineer
4. CHRO / Chief Human Resources Officer / Head of HR

SEARCH AND VERIFICATION RULES:
1. Search for current leadership information (e.g., "{organization_name} CEO 2025")
2. Only include leaders you find in RECENT search results
3. Verify the person is CURRENTLY in the role, not a predecessor
4. If search results are ambiguous or outdated, OMIT that role
5. When in doubt, leave the role empty — a missing leader is better than a wrong one
6. Include their exact current title as shown in search results

BAD EXAMPLE (should NOT do):
Organization: "BASF"
Output: {{"name": "Martin Brudermüller", "title": "CEO"}} from memory without verification
Reason: Leadership may have changed; must verify through current search

GOOD EXAMPLE (correct approach):
Organization: "BASF"
Search finds recent news: "BASF CEO Dr. Markus Kamieth announced..."
Output: {{"name": "Dr. Markus Kamieth", "title": "CEO", "organization": "BASF"}}
Reason: Verified through recent search results

Return a JSON object:
{{
  "leaders": [
    {{"name": "Full Name", "title": "Exact Title", "organization": "{organization_name}"}}
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
