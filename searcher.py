"""Story search and discovery using Gemini AI with Google Search grounding."""

import json
import logging
import re
from datetime import datetime, timedelta
from ddgs import DDGS

from google import genai  # type: ignore
from openai import OpenAI

from config import Config
from database import Database, Story

logger = logging.getLogger(__name__)


class StorySearcher:
    """Search for and process news stories using Gemini AI or Local LLM."""

    def __init__(
        self,
        database: Database,
        client: genai.Client,
        local_client: OpenAI | None = None,
    ):
        """Initialize the story searcher."""
        self.db = database
        self.client = client
        self.local_client = local_client

    def get_search_start_date(self) -> datetime:
        """
        Determine the start date for searching stories.
        Uses last check date if enabled, otherwise uses lookback days.
        """
        if Config.USE_LAST_CHECKED_DATE:
            last_check = self.db.get_last_check_date()
            if last_check:
                logger.info(f"Using last check date: {last_check}")
                return last_check

        # Fallback to lookback days
        lookback = datetime.now() - timedelta(days=Config.SEARCH_LOOKBACK_DAYS)
        logger.info(f"Using lookback of {Config.SEARCH_LOOKBACK_DAYS} days: {lookback}")
        return lookback

    def search_and_process(self) -> int:
        """
        Search for stories and save new ones to the database.
        Returns the number of new stories saved.
        """
        since_date = self.get_search_start_date()
        search_prompt = Config.SEARCH_PROMPT
        summary_words = Config.SUMMARY_WORD_COUNT

        logger.info(
            f"Searching for stories matching: '{search_prompt}' since {since_date.date()}"
        )

        try:
            if self.local_client:
                return self._search_local(search_prompt, since_date, summary_words)
            else:
                return self._search_gemini(search_prompt, since_date, summary_words)

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.error(
                    "Search failed: API quota exceeded (429 RESOURCE_EXHAUSTED)"
                )
                raise RuntimeError(
                    "API quota exceeded. Please try again later or check your Google Cloud billing/quota settings."
                ) from e
            logger.error(f"Search failed: {e}")
            return 0

    def _search_gemini(
        self, search_prompt: str, since_date: datetime, summary_words: int
    ) -> int:
        """Search using Gemini with Google Search grounding."""
        logger.info("Using Gemini with Google Search grounding...")
        prompt = self._build_search_prompt(search_prompt, since_date, summary_words)

        response = self.client.models.generate_content(
            model=Config.MODEL_TEXT,
            contents=prompt,
            config={
                "tools": [{"google_search": {}}],
                "response_mime_type": "application/json",
            },
        )

        if not response.text:
            logger.warning("Empty response from Gemini")
            return 0

        stories_data = self._parse_response(response.text)
        return self._process_stories_data(stories_data)

    def _search_local(
        self, search_prompt: str, since_date: datetime, summary_words: int
    ) -> int:
        """Search using DuckDuckGo and process with Local LLM."""
        logger.info("Using Local LLM with DuckDuckGo search...")

        # Calculate timelimit based on since_date
        days_diff = (datetime.now() - since_date).days
        if days_diff <= 1:
            timelimit = "d"
        elif days_diff <= 7:
            timelimit = "w"
        elif days_diff <= 30:
            timelimit = "m"
        else:
            timelimit = None  # No limit for older searches

        # 1. Search DuckDuckGo
        search_results = []
        try:
            logger.info(f"Querying DuckDuckGo (timelimit={timelimit}): {search_prompt}")
            with DDGS() as ddgs:
                # Try News search first as it's more relevant for "stories"
                logger.info("Attempting DuckDuckGo News search...")
                results = list(
                    ddgs.news(
                        search_prompt,
                        timelimit=timelimit,
                        max_results=10,
                    )
                )

                # If no news, try regular text search
                if not results:
                    logger.info("No news results, trying regular text search...")
                    results = list(
                        ddgs.text(
                            search_prompt,
                            timelimit=timelimit,
                            max_results=10,
                        )
                    )

                # If still no results, try without timelimit as a last resort
                if not results and timelimit:
                    logger.info("No results with timelimit, trying without...")
                    results = list(
                        ddgs.text(
                            search_prompt,
                            timelimit=None,
                            max_results=10,
                        )
                    )

                for i, r in enumerate(results):
                    # Handle different key names between news and text search
                    title = r.get("title") or r.get("title", "No Title")
                    link = r.get("href") or r.get("link") or r.get("url", "")
                    snippet = (
                        r.get("body") or r.get("snippet") or r.get("description", "")
                    )

                    logger.info(f"Found result {i + 1}: {title}")
                    search_results.append(
                        {"title": title, "link": link, "snippet": snippet}
                    )

            logger.info(
                f"DuckDuckGo search complete. Found {len(search_results)} results."
            )
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return 0

        if not search_results:
            logger.warning("No search results from DuckDuckGo")
            return 0

        # 2. Process results with Local LLM
        if not self.local_client:
            logger.error("Local client not available in _search_local")
            return 0

        logger.info(
            f"Processing {len(search_results)} results with Local LLM ({Config.LM_STUDIO_MODEL})..."
        )
        prompt = f"""
You are a news curator. I have found the following search results for the query: "{search_prompt}"

SEARCH RESULTS:
{json.dumps(search_results, indent=2)}

TASK:
1. Select the most relevant and interesting stories.
2. For each story, provide:
   - title: A catchy headline
   - summary: A {summary_words}-word summary
   - sources: A list containing the original link
   - category: One of: Technology, Business, Science, AI, Other
   - quality_score: A score from 1-10 based on relevance and significance

Return the results as a JSON object with a "stories" key containing an array of story objects.
Example:
{{
  "stories": [
    {{
      "title": "Example Story",
      "summary": "This is a summary...",
      "sources": ["https://example.com"],
      "category": "Technology",
      "quality_score": 8
    }}
  ]
}}

Return ONLY the JSON object.
"""

        try:
            response = self.local_client.chat.completions.create(
                model=Config.LM_STUDIO_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
                if "json" in Config.LM_STUDIO_MODEL.lower()
                else {"type": "text"},
                timeout=120,  # Increased timeout for local LLMs
            )
        except Exception as e:
            error_msg = str(e)
            if "No models loaded" in error_msg:
                logger.error(
                    "LM Studio error: No model loaded. Please load a model in LM Studio."
                )
                print("\nERROR: LM Studio has no model loaded.")
                print(
                    "Please open LM Studio, go to the 'Local Server' tab, and load a model."
                )
                return 0
            logger.error(f"Local LLM processing failed: {e}")
            return 0

        content = response.choices[0].message.content
        if not content:
            logger.warning("Empty response from Local LLM")
            return 0

        logger.info("Local LLM processing complete.")
        stories_data = self._parse_response(content)
        return self._process_stories_data(stories_data)

    def _process_stories_data(self, stories_data: list) -> int:
        """Common logic to save stories and update state."""
        if not stories_data:
            logger.warning("No stories found in search results")
            return 0

        logger.info(f"Found {len(stories_data)} potential stories")

        # Save new stories to database
        new_count = self._save_stories(stories_data)

        # Update last check date
        self.db.set_last_check_date()

        logger.info(f"Saved {new_count} new stories to database")
        return new_count

    def _build_search_prompt(
        self, search_prompt: str, since_date: datetime, summary_words: int
    ) -> str:
        """Build the prompt for story search."""
        return f"""
You are a professional news aggregator and analyst.

TASK:
1. Search for recent news stories that match this criteria: "{search_prompt}"
2. Only include stories published after {since_date.strftime("%Y-%m-%d")}.
3. Identify distinct stories. If multiple sources cover the same event/topic, group them as one story with multiple sources.
4. Focus on significant, newsworthy stories that are relevant and well-sourced.

For each story, provide:
- A clear, professional title
- A list of all source URLs covering this story
- A comprehensive summary of exactly {summary_words} words that captures the key facts
- A quality score from 1-10 based on:
  - Relevance to the search criteria (weight: 40%)
  - Significance and newsworthiness (weight: 30%)
  - Source credibility and coverage (weight: 20%)
  - Timeliness and freshness (weight: 10%)

OUTPUT FORMAT (strict JSON):
{{
  "stories": [
    {{
      "title": "Clear Descriptive Story Title",
      "sources": ["https://source1.com/article", "https://source2.com/article"],
      "summary": "The complete {summary_words}-word summary...",
      "quality_score": 8
    }}
  ]
}}

Return an empty array in the "stories" key if no relevant stories are found.
Only include stories that truly match the criteria with quality score >= 5.
"""

    def _parse_response(self, response_text: str) -> list[dict]:
        """Parse the JSON response from the LLM."""
        if not response_text:
            logger.error("Empty response text provided to _parse_response")
            return []

        try:
            # 1. Try direct parsing
            text = response_text.strip()

            # Remove markdown code blocks if present
            if "```" in text:
                # Try to find content between ```json and ``` or just ``` and ```
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
                if json_match:
                    text = json_match.group(1).strip()
                else:
                    # If no closing block, just strip the opening one
                    text = re.sub(r"```(?:json)?", "", text).strip()

            # 2. If it's still not valid, try to find the first [ and last ]
            if not (text.startswith("[") and text.endswith("]")):
                start_idx = text.find("[")
                end_idx = text.rfind("]")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    text = text[start_idx : end_idx + 1]

            stories = json.loads(text)
            if not isinstance(stories, list):
                # If it's a single object, wrap it in a list
                if isinstance(stories, dict):
                    # Check if it's a wrapper object like {"stories": [...]}
                    for key in ["stories", "results", "items"]:
                        if key in stories and isinstance(stories[key], list):
                            return stories[key]
                    return [stories]

                logger.error(f"Response is not a list or valid object: {type(stories)}")
                return []

            return stories

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.info(
                f"Raw response content (first 1000 chars):\n{response_text[:1000]}"
            )

            # Last ditch effort: try to find anything that looks like a JSON array using regex
            try:
                # This is very loose but might catch some cases
                array_match = re.search(r"\[\s*\{.*\}\s*\]", response_text, re.DOTALL)
                if array_match:
                    return json.loads(array_match.group(0))
            except:
                pass

            return []

    def _save_stories(self, stories_data: list[dict]) -> int:
        """Save stories to database, avoiding duplicates. Returns count of new stories."""
        new_count = 0

        for data in stories_data:
            try:
                title = data.get("title", "").strip()
                if not title:
                    continue

                # Get sources from either 'sources' or 'source_links'
                sources = data.get("sources") or data.get("source_links") or []

                # Check for existing story with same title
                existing = self.db.get_story_by_title(title)
                if existing:
                    # Merge sources if same story from different search
                    new_sources = sources
                    existing_sources = set(existing.source_links)
                    updated_sources = list(existing_sources.union(set(new_sources)))

                    if len(updated_sources) > len(existing.source_links):
                        existing.source_links = updated_sources
                        self.db.update_story(existing)
                        logger.debug(f"Updated sources for existing story: {title}")
                    continue

                # Create new story
                quality_score = data.get("quality_score", 5)
                story = Story(
                    title=title,
                    summary=data.get("summary", ""),
                    source_links=sources,
                    acquire_date=datetime.now(),
                    quality_score=quality_score,
                    verification_status="pending",
                    publish_status="unpublished",
                )

                self.db.add_story(story)
                new_count += 1
                logger.info(f"Added new story: {title} (Score: {quality_score})")

            except Exception as e:
                logger.error(f"Failed to save story: {e}")
                continue

        return new_count

    def get_available_story_count(self) -> int:
        """Get the count of stories available for publication."""
        return self.db.count_unpublished_stories()

    def ensure_minimum_stories(self, target_count: int) -> bool:
        """
        Ensure we have at least target_count stories available.
        Returns True if we have enough stories.
        """
        available = self.get_available_story_count()
        if available >= target_count:
            logger.info(f"Have {available} stories available (target: {target_count})")
            return True

        logger.warning(
            f"Only {available} stories available, target is {target_count}. "
            "Previous cycle stories will be used to fill the gap."
        )
        return False
