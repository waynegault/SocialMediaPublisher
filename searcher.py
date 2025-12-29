"""Story search and discovery using Gemini AI with Google Search grounding."""

import json
import logging
from datetime import datetime, timedelta
from duckduckgo_search import DDGS

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

        # 1. Search DuckDuckGo
        search_results = []
        try:
            with DDGS() as ddgs:
                results = ddgs.text(
                    f"{search_prompt} after:{since_date.strftime('%Y-%m-%d')}",
                    max_results=10,
                )
                for r in results:
                    search_results.append(
                        {"title": r["title"], "link": r["href"], "snippet": r["body"]}
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

        prompt = f"""
You are a news curator. I have found the following search results for the query: "{search_prompt}"

SEARCH RESULTS:
{json.dumps(search_results, indent=2)}

TASK:
1. Select the most relevant and interesting stories.
2. For each story, provide:
   - title: A catchy headline
   - summary: A {summary_words}-word summary
   - source_links: A list containing the original link
   - category: One of: Technology, Business, Science, AI, Other

Return the results as a JSON array of objects.
Example:
[
  {{
    "title": "Example Story",
    "summary": "This is a summary...",
    "source_links": ["https://example.com"],
    "category": "Technology"
  }}
]

Return ONLY the JSON array.
"""

        response = self.local_client.chat.completions.create(
            model=Config.MODEL_TEXT,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
            if "json" in Config.LM_STUDIO_MODEL.lower()
            else None,
        )

        content = response.choices[0].message.content
        if not content:
            logger.warning("Empty response from Local LLM")
            return 0

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
[
  {{
    "title": "Clear Descriptive Story Title",
    "sources": ["https://source1.com/article", "https://source2.com/article"],
    "summary": "The complete {summary_words}-word summary...",
    "quality_score": 8
  }}
]

Return an empty array [] if no relevant stories are found.
Only include stories that truly match the criteria with quality score >= 5.
"""

    def _parse_response(self, response_text: str) -> list[dict]:
        """Parse the JSON response from Gemini."""
        try:
            # Clean up response if needed
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            stories = json.loads(text)
            if not isinstance(stories, list):
                logger.error("Response is not a list")
                return []

            return stories

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return []

    def _save_stories(self, stories_data: list[dict]) -> int:
        """Save stories to database, avoiding duplicates. Returns count of new stories."""
        new_count = 0

        for data in stories_data:
            try:
                title = data.get("title", "").strip()
                if not title:
                    continue

                # Check for existing story with same title
                existing = self.db.get_story_by_title(title)
                if existing:
                    # Merge sources if same story from different search
                    new_sources = data.get("sources", [])
                    existing_sources = set(existing.source_links)
                    updated_sources = list(existing_sources.union(set(new_sources)))

                    if len(updated_sources) > len(existing.source_links):
                        existing.source_links = updated_sources
                        self.db.update_story(existing)
                        logger.debug(f"Updated sources for existing story: {title}")
                    continue

                # Create new story
                story = Story(
                    title=title,
                    summary=data.get("summary", ""),
                    source_links=data.get("sources", []),
                    acquire_date=datetime.now(),
                    quality_score=data.get("quality_score", 5),
                    verification_status="pending",
                    publish_status="unpublished",
                )

                self.db.add_story(story)
                new_count += 1
                logger.debug(f"Added new story: {title}")

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
