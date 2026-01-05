"""Story search and discovery using Gemini AI with Google Search grounding."""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any
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
                "max_output_tokens": 8192,
            },
        )

        if not response.text:
            logger.warning("Empty response from Gemini")
            return 0

        logger.debug(f"Gemini response length: {len(response.text)} chars")
        stories_data = self._parse_response(response.text)
        return self._process_stories_data(stories_data)

    def _get_search_query(self, search_prompt: str) -> str:
        """Convert a conversational prompt into a concise search query using Local LLM."""
        if not self.local_client:
            return search_prompt

        # If the prompt is already short (e.g. < 8 words), just use it
        if len(search_prompt.split()) < 8:
            return search_prompt

        logger.info("Distilling conversational prompt into search keywords...")
        prompt = f"""
Convert the following conversational request into a concise, effective search engine query (keywords only).
Request: "{search_prompt}"
Search Query:"""

        try:
            response = self.local_client.chat.completions.create(
                model=Config.LM_STUDIO_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a search query optimizer. Convert long requests into 3-5 keyword search terms.",
                    },
                    {
                        "role": "user",
                        "content": f"Extract search keywords from: {search_prompt}",
                    },
                ],
                max_tokens=50,
                temperature=0.1,
                timeout=30,
            )
            content = response.choices[0].message.content
            if not content:
                logger.warning("Local LLM returned empty content for distillation")
                return self._manual_distill(search_prompt)

            logger.info(f"LLM distillation raw output: '{content.strip()}'")

            query = content.strip().strip('"').strip()
            # Remove common prefixes LLMs add
            query = re.sub(
                r"^(Search Query:|Keywords:|Query:)\s*", "", query, flags=re.IGNORECASE
            )

            # If the LLM just echoed the whole thing or returned something too long, manual distill
            if len(query.split()) > 10:
                logger.info("LLM output too long, using manual distillation")
                return self._manual_distill(search_prompt)

            if not query:
                return self._manual_distill(search_prompt)

            logger.info(f"Distilled query: {query}")
            return query
        except Exception as e:
            logger.warning(
                f"Failed to distill search query: {e}. Using manual distillation."
            )
            return self._manual_distill(search_prompt)

    def _manual_distill(self, text: str) -> str:
        """Fallback to extract keywords if LLM fails."""
        # Remove common conversational filler
        fillers = [
            "i'm a",
            "i am a",
            "looking for",
            "the latest",
            "stories i can",
            "summarise for",
            "publication on",
            "my linkedin profile",
            "professional",
            "please",
            "find",
            "search for",
            "latest",
        ]

        query = text.lower()
        for filler in fillers:
            query = query.replace(filler, " ")

        # Clean up whitespace and take first 8 words
        words = [w for w in query.split() if len(w) >= 2]

        # If we have very few words, try to keep them all
        if len(words) <= 3:
            result = " ".join(words).strip()
        else:
            result = " ".join(words[:8]).strip()

        if not result:
            # Last resort: just take the first 5 words of original
            result = " ".join(text.split()[:5])

        logger.info(f"Manual distillation result: {result}")
        return result

    def _search_local(
        self, search_prompt: str, since_date: datetime, summary_words: int
    ) -> int:
        """Search using DuckDuckGo and process with Local LLM."""
        logger.info("Using Local LLM with DuckDuckGo search...")

        # Distill conversational prompt into keywords for better search results
        search_query = self._get_search_query(search_prompt)

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
            logger.info(f"Querying DuckDuckGo (timelimit={timelimit}): {search_query}")
            with DDGS() as ddgs:
                # Try News search first as it's more relevant for "stories"
                logger.info("Attempting DuckDuckGo News search...")
                results = list(
                    ddgs.news(
                        search_query,
                        timelimit=timelimit,
                        max_results=10,
                    )
                )

                # If no news, try regular text search
                if not results:
                    logger.info("No news results, trying regular text search...")
                    results = list(
                        ddgs.text(
                            search_query,
                            timelimit=timelimit,
                            max_results=10,
                        )
                    )

                # If still no results, try without timelimit as a last resort
                if not results and timelimit:
                    logger.info("No results with timelimit, trying without...")
                    results = list(
                        ddgs.text(
                            search_query,
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
            if "no results" in str(e).lower():
                logger.warning(f"DuckDuckGo search returned no results: {e}")
            else:
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
        # Allow overriding the full instruction via .env with placeholders
        if Config.SEARCH_PROMPT_TEMPLATE:
            try:
                return Config.SEARCH_PROMPT_TEMPLATE.format(
                    criteria=search_prompt,
                    since_date=since_date.strftime("%Y-%m-%d"),
                    summary_words=summary_words,
                )
            except Exception as e:
                logger.warning(
                    f"SEARCH_PROMPT_TEMPLATE formatting failed: {e}. Using default template."
                )

        return f"""
You are a news curator. Find 3-5 recent news stories matching: "{search_prompt}"

REQUIREMENTS:
- Stories must be from after {since_date.strftime("%Y-%m-%d")}
- Each story needs: title, sources (URLs), summary ({summary_words} words max), quality_score (1-10)

RESPOND WITH ONLY THIS JSON FORMAT:
{{
  "stories": [
    {{
      "title": "Story Title",
      "sources": ["https://example.com/article"],
      "summary": "Brief summary here.",
      "quality_score": 8
    }}
  ]
}}

IMPORTANT: Return complete, valid JSON. Keep summaries concise.
"""

    def _parse_response(self, response_text: str) -> list[dict]:
        """Parse the JSON response from the LLM."""
        if not response_text:
            logger.error("Empty response text provided to _parse_response")
            return []

        text = response_text.strip()

        # 1. Try to find JSON block in markdown (more permissive)
        match = re.search(r"```\w*\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()

        # 2. Try to parse as is
        try:
            data = json.loads(text)
            return self._normalize_stories(data)
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed: {e}")

        # 3. Try to find the outermost JSON object or array
        start_brace = text.find("{")
        start_bracket = text.find("[")

        start_idx = -1
        if start_brace != -1 and start_bracket != -1:
            start_idx = min(start_brace, start_bracket)
        elif start_brace != -1:
            start_idx = start_brace
        elif start_bracket != -1:
            start_idx = start_bracket

        if start_idx != -1:
            # Find the corresponding last brace/bracket
            end_idx = max(text.rfind("}"), text.rfind("]"))
            if end_idx > start_idx:
                candidate = text[start_idx : end_idx + 1]
                try:
                    data = json.loads(candidate)
                    return self._normalize_stories(data)
                except json.JSONDecodeError as e:
                    logger.debug(f"Outermost JSON extraction failed: {e}")

        # 4. Fallback: Try to find ANY list in the text
        try:
            match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return self._normalize_stories(data)
        except Exception as e:
            logger.debug(f"List regex extraction failed: {e}")

        # 5. Salvage: Try to extract individual story objects
        logger.info("Attempting to salvage individual story objects from response...")
        stories = []

        # Look specifically for story objects by finding patterns like {"title" or { "title"
        # This skips the outer wrapper and finds actual story objects
        story_starts = []
        for match in re.finditer(r'\{\s*"title"\s*:', text):
            story_starts.append(match.start())

        logger.debug(f"Found {len(story_starts)} potential story start positions")

        for start in story_starts:
            # Try to find the matching closing brace for this story object
            balance = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    balance += 1
                elif text[i] == "}":
                    balance -= 1
                    if balance == 0:
                        # Found a complete object
                        candidate = text[start : i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict) and "title" in obj:
                                stories.append(obj)
                                logger.info(
                                    f"Salvaged story: {obj.get('title', 'Unknown')[:60]}"
                                )
                        except json.JSONDecodeError as e:
                            logger.debug(f"Failed to parse story candidate: {e}")
                        break

        if stories:
            logger.info(f"Salvaged {len(stories)} stories")
            return stories

        logger.error("Failed to parse JSON response")
        logger.info(f"Raw response content (first 1000 chars):\n{response_text[:1000]}")
        return []

    def _normalize_stories(self, data: Any) -> list[dict]:
        """Helper to extract stories list from various JSON structures."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ["stories", "results", "items", "news"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # Maybe the dict itself is a story?
            if "title" in data:
                return [data]
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
