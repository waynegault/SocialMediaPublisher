"""Story search and discovery using Gemini AI with Google Search grounding."""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Callable
from urllib.parse import urlparse
from functools import wraps
from ddgs import DDGS
import requests

from google import genai  # type: ignore
from openai import OpenAI

from config import Config
from database import Database, Story

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int | None = None,
    base_delay: float | None = None,
    retryable_exceptions: tuple = (requests.exceptions.RequestException,),
) -> Callable:
    """
    Decorator for retry with exponential backoff.
    Uses Config values if not specified.
    """
    _max_retries = max_retries if max_retries is not None else Config.API_RETRY_COUNT
    _base_delay = base_delay if base_delay is not None else Config.API_RETRY_DELAY

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception: Exception | None = None
            for attempt in range(_max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < _max_retries:
                        delay = _base_delay * (2**attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {_max_retries + 1} attempts failed.")
                        raise
            # This should never be reached, but satisfies type checker
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic error")

        return wrapper

    return decorator


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts using word sets.
    Returns a value between 0.0 (no similarity) and 1.0 (identical).
    """
    # Normalize: lowercase, remove punctuation, split into words
    words1 = set(re.sub(r"[^\w\s]", "", text1.lower()).split())
    words2 = set(re.sub(r"[^\w\s]", "", text2.lower()).split())

    # Remove very common words that don't carry meaning
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
    }
    words1 -= stopwords
    words2 -= stopwords

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


def validate_url(url: str) -> bool:
    """
    Validate a URL format and optionally check accessibility.
    Returns True if URL is valid and accessible.
    """
    if not url:
        return False

    # Parse URL
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return False
        if parsed.scheme not in ("http", "https"):
            return False
    except Exception:
        return False

    # Optional: Check accessibility (HEAD request with short timeout)
    if Config.VALIDATE_SOURCE_URLS:
        try:
            # Try HEAD first (faster)
            response = requests.head(url, timeout=5, allow_redirects=True)
            # Some servers don't support HEAD, try GET on 405
            if response.status_code == 405:
                response = requests.get(url, timeout=5, allow_redirects=True)
            if response.status_code >= 400:
                logger.debug(f"URL returned {response.status_code}: {url}")
                return False
            return True
        except requests.exceptions.Timeout:
            # Timeout is acceptable - site might be slow but exists
            logger.debug(f"URL timeout (accepting anyway): {url}")
            return True
        except requests.exceptions.RequestException as e:
            # Connection errors, DNS failures, etc. - reject the URL
            logger.debug(f"URL validation failed ({type(e).__name__}): {url}")
            return False

    return True


def extract_article_date(story_data: dict) -> datetime | None:
    """
    Extract article date from story data.
    Looks for common date fields and parses various formats.
    Returns None if no valid date found.
    """
    # Common date field names
    date_fields = [
        "date",
        "published_date",
        "publish_date",
        "article_date",
        "created_at",
    ]

    for field in date_fields:
        if field in story_data and story_data[field]:
            date_str = story_data[field]
            # Try various date formats
            formats = [
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%B %d, %Y",
                "%b %d, %Y",
                "%d %B %Y",
                "%d %b %Y",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

    return None


def filter_stories_by_date(stories: list[dict], since_date: datetime) -> list[dict]:
    """
    Filter stories to only include those published after since_date.
    Stories without a parseable date are included (benefit of the doubt).
    """
    filtered = []
    for story in stories:
        article_date = extract_article_date(story)
        if article_date is None:
            # No date found, include it
            filtered.append(story)
        elif article_date >= since_date:
            filtered.append(story)
        else:
            logger.debug(
                f"Filtered out story (too old): {story.get('title', 'Unknown')[:50]} "
                f"(date: {article_date.date()})"
            )
    return filtered


class StorySearcher:
    """Search for and process news stories using Gemini AI or Local LLM."""

    def __init__(
        self,
        database: Database,
        client: genai.Client,
        local_client: OpenAI | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        """Initialize the story searcher."""
        self.db = database
        self.client = client
        self.local_client = local_client
        self.progress_callback = progress_callback
        # Cache for preview mode - stores discovered stories before save
        self._preview_stories: list[dict] = []
        # Cache for raw search results (for retry without re-fetching)
        self._cached_search_results: list[dict] = []

    def _report_progress(self, message: str) -> None:
        """Report progress via callback if available."""
        if self.progress_callback:
            self.progress_callback(message)
        logger.info(message)

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

        # NOTE: We intentionally do NOT use response_mime_type: application/json
        # because it prevents grounding_chunks from being populated with source URLs.
        # Instead, we ask the model to output JSON in plain text and parse it.
        response = self.client.models.generate_content(
            model=Config.MODEL_TEXT,
            contents=prompt,
            config={
                "tools": [{"google_search": {}}],
                "max_output_tokens": 8192,
            },
        )

        if not response.text:
            logger.warning("Empty response from Gemini")
            return 0

        logger.debug(f"Gemini response length: {len(response.text)} chars")
        stories_data = self._parse_response(response.text)

        # Extract real URLs from grounding metadata (not hallucinated by LLM)
        # This only works when response_mime_type is NOT set to application/json
        grounding_sources = self._extract_grounding_urls(response)
        if grounding_sources:
            logger.info(
                f"Extracted {len(grounding_sources)} real URLs from grounding metadata"
            )
            # Replace hallucinated sources with real grounded URLs
            stories_data = self._replace_sources_with_grounding(
                stories_data, grounding_sources
            )
        else:
            logger.warning(
                "No grounding URLs found in response metadata - sources may be unreliable"
            )

        return self._process_stories_data(stories_data, since_date)

    def _extract_grounding_urls(self, response: Any) -> list[dict]:
        """
        Extract real URLs from Gemini grounding metadata.
        Returns list of dicts with 'uri' and 'title' keys.

        Tries multiple sources:
        1. grounding_chunks (preferred - has individual source URLs)
        2. search_entry_point.rendered_content (fallback - HTML with search links)
        """
        sources: list[dict] = []
        try:
            if not response.candidates:
                return sources
            candidate = response.candidates[0]
            if not hasattr(candidate, "grounding_metadata"):
                return sources
            metadata = candidate.grounding_metadata
            if not metadata:
                return sources

            # Try grounding_chunks first (preferred)
            if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                for chunk in metadata.grounding_chunks:
                    if hasattr(chunk, "web") and hasattr(chunk.web, "uri"):
                        uri = chunk.web.uri
                        title = (
                            getattr(chunk.web, "title", "")
                            if hasattr(chunk, "web")
                            else ""
                        )
                        if uri and uri.startswith("http"):
                            # Resolve Vertex AI Search redirect URLs
                            final_uri = self._resolve_redirect_url(uri)
                            sources.append({"uri": final_uri, "title": title})

            # Fallback: extract from search_entry_point HTML
            if not sources and hasattr(metadata, "search_entry_point"):
                entry_point = metadata.search_entry_point
                if (
                    hasattr(entry_point, "rendered_content")
                    and entry_point.rendered_content
                ):
                    # Parse URLs from the HTML carousel
                    html_content = entry_point.rendered_content
                    # Find all href URLs in the HTML
                    url_matches = re.findall(r'href="([^"]+)"', html_content)
                    for url in url_matches:
                        if url.startswith("http"):
                            final_uri = self._resolve_redirect_url(url)
                            sources.append({"uri": final_uri, "title": ""})

        except Exception as e:
            logger.debug(f"Could not extract grounding URLs: {e}")
        return sources

    def _resolve_redirect_url(self, url: str) -> str:
        """Resolve redirect URLs (like Vertex AI Search) to final destination."""
        # If it's a vertexaisearch redirect, try to follow it
        if "vertexaisearch.cloud.google.com" in url:
            try:
                # First, get the initial redirect without following
                # This is faster and avoids timeouts on slow destination sites
                response = requests.get(
                    url, timeout=10, allow_redirects=False, stream=True
                )
                response.close()

                # Check for redirect (3xx status)
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if location and location.startswith("http"):
                        logger.debug(
                            f"Resolved redirect: {url[:60]}... -> {location[:60]}..."
                        )
                        return location

                # If not a redirect, try following all redirects
                response = requests.get(
                    url, timeout=10, allow_redirects=True, stream=True
                )
                response.close()
                if response.url and response.url != url:
                    logger.debug(
                        f"Resolved redirect: {url[:60]}... -> {response.url[:60]}..."
                    )
                    return response.url
            except Exception as e:
                logger.debug(f"Failed to resolve redirect {url[:50]}: {e}")
        return url

    def _replace_sources_with_grounding(
        self, stories_data: list[dict], grounding_sources: list[dict]
    ) -> list[dict]:
        """
        Replace LLM-generated sources with real grounding URLs.
        Tries to match sources to stories by title similarity.
        """
        if not grounding_sources:
            return stories_data

        num_stories = len(stories_data)
        if num_stories == 0:
            return stories_data

        # Extract just the URIs
        grounding_urls = [s["uri"] for s in grounding_sources]
        grounding_titles = [s.get("title", "").lower() for s in grounding_sources]

        for story in stories_data:
            story_title = story.get("title", "").lower()
            story_summary = story.get("summary", "").lower()
            matched_urls: list[str] = []

            # Try to find matching sources by title/content similarity
            for url, source_title in zip(grounding_urls, grounding_titles):
                # Check if source title appears in story or vice versa
                if source_title and (
                    source_title in story_title
                    or source_title in story_summary
                    or any(
                        word in story_title
                        for word in source_title.split()
                        if len(word) > 4
                    )
                ):
                    matched_urls.append(url)

            # If no matches found, assign based on position
            if not matched_urls and grounding_urls:
                # Give each story at least one URL from the pool
                idx = stories_data.index(story)
                urls_per_story = max(1, len(grounding_urls) // num_stories)
                start = idx * urls_per_story
                end = (
                    start + urls_per_story
                    if idx < num_stories - 1
                    else len(grounding_urls)
                )
                matched_urls = grounding_urls[start:end]

            if matched_urls:
                # Remove duplicates while preserving order
                seen: set[str] = set()
                unique_urls = [
                    u for u in matched_urls if not (u in seen or seen.add(u))
                ]  # type: ignore
                story["sources"] = unique_urls
                logger.debug(
                    f"Assigned {len(unique_urls)} grounded URLs to "
                    f"'{story.get('title', 'Unknown')[:40]}...'"
                )

        return stories_data

    def _get_search_query(self, search_prompt: str) -> str:
        """Convert a conversational prompt into a concise search query using Local LLM."""
        if not self.local_client:
            return search_prompt

        # If the prompt is already short (e.g. < 8 words), just use it
        if len(search_prompt.split()) < 8:
            return search_prompt

        logger.info("Distilling conversational prompt into search keywords...")

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

        self._report_progress(
            f"Processing {len(search_results)} results with Local LLM ({Config.LM_STUDIO_MODEL})..."
        )
        max_stories = Config.MAX_STORIES_PER_SEARCH
        prompt = f"""
You are a news curator. I have found the following search results for the query: "{search_prompt}"

SEARCH RESULTS:
{json.dumps(search_results, indent=2)}

TASK:
1. Select up to {max_stories} of the most relevant and interesting stories.
2. For each story, provide:
   - title: A catchy headline
   - summary: A {summary_words}-word summary
   - sources: A list containing the original link
   - category: One of: Technology, Business, Science, AI, Other
   - quality_score: A score from 1-10 based on relevance and significance
   - quality_justification: Brief explanation of the score

Return the results as a JSON object with a "stories" key containing an array of story objects.
Example:
{{
  "stories": [
    {{
      "title": "Example Story",
      "summary": "This is a summary...",
      "sources": ["https://example.com"],
      "category": "Technology",
      "quality_score": 8,
      "quality_justification": "Highly relevant, reputable source"
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
        return self._process_stories_data(stories_data, since_date)

    def _process_stories_data(
        self, stories_data: list, since_date: datetime | None = None
    ) -> int:
        """Common logic to save stories and update state."""
        if not stories_data:
            logger.warning("No stories found in search results")
            return 0

        logger.info(f"Found {len(stories_data)} potential stories")

        # Apply date post-filtering if since_date provided
        if since_date:
            original_count = len(stories_data)
            stories_data = filter_stories_by_date(stories_data, since_date)
            if len(stories_data) < original_count:
                logger.info(
                    f"Date filter: {original_count} -> {len(stories_data)} stories"
                )

        if not stories_data:
            logger.warning("No stories remain after date filtering")
            return 0

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
        max_stories = Config.MAX_STORIES_PER_SEARCH

        # Allow overriding the full instruction via .env with placeholders
        if Config.SEARCH_PROMPT_TEMPLATE:
            try:
                return Config.SEARCH_PROMPT_TEMPLATE.format(
                    criteria=search_prompt,
                    since_date=since_date.strftime("%Y-%m-%d"),
                    summary_words=summary_words,
                    max_stories=max_stories,
                )
            except Exception as e:
                logger.warning(
                    f"SEARCH_PROMPT_TEMPLATE formatting failed: {e}. Using default template."
                )

        return f"""
You are a news curator. Find {max_stories} recent news stories matching: "{search_prompt}"

REQUIREMENTS:
- Stories must be from after {since_date.strftime("%Y-%m-%d")}
- Each story needs:
  * title: A clear, engaging headline
  * sources: Array of REAL source URLs from your search results
  * summary: {summary_words} words max
  * category: One of: Technology, Business, Science, AI, Other
  * quality_score: 1-10 rating
  * quality_justification: Brief explanation of the score

CRITICAL: Only include URLs you found in your search results. Do NOT invent or guess URLs.
If you cannot find a real URL for a story, omit that story entirely.

RESPOND WITH ONLY THIS JSON FORMAT:
{{
  "stories": [
    {{
      "title": "Story Title",
      "sources": ["https://real-url-from-search.com/article"],
      "summary": "Brief summary here.",
      "category": "Technology",
      "quality_score": 8,
      "quality_justification": "Highly relevant topic, reputable source, timely"
    }}
  ]
}}

IMPORTANT: Return complete, valid JSON. Keep summaries concise. Use ONLY real URLs.
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

        # 6. Last resort: Try LLM-based JSON repair
        logger.info("Attempting LLM-based JSON repair...")
        repaired = self._repair_json_with_llm(response_text)
        if repaired:
            try:
                data = json.loads(repaired)
                stories = self._normalize_stories(data)
                if stories:
                    logger.info(f"LLM repair successful: {len(stories)} stories")
                    return stories
            except json.JSONDecodeError:
                logger.debug("LLM repair produced invalid JSON")

        logger.error("Failed to parse JSON response after all attempts")
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

        # Get existing titles for semantic deduplication
        existing_titles = self.db.get_all_story_titles()
        similarity_threshold = Config.DEDUP_SIMILARITY_THRESHOLD

        for i, data in enumerate(stories_data):
            try:
                title = data.get("title", "").strip()
                if not title:
                    continue

                self._report_progress(
                    f"Processing story {i + 1}/{len(stories_data)}: {title}"
                )

                # Get sources from either 'sources' or 'source_links'
                sources = data.get("sources") or data.get("source_links") or []

                # Validate URLs if enabled
                if Config.VALIDATE_SOURCE_URLS and sources:
                    valid_sources = [url for url in sources if validate_url(url)]
                    if len(valid_sources) < len(sources):
                        logger.debug(
                            f"Filtered {len(sources) - len(valid_sources)} invalid URLs"
                        )
                    sources = valid_sources
                    if not sources:
                        logger.warning(f"Skipping story with no valid sources: {title}")
                        continue

                # Check for exact title match first
                existing = self.db.get_story_by_title(title)
                if existing:
                    # Merge sources if same story from different search
                    existing_sources = set(existing.source_links)
                    updated_sources = list(existing_sources.union(set(sources)))

                    if len(updated_sources) > len(existing.source_links):
                        existing.source_links = updated_sources
                        self.db.update_story(existing)
                        logger.debug(f"Updated sources for existing story: {title}")
                    continue

                # Semantic deduplication: check similarity against existing titles
                is_duplicate = False
                for existing_id, existing_title in existing_titles:
                    similarity = calculate_similarity(title, existing_title)
                    if similarity >= similarity_threshold:
                        logger.info(
                            f"Semantic duplicate detected (similarity={similarity:.2f}): "
                            f"'{title}' matches '{existing_title}'"
                        )
                        # Merge sources into the existing story
                        existing_story = self.db.get_story(existing_id)
                        if existing_story:
                            existing_sources = set(existing_story.source_links)
                            updated_sources = list(existing_sources.union(set(sources)))
                            if len(updated_sources) > len(existing_story.source_links):
                                existing_story.source_links = updated_sources
                                self.db.update_story(existing_story)
                        is_duplicate = True
                        break

                if is_duplicate:
                    continue

                # Create new story with all fields
                quality_score = data.get("quality_score", 5)
                category = data.get("category", "Other")
                quality_justification = data.get("quality_justification", "")

                story = Story(
                    title=title,
                    summary=data.get("summary", ""),
                    source_links=sources,
                    acquire_date=datetime.now(),
                    quality_score=quality_score,
                    category=category,
                    quality_justification=quality_justification,
                    verification_status="pending",
                    publish_status="unpublished",
                )

                self.db.add_story(story)
                new_count += 1
                # Add to existing titles for subsequent duplicate checks
                existing_titles.append((story.id or 0, title))
                logger.info(
                    f"Added new story: {title} "
                    f"(Score: {quality_score}, Category: {category})"
                )

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

    def search_preview(self) -> list[dict]:
        """
        Search for stories but don't save them - return for preview.
        Caches results for later selective saving.
        """
        since_date = self.get_search_start_date()
        search_prompt = Config.SEARCH_PROMPT
        summary_words = Config.SUMMARY_WORD_COUNT

        self._report_progress(f"Searching for preview: '{search_prompt[:50]}...'")

        try:
            if self.local_client:
                # Use DuckDuckGo search
                search_query = self._get_search_query(search_prompt)
                days_diff = (datetime.now() - since_date).days
                if days_diff <= 1:
                    timelimit = "d"
                elif days_diff <= 7:
                    timelimit = "w"
                else:
                    timelimit = "m"

                search_results = self._fetch_duckduckgo_results(search_query, timelimit)
                if not search_results:
                    return []

                # Cache raw results
                self._cached_search_results = search_results

                # Process with LLM
                stories_data = self._process_with_local_llm(
                    search_results, search_prompt, summary_words
                )
            else:
                # Use Gemini
                prompt = self._build_search_prompt(
                    search_prompt, since_date, summary_words
                )
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
                    return []
                stories_data = self._parse_response(response.text)

            self._preview_stories = stories_data
            return stories_data

        except Exception as e:
            logger.error(f"Preview search failed: {e}")
            return []

    def save_selected_stories(self, indices: list[int]) -> int:
        """
        Save selected stories from preview by their indices.
        Returns count of stories saved.
        """
        if not self._preview_stories:
            logger.warning("No preview stories available. Run search_preview first.")
            return 0

        selected = [
            self._preview_stories[i]
            for i in indices
            if 0 <= i < len(self._preview_stories)
        ]

        if not selected:
            logger.warning("No valid indices provided")
            return 0

        saved = self._save_stories(selected)
        self.db.set_last_check_date()

        # Clear preview cache
        self._preview_stories = []
        return saved

    def save_all_preview_stories(self) -> int:
        """Save all stories from preview."""
        if not self._preview_stories:
            return 0
        return self.save_selected_stories(list(range(len(self._preview_stories))))

    def _fetch_duckduckgo_results(
        self, search_query: str, timelimit: str | None
    ) -> list[dict]:
        """Fetch search results from DuckDuckGo with retry logic."""
        search_results = []

        @retry_with_backoff(retryable_exceptions=(Exception,))
        def do_search():
            with DDGS() as ddgs:
                # Try News search first
                results = list(
                    ddgs.news(search_query, timelimit=timelimit, max_results=10)
                )
                if not results:
                    results = list(
                        ddgs.text(search_query, timelimit=timelimit, max_results=10)
                    )
                if not results and timelimit:
                    results = list(
                        ddgs.text(search_query, timelimit=None, max_results=10)
                    )
                return results

        try:
            self._report_progress(f"Searching DuckDuckGo: {search_query}")
            results = do_search()

            for r in results:
                title = r.get("title", "No Title")
                link = r.get("href") or r.get("link") or r.get("url", "")
                snippet = r.get("body") or r.get("snippet") or r.get("description", "")
                search_results.append(
                    {"title": title, "link": link, "snippet": snippet}
                )

            self._report_progress(f"Found {len(search_results)} search results")

        except Exception as e:
            logger.error(f"DuckDuckGo search failed after retries: {e}")

        return search_results

    def _process_with_local_llm(
        self, search_results: list[dict], search_prompt: str, summary_words: int
    ) -> list[dict]:
        """Process search results with local LLM."""
        if not self.local_client:
            return []

        max_stories = Config.MAX_STORIES_PER_SEARCH
        prompt = f"""
You are a news curator. I have found the following search results for: "{search_prompt}"

SEARCH RESULTS:
{json.dumps(search_results, indent=2)}

TASK: Select up to {max_stories} of the BEST stories and provide for each:
- title: A clear, engaging headline based on the search result
- summary: {summary_words} words max summarizing the story
- sources: Array containing ONLY the exact "link" URL from the search result above
- category: One of: Technology, Business, Science, AI, Other
- quality_score: 1-10 rating
- quality_justification: Brief explanation of the score

CRITICAL: For the "sources" field, you MUST use the EXACT "link" URL from the search
results provided above. Do NOT invent, modify, or guess URLs. Copy the link exactly.

Return JSON: {{"stories": [...]}}
"""

        try:
            response = self.local_client.chat.completions.create(
                model=Config.LM_STUDIO_MODEL,
                messages=[{"role": "user", "content": prompt}],
                timeout=120,
            )
            content = response.choices[0].message.content
            if content:
                return self._parse_response(content)
        except Exception as e:
            logger.error(f"Local LLM processing failed: {e}")

        return []

    def _repair_json_with_llm(self, malformed_json: str) -> str | None:
        """
        Attempt to repair malformed JSON using an LLM.
        This is a fallback when initial parsing fails.
        """
        repair_prompt = f"""
The following JSON is malformed. Please fix it and return ONLY the corrected JSON:

{malformed_json[:3000]}

Return ONLY valid JSON, no explanation.
"""

        try:
            if self.local_client:
                response = self.local_client.chat.completions.create(
                    model=Config.LM_STUDIO_MODEL,
                    messages=[{"role": "user", "content": repair_prompt}],
                    timeout=60,
                )
                return response.choices[0].message.content
            else:
                response = self.client.models.generate_content(
                    model=Config.MODEL_TEXT,
                    contents=repair_prompt,
                    config={"response_mime_type": "application/json"},
                )
                return response.text
        except Exception as e:
            logger.error(f"JSON repair failed: {e}")
            return None

    def get_search_feedback(self, results_count: int, query: str) -> str:
        """
        Generate actionable feedback when search returns few or no results.
        """
        if results_count == 0:
            suggestions = [
                "• Try broadening your search terms",
                "• Remove specific date constraints",
                "• Use more general keywords",
                f"• Current query: '{query[:50]}...'",
            ]
            return "No results found. Suggestions:\n" + "\n".join(suggestions)
        elif results_count < 3:
            return (
                f"Only {results_count} result(s) found. "
                "Consider broadening your search criteria for more options."
            )
        return f"Found {results_count} stories."
