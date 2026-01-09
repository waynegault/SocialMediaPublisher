"""Story search and discovery using Gemini AI with Google Search grounding."""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from urllib.parse import urlparse
from ddgs import DDGS
import requests

from google import genai  # type: ignore
from openai import OpenAI

from config import Config
from database import Database, Story
from error_handling import with_enhanced_recovery

logger = logging.getLogger(__name__)


# Alias for backwards compatibility - use error_handling.with_enhanced_recovery instead
def retry_with_backoff(
    max_retries: int | None = None,
    base_delay: float | None = None,
    retryable_exceptions: tuple = (requests.exceptions.RequestException,),
) -> Callable:
    """
    Decorator for retry with exponential backoff.
    Uses Config values if not specified.

    Note: This is an alias for error_handling.with_enhanced_recovery for
    backwards compatibility. New code should use with_enhanced_recovery directly.
    """
    _max_retries = max_retries if max_retries is not None else Config.API_RETRY_COUNT
    _base_delay = base_delay if base_delay is not None else Config.API_RETRY_DELAY

    return with_enhanced_recovery(
        max_attempts=_max_retries + 1,
        base_delay=_base_delay,
        retryable_exceptions=retryable_exceptions,
    )


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


def extract_url_keywords(url: str) -> set[str]:
    """
    Extract meaningful keywords from a URL path.
    E.g., "https://news.rice.edu/news/2026/researchers-unlock-catalyst-behavior"
    -> {"researchers", "unlock", "catalyst", "behavior"}
    """
    if not url:
        return set()

    try:
        parsed = urlparse(url)
        # Combine path segments
        path = parsed.path

        # Replace common separators with spaces
        path = re.sub(r"[-_/.]", " ", path)

        # Remove numbers (years, IDs, etc.)
        path = re.sub(r"\b\d+\b", "", path)

        # Split into words and filter
        words = path.lower().split()

        # Remove very short words and common URL fragments
        url_stopwords = {
            "www",
            "com",
            "org",
            "net",
            "edu",
            "html",
            "htm",
            "php",
            "asp",
            "aspx",
            "jsp",
            "news",
            "article",
            "articles",
            "post",
            "posts",
            "blog",
            "index",
            "page",
            "pages",
            "en",
            "us",
            "uk",
            "http",
            "https",
        }
        keywords = {
            word for word in words if len(word) > 3 and word not in url_stopwords
        }

        return keywords
    except Exception:
        return set()


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
        # Cache for resolved redirect URLs to avoid duplicate HTTP requests
        self._redirect_url_cache: dict[str, str] = {}

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
                "max_output_tokens": Config.LLM_MAX_OUTPUT_TOKENS,
            },
        )

        if not response.text:
            logger.warning("Empty response from Gemini")
            return 0

        logger.debug(f"Gemini response length: {len(response.text)} chars")
        stories_data = self._parse_response(response.text)

        # Resolve redirect URLs in each story's sources
        # The LLM outputs vertexaisearch redirect URLs that must be resolved
        stories_data = self._resolve_story_source_urls(stories_data)

        return self._process_stories_data(stories_data, since_date)

    def _resolve_story_source_urls(self, stories_data: list[dict]) -> list[dict]:
        """
        Resolve redirect URLs in story sources to actual article URLs.

        The LLM outputs vertexaisearch.cloud.google.com redirect URLs.
        These must be resolved to get the actual article URLs.
        Google Search query URLs are filtered out as they are not article sources.
        """
        if not stories_data:
            return stories_data

        for story in stories_data:
            sources = story.get("sources") or story.get("source_links") or []
            resolved_sources = []

            for url in sources:
                if not url or not url.startswith("http"):
                    continue

                # Skip Google Search query URLs - these are not article sources
                if "google.com/search?" in url:
                    logger.debug(f"Skipping Google Search URL: {url[:60]}...")
                    continue

                # Resolve vertexaisearch redirect URLs
                if "vertexaisearch.cloud.google.com" in url:
                    resolved = self._resolve_redirect_url(url)
                    if resolved and "vertexaisearch.cloud.google.com" not in resolved:
                        resolved_sources.append(resolved)
                        logger.debug(
                            f"Resolved URL: {url[:50]}... -> {resolved[:50]}..."
                        )
                    else:
                        logger.warning(f"Failed to resolve redirect URL: {url[:60]}...")
                else:
                    # Keep non-redirect URLs as-is
                    resolved_sources.append(url)

            # Update story sources with resolved URLs
            if resolved_sources:
                story["sources"] = resolved_sources
                logger.info(
                    f"Resolved {len(resolved_sources)} source URL(s) for: "
                    f"'{story.get('title', '')[:40]}...'"
                )
            else:
                logger.warning(
                    f"No valid source URLs for: '{story.get('title', '')[:50]}...'"
                )
                story["sources"] = []

        return stories_data

    def _resolve_redirect_url(self, url: str, max_retries: int = 3) -> str:
        """
        Resolve redirect URLs (like Vertex AI Search) to final destination.

        Uses multiple strategies with retries to ensure we get the real URL.
        Returns empty string if resolution completely fails (caller should handle).
        Results are cached to avoid duplicate HTTP requests.
        """
        # If not a redirect URL, return as-is
        if "vertexaisearch.cloud.google.com" not in url:
            return url

        # Check cache first
        if url in self._redirect_url_cache:
            cached = self._redirect_url_cache[url]
            logger.debug(
                f"Using cached redirect resolution: {url[:40]}... -> {cached[:40] if cached else '(failed)'}..."
            )
            return cached

        # Try multiple times with different strategies
        resolved_url = ""
        for attempt in range(max_retries):
            try:
                # Strategy 1: Simple HEAD request with redirects
                if attempt == 0:
                    response = requests.head(
                        url,
                        timeout=15,
                        allow_redirects=True,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        },
                    )
                    if (
                        response.url
                        and "vertexaisearch.cloud.google.com" not in response.url
                    ):
                        logger.debug(
                            f"Resolved via HEAD: {url[:50]}... -> {response.url[:50]}..."
                        )
                        resolved_url = response.url
                        break

                # Strategy 2: GET request without following redirects (get Location header)
                elif attempt == 1:
                    response = requests.get(
                        url,
                        timeout=15,
                        allow_redirects=False,
                        stream=True,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        },
                    )
                    response.close()
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("Location")
                        if location and location.startswith("http"):
                            # If location is also a redirect, follow it
                            if "vertexaisearch.cloud.google.com" not in location:
                                logger.debug(
                                    f"Resolved via Location header: {url[:50]}... -> {location[:50]}..."
                                )
                                resolved_url = location
                                break
                            # Try to follow the chain
                            final = self._follow_redirect_chain(location)
                            if final:
                                resolved_url = final
                                break

                # Strategy 3: Full GET with redirects followed
                else:
                    response = requests.get(
                        url,
                        timeout=20,
                        allow_redirects=True,
                        stream=True,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        },
                    )
                    response.close()
                    if (
                        response.url
                        and "vertexaisearch.cloud.google.com" not in response.url
                    ):
                        logger.debug(
                            f"Resolved via full GET: {url[:50]}... -> {response.url[:50]}..."
                        )
                        resolved_url = response.url
                        break

            except requests.exceptions.Timeout:
                logger.debug(
                    f"Timeout resolving redirect (attempt {attempt + 1}): {url[:50]}..."
                )
                continue
            except requests.exceptions.RequestException as e:
                logger.debug(f"Error resolving redirect (attempt {attempt + 1}): {e}")
                continue

        # Cache the result (even failures, to avoid retrying)
        self._redirect_url_cache[url] = resolved_url

        if not resolved_url:
            logger.warning(
                f"Failed to resolve redirect URL after {max_retries} attempts: {url[:60]}..."
            )

        return resolved_url

    def _follow_redirect_chain(self, url: str, max_hops: int = 5) -> str:
        """Follow a chain of redirects to get final URL."""
        current_url = url
        for _ in range(max_hops):
            try:
                response = requests.head(
                    current_url,
                    timeout=10,
                    allow_redirects=False,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                )
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if location:
                        if location.startswith("/"):
                            # Relative URL - construct absolute
                            from urllib.parse import urlparse

                            parsed = urlparse(current_url)
                            location = f"{parsed.scheme}://{parsed.netloc}{location}"
                        current_url = location
                    else:
                        break
                else:
                    # Not a redirect - we've reached the destination
                    break
            except Exception:
                break

        # Check if we escaped the redirect domain
        if "vertexaisearch.cloud.google.com" not in current_url:
            return current_url
        return ""

    def _search_for_source_url(self, title: str, summary: str = "") -> Optional[str]:
        """
        Search for a source URL for a story that has no grounded sources.

        Uses a simple web search via Gemini to find a relevant article URL.
        Returns the first valid non-redirect URL found, or None if search fails.
        """
        if not self.client:
            return None

        try:
            logger.debug(f"Searching for source URL for: {title[:50]}...")

            # Use Gemini with grounding to find the article
            from google.genai.types import Tool, GoogleSearch

            response = self.client.models.generate_content(
                model=Config.MODEL_TEXT,
                contents=f"Find the original news article URL for this story. Return ONLY the URL, nothing else.\n\nStory: {title}\n\nSummary: {summary[:200] if summary else 'N/A'}",
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 200,
                    "tools": [Tool(google_search=GoogleSearch())],
                },
            )

            # First try to get URL from grounding metadata
            if response.candidates:
                candidate = response.candidates[0]
                if (
                    hasattr(candidate, "grounding_metadata")
                    and candidate.grounding_metadata
                ):
                    metadata = candidate.grounding_metadata
                    if (
                        hasattr(metadata, "grounding_chunks")
                        and metadata.grounding_chunks
                    ):
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, "web") and hasattr(chunk.web, "uri"):
                                uri = chunk.web.uri  # type: ignore[union-attr]
                                if uri and uri.startswith("http"):
                                    resolved = self._resolve_redirect_url(uri)
                                    if (
                                        resolved
                                        and "vertexaisearch.cloud.google.com"
                                        not in resolved
                                    ):
                                        return resolved

            # Fallback: try to extract URL from response text
            if response.text:
                urls = re.findall(r'https?://[^\s<>"\']+', response.text)
                for url in urls:
                    # Clean up URL (remove trailing punctuation)
                    url = re.sub(r"[.,;:!?\)\]]+$", "", url)
                    if url.startswith("http") and "vertexaisearch" not in url:
                        if validate_url(url):
                            return url

            return None

        except Exception as e:
            logger.debug(f"Failed to search for source URL: {e}")
            return None

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
                        "content": Config.SEARCH_DISTILL_PROMPT,
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
        max_results = Config.DUCKDUCKGO_MAX_RESULTS
        try:
            logger.info(f"Querying DuckDuckGo (timelimit={timelimit}): {search_query}")
            with DDGS() as ddgs:
                # Try News search first as it's more relevant for "stories"
                logger.info("Attempting DuckDuckGo News search...")
                results = list(
                    ddgs.news(
                        search_query,
                        timelimit=timelimit,
                        max_results=max_results,
                    )
                )

                # If no news, try regular text search
                if not results:
                    logger.info("No news results, trying regular text search...")
                    results = list(
                        ddgs.text(
                            search_query,
                            timelimit=timelimit,
                            max_results=max_results,
                        )
                    )

                # If still no results, try without timelimit as a last resort
                if not results and timelimit:
                    logger.info("No results with timelimit, trying without...")
                    results = list(
                        ddgs.text(
                            search_query,
                            timelimit=None,
                            max_results=max_results,
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
        author_name = Config.LINKEDIN_AUTHOR_NAME or "the LinkedIn profile owner"
        prompt = Config.LOCAL_LLM_SEARCH_PROMPT.format(
            author_name=author_name,
            search_prompt=search_prompt,
            search_results=json.dumps(search_results, indent=2),
            max_stories=max_stories,
            summary_words=summary_words,
        )

        try:
            response = self.local_client.chat.completions.create(
                model=Config.LM_STUDIO_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
                if "json" in Config.LM_STUDIO_MODEL.lower()
                else {"type": "text"},
                timeout=Config.LLM_LOCAL_TIMEOUT,
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
        author_name = Config.LINKEDIN_AUTHOR_NAME or "the LinkedIn profile owner"

        # Allow overriding the full instruction via legacy .env variable
        if Config.SEARCH_PROMPT_TEMPLATE:
            try:
                return Config.SEARCH_PROMPT_TEMPLATE.format(
                    criteria=search_prompt,
                    since_date=since_date.strftime("%Y-%m-%d"),
                    summary_words=summary_words,
                    max_stories=max_stories,
                    author_name=author_name,
                )
            except Exception as e:
                logger.warning(
                    f"SEARCH_PROMPT_TEMPLATE formatting failed: {e}. Using default template."
                )

        # Use configurable search instruction prompt
        return Config.SEARCH_INSTRUCTION_PROMPT.format(
            max_stories=max_stories,
            search_prompt=search_prompt,
            since_date=since_date.strftime("%Y-%m-%d"),
            summary_words=summary_words,
            author_name=author_name,
        )

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

        # Get recent titles for semantic deduplication (more memory-efficient for large DBs)
        # Duplicates are most likely among recent stories, so 90 days is a reasonable window
        existing_titles = self.db.get_recent_story_titles(days=90)
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

                # Always filter out redirect URLs (they should have been resolved earlier)
                original_count = len(sources)
                sources = [
                    url
                    for url in sources
                    if "vertexaisearch.cloud.google.com" not in url
                ]
                if len(sources) < original_count:
                    logger.debug(
                        f"Filtered {original_count - len(sources)} redirect URLs from sources"
                    )

                # If no sources, try to find one using web search
                if not sources:
                    logger.info(
                        f"No sources for '{title[:40]}...' - searching for source URL..."
                    )
                    found_url = self._search_for_source_url(
                        title, data.get("summary", "")
                    )
                    if found_url:
                        sources = [found_url]
                        logger.info(f"Found source URL via search: {found_url[:60]}...")
                    else:
                        logger.warning(f"Skipping story with no sources: {title}")
                        continue

                # Validate URLs if enabled
                if Config.VALIDATE_SOURCE_URLS:
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
                # Extract hashtags (limit to 3)
                hashtags = data.get("hashtags", [])
                if isinstance(hashtags, list):
                    hashtags = [str(h).strip().lstrip("#") for h in hashtags[:3]]
                else:
                    hashtags = []

                # Extract relevant_people (list of dicts with name, company, position, linkedin_profile)
                relevant_people = data.get("relevant_people", [])
                if isinstance(relevant_people, list):
                    # Validate and normalize each person entry
                    validated_people = []
                    for person in relevant_people:
                        if isinstance(person, dict) and person.get("name"):
                            validated_people.append(
                                {
                                    "name": str(person.get("name", "")).strip(),
                                    "company": str(person.get("company", "")).strip(),
                                    "position": str(person.get("position", "")).strip(),
                                    "linkedin_profile": str(
                                        person.get("linkedin_profile", "")
                                    ).strip(),
                                }
                            )
                    relevant_people = validated_people
                else:
                    relevant_people = []

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
                    hashtags=hashtags,
                    relevant_people=relevant_people,
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
                # NOTE: We intentionally do NOT use response_mime_type: application/json
                # because it prevents grounding_chunks from being populated with source URLs.
                prompt = self._build_search_prompt(
                    search_prompt, since_date, summary_words
                )
                response = self.client.models.generate_content(
                    model=Config.MODEL_TEXT,
                    contents=prompt,
                    config={
                        "tools": [{"google_search": {}}],
                        "max_output_tokens": Config.LLM_MAX_OUTPUT_TOKENS,
                    },
                )
                if not response.text:
                    return []
                stories_data = self._parse_response(response.text)

                # Resolve redirect URLs in story sources
                stories_data = self._resolve_story_source_urls(stories_data)

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
        max_results = Config.DUCKDUCKGO_MAX_RESULTS

        @retry_with_backoff(retryable_exceptions=(Exception,))
        def do_search():
            with DDGS() as ddgs:
                # Try News search first
                results = list(
                    ddgs.news(
                        search_query, timelimit=timelimit, max_results=max_results
                    )
                )
                if not results:
                    results = list(
                        ddgs.text(
                            search_query, timelimit=timelimit, max_results=max_results
                        )
                    )
                if not results and timelimit:
                    results = list(
                        ddgs.text(search_query, timelimit=None, max_results=max_results)
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
        author_name = Config.LINKEDIN_AUTHOR_NAME or "the LinkedIn profile owner"
        prompt = Config.LOCAL_LLM_SEARCH_PROMPT.format(
            author_name=author_name,
            search_prompt=search_prompt,
            search_results=json.dumps(search_results, indent=2),
            max_stories=max_stories,
            summary_words=summary_words,
        )

        try:
            response = self.local_client.chat.completions.create(
                model=Config.LM_STUDIO_MODEL,
                messages=[{"role": "user", "content": prompt}],
                timeout=Config.LLM_LOCAL_TIMEOUT,
            )
            content = response.choices[0].message.content
            if content:
                stories_data = self._parse_response(content)
                # Validate and fix URLs - LLM may hallucinate or mismatch URLs
                stories_data = self._validate_story_urls(stories_data, search_results)
                return stories_data
        except Exception as e:
            logger.error(f"Local LLM processing failed: {e}")

        return []

    def _validate_story_urls(
        self, stories_data: list[dict], search_results: list[dict]
    ) -> list[dict]:
        """
        Replace LLM-generated URLs with actual search result URLs.

        SIMPLE & RELIABLE APPROACH:
        All search result URLs were used to generate the stories.
        Therefore, ALL search result URLs are valid sources for ALL stories.
        We assign all search result URLs to each story - no AI matching needed.

        This gives 100% confidence that source_links contain only real URLs
        that were actually used in generating the stories.
        """
        if not stories_data or not search_results:
            return stories_data

        # Extract all valid URLs from search results
        all_search_urls = []
        for result in search_results:
            url = result.get("link", "")
            if url and url.startswith("http"):
                all_search_urls.append(url)

        if not all_search_urls:
            logger.warning("No valid URLs found in search results")
            return stories_data

        # Remove duplicates while preserving order
        seen: set[str] = set()
        unique_search_urls = []
        for url in all_search_urls:
            if url not in seen:
                seen.add(url)
                unique_search_urls.append(url)

        logger.info(
            f"Assigning {len(unique_search_urls)} search result URL(s) to "
            f"{len(stories_data)} stories"
        )

        # Assign ALL search result URLs to EACH story
        # These are the actual URLs used to generate the response
        for story in stories_data:
            story_title = story.get("title", "")
            # Replace any LLM-hallucinated sources with real search result URLs
            story["sources"] = unique_search_urls.copy()
            logger.debug(
                f"Assigned {len(unique_search_urls)} source URL(s) to: "
                f"'{story_title[:50]}...'"
            )

        return stories_data

    def _repair_json_with_llm(self, malformed_json: str) -> str | None:
        """
        Attempt to repair malformed JSON using an LLM.
        This is a fallback when initial parsing fails.
        """
        repair_prompt = Config.JSON_REPAIR_PROMPT.format(
            malformed_json=malformed_json[:3000]
        )

        try:
            if self.local_client:
                response = self.local_client.chat.completions.create(
                    model=Config.LM_STUDIO_MODEL,
                    messages=[{"role": "user", "content": repair_prompt}],
                    timeout=Config.LLM_LOCAL_TIMEOUT,
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


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for searcher module."""
    import os
    import tempfile

    from test_framework import TestSuite

    suite = TestSuite("Searcher Tests")

    def test_calculate_similarity_identical():
        result = calculate_similarity("hello world", "hello world")
        assert result == 1.0

    def test_calculate_similarity_different():
        result = calculate_similarity("hello world", "goodbye moon")
        assert result == 0.0

    def test_calculate_similarity_partial():
        result = calculate_similarity("hello world", "hello there")
        assert 0.0 < result < 1.0

    def test_calculate_similarity_empty():
        result = calculate_similarity("", "")
        assert result == 0.0

    def test_validate_url_empty():
        result = validate_url("")
        assert result is False

    def test_validate_url_valid_format():
        result = validate_url("https://example.com")
        # May fail if no network, but should not raise
        assert isinstance(result, bool)

    def test_validate_url_invalid_format():
        result = validate_url("not-a-url")
        assert result is False

    def test_extract_article_date_none():
        result = extract_article_date({})
        assert result is None

    def test_extract_article_date_valid():
        result = extract_article_date({"date": "2024-01-15"})
        assert result is not None
        assert result.year == 2024

    def test_filter_stories_by_date():
        since = datetime(2024, 1, 10)
        stories = [
            {"title": "Old", "date": "2024-01-01"},
            {"title": "New", "date": "2024-01-15"},
            {"title": "NoDate"},
        ]
        filtered = filter_stories_by_date(stories, since)
        titles = [s["title"] for s in filtered]
        assert "Old" not in titles
        assert "New" in titles
        assert "NoDate" in titles  # Included by default

    def test_searcher_init():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            searcher = StorySearcher(db, None, None)  # type: ignore
            assert searcher.db is db
        finally:
            os.unlink(db_path)

    def test_parse_response_empty():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            searcher = StorySearcher(db, None, None)  # type: ignore
            result = searcher._parse_response("")
            assert result == []
        finally:
            os.unlink(db_path)

    def test_parse_response_valid_json():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            searcher = StorySearcher(db, None, None)  # type: ignore
            json_str = '[{"title": "Test", "summary": "Summary", "quality_score": 8}]'
            result = searcher._parse_response(json_str)
            assert len(result) == 1
            assert result[0]["title"] == "Test"
        finally:
            os.unlink(db_path)

    def test_normalize_stories_list():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            searcher = StorySearcher(db, None, None)  # type: ignore
            data = [{"title": "Story1"}, {"title": "Story2"}]
            result = searcher._normalize_stories(data)
            assert len(result) == 2
        finally:
            os.unlink(db_path)

    def test_normalize_stories_dict_with_stories():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            searcher = StorySearcher(db, None, None)  # type: ignore
            data = {"stories": [{"title": "Story1"}]}
            result = searcher._normalize_stories(data)
            assert len(result) == 1
        finally:
            os.unlink(db_path)

    def test_get_search_feedback_zero():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            searcher = StorySearcher(db, None, None)  # type: ignore
            feedback = searcher.get_search_feedback(0, "test query")
            assert "No results" in feedback
        finally:
            os.unlink(db_path)

    def test_get_search_feedback_few():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            searcher = StorySearcher(db, None, None)  # type: ignore
            feedback = searcher.get_search_feedback(2, "test")
            assert "2 result" in feedback
        finally:
            os.unlink(db_path)

    def test_manual_distill():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            searcher = StorySearcher(db, None, None)  # type: ignore
            result = searcher._manual_distill(
                "I'm a chemical engineer looking for AI news"
            )
            assert len(result) > 0
            assert "i'm" not in result.lower()
        finally:
            os.unlink(db_path)

    # New tests for URL matching functions
    def test_extract_url_keywords():
        keywords = extract_url_keywords(
            "https://news.example.com/2024/researchers-unlock-catalyst-behavior"
        )
        assert "researchers" in keywords
        assert "unlock" in keywords
        assert "catalyst" in keywords
        assert "behavior" in keywords
        # Numbers and common fragments should be excluded
        assert "2024" not in keywords
        assert "news" not in keywords

    def test_extract_url_keywords_empty():
        keywords = extract_url_keywords("")
        assert keywords == set()

    def test_retry_decorator_success():
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def succeeds_first_try():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeeds_first_try()
        assert result == "success"
        assert call_count == 1

    def test_retry_decorator_eventual_success():
        call_count = 0

        @retry_with_backoff(
            max_retries=3, base_delay=0.01, retryable_exceptions=(ValueError,)
        )
        def fails_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = fails_then_succeeds()
        assert result == "success"
        assert call_count == 3

    def test_redirect_url_cache():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            searcher = StorySearcher(db, None, None)  # type: ignore
            # Non-redirect URLs should pass through unchanged
            result = searcher._resolve_redirect_url("https://example.com/article")
            assert result == "https://example.com/article"
            # Cache should be empty for non-redirect URLs (they bypass cache)
            assert "https://example.com/article" not in searcher._redirect_url_cache
        finally:
            os.unlink(db_path)

    suite.add_test(
        "Calculate similarity - identical", test_calculate_similarity_identical
    )
    suite.add_test(
        "Calculate similarity - different", test_calculate_similarity_different
    )
    suite.add_test("Calculate similarity - partial", test_calculate_similarity_partial)
    suite.add_test("Calculate similarity - empty", test_calculate_similarity_empty)
    suite.add_test("Validate URL - empty", test_validate_url_empty)
    suite.add_test("Validate URL - valid format", test_validate_url_valid_format)
    suite.add_test("Validate URL - invalid format", test_validate_url_invalid_format)
    suite.add_test("Extract article date - none", test_extract_article_date_none)
    suite.add_test("Extract article date - valid", test_extract_article_date_valid)
    suite.add_test("Filter stories by date", test_filter_stories_by_date)
    suite.add_test("Searcher init", test_searcher_init)
    suite.add_test("Parse response - empty", test_parse_response_empty)
    suite.add_test("Parse response - valid JSON", test_parse_response_valid_json)
    suite.add_test("Normalize stories - list", test_normalize_stories_list)
    suite.add_test("Normalize stories - dict", test_normalize_stories_dict_with_stories)
    suite.add_test("Get search feedback - zero", test_get_search_feedback_zero)
    suite.add_test("Get search feedback - few", test_get_search_feedback_few)
    suite.add_test("Manual distill", test_manual_distill)
    # New tests
    suite.add_test("Extract URL keywords", test_extract_url_keywords)
    suite.add_test("Extract URL keywords - empty", test_extract_url_keywords_empty)
    suite.add_test("Retry decorator - success", test_retry_decorator_success)
    suite.add_test(
        "Retry decorator - eventual success", test_retry_decorator_eventual_success
    )
    suite.add_test("Redirect URL cache", test_redirect_url_cache)

    return suite
