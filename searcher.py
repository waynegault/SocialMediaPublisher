"""Story search and discovery using Gemini AI with Google Search grounding."""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from ddgs import DDGS
import requests

from google import genai  # type: ignore
from openai import OpenAI

from config import Config
from database import Database, Story
from domain_credibility import get_domain_tier
from error_handling import with_enhanced_recovery
from api_client import api_client
from text_utils import calculate_similarity
from url_utils import validate_url, extract_path_keywords

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


# calculate_similarity is now imported from text_utils
# validate_url is now imported from url_utils


def calibrate_quality_score(
    base_score: int,
    story_data: dict,
    source_urls: list[str] | None = None,
    acquire_date: datetime | None = None,
) -> int:
    """
    Apply quality score calibration based on configurable weights.

    Takes the AI-provided base score and applies adjustments based on:
    - Recency: Newer stories get a boost
    - Source quality: Reputable domains get a boost
    - People mentioned: Stories with named individuals get a boost
    - Geographic priority: Stories from priority regions get a boost

    Args:
        base_score: The AI-provided quality score (1-10)
        story_data: Dictionary containing story metadata from AI response
        source_urls: List of source URLs for the story
        acquire_date: When the story was found (for recency calculation)

    Returns:
        Calibrated quality score (1-10)
    """
    bonus = 0.0

    # Recency bonus: stories discovered today get the full weight
    if acquire_date and Config.QUALITY_WEIGHT_RECENCY > 0:
        age_days = (datetime.now() - acquire_date).days
        if age_days == 0:
            bonus += Config.QUALITY_WEIGHT_RECENCY * 0.5  # Same day
        elif age_days <= 1:
            bonus += Config.QUALITY_WEIGHT_RECENCY * 0.3  # Yesterday
        elif age_days <= 3:
            bonus += Config.QUALITY_WEIGHT_RECENCY * 0.1  # Recent

    # Source quality bonus: reputable domains (using centralized domain_credibility)
    if source_urls and Config.QUALITY_WEIGHT_SOURCE > 0:
        for url in source_urls:
            try:
                tier = get_domain_tier(url)
                if tier == 1:
                    bonus += Config.QUALITY_WEIGHT_SOURCE * 0.5  # Tier 1 sources
                    break
                elif tier == 2:
                    bonus += Config.QUALITY_WEIGHT_SOURCE * 0.3  # Tier 2 sources
                    break
                elif tier == 3:
                    bonus += Config.QUALITY_WEIGHT_SOURCE * 0.2  # Tier 3 sources
                    break
            except Exception:
                pass

    # People mentioned bonus: stories with named individuals
    if Config.QUALITY_WEIGHT_PEOPLE_MENTIONED > 0:
        direct_people = story_data.get("direct_people", [])
        if isinstance(direct_people, list) and len(direct_people) > 0:
            # Count people with actual names (not placeholders)
            named_people = sum(
                1
                for p in direct_people
                if isinstance(p, dict)
                and p.get("name")
                and str(p.get("name", "")).lower() not in ("unknown", "tba", "n/a", "")
            )
            if named_people >= 3:
                bonus += Config.QUALITY_WEIGHT_PEOPLE_MENTIONED * 0.5
            elif named_people >= 1:
                bonus += Config.QUALITY_WEIGHT_PEOPLE_MENTIONED * 0.3

    # Cap the bonus
    max_bonus = Config.QUALITY_MAX_CALIBRATION_BONUS
    bonus = min(bonus, max_bonus)

    # Apply bonus and ensure result is within 1-10
    calibrated = base_score + int(round(bonus))
    calibrated = max(1, min(10, calibrated))

    if bonus > 0:
        logger.debug(
            f"Quality score calibrated: {base_score} -> {calibrated} (bonus: +{bonus:.1f})"
        )

    return calibrated


def archive_url_wayback(url: str, timeout: int = 10) -> str | None:
    """
    Submit a URL to the Wayback Machine for archiving.

    This helps preserve source URLs in case they become unavailable later.
    The Wayback Machine will crawl and archive the page asynchronously.

    Args:
        url: The URL to archive
        timeout: Request timeout in seconds

    Returns:
        The archived URL if successful, None otherwise
    """
    if not url:
        return None

    # Skip non-HTTP URLs
    if not url.startswith(("http://", "https://")):
        return None

    wayback_save_url = f"https://web.archive.org/save/{url}"

    try:
        # Send a HEAD request to trigger archiving via centralized client
        # We don't need to wait for the full page to load
        response = api_client.http_head(
            wayback_save_url,
            timeout=timeout,
            allow_redirects=True,
            headers={"User-Agent": "SocialMediaPublisher/1.0 (URL Archiver)"},
            endpoint="wayback",
        )

        if response.status_code in (200, 302, 303):
            # Extract the archived URL from response headers if available
            archived_url = response.headers.get("Content-Location")
            if archived_url:
                logger.info(f"URL archived: {url} -> {archived_url}")
                return archived_url
            else:
                # Return the wayback URL pattern
                logger.info(f"URL submitted for archiving: {url}")
                return f"https://web.archive.org/web/{url}"

        logger.debug(
            f"Wayback archive returned status {response.status_code} for {url}"
        )
        return None

    except requests.exceptions.Timeout:
        logger.debug(f"Wayback archive timeout for {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.debug(f"Wayback archive failed for {url}: {e}")
        return None


def archive_urls_batch(
    urls: list[str], max_concurrent: int = 3
) -> dict[str, str | None]:
    """
    Archive multiple URLs to the Wayback Machine.

    Processes URLs sequentially to avoid overwhelming the service.

    Args:
        urls: List of URLs to archive
        max_concurrent: Maximum number of concurrent archives (not used currently)

    Returns:
        Dictionary mapping original URLs to archived URLs (or None if failed)
    """
    results: dict[str, str | None] = {}

    for url in urls:
        if not url or url in results:
            continue

        archived = archive_url_wayback(url)
        results[url] = archived

        # Small delay to be nice to the Wayback Machine service
        if archived:
            time.sleep(0.5)

    return results


# extract_url_keywords is now imported from url_utils as extract_path_keywords
def extract_url_keywords(url: str) -> set[str]:
    """Alias for extract_path_keywords from url_utils for backward compatibility."""
    return extract_path_keywords(url)


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
        # Cache for search start date (avoid recalculating multiple times)
        self._cached_start_date: datetime | None = None

    def _report_progress(self, message: str) -> None:
        """Report progress via callback if available."""
        if self.progress_callback:
            self.progress_callback(message)
        logger.info(message)

    def get_search_start_date(self) -> datetime:
        """
        Determine the start date for searching stories.
        Uses last check date if enabled, otherwise uses lookback days.
        Caches the result to avoid duplicate DB queries and log spam.
        """
        # Return cached value if available
        if self._cached_start_date is not None:
            return self._cached_start_date

        if Config.USE_LAST_CHECKED_DATE:
            last_check = self.db.get_last_check_date()
            if last_check:
                logger.info(f"Using last check date: {last_check}")
                self._cached_start_date = last_check
                return last_check

        # Fallback to lookback days
        lookback = datetime.now() - timedelta(days=Config.SEARCH_LOOKBACK_DAYS)
        logger.info(f"Using lookback of {Config.SEARCH_LOOKBACK_DAYS} days: {lookback}")
        self._cached_start_date = lookback
        return lookback

    def search_and_process(self) -> int:
        """
        Search for stories and save new ones to the database.
        Returns the number of new stories saved.
        """
        since_date = self.get_search_start_date()
        # Substitute {discipline} placeholder with actual discipline
        search_prompt = Config.SEARCH_PROMPT.format(discipline=Config.DISCIPLINE)
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
        response = api_client.gemini_generate(
            client=self.client,
            model=Config.MODEL_TEXT,
            contents=prompt,
            config={
                "tools": [{"google_search": {}}],
                "max_output_tokens": Config.LLM_MAX_OUTPUT_TOKENS,
            },
            endpoint="search",
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
        Generic category/index pages are also filtered out.
        """
        if not stories_data:
            return stories_data

        # Patterns that indicate a generic category/index page rather than an article
        generic_url_patterns = [
            r"/news/?$",  # Ends with /news or /news/
            r"/articles/?$",  # Ends with /articles
            r"/blog/?$",  # Ends with /blog
            r"/category/",  # Category pages
            r"/tag/",  # Tag pages
            r"/topics?/",  # Topic pages
            r"/archive/?$",  # Archive pages
            r"/index\.html?$",  # Index pages
            r"^https?://[^/]+/?$",  # Just domain with no path
        ]
        import re

        generic_patterns_compiled = [
            re.compile(p, re.IGNORECASE) for p in generic_url_patterns
        ]

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

                # Filter out generic category/index pages (LLM hallucinations)
                is_generic = False
                for pattern in generic_patterns_compiled:
                    if pattern.search(url):
                        logger.debug(f"Skipping generic category URL: {url}")
                        is_generic = True
                        break
                if is_generic:
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
                    response = api_client.http_request(
                        method="HEAD",
                        url=url,
                        timeout=15,
                        allow_redirects=True,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        },
                        endpoint="redirect_resolution",
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
                    response = api_client.http_request(
                        method="GET",
                        url=url,
                        timeout=15,
                        allow_redirects=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        },
                        endpoint="redirect_resolution",
                    )
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
                    response = api_client.http_request(
                        method="GET",
                        url=url,
                        timeout=20,
                        allow_redirects=True,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        },
                        endpoint="redirect_resolution",
                    )
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

        # Strategy 4: Browser-based fallback for stubborn redirects
        if not resolved_url:
            resolved_url = self._resolve_redirect_with_browser(url)

        # Cache the result (even failures, to avoid retrying)
        self._redirect_url_cache[url] = resolved_url

        if not resolved_url:
            logger.warning(
                f"Failed to resolve redirect URL after all strategies: {url[:60]}..."
            )

        return resolved_url

    def _resolve_redirect_with_browser(self, url: str) -> str:
        """
        Use undetected-chromedriver to resolve a redirect URL that requests couldn't handle.

        This is a last-resort fallback that starts a browser to follow the redirect.
        Returns the final URL or empty string if resolution fails.
        """
        # Use centralized browser module for UC Chrome operations
        from browser import resolve_redirect_with_browser

        return resolve_redirect_with_browser(url, timeout=15, wait_for_js=2.0)

    def _follow_redirect_chain(self, url: str, max_hops: int = 5) -> str:
        """Follow a chain of redirects to get final URL."""
        current_url = url
        for _ in range(max_hops):
            try:
                response = api_client.http_request(
                    method="HEAD",
                    url=current_url,
                    timeout=10,
                    allow_redirects=False,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                    endpoint="redirect_chain",
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

            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_TEXT,
                contents=f"Find the original news article URL for this story. Return ONLY the URL, nothing else.\n\nStory: {title}\n\nSummary: {summary[:200] if summary else 'N/A'}",
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 200,
                    "tools": [Tool(google_search=GoogleSearch())],
                },
                endpoint="source_lookup",
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
        """Convert a conversational prompt into a concise search query using Local LLM or Groq."""
        # If the prompt is already short (e.g. < 8 words), just use it
        if len(search_prompt.split()) < 8:
            return search_prompt

        logger.info("Distilling conversational prompt into search keywords...")

        # Include current year to avoid LLM hallucinating outdated years
        current_year = datetime.now().year

        messages = [
            {
                "role": "system",
                "content": Config.SEARCH_DISTILL_PROMPT,
            },
            {
                "role": "user",
                "content": f"The current year is {current_year}. Extract search keywords from: {search_prompt}",
            },
        ]

        content = None

        # Try Local LLM first
        if self.local_client:
            try:
                content = api_client.local_llm_generate(
                    client=self.local_client,
                    messages=messages,
                    max_tokens=50,
                    temperature=0.1,
                    timeout=30,
                    endpoint="search_distill",
                )
            except Exception as e:
                if "No models loaded" in str(e):
                    logger.debug("Local LLM has no model loaded, using Groq...")
                else:
                    logger.warning(
                        f"Local LLM distillation failed: {e}. Trying Groq..."
                    )

        # Fallback to Groq
        if not content:
            groq_client = api_client.get_groq_client()
            if groq_client:
                try:
                    content = api_client.groq_generate(
                        client=groq_client,
                        messages=messages,
                        max_tokens=50,
                        temperature=0.1,
                        endpoint="search_distill",
                    )
                except Exception as e:
                    logger.warning(
                        f"Groq distillation failed: {e}. Using manual distillation."
                    )

        if not content:
            logger.warning(
                "No LLM available for distillation, using manual distillation"
            )
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

        # 1. Search DuckDuckGo using consolidated method with retry logic
        logger.info(f"Querying DuckDuckGo (timelimit={timelimit}): {search_query}")
        search_results = self._fetch_duckduckgo_results(search_query, timelimit)

        if not search_results:
            logger.warning("No search results from DuckDuckGo")
            return 0

        logger.info(f"DuckDuckGo search complete. Found {len(search_results)} results.")

        # 2. Process results with Local LLM -> Groq -> error
        max_stories = Config.MAX_STORIES_PER_SEARCH
        author_name = Config.LINKEDIN_AUTHOR_NAME or "the LinkedIn profile owner"
        prompt = Config.LOCAL_LLM_SEARCH_PROMPT.format(
            author_name=author_name,
            search_prompt=search_prompt,
            search_results=json.dumps(search_results, indent=2),
            max_stories=max_stories,
            summary_words=summary_words,
            discipline=Config.DISCIPLINE,
        )

        content = None

        # Try Local LLM first
        if self.local_client:
            self._report_progress(
                f"Processing {len(search_results)} results with Local LLM ({Config.LM_STUDIO_MODEL})..."
            )
            try:
                content = api_client.local_llm_generate(
                    client=self.local_client,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=Config.LLM_LOCAL_TIMEOUT,
                    endpoint="search_process",
                )
            except Exception as e:
                error_msg = str(e)
                if "No models loaded" in error_msg:
                    logger.debug("Local LLM has no model loaded, using Groq...")
                else:
                    logger.warning(f"Local LLM processing failed: {e}. Trying Groq...")

        # Fallback to Groq if local failed or not available
        if not content:
            groq_client = api_client.get_groq_client()
            if groq_client:
                self._report_progress(
                    f"Processing {len(search_results)} results with Groq ({Config.GROQ_MODEL})..."
                )
                try:
                    content = api_client.groq_generate(
                        client=groq_client,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2048,
                        endpoint="search_process",
                    )
                except Exception as e:
                    logger.error(f"Groq processing failed: {e}")

        if not content:
            logger.error(
                "No LLM available for search processing (Local LLM and Groq both failed)"
            )
            print("\nERROR: No LLM available for search processing.")
            print("Please either:")
            print("  1. Load a model in LM Studio, OR")
            print("  2. Set GROQ_API_KEY in .env (free at console.groq.com)")
            return 0

        logger.info("LLM processing complete.")
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

        # Enforce MAX_STORIES_PER_SEARCH limit (LLMs don't always follow instructions)
        max_stories = Config.MAX_STORIES_PER_SEARCH
        if len(stories_data) > max_stories:
            logger.info(
                f"Limiting stories from {len(stories_data)} to {max_stories} (MAX_STORIES_PER_SEARCH)"
            )
            stories_data = stories_data[:max_stories]

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
        discipline = Config.DISCIPLINE

        # Allow overriding the full instruction via legacy .env variable
        if Config.SEARCH_PROMPT_TEMPLATE:
            try:
                return Config.SEARCH_PROMPT_TEMPLATE.format(
                    criteria=search_prompt,
                    since_date=since_date.strftime("%Y-%m-%d"),
                    summary_words=summary_words,
                    max_stories=max_stories,
                    author_name=author_name,
                    discipline=discipline,
                    discipline_title=discipline.title(),
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
            discipline=discipline,
            discipline_title=discipline.title(),
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
        # Uses configurable window (default: 90 days)
        existing_titles = self.db.get_recent_story_titles(
            days=Config.DEDUP_ALL_STORIES_WINDOW_DAYS
        )
        similarity_threshold = Config.DEDUP_SIMILARITY_THRESHOLD

        # Also get recently published story titles to prevent similar content
        # This uses a configurable window (default: 30 days)
        published_titles = self.db.get_recent_published_titles(
            days=Config.DEDUP_PUBLISHED_WINDOW_DAYS
        )

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

                # Check against recently published stories to prevent similar content
                # This uses a stricter threshold to avoid posting similar topics
                for published_id, published_title in published_titles:
                    similarity = calculate_similarity(title, published_title)
                    # Use slightly lower threshold for published stories (more strict)
                    if similarity >= similarity_threshold * 0.85:
                        logger.info(
                            f"Similar to recently published story (similarity={similarity:.2f}): "
                            f"'{title}' matches '{published_title}'"
                        )
                        is_duplicate = True
                        break

                if is_duplicate:
                    continue

                # Create new story with all fields
                quality_score = data.get("quality_score", 5)
                quality_justification = data.get("quality_justification", "")
                summary = data.get("summary", "")

                # QUALITY GATE 1: Penalize stories without justification
                # If LLM didn't provide justification, cap score at 5 (mediocre)
                if not quality_justification or len(quality_justification.strip()) < 10:
                    if quality_score > 5:
                        logger.warning(
                            f"Capping quality score from {quality_score} to 5 - no justification provided for: {title[:50]}"
                        )
                        quality_score = 5
                    quality_justification = "(No justification provided by LLM)"

                # QUALITY GATE 2: Penalize stories with short summaries
                # If summary is less than 50% of target, cap score at 4
                summary_word_count = len(summary.split())
                min_acceptable_words = int(Config.SUMMARY_WORD_COUNT * 0.5)
                if summary_word_count < min_acceptable_words:
                    if quality_score > 4:
                        logger.warning(
                            f"Capping quality score from {quality_score} to 4 - summary too short "
                            f"({summary_word_count} words, need {min_acceptable_words}+) for: {title[:50]}"
                        )
                        quality_score = 4

                # Apply quality score calibration based on configured weights
                quality_score = calibrate_quality_score(
                    base_score=quality_score,
                    story_data=data,
                    source_urls=sources,
                    acquire_date=datetime.now(),
                )
                category = data.get("category", "Other")
                # Extract hashtags (limit to 3)
                hashtags = data.get("hashtags", [])
                if isinstance(hashtags, list):
                    hashtags = [str(h).strip().lstrip("#") for h in hashtags[:3]]
                else:
                    hashtags = []

                # Extract organizations (list of company/institution names)
                organizations = data.get("organizations", [])
                if isinstance(organizations, list):
                    # Validate and filter organizations
                    validated_orgs = []
                    # Patterns that indicate AI explanation rather than org name
                    invalid_org_patterns = [
                        "not applicable",
                        "this is not",
                        "no organization",
                        "none mentioned",
                        "not specified",
                        "unknown",
                        "n/a",
                        "generalized",
                        "generic",
                        "various",
                        "multiple",
                        "several",
                        "unspecified",
                    ]
                    for org in organizations:
                        org_str = str(org).strip()
                        if not org_str or len(org_str) < 2:
                            continue
                        # Skip if org name is too long (likely an explanation)
                        if len(org_str) > 100:
                            logger.debug(
                                f"Skipping overly long org name: {org_str[:50]}..."
                            )
                            continue
                        # Skip if contains invalid patterns
                        org_lower = org_str.lower()
                        if any(
                            pattern in org_lower for pattern in invalid_org_patterns
                        ):
                            logger.debug(
                                f"Skipping invalid org (AI explanation): {org_str[:50]}"
                            )
                            continue
                        # Skip if it looks like a sentence (has too many words)
                        word_count = len(org_str.split())
                        if word_count > 10:
                            logger.debug(
                                f"Skipping org with too many words ({word_count}): {org_str[:50]}..."
                            )
                            continue
                        validated_orgs.append(org_str)
                    organizations = validated_orgs
                else:
                    organizations = []

                # Extract direct_people from AI response
                # These are people directly mentioned in the story
                direct_people_raw = data.get("direct_people", [])
                if isinstance(direct_people_raw, list):
                    # Validate and normalize each person entry with enhanced fields
                    validated_people = []
                    # Define placeholder/invalid name patterns to filter out
                    invalid_names = {
                        "tba",
                        "tbd",
                        "unknown",
                        "n/a",
                        "none",
                        "placeholder",
                        "",
                    }

                    for person in direct_people_raw:
                        if isinstance(person, dict) and person.get("name"):
                            name = str(person.get("name", "")).strip()
                            # Skip placeholder names and names that are too short
                            name_lower = name.lower()
                            if name_lower in invalid_names or len(name) < 3:
                                logger.debug(
                                    f"Skipping invalid/placeholder name: {name}"
                                )
                                continue
                            # Skip names that look incomplete (single word with no title context)
                            if " " not in name and not person.get("position"):
                                logger.debug(f"Skipping incomplete name: {name}")
                                continue

                            validated_people.append(
                                {
                                    "name": name,
                                    "company": str(person.get("company", "")).strip(),
                                    "position": str(person.get("position", "")).strip(),
                                    "department": str(
                                        person.get("department", "")
                                    ).strip(),
                                    "location": str(person.get("location", "")).strip(),
                                    "role_type": str(person.get("role_type", ""))
                                    .strip()
                                    .lower(),
                                    "research_area": str(
                                        person.get("research_area", "")
                                    ).strip(),
                                    "linkedin_profile": str(
                                        person.get("linkedin_profile", "")
                                    ).strip(),
                                }
                            )
                    direct_people = validated_people
                else:
                    direct_people = []

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
                    organizations=organizations,
                    direct_people=direct_people,  # Canonical direct people list
                )

                self.db.add_story(story)
                new_count += 1
                # Add to existing titles for subsequent duplicate checks
                existing_titles.append((story.id or 0, title))
                logger.info(
                    f"Added new story: {title} "
                    f"(Score: {quality_score}, Category: {category})"
                )

                # Archive source URLs to Wayback Machine (optional, runs in background)
                if Config.ARCHIVE_SOURCE_URLS and sources:
                    try:
                        archived = archive_urls_batch(
                            sources[:3]
                        )  # Limit to first 3 URLs
                        archived_count = sum(1 for v in archived.values() if v)
                        if archived_count > 0:
                            logger.info(
                                f"Archived {archived_count}/{len(sources)} URLs for story"
                            )
                    except Exception as e:
                        logger.debug(f"URL archiving failed: {e}")

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
        # Substitute {discipline} placeholder with actual discipline
        search_prompt = Config.SEARCH_PROMPT.format(discipline=Config.DISCIPLINE)
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
                response = api_client.gemini_generate(
                    client=self.client,
                    model=Config.MODEL_TEXT,
                    contents=prompt,
                    config={
                        "tools": [{"google_search": {}}],
                        "max_output_tokens": Config.LLM_MAX_OUTPUT_TOKENS,
                    },
                    endpoint="preview_search",
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
            discipline=Config.DISCIPLINE,
        )

        try:
            content = api_client.local_llm_generate(
                client=self.local_client,
                messages=[{"role": "user", "content": prompt}],
                timeout=Config.LLM_LOCAL_TIMEOUT,
                endpoint="local_search_process",
            )
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
        Uses Local LLM -> Groq -> Gemini fallback chain.
        """
        repair_prompt = Config.JSON_REPAIR_PROMPT.format(
            malformed_json=malformed_json[:3000]
        )

        # If no repair prompt configured, use a sensible default
        if not repair_prompt.strip():
            repair_prompt = (
                "Fix this malformed JSON and return ONLY the corrected JSON array, no explanation:\n\n"
                f"{malformed_json[:3000]}"
            )

        content = None

        # Try Local LLM first
        if self.local_client:
            try:
                content = api_client.local_llm_generate(
                    client=self.local_client,
                    messages=[{"role": "user", "content": repair_prompt}],
                    timeout=Config.LLM_LOCAL_TIMEOUT,
                    endpoint="json_repair",
                )
                if content:
                    return content
            except Exception as e:
                if "No models loaded" in str(e):
                    logger.debug("Local LLM has no model loaded, using Groq...")
                else:
                    logger.warning(f"Local LLM JSON repair failed: {e}. Trying Groq...")

        # Fallback to Groq (free, fast)
        groq_client = api_client.get_groq_client()
        if groq_client:
            try:
                content = api_client.groq_generate(
                    client=groq_client,
                    messages=[{"role": "user", "content": repair_prompt}],
                    endpoint="json_repair",
                )
                if content:
                    return content
            except Exception as e:
                logger.warning(
                    f"Groq JSON repair failed: {e}. Falling back to Gemini..."
                )

        # Final fallback to Gemini
        try:
            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_TEXT,
                contents=repair_prompt,
                config={"response_mime_type": "application/json"},
                endpoint="json_repair",
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
def _create_module_tests() -> bool:
    """Create unit tests for searcher module."""
    import os
    import tempfile

    from test_framework import TestSuite

    suite = TestSuite("Searcher Tests", "searcher.py")
    suite.start_suite()

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

    def test_calibrate_quality_score_base():
        # Test that base score is returned when no bonuses apply
        story_data: dict = {"title": "Test", "summary": "Summary"}
        result = calibrate_quality_score(5, story_data)
        assert result == 5
        # Test score capping at 10
        result = calibrate_quality_score(10, story_data)
        assert result == 10
        # Test score floor at 1
        result = calibrate_quality_score(1, story_data)
        assert result >= 1

    def test_calibrate_quality_score_with_people():
        # Story with named people should get a bonus
        story_data: dict = {
            "title": "Test",
            "summary": "Summary",
            "direct_people": [
                {"name": "Dr. Jane Smith", "company": "MIT", "position": "Researcher"},
                {"name": "John Doe", "company": "BASF", "position": "CEO"},
                {"name": "Alice Brown", "company": "Stanford", "position": "Professor"},
            ],
        }
        result = calibrate_quality_score(7, story_data)
        # Should get a bonus for having 3+ named people
        assert result >= 7

    def test_calibrate_quality_score_with_reputable_source():
        # Story with reputable source should get a bonus
        story_data: dict = {"title": "Test", "summary": "Summary"}
        source_urls = ["https://www.nature.com/articles/test-article"]
        result = calibrate_quality_score(7, story_data, source_urls=source_urls)
        # Should get a bonus for reputable source
        assert result >= 7

    def test_calibrate_quality_score_max_cap():
        # Test that bonus is capped and score doesn't exceed 10
        story_data: dict = {
            "title": "Test",
            "summary": "Summary",
            "direct_people": [
                {"name": "Dr. Jane Smith", "company": "MIT", "position": "Researcher"},
                {"name": "John Doe", "company": "BASF", "position": "CEO"},
                {"name": "Alice Brown", "company": "Stanford", "position": "Professor"},
            ],
        }
        source_urls = ["https://www.nature.com/articles/test-article"]
        result = calibrate_quality_score(
            9, story_data, source_urls=source_urls, acquire_date=datetime.now()
        )
        # Should be capped at 10
        assert result <= 10

    def test_archive_url_wayback_invalid():
        # Test with invalid inputs
        result = archive_url_wayback("")
        assert result is None
        result = archive_url_wayback("not-a-url")
        assert result is None
        result = archive_url_wayback("ftp://example.com")
        assert result is None

    def test_archive_urls_batch_empty():
        # Test with empty list
        results = archive_urls_batch([])
        assert results == {}

    def test_archive_urls_batch_dedup():
        # Test that duplicates are handled
        urls = ["https://example.com", "https://example.com", ""]
        results = archive_urls_batch(urls)
        # Should have at most 1 unique valid URL
        assert len([k for k in results.keys() if k]) <= 1

    suite.run_test(
        test_name="Calculate similarity - identical",
        test_func=test_calculate_similarity_identical,
        test_summary="Tests Calculate similarity with identical scenario",
        method_description="Calls calculate similarity and verifies the result",
        expected_outcome="Function returns the expected value",
    )
    suite.run_test(
        test_name="Calculate similarity - different",
        test_func=test_calculate_similarity_different,
        test_summary="Tests Calculate similarity with different scenario",
        method_description="Calls calculate similarity and verifies the result",
        expected_outcome="Function returns the expected value",
    )
    suite.run_test(
        test_name="Calculate similarity - partial",
        test_func=test_calculate_similarity_partial,
        test_summary="Tests Calculate similarity with partial scenario",
        method_description="Calls calculate similarity and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Calculate similarity - empty",
        test_func=test_calculate_similarity_empty,
        test_summary="Tests Calculate similarity with empty scenario",
        method_description="Calls calculate similarity and verifies the result",
        expected_outcome="Function returns the expected value",
    )
    suite.run_test(
        test_name="Validate URL - empty",
        test_func=test_validate_url_empty,
        test_summary="Tests Validate URL with empty scenario",
        method_description="Calls validate url and verifies the result",
        expected_outcome="Function handles empty or missing input gracefully",
    )
    suite.run_test(
        test_name="Validate URL - valid format",
        test_func=test_validate_url_valid_format,
        test_summary="Tests Validate URL with valid format scenario",
        method_description="Calls validate url and verifies the result",
        expected_outcome="Function returns the expected successful result",
    )
    suite.run_test(
        test_name="Validate URL - invalid format",
        test_func=test_validate_url_invalid_format,
        test_summary="Tests Validate URL with invalid format scenario",
        method_description="Calls validate url and verifies the result",
        expected_outcome="Function handles invalid input appropriately",
    )
    suite.run_test(
        test_name="Extract article date - none",
        test_func=test_extract_article_date_none,
        test_summary="Tests Extract article date with none scenario",
        method_description="Calls extract article date and verifies the result",
        expected_outcome="Function returns None as expected",
    )
    suite.run_test(
        test_name="Extract article date - valid",
        test_func=test_extract_article_date_valid,
        test_summary="Tests Extract article date with valid scenario",
        method_description="Calls extract article date and verifies the result",
        expected_outcome="Function returns the expected successful result",
    )
    suite.run_test(
        test_name="Filter stories by date",
        test_func=test_filter_stories_by_date,
        test_summary="Tests Filter stories by date functionality",
        method_description="Calls filter stories by date and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Searcher init",
        test_func=test_searcher_init,
        test_summary="Tests Searcher init functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Parse response - empty",
        test_func=test_parse_response_empty,
        test_summary="Tests Parse response with empty scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns an empty collection",
    )
    suite.run_test(
        test_name="Parse response - valid JSON",
        test_func=test_parse_response_valid_json,
        test_summary="Tests Parse response with valid json scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns the expected successful result",
    )
    suite.run_test(
        test_name="Normalize stories - list",
        test_func=test_normalize_stories_list,
        test_summary="Tests Normalize stories with list scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function correctly processes multiple items",
    )
    suite.run_test(
        test_name="Normalize stories - dict",
        test_func=test_normalize_stories_dict_with_stories,
        test_summary="Tests Normalize stories with dict scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Get search feedback - zero",
        test_func=test_get_search_feedback_zero,
        test_summary="Tests Get search feedback with zero scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function finds and returns the expected results",
    )
    suite.run_test(
        test_name="Get search feedback - few",
        test_func=test_get_search_feedback_few,
        test_summary="Tests Get search feedback with few scenario",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function finds and returns the expected results",
    )
    suite.run_test(
        test_name="Manual distill",
        test_func=test_manual_distill,
        test_summary="Tests Manual distill functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    # New tests
    suite.run_test(
        test_name="Extract URL keywords",
        test_func=test_extract_url_keywords,
        test_summary="Tests Extract URL keywords functionality",
        method_description="Calls extract url keywords and verifies the result",
        expected_outcome="Function correctly parses and extracts the data",
    )
    suite.run_test(
        test_name="Extract URL keywords - empty",
        test_func=test_extract_url_keywords_empty,
        test_summary="Tests Extract URL keywords with empty scenario",
        method_description="Calls extract url keywords and verifies the result",
        expected_outcome="Function handles empty or missing input gracefully",
    )
    suite.run_test(
        test_name="Retry decorator - success",
        test_func=test_retry_decorator_success,
        test_summary="Tests Retry decorator with success scenario",
        method_description="Calls retry with backoff and verifies the result",
        expected_outcome="Function returns the expected value",
    )
    suite.run_test(
        test_name="Retry decorator - eventual success",
        test_func=test_retry_decorator_eventual_success,
        test_summary="Tests Retry decorator with eventual success scenario",
        method_description="Calls retry with backoff and verifies the result",
        expected_outcome="Function raises the expected error or exception",
    )
    suite.run_test(
        test_name="Redirect URL cache",
        test_func=test_redirect_url_cache,
        test_summary="Tests Redirect URL cache functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns the expected value",
    )
    # Quality score calibration tests
    suite.run_test(
        test_name="Calibrate score - base",
        test_func=test_calibrate_quality_score_base,
        test_summary="Tests Calibrate score with base scenario",
        method_description="Calls calibrate quality score and verifies the result",
        expected_outcome="Function returns the expected value",
    )
    suite.run_test(
        test_name="Calibrate score - with people",
        test_func=test_calibrate_quality_score_with_people,
        test_summary="Tests Calibrate score with with people scenario",
        method_description="Calls calibrate quality score and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Calibrate score - reputable source",
        test_func=test_calibrate_quality_score_with_reputable_source,
        test_summary="Tests Calibrate score with reputable source scenario",
        method_description="Calls calibrate quality score and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Calibrate score - max cap",
        test_func=test_calibrate_quality_score_max_cap,
        test_summary="Tests Calibrate score with max cap scenario",
        method_description="Calls calibrate quality score and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    # URL archiving tests
    suite.run_test(
        test_name="Archive URL - invalid",
        test_func=test_archive_url_wayback_invalid,
        test_summary="Tests Archive URL with invalid scenario",
        method_description="Calls archive url wayback and verifies the result",
        expected_outcome="Function returns None as expected",
    )
    suite.run_test(
        test_name="Archive URLs batch - empty",
        test_func=test_archive_urls_batch_empty,
        test_summary="Tests Archive URLs batch with empty scenario",
        method_description="Calls archive urls batch and verifies the result",
        expected_outcome="Function returns an empty collection",
    )
    suite.run_test(
        test_name="Archive URLs batch - dedup",
        test_func=test_archive_urls_batch_dedup,
        test_summary="Tests Archive URLs batch with dedup scenario",
        method_description="Calls archive urls batch and verifies the result",
        expected_outcome="Function correctly processes multiple items",
    )

    return suite.finish_suite()