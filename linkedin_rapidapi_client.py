"""
LinkedIn Profile Lookup API Client (RapidAPI)

Provides LinkedIn profile search by name and company using the
Fresh LinkedIn Profile Data API on RapidAPI.

API OPTIONS:
1. Fresh LinkedIn Profile Data ($10/month for 500 requests)
   - https://rapidapi.com/freshdata-freshdata-default/api/fresh-linkedin-profile-data
   - Subscribe to "Basic" plan

2. Browser-based fallback (FREE) - uses DuckDuckGo search via Selenium
   - Works automatically when no RapidAPI key is configured
   - Slower but free and reliable

HOW TO GET RAPIDAPI KEY:
1. Go to RapidAPI.com and create an account
2. Search for "Fresh LinkedIn Profile Data"
3. Subscribe to Basic plan ($10/month, 500 requests)
4. Copy your API key from the X-RapidAPI-Key header
5. Set RAPIDAPI_KEY in your .env file

API Documentation: https://rapidapi.com/freshdata-freshdata-default/api/fresh-linkedin-profile-data
"""

import logging
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable

import requests

from api_client import api_client
from config import Config

logger = logging.getLogger(__name__)


@dataclass
class LinkedInProfileResult:
    """Result from LinkedIn profile search."""

    linkedin_url: str
    public_id: str  # e.g., "john-smith-123"
    first_name: str
    last_name: str
    full_name: str
    headline: str
    job_title: str
    company: str
    company_linkedin_url: str = ""
    location: str = ""
    profile_image_url: str = ""
    about: str = ""
    match_score: float = 0.0

    # Additional fields for validation
    company_domain: str = ""
    company_industry: str = ""
    connection_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching."""
        return {
            "linkedin_url": self.linkedin_url,
            "public_id": self.public_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "headline": self.headline,
            "job_title": self.job_title,
            "company": self.company,
            "company_linkedin_url": self.company_linkedin_url,
            "location": self.location,
            "profile_image_url": self.profile_image_url,
            "about": self.about,
            "match_score": self.match_score,
            "company_domain": self.company_domain,
            "company_industry": self.company_industry,
            "connection_count": self.connection_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinkedInProfileResult":
        """Create from dictionary (cache retrieval)."""
        return cls(
            linkedin_url=data.get("linkedin_url", ""),
            public_id=data.get("public_id", ""),
            first_name=data.get("first_name", ""),
            last_name=data.get("last_name", ""),
            full_name=data.get("full_name", ""),
            headline=data.get("headline", ""),
            job_title=data.get("job_title", ""),
            company=data.get("company", ""),
            company_linkedin_url=data.get("company_linkedin_url", ""),
            location=data.get("location", ""),
            profile_image_url=data.get("profile_image_url", ""),
            about=data.get("about", ""),
            match_score=data.get("match_score", 0.0),
            company_domain=data.get("company_domain", ""),
            company_industry=data.get("company_industry", ""),
            connection_count=data.get("connection_count", 0),
        )


class FreshLinkedInAPIClient:
    """
    Client for Fresh LinkedIn Profile Data API on RapidAPI.

    Provides reliable LinkedIn profile search by name and company.
    Includes caching, rate limiting, and match scoring.

    PRICING: $10/month for 500 requests (Basic plan)

    Subscribe at: https://rapidapi.com/freshdata-freshdata-default/api/fresh-linkedin-profile-data
    """

    # RapidAPI endpoint - Fresh LinkedIn Profile Data
    BASE_URL = "https://fresh-linkedin-profile-data.p.rapidapi.com"
    API_HOST = "fresh-linkedin-profile-data.p.rapidapi.com"

    def __init__(self, api_key: Optional[str] = None, cache: Optional[Any] = None):
        """
        Initialize the LinkedIn API client.

        Args:
            api_key: RapidAPI key. If not provided, reads from Config.RAPIDAPI_KEY
            cache: Optional cache object with get/set methods for storing results
        """
        self.api_key = api_key or getattr(Config, "RAPIDAPI_KEY", None)
        if not self.api_key:
            logger.info(
                "No RapidAPI key configured. Browser-based lookup will be used as fallback."
            )

        self.cache = cache
        # Rate limiting now handled by api_client centrally
        self._request_count = 0
        self._cache_hits = 0
        self._api_calls = 0

        # Use centralized session factory with retry logic
        self._session = api_client.get_session(
            name="linkedin_rapidapi",
            retries=3,
            backoff_factor=0.5,
        )
        self._session.headers.update(
            {
                "X-RapidAPI-Key": self.api_key or "",
                "X-RapidAPI-Host": self.API_HOST,
                "Accept": "application/json",
            }
        )

    def _get_cache_key(self, name: str, company: str) -> str:
        """Generate a cache key for a name+company lookup."""
        normalized = f"{name.lower().strip()}|{company.lower().strip()}"
        return f"linkedin_profile:{hashlib.md5(normalized.encode()).hexdigest()}"

    def _rate_limit(self):
        """Enforce rate limiting using centralized api_client limiter."""
        wait_time = api_client.linkedin_limiter.wait(endpoint="rapidapi")
        if wait_time > 0.5:
            logger.debug(f"RapidAPI rate limiter: waited {wait_time:.1f}s")
        self._request_count += 1

    def _check_from_cache(
        self, name: str, company: str
    ) -> Optional[LinkedInProfileResult]:
        """Check if result exists in cache."""
        if not self.cache:
            return None

        cache_key = self._get_cache_key(name, company)
        cached = self.cache.get(cache_key)

        if cached:
            self._cache_hits += 1
            logger.debug(f"Cache hit for {name} at {company}")
            return LinkedInProfileResult.from_dict(cached)

        return None

    def _save_to_cache(
        self,
        name: str,
        company: str,
        result: LinkedInProfileResult,
        ttl: int = 86400 * 30,
    ):
        """Save result to cache (default 30 days TTL)."""
        if not self.cache:
            return

        cache_key = self._get_cache_key(name, company)
        self.cache.set(cache_key, result.to_dict(), ttl=ttl)
        logger.debug(f"Cached result for {name} at {company}")

    def _extract_company_from_headline(self, headline: str, target_company: str) -> str:
        """
        Extract company name from a headline like "CEO at Microsoft".
        Falls back to target company if company is found in headline.
        """
        if not headline:
            return ""

        # Check if target company is mentioned in headline
        if target_company.lower() in headline.lower():
            return target_company

        # Try to extract company after common patterns
        headline_lower = headline.lower()
        patterns = [" at ", " @ ", " - "]
        for pattern in patterns:
            if pattern in headline_lower:
                parts = headline.split(pattern if pattern != " - " else " - ")
                if len(parts) >= 2:
                    return parts[-1].strip()

        return ""

    def _calculate_match_score(
        self, result: Dict[str, Any], target_name: str, target_company: str
    ) -> float:
        """
        Calculate how well a result matches the search criteria.

        Returns a score from 0.0 to 1.0 where 1.0 is a perfect match.
        """
        score = 0.0

        # Name matching (50% of score)
        # Support both first_name/last_name and full_name
        result_name = result.get("full_name", "").lower().strip()
        if not result_name:
            result_name = f"{result.get('first_name', '')} {result.get('last_name', '')}".lower().strip()
        target_name_lower = target_name.lower().strip()

        if result_name == target_name_lower:
            score += 0.5
        elif target_name_lower in result_name or result_name in target_name_lower:
            score += 0.35
        else:
            # Partial name match
            target_parts = set(target_name_lower.split())
            result_parts = set(result_name.split())
            overlap = len(target_parts & result_parts)
            if overlap > 0:
                score += 0.25 * (overlap / max(len(target_parts), 1))

        # Company matching (40% of score)
        result_company = result.get("company", "").lower().strip()
        target_company_lower = target_company.lower().strip()

        if result_company == target_company_lower:
            score += 0.4
        elif (
            target_company_lower in result_company
            or result_company in target_company_lower
        ):
            score += 0.3
        else:
            # Check headline for company mention
            headline = result.get("headline", "").lower()
            if target_company_lower in headline:
                score += 0.25
            else:
                # Partial company match
                target_parts = set(target_company_lower.split())
                result_parts = set(result_company.split())
                overlap = len(target_parts & result_parts)
                if overlap > 0:
                    score += 0.2 * (overlap / max(len(target_parts), 1))

        # Title/Role bonus (10% of score)
        job_title = result.get("job_title", "").lower()
        headline = result.get("headline", "").lower()

        # Common executive titles get a small bonus
        executive_titles = [
            "ceo",
            "cto",
            "cfo",
            "coo",
            "president",
            "founder",
            "director",
            "vp",
            "vice president",
            "head of",
            "chief",
        ]
        for title in executive_titles:
            if title in job_title or title in headline:
                score += 0.1
                break

        return min(score, 1.0)

    def search_person(
        self,
        name: str,
        company: str,
        use_cache: bool = True,
        min_match_score: float = 0.5,
    ) -> Optional[LinkedInProfileResult]:
        """
        Search for a person by name and company.

        Args:
            name: Person's full name (e.g., "John Smith")
            company: Company name (e.g., "Microsoft")
            use_cache: Whether to check/use cache
            min_match_score: Minimum match score to accept a result (0.0-1.0)

        Returns:
            LinkedInProfileResult if found, None otherwise
        """
        if not self.api_key:
            logger.error("No RapidAPI key configured. Cannot search.")
            return None

        # Check cache first
        if use_cache:
            cached = self._check_from_cache(name, company)
            if cached:
                return cached

        try:
            self._rate_limit()
            self._api_calls += 1

            # Use the Google Full Profiles endpoint from Fresh LinkedIn Profile Data API
            # Endpoint: POST /google-full-profiles
            # Note: This endpoint uses a different host than the base API
            url = "https://web-scraping-api2.p.rapidapi.com/google-full-profiles"

            # Extract job title guess from company context
            # The API requires job_title, so we'll use a generic one if not known
            job_title = ""  # Will rely on name + company matching

            payload = {
                "name": name,
                "job_title": job_title,  # Optional, helps narrow search
                "company_name": company,
                "location": "",  # Optional
                "keywords": "",  # Optional
                "limit": 5,  # Get top 5 matches
            }

            # Update headers for this specific endpoint
            headers = {
                "Content-Type": "application/json",
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": "web-scraping-api2.p.rapidapi.com",
            }

            logger.debug(f"Searching LinkedIn for: {name} at {company}")

            response = self._session.post(
                url, json=payload, headers=headers, timeout=30
            )

            if response.status_code == 403:
                # Not subscribed - don't retry, just log once and return
                error_msg = response.text[:200]
                if "not subscribed" in error_msg.lower():
                    logger.warning(
                        "RapidAPI: Not subscribed to Fresh LinkedIn Profile Data API. "
                        "Subscribe at: https://rapidapi.com/freshdata-freshdata-default/api/fresh-linkedin-profile-data/pricing"
                    )
                else:
                    logger.warning(f"RapidAPI access denied: {error_msg}")
                return None

            if response.status_code == 429:
                logger.warning("Rate limited by RapidAPI. Waiting before retry...")
                api_client.linkedin_limiter.on_429_error(
                    endpoint="rapidapi", retry_after=5.0
                )
                response = self._session.post(
                    url, json=payload, headers=headers, timeout=30
                )

            if response.status_code != 200:
                logger.error(
                    f"API error: {response.status_code} - {response.text[:200]}"
                )
                return None

            # Report success to rate limiter
            api_client.linkedin_limiter.on_success(endpoint="rapidapi")

            data = response.json()

            # Fresh LinkedIn Profile Data returns: {"message": "ok", "data": [...]}
            if data.get("message") != "ok" or not data.get("data"):
                logger.debug(
                    f"No results for {name} at {company}: {data.get('message', 'unknown')}"
                )
                return None

            items = data.get("data", [])
            if not items:
                logger.debug(f"No results for {name} at {company}")
                return None

            # Find best matching result
            best_result = None
            best_score = 0.0

            for item in items:
                # Fresh LinkedIn Profile Data format - map to our expected format
                mapped_item = {
                    "first_name": item.get("first_name", ""),
                    "last_name": item.get("last_name", ""),
                    "full_name": f"{item.get('first_name', '')} {item.get('last_name', '')}".strip(),
                    "headline": item.get("headline", ""),
                    "location": item.get("location", ""),
                    "linkedin_url": item.get("linkedin_url", ""),
                    "profile_image_url": item.get("profile_image_url", ""),
                    "company": item.get("company", ""),
                    "job_title": item.get("job_title", ""),
                    "about": item.get("about", ""),
                }

                score = self._calculate_match_score(mapped_item, name, company)

                # Also consider API's match score if available
                api_score = item.get("_match_score", 0)
                if api_score:
                    score = (score + api_score / 100) / 2

                if score > best_score:
                    best_score = score
                    best_result = mapped_item

            if not best_result or best_score < min_match_score:
                logger.debug(
                    f"No good match for {name} at {company} (best score: {best_score:.2f})"
                )
                return None

            # Build result - extract public_id from URL
            linkedin_url = best_result.get("linkedin_url", "")
            public_id = ""
            if linkedin_url and "/in/" in linkedin_url:
                public_id = linkedin_url.split("/in/")[-1].rstrip("/")

            # Get full name
            full_name = best_result.get("full_name", "")
            if not full_name:
                full_name = f"{best_result.get('first_name', '')} {best_result.get('last_name', '')}".strip()

            # Extract job title from headline if not provided
            headline = best_result.get("headline", "")
            job_title = best_result.get("job_title", "")
            if not job_title and headline:
                job_title = (
                    headline.split(" at ")[0].split(" @ ")[0].split(" - ")[0].strip()
                )

            result = LinkedInProfileResult(
                linkedin_url=linkedin_url,
                public_id=public_id,
                first_name=best_result.get("first_name", ""),
                last_name=best_result.get("last_name", ""),
                full_name=full_name,
                headline=headline,
                job_title=job_title,
                company=best_result.get("company", ""),
                company_linkedin_url=best_result.get("company_linkedin_url", ""),
                location=best_result.get("location", ""),
                profile_image_url=best_result.get("profile_image_url", ""),
                about=best_result.get("about", ""),
                match_score=best_score,
                company_domain=best_result.get("company_domain", ""),
                company_industry=best_result.get("company_industry", ""),
                connection_count=best_result.get("connection_count", 0),
            )

            logger.info(
                f"Found LinkedIn profile for {name}: {result.linkedin_url} (score: {best_score:.2f})"
            )

            # Cache the result
            if use_cache:
                self._save_to_cache(name, company, result)

            return result

        except requests.exceptions.Timeout:
            logger.error(f"Timeout searching for {name} at {company}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error searching for {name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error searching for {name}: {e}")
            return None

    def search_batch(
        self,
        people: List[Dict[str, str]],
        use_cache: bool = True,
        min_match_score: float = 0.5,
        delay_between_requests: float = 0.5,
    ) -> Dict[str, Optional[LinkedInProfileResult]]:
        """
        Search for multiple people in batch.

        Args:
            people: List of dicts with 'name' and 'company' keys
            use_cache: Whether to use caching
            min_match_score: Minimum match score to accept
            delay_between_requests: Additional delay between API calls

        Returns:
            Dict mapping "name|company" to LinkedInProfileResult or None
        """
        results = {}

        for i, person in enumerate(people):
            name = person.get("name", "")
            company = person.get("company", "")

            if not name or not company:
                continue

            key = f"{name}|{company}"

            # Check cache first (no delay for cache hits)
            if use_cache:
                cached = self._check_from_cache(name, company)
                if cached:
                    results[key] = cached
                    continue

            # Add delay between API calls
            if i > 0 and delay_between_requests > 0:
                time.sleep(delay_between_requests)

            result = self.search_person(
                name, company, use_cache=use_cache, min_match_score=min_match_score
            )
            results[key] = result

        return results

    def get_profile_by_url(self, linkedin_url: str) -> Optional[LinkedInProfileResult]:
        """
        Get detailed profile information by LinkedIn URL.

        Args:
            linkedin_url: Full LinkedIn profile URL

        Returns:
            LinkedInProfileResult if found, None otherwise
        """
        if not self.api_key:
            logger.error("No RapidAPI key configured. Cannot get profile.")
            return None

        try:
            self._rate_limit()
            self._api_calls += 1

            # RockApis endpoint for profile by URL
            url = f"{self.BASE_URL}"
            params = {"url": linkedin_url}

            response = self._session.get(url, params=params, timeout=30)

            if response.status_code == 429:
                api_client.linkedin_limiter.on_429_error(
                    endpoint="rapidapi", retry_after=5.0
                )
                logger.warning("Rate limited by RapidAPI on profile lookup")
                return None

            if response.status_code != 200:
                logger.error(f"API error: {response.status_code}")
                return None

            # Report success to rate limiter
            api_client.linkedin_limiter.on_success(endpoint="rapidapi")

            data = response.json()

            # RockApis format: {"success": true, "data": {...}}
            if not data.get("success") or not data.get("data"):
                return None

            item = data["data"]

            # Extract public_id from URL
            public_id = ""
            if linkedin_url and "/in/" in linkedin_url:
                public_id = linkedin_url.split("/in/")[-1].rstrip("/")

            return LinkedInProfileResult(
                linkedin_url=linkedin_url,
                public_id=public_id,
                first_name=item.get("firstName", ""),
                last_name=item.get("lastName", ""),
                full_name=f"{item.get('firstName', '')} {item.get('lastName', '')}".strip(),
                headline=item.get("headline", ""),
                job_title=item.get("headline", "").split(" at ")[0].strip()
                if item.get("headline")
                else "",
                company=item.get("company", {}).get("name", "")
                if isinstance(item.get("company"), dict)
                else "",
                company_linkedin_url="",
                location=item.get("geo", {}).get("full", "")
                if isinstance(item.get("geo"), dict)
                else "",
                profile_image_url=item.get("profilePicture", ""),
                about=item.get("summary", ""),
                match_score=1.0,  # Direct URL lookup
                company_domain="",
                company_industry="",
                connection_count=0,
            )

        except Exception as e:
            logger.error(f"Error getting profile {linkedin_url}: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self._request_count,
            "api_calls": self._api_calls,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": round(
                self._cache_hits / max(self._request_count, 1) * 100, 1
            ),
        }

    def test_connection(self) -> bool:
        """
        Test the API connection with a simple search.

        Returns:
            True if API is working, False otherwise
        """
        if not self.api_key:
            logger.error("No RapidAPI key configured")
            return False

        try:
            # Test with a known search
            result = self.search_person("Satya Nadella", "Microsoft", use_cache=False)

            if result and result.linkedin_url:
                logger.info(f"API test successful: Found {result.full_name}")
                return True
            else:
                logger.warning("API test: No results returned")
                return False

        except Exception as e:
            logger.error(f"API test failed: {e}")
            return False


# NOTE: HybridLinkedInLookup has been consolidated into linkedin_voyager_client.py
# The class there orchestrates RapidAPI, Voyager API, and browser fallback in priority order.
# Import from linkedin_voyager_client instead:
#   from linkedin_voyager_client import HybridLinkedInLookup


def extract_public_id(linkedin_url: str) -> Optional[str]:
    """
    Extract the public ID from a LinkedIn URL.

    Examples:
        https://www.linkedin.com/in/john-smith-123 -> john-smith-123
        https://linkedin.com/in/jane-doe -> jane-doe
    """
    if not linkedin_url:
        return None

    import re

    match = re.search(r"linkedin\.com/in/([^/?]+)", linkedin_url)
    if match:
        return match.group(1)

    return None


def format_linkedin_mention(profile: Dict[str, Any]) -> str:
    """
    Format a LinkedIn profile for @mention in content.

    Returns a Markdown-style mention like:
    [John Smith](https://linkedin.com/in/john-smith)
    """
    name = profile.get("name", profile.get("full_name", "Unknown"))
    url = profile.get("linkedin_url", "")

    if url:
        return f"[{name}]({url})"
    return name
