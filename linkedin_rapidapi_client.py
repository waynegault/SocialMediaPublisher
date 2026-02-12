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
import time
from typing import Optional, List, Dict, Any

import requests

from api_client import api_client
from config import Config
from models import LinkedInProfile
from url_utils import extract_linkedin_public_id

logger = logging.getLogger(__name__)


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

    def _check_from_cache(self, name: str, company: str) -> Optional[LinkedInProfile]:
        """Check if result exists in cache."""
        if not self.cache:
            return None

        cache_key = self._get_cache_key(name, company)
        cached = self.cache.get(cache_key)

        if cached:
            self._cache_hits += 1
            logger.debug(f"Cache hit for {name} at {company}")
            return LinkedInProfile.from_dict(cached)

        return None

    def _save_to_cache(
        self,
        name: str,
        company: str,
        result: LinkedInProfile,
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

        Uses centralized scoring from profile_matcher module.
        """
        # Extract name from result (support both formats)
        result_name = result.get("full_name", "").strip()
        if not result_name:
            result_name = (
                f"{result.get('first_name', '')} {result.get('last_name', '')}".strip()
            )

        # Use centralized scoring function
        from profile_matcher import calculate_match_score

        return calculate_match_score(
            target_name=target_name,
            target_company=target_company,
            result_name=result_name,
            result_company=result.get("company", ""),
            result_headline=result.get("headline", ""),
            result_job_title=result.get("job_title", ""),
        )

    def search_person(
        self,
        name: str,
        company: str,
        use_cache: bool = True,
        min_match_score: float = 0.5,
    ) -> Optional[LinkedInProfile]:
        """
        Search for a person by name and company.

        Args:
            name: Person's full name (e.g., "John Smith")
            company: Company name (e.g., "Microsoft")
            use_cache: Whether to check/use cache
            min_match_score: Minimum match score to accept a result (0.0-1.0)

        Returns:
            LinkedInProfile if found, None otherwise
        """
        # Check master switch - LinkedIn searching is disabled
        if not Config.LINKEDIN_SEARCH_ENABLED:
            logger.info("LinkedIn search disabled (LINKEDIN_SEARCH_ENABLED=false)")
            return None

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
            public_id = extract_linkedin_public_id(linkedin_url) or ""

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

            result = LinkedInProfile(
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
    ) -> Dict[str, Optional[LinkedInProfile]]:
        """
        Search for multiple people in batch.

        Args:
            people: List of dicts with 'name' and 'company' keys
            use_cache: Whether to use caching
            min_match_score: Minimum match score to accept
            delay_between_requests: Additional delay between API calls

        Returns:
            Dict mapping "name|company" to LinkedInProfile or None
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

    def get_profile_by_url(self, linkedin_url: str) -> Optional[LinkedInProfile]:
        """
        Get detailed profile information by LinkedIn URL.

        Args:
            linkedin_url: Full LinkedIn profile URL

        Returns:
            LinkedInProfile if found, None otherwise
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
            public_id = extract_linkedin_public_id(linkedin_url) or ""

            return LinkedInProfile(
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


# ============================================================================
# Module Tests
# ============================================================================


def _create_module_tests():
    """Create and run tests for linkedin_rapidapi_client module."""
    from test_framework import TestSuite

    suite = TestSuite("LinkedIn RapidAPI Client", "linkedin_rapidapi_client.py")

    def _make_client():
        """Create a FreshLinkedInAPIClient with no API key."""
        return FreshLinkedInAPIClient(api_key="test-key-not-real")

    def test_client_init():
        """Test client initialization with explicit key."""
        client = FreshLinkedInAPIClient(api_key="my-test-key")
        assert client.api_key == "my-test-key"
        assert client._request_count == 0
        assert client._cache_hits == 0
        assert client._api_calls == 0

    def test_client_base_url():
        """Test client has correct API endpoint."""
        assert "rapidapi.com" in FreshLinkedInAPIClient.BASE_URL
        assert "rapidapi.com" in FreshLinkedInAPIClient.API_HOST

    def test_get_cache_key_deterministic():
        """Test _get_cache_key produces consistent keys."""
        client = _make_client()
        key1 = client._get_cache_key("John Smith", "Google")
        key2 = client._get_cache_key("John Smith", "Google")
        assert key1 == key2
        assert key1.startswith("linkedin_profile:")

    def test_get_cache_key_case_insensitive():
        """Test _get_cache_key normalizes case."""
        client = _make_client()
        key1 = client._get_cache_key("John Smith", "Google")
        key2 = client._get_cache_key("JOHN SMITH", "GOOGLE")
        assert key1 == key2

    def test_get_cache_key_trims_whitespace():
        """Test _get_cache_key trims whitespace."""
        client = _make_client()
        key1 = client._get_cache_key("John Smith", "Google")
        key2 = client._get_cache_key("  John Smith  ", "  Google  ")
        assert key1 == key2

    def test_get_cache_key_different_inputs():
        """Test _get_cache_key produces different keys for different inputs."""
        client = _make_client()
        key1 = client._get_cache_key("Alice", "Acme")
        key2 = client._get_cache_key("Bob", "Acme")
        assert key1 != key2

    def test_extract_company_from_headline_target():
        """Test _extract_company_from_headline finds target company."""
        client = _make_client()
        result = client._extract_company_from_headline("CTO at Google", "Google")
        assert result == "Google"

    def test_extract_company_from_headline_at_pattern():
        """Test _extract_company_from_headline extracts after 'at'."""
        client = _make_client()
        result = client._extract_company_from_headline("Engineer at Microsoft Corp", "Acme")
        assert "Microsoft" in result

    def test_extract_company_from_headline_empty():
        """Test _extract_company_from_headline handles empty input."""
        client = _make_client()
        result = client._extract_company_from_headline("", "Google")
        assert result == ""

    def test_get_stats_initial():
        """Test get_stats returns zeroed initial stats."""
        client = _make_client()
        stats = client.get_stats()
        assert stats["total_requests"] == 0
        assert stats["api_calls"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_hit_rate"] == 0.0

    def test_check_from_cache_no_cache():
        """Test _check_from_cache returns None when no cache."""
        client = FreshLinkedInAPIClient(api_key="test-key-not-real")
        client.cache = None
        result = client._check_from_cache("John", "Google")
        assert result is None

    suite.run_test(
        test_name="Client initialization",
        test_func=test_client_init,
        test_summary="Tests client stores key and zeroes counters",
        method_description="Creates client with explicit API key",
        expected_outcome="Key stored, counters at zero",
    )
    suite.run_test(
        test_name="Client base URL",
        test_func=test_client_base_url,
        test_summary="Tests API endpoint constants",
        method_description="Checks BASE_URL and API_HOST",
        expected_outcome="URLs reference rapidapi.com",
    )
    suite.run_test(
        test_name="Cache key deterministic",
        test_func=test_get_cache_key_deterministic,
        test_summary="Tests _get_cache_key consistency",
        method_description="Generates same key twice for same input",
        expected_outcome="Keys match and have correct prefix",
    )
    suite.run_test(
        test_name="Cache key case insensitive",
        test_func=test_get_cache_key_case_insensitive,
        test_summary="Tests _get_cache_key case normalization",
        method_description="Compares keys for differently-cased inputs",
        expected_outcome="Keys match regardless of case",
    )
    suite.run_test(
        test_name="Cache key trims whitespace",
        test_func=test_get_cache_key_trims_whitespace,
        test_summary="Tests _get_cache_key whitespace handling",
        method_description="Compares keys for inputs with extra spaces",
        expected_outcome="Keys match regardless of whitespace",
    )
    suite.run_test(
        test_name="Cache key different inputs",
        test_func=test_get_cache_key_different_inputs,
        test_summary="Tests _get_cache_key uniqueness",
        method_description="Generates keys for different people",
        expected_outcome="Keys differ for different inputs",
    )
    suite.run_test(
        test_name="Extract company - target match",
        test_func=test_extract_company_from_headline_target,
        test_summary="Tests headline extraction when target present",
        method_description="Passes headline containing target company",
        expected_outcome="Returns target company name",
    )
    suite.run_test(
        test_name="Extract company - at pattern",
        test_func=test_extract_company_from_headline_at_pattern,
        test_summary="Tests headline extraction with 'at' pattern",
        method_description="Passes headline with 'at Company' pattern",
        expected_outcome="Extracts company after 'at'",
    )
    suite.run_test(
        test_name="Extract company - empty",
        test_func=test_extract_company_from_headline_empty,
        test_summary="Tests headline extraction with empty input",
        method_description="Passes empty headline string",
        expected_outcome="Returns empty string",
    )
    suite.run_test(
        test_name="Get stats initial",
        test_func=test_get_stats_initial,
        test_summary="Tests get_stats returns zeroed stats",
        method_description="Gets stats from fresh client",
        expected_outcome="All counters at zero",
    )
    suite.run_test(
        test_name="Check from cache - no cache",
        test_func=test_check_from_cache_no_cache,
        test_summary="Tests cache check when no cache configured",
        method_description="Calls _check_from_cache with no cache",
        expected_outcome="Returns None",
    )

    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)

if __name__ == "__main__":
    run_comprehensive_tests()
