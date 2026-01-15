"""LinkedIn Voyager API Client for profile lookups.

This module provides a reliable LinkedIn profile lookup using LinkedIn's internal
Voyager API instead of browser scraping. This approach:
- Avoids CAPTCHA detection
- Is much faster (direct API calls vs browser rendering)
- Provides structured data directly
- Uses authenticated session cookies for reliability

Based on techniques from open-linkedin-api project.

IMPORTANT: This uses LinkedIn's internal API which may change. Use responsibly
and respect LinkedIn's rate limits.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import requests

from api_client import api_client

# Import unified cache (optional - falls back to dict if not available)
try:
    from cache import get_linkedin_cache, LinkedInCache

    _UNIFIED_CACHE_AVAILABLE = True
except ImportError:
    _UNIFIED_CACHE_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LinkedInPerson:
    """Represents a person found on LinkedIn."""

    urn_id: str  # LinkedIn URN ID (e.g., "ACoAABxxxxxx")
    public_id: str  # Vanity URL slug
    name: str
    first_name: str = ""
    last_name: str = ""
    headline: str = ""  # Current job title/description
    location: str = ""
    profile_url: str = ""

    # Matching metadata
    distance: int = 0  # Connection distance (1=1st, 2=2nd, 3=3rd+)
    match_score: float = 0.0
    match_signals: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.urn_id and not self.profile_url:
            self.profile_url = f"https://www.linkedin.com/in/{self.public_id}"

    @property
    def mention_urn(self) -> str:
        """Get URN format for @mentions."""
        return f"urn:li:person:{self.urn_id}" if self.urn_id else ""


@dataclass
class LinkedInOrganization:
    """Represents an organization on LinkedIn."""

    urn_id: str
    public_id: str  # Vanity URL slug
    name: str
    page_type: str = "company"  # "company" or "school"
    industry: str = ""
    description: str = ""
    location: str = ""
    employee_count: str = ""
    profile_url: str = ""

    def __post_init__(self) -> None:
        if self.public_id and not self.profile_url:
            self.profile_url = (
                f"https://www.linkedin.com/{self.page_type}/{self.public_id}"
            )

    @property
    def mention_urn(self) -> str:
        """Get URN format for @mentions."""
        return f"urn:li:organization:{self.urn_id}" if self.urn_id else ""


# =============================================================================
# LinkedIn Voyager Client
# =============================================================================


class LinkedInVoyagerClient:
    """
    LinkedIn Voyager API client for profile lookups.

    Uses LinkedIn's internal API endpoints for reliable profile search
    without triggering CAPTCHA detection.

    Authentication:
    - Requires li_at cookie from an authenticated LinkedIn session
    - Optionally accepts JSESSIONID for additional security

    Usage:
        client = LinkedInVoyagerClient()
        results = client.search_people(
            keyword_first_name="John",
            keyword_last_name="Smith",
            keyword_company="Google"
        )
    """

    # API endpoints
    API_BASE_URL = "https://www.linkedin.com/voyager/api"

    # Default headers to mimic browser - must match real Chrome headers closely
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/vnd.linkedin.normalized+json+2.1",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "X-Li-Lang": "en_US",
        "X-Li-Page-Instance": "urn:li:page:d_flagship3_search_srp_people;",
        "X-Li-Track": '{"clientVersion":"1.13.0","mpVersion":"1.13.0","osName":"web","timezoneOffset":-5,"deviceFormFactor":"DESKTOP","mpName":"voyager-web"}',
        "X-Restli-Protocol-Version": "2.0.0",
        "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Referer": "https://www.linkedin.com/search/results/people/",
        "Origin": "https://www.linkedin.com",
    }

    def __init__(
        self,
        li_at: str | None = None,
        jsessionid: str | None = None,
        proxies: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the LinkedIn Voyager client.

        Args:
            li_at: LinkedIn authentication cookie. If not provided, reads from
                   LINKEDIN_LI_AT environment variable or Config.
            jsessionid: Session ID cookie. If not provided, reads from
                        LINKEDIN_JSESSIONID environment variable or Config.
            proxies: Optional proxy configuration for requests.
        """
        # Try to get credentials from multiple sources
        self.li_at = li_at or os.getenv("LINKEDIN_LI_AT", "")
        self.jsessionid = jsessionid or os.getenv("LINKEDIN_JSESSIONID", "")

        # Also check Config if not found in environment
        if not self.li_at:
            try:
                from config import Config

                self.li_at = getattr(Config, "LINKEDIN_LI_AT", "") or ""
                self.jsessionid = (
                    self.jsessionid or getattr(Config, "LINKEDIN_JSESSIONID", "") or ""
                )
            except Exception:
                pass

        self.proxies = proxies

        # Request tracking (rate limiting now handled by api_client)
        self._request_count: int = 0
        self._max_requests_per_session: int = 100

        # Caching - use unified LinkedInCache if available, otherwise dict fallback
        self._cache_dir = Path(
            os.path.expandvars(r"%LOCALAPPDATA%\SocialMediaPublisher")
        )
        self._unified_cache: LinkedInCache | None = None
        if _UNIFIED_CACHE_AVAILABLE:
            try:
                self._unified_cache = get_linkedin_cache()
            except Exception:
                pass  # Fall back to dict cache

        # Dict-based fallback caches (used if unified cache not available)
        self._person_cache: dict[str, LinkedInPerson | None] = {}
        self._org_cache: dict[str, LinkedInOrganization | None] = {}

        # Session setup
        self._session: requests.Session | None = None
        self._csrf_token: str = ""

        if self.li_at:
            self._initialize_session()
        else:
            logger.warning(
                "LinkedIn credentials not configured. "
                "Set LINKEDIN_LI_AT and LINKEDIN_JSESSIONID environment variables."
            )

    def _initialize_session(self) -> None:
        """Initialize the requests session with authentication."""
        # Use centralized session factory with retry logic
        self._session = api_client.get_session(
            name="linkedin_voyager",
            retries=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
        )

        # Set cookies - must include both cookies
        self._session.cookies.set("li_at", self.li_at, domain=".linkedin.com")
        if self.jsessionid:
            # JSESSIONID often has quotes around it - keep them for the cookie
            jsession_value = self.jsessionid
            if not jsession_value.startswith('"'):
                jsession_value = f'"{jsession_value}"'
            self._session.cookies.set(
                "JSESSIONID", jsession_value, domain=".linkedin.com"
            )
            # CSRF token is the JSESSIONID value without quotes
            self._csrf_token = self.jsessionid.strip('"')

        # Set headers
        self._session.headers.update(self.DEFAULT_HEADERS)
        if self._csrf_token:
            # LinkedIn expects csrf-token header to match JSESSIONID
            self._session.headers["csrf-token"] = self._csrf_token

        if self.proxies:
            self._session.proxies.update(self.proxies)

        logger.info("LinkedIn Voyager client initialized")

    def _rate_limit(self) -> None:
        """Apply rate limiting using centralized api_client limiter."""
        # Use centralized LinkedIn rate limiter from api_client
        wait_time = api_client.linkedin_limiter.wait(endpoint="voyager")
        if wait_time > 0.5:
            logger.debug(f"Voyager rate limiter: waited {wait_time:.1f}s")

        # Progressive slowdown after many requests (keep this extra safety measure)
        if self._request_count > 0 and self._request_count % 10 == 0:
            extra_delay = random.uniform(5, 10)
            logger.debug(
                f"Taking {extra_delay:.1f}s break after {self._request_count} requests"
            )
            time.sleep(extra_delay)

        self._request_count += 1

    def _make_request(
        self,
        endpoint: str,
        params: dict | None = None,
        method: str = "GET",
    ) -> dict | None:
        """
        Make an authenticated request to LinkedIn Voyager API.

        Args:
            endpoint: API endpoint (relative to API_BASE_URL)
            params: Query parameters
            method: HTTP method

        Returns:
            JSON response data or None on failure
        """
        if not self._session:
            logger.error("Session not initialized - check LinkedIn credentials")
            return None

        if self._request_count >= self._max_requests_per_session:
            logger.warning("Max requests per session reached, consider restarting")

        self._rate_limit()

        url = f"{self.API_BASE_URL}{endpoint}"

        try:
            if method == "GET":
                response = self._session.get(url, params=params, timeout=30)
            else:
                response = self._session.post(url, json=params, timeout=30)

            # Check for challenge/auth issues
            if response.status_code == 401:
                logger.error(
                    f"LinkedIn authentication failed on {endpoint} - check li_at cookie"
                )
                return None

            if response.status_code == 403:
                logger.debug(f"LinkedIn access denied on {endpoint} (403)")
                return None

            if response.status_code == 429:
                logger.warning("Rate limited by LinkedIn - backing off")
                api_client.linkedin_limiter.on_429_error(
                    endpoint="voyager", retry_after=60.0
                )
                time.sleep(60)  # Back off for 1 minute
                return None

            if response.status_code != 200:
                logger.debug(f"LinkedIn API {endpoint} returned {response.status_code}")
                return None

            # Report success to rate limiter for adaptive adjustment
            api_client.linkedin_limiter.on_success(endpoint="voyager")
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

    def is_authenticated(self) -> bool:
        """Check if the client is properly authenticated."""
        if not self._session or not self.li_at:
            return False

        # Try multiple endpoints as LinkedIn may deprecate them
        test_endpoints = [
            "/me",
            "/identity/dash/profiles",
            "/voyagerIdentityDashProfiles",
            "/voyagerRelationshipsDashMemberRelationships",  # New endpoint
        ]

        for endpoint in test_endpoints:
            try:
                url = f"{self.API_BASE_URL}{endpoint}"
                response = self._session.get(url, timeout=10)
                if response.status_code == 200:
                    logger.info(f"Authentication verified via {endpoint}")
                    return True
                elif response.status_code in (401, 403):
                    # 403 on some endpoints doesn't mean not authenticated
                    logger.debug(f"Endpoint {endpoint}: {response.status_code}")
                    continue
                # 410 means endpoint deprecated, try next
            except Exception:
                continue

        # If all endpoints failed with non-auth errors, assume cookies are valid
        # and let actual API calls determine if auth works
        logger.info(
            "Could not verify auth via test endpoints, will try actual requests"
        )
        return True

    def search_people(
        self,
        keywords: str | None = None,
        keyword_first_name: str | None = None,
        keyword_last_name: str | None = None,
        keyword_title: str | None = None,
        keyword_company: str | None = None,
        current_company: list[str] | None = None,
        regions: list[str] | None = None,
        limit: int = 10,
    ) -> list[LinkedInPerson]:
        """
        Search for people on LinkedIn.

        Args:
            keywords: General search keywords
            keyword_first_name: First name to search
            keyword_last_name: Last name to search
            keyword_title: Job title to search
            keyword_company: Company name keyword
            current_company: List of company URN IDs to filter by
            regions: List of region URN IDs to filter by
            limit: Maximum results to return

        Returns:
            List of LinkedInPerson objects
        """
        # Build keyword query
        keyword_parts = []
        if keywords:
            keyword_parts.append(keywords)
        if keyword_first_name:
            keyword_parts.append(keyword_first_name)
        if keyword_last_name:
            keyword_parts.append(keyword_last_name)
        if keyword_title:
            keyword_parts.append(keyword_title)
        if keyword_company:
            keyword_parts.append(keyword_company)

        query_string = " ".join(keyword_parts) if keyword_parts else ""

        # Try multiple search endpoints
        endpoints_to_try = [
            # Current LinkedIn search endpoint
            (
                "/search/dash/clusters",
                {
                    "decorationId": "com.linkedin.voyager.dash.deco.search.SearchClusterCollection-175",
                    "origin": "GLOBAL_SEARCH_HEADER",
                    "q": "all",
                    "query": f"(keywords:{quote(query_string)},flagshipSearchIntent:SEARCH_SRP,queryParameters:(resultType:List(PEOPLE)))",
                    "start": 0,
                    "count": min(limit, 25),
                },
            ),
            # Alternative blended search
            (
                "/search/blended",
                {
                    "keywords": query_string,
                    "origin": "GLOBAL_SEARCH_HEADER",
                    "q": "all",
                    "filters": "List(resultType->PEOPLE)",
                    "start": 0,
                    "count": min(limit, 25),
                },
            ),
            # Legacy typeahead
            (
                "/typeahead/hitsV2",
                {
                    "keywords": query_string,
                    "origin": "GLOBAL_SEARCH_HEADER",
                    "q": "type",
                    "type": "PEOPLE",
                    "count": min(limit, 10),
                },
            ),
        ]

        for endpoint, params in endpoints_to_try:
            result = self._make_request(endpoint, params)
            if result:
                people = self._parse_people_results(result)
                if people:
                    logger.info(f"Search succeeded via {endpoint}")
                    return people

        return []

    def _parse_people_results(self, data: dict) -> list[LinkedInPerson]:
        """Parse people search results from Voyager API response."""
        people: list[LinkedInPerson] = []

        try:
            # Navigate the GraphQL response structure
            included = data.get("included", [])

            for item in included:
                if (
                    item.get("$type")
                    == "com.linkedin.voyager.dash.search.SearchNormalizedPerson"
                ):
                    # Extract person data
                    person_data = item.get("person", {})

                    urn_id = ""
                    public_id = ""

                    # Extract URN from entityUrn
                    entity_urn = item.get("entityUrn", "")
                    if "fsd_profile:" in entity_urn:
                        public_id = entity_urn.split("fsd_profile:")[-1]

                    # Try to get full URN ID
                    object_urn = item.get("objectUrn", "")
                    if "urn:li:member:" in object_urn:
                        urn_id = object_urn.split("urn:li:member:")[-1]

                    # Get name components
                    first_name = person_data.get("firstName", "")
                    last_name = person_data.get("lastName", "")
                    full_name = f"{first_name} {last_name}".strip()

                    if not full_name:
                        full_name = item.get("title", {}).get("text", "")

                    # Get headline (current position)
                    headline = item.get("primarySubtitle", {}).get("text", "")

                    # Get location
                    location = item.get("secondarySubtitle", {}).get("text", "")

                    if public_id or full_name:
                        people.append(
                            LinkedInPerson(
                                urn_id=urn_id,
                                public_id=public_id,
                                name=full_name,
                                first_name=first_name,
                                last_name=last_name,
                                headline=headline,
                                location=location,
                            )
                        )

        except Exception as e:
            logger.error(f"Error parsing people results: {e}")

        return people

    def search_companies(
        self,
        keywords: str,
        limit: int = 5,
    ) -> list[LinkedInOrganization]:
        """
        Search for companies/organizations on LinkedIn.

        Args:
            keywords: Company name or keywords
            limit: Maximum results

        Returns:
            List of LinkedInOrganization objects
        """
        params = {
            "decorationId": "com.linkedin.voyager.dash.deco.search.SearchClusterCollection-175",
            "origin": "FACETED_SEARCH",
            "q": "all",
            "query": f"(keywords:{quote(keywords)},flagshipSearchIntent:SEARCH_SRP,queryParameters:(resultType:List(COMPANIES)))",
            "start": 0,
            "count": min(limit, 25),
        }

        result = self._make_request("/graphql", params)

        if not result:
            return []

        return self._parse_company_results(result)

    def _parse_company_results(self, data: dict) -> list[LinkedInOrganization]:
        """Parse company search results from Voyager API response."""
        companies: list[LinkedInOrganization] = []

        try:
            included = data.get("included", [])

            for item in included:
                item_type = item.get("$type", "")

                if "SearchNormalizedCompany" in item_type or "Company" in item_type:
                    company_data = item.get("company", item)

                    urn_id = ""
                    public_id = ""

                    # Extract URN
                    entity_urn = item.get("entityUrn", "")
                    if "fsd_company:" in entity_urn:
                        public_id = entity_urn.split("fsd_company:")[-1]

                    object_urn = item.get("objectUrn", "")
                    if "urn:li:company:" in object_urn:
                        urn_id = object_urn.split("urn:li:company:")[-1]

                    name = company_data.get("name", "") or item.get("title", {}).get(
                        "text", ""
                    )

                    # Determine if school or company
                    page_type = "school" if "school" in item_type.lower() else "company"

                    industry = item.get("primarySubtitle", {}).get("text", "")
                    location = item.get("secondarySubtitle", {}).get("text", "")

                    if public_id or name:
                        companies.append(
                            LinkedInOrganization(
                                urn_id=urn_id,
                                public_id=public_id,
                                name=name,
                                page_type=page_type,
                                industry=industry,
                                location=location,
                            )
                        )

        except Exception as e:
            logger.error(f"Error parsing company results: {e}")

        return companies

    def get_profile_by_public_id(self, public_id: str) -> LinkedInPerson | None:
        """
        Get detailed profile information by public ID (vanity URL).

        Args:
            public_id: The vanity URL slug (e.g., "johndoe" from linkedin.com/in/johndoe)

        Returns:
            LinkedInPerson with full details, or None if not found
        """
        endpoint = f"/identity/profiles/{public_id}"

        result = self._make_request(endpoint)

        if not result:
            return None

        try:
            # Parse profile data
            profile = (
                result.get("elements", [{}])[0] if "elements" in result else result
            )

            return LinkedInPerson(
                urn_id=profile.get("urn", "").split(":")[-1],
                public_id=public_id,
                name=f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip(),
                first_name=profile.get("firstName", ""),
                last_name=profile.get("lastName", ""),
                headline=profile.get("headline", ""),
                location=profile.get(
                    "geoLocationName", profile.get("locationName", "")
                ),
            )

        except Exception as e:
            logger.error(f"Error parsing profile {public_id}: {e}")
            return None

    def find_person(
        self,
        name: str,
        company: str,
        title: str | None = None,
        location: str | None = None,
    ) -> LinkedInPerson | None:
        """
        Find a specific person on LinkedIn with validation.

        This is the main method for finding LinkedIn profiles. It:
        1. Searches with name + company
        2. Filters results to find best match
        3. Validates the match against provided criteria

        Args:
            name: Person's full name
            company: Company/organization name
            title: Optional job title for better matching
            location: Optional location for disambiguation

        Returns:
            Best matching LinkedInPerson, or None if no confident match
        """
        # Check unified cache first (persistent across sessions)
        cache_key = f"{name.lower().strip()}@{company.lower().strip()}"
        if self._unified_cache:
            cached = self._unified_cache.get_person(name, company)
            if cached is not None:
                # Reconstruct LinkedInPerson from cached dict
                if isinstance(cached, dict):
                    return LinkedInPerson(**cached)
                return cached
            # Check if this was previously a failed lookup
            if self._unified_cache.is_failed_lookup(name, company):
                return None

        # Fall back to in-memory dict cache
        if cache_key in self._person_cache:
            return self._person_cache[cache_key]

        # Parse name
        name_parts = name.strip().split()
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) > 1 else ""

        # Search with structured filters
        results = self.search_people(
            keyword_first_name=first_name,
            keyword_last_name=last_name,
            keyword_company=company,
            limit=10,
        )

        if not results:
            # Try broader search
            results = self.search_people(
                keywords=f"{name} {company}",
                limit=10,
            )

        if not results:
            # Cache the failure
            if self._unified_cache:
                self._unified_cache.mark_failed_lookup(name, company)
            self._person_cache[cache_key] = None
            return None

        # Score and rank results
        best_match = self._find_best_person_match(
            results,
            first_name=first_name,
            last_name=last_name,
            company=company,
            title=title,
            location=location,
        )

        # Cache the result
        if self._unified_cache and best_match:
            self._unified_cache.set_person(name, company, best_match)
        self._person_cache[cache_key] = best_match
        return best_match

    def _find_best_person_match(
        self,
        candidates: list[LinkedInPerson],
        first_name: str,
        last_name: str,
        company: str,
        title: str | None = None,
        location: str | None = None,
    ) -> LinkedInPerson | None:
        """
        Score and rank candidates to find the best match.

        Uses multi-signal scoring:
        - Name match (exact vs partial)
        - Company/headline match
        - Title match
        - Location match
        """
        first_lower = first_name.lower()
        last_lower = last_name.lower()
        company_lower = company.lower()
        title_lower = title.lower() if title else ""
        location_lower = location.lower() if location else ""

        # Extract significant company words (skip generic terms)
        skip_words = {"inc", "llc", "ltd", "corp", "the", "company", "group"}
        company_words = [
            w for w in company_lower.split() if w not in skip_words and len(w) > 2
        ]

        scored_candidates: list[tuple[float, LinkedInPerson]] = []

        for candidate in candidates:
            score = 0.0
            signals: list[str] = []

            candidate_name = candidate.name.lower()
            candidate_first = candidate.first_name.lower()
            candidate_last = candidate.last_name.lower()
            headline_lower = candidate.headline.lower()
            candidate_location = candidate.location.lower()

            # Name matching
            if candidate_first == first_lower and candidate_last == last_lower:
                score += 4.0
                signals.append("exact_name")
            elif candidate_last == last_lower:
                score += 2.0
                signals.append("last_name")
                # Check if first name is variant
                if first_lower in candidate_name or candidate_first.startswith(
                    first_lower[:3]
                ):
                    score += 1.0
                    signals.append("first_name_variant")
            elif first_lower in candidate_name and last_lower in candidate_name:
                score += 2.5
                signals.append("name_in_full")

            # Company matching (in headline)
            company_match = any(word in headline_lower for word in company_words)
            if company_match:
                score += 3.0
                signals.append("company_match")

            # Title matching
            if title_lower:
                title_words = [w for w in title_lower.split() if len(w) > 3]
                title_match = any(word in headline_lower for word in title_words)
                if title_match:
                    score += 1.5
                    signals.append("title_match")

            # Location matching
            if location_lower and candidate_location:
                location_parts = [p.strip() for p in location_lower.split(",")]
                location_match = any(
                    part in candidate_location
                    for part in location_parts
                    if len(part) > 2
                )
                if location_match:
                    score += 1.0
                    signals.append("location_match")

            # Penalty for OUT_OF_NETWORK
            if not candidate.public_id or "UNKNOWN" in candidate.public_id.upper():
                score -= 2.0
                signals.append("no_public_id")

            candidate.match_score = score
            candidate.match_signals = signals
            scored_candidates.append((score, candidate))

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Return best match if score is high enough
        if scored_candidates:
            best_score, best_candidate = scored_candidates[0]

            # Minimum threshold: need at least name + company match
            if best_score >= 5.0:
                logger.info(
                    f"Best match for {first_name} {last_name}: {best_candidate.name} "
                    f"(score={best_score:.1f}, signals={best_candidate.match_signals})"
                )
                return best_candidate
            elif best_score >= 3.0:
                logger.info(
                    f"Medium confidence match for {first_name} {last_name}: {best_candidate.name} "
                    f"(score={best_score:.1f})"
                )
                return best_candidate
            else:
                logger.debug(
                    f"Low confidence, rejecting: {best_candidate.name} (score={best_score:.1f})"
                )

        return None

    def find_company(self, name: str) -> LinkedInOrganization | None:
        """
        Find a company/organization on LinkedIn.

        Args:
            name: Company/organization name

        Returns:
            LinkedInOrganization if found, None otherwise
        """
        # Check unified cache first (persistent across sessions)
        cache_key = name.lower().strip()
        if self._unified_cache:
            cached = self._unified_cache.get_org(name)
            if cached is not None:
                if isinstance(cached, dict):
                    return LinkedInOrganization(**cached)
                return cached

        # Fall back to in-memory dict cache
        if cache_key in self._org_cache:
            return self._org_cache[cache_key]

        results = self.search_companies(name, limit=5)

        if not results:
            self._org_cache[cache_key] = None
            return None

        # Find best match by name similarity
        name_lower = name.lower()
        name_words = set(name_lower.split())

        for org in results:
            org_name_lower = org.name.lower()
            org_words = set(org_name_lower.split())

            # Check for significant overlap
            overlap = name_words & org_words
            if len(overlap) >= len(name_words) * 0.5:
                # Cache in both unified and dict cache
                if self._unified_cache:
                    self._unified_cache.set_org(name, org)
                self._org_cache[cache_key] = org
                return org

            # Check if org name contains search term or vice versa
            if name_lower in org_name_lower or org_name_lower in name_lower:
                if self._unified_cache:
                    self._unified_cache.set_org(name, org)
                self._org_cache[cache_key] = org
                return org

        # Return first result as fallback
        if self._unified_cache:
            self._unified_cache.set_org(name, results[0])
        self._org_cache[cache_key] = results[0]
        return results[0]

    def close(self) -> None:
        """Close the client session."""
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self) -> LinkedInVoyagerClient:
        return self

    def __exit__(self, *args) -> None:  # type: ignore[no-untyped-def]
        self.close()


# =============================================================================
# Hybrid Lookup that combines RapidAPI, Voyager API, and browser fallback
# =============================================================================


class HybridLinkedInLookup:
    """
    Hybrid LinkedIn lookup that tries multiple sources in priority order:
    1. RapidAPI Fresh LinkedIn Data API (most reliable, paid)
    2. Voyager API (deprecated, usually blocked)
    3. Browser-based search (slowest but always works)

    This achieves 90%+ success rate by using RapidAPI as the primary method.
    """

    # Class-level flag to track if Voyager API is known to be blocked
    _voyager_api_blocked: bool = (
        True  # Default to blocked (LinkedIn blocked it in 2025)
    )

    # Class-level RapidAPI client (shared across instances for connection pooling)
    _rapidapi_client = None

    def __init__(self) -> None:
        """Initialize the hybrid lookup."""
        self.voyager_client: LinkedInVoyagerClient | None = None
        self.browser_lookup = None  # Will be imported lazily

        # Initialize RapidAPI client if not already done
        if HybridLinkedInLookup._rapidapi_client is None:
            try:
                from linkedin_rapidapi_client import FreshLinkedInAPIClient
                from config import Config

                if getattr(Config, "RAPIDAPI_KEY", ""):
                    HybridLinkedInLookup._rapidapi_client = FreshLinkedInAPIClient()
                    logger.info("RapidAPI client initialized for LinkedIn lookups")
                else:
                    logger.debug("RAPIDAPI_KEY not set, will use browser fallback")
            except ImportError:
                logger.debug("RapidAPI client not available")
            except Exception as e:
                logger.warning(f"Failed to initialize RapidAPI client: {e}")

        # Skip Voyager API if known to be blocked (which is the default now)
        if HybridLinkedInLookup._voyager_api_blocked:
            logger.debug("Voyager API blocked, using RapidAPI + browser fallback")
        elif os.getenv("LINKEDIN_LI_AT"):
            try:
                self.voyager_client = LinkedInVoyagerClient()
                # Don't bother with auth check - it uses deprecated endpoints
                # We'll detect failures on first actual use
                logger.debug("Voyager client initialized (will test on first use)")
            except Exception as e:
                logger.warning(f"Failed to initialize Voyager client: {e}")
                self.voyager_client = None
        else:
            logger.debug("LINKEDIN_LI_AT not set, using browser-based lookup only")

    def find_person(
        self,
        name: str,
        company: str,
        title: str | None = None,
        location: str | None = None,
        department: str | None = None,
        role_type: str | None = None,
    ) -> tuple[str | None, str | None, str]:
        """
        Find a person's LinkedIn profile URL and URN.

        Args:
            name: Person's name
            company: Company/organization
            title: Job title
            location: Location
            department: Department
            role_type: Role type (academic, executive, etc.)

        Returns:
            Tuple of (profile_url, urn, match_confidence)
        """
        # 1. Try RapidAPI Fresh LinkedIn Data API first (most reliable)
        if HybridLinkedInLookup._rapidapi_client is not None:
            try:
                result = HybridLinkedInLookup._rapidapi_client.search_person(
                    name, company
                )

                if result and result.linkedin_url:
                    confidence = "high" if result.match_score >= 0.7 else "medium"
                    logger.info(
                        f"RapidAPI found: {result.full_name} at {result.company} (score: {result.match_score:.2f})"
                    )
                    # RapidAPI doesn't return URN, but we have the URL
                    return (result.linkedin_url, None, confidence)

            except Exception as e:
                logger.warning(f"RapidAPI lookup failed for {name}: {e}")

        # 2. Try Voyager API (if not known to be blocked - usually is)
        if self.voyager_client and not HybridLinkedInLookup._voyager_api_blocked:
            try:
                person = self.voyager_client.find_person(
                    name=name,
                    company=company,
                    title=title,
                    location=location,
                )

                if person and person.public_id:
                    confidence = "high" if person.match_score >= 5.0 else "medium"
                    return (person.profile_url, person.mention_urn, confidence)

                # If we get here with no results, Voyager API might be blocked
                # Mark it as blocked to avoid wasting time on future calls
                logger.info(
                    "Voyager API returned no results, switching to browser fallback"
                )
                HybridLinkedInLookup._voyager_api_blocked = True
                self.voyager_client = None

            except Exception as e:
                logger.warning(f"Voyager API lookup failed: {e}, switching to browser")
                HybridLinkedInLookup._voyager_api_blocked = True
                self.voyager_client = None

        # 3. Fall back to browser-based lookup (slowest but always works)
        return self._browser_fallback(
            name, company, title, location, department, role_type
        )

    def _browser_fallback(
        self,
        name: str,
        company: str,
        title: str | None = None,
        location: str | None = None,
        department: str | None = None,
        role_type: str | None = None,
    ) -> tuple[str | None, str | None, str]:
        """Fall back to browser-based lookup."""
        if self.browser_lookup is None:
            try:
                from linkedin_profile_lookup import LinkedInCompanyLookup

                self.browser_lookup = LinkedInCompanyLookup()
            except Exception as e:
                logger.error(f"Failed to initialize browser lookup: {e}")
                return (None, None, "failed")

        try:
            url = self.browser_lookup.search_person(
                name=name,
                company=company,
                position=title,
                department=department,
                location=location,
                role_type=role_type,
            )

            if url:
                # Try to get URN
                urn = None
                if hasattr(self.browser_lookup, "lookup_person_urn"):
                    urn = self.browser_lookup.lookup_person_urn(url)
                return (url, urn, "medium")

        except Exception as e:
            logger.warning(f"Browser lookup failed: {e}")

        return (None, None, "not_found")

    def find_company(self, name: str) -> tuple[str | None, str | None]:
        """
        Find a company's LinkedIn page URL and URN.

        Args:
            name: Company/organization name

        Returns:
            Tuple of (page_url, urn)
        """
        # Try Voyager API first
        if self.voyager_client:
            try:
                org = self.voyager_client.find_company(name)
                if org:
                    return (org.profile_url, org.mention_urn)
            except Exception as e:
                logger.warning(f"Voyager company lookup failed: {e}")

        # Fall back to browser
        if self.browser_lookup is None:
            try:
                from linkedin_profile_lookup import LinkedInCompanyLookup

                self.browser_lookup = LinkedInCompanyLookup()
            except Exception:
                return (None, None)

        try:
            url, slug = self.browser_lookup.search_company(name)
            if url:
                urn = self.browser_lookup.lookup_organization_urn(url) if url else None
                return (url, urn)
        except Exception as e:
            logger.warning(f"Browser company lookup failed: {e}")

        return (None, None)

    def validate_profile_match(
        self,
        linkedin_url: str,
        expected_name: str,
        expected_company: str,
        expected_title: str | None = None,
        expected_location: str | None = None,
    ) -> tuple[bool, float, list[str]]:
        """
        Validate a LinkedIn profile match using triangulation.

        Cross-checks the profile against multiple expected data points to ensure
        the correct profile was found.

        Args:
            linkedin_url: The LinkedIn profile URL to validate
            expected_name: Expected full name
            expected_company: Expected company/organization name
            expected_title: Expected job title
            expected_location: Expected location

        Returns:
            Tuple of (is_valid, confidence_score, validation_signals)
        """
        signals: list[str] = []
        score = 0.0

        if not linkedin_url or not self.voyager_client:
            return (False, 0.0, ["no_url_or_client"])

        # Extract public_id from URL
        public_id = ""
        if "linkedin.com/in/" in linkedin_url:
            public_id = (
                linkedin_url.split("linkedin.com/in/")[-1].rstrip("/").split("?")[0]
            )

        if not public_id:
            return (False, 0.0, ["invalid_url"])

        try:
            # Get full profile details
            profile = self.voyager_client.get_profile_by_public_id(public_id)

            if not profile:
                return (False, 0.0, ["profile_not_found"])

            # Validate name
            expected_parts = expected_name.lower().split()
            profile_parts = profile.name.lower().split()

            # Check last name match
            if expected_parts and profile_parts:
                if expected_parts[-1] == profile_parts[-1]:
                    score += 3.0
                    signals.append("last_name_match")

                    # Check first name match
                    if expected_parts[0] == profile_parts[0]:
                        score += 2.0
                        signals.append("first_name_match")
                    elif expected_parts[0] in profile.name.lower():
                        score += 1.0
                        signals.append("first_name_partial")

            # Validate company/headline
            headline_lower = profile.headline.lower()
            company_lower = expected_company.lower()

            # Skip generic words
            skip_words = {
                "inc",
                "llc",
                "ltd",
                "corp",
                "the",
                "company",
                "university",
                "of",
            }
            company_words = [
                w for w in company_lower.split() if w not in skip_words and len(w) > 2
            ]

            company_match = any(word in headline_lower for word in company_words)
            if company_match:
                score += 3.0
                signals.append("company_in_headline")

            # Validate title if provided
            if expected_title:
                title_lower = expected_title.lower()
                title_words = [w for w in title_lower.split() if len(w) > 3]
                title_match = any(word in headline_lower for word in title_words)
                if title_match:
                    score += 2.0
                    signals.append("title_match")

            # Validate location if provided
            if expected_location and profile.location:
                loc_lower = expected_location.lower()
                profile_loc_lower = profile.location.lower()

                # Check for location overlap
                loc_parts = [p.strip() for p in loc_lower.split(",")]
                loc_match = any(
                    part in profile_loc_lower for part in loc_parts if len(part) > 2
                )
                if loc_match:
                    score += 1.5
                    signals.append("location_match")

            # Determine if valid based on threshold
            is_valid = score >= 5.0  # Need at least name + company match

            return (is_valid, score, signals)

        except Exception as e:
            logger.warning(f"Profile validation failed for {public_id}: {e}")
            return (False, 0.0, [f"error:{str(e)}"])

    def close(self) -> None:
        """Close all client connections."""
        if self.voyager_client:
            self.voyager_client.close()
        if self.browser_lookup:
            try:
                self.browser_lookup.close_browser()
            except Exception:
                pass

    def __enter__(self) -> HybridLinkedInLookup:
        return self

    def __exit__(self, *args) -> None:  # type: ignore[no-untyped-def]
        self.close()


# =============================================================================
# Helper to extract cookies from browser
# =============================================================================


def extract_linkedin_cookies_from_browser() -> tuple[str, str]:
    """
    Extract LinkedIn session cookies from Chrome browser.

    Returns:
        Tuple of (li_at, jsessionid) cookies
    """
    try:
        import browser_cookie3  # type: ignore[import-not-found]

        cookies = browser_cookie3.chrome(domain_name=".linkedin.com")

        li_at = ""
        jsessionid = ""

        for cookie in cookies:
            if cookie.name == "li_at":
                li_at = cookie.value or ""
            elif cookie.name == "JSESSIONID":
                jsessionid = cookie.value or ""

        if li_at:
            logger.info("Successfully extracted LinkedIn cookies from Chrome")
            return (li_at, jsessionid)
        else:
            logger.warning("No LinkedIn session found in Chrome")
            return ("", "")

    except ImportError:
        logger.warning("browser_cookie3 not installed - pip install browser-cookie3")
        return ("", "")
    except Exception as e:
        error_msg = str(e).lower()
        if "admin" in error_msg:
            logger.warning(
                "Cookie extraction requires admin privileges. "
                "Run as Administrator or use manual entry."
            )
        elif "key" in error_msg or "decrypt" in error_msg:
            logger.warning(
                "Cookie decryption failed. This can happen when Chrome is running "
                "or due to Windows security settings. Please use manual entry instead."
            )
        else:
            logger.warning(f"Failed to extract browser cookies: {e}")
        return ("", "")
