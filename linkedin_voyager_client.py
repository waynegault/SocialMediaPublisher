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
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import requests

from api_client import api_client
from config import Config
from models import LinkedInProfile, LinkedInOrganization
from profile_matcher import score_person_candidate

# Import unified cache (optional - falls back to dict if not available)
try:
    from cache import get_linkedin_cache, LinkedInCache

    _UNIFIED_CACHE_AVAILABLE = True
except ImportError:
    _UNIFIED_CACHE_AVAILABLE = False
    get_linkedin_cache = None
    LinkedInCache = None

logger = logging.getLogger(__name__)


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

    # Minimal headers — matches the proven open-linkedin-api approach.
    # Extra browser-fingerprinting headers (Sec-*, X-Li-Track, X-Li-Page-Instance,
    # Referer, Origin) actually trigger LinkedIn's anti-bot detection for search
    # endpoints because the static values don't match a real browser's dynamic state.
    DEFAULT_HEADERS = {
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/133.0.0.0 Safari/537.36"
        ),
        "accept-language": "en-US,en;q=0.9",
        "x-li-lang": "en_US",
        "x-restli-protocol-version": "2.0.0",
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
        # First check passed parameters, then Config class
        self.li_at = li_at or Config.LINKEDIN_LI_AT
        self.jsessionid = jsessionid or Config.LINKEDIN_JSESSIONID

        self.proxies = proxies

        # Request tracking (rate limiting now handled by api_client)
        self._request_count: int = 0
        self._max_requests_per_session: int = 100

        # Caching - use unified LinkedInCache if available, otherwise dict fallback
        self._cache_dir = Path(
            os.path.expandvars(r"%LOCALAPPDATA%\SocialMediaPublisher")
        )
        self._unified_cache: Any = None
        if _UNIFIED_CACHE_AVAILABLE and get_linkedin_cache is not None:
            try:
                self._unified_cache = get_linkedin_cache()
            except Exception:
                pass  # Fall back to dict cache

        # Dict-based fallback caches (used if unified cache not available)
        self._person_cache: dict[str, LinkedInProfile | None] = {}
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

        LinkedIn's Voyager API uses Rest.li format for query parameters,
        which includes literal ``()``, ``:``, and ``,`` characters. The
        ``requests`` library percent-encodes these, causing 400 errors.

        This method builds the URL manually with ``urlencode(safe="(),:")``
        and then overrides the prepared request URL to prevent re-encoding.

        Args:
            endpoint: API endpoint (relative to API_BASE_URL)
            params: Query parameters (for GET, encoded with restli-safe chars)
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

        base_url = f"{self.API_BASE_URL}{endpoint}"

        try:
            if method == "GET":
                if params:
                    # Build URL manually to preserve restli-format characters.
                    # urlencode with safe="(),:" keeps (key:value,List(...))
                    # intact while still encoding spaces → %20, etc.
                    # quote_via=quote gives %20 for spaces (not +).
                    qs = urlencode(params, safe="(),:", quote_via=quote)
                    raw_url = f"{base_url}?{qs}"
                else:
                    raw_url = base_url

                # Use PreparedRequest so session cookies/headers are attached,
                # then override .url to prevent requests from re-encoding it.
                req = requests.Request("GET", raw_url)
                prepared = self._session.prepare_request(req)
                prepared.url = raw_url
                response = self._session.send(
                    prepared, timeout=30, allow_redirects=False
                )
            else:
                response = self._session.post(
                    base_url, json=params, timeout=30, allow_redirects=False
                )

            # 302 with cookie deletion = expired authentication
            if response.status_code in (301, 302):
                location = response.headers.get("Location", "")
                set_cookie = response.headers.get("Set-Cookie", "")
                request_url = response.request.url or ""
                if "delete me" in set_cookie or location == request_url:
                    logger.error(
                        "LinkedIn cookies expired — re-enter them via "
                        "action 7 in the menu"
                    )
                else:
                    logger.debug(
                        f"LinkedIn API {endpoint} redirected to {location[:120]}"
                    )
                return None

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

        except requests.exceptions.TooManyRedirects:
            logger.error(
                "LinkedIn redirect loop — cookies likely expired. "
                "Re-enter them via action 7 in the menu."
            )
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

    def is_authenticated(self) -> bool:
        """Check if the client is properly authenticated.

        Uses ``allow_redirects=False`` so that a 302 with cookie deletion
        is detected as expired auth instead of causing a redirect loop.
        """
        if not self._session or not self.li_at:
            return False

        # /me is the most reliable lightweight auth check
        try:
            url = f"{self.API_BASE_URL}/me"
            response = self._session.get(
                url, timeout=10, allow_redirects=False
            )
            if response.status_code == 200:
                logger.info("Authentication verified via /me")
                return True

            # 302 with "delete me" cookie = expired session
            if response.status_code in (301, 302):
                set_cookie = response.headers.get("Set-Cookie", "")
                if "delete me" in set_cookie:
                    logger.warning(
                        "LinkedIn cookies expired (server deleted li_at)"
                    )
                    return False
                logger.debug(f"/me returned {response.status_code} redirect")

            if response.status_code in (401, 403):
                logger.warning(f"/me returned {response.status_code} — not authenticated")
                return False

        except requests.exceptions.TooManyRedirects:
            logger.warning("LinkedIn redirect loop — cookies expired")
            return False
        except Exception as e:
            logger.debug(f"Auth check failed: {e}")

        # Fallback: try another endpoint
        try:
            url = f"{self.API_BASE_URL}/identity/dash/profiles"
            response = self._session.get(
                url, timeout=10, allow_redirects=False
            )
            if response.status_code == 200:
                logger.info("Authentication verified via /identity/dash/profiles")
                return True
        except Exception:
            pass

        logger.warning("Could not verify authentication — cookies may be expired")
        return False

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
    ) -> list[LinkedInProfile]:
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
            List of LinkedInProfile objects
        """
        # Check master switch - LinkedIn searching is disabled
        if not Config.LINKEDIN_SEARCH_ENABLED:
            logger.info("LinkedIn search disabled (LINKEDIN_SEARCH_ENABLED=false)")
            return []

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

        # Build filter list (matches open-linkedin-api format)
        filters = ["(key:resultType,value:List(PEOPLE))"]
        if current_company:
            stringify = ",".join(current_company)
            filters.append(f"(key:currentCompany,value:List({stringify}))")
        if regions:
            stringify = ",".join(regions)
            filters.append(f"(key:geoUrn,value:List({stringify}))")

        filter_str = f"List({','.join(filters)})"

        keywords_part = f"keywords:{query_string}," if query_string else ""

        # Try graphql first (proven approach from open-linkedin-api), then
        # dash/clusters as fallback.
        count = min(limit, 49)  # LinkedIn max is 49 per page
        endpoints_to_try = [
            # GraphQL — the approach used by open-linkedin-api
            (
                "/graphql",
                {
                    "variables": (
                        f"(start:0,origin:GLOBAL_SEARCH_HEADER,"
                        f"query:({keywords_part}"
                        f"flagshipSearchIntent:SEARCH_SRP,"
                        f"queryParameters:{filter_str},"
                        f"includeFiltersInResponse:false))"
                    ),
                    "queryId": "voyagerSearchDashClusters.b0928897b71bd00a5a7291755dcd64f0",
                },
            ),
            # Dash clusters fallback
            (
                "/search/dash/clusters",
                {
                    "decorationId": "com.linkedin.voyager.dash.deco.search.SearchClusterCollection-175",
                    "origin": "GLOBAL_SEARCH_HEADER",
                    "q": "all",
                    "query": (
                        f"({keywords_part}"
                        f"flagshipSearchIntent:SEARCH_SRP,"
                        f"queryParameters:{filter_str},"
                        f"includeFiltersInResponse:false)"
                    ),
                    "start": 0,
                    "count": count,
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

    def _parse_people_results(self, data: dict) -> list[LinkedInProfile]:
        """Parse people search results from Voyager API response.

        Handles two response formats:
        1. ``included`` array (dash/clusters endpoint) — items typed as
           ``SearchNormalizedPerson`` or ``EntityResult``.
        2. ``data.searchDashClustersByAll`` (graphql endpoint) — items
           nested under ``elements[].items[].item.entityResult``.
        """
        people: list[LinkedInProfile] = []

        try:
            # --- Format 1: included array (dash/clusters) ---
            included = data.get("included", [])

            for item in included:
                item_type = item.get("$type", "")

                # Original format: SearchNormalizedPerson
                if "SearchNormalizedPerson" in item_type:
                    person_data = item.get("person", {})

                    urn_id = ""
                    public_id = ""

                    entity_urn = item.get("entityUrn", "")
                    if "fsd_profile:" in entity_urn:
                        public_id = entity_urn.split("fsd_profile:")[-1]

                    object_urn = item.get("objectUrn", "")
                    if "urn:li:member:" in object_urn:
                        urn_id = object_urn.split("urn:li:member:")[-1]

                    first_name = person_data.get("firstName", "")
                    last_name = person_data.get("lastName", "")
                    full_name = f"{first_name} {last_name}".strip()

                    if not full_name:
                        full_name = item.get("title", {}).get("text", "")

                    headline = item.get("primarySubtitle", {}).get("text", "")
                    location = item.get("secondarySubtitle", {}).get("text", "")

                    if public_id or full_name:
                        people.append(
                            LinkedInProfile(
                                urn_id=urn_id,
                                public_id=public_id,
                                full_name=full_name,
                                first_name=first_name,
                                last_name=last_name,
                                headline=headline,
                                location=location,
                            )
                        )

                # Newer format: EntityResult in included array
                elif "EntityResult" in item_type:
                    title_obj = item.get("title", {})
                    full_name = title_obj.get("text", "") if isinstance(title_obj, dict) else ""
                    headline = ""
                    location = ""
                    sub = item.get("primarySubtitle", {})
                    if isinstance(sub, dict):
                        headline = sub.get("text", "")
                    sec_sub = item.get("secondarySubtitle", {})
                    if isinstance(sec_sub, dict):
                        location = sec_sub.get("text", "")

                    public_id = ""
                    urn_id = ""
                    nav_url = item.get("navigationUrl", "")
                    if "/in/" in nav_url:
                        public_id = nav_url.split("/in/")[-1].split("?")[0].strip("/")
                    entity_urn = item.get("entityUrn", "")
                    if "fsd_profile:" in entity_urn:
                        public_id = public_id or entity_urn.split("fsd_profile:")[-1]

                    if full_name:
                        people.append(
                            LinkedInProfile(
                                urn_id=urn_id,
                                public_id=public_id,
                                full_name=full_name,
                                first_name="",
                                last_name="",
                                headline=headline,
                                location=location,
                            )
                        )

            # --- Format 2: graphql structured data ---
            if not people:
                clusters = (
                    data.get("data", {})
                    .get("searchDashClustersByAll", {})
                )
                if clusters and isinstance(clusters, dict):
                    for element in clusters.get("elements", []):
                        for item_wrapper in element.get("items", []):
                            entity = (
                                item_wrapper.get("item", {})
                                .get("entityResult", {})
                            )
                            if not entity:
                                continue

                            title_obj = entity.get("title", {})
                            full_name = (
                                title_obj.get("text", "")
                                if isinstance(title_obj, dict) else ""
                            )
                            headline = ""
                            location = ""
                            sub = entity.get("primarySubtitle", {})
                            if isinstance(sub, dict):
                                headline = sub.get("text", "")
                            sec_sub = entity.get("secondarySubtitle", {})
                            if isinstance(sec_sub, dict):
                                location = sec_sub.get("text", "")

                            public_id = ""
                            nav_url = entity.get("navigationUrl", "")
                            if "/in/" in nav_url:
                                public_id = (
                                    nav_url.split("/in/")[-1]
                                    .split("?")[0]
                                    .strip("/")
                                )
                            entity_urn = entity.get("entityUrn", "")
                            if "fsd_profile:" in entity_urn:
                                public_id = public_id or entity_urn.split(
                                    "fsd_profile:"
                                )[-1]

                            if full_name:
                                people.append(
                                    LinkedInProfile(
                                        urn_id="",
                                        public_id=public_id,
                                        full_name=full_name,
                                        first_name="",
                                        last_name="",
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
        # Check master switch - LinkedIn searching is disabled
        if not Config.LINKEDIN_SEARCH_ENABLED:
            logger.info("LinkedIn search disabled (LINKEDIN_SEARCH_ENABLED=false)")
            return []

        filter_str = "List((key:resultType,value:List(COMPANIES)))"
        keywords_part = f"keywords:{keywords}," if keywords else ""

        # Try graphql first (proven), then dash/clusters
        endpoints_to_try = [
            (
                "/graphql",
                {
                    "variables": (
                        f"(start:0,origin:FACETED_SEARCH,"
                        f"query:({keywords_part}"
                        f"flagshipSearchIntent:SEARCH_SRP,"
                        f"queryParameters:{filter_str},"
                        f"includeFiltersInResponse:false))"
                    ),
                    "queryId": "voyagerSearchDashClusters.b0928897b71bd00a5a7291755dcd64f0",
                },
            ),
            (
                "/search/dash/clusters",
                {
                    "decorationId": "com.linkedin.voyager.dash.deco.search.SearchClusterCollection-175",
                    "origin": "FACETED_SEARCH",
                    "q": "all",
                    "query": (
                        f"({keywords_part}"
                        f"flagshipSearchIntent:SEARCH_SRP,"
                        f"queryParameters:{filter_str},"
                        f"includeFiltersInResponse:false)"
                    ),
                    "start": 0,
                    "count": min(limit, 25),
                },
            ),
        ]

        for endpoint, params in endpoints_to_try:
            result = self._make_request(endpoint, params)
            if result:
                companies = self._parse_company_results(result)
                if companies:
                    logger.info(f"Company search succeeded via {endpoint}")
                    return companies

    def _parse_company_results(self, data: dict) -> list[LinkedInOrganization]:
        """Parse company search results from Voyager API response.

        Handles two formats (same pattern as ``_parse_people_results``):
        1. ``included`` array — items typed as ``EntityResult`` or
           ``SearchNormalizedCompany``.
        2. ``data.searchDashClustersByAll`` (graphql) — nested
           ``elements[].items[].item.entityResult``.
        """
        companies: list[LinkedInOrganization] = []

        def _extract_company(entity: dict) -> LinkedInOrganization | None:
            """Extract a company from an entity-result-like dict."""
            title_obj = entity.get("title", {})
            name = title_obj.get("text", "") if isinstance(title_obj, dict) else ""
            if not name:
                name = entity.get("name", "")

            public_id = ""
            urn_id = ""

            entity_urn = entity.get("entityUrn", "")
            if "fsd_company:" in entity_urn:
                public_id = entity_urn.split("fsd_company:")[-1]

            object_urn = entity.get("objectUrn", "")
            if "urn:li:company:" in object_urn:
                urn_id = object_urn.split("urn:li:company:")[-1]

            # Try trackingUrn (open-linkedin-api style)
            tracking_urn = entity.get("trackingUrn", "")
            if not urn_id and "company:" in tracking_urn:
                urn_id = tracking_urn.split("company:")[-1]

            nav_url = entity.get("navigationUrl", "")
            if not public_id and "/company/" in nav_url:
                public_id = nav_url.split("/company/")[-1].split("?")[0].strip("/")

            item_type = entity.get("$type", "")
            page_type = "school" if "school" in item_type.lower() else "company"

            sub = entity.get("primarySubtitle", {})
            industry = sub.get("text", "") if isinstance(sub, dict) else ""
            sec_sub = entity.get("secondarySubtitle", {})
            location = sec_sub.get("text", "") if isinstance(sec_sub, dict) else ""

            if public_id or name:
                return LinkedInOrganization(
                    urn_id=urn_id,
                    public_id=public_id,
                    name=name,
                    page_type=page_type,
                    industry=industry,
                    location=location,
                )
            return None

        try:
            # --- Format 1: included array ---
            included = data.get("included", [])
            for item in included:
                item_type = item.get("$type", "")
                if any(
                    t in item_type
                    for t in ("SearchNormalizedCompany", "Company", "EntityResult")
                ):
                    # Skip people EntityResults
                    if "EntityResult" in item_type:
                        tracking = item.get("trackingUrn", "")
                        if "company" not in tracking and "fsd_company" not in item.get(
                            "entityUrn", ""
                        ):
                            continue
                    company_data = item.get("company", item)
                    org = _extract_company(company_data)
                    if org:
                        companies.append(org)

            # --- Format 2: graphql structured data ---
            if not companies:
                clusters = data.get("data", {}).get("searchDashClustersByAll", {})
                if clusters and isinstance(clusters, dict):
                    for element in clusters.get("elements", []):
                        for item_wrapper in element.get("items", []):
                            entity = (
                                item_wrapper.get("item", {}).get("entityResult", {})
                            )
                            if not entity:
                                continue
                            org = _extract_company(entity)
                            if org:
                                companies.append(org)

        except Exception as e:
            logger.error(f"Error parsing company results: {e}")

        return companies

    def get_profile_by_public_id(self, public_id: str) -> LinkedInProfile | None:
        """
        Get detailed profile information by public ID (vanity URL).

        Args:
            public_id: The vanity URL slug (e.g., "johndoe" from linkedin.com/in/johndoe)

        Returns:
            LinkedInProfile with full details, or None if not found
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

            return LinkedInProfile(
                urn_id=profile.get("urn", "").split(":")[-1],
                public_id=public_id,
                full_name=f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip(),
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
    ) -> LinkedInProfile | None:
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
            Best matching LinkedInProfile, or None if no confident match
        """
        # Check unified cache first (persistent across sessions)
        cache_key = f"{name.lower().strip()}@{company.lower().strip()}"
        if self._unified_cache:
            cached = self._unified_cache.get_person(name, company)
            if cached is not None:
                # Reconstruct LinkedInProfile from cached dict using from_dict
                if isinstance(cached, dict):
                    # Handle legacy cache entries that used 'name' instead of 'full_name'
                    if "name" in cached and "full_name" not in cached:
                        cached["full_name"] = cached.pop("name")
                    return LinkedInProfile.from_dict(cached)
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
        candidates: list[LinkedInProfile],
        first_name: str,
        last_name: str,
        company: str,
        title: str | None = None,
        location: str | None = None,
    ) -> LinkedInProfile | None:
        """
        Score and rank candidates to find the best match.

        Uses multi-signal scoring:
        - Name match (exact vs partial)
        - Company/headline match
        - Title match
        - Location match
        """
        scored_candidates: list[tuple[float, LinkedInProfile]] = []

        for candidate in candidates:
            # Use centralized scoring from profile_matcher
            result = score_person_candidate(
                candidate_first_name=candidate.first_name,
                candidate_last_name=candidate.last_name,
                candidate_headline=candidate.headline,
                candidate_location=candidate.location,
                candidate_public_id=candidate.public_id or "",
                target_first_name=first_name,
                target_last_name=last_name,
                target_company=company,
                target_title=title,
                target_location=location,
            )

            candidate.match_score = result.score
            candidate.match_signals = result.signals
            scored_candidates.append((result.score, candidate))

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
                    return LinkedInOrganization.from_dict(cached)
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
    1. Voyager API (uses your LinkedIn session cookies, fast & free)
    2. Browser-based search (Playwright/Bing, slower but always works)

    Configure Voyager cookies via Action 7 in the main menu.
    """

    # Class-level flag to track if Voyager API is known to be blocked
    _voyager_api_blocked: bool = False

    def __init__(self) -> None:
        """Initialize the hybrid lookup."""
        self.voyager_client: LinkedInVoyagerClient | None = None
        self.browser_lookup = None  # Will be imported lazily

        # Initialize Voyager client if cookies are configured
        if not HybridLinkedInLookup._voyager_api_blocked and Config.LINKEDIN_LI_AT:
            try:
                self.voyager_client = LinkedInVoyagerClient()
                logger.debug("Voyager client initialized for LinkedIn lookups")
            except Exception as e:
                logger.warning(f"Failed to initialize Voyager client: {e}")
                self.voyager_client = None
        elif not Config.LINKEDIN_LI_AT:
            logger.debug(
                "LINKEDIN_LI_AT not set — using browser-based lookup only. "
                "Use Action 7 to configure Voyager cookies for faster lookups."
            )

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
        # Check master switch - LinkedIn searching is disabled
        if not Config.LINKEDIN_SEARCH_ENABLED:
            logger.info("LinkedIn search disabled (LINKEDIN_SEARCH_ENABLED=false)")
            return (None, None, "disabled")

        # 1. Try Voyager API first (fast, free, uses your session cookies)
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

                # No results this time doesn't mean blocked
                logger.debug(
                    f"Voyager found no match for {name}, trying browser fallback"
                )

            except Exception as e:
                error_msg = str(e).lower()
                if "302" in error_msg or "redirect" in error_msg or "auth" in error_msg:
                    logger.warning(
                        f"Voyager API auth failed: {e} — cookies may be expired. "
                        f"Use Action 7 to refresh them."
                    )
                    HybridLinkedInLookup._voyager_api_blocked = True
                    self.voyager_client = None
                else:
                    logger.warning(f"Voyager lookup failed for {name}: {e}")

        # 2. Fall back to browser-based lookup (slower but always works)
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
        # Check master switch - LinkedIn searching is disabled
        if not Config.LINKEDIN_SEARCH_ENABLED:
            logger.info("LinkedIn search disabled (LINKEDIN_SEARCH_ENABLED=false)")
            return (None, None)

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
    Extract LinkedIn session cookies by opening a Playwright browser window
    where the user can log in to LinkedIn. Cookies are captured automatically
    once the login completes.

    Falls back to reading Chrome/Edge cookie databases directly if the browser
    is closed (no file-lock).

    Returns:
        Tuple of (li_at, jsessionid) cookies, or ("", "") on failure.
    """
    # --- Method 1: Try reading cookie DB directly (works if browser is closed) ---
    li_at, jsessionid = _try_read_cookie_db()
    if li_at:
        return li_at, jsessionid

    # --- Method 2: Playwright interactive login ---
    return _extract_via_playwright()


def _try_read_cookie_db() -> tuple[str, str]:
    """Try to read LinkedIn cookies from Chrome/Edge SQLite databases.

    Only works when the browser is *not* running (file not locked).
    Handles AES-256-GCM decryption via Windows DPAPI.
    """
    import os
    import json
    import base64
    import sqlite3
    import shutil
    import tempfile

    try:
        import ctypes
        import ctypes.wintypes
    except Exception:
        return ("", "")

    browser_paths: list[tuple[str, str]] = [
        (
            "Chrome",
            os.path.expandvars(
                r"%LOCALAPPDATA%\Google\Chrome\User Data"
            ),
        ),
        (
            "Edge",
            os.path.expandvars(
                r"%LOCALAPPDATA%\Microsoft\Edge\User Data"
            ),
        ),
    ]

    for browser_name, user_data_dir in browser_paths:
        cookies_path = os.path.join(
            user_data_dir, "Default", "Network", "Cookies"
        )
        local_state_path = os.path.join(user_data_dir, "Local State")
        if not os.path.exists(cookies_path) or not os.path.exists(local_state_path):
            continue

        tmp_db = os.path.join(tempfile.gettempdir(), "smp_cookies.db")
        try:
            shutil.copy2(cookies_path, tmp_db)
        except (PermissionError, OSError):
            # Browser is running — file locked
            continue

        try:
            # Read the encryption key from Local State
            with open(local_state_path, "r", encoding="utf-8") as f:
                local_state = json.load(f)
            encrypted_key_b64 = local_state["os_crypt"]["encrypted_key"]
            encrypted_key = base64.b64decode(encrypted_key_b64)
            # Strip the "DPAPI" prefix (first 5 bytes)
            encrypted_key = encrypted_key[5:]

            # Decrypt with Windows DPAPI
            class DATA_BLOB(ctypes.Structure):
                _fields_ = [
                    ("cbData", ctypes.wintypes.DWORD),
                    ("pbData", ctypes.POINTER(ctypes.c_char)),
                ]

            key_blob_in = DATA_BLOB(
                len(encrypted_key),
                ctypes.cast(
                    ctypes.create_string_buffer(encrypted_key, len(encrypted_key)),
                    ctypes.POINTER(ctypes.c_char),
                ),
            )
            key_blob_out = DATA_BLOB()
            if not ctypes.windll.crypt32.CryptUnprotectData(
                ctypes.byref(key_blob_in),
                None, None, None, None, 0,
                ctypes.byref(key_blob_out),
            ):
                logger.debug(f"DPAPI decryption failed for {browser_name}")
                continue

            aes_key = ctypes.string_at(
                key_blob_out.pbData, key_blob_out.cbData
            )

            # Read cookies from SQLite
            conn = sqlite3.connect(tmp_db)
            cur = conn.cursor()
            cur.execute(
                "SELECT name, encrypted_value FROM cookies "
                "WHERE host_key LIKE '%.linkedin.com%' "
                "AND name IN ('li_at', 'JSESSIONID')"
            )
            rows = cur.fetchall()
            conn.close()

            li_at = ""
            jsessionid = ""
            for name, enc_val in rows:
                if not enc_val:
                    continue
                # Chrome v80+ uses AES-256-GCM: v10 + 12-byte nonce + ciphertext + 16-byte tag
                if enc_val[:3] == b"v10" or enc_val[:3] == b"v20":
                    nonce = enc_val[3:15]
                    ciphertext_tag = enc_val[15:]
                    try:
                        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
                        aes_gcm = AESGCM(aes_key)
                        decrypted = aes_gcm.decrypt(nonce, ciphertext_tag, None)
                        value = decrypted.decode("utf-8", errors="replace")
                    except ImportError:
                        logger.debug(
                            "cryptography package needed for cookie decryption"
                        )
                        break
                    except Exception as e:
                        logger.debug(f"AES decryption failed for {name}: {e}")
                        continue
                else:
                    # Fallback: old-style DPAPI-only encryption
                    blob_in = DATA_BLOB(
                        len(enc_val),
                        ctypes.cast(
                            ctypes.create_string_buffer(enc_val, len(enc_val)),
                            ctypes.POINTER(ctypes.c_char),
                        ),
                    )
                    blob_out = DATA_BLOB()
                    if ctypes.windll.crypt32.CryptUnprotectData(
                        ctypes.byref(blob_in),
                        None, None, None, None, 0,
                        ctypes.byref(blob_out),
                    ):
                        value = ctypes.string_at(
                            blob_out.pbData, blob_out.cbData
                        ).decode("utf-8", errors="replace")
                    else:
                        continue

                if name == "li_at":
                    li_at = value
                elif name == "JSESSIONID":
                    jsessionid = value.strip('"')

            if li_at:
                logger.info(
                    f"Extracted LinkedIn cookies from {browser_name} database"
                )
                return li_at, jsessionid

        except Exception as e:
            logger.debug(f"Cookie DB read failed for {browser_name}: {e}")
        finally:
            try:
                os.remove(tmp_db)
            except OSError:
                pass

    return ("", "")


def _extract_via_playwright() -> tuple[str, str]:
    """Open the user's real Chrome or Edge browser for LinkedIn login.

    Uses the installed browser (not Playwright's bundled Chromium) so LinkedIn
    doesn't block the login as bot/automation.  Waits up to 3 minutes for the
    ``li_at`` cookie to appear after login (including any 2FA prompts).

    Returns:
        Tuple of (li_at, jsessionid) or ("", "") on failure/cancel.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("Playwright not installed — pip install playwright")
        return ("", "")

    import os
    import time as _time

    # Use a dedicated profile so it doesn't interfere with the user's
    # main browser profile, but persists across runs so LinkedIn remembers
    # the device and avoids extra verification on repeat logins.
    profile_dir = os.path.join(
        os.path.expandvars(r"%LOCALAPPDATA%"),
        "SocialMediaPublisher",
        "playwright_profile",
    )
    os.makedirs(profile_dir, exist_ok=True)

    # Try Chrome first, then Edge — both are real browsers that bypass
    # LinkedIn's automation detection.
    channels = ["chrome", "msedge"]

    print("\n  Opening your browser — please log in to LinkedIn.")
    print("  (The window will close automatically once login is detected.)")
    print("  You have up to 3 minutes to complete login + any 2FA.\n")

    li_at = ""
    jsessionid = ""
    last_error = ""

    for channel in channels:
        try:
            with sync_playwright() as pw:
                context = pw.chromium.launch_persistent_context(
                    user_data_dir=profile_dir,
                    channel=channel,
                    headless=False,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                    ],
                    ignore_default_args=["--enable-automation"],
                    viewport={"width": 1280, "height": 900},
                )

                # Navigate the first tab (persistent context opens one)
                page = context.pages[0] if context.pages else context.new_page()
                page.goto(
                    "https://www.linkedin.com/login",
                    wait_until="domcontentloaded",
                )

                # Poll for the li_at cookie (set after successful login)
                deadline = _time.time() + 180  # 3 minutes
                while _time.time() < deadline:
                    try:
                        cookies = context.cookies("https://www.linkedin.com")
                    except Exception:
                        # Browser was closed by user
                        break
                    for c in cookies:
                        if c["name"] == "li_at" and c["value"]:
                            li_at = c["value"]
                        elif c["name"] == "JSESSIONID" and c["value"]:
                            jsessionid = c["value"].strip('"')
                    if li_at:
                        break
                    _time.sleep(1.5)

                context.close()

            if li_at:
                break

        except Exception as e:
            last_error = str(e)
            logger.debug(f"Playwright with {channel} failed: {e}")
            continue

    if li_at:
        logger.info("Extracted LinkedIn cookies via browser login")
    else:
        if last_error:
            logger.warning(f"Browser cookie extraction failed: {last_error}")
        else:
            logger.warning(
                "Login not completed within 3 minutes — no cookies captured"
            )

    return li_at, jsessionid


# ============================================================================
# Module Tests
# ============================================================================


def _create_module_tests():
    """Create and run tests for linkedin_voyager_client module."""
    from test_framework import TestSuite

    suite = TestSuite("LinkedIn Voyager Client", "linkedin_voyager_client.py")

    def test_client_init_no_auth():
        """Test LinkedInVoyagerClient without authentication."""
        client = LinkedInVoyagerClient(li_at=None, jsessionid=None)
        assert client.is_authenticated() is False

    def test_client_init_with_auth():
        """Test LinkedInVoyagerClient stores auth tokens."""
        client = LinkedInVoyagerClient(li_at="test-token", jsessionid="test-session")
        assert client.li_at == "test-token"
        assert client.jsessionid == "test-session"

    def test_client_close():
        """Test LinkedInVoyagerClient close method."""
        client = LinkedInVoyagerClient(li_at=None)
        client.close()
        assert client._session is None

    def test_client_context_manager():
        """Test LinkedInVoyagerClient as context manager."""
        with LinkedInVoyagerClient(li_at=None) as client:
            assert client is not None
        assert client._session is None

    def test_parse_people_results_empty():
        """Test _parse_people_results with empty data."""
        client = LinkedInVoyagerClient(li_at=None)
        result = client._parse_people_results({})
        assert isinstance(result, list)
        assert len(result) == 0

    def test_parse_people_results_no_included():
        """Test _parse_people_results with missing included key."""
        client = LinkedInVoyagerClient(li_at=None)
        result = client._parse_people_results({"data": "something"})
        assert len(result) == 0

    def test_parse_people_results_with_person():
        """Test _parse_people_results with valid person data."""
        client = LinkedInVoyagerClient(li_at=None)
        data = {
            "included": [
                {
                    "$type": "com.linkedin.voyager.dash.search.SearchNormalizedPerson",
                    "entityUrn": "urn:li:fsd_profile:johndoe",
                    "objectUrn": "urn:li:member:12345",
                    "person": {
                        "firstName": "John",
                        "lastName": "Doe",
                    },
                    "primarySubtitle": {"text": "CTO at Acme Corp"},
                    "secondarySubtitle": {"text": "San Francisco, CA"},
                }
            ]
        }
        result = client._parse_people_results(data)
        assert len(result) == 1
        assert result[0].full_name == "John Doe"
        assert result[0].public_id == "johndoe"
        assert result[0].headline == "CTO at Acme Corp"
        assert result[0].location == "San Francisco, CA"

    def test_parse_company_results_empty():
        """Test _parse_company_results with empty data."""
        client = LinkedInVoyagerClient(li_at=None)
        result = client._parse_company_results({})
        assert isinstance(result, list)
        assert len(result) == 0

    def test_hybrid_lookup_init():
        """Test HybridLinkedInLookup initialization."""
        lookup = HybridLinkedInLookup()
        assert lookup is not None

    suite.run_test(
        test_name="Client init no auth",
        test_func=test_client_init_no_auth,
        test_summary="Tests client without authentication",
        method_description="Creates client with no tokens",
        expected_outcome="is_authenticated returns False",
    )
    suite.run_test(
        test_name="Client init with auth",
        test_func=test_client_init_with_auth,
        test_summary="Tests client stores auth tokens",
        method_description="Creates client with test tokens",
        expected_outcome="Tokens stored correctly",
    )
    suite.run_test(
        test_name="Client close",
        test_func=test_client_close,
        test_summary="Tests close method",
        method_description="Closes a client session",
        expected_outcome="HTTP client set to None",
    )
    suite.run_test(
        test_name="Client context manager",
        test_func=test_client_context_manager,
        test_summary="Tests with statement usage",
        method_description="Uses client as context manager",
        expected_outcome="Client cleaned up on exit",
    )
    suite.run_test(
        test_name="Parse people results - empty",
        test_func=test_parse_people_results_empty,
        test_summary="Tests parsing empty response",
        method_description="Parses empty dict",
        expected_outcome="Returns empty list",
    )
    suite.run_test(
        test_name="Parse people results - no included",
        test_func=test_parse_people_results_no_included,
        test_summary="Tests parsing response without included key",
        method_description="Parses dict without included key",
        expected_outcome="Returns empty list",
    )
    suite.run_test(
        test_name="Parse people results - with person",
        test_func=test_parse_people_results_with_person,
        test_summary="Tests parsing valid person data",
        method_description="Parses realistic Voyager response",
        expected_outcome="Returns profile with correct fields",
    )
    suite.run_test(
        test_name="Parse company results - empty",
        test_func=test_parse_company_results_empty,
        test_summary="Tests parsing empty company response",
        method_description="Parses empty dict for companies",
        expected_outcome="Returns empty list",
    )
    suite.run_test(
        test_name="HybridLinkedInLookup init",
        test_func=test_hybrid_lookup_init,
        test_summary="Tests HybridLinkedInLookup creation",
        method_description="Creates HybridLinkedInLookup instance",
        expected_outcome="Instance created successfully",
    )

    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)

if __name__ == "__main__":
    run_comprehensive_tests()
