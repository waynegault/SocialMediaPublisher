"""LinkedIn company page lookup using Gemini with Google Search grounding."""

import logging
import re
import time
from typing import Optional

import httpx
from google import genai  # type: ignore

from config import Config

logger = logging.getLogger(__name__)


class LinkedInCompanyLookup:
    """Look up LinkedIn company pages using Gemini with Google Search grounding."""

    def __init__(self, genai_client: Optional[genai.Client] = None):
        """Initialize the LinkedIn company lookup service.

        Args:
            genai_client: Optional Gemini client. If not provided, creates one using Config.
        """
        if genai_client:
            self.client = genai_client
        elif Config.GEMINI_API_KEY:
            self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        else:
            self.client = None
            logger.warning(
                "GEMINI_API_KEY not configured - LinkedIn company lookup disabled"
            )

        # HTTP client for LinkedIn API calls
        self._http_client: Optional[httpx.Client] = None

    def _get_http_client(self) -> httpx.Client:
        """Get or create HTTP client for LinkedIn API calls."""
        if self._http_client is None:
            self._http_client = httpx.Client(
                timeout=10.0,
                headers={
                    "Authorization": f"Bearer {Config.LINKEDIN_ACCESS_TOKEN}",
                    "X-Restli-Protocol-Version": "2.0.0",
                    "LinkedIn-Version": "202401",
                },
            )
        return self._http_client

    def lookup_organization_by_vanity_name(
        self, vanity_name: str
    ) -> Optional[tuple[str, int]]:
        """
        Look up organization URN using LinkedIn's official API.

        Uses the endpoint: https://api.linkedin.com/v2/organizations?q=vanityName&vanityName=xxx

        Args:
            vanity_name: The vanity name (slug) from the LinkedIn URL, e.g., "stanford-university"

        Returns:
            Tuple of (urn, organization_id) if found, None otherwise
            e.g., ("urn:li:organization:12345", 12345)
        """
        if not Config.LINKEDIN_ACCESS_TOKEN:
            logger.warning("LINKEDIN_ACCESS_TOKEN not configured - cannot lookup URN")
            return None

        if not vanity_name:
            return None

        # Clean the vanity name (remove any URL parts if present)
        vanity_name = vanity_name.strip().lower()
        if "/" in vanity_name:
            # Extract just the slug if a URL was passed
            match = re.search(
                r"linkedin\.com/(?:company|school)/([\w\-]+)", vanity_name
            )
            if match:
                vanity_name = match.group(1)

        logger.info(f"Looking up LinkedIn organization by vanity name: {vanity_name}")

        try:
            client = self._get_http_client()
            response = client.get(
                "https://api.linkedin.com/v2/organizations",
                params={"q": "vanityName", "vanityName": vanity_name},
            )

            if response.status_code == 200:
                data = response.json()
                elements = data.get("elements", [])

                if elements:
                    org = elements[0]
                    org_id = org.get("id")
                    if org_id:
                        urn = f"urn:li:organization:{org_id}"
                        logger.info(
                            f"Found organization via API: {vanity_name} -> {urn}"
                        )
                        return (urn, org_id)

                logger.info(f"No organization found for vanity name: {vanity_name}")
                return None

            elif response.status_code == 401:
                logger.error("LinkedIn API authentication failed - check access token")
                return None
            elif response.status_code == 403:
                logger.error(
                    "LinkedIn API access denied - token may lack required permissions"
                )
                return None
            else:
                logger.warning(
                    f"LinkedIn API returned {response.status_code} for {vanity_name}"
                )
                return None

        except httpx.TimeoutException:
            logger.warning(f"Timeout calling LinkedIn API for: {vanity_name}")
            return None
        except Exception as e:
            logger.error(f"Error calling LinkedIn API for {vanity_name}: {e}")
            return None

    def validate_linkedin_url(self, url: str) -> tuple[bool, Optional[str]]:
        """
        Validate that a LinkedIn URL is well-formed and extract the organization slug.

        Note: LinkedIn blocks direct HTTP requests with 999 status, so we cannot
        validate by fetching the page. Instead we validate the URL format and
        extract the organization slug which can be used for further lookups.

        Args:
            url: LinkedIn company or school URL

        Returns:
            Tuple of (is_valid, organization_slug)
            organization_slug is the path component like "stanford-university" or "basf"
        """
        if not url:
            return (False, None)

        # Validate URL format
        pattern = r"https?://(?:www\.)?linkedin\.com/(company|school)/([\w\-]+)"
        match = re.match(pattern, url)

        if not match:
            logger.info(f"Invalid LinkedIn URL format: {url}")
            return (False, None)

        org_type = match.group(1)  # "company" or "school"
        org_slug = match.group(2)  # e.g., "stanford-university"

        logger.info(f"Valid LinkedIn {org_type} URL, slug: {org_slug}")
        return (True, org_slug)

    def lookup_organization_urn(self, url: str) -> Optional[str]:
        """
        Find the numeric organization ID for a LinkedIn company/school URL.

        First tries LinkedIn's official API (requires valid access token).
        Falls back to Gemini with Google Search if API fails.

        Args:
            url: LinkedIn company or school URL

        Returns:
            URN string like "urn:li:organization:12345" if found, None otherwise
        """
        if not url:
            return None

        # Extract the slug from the URL
        pattern = r"linkedin\.com/(company|school)/([\w\-]+)"
        match = re.search(pattern, url)
        if not match:
            return None

        org_slug = match.group(2)

        # Try 1: Use LinkedIn's official API
        result = self.lookup_organization_by_vanity_name(org_slug)
        if result:
            urn, _ = result
            return urn

        # Try 2: Fall back to Gemini search (usually doesn't work but worth trying)
        return self._lookup_urn_via_gemini(url, org_slug)

    def _lookup_urn_via_gemini(self, url: str, org_slug: str) -> Optional[str]:
        """Fall back to Gemini search for organization URN (rarely works)."""
        if not self.client:
            return None

        org_type = "company" if "/company/" in url else "school"

        prompt = f"""Find the LinkedIn organization ID for: {url}

I need the numeric organization ID (also called company ID) for this LinkedIn page.
The slug is: {org_slug}

Search for information about this LinkedIn {org_type} page's numeric ID.
The ID is typically a number like 1234567 or similar.

RESPONSE FORMAT:
If you find the organization ID, respond with just the number:
12345678

If you cannot find it, respond with:
NOT_FOUND"""

        try:
            response = self.client.models.generate_content(
                model=Config.MODEL_TEXT,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "max_output_tokens": 64,
                },
            )

            if not response.text:
                return None

            result = response.text.strip()

            if "NOT_FOUND" in result.upper():
                return None

            # Extract numeric ID from response
            id_match = re.search(r"\b(\d{5,15})\b", result)
            if id_match:
                org_id = id_match.group(1)
                urn = f"urn:li:organization:{org_id}"
                logger.info(f"Found organization URN via Gemini for {org_slug}: {urn}")
                return urn

            return None

        except Exception as e:
            logger.error(f"Error looking up organization URN for {url}: {e}")
            return None

    def search_company(
        self, company_name: str, validate: bool = True
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Search for a LinkedIn company page by company name.

        Uses Gemini with Google Search grounding to find the LinkedIn company URL.
        Tries multiple search strategies if initial search fails.

        Args:
            company_name: Company/organization name (required)
            validate: Whether to validate the URL format (default True)

        Returns:
            Tuple of (linkedin_url, organization_slug) if found, (None, None) otherwise
        """
        if not self.client:
            logger.warning("Cannot search - Gemini client not initialized")
            return (None, None)

        if not company_name or not company_name.strip():
            logger.warning("Cannot search - company name is required")
            return (None, None)

        company_name = company_name.strip()
        logger.info(f"Searching LinkedIn for company: {company_name}")

        # Try multiple search strategies
        search_strategies = [
            # Strategy 1: Direct company name search
            f'"{company_name}" site:linkedin.com/company OR site:linkedin.com/school',
            # Strategy 2: Simplified name (remove common suffixes)
            self._get_simplified_search(company_name),
            # Strategy 3: Acronym if applicable
            self._get_acronym_search(company_name),
        ]

        # Remove None strategies
        search_strategies = [s for s in search_strategies if s]

        for i, search_query in enumerate(search_strategies):
            if i > 0:
                logger.info(
                    f"Trying alternative search strategy {i + 1} for: {company_name}"
                )

            url, urn = self._search_with_query(company_name, search_query, validate)
            if url:
                return (url, urn)

            # Small delay between retries
            if i < len(search_strategies) - 1:
                time.sleep(0.5)

        logger.info(f"No LinkedIn company page found for: {company_name}")
        return (None, None)

    def _get_simplified_search(self, company_name: str) -> Optional[str]:
        """Generate a simplified search query by removing common suffixes."""
        # Common suffixes to remove for cleaner searching
        suffixes_to_remove = [
            " Corporation",
            " Corp",
            " Inc",
            " LLC",
            " Ltd",
            " Limited",
            " GmbH",
            " AG",
            " SE",
            " PLC",
            " of Technology",
            " University",
            " School of Engineering",
            " Samueli School of Engineering",
            " School of",
        ]

        simplified = company_name
        for suffix in suffixes_to_remove:
            if simplified.lower().endswith(suffix.lower()):
                simplified = simplified[: -len(suffix)].strip()

        # Also try removing "University of" prefix
        if simplified.lower().startswith("university of "):
            simplified = simplified[14:].strip()

        if simplified != company_name:
            return f'"{simplified}" linkedin company OR school'
        return None

    def _get_acronym_search(self, company_name: str) -> Optional[str]:
        """Generate an acronym-based search if company name is multi-word."""
        # Common known acronyms
        known_acronyms = {
            "Massachusetts Institute of Technology": "MIT",
            "University of California, Santa Barbara": "UCSB",
            "University of California, Los Angeles": "UCLA",
            "University of California, Berkeley": "UC Berkeley",
            "California Institute of Technology": "Caltech",
            "Georgia Institute of Technology": "Georgia Tech",
        }

        if company_name in known_acronyms:
            acronym = known_acronyms[company_name]
            return f'"{acronym}" site:linkedin.com/school OR site:linkedin.com/company'

        return None

    def _search_with_query(
        self, company_name: str, search_query: str, validate: bool
    ) -> tuple[Optional[str], Optional[str]]:
        """Execute a search with a specific query."""
        prompt = f"""Find the official LinkedIn page for: {company_name}

Search for: {search_query}

TASK: Find the MAIN LinkedIn company or school page URL.

IMPORTANT RULES:
1. Find the MAIN/OFFICIAL page for the whole organization, NOT a department or subsidiary
2. For universities, look for the main university page, not individual departments
3. The URL must be linkedin.com/company/xxx or linkedin.com/school/xxx
4. Return the most official/verified page if multiple exist
5. If the organization has multiple entities, return the main/parent one

EXAMPLES of what to return:
- For "Stanford University" → https://www.linkedin.com/school/stanford-university
- For "Google" → https://www.linkedin.com/company/google
- For "MIT" → https://www.linkedin.com/school/massachusetts-institute-of-technology

RESPONSE FORMAT:
If found, respond with ONLY the LinkedIn URL on a single line:
https://www.linkedin.com/company/company-name
OR
https://www.linkedin.com/school/school-name

If NOT found or only department pages exist, respond with exactly:
NOT_FOUND"""

        try:
            response = self.client.models.generate_content(
                model=Config.MODEL_TEXT,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "max_output_tokens": 256,
                },
            )

            if not response.text:
                return (None, None)

            result = response.text.strip()

            # Check for NOT_FOUND response
            if "NOT_FOUND" in result.upper():
                return (None, None)

            # Extract LinkedIn company URL from response
            linkedin_url = self._extract_company_url(result)

            if not linkedin_url:
                return (None, None)

            # Validate the URL if requested
            if validate:
                is_valid, org_urn = self.validate_linkedin_url(linkedin_url)
                if is_valid:
                    logger.info(
                        f"Found and validated LinkedIn page for {company_name}: {linkedin_url}"
                    )
                    return (linkedin_url, org_urn)
                else:
                    logger.info(
                        f"LinkedIn URL failed validation for {company_name}: {linkedin_url}"
                    )
                    return (None, None)
            else:
                logger.info(
                    f"Found LinkedIn page for {company_name}: {linkedin_url} (not validated)"
                )
                return (linkedin_url, None)

        except Exception as e:
            logger.error(f"Error searching LinkedIn for {company_name}: {e}")
            return (None, None)

    def _extract_company_url(self, text: str) -> Optional[str]:
        """Extract a LinkedIn company/school URL from text."""
        # Pattern for LinkedIn company or school URLs
        pattern = r"https?://(?:www\.)?linkedin\.com/(?:company|school)/[\w\-]+"

        match = re.search(pattern, text)
        if match:
            url = match.group(0)
            # Normalize to https://www.linkedin.com format
            if not url.startswith("https://www."):
                url = url.replace("http://", "https://")
                url = url.replace("https://linkedin.com", "https://www.linkedin.com")
            return url
        return None

    def populate_company_profiles(
        self,
        relevant_people: list[dict],
        delay_between_requests: float = 1.0,
    ) -> tuple[int, int, dict[str, tuple[str, Optional[str]]]]:
        """
        Look up LinkedIn company pages for the companies mentioned in relevant_people.

        Instead of looking up individual profiles, we look up company pages which
        are public and easier to find. The company URLs can be used for tagging
        in LinkedIn posts.

        Args:
            relevant_people: List of person dicts with name, company, position, linkedin_profile
            delay_between_requests: Seconds to wait between API calls (rate limiting)

        Returns:
            Tuple of (companies_found, companies_not_found, company_data_dict)
            company_data_dict maps company names to (url, urn) tuples
        """
        if not self.client:
            logger.warning("Cannot populate profiles - Gemini client not initialized")
            return (0, 0, {})

        # Extract unique company names
        companies = set()
        for person in relevant_people:
            company = person.get("company", "").strip()
            if company:
                companies.add(company)

        if not companies:
            logger.info("No companies found in relevant_people")
            return (0, 0, {})

        companies_found = 0
        companies_not_found = 0
        company_data: dict[str, tuple[str, Optional[str]]] = {}

        for i, company in enumerate(companies):
            linkedin_url, org_urn = self.search_company(company)

            if linkedin_url:
                company_data[company] = (linkedin_url, org_urn)
                companies_found += 1
            else:
                companies_not_found += 1

            # Rate limiting - be respectful of API limits
            if i < len(companies) - 1:
                time.sleep(delay_between_requests)

        # Update the relevant_people entries with company LinkedIn URLs
        # Store in linkedin_profile field as company page (better than nothing)
        for person in relevant_people:
            company = person.get("company", "").strip()
            if company and company in company_data:
                url, slug = company_data[company]
                # Only update if no profile is set yet
                if not person.get("linkedin_profile", "").strip():
                    person["linkedin_profile"] = url
                # Store the slug in a new field if available
                if slug:
                    person["linkedin_slug"] = slug

                    # Try to look up the organization URN via LinkedIn API
                    urn = self.lookup_organization_urn(url)
                    if urn:
                        person["linkedin_urn"] = urn

        return (companies_found, companies_not_found, company_data)

    def close(self) -> None:
        """Clean up HTTP client resources."""
        if self._http_client:
            self._http_client.close()
            self._http_client = None


# Backwards compatibility alias
LinkedInProfileLookup = LinkedInCompanyLookup
