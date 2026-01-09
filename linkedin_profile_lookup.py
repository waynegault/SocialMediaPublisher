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
        # Note: caller (search_company) guarantees self.client is not None
        assert self.client is not None

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

    def _extract_person_url(self, text: str) -> Optional[str]:
        """Extract a LinkedIn personal profile URL from text."""
        # Pattern for LinkedIn personal profile URLs (linkedin.com/in/username)
        pattern = r"https?://(?:www\.)?linkedin\.com/in/([\w\-]+)"

        match = re.search(pattern, text)
        if match:
            url = match.group(0)
            # Normalize to https://www.linkedin.com format
            if not url.startswith("https://www."):
                url = url.replace("http://", "https://")
                url = url.replace("https://linkedin.com", "https://www.linkedin.com")
            return url
        return None

    def search_person(
        self, name: str, company: str, position: Optional[str] = None
    ) -> Optional[str]:
        """
        Search for an individual person's LinkedIn profile using Gemini with Google Search.

        Args:
            name: Person's name (e.g., "Suzanne Farid")
            company: Company or organization they work at (e.g., "UCL")
            position: Optional job title to help identify the right person

        Returns:
            LinkedIn profile URL if found, None otherwise
        """
        if not self.client:
            return None

        if not name or not company:
            return None

        logger.info(f"Searching for person LinkedIn profile: {name} at {company}")

        # Build search context
        position_context = f", {position}" if position else ""

        prompt = f"""Find the LinkedIn profile for this specific person:

Name: {name}
Company/Organization: {company}{position_context}

TASK: Search for the LinkedIn personal profile page for this individual.

RULES:
1. Find a linkedin.com/in/username profile URL for THIS SPECIFIC PERSON
2. The person should work at or be affiliated with {company}
3. Do NOT return company pages (linkedin.com/company/...)
4. Do NOT return school pages (linkedin.com/school/...)
5. Only return a profile if you're confident it's the right person

RESPONSE FORMAT:
If you find their LinkedIn profile, respond with ONLY the URL like:
https://www.linkedin.com/in/username

If you cannot find their personal LinkedIn profile, respond exactly:
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
                return None

            result = response.text.strip()

            if "NOT_FOUND" in result.upper():
                logger.info(f"Person profile not found: {name} at {company}")
                return None

            # Extract LinkedIn personal profile URL
            linkedin_url = self._extract_person_url(result)
            if linkedin_url:
                logger.info(f"Found person profile: {name} -> {linkedin_url}")
                return linkedin_url

            return None

        except Exception as e:
            logger.error(f"Error searching for person {name}: {e}")
            return None

    def _extract_department_name(self, position: str) -> Optional[str]:
        """Extract department name from a position title if present."""
        if not position:
            return None

        position_lower = position.lower()

        # Common patterns for department positions
        department_patterns = [
            r"(?:head|director|chair|dean|professor)\s+(?:of\s+)?(?:the\s+)?(?:department\s+(?:of|for)\s+)?(.+?)(?:\s+department)?$",
            r"(?:department|dept\.?)\s+(?:of|for)\s+(.+?)(?:\s+(?:head|director|chair|dean))?$",
            r"(.+?)\s+department\s+head",
        ]

        for pattern in department_patterns:
            match = re.search(pattern, position_lower)
            if match:
                dept = match.group(1).strip()
                # Clean up common suffixes
                dept = re.sub(r"\s+at\s+.*$", "", dept)
                # Remove trailing title words
                dept = re.sub(r"\s+(?:head|director|chair|dean)$", "", dept)
                if len(dept) > 3:  # Avoid very short matches
                    return dept.title()

        return None

    def _generate_department_slug_candidates(
        self, department: str, parent_org: str, parent_slug: Optional[str] = None
    ) -> list[str]:
        """Generate likely LinkedIn slug patterns for a department."""
        candidates = []

        # Normalize names
        dept_lower = department.lower().strip()
        org_lower = parent_org.lower().strip()

        # Clean department name for slug (with hyphens)
        dept_slug = re.sub(r"[^a-z0-9\s]", "", dept_lower)
        dept_slug = re.sub(r"\s+", "-", dept_slug.strip())

        # Clean org name for slug (with hyphens)
        org_slug = re.sub(r"[^a-z0-9\s]", "", org_lower)
        org_slug = re.sub(r"\s+", "-", org_slug.strip())

        # No-hyphen versions (e.g., "uclaengineering" instead of "ucla-engineering")
        dept_nohyphen = dept_slug.replace("-", "")
        org_nohyphen = org_slug.replace("-", "")

        # Common patterns - try no-hyphen versions first (more likely for schools)
        # Pattern: orgdept (e.g., uclaengineering)
        candidates.append(f"{org_nohyphen}{dept_nohyphen}")

        # Pattern: org-dept (e.g., ucla-engineering)
        candidates.append(f"{org_slug}-{dept_slug}")

        # If parent_slug provided, try with it
        if parent_slug:
            parent_nohyphen = parent_slug.replace("-", "")
            candidates.append(f"{parent_nohyphen}{dept_nohyphen}")
            candidates.append(f"{parent_slug}-{dept_slug}")

        # Additional patterns
        candidates.append(f"{dept_slug}-{org_slug}")
        candidates.append(f"{org_slug}-{dept_nohyphen}")

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        return unique_candidates

    def search_department(
        self, department: str, parent_org: str, parent_slug: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Search for a department-specific LinkedIn page.

        First tries common URL patterns via LinkedIn API, then falls back to Gemini search.

        Args:
            department: Department name (e.g., "Biochemical Engineering")
            parent_org: Parent organization name (e.g., "UCL")
            parent_slug: Optional parent org slug to help narrow search

        Returns:
            Tuple of (linkedin_url, slug) if found, (None, None) otherwise
        """
        logger.info(
            f"Searching for department LinkedIn page: {department} at {parent_org}"
        )

        # Strategy 1: Try common URL patterns directly via LinkedIn API
        slug_candidates = self._generate_department_slug_candidates(
            department, parent_org, parent_slug
        )

        for candidate_slug in slug_candidates:
            result = self.lookup_organization_by_vanity_name(candidate_slug)
            if result:
                urn, org_id = result
                if urn:
                    url = f"https://www.linkedin.com/company/{candidate_slug}"
                    logger.info(
                        f"Found department page via API: {department} at {parent_org} -> {url}"
                    )
                    return (url, candidate_slug)

        # Strategy 2: Fall back to Gemini search
        if not self.client:
            return (None, None)

        prompt = f"""Find the LinkedIn company/school page for this specific department, faculty, or school:

Department/Faculty: {department}
Parent Organization: {parent_org}
{f"Parent LinkedIn slug: {parent_slug}" if parent_slug else ""}

TASK: Search for a LinkedIn page that is specifically for this department, faculty, or school unit - NOT the main parent organization.

SEARCH EXAMPLES:
- "UCL Biochemical Engineering" might have linkedin.com/school/ucl-biochemical-engineering
- "MIT Sloan" might have linkedin.com/school/mit-sloan
- "Stanford Graduate School of Business" might have linkedin.com/school/stanford-gsb

IMPORTANT:
- Many university departments have their own LinkedIn pages
- Look for pages with names like "{parent_org} {department}" or "{department} at {parent_org}"
- The URL format is linkedin.com/company/xxx or linkedin.com/school/xxx
- Do NOT return the main parent organization page (e.g., don't return UCL main page)
- The page name should specifically reference the department or faculty

RESPONSE:
If you find a department-specific LinkedIn page, respond with ONLY the full URL, like:
https://www.linkedin.com/school/ucl-biochemical-engineering

If you cannot find a department-specific page (only the parent org exists), respond with exactly:
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

            if "NOT_FOUND" in result.upper():
                return (None, None)

            # Extract LinkedIn URL
            linkedin_url = self._extract_company_url(result)
            if not linkedin_url:
                return (None, None)

            # Validate the URL format
            is_valid, slug = self.validate_linkedin_url(linkedin_url)
            if is_valid:
                logger.info(
                    f"Found department page via search: {department} at {parent_org} -> {linkedin_url}"
                )
                return (linkedin_url, slug)

            return (None, None)

        except Exception as e:
            logger.error(f"Error searching for department {department}: {e}")
            return (None, None)

    def populate_company_profiles(
        self,
        relevant_people: list[dict],
        delay_between_requests: float = 1.0,
    ) -> tuple[int, int, dict[str, tuple[str, Optional[str]]]]:
        """
        Look up LinkedIn profiles for people using a hierarchical approach.

        For each person, tries to find profiles in this order:
        1. Personal LinkedIn profile (linkedin.com/in/username)
        2. Department-specific page (if position indicates a department)
        3. Parent organization page

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

        # First, look up parent organizations (needed as fallback)
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
        company_data: dict[str, tuple[str, Optional[str], Optional[str]]] = {}

        # Look up parent organizations first (as fallback)
        for i, company in enumerate(companies):
            linkedin_url, slug = self.search_company(company)

            if linkedin_url:
                # Look up the organization URN via LinkedIn API
                urn = self.lookup_organization_urn(linkedin_url)
                company_data[company] = (linkedin_url, slug, urn)
                companies_found += 1
            else:
                companies_not_found += 1

            # Rate limiting
            if i < len(companies) - 1:
                time.sleep(delay_between_requests)

        # Cache for department lookups to avoid duplicate searches
        department_cache: dict[
            str, tuple[Optional[str], Optional[str], Optional[str]]
        ] = {}

        # Cache for person lookups to avoid duplicate searches
        person_cache: dict[str, Optional[str]] = {}

        # Process each person with hierarchical lookup
        for person in relevant_people:
            name = person.get("name", "").strip()
            company = person.get("company", "").strip()
            position = person.get("position", "").strip()

            if not company:
                continue

            # ============================================================
            # HIERARCHY LEVEL 1: Try to find the individual's personal profile
            # ============================================================
            if name:
                person_cache_key = f"{name}@{company}"

                if person_cache_key not in person_cache:
                    logger.info(f"Level 1: Searching for personal profile: {name} at {company}")
                    person_url = self.search_person(name, company, position)
                    person_cache[person_cache_key] = person_url
                    time.sleep(delay_between_requests)

                person_url = person_cache[person_cache_key]
                if person_url:
                    person["linkedin_profile"] = person_url
                    # Extract slug from personal profile URL
                    match = re.search(r"linkedin\.com/in/([\w\-]+)", person_url)
                    if match:
                        person["linkedin_slug"] = match.group(1)
                    # Personal profiles don't have organization URNs
                    person["linkedin_urn"] = None
                    person["linkedin_profile_type"] = "personal"
                    continue  # Found personal profile, skip to next person

            # ============================================================
            # HIERARCHY LEVEL 2: Try to find department-specific page
            # ============================================================
            department = self._extract_department_name(position)

            if department:
                dept_cache_key = f"{department}@{company}"

                if dept_cache_key not in department_cache:
                    # Get parent org slug if available
                    parent_slug = None
                    if company in company_data:
                        parent_slug = company_data[company][1]

                    logger.info(
                        f"Level 2: Looking for department page: {department} at {company}"
                    )
                    dept_url, dept_slug = self.search_department(
                        department, company, parent_slug
                    )

                    if dept_url:
                        # Look up URN for department
                        dept_urn = self.lookup_organization_urn(dept_url)
                        department_cache[dept_cache_key] = (dept_url, dept_slug, dept_urn)
                        time.sleep(delay_between_requests)
                    else:
                        department_cache[dept_cache_key] = (None, None, None)
                        time.sleep(delay_between_requests * 0.5)

                # Use department data if found
                dept_data = department_cache[dept_cache_key]
                if dept_data[0]:  # Department page found
                    person["linkedin_profile"] = dept_data[0]
                    if dept_data[1]:
                        person["linkedin_slug"] = dept_data[1]
                    if dept_data[2]:
                        person["linkedin_urn"] = dept_data[2]
                    person["linkedin_profile_type"] = "department"
                    continue  # Found department, skip to next person

            # ============================================================
            # HIERARCHY LEVEL 3: Fall back to parent organization
            # ============================================================
            if company in company_data:
                url, slug, urn = company_data[company]
                logger.info(f"Level 3: Using organization fallback for {name}: {company}")
                if not person.get("linkedin_profile", "").strip():
                    person["linkedin_profile"] = url
                if slug:
                    person["linkedin_slug"] = slug
                if urn:
                    person["linkedin_urn"] = urn
                person["linkedin_profile_type"] = "organization"
            else:
                # ============================================================
                # HIERARCHY LEVEL 4: Nothing found
                # ============================================================
                logger.warning(f"Level 4: No LinkedIn profile found for {name} at {company}")
                person["linkedin_profile"] = None
                person["linkedin_slug"] = None
                person["linkedin_urn"] = None
                person["linkedin_profile_type"] = None

        # Return company_data with just (url, slug) for backwards compatibility
        return_data = {k: (v[0], v[1]) for k, v in company_data.items()}
        return (companies_found, companies_not_found, return_data)

    def close(self) -> None:
        """Clean up HTTP client resources."""
        if self._http_client:
            self._http_client.close()
            self._http_client = None


# Backwards compatibility alias
LinkedInProfileLookup = LinkedInCompanyLookup
