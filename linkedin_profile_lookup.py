"""LinkedIn company and person profile lookup using Gemini with Google Search grounding and undetected-chromedriver."""

import base64
import logging
import os
import re
import sys
import time
from typing import TYPE_CHECKING, Any, Optional, cast

import httpx
from google import genai  # type: ignore

from api_client import api_client
from config import Config

logger = logging.getLogger(__name__)

# Import undetected-chromedriver for CAPTCHA-resistant browser automation
try:
    import undetected_chromedriver as uc  # type: ignore

    UC_AVAILABLE = True
except ImportError:
    UC_AVAILABLE = False
    logger.warning(
        "undetected-chromedriver not installed - pip install undetected-chromedriver"
    )


def _suppress_uc_cleanup_errors():
    """Suppress Windows handle errors from UC Chrome cleanup."""
    if sys.platform == "win32":
        # Monkey-patch time.sleep in the UC module to suppress cleanup errors
        original_sleep = time.sleep

        def patched_sleep(seconds):
            try:
                original_sleep(seconds)
            except OSError:
                pass  # Suppress WinError 6: handle is invalid

        # Apply to UC module if loaded
        if UC_AVAILABLE:
            import undetected_chromedriver

            undetected_chromedriver.time = type(sys)("time")
            undetected_chromedriver.time.sleep = patched_sleep


# Apply the patch on module load
_suppress_uc_cleanup_errors()


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

        # Shared browser session for UC Chrome searches (reused across multiple searches)
        self._uc_driver: Optional[Any] = None

        # Track if we've verified LinkedIn login this session
        self._linkedin_login_verified = False

    def __enter__(self):
        """Context manager entry - returns self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes browser session."""
        self.close_browser()
        return False

    def _get_uc_driver(self) -> Optional[Any]:
        """Get or create a shared UC Chrome driver instance.

        The driver is reused across multiple searches for efficiency.
        Call close_browser() when done with all searches.

        Returns:
            UC Chrome driver instance, or None if UC not available
        """
        if not UC_AVAILABLE:
            logger.warning(
                "undetected-chromedriver not installed - pip install undetected-chromedriver"
            )
            return None

        # Return existing driver if still valid
        if self._uc_driver is not None:
            try:
                # Check if driver is still alive
                _ = self._uc_driver.current_url
                return self._uc_driver
            except Exception:
                # Driver is dead, clean up and create new one
                self._uc_driver = None

        # Create new driver
        try:
            from selenium.webdriver.common.by import By

            options = uc.ChromeOptions()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--disable-software-rasterizer")
            options.add_argument("--disable-extensions")
            options.add_argument("--no-first-run")
            options.add_argument("--no-default-browser-check")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-popup-blocking")

            # Use a dedicated profile directory for automation
            # This persists LinkedIn login between runs without conflicting with main Chrome
            automation_profile = os.path.expandvars(
                r"%LOCALAPPDATA%\SocialMediaPublisher\ChromeProfile"
            )
            os.makedirs(automation_profile, exist_ok=True)
            options.add_argument(f"--user-data-dir={automation_profile}")
            logger.info(f"Using automation profile at {automation_profile}")

            # Consistent user agent (random user agents are red flags)
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
            options.add_argument(f"--user-agent={user_agent}")

            self._uc_driver = uc.Chrome(
                options=options,
                use_subprocess=False,  # More stable on Windows
                suppress_welcome=True,
            )
            self._uc_driver.set_page_load_timeout(30)
            logger.debug("Created new UC Chrome driver session")
            return self._uc_driver

        except Exception as e:
            logger.error(f"Failed to create UC Chrome driver: {e}")
            return None

    def _ensure_linkedin_login(self, driver) -> bool:
        """Ensure user is logged in to LinkedIn, attempting auto-login if credentials available.

        Returns:
            True if logged in, False if user cancelled or login failed
        """
        if self._linkedin_login_verified:
            return True

        # Navigate to LinkedIn to check login status
        try:
            driver.get("https://www.linkedin.com/feed/")
            time.sleep(3)

            current_url = driver.current_url

            # If redirected to login page, need to log in
            if (
                "/login" in current_url
                or "/authwall" in current_url
                or "/checkpoint" in current_url
            ):
                # Try automatic login if credentials are configured
                if Config.LINKEDIN_USERNAME and Config.LINKEDIN_PASSWORD:
                    logger.info("Attempting automatic LinkedIn login...")
                    if self._auto_login(driver):
                        self._linkedin_login_verified = True
                        logger.info("Automatic LinkedIn login successful")
                        return True
                    else:
                        logger.warning("Automatic login failed, falling back to manual")

                # Fall back to manual login
                print("\n" + "=" * 60)
                print("LinkedIn Login Required")
                print("=" * 60)
                print("A browser window has opened. Please log in to LinkedIn.")
                print("This is required to extract @mention URNs from profiles.")
                print("Your session will be saved for future runs.")
                print(
                    "\nTip: Add LINKEDIN_USERNAME and LINKEDIN_PASSWORD to .env for auto-login."
                )
                print("\nAfter logging in, press Enter to continue...")
                input()

                # Check if login succeeded
                time.sleep(2)
                current_url = driver.current_url
                if "/login" in current_url or "/authwall" in current_url:
                    print("Login not detected. Please try again.")
                    return False

            self._linkedin_login_verified = True
            logger.info("LinkedIn login verified")
            return True

        except Exception as e:
            logger.error(f"Error checking LinkedIn login: {e}")
            return False

    def _auto_login(self, driver) -> bool:
        """Attempt automatic login using credentials from config.

        Returns:
            True if login successful, False otherwise
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            # Navigate to login page
            driver.get("https://www.linkedin.com/login")
            time.sleep(2)

            # Wait for and fill username field
            username_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            username_field.clear()
            username_field.send_keys(Config.LINKEDIN_USERNAME)

            # Fill password field
            password_field = driver.find_element(By.ID, "password")
            password_field.clear()
            password_field.send_keys(Config.LINKEDIN_PASSWORD)

            # Click sign in button
            sign_in_button = driver.find_element(
                By.CSS_SELECTOR, "button[type='submit']"
            )
            sign_in_button.click()

            # Wait for redirect (successful login redirects away from /login)
            time.sleep(5)

            current_url = driver.current_url

            # Check for security checkpoint (2FA, CAPTCHA, etc.)
            if "/checkpoint" in current_url:
                print("\n" + "=" * 60)
                print("LinkedIn Security Checkpoint Detected")
                print("=" * 60)
                print("Please complete the security verification in the browser.")
                print("\nPress Enter after completing verification...")
                input()
                time.sleep(2)
                current_url = driver.current_url

            # Success if we're no longer on login/authwall page
            if "/login" not in current_url and "/authwall" not in current_url:
                return True

            return False

        except Exception as e:
            logger.error(f"Auto-login failed: {e}")
            return False

    def close_browser(self):
        """Close the shared browser session.

        Call this when done with all searches to clean up resources.
        """
        if self._uc_driver is not None:
            try:
                self._uc_driver.quit()
            except Exception:
                pass
            self._uc_driver = None
            logger.debug("Closed UC Chrome driver session")

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
            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_TEXT,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "max_output_tokens": 64,
                },
                endpoint="org_id_lookup",
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
            # Strategy 4: Try with common suffixes added (Inc, LLC, etc.)
            self._get_suffix_search(company_name),
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

        # Fallback: Try Playwright/Bing search (most reliable)
        logger.info(f"Trying Playwright/Bing search for: {company_name}")
        url = self._search_company_playwright(company_name)
        if url:
            is_valid, slug = self.validate_linkedin_url(url)
            if is_valid:
                return (url, slug)

        # Try Playwright search with acronym if we can generate one
        acronym = self._generate_acronym(company_name)
        if acronym and acronym.upper() != company_name.upper():
            logger.info(f"Trying Playwright/Bing search with acronym: {acronym}")
            url = self._search_company_playwright(acronym)
            if url:
                is_valid, slug = self.validate_linkedin_url(url)
                if is_valid:
                    return (url, slug)

        logger.info(f"No LinkedIn company page found for: {company_name}")
        return (None, None)

    def _generate_acronym(self, company_name: str) -> Optional[str]:
        """Generate an acronym from a multi-word company name."""
        # Known acronyms
        known_acronyms = {
            "Massachusetts Institute of Technology": "MIT",
            "University of California, Santa Barbara": "UCSB",
            "University of California, Los Angeles": "UCLA",
            "University of California, Berkeley": "UC Berkeley",
            "California Institute of Technology": "Caltech",
            "Georgia Institute of Technology": "Georgia Tech",
            "International Union of Pure and Applied Chemistry": "IUPAC",
        }

        if company_name in known_acronyms:
            return known_acronyms[company_name]

        # Generate acronym from multi-word names
        words = company_name.split()
        skip_words = {"of", "the", "and", "for", "in", "on", "at", "to", "a", "an"}
        significant_words = [w for w in words if w.lower() not in skip_words]

        if len(significant_words) >= 3:
            acronym = "".join(w[0].upper() for w in significant_words if w)
            if len(acronym) >= 3:
                return acronym

        return None

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

    def _get_suffix_search(self, company_name: str) -> Optional[str]:
        """Generate search queries by adding common company suffixes.

        Many companies have LinkedIn pages with suffixes like Inc, LLC, etc.
        that don't appear in how they're commonly referred to.
        """
        company_lower = company_name.lower()

        # Check if name already has a suffix
        existing_suffixes = [
            "inc",
            "llc",
            "ltd",
            "corp",
            "corporation",
            "limited",
            "gmbh",
            "plc",
        ]
        has_suffix = any(
            company_lower.endswith(f" {s}") or company_lower.endswith(f", {s}")
            for s in existing_suffixes
        )

        if has_suffix:
            return None

        # Try adding common suffixes - Inc is most common
        return (
            f'"{company_name} Inc" OR "{company_name}, Inc" site:linkedin.com/company'
        )

    def _get_acronym_search(self, company_name: str) -> Optional[str]:
        """Generate an acronym-based search if company name is multi-word."""
        acronym = self._generate_acronym(company_name)
        if acronym:
            return f'"{acronym}" site:linkedin.com/company OR site:linkedin.com/school'
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
            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_TEXT,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "max_output_tokens": 256,
                },
                endpoint="company_search",
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
        Search for an individual person's LinkedIn profile.

        Uses Playwright with Bing search as the primary method (most reliable),
        with a two-pass strategy: first requiring org match, then without.

        Args:
            name: Person's name (e.g., "Suzanne Farid")
            company: Company or organization they work at (e.g., "UCL")
            position: Optional job title to help identify the right person

        Returns:
            LinkedIn profile URL if found, None otherwise
        """
        if not name or not company:
            return None

        logger.info(f"Searching for person LinkedIn profile: {name} at {company}")

        # Primary method: Playwright/Bing search with org matching
        profile_url = self._search_person_playwright(
            name, location=None, company=company, require_org_match=True
        )

        # Fallback: Retry without org matching if needed
        if not profile_url:
            logger.info(f"Retrying search for {name} without org matching...")
            profile_url = self._search_person_playwright(
                name, location=None, company=company, require_org_match=False
            )

        if profile_url:
            logger.info(f"Found person profile: {name} -> {profile_url}")
            return profile_url

        # Fallback to Gemini search (rarely works but worth trying)
        return self._search_person_gemini(name, company, position)

    def _search_person_gemini(
        self, name: str, company: str, position: Optional[str] = None
    ) -> Optional[str]:
        """
        Fallback: Search for a person's LinkedIn profile using Gemini with Google Search.

        Note: This method is less reliable than Playwright/Bing search.
        """
        if not self.client:
            return None

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
            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_TEXT,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "max_output_tokens": 256,
                },
                endpoint="person_search",
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

    def _search_company_playwright(self, company_name: str) -> Optional[str]:
        """
        Search for LinkedIn company page using undetected-chromedriver to render Bing search results.

        Uses shared browser session for efficiency across multiple searches.

        Args:
            company_name: Company name to search for

        Returns:
            LinkedIn company/school URL if found, None otherwise
        """
        driver = self._get_uc_driver()
        if driver is None:
            return None

        from selenium.webdriver.common.by import By

        # Build search query
        search_query = f"{company_name} LinkedIn company"
        logger.debug(f"UC Chrome Bing company search: {search_query}")

        try:
            url = f"https://www.bing.com/search?q={search_query.replace(' ', '+')}"
            driver.get(url)
            time.sleep(2)  # Wait for page to load

            # Prepare company name parts for matching
            company_lower = company_name.lower()
            company_words = [w for w in company_lower.split() if len(w) > 2]

            result_items = driver.find_elements(By.CSS_SELECTOR, ".b_algo")
            for item in result_items:
                try:
                    heading = item.find_element(By.CSS_SELECTOR, "h2")
                    title = heading.text.lower()
                    link = heading.find_element(By.CSS_SELECTOR, "a")
                    href = link.get_attribute("href") or ""

                    # Decode Bing redirect URL
                    u_match = re.search(r"[&?]u=a1([^&]+)", href)
                    if not u_match:
                        continue

                    try:
                        encoded = u_match.group(1)
                        padding = 4 - len(encoded) % 4
                        if padding != 4:
                            encoded += "=" * padding
                        decoded_url = base64.urlsafe_b64decode(encoded).decode("utf-8")
                    except Exception:
                        continue

                    # Check if it's a LinkedIn company or school page
                    if (
                        "linkedin.com/company/" not in decoded_url
                        and "linkedin.com/school/" not in decoded_url
                    ):
                        continue

                    # Verify company name appears in title or URL
                    url_lower = decoded_url.lower()
                    matches = sum(
                        1
                        for word in company_words
                        if word in title or word in url_lower
                    )
                    if matches < len(company_words) * 0.5:
                        logger.debug(f"Skipping '{title[:40]}' - name mismatch")
                        continue

                    # Extract and return the URL
                    match = re.search(
                        r"linkedin\.com/(company|school)/([\w\-]+)", decoded_url
                    )
                    if match:
                        page_type = match.group(1)
                        slug = match.group(2)
                        result_url = f"https://www.linkedin.com/{page_type}/{slug}"
                        logger.info(f"Found company via UC Chrome: {result_url}")
                        return result_url

                except Exception as e:
                    logger.debug(f"Error processing result item: {e}")
                    continue

            logger.debug("No matching LinkedIn company found via UC Chrome")

        except Exception as e:
            logger.error(f"UC Chrome company search error: {e}")
            # If browser crashed, reset it
            self._uc_driver = None

        return None

    def _search_person_playwright(
        self,
        name: str,
        location: Optional[str] = None,
        company: Optional[str] = None,
        require_org_match: bool = True,
    ) -> Optional[str]:
        """
        Search for LinkedIn profile using undetected-chromedriver to render Bing search results.

        Uses shared browser session for efficiency across multiple searches.

        Args:
            name: Person's name
            location: Optional location
            company: Optional company/organisation
            require_org_match: If True, only return results that match the org.
                             If False, return first result matching the name.

        Returns:
            LinkedIn profile URL if found, None otherwise
        """
        driver = self._get_uc_driver()
        if driver is None:
            return None

        from selenium.webdriver.common.by import By

        # Build search query
        parts = [name]
        if location:
            parts.append(location)
        if company:
            parts.append(company)
        parts.append("LinkedIn")
        search_query = " ".join(parts)
        logger.debug(f"UC Chrome Bing search: {search_query}")

        try:
            url = f"https://www.bing.com/search?q={search_query.replace(' ', '+')}"
            driver.get(url)
            time.sleep(2)  # Wait for page to load

            # Parse name for matching (first and last, ignoring middle names)
            name_parts = name.lower().split()
            first_name = name_parts[0] if name_parts else ""
            last_name = name_parts[-1] if len(name_parts) >= 2 else ""

            result_items = driver.find_elements(By.CSS_SELECTOR, ".b_algo")
            for item in result_items:
                try:
                    heading = item.find_element(By.CSS_SELECTOR, "h2")
                    title = heading.text
                    link = heading.find_element(By.CSS_SELECTOR, "a")
                    href = link.get_attribute("href") or ""

                    # Decode Bing redirect URL (base64 encoded)
                    u_match = re.search(r"[&?]u=a1([^&]+)", href)
                    if not u_match:
                        continue

                    try:
                        encoded = u_match.group(1)
                        padding = 4 - len(encoded) % 4
                        if padding != 4:
                            encoded += "=" * padding
                        decoded_url = base64.urlsafe_b64decode(encoded).decode("utf-8")
                    except Exception:
                        continue

                    if "linkedin.com/in/" not in decoded_url:
                        continue

                    # Get result text for matching
                    try:
                        snippet_el = item.find_element(By.CSS_SELECTOR, ".b_caption p")
                        snippet = snippet_el.text
                    except Exception:
                        snippet = ""
                    result_text = f"{title} {snippet}".lower()

                    # Check name match (required) - first and last name
                    if first_name and last_name:
                        if (
                            first_name not in result_text
                            or last_name not in result_text
                        ):
                            logger.debug(f"Skipping '{title[:40]}' - name mismatch")
                            continue

                    # Check org match (optional based on require_org_match)
                    if company and require_org_match:
                        company_lower = company.lower()
                        company_words = [w for w in company_lower.split() if len(w) > 3]
                        org_match = any(word in result_text for word in company_words)
                        org_match = org_match or company_lower in result_text

                        if not org_match:
                            logger.debug(f"Skipping '{title[:40]}' - org mismatch")
                            continue

                    # Extract vanity name and return
                    vanity_match = re.search(r"linkedin\.com/in/([\w\-]+)", decoded_url)
                    if vanity_match:
                        vanity = vanity_match.group(1)
                        result_url = f"https://www.linkedin.com/in/{vanity}"
                        logger.info(f"Found profile via UC Chrome: {result_url}")
                        return result_url

                except Exception as e:
                    logger.debug(f"Error processing result item: {e}")
                    continue

            logger.debug("No matching LinkedIn profile found via UC Chrome")

        except Exception as e:
            logger.error(f"UC Chrome search error: {e}")
            # If browser crashed, reset it
            self._uc_driver = None

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
            response = api_client.gemini_generate(
                client=self.client,
                model=Config.MODEL_TEXT,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "max_output_tokens": 256,
                },
                endpoint="department_search",
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
                    logger.info(
                        f"Level 1: Searching for personal profile: {name} at {company}"
                    )
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
                        department_cache[dept_cache_key] = (
                            dept_url,
                            dept_slug,
                            dept_urn,
                        )
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
                logger.info(
                    f"Level 3: Using organization fallback for {name}: {company}"
                )
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
                logger.warning(
                    f"Level 4: No LinkedIn profile found for {name} at {company}"
                )
                person["linkedin_profile"] = None
                person["linkedin_slug"] = None
                person["linkedin_urn"] = None
                person["linkedin_profile_type"] = None

        # Return company_data with just (url, slug) for backwards compatibility
        return_data = {k: (v[0], v[1]) for k, v in company_data.items()}
        return (companies_found, companies_not_found, return_data)

    def lookup_person_urn(self, profile_url: str) -> Optional[str]:
        """
        Look up a person's URN from their LinkedIn profile URL.

        LinkedIn embeds the member URN in the profile page source code.
        This method loads the profile page and extracts the URN.

        Args:
            profile_url: LinkedIn profile URL (e.g., https://www.linkedin.com/in/username)

        Returns:
            Person URN (e.g., "urn:li:person:ABC123XYZ") if found, None otherwise
        """
        if not profile_url or "linkedin.com/in/" not in profile_url:
            return None

        driver = self._get_uc_driver()
        if driver is None:
            logger.warning("Cannot lookup person URN - browser driver not available")
            return None

        # Ensure user is logged in to LinkedIn (required to see URNs)
        if not self._ensure_linkedin_login(driver):
            logger.warning("LinkedIn login required for URN extraction")
            return None

        logger.debug(f"Looking up person URN for: {profile_url}")

        try:
            driver.get(profile_url)
            time.sleep(4)  # Wait for page to fully load

            # Extract the vanity name from URL to help identify the correct profile
            vanity_match = re.search(r"linkedin\.com/in/([\w\-]+)", profile_url)
            vanity_name = vanity_match.group(1) if vanity_match else None

            # LinkedIn embeds the member URN in several places in the page source
            # We need to find the URN that belongs to the PROFILE OWNER, not the viewer

            page_source = driver.page_source

            if vanity_name:
                # BEST PATTERN: Look for memberRelationship URN after the target's publicIdentifier
                # Format: "publicIdentifier":"username",...,"*memberRelationship":"urn:li:fsd_memberRelationship:ID"
                # Use [\s\S] instead of [^{}] to handle nested braces in JSON
                member_rel_pattern = rf'"publicIdentifier":"{re.escape(vanity_name)}"[\s\S]{{0,500}}?"\*memberRelationship":"urn:li:fsd_memberRelationship:([A-Za-z0-9_-]+)"'
                member_match = re.search(member_rel_pattern, page_source)
                if member_match:
                    profile_id = member_match.group(1)
                    urn = f"urn:li:person:{profile_id}"
                    logger.info(f"Found person URN (memberRelationship): {urn}")
                    return urn

                # Alternative: Look for fsd_profile after publicIdentifier
                profile_block_pattern = rf'"publicIdentifier":"{re.escape(vanity_name)}"[\s\S]{{0,500}}?"entityUrn":"urn:li:fsd_profile:([A-Za-z0-9_-]+)"'
                profile_match = re.search(profile_block_pattern, page_source)
                if profile_match:
                    profile_id = profile_match.group(1)
                    urn = f"urn:li:person:{profile_id}"
                    logger.info(f"Found person URN (matched publicIdentifier): {urn}")
                    return urn

                # Alternative: Look for miniProfile with matching publicIdentifier
                mini_pattern = rf'"publicIdentifier":"{re.escape(vanity_name)}"[\s\S]{{0,500}}?"urn":"urn:li:fs_miniProfile:([A-Za-z0-9_-]+)"'
                mini_match = re.search(mini_pattern, page_source)
                if mini_match:
                    profile_id = mini_match.group(1)
                    urn = f"urn:li:person:{profile_id}"
                    logger.info(f"Found person URN (miniProfile with publicId): {urn}")
                    return urn

            # Fallback: Look for the first fsd_profile in a profile context
            # Search specifically in profile-related JSON blocks
            profile_data_match = re.search(
                r'"Profile"[^}]*"entityUrn":"(urn:li:fsd_profile:[A-Za-z0-9_-]+)"',
                page_source,
            )
            if profile_data_match:
                profile_id = profile_data_match.group(1).split(":")[-1]
                urn = f"urn:li:person:{profile_id}"
                logger.info(f"Found person URN (Profile entity): {urn}")
                return urn

            logger.warning(f"Could not extract URN from profile: {profile_url}")
            return None

        except Exception as e:
            logger.error(f"Error looking up person URN for {profile_url}: {e}")
            # Reset browser if it crashed
            self._uc_driver = None
            return None

    def close(self) -> None:
        """Clean up HTTP client and browser resources."""
        if self._http_client:
            self._http_client.close()
            self._http_client = None
        self.close_browser()


# Backwards compatibility alias
LinkedInProfileLookup = LinkedInCompanyLookup


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for linkedin_profile_lookup module."""
    from test_framework import TestSuite

    suite = TestSuite("LinkedIn Profile Lookup Tests")

    def test_validate_linkedin_url_valid_company():
        lookup = LinkedInCompanyLookup(genai_client=None)
        valid, slug = lookup.validate_linkedin_url(
            "https://www.linkedin.com/company/google"
        )
        assert valid is True
        assert slug == "google"
        lookup.close()

    def test_validate_linkedin_url_valid_school():
        lookup = LinkedInCompanyLookup(genai_client=None)
        valid, slug = lookup.validate_linkedin_url(
            "https://www.linkedin.com/school/stanford-university"
        )
        assert valid is True
        assert slug == "stanford-university"
        lookup.close()

    def test_validate_linkedin_url_invalid():
        lookup = LinkedInCompanyLookup(genai_client=None)
        valid, slug = lookup.validate_linkedin_url("https://example.com/not-linkedin")
        assert valid is False
        assert slug is None
        lookup.close()

    def test_validate_linkedin_url_empty():
        lookup = LinkedInCompanyLookup(genai_client=None)
        valid, slug = lookup.validate_linkedin_url("")
        assert valid is False
        assert slug is None
        lookup.close()

    def test_generate_acronym_long():
        lookup = LinkedInCompanyLookup(genai_client=None)
        # Acronym for 3+ word phrases
        acronym = lookup._generate_acronym(
            "International Business Machines Corporation"
        )
        assert acronym == "IBMC"
        lookup.close()

    def test_generate_acronym_short():
        lookup = LinkedInCompanyLookup(genai_client=None)
        # Short names don't generate acronyms
        acronym = lookup._generate_acronym("MIT")
        assert acronym is None
        lookup.close()

    def test_extract_company_url():
        lookup = LinkedInCompanyLookup(genai_client=None)
        text = "Check out https://www.linkedin.com/company/acme-corp for more info"
        url = lookup._extract_company_url(text)
        assert url == "https://www.linkedin.com/company/acme-corp"
        lookup.close()

    def test_extract_person_url():
        lookup = LinkedInCompanyLookup(genai_client=None)
        text = "Visit https://www.linkedin.com/in/jane-doe-12345 for profile"
        url = lookup._extract_person_url(text)
        assert url is not None
        assert "jane-doe" in url
        lookup.close()

    def test_lookup_class_init():
        lookup = LinkedInCompanyLookup(genai_client=None)
        # When no genai_client passed and no API key, client is None
        assert lookup._uc_driver is None
        assert lookup._http_client is None
        lookup.close()

    def test_context_manager():
        with LinkedInCompanyLookup(genai_client=None) as lookup:
            assert lookup is not None
        # Should close without error

    suite.add_test(
        "Validate LinkedIn URL - valid company",
        test_validate_linkedin_url_valid_company,
    )
    suite.add_test(
        "Validate LinkedIn URL - valid school", test_validate_linkedin_url_valid_school
    )
    suite.add_test(
        "Validate LinkedIn URL - invalid", test_validate_linkedin_url_invalid
    )
    suite.add_test("Validate LinkedIn URL - empty", test_validate_linkedin_url_empty)
    suite.add_test("Generate acronym - long phrase", test_generate_acronym_long)
    suite.add_test("Generate acronym - short name", test_generate_acronym_short)
    suite.add_test("Extract company URL from text", test_extract_company_url)
    suite.add_test("Extract person URL from text", test_extract_person_url)
    suite.add_test("Lookup class initialization", test_lookup_class_init)
    suite.add_test("Context manager works", test_context_manager)

    return suite
