"""LinkedIn company and person profile lookup using Gemini with Google Search grounding and undetected-chromedriver."""

import base64
import concurrent.futures
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, cast

import httpx
from google import genai  # type: ignore

from api_client import api_client
from config import Config
from error_handling import with_enhanced_recovery, NetworkTimeoutError
from rate_limiter import AdaptiveRateLimiter

logger = logging.getLogger(__name__)

# Import undetected-chromedriver for CAPTCHA-resistant browser automation
# Type stubs for optional dependencies
uc: Any = None
UC_AVAILABLE = False

try:
    import undetected_chromedriver as _uc

    uc = _uc
    UC_AVAILABLE = True
except ImportError:
    logger.warning(
        "undetected-chromedriver not installed - pip install undetected-chromedriver"
    )


def _suppress_uc_cleanup_errors() -> None:
    """Suppress Windows handle errors from UC Chrome cleanup."""
    if sys.platform == "win32":
        # Monkey-patch time.sleep in the UC module to suppress cleanup errors
        original_sleep = time.sleep

        def patched_sleep(seconds: float) -> None:
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

    def __init__(self, genai_client: Optional[genai.Client] = None) -> None:
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

        # Cache person search results across multiple stories
        # Key: "name@company|department|location" -> LinkedIn URL or None (not found)
        self._person_cache: dict[str, Optional[str]] = {}

        # Cache company search results across multiple stories
        # Key: company name -> (url, slug, urn) or None
        self._company_cache: dict[
            str, Optional[tuple[str, Optional[str], Optional[str]]]
        ] = {}

        # Cache department search results across multiple stories
        # Key: "dept@company" -> (url, slug, urn) or None
        self._department_cache: dict[
            str, tuple[Optional[str], Optional[str], Optional[str]]
        ] = {}

        # Track Gemini fallback success rate - disable after too many failures
        self._gemini_attempts = 0
        self._gemini_successes = 0
        self._gemini_disabled = False

        # Timing metrics for search operations
        self._timing_stats: dict[str, list[float]] = {
            "person_search": [],
            "company_search": [],
            "department_search": [],
        }

        # Cache persistence file path
        self._cache_dir = Path(
            os.path.expandvars(r"%LOCALAPPDATA%\SocialMediaPublisher")
        )
        self._cache_file = self._cache_dir / "linkedin_cache.json"

        # Load persisted cache on startup
        self._load_cache_from_disk()

    def get_cache_stats(self) -> dict[str, dict[str, int]]:
        """Get statistics about all search caches.

        Returns:
            Dict with stats for 'person', 'company', 'department' caches
        """
        person_total = len(self._person_cache)
        person_found = sum(1 for url in self._person_cache.values() if url is not None)

        company_total = len(self._company_cache)
        company_found = sum(1 for v in self._company_cache.values() if v is not None)

        dept_total = len(self._department_cache)
        dept_found = sum(1 for v in self._department_cache.values() if v[0] is not None)

        return {
            "person": {
                "total": person_total,
                "found": person_found,
                "not_found": person_total - person_found,
            },
            "company": {
                "total": company_total,
                "found": company_found,
                "not_found": company_total - company_found,
            },
            "department": {
                "total": dept_total,
                "found": dept_found,
                "not_found": dept_total - dept_found,
            },
        }

    def get_timing_stats(self) -> dict[str, dict[str, float]]:
        """Get timing statistics for search operations.

        Returns:
            Dict with avg/min/max/total times in seconds for each operation type
        """
        result = {}
        for op_type, times in self._timing_stats.items():
            if times:
                result[op_type] = {
                    "count": len(times),
                    "total": sum(times),
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }
            else:
                result[op_type] = {"count": 0, "total": 0, "avg": 0, "min": 0, "max": 0}
        return result

    def get_gemini_stats(self) -> dict[str, Any]:
        """Get Gemini fallback statistics.

        Returns:
            Dict with attempts, successes, success_rate, and disabled status
        """
        rate = (
            self._gemini_successes / self._gemini_attempts * 100
            if self._gemini_attempts > 0
            else 0
        )
        return {
            "attempts": self._gemini_attempts,
            "successes": self._gemini_successes,
            "success_rate": round(rate, 1),
            "disabled": self._gemini_disabled,
        }

    def _load_cache_from_disk(self) -> None:
        """Load persisted cache from disk if available."""
        if not self._cache_file.exists():
            return

        try:
            with open(self._cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Restore person cache
            self._person_cache = data.get("person", {})

            # Restore company cache (convert lists back to tuples)
            for key, val in data.get("company", {}).items():
                if val is None:
                    self._company_cache[key] = None
                else:
                    self._company_cache[key] = tuple(val)  # type: ignore

            # Restore department cache (convert lists back to tuples)
            for key, val in data.get("department", {}).items():
                self._department_cache[key] = tuple(val)  # type: ignore

            total = (
                len(self._person_cache)
                + len(self._company_cache)
                + len(self._department_cache)
            )
            logger.info(f"Loaded {total} cached entries from {self._cache_file}")

        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")

    def save_cache_to_disk(self) -> None:
        """Persist cache to disk for future runs."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # Convert tuples to lists for JSON serialization
            data = {
                "person": self._person_cache,
                "company": {
                    k: list(v) if v else None for k, v in self._company_cache.items()
                },
                "department": {k: list(v) for k, v in self._department_cache.items()},
            }

            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            total = (
                len(self._person_cache)
                + len(self._company_cache)
                + len(self._department_cache)
            )
            logger.info(f"Saved {total} cache entries to {self._cache_file}")

        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")

    def get_person_cache_stats(self) -> dict[str, int]:
        """Get statistics about the person search cache.

        Returns:
            Dict with 'total', 'found' (with URLs), 'not_found' (None values)
        """
        return self.get_cache_stats()["person"]

    def clear_all_caches(self) -> dict[str, int]:
        """Clear all search caches.

        Returns:
            Dict with count of entries cleared per cache type
        """
        counts = {
            "person": len(self._person_cache),
            "company": len(self._company_cache),
            "department": len(self._department_cache),
        }
        self._person_cache.clear()
        self._company_cache.clear()
        self._department_cache.clear()
        logger.info(f"Cleared all caches: {counts}")
        return counts

    def clear_person_cache(self) -> int:
        """Clear the person search cache.

        Returns:
            Number of entries cleared
        """
        count = len(self._person_cache)
        self._person_cache.clear()
        logger.info(f"Cleared person cache ({count} entries)")
        return count

    def __enter__(self) -> "LinkedInCompanyLookup":
        """Context manager entry - returns self."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
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
            if self._uc_driver is not None:
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

    def close_browser(self) -> None:
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

    def validate_linkedin_org_url(self, url: str) -> tuple[bool, Optional[str]]:
        """
        Validate that a LinkedIn company/school URL is well-formed and extract the slug.

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

        start_time = time.time()
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
                elapsed = time.time() - start_time
                self._timing_stats["company_search"].append(elapsed)
                return (url, urn)

            # Small delay between retries
            if i < len(search_strategies) - 1:
                time.sleep(0.5)

        # Fallback: Try Playwright/Bing search (most reliable)
        logger.info(f"Trying Playwright/Bing search for: {company_name}")
        url = self._search_company_playwright(company_name)
        if url:
            is_valid, slug = self.validate_linkedin_org_url(url)
            if is_valid:
                elapsed = time.time() - start_time
                self._timing_stats["company_search"].append(elapsed)
                return (url, slug)

        # Try Playwright search with acronym if we can generate one
        acronym = self._generate_acronym(company_name)
        if acronym and acronym.upper() != company_name.upper():
            logger.info(f"Trying Playwright/Bing search with acronym: {acronym}")
            url = self._search_company_playwright(acronym)
            if url:
                is_valid, slug = self.validate_linkedin_org_url(url)
                if is_valid:
                    elapsed = time.time() - start_time
                    self._timing_stats["company_search"].append(elapsed)
                    return (url, slug)

        elapsed = time.time() - start_time
        self._timing_stats["company_search"].append(elapsed)
        logger.info(
            f"No LinkedIn company page found for: {company_name} ({elapsed:.1f}s)"
        )
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
                is_valid, org_urn = self.validate_linkedin_org_url(linkedin_url)
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
        self,
        name: str,
        company: str,
        position: Optional[str] = None,
        department: Optional[str] = None,
        location: Optional[str] = None,
        role_type: Optional[str] = None,
        research_area: Optional[str] = None,
    ) -> Optional[str]:
        """
        Search for an individual person's LinkedIn profile.

        Uses Playwright with Bing search as the primary method (most reliable),
        with a two-pass strategy: first requiring org match, then without.

        Args:
            name: Person's name (e.g., "Suzanne Farid")
            company: Company or organization they work at (e.g., "UCL")
            position: Optional job title to help identify the right person
            department: Optional department name (e.g., "Chemical Engineering")
            location: Optional location (e.g., "Cambridge, MA, USA")
            role_type: Optional role type (academic, executive, researcher, etc.)
            research_area: Optional research field for academics

        Returns:
            LinkedIn profile URL if found, None otherwise
        """
        if not name or not company:
            return None

        start_time = time.time()

        # Build enhanced context for logging
        context_parts = [f"{name} at {company}"]
        if position:
            context_parts.append(f"({position})")
        if department:
            context_parts.append(f"in {department}")
        if location:
            context_parts.append(f"from {location}")
        logger.info(f"Searching for person LinkedIn profile: {' '.join(context_parts)}")

        # Primary method: Playwright/Bing search with org matching
        profile_url = self._search_person_playwright(
            name,
            location=location,
            company=company,
            position=position,
            department=department,
            role_type=role_type,
            research_area=research_area,
            require_org_match=True,
        )

        # Fallback: Retry without org matching if needed
        if not profile_url:
            logger.info(f"Retrying search for {name} without org matching...")
            profile_url = self._search_person_playwright(
                name,
                location=location,
                company=company,
                position=position,
                department=department,
                role_type=role_type,
                research_area=research_area,
                require_org_match=False,
            )

        # Fallback: Try with just name + company (no department/location that might be wrong)
        if not profile_url and (department or location):
            logger.info(f"Retrying search for {name} with simplified query...")
            profile_url = self._search_person_playwright(
                name,
                company=company,
                require_org_match=False,
            )

        if profile_url:
            elapsed = time.time() - start_time
            self._timing_stats["person_search"].append(elapsed)
            logger.info(
                f"Found person profile: {name} -> {profile_url} ({elapsed:.1f}s)"
            )
            return profile_url

        # Fallback to Gemini search (skip if disabled due to low success rate)
        if self._gemini_disabled:
            elapsed = time.time() - start_time
            self._timing_stats["person_search"].append(elapsed)
            logger.debug(f"Skipping Gemini fallback (disabled) for {name}")
            return None

        result = self._search_person_gemini(
            name, company, position, department, location, role_type, research_area
        )

        elapsed = time.time() - start_time
        self._timing_stats["person_search"].append(elapsed)
        return result

    def _search_person_gemini(
        self,
        name: str,
        company: str,
        position: Optional[str] = None,
        department: Optional[str] = None,
        location: Optional[str] = None,
        role_type: Optional[str] = None,
        research_area: Optional[str] = None,
    ) -> Optional[str]:
        """
        Fallback: Search for a person's LinkedIn profile using Gemini with Google Search.

        Note: This method is less reliable than Playwright/Bing search.
        Automatically disables itself after 10+ attempts with <10% success rate.
        """
        self._gemini_attempts += 1

        if not self.client:
            return None

        # Build rich context for better matching
        context_lines = [f"Name: {name}", f"Company/Organization: {company}"]

        if position:
            context_lines.append(f"Position/Title: {position}")
        if department:
            context_lines.append(f"Department: {department}")
        if location:
            context_lines.append(f"Location: {location}")
        if role_type:
            context_lines.append(f"Role Type: {role_type}")
        if research_area:
            context_lines.append(f"Research Area/Field: {research_area}")

        person_context = "\n".join(context_lines)

        # Build role-specific matching guidance
        matching_guidance = ""
        if role_type == "academic":
            matching_guidance = """
MATCHING TIPS FOR ACADEMIC PROFILE:
- Look for university affiliations and department matches
- Research area/field should align with their expertise
- Academic titles: Professor, Dr., Researcher, Postdoc, PhD
- May list publications or research interests"""
        elif role_type == "executive":
            matching_guidance = """
MATCHING TIPS FOR EXECUTIVE PROFILE:
- Look for C-suite titles or VP/Director positions
- Industry experience should match the company sector
- Senior leadership roles at the mentioned organization"""
        elif role_type == "researcher":
            matching_guidance = """
MATCHING TIPS FOR RESEARCHER PROFILE:
- Look for research-focused titles and affiliations
- Research area should match their published work
- May be affiliated with universities, labs, or R&D divisions"""

        prompt = f"""Find the LinkedIn profile for this specific person:

{person_context}

TASK: Search for the LinkedIn personal profile page for this individual.

CRITICAL MATCHING RULES:
1. Find a linkedin.com/in/username profile URL for THIS SPECIFIC PERSON
2. The person MUST work at or be affiliated with {company}
3. ALL available context (position, department, location) should match
4. Be VERY careful with common names - require multiple matching attributes
5. Do NOT return company pages (linkedin.com/company/...)
6. Do NOT return school pages (linkedin.com/school/...)
7. Only return a profile if you're HIGHLY CONFIDENT it's the right person
{matching_guidance}

CONTRADICTION DETECTION - REJECT if ANY of these are true:
- Profile shows a completely DIFFERENT field of work (e.g., real estate agent when expecting researcher)
- Profile shows a conflicting location (e.g., India when expecting USA, unless recent move indicated)
- Profile shows unrelated industry (e.g., hospitality, retail when expecting engineering/science)
- Profile title/role is fundamentally incompatible (e.g., "Marketing Manager" when expecting "Professor")
- Same name but clearly different person (different photo context, different career entirely)

VERIFICATION CHECKLIST:
- Name matches (including possible variations like Dr., Prof.)
- Organization/company affiliation matches
- Job title/position is consistent with the context
- Location is consistent (if provided)
- Department or field aligns (if provided)
- NO contradictory information present

RESPONSE FORMAT:
If you find their LinkedIn profile with HIGH CONFIDENCE, respond with ONLY the URL like:
https://www.linkedin.com/in/username

If you cannot find their personal LinkedIn profile OR are uncertain OR found contradictory information, respond exactly:
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
                self._check_gemini_disable()
                return None

            # Extract LinkedIn personal profile URL
            linkedin_url = self._extract_person_url(result)
            if linkedin_url:
                self._gemini_successes += 1
                logger.info(
                    f"Found person profile via Gemini: {name} -> {linkedin_url}"
                )
                return linkedin_url

            self._check_gemini_disable()
            return None

        except Exception as e:
            logger.error(f"Error searching for person {name}: {e}")
            self._check_gemini_disable()
            return None

    def _check_gemini_disable(self) -> None:
        """Check if Gemini fallback should be disabled due to low success rate."""
        if self._gemini_disabled:
            return
        # Only evaluate after 10+ attempts
        if self._gemini_attempts >= 10:
            success_rate = self._gemini_successes / self._gemini_attempts
            if success_rate < 0.10:  # Less than 10% success
                self._gemini_disabled = True
                logger.info(
                    f"Disabling Gemini fallback: {self._gemini_successes}/{self._gemini_attempts} "
                    f"({success_rate * 100:.0f}%) success rate"
                )

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

    def _validate_profile_name(
        self, driver: Any, profile_url: str, first_name: str, last_name: str
    ) -> bool:
        """Visit a LinkedIn profile page and validate that the name matches.

        This is the most reliable way to verify we have the correct person,
        as it reads the actual profile name from the page.

        Args:
            driver: Selenium WebDriver instance
            profile_url: LinkedIn profile URL to validate
            first_name: Expected first name (lowercase)
            last_name: Expected last name (lowercase)

        Returns:
            True if the profile name matches, False otherwise
        """
        from selenium.webdriver.common.by import By

        try:
            driver.get(profile_url)
            time.sleep(1.5)  # Wait for page to load

            # Extract the profile name from the page
            # LinkedIn profile titles are typically "FirstName LastName - Title | LinkedIn"
            page_title = driver.title.lower() if driver.title else ""

            # Also try to get the main profile name heading
            profile_name = ""
            for selector in [
                "h1",
                ".text-heading-xlarge",
                "[data-generated-suggestion-target]",
            ]:
                try:
                    name_elem = driver.find_element(By.CSS_SELECTOR, selector)
                    if name_elem and name_elem.text:
                        profile_name = name_elem.text.lower().strip()
                        break
                except Exception:
                    continue

            # Use profile name if found, otherwise fall back to page title
            name_text = profile_name if profile_name else page_title

            if not name_text:
                logger.debug(f"Could not extract name from profile page: {profile_url}")
                return False

            # Validate: BOTH first and last name must appear in the profile name
            # Use word boundary matching to avoid partial matches
            if first_name and last_name:
                first_pattern = rf"\b{re.escape(first_name)}\b"
                last_pattern = rf"\b{re.escape(last_name)}\b"
                first_match = bool(re.search(first_pattern, name_text))
                last_match = bool(re.search(last_pattern, name_text))

                if first_match and last_match:
                    logger.debug(
                        f"Profile name validated: '{name_text[:50]}' matches '{first_name} {last_name}'"
                    )
                    return True
                else:
                    logger.debug(
                        f"Profile name mismatch: expected '{first_name} {last_name}', found '{name_text[:50]}'"
                    )
                    return False
            elif first_name:
                first_pattern = rf"\b{re.escape(first_name)}\b"
                if re.search(first_pattern, name_text):
                    return True
                else:
                    logger.debug(
                        f"Profile name mismatch: expected '{first_name}', found '{name_text[:50]}'"
                    )
                    return False
            else:
                # No name parts to validate - accept
                return True

        except Exception as e:
            logger.debug(f"Error validating profile name for {profile_url}: {e}")
            return False

    def _search_person_playwright(
        self,
        name: str,
        location: Optional[str] = None,
        company: Optional[str] = None,
        position: Optional[str] = None,
        department: Optional[str] = None,
        role_type: Optional[str] = None,
        research_area: Optional[str] = None,
        require_org_match: bool = True,
    ) -> Optional[str]:
        """
        Search for LinkedIn profile using undetected-chromedriver to render Bing search results.

        Uses shared browser session for efficiency across multiple searches.

        Args:
            name: Person's name
            location: Optional location (city, state, country)
            company: Optional company/organisation
            position: Optional job title
            department: Optional department name
            role_type: Optional role type (academic, executive, researcher)
            research_area: Optional research field for academics
            require_org_match: If True, only return results that match the org.
                             If False, return first result matching the name.

        Returns:
            LinkedIn profile URL if found, None otherwise
        """
        driver = self._get_uc_driver()
        if driver is None:
            return None

        from selenium.webdriver.common.by import By

        # Build enhanced search query with available context
        parts = [name]
        if company:
            parts.append(company)
        # Add department for more specific searches (especially useful for academics)
        if department:
            parts.append(department)
        # Add location for disambiguation
        if location:
            # Extract just city/region for search (avoid full country name)
            location_parts = location.split(",")
            if location_parts:
                parts.append(location_parts[0].strip())
        # Use site: restriction for more targeted results
        parts.append("site:linkedin.com/in")
        search_query = " ".join(parts)
        logger.debug(f"UC Chrome Google search: {search_query}")

        # Common FULL names or extremely common first names that need extra signals
        # Note: We removed ethnic surnames (Wang, Zhang, Kim, Patel, etc.) because
        # when combined with first names they're usually distinctive enough.
        # Only flag the most generic Western first names that could match anyone.
        COMMON_FIRST_NAMES = {
            "john",
            "james",
            "michael",
            "david",
            "robert",
            "mary",
            "jennifer",
            "sarah",
            "smith",
            "johnson",
            "williams",
            "brown",
            "jones",
        }

        try:
            # Use Google Search (better LinkedIn results than Bing)
            import urllib.parse

            encoded_query = urllib.parse.quote(search_query)
            url = f"https://www.google.com/search?q={encoded_query}"
            driver.get(url)
            time.sleep(2)  # Wait for page to load

            # Parse name for matching (first and last, ignoring middle names and titles)
            name_clean = re.sub(r"^(dr\.?|prof\.?|professor)\s+", "", name.lower())
            name_parts = name_clean.split()
            first_name = name_parts[0] if name_parts else ""
            last_name = name_parts[-1] if len(name_parts) >= 2 else ""

            # Check if this is a very common Western name (requires slightly higher confidence)
            is_common_name = (
                first_name in COMMON_FIRST_NAMES or last_name in COMMON_FIRST_NAMES
            )

            # Build context matching keywords from all available metadata
            context_keywords = set()
            if company:
                context_keywords.update(
                    w.lower() for w in company.split() if len(w) > 2
                )
            if department:
                context_keywords.update(
                    w.lower() for w in department.split() if len(w) > 3
                )
            if position:
                # Add significant words from position
                context_keywords.update(
                    w.lower() for w in position.split() if len(w) > 4
                )
            if research_area:
                context_keywords.update(
                    w.lower() for w in research_area.split() if len(w) > 3
                )

            # Remove common stopwords from context keywords
            stopwords = {
                "the",
                "and",
                "for",
                "with",
                "from",
                "research",
                "center",
                "centre",
                "department",
            }
            context_keywords -= stopwords

            # Extract LinkedIn URLs directly from page source (most reliable for Google)
            page_source = driver.page_source
            linkedin_urls = re.findall(
                r"https://(?:www\.)?linkedin\.com/in/[\w\-]+/?", page_source
            )

            # Deduplicate while preserving order
            seen = set()
            unique_urls = []
            for u in linkedin_urls:
                u_clean = u.rstrip("/")
                if u_clean not in seen:
                    seen.add(u_clean)
                    unique_urls.append(u_clean)

            # Track all candidates with their scores
            candidates: List[
                Tuple[int, str, List[str]]
            ] = []  # (score, url, matched_keywords)

            # For each LinkedIn URL found, try to get context from the search result
            for linkedin_url in unique_urls[:10]:  # Check up to 10 results
                try:
                    # Find the search result containing this URL
                    # Google results are in divs with class 'g' or we can search for the URL in any element
                    result_text = ""

                    # Try to find result elements containing this URL
                    try:
                        # Look for any element containing this URL
                        elements = driver.find_elements(
                            By.XPATH, f"//*[contains(@href, '{linkedin_url}')]"
                        )
                        for elem in elements:
                            # Get parent container text
                            parent = elem
                            for _ in range(5):  # Go up to 5 levels
                                try:
                                    parent = parent.find_element(By.XPATH, "..")
                                    parent_text = parent.text
                                    if (
                                        len(parent_text) > len(result_text)
                                        and len(parent_text) < 1000
                                    ):
                                        result_text = parent_text
                                except Exception:
                                    break
                    except Exception:
                        pass

                    result_text = result_text.lower() if result_text else ""

                    # If no context found from search results, we can still use the URL
                    # The URL slug often contains the person's name
                    url_slug = (
                        linkedin_url.split("/in/")[-1] if "/in/" in linkedin_url else ""
                    )
                    url_text = url_slug.replace("-", " ").lower()

                    # === CRITICAL NAME VALIDATION ===
                    # The URL slug MUST contain the person's name to be valid
                    # This prevents matching completely wrong profiles
                    name_in_url = True
                    if first_name and last_name:
                        # Both first and last name should appear in URL slug
                        # Use word boundary matching to avoid partial matches
                        first_pattern = r"\b" + re.escape(first_name) + r"\b"
                        last_pattern = r"\b" + re.escape(last_name) + r"\b"
                        first_in_url = bool(re.search(first_pattern, url_text))
                        last_in_url = bool(re.search(last_pattern, url_text))

                        if not first_in_url and not last_in_url:
                            # Neither name part in URL - definitely wrong person
                            logger.debug(
                                f"Skipping '{linkedin_url}' - name not in URL slug: '{url_text}'"
                            )
                            continue
                        elif not (first_in_url and last_in_url):
                            # Only one part matches - could be partial match, flag it
                            name_in_url = False
                    elif first_name:
                        # Single word name - must be in URL
                        first_pattern = r"\b" + re.escape(first_name) + r"\b"
                        if not re.search(first_pattern, url_text):
                            logger.debug(
                                f"Skipping '{linkedin_url}' - name not in URL slug: '{url_text}'"
                            )
                            continue

                    # Secondary check: name in search result text (less reliable but helpful)
                    text_to_check = f"{result_text} {url_text}"

                    if first_name and last_name:
                        # If name wasn't fully in URL, it MUST be in result text
                        if not name_in_url:
                            if (
                                first_name not in result_text
                                or last_name not in result_text
                            ):
                                logger.debug(
                                    f"Skipping '{linkedin_url}' - partial URL match but name not in result text"
                                )
                                continue
                    elif first_name:
                        # Single word name - be more careful
                        if first_name not in text_to_check:
                            logger.debug(f"Skipping '{linkedin_url}' - name mismatch")
                            continue

                    # Calculate match score based on context keywords
                    match_score = 0
                    matched_keywords = []
                    for keyword in context_keywords:
                        if keyword in result_text:
                            match_score += 1
                            matched_keywords.append(keyword)

                    # Strong boost if full name is in URL slug (high confidence)
                    if name_in_url:
                        match_score += 3
                        matched_keywords.append("name_in_url")

                    # Check org match (optional based on require_org_match)
                    org_matched = False
                    if company:
                        company_lower = company.lower()
                        company_words = [w for w in company_lower.split() if len(w) > 3]
                        org_matched = any(word in result_text for word in company_words)
                        org_matched = org_matched or company_lower in result_text

                    if require_org_match and not org_matched:
                        logger.debug(f"Skipping '{linkedin_url}' - org mismatch")
                        continue

                    # Boost score for org match
                    if org_matched:
                        match_score += 2

                    # Boost score for location match
                    if location:
                        location_lower = location.lower()
                        location_parts = [p.strip() for p in location_lower.split(",")]
                        if any(
                            part in result_text
                            for part in location_parts
                            if len(part) > 2
                        ):
                            match_score += 1

                    # Boost score for role-type specific matches
                    if role_type == "academic":
                        academic_indicators = [
                            "professor",
                            "researcher",
                            "phd",
                            "dr.",
                            "university",
                        ]
                        if any(ind in result_text for ind in academic_indicators):
                            match_score += 1
                    elif role_type == "executive":
                        exec_indicators = [
                            "ceo",
                            "cto",
                            "cfo",
                            "vp",
                            "president",
                            "director",
                            "chief",
                        ]
                        if any(ind in result_text for ind in exec_indicators):
                            match_score += 1

                    # === CONTRADICTION DETECTION ===
                    # Penalize profiles with indicators of wrong person
                    contradiction_penalty = 0
                    contradiction_reasons = []

                    # Wrong field of work - unrelated professions
                    if role_type == "academic":
                        # Academic should not be primarily sales, marketing, real estate
                        wrong_field_indicators = [
                            "real estate agent",
                            "realtor",
                            "sales manager",
                            "marketing manager",
                            "insurance agent",
                            "financial advisor",
                            "fitness trainer",
                            "hair stylist",
                            "photographer",
                            "life coach",
                        ]
                        for wrong_field in wrong_field_indicators:
                            if wrong_field in result_text:
                                contradiction_penalty += 3
                                contradiction_reasons.append(
                                    f"wrong field: {wrong_field}"
                                )
                    elif role_type == "executive":
                        # Executive at company X should not show as professor
                        if "professor" in result_text and company:
                            company_lower = company.lower()
                            # If they're a professor but not at expected org
                            if not any(
                                word in result_text
                                for word in company_lower.split()
                                if len(word) > 3
                            ):
                                contradiction_penalty += 2
                                contradiction_reasons.append(
                                    "professor at different org"
                                )

                    # Wrong industry indicators
                    if department or research_area:
                        # Engineering/science person should not be primarily in unrelated fields
                        expected_terms = set()
                        if department:
                            expected_terms.update(
                                w.lower() for w in department.split() if len(w) > 4
                            )
                        if research_area:
                            expected_terms.update(
                                w.lower() for w in research_area.split() if len(w) > 4
                            )

                        # If we expect engineering/science but find completely different industry
                        science_expected = any(
                            term in expected_terms
                            for term in [
                                "chemical",
                                "engineering",
                                "biology",
                                "chemistry",
                                "physics",
                                "research",
                            ]
                        )
                        if science_expected:
                            unrelated_industries = [
                                "real estate",
                                "retail sales",
                                "hospitality",
                                "food service",
                                "beauty salon",
                            ]
                            for industry in unrelated_industries:
                                if industry in result_text:
                                    contradiction_penalty += 2
                                    contradiction_reasons.append(
                                        f"unrelated industry: {industry}"
                                    )

                    # Location contradiction - different country/continent
                    if location:
                        location_lower = location.lower()
                        # Extract expected country/region
                        location_parts = [p.strip() for p in location_lower.split(",")]
                        expected_country = location_parts[-1] if location_parts else ""

                        # Check for conflicting country indicators
                        country_conflicts = {
                            "usa": [
                                "india",
                                "china",
                                "brazil",
                                "indonesia",
                                "nigeria",
                                "pakistan",
                            ],
                            "uk": [
                                "india",
                                "china",
                                "brazil",
                                "indonesia",
                                "nigeria",
                                "pakistan",
                            ],
                            "canada": [
                                "india",
                                "china",
                                "brazil",
                                "indonesia",
                                "nigeria",
                            ],
                            "australia": ["india", "china", "brazil", "indonesia"],
                            "germany": ["india", "china", "brazil", "indonesia"],
                        }

                        for expected, conflicts in country_conflicts.items():
                            if expected in expected_country:
                                for conflict_country in conflicts:
                                    # Check if the snippet mentions the conflicting country prominently
                                    if conflict_country in result_text:
                                        # Don't penalize if expected location also appears
                                        if not any(
                                            part in result_text
                                            for part in location_parts
                                            if len(part) > 2
                                        ):
                                            contradiction_penalty += 2
                                            contradiction_reasons.append(
                                                f"location mismatch: {conflict_country}"
                                            )
                                            break

                    # Apply contradiction penalty
                    if contradiction_penalty > 0:
                        match_score -= contradiction_penalty
                        logger.debug(
                            f"Contradiction detected for '{linkedin_url}': {contradiction_reasons}, penalty={contradiction_penalty}"
                        )

                    # Skip if contradictions outweigh matches
                    if match_score < 0:
                        logger.debug(
                            f"Skipping '{linkedin_url}' - contradictions outweigh matches (score={match_score})"
                        )
                        continue

                    # Track all candidates with positive scores (not just best)
                    # We'll validate them in order of score if needed
                    vanity_match = re.search(
                        r"linkedin\.com/in/([\w\-]+)", linkedin_url
                    )
                    if vanity_match:
                        vanity = vanity_match.group(1)
                        candidate_url = f"https://www.linkedin.com/in/{vanity}"
                        candidates.append(
                            (match_score, candidate_url, matched_keywords)
                        )
                        logger.debug(
                            f"Candidate: {candidate_url} (score={match_score}, keywords={matched_keywords})"
                        )

                except Exception as e:
                    logger.debug(f"Error processing result item: {e}")
                    continue

            # Sort candidates by score (highest first)
            candidates.sort(key=lambda x: x[0], reverse=True)

            # HIGH PRECISION THRESHOLD: Require minimum score for confidence
            # Score breakdown:
            # - Name in URL slug: +3 (strong signal)
            # - Org match: +2
            # - Location match: +1
            # - Role type match: +1
            # - Keyword matches: +1 each
            # Require name_in_url (+3) or org_match (+2) + other signal
            MIN_CONFIDENCE_SCORE = 3  # Name in URL alone is sufficient
            MIN_CONFIDENCE_SCORE_COMMON_NAME = (
                4  # Common names need name_in_url + at least one other signal
            )

            threshold = (
                MIN_CONFIDENCE_SCORE_COMMON_NAME
                if is_common_name
                else MIN_CONFIDENCE_SCORE
            )

            # Try candidates in order of score until we find a valid one
            for candidate_score, candidate_url, matched_kws in candidates:
                if candidate_score < threshold:
                    logger.info(
                        f"Remaining candidates below threshold (best: {candidate_url}, score={candidate_score}, threshold={threshold})"
                    )
                    break

                # FINAL VALIDATION: Visit the profile page and verify the name
                if self._validate_profile_name(
                    driver, candidate_url, first_name, last_name
                ):
                    logger.info(
                        f"Found profile via UC Chrome: {candidate_url} (score={candidate_score}, threshold={threshold})"
                    )
                    return candidate_url
                else:
                    logger.info(
                        f"Rejecting candidate after page validation: {candidate_url} (name mismatch on actual profile), trying next..."
                    )

            if candidates:
                logger.debug(
                    f"No valid candidate found from {len(candidates)} candidates"
                )
            else:
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
        start_time = time.time()

        # Skip department lookup for generic academic departments that rarely have LinkedIn pages
        # This saves 4+ API calls per department that will almost certainly fail
        dept_lower = department.lower()
        skip_patterns = [
            "department of",  # Generic academic department naming
            "graduate school of",  # Graduate programs
            "school of",  # Generic school naming (but keep business schools)
            "faculty of",  # Faculty naming
            "institute of",  # Research institutes (too generic)
            "center for",  # Research centers
            "centre for",  # UK spelling
            "division of",  # Academic divisions
            "college of",  # Generic college naming
        ]

        # Don't skip business schools, engineering schools, etc. that often have LinkedIn pages
        keep_patterns = ["business", "management", "mba", "sloan", "gsb", "engineering"]

        should_skip = any(pattern in dept_lower for pattern in skip_patterns)
        should_keep = any(pattern in dept_lower for pattern in keep_patterns)

        if should_skip and not should_keep:
            logger.debug(
                f"Skipping department lookup (unlikely to have LinkedIn page): {department}"
            )
            return (None, None)

        logger.info(
            f"Searching for department LinkedIn page: {department} at {parent_org}"
        )

        # Strategy 1: Try common URL patterns directly via LinkedIn API
        # Limit to 2 most likely patterns to reduce API calls
        slug_candidates = self._generate_department_slug_candidates(
            department, parent_org, parent_slug
        )[:2]  # Only try top 2 candidates

        for candidate_slug in slug_candidates:
            result = self.lookup_organization_by_vanity_name(candidate_slug)
            if result:
                urn, org_id = result
                if urn:
                    url = f"https://www.linkedin.com/company/{candidate_slug}"
                    elapsed = time.time() - start_time
                    self._timing_stats["department_search"].append(elapsed)
                    logger.info(
                        f"Found department page via API: {department} at {parent_org} -> {url} ({elapsed:.1f}s)"
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
                elapsed = time.time() - start_time
                self._timing_stats["department_search"].append(elapsed)
                return (None, None)

            # Extract LinkedIn URL
            linkedin_url = self._extract_company_url(result)
            if not linkedin_url:
                elapsed = time.time() - start_time
                self._timing_stats["department_search"].append(elapsed)
                return (None, None)

            # Validate the URL format
            is_valid, slug = self.validate_linkedin_org_url(linkedin_url)
            if is_valid:
                elapsed = time.time() - start_time
                self._timing_stats["department_search"].append(elapsed)
                logger.info(
                    f"Found department page via search: {department} at {parent_org} -> {linkedin_url} ({elapsed:.1f}s)"
                )
                return (linkedin_url, slug)

            elapsed = time.time() - start_time
            self._timing_stats["department_search"].append(elapsed)
            return (None, None)

        except Exception as e:
            elapsed = time.time() - start_time
            self._timing_stats["department_search"].append(elapsed)
            logger.error(f"Error searching for department {department}: {e}")
            return (None, None)

    def populate_company_profiles(
        self,
        people: list[dict],
        delay_between_requests: float = 1.0,
    ) -> tuple[int, int, dict[str, tuple[str, Optional[str]]]]:
        """
        Look up LinkedIn profiles for people using a hierarchical approach.

        For each person, tries to find profiles in this order:
        1. Personal LinkedIn profile (linkedin.com/in/username)
        2. Department-specific page (if position indicates a department)
        3. Parent organization page

        Args:
            people: List of person dicts with name, company, position, linkedin_profile
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
        for person in people:
            company = person.get("company", "").strip()
            if company:
                companies.add(company)

        if not companies:
            logger.info("No companies found in people list")
            return (0, 0, {})

        companies_found = 0
        companies_not_found = 0
        company_data: dict[str, tuple[str, Optional[str], Optional[str]]] = {}

        # Look up parent organizations first (as fallback), using instance cache
        for i, company in enumerate(companies):
            if company in self._company_cache:
                # Cache hit
                cached = self._company_cache[company]
                if cached:
                    logger.info(f"Company cache hit: {company} -> {cached[0]}")
                    company_data[company] = cached
                    companies_found += 1
                else:
                    logger.debug(f"Company cache hit (not found): {company}")
                    companies_not_found += 1
                continue

            # Cache miss - do the lookup
            linkedin_url, slug = self.search_company(company)

            if linkedin_url:
                # Look up the organization URN via LinkedIn API
                urn = self.lookup_organization_urn(linkedin_url)
                company_data[company] = (linkedin_url, slug, urn)
                self._company_cache[company] = (linkedin_url, slug, urn)
                companies_found += 1
            else:
                self._company_cache[company] = None
                companies_not_found += 1

            # Rate limiting
            if i < len(companies) - 1:
                time.sleep(delay_between_requests)

        # Use instance-level caches to remember results across multiple stories
        # This avoids re-searching for the same person/department appearing in multiple articles
        person_cache = self._person_cache
        department_cache = self._department_cache

        # Process each person with hierarchical lookup
        for person in people:
            # Skip if already has a profile (from previous run or earlier in this run)
            if person.get("linkedin_profile"):
                logger.debug(f"Skipping {person.get('name')} - already has profile")
                continue

            name = person.get("name", "").strip()
            company = person.get("company", "").strip()
            position = person.get("position", "").strip()
            # Extract enhanced person metadata for better matching
            department = person.get("department", "").strip()
            location = person.get("location", "").strip()
            role_type = person.get("role_type", "").strip()
            research_area = person.get("research_area", "").strip()

            if not company:
                continue

            # ============================================================
            # HIERARCHY LEVEL 1: Try to find the individual's personal profile
            # ============================================================
            if name:
                # Include department and location in cache key for better specificity
                person_cache_key = f"{name}@{company}|{department}|{location}"

                if person_cache_key in person_cache:
                    # Cache hit - avoid redundant search
                    cached_url = person_cache[person_cache_key]
                    if cached_url:
                        logger.info(
                            f"Level 1: Cache hit for {name} at {company} -> {cached_url}"
                        )
                    else:
                        logger.debug(
                            f"Level 1: Cache hit (not found) for {name} at {company}"
                        )
                else:
                    logger.info(
                        f"Level 1: Searching for personal profile: {name} at {company}"
                    )
                    person_url = self.search_person(
                        name,
                        company,
                        position=position or None,
                        department=department or None,
                        location=location or None,
                        role_type=role_type or None,
                        research_area=research_area or None,
                    )
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
            # Use actual department field first; only parse from position as fallback
            dept_for_lookup = (
                department if department else self._extract_department_name(position)
            )

            if dept_for_lookup:
                dept_cache_key = f"{dept_for_lookup}@{company}"

                if dept_cache_key in department_cache:
                    # Cache hit
                    cached_dept = department_cache[dept_cache_key]
                    if cached_dept[0]:
                        logger.info(
                            f"Level 2: Cache hit for {dept_for_lookup}@{company} -> {cached_dept[0]}"
                        )
                    else:
                        logger.debug(
                            f"Level 2: Cache hit (not found) for {dept_for_lookup}@{company}"
                        )
                else:
                    # Cache miss - do the lookup
                    # Get parent org slug if available
                    parent_slug = None
                    if company in company_data:
                        parent_slug = company_data[company][1]

                    logger.info(
                        f"Level 2: Looking for department page: {dept_for_lookup} at {company}"
                    )
                    dept_url, dept_slug = self.search_department(
                        dept_for_lookup, company, parent_slug
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

        Uses retry logic with exponential backoff for reliability.

        Args:
            profile_url: LinkedIn profile URL (e.g., https://www.linkedin.com/in/username)

        Returns:
            Person URN (e.g., "urn:li:person:ABC123XYZ") if found, None otherwise
        """
        if not profile_url or "linkedin.com/in/" not in profile_url:
            return None

        # Retry the actual lookup with exponential backoff
        try:
            return self._lookup_person_urn_with_retry(profile_url)
        except Exception as e:
            logger.error(f"All retry attempts failed for {profile_url}: {e}")
            return None

    @with_enhanced_recovery(
        max_attempts=3,
        base_delay=5.0,
        retryable_exceptions=(Exception,),  # Broad retry for network/browser issues
    )
    def _lookup_person_urn_with_retry(self, profile_url: str) -> Optional[str]:
        """Internal method with retry logic for URN lookup."""
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
        """Clean up HTTP client and browser resources, and save cache."""
        # Save cache before closing
        self.save_cache_to_disk()

        if self._http_client:
            self._http_client.close()
            self._http_client = None
        self.close_browser()


# Backwards compatibility alias
LinkedInProfileLookup = LinkedInCompanyLookup


# =============================================================================
# TASK 2.1: Robust LinkedIn Profile URN Resolution
# =============================================================================


@dataclass
class URNLookupResult:
    """Result of a URN lookup attempt."""

    urn: Optional[str] = None
    source: str = ""  # "cache", "api", "playwright", "uc_chrome", "google_search"
    success: bool = False
    error: Optional[str] = None
    lookup_time_ms: float = 0.0


@dataclass
class URNCache:
    """Simple in-memory cache for URN lookups."""

    _cache: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _max_size: int = 1000

    def get(self, url: str) -> Optional[str]:
        """Get URN from cache."""
        with self._lock:
            return self._cache.get(url)

    def set(self, url: str, urn: str) -> None:
        """Set URN in cache."""
        with self._lock:
            # Simple LRU: if full, remove oldest entry
            if len(self._cache) >= self._max_size:
                first_key = next(iter(self._cache))
                del self._cache[first_key]
            self._cache[url] = urn

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()


# Type hint for Playwright
if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page, Playwright as PlaywrightSync


# Global URN cache
_urn_cache = URNCache()


class RobustURNResolver:
    """
    Robust LinkedIn URN resolver with multiple fallback strategies.

    TASK 2.1: Implements fallback chain for reliable URN extraction:
    1. Cache lookup (fastest)
    2. API lookup (if LinkedIn API configured)
    3. Playwright browser automation (preferred for reliability)
    4. UC Chrome fallback (if Playwright unavailable)
    5. Google Search (last resort via Gemini)

    Features:
    - Rate-limited concurrent lookups
    - Browser cookie persistence
    - Automatic retry with exponential backoff
    """

    # Rate limiting for LinkedIn: 60 lookups per minute
    _rate_limiter = AdaptiveRateLimiter(
        initial_fill_rate=1.0,  # 1 lookup per second
        capacity=10.0,  # Allow burst of 10
        min_fill_rate=0.5,
        max_fill_rate=2.0,
    )

    def __init__(
        self,
        use_playwright: bool = True,
        use_uc_chrome: bool = True,
        use_gemini: bool = True,
        max_concurrent: int = 3,
    ) -> None:
        """Initialize the robust URN resolver.

        Args:
            use_playwright: Enable Playwright for browser automation
            use_uc_chrome: Enable UC Chrome as fallback
            use_gemini: Enable Gemini/Google Search as last resort
            max_concurrent: Maximum concurrent lookups
        """
        self.use_playwright = use_playwright
        self.use_uc_chrome = use_uc_chrome
        self.use_gemini = use_gemini
        self.max_concurrent = max_concurrent

        # Playwright resources (use Any to avoid type issues with optional Playwright)
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._playwright_lock = threading.Lock()

        # UC Chrome fallback
        self._uc_lookup: Optional[LinkedInCompanyLookup] = None

        # Semaphore for concurrent lookup limiting
        self._semaphore = threading.Semaphore(max_concurrent)

        # Thread pool for concurrent lookups
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def __enter__(self) -> "RobustURNResolver":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Context manager exit."""
        self.close()
        return False

    def _get_playwright_page(self) -> Optional["Page"]:
        """Get or create a Playwright page with persistent session.

        Returns:
            Playwright Page object, or None if Playwright unavailable
        """
        with self._playwright_lock:
            if self._page is not None:
                return self._page

            try:
                from playwright.sync_api import sync_playwright

                # Use persistent context to save cookies/session
                playwright_profile = os.path.expandvars(
                    r"%LOCALAPPDATA%\SocialMediaPublisher\PlaywrightProfile"
                )
                os.makedirs(playwright_profile, exist_ok=True)

                self._playwright = sync_playwright().start()
                self._browser = self._playwright.chromium.launch_persistent_context(
                    playwright_profile,
                    headless=False,  # Visible for LinkedIn auth
                    viewport={"width": 1280, "height": 800},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
                )
                self._page = self._browser.new_page()
                logger.info(
                    f"Created Playwright session with profile: {playwright_profile}"
                )
                return self._page

            except ImportError:
                logger.warning(
                    "Playwright not installed - pip install playwright && playwright install chromium"
                )
                return None
            except Exception as e:
                logger.error(f"Failed to create Playwright session: {e}")
                return None

    def _ensure_linkedin_login_playwright(self, page: "Page") -> bool:
        """Ensure LinkedIn login in Playwright browser."""
        try:
            page.goto("https://www.linkedin.com/feed/", timeout=30000)
            page.wait_for_load_state("networkidle", timeout=10000)

            current_url = page.url

            if "/login" in current_url or "/authwall" in current_url:
                # Try auto-login if credentials available
                if Config.LINKEDIN_USERNAME and Config.LINKEDIN_PASSWORD:
                    logger.info("Attempting Playwright auto-login...")
                    page.goto("https://www.linkedin.com/login", timeout=30000)
                    page.wait_for_load_state("networkidle")

                    page.fill("#username", Config.LINKEDIN_USERNAME)
                    page.fill("#password", Config.LINKEDIN_PASSWORD)
                    page.click('button[type="submit"]')

                    page.wait_for_load_state("networkidle", timeout=30000)

                    if "/feed" in page.url:
                        logger.info("Playwright auto-login successful")
                        return True

                # Manual login fallback
                print("\n" + "=" * 60)
                print("LinkedIn Login Required (Playwright)")
                print("=" * 60)
                print("Please log in to LinkedIn in the browser window.")
                input("\nPress Enter after logging in...")

                page.goto("https://www.linkedin.com/feed/", timeout=30000)
                return "/feed" in page.url

            return True

        except Exception as e:
            logger.error(f"Playwright LinkedIn login error: {e}")
            return False

    def _lookup_via_playwright(self, profile_url: str) -> URNLookupResult:
        """Look up URN using Playwright browser automation."""
        start_time = time.perf_counter()

        page = self._get_playwright_page()
        if page is None:
            return URNLookupResult(
                error="Playwright not available",
                lookup_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        if not self._ensure_linkedin_login_playwright(page):
            return URNLookupResult(
                error="LinkedIn login required",
                lookup_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        try:
            page.goto(profile_url, timeout=30000)
            page.wait_for_load_state("networkidle", timeout=10000)

            # Extract URN from page source
            content = page.content()

            # Extract vanity name for matching
            vanity_match = re.search(r"linkedin\.com/in/([\w\-]+)", profile_url)
            vanity_name = vanity_match.group(1) if vanity_match else None

            if vanity_name:
                # Look for memberRelationship URN
                pattern = rf'"publicIdentifier":"{re.escape(vanity_name)}"[\s\S]{{0,500}}?"\*memberRelationship":"urn:li:fsd_memberRelationship:([A-Za-z0-9_-]+)"'
                match = re.search(pattern, content)
                if match:
                    urn = f"urn:li:person:{match.group(1)}"
                    return URNLookupResult(
                        urn=urn,
                        source="playwright",
                        success=True,
                        lookup_time_ms=(time.perf_counter() - start_time) * 1000,
                    )

                # Fallback: fsd_profile
                pattern = rf'"publicIdentifier":"{re.escape(vanity_name)}"[\s\S]{{0,500}}?"entityUrn":"urn:li:fsd_profile:([A-Za-z0-9_-]+)"'
                match = re.search(pattern, content)
                if match:
                    urn = f"urn:li:person:{match.group(1)}"
                    return URNLookupResult(
                        urn=urn,
                        source="playwright",
                        success=True,
                        lookup_time_ms=(time.perf_counter() - start_time) * 1000,
                    )

            return URNLookupResult(
                error="URN not found in page",
                lookup_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        except Exception as e:
            return URNLookupResult(
                error=str(e),
                lookup_time_ms=(time.perf_counter() - start_time) * 1000,
            )

    def _lookup_via_uc_chrome(self, profile_url: str) -> URNLookupResult:
        """Look up URN using UC Chrome (fallback)."""
        start_time = time.perf_counter()

        if not UC_AVAILABLE:
            return URNLookupResult(
                error="UC Chrome not available",
                lookup_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        try:
            if self._uc_lookup is None:
                self._uc_lookup = LinkedInCompanyLookup(genai_client=None)

            urn = self._uc_lookup.lookup_person_urn(profile_url)

            if urn:
                return URNLookupResult(
                    urn=urn,
                    source="uc_chrome",
                    success=True,
                    lookup_time_ms=(time.perf_counter() - start_time) * 1000,
                )
            else:
                return URNLookupResult(
                    error="URN not found via UC Chrome",
                    lookup_time_ms=(time.perf_counter() - start_time) * 1000,
                )

        except Exception as e:
            return URNLookupResult(
                error=str(e),
                lookup_time_ms=(time.perf_counter() - start_time) * 1000,
            )

    def resolve_person_urn(self, profile_url: str) -> URNLookupResult:
        """
        Resolve a person's URN using the fallback chain.

        Fallback order:
        1. Cache (instant)
        2. Playwright (preferred browser automation)
        3. UC Chrome (fallback)
        4. Returns error if all methods fail

        Args:
            profile_url: LinkedIn profile URL

        Returns:
            URNLookupResult with URN or error details
        """
        if not profile_url or "linkedin.com/in/" not in profile_url:
            return URNLookupResult(error="Invalid LinkedIn profile URL")

        # Normalize URL
        profile_url = profile_url.split("?")[0].rstrip("/")

        # 1. Check cache first
        cached_urn = _urn_cache.get(profile_url)
        if cached_urn:
            return URNLookupResult(
                urn=cached_urn,
                source="cache",
                success=True,
                lookup_time_ms=0.0,
            )

        # Rate limit before browser lookups
        self._rate_limiter.wait(endpoint="urn_lookup")

        with self._semaphore:
            # 2. Try Playwright first (if enabled)
            if self.use_playwright:
                result = self._lookup_via_playwright(profile_url)
                if result.success and result.urn:
                    _urn_cache.set(profile_url, result.urn)
                    self._rate_limiter.on_success(endpoint="urn_lookup")
                    return result

            # 3. Try UC Chrome fallback (if enabled)
            if self.use_uc_chrome:
                result = self._lookup_via_uc_chrome(profile_url)
                if result.success and result.urn:
                    _urn_cache.set(profile_url, result.urn)
                    self._rate_limiter.on_success(endpoint="urn_lookup")
                    return result

        # All methods failed
        return URNLookupResult(error="All lookup methods failed")

    def resolve_multiple_urns(
        self, profile_urls: list[str]
    ) -> dict[str, URNLookupResult]:
        """
        Resolve multiple URNs concurrently with rate limiting.

        Args:
            profile_urls: List of LinkedIn profile URLs

        Returns:
            Dictionary mapping URLs to their lookup results
        """
        results: dict[str, URNLookupResult] = {}

        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_concurrent
            )

        # Submit all lookups
        futures = {
            self._executor.submit(self.resolve_person_urn, url): url
            for url in profile_urls
        }

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                results[url] = future.result(timeout=60)
            except Exception as e:
                results[url] = URNLookupResult(error=str(e))

        return results

    def close(self) -> None:
        """Clean up all resources."""
        # Close Playwright
        with self._playwright_lock:
            if self._page:
                try:
                    self._page.close()
                except Exception:
                    pass
                self._page = None

            if self._browser:
                try:
                    self._browser.close()
                except Exception:
                    pass
                self._browser = None

            if self._playwright:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None

        # Close UC Chrome
        if self._uc_lookup:
            self._uc_lookup.close()
            self._uc_lookup = None

        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None


# Singleton instance for easy access
_robust_resolver: Optional[RobustURNResolver] = None


def get_robust_urn_resolver() -> RobustURNResolver:
    """Get or create the global RobustURNResolver instance."""
    global _robust_resolver
    if _robust_resolver is None:
        _robust_resolver = RobustURNResolver()
    return _robust_resolver


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for linkedin_profile_lookup module."""
    from test_framework import TestSuite

    suite = TestSuite("LinkedIn Profile Lookup Tests")

    def test_validate_linkedin_org_url_valid_company():
        lookup = LinkedInCompanyLookup(genai_client=None)
        valid, slug = lookup.validate_linkedin_org_url(
            "https://www.linkedin.com/company/google"
        )
        assert valid is True
        assert slug == "google"
        lookup.close()

    def test_validate_linkedin_org_url_valid_school():
        lookup = LinkedInCompanyLookup(genai_client=None)
        valid, slug = lookup.validate_linkedin_org_url(
            "https://www.linkedin.com/school/stanford-university"
        )
        assert valid is True
        assert slug == "stanford-university"
        lookup.close()

    def test_validate_linkedin_org_url_invalid():
        lookup = LinkedInCompanyLookup(genai_client=None)
        valid, slug = lookup.validate_linkedin_org_url(
            "https://example.com/not-linkedin"
        )
        assert valid is False
        assert slug is None
        lookup.close()

    def test_validate_linkedin_org_url_empty():
        lookup = LinkedInCompanyLookup(genai_client=None)
        valid, slug = lookup.validate_linkedin_org_url("")
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
        text = "Visit https://www.linkedin.com/in/wayne-gault for profile"
        url = lookup._extract_person_url(text)
        assert url is not None
        assert "wayne-gault" in url
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

    def test_lookup_person_urn_invalid_url():
        """Test that invalid URLs return None without error."""
        lookup = LinkedInCompanyLookup(genai_client=None)
        # Test with invalid URL
        result = lookup.lookup_person_urn("https://example.com/not-linkedin")
        assert result is None
        # Test with empty URL
        result = lookup.lookup_person_urn("")
        assert result is None
        lookup.close()

    def test_lookup_person_urn_no_driver():
        """Test that lookup handles gracefully with or without browser driver."""
        lookup = LinkedInCompanyLookup(genai_client=None)
        # Test with valid LinkedIn profile URL
        result = lookup.lookup_person_urn("https://www.linkedin.com/in/wayne-gault")
        # Result could be None (no driver) or a URN string (driver available)
        assert result is None or (
            isinstance(result, str) and result.startswith("urn:li:")
        )
        lookup.close()

    def test_urn_cache():
        """Test URN cache operations."""
        cache = URNCache()
        cache.clear()

        # Test set and get
        cache.set("https://linkedin.com/in/test", "urn:li:person:ABC123")
        result = cache.get("https://linkedin.com/in/test")
        assert result == "urn:li:person:ABC123"

        # Test missing key
        result = cache.get("https://linkedin.com/in/nonexistent")
        assert result is None

    def test_robust_resolver_invalid_url():
        """Test robust resolver with invalid URL."""
        resolver = RobustURNResolver(use_playwright=False, use_uc_chrome=False)
        result = resolver.resolve_person_urn("https://example.com/not-linkedin")
        assert not result.success
        assert result.error == "Invalid LinkedIn profile URL"
        resolver.close()

    suite.add_test(
        "Validate LinkedIn URL - valid company",
        test_validate_linkedin_org_url_valid_company,
    )
    suite.add_test(
        "Validate LinkedIn URL - valid school",
        test_validate_linkedin_org_url_valid_school,
    )
    suite.add_test(
        "Validate LinkedIn URL - invalid", test_validate_linkedin_org_url_invalid
    )
    suite.add_test(
        "Validate LinkedIn URL - empty", test_validate_linkedin_org_url_empty
    )
    suite.add_test("Generate acronym - long phrase", test_generate_acronym_long)
    suite.add_test("Generate acronym - short name", test_generate_acronym_short)
    suite.add_test("Extract company URL from text", test_extract_company_url)
    suite.add_test("Extract person URL from text", test_extract_person_url)
    suite.add_test("Lookup class initialization", test_lookup_class_init)
    suite.add_test("Context manager works", test_context_manager)
    suite.add_test(
        "Lookup person URN - invalid URL", test_lookup_person_urn_invalid_url
    )
    suite.add_test("Lookup person URN - no driver", test_lookup_person_urn_no_driver)
    suite.add_test("URN cache operations", test_urn_cache)
    suite.add_test("Robust resolver - invalid URL", test_robust_resolver_invalid_url)

    return suite
