"""
Browser automation backends for anti-detection web scraping.

Provides two backend implementations:
1. undetected-chromedriver (uc) - The traditional approach
2. nodriver - The modern successor with better anti-detection

Use the factory function get_browser_backend() to get the configured backend.
"""

import asyncio
import logging
import os
import random
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from config import Config

logger = logging.getLogger(__name__)

# ============================================================================
# Type stubs for optional dependencies
# ============================================================================

uc: Any = None
UC_AVAILABLE = False
nodriver: Any = None
NODRIVER_AVAILABLE = False

try:
    import undetected_chromedriver as _uc
    from selenium.webdriver.common.by import By as _By

    uc = _uc
    By = _By
    UC_AVAILABLE = True
except ImportError:
    By = None
    logger.debug("undetected-chromedriver not available")

try:
    import nodriver as _nodriver

    nodriver = _nodriver
    NODRIVER_AVAILABLE = True
except ImportError:
    logger.debug("nodriver not available")


# ============================================================================
# Abstract Base Class for Browser Backends
# ============================================================================


class BrowserBackend(ABC):
    """Abstract base class for browser automation backends."""

    @abstractmethod
    def search(
        self,
        query: str,
        engine: str = "google",
        timeout: int = 30,
    ) -> Optional[str]:
        """
        Perform a search and return the page source.

        Args:
            query: Search query string
            engine: Search engine to use (google, bing, duckduckgo)
            timeout: Page load timeout in seconds

        Returns:
            Page source HTML or None if search failed
        """
        pass

    @abstractmethod
    def navigate(self, url: str, timeout: int = 30) -> Optional[str]:
        """
        Navigate to a URL and return the page source.

        Args:
            url: URL to navigate to
            timeout: Page load timeout in seconds

        Returns:
            Page source HTML or None if navigation failed
        """
        pass

    @abstractmethod
    def get_current_url(self) -> Optional[str]:
        """Get the current page URL."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the browser session."""
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        """Check if the browser session is still alive."""
        pass


# ============================================================================
# Undetected ChromeDriver Backend
# ============================================================================


class UCBrowserBackend(BrowserBackend):
    """Browser backend using undetected-chromedriver."""

    def __init__(self):
        self._driver: Any = None
        self._search_count = 0
        self._max_searches = 8

    def _create_driver(self) -> Optional[Any]:
        """Create a new UC Chrome driver with anti-detection settings."""
        if not UC_AVAILABLE:
            logger.error("undetected-chromedriver not available")
            return None

        try:
            options = uc.ChromeOptions()

            # Core stability options
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--log-level=3")  # Suppress Chrome warnings
            options.add_argument("--disable-software-rasterizer")
            options.add_argument("--no-first-run")
            options.add_argument("--no-default-browser-check")

            # Anti-detection flags
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-popup-blocking")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-plugins-discovery")
            options.add_argument("--disable-automation")

            # Preferences
            prefs = {
                "credentials_enable_service": False,
                "profile.password_manager_enabled": False,
                "profile.default_content_setting_values.notifications": 2,
            }
            options.add_experimental_option("prefs", prefs)

            # User profile directory
            automation_profile = os.path.expandvars(
                r"%LOCALAPPDATA%\SocialMediaPublisher\ChromeProfile"
            )
            os.makedirs(automation_profile, exist_ok=True)
            options.add_argument(f"--user-data-dir={automation_profile}")
            options.add_argument("--profile-directory=Default")

            # User agent - should match installed Chrome version
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
            options.add_argument(f"--user-agent={user_agent}")

            # Window size (randomized slightly)
            width = 1920 + random.randint(-100, 100)
            height = 1080 + random.randint(-50, 50)
            options.add_argument(f"--window-size={width},{height}")

            driver = uc.Chrome(
                options=options,
                use_subprocess=True,
                suppress_welcome=True,
                headless=False,  # Must be False for anti-detection
            )
            driver.set_page_load_timeout(30)

            # Additional stealth via JavaScript
            try:
                driver.execute_script("""
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5]
                    });
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en']
                    });
                """)
            except Exception:
                pass

            logger.debug("Created UC Chrome driver with anti-detection settings")
            return driver

        except Exception as e:
            logger.error(f"Failed to create UC Chrome driver: {e}")
            return None

    def _ensure_driver(self) -> Optional[Any]:
        """Ensure we have a valid driver, creating one if needed."""
        if self._driver is not None:
            # Check if driver is still alive
            try:
                _ = self._driver.current_url
                # Rotate driver after too many searches
                if self._search_count >= self._max_searches:
                    self._search_count = 0
                    time.sleep(5 + random.random() * 3)
                    try:
                        self._driver.delete_all_cookies()
                    except Exception:
                        pass
                return self._driver
            except Exception:
                self._cleanup_driver()

        # Create new driver
        self._driver = self._create_driver()
        return self._driver

    def _cleanup_driver(self) -> None:
        """Clean up the driver and kill orphan processes."""
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None

        self._search_count = 0

        # Kill orphan chromedriver processes on Windows
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "chromedriver.exe"],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass

        time.sleep(2)

    def search(
        self,
        query: str,
        engine: str = "google",
        timeout: int = 30,
    ) -> Optional[str]:
        """Perform a search using UC Chrome."""
        driver = self._ensure_driver()
        if driver is None:
            return None

        try:
            import urllib.parse

            encoded_query = urllib.parse.quote(query)

            engines = {
                "google": f"https://www.google.com/search?q={encoded_query}",
                "bing": f"https://www.bing.com/search?q={encoded_query}",
                "duckduckgo": f"https://duckduckgo.com/?q={encoded_query}",
            }

            url = engines.get(engine.lower(), engines["google"])
            driver.get(url)

            # Human-like delay
            time.sleep(4 + random.random() * 4)

            # Simulate human behavior
            self._simulate_human_behavior(driver)

            self._search_count += 1
            return driver.page_source

        except Exception as e:
            logger.error(f"UC Chrome search error: {e}")
            self._cleanup_driver()
            return None

    def _simulate_human_behavior(self, driver: Any) -> None:
        """Simulate human-like browser interactions."""
        try:
            # Scroll down
            driver.execute_script("window.scrollBy(0, 200 + Math.random() * 300);")
            time.sleep(0.3 + random.random() * 0.5)

            # Mouse movement simulation
            driver.execute_script("""
                document.dispatchEvent(new MouseEvent('mousemove', {
                    clientX: 100 + Math.random() * 500,
                    clientY: 100 + Math.random() * 300,
                    bubbles: true
                }));
            """)
            time.sleep(0.2 + random.random() * 0.3)

            # Scroll back up slightly
            driver.execute_script("window.scrollBy(0, -(50 + Math.random() * 100));")
            time.sleep(0.3 + random.random() * 0.4)
        except Exception:
            pass

    def navigate(self, url: str, timeout: int = 30) -> Optional[str]:
        """Navigate to a URL."""
        driver = self._ensure_driver()
        if driver is None:
            return None

        try:
            driver.set_page_load_timeout(timeout)
            driver.get(url)
            time.sleep(2 + random.random() * 2)
            return driver.page_source
        except Exception as e:
            logger.error(f"UC Chrome navigation error: {e}")
            return None

    def get_current_url(self) -> Optional[str]:
        """Get the current URL."""
        if self._driver is None:
            return None
        try:
            return self._driver.current_url
        except Exception:
            return None

    def close(self) -> None:
        """Close the browser."""
        self._cleanup_driver()

    def is_alive(self) -> bool:
        """Check if the driver is still alive."""
        if self._driver is None:
            return False
        try:
            _ = self._driver.current_url
            return True
        except Exception:
            return False


# ============================================================================
# Nodriver Backend (Recommended)
# ============================================================================


class NodriverBackend(BrowserBackend):
    """
    Browser backend using nodriver - the successor to undetected-chromedriver.

    Nodriver provides better anti-detection and is async-native. This wrapper
    provides a synchronous interface for compatibility with existing code.
    """

    def __init__(self):
        self._browser: Any = None
        self._tab: Any = None
        self._loop: Any = None
        self._search_count = 0
        self._max_searches = 12  # nodriver can handle more searches

    def _get_loop(self) -> Any:
        """Get or create an event loop."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    async def _create_browser_async(self) -> bool:
        """Create a new nodriver browser instance asynchronously."""
        if not NODRIVER_AVAILABLE:
            logger.error("nodriver not available - pip install nodriver")
            return False

        try:
            # Start browser with optimal settings for anti-detection
            self._browser = await nodriver.start(
                headless=False,  # Must be False for anti-detection
                sandbox=False,
                lang="en-US",
            )

            # Get the initial tab
            self._tab = await self._browser.get("about:blank")

            logger.debug("Created nodriver browser with anti-detection settings")
            return True

        except Exception as e:
            logger.error(f"Failed to create nodriver browser: {e}")
            return False

    def _ensure_browser(self) -> bool:
        """Ensure we have a valid browser, creating one if needed."""
        if self._browser is not None and self._tab is not None:
            # Rotate after many searches
            if self._search_count >= self._max_searches:
                self._search_count = 0
                # Clear cookies via CDP command
                try:
                    self._run_async(self._clear_cookies_async())
                except Exception:
                    pass
            return True

        # Create new browser
        return self._run_async(self._create_browser_async())

    async def _clear_cookies_async(self) -> None:
        """Clear browser cookies."""
        if self._tab is not None:
            try:
                await self._tab.send(nodriver.cdp.network.clear_browser_cookies())
            except Exception:
                pass

    async def _search_async(
        self,
        query: str,
        engine: str = "google",
        timeout: int = 30,
    ) -> Optional[str]:
        """Perform an async search."""
        import urllib.parse

        encoded_query = urllib.parse.quote(query)

        engines = {
            "google": f"https://www.google.com/search?q={encoded_query}",
            "bing": f"https://www.bing.com/search?q={encoded_query}",
            "duckduckgo": f"https://duckduckgo.com/?q={encoded_query}",
        }

        url = engines.get(engine.lower(), engines["google"])

        try:
            self._tab = await self._browser.get(url)

            # Wait for page load
            await self._tab.sleep(3 + random.random() * 3)

            # Simulate human behavior
            await self._simulate_human_behavior_async()

            # Check for CAPTCHA/challenge page
            page_source = await self._tab.get_content()

            # Try to handle Cloudflare challenge if detected
            if "challenge" in page_source.lower() or "captcha" in page_source.lower():
                logger.warning("Challenge detected, attempting to bypass...")
                try:
                    await self._tab.verify_cf()
                    await self._tab.sleep(3)
                    page_source = await self._tab.get_content()
                except Exception as e:
                    logger.warning(f"Challenge bypass failed: {e}")

            self._search_count += 1
            return page_source

        except Exception as e:
            logger.error(f"Nodriver search error: {e}")
            return None

    async def _simulate_human_behavior_async(self) -> None:
        """Simulate human-like behavior asynchronously."""
        try:
            # Scroll down randomly
            await self._tab.scroll_down(200 + random.randint(0, 300))
            await self._tab.sleep(0.3 + random.random() * 0.5)

            # Scroll up slightly
            await self._tab.scroll_up(50 + random.randint(0, 100))
            await self._tab.sleep(0.3 + random.random() * 0.4)
        except Exception:
            pass

    def search(
        self,
        query: str,
        engine: str = "google",
        timeout: int = 30,
    ) -> Optional[str]:
        """Perform a search using nodriver."""
        if not self._ensure_browser():
            return None

        try:
            return self._run_async(self._search_async(query, engine, timeout))
        except Exception as e:
            logger.error(f"Nodriver search failed: {e}")
            return None

    async def _navigate_async(self, url: str, timeout: int = 30) -> Optional[str]:
        """Navigate to a URL asynchronously."""
        try:
            self._tab = await self._browser.get(url)
            await self._tab.sleep(2 + random.random() * 2)
            return await self._tab.get_content()
        except Exception as e:
            logger.error(f"Nodriver navigation error: {e}")
            return None

    def navigate(self, url: str, timeout: int = 30) -> Optional[str]:
        """Navigate to a URL."""
        if not self._ensure_browser():
            return None

        try:
            return self._run_async(self._navigate_async(url, timeout))
        except Exception as e:
            logger.error(f"Nodriver navigation failed: {e}")
            return None

    def get_current_url(self) -> Optional[str]:
        """Get the current URL."""
        if self._tab is None:
            return None
        try:
            return self._tab.url
        except Exception:
            return None

    def close(self) -> None:
        """Close the browser."""
        if self._browser is not None:
            try:
                self._run_async(self._browser.stop())
            except Exception:
                pass
            self._browser = None
            self._tab = None

    def is_alive(self) -> bool:
        """Check if the browser is still alive."""
        return self._browser is not None and self._tab is not None


# ============================================================================
# Factory Function
# ============================================================================

_backend_instance: Optional[BrowserBackend] = None


def get_browser_backend(
    force_backend: Optional[str] = None,
) -> Optional[BrowserBackend]:
    """
    Get the configured browser backend instance.

    Args:
        force_backend: Override the config setting ("uc" or "nodriver")

    Returns:
        Browser backend instance or None if no backend is available
    """
    global _backend_instance

    backend_type = force_backend or Config.BROWSER_BACKEND

    # If we have an existing instance of the right type, return it
    if _backend_instance is not None:
        if backend_type == "nodriver" and isinstance(
            _backend_instance, NodriverBackend
        ):
            return _backend_instance
        if backend_type == "uc" and isinstance(_backend_instance, UCBrowserBackend):
            return _backend_instance
        # Wrong type, close and create new
        _backend_instance.close()
        _backend_instance = None

    # Try to create the requested backend
    if backend_type == "nodriver":
        if NODRIVER_AVAILABLE:
            _backend_instance = NodriverBackend()
            logger.info(
                "Using nodriver browser backend (recommended for anti-detection)"
            )
            return _backend_instance
        else:
            logger.warning(
                "nodriver not available, falling back to undetected-chromedriver"
            )

    # Fall back to UC
    if UC_AVAILABLE:
        _backend_instance = UCBrowserBackend()
        logger.info("Using undetected-chromedriver browser backend")
        return _backend_instance

    logger.error(
        "No browser backend available. Install nodriver or undetected-chromedriver"
    )
    return None


def close_browser_backend() -> None:
    """Close the global browser backend instance."""
    global _backend_instance
    if _backend_instance is not None:
        _backend_instance.close()
        _backend_instance = None


def is_captcha_page(page_source: str) -> bool:
    """
    Check if the page source indicates a CAPTCHA or challenge page.

    Args:
        page_source: HTML content of the page

    Returns:
        True if CAPTCHA indicators are detected
    """
    if not page_source:
        return False

    lower_source = page_source.lower()

    indicators = [
        "captcha",
        "i'm not a robot",
        "unusual traffic",
        "verify you are human",
        "recaptcha",
        "hcaptcha",
        "challenge-form",
        "please verify",
        "security check",
        "automated queries",
    ]

    return any(indicator in lower_source for indicator in indicators)


# =============================================================================
# Module Tests
# =============================================================================


def _create_module_tests() -> bool:
    """Create unit tests for browser_backends module."""
    from test_framework import TestSuite

    suite = TestSuite("Browser Backends", "browser_backends.py")
    suite.start_suite()

    def test_is_captcha_page_true():
        html = "<html><body>Please complete the captcha to continue</body></html>"
        assert is_captcha_page(html) is True

    def test_is_captcha_page_recaptcha():
        html = '<html><body><div class="recaptcha">Verify</div></body></html>'
        assert is_captcha_page(html) is True

    def test_is_captcha_page_hcaptcha():
        html = "<html><body>hcaptcha challenge detected</body></html>"
        assert is_captcha_page(html) is True

    def test_is_captcha_page_false():
        html = "<html><body><h1>Welcome to LinkedIn</h1></body></html>"
        assert is_captcha_page(html) is False

    def test_is_captcha_page_empty():
        assert is_captcha_page("") is False

    def test_is_captcha_page_unusual_traffic():
        html = "<html><body>We detected unusual traffic from your network</body></html>"
        assert is_captcha_page(html) is True

    def test_is_captcha_verify_human():
        html = "<html><body>Please verify you are human</body></html>"
        assert is_captcha_page(html) is True

    def test_browser_backend_abstract():
        # Ensure BrowserBackend can't be instantiated directly
        try:
            BrowserBackend()  # type: ignore
            assert False, "Should have raised TypeError"
        except TypeError:
            pass

    def test_uc_backend_class_exists():
        assert issubclass(UCBrowserBackend, BrowserBackend)
        assert hasattr(UCBrowserBackend, "search")
        assert hasattr(UCBrowserBackend, "navigate")
        assert hasattr(UCBrowserBackend, "close")

    def test_nodriver_backend_class_exists():
        assert issubclass(NodriverBackend, BrowserBackend)
        assert hasattr(NodriverBackend, "search")
        assert hasattr(NodriverBackend, "navigate")
        assert hasattr(NodriverBackend, "close")

    suite.run_test(

        test_name="is_captcha_page - captcha",

        test_func=test_is_captcha_page_true,

        test_summary="is_captcha_page behavior with captcha input",

        method_description="Testing is_captcha_page with captcha input using boolean return verification",

        expected_outcome="Function returns True for captcha input",

    )
    suite.run_test(
        test_name="is_captcha_page - recaptcha",
        test_func=test_is_captcha_page_recaptcha,
        test_summary="is_captcha_page behavior with recaptcha input",
        method_description="Testing is_captcha_page with recaptcha input using boolean return verification",
        expected_outcome="Function returns True for recaptcha input",
    )
    suite.run_test(
        test_name="is_captcha_page - hcaptcha",
        test_func=test_is_captcha_page_hcaptcha,
        test_summary="is_captcha_page behavior with hcaptcha input",
        method_description="Testing is_captcha_page with hcaptcha input using boolean return verification",
        expected_outcome="Function returns True for hcaptcha input",
    )
    suite.run_test(
        test_name="is_captcha_page - false",
        test_func=test_is_captcha_page_false,
        test_summary="is_captcha_page behavior with false input",
        method_description="Testing is_captcha_page with false input using boolean return verification",
        expected_outcome="Function returns False for false input",
    )
    suite.run_test(
        test_name="is_captcha_page - empty",
        test_func=test_is_captcha_page_empty,
        test_summary="is_captcha_page behavior with empty input",
        method_description="Testing is_captcha_page with empty input using boolean return verification",
        expected_outcome="Function returns False for empty input",
    )
    suite.run_test(
        test_name="is_captcha_page - unusual traffic",
        test_func=test_is_captcha_page_unusual_traffic,
        test_summary="is_captcha_page behavior with unusual traffic input",
        method_description="Testing is_captcha_page with unusual traffic input using boolean return verification",
        expected_outcome="Function returns True for unusual traffic input",
    )
    suite.run_test(
        test_name="is_captcha_page - verify human",
        test_func=test_is_captcha_verify_human,
        test_summary="is_captcha_page behavior with verify human input",
        method_description="Testing is_captcha_page with verify human input using boolean return verification",
        expected_outcome="Function returns True for verify human input",
    )
    suite.run_test(
        test_name="BrowserBackend abstract",
        test_func=test_browser_backend_abstract,
        test_summary="Verify BrowserBackend abstract produces correct results",
        method_description="Testing BrowserBackend abstract using error handling validation",
        expected_outcome="BrowserBackend abstract raises appropriate error for invalid input",
    )
    suite.run_test(
        test_name="UCBrowserBackend class",
        test_func=test_uc_backend_class_exists,
        test_summary="Verify UCBrowserBackend class produces correct results",
        method_description="Testing UCBrowserBackend class",
        expected_outcome="All assertions pass confirming UCBrowserBackend class works correctly",
    )
    suite.run_test(
        test_name="NodriverBackend class",
        test_func=test_nodriver_backend_class_exists,
        test_summary="Verify NodriverBackend class produces correct results",
        method_description="Testing NodriverBackend class",
        expected_outcome="All assertions pass confirming NodriverBackend class works correctly",
    )
    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
