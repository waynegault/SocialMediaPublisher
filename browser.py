"""
Unified browser automation module using undetected-chromedriver.

Consolidates duplicate UC Chrome setup code from:
- linkedin_profile_lookup.py
- find_indirect_people.py
- searcher.py

Provides:
- UC Chrome driver creation with consistent options
- Windows error suppression for driver cleanup
- Session management for driver reuse
"""

import logging
import sys
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Type stubs for optional dependencies
uc: Any = None
By: Any = None
UC_AVAILABLE = False

try:
    import undetected_chromedriver as _uc
    from selenium.webdriver.common.by import By as _By
    from selenium.webdriver.support.ui import WebDriverWait as _WebDriverWait
    from selenium.webdriver.support import expected_conditions as _EC

    uc = _uc
    By = _By
    WebDriverWait = _WebDriverWait
    EC = _EC
    UC_AVAILABLE = True
except ImportError:
    logger.warning(
        "undetected-chromedriver not installed - pip install undetected-chromedriver selenium"
    )
    # Provide stub types for type checking
    WebDriverWait = None
    EC = None


def _suppress_uc_cleanup_errors() -> None:
    """
    Suppress Windows handle errors from UC Chrome cleanup.

    Windows can throw 'OSError: [WinError 6] The handle is invalid' during
    Chrome process cleanup. This patches the time module used by UC to
    suppress these harmless errors.
    """
    if sys.platform == "win32" and UC_AVAILABLE:
        original_sleep = time.sleep

        def patched_sleep(seconds: float) -> None:
            try:
                original_sleep(seconds)
            except OSError:
                pass  # Suppress WinError 6: handle is invalid

        # Apply to UC module
        try:
            import undetected_chromedriver

            undetected_chromedriver.time = type(sys)("time")
            undetected_chromedriver.time.sleep = patched_sleep
        except Exception:
            pass  # Ignore if patching fails


# Apply the Windows error suppression on module load
_suppress_uc_cleanup_errors()


def create_chrome_driver(
    headless: bool = True,
    timeout: int = 30,
    extra_options: Optional[list[str]] = None,
) -> Any:
    """
    Create a new UC Chrome driver with consistent configuration.

    Args:
        headless: Whether to run in headless mode (default True)
        timeout: Page load timeout in seconds (default 30)
        extra_options: Additional Chrome options to add

    Returns:
        A configured undetected_chromedriver.Chrome instance

    Raises:
        ImportError: If undetected-chromedriver is not installed
        Exception: If driver creation fails
    """
    if not UC_AVAILABLE:
        raise ImportError(
            "undetected-chromedriver not installed. "
            "Install with: pip install undetected-chromedriver selenium"
        )

    options = uc.ChromeOptions()

    # Standard options for reliability
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--log-level=3")  # Suppress Chrome warnings
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")

    # Add any extra options
    if extra_options:
        for opt in extra_options:
            options.add_argument(opt)

    # Create driver with subprocess for better isolation
    driver = uc.Chrome(options=options, use_subprocess=True)
    driver.set_page_load_timeout(timeout)

    return driver


def quit_driver_safely(driver: Any) -> None:
    """
    Safely quit a Chrome driver, suppressing cleanup errors.

    Args:
        driver: The Chrome driver instance to quit
    """
    if driver is None:
        return

    try:
        driver.quit()
    except Exception as e:
        logger.debug(f"Driver cleanup warning (harmless): {e}")


def resolve_redirect_with_browser(
    url: str,
    timeout: int = 15,
    wait_for_js: float = 2.0,
) -> str:
    """
    Use UC Chrome to resolve a redirect URL that requests couldn't handle.

    This is a fallback for redirects that require JavaScript execution.

    Args:
        url: The URL to resolve
        timeout: Page load timeout in seconds
        wait_for_js: Seconds to wait for JavaScript redirects

    Returns:
        The final resolved URL, or empty string if resolution fails
    """
    if not UC_AVAILABLE:
        logger.debug("UC Chrome not available for redirect resolution")
        return ""

    driver = None
    try:
        logger.debug(f"Resolving redirect via browser: {url[:50]}...")

        driver = create_chrome_driver(headless=True, timeout=timeout)
        driver.get(url)

        # Wait for JavaScript redirects
        time.sleep(wait_for_js)

        final_url = driver.current_url

        # Validate the result
        if final_url and "vertexaisearch.cloud.google.com" not in final_url:
            logger.info(f"Browser resolved: {url[:40]}... -> {final_url[:50]}...")
            return final_url

    except ImportError:
        logger.debug("UC Chrome not available")
    except Exception as e:
        logger.debug(f"Browser redirect resolution failed: {e}")
    finally:
        quit_driver_safely(driver)

    return ""


class BrowserSession:
    """
    Manages a reusable browser session for multiple operations.

    Use as a context manager for automatic cleanup:

        with BrowserSession() as session:
            driver = session.driver
            driver.get("https://example.com")
            # ... use driver ...
    """

    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30,
        extra_options: Optional[list[str]] = None,
    ):
        """
        Initialize a browser session.

        Args:
            headless: Whether to run in headless mode
            timeout: Page load timeout in seconds
            extra_options: Additional Chrome options
        """
        self.headless = headless
        self.timeout = timeout
        self.extra_options = extra_options
        self._driver: Any = None

    @property
    def driver(self) -> Any:
        """Get the Chrome driver, creating it if necessary."""
        if self._driver is None:
            self._driver = create_chrome_driver(
                headless=self.headless,
                timeout=self.timeout,
                extra_options=self.extra_options,
            )
        return self._driver

    def close(self) -> None:
        """Close the browser session."""
        if self._driver is not None:
            quit_driver_safely(self._driver)
            self._driver = None

    def __enter__(self) -> "BrowserSession":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
