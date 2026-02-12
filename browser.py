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


# ============================================================================
# Module Tests
# ============================================================================


def _create_module_tests():
    """Create and run tests for browser module."""
    from test_framework import TestSuite

    suite = TestSuite("Browser", "browser.py")

    def test_uc_available_is_bool():
        """Test UC_AVAILABLE is a boolean."""
        assert isinstance(UC_AVAILABLE, bool)

    def test_quit_driver_safely_none():
        """Test quit_driver_safely handles None gracefully."""
        quit_driver_safely(None)  # Should not raise

    def test_quit_driver_safely_mock():
        """Test quit_driver_safely calls quit on driver."""
        class MockDriver:
            def __init__(self):
                self.quit_called = False

            def quit(self):
                self.quit_called = True

        driver = MockDriver()
        quit_driver_safely(driver)
        assert driver.quit_called

    def test_quit_driver_safely_error():
        """Test quit_driver_safely suppresses exceptions."""
        class BadDriver:
            def quit(self):
                raise RuntimeError("Browser crashed")

        quit_driver_safely(BadDriver())  # Should not raise

    def test_browser_session_init():
        """Test BrowserSession stores parameters."""
        session = BrowserSession(headless=False, timeout=60, extra_options=["--incognito"])
        assert session.headless is False
        assert session.timeout == 60
        assert session.extra_options == ["--incognito"]
        assert session._driver is None

    def test_browser_session_close_without_driver():
        """Test BrowserSession close when no driver has been created."""
        session = BrowserSession()
        session.close()  # Should not raise
        assert session._driver is None

    def test_browser_session_context_manager():
        """Test BrowserSession context manager protocol."""
        session = BrowserSession()
        assert hasattr(session, "__enter__")
        assert hasattr(session, "__exit__")
        # Enter returns self
        result = session.__enter__()
        assert result is session

    def test_create_chrome_driver_no_uc():
        """Test create_chrome_driver raises ImportError when UC not available."""
        if not UC_AVAILABLE:
            try:
                create_chrome_driver()
                assert False, "Should have raised ImportError"
            except ImportError as e:
                assert "undetected-chromedriver" in str(e)

    suite.run_test(
        test_name="UC_AVAILABLE is boolean",
        test_func=test_uc_available_is_bool,
        test_summary="Tests UC_AVAILABLE constant type",
        method_description="Checks that UC_AVAILABLE is a boolean",
        expected_outcome="UC_AVAILABLE is a bool",
    )
    suite.run_test(
        test_name="quit_driver_safely handles None",
        test_func=test_quit_driver_safely_none,
        test_summary="Tests quit_driver_safely with None input",
        method_description="Passes None to quit_driver_safely",
        expected_outcome="No exception raised",
    )
    suite.run_test(
        test_name="quit_driver_safely calls quit",
        test_func=test_quit_driver_safely_mock,
        test_summary="Tests quit_driver_safely calls quit on driver",
        method_description="Passes a mock driver to quit_driver_safely",
        expected_outcome="Driver.quit() is called",
    )
    suite.run_test(
        test_name="quit_driver_safely suppresses errors",
        test_func=test_quit_driver_safely_error,
        test_summary="Tests quit_driver_safely suppresses exceptions",
        method_description="Passes a driver that raises on quit",
        expected_outcome="No exception propagated",
    )
    suite.run_test(
        test_name="BrowserSession initialization",
        test_func=test_browser_session_init,
        test_summary="Tests BrowserSession stores parameters",
        method_description="Creates BrowserSession with custom params",
        expected_outcome="Parameters stored correctly",
    )
    suite.run_test(
        test_name="BrowserSession close without driver",
        test_func=test_browser_session_close_without_driver,
        test_summary="Tests BrowserSession close without active driver",
        method_description="Calls close on a fresh session",
        expected_outcome="No exception raised",
    )
    suite.run_test(
        test_name="BrowserSession context manager",
        test_func=test_browser_session_context_manager,
        test_summary="Tests BrowserSession context manager protocol",
        method_description="Verifies __enter__ and __exit__",
        expected_outcome="Enter returns self",
    )
    suite.run_test(
        test_name="create_chrome_driver without UC",
        test_func=test_create_chrome_driver_no_uc,
        test_summary="Tests ImportError when UC unavailable",
        method_description="Calls create_chrome_driver without UC",
        expected_outcome="ImportError raised with helpful message",
    )

    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)

if __name__ == "__main__":
    run_comprehensive_tests()
