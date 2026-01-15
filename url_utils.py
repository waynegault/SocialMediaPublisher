"""URL validation and parsing utilities.

This module provides centralized URL handling functions used across the codebase.
Domain credibility logic remains in domain_credibility.py; this module focuses
on URL structure, validation, and normalization.
"""

import logging
import re
from urllib.parse import urlparse, urljoin

import requests

from config import Config
from api_client import api_client

logger = logging.getLogger(__name__)


def validate_url_format(url: str) -> bool:
    """
    Validate URL structure without making network requests.

    Args:
        url: URL string to validate

    Returns:
        True if URL has valid format (scheme and netloc), False otherwise
    """
    if not url:
        return False

    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return False
        if parsed.scheme not in ("http", "https"):
            return False
        return True
    except Exception:
        return False


def validate_url_accessible(
    url: str,
    timeout: int = 5,
    use_get_fallback: bool = True,
) -> bool:
    """
    Check if a URL is accessible via HTTP request.

    Args:
        url: URL to check
        timeout: Request timeout in seconds
        use_get_fallback: If True, fall back to GET when HEAD returns 405

    Returns:
        True if URL is accessible or times out (site might be slow),
        False on connection errors or 4xx/5xx status codes
    """
    if not validate_url_format(url):
        return False

    try:
        # Try HEAD first (faster)
        response = api_client.http_request(
            method="HEAD",
            url=url,
            timeout=timeout,
            allow_redirects=True,
            endpoint="url_validation",
        )

        # Some servers don't support HEAD, try GET on 405
        if response.status_code == 405 and use_get_fallback:
            response = api_client.http_get(
                url=url,
                timeout=timeout,
                allow_redirects=True,
                endpoint="url_validation",
            )

        if response.status_code >= 400:
            logger.debug(f"URL returned {response.status_code}: {url}")
            return False

        return True

    except requests.exceptions.Timeout:
        # Timeout is acceptable - site might be slow but exists
        logger.debug(f"URL timeout (accepting anyway): {url}")
        return True

    except requests.exceptions.RequestException as e:
        # Connection errors, DNS failures, etc. - reject the URL
        logger.debug(f"URL validation failed ({type(e).__name__}): {url}")
        return False


def validate_url(url: str) -> bool:
    """
    Validate a URL format and optionally check accessibility.

    This is the main entry point for URL validation - it checks format
    and optionally accessibility based on Config.VALIDATE_SOURCE_URLS.

    Args:
        url: URL to validate

    Returns:
        True if URL is valid (and accessible if checking is enabled)
    """
    if not validate_url_format(url):
        return False

    if Config.VALIDATE_SOURCE_URLS:
        return validate_url_accessible(url)

    return True


def resolve_relative_url(relative_url: str, base_url: str) -> str:
    """
    Resolve a relative URL against a base URL.

    Handles:
    - Protocol-relative URLs (//example.com/path)
    - Absolute paths (/path/to/resource)
    - Relative paths (path/to/resource)
    - Full URLs (returned as-is)

    Args:
        relative_url: The relative or absolute URL to resolve
        base_url: The base URL for resolution

    Returns:
        Fully qualified URL string
    """
    if not relative_url:
        return base_url

    # Protocol-relative URL
    if relative_url.startswith("//"):
        return "https:" + relative_url

    # Already absolute
    if relative_url.startswith("http://") or relative_url.startswith("https://"):
        return relative_url

    # Use urljoin for proper resolution
    return urljoin(base_url, relative_url)


def get_base_url(url: str) -> str:
    """
    Extract base URL (scheme + netloc) from a full URL.

    Args:
        url: Full URL string

    Returns:
        Base URL like "https://example.com"
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
        return ""
    except Exception:
        return ""


def extract_path_keywords(url: str) -> set[str]:
    """
    Extract meaningful keywords from a URL path.

    E.g., "https://news.rice.edu/news/2026/researchers-unlock-catalyst-behavior"
    -> {"researchers", "unlock", "catalyst", "behavior"}

    Args:
        url: URL to extract keywords from

    Returns:
        Set of lowercase keywords from the path
    """
    if not url:
        return set()

    try:
        parsed = urlparse(url)
        path = parsed.path

        # Replace common separators with spaces
        path = re.sub(r"[-_/.]", " ", path)

        # Remove numbers (years, IDs, etc.)
        path = re.sub(r"\b\d+\b", "", path)

        # Split into words
        words = path.lower().split()

        # Remove very short words and common URL fragments
        url_stopwords = {
            "www",
            "com",
            "org",
            "net",
            "edu",
            "html",
            "htm",
            "php",
            "asp",
            "aspx",
            "jsp",
            "news",
            "article",
            "articles",
            "post",
            "posts",
            "blog",
            "index",
            "page",
            "category",
            "tag",
            "tags",
            "id",
            "view",
            "print",
            "content",
            "story",
            "stories",
        }

        return {word for word in words if len(word) > 2 and word not in url_stopwords}

    except Exception:
        return set()


def validate_linkedin_url(url: str, url_type: str = "profile") -> bool:
    """
    Validate LinkedIn URL format (without HTTP request).

    Args:
        url: LinkedIn URL to validate
        url_type: Type of URL - "profile", "company", or "school"

    Returns:
        True if URL appears to be a valid LinkedIn URL of the specified type
    """
    if not url:
        return False

    patterns = {
        "profile": r"linkedin\.com/in/([\w\-]+)",
        "company": r"linkedin\.com/company/([\w\-]+)",
        "school": r"linkedin\.com/school/([\w\-]+)",
    }

    pattern = patterns.get(url_type, patterns["profile"])
    match = re.search(pattern, url)

    if not match:
        return False

    slug = match.group(1)

    # Basic format validation
    if len(slug) < 2 or len(slug) > 100:
        return False

    # Reject common error page slugs
    invalid_slugs = {"login", "authwall", "error", "404", "unavailable", "uas"}
    if slug.lower() in invalid_slugs:
        return False

    return True


# ============================================================================
# Module Tests
# ============================================================================


def _create_module_tests():
    """Create tests for this module. Call from run_tests.py."""
    from test_framework import TestSuite

    suite = TestSuite("URL Utilities")

    # validate_url_format tests
    suite.add_test(
        "validate_url_format_empty",
        lambda: not validate_url_format(""),
    )

    suite.add_test(
        "validate_url_format_no_scheme",
        lambda: not validate_url_format("example.com/path"),
    )

    suite.add_test(
        "validate_url_format_invalid_scheme",
        lambda: not validate_url_format("ftp://example.com"),
    )

    suite.add_test(
        "validate_url_format_valid_http",
        lambda: validate_url_format("http://example.com/path"),
    )

    suite.add_test(
        "validate_url_format_valid_https",
        lambda: validate_url_format("https://example.com/path?query=1"),
    )

    # get_base_url tests
    suite.add_test(
        "get_base_url_full_url",
        lambda: get_base_url("https://news.example.com/path/article")
        == "https://news.example.com",
    )

    suite.add_test(
        "get_base_url_empty",
        lambda: get_base_url("") == "",
    )

    # resolve_relative_url tests
    suite.add_test(
        "resolve_relative_url_protocol_relative",
        lambda: resolve_relative_url(
            "//cdn.example.com/image.jpg", "https://example.com"
        )
        == "https://cdn.example.com/image.jpg",
    )

    suite.add_test(
        "resolve_relative_url_absolute_path",
        lambda: resolve_relative_url(
            "/images/photo.jpg", "https://example.com/news/article"
        )
        == "https://example.com/images/photo.jpg",
    )

    suite.add_test(
        "resolve_relative_url_relative_path",
        lambda: resolve_relative_url("photo.jpg", "https://example.com/news/")
        == "https://example.com/news/photo.jpg",
    )

    suite.add_test(
        "resolve_relative_url_full_url",
        lambda: resolve_relative_url(
            "https://other.com/image.jpg", "https://example.com"
        )
        == "https://other.com/image.jpg",
    )

    # extract_path_keywords tests
    suite.add_test(
        "extract_path_keywords_research_url",
        lambda: {"researchers", "unlock", "catalyst", "behavior"}
        <= extract_path_keywords(
            "https://news.rice.edu/news/2026/researchers-unlock-catalyst-behavior"
        ),
    )

    suite.add_test(
        "extract_path_keywords_empty",
        lambda: extract_path_keywords("") == set(),
    )

    # validate_linkedin_url tests
    suite.add_test(
        "validate_linkedin_url_valid_profile",
        lambda: validate_linkedin_url("https://linkedin.com/in/john-doe-12345"),
    )

    suite.add_test(
        "validate_linkedin_url_invalid_profile",
        lambda: not validate_linkedin_url("https://linkedin.com/in/login"),
    )

    suite.add_test(
        "validate_linkedin_url_valid_company",
        lambda: validate_linkedin_url(
            "https://linkedin.com/company/acme-corp", url_type="company"
        ),
    )

    suite.add_test(
        "validate_linkedin_url_empty",
        lambda: not validate_linkedin_url(""),
    )

    return suite
