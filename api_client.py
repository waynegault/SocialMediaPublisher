"""
Centralized API Client with Adaptive Rate Limiting.

Provides rate-limited wrappers for all external API calls:
- Google Gemini (generate_content)
- Google Imagen (generate_images)
- LinkedIn API
- HTTP requests (for web scraping, etc.)

Usage:
    from api_client import api_client

    # Rate-limited Gemini call
    response = api_client.gemini_generate(client, model, contents, config)

    # Rate-limited HTTP request
    response = api_client.http_get(url, headers, timeout)
"""

import logging
import re
import requests
from typing import Any, Optional

from rate_limiter import AdaptiveRateLimiter

logger = logging.getLogger(__name__)


class RateLimitedAPIClient:
    """Centralized API client with per-endpoint adaptive rate limiting."""

    def __init__(self):
        """Initialize rate limiters for each API category."""
        # Gemini API - generous limits (60 RPM for free tier)
        self.gemini_limiter = AdaptiveRateLimiter(
            initial_fill_rate=1.0,  # 1 request per second
            min_fill_rate=0.1,  # Minimum: 1 per 10 seconds
            max_fill_rate=2.0,  # Maximum: 2 per second
            success_threshold=5,
            rate_limiter_429_backoff=0.5,
        )

        # Imagen API - stricter limits (image generation is expensive)
        self.imagen_limiter = AdaptiveRateLimiter(
            initial_fill_rate=0.5,  # 1 request per 2 seconds
            min_fill_rate=0.1,  # Minimum: 1 per 10 seconds
            max_fill_rate=1.0,  # Maximum: 1 per second
            success_threshold=3,
            rate_limiter_429_backoff=0.3,  # More aggressive backoff
        )

        # LinkedIn API - conservative (strict rate limits)
        self.linkedin_limiter = AdaptiveRateLimiter(
            initial_fill_rate=0.5,  # 1 request per 2 seconds
            min_fill_rate=0.1,  # Minimum: 1 per 10 seconds
            max_fill_rate=1.0,  # Maximum: 1 per second
            success_threshold=5,
            rate_limiter_429_backoff=0.5,
        )

        # HTTP requests (web scraping) - moderate
        self.http_limiter = AdaptiveRateLimiter(
            initial_fill_rate=2.0,  # 2 requests per second
            min_fill_rate=0.2,  # Minimum: 1 per 5 seconds
            max_fill_rate=5.0,  # Maximum: 5 per second
            success_threshold=10,
            rate_limiter_429_backoff=0.5,
        )

    def _parse_retry_after(self, error_msg: str) -> float:
        """Extract retry-after value from error message if present."""
        match = re.search(r"retry[- ]?after[:\s]+(\d+)", error_msg.lower())
        if match:
            return float(match.group(1))
        return 30.0  # Default penalty

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an exception indicates a rate limit error."""
        error_msg = str(error).lower()
        return (
            "429" in error_msg
            or "resource_exhausted" in error_msg
            or "quota" in error_msg
            or "rate limit" in error_msg
            or "too many requests" in error_msg
        )

    # =========================================================================
    # Gemini API
    # =========================================================================

    def gemini_generate(
        self,
        client: Any,
        model: str,
        contents: Any,
        config: Optional[Any] = None,
        endpoint: str = "default",
    ) -> Any:
        """
        Rate-limited Gemini generate_content call.

        Args:
            client: google.genai.Client instance
            model: Model name (e.g., 'gemini-2.0-flash')
            contents: Prompt or content to generate from
            config: Optional GenerateContentConfig
            endpoint: Logical endpoint name for rate limiting (e.g., 'search', 'verify')

        Returns:
            GenerateContentResponse from Gemini
        """
        full_endpoint = f"gemini_{endpoint}"

        # Wait according to rate limiter
        wait_time = self.gemini_limiter.wait(endpoint=full_endpoint)
        if wait_time > 0.5:
            logger.debug(f"Gemini rate limiter ({endpoint}): waited {wait_time:.1f}s")

        try:
            if config:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            else:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                )
            self.gemini_limiter.on_success(endpoint=full_endpoint)
            return response

        except Exception as e:
            if self._is_rate_limit_error(e):
                retry_after = self._parse_retry_after(str(e))
                self.gemini_limiter.on_429_error(
                    endpoint=full_endpoint, retry_after=retry_after
                )
            raise

    # =========================================================================
    # Imagen API
    # =========================================================================

    def imagen_generate(
        self,
        client: Any,
        model: str,
        prompt: str,
        config: Any,
    ) -> Any:
        """
        Rate-limited Imagen generate_images call.

        Args:
            client: google.genai.Client instance
            model: Model name (e.g., 'imagen-4.0-generate-001')
            prompt: Image prompt
            config: GenerateImagesConfig

        Returns:
            GenerateImagesResponse from Imagen
        """
        wait_time = self.imagen_limiter.wait(endpoint="imagen")
        if wait_time > 0.5:
            logger.debug(f"Imagen rate limiter: waited {wait_time:.1f}s")

        try:
            response = client.models.generate_images(
                model=model,
                prompt=prompt,
                config=config,
            )
            self.imagen_limiter.on_success(endpoint="imagen")
            return response

        except Exception as e:
            if self._is_rate_limit_error(e):
                retry_after = self._parse_retry_after(str(e))
                self.imagen_limiter.on_429_error(
                    endpoint="imagen", retry_after=retry_after
                )
            raise

    # =========================================================================
    # LinkedIn API
    # =========================================================================

    def linkedin_request(
        self,
        method: str,
        url: str,
        headers: Optional[dict] = None,
        json: Optional[dict] = None,
        data: Optional[Any] = None,
        params: Optional[dict] = None,
        timeout: int = 30,
        endpoint: str = "default",
    ) -> requests.Response:
        """
        Rate-limited LinkedIn API request.

        Args:
            method: HTTP method ('GET', 'POST', 'PUT', 'DELETE')
            url: Full URL
            headers: Request headers
            json: JSON body
            data: Raw data body
            params: URL query parameters
            timeout: Request timeout
            endpoint: Logical endpoint name (e.g., 'publish', 'analytics')

        Returns:
            requests.Response
        """
        full_endpoint = f"linkedin_{endpoint}"

        wait_time = self.linkedin_limiter.wait(endpoint=full_endpoint)
        if wait_time > 0.5:
            logger.debug(f"LinkedIn rate limiter ({endpoint}): waited {wait_time:.1f}s")

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                data=data,
                params=params,
                timeout=timeout,
            )

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 60))
                self.linkedin_limiter.on_429_error(
                    endpoint=full_endpoint, retry_after=retry_after
                )
                raise requests.exceptions.HTTPError(
                    f"429 Too Many Requests (retry after {retry_after}s)"
                )

            self.linkedin_limiter.on_success(endpoint=full_endpoint)
            return response

        except requests.exceptions.HTTPError:
            raise
        except Exception as e:
            if self._is_rate_limit_error(e):
                self.linkedin_limiter.on_429_error(
                    endpoint=full_endpoint, retry_after=30
                )
            raise

    # =========================================================================
    # General HTTP Requests
    # =========================================================================

    def http_get(
        self,
        url: str,
        headers: Optional[dict] = None,
        timeout: int = 10,
        allow_redirects: bool = True,
        endpoint: str = "default",
    ) -> requests.Response:
        """
        Rate-limited HTTP GET request.

        Args:
            url: URL to fetch
            headers: Request headers
            timeout: Request timeout
            allow_redirects: Follow redirects
            endpoint: Logical endpoint name for rate limiting

        Returns:
            requests.Response
        """
        full_endpoint = f"http_{endpoint}"

        wait_time = self.http_limiter.wait(endpoint=full_endpoint)
        if wait_time > 1.0:
            logger.debug(f"HTTP rate limiter ({endpoint}): waited {wait_time:.1f}s")

        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=allow_redirects,
            )

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 30))
                self.http_limiter.on_429_error(
                    endpoint=full_endpoint, retry_after=retry_after
                )

            self.http_limiter.on_success(endpoint=full_endpoint)
            return response

        except Exception as e:
            if self._is_rate_limit_error(e):
                self.http_limiter.on_429_error(endpoint=full_endpoint, retry_after=30)
            raise

    def http_post(
        self,
        url: str,
        headers: Optional[dict] = None,
        json: Optional[dict] = None,
        data: Optional[Any] = None,
        timeout: int = 30,
        endpoint: str = "default",
    ) -> requests.Response:
        """
        Rate-limited HTTP POST request.

        Args:
            url: URL to post to
            headers: Request headers
            json: JSON body
            data: Form data or raw data
            timeout: Request timeout
            endpoint: Logical endpoint name

        Returns:
            requests.Response
        """
        full_endpoint = f"http_{endpoint}"

        wait_time = self.http_limiter.wait(endpoint=full_endpoint)
        if wait_time > 1.0:
            logger.debug(f"HTTP rate limiter ({endpoint}): waited {wait_time:.1f}s")

        try:
            response = requests.post(
                url,
                headers=headers,
                json=json,
                data=data,
                timeout=timeout,
            )

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 30))
                self.http_limiter.on_429_error(
                    endpoint=full_endpoint, retry_after=retry_after
                )

            self.http_limiter.on_success(endpoint=full_endpoint)
            return response

        except Exception as e:
            if self._is_rate_limit_error(e):
                self.http_limiter.on_429_error(endpoint=full_endpoint, retry_after=30)
            raise

    def http_request(
        self,
        method: str,
        url: str,
        headers: Optional[dict] = None,
        json: Optional[dict] = None,
        data: Optional[Any] = None,
        params: Optional[dict] = None,
        timeout: int = 10,
        allow_redirects: bool = True,
        endpoint: str = "default",
    ) -> requests.Response:
        """
        Rate-limited HTTP request with any method (HEAD, GET, POST, etc).

        Args:
            method: HTTP method ('GET', 'POST', 'HEAD', 'PUT', 'DELETE', etc.)
            url: URL to request
            headers: Request headers
            json: JSON body (for POST/PUT)
            data: Form data or raw data
            params: URL query parameters
            timeout: Request timeout
            allow_redirects: Follow redirects
            endpoint: Logical endpoint name for rate limiting

        Returns:
            requests.Response
        """
        full_endpoint = f"http_{endpoint}"

        wait_time = self.http_limiter.wait(endpoint=full_endpoint)
        if wait_time > 1.0:
            logger.debug(f"HTTP rate limiter ({endpoint}): waited {wait_time:.1f}s")

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                data=data,
                params=params,
                timeout=timeout,
                allow_redirects=allow_redirects,
            )

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 30))
                self.http_limiter.on_429_error(
                    endpoint=full_endpoint, retry_after=retry_after
                )

            self.http_limiter.on_success(endpoint=full_endpoint)
            return response

        except Exception as e:
            if self._is_rate_limit_error(e):
                self.http_limiter.on_429_error(endpoint=full_endpoint, retry_after=30)
            raise

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_all_metrics(self) -> dict:
        """Get metrics from all rate limiters."""
        return {
            "gemini": self.gemini_limiter.get_metrics(),
            "imagen": self.imagen_limiter.get_metrics(),
            "linkedin": self.linkedin_limiter.get_metrics(),
            "http": self.http_limiter.get_metrics(),
        }

    def log_metrics(self) -> None:
        """Log current rate limiter metrics."""
        metrics = self.get_all_metrics()
        for name, m in metrics.items():
            if m.total_requests > 0:
                logger.info(
                    f"Rate limiter [{name}]: {m.total_requests} requests, "
                    f"{m.error_429_count} 429s, rate: {m.current_fill_rate:.2f} req/s"
                )


# Global singleton instance
api_client = RateLimitedAPIClient()
