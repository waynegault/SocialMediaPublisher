"""
Centralized API Client with Adaptive Rate Limiting.

Provides rate-limited wrappers for all external API calls:
- Google Gemini (generate_content)
- Google Imagen (generate_images)
- LinkedIn API
- HTTP requests (for web scraping, etc.)
- HTTP session factory with retry logic

Usage:
    from api_client import api_client

    # Rate-limited Gemini call
    response = api_client.gemini_generate(client, model, contents, config)

    # Rate-limited HTTP request
    response = api_client.http_get(url, headers, timeout)

    # Get a managed session with retry logic
    session = api_client.get_session("linkedin")
"""

import logging
import re
import requests
from requests.adapters import HTTPAdapter
from typing import Any, Optional
from urllib3.util.retry import Retry

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

        # Groq API - generous free tier (30 RPM, 14400 RPD)
        self.groq_limiter = AdaptiveRateLimiter(
            initial_fill_rate=0.5,  # 1 request per 2 seconds (conservative)
            min_fill_rate=0.1,  # Minimum: 1 per 10 seconds
            max_fill_rate=1.0,  # Maximum: 1 per second
            success_threshold=5,
            rate_limiter_429_backoff=0.5,
        )

        # OpenAI API - generous limits for paid tier
        self.openai_limiter = AdaptiveRateLimiter(
            initial_fill_rate=1.0,
            min_fill_rate=0.1,
            max_fill_rate=2.0,
            success_threshold=5,
            rate_limiter_429_backoff=0.5,
        )

        # Anthropic (Claude) API
        self.anthropic_limiter = AdaptiveRateLimiter(
            initial_fill_rate=0.5,
            min_fill_rate=0.1,
            max_fill_rate=1.0,
            success_threshold=5,
            rate_limiter_429_backoff=0.5,
        )

        # DeepSeek API
        self.deepseek_limiter = AdaptiveRateLimiter(
            initial_fill_rate=0.5,
            min_fill_rate=0.1,
            max_fill_rate=1.0,
            success_threshold=5,
            rate_limiter_429_backoff=0.5,
        )

        # Moonshot (Kimi) API
        self.moonshot_limiter = AdaptiveRateLimiter(
            initial_fill_rate=0.5,
            min_fill_rate=0.1,
            max_fill_rate=1.0,
            success_threshold=5,
            rate_limiter_429_backoff=0.5,
        )

        # Browser-based searches - very conservative (CAPTCHA prevention)
        # Used by linkedin_profile_lookup for browser-based searches
        self.browser_limiter = AdaptiveRateLimiter(
            initial_fill_rate=0.125,  # 1 request per 8 seconds
            min_fill_rate=0.05,  # Minimum: 1 per 20 seconds
            max_fill_rate=0.2,  # Maximum: 1 per 5 seconds
            success_threshold=3,
            rate_limiter_429_backoff=0.3,  # Aggressive backoff on CAPTCHA
        )

        # Session pool for connection reuse
        self._sessions: dict[str, requests.Session] = {}

        # Client singletons (lazy-initialized)
        self._groq_client: Any = None
        self._openai_client: Any = None
        self._anthropic_client: Any = None
        self._deepseek_client: Any = None
        self._moonshot_client: Any = None

    def get_groq_client(self) -> Any:
        """Get or create a Groq client singleton."""
        if self._groq_client is None:
            from config import Config

            if Config.GROQ_API_KEY:
                try:
                    from groq import Groq

                    self._groq_client = Groq(api_key=Config.GROQ_API_KEY)
                    logger.debug("Groq client initialized")
                except ImportError:
                    logger.warning("groq package not installed")
                except Exception as e:
                    logger.warning(f"Failed to initialize Groq client: {e}")
        return self._groq_client

    def get_openai_client(self) -> Any:
        """Get or create an OpenAI client singleton (for text generation)."""
        if self._openai_client is None:
            from config import Config

            if Config.OPENAI_API_KEY:
                try:
                    from openai import OpenAI

                    self._openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
                    logger.debug("OpenAI client initialized")
                except ImportError:
                    logger.warning("openai package not installed")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI client: {e}")
        return self._openai_client

    def get_anthropic_client(self) -> Any:
        """Get or create an Anthropic (Claude) client singleton."""
        if self._anthropic_client is None:
            from config import Config

            if Config.ANTHROPIC_API_KEY:
                try:
                    from anthropic import Anthropic

                    self._anthropic_client = Anthropic(
                        api_key=Config.ANTHROPIC_API_KEY
                    )
                    logger.debug("Anthropic client initialized")
                except ImportError:
                    logger.warning(
                        "anthropic package not installed (pip install anthropic)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize Anthropic client: {e}")
        return self._anthropic_client

    def get_deepseek_client(self) -> Any:
        """Get or create a DeepSeek client singleton (OpenAI-compatible)."""
        if self._deepseek_client is None:
            from config import Config

            if Config.DEEPSEEK_API_KEY:
                try:
                    from openai import OpenAI

                    self._deepseek_client = OpenAI(
                        api_key=Config.DEEPSEEK_API_KEY,
                        base_url="https://api.deepseek.com",
                    )
                    logger.debug("DeepSeek client initialized")
                except ImportError:
                    logger.warning("openai package not installed")
                except Exception as e:
                    logger.warning(f"Failed to initialize DeepSeek client: {e}")
        return self._deepseek_client

    def get_moonshot_client(self) -> Any:
        """Get or create a Moonshot (Kimi) client singleton (OpenAI-compatible)."""
        if self._moonshot_client is None:
            from config import Config

            if Config.MOONSHOT_API_KEY:
                try:
                    from openai import OpenAI

                    self._moonshot_client = OpenAI(
                        api_key=Config.MOONSHOT_API_KEY,
                        base_url="https://api.moonshot.cn/v1",
                    )
                    logger.debug("Moonshot client initialized")
                except ImportError:
                    logger.warning("openai package not installed")
                except Exception as e:
                    logger.warning(f"Failed to initialize Moonshot client: {e}")
        return self._moonshot_client

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        endpoint: str = "default",
        gemini_client: Any = None,
        gemini_model: Optional[str] = None,
        gemini_config: Optional[Any] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Unified text generation using the configured LLM_PROVIDER.

        Uses Config.LLM_PROVIDER to select the primary provider, with
        Gemini as the final fallback.

        Supported providers: gemini, groq, local, openai, claude, deepseek, kimi

        Args:
            prompt: The text prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            endpoint: Endpoint name for rate limiting/logging
            gemini_client: Gemini client (for Gemini provider / fallback)
            gemini_model: Gemini model name (for Gemini provider / fallback)
            gemini_config: Gemini config (for Gemini provider / fallback)
            system_prompt: Optional system-level instruction prepended to messages

        Returns:
            Generated text content

        Raises:
            Exception: If all providers fail
        """
        from config import Config

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # For Gemini, prepend system prompt to the user prompt since Gemini
        # uses a different API that doesn't have a system message role
        gemini_prompt = prompt
        if system_prompt:
            gemini_prompt = f"{system_prompt}\n\n{prompt}"

        # Determine primary provider from LLM_PROVIDER (with legacy flag compat)
        provider = getattr(Config, "LLM_PROVIDER", "").lower()
        if not provider:
            if Config.PREFER_LOCAL_LLM:
                provider = "local"
            elif Config.PREFER_GROQ:
                provider = "groq"
            else:
                provider = "gemini"

        errors: list[str] = []

        # --- Try the selected provider first ---
        if provider == "openai":
            client = self.get_openai_client()
            if client:
                try:
                    return self.openai_generate(
                        client=client, messages=messages,
                        max_tokens=max_tokens, temperature=temperature,
                        endpoint=endpoint,
                    )
                except Exception as e:
                    errors.append(f"OpenAI: {e}")
                    logger.warning(f"OpenAI failed, trying fallback: {e}")

        elif provider == "claude":
            client = self.get_anthropic_client()
            if client:
                try:
                    return self.anthropic_generate(
                        client=client, messages=messages,
                        max_tokens=max_tokens, temperature=temperature,
                        endpoint=endpoint,
                    )
                except Exception as e:
                    errors.append(f"Anthropic: {e}")
                    logger.warning(f"Anthropic (Claude) failed, trying fallback: {e}")

        elif provider == "deepseek":
            client = self.get_deepseek_client()
            if client:
                try:
                    return self.deepseek_generate(
                        client=client, messages=messages,
                        max_tokens=max_tokens, temperature=temperature,
                        endpoint=endpoint,
                    )
                except Exception as e:
                    errors.append(f"DeepSeek: {e}")
                    logger.warning(f"DeepSeek failed, trying fallback: {e}")

        elif provider == "kimi":
            client = self.get_moonshot_client()
            if client:
                try:
                    return self.moonshot_generate(
                        client=client, messages=messages,
                        max_tokens=max_tokens, temperature=temperature,
                        endpoint=endpoint,
                    )
                except Exception as e:
                    errors.append(f"Moonshot (Kimi): {e}")
                    logger.warning(f"Moonshot (Kimi) failed, trying fallback: {e}")

        elif provider == "groq":
            groq_client = self.get_groq_client()
            if groq_client:
                try:
                    return self.groq_generate(
                        client=groq_client, messages=messages,
                        max_tokens=max_tokens, temperature=temperature,
                        endpoint=endpoint,
                    )
                except Exception as e:
                    errors.append(f"Groq: {e}")
                    logger.warning(f"Groq failed, trying fallback: {e}")

        elif provider == "local":
            try:
                from openai import OpenAI as _OAI

                local_client = _OAI(
                    base_url=Config.LM_STUDIO_BASE_URL,
                    api_key="lm-studio",
                )
                return self.local_llm_generate(
                    client=local_client, messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                    endpoint=endpoint,
                )
            except Exception as e:
                errors.append(f"Local LLM: {e}")
                logger.warning(f"Local LLM failed, trying fallback: {e}")

        elif provider == "gemini":
            if gemini_client:
                try:
                    response = self.gemini_generate(
                        client=gemini_client,
                        model=gemini_model or Config.MODEL_TEXT,
                        contents=gemini_prompt,
                        config=gemini_config,
                        endpoint=endpoint,
                    )
                    return response.text if hasattr(response, "text") else str(response)
                except Exception as e:
                    errors.append(f"Gemini: {e}")
                    logger.warning(f"Gemini failed: {e}")

        # --- Fallback chain: Groq → Gemini ---
        if provider != "groq" and Config.GROQ_API_KEY:
            groq_client = self.get_groq_client()
            if groq_client:
                try:
                    return self.groq_generate(
                        client=groq_client, messages=messages,
                        max_tokens=max_tokens, temperature=temperature,
                        endpoint=endpoint,
                    )
                except Exception as e:
                    errors.append(f"Groq (fallback): {e}")

        if provider != "gemini" and gemini_client:
            try:
                response = self.gemini_generate(
                    client=gemini_client,
                    model=gemini_model or Config.MODEL_TEXT,
                    contents=gemini_prompt,
                    config=gemini_config,
                    endpoint=endpoint,
                )
                return response.text if hasattr(response, "text") else str(response)
            except Exception as e:
                errors.append(f"Gemini (fallback): {e}")

        error_details = "; ".join(errors) if errors else "no provider configured"
        raise RuntimeError(f"All LLM providers failed: {error_details}")

    # =========================================================================
    # Session Factory
    # =========================================================================

    def get_session(
        self,
        name: str = "default",
        retries: int = 3,
        backoff_factor: float = 1.0,
        status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
    ) -> requests.Session:
        """
        Get or create a named session with retry logic.

        Sessions are reused across calls with the same name, providing
        connection pooling benefits.

        Args:
            name: Session identifier (e.g., 'linkedin', 'rapidapi', 'scraping')
            retries: Number of retry attempts for failed requests
            backoff_factor: Multiplier for retry delays
            status_forcelist: HTTP status codes that trigger retries

        Returns:
            A requests.Session configured with retry logic
        """
        if name not in self._sessions:
            session = requests.Session()
            retry = Retry(
                total=retries,
                backoff_factor=backoff_factor,
                status_forcelist=list(status_forcelist),
                allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS"],
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._sessions[name] = session
            logger.debug(f"Created new session '{name}' with {retries} retries")
        return self._sessions[name]

    def close_session(self, name: str) -> None:
        """Close and remove a named session."""
        if name in self._sessions:
            self._sessions[name].close()
            del self._sessions[name]
            logger.debug(f"Closed session '{name}'")

    def close_all_sessions(self) -> None:
        """Close all managed sessions."""
        for name in list(self._sessions.keys()):
            self.close_session(name)

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
    # Groq API (Free cloud LLM)
    # =========================================================================

    def groq_generate(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        endpoint: str = "default",
    ) -> str:
        """
        Rate-limited Groq API call.

        Groq provides free, fast inference for open-source models like Llama.
        Free tier: 30 requests/min, 14,400 requests/day.

        Args:
            client: Groq client instance
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name (defaults to Config.GROQ_MODEL)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            endpoint: Endpoint name for rate limiting

        Returns:
            The text content from the response

        Raises:
            Exception: If the API request fails
        """
        from config import Config

        model_name = model or Config.GROQ_MODEL
        full_endpoint = f"groq_{endpoint}"

        # Wait according to rate limiter
        wait_time = self.groq_limiter.wait(endpoint=full_endpoint)
        if wait_time > 0.5:
            logger.debug(f"Groq rate limiter ({endpoint}): waited {wait_time:.1f}s")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = response.choices[0].message.content or ""
            self.groq_limiter.on_success(endpoint=full_endpoint)
            logger.debug(f"Groq response [{endpoint}]: {len(content)} chars")
            return content

        except Exception as e:
            if self._is_rate_limit_error(e):
                retry_after = self._parse_retry_after(str(e))
                self.groq_limiter.on_429_error(
                    endpoint=full_endpoint, retry_after=retry_after
                )
            raise

    # =========================================================================
    # Local LLM API (LM Studio compatible)
    # =========================================================================

    def local_llm_generate(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.1,
        timeout: int = 60,
        endpoint: str = "default",
    ) -> str:
        """Make a rate-limited request to a local LLM (LM Studio compatible).

        This provides a unified interface for all local LLM calls with:
        - Consistent error handling
        - Response content extraction
        - Logging

        Args:
            client: OpenAI-compatible client (e.g., from openai.OpenAI)
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name (defaults to Config.LM_STUDIO_MODEL)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            endpoint: Endpoint name for logging/metrics

        Returns:
            The text content from the LLM response

        Raises:
            Exception: If the LLM request fails
        """
        from config import Config

        model_name = model or Config.LM_STUDIO_MODEL

        logger.debug(f"Local LLM request [{endpoint}]: {len(messages)} messages")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )

            content = response.choices[0].message.content or ""
            logger.debug(f"Local LLM response [{endpoint}]: {len(content)} chars")
            return content

        except Exception as e:
            # Use debug for "no model loaded" since fallback to Groq is expected behavior
            if "No models loaded" in str(e):
                logger.debug(f"Local LLM has no model [{endpoint}]: using fallback")
            else:
                logger.warning(f"Local LLM request failed [{endpoint}]: {e}")
            raise

    # =========================================================================
    # OpenAI API (text generation)
    # =========================================================================

    def openai_generate(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        endpoint: str = "default",
    ) -> str:
        """Rate-limited OpenAI chat completion call.

        Args:
            client: OpenAI client instance
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name (defaults to Config.OPENAI_TEXT_MODEL)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            endpoint: Endpoint name for rate limiting

        Returns:
            The text content from the response
        """
        from config import Config

        model_name = model or Config.OPENAI_TEXT_MODEL
        full_endpoint = f"openai_{endpoint}"

        wait_time = self.openai_limiter.wait(endpoint=full_endpoint)
        if wait_time > 0.5:
            logger.debug(f"OpenAI rate limiter ({endpoint}): waited {wait_time:.1f}s")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = response.choices[0].message.content or ""
            self.openai_limiter.on_success(endpoint=full_endpoint)
            logger.debug(f"OpenAI response [{endpoint}]: {len(content)} chars")
            return content

        except Exception as e:
            if self._is_rate_limit_error(e):
                retry_after = self._parse_retry_after(str(e))
                self.openai_limiter.on_429_error(
                    endpoint=full_endpoint, retry_after=retry_after
                )
            raise

    # =========================================================================
    # Anthropic (Claude) API
    # =========================================================================

    def anthropic_generate(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        endpoint: str = "default",
    ) -> str:
        """Rate-limited Anthropic (Claude) API call.

        Uses the Anthropic SDK messages API (not OpenAI-compatible).

        Args:
            client: Anthropic client instance
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name (defaults to Config.ANTHROPIC_MODEL)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            endpoint: Endpoint name for rate limiting

        Returns:
            The text content from the response
        """
        from config import Config

        model_name = model or Config.ANTHROPIC_MODEL
        full_endpoint = f"anthropic_{endpoint}"

        wait_time = self.anthropic_limiter.wait(endpoint=full_endpoint)
        if wait_time > 0.5:
            logger.debug(
                f"Anthropic rate limiter ({endpoint}): waited {wait_time:.1f}s"
            )

        try:
            response = client.messages.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = response.content[0].text if response.content else ""
            self.anthropic_limiter.on_success(endpoint=full_endpoint)
            logger.debug(f"Anthropic response [{endpoint}]: {len(content)} chars")
            return content

        except Exception as e:
            if self._is_rate_limit_error(e):
                retry_after = self._parse_retry_after(str(e))
                self.anthropic_limiter.on_429_error(
                    endpoint=full_endpoint, retry_after=retry_after
                )
            raise

    # =========================================================================
    # DeepSeek API (OpenAI-compatible)
    # =========================================================================

    def deepseek_generate(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        endpoint: str = "default",
    ) -> str:
        """Rate-limited DeepSeek API call (OpenAI-compatible).

        Args:
            client: OpenAI-compatible client pointing to DeepSeek API
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name (defaults to Config.DEEPSEEK_MODEL)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            endpoint: Endpoint name for rate limiting

        Returns:
            The text content from the response
        """
        from config import Config

        model_name = model or Config.DEEPSEEK_MODEL
        full_endpoint = f"deepseek_{endpoint}"

        wait_time = self.deepseek_limiter.wait(endpoint=full_endpoint)
        if wait_time > 0.5:
            logger.debug(
                f"DeepSeek rate limiter ({endpoint}): waited {wait_time:.1f}s"
            )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = response.choices[0].message.content or ""
            self.deepseek_limiter.on_success(endpoint=full_endpoint)
            logger.debug(f"DeepSeek response [{endpoint}]: {len(content)} chars")
            return content

        except Exception as e:
            if self._is_rate_limit_error(e):
                retry_after = self._parse_retry_after(str(e))
                self.deepseek_limiter.on_429_error(
                    endpoint=full_endpoint, retry_after=retry_after
                )
            raise

    # =========================================================================
    # Moonshot (Kimi K2) API (OpenAI-compatible)
    # =========================================================================

    def moonshot_generate(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        endpoint: str = "default",
    ) -> str:
        """Rate-limited Moonshot (Kimi) API call (OpenAI-compatible).

        Args:
            client: OpenAI-compatible client pointing to Moonshot API
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name (defaults to Config.MOONSHOT_MODEL)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            endpoint: Endpoint name for rate limiting

        Returns:
            The text content from the response
        """
        from config import Config

        model_name = model or Config.MOONSHOT_MODEL
        full_endpoint = f"moonshot_{endpoint}"

        wait_time = self.moonshot_limiter.wait(endpoint=full_endpoint)
        if wait_time > 0.5:
            logger.debug(
                f"Moonshot rate limiter ({endpoint}): waited {wait_time:.1f}s"
            )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = response.choices[0].message.content or ""
            self.moonshot_limiter.on_success(endpoint=full_endpoint)
            logger.debug(f"Moonshot response [{endpoint}]: {len(content)} chars")
            return content

        except Exception as e:
            if self._is_rate_limit_error(e):
                retry_after = self._parse_retry_after(str(e))
                self.moonshot_limiter.on_429_error(
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

    def http_head(
        self,
        url: str,
        headers: Optional[dict] = None,
        timeout: int = 10,
        allow_redirects: bool = True,
        endpoint: str = "default",
    ) -> requests.Response:
        """
        Rate-limited HTTP HEAD request.

        Args:
            url: URL to check
            headers: Request headers
            timeout: Request timeout
            allow_redirects: Follow redirects
            endpoint: Logical endpoint name for rate limiting

        Returns:
            requests.Response
        """
        return self.http_request(
            method="HEAD",
            url=url,
            headers=headers,
            timeout=timeout,
            allow_redirects=allow_redirects,
            endpoint=endpoint,
        )

    def http_put(
        self,
        url: str,
        headers: Optional[dict] = None,
        data: Optional[Any] = None,
        json: Optional[dict] = None,
        timeout: int = 10,
        endpoint: str = "default",
    ) -> requests.Response:
        """
        Rate-limited HTTP PUT request.

        Args:
            url: URL to put to
            headers: Request headers
            data: Raw data or binary content
            json: JSON body
            timeout: Request timeout
            endpoint: Logical endpoint name for rate limiting

        Returns:
            requests.Response
        """
        return self.http_request(
            method="PUT",
            url=url,
            headers=headers,
            data=data,
            json=json,
            timeout=timeout,
            endpoint=endpoint,
        )

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
            "browser": self.browser_limiter.get_metrics(),
        }

    def browser_wait(self, endpoint: str = "search") -> float:
        """
        Wait according to browser rate limiter.

        Used by browser-based search operations that need CAPTCHA prevention.
        Returns the actual wait time.

        Args:
            endpoint: Logical endpoint (e.g., 'google', 'bing', 'linkedin')

        Returns:
            Actual time waited in seconds
        """
        full_endpoint = f"browser_{endpoint}"
        return self.browser_limiter.wait(endpoint=full_endpoint)

    def browser_on_success(self, endpoint: str = "search") -> None:
        """Mark a successful browser operation."""
        full_endpoint = f"browser_{endpoint}"
        self.browser_limiter.on_success(endpoint=full_endpoint)

    def browser_on_captcha(self, endpoint: str = "search") -> None:
        """Mark a CAPTCHA/rate limit event for browser operations."""
        full_endpoint = f"browser_{endpoint}"
        self.browser_limiter.on_429_error(endpoint=full_endpoint)

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


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests() -> bool:
    """Create unit tests for api_client module."""
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from test_framework import TestSuite

    from test_framework import TestSuite

    suite = TestSuite("API Client Tests", "api_client.py")
    suite.start_suite()

    def test_api_client_creation():
        """Test RateLimitedAPIClient creation."""
        client = RateLimitedAPIClient()
        assert client.gemini_limiter is not None
        assert client.imagen_limiter is not None
        assert client.linkedin_limiter is not None
        assert client.http_limiter is not None

    def test_parse_retry_after_found():
        """Test parsing retry-after from error message."""
        client = RateLimitedAPIClient()
        result = client._parse_retry_after("Rate limit exceeded. Retry-After: 60")
        assert result == 60.0

    def test_parse_retry_after_not_found():
        """Test parsing when retry-after is missing."""
        client = RateLimitedAPIClient()
        result = client._parse_retry_after("Some other error")
        assert result == 30.0  # Default

    def test_is_rate_limit_error_429():
        """Test detection of 429 error."""
        client = RateLimitedAPIClient()
        error = Exception("HTTP 429 Too Many Requests")
        assert client._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_quota():
        """Test detection of quota error."""
        client = RateLimitedAPIClient()
        error = Exception("RESOURCE_EXHAUSTED: Quota exceeded")
        assert client._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_not_rate_limit():
        """Test non-rate-limit error."""
        client = RateLimitedAPIClient()
        error = Exception("Connection timeout")
        assert client._is_rate_limit_error(error) is False

    def test_get_all_metrics():
        """Test getting all rate limiter metrics."""
        client = RateLimitedAPIClient()
        metrics = client.get_all_metrics()
        assert "gemini" in metrics
        assert "imagen" in metrics
        assert "linkedin" in metrics
        assert "http" in metrics

    def test_gemini_limiter_config():
        """Test Gemini rate limiter configuration."""
        client = RateLimitedAPIClient()
        assert client.gemini_limiter.fill_rate == 1.0

    def test_imagen_limiter_config():
        """Test Imagen rate limiter configuration."""
        client = RateLimitedAPIClient()
        assert client.imagen_limiter.fill_rate == 0.5

    def test_linkedin_limiter_config():
        """Test LinkedIn rate limiter configuration."""
        client = RateLimitedAPIClient()
        assert client.linkedin_limiter.fill_rate == 0.5

    def test_http_limiter_config():
        """Test HTTP rate limiter configuration."""
        client = RateLimitedAPIClient()
        assert client.http_limiter.fill_rate == 2.0

    def test_global_api_client_exists():
        """Test that global api_client is created."""
        assert api_client is not None
        assert isinstance(api_client, RateLimitedAPIClient)

    def test_log_metrics_no_error():
        """Test log_metrics doesn't raise."""
        client = RateLimitedAPIClient()
        client.log_metrics()  # Should not raise

    suite.run_test(
        test_name="API client creation",
        test_func=test_api_client_creation,
        test_summary="Tests API client creation functionality",
        method_description="Calls RateLimitedAPIClient and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Parse retry-after found",
        test_func=test_parse_retry_after_found,
        test_summary="Tests Parse retry-after found functionality",
        method_description="Calls RateLimitedAPIClient and verifies the result",
        expected_outcome="Function returns the expected value",
    )
    suite.run_test(
        test_name="Parse retry-after not found",
        test_func=test_parse_retry_after_not_found,
        test_summary="Tests Parse retry-after not found functionality",
        method_description="Calls RateLimitedAPIClient and verifies the result",
        expected_outcome="Function returns the expected value",
    )
    suite.run_test(
        test_name="Is rate limit error - 429",
        test_func=test_is_rate_limit_error_429,
        test_summary="Tests Is rate limit error with 429 scenario",
        method_description="Calls RateLimitedAPIClient and verifies the result",
        expected_outcome="Function handles invalid input appropriately",
    )
    suite.run_test(
        test_name="Is rate limit error - quota",
        test_func=test_is_rate_limit_error_quota,
        test_summary="Tests Is rate limit error with quota scenario",
        method_description="Calls RateLimitedAPIClient and verifies the result",
        expected_outcome="Function handles invalid input appropriately",
    )
    suite.run_test(
        test_name="Is not rate limit error",
        test_func=test_is_rate_limit_error_not_rate_limit,
        test_summary="Tests Is not rate limit error functionality",
        method_description="Calls RateLimitedAPIClient and verifies the result",
        expected_outcome="Function handles invalid input appropriately",
    )
    suite.run_test(
        test_name="Get all metrics",
        test_func=test_get_all_metrics,
        test_summary="Tests Get all metrics functionality",
        method_description="Calls RateLimitedAPIClient and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Gemini limiter config",
        test_func=test_gemini_limiter_config,
        test_summary="Tests Gemini limiter config functionality",
        method_description="Calls RateLimitedAPIClient and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Imagen limiter config",
        test_func=test_imagen_limiter_config,
        test_summary="Tests Imagen limiter config functionality",
        method_description="Calls RateLimitedAPIClient and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="LinkedIn limiter config",
        test_func=test_linkedin_limiter_config,
        test_summary="Tests LinkedIn limiter config functionality",
        method_description="Calls RateLimitedAPIClient and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="HTTP limiter config",
        test_func=test_http_limiter_config,
        test_summary="Tests HTTP limiter config functionality",
        method_description="Calls RateLimitedAPIClient and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Global api_client exists",
        test_func=test_global_api_client_exists,
        test_summary="Tests Global api client exists functionality",
        method_description="Invokes the function under test and validates behavior",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Log metrics no error",
        test_func=test_log_metrics_no_error,
        test_summary="Tests Log metrics no error functionality",
        method_description="Calls RateLimitedAPIClient and verifies the result",
        expected_outcome="Function handles empty or missing input gracefully",
    )

    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
