"""Async Utilities for Social Media Publisher.

This module provides async/await patterns for improved performance.

TASK 4.1: Async/Await Pattern Throughout

Features:
- Async HTTP client using aiohttp
- Concurrent story processing
- Backward compatibility with sync wrappers
- Async context managers

This module enables gradual migration to async without breaking existing code.
"""

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar("T")
P = ParamSpec("P")


# =============================================================================
# Async HTTP Client
# =============================================================================

# Optional aiohttp import
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not installed - pip install aiohttp for async HTTP")


@dataclass
class AsyncHTTPResponse:
    """Response from async HTTP request."""

    status: int
    headers: dict[str, str]
    text: str
    json_data: Optional[Any] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """Check if response was successful."""
        return 200 <= self.status < 300


class AsyncHTTPClient:
    """Async HTTP client using aiohttp.

    Provides async HTTP methods with retry logic and connection pooling.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_connections: int = 20,
        retry_attempts: int = 3,
    ) -> None:
        """Initialize the async HTTP client.

        Args:
            timeout: Request timeout in seconds
            max_connections: Maximum concurrent connections
            retry_attempts: Number of retry attempts on failure
        """
        self.timeout = timeout
        self.max_connections = max_connections
        self.retry_attempts = retry_attempts
        self._session: Optional[Any] = None

    async def _get_session(self) -> Any:
        """Get or create the aiohttp session."""
        if not AIOHTTP_AVAILABLE or aiohttp is None:
            raise RuntimeError("aiohttp not installed - pip install aiohttp")

        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=self.max_connections)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
        return self._session

    async def get(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, str]] = None,
    ) -> AsyncHTTPResponse:
        """Perform async GET request.

        Args:
            url: URL to fetch
            headers: Optional request headers
            params: Optional query parameters

        Returns:
            AsyncHTTPResponse with response data
        """
        return await self._request("GET", url, headers=headers, params=params)

    async def post(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        json_data: Optional[Any] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> AsyncHTTPResponse:
        """Perform async POST request.

        Args:
            url: URL to post to
            headers: Optional request headers
            json_data: Optional JSON body
            data: Optional form data

        Returns:
            AsyncHTTPResponse with response data
        """
        return await self._request(
            "POST", url, headers=headers, json_data=json_data, data=data
        )

    async def _request(
        self,
        method: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, str]] = None,
        json_data: Optional[Any] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> AsyncHTTPResponse:
        """Perform async HTTP request with retry logic."""
        session = await self._get_session()

        last_error: Optional[str] = None

        for attempt in range(self.retry_attempts):
            try:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json_data,
                    data=data,
                ) as response:
                    text = await response.text()

                    # Try to parse JSON
                    response_json = None
                    try:
                        response_json = await response.json()
                    except Exception:
                        pass

                    return AsyncHTTPResponse(
                        status=response.status,
                        headers=dict(response.headers),
                        text=text,
                        json_data=response_json,
                    )

            except asyncio.TimeoutError:
                last_error = "Request timeout"
                logger.warning(
                    f"Timeout on attempt {attempt + 1}/{self.retry_attempts} for {url}"
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Error on attempt {attempt + 1}/{self.retry_attempts} for {url}: {e}"
                )

            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff

        return AsyncHTTPResponse(
            status=0,
            headers={},
            text="",
            error=last_error,
        )

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "AsyncHTTPClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()


# Global async HTTP client
_async_http_client: Optional[AsyncHTTPClient] = None


def get_async_http_client() -> AsyncHTTPClient:
    """Get the global async HTTP client instance."""
    global _async_http_client
    if _async_http_client is None:
        _async_http_client = AsyncHTTPClient()
    return _async_http_client


# =============================================================================
# Sync-to-Async and Async-to-Sync Wrappers
# =============================================================================


def run_async(coro: Awaitable[T]) -> T:
    """Run an async coroutine synchronously.

    This is a convenience function for running async code from sync contexts.
    Creates a new event loop if one doesn't exist.

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine
    """
    try:
        asyncio.get_running_loop()
        # If we're already in an async context, we can't use asyncio.run
        # Use a thread instead
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)  # type: ignore[arg-type]
            return future.result()
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        return asyncio.run(coro)  # type: ignore[arg-type]


def async_to_sync(func: Callable[P, Awaitable[T]]) -> Callable[P, T]:
    """Decorator to convert async function to sync function.

    Enables calling async functions from sync code.

    Args:
        func: Async function to wrap

    Returns:
        Sync function that runs the async function
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return run_async(func(*args, **kwargs))

    return wrapper


def sync_to_async(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """Decorator to convert sync function to async function.

    Runs sync function in a thread pool to avoid blocking.

    Args:
        func: Sync function to wrap

    Returns:
        Async function that runs the sync function in a thread
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    return wrapper


# =============================================================================
# Concurrent Story Processing
# =============================================================================


@dataclass
class ProcessingResult:
    """Result of processing a single item."""

    item_id: Any
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0.0


class ConcurrentProcessor:
    """Process multiple items concurrently with async/await.

    Provides controlled concurrent processing with rate limiting.
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        rate_limit_per_second: float = 2.0,
    ) -> None:
        """Initialize the concurrent processor.

        Args:
            max_concurrent: Maximum concurrent tasks
            rate_limit_per_second: Maximum tasks per second
        """
        self.max_concurrent = max_concurrent
        self.rate_limit_per_second = rate_limit_per_second
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def process_items(
        self,
        items: list[Any],
        process_func: Callable[[Any], Awaitable[Any]],
        item_id_func: Callable[[Any], Any] = lambda x: x,
    ) -> list[ProcessingResult]:
        """Process multiple items concurrently.

        Args:
            items: Items to process
            process_func: Async function to process each item
            item_id_func: Function to extract ID from item

        Returns:
            List of ProcessingResult for each item
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)

        results: list[ProcessingResult] = []
        min_interval = 1.0 / self.rate_limit_per_second

        async def process_with_rate_limit(item: Any, index: int) -> ProcessingResult:
            """Process a single item with rate limiting."""
            # Apply rate limiting delay
            await asyncio.sleep(index * min_interval)

            async with self._semaphore:  # type: ignore
                import time

                start = time.perf_counter()
                item_id = item_id_func(item)

                try:
                    result = await process_func(item)
                    duration = (time.perf_counter() - start) * 1000

                    return ProcessingResult(
                        item_id=item_id,
                        success=True,
                        result=result,
                        duration_ms=duration,
                    )

                except Exception as e:
                    duration = (time.perf_counter() - start) * 1000
                    logger.error(f"Error processing item {item_id}: {e}")

                    return ProcessingResult(
                        item_id=item_id,
                        success=False,
                        error=str(e),
                        duration_ms=duration,
                    )

        # Create tasks for all items
        tasks = [process_with_rate_limit(item, i) for i, item in enumerate(items)]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        return list(results)

    async def process_items_with_callback(
        self,
        items: list[Any],
        process_func: Callable[[Any], Awaitable[Any]],
        on_complete: Callable[[ProcessingResult], None],
        item_id_func: Callable[[Any], Any] = lambda x: x,
    ) -> list[ProcessingResult]:
        """Process items with a callback for each completion.

        Args:
            items: Items to process
            process_func: Async function to process each item
            on_complete: Callback for each completed item
            item_id_func: Function to extract ID from item

        Returns:
            List of all ProcessingResults
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)

        results: list[ProcessingResult] = []
        min_interval = 1.0 / self.rate_limit_per_second

        async def process_and_callback(item: Any, index: int) -> ProcessingResult:
            """Process item and call callback on completion."""
            await asyncio.sleep(index * min_interval)

            async with self._semaphore:  # type: ignore
                import time

                start = time.perf_counter()
                item_id = item_id_func(item)

                try:
                    result = await process_func(item)
                    duration = (time.perf_counter() - start) * 1000

                    proc_result = ProcessingResult(
                        item_id=item_id,
                        success=True,
                        result=result,
                        duration_ms=duration,
                    )

                except Exception as e:
                    duration = (time.perf_counter() - start) * 1000

                    proc_result = ProcessingResult(
                        item_id=item_id,
                        success=False,
                        error=str(e),
                        duration_ms=duration,
                    )

                # Call the callback
                on_complete(proc_result)
                return proc_result

        tasks = [process_and_callback(item, i) for i, item in enumerate(items)]

        results = await asyncio.gather(*tasks)
        return list(results)


# =============================================================================
# Batch Enrichment Processor (Phase 2)
# =============================================================================


@dataclass
class EnrichmentResult:
    """Result of enriching a single story."""

    story_id: int
    success: bool
    people_matched: int = 0
    people_total: int = 0
    high_confidence: int = 0
    org_fallback: int = 0
    rejected: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None
    enrichment_log: dict = field(default_factory=dict)


@dataclass
class BatchEnrichmentStats:
    """Aggregate stats for a batch enrichment run."""

    stories_processed: int = 0
    stories_succeeded: int = 0
    stories_failed: int = 0
    total_people: int = 0
    total_matched: int = 0
    high_confidence: int = 0
    org_fallback: int = 0
    rejected: int = 0
    total_duration_ms: float = 0.0
    api_calls: int = 0
    cache_hits: int = 0

    @property
    def match_rate(self) -> float:
        """Calculate overall match rate."""
        if self.total_people == 0:
            return 0.0
        return self.total_matched / self.total_people

    @property
    def high_confidence_rate(self) -> float:
        """Calculate high-confidence match rate."""
        if self.total_matched == 0:
            return 0.0
        return self.high_confidence / self.total_matched

    def to_log_dict(self) -> dict:
        """Convert to dict for logging."""
        return {
            "stories_processed": self.stories_processed,
            "stories_succeeded": self.stories_succeeded,
            "stories_failed": self.stories_failed,
            "total_people": self.total_people,
            "total_matched": self.total_matched,
            "match_rate": f"{self.match_rate:.1%}",
            "high_confidence": self.high_confidence,
            "high_confidence_rate": f"{self.high_confidence_rate:.1%}",
            "org_fallback": self.org_fallback,
            "rejected": self.rejected,
            "duration_ms": self.total_duration_ms,
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
        }


class BatchEnrichmentProcessor:
    """Process multiple stories with controlled concurrency and caching.

    Phase 2 implementation per spec Step 6.1:
    - Controlled concurrency for stories and validations
    - Deduplication of validation requests across stories
    - Caching of organization leadership
    - High-priority story processing
    """

    def __init__(
        self,
        max_concurrent_stories: int = 3,
        max_concurrent_validations: int = 5,
        rate_limit_per_second: float = 1.0,
    ) -> None:
        """Initialize the batch processor.

        Args:
            max_concurrent_stories: Max stories to process in parallel
            max_concurrent_validations: Max validation calls per story
            rate_limit_per_second: Rate limit for API calls
        """
        self.max_concurrent_stories = max_concurrent_stories
        self.max_concurrent_validations = max_concurrent_validations
        self.rate_limit_per_second = rate_limit_per_second

        # Validation cache: (person_name, org) -> validation_result
        self._validation_cache: dict[str, dict] = {}

        # Indirect people cache: org_name -> [people]
        self._indirect_people_cache: dict[str, list[dict]] = {}

        # Stats tracking
        self._stats = BatchEnrichmentStats()
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _cache_key(self, name: str, org: str) -> str:
        """Generate cache key for person validation."""
        return f"{name.lower().strip()}@{org.lower().strip()}"

    def get_cached_validation(self, name: str, org: str) -> Optional[dict]:
        """Get cached validation result if available."""
        key = self._cache_key(name, org)
        if key in self._validation_cache:
            self._stats.cache_hits += 1
            return self._validation_cache[key]
        return None

    def cache_validation(self, name: str, org: str, result: dict) -> None:
        """Cache a validation result."""
        key = self._cache_key(name, org)
        self._validation_cache[key] = result

    def get_cached_indirect_people(self, org: str) -> Optional[list[dict]]:
        """Get cached indirect people if available."""
        key = org.lower().strip()
        if key in self._indirect_people_cache:
            self._stats.cache_hits += 1
            return self._indirect_people_cache[key]
        return None

    def cache_indirect_people(self, org: str, people: list[dict]) -> None:
        """Cache indirect people for an org."""
        key = org.lower().strip()
        self._indirect_people_cache[key] = people

    async def process_story(
        self,
        story: Any,
        enrich_func: Callable[[Any], Awaitable[EnrichmentResult]],
    ) -> EnrichmentResult:
        """Process a single story with rate limiting.

        Args:
            story: Story object to process
            enrich_func: Async function that performs enrichment

        Returns:
            EnrichmentResult with outcome details
        """
        import time

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_stories)

        async with self._semaphore:
            # Apply rate limiting
            await asyncio.sleep(1.0 / self.rate_limit_per_second)

            start = time.perf_counter()
            self._stats.api_calls += 1

            try:
                result = await enrich_func(story)
                duration = (time.perf_counter() - start) * 1000
                result.duration_ms = duration

                # Update stats
                self._stats.stories_processed += 1
                if result.success:
                    self._stats.stories_succeeded += 1
                else:
                    self._stats.stories_failed += 1

                self._stats.total_people += result.people_total
                self._stats.total_matched += result.people_matched
                self._stats.high_confidence += result.high_confidence
                self._stats.org_fallback += result.org_fallback
                self._stats.rejected += result.rejected
                self._stats.total_duration_ms += duration

                return result

            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                self._stats.stories_failed += 1
                self._stats.total_duration_ms += duration

                logger.error(f"Error enriching story {getattr(story, 'id', '?')}: {e}")
                return EnrichmentResult(
                    story_id=getattr(story, "id", 0),
                    success=False,
                    error=str(e),
                    duration_ms=duration,
                )

    async def process_batch(
        self,
        stories: list[Any],
        enrich_func: Callable[[Any], Awaitable[EnrichmentResult]],
        priority_func: Optional[Callable[[Any], int]] = None,
    ) -> list[EnrichmentResult]:
        """Process multiple stories with controlled concurrency.

        Args:
            stories: List of Story objects to process
            enrich_func: Async function that performs enrichment
            priority_func: Optional function to get priority (higher = first)

        Returns:
            List of EnrichmentResult for each story
        """
        # Sort by priority if provided (higher priority first)
        if priority_func:
            stories = sorted(stories, key=priority_func, reverse=True)

        # Process all stories concurrently (semaphore controls actual concurrency)
        tasks = [self.process_story(story, enrich_func) for story in stories]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to EnrichmentResult
        final_results: list[EnrichmentResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                final_results.append(
                    EnrichmentResult(
                        story_id=getattr(stories[i], "id", 0),
                        success=False,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)  # type: ignore[arg-type]

        return final_results

    def get_stats(self) -> BatchEnrichmentStats:
        """Get current batch processing stats."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset stats for a new batch run."""
        self._stats = BatchEnrichmentStats()

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._validation_cache.clear()
        self._indirect_people_cache.clear()

    def log_stats(self) -> None:
        """Log current stats summary."""
        stats = self._stats.to_log_dict()
        logger.info(f"BATCH_ENRICHMENT: {stats}")


# =============================================================================
# Async Context Managers
# =============================================================================


@asynccontextmanager
async def async_timeout(seconds: float):
    """Async context manager for timeout.

    Args:
        seconds: Timeout in seconds

    Raises:
        asyncio.TimeoutError: If timeout is exceeded

    Example:
        async with async_timeout(10.0):
            await some_long_operation()
    """
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {seconds}s")
        raise


@asynccontextmanager
async def async_semaphore(sem: asyncio.Semaphore):
    """Async context manager for semaphore with logging.

    Args:
        sem: Semaphore to acquire

    Example:
        sem = asyncio.Semaphore(5)
        async with async_semaphore(sem):
            await limited_operation()
    """
    async with sem:
        yield


# =============================================================================
# Thread Pool for CPU-bound Operations
# =============================================================================


class AsyncThreadPool:
    """Thread pool for running CPU-bound operations in async context.

    Use this for operations that are CPU-bound but need to be called
    from async code without blocking the event loop.
    """

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize the thread pool.

        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None

    @property
    def executor(self) -> ThreadPoolExecutor:
        """Get or create the executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor

    async def run(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Run a sync function in the thread pool.

        Args:
            func: Function to run
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the function
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            functools.partial(func, *args, **kwargs),
        )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool.

        Args:
            wait: Whether to wait for pending tasks
        """
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None


# Global thread pool
_thread_pool: Optional[AsyncThreadPool] = None


def get_thread_pool() -> AsyncThreadPool:
    """Get the global thread pool instance."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = AsyncThreadPool()
    return _thread_pool


# =============================================================================
# Convenience Functions
# =============================================================================


async def gather_with_limit(
    *coros: Awaitable[T],
    limit: int = 5,
    return_exceptions: bool = True,
) -> list[T | BaseException]:
    """Like asyncio.gather but with concurrency limit.

    Args:
        *coros: Coroutines to run
        limit: Maximum concurrent coroutines
        return_exceptions: Whether to return exceptions instead of raising

    Returns:
        List of results (or exceptions if return_exceptions=True)
    """
    semaphore = asyncio.Semaphore(limit)

    async def limited_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *[limited_coro(coro) for coro in coros],
        return_exceptions=return_exceptions,
    )


async def retry_async(
    func: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exceptions: Exception types to catch and retry

    Returns:
        Result of successful function call

    Raises:
        Last exception if all attempts fail
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed: {e}. Retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

    raise last_exception  # type: ignore


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests() -> bool:
    """Create unit tests for async_utils module."""
    from test_framework import TestSuite

    suite = TestSuite("Async Utilities Tests", "async_utils.py")
    suite.start_suite()

    def test_run_async():
        """Test sync wrapper for async code."""

        async def async_func() -> int:
            await asyncio.sleep(0.01)
            return 42

        result = run_async(async_func())
        assert result == 42

    def test_async_to_sync_decorator():
        """Test async to sync decorator."""

        @async_to_sync
        async def async_add(a: int, b: int) -> int:
            await asyncio.sleep(0.01)
            return a + b

        result = async_add(2, 3)
        assert result == 5

    def test_sync_to_async_decorator():
        """Test sync to async decorator."""

        @sync_to_async
        def sync_multiply(a: int, b: int) -> int:
            return a * b

        async def run_test() -> int:
            return await sync_multiply(3, 4)

        result = run_async(run_test())
        assert result == 12

    def test_concurrent_processor():
        """Test concurrent processing."""

        async def process_item(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        processor = ConcurrentProcessor(max_concurrent=3, rate_limit_per_second=100)

        async def run_test() -> list[ProcessingResult]:
            return await processor.process_items([1, 2, 3], process_item)

        results = run_async(run_test())
        assert len(results) == 3
        assert all(r.success for r in results)
        assert [r.result for r in results] == [2, 4, 6]

    def test_processing_result():
        """Test ProcessingResult dataclass."""
        result = ProcessingResult(
            item_id=1,
            success=True,
            result="test",
            duration_ms=100.0,
        )
        assert result.item_id == 1
        assert result.success is True
        assert result.error is None

    def test_async_http_response():
        """Test AsyncHTTPResponse dataclass."""
        response = AsyncHTTPResponse(
            status=200,
            headers={"Content-Type": "application/json"},
            text='{"key": "value"}',
        )
        assert response.ok is True
        assert response.status == 200

        error_response = AsyncHTTPResponse(status=500, headers={}, text="")
        assert error_response.ok is False

    suite.run_test(
        test_name="run_async wrapper",
        test_func=test_run_async,
        test_summary="Tests run async wrapper functionality",
        method_description="Calls async func and verifies the result",
        expected_outcome="Function returns the expected value",
    )
    suite.run_test(
        test_name="async_to_sync decorator",
        test_func=test_async_to_sync_decorator,
        test_summary="Tests async to sync decorator functionality",
        method_description="Calls async add and verifies the result",
        expected_outcome="Function returns the expected value",
    )
    suite.run_test(
        test_name="sync_to_async decorator",
        test_func=test_sync_to_async_decorator,
        test_summary="Tests sync to async decorator functionality",
        method_description="Calls sync multiply and verifies the result",
        expected_outcome="Function returns the expected value",
    )
    suite.run_test(
        test_name="ConcurrentProcessor",
        test_func=test_concurrent_processor,
        test_summary="Tests ConcurrentProcessor functionality",
        method_description="Calls process item and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="ProcessingResult dataclass",
        test_func=test_processing_result,
        test_summary="Tests ProcessingResult dataclass functionality",
        method_description="Calls ProcessingResult and verifies the result",
        expected_outcome="Function returns None as expected",
    )
    suite.run_test(
        test_name="AsyncHTTPResponse dataclass",
        test_func=test_async_http_response,
        test_summary="Tests AsyncHTTPResponse dataclass functionality",
        method_description="Calls AsyncHTTPResponse and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )

    return suite.finish_suite()


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
