"""
Caching layer for API responses and expensive operations.

This module provides:
- SQLite-based persistent caching with TTL support
- In-memory LRU caching for hot data
- Cache key generation utilities
- Cache statistics and metrics
- Automatic cache invalidation

Implements TASK 5.3 from IMPROVEMENT_TASKS.md.
"""

import functools
import hashlib
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar


# Type variable for generic cached functions
T = TypeVar("T")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CacheEntry:
    """Represents a cached value with metadata."""

    key: str
    value: Any
    created_at: float
    expires_at: float
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        return time.time() > self.expires_at

    @property
    def ttl_remaining(self) -> float:
        """Get remaining TTL in seconds."""
        return max(0, self.expires_at - time.time())


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_entries: int = 0
    memory_entries: int = 0
    disk_entries: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


# =============================================================================
# In-Memory LRU Cache
# =============================================================================


class LRUCache:
    """
    Thread-safe in-memory LRU cache with TTL support.

    Features:
    - Least Recently Used eviction
    - Per-entry TTL
    - Thread-safe operations
    - Hit/miss statistics
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600.0) -> None:
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (1 hour)
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._stats = CacheStats()

    def get(self, key: str) -> Any | None:
        """
        Get a value from the cache.

        Args:
            key: The cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hit_count += 1
            self._stats.hits += 1

            return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: The cache key
            value: The value to cache
            ttl: Optional TTL in seconds (uses default if not specified)
        """
        with self._lock:
            ttl = ttl if ttl is not None else self._default_ttl
            now = time.time()

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=now + ttl,
            )

            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]

            # Evict LRU entries if at capacity
            while len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1

            self._cache[key] = entry

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: The cache key

        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired]
            for key in expired_keys:
                del self._cache[key]
                self._stats.expirations += 1
            return len(expired_keys)

    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            self._stats.memory_entries = len(self._cache)
            self._stats.total_entries = len(self._cache)
            return self._stats

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# =============================================================================
# SQLite Persistent Cache
# =============================================================================


class SQLiteCache:
    """
    Persistent SQLite-based cache with TTL support.

    Features:
    - Persistent storage across restarts
    - JSON serialization for complex values
    - Automatic expired entry cleanup
    - Thread-safe operations
    """

    def __init__(
        self,
        db_path: Path | str = "cache.db",
        default_ttl: float = 86400.0,  # 24 hours
        cleanup_interval: int = 100,  # Cleanup every N operations
    ) -> None:
        """
        Initialize the SQLite cache.

        Args:
            db_path: Path to the SQLite database file
            default_ttl: Default TTL in seconds
            cleanup_interval: Operations between cleanup runs
        """
        self._db_path = Path(db_path)
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        self._operation_count = 0
        self._lock = threading.RLock()
        self._stats = CacheStats()

        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON cache(expires_at)
            """)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(str(self._db_path), timeout=10.0)

    def _maybe_cleanup(self) -> None:
        """Run cleanup if interval has elapsed."""
        self._operation_count += 1
        if self._operation_count >= self._cleanup_interval:
            self._operation_count = 0
            self.cleanup_expired()

    def get(self, key: str) -> Any | None:
        """
        Get a value from the cache.

        Args:
            key: The cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            self._maybe_cleanup()

            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT value, expires_at FROM cache WHERE key = ?",
                    (key,),
                )
                row = cursor.fetchone()

                if not row:
                    self._stats.misses += 1
                    return None

                value_json, expires_at = row

                # Check expiration
                if time.time() > expires_at:
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
                    self._stats.expirations += 1
                    self._stats.misses += 1
                    return None

                # Update hit count
                conn.execute(
                    "UPDATE cache SET hit_count = hit_count + 1 WHERE key = ?",
                    (key,),
                )
                conn.commit()
                self._stats.hits += 1

                try:
                    return json.loads(value_json)
                except json.JSONDecodeError:
                    return value_json

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: The cache key
            value: The value to cache
            ttl: Optional TTL in seconds
        """
        with self._lock:
            ttl = ttl if ttl is not None else self._default_ttl
            now = time.time()
            expires_at = now + ttl

            try:
                value_json = json.dumps(value)
            except (TypeError, ValueError):
                value_json = json.dumps(str(value))

            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache
                    (key, value, created_at, expires_at, hit_count)
                    VALUES (?, ?, ?, ?, 0)
                    """,
                    (key, value_json, now, expires_at),
                )
                conn.commit()

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: The cache key

        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM cache")
                conn.commit()
            self._stats = CacheStats()

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            now = time.time()
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM cache WHERE expires_at < ?", (now,))
                conn.commit()
                removed = cursor.rowcount
                self._stats.expirations += removed
                return removed

    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM cache")
                count = cursor.fetchone()[0]
                self._stats.disk_entries = count
                self._stats.total_entries = count
            return self._stats

    def close(self) -> None:
        """Close the cache (for cleanup in tests)."""
        # SQLite connections are closed automatically via context manager
        # But we force a cleanup to release any locks
        try:
            import gc
            gc.collect()  # Force garbage collection to close any lingering connections
        except Exception:
            pass

    def __len__(self) -> int:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM cache")
                return cursor.fetchone()[0]


# =============================================================================
# Two-Level Cache (Memory + Disk)
# =============================================================================


class TwoLevelCache:
    """
    Two-level cache combining fast in-memory cache with persistent disk cache.

    Features:
    - L1: Fast in-memory LRU cache
    - L2: Persistent SQLite cache
    - Automatic promotion from L2 to L1 on hits
    - Write-through to both levels
    """

    def __init__(
        self,
        memory_size: int = 500,
        memory_ttl: float = 300.0,  # 5 minutes
        disk_path: Path | str = "cache.db",
        disk_ttl: float = 86400.0,  # 24 hours
    ) -> None:
        """
        Initialize the two-level cache.

        Args:
            memory_size: Max entries in memory cache
            memory_ttl: TTL for memory cache
            disk_path: Path to SQLite database
            disk_ttl: TTL for disk cache
        """
        self._l1 = LRUCache(max_size=memory_size, default_ttl=memory_ttl)
        self._l2 = SQLiteCache(db_path=disk_path, default_ttl=disk_ttl)

    def get(self, key: str) -> Any | None:
        """
        Get a value, checking L1 first then L2.

        Args:
            key: The cache key

        Returns:
            Cached value or None if not found
        """
        # Try L1 first
        value = self._l1.get(key)
        if value is not None:
            return value

        # Try L2
        value = self._l2.get(key)
        if value is not None:
            # Promote to L1
            self._l1.set(key, value)
            return value

        return None

    def set(
        self,
        key: str,
        value: Any,
        memory_ttl: float | None = None,
        disk_ttl: float | None = None,
    ) -> None:
        """
        Set a value in both cache levels.

        Args:
            key: The cache key
            value: The value to cache
            memory_ttl: Optional TTL for memory cache
            disk_ttl: Optional TTL for disk cache
        """
        self._l1.set(key, value, memory_ttl)
        self._l2.set(key, value, disk_ttl)

    def delete(self, key: str) -> bool:
        """Delete from both cache levels."""
        l1_deleted = self._l1.delete(key)
        l2_deleted = self._l2.delete(key)
        return l1_deleted or l2_deleted

    def clear(self) -> None:
        """Clear both cache levels."""
        self._l1.clear()
        self._l2.clear()

    def get_stats(self) -> dict[str, CacheStats]:
        """Get statistics for both cache levels."""
        return {
            "l1_memory": self._l1.get_stats(),
            "l2_disk": self._l2.get_stats(),
        }

    def close(self) -> None:
        """Close the cache (for cleanup in tests)."""
        self._l2.close()


# =============================================================================
# Cache Key Generation
# =============================================================================


def generate_cache_key(*args: Any, prefix: str = "", **kwargs: Any) -> str:
    """
    Generate a deterministic cache key from arguments.

    Args:
        *args: Positional arguments to include in key
        prefix: Optional prefix for the key
        **kwargs: Keyword arguments to include in key

    Returns:
        MD5 hash-based cache key
    """
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_data = ":".join(key_parts)

    hash_value = hashlib.md5(key_data.encode()).hexdigest()[:16]

    if prefix:
        return f"{prefix}:{hash_value}"
    return hash_value


def generate_url_cache_key(url: str) -> str:
    """Generate a cache key for a URL."""
    return generate_cache_key(url, prefix="url")


def generate_profile_cache_key(profile_url: str) -> str:
    """Generate a cache key for a LinkedIn profile."""
    return generate_cache_key(profile_url, prefix="profile")


def generate_entity_cache_key(entity_name: str, entity_type: str) -> str:
    """Generate a cache key for an entity lookup."""
    return generate_cache_key(entity_name, entity_type, prefix="entity")


# =============================================================================
# Caching Decorator
# =============================================================================


def cached(
    cache: LRUCache | SQLiteCache | TwoLevelCache,
    ttl: float | None = None,
    key_prefix: str = "",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to cache function results.

    Args:
        cache: The cache instance to use
        ttl: Optional TTL in seconds
        key_prefix: Optional prefix for cache keys

    Example:
        @cached(my_cache, ttl=3600)
        def fetch_profile(url: str) -> dict:
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate cache key
            prefix = key_prefix or func.__name__
            key = generate_cache_key(*args, prefix=prefix, **kwargs)

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Call function and cache result
            result = func(*args, **kwargs)
            if result is not None:
                if ttl is not None:
                    cache.set(key, result, ttl)
                else:
                    cache.set(key, result)

            return result

        return wrapper

    return decorator


# =============================================================================
# Global Cache Instance
# =============================================================================


_global_cache: TwoLevelCache | None = None


def get_cache(
    memory_size: int = 500,
    disk_path: Path | str | None = None,
) -> TwoLevelCache:
    """
    Get or create the global cache instance.

    Args:
        memory_size: Max entries in memory cache
        disk_path: Optional path to SQLite database

    Returns:
        TwoLevelCache instance
    """
    global _global_cache

    if _global_cache is None:
        if disk_path is None:
            # Use default path in project directory
            disk_path = Path(__file__).parent / "cache.db"

        _global_cache = TwoLevelCache(
            memory_size=memory_size,
            disk_path=disk_path,
        )

    return _global_cache


def clear_global_cache() -> None:
    """Clear the global cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for cache module."""
    from test_framework import TestSuite
    import tempfile

    suite = TestSuite("Cache Tests")

    def test_lru_cache_basic():
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None

    def test_lru_cache_ttl():
        cache = LRUCache(default_ttl=0.1)  # 100ms TTL
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(0.15)
        assert cache.get("key1") is None  # Expired

    def test_lru_cache_eviction():
        cache = LRUCache(max_size=3)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"

    def test_lru_cache_lru_order():
        cache = LRUCache(max_size=3)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.get("key1")  # Access key1 to make it recently used
        cache.set("key4", "value4")  # Should evict key2 (LRU)
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

    def test_lru_cache_stats():
        cache = LRUCache()
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss
        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 1

    def test_sqlite_cache_basic():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = SQLiteCache(db_path=db_path)
            try:
                cache.set("key1", {"nested": "value"})
                assert cache.get("key1") == {"nested": "value"}
                assert cache.get("nonexistent") is None
            finally:
                cache.close()

    def test_sqlite_cache_ttl():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = SQLiteCache(db_path=db_path, default_ttl=0.1)
            try:
                cache.set("key1", "value1")
                assert cache.get("key1") == "value1"
                time.sleep(0.15)
                assert cache.get("key1") is None
            finally:
                cache.close()

    def test_sqlite_cache_cleanup():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = SQLiteCache(db_path=db_path, default_ttl=0.1)
            try:
                cache.set("key1", "value1")
                cache.set("key2", "value2")
                time.sleep(0.15)
                removed = cache.cleanup_expired()
                assert removed == 2
            finally:
                cache.close()

    def test_two_level_cache():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = TwoLevelCache(disk_path=db_path)
            try:
                cache.set("key1", "value1")
                assert cache.get("key1") == "value1"
            finally:
                cache.close()

    def test_two_level_cache_promotion():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"
            cache = TwoLevelCache(
                memory_ttl=0.05,  # Short memory TTL
                disk_path=db_path,
                disk_ttl=10.0,
            )
            try:
                cache.set("key1", "value1")
                time.sleep(0.1)  # Let L1 expire
                # Value should still be in L2 and get promoted to L1
                assert cache.get("key1") == "value1"
                # Now it should be in L1 again
                assert cache._l1.get("key1") == "value1"
            finally:
                cache.close()

    def test_generate_cache_key():
        key1 = generate_cache_key("arg1", "arg2", prefix="test")
        key2 = generate_cache_key("arg1", "arg2", prefix="test")
        key3 = generate_cache_key("arg1", "arg3", prefix="test")
        assert key1 == key2  # Same args = same key
        assert key1 != key3  # Different args = different key
        assert key1.startswith("test:")

    def test_generate_cache_key_with_kwargs():
        key1 = generate_cache_key("arg1", foo="bar", prefix="test")
        key2 = generate_cache_key("arg1", foo="bar", prefix="test")
        key3 = generate_cache_key("arg1", foo="baz", prefix="test")
        assert key1 == key2
        assert key1 != key3

    def test_cached_decorator():
        cache = LRUCache()
        call_count = 0

        @cached(cache, ttl=60)
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive_function(5) == 10
        assert call_count == 1
        assert expensive_function(5) == 10  # Cached
        assert call_count == 1  # Not called again
        assert expensive_function(10) == 20  # Different arg
        assert call_count == 2

    def test_cache_delete():
        cache = LRUCache()
        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("nonexistent") is False

    def test_cache_clear():
        cache = LRUCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert len(cache) == 0

    suite.add_test("LRU cache basic", test_lru_cache_basic)
    suite.add_test("LRU cache TTL", test_lru_cache_ttl)
    suite.add_test("LRU cache eviction", test_lru_cache_eviction)
    suite.add_test("LRU cache LRU order", test_lru_cache_lru_order)
    suite.add_test("LRU cache stats", test_lru_cache_stats)
    suite.add_test("SQLite cache basic", test_sqlite_cache_basic)
    suite.add_test("SQLite cache TTL", test_sqlite_cache_ttl)
    suite.add_test("SQLite cache cleanup", test_sqlite_cache_cleanup)
    suite.add_test("Two-level cache", test_two_level_cache)
    suite.add_test("Two-level cache promotion", test_two_level_cache_promotion)
    suite.add_test("Generate cache key", test_generate_cache_key)
    suite.add_test("Cache key with kwargs", test_generate_cache_key_with_kwargs)
    suite.add_test("Cached decorator", test_cached_decorator)
    suite.add_test("Cache delete", test_cache_delete)
    suite.add_test("Cache clear", test_cache_clear)

    return suite
