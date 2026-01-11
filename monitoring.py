"""
Comprehensive logging and monitoring module.

This module provides:
- Structured JSON logging for production environments
- Metrics collection for tracking system performance
- Performance timing decorators
- Event tracking for analytics

Implements TASK 4.4 from IMPROVEMENT_TASKS.md.
"""

import functools
import json
import logging
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


# =============================================================================
# Structured JSON Logging
# =============================================================================


class JSONFormatter(logging.Formatter):
    """
    Format log records as JSON for structured logging.

    Produces machine-readable logs suitable for log aggregation
    systems like ELK, Loki, or CloudWatch.
    """

    def __init__(self, include_extras: bool = True):
        """
        Initialize the JSON formatter.

        Args:
            include_extras: Whether to include extra fields from the record
        """
        super().__init__()
        self.include_extras = include_extras
        self._skip_fields = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "taskName",
            "message",
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if self.include_extras:
            for key, value in record.__dict__.items():
                if key not in self._skip_fields and not key.startswith("_"):
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for development environments.

    Uses ANSI color codes for visual log level differentiation.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format with color codes."""
        color = self.COLORS.get(record.levelname, "")
        formatted = super().format(record)
        return f"{color}{formatted}{self.RESET}"


def setup_logging(
    log_level: str = "INFO",
    json_format: bool = False,
    log_file: Path | None = None,
    console_colors: bool = True,
) -> None:
    """
    Configure application logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON structured logging
        log_file: Optional file path for log output
        console_colors: Use colored output for console (dev mode)
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if json_format:
        console_handler.setFormatter(JSONFormatter())
    elif console_colors:
        console_handler.setFormatter(
            ColoredFormatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
    else:
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )

    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# =============================================================================
# Metrics Collection
# =============================================================================


@dataclass
class MetricValue:
    """Stores a single metric value with metadata."""

    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricStats:
    """Statistics for a metric over time."""

    count: int = 0
    total: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    last_value: float = 0.0
    last_timestamp: float = 0.0

    @property
    def mean(self) -> float:
        """Calculate mean value."""
        return self.total / self.count if self.count > 0 else 0.0

    def record(self, value: float) -> None:
        """Record a new value."""
        self.count += 1
        self.total += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.last_value = value
        self.last_timestamp = time.time()


class MetricsCollector:
    """
    Thread-safe metrics collector for application monitoring.

    Supports counters, gauges, and histograms. Metrics can be
    exported in Prometheus-compatible format.
    """

    _instance: "MetricsCollector | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        """Singleton pattern for global metrics collection."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        if self._initialized:
            return

        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, MetricStats] = defaultdict(MetricStats)
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._labels: dict[str, dict[str, str]] = {}
        self._data_lock = threading.Lock()
        self._initialized = True

    def increment(
        self, name: str, value: int = 1, labels: dict[str, str] | None = None
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name
            value: Amount to increment by
            labels: Optional labels for the metric
        """
        key = self._build_key(name, labels)
        with self._data_lock:
            self._counters[key] += value
            if labels:
                self._labels[key] = labels

    def gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """
        Set a gauge metric value.

        Args:
            name: Metric name
            value: Current value
            labels: Optional labels for the metric
        """
        key = self._build_key(name, labels)
        with self._data_lock:
            self._gauges[key] = value
            if labels:
                self._labels[key] = labels

    def histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """
        Record a histogram/distribution metric value.

        Args:
            name: Metric name
            value: Observed value
            labels: Optional labels for the metric
        """
        key = self._build_key(name, labels)
        with self._data_lock:
            self._histograms[key].record(value)
            if labels:
                self._labels[key] = labels

    def timing(
        self, name: str, duration_seconds: float, labels: dict[str, str] | None = None
    ) -> None:
        """
        Record a timing metric.

        Args:
            name: Metric name
            duration_seconds: Duration in seconds
            labels: Optional labels for the metric
        """
        key = self._build_key(name, labels)
        with self._data_lock:
            self._timings[key].append(duration_seconds)
            # Keep only last 1000 timings
            if len(self._timings[key]) > 1000:
                self._timings[key] = self._timings[key][-1000:]
            if labels:
                self._labels[key] = labels

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> int:
        """Get current counter value."""
        key = self._build_key(name, labels)
        with self._data_lock:
            return self._counters.get(key, 0)

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get current gauge value."""
        key = self._build_key(name, labels)
        with self._data_lock:
            return self._gauges.get(key, 0.0)

    def get_histogram_stats(
        self, name: str, labels: dict[str, str] | None = None
    ) -> MetricStats | None:
        """Get histogram statistics."""
        key = self._build_key(name, labels)
        with self._data_lock:
            return self._histograms.get(key)

    def get_timing_stats(
        self, name: str, labels: dict[str, str] | None = None
    ) -> dict[str, float]:
        """
        Get timing statistics.

        Returns:
            Dictionary with count, mean, min, max, p50, p95, p99
        """
        key = self._build_key(name, labels)
        with self._data_lock:
            timings = self._timings.get(key, [])
            if not timings:
                return {}

            sorted_timings = sorted(timings)
            count = len(sorted_timings)

            return {
                "count": count,
                "mean": sum(sorted_timings) / count,
                "min": sorted_timings[0],
                "max": sorted_timings[-1],
                "p50": sorted_timings[int(count * 0.5)],
                "p95": sorted_timings[int(count * 0.95)]
                if count >= 20
                else sorted_timings[-1],
                "p99": sorted_timings[int(count * 0.99)]
                if count >= 100
                else sorted_timings[-1],
            }

    def get_all_metrics(self) -> dict[str, Any]:
        """
        Get all collected metrics.

        Returns:
            Dictionary of all metrics organized by type
        """
        with self._data_lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: {
                        "count": v.count,
                        "mean": v.mean,
                        "min": v.min_value if v.count > 0 else 0,
                        "max": v.max_value if v.count > 0 else 0,
                        "last": v.last_value,
                    }
                    for k, v in self._histograms.items()
                },
                "timings": {
                    k: self.get_timing_stats(k) or {} for k in self._timings.keys()
                },
            }

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._data_lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timings.clear()
            self._labels.clear()

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-compatible metrics string
        """
        lines = []

        with self._data_lock:
            # Export counters
            for name, value in self._counters.items():
                labels_str = self._format_labels(self._labels.get(name, {}))
                lines.append(f"{name}{labels_str} {value}")

            # Export gauges
            for name, value in self._gauges.items():
                labels_str = self._format_labels(self._labels.get(name, {}))
                lines.append(f"{name}{labels_str} {value}")

            # Export histograms as summary
            for name, stats in self._histograms.items():
                labels_str = self._format_labels(self._labels.get(name, {}))
                lines.append(f"{name}_count{labels_str} {stats.count}")
                lines.append(f"{name}_sum{labels_str} {stats.total}")

        return "\n".join(lines)

    @staticmethod
    def _build_key(name: str, labels: dict[str, str] | None) -> str:
        """Build a unique key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    @staticmethod
    def _format_labels(labels: dict[str, str]) -> str:
        """Format labels for Prometheus output."""
        if not labels:
            return ""
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{{{label_str}}}"


# Global metrics instance
metrics = MetricsCollector()


# =============================================================================
# Performance Timing Decorators
# =============================================================================


def timed(metric_name: str | None = None, labels: dict[str, str] | None = None):
    """
    Decorator to measure function execution time.

    Args:
        metric_name: Optional metric name (defaults to function name)
        labels: Optional labels for the metric

    Example:
        @timed("api_call")
        def fetch_data():
            ...
    """

    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                metrics.increment(f"{name}_success", labels=labels)
                return result
            except Exception:
                metrics.increment(f"{name}_error", labels=labels)
                raise
            finally:
                duration = time.perf_counter() - start_time
                metrics.timing(f"{name}_duration_seconds", duration, labels=labels)

        return wrapper

    return decorator


def counted(metric_name: str | None = None, labels: dict[str, str] | None = None):
    """
    Decorator to count function calls.

    Args:
        metric_name: Optional metric name (defaults to function name)
        labels: Optional labels for the metric

    Example:
        @counted("story_processed")
        def process_story(story):
            ...
    """

    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}_calls"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metrics.increment(name, labels=labels)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Event Tracking
# =============================================================================


@dataclass
class Event:
    """Represents a tracked event."""

    name: str
    timestamp: float
    data: dict[str, Any]
    labels: dict[str, str] = field(default_factory=dict)


class EventTracker:
    """
    Track application events for analytics and debugging.

    Events are stored in memory with a configurable buffer size.
    """

    def __init__(self, max_events: int = 10000):
        """
        Initialize the event tracker.

        Args:
            max_events: Maximum number of events to keep in memory
        """
        self._events: list[Event] = []
        self._max_events = max_events
        self._lock = threading.Lock()
        self._logger = get_logger(__name__)

    def track(
        self,
        event_name: str,
        data: dict[str, Any] | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Track an event.

        Args:
            event_name: Name of the event
            data: Optional event data
            labels: Optional event labels
        """
        event = Event(
            name=event_name,
            timestamp=time.time(),
            data=data or {},
            labels=labels or {},
        )

        with self._lock:
            self._events.append(event)
            # Trim old events if over limit
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]

        # Also log the event
        self._logger.debug(
            f"Event: {event_name}",
            extra={"event_data": data, "event_labels": labels},
        )

        # Update metrics
        metrics.increment(f"event_{event_name}", labels=labels)

    def get_events(
        self,
        event_name: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """
        Get tracked events.

        Args:
            event_name: Filter by event name
            since: Only events after this timestamp
            limit: Maximum number of events to return

        Returns:
            List of matching events
        """
        with self._lock:
            events = self._events.copy()

        if event_name:
            events = [e for e in events if e.name == event_name]

        if since:
            events = [e for e in events if e.timestamp > since]

        return events[-limit:]

    def get_event_counts(self, since: float | None = None) -> dict[str, int]:
        """
        Get event counts by name.

        Args:
            since: Only count events after this timestamp

        Returns:
            Dictionary mapping event names to counts
        """
        with self._lock:
            events = self._events.copy()

        if since:
            events = [e for e in events if e.timestamp > since]

        counts: dict[str, int] = defaultdict(int)
        for event in events:
            counts[event.name] += 1

        return dict(counts)

    def clear(self) -> None:
        """Clear all tracked events."""
        with self._lock:
            self._events.clear()


# Global event tracker instance
events = EventTracker()


# =============================================================================
# Convenience Functions
# =============================================================================


def log_and_track(
    event_name: str,
    message: str,
    level: str = "INFO",
    data: dict[str, Any] | None = None,
) -> None:
    """
    Log a message and track it as an event.

    Args:
        event_name: Event name for tracking
        message: Log message
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        data: Optional event data
    """
    logger = get_logger("app")
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message, extra=data or {})
    events.track(event_name, data)


def get_health_status() -> dict[str, Any]:
    """
    Get application health status including key metrics.

    Returns:
        Dictionary with health information
    """
    all_metrics = metrics.get_all_metrics()
    event_counts = events.get_event_counts(since=time.time() - 3600)  # Last hour

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics_summary": {
            "total_counters": len(all_metrics["counters"]),
            "total_gauges": len(all_metrics["gauges"]),
            "total_histograms": len(all_metrics["histograms"]),
        },
        "recent_events": event_counts,
    }


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for monitoring module."""
    from test_framework import TestSuite

    suite = TestSuite("Monitoring Tests")

    def test_json_formatter():
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed

    def test_colored_formatter():
        formatter = ColoredFormatter("%(levelname)s - %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        # Should contain ANSI color codes
        assert "\033[31m" in output  # Red for ERROR
        assert "Error message" in output

    def test_metrics_counter():
        m = MetricsCollector()
        m.reset()
        m.increment("test_counter")
        m.increment("test_counter", 5)
        assert m.get_counter("test_counter") == 6

    def test_metrics_gauge():
        m = MetricsCollector()
        m.reset()
        m.gauge("test_gauge", 42.5)
        assert m.get_gauge("test_gauge") == 42.5
        m.gauge("test_gauge", 100.0)
        assert m.get_gauge("test_gauge") == 100.0

    def test_metrics_histogram():
        m = MetricsCollector()
        m.reset()
        for i in range(10):
            m.histogram("test_histogram", float(i))
        stats = m.get_histogram_stats("test_histogram")
        assert stats is not None
        assert stats.count == 10
        assert stats.min_value == 0.0
        assert stats.max_value == 9.0

    def test_metrics_timing():
        m = MetricsCollector()
        m.reset()
        for i in range(100):
            m.timing("test_timing", 0.01 * i)
        stats = m.get_timing_stats("test_timing")
        assert stats["count"] == 100
        assert stats["min"] == 0.0

    def test_metrics_with_labels():
        m = MetricsCollector()
        m.reset()
        m.increment("api_calls", labels={"endpoint": "/users"})
        m.increment("api_calls", labels={"endpoint": "/posts"})
        assert m.get_counter("api_calls", labels={"endpoint": "/users"}) == 1
        assert m.get_counter("api_calls", labels={"endpoint": "/posts"}) == 1

    def test_prometheus_export():
        m = MetricsCollector()
        m.reset()
        m.increment("http_requests", 10)
        m.gauge("active_connections", 5.0)
        output = m.export_prometheus()
        assert "http_requests" in output
        assert "active_connections" in output

    def test_event_tracker():
        tracker = EventTracker(max_events=100)
        tracker.clear()
        tracker.track("test_event", {"key": "value"})
        events_list = tracker.get_events("test_event")
        assert len(events_list) == 1
        assert events_list[0].data == {"key": "value"}

    def test_event_tracker_limit():
        tracker = EventTracker(max_events=5)
        tracker.clear()
        for i in range(10):
            tracker.track("event", {"index": i})
        events_list = tracker.get_events()
        assert len(events_list) == 5
        # Should have most recent events
        assert events_list[-1].data["index"] == 9

    def test_event_counts():
        tracker = EventTracker()
        tracker.clear()
        tracker.track("event_a")
        tracker.track("event_a")
        tracker.track("event_b")
        counts = tracker.get_event_counts()
        assert counts["event_a"] == 2
        assert counts["event_b"] == 1

    def test_timed_decorator():
        m = MetricsCollector()
        m.reset()

        @timed("test_function")
        def slow_function():
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"
        assert m.get_counter("test_function_success") == 1
        stats = m.get_timing_stats("test_function_duration_seconds")
        assert stats["count"] == 1
        assert stats["mean"] >= 0.01

    def test_counted_decorator():
        m = MetricsCollector()
        m.reset()

        @counted("my_function")
        def simple_function():
            return 42

        for _ in range(5):
            simple_function()

        assert m.get_counter("my_function") == 5

    def test_health_status():
        m = MetricsCollector()
        m.reset()
        m.increment("health_check")
        status = get_health_status()
        assert status["status"] == "healthy"
        assert "timestamp" in status
        assert "metrics_summary" in status

    def test_metric_stats():
        stats = MetricStats()
        stats.record(10.0)
        stats.record(20.0)
        stats.record(30.0)
        assert stats.count == 3
        assert stats.mean == 20.0
        assert stats.min_value == 10.0
        assert stats.max_value == 30.0

    suite.add_test("JSON formatter", test_json_formatter)
    suite.add_test("Colored formatter", test_colored_formatter)
    suite.add_test("Metrics counter", test_metrics_counter)
    suite.add_test("Metrics gauge", test_metrics_gauge)
    suite.add_test("Metrics histogram", test_metrics_histogram)
    suite.add_test("Metrics timing", test_metrics_timing)
    suite.add_test("Metrics with labels", test_metrics_with_labels)
    suite.add_test("Prometheus export", test_prometheus_export)
    suite.add_test("Event tracker", test_event_tracker)
    suite.add_test("Event tracker limit", test_event_tracker_limit)
    suite.add_test("Event counts", test_event_counts)
    suite.add_test("Timed decorator", test_timed_decorator)
    suite.add_test("Counted decorator", test_counted_decorator)
    suite.add_test("Health status", test_health_status)
    suite.add_test("Metric stats", test_metric_stats)

    return suite
