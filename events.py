"""Event-driven architecture for Social Media Publisher.

This module provides an event bus for decoupled, inter-component communication,
enabling extensibility through event listeners and plugin architecture.

Features:
- Type-safe event definitions
- Synchronous and async event dispatching
- Event filtering and prioritization
- Built-in events for the publishing pipeline
- Webhook support for external integrations

Example:
    bus = EventBus()

    @bus.on(StoryDiscovered)
    def handle_story(event: StoryDiscovered):
        print(f"New story: {event.title}")

    bus.emit(StoryDiscovered(title="Tech News", url="..."))
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    TypeVar,
)
from urllib.request import Request, urlopen
from urllib.error import URLError

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# =============================================================================
# Event Base Classes
# =============================================================================


class EventPriority(Enum):
    """Priority levels for event handlers."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event(ABC):
    """Base class for all events."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    correlation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "type": self.__class__.__name__,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id,
            "data": self._get_data(),
        }

    def _get_data(self) -> dict[str, Any]:
        """Get event-specific data. Override in subclasses."""
        return {}


# Type variable for generic event handling
E = TypeVar("E", bound=Event)


# =============================================================================
# Built-in Events
# =============================================================================


@dataclass
class StoryDiscovered(Event):
    """Emitted when a new story is discovered."""

    title: str = ""
    url: str = ""
    source: str = ""
    quality_score: float = 0.0

    def _get_data(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "quality_score": self.quality_score,
        }


@dataclass
class StoryVerified(Event):
    """Emitted when a story passes verification."""

    story_id: int = 0
    title: str = ""
    verification_result: str = ""
    confidence: float = 0.0

    def _get_data(self) -> dict[str, Any]:
        return {
            "story_id": self.story_id,
            "title": self.title,
            "verification_result": self.verification_result,
            "confidence": self.confidence,
        }


@dataclass
class StoryRejected(Event):
    """Emitted when a story is rejected."""

    story_id: int = 0
    title: str = ""
    reason: str = ""

    def _get_data(self) -> dict[str, Any]:
        return {
            "story_id": self.story_id,
            "title": self.title,
            "reason": self.reason,
        }


@dataclass
class ImageGenerated(Event):
    """Emitted when an image is generated for a story."""

    story_id: int = 0
    image_path: str = ""
    style: str = ""
    quality_score: float = 0.0

    def _get_data(self) -> dict[str, Any]:
        return {
            "story_id": self.story_id,
            "image_path": self.image_path,
            "style": self.style,
            "quality_score": self.quality_score,
        }


@dataclass
class StoryPublished(Event):
    """Emitted when a story is published to LinkedIn."""

    story_id: int = 0
    post_id: str = ""
    linkedin_url: str = ""
    title: str = ""

    def _get_data(self) -> dict[str, Any]:
        return {
            "story_id": self.story_id,
            "post_id": self.post_id,
            "linkedin_url": self.linkedin_url,
            "title": self.title,
        }


@dataclass
class PublishFailed(Event):
    """Emitted when publishing fails."""

    story_id: int = 0
    title: str = ""
    error: str = ""
    retry_count: int = 0

    def _get_data(self) -> dict[str, Any]:
        return {
            "story_id": self.story_id,
            "title": self.title,
            "error": self.error,
            "retry_count": self.retry_count,
        }


@dataclass
class AnalyticsUpdated(Event):
    """Emitted when analytics are updated for a post."""

    post_id: str = ""
    impressions: int = 0
    engagement_count: int = 0
    engagement_rate: float = 0.0

    def _get_data(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "impressions": self.impressions,
            "engagement_count": self.engagement_count,
            "engagement_rate": self.engagement_rate,
        }


@dataclass
class RateLimitHit(Event):
    """Emitted when a rate limit is encountered."""

    service: str = ""
    limit_type: str = ""
    retry_after: float = 0.0

    def _get_data(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "limit_type": self.limit_type,
            "retry_after": self.retry_after,
        }


@dataclass
class SystemHealth(Event):
    """Emitted for system health checks."""

    component: str = ""
    status: str = "healthy"  # healthy, degraded, unhealthy
    details: dict[str, Any] = field(default_factory=dict)

    def _get_data(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status,
            "details": self.details,
        }


# =============================================================================
# Event Handler
# =============================================================================


@dataclass
class EventHandler(Generic[E]):
    """Wrapper for event handlers with metadata."""

    handler: Callable[[E], None]
    priority: EventPriority = EventPriority.NORMAL
    filter_func: Callable[[E], bool] | None = None
    name: str = ""
    once: bool = False  # If True, handler is removed after first call

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.handler.__name__

    def should_handle(self, event: E) -> bool:
        """Check if this handler should process the event."""
        if self.filter_func is not None:
            return self.filter_func(event)
        return True

    def __call__(self, event: E) -> None:
        """Call the handler."""
        self.handler(event)


# =============================================================================
# Event Bus
# =============================================================================


class EventBus:
    """Central event bus for publishing and subscribing to events."""

    def __init__(self) -> None:
        """Initialize the event bus."""
        self._handlers: dict[type[Event], list[EventHandler[Any]]] = {}
        self._global_handlers: list[EventHandler[Any]] = []
        self._history: list[Event] = []
        self._history_limit: int = 1000
        self._webhooks: list[WebhookConfig] = []
        self._paused: bool = False

    def on(
        self,
        event_type: type[E],
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Callable[[E], bool] | None = None,
        once: bool = False,
    ) -> Callable[[Callable[[E], None]], Callable[[E], None]]:
        """Decorator to register an event handler.

        Args:
            event_type: Type of event to handle
            priority: Handler priority
            filter_func: Optional filter function
            once: If True, handler runs only once

        Returns:
            Decorator function
        """

        def decorator(func: Callable[[E], None]) -> Callable[[E], None]:
            handler = EventHandler(
                handler=func,
                priority=priority,
                filter_func=filter_func,
                once=once,
                name=func.__name__,
            )
            self.subscribe(event_type, handler)
            return func

        return decorator

    def subscribe(
        self,
        event_type: type[E],
        handler: EventHandler[E] | Callable[[E], None],
    ) -> None:
        """Subscribe a handler to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Handler function or EventHandler wrapper
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        if callable(handler) and not isinstance(handler, EventHandler):
            handler = EventHandler(handler=handler)

        self._handlers[event_type].append(handler)
        # Sort by priority (highest first)
        self._handlers[event_type].sort(key=lambda h: h.priority.value, reverse=True)

    def subscribe_all(
        self, handler: EventHandler[Event] | Callable[[Event], None]
    ) -> None:
        """Subscribe a handler to all events.

        Args:
            handler: Handler function or EventHandler wrapper
        """
        if callable(handler) and not isinstance(handler, EventHandler):
            handler = EventHandler(handler=handler)

        self._global_handlers.append(handler)
        self._global_handlers.sort(key=lambda h: h.priority.value, reverse=True)

    def unsubscribe(self, event_type: type[E], handler_name: str | None = None) -> int:
        """Unsubscribe handlers from an event type.

        Args:
            event_type: Type of event
            handler_name: Optional specific handler name to remove

        Returns:
            Number of handlers removed
        """
        if event_type not in self._handlers:
            return 0

        if handler_name is None:
            count = len(self._handlers[event_type])
            self._handlers[event_type] = []
            return count

        original_count = len(self._handlers[event_type])
        self._handlers[event_type] = [
            h for h in self._handlers[event_type] if h.name != handler_name
        ]
        return original_count - len(self._handlers[event_type])

    def emit(self, event: Event) -> int:
        """Emit an event to all subscribed handlers.

        Args:
            event: Event to emit

        Returns:
            Number of handlers that processed the event
        """
        if self._paused:
            logger.debug("Event bus paused, skipping event: %s", type(event).__name__)
            return 0

        # Record in history
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit :]

        handlers_called = 0
        handlers_to_remove: list[tuple[type[Event], EventHandler[Any]]] = []

        # Get handlers for this event type
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        # Include handlers for parent event types
        for registered_type, type_handlers in self._handlers.items():
            if registered_type != event_type and isinstance(event, registered_type):
                handlers = handlers + type_handlers

        # Process specific handlers
        for handler in handlers:
            if handler.should_handle(event):
                try:
                    handler(event)
                    handlers_called += 1

                    if handler.once:
                        handlers_to_remove.append((event_type, handler))

                except Exception as e:
                    logger.error(
                        "Error in event handler %s: %s",
                        handler.name,
                        e,
                    )

        # Process global handlers
        for handler in self._global_handlers:
            if handler.should_handle(event):
                try:
                    handler(event)
                    handlers_called += 1
                except Exception as e:
                    logger.error(
                        "Error in global event handler %s: %s",
                        handler.name,
                        e,
                    )

        # Remove one-time handlers
        for evt_type, handler in handlers_to_remove:
            if evt_type in self._handlers:
                self._handlers[evt_type] = [
                    h for h in self._handlers[evt_type] if h != handler
                ]

        # Send to webhooks
        self._send_to_webhooks(event)

        return handlers_called

    def pause(self) -> None:
        """Pause event processing."""
        self._paused = True

    def resume(self) -> None:
        """Resume event processing."""
        self._paused = False

    @property
    def is_paused(self) -> bool:
        """Check if event bus is paused."""
        return self._paused

    def get_history(
        self,
        event_type: type[Event] | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """Get recent event history.

        Args:
            event_type: Optional filter by event type
            limit: Maximum events to return

        Returns:
            List of recent events
        """
        if event_type is None:
            return self._history[-limit:]

        filtered: list[Event] = [e for e in self._history if isinstance(e, event_type)]
        return filtered[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._history = []

    def get_handler_count(self, event_type: type[Event] | None = None) -> int:
        """Get number of registered handlers.

        Args:
            event_type: Optional specific event type

        Returns:
            Handler count
        """
        if event_type is None:
            return sum(len(h) for h in self._handlers.values()) + len(
                self._global_handlers
            )
        return len(self._handlers.get(event_type, []))

    # Webhook support
    def add_webhook(self, config: WebhookConfig) -> None:
        """Add a webhook endpoint.

        Args:
            config: Webhook configuration
        """
        self._webhooks.append(config)

    def remove_webhook(self, url: str) -> bool:
        """Remove a webhook by URL.

        Args:
            url: Webhook URL to remove

        Returns:
            True if removed, False if not found
        """
        original_count = len(self._webhooks)
        self._webhooks = [w for w in self._webhooks if w.url != url]
        return len(self._webhooks) < original_count

    def _send_to_webhooks(self, event: Event) -> None:
        """Send event to configured webhooks."""
        for webhook in self._webhooks:
            if webhook.should_send(event):
                try:
                    webhook.send(event)
                except Exception as e:
                    logger.error("Webhook error for %s: %s", webhook.url, e)


# =============================================================================
# Webhook Support
# =============================================================================


@dataclass
class WebhookConfig:
    """Configuration for webhook endpoints."""

    url: str
    event_types: list[type[Event]] | None = None  # None = all events
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 5.0
    retry_count: int = 3
    enabled: bool = True

    def should_send(self, event: Event) -> bool:
        """Check if this event should be sent to the webhook."""
        if not self.enabled:
            return False
        if self.event_types is None:
            return True
        return any(isinstance(event, t) for t in self.event_types)

    def send(self, event: Event) -> bool:
        """Send event to webhook.

        Args:
            event: Event to send

        Returns:
            True if successful
        """
        data = json.dumps(event.to_dict()).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}

        for attempt in range(self.retry_count):
            try:
                req = Request(self.url, data=data, headers=headers, method="POST")
                with urlopen(req, timeout=self.timeout) as response:
                    if response.status < 300:
                        return True
            except URLError as e:
                logger.warning(
                    "Webhook attempt %d failed for %s: %s",
                    attempt + 1,
                    self.url,
                    e,
                )
                if attempt < self.retry_count - 1:
                    time.sleep(0.5 * (attempt + 1))

        return False


# =============================================================================
# Event Aggregator
# =============================================================================


class EventAggregator:
    """Aggregates events for batch processing or reporting."""

    def __init__(self, bus: EventBus) -> None:
        """Initialize aggregator.

        Args:
            bus: Event bus to monitor
        """
        self.bus = bus
        self._counts: dict[str, int] = {}
        self._start_time = datetime.now(timezone.utc)

        # Subscribe to all events
        bus.subscribe_all(self._track_event)

    def _track_event(self, event: Event) -> None:
        """Track event for aggregation."""
        event_type = type(event).__name__
        self._counts[event_type] = self._counts.get(event_type, 0) + 1

    def get_counts(self) -> dict[str, int]:
        """Get event counts.

        Returns:
            Dictionary of event type to count
        """
        return self._counts.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get aggregation summary.

        Returns:
            Summary dictionary
        """
        elapsed = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        total = sum(self._counts.values())

        return {
            "total_events": total,
            "events_per_minute": (total / elapsed) * 60 if elapsed > 0 else 0,
            "event_counts": self._counts,
            "tracking_duration_seconds": elapsed,
        }

    def reset(self) -> None:
        """Reset aggregation counters."""
        self._counts = {}
        self._start_time = datetime.now(timezone.utc)


# =============================================================================
# Convenience Functions
# =============================================================================

# Global event bus instance
_default_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the default event bus instance.

    Returns:
        Default EventBus
    """
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def emit(event: Event) -> int:
    """Emit an event on the default bus.

    Args:
        event: Event to emit

    Returns:
        Number of handlers called
    """
    return get_event_bus().emit(event)


def on(
    event_type: type[E],
    priority: EventPriority = EventPriority.NORMAL,
) -> Callable[[Callable[[E], None]], Callable[[E], None]]:
    """Decorator to register handler on default bus.

    Args:
        event_type: Type of event to handle
        priority: Handler priority

    Returns:
        Decorator function
    """
    return get_event_bus().on(event_type, priority)


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests() -> bool:
    """Create unit tests for this module."""
    from test_framework import TestSuite

    suite = TestSuite("Event System", "events.py")
    suite.start_suite()

    def test_event_base():
        event = StoryDiscovered(title="Test", url="http://test.com")
        assert event.title == "Test"
        data = event.to_dict()
        assert data["type"] == "StoryDiscovered"

    def test_event_timestamp():
        event = StoryPublished()
        assert event.timestamp is not None
        assert event.timestamp <= datetime.now(timezone.utc)

    def test_event_bus_init():
        bus = EventBus()
        assert bus.get_handler_count() == 0

    def test_event_bus_subscribe():
        bus = EventBus()
        bus.subscribe(StoryDiscovered, lambda e: None)
        assert bus.get_handler_count(StoryDiscovered) == 1

    def test_event_bus_emit():
        bus = EventBus()
        received = []

        def handler(e: StoryDiscovered) -> None:
            received.append(e)

        bus.subscribe(StoryDiscovered, handler)
        bus.emit(StoryDiscovered(title="Test"))
        assert len(received) == 1

    def test_event_bus_decorator():
        bus = EventBus()
        received = []

        @bus.on(StoryPublished)
        def handler(e: StoryPublished) -> None:
            received.append(e)

        bus.emit(StoryPublished(title="Test"))
        assert len(received) == 1

    def test_event_bus_priority():
        bus = EventBus()
        order = []

        bus.subscribe(
            StoryDiscovered,
            EventHandler(lambda e: order.append("low"), priority=EventPriority.LOW),
        )
        bus.subscribe(
            StoryDiscovered,
            EventHandler(lambda e: order.append("high"), priority=EventPriority.HIGH),
        )
        bus.emit(StoryDiscovered())
        assert order == ["high", "low"]

    def test_event_bus_filter():
        bus = EventBus()
        received: list[StoryDiscovered] = []

        def filter_quality(e: StoryDiscovered) -> bool:
            return e.quality_score > 0.5

        handler: EventHandler[StoryDiscovered] = EventHandler(
            handler=lambda e: received.append(e),
            filter_func=filter_quality,
        )
        bus.subscribe(StoryDiscovered, handler)

        bus.emit(StoryDiscovered(quality_score=0.3))
        bus.emit(StoryDiscovered(quality_score=0.7))
        assert len(received) == 1

    def test_event_bus_unsubscribe():
        bus = EventBus()
        bus.subscribe(StoryDiscovered, EventHandler(lambda e: None, name="test"))
        assert bus.get_handler_count(StoryDiscovered) == 1
        bus.unsubscribe(StoryDiscovered, "test")
        assert bus.get_handler_count(StoryDiscovered) == 0

    def test_event_bus_history():
        bus = EventBus()
        bus.emit(StoryDiscovered(title="One"))
        bus.emit(StoryDiscovered(title="Two"))
        history = bus.get_history()
        assert len(history) == 2

    def test_event_bus_pause():
        bus = EventBus()
        received = []
        bus.subscribe(StoryDiscovered, lambda e: received.append(e))

        bus.pause()
        bus.emit(StoryDiscovered())
        assert len(received) == 0

        bus.resume()
        bus.emit(StoryDiscovered())
        assert len(received) == 1

    def test_event_bus_global_handler():
        bus = EventBus()
        received = []
        bus.subscribe_all(lambda e: received.append(e))

        bus.emit(StoryDiscovered())
        bus.emit(StoryPublished())
        assert len(received) == 2

    def test_webhook_config():
        config = WebhookConfig(url="http://test.com/hook")
        assert config.enabled is True
        assert config.should_send(StoryDiscovered())

    def test_event_aggregator():
        bus = EventBus()
        agg = EventAggregator(bus)

        bus.emit(StoryDiscovered())
        bus.emit(StoryDiscovered())
        bus.emit(StoryPublished())

        counts = agg.get_counts()
        assert counts.get("StoryDiscovered") == 2
        assert counts.get("StoryPublished") == 1

    suite.run_test(
        test_name="Event base class",
        test_func=test_event_base,
        test_summary="Tests Event base class functionality",
        method_description="Calls StoryDiscovered and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Event timestamp",
        test_func=test_event_timestamp,
        test_summary="Tests Event timestamp functionality",
        method_description="Calls StoryPublished and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Event bus init",
        test_func=test_event_bus_init,
        test_summary="Tests Event bus init functionality",
        method_description="Calls EventBus and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Event bus subscribe",
        test_func=test_event_bus_subscribe,
        test_summary="Tests Event bus subscribe functionality",
        method_description="Calls EventBus and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Event bus emit",
        test_func=test_event_bus_emit,
        test_summary="Tests Event bus emit functionality",
        method_description="Calls EventBus and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Event bus decorator",
        test_func=test_event_bus_decorator,
        test_summary="Tests Event bus decorator functionality",
        method_description="Calls EventBus and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Event bus priority",
        test_func=test_event_bus_priority,
        test_summary="Tests Event bus priority functionality",
        method_description="Calls EventBus and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Event bus filter",
        test_func=test_event_bus_filter,
        test_summary="Tests Event bus filter functionality",
        method_description="Calls EventBus and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Event bus unsubscribe",
        test_func=test_event_bus_unsubscribe,
        test_summary="Tests Event bus unsubscribe functionality",
        method_description="Calls EventBus and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Event bus history",
        test_func=test_event_bus_history,
        test_summary="Tests Event bus history functionality",
        method_description="Calls EventBus and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Event bus pause",
        test_func=test_event_bus_pause,
        test_summary="Tests Event bus pause functionality",
        method_description="Calls EventBus and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Event bus global handler",
        test_func=test_event_bus_global_handler,
        test_summary="Tests Event bus global handler functionality",
        method_description="Calls EventBus and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Webhook config",
        test_func=test_webhook_config,
        test_summary="Tests Webhook config functionality",
        method_description="Calls WebhookConfig and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Event aggregator",
        test_func=test_event_aggregator,
        test_summary="Tests Event aggregator functionality",
        method_description="Calls EventBus and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )

    return suite.finish_suite()
if __name__ == "__main__":
    # Demo usage
    print("Event-Driven Architecture Demo")
    print("=" * 50)

    bus = EventBus()

    @bus.on(StoryDiscovered)
    def on_story_discovered(event: StoryDiscovered) -> None:
        print(f"📰 Story discovered: {event.title}")

    @bus.on(StoryPublished)
    def on_story_published(event: StoryPublished) -> None:
        print(f"✅ Story published: {event.title}")

    @bus.on(PublishFailed)
    def on_publish_failed(event: PublishFailed) -> None:
        print(f"❌ Publish failed: {event.title} - {event.error}")

    # Emit some events
    bus.emit(StoryDiscovered(title="Tech News Story", quality_score=0.85))
    bus.emit(StoryPublished(title="Tech News Story", post_id="123"))

    # Show history
    print(f"\nEvent history: {len(bus.get_history())} events")
