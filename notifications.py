"""Notification system for Social Media Publisher.

This module provides notification capabilities for keeping users
informed about operations, publications, and analytics.

Features:
- Email notifications via SMTP
- Slack webhook integration
- Multiple notification channels
- Rate limiting to prevent spam
- Notification templates
- Daily/weekly digest support

Example:
    notifier = NotificationManager()
    notifier.add_channel(SlackChannel(webhook_url="..."))
    notifier.notify(
        "post_published",
        {"title": "My Post", "url": "https://linkedin.com/..."}
    )
"""

from __future__ import annotations

import json
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from api_client import api_client

if TYPE_CHECKING:
    from test_framework import TestSuite


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Notification:
    """A notification to be sent."""

    event_type: str
    title: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"  # low, normal, high, critical
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sent_at: datetime | None = None
    channel: str = ""
    success: bool = False
    error: str | None = None


@dataclass
class NotificationResult:
    """Result of sending a notification."""

    success: bool
    channel: str
    message: str = ""
    error: str | None = None
    sent_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Notification Templates
# =============================================================================


NOTIFICATION_TEMPLATES: dict[str, dict[str, str]] = {
    "post_published": {
        "title": "✅ Post Published",
        "message": 'Your post "{title}" was published successfully.\n\nView it here: {url}',
        "priority": "normal",
    },
    "post_failed": {
        "title": "❌ Post Failed",
        "message": 'Failed to publish post "{title}".\n\nError: {error}',
        "priority": "high",
    },
    "image_generated": {
        "title": "🖼️ Image Generated",
        "message": 'Image generated for "{title}".\n\nQuality score: {quality_score}',
        "priority": "low",
    },
    "image_failed": {
        "title": "⚠️ Image Generation Failed",
        "message": 'Failed to generate image for "{title}".\n\nError: {error}',
        "priority": "normal",
    },
    "engagement_milestone": {
        "title": "🎉 Engagement Milestone",
        "message": 'Your post "{title}" reached {milestone}!\n\nCurrent stats:\n- Likes: {likes}\n- Comments: {comments}\n- Shares: {shares}',
        "priority": "normal",
    },
    "daily_summary": {
        "title": "📊 Daily Summary",
        "message": "Daily activity summary for {date}:\n\n- Posts published: {posts_published}\n- Total engagement: {total_engagement}\n- Stories processed: {stories_processed}",
        "priority": "low",
    },
    "weekly_digest": {
        "title": "📈 Weekly Digest",
        "message": "Weekly performance report ({start_date} to {end_date}):\n\n- Posts: {total_posts}\n- Total reach: {total_reach}\n- Best performing: {best_post}",
        "priority": "low",
    },
    "error_alert": {
        "title": "🚨 Error Alert",
        "message": "An error occurred in {component}:\n\n{error_message}\n\nTime: {timestamp}",
        "priority": "high",
    },
    "rate_limit_warning": {
        "title": "⏳ Rate Limit Warning",
        "message": "Approaching rate limit for {service}.\n\nRemaining: {remaining}\nResets at: {reset_time}",
        "priority": "normal",
    },
    "system_health": {
        "title": "💚 System Health",
        "message": "System health check:\n\n- Status: {status}\n- Uptime: {uptime}\n- Last error: {last_error}",
        "priority": "low",
    },
}


def format_notification(
    event_type: str,
    data: dict[str, Any],
) -> tuple[str, str, str]:
    """
    Format a notification using templates.

    Args:
        event_type: Type of notification event
        data: Data to populate the template

    Returns:
        Tuple of (title, message, priority)
    """
    template = NOTIFICATION_TEMPLATES.get(
        event_type,
        {
            "title": event_type.replace("_", " ").title(),
            "message": str(data),
            "priority": "normal",
        },
    )

    title = template["title"]

    # Format message with data, handling missing keys gracefully
    message = template["message"]
    for key, value in data.items():
        message = message.replace(f"{{{key}}}", str(value))

    # Remove any unformatted placeholders
    import re

    message = re.sub(r"\{[^}]+\}", "[N/A]", message)

    priority = template.get("priority", "normal")

    return title, message, priority


# =============================================================================
# Notification Channels
# =============================================================================


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the channel name."""
        pass

    @abstractmethod
    def send(self, notification: Notification) -> NotificationResult:
        """Send a notification through this channel."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the channel is properly configured."""
        pass


class SlackChannel(NotificationChannel):
    """Slack webhook notification channel."""

    def __init__(
        self,
        webhook_url: str,
        channel_name: str = "#notifications",
        username: str = "Social Media Publisher",
        icon_emoji: str = ":robot_face:",
    ) -> None:
        """
        Initialize Slack channel.

        Args:
            webhook_url: Slack incoming webhook URL
            channel_name: Slack channel name (for display)
            username: Bot username
            icon_emoji: Bot emoji icon
        """
        self._webhook_url = webhook_url
        self._channel_name = channel_name
        self._username = username
        self._icon_emoji = icon_emoji

    @property
    def name(self) -> str:
        return f"slack:{self._channel_name}"

    def send(self, notification: Notification) -> NotificationResult:
        """Send notification to Slack."""
        if not self._webhook_url:
            return NotificationResult(
                success=False,
                channel=self.name,
                error="Webhook URL not configured",
            )

        # Build Slack message payload
        color = self._get_color_for_priority(notification.priority)

        payload = {
            "username": self._username,
            "icon_emoji": self._icon_emoji,
            "attachments": [
                {
                    "color": color,
                    "title": notification.title,
                    "text": notification.message,
                    "footer": "Social Media Publisher",
                    "ts": int(notification.created_at.timestamp()),
                }
            ],
        }

        try:
            # Use centralized HTTP client for rate limiting
            response = api_client.http_post(
                self._webhook_url,
                json=payload,
                timeout=10,
                endpoint="slack_webhook",
            )

            if response.status_code == 200:
                return NotificationResult(
                    success=True,
                    channel=self.name,
                    message="Sent to Slack",
                )
            else:
                return NotificationResult(
                    success=False,
                    channel=self.name,
                    error=f"HTTP {response.status_code}",
                )

        except requests.exceptions.RequestException as e:
            return NotificationResult(
                success=False,
                channel=self.name,
                error=str(e),
            )
        except Exception as e:
            return NotificationResult(
                success=False,
                channel=self.name,
                error=str(e),
            )

    def _get_color_for_priority(self, priority: str) -> str:
        """Get Slack attachment color for priority."""
        colors = {
            "low": "#36a64f",  # Green
            "normal": "#2196F3",  # Blue
            "high": "#ff9800",  # Orange
            "critical": "#f44336",  # Red
        }
        return colors.get(priority, colors["normal"])

    def test_connection(self) -> bool:
        """Test Slack webhook connectivity."""
        if not self._webhook_url:
            return False

        # Just check URL format, don't actually send
        return self._webhook_url.startswith("https://hooks.slack.com/")


class EmailChannel(NotificationChannel):
    """Email notification channel via SMTP."""

    def __init__(
        self,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        from_email: str = "",
        to_emails: list[str] | None = None,
        use_tls: bool = True,
    ) -> None:
        """
        Initialize email channel.

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: Sender email address
            to_emails: List of recipient email addresses
            use_tls: Whether to use TLS
        """
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._username = username
        self._password = password
        self._from_email = from_email or username
        self._to_emails = to_emails or []
        self._use_tls = use_tls

    @property
    def name(self) -> str:
        return f"email:{self._from_email}"

    def send(self, notification: Notification) -> NotificationResult:
        """Send notification via email."""
        if not self._to_emails:
            return NotificationResult(
                success=False,
                channel=self.name,
                error="No recipients configured",
            )

        if not self._username or not self._password:
            return NotificationResult(
                success=False,
                channel=self.name,
                error="SMTP credentials not configured",
            )

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = notification.title
            msg["From"] = self._from_email
            msg["To"] = ", ".join(self._to_emails)

            # Create plain text body
            text_body = f"{notification.title}\n\n{notification.message}"
            msg.attach(MIMEText(text_body, "plain"))

            # Create HTML body
            html_body = self._create_html_body(notification)
            msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                if self._use_tls:
                    server.starttls()
                server.login(self._username, self._password)
                server.sendmail(
                    self._from_email,
                    self._to_emails,
                    msg.as_string(),
                )

            return NotificationResult(
                success=True,
                channel=self.name,
                message=f"Email sent to {len(self._to_emails)} recipients",
            )

        except smtplib.SMTPException as e:
            return NotificationResult(
                success=False,
                channel=self.name,
                error=f"SMTP error: {e}",
            )
        except Exception as e:
            return NotificationResult(
                success=False,
                channel=self.name,
                error=str(e),
            )

    def _create_html_body(self, notification: Notification) -> str:
        """Create HTML email body."""
        priority_color = {
            "low": "#4CAF50",
            "normal": "#2196F3",
            "high": "#FF9800",
            "critical": "#F44336",
        }.get(notification.priority, "#2196F3")

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <div style="border-left: 4px solid {priority_color}; padding-left: 15px;">
                <h2 style="margin: 0 0 10px 0; color: #333;">{notification.title}</h2>
                <p style="color: #666; white-space: pre-wrap;">{notification.message}</p>
                <hr style="border: none; border-top: 1px solid #eee; margin: 15px 0;">
                <p style="color: #999; font-size: 12px;">
                    Sent by Social Media Publisher • {notification.created_at.strftime("%Y-%m-%d %H:%M UTC")}
                </p>
            </div>
        </body>
        </html>
        """

    def test_connection(self) -> bool:
        """Test SMTP connectivity."""
        if not self._smtp_host or not self._username:
            return False

        try:
            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=5) as server:
                if self._use_tls:
                    server.starttls()
                server.noop()  # Just test connection
            return True
        except Exception:
            return False


class ConsoleChannel(NotificationChannel):
    """Console/stdout notification channel for debugging."""

    def __init__(self, prefix: str = "[NOTIFY]") -> None:
        """Initialize console channel."""
        self._prefix = prefix

    @property
    def name(self) -> str:
        return "console"

    def send(self, notification: Notification) -> NotificationResult:
        """Print notification to console."""
        timestamp = notification.created_at.strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{self._prefix} [{notification.priority.upper()}] {timestamp}")
        print(f"  {notification.title}")
        print(f"  {notification.message.replace(chr(10), chr(10) + '  ')}")
        print()

        return NotificationResult(
            success=True,
            channel=self.name,
            message="Printed to console",
        )

    def test_connection(self) -> bool:
        """Console is always available."""
        return True


class FileChannel(NotificationChannel):
    """File-based notification channel for logging."""

    def __init__(
        self,
        log_path: str | Path = "notifications.log",
        json_format: bool = False,
    ) -> None:
        """
        Initialize file channel.

        Args:
            log_path: Path to log file
            json_format: Whether to use JSON format
        """
        self._log_path = Path(log_path)
        self._json_format = json_format

    @property
    def name(self) -> str:
        return f"file:{self._log_path.name}"

    def send(self, notification: Notification) -> NotificationResult:
        """Write notification to file."""
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                if self._json_format:
                    record = {
                        "timestamp": notification.created_at.isoformat(),
                        "event_type": notification.event_type,
                        "title": notification.title,
                        "message": notification.message,
                        "priority": notification.priority,
                        "data": notification.data,
                    }
                    f.write(json.dumps(record) + "\n")
                else:
                    timestamp = notification.created_at.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(
                        f"[{timestamp}] [{notification.priority.upper()}] {notification.title}\n"
                    )
                    f.write(f"  {notification.message}\n\n")

            return NotificationResult(
                success=True,
                channel=self.name,
                message=f"Written to {self._log_path}",
            )

        except Exception as e:
            return NotificationResult(
                success=False,
                channel=self.name,
                error=str(e),
            )

    def test_connection(self) -> bool:
        """Check if file path is writable."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False


# =============================================================================
# Notification Manager
# =============================================================================


class NotificationManager:
    """
    Central manager for all notification channels.

    Features:
    - Multiple channel support
    - Rate limiting
    - Priority filtering
    - History tracking
    """

    def __init__(
        self,
        rate_limit_per_hour: int = 50,
        min_priority: str = "low",
    ) -> None:
        """
        Initialize notification manager.

        Args:
            rate_limit_per_hour: Max notifications per hour
            min_priority: Minimum priority to send
        """
        self._channels: list[NotificationChannel] = []
        self._history: list[Notification] = []
        self._max_history = 500
        self._rate_limit = rate_limit_per_hour
        self._min_priority = min_priority
        self._priority_levels = ["low", "normal", "high", "critical"]

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self._channels.append(channel)

    def remove_channel(self, channel_name: str) -> bool:
        """Remove a channel by name."""
        for i, channel in enumerate(self._channels):
            if channel.name == channel_name:
                self._channels.pop(i)
                return True
        return False

    def list_channels(self) -> list[str]:
        """List all configured channel names."""
        return [c.name for c in self._channels]

    def notify(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        channels: list[str] | None = None,
    ) -> list[NotificationResult]:
        """
        Send a notification.

        Args:
            event_type: Type of notification event
            data: Data for the notification template
            channels: Optional list of specific channels to use

        Returns:
            List of results from each channel
        """
        data = data or {}

        # Format notification from template
        title, message, priority = format_notification(event_type, data)

        # Check priority filter
        if self._priority_levels.index(priority) < self._priority_levels.index(
            self._min_priority
        ):
            return []

        # Check rate limit
        if self._is_rate_limited():
            return [
                NotificationResult(
                    success=False,
                    channel="all",
                    error="Rate limit exceeded",
                )
            ]

        # Create notification object
        notification = Notification(
            event_type=event_type,
            title=title,
            message=message,
            data=data,
            priority=priority,
        )

        # Send to channels
        results: list[NotificationResult] = []
        target_channels = self._get_target_channels(channels)

        for channel in target_channels:
            result = channel.send(notification)
            results.append(result)

            # Update notification with result
            notification.channel = channel.name
            notification.success = result.success
            notification.sent_at = result.sent_at
            notification.error = result.error

        # Store in history
        self._add_to_history(notification)

        return results

    def _get_target_channels(
        self,
        channel_names: list[str] | None,
    ) -> list[NotificationChannel]:
        """Get channels to send to."""
        if not channel_names:
            return self._channels

        return [c for c in self._channels if c.name in channel_names]

    def _is_rate_limited(self) -> bool:
        """Check if rate limit is exceeded."""
        if self._rate_limit <= 0:
            return False

        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_count = sum(1 for n in self._history if n.created_at > cutoff)

        return recent_count >= self._rate_limit

    def _add_to_history(self, notification: Notification) -> None:
        """Add notification to history."""
        self._history.append(notification)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

    def get_history(
        self,
        limit: int = 50,
        event_type: str | None = None,
    ) -> list[Notification]:
        """Get notification history."""
        history = self._history

        if event_type:
            history = [n for n in history if n.event_type == event_type]

        return list(reversed(history[-limit:]))

    def test_all_channels(self) -> dict[str, bool]:
        """Test connectivity for all channels."""
        return {c.name: c.test_connection() for c in self._channels}

    def send_test_notification(self) -> list[NotificationResult]:
        """Send a test notification to all channels."""
        return self.notify(
            "system_health",
            {
                "status": "OK",
                "uptime": "test",
                "last_error": "None (test)",
            },
        )


# =============================================================================
# Digest Generator
# =============================================================================


class DigestGenerator:
    """
    Generate daily and weekly digest notifications.
    """

    def __init__(self, manager: NotificationManager) -> None:
        """Initialize digest generator."""
        self._manager = manager
        self._daily_stats: dict[str, Any] = {}
        self._weekly_stats: list[dict[str, Any]] = []

    def record_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Record an event for digest inclusion."""
        data = data or {}

        # Update daily stats
        if event_type not in self._daily_stats:
            self._daily_stats[event_type] = {
                "count": 0,
                "data": [],
            }

        self._daily_stats[event_type]["count"] += 1
        self._daily_stats[event_type]["data"].append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **data,
            }
        )

    def generate_daily_summary(self) -> dict[str, Any]:
        """Generate daily summary data."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        posts_published = self._daily_stats.get("post_published", {}).get("count", 0)
        posts_failed = self._daily_stats.get("post_failed", {}).get("count", 0)
        stories_processed = self._daily_stats.get("story_processed", {}).get("count", 0)

        # Calculate total engagement from events
        engagement_events = self._daily_stats.get("engagement_update", {}).get(
            "data", []
        )
        total_engagement = sum(e.get("engagement", 0) for e in engagement_events)

        return {
            "date": today,
            "posts_published": posts_published,
            "posts_failed": posts_failed,
            "stories_processed": stories_processed,
            "total_engagement": total_engagement,
        }

    def send_daily_summary(self) -> list[NotificationResult]:
        """Send daily summary notification."""
        summary = self.generate_daily_summary()
        results = self._manager.notify("daily_summary", summary)

        # Archive daily stats to weekly
        self._weekly_stats.append(summary)

        # Reset daily stats
        self._daily_stats = {}

        return results

    def send_weekly_digest(self) -> list[NotificationResult]:
        """Send weekly digest notification."""
        if not self._weekly_stats:
            return []

        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")

        total_posts = sum(d.get("posts_published", 0) for d in self._weekly_stats)
        total_reach = sum(d.get("total_engagement", 0) for d in self._weekly_stats)

        # Find best post (placeholder - would need actual tracking)
        best_post = "N/A"

        digest_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_posts": total_posts,
            "total_reach": total_reach,
            "best_post": best_post,
        }

        results = self._manager.notify("weekly_digest", digest_data)

        # Reset weekly stats
        self._weekly_stats = []

        return results


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests() -> "TestSuite":
    """Create unit tests for this module."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_framework import TestSuite

    suite = TestSuite("Notifications Tests")

    def test_notification_creation():
        notif = Notification(
            event_type="test",
            title="Test Title",
            message="Test message",
        )
        assert notif.event_type == "test"
        assert notif.priority == "normal"
        assert notif.success is False

    def test_format_notification():
        title, message, priority = format_notification(
            "post_published",
            {"title": "My Post", "url": "https://example.com"},
        )
        assert "Published" in title
        assert "My Post" in message
        assert "example.com" in message

    def test_format_notification_missing_key():
        title, message, priority = format_notification(
            "post_published",
            {"title": "My Post"},  # Missing url
        )
        assert "[N/A]" in message

    def test_format_notification_unknown():
        title, message, priority = format_notification(
            "unknown_event",
            {"key": "value"},
        )
        assert "Unknown Event" in title

    def test_console_channel():
        channel = ConsoleChannel()
        assert channel.name == "console"
        assert channel.test_connection() is True

    def test_console_channel_send():
        import io
        import sys

        channel = ConsoleChannel("[TEST]")
        notif = Notification(
            event_type="test",
            title="Test",
            message="Message",
        )

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        result = channel.send(notif)

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        assert result.success is True
        assert "[TEST]" in output

    def test_file_channel_name():
        channel = FileChannel("test.log")
        assert "file:" in channel.name
        assert "test.log" in channel.name

    def test_slack_channel_name():
        channel = SlackChannel(
            webhook_url="https://hooks.slack.com/test",
            channel_name="#test",
        )
        assert "slack:" in channel.name
        assert "#test" in channel.name

    def test_slack_channel_test_connection():
        channel = SlackChannel(webhook_url="")
        assert channel.test_connection() is False

        channel = SlackChannel(webhook_url="https://hooks.slack.com/test")
        assert channel.test_connection() is True

    def test_email_channel_name():
        channel = EmailChannel(username="test@example.com")
        assert "email:" in channel.name

    def test_notification_manager_init():
        manager = NotificationManager()
        assert manager.list_channels() == []

    def test_notification_manager_add_channel():
        manager = NotificationManager()
        manager.add_channel(ConsoleChannel())
        assert len(manager.list_channels()) == 1

    def test_notification_manager_remove_channel():
        manager = NotificationManager()
        manager.add_channel(ConsoleChannel())
        removed = manager.remove_channel("console")
        assert removed is True
        assert len(manager.list_channels()) == 0

    def test_notification_manager_test_channels():
        manager = NotificationManager()
        manager.add_channel(ConsoleChannel())
        results = manager.test_all_channels()
        assert results["console"] is True

    def test_digest_generator():
        manager = NotificationManager()
        digest = DigestGenerator(manager)

        digest.record_event("post_published", {"title": "Test"})
        digest.record_event("post_published", {"title": "Test 2"})

        summary = digest.generate_daily_summary()
        assert summary["posts_published"] == 2

    suite.add_test("Notification creation", test_notification_creation)
    suite.add_test("Format notification", test_format_notification)
    suite.add_test(
        "Format notification missing key", test_format_notification_missing_key
    )
    suite.add_test("Format notification unknown", test_format_notification_unknown)
    suite.add_test("Console channel", test_console_channel)
    suite.add_test("Console channel send", test_console_channel_send)
    suite.add_test("File channel name", test_file_channel_name)
    suite.add_test("Slack channel name", test_slack_channel_name)
    suite.add_test("Slack channel test connection", test_slack_channel_test_connection)
    suite.add_test("Email channel name", test_email_channel_name)
    suite.add_test("Notification manager init", test_notification_manager_init)
    suite.add_test(
        "Notification manager add channel", test_notification_manager_add_channel
    )
    suite.add_test(
        "Notification manager remove channel", test_notification_manager_remove_channel
    )
    suite.add_test(
        "Notification manager test channels", test_notification_manager_test_channels
    )
    suite.add_test("Digest generator", test_digest_generator)

    return suite


if __name__ == "__main__":
    suite = _create_module_tests()
    suite.run()
