"""Background daemon for scheduled LinkedIn publishing.

This daemon runs continuously and publishes stories when their scheduled time is due.
Only human-approved stories will be published.

Usage:
    python publish_daemon.py              # Run daemon in foreground
    python publish_daemon.py --once       # Check and publish once, then exit
    python publish_daemon.py --status     # Show daemon status and pending posts
    python publish_daemon.py --interval 60  # Check every 60 seconds (default: 300)

The daemon can be run as a Windows service, systemd service, or in a terminal.
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from database import Database
from linkedin_publisher import LinkedInPublisher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("publish_daemon.log"),
    ],
)
logger = logging.getLogger(__name__)


class PublishDaemon:
    """Background daemon for scheduled publishing."""

    def __init__(
        self,
        check_interval: int = 300,  # 5 minutes default
        db_path: Optional[str] = None,
    ):
        """Initialize the publish daemon.

        Args:
            check_interval: Seconds between checks for due posts
            db_path: Path to database (uses default if not specified)
        """
        self.check_interval = check_interval
        self.db = Database(db_path or Config.DB_PATH)
        self.publisher = LinkedInPublisher(self.db)
        self.running = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up graceful shutdown handlers."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def check_and_publish(self) -> tuple[int, int]:
        """Check for due posts and publish them.

        Returns:
            Tuple of (success_count, failure_count)
        """
        due_stories = self.db.get_scheduled_stories_due()

        if not due_stories:
            logger.debug("No stories due for publishing")
            return (0, 0)

        logger.info(f"Found {len(due_stories)} stories due for publishing")

        success_count = 0
        failure_count = 0

        for story in due_stories:
            # Double-check human approval (safety check)
            if not story.human_approved:
                logger.warning(
                    f"Skipping story {story.id} - not human approved "
                    "(this should not happen - check database query)"
                )
                continue

            logger.info(f"Publishing story {story.id}: {story.title[:50]}...")

            try:
                post_id = self.publisher.publish_immediately(
                    story, skip_validation=False
                )

                if post_id:
                    success_count += 1
                    logger.info(
                        f"✓ Successfully published story {story.id} "
                        f"(post ID: {post_id})"
                    )
                else:
                    failure_count += 1
                    logger.error(f"✗ Failed to publish story {story.id}")

            except Exception as e:
                failure_count += 1
                logger.exception(f"✗ Exception publishing story {story.id}: {e}")

            # Small delay between posts to avoid rate limiting
            if due_stories.index(story) < len(due_stories) - 1:
                time.sleep(5)

        return (success_count, failure_count)

    def get_status(self) -> dict:
        """Get current daemon status and pending posts."""
        scheduled = self.db.get_scheduled_stories()
        due = self.db.get_scheduled_stories_due()
        human_approved = self.db.get_approved_unpublished_stories(limit=100)

        now = datetime.now()

        return {
            "current_time": now.isoformat(),
            "scheduled_count": len(scheduled),
            "due_now_count": len(due),
            "human_approved_awaiting_schedule": len(human_approved),
            "next_scheduled": scheduled[0].scheduled_time.isoformat()
            if scheduled and scheduled[0].scheduled_time
            else None,
            "scheduled_stories": [
                {
                    "id": s.id,
                    "title": s.title[:50],
                    "scheduled_time": s.scheduled_time.isoformat()
                    if s.scheduled_time
                    else None,
                    "is_due": s.scheduled_time <= now if s.scheduled_time else False,
                    "human_approved": s.human_approved,
                }
                for s in scheduled[:10]
            ],
        }

    def run(self):
        """Run the daemon continuously."""
        logger.info("=" * 60)
        logger.info("PUBLISH DAEMON STARTED")
        logger.info(f"Check interval: {self.check_interval} seconds")
        logger.info("=" * 60)

        self.running = True

        while self.running:
            try:
                success, failures = self.check_and_publish()

                if success > 0 or failures > 0:
                    logger.info(
                        f"Publishing cycle complete: {success} succeeded, {failures} failed"
                    )

            except Exception as e:
                logger.exception(f"Error in publishing cycle: {e}")

            # Sleep in small increments to allow graceful shutdown
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)

        logger.info("Publish daemon stopped")

    def run_once(self):
        """Check and publish once, then exit."""
        logger.info("Running single publish check...")
        success, failures = self.check_and_publish()
        logger.info(f"Complete: {success} published, {failures} failed")
        return success, failures


def print_status(daemon: PublishDaemon):
    """Print daemon status in a readable format."""
    status = daemon.get_status()

    print("\n" + "=" * 60)
    print("PUBLISH DAEMON STATUS")
    print("=" * 60)
    print(f"Current time: {status['current_time']}")
    print(f"Scheduled posts: {status['scheduled_count']}")
    print(f"Due now: {status['due_now_count']}")
    print(
        f"Human-approved (not scheduled): {status['human_approved_awaiting_schedule']}"
    )

    if status["next_scheduled"]:
        print(f"Next scheduled: {status['next_scheduled']}")

    if status["scheduled_stories"]:
        print("\nUpcoming scheduled posts:")
        print("-" * 60)
        for story in status["scheduled_stories"]:
            due_marker = " [DUE NOW]" if story["is_due"] else ""
            approved_marker = "✓" if story["human_approved"] else "⚠"
            print(
                f"  {approved_marker} [{story['id']}] {story['title']} "
                f"@ {story['scheduled_time']}{due_marker}"
            )
    else:
        print("\nNo scheduled posts.")

    print("=" * 60)


def main():
    """Main entry point for the publish daemon."""
    parser = argparse.ArgumentParser(
        description="Background daemon for scheduled LinkedIn publishing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python publish_daemon.py              # Run daemon continuously
    python publish_daemon.py --once       # Publish due posts once and exit
    python publish_daemon.py --status     # Show status and exit
    python publish_daemon.py --interval 60  # Check every 60 seconds
        """,
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check and publish once, then exit",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show daemon status and pending posts",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between publish checks (default: 300)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to database file (uses default if not specified)",
    )

    args = parser.parse_args()

    # Validate LinkedIn credentials before starting
    if not args.status:
        if not Config.LINKEDIN_ACCESS_TOKEN:
            logger.error("LINKEDIN_ACCESS_TOKEN not configured")
            sys.exit(1)
        if not Config.LINKEDIN_AUTHOR_URN:
            logger.error("LINKEDIN_AUTHOR_URN not configured")
            sys.exit(1)

    daemon = PublishDaemon(
        check_interval=args.interval,
        db_path=args.db,
    )

    if args.status:
        print_status(daemon)
    elif args.once:
        success, failures = daemon.run_once()
        sys.exit(0 if failures == 0 else 1)
    else:
        daemon.run()


if __name__ == "__main__":
    main()
