"""
Social Media Publisher - Main Entry Point

Automated news story discovery, image generation, and LinkedIn publishing.
"""

import time
import logging
import argparse
from datetime import datetime, timedelta

from config import Config
from database import Database
from searcher import StorySearcher
from image_generator import ImageGenerator
from verifier import ContentVerifier
from scheduler import Scheduler
from linkedin_publisher import LinkedInPublisher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ContentEngine:
    """Main orchestrator for the content publishing pipeline."""

    def __init__(self):
        """Initialize all components."""
        logger.info("Initializing Content Engine...")

        # Ensure directories exist
        Config.ensure_directories()

        # Initialize database
        self.db = Database()

        # Initialize components
        self.searcher = StorySearcher(self.db)
        self.image_generator = ImageGenerator(self.db)
        self.verifier = ContentVerifier(self.db)
        self.scheduler = Scheduler(self.db)
        self.publisher = LinkedInPublisher(self.db)

        # Track last search cycle
        self._last_search_cycle: datetime | None = None

        logger.info("Content Engine initialized successfully")

    def validate_configuration(self) -> bool:
        """Validate all required configuration is present."""
        errors = Config.validate()
        if errors:
            logger.error("Configuration errors:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        return True

    def run_search_cycle(self) -> None:
        """
        Run a complete search cycle:
        1. Search for new stories
        2. Generate images for qualifying stories
        3. Verify content quality
        4. Schedule for publication
        5. Clean up old unused stories
        """
        logger.info("=" * 60)
        logger.info("Starting Search Cycle")
        logger.info("=" * 60)

        try:
            # Step 1: Search for new stories
            logger.info("Step 1: Searching for new stories...")
            new_stories = self.searcher.search_and_process()
            logger.info(f"Found {new_stories} new stories")

            # Step 2: Generate images
            logger.info("Step 2: Generating images...")
            images_generated = self.image_generator.generate_images_for_stories()
            logger.info(f"Generated {images_generated} images")

            # Step 3: Verify content
            logger.info("Step 3: Verifying content...")
            approved, rejected = self.verifier.verify_pending_content()
            logger.info(f"Verification: {approved} approved, {rejected} rejected")

            # Step 4: Schedule stories
            logger.info("Step 4: Scheduling stories...")
            scheduled = self.scheduler.schedule_stories()
            logger.info(f"Scheduled {len(scheduled)} stories for publication")

            # Log schedule summary
            schedule_summary = self.scheduler.get_schedule_summary()
            logger.info(schedule_summary)

            # Step 5: Cleanup old stories
            logger.info("Step 5: Cleaning up old stories...")
            cutoff = datetime.now() - timedelta(days=Config.EXCLUSION_PERIOD_DAYS)
            self.db.delete_old_unused_stories(cutoff)
            self.image_generator.cleanup_orphaned_images()

            # Update tracking
            self._last_search_cycle = datetime.now()

            # Print statistics
            stats = self.db.get_statistics()
            logger.info("Database Statistics:")
            logger.info(f"  Total stories: {stats['total_stories']}")
            logger.info(f"  Published: {stats['published_count']}")
            logger.info(f"  Scheduled: {stats['scheduled_count']}")
            logger.info(f"  Available: {stats['available_count']}")
            logger.info(f"  Pending verification: {stats['pending_verification']}")

        except Exception as e:
            logger.error(f"Search cycle failed: {e}")
            raise

        logger.info("Search cycle complete")
        logger.info("=" * 60)

    def run_publish_check(self) -> tuple[int, int]:
        """
        Check for and publish any due stories.
        Returns (success_count, failure_count).
        """
        return self.publisher.publish_due_stories()

    def should_run_search_cycle(self) -> bool:
        """Determine if a new search cycle should be run."""
        if self._last_search_cycle is None:
            return True

        hours_since_last = (
            datetime.now() - self._last_search_cycle
        ).total_seconds() / 3600

        return hours_since_last >= Config.SEARCH_CYCLE_HOURS

    def run_continuous(self) -> None:
        """
        Run the content engine continuously.
        - Checks for due publications every minute
        - Runs search cycles at configured intervals
        """
        logger.info("Starting continuous operation mode")
        logger.info(f"Search cycle interval: {Config.SEARCH_CYCLE_HOURS} hours")
        logger.info(
            f"Publisher check interval: {Config.PUBLISHER_CHECK_INTERVAL_SECONDS} seconds"
        )

        # Initial search cycle
        self.run_search_cycle()

        try:
            while True:
                # Check for due publications
                success, failures = self.run_publish_check()
                if success > 0 or failures > 0:
                    logger.info(f"Publishing: {success} successful, {failures} failed")

                # Check if search cycle is due
                if self.should_run_search_cycle():
                    self.run_search_cycle()

                # Wait before next check
                time.sleep(Config.PUBLISHER_CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Shutdown requested. Exiting...")

    def run_once(self) -> None:
        """Run a single search cycle and exit."""
        self.run_search_cycle()

    def status(self) -> None:
        """Print current status."""
        stats = self.db.get_statistics()

        print("\n" + "=" * 60)
        print("Social Media Publisher - Status")
        print("=" * 60)

        # Configuration
        Config.print_config()

        # Database stats
        print("\n--- Database Statistics ---")
        print(f"  Total stories: {stats['total_stories']}")
        print(f"  Published: {stats['published_count']}")
        print(f"  Scheduled: {stats['scheduled_count']}")
        print(f"  Available for scheduling: {stats['available_count']}")
        print(f"  Pending verification: {stats['pending_verification']}")

        # Schedule
        print("\n--- Current Schedule ---")
        print(self.scheduler.get_schedule_summary())

        # LinkedIn status
        print("\n--- LinkedIn Connection ---")
        if self.publisher.test_connection():
            profile = self.publisher.get_profile_info()
            if profile:
                print(f"  Connected as: {profile.get('localizedFirstName', 'Unknown')}")
            else:
                print("  Connected (profile unavailable)")
        else:
            print("  Not connected or credentials invalid")

        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Social Media Publisher - Automated news publishing to LinkedIn"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one search cycle and exit",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status and exit",
    )
    parser.add_argument(
        "--publish-now",
        action="store_true",
        help="Publish any due stories and exit",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show configuration and exit",
    )

    args = parser.parse_args()

    # Create engine
    engine = ContentEngine()

    # Validate configuration
    if not engine.validate_configuration():
        logger.error("Configuration validation failed. Please check your .env file.")
        return 1

    # Handle commands
    if args.config:
        Config.print_config()
        return 0

    if args.status:
        engine.status()
        return 0

    if args.publish_now:
        success, failures = engine.run_publish_check()
        print(f"Published: {success} successful, {failures} failed")
        return 0

    if args.once:
        engine.run_once()
        return 0

    # Default: run continuously
    engine.run_continuous()
    return 0


if __name__ == "__main__":
    exit(main())
