"""
Social Media Publisher - Main Entry Point

Automated news story discovery, image generation, and LinkedIn publishing.
"""

import time
import logging
import argparse
import requests
import os
import subprocess
from datetime import datetime, timedelta

from google import genai  # type: ignore
from openai import OpenAI

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

        # Initialize GenAI client
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured")
        self.genai_client = genai.Client(api_key=Config.GEMINI_API_KEY)

        # Initialize Local LLM client (LM Studio)
        self.local_client = self._initialize_local_client()

        # Initialize components
        self.searcher = StorySearcher(self.db, self.genai_client, self.local_client)
        self.image_generator = ImageGenerator(
            self.db, self.genai_client, self.local_client
        )
        self.verifier = ContentVerifier(self.db, self.genai_client, self.local_client)
        self.scheduler = Scheduler(self.db)
        self.publisher = LinkedInPublisher(self.db)

        # Track last search cycle
        self._last_search_cycle: datetime | None = None

        logger.info("Content Engine initialized successfully")

    def _initialize_local_client(self) -> OpenAI | None:
        """Check if LM Studio is running, start it if necessary, and return a client."""
        if not Config.PREFER_LOCAL_LLM:
            return None

        logger.info(f"Checking for LM Studio at {Config.LM_STUDIO_BASE_URL}...")

        # Try to connect first
        client = self._get_lm_studio_client()
        if client:
            return client

        # If not running, try to start it
        logger.info("LM Studio not detected. Attempting to locate and start it...")
        lm_studio_path = self._find_lm_studio_executable()

        if lm_studio_path:
            try:
                logger.info(f"Starting LM Studio from {lm_studio_path}...")
                subprocess.Popen([lm_studio_path], start_new_session=True)

                # Wait for it to start (up to 30 seconds)
                logger.info("Waiting for LM Studio to initialize...")
                for _ in range(15):
                    time.sleep(2)
                    client = self._get_lm_studio_client()
                    if client:
                        return client
            except Exception as e:
                logger.error(f"Failed to start LM Studio: {e}")
        else:
            logger.warning(
                "Could not find LM Studio executable. Please start it manually."
            )

        return None

    def _get_lm_studio_client(self) -> OpenAI | None:
        """Attempt to get an OpenAI client for LM Studio and verify a model is loaded."""
        try:
            response = requests.get(f"{Config.LM_STUDIO_BASE_URL}/models", timeout=2)
            if response.status_code == 200:
                models = response.json().get("data", [])
                if not models:
                    print("\n" + "=" * 50)
                    print("LM Studio is running but NO MODEL IS LOADED.")
                    print("Please load a model in LM Studio to use local LLM features.")
                    print("=" * 50 + "\n")
                    input(
                        "Press Enter once you have loaded a model to continue, or Ctrl+C to skip..."
                    )
                    # Re-check after user input
                    return self._get_lm_studio_client()

                logger.info(f"LM Studio detected with {len(models)} model(s) loaded")
                return OpenAI(base_url=Config.LM_STUDIO_BASE_URL, api_key="lm-studio")
        except requests.exceptions.RequestException:
            pass
        return None

    def _find_lm_studio_executable(self) -> str | None:
        """Search for LM Studio executable in common Windows locations."""
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("ProgramFiles", "")

        potential_paths = [
            os.path.join(local_app_data, "Programs", "lm-studio", "LM Studio.exe"),
            os.path.join(program_files, "LM Studio", "LM Studio.exe"),
        ]

        for path in potential_paths:
            if os.path.exists(path):
                return path
        return None

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

        except RuntimeError as e:
            if "quota exceeded" in str(e).lower():
                logger.warning(f"Search cycle interrupted: {e}")
                print(f"\n[!] API Quota Exceeded: {e}")
            else:
                logger.error(f"Search cycle failed: {e}")
                raise
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


def interactive_menu(engine: ContentEngine) -> None:
    """Run interactive CLI menu for component testing."""
    while True:
        print("\n" + "=" * 60)
        print("Social Media Publisher - Debug Menu")
        print("=" * 60)
        print("\n  Component Testing:")
        print("    1. Test Story Search")
        print("    2. Test Image Generation")
        print("    3. Test Content Verification")
        print("    4. Test Scheduling")
        print("    5. Test LinkedIn Connection")
        print("    6. Test LinkedIn Publish (due stories)")
        print("\n  Database Operations:")
        print("    7. View Database Statistics")
        print("    8. List All Stories")
        print("    9. List Pending Stories")
        print("   10. List Scheduled Stories")
        print("   11. Cleanup Old Stories")
        print("\n  Configuration:")
        print("   12. Show Configuration")
        print("   13. Show Full Status")
        print("\n  Pipeline:")
        print("   14. Run Full Search Cycle")
        print("\n   0. Exit")
        print("=" * 60)

        try:
            choice = input("\nEnter choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if choice == "0":
            print("Exiting...")
            break
        elif choice == "1":
            _test_search(engine)
        elif choice == "2":
            _test_image_generation(engine)
        elif choice == "3":
            _test_verification(engine)
        elif choice == "4":
            _test_scheduling(engine)
        elif choice == "5":
            _test_linkedin_connection(engine)
        elif choice == "6":
            _test_linkedin_publish(engine)
        elif choice == "7":
            _show_database_stats(engine)
        elif choice == "8":
            _list_all_stories(engine)
        elif choice == "9":
            _list_pending_stories(engine)
        elif choice == "10":
            _list_scheduled_stories(engine)
        elif choice == "11":
            _cleanup_old_stories(engine)
        elif choice == "12":
            Config.print_config()
        elif choice == "13":
            engine.status()
        elif choice == "14":
            _run_full_cycle(engine)
        else:
            print("Invalid choice. Please try again.")

        input("\nPress Enter to continue...")


def _test_search(engine: ContentEngine) -> None:
    """Test the story search component."""
    print("\n--- Testing Story Search ---")
    print(f"Search prompt: {Config.SEARCH_PROMPT[:80]}...")
    print(f"Lookback days: {Config.SEARCH_LOOKBACK_DAYS}")
    print(f"Use last checked date: {Config.USE_LAST_CHECKED_DATE}")

    confirm = input("\nProceed with search? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        start_date = engine.searcher.get_search_start_date()
        print(f"Searching for stories since: {start_date}")

        new_count = engine.searcher.search_and_process()
        print(f"\nResult: Found and saved {new_count} new stories")
    except RuntimeError as e:
        if "quota exceeded" in str(e).lower():
            print(f"\n[!] API Quota Exceeded: {e}")
            print(
                "The search could not be completed because the Google GenAI API quota has been reached."
            )
            print("Please wait a few minutes or check your API usage limits.")
        else:
            print(f"\nError: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Search test failed")


def _test_image_generation(engine: ContentEngine) -> None:
    """Test the image generation component."""
    print("\n--- Testing Image Generation ---")

    stories = engine.db.get_stories_needing_images(Config.MIN_QUALITY_SCORE)
    print(f"Stories needing images: {len(stories)}")

    if not stories:
        print("No stories need images.")
        return

    for story in stories[:5]:  # Show first 5
        print(f"  - [{story.id}] {story.title[:50]}... (score: {story.quality_score})")

    confirm = input("\nGenerate images for these stories? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        count = engine.image_generator.generate_images_for_stories()
        print(f"\nResult: Generated {count} images")
    except RuntimeError as e:
        if "quota exceeded" in str(e).lower():
            print(f"\n[!] API Quota Exceeded: {e}")
            print(
                "Image generation could not be completed because the API quota has been reached."
            )
        else:
            print(f"\nError: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Image generation test failed")


def _test_verification(engine: ContentEngine) -> None:
    """Test the content verification component."""
    print("\n--- Testing Content Verification ---")

    stories = engine.db.get_stories_needing_verification()
    print(f"Stories pending verification: {len(stories)}")

    if not stories:
        print("No stories need verification.")
        return

    for story in stories[:5]:  # Show first 5
        print(f"  - [{story.id}] {story.title[:50]}...")

    confirm = input("\nVerify these stories? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        approved, rejected = engine.verifier.verify_pending_content()
        print(f"\nResult: {approved} approved, {rejected} rejected")
    except RuntimeError as e:
        if "quota exceeded" in str(e).lower():
            print(f"\n[!] API Quota Exceeded: {e}")
            print(
                "Content verification could not be completed because the API quota has been reached."
            )
        else:
            print(f"\nError: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Verification test failed")


def _test_scheduling(engine: ContentEngine) -> None:
    """Test the scheduling component."""
    print("\n--- Testing Scheduling ---")
    print(f"Stories per cycle: {Config.STORIES_PER_CYCLE}")
    print(f"Publish window: {Config.PUBLISH_WINDOW_HOURS} hours")
    print(
        f"Publish hours: {Config.PUBLISH_START_HOUR}:00 - {Config.PUBLISH_END_HOUR}:00"
    )
    print(f"Jitter: ±{Config.JITTER_MINUTES} minutes")

    available = engine.db.count_unpublished_stories()
    print(f"\nAvailable approved stories: {available}")

    if available == 0:
        print("No approved stories available to schedule.")
        return

    confirm = input("\nSchedule stories? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        scheduled = engine.scheduler.schedule_stories()
        print(f"\nResult: Scheduled {len(scheduled)} stories")
        print(engine.scheduler.get_schedule_summary())
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Scheduling test failed")


def _test_linkedin_connection(engine: ContentEngine) -> None:
    """Test LinkedIn API connection."""
    print("\n--- Testing LinkedIn Connection ---")
    print(f"Author URN: {Config.LINKEDIN_AUTHOR_URN or 'NOT SET'}")
    print(f"Access Token: {'SET' if Config.LINKEDIN_ACCESS_TOKEN else 'NOT SET'}")

    if not Config.LINKEDIN_ACCESS_TOKEN or not Config.LINKEDIN_AUTHOR_URN:
        print("\nLinkedIn credentials not configured.")
        return

    print("\nTesting connection...")
    try:
        if engine.publisher.test_connection():
            print("✓ Connection successful!")
            profile = engine.publisher.get_profile_info()
            if profile:
                print(
                    f"  Profile: {profile.get('localizedFirstName', '')} "
                    f"{profile.get('localizedLastName', '')}"
                )
        else:
            print("✗ Connection failed. Check your credentials.")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("LinkedIn connection test failed")


def _test_linkedin_publish(engine: ContentEngine) -> None:
    """Test publishing due stories to LinkedIn."""
    print("\n--- Testing LinkedIn Publish ---")

    due = engine.db.get_scheduled_stories_due()
    print(f"Stories due for publication: {len(due)}")

    if not due:
        print("No stories are due for publication.")
        return

    for story in due:
        print(f"  - [{story.id}] {story.title[:50]}...")

    confirm = input("\nPublish these stories now? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        success, failures = engine.publisher.publish_due_stories()
        print(f"\nResult: {success} published, {failures} failed")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("LinkedIn publish test failed")


def _show_database_stats(engine: ContentEngine) -> None:
    """Show database statistics."""
    print("\n--- Database Statistics ---")
    stats = engine.db.get_statistics()
    print(f"  Total stories: {stats['total_stories']}")
    print(f"  Published: {stats['published_count']}")
    print(f"  Scheduled: {stats['scheduled_count']}")
    print(f"  Available (approved, unpublished): {stats['available_count']}")
    print(f"  Pending verification: {stats['pending_verification']}")


def _list_all_stories(engine: ContentEngine) -> None:
    """List all stories in the database."""
    print("\n--- All Stories ---")
    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, quality_score, verification_status, publish_status, acquire_date
            FROM stories ORDER BY acquire_date DESC LIMIT 20
        """)
        rows = cursor.fetchall()

    if not rows:
        print("No stories in database.")
        return

    print(f"{'ID':<4} {'Score':<5} {'Verify':<10} {'Publish':<12} {'Title':<40}")
    print("-" * 75)
    for row in rows:
        title = row["title"][:38] + ".." if len(row["title"]) > 40 else row["title"]
        print(
            f"{row['id']:<4} {row['quality_score']:<5} {row['verification_status']:<10} "
            f"{row['publish_status']:<12} {title:<40}"
        )

    if len(rows) == 20:
        print("\n(Showing first 20 stories)")


def _list_pending_stories(engine: ContentEngine) -> None:
    """List stories pending verification."""
    print("\n--- Stories Pending Verification ---")
    stories = engine.db.get_stories_needing_verification()

    if not stories:
        print("No stories pending verification.")
        return

    for story in stories:
        has_image = "✓" if story.image_path else "✗"
        print(
            f"  [{story.id}] {has_image} {story.title[:50]}... (score: {story.quality_score})"
        )


def _list_scheduled_stories(engine: ContentEngine) -> None:
    """List scheduled stories."""
    print("\n--- Scheduled Stories ---")
    print(engine.scheduler.get_schedule_summary())


def _cleanup_old_stories(engine: ContentEngine) -> None:
    """Clean up old unused stories."""
    print("\n--- Cleanup Old Stories ---")
    print(f"Exclusion period: {Config.EXCLUSION_PERIOD_DAYS} days")

    cutoff = datetime.now() - timedelta(days=Config.EXCLUSION_PERIOD_DAYS)
    print(f"Will delete unpublished stories acquired before: {cutoff.date()}")

    confirm = input("\nProceed with cleanup? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        deleted = engine.db.delete_old_unused_stories(cutoff)
        orphaned = engine.image_generator.cleanup_orphaned_images()
        print(f"\nResult: Deleted {deleted} stories, {orphaned} orphaned images")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Cleanup failed")


def _run_full_cycle(engine: ContentEngine) -> None:
    """Run the full search cycle."""
    print("\n--- Run Full Search Cycle ---")
    print("This will:")
    print("  1. Search for new stories")
    print("  2. Generate images")
    print("  3. Verify content")
    print("  4. Schedule for publication")
    print("  5. Clean up old stories")

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        engine.run_search_cycle()
        print("\nSearch cycle completed successfully.")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Search cycle failed")


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
    parser.add_argument(
        "--menu",
        action="store_true",
        help="Run interactive debug menu",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run in continuous mode (scheduler)",
    )

    args = parser.parse_args()

    # Create engine
    engine = ContentEngine()

    # Handle menu first (doesn't require full validation)
    if args.menu:
        interactive_menu(engine)
        return 0

    # Validate configuration for other operations
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

    if args.continuous:
        engine.run_continuous()
        return 0

    # Default: run interactive menu
    interactive_menu(engine)
    return 0


if __name__ == "__main__":
    exit(main())
