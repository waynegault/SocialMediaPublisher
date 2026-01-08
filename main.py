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
from pathlib import Path

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

        # Try to connect first (poll for a short period in case it's starting)
        for _ in range(10):  # ~20 seconds
            client = self._get_lm_studio_client()
            if client:
                return client
            time.sleep(2)

        # If not running, try to start it
        logger.info("LM Studio not detected. Attempting to locate and start it...")
        lm_studio_path = self._find_lm_studio_executable()

        if lm_studio_path:
            try:
                logger.info(f"Starting LM Studio from {lm_studio_path}...")
                subprocess.Popen([lm_studio_path], start_new_session=True)

                # Whether we started a new instance or another instance is already running,
                # just poll until the local server becomes available.
                logger.info(
                    "Waiting for LM Studio to initialize or detect existing instance..."
                )
                for _ in range(30):  # up to ~60 seconds
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
                    logger.warning(
                        "LM Studio server detected but no models are loaded. You may need to load a model in the LM Studio UI."
                    )
                else:
                    logger.info(
                        f"LM Studio detected with {len(models)} model(s) loaded"
                    )
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
    # Build menu as a single string to avoid any output interleaving issues
    menu_text = """
============================================================
Social Media Publisher - Debug Menu
============================================================

  Component Testing:
    1. Test Story Search
    2. Test Image Generation
    3. Test Content Verification
    4. Test Scheduling
    5. Test LinkedIn Connection
    6. Test LinkedIn Publish (due stories)

  Database Operations:
    7. View Database Statistics
    8. List All Stories
    9. List Pending Stories
   10. List Scheduled Stories
   11. Cleanup Old Stories
   16. Backup Database
   17. Verify Database Integrity
   18. Restore Database from Backup
   19. Retry Rejected Stories (regenerate image + re-verify)

  Configuration:
   12. Show Configuration
   13. Show Full Status

  Pipeline:
   14. Run Full Search Cycle

  Testing:
   15. Run Unit Tests

   0. Exit
============================================================
"""
    import subprocess

    while True:
        # Use subprocess to call cls - this ensures a fresh terminal state
        subprocess.call("cls", shell=True)
        # Print menu using standard print
        print(menu_text)

        try:
            choice = input("Enter choice: ").strip()
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
            _test_api_keys(engine)
        elif choice == "13":
            engine.status()
        elif choice == "14":
            _run_full_cycle(engine)
        elif choice == "15":
            _run_unit_tests()
        elif choice == "16":
            _backup_database(engine)
        elif choice == "17":
            _verify_database(engine)
        elif choice == "18":
            _restore_database(engine)
        elif choice == "19":
            _retry_rejected_stories(engine)
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

        if new_count > 0:
            print("\nNewly saved stories:")
            # Get the most recent stories
            with engine.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, title, quality_score FROM stories ORDER BY id DESC LIMIT ?",
                    (new_count,),
                )
                for row in cursor.fetchall():
                    print(f"  - [{row[0]}] {row[1][:60]}... (Score: {row[2]})")
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

    # Show which image provider will be used
    if Config.HUGGINGFACE_API_TOKEN and Config.HF_PREFER_IF_CONFIGURED:
        provider = f"HuggingFace ({Config.HF_TTI_MODEL}) with Imagen fallback"
    else:
        provider = f"Google Imagen ({Config.MODEL_IMAGE})"
    print(f"Image Provider: {provider}")
    print("-" * 40)

    # First, check for missing image files and handle them
    missing_files_handled = _check_and_fix_missing_images(engine)
    if missing_files_handled > 0:
        print(f"\nHandled {missing_files_handled} stories with missing image files.")
        print("-" * 40)

    stories = engine.db.get_stories_needing_images(Config.MIN_QUALITY_SCORE)

    if not stories:
        # Check if there are stories that were skipped due to quality score
        all_pending = engine.db.get_stories_needing_images(0)
        if all_pending:
            print(
                f"Found {len(all_pending)} stories needing images, but NONE meet "
                f"the minimum quality score of {Config.MIN_QUALITY_SCORE}."
            )
            for story in all_pending[:5]:
                print(f"  - [{story.id}] {story.title}")
                print(f"      Score: {story.quality_score}")
            print(
                "\nYou can lower MIN_QUALITY_SCORE in your .env file to include these."
            )
        else:
            print(
                "No stories in the database need images (all have images or none found)."
            )
        return

    print(
        f"\nStories meeting quality threshold ({Config.MIN_QUALITY_SCORE}+): {len(stories)}"
    )
    for story in stories[:5]:  # Show first 5
        print(f"  - [{story.id}] {story.title}")
        print(f"      Score: {story.quality_score}")
    if len(stories) > 5:
        print(f"  ... and {len(stories) - 5} more")

    # Estimate time (roughly 10-30 seconds per image)
    est_time = len(stories) * 20  # 20 seconds average
    est_min = est_time // 60
    est_sec = est_time % 60
    time_str = f"{est_min}m {est_sec}s" if est_min > 0 else f"{est_sec}s"
    print(f"\nEstimated time: ~{time_str} for {len(stories)} image(s)")

    confirm = input("Generate images for these stories? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        print(f"\nGenerating {len(stories)} image(s)...")
        count = engine.image_generator.generate_images_for_stories()
        print(f"\n✓ Result: Generated {count}/{len(stories)} images successfully")
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


def _check_and_fix_missing_images(engine: ContentEngine) -> int:
    """
    Check all stories with image_path set and verify the files exist.

    - If file missing and story < 14 days old: clear image_path (will be regenerated)
    - If file missing and story >= 14 days old: clear image_path (too old to regenerate)

    Returns the number of stories that were updated.
    """
    stories_with_images = engine.db.get_stories_with_images()
    if not stories_with_images:
        return 0

    cutoff_date = datetime.now() - timedelta(days=14)
    updated_count = 0
    regenerate_count = 0
    expired_count = 0

    print(f"Checking {len(stories_with_images)} stories with image references...")

    for story in stories_with_images:
        if not story.image_path:
            continue

        image_file = Path(story.image_path)

        if image_file.exists():
            continue  # File exists, nothing to do

        # File is missing - determine action based on story age
        story_date = story.acquire_date or datetime.min

        if story_date >= cutoff_date:
            # Story is less than 14 days old - clear path so it can be regenerated
            print(f"  [REGENERATE] Story {story.id}: image missing, will regenerate")
            print(f"      {story.title}")
            story.image_path = None
            engine.db.update_story(story)
            regenerate_count += 1
            updated_count += 1
        else:
            # Story is 14+ days old - just clear the reference
            days_old = (datetime.now() - story_date).days
            print(
                f"  [EXPIRED] Story {story.id}: image missing, story is {days_old} days old"
            )
            print(f"      {story.title}")
            story.image_path = None
            engine.db.update_story(story)
            expired_count += 1
            updated_count += 1

    if updated_count > 0:
        print(
            f"\nSummary: {regenerate_count} queued for regeneration, {expired_count} expired (reference cleared)"
        )

    return updated_count


def _test_verification(engine: ContentEngine) -> None:
    """Test the content verification component."""
    print("\n--- Testing Content Verification ---")

    stories = engine.db.get_stories_needing_verification()

    if not stories:
        # Check why none are pending - provide helpful diagnostics
        stats = engine.db.get_statistics()

        if stats.get("total_stories", 0) == 0:
            print("No stories in database. Search for stories first (Choice 1).")
        elif stats.get("needing_images", 0) > 0:
            print(
                f"Found {stats.get('needing_images', 0)} stories without images. "
                "Generate images first (Choice 2)."
            )
        elif (
            stats.get("available_count", 0) > 0
            and stats.get("ready_for_verification", 0) == 0
        ):
            print(
                f"All stories have been verified. {stats.get('available_count', 0)} are approved and ready."
            )
        else:
            print("No stories are currently pending verification.")
            print(f"  Total stories: {stats.get('total_stories', 0)}")
            print(f"  Needing images: {stats.get('needing_images', 0)}")
            print(f"  Ready for verification: {stats.get('ready_for_verification', 0)}")
            print(f"  Available (approved): {stats.get('available_count', 0)}")
        return

    print(f"Stories pending verification: {len(stories)}")
    for story in stories[:5]:  # Show first 5
        print(f"  - [{story.id}] {story.title}")

    confirm = input("\nVerify these stories? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        approved, rejected = engine.verifier.verify_pending_content()
        print(f"\nResult: {approved} approved, {rejected} rejected")
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user.")
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
        print(f"  - [{story.id}] {story.title}")

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
            SELECT id, title, quality_score, verification_status, verification_reason,
                   publish_status, acquire_date, image_path
            FROM stories ORDER BY acquire_date DESC LIMIT 20
        """)
        rows = cursor.fetchall()

    if not rows:
        print("No stories in database.")
        return

    for row in rows:
        has_image = "Yes" if row["image_path"] else "No"
        verify_status = row["verification_status"] or "pending"
        verify_reason = row["verification_reason"] or ""
        print(f"\n[{row['id']}] {row['title']}")
        print(
            f"    Score: {row['quality_score']} | Verify: {verify_status} | "
            f"Publish: {row['publish_status']} | Image: {has_image}"
        )
        if verify_reason:
            print(f"    Reason: {verify_reason}")

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
        has_image = "Yes" if story.image_path else "No"
        print(f"\n[{story.id}] {story.title}")
        print(f"    Score: {story.quality_score} | Image: {has_image}")


def _list_scheduled_stories(engine: ContentEngine) -> None:
    """List scheduled stories."""
    print("\n--- Scheduled Stories ---")
    print(engine.scheduler.get_schedule_summary())


def _retry_rejected_stories(engine: ContentEngine) -> None:
    """Retry rejected stories by regenerating images and re-verifying."""
    print("\n--- Retry Rejected Stories ---")

    rejected = engine.db.get_rejected_stories()
    if not rejected:
        print("No rejected stories found.")
        return

    print(f"Found {len(rejected)} rejected stories:\n")
    for story in rejected:
        print(f"[{story.id}] {story.title}")
        print(f"    Reason: {story.verification_reason or 'Unknown'}")
        print()

    # Ask user which stories to retry
    story_input = (
        input(
            "Enter story IDs to retry (comma-separated), 'all' for all, or 'q' to cancel: "
        )
        .strip()
        .lower()
    )

    if story_input == "q" or not story_input:
        print("Cancelled.")
        return

    # Determine which stories to retry
    if story_input == "all":
        stories_to_retry = rejected
    else:
        try:
            ids_to_retry = [int(x.strip()) for x in story_input.split(",")]
            stories_to_retry = [s for s in rejected if s.id in ids_to_retry]
            if not stories_to_retry:
                print("No matching stories found.")
                return
        except ValueError:
            print("Invalid input. Please enter comma-separated IDs or 'all'.")
            return

    print(f"\nRetrying {len(stories_to_retry)} story/stories...")

    success_count = 0
    for story in stories_to_retry:
        print(f"\n--- Processing story {story.id}: {story.title} ---")

        # Step 1: Delete old image if exists
        if story.image_path:
            old_image = Path(story.image_path)
            if old_image.exists():
                try:
                    old_image.unlink()
                    print(f"  Deleted old image: {story.image_path}")
                except Exception as e:
                    print(f"  Warning: Could not delete old image: {e}")

        # Step 2: Clear image path and reset status
        story.image_path = None
        story.verification_status = "pending"
        story.verification_reason = None
        engine.db.update_story(story)

        # Step 3: Generate new image
        print("  Generating new image...")
        try:
            image_path = engine.image_generator._generate_image_for_story(story)
            if image_path:
                story.image_path = image_path
                engine.db.update_story(story)
                print(f"  ✓ New image generated: {image_path}")
            else:
                print("  ✗ Failed to generate new image")
                continue
        except Exception as e:
            print(f"  ✗ Image generation error: {e}")
            continue

        # Step 4: Re-verify
        print("  Re-verifying...")
        try:
            is_approved, reason = engine.verifier._verify_story(story)
            story.verification_status = "approved" if is_approved else "rejected"
            story.verification_reason = reason
            engine.db.update_story(story)

            if is_approved:
                print(f"  ✓ APPROVED: {reason}")
                success_count += 1
            else:
                print(f"  ✗ REJECTED again: {reason}")
        except Exception as e:
            print(f"  ✗ Verification error: {e}")

    print(
        f"\n=== Summary: {success_count}/{len(stories_to_retry)} stories now approved ==="
    )


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


def _backup_database(engine: ContentEngine) -> None:
    """Create a database backup."""
    print("\n--- Database Backup ---")
    print(f"Database: {engine.db.db_name}")

    backup_exists = engine.db.backup_exists()
    if backup_exists:
        print("Warning: An existing backup will be overwritten.")

    confirm = input("\nCreate backup? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        backup_path = engine.db.create_backup()
        print(f"\n✓ Backup created: {backup_path}")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Backup failed")


def _verify_database(engine: ContentEngine) -> None:
    """Verify database integrity."""
    print("\n--- Database Integrity Check ---")
    print(f"Database: {engine.db.db_name}")

    print("\nRunning integrity check...")
    is_valid, message = engine.db.verify_integrity()

    if is_valid:
        print(f"\n✓ {message}")
    else:
        print(f"\n✗ {message}")
        print("\nConsider restoring from backup if available.")
        if engine.db.backup_exists():
            print("  Backup file exists: Use option 18 to restore.")


def _restore_database(engine: ContentEngine) -> None:
    """Restore database from backup."""
    print("\n--- Database Restore ---")
    print(f"Database: {engine.db.db_name}")

    if not engine.db.backup_exists():
        print("\n✗ No backup file found.")
        print("  Create a backup first using option 16.")
        return

    print("\nWarning: This will replace the current database with the backup.")
    print("         All changes since the backup was created will be lost.")

    confirm = input("\nRestore from backup? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        if engine.db.restore_from_backup():
            print("\n✓ Database restored from backup.")
            print("  Reinitializing database connection...")
            engine.db._init_db()  # Reinitialize to pick up restored data
        else:
            print("\n✗ Restore failed. See logs for details.")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Restore failed")


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


def _run_unit_tests() -> None:
    """Run the unit test suite."""
    print("\n--- Running Unit Tests ---")
    print("Executing all unit tests...\n")

    try:
        # Import and run tests from unit_tests module
        from unit_tests import run_tests

        success = run_tests()

        if success:
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed. Review output above.")
    except ImportError as e:
        print(f"\nError importing test module: {e}")
        print("Make sure unit_tests.py and test_framework.py are present.")
    except Exception as e:
        print(f"\nError running tests: {e}")
        logger.exception("Unit test execution failed")


def _test_api_keys(engine: ContentEngine) -> None:
    """Test all configured API keys to verify they are valid and working."""
    print("\n--- API Key Validation ---")
    print("Testing configured API connections...\n")

    results: list[tuple[str, bool, str]] = []

    # 1. Test Gemini API
    print("  Testing Gemini API...", end=" ", flush=True)
    if Config.GEMINI_API_KEY:
        try:
            response = engine.genai_client.models.generate_content(
                model=Config.MODEL_TEXT,
                contents="Say 'API key is valid' in exactly 4 words.",
                config={"max_output_tokens": 20},
            )
            if response.text:
                results.append(("Gemini API", True, f"Model: {Config.MODEL_TEXT}"))
                print("✓ OK")
            else:
                results.append(("Gemini API", False, "Empty response"))
                print("✗ FAILED (empty response)")
        except Exception as e:
            error_msg = str(e)[:50]
            results.append(("Gemini API", False, error_msg))
            print(f"✗ FAILED ({error_msg})")
    else:
        results.append(("Gemini API", False, "NOT CONFIGURED"))
        print("⚠ NOT CONFIGURED")

    # 2. Test Hugging Face API (if configured)
    print("  Testing Hugging Face API...", end=" ", flush=True)
    if Config.HUGGINGFACE_API_TOKEN:
        try:
            # Test with a simple whoami endpoint
            headers = {"Authorization": f"Bearer {Config.HUGGINGFACE_API_TOKEN}"}
            response = requests.get(
                "https://huggingface.co/api/whoami",
                headers=headers,
                timeout=10,
            )
            if response.status_code == 200:
                user_info = response.json()
                username = user_info.get("name", "Unknown")
                results.append(("Hugging Face API", True, f"User: {username}"))
                print(f"✓ OK (User: {username})")
            elif response.status_code == 401:
                results.append(("Hugging Face API", False, "Invalid token"))
                print("✗ FAILED (Invalid token)")
            else:
                results.append(
                    ("Hugging Face API", False, f"HTTP {response.status_code}")
                )
                print(f"✗ FAILED (HTTP {response.status_code})")
        except Exception as e:
            error_msg = str(e)[:50]
            results.append(("Hugging Face API", False, error_msg))
            print(f"✗ FAILED ({error_msg})")
    else:
        results.append(("Hugging Face API", False, "NOT CONFIGURED (optional)"))
        print("⚠ NOT CONFIGURED (optional)")

    # 3. Test LinkedIn API
    print("  Testing LinkedIn API...", end=" ", flush=True)
    if Config.LINKEDIN_ACCESS_TOKEN and Config.LINKEDIN_AUTHOR_URN:
        try:
            headers = {
                "Authorization": f"Bearer {Config.LINKEDIN_ACCESS_TOKEN}",
                "X-Restli-Protocol-Version": "2.0.0",
            }
            response = requests.get(
                "https://api.linkedin.com/v2/userinfo",
                headers=headers,
                timeout=10,
            )
            if response.status_code == 200:
                user_info = response.json()
                name = user_info.get("name", "Unknown")
                results.append(("LinkedIn API", True, f"User: {name}"))
                print(f"✓ OK (User: {name})")
            elif response.status_code == 401:
                results.append(("LinkedIn API", False, "Token expired or invalid"))
                print("✗ FAILED (Token expired or invalid)")
            else:
                results.append(("LinkedIn API", False, f"HTTP {response.status_code}"))
                print(f"✗ FAILED (HTTP {response.status_code})")
        except Exception as e:
            error_msg = str(e)[:50]
            results.append(("LinkedIn API", False, error_msg))
            print(f"✗ FAILED ({error_msg})")
    elif Config.LINKEDIN_ACCESS_TOKEN and not Config.LINKEDIN_AUTHOR_URN:
        results.append(("LinkedIn API", False, "LINKEDIN_AUTHOR_URN not set"))
        print("⚠ MISSING LINKEDIN_AUTHOR_URN")
    else:
        results.append(("LinkedIn API", False, "NOT CONFIGURED"))
        print("⚠ NOT CONFIGURED")

    # 4. Test Local LLM (LM Studio) if preferred
    print("  Testing Local LLM (LM Studio)...", end=" ", flush=True)
    if Config.PREFER_LOCAL_LLM:
        try:
            response = requests.get(
                f"{Config.LM_STUDIO_BASE_URL}/models",
                timeout=5,
            )
            if response.status_code == 200:
                models = response.json().get("data", [])
                if models:
                    model_id = models[0].get("id", "Unknown")
                    results.append(("Local LLM", True, f"Model: {model_id}"))
                    print(f"✓ OK ({len(models)} model(s) loaded)")
                else:
                    results.append(("Local LLM", False, "No models loaded"))
                    print("⚠ Server running but no models loaded")
            else:
                results.append(("Local LLM", False, f"HTTP {response.status_code}"))
                print(f"✗ FAILED (HTTP {response.status_code})")
        except requests.exceptions.ConnectionError:
            results.append(("Local LLM", False, "Server not running"))
            print("⚠ Server not running (start LM Studio)")
        except Exception as e:
            error_msg = str(e)[:50]
            results.append(("Local LLM", False, error_msg))
            print(f"✗ FAILED ({error_msg})")
    else:
        results.append(("Local LLM", False, "DISABLED (PREFER_LOCAL_LLM=False)"))
        print("⚠ DISABLED (PREFER_LOCAL_LLM=False)")

    # 5. Test Imagen API (Gemini image generation)
    print("  Testing Imagen API...", end=" ", flush=True)
    if Config.GEMINI_API_KEY:
        try:
            # Just verify the model exists by listing models
            # We don't actually generate an image (costs money/quota)
            models_list = list(engine.genai_client.models.list())
            imagen_available = any(
                Config.MODEL_IMAGE in str(m.name) for m in models_list
            )
            if imagen_available:
                results.append(("Imagen API", True, f"Model: {Config.MODEL_IMAGE}"))
                print(f"✓ OK (Model: {Config.MODEL_IMAGE})")
            else:
                results.append(
                    ("Imagen API", False, f"Model {Config.MODEL_IMAGE} not found")
                )
                print(f"⚠ Model {Config.MODEL_IMAGE} not in available models")
        except Exception as e:
            error_msg = str(e)[:50]
            results.append(("Imagen API", False, error_msg))
            print(f"✗ FAILED ({error_msg})")
    else:
        results.append(("Imagen API", False, "Requires GEMINI_API_KEY"))
        print("⚠ Requires GEMINI_API_KEY")

    # Print summary
    print("\n--- Summary ---")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    required_services = ["Gemini API", "LinkedIn API"]
    required_ok = all(ok for name, ok, _ in results if name in required_services)

    for name, ok, detail in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: {detail}")

    print(f"\n  {passed}/{total} services operational")
    if required_ok:
        print("  ✓ All required services are working")
    else:
        print("  ⚠ Some required services need attention")


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
