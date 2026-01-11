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

from api_client import api_client
from config import Config
from database import Database
from searcher import StorySearcher
from image_generator import ImageGenerator
from verifier import ContentVerifier
from company_mention_enricher import CompanyMentionEnricher
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

    def __init__(self) -> None:
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
        self.enricher = CompanyMentionEnricher(
            self.db, self.genai_client, self.local_client
        )
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
            response = api_client.http_get(
                url=f"{Config.LM_STUDIO_BASE_URL}/models",
                timeout=2,
                endpoint="lm_studio_check",
            )
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
        2. Enrich with company mentions from sources
        3. Generate images for qualifying stories
        4. Assign promotion messages
        5. Verify content quality
        6. Clean up old unused stories

        Note: Scheduling is handled separately via Action 20.
        """
        logger.info("=" * 60)
        logger.info("Starting Search Cycle")
        logger.info("=" * 60)

        try:
            # Step 1: Search for new stories
            logger.info("Step 1: Searching for new stories...")
            new_stories = self.searcher.search_and_process()
            logger.info(f"Found {new_stories} new stories")

            # Step 2: Enrich with company mentions
            logger.info("Step 2: Enriching with company mentions...")
            enriched, skipped = self.enricher.enrich_pending_stories()
            logger.info(f"Enrichment: {enriched} enriched, {skipped} skipped")

            # Step 3: Generate images
            logger.info("Step 3: Generating images...")
            images_generated = self.image_generator.generate_images_for_stories()
            logger.info(f"Generated {images_generated} images")

            # Step 4: Assign promotion messages
            logger.info("Step 4: Assigning promotion messages...")
            promo_assigned = self._assign_promotions_batch()
            logger.info(f"Assigned {promo_assigned} promotion messages")

            # Step 5: Verify content
            logger.info("Step 5: Verifying content...")
            approved, rejected = self.verifier.verify_pending_content()
            logger.info(f"Verification: {approved} approved, {rejected} rejected")

            # Step 6: Cleanup old stories
            logger.info("Step 6: Cleaning up old stories...")
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

    def _assign_promotions_batch(
        self, require_image: bool = True, verbose: bool = False
    ) -> int:
        """Assign AI-generated promotion messages to stories that don't have one.

        Args:
            require_image: If True, only process stories with images (for pipeline).
                          If False, process all stories without promotions (for manual regeneration).
            verbose: If True, print progress to console.
        """
        import random

        style_examples = [
            "MEng Process & Chemical Engineering | Know someone hiring in this space? I'd love an intro! Follow me for more insights.",
            "This resonates with my chemical engineering background. Know of opportunities? DM me or tag someone who should connect!",
            "Follow for more process engineering insights! If you know of roles in this area, I'd appreciate a connection.",
            "Strathclyde-trained engineer exploring opportunities in this sector. Hit follow & let me know if you hear of openings!",
            "Passionate about sustainable processes. Know someone working in this space? Tag them below or send me a message!",
        ]

        # Get stories that need a promotion message
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            if require_image:
                cursor.execute("""
                    SELECT id, title, summary
                    FROM stories
                    WHERE (promotion IS NULL OR promotion = '')
                      AND image_path IS NOT NULL
                    ORDER BY id DESC
                """)
            else:
                cursor.execute("""
                    SELECT id, title, summary
                    FROM stories
                    WHERE promotion IS NULL OR promotion = ''
                    ORDER BY id DESC
                """)
            rows = cursor.fetchall()

        if not rows:
            return 0

        examples_text = "\n".join(f"- {ex}" for ex in style_examples)
        assigned = 0

        for row in rows:
            story_id = row["id"]
            title = row["title"]
            summary = row["summary"] or ""

            prompt = f"""Generate a professional LinkedIn message with a clear call-to-action that connects to this story's topic.

STORY:
Title: {title}
Summary: {summary[:500]}

REQUIREMENTS:
1. Write in FIRST PERSON (use "I", "my", "I'm")
2. Reference the specific technology, company, industry, or theme from the story
3. Briefly mention MEng Process & Chemical Engineering background
4. Include ONE clear call-to-action from this list:
   - Ask followers to tag someone who should connect with you
   - Invite people to DM you about opportunities
   - Ask if anyone knows of openings in this area
   - Encourage people to follow for more insights
   - Ask for introductions to contacts in this space
5. Maintain professional confidence - show expertise and genuine interest
6. Keep it to 1-2 sentences, max 250 characters
7. Do NOT include emojis

STYLE EXAMPLES (match this tone and include CTA):
{examples_text}

OUTPUT: Write ONLY the promotion message, nothing else."""

            try:
                response = api_client.gemini_generate(
                    client=self.genai_client,
                    model="gemini-2.0-flash",
                    contents=prompt,
                    endpoint="promotion_message",
                )
                if response.text:
                    promotion = response.text.strip().strip('"').strip("'")
                else:
                    promotion = random.choice(style_examples)

                if len(promotion) > 300:
                    promotion = promotion[:297] + "..."

            except Exception as e:
                logger.warning(f"AI generation failed for story {story_id}: {e}")
                promotion = random.choice(style_examples)

            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE stories SET promotion = ? WHERE id = ?",
                    (promotion, story_id),
                )

            assigned += 1

            if verbose:
                print(f"  [{story_id}] {title[:50]}...")
                print(f"      → {promotion[:80]}...")

        return assigned

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
                # OpenID userinfo returns 'name' or 'given_name'/'family_name'
                name = profile.get("name") or profile.get("given_name", "Unknown")
                print(f"  Connected as: {name}")
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

  Pipeline:
    1. Run Full Pipeline (search → enrich → images → promo → verify → cleanup)

  Configuration:
    2. Show Configuration
    3. View All Prompts (full text)
    4. Show Full Status
    5. Install Dependencies (requirements.txt)

  Database Operations:
    6. View Database Statistics
    7. List All Stories
    8. List Pending Stories
    9. List Scheduled Stories
   10. List Human Approved Stories (not published)
   11. Cleanup Old Stories
   12. Backup Database
   13. Restore Database from Backup
   14. Verify Database Integrity
   15. Retry Rejected Stories (regenerate image + re-verify)

  Component Testing:
   16. Search, Enrich & Assign Promotions
   17. Test Image Generation
   18. Test Content Verification
   19. Test Scheduling
   20. Human Validation (Web GUI)
   21. Test LinkedIn Connection
   22. Send LinkedIn Connection Requests
   23. Test LinkedIn Publish (due stories)
   24. Publish One Scheduled Story Now
   25. Run Unit Tests

  Analytics:
   26. View LinkedIn Analytics
   27. Refresh All Analytics

  Danger Zone:
   28. Reset (delete database and images)

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
        # Pipeline
        elif choice == "1":
            _run_full_cycle(engine)
        # Configuration
        elif choice == "2":
            Config.print_config()
            _test_api_keys(engine)
        elif choice == "3":
            _show_all_prompts()
        elif choice == "4":
            engine.status()
        elif choice == "5":
            _install_dependencies()
        # Database Operations
        elif choice == "6":
            _show_database_stats(engine)
        elif choice == "7":
            _list_all_stories(engine)
        elif choice == "8":
            _list_pending_stories(engine)
        elif choice == "9":
            _list_scheduled_stories(engine)
        elif choice == "10":
            _list_human_approved_stories(engine)
        elif choice == "11":
            _cleanup_old_stories(engine)
        elif choice == "12":
            _backup_database(engine)
        elif choice == "13":
            _restore_database(engine)
        elif choice == "14":
            _verify_database(engine)
        elif choice == "15":
            _retry_rejected_stories(engine)
        # Component Testing
        elif choice == "16":
            _test_search(engine)
        elif choice == "17":
            _test_image_generation(engine)
        elif choice == "18":
            _test_verification(engine)
        elif choice == "19":
            _test_scheduling(engine)
        elif choice == "20":
            _human_validation(engine)
        elif choice == "21":
            _test_linkedin_connection(engine)
        elif choice == "22":
            _send_linkedin_connections(engine)
        elif choice == "23":
            _test_linkedin_publish(engine)
        elif choice == "24":
            _test_publish_one_story(engine)
        elif choice == "25":
            _run_unit_tests()
        # Analytics
        elif choice == "26":
            _view_linkedin_analytics(engine)
        elif choice == "27":
            _refresh_linkedin_analytics(engine)
        # Danger Zone
        elif choice == "28":
            _reset_all(engine)
        else:
            print("Invalid choice. Please try again.")

        try:
            input("\nPress Enter to continue...")
        except EOFError:
            pass  # Gracefully handle piped input


def _test_search(engine: ContentEngine) -> None:
    """Search for stories, enrich with LinkedIn profiles/URNs, and assign promotions."""
    print("\n--- Story Search, Enrichment & Promotions ---")
    print(f"Search prompt: {Config.SEARCH_PROMPT[:80]}...")
    print(f"Lookback days: {Config.SEARCH_LOOKBACK_DAYS}")
    print(f"Use last checked date: {Config.USE_LAST_CHECKED_DATE}")

    try:
        start_date = engine.searcher.get_search_start_date()
        print(f"Searching for stories since: {start_date}")

        # Step 1: Search for stories
        print("\n[Step 1/4] Searching for stories...")
        new_count = engine.searcher.search_and_process()
        print(f"  → Found and saved {new_count} new stories")

        if new_count > 0:
            print("\nNewly saved stories (by quality score):")
            # Get the most recent stories, sorted by quality score
            with engine.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, title, quality_score FROM stories ORDER BY id DESC LIMIT ?",
                    (new_count,),
                )
                rows = cursor.fetchall()
                # Sort by quality score descending (highest first)
                for row in sorted(rows, key=lambda r: r[2], reverse=True):
                    print(f"  - [{row[0]}] {row[1][:60]}... (Score: {row[2]})")

        # Step 2: Find LinkedIn profiles for all people needing them
        print("\n[Step 2/4] Finding LinkedIn profiles...")
        _run_profile_lookup_silent(engine)

        # Step 3: Extract URNs for @mentions
        print("\n[Step 3/4] Extracting LinkedIn URNs for @mentions...")
        _run_urn_extraction_silent(engine)

        # Step 4: Assign promotion messages
        print("\n[Step 4/4] Assigning promotion messages...")
        _run_promotion_assignment_silent(engine)

        # Show summary
        print("\n" + "=" * 60)
        print("Search, Enrichment & Promotions Complete")
        print("=" * 60)
        _show_enrichment_summary(engine)

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


def _get_stories_needing_profile_lookup(engine: ContentEngine) -> list:
    """Get all stories that have people needing LinkedIn profile lookup."""
    import json as json_module

    stories_needing_profiles = []
    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        # Check both story_people and org_leaders (the new primary fields)
        cursor.execute("""
            SELECT id, story_people, org_leaders FROM stories
            WHERE (story_people IS NOT NULL AND story_people != '[]')
               OR (org_leaders IS NOT NULL AND org_leaders != '[]')
        """)
        for row in cursor.fetchall():
            story_people = (
                json_module.loads(row["story_people"]) if row["story_people"] else []
            )
            org_leaders = (
                json_module.loads(row["org_leaders"]) if row["org_leaders"] else []
            )
            all_people = story_people + org_leaders
            # Check if any people need profiles
            needs_lookup = any(
                not (p.get("linkedin_profile") or "").strip() for p in all_people
            )
            if needs_lookup:
                story = engine.db.get_story(row["id"])
                if story:
                    stories_needing_profiles.append(story)
    return stories_needing_profiles


def _run_profile_lookup_silent(engine: ContentEngine) -> int:
    """Run LinkedIn profile lookup without verbose output. Returns count of stories processed."""
    # Use a single query to get stories with people needing profiles
    stories_needing_profiles = _get_stories_needing_profile_lookup(engine)

    if not stories_needing_profiles:
        print("  → All people already have LinkedIn profiles")
        return 0

    print(f"  → Looking up profiles for {len(stories_needing_profiles)} stories...")
    _lookup_linkedin_profiles_for_people(engine, stories_needing_profiles)
    return len(stories_needing_profiles)


def _run_urn_extraction_silent(engine: ContentEngine) -> tuple[int, int]:
    """Run URN extraction without verbose output. Returns (enriched, pending) counts."""
    import json as json_module

    # Check how many people need URN extraction across story_people and org_leaders
    pending = 0
    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT story_people, org_leaders FROM stories
            WHERE (story_people LIKE '%linkedin_profile%')
               OR (org_leaders LIKE '%linkedin_profile%')
        """)
        for row in cursor.fetchall():
            story_people = (
                json_module.loads(row["story_people"]) if row["story_people"] else []
            )
            org_leaders = (
                json_module.loads(row["org_leaders"]) if row["org_leaders"] else []
            )
            for p in story_people + org_leaders:
                if p.get("linkedin_profile") and not p.get("linkedin_urn"):
                    pending += 1

    if pending == 0:
        print("  → All people already have URNs")
        return 0, 0

    print(f"  → Extracting URNs for {pending} people...")

    try:
        enriched, skipped = engine.enricher.populate_linkedin_mentions()
        print(f"  → {enriched} stories updated, {skipped} skipped")
        return enriched, pending
    except Exception as e:
        print(f"  → Error: {e}")
        logger.exception("URN extraction failed")
        return 0, pending


def _run_promotion_assignment_silent(engine: ContentEngine) -> int:
    """Assign promotion messages to stories without verbose output. Returns count assigned."""
    # Count stories needing promotion
    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM stories
            WHERE promotion IS NULL OR promotion = ''
        """)
        count = cursor.fetchone()["count"]

    if count == 0:
        print("  → All stories already have promotion messages")
        return 0

    print(f"  → Assigning promotions to {count} stories...")

    try:
        assigned = engine._assign_promotions_batch(require_image=False, verbose=False)
        print(f"  → {assigned} stories updated with promotions")
        return assigned
    except Exception as e:
        print(f"  → Error: {e}")
        logger.exception("Promotion assignment failed")
        return 0


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


def _assign_promotion_message(engine: ContentEngine) -> None:
    """Assign an AI-generated promotion message aligned to each story's content."""
    print("\n--- Assign Promotion Message ---")

    # Check if there are stories needing promotion
    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM stories
            WHERE promotion IS NULL OR promotion = ''
        """)
        count = cursor.fetchone()["count"]

    if count == 0:
        print("\nNo stories need a promotion message. All stories already have one.")

        # Show a sample of existing promotions
        with engine.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, promotion
                FROM stories
                WHERE promotion IS NOT NULL AND promotion != ''
                ORDER BY id DESC
                LIMIT 3
            """)
            sample = cursor.fetchall()
            if sample:
                print("\nSample of existing promotions:")
                for row in sample:
                    print(f"  [{row['id']}] {row['title'][:50]}...")
                    print(f"      → {row['promotion'][:80]}...")
        return

    print(f"\nGenerating personalized promotion messages for {count} stories...")

    # Reuse the method from ContentEngine (require_image=False to process all)
    assigned = engine._assign_promotions_batch(require_image=False, verbose=True)

    print(f"\n✓ Generated personalized promotion messages for {assigned} stories.")


def _check_and_fix_missing_images(engine: ContentEngine) -> int:
    """
    Check all stories with image_path set and verify the files exist.

    - If file missing and story is recent: clear image_path (will be regenerated)
    - If file missing and story is too old: clear image_path (too old to regenerate)

    Returns the number of stories that were updated.
    """
    stories_with_images = engine.db.get_stories_with_images()
    if not stories_with_images:
        return 0

    cutoff_date = datetime.now() - timedelta(days=Config.IMAGE_REGEN_CUTOFF_DAYS)
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
            # Reset verification so the new image will be verified
            story.verification_status = "pending"
            story.verification_reason = None
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
            # Reset verification for expired stories too
            story.verification_status = "pending"
            story.verification_reason = None
            engine.db.update_story(story)
            expired_count += 1
            updated_count += 1

    if updated_count > 0:
        print(
            f"\nSummary: {regenerate_count} queued for regeneration, {expired_count} expired (reference cleared)"
        )

    return updated_count


def _check_and_offer_image_retry(engine: ContentEngine) -> None:
    """Check for image-related rejections and offer to retry them."""
    rejected_stories = engine.db.get_rejected_stories()

    # Filter for image-related rejections
    image_keywords = ["image", "picture", "photo", "visual", "graphic", "illustration"]
    image_rejected = [
        s
        for s in rejected_stories
        if s.verification_reason
        and any(keyword in s.verification_reason.lower() for keyword in image_keywords)
    ]

    if not image_rejected:
        return

    print(f"\n--- Found {len(image_rejected)} stories rejected due to image issues ---")
    for story in image_rejected:
        print(f"  [{story.id}] {story.title}")
        print(f"       Reason: {story.verification_reason}")

    retry = input("\nRetry these with new images? (y/n): ").strip().lower()
    if retry != "y":
        return

    # Run the retry logic for these specific stories
    print(f"\nRetrying {len(image_rejected)} stories...")

    success_count = 0
    for story in image_rejected:
        print(f"\n--- Processing story {story.id}: {story.title} ---")

        # Delete old image if exists
        if story.image_path:
            old_image = Path(story.image_path)
            if old_image.exists():
                try:
                    old_image.unlink()
                    print(f"  Deleted old image: {story.image_path}")
                except Exception as e:
                    print(f"  Warning: Could not delete old image: {e}")

        # Clear image path and reset status
        story.image_path = None
        story.verification_status = "pending"
        story.verification_reason = None
        engine.db.update_story(story)

        # Generate new image
        print("  Generating new image...")
        try:
            result = engine.image_generator._generate_image_for_story(story)
            if result:
                image_path, image_alt_text = result
                story.image_path = image_path
                story.image_alt_text = image_alt_text
                engine.db.update_story(story)
                print(f"  ✓ New image generated: {image_path}")
            else:
                print("  ✗ Failed to generate new image")
                continue
        except Exception as e:
            print(f"  ✗ Image generation error: {e}")
            continue

        # Re-verify
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
        f"\n=== Retry Summary: {success_count}/{len(image_rejected)} stories now approved ==="
    )


def _test_verification(engine: ContentEngine) -> None:
    """Test the content verification component."""
    print("\n--- Testing Content Verification ---")

    stories = engine.db.get_stories_needing_verification()

    if not stories:
        # Check why none are pending - provide helpful diagnostics
        stats = engine.db.get_statistics()

        if stats.get("total_stories", 0) == 0:
            print("No stories in database. Search for stories first (Choice 1).")
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
            print(f"  Available (approved): {stats.get('available_count', 0)}")
        return

    print(f"Stories pending verification: {len(stories)}")
    with_images = sum(1 for s in stories if s.image_path)
    low_quality = sum(
        1
        for s in stories
        if not s.image_path and s.quality_score < Config.MIN_QUALITY_SCORE
    )
    print(f"  - {with_images} with images (will be AI-verified)")
    print(f"  - {low_quality} low quality without images (will be auto-rejected)")
    print()
    for story in stories[:5]:  # Show first 5
        img_status = "✓" if story.image_path else f"✗ (score={story.quality_score})"
        print(f"  [{story.id}] {story.title}")
        print(f"       Image: {img_status}")

    # Create a LinkedIn lookup callback for auto-retry on insufficient coverage
    def linkedin_lookup_callback(stories_to_process: list) -> None:
        """Callback to run LinkedIn profile lookup for stories with insufficient coverage."""
        _lookup_linkedin_profiles_for_people(engine, stories_to_process)

    try:
        approved, rejected = engine.verifier.verify_pending_content(
            linkedin_lookup_callback=linkedin_lookup_callback
        )
        print(f"\nResult: {approved} approved, {rejected} rejected")

        # Check for image-related rejections and offer to retry
        if rejected > 0:
            _check_and_offer_image_retry(engine)

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


def _get_stories_needing_linkedin_profiles(engine: ContentEngine) -> list:
    """Get stories that have story_people or org_leaders with missing LinkedIn profiles."""
    import json as json_module

    stories_needing_profiles = []
    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, story_people, org_leaders
            FROM stories
            WHERE (story_people IS NOT NULL AND story_people != '[]')
               OR (org_leaders IS NOT NULL AND org_leaders != '[]')
        """)
        for row in cursor.fetchall():
            try:
                story_people = (
                    json_module.loads(row["story_people"])
                    if row["story_people"]
                    else []
                )
                org_leaders = (
                    json_module.loads(row["org_leaders"]) if row["org_leaders"] else []
                )
                all_people = story_people + org_leaders
                if not all_people:
                    continue

                # Check if any person is missing a LinkedIn profile
                people_needing_lookup = [
                    p
                    for p in all_people
                    if not (p.get("linkedin_profile") or "").strip()
                ]
                if people_needing_lookup:
                    # Fetch the full story object
                    story = engine.db.get_story(row["id"])
                    if story:
                        stories_needing_profiles.append(story)
            except json_module.JSONDecodeError:
                continue

    return stories_needing_profiles


def _lookup_linkedin_profiles_for_people(
    engine: ContentEngine,
    stories: list,
) -> None:
    """Look up LinkedIn company pages for organizations in story_people and org_leaders."""
    from linkedin_profile_lookup import LinkedInCompanyLookup

    # Use Gemini to search for company LinkedIn pages
    lookup = LinkedInCompanyLookup(genai_client=engine.genai_client)

    if not lookup.client:
        print("\n⚠ Gemini API not configured.")
        print("LinkedIn company lookup requires a configured GEMINI_API_KEY.")
        return

    total_companies_found = 0
    total_companies_not_found = 0
    stories_updated = 0
    all_company_data: dict[str, tuple[str, str | None]] = {}
    all_companies_not_found: set[str] = set()

    # Track people results for summary
    people_with_profiles: list[dict] = []  # People who got LinkedIn profiles
    people_without_profiles: list[dict] = []  # People who didn't get profiles

    try:
        for story in stories:
            # Combine story_people and org_leaders for lookup
            all_people = (story.story_people or []) + (story.org_leaders or [])

            # If story has no people, try to extract them using AI
            if not all_people:
                print(f"\n[{story.id}] {story.title[:60]}...")
                print("  No people - extracting with AI...")
                result = engine.enricher._extract_orgs_and_people(story)
                if result:
                    people = result.get("story_people", [])
                    orgs = result.get("organizations", [])
                    if people:
                        # Store in story_people (the new primary field)
                        story.story_people = [
                            {
                                "name": p.get("name", ""),
                                "company": p.get("affiliation", ""),
                                "position": p.get("title", ""),
                                "linkedin_profile": "",
                            }
                            for p in people
                            if p.get("name")
                        ]
                        story.organizations = orgs
                        story.enrichment_status = "enriched"
                        engine.db.update_story(story)
                        all_people = story.story_people
                        print(
                            f"  ✓ Extracted {len(story.story_people)} people, {len(orgs)} organizations"
                        )
                    else:
                        print("  ⚠ No people found in story")
                        continue
                else:
                    print("  ⚠ AI extraction failed")
                    continue

            print(f"\n[{story.id}] {story.title[:60]}...")

            # Get unique companies from story_people and org_leaders
            companies = set()
            for person in all_people:
                company = person.get("company", "").strip()
                if (
                    company
                    and company not in all_company_data
                    and company not in all_companies_not_found
                ):
                    companies.add(company)

            if not companies:
                print("  No new companies to look up")
                continue

            print(f"  Looking up {len(companies)} company LinkedIn page(s)...")

            # Look up profiles for story_people first, then org_leaders
            story_updated = False
            if story.story_people:
                found, not_found, company_data = lookup.populate_company_profiles(
                    story.story_people,
                    delay_between_requests=1.5,
                )
                all_company_data.update(company_data)
                if found > 0:
                    story_updated = True
                total_companies_found += found
                total_companies_not_found += not_found

            if story.org_leaders:
                found, not_found, company_data = lookup.populate_company_profiles(
                    story.org_leaders,
                    delay_between_requests=1.5,
                )
                all_company_data.update(company_data)
                if found > 0:
                    story_updated = True
                total_companies_found += found
                total_companies_not_found += not_found

            # Track companies that weren't found
            for person in all_people:
                company = person.get("company", "").strip()
                if company and company not in all_company_data:
                    all_companies_not_found.add(company)

            # Track people results
            for person in all_people:
                person_info = {
                    "name": person.get("name", "Unknown"),
                    "company": person.get("company", ""),
                    "position": person.get("position", ""),
                    "linkedin_profile": person.get("linkedin_profile"),
                    "linkedin_profile_type": person.get("linkedin_profile_type"),
                    "story_id": story.id,
                }
                if person.get("linkedin_profile"):
                    people_with_profiles.append(person_info)
                else:
                    people_without_profiles.append(person_info)

            if story_updated:
                # Update the story in the database
                engine.db.update_story(story)
                stories_updated += 1
                profiles_found = sum(1 for p in all_people if p.get("linkedin_profile"))
                print(f"  ✓ Found profiles for {profiles_found} people")

                # Show what was found
                for company, (url, slug) in all_company_data.items():
                    slug_display = f" (slug: {slug})" if slug else ""
                    print(f"    • {company}: {url}{slug_display}")
            else:
                print("  ✗ No company pages found")
    finally:
        # Clean up resources
        lookup.close()

    print("\n" + "=" * 60)
    print("LinkedIn Profile Lookup Summary")
    print("=" * 60)
    print(f"Stories processed: {len(stories)}")
    print(f"Stories updated: {stories_updated}")
    print(f"Companies found: {total_companies_found}")
    print(f"Companies not found: {total_companies_not_found}")

    # Summary of people with profiles
    if people_with_profiles:
        print(f"\n✓ People WITH LinkedIn Profiles ({len(people_with_profiles)}):")
        # Group by profile type
        by_type: dict[str, list] = {}
        for p in people_with_profiles:
            ptype = p.get("linkedin_profile_type") or "unknown"
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append(p)

        for ptype, people in sorted(by_type.items()):
            type_label = {
                "personal": "👤 Personal profiles",
                "department": "🏢 Department pages",
                "organization": "🏛️ Organization pages",
            }.get(ptype, f"  {ptype}")
            print(f"  {type_label}: {len(people)}")
            for p in people[:5]:  # Show first 5 of each type
                company_str = f" @ {p['company']}" if p["company"] else ""
                print(f"    • {p['name']}{company_str}")
            if len(people) > 5:
                print(f"    ... and {len(people) - 5} more")

    # Summary of people without profiles
    if people_without_profiles:
        print(f"\n✗ People WITHOUT LinkedIn Profiles ({len(people_without_profiles)}):")
        for p in people_without_profiles[:10]:  # Show first 10
            company_str = f" @ {p['company']}" if p["company"] else ""
            position_str = f" ({p['position']})" if p["position"] else ""
            print(f"  • {p['name']}{company_str}{position_str}")
        if len(people_without_profiles) > 10:
            print(f"  ... and {len(people_without_profiles) - 10} more")

    # Company details
    if all_companies_not_found:
        print("\n✗ Companies NOT Found:")
        for company in sorted(all_companies_not_found):
            print(f"  • {company}")

    if all_company_data:
        print("\n✓ Company LinkedIn Pages Found:")
        for company, (url, slug) in sorted(all_company_data.items()):
            slug_display = f" (slug: {slug})" if slug else ""
            print(f"  • {company}: {url}{slug_display}")


def _test_enrichment(engine: ContentEngine) -> None:
    """LinkedIn Profile Enrichment - finds profiles and populates URNs."""
    print("\n--- LinkedIn Profile Enrichment ---")

    # Get stats for all stories
    import json as json_module

    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, story_people, org_leaders FROM stories
            WHERE (story_people IS NOT NULL AND story_people != '[]')
               OR (org_leaders IS NOT NULL AND org_leaders != '[]')
        """)
        rows = cursor.fetchall()

    total_stories = len(rows)
    with_profiles = 0
    with_urns = 0
    total_people = 0
    people_with_profiles = 0
    people_with_urns = 0

    for row in rows:
        story_people = (
            json_module.loads(row["story_people"]) if row["story_people"] else []
        )
        org_leaders = (
            json_module.loads(row["org_leaders"]) if row["org_leaders"] else []
        )
        people = story_people + org_leaders
        total_people += len(people)

        has_profile = False
        has_urn = False
        for p in people:
            if p.get("linkedin_profile"):
                people_with_profiles += 1
                has_profile = True
            if p.get("linkedin_urn"):
                people_with_urns += 1
                has_urn = True

        if has_profile:
            with_profiles += 1
        if has_urn:
            with_urns += 1

    if total_stories == 0:
        print("No stories with people. Search for stories first (Choice 16).")
        return

    # Show current status
    print("Current Status:")
    print(f"  Stories with people: {total_stories}")
    print(f"  People with LinkedIn profiles: {people_with_profiles}/{total_people}")
    print(f"  People with URNs (ready for @mention): {people_with_urns}/{total_people}")

    # Calculate what needs to be done
    need_profiles = total_people - people_with_profiles
    need_urns = people_with_profiles - people_with_urns

    if need_profiles == 0 and need_urns == 0:
        print("\n✓ All enrichment complete!")
        _show_enrichment_summary(engine)
        return

    # Run profile lookup if needed
    if need_profiles > 0:
        print(f"\nStep 1: Looking up LinkedIn profiles for {need_profiles} people...")
        _run_profile_lookup(engine)

    # Run URN extraction if needed
    if need_urns > 0 or need_profiles > 0:
        print("\nStep 2: Extracting URNs...")
        _run_mention_extraction(engine)

    print("\n" + "=" * 60)
    print("Enrichment Complete")
    print("=" * 60)
    _show_enrichment_summary(engine)


def _show_enrichment_summary(engine: ContentEngine) -> None:
    """Show comprehensive summary of enrichment status."""
    import json as json_module

    with engine.db._get_connection() as conn:
        cursor = conn.cursor()

        # Get overall stats in one query
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN (story_people IS NOT NULL AND story_people != '[]') OR (org_leaders IS NOT NULL AND org_leaders != '[]') THEN 1 ELSE 0 END) as with_people,
                SUM(CASE WHEN promotion IS NOT NULL AND promotion != '' THEN 1 ELSE 0 END) as with_promotion,
                SUM(CASE WHEN image_path IS NOT NULL AND image_path != '' THEN 1 ELSE 0 END) as with_image,
                SUM(CASE WHEN verification_status = 'approved' THEN 1 ELSE 0 END) as approved
            FROM stories
        """)
        stats = cursor.fetchone()
        total = stats["total"] or 0
        with_people = stats["with_people"] or 0
        with_promotion = stats["with_promotion"] or 0
        with_image = stats["with_image"] or 0
        approved = stats["approved"] or 0

        # Count LinkedIn enrichment details
        cursor.execute("""
            SELECT story_people, org_leaders FROM stories
            WHERE (story_people IS NOT NULL AND story_people != '[]')
               OR (org_leaders IS NOT NULL AND org_leaders != '[]')
        """)
        total_people = 0
        with_profiles = 0
        with_urns = 0
        for row in cursor.fetchall():
            story_people = (
                json_module.loads(row["story_people"]) if row["story_people"] else []
            )
            org_leaders = (
                json_module.loads(row["org_leaders"]) if row["org_leaders"] else []
            )
            for p in story_people + org_leaders:
                total_people += 1
                if p.get("linkedin_profile"):
                    with_profiles += 1
                if p.get("linkedin_urn"):
                    with_urns += 1

    # Display summary
    print(f"\nStories: {total} total")
    print(f"  • With people: {with_people}")
    print(f"  • With promotion message: {with_promotion}")
    print(f"  • With image: {with_image}")
    print(f"  • Approved: {approved}")

    if total_people > 0:
        print(f"\nLinkedIn Enrichment: {total_people} people identified")
        print(
            f"  • With LinkedIn profiles: {with_profiles} ({100 * with_profiles // total_people}%)"
        )
        print(
            f"  • With URNs for @mentions: {with_urns} ({100 * with_urns // total_people}%)"
        )

    # Show what's missing/next steps
    missing = []
    if with_people < total:
        missing.append(f"{total - with_people} stories need people extraction")
    if with_profiles < total_people:
        missing.append(f"{total_people - with_profiles} people need LinkedIn profiles")
    if with_urns < with_profiles:
        missing.append(f"{with_profiles - with_urns} profiles need URN extraction")
    if with_promotion < total:
        missing.append(f"{total - with_promotion} stories need promotion messages")

    if missing:
        print("\nNext steps:")
        for m in missing:
            print(f"  ⚠ {m}")
    else:
        print("\n✓ All enrichment complete!")


def _run_profile_lookup(engine: ContentEngine) -> None:
    """Run LinkedIn profile lookup for stories needing profiles."""
    stories_needing_profiles = _get_stories_needing_profile_lookup(engine)

    if not stories_needing_profiles:
        print("✓ All stories already have LinkedIn profiles.")
        return

    print(f"Looking up profiles for {len(stories_needing_profiles)} stories...")
    _lookup_linkedin_profiles_for_people(engine, stories_needing_profiles)


def _run_mention_extraction(engine: ContentEngine) -> None:
    """Run URN extraction for people with profiles but no URNs."""
    import json as json_module

    # Check how many people need URN extraction across story_people and org_leaders
    pending = 0
    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT story_people, org_leaders FROM stories
            WHERE (story_people LIKE '%linkedin_profile%')
               OR (org_leaders LIKE '%linkedin_profile%')
        """)
        for row in cursor.fetchall():
            story_people = (
                json_module.loads(row["story_people"]) if row["story_people"] else []
            )
            org_leaders = (
                json_module.loads(row["org_leaders"]) if row["org_leaders"] else []
            )
            for p in story_people + org_leaders:
                if p.get("linkedin_profile") and not p.get("linkedin_urn"):
                    pending += 1

    if pending == 0:
        print("✓ All people already have URNs.")
        return

    print(f"Extracting URNs for {pending} people...")
    print("Note: This loads each LinkedIn profile page to extract URNs.\n")

    try:
        enriched, skipped = engine.enricher.populate_linkedin_mentions()
        print(f"\nResult: {enriched} stories updated, {skipped} skipped")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("URN extraction failed")


def _find_org_leaders(engine: ContentEngine) -> None:
    """Find key leaders for organizations in enriched stories."""
    print("\n--- Finding Organization Leaders ---")

    try:
        enriched, skipped = engine.enricher.find_org_leaders()
        print(f"\nResult: {enriched} stories with leaders found, {skipped} skipped")

        if enriched > 0:
            print("\nStories with organization leaders:")
            with engine.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, organizations, org_leaders
                    FROM stories
                    WHERE org_leaders IS NOT NULL AND org_leaders != '[]'
                    ORDER BY id DESC LIMIT 10
                """)
                for row in cursor.fetchall():
                    import json

                    orgs = (
                        json.loads(row["organizations"]) if row["organizations"] else []
                    )
                    leaders = (
                        json.loads(row["org_leaders"]) if row["org_leaders"] else []
                    )
                    print(f"\n  [{row['id']}] {row['title'][:50]}...")
                    print(f"       Organizations: {', '.join(orgs[:3])}")
                    print(f"       Leaders found: {len(leaders)}")
                    for leader in leaders[:5]:
                        name = leader.get("name", "Unknown")
                        title = leader.get("title", "")
                        org = leader.get("organization", "")
                        print(f"         → {name} ({title}) @ {org}")

    except Exception as e:
        print(f"\nError finding leaders: {e}")
        logger.exception("Leader enrichment failed")


def _enrich_individuals(engine: ContentEngine) -> None:
    """Find key individuals and their LinkedIn profiles for enriched stories (legacy)."""
    print("\n--- Finding Key Individuals ---")

    try:
        enriched, skipped = engine.enricher.enrich_individuals_for_stories()
        print(f"\nResult: {enriched} stories with individuals found, {skipped} skipped")

        if enriched > 0:
            print("\nStories with identified individuals:")
            with engine.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, individuals, linkedin_profiles
                    FROM stories
                    WHERE individuals IS NOT NULL AND individuals != '[]'
                    ORDER BY id DESC LIMIT 10
                """)
                for row in cursor.fetchall():
                    import json

                    individuals = (
                        json.loads(row["individuals"]) if row["individuals"] else []
                    )
                    profiles = (
                        json.loads(row["linkedin_profiles"])
                        if row["linkedin_profiles"]
                        else []
                    )
                    print(f"\n  [{row['id']}] {row['title'][:50]}...")
                    print(f"       Individuals: {', '.join(individuals)}")
                    print(f"       LinkedIn profiles: {len(profiles)} found")
                    for profile in profiles[:3]:
                        name = profile.get("name", "Unknown")
                        title = profile.get("title", "")
                        url = profile.get("linkedin_url", "No URL")
                        print(f"         → {name} ({title}): {url[:60]}...")

    except Exception as e:
        print(f"\nError finding individuals: {e}")
        logger.exception("Individual enrichment failed")


def _test_scheduling(engine: ContentEngine) -> None:
    """Test the scheduling component - schedules ALL available stories."""
    print("\n--- Schedule All Available Stories ---")
    print(f"Publish hours: {Config.START_PUB_TIME} - {Config.END_PUB_TIME}")
    print(f"Max stories per day: {Config.MAX_STORIES_PER_DAY}")
    print(f"Jitter: ±{Config.JITTER_MINUTES} minutes")

    # Show existing scheduled stories
    existing_scheduled = engine.db.get_scheduled_stories()
    if existing_scheduled:
        print(f"\nAlready scheduled: {len(existing_scheduled)} stories")
        latest = max(s.scheduled_time for s in existing_scheduled if s.scheduled_time)
        print(f"Latest scheduled for: {latest.strftime('%Y-%m-%d %H:%M')}")
        print("New stories will be scheduled AFTER existing ones.")
    else:
        print("\nNo existing scheduled stories.")

    # Count available stories
    available = engine.db.count_unpublished_stories()
    print(f"\nAvailable approved stories to schedule: {available}")

    if available == 0:
        print("\nNo approved stories available to schedule.")
        print("Running full pipeline to generate content...")
        try:
            engine.run_search_cycle()
            # Re-check available stories after pipeline
            available = engine.db.count_unpublished_stories()
            print(f"\nAfter pipeline: {available} stories available to schedule")
            if available == 0:
                print(
                    "Still no stories available after pipeline. Check search sources."
                )
                return
        except Exception as e:
            print(f"\nError running pipeline: {e}")
            logger.exception("Pipeline failed during scheduling")
            return

    try:
        scheduled = engine.scheduler.schedule_stories(schedule_all=True)
        print(f"\nResult: Scheduled {len(scheduled)} new stories")
        if scheduled:
            first_date = min(s.scheduled_time for s in scheduled if s.scheduled_time)
            last_date = max(s.scheduled_time for s in scheduled if s.scheduled_time)
            print(f"  From: {first_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"  To:   {last_date.strftime('%Y-%m-%d %H:%M')}")
        print("\n" + engine.scheduler.get_schedule_summary())
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Scheduling test failed")


def _human_validation(engine: ContentEngine) -> None:
    """Launch the web-based human validation GUI."""
    print("\n--- Human Validation (Web GUI) ---")
    print("This will open a web browser for reviewing and approving stories.")
    print("You can Accept, Reject, or Edit each story before publication.")

    try:
        from validation_server import run_human_validation

        run_human_validation(engine.db, port=5000)
    except ImportError as e:
        print(f"\nError: Could not import validation server: {e}")
        print("Make sure Flask is installed: pip install flask")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Human validation failed")


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
                # OpenID userinfo endpoint returns given_name/family_name, not localizedFirstName
                first_name = profile.get("given_name") or profile.get(
                    "localizedFirstName", ""
                )
                last_name = profile.get("family_name") or profile.get(
                    "localizedLastName", ""
                )
                name = profile.get("name") or f"{first_name} {last_name}".strip()
                print(f"  Profile: {name}")
        else:
            print("✗ Connection failed. Check your credentials.")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("LinkedIn connection test failed")


def _send_linkedin_connections(engine: ContentEngine) -> None:
    """Send LinkedIn connection requests to people found in stories."""
    print("\n--- Send LinkedIn Connection Requests ---")

    if not Config.LINKEDIN_ACCESS_TOKEN:
        print("LinkedIn credentials not configured.")
        return

    # Gather all unique people with LinkedIn URNs from stories
    people_map: dict[str, dict] = {}  # urn -> person info

    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT story_people, org_leaders FROM stories
            WHERE (story_people LIKE '%linkedin_urn%')
               OR (org_leaders LIKE '%linkedin_urn%')
        """)
        rows = cursor.fetchall()

    import json

    for row in rows:
        try:
            story_people = (
                json.loads(row["story_people"]) if row["story_people"] else []
            )
            org_leaders = json.loads(row["org_leaders"]) if row["org_leaders"] else []
            for person in story_people + org_leaders:
                urn = person.get("linkedin_urn", "")
                if urn and urn.startswith("urn:li:person:"):
                    # Deduplicate by URN
                    if urn not in people_map:
                        people_map[urn] = {
                            "name": person.get("name", "Unknown"),
                            "linkedin_urn": urn,
                            "company": person.get("company", ""),
                            "linkedin_profile": person.get("linkedin_profile", ""),
                        }
        except (json.JSONDecodeError, TypeError):
            continue

    if not people_map:
        print("\nNo people with valid LinkedIn person URNs found in database.")
        print("Run LinkedIn Profile Enrichment (Option 17) first.")
        return

    people_list = list(people_map.values())
    print(f"\nFound {len(people_list)} unique people with LinkedIn person URNs:\n")

    for i, person in enumerate(people_list, 1):
        company_info = f" ({person['company']})" if person.get("company") else ""
        print(f"  {i}. {person['name']}{company_info}")
        if person.get("linkedin_profile"):
            print(f"      {person['linkedin_profile']}")

    print("\nOptions:")
    print("  a - Send connection requests to ALL")
    print("  1,2,3 - Send to specific people (comma-separated numbers)")
    print("  q - Cancel")

    choice = input("\nYour choice: ").strip().lower()

    if choice == "q":
        print("Cancelled.")
        return

    selected_people: list[dict] = []

    if choice == "a":
        selected_people = people_list
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            for idx in indices:
                if 0 <= idx < len(people_list):
                    selected_people.append(people_list[idx])
        except ValueError:
            print("Invalid selection.")
            return

    if not selected_people:
        print("No valid people selected.")
        return

    print(f"\nWill send connection requests to {len(selected_people)} people:")
    for p in selected_people:
        print(f"  - {p['name']}")

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    print("\nSending connection requests (2 second delay between each)...")
    success, failed, errors = engine.publisher.send_bulk_connection_requests(
        selected_people, delay_seconds=2.0
    )

    print("\n--- Results ---")
    print(f"  Successful: {success}")
    print(f"  Failed: {failed}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")


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


def _test_publish_one_story(engine: ContentEngine) -> None:
    """Test publishing by immediately publishing one approved story and verifying it."""
    print("\n--- Test Publish One Story ---")
    print("This will publish ONE approved story immediately to LinkedIn,")
    print("then verify the post exists on the platform.\n")

    # Check LinkedIn credentials
    if not Config.LINKEDIN_ACCESS_TOKEN or not Config.LINKEDIN_AUTHOR_URN:
        print("✗ LinkedIn credentials not configured.")
        return

    # Test connection first
    print("Testing LinkedIn connection...")
    if not engine.publisher.test_connection():
        print("✗ LinkedIn connection failed. Check your credentials.")
        return
    print("✓ LinkedIn connection OK\n")

    # Get approved unpublished stories
    stories = engine.db.get_approved_unpublished_stories(limit=5)
    if not stories:
        print("No approved unpublished stories available.")
        print("Run Actions 1-3 first to get stories ready for publishing.")
        return

    print(f"Available approved stories ({len(stories)}):")
    for story in stories:
        has_image = "✓" if story.image_path else "✗"
        print(f"  [{story.id}] {story.title}")
        print(f"       Image: {has_image} | Score: {story.quality_score}")

    # Ask which story to publish
    story_input = input("\nEnter story ID to publish (or 'q' to cancel): ").strip()
    if story_input.lower() == "q" or not story_input:
        print("Cancelled.")
        return

    try:
        story_id = int(story_input)
        story = next((s for s in stories if s.id == story_id), None)
        if not story:
            print(f"Story ID {story_id} not found in the available list.")
            return
    except ValueError:
        print("Invalid input. Please enter a numeric story ID.")
        return

    # Final confirmation
    print("\n--- About to publish ---")
    print(f"Title: {story.title}")
    print(f"Summary: {story.summary[:100]}...")
    if story.image_path:
        print(f"Image: {story.image_path}")

    confirm = input("\nPublish this story NOW to LinkedIn? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    # Publish immediately
    print("\n[1/3] Publishing to LinkedIn...")
    try:
        post_id = engine.publisher.publish_immediately(story)
        if not post_id:
            print("✗ Publishing failed. Check logs for details.")
            return
        print(f"✓ Published! Post ID: {post_id}")
    except Exception as e:
        print(f"✗ Publishing error: {e}")
        logger.exception("Test publish failed")
        return

    # Wait a moment for LinkedIn to process
    print("\n[2/3] Waiting for LinkedIn to process...")
    import time

    time.sleep(3)

    # Verify the post exists
    print("\n[3/3] Verifying post exists on LinkedIn...")
    try:
        exists, post_data = engine.publisher.verify_post_exists(post_id)
        if exists:
            print("✓ Post verified on LinkedIn!")
            if post_data:
                created = post_data.get("created", {}).get("time", "Unknown")
                print(f"  Created: {created}")
        else:
            print(
                "⚠ Could not verify post (may still exist - API permissions may be limited)"
            )
    except Exception as e:
        print(f"⚠ Verification error: {e}")
        print("  (Post may still have been published successfully)")

    print("\n=== Test Complete ===")
    print(f"Story {story.id} has been published to LinkedIn.")
    print("Check your LinkedIn profile to confirm the post is visible.")


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


def _list_human_approved_stories(engine: ContentEngine) -> None:
    """List human-approved stories that haven't been published yet."""
    print("\n--- Human Approved Stories (Not Yet Published) ---")
    stories = engine.db.get_approved_unpublished_stories(limit=100)

    if not stories:
        print("No human-approved unpublished stories found.")
        print("\nTo approve stories for publication, use Action 22 (Human Validation).")
        return

    print(f"Found {len(stories)} approved stories awaiting publication:\n")
    for story in stories:
        has_image = "Yes" if story.image_path else "No"
        has_promotion = "Yes" if story.promotion else "No"
        scheduled = (
            story.scheduled_time.strftime("%Y-%m-%d %H:%M")
            if story.scheduled_time
            else "Not scheduled"
        )
        print(f"[{story.id}] {story.title}")
        print(
            f"    Score: {story.quality_score} | Image: {has_image} | Promo: {has_promotion}"
        )
        print(f"    Scheduled: {scheduled}")
        print()


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
            result = engine.image_generator._generate_image_for_story(story)
            if result:
                image_path, image_alt_text = result
                story.image_path = image_path
                story.image_alt_text = image_alt_text
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
    print("\n--- Run Full Pipeline ---")
    print("This will:")
    print("  1. Search for new stories")
    print("  2. Enrich with company mentions")
    print("  3. Generate images")
    print("  4. Assign promotion messages")
    print("  5. Verify content quality")
    print("  6. Clean up old stories")
    print("\nNote: Use Action 20 to schedule stories for publication.")

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
        # Import and run tests from run_tests module
        from run_tests import main_with_failures

        exit_code, failed_tests = main_with_failures()

        if exit_code == 0:
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed:")
            for suite_name, test_name in failed_tests:
                print(f"    - [{suite_name}] {test_name}")
    except ImportError as e:
        print(f"\nError importing test module: {e}")
        print("Make sure run_tests.py and test_framework.py are present.")
    except Exception as e:
        print(f"\nError running tests: {e}")
        logger.exception("Unit test execution failed")


def _view_linkedin_analytics(engine: ContentEngine) -> None:
    """View LinkedIn analytics for published stories."""
    print("\n--- LinkedIn Analytics ---")

    published_stories = engine.db.get_published_stories()

    if not published_stories:
        print("\nNo published stories found.")
        return

    print(f"\nFound {len(published_stories)} published stories:\n")
    print("-" * 100)
    print(
        f"{'ID':>4} | {'Title':<40} | {'👁 Impr':>8} | {'👍 Like':>7} | "
        f"{'💬 Cmnt':>7} | {'📤 Share':>8} | {'Last Updated':<16}"
    )
    print("-" * 100)

    for story in published_stories:
        title = (story.title or "Untitled")[:38]
        if len(story.title or "") > 38:
            title += ".."

        last_updated = "Never"
        if story.linkedin_analytics_fetched_at:
            last_updated = story.linkedin_analytics_fetched_at.strftime(
                "%Y-%m-%d %H:%M"
            )

        print(
            f"{story.id:>4} | {title:<40} | {story.linkedin_impressions or 0:>8} | "
            f"{story.linkedin_likes or 0:>7} | {story.linkedin_comments or 0:>7} | "
            f"{story.linkedin_shares or 0:>8} | {last_updated:<16}"
        )

    print("-" * 100)

    # Summary statistics
    total_impressions = sum(s.linkedin_impressions or 0 for s in published_stories)
    total_likes = sum(s.linkedin_likes or 0 for s in published_stories)
    total_comments = sum(s.linkedin_comments or 0 for s in published_stories)
    total_shares = sum(s.linkedin_shares or 0 for s in published_stories)

    print(
        f"\n{'TOTALS':>4} | {'':<40} | {total_impressions:>8} | "
        f"{total_likes:>7} | {total_comments:>7} | {total_shares:>8}"
    )

    # Calculate engagement rate if we have impressions
    if total_impressions > 0:
        engagement_rate = (
            (total_likes + total_comments + total_shares) / total_impressions * 100
        )
        print(f"\nOverall Engagement Rate: {engagement_rate:.2f}%")


def _refresh_linkedin_analytics(engine: ContentEngine) -> None:
    """Refresh LinkedIn analytics for all published stories."""
    print("\n--- Refresh LinkedIn Analytics ---")

    published_stories = engine.db.get_published_stories()

    if not published_stories:
        print("\nNo published stories to refresh.")
        return

    print(f"\nFound {len(published_stories)} published stories.")
    confirm = input("Refresh analytics for all? (y/n): ").strip().lower()

    if confirm != "y":
        print("Cancelled.")
        return

    print("\nFetching analytics from LinkedIn API...")

    success_count, failure_count = engine.publisher.refresh_all_analytics()

    print(f"\n✓ Successfully updated: {success_count}")
    if failure_count > 0:
        print(f"✗ Failed to update: {failure_count}")

    # Show updated analytics
    print("\nUpdated analytics:")
    _view_linkedin_analytics(engine)


def _reset_all(engine: ContentEngine) -> None:
    """Reset everything: delete database and all generated images."""
    print("\n--- RESET ALL DATA ---")
    print("\n⚠️  WARNING: This will permanently delete:")
    print(f"    • Database: {Config.DB_NAME}")
    print(f"    • All files in: {Config.IMAGE_DIR}/")
    print("\n    This action cannot be undone!")

    confirm = input("\nAre you sure you want to reset? Type 'yes' to confirm: ").strip()
    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    # Double-check
    confirm2 = input("This is your last chance. Type 'RESET' to proceed: ").strip()
    if confirm2 != "RESET":
        print("Cancelled.")
        return

    print("\nResetting...")

    # Delete the database file
    db_path = Path(Config.DB_NAME)
    if db_path.exists():
        try:
            db_path.unlink()
            print(f"  ✓ Deleted database: {Config.DB_NAME}")
        except Exception as e:
            print(f"  ✗ Failed to delete database: {e}")
            return

    # Delete all files in generated_images
    image_dir = Path(Config.IMAGE_DIR)
    if image_dir.exists():
        deleted_count = 0
        for file in image_dir.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"  ✗ Failed to delete {file.name}: {e}")
        print(f"  ✓ Deleted {deleted_count} images from {Config.IMAGE_DIR}/")

    # Recreate the database
    from database import Database

    new_db = Database(Config.DB_NAME)
    engine.db = new_db
    print(f"  ✓ Created fresh database: {Config.DB_NAME}")

    print("\n✓ Reset complete!")


def _show_all_prompts() -> None:
    """Display all configured prompts in full."""
    print("\n" + "=" * 80)
    print("ALL CONFIGURED PROMPTS")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("1. SEARCH_PROMPT (topic/criteria for finding stories)")
    print("-" * 80)
    print(Config.SEARCH_PROMPT)

    print("\n" + "-" * 80)
    print("2. SEARCH_INSTRUCTION_PROMPT (system prompt for LLM search)")
    print(
        "   Placeholders: {max_stories}, {search_prompt}, {since_date}, {summary_words}"
    )
    print("-" * 80)
    print(Config.SEARCH_INSTRUCTION_PROMPT)

    print("\n" + "-" * 80)
    print("3. IMAGE_STYLE (style directive for image generation)")
    print("-" * 80)
    print(Config.IMAGE_STYLE)

    print("\n" + "-" * 80)
    print("4. IMAGE_REFINEMENT_PROMPT (LLM prompt to create image prompts)")
    print("   Placeholders: {story_title}, {story_summary}, {image_style}")
    print("-" * 80)
    print(Config.IMAGE_REFINEMENT_PROMPT)

    print("\n" + "-" * 80)
    print("5. IMAGE_FALLBACK_PROMPT (fallback when LLM refinement fails)")
    print("   Placeholders: {story_title}")
    print("-" * 80)
    print(Config.IMAGE_FALLBACK_PROMPT)

    print("\n" + "-" * 80)
    print("6. VERIFICATION_PROMPT (content verification prompt)")
    print(
        "   Placeholders: {search_prompt}, {story_title}, {story_summary}, {story_sources}"
    )
    print("-" * 80)
    print(Config.VERIFICATION_PROMPT)

    if Config.SEARCH_PROMPT_TEMPLATE:
        print("\n" + "-" * 80)
        print(
            "7. SEARCH_PROMPT_TEMPLATE (legacy override - if set, replaces SEARCH_INSTRUCTION_PROMPT)"
        )
        print(
            "   Placeholders: {criteria}, {since_date}, {summary_words}, {max_stories}"
        )
        print("-" * 80)
        print(Config.SEARCH_PROMPT_TEMPLATE)

    print("\n" + "=" * 80)
    print("To customize these prompts, add them to your .env file.")
    print("=" * 80)


def _install_dependencies() -> None:
    """Install dependencies from requirements.txt into the virtual environment."""
    import subprocess
    import sys
    from pathlib import Path

    print("\n--- Install Dependencies ---")

    # Determine the project root (where requirements.txt should be)
    project_root = Path(__file__).parent
    requirements_file = project_root / "requirements.txt"

    if not requirements_file.exists():
        print(f"[!] requirements.txt not found at: {requirements_file}")
        return

    # Determine the pip executable in the virtual environment
    venv_path = project_root / ".venv"
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:
        pip_path = venv_path / "bin" / "pip"

    if not pip_path.exists():
        print(f"[!] Virtual environment pip not found at: {pip_path}")
        print("    Make sure .venv is created. Run: python -m venv .venv")
        return

    print(f"Requirements file: {requirements_file}")
    print(f"Installing to: {venv_path}")
    print("-" * 40)

    try:
        # Run pip install
        result = subprocess.run(
            [str(pip_path), "install", "-r", str(requirements_file)],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        if result.returncode == 0:
            print("✓ Dependencies installed successfully!")
            # Show summary of what was installed/updated
            output_lines = result.stdout.strip().split("\n")
            for line in output_lines[-10:]:  # Show last 10 lines
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"[!] Installation failed with exit code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")

    except Exception as e:
        print(f"[!] Error running pip: {e}")


def _test_api_keys(engine: ContentEngine) -> None:
    """Test all configured API keys to verify they are valid and working."""
    print("\n--- API Key Validation ---")
    print("Testing configured API connections...\n")

    results: list[tuple[str, bool, str]] = []

    # 1. Test Gemini API
    print("  Testing Gemini API...", end=" ", flush=True)
    if Config.GEMINI_API_KEY:
        try:
            response = api_client.gemini_generate(
                client=engine.genai_client,
                model=Config.MODEL_TEXT,
                contents="Say 'API key is valid' in exactly 4 words.",
                config={"max_output_tokens": 20},
                endpoint="api_test",
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
            # Test with the whoami-v2 endpoint (v1 is deprecated)
            headers = {"Authorization": f"Bearer {Config.HUGGINGFACE_API_TOKEN}"}
            response = api_client.http_get(
                url="https://huggingface.co/api/whoami-v2",
                headers=headers,
                timeout=10,
                endpoint="huggingface_test",
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
            response = api_client.linkedin_request(
                method="GET",
                url="https://api.linkedin.com/v2/userinfo",
                headers=headers,
                timeout=10,
                endpoint="api_test",
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
            response = api_client.http_get(
                url=f"{Config.LM_STUDIO_BASE_URL}/models",
                timeout=5,
                endpoint="lm_studio_test",
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


def main() -> int:
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
