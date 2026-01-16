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
from company_mention_enricher import CompanyMentionEnricher, get_validation_cache
from scheduler import Scheduler
from linkedin_publisher import LinkedInPublisher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("google.genai.models").setLevel(logging.WARNING)
logging.getLogger("undetected_chromedriver").setLevel(logging.WARNING)
logging.getLogger("undetected_chromedriver.patcher").setLevel(logging.WARNING)
logging.getLogger("selenium").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


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

            # Step 2b: Find indirect people (organization leaders)
            logger.info("Step 2b: Finding indirect people...")
            indirect_enriched, indirect_skipped = self.enricher.find_indirect_people()
            logger.info(
                f"Indirect people: {indirect_enriched} enriched, {indirect_skipped} skipped"
            )

            # Log validation cache stats
            cache_stats = get_validation_cache().get_stats()
            logger.info(f"CACHE_STATS: {cache_stats}")

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

        signature_detail = Config.SIGNATURE_BLOCK_DETAIL.strip()
        signature_phrase = f"As a {signature_detail}, " if signature_detail else ""
        base_examples = [
            "This catalyst development aligns perfectly with my background. I'm actively seeking roles in catalysis R&D — please feel free to tag a hiring manager or reach out directly!",
            "Hydrogen production is exactly where I want to contribute. I'm actively job hunting; I’d be grateful if you could connect me with anyone who's hiring!",
            "Carbon capture is central to my career goals. I'm currently seeking process engineering roles — if you're hiring or know someone who is, I'd love to hear from you!",
            "This breakthrough resonates with my sustainable chemistry interests. I'm actively interviewing and would welcome the opportunity to connect with hiring managers!",
            "Process scale-up is my specialty. I'm actively job hunting for roles like this — I'd really appreciate any introductions to recruiters or hiring managers in this field!",
        ]
        style_examples = [
            (signature_phrase + ex) if signature_phrase else ex for ex in base_examples
        ]

        # Get stories that need a promotion message
        stories = self.db.get_stories_needing_promotion(require_image=require_image)

        if not stories:
            return 0

        examples_text = "\n".join(f"- {ex}" for ex in style_examples)
        assigned = 0

        for story in stories:
            story_id = story.id
            if story_id is None:
                continue
            title = story.title
            summary = story.summary or ""

            prompt = f"""Generate a DIRECT, ACTION-ORIENTED job-seeking message for LinkedIn that connects to this specific story.

STORY:
Title: {title}
Summary: {summary[:500]}

YOUR GOAL: You are a {Config.DISCIPLINE} actively seeking employment. This message must:
1. Directly connect YOUR job search to the SPECIFIC technology/topic in this story
2. Make it crystal clear you are ACTIVELY JOB HUNTING (use phrases like "actively seeking", "job hunting", "interviewing", "looking for roles")
3. Include a DIRECT call-to-action asking people to help (tag hiring managers, DM you, connect, refer you)

REQUIREMENTS:
1. Write in FIRST PERSON ("I", "my", "I'm")
2. MUST reference the specific technology, company, or topic from THIS story (not generic)
3. {("Weave this credential naturally: " + signature_detail + " (no trailing tagline)") if signature_detail else "Mention a concise qualification (e.g., MEng) relevant to the story"}
4. MUST include ONE polite, first-person call-to-action:
   - "I'd really appreciate it if you could tag a hiring manager!"
   - "Please feel free to DM me if you're hiring!"
   - "I'd welcome introductions to recruiters in this space!"
   - "If you know of any openings, I'd love to hear from you!"
   - "I'd be grateful for any connections to people hiring in this field!"
5. Sound confident and eager, NOT passive or note-like
6. Keep it to 1-2 punchy sentences, max 250 characters
7. No emojis

BAD EXAMPLES (too passive/vague - DO NOT write like this):
- "Interesting developments in the field." (no job ask)
- "This is relevant to my background." (no CTA)
- "I find this topic fascinating." (sounds like a note)

GOOD EXAMPLES (direct job-seeking with clear CTA):
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

                # Ensure signature detail is woven in when provided
                if (
                    signature_detail
                    and signature_detail.lower() not in promotion.lower()
                ):
                    promotion = (
                        f"As a {signature_detail}, {promotion[0].lower()}{promotion[1:]}"
                        if promotion
                        else f"As a {signature_detail}, actively seeking opportunities."
                    )

                if len(promotion) > 300:
                    promotion = promotion[:297] + "..."

            except Exception as e:
                logger.warning(f"AI generation failed for story {story_id}: {e}")
                promotion = random.choice(style_examples)

                if (
                    signature_detail
                    and signature_detail.lower() not in promotion.lower()
                ):
                    promotion = (
                        f"As a {signature_detail}, {promotion[0].lower()}{promotion[1:]}"
                        if promotion
                        else f"As a {signature_detail}, actively seeking opportunities."
                    )

            self.db.update_story_promotion(story_id, promotion)

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
   23. Test LinkedIn Publish (due stories)
   24. Publish One Scheduled Story Now
   25. Run Unit Tests

  Analytics:
   26. View LinkedIn Analytics
   27. Refresh All Analytics

  Advanced Tools:
   29. Launch Dashboard (Web GUI)
   30. View A/B Tests
   31. Analyze Post Optimization
   32. Check Story Originality
   33. View Source Credibility
   34. Analyze Trends & Freshness
   35. Analyze Intent Classification
   36. Test Notifications

  Danger Zone:
   37. Reset (delete database and images)
   38. Configure LinkedIn Voyager API Cookies
   39. Configure RapidAPI Key (LinkedIn Lookups)

  Connection Management:
   40. Process Connection Queue (send queued requests)
   41. View Connection Tracking Dashboard

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
        # Advanced Tools
        elif choice == "29":
            _launch_dashboard(engine)
        elif choice == "30":
            _view_ab_tests(engine)
        elif choice == "31":
            _analyze_post_optimization(engine)
        elif choice == "32":
            _check_story_originality(engine)
        elif choice == "33":
            _view_source_credibility(engine)
        elif choice == "34":
            _analyze_trends(engine)
        elif choice == "35":
            _analyze_intent_classification(engine)
        elif choice == "36":
            _test_notifications(engine)
        # Danger Zone
        elif choice == "37":
            _reset_all(engine)
        elif choice == "38":
            _configure_linkedin_voyager()
        elif choice == "39":
            _configure_rapidapi_key()
        elif choice == "40":
            _process_connection_queue(engine)
        elif choice == "41":
            _view_connection_dashboard(engine)
        else:
            print("Invalid choice. Please try again.")

        try:
            input("\nPress Enter to continue...")
        except EOFError:
            pass  # Gracefully handle piped input


def _test_search(engine: ContentEngine) -> None:
    """Search for stories, enrich with LinkedIn profiles/URNs, and assign promotions.

    This is the main enrichment pipeline (Action 16) that:
    1. Searches for new stories matching the configured search prompt
    2. Extracts direct people (mentioned in stories) using NER + AI
    3. Finds indirect people (org leadership: executives, academics, HR, PR)
    4. Looks up LinkedIn profiles with triangulation validation
    5. Extracts URNs for @mentions in LinkedIn posts
    6. Assigns promotional messages
    7. Offers to send connection requests

    Optimized flow (5 steps):
    1. Search for stories
    2. Enrich stories (direct people, indirect people, LinkedIn profiles, URNs - single pass)
    3. Assign promotional messages
    4. Show people discovery summary
    5. Offer connection requests
    """
    print("\n--- Story Search, Enrichment & Promotions ---")
    print(f"Search prompt: {Config.SEARCH_PROMPT[:80]}...")
    print(f"Lookback days: {Config.SEARCH_LOOKBACK_DAYS}")
    print(f"Use last checked date: {Config.USE_LAST_CHECKED_DATE}")
    print(f"Max stories per search: {Config.MAX_STORIES_PER_SEARCH}")
    print(f"Max people per story: {Config.MAX_PEOPLE_PER_STORY}")

    try:
        start_date = engine.searcher.get_search_start_date()
        print(f"Searching for stories since: {start_date}")

        # Step 1: Search for stories
        print("\n[Step 1/5] Searching for stories...")
        new_count = engine.searcher.search_and_process()
        print(f"  → Found and saved {new_count} new stories")

        if new_count > 0:
            print("\nNewly saved stories (by quality score):")
            # Get the most recent stories, sorted by quality score
            recent_stories = engine.db.get_recent_stories(limit=new_count)
            # Sort by quality score descending (highest first)
            for story_id, title, quality_score in sorted(
                recent_stories, key=lambda r: r[2], reverse=True
            ):
                print(f"  - [{story_id}] {title[:60]}... (Score: {quality_score})")

        # Step 2: Full enrichment (direct+indirect people, LinkedIn profiles, URNs)
        # This consolidates the previous Steps 2-4 into a single efficient pass
        print("\n[Step 2/5] Enriching stories (people, profiles, URNs)...")
        enriched, skipped = engine.enricher.enrich_pending_stories()
        print(f"  → Enriched {enriched} stories, skipped {skipped}")

        # Extract URNs for any profiles that don't have them yet
        # (This uses the same browser session if already open)
        urn_enriched, urn_pending = _run_urn_extraction_silent(engine)

        # Step 3: Assign promotion messages
        print("\n[Step 3/5] Assigning promotion messages...")
        _run_promotion_assignment_silent(engine)

        # Step 4: Summary of people found
        print("\n[Step 4/5] People Discovery Summary...")
        _show_people_discovery_summary(engine)

        # Show detailed story information
        print("\n" + "=" * 60)
        print("Detailed Story Results")
        print("=" * 60)
        log_story_details(engine)

        # Show summary
        print("\n" + "=" * 60)
        print("Search, Enrichment & Promotions Complete")
        print("=" * 60)
        _show_enrichment_summary(engine)

        # Step 5: Offer to send connection requests to discovered people
        print("\n[Step 5/5] Connection Requests...")
        _offer_connection_requests(engine)

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
    return engine.db.get_stories_needing_profile_lookup()


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
    # Check how many people need URN extraction
    pending = engine.db.count_people_needing_urns()

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
    stories_needing = engine.db.get_stories_needing_promotion()
    count = len(stories_needing)

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


def _run_indirect_people_enrichment_silent(engine: ContentEngine) -> int:
    """Find indirect people (org leadership) for organizations. Returns count enriched."""
    # Get stories with organizations but no indirect_people
    stories = engine.db.get_stories_needing_indirect_people()

    if not stories:
        print("  → All stories already have indirect people")
        return 0

    print(
        f"  → Finding indirect people for {len(stories)} stories with organizations..."
    )

    enriched = 0
    for story in stories:
        story_id = story.id
        if story_id is None:
            continue
        orgs = story.organizations

        if not orgs:
            continue

        indirect_people = []
        for org in orgs:
            try:
                people = engine.enricher._get_indirect_people(org)
                if people:
                    indirect_people.extend(people)
                    print(f"    ✓ {org}: Found {len(people)} indirect people")
                else:
                    print(f"    ⏭ {org}: No indirect people found")
            except Exception as e:
                logger.debug(f"Error getting indirect people for {org}: {e}")
                continue

        if indirect_people:
            # Look up LinkedIn profiles for indirect_people (same as direct_people)
            people_for_lookup = [
                {
                    "name": person.get("name", ""),
                    "title": person.get("title", ""),
                    "affiliation": person.get("organization", ""),
                }
                for person in indirect_people
                if person.get("name")
            ]

            if people_for_lookup:
                try:
                    linkedin_profiles = engine.enricher._find_linkedin_profiles_batch(
                        people_for_lookup
                    )

                    # Update indirect_people with found LinkedIn profiles
                    if linkedin_profiles:
                        profiles_by_name = {
                            p.get("name", "").lower(): p for p in linkedin_profiles
                        }
                        profiles_found = 0
                        for person in indirect_people:
                            name_lower = person.get("name", "").lower()
                            if name_lower in profiles_by_name:
                                profile = profiles_by_name[name_lower]
                                person["linkedin_profile"] = profile.get(
                                    "linkedin_url", ""
                                )
                                profiles_found += 1
                        if profiles_found > 0:
                            print(
                                f"    ✓ Found {profiles_found} LinkedIn profiles for indirect people"
                            )
                except Exception as e:
                    logger.debug(f"Error looking up LinkedIn profiles: {e}")

            # Update story with indirect_people
            engine.db.update_story_indirect_people(story_id, indirect_people)
            enriched += 1

    print(f"  → Updated {enriched}/{len(stories)} stories with indirect people")
    return enriched


def _mark_stories_enriched(engine: ContentEngine) -> int:
    """Mark stories as enriched after all enrichment steps complete. Returns count updated."""
    updated = engine.db.mark_stories_enriched()

    if updated > 0:
        print(f"  → Marked {updated} stories as enriched")
    else:
        print("  → No stories needed status update")
    return updated


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
    """Get stories that have direct_people or indirect_people with missing LinkedIn profiles."""
    import json as json_module

    stories_needing_profiles = []
    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, direct_people, indirect_people
            FROM stories
            WHERE (direct_people IS NOT NULL AND direct_people != '[]')
               OR (indirect_people IS NOT NULL AND indirect_people != '[]')
        """)
        for row in cursor.fetchall():
            try:
                direct_people = (
                    json_module.loads(row["direct_people"])
                    if row["direct_people"]
                    else []
                )
                indirect_people = (
                    json_module.loads(row["indirect_people"])
                    if row["indirect_people"]
                    else []
                )
                all_people = direct_people + indirect_people
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
    """Look up LinkedIn company pages for organizations in direct_people and indirect_people."""
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
            # Combine direct_people and indirect_people for lookup
            all_people = (story.direct_people or []) + (story.indirect_people or [])

            # If story has no people, try to extract them using AI
            if not all_people:
                print(f"\n[{story.id}] {story.title[:60]}...")
                print("  No people - extracting with AI...")
                result = engine.enricher._extract_orgs_and_people(story)
                if result:
                    people = result.get("direct_people", [])
                    orgs = result.get("organizations", [])
                    if people:
                        # Store in direct_people (the new primary field)
                        story.direct_people = [
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
                        all_people = story.direct_people
                        print(
                            f"  ✓ Extracted {len(story.direct_people)} people, {len(orgs)} organizations"
                        )
                    else:
                        print("  ⚠ No people found in story")
                        continue
                else:
                    print("  ⚠ AI extraction failed")
                    continue

            print(f"\n[{story.id}] {story.title[:60]}...")

            # Get unique companies from direct_people and indirect_people
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

            # Look up profiles for direct_people first, then indirect_people
            story_updated = False
            if story.direct_people:
                found, not_found, company_data = lookup.populate_company_profiles(
                    story.direct_people,
                    delay_between_requests=1.5,
                )
                all_company_data.update(company_data)
                if found > 0:
                    story_updated = True
                total_companies_found += found
                total_companies_not_found += not_found

            if story.indirect_people:
                found, not_found, company_data = lookup.populate_company_profiles(
                    story.indirect_people,
                    delay_between_requests=1.5,
                )
                all_company_data.update(company_data)
                if found > 0:
                    story_updated = True
                total_companies_found += found
                total_companies_not_found += not_found

            # Sync profiles between direct_people and indirect_people
            # (same person may appear in both lists)
            profile_lookup = {}
            for person in (story.direct_people or []) + (story.indirect_people or []):
                name = person.get("name", "").strip().lower()
                company = person.get("company", "").strip().lower()
                key = f"{name}@{company}"
                if person.get("linkedin_profile") and key not in profile_lookup:
                    profile_lookup[key] = {
                        "linkedin_profile": person.get("linkedin_profile"),
                        "linkedin_profile_type": person.get("linkedin_profile_type"),
                        "linkedin_slug": person.get("linkedin_slug"),
                        "linkedin_urn": person.get("linkedin_urn"),
                    }

            # Apply found profiles to any matching people who don't have them
            for person in (story.direct_people or []) + (story.indirect_people or []):
                if not person.get("linkedin_profile"):
                    name = person.get("name", "").strip().lower()
                    company = person.get("company", "").strip().lower()
                    key = f"{name}@{company}"
                    if key in profile_lookup:
                        person.update(profile_lookup[key])

            # Rebuild all_people after sync
            all_people = (story.direct_people or []) + (story.indirect_people or [])

            # Track companies that weren't found
            for person in all_people:
                company = person.get("company", "").strip()
                if company and company not in all_company_data:
                    all_companies_not_found.add(company)

            # Track people results (deduplicate by name+company)
            for person in all_people:
                name = person.get("name", "Unknown")
                company = person.get("company", "")
                dedup_key = f"{name.lower().strip()}@{company.lower().strip()}"

                person_info = {
                    "name": name,
                    "company": company,
                    "position": person.get("position", ""),
                    "linkedin_profile": person.get("linkedin_profile"),
                    "linkedin_profile_type": person.get("linkedin_profile_type"),
                    "story_id": story.id,
                }

                if person.get("linkedin_profile"):
                    # Only add if not already tracked (avoid duplicates)
                    if not any(
                        f"{p['name'].lower().strip()}@{p['company'].lower().strip()}"
                        == dedup_key
                        for p in people_with_profiles
                    ):
                        people_with_profiles.append(person_info)
                else:
                    # Only add if not already tracked and doesn't have profile elsewhere
                    already_has_profile = any(
                        f"{p['name'].lower().strip()}@{p['company'].lower().strip()}"
                        == dedup_key
                        for p in people_with_profiles
                    )
                    already_in_without = any(
                        f"{p['name'].lower().strip()}@{p['company'].lower().strip()}"
                        == dedup_key
                        for p in people_without_profiles
                    )
                    if not already_has_profile and not already_in_without:
                        people_without_profiles.append(person_info)

            if story_updated:
                # Update the story in the database
                engine.db.update_story(story)
                stories_updated += 1
                profiles_found = sum(1 for p in all_people if p.get("linkedin_profile"))
                print(f"  ✓ Found profiles for {profiles_found} people")

                # Log per-story metrics
                _log_story_enrichment_metrics(story, context="after_profile_lookup")

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

    # Cache statistics (shows cross-story caching efficiency)
    cache_stats = lookup.get_cache_stats()
    unified = cache_stats.get("unified_cache", {})
    total_entries = unified.get("total_entries", 0)
    if total_entries > 0:
        print("\n📊 Cache Statistics (cross-story efficiency):")
        print(
            f"  • Total: {total_entries} entries "
            f"(hits: {unified.get('hits', 0)}, misses: {unified.get('misses', 0)}, "
            f"hit rate: {unified.get('hit_rate', '0%')})"
        )

    # Timing statistics
    timing_stats = lookup.get_timing_stats()
    total_searches = sum(s["count"] for s in timing_stats.values())  # type: ignore[call-overload]
    if total_searches > 0:
        print("\n⏱️ Search Timing Statistics:")
        for op_type, stats in timing_stats.items():
            if stats["count"] > 0:
                print(
                    f"  • {op_type.replace('_', ' ').title()}: "
                    f"{stats['count']} searches, {stats['total']:.1f}s total, "
                    f"{stats['avg']:.1f}s avg"
                )

    # Gemini fallback statistics
    gemini_stats = lookup.get_gemini_stats()
    if gemini_stats["attempts"] > 0:
        status = " (DISABLED)" if gemini_stats["disabled"] else ""
        print(
            f"\n🤖 Gemini Fallback{status}: "
            f"{gemini_stats['successes']}/{gemini_stats['attempts']} successful "
            f"({gemini_stats['success_rate']}%)"
        )

    # Log metrics for this batch
    total_processed = len(people_with_profiles) + len(people_without_profiles)
    match_rate = (
        len(people_with_profiles) / total_processed * 100 if total_processed > 0 else 0
    )
    logger.info(
        f"BASELINE:profile_lookup_batch "
        f"stories_processed={len(stories)} "
        f"stories_updated={stories_updated} "
        f"total_people={total_processed} "
        f"with_linkedin={len(people_with_profiles)} "
        f"without_linkedin={len(people_without_profiles)} "
        f"match_rate={match_rate:.1f}% "
        f"companies_found={total_companies_found} "
        f"companies_not_found={total_companies_not_found}"
    )


def _test_enrichment(engine: ContentEngine) -> None:
    """LinkedIn Profile Enrichment - finds profiles and populates URNs."""
    print("\n--- LinkedIn Profile Enrichment ---")

    # Get stats for all stories
    import json as json_module

    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, direct_people, indirect_people FROM stories
            WHERE (direct_people IS NOT NULL AND direct_people != '[]')
               OR (indirect_people IS NOT NULL AND indirect_people != '[]')
        """)
        rows = cursor.fetchall()

    total_stories = len(rows)
    with_profiles = 0
    with_urns = 0
    total_people = 0
    people_with_profiles = 0
    people_with_urns = 0

    for row in rows:
        direct_people = (
            json_module.loads(row["direct_people"]) if row["direct_people"] else []
        )
        indirect_people = (
            json_module.loads(row["indirect_people"]) if row["indirect_people"] else []
        )
        people = direct_people + indirect_people
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


# =============================================================================
# ENRICHMENT METRICS
# =============================================================================


def log_enrichment_baseline_metrics(engine: ContentEngine) -> dict:
    """
    Log comprehensive enrichment metrics for baseline measurement.

    This function captures the current state of enrichment quality to establish
    baseline metrics BEFORE any pipeline changes are made. Run for 1 week to
    collect sufficient data for comparison.

    Metrics logged (searchable via BASELINE: prefix):
    - linkedin_match_rate: % of people with valid LinkedIn profiles
    - high_confidence_rate: % of matches marked as high confidence
    - zero_match_stories: % of stories with no LinkedIn profiles found
    - urn_extraction_rate: % of profiles with URNs for @mentions

    Returns:
        dict with all computed metrics for programmatic access
    """
    import json as json_module
    from datetime import datetime

    with engine.db._get_connection() as conn:
        cursor = conn.cursor()

        # Get stories with people/leaders for analysis
        cursor.execute("""
            SELECT id, title, direct_people, indirect_people, organizations, enrichment_quality
            FROM stories
            WHERE (direct_people IS NOT NULL AND direct_people != '[]')
               OR (indirect_people IS NOT NULL AND indirect_people != '[]')
               OR (organizations IS NOT NULL AND organizations != '[]')
        """)
        rows = cursor.fetchall()

    # Aggregate metrics
    total_stories = len(rows)
    stories_with_any_linkedin = 0
    stories_with_zero_linkedin = 0
    total_people = 0
    people_with_profile = 0
    people_with_urn = 0
    high_confidence_matches = 0
    medium_confidence_matches = 0
    low_confidence_matches = 0
    org_fallback_matches = 0
    quality_high = 0
    quality_medium = 0
    quality_low = 0
    quality_failed = 0

    for row in rows:
        direct_people = (
            json_module.loads(row["direct_people"]) if row["direct_people"] else []
        )
        indirect_people = (
            json_module.loads(row["indirect_people"]) if row["indirect_people"] else []
        )
        all_people = direct_people + indirect_people
        total_people += len(all_people)

        # Track enrichment quality per story
        eq = row["enrichment_quality"] or ""
        if eq == "high":
            quality_high += 1
        elif eq == "medium":
            quality_medium += 1
        elif eq == "low":
            quality_low += 1
        elif eq == "failed":
            quality_failed += 1

        story_has_linkedin = False
        for p in all_people:
            if p.get("linkedin_profile"):
                people_with_profile += 1
                story_has_linkedin = True

                # Track confidence levels (if available)
                confidence = p.get("match_confidence", "").lower()
                if confidence in ("high", "verified", "validated_high"):
                    high_confidence_matches += 1
                elif confidence in ("medium", "validated_medium"):
                    medium_confidence_matches += 1
                elif confidence == "org_fallback":
                    org_fallback_matches += 1
                else:
                    low_confidence_matches += 1

            if p.get("linkedin_urn"):
                people_with_urn += 1

        if story_has_linkedin:
            stories_with_any_linkedin += 1
        elif all_people:  # Had people but none matched
            stories_with_zero_linkedin += 1

    # Calculate rates (avoid division by zero)
    linkedin_match_rate = (
        (people_with_profile / total_people * 100) if total_people > 0 else 0
    )
    urn_rate = (
        (people_with_urn / people_with_profile * 100) if people_with_profile > 0 else 0
    )
    zero_match_rate = (
        (stories_with_zero_linkedin / total_stories * 100) if total_stories > 0 else 0
    )
    quality_total = quality_high + quality_medium + quality_low + quality_failed
    quality_high_rate = (quality_high / quality_total * 100) if quality_total > 0 else 0
    high_confidence_rate = (
        (high_confidence_matches / people_with_profile * 100)
        if people_with_profile > 0
        else 0
    )

    # Build metrics dict
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_stories": total_stories,
        "total_people": total_people,
        "people_with_profile": people_with_profile,
        "people_with_urn": people_with_urn,
        "linkedin_match_rate_pct": round(linkedin_match_rate, 1),
        "urn_extraction_rate_pct": round(urn_rate, 1),
        "zero_match_stories": stories_with_zero_linkedin,
        "zero_match_rate_pct": round(zero_match_rate, 1),
        "high_confidence_matches": high_confidence_matches,
        "medium_confidence_matches": medium_confidence_matches,
        "low_confidence_matches": low_confidence_matches,
        "org_fallback_matches": org_fallback_matches,
        "high_confidence_rate_pct": round(high_confidence_rate, 1),
        # Enrichment quality distribution
        "quality_high": quality_high,
        "quality_medium": quality_medium,
        "quality_low": quality_low,
        "quality_failed": quality_failed,
        "quality_high_rate_pct": round(quality_high_rate, 1),
    }

    # Log in structured format for easy parsing
    logger.info(
        f"BASELINE:enrichment_summary "
        f"stories={total_stories} "
        f"people={total_people} "
        f"with_linkedin={people_with_profile} "
        f"linkedin_match_rate={linkedin_match_rate:.1f}% "
        f"with_urn={people_with_urn} "
        f"urn_rate={urn_rate:.1f}% "
        f"zero_match_stories={stories_with_zero_linkedin} "
        f"zero_match_rate={zero_match_rate:.1f}% "
        f"high_conf={high_confidence_matches} "
        f"medium_conf={medium_confidence_matches} "
        f"low_conf={low_confidence_matches} "
        f"org_fallback={org_fallback_matches} "
        f"quality_high={quality_high} "
        f"quality_medium={quality_medium} "
        f"quality_low={quality_low} "
        f"quality_failed={quality_failed} "
        f"quality_high_rate={quality_high_rate:.1f}%"
    )

    return metrics


def _log_story_enrichment_metrics(story, context: str = "") -> None:
    """
    Log enrichment metrics for a single story.

    Call this after each story is enriched to track per-story performance.

    Args:
        story: Story object with enrichment data
        context: Optional context string (e.g., "after_profile_lookup")
    """
    direct_people = story.direct_people or []
    indirect_people = story.indirect_people or []
    all_people = direct_people + indirect_people

    total = len(all_people)
    with_linkedin = sum(1 for p in all_people if p.get("linkedin_profile"))
    with_urn = sum(1 for p in all_people if p.get("linkedin_urn"))
    match_rate = (with_linkedin / total * 100) if total > 0 else 0

    logger.info(
        f"BASELINE:story_enrichment "
        f"story_id={story.id} "
        f"context={context} "
        f"direct_people={len(direct_people)} "
        f"indirect_people={len(indirect_people)} "
        f"total_people={total} "
        f"with_linkedin={with_linkedin} "
        f"with_urn={with_urn} "
        f"match_rate={match_rate:.1f}%"
    )


def log_story_details(
    engine: ContentEngine, story_ids: list[int] | None = None
) -> None:
    """
    Log detailed information for each story including people and organizations.

    Args:
        engine: ContentEngine instance
        story_ids: Optional list of story IDs to log. If None, logs all recent stories.
    """
    import json as json_module

    with engine.db._get_connection() as conn:
        cursor = conn.cursor()

        if story_ids:
            placeholders = ",".join("?" * len(story_ids))
            cursor.execute(
                f"""SELECT id, title, summary, direct_people, indirect_people, organizations
                   FROM stories WHERE id IN ({placeholders}) ORDER BY id""",
                story_ids,
            )
        else:
            # Get most recent stories
            cursor.execute("""
                SELECT id, title, summary, direct_people, indirect_people, organizations
                FROM stories ORDER BY id DESC LIMIT 10
            """)

        rows = cursor.fetchall()

    for row in rows:
        story_id = row["id"]
        title = row["title"] or "Untitled"
        summary = row["summary"] or "No summary"
        direct_people = (
            json_module.loads(row["direct_people"]) if row["direct_people"] else []
        )
        indirect_people = (
            json_module.loads(row["indirect_people"]) if row["indirect_people"] else []
        )
        organizations = (
            json_module.loads(row["organizations"]) if row["organizations"] else []
        )

        print(f"\n{'=' * 70}")
        print(f"STORY [{story_id}]: {title}")
        print(f"{'=' * 70}")
        print(f"\nSUMMARY:\n{summary[:500]}{'...' if len(summary) > 500 else ''}")

        # Direct people (mentioned in the story)
        print(f"\nDIRECT PEOPLE ({len(direct_people)}):")
        if direct_people:
            for person in direct_people:
                name = person.get("name", "Unknown")
                title = (
                    person.get("job_title", "")
                    or person.get("title", "")
                    or person.get("position", "")
                )
                org = (
                    person.get("employer", "")
                    or person.get("company", "")
                    or person.get("organization", "")
                )
                location = person.get("location", "")
                specialty = person.get("specialty", "") or person.get("department", "")
                linkedin = person.get("linkedin_profile", "") or person.get(
                    "linkedin_urn", ""
                )
                linkedin_display = linkedin if linkedin else "❌ Not found"

                # Validation score and confidence
                match_confidence = person.get("match_confidence", "")
                validation_score = person.get("validation_score", "")
                validation_signals = person.get("validation_signals", [])

                print(f"  • {name}")
                if title:
                    print(f"      Title: {title}")
                if org:
                    print(f"      Organization: {org}")
                if location:
                    print(f"      Location: {location}")
                if specialty:
                    print(f"      Specialty: {specialty}")
                print(f"      LinkedIn: {linkedin_display}")

                # Show validation details if available
                if match_confidence:
                    confidence_emoji = {
                        "high": "🟢",
                        "validated_high": "🟢✓",
                        "medium": "🟡",
                        "validated_medium": "🟡✓",
                        "low": "🔴",
                        "org_fallback": "🏢",
                    }.get(match_confidence, "⚪")
                    print(f"      Confidence: {confidence_emoji} {match_confidence}")
                if validation_score:
                    print(
                        f"      Validation Score: {validation_score:.1f}/10"
                        if isinstance(validation_score, (int, float))
                        else f"      Validation Score: {validation_score}"
                    )
                if validation_signals:
                    signals_str = ", ".join(str(s) for s in validation_signals[:5])
                    print(f"      Signals: {signals_str}")
        else:
            print("  None")

        # Organizations extracted
        print(f"\nORGANIZATIONS ({len(organizations)}):")
        if organizations:
            for org in organizations:
                print(f"  • {org}")
        else:
            print("  None")

        # Indirect people (org leadership)
        print(f"\nINDIRECT PEOPLE ({len(indirect_people)}):")
        if indirect_people:
            for person in indirect_people:
                name = person.get("name", "Unknown")
                title = person.get("title", "") or person.get("job_title", "")
                org = (
                    person.get("organization", "")
                    or person.get("company", "")
                    or person.get("employer", "")
                )
                location = person.get("location", "")
                specialty = person.get("specialty", "") or person.get("department", "")
                role_type = person.get("role_type", "")
                linkedin = person.get("linkedin_profile", "") or person.get(
                    "linkedin_urn", ""
                )
                linkedin_display = linkedin if linkedin else "❌ Not found"

                # Validation score and confidence
                match_confidence = person.get("match_confidence", "")
                validation_score = person.get("validation_score", "")
                validation_signals = person.get("validation_signals", [])

                print(f"  • {name}")
                if title:
                    print(f"      Title: {title}")
                if org:
                    print(f"      Organization: {org}")
                if role_type:
                    print(f"      Role Type: {role_type}")
                if location:
                    print(f"      Location: {location}")
                if specialty:
                    print(f"      Specialty: {specialty}")
                print(f"      LinkedIn: {linkedin_display}")

                # Show validation details if available
                if match_confidence:
                    confidence_emoji = {
                        "high": "🟢",
                        "validated_high": "🟢✓",
                        "medium": "🟡",
                        "validated_medium": "🟡✓",
                        "low": "🔴",
                        "org_fallback": "🏢",
                    }.get(match_confidence, "⚪")
                    print(f"      Confidence: {confidence_emoji} {match_confidence}")
                if validation_score:
                    print(
                        f"      Validation Score: {validation_score:.1f}/10"
                        if isinstance(validation_score, (int, float))
                        else f"      Validation Score: {validation_score}"
                    )
                if validation_signals:
                    signals_str = ", ".join(str(s) for s in validation_signals[:5])
                    print(f"      Signals: {signals_str}")
        else:
            print("  None")

        # Log to file as well
        total_people = len(direct_people) + len(indirect_people)
        with_linkedin = sum(
            1 for p in direct_people + indirect_people if p.get("linkedin_profile")
        )
        logger.info(
            f"STORY_DETAIL story_id={story_id} "
            f'title="{title[:50]}" '
            f"direct_people={len(direct_people)} "
            f"indirect_people={len(indirect_people)} "
            f"organizations={len(organizations)} "
            f"linkedin_coverage={with_linkedin}/{total_people}"
        )


def _show_enrichment_summary(engine: ContentEngine) -> None:
    """Show comprehensive summary of enrichment status."""
    import json as json_module

    # Log enrichment metrics
    log_enrichment_baseline_metrics(engine)

    with engine.db._get_connection() as conn:
        cursor = conn.cursor()

        # Get overall stats in one query
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN (direct_people IS NOT NULL AND direct_people != '[]') THEN 1 ELSE 0 END) as with_direct_people,
                SUM(CASE WHEN (indirect_people IS NOT NULL AND indirect_people != '[]') THEN 1 ELSE 0 END) as with_indirect_people,
                SUM(CASE WHEN (organizations IS NOT NULL AND organizations != '[]') THEN 1 ELSE 0 END) as with_organizations,
                SUM(CASE WHEN promotion IS NOT NULL AND promotion != '' THEN 1 ELSE 0 END) as with_promotion,
                SUM(CASE WHEN image_path IS NOT NULL AND image_path != '' THEN 1 ELSE 0 END) as with_image,
                SUM(CASE WHEN verification_status = 'approved' THEN 1 ELSE 0 END) as approved,
                SUM(CASE WHEN enrichment_quality = 'high' THEN 1 ELSE 0 END) as quality_high,
                SUM(CASE WHEN enrichment_quality = 'medium' THEN 1 ELSE 0 END) as quality_medium,
                SUM(CASE WHEN enrichment_quality = 'low' THEN 1 ELSE 0 END) as quality_low,
                SUM(CASE WHEN enrichment_quality = 'failed' THEN 1 ELSE 0 END) as quality_failed
            FROM stories
        """)
        stats = cursor.fetchone()
        total = stats["total"] or 0
        with_direct_people = stats["with_direct_people"] or 0
        with_indirect_people = stats["with_indirect_people"] or 0
        with_organizations = stats["with_organizations"] or 0
        with_promotion = stats["with_promotion"] or 0
        with_image = stats["with_image"] or 0
        approved = stats["approved"] or 0
        quality_high = stats["quality_high"] or 0
        quality_medium = stats["quality_medium"] or 0
        quality_low = stats["quality_low"] or 0
        quality_failed = stats["quality_failed"] or 0

        # Count LinkedIn enrichment details
        cursor.execute("""
            SELECT direct_people, indirect_people FROM stories
            WHERE (direct_people IS NOT NULL AND direct_people != '[]')
               OR (indirect_people IS NOT NULL AND indirect_people != '[]')
        """)
        total_direct_people = 0
        total_indirect_people = 0
        with_profiles = 0
        with_urns = 0
        for row in cursor.fetchall():
            direct_people = (
                json_module.loads(row["direct_people"]) if row["direct_people"] else []
            )
            indirect_people = (
                json_module.loads(row["indirect_people"])
                if row["indirect_people"]
                else []
            )
            total_direct_people += len(direct_people)
            total_indirect_people += len(indirect_people)
            for p in direct_people + indirect_people:
                if p.get("linkedin_profile"):
                    with_profiles += 1
                if p.get("linkedin_urn"):
                    with_urns += 1

        total_people = total_direct_people + total_indirect_people

    # Display summary
    print(f"\nStories: {total} total")
    print(f"  • With organizations: {with_organizations}")
    print(
        f"  • With direct people: {with_direct_people} ({total_direct_people} people)"
    )
    print(
        f"  • With indirect people: {with_indirect_people} ({total_indirect_people} leaders)"
    )
    print(f"  • With promotion message: {with_promotion}")
    print(f"  • With image: {with_image}")
    print(f"  • Approved: {approved}")

    # Display enrichment quality distribution
    enriched_total = quality_high + quality_medium + quality_low + quality_failed
    if enriched_total > 0:
        print(f"\nEnrichment Quality: {enriched_total} stories assessed")
        print(
            f"  • High quality: {quality_high} ({100 * quality_high // enriched_total}%)"
        )
        print(
            f"  • Medium quality: {quality_medium} ({100 * quality_medium // enriched_total}%)"
        )
        print(
            f"  • Low quality: {quality_low} ({100 * quality_low // enriched_total}%)"
        )
        if quality_failed > 0:
            print(
                f"  • Failed: {quality_failed} ({100 * quality_failed // enriched_total}%)"
            )

    if total_people > 0:
        print(f"\nLinkedIn Enrichment: {total_people} people/leaders identified")
        print(
            f"  • With LinkedIn profiles: {with_profiles} ({100 * with_profiles // total_people}%)"
        )
        print(
            f"  • With URNs for @mentions: {with_urns} ({100 * with_urns // total_people}%)"
        )

    # Show what's missing/next steps
    missing = []
    if with_organizations > with_indirect_people:
        missing.append(
            f"{with_organizations - with_indirect_people} stories with orgs need leader lookup"
        )
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


def _show_people_discovery_summary(engine: ContentEngine) -> None:
    """Show a detailed summary of people discovered with validation confidence levels.

    Displays:
    - Total direct people (mentioned in stories)
    - Total indirect people (organization leadership)
    - LinkedIn profile match rates by confidence level
    - People ready for connection requests
    """
    import json as json_module

    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, direct_people, indirect_people FROM stories
            WHERE (direct_people IS NOT NULL AND direct_people != '[]')
               OR (indirect_people IS NOT NULL AND indirect_people != '[]')
            ORDER BY id DESC
        """)
        rows = cursor.fetchall()

    # Aggregate people data
    all_direct: list[dict] = []
    all_indirect: list[dict] = []
    confidence_counts = {
        "validated_high": 0,
        "high": 0,
        "validated_medium": 0,
        "medium": 0,
        "low": 0,
        "org_fallback": 0,
        "none": 0,
    }

    for row in rows:
        direct = json_module.loads(row["direct_people"]) if row["direct_people"] else []
        indirect = (
            json_module.loads(row["indirect_people"]) if row["indirect_people"] else []
        )
        all_direct.extend(direct)
        all_indirect.extend(indirect)

        for person in direct + indirect:
            conf = person.get("match_confidence", "none")
            if conf in confidence_counts:
                confidence_counts[conf] += 1
            else:
                confidence_counts["none"] += 1

    total_direct = len(all_direct)
    total_indirect = len(all_indirect)
    total_people = total_direct + total_indirect

    # Count profiles
    with_profile = sum(
        1 for p in all_direct + all_indirect if p.get("linkedin_profile")
    )
    with_urn = sum(1 for p in all_direct + all_indirect if p.get("linkedin_urn"))

    print(f"\n{'=' * 60}")
    print("PEOPLE DISCOVERY SUMMARY")
    print(f"{'=' * 60}")

    print(f"\nTotal People Discovered: {total_people}")
    print(f"  • Direct (mentioned in stories): {total_direct}")
    print(f"  • Indirect (organization leaders): {total_indirect}")

    print("\nLinkedIn Profile Matching:")
    print(
        f"  • With profiles: {with_profile}/{total_people} ({100 * with_profile // total_people if total_people > 0 else 0}%)"
    )
    print(
        f"  • With URNs: {with_urn}/{total_people} ({100 * with_urn // total_people if total_people > 0 else 0}%)"
    )

    print("\nMatch Confidence Distribution:")
    high_quality = confidence_counts["validated_high"] + confidence_counts["high"]
    medium_quality = confidence_counts["validated_medium"] + confidence_counts["medium"]
    low_quality = confidence_counts["low"] + confidence_counts["org_fallback"]
    no_match = confidence_counts["none"]

    if with_profile > 0:
        print(
            f"  🟢 High confidence: {high_quality} ({100 * high_quality // with_profile}%)"
        )
        print(
            f"  🟡 Medium confidence: {medium_quality} ({100 * medium_quality // with_profile}%)"
        )
        print(
            f"  🔴 Low/Fallback: {low_quality} ({100 * low_quality // with_profile}%)"
        )
    print(f"  ⚪ No profile found: {no_match}")

    # Show people awaiting connection
    people_stats = engine.db.get_people_stats()
    awaiting = people_stats.get("awaiting_connection", 0)
    pending = people_stats.get("pending_connections", 0)
    connected = people_stats.get("connected", 0)

    print("\nConnection Status:")
    print(f"  • Awaiting connection request: {awaiting}")
    print(f"  • Pending (sent, awaiting acceptance): {pending}")
    print(f"  • Connected: {connected}")

    # Show top people by confidence for quick review
    high_conf_people = [
        p
        for p in all_direct + all_indirect
        if p.get("match_confidence") in ("validated_high", "high")
        and p.get("linkedin_profile")
    ]
    if high_conf_people[:5]:
        print("\nTop High-Confidence Matches (sample):")
        for p in high_conf_people[:5]:
            name = p.get("name", "Unknown")
            org = p.get("employer") or p.get("company") or p.get("organization", "")
            url = p.get("linkedin_profile", "")[:50]
            print(f"  ✓ {name} ({org[:30]})")
            print(f"      {url}...")


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

    # Check how many people need URN extraction across direct_people and indirect_people
    pending = 0
    with engine.db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT direct_people, indirect_people FROM stories
            WHERE (direct_people LIKE '%linkedin_profile%')
               OR (indirect_people LIKE '%linkedin_profile%')
        """)
        for row in cursor.fetchall():
            direct_people = (
                json_module.loads(row["direct_people"]) if row["direct_people"] else []
            )
            indirect_people = (
                json_module.loads(row["indirect_people"])
                if row["indirect_people"]
                else []
            )
            for p in direct_people + indirect_people:
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


def _find_indirect_people(engine: ContentEngine) -> None:
    """Find indirect people (org leadership) for enriched stories."""
    print("\n--- Finding Indirect People (Org Leadership) ---")

    try:
        enriched, skipped = engine.enricher.find_indirect_people()
        print(
            f"\nResult: {enriched} stories with indirect people found, {skipped} skipped"
        )

        if enriched > 0:
            print("\nStories with indirect people:")
            with engine.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, organizations, indirect_people
                    FROM stories
                    WHERE indirect_people IS NOT NULL AND indirect_people != '[]'
                    ORDER BY id DESC LIMIT 10
                """)
                for row in cursor.fetchall():
                    import json

                    orgs = (
                        json.loads(row["organizations"]) if row["organizations"] else []
                    )
                    indirect_people = (
                        json.loads(row["indirect_people"])
                        if row["indirect_people"]
                        else []
                    )
                    print(f"\n  [{row['id']}] {row['title'][:50]}...")
                    print(f"       Organizations: {', '.join(orgs[:3])}")
                    print(f"       Indirect people found: {len(indirect_people)}")
                    for person in indirect_people[:5]:
                        name = person.get("name", "Unknown")
                        title = person.get("title", "")
                        org = person.get("organization", "")
                        print(f"         → {name} ({title}) @ {org}")

    except Exception as e:
        print(f"\nError finding indirect people: {e}")
        logger.exception("Indirect people enrichment failed")


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
        print(
            "Use option 1 (Full Pipeline) or option 16 (Search) to generate content first."
        )
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
        from web_server import run_validation

        run_validation(engine.db, port=5000)
    except ImportError as e:
        print(f"\nError: Could not import web server: {e}")
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


def _offer_connection_requests(engine: ContentEngine) -> None:
    """Offer to send LinkedIn connection requests to people found in stories using browser automation."""
    from config import Config
    from database import Person

    # Get unique people with LinkedIn profiles who haven't been connected yet
    people_needing_connection = engine.db.get_people_needing_connection()

    if not people_needing_connection:
        return  # No people needing connections, skip silently

    # Get story details for personalized messages
    story_data: dict[int, dict] = {}
    for person in people_needing_connection:
        if person.story_id and person.story_id not in story_data:
            story = engine.db.get_story(person.story_id)
            if story:
                story_data[person.story_id] = {
                    "title": story.title,
                    "summary": story.summary or "",
                    "category": story.category or "",
                }

    print("\n" + "=" * 60)
    print("LinkedIn Connection Requests")
    print("=" * 60)
    print(
        f"Found {len(people_needing_connection)} people with LinkedIn profiles awaiting connection."
    )

    # Show queue stats
    queue_stats = engine.db.get_queue_stats()
    if queue_stats["pending"] > 0:
        print(f"({queue_stats['pending']} already in queue)")

    print("\nOptions:")
    print("  1 - Send connection requests NOW (browser automation)")
    print("  2 - Add to queue for batch sending later")
    print("  3 - Skip for now")

    choice = input("\nYour choice (1/2/3): ").strip()

    if choice == "3":
        print("Skipped connection requests.")
        return

    print("\nPeople with LinkedIn profiles:\n")
    for i, person in enumerate(people_needing_connection, 1):
        org_info = f" ({person.organization})" if person.organization else ""
        story_info = story_data.get(person.story_id or 0, {})
        story_title = story_info.get("title", "")
        story_ref = f" [Story: {story_title[:40]}...]" if story_title else ""
        rel_type = "direct" if person.relationship_type == "direct" else "indirect"
        print(f"  {i}. {person.name}{org_info} [{rel_type}]{story_ref}")
        print(f"      {person.linkedin_profile}")

    print("\nSelection options:")
    print("  a - ALL people")
    print("  1,2,3 - Specific people (comma-separated numbers)")
    print("  q - Cancel")

    selection = input("\nYour choice: ").strip().lower()

    if selection == "q":
        print("Cancelled.")
        return

    selected_people: list[Person] = []

    if selection == "a":
        selected_people = people_needing_connection
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            for idx in indices:
                if 0 <= idx < len(people_needing_connection):
                    selected_people.append(people_needing_connection[idx])
        except ValueError:
            print("Invalid selection.")
            return

    if not selected_people:
        print("No valid people selected.")
        return

    # Handle based on choice (1 = send now, 2 = queue)
    if choice == "2":
        # Queue for later
        queued_count = 0
        for person in selected_people:
            story_info = story_data.get(person.story_id or 0, {})
            message = _generate_connection_message(
                discipline=Config.DISCIPLINE,
                person_name=person.name or "",
                story_title=story_info.get("title", ""),
                story_summary=story_info.get("summary", ""),
                person_title=person.title or "",
                person_org=person.organization or "",
            )

            queue_id = engine.db.queue_connection_request(
                person_id=person.id,
                linkedin_profile=person.linkedin_profile or "",
                name=person.name or "",
                title=person.title or "",
                organization=person.organization or "",
                story_id=person.story_id,
                story_title=story_info.get("title", ""),
                story_summary=story_info.get("summary", "")[:500],
                message=message,
                priority=3 if person.relationship_type == "direct" else 5,
            )
            if queue_id:
                queued_count += 1
                print(f"  ✓ Queued: {person.name}")
            else:
                print(f"  ⊘ Already queued: {person.name}")

        print(f"\nQueued {queued_count} connection requests for later processing.")
        print("Use menu option 40 to process the queue.")
        return

    # choice == "1" - send now (original behavior)
    print(f"\nWill send connection requests to {len(selected_people)} people:")
    for p in selected_people:
        print(f"  - {p.name} ({p.linkedin_profile})")

    print("\nNote: A browser window will open. You may need to log in to LinkedIn.")
    print("The browser will visit each profile and click the Connect button.")
    print("There will be a 5-8 second delay between each connection request.")
    print(f"\nConnection message will reference your {Config.DISCIPLINE} network.")
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    # Use browser automation for connections
    from linkedin_profile_lookup import LinkedInCompanyLookup

    lookup = LinkedInCompanyLookup()

    print("\nSending connection requests via browser...")
    print("(Watch the browser window for progress)\n")

    success_count = 0
    failed_count = 0
    errors = []

    try:
        for i, person in enumerate(selected_people):
            profile_url = person.linkedin_profile or ""
            person_name = person.name or ""
            story_info = story_data.get(person.story_id or 0, {})

            # Generate personalized connection message with enhanced context
            message = _generate_connection_message(
                discipline=Config.DISCIPLINE,
                person_name=person_name,
                story_title=story_info.get("title", ""),
                story_summary=story_info.get("summary", ""),
                person_title=person.title or "",
                person_org=person.organization or "",
            )

            print(f"[{i + 1}/{len(selected_people)}] Connecting to {person_name}...")
            print(f"    Message: {message[:60]}...")

            success, result_msg = lookup.send_connection_via_browser(
                profile_url, message=message
            )

            if success:
                success_count += 1
                # Update database with connection status
                engine.db.mark_connection_sent(person.id or 0, message, "pending")
                print(f"    ✓ {result_msg}")
            else:
                failed_count += 1
                # Mark as failed in database
                engine.db.mark_connection_sent(person.id or 0, message, "failed")
                errors.append(f"{person_name}: {result_msg}")
                print(f"    ✗ {result_msg}")

            # Delay between requests
            if i < len(selected_people) - 1:
                import time
                import random

                delay = 5.0 + random.random() * 3
                time.sleep(delay)

    finally:
        # Don't close browser - let user see the results
        pass

    print("\n--- Connection Request Results ---")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failed_count}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")

    print("\nNote: Browser window left open. Close it manually when done.")


def _generate_connection_message(
    discipline: str,
    person_name: str,
    story_title: str,
    story_summary: str = "",
    person_title: str = "",
    person_org: str = "",
    person_specialty: str = "",
) -> str:
    """Generate a personalized connection request message with enhanced context.

    Creates compelling, context-aware connection messages that reference:
    - The story/topic that led to finding this person
    - Their professional role and organization
    - Their specialty/field of expertise
    - Your professional discipline

    Args:
        discipline: Professional discipline (e.g., "chemical engineer")
        person_name: Name of the person being connected to
        story_title: Title of the story they're associated with
        story_summary: Summary of the story for context
        person_title: Person's job title
        person_org: Person's organization
        person_specialty: Person's specialty/field (e.g., "catalysis", "process safety")

    Returns:
        Connection message (max 300 chars for LinkedIn)
    """
    # Extract first name if available
    first_name = person_name.split()[0] if person_name else ""
    greeting = f"Hi {first_name}," if first_name else "Hi,"

    # Extract topic from story or use specialty
    topic = _extract_topic_from_story(story_title, story_summary)
    specialty_topic = person_specialty.lower() if person_specialty else ""

    # Use specialty if we have it and no topic from story
    if not topic and specialty_topic:
        topic = specialty_topic

    # Build personalized message based on available context
    message = ""

    if topic and person_title:
        # Best case: we have topic and role
        message = (
            f"{greeting} Your work on {topic} caught my attention. "
            f"As a {discipline}, I'd love to connect with professionals in this space."
        )
    elif topic and person_org:
        # Have topic and org
        message = (
            f"{greeting} I came across your work on {topic} at {person_org[:25]}. "
            f"I'm building my {discipline} network and would value connecting."
        )
    elif story_title and person_title:
        # Have story title and role
        short_title = story_title[:50] + "..." if len(story_title) > 50 else story_title
        message = (
            f'{greeting} Your involvement in "{short_title}" caught my eye. '
            f"I'm expanding my {discipline} network and would appreciate connecting."
        )
    elif story_title:
        # Have story but no title
        short_title = story_title[:60] + "..." if len(story_title) > 60 else story_title
        message = (
            f'{greeting} I noticed your work related to "{short_title}". '
            f"I'm building my {discipline} network and would be grateful to connect."
        )
    elif person_org:
        # No story but have organization
        message = (
            f"{greeting} I see you're with {person_org[:30]}. "
            f"I'm expanding my professional {discipline} network and would appreciate connecting."
        )
    else:
        # Minimal context fallback
        message = (
            f"{greeting} I'm building my professional {discipline} network "
            f"and would be grateful if you'd connect."
        )

    # Ensure we don't exceed LinkedIn's 300 char limit
    if len(message) > 300:
        message = message[:297] + "..."

    return message


def _extract_topic_from_story(title: str, summary: str) -> str:
    """Extract a short, compelling topic phrase from story title/summary.

    Looks for domain-specific terminology that indicates what the story
    is about, suitable for use in connection messages.

    Args:
        title: Story title
        summary: Story summary

    Returns:
        Short topic phrase (e.g., "carbon capture", "hydrogen technology") or empty string
    """
    import re

    # Combine and look for common patterns
    text = f"{title} {summary}"[:500]

    # Technology/project patterns - ordered by specificity
    # More specific patterns first (e.g., "green hydrogen" before "hydrogen")
    patterns = [
        # Energy & Sustainability
        (r"\b(carbon capture and storage|CCS|CCUS)\b", "carbon capture"),
        (r"\b(green hydrogen|blue hydrogen)\b", lambda m: m.group(0).lower()),
        (r"\b(hydrogen production|H2 production)\b", "hydrogen technology"),
        (r"\b(hydrogen|H2)\b", "hydrogen"),
        (r"\b(electrolysis|electrolyzer|electrolyser)\b", "electrolysis technology"),
        (r"\b(net zero|net-zero|carbon neutral)\b", "net-zero initiatives"),
        (r"\b(sustainability|decarbonization|decarbonisation)\b", "sustainability"),
        (r"\b(renewable energy|solar|wind power)\b", "renewable energy"),
        (r"\b(biofuels|biodiesel|bioethanol)\b", "biofuels"),
        (r"\b(circular economy|recycling)\b", "circular economy"),
        # Chemical/Process Engineering
        (r"\b(catalyst|catalysis|catalytic)\b", "catalysis"),
        (r"\b(process intensification)\b", "process intensification"),
        (r"\b(process safety|safety engineering)\b", "process safety"),
        (r"\b(reactor design|chemical reactor)\b", "reactor technology"),
        (
            r"\b(separation process|distillation|crystallization)\b",
            "separation technology",
        ),
        (r"\b(polymer|polyethylene|polypropylene|plastics)\b", "polymers"),
        (r"\b(petrochemical|refinery|refining)\b", "petrochemicals"),
        (r"\b(LNG|natural gas|methane)\b", "natural gas"),
        (r"\b(ammonia|fertilizer|fertiliser)\b", "ammonia technology"),
        # Digital/Innovation
        (r"\b(digital twin|digital transformation)\b", "digital transformation"),
        (r"\b(artificial intelligence|AI|machine learning)\b", "AI applications"),
        (r"\b(process automation|industrial automation)\b", "automation"),
        (r"\b(Industry 4\.0|smart manufacturing)\b", "Industry 4.0"),
        # Energy Storage
        (r"\b(battery technology|energy storage)\b", "energy storage"),
        (r"\b(lithium-ion|lithium ion)\b", "battery technology"),
        # Biotech
        (r"\b(biotech|biotechnology|fermentation)\b", "biotechnology"),
        (r"\b(biochemical|biochemistry)\b", "biochemistry"),
        # Water
        (r"\b(water treatment|wastewater|desalination)\b", "water technology"),
        # Materials
        (
            r"\b(materials science|advanced materials|nanomaterials)\b",
            "advanced materials",
        ),
    ]

    for pattern, replacement in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if callable(replacement):
                return str(replacement(match))
            return str(replacement)

    # No pattern match - return empty
    return ""


def _process_connection_queue(engine: ContentEngine) -> None:
    """Process queued connection requests via browser automation."""
    import time
    import random

    print("\n--- Process Connection Queue ---")

    # Get queue stats
    stats = engine.db.get_queue_stats()
    print("\nQueue Statistics:")
    print(f"  Pending: {stats['pending']}")
    print(f"  Sent: {stats['sent']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total: {stats['total_queued']}")

    if stats["pending"] == 0:
        print("\nNo pending connection requests in queue.")
        return

    if stats.get("by_priority"):
        print("\n  By Priority (1=highest):")
        for priority, count in sorted(stats["by_priority"].items()):
            print(f"    Priority {priority}: {count}")

    # Get queued connections
    print("\nOptions:")
    print("  1 - Process all pending requests")
    print("  2 - Process next N requests")
    print("  3 - View queue contents")
    print("  4 - Clear failed entries")
    print("  5 - Clear entire queue")
    print("  q - Cancel")

    choice = input("\nYour choice: ").strip().lower()

    if choice == "q":
        print("Cancelled.")
        return

    if choice == "3":
        # View queue contents
        queued = engine.db.get_queued_connections(limit=50, status="all")
        print(f"\n{'Name':<30} {'Org':<25} {'Status':<10} {'Priority'}")
        print("-" * 80)
        for entry in queued:
            status_icon = {"queued": "⏳", "sent": "✓", "failed": "✗"}.get(
                entry["status"], "?"
            )
            print(
                f"{entry['name'][:29]:<30} {(entry['organization'] or '')[:24]:<25} "
                f"{status_icon} {entry['status']:<8} {entry['priority']}"
            )
        return

    if choice == "4":
        # Clear failed entries
        cleared = engine.db.clear_queue(status="failed")
        print(f"\nCleared {cleared} failed entries from queue.")
        return

    if choice == "5":
        # Clear entire queue
        confirm = input("Are you sure you want to clear the ENTIRE queue? (yes/no): ")
        if confirm.lower() == "yes":
            cleared = engine.db.clear_queue()
            print(f"\nCleared {cleared} entries from queue.")
        else:
            print("Cancelled.")
        return

    # Process requests
    limit = stats["pending"]
    if choice == "2":
        try:
            limit = int(
                input(f"How many to process (max {stats['pending']}): ").strip()
            )
            limit = min(limit, stats["pending"])
        except ValueError:
            print("Invalid number.")
            return

    queued = engine.db.get_queued_connections(limit=limit, status="queued")

    if not queued:
        print("No pending requests to process.")
        return

    print(f"\nWill process {len(queued)} connection requests:")
    for entry in queued[:10]:
        print(f"  - {entry['name']} ({entry['organization'] or 'Unknown org'})")
    if len(queued) > 10:
        print(f"  ... and {len(queued) - 10} more")

    print("\nNote: A browser window will open. You may need to log in to LinkedIn.")
    print("There will be a 5-8 second delay between each request.")
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    # Use browser automation
    from linkedin_profile_lookup import LinkedInCompanyLookup

    lookup = LinkedInCompanyLookup()

    print("\nProcessing connection queue...")
    print("(Watch the browser window for progress)\n")

    success_count = 0
    failed_count = 0
    errors = []

    try:
        for i, entry in enumerate(queued):
            profile_url = entry["linkedin_profile"]
            person_name = entry["name"]
            message = entry["message"]

            print(f"[{i + 1}/{len(queued)}] Connecting to {person_name}...")

            success, result_msg = lookup.send_connection_via_browser(
                profile_url, message=message
            )

            if success:
                success_count += 1
                engine.db.update_queue_status(entry["id"], "sent")
                # Also update person record if linked
                if entry.get("person_id"):
                    engine.db.mark_connection_sent(
                        entry["person_id"], message, "pending"
                    )
                print(f"    ✓ {result_msg}")
            else:
                failed_count += 1
                engine.db.update_queue_status(entry["id"], "failed", result_msg)
                errors.append(f"{person_name}: {result_msg}")
                print(f"    ✗ {result_msg}")

            # Delay between requests (rate limiting)
            if i < len(queued) - 1:
                delay = 5.0 + random.random() * 3
                time.sleep(delay)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")

    print("\n--- Queue Processing Results ---")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Remaining in queue: {stats['pending'] - success_count}")

    if errors:
        print("\nErrors:")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")


def _view_connection_dashboard(engine: ContentEngine) -> None:
    """View connection tracking dashboard with stats and history."""
    print("\n" + "=" * 70)
    print("CONNECTION TRACKING DASHBOARD")
    print("=" * 70)

    # Get acceptance rate stats
    acceptance = engine.db.get_connection_acceptance_rate(days=30)
    queue_stats = engine.db.get_queue_stats()
    people_stats = engine.db.get_people_stats()

    # Summary section
    print("\n📊 SUMMARY (Last 30 Days)")
    print("-" * 40)
    print(f"  Connection Requests Sent: {acceptance['total_sent']}")
    print(f"  Accepted (Connected):     {acceptance['accepted']}")
    print(f"  Still Pending:            {acceptance['pending']}")
    print(f"  Failed to Send:           {acceptance['failed']}")
    if acceptance["total_sent"] > 0:
        rate = acceptance["acceptance_rate"] * 100
        print(f"\n  Acceptance Rate: {rate:.1f}%")
        # Visual bar
        bar_filled = int(rate / 5)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"  [{bar}]")

    # Queue section
    print("\n📋 CONNECTION QUEUE")
    print("-" * 40)
    print(f"  Pending to Send: {queue_stats['pending']}")
    print(f"  Already Sent:    {queue_stats['sent']}")
    print(f"  Failed:          {queue_stats['failed']}")

    if queue_stats.get("by_priority"):
        print("\n  By Priority:")
        for priority, count in sorted(queue_stats["by_priority"].items()):
            priority_label = {1: "High", 3: "Direct", 5: "Indirect", 7: "Low"}.get(
                priority, f"P{priority}"
            )
            print(f"    {priority_label}: {count}")

    # People database section
    print("\n👥 PEOPLE DATABASE")
    print("-" * 40)
    print(f"  Total People Extracted: {people_stats.get('total', 0)}")
    print(f"  With LinkedIn Profiles: {people_stats.get('unique_profiles', 0)}")
    print(f"  Direct Mentions:        {people_stats.get('direct_count', 0)}")
    print(f"  Indirect (Leadership):  {people_stats.get('indirect_count', 0)}")
    print(f"  Connections Sent:       {people_stats.get('connections_sent', 0)}")

    # Recent activity
    print("\n📜 RECENT CONNECTION ACTIVITY")
    print("-" * 40)
    history = engine.db.get_connection_history(days=30, limit=15)

    if history:
        print(f"  {'Name':<25} {'Status':<12} {'Sent Date'}")
        print("  " + "-" * 60)
        for record in history:
            name = (record["name"] or "Unknown")[:24]
            status = record["connection_status"] or "unknown"
            status_icon = {"pending": "⏳", "connected": "✅", "failed": "❌"}.get(
                status, "?"
            )
            sent_date = record.get("connection_sent_at", "")
            if sent_date:
                # Format date nicely
                sent_date = str(sent_date)[:10]
            print(f"  {name:<25} {status_icon} {status:<10} {sent_date}")
    else:
        print("  No recent connection activity.")

    # Action menu
    print("\n" + "-" * 40)
    print("Actions:")
    print("  1 - Export connection history to CSV")
    print("  2 - View detailed history")
    print("  q - Return to menu")

    choice = input("\nYour choice: ").strip().lower()

    if choice == "1":
        _export_connection_history(engine)
    elif choice == "2":
        _view_detailed_connection_history(engine)


def _export_connection_history(engine: ContentEngine) -> None:
    """Export connection history to CSV file."""
    import csv
    from datetime import datetime

    history = engine.db.get_connection_history(days=365, limit=1000)

    if not history:
        print("No connection history to export.")
        return

    filename = f"connection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    try:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "title",
                    "organization",
                    "linkedin_profile",
                    "connection_status",
                    "connection_sent_at",
                    "story_title",
                ],
            )
            writer.writeheader()
            writer.writerows(history)

        print(f"\n✓ Exported {len(history)} records to {filename}")
    except Exception as e:
        print(f"\n✗ Export failed: {e}")


def _view_detailed_connection_history(engine: ContentEngine) -> None:
    """View detailed connection history with messages."""
    history = engine.db.get_connection_history(days=90, limit=50)

    if not history:
        print("\nNo connection history found.")
        return

    print("\n--- Detailed Connection History (Last 90 Days) ---\n")

    for i, record in enumerate(history, 1):
        status_icon = {
            "pending": "⏳",
            "connected": "✅",
            "failed": "❌",
        }.get(record["connection_status"], "?")

        print(f"{i}. {record['name']}")
        print(f"   Title: {record.get('title') or 'N/A'}")
        print(f"   Org: {record.get('organization') or 'N/A'}")
        print(f"   Status: {status_icon} {record['connection_status']}")
        print(f"   Sent: {record.get('connection_sent_at', 'N/A')}")
        print(f"   Story: {(record.get('story_title') or 'N/A')[:60]}")
        if record.get("connection_message"):
            print(f"   Message: {record['connection_message'][:80]}...")
        print()


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

    # Run pre-publish validation
    print("\n--- Pre-Publish Validation ---")
    validation = engine.publisher.validate_before_publish(story)

    # Show validation results
    if validation.author_verified:
        print(f"✓ Author verified: {validation.author_name}")
        print(f"  URN: {validation.author_urn_from_api}")
    elif validation.author_name:
        print(f"⚠ Author: {validation.author_name} (not fully verified)")
    else:
        print("✗ Could not verify author account")

    # Show mention validation
    if validation.mention_validations:
        valid_mentions = sum(
            1 for m in validation.mention_validations if m.get("urn_valid")
        )
        total_mentions = len(validation.mention_validations)
        print(f"\n@Mentions: {valid_mentions}/{total_mentions} have valid URNs")

        for mv in validation.mention_validations:
            name = mv.get("name", "Unknown")
            urn_valid = "✓" if mv.get("urn_valid") else "✗"
            urn = mv.get("urn", "No URN")
            confidence = mv.get("confidence", "unknown")
            print(f"  {urn_valid} {name}")
            print(f"      URN: {urn}")
            print(f"      Confidence: {confidence}")
            if mv.get("issues"):
                for issue in mv["issues"]:
                    print(f"      ⚠ {issue}")

    # Show warnings
    if validation.warnings:
        print(f"\n⚠ Warnings ({len(validation.warnings)}):")
        for warning in validation.warnings:
            print(f"  • {warning}")

    # Show errors (will block publishing)
    if validation.errors:
        print(f"\n❌ Errors ({len(validation.errors)}):")
        for error in validation.errors:
            print(f"  • {error}")

    if not validation.is_valid:
        print("\n✗ Pre-publish validation FAILED. Cannot publish.")
        return

    print("\n✓ Pre-publish validation PASSED")

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

    # Publish immediately (skip_validation since we just validated)
    print("\n[1/3] Publishing to LinkedIn...")
    try:
        post_id = engine.publisher.publish_immediately(story, skip_validation=True)
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
    _list_stories(engine, "all")


def _list_pending_stories(engine: ContentEngine) -> None:
    """List stories pending verification."""
    _list_stories(engine, "pending")


def _list_scheduled_stories(engine: ContentEngine) -> None:
    """List scheduled stories."""
    print("\n--- Scheduled Stories ---")
    print(engine.scheduler.get_schedule_summary())


def _list_human_approved_stories(engine: ContentEngine) -> None:
    """List human-approved stories that haven't been published yet."""
    _list_stories(engine, "approved")


def _list_stories(
    engine: ContentEngine,
    filter_type: str = "all",
    limit: int = 20,
) -> list:
    """Unified story listing function.

    Consolidates common logic for listing stories with different filters.

    Args:
        engine: ContentEngine instance
        filter_type: One of "all", "pending", "approved", "rejected", "published"
        limit: Maximum number of stories to show

    Returns:
        List of Story objects shown
    """
    from database import Story

    stories: list[Story] = []
    rows: list = []
    use_rows = False

    # Get stories based on filter type
    if filter_type == "all":
        title = "All Stories"
        use_rows = True
        with engine.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, quality_score, verification_status, verification_reason,
                       publish_status, acquire_date, image_path, promotion
                FROM stories ORDER BY acquire_date DESC LIMIT ?
            """,
                (limit,),
            )
            rows = list(cursor.fetchall())
    elif filter_type == "pending":
        title = "Stories Pending Verification"
        stories = engine.db.get_stories_needing_verification()
    elif filter_type == "approved":
        title = "Human Approved Stories (Not Yet Published)"
        stories = engine.db.get_approved_unpublished_stories(limit=limit)
    elif filter_type == "rejected":
        title = "Rejected Stories"
        stories = engine.db.get_rejected_stories()
    elif filter_type == "published":
        title = "Published Stories"
        stories = engine.db.get_published_stories()
    else:
        print(f"Unknown filter type: {filter_type}")
        return []

    print(f"\n--- {title} ---")

    # Handle empty results
    if use_rows and not rows:
        print("No stories found.")
        return []
    if not use_rows and not stories:
        print("No stories found.")
        if filter_type == "approved":
            print(
                "\nTo approve stories for publication, use Action 22 (Human Validation)."
            )
        return []

    # Display stories
    if use_rows:
        # Raw row display (for "all" query)
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
        if len(rows) >= limit:
            print(f"\n(Showing first {limit} stories)")
        return rows

    # Story object display
    result_list: list[Story] = []
    for story in stories[:limit]:
        result_list.append(story)
        has_image = "Yes" if story.image_path else "No"
        has_promotion = "Yes" if story.promotion else "No"
        print(f"\n[{story.id}] {story.title}")
        print(f"    Score: {story.quality_score} | Image: {has_image}")

        if filter_type == "approved":
            scheduled = (
                story.scheduled_time.strftime("%Y-%m-%d %H:%M")
                if story.scheduled_time
                else "Not scheduled"
            )
            print(f"    Promo: {has_promotion} | Scheduled: {scheduled}")
        elif filter_type == "rejected":
            print(f"    Reason: {story.verification_reason or 'Unknown'}")

    if len(stories) > limit:
        print(f"\n(Showing first {limit} of {len(stories)} stories)")
    return result_list


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
            f"{story.linkedin_shares or 0:>8} | {last_updated:!<16}"
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


def _launch_dashboard(engine: ContentEngine) -> None:
    """Launch the web-based dashboard for real-time monitoring."""
    print("\n--- Launch Dashboard ---")
    print("Starting the web dashboard for real-time pipeline monitoring.")

    try:
        from web_server import run_dashboard

        run_dashboard(engine.db, port=5000)
    except ImportError as e:
        print(f"\nError: Could not import web server: {e}")
        print("Make sure Flask is installed: pip install flask")
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Dashboard launch failed")


def _view_ab_tests(engine: ContentEngine) -> None:
    """View and manage A/B tests."""
    print("\n--- A/B Testing ---")

    try:
        from ab_testing import ABTestManager

        manager = ABTestManager()
        all_tests = manager.get_all_tests()

        if not all_tests:
            print("\nNo A/B tests found.")
            print(
                "\nA/B tests are automatically created when content variants are generated."
            )
            print("Tests track engagement metrics for different content approaches.")
            return

        active_tests = [t for t in all_tests if t.is_active]
        completed_tests = [t for t in all_tests if not t.is_active]

        print(f"\nTotal tests: {len(all_tests)}")
        print(f"  Active: {len(active_tests)}")
        print(f"  Completed: {len(completed_tests)}")

        if active_tests:
            print("\n--- Active Tests ---")
            for test in active_tests[:5]:
                print(f"\n[{test.id}] {test.name}")
                print(f"    Type: {test.variant_type.value}")
                print(f"    Created: {test.created_at.strftime('%Y-%m-%d')}")
                print(f"    Variants: {len(test.variants)}")

                # Get summary if available
                summary = manager.get_test_summary(test.id)
                if summary:
                    print(f"    Samples: {summary.total_assignments}")
                    if summary.recommendation:
                        print(f"    Recommendation: {summary.recommendation}")

        if completed_tests:
            print("\n--- Recently Completed Tests ---")
            for test in completed_tests[:3]:
                print(f"\n[{test.id}] {test.name}")
                print(f"    Winner: {test.winner_id or 'Not declared'}")

    except ImportError as e:
        print(f"\nError: Could not import ab_testing module: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("A/B test view failed")


def _analyze_post_optimization(engine: ContentEngine) -> None:
    """Analyze and optimize a story's LinkedIn post content."""
    print("\n--- Post Optimization Analysis ---")

    try:
        from linkedin_optimizer import LinkedInOptimizer

        optimizer = LinkedInOptimizer()

        # Get a story to analyze
        stories = engine.db.get_approved_unpublished_stories(limit=10)
        if not stories:
            stories = engine.db.get_published_stories()[:10]

        if not stories:
            print("\nNo stories available to analyze.")
            return

        print("\nSelect a story to analyze:")
        for i, story in enumerate(stories[:10], 1):
            title = (story.title or "Untitled")[:50]
            print(f"  {i}. [{story.id}] {title}")

        choice = input("\nEnter number (or 'q' to cancel): ").strip()
        if choice.lower() == "q" or not choice:
            print("Cancelled.")
            return

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(stories):
                print("Invalid selection.")
                return
            story = stories[idx]
        except ValueError:
            print("Invalid input.")
            return

        # Build the post content as it would appear on LinkedIn
        content = f"{story.summary or ''}\n\n{story.promotion or ''}"

        print(f"\n--- Analyzing: {story.title} ---\n")

        # Get analysis
        analysis = optimizer.analyze_post(content)

        print(f"Readability Score: {analysis.readability_score:.1%}")
        print(f"Has Strong Hook: {'Yes' if analysis.has_hook else 'No'}")
        print(f"Has Call-to-Action: {'Yes' if analysis.has_cta else 'No'}")
        print(
            f"Post Length: {analysis.optimized_length} chars ({analysis.paragraph_count} paragraphs)"
        )

        if analysis.warnings:
            print("\nWarnings:")
            for warning in analysis.warnings:
                print(f"  ⚠ {warning}")

        if analysis.suggestions:
            print("\nSuggestions:")
            for suggestion in analysis.suggestions:
                print(f"  → {suggestion}")

        # Show optimization summary
        print("\n" + optimizer.get_optimization_summary(content))

    except ImportError as e:
        print(f"\nError: Could not import linkedin_optimizer module: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Post optimization analysis failed")


def _check_story_originality(engine: ContentEngine) -> None:
    """Check originality of stories against their sources."""
    print("\n--- Story Originality Check ---")

    try:
        from originality_checker import OriginalityChecker

        checker = OriginalityChecker(
            client=engine.genai_client,
            local_client=engine.local_client,
        )

        # Get stories to check
        with engine.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, summary, source_links
                FROM stories
                WHERE summary IS NOT NULL
                ORDER BY id DESC LIMIT 10
            """)
            rows = cursor.fetchall()

        if not rows:
            print("\nNo stories with summaries to check.")
            return

        print("\nSelect a story to check originality:")
        for i, row in enumerate(rows, 1):
            title = (row["title"] or "Untitled")[:50]
            print(f"  {i}. [{row['id']}] {title}")

        print(f"  a. Check all {len(rows)} stories")

        choice = (
            input("\nEnter number, 'a' for all, or 'q' to cancel: ").strip().lower()
        )
        if choice == "q" or not choice:
            print("Cancelled.")
            return

        stories_to_check = []
        if choice == "a":
            stories_to_check = rows
        else:
            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(rows):
                    print("Invalid selection.")
                    return
                stories_to_check = [rows[idx]]
            except ValueError:
                print("Invalid input.")
                return

        for row in stories_to_check:
            story = engine.db.get_story(row["id"])
            if not story:
                continue

            print(f"\n--- Checking: [{story.id}] {story.title[:50]} ---")

            result = checker.check_story_originality(story)

            print(f"  Originality Score: {1 - result.similarity_score:.0%}")
            print(f"  Is Original: {'Yes' if result.is_original else 'No'}")
            print(f"  N-gram Overlap: {result.ngram_overlap_score:.0%}")

            if result.flagged_phrases:
                print("  Flagged Phrases:")
                for phrase in result.flagged_phrases[:5]:
                    print(f"    ⚠ {phrase}")

            print(f"  Recommendation: {result.recommendation}")

    except ImportError as e:
        print(f"\nError: Could not import originality_checker module: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Originality check failed")


def _view_source_credibility(engine: ContentEngine) -> None:
    """View source credibility analysis for stories."""
    print("\n--- Source Credibility Analysis ---")

    try:
        from source_verifier import SourceVerifier

        verifier = SourceVerifier()

        # Get stories with sources
        with engine.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, source_links
                FROM stories
                WHERE source_links IS NOT NULL AND source_links != '[]'
                ORDER BY id DESC LIMIT 10
            """)
            rows = cursor.fetchall()

        if not rows:
            print("\nNo stories with sources to analyze.")
            return

        print("\nSelect a story to analyze sources:")
        for i, row in enumerate(rows, 1):
            title = (row["title"] or "Untitled")[:50]
            print(f"  {i}. [{row['id']}] {title}")

        print(f"  a. Analyze all {len(rows)} stories")

        choice = (
            input("\nEnter number, 'a' for all, or 'q' to cancel: ").strip().lower()
        )
        if choice == "q" or not choice:
            print("Cancelled.")
            return

        stories_to_check = []
        if choice == "a":
            stories_to_check = rows
        else:
            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(rows):
                    print("Invalid selection.")
                    return
                stories_to_check = [rows[idx]]
            except ValueError:
                print("Invalid input.")
                return

        for row in stories_to_check:
            story = engine.db.get_story(row["id"])
            if not story:
                continue

            print(f"\n--- Sources for: [{story.id}] {story.title[:50]} ---")

            result = verifier.verify_story_sources(story)

            print(f"  Average Credibility: {result.average_credibility:.0%}")
            print(f"  Sources Checked: {len(result.source_results)}")
            print(
                f"  Has Primary Source: {'Yes' if result.has_primary_source else 'No'}"
            )

            for src_result in result.source_results:
                credibility_icon = (
                    "✓"
                    if src_result.credibility_score >= 0.7
                    else "⚠"
                    if src_result.credibility_score >= 0.4
                    else "✗"
                )
                tier_str = (
                    f"Tier {src_result.tier}" if src_result.tier > 0 else "Unknown"
                )
                print(f"\n  {credibility_icon} {src_result.domain}")
                print(f"      Score: {src_result.credibility_score:.0%} | {tier_str}")
                if src_result.notes:
                    print(f"      Notes: {', '.join(src_result.notes)}")

            # Show summary
            print(f"\n  Summary: {verifier.get_source_summary(story)}")

    except ImportError as e:
        print(f"\nError: Could not import source_verifier module: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Source credibility analysis failed")


def _analyze_trends(engine: ContentEngine) -> None:
    """Analyze trending topics and content freshness."""
    print("\n--- Trend & Freshness Analysis ---")

    try:
        from trend_detector import TrendDetector

        detector = TrendDetector()

        # Get recent stories
        with engine.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, summary, acquire_date
                FROM stories
                WHERE summary IS NOT NULL
                ORDER BY acquire_date DESC LIMIT 15
            """)
            rows = cursor.fetchall()

        if not rows:
            print("\nNo stories to analyze.")
            return

        print(f"\nAnalyzing {len(rows)} recent stories for trends...\n")

        trending_stories = []
        fresh_stories = []

        for row in rows:
            story = engine.db.get_story(row["id"])
            if not story or not story.summary:
                continue

            content = f"{story.title or ''} {story.summary or ''}"

            # Check for trending topics
            trending_topics = detector.detect_trending_topics(content)
            boost, matched_topics = detector.calculate_trending_boost(content)

            # Calculate freshness (returns float, not FreshnessScore)
            freshness_score = detector.calculate_freshness_score(story.acquire_date)

            title_short = (story.title or "Untitled")[:45]

            if trending_topics:
                trending_stories.append((story, trending_topics, boost))
            if freshness_score >= 0.7:
                fresh_stories.append((story, freshness_score))

            # Show each story's analysis
            trend_icon = "🔥" if boost > 0.1 else "  "
            fresh_icon = "⚡" if freshness_score >= 0.7 else "  "
            print(f"[{story.id}] {title_short}...")
            print(
                f"    {trend_icon} Trend boost: +{boost:.0%} | {fresh_icon} Freshness: {freshness_score:.0%}"
            )
            if matched_topics:
                print(f"       Topics: {', '.join(matched_topics[:3])}")

        # Summary
        print("\n" + "=" * 60)
        print(f"Trending Stories: {len(trending_stories)}")
        print(f"Fresh Stories (>70%): {len(fresh_stories)}")

        if trending_stories:
            print("\nTop Trending:")
            for story, topics, boost in sorted(
                trending_stories, key=lambda x: x[2], reverse=True
            )[:3]:
                print(f"  🔥 [{story.id}] {story.title[:40]}... (+{boost:.0%})")

    except ImportError as e:
        print(f"\nError: Could not import trend_detector module: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Trend analysis failed")


def _analyze_intent_classification(engine: ContentEngine) -> None:
    """Analyze story intent and career alignment."""
    print("\n--- Intent Classification Analysis ---")

    try:
        from intent_classifier import IntentClassifier

        classifier = IntentClassifier()

        # Get recent stories
        with engine.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, summary
                FROM stories
                WHERE summary IS NOT NULL
                ORDER BY id DESC LIMIT 15
            """)
            rows = cursor.fetchall()

        if not rows:
            print("\nNo stories to classify.")
            return

        print(f"\nClassifying {len(rows)} stories...\n")

        intent_counts: dict[str, int] = {}
        career_aligned = 0
        results = []

        for row in rows:
            title = row["title"] or ""
            summary = row["summary"] or ""

            result = classifier.classify(title, summary)
            results.append((row, result))

            # Track intents
            for intent in result.top_intents:
                intent_name = intent.value
                intent_counts[intent_name] = intent_counts.get(intent_name, 0) + 1

            # Track career alignment
            if classifier.is_career_aligned(title, summary):
                career_aligned += 1

            # Display
            title_short = title[:45] if title else "Untitled"
            top_intent = (
                result.top_intents[0].value if result.top_intents else "unknown"
            )
            confidence = max((s.score for s in result.all_scores), default=0)
            aligned_icon = "✓" if classifier.is_career_aligned(title, summary) else " "

            print(f"[{row['id']}] {title_short}...")
            print(
                f"    {aligned_icon} Intent: {top_intent} ({confidence:.0%}) | Career Score: {result.career_alignment_score:.0%}"
            )

        # Summary
        print("\n" + "=" * 60)
        print("Intent Distribution:")
        for intent, count in sorted(
            intent_counts.items(), key=lambda x: x[1], reverse=True
        ):
            bar = "█" * count
            print(f"  {intent:<20} {bar} ({count})")

        print(
            f"\nCareer Aligned: {career_aligned}/{len(rows)} ({100 * career_aligned // len(rows)}%)"
        )

        # Show intent legend
        print("\nIntent Types:")
        print("  • skill_showcase: Demonstrates expertise")
        print("  • network_building: Encourages connections")
        print("  • thought_leadership: Establishes authority")
        print("  • industry_awareness: Shows market knowledge")

    except ImportError as e:
        print(f"\nError: Could not import intent_classifier module: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Intent classification failed")


def _test_notifications(engine: ContentEngine) -> None:
    """Test the notification system."""
    print("\n" + "=" * 60)
    print("  NOTIFICATION SYSTEM TEST")
    print("=" * 60)

    try:
        from notifications import (
            Notification,
            ConsoleChannel,
            SlackChannel,
        )

        results = []  # Track results for summary

        print("\n📋 Available Notification Channels:")
        print("   ├─ Console (always available)")
        print("   ├─ Slack (requires SLACK_WEBHOOK_URL)")
        print("   └─ Email (requires SMTP configuration)")

        # Test console notification
        print("\n" + "-" * 40)
        print("🖥️  Testing Console Channel")
        print("-" * 40)
        console = ConsoleChannel()

        test_notification = Notification(
            event_type="test",
            title="Test Notification",
            message="This is a test notification from Social Media Publisher.",
            priority="normal",
            data={"source": "menu_test", "timestamp": datetime.now().isoformat()},
        )

        result = console.send(test_notification)
        if result.success:
            print("   ✅ Console notification sent successfully")
            results.append(("Console", "✅ Passed", "Message displayed above"))
        else:
            print(f"   ❌ Console notification failed: {result.error}")
            results.append(("Console", "❌ Failed", result.error or "Unknown error"))

        # Check if Slack is configured
        from config import Config as AppConfig

        slack_url = getattr(AppConfig, "SLACK_WEBHOOK_URL", None)
        print("\n" + "-" * 40)
        print("💬 Testing Slack Channel")
        print("-" * 40)
        if slack_url:
            print(f"   Webhook URL: {slack_url[:40]}...")
            confirm = (
                input("   Send test notification to Slack? (y/n): ").strip().lower()
            )
            if confirm == "y":
                try:
                    slack = SlackChannel(webhook_url=slack_url)
                    slack_result = slack.send(test_notification)
                    if slack_result.success:
                        print("   ✅ Slack notification sent successfully")
                        results.append(
                            ("Slack", "✅ Passed", "Message sent to channel")
                        )
                    else:
                        print(f"   ❌ Slack notification failed: {slack_result.error}")
                        results.append(
                            (
                                "Slack",
                                "❌ Failed",
                                slack_result.error or "Unknown error",
                            )
                        )
                except Exception as e:
                    print(f"   ❌ Slack error: {e}")
                    results.append(("Slack", "❌ Error", str(e)))
            else:
                print("   ⏭️  Skipped by user")
                results.append(("Slack", "⏭️ Skipped", "User declined test"))
        else:
            print("   ⚠️  Not configured (SLACK_WEBHOOK_URL not set)")
            results.append(
                ("Slack", "⚠️ Not configured", "Set SLACK_WEBHOOK_URL in .env")
            )

        # Check if Email is configured
        smtp_host = getattr(AppConfig, "SMTP_HOST", None)
        print("\n" + "-" * 40)
        print("📧 Testing Email Channel")
        print("-" * 40)
        if smtp_host:
            print(f"   SMTP Host: {smtp_host}")
            print(
                "   ⚠️  Email test requires recipient address (not implemented in menu)"
            )
            results.append(("Email", "⚠️ Configured", "Manual test required"))
        else:
            print("   ⚠️  Not configured (SMTP_HOST not set)")
            results.append(("Email", "⚠️ Not configured", "Set SMTP_HOST in .env"))

        # Print summary table
        print("\n" + "=" * 60)
        print("  NOTIFICATION TEST SUMMARY")
        print("=" * 60)
        print(f"  {'Channel':<12} {'Status':<18} {'Details'}")
        print("  " + "-" * 56)
        for channel, status, details in results:
            print(f"  {channel:<12} {status:<18} {details[:35]}")
        print("=" * 60)

    except ImportError as e:
        print(f"\n❌ Error: Could not import notifications module: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Notification test failed")


def _reset_all(engine: ContentEngine) -> None:
    """Reset everything: delete database, all generated images, and all caches."""
    print("\n--- RESET ALL DATA ---")
    print("\n⚠️  WARNING: This will permanently delete:")
    print(f"    • Database: {Config.DB_NAME}")
    print(f"    • All files in: {Config.IMAGE_DIR}/")
    print("    • All caches (memory + SQLite + LinkedIn)")
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

    # Delete old LinkedIn cache (JSON format - legacy)
    cache_file = Path(
        os.path.expandvars(r"%LOCALAPPDATA%\SocialMediaPublisher\linkedin_cache.json")
    )
    if cache_file.exists():
        try:
            cache_file.unlink()
            print(f"  ✓ Deleted LinkedIn cache: {cache_file}")
        except Exception as e:
            print(f"  ✗ Failed to delete LinkedIn cache: {e}")

    # Clear the unified SQLite cache (cache.db)
    try:
        from cache import clear_global_cache

        # Clear memory cache
        clear_global_cache()
        print("  ✓ Cleared in-memory cache")

        # Delete the SQLite cache file (cache.db in project directory)
        cache_db_path = Path(__file__).parent / "cache.db"
        if cache_db_path.exists():
            try:
                cache_db_path.unlink()
                print(f"  ✓ Deleted SQLite cache: {cache_db_path}")
            except Exception as e:
                print(f"  ✗ Failed to delete SQLite cache: {e}")
    except ImportError:
        pass  # Cache module not available

    # Clear LinkedIn profile lookup caches (now uses LinkedInCache via SQLite)
    try:
        from cache import get_linkedin_cache

        # LinkedInCache uses SQLite backend which persists across sessions
        # The clear_person_cache() method clears person-related entries
        linkedin_cache = get_linkedin_cache()
        linkedin_cache.clear_person_cache()
        print("  ✓ Cleared LinkedIn profile cache")
    except (ImportError, AttributeError):
        pass  # Module not available or method doesn't exist

    print("\n✓ Reset complete!")


def _configure_linkedin_voyager() -> None:
    """Configure LinkedIn Voyager API cookies for reliable profile lookups."""
    print("\n" + "=" * 70)
    print("LINKEDIN VOYAGER API CONFIGURATION")
    print("=" * 70)
    print("""
The Voyager API uses LinkedIn's internal API for reliable profile lookups
without CAPTCHA detection. It requires two cookies from your browser:

  1. li_at - Main authentication cookie (long-lived)
  2. JSESSIONID - Session cookie (helps with CSRF)

To get these cookies:
  1. Open Chrome and log into LinkedIn
  2. Press F12 to open Developer Tools
  3. Go to Application tab → Cookies → www.linkedin.com
  4. Copy the values for 'li_at' and 'JSESSIONID'
""")

    # Check current status
    li_at = Config.LINKEDIN_LI_AT
    jsessionid = Config.LINKEDIN_JSESSIONID

    if li_at:
        print(
            f"Current li_at: {li_at[:20]}...{li_at[-10:]}"
            if len(li_at) > 30
            else f"Current li_at: {li_at}"
        )
        print(
            f"Current JSESSIONID: {jsessionid[:20]}..."
            if jsessionid
            else "Current JSESSIONID: (not set)"
        )
    else:
        print("Status: No LinkedIn Voyager cookies configured")

    print("-" * 70)

    # Option 1: Try auto-extract from browser
    print("\nOptions:")
    print("  1. Auto-extract from Chrome (requires browser-cookie3)")
    print("  2. Manual entry")
    print("  3. Test current configuration")
    print("  0. Cancel")

    choice = input("\nChoice: ").strip()

    if choice == "1":
        try:
            from linkedin_voyager_client import extract_linkedin_cookies_from_browser

            print("\nExtracting cookies from Chrome...")
            print("(Make sure Chrome is CLOSED for best results)")

            li_at, jsessionid = extract_linkedin_cookies_from_browser()

            if li_at:
                print(f"\n✓ Extracted li_at: {li_at[:20]}...")
                if jsessionid:
                    print(f"✓ Extracted JSESSIONID: {jsessionid[:20]}...")

                # Save to .env file
                _save_voyager_cookies_to_env(li_at, jsessionid)
            else:
                print("\n✗ Automatic extraction failed.")
                print("\n  Common causes:")
                print("  - Chrome is still running (close it completely)")
                print("  - Windows cookie encryption issues")
                print("  - Not logged into LinkedIn in Chrome")
                print("\n  → Please use option 2 (Manual entry) instead.")
                print(
                    "    It only takes 30 seconds to copy the cookies from Chrome DevTools."
                )
        except ImportError:
            print("\n✗ browser-cookie3 not installed.")
            print("  Install with: pip install browser-cookie3")
        except Exception as e:
            print(f"\n✗ Failed to extract cookies: {e}")
            print("\n  → Please use option 2 (Manual entry) instead.")

    elif choice == "2":
        print("\nEnter the cookie values (paste from browser dev tools):")
        li_at = input("li_at: ").strip()
        jsessionid = input("JSESSIONID: ").strip()

        if li_at:
            _save_voyager_cookies_to_env(li_at, jsessionid)
        else:
            print("Cancelled - no li_at provided.")

    elif choice == "3":
        print("\nTesting Voyager API connection...")
        try:
            from linkedin_voyager_client import LinkedInVoyagerClient

            client = LinkedInVoyagerClient()
            print("✓ Client initialized with cookies")

            # Skip auth check (endpoints deprecated) and go straight to search test
            print("\nTesting people search...")
            results = client.search_people(keywords=Config.DISCIPLINE, limit=3)
            if results:
                print(f"✓ Search successful! Found {len(results)} profiles:")
                for p in results[:3]:
                    headline_display = (
                        p.headline[:50] + "..." if len(p.headline) > 50 else p.headline
                    )
                    print(f"  - {p.name}: {headline_display}")
                print("\n✓ Voyager API is working correctly!")
            else:
                print("! Search returned no results.")
                print("  This could mean:")
                print("  - Cookies expired (re-enter them)")
                print("  - LinkedIn blocked the request")
                print("  - API endpoints changed")
            client.close()
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback

            traceback.print_exc()

    else:
        print("Cancelled.")


def _save_voyager_cookies_to_env(li_at: str, jsessionid: str) -> None:
    """Save Voyager cookies to .env file."""
    env_file = Path(".env")

    # Read existing .env content
    existing_lines = []
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

    # Update or add the cookie settings
    updated = False
    new_lines = []
    for line in existing_lines:
        if line.startswith("LINKEDIN_LI_AT="):
            new_lines.append(f"LINKEDIN_LI_AT={li_at}\n")
            updated = True
        elif line.startswith("LINKEDIN_JSESSIONID="):
            if jsessionid:
                new_lines.append(f"LINKEDIN_JSESSIONID={jsessionid}\n")
            # Skip empty jsessionid
        else:
            new_lines.append(line)

    # Add if not found
    if not updated:
        new_lines.append(f"LINKEDIN_LI_AT={li_at}\n")
        if jsessionid:
            new_lines.append(f"LINKEDIN_JSESSIONID={jsessionid}\n")

    # Write back
    with open(env_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"\n✓ Saved cookies to {env_file}")
    print("  Restart the application for changes to take effect.")


def _configure_rapidapi_key() -> None:
    """Configure RapidAPI key for Fresh LinkedIn Profile Data API lookups."""
    print("\n" + "=" * 70)
    print("RAPIDAPI KEY CONFIGURATION (Fresh LinkedIn Profile Data)")
    print("=" * 70)
    print("""
The Fresh LinkedIn Profile Data API provides reliable LinkedIn profile
lookups for @mentions. This is the primary method for finding profiles.

╔══════════════════════════════════════════════════════════════════════╗
║  HOW TO SUBSCRIBE (5 minutes):                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  1. Go to: https://rapidapi.com/freshdata-freshdata-default/         ║
║             api/fresh-linkedin-profile-data/pricing                  ║
║                                                                      ║
║  2. Sign up for a FREE RapidAPI account (if you don't have one)      ║
║                                                                      ║
║  3. Click "Subscribe" → Select "BASIC" plan ($10/month)              ║
║     • 500 requests/month (enough for ~150-200 lookups)               ║
║     • Cancel anytime from RapidAPI dashboard                         ║
║                                                                      ║
║  4. Copy your API key from the "X-RapidAPI-Key" header               ║
║     (same key works for all RapidAPI subscriptions)                  ║
╚══════════════════════════════════════════════════════════════════════╝

ALTERNATIVE (FREE): Browser-based lookup uses DuckDuckGo + Selenium.
                    Slower but works without API key.
""")

    # Check current status
    current_key = Config.RAPIDAPI_KEY

    if current_key:
        masked = (
            current_key[:8] + "..." + current_key[-4:]
            if len(current_key) > 12
            else current_key
        )
        print(f"Current API key: {masked}")
    else:
        print("Status: No RapidAPI key configured")

    print("-" * 70)

    print("\nOptions:")
    print("  1. Enter/update API key")
    print("  2. Test current configuration")
    print("  0. Cancel")

    choice = input("\nChoice: ").strip()

    if choice == "1":
        print("\nEnter your RapidAPI key:")
        api_key = input("API Key: ").strip()

        if api_key:
            _save_rapidapi_key_to_env(api_key)
        else:
            print("Cancelled - no API key provided.")

    elif choice == "2":
        if not current_key:
            print("\n✗ No API key configured. Please enter one first.")
            return

        print("\nTesting Fresh LinkedIn Profile Data API connection...")
        try:
            from linkedin_rapidapi_client import FreshLinkedInAPIClient

            client = FreshLinkedInAPIClient(api_key=current_key)
            print("✓ Client initialized")

            print("\nTesting search for 'Satya Nadella' at 'Microsoft'...")
            result = client.search_person("Satya Nadella", "Microsoft")

            if result and result.linkedin_url:
                print("✓ Search successful!")
                print(f"  Name: {result.full_name}")
                print(f"  Title: {result.job_title}")
                print(f"  Company: {result.company}")
                print(f"  LinkedIn: {result.linkedin_url}")
                print(f"  Match Score: {result.match_score:.2f}")
                print("\n✓ Fresh LinkedIn Profile Data API is working!")
            else:
                print("✗ Search returned no results.")
                print("  This could mean:")
                print("  - Not subscribed to Fresh LinkedIn Profile Data API")
                print(
                    "    → Subscribe at: https://rapidapi.com/freshdata-freshdata-default/"
                )
                print("                    api/fresh-linkedin-profile-data/pricing")
                print("  - API rate limit exceeded")
                print("  - Service temporarily unavailable")

        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("Cancelled.")


def _save_rapidapi_key_to_env(api_key: str) -> None:
    """Save RapidAPI key to .env file."""
    env_file = Path(".env")

    # Read existing .env content
    existing_lines = []
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

    # Update or add the API key
    updated = False
    new_lines = []
    for line in existing_lines:
        if line.startswith("RAPIDAPI_KEY="):
            new_lines.append(f"RAPIDAPI_KEY={api_key}\n")
            updated = True
        else:
            new_lines.append(line)

    # Add if not found
    if not updated:
        new_lines.append(f"RAPIDAPI_KEY={api_key}\n")

    # Write back
    with open(env_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"\n✓ Saved API key to {env_file}")
    print("  Restart the application for changes to take effect.")


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
    print("   Placeholders: {story_title}, {appearance}, {discipline}")
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
    print(
        "Prompts are defined in config.py defaults; override by setting the matching entries in your .env (e.g., SEARCH_INSTRUCTION_PROMPT, IMAGE_REFINEMENT_PROMPT, IMAGE_FALLBACK_PROMPT, VERIFICATION_PROMPT)."
    )
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
