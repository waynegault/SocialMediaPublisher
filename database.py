"""Database management for Social Media Publisher."""

import sqlite3
import json
import logging
import shutil
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional
from contextlib import contextmanager

from config import Config

logger = logging.getLogger(__name__)


def _adapt_datetime(value: datetime) -> str:
    """Serialize datetime for SQLite using ISO format."""
    return value.isoformat(sep=" ", timespec="microseconds")


def _convert_datetime(value: bytes) -> datetime:
    """Deserialize SQLite TIMESTAMP column to datetime."""
    return datetime.fromisoformat(value.decode())


sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("TIMESTAMP", _convert_datetime)


@dataclass
class Story:
    """Represents a story in the database.

    Fields are organized to follow the workflow sequence:
    1. Identity - Core story identification
    2. Search/Discovery - Data from initial story search
    3. Enrichment - LinkedIn profile lookup results
    4. Image Generation - Generated image data
    5. Verification - Content quality verification
    6. Scheduling/Publishing - Publication workflow
    7. Analytics - Post-publication metrics
    8. Legacy - Deprecated fields for backward compatibility
    """

    # --- 1. IDENTITY ---
    id: Optional[int] = None
    title: str = ""
    summary: str = ""

    # --- 2. SEARCH/DISCOVERY (from story search) ---
    source_links: list[str] = field(default_factory=list)
    acquire_date: Optional[datetime] = None
    quality_score: int = 0
    quality_justification: str = ""  # Reasoning for the quality score
    category: str = "Other"  # Medicine, Hydrogen, Research, Technology, Business, Science, AI, Other
    hashtags: list[str] = field(
        default_factory=list
    )  # Hashtags for LinkedIn posts (max 3)

    # --- 3. ENRICHMENT (LinkedIn profile lookup) ---
    enrichment_status: str = "pending"  # pending, enriched, skipped, error
    # Organizations mentioned in the story (just names)
    organizations: list[str] = field(default_factory=list)  # ["BASF", "MIT", "IChemE"]
    # People mentioned directly in the story (PRIMARY field for @mentions)
    # [{"name": "Dr. Jane Smith", "title": "Lead Researcher", "affiliation": "MIT", "linkedin_profile": "", "linkedin_urn": ""}]
    story_people: list[dict] = field(default_factory=list)
    # Key leaders from the organizations (CEO, CTO, etc.) - for secondary @mentions
    # [{"name": "John Doe", "title": "CEO", "organization": "BASF", "linkedin_profile": "", "linkedin_urn": ""}]
    org_leaders: list[dict] = field(
        default_factory=list
    )  # [{"name": "John Doe", "title": "CEO", "organization": "BASF"}]
    # New canonical fields for enrichment pipeline
    direct_people: list[dict] = field(default_factory=list)
    indirect_people: list[dict] = field(default_factory=list)

    # --- 3b. ENRICHMENT METADATA (Phase 1) ---
    # Processing metrics, errors, and audit trail for debugging
    # Structure: {"completed_at": "ISO8601", "processing_time_seconds": N, "errors": [], ...}
    enrichment_log: dict = field(default_factory=dict)
    # Overall quality assessment: "high", "medium", "low", "failed"
    enrichment_quality: str = ""

    # --- 4. IMAGE GENERATION ---
    image_path: Optional[str] = None
    image_alt_text: Optional[str] = None  # Accessibility alt text describing the image

    # --- 5. VERIFICATION ---
    verification_status: str = "pending"  # pending, approved, rejected
    verification_reason: Optional[str] = None  # AI's reason for approval/rejection

    # --- 6. SCHEDULING/PUBLISHING ---
    scheduled_time: Optional[datetime] = None
    publish_status: str = "unpublished"  # unpublished, scheduled, published
    published_time: Optional[datetime] = None
    linkedin_post_id: Optional[str] = None
    linkedin_post_url: Optional[str] = None
    # Promotional message to append to LinkedIn posts (randomly selected from promotion.json)
    promotion: Optional[str] = None

    # --- 7. ANALYTICS (fetched from API after publication) ---
    linkedin_impressions: int = 0
    linkedin_clicks: int = 0
    linkedin_likes: int = 0
    linkedin_comments: int = 0
    linkedin_shares: int = 0
    linkedin_engagement: float = (
        0.0  # Engagement rate (clicks+likes+comments+shares / impressions)
    )
    linkedin_analytics_fetched_at: Optional[datetime] = None

    # --- 8. LEGACY FIELDS (deprecated, kept for backward compatibility) ---
    company_mention_enrichment: Optional[str] = None  # DEPRECATED - use organizations
    individuals: list[str] = field(
        default_factory=list
    )  # DEPRECATED - use story_people
    linkedin_profiles: list[dict] = field(
        default_factory=list
    )  # DEPRECATED - use story_people.linkedin_profile

    def to_dict(self) -> dict:
        """Convert story to dictionary (organized by workflow sequence)."""
        return {
            # 1. Identity
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            # 2. Search/Discovery
            "source_links": self.source_links,
            "acquire_date": self.acquire_date.isoformat()
            if self.acquire_date
            else None,
            "quality_score": self.quality_score,
            "quality_justification": self.quality_justification,
            "category": self.category,
            "hashtags": self.hashtags,
            # 3. Enrichment
            "enrichment_status": self.enrichment_status,
            "organizations": self.organizations,
            "story_people": self.story_people,
            "org_leaders": self.org_leaders,
            "direct_people": self.direct_people,
            "indirect_people": self.indirect_people,
            "enrichment_log": self.enrichment_log,
            "enrichment_quality": self.enrichment_quality,
            # 4. Image Generation
            "image_path": self.image_path,
            "image_alt_text": self.image_alt_text,
            # 5. Verification
            "verification_status": self.verification_status,
            "verification_reason": self.verification_reason,
            # 6. Scheduling/Publishing
            "scheduled_time": self.scheduled_time.isoformat()
            if self.scheduled_time
            else None,
            "publish_status": self.publish_status,
            "published_time": self.published_time.isoformat()
            if self.published_time
            else None,
            "linkedin_post_id": self.linkedin_post_id,
            "linkedin_post_url": self.linkedin_post_url,
            "promotion": self.promotion,
            # 7. Analytics
            "linkedin_impressions": self.linkedin_impressions,
            "linkedin_clicks": self.linkedin_clicks,
            "linkedin_likes": self.linkedin_likes,
            "linkedin_comments": self.linkedin_comments,
            "linkedin_shares": self.linkedin_shares,
            "linkedin_engagement": self.linkedin_engagement,
            "linkedin_analytics_fetched_at": self.linkedin_analytics_fetched_at.isoformat()
            if self.linkedin_analytics_fetched_at
            else None,
            # 8. Legacy (deprecated)
            "company_mention_enrichment": self.company_mention_enrichment,
            "individuals": self.individuals,
            "linkedin_profiles": self.linkedin_profiles,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Story":
        """Create a Story from a database row."""
        source_links = []
        if row["source_links"]:
            try:
                source_links = json.loads(row["source_links"])
            except json.JSONDecodeError:
                source_links = [row["source_links"]]

        # Handle optional columns that may not exist in older databases
        keys = row.keys()

        # Parse hashtags (JSON array)
        hashtags = []
        if "hashtags" in keys and row["hashtags"]:
            try:
                hashtags = json.loads(row["hashtags"])
            except json.JSONDecodeError:
                hashtags = []

        organizations = (
            json.loads(row["organizations"])
            if "organizations" in keys and row["organizations"]
            else []
        )
        story_people = (
            json.loads(row["story_people"])
            if "story_people" in keys and row["story_people"]
            else []
        )
        org_leaders = (
            json.loads(row["org_leaders"])
            if "org_leaders" in keys and row["org_leaders"]
            else []
        )
        direct_people = (
            json.loads(row["direct_people"])
            if "direct_people" in keys and row["direct_people"]
            else []
        )
        indirect_people = (
            json.loads(row["indirect_people"])
            if "indirect_people" in keys and row["indirect_people"]
            else []
        )

        return cls(
            id=row["id"],
            title=row["title"],
            summary=row["summary"],
            source_links=source_links,
            acquire_date=_parse_datetime(row["acquire_date"]),
            quality_score=row["quality_score"],
            category=row["category"] if "category" in keys else "Other",
            quality_justification=row["quality_justification"]
            if "quality_justification" in keys
            else "",
            image_path=row["image_path"],
            image_alt_text=row["image_alt_text"] if "image_alt_text" in keys else None,
            verification_status=row["verification_status"],
            verification_reason=row["verification_reason"]
            if "verification_reason" in keys
            else None,
            publish_status=row["publish_status"],
            scheduled_time=_parse_datetime(row["scheduled_time"]),
            published_time=_parse_datetime(row["published_time"]),
            linkedin_post_id=row["linkedin_post_id"],
            linkedin_post_url=row["linkedin_post_url"]
            if "linkedin_post_url" in keys
            else None,
            hashtags=hashtags,
            linkedin_impressions=row["linkedin_impressions"]
            if "linkedin_impressions" in keys
            else 0,
            linkedin_clicks=row["linkedin_clicks"] if "linkedin_clicks" in keys else 0,
            linkedin_likes=row["linkedin_likes"] if "linkedin_likes" in keys else 0,
            linkedin_comments=row["linkedin_comments"]
            if "linkedin_comments" in keys
            else 0,
            linkedin_shares=row["linkedin_shares"] if "linkedin_shares" in keys else 0,
            linkedin_engagement=row["linkedin_engagement"]
            if "linkedin_engagement" in keys
            else 0.0,
            linkedin_analytics_fetched_at=_parse_datetime(
                row["linkedin_analytics_fetched_at"]
            )
            if "linkedin_analytics_fetched_at" in keys
            else None,
            enrichment_status=row["enrichment_status"]
            if "enrichment_status" in keys
            else "pending",
            # New fields
            organizations=organizations,
            direct_people=direct_people,
            indirect_people=indirect_people,
            story_people=story_people,
            org_leaders=org_leaders,
            # Phase 1: Enrichment metadata
            enrichment_log=json.loads(row["enrichment_log"])
            if "enrichment_log" in keys and row["enrichment_log"]
            else {},
            enrichment_quality=row["enrichment_quality"]
            if "enrichment_quality" in keys
            else "",
            # Promotion message
            promotion=row["promotion"] if "promotion" in keys else None,
            # Legacy fields
            company_mention_enrichment=row["company_mention_enrichment"]
            if "company_mention_enrichment" in keys
            else None,
            individuals=json.loads(row["individuals"])
            if "individuals" in keys and row["individuals"]
            else [],
            linkedin_profiles=json.loads(row["linkedin_profiles"])
            if "linkedin_profiles" in keys and row["linkedin_profiles"]
            else [],
        )


def _parse_datetime(value: Optional[datetime | str]) -> Optional[datetime]:
    """Parse datetime from database value, supporting string or datetime inputs."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class Database:
    """Database manager for story storage and retrieval."""

    def __init__(self, db_name: Optional[str] = None) -> None:
        """Initialize database connection."""
        self.db_name = db_name or Config.DB_NAME
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(
            self.db_name, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _get_existing_columns(
        self, cursor: sqlite3.Cursor, table_name: str
    ) -> set[str]:
        """Get the set of existing column names for a table."""
        cursor.execute(f"PRAGMA table_info({table_name})")
        return {row[1] for row in cursor.fetchall()}

    def _migrate_add_column(
        self,
        cursor: sqlite3.Cursor,
        column_name: str,
        column_def: str,
        existing_columns: set[str] | None = None,
    ) -> None:
        """Add a column to the stories table if it doesn't exist.

        Args:
            cursor: Database cursor
            column_name: Name of the column to add
            column_def: Column definition (e.g., 'TEXT DEFAULT NULL')
            existing_columns: Optional pre-fetched set of existing column names
                             (avoids repeated PRAGMA calls)
        """
        # If existing_columns provided, do a quick check first
        if existing_columns is not None:
            if column_name in existing_columns:
                return  # Column already exists, skip

        try:
            cursor.execute(f"ALTER TABLE stories ADD COLUMN {column_name} {column_def}")
            logger.debug(f"Added column '{column_name}' to stories table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                pass  # Column already exists
            else:
                raise

    def _init_db(self) -> None:
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Stories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    summary TEXT,
                    source_links TEXT,
                    acquire_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    quality_score INTEGER DEFAULT 0,
                    category TEXT DEFAULT 'Other',
                    quality_justification TEXT DEFAULT '',
                    image_path TEXT,
                    verification_status TEXT DEFAULT 'pending',
                    verification_reason TEXT,
                    publish_status TEXT DEFAULT 'unpublished',
                    scheduled_time TIMESTAMP,
                    published_time TIMESTAMP,
                    linkedin_post_id TEXT
                )
            """)

            # Get existing columns once for all migration checks
            existing_columns = self._get_existing_columns(cursor, "stories")

            # Migrate existing databases: add new columns if they don't exist
            self._migrate_add_column(
                cursor, "category", "TEXT DEFAULT 'Other'", existing_columns
            )
            self._migrate_add_column(
                cursor, "quality_justification", "TEXT DEFAULT ''", existing_columns
            )
            self._migrate_add_column(
                cursor, "verification_reason", "TEXT", existing_columns
            )
            self._migrate_add_column(
                cursor, "linkedin_post_url", "TEXT", existing_columns
            )
            self._migrate_add_column(
                cursor, "hashtags", "TEXT DEFAULT '[]'", existing_columns
            )
            self._migrate_add_column(
                cursor, "linkedin_mentions", "TEXT DEFAULT '[]'", existing_columns
            )
            # LinkedIn analytics columns
            self._migrate_add_column(
                cursor, "linkedin_impressions", "INTEGER DEFAULT 0", existing_columns
            )
            self._migrate_add_column(
                cursor, "linkedin_clicks", "INTEGER DEFAULT 0", existing_columns
            )
            self._migrate_add_column(
                cursor, "linkedin_likes", "INTEGER DEFAULT 0", existing_columns
            )
            self._migrate_add_column(
                cursor, "linkedin_comments", "INTEGER DEFAULT 0", existing_columns
            )
            self._migrate_add_column(
                cursor, "linkedin_shares", "INTEGER DEFAULT 0", existing_columns
            )
            self._migrate_add_column(
                cursor, "linkedin_engagement", "REAL DEFAULT 0.0", existing_columns
            )
            self._migrate_add_column(
                cursor, "linkedin_analytics_fetched_at", "TIMESTAMP", existing_columns
            )
            # Company mention enrichment columns
            self._migrate_add_column(
                cursor, "company_mention_enrichment", "TEXT", existing_columns
            )
            self._migrate_add_column(
                cursor, "enrichment_status", "TEXT DEFAULT 'pending'", existing_columns
            )
            # Individual people and their LinkedIn profiles (legacy)
            self._migrate_add_column(
                cursor, "individuals", "TEXT DEFAULT '[]'", existing_columns
            )
            self._migrate_add_column(
                cursor, "linkedin_profiles", "TEXT DEFAULT '[]'", existing_columns
            )
            # New enrichment fields (cleaner structure)
            self._migrate_add_column(
                cursor, "organizations", "TEXT DEFAULT '[]'", existing_columns
            )
            self._migrate_add_column(
                cursor, "story_people", "TEXT DEFAULT '[]'", existing_columns
            )
            self._migrate_add_column(
                cursor, "org_leaders", "TEXT DEFAULT '[]'", existing_columns
            )
            self._migrate_add_column(
                cursor, "direct_people", "TEXT DEFAULT '[]'", existing_columns
            )
            self._migrate_add_column(
                cursor, "indirect_people", "TEXT DEFAULT '[]'", existing_columns
            )
            self._migrate_add_column(
                cursor, "linkedin_handles", "TEXT DEFAULT '[]'", existing_columns
            )
            # Phase 1: Enrichment metadata fields
            self._migrate_add_column(
                cursor, "enrichment_log", "TEXT DEFAULT '{}'", existing_columns
            )
            self._migrate_add_column(
                cursor, "enrichment_quality", "TEXT DEFAULT ''", existing_columns
            )
            # Promotion message for LinkedIn posts
            self._migrate_add_column(cursor, "promotion", "TEXT", existing_columns)
            # Image alt text for accessibility
            self._migrate_add_column(cursor, "image_alt_text", "TEXT", existing_columns)

            # System state table for tracking last check date, etc.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            # Indices for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stories_status
                ON stories(publish_status, verification_status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stories_quality
                ON stories(quality_score DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stories_category
                ON stories(category)
            """)

            logger.debug("Database initialized successfully")

    # --- Story CRUD Operations ---

    def add_story(self, story: Story) -> int:
        """Add a new story to the database. Returns the story ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO stories
                (title, summary, source_links, acquire_date, quality_score,
                 category, quality_justification, image_path, verification_status,
                 publish_status, hashtags, company_mention_enrichment,
                 enrichment_status, story_people, direct_people, organizations,
                 org_leaders, indirect_people, enrichment_log, enrichment_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    story.title,
                    story.summary,
                    json.dumps(story.source_links),
                    story.acquire_date or datetime.now(),
                    story.quality_score,
                    story.category,
                    story.quality_justification,
                    story.image_path,
                    story.verification_status,
                    story.publish_status,
                    json.dumps(story.hashtags),
                    story.company_mention_enrichment,
                    story.enrichment_status,
                    json.dumps(story.story_people),
                    json.dumps(story.direct_people),
                    json.dumps(story.organizations),
                    json.dumps(story.org_leaders),
                    json.dumps(story.indirect_people),
                    json.dumps(story.enrichment_log),
                    story.enrichment_quality,
                ),
            )
            story_id = cursor.lastrowid or 0
            logger.debug(f"Added story with ID {story_id}: {story.title}")
            return story_id

    def get_story(self, story_id: int) -> Optional[Story]:
        """Get a story by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM stories WHERE id = ?", (story_id,))
            row = cursor.fetchone()
            return Story.from_row(row) if row else None

    def get_story_by_title(self, title: str) -> Optional[Story]:
        """Get a story by title (for deduplication)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM stories WHERE title = ?", (title,))
            row = cursor.fetchone()
            return Story.from_row(row) if row else None

    def get_all_story_titles(self, limit: int | None = None) -> list[tuple[int, str]]:
        """Get story IDs and titles for semantic deduplication.

        Args:
            limit: Optional maximum number of stories to return (most recent first).
                   Use this for large databases to limit memory usage.
                   Default is None (all stories, for backwards compatibility).

        Returns:
            List of (id, title) tuples ordered by acquire_date descending (most recent first).
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if limit is not None:
                cursor.execute(
                    "SELECT id, title FROM stories ORDER BY acquire_date DESC LIMIT ?",
                    (limit,),
                )
            else:
                cursor.execute(
                    "SELECT id, title FROM stories ORDER BY acquire_date DESC"
                )
            return [(row["id"], row["title"]) for row in cursor.fetchall()]

    def get_recent_story_titles(self, days: int = 90) -> list[tuple[int, str]]:
        """Get story IDs and titles from the last N days for deduplication.

        This is more memory-efficient than get_all_story_titles for large databases
        since duplicates are most likely to be found among recent stories.

        Args:
            days: Number of days to look back (default: 90)

        Returns:
            List of (id, title) tuples from the last N days.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title FROM stories
                WHERE acquire_date >= datetime('now', '-' || ? || ' days')
                ORDER BY acquire_date DESC
                """,
                (days,),
            )
            return [(row["id"], row["title"]) for row in cursor.fetchall()]

    def get_recent_published_titles(self, days: int = 30) -> list[tuple[int, str]]:
        """Get published story IDs and titles from the last N days for deduplication.

        This specifically returns only published stories, which is useful for
        preventing similar content from being posted within a configurable window.

        Args:
            days: Number of days to look back (default: 30)

        Returns:
            List of (id, title) tuples from published stories in the last N days.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title FROM stories
                WHERE publish_status = 'published'
                AND published_time >= datetime('now', '-' || ? || ' days')
                ORDER BY published_time DESC
                """,
                (days,),
            )
            return [(row["id"], row["title"]) for row in cursor.fetchall()]

    def update_story(self, story: Story) -> bool:
        """Update an existing story."""
        if story.id is None:
            return False

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE stories SET
                    title = ?,
                    summary = ?,
                    source_links = ?,
                    quality_score = ?,
                    category = ?,
                    quality_justification = ?,
                    image_path = ?,
                    image_alt_text = ?,
                    verification_status = ?,
                    verification_reason = ?,
                    publish_status = ?,
                    scheduled_time = ?,
                    published_time = ?,
                    linkedin_post_id = ?,
                    linkedin_post_url = ?,
                    hashtags = ?,
                    linkedin_impressions = ?,
                    linkedin_clicks = ?,
                    linkedin_likes = ?,
                    linkedin_comments = ?,
                    linkedin_shares = ?,
                    linkedin_engagement = ?,
                    linkedin_analytics_fetched_at = ?,
                    enrichment_status = ?,
                    organizations = ?,
                    story_people = ?,
                    direct_people = ?,
                    org_leaders = ?,
                    indirect_people = ?,
                    enrichment_log = ?,
                    enrichment_quality = ?,
                    company_mention_enrichment = ?,
                    individuals = ?,
                    linkedin_profiles = ?
                WHERE id = ?
                """,
                (
                    story.title,
                    story.summary,
                    json.dumps(story.source_links),
                    story.quality_score,
                    story.category,
                    story.quality_justification,
                    story.image_path,
                    story.image_alt_text,
                    story.verification_status,
                    story.verification_reason,
                    story.publish_status,
                    story.scheduled_time,
                    story.published_time,
                    story.linkedin_post_id,
                    story.linkedin_post_url,
                    json.dumps(story.hashtags),
                    story.linkedin_impressions,
                    story.linkedin_clicks,
                    story.linkedin_likes,
                    story.linkedin_comments,
                    story.linkedin_shares,
                    story.linkedin_engagement,
                    story.linkedin_analytics_fetched_at,
                    story.enrichment_status,
                    json.dumps(story.organizations),
                    json.dumps(story.story_people),
                    json.dumps(story.direct_people),
                    json.dumps(story.org_leaders),
                    json.dumps(story.indirect_people),
                    json.dumps(story.enrichment_log),
                    story.enrichment_quality,
                    story.company_mention_enrichment,
                    json.dumps(story.individuals),
                    json.dumps(story.linkedin_profiles),
                    story.id,
                ),
            )
            return cursor.rowcount > 0

    def delete_story(self, story_id: int) -> bool:
        """Delete a story by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM stories WHERE id = ?", (story_id,))
            return cursor.rowcount > 0

    # --- Query Operations ---

    def get_stories_with_images(self) -> list[Story]:
        """Get all stories that have an image_path set (regardless of file existence)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM stories
                WHERE image_path IS NOT NULL
                ORDER BY acquire_date DESC
                """
            )
            return [Story.from_row(row) for row in cursor.fetchall()]

    def get_stories_needing_images(self, min_quality: int) -> list[Story]:
        """Get stories that need image generation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM stories
                WHERE image_path IS NULL
                AND quality_score >= ?
                AND publish_status IN ('unpublished', 'scheduled')
                ORDER BY quality_score DESC
                """,
                (min_quality,),
            )
            return [Story.from_row(row) for row in cursor.fetchall()]

    def get_stories_needing_verification(self) -> list[Story]:
        """Get stories that need content verification.

        Returns all pending stories regardless of image status.
        Stories without images will be auto-rejected during verification
        if their quality score is too low.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM stories
                WHERE verification_status = 'pending'
                ORDER BY quality_score DESC
                """
            )
            return [Story.from_row(row) for row in cursor.fetchall()]

    def get_stories_needing_enrichment(self) -> list[Story]:
        """Get stories that need company mention enrichment.

        Returns all stories with pending enrichment status.
        Enrichment runs early in the pipeline (before verification),
        so we don't require verification_status='approved'.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM stories
                WHERE enrichment_status = 'pending'
                ORDER BY quality_score DESC
                """
            )
            return [Story.from_row(row) for row in cursor.fetchall()]

    def get_approved_unpublished_stories(self, limit: int) -> list[Story]:
        """Get approved stories that haven't been published."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM stories
                WHERE verification_status = 'approved'
                AND publish_status = 'unpublished'
                ORDER BY quality_score DESC, acquire_date DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [Story.from_row(row) for row in cursor.fetchall()]

    def get_rejected_stories(self) -> list[Story]:
        """Get stories that were rejected during verification."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM stories
                WHERE verification_status = 'rejected'
                ORDER BY quality_score DESC, acquire_date DESC
                """
            )
            return [Story.from_row(row) for row in cursor.fetchall()]

    def get_scheduled_stories_due(self) -> list[Story]:
        """Get stories that are scheduled and due for publishing."""
        now = datetime.now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM stories
                WHERE publish_status = 'scheduled'
                AND scheduled_time <= ?
                ORDER BY scheduled_time ASC
                """,
                (now,),
            )
            return [Story.from_row(row) for row in cursor.fetchall()]

    def get_scheduled_stories(self) -> list[Story]:
        """Get all currently scheduled stories."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM stories
                WHERE publish_status = 'scheduled'
                ORDER BY scheduled_time ASC
                """
            )
            return [Story.from_row(row) for row in cursor.fetchall()]

    def get_published_stories(self, limit: int | None = None) -> list[Story]:
        """Get all published stories, ordered by publish time (most recent first)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT * FROM stories
                WHERE publish_status = 'published'
                AND linkedin_post_id IS NOT NULL
                ORDER BY published_time DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query)
            return [Story.from_row(row) for row in cursor.fetchall()]

    def clear_scheduled_status(self) -> int:
        """Reset all scheduled stories to unpublished. Returns count."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE stories
                SET publish_status = 'unpublished', scheduled_time = NULL
                WHERE publish_status = 'scheduled'
                """
            )
            return cursor.rowcount

    def count_unpublished_stories(self) -> int:
        """Count available unpublished stories."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) FROM stories
                WHERE verification_status = 'approved'
                AND publish_status = 'unpublished'
                """
            )
            return cursor.fetchone()[0]

    # --- Cleanup Operations ---

    def delete_old_unused_stories(self, cutoff_date: datetime) -> int:
        """Delete old stories that were never published. Returns count deleted."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM stories
                WHERE publish_status != 'published'
                AND acquire_date < ?
                """,
                (cutoff_date,),
            )
            count = cursor.rowcount
            if count > 0:
                logger.info(f"Deleted {count} old unused stories")
            return count

    # --- System State Operations ---

    def get_state(self, key: str) -> Optional[str]:
        """Get a system state value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM system_state WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else None

    def set_state(self, key: str, value: str) -> None:
        """Set a system state value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
                (key, value),
            )

    def get_last_check_date(self) -> Optional[datetime]:
        """Get the last time we checked for stories."""
        value = self.get_state("last_check")
        return _parse_datetime(value)

    def set_last_check_date(self, dt: Optional[datetime] = None) -> None:
        """Set the last check date (defaults to now)."""
        dt = dt or datetime.now()
        self.set_state("last_check", dt.isoformat())

    # --- Statistics ---

    def get_statistics(self) -> dict:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            cursor.execute("SELECT COUNT(*) FROM stories")
            stats["total_stories"] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM stories WHERE publish_status = 'published'"
            )
            stats["published_count"] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM stories WHERE publish_status = 'scheduled'"
            )
            stats["scheduled_count"] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM stories WHERE verification_status = 'approved' AND publish_status = 'unpublished'"
            )
            stats["available_count"] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM stories WHERE verification_status = 'pending'"
            )
            stats["pending_verification"] = cursor.fetchone()[0]

            # Stories with images ready for verification
            cursor.execute(
                "SELECT COUNT(*) FROM stories WHERE verification_status = 'pending' AND image_path IS NOT NULL"
            )
            stats["ready_for_verification"] = cursor.fetchone()[0]

            # Stories needing images
            cursor.execute(
                "SELECT COUNT(*) FROM stories WHERE image_path IS NULL AND publish_status IN ('unpublished', 'scheduled')"
            )
            stats["needing_images"] = cursor.fetchone()[0]

            # Enrichment stats
            cursor.execute(
                "SELECT COUNT(*) FROM stories WHERE enrichment_status = 'pending' AND verification_status = 'approved' AND image_path IS NOT NULL"
            )
            stats["pending_enrichment"] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM stories WHERE enrichment_status = 'enriched'"
            )
            stats["enriched_count"] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM stories WHERE enrichment_status = 'enriched' AND company_mention_enrichment IS NOT NULL"
            )
            stats["with_mentions"] = cursor.fetchone()[0]

            stats["no_mentions"] = stats["enriched_count"] - stats["with_mentions"]

            return stats

    # ==========================================================================
    # PHASE 3: CROSS-STORY ENTITY RESOLUTION
    # ==========================================================================

    def find_person_by_linkedin_urn(self, urn: str) -> dict | None:
        """Find a person across all stories by their LinkedIn URN.

        This enables cross-story entity resolution - if we've already validated
        a person in one story, we can reuse that validation in another story.

        Args:
            urn: LinkedIn URN (e.g., "urn:li:person:ABC123")

        Returns:
            Person dict with all known attributes, or None if not found
        """
        if not urn:
            return None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Search story_people first
            cursor.execute(
                """
                SELECT story_people, org_leaders FROM stories
                WHERE story_people LIKE ? OR org_leaders LIKE ?
            """,
                (f"%{urn}%", f"%{urn}%"),
            )

            for row in cursor.fetchall():
                # Check story_people
                if row["story_people"]:
                    people = json.loads(row["story_people"])
                    for person in people:
                        if person.get("linkedin_urn") == urn:
                            return person

                # Check org_leaders
                if row["org_leaders"]:
                    leaders = json.loads(row["org_leaders"])
                    for leader in leaders:
                        if leader.get("linkedin_urn") == urn:
                            return leader

        return None

    def find_person_by_attributes(
        self,
        name: str,
        employer: str | None = None,
        fuzzy: bool = True,
    ) -> dict | None:
        """Find a person across all stories by name and optional employer.

        Args:
            name: Person's name to search for
            employer: Optional employer/organization name
            fuzzy: If True, use case-insensitive partial matching

        Returns:
            Person dict with all known attributes, or None if not found
        """
        if not name:
            return None

        name_lower = name.lower().strip()
        employer_lower = (employer or "").lower().strip()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT story_people, org_leaders FROM stories")

            best_match: dict | None = None
            best_score = 0

            for row in cursor.fetchall():
                for field_name in ["story_people", "org_leaders"]:
                    data = row[field_name]
                    if not data:
                        continue

                    people = json.loads(data)
                    for person in people:
                        person_name = (person.get("name") or "").lower().strip()

                        # Skip if no name match
                        if fuzzy:
                            if (
                                name_lower not in person_name
                                and person_name not in name_lower
                            ):
                                continue
                        else:
                            if person_name != name_lower:
                                continue

                        # Calculate match score
                        score = 1  # Base score for name match

                        # Bonus for exact name match
                        if person_name == name_lower:
                            score += 2

                        # Bonus for employer match
                        if employer_lower:
                            person_org = (
                                (
                                    person.get("company")
                                    or person.get("affiliation")
                                    or person.get("organization")
                                    or ""
                                )
                                .lower()
                                .strip()
                            )
                            if (
                                employer_lower in person_org
                                or person_org in employer_lower
                            ):
                                score += 3

                        # Bonus for having LinkedIn profile
                        if person.get("linkedin_profile") or person.get("linkedin_urn"):
                            score += 2

                        if score > best_score:
                            best_score = score
                            best_match = person

            return best_match

    def get_enrichment_dashboard_stats(self) -> dict:
        """Get enrichment-specific stats for dashboard display.

        Phase 3: Aggregate enrichment metrics for monitoring.

        Returns:
            Dict with enrichment stats
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats: dict = {}

            # Total people counts
            cursor.execute("SELECT story_people, org_leaders FROM stories")
            total_direct = 0
            total_indirect = 0
            with_linkedin = 0
            high_confidence = 0
            org_fallback = 0

            for row in cursor.fetchall():
                if row["story_people"]:
                    people = json.loads(row["story_people"])
                    total_direct += len(people)
                    for p in people:
                        if p.get("linkedin_profile") or p.get("linkedin_urn"):
                            with_linkedin += 1
                        conf = p.get("match_confidence", "")
                        if conf == "high":
                            high_confidence += 1
                        elif conf == "org_fallback":
                            org_fallback += 1

                if row["org_leaders"]:
                    leaders = json.loads(row["org_leaders"])
                    total_indirect += len(leaders)
                    for l in leaders:
                        if l.get("linkedin_profile") or l.get("linkedin_urn"):
                            with_linkedin += 1
                        conf = l.get("match_confidence", "")
                        if conf == "high":
                            high_confidence += 1
                        elif conf == "org_fallback":
                            org_fallback += 1

            total_people = total_direct + total_indirect

            stats["total_direct_people"] = total_direct
            stats["total_indirect_people"] = total_indirect
            stats["total_people"] = total_people
            stats["with_linkedin"] = with_linkedin
            stats["linkedin_rate"] = (
                f"{with_linkedin / total_people:.1%}" if total_people > 0 else "0%"
            )
            stats["high_confidence"] = high_confidence
            stats["high_confidence_rate"] = (
                f"{high_confidence / with_linkedin:.1%}" if with_linkedin > 0 else "0%"
            )
            stats["org_fallback"] = org_fallback
            stats["org_fallback_rate"] = (
                f"{org_fallback / with_linkedin:.1%}" if with_linkedin > 0 else "0%"
            )

            # Enrichment quality breakdown
            cursor.execute("""
                SELECT enrichment_quality, COUNT(*) as cnt
                FROM stories
                WHERE enrichment_quality != ''
                GROUP BY enrichment_quality
            """)
            stats["quality_breakdown"] = {
                row["enrichment_quality"]: row["cnt"] for row in cursor.fetchall()
            }

            # Stories needing manual review
            cursor.execute("""
                SELECT COUNT(*) FROM stories
                WHERE enrichment_quality = 'low'
                   OR enrichment_quality = 'failed'
            """)
            stats["needs_review"] = cursor.fetchone()[0]

            return stats

    def get_stories_needing_review(self, limit: int = 50) -> list["Story"]:
        """Get stories flagged for manual review.

        Phase 3: Manual review workflow support.

        Args:
            limit: Maximum stories to return

        Returns:
            List of Story objects needing review
        """
        stories = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM stories
                WHERE enrichment_quality IN ('low', 'failed')
                   OR (enrichment_status = 'error')
                ORDER BY acquire_date DESC
                LIMIT ?
            """,
                (limit,),
            )
            for row in cursor.fetchall():
                stories.append(Story.from_row(row))
        return stories

    def mark_story_reviewed(self, story_id: int, reviewer_notes: str = "") -> bool:
        """Mark a story as manually reviewed.

        Args:
            story_id: Story ID to mark
            reviewer_notes: Optional notes from reviewer

        Returns:
            True if update succeeded
        """
        try:
            story = self.get_story(story_id)
            if not story:
                return False

            # Update enrichment_log with review info
            log = story.enrichment_log or {}
            if isinstance(log, str):
                log = json.loads(log) if log else {}
            log["manual_review"] = {
                "reviewed_at": datetime.now().isoformat(),
                "notes": reviewer_notes,
            }

            # Update the story object and save
            story.enrichment_log = log
            story.enrichment_quality = "reviewed"
            self.update_story(story)
            return True
        except Exception as e:
            logger.error(f"Failed to mark story {story_id} as reviewed: {e}")
            return False

    def create_backup(self, suffix: str = ".backup") -> str:
        """Create a backup of the database file.

        Args:
            suffix: Suffix to append to database name for backup file

        Returns:
            Path to the backup file
        """
        backup_path = self.db_name + suffix
        try:
            shutil.copy2(self.db_name, backup_path)
            logger.info(f"Created database backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise

    def restore_from_backup(self, backup_path: Optional[str] = None) -> bool:
        """Restore database from a backup file.

        Args:
            backup_path: Path to backup file. Defaults to db_name + ".backup"

        Returns:
            True if restore succeeded
        """
        if backup_path is None:
            backup_path = self.db_name + ".backup"

        if not Path(backup_path).exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False

        try:
            shutil.copy2(backup_path, self.db_name)
            logger.info(f"Restored database from: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False

    def verify_integrity(self) -> tuple[bool, str]:
        """Verify database integrity using SQLite's integrity_check.

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()[0]

                if result == "ok":
                    logger.info("Database integrity check passed")
                    return True, "Database integrity check passed"
                else:
                    logger.error(f"Database integrity check failed: {result}")
                    return False, f"Integrity check failed: {result}"
        except Exception as e:
            logger.error(f"Error checking database integrity: {e}")
            return False, f"Error: {e}"

    def backup_exists(self, suffix: str = ".backup") -> bool:
        """Check if a backup file exists.

        Args:
            suffix: Suffix of backup file to check

        Returns:
            True if backup exists
        """
        return Path(self.db_name + suffix).exists()


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for database module."""
    import os
    import tempfile

    from test_framework import TestSuite

    suite = TestSuite("Database Tests")

    def test_story_dataclass():
        story = Story(title="Test", summary="Summary", quality_score=8)
        assert story.title == "Test"
        assert story.quality_score == 8
        assert story.category == "Other"

    def test_story_to_dict():
        story = Story(
            title="Test", summary="Summary", source_links=["http://example.com"]
        )
        d = story.to_dict()
        assert d["title"] == "Test"
        assert "http://example.com" in d["source_links"]

    def test_story_from_row():
        # Test _parse_datetime helper
        result = _parse_datetime("2024-01-01T12:00:00")
        assert result is not None
        assert result.year == 2024
        assert _parse_datetime(None) is None
        assert _parse_datetime("invalid") is None

    def test_database_init():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            Database(db_path)
            assert os.path.exists(db_path)
        finally:
            os.unlink(db_path)

    def test_database_add_story():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            story = Story(title="Test Story", summary="Test summary", quality_score=7)
            story_id = db.add_story(story)
            assert story_id is not None
            assert story_id > 0
        finally:
            os.unlink(db_path)

    def test_database_get_story():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            story = Story(title="Fetch Test", summary="Summary", quality_score=8)
            story_id = db.add_story(story)
            fetched = db.get_story(story_id)
            assert fetched is not None
            assert fetched.title == "Fetch Test"
        finally:
            os.unlink(db_path)

    def test_database_get_story_by_title():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            story = Story(title="Unique Title", summary="Summary", quality_score=7)
            db.add_story(story)
            found = db.get_story_by_title("Unique Title")
            assert found is not None
            assert found.title == "Unique Title"
            not_found = db.get_story_by_title("Nonexistent")
            assert not_found is None
        finally:
            os.unlink(db_path)

    def test_database_update_story():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            story = Story(title="Update Test", summary="Original", quality_score=5)
            story_id = db.add_story(story)
            story.id = story_id
            story.summary = "Updated"
            result = db.update_story(story)
            assert result is True
            fetched = db.get_story(story_id)
            assert fetched is not None
            assert fetched.summary == "Updated"
        finally:
            os.unlink(db_path)

    def test_database_delete_story():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            story = Story(title="Delete Test", summary="Summary", quality_score=5)
            story_id = db.add_story(story)
            result = db.delete_story(story_id)
            assert result is True
            fetched = db.get_story(story_id)
            assert fetched is None
        finally:
            os.unlink(db_path)

    def test_database_statistics():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            stats = db.get_statistics()
            assert "total_stories" in stats
            assert stats["total_stories"] == 0
        finally:
            os.unlink(db_path)

    def test_database_state():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            db.set_state("test_key", "test_value")
            value = db.get_state("test_key")
            assert value == "test_value"
        finally:
            os.unlink(db_path)

    def test_database_published_titles():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            from datetime import datetime

            db = Database(db_path)
            # Add an unpublished story
            story1 = Story(
                title="Unpublished Story",
                summary="Test summary",
                source_links=["https://example.com"],
                quality_score=7,
            )
            story1_id = db.add_story(story1)

            # Add a story that will be published
            story2 = Story(
                title="Published Story",
                summary="Test summary 2",
                source_links=["https://example2.com"],
                quality_score=8,
            )
            story2_id = db.add_story(story2)

            # Mark story2 as published using update_story
            story2_from_db = db.get_story(story2_id)
            assert story2_from_db is not None
            story2_from_db.publish_status = "published"
            story2_from_db.published_time = datetime.now()
            db.update_story(story2_from_db)

            # Get published titles
            published = db.get_recent_published_titles(days=30)
            # Should only return the published story
            assert len(published) == 1
            assert published[0][1] == "Published Story"

            # Get all recent titles
            all_titles = db.get_recent_story_titles(days=90)
            # Should return both stories
            assert len(all_titles) == 2
        finally:
            os.unlink(db_path)

    suite.add_test("Story dataclass", test_story_dataclass)
    suite.add_test("Story to_dict", test_story_to_dict)
    suite.add_test("Story from_row and _parse_datetime", test_story_from_row)
    suite.add_test("Database init", test_database_init)
    suite.add_test("Database add story", test_database_add_story)
    suite.add_test("Database get story", test_database_get_story)
    suite.add_test("Database get story by title", test_database_get_story_by_title)
    suite.add_test("Database update story", test_database_update_story)
    suite.add_test("Database delete story", test_database_delete_story)
    suite.add_test("Database statistics", test_database_statistics)
    suite.add_test("Database state", test_database_state)
    suite.add_test("Database published titles", test_database_published_titles)

    return suite
