"""Database management for Social Media Publisher."""

import sqlite3
import json
import logging
import shutil
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class Story:
    """Represents a story in the database."""

    id: Optional[int] = None
    title: str = ""
    summary: str = ""
    source_links: list[str] = field(default_factory=list)
    acquire_date: Optional[datetime] = None
    quality_score: int = 0
    category: str = "Other"  # Technology, Business, Science, AI, Other
    quality_justification: str = ""  # Reasoning for the quality score
    image_path: Optional[str] = None
    verification_status: str = "pending"  # pending, approved, rejected
    verification_reason: Optional[str] = None  # AI's reason for approval/rejection
    publish_status: str = "unpublished"  # unpublished, scheduled, published
    scheduled_time: Optional[datetime] = None
    published_time: Optional[datetime] = None
    linkedin_post_id: Optional[str] = None
    linkedin_post_url: Optional[str] = None
    # Hashtags for LinkedIn posts (max 3)
    hashtags: list[str] = field(default_factory=list)
    # LinkedIn mentions: list of {"name": "...", "urn": "urn:li:person/organization:...", "type": "person/organization"}
    linkedin_mentions: list[dict] = field(default_factory=list)
    # LinkedIn analytics (fetched from API after publication)
    linkedin_impressions: int = 0
    linkedin_clicks: int = 0
    linkedin_likes: int = 0
    linkedin_comments: int = 0
    linkedin_shares: int = 0
    linkedin_engagement: float = (
        0.0  # Engagement rate (clicks+likes+comments+shares / impressions)
    )
    linkedin_analytics_fetched_at: Optional[datetime] = None
    # --- Enrichment fields (cleaner structure) ---
    enrichment_status: str = "pending"  # pending, enriched, skipped, error
    # Organizations mentioned in the story (just names)
    organizations: list[str] = field(default_factory=list)  # ["BASF", "MIT", "IChemE"]
    # People mentioned directly in the story
    story_people: list[dict] = field(
        default_factory=list
    )  # [{"name": "Dr. Jane Smith", "title": "Lead Researcher", "affiliation": "MIT"}]
    # Key leaders from the organizations (CEO, CTO, etc.)
    org_leaders: list[dict] = field(
        default_factory=list
    )  # [{"name": "John Doe", "title": "CEO", "organization": "BASF"}]
    # LinkedIn handles for @ mentions (consolidated from all people)
    linkedin_handles: list[dict] = field(
        default_factory=list
    )  # [{"name": "Dr. Jane Smith", "handle": "janesmith", "urn": "urn:li:person:123", "url": "https://linkedin.com/in/janesmith"}]
    # Legacy fields (kept for backward compatibility, will be migrated)
    company_mention_enrichment: Optional[str] = None  # DEPRECATED - use organizations
    individuals: list[str] = field(
        default_factory=list
    )  # DEPRECATED - use story_people
    linkedin_profiles: list[dict] = field(
        default_factory=list
    )  # DEPRECATED - use linkedin_handles

    def to_dict(self) -> dict:
        """Convert story to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "source_links": self.source_links,
            "acquire_date": self.acquire_date.isoformat()
            if self.acquire_date
            else None,
            "quality_score": self.quality_score,
            "category": self.category,
            "quality_justification": self.quality_justification,
            "image_path": self.image_path,
            "verification_status": self.verification_status,
            "verification_reason": self.verification_reason,
            "publish_status": self.publish_status,
            "scheduled_time": self.scheduled_time.isoformat()
            if self.scheduled_time
            else None,
            "published_time": self.published_time.isoformat()
            if self.published_time
            else None,
            "linkedin_post_id": self.linkedin_post_id,
            "linkedin_post_url": self.linkedin_post_url,
            "hashtags": self.hashtags,
            "linkedin_mentions": self.linkedin_mentions,
            "linkedin_impressions": self.linkedin_impressions,
            "linkedin_clicks": self.linkedin_clicks,
            "linkedin_likes": self.linkedin_likes,
            "linkedin_comments": self.linkedin_comments,
            "linkedin_shares": self.linkedin_shares,
            "linkedin_engagement": self.linkedin_engagement,
            "linkedin_analytics_fetched_at": self.linkedin_analytics_fetched_at.isoformat()
            if self.linkedin_analytics_fetched_at
            else None,
            "enrichment_status": self.enrichment_status,
            "organizations": self.organizations,
            "story_people": self.story_people,
            "org_leaders": self.org_leaders,
            "linkedin_handles": self.linkedin_handles,
            # Legacy fields (deprecated)
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

        # Parse linkedin_mentions (JSON array of objects)
        linkedin_mentions = []
        if "linkedin_mentions" in keys and row["linkedin_mentions"]:
            try:
                linkedin_mentions = json.loads(row["linkedin_mentions"])
            except json.JSONDecodeError:
                linkedin_mentions = []

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
            linkedin_mentions=linkedin_mentions,
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
            organizations=json.loads(row["organizations"])
            if "organizations" in keys and row["organizations"]
            else [],
            story_people=json.loads(row["story_people"])
            if "story_people" in keys and row["story_people"]
            else [],
            org_leaders=json.loads(row["org_leaders"])
            if "org_leaders" in keys and row["org_leaders"]
            else [],
            linkedin_handles=json.loads(row["linkedin_handles"])
            if "linkedin_handles" in keys and row["linkedin_handles"]
            else [],
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


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse datetime from database value."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class Database:
    """Database manager for story storage and retrieval."""

    def __init__(self, db_name: Optional[str] = None):
        """Initialize database connection."""
        self.db_name = db_name or Config.DB_NAME
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_name)
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
                cursor, "linkedin_handles", "TEXT DEFAULT '[]'", existing_columns
            )

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
                 publish_status, hashtags, linkedin_mentions, company_mention_enrichment,
                 enrichment_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    json.dumps(story.linkedin_mentions),
                    story.company_mention_enrichment,
                    story.enrichment_status,
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
                    verification_status = ?,
                    verification_reason = ?,
                    publish_status = ?,
                    scheduled_time = ?,
                    published_time = ?,
                    linkedin_post_id = ?,
                    linkedin_post_url = ?,
                    hashtags = ?,
                    linkedin_mentions = ?,
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
                    org_leaders = ?,
                    linkedin_handles = ?,
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
                    story.verification_status,
                    story.verification_reason,
                    story.publish_status,
                    story.scheduled_time,
                    story.published_time,
                    story.linkedin_post_id,
                    story.linkedin_post_url,
                    json.dumps(story.hashtags),
                    json.dumps(story.linkedin_mentions),
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
                    json.dumps(story.org_leaders),
                    json.dumps(story.linkedin_handles),
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
                AND publish_status = 'unpublished'
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

        Returns all pending enrichment stories that have been approved and have images.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM stories
                WHERE enrichment_status = 'pending'
                AND verification_status = 'approved'
                AND image_path IS NOT NULL
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
                "SELECT COUNT(*) FROM stories WHERE image_path IS NULL AND publish_status = 'unpublished'"
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

    return suite
