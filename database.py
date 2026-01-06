"""Database management for Social Media Publisher."""

import sqlite3
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
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
    publish_status: str = "unpublished"  # unpublished, scheduled, published
    scheduled_time: Optional[datetime] = None
    published_time: Optional[datetime] = None
    linkedin_post_id: Optional[str] = None

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
            "publish_status": self.publish_status,
            "scheduled_time": self.scheduled_time.isoformat()
            if self.scheduled_time
            else None,
            "published_time": self.published_time.isoformat()
            if self.published_time
            else None,
            "linkedin_post_id": self.linkedin_post_id,
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
            publish_status=row["publish_status"],
            scheduled_time=_parse_datetime(row["scheduled_time"]),
            published_time=_parse_datetime(row["published_time"]),
            linkedin_post_id=row["linkedin_post_id"],
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

    def _migrate_add_column(
        self, cursor: sqlite3.Cursor, column_name: str, column_def: str
    ) -> None:
        """Add a column to the stories table if it doesn't exist."""
        try:
            cursor.execute(f"ALTER TABLE stories ADD COLUMN {column_name} {column_def}")
            logger.info(f"Added column '{column_name}' to stories table")
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
                    publish_status TEXT DEFAULT 'unpublished',
                    scheduled_time TIMESTAMP,
                    published_time TIMESTAMP,
                    linkedin_post_id TEXT
                )
            """)

            # Migrate existing databases: add new columns if they don't exist
            self._migrate_add_column(cursor, "category", "TEXT DEFAULT 'Other'")
            self._migrate_add_column(cursor, "quality_justification", "TEXT DEFAULT ''")

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

            logger.info("Database initialized successfully")

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
                 publish_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def get_all_story_titles(self) -> list[tuple[int, str]]:
        """Get all story IDs and titles for semantic deduplication."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title FROM stories")
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
                    publish_status = ?,
                    scheduled_time = ?,
                    published_time = ?,
                    linkedin_post_id = ?
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
                    story.publish_status,
                    story.scheduled_time,
                    story.published_time,
                    story.linkedin_post_id,
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
        """Get stories that need content verification."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM stories
                WHERE verification_status = 'pending'
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

            return stats
