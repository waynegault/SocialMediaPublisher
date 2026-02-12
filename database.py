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
    direct_people: list[dict] = field(default_factory=list)
    # Key leaders from the organizations (CEO, CTO, etc.) - for secondary @mentions
    # [{"name": "John Doe", "title": "CEO", "organization": "BASF", "linkedin_profile": "", "linkedin_urn": ""}]
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
    verification_status: str = (
        "pending"  # pending, approved, rejected (AI recommendation)
    )
    verification_reason: Optional[str] = None  # AI's reason for approval/rejection
    human_approved: bool = False  # True only when human explicitly approves via GUI
    human_approved_at: Optional[datetime] = None  # When human approved

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
            "human_approved": self.human_approved,
            "human_approved_at": self.human_approved_at.isoformat()
            if self.human_approved_at
            else None,
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
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Story":
        """Create a Story from a database row."""

        # Parse JSON fields
        def _parse_json_list(value: str | None) -> list:
            if not value:
                return []
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return [value] if value else []

        def _parse_json_dict(value: str | None) -> dict:
            if not value:
                return {}
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}

        return cls(
            id=row["id"],
            title=row["title"],
            summary=row["summary"],
            source_links=_parse_json_list(row["source_links"]),
            acquire_date=_parse_datetime(row["acquire_date"]),
            quality_score=row["quality_score"],
            category=row["category"],
            quality_justification=row["quality_justification"] or "",
            image_path=row["image_path"],
            image_alt_text=row["image_alt_text"],
            verification_status=row["verification_status"],
            verification_reason=row["verification_reason"],
            human_approved=bool(row["human_approved"])
            if "human_approved" in row.keys()
            else False,
            human_approved_at=_parse_datetime(row["human_approved_at"])
            if "human_approved_at" in row.keys()
            else None,
            publish_status=row["publish_status"],
            scheduled_time=_parse_datetime(row["scheduled_time"]),
            published_time=_parse_datetime(row["published_time"]),
            linkedin_post_id=row["linkedin_post_id"],
            linkedin_post_url=row["linkedin_post_url"],
            hashtags=_parse_json_list(row["hashtags"]),
            linkedin_impressions=row["linkedin_impressions"] or 0,
            linkedin_clicks=row["linkedin_clicks"] or 0,
            linkedin_likes=row["linkedin_likes"] or 0,
            linkedin_comments=row["linkedin_comments"] or 0,
            linkedin_shares=row["linkedin_shares"] or 0,
            linkedin_engagement=row["linkedin_engagement"] or 0.0,
            linkedin_analytics_fetched_at=_parse_datetime(
                row["linkedin_analytics_fetched_at"]
            ),
            enrichment_status=row["enrichment_status"] or "pending",
            organizations=_parse_json_list(row["organizations"]),
            direct_people=_parse_json_list(row["direct_people"]),
            indirect_people=_parse_json_list(row["indirect_people"]),
            enrichment_log=_parse_json_dict(row["enrichment_log"]),
            enrichment_quality=row["enrichment_quality"] or "",
            promotion=row["promotion"],
        )


@dataclass
class Person:
    """Represents a person associated with a story.

    People are linked to stories via the story_people table.
    Each person has LinkedIn profile information and connection tracking.
    """

    # --- Identity ---
    id: Optional[int] = None
    name: str = ""
    title: Optional[str] = None  # Job title (e.g., "Lead Researcher", "CEO")
    organization: Optional[str] = None  # Company/institution they work at

    # --- Enhanced Matching Fields ---
    location: Optional[str] = None  # Location (e.g., "Cambridge, UK")
    specialty: Optional[str] = (
        None  # Field of expertise (e.g., "catalysis", "process safety")
    )
    department: Optional[str] = None  # Department name (e.g., "Chemical Engineering")

    # --- Story Relationship ---
    story_id: Optional[int] = None  # Which story this person is associated with
    relationship_type: str = (
        "direct"  # "direct" (mentioned in story) or "indirect" (org leader)
    )

    # --- LinkedIn Profile ---
    linkedin_profile: Optional[str] = (
        None  # Profile URL (e.g., "https://www.linkedin.com/in/username")
    )
    linkedin_urn: Optional[str] = (
        None  # LinkedIn URN for @mentions (e.g., "urn:li:person:ABC123")
    )

    # --- Connection Tracking ---
    connection_status: str = "none"  # none, pending, connected, failed
    connection_sent_at: Optional[datetime] = None
    connection_message: Optional[str] = None  # The personalized intro message sent

    # --- Metadata ---
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert person to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "title": self.title,
            "organization": self.organization,
            "location": self.location,
            "specialty": self.specialty,
            "department": self.department,
            "story_id": self.story_id,
            "relationship_type": self.relationship_type,
            "linkedin_profile": self.linkedin_profile,
            "linkedin_urn": self.linkedin_urn,
            "connection_status": self.connection_status,
            "connection_sent_at": self.connection_sent_at.isoformat()
            if self.connection_sent_at
            else None,
            "connection_message": self.connection_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Person":
        """Create a Person from a database row."""
        keys = row.keys()
        return cls(
            id=row["id"],
            name=row["name"],
            title=row["title"] if "title" in keys else None,
            organization=row["organization"] if "organization" in keys else None,
            location=row["location"] if "location" in keys else None,
            specialty=row["specialty"] if "specialty" in keys else None,
            department=row["department"] if "department" in keys else None,
            story_id=row["story_id"] if "story_id" in keys else None,
            relationship_type=row["relationship_type"]
            if "relationship_type" in keys
            else "direct",
            linkedin_profile=row["linkedin_profile"]
            if "linkedin_profile" in keys
            else None,
            linkedin_urn=row["linkedin_urn"] if "linkedin_urn" in keys else None,
            connection_status=row["connection_status"]
            if "connection_status" in keys
            else "none",
            connection_sent_at=_parse_datetime(row["connection_sent_at"])
            if "connection_sent_at" in keys
            else None,
            connection_message=row["connection_message"]
            if "connection_message" in keys
            else None,
            created_at=_parse_datetime(row["created_at"])
            if "created_at" in keys
            else None,
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
        table_name: str = "stories",
    ) -> None:
        """Add a column to a table if it doesn't exist.

        Args:
            cursor: Database cursor
            column_name: Name of the column to add
            column_def: Column definition (e.g., 'TEXT DEFAULT NULL')
            existing_columns: Optional pre-fetched set of existing column names
                             (avoids repeated PRAGMA calls)
            table_name: Name of the table to add the column to (default: 'stories')
        """
        # If existing_columns provided, do a quick check first
        if existing_columns is not None:
            if column_name in existing_columns:
                return  # Column already exists, skip

        try:
            cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}"
            )
            logger.debug(f"Added column '{column_name}' to {table_name} table")
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
            # Enrichment columns
            self._migrate_add_column(
                cursor, "enrichment_status", "TEXT DEFAULT 'pending'", existing_columns
            )
            self._migrate_add_column(
                cursor, "organizations", "TEXT DEFAULT '[]'", existing_columns
            )
            self._migrate_add_column(
                cursor, "direct_people", "TEXT DEFAULT '[]'", existing_columns
            )
            self._migrate_add_column(
                cursor, "indirect_people", "TEXT DEFAULT '[]'", existing_columns
            )
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
            # Human approval tracking
            self._migrate_add_column(
                cursor, "human_approved", "INTEGER DEFAULT 0", existing_columns
            )
            self._migrate_add_column(
                cursor, "human_approved_at", "TEXT", existing_columns
            )

            # System state table for tracking last check date, etc.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            # People table for tracking individuals mentioned in stories
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    title TEXT,
                    organization TEXT,
                    story_id INTEGER REFERENCES stories(id) ON DELETE CASCADE,
                    relationship_type TEXT DEFAULT 'direct',
                    linkedin_profile TEXT,
                    linkedin_urn TEXT,
                    connection_status TEXT DEFAULT 'none',
                    connection_sent_at TIMESTAMP,
                    connection_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(linkedin_profile, story_id)
                )
            """)

            # Migrate people table: add new columns if they don't exist
            existing_people_columns = self._get_existing_columns(cursor, "people")
            self._migrate_add_column(
                cursor, "location", "TEXT", existing_people_columns, "people"
            )
            self._migrate_add_column(
                cursor, "specialty", "TEXT", existing_people_columns, "people"
            )
            self._migrate_add_column(
                cursor, "department", "TEXT", existing_people_columns, "people"
            )

            # Connection queue table for batch connection requests
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS connection_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER REFERENCES people(id) ON DELETE CASCADE,
                    linkedin_profile TEXT NOT NULL,
                    name TEXT NOT NULL,
                    title TEXT,
                    organization TEXT,
                    story_id INTEGER,
                    story_title TEXT,
                    story_summary TEXT,
                    message TEXT,
                    priority INTEGER DEFAULT 5,
                    status TEXT DEFAULT 'queued',
                    queued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sent_at TIMESTAMP,
                    error_message TEXT,
                    UNIQUE(linkedin_profile)
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

            # Indices for people table
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_people_story
                ON people(story_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_people_linkedin_profile
                ON people(linkedin_profile)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_people_connection_status
                ON people(connection_status)
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
                 publish_status, hashtags, enrichment_status, direct_people,
                 organizations, indirect_people, enrichment_log, enrichment_quality,
                 promotion)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    story.enrichment_status,
                    json.dumps(story.direct_people),
                    json.dumps(story.organizations),
                    json.dumps(story.indirect_people),
                    json.dumps(story.enrichment_log),
                    story.enrichment_quality,
                    story.promotion,
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
                    human_approved = ?,
                    human_approved_at = ?,
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
                    direct_people = ?,
                    indirect_people = ?,
                    enrichment_log = ?,
                    enrichment_quality = ?,
                    promotion = ?
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
                    1 if story.human_approved else 0,
                    story.human_approved_at,
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
                    json.dumps(story.direct_people),
                    json.dumps(story.indirect_people),
                    json.dumps(story.enrichment_log),
                    story.enrichment_quality,
                    story.promotion,
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

    def get_stories_needing_promotion(self, require_image: bool = False) -> list[Story]:
        """Get stories that need a promotion message assigned.

        Args:
            require_image: If True, only return stories with images

        Returns:
            List of stories needing promotion messages
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if require_image:
                cursor.execute("""
                    SELECT * FROM stories
                    WHERE (promotion IS NULL OR promotion = '')
                      AND image_path IS NOT NULL
                    ORDER BY id DESC
                """)
            else:
                cursor.execute("""
                    SELECT * FROM stories
                    WHERE promotion IS NULL OR promotion = ''
                    ORDER BY id DESC
                """)
            return [Story.from_row(row) for row in cursor.fetchall()]

    def get_stories_needing_profile_lookup(self) -> list[Story]:
        """Get stories with people who need LinkedIn profile lookup.

        Returns stories where direct_people or indirect_people exist
        but some entries are missing linkedin_profile URLs.
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM stories
                WHERE (direct_people IS NOT NULL AND direct_people != '[]')
                   OR (indirect_people IS NOT NULL AND indirect_people != '[]')
            """)
            stories_needing_profiles = []
            for row in cursor.fetchall():
                story = Story.from_row(row)
                all_people = story.direct_people + story.indirect_people
                # Check if any people need profiles
                needs_lookup = any(
                    not (p.get("linkedin_profile") or "").strip() for p in all_people
                )
                if needs_lookup:
                    stories_needing_profiles.append(story)
            return stories_needing_profiles

    def get_stories_needing_urn_extraction(self) -> list[tuple[list, list]]:
        """Get people entries that need URN extraction.

        Returns list of (direct_people, indirect_people) tuples for stories
        where people have linkedin_profile but no linkedin_urn.
        """
        import json as json_module

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT direct_people, indirect_people FROM stories
                WHERE (direct_people LIKE '%linkedin_profile%')
                   OR (indirect_people LIKE '%linkedin_profile%')
            """)
            results = []
            for row in cursor.fetchall():
                direct = (
                    json_module.loads(row["direct_people"])
                    if row["direct_people"]
                    else []
                )
                indirect = (
                    json_module.loads(row["indirect_people"])
                    if row["indirect_people"]
                    else []
                )
                results.append((direct, indirect))
            return results

    def count_people_needing_urns(self) -> int:
        """Count people entries that have linkedin_profile but no linkedin_urn."""
        results = self.get_stories_needing_urn_extraction()
        count = 0
        for direct, indirect in results:
            for p in direct + indirect:
                if p.get("linkedin_profile") and not p.get("linkedin_urn"):
                    count += 1
        return count

    def get_stories_needing_indirect_people(self) -> list[Story]:
        """Get stories with organizations but no indirect_people.

        Returns stories that have organizations extracted but haven't
        had indirect people (org leadership) looked up yet.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM stories
                WHERE organizations IS NOT NULL
                AND organizations != '[]'
                AND (indirect_people IS NULL OR indirect_people = '[]')
            """)
            return [Story.from_row(row) for row in cursor.fetchall()]

    def update_story_indirect_people(
        self, story_id: int, indirect_people: list
    ) -> None:
        """Update a story's indirect_people field.

        Args:
            story_id: ID of the story to update
            indirect_people: List of indirect people dicts
        """
        import json as json_module

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE stories SET indirect_people = ? WHERE id = ?",
                (json_module.dumps(indirect_people), story_id),
            )

    def update_story_promotion(self, story_id: int, promotion: str) -> None:
        """Update a story's promotion message.

        Args:
            story_id: ID of the story to update
            promotion: Promotion message text
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE stories SET promotion = ? WHERE id = ?",
                (promotion, story_id),
            )

    def mark_stories_enriched(self) -> int:
        """Mark stories as enriched after all enrichment steps complete.

        Updates stories from 'pending' to 'enriched' if they have
        direct_people or indirect_people populated.

        Returns:
            Count of stories updated
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE stories
                SET enrichment_status = 'enriched'
                WHERE enrichment_status = 'pending'
                AND (
                    (direct_people IS NOT NULL AND direct_people != '[]')
                    OR (indirect_people IS NOT NULL AND indirect_people != '[]')
                )
            """)
            return cursor.rowcount

    def get_recent_stories(self, limit: int = 100) -> list[tuple[int, str, int]]:
        """Get recent stories with basic info for listing.

        Returns:
            List of (id, title, quality_score) tuples
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, quality_score FROM stories ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [
                (row["id"], row["title"], row["quality_score"])
                for row in cursor.fetchall()
            ]

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
        """Get HUMAN-approved stories that haven't been published.

        Only returns stories where human_approved=True, not just AI-recommended.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM stories
                WHERE human_approved = 1
                AND publish_status = 'unpublished'
                ORDER BY quality_score DESC, acquire_date DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [Story.from_row(row) for row in cursor.fetchall()]

    def get_ai_recommended_stories(self, limit: int = 100) -> list[Story]:
        """Get AI-recommended stories awaiting human review.

        Returns stories where AI approved but human hasn't reviewed yet.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM stories
                WHERE verification_status = 'approved'
                AND human_approved = 0
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
        """Get stories that are scheduled, human-approved, and due for publishing."""
        now = datetime.now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM stories
                WHERE publish_status = 'scheduled'
                AND human_approved = 1
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

            # Count stories with organizations (replaces company_mention_enrichment)
            cursor.execute(
                "SELECT COUNT(*) FROM stories WHERE enrichment_status = 'enriched' AND organizations != '[]' AND organizations IS NOT NULL"
            )
            stats["with_organizations"] = cursor.fetchone()[0]

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
            # Search direct_people and indirect_people
            cursor.execute(
                """
                SELECT direct_people, indirect_people FROM stories
                WHERE direct_people LIKE ? OR indirect_people LIKE ?
            """,
                (f"%{urn}%", f"%{urn}%"),
            )

            for row in cursor.fetchall():
                # Check direct_people
                if row["direct_people"]:
                    people = json.loads(row["direct_people"])
                    for person in people:
                        if person.get("linkedin_urn") == urn:
                            return person

                # Check indirect_people
                if row["indirect_people"]:
                    leaders = json.loads(row["indirect_people"])
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
            cursor.execute("SELECT direct_people, indirect_people FROM stories")

            best_match: dict | None = None
            best_score = 0

            for row in cursor.fetchall():
                for field_name in ["direct_people", "indirect_people"]:
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
            cursor.execute("SELECT direct_people, indirect_people FROM stories")
            total_direct = 0
            total_indirect = 0
            with_linkedin = 0
            high_confidence = 0
            org_fallback = 0

            for row in cursor.fetchall():
                if row["direct_people"]:
                    people = json.loads(row["direct_people"])
                    total_direct += len(people)
                    for p in people:
                        if p.get("linkedin_profile") or p.get("linkedin_urn"):
                            with_linkedin += 1
                        conf = p.get("match_confidence", "")
                        if conf == "high":
                            high_confidence += 1
                        elif conf == "org_fallback":
                            org_fallback += 1

                if row["indirect_people"]:
                    leaders = json.loads(row["indirect_people"])
                    total_indirect += len(leaders)
                    for leader in leaders:
                        if leader.get("linkedin_profile") or leader.get("linkedin_urn"):
                            with_linkedin += 1
                        conf = leader.get("match_confidence", "")
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
        resolved_path: str = (
            backup_path if backup_path is not None else self.db_name + ".backup"
        )

        # Type narrowing: resolved_path is now guaranteed to be str
        backup_file = Path(resolved_path)
        if not backup_file.exists():
            logger.error(f"Backup file not found: {resolved_path}")
            return False

        try:
            shutil.copy2(backup_file, self.db_name)
            logger.info(f"Restored database from: {resolved_path}")
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

    # ==========================================================================
    # PEOPLE CRUD OPERATIONS
    # ==========================================================================

    def add_person(self, person: Person) -> int:
        """Add a new person to the database. Returns the person ID.

        If a person with the same linkedin_profile and story_id already exists,
        returns the existing person's ID instead of creating a duplicate.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check for existing person with same linkedin_profile and story_id
            if person.linkedin_profile and person.story_id:
                cursor.execute(
                    """
                    SELECT id FROM people
                    WHERE linkedin_profile = ? AND story_id = ?
                    """,
                    (person.linkedin_profile, person.story_id),
                )
                existing = cursor.fetchone()
                if existing:
                    return existing["id"]

            cursor.execute(
                """
                INSERT INTO people
                (name, title, organization, location, specialty, department,
                 story_id, relationship_type,
                 linkedin_profile, linkedin_urn, connection_status,
                 connection_sent_at, connection_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person.name,
                    person.title,
                    person.organization,
                    person.location,
                    person.specialty,
                    person.department,
                    person.story_id,
                    person.relationship_type,
                    person.linkedin_profile,
                    person.linkedin_urn,
                    person.connection_status,
                    person.connection_sent_at,
                    person.connection_message,
                    person.created_at or datetime.now(),
                ),
            )
            return cursor.lastrowid or 0

    def get_person(self, person_id: int) -> Optional[Person]:
        """Get a person by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM people WHERE id = ?", (person_id,))
            row = cursor.fetchone()
            return Person.from_row(row) if row else None

    def get_person_by_linkedin_profile(self, linkedin_profile: str) -> Optional[Person]:
        """Get a person by LinkedIn profile URL.

        Returns the first matching person (may have multiple across stories).
        """
        if not linkedin_profile:
            return None
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM people WHERE linkedin_profile = ? LIMIT 1",
                (linkedin_profile,),
            )
            row = cursor.fetchone()
            return Person.from_row(row) if row else None

    def get_people_for_story(self, story_id: int) -> list[Person]:
        """Get all people associated with a story."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM people WHERE story_id = ? ORDER BY relationship_type, name",
                (story_id,),
            )
            return [Person.from_row(row) for row in cursor.fetchall()]

    def get_direct_people_for_story(self, story_id: int) -> list[Person]:
        """Get direct people (mentioned in story) for a story."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM people
                WHERE story_id = ? AND relationship_type = 'direct'
                ORDER BY name
                """,
                (story_id,),
            )
            return [Person.from_row(row) for row in cursor.fetchall()]

    def get_indirect_people_for_story(self, story_id: int) -> list[Person]:
        """Get indirect people (org leaders) for a story."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM people
                WHERE story_id = ? AND relationship_type = 'indirect'
                ORDER BY name
                """,
                (story_id,),
            )
            return [Person.from_row(row) for row in cursor.fetchall()]

    def update_person(self, person: Person) -> bool:
        """Update an existing person."""
        if person.id is None:
            return False

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE people SET
                    name = ?,
                    title = ?,
                    organization = ?,
                    story_id = ?,
                    relationship_type = ?,
                    linkedin_profile = ?,
                    linkedin_urn = ?,
                    connection_status = ?,
                    connection_sent_at = ?,
                    connection_message = ?
                WHERE id = ?
                """,
                (
                    person.name,
                    person.title,
                    person.organization,
                    person.story_id,
                    person.relationship_type,
                    person.linkedin_profile,
                    person.linkedin_urn,
                    person.connection_status,
                    person.connection_sent_at,
                    person.connection_message,
                    person.id,
                ),
            )
            return cursor.rowcount > 0

    def delete_person(self, person_id: int) -> bool:
        """Delete a person by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM people WHERE id = ?", (person_id,))
            return cursor.rowcount > 0

    def get_people_needing_connection(self) -> list[Person]:
        """Get all people with LinkedIn profiles who haven't been connected yet."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM people
                WHERE linkedin_profile IS NOT NULL
                  AND linkedin_profile != ''
                  AND connection_status = 'none'
                ORDER BY created_at DESC
                """
            )
            return [Person.from_row(row) for row in cursor.fetchall()]

    def get_people_with_pending_connections(self) -> list[Person]:
        """Get all people with pending connection requests."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM people
                WHERE connection_status = 'pending'
                ORDER BY connection_sent_at DESC
                """
            )
            return [Person.from_row(row) for row in cursor.fetchall()]

    def get_unique_people_with_profiles(self) -> list[Person]:
        """Get unique people by LinkedIn profile URL (deduped across stories).

        Returns one Person record per unique linkedin_profile, preferring
        the most recent record.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT p.* FROM people p
                INNER JOIN (
                    SELECT linkedin_profile, MAX(id) as max_id
                    FROM people
                    WHERE linkedin_profile IS NOT NULL AND linkedin_profile != ''
                    GROUP BY linkedin_profile
                ) latest ON p.id = latest.max_id
                ORDER BY p.name
                """
            )
            return [Person.from_row(row) for row in cursor.fetchall()]

    def mark_connection_sent(
        self, person_id: int, message: str, status: str = "pending"
    ) -> bool:
        """Mark a connection request as sent for a person.

        Args:
            person_id: The person ID
            message: The connection message that was sent
            status: The connection status (default: "pending")

        Returns:
            True if update succeeded
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE people SET
                    connection_status = ?,
                    connection_sent_at = ?,
                    connection_message = ?
                WHERE id = ?
                """,
                (status, datetime.now(), message, person_id),
            )
            return cursor.rowcount > 0

    def update_connection_status(self, person_id: int, status: str) -> bool:
        """Update the connection status for a person.

        Args:
            person_id: The person ID
            status: New status (none, pending, connected, failed)

        Returns:
            True if update succeeded
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE people SET connection_status = ? WHERE id = ?",
                (status, person_id),
            )
            return cursor.rowcount > 0

    def get_people_stats(self) -> dict:
        """Get statistics about people in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            stats = {}

            cursor.execute("SELECT COUNT(*) FROM people")
            stats["total"] = cursor.fetchone()[
                0
            ]  # Fixed: key was 'total_people' but display used 'total'

            cursor.execute(
                "SELECT COUNT(*) FROM people WHERE relationship_type = 'direct'"
            )
            stats["direct_count"] = cursor.fetchone()[0]  # Fixed: key to match display

            cursor.execute(
                "SELECT COUNT(*) FROM people WHERE relationship_type = 'indirect'"
            )
            stats["indirect_count"] = cursor.fetchone()[
                0
            ]  # Fixed: key to match display

            cursor.execute(
                "SELECT COUNT(*) FROM people WHERE linkedin_profile IS NOT NULL AND linkedin_profile != ''"
            )
            stats["with_linkedin"] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM people WHERE connection_status = 'none' AND linkedin_profile IS NOT NULL"
            )
            stats["awaiting_connection"] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM people WHERE connection_status = 'pending'"
            )
            stats["pending_connections"] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM people WHERE connection_status = 'connected'"
            )
            stats["connected"] = cursor.fetchone()[0]

            # Count connections actually sent (pending + connected)
            stats["connections_sent"] = (
                stats["pending_connections"] + stats["connected"]
            )

            cursor.execute(
                "SELECT COUNT(DISTINCT linkedin_profile) FROM people WHERE linkedin_profile IS NOT NULL"
            )
            stats["unique_profiles"] = cursor.fetchone()[0]

            return stats

    # ==========================================================================
    # CONNECTION QUEUE OPERATIONS
    # ==========================================================================

    def queue_connection_request(
        self,
        person_id: Optional[int],
        linkedin_profile: str,
        name: str,
        title: str = "",
        organization: str = "",
        story_id: Optional[int] = None,
        story_title: str = "",
        story_summary: str = "",
        message: str = "",
        priority: int = 5,
    ) -> int:
        """Add a person to the connection request queue.

        Args:
            person_id: Optional ID from people table
            linkedin_profile: LinkedIn profile URL (required)
            name: Person's name (required)
            title: Job title
            organization: Company/institution
            story_id: Related story ID
            story_title: Story title for message personalization
            story_summary: Story summary for context
            message: Pre-generated connection message
            priority: Queue priority (1=highest, 10=lowest)

        Returns:
            Queue entry ID, or 0 if already queued
        """
        if not linkedin_profile or not name:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if already queued
            cursor.execute(
                "SELECT id FROM connection_queue WHERE linkedin_profile = ?",
                (linkedin_profile,),
            )
            existing = cursor.fetchone()
            if existing:
                return existing["id"]

            cursor.execute(
                """
                INSERT INTO connection_queue
                (person_id, linkedin_profile, name, title, organization,
                 story_id, story_title, story_summary, message, priority, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued')
                """,
                (
                    person_id,
                    linkedin_profile,
                    name,
                    title,
                    organization,
                    story_id,
                    story_title,
                    story_summary[:500] if story_summary else "",
                    message,
                    priority,
                ),
            )
            queue_id = cursor.lastrowid or 0
            logger.debug(f"Queued connection request for {name}: ID {queue_id}")
            return queue_id

    def get_queued_connections(
        self, limit: int = 20, status: str = "queued"
    ) -> list[dict]:
        """Get connection requests from the queue.

        Args:
            limit: Maximum number to return
            status: Filter by status ('queued', 'sent', 'failed', 'all')

        Returns:
            List of queue entry dicts
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if status == "all":
                cursor.execute(
                    """
                    SELECT * FROM connection_queue
                    ORDER BY priority ASC, queued_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM connection_queue
                    WHERE status = ?
                    ORDER BY priority ASC, queued_at ASC
                    LIMIT ?
                    """,
                    (status, limit),
                )

            return [dict(row) for row in cursor.fetchall()]

    def update_queue_status(
        self,
        queue_id: int,
        status: str,
        error_message: str = "",
    ) -> bool:
        """Update the status of a queued connection request.

        Args:
            queue_id: Queue entry ID
            status: New status ('sent', 'failed', 'skipped')
            error_message: Error message if failed

        Returns:
            True if updated
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if status == "sent":
                cursor.execute(
                    """
                    UPDATE connection_queue
                    SET status = ?, sent_at = ?, error_message = NULL
                    WHERE id = ?
                    """,
                    (status, datetime.now(), queue_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE connection_queue
                    SET status = ?, error_message = ?
                    WHERE id = ?
                    """,
                    (status, error_message, queue_id),
                )

            return cursor.rowcount > 0

    def remove_from_queue(self, queue_id: int) -> bool:
        """Remove an entry from the connection queue.

        Args:
            queue_id: Queue entry ID

        Returns:
            True if deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM connection_queue WHERE id = ?", (queue_id,))
            return cursor.rowcount > 0

    def clear_queue(self, status: Optional[str] = None) -> int:
        """Clear the connection queue.

        Args:
            status: If provided, only clear entries with this status.
                    If None, clear all entries.

        Returns:
            Number of entries deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    "DELETE FROM connection_queue WHERE status = ?", (status,)
                )
            else:
                cursor.execute("DELETE FROM connection_queue")
            return cursor.rowcount

    def get_queue_stats(self) -> dict:
        """Get connection queue statistics.

        Returns:
            Dict with queue statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            stats = {}

            cursor.execute("SELECT COUNT(*) FROM connection_queue")
            stats["total_queued"] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM connection_queue WHERE status = 'queued'"
            )
            stats["pending"] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM connection_queue WHERE status = 'sent'"
            )
            stats["sent"] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM connection_queue WHERE status = 'failed'"
            )
            stats["failed"] = cursor.fetchone()[0]

            # Get priority breakdown
            cursor.execute("""
                SELECT priority, COUNT(*) as cnt
                FROM connection_queue
                WHERE status = 'queued'
                GROUP BY priority
                ORDER BY priority
            """)
            stats["by_priority"] = {
                row["priority"]: row["cnt"] for row in cursor.fetchall()
            }

            return stats

    def get_connection_history(self, days: int = 30, limit: int = 100) -> list[dict]:
        """Get connection request history for dashboard.

        Args:
            days: Number of days to look back
            limit: Maximum number of records

        Returns:
            List of connection history records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    p.name,
                    p.title,
                    p.organization,
                    p.linkedin_profile,
                    p.connection_status,
                    p.connection_sent_at,
                    p.connection_message,
                    s.title as story_title
                FROM people p
                LEFT JOIN stories s ON p.story_id = s.id
                WHERE p.connection_status != 'none'
                  AND p.connection_sent_at >= datetime('now', '-' || ? || ' days')
                ORDER BY p.connection_sent_at DESC
                LIMIT ?
                """,
                (days, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_connection_acceptance_rate(self, days: int = 30) -> dict:
        """Calculate connection acceptance rate over a period.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with acceptance statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {
                "total_sent": 0,
                "accepted": 0,
                "pending": 0,
                "failed": 0,
                "acceptance_rate": 0.0,
            }

            cursor.execute(
                """
                SELECT connection_status, COUNT(*) as cnt
                FROM people
                WHERE connection_status != 'none'
                  AND connection_sent_at >= datetime('now', '-' || ? || ' days')
                GROUP BY connection_status
                """,
                (days,),
            )

            for row in cursor.fetchall():
                status = row["connection_status"]
                count = row["cnt"]
                stats["total_sent"] += count

                if status == "connected":
                    stats["accepted"] = count
                elif status == "pending":
                    stats["pending"] = count
                elif status == "failed":
                    stats["failed"] = count

            # Calculate rate (excluding pending)
            responded = stats["accepted"] + stats["failed"]
            if responded > 0:
                stats["acceptance_rate"] = stats["accepted"] / responded

            return stats


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests() -> bool:
    """Create unit tests for database module."""
    import os
    import tempfile

    from test_framework import TestSuite

    suite = TestSuite("Database Tests", "database.py")
    suite.start_suite()

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
            _story1_id = db.add_story(story1)

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

    suite.run_test(
        test_name="Story dataclass",
        test_func=test_story_dataclass,
        test_summary="Tests Story dataclass functionality",
        method_description="Calls Story and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Story to_dict",
        test_func=test_story_to_dict,
        test_summary="Tests Story to dict functionality",
        method_description="Calls Story and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Story from_row and _parse_datetime",
        test_func=test_story_from_row,
        test_summary="Tests Story from row and parse datetime functionality",
        method_description="Calls  parse datetime and verifies the result",
        expected_outcome="Function returns None as expected",
    )
    suite.run_test(
        test_name="Database init",
        test_func=test_database_init,
        test_summary="Tests Database init functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Database add story",
        test_func=test_database_add_story,
        test_summary="Tests Database add story functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Database get story",
        test_func=test_database_get_story,
        test_summary="Tests Database get story functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Database get story by title",
        test_func=test_database_get_story_by_title,
        test_summary="Tests Database get story by title functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns None as expected",
    )
    suite.run_test(
        test_name="Database update story",
        test_func=test_database_update_story,
        test_summary="Tests Database update story functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns True",
    )
    suite.run_test(
        test_name="Database delete story",
        test_func=test_database_delete_story,
        test_summary="Tests Database delete story functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function returns None as expected",
    )
    suite.run_test(
        test_name="Database statistics",
        test_func=test_database_statistics,
        test_summary="Tests Database statistics functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Database state",
        test_func=test_database_state,
        test_summary="Tests Database state functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Database published titles",
        test_func=test_database_published_titles,
        test_summary="Tests Database published titles functionality",
        method_description="Calls NamedTemporaryFile and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )

    return suite.finish_suite()