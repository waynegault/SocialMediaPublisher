"""Database migration system for Social Media Publisher.

This module provides a simple, version-controlled database migration system
for managing schema evolution without requiring external dependencies like Alembic.

Features:
- Versioned migrations with up/down methods
- Migration history tracking
- Rollback capability
- Dry-run mode for testing
- Migration validation

Example:
    migrator = Migrator(db_path="content_engine.db")
    migrator.migrate()  # Apply all pending migrations
    migrator.rollback(steps=1)  # Rollback last migration
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MigrationRecord:
    """Record of an applied migration."""

    version: int
    name: str
    applied_at: datetime
    checksum: str
    success: bool
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "name": self.name,
            "applied_at": self.applied_at.isoformat(),
            "checksum": self.checksum,
            "success": self.success,
            "duration_ms": self.duration_ms,
        }


@dataclass
class MigrationResult:
    """Result of running migrations."""

    applied: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    dry_run: bool = False
    success: bool = True
    error_message: str = ""

    @property
    def total_applied(self) -> int:
        """Number of migrations applied."""
        return len(self.applied)

    @property
    def total_failed(self) -> int:
        """Number of migrations failed."""
        return len(self.failed)


# =============================================================================
# Migration Base Class
# =============================================================================


class Migration(ABC):
    """Base class for database migrations."""

    # Override in subclass
    version: int = 0
    name: str = "Base Migration"
    description: str = ""

    @abstractmethod
    def up(self, cursor: sqlite3.Cursor) -> None:
        """Apply the migration.

        Args:
            cursor: Database cursor for executing SQL
        """
        pass

    @abstractmethod
    def down(self, cursor: sqlite3.Cursor) -> None:
        """Rollback the migration.

        Args:
            cursor: Database cursor for executing SQL
        """
        pass

    def get_checksum(self) -> str:
        """Get a checksum for this migration's code."""
        import inspect

        source = inspect.getsource(self.__class__)
        return hashlib.md5(source.encode()).hexdigest()[:12]

    def __lt__(self, other: Migration) -> bool:
        """Sort migrations by version."""
        return self.version < other.version


# =============================================================================
# Built-in Migrations
# =============================================================================


class Migration001AddAnalyticsColumns(Migration):
    """Add analytics columns to stories table."""

    version = 1
    name = "add_analytics_columns"
    description = "Add impression and engagement tracking columns"

    def up(self, cursor: sqlite3.Cursor) -> None:
        """Add analytics columns."""
        # Check if columns exist first
        cursor.execute("PRAGMA table_info(stories)")
        columns = {row[1] for row in cursor.fetchall()}

        if "impressions" not in columns:
            cursor.execute(
                "ALTER TABLE stories ADD COLUMN impressions INTEGER DEFAULT 0"
            )
        if "engagement_count" not in columns:
            cursor.execute(
                "ALTER TABLE stories ADD COLUMN engagement_count INTEGER DEFAULT 0"
            )
        if "last_analytics_update" not in columns:
            cursor.execute("ALTER TABLE stories ADD COLUMN last_analytics_update TEXT")

    def down(self, cursor: sqlite3.Cursor) -> None:
        """Remove analytics columns (SQLite doesn't support DROP COLUMN easily)."""
        # SQLite doesn't support DROP COLUMN before 3.35.0
        # We create a new table without the columns and copy data
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS stories_backup AS
            SELECT id, title, url, summary, quality_score, source, published_at,
                   post_id, image_path, linkedin_url, search_prompt, created_at
            FROM stories
        """
        )
        cursor.execute("DROP TABLE IF EXISTS stories")
        cursor.execute("ALTER TABLE stories_backup RENAME TO stories")


class Migration002AddPostMetadata(Migration):
    """Add post metadata tracking."""

    version = 2
    name = "add_post_metadata"
    description = "Add columns for post format and variant tracking"

    def up(self, cursor: sqlite3.Cursor) -> None:
        """Add metadata columns."""
        cursor.execute("PRAGMA table_info(stories)")
        columns = {row[1] for row in cursor.fetchall()}

        if "post_format" not in columns:
            cursor.execute("ALTER TABLE stories ADD COLUMN post_format TEXT")
        if "ab_variant_id" not in columns:
            cursor.execute("ALTER TABLE stories ADD COLUMN ab_variant_id TEXT")
        if "style_preset" not in columns:
            cursor.execute("ALTER TABLE stories ADD COLUMN style_preset TEXT")

    def down(self, cursor: sqlite3.Cursor) -> None:
        """Remove metadata columns."""
        pass  # Keep columns on rollback for data safety


class Migration003AddContentHash(Migration):
    """Add content hash for deduplication."""

    version = 3
    name = "add_content_hash"
    description = "Add hash column for content deduplication"

    def up(self, cursor: sqlite3.Cursor) -> None:
        """Add hash column."""
        cursor.execute("PRAGMA table_info(stories)")
        columns = {row[1] for row in cursor.fetchall()}

        if "content_hash" not in columns:
            cursor.execute("ALTER TABLE stories ADD COLUMN content_hash TEXT")
            # Create index for fast lookups
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_content_hash ON stories(content_hash)"
            )

    def down(self, cursor: sqlite3.Cursor) -> None:
        """Remove hash column and index."""
        cursor.execute("DROP INDEX IF EXISTS idx_content_hash")


class Migration004AddSourceCredibility(Migration):
    """Add source credibility tracking."""

    version = 4
    name = "add_source_credibility"
    description = "Add columns for source verification and credibility"

    def up(self, cursor: sqlite3.Cursor) -> None:
        """Add credibility columns."""
        cursor.execute("PRAGMA table_info(stories)")
        columns = {row[1] for row in cursor.fetchall()}

        if "source_tier" not in columns:
            cursor.execute("ALTER TABLE stories ADD COLUMN source_tier INTEGER")
        if "verified_sources" not in columns:
            cursor.execute("ALTER TABLE stories ADD COLUMN verified_sources TEXT")
        if "originality_score" not in columns:
            cursor.execute(
                "ALTER TABLE stories ADD COLUMN originality_score REAL DEFAULT 1.0"
            )

    def down(self, cursor: sqlite3.Cursor) -> None:
        """Remove credibility columns."""
        pass  # Keep for data safety


class Migration005AddSchedulingInfo(Migration):
    """Add scheduling metadata."""

    version = 5
    name = "add_scheduling_info"
    description = "Add columns for advanced scheduling"

    def up(self, cursor: sqlite3.Cursor) -> None:
        """Add scheduling columns."""
        cursor.execute("PRAGMA table_info(stories)")
        columns = {row[1] for row in cursor.fetchall()}

        if "scheduled_for" not in columns:
            cursor.execute("ALTER TABLE stories ADD COLUMN scheduled_for TEXT")
        if "publish_priority" not in columns:
            cursor.execute(
                "ALTER TABLE stories ADD COLUMN publish_priority INTEGER DEFAULT 0"
            )
        if "retry_count" not in columns:
            cursor.execute(
                "ALTER TABLE stories ADD COLUMN retry_count INTEGER DEFAULT 0"
            )

    def down(self, cursor: sqlite3.Cursor) -> None:
        """Remove scheduling columns."""
        pass  # Keep for data safety


class Migration006AddHumanApproval(Migration):
    """Add human approval tracking - posts can only be published after human approval."""

    version = 6
    name = "add_human_approval"
    description = "Add human_approved flag to enforce human review before publishing"

    def up(self, cursor: sqlite3.Cursor) -> None:
        """Add human approval columns."""
        cursor.execute("PRAGMA table_info(stories)")
        columns = {row[1] for row in cursor.fetchall()}

        if "human_approved" not in columns:
            cursor.execute(
                "ALTER TABLE stories ADD COLUMN human_approved INTEGER DEFAULT 0"
            )
        if "human_approved_at" not in columns:
            cursor.execute("ALTER TABLE stories ADD COLUMN human_approved_at TEXT")

    def down(self, cursor: sqlite3.Cursor) -> None:
        """Remove human approval columns."""
        pass  # Keep for data safety


# =============================================================================
# Migrator Class
# =============================================================================


class Migrator:
    """Database migration manager."""

    # Register all migrations here
    MIGRATIONS: list[type[Migration]] = [
        Migration001AddAnalyticsColumns,
        Migration002AddPostMetadata,
        Migration003AddContentHash,
        Migration004AddSourceCredibility,
        Migration005AddSchedulingInfo,
        Migration006AddHumanApproval,
    ]

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize migrator.

        Args:
            db_path: Path to SQLite database. If None, uses in-memory DB.
        """
        self.db_path = db_path or ":memory:"
        self._conn: sqlite3.Connection | None = None
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if self.db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:")
                self._conn.row_factory = sqlite3.Row
            return self._conn
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn

    def _close_connection(self, conn: sqlite3.Connection) -> None:
        """Close connection if not in-memory."""
        if self.db_path != ":memory:":
            conn.close()

    def _init_schema(self) -> None:
        """Initialize migration tracking table."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS _migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL,
                checksum TEXT NOT NULL,
                success INTEGER NOT NULL,
                duration_ms REAL DEFAULT 0.0
            )
        """
        )

        conn.commit()
        self._close_connection(conn)

    def get_applied_versions(self) -> set[int]:
        """Get set of applied migration versions.

        Returns:
            Set of version numbers that have been applied
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT version FROM _migrations WHERE success = 1")
        versions = {row["version"] for row in cursor.fetchall()}

        self._close_connection(conn)
        return versions

    def get_current_version(self) -> int:
        """Get the current schema version.

        Returns:
            Highest applied migration version, or 0 if none
        """
        versions = self.get_applied_versions()
        return max(versions) if versions else 0

    def get_pending_migrations(self) -> list[Migration]:
        """Get list of migrations that haven't been applied.

        Returns:
            List of pending Migration instances
        """
        applied = self.get_applied_versions()
        pending = []

        for migration_class in self.MIGRATIONS:
            migration = migration_class()
            if migration.version not in applied:
                pending.append(migration)

        return sorted(pending)

    def get_migration_history(self) -> list[MigrationRecord]:
        """Get history of applied migrations.

        Returns:
            List of MigrationRecord objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT version, name, applied_at, checksum, success, duration_ms
            FROM _migrations
            ORDER BY version
        """
        )

        history = []
        for row in cursor.fetchall():
            history.append(
                MigrationRecord(
                    version=row["version"],
                    name=row["name"],
                    applied_at=datetime.fromisoformat(row["applied_at"]),
                    checksum=row["checksum"],
                    success=bool(row["success"]),
                    duration_ms=row["duration_ms"],
                )
            )

        self._close_connection(conn)
        return history

    def migrate(
        self, target_version: int | None = None, dry_run: bool = False
    ) -> MigrationResult:
        """Apply pending migrations.

        Args:
            target_version: Optional target version (None = latest)
            dry_run: If True, don't actually apply migrations

        Returns:
            MigrationResult with details of what was done
        """
        result = MigrationResult(dry_run=dry_run)
        pending = self.get_pending_migrations()

        if target_version is not None:
            pending = [m for m in pending if m.version <= target_version]

        if not pending:
            logger.info("No pending migrations")
            return result

        conn = self._get_connection()

        for migration in pending:
            if dry_run:
                result.applied.append(
                    f"[DRY RUN] {migration.version}: {migration.name}"
                )
                continue

            cursor = conn.cursor()
            start_time = datetime.now(timezone.utc)

            try:
                migration.up(cursor)
                conn.commit()

                duration_ms = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000

                # Record success
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO _migrations
                    (version, name, applied_at, checksum, success, duration_ms)
                    VALUES (?, ?, ?, ?, 1, ?)
                """,
                    (
                        migration.version,
                        migration.name,
                        datetime.now(timezone.utc).isoformat(),
                        migration.get_checksum(),
                        duration_ms,
                    ),
                )
                conn.commit()

                result.applied.append(f"{migration.version}: {migration.name}")
                logger.info(
                    "Applied migration %d: %s (%.1fms)",
                    migration.version,
                    migration.name,
                    duration_ms,
                )

            except Exception as e:
                conn.rollback()
                result.failed.append(f"{migration.version}: {migration.name} - {e}")
                result.success = False
                result.error_message = str(e)
                logger.error(
                    "Failed migration %d: %s - %s",
                    migration.version,
                    migration.name,
                    e,
                )
                break

        self._close_connection(conn)
        return result

    def rollback(self, steps: int = 1, dry_run: bool = False) -> MigrationResult:
        """Rollback applied migrations.

        Args:
            steps: Number of migrations to rollback
            dry_run: If True, don't actually rollback

        Returns:
            MigrationResult with details of what was done
        """
        result = MigrationResult(dry_run=dry_run)

        history = self.get_migration_history()
        if not history:
            logger.info("No migrations to rollback")
            return result

        # Get last N successful migrations
        to_rollback = [h for h in reversed(history) if h.success][:steps]

        if not to_rollback:
            logger.info("No successful migrations to rollback")
            return result

        # Find Migration classes for these versions
        version_to_class = {m().version: m for m in self.MIGRATIONS}

        conn = self._get_connection()

        for record in to_rollback:
            if record.version not in version_to_class:
                result.skipped.append(
                    f"{record.version}: {record.name} - No migration class found"
                )
                continue

            migration = version_to_class[record.version]()

            if dry_run:
                result.applied.append(
                    f"[DRY RUN ROLLBACK] {migration.version}: {migration.name}"
                )
                continue

            cursor = conn.cursor()

            try:
                migration.down(cursor)
                conn.commit()

                # Remove from history
                cursor.execute(
                    "DELETE FROM _migrations WHERE version = ?", (migration.version,)
                )
                conn.commit()

                result.applied.append(
                    f"Rolled back {migration.version}: {migration.name}"
                )
                logger.info(
                    "Rolled back migration %d: %s",
                    migration.version,
                    migration.name,
                )

            except Exception as e:
                conn.rollback()
                result.failed.append(f"{migration.version}: {migration.name} - {e}")
                result.success = False
                result.error_message = str(e)
                logger.error(
                    "Failed rollback %d: %s - %s",
                    migration.version,
                    migration.name,
                    e,
                )
                break

        self._close_connection(conn)
        return result

    def validate(self) -> list[str]:
        """Validate migrations for common issues.

        Returns:
            List of warning/error messages
        """
        issues = []

        # Check for duplicate versions
        versions = [m().version for m in self.MIGRATIONS]
        if len(versions) != len(set(versions)):
            issues.append("ERROR: Duplicate migration versions found")

        # Check for gaps in versions
        if versions:
            expected = set(range(1, max(versions) + 1))
            actual = set(versions)
            missing = expected - actual
            if missing:
                issues.append(f"WARNING: Missing versions: {sorted(missing)}")

        # Check that applied migrations have matching checksums
        history = self.get_migration_history()
        version_to_class = {m().version: m for m in self.MIGRATIONS}

        for record in history:
            if record.version in version_to_class:
                migration = version_to_class[record.version]()
                if migration.get_checksum() != record.checksum:
                    issues.append(
                        f"WARNING: Migration {record.version} checksum mismatch - "
                        f"code may have changed since applied"
                    )

        return issues

    def get_status(self) -> dict[str, Any]:
        """Get current migration status.

        Returns:
            Dictionary with status information
        """
        applied = self.get_applied_versions()
        pending = self.get_pending_migrations()
        history = self.get_migration_history()

        return {
            "current_version": self.get_current_version(),
            "latest_available": max(m().version for m in self.MIGRATIONS)
            if self.MIGRATIONS
            else 0,
            "applied_count": len(applied),
            "pending_count": len(pending),
            "pending": [{"version": m.version, "name": m.name} for m in pending],
            "recent_history": [h.to_dict() for h in history[-5:]],
            "issues": self.validate(),
        }

    def close(self) -> None:
        """Close the migrator connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# =============================================================================
# Convenience Functions
# =============================================================================


def create_migrator(db_path: str | None = None) -> Migrator:
    """Create a migrator instance.

    Args:
        db_path: Optional database path

    Returns:
        Migrator instance
    """
    return Migrator(db_path)


def run_migrations(db_path: str, dry_run: bool = False) -> MigrationResult:
    """Run all pending migrations.

    Args:
        db_path: Path to database
        dry_run: If True, don't apply changes

    Returns:
        MigrationResult
    """
    migrator = Migrator(db_path)
    return migrator.migrate(dry_run=dry_run)


# =============================================================================
# Unit Tests
# =============================================================================


def _create_module_tests() -> bool:
    """Create unit tests for this module."""
    from test_framework import TestSuite

    suite = TestSuite("Database Migrations", "migrations.py")
    suite.start_suite()

    def test_migration_record():
        record = MigrationRecord(
            version=1,
            name="test",
            applied_at=datetime.now(timezone.utc),
            checksum="abc123",
            success=True,
        )
        assert record.version == 1
        data = record.to_dict()
        assert "version" in data

    def test_migration_result():
        result = MigrationResult()
        assert result.total_applied == 0
        assert result.success is True
        result.applied.append("test")
        assert result.total_applied == 1

    def test_migration_result_failed():
        result = MigrationResult(failed=["test"])
        assert result.total_failed == 1

    def test_migrator_init():
        migrator = Migrator()
        assert migrator.db_path == ":memory:"

    def test_migrator_get_current_version():
        migrator = Migrator()
        # Fresh DB should have version 0
        assert migrator.get_current_version() == 0

    def test_migrator_get_pending():
        migrator = Migrator()
        pending = migrator.get_pending_migrations()
        assert len(pending) > 0
        assert all(isinstance(m, Migration) for m in pending)

    def test_migrator_migrate_dry_run():
        migrator = Migrator()
        result = migrator.migrate(dry_run=True)
        assert result.dry_run is True
        assert len(result.applied) > 0
        # Should not actually apply
        assert migrator.get_current_version() == 0

    def test_migrator_migrate():
        # Create a minimal stories table first
        migrator = Migrator()
        conn = migrator._get_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stories (
                id INTEGER PRIMARY KEY,
                title TEXT,
                url TEXT,
                summary TEXT,
                quality_score REAL,
                source TEXT,
                published_at TEXT,
                post_id TEXT,
                image_path TEXT,
                linkedin_url TEXT,
                search_prompt TEXT,
                created_at TEXT
            )
        """
        )
        conn.commit()

        result = migrator.migrate()
        assert result.success is True
        assert len(result.applied) > 0
        assert migrator.get_current_version() > 0

    def test_migrator_history():
        migrator = Migrator()
        # Create stories table
        conn = migrator._get_connection()
        conn.execute("CREATE TABLE IF NOT EXISTS stories (id INTEGER PRIMARY KEY)")
        conn.commit()

        migrator.migrate()
        history = migrator.get_migration_history()
        assert len(history) > 0

    def test_migrator_rollback_dry_run():
        migrator = Migrator()
        # Create stories table and migrate
        conn = migrator._get_connection()
        conn.execute("CREATE TABLE IF NOT EXISTS stories (id INTEGER PRIMARY KEY)")
        conn.commit()
        migrator.migrate()

        version_before = migrator.get_current_version()
        result = migrator.rollback(steps=1, dry_run=True)
        assert result.dry_run is True
        # Version should not change
        assert migrator.get_current_version() == version_before

    def test_migrator_validate():
        migrator = Migrator()
        issues = migrator.validate()
        # Should have no errors for built-in migrations
        assert not any("ERROR" in i for i in issues)

    def test_migrator_status():
        migrator = Migrator()
        status = migrator.get_status()
        assert "current_version" in status
        assert "pending_count" in status
        assert status["pending_count"] > 0

    def test_migration_checksum():
        m = Migration001AddAnalyticsColumns()
        checksum = m.get_checksum()
        assert len(checksum) == 12
        # Should be consistent
        assert m.get_checksum() == checksum

    def test_migration_sorting():
        m1 = Migration001AddAnalyticsColumns()
        m2 = Migration002AddPostMetadata()
        assert m1 < m2

    suite.run_test(
        test_name="Migration record",
        test_func=test_migration_record,
        test_summary="Tests Migration record functionality",
        method_description="Calls MigrationRecord and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Migration result",
        test_func=test_migration_result,
        test_summary="Tests Migration result functionality",
        method_description="Calls MigrationResult and verifies the result",
        expected_outcome="Function returns True",
    )
    suite.run_test(
        test_name="Migration result failed",
        test_func=test_migration_result_failed,
        test_summary="Tests Migration result failed functionality",
        method_description="Calls MigrationResult and verifies the result",
        expected_outcome="Function handles invalid input appropriately",
    )
    suite.run_test(
        test_name="Migrator init",
        test_func=test_migrator_init,
        test_summary="Tests Migrator init functionality",
        method_description="Calls Migrator and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Get current version",
        test_func=test_migrator_get_current_version,
        test_summary="Tests Get current version functionality",
        method_description="Calls Migrator and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Get pending migrations",
        test_func=test_migrator_get_pending,
        test_summary="Tests Get pending migrations functionality",
        method_description="Calls Migrator and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Migrate dry run",
        test_func=test_migrator_migrate_dry_run,
        test_summary="Tests Migrate dry run functionality",
        method_description="Calls Migrator and verifies the result",
        expected_outcome="Function returns True",
    )
    suite.run_test(
        test_name="Migrate",
        test_func=test_migrator_migrate,
        test_summary="Tests Migrate functionality",
        method_description="Calls Migrator and verifies the result",
        expected_outcome="Function returns True",
    )
    suite.run_test(
        test_name="Migration history",
        test_func=test_migrator_history,
        test_summary="Tests Migration history functionality",
        method_description="Calls Migrator and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Rollback dry run",
        test_func=test_migrator_rollback_dry_run,
        test_summary="Tests Rollback dry run functionality",
        method_description="Calls Migrator and verifies the result",
        expected_outcome="Function returns True",
    )
    suite.run_test(
        test_name="Validate migrations",
        test_func=test_migrator_validate,
        test_summary="Tests Validate migrations functionality",
        method_description="Calls Migrator and verifies the result",
        expected_outcome="Function returns False or falsy value",
    )
    suite.run_test(
        test_name="Status check",
        test_func=test_migrator_status,
        test_summary="Tests Status check functionality",
        method_description="Calls Migrator and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Migration checksum",
        test_func=test_migration_checksum,
        test_summary="Tests Migration checksum functionality",
        method_description="Calls Migration001AddAnalyticsColumns and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Migration sorting",
        test_func=test_migration_sorting,
        test_summary="Tests Migration sorting functionality",
        method_description="Calls Migration001AddAnalyticsColumns and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )

    return suite.finish_suite()
if __name__ == "__main__":
    # Demo usage
    print("Database Migration System Demo")
    print("=" * 50)

    migrator = Migrator()

    # Show status
    status = migrator.get_status()
    print(f"\nCurrent Version: {status['current_version']}")
    print(f"Latest Available: {status['latest_available']}")
    print(f"Pending: {status['pending_count']}")

    print("\nPending Migrations:")
    for m in status["pending"]:
        print(f"  - Version {m['version']}: {m['name']}")

    # Run dry-run
    print("\nDry Run:")
    result = migrator.migrate(dry_run=True)
    for applied in result.applied:
        print(f"  {applied}")
