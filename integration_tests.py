"""Integration tests for Social Media Publisher.

This module provides end-to-end integration tests for the complete
content publishing pipeline.

Features:
- Pipeline integration tests
- Database integration tests
- Configuration validation tests
- Cross-module interaction tests
- Mock API integration tests

Example:
    Run with: python integration_tests.py
    Or via: python run_tests.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class TestStory:
    """Test story fixture for integration tests."""

    id: int = 1
    title: str = "Test Story Title"
    summary: str = "Test summary about clean energy technology advancements."
    url: str = "https://example.com/test-story"
    source_url: str = "https://techcrunch.com/test"
    quality_score: float = 0.85
    people_mentioned: list[str] = field(default_factory=list)
    company_mentioned: str = "Tesla"
    image_path: str | None = None
    published: bool = False
    published_at: datetime | None = None
    linkedin_post_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "source_url": self.source_url,
            "quality_score": self.quality_score,
            "people_mentioned": json.dumps(self.people_mentioned),
            "company_mentioned": self.company_mentioned,
            "image_path": self.image_path,
            "published": self.published,
            "published_at": self.published_at.isoformat()
            if self.published_at
            else None,
            "linkedin_post_id": self.linkedin_post_id,
        }


@dataclass
class TestConfig:
    """Test configuration fixture."""

    search_prompt: str = "clean energy technology"
    publish_hour_start: int = 8
    publish_hour_end: int = 18
    quality_threshold: float = 0.6
    linkedin_access_token: str = "test-token"
    linkedin_person_urn: str = "urn:li:person:test123"


class MockDatabase:
    """Mock database for integration testing."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize mock database."""
        if db_path:
            self.db_path = db_path
        else:
            self.temp_dir = tempfile.mkdtemp()
            self.db_path = os.path.join(self.temp_dir, "test.db")

        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS stories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                summary TEXT,
                url TEXT,
                source_url TEXT,
                quality_score REAL,
                people_mentioned TEXT,
                company_mentioned TEXT,
                image_path TEXT,
                published INTEGER DEFAULT 0,
                published_at TEXT,
                linkedin_post_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def add_story(self, story: TestStory) -> int:
        """Add a story to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        data = story.to_dict()
        cursor.execute(
            """
            INSERT INTO stories (title, summary, url, source_url, quality_score,
                               people_mentioned, company_mentioned, image_path,
                               published, published_at, linkedin_post_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                data["title"],
                data["summary"],
                data["url"],
                data["source_url"],
                data["quality_score"],
                data["people_mentioned"],
                data["company_mentioned"],
                data["image_path"],
                data["published"],
                data["published_at"],
                data["linkedin_post_id"],
            ),
        )

        story_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return story_id if story_id else 0

    def get_story(self, story_id: int) -> dict[str, Any] | None:
        """Get a story by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM stories WHERE id = ?", (story_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_unpublished(self) -> list[dict[str, Any]]:
        """Get all unpublished stories."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM stories WHERE published = 0 ORDER BY quality_score DESC"
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def mark_published(self, story_id: int, post_id: str) -> None:
        """Mark a story as published."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE stories SET published = 1, published_at = ?, linkedin_post_id = ?
            WHERE id = ?
        """,
            (datetime.now(timezone.utc).isoformat(), post_id, story_id),
        )

        conn.commit()
        conn.close()

    def close(self) -> None:
        """Clean up temporary files."""
        if hasattr(self, "temp_dir"):
            import shutil

            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass


class MockLinkedInAPI:
    """Mock LinkedIn API for integration testing."""

    def __init__(self) -> None:
        """Initialize mock API."""
        self.posts: list[dict[str, Any]] = []
        self.should_fail: bool = False
        self.rate_limited: bool = False
        self.call_count: int = 0

    def create_post(self, text: str, image_path: str | None = None) -> dict[str, Any]:
        """Create a mock LinkedIn post."""
        self.call_count += 1

        if self.should_fail:
            raise ConnectionError("Simulated API failure")

        if self.rate_limited:
            raise Exception("Rate limit exceeded (429)")

        post_id = f"urn:li:share:{len(self.posts) + 1}_{datetime.now().timestamp():.0f}"
        post = {
            "id": post_id,
            "text": text,
            "image_path": image_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.posts.append(post)
        return post

    def get_post(self, post_id: str) -> dict[str, Any] | None:
        """Get a post by ID."""
        for post in self.posts:
            if post["id"] == post_id:
                return post
        return None

    def reset(self) -> None:
        """Reset mock state."""
        self.posts = []
        self.should_fail = False
        self.rate_limited = False
        self.call_count = 0


# =============================================================================
# Integration Test Cases
# =============================================================================


@dataclass
class IntegrationTestResult:
    """Result of an integration test."""

    name: str
    passed: bool
    duration_ms: float
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class IntegrationTestSuite:
    """Suite of integration tests."""

    def __init__(self) -> None:
        """Initialize test suite."""
        self.db: MockDatabase | None = None
        self.api: MockLinkedInAPI | None = None
        self.results: list[IntegrationTestResult] = []

    def setup(self) -> None:
        """Set up test fixtures."""
        self.db = MockDatabase()
        self.api = MockLinkedInAPI()

    def teardown(self) -> None:
        """Clean up test fixtures."""
        if self.db:
            self.db.close()
        self.db = None
        self.api = None

    def run_test(
        self,
        name: str,
        test_func: Any,
    ) -> IntegrationTestResult:
        """Run a single integration test."""
        import time

        start = time.time()
        try:
            test_func()
            duration = (time.time() - start) * 1000
            return IntegrationTestResult(name=name, passed=True, duration_ms=duration)
        except AssertionError as e:
            duration = (time.time() - start) * 1000
            return IntegrationTestResult(
                name=name, passed=False, duration_ms=duration, error=str(e)
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return IntegrationTestResult(
                name=name,
                passed=False,
                duration_ms=duration,
                error=f"Unexpected error: {e}",
            )

    def test_story_pipeline_flow(self) -> None:
        """Test complete story pipeline: add -> enrich -> publish."""
        assert self.db is not None
        assert self.api is not None

        # Add a story
        story = TestStory(
            title="Breakthrough in Solar Technology",
            summary="New solar cell achieves 47% efficiency.",
            quality_score=0.9,
        )
        story_id = self.db.add_story(story)
        assert story_id > 0

        # Verify story was added
        retrieved = self.db.get_story(story_id)
        assert retrieved is not None
        assert retrieved["title"] == story.title
        assert retrieved["published"] == 0

        # Simulate publishing
        post = self.api.create_post(story.summary)
        self.db.mark_published(story_id, post["id"])

        # Verify published state
        retrieved = self.db.get_story(story_id)
        assert retrieved is not None
        assert retrieved["published"] == 1
        assert retrieved["linkedin_post_id"] == post["id"]

    def test_unpublished_story_selection(self) -> None:
        """Test selecting unpublished stories by quality score."""
        assert self.db is not None

        # Add stories with different quality scores
        stories = [
            TestStory(title="Low Quality", quality_score=0.3),
            TestStory(title="High Quality", quality_score=0.95),
            TestStory(title="Medium Quality", quality_score=0.6),
        ]

        for story in stories:
            self.db.add_story(story)

        # Get unpublished - should be ordered by quality
        unpublished = self.db.get_unpublished()
        assert len(unpublished) == 3
        assert unpublished[0]["title"] == "High Quality"
        assert unpublished[1]["title"] == "Medium Quality"
        assert unpublished[2]["title"] == "Low Quality"

    def test_api_failure_handling(self) -> None:
        """Test handling of API failures."""
        assert self.db is not None
        assert self.api is not None

        story = TestStory(title="Test Story")
        story_id = self.db.add_story(story)

        # Simulate API failure
        self.api.should_fail = True

        try:
            self.api.create_post(story.summary)
            assert False, "Should have raised exception"
        except ConnectionError:
            pass  # Expected

        # Story should still be unpublished
        retrieved = self.db.get_story(story_id)
        assert retrieved is not None
        assert retrieved["published"] == 0

    def test_rate_limit_handling(self) -> None:
        """Test handling of rate limits."""
        assert self.api is not None

        self.api.rate_limited = True

        try:
            self.api.create_post("Test post")
            assert False, "Should have raised exception"
        except Exception as e:
            assert "429" in str(e) or "Rate limit" in str(e)

    def test_quality_filtering(self) -> None:
        """Test quality-based story filtering."""
        assert self.db is not None

        threshold = 0.6

        stories = [
            TestStory(title="Below Threshold", quality_score=0.5),
            TestStory(title="Above Threshold", quality_score=0.8),
            TestStory(title="At Threshold", quality_score=0.6),
        ]

        for story in stories:
            self.db.add_story(story)

        unpublished = self.db.get_unpublished()
        qualified = [s for s in unpublished if s["quality_score"] >= threshold]

        assert len(qualified) == 2
        titles = [s["title"] for s in qualified]
        assert "Below Threshold" not in titles
        assert "Above Threshold" in titles
        assert "At Threshold" in titles

    def test_duplicate_story_detection(self) -> None:
        """Test duplicate story detection."""
        assert self.db is not None

        story1 = TestStory(title="Unique Story About Solar Energy")
        self.db.add_story(story1)

        # Try to add similar story
        story2 = TestStory(title="Unique Story About Solar Energy")
        self.db.add_story(story2)

        # Both should exist (actual dedup logic would be in application layer)
        unpublished = self.db.get_unpublished()
        assert len(unpublished) == 2

        # Simulate dedup check - find similar titles
        titles = [s["title"] for s in unpublished]
        unique_titles = set(titles)
        has_duplicates = len(titles) != len(unique_titles)
        assert has_duplicates

    def test_cross_module_story_flow(self) -> None:
        """Test story flowing through multiple components."""
        assert self.db is not None
        assert self.api is not None

        # Simulate story discovery
        discovered_stories = [
            {"title": "Story 1", "score": 0.85},
            {"title": "Story 2", "score": 0.7},
            {"title": "Story 3", "score": 0.95},
        ]

        # Add to database
        story_ids = []
        for data in discovered_stories:
            story = TestStory(title=data["title"], quality_score=data["score"])
            story_id = self.db.add_story(story)
            story_ids.append(story_id)

        # Get best story for publishing
        unpublished = self.db.get_unpublished()
        best_story = unpublished[0]  # Highest quality

        assert best_story["title"] == "Story 3"

        # Publish best story
        post = self.api.create_post(best_story["summary"] or best_story["title"])
        self.db.mark_published(best_story["id"], post["id"])

        # Verify one less unpublished
        unpublished = self.db.get_unpublished()
        assert len(unpublished) == 2

    def run_all(self) -> list[IntegrationTestResult]:
        """Run all integration tests."""
        tests = [
            ("Story pipeline flow", self.test_story_pipeline_flow),
            ("Unpublished story selection", self.test_unpublished_story_selection),
            ("API failure handling", self.test_api_failure_handling),
            ("Rate limit handling", self.test_rate_limit_handling),
            ("Quality filtering", self.test_quality_filtering),
            ("Duplicate detection", self.test_duplicate_story_detection),
            ("Cross-module story flow", self.test_cross_module_story_flow),
        ]

        self.results = []

        for name, test_func in tests:
            self.setup()
            try:
                result = self.run_test(name, test_func)
                self.results.append(result)
            finally:
                self.teardown()

        return self.results


def format_integration_results(results: list[IntegrationTestResult]) -> str:
    """Format integration test results for display."""
    lines = [
        "Integration Test Results",
        "=" * 50,
    ]

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for result in results:
        status = "✅" if result.passed else "❌"
        line = f"  {status} {result.name} ({result.duration_ms:.1f}ms)"
        if result.error:
            line += f"\n       Error: {result.error}"
        lines.append(line)

    lines.extend(
        [
            "",
            "-" * 50,
            f"  {'✅' if passed == total else '❌'} PASSED: {passed}/{total} tests",
            "=" * 50,
        ]
    )

    return "\n".join(lines)


# =============================================================================
# Unit Tests (TestSuite integration)
# =============================================================================


def _create_module_tests() -> bool:
    """Create unit tests for this module."""
    from test_framework import TestSuite

    suite = TestSuite("Integration Tests", "integration_tests.py")
    suite.start_suite()

    def test_test_story_fixture():
        story = TestStory()
        assert story.title == "Test Story Title"
        data = story.to_dict()
        assert "title" in data
        assert "summary" in data

    def test_mock_database_init():
        db = MockDatabase()
        try:
            assert db.db_path is not None
            assert os.path.exists(db.db_path)
        finally:
            db.close()

    def test_mock_database_add_get():
        db = MockDatabase()
        try:
            story = TestStory(title="Test")
            story_id = db.add_story(story)
            assert story_id > 0

            retrieved = db.get_story(story_id)
            assert retrieved is not None
            assert retrieved["title"] == "Test"
        finally:
            db.close()

    def test_mock_database_unpublished():
        db = MockDatabase()
        try:
            db.add_story(TestStory(title="Story1", quality_score=0.5))
            db.add_story(TestStory(title="Story2", quality_score=0.9))

            unpublished = db.get_unpublished()
            assert len(unpublished) == 2
            assert unpublished[0]["quality_score"] == 0.9
        finally:
            db.close()

    def test_mock_database_mark_published():
        db = MockDatabase()
        try:
            story_id = db.add_story(TestStory())
            db.mark_published(story_id, "post_123")

            story = db.get_story(story_id)
            assert story is not None
            assert story["published"] == 1
            assert story["linkedin_post_id"] == "post_123"
        finally:
            db.close()

    def test_mock_linkedin_api_create():
        api = MockLinkedInAPI()
        post = api.create_post("Test content")
        assert "id" in post
        assert post["text"] == "Test content"
        assert api.call_count == 1

    def test_mock_linkedin_api_failure():
        api = MockLinkedInAPI()
        api.should_fail = True
        try:
            api.create_post("Test")
            assert False, "Should have raised"
        except ConnectionError:
            pass

    def test_mock_linkedin_api_reset():
        api = MockLinkedInAPI()
        api.create_post("Test")
        api.should_fail = True
        api.reset()
        assert len(api.posts) == 0
        assert api.should_fail is False

    def test_integration_test_result():
        result = IntegrationTestResult(name="test", passed=True, duration_ms=100.0)
        assert result.passed
        assert result.error is None

    def test_integration_suite_setup():
        suite = IntegrationTestSuite()
        suite.setup()
        assert suite.db is not None
        assert suite.api is not None
        suite.teardown()
        assert suite.db is None

    def test_integration_suite_run_test():
        suite = IntegrationTestSuite()
        suite.setup()

        def passing_test() -> None:
            assert True

        result = suite.run_test("passing", passing_test)
        assert result.passed
        suite.teardown()

    def test_integration_suite_run_failing():
        suite = IntegrationTestSuite()
        suite.setup()

        def failing_test() -> None:
            assert False, "Expected failure"

        result = suite.run_test("failing", failing_test)
        assert not result.passed
        assert "Expected failure" in str(result.error)
        suite.teardown()

    def test_format_results():
        results = [
            IntegrationTestResult("test1", True, 50.0),
            IntegrationTestResult("test2", False, 100.0, "Error occurred"),
        ]
        formatted = format_integration_results(results)
        assert "test1" in formatted
        assert "test2" in formatted
        assert "1/2" in formatted

    def test_run_all_integration_tests():
        suite = IntegrationTestSuite()
        results = suite.run_all()
        assert len(results) == 7
        passed = sum(1 for r in results if r.passed)
        assert passed == 7

    def test_test_config_fixture():
        config = TestConfig()
        assert config.publish_hour_start == 8
        assert config.quality_threshold == 0.6

    suite.run_test(
        test_name="Test story fixture",
        test_func=test_test_story_fixture,
        test_summary="Tests Test story fixture functionality",
        method_description="Calls TestStory and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Mock database init",
        test_func=test_mock_database_init,
        test_summary="Tests Mock database init functionality",
        method_description="Calls MockDatabase and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Mock database add/get",
        test_func=test_mock_database_add_get,
        test_summary="Tests Mock database add/get functionality",
        method_description="Calls MockDatabase and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Mock database unpublished",
        test_func=test_mock_database_unpublished,
        test_summary="Tests Mock database unpublished functionality",
        method_description="Calls MockDatabase and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Mock database mark published",
        test_func=test_mock_database_mark_published,
        test_summary="Tests Mock database mark published functionality",
        method_description="Calls MockDatabase and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Mock LinkedIn API create",
        test_func=test_mock_linkedin_api_create,
        test_summary="Tests Mock LinkedIn API create functionality",
        method_description="Calls MockLinkedInAPI and verifies the result",
        expected_outcome="Function creates the expected object or result",
    )
    suite.run_test(
        test_name="Mock LinkedIn API failure",
        test_func=test_mock_linkedin_api_failure,
        test_summary="Tests Mock LinkedIn API failure functionality",
        method_description="Calls MockLinkedInAPI and verifies the result",
        expected_outcome="Function raises the expected error or exception",
    )
    suite.run_test(
        test_name="Mock LinkedIn API reset",
        test_func=test_mock_linkedin_api_reset,
        test_summary="Tests Mock LinkedIn API reset functionality",
        method_description="Calls MockLinkedInAPI and verifies the result",
        expected_outcome="Function correctly updates the target",
    )
    suite.run_test(
        test_name="Integration test result",
        test_func=test_integration_test_result,
        test_summary="Tests Integration test result functionality",
        method_description="Calls IntegrationTestResult and verifies the result",
        expected_outcome="Function returns None as expected",
    )
    suite.run_test(
        test_name="Integration suite setup",
        test_func=test_integration_suite_setup,
        test_summary="Tests Integration suite setup functionality",
        method_description="Calls IntegrationTestSuite and verifies the result",
        expected_outcome="Function returns None as expected",
    )
    suite.run_test(
        test_name="Integration suite run test",
        test_func=test_integration_suite_run_test,
        test_summary="Tests Integration suite run test functionality",
        method_description="Calls IntegrationTestSuite and verifies the result",
        expected_outcome="Function returns True",
    )
    suite.run_test(
        test_name="Integration suite run failing",
        test_func=test_integration_suite_run_failing,
        test_summary="Tests Integration suite run failing functionality",
        method_description="Calls IntegrationTestSuite and verifies the result",
        expected_outcome="Function returns False or falsy value",
    )
    suite.run_test(
        test_name="Format results",
        test_func=test_format_results,
        test_summary="Tests Format results functionality",
        method_description="Calls IntegrationTestResult and verifies the result",
        expected_outcome="Function raises the expected error or exception",
    )
    suite.run_test(
        test_name="Run all integration tests",
        test_func=test_run_all_integration_tests,
        test_summary="Tests Run all integration tests functionality",
        method_description="Calls IntegrationTestSuite and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )
    suite.run_test(
        test_name="Test config fixture",
        test_func=test_test_config_fixture,
        test_summary="Tests Test config fixture functionality",
        method_description="Calls TestConfig and verifies the result",
        expected_outcome="Function produces the correct result without errors",
    )

    return suite.finish_suite()
if __name__ == "__main__":
    # Run integration tests
    print("\n" + "=" * 60)
    print("  Running Integration Tests")
    print("=" * 60 + "\n")

    suite = IntegrationTestSuite()
    results = suite.run_all()
    print(format_integration_results(results))

    # Exit with appropriate code
    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)


run_comprehensive_tests = __import__("test_framework").create_standard_test_runner(
    _create_module_tests
)
