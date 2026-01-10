"""Publication scheduling for stories."""

import random
import logging
from datetime import datetime, timedelta

from config import Config
from database import Database, Story

logger = logging.getLogger(__name__)


class Scheduler:
    """Schedule stories for publication at optimal times."""

    def __init__(self, database: Database):
        """Initialize the scheduler."""
        self.db = database

    def schedule_stories(self, schedule_all: bool = False) -> list[Story]:
        """
        Schedule stories for publication.

        Args:
            schedule_all: If True, schedule ALL available stories across multiple days.
                         If False (default), schedule up to MAX_STORIES_PER_DAY for today only.

        Returns the list of newly scheduled stories.
        """
        if schedule_all:
            # Get ALL available approved unpublished stories (sorted by quality score)
            candidates = self.db.get_approved_unpublished_stories(limit=1000)
        else:
            # For single-day scheduling, limit to MAX_STORIES_PER_DAY
            target_count = Config.MAX_STORIES_PER_DAY
            # Clear any existing scheduled stories when doing cycle-based scheduling
            cleared = self.db.clear_scheduled_status()
            if cleared > 0:
                logger.info(f"Cleared {cleared} previously scheduled stories")
            candidates = self.db.get_approved_unpublished_stories(target_count)

        if not candidates:
            logger.warning("No approved stories available to schedule")
            return []

        logger.info(f"Found {len(candidates)} stories to schedule")

        # Get existing scheduled times to avoid conflicts
        existing_scheduled = self.db.get_scheduled_stories()
        existing_times = set()
        latest_scheduled_time = None

        for story in existing_scheduled:
            if story.scheduled_time:
                existing_times.add(story.scheduled_time.date())
                if (
                    latest_scheduled_time is None
                    or story.scheduled_time > latest_scheduled_time
                ):
                    latest_scheduled_time = story.scheduled_time

        # Calculate publication times, starting after any existing scheduled stories
        scheduled_times = self._calculate_schedule_times(
            len(candidates),
            start_after=latest_scheduled_time,
            exclude_dates=existing_times if schedule_all else set(),
        )

        # Assign times to stories
        scheduled_stories = []
        for story, scheduled_time in zip(candidates, scheduled_times):
            story.scheduled_time = scheduled_time
            story.publish_status = "scheduled"
            self.db.update_story(story)
            scheduled_stories.append(story)
            logger.info(
                f"Scheduled story {story.id} for {scheduled_time.strftime('%Y-%m-%d %H:%M')}: "
                f"{story.title}"
            )

        logger.info(f"Scheduled {len(scheduled_stories)} stories for publication")
        return scheduled_stories

    def _calculate_schedule_times(
        self,
        count: int,
        start_after: datetime | None = None,
        exclude_dates: set | None = None,
    ) -> list[datetime]:
        """
        Calculate publication times for a given number of stories.
        Distributes them evenly across the publication window with jitter.

        Args:
            count: Number of time slots to generate
            start_after: If provided, start scheduling after this datetime
            exclude_dates: Set of dates to skip (already have scheduled stories)
        """
        if count == 0:
            return []

        now = datetime.now()
        start_hour = Config.get_pub_start_hour()
        end_hour = Config.get_pub_end_hour()
        max_per_day = Config.MAX_STORIES_PER_DAY
        jitter_minutes = Config.JITTER_MINUTES

        # Determine the base start time
        if start_after and start_after > now:
            # Start from the day after the latest scheduled story
            base_time = start_after.replace(
                hour=start_hour, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
        else:
            base_time = self._get_next_valid_start_time(now, start_hour, end_hour)

        # Calculate daily hours available
        daily_hours = end_hour - start_hour

        # Calculate time slots: up to max_per_day per day, spread evenly
        scheduled_times = []
        current_date = base_time.date()
        stories_on_current_day = 0

        for i in range(count):
            # Skip dates that already have scheduled stories
            if exclude_dates:
                while current_date in exclude_dates:
                    current_date += timedelta(days=1)
                    stories_on_current_day = 0

            # If we've hit max for this day, move to next day
            if stories_on_current_day >= max_per_day:
                current_date += timedelta(days=1)
                stories_on_current_day = 0
                # Skip excluded dates again
                if exclude_dates:
                    while current_date in exclude_dates:
                        current_date += timedelta(days=1)

            # Calculate time slot within the day
            # Spread stories evenly across the publishing window
            slot_duration = daily_hours / max(max_per_day, 1)
            slot_start = start_hour + (stories_on_current_day * slot_duration)
            slot_end = slot_start + slot_duration

            # Pick a random time within this slot
            random_minutes = random.randint(0, int(slot_duration * 60) - 1)
            scheduled_hour = int(slot_start) + (random_minutes // 60)
            scheduled_minute = random_minutes % 60

            scheduled_time = datetime.combine(
                current_date, datetime.min.time()
            ).replace(
                hour=min(scheduled_hour, end_hour - 1),
                minute=scheduled_minute,
                second=0,
                microsecond=0,
            )

            # Add jitter (but stay within valid hours)
            jitter = random.randint(-jitter_minutes, jitter_minutes)
            scheduled_time += timedelta(minutes=jitter)

            # Ensure time is within valid hours
            scheduled_time = self._adjust_to_valid_hours(
                scheduled_time, start_hour, end_hour
            )

            # If adjustment moved to next day, update tracking
            if scheduled_time.date() != current_date:
                current_date = scheduled_time.date()
                stories_on_current_day = 0

            scheduled_times.append(scheduled_time)
            stories_on_current_day += 1

        return scheduled_times

    def _get_next_valid_start_time(
        self, now: datetime, start_hour: int, end_hour: int
    ) -> datetime:
        """Get the next valid time to start scheduling from."""
        result = now

        # If we're before start hour today, use start hour today
        if now.hour < start_hour:
            result = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)

        # If we're after end hour today, use start hour tomorrow
        elif now.hour >= end_hour:
            result = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
            result += timedelta(days=1)

        return result

    def _adjust_to_valid_hours(
        self, dt: datetime, start_hour: int, end_hour: int
    ) -> datetime:
        """Adjust a datetime to be within valid publishing hours."""
        if dt.hour < start_hour:
            return dt.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        elif dt.hour >= end_hour:
            # Move to start hour of next day
            dt = dt.replace(hour=start_hour, minute=0, second=0, microsecond=0)
            return dt + timedelta(days=1)
        return dt

    def get_scheduled_stories(self) -> list[Story]:
        """Get all currently scheduled stories."""
        return self.db.get_scheduled_stories()

    def get_due_stories(self) -> list[Story]:
        """Get stories that are due for publication now."""
        return self.db.get_scheduled_stories_due()

    def get_next_scheduled_time(self) -> datetime | None:
        """Get the time of the next scheduled publication."""
        scheduled = self.db.get_scheduled_stories()
        if not scheduled:
            return None
        return min(s.scheduled_time for s in scheduled if s.scheduled_time)

    def get_schedule_summary(self) -> str:
        """Get a human-readable summary of the current schedule."""
        scheduled = self.get_scheduled_stories()
        if not scheduled:
            return "No stories currently scheduled"

        lines = ["Current Publication Schedule:"]
        for story in sorted(scheduled, key=lambda s: s.scheduled_time or datetime.max):
            time_str = (
                story.scheduled_time.strftime("%Y-%m-%d %H:%M")
                if story.scheduled_time
                else "Unknown"
            )
            lines.append(f"  [{time_str}] {story.title}")

        return "\n".join(lines)


# ============================================================================
# Unit Tests
# ============================================================================
def _create_module_tests():  # pyright: ignore[reportUnusedFunction]
    """Create unit tests for scheduler module."""
    import os
    import tempfile

    from test_framework import TestSuite

    suite = TestSuite("Scheduler Tests")

    def test_scheduler_init():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            scheduler = Scheduler(db)
            assert scheduler.db is db
        finally:
            os.unlink(db_path)

    def test_get_next_valid_start_time():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            scheduler = Scheduler(db)
            # Test when current time is before start hour
            morning = datetime(2024, 1, 15, 6, 0, 0)
            result = scheduler._get_next_valid_start_time(morning, 8, 18)
            assert result.hour == 8
            # Test when current time is after end hour
            evening = datetime(2024, 1, 15, 20, 0, 0)
            result = scheduler._get_next_valid_start_time(evening, 8, 18)
            assert result.hour == 8
            assert result.day == 16
        finally:
            os.unlink(db_path)

    def test_adjust_to_valid_hours():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            scheduler = Scheduler(db)
            # Test before start hour
            early = datetime(2024, 1, 15, 5, 0, 0)
            result = scheduler._adjust_to_valid_hours(early, 8, 18)
            assert result.hour == 8
            # Test after end hour
            late = datetime(2024, 1, 15, 20, 0, 0)
            result = scheduler._adjust_to_valid_hours(late, 8, 18)
            assert result.hour == 8
            assert result.day == 16
            # Test within valid hours
            valid = datetime(2024, 1, 15, 12, 0, 0)
            result = scheduler._adjust_to_valid_hours(valid, 8, 18)
            assert result.hour == 12
        finally:
            os.unlink(db_path)

    def test_calculate_schedule_times():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            scheduler = Scheduler(db)
            times = scheduler._calculate_schedule_times(3)
            assert len(times) == 3
            # All times should be unique
            assert len(set(times)) == 3
            # All times should be in valid hours
            for t in times:
                assert Config.PUBLISH_START_HOUR <= t.hour < Config.PUBLISH_END_HOUR
        finally:
            os.unlink(db_path)

    def test_get_schedule_summary_empty():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db = Database(db_path)
            scheduler = Scheduler(db)
            summary = scheduler.get_schedule_summary()
            assert "No stories currently scheduled" in summary
        finally:
            os.unlink(db_path)

    suite.add_test("Scheduler init", test_scheduler_init)
    suite.add_test("Get next valid start time", test_get_next_valid_start_time)
    suite.add_test("Adjust to valid hours", test_adjust_to_valid_hours)
    suite.add_test("Calculate schedule times", test_calculate_schedule_times)
    suite.add_test("Get schedule summary empty", test_get_schedule_summary_empty)

    return suite
