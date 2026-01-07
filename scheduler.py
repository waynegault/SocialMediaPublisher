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

    def schedule_stories(self) -> list[Story]:
        """
        Schedule stories for publication.
        Returns the list of scheduled stories.
        """
        target_count = Config.STORIES_PER_CYCLE

        # Clear any existing scheduled stories (re-scheduling)
        cleared = self.db.clear_scheduled_status()
        if cleared > 0:
            logger.info(f"Cleared {cleared} previously scheduled stories")

        # Get candidates in quality order
        candidates = self.db.get_approved_unpublished_stories(target_count)

        if not candidates:
            logger.warning("No approved stories available to schedule")
            return []

        if len(candidates) < target_count:
            logger.warning(
                f"Only {len(candidates)} stories available (target: {target_count})"
            )

        # Calculate publication times
        scheduled_times = self._calculate_schedule_times(len(candidates))

        # Assign times to stories
        scheduled_stories = []
        for story, scheduled_time in zip(candidates, scheduled_times):
            story.scheduled_time = scheduled_time
            story.publish_status = "scheduled"
            self.db.update_story(story)
            scheduled_stories.append(story)
            logger.info(
                f"Scheduled story {story.id} for {scheduled_time.strftime('%Y-%m-%d %H:%M')}: "
                f"{story.title[:50]}..."
            )

        logger.info(f"Scheduled {len(scheduled_stories)} stories for publication")
        return scheduled_stories

    def _calculate_schedule_times(self, count: int) -> list[datetime]:
        """
        Calculate publication times for a given number of stories.
        Distributes them evenly across the publication window with jitter.
        """
        if count == 0:
            return []

        now = datetime.now()
        start_hour = Config.PUBLISH_START_HOUR
        end_hour = Config.PUBLISH_END_HOUR
        window_hours = Config.PUBLISH_WINDOW_HOURS
        jitter_minutes = Config.JITTER_MINUTES

        # Determine the base start time
        base_time = self._get_next_valid_start_time(now, start_hour, end_hour)

        # Calculate the available publishing hours per day
        daily_hours = end_hour - start_hour

        # Calculate interval between posts
        if count == 1:
            interval_hours = 0
        else:
            # Spread across the window, but constrained to valid hours
            interval_hours = min(window_hours / count, daily_hours / count)

        scheduled_times = []
        current_time = base_time

        for _ in range(count):
            # Add jitter
            jitter = random.randint(-jitter_minutes, jitter_minutes)
            scheduled_time = current_time + timedelta(minutes=jitter)

            # Ensure time is within valid hours
            scheduled_time = self._adjust_to_valid_hours(
                scheduled_time, start_hour, end_hour
            )

            scheduled_times.append(scheduled_time)

            # Move to next slot
            current_time += timedelta(hours=interval_hours)

            # If we've gone past end hour, move to next day
            if current_time.hour >= end_hour:
                current_time = current_time.replace(
                    hour=start_hour, minute=0, second=0, microsecond=0
                )
                current_time += timedelta(days=1)

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
            lines.append(f"  [{time_str}] {story.title[:60]}...")

        return "\n".join(lines)
