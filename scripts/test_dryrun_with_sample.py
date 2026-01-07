"""Create a temporary approved story, run the linkedin dry-run publish to show payloads, then clean up the story."""
import sys
import os
# Ensure project root is on sys.path when run from the scripts/ folder
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database import Database, Story

from pathlib import Path

def main():
    db = Database()
    # Create a temporary approved, unpublished story
    story = Story(
        title="TEST: Dry-run publish sample",
        summary="This is a test summary for dry-run publish payload inspection.",
        source_links=["https://example.com/test-article"],
        quality_score=9,
        verification_status="approved",
        publish_status="unpublished",
    )
    story_id = db.add_story(story)
    print(f"Added temporary story with ID {story_id}")

    try:
        # Run the dry-run publication (prints payloads)
        import linkedin_dryrun_publish

        linkedin_dryrun_publish.main()
    finally:
        # Clean up
        deleted = db.delete_story(story_id)
        print(f"Cleaned up story {story_id}: deleted={deleted}")


if __name__ == '__main__':
    main()
