"""Dry-run LinkedIn publish utility.

This script builds the same payloads the publisher would send to LinkedIn
but does not make any network requests. It prints masked payloads for
inspection so you can verify content, media attachments, and author URN.

Usage:
  python linkedin_dryrun_publish.py

It will prefer scheduled due stories; if none are due it will use
approved unpublished stories up to the configured STORIES_PER_CYCLE.
"""

from __future__ import annotations
from pathlib import Path
import json
import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / '.env', override=False)

from config import Config
from database import Database
from linkedin_publisher import LinkedInPublisher


def mask_url(u: str) -> str:
    # Simple masking for long URLs
    if not u:
        return u
    return u[:40] + '...'


def main() -> int:
    print("=== Dry-run LinkedIn publish ===")
    db = Database()
    publisher = LinkedInPublisher(db)

    # Prefer scheduled due stories
    due = db.get_scheduled_stories_due()
    if due:
        stories = due
        source = 'scheduled_due'
    else:
        stories = db.get_approved_unpublished_stories(Config.STORIES_PER_CYCLE)
        source = 'approved_unpublished'

    if not stories:
        print("No stories found to publish (no scheduled due or approved unpublished).")
        return 0

    print(f"Found {len(stories)} stories from {source} to dry-run publish:")

    for s in stories:
        print('\n---')
        print(f"Story ID: {s.id} | Title: {s.title}")
        print(f"Quality: {s.quality_score} | Scheduled: {s.scheduled_time}")
        text = publisher._format_post_text(s)
        print("\n-- Post text preview --")
        # Print first 500 characters of the post for review
        print(text[:500])
        if len(text) > 500:
            print("... (truncated)")

        # Build payloads
        if s.image_path and Path(s.image_path).exists():
            # Image would be uploaded then used as asset URN; show placeholder
            asset_urn = 'urn:li:digitalmediaAsset:DRY_RUN_ASSET'
            payload = publisher._build_image_post_payload(text, s, asset_urn)
            print("\n-- Image post payload --")
            # Mask any long URLs in the payload
            payload_copy = json.loads(json.dumps(payload))
            for m in payload_copy.get('specificContent', {}).get('com.linkedin.ugc.ShareContent', {}).get('media', []):
                if m.get('media'):
                    m['media'] = mask_url(m['media'])
            print(json.dumps(payload_copy, indent=2))
        else:
            payload = publisher._build_article_post_payload(text, s)
            print("\n-- Article post payload --")
            payload_copy = json.loads(json.dumps(payload))
            # Mask originalUrl if present
            medias = payload_copy.get('specificContent', {}).get('com.linkedin.ugc.ShareContent', {}).get('media', [])
            for m in medias:
                if m.get('originalUrl'):
                    m['originalUrl'] = mask_url(m['originalUrl'])
            print(json.dumps(payload_copy, indent=2))

        # Note about actions that would be taken
        print("\nActions that would be performed:")
        if s.image_path and Path(s.image_path).exists():
            print(f"  - Register upload (assets API) for owner {Config.LINKEDIN_AUTHOR_URN}")
            print(f"  - Upload binary image from: {s.image_path}")
        print("  - POST to /ugcPosts with payload above")
        print("  - On success, update story.publish_status -> 'published' and store linkedin_post_id")

    print('\nDry-run complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())