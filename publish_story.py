"""Publish a specific story to LinkedIn (live). Use with caution.

Usage: python publish_story.py <story_id>
"""
import sys
from datetime import datetime
from database import Database
from linkedin_publisher import LinkedInPublisher

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python publish_story.py <story_id>')
        sys.exit(2)
    sid = int(sys.argv[1])
    db = Database()
    s = db.get_story(sid)
    if not s:
        print('Story not found', sid)
        sys.exit(1)
    print('Publishing story:', s.title)
    pub = LinkedInPublisher(db)
    post_id = pub._publish_story(s)
    print('post_id:', post_id)
    if post_id:
        s.publish_status = 'published'
        s.published_time = datetime.now()
        s.linkedin_post_id = post_id
        ok = db.update_story(s)
        print('DB update OK:', ok)
    else:
        print('Publish failed')
