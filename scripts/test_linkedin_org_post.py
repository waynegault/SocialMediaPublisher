"""One-off test script to publish a short test post to an organization page.

Usage:
  python scripts/test_linkedin_org_post.py --message "Test message" --yes

By default this will use LINKEDIN_ORGANIZATION_URN from .env. Pass --yes to skip interactive confirmation.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import os
import sys

def _ensure_venv():
    proj_root = Path(__file__).parent.parent
    venv_dir = proj_root / '.venv'
    if venv_dir.exists():
        venv_py = venv_dir / ('Scripts' if os.name == 'nt' else 'bin') / (
            'python.exe' if os.name == 'nt' else 'python'
        )
        if venv_py.exists():
            venv_path = str(venv_py)
            if os.path.abspath(sys.executable) != os.path.abspath(venv_path):
                os.execv(venv_path, [venv_path] + sys.argv)

_ensure_venv()

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env', override=False)

from config import Config
from database import Database
from linkedin_publisher import LinkedInPublisher


def main() -> int:
    parser = argparse.ArgumentParser(description='Publish a one-off test post to a LinkedIn organization.')
    parser.add_argument('--org', help='Organization URN (overrides env)')
    parser.add_argument('--message', help='Message text to post', default=None)
    parser.add_argument('--yes', help='Skip confirmation and post immediately', action='store_true')
    args = parser.parse_args()

    org_urn = args.org or getattr(Config, 'LINKEDIN_ORGANIZATION_URN', None)
    if not org_urn:
        print('ERROR: No organization URN provided and LINKEDIN_ORGANIZATION_URN not set in env.')
        return 2

    message = args.message or f"Test post from SocialMediaPublisher at {datetime.utcnow().isoformat()} UTC - please ignore"

    print('Organization URN:', org_urn)
    print('Message preview:')
    print(message)

    if not args.yes:
        resp = input('Proceed to POST this message to LinkedIn (y/n)? ')
        if resp.strip().lower() not in ('y', 'yes'):
            print('Aborted by user.')
            return 0

    db = Database()
    publisher = LinkedInPublisher(db)

    print('Posting...')
    post_id = publisher.publish_one_off(org_urn, message)

    if post_id:
        print(f'Post successful. Post ID: {post_id}')
        return 0
    else:
        print('Post failed. Check logs for details.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())