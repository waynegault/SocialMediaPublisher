#!/usr/bin/env python3
"""Show story details including mentions."""

from database import Database
import sys

db = Database()

# Get story by title or ID
if len(sys.argv) > 1:
    title = " ".join(sys.argv[1:])
    s = db.get_story_by_title(title)
else:
    # Get the latest approved story
    stories = db.get_approved_unpublished_stories(1)
    if not stories:
        print("No approved stories found")
        sys.exit(1)
    s = stories[0]

if not s:
    print(f"Story not found")
    sys.exit(1)

print("=" * 70)
print(f"STORY: {s.title}")
print("=" * 70)
print(f"\nSOURCE: {s.source_links}")
print(f"\nSUMMARY:\n{s.summary}")
print(f"\nPROMOTION:\n{s.promotion}")
print(f"\nORGANIZATIONS: {s.organizations}")
print("\n" + "-" * 70)
print("DIRECT PEOPLE (direct_people):")
print("-" * 70)
if s.direct_people:
    for i, p in enumerate(s.direct_people, 1):
        name = p.get("name", "Unknown")
        title = p.get("position", p.get("title", "N/A"))
        company = p.get("company", p.get("affiliation", "N/A"))
        linkedin = p.get("linkedin_profile", p.get("linkedin_url", "Not found"))
        print(f"  {i}. {name}")
        print(f"     Title: {title}")
        print(f"     Company: {company}")
        print(f"     LinkedIn: {linkedin}")
        print()
else:
    print("  None")
print("-" * 70)
print("INDIRECT PEOPLE (indirect_people):")
print("-" * 70)
if s.indirect_people:
    for i, p in enumerate(s.indirect_people, 1):
        name = p.get("name", "Unknown")
        title = p.get("title", "N/A")
        org = p.get("organization", "N/A")
        linkedin = p.get("linkedin_profile", "Not found")
        print(f"  {i}. {name}")
        print(f"     Title: {title}")
        print(f"     Organization: {org}")
        print(f"     LinkedIn: {linkedin}")
        print()
else:
    print("  None")
