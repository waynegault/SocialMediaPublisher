"""Verify the three fixes are working."""

import sqlite3
import json

conn = sqlite3.connect("content_engine.db")
cursor = conn.cursor()
cursor.execute(
    "SELECT id, title, summary, story_people, org_leaders, verification_status FROM stories"
)


def get_name_key(n):
    """Get first+last name for fuzzy matching."""
    parts = n.lower().split()
    if len(parts) >= 2:
        return (parts[0], parts[-1])
    return (n.lower(), "")


for row in cursor.fetchall():
    print("=" * 80)
    print(f"Story {row[0]}: {row[1][:60]}...")
    print(f"Status: {row[5]}")
    print(f"\nSummary:\n{row[2][:500]}")

    story_people = json.loads(row[3]) if row[3] else []
    org_leaders = json.loads(row[4]) if row[4] else []

    print(f"\n--- Story People ({len(story_people)}) ---")
    for p in story_people[:8]:
        name = p.get("name", "Unknown")
        url = p.get("linkedin_profile") or p.get("linkedin_url") or "NO URL"
        print(f"  {name}: {url}")

    print(f"\n--- Org Leaders ({len(org_leaders)}) ---")
    for p in org_leaders[:8]:
        name = p.get("name", "Unknown")
        url = p.get("linkedin_profile") or p.get("linkedin_url") or "NO URL"
        print(f"  {name}: {url}")

    # Check for exact duplicates
    story_names = [p.get("name", "").lower() for p in story_people]
    leader_names = [p.get("name", "").lower() for p in org_leaders]
    exact_dups = set(story_names) & set(leader_names)
    if exact_dups:
        print(f"\n⚠️ EXACT DUPLICATES: {exact_dups}")

    # Check for near-duplicates (first+last name match)
    story_keys = {get_name_key(p.get("name", "")) for p in story_people}
    near_dups = []
    for p in org_leaders:
        key = get_name_key(p.get("name", ""))
        if key in story_keys:
            near_dups.append(p.get("name"))

    if near_dups:
        print(f"\n⚠️ NEAR-DUPLICATES in org_leaders: {near_dups}")
    else:
        print("\n✅ No duplicates between story_people and org_leaders")

    # Check LinkedIn URLs
    all_people = story_people + org_leaders
    with_urls = [
        p for p in all_people if p.get("linkedin_profile") or p.get("linkedin_url")
    ]
    without_urls = [
        p
        for p in all_people
        if not (p.get("linkedin_profile") or p.get("linkedin_url"))
    ]

    print(f"\n📊 LinkedIn Coverage: {len(with_urls)}/{len(all_people)} have URLs")
    if with_urls:
        print("URLs found:")
        for p in with_urls[:5]:
            url = p.get("linkedin_profile") or p.get("linkedin_url")
            print(f"  ✓ {p['name']}: {url}")

conn.close()
