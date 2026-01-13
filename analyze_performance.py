#!/usr/bin/env python3
"""Analyze database and log for enrichment performance."""

import sqlite3
import json
from datetime import datetime
from pathlib import Path


def analyze_database():
    """Analyze current database state."""
    print("\n" + "=" * 70)
    print("DATABASE ANALYSIS")
    print("=" * 70)

    conn = sqlite3.connect("content_engine.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all stories - check which columns exist
    cursor.execute("PRAGMA table_info(stories)")
    columns = {row[1] for row in cursor.fetchall()}

    # Build query based on available columns
    base_cols = "id, title, verification_status, enrichment_status, image_path, story_people, org_leaders"
    optional_cols = []
    if "enrichment_log" in columns:
        optional_cols.append("enrichment_log")
    if "match_confidence" in columns:
        optional_cols.append("match_confidence")
    if "enrichment_quality" in columns:
        optional_cols.append("enrichment_quality")

    all_cols = base_cols + (", " + ", ".join(optional_cols) if optional_cols else "")
    cursor.execute(f"SELECT {all_cols} FROM stories")
    rows = cursor.fetchall()

    print(f"\nTotal stories: {len(rows)}")
    print("-" * 70)

    total_people = 0
    linkedin_found = 0
    enriched_count = 0
    pending_count = 0

    for row in rows:
        story_id = row["id"]
        title = row["title"][:50] + "..." if len(row["title"]) > 50 else row["title"]
        v_status = row["verification_status"]
        e_status = row["enrichment_status"]
        has_image = bool(row["image_path"])
        quality = (
            row["enrichment_quality"] if "enrichment_quality" in row.keys() else "n/a"
        )
        confidence = (
            row["match_confidence"] if "match_confidence" in row.keys() else "n/a"
        )

        # Parse story_people
        story_people = []
        if row["story_people"]:
            try:
                story_people = json.loads(row["story_people"])
            except Exception:
                pass

        # Parse org_leaders
        org_leaders = []
        if row["org_leaders"]:
            try:
                org_leaders = json.loads(row["org_leaders"])
            except Exception:
                pass

        people_count = len(story_people)
        leaders_count = len(org_leaders)
        total_people += people_count + leaders_count

        # Count LinkedIn profiles found
        linkedin_in_people = sum(
            1
            for p in story_people
            if p.get("linkedin_profile") or p.get("linkedin_url")
        )
        linkedin_in_leaders = sum(
            1 for p in org_leaders if p.get("linkedin_profile") or p.get("linkedin_url")
        )
        linkedin_found += linkedin_in_people + linkedin_in_leaders

        if e_status == "enriched":
            enriched_count += 1
        elif e_status == "pending":
            pending_count += 1

        print(f"\nStory {story_id}: {title}")
        print(f"  Status: verify={v_status}, enrich={e_status}, image={has_image}")
        print(f"  Quality: {quality}, Confidence: {confidence}")
        print(f"  People: {people_count} story_people, {leaders_count} org_leaders")
        print(f"  LinkedIn: {linkedin_in_people + linkedin_in_leaders} profiles found")

        # Show enrichment log if available
        if "enrichment_log" in row.keys() and row["enrichment_log"]:
            try:
                log = json.loads(row["enrichment_log"])
                if log:
                    print(f"  Enrichment Log: {json.dumps(log, indent=4)[:200]}...")
            except Exception:
                pass

    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  Total Stories: {len(rows)}")
    print(f"  Enriched: {enriched_count}")
    print(f"  Pending Enrichment: {pending_count}")
    print(f"  Total People Identified: {total_people}")
    print(f"  LinkedIn Profiles Found: {linkedin_found}")
    if total_people > 0:
        success_rate = (linkedin_found / total_people) * 100
        print(f"  LinkedIn Match Rate: {success_rate:.1f}%")
    else:
        print("  LinkedIn Match Rate: N/A (no people)")

    conn.close()


def analyze_log():
    """Analyze the search run log."""
    print("\n" + "=" * 70)
    print("LOG ANALYSIS")
    print("=" * 70)

    log_path = Path("search_run.log")
    if not log_path.exists():
        print("  No search_run.log found")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    # Key metrics from log
    lines = log_content.split("\n")

    # Find enrichment-related lines
    enrichment_lines = [l for l in lines if "enrich" in l.lower()]
    print(f"\n  Enrichment log lines: {len(enrichment_lines)}")
    for line in enrichment_lines[:10]:
        print(f"    {line}")

    # Find verification rejections
    rejection_lines = [l for l in lines if "REJECTED" in l]
    print(f"\n  Verification rejections: {len(rejection_lines)}")
    for line in rejection_lines:
        # Extract reason
        if ":" in line:
            reason = line.split("REJECTED:")[-1].strip()
            print(f"    - {reason}")

    # Find cache stats
    cache_lines = [l for l in lines if "CACHE_STATS" in l]
    if cache_lines:
        print(f"\n  Cache Stats: {cache_lines[-1]}")


def identify_issues():
    """Identify the root cause of poor @mention discovery."""
    print("\n" + "=" * 70)
    print("ISSUE ANALYSIS")
    print("=" * 70)

    conn = sqlite3.connect("content_engine.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check stories needing enrichment query
    cursor.execute(
        """
        SELECT id, title, enrichment_status, verification_status, image_path
        FROM stories
        WHERE enrichment_status = 'pending'
        """
    )
    pending = cursor.fetchall()

    cursor.execute(
        """
        SELECT id, title
        FROM stories
        WHERE enrichment_status = 'pending'
        AND verification_status = 'approved'
        AND image_path IS NOT NULL
        """
    )
    matches_query = cursor.fetchall()

    print(f"\n  Stories with enrichment_status='pending': {len(pending)}")
    print(
        f"  Stories matching get_stories_needing_enrichment query: {len(matches_query)}"
    )

    if len(pending) > 0 and len(matches_query) == 0:
        print("\n  ⚠️  ISSUE FOUND: Pipeline order mismatch!")
        print(
            "  The query requires verification_status='approved' AND image_path IS NOT NULL"
        )
        print("  But enrichment runs BEFORE verification in the pipeline.")
        print("")
        print("  Current pending stories that don't match:")
        for row in pending:
            print(
                f"    Story {row['id']}: verify={row['verification_status']}, "
                f"image={bool(row['image_path'])}"
            )

    conn.close()


def main():
    print("\n" + "#" * 70)
    print("# ENRICHMENT PERFORMANCE ANALYSIS")
    print(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    analyze_database()
    analyze_log()
    identify_issues()

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. FIX DATABASE QUERY: The get_stories_needing_enrichment() query is too
   restrictive. It requires verification_status='approved', but enrichment
   runs BEFORE verification in the pipeline.

2. OPTION A - Change query to not require verification:
   Remove 'AND verification_status = 'approved'' from the query

3. OPTION B - Reorder pipeline:
   Run enrichment AFTER verification (but this delays LinkedIn lookups)

4. RECOMMENDED: Option A - stories should be enriched regardless of
   verification status since verification DEPENDS on enrichment.
""")


if __name__ == "__main__":
    main()
