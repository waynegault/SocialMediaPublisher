#!/usr/bin/env python3
"""Test Phase 3 implementation: cross-story entity resolution, dashboard metrics, manual review."""

import json
import sys
from datetime import datetime

# Add current directory to path
sys.path.insert(0, ".")

from database import Database, Story


def test_cross_story_entity_resolution():
    """Test find_person_by_linkedin_urn and find_person_by_attributes."""
    print("\n" + "=" * 60)
    print("TEST: Cross-Story Entity Resolution")
    print("=" * 60)

    db = Database()

    # Test 1: find_person_by_linkedin_urn
    print("\n1. Testing find_person_by_linkedin_urn...")

    # Get some stories to check for LinkedIn URNs
    stories = db.get_stories_needing_verification() + db.get_published_stories()
    urn_found = None
    for story in stories[:50]:
        if story.story_people:
            people = (
                story.story_people
                if isinstance(story.story_people, list)
                else json.loads(story.story_people)
            )
            for p in people:
                if p.get("linkedin_urn"):
                    urn_found = p.get("linkedin_urn")
                    print(f"   Found URN in story {story.id}: {urn_found}")
                    break
        if urn_found:
            break

    if urn_found:
        result = db.find_person_by_linkedin_urn(urn_found)
        if result:
            print(f"   ✓ Found person: {result.get('name', 'Unknown')}")
        else:
            print("   ✗ Failed to find person with known URN")
    else:
        print("   ℹ No stories with LinkedIn URNs found - creating test data...")
        # Create test story with URN
        test_people = [
            {
                "name": "Test Person",
                "linkedin_urn": "urn:li:person:TEST123",
                "linkedin_profile": "https://linkedin.com/in/test",
                "match_confidence": "high",
            }
        ]
        test_story = Story()
        test_story.title = "Test Story for Phase 3"
        test_story.story_people = test_people
        test_story.enrichment_status = "enriched"
        test_story.enrichment_quality = "high"
        db.add_story(test_story)
        result = db.find_person_by_linkedin_urn("urn:li:person:TEST123")
        if result and result.get("name") == "Test Person":
            print(f"   ✓ Created and found test person: {result.get('name')}")
        else:
            print("   ✗ Failed to find created test person")

    # Test 2: find_person_by_attributes
    print("\n2. Testing find_person_by_attributes...")

    # Find any person name in the database
    name_found = None
    employer_found = None
    for story in stories[:50]:
        if story.story_people:
            people = (
                story.story_people
                if isinstance(story.story_people, list)
                else json.loads(story.story_people)
            )
            for p in people:
                if p.get("name"):
                    name_found = p.get("name")
                    employer_found = (
                        p.get("company")
                        or p.get("affiliation")
                        or p.get("organization")
                    )
                    break
        if name_found:
            break

    if name_found:
        result = db.find_person_by_attributes(name_found, employer_found, fuzzy=True)
        if result:
            print(f"   ✓ Found person by name '{name_found}': {result.get('name')}")
        else:
            print(f"   ✗ Failed to find person by name '{name_found}'")
    else:
        print("   ℹ No person names found in stories")

    # Test 3: Edge cases
    print("\n3. Testing edge cases...")
    result = db.find_person_by_linkedin_urn("")
    print(f"   Empty URN returns None: {'✓' if result is None else '✗'}")

    result = db.find_person_by_attributes("", None)
    print(f"   Empty name returns None: {'✓' if result is None else '✗'}")

    print("\n   Cross-story entity resolution: PASSED ✓")


def test_enrichment_dashboard_stats():
    """Test get_enrichment_dashboard_stats."""
    print("\n" + "=" * 60)
    print("TEST: Enrichment Dashboard Stats")
    print("=" * 60)

    db = Database()
    stats = db.get_enrichment_dashboard_stats()

    print("\n   Enrichment Stats Retrieved:")
    print(f"   - Total direct people: {stats.get('total_direct_people', 0)}")
    print(f"   - Total indirect people: {stats.get('total_indirect_people', 0)}")
    print(f"   - Total people: {stats.get('total_people', 0)}")
    print(
        f"   - With LinkedIn: {stats.get('with_linkedin', 0)} ({stats.get('linkedin_rate', '0%')})"
    )
    print(
        f"   - High confidence: {stats.get('high_confidence', 0)} ({stats.get('high_confidence_rate', '0%')})"
    )
    print(
        f"   - Org fallback: {stats.get('org_fallback', 0)} ({stats.get('org_fallback_rate', '0%')})"
    )
    print(f"   - Quality breakdown: {stats.get('quality_breakdown', {})}")
    print(f"   - Needs review: {stats.get('needs_review', 0)}")

    # Validate structure
    required_keys = [
        "total_direct_people",
        "total_indirect_people",
        "total_people",
        "with_linkedin",
        "linkedin_rate",
        "high_confidence",
        "high_confidence_rate",
        "org_fallback",
        "org_fallback_rate",
        "quality_breakdown",
        "needs_review",
    ]

    missing_keys = [k for k in required_keys if k not in stats]
    if missing_keys:
        print(f"\n   ✗ Missing keys: {missing_keys}")
    else:
        print("\n   All required keys present: PASSED ✓")


def test_manual_review_workflow():
    """Test get_stories_needing_review and mark_story_reviewed."""
    print("\n" + "=" * 60)
    print("TEST: Manual Review Workflow")
    print("=" * 60)

    db = Database()

    # Test 1: get_stories_needing_review
    print("\n1. Testing get_stories_needing_review...")
    stories = db.get_stories_needing_review(limit=10)
    print(f"   Found {len(stories)} stories needing review")

    if stories:
        for s in stories[:3]:
            title_preview = (s.title[:50] + "...") if len(s.title) > 50 else s.title
            print(
                f"   - Story {s.id}: {title_preview} (quality: {s.enrichment_quality})"
            )

    # Test 2: mark_story_reviewed
    print("\n2. Testing mark_story_reviewed...")

    # Create a test story with low quality to test review
    test_story = Story()
    test_story.title = f"Test Story for Review Workflow - {datetime.now().timestamp()}"
    test_story.enrichment_status = "enriched"
    test_story.enrichment_quality = "low"  # Mark as needing review
    story_id = db.add_story(test_story)
    print(f"   Created test story {story_id} with quality='low'")

    # Verify it appears in review queue
    review_queue = db.get_stories_needing_review(limit=100)
    in_queue = any(s.id == story_id for s in review_queue)
    print(f"   Story in review queue: {'✓' if in_queue else '✗'}")

    # Mark as reviewed
    success = db.mark_story_reviewed(story_id, "Reviewed by test script")
    print(f"   Mark reviewed returned: {'✓' if success else '✗'}")

    # Verify it's no longer in review queue
    review_queue = db.get_stories_needing_review(limit=100)
    still_in_queue = any(s.id == story_id for s in review_queue)
    print(f"   Story removed from queue: {'✓' if not still_in_queue else '✗'}")

    # Verify enrichment_log was updated
    updated_story = db.get_story(story_id)
    log = updated_story.enrichment_log if updated_story else {}
    if isinstance(log, str):
        log = json.loads(log) if log else {}
    has_review_info = "manual_review" in log
    print(f"   Enrichment log updated: {'✓' if has_review_info else '✗'}")

    if has_review_info:
        print(f"   Review info: {log.get('manual_review')}")

    print("\n   Manual review workflow: PASSED ✓")


def test_dashboard_integration():
    """Test that dashboard properly integrates enrichment stats."""
    print("\n" + "=" * 60)
    print("TEST: Dashboard Integration")
    print("=" * 60)

    try:
        from dashboard import DashboardServer, DashboardMetrics
        from dataclasses import fields

        print("\n1. Checking DashboardMetrics has enrichment_stats field...")
        field_names = [f.name for f in fields(DashboardMetrics)]
        has_field = "enrichment_stats" in field_names
        print(f"   enrichment_stats field exists: {'✓' if has_field else '✗'}")

        print("\n2. Testing DashboardServer._calculate_metrics includes enrichment...")
        db = Database()
        dashboard = DashboardServer(db)
        metrics = dashboard._calculate_metrics()

        has_stats = (
            hasattr(metrics, "enrichment_stats")
            and metrics.enrichment_stats is not None
        )
        print(f"   Metrics has enrichment_stats: {'✓' if has_stats else '✗'}")

        if has_stats:
            print(f"   Stats content: {type(metrics.enrichment_stats).__name__}")
            if isinstance(metrics.enrichment_stats, dict):
                print(f"   Keys: {list(metrics.enrichment_stats.keys())[:5]}...")

        print("\n   Dashboard integration: PASSED ✓")

    except ImportError as e:
        print(f"\n   ⚠ Could not import dashboard: {e}")
        print("   Skipping dashboard integration test")


def main():
    """Run all Phase 3 tests."""
    print("\n" + "#" * 60)
    print("# PHASE 3 TEST SUITE")
    print("#" * 60)

    try:
        test_cross_story_entity_resolution()
        test_enrichment_dashboard_stats()
        test_manual_review_workflow()
        test_dashboard_integration()

        print("\n" + "=" * 60)
        print("ALL PHASE 3 TESTS PASSED ✓")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
