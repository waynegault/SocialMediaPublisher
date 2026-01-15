#!/usr/bin/env python3
"""Test remaining spec implementation: alerts, incremental updates, exports."""

import json
import sys

sys.path.insert(0, ".")

from database import Database, Story
from company_mention_enricher import (
    EnrichmentMetrics,
    EnrichmentAlerts,
    needs_refresh,
    get_stories_needing_refresh,
    export_direct_people,
    export_all_people,
)


def test_enrichment_alerts():
    """Test Step 5.4: Alerts and Notifications."""
    print("\n" + "=" * 60)
    print("TEST: Step 5.4 - Enrichment Alerts")
    print("=" * 60)

    alerts = EnrichmentAlerts()

    # Test 1: Critical alert - low success rate
    print("\n1. Testing critical alert (low success rate)...")
    metrics = EnrichmentMetrics()
    metrics.stories_processed = 10
    metrics.direct_people_found = 50
    metrics.indirect_people_found = 50
    metrics.linkedin_matches = 20  # 20% match rate - critical!

    alert_list = alerts.check_metrics(metrics)
    critical = [a for a in alert_list if a["severity"] == "critical"]
    assert len(critical) >= 1, "Should have critical alert for low success"
    print(f"   ✓ Critical alert triggered: {critical[0]['message']}")

    # Test 2: Warning alert - high low-confidence rate
    print("\n2. Testing warning alert (high low-confidence)...")
    metrics2 = EnrichmentMetrics()
    metrics2.stories_processed = 10
    metrics2.direct_people_found = 20
    metrics2.indirect_people_found = 10
    metrics2.linkedin_matches = 25
    metrics2.low_confidence_matches = 12  # 48% low confidence - warning!
    metrics2.high_confidence_matches = 13

    alerts2 = EnrichmentAlerts()
    alert_list2 = alerts2.check_metrics(metrics2)
    warnings = [a for a in alert_list2 if a["severity"] == "warning"]
    assert any("low-confidence" in w["message"].lower() for w in warnings)
    print(f"   ✓ Warning alert triggered: {len(warnings)} warnings")

    # Test 3: Info alert - excellent high confidence
    print("\n3. Testing info alert (excellent performance)...")
    metrics3 = EnrichmentMetrics()
    metrics3.stories_processed = 10
    metrics3.direct_people_found = 20
    metrics3.indirect_people_found = 10
    metrics3.linkedin_matches = 28
    metrics3.high_confidence_matches = 25  # 89% high confidence - excellent!

    alerts3 = EnrichmentAlerts()
    alert_list3 = alerts3.check_metrics(metrics3)
    infos = [a for a in alert_list3 if a["severity"] == "info"]
    assert len(infos) >= 1, "Should have info alert for excellent performance"
    print(f"   ✓ Info alert triggered: {infos[0]['message']}")

    # Test 4: Alert structure
    print("\n4. Testing alert structure...")
    assert all("severity" in a for a in alert_list)
    assert all("message" in a for a in alert_list)
    assert all("context" in a for a in alert_list)
    assert all("timestamp" in a for a in alert_list)
    print("   ✓ All alerts have correct structure")

    print("\n   Step 5.4 Alerts: PASSED ✓")


def test_incremental_updates():
    """Test Step 6.2: Incremental Updates."""
    print("\n" + "=" * 60)
    print("TEST: Step 6.2 - Incremental Updates")
    print("=" * 60)

    # Test 1: Never enriched story needs refresh
    print("\n1. Testing never-enriched story...")
    story = Story()
    story.id = 1
    story.title = "Test Story"
    story.enrichment_status = "pending"
    assert needs_refresh(story) is True
    print("   ✓ Pending story needs refresh")

    # Test 2: Error story needs refresh
    print("\n2. Testing error story...")
    story2 = Story()
    story2.id = 2
    story2.title = "Error Story"
    story2.enrichment_status = "error"
    assert needs_refresh(story2) is True
    print("   ✓ Error story needs refresh")

    # Test 3: Recently enriched story does NOT need refresh
    print("\n3. Testing recently enriched story...")
    from datetime import datetime

    story3 = Story()
    story3.id = 3
    story3.title = "Recent Story"
    story3.enrichment_status = "enriched"
    story3.enrichment_quality = "high"
    story3.enrichment_log = {"completed_at": datetime.now().isoformat()}
    assert needs_refresh(story3) is False
    print("   ✓ Recently enriched story does NOT need refresh")

    # Test 4: Low quality story needs refresh after 7 days
    print("\n4. Testing low quality story (old)...")
    from datetime import timedelta

    old_date = (datetime.now() - timedelta(days=10)).isoformat()
    story4 = Story()
    story4.id = 4
    story4.title = "Low Quality Story"
    story4.enrichment_status = "enriched"
    story4.enrichment_quality = "low"
    story4.enrichment_log = {"completed_at": old_date}
    assert needs_refresh(story4) is True
    print("   ✓ Old low-quality story needs refresh")

    # Test 5: get_stories_needing_refresh
    print("\n5. Testing get_stories_needing_refresh...")
    db = Database()
    stories = get_stories_needing_refresh(db, limit=5)
    print(f"   ✓ Found {len(stories)} stories needing refresh")

    print("\n   Step 6.2 Incremental Updates: PASSED ✓")


def test_export_options():
    """Test Step 6.5: Export Options."""
    print("\n" + "=" * 60)
    print("TEST: Step 6.5 - Export Options")
    print("=" * 60)

    # Create test story with people
    story = Story()
    story.id = 999
    story.title = "Export Test Story"
    story.direct_people = [
        {
            "name": "Dr. Jane Smith",
            "position": "Lead Researcher",
            "company": "MIT",
            "linkedin_profile": "https://linkedin.com/in/janesmith",
            "match_confidence": "high",
        },
        {
            "name": "John Doe",
            "title": "Engineer",
            "affiliation": "Stanford",
            "linkedin_url": "https://linkedin.com/in/johndoe",
            "match_confidence": "medium",
        },
    ]
    story.indirect_people = [
        {
            "name": "CEO Person",
            "title": "CEO",
            "organization": "MIT",
            "linkedin_profile": "https://linkedin.com/in/ceo",
            "match_confidence": "high",
        }
    ]
    story.enrichment_quality = "high"

    # Test 1: JSON export
    print("\n1. Testing JSON export...")
    json_output = export_direct_people(story, "json")
    parsed = json.loads(json_output)
    assert "direct_people" in parsed
    assert "indirect_people" in parsed
    assert len(parsed["direct_people"]) == 2
    assert len(parsed["indirect_people"]) == 1
    print(
        f"   ✓ JSON export: {len(parsed['direct_people'])} direct, {len(parsed['indirect_people'])} indirect"
    )

    # Test 2: CSV export
    print("\n2. Testing CSV export...")
    csv_output = export_direct_people(story, "csv")
    lines = csv_output.strip().split("\n")
    assert len(lines) == 4  # Header + 3 people
    assert "category,name,title" in lines[0]
    print(f"   ✓ CSV export: {len(lines) - 1} people rows")

    # Test 3: Markdown export
    print("\n3. Testing Markdown export...")
    md_output = export_direct_people(story, "markdown")
    assert "# People in Story:" in md_output
    assert "## Direct People" in md_output
    assert "## Indirect People" in md_output
    assert "Dr. Jane Smith" in md_output
    print(f"   ✓ Markdown export: {len(md_output)} chars")

    # Test 4: Export all people
    print("\n4. Testing export_all_people...")
    db = Database()
    all_json = export_all_people(db, "json")
    parsed_all = json.loads(all_json)
    assert "total_people" in parsed_all
    assert "people" in parsed_all
    print(f"   ✓ All people export: {parsed_all['total_people']} total")

    # Test 5: CSV all export
    print("\n5. Testing CSV all export...")
    all_csv = export_all_people(db, "csv")
    csv_lines = all_csv.strip().split("\n")
    print(f"   ✓ CSV all export: {len(csv_lines) - 1} people rows")

    print("\n   Step 6.5 Export Options: PASSED ✓")


def test_dashboard_endpoints():
    """Test dashboard API endpoints for new features."""
    print("\n" + "=" * 60)
    print("TEST: Dashboard API Endpoints")
    print("=" * 60)

    # We can't easily test Flask routes without running the server,
    # but we can verify the imports work
    print("\n1. Testing dashboard imports...")
    from dashboard import DashboardServer

    db = Database()
    server = DashboardServer(db)

    # Check that routes are registered
    print("\n2. Checking route registration...")
    routes = [rule.rule for rule in server.app.url_map.iter_rules()]

    assert "/api/dashboard/export/<int:story_id>" in routes
    print("   ✓ /api/dashboard/export/<story_id> registered")

    assert "/api/dashboard/export/all" in routes
    print("   ✓ /api/dashboard/export/all registered")

    assert "/api/dashboard/alerts" in routes
    print("   ✓ /api/dashboard/alerts registered")

    assert "/api/dashboard/refresh-needed" in routes
    print("   ✓ /api/dashboard/refresh-needed registered")

    print("\n   Dashboard Endpoints: PASSED ✓")


def main():
    """Run all remaining spec tests."""
    print("\n" + "#" * 60)
    print("# REMAINING SPEC IMPLEMENTATION TESTS")
    print("#" * 60)

    try:
        test_enrichment_alerts()
        test_incremental_updates()
        test_export_options()
        test_dashboard_endpoints()

        print("\n" + "=" * 60)
        print("ALL REMAINING SPEC TESTS PASSED ✓")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
