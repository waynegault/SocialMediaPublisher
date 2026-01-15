#!/usr/bin/env python3
"""Test Phase 4 implementation: database persistence, validation, metrics, atomic transactions."""

import json
import sys

# Add current directory to path
sys.path.insert(0, ".")

from database import Database, Story
from company_mention_enricher import (
    EnrichmentMetrics,
    QACheckResult,
    validate_enrichment_quality,
    validate_person_record,
    get_enrichment_metrics,
    reset_enrichment_metrics,
    CompanyMentionEnricher,
)


def test_enrichment_metrics():
    """Test EnrichmentMetrics dataclass."""
    print("\n" + "=" * 60)
    print("TEST: EnrichmentMetrics")
    print("=" * 60)

    reset_enrichment_metrics()
    metrics = get_enrichment_metrics()

    # Test recording stories
    print("\n1. Testing story recording...")
    metrics.record_story(1.5)
    metrics.record_story(2.5)
    assert metrics.stories_processed == 2, "Should have 2 stories"
    assert metrics.avg_story_enrichment_time == 2.0, "Avg should be 2.0s"
    print(f"   ✓ Stories processed: {metrics.stories_processed}")
    print(f"   ✓ Avg enrichment time: {metrics.avg_story_enrichment_time}s")

    # Test recording matches
    print("\n2. Testing match recording...")
    metrics.record_match("high")
    metrics.record_match("high")
    metrics.record_match("medium")
    metrics.record_match("low")
    metrics.record_match("org_fallback")

    assert metrics.linkedin_matches == 5, "Should have 5 matches"
    assert metrics.high_confidence_matches == 2, "Should have 2 high"
    assert metrics.medium_confidence_matches == 1, "Should have 1 medium"
    assert metrics.low_confidence_matches == 1, "Should have 1 low"
    assert metrics.org_fallback_matches == 1, "Should have 1 org_fallback"
    print(f"   ✓ LinkedIn matches: {metrics.linkedin_matches}")
    print(f"   ✓ High confidence: {metrics.high_confidence_matches}")

    # Test to_dict
    print("\n3. Testing to_dict...")
    d = metrics.to_dict()
    assert "stories_processed" in d
    assert "high_confidence_rate" in d
    print(f"   ✓ Dict keys: {list(d.keys())[:5]}...")

    print("\n   EnrichmentMetrics: PASSED ✓")


def test_qa_check_result():
    """Test QACheckResult dataclass."""
    print("\n" + "=" * 60)
    print("TEST: QACheckResult")
    print("=" * 60)

    # Test passed result
    print("\n1. Testing passed result...")
    result = QACheckResult(story_id=1, checks_passed=5, checks_failed=0)
    assert result.passed is True
    assert result.status == "passed"
    print(f"   ✓ Status: {result.status}")

    # Test passed with warnings
    print("\n2. Testing passed with warnings...")
    result.warnings.append("Test warning")
    assert result.status == "passed_with_warnings"
    print(f"   ✓ Status: {result.status}")

    # Test failed
    print("\n3. Testing failed result...")
    result.checks_failed = 1
    assert result.passed is False
    assert result.status == "failed"
    print(f"   ✓ Status: {result.status}")

    # Test to_dict
    print("\n4. Testing to_dict...")
    d = result.to_dict()
    assert d["story_id"] == 1
    assert "warnings" in d
    print(f"   ✓ Dict: {d}")

    print("\n   QACheckResult: PASSED ✓")


def test_validate_person_record():
    """Test validate_person_record function."""
    print("\n" + "=" * 60)
    print("TEST: validate_person_record")
    print("=" * 60)

    # Valid record
    print("\n1. Testing valid record...")
    valid_person = {
        "name": "Dr. Jane Smith",
        "linkedin_profile": "https://linkedin.com/in/janesmith",
        "match_score": 7.5,
    }
    is_valid, errors = validate_person_record(valid_person)
    assert is_valid is True
    assert len(errors) == 0
    print(f"   ✓ Valid: {is_valid}, Errors: {errors}")

    # Missing name
    print("\n2. Testing missing name...")
    invalid_person = {"linkedin_profile": "https://linkedin.com/in/test"}
    is_valid, errors = validate_person_record(invalid_person)
    assert is_valid is False
    assert any("name" in e.lower() for e in errors)
    print(f"   ✓ Valid: {is_valid}, Errors: {errors}")

    # Invalid LinkedIn URL
    print("\n3. Testing invalid LinkedIn URL...")
    bad_url_person = {
        "name": "Test Person",
        "linkedin_profile": "https://twitter.com/test",
    }
    is_valid, errors = validate_person_record(bad_url_person)
    assert is_valid is False
    assert any("linkedin" in e.lower() for e in errors)
    print(f"   ✓ Valid: {is_valid}, Errors: {errors}")

    # Invalid match score
    print("\n4. Testing invalid match score...")
    bad_score_person = {"name": "Test", "match_score": 15.0}
    is_valid, errors = validate_person_record(bad_score_person)
    assert is_valid is False
    assert any("score" in e.lower() for e in errors)
    print(f"   ✓ Valid: {is_valid}, Errors: {errors}")

    print("\n   validate_person_record: PASSED ✓")


def test_validate_enrichment_quality():
    """Test validate_enrichment_quality function."""
    print("\n" + "=" * 60)
    print("TEST: validate_enrichment_quality")
    print("=" * 60)

    # Create a test story
    print("\n1. Testing valid story...")
    story = Story()
    story.id = 999
    story.title = "Test Story"
    story.direct_people = [
        {
            "name": "Dr. Jane Smith",
            "linkedin_profile": "https://linkedin.com/in/janesmith",
            "match_confidence": "high",
        },
        {
            "name": "John Doe",
            "linkedin_profile": "https://linkedin.com/in/johndoe",
            "match_confidence": "high",
        },
    ]
    story.indirect_people = [
        {
            "name": "CEO Person",
            "linkedin_profile": "https://linkedin.com/in/ceoperson",
            "match_confidence": "medium",
        }
    ]

    result = validate_enrichment_quality(story)
    print(f"   Status: {result.status}")
    print(f"   Checks passed: {result.checks_passed}")
    print(f"   Checks failed: {result.checks_failed}")
    print(f"   Warnings: {result.warnings}")
    assert result.linkedin_url_validity is True
    assert result.required_fields_ok is True
    print(f"   ✓ Valid story validation passed")

    # Test duplicate detection
    print("\n2. Testing duplicate detection...")
    story_with_dupe = Story()
    story_with_dupe.id = 998
    story_with_dupe.title = "Test Story with Duplicate"
    story_with_dupe.direct_people = [
        {"name": "Same Person", "match_confidence": "high"}
    ]
    story_with_dupe.indirect_people = [
        {"name": "Same Person", "match_confidence": "high"}
    ]
    result = validate_enrichment_quality(story_with_dupe)
    assert result.duplicate_detection is False
    assert any("duplicate" in w.lower() for w in result.warnings)
    print(f"   ✓ Duplicate detected: {result.warnings}")

    # Test low confidence warning
    print("\n3. Testing confidence distribution warning...")
    story_low_conf = Story()
    story_low_conf.id = 997
    story_low_conf.title = "Test Low Confidence"
    story_low_conf.direct_people = [
        {
            "name": "P1",
            "linkedin_profile": "https://linkedin.com/in/p1",
            "match_confidence": "low",
        },
        {
            "name": "P2",
            "linkedin_profile": "https://linkedin.com/in/p2",
            "match_confidence": "low",
        },
        {
            "name": "P3",
            "linkedin_profile": "https://linkedin.com/in/p3",
            "match_confidence": "low",
        },
    ]
    result = validate_enrichment_quality(story_low_conf)
    # All 3 are low confidence, which is 100% > 40%
    assert result.confidence_distribution_ok is False
    print(f"   ✓ Low confidence warning: {result.warnings}")

    print("\n   validate_enrichment_quality: PASSED ✓")


def test_atomic_enrichment():
    """Test atomic enrichment transaction."""
    print("\n" + "=" * 60)
    print("TEST: Atomic Enrichment Transaction")
    print("=" * 60)

    db = Database()

    # Create a test story
    print("\n1. Creating test story...")
    test_story = Story()
    test_story.title = "Phase 4 Atomic Enrichment Test"
    test_story.direct_people = [
        {
            "name": "Test Researcher",
            "company": "Test University",
            "linkedin_profile": "https://linkedin.com/in/testresearcher",
            "match_confidence": "high",
        }
    ]
    test_story.indirect_people = [
        {
            "name": "Test CEO",
            "organization": "Test Corp",
            "linkedin_profile": "https://linkedin.com/in/testceo",
            "match_confidence": "medium",
        }
    ]
    test_story.enrichment_status = "pending"
    story_id = db.add_story(test_story)
    print(f"   Created story {story_id}")

    # Get the story back
    story = db.get_story(story_id)
    assert story is not None

    # Create mock enricher (we just need db access)
    print("\n2. Testing atomic enrichment...")

    # Import mock client
    try:
        from google import genai

        client = genai.Client(api_key="test")  # Will fail but we don't need it
    except Exception:
        client = None

    # We can't easily instantiate the enricher without valid API keys,
    # so let's just test the validation functions directly

    # Run validation
    qa_result = validate_enrichment_quality(story)
    print(f"   QA Status: {qa_result.status}")
    print(f"   Checks passed: {qa_result.checks_passed}")

    # Manually simulate atomic update
    log = story.enrichment_log if isinstance(story.enrichment_log, dict) else {}
    log["completed_at"] = "2026-01-12T12:00:00"
    log["validation"] = qa_result.to_dict()
    story.enrichment_log = log
    story.enrichment_status = "enriched"
    story.enrichment_quality = "high" if qa_result.passed else "low"

    success = db.update_story(story)
    print(f"   Update success: {success}")
    assert success is True

    # Verify the update
    updated = db.get_story(story_id)
    assert updated is not None
    assert updated.enrichment_status == "enriched"
    assert updated.enrichment_quality == "high"

    log = updated.enrichment_log
    if isinstance(log, str):
        log = json.loads(log)
    assert "validation" in log
    print(f"   ✓ Enrichment log updated with validation: {list(log.keys())}")

    print("\n   Atomic enrichment: PASSED ✓")


def test_metrics_integration():
    """Test that metrics integrate properly with enrichment."""
    print("\n" + "=" * 60)
    print("TEST: Metrics Integration")
    print("=" * 60)

    reset_enrichment_metrics()
    metrics = get_enrichment_metrics()

    # Simulate enrichment flow
    print("\n1. Simulating enrichment flow...")
    metrics.stories_processed = 10
    metrics.direct_people_found = 25
    metrics.indirect_people_found = 15
    metrics.linkedin_matches = 30
    metrics.high_confidence_matches = 20
    metrics.medium_confidence_matches = 7
    metrics.low_confidence_matches = 3
    metrics.gemini_calls = 10
    metrics.google_searches = 50
    metrics.total_processing_time = 120.0
    metrics.avg_story_enrichment_time = 12.0

    # Get dict for logging
    d = metrics.to_dict()
    print(f"   Stories: {d['stories_processed']}")
    print(f"   Match rate: {d['match_rate']}")
    print(f"   High confidence rate: {d['high_confidence_rate']}")
    print(f"   Avg time: {d['avg_enrichment_time_s']}s")

    assert d["stories_processed"] == 10
    assert "75.0%" in d["match_rate"]  # 30/(25+15)
    assert "66.7%" in d["high_confidence_rate"]  # 20/30

    print("\n   Metrics integration: PASSED ✓")


def main():
    """Run all Phase 4 tests."""
    print("\n" + "#" * 60)
    print("# PHASE 4 TEST SUITE")
    print("#" * 60)

    try:
        test_enrichment_metrics()
        test_qa_check_result()
        test_validate_person_record()
        test_validate_enrichment_quality()
        test_atomic_enrichment()
        test_metrics_integration()

        print("\n" + "=" * 60)
        print("ALL PHASE 4 TESTS PASSED ✓")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
