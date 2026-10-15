"""Regression tests for strict, shared story-update API validation."""

import tempfile
from datetime import datetime
from pathlib import Path

from flask import Flask
from pydantic import ValidationError

from database import Database, Story
from story_update import StoryUpdateRequest, apply_story_update
from test_framework import TestSuite, create_standard_test_runner
from validation_server import ValidationServer
from web_server import create_stories_api_blueprint


def _clients_for_database(database: Database):
    app = Flask(__name__)
    app.register_blueprint(create_stories_api_blueprint(database))
    validation_app = ValidationServer(database, port=0).app
    return [app.test_client(), validation_app.test_client()]


def _create_database(directory: str) -> tuple[Database, int]:
    database = Database(str(Path(directory) / "stories.db"))
    story_id = database.add_story(Story(title="Original", summary="Initial"))
    return database, story_id


def _test_request_model_is_strict() -> None:
    valid = StoryUpdateRequest.model_validate(
        {
            "title": "Updated",
            "hashtags": ["#python"],
            "scheduled_time": "2026-05-01T12:30:00",
        }
    )
    assert valid.title == "Updated"
    assert valid.scheduled_time == datetime(2026, 5, 1, 12, 30)

    invalid_payloads = [
        {"title": 123},
        {"hashtags": "#not-a-list"},
        {"hashtags": ["valid", 123]},
        {"title": None},
        {"unknown": "not accepted"},
        {"scheduled_time": "not-a-datetime"},
    ]
    for payload in invalid_payloads:
        try:
            StoryUpdateRequest.model_validate(payload)
        except ValidationError:
            continue
        raise AssertionError(f"Invalid update payload was accepted: {payload!r}")


def _test_update_is_validated_before_mutation() -> None:
    story = Story(title="Original", summary="Initial")
    try:
        apply_story_update(story, {"title": "Updated", "hashtags": [3]})
    except ValidationError:
        pass
    else:
        raise AssertionError("Invalid update did not fail validation")
    assert story.title == "Original"
    assert story.hashtags == []


def _test_both_routes_validate_and_persist_updates() -> None:
    with tempfile.TemporaryDirectory() as directory:
        database, story_id = _create_database(directory)
        for index, client in enumerate(_clients_for_database(database)):
            original_title = database.get_story(story_id).title
            invalid = client.put(
                f"/api/stories/{story_id}",
                json={"title": 42},
            )
            assert invalid.status_code == 400
            assert invalid.get_json()["error"] == "Invalid story update"
            assert database.get_story(story_id).title == original_title

            response = client.put(
                f"/api/stories/{story_id}",
                json={
                    "title": f"Updated {index}",
                    "summary": "Validated summary",
                    "hashtags": ["#validated"],
                    "promotion": None,
                    "scheduled_time": "2026-05-01T12:30:00",
                },
            )
            assert response.status_code == 200
            assert response.get_json()["title"] == f"Updated {index}"
            updated = database.get_story(story_id)
            assert updated.title == f"Updated {index}"
            assert updated.hashtags == ["#validated"]
            assert updated.promotion is None
            assert updated.scheduled_time == datetime(2026, 5, 1, 12, 30)


def _test_both_routes_reject_malformed_json() -> None:
    with tempfile.TemporaryDirectory() as directory:
        database, story_id = _create_database(directory)
        for client in _clients_for_database(database):
            response = client.put(
                f"/api/stories/{story_id}",
                data="{",
                content_type="application/json",
            )
            assert response.status_code == 400
            assert response.get_json()["error"]
            assert database.get_story(story_id).title == "Original"


def _create_module_tests() -> bool:
    suite = TestSuite("Story Update API Tests", "test_story_update_api.py")
    suite.start_suite()
    tests = [
        (
            "Strict request model",
            _test_request_model_is_strict,
            "Wrong field types, explicit nulls, unknown keys, and invalid timestamps are rejected",
        ),
        (
            "Validate before mutation",
            _test_update_is_validated_before_mutation,
            "Invalid update payloads leave the in-memory story unchanged",
        ),
        (
            "Both update routes",
            _test_both_routes_validate_and_persist_updates,
            "Both Flask APIs reject invalid data and persist valid partial updates",
        ),
        (
            "Malformed JSON",
            _test_both_routes_reject_malformed_json,
            "Both Flask APIs return a client error without changing stored data",
        ),
    ]
    for name, test, expected_outcome in tests:
        suite.run_test(
            test_name=name,
            test_func=test,
            test_summary=f"Verify {name.lower()}",
            expected_outcome=expected_outcome,
        )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(_create_module_tests)
