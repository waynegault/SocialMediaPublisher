"""Strict validation and shared application logic for story update requests."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

from database import Story


class StoryUpdateRequest(BaseModel):
    """Fields accepted by the story-editing API."""

    model_config = ConfigDict(strict=True, extra="forbid")

    title: StrictStr | None = None
    summary: StrictStr | None = None
    hashtags: list[StrictStr] | None = None
    promotion: StrictStr | None = None
    scheduled_time: datetime | None = None

    @field_validator("title", "summary", "hashtags", mode="before")
    @classmethod
    def reject_null_non_nullable_fields(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("Field cannot be null")
        return value

    @field_validator("scheduled_time", mode="before")
    @classmethod
    def parse_scheduled_time(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError as error:
                raise ValueError("scheduled_time must be an ISO 8601 datetime") from error
        return value


def apply_story_update(story: Story, payload: Any) -> None:
    """Validate a partial update before mutating the story."""
    update = StoryUpdateRequest.model_validate(payload)
    for field_name in update.model_fields_set:
        setattr(story, field_name, getattr(update, field_name))
