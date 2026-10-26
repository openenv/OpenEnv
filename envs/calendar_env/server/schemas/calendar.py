"""
Calendar models for Calendar API following Google Calendar API v3 structure
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# Allowed conference solution types per Google Calendar API v3 spec
ALLOWED_CONFERENCE_SOLUTION_TYPES = {
    "eventHangout",
    "eventNamedHangout",
    "hangoutsMeet",
}


class ConferenceProperties(BaseModel):
    """Conference properties for calendar"""

    allowedConferenceSolutionTypes: Optional[List[str]] = Field(
        default=None,
        description=(
            "Conference solution types. Allowed values: 'eventHangout', 'eventNamedHangout', 'hangoutsMeet'"
        ),
    )

    @field_validator("allowedConferenceSolutionTypes")
    @classmethod
    def validate_allowed_conference_solution_types(
        cls, v: Optional[List[str]]
    ) -> Optional[List[str]]:
        """Ensure provided values are restricted to the API-supported set.

        The Google Calendar API v3 permits only the following values for
        conferenceProperties.allowedConferenceSolutionTypes:
        - "eventHangout"
        - "eventNamedHangout"
        - "hangoutsMeet"
        """
        if v is None:
            return v
        # Allow empty list, but every provided value must be valid
        invalid = [
            item
            for item in v
            if item not in ALLOWED_CONFERENCE_SOLUTION_TYPES and item not in [None, ""]
        ]
        if invalid:
            allowed_sorted = sorted(ALLOWED_CONFERENCE_SOLUTION_TYPES)
            raise ValueError(
                "Invalid values for conferenceProperties.allowedConferenceSolutionTypes: "
                f"{invalid}. Allowed values are: {allowed_sorted}"
            )
        return v


class Calendar(BaseModel):
    """Calendar model following Google Calendar API v3 structure"""

    kind: str = Field(default="calendar#calendar", description="Calendar resource type")
    etag: Optional[str] = Field(None, description="ETag of the resource")
    id: Optional[str] = Field(None, description="Unique calendar identifier")
    summary: str = Field(
        ..., min_length=1, max_length=255, description="Calendar title"
    )
    description: Optional[str] = Field(
        None, max_length=1000, description="Calendar description"
    )
    location: Optional[str] = Field(
        None, max_length=500, description="Calendar location"
    )
    timeZone: str = Field(default="UTC", description="Calendar timezone in IANA format")
    conferenceProperties: Optional[ConferenceProperties] = Field(
        None, description="Conference properties"
    )


class CalendarCreateRequest(BaseModel):
    """Request model for creating a calendar (POST /calendars)"""

    summary: str = Field(
        ..., min_length=1, max_length=255, description="Calendar title (required)"
    )
    description: Optional[str] = Field(
        None, max_length=1000, description="Calendar description"
    )
    location: Optional[str] = Field(
        None, max_length=500, description="Calendar location"
    )
    timeZone: Optional[str] = Field(
        None, description="Calendar timezone in IANA format"
    )
    conferenceProperties: Optional[ConferenceProperties] = Field(
        None, description="Conference properties"
    )

    @field_validator("summary")
    @classmethod
    def validate_summary_not_blank(cls, v: str) -> str:
        if v is None:
            return v
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("summary must not be blank")
        return v

    @field_validator("timeZone")
    @classmethod
    def validate_timezone(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            from dateutil.tz import gettz

            if gettz(v) is None:
                raise ValueError("Invalid timeZone; must be a valid IANA timezone name")
        except Exception:
            # If dateutil is unavailable or another error occurs
            raise ValueError("Invalid timeZone; validation failed")
        return v


class CalendarUpdateRequest(BaseModel):
    """Request model for updating a calendar (PATCH/PUT /calendars/{calendarId})"""

    summary: str = Field(
        None, min_length=1, max_length=255, description="Calendar title"
    )
    description: Optional[str] = Field(
        None, max_length=1000, description="Calendar description"
    )
    location: Optional[str] = Field(
        None, max_length=500, description="Calendar location"
    )
    timeZone: Optional[str] = Field(
        None, description="Calendar timezone in IANA format"
    )
    conferenceProperties: Optional[ConferenceProperties] = Field(
        None, description="Conference properties"
    )

    @field_validator("summary")
    @classmethod
    def validate_update_summary_not_blank(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("summary must not be blank when provided")
        return v

    @field_validator("timeZone")
    @classmethod
    def validate_update_timezone(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            from dateutil.tz import gettz

            if gettz(v) is None:
                raise ValueError("Invalid timeZone; must be a valid IANA timezone name")
        except Exception:
            raise ValueError("Invalid timeZone; validation failed")
        return v


class CalendarListResponse(BaseModel):
    """Response model for listing calendars"""

    kind: str = Field(
        default="calendar#calendarList", description="Calendar list resource type"
    )
    etag: Optional[str] = Field(None, description="ETag of the collection")
    items: List[Calendar] = Field(default_factory=list, description="List of calendars")
