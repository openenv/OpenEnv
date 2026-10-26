"""
FreeBusy models following Google Calendar API v3 structure
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class FreeBusyError(BaseModel):
    """Error model for FreeBusy API responses"""

    domain: str = Field(..., description="Error domain")
    reason: str = Field(..., description="Error reason")


class TimePeriod(BaseModel):
    """Time period model for busy times"""

    start: str = Field(..., description="Start time in RFC3339 format")
    end: str = Field(..., description="End time in RFC3339 format")


class CalendarItem(BaseModel):
    """Calendar item for FreeBusy query request"""

    id: str = Field(..., description="Calendar identifier")


class FreeBusyQueryRequest(BaseModel):
    """Request model for FreeBusy query"""

    timeMin: str = Field(..., description="Lower bound for the query (RFC3339)")
    timeMax: str = Field(..., description="Upper bound for the query (RFC3339)")
    timeZone: Optional[str] = Field("UTC", description="Time zone for the query")
    groupExpansionMax: Optional[int] = Field(
        None, ge=1, description="Maximum number of calendars to expand for groups"
    )
    calendarExpansionMax: Optional[int] = Field(
        None, ge=1, description="Maximum number of events to expand for calendars"
    )
    items: List[CalendarItem] = Field(..., description="List of calendars to query")

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

    @field_validator("items")
    @classmethod
    def validate_items_not_empty(cls, v):
        """Validate that items list is not empty."""
        if not v:
            raise ValueError("At least one calendar item is required")
        return v


class FreeBusyCalendarResult(BaseModel):
    """FreeBusy result for a single calendar"""

    errors: Optional[List[FreeBusyError]] = Field(None, description="List of errors")
    busy: Optional[List[TimePeriod]] = Field(
        None, description="List of busy time periods"
    )


class FreeBusyGroupResult(BaseModel):
    """FreeBusy result for a group"""

    calendars: Optional[List[str]] = Field(
        None, description="List of calendar IDs in the group"
    )
    errors: Optional[List[FreeBusyError]] = Field(None, description="List of errors")


class FreeBusyQueryResponse(BaseModel):
    """Response model for FreeBusy query"""

    kind: str = Field(default="calendar#freeBusy", description="Resource type")
    timeMin: str = Field(..., description="Lower bound for the query (RFC3339)")
    timeMax: str = Field(..., description="Upper bound for the query (RFC3339)")
    calendars: Dict[str, FreeBusyCalendarResult] = Field(
        default_factory=dict, description="Calendar-specific results"
    )
    groups: Optional[Dict[str, FreeBusyGroupResult]] = Field(
        None, description="Group-specific results"
    )


# Additional helper models for internal use


class FreeBusyCalendarInput(BaseModel):
    """Internal model for calendar input validation"""

    calendar_id: str = Field(..., description="Calendar identifier")
    user_id: str = Field(..., description="User identifier")


class FreeBusyTimeRange(BaseModel):
    """Internal model for time range validation"""

    start: datetime = Field(..., description="Start datetime")
    end: datetime = Field(..., description="End datetime")
    timezone: str = Field(default="UTC", description="Timezone")

    def validate_range(self) -> bool:
        """Validate that end time is after start time"""
        return self.end > self.start


class FreeBusyEventOverlap(BaseModel):
    """Internal model for event overlap calculation"""

    event_id: str = Field(..., description="Event identifier")
    start: datetime = Field(..., description="Event start time")
    end: datetime = Field(..., description="Event end time")
    transparency: Optional[str] = Field("opaque", description="Event transparency")

    def is_busy(self) -> bool:
        """Check if event blocks time (is not transparent)"""
        return self.transparency != "transparent"


class FreeBusyQueryValidation(BaseModel):
    """Internal model for query validation"""

    time_min: datetime = Field(..., description="Query start time")
    time_max: datetime = Field(..., description="Query end time")
    calendar_ids: List[str] = Field(..., description="Calendar IDs to query")
    user_id: str = Field(..., description="User making the query")

    def validate_time_range(self) -> bool:
        """Validate time range is reasonable"""
        max_duration_days = 366  # Maximum 1 year + 1 day
        # Calculate total duration in seconds to handle same-day queries
        duration_timedelta = self.time_max - self.time_min
        duration_seconds = duration_timedelta.total_seconds()

        # Ensure the time range is valid (end after start)
        if duration_seconds <= 0:
            return False

        # Convert to days for comparison with maximum limit
        duration_days = duration_seconds / (24 * 60 * 60)  # seconds per day

        return duration_days <= max_duration_days

    def validate_calendar_count(self) -> bool:
        """Validate number of calendars is reasonable"""
        max_calendars = 50  # Maximum calendars per query
        return 0 < len(self.calendar_ids) <= max_calendars
