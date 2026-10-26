"""
Event Import schemas following Google Calendar API v3 structure
"""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator, validator


class EventDateTime(BaseModel):
    """DateTime model for event start/end times"""

    dateTime: Optional[str] = Field(None, description="RFC3339 timestamp with timezone")
    date: Optional[str] = Field(
        None, description="Date in YYYY-MM-DD format for all-day events"
    )
    timeZone: Optional[str] = Field(None, description="IANA timezone identifier")

    @model_validator(mode="after")
    def validate_datetime_or_date(self) -> "EventDateTime":
        """Ensure either dateTime or date is provided, but not both"""
        has_datetime = self.dateTime is not None
        has_date = self.date is not None

        if has_datetime and has_date:
            raise ValueError(
                "Cannot specify both dateTime and date - use either dateTime for timed events or date for all-day events"
            )

        if not has_datetime and not has_date:
            raise ValueError(
                "Must specify either dateTime or date - dateTime for timed events, date for all-day events"
            )

        return self

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


# Conference Data Enums
class ConferenceSolutionType(str, Enum):
    """Conference solution types according to Google Calendar API"""

    HANGOUTS_MEET = "hangoutsMeet"
    ADD_ON = "addOn"


class EntryPointType(str, Enum):
    """Conference entry point types"""

    VIDEO = "video"
    PHONE = "phone"
    SIP = "sip"
    MORE = "more"


class CreateRequestStatusCode(str, Enum):
    """Conference create request status codes"""

    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"


class ResponseStatusEnum(str, Enum):
    needsAction = "needsAction"
    declined = "declined"
    tentative = "tentative"
    accepted = "accepted"


# Conference Data Models
class ConferenceSolutionKey(BaseModel):
    """Conference solution key with type validation"""

    type: Optional[ConferenceSolutionType] = Field(
        None, description="Conference solution type"
    )


class ConferenceSolution(BaseModel):
    """Conference solution details"""

    iconUri: Optional[str] = Field(
        None, description="User-visible icon for this solution"
    )
    key: Optional[ConferenceSolutionKey] = Field(
        None, description="Key which uniquely identifies the conference solution"
    )
    name: Optional[str] = Field(
        None, description="User-visible name of this solution (not localized)"
    )


class CreateRequestStatus(BaseModel):
    """Status of conference create request"""

    statusCode: CreateRequestStatusCode = Field(
        ..., description="Current status of the conference create request"
    )


class CreateRequest(BaseModel):
    """Request to generate a new conference and attach it to the event"""

    conferenceSolutionKey: Optional[ConferenceSolutionKey] = Field(
        None, description="Conference solution such as Hangouts or Google Meet"
    )
    requestId: Optional[str] = Field(
        None, description="Client-generated unique ID for this request"
    )
    status: Optional[CreateRequestStatus] = Field(
        None, description="Status of the conference create request"
    )


class EntryPoint(BaseModel):
    """Individual conference entry point (URLs or phone numbers)"""

    entryPointType: EntryPointType = Field(
        ..., description="Type of the conference entry point"
    )
    uri: str = Field(..., max_length=1300, description="URI of the entry point")

    # Optional access credentials (only populate subset based on provider terminology)
    accessCode: Optional[str] = Field(
        None, max_length=128, description="Access code to access the conference"
    )
    meetingCode: Optional[str] = Field(
        None, max_length=128, description="Meeting code to access the conference"
    )
    passcode: Optional[str] = Field(
        None, max_length=128, description="Passcode to access the conference"
    )
    password: Optional[str] = Field(
        None, max_length=128, description="Password to access the conference"
    )
    pin: Optional[str] = Field(
        None, max_length=128, description="PIN to access the conference"
    )

    # Optional display properties
    label: Optional[str] = Field(
        None, max_length=512, description="Label for the URI, visible to end users"
    )

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, v: str, info) -> str:
        """Validate URI format based on entry point type"""
        if not v:
            raise ValueError("URI is required")

        # Get the entry point type from the model context
        entry_point_type = info.data.get("entryPointType")

        parsed = urlparse(v)

        if entry_point_type == EntryPointType.VIDEO:
            if parsed.scheme not in ["http", "https"]:
                raise ValueError("Video entry point URI must use http or https schema")
        elif entry_point_type == EntryPointType.PHONE:
            if parsed.scheme != "tel":
                raise ValueError("Phone entry point URI must use tel schema")
        elif entry_point_type == EntryPointType.SIP:
            if parsed.scheme != "sip":
                raise ValueError("SIP entry point URI must use sip schema")
        elif entry_point_type == EntryPointType.MORE:
            if parsed.scheme not in ["http", "https"]:
                raise ValueError("More entry point URI must use http or https schema")

        return v


class ConferenceData(BaseModel):
    """Conference-related information such as Google Meet details"""

    # Basic properties
    conferenceId: Optional[str] = Field(None, description="ID of the conference")
    signature: Optional[str] = Field(
        None, description="Signature of the conference data"
    )
    notes: Optional[str] = Field(
        None, max_length=2048, description="Additional notes to display to user"
    )

    # Either conferenceSolution + entryPoints OR createRequest is required
    conferenceSolution: Optional[ConferenceSolution] = Field(
        None, description="Conference solution such as Google Meet"
    )
    entryPoints: Optional[List[EntryPoint]] = Field(
        None, description="Conference entry points such as URLs or phone numbers"
    )
    createRequest: Optional[CreateRequest] = Field(
        None, description="Request to generate a new conference"
    )

    @model_validator(mode="after")
    def validate_conference_data(self) -> "ConferenceData":
        """Validate that either (conferenceSolution + entryPoints) or createRequest is provided"""
        has_solution = self.conferenceSolution is not None
        has_entry_points = self.entryPoints is not None and len(self.entryPoints) > 0
        has_create_request = self.createRequest is not None

        if has_create_request:
            # If createRequest is provided, conferenceSolution and entryPoints should not be set
            if has_solution or has_entry_points:
                raise ValueError(
                    "Cannot specify both createRequest and conferenceSolution/entryPoints"
                )
        else:
            # If no createRequest, must have conferenceSolution and at least one entryPoint
            if not (has_solution and has_entry_points):
                raise ValueError(
                    "Must specify either createRequest OR both conferenceSolution and at least one entryPoint"
                )

        return self

    @field_validator("entryPoints")
    @classmethod
    def validate_entry_points(
        cls, v: Optional[List[EntryPoint]]
    ) -> Optional[List[EntryPoint]]:
        """Validate entry point constraints"""
        if not v:
            return v

        # Count entry points by type
        type_counts = {}
        for entry_point in v:
            entry_type = entry_point.entryPointType
            type_counts[entry_type] = type_counts.get(entry_type, 0) + 1

        # Validate type constraints
        if type_counts.get(EntryPointType.VIDEO, 0) > 1:
            raise ValueError("Conference can have at most one video entry point")
        if type_counts.get(EntryPointType.SIP, 0) > 1:
            raise ValueError("Conference can have at most one SIP entry point")
        if type_counts.get(EntryPointType.MORE, 0) > 1:
            raise ValueError("Conference can have at most one more entry point")

        # A conference with only a 'more' entry point is not valid
        if len(v) == 1 and v[0].entryPointType == EntryPointType.MORE:
            raise ValueError("Conference with only a 'more' entry point is not valid")

        return v


class AttendeeRequest(BaseModel):
    """Attendee model for import requests"""

    email: str = Field(..., description="Attendee email address")
    displayName: Optional[str] = Field(None, description="Attendee display name")
    optional: Optional[bool] = Field(False, description="Whether attendee is optional")
    resource: Optional[bool] = Field(
        False, description="Whether attendee is a resource"
    )
    responseStatus: ResponseStatusEnum = Field(
        "needsAction", description="Response status"
    )
    comment: Optional[str] = Field(None, description="Attendee comment")
    additionalGuests: Optional[int] = Field(
        0, description="Number of additional guests"
    )


class EventAttachment(BaseModel):
    """Event attachment model"""

    fileUrl: str = Field(..., description="URL of the attached file")
    title: Optional[str] = Field(None, description="Attachment title")
    mimeType: Optional[str] = Field(None, description="MIME type of the attachment")
    iconLink: Optional[str] = Field(None, description="URL to attachment icon")
    fileId: Optional[str] = Field(None, description="ID of the attached file")

    @field_validator("fileUrl")
    @classmethod
    def _validate_fileUrl(cls, v: str) -> str:
        if v is None:
            return None
        s = str(v).strip()
        parsed = urlparse(s)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Invalid 'url' in source: must be an https or http URL")
        return v


class EventReminder(BaseModel):
    """Event reminder model"""

    method: str = Field(..., description="Reminder method: email or popup")
    minutes: int = Field(
        ..., gt=0, lt=40320, description="Minutes before event to remind"
    )

    @validator("method")
    def validate_method(cls, v):
        if v not in ["email", "popup"]:
            raise ValueError("Method must be either 'email' or 'popup'")
        return v

    @validator("minutes")
    def validate_minutes(cls, v):
        if v < 0:
            raise ValueError("Please enter the non-negative value for minutes")
        return v


class EventReminders(BaseModel):
    """Event reminders configuration"""

    useDefault: bool = Field(True, description="Whether to use default reminders")
    overrides: Optional[List[EventReminder]] = Field(
        None, description="Custom reminder overrides"
    )


class EventSource(BaseModel):
    """Event source information"""

    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Source title")


class ExtendedProperties(BaseModel):
    """Extended properties for events"""

    private: Optional[Dict[str, str]] = Field(
        None, description="Private extended properties"
    )
    shared: Optional[Dict[str, str]] = Field(
        None, description="Shared extended properties"
    )


class WorkingLocationProperties(BaseModel):
    """Working location properties for events"""

    type: str = Field(..., description="Working location type")
    homeOffice: Optional[bool] = Field(None, description="Whether working from home")
    customLocation: Optional[Dict[str, Any]] = Field(
        None, description="Custom location details"
    )
    officeLocation: Optional[Dict[str, Any]] = Field(
        None, description="Office location details"
    )

    @validator("type")
    def validate_type(cls, v):
        valid_types = ["homeOffice", "officeLocation", "customLocation"]
        if v not in valid_types:
            raise ValueError(f"Type must be one of: {valid_types}")
        return v


class FocusTimeProperties(BaseModel):
    """Focus time properties for focus time events"""

    autoDeclineMode: Optional[
        Literal[
            "declineNone",
            "declineAllConflictingInvitations",
            "declineOnlyNewConflictingInvitations",
        ]
    ] = Field(
        "",
        description="Whether to decline meeting invitations which overlap Focus Time events. Valid values are declineNone, declineAllConflictingInvitations, and declineOnlyNewConflictingInvitations",
    )
    chatStatus: Optional[Literal["available", "doNotDisturb"]] = Field(
        "",
        description="The status to mark the user in Chat and related products. This can be available or doNotDisturb",
    )
    declineMessage: Optional[str] = Field(
        "",
        description="Response message to set if an existing event or new invitation is automatically declined by Calendar",
    )


class EventOrganizerImport(BaseModel):
    """Event organizer model for import requests"""

    email: str = Field(
        ...,
        description="The organizer's email address. Must be a valid email address as per RFC5322.",
    )
    displayName: Optional[str] = Field(
        None, description="The organizer's name, if available."
    )

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email format according to RFC5322"""
        if not v:
            raise ValueError("Organizer email is required")

        # Basic RFC5322 email validation pattern
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError(
                "Invalid email format. Must be a valid email address as per RFC5322."
            )

        return v


class OutOfOfficeProperties(BaseModel):
    """Out of office properties for outOfOfficeProperties events"""

    autoDeclineMode: Optional[
        Literal[
            "declineNone",
            "declineAllConflictingInvitations",
            "declineOnlyNewConflictingInvitations",
        ]
    ] = Field(
        "",
        description="Whether to decline meeting invitations which overlap Focus Time events. Valid values are declineNone, declineAllConflictingInvitations, and declineOnlyNewConflictingInvitations",
    )
    declineMessage: Optional[str] = Field(
        "",
        description="Response message to set if an existing event or new invitation is automatically declined by Calendar",
    )


class EventImportRequest(BaseModel):
    """Request model for importing an event"""

    # Required fields
    start: EventDateTime = Field(..., description="Event start time")
    end: EventDateTime = Field(..., description="Event end time")
    iCalUID: str = Field(None, description="iCalendar UID")

    # Optional fields
    summary: Optional[str] = Field(None, description="Event title/summary")
    description: Optional[str] = Field(None, description="Event description")
    location: Optional[str] = Field(None, description="Event location")
    colorId: Optional[str] = Field(None, description="Event color ID")
    status: Optional[str] = Field("confirmed", description="Event status")
    transparency: Optional[str] = Field("opaque", description="Event transparency")
    visibility: Optional[str] = Field("default", description="Event visibility")

    # Organizer (writable only when importing)
    organizer: Optional[EventOrganizerImport] = Field(
        None,
        description="The organizer of the event. If the organizer is also an attendee, this is indicated with a separate entry in attendees with the organizer field set to True. To change the organizer, use the move operation. Read-only, except when importing an event.",
    )

    # Attendees and guests
    attendees: Optional[List[AttendeeRequest]] = Field(
        None, description="Event attendees"
    )
    attendeesOmitted: Optional[bool] = Field(
        False,
        description="Whether attendees may have been omitted from the event's representation",
    )
    guestsCanInviteOthers: Optional[bool] = Field(
        True, description="Can guests invite others"
    )
    guestsCanModify: Optional[bool] = Field(
        False, description="Can guests modify event"
    )
    guestsCanSeeOtherGuests: Optional[bool] = Field(
        True, description="Can guests see other guests"
    )

    # Gadget
    gadget: Optional[Dict[str, Any]] = Field(None, description="Gadget extension")

    # Recurring events
    recurrence: Optional[List[str]] = Field(
        None, description="List of RRULE, EXRULE, RDATE and EXDATE"
    )
    recurringEventId: Optional[str] = Field(None, description="Recurring event ID")
    originalStartTime: Optional[EventDateTime] = Field(
        None, description="Original start time for recurring events"
    )

    # Additional properties
    reminders: Optional[EventReminders] = Field(None, description="Event reminders")
    attachments: Optional[List[EventAttachment]] = Field(
        None, description="Event attachments"
    )
    conferenceData: Optional[ConferenceData] = Field(
        None, description="Conference data"
    )
    source: EventSource = Field(None, description="Event source information")
    extendedProperties: Optional[ExtendedProperties] = Field(
        None, description="Extended properties"
    )

    # Focus time properties (for focusTime event type)
    focusTimeProperties: Optional[FocusTimeProperties] = Field(
        None, description="Focus time properties"
    )

    # Out of office properties (for outOfOffice event type)
    outOfOfficeProperties: Optional[OutOfOfficeProperties] = Field(
        None, description="Out of office properties"
    )

    sequence: int = Field(None, description="iCalendar sequence number")

    @field_validator("iCalUID")
    @classmethod
    def _validate_iCalUID(cls, v: Optional[str]) -> Optional[str]:
        pattern = r"^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError(
                "Invalid iCalUID format. Expected something like 'abcd123@google.com'."
            )
        return v

    @validator("status")
    def validate_status(cls, v):
        valid_statuses = ["confirmed", "tentative", "cancelled"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return v

    @validator("transparency")
    def validate_transparency(cls, v):
        if v not in ["opaque", "transparent"]:
            raise ValueError("Transparency must be either 'opaque' or 'transparent'")
        return v

    @validator("visibility")
    def validate_visibility(cls, v):
        valid_visibilities = ["default", "public", "private", "confidential"]
        if v not in valid_visibilities:
            raise ValueError(f"Visibility must be one of: {valid_visibilities}")
        return v

    @field_validator("source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        if v is None:
            return None
        s = str(v.url).strip()
        parsed = urlparse(s)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Invalid 'url' in source: must be an https or http URL")
        return v


class EventImportQueryParams(BaseModel):
    """Query parameters for event import API"""

    conferenceDataVersion: Optional[int] = Field(
        None,
        ge=0,
        le=1,
        description="Version number of conference data supported by API client",
    )
    supportsAttachments: Optional[bool] = Field(
        None, description="Whether API client supports event attachments"
    )


class EventImportResponse(BaseModel):
    """Response model for successful event import"""

    kind: str = Field(default="calendar#event", description="Resource type")
    id: str = Field(..., description="Event identifier")
    etag: str = Field(..., description="ETag for the event")
    status: str = Field(..., description="Event status")
    htmlLink: str = Field(..., description="Absolute link to event in Google Calendar")
    created: str = Field(..., description="Creation time (RFC3339)")
    updated: str = Field(..., description="Last modification time (RFC3339)")
    summary: str = Field(..., description="Event title")
    creator: Dict[str, Any] = Field(..., description="Event creator")
    organizer: Dict[str, Any] = Field(..., description="Event organizer")
    start: EventDateTime = Field(..., description="Event start time")
    end: EventDateTime = Field(..., description="Event end time")

    # Optional response fields
    description: Optional[str] = Field(None, description="Event description")
    location: Optional[str] = Field(None, description="Event location")
    colorId: Optional[str] = Field(None, description="Event color ID")
    transparency: Optional[str] = Field(None, description="Event transparency")
    visibility: Optional[str] = Field(None, description="Event visibility")
    eventType: Optional[str] = Field(None, description="Event type")
    attendees: Optional[List[Dict[str, Any]]] = Field(
        None, description="Event attendees"
    )
    recurrence: Optional[List[str]] = Field(None, description="Recurrence rules")
    reminders: Optional[Dict[str, Any]] = Field(None, description="Event reminders")
    attachments: Optional[List[Dict[str, Any]]] = Field(
        None, description="Event attachments"
    )
    conferenceData: Optional[Dict[str, Any]] = Field(
        None, description="Conference data"
    )
    source: Optional[Dict[str, Any]] = Field(None, description="Event source")
    extendedProperties: Optional[Dict[str, Any]] = Field(
        None, description="Extended properties"
    )
    workingLocationProperties: Optional[Dict[str, Any]] = Field(
        None, description="Working location"
    )
    focusTimeProperties: Optional[Dict[str, Any]] = Field(
        None, description="Focus time properties"
    )
    outOfOfficeProperties: Optional[Dict[str, Any]] = Field(
        None, description="Out of office properties"
    )

    # Guest permissions
    guestsCanInviteOthers: Optional[bool] = Field(
        None, description="Whether guests can invite others"
    )
    guestsCanModify: Optional[bool] = Field(
        None, description="Whether guests can modify event"
    )
    guestsCanSeeOtherGuests: Optional[bool] = Field(
        None, description="Whether guests can see other guests"
    )

    # iCalendar properties
    iCalUID: Optional[str] = Field(None, description="iCalendar UID")
    sequence: Optional[int] = Field(None, description="iCalendar sequence number")


class EventImportError(BaseModel):
    """Error model for event import failures"""

    domain: str = Field(..., description="Error domain")
    reason: str = Field(..., description="Error reason")
    message: str = Field(..., description="Error message")
    locationType: Optional[str] = Field(None, description="Location type causing error")
    location: Optional[str] = Field(None, description="Location causing error")


class EventImportValidation(BaseModel):
    """Internal model for event import validation"""

    calendar_id: str = Field(..., description="Target calendar ID")
    user_id: str = Field(..., description="User importing the event")
    event_data: EventImportRequest = Field(..., description="Event data to import")
    preserve_event_id: bool = Field(
        False, description="Whether to preserve original event ID"
    )
    generate_new_id: bool = Field(True, description="Whether to generate new event ID")

    def validate_calendar_access(self) -> bool:
        """Validate user has access to target calendar"""
        # This would be implemented with actual calendar access checks
        return True

    def validate_event_conflicts(self) -> bool:
        """Validate no conflicts with existing events"""
        # This would check for time conflicts, duplicate IDs, etc.
        return True

    def validate_import_permissions(self) -> bool:
        """Validate user has import permissions"""
        return True


class EventImportResult(BaseModel):
    """Internal model for event import operation result"""

    success: bool = Field(..., description="Whether import was successful")
    event_id: Optional[str] = Field(None, description="ID of imported event")
    original_event_type: Optional[str] = Field(
        None, description="Original event type before conversion"
    )
    converted_to_default: bool = Field(
        False, description="Whether event was converted to default type"
    )
    properties_dropped: Optional[List[str]] = Field(
        None, description="Properties dropped during conversion"
    )
    warnings: Optional[List[str]] = Field(None, description="Import warnings")
    error: Optional[EventImportError] = Field(
        None, description="Import error if failed"
    )
