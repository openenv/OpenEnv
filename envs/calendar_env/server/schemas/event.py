"""
Event models following Google Calendar API v3 structure
"""

import json
import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Extra, Field, field_validator, model_validator


class EventStatus(str, Enum):
    """Event status enum"""

    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class EventVisibility(str, Enum):
    """Event visibility enum"""

    DEFAULT = "default"
    PUBLIC = "public"
    PRIVATE = "private"
    CONFIDENTIAL = "confidential"


class Transparency(str, Enum):
    """Event transparency enum"""

    OPAQUE = "opaque"
    TRANSPARENT = "transparent"


class OrderByEnum(str, Enum):
    """Event list ordering enum"""

    START_TIME = "startTime"
    UPDATED = "updated"


class EventTypesEnum(str, Enum):
    """Event types enum for filtering"""

    DEFAULT = "default"
    BIRTHDAY = "birthday"
    FOCUS_TIME = "focusTime"
    FROM_GMAIL = "fromGmail"
    OUT_OF_OFFICE = "outOfOffice"
    WORKING_LOCATION = "workingLocation"


class ResponseStatusEnum(str, Enum):
    needsAction = "needsAction"
    declined = "declined"
    tentative = "tentative"
    accepted = "accepted"


class ReminderMethodEnum(str, Enum):
    email = "email"
    popup = "popup"


class EntryPointType(str, Enum):
    """Conference entry point types"""

    VIDEO = "video"
    PHONE = "phone"
    SIP = "sip"
    MORE = "more"


class DateTime(BaseModel):
    """DateTime object for event start/end times"""

    dateTime: Optional[str] = Field(None, description="RFC3339 timestamp")
    date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format")
    timeZone: Optional[str] = Field(None, description="IANA timezone")

    @model_validator(mode="after")
    def validate_datetime_or_date(self) -> "DateTime":
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


class Person(BaseModel):
    """Person object for organizer and attendees"""

    id: Optional[str] = Field(None, description="Person identifier")
    email: Optional[str] = Field(None, description="Email address")
    displayName: Optional[str] = Field(None, description="Display name")
    self: Optional[bool] = Field(None, description="Whether this person is you")


class Attendee(Person):
    """Event attendee"""

    organizer: Optional[bool] = Field(
        False, description="Whether attendee is the event organizer"
    )
    responseStatus: Optional[ResponseStatusEnum] = Field(
        "needsAction",
        description="Response status: needsAction, declined, tentative, accepted",
    )
    comment: Optional[str] = Field(None, description="Attendee comment")
    additionalGuests: Optional[int] = Field(
        0, description="Number of additional guests"
    )
    optional: Optional[bool] = Field(
        False, description="Whether attendance is optional"
    )
    resource: Optional[bool] = Field(False, description="Whether this is a resource")


class EventReminder(BaseModel):
    """Event reminder settings"""

    method: ReminderMethodEnum = Field(..., description="Reminder method: email, popup")
    minutes: int = Field(..., gt=0, lt=40320, description="Minutes before event")


class ReminderOverrides(BaseModel):
    """Event reminder overrides"""

    overrides: Optional[List[EventReminder]] = Field(
        None, description="List of reminders"
    )
    useDefault: bool = Field(True, description="Use default reminders")


class ConferenceSolutionKey(BaseModel):
    """Conference solution key"""

    type: Optional[str] = Field(None, description="Conference solution type")


class ConferenceSolution(BaseModel):
    """Conference solution details"""

    iconUri: Optional[str] = Field(
        None, description="Icon URI for the conference solution"
    )
    key: Optional[ConferenceSolutionKey] = Field(
        None, description="Conference solution key"
    )
    name: Optional[str] = Field(None, description="Name of the conference solution")


class CreateRequestStatus(BaseModel):
    """Create request status"""

    statusCode: Optional[str] = Field(
        None, description="Status code of the create request"
    )


class CreateRequest(BaseModel):
    """Conference create request details"""

    conferenceSolutionKey: Optional[ConferenceSolutionKey] = Field(
        None, description="Conference solution for the create request"
    )
    requestId: Optional[str] = Field(
        None, description="Request ID for creating the conference"
    )
    status: Optional[CreateRequestStatus] = Field(
        None, description="Status of the create request"
    )


class EntryPoint(BaseModel):
    """Conference entry point details"""

    accessCode: Optional[str] = Field(
        None, max_length=128, description="Access code for the conference"
    )
    entryPointType: Optional[EntryPointType] = Field(
        None, description="Type of entry point"
    )
    label: Optional[str] = Field(
        None, max_length=512, description="Label for the URI, visible to end users"
    )
    meetingCode: Optional[str] = Field(
        None, max_length=128, description="Meeting code for the conference"
    )
    passcode: Optional[str] = Field(
        None, max_length=128, description="Passcode for the conference"
    )
    password: Optional[str] = Field(
        None, max_length=128, description="Password to access the conference"
    )
    pin: Optional[str] = Field(
        None, max_length=128, description="PIN for the conference"
    )
    uri: Optional[str] = Field(
        None, max_length=1300, description="URI for the conference entry point"
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
    """Conference/meeting data"""

    conferenceId: Optional[str] = Field(None, description="Conference ID")
    conferenceSolution: Optional[ConferenceSolution] = Field(
        None, description="Conference solution details"
    )
    createRequest: Optional[CreateRequest] = Field(
        None, description="Conference create request details"
    )
    entryPoints: Optional[List[EntryPoint]] = Field(
        None, description="Conference entry points"
    )
    notes: Optional[str] = Field(None, max_length=2048, description="Conference notes")
    signature: Optional[str] = Field(None, description="Conference signature")

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


class ConferenceDataOutput(BaseModel):
    conferenceSolution: Optional[ConferenceSolution] = Field(
        None, description="Conference solution details"
    )
    createRequest: Optional[CreateRequest] = Field(
        None, description="Conference create request details"
    )
    entryPoints: Optional[List[EntryPoint]] = Field(
        None, description="Conference entry points"
    )
    notes: Optional[str] = Field(None, description="Conference notes")
    signature: Optional[str] = Field(None, description="Conference signature")


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


class CustomLocation(BaseModel):
    """Custom location details for working location events"""

    label: Optional[str] = Field(
        None, description="An optional extra label for additional information"
    )


class OfficeLocation(BaseModel):
    """Office location details for working location events"""

    buildingId: Optional[str] = Field(
        None,
        description="An optional building identifier. This should reference a building ID in the organization's Resources database",
    )
    deskId: Optional[str] = Field(None, description="An optional desk identifier")
    floorId: Optional[str] = Field(None, description="An optional floor identifier")
    floorSectionId: Optional[str] = Field(
        None, description="An optional floor section identifier"
    )
    label: Optional[str] = Field(
        None,
        description="The office name that's displayed in Calendar Web and Mobile clients. We recommend you reference a building name in the organization's Resources database",
    )


class WorkingLocationProperties(BaseModel):
    """Working location properties for working location events"""

    type: Literal["homeOffice", "officeLocation", "customLocation"] = Field(
        ...,
        description="Type of the working location. Required when adding working location properties",
    )
    customLocation: Optional[CustomLocation] = Field(
        None,
        description="If present, specifies that the user is working from a custom location",
    )
    homeOffice: Optional[Any] = Field(
        None, description="If present, specifies that the user is working at home"
    )
    officeLocation: Optional[OfficeLocation] = Field(
        None,
        description="If present, specifies that the user is working from an office",
    )

    @field_validator("homeOffice")
    @classmethod
    def _validate_homeOffice(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v


class BirthdayProperties(BaseModel):
    """Birthday properties for birthday events"""

    type: Literal["birthday"] = Field(
        "birthday",
        description="Type of birthday event, must be 'birthday'. Cannot be changed after event creation.",
    )


class ExtendedProperties(BaseModel):
    """Extended properties for events"""

    private: Optional[Dict[str, str]] = Field(
        None, description="Private extended properties"
    )
    shared: Optional[Dict[str, str]] = Field(
        None, description="Shared extended properties"
    )


class EventSource(BaseModel):
    """Event source information"""

    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Source title")


class Event(BaseModel):
    """Event model following Google Calendar API v3 structure"""

    kind: str = Field(default="calendar#event", description="Resource type")
    etag: Optional[str] = Field(None, description="ETag of the resource")
    id: Optional[str] = Field(None, description="Event identifier")
    status: EventStatus = Field(
        default=EventStatus.CONFIRMED, description="Event status"
    )
    htmlLink: Optional[str] = Field(
        None, description="Absolute link to event in Google Calendar"
    )
    created: Optional[str] = Field(None, description="Creation time (RFC3339)")
    updated: Optional[str] = Field(None, description="Last modification time (RFC3339)")

    summary: Optional[str] = Field(None, description="Event title")
    description: Optional[str] = Field(None, description="Event description")
    location: Optional[str] = Field(None, description="Geographic location")
    colorId: Optional[str] = Field(None, description="Color ID")

    creator: Optional[Person] = Field(None, description="Event creator")
    organizer: Optional[Person] = Field(None, description="Event organizer")

    start: DateTime = Field(..., description="Event start time")
    end: DateTime = Field(..., description="Event end time")
    endTimeUnspecified: Optional[bool] = Field(
        False, description="Whether end time is actually unspecified"
    )

    recurrence: Optional[List[str]] = Field(
        None, description="List of RRULE, EXRULE, RDATE and EXDATE"
    )
    recurringEventId: Optional[str] = Field(
        None, description="Recurring event ID for instances of recurring events"
    )
    originalStartTime: Optional[DateTime] = Field(
        None, description="Original start time for recurring event instances"
    )

    transparency: Optional[Transparency] = Field(
        None, description="Whether event blocks time"
    )
    visibility: Optional[EventVisibility] = Field(
        EventVisibility.DEFAULT, description="Visibility of event"
    )
    iCalUID: Optional[str] = Field(None, description="iCalendar UID")
    sequence: Optional[int] = Field(0, description="Sequence number")

    attendees: Optional[List[Attendee]] = Field(None, description="List of attendees")
    attendeesOmitted: Optional[bool] = Field(
        None, description="Whether attendees are omitted"
    )

    extendedProperties: Optional[Dict[str, Any]] = Field(
        None, description="Extended properties"
    )
    hangoutLink: Optional[str] = Field(None, description="Hangout link")
    conferenceData: Optional[ConferenceDataOutput] = Field(
        None, description="Conference data"
    )

    guestsCanInviteOthers: Optional[bool] = Field(
        True, description="Whether guests can invite others"
    )
    guestsCanModify: Optional[bool] = Field(
        False, description="Whether guests can modify event"
    )
    guestsCanSeeOtherGuests: Optional[bool] = Field(
        True, description="Whether guests can see other guests"
    )
    privateCopy: Optional[bool] = Field(
        None, description="Whether this is a private copy"
    )
    locked: Optional[bool] = Field(False, description="Whether event is locked")

    reminders: Optional[ReminderOverrides] = Field(
        None, description="Reminder settings"
    )
    source: Optional[EventSource] = Field(
        None, description="Source from which event was created"
    )
    attachments: Optional[List[Dict[str, Any]]] = Field(
        None, description="File attachments"
    )
    eventType: Optional[str] = Field(
        "default",
        description="Event type: default, outOfOffice, focusTime, workingLocation",
    )

    # Type-specific properties
    birthdayProperties: Optional[BirthdayProperties] = Field(
        None, description="Birthday properties for birthday events"
    )
    focusTimeProperties: Optional[FocusTimeProperties] = Field(
        None, description="Focus time properties for focusTime events"
    )
    outOfOfficeProperties: Optional[OutOfOfficeProperties] = Field(
        None, description="Out of office properties for outOfOffice events"
    )
    workingLocationProperties: Optional[WorkingLocationProperties] = Field(
        None, description="Working location properties for workingLocation events"
    )


class EventCreateRequest(BaseModel):
    """Request model for creating an event - matches Google Calendar API v3 Events.insert"""

    # Required properties
    end: DateTime = Field(..., description="Event end time")
    start: DateTime = Field(..., description="Event start time")

    # Optional properties (alphabetically ordered as per Google API docs)
    attachments: Optional[List[Dict[str, Any]]] = Field(
        None, description="File attachments for the event"
    )
    attendees: Optional[List[Attendee]] = Field(None, description="List of attendees")
    birthdayProperties: Optional[BirthdayProperties] = Field(
        None, description="Birthday properties for birthday events"
    )
    colorId: Optional[str] = Field(None, description="Color ID of the event")
    conferenceData: Optional[ConferenceData] = Field(
        None, description="Conference data"
    )
    description: Optional[str] = Field(None, description="Event description")
    eventType: Optional[EventTypesEnum] = Field(
        "default",
        description="Event type: default, birthday, outOfOffice, focusTime, workingLocation",
    )
    extendedProperties: Optional[Dict[str, Any]] = Field(
        None, description="Extended properties"
    )
    focusTimeProperties: Optional[FocusTimeProperties] = Field(
        None, description="Focus time properties for focusTime events"
    )
    guestsCanInviteOthers: Optional[bool] = Field(
        True, description="Whether guests can invite others"
    )
    guestsCanModify: Optional[bool] = Field(
        False, description="Whether guests can modify event"
    )
    guestsCanSeeOtherGuests: Optional[bool] = Field(
        True, description="Whether guests can see other guests"
    )
    hangoutLink: Optional[str] = Field(None, description="Hangout link")
    iCalUID: Optional[str] = Field(None, description="iCalendar UID")
    location: Optional[str] = Field(None, description="Geographic location")
    originalStartTime: Optional[DateTime] = Field(
        None, description="Original start time for recurring event instances"
    )
    outOfOfficeProperties: Optional[OutOfOfficeProperties] = Field(
        None, description="Out of office properties for outOfOffice events"
    )
    recurrence: Optional[List[str]] = Field(
        None, description="List of RRULE, EXRULE, RDATE and EXDATE"
    )
    reminders: Optional[ReminderOverrides] = Field(
        None, description="Reminder settings"
    )
    sequence: Optional[int] = Field(0, description="Sequence number")
    source: Optional[Dict[str, Any]] = Field(
        None, description="Source from which event was created"
    )
    status: Optional[EventStatus] = Field(
        EventStatus.CONFIRMED, description="Event status"
    )
    summary: Optional[str] = Field(None, description="Event title")
    transparency: Optional[Transparency] = Field(
        Transparency.OPAQUE, description="Whether event blocks time"
    )
    visibility: Optional[EventVisibility] = Field(
        EventVisibility.DEFAULT, description="Visibility of event"
    )
    workingLocationProperties: Optional[WorkingLocationProperties] = Field(
        None, description="Working location properties for workingLocation events"
    )

    @field_validator("iCalUID")
    @classmethod
    def _validate_iCalUID(cls, v: Optional[str]) -> Optional[str]:
        pattern = r"^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError(
                "Invalid iCalUID format. Expected something like 'abcd123@google.com'."
            )
        return v

    @field_validator("source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        if v is None:
            return None
        s = str(v.get("url")).strip()
        parsed = urlparse(s)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Invalid 'url' in source: must be an https or http URL")
        return v


class EventUpdateRequest(BaseModel):
    """Request model for updating an event - matches Google Calendar API v3 Events.patch"""

    # Basic event properties
    summary: Optional[str] = Field(None, description="Event title")
    description: Optional[str] = Field(None, description="Event description")
    location: Optional[str] = Field(None, description="Geographic location")
    colorId: Optional[str] = Field(None, description="Color ID")

    # Date/time properties
    start: Optional[DateTime] = Field(None, description="Event start time")
    end: Optional[DateTime] = Field(None, description="Event end time")
    endTimeUnspecified: Optional[bool] = Field(
        None, description="Whether end time is actually unspecified"
    )

    eventType: Optional[EventTypesEnum] = Field(
        None,
        description="Event type: default, birthday, outOfOffice, focusTime, workingLocation",
    )

    # Recurrence properties
    recurrence: Optional[List[str]] = Field(
        None, description="List of RRULE, EXRULE, RDATE and EXDATE"
    )

    # Status and visibility
    status: Optional[EventStatus] = Field(None, description="Event status")
    visibility: Optional[EventVisibility] = Field(
        None, description="Visibility of event"
    )
    transparency: Optional[Transparency] = Field(
        None, description="Whether event blocks time"
    )

    # Attendee management
    attendees: Optional[List[Attendee]] = Field(None, description="List of attendees")
    guestsCanInviteOthers: Optional[bool] = Field(
        None, description="Whether guests can invite others"
    )
    guestsCanModify: Optional[bool] = Field(
        None, description="Whether guests can modify event"
    )
    guestsCanSeeOtherGuests: Optional[bool] = Field(
        None, description="Whether guests can see other guests"
    )

    # Advanced properties requiring query_params
    attachments: Optional[List[Dict[str, Any]]] = Field(
        None, description="File attachments (requires supportsAttachments=true)"
    )
    conferenceData: Optional[ConferenceData] = Field(
        None, description="Conference data (requires conferenceDataVersion)"
    )

    # Type-specific properties (note: birthdayProperties.type cannot be changed after creation)
    birthdayProperties: Optional[BirthdayProperties] = Field(
        None, description="Birthday properties (type cannot be changed)"
    )
    focusTimeProperties: Optional[FocusTimeProperties] = Field(
        None, description="Focus time properties for focusTime events"
    )
    outOfOfficeProperties: Optional[Dict[str, Any]] = Field(
        None, description="Out of office properties for outOfOffice events"
    )
    workingLocationProperties: Optional[WorkingLocationProperties] = Field(
        None, description="Working location properties for workingLocation events"
    )

    # Other modifiable properties
    extendedProperties: Optional[Dict[str, Any]] = Field(
        None, description="Extended properties"
    )
    hangoutLink: Optional[str] = Field(None, description="Hangout link")
    iCalUID: Optional[str] = Field(None, description="iCalendar UID")
    privateCopy: Optional[bool] = Field(
        None, description="Whether this is a private copy"
    )
    locked: Optional[bool] = Field(None, description="Whether event is locked")
    reminders: Optional[ReminderOverrides] = Field(
        None, description="Reminder settings"
    )
    sequence: Optional[int] = Field(None, description="Sequence number")
    source: Optional[Dict[str, Any]] = Field(
        None, description="Source from which event was created"
    )

    @field_validator("iCalUID")
    @classmethod
    def _validate_iCalUID(cls, v: Optional[str]) -> Optional[str]:
        pattern = r"^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError(
                "Invalid iCalUID format. Expected something like 'abcd123@google.com'."
            )
        return v

    @field_validator("source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        if v is None:
            return None
        s = str(v.get("url")).strip()
        parsed = urlparse(s)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Invalid 'url' in source: must be an https or http URL")
        return v


class EventListResponse(BaseModel):
    """Response model for events list"""

    kind: str = Field(default="calendar#events", description="Resource type")
    etag: Optional[str] = Field(None, description="ETag of the collection")
    summary: Optional[str] = Field(None, description="Calendar title")
    description: Optional[str] = Field(None, description="Calendar description")
    updated: Optional[str] = Field(None, description="Last modification time (RFC3339)")
    timeZone: Optional[str] = Field(None, description="Calendar timezone")
    accessRole: Optional[str] = Field(None, description="User's access role")
    defaultReminders: Optional[List[EventReminder]] = Field(
        None, description="Default reminders"
    )
    nextPageToken: Optional[str] = Field(None, description="Token for next page")
    nextSyncToken: Optional[str] = Field(None, description="Token for incremental sync")
    items: List[Event] = Field(..., description="List of events")


class EventMoveRequest(BaseModel):
    """Request model for moving an event to another calendar"""

    destination: str = Field(..., description="ID of destination calendar")
    sendUpdates: Optional[str] = Field(
        None, description="Guests who should receive notifications"
    )


class EventQuickAddRequest(BaseModel):
    """Request model for quick adding an event"""

    text: str = Field(..., min_length=1, max_length=1000, description="Quick add text")
    sendUpdates: Optional[str] = Field(
        None, description="Guests who should receive notifications"
    )


class EventImportRequest(BaseModel):
    """Request model for importing an event"""

    summary: str = Field(..., min_length=1, max_length=255, description="Event title")
    description: Optional[str] = Field(
        None, max_length=2000, description="Event description"
    )
    location: Optional[str] = Field(None, max_length=500, description="Event location")
    start_datetime: str = Field(..., description="Event start datetime in ISO format")
    end_datetime: str = Field(..., description="Event end datetime in ISO format")
    start_timezone: Optional[str] = Field(None, description="Start datetime timezone")
    end_timezone: Optional[str] = Field(None, description="End datetime timezone")
    recurrence: Optional[str] = Field(
        None, description="Recurrence rules in RRULE format"
    )
    status: str = Field(default="confirmed", description="Event status")
    visibility: str = Field(default="default", description="Event visibility")
    supportsImport: Optional[bool] = Field(
        None, description="Whether import is supported"
    )


class EventInstancesResponse(BaseModel):
    """Response model for recurring event instances"""

    kind: str = Field(default="calendar#events", description="Resource type")
    etag: Optional[str] = Field(None, description="ETag of the collection")
    summary: Optional[str] = Field(None, description="Calendar title")
    description: Optional[str] = Field(None, description="Calendar description")
    updated: Optional[str] = Field(None, description="Last modification time (RFC3339)")
    timeZone: Optional[str] = Field(None, description="Calendar timezone")
    accessRole: Optional[str] = Field(None, description="User's access role")
    defaultReminders: Optional[List[EventReminder]] = Field(
        None, description="Default reminders"
    )
    nextPageToken: Optional[str] = Field(None, description="Token for next page")
    items: List[Event] = Field(..., description="List of event instances")


class Channel(BaseModel):
    """Channel model for watch notifications"""

    kind: str = Field(default="api#channel", description="Resource type identifier")
    id: str = Field(..., description="Channel identifier")
    resourceId: Optional[str] = Field(None, description="Resource ID")
    resourceUri: Optional[str] = Field(None, description="Resource URI")
    token: Optional[str] = Field(None, description="Channel token")
    expiration: Optional[str] = Field(None, description="Expiration time")


class WatchParams(BaseModel):
    """Watch parameters"""

    ttl: Optional[str] = Field(None, description="Time to live (seconds)")

    class Config:
        extra = Extra.forbid

    @field_validator("ttl")
    @classmethod
    def _validate_ttl(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        s = str(v).strip()
        if not s.isdigit():
            raise ValueError(
                "params.ttl must be an integer string representing seconds"
            )
        if int(s) <= 0:
            raise ValueError("params.ttl must be greater than 0")
        return s


class EventWatchRequest(BaseModel):
    """Request model for watching events"""

    id: str = Field(..., description="Channel identifier")
    type: str = Field(description="Channel type")
    address: str = Field(..., description="Webhook address")
    token: Optional[str] = Field(None, description="Channel token")
    params: Optional[WatchParams] = Field(
        None,
        description="Optional parameters object; supports 'ttl' as string seconds per Google spec",
    )

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v is None:
            raise ValueError("type is required")
        s = str(v).strip().lower()
        if s not in ("web_hook", "webhook"):
            raise ValueError("Only channel type 'web_hook' or 'webhook' is supported")
        # Normalize to canonical 'web_hook'
        return "web_hook"

    @field_validator("address")
    @classmethod
    def _validate_address(cls, v: str) -> str:
        if v is None:
            raise ValueError("address is required")
        s = str(v).strip()
        parsed = urlparse(s)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Invalid 'address': must be an https URL")
        return s
