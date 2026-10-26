"""
Events API endpoints following Google Calendar API v3 structure
Handles all 11 Events operations with exact Google API compatibility
"""

import logging
import re
from typing import List, Optional

from apis.calendars.router import get_calendar_manager
from database.managers.calendar_manager import CalendarManager
from database.managers.event_manager import EventManager
from database.models.user import User
from database.session_manager import CalendarSessionManager
from database.session_utils import get_session
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from middleware.auth import get_user_context
from pydantic import ValidationError
from schemas.event import (
    Channel,
    Event,
    EventCreateRequest,
    EventInstancesResponse,
    EventListResponse,
    EventMoveRequest,
    EventQuickAddRequest,
    EventTypesEnum,
    EventUpdateRequest,
    EventWatchRequest,
    OrderByEnum,
)
from schemas.import_event import (
    EventImportError,
    EventImportQueryParams,
    EventImportRequest,
    EventImportResponse,
)
from utils.validation import validate_request_colors

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calendars/{calendarId}/events", tags=["events"])

# Initialize managers
session_manager = CalendarSessionManager()


def get_event_manager(database_id: str) -> EventManager:
    """Get event manager for the specified database"""
    return EventManager(database_id)


def check_event_acl_permissions(
    calendar_manager: CalendarManager,
    user_id: str,
    calendar_id: str,
    allowed_roles: list[str],
    operation: str = "access",
):
    """Check ACL permissions for event operations"""
    calendar = calendar_manager.get_calendar_by_id(user_id, calendar_id, allowed_roles)
    if not calendar:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user_id}' lacks required roles {allowed_roles} for {operation} on calendar '{calendar_id}'",
        )
    return calendar


@router.get("", response_model=EventListResponse)
async def list_events(
    calendarId: str,
    user_context: tuple[str, str] = Depends(get_user_context),
    eventTypes: Optional[List[EventTypesEnum]] = Query(
        None,
        description="Event types to return. Acceptable values are: 'birthday' - Special all-day events with an annual recurrence, 'default' - Regular events, 'focusTime' - Focus time events, 'fromGmail' - Events from Gmail, 'outOfOffice' - Out of office events, 'workingLocation' - Working location events. Optional. Multiple event types can be provided using repeated parameter instances",
    ),
    iCalUID: Optional[str] = Query(
        None,
        description="Specifies an event ID in the iCalendar format to be provided in the response. Optional. Use this if you want to search for an event by its iCalendar ID. Mutually exclusive with q. Optional.",
    ),
    maxAttendees: Optional[int] = Query(
        None,
        gt=0,
        description="The maximum number of attendees to include in the response. If there are more than the specified number of attendees, only the participant is returned. Optional.",
    ),
    maxResults: Optional[int] = Query(
        250,
        gt=0,
        le=2500,
        description="Maximum number of events returned on one result page. By default the value is 250 events. The page size can never be larger than 2500 events. Optional.",
    ),
    orderBy: Optional[OrderByEnum] = Query(
        OrderByEnum.START_TIME,
        description="The order of the events returned in the result. Optional. The default is an unspecified, stable order.",
    ),
    pageToken: Optional[str] = Query(
        None, description="Token specifying which result page to return. Optional."
    ),
    privateExtendedProperty: Optional[str] = Query(
        None,
        description="Extended properties constraint specified as propertyName=value. Matches only private properties. This parameter might be repeated multiple times to return events that match all given constraints.",
    ),
    q: Optional[str] = Query(
        None,
        description="Free text search terms to find events that match these terms in any field, except for extended properties. Optional.",
    ),
    sharedExtendedProperty: Optional[str] = Query(
        None,
        description="Extended properties constraint specified as propertyName=value. Matches only shared properties. This parameter might be repeated multiple times to return events that match all given constraints.",
    ),
    showDeleted: Optional[bool] = Query(
        False,
        description="Whether to include deleted events (with status equals 'cancelled') in the result. Cancelled instances of recurring events (but not the underlying recurring event) will still be included if showDeleted and singleEvents are both False. If showDeleted and singleEvents are both True, only single instances of deleted events (but not the underlying recurring events) are returned. Optional. The default is False.",
    ),
    showHiddenInvitations: Optional[bool] = Query(
        False,
        description="Whether to include hidden invitations in the result. Optional. The default is False.",
    ),
    singleEvents: Optional[bool] = Query(
        False,
        description="Whether to expand recurring events into instances and only return single one-off events and instances of recurring events, but not the underlying recurring events themselves. Optional. The default is False.",
    ),
    syncToken: Optional[str] = Query(
        None,
        description="Token obtained from the nextSyncToken field returned on the last page of results from the previous list request. It makes the result of this list request contain only entries that have changed since then. All events deleted since the previous list request will always be in the result set and it is not allowed to set showDeleted to False. There are several query parameters that cannot be specified together with nextSyncToken to ensure consistency of the client state.",
    ),
    timeMax: Optional[str] = Query(
        None,
        description="Upper bound (exclusive) for an event's start time to filter by. Optional. The default is not to filter by start time. Must be an RFC3339 timestamp with mandatory time zone offset, for example, 2011-06-03T10:00:00-07:00, 2011-06-03T10:00:00Z. Milliseconds may be provided but are ignored. If timeMin is set, timeMax must be greater than timeMin.",
    ),
    timeMin: Optional[str] = Query(
        None,
        description="Lower bound (exclusive) for an event's end time to filter by. Optional. The default is not to filter by end time. Must be an RFC3339 timestamp with mandatory time zone offset, for example, 2011-06-03T10:00:00-07:00, 2011-06-03T10:00:00Z. Milliseconds may be provided but are ignored. If timeMax is set, timeMin must be less than timeMax.",
    ),
    timeZone: Optional[str] = Query(
        None,
        description="Time zone used in the response. Optional. The default is the time zone of the calendar.",
    ),
    updatedMin: Optional[str] = Query(
        None,
        description="Lower bound for an event's last modification time (as a RFC3339 timestamp) to filter by. When specified, entries deleted since this time will always be included regardless of showDeleted. Optional. The default is not to filter by last modification time.",
    ),
):
    """
    Returns events on the specified calendar
    ACL Required: reader, writer, or owner role

    GET /calendars/{calendarId}/events
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)
        event_manager = get_event_manager(database_id)

        # Check ACL permissions - require at least reader role
        check_event_acl_permissions(
            calendar_manager,
            user_id,
            calendarId,
            ["reader", "writer", "owner"],
            "list events",
        )

        # Validate iCalUID
        if iCalUID:
            pattern = r"^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if not re.match(pattern, iCalUID):
                raise ValueError(
                    "Invalid iCalUID format. Expected something like 'abcd123@google.com'."
                )

        # Validate pageToken
        if pageToken:
            try:
                page_token_int = int(pageToken)
                if not page_token_int >= 0:
                    raise ValueError(
                        "Please enter a valid pageToken value. Page token must be greater than equal to 0 and must be string integer"
                    )
            except:
                raise ValueError(
                    "Please enter a valid pageToken value. Page token must be greater than equal to 0 and must be string integer"
                )

        # Convert eventTypes enum list to comma-separated string for the manager
        event_types_str = None
        if eventTypes:
            event_types_str = ",".join([event_type.value for event_type in eventTypes])

        response = event_manager.list_events(
            user_id=user_id,
            calendar_id=calendarId,
            event_types=event_types_str,
            ical_uid=iCalUID,
            max_attendees=maxAttendees,
            max_results=maxResults,
            order_by=orderBy,
            page_token=pageToken,
            private_extended_property=privateExtendedProperty,
            q=q,
            shared_extended_property=sharedExtendedProperty,
            show_deleted=showDeleted,
            show_hidden_invitations=showHiddenInvitations,
            single_events=singleEvents,
            sync_token=syncToken,
            time_max=timeMax,
            time_min=timeMin,
            time_zone=timeZone,
            updated_min=updatedMin,
        )

        return response

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{str(e)}")
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing events for calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("", response_model=Event, status_code=status.HTTP_201_CREATED)
async def create_event(
    calendarId: str,
    event_request: EventCreateRequest,
    user_context: tuple[str, str] = Depends(get_user_context),
    conferenceDataVersion: Optional[int] = Query(
        None,
        ge=0,
        le=1,
        description="Version number of conference data supported by API client",
    ),
    maxAttendees: Optional[int] = Query(
        None,
        gt=0,
        description="The maximum number of attendees to include in the response",
    ),
    sendUpdates: Optional[str] = Query(
        "none",
        description="Whether to send notifications about the creation of the new event. Note that some emails might still be sent. The default is 'none'. Acceptable values are: 'all' (notifications sent to all guests), 'externalOnly' (notifications sent to non-Google Calendar guests only), 'none' (no notifications sent)",
    ),
    supportsAttachments: Optional[bool] = Query(
        False,
        description="Whether API client performing operation supports event attachments",
    ),
):
    """
    Creates an event
    ACL Required: writer or owner role

    The sendUpdates parameter controls notification behavior:
    - "all": Notifications are sent to all guests
    - "externalOnly": Notifications are sent to non-Google Calendar guests only
    - "none": No notifications are sent (default)

    POST /calendars/{calendarId}/events
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)
        event_manager = get_event_manager(database_id)

        # Check ACL permissions - require writer or owner role
        check_event_acl_permissions(
            calendar_manager, user_id, calendarId, ["writer", "owner"], "create events"
        )

        # Validate sendUpdates parameter
        if sendUpdates and sendUpdates not in ["all", "externalOnly", "none"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sendUpdates value '{sendUpdates}'. Acceptable values are: 'all', 'externalOnly', 'none'",
            )

        # Validate colorId if provided (events use event colors)
        event_data = event_request.model_dump(exclude_none=True)
        color_error = validate_request_colors(event_data, "event", database_id)
        if color_error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=color_error
            )

        # Validate required fields per Google API specification
        if not event_request.start or not event_request.end:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both 'start' and 'end' fields are required",
            )

        # Create query parameters object for event creation
        query_params = {
            "conferenceDataVersion": conferenceDataVersion,
            "maxAttendees": maxAttendees,
            "sendUpdates": sendUpdates,
            "supportsAttachments": supportsAttachments,
        }

        event = event_manager.create_event(
            user_id, calendarId, event_request, query_params
        )

        if not event:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create event"
            )

        # Filter attendees based on maxAttendees parameter
        if (
            maxAttendees is not None
            and event.attendees
            and len(event.attendees) > maxAttendees
        ):
            event.attendees = event.attendees[:maxAttendees]
            event.attendeesOmitted = True

        return event

    except ValueError as verr:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(verr))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.errors())
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating event in calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{eventId}", response_model=Event)
async def get_event(
    calendarId: str,
    eventId: str,
    user_context: tuple[str, str] = Depends(get_user_context),
    timeZone: Optional[str] = Query(None, description="Time zone for returned times"),
    maxAttendees: Optional[int] = Query(
        None,
        description="Maximum number of attendees to include in the response. If there are more than the specified number of attendees, only the participant is returned. Optional.",
    ),
):
    """
    Returns an event

    GET /calendars/{calendarId}/events/{eventId}
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)
        event_manager = get_event_manager(database_id)

        # Check ACL permissions - require at least reader role
        check_event_acl_permissions(
            calendar_manager,
            user_id,
            calendarId,
            ["reader", "writer", "owner"],
            "read event",
        )

        event = event_manager.get_event(
            user_id, calendarId, eventId, timeZone, maxAttendees
        )

        if not event:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event not found: {eventId}",
            )

        return event

    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting event {eventId} from calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.patch("/{eventId}", response_model=Event)
async def patch_event(
    calendarId: str,
    eventId: str,
    event_request: EventUpdateRequest,
    user_context: tuple[str, str] = Depends(get_user_context),
    conferenceDataVersion: Optional[int] = Query(
        None,
        ge=0,
        le=1,
        description="Version number of conference data supported by API client",
    ),
    maxAttendees: Optional[int] = Query(
        None,
        gt=0,
        description="The maximum number of attendees to include in the response",
    ),
    sendUpdates: Optional[str] = Query(
        "none",
        description="Guests who should receive notifications (all, externalOnly, none)",
    ),
    supportsAttachments: Optional[bool] = Query(
        False,
        description="Whether API client performing operation supports event attachments",
    ),
):
    """
    Updates an event (partial update) following Google Calendar API v3 specification.

    This method supports patch semantics and only updates the fields that are explicitly provided.
    To do a full update, use PUT which replaces the entire event resource.

    ACL Required: writer or owner role

    PATCH /calendars/{calendarId}/events/{eventId}
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)
        event_manager = get_event_manager(database_id)

        # Check ACL permissions - require writer or owner role
        check_event_acl_permissions(
            calendar_manager, user_id, calendarId, ["writer", "owner"], "update event"
        )

        # Validate sendUpdates parameter
        if sendUpdates and sendUpdates not in ["all", "externalOnly", "none"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sendUpdates value '{sendUpdates}'. Acceptable values are: 'all', 'externalOnly', 'none'",
            )

        # Get update data, excluding None values for partial update
        update_data = event_request.model_dump(exclude_none=True)

        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields provided for update",
            )

        # Validate colorId if provided (events use event colors)
        color_error = validate_request_colors(update_data, "event", database_id)
        if color_error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=color_error
            )

        # Create query parameters object for event update
        query_params = {
            "conferenceDataVersion": conferenceDataVersion,
            "maxAttendees": maxAttendees,
            "sendUpdates": sendUpdates,
            "supportsAttachments": supportsAttachments,
        }

        event = event_manager.update_event(
            user_id,
            calendarId,
            eventId,
            event_request,
            is_patch=True,
            query_params=query_params,
        )

        if not event:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event not found: {eventId}",
            )

        # Filter attendees based on maxAttendees parameter
        if (
            maxAttendees is not None
            and event.attendees
            and len(event.attendees) > maxAttendees
        ):
            event.attendees = event.attendees[:maxAttendees]
            event.attendeesOmitted = True

        return event

    except ValueError as verr:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(verr))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.errors())
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating event {eventId} in calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put("/{eventId}", response_model=Event)
async def update_event(
    calendarId: str,
    eventId: str,
    event_request: EventUpdateRequest,
    user_context: tuple[str, str] = Depends(get_user_context),
    conferenceDataVersion: Optional[int] = Query(
        None,
        ge=0,
        le=1,
        description="Version number of conference data supported by API client",
    ),
    maxAttendees: Optional[int] = Query(
        None,
        gt=0,
        description="The maximum number of attendees to include in the response",
    ),
    sendUpdates: Optional[str] = Query(
        "none",
        description="Guests who should receive notifications (all, externalOnly, none)",
    ),
    supportsAttachments: Optional[bool] = Query(
        False,
        description="Whether API client performing operation supports event attachments",
    ),
):
    """
    Updates an event (full update) following Google Calendar API v3 specification.

    This method does not support patch semantics and always updates the entire event resource.
    To do a partial update, perform a get followed by an update using etags to ensure atomicity.

    PUT /calendars/{calendarId}/events/{eventId}
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)
        event_manager = get_event_manager(database_id)

        # Check ACL permissions - require writer or owner role
        check_event_acl_permissions(
            calendar_manager, user_id, calendarId, ["writer", "owner"], "update event"
        )

        # For PUT operations, validate that start and end are provided (required per Google API v3)
        if not event_request.start or not event_request.end:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both 'start' and 'end' fields are required for PUT operations",
            )

        # Validate sendUpdates parameter
        if sendUpdates and sendUpdates not in ["all", "externalOnly", "none"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sendUpdates value '{sendUpdates}'. Acceptable values are: 'all', 'externalOnly', 'none'",
            )

        # Ensure required fields have defaults for PUT requests
        update_data = event_request.model_dump()
        if not update_data.get("status"):
            event_request.status = "confirmed"
        if not update_data.get("visibility"):
            event_request.visibility = "default"

        # Validate colorId if provided (events use event colors)
        color_error = validate_request_colors(update_data, "event", database_id)
        if color_error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=color_error
            )

        # Create query parameters object for event update
        query_params = {
            "conferenceDataVersion": conferenceDataVersion,
            "maxAttendees": maxAttendees,
            "sendUpdates": sendUpdates,
            "supportsAttachments": supportsAttachments,
        }

        event = event_manager.update_event(
            user_id,
            calendarId,
            eventId,
            event_request,
            is_patch=False,
            query_params=query_params,
        )

        if not event:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event not found: {eventId}",
            )

        # Filter attendees based on maxAttendees parameter
        if (
            maxAttendees is not None
            and event.attendees
            and len(event.attendees) > maxAttendees
        ):
            event.attendees = event.attendees[:maxAttendees]
            event.attendeesOmitted = True

        return event

    except ValueError as verr:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(verr))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.errors())
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating event {eventId} in calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/{eventId}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_event(
    calendarId: str,
    eventId: str,
    user_context: tuple[str, str] = Depends(get_user_context),
    sendUpdates: Optional[str] = Query(
        "all",
        description="Guests who should receive notifications. Acceptable values are: 'all' (notifications sent to all guests), 'externalOnly' (notifications sent to non-Google Calendar guests only), 'none' (no notifications sent)",
    ),
):
    """
    Deletes an event
    ACL Required: writer or owner role (owner required for events created by others)

    DELETE /calendars/{calendarId}/events/{eventId}
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)
        event_manager = get_event_manager(database_id)

        # Validate sendUpdates parameter
        if sendUpdates not in ["all", "externalOnly", "none"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sendUpdates value '{sendUpdates}'. Acceptable values are: 'all', 'externalOnly', 'none'",
            )

        # Check ACL permissions - require writer or owner role
        check_event_acl_permissions(
            calendar_manager, user_id, calendarId, ["writer", "owner"], "delete event"
        )

        # Check if event exists first
        existing_event = event_manager.get_event(user_id, calendarId, eventId)
        if not existing_event:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event not found: {eventId}",
            )

        # Delete event with sendUpdates parameter
        success = event_manager.delete_event(user_id, calendarId, eventId, sendUpdates)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete event")

        # Return 204 No Content (no response body)
        return None

    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting event {eventId} from calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/{eventId}/move", response_model=Event)
async def move_event(
    calendarId: str,
    eventId: str,
    user_context: tuple[str, str] = Depends(get_user_context),
    destination: str = Query(
        ..., description="Calendar identifier of the target calendar"
    ),
    sendUpdates: Optional[str] = Query(
        "all", description="Guests who should receive notifications"
    ),
):
    """
    Moves an event to another calendar
    ACL Required:
    - writer or owner role on source calendar
    - writer or owner role on destination calendar

    POST /calendars/{calendarId}/events/{eventId}/move
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)
        event_manager = get_event_manager(database_id)

        # Validate sendUpdates parameter
        if sendUpdates not in ["all", "externalOnly", "none"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sendUpdates value '{sendUpdates}'. Acceptable values are: 'all', 'externalOnly', 'none'",
            )

        # Check ACL permissions on source calendar
        check_event_acl_permissions(
            calendar_manager,
            user_id,
            calendarId,
            ["writer", "owner"],
            "move event from",
        )

        # Check if event exists and validate event type for move operation
        existing_event = event_manager.get_event(user_id, calendarId, eventId)
        if not existing_event:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event not found: {eventId}",
            )

        # Only default events can be moved - validate event type
        if existing_event.eventType and existing_event.eventType != "default":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot move event of type '{existing_event.eventType}'. Only default events can be moved. Events of type 'birthday', 'focusTime', 'fromGmail', 'outOfOffice', and 'workingLocation' cannot be moved.",
            )

        move_request = EventMoveRequest(
            destination=destination, sendUpdates=sendUpdates
        )

        event = event_manager.move_event(user_id, calendarId, eventId, move_request)

        if not event:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event not found: {eventId}",
            )

        return event

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error moving event {eventId} from calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/quickAdd", response_model=Event, status_code=status.HTTP_201_CREATED)
async def quick_add_event(
    calendarId: str,
    user_context: tuple[str, str] = Depends(get_user_context),
    text: str = Query(..., description="The text describing the event to be created"),
    sendUpdates: Optional[str] = Query(
        "all", description="Guests who should receive notifications"
    ),
):
    """
    Creates an event based on a simple text string

    POST /calendars/{calendarId}/events/quickAdd
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)
        event_manager = get_event_manager(database_id)

        # Validate sendUpdates parameter
        if sendUpdates not in ["all", "externalOnly", "none"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sendUpdates value '{sendUpdates}'. Acceptable values are: 'all', 'externalOnly', 'none'",
            )

        # Check ACL permissions - require writer or owner role
        check_event_acl_permissions(
            calendar_manager,
            user_id,
            calendarId,
            ["writer", "owner"],
            "quick add event",
        )

        quick_add_request = EventQuickAddRequest(text=text, sendUpdates=sendUpdates)

        event = event_manager.quick_add_event(user_id, calendarId, quick_add_request)

        return event

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.errors())
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error quick adding event to calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/import", response_model=Event, status_code=status.HTTP_201_CREATED)
async def import_event(
    calendarId: str,
    event_request: EventImportRequest,
    user_context: tuple[str, str] = Depends(get_user_context),
    conferenceDataVersion: Optional[int] = Query(
        None,
        ge=0,
        le=1,
        description="Version number of conference data supported by API client",
    ),
    supportsAttachments: Optional[bool] = Query(
        False,
        description="Whether API client performing operation supports event attachments",
    ),
):
    """
    Imports an event. This operation is used to add a private copy of an
    existing event to a calendar.

    Requires authorization with at least one of the following scopes:
    - https://www.googleapis.com/auth/calendar
    - https://www.googleapis.com/auth/calendar.events
    - https://www.googleapis.com/auth/calendar.app.created
    - https://www.googleapis.com/auth/calendar.events.owned

    POST /calendars/{calendarId}/events/import
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)
        event_manager = get_event_manager(database_id)

        # Check ACL permissions - require writer or owner role
        check_event_acl_permissions(
            calendar_manager, user_id, calendarId, ["writer", "owner"], "import event"
        )

        # Validate required fields
        if not event_request.start or not event_request.end:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both 'start' and 'end' fields are required for event import",
            )

        # Validate iCalUID
        if not event_request.iCalUID:
            raise ValueError("iCalUID is required field")

        # Validate attendees email addresses exist in database
        if event_request.attendees:
            session = get_session(database_id)
            try:
                for attendee in event_request.attendees:
                    if attendee.email:
                        user = (
                            session.query(User)
                            .filter(User.email == attendee.email)
                            .first()
                        )
                        if not user:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Attendee email '{attendee.email}' not found in database. All attendee emails must be valid and present in the system.",
                            )
            finally:
                session.close()

        # Validate colorId if provided (events use event colors)
        event_data = event_request.model_dump(exclude_none=True)
        color_error = validate_request_colors(event_data, "event", database_id)
        if color_error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=color_error
            )

        # Create query params object
        query_params = EventImportQueryParams(
            conferenceDataVersion=conferenceDataVersion,
            supportsAttachments=supportsAttachments,
        )
        # Import the event as a private copy
        imported_event = event_manager.import_event(
            user_id=user_id,
            calendar_id=calendarId,
            event_request=event_request,
            query_params=query_params,
        )

        if not imported_event:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to import event"
            )

        return imported_event

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.errors())
    except ValueError as e:
        logger.error(f"Validation error importing event to calendar {calendarId}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error importing event to calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{eventId}/instances", response_model=EventInstancesResponse)
async def get_event_instances(
    calendarId: str,
    eventId: str,
    user_context: tuple[str, str] = Depends(get_user_context),
    maxAttendees: Optional[int] = Query(
        None,
        description="The maximum number of attendees to include in the response. If there are more than the specified number of attendees, only the participant is returned. Optional.",
    ),
    maxResults: Optional[int] = Query(
        250,
        lt=2500,
        description="Maximum number of events returned on one result page. By default the value is 250 events. The page size can never be larger than 2500 events. Optional.",
        gt=0,
        le=2500,
    ),
    originalStart: Optional[str] = Query(
        None,
        description="The original start time of the instance in the result. Optional.",
    ),
    pageToken: Optional[str] = Query(
        None, description="Token specifying which result page to return. Optional."
    ),
    showDeleted: Optional[bool] = Query(
        False,
        description="Whether to include deleted events (with status equals 'cancelled') in the result. Cancelled instances of recurring events will still be included if singleEvents is False. Optional.",
    ),
    timeMin: Optional[str] = Query(
        None,
        description="Lower bound (inclusive) for an event's end time to filter by. Optional. The default is not to filter by end time. Must be an RFC3339 timestamp with mandatory time zone offset.",
    ),
    timeMax: Optional[str] = Query(
        None,
        description="Upper bound (exclusive) for an event's start time to filter by. Optional. The default is not to filter by start time. Must be an RFC3339 timestamp with mandatory time zone offset.",
    ),
    timeZone: Optional[str] = Query(
        None,
        description="Time zone used in the response. Optional. The default is the time zone of the calendar.",
    ),
):
    """
    Returns instances of the specified recurring event

    GET /calendars/{calendarId}/events/{eventId}/instances
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)
        event_manager = get_event_manager(database_id)

        # Check ACL permissions - require at least reader role
        check_event_acl_permissions(
            calendar_manager,
            user_id,
            calendarId,
            ["reader", "writer", "owner"],
            "read event instances",
        )

        # Validate pageToken
        if pageToken:
            try:
                page_token_int = int(pageToken)
                if not page_token_int >= 0:
                    raise ValueError(
                        "Please enter a valid pageToken value. Page token must be greater than equal to 0 and must be string integer"
                    )
            except:
                raise ValueError(
                    "Please enter a valid pageToken value. Page token must be greater than equal to 0 and must be string integer"
                )

        response = event_manager.get_event_instances(
            user_id=user_id,
            calendar_id=calendarId,
            event_id=eventId,
            max_attendees=maxAttendees,
            max_results=maxResults,
            original_start=originalStart,
            page_token=pageToken,
            show_deleted=showDeleted,
            time_min=timeMin,
            time_max=timeMax,
            time_zone=timeZone,
        )

        return response

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting instances for event {eventId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/watch", response_model=Channel)
async def watch_events(
    calendarId: str,
    watch_request: EventWatchRequest,
    user_context: tuple[str, str] = Depends(get_user_context),
    eventTypes: Optional[str] = Query(
        None,
        description="Event types of resources to watch. Optional. This parameter can be repeated multiple times to watch resources of different types. If unset, returns all event types. Acceptable values are: 'birthday' - Special all-day events with an annual recurrence, 'default' - Regular events, 'focusTime' - Focus time events, 'fromGmail' - Events from Gmail, 'outOfOffice' - Out of office events, 'workingLocation' - Working location events.",
    ),
):
    """
    Watch for changes to Events resources

    POST /calendars/{calendarId}/events/watch
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)
        event_manager = get_event_manager(database_id)

        if eventTypes and eventTypes not in [
            "birthday",
            "default",
            "focusTime",
            "fromGmail",
            "outOfOffice",
            "workingLocation",
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid eventType value '{eventTypes}'. Acceptable values are: 'birthday', 'default', 'focusTime', 'fromGmail', 'outOfOffice', 'workingLocation'",
            )

        from database.models.user import User

        # Validate user exists in this database (ensures ownership context)
        from database.session_utils import get_session

        session = get_session(database_id)

        try:
            user_row = session.query(User).filter(User.user_id == user_id).first()
            if not user_row:
                raise HTTPException(
                    status_code=404, detail=f"User not found: {user_id}"
                )
        finally:
            session.close()

        # Check ACL permissions - require at least reader role
        check_event_acl_permissions(
            calendar_manager,
            user_id,
            calendarId,
            ["reader", "writer", "owner"],
            "watch events",
        )

        # Set up watch channel with event types filter
        channel = event_manager.watch_events(
            user_id, calendarId, watch_request, eventTypes
        )
        return channel

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{str(e)}")
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error setting up events watch for calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
