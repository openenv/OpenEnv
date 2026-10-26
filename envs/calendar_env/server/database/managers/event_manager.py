"""
Event database manager with Google Calendar API v3 compatible operations
Handles all 11 Events API operations with database-per-user architecture
"""

import base64
import json
import logging
import random
import re
import string
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from database.managers.calendar_manager import CalendarManager
from database.models.calendar import Calendar
from database.models.event import (
    Attendees,
    ConferenceData as ConferenceDataModel,
    Event,
    EventTypeEnum,
    ExtendedPropertyScope,
    RecurringEvent,
    Reminder,
)
from database.models.user import User
from database.models.watch_channel import WatchChannel
from database.session_utils import get_session, init_database
from dateutil import parser
from schemas.event import (
    Attendee,
    Channel,
    DateTime,
    Event as EventSchema,
    EventCreateRequest,
    EventInstancesResponse,
    EventListResponse,
    EventMoveRequest,
    EventQuickAddRequest,
    EventStatus,
    EventUpdateRequest,
    EventVisibility,
    Person,
)
from schemas.import_event import (
    ConferenceData,
    EventDateTime,
    EventImportError,
    EventImportQueryParams,
    EventImportRequest,
    EventImportResponse,
    EventImportResult,
    EventImportValidation,
)
from sqlalchemy import and_, asc, desc, or_
from sqlalchemy.orm import joinedload, sessionmaker
from utils.recurrence_utils import RecurrenceParseError, RecurrenceParser


logger = logging.getLogger(__name__)


class EventManager:
    """Event manager for database operations"""

    def __init__(self, database_id: str):
        self.database_id = database_id
        # Initialize database on first use
        init_database(database_id)
        self.calendar_manager = CalendarManager(database_id)
        # self.recurring_event_manager = RecurringEventManager(database_id)

    def _get_user_calendar_role(self, user_id: str, calendar: Calendar) -> str:
        """Get the highest role a user has on a calendar"""
        from database.models.acl import ACLs, Scope

        session = get_session(self.database_id)
        try:
            user = session.query(User).filter(User.user_id == user_id).first()
            if not user:
                return "none"

            acls = (
                session.query(ACLs)
                .join(Scope, ACLs.scope_id == Scope.id)
                .filter(
                    ACLs.calendar_id == calendar.calendar_id,
                    Scope.type == "user",
                    Scope.value == user.email,
                )
                .all()
            )

            if not acls:
                return "none"

            # Return the highest permission level found
            role_hierarchy = {
                "none": 0,
                "freeBusyReader": 1,
                "reader": 2,
                "writer": 3,
                "owner": 4,
            }
            highest_role = "none"
            highest_weight = 0

            for acl in acls:
                role_weight = role_hierarchy.get(acl.role.value, 0)
                if role_weight > highest_weight:
                    highest_weight = role_weight
                    highest_role = acl.role.value

            return highest_role
        finally:
            session.close()

    def check_event_permissions(
        self, user_id: str, calendar_id: str, event_id: str, required_roles: list[str]
    ) -> bool:
        """
        Check if user has required permissions for an event operation

        Args:
            user_id: User performing the operation
            calendar_id: Calendar containing the event
            event_id: Event ID
            required_roles: List of roles that allow the operation

        Returns:
            True if user has permission, False otherwise
        """
        session = get_session(self.database_id)
        try:
            # Get the event and calendar
            db_event = (
                session.query(Event)
                .filter(
                    and_(Event.calendar_id == calendar_id, Event.event_id == event_id)
                )
                .first()
            )

            if not db_event:
                return False

            calendar = db_event.calendar
            user_role = self._get_user_calendar_role(user_id, calendar)

            # Check if user has required calendar role
            if user_role in required_roles:
                return True

            # Special case: event creator can modify their own events if they have writer+ access
            if db_event.user_id == user_id and user_role in ["writer", "owner"]:
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking event permissions: {e}")
            return False
        finally:
            session.close()

    def check_event_visibility_permission(
        self, user_id: str, calendar_id: str, event_id: str
    ) -> bool:
        """
        Check if user can see a specific event based on visibility settings

        Visibility Rules:
        - default/public: Anyone with calendar read access can see
        - private: Only event creator and calendar owners can see
        - confidential: Only calendar owners can see
        """
        session = get_session(self.database_id)
        try:
            db_event = (
                session.query(Event)
                .filter(
                    and_(Event.calendar_id == calendar_id, Event.event_id == event_id)
                )
                .first()
            )

            if not db_event:
                return False

            calendar = db_event.calendar
            user_role = self._get_user_calendar_role(user_id, calendar)

            # Check basic calendar access first
            if user_role == "none":
                return False

            # Check visibility rules
            visibility = db_event.visibility or "default"

            if visibility in ["default", "public"]:
                # Anyone with calendar access can see
                return user_role in ["freeBusyReader", "reader", "writer", "owner"]
            elif visibility == "private":
                # Only event creator and calendar owners can see
                return db_event.user_id == user_id or user_role == "owner"
            elif visibility == "confidential":
                # Only calendar owners can see
                return user_role == "owner"

            return False

        except Exception as e:
            logger.error(f"Error checking event visibility permission: {e}")
            return False
        finally:
            session.close()

    def check_event_delete_permission(
        self, user_id: str, calendar_id: str, event_id: str
    ) -> bool:
        """
        Check if user has permission to delete a specific event

        Rules:
        - Event creator can delete their events (if they have writer+ access)
        - Calendar owner can delete any event
        - Users with writer access can delete events they created
        """
        session = get_session(self.database_id)
        try:
            # Get the event
            db_event = (
                session.query(Event)
                .filter(
                    and_(Event.calendar_id == calendar_id, Event.event_id == event_id)
                )
                .first()
            )

            if not db_event:
                return False

            calendar = db_event.calendar
            user_role = self._get_user_calendar_role(user_id, calendar)

            # Calendar owner can delete any event
            if user_role == "owner":
                return True

            # Event creator can delete if they have writer+ access to calendar
            if db_event.user_id == user_id and user_role in ["writer", "owner"]:
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking event delete permission: {e}")
            return False
        finally:
            session.close()

    def check_event_modification_permission(
        self, user_id: str, calendar_id: str, event_id: str
    ) -> bool:
        """
        Check if user can modify a specific event

        Rules:
        - Calendar owner can modify any event
        - Calendar writer can modify events they created
        - Guest permissions (guestsCanModify) are checked for attendees
        """
        session = get_session(self.database_id)
        try:
            db_event = (
                session.query(Event)
                .filter(
                    and_(Event.calendar_id == calendar_id, Event.event_id == event_id)
                )
                .first()
            )

            if not db_event:
                return False

            calendar = db_event.calendar
            user_role = self._get_user_calendar_role(user_id, calendar)

            # Owner can modify any event
            if user_role == "owner":
                return True

            # Writer can modify events they created
            if user_role == "writer" and db_event.user_id == user_id:
                return True

            # Check if user is an attendee with modify permissions
            if db_event.guestsCanModify:
                from database.models.user import User

                user = session.query(User).filter(User.user_id == user_id).first()
                if user:
                    attendee = (
                        session.query(Attendees)
                        .filter(
                            and_(
                                Attendees.event_id == event_id,
                                Attendees.user_id == user_id,
                            )
                        )
                        .first()
                    )
                    if attendee:
                        return True

            return False

        except Exception as e:
            logger.error(f"Error checking event modification permission: {e}")
            return False
        finally:
            session.close()

    def _convert_db_event_to_schema(self, db_event: Event) -> EventSchema:
        """Convert database Event model to EventSchema"""
        try:
            # Build organizer info from stored database fields
            organizer_info = {}
            if hasattr(db_event, "organizer_id") and db_event.organizer_id:
                organizer_info = {
                    "id": db_event.organizer_id,
                    "email": db_event.organizer_email or "",
                    "displayName": db_event.organizer_display_name or "",
                    "self": db_event.organizer_self or False,
                }

            # Basic event data
            event_data = {
                "id": db_event.event_id,
                "kind": "calendar#event",
                "etag": f'"{db_event.updated_at.timestamp()}"'
                if db_event.updated_at
                else None,
                "status": db_event.status,
                "htmlLink": f"https://calendar.google.com/event?eid={db_event.event_id}",
                "created": db_event.created_at.isoformat()
                if db_event.created_at
                else None,
                "updated": db_event.updated_at.isoformat()
                if db_event.updated_at
                else None,
                "summary": db_event.summary,
                "description": db_event.description,
                "location": db_event.location,
                "start": self._build_datetime_for_schema(
                    db_event.start_datetime, db_event.start_timezone
                ),
                "end": self._build_datetime_for_schema(
                    db_event.end_datetime, db_event.end_timezone
                ),
                "originalStartTime": self._build_original_start_time_for_schema(
                    db_event
                ),
                "recurringEventId": db_event.recurring_event_id,
                "recurrence": json.loads(db_event.recurrence)
                if db_event.recurrence
                else [],
                "visibility": db_event.visibility,
                "transparency": db_event.transparency,
                "iCalUID": db_event.iCalUID,
                "guestsCanInviteOthers": db_event.guestsCanInviteOthers,
                "guestsCanModify": db_event.guestsCanModify,
                "guestsCanSeeOtherGuests": db_event.guestsCanSeeOtherGuests,
                "privateCopy": db_event.privateCopy,
                "locked": db_event.locked,
                "hangoutLink": db_event.hangoutLink,
                "creator": organizer_info,
                "organizer": organizer_info,
                "sequence": db_event.sequence,
            }

            # Add eventType if present
            if hasattr(db_event, "eventType") and db_event.eventType:
                event_data["eventType"] = (
                    db_event.eventType.value
                    if hasattr(db_event.eventType, "value")
                    else str(db_event.eventType)
                )

            # Add colorId if present
            if hasattr(db_event, "color_id") and db_event.color_id:
                event_data["colorId"] = db_event.color_id

            # Convert attendees from database relationships
            attendees_list = []
            if db_event.attendees:
                for attendee in db_event.attendees:
                    attendee_data = {
                        "id": attendee.attendees_id,
                        "email": attendee.user.email if attendee.user else None,
                        "displayName": attendee.displayName,
                        "responseStatus": attendee.responseStatus,
                        "optional": attendee.optional,
                        "comment": attendee.comment,
                        "additionalGuests": attendee.additionalGuests,
                        "resource": attendee.resource,
                    }

                    # Set organizer and self flags based on stored organizer info
                    if organizer_info and "email" in organizer_info and attendee.user:
                        if attendee.user.email == organizer_info["email"]:
                            attendee_data["organizer"] = True
                            attendee_data["self"] = organizer_info.get("self", False)
                        else:
                            attendee_data["organizer"] = False
                            attendee_data["self"] = False
                    else:
                        attendee_data["organizer"] = False
                        attendee_data["self"] = False

                    attendees_list.append(attendee_data)
            event_data["attendees"] = attendees_list

            event_data["attendeesOmitted"] = False
            # Convert attachments from database relationships
            attachments_list = []
            if db_event.attachments:
                for attachment in db_event.attachments:
                    attachment_data = {
                        "fileUrl": attachment.file_url,
                        "title": attachment.file_url.split("/")[-1]
                        if attachment.file_url
                        else "attachment",
                    }
                    attachments_list.append(attachment_data)
            event_data["attachments"] = attachments_list

            # Add conference data if present
            if hasattr(db_event, "conferenceData") and db_event.conferenceData:
                conf = db_event.conferenceData
                conference_data = {}

                # Add conferenceSolution if available
                if conf.solution_type or conf.solution_name or conf.solution_icon_uri:
                    conference_solution = {}
                    if conf.solution_icon_uri:
                        conference_solution["iconUri"] = conf.solution_icon_uri
                    if conf.solution_type:
                        conference_solution["key"] = {"type": conf.solution_type}
                    if conf.solution_name:
                        conference_solution["name"] = conf.solution_name
                    conference_data["conferenceSolution"] = conference_solution

                # Add createRequest if available
                if conf.request_id or conf.create_solution_type or conf.status_code:
                    create_request = {}
                    if conf.request_id:
                        create_request["requestId"] = conf.request_id
                    if conf.create_solution_type:
                        create_request["conferenceSolutionKey"] = {
                            "type": conf.create_solution_type
                        }
                    if conf.status_code:
                        create_request["status"] = {"statusCode": conf.status_code}
                    conference_data["createRequest"] = create_request

                # Add entryPoints from JSON array or legacy field
                entry_points = []
                if conf.entry_points:
                    # Use new JSON array format
                    entry_points = conf.entry_points
                elif conf.meeting_uri:
                    # Fallback to legacy format
                    entry_points = [
                        {"entryPointType": "video", "uri": conf.meeting_uri}
                    ]

                if entry_points:
                    conference_data["entryPoints"] = entry_points

                # Add notes and signature if available
                if conf.notes:
                    conference_data["notes"] = conf.notes
                if conf.signature:
                    conference_data["signature"] = conf.signature

                event_data["conferenceData"] = conference_data

            # Add source if present
            if hasattr(db_event, "source") and db_event.source:
                event_data["source"] = db_event.source

            # Add extended properties if present
            if hasattr(db_event, "extendedProperties") and db_event.extendedProperties:
                ext_props = {}
                private_props = {}
                shared_props = {}

                for prop in db_event.extendedProperties:
                    if prop.scope == ExtendedPropertyScope.private:
                        private_props.update(prop.properties or {})
                    elif prop.scope == ExtendedPropertyScope.shared:
                        shared_props.update(prop.properties or {})

                if private_props:
                    ext_props["private"] = private_props
                if shared_props:
                    ext_props["shared"] = shared_props

                if ext_props:
                    event_data["extendedProperties"] = ext_props

            # Add working location properties if present
            if (
                hasattr(db_event, "workingLocationProperties")
                and db_event.workingLocationProperties
            ):
                working_loc = {
                    "type": db_event.workingLocationProperties.type,
                    "homeOffice": db_event.workingLocationProperties.homeOffice,
                }

                if db_event.workingLocationProperties.customLocationLabel:
                    working_loc["customLocation"] = {
                        "label": db_event.workingLocationProperties.customLocationLabel
                    }

                if db_event.workingLocationProperties.officeLocation:
                    working_loc["officeLocation"] = {
                        "buildingId": db_event.workingLocationProperties.officeLocation.buildingId,
                        "floorId": db_event.workingLocationProperties.officeLocation.floorId,
                        "deskId": db_event.workingLocationProperties.officeLocation.deskId,
                        "floorSectionId": db_event.workingLocationProperties.officeLocation.floorSectionId,
                        "label": db_event.workingLocationProperties.officeLocation.label,
                    }

                event_data["workingLocationProperties"] = working_loc

            # Add birthday properties if present
            if hasattr(db_event, "birthdayProperties") and db_event.birthdayProperties:
                from schemas.event import BirthdayProperties

                event_data["birthdayProperties"] = BirthdayProperties(
                    type=db_event.birthdayProperties.type
                )

            # Add focus time properties if present
            if (
                hasattr(db_event, "focusTimeProperties")
                and db_event.focusTimeProperties
            ):
                event_data["focusTimeProperties"] = db_event.focusTimeProperties

            # Add out of office properties if present
            if (
                hasattr(db_event, "outOfOfficeProperties")
                and db_event.outOfOfficeProperties
            ):
                event_data["outOfOfficeProperties"] = db_event.outOfOfficeProperties

            # Build reminders response format
            reminders_response = {"useDefault": False, "overrides": []}
            # Add reminders if present
            if db_event.reminders:
                # Check if any reminder uses default
                use_default = any(
                    reminder.use_default for reminder in db_event.reminders
                )

                if use_default:
                    reminders_response["useDefault"] = True

                # Build overrides list
                overrides = []
                for reminder in db_event.reminders:
                    overrides.append(
                        {
                            "method": reminder.method.value
                            if hasattr(reminder.method, "value")
                            else reminder.method,
                            "minutes": reminder.minutes,
                        }
                    )

                reminders_response["overrides"] = overrides

            event_data["reminders"] = reminders_response

            return EventSchema(**event_data)

        except Exception as e:
            logger.error(f"Error converting DB event to schema: {e}")
            raise

    def _parse_datetime_string(self, datetime_str: str) -> datetime:
        """Parse ISO datetime string to datetime object"""
        try:
            return parser.isoparse(datetime_str.replace("Z", "+00:00"))
        except Exception as e:
            logger.error(f"Error parsing datetime string {datetime_str}: {e}")
            raise ValueError(f"Invalid datetime format: {datetime_str}")

    def _parse_datetime_from_api_format(
        self, datetime_obj: DateTime
    ) -> tuple[datetime, str]:
        """Parse Google Calendar API DateTime format to datetime object and timezone"""
        try:
            if isinstance(datetime_obj, dict):
                # Validate timezone
                self._validate_timezone(datetime_obj["timeZone"])
                if "dateTime" in datetime_obj:
                    # Timed event
                    dt = self._parse_datetime_string(datetime_obj["dateTime"])
                    tz = datetime_obj["timeZone"] or "UTC"
                    return dt, tz
                elif "date" in datetime_obj:
                    # All-day event - parse date string and create datetime at midnight
                    from datetime import date

                    parsed_date = date.fromisoformat(datetime_obj["date"])
                    dt = datetime.combine(parsed_date, datetime.min.time())
                    tz = datetime_obj["timeZone"] or "UTC"
                    return dt, tz
                else:
                    raise ValueError(
                        "DateTime object must have either dateTime or date"
                    )
            else:
                self._validate_timezone(datetime_obj.timeZone)
                if datetime_obj.dateTime:
                    # Timed event
                    dt = self._parse_datetime_string(datetime_obj.dateTime)
                    tz = datetime_obj.timeZone or "UTC"
                    return dt, tz
                elif datetime_obj.date:
                    # All-day event - parse date string and create datetime at midnight
                    from datetime import date

                    parsed_date = date.fromisoformat(datetime_obj.date)
                    dt = datetime.combine(parsed_date, datetime.min.time())
                    tz = datetime_obj.timeZone or "UTC"
                    return dt, tz
                else:
                    raise ValueError(
                        "DateTime object must have either dateTime or date"
                    )
        except Exception as e:
            logger.error(f"Error parsing API DateTime format: {e}")
            raise ValueError(f"Invalid DateTime format: {e}")

    def _process_create_attendees(self, db_event: Event, attendees_data: list, session):
        """Process attendees from create request"""
        if not attendees_data:
            return

        from database.models.event import Attendees

        # Check whether email id exist
        self.check_attendees_email_id(session, attendees_data)

        for attendee_data in attendees_data:
            # Find user based on email
            user = session.query(User).filter(User.email == attendee_data.email).first()
            if not user:
                logger.warning(
                    f"Attendee email '{attendee_data.email}' not found in database, skipping"
                )
                continue

            attendee = Attendees(
                attendees_id=str(uuid.uuid4()),
                event_id=db_event.event_id,
                user_id=user.user_id,
                displayName=attendee_data.displayName,
                optional=attendee_data.optional or False,
                resource=attendee_data.resource or False,
                responseStatus=attendee_data.responseStatus or "needsAction",
                comment=attendee_data.comment,
                additionalGuests=attendee_data.additionalGuests or 0,
            )
            session.add(attendee)

    def _validate_attachment_file_url(self, attachments_data):
        from urllib.parse import urlparse

        for data in attachments_data:
            if "fileUrl" not in data.keys():
                raise ValueError("fileUrl is required")
            s = str(data.get("fileUrl")).strip()
            parsed = urlparse(s)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError(f"Invalid 'fileUrl': must be an https URL {s}")

    def _process_create_attachments(
        self, db_event: Event, attachments_data: list, session
    ):
        """Process attachments from create request"""
        if not attachments_data:
            return

        from database.models.event import Attachment

        # Validate attachment file url
        self._validate_attachment_file_url(attachments_data)

        for attachment_data in attachments_data:
            attachment = Attachment(
                attachment_id=str(uuid.uuid4()),
                event_id=db_event.event_id,
                file_url=attachment_data.get("fileUrl"),
            )
            session.add(attachment)

    def _process_create_conference_data(
        self, db_event: Event, conference_data, session
    ):
        """Process conference data from create request using new comprehensive schema"""
        from database.models.event import ConferenceData as DBConferenceData

        # Handle both old dict format and new Pydantic ConferenceData model
        conference_id = None
        request_id = None
        solution_type = None
        solution_name = None
        solution_icon_uri = None
        create_solution_type = None
        status_code = None
        entry_points_json = None
        notes = None
        signature = None
        meeting_uri = None  # Legacy field
        label = None  # Legacy field

        if hasattr(conference_data, "conferenceId"):
            # New Pydantic ConferenceData model
            conference_id = conference_data.conferenceId

            # Validate conference Id uniqueness
            conference_data_obj = (
                session.query(ConferenceDataModel)
                .filter(ConferenceDataModel.id == conference_id)
                .first()
            )
            if conference_data_obj is not None:
                raise ValueError(
                    f"Conference Id '{conference_id}' already exists. Please use different ConferenceId"
                )

            # Extract from conferenceSolution if present
            if conference_data.conferenceSolution:
                solution_type = (
                    conference_data.conferenceSolution.key.type
                    if conference_data.conferenceSolution.key
                    else None
                )
                solution_name = conference_data.conferenceSolution.name
                solution_icon_uri = conference_data.conferenceSolution.iconUri
                label = (
                    conference_data.conferenceSolution.name
                )  # For backward compatibility

            # Process entryPoints array as JSON
            if conference_data.entryPoints:
                entry_points_list = []
                for entry_point in conference_data.entryPoints:
                    entry_point_data = {
                        "accessCode": entry_point.accessCode
                        if hasattr(entry_point, "accessCode")
                        else None,
                        "entryPointType": entry_point.entryPointType
                        if hasattr(entry_point, "entryPointType")
                        else None,
                        "label": entry_point.label
                        if hasattr(entry_point, "label")
                        else None,
                        "meetingCode": entry_point.meetingCode
                        if hasattr(entry_point, "meetingCode")
                        else None,
                        "passcode": entry_point.passcode
                        if hasattr(entry_point, "passcode")
                        else None,
                        "pin": entry_point.pin if hasattr(entry_point, "pin") else None,
                        "uri": entry_point.uri if hasattr(entry_point, "uri") else None,
                    }
                    entry_points_list.append(entry_point_data)

                    # Set legacy meeting_uri from first video entry point
                    if (
                        hasattr(entry_point, "entryPointType")
                        and entry_point.entryPointType == "video"
                        and hasattr(entry_point, "uri")
                        and not meeting_uri
                    ):
                        meeting_uri = entry_point.uri

                entry_points_json = entry_points_list

            # Extract notes and signature
            notes = conference_data.notes if hasattr(conference_data, "notes") else None
            signature = (
                conference_data.signature
                if hasattr(conference_data, "signature")
                else None
            )

            # Extract from createRequest if present
            if conference_data.createRequest:
                request_id = conference_data.createRequest.requestId
                if (
                    session.query(ConferenceDataModel)
                    .filter(ConferenceDataModel.request_id == request_id)
                    .first()
                ):
                    status_code = "failure"
                    # Create database conference record with all fields
                    db_conference = DBConferenceData(
                        id=conference_id or str(uuid.uuid4()),
                        event_id=db_event.event_id,
                        solution_type=solution_type,
                        solution_name=solution_name,
                        solution_icon_uri=solution_icon_uri,
                        status_code=status_code,
                        notes=notes,
                        signature=signature,
                        meeting_uri=meeting_uri,  # Legacy field for backward compatibility
                        label=label,  # Legacy field for backward compatibility
                    )
                    session.add(db_conference)
                else:
                    if (
                        conference_data.createRequest.conferenceSolutionKey
                        and conference_data.createRequest.conferenceSolutionKey.type
                    ):
                        create_solution_type = (
                            conference_data.createRequest.conferenceSolutionKey.type
                        )

                    status_code = "success"

                    # Create database conference record with all fields
                    db_conference = DBConferenceData(
                        id=conference_id or str(uuid.uuid4()),
                        event_id=db_event.event_id,
                        solution_type=solution_type,
                        solution_name=solution_name,
                        solution_icon_uri=solution_icon_uri,
                        request_id=request_id,
                        create_solution_type=create_solution_type,
                        status_code=status_code,
                        entry_points=entry_points_json,
                        notes=notes,
                        signature=signature,
                        meeting_uri=meeting_uri,  # Legacy field for backward compatibility
                        label=label,  # Legacy field for backward compatibility
                    )
                    session.add(db_conference)

        else:
            # Legacy dict format support
            conference_id = conference_data.get("conferenceId")
            # Validate conference Id uniqueness
            conference_data_obj = (
                session.query(ConferenceDataModel)
                .filter(
                    ConferenceDataModel.id == conference_id,
                    ConferenceDataModel.event_id == db_event.event_id,
                )
                .first()
            )
            if conference_data_obj is not None:
                raise ValueError(
                    f"Conference Id '{conference_id}' already exists. Please use different ConferenceId"
                )

            notes = conference_data.get("notes")
            signature = conference_data.get("signature")

            if conference_data.get("conferenceSolution"):
                solution_type = (
                    conference_data["conferenceSolution"].get("key", {}).get("type")
                )
                solution_name = conference_data["conferenceSolution"].get("name")
                solution_icon_uri = conference_data["conferenceSolution"].get("iconUri")
                label = conference_data["conferenceSolution"].get(
                    "name"
                )  # For backward compatibility

            if conference_data.get("entryPoints"):
                entry_points_list = []
                for entry_point in conference_data["entryPoints"]:
                    entry_points_list.append(
                        {
                            "accessCode": entry_point.get("accessCode"),
                            "entryPointType": entry_point.get("entryPointType"),
                            "label": entry_point.get("label"),
                            "meetingCode": entry_point.get("meetingCode"),
                            "passcode": entry_point.get("passcode"),
                            "pin": entry_point.get("pin"),
                            "uri": entry_point.get("uri"),
                        }
                    )

                    # Set legacy meeting_uri from first video entry point
                    if (
                        entry_point.get("entryPointType") == "video"
                        and entry_point.get("uri")
                        and not meeting_uri
                    ):
                        meeting_uri = entry_point.get("uri")

                entry_points_json = entry_points_list

            if conference_data.get("createRequest"):
                request_id = conference_data["createRequest"].get("requestId")
                if (
                    session.query(ConferenceDataModel)
                    .filter(ConferenceDataModel.request_id == request_id)
                    .first()
                ):
                    status_code = "failure"
                    # Create database conference record with all fields
                    db_conference = DBConferenceData(
                        id=conference_id or str(uuid.uuid4()),
                        event_id=db_event.event_id,
                        solution_type=solution_type,
                        solution_name=solution_name,
                        solution_icon_uri=solution_icon_uri,
                        status_code=status_code,
                        notes=notes,
                        signature=signature,
                        meeting_uri=meeting_uri,  # Legacy field for backward compatibility
                        label=label,  # Legacy field for backward compatibility
                    )
                    session.add(db_conference)
                else:
                    if (
                        conference_data["createRequest"]
                        .get("conferenceSolutionKey", {})
                        .get("type")
                    ):
                        create_solution_type = conference_data["createRequest"][
                            "conferenceSolutionKey"
                        ].get("type")
                    status_code = "success"
                    # Create database conference record with all fields
                    db_conference = DBConferenceData(
                        id=conference_id or str(uuid.uuid4()),
                        event_id=db_event.event_id,
                        solution_type=solution_type,
                        solution_name=solution_name,
                        solution_icon_uri=solution_icon_uri,
                        request_id=request_id,
                        create_solution_type=create_solution_type,
                        status_code=status_code,
                        entry_points=entry_points_json,
                        notes=notes,
                        signature=signature,
                        meeting_uri=meeting_uri,  # Legacy field for backward compatibility
                        label=label,  # Legacy field for backward compatibility
                    )

                    session.add(db_conference)

    def _process_create_extended_properties(self, db_event: Event, ext_props, session):
        """Process extended properties from create request"""
        from database.models.event import ExtendedProperty

        if ext_props.get("private"):
            private_prop = ExtendedProperty(
                id=str(uuid.uuid4()),
                event_id=db_event.event_id,
                scope="private",
                properties=ext_props["private"],
            )
            session.add(private_prop)

        if ext_props.get("shared"):
            shared_prop = ExtendedProperty(
                id=str(uuid.uuid4()),
                event_id=db_event.event_id,
                scope="shared",
                properties=ext_props["shared"],
            )
            session.add(shared_prop)

    def _process_create_reminders(self, db_event: Event, reminders_data, session):
        """Process reminders from create request"""
        if not reminders_data:
            return

        try:
            # Handle useDefault logic
            if reminders_data.useDefault:
                # For useDefault=True, create a standard default reminder
                default_reminder = Reminder(
                    id=str(uuid.uuid4()),
                    event_id=db_event.event_id,
                    use_default=True,
                    method="popup",  # Default method
                    minutes=10,  # Default 10 minutes before
                )
                session.add(default_reminder)
                logger.info(f"Added default reminder for event {db_event.event_id}")

            else:
                # Handle custom reminder overrides
                if reminders_data.overrides and len(reminders_data.overrides) > 0:
                    for override in reminders_data.overrides:
                        # Validate reminder data
                        if not hasattr(override, "method") or not hasattr(
                            override, "minutes"
                        ):
                            logger.warning(
                                f"Invalid reminder override for event {db_event.event_id}: missing method or minutes"
                            )
                            continue

                        # Validate method
                        if override.method not in ["email", "popup"]:
                            logger.warning(
                                f"Invalid reminder method '{override.method}' for event {db_event.event_id}, skipping"
                            )
                            continue

                        # Validate minutes (should be non-negative)
                        if override.minutes < 0:
                            logger.warning(
                                f"Invalid reminder minutes '{override.minutes}' for event {db_event.event_id}, skipping"
                            )
                            continue

                        custom_reminder = Reminder(
                            id=str(uuid.uuid4()),
                            event_id=db_event.event_id,
                            use_default=False,
                            method=override.method,
                            minutes=override.minutes,
                        )
                        session.add(custom_reminder)
                        logger.info(
                            f"Added custom reminder for event {db_event.event_id}: {override.method} {override.minutes} minutes before"
                        )

        except Exception as e:
            logger.error(
                f"Error processing reminders for event {db_event.event_id}: {e}"
            )
            # Don't raise the exception - reminders are not critical for event creation

    def _process_create_reminders_for_update(
        self, db_event: Event, reminders_data, session
    ):
        """Process reminders from create request"""
        if not reminders_data:
            return

        try:
            # Handle useDefault logic
            if reminders_data.get("useDefault"):
                # For useDefault=True, create a standard default reminder
                default_reminder = Reminder(
                    id=str(uuid.uuid4()),
                    event_id=db_event.event_id,
                    use_default=True,
                    method="popup",  # Default method
                    minutes=10,  # Default 10 minutes before
                )
                session.add(default_reminder)
                logger.info(f"Added default reminder for event {db_event.event_id}")

            else:
                # Handle custom reminder overrides
                if (
                    reminders_data.get("overrides")
                    and len(reminders_data.get("overrides")) > 0
                ):
                    for override in reminders_data.get("overrides"):
                        # Validate reminder data

                        custom_reminder = Reminder(
                            id=str(uuid.uuid4()),
                            event_id=db_event.event_id,
                            use_default=False,
                            method=override.get("method").value,
                            minutes=override.get("minutes"),
                        )
                        session.add(custom_reminder)
                        logger.info(
                            f"Added custom reminder for event {db_event.event_id}: {override.get('method')} {override.get('minutes')} minutes before"
                        )

        except Exception as e:
            logger.error(
                f"Error processing reminders for event {db_event.event_id}: {e}"
            )
            # Don't raise the exception - reminders are not critical for event creation

    def _process_create_working_location(
        self, db_event: Event, working_location, session
    ):
        """Process working location properties from create request"""
        from database.models.event import OfficeLocation, WorkingLocationProperties

        # Convert to dictionary if it's a Pydantic model
        if not isinstance(working_location, dict):
            if hasattr(working_location, "model_dump"):
                working_location = working_location.model_dump()
            elif hasattr(working_location, "dict"):
                working_location = working_location.dict()

        # Create working location
        working_loc = WorkingLocationProperties(
            working_location_id=str(uuid.uuid4()),
            event_id=db_event.event_id,
            type=working_location.get("type"),
            homeOffice=working_location.get("homeOffice"),
            customLocationLabel=working_location.get("customLocation", {}).get("label")
            if working_location.get("customLocation")
            else None,
        )

        # Handle office location if specified
        if working_location.get("officeLocation"):
            office_loc = OfficeLocation(
                id=str(uuid.uuid4()),
                label=working_location["officeLocation"].get("label", "")
                if working_location.get("officeLocation")
                and working_location.get("officeLocation").get("label")
                else "",
                buildingId=working_location["officeLocation"].get("buildingId")
                if working_location.get("officeLocation")
                and working_location.get("officeLocation").get("buildingId")
                else "",
                floorId=working_location["officeLocation"].get("floorId")
                if working_location.get("officeLocation")
                and working_location.get("officeLocation").get("floorId")
                else "",
                deskId=working_location["officeLocation"].get("deskId")
                if working_location.get("officeLocation")
                and working_location.get("officeLocation").get("deskId")
                else "",
                floorSectionId=working_location["officeLocation"].get("floorSectionId")
                if working_location.get("officeLocation")
                and working_location.get("officeLocation").get("floorSectionId")
                else "",
            )
            session.add(office_loc)
            session.flush()
            working_loc.officeLocationId = office_loc.id

        session.add(working_loc)

    def _process_create_birthday_properties(
        self, db_event: Event, birthday_props, session
    ):
        """Process birthday properties from create request"""
        from database.models.event import BirthdayProperties

        # Handle both dict and object formats
        if isinstance(birthday_props, dict):
            birthday_type = birthday_props.get("type", "birthday")
        else:
            birthday_type = getattr(birthday_props, "type", "birthday")

        # Validate that type is "birthday"
        if birthday_type != "birthday":
            raise ValueError(
                f"Invalid birthday properties type: {birthday_type}. Must be 'birthday'."
            )

        birthday_property = BirthdayProperties(
            id=str(uuid.uuid4()),
            event_id=db_event.event_id,
            type="birthday",  # Always enforce "birthday" for birthday events
        )
        session.add(birthday_property)

    def _process_import_birthday_properties(
        self, db_event: Event, birthday_props, session
    ):
        """Process birthday properties from import request"""
        from database.models.event import BirthdayProperties

        # Validate that type is "birthday"
        if hasattr(birthday_props, "type") and birthday_props.type != "birthday":
            raise ValueError(
                f"Invalid birthday properties type: {birthday_props.type}. Must be 'birthday'."
            )

        birthday_property = BirthdayProperties(
            id=str(uuid.uuid4()),
            event_id=db_event.event_id,
            type="birthday",  # Always enforce "birthday" for birthday events
        )
        session.add(birthday_property)

    def list_events(
        self,
        user_id: str,
        calendar_id: str,
        event_types: Optional[str] = None,
        ical_uid: Optional[str] = None,
        max_attendees: Optional[int] = None,
        max_results: Optional[int] = None,
        order_by: Optional[str] = None,
        page_token: Optional[str] = None,
        private_extended_property: Optional[str] = None,
        q: Optional[str] = None,
        shared_extended_property: Optional[str] = None,
        show_deleted: Optional[bool] = None,
        show_hidden_invitations: Optional[bool] = None,
        single_events: Optional[bool] = None,
        sync_token: Optional[str] = None,
        time_max: Optional[str] = None,
        time_min: Optional[str] = None,
        time_zone: Optional[str] = None,
        updated_min: Optional[str] = None,
    ) -> EventListResponse:
        """
        List events from calendar with ACL permission and visibility checking
        Only returns events the user has permission to see based on visibility settings

        GET /calendars/{calendarId}/events
        """
        session = get_session(self.database_id)
        try:
            # Parse page token to get offset
            offset = 0
            if page_token:
                try:
                    offset = self._decode_page_token(page_token)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid page token: {page_token}, error: {e}")
                    offset = 0

            # Get calendar and check basic access, optionally filtering by timezone
            calendar_query = session.query(Calendar).filter(
                Calendar.calendar_id == calendar_id
            )

            # Add timezone filter if provided
            if time_zone:
                # Validate timezone
                self._validate_timezone(time_zone)
                calendar_query = calendar_query.filter(Calendar.time_zone == time_zone)

            calendar = calendar_query.first()
            if not calendar:
                if time_zone:
                    raise ValueError(
                        f"Calendar {calendar_id} not found with timezone {time_zone}"
                    )
                else:
                    raise ValueError(f"Calendar {calendar_id} not found")

            # Get user's role on this calendar
            user_role = self._get_user_calendar_role(user_id, calendar)
            if user_role == "none":
                raise PermissionError(
                    f"User '{user_id}' has no access to calendar '{calendar_id}'"
                )

            # Expand recurring events in the time range if time filters are provided
            if time_min or time_max:
                try:
                    time_min_dt = (
                        self._parse_datetime_string(time_min)
                        if time_min
                        else datetime.now(timezone.utc) - timedelta(days=30)
                    )
                    time_max_dt = (
                        self._parse_datetime_string(time_max)
                        if time_max
                        else datetime.now(timezone.utc) + timedelta(days=90)
                    )

                    # Generate recurring event instances for the time range
                    self.expand_recurring_events(
                        user_id, calendar_id, time_min_dt, time_max_dt
                    )
                except Exception as e:
                    logger.warning(f"Error expanding recurring events: {e}")

            # Build base query for all events in the calendar
            query = session.query(Event).filter(Event.calendar_id == calendar_id)

            # Apply visibility filters based on user role
            if user_role == "freeBusyReader":
                # FreeBusyReader can only see basic event info, no detailed content
                # For simplicity, we'll allow them to see public/default events
                query = query.filter(Event.visibility.in_(["default", "public"]))
            elif user_role == "reader":
                # Readers can see public/default events and their own private events

                user = session.query(User).filter(User.user_id == user_id).first()
                if user:
                    query = query.filter(
                        or_(
                            Event.visibility.in_(["default", "public"]),
                            and_(
                                Event.visibility == "private", Event.user_id == user_id
                            ),
                        )
                    )
            elif user_role == "writer":
                # Writers can see public/default events and their own private events
                query = query.filter(
                    or_(
                        Event.visibility.in_(["default", "public"]),
                        and_(Event.visibility == "private", Event.user_id == user_id),
                    )
                )
            # Owners can see all events (no additional filter)

            # Apply other filters
            if not show_deleted:
                query = query.filter(Event.status != "cancelled")
            else:
                query = query.filter(Event.status == "cancelled")

            if time_min:
                time_min_dt = self._parse_datetime_string(time_min)
                query = query.filter(Event.end_datetime >= time_min_dt)

            if time_max:
                time_max_dt = self._parse_datetime_string(time_max)
                query = query.filter(Event.start_datetime < time_max_dt)

            if updated_min:
                updated_min_dt = self._parse_datetime_string(updated_min)
                query = query.filter(Event.updated_at >= updated_min_dt)

            # Handle iCalUID filter (mutually exclusive with q)
            if ical_uid:
                query = query.filter(Event.iCalUID == ical_uid)
            elif q:
                # Search in summary, description, location (only if no iCalUID)
                search_filter = or_(
                    Event.summary.ilike(f"%{q}%"),
                    Event.description.ilike(f"%{q}%"),
                    Event.location.ilike(f"%{q}%"),
                )
                query = query.filter(search_filter)

            # Handle eventTypes filter
            if event_types:
                # Parse comma-separated event types
                event_type_list = [
                    t.strip() for t in event_types.split(",") if t.strip()
                ]
                if event_type_list:
                    # Convert string event types to enum values for filtering
                    enum_values = []
                    for event_type in event_type_list:
                        if event_type == "default":
                            enum_values.append("default")
                        elif event_type == "birthday":
                            enum_values.append("birthday")
                        elif event_type == "focusTime":
                            enum_values.append("focusTime")
                        elif event_type == "fromGmail":
                            enum_values.append("fromGmail")
                        elif event_type == "outOfOffice":
                            enum_values.append("outOfOffice")
                        elif event_type == "workingLocation":
                            enum_values.append("workingLocation")
                    if enum_values:
                        query = query.filter(Event.eventType.in_(enum_values))

            # Handle extended properties filters
            if private_extended_property or shared_extended_property:
                from database.models.event import ExtendedProperty

                if private_extended_property:
                    # Parse propertyName=value format
                    if "=" in private_extended_property:
                        prop_name, prop_value = private_extended_property.split("=", 1)
                        query = query.join(
                            ExtendedProperty,
                            Event.event_id == ExtendedProperty.event_id,
                        ).filter(
                            and_(
                                ExtendedProperty.scope == "private",
                                ExtendedProperty.properties[prop_name].astext
                                == prop_value,
                            )
                        )

                if shared_extended_property:
                    # Parse propertyName=value format
                    if "=" in shared_extended_property:
                        prop_name, prop_value = shared_extended_property.split("=", 1)
                        query = query.join(
                            ExtendedProperty,
                            Event.event_id == ExtendedProperty.event_id,
                        ).filter(
                            and_(
                                ExtendedProperty.scope == "shared",
                                ExtendedProperty.properties[prop_name].astext
                                == prop_value,
                            )
                        )

            # Handle showHiddenInvitations filter
            # Note: This would require additional database schema to track hidden invitations
            # For now, we'll ignore this parameter as it's not implemented in our schema

            # Apply ordering
            if order_by == "updated":
                query = query.order_by(desc(Event.updated_at))
            else:  # Default to startTime
                query = query.order_by(asc(Event.start_datetime))

            # Apply pagination
            if max_results or max_results == 0:
                query = query.limit(max_results)

            # Apply offset for pagination
            if offset > 0:
                query = query.offset(offset)

            events = query.all()

            # Filter events based on individual visibility permissions
            visible_events = []
            for event in events:
                if self.check_event_visibility_permission(
                    user_id, calendar_id, event.event_id
                ):
                    visible_events.append(event)

            # Convert to schema
            event_schemas = []
            for event in visible_events:
                event_schema = self._convert_db_event_to_schema(event)

                # Apply maxAttendees filtering to each event
                if (
                    max_attendees is not None
                    and event_schema.attendees
                    and len(event_schema.attendees) > max_attendees
                ):
                    event_schema.attendees = event_schema.attendees[:max_attendees]
                    event_schema.attendeesOmitted = True

                event_schemas.append(event_schema)

            # Determine response timezone
            response_timezone = time_zone or (calendar.time_zone if calendar else "UTC")

            # Determine access role for response
            access_role = user_role if user_role != "freeBusyReader" else "reader"

            return EventListResponse(
                kind="calendar#events",
                etag=f"etag-events-{calendar_id}",
                summary=calendar.summary if calendar else None,
                description=calendar.description if calendar else None,
                updated=datetime.now(timezone.utc).isoformat(),
                timeZone=response_timezone,
                accessRole=access_role,
                defaultReminders=[],
                nextPageToken=None,  # Pagination not implemented in this simple version
                nextSyncToken=None,  # Sync not implemented in this simple version
                items=event_schemas,
            )

        except Exception as e:
            logger.error(f"Error listing events for calendar {calendar_id}: {e}")
            raise
        finally:
            session.close()

    def create_recurrence_event_instance(
        self,
        session,
        user_id,
        recurring_event_id,
        calendar_id,
        event_request,
        query_params,
        total_occurences,
        datetime_dict,
    ):
        # Create database event with Google API fields
        # Generate event ID for single event
        try:
            for occurence in total_occurences:
                # Convert string datetime to datetime object
                occurrence_dt = self._parse_datetime_string(occurence)

                # Calculate end datetime by adding duration
                duration = datetime_dict.get("duration")
                end_datetime = occurrence_dt + duration
                event_id = str(uuid.uuid4())
                db_event = Event(
                    event_id=event_id,
                    calendar_id=calendar_id,
                    user_id=user_id,
                    recurring_event_id=recurring_event_id,
                    summary=event_request.summary,
                    description=event_request.description,
                    location=event_request.location,
                    start_datetime=occurrence_dt,
                    end_datetime=end_datetime,
                    start_timezone=datetime_dict.get("start_timezone") or "UTC",
                    end_timezone=datetime_dict.get("end_timezone") or "UTC",
                    recurrence=json.dumps(event_request.recurrence)
                    if event_request.recurrence
                    else None,
                    status=event_request.status or "confirmed",
                    visibility=event_request.visibility or "default",
                    sequence=event_request.sequence or 0,
                    source=event_request.source,
                    # Store originalStartTime fields
                    originalStartTime_date=occurrence_dt
                    if "T" not in occurence
                    else None,
                    originalStartTime_dateTime=occurrence_dt
                    if "T" in occurence
                    else None,
                    originalStartTime_timeZone=datetime_dict.get("start_timezone")
                    or "UTC",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )

                # Handle optional Google API fields
                if hasattr(event_request, "eventType") and event_request.eventType:
                    try:
                        db_event.eventType = EventTypeEnum(event_request.eventType)
                    except ValueError:
                        # Default to 'default' if eventType is not supported
                        db_event.eventType = EventTypeEnum.DEFAULT

                if hasattr(event_request, "colorId") and event_request.colorId:
                    db_event.color_id = event_request.colorId

                if (
                    hasattr(event_request, "transparency")
                    and event_request.transparency
                ):
                    db_event.transparency = event_request.transparency

                if hasattr(event_request, "iCalUID") and event_request.iCalUID:
                    db_event.iCalUID = event_request.iCalUID
                else:
                    db_event.iCalUID = datetime_dict.get("iCalUID")

                # Handle guest permissions
                if (
                    hasattr(event_request, "guestsCanInviteOthers")
                    and event_request.guestsCanInviteOthers is not None
                ):
                    db_event.guestsCanInviteOthers = event_request.guestsCanInviteOthers
                if (
                    hasattr(event_request, "guestsCanModify")
                    and event_request.guestsCanModify is not None
                ):
                    db_event.guestsCanModify = event_request.guestsCanModify
                if (
                    hasattr(event_request, "guestsCanSeeOtherGuests")
                    and event_request.guestsCanSeeOtherGuests is not None
                ):
                    db_event.guestsCanSeeOtherGuests = (
                        event_request.guestsCanSeeOtherGuests
                    )

                # Set organizer information for recurring event instances
                creator_user = (
                    session.query(User).filter(User.user_id == user_id).first()
                )
                if creator_user:
                    db_event.organizer_id = creator_user.user_id
                    db_event.organizer_email = creator_user.email
                    db_event.organizer_display_name = creator_user.name
                    db_event.organizer_self = True

                session.add(db_event)
                session.flush()  # Get the event ID for related records

                # Handle attendees if provided
                if event_request.attendees:
                    self._process_create_attendees(
                        db_event, event_request.attendees, session
                    )

                # Handle attachments if provided (only if supported by client)
                if event_request.attachments and (
                    query_params is None
                    or (
                        query_params.get("supportsAttachments")
                        if query_params.get("supportsAttachments") is not None
                        else True
                    )
                ):
                    self._process_create_attachments(
                        db_event, event_request.attachments, session
                    )

                # Handle conference data if provided (based on version)
                if event_request.conferenceData and (
                    query_params is None
                    or (query_params.get("conferenceDataVersion") or 0) >= 1
                ):
                    try:
                        self._process_create_conference_data(
                            db_event, event_request.conferenceData, session
                        )
                    except Exception as e:
                        raise

                # Handle extended properties if provided
                if event_request.extendedProperties:
                    self._process_create_extended_properties(
                        db_event, event_request.extendedProperties, session
                    )

                # Handle reminders if provided
                if event_request.reminders:
                    self._process_create_reminders(
                        db_event, event_request.reminders, session
                    )

                # Handle working location properties if provided
                if event_request.workingLocationProperties:
                    self._process_create_working_location(
                        db_event, event_request.workingLocationProperties, session
                    )

                # Handle type-specific properties
                if (
                    event_request.eventType == "birthday"
                    and event_request.birthdayProperties
                ):
                    self._process_create_birthday_properties(
                        db_event, event_request.birthdayProperties, session
                    )
                elif (
                    event_request.eventType == "focusTime"
                    and event_request.focusTimeProperties
                ):
                    # Convert Pydantic model to dict for JSON storage
                    if hasattr(event_request.focusTimeProperties, "model_dump"):
                        db_event.focusTimeProperties = (
                            event_request.focusTimeProperties.model_dump()
                        )
                    elif hasattr(event_request.focusTimeProperties, "dict"):
                        db_event.focusTimeProperties = (
                            event_request.focusTimeProperties.dict()
                        )
                    else:
                        db_event.focusTimeProperties = event_request.focusTimeProperties
                elif (
                    event_request.eventType == "outOfOffice"
                    and event_request.outOfOfficeProperties
                ):
                    # Convert Pydantic model to dict for JSON storage
                    if hasattr(event_request.outOfOfficeProperties, "model_dump"):
                        db_event.outOfOfficeProperties = (
                            event_request.outOfOfficeProperties.model_dump()
                        )
                    elif hasattr(event_request.outOfOfficeProperties, "dict"):
                        db_event.outOfOfficeProperties = (
                            event_request.outOfOfficeProperties.dict()
                        )
                    else:
                        db_event.outOfOfficeProperties = (
                            event_request.outOfOfficeProperties
                        )

                # Handle source if provided
                if event_request.source:
                    db_event.source = event_request.source

                session.commit()
                session.refresh(db_event)

        except Exception as e:
            logger.error(f"Error while creating recurring event {e}")

    def create_event(
        self,
        user_id: str,
        calendar_id: str,
        event_request: EventCreateRequest,
        query_params,
    ) -> EventSchema:
        """
        Create a new event following Google Calendar API v3 structure
        Handles both single events and recurring events based on recurrence data

        POST /calendars/{calendarId}/events
        """
        session = get_session(self.database_id)
        try:
            # Verify calendar belongs to user
            calendar = (
                session.query(Calendar)
                .filter(
                    Calendar.calendar_id == calendar_id, Calendar.user_id == user_id
                )
                .first()
            )
            if not calendar:
                raise ValueError(f"Calendar {calendar_id} not found for user {user_id}")

            # Generate event ID for single event
            event_id = str(uuid.uuid4())

            # Parse start and end datetime from Google API format
            start_dt, start_tz = self._parse_datetime_from_api_format(
                event_request.start
            )
            end_dt, end_tz = self._parse_datetime_from_api_format(event_request.end)

            # Validate datetime consistency - handle timezone-aware vs naive comparison
            def safe_datetime_compare(dt1, dt2):
                """Compare two datetimes, handling timezone awareness differences"""
                # If both are naive or both are aware, compare directly
                if (dt1.tzinfo is None) == (dt2.tzinfo is None):
                    return dt1 <= dt2

                # If one is naive and one is aware, make both naive for comparison
                # This is safe for validation since we're just checking chronological order
                naive_dt1 = dt1.replace(tzinfo=None) if dt1.tzinfo is not None else dt1
                naive_dt2 = dt2.replace(tzinfo=None) if dt2.tzinfo is not None else dt2
                return naive_dt1 <= naive_dt2

            # Validate datetime consistency - end_dt should be after start_dt
            if safe_datetime_compare(end_dt, start_dt):
                raise ValueError("Event end time must be after start time")

            # Validate originalStartTime fields if provided and prepare values for storage
            original_start_date = None
            original_start_datetime = None
            original_start_timezone = None

            if event_request.originalStartTime:
                # Validate that originalStartTime matches start field values
                if event_request.start.date and event_request.originalStartTime.date:
                    # Both are all-day events - dates must match
                    if event_request.start.date != event_request.originalStartTime.date:
                        raise ValueError(
                            "originalStartTime.date must match start.date when both are provided"
                        )
                    original_start_date = parser.parse(
                        event_request.originalStartTime.date
                    ).date()
                elif (
                    event_request.start.dateTime
                    and event_request.originalStartTime.dateTime
                ):
                    # Both are timed events - dateTime must match
                    if (
                        event_request.start.dateTime
                        != event_request.originalStartTime.dateTime
                    ):
                        raise ValueError(
                            "originalStartTime.dateTime must match start.dateTime when both are provided"
                        )
                    original_start_datetime = self._parse_datetime_string(
                        event_request.originalStartTime.dateTime
                    )
                elif (
                    event_request.start.date
                    and event_request.originalStartTime.dateTime
                ) or (
                    event_request.start.dateTime
                    and event_request.originalStartTime.date
                ):
                    # Mismatched types (one is all-day, other is timed)
                    raise ValueError(
                        "originalStartTime and start must both be either all-day (date) or timed (dateTime) events"
                    )

                # Validate timezone matches
                if (
                    event_request.start.timeZone
                    and event_request.originalStartTime.timeZone
                ):
                    if (
                        event_request.start.timeZone
                        != event_request.originalStartTime.timeZone
                    ):
                        raise ValueError(
                            "originalStartTime.timeZone must match start.timeZone when both are provided"
                        )

                # Store originalStartTime timezone
                original_start_timezone = event_request.originalStartTime.timeZone
            else:
                # If originalStartTime not provided, use start field values
                if event_request.start.date:
                    original_start_date = parser.parse(event_request.start.date).date()
                elif event_request.start.dateTime:
                    original_start_datetime = start_dt
                original_start_timezone = start_tz

            # Validate eventType and birthdayproperties value
            if (
                event_request.eventType != "birthday"
                and event_request.birthdayProperties
            ):
                raise ValueError(
                    "Use birthday properties only when eventType is set to 'birthday'"
                )

            # Validate eventType and focusTimeProperties
            if (
                event_request.eventType != "focusTime"
                and event_request.focusTimeProperties
            ):
                raise ValueError(
                    "Use focusTimeProperties only when eventType is set to 'focusTime'"
                )

            # Validate eventType and outOfOfficeProperties
            if (
                event_request.eventType != "outOfOffice"
                and event_request.outOfOfficeProperties
            ):
                raise ValueError(
                    "Use outOfOfficeProperties only when eventType is set to 'outOfOffice'"
                )

            # Validate maximum number of override reminder
            if event_request.reminders and event_request.reminders.overrides:
                if len(event_request.reminders.overrides) > 5:
                    raise ValueError("The maximum number of override reminders is 5")

            # Validate conferenceId based on conference solution type
            if (
                event_request.conferenceData
                and event_request.conferenceData.conferenceSolution
            ):
                if (
                    event_request.conferenceData.conferenceSolution.key
                    and event_request.conferenceData.conferenceSolution.key.type
                ):
                    if event_request.conferenceData.conferenceSolution.key.type not in [
                        "hangoutsMeet",
                        "addOn",
                    ]:
                        raise ValueError(
                            "Value for type under key field in the conferenceSolution is not valid value. Please provide either 'hangoutsMeet' or 'addOn'"
                        )
                    if (
                        event_request.conferenceData.conferenceSolution.key.type
                        == "hangoutsMeet"
                    ):
                        if event_request.conferenceData.conferenceId:
                            raise ValueError(
                                "ConferenceId will be set internally. Please remove this field"
                            )
                        else:
                            chars = string.ascii_lowercase
                            parts = [
                                "".join(random.choices(chars, k=3)),
                                "".join(random.choices(chars, k=4)),
                                "".join(random.choices(chars, k=3)),
                            ]
                            code = "-".join(parts)

                            event_request.conferenceData.conferenceId = code

                    elif (
                        event_request.conferenceData.conferenceSolution.key.type
                        == "addOn"
                    ):
                        if event_request.conferenceData.conferenceId is None:
                            raise ValueError(
                                "Please provide valid value of conferenceId key for conferenceData or add conferenceId key with valid value if not present"
                            )

            if (
                event_request.conferenceData
                and event_request.conferenceData.createRequest
                and event_request.conferenceData.createRequest.conferenceSolutionKey
            ):
                if (
                    event_request.conferenceData.createRequest.conferenceSolutionKey.type
                    not in ["hangoutsMeet", "addOn"]
                ):
                    raise ValueError(
                        "Value for type under conferenceSolutionKey  inside the createRequest field is not valid value. Please provide either 'hangoutsMeet' or 'addOn'"
                    )

            recurring_event_id = None
            total_occurences = []
            # Validate recurrence if present
            if event_request.recurrence and len(event_request.recurrence) > 0:
                RECURRENCE_LIMIT = 365

                # Ensure start_dt is timezone-aware before passing to parser
                # RecurrenceParser needs timezone-aware datetimes for proper comparison
                recurrence_start_dt = start_dt
                if recurrence_start_dt.tzinfo is None:
                    recurrence_start_dt = recurrence_start_dt.replace(
                        tzinfo=timezone.utc
                    )

                try:
                    # Parse recurrence data with timezone-aware start datetime
                    parsed_recurrence, rset = RecurrenceParser.parse_recurrence_list(
                        event_request.recurrence, recurrence_start_dt
                    )

                    if not parsed_recurrence["rrule"]:
                        raise ValueError("RRULE is required for recurring events")

                    logger.info(f"Recurrence rule passed all the validation tests")

                    # Generate sample occurrences to validate the rule
                    sample_occurrences = list(rset[:RECURRENCE_LIMIT])
                    total_occurences = [dt.isoformat() for dt in sample_occurrences]
                    logger.info(
                        f"Generated {len(total_occurences)} sample occurrences from recurrence rule"
                    )

                    # Create Recurring Event

                    # Generate recurring event ID
                    recurring_event_id = str(uuid.uuid4())

                    # 1. Create a Recurring Event instance
                    rec_event = RecurringEvent(
                        recurring_event_id=recurring_event_id,
                        original_recurrence=json.dumps(event_request.recurrence),
                    )
                    session.add(rec_event)
                    session.commit()

                except RecurrenceParseError as rpe:
                    raise ValueError(f"Invalid recurrence pattern: {rpe}")
                except Exception as e:
                    logger.error(f"Error validating recurrence pattern: {e}")
                    raise ValueError(f"Recurrence validation failed: {e}")

            # Create database event with Google API fields
            db_event = Event(
                event_id=event_id,
                calendar_id=calendar_id,
                user_id=user_id,
                recurring_event_id=recurring_event_id,
                summary=event_request.summary,
                description=event_request.description,
                location=event_request.location,
                start_datetime=start_dt,
                end_datetime=end_dt,
                start_timezone=start_tz,
                end_timezone=end_tz,
                recurrence=json.dumps(event_request.recurrence)
                if event_request.recurrence
                else None,
                status=event_request.status or "confirmed",
                visibility=event_request.visibility or "default",
                sequence=event_request.sequence or 0,
                source=event_request.source,
                # Store originalStartTime fields
                originalStartTime_date=original_start_date,
                originalStartTime_dateTime=original_start_datetime,
                originalStartTime_timeZone=original_start_timezone,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            # Handle optional Google API fields
            if hasattr(event_request, "eventType") and event_request.eventType:
                try:
                    db_event.eventType = EventTypeEnum(event_request.eventType)
                except ValueError:
                    # Default to 'default' if eventType is not supported
                    db_event.eventType = EventTypeEnum.DEFAULT

            if hasattr(event_request, "colorId") and event_request.colorId:
                db_event.color_id = event_request.colorId

            if hasattr(event_request, "transparency") and event_request.transparency:
                db_event.transparency = event_request.transparency

            if hasattr(event_request, "iCalUID") and event_request.iCalUID:
                db_event.iCalUID = event_request.iCalUID
            else:
                db_event.iCalUID = f"{event_id}@calendar.google.com"

            # Handle guest permissions
            if (
                hasattr(event_request, "guestsCanInviteOthers")
                and event_request.guestsCanInviteOthers is not None
            ):
                db_event.guestsCanInviteOthers = event_request.guestsCanInviteOthers
            if (
                hasattr(event_request, "guestsCanModify")
                and event_request.guestsCanModify is not None
            ):
                db_event.guestsCanModify = event_request.guestsCanModify
            if (
                hasattr(event_request, "guestsCanSeeOtherGuests")
                and event_request.guestsCanSeeOtherGuests is not None
            ):
                db_event.guestsCanSeeOtherGuests = event_request.guestsCanSeeOtherGuests

            # Set organizer information - organizer is the user creating the event
            creator_user = session.query(User).filter(User.user_id == user_id).first()
            if creator_user:
                db_event.organizer_id = creator_user.user_id
                db_event.organizer_email = creator_user.email
                db_event.organizer_display_name = creator_user.name
                db_event.organizer_self = True

            session.add(db_event)
            session.flush()  # Get the event ID for related records

            # Handle attendees if provided
            if event_request.attendees:
                self._process_create_attendees(
                    db_event, event_request.attendees, session
                )

            # Handle attachments if provided (only if supported by client)
            if event_request.attachments and (
                query_params is None
                or (
                    query_params.get("supportsAttachments")
                    if query_params.get("supportsAttachments") is not None
                    else True
                )
            ):
                self._process_create_attachments(
                    db_event, event_request.attachments, session
                )

            # Handle conference data if provided (based on version)
            if event_request.conferenceData and (
                query_params is None
                or (query_params.get("conferenceDataVersion") or 0) >= 1
            ):
                try:
                    self._process_create_conference_data(
                        db_event, event_request.conferenceData, session
                    )
                except Exception as e:
                    raise

            # Handle extended properties if provided
            if event_request.extendedProperties:
                self._process_create_extended_properties(
                    db_event, event_request.extendedProperties, session
                )

            # Handle reminders if provided
            if event_request.reminders:
                self._process_create_reminders(
                    db_event, event_request.reminders, session
                )

            # Handle working location properties if provided
            if event_request.workingLocationProperties:
                self._process_create_working_location(
                    db_event, event_request.workingLocationProperties, session
                )

            # Handle type-specific properties
            if (
                event_request.eventType == "birthday"
                and event_request.birthdayProperties
            ):
                self._process_create_birthday_properties(
                    db_event, event_request.birthdayProperties, session
                )
            elif (
                event_request.eventType == "focusTime"
                and event_request.focusTimeProperties
            ):
                # Convert Pydantic model to dict for JSON storage
                if hasattr(event_request.focusTimeProperties, "model_dump"):
                    db_event.focusTimeProperties = (
                        event_request.focusTimeProperties.model_dump()
                    )
                elif hasattr(event_request.focusTimeProperties, "dict"):
                    db_event.focusTimeProperties = (
                        event_request.focusTimeProperties.dict()
                    )
                else:
                    db_event.focusTimeProperties = event_request.focusTimeProperties
            elif (
                event_request.eventType == "outOfOffice"
                and event_request.outOfOfficeProperties
            ):
                # Convert Pydantic model to dict for JSON storage
                if hasattr(event_request.outOfOfficeProperties, "model_dump"):
                    db_event.outOfOfficeProperties = (
                        event_request.outOfOfficeProperties.model_dump()
                    )
                elif hasattr(event_request.outOfOfficeProperties, "dict"):
                    db_event.outOfOfficeProperties = (
                        event_request.outOfOfficeProperties.dict()
                    )
                else:
                    db_event.outOfOfficeProperties = event_request.outOfOfficeProperties

            # Handle source if provided
            if event_request.source:
                db_event.source = event_request.source

            session.commit()
            session.refresh(db_event)

            # Handle notifications based on sendUpdates parameter
            try:
                send_updates = query_params.get("sendUpdates") if query_params else None
                # Set default to 'none' if not provided or is None
                if send_updates is None:
                    send_updates = "none"
                self._handle_create_notifications(db_event, send_updates, session)
            except Exception as e:
                logger.info(f"Error sending notifications: {str(e)}")
                pass

            # Generate Event instance based on total_occurence
            if event_request.recurrence and len(event_request.recurrence) > 0:
                # Ensure both datetimes have the same timezone awareness for duration calculation
                duration_start_dt = start_dt
                duration_end_dt = end_dt

                # Handle timezone awareness consistency for duration calculation
                if duration_start_dt.tzinfo is None and duration_end_dt.tzinfo is None:
                    # Both are naive (all-day events) - this is fine
                    pass
                elif (
                    duration_start_dt.tzinfo is None
                    and duration_end_dt.tzinfo is not None
                ):
                    # start is naive, end is aware - make start aware
                    duration_start_dt = duration_start_dt.replace(tzinfo=timezone.utc)
                elif (
                    duration_start_dt.tzinfo is not None
                    and duration_end_dt.tzinfo is None
                ):
                    # start is aware, end is naive - make end aware
                    duration_end_dt = duration_end_dt.replace(tzinfo=timezone.utc)

                duration = duration_end_dt - duration_start_dt
                datetime_dict = {
                    "duration": duration,
                    "start_timezone": start_tz,
                    "end_timezone": end_tz,
                    "iCalUID": db_event.iCalUID,
                }
                self.create_recurrence_event_instance(
                    session,
                    user_id,
                    recurring_event_id,
                    calendar_id,
                    event_request,
                    query_params,
                    total_occurences,
                    datetime_dict,
                )

            logger.info(f"Created event {event_id} in calendar {calendar_id}")
            return self._convert_db_event_to_schema(db_event)

        except RecurrenceParseError as rperr:
            raise ValueError(f"{str(rperr)}")
        except ValueError as verr:
            session.rollback()
            raise ValueError(f"{str(verr)}")
        except Exception as e:
            logger.error(f"Error creating event in calendar {calendar_id}: {e}")
            raise
        finally:
            session.close()

    def get_event(
        self,
        user_id: str,
        calendar_id: str,
        event_id: str,
        timeZone: str = None,
        maxAttendees: Optional[int] = None,
    ) -> Optional[EventSchema]:
        """
        Get a specific event

        GET /calendars/{calendarId}/events/{eventId}
        """
        session = get_session(self.database_id)
        try:
            # Verify calendar belongs to user
            if not timeZone or not timeZone.strip():
                calendar = (
                    session.query(Calendar)
                    .filter(
                        Calendar.calendar_id == calendar_id, Calendar.user_id == user_id
                    )
                    .first()
                )

            else:
                calendar = (
                    session.query(Calendar)
                    .filter(
                        Calendar.calendar_id == calendar_id,
                        Calendar.user_id == user_id,
                        Calendar.time_zone == timeZone,
                    )
                    .first()
                )

            if not calendar:
                raise ValueError(
                    f"Calendar {calendar_id} not found for user {user_id} in the timezone {timeZone}"
                )
            db_event = (
                session.query(Event)
                .options(
                    joinedload(Event.attendees).joinedload(Attendees.user),
                    joinedload(Event.attachments),
                    joinedload(Event.reminders),
                )
                .filter(
                    and_(
                        Event.calendar_id == calendar_id,
                        Event.event_id == event_id,
                        Event.user_id == user_id,
                    )
                )
                .first()
            )

            if not db_event:
                return None

            event_schema = self._convert_db_event_to_schema(db_event)

            # Filter attendees based on maxAttendees parameter
            if (
                maxAttendees is not None
                and event_schema.attendees
                and len(event_schema.attendees) > maxAttendees
            ):
                event_schema.attendees = event_schema.attendees[:maxAttendees]
                event_schema.attendeesOmitted = True

            return event_schema

        except Exception as e:
            logger.error(
                f"Error getting event {event_id} from calendar {calendar_id}: {e}"
            )
            raise
        finally:
            session.close()

    def _validate_email_pattern(self, email: str) -> bool:
        """Validate email pattern using regex"""
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return re.match(email_pattern, email) is not None

    def _create_user_from_email(self, session, email: str) -> User:
        """Create a new user from email address"""
        try:
            # Generate user data from email
            name = email.split("@")[0]  # Use email prefix as default name
            user_data = {
                "user_id": str(uuid.uuid4()),
                "email": email,
                "name": name,
                "static_token": str(uuid.uuid4()),  # Required field for User model
            }

            # Create new user instance
            user = User(
                user_id=user_data["user_id"],
                email=user_data["email"],
                name=user_data["name"],
                static_token=user_data["static_token"],
                is_active=True,
                is_verified=False,
                timezone="UTC",
            )

            session.add(user)
            session.commit()
            logger.info(f"Created new user with email: {email}")
            return user

        except Exception as e:
            logger.error(f"Error creating user from email {email}: {e}")
            raise ValueError(f"Failed to create user for email {email}: {e}")

    def check_attendees_email_id(self, session, value):
        for v in value:
            # Handle both dictionary and Pydantic model formats
            if isinstance(v, dict):
                email_id = v.get("email")
            else:
                # Assume it's a Pydantic model with .email attribute
                email_id = getattr(v, "email", None)

            if not email_id:
                raise ValueError("Email address is required for all attendees")

            user = session.query(User).filter(User.email == email_id).first()
            if user is None:
                # Validate email pattern
                if not self._validate_email_pattern(email_id):
                    raise ValueError(f"Invalid email format: {email_id}")

                # Create new user based on email
                try:
                    user = self._create_user_from_email(session, email_id)
                except Exception as e:
                    raise ValueError(
                        f"Failed to create attendees for email {email_id}: {e}"
                    )

    def update_event(
        self,
        user_id: str,
        calendar_id: str,
        event_id: str,
        event_request: EventUpdateRequest,
        is_patch: bool = True,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[EventSchema]:
        """
        Update an event (PATCH or PUT)

        PATCH /calendars/{calendarId}/events/{eventId}
        PUT /calendars/{calendarId}/events/{eventId}
        """
        session = get_session(self.database_id)
        try:
            # Set default query_params if not provided
            if query_params is None:
                query_params = {}

            # Verify calendar belongs to user
            calendar = (
                session.query(Calendar)
                .filter(
                    Calendar.calendar_id == calendar_id, Calendar.user_id == user_id
                )
                .first()
            )
            if not calendar:
                raise ValueError(f"Calendar {calendar_id} not found for user {user_id}")

            db_event = (
                session.query(Event)
                .options(
                    joinedload(Event.attendees).joinedload(Attendees.user),
                    joinedload(Event.attachments),
                )
                .filter(
                    and_(
                        Event.calendar_id == calendar_id,
                        Event.event_id == event_id,
                        Event.user_id == user_id,
                    )
                )
                .first()
            )

            if not db_event:
                return None

            # Update fields based on request
            update_data = event_request.model_dump(exclude_none=is_patch)

            # Validation to check the eventType is not getting changed
            if (
                "eventType" in update_data.keys()
                and update_data.get("eventType") is not None
            ):
                if db_event.eventType.value != update_data.get("eventType").value:
                    raise ValueError(
                        f"EventType once set can not be modified. Expecting value: {db_event.eventType.value}. Got {update_data.get('eventType').value}"
                    )

            for field, value in update_data.items():
                if field == "start" and value:
                    # Handle start DateTime object with timezone
                    start_dt, start_tz = self._parse_datetime_from_api_format(value)
                    db_event.start_datetime = start_dt
                    db_event.start_timezone = start_tz

                    # Validate datetime consistency if end is also being updated
                    if hasattr(event_request, "end") and event_request.end:
                        end_dt, _ = self._parse_datetime_from_api_format(
                            event_request.end
                        )
                        if end_dt <= start_dt:
                            raise ValueError("Event end time must be after start time")
                elif field == "end" and value:
                    # Handle end DateTime object with timezone
                    end_dt, end_tz = self._parse_datetime_from_api_format(value)
                    db_event.end_datetime = end_dt
                    db_event.end_timezone = end_tz

                    # Validate datetime consistency if start is also being updated
                    if hasattr(event_request, "start") and event_request.start:
                        start_dt, _ = self._parse_datetime_from_api_format(
                            event_request.start
                        )
                        if end_dt <= start_dt:
                            raise ValueError("Event end time must be after start time")
                    # Also validate against existing start time if start is not being updated
                    elif end_dt <= db_event.start_datetime:
                        raise ValueError("Event end time must be after start time")
                elif field == "colorId" and value:
                    db_event.color_id = value
                elif field == "attendees" and value:
                    # Check whether email id exist
                    self.check_attendees_email_id(session, value)

                    # Remove all the attendees before setting
                    attendees = (
                        session.query(Attendees)
                        .filter(Attendees.event_id == event_id)
                        .all()
                    )
                    for att in attendees:
                        session.delete(att)
                    session.commit()

                    for v in value:
                        # Handle both dictionary and Pydantic model formats consistently
                        if isinstance(v, dict):
                            email = v.get("email")
                            display_name = v.get("displayName")
                            response_status = v.get("responseStatus", "needsAction")
                            comment = v.get("comment")
                            additional_guests = v.get("additionalGuests", 0)
                            resource = v.get("resource", False)
                            optional = v.get("optional", False)
                        else:
                            # Assume it's a Pydantic model
                            email = getattr(v, "email", None)
                            display_name = getattr(v, "displayName", None)
                            response_status = getattr(
                                v, "responseStatus", "needsAction"
                            )
                            comment = getattr(v, "comment", None)
                            additional_guests = getattr(v, "additionalGuests", 0)
                            resource = getattr(v, "resource", False)
                            optional = getattr(v, "optional", False)

                        if not email:
                            continue

                        user = session.query(User).filter(User.email == email).first()
                        if user is None:
                            continue

                        att_obj = (
                            session.query(Attendees)
                            .filter(
                                and_(
                                    Attendees.event_id == event_id,
                                    Attendees.user_id == user.user_id,
                                )
                            )
                            .first()
                        )
                        if att_obj is None:
                            att_obj = Attendees(
                                attendees_id=str(uuid.uuid4()),
                                event_id=event_id,
                                user_id=user.user_id,
                            )
                            session.add(att_obj)
                            session.flush()

                        # Update attendee attributes
                        att_obj.responseStatus = response_status
                        att_obj.comment = comment
                        att_obj.displayName = display_name
                        att_obj.additionalGuests = additional_guests
                        att_obj.resource = resource
                        att_obj.optional = optional

                        session.commit()
                elif field == "recurrence" and value:
                    if isinstance(value, list):
                        db_event.recurrence = json.dumps(value)
                    else:
                        db_event.recurrence = value
                elif field == "attachments" and value:
                    # Handle attachments (only if supported by client)
                    if query_params.get("supportsAttachments", True):
                        # Clear existing attachments first
                        from database.models.event import Attachment

                        session.query(Attachment).filter(
                            Attachment.event_id == event_id
                        ).delete()
                        session.flush()

                        # Process new attachments
                        self._process_create_attachments(db_event, value, session)
                elif field == "conferenceData" and value and value != {}:
                    # Handle conference data (based on version)
                    conference_version = query_params.get("conferenceDataVersion", 0)
                    if (
                        conference_version
                        and conference_version >= 1
                        and value
                        and value != {}
                    ):
                        # Clear existing conference data first
                        from database.models.event import (
                            ConferenceData as DBConferenceData,
                        )

                        session.query(DBConferenceData).filter(
                            DBConferenceData.event_id == event_id
                        ).delete()
                        session.flush()

                        # Process new conference data - ensure it's treated as dictionary
                        if not isinstance(value, dict):
                            # Convert Pydantic model to dict if needed
                            if hasattr(value, "model_dump"):
                                value = value.model_dump()
                            elif hasattr(value, "dict"):
                                value = value.dict()
                            else:
                                # Try to convert to dict manually
                                value = {
                                    k: getattr(value, k)
                                    for k in dir(value)
                                    if not k.startswith("_")
                                    and not callable(getattr(value, k))
                                }

                        # Process new conference data
                        self._process_create_conference_data(db_event, value, session)
                elif field == "birthdayProperties" and value and value != {}:
                    # Handle birthday properties with type immutability check
                    existing_birthday = None
                    if (
                        hasattr(db_event, "birthdayProperties")
                        and db_event.birthdayProperties
                    ):
                        existing_birthday = db_event.birthdayProperties

                    # Get birthday type from value
                    birthday_type = (
                        value.get("type", "birthday")
                        if isinstance(value, dict)
                        else getattr(value, "type", "birthday")
                    )

                    # Validate type immutability - birthday event type cannot be changed
                    if existing_birthday and birthday_type != existing_birthday.type:
                        raise ValueError(
                            f"Birthday event type cannot be changed. Current type: {existing_birthday.type}"
                        )

                    # For new birthday properties, validate type is "birthday"
                    if birthday_type != "birthday":
                        raise ValueError(
                            f"Invalid birthday properties type: {birthday_type}. Must be 'birthday'."
                        )

                    # If no existing birthday properties and we have valid data, create new ones
                    if not existing_birthday:
                        self._process_create_birthday_properties(
                            db_event, value, session
                        )
                elif field == "extendedProperties" and value and value != {}:
                    # Handle extended properties
                    from database.models.event import ExtendedProperty

                    # Clear existing extended properties first
                    session.query(ExtendedProperty).filter(
                        ExtendedProperty.event_id == event_id
                    ).delete()
                    session.flush()

                    # Ensure value is a dictionary
                    if not isinstance(value, dict):
                        if hasattr(value, "model_dump"):
                            value = value.model_dump()
                        elif hasattr(value, "dict"):
                            value = value.dict()

                    # Process new extended properties
                    self._process_create_extended_properties(db_event, value, session)
                elif field == "reminders" and value and value != {}:
                    # Handle reminders - only if not empty dict
                    # Ensure value is in the right format
                    processed_value = value
                    if not isinstance(value, dict):
                        if hasattr(value, "model_dump"):
                            processed_value = value.model_dump()
                        elif hasattr(value, "dict"):
                            processed_value = value.dict()

                    # Check if it has the expected reminder structure
                    if (
                        hasattr(processed_value, "useDefault")
                        or hasattr(processed_value, "overrides")
                        or (
                            isinstance(processed_value, dict)
                            and (
                                "useDefault" in processed_value
                                or "overrides" in processed_value
                            )
                        )
                    ):
                        # Clear existing reminders first

                        session.query(Reminder).filter(
                            Reminder.event_id == event_id
                        ).delete()
                        session.flush()

                        # Process new reminders
                        self._process_create_reminders_for_update(
                            db_event, processed_value, session
                        )
                elif field in [
                    "guestsCanInviteOthers",
                    "guestsCanModify",
                    "guestsCanSeeOtherGuests",
                ]:
                    # Only update Boolean fields if value is not None to avoid null constraint violations
                    if value is not None:
                        setattr(db_event, field, value)
                elif (
                    field
                    in [
                        "focusTimeProperties",
                        "outOfOfficeProperties",
                        "workingLocationProperties",
                        "source",
                    ]
                    and value
                    and value != {}
                ):
                    # Handle other object fields - only if not empty dict
                    if hasattr(db_event, field):
                        # Convert Pydantic models to dict for JSON storage
                        if field in ["focusTimeProperties", "outOfOfficeProperties"]:
                            try:
                                # Convert Pydantic model to dict
                                if hasattr(value, "model_dump"):
                                    processed_value = value.model_dump()
                                elif hasattr(value, "dict"):
                                    processed_value = value.dict()
                                setattr(db_event, field, processed_value)
                            except:
                                pass
                        elif field == "workingLocationProperties":
                            # Handle working location properties separately to avoid issues
                            if hasattr(value, "model_dump"):
                                working_location_data = value.model_dump()
                            elif hasattr(value, "dict"):
                                working_location_data = value.dict()
                            else:
                                working_location_data = value

                            # Clear existing working location properties first
                            from database.models.event import WorkingLocationProperties

                            session.query(WorkingLocationProperties).filter(
                                WorkingLocationProperties.event_id == event_id
                            ).delete()
                            session.flush()

                            # Process new working location properties
                            self._process_create_working_location(
                                db_event, working_location_data, session
                            )
                            continue

                elif hasattr(db_event, field) and value is not None and value != {}:
                    # Only set attribute if value is not None and not empty dict to avoid null constraint violations
                    setattr(db_event, field, value)
            # Increment event versioning
            try:
                db_event.sequence += 1
            except:
                db_event.sequence = 0
            db_event.updated_at = datetime.now(timezone.utc)

            session.commit()

            # Reload the event with all relationships for the response
            db_event_with_relations = (
                session.query(Event)
                .options(
                    joinedload(Event.attendees).joinedload(Attendees.user),
                    joinedload(Event.attachments),
                )
                .filter(
                    and_(
                        Event.calendar_id == calendar_id,
                        Event.event_id == event_id,
                        Event.user_id == user_id,
                    )
                )
                .first()
            )

            logger.info(f"Updated event {event_id} in calendar {calendar_id}")
            logger.info(
                f"Event has {len(db_event_with_relations.attendees) if db_event_with_relations.attendees else 0} attendees"
            )
            logger.info(
                f"Event has {len(db_event_with_relations.attachments) if db_event_with_relations.attachments else 0} attachments"
            )

            return self._convert_db_event_to_schema(db_event_with_relations)

        except Exception as e:
            session.rollback()
            logger.error(
                f"Error updating event {event_id} in calendar {calendar_id}: {e}"
            )
            raise
        finally:
            session.close()

    def delete_event(
        self,
        user_id: str,
        calendar_id: str,
        event_id: str,
        send_updates: Optional[str] = "all",
    ) -> bool:
        """
        Delete an event with notification control

        Args:
            user_id: User performing the deletion
            calendar_id: Calendar containing the event
            event_id: Event to delete
            send_updates: Who should receive notifications:
                - "all": Notifications sent to all guests
                - "externalOnly": Notifications sent to non-Google Calendar guests only
                - "none": No notifications sent

        DELETE /calendars/{calendarId}/events/{eventId}
        """
        session = get_session(self.database_id)
        try:
            # Verify calendar belongs to user
            calendar = (
                session.query(Calendar)
                .filter(
                    Calendar.calendar_id == calendar_id, Calendar.user_id == user_id
                )
                .first()
            )
            if not calendar:
                raise ValueError(f"Calendar {calendar_id} not found for user {user_id}")

            db_event = (
                session.query(Event)
                .filter(
                    and_(
                        Event.calendar_id == calendar_id,
                        Event.event_id == event_id,
                        Event.user_id == user_id,
                    )
                )
                .first()
            )

            if not db_event:
                return False

            try:
                # Handle notification logic based on sendUpdates parameter
                self._handle_notifications(db_event, send_updates, session)
            except Exception as e:
                logger.info(f"Error sending notifications: {str(e)}")
                pass

            session.delete(db_event)
            session.commit()

            logger.info(f"Deleted event {event_id} from calendar {calendar_id}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(
                f"Error deleting event {event_id} from calendar {calendar_id}: {e}"
            )
            raise
        finally:
            session.close()

    def move_event(
        self,
        user_id: str,
        calendar_id: str,
        event_id: str,
        move_request: EventMoveRequest,
    ) -> Optional[EventSchema]:
        """
        Move an event to another calendar

        POST /calendars/{calendarId}/events/{eventId}/move
        """
        session = get_session(self.database_id)
        try:
            # Verify source calendar belongs to user
            source_calendar = (
                session.query(Calendar)
                .filter(
                    Calendar.calendar_id == calendar_id, Calendar.user_id == user_id
                )
                .first()
            )
            if not source_calendar:
                raise ValueError(
                    f"Source calendar {calendar_id} not found for user {user_id}"
                )

            db_event = (
                session.query(Event)
                .filter(
                    and_(
                        Event.calendar_id == calendar_id,
                        Event.event_id == event_id,
                        Event.user_id == user_id,
                    )
                )
                .first()
            )

            if not db_event:
                return None

            # Check if destination calendar exists and belongs to user
            dest_calendar = (
                session.query(Calendar)
                .filter(
                    Calendar.calendar_id == move_request.destination,
                    Calendar.user_id == user_id,
                )
                .first()
            )

            if not dest_calendar:
                raise ValueError(
                    f"Destination calendar {move_request.destination} not found for user {user_id}"
                )

            # Move event to destination calendar
            db_event.calendar_id = move_request.destination
            db_event.updated_at = datetime.now(timezone.utc)

            session.commit()
            session.refresh(db_event)

            try:
                # Handle notification logic based on sendUpdates parameter
                self._handle_notifications(db_event, move_request.sendUpdates, session)
            except Exception as e:
                logger.info(f"Error sending notifications: {str(e)}")
                pass

            logger.info(
                f"Moved event {event_id} from {calendar_id} to {move_request.destination}"
            )
            return self._convert_db_event_to_schema(db_event)

        except Exception as e:
            session.rollback()
            logger.error(
                f"Error moving event {event_id} from calendar {calendar_id}: {e}"
            )
            raise
        finally:
            session.close()

    def quick_add_event(
        self, user_id: str, calendar_id: str, quick_add_request: EventQuickAddRequest
    ) -> EventSchema:
        """
        Quick add an event using natural language

        POST /calendars/{calendarId}/events/quickAdd
        """
        session = get_session(self.database_id)
        try:
            # Verify calendar belongs to user
            calendar = (
                session.query(Calendar)
                .filter(
                    Calendar.calendar_id == calendar_id, Calendar.user_id == user_id
                )
                .first()
            )
            if not calendar:
                raise ValueError(f"Calendar {calendar_id} not found for user {user_id}")

            # Simple parsing for quick add (in real implementation, use NLP)
            text = quick_add_request.text

            # Basic parsing - just create event with text as summary
            # In real implementation, parse text for date/time/location
            import uuid

            event_id = str(uuid.uuid4())

            # Use current time + 1 hour as default
            now = datetime.now(timezone.utc)
            start_dt = now.replace(minute=0, second=0, microsecond=0)
            end_dt = start_dt.replace(hour=start_dt.hour + 1)

            db_event = Event(
                event_id=event_id,
                calendar_id=calendar_id,
                user_id=user_id,
                summary=text,
                start_datetime=start_dt,
                end_datetime=end_dt,
                status="confirmed",
                visibility="default",
                created_at=now,
                updated_at=now,
            )

            # Set organizer information for quick add events
            creator_user = session.query(User).filter(User.user_id == user_id).first()
            if creator_user:
                db_event.organizer_id = creator_user.user_id
                db_event.organizer_email = creator_user.email
                db_event.organizer_display_name = creator_user.name
                db_event.organizer_self = True

            session.add(db_event)
            session.commit()
            session.refresh(db_event)

            try:
                # Handle notification logic based on sendUpdates parameter
                self._handle_notifications(
                    db_event, quick_add_request.sendUpdates, session
                )
            except Exception as e:
                logger.info(f"Error sending notifications: {str(e)}")
                pass

            logger.info(f"Quick added event {event_id} in calendar {calendar_id}")
            return self._convert_db_event_to_schema(db_event)

        except Exception as e:
            session.rollback()
            logger.error(f"Error quick adding event to calendar {calendar_id}: {e}")
            raise
        finally:
            session.close()

    def import_event(
        self,
        user_id: str,
        calendar_id: str,
        event_request: EventImportRequest,
        query_params: EventImportQueryParams,
    ) -> EventImportResponse:
        """
        Import an event as a private copy to the specified calendar.
        This operation is used to add a private copy of an existing event to a calendar.

        Features:
        - Validates event data and user permissions
        - Handles event type conversion (non-default types may be converted to default)
        - Manages iCalUID uniqueness
        - Processes attachments and conference data based on client capabilities
        - Creates a complete private copy with proper metadata

        POST /calendars/{calendarId}/events/import
        """
        session = get_session(self.database_id)
        try:
            # Step 1: Validate calendar access
            calendar = (
                session.query(Calendar)
                .filter(
                    Calendar.calendar_id == calendar_id, Calendar.user_id == user_id
                )
                .first()
            )
            if not calendar:
                raise ValueError(f"Calendar {calendar_id} not found for user {user_id}")

            # Step 2: Validate import request
            validation_result = self._validate_import_request(
                user_id, calendar_id, event_request, session
            )
            if not validation_result.success:
                raise ValueError(validation_result.error.message)

            # Check for existing iCalUID conflicts
            existing_event = (
                session.query(Event)
                .filter(
                    Event.user_id == user_id,
                    Event.calendar_id == calendar_id,
                    Event.iCalUID == event_request.iCalUID,
                )
                .first()
            )

            if existing_event is None:
                raise ValueError(
                    "Event does not exist with this calendar_id and iCalUID"
                )
            if existing_event.eventType.value != "default":
                raise ValueError("Only event with eventType 'default' can be imported")

            # Step 4: Generate event ID and handle iCalUID

            event_id = str(uuid.uuid4())

            # Step 5: Parse and validate datetime fields
            start_date, start_dt, end_date, end_dt, start_tz, end_tz = (
                self._parse_import_datetimes(event_request)
            )

            # Validate originalStartTime fields if provided and prepare values for storage
            original_start_date = None
            original_start_datetime = None
            original_start_timezone = None

            if event_request.originalStartTime:
                # Validate that originalStartTime matches start field values
                if event_request.start.date and event_request.originalStartTime.date:
                    # Both are all-day events - dates must match
                    if event_request.start.date != event_request.originalStartTime.date:
                        raise ValueError(
                            "originalStartTime.date must match start.date when both are provided"
                        )
                    original_start_date = parser.parse(
                        event_request.originalStartTime.date
                    ).date()
                elif (
                    event_request.start.dateTime
                    and event_request.originalStartTime.dateTime
                ):
                    # Both are timed events - dateTime must match
                    if (
                        event_request.start.dateTime
                        != event_request.originalStartTime.dateTime
                    ):
                        raise ValueError(
                            "originalStartTime.dateTime must match start.dateTime when both are provided"
                        )
                    original_start_datetime = self._parse_datetime_string(
                        event_request.originalStartTime.dateTime
                    )
                elif (
                    event_request.start.date
                    and event_request.originalStartTime.dateTime
                ) or (
                    event_request.start.dateTime
                    and event_request.originalStartTime.date
                ):
                    # Mismatched types (one is all-day, other is timed)
                    raise ValueError(
                        "originalStartTime and start must both be either all-day (date) or timed (dateTime) events"
                    )

                # Validate timezone matches
                if (
                    event_request.start.timeZone
                    and event_request.originalStartTime.timeZone
                ):
                    if (
                        event_request.start.timeZone
                        != event_request.originalStartTime.timeZone
                    ):
                        raise ValueError(
                            "originalStartTime.timeZone must match start.timeZone when both are provided"
                        )

                # Store originalStartTime timezone
                original_start_timezone = event_request.originalStartTime.timeZone
            else:
                # If originalStartTime not provided, use start field values
                if event_request.start.date:
                    original_start_date = parser.parse(event_request.start.date).date()
                elif event_request.start.dateTime:
                    original_start_datetime = start_dt
                original_start_timezone = start_tz

            # Validate eventType and focusTimeProperties
            if event_request.focusTimeProperties:
                raise ValueError(
                    "focusTimeProperties is not required as only 'default' eventType can be imported"
                )

            # Validate eventType and outOfOfficeProperties
            if event_request.outOfOfficeProperties:
                raise ValueError(
                    "outOfOfficeProperties is not required as only 'default' eventType can be imported"
                )

            # Validate maximum number of override reminder
            if event_request.reminders and event_request.reminders.overrides:
                if len(event_request.reminders.overrides) > 5:
                    raise ValueError("The maximum number of override reminders is 5")

            # Validate organizer email and displayName
            if event_request.organizer:
                # Find or create user for organizer
                organizer_user = (
                    session.query(User)
                    .filter(User.email == event_request.organizer.email)
                    .first()
                )
                if organizer_user is None:
                    raise ValueError("Organizer email does not exist")
                else:
                    organizer_display_name = (
                        session.query(User)
                        .filter(
                            User.email == event_request.organizer.email,
                            User.name == event_request.organizer.displayName,
                        )
                        .first()
                    )
                    if organizer_display_name is None:
                        raise ValueError(
                            "Please enter valid name in the organizer displayName"
                        )

            # Validate conferenceId based on conference solution type
            if (
                event_request.conferenceData
                and event_request.conferenceData.conferenceSolution
            ):
                if (
                    event_request.conferenceData.conferenceSolution.key
                    and event_request.conferenceData.conferenceSolution.key.type
                ):
                    if event_request.conferenceData.conferenceSolution.key.type not in [
                        "hangoutsMeet",
                        "addOn",
                    ]:
                        raise ValueError(
                            "Value for type under key field in the conferenceSolution is not valid value. Please provide either 'hangoutsMeet' or 'addOn'"
                        )
                    if (
                        event_request.conferenceData.conferenceSolution.key.type
                        == "hangoutsMeet"
                    ):
                        if event_request.conferenceData.conferenceId:
                            raise ValueError(
                                "ConferenceId will be set internally. Please remove this field"
                            )
                        else:
                            chars = string.ascii_lowercase
                            parts = [
                                "".join(random.choices(chars, k=3)),
                                "".join(random.choices(chars, k=4)),
                                "".join(random.choices(chars, k=3)),
                            ]
                            code = "-".join(parts)

                            event_request.conferenceData.conferenceId = code

                    elif (
                        event_request.conferenceData.conferenceSolution.key.type
                        == "addOn"
                    ):
                        if event_request.conferenceData.conferenceId is None:
                            raise ValueError(
                                "Please provide valid value of conferenceId key for conferenceData or add conferenceId key with valid value if not present"
                            )

            # Check if this is an all-day event
            is_all_day = self._is_all_day_event(start_date, end_date)

            # Step 6: Create the imported event as private copy
            db_event = Event(
                event_id=event_id,
                calendar_id=calendar_id,
                user_id=user_id,
                summary=event_request.summary,
                description=event_request.description,
                location=event_request.location,
                start_datetime=start_dt,
                end_datetime=end_dt,
                start_timezone=start_tz,
                end_timezone=end_tz,
                iCalUID=event_request.iCalUID,
                status=event_request.status or "confirmed",
                visibility=event_request.visibility or "default",
                sequence=event_request.sequence or 0,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            # Handle recurrence
            if event_request.recurrence:
                db_event.recurrence = json.dumps(event_request.recurrence)

            # Handle guest permissions
            if (
                hasattr(event_request, "guestsCanInviteOthers")
                and event_request.guestsCanInviteOthers is not None
            ):
                db_event.guestsCanInviteOthers = event_request.guestsCanInviteOthers
            if (
                hasattr(event_request, "guestsCanModify")
                and event_request.guestsCanModify is not None
            ):
                db_event.guestsCanModify = event_request.guestsCanModify
            if (
                hasattr(event_request, "guestsCanSeeOtherGuests")
                and event_request.guestsCanSeeOtherGuests is not None
            ):
                db_event.guestsCanSeeOtherGuests = event_request.guestsCanSeeOtherGuests

            # Set organizer information based on import request or default to creator
            if event_request.organizer:
                # Find organizer user
                organizer_user = (
                    session.query(User)
                    .filter(User.email == event_request.organizer.email)
                    .first()
                )
                if organizer_user:
                    # Set organizer fields from the provided organizer
                    db_event.organizer_id = organizer_user.user_id
                    db_event.organizer_email = organizer_user.email
                    db_event.organizer_display_name = (
                        event_request.organizer.displayName or organizer_user.name
                    )
                    # Check if organizer is the current user
                    db_event.organizer_self = organizer_user.user_id == user_id

                    # Update event's user_id to organizer (import allows organizer change)
                    db_event.user_id = organizer_user.user_id
                else:
                    logger.warning(
                        f"Organizer email '{event_request.organizer.email}' not found in database"
                    )
                    # Default to current user as organizer
                    creator_user = (
                        session.query(User).filter(User.user_id == user_id).first()
                    )
                    if creator_user:
                        db_event.organizer_id = creator_user.user_id
                        db_event.organizer_email = creator_user.email
                        db_event.organizer_display_name = creator_user.name
                        db_event.organizer_self = True
            else:
                # Default to current user as organizer
                creator_user = (
                    session.query(User).filter(User.user_id == user_id).first()
                )
                if creator_user:
                    db_event.organizer_id = creator_user.user_id
                    db_event.organizer_email = creator_user.email
                    db_event.organizer_display_name = creator_user.name
                    db_event.organizer_self = True

            session.add(db_event)
            session.flush()  # Get the event ID for related records

            # Step 8: Handle attendees
            self._process_import_attendees(
                db_event, event_request.attendees, session, event_request.organizer
            )

            # Step 9: Handle attachments (if supported)
            if (
                query_params.supportsAttachments
                if query_params.supportsAttachments is not None
                else True
            ) and event_request.attachments:
                self._process_import_attachments(
                    db_event, event_request.attachments, session
                )

            # Step 10: Handle conference data (based on version)
            try:
                if (
                    event_request.conferenceData
                    and (query_params.conferenceDataVersion or 0) >= 1
                ):
                    self._process_create_conference_data(
                        db_event, event_request.conferenceData, session
                    )
            except Exception as e:
                logger.warning(f"Error while creating conference data{e}")

            # Step 11: Handle extended properties
            if event_request.extendedProperties:
                self._process_import_extended_properties(
                    db_event, event_request.extendedProperties, session
                )

            # Step 12: Handle reminders
            if event_request.reminders:
                self._process_import_reminders(
                    db_event, event_request.reminders, session
                )

            # Step 13: Handle source
            if event_request.source:
                db_event.source = event_request.source.model_dump()

            session.commit()
            session.refresh(db_event)

            # Step 15: Build import response
            response = self._convert_db_event_to_schema(db_event)

            # Handle attendeesOmitted from import request
            if (
                hasattr(event_request, "attendeesOmitted")
                and event_request.attendeesOmitted
            ):
                response.attendeesOmitted = True

            logger.info(
                f"Successfully imported event {event_id} to calendar {calendar_id}"
            )
            return response

        except Exception as e:
            session.rollback()
            logger.error(f"Error importing event to calendar {calendar_id}: {e}")
            raise
        finally:
            session.close()

    def _validate_import_request(
        self, user_id: str, calendar_id: str, event_request: EventImportRequest, session
    ) -> EventImportResult:
        """Validate import request and permissions"""
        try:
            # Validate required fields
            if not event_request.start or not event_request.end:
                return EventImportResult(
                    success=False,
                    error=EventImportError(
                        domain="calendar",
                        reason="required",
                        message="Both 'start' and 'end' fields are required",
                    ),
                )

            # Validate datetime consistency
            if event_request.start.dateTime and event_request.end.dateTime:
                start_dt = self._parse_datetime_string(event_request.start.dateTime)
                end_dt = self._parse_datetime_string(event_request.end.dateTime)
                if end_dt <= start_dt:
                    return EventImportResult(
                        success=False,
                        error=EventImportError(
                            domain="calendar",
                            reason="invalid",
                            message="Event end time must be after start time",
                        ),
                    )

            return EventImportResult(success=True)

        except Exception as e:
            return EventImportResult(
                success=False,
                error=EventImportError(
                    domain="calendar",
                    reason="invalid",
                    message=f"Validation error: {str(e)}",
                ),
            )

    def _handle_event_type_conversion(
        self, original_type: str, event_request: EventImportRequest
    ) -> tuple[str, list[str]]:
        """Handle event type conversion and return warnings"""
        warnings = []

        # Check if event type is supported
        supported_types = [
            "default",
            "birthday",
            "focusTime",
            "fromGmail",
            "outOfOffice",
            "workingLocation",
        ]

        if original_type not in supported_types:
            warnings.append(
                f"Unsupported event type '{original_type}' converted to 'default'"
            )
            return "default", warnings

        # Check if type-specific properties are present
        if (
            original_type == "workingLocation"
            and not event_request.workingLocationProperties
        ):
            warnings.append(
                "workingLocation event type requires workingLocationProperties"
            )
            return "default", warnings

        if original_type == "focusTime" and not event_request.focusTimeProperties:
            warnings.append("focusTime event type requires focusTimeProperties")
            return "default", warnings

        if original_type == "outOfOffice" and not event_request.outOfOfficeProperties:
            warnings.append("outOfOffice event type requires outOfOfficeProperties")
            return "default", warnings

        return original_type, warnings

    def _parse_import_datetimes(
        self, event_request: EventImportRequest
    ) -> tuple[str, datetime, str, datetime, str, str]:
        """Parse start and end datetime from import request with proper all-day event handling"""
        from datetime import date

        # Handle start datetime
        start_date = None
        if event_request.start.dateTime:
            start_dt = self._parse_datetime_string(event_request.start.dateTime)
            start_tz = self._validate_timezone(event_request.start.timeZone or "UTC")
        elif event_request.start.date:
            # All-day event - parse date string and create datetime at midnight UTC
            try:
                start_date = event_request.start.date
                parsed_date = date.fromisoformat(start_date)
                start_dt = datetime.combine(parsed_date, datetime.min.time())
                start_tz = self._validate_timezone(
                    event_request.start.timeZone or "UTC"
                )
            except ValueError as e:
                raise ValueError(
                    f"Invalid start date format '{event_request.start.date}': {e}"
                )
        else:
            raise ValueError("Event start must have either dateTime or date")

        # Handle end datetime
        end_date = None
        if event_request.end.dateTime:
            end_dt = self._parse_datetime_string(event_request.end.dateTime)
            end_tz = self._validate_timezone(event_request.end.timeZone or "UTC")
        elif event_request.end.date:
            # All-day event - parse date string and create datetime at midnight UTC
            try:
                end_date = event_request.end.date
                parsed_date = date.fromisoformat(end_date)
                end_dt = datetime.combine(parsed_date, datetime.min.time())
                end_tz = self._validate_timezone(event_request.end.timeZone or "UTC")
            except ValueError as e:
                raise ValueError(
                    f"Invalid end date format '{event_request.end.date}': {e}"
                )
        else:
            raise ValueError("Event end must have either dateTime or date")

        # Validate datetime consistency
        if end_dt <= start_dt:
            raise ValueError("Event end time must be after start time")

        return start_date, start_dt, end_date, end_dt, start_tz, end_tz

    def _validate_timezone(self, tz_name: str) -> str:
        """Validate IANA timezone name"""
        if tz_name is None:
            return tz_name
        try:
            from dateutil.tz import gettz

            if gettz(tz_name) is None:
                raise ValueError("Invalid timeZone; must be a valid IANA timezone name")
        except Exception:
            # If dateutil is unavailable or another error occurs
            raise ValueError("Invalid timeZone; must be a valid IANA timezone name.")
        return tz_name

    def _is_all_day_event(self, start_date: str, end_date: str) -> bool:
        """Check if the event is an all-day event based on date fields"""
        return start_date is not None and end_date is not None

    def _process_import_attendees(
        self, db_event: Event, attendees_data: list, session, organizer=None
    ):
        """Process attendees from import request with organizer handling"""
        if not attendees_data:
            return

        from database.models.event import Attendees

        organizer_email = organizer.email if organizer else None

        for attendee_data in attendees_data:
            # Find or create user based on email
            user = session.query(User).filter(User.email == attendee_data.email).first()
            if not user:
                continue  # Skip unknown users

            # Check if this attendee is the organizer
            is_organizer = organizer_email and attendee_data.email == organizer_email

            attendee = Attendees(
                attendees_id=str(uuid.uuid4()),
                event_id=db_event.event_id,
                user_id=user.user_id,
                displayName=attendee_data.displayName
                or (organizer.displayName if is_organizer and organizer else None),
                optional=attendee_data.optional or False,
                resource=attendee_data.resource or False,
                responseStatus=attendee_data.responseStatus or "needsAction",
                comment=attendee_data.comment,
                additionalGuests=attendee_data.additionalGuests or 0,
            )
            session.add(attendee)

    def _process_import_attachments(
        self, db_event: Event, attachments_data: list, session
    ):
        """Process attachments from import request"""
        if not attachments_data:
            return

        from database.models.event import Attachment

        for attachment_data in attachments_data:
            attachment = Attachment(
                attachment_id=str(uuid.uuid4()),
                event_id=db_event.event_id,
                file_url=attachment_data.fileUrl,
            )
            session.add(attachment)

    def _process_import_conference_data(
        self, db_event: Event, conference_data, version: int, session
    ):
        """Process conference data from import request using new comprehensive schema"""
        if version < 1:
            return  # Conference data not supported in version 0

        from database.models.event import ConferenceData as DBConferenceData

        # Handle both old dict format and new Pydantic ConferenceData model
        conference_id = None
        request_id = None
        solution_type = None
        meeting_uri = None
        label = None
        status_code = None

        if hasattr(conference_data, "conferenceId"):
            # New Pydantic ConferenceData model
            conference_id = conference_data.conferenceId

            # Extract from conferenceSolution if present
            if conference_data.conferenceSolution:
                solution_type = conference_data.conferenceSolution.key.type.value
                label = conference_data.conferenceSolution.name

            # Extract meeting URI from first video entry point
            if conference_data.entryPoints:
                for entry_point in conference_data.entryPoints:
                    if (
                        hasattr(entry_point, "entryPointType")
                        and entry_point.entryPointType.value == "video"
                    ):
                        meeting_uri = entry_point.uri
                        break

            # Extract from createRequest if present
            if conference_data.createRequest:
                request_id = conference_data.createRequest.requestId
                solution_type = (
                    conference_data.createRequest.conferenceSolutionKey.type.value
                )
                if conference_data.createRequest.status:
                    status_code = conference_data.createRequest.status.statusCode.value
        else:
            # Legacy dict format support for backward compatibility
            conference_id = (
                conference_data.get("conferenceId")
                if isinstance(conference_data, dict)
                else None
            )
            if isinstance(conference_data, dict):
                if conference_data.get("createRequest"):
                    request_id = conference_data["createRequest"].get("requestId")
                if conference_data.get("conferenceSolution"):
                    solution_type = (
                        conference_data["conferenceSolution"].get("key", {}).get("type")
                    )
                    label = conference_data["conferenceSolution"].get("name")
                if (
                    conference_data.get("entryPoints")
                    and len(conference_data["entryPoints"]) > 0
                ):
                    meeting_uri = conference_data["entryPoints"][0].get("uri")

        # Create database conference record
        db_conference = DBConferenceData(
            id=conference_id or str(uuid.uuid4()),
            event_id=db_event.event_id,
            request_id=request_id,
            solution_type=solution_type,
            status_code=status_code,
            meeting_uri=meeting_uri,
            label=label,
        )
        session.add(db_conference)

    def _process_import_extended_properties(self, db_event: Event, ext_props, session):
        """Process extended properties from import request"""
        from database.models.event import ExtendedProperty

        if ext_props.private:
            private_prop = ExtendedProperty(
                id=str(uuid.uuid4()),
                event_id=db_event.event_id,
                scope="private",
                properties=ext_props.private,
            )
            session.add(private_prop)

        if ext_props.shared:
            shared_prop = ExtendedProperty(
                id=str(uuid.uuid4()),
                event_id=db_event.event_id,
                scope="shared",
                properties=ext_props.shared,
            )
            session.add(shared_prop)

    def _process_import_working_location(
        self, db_event: Event, working_location, session
    ):
        """Process working location properties from import request"""
        from database.models.event import OfficeLocation, WorkingLocationProperties

        # Create working location
        working_loc = WorkingLocationProperties(
            working_location_id=str(uuid.uuid4()),
            event_id=db_event.event_id,
            type=working_location.type,
            homeOffice=working_location.homeOffice,
            customLocationLabel=working_location.customLocation.get("label")
            if working_location.customLocation
            else None,
        )

        # Handle office location if specified
        if working_location.officeLocation:
            office_loc = OfficeLocation(
                id=str(uuid.uuid4()),
                label=working_location.officeLocation.get("label", ""),
                buildingId=working_location.officeLocation.get("buildingId"),
                floorId=working_location.officeLocation.get("floorId"),
                deskId=working_location.officeLocation.get("deskId"),
                floorSectionId=working_location.officeLocation.get("floorSectionId"),
            )
            session.add(office_loc)
            session.flush()
            working_loc.officeLocationId = office_loc.id

        session.add(working_loc)

    def _process_import_reminders(self, db_event: Event, reminders_data, session):
        """
        Process reminders from import request with proper Google Calendar logic

        Google Calendar reminder logic:
        - If useDefault is True: Use calendar's default reminders (ignore overrides)
        - If useDefault is False: Use custom overrides (if provided)
        - If no reminders specified: No reminders for this event
        """
        if not reminders_data:
            return

        try:
            # Handle useDefault logic
            if reminders_data.useDefault:
                # For useDefault=True, we should use calendar's default reminders
                # Since we don't have calendar defaults in our schema, we'll create
                # a standard default reminder (10 minutes popup)
                default_reminder = Reminder(
                    id=str(uuid.uuid4()),
                    event_id=db_event.event_id,
                    use_default=True,
                    method="popup",  # Default method
                    minutes=10,  # Default 10 minutes before
                )
                session.add(default_reminder)
                logger.info(f"Added default reminder for event {db_event.event_id}")

            else:
                # Handle custom reminder overrides
                if reminders_data.overrides and len(reminders_data.overrides) > 0:
                    for override in reminders_data.overrides:
                        # Validate reminder data
                        if not hasattr(override, "method") or not hasattr(
                            override, "minutes"
                        ):
                            logger.warning(
                                f"Invalid reminder override for event {db_event.event_id}: missing method or minutes"
                            )
                            continue

                        # Validate method
                        if override.method not in ["email", "popup"]:
                            logger.warning(
                                f"Invalid reminder method '{override.method}' for event {db_event.event_id}, skipping"
                            )
                            continue

                        # Validate minutes (should be non-negative)
                        if override.minutes < 0:
                            logger.warning(
                                f"Invalid reminder minutes '{override.minutes}' for event {db_event.event_id}, skipping"
                            )
                            continue

                        custom_reminder = Reminder(
                            id=str(uuid.uuid4()),
                            event_id=db_event.event_id,
                            use_default=False,
                            method=override.method,
                            minutes=override.minutes,
                        )
                        session.add(custom_reminder)
                        logger.info(
                            f"Added custom reminder for event {db_event.event_id}: {override.method} {override.minutes} minutes before"
                        )

                else:
                    # useDefault=False but no overrides means no reminders
                    logger.info(
                        f"No reminders set for event {db_event.event_id} (useDefault=False, no overrides)"
                    )

        except Exception as e:
            logger.error(
                f"Error processing reminders for event {db_event.event_id}: {e}"
            )
            # Don't raise the exception - reminders are not critical for event creation
            # Just log the error and continue

    def _build_import_response(
        self,
        db_event: Event,
        original_type: str,
        final_type: str,
        warnings: list[str],
        ical_uid: str,
        is_all_day: bool = False,
    ) -> EventImportResponse:
        """Build comprehensive import response"""
        import uuid

        # Build creator/organizer info
        creator_organizer = {
            "email": f"user-{db_event.user_id}@calendar.google.com",
            "displayName": "Calendar User",
            "self": True,
        }

        response = EventImportResponse(
            kind="calendar#event",
            id=db_event.event_id,
            etag=f'"{int(db_event.updated_at.timestamp())}"',
            status=db_event.status,
            htmlLink=f"https://calendar.google.com/event?eid={db_event.event_id}",
            created=db_event.created_at.isoformat(),
            updated=db_event.updated_at.isoformat(),
            summary=db_event.summary,
            creator=creator_organizer,
            organizer=creator_organizer,
            start=self._build_event_datetime_response(
                db_event.start_datetime, db_event.start_timezone, is_all_day
            ),
            end=self._build_event_datetime_response(
                db_event.end_datetime, db_event.end_timezone, is_all_day
            ),
            description=db_event.description,
            location=db_event.location,
            transparency="opaque",  # Default for imported events
            visibility=db_event.visibility,
            eventType=final_type,
            iCalUID=ical_uid,
            sequence=db_event.sequence,
            # Guest permissions
            guestsCanInviteOthers=db_event.guestsCanInviteOthers,
            guestsCanModify=db_event.guestsCanModify,
            guestsCanSeeOtherGuests=db_event.guestsCanSeeOtherGuests,
        )

        # Add colorId if present
        if hasattr(db_event, "color_id") and db_event.color_id:
            response.colorId = db_event.color_id

        # Add recurrence if present
        if db_event.recurrence:
            response.recurrence = json.loads(db_event.recurrence)

        # Add attendees if present
        if db_event.attendees:
            attendees_list = []
            for attendee in db_event.attendees:
                attendee_data = {
                    "email": attendee.user.email if attendee.user else None,
                    "displayName": attendee.displayName,
                    "responseStatus": attendee.responseStatus,
                    "optional": attendee.optional,
                    "comment": attendee.comment,
                    "additionalGuests": attendee.additionalGuests,
                    "resource": attendee.resource,
                }
                attendees_list.append(attendee_data)
            response.attendees = attendees_list

        # Add attachments if present
        if db_event.attachments:
            attachments_list = []
            for attachment in db_event.attachments:
                attachment_data = {
                    "fileUrl": attachment.file_url,
                    "title": attachment.file_url.split("/")[-1]
                    if attachment.file_url
                    else "attachment",
                }
                attachments_list.append(attachment_data)
            response.attachments = attachments_list

        # Add conference data if present
        if hasattr(db_event, "conferenceData") and db_event.conferenceData:
            conf = db_event.conferenceData
            conference_data = {"conferenceId": conf.id}

            # Add conferenceSolution if available
            if conf.solution_type or conf.solution_name or conf.solution_icon_uri:
                conference_solution = {}
                if conf.solution_icon_uri:
                    conference_solution["iconUri"] = conf.solution_icon_uri
                if conf.solution_type:
                    conference_solution["key"] = {"type": conf.solution_type}
                if conf.solution_name:
                    conference_solution["name"] = conf.solution_name
                conference_data["conferenceSolution"] = conference_solution

            # Add createRequest if available
            if conf.request_id or conf.create_solution_type or conf.status_code:
                create_request = {}
                if conf.request_id:
                    create_request["requestId"] = conf.request_id
                if conf.create_solution_type:
                    create_request["conferenceSolution"] = {
                        "key": {"type": conf.create_solution_type}
                    }
                if conf.status_code:
                    create_request["status"] = {"statusCode": conf.status_code}
                conference_data["createRequest"] = create_request

            # Add entryPoints from JSON array or legacy field
            entry_points = []
            if conf.entry_points:
                # Use new JSON array format
                entry_points = conf.entry_points
            elif conf.meeting_uri:
                # Fallback to legacy format
                entry_points = [{"entryPointType": "video", "uri": conf.meeting_uri}]

            if entry_points:
                conference_data["entryPoints"] = entry_points

            # Add notes and signature if available
            if conf.notes:
                conference_data["notes"] = conf.notes
            if conf.signature:
                conference_data["signature"] = conf.signature

            response.conferenceData = conference_data

        # Add source if present
        if hasattr(db_event, "source") and db_event.source:
            response.source = db_event.source

        # Add extended properties if present
        if hasattr(db_event, "extendedProperties") and db_event.extendedProperties:
            ext_props = {}
            private_props = {}
            shared_props = {}

            for prop in db_event.extendedProperties:
                if prop.scope == "private":
                    private_props.update(prop.properties or {})
                elif prop.scope == "shared":
                    shared_props.update(prop.properties or {})

            if private_props:
                ext_props["private"] = private_props
            if shared_props:
                ext_props["shared"] = shared_props

            if ext_props:
                response.extendedProperties = ext_props

        # Add working location properties if present
        if (
            hasattr(db_event, "workingLocationProperties")
            and db_event.workingLocationProperties
        ):
            working_loc = {
                "type": db_event.workingLocationProperties.type,
                "homeOffice": db_event.workingLocationProperties.homeOffice,
            }

            if db_event.workingLocationProperties.customLocationLabel:
                working_loc["customLocation"] = {
                    "label": db_event.workingLocationProperties.customLocationLabel
                }

            if db_event.workingLocationProperties.officeLocation:
                working_loc["officeLocation"] = {
                    "buildingId": db_event.workingLocationProperties.officeLocation.buildingId,
                    "floorId": db_event.workingLocationProperties.officeLocation.floorId,
                    "deskId": db_event.workingLocationProperties.officeLocation.deskId,
                    "floorSectionId": db_event.workingLocationProperties.officeLocation.floorSectionId,
                    "label": db_event.workingLocationProperties.officeLocation.label,
                }

            response.workingLocationProperties = working_loc

        # Add birthday properties if present
        if hasattr(db_event, "birthdayProperties") and db_event.birthdayProperties:
            response.birthdayProperties = db_event.birthdayProperties

        # Add focus time properties if present
        if hasattr(db_event, "focusTimeProperties") and db_event.focusTimeProperties:
            response.focusTimeProperties = db_event.focusTimeProperties

        # Add out of office properties if present
        if (
            hasattr(db_event, "outOfOfficeProperties")
            and db_event.outOfOfficeProperties
        ):
            response.outOfOfficeProperties = db_event.outOfOfficeProperties

        # Add reminders if present
        if db_event.reminders:
            # Build reminders response format
            reminders_response = {"useDefault": False, "overrides": []}

            # Check if any reminder uses default
            use_default = any(reminder.use_default for reminder in db_event.reminders)

            if use_default:
                reminders_response["useDefault"] = True

            # Build overrides list
            overrides = []
            for reminder in db_event.reminders:
                overrides.append(
                    {
                        "method": reminder.method.value
                        if hasattr(reminder.method, "value")
                        else reminder.method,
                        "minutes": reminder.minutes,
                    }
                )

            reminders_response["overrides"] = overrides

            response.reminders = reminders_response

        return response

    def _build_datetime_for_schema(self, dt: datetime, tz: str) -> DateTime:
        """Build DateTime response for regular event schema"""
        # Check if this appears to be an all-day event (time is 00:00:00 and timezone is UTC)
        is_all_day = dt.time() == dt.min.time()

        if is_all_day:
            # For all-day events, return date format
            return DateTime(date=dt.date().isoformat(), timeZone=tz)
        else:
            # For timed events, return dateTime format
            return DateTime(dateTime=dt.isoformat() if dt else None, timeZone=tz)

    def _build_original_start_time_for_schema(
        self, db_event: Event
    ) -> Optional[DateTime]:
        """Build originalStartTime DateTime response for event schema"""
        if (
            hasattr(db_event, "originalStartTime_date")
            and db_event.originalStartTime_date
        ):
            # All-day event
            return DateTime(
                date=db_event.originalStartTime_date.isoformat(),
                timeZone=db_event.originalStartTime_timeZone,
            )
        elif (
            hasattr(db_event, "originalStartTime_dateTime")
            and db_event.originalStartTime_dateTime
        ):
            # Timed event
            return DateTime(
                dateTime=db_event.originalStartTime_dateTime.isoformat(),
                timeZone=db_event.originalStartTime_timeZone,
            )
        else:
            # No originalStartTime data stored
            return None

    def _build_event_datetime_response(
        self, dt: datetime, tz: str, is_all_day: bool
    ) -> EventDateTime:
        """Build EventDateTime response with proper format for all-day vs timed events"""
        if is_all_day:
            # For all-day events, return date format (YYYY-MM-DD)
            return EventDateTime(
                date=dt.date().isoformat() if dt else None, timeZone=tz
            )
        else:
            # For timed events, return dateTime format with timezone
            return EventDateTime(dateTime=dt.isoformat() if dt else None, timeZone=tz)

    def get_event_instances(
        self,
        user_id: str,
        calendar_id: str,
        event_id: str,
        max_attendees: Optional[int] = None,
        max_results: Optional[int] = None,
        original_start: Optional[str] = None,
        page_token: Optional[str] = None,
        show_deleted: Optional[bool] = None,
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        time_zone: Optional[str] = None,
    ) -> EventInstancesResponse:
        """
        Returns instances of the specified recurring event following Google Calendar API v3 specification

        Supports all query parameters as defined in the official documentation:
        - maxAttendees (limits attendees in response)
        - maxResults (limits number of instances returned, max 2500)
        - originalStart (filter for specific instance)
        - pageToken (pagination support)
        - showDeleted (includes cancelled instances)
        - timeMin/timeMax (time range filtering)
        - timeZone (response timezone)

        GET /calendars/{calendarId}/events/{eventId}/instances
        """
        session = get_session(self.database_id)
        try:
            # Parse page token to get offset
            offset = 0
            if page_token:
                try:
                    offset = self._decode_page_token(page_token)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid page token: {page_token}, error: {e}")
                    offset = 0

            # Verify calendar access with proper ACL permissions
            calendar = (
                session.query(Calendar)
                .filter(Calendar.calendar_id == calendar_id)
                .first()
            )
            if not calendar:
                raise ValueError(f"Calendar {calendar_id} not found")

            # Check user access to calendar and event
            user_role = self._get_user_calendar_role(user_id, calendar)
            if user_role == "none":
                raise PermissionError(
                    f"User '{user_id}' has no access to calendar '{calendar_id}'"
                )

            # Get the recurring event with proper permissions check
            db_event_query = (
                session.query(Event)
                .options(
                    joinedload(Event.attendees).joinedload(Attendees.user),
                    joinedload(Event.attachments),
                    joinedload(Event.reminders),
                )
                .filter(
                    and_(
                        Event.calendar_id == calendar_id,
                        Event.recurring_event_id == event_id,
                    )
                )
            )

            db_event = db_event_query.first()

            if not db_event:
                raise ValueError(f"Recurring Event not found: {event_id}")

            # Check event visibility permissions
            if not self.check_event_visibility_permission(
                user_id, calendar_id, db_event.event_id
            ):
                raise PermissionError(
                    f"User '{user_id}' cannot view the recurring event '{event_id}'"
                )

            if not db_event.recurrence:
                raise ValueError(f"Event is not recurring: {event_id}")

            # Add timezone filter if provided
            if time_zone:
                # Validate timezone
                self._validate_timezone(time_zone)
            else:
                time_zone = calendar.time_zone

            # Apply other filters
            if not show_deleted:
                db_event_query = db_event_query.filter(Event.status != "cancelled")

            if time_min:
                time_min_dt = self._parse_datetime_string(time_min)
                db_event_query = db_event_query.filter(
                    Event.end_datetime >= time_min_dt
                )

            if time_max:
                time_max_dt = self._parse_datetime_string(time_max)
                db_event_query = db_event_query.filter(
                    Event.start_datetime < time_max_dt
                )

            if original_start:
                original_start_time_dt = self._parse_datetime_string(original_start)
                db_event_query = db_event_query.filter(
                    Event.originalStartTime_dateTime == original_start_time_dt
                )

            # Apply offset for pagination
            if offset > 0:
                db_event_query = db_event_query.offset(offset)

            db_event = db_event_query.all()

            instances = []
            for event in db_event:
                base_instance = self._convert_db_event_to_schema(event)
                if (
                    max_attendees is not None
                    and base_instance.attendees
                    and len(base_instance.attendees) > max_attendees
                ):
                    base_instance.attendees = base_instance.attendees[:max_attendees]
                    base_instance.attendeesOmitted = True
                instances.append(base_instance)

            # Apply maxResults limit (default 250, max 2500)
            if max_results is not None:
                max_results = min(max_results, 2500)  # Enforce API limit
                instances = instances[:max_results]

            # Determine response timezone
            response_timezone = time_zone

            # Build pagination response
            next_page_token = None
            # In a real implementation, this would handle pagination based on page_token

            # Determine access role for response
            access_role = user_role if user_role != "freeBusyReader" else "reader"

            return EventInstancesResponse(
                kind="calendar#events",
                etag=f'"instances-{event_id}-{int(datetime.now(timezone.utc).timestamp())}"',
                summary=calendar.summary if calendar else db_event.summary,
                description=calendar.description if calendar else None,
                updated=datetime.now(timezone.utc).isoformat(),
                timeZone=response_timezone,
                accessRole=access_role,
                defaultReminders=[],  # Calendar defaults would go here
                nextPageToken=next_page_token,
                items=instances,
            )

        except Exception as e:
            logger.error(f"Error getting instances for event {event_id}: {e}")
            raise
        finally:
            session.close()

    def watch_events(
        self,
        user_id: str,
        calendar_id: str,
        watch_request: Dict[str, Any],
        event_types: Optional[str] = None,
    ) -> Channel:
        """
        Watch for changes to events with optional event type filtering

        Args:
            user_id: User setting up the watch
            calendar_id: Calendar to watch
            watch_request: Watch request details
            event_types: Optional comma-separated string of event types to watch
                        Acceptable values: "birthday", "default", "focusTime", "fromGmail", "outOfOffice", "workingLocation"

        POST /calendars/{calendarId}/events/watch
        """
        session = get_session(self.database_id)
        try:
            # Generate unique resource ID for events watch
            resource_id = f"events-{calendar_id}-{uuid.uuid4().hex[:8]}"
            if event_types is not None:
                resource_uri = (
                    f"/calendars/{calendar_id}/events?eventTypes={event_types}"
                )
            else:
                resource_uri = f"/calendars/{calendar_id}/events"

            # Calculate expiration time (max 24 hours from now if not specified)
            now = datetime.utcnow()
            expires_at = now + timedelta(hours=24)

            # Verify calendar belongs to user
            calendar = (
                session.query(Calendar)
                .filter(
                    Calendar.calendar_id == calendar_id, Calendar.user_id == user_id
                )
                .first()
            )
            if not calendar:
                raise ValueError(f"Calendar {calendar_id} not found for user {user_id}")

            if (
                session.query(WatchChannel)
                .filter(WatchChannel.id == watch_request.id)
                .first()
            ):
                raise ValueError(f"Channel with Id {watch_request.id} already exists")

            # Prepare watch parameters including event types filter
            watch_params = {}
            if watch_request.params:
                watch_params = watch_request.params.model_dump()

            # Add event types filter if specified
            if event_types:
                watch_params["eventTypes"] = event_types
                logger.info(f"Watch channel will filter for event types: {event_types}")
            else:
                logger.info(f"Watch channel will monitor all event types")

            # Create watch channel record
            watch_channel = WatchChannel(
                id=watch_request.id,
                resource_id=resource_id,
                resource_uri=resource_uri,
                resource_type="event",
                calendar_id=calendar_id,
                user_id=user_id,
                webhook_address=watch_request.address,
                webhook_token=watch_request.token,
                webhook_type=watch_request.type,
                params=json.dumps(watch_params) if watch_params else None,
                created_at=now,
                expires_at=expires_at,
                is_active="true",
                notification_count=0,
            )

            # Save to database
            session.add(watch_channel)
            session.commit()

            logger.info(
                f"Created settings watch channel {watch_request.id} for user {user_id}"
            )

            channel = Channel(
                id=watch_request.id,
                resourceId=resource_id,
                resourceUri=resource_uri,
                token=watch_channel.webhook_token,
                expiration=expires_at.isoformat() + "Z" if expires_at else None,
            )

            logger.info(f"Set up watch for events in calendar {calendar_id}")
            return channel

        except Exception as e:
            logger.error(
                f"Error setting up events watch for calendar {calendar_id}: {e}"
            )
            raise
        finally:
            session.close()

    def _handle_create_notifications(self, db_event: Event, send_updates: str, session):
        """
        Handle notification logic for event creation based on sendUpdates parameter

        Args:
            db_event: The event being created
            send_updates: Notification scope ("all", "externalOnly", "none")
            session: Database session
        """

        if send_updates == "none":
            logger.info(
                f"No notifications will be sent for created event {db_event.event_id}"
            )
            return

        # Get event attendees
        attendees = db_event.attendees if db_event.attendees else []

        if not attendees:
            logger.info(
                f"No attendees found for created event {db_event.event_id}, no notifications to send"
            )
            return

        # Determine which attendees should receive notifications
        notification_recipients = []

        for attendee in attendees:
            if not attendee.user or not attendee.user.email:
                continue

            should_notify = False

            if send_updates == "all":
                # Send to all attendees
                should_notify = True
            elif send_updates == "externalOnly":
                # Send only to non-Google Calendar guests (external email domains)
                # For this implementation, we'll consider emails not ending with common Google domains as external
                email_domain = attendee.user.email.split("@")[-1].lower()
                google_domains = ["gmail.com", "googlemail.com", "google.com"]
                if email_domain not in google_domains:
                    should_notify = True

            if should_notify:
                notification_recipients.append(
                    {
                        "email": attendee.user.email,
                        "displayName": attendee.displayName or attendee.user.email,
                    }
                )

        # Log notification details (in a real implementation, this would send actual notifications)
        if notification_recipients:
            logger.info(
                f"Event creation notifications will be sent to {len(notification_recipients)} recipients:"
            )
            for recipient in notification_recipients:
                logger.info(f"  - {recipient['displayName']} ({recipient['email']})")
        else:
            logger.info(
                f"No attendees match notification criteria for sendUpdates='{send_updates}'"
            )

    def _handle_notifications(self, db_event: Event, send_updates: str, session):
        """
        Handle notification logic for event operations based on sendUpdates parameter

        Args:
            db_event: The event being processed
            send_updates: Notification scope ("all", "externalOnly", "none")
            session: Database session
        """

        if send_updates == "none":
            logger.info(f"No notifications will be sent for event {db_event.event_id}")
            return

        # Get event attendees
        attendees = db_event.attendees if db_event.attendees else []

        if not attendees:
            logger.info(
                f"No attendees found for event {db_event.event_id}, no notifications to send"
            )
            return

        # Determine which attendees should receive notifications
        notification_recipients = []

        for attendee in attendees:
            if not attendee.user or not attendee.user.email:
                continue

            should_notify = False

            if send_updates == "all":
                # Send to all attendees
                should_notify = True
            elif send_updates == "externalOnly":
                # Send only to non-Google Calendar guests (external email domains)
                # For this implementation, we'll consider emails not ending with common Google domains as external
                email_domain = attendee.user.email.split("@")[-1].lower()
                google_domains = ["gmail.com", "googlemail.com", "google.com"]
                if email_domain not in google_domains:
                    should_notify = True

            if should_notify:
                notification_recipients.append(
                    {
                        "email": attendee.user.email,
                        "displayName": attendee.displayName or attendee.user.email,
                    }
                )

        # Log notification details (in a real implementation, this would send actual notifications)
        if notification_recipients:
            logger.info(
                f"Event notifications will be sent to {len(notification_recipients)} recipients:"
            )
            for recipient in notification_recipients:
                logger.info(f"  - {recipient['displayName']} ({recipient['email']})")
        else:
            logger.info(
                f"No attendees match notification criteria for sendUpdates='{send_updates}'"
            )

    def _encode_page_token(self, offset: int) -> str:
        """Encode offset as a page token"""
        try:
            token_data = str(offset)
            return base64.b64encode(token_data.encode("utf-8")).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding page token: {e}")
            return ""

    def _decode_page_token(self, token: str) -> int:
        """Decode page token to get offset"""
        try:
            # Handle legacy case where raw numbers might be passed
            if token.isdigit():
                logger.warning(
                    f"Raw numeric page token received: {token}. This should be a base64-encoded token."
                )
                return int(token)

            # Add padding if needed for base64 decoding
            missing_padding = len(token) % 4
            if missing_padding:
                token += "=" * (4 - missing_padding)

            decoded = base64.b64decode(token.encode("utf-8")).decode("utf-8")
            return int(decoded)
        except Exception as e:
            logger.error(f"Error decoding page token: {e}")
            raise ValueError(
                f"Invalid page token: {token}. Page tokens should only be generated by the API."
            )

    def _encode_sync_token(self, timestamp: datetime) -> str:
        """Encode timestamp as a sync token"""
        try:
            # Use ISO format timestamp for sync token
            token_data = timestamp.isoformat()
            return base64.b64encode(token_data.encode("utf-8")).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding sync token: {e}")
            return ""

    def _decode_sync_token(self, token: str) -> datetime:
        """Decode sync token to get timestamp"""
        try:
            # Add padding if needed for base64 decoding
            missing_padding = len(token) % 4
            if missing_padding:
                token += "=" * (4 - missing_padding)

            decoded = base64.b64decode(token.encode("utf-8")).decode("utf-8")
            return datetime.fromisoformat(decoded)
        except Exception as e:
            logger.error(f"Error decoding sync token: {e}")
            raise ValueError(
                f"Invalid sync token. Token may have expired or is malformed."
            )
