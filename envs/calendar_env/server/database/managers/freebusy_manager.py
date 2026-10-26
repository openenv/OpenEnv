"""
FreeBusy database manager with Google Calendar API v3 compatible operations
Handles FreeBusy query operations with database-per-user architecture
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from database.models.calendar import Calendar
from database.models.event import Event
from database.models.user import User
from database.session_utils import get_session, init_database
from dateutil import parser
from schemas.freebusy import (
    FreeBusyCalendarResult as FreeBusyCalendarResultSchema,
    FreeBusyError,
    FreeBusyEventOverlap,
    FreeBusyQueryRequest,
    FreeBusyQueryResponse,
    FreeBusyQueryValidation,
    TimePeriod,
)
from sqlalchemy import and_, asc, desc, or_
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class FreeBusyManager:
    """FreeBusy manager for database operations"""

    def __init__(self, database_id: str):
        self.database_id = database_id
        # Initialize database on first use
        init_database(database_id)

    def _parse_datetime_string(self, datetime_str: str) -> datetime:
        """Parse ISO datetime string to datetime object"""
        try:
            return parser.isoparse(datetime_str.replace("Z", "+00:00"))
        except Exception as e:
            logger.error(f"Error parsing datetime string {datetime_str}: {e}")
            raise ValueError(f"Invalid datetime format: {datetime_str}")

    def _validate_query_request(
        self, request: FreeBusyQueryRequest, user_id: str
    ) -> FreeBusyQueryValidation:
        """Validate FreeBusy query request"""
        time_min = self._parse_datetime_string(request.timeMin)
        time_max = self._parse_datetime_string(request.timeMax)
        calendar_ids = [item.id for item in request.items]

        validation = FreeBusyQueryValidation(
            time_min=time_min,
            time_max=time_max,
            calendar_ids=calendar_ids,
            user_id=user_id,
        )

        if not validation.validate_time_range():
            raise ValueError("Time range is invalid or too large (max 366 days)")

        if not validation.validate_calendar_count():
            raise ValueError("Too many calendars requested (max 50)")

        # Validate that all calendar IDs exist in the database
        self._validate_calendar_ids_exist(calendar_ids, user_id)

        return validation

    def _validate_calendar_ids_exist(
        self, calendar_ids: List[str], user_id: str
    ) -> None:
        """Validate that all calendar IDs exist in the database for the user"""
        session = get_session(self.database_id)
        try:
            for calendar_id in calendar_ids:
                calendar = (
                    session.query(Calendar)
                    .filter(
                        Calendar.calendar_id == calendar_id, Calendar.user_id == user_id
                    )
                    .first()
                )

                if not calendar:
                    raise ValueError(
                        f"Calendar with ID '{calendar_id}' does not exist or is not accessible"
                    )

        finally:
            session.close()

    def _get_busy_periods_for_calendar(
        self,
        session,
        calendar_id: str,
        user_id: str,
        time_min: datetime,
        time_max: datetime,
    ) -> List[TimePeriod]:
        """Get busy periods for a specific calendar"""
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
                # This should not happen if validation passed, but handle gracefully
                logger.error(
                    f"Calendar {calendar_id} not found for user {user_id} during query execution"
                )
                raise ValueError(
                    f"Calendar with ID '{calendar_id}' does not exist or is not accessible"
                )

            # Query events that overlap with the time range
            events = (
                session.query(Event)
                .filter(
                    and_(
                        Event.calendar_id == calendar_id,
                        Event.user_id == user_id,
                        Event.status == "confirmed",  # Only confirmed events block time
                        # Event overlaps with query range
                        Event.start_datetime < time_max,
                        Event.end_datetime > time_min,
                    )
                )
                .all()
            )

            busy_periods = []
            for event in events:
                # Check if event is transparent (doesn't block time)
                if (
                    hasattr(event, "transparency")
                    and event.transparency == "transparent"
                ):
                    continue

                # Create overlap object for validation
                overlap = FreeBusyEventOverlap(
                    event_id=event.event_id,
                    start=event.start_datetime,
                    end=event.end_datetime,
                    transparency=getattr(event, "transparency", "opaque"),
                )

                if overlap.is_busy():
                    # Clip event times to query range
                    period_start = max(event.start_datetime, time_min)
                    period_end = min(event.end_datetime, time_max)

                    busy_periods.append(
                        TimePeriod(
                            start=period_start.isoformat(), end=period_end.isoformat()
                        )
                    )

            # Merge overlapping periods
            busy_periods = self._merge_overlapping_periods(busy_periods)

            return busy_periods

        except Exception as e:
            logger.error(f"Error getting busy periods for calendar {calendar_id}: {e}")
            raise

    def _merge_overlapping_periods(self, periods: List[TimePeriod]) -> List[TimePeriod]:
        """Merge overlapping time periods"""
        if not periods:
            return []

        # Sort periods by start time
        sorted_periods = sorted(periods, key=lambda p: p.start)

        merged = [sorted_periods[0]]

        for current in sorted_periods[1:]:
            last_merged = merged[-1]

            # Parse times for comparison
            current_start = self._parse_datetime_string(current.start)
            current_end = self._parse_datetime_string(current.end)
            last_end = self._parse_datetime_string(last_merged.end)

            # If current period overlaps with last merged period
            if current_start <= last_end:
                # Extend the last merged period if necessary
                if current_end > last_end:
                    merged[-1] = TimePeriod(start=last_merged.start, end=current.end)
            else:
                # No overlap, add as new period
                merged.append(current)

        return merged

    def query_freebusy(
        self, user_id: str, request: FreeBusyQueryRequest
    ) -> FreeBusyQueryResponse:
        """
        Query free/busy information for calendars

        POST /freeBusy
        """
        session = get_session(self.database_id)
        try:
            # Validate request
            validation = self._validate_query_request(request, user_id)

            # Process each calendar
            calendar_results = {}

            for calendar_item in request.items:
                calendar_id = calendar_item.id

                try:
                    # Get busy periods for this calendar
                    busy_periods = self._get_busy_periods_for_calendar(
                        session,
                        calendar_id,
                        user_id,
                        validation.time_min,
                        validation.time_max,
                    )

                    # Add to response
                    calendar_results[calendar_id] = FreeBusyCalendarResultSchema(
                        busy=busy_periods
                    )

                except Exception as e:
                    logger.error(f"Error processing calendar {calendar_id}: {e}")

                    # Add error to response
                    calendar_results[calendar_id] = FreeBusyCalendarResultSchema(
                        errors=[FreeBusyError(domain="calendar", reason="backendError")]
                    )

            response = FreeBusyQueryResponse(
                timeMin=request.timeMin,
                timeMax=request.timeMax,
                calendars=calendar_results,
            )

            logger.info(f"Processed FreeBusy query for user {user_id}")
            return response

        except Exception as e:
            session.rollback()
            logger.error(f"Error processing FreeBusy query for user {user_id}: {e}")
            raise ValueError(f"Error processing FreeBusy query for user {user_id}: {e}")
        finally:
            session.close()
