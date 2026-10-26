"""
Recurrence utilities for parsing RFC 5545 RRULE strings and generating event instances
"""

import json

# from database.models.recurring_event import RecurrenceFrequency
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as date_parser, rrule
from dateutil.rrule import rruleset, rrulestr
from dateutil.tz import gettz

logger = logging.getLogger(__name__)


class RecurrenceParseError(Exception):
    """Custom exception for recurrence parsing errors"""

    pass


class RecurrenceParser:
    """
    Parser for RFC 5545 recurrence rules (RRULE, RDATE, EXDATE)
    """

    # Mapping from RFC 5545 frequency to dateutil.rrule frequency
    FREQ_MAP = {
        "SECONDLY": rrule.SECONDLY,
        "MINUTELY": rrule.MINUTELY,
        "HOURLY": rrule.HOURLY,
        "DAILY": rrule.DAILY,
        "WEEKLY": rrule.WEEKLY,
        "MONTHLY": rrule.MONTHLY,
        "YEARLY": rrule.YEARLY,
    }

    # Mapping from RFC 5545 weekdays to dateutil.rrule weekdays
    WEEKDAY_MAP = {
        "MO": rrule.MO,
        "TU": rrule.TU,
        "WE": rrule.WE,
        "TH": rrule.TH,
        "FR": rrule.FR,
        "SA": rrule.SA,
        "SU": rrule.SU,
    }

    @staticmethod
    def parse_recurrence_list(
        recurrence_strings: List[str], event_start: datetime
    ) -> Dict[str, Any]:
        """
        Parse a list of recurrence strings (RRULE, RDATE, EXDATE)

        Args:
            recurrence_strings: List of RFC 5545 recurrence strings

        Returns:
            Dictionary with parsed recurrence components
        """
        parsed_data = {"rrule": None, "rdate_list": [], "exdate_list": []}
        rset = rruleset()

        for rec_string in recurrence_strings:
            rec_string = rec_string.strip()

            if rec_string.startswith("RRULE:"):
                if parsed_data["rrule"] is not None:
                    raise RecurrenceParseError(
                        "Multiple RRULE entries found - only one is allowed"
                    )
                parsed_data["rrule"] = RecurrenceParser.parse_rrule(
                    rec_string[6:]
                )  # Remove 'RRULE:' prefix

                rule_str = rec_string.replace("RRULE:", "")
                rule = rrulestr(rule_str, dtstart=event_start)
                rset.rrule(rule)

            elif rec_string.startswith("RDATE:"):
                rdate_values = RecurrenceParser.parse_rdate_exdate(
                    rec_string[6:]
                )  # Remove 'RDATE:' prefix
                parsed_data["rdate_list"].extend(rdate_values)

                rdates_str = rec_string.replace("RDATE:", "")
                for d in rdates_str.split(","):
                    rdate_dt = datetime.strptime(d.strip(), "%Y%m%dT%H%M%SZ").replace(
                        tzinfo=timezone.utc
                    )
                    rset.rdate(rdate_dt)

            elif rec_string.startswith("EXDATE:"):
                exdate_values = RecurrenceParser.parse_rdate_exdate(
                    rec_string[7:]
                )  # Remove 'EXDATE:' prefix
                parsed_data["exdate_list"].extend(exdate_values)

                exdates_str = rec_string.replace("EXDATE:", "")
                for d in exdates_str.split(","):
                    exdate_dt = datetime.strptime(d.strip(), "%Y%m%dT%H%M%SZ").replace(
                        tzinfo=timezone.utc
                    )
                    rset.exdate(exdate_dt)

            else:
                logger.warning(f"Unknown recurrence string format: {rec_string}")

        return parsed_data, rset

    @staticmethod
    def parse_rrule(rrule_string: str) -> Dict[str, Any]:
        """
        Parse an RRULE string according to RFC 5545

        Args:
            rrule_string: RRULE parameters (without 'RRULE:' prefix)

        Returns:
            Dictionary with parsed RRULE parameters
        """
        rrule_data = {}

        # Split parameters by semicolon
        parameters = rrule_string.split(";")

        for param in parameters:
            if "=" not in param:
                continue

            key, value = param.split("=", 1)
            key = key.upper()

            if key == "FREQ":
                if value.upper() not in RecurrenceParser.FREQ_MAP:
                    raise RecurrenceParseError(f"Invalid FREQ value: {value}")
                rrule_data["freq"] = value.upper()

            elif key == "UNTIL":
                try:
                    # Parse UNTIL datetime
                    until_dt = RecurrenceParser.parse_datetime(value)
                    rrule_data["until"] = until_dt
                except Exception as e:
                    raise RecurrenceParseError(
                        f"Invalid UNTIL value: {value}, error: {e}"
                    )

            elif key == "COUNT":
                try:
                    count = int(value)
                    if count <= 0:
                        raise RecurrenceParseError("COUNT must be positive")
                    rrule_data["count"] = count
                except ValueError:
                    raise RecurrenceParseError(f"Invalid COUNT value: {value}")

            elif key == "INTERVAL":
                try:
                    interval = int(value)
                    if interval <= 0:
                        raise RecurrenceParseError("INTERVAL must be positive")
                    rrule_data["interval"] = interval
                except ValueError:
                    raise RecurrenceParseError(f"Invalid INTERVAL value: {value}")

            elif key == "BYSECOND":
                rrule_data["by_second"] = RecurrenceParser.parse_int_list(value, 0, 60)

            elif key == "BYMINUTE":
                rrule_data["by_minute"] = RecurrenceParser.parse_int_list(value, 0, 59)

            elif key == "BYHOUR":
                rrule_data["by_hour"] = RecurrenceParser.parse_int_list(value, 0, 23)

            elif key == "BYDAY":
                rrule_data["by_day"] = RecurrenceParser.parse_by_day(value)

            elif key == "BYMONTHDAY":
                rrule_data["by_monthday"] = RecurrenceParser.parse_int_list(
                    value, -31, 31, allow_zero=False
                )

            elif key == "BYYEARDAY":
                rrule_data["by_yearday"] = RecurrenceParser.parse_int_list(
                    value, -366, 366, allow_zero=False
                )

            elif key == "BYWEEKNO":
                rrule_data["by_weekno"] = RecurrenceParser.parse_int_list(
                    value, -53, 53, allow_zero=False
                )

            elif key == "BYMONTH":
                rrule_data["by_month"] = RecurrenceParser.parse_int_list(value, 1, 12)

            elif key == "BYSETPOS":
                rrule_data["by_setpos"] = RecurrenceParser.parse_int_list(
                    value, -366, 366, allow_zero=False
                )

            elif key == "WKST":
                if value.upper() not in RecurrenceParser.WEEKDAY_MAP:
                    raise RecurrenceParseError(f"Invalid WKST value: {value}")
                rrule_data["wkst"] = value.upper()

            else:
                logger.warning(f"Unknown RRULE parameter: {key}")

        # Validate required FREQ parameter
        if "freq" not in rrule_data:
            raise RecurrenceParseError("FREQ parameter is required in RRULE")

        # Validate COUNT and UNTIL are mutually exclusive
        if "count" in rrule_data and "until" in rrule_data:
            raise RecurrenceParseError(
                "COUNT and UNTIL cannot both be specified in RRULE"
            )

        return rrule_data

    @staticmethod
    def parse_rdate_exdate(date_string: str) -> List[datetime]:
        """
        Parse RDATE or EXDATE values

        Args:
            date_string: Date list string (comma-separated dates)

        Returns:
            List of datetime objects
        """
        dates = []
        date_values = date_string.split(",")

        for date_value in date_values:
            date_value = date_value.strip()
            if date_value:
                try:
                    dt = RecurrenceParser.parse_datetime(date_value)
                    dates.append(dt)
                except Exception as e:
                    logger.warning(f"Failed to parse date value '{date_value}': {e}")

        return dates

    @staticmethod
    def parse_datetime(datetime_string: str) -> datetime:
        """
        Parse RFC 5545 datetime string

        Args:
            datetime_string: RFC 5545 datetime string

        Returns:
            datetime object
        """
        # Handle UTC times ending with 'Z'
        if datetime_string.endswith("Z"):
            datetime_string = datetime_string[:-1] + "+00:00"

        # Use dateutil parser for flexibility
        try:
            dt = date_parser.isoparse(datetime_string)
            # Ensure timezone-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            # Fallback to manual parsing for RFC 5545 format
            try:
                if "T" in datetime_string:
                    # YYYYMMDDTHHMMSS format
                    if "+" in datetime_string or "-" in datetime_string:
                        # Has timezone
                        dt_part = datetime_string.split("+")[0].split("-")[0]
                        dt = datetime.strptime(dt_part, "%Y%m%dT%H%M%S")
                    else:
                        dt = datetime.strptime(datetime_string, "%Y%m%dT%H%M%S")
                else:
                    # YYYYMMDD format (date only)
                    dt = datetime.strptime(datetime_string, "%Y%m%d")

                # Always ensure timezone-aware
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception as e:
                raise RecurrenceParseError(
                    f"Cannot parse datetime: {datetime_string}, error: {e}"
                )

    @staticmethod
    def parse_int_list(
        value_string: str, min_val: int, max_val: int, allow_zero: bool = True
    ) -> List[int]:
        """
        Parse comma-separated list of integers with validation

        Args:
            value_string: Comma-separated string of integers
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_zero: Whether zero is allowed

        Returns:
            List of validated integers
        """
        values = []
        parts = value_string.split(",")

        for part in parts:
            part = part.strip()
            if part:
                try:
                    val = int(part)
                    if not allow_zero and val == 0:
                        raise RecurrenceParseError(
                            f"Zero not allowed in this context: {part}"
                        )
                    if val < min_val or val > max_val:
                        raise RecurrenceParseError(
                            f"Value {val} out of range [{min_val}, {max_val}]"
                        )
                    values.append(val)
                except ValueError:
                    raise RecurrenceParseError(f"Invalid integer value: {part}")

        return values

    @staticmethod
    def parse_by_day(value_string: str) -> List[str]:
        """
        Parse BYDAY parameter (e.g., "MO,TU,WE" or "1MO,-1FR")

        Args:
            value_string: BYDAY parameter value

        Returns:
            List of weekday specifications
        """
        weekdays = []
        parts = value_string.split(",")

        weekday_pattern = re.compile(r"^([+-]?\d*)([A-Z]{2})$")

        for part in parts:
            part = part.strip().upper()
            if part:
                match = weekday_pattern.match(part)
                if not match:
                    raise RecurrenceParseError(f"Invalid BYDAY value: {part}")

                ordinal, weekday = match.groups()

                if weekday not in RecurrenceParser.WEEKDAY_MAP:
                    raise RecurrenceParseError(f"Invalid weekday: {weekday}")

                # Validate ordinal if present
                if ordinal:
                    try:
                        ord_val = int(ordinal)
                        if ord_val == 0 or abs(ord_val) > 53:
                            raise RecurrenceParseError(
                                f"Invalid ordinal in BYDAY: {ordinal}"
                            )
                    except ValueError:
                        raise RecurrenceParseError(
                            f"Invalid ordinal in BYDAY: {ordinal}"
                        )

                weekdays.append(part)

        return weekdays


class EventInstanceGenerator:
    """
    Generator for creating event instances from recurring event templates
    """

    @staticmethod
    def generate_instances(
        recurring_event,
        start_date: datetime,
        end_date: datetime,
        max_instances: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Generate event instances for a recurring event within a date range

        Args:
            recurring_event: RecurringEvent model instance
            start_date: Start of the generation window
            end_date: End of the generation window
            max_instances: Maximum number of instances to generate

        Returns:
            List of event instance data dictionaries
        """
        if not recurring_event.rrule_freq:
            logger.warning(
                f"No RRULE frequency found for recurring event {recurring_event.recurring_event_id}"
            )
            return []

        try:
            # Build dateutil rrule from recurring event data
            rrule_obj = EventInstanceGenerator._build_rrule(
                recurring_event, start_date, end_date
            )

            # Generate base occurrences from RRULE
            base_occurrences = list(rrule_obj)

            # Add RDATE occurrences
            rdate_occurrences = EventInstanceGenerator._get_rdate_occurrences(
                recurring_event, start_date, end_date
            )

            # Combine and sort all occurrences
            all_occurrences = sorted(set(base_occurrences + rdate_occurrences))

            # Remove EXDATE occurrences
            filtered_occurrences = EventInstanceGenerator._filter_exdate_occurrences(
                all_occurrences, recurring_event
            )

            # Limit to max_instances
            if len(filtered_occurrences) > max_instances:
                filtered_occurrences = filtered_occurrences[:max_instances]
                logger.warning(
                    f"Limited instances to {max_instances} for recurring event {recurring_event.recurring_event_id}"
                )

            # Generate event instance data
            instances = []
            for occurrence_dt in filtered_occurrences:
                instance_data = EventInstanceGenerator._create_instance_data(
                    recurring_event, occurrence_dt
                )
                instances.append(instance_data)

            return instances

        except Exception as e:
            logger.error(
                f"Error generating instances for recurring event {recurring_event.recurring_event_id}: {e}"
            )
            raise RecurrenceParseError(f"Failed to generate event instances: {e}")

    @staticmethod
    def _build_rrule(recurring_event, start_date: datetime, end_date: datetime):
        """Build dateutil rrule object from RecurringEvent data"""

        # Get frequency
        freq = RecurrenceParser.FREQ_MAP[recurring_event.rrule_freq.value]

        # Start from DTSTART
        dtstart = recurring_event.dtstart

        # Build rrule parameters
        rrule_kwargs = {
            "freq": freq,
            "dtstart": dtstart,
            "interval": recurring_event.rrule_interval or 1,
        }

        # Add UNTIL or COUNT
        if recurring_event.rrule_until:
            rrule_kwargs["until"] = min(recurring_event.rrule_until, end_date)
        elif recurring_event.rrule_count:
            rrule_kwargs["count"] = recurring_event.rrule_count
        else:
            # Limit to end_date if no UNTIL or COUNT specified
            rrule_kwargs["until"] = end_date

        # Add BY* parameters
        if recurring_event.rrule_by_second:
            rrule_kwargs["bysecond"] = recurring_event.rrule_by_second
        if recurring_event.rrule_by_minute:
            rrule_kwargs["byminute"] = recurring_event.rrule_by_minute
        if recurring_event.rrule_by_hour:
            rrule_kwargs["byhour"] = recurring_event.rrule_by_hour
        if recurring_event.rrule_by_day:
            rrule_kwargs["byweekday"] = EventInstanceGenerator._parse_by_day_for_rrule(
                recurring_event.rrule_by_day
            )
        if recurring_event.rrule_by_monthday:
            rrule_kwargs["bymonthday"] = recurring_event.rrule_by_monthday
        if recurring_event.rrule_by_yearday:
            rrule_kwargs["byyearday"] = recurring_event.rrule_by_yearday
        if recurring_event.rrule_by_weekno:
            rrule_kwargs["byweekno"] = recurring_event.rrule_by_weekno
        if recurring_event.rrule_by_month:
            rrule_kwargs["bymonth"] = recurring_event.rrule_by_month
        if recurring_event.rrule_by_setpos:
            rrule_kwargs["bysetpos"] = recurring_event.rrule_by_setpos
        if recurring_event.rrule_wkst:
            rrule_kwargs["wkst"] = RecurrenceParser.WEEKDAY_MAP[
                recurring_event.rrule_wkst
            ]

        return rrule.rrule(**rrule_kwargs)

    @staticmethod
    def _parse_by_day_for_rrule(by_day_list: List[str]) -> List:
        """Convert BYDAY strings to dateutil weekday objects"""
        weekdays = []
        weekday_pattern = re.compile(r"^([+-]?\d*)([A-Z]{2})$")

        for by_day in by_day_list:
            match = weekday_pattern.match(by_day)
            if match:
                ordinal_str, weekday_str = match.groups()
                weekday_obj = RecurrenceParser.WEEKDAY_MAP[weekday_str]

                if ordinal_str:
                    ordinal = int(ordinal_str)
                    weekdays.append(weekday_obj(ordinal))
                else:
                    weekdays.append(weekday_obj)

        return weekdays

    @staticmethod
    def _get_rdate_occurrences(
        recurring_event, start_date: datetime, end_date: datetime
    ) -> List[datetime]:
        """Get RDATE occurrences within the date range"""
        if not recurring_event.rdate_list:
            return []

        rdate_occurrences = []
        for rdate_str in recurring_event.rdate_list:
            try:
                rdate_dt = date_parser.isoparse(rdate_str)
                if start_date <= rdate_dt <= end_date:
                    rdate_occurrences.append(rdate_dt)
            except Exception as e:
                logger.warning(f"Failed to parse RDATE '{rdate_str}': {e}")

        return rdate_occurrences

    @staticmethod
    def _filter_exdate_occurrences(
        occurrences: List[datetime], recurring_event
    ) -> List[datetime]:
        """Remove EXDATE occurrences from the list"""
        if not recurring_event.exdate_list:
            return occurrences

        exdates = set()
        for exdate_str in recurring_event.exdate_list:
            try:
                exdate_dt = date_parser.isoparse(exdate_str)
                exdates.add(exdate_dt)
            except Exception as e:
                logger.warning(f"Failed to parse EXDATE '{exdate_str}': {e}")

        return [occ for occ in occurrences if occ not in exdates]

    @staticmethod
    def _create_instance_data(
        recurring_event, occurrence_dt: datetime
    ) -> Dict[str, Any]:
        """Create event instance data from recurring event template and occurrence datetime"""

        # Calculate event duration from template
        duration = (
            recurring_event.template_end_datetime
            - recurring_event.template_start_datetime
        )

        # Calculate instance start and end times
        instance_start = occurrence_dt
        instance_end = occurrence_dt + duration

        # Build event instance data
        instance_data = {
            "recurring_event_id": recurring_event.recurring_event_id,
            "calendar_id": recurring_event.calendar_id,
            "user_id": recurring_event.user_id,
            "summary": recurring_event.summary,
            "description": recurring_event.description,
            "location": recurring_event.location,
            "start_datetime": instance_start,
            "end_datetime": instance_end,
            "start_timezone": recurring_event.template_start_timezone,
            "end_timezone": recurring_event.template_end_timezone,
            "status": recurring_event.status,
            "visibility": recurring_event.visibility,
            "color_id": recurring_event.color_id,
            "eventType": recurring_event.eventType,
            "guestsCanInviteOthers": recurring_event.guestsCanInviteOthers,
            "guestsCanModify": recurring_event.guestsCanModify,
            "guestsCanSeeOtherGuests": recurring_event.guestsCanSeeOtherGuests,
            "transparency": recurring_event.transparency,
            "privateCopy": recurring_event.privateCopy,
            "locked": recurring_event.locked,
            "sequence": recurring_event.sequence,
            "focusTimeProperties": recurring_event.focusTimeProperties,
            "outOfOfficeProperties": recurring_event.outOfOfficeProperties,
            "source": recurring_event.source,
            # Set original start time for recurring event instances
            "originalStartTime_dateTime": instance_start,
            "originalStartTime_timeZone": recurring_event.template_start_timezone,
            # Reference to parent recurring event
            "recurringEventId": recurring_event.recurring_event_id,
        }

        return instance_data
