"""
CalendarList API endpoints following Google Calendar API v3 structure
Handles user's calendar list operations with exact Google API compatibility
"""

import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib.parse import urlencode, urlparse

from database.managers.calendar_list_manager import CalendarListManager
from database.session_manager import CalendarSessionManager
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from middleware.auth import get_user_context
from schemas.calendar_list import (
    CalendarListEntry,
    CalendarListInsertRequest,
    CalendarListPatchRequest,
    CalendarListResponse,
    CalendarListUpdateRequest,
    Channel,
    WatchRequest,
)
from schemas.common import SuccessResponse
from utils.validation import (
    set_colors_from_color_id,
    validate_and_set_color_id,
    validate_request_colors,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users/me/calendarList", tags=["calendarList"])

# Initialize managers
session_manager = CalendarSessionManager()


def get_calendar_list_manager(database_id: str) -> CalendarListManager:
    """Get calendar list manager for the specified database"""
    return CalendarListManager(database_id)


@router.get("", response_model=CalendarListResponse)
async def get_calendar_list(
    user_context: tuple[str, str] = Depends(get_user_context),
    maxResults: Optional[int] = Query(
        None,
        gt=0,
        le=250,
        description="Maximum number of entries returned (0-250). If 0, returns no items.",
    ),
    minAccessRole: Optional[str] = Query(
        None,
        description="Minimum access role filter",
        regex="^(freeBusyReader|reader|writer|owner)$",
    ),
    pageToken: Optional[str] = Query(None, description="Token for pagination"),
    showDeleted: Optional[bool] = Query(False, description="Include deleted calendars"),
    showHidden: Optional[bool] = Query(False, description="Include hidden calendars"),
    syncToken: Optional[str] = Query(None, description="Token for incremental sync"),
):
    """
    Returns the calendars on the user's calendar list

    GET /users/me/calendarList
    """
    try:
        database_id, user_id = user_context
        calendar_list_manager = get_calendar_list_manager(database_id)

        # Validate minAccessRole parameter
        if minAccessRole and minAccessRole not in [
            "freeBusyReader",
            "reader",
            "writer",
            "owner",
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid minAccessRole: {minAccessRole}. Must be one of: freeBusyReader, reader, writer, owner",
            )

        # Handle syncToken constraints
        if syncToken:
            # syncToken cannot be used with minAccessRole
            if minAccessRole:
                raise ValueError(
                    f"minAccessRole query parameter cannot be specified together with syncToken"
                )

            # When using syncToken, deleted and hidden entries must be included
            showDeleted = True
            showHidden = True

        # Get calendar entries with pagination and/or sync
        try:
            entries, next_page_token, next_sync_token = (
                calendar_list_manager.list_calendar_entries(
                    user_id=user_id,
                    max_results=maxResults,
                    min_access_role=minAccessRole,
                    show_deleted=showDeleted,
                    show_hidden=showHidden,
                    page_token=pageToken,
                    sync_token=syncToken,
                )
            )
        except ValueError as e:
            # Handle expired syncToken
            if "expired" in str(e).lower() or "invalid sync token" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_410_GONE,
                    detail="Sync token has expired. Client should clear storage and perform full synchronization.",
                )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        return CalendarListResponse(
            kind="calendar#calendarList",
            etag=f"etag-list-{database_id}",
            items=entries,
            nextPageToken=next_page_token,
            nextSyncToken=next_sync_token,
        )

    except ValueError as verr:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"{str(verr)}"
        )
    except Exception as e:
        logger.error(f"Error listing calendar list: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{calendarId}", response_model=CalendarListEntry)
async def get_calendar_from_list(
    calendarId: str, user_context: tuple[str, str] = Depends(get_user_context)
):
    """
    Returns a calendar from the user's calendar list

    GET /users/me/calendarList/{calendarId}
    """
    try:
        database_id, user_id = user_context
        calendar_list_manager = get_calendar_list_manager(database_id)

        # Support keyword 'primary' to fetch the user's primary calendar list entry
        if isinstance(calendarId, str) and calendarId.lower() == "primary":
            entries, _, _ = calendar_list_manager.list_calendar_entries(
                user_id=user_id, show_hidden=True
            )
            entry = next((e for e in entries if e.get("primary") is True), None)
            if entry:
                # Check ACL permissions for primary calendar
                try:
                    calendar_list_manager.check_calendar_acl_permissions(
                        user_id, entry["id"], ["reader", "writer", "owner"]
                    )
                except PermissionError:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"User '{user_id}' lacks permission on calendar '{entry['id']}'",
                    )
        else:
            # Check ACL permissions before getting calendar entry
            try:
                calendar_list_manager.check_calendar_acl_permissions(
                    user_id, calendarId, ["reader", "writer", "owner"]
                )
            except PermissionError:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"User '{user_id}' lacks permission on calendar '{calendarId}'",
                )
            entry = calendar_list_manager.get_calendar_entry(user_id, calendarId)

        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calendar not found: {calendarId}",
            )

        return entry

    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error getting calendar list entry {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("", response_model=CalendarListEntry, status_code=status.HTTP_201_CREATED)
async def add_calendar_to_list(
    calendar_request: CalendarListInsertRequest,
    colorRgbFormat: Optional[bool] = Query(
        False,
        description="Use RGB color fields when writing colors (backgroundColor/foregroundColor)",
    ),
    user_context: tuple[str, str] = Depends(get_user_context),
):
    """
    Inserts an existing calendar into the user's calendar list

    POST /users/me/calendarList
    """
    try:
        database_id, user_id = user_context
        calendar_list_manager = get_calendar_list_manager(database_id)

        # Convert request to dict
        entry_data = calendar_request.model_dump(exclude_none=True)

        # Validate colorId if provided (calendar list uses calendar colors)
        color_error = validate_request_colors(entry_data, "calendar", database_id)
        if color_error:
            raise ValueError(color_error)

        # Set backgroundColor and foregroundColor from colorId if they are not provided
        color_set_error = set_colors_from_color_id(entry_data, "calendar")
        if color_set_error:
            raise ValueError(color_set_error)

        # Normalize empty strings: drop to avoid triggering validation
        for key in ["summaryOverride", "colorId", "backgroundColor", "foregroundColor"]:
            if (
                key in entry_data
                and isinstance(entry_data[key], str)
                and entry_data[key].strip() == ""
            ):
                entry_data.pop(key)

        # Enforce colorRgbFormat rules for RGB color fields
        has_rgb_value = any(
            (
                k in entry_data
                and isinstance(entry_data[k], str)
                and entry_data[k].strip() != ""
            )
            for k in ["backgroundColor", "foregroundColor"]
        )
        if has_rgb_value:
            if not colorRgbFormat:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="To set backgroundColor/foregroundColor you must set colorRgbFormat=true",
                )
            # Validate hex color pattern #RRGGBB
            for key in ["backgroundColor", "foregroundColor"]:
                if (
                    key in entry_data
                    and isinstance(entry_data[key], str)
                    and entry_data[key].strip() != ""
                ):
                    val = entry_data[key]
                    if not re.match(r"^#[0-9A-Fa-f]{6}$", val):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid {key}: must be a hex color like #AABBCC",
                        )

            # Validate color combination and set colorId if both colors are provided
            if (
                "backgroundColor" in entry_data
                and entry_data["backgroundColor"]
                and "foregroundColor" in entry_data
                and entry_data["foregroundColor"]
            ):
                combination_error = validate_and_set_color_id(entry_data, "calendar")
                if combination_error:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=combination_error,
                    )
            # When RGB colors are provided but no colorId was found/set, remove any existing colorId
            elif "colorId" in entry_data:
                entry_data.pop("colorId")
        calendar_id = entry_data.pop("id")

        # Check ACL permissions before adding calendar to list
        try:
            calendar_list_manager.check_calendar_acl_permissions(
                user_id, calendar_id, ["reader", "writer", "owner"]
            )
        except PermissionError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User '{user_id}' lacks permission on calendar '{calendar_id}'",
            )

        # Enforce Google spec: required fields when ADDING reminders/notifications
        allowed_reminder_methods = {"email", "popup"}
        allowed_notification_types = {
            "eventCreation",
            "eventChange",
            "eventCancellation",
            "eventResponse",
            "agenda",
        }

        # defaultReminders: each item must include method (enum) and minutes (>=0)
        if (
            "defaultReminders" in entry_data
            and entry_data["defaultReminders"] is not None
        ):
            dr = entry_data["defaultReminders"]
            if not isinstance(dr, list):
                raise HTTPException(
                    status_code=400, detail="defaultReminders must be a list"
                )
            for idx, item in enumerate(dr):
                if not isinstance(item, dict):
                    raise HTTPException(
                        status_code=400,
                        detail=f"defaultReminders[{idx}] must be an object",
                    )
                method = (item.get("method") or "").strip()
                minutes = item.get("minutes")
                if method == "" or method not in allowed_reminder_methods:
                    raise HTTPException(
                        status_code=400,
                        detail="defaultReminders[].method is required when adding and must be 'email' or 'popup'",
                    )
                if not isinstance(minutes, int) or minutes < 0:
                    raise HTTPException(
                        status_code=400,
                        detail="defaultReminders[].minutes is required when adding and must be >= 0",
                    )

        # notificationSettings.notifications: each item must include method ('email') and type (enum)
        if (
            "notificationSettings" in entry_data
            and entry_data["notificationSettings"] is not None
        ):
            ns = entry_data["notificationSettings"]
            if not isinstance(ns, dict):
                raise HTTPException(
                    status_code=400, detail="notificationSettings must be an object"
                )
            notifs = ns.get("notifications")
            if notifs is not None:
                if not isinstance(notifs, list):
                    raise HTTPException(
                        status_code=400,
                        detail="notificationSettings.notifications must be a list",
                    )
                for idx, n in enumerate(notifs):
                    if not isinstance(n, dict):
                        raise HTTPException(
                            status_code=400,
                            detail=f"notificationSettings.notifications[{idx}] must be an object",
                        )
                    method = (n.get("method") or "").strip()
                    ntype = (n.get("type") or "").strip()
                    if method == "" or method != "email":
                        raise HTTPException(
                            status_code=400,
                            detail="notificationSettings.notifications[].method is required when adding and must be 'email'",
                        )
                    if ntype == "" or ntype not in allowed_notification_types:
                        raise HTTPException(
                            status_code=400,
                            detail="notificationSettings.notifications[].type is required when adding and must be a valid type",
                        )

        # Insert calendar into list
        entry = calendar_list_manager.insert_calendar_entry(
            user_id, calendar_id, entry_data
        )

        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calendar not found: {calendar_id}",
            )

        return entry

    except ValueError as verr:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"{str(verr)}"
        )
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error inserting calendar list entry: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.patch("/{calendarId}", response_model=CalendarListEntry)
async def update_calendar_in_list(
    calendarId: str,
    calendar_request: CalendarListUpdateRequest,
    colorRgbFormat: Optional[bool] = Query(
        False,
        description="Use RGB color fields when writing colors (backgroundColor/foregroundColor)",
    ),
    user_context: tuple[str, str] = Depends(get_user_context),
):
    """
    Updates an entry on the user's calendar list (partial update)

    PATCH /users/me/calendarList/{calendarId}
    """
    try:
        database_id, user_id = user_context
        calendar_list_manager = get_calendar_list_manager(database_id)

        # Support keyword 'primary' for calendarId (resolve to actual primary calendar ID)
        if isinstance(calendarId, str) and calendarId.lower() == "primary":
            entries, _, _ = calendar_list_manager.list_calendar_entries(
                user_id=user_id, show_hidden=True
            )
            primary_entry = next((e for e in entries if e.get("primary") is True), None)
            if not primary_entry:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Primary calendar not found",
                )
            calendarId = primary_entry["id"]

        # Check ACL permissions before updating calendar list entry
        try:
            calendar_list_manager.check_calendar_acl_permissions(
                user_id, calendarId, ["writer", "owner"]
            )
        except PermissionError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User '{user_id}' lacks permission on calendar '{calendarId}'",
            )

        # Convert request to dict for PATCH
        # Use exclude_unset=True so explicitly provided nulls are preserved (to allow clearing fields)
        update_data = calendar_request.model_dump(exclude_unset=True)

        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields provided for update",
            )

        # Validate colorId if provided (calendar list uses calendar colors)
        color_error = validate_request_colors(update_data, "calendar", database_id)
        if color_error:
            raise ValueError(color_error)

        # Set backgroundColor and foregroundColor from colorId if they are not provided
        color_set_error = set_colors_from_color_id(update_data, "calendar")
        if color_set_error:
            raise ValueError(color_set_error)

        # Reject null values for boolean fields in PATCH
        if "hidden" in update_data and update_data["hidden"] is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="'hidden' cannot be null in PATCH",
            )
        if "selected" in update_data and update_data["selected"] is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="'selected' cannot be null in PATCH",
            )

        # Normalize empty strings for PATCH: treat as omitted so they don't trigger validation
        for key in ["summaryOverride", "colorId", "backgroundColor", "foregroundColor"]:
            if (
                key in update_data
                and isinstance(update_data[key], str)
                and update_data[key].strip() == ""
            ):
                update_data.pop(key)

        # Enforce colorRgbFormat rules for RGB color fields
        has_rgb_value = any(
            (
                k in update_data
                and isinstance(update_data[k], str)
                and update_data[k].strip() != ""
            )
            for k in ["backgroundColor", "foregroundColor"]
        )
        if has_rgb_value:
            if not colorRgbFormat:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="To set backgroundColor/foregroundColor you must set colorRgbFormat=true",
                )
            # Validate hex color pattern #RRGGBB
            for key in ["backgroundColor", "foregroundColor"]:
                if (
                    key in update_data
                    and isinstance(update_data[key], str)
                    and update_data[key].strip() != ""
                ):
                    val = update_data[key]
                    if not re.match(r"^#[0-9A-Fa-f]{6}$", val):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid {key}: must be a hex color like #AABBCC",
                        )

            # Validate color combination and set colorId if both colors are provided
            if (
                "backgroundColor" in update_data
                and update_data["backgroundColor"]
                and "foregroundColor" in update_data
                and update_data["foregroundColor"]
            ):
                combination_error = validate_and_set_color_id(update_data, "calendar")
                if combination_error:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=combination_error,
                    )
            # When RGB colors are provided but no colorId was found/set, remove any existing colorId
            elif "colorId" in update_data:
                update_data.pop("colorId")

        # Set backgroundColor and foregroundColor from colorId if they are not provided
        color_set_error = set_colors_from_color_id(update_data, "calendar")
        if color_set_error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=color_set_error
            )

        # Coupling rule: only enforce selected=false when hidden=true.
        # When hidden=false, do not override selected; honor client's value if provided.
        if "hidden" in update_data and update_data["hidden"] is True:
            update_data["selected"] = False
        if (
            "hidden" in update_data
            and update_data["hidden"] is False
            and "selected" not in update_data
        ):
            update_data["selected"] = True

        # Sanitize nested lists for PATCH: drop invalid/empty reminder/notification items; clear when empty arrays provided
        if "defaultReminders" in update_data:
            dr = update_data["defaultReminders"]
            if isinstance(dr, list):
                cleaned = []
                for item in dr:
                    if not isinstance(item, dict):
                        continue
                    method = (item.get("method") or "").strip()
                    minutes = item.get("minutes")
                    if method == "":
                        # treat as cleared; skip item
                        continue
                    if (
                        method in {"email", "popup"}
                        and isinstance(minutes, int)
                        and minutes >= 0
                    ):
                        cleaned.append({"method": method, "minutes": minutes})
                update_data["defaultReminders"] = cleaned if cleaned else None
            elif dr in (None, ""):
                update_data["defaultReminders"] = None

        if "notificationSettings" in update_data:
            ns = update_data["notificationSettings"]
            if isinstance(ns, dict):
                notifs = ns.get("notifications")
                if isinstance(notifs, list):
                    cleaned_n = []
                    for n in notifs:
                        if not isinstance(n, dict):
                            continue
                        method = (n.get("method") or "").strip()
                        ntype = (n.get("type") or "").strip()
                        if method == "" and ntype == "":
                            # cleared
                            continue
                        if method == "email" and ntype in {
                            "eventCreation",
                            "eventChange",
                            "eventCancellation",
                            "eventResponse",
                            "agenda",
                        }:
                            cleaned_n.append({"method": method, "type": ntype})
                    update_data["notificationSettings"] = (
                        {"notifications": cleaned_n} if cleaned_n else None
                    )
                elif notifs in (None, ""):
                    update_data["notificationSettings"] = None
            elif ns in (None, ""):
                update_data["notificationSettings"] = None

        # Update calendar entry
        entry = calendar_list_manager.update_calendar_entry(
            user_id, calendarId, update_data, is_patch=True
        )

        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calendar not found: {calendarId}",
            )

        return entry

    except ValueError as verr:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"{str(verr)}"
        )
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error updating calendar list entry {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put("/{calendarId}", response_model=CalendarListEntry)
async def replace_calendar_in_list(
    calendarId: str,
    calendar_request: CalendarListPatchRequest,
    colorRgbFormat: Optional[bool] = Query(
        False,
        description="Use RGB color fields when writing colors (backgroundColor/foregroundColor)",
    ),
    user_context: tuple[str, str] = Depends(get_user_context),
):
    """
    Updates an entry on the user's calendar list (full update)

    PUT /users/me/calendarList/{calendarId}
    """
    try:
        database_id, user_id = user_context
        calendar_list_manager = get_calendar_list_manager(database_id)

        # Support keyword 'primary' for calendarId (resolve to actual primary calendar ID)
        if isinstance(calendarId, str) and calendarId.lower() == "primary":
            entries, _, _ = calendar_list_manager.list_calendar_entries(
                user_id=user_id, show_hidden=True
            )
            primary_entry = next((e for e in entries if e.get("primary") is True), None)
            if not primary_entry:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Primary calendar not found",
                )
            calendarId = primary_entry["id"]

        # Check ACL permissions before updating calendar list entry
        try:
            calendar_list_manager.check_calendar_acl_permissions(
                user_id, calendarId, ["writer", "owner"]
            )
        except PermissionError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User '{user_id}' lacks permission on calendar '{calendarId}'",
            )

        # Convert request to dict, including None values for full update
        update_data = calendar_request.model_dump()

        # Validate colorId if provided (calendar list uses calendar colors)
        color_error = validate_request_colors(update_data, "calendar", database_id)
        if color_error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=color_error
            )

        # Set backgroundColor and foregroundColor from colorId if they are not provided
        color_set_error = set_colors_from_color_id(update_data, "calendar")
        if color_set_error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=color_set_error
            )

        # Ensure NOT NULL fields have defaults for PUT requests
        if "hidden" not in update_data or update_data["hidden"] is None:
            update_data["hidden"] = False
        if "selected" not in update_data or update_data["selected"] is None:
            update_data["selected"] = True

        # Normalize empty strings to None for PUT (clear fields)
        for key in ["summaryOverride", "colorId", "backgroundColor", "foregroundColor"]:
            if (
                key in update_data
                and isinstance(update_data[key], str)
                and update_data[key].strip() == ""
            ):
                update_data[key] = None

        # Enforce colorRgbFormat rules for RGB color fields
        has_rgb_value = any(
            (
                k in update_data
                and isinstance(update_data[k], str)
                and update_data[k].strip() != ""
            )
            for k in ["backgroundColor", "foregroundColor"]
        )
        if has_rgb_value:
            if not colorRgbFormat:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="To set backgroundColor/foregroundColor you must set colorRgbFormat=true",
                )
            # Validate hex color pattern #RRGGBB
            for key in ["backgroundColor", "foregroundColor"]:
                if (
                    key in update_data
                    and isinstance(update_data[key], str)
                    and update_data[key].strip() != ""
                ):
                    val = update_data[key]
                    if not re.match(r"^#[0-9A-Fa-f]{6}$", val):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid {key}: must be a hex color like #AABBCC",
                        )

            # Validate color combination and set colorId if both colors are provided
            if (
                "backgroundColor" in update_data
                and update_data["backgroundColor"]
                and "foregroundColor" in update_data
                and update_data["foregroundColor"]
            ):
                combination_error = validate_and_set_color_id(update_data, "calendar")
                if combination_error:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=combination_error,
                    )
            # When RGB colors are provided but no colorId was found/set, remove any existing colorId
            elif "colorId" in update_data:
                update_data.pop("colorId")

        if "hidden" in update_data and update_data["hidden"] is True:
            update_data["selected"] = False
        if (
            "hidden" in update_data
            and update_data["hidden"] is False
            and "selected" not in update_data
        ):
            update_data["selected"] = True

        # Sanitize nested lists for PUT: drop invalid/empty items; empty array clears
        if "defaultReminders" in update_data:
            dr = update_data["defaultReminders"]
            if isinstance(dr, list):
                cleaned = []
                for item in dr:
                    if not isinstance(item, dict):
                        continue
                    method = (item.get("method") or "").strip()
                    minutes = item.get("minutes")
                    if method == "":
                        continue
                    if (
                        method in {"email", "popup"}
                        and isinstance(minutes, int)
                        and minutes >= 0
                    ):
                        cleaned.append({"method": method, "minutes": minutes})
                update_data["defaultReminders"] = cleaned if cleaned else None
            elif dr in (None, ""):
                update_data["defaultReminders"] = None

        if "notificationSettings" in update_data:
            ns = update_data["notificationSettings"]
            if isinstance(ns, dict):
                notifs = ns.get("notifications")
                if isinstance(notifs, list):
                    cleaned_n = []
                    for n in notifs:
                        if not isinstance(n, dict):
                            continue
                        method = (n.get("method") or "").strip()
                        ntype = (n.get("type") or "").strip()
                        if method == "" and ntype == "":
                            continue
                        if method == "email" and ntype in {
                            "eventCreation",
                            "eventChange",
                            "eventCancellation",
                            "eventResponse",
                            "agenda",
                        }:
                            cleaned_n.append({"method": method, "type": ntype})
                    update_data["notificationSettings"] = (
                        {"notifications": cleaned_n} if cleaned_n else None
                    )
                elif notifs in (None, ""):
                    update_data["notificationSettings"] = None
            elif ns in (None, ""):
                update_data["notificationSettings"] = None

        # Update calendar entry
        entry = calendar_list_manager.update_calendar_entry(
            user_id, calendarId, update_data, is_patch=False
        )

        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calendar not found: {calendarId}",
            )

        return entry

    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error updating calendar list entry {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/{calendarId}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_calendar_from_list(
    calendarId: str, user_context: tuple[str, str] = Depends(get_user_context)
):
    """
    Removes a calendar from the user's calendar list

    DELETE /users/me/calendarList/{calendarId}
    """
    try:
        database_id, user_id = user_context
        calendar_list_manager = get_calendar_list_manager(database_id)

        # Support keyword 'primary' for calendarId (resolve to actual primary calendar ID)
        if isinstance(calendarId, str) and calendarId.lower() == "primary":
            entries, _, _ = calendar_list_manager.list_calendar_entries(
                user_id=user_id, show_hidden=True
            )
            primary_entry = next((e for e in entries if e.get("primary") is True), None)
            if not primary_entry:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Primary calendar not found",
                )
            calendarId = primary_entry["id"]

        # Check ACL permissions before removing calendar from list
        try:
            calendar_list_manager.check_calendar_acl_permissions(
                user_id, calendarId, ["owner"]
            )
        except PermissionError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User '{user_id}' lacks permission on calendar '{calendarId}'",
            )

        # Check if calendar exists first
        existing_entry = calendar_list_manager.get_calendar_entry(user_id, calendarId)
        if not existing_entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calendar not found: {calendarId}",
            )

        # Delete calendar entry
        try:
            success = calendar_list_manager.delete_calendar_entry(user_id, calendarId)

            if not success:
                raise HTTPException(
                    status_code=500, detail="Failed to remove calendar from list"
                )
        except ValueError as e:
            # Handle primary calendar removal attempt
            if "Cannot remove primary calendar" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot remove primary calendar from calendar list",
                )
            raise

        # Return 204 No Content (no response body)
        return None

    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error deleting calendar list entry {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/watch")
async def watch_calendar_list(
    watch_request: WatchRequest,
    user_context: tuple[str, str] = Depends(get_user_context),
):
    """
    Watch for changes to CalendarList resources

    POST /users/me/calendarList/watch
    """
    try:
        database_id, user_id = user_context
        calendar_list_manager = get_calendar_list_manager(database_id)

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

        # Create the watch channel
        channel = calendar_list_manager.watch_calendar_list(watch_request, user_id)
        # Return Channel response
        return channel

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up calendar list watch: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
