"""
Calendar API endpoints following Google Calendar API v3 structure
Handles CRUD operations for calendars
"""

import logging
from typing import List

from database.managers.calendar_manager import CalendarManager
from database.session_manager import CalendarSessionManager
from fastapi import APIRouter, Body, Depends, HTTPException, status
from middleware.auth import get_user_context
from schemas.calendar import Calendar, CalendarCreateRequest, CalendarUpdateRequest
from schemas.common import SuccessResponse
from utils.validation import validate_request_colors

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/calendars", tags=["calendars"])

# Initialize managers
session_manager = CalendarSessionManager()


def get_calendar_manager(database_id: str) -> CalendarManager:
    """Get calendar manager for the specified database"""
    return CalendarManager(database_id)


def check_acl_permissions(
    manager: CalendarManager, user_id: str, calendar_id: str, allowed_roles: List[str]
):
    calendar = manager.get_calendar_by_id(user_id, calendar_id, allowed_roles)
    if not calendar:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user_id}' lacks permission on calendar '{calendar_id}'",
        )
    return calendar


@router.get("/{calendarId}", response_model=Calendar)
async def get_calendar(
    calendarId: str, user_context: tuple[str, str] = Depends(get_user_context)
):
    """
    Returns metadata for a calendar

    GET /calendars/{calendarId}
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)

        if calendarId.lower() == "primary":
            calendar = calendar_manager.get_primary_calendar(user_id)
            if not calendar:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User '{user_id}' has no primary calendar",
                )
            formatted_response = calendar_manager._format_calendar_response(calendar)
            return Calendar(**formatted_response)

        calendar = check_acl_permissions(
            calendar_manager, user_id, calendarId, ["reader", "writer", "owner"]
        )
        formatted_response = calendar_manager._format_calendar_response(calendar)
        return Calendar(**formatted_response)
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error getting calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.patch("/{calendarId}", response_model=Calendar)
def patch_calendar(
    calendarId: str,
    update: CalendarUpdateRequest = Body(...),
    user_context: tuple[str, str] = Depends(get_user_context),
):
    """
    Updates calendar metadata (partial update)

    PATCH /calendars/{calendarId}
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)

        update_data = update.model_dump(exclude_none=True)
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields provided for patch",
            )

        color_error = validate_request_colors(update_data, "calendar", database_id)
        if color_error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=color_error
            )

        resolved_calendar_id = calendarId
        if calendarId.lower() == "primary":
            primary_calendar = calendar_manager.get_primary_calendar(user_id)
            if not primary_calendar:
                raise HTTPException(
                    status_code=404, detail="Primary calendar not found"
                )
            resolved_calendar_id = primary_calendar.calendar_id

        check_acl_permissions(
            calendar_manager, user_id, resolved_calendar_id, ["writer", "owner"]
        )

        updated_calendar = calendar_manager.update_calendar(
            user_id, resolved_calendar_id, update_data
        )
        if not updated_calendar:
            raise HTTPException(status_code=404, detail="Calendar not found")

        calendar_manager.session.refresh(updated_calendar)
        # Format the response to match the Pydantic schema
        formatted_response = calendar_manager._format_calendar_response(
            updated_calendar
        )
        return Calendar(**formatted_response)
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error updating calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put("/{calendarId}", response_model=Calendar)
async def update_calendar(
    calendarId: str,
    calendar_request: CalendarUpdateRequest,
    user_context: tuple[str, str] = Depends(get_user_context),
):
    """
    Updates calendar metadata (full update)

    PUT /calendars/{calendarId}
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)

        update_data = calendar_request.model_dump(exclude_unset=True, exclude_none=True)
        color_error = validate_request_colors(update_data, "calendar", database_id)
        if color_error:
            raise HTTPException(status_code=400, detail=color_error)

        resolved_calendar_id = calendarId
        if calendarId.lower() == "primary":
            primary_calendar = calendar_manager.get_primary_calendar(user_id)
            if not primary_calendar:
                raise HTTPException(
                    status_code=404, detail="Primary calendar not found"
                )
            resolved_calendar_id = primary_calendar.calendar_id

        check_acl_permissions(
            calendar_manager, user_id, resolved_calendar_id, ["writer", "owner"]
        )

        logger.debug(f"Update data: {update_data}")
        updated_calendar = calendar_manager.update_calendar(
            user_id, resolved_calendar_id, update_data
        )
        if not updated_calendar:
            raise HTTPException(status_code=404, detail="Calendar not found")

        calendar_manager.session.refresh(updated_calendar)

        # Format the response to match the Pydantic schema
        formatted_response = calendar_manager._format_calendar_response(
            updated_calendar
        )
        return Calendar(**formatted_response)
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error updating calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("", response_model=Calendar, status_code=status.HTTP_201_CREATED)
async def create_calendar(
    calendar_request: CalendarCreateRequest,
    user_context: tuple[str, str] = Depends(get_user_context),
):
    """
    Creates a secondary calendar

    POST /calendars
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)

        from database.managers.user_manager import UserManager

        user_manager = UserManager(database_id)
        user = user_manager.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        calendar_data = calendar_request.model_dump(exclude_unset=True)
        color_error = validate_request_colors(calendar_data, "calendar", database_id)
        if color_error:
            raise HTTPException(status_code=400, detail=color_error)

        created_calendar = calendar_manager.create_calendar(user_id, calendar_data)
        if not created_calendar:
            raise HTTPException(status_code=500, detail="Failed to create calendar")

        # Format the response to match the Pydantic schema
        formatted_response = calendar_manager._format_calendar_response(
            created_calendar
        )
        return Calendar(**formatted_response)
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error creating calendar: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/{calendarId}")
def delete_calendar(
    calendarId: str, user_context: tuple[str, str] = Depends(get_user_context)
):
    """
    Deletes a secondary calendar

    DELETE /calendars/{calendarId}
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)

        resolved_calendar_id = calendarId
        if calendarId.lower() == "primary":
            primary_calendar = calendar_manager.get_primary_calendar(user_id)
            if not primary_calendar:
                raise HTTPException(
                    status_code=404, detail="Primary calendar not found"
                )
            resolved_calendar_id = primary_calendar.calendar_id

        # Check ACL permissions before attempting to delete
        check_acl_permissions(
            calendar_manager, user_id, resolved_calendar_id, ["owner"]
        )

        deleted = calendar_manager.delete_calendar(user_id, resolved_calendar_id)
        if not deleted:
            raise HTTPException(
                status_code=404, detail="Calendar not found or already deleted"
            )

        return {"message": "Calendar deleted successfully"}
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error deleting calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/{calendarId}/clear", response_model=int)
def clear_calendar(
    calendarId: str, user_context: tuple[str, str] = Depends(get_user_context)
):
    """
    Clears a primary calendar (deletes all events)

    POST /calendars/{calendarId}/clear
    """
    try:
        database_id, user_id = user_context
        calendar_manager = get_calendar_manager(database_id)

        resolved_calendar_id = calendarId
        if calendarId.lower() == "primary":
            primary_calendar = calendar_manager.get_primary_calendar(user_id)
            if not primary_calendar:
                raise HTTPException(
                    status_code=404, detail="Primary calendar not found"
                )
            resolved_calendar_id = primary_calendar.calendar_id

        # Check ACL permissions before attempting to clear
        check_acl_permissions(
            calendar_manager, user_id, resolved_calendar_id, ["owner", "writer"]
        )

        cleared_count = calendar_manager.clear_calendar(user_id, resolved_calendar_id)
        return cleared_count
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error clearing calendar {calendarId}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
