"""
FreeBusy API endpoints following Google Calendar API v3 structure
Handles FreeBusy query operations with exact Google API compatibility
"""

import logging
from typing import Optional

from database.managers.freebusy_manager import FreeBusyManager
from database.session_manager import CalendarSessionManager
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from middleware.auth import get_user_context
from pydantic import ValidationError
from schemas.freebusy import FreeBusyQueryRequest, FreeBusyQueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/freebusy", tags=["freebusy"])

# Initialize managers
session_manager = CalendarSessionManager()


def get_freebusy_manager(database_id: str) -> FreeBusyManager:
    """Get freebusy manager for the specified database"""
    return FreeBusyManager(database_id)


@router.post("/query", response_model=FreeBusyQueryResponse)
async def query_freebusy(
    request: FreeBusyQueryRequest,
    user_context: tuple[str, str] = Depends(get_user_context),
):
    """
    Returns free/busy information for a set of calendars

    POST /freeBusy
    """
    try:
        database_id, user_id = user_context
        freebusy_manager = get_freebusy_manager(database_id)

        # Validate request
        if not request.items:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one calendar item is required",
            )

        if len(request.items) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 50 calendars allowed per query",
            )

        response = freebusy_manager.query_freebusy(user_id, request)

        return response

    except ValidationError as e:
        # Handle pydantic validation errors (timezone, schema validation)
        logger.error(f"Schema validation error in FreeBusy query: {e}")
        error_messages = []
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            error_messages.append(f"{field}: {msg}")
        detail = "Validation failed: " + "; ".join(error_messages)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
    except ValueError as e:
        # Handle business logic validation errors (calendar ID existence, time range, etc.)
        logger.error(f"Business validation error in FreeBusy query: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error processing FreeBusy query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred while processing the request",
        )
