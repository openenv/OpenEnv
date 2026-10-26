"""
Common models and response schemas for Calendar API
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response model"""

    success: bool
    message: Optional[str] = None
    timestamp: Optional[str] = None


class ErrorResponse(BaseResponse):
    """Error response model"""

    success: bool = False
    error: str
    error_code: Optional[str] = None


class SuccessResponse(BaseResponse):
    """Success response model"""

    success: bool = True
    data: Optional[Any] = None


class PaginationModel(BaseModel):
    """Pagination model"""

    limit: int = Field(
        default=100, ge=1, le=1000, description="Number of records to return"
    )
    offset: int = Field(default=0, ge=0, description="Number of records to skip")


class DatabaseStateResponse(BaseModel):
    """Database state response model"""

    success: bool
    service: str
    database_info: Dict[str, Any]
    table_counts: Dict[str, int]
    table_data: Dict[str, List[Dict[str, Any]]]
    timestamp: str


# Enums for calendar status and visibility
class EventStatus:
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class EventVisibility:
    DEFAULT = "default"
    PUBLIC = "public"
    PRIVATE = "private"
    CONFIDENTIAL = "confidential"


class CalendarAccessRole:
    OWNER = "owner"
    READER = "reader"
    WRITER = "writer"
    FREE_BUSY_READER = "freeBusyReader"


class EventTransparency:
    OPAQUE = "opaque"
    TRANSPARENT = "transparent"
