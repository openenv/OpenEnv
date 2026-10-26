"""
Settings models for Calendar Settings API following Google Calendar API v3 structure
"""

from typing import Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Extra, Field, field_validator


class SettingItem(BaseModel):
    """Single settings resource (Google Calendar API v3 style)"""

    model_config = {"from_attributes": True}

    etag: Optional[str] = Field(None, description="ETag of the settings")
    id: str = Field(..., description="ID of the settings (e.g., timezone)")
    value: str = Field(..., description="Value of the settings")
    user_id: str = Field(..., description="User ID to whom this setting belongs")


class SettingsListResponse(BaseModel):
    """Paginated list of settings (Google Calendar API v3 style)"""

    etag: Optional[str] = Field(None, description="ETag of the collection")
    items: List[SettingItem] = Field(
        default_factory=list, description="List of settings resources"
    )
    nextPageToken: Optional[str] = Field(
        None, description="Token for next page if pagination is supported"
    )


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


class SettingsWatchRequest(BaseModel):
    """Request model for watching settings changes"""

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

    class Config:
        extra = "ignore"
