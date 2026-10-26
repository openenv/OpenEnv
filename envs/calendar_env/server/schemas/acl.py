import re
from enum import Enum
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Extra, Field, field_validator, model_validator


class ScopeType(str, Enum):
    default = "default"
    user = "user"
    group = "group"
    domain = "domain"


class AclRole(str, Enum):
    none = "none"
    freeBusyReader = "freeBusyReader"
    reader = "reader"
    writer = "writer"
    owner = "owner"


class ScopeInput(BaseModel):
    type: ScopeType
    value: Optional[str] = Field(
        None,
        description="The email address of a user or group, or the name of a domain, depending on the scope type. Omitted for type 'default'.",
    )  # Optional only for default

    @model_validator(mode="after")
    def validate_scope_value(self) -> "ScopeInput":
        """
        Validate scope value based on type:
        - default: value must be None/omitted
        - user/group: value must be a valid email address
        - domain: value must be a valid domain name
        """
        if self.type == ScopeType.default:
            if self.value is not None:
                raise ValueError("scope.value must be omitted for type 'default'")
        else:
            value = self.value.strip()

            if self.type in (ScopeType.user, ScopeType.group):
                # Validate email address
                email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                if not re.match(email_pattern, value):
                    raise ValueError(
                        f"scope.value must be a valid email address for type '{self.type.value}'"
                    )

            elif self.type == ScopeType.domain:
                # Validate domain name
                domain_pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
                if not re.match(domain_pattern, value) or len(value) > 253:
                    raise ValueError(
                        f"scope.value must be a valid domain name for type '{self.type.value}'"
                    )

        return self


class ACLRuleInput(BaseModel):
    scope: ScopeInput
    role: Optional[AclRole] = Field(None, description="The role assigned to the scope")


class PatchACLRuleInput(BaseModel):
    scope: Optional[ScopeInput] = Field(
        None, description="The extent to which calenda access is granted"
    )
    role: Optional[AclRole] = Field(None, description="The role assigned to the scope")


class ScopeOutput(BaseModel):
    type: ScopeType
    value: Optional[str] = Field(
        None,
        description="The email address of a user or group, or the name of a domain, depending on the scope type. Omitted for type 'default'.",
    )  # Optional only for default


class ACLRule(BaseModel):
    id: str
    calendar_id: str
    user_id: str
    role: AclRole
    etag: str
    scope: ScopeOutput


class InsertACLRule(BaseModel):
    kind: str
    etag: str
    id: str
    scope: ScopeInput
    role: str


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


class ACLWatchRequest(BaseModel):
    """Request model for watching ACL changes"""

    id: str = Field(..., description="Channel identifier")
    type: str = Field(default="web_hook", description="Channel type")
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
            raise ValueError("Only channel type 'web_hook' is supported")
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


class NotificationEvent(BaseModel):
    """Model for notification event payload"""

    kind: str = Field(default="api#channel", description="Resource type identifier")
    id: str = Field(..., description="Channel identifier")
    resourceId: str = Field(..., description="Resource ID")
    resourceUri: str = Field(..., description="Resource URI")
    eventType: str = Field(..., description="Type of event (insert, update, delete)")
    resourceState: str = Field(default="sync", description="State of the resource")
    timestamp: str = Field(..., description="Event timestamp")
    data: Dict[str, Any] = Field(..., description="The actual data that changed")


class WatchChannelInfo(BaseModel):
    """Information about a watch channel"""

    id: str = Field(..., description="Channel identifier")
    resource_id: str = Field(..., description="Resource ID")
    resource_uri: str = Field(..., description="Resource URI")
    calendar_id: str = Field(..., description="Calendar ID")
    webhook_address: str = Field(..., description="Webhook address")
    webhook_type: str = Field(..., description="Webhook type")
    created_at: str = Field(..., description="Creation timestamp")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")
    is_active: str = Field(..., description="Whether channel is active")
    notification_count: int = Field(..., description="Number of notifications sent")


class ACLListResponse(BaseModel):
    """Response model for paginated ACL list"""

    kind: str = Field(default="calendar#acl", description="Resource type identifier")
    etag: str = Field(..., description="ETag for the collection")
    items: list[ACLRule] = Field(default_factory=list, description="List of ACL rules")
    nextPageToken: Optional[str] = Field(None, description="Token for next page")
    nextSyncToken: Optional[str] = Field(
        None, description="Token for next sync operation"
    )
