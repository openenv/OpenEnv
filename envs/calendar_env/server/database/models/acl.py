import enum
from datetime import datetime

from sqlalchemy import Column, DateTime, Enum, ForeignKey, String
from sqlalchemy.orm import relationship

from .base import Base


class AclRole(enum.Enum):
    none = "none"
    freeBusyReader = "freeBusyReader"
    reader = "reader"
    writer = "writer"
    owner = "owner"


class ScopeType(enum.Enum):
    default = "default"
    user = "user"
    group = "group"
    domain = "domain"


class Scope(Base):
    __tablename__ = "scopes"

    id = Column(String, primary_key=True, index=True)
    type = Column(Enum(ScopeType), nullable=False)
    value = Column(String, nullable=True)  # Optional for default

    # Relationship to ACLs
    acls = relationship("ACLs", back_populates="scope", cascade="all, delete-orphan")

    def as_dict(self):
        return {"type": self.type.value, "value": self.value}


class ACLs(Base):
    __tablename__ = "acls"

    id = Column(String, primary_key=True, index=True)

    calendar_id = Column(
        String, ForeignKey("calendars.calendar_id"), nullable=False, index=True
    )
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    scope_id = Column(String, ForeignKey("scopes.id"), nullable=False)

    role = Column(Enum(AclRole), nullable=False)

    etag = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    calendar = relationship("Calendar", back_populates="acls")
    user = relationship("User", back_populates="acls")
    scope = relationship("Scope", back_populates="acls")
