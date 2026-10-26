"""
MCP Tools Module - Calendar API Tools

This module aggregates all MCP tools for the Calendar API endpoints.
Includes calendar management tools:
- Calendars (create, read, update, delete, clear, list)
- CalendarList (list, get, insert, patch, put, delete, watch)
- Events (list, create, get, patch, update, delete, move, quickAdd, import, instances, watch)
- Colors (get color definitions)
- FreeBusy (query)
- ACL (access control list management)
- Users (user management)
- Settings (calendar settings)

Each tool corresponds to Calendar API endpoints
"""

# Import calendar tool categories
from .acl import ACL_TOOLS
from .calendar_list import CALENDAR_LIST_TOOLS
from .calendars import CALENDARS_TOOLS
from .colors import COLORS_TOOLS
from .events import EVENTS_TOOLS
from .freebusy import FREEBUSY_TOOLS
from .settings import SETTINGS_TOOLS
from .users import USERS_TOOLS

# Combine all tools into the main MCP_TOOLS list
MCP_TOOLS = []
MCP_TOOLS.extend(CALENDARS_TOOLS)
MCP_TOOLS.extend(CALENDAR_LIST_TOOLS)
MCP_TOOLS.extend(EVENTS_TOOLS)
MCP_TOOLS.extend(COLORS_TOOLS)
MCP_TOOLS.extend(USERS_TOOLS)
MCP_TOOLS.extend(SETTINGS_TOOLS)
MCP_TOOLS.extend(ACL_TOOLS)
MCP_TOOLS.extend(FREEBUSY_TOOLS)


print(
    f"📦 MCP Tools Module Loaded: {len(MCP_TOOLS)} calendar API tools across 8 modules"
)
