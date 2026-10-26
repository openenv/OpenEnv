"""
MCP (Model Context Protocol) API Router
Handles MCP protocol messages ONLY for Calendar API
Database management APIs are in apis.database_router
"""

import logging

from fastapi import APIRouter, Request, Response
from handlers.mcp_handler import handle_mcp_request

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/mcp")
async def handle_mcp(request: Request):
    """Handle MCP protocol messages"""
    response = await handle_mcp_request(request)
    if response is None:
        # For notifications, return 204 No Content
        return Response(content="", status_code=204, headers={"Content-Length": "0"})

    # Check if response is already a dict or needs to be converted
    if hasattr(response, "model_dump"):
        return response.model_dump()
    else:
        return response


# MCP protocol endpoint only - all other routes are at root level (no /api prefix)
