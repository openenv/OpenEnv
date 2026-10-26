"""
Core API endpoints
Handles health check and other core endpoints for Calendar API clone
"""

import logging

from fastapi import APIRouter

# Configure logging
logger = logging.getLogger(__name__)

# Create router for core APIs
router = APIRouter(tags=["core"])


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "calendar-env"}
