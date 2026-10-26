"""
Simple middleware for multi-user support
"""

import logging

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)


def authenticate_user_with_token(
    x_database_id: str = Header(alias="x-database-id"),
    x_access_token: str = Header(alias="x-access-token"),
) -> tuple[str, str]:
    """
    Authenticate user with access token and return database_id and user_id.
    """
    from database.managers.user_manager import UserManager

    try:
        # Validate header format
        missing_headers = []
        if not x_database_id:
            missing_headers.append("x-database-id")
        if not x_access_token:
            missing_headers.append("x-access-token")

        if missing_headers:
            detail = f"Missing required headers: {', '.join(missing_headers)}"
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

        # Get user manager and authenticate with token
        user_manager = UserManager(x_database_id)
        user_info = user_manager.get_user_by_access_token(x_access_token)

        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token"
            )

        if not user_info.get("is_active", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive",
            )

        return x_database_id, user_info["id"]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error authenticating user with token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication error",
        )


def get_user_context(
    x_database_id: str = Header(alias="x-database-id"),
    x_user_id: str = Header(alias="x-user-id"),
) -> tuple[str, str]:
    """
    Extract database_id and user_id from headers for multi-user support.
    """
    try:
        # Validate header format if needed
        missing_headers = []
        if not x_database_id:
            missing_headers.append("x-database-id")
        if not x_user_id:
            missing_headers.append("x-user-id")

        if missing_headers:
            detail = f"Missing required headers: {', '.join(missing_headers)}"
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

        return x_database_id, x_user_id

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting user context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User context error",
        )
