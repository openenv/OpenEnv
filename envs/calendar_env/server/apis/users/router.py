"""
User API endpoints for user management operations
"""

import logging

from database.managers.user_manager import UserManager
from fastapi import APIRouter, Depends, Header, HTTPException, status
from middleware.auth import authenticate_user_with_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])

# Separate router for API endpoints that don't follow the "/users" pattern
api_router = APIRouter(tags=["user-info"])


def get_user_manager(database_id: str) -> UserManager:
    """Get user manager for the specified database"""
    return UserManager(database_id)


@router.get("/email/{email}")
async def get_user_by_email(
    email: str, x_database_id: str = Header(alias="x-database-id")
):
    """
    Get user details by email address

    GET /users/email/{email}
    """
    try:
        if not x_database_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required header: x-database-id",
            )

        user_manager = get_user_manager(x_database_id)
        user_info = user_manager.get_user_by_email(email)

        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User not found with email: {email}",
            )

        return user_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user by email {email}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_router.get("/api/user-info")
async def get_authenticated_user_info(
    auth_context: tuple[str, str] = Depends(authenticate_user_with_token),
):
    """
    Get authenticated user information using access token

    GET /api/user-info

    Headers:
    - x-database-id: Database identifier
    - x-access-token: User access token

    Returns:
    - user_id: User identifier
    - name: User's display name
    - email: User's email address
    """
    try:
        database_id, user_id = auth_context

        user_manager = get_user_manager(database_id)
        user_info = user_manager.get_user_by_id(user_id)

        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        # Return only the required fields: user_id, name, email
        response = {
            "user_id": user_info["id"],
            "name": user_info["name"],
            "email": user_info["email"],
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting authenticated user info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )
