"""
User Manager - Database operations for user management using SQLAlchemy
"""

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from database.models import User
from database.session_utils import get_session, init_database
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _token_fingerprint(token: Optional[str]) -> str:
    """Return a short, non-reversible fingerprint of an access token"""
    if not token:
        return "<empty>"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]


class UserManager:
    """Manager for user database operations using SQLAlchemy"""

    def __init__(self, database_id: str):
        self.database_id = database_id
        # Initialize database on first use
        init_database(database_id)

    def create_user(self, user_data: Dict) -> Dict:
        """Create a new user"""
        session = get_session(self.database_id)
        try:
            # Generate unique user ID if not provided
            user_id = user_data.get("user_id") or str(uuid.uuid4())

            # Check if user with email already exists
            existing_user = (
                session.query(User).filter(User.email == user_data["email"]).first()
            )
            if existing_user:
                raise ValueError(f"User with email {user_data['email']} already exists")

            # Create User model instance
            user = User(
                user_id=user_id,
                email=user_data["email"],
                name=user_data["name"],
                given_name=user_data.get("given_name"),
                family_name=user_data.get("family_name"),
                picture=user_data.get("picture"),
                locale=user_data.get("locale", "en"),
                timezone=user_data.get("timezone", "UTC"),
                is_active=user_data.get("is_active", True),
                is_verified=user_data.get("is_verified", False),
                provider=user_data.get("provider"),
                provider_id=user_data.get("provider_id"),
                access_token_hash=user_data.get("access_token_hash"),
                refresh_token_hash=user_data.get("refresh_token_hash"),
            )

            session.add(user)
            session.commit()

            # Return the created user
            result = self._format_user_response(user)
            return result

        except Exception as e:
            session.rollback()
            logger.error(f"Error creating user: {e}")
            raise
        finally:
            session.close()

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get a user by ID"""
        session = get_session(self.database_id)
        try:
            user = session.query(User).filter(User.user_id == user_id).first()

            if not user:
                return None

            return self._format_user_response(user)

        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
            raise
        finally:
            session.close()

    def get_first_user_from_db(self) -> Optional[Dict]:
        """Get the first user from db"""
        session = get_session(self.database_id)
        try:
            user = session.query(User).first()
            if not user:
                return None

            return self._format_user_response(user)
        except Exception as e:
            logger.error(f"Error getting user from db: {e}")
            raise
        finally:
            session.close()

    def get_user_by_access_token(self, static_token: str) -> Optional[Dict]:
        """Get a user by access token"""
        session = get_session(self.database_id)
        try:
            user = session.query(User).filter(User.static_token == static_token).first()

            if not user:
                return None

            return self._format_user_response(user)

        except Exception as e:
            # Never log the raw token; emit only a short fingerprint so we
            # can correlate failures without leaking credentials to the log sink.
            logger.error(
                f"Error getting user by access token "
                f"(fingerprint={_token_fingerprint(static_token)}): {type(e).__name__}"
            )
            raise
        finally:
            session.close()

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get a user by email"""
        session = get_session(self.database_id)
        try:
            user = session.query(User).filter(User.email == email).first()

            if not user:
                return None

            return self._format_user_response(user)

        except Exception as e:
            logger.error(f"Error getting user by email {email}: {e}")
            raise
        finally:
            session.close()

    def update_user(self, user_id: str, user_data: Dict) -> Optional[Dict]:
        """Update a user"""
        session = get_session(self.database_id)
        try:
            user = session.query(User).filter(User.user_id == user_id).first()

            if not user:
                return None

            # Update fields if provided
            updateable_fields = [
                "name",
                "given_name",
                "family_name",
                "picture",
                "locale",
                "timezone",
                "is_active",
                "is_verified",
                "provider",
                "provider_id",
                "access_token_hash",
                "refresh_token_hash",
            ]

            for field in updateable_fields:
                if field in user_data:
                    setattr(user, field, user_data[field])

            # Update last login if provided
            if user_data.get("update_last_login"):
                user.last_login_at = datetime.now(timezone.utc)

            # SQLAlchemy will automatically update updated_at due to onupdate=datetime.utcnow
            session.commit()

            return self._format_user_response(user)

        except Exception as e:
            session.rollback()
            logger.error(f"Error updating user {user_id}: {e}")
            raise
        finally:
            session.close()

    def delete_user(self, user_id: str) -> bool:
        """Delete a user and all related data"""
        session = get_session(self.database_id)
        try:
            user = session.query(User).filter(User.user_id == user_id).first()

            if not user:
                return False

            # Delete user (cascade will delete related calendars and events)
            session.delete(user)
            session.commit()
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting user {user_id}: {e}")
            raise
        finally:
            session.close()

    def deactivate_user(self, user_id: str) -> Optional[Dict]:
        """Deactivate a user (soft delete)"""
        session = get_session(self.database_id)
        try:
            user = session.query(User).filter(User.user_id == user_id).first()

            if not user:
                return None

            user.is_active = False
            session.commit()

            return self._format_user_response(user)

        except Exception as e:
            session.rollback()
            logger.error(f"Error deactivating user {user_id}: {e}")
            raise
        finally:
            session.close()

    def list_users(
        self, limit: int = 100, offset: int = 0, active_only: bool = True
    ) -> List[Dict]:
        """List users"""
        session = get_session(self.database_id)
        try:
            query = session.query(User)

            if active_only:
                query = query.filter(User.is_active == True)

            users = (
                query.order_by(User.created_at.desc()).offset(offset).limit(limit).all()
            )

            return [self._format_user_response(user) for user in users]

        except Exception as e:
            logger.error(f"Error listing users: {e}")
            raise
        finally:
            session.close()

    def authenticate_user(
        self, email: str, provider: str = None, provider_id: str = None
    ) -> Optional[Dict]:
        """Authenticate user by email and optionally by provider"""
        session = get_session(self.database_id)
        try:
            query = session.query(User).filter(
                User.email == email, User.is_active == True
            )

            if provider:
                query = query.filter(User.provider == provider)

            if provider_id:
                query = query.filter(User.provider_id == provider_id)

            user = query.first()

            if not user:
                return None

            # Update last login
            user.last_login_at = datetime.now(timezone.utc)
            session.commit()

            return self._format_user_response(user)

        except Exception as e:
            logger.error(f"Error authenticating user {email}: {e}")
            raise
        finally:
            session.close()

    def _format_user_response(self, user: User) -> Dict:
        """Format user model for API response (without sensitive data)"""
        return {
            "id": user.user_id,
            "email": user.email,
            "name": user.name,
            "given_name": user.given_name,
            "family_name": user.family_name,
            "picture": user.picture,
            "locale": user.locale,
            "timezone": user.timezone,
            "is_active": user.is_active,
            "is_verified": user.is_verified,
            "provider": user.provider,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None,
            "last_login_at": user.last_login_at.isoformat()
            if user.last_login_at
            else None,
        }
