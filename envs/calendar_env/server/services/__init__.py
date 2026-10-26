"""
Services package for calendar application
"""

from .notification_service import get_notification_service, NotificationService

__all__ = ["NotificationService", "get_notification_service"]
