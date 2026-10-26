"""
Watch Channel Seed Data for Calendar Application
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List


def get_watch_channel_sample_data() -> Dict[str, Any]:
    """
    Generate sample watch channel data for multi-user scenarios
    """

    # Base time for calculations
    now = datetime.utcnow()

    # Sample watch channels demonstrating various scenarios
    watch_channels = [
        # Alice's active watch channels
        {
            "id": "watch-alice-projects-001",
            "resource_id": "acl-alice-projects",
            "resource_uri": "/calendars/alice-projects/acl",
            "resource_type": "acl",
            "calendar_id": "alice-projects",
            "user_id": "alice_manager",
            "webhook_address": "https://techcorp.com/webhooks/alice/acl-notifications",
            "webhook_token": "alice_token_abc123",
            "webhook_type": "web_hook",
            "params": json.dumps({"ttl": 14200}),
            "created_at": now - timedelta(days=5),
            "expires_at": now + timedelta(seconds=14200),
            "last_notification_at": now - timedelta(hours=2),
            "is_active": "true",
            "notification_count": 12,
        },
        {
            "id": "watch-alice-team-002",
            "resource_id": "acl-alice-team",
            "resource_uri": "/calendars/alice-team/acl",
            "resource_type": "acl",
            "calendar_id": "alice-team",
            "user_id": "alice_manager",
            "webhook_address": "https://api.techcorp.com/calendar/notifications/team-acl",
            "webhook_token": "team_secure_token_xyz789",
            "webhook_type": "web_hook",
            "params": json.dumps({"ttl": 11200}),
            "created_at": now - timedelta(days=2),
            "expires_at": now + timedelta(seconds=11200),
            "last_notification_at": now - timedelta(hours=6),
            "is_active": "true",
            "notification_count": 5,
        },
        # Bob's watch channels
        {
            "id": "watch-bob-dev-001",
            "resource_id": "acl-bob-development",
            "resource_uri": "/calendars/bob-development/acl",
            "resource_type": "acl",
            "calendar_id": "bob-development",
            "user_id": "bob_developer",
            "webhook_address": "https://hooks.slack.com/services/T123/B456/dev-calendar-notifications",
            "webhook_token": None,  # No token for Slack webhook
            "webhook_type": "web_hook",
            "params": json.dumps({"ttl": 9700}),
            "created_at": now - timedelta(days=10),
            "expires_at": now + timedelta(seconds=9700),
            "last_notification_at": now - timedelta(days=1),
            "is_active": "true",
            "notification_count": 23,
        },
        {
            "id": "watch-bob-personal-002",
            "resource_id": "acl-bob-personal",
            "resource_uri": "/calendars/bob-personal/acl",
            "resource_type": "acl",
            "calendar_id": "bob-personal",
            "user_id": "bob_developer",
            "webhook_address": "https://api.example.com/personal/calendar/webhook",
            "webhook_token": "personal_webhook_token_def456",
            "webhook_type": "web_hook",
            "params": json.dumps({"ttl": 24390}),
            "created_at": now - timedelta(days=1),
            "expires_at": now + timedelta(seconds=24390),
            "last_notification_at": None,  # No notifications sent yet
            "is_active": "true",
            "notification_count": 0,
        },
        # Carol's watch channels
        {
            "id": "watch-carol-design-001",
            "resource_id": "acl-carol-design",
            "resource_uri": "/calendars/carol-design/acl",
            "resource_type": "acl",
            "calendar_id": "carol-design",
            "user_id": "carol_designer",
            "webhook_address": "https://design-tools.techcorp.com/calendar/acl-updates",
            "webhook_token": "design_team_token_ghi789",
            "webhook_type": "web_hook",
            "params": json.dumps({"ttl": 13200}),
            "created_at": now - timedelta(days=7),
            "expires_at": now + timedelta(seconds=13200),
            "last_notification_at": now - timedelta(hours=12),
            "is_active": "true",
            "notification_count": 8,
        },
        # Dave's watch channels
        {
            "id": "watch-dave-sales-001",
            "resource_id": "acl-dave-sales",
            "resource_uri": "/calendars/dave-sales/acl",
            "resource_type": "acl",
            "calendar_id": "dave-sales",
            "user_id": "dave_sales",
            "webhook_address": "https://crm.techcorp.com/calendar/sales-acl-sync",
            "webhook_token": "sales_crm_token_jkl012",
            "webhook_type": "web_hook",
            "params": json.dumps({"ttl": 15600}),
            "created_at": now - timedelta(days=14),
            "expires_at": now + timedelta(seconds=15600),
            "last_notification_at": now - timedelta(hours=4),
            "is_active": "true",
            "notification_count": 31,
        },
        # Expired/Inactive watch channels (for testing cleanup)
        {
            "id": "watch-alice-expired-001",
            "resource_id": "acl-alice-primary",
            "resource_uri": "/calendars/alice-primary/acl",
            "resource_type": "acl",
            "calendar_id": "alice-primary",
            "user_id": "alice_manager",
            "webhook_address": "https://old-system.techcorp.com/webhooks/acl",
            "webhook_token": "expired_token_mno345",
            "webhook_type": "web_hook",
            "params": json.dumps({"ttl": 2592000}),
            "created_at": now - timedelta(days=35),
            "expires_at": (now - timedelta(days=35))
            + timedelta(seconds=2592000),  # Expired 5 days ago
            "last_notification_at": now - timedelta(days=6),
            "is_active": "true",  # Not yet cleaned up
            "notification_count": 47,
        },
        {
            "id": "watch-bob-stopped-001",
            "resource_id": "acl-bob-primary",
            "resource_uri": "/calendars/bob-primary/acl",
            "resource_type": "acl",
            "calendar_id": "bob-primary",
            "user_id": "bob_developer",
            "webhook_address": "https://temp-webhook.example.com/test",
            "webhook_token": "temp_token_pqr678",
            "webhook_type": "web_hook",
            "params": json.dumps({"ttl": 12000}),
            "created_at": now - timedelta(days=8),
            "expires_at": now + timedelta(seconds=12000),
            "last_notification_at": now - timedelta(days=3),
            "is_active": "false",  # Manually stopped
            "notification_count": 15,
        },
        # Cross-calendar watch example (someone watching another user's shared calendar)
        {
            "id": "watch-bob-alice-shared-001",
            "resource_id": "acl-alice-projects",
            "resource_uri": "/calendars/alice-projects/acl",
            "resource_type": "acl",
            "calendar_id": "alice-projects",  # Alice's calendar
            "user_id": "bob_developer",  # But Bob is watching it
            "webhook_address": "https://api.example.com/shared-calendar/acl-watch",
            "webhook_token": "shared_access_token_stu901",
            "webhook_type": "web_hook",
            "params": json.dumps({"ttl": 16800}),
            "created_at": now - timedelta(days=3),
            "expires_at": now + timedelta(seconds=16800),
            "last_notification_at": now - timedelta(hours=18),
            "is_active": "true",
            "notification_count": 3,
        },
    ]

    return {
        "watch_channels": watch_channels,
        "description": "Sample watch channel data demonstrating various ACL notification scenarios across multiple users",
    }


def get_watch_channel_sql(sql_statements) -> str:
    """
    Generate SQL statements for watch channel sample data
    """
    data = get_watch_channel_sample_data()

    # Header
    sql_statements.append(f"-- Generated on: {datetime.now().isoformat()}")
    sql_statements.append("-- Contains sample watch channel data for ACL notifications")
    sql_statements.append("")

    # Watch Channels
    sql_statements.append("-- Watch Channels for ACL Notifications")
    sql_statements.append("INSERT INTO watch_channels (")
    sql_statements.append(
        "    id, resource_id, resource_uri, resource_type, calendar_id, user_id,"
    )
    sql_statements.append("    webhook_address, webhook_token, webhook_type, params,")
    sql_statements.append(
        "    created_at, expires_at, last_notification_at, is_active, notification_count"
    )
    sql_statements.append(") VALUES")

    channel_values = []
    for channel in data["watch_channels"]:
        # Handle nullable fields
        webhook_token = (
            f"'{channel['webhook_token']}'" if channel["webhook_token"] else "NULL"
        )
        params = f"'{channel['params']}'" if channel["params"] else "NULL"
        last_notification = (
            f"'{channel['last_notification_at'].isoformat()}'"
            if channel["last_notification_at"]
            else "NULL"
        )
        expires_at = (
            f"'{channel['expires_at'].isoformat()}'"
            if channel["expires_at"]
            else "NULL"
        )

        # Escape single quotes in webhook addresses
        webhook_address = channel["webhook_address"].replace("'", "''")

        channel_values.append(
            f"('{channel['id']}', '{channel['resource_id']}', '{channel['resource_uri']}', "
            f"'{channel['resource_type']}', '{channel['calendar_id']}', '{channel['user_id']}', "
            f"'{webhook_address}', {webhook_token}, '{channel['webhook_type']}', {params}, "
            f"'{channel['created_at'].isoformat()}', {expires_at}, {last_notification}, "
            f"'{channel['is_active']}', {channel['notification_count']})"
        )

    sql_statements.append(",\n".join(channel_values) + ";")
    sql_statements.append("")

    # Add some comments about the data
    sql_statements.append("-- Watch Channel Data Summary:")
    sql_statements.append("-- - Alice: 2 active channels + 1 expired")
    sql_statements.append(
        "-- - Bob: 2 active channels + 1 stopped + 1 shared calendar watch"
    )
    sql_statements.append("-- - Carol: 1 active channel")
    sql_statements.append("-- - Dave: 1 active channel")
    sql_statements.append("-- - Total: 6 active, 1 expired, 1 manually stopped")
    sql_statements.append(
        "-- - Demonstrates various webhook URLs (Slack, CRM, internal APIs)"
    )
    sql_statements.append("-- - Shows different notification patterns and usage levels")
    sql_statements.append("")

    return sql_statements
