"""Webhook notification system for security events.

Allows users to subscribe to real-time alerts when security-relevant events
occur (anomaly detected, policy violated, emergency revoke, step-up triggered,
delegation created/revoked, etc.).
"""

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum

import aiosqlite

from .database import DB_PATH


class WebhookEvent(str, Enum):
    """Security events that can trigger webhook notifications."""
    POLICY_VIOLATED = "policy.violated"
    POLICY_CREATED = "policy.created"
    POLICY_DELETED = "policy.deleted"
    POLICY_TOGGLED = "policy.toggled"
    ANOMALY_DETECTED = "anomaly.detected"
    EMERGENCY_REVOKE = "emergency.revoke"
    STEP_UP_TRIGGERED = "step_up.triggered"
    STEP_UP_APPROVED = "step_up.approved"
    STEP_UP_DENIED = "step_up.denied"
    RATE_LIMIT_HIT = "rate_limit.hit"
    API_KEY_CREATED = "api_key.created"
    API_KEY_REVOKED = "api_key.revoked"
    DELEGATION_CREATED = "delegation.created"
    DELEGATION_REVOKED = "delegation.revoked"
    TOKEN_ISSUED = "token.issued"
    CONSENT_REQUESTED = "consent.requested"
    CONSENT_REVOKED = "consent.revoked"


@dataclass
class WebhookSubscription:
    """A webhook subscription for a specific set of events."""
    id: str = ""
    user_id: str = ""
    url: str = ""
    secret: str = ""  # Used for HMAC signature verification
    events: list[str] = field(default_factory=list)  # List of WebhookEvent values
    is_active: bool = True
    created_at: float = 0.0
    description: str = ""
    failure_count: int = 0
    last_triggered_at: float = 0.0
    last_status_code: int = 0


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""
    id: str = ""
    subscription_id: str = ""
    event: str = ""
    payload: str = ""
    status_code: int = 0
    response_body: str = ""
    delivered_at: float = 0.0
    success: bool = False
    duration_ms: float = 0.0


async def init_webhook_tables():
    """Create webhook-related database tables."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS webhook_subscriptions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                url TEXT NOT NULL,
                secret TEXT NOT NULL,
                events TEXT NOT NULL DEFAULT '[]',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                failure_count INTEGER NOT NULL DEFAULT 0,
                last_triggered_at REAL NOT NULL DEFAULT 0,
                last_status_code INTEGER NOT NULL DEFAULT 0
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS webhook_deliveries (
                id TEXT PRIMARY KEY,
                subscription_id TEXT NOT NULL,
                event TEXT NOT NULL,
                payload TEXT NOT NULL,
                status_code INTEGER NOT NULL DEFAULT 0,
                response_body TEXT NOT NULL DEFAULT '',
                delivered_at REAL NOT NULL,
                success INTEGER NOT NULL DEFAULT 0,
                duration_ms REAL NOT NULL DEFAULT 0
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_webhook_sub_user
            ON webhook_subscriptions(user_id)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_sub
            ON webhook_deliveries(subscription_id, delivered_at DESC)
        """)
        await db.commit()


def compute_signature(payload: str, secret: str) -> str:
    """Compute HMAC-SHA256 signature for webhook payload verification."""
    return hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()


def verify_signature(payload: str, secret: str, signature: str) -> bool:
    """Verify an HMAC-SHA256 webhook signature (constant-time comparison)."""
    expected = compute_signature(payload, secret)
    return hmac.compare_digest(expected, signature)


async def create_webhook(
    user_id: str,
    url: str,
    events: list[str],
    description: str = "",
) -> WebhookSubscription:
    """Create a new webhook subscription.

    Returns the subscription with its secret (only shown at creation).
    """
    # Validate events
    valid_events = {e.value for e in WebhookEvent}
    invalid = set(events) - valid_events
    if invalid:
        raise ValueError(f"Invalid events: {', '.join(invalid)}")

    if not events:
        raise ValueError("At least one event is required")

    if not url:
        raise ValueError("URL is required")

    sub_id = secrets.token_urlsafe(16)
    secret = f"whsec_{secrets.token_urlsafe(32)}"
    now = time.time()

    sub = WebhookSubscription(
        id=sub_id,
        user_id=user_id,
        url=url,
        secret=secret,
        events=events,
        is_active=True,
        created_at=now,
        description=description,
    )

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO webhook_subscriptions
               (id, user_id, url, secret, events, is_active, created_at, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (sub.id, sub.user_id, sub.url, sub.secret,
             json.dumps(sub.events), int(sub.is_active),
             sub.created_at, sub.description),
        )
        await db.commit()

    return sub


async def list_webhooks(user_id: str) -> list[WebhookSubscription]:
    """List all webhook subscriptions for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM webhook_subscriptions WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [_sub_from_row(row) for row in rows]


async def get_webhook(webhook_id: str, user_id: str) -> WebhookSubscription | None:
    """Get a specific webhook subscription."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM webhook_subscriptions WHERE id = ? AND user_id = ?",
            (webhook_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return _sub_from_row(row)


async def update_webhook(
    webhook_id: str,
    user_id: str,
    url: str | None = None,
    events: list[str] | None = None,
    is_active: bool | None = None,
    description: str | None = None,
) -> WebhookSubscription | None:
    """Update a webhook subscription. Returns updated sub or None if not found."""
    sub = await get_webhook(webhook_id, user_id)
    if not sub:
        return None

    if events is not None:
        valid_events = {e.value for e in WebhookEvent}
        invalid = set(events) - valid_events
        if invalid:
            raise ValueError(f"Invalid events: {', '.join(invalid)}")
        sub.events = events

    if url is not None:
        sub.url = url
    if is_active is not None:
        sub.is_active = is_active
    if description is not None:
        sub.description = description

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE webhook_subscriptions
               SET url = ?, events = ?, is_active = ?, description = ?
               WHERE id = ? AND user_id = ?""",
            (sub.url, json.dumps(sub.events), int(sub.is_active),
             sub.description, webhook_id, user_id),
        )
        await db.commit()

    return sub


async def delete_webhook(webhook_id: str, user_id: str) -> bool:
    """Delete a webhook subscription. Returns True if found and deleted."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM webhook_subscriptions WHERE id = ? AND user_id = ?",
            (webhook_id, user_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def rotate_webhook_secret(webhook_id: str, user_id: str) -> str | None:
    """Generate a new secret for a webhook. Returns new secret or None if not found."""
    new_secret = f"whsec_{secrets.token_urlsafe(32)}"
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "UPDATE webhook_subscriptions SET secret = ? WHERE id = ? AND user_id = ?",
            (new_secret, webhook_id, user_id),
        )
        await db.commit()
        if cursor.rowcount == 0:
            return None
    return new_secret


async def get_subscriptions_for_event(
    user_id: str, event: str,
) -> list[WebhookSubscription]:
    """Get all active subscriptions that match a given event for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM webhook_subscriptions WHERE user_id = ? AND is_active = 1",
            (user_id,),
        )
        rows = await cursor.fetchall()
        subs = []
        for row in rows:
            sub = _sub_from_row(row)
            if event in sub.events:
                subs.append(sub)
        return subs


async def dispatch_webhook(
    user_id: str,
    event: str,
    data: dict,
) -> list[WebhookDelivery]:
    """Dispatch a webhook event to all matching subscriptions.

    Builds the payload, signs it, and records the delivery attempt.
    Actual HTTP delivery is simulated in the record (real delivery would use
    an async task queue in production).
    """
    subs = await get_subscriptions_for_event(user_id, event)
    deliveries = []
    now = time.time()

    payload = json.dumps({
        "event": event,
        "timestamp": now,
        "data": data,
    })

    for sub in subs:
        delivery_id = secrets.token_urlsafe(16)
        signature = compute_signature(payload, sub.secret)

        delivery = WebhookDelivery(
            id=delivery_id,
            subscription_id=sub.id,
            event=event,
            payload=payload,
            status_code=0,  # Will be updated by actual delivery
            delivered_at=now,
            success=False,
            duration_ms=0,
        )

        # Record the delivery attempt
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """INSERT INTO webhook_deliveries
                   (id, subscription_id, event, payload, status_code,
                    response_body, delivered_at, success, duration_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (delivery.id, delivery.subscription_id, delivery.event,
                 delivery.payload, delivery.status_code, delivery.response_body,
                 delivery.delivered_at, int(delivery.success), delivery.duration_ms),
            )

            # Update subscription metadata
            await db.execute(
                """UPDATE webhook_subscriptions
                   SET last_triggered_at = ?
                   WHERE id = ?""",
                (now, sub.id),
            )
            await db.commit()

        delivery._signature = signature  # Attach for testing/inspection
        deliveries.append(delivery)

    return deliveries


async def record_delivery_result(
    delivery_id: str,
    status_code: int,
    response_body: str = "",
    duration_ms: float = 0,
) -> bool:
    """Record the result of an HTTP delivery attempt."""
    success = 200 <= status_code < 300
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """UPDATE webhook_deliveries
               SET status_code = ?, response_body = ?, success = ?, duration_ms = ?
               WHERE id = ?""",
            (status_code, response_body[:1000], int(success), duration_ms, delivery_id),
        )
        if cursor.rowcount == 0:
            await db.commit()
            return False

        # Get the subscription_id to update failure count
        cursor2 = await db.execute(
            "SELECT subscription_id FROM webhook_deliveries WHERE id = ?",
            (delivery_id,),
        )
        row = await cursor2.fetchone()
        if row:
            sub_id = row[0]
            if success:
                await db.execute(
                    "UPDATE webhook_subscriptions SET failure_count = 0, last_status_code = ? WHERE id = ?",
                    (status_code, sub_id),
                )
            else:
                await db.execute(
                    """UPDATE webhook_subscriptions
                       SET failure_count = failure_count + 1, last_status_code = ?
                       WHERE id = ?""",
                    (status_code, sub_id),
                )
                # Auto-disable after 10 consecutive failures
                await db.execute(
                    """UPDATE webhook_subscriptions
                       SET is_active = 0
                       WHERE id = ? AND failure_count >= 10""",
                    (sub_id,),
                )

        await db.commit()
    return True


async def get_webhook_deliveries(
    webhook_id: str, user_id: str, limit: int = 50,
) -> list[WebhookDelivery]:
    """Get recent delivery attempts for a webhook subscription."""
    # Verify ownership
    sub = await get_webhook(webhook_id, user_id)
    if not sub:
        return []

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT * FROM webhook_deliveries
               WHERE subscription_id = ?
               ORDER BY delivered_at DESC LIMIT ?""",
            (webhook_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            WebhookDelivery(
                id=row["id"],
                subscription_id=row["subscription_id"],
                event=row["event"],
                payload=row["payload"],
                status_code=row["status_code"],
                response_body=row["response_body"],
                delivered_at=row["delivered_at"],
                success=bool(row["success"]),
                duration_ms=row["duration_ms"],
            )
            for row in rows
        ]


def _sub_from_row(row) -> WebhookSubscription:
    """Construct a WebhookSubscription from a database row."""
    return WebhookSubscription(
        id=row["id"],
        user_id=row["user_id"],
        url=row["url"],
        secret=row["secret"],
        events=json.loads(row["events"]),
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        description=row["description"],
        failure_count=row["failure_count"],
        last_triggered_at=row["last_triggered_at"],
        last_status_code=row["last_status_code"],
    )
