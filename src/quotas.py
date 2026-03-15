"""Token usage quotas for budget-based access control.

Complements rate limiting (per-minute burst control) with longer-term
usage budgets (daily/monthly token issuance limits per agent).
"""

import json
import secrets
import time
from dataclasses import dataclass, field

import aiosqlite

from .database import DB_PATH


@dataclass
class UsageQuota:
    """A usage quota for an agent."""
    id: str = ""
    user_id: str = ""
    agent_id: str = ""
    quota_type: str = "daily"  # "daily", "monthly", "total"
    max_tokens: int = 0  # Maximum token requests allowed
    current_usage: int = 0
    period_start: float = 0.0  # Start of current period
    created_at: float = 0.0
    is_active: bool = True
    action_on_exceed: str = "deny"  # "deny", "warn", "step_up"
    notify_at_percent: int = 80  # Send notification at this usage percentage


@dataclass
class QuotaStatus:
    """Current status of a quota check."""
    allowed: bool = True
    quota_id: str = ""
    quota_type: str = ""
    current_usage: int = 0
    max_tokens: int = 0
    remaining: int = 0
    usage_percent: float = 0.0
    threshold_reached: bool = False
    action: str = ""  # "" if allowed, or the action_on_exceed value
    resets_at: float = 0.0  # When the current period resets

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "quota_id": self.quota_id,
            "quota_type": self.quota_type,
            "current_usage": self.current_usage,
            "max_tokens": self.max_tokens,
            "remaining": self.remaining,
            "usage_percent": round(self.usage_percent, 1),
            "threshold_reached": self.threshold_reached,
            "action": self.action,
            "resets_at": self.resets_at if self.resets_at > 0 else None,
        }


async def init_quota_tables():
    """Create quota-related database tables."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS usage_quotas (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                quota_type TEXT NOT NULL DEFAULT 'daily',
                max_tokens INTEGER NOT NULL DEFAULT 0,
                current_usage INTEGER NOT NULL DEFAULT 0,
                period_start REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                action_on_exceed TEXT NOT NULL DEFAULT 'deny',
                notify_at_percent INTEGER NOT NULL DEFAULT 80
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS quota_usage_log (
                id TEXT PRIMARY KEY,
                quota_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                service TEXT NOT NULL DEFAULT '',
                timestamp REAL NOT NULL,
                action TEXT NOT NULL DEFAULT 'token_issued'
            )
        """)
        await db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_quota_user_agent_type
            ON usage_quotas(user_id, agent_id, quota_type)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_quota_usage_log_quota
            ON quota_usage_log(quota_id, timestamp DESC)
        """)
        await db.commit()


def _get_period_bounds(quota_type: str, now: float) -> tuple[float, float]:
    """Get the start and end timestamps for the current quota period."""
    from datetime import datetime, timezone, timedelta

    dt = datetime.fromtimestamp(now, tz=timezone.utc)

    if quota_type == "daily":
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif quota_type == "monthly":
        start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)
    elif quota_type == "total":
        return 0.0, 0.0  # No period for total quotas
    else:
        return 0.0, 0.0

    return start.timestamp(), end.timestamp()


async def create_quota(
    user_id: str,
    agent_id: str,
    quota_type: str = "daily",
    max_tokens: int = 100,
    action_on_exceed: str = "deny",
    notify_at_percent: int = 80,
) -> UsageQuota:
    """Create or update a usage quota for an agent."""
    valid_types = {"daily", "monthly", "total"}
    if quota_type not in valid_types:
        raise ValueError(f"Invalid quota type: {quota_type}. Must be one of {valid_types}")

    valid_actions = {"deny", "warn", "step_up"}
    if action_on_exceed not in valid_actions:
        raise ValueError(f"Invalid action: {action_on_exceed}. Must be one of {valid_actions}")

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    if not (0 <= notify_at_percent <= 100):
        raise ValueError("notify_at_percent must be 0-100")

    now = time.time()
    period_start, _ = _get_period_bounds(quota_type, now)
    quota_id = secrets.token_urlsafe(16)

    quota = UsageQuota(
        id=quota_id,
        user_id=user_id,
        agent_id=agent_id,
        quota_type=quota_type,
        max_tokens=max_tokens,
        current_usage=0,
        period_start=period_start,
        created_at=now,
        is_active=True,
        action_on_exceed=action_on_exceed,
        notify_at_percent=notify_at_percent,
    )

    async with aiosqlite.connect(DB_PATH) as db:
        # Upsert based on user_id + agent_id + quota_type
        await db.execute(
            """INSERT INTO usage_quotas
               (id, user_id, agent_id, quota_type, max_tokens, current_usage,
                period_start, created_at, is_active, action_on_exceed, notify_at_percent)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, agent_id, quota_type) DO UPDATE SET
                   max_tokens = excluded.max_tokens,
                   action_on_exceed = excluded.action_on_exceed,
                   notify_at_percent = excluded.notify_at_percent,
                   is_active = excluded.is_active""",
            (quota.id, quota.user_id, quota.agent_id, quota.quota_type,
             quota.max_tokens, quota.current_usage, quota.period_start,
             quota.created_at, int(quota.is_active), quota.action_on_exceed,
             quota.notify_at_percent),
        )
        await db.commit()

    return quota


async def check_quota(
    user_id: str,
    agent_id: str,
) -> list[QuotaStatus]:
    """Check all quotas for an agent. Returns status for each quota.

    Automatically resets period-based quotas when the period has elapsed.
    """
    now = time.time()
    results = []

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM usage_quotas WHERE user_id = ? AND agent_id = ? AND is_active = 1",
            (user_id, agent_id),
        )
        rows = await cursor.fetchall()

        for row in rows:
            quota_type = row["quota_type"]
            current_usage = row["current_usage"]
            max_tokens = row["max_tokens"]
            period_start = row["period_start"]

            # Check if period has elapsed and reset if needed
            period_bounds = _get_period_bounds(quota_type, now)
            current_period_start = period_bounds[0]
            resets_at = period_bounds[1]

            if quota_type != "total" and current_period_start > period_start:
                # Period has elapsed, reset counter
                current_usage = 0
                await db.execute(
                    "UPDATE usage_quotas SET current_usage = 0, period_start = ? WHERE id = ?",
                    (current_period_start, row["id"]),
                )
                await db.commit()

            remaining = max(0, max_tokens - current_usage)
            usage_pct = (current_usage / max_tokens * 100) if max_tokens > 0 else 0
            threshold = row["notify_at_percent"]
            exceeded = current_usage >= max_tokens

            status = QuotaStatus(
                allowed=not exceeded or row["action_on_exceed"] == "warn",
                quota_id=row["id"],
                quota_type=quota_type,
                current_usage=current_usage,
                max_tokens=max_tokens,
                remaining=remaining,
                usage_percent=usage_pct,
                threshold_reached=usage_pct >= threshold,
                action=row["action_on_exceed"] if exceeded else "",
                resets_at=resets_at,
            )
            results.append(status)

    return results


async def record_quota_usage(
    user_id: str,
    agent_id: str,
    service: str = "",
) -> list[QuotaStatus]:
    """Record a token usage against all active quotas for this agent.

    Returns the updated quota statuses after incrementing.
    """
    now = time.time()

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM usage_quotas WHERE user_id = ? AND agent_id = ? AND is_active = 1",
            (user_id, agent_id),
        )
        rows = await cursor.fetchall()

        for row in rows:
            quota_type = row["quota_type"]
            period_bounds = _get_period_bounds(quota_type, now)
            current_period_start = period_bounds[0]

            # Reset if period elapsed
            if quota_type != "total" and current_period_start > row["period_start"]:
                await db.execute(
                    "UPDATE usage_quotas SET current_usage = 1, period_start = ? WHERE id = ?",
                    (current_period_start, row["id"]),
                )
            else:
                await db.execute(
                    "UPDATE usage_quotas SET current_usage = current_usage + 1 WHERE id = ?",
                    (row["id"],),
                )

            # Log the usage event
            log_id = secrets.token_urlsafe(16)
            await db.execute(
                """INSERT INTO quota_usage_log (id, quota_id, agent_id, service, timestamp, action)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (log_id, row["id"], agent_id, service, now, "token_issued"),
            )

        await db.commit()

    # Return fresh status
    return await check_quota(user_id, agent_id)


async def get_quotas(user_id: str, agent_id: str = "") -> list[UsageQuota]:
    """List all quotas for a user, optionally filtered by agent_id."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if agent_id:
            cursor = await db.execute(
                "SELECT * FROM usage_quotas WHERE user_id = ? AND agent_id = ?",
                (user_id, agent_id),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM usage_quotas WHERE user_id = ?",
                (user_id,),
            )
        rows = await cursor.fetchall()
        return [_quota_from_row(row) for row in rows]


async def delete_quota(quota_id: str, user_id: str) -> bool:
    """Delete a quota. Returns True if found and deleted."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM usage_quotas WHERE id = ? AND user_id = ?",
            (quota_id, user_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def reset_quota(quota_id: str, user_id: str) -> bool:
    """Reset a quota's current usage to zero. Returns True if found."""
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "UPDATE usage_quotas SET current_usage = 0, period_start = ? WHERE id = ? AND user_id = ?",
            (now, quota_id, user_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def get_quota_usage_history(
    quota_id: str, limit: int = 100,
) -> list[dict]:
    """Get usage history for a specific quota."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT * FROM quota_usage_log
               WHERE quota_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (quota_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "agent_id": row["agent_id"],
                "service": row["service"],
                "timestamp": row["timestamp"],
                "action": row["action"],
            }
            for row in rows
        ]


def _quota_from_row(row) -> UsageQuota:
    """Construct a UsageQuota from a database row."""
    return UsageQuota(
        id=row["id"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        quota_type=row["quota_type"],
        max_tokens=row["max_tokens"],
        current_usage=row["current_usage"],
        period_start=row["period_start"],
        created_at=row["created_at"],
        is_active=bool(row["is_active"]),
        action_on_exceed=row["action_on_exceed"],
        notify_at_percent=row["notify_at_percent"],
    )
