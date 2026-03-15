"""Consent management for granular user approval of agent actions.

Provides a structured consent workflow where users can:
- Pre-approve specific agent actions with conditions
- Require explicit consent for sensitive operations
- Set consent expiration and usage limits
- Review and revoke consent grants
- Track consent usage history

This integrates with Auth0's authorization model to ensure agents only
act with explicit, auditable user permission.
"""

import json
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum

import aiosqlite

from .database import DB_PATH, log_audit


class ConsentStatus(str, Enum):
    """Status of a consent grant."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    REVOKED = "revoked"
    EXPIRED = "expired"


class ConsentScope(str, Enum):
    """Granularity level for consent grants."""
    BLANKET = "blanket"        # Approve all actions for a service
    SERVICE = "service"        # Approve specific service access
    ACTION = "action"          # Approve a specific action type
    RESOURCE = "resource"      # Approve access to a specific resource
    ONE_TIME = "one_time"      # Single-use consent


@dataclass
class ConsentGrant:
    """A consent grant from a user to an agent."""
    id: str = ""
    user_id: str = ""
    agent_id: str = ""
    service: str = ""
    scope_type: str = "service"        # ConsentScope value
    action_pattern: str = "*"          # Glob pattern for allowed actions
    resource_pattern: str = "*"        # Glob pattern for allowed resources
    conditions: dict = field(default_factory=dict)  # Additional conditions
    status: str = "pending"            # ConsentStatus value
    max_uses: int = 0                  # 0 = unlimited
    current_uses: int = 0
    created_at: float = 0.0
    expires_at: float = 0.0            # 0 = never
    approved_at: float = 0.0
    revoked_at: float = 0.0
    revoke_reason: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ConsentRequest:
    """A request for user consent from an agent."""
    id: str = ""
    user_id: str = ""
    agent_id: str = ""
    service: str = ""
    action: str = ""
    resource: str = ""
    reason: str = ""
    urgency: str = "normal"            # low, normal, high, critical
    status: str = "pending"
    grant_id: str = ""                 # Set when consent is granted
    created_at: float = 0.0
    resolved_at: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ConsentAuditEntry:
    """An audit trail entry for consent operations."""
    id: int = 0
    timestamp: float = 0.0
    user_id: str = ""
    agent_id: str = ""
    consent_id: str = ""
    action: str = ""
    details: str = ""


async def init_consent_tables():
    """Create consent management tables."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS consent_grants (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                service TEXT NOT NULL,
                scope_type TEXT NOT NULL DEFAULT 'service',
                action_pattern TEXT NOT NULL DEFAULT '*',
                resource_pattern TEXT NOT NULL DEFAULT '*',
                conditions TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                max_uses INTEGER NOT NULL DEFAULT 0,
                current_uses INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL DEFAULT 0,
                approved_at REAL NOT NULL DEFAULT 0,
                revoked_at REAL NOT NULL DEFAULT 0,
                revoke_reason TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_consent_user_agent
            ON consent_grants(user_id, agent_id, status)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_consent_service
            ON consent_grants(service, status)
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS consent_requests (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                service TEXT NOT NULL,
                action TEXT NOT NULL DEFAULT '',
                resource TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                urgency TEXT NOT NULL DEFAULT 'normal',
                status TEXT NOT NULL DEFAULT 'pending',
                grant_id TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                resolved_at REAL NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_consent_req_user
            ON consent_requests(user_id, status)
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS consent_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                consent_id TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT ''
            )
        """)
        await db.commit()


async def _log_consent_audit(
    user_id: str, agent_id: str, consent_id: str, action: str, details: str = ""
):
    """Log a consent audit entry."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO consent_audit (timestamp, user_id, agent_id, consent_id, action, details)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (time.time(), user_id, agent_id, consent_id, action, details),
        )
        await db.commit()


async def create_consent_grant(
    user_id: str,
    agent_id: str,
    service: str,
    scope_type: str = "service",
    action_pattern: str = "*",
    resource_pattern: str = "*",
    conditions: dict | None = None,
    max_uses: int = 0,
    expires_at: float = 0.0,
    auto_approve: bool = False,
    metadata: dict | None = None,
) -> ConsentGrant:
    """Create a new consent grant.

    Args:
        user_id: The user granting consent
        agent_id: The agent receiving consent
        service: The service this consent applies to
        scope_type: Granularity level (blanket, service, action, resource, one_time)
        action_pattern: Glob pattern for allowed actions
        resource_pattern: Glob pattern for allowed resources
        conditions: Additional conditions (e.g., time windows, IP restrictions)
        max_uses: Maximum number of times this consent can be used (0 = unlimited)
        expires_at: When this consent expires (0 = never)
        auto_approve: Whether to auto-approve the grant
        metadata: Additional metadata

    Returns:
        The created ConsentGrant
    """
    if not user_id:
        raise ConsentError("user_id is required")
    if not agent_id:
        raise ConsentError("agent_id is required")
    if not service:
        raise ConsentError("service is required")

    # Validate scope type
    valid_scopes = {s.value for s in ConsentScope}
    if scope_type not in valid_scopes:
        raise ConsentError(f"Invalid scope_type '{scope_type}'. Must be one of: {valid_scopes}")

    # Validate urgency-related conditions
    if conditions and "require_mfa" in conditions:
        if not isinstance(conditions["require_mfa"], bool):
            raise ConsentError("condition 'require_mfa' must be boolean")

    grant_id = secrets.token_urlsafe(16)
    now = time.time()
    status = ConsentStatus.APPROVED.value if auto_approve else ConsentStatus.PENDING.value

    grant = ConsentGrant(
        id=grant_id,
        user_id=user_id,
        agent_id=agent_id,
        service=service,
        scope_type=scope_type,
        action_pattern=action_pattern,
        resource_pattern=resource_pattern,
        conditions=conditions or {},
        status=status,
        max_uses=max_uses,
        current_uses=0,
        created_at=now,
        expires_at=expires_at,
        approved_at=now if auto_approve else 0.0,
        metadata=metadata or {},
    )

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO consent_grants
               (id, user_id, agent_id, service, scope_type, action_pattern,
                resource_pattern, conditions, status, max_uses, current_uses,
                created_at, expires_at, approved_at, revoked_at, revoke_reason, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                grant.id, grant.user_id, grant.agent_id, grant.service,
                grant.scope_type, grant.action_pattern, grant.resource_pattern,
                json.dumps(grant.conditions), grant.status, grant.max_uses,
                grant.current_uses, grant.created_at, grant.expires_at,
                grant.approved_at, grant.revoked_at, grant.revoke_reason,
                json.dumps(grant.metadata),
            ),
        )
        await db.commit()

    await _log_consent_audit(
        user_id, agent_id, grant_id,
        "grant_created",
        f"scope={scope_type}, service={service}, auto_approve={auto_approve}",
    )

    return grant


class ConsentError(Exception):
    """Raised when a consent operation fails."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


async def approve_consent(grant_id: str, user_id: str) -> ConsentGrant | None:
    """Approve a pending consent grant.

    Only the grant owner can approve it.
    Returns the updated grant, or None if not found.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM consent_grants WHERE id = ? AND user_id = ?",
            (grant_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        if row["status"] != ConsentStatus.PENDING.value:
            raise ConsentError(
                f"Cannot approve grant in status '{row['status']}' (must be 'pending')"
            )

        now = time.time()
        await db.execute(
            "UPDATE consent_grants SET status = ?, approved_at = ? WHERE id = ?",
            (ConsentStatus.APPROVED.value, now, grant_id),
        )
        await db.commit()

    await _log_consent_audit(user_id, row["agent_id"], grant_id, "grant_approved")

    return _grant_from_row(row, override_status=ConsentStatus.APPROVED.value, override_approved_at=now)


async def deny_consent(grant_id: str, user_id: str, reason: str = "") -> ConsentGrant | None:
    """Deny a pending consent grant."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM consent_grants WHERE id = ? AND user_id = ?",
            (grant_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        if row["status"] != ConsentStatus.PENDING.value:
            raise ConsentError(
                f"Cannot deny grant in status '{row['status']}' (must be 'pending')"
            )

        now = time.time()
        await db.execute(
            "UPDATE consent_grants SET status = ?, revoked_at = ?, revoke_reason = ? WHERE id = ?",
            (ConsentStatus.DENIED.value, now, reason, grant_id),
        )
        await db.commit()

    await _log_consent_audit(
        user_id, row["agent_id"], grant_id, "grant_denied", reason
    )

    return _grant_from_row(row, override_status=ConsentStatus.DENIED.value)


async def revoke_consent(grant_id: str, user_id: str, reason: str = "") -> ConsentGrant | None:
    """Revoke an approved consent grant."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM consent_grants WHERE id = ? AND user_id = ?",
            (grant_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        if row["status"] not in (ConsentStatus.APPROVED.value, ConsentStatus.PENDING.value):
            raise ConsentError(
                f"Cannot revoke grant in status '{row['status']}'"
            )

        now = time.time()
        await db.execute(
            "UPDATE consent_grants SET status = ?, revoked_at = ?, revoke_reason = ? WHERE id = ?",
            (ConsentStatus.REVOKED.value, now, reason, grant_id),
        )
        await db.commit()

    await _log_consent_audit(
        user_id, row["agent_id"], grant_id, "grant_revoked", reason
    )

    return _grant_from_row(row, override_status=ConsentStatus.REVOKED.value)


async def check_consent(
    user_id: str,
    agent_id: str,
    service: str,
    action: str = "",
    resource: str = "",
) -> tuple[bool, ConsentGrant | None]:
    """Check if an agent has consent for a specific operation.

    Returns (has_consent, matching_grant).

    Matches consent grants in order of specificity:
    1. Resource-level grants
    2. Action-level grants
    3. Service-level grants
    4. Blanket grants
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT * FROM consent_grants
               WHERE user_id = ? AND agent_id = ? AND service = ?
                 AND status = ?
               ORDER BY
                 CASE scope_type
                   WHEN 'resource' THEN 1
                   WHEN 'one_time' THEN 2
                   WHEN 'action' THEN 3
                   WHEN 'service' THEN 4
                   WHEN 'blanket' THEN 5
                 END""",
            (user_id, agent_id, service, ConsentStatus.APPROVED.value),
        )
        rows = await cursor.fetchall()

    now = time.time()
    for row in rows:
        grant = _grant_from_row(row)

        # Check expiration
        if grant.expires_at > 0 and now > grant.expires_at:
            # Mark as expired
            await _expire_grant(grant.id)
            continue

        # Check usage limit
        if grant.max_uses > 0 and grant.current_uses >= grant.max_uses:
            continue

        # Check action pattern match
        if action and grant.action_pattern != "*":
            if not _pattern_matches(grant.action_pattern, action):
                continue

        # Check resource pattern match
        if resource and grant.resource_pattern != "*":
            if not _pattern_matches(grant.resource_pattern, resource):
                continue

        # Check conditions
        if grant.conditions:
            if not _check_conditions(grant.conditions, now):
                continue

        return True, grant

    return False, None


async def use_consent(grant_id: str) -> bool:
    """Record a usage of a consent grant. Returns False if grant is exhausted."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM consent_grants WHERE id = ? AND status = ?",
            (grant_id, ConsentStatus.APPROVED.value),
        )
        row = await cursor.fetchone()
        if not row:
            return False

        new_count = row["current_uses"] + 1

        # Check if this would exceed max_uses
        if row["max_uses"] > 0 and new_count > row["max_uses"]:
            return False

        await db.execute(
            "UPDATE consent_grants SET current_uses = ? WHERE id = ?",
            (new_count, grant_id),
        )

        # Auto-expire one-time grants
        if row["scope_type"] == ConsentScope.ONE_TIME.value and new_count >= 1:
            await db.execute(
                "UPDATE consent_grants SET status = ? WHERE id = ?",
                (ConsentStatus.EXPIRED.value, grant_id),
            )

        # Auto-expire grants that hit max_uses
        if row["max_uses"] > 0 and new_count >= row["max_uses"]:
            await db.execute(
                "UPDATE consent_grants SET status = ? WHERE id = ?",
                (ConsentStatus.EXPIRED.value, grant_id),
            )

        await db.commit()

    await _log_consent_audit(
        row["user_id"], row["agent_id"], grant_id,
        "consent_used", f"use {new_count}/{row['max_uses'] or 'unlimited'}",
    )
    return True


async def get_consent_grant(grant_id: str) -> ConsentGrant | None:
    """Get a consent grant by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM consent_grants WHERE id = ?",
            (grant_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return _grant_from_row(row)


async def list_consent_grants(
    user_id: str,
    agent_id: str = "",
    service: str = "",
    status: str = "",
    include_expired: bool = False,
) -> list[ConsentGrant]:
    """List consent grants with optional filters."""
    query = "SELECT * FROM consent_grants WHERE user_id = ?"
    params: list = [user_id]

    if agent_id:
        query += " AND agent_id = ?"
        params.append(agent_id)
    if service:
        query += " AND service = ?"
        params.append(service)
    if status:
        query += " AND status = ?"
        params.append(status)
    elif not include_expired:
        query += " AND status NOT IN (?, ?)"
        params.extend([ConsentStatus.EXPIRED.value, ConsentStatus.DENIED.value])

    query += " ORDER BY created_at DESC"

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [_grant_from_row(row) for row in rows]


async def create_consent_request(
    user_id: str,
    agent_id: str,
    service: str,
    action: str = "",
    resource: str = "",
    reason: str = "",
    urgency: str = "normal",
    metadata: dict | None = None,
) -> ConsentRequest:
    """Create a consent request from an agent to a user.

    This is used when an agent needs to perform an action that requires
    explicit user consent. The user can then approve or deny the request.
    """
    if not user_id:
        raise ConsentError("user_id is required")
    if not agent_id:
        raise ConsentError("agent_id is required")
    if not service:
        raise ConsentError("service is required")

    valid_urgencies = {"low", "normal", "high", "critical"}
    if urgency not in valid_urgencies:
        raise ConsentError(f"Invalid urgency '{urgency}'. Must be one of: {valid_urgencies}")

    request_id = secrets.token_urlsafe(16)
    now = time.time()

    request = ConsentRequest(
        id=request_id,
        user_id=user_id,
        agent_id=agent_id,
        service=service,
        action=action,
        resource=resource,
        reason=reason,
        urgency=urgency,
        status="pending",
        created_at=now,
        metadata=metadata or {},
    )

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO consent_requests
               (id, user_id, agent_id, service, action, resource, reason,
                urgency, status, grant_id, created_at, resolved_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                request.id, request.user_id, request.agent_id, request.service,
                request.action, request.resource, request.reason, request.urgency,
                request.status, request.grant_id, request.created_at,
                request.resolved_at, json.dumps(request.metadata),
            ),
        )
        await db.commit()

    await _log_consent_audit(
        user_id, agent_id, request_id,
        "consent_requested",
        f"service={service}, action={action}, urgency={urgency}",
    )

    return request


async def resolve_consent_request(
    request_id: str,
    user_id: str,
    approved: bool,
    grant_options: dict | None = None,
) -> tuple[ConsentRequest | None, ConsentGrant | None]:
    """Resolve a consent request by approving or denying it.

    If approved, creates a consent grant based on grant_options.
    Returns (updated_request, grant_or_none).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM consent_requests WHERE id = ? AND user_id = ?",
            (request_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None, None

        if row["status"] != "pending":
            raise ConsentError(f"Request already resolved with status '{row['status']}'")

        now = time.time()
        new_status = "approved" if approved else "denied"

        grant = None
        grant_id = ""

        if approved:
            options = grant_options or {}
            grant = await create_consent_grant(
                user_id=user_id,
                agent_id=row["agent_id"],
                service=row["service"],
                scope_type=options.get("scope_type", "action"),
                action_pattern=options.get("action_pattern", row["action"] or "*"),
                resource_pattern=options.get("resource_pattern", row["resource"] or "*"),
                max_uses=options.get("max_uses", 0),
                expires_at=options.get("expires_at", 0.0),
                auto_approve=True,
                metadata=options.get("metadata", {}),
            )
            grant_id = grant.id

        await db.execute(
            "UPDATE consent_requests SET status = ?, resolved_at = ?, grant_id = ? WHERE id = ?",
            (new_status, now, grant_id, request_id),
        )
        await db.commit()

    request_obj = ConsentRequest(
        id=row["id"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        service=row["service"],
        action=row["action"],
        resource=row["resource"],
        reason=row["reason"],
        urgency=row["urgency"],
        status=new_status,
        grant_id=grant_id,
        created_at=row["created_at"],
        resolved_at=now,
        metadata=json.loads(row["metadata"]),
    )

    return request_obj, grant


async def list_consent_requests(
    user_id: str,
    status: str = "",
    agent_id: str = "",
) -> list[ConsentRequest]:
    """List consent requests for a user."""
    query = "SELECT * FROM consent_requests WHERE user_id = ?"
    params: list = [user_id]

    if status:
        query += " AND status = ?"
        params.append(status)
    if agent_id:
        query += " AND agent_id = ?"
        params.append(agent_id)

    query += " ORDER BY created_at DESC"

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [
            ConsentRequest(
                id=row["id"],
                user_id=row["user_id"],
                agent_id=row["agent_id"],
                service=row["service"],
                action=row["action"],
                resource=row["resource"],
                reason=row["reason"],
                urgency=row["urgency"],
                status=row["status"],
                grant_id=row["grant_id"],
                created_at=row["created_at"],
                resolved_at=row["resolved_at"],
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]


async def get_consent_audit_log(
    user_id: str,
    limit: int = 100,
) -> list[ConsentAuditEntry]:
    """Get consent audit entries for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT * FROM consent_audit
               WHERE user_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            ConsentAuditEntry(
                id=row["id"],
                timestamp=row["timestamp"],
                user_id=row["user_id"],
                agent_id=row["agent_id"],
                consent_id=row["consent_id"],
                action=row["action"],
                details=row["details"],
            )
            for row in rows
        ]


async def revoke_all_agent_consent(user_id: str, agent_id: str) -> int:
    """Revoke all consent grants for a specific agent. Returns count revoked."""
    async with aiosqlite.connect(DB_PATH) as db:
        now = time.time()
        cursor = await db.execute(
            """UPDATE consent_grants
               SET status = ?, revoked_at = ?, revoke_reason = 'bulk_revoke'
               WHERE user_id = ? AND agent_id = ? AND status IN (?, ?)""",
            (
                ConsentStatus.REVOKED.value, now,
                user_id, agent_id,
                ConsentStatus.APPROVED.value, ConsentStatus.PENDING.value,
            ),
        )
        count = cursor.rowcount
        await db.commit()

    if count > 0:
        await _log_consent_audit(
            user_id, agent_id, "",
            "bulk_revoke",
            f"Revoked {count} consent grant(s)",
        )

    return count


async def get_consent_summary(user_id: str) -> dict:
    """Get a summary of consent grants for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Count by status
        cursor = await db.execute(
            """SELECT status, COUNT(*) as count
               FROM consent_grants WHERE user_id = ?
               GROUP BY status""",
            (user_id,),
        )
        status_counts = {row["status"]: row["count"] for row in await cursor.fetchall()}

        # Count by agent
        cursor = await db.execute(
            """SELECT agent_id, COUNT(*) as count
               FROM consent_grants WHERE user_id = ? AND status = ?
               GROUP BY agent_id""",
            (user_id, ConsentStatus.APPROVED.value),
        )
        agent_counts = {row["agent_id"]: row["count"] for row in await cursor.fetchall()}

        # Count by service
        cursor = await db.execute(
            """SELECT service, COUNT(*) as count
               FROM consent_grants WHERE user_id = ? AND status = ?
               GROUP BY service""",
            (user_id, ConsentStatus.APPROVED.value),
        )
        service_counts = {row["service"]: row["count"] for row in await cursor.fetchall()}

        # Pending requests
        cursor = await db.execute(
            "SELECT COUNT(*) as count FROM consent_requests WHERE user_id = ? AND status = 'pending'",
            (user_id,),
        )
        row = await cursor.fetchone()
        pending_requests = row["count"] if row else 0

    return {
        "total_grants": sum(status_counts.values()),
        "status_breakdown": status_counts,
        "active_grants": status_counts.get(ConsentStatus.APPROVED.value, 0),
        "pending_grants": status_counts.get(ConsentStatus.PENDING.value, 0),
        "revoked_grants": status_counts.get(ConsentStatus.REVOKED.value, 0),
        "agents_with_consent": agent_counts,
        "services_with_consent": service_counts,
        "pending_requests": pending_requests,
    }


# --- Helper functions ---


def _grant_from_row(
    row,
    override_status: str = "",
    override_approved_at: float = 0.0,
) -> ConsentGrant:
    """Build a ConsentGrant from a database row."""
    return ConsentGrant(
        id=row["id"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        service=row["service"],
        scope_type=row["scope_type"],
        action_pattern=row["action_pattern"],
        resource_pattern=row["resource_pattern"],
        conditions=json.loads(row["conditions"]),
        status=override_status or row["status"],
        max_uses=row["max_uses"],
        current_uses=row["current_uses"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        approved_at=override_approved_at or row["approved_at"],
        revoked_at=row["revoked_at"],
        revoke_reason=row["revoke_reason"],
        metadata=json.loads(row["metadata"]),
    )


async def _expire_grant(grant_id: str):
    """Mark a grant as expired."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE consent_grants SET status = ? WHERE id = ?",
            (ConsentStatus.EXPIRED.value, grant_id),
        )
        await db.commit()


def _pattern_matches(pattern: str, value: str) -> bool:
    """Simple glob-like pattern matching.

    Supports:
    - '*' matches everything
    - 'prefix*' matches strings starting with prefix
    - '*suffix' matches strings ending with suffix
    - 'exact' matches exact string
    - 'a,b,c' matches any of the comma-separated values
    """
    if pattern == "*":
        return True

    # Comma-separated alternatives
    if "," in pattern:
        alternatives = [p.strip() for p in pattern.split(",")]
        return any(_pattern_matches(alt, value) for alt in alternatives)

    # Wildcard at end
    if pattern.endswith("*") and not pattern.startswith("*"):
        return value.startswith(pattern[:-1])

    # Wildcard at start
    if pattern.startswith("*") and not pattern.endswith("*"):
        return value.endswith(pattern[1:])

    # Wildcards on both sides
    if pattern.startswith("*") and pattern.endswith("*") and len(pattern) > 2:
        return pattern[1:-1] in value

    # Exact match
    return pattern == value


def _check_conditions(conditions: dict, now: float) -> bool:
    """Check additional conditions on a consent grant.

    Supported conditions:
    - time_after: Unix timestamp - only valid after this time
    - time_before: Unix timestamp - only valid before this time
    - allowed_hours: list of hours (0-23) when consent is valid
    - allowed_days: list of days (0=Mon..6=Sun) when consent is valid
    """
    from datetime import datetime, timezone

    if "time_after" in conditions:
        if now < conditions["time_after"]:
            return False

    if "time_before" in conditions:
        if now > conditions["time_before"]:
            return False

    dt = datetime.fromtimestamp(now, tz=timezone.utc)

    if "allowed_hours" in conditions:
        if dt.hour not in conditions["allowed_hours"]:
            return False

    if "allowed_days" in conditions:
        if dt.weekday() not in conditions["allowed_days"]:
            return False

    return True
