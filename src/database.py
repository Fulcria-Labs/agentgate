"""Database models and connection for audit logging and agent policies."""

import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field

import aiosqlite

DB_PATH = "agentgate.db"


@dataclass
class ApiKey:
    """An API key issued to an agent for authentication."""
    id: str = ""
    key_hash: str = ""
    key_prefix: str = ""  # First 8 chars for display
    user_id: str = ""
    agent_id: str = ""
    name: str = ""
    created_at: float = 0.0
    expires_at: float = 0.0  # 0 = never
    is_revoked: bool = False
    last_used_at: float = 0.0


@dataclass
class AgentPolicy:
    """Defines what services and scopes an agent can access."""
    agent_id: str
    agent_name: str
    allowed_services: list[str] = field(default_factory=list)
    allowed_scopes: dict[str, list[str]] = field(default_factory=dict)
    rate_limit_per_minute: int = 60
    requires_step_up: list[str] = field(default_factory=list)
    created_by: str = ""
    created_at: float = 0.0
    is_active: bool = True


@dataclass
class AuditEntry:
    """Records every token request and action."""
    id: int = 0
    timestamp: float = 0.0
    user_id: str = ""
    agent_id: str = ""
    service: str = ""
    scopes: str = ""
    action: str = ""
    status: str = ""
    ip_address: str = ""
    details: str = ""


async def init_db():
    """Initialize the database schema."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agent_policies (
                agent_id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                allowed_services TEXT NOT NULL DEFAULT '[]',
                allowed_scopes TEXT NOT NULL DEFAULT '{}',
                rate_limit_per_minute INTEGER NOT NULL DEFAULT 60,
                requires_step_up TEXT NOT NULL DEFAULT '[]',
                created_by TEXT NOT NULL,
                created_at REAL NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                service TEXT NOT NULL,
                scopes TEXT NOT NULL DEFAULT '',
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                ip_address TEXT NOT NULL DEFAULT '',
                details TEXT NOT NULL DEFAULT ''
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS connected_services (
                user_id TEXT NOT NULL,
                service TEXT NOT NULL,
                connected_at REAL NOT NULL,
                connection_id TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (user_id, service)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                key_hash TEXT NOT NULL UNIQUE,
                key_prefix TEXT NOT NULL,
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL DEFAULT 0,
                is_revoked INTEGER NOT NULL DEFAULT 0,
                last_used_at REAL NOT NULL DEFAULT 0
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_user
            ON audit_log(user_id, timestamp DESC)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_agent
            ON audit_log(agent_id, timestamp DESC)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_keys_hash
            ON api_keys(key_hash)
        """)
        await db.commit()


async def log_audit(
    user_id: str,
    agent_id: str,
    service: str,
    action: str,
    status: str,
    scopes: str = "",
    ip_address: str = "",
    details: str = "",
):
    """Write an audit log entry."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO audit_log
               (timestamp, user_id, agent_id, service, scopes, action, status, ip_address, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), user_id, agent_id, service, scopes, action, status, ip_address, details),
        )
        await db.commit()


async def get_audit_log(user_id: str, limit: int = 50) -> list[AuditEntry]:
    """Retrieve recent audit entries for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM audit_log WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            AuditEntry(
                id=row["id"],
                timestamp=row["timestamp"],
                user_id=row["user_id"],
                agent_id=row["agent_id"],
                service=row["service"],
                scopes=row["scopes"],
                action=row["action"],
                status=row["status"],
                ip_address=row["ip_address"],
                details=row["details"],
            )
            for row in rows
        ]


async def create_agent_policy(policy: AgentPolicy):
    """Create or update an agent policy."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO agent_policies
               (agent_id, agent_name, allowed_services, allowed_scopes,
                rate_limit_per_minute, requires_step_up, created_by, created_at, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                policy.agent_id,
                policy.agent_name,
                json.dumps(policy.allowed_services),
                json.dumps(policy.allowed_scopes),
                policy.rate_limit_per_minute,
                json.dumps(policy.requires_step_up),
                policy.created_by,
                policy.created_at or time.time(),
                int(policy.is_active),
            ),
        )
        await db.commit()


async def get_agent_policy(agent_id: str) -> AgentPolicy | None:
    """Retrieve an agent's policy."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM agent_policies WHERE agent_id = ?", (agent_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return AgentPolicy(
            agent_id=row["agent_id"],
            agent_name=row["agent_name"],
            allowed_services=json.loads(row["allowed_services"]),
            allowed_scopes=json.loads(row["allowed_scopes"]),
            rate_limit_per_minute=row["rate_limit_per_minute"],
            requires_step_up=json.loads(row["requires_step_up"]),
            created_by=row["created_by"],
            created_at=row["created_at"],
            is_active=bool(row["is_active"]),
        )


async def get_all_policies(user_id: str) -> list[AgentPolicy]:
    """Get all agent policies created by a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM agent_policies WHERE created_by = ?", (user_id,)
        )
        rows = await cursor.fetchall()
        return [
            AgentPolicy(
                agent_id=row["agent_id"],
                agent_name=row["agent_name"],
                allowed_services=json.loads(row["allowed_services"]),
                allowed_scopes=json.loads(row["allowed_scopes"]),
                rate_limit_per_minute=row["rate_limit_per_minute"],
                requires_step_up=json.loads(row["requires_step_up"]),
                created_by=row["created_by"],
                created_at=row["created_at"],
                is_active=bool(row["is_active"]),
            )
            for row in rows
        ]


async def add_connected_service(user_id: str, service: str, connection_id: str = ""):
    """Record that a user has connected a service."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO connected_services
               (user_id, service, connected_at, connection_id)
               VALUES (?, ?, ?, ?)""",
            (user_id, service, time.time(), connection_id),
        )
        await db.commit()


async def remove_connected_service(user_id: str, service: str):
    """Remove a connected service for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM connected_services WHERE user_id = ? AND service = ?",
            (user_id, service),
        )
        await db.commit()


async def get_connected_services(user_id: str) -> list[dict]:
    """Get all connected services for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM connected_services WHERE user_id = ? ORDER BY connected_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "service": row["service"],
                "connected_at": row["connected_at"],
                "connection_id": row["connection_id"],
            }
            for row in rows
        ]


def _hash_key(raw_key: str) -> str:
    """Hash an API key using SHA-256."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def create_api_key(user_id: str, agent_id: str, name: str = "",
                         expires_in: int = 0) -> tuple[ApiKey, str]:
    """Create a new API key. Returns (ApiKey, raw_key). Raw key is only available at creation."""
    raw_key = f"ag_{secrets.token_urlsafe(32)}"
    key_id = secrets.token_urlsafe(16)
    key_hash = _hash_key(raw_key)
    now = time.time()
    expires_at = now + expires_in if expires_in != 0 else 0

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO api_keys
               (id, key_hash, key_prefix, user_id, agent_id, name, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (key_id, key_hash, raw_key[:8], user_id, agent_id, name, now, expires_at),
        )
        await db.commit()

    api_key = ApiKey(
        id=key_id, key_hash=key_hash, key_prefix=raw_key[:8],
        user_id=user_id, agent_id=agent_id, name=name,
        created_at=now, expires_at=expires_at,
    )
    return api_key, raw_key


async def validate_api_key(raw_key: str) -> ApiKey | None:
    """Validate an API key and return the associated ApiKey, or None if invalid."""
    key_hash = _hash_key(raw_key)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
        )
        row = await cursor.fetchone()
        if not row:
            return None

        api_key = ApiKey(
            id=row["id"], key_hash=row["key_hash"], key_prefix=row["key_prefix"],
            user_id=row["user_id"], agent_id=row["agent_id"], name=row["name"],
            created_at=row["created_at"], expires_at=row["expires_at"],
            is_revoked=bool(row["is_revoked"]), last_used_at=row["last_used_at"],
        )

        # Check revocation
        if api_key.is_revoked:
            return None

        # Check expiration
        if api_key.expires_at > 0 and time.time() > api_key.expires_at:
            return None

        # Update last_used_at
        now = time.time()
        await db.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
            (now, api_key.id),
        )
        await db.commit()
        api_key.last_used_at = now
        return api_key


async def revoke_api_key(key_id: str, user_id: str) -> bool:
    """Revoke an API key. Returns True if found and revoked."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "UPDATE api_keys SET is_revoked = 1 WHERE id = ? AND user_id = ?",
            (key_id, user_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def get_api_keys(user_id: str) -> list[ApiKey]:
    """Get all API keys for a user (excluding hash)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM api_keys WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [
            ApiKey(
                id=row["id"], key_hash="", key_prefix=row["key_prefix"],
                user_id=row["user_id"], agent_id=row["agent_id"], name=row["name"],
                created_at=row["created_at"], expires_at=row["expires_at"],
                is_revoked=bool(row["is_revoked"]), last_used_at=row["last_used_at"],
            )
            for row in rows
        ]


async def toggle_agent_policy(agent_id: str, user_id: str) -> bool | None:
    """Toggle an agent's active state. Returns new state, or None if not found."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT is_active FROM agent_policies WHERE agent_id = ? AND created_by = ?",
            (agent_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        new_state = not bool(row["is_active"])
        await db.execute(
            "UPDATE agent_policies SET is_active = ? WHERE agent_id = ? AND created_by = ?",
            (int(new_state), agent_id, user_id),
        )
        await db.commit()
        return new_state
