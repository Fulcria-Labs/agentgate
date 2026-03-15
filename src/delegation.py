"""Agent delegation chains for multi-agent authorization.

Enables an agent to delegate a subset of its permissions to another agent,
creating a verifiable chain of delegation with scope narrowing at each hop.
"""

import json
import secrets
import time

import aiosqlite

from .database import DB_PATH, AgentPolicy, get_agent_policy, log_audit


MAX_DELEGATION_DEPTH = 3


class DelegationError(Exception):
    """Raised when a delegation operation fails."""
    def __init__(self, reason: str, parent_id: str = "", child_id: str = ""):
        self.reason = reason
        self.parent_id = parent_id
        self.child_id = child_id
        super().__init__(reason)


async def init_delegation_tables():
    """Create delegation tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS delegation_chains (
                id TEXT PRIMARY KEY,
                parent_agent_id TEXT NOT NULL,
                child_agent_id TEXT NOT NULL,
                delegated_services TEXT NOT NULL DEFAULT '[]',
                delegated_scopes TEXT NOT NULL DEFAULT '{}',
                max_depth INTEGER NOT NULL DEFAULT 3,
                created_by TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                chain_path TEXT NOT NULL DEFAULT '[]'
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_delegation_parent
            ON delegation_chains(parent_agent_id, is_active)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_delegation_child
            ON delegation_chains(child_agent_id, is_active)
        """)
        await db.commit()


async def create_delegation(
    user_id: str,
    parent_agent_id: str,
    child_agent_id: str,
    services: list[str],
    scopes: dict[str, list[str]],
    expires_at: float = 0.0,
    max_depth: int = MAX_DELEGATION_DEPTH,
) -> dict:
    """Create a delegation from a parent agent to a child agent.

    The child agent receives a subset of the parent's permissions.
    Validates:
      - Parent agent exists and is active
      - User owns the parent agent's policy
      - No circular delegation
      - Delegation depth limit not exceeded
      - Delegated scopes are a subset of parent's scopes
    """
    if parent_agent_id == child_agent_id:
        raise DelegationError("Cannot delegate to self", parent_agent_id, child_agent_id)

    # Validate parent agent
    parent_policy = await get_agent_policy(parent_agent_id)
    if not parent_policy:
        raise DelegationError("Parent agent not found", parent_agent_id, child_agent_id)
    if not parent_policy.is_active:
        raise DelegationError("Parent agent is disabled", parent_agent_id, child_agent_id)
    if parent_policy.created_by != user_id:
        raise DelegationError(
            "Not authorized: you do not own the parent agent's policy",
            parent_agent_id, child_agent_id,
        )

    # Validate delegated services are subset of parent's services
    parent_services = set(parent_policy.allowed_services)
    for svc in services:
        if svc not in parent_services:
            raise DelegationError(
                f"Cannot delegate service '{svc}': parent agent does not have access",
                parent_agent_id, child_agent_id,
            )

    # Validate delegated scopes are subset of parent's scopes
    for svc, scope_list in scopes.items():
        if svc not in parent_services:
            raise DelegationError(
                f"Cannot delegate scopes for '{svc}': parent agent does not have access",
                parent_agent_id, child_agent_id,
            )
        parent_scopes = set(parent_policy.allowed_scopes.get(svc, []))
        excess = set(scope_list) - parent_scopes
        if excess:
            raise DelegationError(
                f"Cannot delegate scopes {excess} for '{svc}': exceeds parent's permissions",
                parent_agent_id, child_agent_id,
            )

    # Check for circular delegation
    chain = await _get_delegation_chain(parent_agent_id)
    if child_agent_id in chain:
        raise DelegationError(
            "Circular delegation detected",
            parent_agent_id, child_agent_id,
        )

    # Check depth limit
    current_depth = len(chain)
    if current_depth >= max_depth:
        raise DelegationError(
            f"Delegation depth limit exceeded (max {max_depth})",
            parent_agent_id, child_agent_id,
        )

    # Create delegation
    delegation_id = secrets.token_urlsafe(16)
    chain_path = chain + [parent_agent_id]

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO delegation_chains
               (id, parent_agent_id, child_agent_id, delegated_services,
                delegated_scopes, max_depth, created_by, created_at,
                expires_at, is_active, chain_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
            (
                delegation_id,
                parent_agent_id,
                child_agent_id,
                json.dumps(services),
                json.dumps(scopes),
                max_depth,
                user_id,
                time.time(),
                expires_at,
                json.dumps(chain_path),
            ),
        )
        await db.commit()

    await log_audit(
        user_id, parent_agent_id, "",
        "delegation_created", "success",
        details=f"Delegated to {child_agent_id}: services={services}",
    )

    return {
        "delegation_id": delegation_id,
        "parent_agent_id": parent_agent_id,
        "child_agent_id": child_agent_id,
        "delegated_services": services,
        "chain_depth": current_depth + 1,
        "chain_path": chain_path,
    }


async def _get_delegation_chain(agent_id: str) -> list[str]:
    """Trace the delegation chain back to the root agent.

    Returns list of agent IDs from root to the given agent (exclusive).
    """
    visited = set()
    chain = []
    current = agent_id

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        while True:
            if current in visited:
                break  # Prevent infinite loops
            visited.add(current)

            cursor = await db.execute(
                """SELECT parent_agent_id FROM delegation_chains
                   WHERE child_agent_id = ? AND is_active = 1
                   ORDER BY created_at DESC LIMIT 1""",
                (current,),
            )
            row = await cursor.fetchone()
            if not row:
                break
            chain.insert(0, row["parent_agent_id"])
            current = row["parent_agent_id"]

    return chain


async def get_delegated_permissions(child_agent_id: str) -> dict | None:
    """Get the effective permissions for a child agent via delegation.

    Returns the intersection of the parent's policy and the delegation grant.
    Returns None if no active delegation exists for this child agent.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT * FROM delegation_chains
               WHERE child_agent_id = ? AND is_active = 1
               ORDER BY created_at DESC LIMIT 1""",
            (child_agent_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        # Check expiration
        if row["expires_at"] > 0 and time.time() > row["expires_at"]:
            return None

        parent_policy = await get_agent_policy(row["parent_agent_id"])
        if not parent_policy or not parent_policy.is_active:
            return None

        # Effective services = intersection of parent's services and delegated services
        delegated_services = json.loads(row["delegated_services"])
        effective_services = [
            s for s in delegated_services
            if s in parent_policy.allowed_services
        ]

        # Effective scopes = intersection of parent's scopes and delegated scopes
        delegated_scopes = json.loads(row["delegated_scopes"])
        effective_scopes = {}
        for svc in effective_services:
            parent_svc_scopes = set(parent_policy.allowed_scopes.get(svc, []))
            delegated_svc_scopes = set(delegated_scopes.get(svc, []))
            effective_scopes[svc] = list(parent_svc_scopes & delegated_svc_scopes)

        return {
            "delegation_id": row["id"],
            "parent_agent_id": row["parent_agent_id"],
            "child_agent_id": child_agent_id,
            "effective_services": effective_services,
            "effective_scopes": effective_scopes,
            "chain_path": json.loads(row["chain_path"]),
            "expires_at": row["expires_at"],
        }


async def revoke_delegation(delegation_id: str, user_id: str) -> bool:
    """Revoke a delegation. Also cascades to child delegations."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Find the delegation
        cursor = await db.execute(
            "SELECT * FROM delegation_chains WHERE id = ? AND created_by = ?",
            (delegation_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return False

        child_agent_id = row["child_agent_id"]

        # Revoke the delegation
        await db.execute(
            "UPDATE delegation_chains SET is_active = 0 WHERE id = ?",
            (delegation_id,),
        )

        # Cascade: revoke any delegations where the child was a parent
        await _cascade_revoke(db, child_agent_id)

        await db.commit()

    await log_audit(
        user_id, row["parent_agent_id"], "",
        "delegation_revoked", "success",
        details=f"Revoked delegation to {child_agent_id} (cascade applied)",
    )

    return True


async def _cascade_revoke(db, agent_id: str, visited: set | None = None):
    """Recursively revoke all delegations where agent_id is a parent."""
    if visited is None:
        visited = set()
    if agent_id in visited:
        return
    visited.add(agent_id)

    cursor = await db.execute(
        "SELECT id, child_agent_id FROM delegation_chains WHERE parent_agent_id = ? AND is_active = 1",
        (agent_id,),
    )
    rows = await cursor.fetchall()
    for row in rows:
        await db.execute(
            "UPDATE delegation_chains SET is_active = 0 WHERE id = ?",
            (row["id"],),
        )
        await _cascade_revoke(db, row["child_agent_id"], visited)


async def list_delegations(user_id: str) -> list[dict]:
    """List all delegations created by a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT * FROM delegation_chains
               WHERE created_by = ? ORDER BY created_at DESC""",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "delegation_id": row["id"],
                "parent_agent_id": row["parent_agent_id"],
                "child_agent_id": row["child_agent_id"],
                "delegated_services": json.loads(row["delegated_services"]),
                "delegated_scopes": json.loads(row["delegated_scopes"]),
                "chain_path": json.loads(row["chain_path"]),
                "max_depth": row["max_depth"],
                "is_active": bool(row["is_active"]),
                "created_at": row["created_at"],
                "expires_at": row["expires_at"],
            }
            for row in rows
        ]
