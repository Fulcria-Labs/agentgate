"""Tests for agent delegation chains."""

import json
import time
import pytest
import pytest_asyncio

from src.database import (
    init_db, create_agent_policy, AgentPolicy, DB_PATH,
)
from src.delegation import (
    create_delegation,
    DelegationError,
    get_delegated_permissions,
    init_delegation_tables,
    list_delegations,
    revoke_delegation,
    MAX_DELEGATION_DEPTH,
    _get_delegation_chain,
)


USER_ID = "user|123"
OTHER_USER = "user|999"


def make_policy(agent_id, name="TestAgent", services=None, scopes=None,
                user_id=USER_ID, is_active=True):
    return AgentPolicy(
        agent_id=agent_id,
        agent_name=name,
        allowed_services=services or ["github", "slack"],
        allowed_scopes=scopes or {
            "github": ["repo", "read:user", "write:org"],
            "slack": ["chat:write", "channels:read"],
        },
        rate_limit_per_minute=60,
        created_by=user_id,
        created_at=time.time(),
        is_active=is_active,
    )


@pytest_asyncio.fixture
async def delegation_db(db, monkeypatch):
    """Set up delegation tables alongside normal DB."""
    monkeypatch.setattr("src.delegation.DB_PATH", db)
    await init_delegation_tables()
    return db


@pytest_asyncio.fixture
async def parent_agent(delegation_db):
    policy = make_policy("agent-parent", "ParentBot")
    await create_agent_policy(policy)
    return policy


@pytest_asyncio.fixture
async def child_agent(delegation_db):
    policy = make_policy("agent-child", "ChildBot")
    await create_agent_policy(policy)
    return policy


# ── Creation tests ──


@pytest.mark.asyncio
async def test_basic_delegation(parent_agent, child_agent):
    result = await create_delegation(
        user_id=USER_ID,
        parent_agent_id="agent-parent",
        child_agent_id="agent-child",
        services=["github"],
        scopes={"github": ["repo"]},
    )
    assert result["parent_agent_id"] == "agent-parent"
    assert result["child_agent_id"] == "agent-child"
    assert "delegation_id" in result
    assert result["delegated_services"] == ["github"]


@pytest.mark.asyncio
async def test_cannot_delegate_to_self(parent_agent):
    with pytest.raises(DelegationError, match="Cannot delegate to self"):
        await create_delegation(
            USER_ID, "agent-parent", "agent-parent",
            ["github"], {"github": ["repo"]},
        )


@pytest.mark.asyncio
async def test_parent_not_found(delegation_db):
    with pytest.raises(DelegationError, match="Parent agent not found"):
        await create_delegation(
            USER_ID, "nonexistent", "agent-child",
            ["github"], {},
        )


@pytest.mark.asyncio
async def test_parent_disabled(delegation_db):
    policy = make_policy("agent-disabled", is_active=False)
    await create_agent_policy(policy)
    policy2 = make_policy("agent-child2")
    await create_agent_policy(policy2)
    with pytest.raises(DelegationError, match="disabled"):
        await create_delegation(
            USER_ID, "agent-disabled", "agent-child2",
            ["github"], {},
        )


@pytest.mark.asyncio
async def test_wrong_owner(parent_agent, child_agent):
    with pytest.raises(DelegationError, match="Not authorized"):
        await create_delegation(
            OTHER_USER, "agent-parent", "agent-child",
            ["github"], {"github": ["repo"]},
        )


@pytest.mark.asyncio
async def test_cannot_delegate_service_parent_lacks(parent_agent, child_agent):
    with pytest.raises(DelegationError, match="does not have access"):
        await create_delegation(
            USER_ID, "agent-parent", "agent-child",
            ["google"], {},
        )


@pytest.mark.asyncio
async def test_cannot_delegate_excess_scopes(parent_agent, child_agent):
    with pytest.raises(DelegationError, match="exceeds parent"):
        await create_delegation(
            USER_ID, "agent-parent", "agent-child",
            ["github"], {"github": ["repo", "admin:org"]},
        )


@pytest.mark.asyncio
async def test_scope_for_nonexistent_service(parent_agent, child_agent):
    with pytest.raises(DelegationError, match="does not have access"):
        await create_delegation(
            USER_ID, "agent-parent", "agent-child",
            ["github"], {"google": ["mail.read"]},
        )


@pytest.mark.asyncio
async def test_delegation_with_expiration(parent_agent, child_agent):
    future = time.time() + 3600
    result = await create_delegation(
        USER_ID, "agent-parent", "agent-child",
        ["github"], {"github": ["repo"]},
        expires_at=future,
    )
    assert result["delegation_id"]


@pytest.mark.asyncio
async def test_delegation_multiple_services(parent_agent, child_agent):
    result = await create_delegation(
        USER_ID, "agent-parent", "agent-child",
        ["github", "slack"],
        {"github": ["repo"], "slack": ["chat:write"]},
    )
    assert set(result["delegated_services"]) == {"github", "slack"}


@pytest.mark.asyncio
async def test_delegation_empty_scopes(parent_agent, child_agent):
    result = await create_delegation(
        USER_ID, "agent-parent", "agent-child",
        ["github"], {},
    )
    assert result["delegated_services"] == ["github"]


@pytest.mark.asyncio
async def test_delegation_all_parent_scopes(parent_agent, child_agent):
    result = await create_delegation(
        USER_ID, "agent-parent", "agent-child",
        ["github"], {"github": ["repo", "read:user", "write:org"]},
    )
    assert result["delegation_id"]


# ── Delegation chain depth tests ──


@pytest.mark.asyncio
async def test_chain_depth_limit(delegation_db):
    agents = []
    for i in range(MAX_DELEGATION_DEPTH + 2):
        agent_id = f"chain-agent-{i}"
        policy = make_policy(agent_id, f"ChainAgent{i}")
        await create_agent_policy(policy)
        agents.append(agent_id)

    for i in range(MAX_DELEGATION_DEPTH):
        await create_delegation(
            USER_ID, agents[i], agents[i + 1],
            ["github"], {"github": ["repo"]},
        )

    with pytest.raises(DelegationError, match="depth limit"):
        await create_delegation(
            USER_ID, agents[MAX_DELEGATION_DEPTH],
            agents[MAX_DELEGATION_DEPTH + 1],
            ["github"], {"github": ["repo"]},
        )


@pytest.mark.asyncio
async def test_custom_max_depth(delegation_db):
    for i in range(3):
        await create_agent_policy(make_policy(f"depth-agent-{i}"))

    await create_delegation(
        USER_ID, "depth-agent-0", "depth-agent-1",
        ["github"], {"github": ["repo"]},
        max_depth=1,
    )
    with pytest.raises(DelegationError, match="depth limit"):
        await create_delegation(
            USER_ID, "depth-agent-1", "depth-agent-2",
            ["github"], {"github": ["repo"]},
            max_depth=1,
        )


# ── Circular delegation tests ──


@pytest.mark.asyncio
async def test_circular_a_to_b_to_a(delegation_db):
    await create_agent_policy(make_policy("circ-a"))
    await create_agent_policy(make_policy("circ-b"))

    await create_delegation(
        USER_ID, "circ-a", "circ-b",
        ["github"], {"github": ["repo"]},
    )
    with pytest.raises(DelegationError, match="Circular delegation"):
        await create_delegation(
            USER_ID, "circ-b", "circ-a",
            ["github"], {"github": ["repo"]},
        )


@pytest.mark.asyncio
async def test_circular_three_way(delegation_db):
    for name in ["circ-x", "circ-y", "circ-z"]:
        await create_agent_policy(make_policy(name))

    await create_delegation(USER_ID, "circ-x", "circ-y", ["github"], {})
    await create_delegation(USER_ID, "circ-y", "circ-z", ["github"], {})
    with pytest.raises(DelegationError, match="Circular delegation"):
        await create_delegation(USER_ID, "circ-z", "circ-x", ["github"], {})


# ── Get delegated permissions tests ──


@pytest.mark.asyncio
async def test_basic_permissions(parent_agent, child_agent):
    await create_delegation(
        USER_ID, "agent-parent", "agent-child",
        ["github"], {"github": ["repo", "read:user"]},
    )
    perms = await get_delegated_permissions("agent-child")
    assert perms is not None
    assert perms["parent_agent_id"] == "agent-parent"
    assert "github" in perms["effective_services"]
    assert "repo" in perms["effective_scopes"]["github"]
    assert "read:user" in perms["effective_scopes"]["github"]


@pytest.mark.asyncio
async def test_no_delegation_returns_none(delegation_db):
    perms = await get_delegated_permissions("random-agent")
    assert perms is None


@pytest.mark.asyncio
async def test_expired_delegation_returns_none(parent_agent, child_agent):
    await create_delegation(
        USER_ID, "agent-parent", "agent-child",
        ["github"], {"github": ["repo"]},
        expires_at=time.time() - 3600,
    )
    perms = await get_delegated_permissions("agent-child")
    assert perms is None


@pytest.mark.asyncio
async def test_parent_disabled_returns_none(delegation_db):
    parent = make_policy("perm-parent")
    await create_agent_policy(parent)
    child = make_policy("perm-child")
    await create_agent_policy(child)

    await create_delegation(
        USER_ID, "perm-parent", "perm-child",
        ["github"], {"github": ["repo"]},
    )

    import aiosqlite
    async with aiosqlite.connect(delegation_db) as db:
        await db.execute(
            "UPDATE agent_policies SET is_active = 0 WHERE agent_id = ?",
            ("perm-parent",),
        )
        await db.commit()

    perms = await get_delegated_permissions("perm-child")
    assert perms is None


@pytest.mark.asyncio
async def test_effective_scopes_are_intersection(delegation_db):
    parent = make_policy("scope-parent", scopes={"github": ["repo", "read:user"]})
    await create_agent_policy(parent)
    child = make_policy("scope-child")
    await create_agent_policy(child)

    await create_delegation(
        USER_ID, "scope-parent", "scope-child",
        ["github"], {"github": ["repo", "read:user"]},
    )

    parent.allowed_scopes = {"github": ["repo"]}
    await create_agent_policy(parent)

    perms = await get_delegated_permissions("scope-child")
    assert perms is not None
    assert perms["effective_scopes"]["github"] == ["repo"]


# ── Revocation tests ──


@pytest.mark.asyncio
async def test_basic_revocation(parent_agent, child_agent):
    result = await create_delegation(
        USER_ID, "agent-parent", "agent-child",
        ["github"], {"github": ["repo"]},
    )
    delegation_id = result["delegation_id"]

    revoked = await revoke_delegation(delegation_id, USER_ID)
    assert revoked is True

    perms = await get_delegated_permissions("agent-child")
    assert perms is None


@pytest.mark.asyncio
async def test_revoke_wrong_user(parent_agent, child_agent):
    result = await create_delegation(
        USER_ID, "agent-parent", "agent-child",
        ["github"], {},
    )
    revoked = await revoke_delegation(result["delegation_id"], OTHER_USER)
    assert revoked is False


@pytest.mark.asyncio
async def test_revoke_nonexistent(delegation_db):
    revoked = await revoke_delegation("nonexistent-id", USER_ID)
    assert revoked is False


@pytest.mark.asyncio
async def test_cascade_revocation(delegation_db):
    for name in ["cascade-a", "cascade-b", "cascade-c"]:
        await create_agent_policy(make_policy(name))

    r1 = await create_delegation(
        USER_ID, "cascade-a", "cascade-b",
        ["github"], {"github": ["repo"]},
    )
    await create_delegation(
        USER_ID, "cascade-b", "cascade-c",
        ["github"], {"github": ["repo"]},
    )

    assert (await get_delegated_permissions("cascade-b")) is not None
    assert (await get_delegated_permissions("cascade-c")) is not None

    await revoke_delegation(r1["delegation_id"], USER_ID)

    assert (await get_delegated_permissions("cascade-b")) is None
    assert (await get_delegated_permissions("cascade-c")) is None


# ── List delegations tests ──


@pytest.mark.asyncio
async def test_list_empty(delegation_db):
    delegations = await list_delegations(USER_ID)
    assert delegations == []


@pytest.mark.asyncio
async def test_list_own_delegations(parent_agent, child_agent):
    await create_delegation(
        USER_ID, "agent-parent", "agent-child",
        ["github"], {"github": ["repo"]},
    )
    delegations = await list_delegations(USER_ID)
    assert len(delegations) == 1
    assert delegations[0]["parent_agent_id"] == "agent-parent"
    assert delegations[0]["child_agent_id"] == "agent-child"
    assert delegations[0]["is_active"] is True


@pytest.mark.asyncio
async def test_list_does_not_show_other_users(parent_agent, child_agent):
    await create_delegation(
        USER_ID, "agent-parent", "agent-child",
        ["github"], {},
    )
    delegations = await list_delegations(OTHER_USER)
    assert delegations == []


@pytest.mark.asyncio
async def test_list_shows_revoked(parent_agent, child_agent):
    result = await create_delegation(
        USER_ID, "agent-parent", "agent-child",
        ["github"], {},
    )
    await revoke_delegation(result["delegation_id"], USER_ID)
    delegations = await list_delegations(USER_ID)
    assert len(delegations) == 1
    assert delegations[0]["is_active"] is False


@pytest.mark.asyncio
async def test_list_multiple_delegations(delegation_db):
    for i in range(4):
        await create_agent_policy(make_policy(f"multi-agent-{i}"))

    await create_delegation(USER_ID, "multi-agent-0", "multi-agent-1", ["github"], {})
    await create_delegation(USER_ID, "multi-agent-2", "multi-agent-3", ["slack"], {})
    delegations = await list_delegations(USER_ID)
    assert len(delegations) == 2


# ── Chain path tests ──


@pytest.mark.asyncio
async def test_root_delegation_has_parent_in_chain(parent_agent, child_agent):
    result = await create_delegation(
        USER_ID, "agent-parent", "agent-child",
        ["github"], {},
    )
    assert result["chain_path"] == ["agent-parent"]


@pytest.mark.asyncio
async def test_two_hop_chain(delegation_db):
    for name in ["root", "mid", "leaf"]:
        await create_agent_policy(make_policy(name))

    r1 = await create_delegation(USER_ID, "root", "mid", ["github"], {"github": ["repo"]})
    assert r1["chain_path"] == ["root"]

    r2 = await create_delegation(USER_ID, "mid", "leaf", ["github"], {"github": ["repo"]})
    assert r2["chain_path"] == ["root", "mid"]


# ── Edge cases ──


@pytest.mark.asyncio
async def test_delegation_with_zero_expiration(parent_agent, child_agent):
    await create_delegation(
        USER_ID, "agent-parent", "agent-child",
        ["github"], {}, expires_at=0,
    )
    perms = await get_delegated_permissions("agent-child")
    assert perms is not None


@pytest.mark.asyncio
async def test_multiple_delegations_to_same_child(delegation_db):
    for name in ["p1", "p2", "c1"]:
        await create_agent_policy(make_policy(name))

    await create_delegation(USER_ID, "p1", "c1", ["github"], {"github": ["repo"]})
    await create_delegation(USER_ID, "p2", "c1", ["slack"], {"slack": ["chat:write"]})
    perms = await get_delegated_permissions("c1")
    assert perms is not None
    assert perms["parent_agent_id"] == "p2"


@pytest.mark.asyncio
async def test_delegation_single_scope(parent_agent, child_agent):
    await create_delegation(
        USER_ID, "agent-parent", "agent-child",
        ["github"], {"github": ["repo"]},
    )
    perms = await get_delegated_permissions("agent-child")
    assert perms["effective_scopes"]["github"] == ["repo"]


@pytest.mark.asyncio
async def test_get_delegation_chain_no_chain(delegation_db):
    chain = await _get_delegation_chain("no-such-agent")
    assert chain == []


@pytest.mark.asyncio
async def test_delegation_preserves_chain_after_revoke(delegation_db):
    for name in ["iso-a", "iso-b", "iso-c", "iso-d"]:
        await create_agent_policy(make_policy(name))

    r1 = await create_delegation(USER_ID, "iso-a", "iso-b", ["github"], {"github": ["repo"]})
    await create_delegation(USER_ID, "iso-c", "iso-d", ["slack"], {"slack": ["chat:write"]})

    await revoke_delegation(r1["delegation_id"], USER_ID)

    perms = await get_delegated_permissions("iso-d")
    assert perms is not None
    assert perms["parent_agent_id"] == "iso-c"
