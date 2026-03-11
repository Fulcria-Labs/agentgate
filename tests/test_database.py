"""Tests for database operations."""

import time
import pytest

from src.database import (
    AgentPolicy,
    add_connected_service,
    create_agent_policy,
    create_api_key,
    delete_agent_policy,
    emergency_revoke_all,
    get_agent_policy,
    get_all_policies,
    get_api_keys,
    get_audit_log,
    get_connected_services,
    init_db,
    log_audit,
    remove_connected_service,
    revoke_api_key,
    toggle_agent_policy,
    validate_api_key,
)


@pytest.mark.asyncio
async def test_init_db(db):
    """Database initializes without error."""
    # db fixture already calls init_db
    pass


@pytest.mark.asyncio
async def test_audit_logging(db):
    """Audit entries are written and retrieved correctly."""
    await log_audit("user1", "agent1", "github", "token_request", "success",
                   scopes="repo", ip_address="127.0.0.1")
    await log_audit("user1", "agent1", "slack", "token_request", "denied",
                   details="Service not allowed")

    entries = await get_audit_log("user1")
    assert len(entries) == 2
    # Most recent first
    assert entries[0].service == "slack"
    assert entries[0].status == "denied"
    assert entries[1].service == "github"
    assert entries[1].status == "success"
    assert entries[1].scopes == "repo"


@pytest.mark.asyncio
async def test_audit_log_limit(db):
    """Audit log respects the limit parameter."""
    for i in range(10):
        await log_audit("user1", "agent1", "github", f"action_{i}", "success")

    entries = await get_audit_log("user1", limit=5)
    assert len(entries) == 5


@pytest.mark.asyncio
async def test_audit_log_user_isolation(db):
    """Audit entries are isolated per user."""
    await log_audit("user1", "agent1", "github", "action1", "success")
    await log_audit("user2", "agent1", "github", "action2", "success")

    entries1 = await get_audit_log("user1")
    entries2 = await get_audit_log("user2")
    assert len(entries1) == 1
    assert len(entries2) == 1
    assert entries1[0].action == "action1"
    assert entries2[0].action == "action2"


@pytest.mark.asyncio
async def test_create_and_get_policy(db):
    """Agent policies are created and retrieved correctly."""
    policy = AgentPolicy(
        agent_id="agent-123",
        agent_name="Code Review Bot",
        allowed_services=["github", "slack"],
        allowed_scopes={"github": ["repo", "read:user"], "slack": ["channels:read"]},
        rate_limit_per_minute=30,
        requires_step_up=["slack"],
        created_by="user1",
        created_at=time.time(),
    )
    await create_agent_policy(policy)

    retrieved = await get_agent_policy("agent-123")
    assert retrieved is not None
    assert retrieved.agent_name == "Code Review Bot"
    assert "github" in retrieved.allowed_services
    assert "slack" in retrieved.allowed_services
    assert retrieved.allowed_scopes["github"] == ["repo", "read:user"]
    assert retrieved.rate_limit_per_minute == 30
    assert "slack" in retrieved.requires_step_up


@pytest.mark.asyncio
async def test_get_nonexistent_policy(db):
    """Getting a non-existent policy returns None."""
    result = await get_agent_policy("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_update_policy(db):
    """Creating a policy with the same agent_id updates it."""
    policy1 = AgentPolicy(
        agent_id="agent-1",
        agent_name="Bot v1",
        allowed_services=["github"],
        created_by="user1",
    )
    await create_agent_policy(policy1)

    policy2 = AgentPolicy(
        agent_id="agent-1",
        agent_name="Bot v2",
        allowed_services=["github", "slack"],
        created_by="user1",
    )
    await create_agent_policy(policy2)

    retrieved = await get_agent_policy("agent-1")
    assert retrieved.agent_name == "Bot v2"
    assert len(retrieved.allowed_services) == 2


@pytest.mark.asyncio
async def test_get_all_policies(db):
    """Get all policies for a specific user."""
    for i in range(3):
        await create_agent_policy(AgentPolicy(
            agent_id=f"agent-{i}",
            agent_name=f"Bot {i}",
            created_by="user1",
        ))
    await create_agent_policy(AgentPolicy(
        agent_id="agent-other",
        agent_name="Other Bot",
        created_by="user2",
    ))

    user1_policies = await get_all_policies("user1")
    user2_policies = await get_all_policies("user2")
    assert len(user1_policies) == 3
    assert len(user2_policies) == 1


@pytest.mark.asyncio
async def test_connected_services(db):
    """Connected services CRUD operations work correctly."""
    await add_connected_service("user1", "github", "conn-123")
    await add_connected_service("user1", "slack", "conn-456")

    services = await get_connected_services("user1")
    assert len(services) == 2
    service_names = {s["service"] for s in services}
    assert "github" in service_names
    assert "slack" in service_names

    await remove_connected_service("user1", "github")
    services = await get_connected_services("user1")
    assert len(services) == 1
    assert services[0]["service"] == "slack"


@pytest.mark.asyncio
async def test_connected_services_user_isolation(db):
    """Connected services are isolated per user."""
    await add_connected_service("user1", "github")
    await add_connected_service("user2", "slack")

    s1 = await get_connected_services("user1")
    s2 = await get_connected_services("user2")
    assert len(s1) == 1
    assert len(s2) == 1
    assert s1[0]["service"] == "github"
    assert s2[0]["service"] == "slack"


@pytest.mark.asyncio
async def test_connected_service_upsert(db):
    """Re-connecting a service updates it rather than duplicating."""
    await add_connected_service("user1", "github", "conn-1")
    await add_connected_service("user1", "github", "conn-2")

    services = await get_connected_services("user1")
    assert len(services) == 1
    assert services[0]["connection_id"] == "conn-2"


# --- API Key Tests ---

@pytest.mark.asyncio
async def test_create_api_key(db):
    """API key creation returns key and metadata."""
    api_key, raw_key = await create_api_key("user1", "agent-1", "test-key")
    assert raw_key.startswith("ag_")
    assert api_key.key_prefix == raw_key[:8]
    assert api_key.agent_id == "agent-1"
    assert api_key.user_id == "user1"


@pytest.mark.asyncio
async def test_validate_api_key(db):
    """Valid API key passes validation."""
    api_key, raw_key = await create_api_key("user1", "agent-1")
    result = await validate_api_key(raw_key)
    assert result is not None
    assert result.agent_id == "agent-1"
    assert result.user_id == "user1"


@pytest.mark.asyncio
async def test_validate_invalid_key(db):
    """Invalid key returns None."""
    result = await validate_api_key("ag_invalid_key_here")
    assert result is None


@pytest.mark.asyncio
async def test_revoke_api_key(db):
    """Revoked keys fail validation."""
    api_key, raw_key = await create_api_key("user1", "agent-1")
    assert await revoke_api_key(api_key.id, "user1")
    result = await validate_api_key(raw_key)
    assert result is None


@pytest.mark.asyncio
async def test_revoke_wrong_user(db):
    """Cannot revoke another user's key."""
    api_key, raw_key = await create_api_key("user1", "agent-1")
    assert not await revoke_api_key(api_key.id, "user2")
    # Key still valid
    result = await validate_api_key(raw_key)
    assert result is not None


@pytest.mark.asyncio
async def test_expired_api_key(db):
    """Expired keys fail validation."""
    api_key, raw_key = await create_api_key("user1", "agent-1", expires_in=-1)
    result = await validate_api_key(raw_key)
    assert result is None


@pytest.mark.asyncio
async def test_get_api_keys(db):
    """List API keys for a user (no hashes exposed)."""
    await create_api_key("user1", "agent-1", "key-1")
    await create_api_key("user1", "agent-2", "key-2")
    await create_api_key("user2", "agent-3", "key-3")

    keys = await get_api_keys("user1")
    assert len(keys) == 2
    assert all(k.key_hash == "" for k in keys)  # Hash not exposed


@pytest.mark.asyncio
async def test_validate_updates_last_used(db):
    """Validating a key updates last_used_at."""
    api_key, raw_key = await create_api_key("user1", "agent-1")
    assert api_key.last_used_at == 0.0
    result = await validate_api_key(raw_key)
    assert result.last_used_at > 0


# --- Toggle Tests ---

@pytest.mark.asyncio
async def test_toggle_agent_policy(db):
    """Toggle flips agent active state."""
    await create_agent_policy(AgentPolicy(
        agent_id="agent-1", agent_name="Bot", created_by="user1",
    ))
    # Initially active (True), toggle to False
    new_state = await toggle_agent_policy("agent-1", "user1")
    assert new_state is False

    policy = await get_agent_policy("agent-1")
    assert policy.is_active is False

    # Toggle back to True
    new_state = await toggle_agent_policy("agent-1", "user1")
    assert new_state is True


@pytest.mark.asyncio
async def test_toggle_nonexistent_policy(db):
    """Toggle on non-existent policy returns None."""
    result = await toggle_agent_policy("fake", "user1")
    assert result is None


@pytest.mark.asyncio
async def test_toggle_wrong_user(db):
    """Cannot toggle another user's policy."""
    await create_agent_policy(AgentPolicy(
        agent_id="agent-1", agent_name="Bot", created_by="user1",
    ))
    result = await toggle_agent_policy("agent-1", "user2")
    assert result is None
