"""Multi-tenant isolation tests — verify that user data, policies, API keys,
audit logs, and connected services are completely isolated between users."""

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
    log_audit,
    remove_connected_service,
    revoke_api_key,
    toggle_agent_policy,
    validate_api_key,
)
from src.policy import (
    PolicyDenied,
    _rate_counters,
    enforce_policy,
)


@pytest.fixture(autouse=True)
def clear_rate_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


class TestPolicyIsolation:
    """Policies are isolated between users."""

    @pytest.mark.asyncio
    async def test_users_see_only_own_policies(self, db):
        for i in range(3):
            await create_agent_policy(AgentPolicy(
                agent_id=f"u1-agent-{i}", agent_name=f"Bot {i}", created_by="user1",
            ))
        for i in range(2):
            await create_agent_policy(AgentPolicy(
                agent_id=f"u2-agent-{i}", agent_name=f"Bot {i}", created_by="user2",
            ))
        u1 = await get_all_policies("user1")
        u2 = await get_all_policies("user2")
        assert len(u1) == 3
        assert len(u2) == 2
        assert all(p.created_by == "user1" for p in u1)
        assert all(p.created_by == "user2" for p in u2)

    @pytest.mark.asyncio
    async def test_user_cannot_enforce_other_users_policy(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="private-agent", agent_name="Private Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            created_by="owner", created_at=time.time(), is_active=True,
        ))
        with pytest.raises(PolicyDenied, match="do not own"):
            await enforce_policy("attacker", "private-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_user_cannot_toggle_other_users_policy(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="toggle-agent", agent_name="Bot", created_by="owner",
        ))
        result = await toggle_agent_policy("toggle-agent", "attacker")
        assert result is None

    @pytest.mark.asyncio
    async def test_user_cannot_delete_other_users_policy(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="del-agent", agent_name="Bot", created_by="owner",
        ))
        result = await delete_agent_policy("del-agent", "attacker")
        assert result is False
        # Still exists
        p = await get_agent_policy("del-agent")
        assert p is not None

    @pytest.mark.asyncio
    async def test_same_agent_id_different_owners(self, db):
        """Two users can't have the same agent_id (it's a PK)."""
        await create_agent_policy(AgentPolicy(
            agent_id="shared-id", agent_name="Bot v1", created_by="user1",
        ))
        # INSERT OR REPLACE will overwrite
        await create_agent_policy(AgentPolicy(
            agent_id="shared-id", agent_name="Bot v2", created_by="user2",
        ))
        p = await get_agent_policy("shared-id")
        assert p.created_by == "user2"  # last writer wins


class TestApiKeyIsolation:
    """API keys are isolated between users."""

    @pytest.mark.asyncio
    async def test_users_see_only_own_keys(self, db):
        await create_api_key("u1", "a1", name="k1")
        await create_api_key("u1", "a2", name="k2")
        await create_api_key("u2", "a3", name="k3")
        k1 = await get_api_keys("u1")
        k2 = await get_api_keys("u2")
        assert len(k1) == 2
        assert len(k2) == 1

    @pytest.mark.asyncio
    async def test_user_cannot_revoke_other_users_key(self, db):
        key_obj, raw = await create_api_key("u1", "a1")
        result = await revoke_api_key(key_obj.id, "u2")
        assert result is False
        # Key still valid
        v = await validate_api_key(raw)
        assert v is not None

    @pytest.mark.asyncio
    async def test_key_validation_returns_owner_info(self, db):
        _, raw = await create_api_key("user-abc", "agent-xyz")
        result = await validate_api_key(raw)
        assert result.user_id == "user-abc"
        assert result.agent_id == "agent-xyz"


class TestAuditLogIsolation:
    """Audit logs are isolated between users."""

    @pytest.mark.asyncio
    async def test_users_see_only_own_audit_entries(self, db):
        await log_audit("u1", "a1", "github", "request", "success")
        await log_audit("u1", "a1", "slack", "request", "denied")
        await log_audit("u2", "a2", "github", "request", "success")
        e1 = await get_audit_log("u1")
        e2 = await get_audit_log("u2")
        assert len(e1) == 2
        assert len(e2) == 1

    @pytest.mark.asyncio
    async def test_audit_entries_contain_correct_user_id(self, db):
        await log_audit("specific-user", "a1", "github", "action", "status")
        entries = await get_audit_log("specific-user")
        assert entries[0].user_id == "specific-user"

    @pytest.mark.asyncio
    async def test_no_entries_for_new_user(self, db):
        entries = await get_audit_log("brand-new-user")
        assert entries == []


class TestConnectedServiceIsolation:
    """Connected services are isolated between users."""

    @pytest.mark.asyncio
    async def test_users_see_only_own_services(self, db):
        await add_connected_service("u1", "github")
        await add_connected_service("u1", "slack")
        await add_connected_service("u2", "google")
        s1 = await get_connected_services("u1")
        s2 = await get_connected_services("u2")
        assert len(s1) == 2
        assert len(s2) == 1

    @pytest.mark.asyncio
    async def test_removing_service_doesnt_affect_other_users(self, db):
        await add_connected_service("u1", "github")
        await add_connected_service("u2", "github")
        await remove_connected_service("u1", "github")
        s1 = await get_connected_services("u1")
        s2 = await get_connected_services("u2")
        assert len(s1) == 0
        assert len(s2) == 1


class TestEmergencyRevokeIsolation:
    """Emergency revoke only affects the target user."""

    @pytest.mark.asyncio
    async def test_emergency_revoke_isolates_to_user(self, db):
        # Setup: two users with policies and keys
        await create_agent_policy(AgentPolicy(
            agent_id="u1-agent", agent_name="U1 Bot", created_by="u1",
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="u2-agent", agent_name="U2 Bot", created_by="u2",
        ))
        _, raw1 = await create_api_key("u1", "u1-agent")
        _, raw2 = await create_api_key("u2", "u2-agent")

        # Emergency revoke u1 only
        result = await emergency_revoke_all("u1")
        assert result["policies_disabled"] == 1
        assert result["keys_revoked"] == 1

        # u1's resources are revoked
        p1 = await get_agent_policy("u1-agent")
        assert p1.is_active is False
        assert await validate_api_key(raw1) is None

        # u2's resources are unaffected
        p2 = await get_agent_policy("u2-agent")
        assert p2.is_active is True
        assert await validate_api_key(raw2) is not None

    @pytest.mark.asyncio
    async def test_emergency_revoke_on_user_with_no_resources(self, db):
        result = await emergency_revoke_all("empty-user")
        assert result["policies_disabled"] == 0
        assert result["keys_revoked"] == 0


class TestCrossTenantPolicyEnforcement:
    """Enforce policy respects tenant boundaries."""

    @pytest.mark.asyncio
    async def test_ten_tenants_isolated(self, db):
        """Create 10 tenants, each with a policy, and verify isolation."""
        for i in range(10):
            await create_agent_policy(AgentPolicy(
                agent_id=f"tenant{i}-agent", agent_name=f"Bot {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=60,
                created_by=f"tenant{i}", created_at=time.time(), is_active=True,
            ))

        # Each tenant can access their own agent
        for i in range(10):
            result = await enforce_policy(f"tenant{i}", f"tenant{i}-agent", "github", ["repo"])
            assert result.agent_id == f"tenant{i}-agent"

        # Cross-tenant access is denied
        with pytest.raises(PolicyDenied, match="do not own"):
            await enforce_policy("tenant0", "tenant1-agent", "github", ["repo"])
