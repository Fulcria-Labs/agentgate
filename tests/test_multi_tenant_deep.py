"""Deep multi-tenant isolation testing -- ensure tenant A cannot access,
modify, view, or interfere with tenant B's policies, keys, services,
and audit trail across all database operations."""

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


TENANT_A = "auth0|tenant-alpha"
TENANT_B = "auth0|tenant-beta"
TENANT_C = "auth0|tenant-gamma"


@pytest.fixture(autouse=True)
def clear_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


# ---------------------------------------------------------------------------
# 1. Policy ownership isolation
# ---------------------------------------------------------------------------

class TestPolicyOwnershipIsolation:
    """Policies created by one tenant are invisible and inaccessible to another."""

    @pytest.mark.asyncio
    async def test_enforce_denies_cross_tenant_access(self, db):
        """Tenant B cannot use enforce_policy on Tenant A's agent."""
        await create_agent_policy(AgentPolicy(
            agent_id="alpha-bot",
            agent_name="Alpha Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by=TENANT_A,
            created_at=time.time(),
            is_active=True,
        ))
        with pytest.raises(PolicyDenied, match="do not own"):
            await enforce_policy(TENANT_B, "alpha-bot", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_get_all_policies_is_tenant_scoped(self, db):
        """get_all_policies only returns policies owned by the requesting tenant."""
        for i in range(3):
            await create_agent_policy(AgentPolicy(
                agent_id=f"a-{i}", agent_name=f"A Bot {i}",
                created_by=TENANT_A, created_at=time.time(),
            ))
        for i in range(2):
            await create_agent_policy(AgentPolicy(
                agent_id=f"b-{i}", agent_name=f"B Bot {i}",
                created_by=TENANT_B, created_at=time.time(),
            ))
        policies_a = await get_all_policies(TENANT_A)
        policies_b = await get_all_policies(TENANT_B)
        policies_c = await get_all_policies(TENANT_C)
        assert len(policies_a) == 3
        assert len(policies_b) == 2
        assert len(policies_c) == 0

    @pytest.mark.asyncio
    async def test_toggle_fails_cross_tenant(self, db):
        """Tenant B cannot toggle Tenant A's policy."""
        await create_agent_policy(AgentPolicy(
            agent_id="a-toggle", agent_name="Bot",
            created_by=TENANT_A, created_at=time.time(), is_active=True,
        ))
        result = await toggle_agent_policy("a-toggle", TENANT_B)
        assert result is None
        # A's policy unchanged
        p = await get_agent_policy("a-toggle")
        assert p.is_active is True

    @pytest.mark.asyncio
    async def test_delete_fails_cross_tenant(self, db):
        """Tenant B cannot delete Tenant A's policy."""
        await create_agent_policy(AgentPolicy(
            agent_id="a-delete", agent_name="Bot",
            created_by=TENANT_A, created_at=time.time(),
        ))
        deleted = await delete_agent_policy("a-delete", TENANT_B)
        assert deleted is False
        assert await get_agent_policy("a-delete") is not None


# ---------------------------------------------------------------------------
# 2. API key tenant isolation
# ---------------------------------------------------------------------------

class TestApiKeyTenantIsolation:
    """API keys created by one tenant cannot be listed or revoked by another."""

    @pytest.mark.asyncio
    async def test_get_api_keys_scoped_to_tenant(self, db):
        """Each tenant only sees their own API keys."""
        await create_api_key(TENANT_A, "agent-a", "key-a")
        await create_api_key(TENANT_A, "agent-a2", "key-a2")
        await create_api_key(TENANT_B, "agent-b", "key-b")

        keys_a = await get_api_keys(TENANT_A)
        keys_b = await get_api_keys(TENANT_B)
        keys_c = await get_api_keys(TENANT_C)

        assert len(keys_a) == 2
        assert len(keys_b) == 1
        assert len(keys_c) == 0

    @pytest.mark.asyncio
    async def test_revoke_fails_cross_tenant(self, db):
        """Tenant B cannot revoke Tenant A's API key."""
        key, raw = await create_api_key(TENANT_A, "agent-a", "key-a")
        revoked = await revoke_api_key(key.id, TENANT_B)
        assert revoked is False
        # Key still valid
        assert await validate_api_key(raw) is not None

    @pytest.mark.asyncio
    async def test_api_key_validates_regardless_of_tenant(self, db):
        """Any bearer of the raw key can validate it (binding is via user_id)."""
        _, raw = await create_api_key(TENANT_A, "agent-a", "key-a")
        result = await validate_api_key(raw)
        assert result is not None
        assert result.user_id == TENANT_A

    @pytest.mark.asyncio
    async def test_api_key_agent_id_bound_to_owner(self, db):
        """An API key's agent_id is bound to the creating tenant."""
        _, raw = await create_api_key(TENANT_A, "shared-agent-id", "k")
        result = await validate_api_key(raw)
        assert result.user_id == TENANT_A
        assert result.agent_id == "shared-agent-id"


# ---------------------------------------------------------------------------
# 3. Audit log tenant isolation
# ---------------------------------------------------------------------------

class TestAuditLogTenantIsolation:
    """Audit entries are strictly scoped to the creating user."""

    @pytest.mark.asyncio
    async def test_audit_logs_isolated(self, db):
        """Each tenant only sees their own audit entries."""
        for i in range(5):
            await log_audit(TENANT_A, f"agent-{i}", "github", "action", "success")
        for i in range(3):
            await log_audit(TENANT_B, f"agent-{i}", "slack", "action", "denied")

        entries_a = await get_audit_log(TENANT_A)
        entries_b = await get_audit_log(TENANT_B)
        entries_c = await get_audit_log(TENANT_C)

        assert len(entries_a) == 5
        assert len(entries_b) == 3
        assert len(entries_c) == 0
        assert all(e.user_id == TENANT_A for e in entries_a)
        assert all(e.user_id == TENANT_B for e in entries_b)

    @pytest.mark.asyncio
    async def test_enforcement_denial_audit_uses_caller_tenant(self, db):
        """When tenant B is denied access to A's policy, the audit entry is under B."""
        await create_agent_policy(AgentPolicy(
            agent_id="cross-audit", agent_name="Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            created_by=TENANT_A, created_at=time.time(), is_active=True,
        ))
        with pytest.raises(PolicyDenied):
            await enforce_policy(TENANT_B, "cross-audit", "github", ["repo"])
        entries_b = await get_audit_log(TENANT_B)
        assert len(entries_b) >= 1
        assert entries_b[0].status == "denied"
        # Tenant A should have no audit entries from this
        entries_a = await get_audit_log(TENANT_A)
        assert len(entries_a) == 0


# ---------------------------------------------------------------------------
# 4. Connected services tenant isolation
# ---------------------------------------------------------------------------

class TestConnectedServicesTenantIsolation:
    """Connected services are completely isolated between tenants."""

    @pytest.mark.asyncio
    async def test_services_isolated(self, db):
        """Each tenant's connected services are independent."""
        await add_connected_service(TENANT_A, "github", "conn-a")
        await add_connected_service(TENANT_A, "slack", "conn-a2")
        await add_connected_service(TENANT_B, "google", "conn-b")

        svcs_a = await get_connected_services(TENANT_A)
        svcs_b = await get_connected_services(TENANT_B)
        svcs_c = await get_connected_services(TENANT_C)

        assert len(svcs_a) == 2
        assert len(svcs_b) == 1
        assert len(svcs_c) == 0

    @pytest.mark.asyncio
    async def test_remove_only_affects_own_service(self, db):
        """Removing a service for tenant A doesn't affect tenant B."""
        await add_connected_service(TENANT_A, "github")
        await add_connected_service(TENANT_B, "github")
        await remove_connected_service(TENANT_A, "github")

        svcs_a = await get_connected_services(TENANT_A)
        svcs_b = await get_connected_services(TENANT_B)
        assert len(svcs_a) == 0
        assert len(svcs_b) == 1


# ---------------------------------------------------------------------------
# 5. Emergency revoke tenant isolation
# ---------------------------------------------------------------------------

class TestEmergencyRevokeTenantIsolation:
    """Emergency revoke for one tenant must not affect others."""

    @pytest.mark.asyncio
    async def test_emergency_revoke_only_affects_caller(self, db):
        """Emergency revoke only disables the calling tenant's resources."""
        # Tenant A resources
        await create_agent_policy(AgentPolicy(
            agent_id="a-em", agent_name="A Bot",
            created_by=TENANT_A, created_at=time.time(), is_active=True,
        ))
        _, raw_a = await create_api_key(TENANT_A, "a-em", "key-a")

        # Tenant B resources
        await create_agent_policy(AgentPolicy(
            agent_id="b-em", agent_name="B Bot",
            created_by=TENANT_B, created_at=time.time(), is_active=True,
        ))
        _, raw_b = await create_api_key(TENANT_B, "b-em", "key-b")

        # Tenant A triggers emergency revoke
        result = await emergency_revoke_all(TENANT_A)
        assert result["policies_disabled"] == 1
        assert result["keys_revoked"] == 1

        # Tenant A's resources are revoked
        p_a = await get_agent_policy("a-em")
        assert p_a.is_active is False
        assert await validate_api_key(raw_a) is None

        # Tenant B's resources are untouched
        p_b = await get_agent_policy("b-em")
        assert p_b.is_active is True
        assert await validate_api_key(raw_b) is not None

    @pytest.mark.asyncio
    async def test_emergency_revoke_connected_services_preserved(self, db):
        """Emergency revoke does not disconnect services for any tenant."""
        await add_connected_service(TENANT_A, "github", "conn-a")
        await add_connected_service(TENANT_B, "slack", "conn-b")

        await emergency_revoke_all(TENANT_A)

        svcs_a = await get_connected_services(TENANT_A)
        svcs_b = await get_connected_services(TENANT_B)
        assert len(svcs_a) == 1  # Preserved
        assert len(svcs_b) == 1


# ---------------------------------------------------------------------------
# 6. Three-tenant enforcement scenarios
# ---------------------------------------------------------------------------

class TestThreeTenantEnforcement:
    """Enforcement scenarios with three distinct tenants."""

    @pytest.mark.asyncio
    async def test_same_agent_id_different_tenants(self, db):
        """Two tenants can have policies with the same agent_id
        (agent_id is PRIMARY KEY, so only one exists -- tests ownership)."""
        await create_agent_policy(AgentPolicy(
            agent_id="shared-id", agent_name="A Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            created_by=TENANT_A, created_at=time.time(), is_active=True,
        ))
        # B tries to use A's agent
        with pytest.raises(PolicyDenied, match="do not own"):
            await enforce_policy(TENANT_B, "shared-id", "github", ["repo"])
        # A can use it
        result = await enforce_policy(TENANT_A, "shared-id", "github", ["repo"])
        assert result.agent_id == "shared-id"

    @pytest.mark.asyncio
    async def test_three_tenants_independent_rate_limits(self, db):
        """Rate limits are per agent:service, not per tenant, but ownership
        prevents cross-tenant access anyway."""
        for tenant, agent_id in [
            (TENANT_A, "rl-a"), (TENANT_B, "rl-b"), (TENANT_C, "rl-c"),
        ]:
            await create_agent_policy(AgentPolicy(
                agent_id=agent_id, agent_name=f"Bot {agent_id}",
                allowed_services=["github"], allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=1,
                created_by=tenant, created_at=time.time(), is_active=True,
            ))
        # Each tenant uses their own agent
        await enforce_policy(TENANT_A, "rl-a", "github", ["repo"])
        await enforce_policy(TENANT_B, "rl-b", "github", ["repo"])
        await enforce_policy(TENANT_C, "rl-c", "github", ["repo"])
        # Each is now rate limited independently
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy(TENANT_A, "rl-a", "github", ["repo"])
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy(TENANT_B, "rl-b", "github", ["repo"])
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy(TENANT_C, "rl-c", "github", ["repo"])
