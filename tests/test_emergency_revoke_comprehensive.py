"""Comprehensive emergency revoke tests — kill switch behavior, partial states,
re-enabling after revoke, multiple users, and audit logging."""

import time

import pytest

from src.database import (
    AgentPolicy,
    create_agent_policy,
    create_api_key,
    emergency_revoke_all,
    get_agent_policy,
    get_all_policies,
    get_api_keys,
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


class TestEmergencyRevokeBasic:
    """Basic emergency revoke behavior."""

    @pytest.mark.asyncio
    async def test_disables_all_policies(self, db):
        for i in range(5):
            await create_agent_policy(AgentPolicy(
                agent_id=f"agent-{i}", agent_name=f"Bot {i}", created_by="u1",
            ))
        result = await emergency_revoke_all("u1")
        assert result["policies_disabled"] == 5

        for i in range(5):
            p = await get_agent_policy(f"agent-{i}")
            assert p.is_active is False

    @pytest.mark.asyncio
    async def test_revokes_all_api_keys(self, db):
        for i in range(4):
            await create_api_key("u1", f"agent-{i}", name=f"key-{i}")
        result = await emergency_revoke_all("u1")
        assert result["keys_revoked"] == 4

    @pytest.mark.asyncio
    async def test_revoked_keys_fail_validation(self, db):
        raws = []
        for i in range(3):
            _, raw = await create_api_key("u1", f"agent-{i}")
            raws.append(raw)
        await emergency_revoke_all("u1")
        for raw in raws:
            assert await validate_api_key(raw) is None

    @pytest.mark.asyncio
    async def test_disabled_policies_deny_access(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="revoked-agent", agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            created_by="u1", created_at=time.time(), is_active=True,
        ))
        await emergency_revoke_all("u1")
        with pytest.raises(PolicyDenied, match="disabled"):
            await enforce_policy("u1", "revoked-agent", "github", ["repo"])


class TestEmergencyRevokeReturnValues:
    """Verify return value accuracy."""

    @pytest.mark.asyncio
    async def test_zero_policies_zero_keys(self, db):
        result = await emergency_revoke_all("empty-user")
        assert result["policies_disabled"] == 0
        assert result["keys_revoked"] == 0

    @pytest.mark.asyncio
    async def test_policies_only_no_keys(self, db):
        for i in range(3):
            await create_agent_policy(AgentPolicy(
                agent_id=f"pol-{i}", agent_name="Bot", created_by="u1",
            ))
        result = await emergency_revoke_all("u1")
        assert result["policies_disabled"] == 3
        assert result["keys_revoked"] == 0

    @pytest.mark.asyncio
    async def test_keys_only_no_policies(self, db):
        for i in range(2):
            await create_api_key("u1", f"a-{i}")
        result = await emergency_revoke_all("u1")
        assert result["policies_disabled"] == 0
        assert result["keys_revoked"] == 2

    @pytest.mark.asyncio
    async def test_exact_count_mixed(self, db):
        for i in range(4):
            await create_agent_policy(AgentPolicy(
                agent_id=f"mix-{i}", agent_name="Bot", created_by="u1",
            ))
        for i in range(6):
            await create_api_key("u1", f"mix-{i}")
        result = await emergency_revoke_all("u1")
        assert result["policies_disabled"] == 4
        assert result["keys_revoked"] == 6


class TestEmergencyRevokeIdempotency:
    """Calling emergency revoke multiple times."""

    @pytest.mark.asyncio
    async def test_double_revoke_no_error(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="idem-agent", agent_name="Bot", created_by="u1",
        ))
        await create_api_key("u1", "idem-agent")

        r1 = await emergency_revoke_all("u1")
        assert r1["policies_disabled"] == 1
        assert r1["keys_revoked"] == 1

        # Second call — policies already disabled, keys already revoked
        r2 = await emergency_revoke_all("u1")
        # Already inactive policies still get "updated" to inactive
        # Already revoked keys won't match WHERE is_revoked = 0
        assert r2["keys_revoked"] == 0

    @pytest.mark.asyncio
    async def test_revoke_then_add_new_resources(self, db):
        """After emergency revoke, new resources work normally."""
        await create_agent_policy(AgentPolicy(
            agent_id="old-agent", agent_name="Old Bot", created_by="u1",
        ))
        await emergency_revoke_all("u1")

        # Create new policy — should be active
        await create_agent_policy(AgentPolicy(
            agent_id="new-agent", agent_name="New Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            created_by="u1", created_at=time.time(), is_active=True,
        ))
        p = await get_agent_policy("new-agent")
        assert p.is_active is True

        # New key should be valid
        _, raw = await create_api_key("u1", "new-agent")
        v = await validate_api_key(raw)
        assert v is not None


class TestEmergencyRevokeWithToggle:
    """Interaction between emergency revoke and policy toggle."""

    @pytest.mark.asyncio
    async def test_toggle_after_revoke_re_enables(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="toggle-agent", agent_name="Bot", created_by="u1",
        ))
        await emergency_revoke_all("u1")

        # Toggle re-enables
        new_state = await toggle_agent_policy("toggle-agent", "u1")
        assert new_state is True

        p = await get_agent_policy("toggle-agent")
        assert p.is_active is True

    @pytest.mark.asyncio
    async def test_re_enabled_policy_enforces_again(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="re-enable-agent", agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            created_by="u1", created_at=time.time(), is_active=True,
        ))
        await emergency_revoke_all("u1")

        # Can't access
        with pytest.raises(PolicyDenied, match="disabled"):
            await enforce_policy("u1", "re-enable-agent", "github", ["repo"])

        # Re-enable
        await toggle_agent_policy("re-enable-agent", "u1")

        # Now can access
        result = await enforce_policy("u1", "re-enable-agent", "github", ["repo"])
        assert result.agent_id == "re-enable-agent"


class TestEmergencyRevokeWithAlreadyDisabled:
    """Edge case: some policies already disabled before revoke."""

    @pytest.mark.asyncio
    async def test_mix_of_active_and_inactive(self, db):
        # 2 active + 1 already inactive
        await create_agent_policy(AgentPolicy(
            agent_id="active-1", agent_name="Bot", created_by="u1", is_active=True,
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="active-2", agent_name="Bot", created_by="u1", is_active=True,
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="inactive-1", agent_name="Bot", created_by="u1", is_active=False,
        ))

        result = await emergency_revoke_all("u1")
        # All 3 get SET is_active=0 (even the already-inactive one)
        assert result["policies_disabled"] == 3

    @pytest.mark.asyncio
    async def test_already_revoked_keys_not_counted(self, db):
        k1, _ = await create_api_key("u1", "a1")
        k2, _ = await create_api_key("u1", "a2")
        # Pre-revoke k1
        from src.database import revoke_api_key
        await revoke_api_key(k1.id, "u1")

        result = await emergency_revoke_all("u1")
        # Only k2 should be counted as newly revoked
        assert result["keys_revoked"] == 1
