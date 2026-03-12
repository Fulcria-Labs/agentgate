"""Deep emergency kill switch testing -- comprehensive scenarios for the
emergency_revoke_all functionality including partial states, idempotency,
interaction with already-revoked resources, and recovery patterns."""

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
    get_connected_services,
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
def clear_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


USER = "auth0|kill-switch-user"


# ---------------------------------------------------------------------------
# 1. Basic kill switch behaviour
# ---------------------------------------------------------------------------

class TestKillSwitchBasics:
    """Fundamental kill switch operations."""

    @pytest.mark.asyncio
    async def test_empty_user_returns_zero_counts(self, db):
        """Kill switch on user with no resources returns zeroes."""
        result = await emergency_revoke_all(USER)
        assert result["policies_disabled"] == 0
        assert result["keys_revoked"] == 0

    @pytest.mark.asyncio
    async def test_single_policy_disabled(self, db):
        """A single active policy is disabled."""
        await create_agent_policy(AgentPolicy(
            agent_id="bot-1", agent_name="Bot",
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        result = await emergency_revoke_all(USER)
        assert result["policies_disabled"] == 1
        p = await get_agent_policy("bot-1")
        assert p.is_active is False

    @pytest.mark.asyncio
    async def test_single_key_revoked(self, db):
        """A single active key is revoked."""
        _, raw = await create_api_key(USER, "bot-1", "key-1")
        result = await emergency_revoke_all(USER)
        assert result["keys_revoked"] == 1
        assert await validate_api_key(raw) is None

    @pytest.mark.asyncio
    async def test_multiple_policies_all_disabled(self, db):
        """All active policies for a user are disabled."""
        for i in range(5):
            await create_agent_policy(AgentPolicy(
                agent_id=f"bot-{i}", agent_name=f"Bot {i}",
                created_by=USER, created_at=time.time(), is_active=True,
            ))
        result = await emergency_revoke_all(USER)
        assert result["policies_disabled"] == 5
        for i in range(5):
            p = await get_agent_policy(f"bot-{i}")
            assert p.is_active is False

    @pytest.mark.asyncio
    async def test_multiple_keys_all_revoked(self, db):
        """All active keys for a user are revoked."""
        raws = []
        for i in range(5):
            _, raw = await create_api_key(USER, f"bot-{i}", f"key-{i}")
            raws.append(raw)
        result = await emergency_revoke_all(USER)
        assert result["keys_revoked"] == 5
        for raw in raws:
            assert await validate_api_key(raw) is None


# ---------------------------------------------------------------------------
# 2. Pre-existing disabled/revoked resources
# ---------------------------------------------------------------------------

class TestKillSwitchPreExisting:
    """Kill switch with already-disabled policies and revoked keys."""

    @pytest.mark.asyncio
    async def test_already_inactive_policy_counted(self, db):
        """Policies already inactive are still counted (UPDATE SET hits them)."""
        await create_agent_policy(AgentPolicy(
            agent_id="off-bot", agent_name="Off Bot",
            created_by=USER, created_at=time.time(), is_active=False,
        ))
        result = await emergency_revoke_all(USER)
        # UPDATE sets is_active=0 for all, already-inactive still in rowcount
        assert result["policies_disabled"] >= 1

    @pytest.mark.asyncio
    async def test_already_revoked_key_not_double_counted(self, db):
        """Already revoked keys are NOT counted again."""
        key, raw = await create_api_key(USER, "bot-1", "key-1")
        await revoke_api_key(key.id, USER)
        result = await emergency_revoke_all(USER)
        # The SQL filters is_revoked=0, so already-revoked skipped
        assert result["keys_revoked"] == 0

    @pytest.mark.asyncio
    async def test_mix_active_and_revoked_keys(self, db):
        """Only active keys get revoked in the count."""
        k1, r1 = await create_api_key(USER, "bot-1", "key-1")
        _, r2 = await create_api_key(USER, "bot-2", "key-2")
        _, r3 = await create_api_key(USER, "bot-3", "key-3")
        await revoke_api_key(k1.id, USER)
        result = await emergency_revoke_all(USER)
        assert result["keys_revoked"] == 2  # Only key-2 and key-3

    @pytest.mark.asyncio
    async def test_mix_active_and_inactive_policies(self, db):
        """Active and inactive policies both affected by kill switch."""
        await create_agent_policy(AgentPolicy(
            agent_id="active-bot", agent_name="Active",
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="inactive-bot", agent_name="Inactive",
            created_by=USER, created_at=time.time(), is_active=False,
        ))
        result = await emergency_revoke_all(USER)
        # Both get SET is_active=0 (rowcount includes both)
        assert result["policies_disabled"] == 2


# ---------------------------------------------------------------------------
# 3. Idempotency
# ---------------------------------------------------------------------------

class TestKillSwitchIdempotency:
    """Running the kill switch multiple times is safe."""

    @pytest.mark.asyncio
    async def test_double_kill_switch(self, db):
        """Running kill switch twice: second time yields zero new revocations."""
        await create_agent_policy(AgentPolicy(
            agent_id="bot-1", agent_name="Bot",
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        _, raw = await create_api_key(USER, "bot-1", "key-1")

        r1 = await emergency_revoke_all(USER)
        assert r1["policies_disabled"] == 1
        assert r1["keys_revoked"] == 1

        r2 = await emergency_revoke_all(USER)
        # Policies still get SET is_active=0 (already 0, rowcount still 1)
        # Keys with is_revoked=0 filter means 0
        assert r2["keys_revoked"] == 0

    @pytest.mark.asyncio
    async def test_triple_kill_switch_keys_stay_invalid(self, db):
        """Keys remain invalid after multiple kill switch invocations."""
        _, raw = await create_api_key(USER, "bot-1", "key-1")
        for _ in range(3):
            await emergency_revoke_all(USER)
        assert await validate_api_key(raw) is None


# ---------------------------------------------------------------------------
# 4. Kill switch and enforcement
# ---------------------------------------------------------------------------

class TestKillSwitchEnforcement:
    """Kill switch prevents subsequent enforce_policy calls."""

    @pytest.mark.asyncio
    async def test_enforce_denied_after_kill_switch(self, db):
        """enforce_policy raises PolicyDenied after kill switch disables policy."""
        await create_agent_policy(AgentPolicy(
            agent_id="enf-bot", agent_name="Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        # Works before
        result = await enforce_policy(USER, "enf-bot", "github", ["repo"])
        assert result.agent_id == "enf-bot"

        await emergency_revoke_all(USER)

        # Denied after
        with pytest.raises(PolicyDenied, match="disabled"):
            await enforce_policy(USER, "enf-bot", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_api_key_invalid_after_kill_switch_then_enforce(self, db):
        """API key from before kill switch is invalid after."""
        await create_agent_policy(AgentPolicy(
            agent_id="key-enf", agent_name="Bot",
            allowed_services=["slack"], allowed_scopes={"slack": ["chat:write"]},
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        _, raw = await create_api_key(USER, "key-enf", "k")
        assert await validate_api_key(raw) is not None

        await emergency_revoke_all(USER)
        assert await validate_api_key(raw) is None


# ---------------------------------------------------------------------------
# 5. Kill switch does NOT affect connected services
# ---------------------------------------------------------------------------

class TestKillSwitchConnectedServices:
    """Verify that kill switch intentionally preserves connected services."""

    @pytest.mark.asyncio
    async def test_connected_services_preserved(self, db):
        """Connected services remain after kill switch."""
        await add_connected_service(USER, "github", "gh-conn")
        await add_connected_service(USER, "slack", "sl-conn")
        await emergency_revoke_all(USER)
        svcs = await get_connected_services(USER)
        assert len(svcs) == 2

    @pytest.mark.asyncio
    async def test_service_connection_after_kill_switch(self, db):
        """New services can be connected after kill switch."""
        await emergency_revoke_all(USER)
        await add_connected_service(USER, "notion", "notion-conn")
        svcs = await get_connected_services(USER)
        assert len(svcs) == 1
        assert svcs[0]["service"] == "notion"


# ---------------------------------------------------------------------------
# 6. Recovery after kill switch
# ---------------------------------------------------------------------------

class TestKillSwitchRecovery:
    """Verify that resources can be re-enabled after a kill switch."""

    @pytest.mark.asyncio
    async def test_policy_re_enable_after_kill_switch(self, db):
        """A disabled policy can be toggled back on."""
        await create_agent_policy(AgentPolicy(
            agent_id="recovery-bot", agent_name="Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        await emergency_revoke_all(USER)
        p = await get_agent_policy("recovery-bot")
        assert p.is_active is False

        new_state = await toggle_agent_policy("recovery-bot", USER)
        assert new_state is True
        result = await enforce_policy(USER, "recovery-bot", "github", ["repo"])
        assert result.agent_id == "recovery-bot"

    @pytest.mark.asyncio
    async def test_new_key_works_after_kill_switch(self, db):
        """New API keys created after kill switch are valid."""
        _, old_raw = await create_api_key(USER, "bot-1", "old-key")
        await emergency_revoke_all(USER)
        assert await validate_api_key(old_raw) is None

        _, new_raw = await create_api_key(USER, "bot-1", "new-key")
        result = await validate_api_key(new_raw)
        assert result is not None
        assert result.agent_id == "bot-1"

    @pytest.mark.asyncio
    async def test_new_policy_works_after_kill_switch(self, db):
        """New policies created after kill switch function normally."""
        await emergency_revoke_all(USER)
        await create_agent_policy(AgentPolicy(
            agent_id="new-bot", agent_name="New Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        result = await enforce_policy(USER, "new-bot", "github", ["repo"])
        assert result.agent_id == "new-bot"

    @pytest.mark.asyncio
    async def test_delete_policy_after_kill_switch(self, db):
        """Policies can be fully deleted after kill switch."""
        await create_agent_policy(AgentPolicy(
            agent_id="del-bot", agent_name="Bot",
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        await emergency_revoke_all(USER)
        deleted = await delete_agent_policy("del-bot", USER)
        assert deleted is True
        assert await get_agent_policy("del-bot") is None


# ---------------------------------------------------------------------------
# 7. Kill switch with large resource sets
# ---------------------------------------------------------------------------

class TestKillSwitchScale:
    """Kill switch with many resources."""

    @pytest.mark.asyncio
    async def test_20_policies_all_disabled(self, db):
        """Kill switch disables 20 policies."""
        for i in range(20):
            await create_agent_policy(AgentPolicy(
                agent_id=f"scale-bot-{i}", agent_name=f"Bot {i}",
                created_by=USER, created_at=time.time(), is_active=True,
            ))
        result = await emergency_revoke_all(USER)
        assert result["policies_disabled"] == 20
        policies = await get_all_policies(USER)
        assert all(not p.is_active for p in policies)

    @pytest.mark.asyncio
    async def test_20_keys_all_revoked(self, db):
        """Kill switch revokes 20 API keys."""
        raws = []
        for i in range(20):
            _, raw = await create_api_key(USER, f"bot-{i}", f"key-{i}")
            raws.append(raw)
        result = await emergency_revoke_all(USER)
        assert result["keys_revoked"] == 20
        for raw in raws:
            assert await validate_api_key(raw) is None

    @pytest.mark.asyncio
    async def test_mixed_scale_resources(self, db):
        """Kill switch on mixed 10 policies + 10 keys + 5 services."""
        for i in range(10):
            await create_agent_policy(AgentPolicy(
                agent_id=f"mix-bot-{i}", agent_name=f"Bot {i}",
                created_by=USER, created_at=time.time(), is_active=True,
            ))
        raws = []
        for i in range(10):
            _, raw = await create_api_key(USER, f"mix-bot-{i}", f"key-{i}")
            raws.append(raw)
        for svc in ["github", "slack", "google", "linear", "notion"]:
            await add_connected_service(USER, svc, f"conn-{svc}")

        result = await emergency_revoke_all(USER)
        assert result["policies_disabled"] == 10
        assert result["keys_revoked"] == 10
        # Services preserved
        svcs = await get_connected_services(USER)
        assert len(svcs) == 5


# ---------------------------------------------------------------------------
# 8. Kill switch with expired keys
# ---------------------------------------------------------------------------

class TestKillSwitchExpiredKeys:
    """Kill switch interaction with expired API keys."""

    @pytest.mark.asyncio
    async def test_expired_key_still_revoked_by_kill_switch(self, db):
        """An expired but not-revoked key is still revoked by kill switch."""
        _, raw = await create_api_key(USER, "bot-1", "exp-key", expires_in=-1)
        # Already expired, but not revoked
        result = await emergency_revoke_all(USER)
        assert result["keys_revoked"] == 1

    @pytest.mark.asyncio
    async def test_far_future_key_revoked_by_kill_switch(self, db):
        """A key with far-future expiry is revoked by kill switch."""
        _, raw = await create_api_key(USER, "bot-1", "far-key", expires_in=365*86400)
        result = await emergency_revoke_all(USER)
        assert result["keys_revoked"] == 1
        assert await validate_api_key(raw) is None

    @pytest.mark.asyncio
    async def test_never_expiring_key_revoked_by_kill_switch(self, db):
        """A never-expiring key is revoked by kill switch."""
        _, raw = await create_api_key(USER, "bot-1", "forever-key", expires_in=0)
        result = await emergency_revoke_all(USER)
        assert result["keys_revoked"] == 1
        assert await validate_api_key(raw) is None
