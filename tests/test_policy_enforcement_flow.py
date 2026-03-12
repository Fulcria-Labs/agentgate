"""Policy enforcement flow tests — full end-to-end enforcement covering
all 8 steps in order, complex multi-constraint scenarios, and
boundary conditions across enforcement stages."""

import time
from unittest.mock import patch

import pytest

from src.database import AgentPolicy, create_agent_policy, init_db
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


def _full_policy(**overrides):
    defaults = dict(
        agent_id="flow-agent",
        agent_name="Flow Bot",
        allowed_services=["github", "slack"],
        allowed_scopes={"github": ["repo", "read:user"], "slack": ["channels:read"]},
        rate_limit_per_minute=60,
        requires_step_up=[],
        created_by="user1",
        created_at=time.time(),
        is_active=True,
        allowed_hours=[],
        allowed_days=[],
        expires_at=0.0,
        ip_allowlist=[],
    )
    defaults.update(overrides)
    return AgentPolicy(**defaults)


class TestEnforcementStep1AgentExistence:
    """Step 1: Agent must exist."""

    @pytest.mark.asyncio
    async def test_nonexistent_agent_denied(self, db):
        with pytest.raises(PolicyDenied, match="not registered"):
            await enforce_policy("u1", "ghost-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_existent_agent_passes_step1(self, db):
        await create_agent_policy(_full_policy())
        result = await enforce_policy("user1", "flow-agent", "github", ["repo"])
        assert result is not None


class TestEnforcementStep1BAgentActive:
    """Step 1b: Agent must be active."""

    @pytest.mark.asyncio
    async def test_inactive_agent_denied(self, db):
        await create_agent_policy(_full_policy(is_active=False))
        with pytest.raises(PolicyDenied, match="disabled"):
            await enforce_policy("user1", "flow-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_active_agent_passes(self, db):
        await create_agent_policy(_full_policy(is_active=True))
        result = await enforce_policy("user1", "flow-agent", "github", ["repo"])
        assert result.is_active is True


class TestEnforcementStep2Ownership:
    """Step 2: User must own the policy."""

    @pytest.mark.asyncio
    async def test_non_owner_denied(self, db):
        await create_agent_policy(_full_policy(created_by="owner"))
        with pytest.raises(PolicyDenied, match="do not own"):
            await enforce_policy("intruder", "flow-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_owner_passes(self, db):
        await create_agent_policy(_full_policy(created_by="rightful-owner"))
        result = await enforce_policy("rightful-owner", "flow-agent", "github", ["repo"])
        assert result.created_by == "rightful-owner"


class TestEnforcementStep3Expiration:
    """Step 3: Policy must not be expired."""

    @pytest.mark.asyncio
    async def test_expired_policy_denied(self, db):
        await create_agent_policy(_full_policy(expires_at=time.time() - 1))
        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy("user1", "flow-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_future_expiry_passes(self, db):
        await create_agent_policy(_full_policy(expires_at=time.time() + 86400))
        result = await enforce_policy("user1", "flow-agent", "github", ["repo"])
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_expiry_passes(self, db):
        await create_agent_policy(_full_policy(expires_at=0.0))
        result = await enforce_policy("user1", "flow-agent", "github", ["repo"])
        assert result is not None

    @pytest.mark.asyncio
    async def test_just_expired_denied(self, db):
        """Policy that expired 1 second ago is denied."""
        await create_agent_policy(_full_policy(expires_at=time.time() - 1))
        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy("user1", "flow-agent", "github", ["repo"])


class TestEnforcementStep4TimeWindow:
    """Step 4: Time window check."""

    @pytest.mark.asyncio
    async def test_time_window_denied(self, db):
        await create_agent_policy(_full_policy(allowed_hours=[9]))
        with patch("src.policy.check_time_window", return_value="denied: bad time"):
            with pytest.raises(PolicyDenied, match="bad time"):
                await enforce_policy("user1", "flow-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_time_window_allowed(self, db):
        await create_agent_policy(_full_policy(allowed_hours=[9]))
        with patch("src.policy.check_time_window", return_value=None):
            result = await enforce_policy("user1", "flow-agent", "github", ["repo"])
            assert result is not None


class TestEnforcementStep5IPAllowlist:
    """Step 5: IP allowlist check."""

    @pytest.mark.asyncio
    async def test_ip_denied(self, db):
        await create_agent_policy(_full_policy(ip_allowlist=["10.0.0.1"]))
        with pytest.raises(PolicyDenied, match="not in allowlist"):
            await enforce_policy("user1", "flow-agent", "github", ["repo"], ip_address="192.168.1.1")

    @pytest.mark.asyncio
    async def test_ip_allowed(self, db):
        await create_agent_policy(_full_policy(ip_allowlist=["10.0.0.1"]))
        result = await enforce_policy(
            "user1", "flow-agent", "github", ["repo"], ip_address="10.0.0.1"
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_empty_allowlist_passes(self, db):
        await create_agent_policy(_full_policy(ip_allowlist=[]))
        result = await enforce_policy(
            "user1", "flow-agent", "github", ["repo"], ip_address="any.ip"
        )
        assert result is not None


class TestEnforcementStep6ServiceAuth:
    """Step 6: Service authorization."""

    @pytest.mark.asyncio
    async def test_unauthorized_service_denied(self, db):
        await create_agent_policy(_full_policy(allowed_services=["github"]))
        with pytest.raises(PolicyDenied, match="not authorized to access"):
            await enforce_policy("user1", "flow-agent", "google", ["gmail.readonly"])

    @pytest.mark.asyncio
    async def test_authorized_service_passes(self, db):
        await create_agent_policy(_full_policy(allowed_services=["github"]))
        result = await enforce_policy("user1", "flow-agent", "github", ["repo"])
        assert result is not None


class TestEnforcementStep7ScopeRestriction:
    """Step 7: Scope restrictions."""

    @pytest.mark.asyncio
    async def test_excess_scopes_denied(self, db):
        await create_agent_policy(_full_policy())
        with pytest.raises(PolicyDenied, match="Scopes not permitted"):
            await enforce_policy("user1", "flow-agent", "github", ["repo", "admin:org"])

    @pytest.mark.asyncio
    async def test_allowed_scopes_pass(self, db):
        await create_agent_policy(_full_policy())
        result = await enforce_policy("user1", "flow-agent", "github", ["repo", "read:user"])
        assert result is not None

    @pytest.mark.asyncio
    async def test_empty_scopes_pass(self, db):
        await create_agent_policy(_full_policy())
        result = await enforce_policy("user1", "flow-agent", "github", [])
        assert result is not None

    @pytest.mark.asyncio
    async def test_single_excess_scope_denied(self, db):
        await create_agent_policy(_full_policy())
        with pytest.raises(PolicyDenied, match="Scopes not permitted"):
            await enforce_policy("user1", "flow-agent", "github", ["delete_repo"])


class TestEnforcementStep8RateLimit:
    """Step 8: Rate limiting."""

    @pytest.mark.asyncio
    async def test_within_rate_limit(self, db):
        await create_agent_policy(_full_policy(rate_limit_per_minute=5))
        for _ in range(5):
            await enforce_policy("user1", "flow-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_over_rate_limit(self, db):
        await create_agent_policy(_full_policy(rate_limit_per_minute=2))
        for _ in range(2):
            await enforce_policy("user1", "flow-agent", "github", ["repo"])
        with pytest.raises(PolicyDenied, match="Rate limit exceeded"):
            await enforce_policy("user1", "flow-agent", "github", ["repo"])


class TestMultiConstraintScenarios:
    """Scenarios with multiple constraints active simultaneously."""

    @pytest.mark.asyncio
    async def test_all_constraints_pass(self, db):
        policy = _full_policy(
            ip_allowlist=["10.0.0.1"],
            rate_limit_per_minute=10,
            expires_at=time.time() + 3600,
        )
        await create_agent_policy(policy)
        with patch("src.policy.check_time_window", return_value=None):
            result = await enforce_policy(
                "user1", "flow-agent", "github", ["repo"], ip_address="10.0.0.1"
            )
            assert result.agent_id == "flow-agent"

    @pytest.mark.asyncio
    async def test_multiple_services_same_agent(self, db):
        """Agent can access different allowed services in sequence."""
        await create_agent_policy(_full_policy())
        r1 = await enforce_policy("user1", "flow-agent", "github", ["repo"])
        r2 = await enforce_policy("user1", "flow-agent", "slack", ["channels:read"])
        assert r1.agent_id == r2.agent_id

    @pytest.mark.asyncio
    async def test_enforcement_returns_policy_object(self, db):
        await create_agent_policy(_full_policy())
        result = await enforce_policy("user1", "flow-agent", "github", ["repo"])
        assert isinstance(result, AgentPolicy)
        assert result.agent_name == "Flow Bot"
        assert result.rate_limit_per_minute == 60
