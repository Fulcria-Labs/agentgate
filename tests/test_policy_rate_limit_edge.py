"""Rate limiting edge case tests — boundary conditions, window expiry,
concurrent agents, counter isolation, and burst patterns."""

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


def _make_policy(agent_id="rate-agent", user="user1", rate_limit=5, services=None):
    return AgentPolicy(
        agent_id=agent_id,
        agent_name="Rate Bot",
        allowed_services=services or ["github"],
        allowed_scopes={"github": ["repo"], "slack": ["channels:read"]},
        rate_limit_per_minute=rate_limit,
        created_by=user,
        created_at=time.time(),
        is_active=True,
    )


class TestRateLimitExactBoundary:
    """Verify behaviour exactly at the rate limit boundary."""

    @pytest.mark.asyncio
    async def test_exactly_at_limit_is_allowed(self, db):
        """N requests where N == limit should all succeed."""
        policy = _make_policy(rate_limit=3)
        await create_agent_policy(policy)
        for _ in range(3):
            result = await enforce_policy("user1", "rate-agent", "github", ["repo"])
            assert result.agent_id == "rate-agent"

    @pytest.mark.asyncio
    async def test_one_over_limit_is_denied(self, db):
        """N+1 request where N == limit should fail."""
        policy = _make_policy(rate_limit=3)
        await create_agent_policy(policy)
        for _ in range(3):
            await enforce_policy("user1", "rate-agent", "github", ["repo"])
        with pytest.raises(PolicyDenied, match="Rate limit exceeded"):
            await enforce_policy("user1", "rate-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_limit_of_one(self, db):
        """Rate limit of 1 allows exactly one request."""
        policy = _make_policy(rate_limit=1)
        await create_agent_policy(policy)
        await enforce_policy("user1", "rate-agent", "github", ["repo"])
        with pytest.raises(PolicyDenied, match="Rate limit exceeded"):
            await enforce_policy("user1", "rate-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_limit_of_zero_denies_all(self, db):
        """Rate limit of 0 denies the very first request."""
        policy = _make_policy(rate_limit=0)
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="Rate limit exceeded"):
            await enforce_policy("user1", "rate-agent", "github", ["repo"])


class TestRateLimitWindowExpiry:
    """Verify that old counter entries expire after the 60s window."""

    @pytest.mark.asyncio
    async def test_old_entries_expire(self, db):
        """Requests older than 60s are removed from the counter window."""
        policy = _make_policy(rate_limit=2)
        await create_agent_policy(policy)

        # Manually inject old entries
        key = "rate-agent:github"
        _rate_counters[key] = [time.time() - 120, time.time() - 90]

        # Should succeed because old entries expire
        result = await enforce_policy("user1", "rate-agent", "github", ["repo"])
        assert result.agent_id == "rate-agent"

    @pytest.mark.asyncio
    async def test_mix_of_old_and_new_entries(self, db):
        """Only recent entries count toward the limit."""
        policy = _make_policy(rate_limit=2)
        await create_agent_policy(policy)

        now = time.time()
        key = "rate-agent:github"
        _rate_counters[key] = [now - 120, now - 30]  # 1 old + 1 recent

        # 1 recent entry + 1 new = 2, at limit but OK
        await enforce_policy("user1", "rate-agent", "github", ["repo"])
        # Now at limit (2), next one denied
        with pytest.raises(PolicyDenied, match="Rate limit exceeded"):
            await enforce_policy("user1", "rate-agent", "github", ["repo"])


class TestRateLimitCounterIsolation:
    """Rate limit counters are isolated by agent_id:service."""

    @pytest.mark.asyncio
    async def test_different_agents_separate_counters(self, db):
        """Two agents have independent rate limit counters."""
        p1 = _make_policy(agent_id="agent-A", rate_limit=2)
        p2 = _make_policy(agent_id="agent-B", rate_limit=2)
        await create_agent_policy(p1)
        await create_agent_policy(p2)

        # Exhaust agent-A
        for _ in range(2):
            await enforce_policy("user1", "agent-A", "github", ["repo"])
        with pytest.raises(PolicyDenied):
            await enforce_policy("user1", "agent-A", "github", ["repo"])

        # agent-B still OK
        result = await enforce_policy("user1", "agent-B", "github", ["repo"])
        assert result.agent_id == "agent-B"

    @pytest.mark.asyncio
    async def test_different_services_separate_counters(self, db):
        """Same agent accessing different services has separate counters."""
        policy = _make_policy(rate_limit=2, services=["github", "slack"])
        await create_agent_policy(policy)

        # Exhaust github counter
        for _ in range(2):
            await enforce_policy("user1", "rate-agent", "github", ["repo"])
        with pytest.raises(PolicyDenied):
            await enforce_policy("user1", "rate-agent", "github", ["repo"])

        # slack counter is independent
        result = await enforce_policy("user1", "rate-agent", "slack", ["channels:read"])
        assert result.agent_id == "rate-agent"


class TestRateLimitCounterKey:
    """Verify rate counter key format."""

    @pytest.mark.asyncio
    async def test_counter_key_format(self, db):
        """Rate counters use 'agent_id:service' as key."""
        policy = _make_policy(agent_id="my-agent", rate_limit=10)
        await create_agent_policy(policy)
        await enforce_policy("user1", "my-agent", "github", ["repo"])
        assert "my-agent:github" in _rate_counters

    @pytest.mark.asyncio
    async def test_counter_incremented_on_success(self, db):
        """Each successful request adds an entry to the counter."""
        policy = _make_policy(agent_id="cnt-agent", rate_limit=10)
        await create_agent_policy(policy)

        await enforce_policy("user1", "cnt-agent", "github", ["repo"])
        assert len(_rate_counters["cnt-agent:github"]) == 1

        await enforce_policy("user1", "cnt-agent", "github", ["repo"])
        assert len(_rate_counters["cnt-agent:github"]) == 2


class TestRateLimitHighVolume:
    """High rate limits still work correctly."""

    @pytest.mark.asyncio
    async def test_high_rate_limit(self, db):
        """Rate limit of 1000 allows many requests."""
        policy = _make_policy(rate_limit=1000)
        await create_agent_policy(policy)
        for _ in range(100):
            result = await enforce_policy("user1", "rate-agent", "github", ["repo"])
            assert result.agent_id == "rate-agent"

    @pytest.mark.asyncio
    async def test_rate_limit_message_includes_limit(self, db):
        """Denial message includes the configured rate limit."""
        policy = _make_policy(rate_limit=7)
        await create_agent_policy(policy)
        for _ in range(7):
            await enforce_policy("user1", "rate-agent", "github", ["repo"])
        with pytest.raises(PolicyDenied, match="7/min"):
            await enforce_policy("user1", "rate-agent", "github", ["repo"])
