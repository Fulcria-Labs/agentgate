"""Tests for the policy enforcement engine."""

import time
import pytest

from src.database import AgentPolicy, create_agent_policy, init_db
from src.policy import (
    PolicyDenied,
    enforce_policy,
    get_effective_scopes,
    requires_step_up,
    _rate_counters,
)


@pytest.fixture
def sample_policy():
    return AgentPolicy(
        agent_id="test-agent",
        agent_name="Test Bot",
        allowed_services=["github", "slack"],
        allowed_scopes={
            "github": ["repo", "read:user"],
            "slack": ["channels:read", "chat:write"],
        },
        rate_limit_per_minute=5,
        requires_step_up=["slack"],
        created_by="user1",
        created_at=time.time(),
        is_active=True,
    )


@pytest.fixture(autouse=True)
def clear_rate_counters():
    """Clear rate limit counters between tests."""
    _rate_counters.clear()
    yield
    _rate_counters.clear()


@pytest.mark.asyncio
async def test_enforce_policy_success(db, sample_policy):
    """Valid request passes policy enforcement."""
    await create_agent_policy(sample_policy)
    result = await enforce_policy("user1", "test-agent", "github", ["repo"])
    assert result.agent_id == "test-agent"


@pytest.mark.asyncio
async def test_enforce_policy_unknown_agent(db):
    """Unknown agent is denied."""
    with pytest.raises(PolicyDenied, match="not registered"):
        await enforce_policy("user1", "unknown-agent", "github", ["repo"])


@pytest.mark.asyncio
async def test_enforce_policy_disabled_agent(db, sample_policy):
    """Disabled agent is denied."""
    sample_policy.is_active = False
    await create_agent_policy(sample_policy)
    with pytest.raises(PolicyDenied, match="disabled"):
        await enforce_policy("user1", "test-agent", "github", ["repo"])


@pytest.mark.asyncio
async def test_enforce_policy_unauthorized_service(db, sample_policy):
    """Agent requesting unauthorized service is denied."""
    await create_agent_policy(sample_policy)
    with pytest.raises(PolicyDenied, match="not authorized to access"):
        await enforce_policy("user1", "test-agent", "google", ["gmail.readonly"])


@pytest.mark.asyncio
async def test_enforce_policy_excess_scopes(db, sample_policy):
    """Agent requesting excess scopes is denied."""
    await create_agent_policy(sample_policy)
    with pytest.raises(PolicyDenied, match="Scopes not permitted"):
        await enforce_policy("user1", "test-agent", "github", ["repo", "admin:org"])


@pytest.mark.asyncio
async def test_enforce_policy_rate_limit(db, sample_policy):
    """Agent exceeding rate limit is denied."""
    sample_policy.rate_limit_per_minute = 3
    await create_agent_policy(sample_policy)

    # First 3 requests succeed
    for _ in range(3):
        await enforce_policy("user1", "test-agent", "github", ["repo"])

    # 4th request is rate limited
    with pytest.raises(PolicyDenied, match="Rate limit exceeded"):
        await enforce_policy("user1", "test-agent", "github", ["repo"])


@pytest.mark.asyncio
async def test_enforce_policy_empty_scopes(db, sample_policy):
    """Empty scopes list passes (no excess scopes)."""
    await create_agent_policy(sample_policy)
    result = await enforce_policy("user1", "test-agent", "github", [])
    assert result.agent_id == "test-agent"


def test_requires_step_up_true(sample_policy):
    """Services in requires_step_up list return True."""
    assert requires_step_up(sample_policy, "slack") is True


def test_requires_step_up_false(sample_policy):
    """Services not in requires_step_up list return False."""
    assert requires_step_up(sample_policy, "github") is False


def test_get_effective_scopes(sample_policy):
    """Effective scopes are the intersection of requested and allowed."""
    result = get_effective_scopes(sample_policy, "github", ["repo", "admin:org", "read:user"])
    assert set(result) == {"repo", "read:user"}


def test_get_effective_scopes_no_overlap(sample_policy):
    """No overlap returns empty list."""
    result = get_effective_scopes(sample_policy, "github", ["admin:org", "delete_repo"])
    assert result == []


def test_get_effective_scopes_unknown_service(sample_policy):
    """Unknown service returns empty list."""
    result = get_effective_scopes(sample_policy, "unknown", ["anything"])
    assert result == []
