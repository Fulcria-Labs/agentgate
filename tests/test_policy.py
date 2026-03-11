"""Tests for the policy enforcement engine."""

import time
from datetime import datetime, timezone
import pytest

from src.database import AgentPolicy, create_agent_policy, init_db
from src.policy import (
    PolicyDenied,
    check_ip_allowlist,
    check_time_window,
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


# --- Ownership Verification Tests ---

@pytest.mark.asyncio
async def test_enforce_policy_ownership_denied(db, sample_policy):
    """User who doesn't own the policy is denied access."""
    await create_agent_policy(sample_policy)  # created_by = "user1"
    with pytest.raises(PolicyDenied, match="do not own"):
        await enforce_policy("user2", "test-agent", "github", ["repo"])


@pytest.mark.asyncio
async def test_enforce_policy_ownership_success(db, sample_policy):
    """User who owns the policy can access."""
    await create_agent_policy(sample_policy)  # created_by = "user1"
    result = await enforce_policy("user1", "test-agent", "github", ["repo"])
    assert result.agent_id == "test-agent"


# --- Time Window Tests ---

def test_check_time_window_allowed_hours():
    """Access during allowed hours passes."""
    policy = AgentPolicy(
        agent_id="t", agent_name="t",
        allowed_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
    )
    # 10:30 UTC should pass
    now = datetime(2026, 3, 11, 10, 30, tzinfo=timezone.utc)
    assert check_time_window(policy, now) is None


def test_check_time_window_denied_hours():
    """Access outside allowed hours is denied."""
    policy = AgentPolicy(
        agent_id="t", agent_name="t",
        allowed_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
    )
    # 3:00 UTC should be denied
    now = datetime(2026, 3, 11, 3, 0, tzinfo=timezone.utc)
    result = check_time_window(policy, now)
    assert result is not None
    assert "not within allowed hours" in result


def test_check_time_window_allowed_days():
    """Access on allowed weekdays passes."""
    policy = AgentPolicy(
        agent_id="t", agent_name="t",
        allowed_days=[0, 1, 2, 3, 4],  # Mon-Fri
    )
    # Wednesday = weekday 2
    now = datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)  # Mar 11 2026 = Wednesday
    assert check_time_window(policy, now) is None


def test_check_time_window_denied_days():
    """Access on disallowed weekdays is denied."""
    policy = AgentPolicy(
        agent_id="t", agent_name="t",
        allowed_days=[0, 1, 2, 3, 4],  # Mon-Fri only
    )
    # Saturday = weekday 5
    now = datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)  # Mar 14 2026 = Saturday
    result = check_time_window(policy, now)
    assert result is not None
    assert "not within allowed days" in result


def test_check_time_window_empty_means_always():
    """Empty allowed_hours and allowed_days means always allowed."""
    policy = AgentPolicy(agent_id="t", agent_name="t")
    # Any time should pass
    now = datetime(2026, 3, 14, 3, 0, tzinfo=timezone.utc)
    assert check_time_window(policy, now) is None


def test_check_time_window_both_constraints():
    """Both day and hour constraints are enforced together."""
    policy = AgentPolicy(
        agent_id="t", agent_name="t",
        allowed_days=[0, 1, 2, 3, 4],  # Mon-Fri
        allowed_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],  # 9-17 UTC
    )
    # Wednesday 10:00 UTC - both pass
    now = datetime(2026, 3, 11, 10, 0, tzinfo=timezone.utc)
    assert check_time_window(policy, now) is None

    # Wednesday 3:00 UTC - day OK but hour denied
    now = datetime(2026, 3, 11, 3, 0, tzinfo=timezone.utc)
    result = check_time_window(policy, now)
    assert "hours" in result

    # Saturday 10:00 UTC - hour OK but day denied
    now = datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc)
    result = check_time_window(policy, now)
    assert "days" in result


@pytest.mark.asyncio
async def test_enforce_policy_time_window_denied(db):
    """Policy enforcement denies access outside time window."""
    policy = AgentPolicy(
        agent_id="time-agent",
        agent_name="Time Bot",
        allowed_services=["github"],
        allowed_scopes={"github": ["repo"]},
        allowed_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
        rate_limit_per_minute=60,
        created_by="user1",
        created_at=time.time(),
        is_active=True,
    )
    await create_agent_policy(policy)

    # Patch the time to be outside allowed hours
    from unittest.mock import patch
    with patch("src.policy.check_time_window", return_value="Access denied: not within allowed hours"):
        with pytest.raises(PolicyDenied, match="not within allowed hours"):
            await enforce_policy("user1", "time-agent", "github", ["repo"])


# --- IP Allowlist Tests ---

def test_check_ip_allowlist_empty():
    """Empty allowlist means allow all IPs."""
    policy = AgentPolicy(agent_id="t", agent_name="t", ip_allowlist=[])
    assert check_ip_allowlist(policy, "192.168.1.1") is None


def test_check_ip_allowlist_allowed():
    """IP in allowlist is accepted."""
    policy = AgentPolicy(agent_id="t", agent_name="t", ip_allowlist=["10.0.0.1", "10.0.0.2"])
    assert check_ip_allowlist(policy, "10.0.0.1") is None


def test_check_ip_allowlist_denied():
    """IP not in allowlist is denied."""
    policy = AgentPolicy(agent_id="t", agent_name="t", ip_allowlist=["10.0.0.1", "10.0.0.2"])
    result = check_ip_allowlist(policy, "192.168.1.1")
    assert result is not None
    assert "not in allowlist" in result


@pytest.mark.asyncio
async def test_enforce_policy_ip_denied(db):
    """Policy enforcement denies access from unauthorized IP."""
    policy = AgentPolicy(
        agent_id="ip-agent",
        agent_name="IP Bot",
        allowed_services=["github"],
        allowed_scopes={"github": ["repo"]},
        ip_allowlist=["10.0.0.1"],
        rate_limit_per_minute=60,
        created_by="user1",
        created_at=time.time(),
        is_active=True,
    )
    await create_agent_policy(policy)
    with pytest.raises(PolicyDenied, match="not in allowlist"):
        await enforce_policy("user1", "ip-agent", "github", ["repo"], ip_address="192.168.1.99")


@pytest.mark.asyncio
async def test_enforce_policy_ip_allowed(db):
    """Policy enforcement allows access from authorized IP."""
    policy = AgentPolicy(
        agent_id="ip-agent2",
        agent_name="IP Bot",
        allowed_services=["github"],
        allowed_scopes={"github": ["repo"]},
        ip_allowlist=["10.0.0.1"],
        rate_limit_per_minute=60,
        created_by="user1",
        created_at=time.time(),
        is_active=True,
    )
    await create_agent_policy(policy)
    result = await enforce_policy("user1", "ip-agent2", "github", ["repo"], ip_address="10.0.0.1")
    assert result.agent_id == "ip-agent2"


# --- Policy Expiration Tests ---

@pytest.mark.asyncio
async def test_enforce_policy_expired(db):
    """Expired policy is denied."""
    policy = AgentPolicy(
        agent_id="expired-agent",
        agent_name="Expired Bot",
        allowed_services=["github"],
        allowed_scopes={"github": ["repo"]},
        rate_limit_per_minute=60,
        created_by="user1",
        created_at=time.time(),
        is_active=True,
        expires_at=time.time() - 3600,  # Expired 1 hour ago
    )
    await create_agent_policy(policy)
    with pytest.raises(PolicyDenied, match="expired"):
        await enforce_policy("user1", "expired-agent", "github", ["repo"])


@pytest.mark.asyncio
async def test_enforce_policy_not_expired(db):
    """Non-expired policy passes."""
    policy = AgentPolicy(
        agent_id="valid-agent",
        agent_name="Valid Bot",
        allowed_services=["github"],
        allowed_scopes={"github": ["repo"]},
        rate_limit_per_minute=60,
        created_by="user1",
        created_at=time.time(),
        is_active=True,
        expires_at=time.time() + 3600,  # Expires in 1 hour
    )
    await create_agent_policy(policy)
    result = await enforce_policy("user1", "valid-agent", "github", ["repo"])
    assert result.agent_id == "valid-agent"


@pytest.mark.asyncio
async def test_enforce_policy_no_expiration(db):
    """Policy with expires_at=0 never expires."""
    policy = AgentPolicy(
        agent_id="forever-agent",
        agent_name="Forever Bot",
        allowed_services=["github"],
        allowed_scopes={"github": ["repo"]},
        rate_limit_per_minute=60,
        created_by="user1",
        created_at=time.time(),
        is_active=True,
        expires_at=0.0,
    )
    await create_agent_policy(policy)
    result = await enforce_policy("user1", "forever-agent", "github", ["repo"])
    assert result.agent_id == "forever-agent"


# --- CIDR IP Allowlist Tests ---

def test_check_ip_allowlist_cidr_match():
    """IP within a CIDR range is accepted."""
    policy = AgentPolicy(
        agent_id="t", agent_name="t",
        ip_allowlist=["10.0.0.0/24"],
    )
    assert check_ip_allowlist(policy, "10.0.0.42") is None


def test_check_ip_allowlist_cidr_no_match():
    """IP outside a CIDR range is denied."""
    policy = AgentPolicy(
        agent_id="t", agent_name="t",
        ip_allowlist=["10.0.0.0/24"],
    )
    result = check_ip_allowlist(policy, "10.0.1.1")
    assert result is not None
    assert "not in allowlist" in result


def test_check_ip_allowlist_mixed_exact_and_cidr():
    """Mix of exact IPs and CIDR ranges works correctly."""
    policy = AgentPolicy(
        agent_id="t", agent_name="t",
        ip_allowlist=["192.168.1.1", "10.0.0.0/16"],
    )
    # Exact match
    assert check_ip_allowlist(policy, "192.168.1.1") is None
    # CIDR match
    assert check_ip_allowlist(policy, "10.0.5.100") is None
    # No match
    result = check_ip_allowlist(policy, "172.16.0.1")
    assert "not in allowlist" in result


def test_check_ip_allowlist_ipv6_cidr():
    """IPv6 CIDR ranges are supported."""
    policy = AgentPolicy(
        agent_id="t", agent_name="t",
        ip_allowlist=["2001:db8::/32"],
    )
    assert check_ip_allowlist(policy, "2001:db8::1") is None
    result = check_ip_allowlist(policy, "2001:db9::1")
    assert "not in allowlist" in result


def test_check_ip_allowlist_invalid_ip():
    """Invalid requesting IP is denied with clear message."""
    policy = AgentPolicy(
        agent_id="t", agent_name="t",
        ip_allowlist=["10.0.0.1"],
    )
    result = check_ip_allowlist(policy, "not-an-ip")
    assert result is not None
    assert "invalid IP" in result
