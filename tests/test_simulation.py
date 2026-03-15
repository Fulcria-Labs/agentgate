"""Tests for policy simulation engine."""

import time
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from src.database import (
    init_db, create_agent_policy, AgentPolicy,
)
from src.simulation import simulate_token_request, SimulationResult, CheckResult


USER_ID = "user|sim"
OTHER_USER = "user|other"


def make_policy(agent_id="sim-agent", services=None, scopes=None,
                user_id=USER_ID, is_active=True, **kwargs):
    return AgentPolicy(
        agent_id=agent_id,
        agent_name=f"Agent-{agent_id}",
        allowed_services=services or ["github", "slack"],
        allowed_scopes=scopes or {
            "github": ["repo", "read:user"],
            "slack": ["chat:write", "channels:read"],
        },
        rate_limit_per_minute=kwargs.get("rate_limit", 60),
        created_by=user_id,
        created_at=time.time(),
        is_active=is_active,
        allowed_hours=kwargs.get("allowed_hours", []),
        allowed_days=kwargs.get("allowed_days", []),
        expires_at=kwargs.get("expires_at", 0.0),
        ip_allowlist=kwargs.get("ip_allowlist", []),
        requires_step_up=kwargs.get("requires_step_up", []),
    )


@pytest_asyncio.fixture
async def agent(db):
    policy = make_policy()
    await create_agent_policy(policy)
    return policy


# ── Basic simulation tests ──


@pytest.mark.asyncio
async def test_successful_simulation(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    assert result.would_succeed is True
    assert all(c.passed for c in result.checks)


@pytest.mark.asyncio
async def test_simulation_returns_effective_scopes(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo", "read:user"],
    )
    assert result.would_succeed is True
    assert set(result.effective_scopes) == {"repo", "read:user"}


@pytest.mark.asyncio
async def test_result_to_dict(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    d = result.to_dict()
    assert "would_succeed" in d
    assert "checks" in d
    assert "checks_passed" in d
    assert "checks_failed" in d
    assert "total_checks" in d
    assert d["would_succeed"] is True
    assert d["checks_failed"] == 0


@pytest.mark.asyncio
async def test_simulation_has_timestamp(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    assert result.timestamp > 0


# ── Agent existence check ──


@pytest.mark.asyncio
async def test_nonexistent_agent(db):
    result = await simulate_token_request(
        USER_ID, "ghost", "github", ["repo"],
    )
    assert result.would_succeed is False
    assert any(c.name == "agent_exists" and not c.passed for c in result.checks)
    assert len(result.checks) == 1


@pytest.mark.asyncio
async def test_existing_agent_passes(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    exists_check = next(c for c in result.checks if c.name == "agent_exists")
    assert exists_check.passed is True


# ── Active status check ──


@pytest.mark.asyncio
async def test_disabled_agent_fails(db):
    await create_agent_policy(make_policy("disabled-agent", is_active=False))
    result = await simulate_token_request(
        USER_ID, "disabled-agent", "github", ["repo"],
    )
    assert result.would_succeed is False
    check = next(c for c in result.checks if c.name == "agent_active")
    assert check.passed is False


@pytest.mark.asyncio
async def test_active_agent_passes(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    check = next(c for c in result.checks if c.name == "agent_active")
    assert check.passed is True


# ── Ownership check ──


@pytest.mark.asyncio
async def test_wrong_user_fails(agent):
    result = await simulate_token_request(
        OTHER_USER, "sim-agent", "github", ["repo"],
    )
    assert result.would_succeed is False
    check = next(c for c in result.checks if c.name == "ownership")
    assert check.passed is False


@pytest.mark.asyncio
async def test_correct_user_passes(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    check = next(c for c in result.checks if c.name == "ownership")
    assert check.passed is True


# ── Expiration check ──


@pytest.mark.asyncio
async def test_expired_policy_fails(db):
    await create_agent_policy(make_policy(
        "expired-agent", expires_at=time.time() - 3600,
    ))
    result = await simulate_token_request(
        USER_ID, "expired-agent", "github", ["repo"],
    )
    assert result.would_succeed is False
    check = next(c for c in result.checks if c.name == "expiration")
    assert check.passed is False


@pytest.mark.asyncio
async def test_future_expiration_passes(db):
    await create_agent_policy(make_policy(
        "future-agent", expires_at=time.time() + 3600,
    ))
    result = await simulate_token_request(
        USER_ID, "future-agent", "github", ["repo"],
    )
    check = next(c for c in result.checks if c.name == "expiration")
    assert check.passed is True
    assert "expires in" in check.detail


@pytest.mark.asyncio
async def test_no_expiration_passes(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    check = next(c for c in result.checks if c.name == "expiration")
    assert check.passed is True
    assert "No expiration" in check.detail


# ── Time window check ──


@pytest.mark.asyncio
async def test_outside_allowed_hours(db):
    await create_agent_policy(make_policy(
        "hour-agent", allowed_hours=[9, 10, 11],
    ))
    result = await simulate_token_request(
        USER_ID, "hour-agent", "github", ["repo"],
        now=datetime(2026, 3, 15, 15, 0, 0, tzinfo=timezone.utc),
    )
    assert result.would_succeed is False
    check = next(c for c in result.checks if c.name == "time_window")
    assert check.passed is False


@pytest.mark.asyncio
async def test_inside_allowed_hours(db):
    await create_agent_policy(make_policy(
        "hour-agent2", allowed_hours=[9, 10, 11],
    ))
    result = await simulate_token_request(
        USER_ID, "hour-agent2", "github", ["repo"],
        now=datetime(2026, 3, 15, 10, 30, 0, tzinfo=timezone.utc),
    )
    check = next(c for c in result.checks if c.name == "time_window")
    assert check.passed is True


@pytest.mark.asyncio
async def test_outside_allowed_days(db):
    await create_agent_policy(make_policy(
        "day-agent", allowed_days=[0, 1, 2, 3, 4],
    ))
    result = await simulate_token_request(
        USER_ID, "day-agent", "github", ["repo"],
        now=datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc),
    )
    check = next(c for c in result.checks if c.name == "time_window")
    assert check.passed is False


@pytest.mark.asyncio
async def test_no_time_restrictions(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    check = next(c for c in result.checks if c.name == "time_window")
    assert check.passed is True


# ── IP allowlist check ──


@pytest.mark.asyncio
async def test_ip_not_in_allowlist(db):
    await create_agent_policy(make_policy(
        "ip-agent", ip_allowlist=["10.0.0.1", "192.168.1.0/24"],
    ))
    result = await simulate_token_request(
        USER_ID, "ip-agent", "github", ["repo"],
        ip_address="172.16.0.5",
    )
    assert result.would_succeed is False
    check = next(c for c in result.checks if c.name == "ip_allowlist")
    assert check.passed is False


@pytest.mark.asyncio
async def test_ip_in_allowlist(db):
    await create_agent_policy(make_policy(
        "ip-agent2", ip_allowlist=["10.0.0.0/8"],
    ))
    result = await simulate_token_request(
        USER_ID, "ip-agent2", "github", ["repo"],
        ip_address="10.5.3.2",
    )
    check = next(c for c in result.checks if c.name == "ip_allowlist")
    assert check.passed is True


@pytest.mark.asyncio
async def test_no_ip_provided_skips(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    check = next(c for c in result.checks if c.name == "ip_allowlist")
    assert check.passed is True
    assert "skipped" in check.detail


@pytest.mark.asyncio
async def test_no_ip_restrictions(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
        ip_address="1.2.3.4",
    )
    check = next(c for c in result.checks if c.name == "ip_allowlist")
    assert check.passed is True


# ── Service authorization check ──


@pytest.mark.asyncio
async def test_unauthorized_service(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "google", ["mail.read"],
    )
    assert result.would_succeed is False
    check = next(c for c in result.checks if c.name == "service_auth")
    assert check.passed is False


@pytest.mark.asyncio
async def test_authorized_service(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    check = next(c for c in result.checks if c.name == "service_auth")
    assert check.passed is True


# ── Scope validation check ──


@pytest.mark.asyncio
async def test_excess_scopes_fail(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo", "admin:org"],
    )
    assert result.would_succeed is False
    check = next(c for c in result.checks if c.name == "scope_validation")
    assert check.passed is False
    assert "admin:org" in check.detail


@pytest.mark.asyncio
async def test_valid_scopes_pass(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    check = next(c for c in result.checks if c.name == "scope_validation")
    assert check.passed is True


@pytest.mark.asyncio
async def test_empty_scopes_pass(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", [],
    )
    check = next(c for c in result.checks if c.name == "scope_validation")
    assert check.passed is True


@pytest.mark.asyncio
async def test_all_allowed_scopes_pass(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo", "read:user"],
    )
    check = next(c for c in result.checks if c.name == "scope_validation")
    assert check.passed is True
    assert set(result.effective_scopes) == {"repo", "read:user"}


# ── Rate limit check ──


@pytest.mark.asyncio
async def test_within_rate_limit(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    check = next(c for c in result.checks if c.name == "rate_limit")
    assert check.passed is True


@pytest.mark.asyncio
async def test_rate_limit_shows_count(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    check = next(c for c in result.checks if c.name == "rate_limit")
    assert "/60" in check.detail


# ── Step-up auth check ──


@pytest.mark.asyncio
async def test_step_up_required(db):
    await create_agent_policy(make_policy(
        "stepup-agent", requires_step_up=["github"],
    ))
    result = await simulate_token_request(
        USER_ID, "stepup-agent", "github", ["repo"],
    )
    check = next(c for c in result.checks if c.name == "step_up_auth")
    assert check.passed is True
    assert "CIBA" in check.detail


@pytest.mark.asyncio
async def test_no_step_up(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    check = next(c for c in result.checks if c.name == "step_up_auth")
    assert check.passed is True
    assert "No step-up" in check.detail


# ── Multiple failures ──


@pytest.mark.asyncio
async def test_all_checks_reported(db):
    await create_agent_policy(make_policy(
        "multi-fail",
        is_active=False,
        user_id=OTHER_USER,
        expires_at=time.time() - 3600,
    ))
    result = await simulate_token_request(
        USER_ID, "multi-fail", "google", ["admin"],
    )
    assert result.would_succeed is False
    failed_checks = [c for c in result.checks if not c.passed]
    assert len(failed_checks) >= 3


@pytest.mark.asyncio
async def test_summary_counts(db):
    await create_agent_policy(make_policy("count-agent"))
    result = await simulate_token_request(
        USER_ID, "count-agent", "github", ["repo"],
    )
    d = result.to_dict()
    assert d["checks_passed"] + d["checks_failed"] == d["total_checks"]


# ── Integration with various policy configurations ──


@pytest.mark.asyncio
async def test_fully_restricted_agent(db):
    await create_agent_policy(make_policy(
        "restricted",
        allowed_hours=[2, 3, 4],
        allowed_days=[0],
        ip_allowlist=["10.0.0.1"],
        requires_step_up=["github"],
    ))
    result = await simulate_token_request(
        USER_ID, "restricted", "github", ["repo"],
        ip_address="192.168.1.1",
        now=datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert result.would_succeed is False
    failed = {c.name for c in result.checks if not c.passed}
    assert "time_window" in failed
    assert "ip_allowlist" in failed


@pytest.mark.asyncio
async def test_fully_open_agent(db):
    await create_agent_policy(make_policy("open-agent"))
    result = await simulate_token_request(
        USER_ID, "open-agent", "github", ["repo"],
    )
    assert result.would_succeed is True
    assert all(c.passed for c in result.checks)


@pytest.mark.asyncio
async def test_slack_service(agent):
    result = await simulate_token_request(
        USER_ID, "sim-agent", "slack", ["chat:write"],
    )
    assert result.would_succeed is True


@pytest.mark.asyncio
async def test_simulation_does_not_modify_state(agent):
    r1 = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    r2 = await simulate_token_request(
        USER_ID, "sim-agent", "github", ["repo"],
    )
    assert r1.would_succeed is True
    assert r2.would_succeed is True
