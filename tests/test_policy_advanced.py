"""Advanced policy engine tests — edge cases, multi-constraint, and security scenarios."""

import time
from datetime import datetime, timezone
import pytest

from src.database import AgentPolicy, create_agent_policy
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
def base_policy():
    """A fully-configured policy for testing."""
    return AgentPolicy(
        agent_id="adv-agent",
        agent_name="Advanced Bot",
        allowed_services=["github", "slack", "jira"],
        allowed_scopes={
            "github": ["repo", "read:user", "write:packages"],
            "slack": ["channels:read", "chat:write", "files:read"],
            "jira": ["read:jira-work", "write:jira-work"],
        },
        rate_limit_per_minute=10,
        requires_step_up=["jira"],
        allowed_hours=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        allowed_days=[0, 1, 2, 3, 4],  # Mon-Fri
        ip_allowlist=["10.0.0.0/8", "172.16.0.0/12"],
        created_by="admin-user",
        created_at=time.time(),
        is_active=True,
        expires_at=time.time() + 86400 * 30,  # 30 days from now
    )


@pytest.fixture(autouse=True)
def clear_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


# --- Effective Scopes Edge Cases ---

class TestEffectiveScopes:
    def test_empty_requested_returns_empty(self, base_policy):
        result = get_effective_scopes(base_policy, "github", [])
        assert result == []

    def test_all_requested_are_allowed(self, base_policy):
        result = get_effective_scopes(base_policy, "github", ["repo", "read:user"])
        assert set(result) == {"repo", "read:user"}

    def test_partial_overlap(self, base_policy):
        result = get_effective_scopes(base_policy, "github", ["repo", "admin:org"])
        assert result == ["repo"]

    def test_empty_allowed_for_service(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={"github": []},
        )
        result = get_effective_scopes(policy, "github", ["repo"])
        assert result == []

    def test_multiple_services_isolation(self, base_policy):
        github_scopes = get_effective_scopes(base_policy, "github", ["repo", "channels:read"])
        slack_scopes = get_effective_scopes(base_policy, "slack", ["repo", "channels:read"])
        assert set(github_scopes) == {"repo"}
        assert set(slack_scopes) == {"channels:read"}

    def test_case_sensitive_scopes(self, base_policy):
        result = get_effective_scopes(base_policy, "github", ["Repo", "REPO", "repo"])
        assert result == ["repo"]


# --- Step-up Auth ---

class TestStepUpAuth:
    def test_step_up_required_for_configured_service(self, base_policy):
        assert requires_step_up(base_policy, "jira") is True

    def test_step_up_not_required_for_other_service(self, base_policy):
        assert requires_step_up(base_policy, "github") is False
        assert requires_step_up(base_policy, "slack") is False

    def test_step_up_not_required_when_empty(self):
        policy = AgentPolicy(agent_id="t", agent_name="t", requires_step_up=[])
        assert requires_step_up(policy, "anything") is False

    def test_step_up_for_unknown_service(self, base_policy):
        assert requires_step_up(base_policy, "unknown") is False


# --- Time Window Edge Cases ---

class TestTimeWindowEdgeCases:
    def test_midnight_boundary(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_hours=[23, 0, 1],
        )
        assert check_time_window(policy, datetime(2026, 3, 11, 23, 59, tzinfo=timezone.utc)) is None
        assert check_time_window(policy, datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc)) is None
        assert check_time_window(policy, datetime(2026, 3, 12, 1, 30, tzinfo=timezone.utc)) is None
        assert check_time_window(policy, datetime(2026, 3, 12, 2, 0, tzinfo=timezone.utc)) is not None

    def test_single_hour_window(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_hours=[12],
        )
        assert check_time_window(policy, datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)) is None
        assert check_time_window(policy, datetime(2026, 3, 11, 12, 59, tzinfo=timezone.utc)) is None
        assert check_time_window(policy, datetime(2026, 3, 11, 13, 0, tzinfo=timezone.utc)) is not None

    def test_weekend_only(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_days=[5, 6],  # Sat, Sun
        )
        # Saturday
        assert check_time_window(policy, datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)) is None
        # Sunday
        assert check_time_window(policy, datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)) is None
        # Monday
        assert check_time_window(policy, datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc)) is not None

    def test_all_hours_all_days(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_hours=list(range(24)),
            allowed_days=list(range(7)),
        )
        assert check_time_window(policy, datetime(2026, 3, 14, 3, 0, tzinfo=timezone.utc)) is None

    def test_denial_message_includes_day_names(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_days=[0, 4],  # Mon, Fri
        )
        result = check_time_window(policy, datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc))  # Wednesday
        assert result is not None
        assert "Mon" in result
        assert "Fri" in result


# --- IP Allowlist Edge Cases ---

class TestIPAllowlistEdgeCases:
    def test_localhost_ipv4(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["127.0.0.1"],
        )
        assert check_ip_allowlist(policy, "127.0.0.1") is None
        assert check_ip_allowlist(policy, "127.0.0.2") is not None

    def test_localhost_ipv6(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["::1"],
        )
        assert check_ip_allowlist(policy, "::1") is None

    def test_wide_cidr_slash_0(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["0.0.0.0/0"],
        )
        assert check_ip_allowlist(policy, "1.2.3.4") is None
        assert check_ip_allowlist(policy, "255.255.255.255") is None

    def test_single_host_cidr_slash_32(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["10.0.0.1/32"],
        )
        assert check_ip_allowlist(policy, "10.0.0.1") is None
        assert check_ip_allowlist(policy, "10.0.0.2") is not None

    def test_malformed_allowlist_entry_skipped(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["not-valid", "10.0.0.1"],
        )
        # Should still match the valid entry
        assert check_ip_allowlist(policy, "10.0.0.1") is None
        # Should deny non-matching
        assert check_ip_allowlist(policy, "10.0.0.2") is not None

    def test_empty_string_ip(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["10.0.0.1"],
        )
        result = check_ip_allowlist(policy, "")
        assert result is not None
        assert "invalid IP" in result

    def test_multiple_cidr_ranges(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"],
        )
        assert check_ip_allowlist(policy, "10.1.2.3") is None
        assert check_ip_allowlist(policy, "172.20.0.1") is None
        assert check_ip_allowlist(policy, "192.168.5.5") is None
        assert check_ip_allowlist(policy, "8.8.8.8") is not None


# --- Multi-Constraint Enforcement Tests ---

class TestMultiConstraintEnforcement:
    @pytest.mark.asyncio
    async def test_all_constraints_pass(self, db, base_policy):
        await create_agent_policy(base_policy)
        # This test mocks time to be within allowed window
        from unittest.mock import patch
        with patch("src.policy.check_time_window", return_value=None):
            result = await enforce_policy(
                "admin-user", "adv-agent", "github", ["repo"],
                ip_address="10.0.0.5",
            )
            assert result.agent_id == "adv-agent"

    @pytest.mark.asyncio
    async def test_ip_fails_but_other_constraints_pass(self, db, base_policy):
        await create_agent_policy(base_policy)
        from unittest.mock import patch
        with patch("src.policy.check_time_window", return_value=None):
            with pytest.raises(PolicyDenied, match="not in allowlist"):
                await enforce_policy(
                    "admin-user", "adv-agent", "github", ["repo"],
                    ip_address="8.8.8.8",
                )

    @pytest.mark.asyncio
    async def test_rate_limit_independent_per_service(self, db):
        policy = AgentPolicy(
            agent_id="rate-agent",
            agent_name="Rate Bot",
            allowed_services=["svc-a", "svc-b"],
            allowed_scopes={"svc-a": ["read"], "svc-b": ["read"]},
            rate_limit_per_minute=2,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)

        # 2 requests to svc-a: succeed
        await enforce_policy("user1", "rate-agent", "svc-a", ["read"])
        await enforce_policy("user1", "rate-agent", "svc-a", ["read"])

        # 3rd to svc-a: rate limited
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy("user1", "rate-agent", "svc-a", ["read"])

        # But svc-b should still work (independent counter)
        result = await enforce_policy("user1", "rate-agent", "svc-b", ["read"])
        assert result.agent_id == "rate-agent"


# --- PolicyDenied Exception Tests ---

class TestPolicyDeniedException:
    def test_exception_message(self):
        ex = PolicyDenied("test reason", "agent-1", "github")
        assert str(ex) == "test reason"
        assert ex.agent_id == "agent-1"
        assert ex.service == "github"

    def test_exception_default_fields(self):
        ex = PolicyDenied("reason")
        assert ex.agent_id == ""
        assert ex.service == ""

    def test_exception_is_catchable(self):
        with pytest.raises(PolicyDenied):
            raise PolicyDenied("denied")

    def test_exception_inherits_from_exception(self):
        ex = PolicyDenied("test")
        assert isinstance(ex, Exception)
