"""Comprehensive policy enforcement tests — covering every denial path,
boundary conditions, and multi-factor constraint combinations."""

import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.database import AgentPolicy, create_agent_policy
from src.policy import (
    PolicyDenied,
    _rate_counters,
    check_ip_allowlist,
    check_time_window,
    enforce_policy,
    get_effective_scopes,
    requires_step_up,
)


@pytest.fixture(autouse=True)
def clear_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


# ---------------------------------------------------------------------------
# 1. Enforcement with all constraints simultaneously
# ---------------------------------------------------------------------------

class TestFullConstraintStack:
    """Test enforcement when ALL constraint types are active at once."""

    @pytest.mark.asyncio
    async def test_all_constraints_satisfied(self, db):
        policy = AgentPolicy(
            agent_id="full-ok",
            agent_name="Full Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=10,
            allowed_hours=list(range(24)),
            allowed_days=list(range(7)),
            ip_allowlist=["10.0.0.0/8"],
            expires_at=time.time() + 86400,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        result = await enforce_policy("user1", "full-ok", "github", ["repo"],
                                     ip_address="10.0.0.1")
        assert result.agent_id == "full-ok"

    @pytest.mark.asyncio
    async def test_ip_denied_in_full_stack(self, db):
        policy = AgentPolicy(
            agent_id="full-ip-fail",
            agent_name="Full Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=10,
            ip_allowlist=["10.0.0.0/8"],
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="not in allowlist"):
            await enforce_policy("user1", "full-ip-fail", "github", ["repo"],
                               ip_address="192.168.1.1")

    @pytest.mark.asyncio
    async def test_time_denied_in_full_stack(self, db):
        policy = AgentPolicy(
            agent_id="full-time-fail",
            agent_name="Full Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=10,
            allowed_hours=[9, 10, 11],
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with patch("src.policy.check_time_window",
                   return_value="Access denied: not within allowed hours"):
            with pytest.raises(PolicyDenied, match="not within allowed hours"):
                await enforce_policy("user1", "full-time-fail", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_expired_in_full_stack(self, db):
        policy = AgentPolicy(
            agent_id="full-exp-fail",
            agent_name="Full Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=10,
            expires_at=time.time() - 1,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy("user1", "full-exp-fail", "github", ["repo"])


# ---------------------------------------------------------------------------
# 2. Scope enforcement edge cases
# ---------------------------------------------------------------------------

class TestScopeEnforcementEdgeCases:
    """Edge cases for scope checking in enforce_policy."""

    @pytest.mark.asyncio
    async def test_subset_scopes_pass(self, db):
        policy = AgentPolicy(
            agent_id="scope-sub",
            agent_name="Scope Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo", "read:user", "gist"]},
            rate_limit_per_minute=60,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        result = await enforce_policy("user1", "scope-sub", "github", ["repo"])
        assert result.agent_id == "scope-sub"

    @pytest.mark.asyncio
    async def test_exact_scopes_pass(self, db):
        policy = AgentPolicy(
            agent_id="scope-exact",
            agent_name="Scope Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        result = await enforce_policy("user1", "scope-exact", "github", ["repo"])
        assert result.agent_id == "scope-exact"

    @pytest.mark.asyncio
    async def test_single_excess_scope_denied(self, db):
        policy = AgentPolicy(
            agent_id="scope-1ex",
            agent_name="Scope Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="Scopes not permitted"):
            await enforce_policy("user1", "scope-1ex", "github", ["admin:org"])

    @pytest.mark.asyncio
    async def test_mixed_valid_and_invalid_scopes_denied(self, db):
        policy = AgentPolicy(
            agent_id="scope-mix",
            agent_name="Scope Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo", "read:user"]},
            rate_limit_per_minute=60,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="Scopes not permitted"):
            await enforce_policy("user1", "scope-mix", "github",
                               ["repo", "admin:org"])

    @pytest.mark.asyncio
    async def test_no_scopes_configured_allows_empty_request(self, db):
        policy = AgentPolicy(
            agent_id="scope-none",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": []},
            rate_limit_per_minute=60,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        result = await enforce_policy("user1", "scope-none", "github", [])
        assert result.agent_id == "scope-none"

    @pytest.mark.asyncio
    async def test_no_scopes_configured_denies_any_scope(self, db):
        policy = AgentPolicy(
            agent_id="scope-deny-any",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": []},
            rate_limit_per_minute=60,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="Scopes not permitted"):
            await enforce_policy("user1", "scope-deny-any", "github", ["repo"])


# ---------------------------------------------------------------------------
# 3. Service enforcement edge cases
# ---------------------------------------------------------------------------

class TestServiceEnforcementEdgeCases:
    """Edge cases for service authorization checking."""

    @pytest.mark.asyncio
    async def test_empty_services_denies_all(self, db):
        policy = AgentPolicy(
            agent_id="no-svc",
            agent_name="Bot",
            allowed_services=[],
            allowed_scopes={},
            rate_limit_per_minute=60,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="not authorized"):
            await enforce_policy("user1", "no-svc", "github", [])

    @pytest.mark.asyncio
    async def test_multiple_services_each_accessible(self, db):
        policy = AgentPolicy(
            agent_id="multi-svc",
            agent_name="Bot",
            allowed_services=["github", "slack", "google"],
            allowed_scopes={
                "github": ["repo"], "slack": ["chat:write"],
                "google": ["gmail.readonly"],
            },
            rate_limit_per_minute=60,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        for svc, scope in [("github", "repo"), ("slack", "chat:write"),
                           ("google", "gmail.readonly")]:
            result = await enforce_policy("user1", "multi-svc", svc, [scope])
            assert result.agent_id == "multi-svc"


# ---------------------------------------------------------------------------
# 4. Effective scopes comprehensive
# ---------------------------------------------------------------------------

class TestEffectiveScopesComprehensive:
    """Additional get_effective_scopes tests."""

    def test_empty_requested_and_empty_allowed(self):
        policy = AgentPolicy(agent_id="t", agent_name="t", allowed_scopes={})
        result = get_effective_scopes(policy, "github", [])
        assert result == []

    def test_duplicate_requested_scopes(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={"github": ["repo"]},
        )
        result = get_effective_scopes(policy, "github", ["repo", "repo", "repo"])
        assert result == ["repo"]

    def test_many_scopes_intersection(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={"github": ["repo", "read:user", "gist", "notifications"]},
        )
        result = get_effective_scopes(policy, "github",
                                      ["repo", "admin:org", "gist", "delete:repo"])
        assert set(result) == {"repo", "gist"}


# ---------------------------------------------------------------------------
# 5. Time window comprehensive
# ---------------------------------------------------------------------------

class TestTimeWindowComprehensive:
    """Additional time window tests for boundary conditions."""

    def test_hour_23_boundary(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_hours=[23],
        )
        assert check_time_window(policy,
            datetime(2026, 1, 1, 23, 0, tzinfo=timezone.utc)) is None
        assert check_time_window(policy,
            datetime(2026, 1, 1, 23, 59, tzinfo=timezone.utc)) is None
        assert check_time_window(policy,
            datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)) is not None

    def test_hour_0_boundary(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_hours=[0],
        )
        assert check_time_window(policy,
            datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)) is None
        assert check_time_window(policy,
            datetime(2026, 1, 1, 0, 59, tzinfo=timezone.utc)) is None
        assert check_time_window(policy,
            datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)) is not None

    def test_single_day_monday(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_days=[0],  # Monday only
        )
        # March 16, 2026 is a Monday
        assert check_time_window(policy,
            datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc)) is None
        # March 17, 2026 is a Tuesday
        assert check_time_window(policy,
            datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)) is not None

    def test_single_day_sunday(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_days=[6],  # Sunday only
        )
        # March 15, 2026 is a Sunday
        assert check_time_window(policy,
            datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)) is None
        # March 14, 2026 is a Saturday
        assert check_time_window(policy,
            datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)) is not None


# ---------------------------------------------------------------------------
# 6. IP allowlist comprehensive
# ---------------------------------------------------------------------------

class TestIPAllowlistComprehensive:
    """Additional IP allowlist edge cases."""

    def test_ipv6_exact_match(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["2001:db8::1"],
        )
        assert check_ip_allowlist(policy, "2001:db8::1") is None
        assert check_ip_allowlist(policy, "2001:db8::2") is not None

    def test_large_cidr_block(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["10.0.0.0/8"],
        )
        assert check_ip_allowlist(policy, "10.255.255.255") is None
        assert check_ip_allowlist(policy, "11.0.0.0") is not None

    def test_slash_24_cidr(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["192.168.1.0/24"],
        )
        assert check_ip_allowlist(policy, "192.168.1.0") is None
        assert check_ip_allowlist(policy, "192.168.1.255") is None
        assert check_ip_allowlist(policy, "192.168.2.0") is not None

    def test_multiple_valid_entries_first_match_wins(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["192.168.1.0/24", "10.0.0.0/8"],
        )
        # Match from first entry
        assert check_ip_allowlist(policy, "192.168.1.100") is None
        # Match from second entry
        assert check_ip_allowlist(policy, "10.5.5.5") is None

    def test_all_malformed_entries_deny(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["not-valid", "also-not-valid"],
        )
        result = check_ip_allowlist(policy, "10.0.0.1")
        assert result is not None
        assert "not in allowlist" in result


# ---------------------------------------------------------------------------
# 7. Step-up auth comprehensive
# ---------------------------------------------------------------------------

class TestStepUpComprehensive:
    """Additional step-up auth tests."""

    def test_multiple_services_require_step_up(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            requires_step_up=["github", "slack", "google"],
        )
        assert requires_step_up(policy, "github") is True
        assert requires_step_up(policy, "slack") is True
        assert requires_step_up(policy, "google") is True
        assert requires_step_up(policy, "linear") is False

    def test_empty_step_up_list(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            requires_step_up=[],
        )
        assert requires_step_up(policy, "github") is False
        assert requires_step_up(policy, "") is False


# ---------------------------------------------------------------------------
# 8. PolicyDenied exception attributes
# ---------------------------------------------------------------------------

class TestPolicyDeniedAttributes:
    """Verify PolicyDenied exception construction and attributes."""

    def test_full_construction(self):
        ex = PolicyDenied("reason", "agent-x", "github")
        assert str(ex) == "reason"
        assert ex.reason == "reason"
        assert ex.agent_id == "agent-x"
        assert ex.service == "github"

    def test_partial_construction(self):
        ex = PolicyDenied("reason", agent_id="a")
        assert ex.agent_id == "a"
        assert ex.service == ""

    def test_is_exception_subclass(self):
        assert issubclass(PolicyDenied, Exception)

    def test_can_be_caught_as_exception(self):
        try:
            raise PolicyDenied("test")
        except Exception as e:
            assert str(e) == "test"
