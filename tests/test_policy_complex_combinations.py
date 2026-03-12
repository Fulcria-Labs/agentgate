"""Policy engine complex combinations -- deep edge cases testing interactions
between time windows, IP ranges, rate limits, expiration, ownership, and
service/scope constraints when multiple constraints combine simultaneously."""

import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.database import AgentPolicy, create_agent_policy, get_audit_log
from src.policy import (
    PolicyDenied,
    _rate_counters,
    check_ip_allowlist,
    check_time_window,
    enforce_policy,
    get_effective_scopes,
)


@pytest.fixture(autouse=True)
def clear_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


# ---------------------------------------------------------------------------
# 1. Complex time window + IP combinations
# ---------------------------------------------------------------------------

class TestTimeWindowIPCombinations:
    """Test interactions between time windows and IP allowlists."""

    @pytest.mark.asyncio
    async def test_both_time_and_ip_pass(self, db):
        """When both time window and IP are satisfied, enforcement passes."""
        policy = AgentPolicy(
            agent_id="combo-ok",
            agent_name="Combo Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            allowed_hours=list(range(24)),
            allowed_days=list(range(7)),
            ip_allowlist=["10.0.0.0/8"],
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        result = await enforce_policy("user1", "combo-ok", "github", ["repo"],
                                     ip_address="10.0.0.5")
        assert result.agent_id == "combo-ok"

    @pytest.mark.asyncio
    async def test_time_passes_ip_fails(self, db):
        """Time window passes but IP fails -- IP denial is reported."""
        policy = AgentPolicy(
            agent_id="combo-ip-fail",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            ip_allowlist=["10.0.0.1"],
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="not in allowlist"):
            await enforce_policy("user1", "combo-ip-fail", "github", ["repo"],
                               ip_address="192.168.1.1")

    @pytest.mark.asyncio
    async def test_ip_passes_time_fails(self, db):
        """IP passes but time window fails -- time denial is reported."""
        policy = AgentPolicy(
            agent_id="combo-time-fail",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            allowed_hours=[9, 10, 11],
            ip_allowlist=["10.0.0.0/8"],
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with patch("src.policy.check_time_window",
                   return_value="Access denied: not within allowed hours"):
            with pytest.raises(PolicyDenied, match="not within allowed hours"):
                await enforce_policy("user1", "combo-time-fail", "github", ["repo"],
                                   ip_address="10.0.0.5")

    @pytest.mark.asyncio
    async def test_both_time_and_ip_fail_time_checked_first(self, db):
        """When both time and IP fail, time is checked before IP (step 4 before 5)."""
        policy = AgentPolicy(
            agent_id="combo-both-fail",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            allowed_hours=[9],
            ip_allowlist=["10.0.0.1"],
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with patch("src.policy.check_time_window",
                   return_value="Access denied: not within allowed hours"):
            with pytest.raises(PolicyDenied, match="not within allowed hours"):
                await enforce_policy("user1", "combo-both-fail", "github", ["repo"],
                                   ip_address="192.168.1.1")


# ---------------------------------------------------------------------------
# 2. Rate limit + scope + service combinations
# ---------------------------------------------------------------------------

class TestRateLimitScopeServiceCombinations:
    """Complex interactions between rate limits, scopes, and services."""

    @pytest.mark.asyncio
    async def test_rate_limit_not_consumed_on_scope_denial(self, db):
        """A scope denial should not consume a rate limit slot
        (scopes are checked before rate limit in enforcement order)."""
        policy = AgentPolicy(
            agent_id="rl-scope",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=1,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        # Scope denial first
        with pytest.raises(PolicyDenied, match="Scopes not permitted"):
            await enforce_policy("user1", "rl-scope", "github", ["admin:org"])
        # Rate limit should still be available
        result = await enforce_policy("user1", "rl-scope", "github", ["repo"])
        assert result.agent_id == "rl-scope"

    @pytest.mark.asyncio
    async def test_rate_limit_not_consumed_on_service_denial(self, db):
        """A service denial should not consume a rate limit slot."""
        policy = AgentPolicy(
            agent_id="rl-svc",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=1,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="not authorized"):
            await enforce_policy("user1", "rl-svc", "slack", ["chat:write"])
        result = await enforce_policy("user1", "rl-svc", "github", ["repo"])
        assert result.agent_id == "rl-svc"

    @pytest.mark.asyncio
    async def test_rate_limit_consumed_only_on_success(self, db):
        """Rate limit counter increments only on successful policy checks."""
        policy = AgentPolicy(
            agent_id="rl-success-only",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=2,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        # Fail with service denial (doesn't consume RL)
        with pytest.raises(PolicyDenied):
            await enforce_policy("user1", "rl-success-only", "slack", [])
        # Succeed twice (consumes both RL slots)
        await enforce_policy("user1", "rl-success-only", "github", ["repo"])
        await enforce_policy("user1", "rl-success-only", "github", ["repo"])
        # Third should be rate limited
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy("user1", "rl-success-only", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_multi_service_rate_limits_independent(self, db):
        """Rate limits for different services on the same agent are independent."""
        policy = AgentPolicy(
            agent_id="rl-multi-svc",
            agent_name="Bot",
            allowed_services=["github", "slack"],
            allowed_scopes={"github": ["repo"], "slack": ["chat:write"]},
            rate_limit_per_minute=2,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        # Exhaust github rate limit
        await enforce_policy("user1", "rl-multi-svc", "github", ["repo"])
        await enforce_policy("user1", "rl-multi-svc", "github", ["repo"])
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy("user1", "rl-multi-svc", "github", ["repo"])
        # Slack still has its own rate limit
        result = await enforce_policy("user1", "rl-multi-svc", "slack", ["chat:write"])
        assert result.agent_id == "rl-multi-svc"


# ---------------------------------------------------------------------------
# 3. Expiration + time window + IP triple constraint
# ---------------------------------------------------------------------------

class TestTripleConstraint:
    """Three constraints active simultaneously."""

    @pytest.mark.asyncio
    async def test_all_three_pass(self, db):
        """Expiration, time window, and IP all pass."""
        policy = AgentPolicy(
            agent_id="triple-ok",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            expires_at=time.time() + 86400,
            ip_allowlist=["10.0.0.0/8"],
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        result = await enforce_policy("user1", "triple-ok", "github", ["repo"],
                                     ip_address="10.0.0.1")
        assert result.agent_id == "triple-ok"

    @pytest.mark.asyncio
    async def test_expired_blocks_even_with_valid_ip_and_time(self, db):
        """Expiration denial fires before time and IP checks."""
        policy = AgentPolicy(
            agent_id="triple-exp",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            expires_at=time.time() - 1,
            ip_allowlist=["10.0.0.0/8"],
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy("user1", "triple-exp", "github", ["repo"],
                               ip_address="10.0.0.1")


# ---------------------------------------------------------------------------
# 4. Non-contiguous time windows
# ---------------------------------------------------------------------------

class TestNonContiguousTimeWindows:
    """Time windows with gaps (e.g., morning and evening but not afternoon)."""

    def test_split_hour_window_morning_allowed(self):
        """Morning access in a split window (8-12, 18-22)."""
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_hours=[8, 9, 10, 11, 18, 19, 20, 21],
        )
        now = datetime(2026, 3, 12, 10, 30, tzinfo=timezone.utc)
        assert check_time_window(policy, now) is None

    def test_split_hour_window_evening_allowed(self):
        """Evening access in a split window."""
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_hours=[8, 9, 10, 11, 18, 19, 20, 21],
        )
        now = datetime(2026, 3, 12, 19, 30, tzinfo=timezone.utc)
        assert check_time_window(policy, now) is None

    def test_split_hour_window_gap_denied(self):
        """Access during the gap period is denied."""
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_hours=[8, 9, 10, 11, 18, 19, 20, 21],
        )
        now = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        result = check_time_window(policy, now)
        assert result is not None
        assert "not within allowed hours" in result

    def test_alternate_days_mon_wed_fri(self):
        """Access only on Mon/Wed/Fri."""
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_days=[0, 2, 4],  # Mon, Wed, Fri
        )
        # Wednesday
        assert check_time_window(policy,
            datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)) is None
        # Thursday - denied
        result = check_time_window(policy,
            datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc))
        assert result is not None


# ---------------------------------------------------------------------------
# 5. Narrow CIDR ranges
# ---------------------------------------------------------------------------

class TestNarrowCIDRRanges:
    """Test narrow CIDR ranges (/30, /31, /28)."""

    def test_slash_30_has_4_addresses(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["192.168.1.0/30"],
        )
        assert check_ip_allowlist(policy, "192.168.1.0") is None
        assert check_ip_allowlist(policy, "192.168.1.1") is None
        assert check_ip_allowlist(policy, "192.168.1.2") is None
        assert check_ip_allowlist(policy, "192.168.1.3") is None
        assert check_ip_allowlist(policy, "192.168.1.4") is not None

    def test_slash_28_has_16_addresses(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["172.16.0.0/28"],
        )
        assert check_ip_allowlist(policy, "172.16.0.0") is None
        assert check_ip_allowlist(policy, "172.16.0.15") is None
        assert check_ip_allowlist(policy, "172.16.0.16") is not None

    def test_slash_31_has_2_addresses(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["10.0.0.0/31"],
        )
        assert check_ip_allowlist(policy, "10.0.0.0") is None
        assert check_ip_allowlist(policy, "10.0.0.1") is None
        assert check_ip_allowlist(policy, "10.0.0.2") is not None


# ---------------------------------------------------------------------------
# 6. Overlapping CIDR ranges in allowlist
# ---------------------------------------------------------------------------

class TestOverlappingCIDR:
    """Allowlist with overlapping CIDR ranges."""

    def test_overlapping_ranges_both_match(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["10.0.0.0/8", "10.0.0.0/24"],
        )
        assert check_ip_allowlist(policy, "10.0.0.5") is None

    def test_superset_and_subset_ranges(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["10.0.0.0/16", "10.0.1.0/24"],
        )
        # Matches superset
        assert check_ip_allowlist(policy, "10.0.0.5") is None
        # Matches both
        assert check_ip_allowlist(policy, "10.0.1.5") is None
        # Outside both
        assert check_ip_allowlist(policy, "10.1.0.5") is not None


# ---------------------------------------------------------------------------
# 7. Effective scopes with multiple services
# ---------------------------------------------------------------------------

class TestEffectiveScopesMultiService:
    """Effective scope computation across multiple services."""

    def test_scopes_dont_bleed_across_services(self):
        """Scopes from one service don't appear in another."""
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={
                "github": ["repo", "read:user"],
                "slack": ["channels:read", "chat:write"],
            },
        )
        # Requesting github scopes against slack should yield nothing
        result = get_effective_scopes(policy, "slack", ["repo", "read:user"])
        assert result == []

    def test_intersections_are_independent(self):
        """Each service scope intersection is computed independently."""
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={
                "github": ["repo", "gist", "read:user"],
                "slack": ["channels:read", "chat:write"],
            },
        )
        gh_result = get_effective_scopes(policy, "github",
                                          ["repo", "channels:read", "gist"])
        sl_result = get_effective_scopes(policy, "slack",
                                          ["repo", "channels:read", "chat:write"])
        assert set(gh_result) == {"repo", "gist"}
        assert set(sl_result) == {"channels:read", "chat:write"}


# ---------------------------------------------------------------------------
# 8. Audit trail completeness for combined failures
# ---------------------------------------------------------------------------

class TestAuditTrailCombinedFailures:
    """Verify audit entries contain correct details for combined-constraint failures."""

    @pytest.mark.asyncio
    async def test_ip_denial_audit_records_ip(self, db):
        """IP denial audit entry records the denied IP address."""
        policy = AgentPolicy(
            agent_id="audit-ip-combo",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            ip_allowlist=["10.0.0.0/8"],
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied):
            await enforce_policy("user1", "audit-ip-combo", "github", ["repo"],
                               ip_address="192.168.1.99")
        entries = await get_audit_log("user1")
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 1
        assert denied[0].ip_address == "192.168.1.99"

    @pytest.mark.asyncio
    async def test_expiration_denial_records_details(self, db):
        """Expiration denial audit entry mentions expired."""
        policy = AgentPolicy(
            agent_id="audit-exp-combo",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            expires_at=time.time() - 3600,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied):
            await enforce_policy("user1", "audit-exp-combo", "github", ["repo"])
        entries = await get_audit_log("user1")
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 1
        assert "expired" in denied[0].details.lower()

    @pytest.mark.asyncio
    async def test_multiple_denials_each_logged(self, db):
        """Multiple sequential denials each produce their own audit entry."""
        policy = AgentPolicy(
            agent_id="audit-multi-deny",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        # Service denial
        with pytest.raises(PolicyDenied):
            await enforce_policy("user1", "audit-multi-deny", "slack", [])
        # Scope denial
        with pytest.raises(PolicyDenied):
            await enforce_policy("user1", "audit-multi-deny", "github", ["admin:org"])
        entries = await get_audit_log("user1")
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 2
