"""Policy engine edge cases — rate counter manipulation, enforcement order
verification with audit trail, IPv6 edge cases, scope boundary conditions,
and combined constraint denial chains."""

import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.database import AgentPolicy, create_agent_policy, get_audit_log, cleanup_rate_limit_events
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
# 1. Rate counter manipulation and edge cases
# ---------------------------------------------------------------------------

class TestRateCounterManipulation:
    """Test rate counter pruning and edge cases."""

    @pytest.mark.asyncio
    async def test_old_entries_pruned_on_check(self, db):
        """Entries older than 60s are pruned during enforcement check."""
        policy = AgentPolicy(
            agent_id="prune-bot",
            agent_name="Prune Bot",
            allowed_services=["svc"],
            allowed_scopes={"svc": ["r"]},
            rate_limit_per_minute=2,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)

        # Fill up the rate limit
        await enforce_policy("user1", "prune-bot", "svc", ["r"])
        await enforce_policy("user1", "prune-bot", "svc", ["r"])

        # Blocked
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy("user1", "prune-bot", "svc", ["r"])

        # Simulate all entries aging out
        key = "prune-bot:svc"
        _rate_counters[key] = [t - 120 for t in _rate_counters[key]]
        # Also clean persistent rate limit events
        await cleanup_rate_limit_events(max_age_seconds=0)

        # Now should succeed again
        result = await enforce_policy("user1", "prune-bot", "svc", ["r"])
        assert result.agent_id == "prune-bot"

    @pytest.mark.asyncio
    async def test_rate_counter_key_format(self, db):
        """Rate counter key is agent_id:service."""
        policy = AgentPolicy(
            agent_id="key-format",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=100,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        await enforce_policy("user1", "key-format", "github", ["repo"])
        assert "key-format:github" in _rate_counters

    @pytest.mark.asyncio
    async def test_rate_limit_per_service_isolation(self, db):
        """Rate limits are tracked independently per service for the same agent."""
        policy = AgentPolicy(
            agent_id="iso-rl",
            agent_name="Bot",
            allowed_services=["github", "slack"],
            allowed_scopes={"github": ["repo"], "slack": ["chat:write"]},
            rate_limit_per_minute=1,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)

        # Use up github rate limit
        await enforce_policy("user1", "iso-rl", "github", ["repo"])
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy("user1", "iso-rl", "github", ["repo"])

        # Slack should still work
        result = await enforce_policy("user1", "iso-rl", "slack", ["chat:write"])
        assert result.agent_id == "iso-rl"


# ---------------------------------------------------------------------------
# 2. Enforcement audit trail verification
# ---------------------------------------------------------------------------

class TestEnforcementAuditTrail:
    """Verify that policy enforcement creates proper audit entries."""

    @pytest.mark.asyncio
    async def test_denied_unknown_agent_logs_audit(self, db):
        """Denying an unknown agent creates an audit entry."""
        with pytest.raises(PolicyDenied):
            await enforce_policy("user1", "unknown", "github", ["repo"])
        entries = await get_audit_log("user1")
        assert len(entries) >= 1
        assert any(e.status == "denied" for e in entries)

    @pytest.mark.asyncio
    async def test_denied_disabled_agent_logs_audit(self, db):
        """Denying a disabled agent creates an audit entry."""
        await create_agent_policy(AgentPolicy(
            agent_id="disabled-audit",
            agent_name="Bot",
            allowed_services=["github"],
            created_by="user1",
            created_at=time.time(),
            is_active=False,
        ))
        with pytest.raises(PolicyDenied):
            await enforce_policy("user1", "disabled-audit", "github", [])
        entries = await get_audit_log("user1")
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 1
        assert "disabled" in denied[0].details

    @pytest.mark.asyncio
    async def test_denied_ownership_logs_audit(self, db):
        """Denying an ownership violation creates an audit entry."""
        await create_agent_policy(AgentPolicy(
            agent_id="own-audit",
            agent_name="Bot",
            allowed_services=["github"],
            created_by="owner",
            created_at=time.time(),
            is_active=True,
        ))
        with pytest.raises(PolicyDenied):
            await enforce_policy("intruder", "own-audit", "github", [])
        entries = await get_audit_log("intruder")
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 1
        assert "Ownership" in denied[0].details

    @pytest.mark.asyncio
    async def test_denied_service_logs_audit(self, db):
        """Denying an unauthorized service creates an audit entry."""
        await create_agent_policy(AgentPolicy(
            agent_id="svc-audit",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        ))
        with pytest.raises(PolicyDenied):
            await enforce_policy("user1", "svc-audit", "slack", [])
        entries = await get_audit_log("user1")
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 1

    @pytest.mark.asyncio
    async def test_denied_scopes_logs_audit(self, db):
        """Denying excess scopes creates an audit entry with scope info."""
        await create_agent_policy(AgentPolicy(
            agent_id="scope-audit",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        ))
        with pytest.raises(PolicyDenied):
            await enforce_policy("user1", "scope-audit", "github", ["admin:org"])
        entries = await get_audit_log("user1")
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 1
        assert "Excess scopes" in denied[0].details

    @pytest.mark.asyncio
    async def test_rate_limited_logs_audit(self, db):
        """Rate-limited requests create an audit entry."""
        await create_agent_policy(AgentPolicy(
            agent_id="rl-audit",
            agent_name="Bot",
            allowed_services=["svc"],
            allowed_scopes={"svc": ["r"]},
            rate_limit_per_minute=1,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        ))
        await enforce_policy("user1", "rl-audit", "svc", ["r"])
        with pytest.raises(PolicyDenied):
            await enforce_policy("user1", "rl-audit", "svc", ["r"])

        entries = await get_audit_log("user1")
        rl_entries = [e for e in entries if e.status == "rate_limited"]
        assert len(rl_entries) >= 1


# ---------------------------------------------------------------------------
# 3. IPv6 edge cases
# ---------------------------------------------------------------------------

class TestIPv6EdgeCases:
    """IPv6 specific edge cases for IP allowlist."""

    def test_ipv6_loopback(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["::1"],
        )
        assert check_ip_allowlist(policy, "::1") is None
        assert check_ip_allowlist(policy, "::2") is not None

    def test_ipv6_full_address(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["2001:0db8:85a3:0000:0000:8a2e:0370:7334"],
        )
        assert check_ip_allowlist(policy, "2001:db8:85a3::8a2e:370:7334") is None

    def test_ipv6_cidr_64(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["2001:db8::/64"],
        )
        assert check_ip_allowlist(policy, "2001:db8::1") is None
        assert check_ip_allowlist(policy, "2001:db8::ffff") is None
        assert check_ip_allowlist(policy, "2001:db9::1") is not None

    def test_ipv6_cidr_128(self):
        """A /128 CIDR matches exactly one address."""
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["2001:db8::1/128"],
        )
        assert check_ip_allowlist(policy, "2001:db8::1") is None
        assert check_ip_allowlist(policy, "2001:db8::2") is not None

    def test_mixed_ipv4_and_ipv6(self):
        """Allowlist with both IPv4 and IPv6 entries."""
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            ip_allowlist=["10.0.0.1", "2001:db8::1"],
        )
        assert check_ip_allowlist(policy, "10.0.0.1") is None
        assert check_ip_allowlist(policy, "2001:db8::1") is None
        assert check_ip_allowlist(policy, "10.0.0.2") is not None
        assert check_ip_allowlist(policy, "2001:db8::2") is not None


# ---------------------------------------------------------------------------
# 4. Scope boundary conditions
# ---------------------------------------------------------------------------

class TestScopeBoundaryConditions:
    """Boundary conditions for scope checks."""

    def test_single_scope_match(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={"svc": ["read"]},
        )
        result = get_effective_scopes(policy, "svc", ["read"])
        assert result == ["read"]

    def test_single_scope_no_match(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={"svc": ["read"]},
        )
        result = get_effective_scopes(policy, "svc", ["write"])
        assert result == []

    def test_many_allowed_few_requested(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={"svc": ["a", "b", "c", "d", "e", "f"]},
        )
        result = get_effective_scopes(policy, "svc", ["b"])
        assert result == ["b"]

    def test_few_allowed_many_requested(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={"svc": ["a"]},
        )
        result = get_effective_scopes(policy, "svc", ["a", "b", "c", "d", "e"])
        assert result == ["a"]

    def test_all_overlap(self):
        scopes = ["x", "y", "z"]
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={"svc": scopes},
        )
        result = get_effective_scopes(policy, "svc", scopes)
        assert set(result) == set(scopes)

    def test_no_overlap(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={"svc": ["a", "b"]},
        )
        result = get_effective_scopes(policy, "svc", ["c", "d"])
        assert result == []


# ---------------------------------------------------------------------------
# 5. Combined constraint denial chains
# ---------------------------------------------------------------------------

class TestCombinedConstraintDenialChains:
    """Test that the correct denial fires when multiple constraints fail."""

    @pytest.mark.asyncio
    async def test_inactive_wins_over_everything(self, db):
        """Inactive policy is denied even if all other constraints pass."""
        await create_agent_policy(AgentPolicy(
            agent_id="chain-inactive",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
            is_active=False,
            expires_at=time.time() - 1,  # Also expired
        ))
        with pytest.raises(PolicyDenied, match="disabled"):
            await enforce_policy("user1", "chain-inactive", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_expiration_wins_over_ip(self, db):
        """Expired policy is denied before IP check."""
        await create_agent_policy(AgentPolicy(
            agent_id="chain-expired",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
            is_active=True,
            expires_at=time.time() - 1,
            ip_allowlist=["10.0.0.1"],
        ))
        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy("user1", "chain-expired", "github", ["repo"],
                               ip_address="192.168.1.1")

    @pytest.mark.asyncio
    async def test_service_denial_wins_over_scope(self, db):
        """Unauthorized service is denied before scope check."""
        await create_agent_policy(AgentPolicy(
            agent_id="chain-svc",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        ))
        with pytest.raises(PolicyDenied, match="not authorized"):
            # "slack" is not allowed, and "admin" scope would also fail
            await enforce_policy("user1", "chain-svc", "slack", ["admin"])

    @pytest.mark.asyncio
    async def test_ownership_wins_over_expiration(self, db):
        """Ownership denial fires before expiration check."""
        await create_agent_policy(AgentPolicy(
            agent_id="chain-own",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="owner",
            created_at=time.time(),
            is_active=True,
            expires_at=time.time() - 1,
        ))
        with pytest.raises(PolicyDenied, match="do not own"):
            await enforce_policy("intruder", "chain-own", "github", ["repo"])


# ---------------------------------------------------------------------------
# 6. Time window with check_time_window defaulting to now
# ---------------------------------------------------------------------------

class TestTimeWindowDefaultNow:
    """Test check_time_window with default now (no explicit time)."""

    def test_all_hours_all_days_passes_with_default_now(self):
        """Policy allowing all hours and days passes with default now."""
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_hours=list(range(24)),
            allowed_days=list(range(7)),
        )
        # No explicit `now` parameter - uses default
        result = check_time_window(policy)
        assert result is None

    def test_empty_constraints_passes_with_default_now(self):
        """Empty time constraints always pass."""
        policy = AgentPolicy(agent_id="t", agent_name="t")
        result = check_time_window(policy)
        assert result is None

    def test_single_day_saturday(self):
        """Saturday-only policy denies on non-Saturday."""
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_days=[5],  # Saturday
        )
        # March 14, 2026 is a Saturday
        result = check_time_window(policy, datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc))
        assert result is None
        # March 16, 2026 is a Monday
        result = check_time_window(policy, datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc))
        assert result is not None


# ---------------------------------------------------------------------------
# 7. PolicyDenied exception additional tests
# ---------------------------------------------------------------------------

class TestPolicyDeniedAdditional:
    """Additional PolicyDenied exception tests."""

    def test_reason_attribute(self):
        ex = PolicyDenied("my reason", "agent-1", "github")
        assert ex.reason == "my reason"

    def test_str_representation(self):
        ex = PolicyDenied("test message")
        assert str(ex) == "test message"

    def test_args_tuple(self):
        ex = PolicyDenied("reason")
        assert ex.args == ("reason",)

    def test_with_all_empty_strings(self):
        ex = PolicyDenied("", "", "")
        assert ex.reason == ""
        assert ex.agent_id == ""
        assert ex.service == ""
        assert str(ex) == ""
