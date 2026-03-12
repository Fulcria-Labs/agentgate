"""
Comprehensive tests for policy IP allowlist and time window checks.

Tests cover:
- IPv4 and IPv6 address validation
- CIDR notation matching
- Mixed IP entries in allowlists
- Time window boundary conditions
- Day of week filtering
- Hour range filtering
- Combined time + day restrictions
- Edge cases: malformed IPs, empty allowlists, midnight crossings
"""

import pytest
from datetime import datetime, timezone
from src.policy import check_ip_allowlist, check_time_window, PolicyDenied, requires_step_up, get_effective_scopes
from src.database import AgentPolicy


def make_policy(**kwargs) -> AgentPolicy:
    """Create a test policy with defaults."""
    defaults = {
        "agent_id": "test-agent",
        "agent_name": "Test Agent",
        "allowed_services": ["github"],
        "allowed_scopes": {"github": ["read"]},
        "rate_limit_per_minute": 60,
        "requires_step_up": [],
        "created_by": "user-1",
        "created_at": 1000.0,
        "is_active": True,
        "allowed_hours": [],
        "allowed_days": [],
        "expires_at": 0.0,
        "ip_allowlist": [],
    }
    defaults.update(kwargs)
    return AgentPolicy(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# IP ALLOWLIST - IPv4
# ═══════════════════════════════════════════════════════════════════════════════

class TestIPAllowlistIPv4:
    def test_empty_allowlist_allows_all(self):
        policy = make_policy(ip_allowlist=[])
        assert check_ip_allowlist(policy, "1.2.3.4") is None

    def test_exact_match_allows(self):
        policy = make_policy(ip_allowlist=["192.168.1.1"])
        assert check_ip_allowlist(policy, "192.168.1.1") is None

    def test_exact_mismatch_denies(self):
        policy = make_policy(ip_allowlist=["192.168.1.1"])
        result = check_ip_allowlist(policy, "192.168.1.2")
        assert result is not None
        assert "not in allowlist" in result

    def test_cidr_24_allows_matching(self):
        policy = make_policy(ip_allowlist=["192.168.1.0/24"])
        assert check_ip_allowlist(policy, "192.168.1.100") is None
        assert check_ip_allowlist(policy, "192.168.1.255") is None

    def test_cidr_24_denies_outside(self):
        policy = make_policy(ip_allowlist=["192.168.1.0/24"])
        result = check_ip_allowlist(policy, "192.168.2.1")
        assert result is not None

    def test_cidr_16_allows_wide_range(self):
        policy = make_policy(ip_allowlist=["10.0.0.0/16"])
        assert check_ip_allowlist(policy, "10.0.255.255") is None
        assert check_ip_allowlist(policy, "10.0.0.1") is None

    def test_cidr_16_denies_outside(self):
        policy = make_policy(ip_allowlist=["10.0.0.0/16"])
        result = check_ip_allowlist(policy, "10.1.0.1")
        assert result is not None

    def test_cidr_32_is_exact_match(self):
        policy = make_policy(ip_allowlist=["10.0.0.1/32"])
        assert check_ip_allowlist(policy, "10.0.0.1") is None
        result = check_ip_allowlist(policy, "10.0.0.2")
        assert result is not None

    def test_multiple_entries_any_match(self):
        policy = make_policy(ip_allowlist=["192.168.1.1", "10.0.0.0/24", "172.16.0.5"])
        assert check_ip_allowlist(policy, "192.168.1.1") is None
        assert check_ip_allowlist(policy, "10.0.0.50") is None
        assert check_ip_allowlist(policy, "172.16.0.5") is None

    def test_multiple_entries_none_match(self):
        policy = make_policy(ip_allowlist=["192.168.1.1", "10.0.0.0/24"])
        result = check_ip_allowlist(policy, "8.8.8.8")
        assert result is not None

    def test_localhost_ipv4(self):
        policy = make_policy(ip_allowlist=["127.0.0.1"])
        assert check_ip_allowlist(policy, "127.0.0.1") is None

    def test_all_zeros_address(self):
        policy = make_policy(ip_allowlist=["0.0.0.0/0"])
        assert check_ip_allowlist(policy, "1.2.3.4") is None  # /0 matches everything

    def test_broadcast_address(self):
        policy = make_policy(ip_allowlist=["255.255.255.255"])
        assert check_ip_allowlist(policy, "255.255.255.255") is None


# ═══════════════════════════════════════════════════════════════════════════════
# IP ALLOWLIST - IPv6
# ═══════════════════════════════════════════════════════════════════════════════

class TestIPAllowlistIPv6:
    def test_ipv6_exact_match(self):
        policy = make_policy(ip_allowlist=["::1"])
        assert check_ip_allowlist(policy, "::1") is None

    def test_ipv6_full_notation_match(self):
        policy = make_policy(ip_allowlist=["2001:db8::1"])
        assert check_ip_allowlist(policy, "2001:db8::1") is None

    def test_ipv6_cidr_match(self):
        policy = make_policy(ip_allowlist=["2001:db8::/32"])
        assert check_ip_allowlist(policy, "2001:db8::1") is None
        assert check_ip_allowlist(policy, "2001:db8:ffff::1") is None

    def test_ipv6_cidr_denies_outside(self):
        policy = make_policy(ip_allowlist=["2001:db8::/32"])
        result = check_ip_allowlist(policy, "2001:db9::1")
        assert result is not None

    def test_ipv6_mismatch(self):
        policy = make_policy(ip_allowlist=["2001:db8::1"])
        result = check_ip_allowlist(policy, "2001:db8::2")
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# IP ALLOWLIST - EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestIPAllowlistEdgeCases:
    def test_invalid_ip_address(self):
        policy = make_policy(ip_allowlist=["192.168.1.1"])
        result = check_ip_allowlist(policy, "not-an-ip")
        assert result is not None
        assert "invalid IP" in result

    def test_empty_string_ip(self):
        policy = make_policy(ip_allowlist=["192.168.1.1"])
        result = check_ip_allowlist(policy, "")
        assert result is not None

    def test_malformed_allowlist_entry_skipped(self):
        policy = make_policy(ip_allowlist=["bad-entry", "192.168.1.1"])
        assert check_ip_allowlist(policy, "192.168.1.1") is None

    def test_all_malformed_entries_deny(self):
        policy = make_policy(ip_allowlist=["bad", "also-bad", "nope/24"])
        result = check_ip_allowlist(policy, "1.2.3.4")
        assert result is not None

    def test_mixed_ipv4_ipv6_allowlist(self):
        policy = make_policy(ip_allowlist=["192.168.1.1", "::1"])
        assert check_ip_allowlist(policy, "192.168.1.1") is None
        assert check_ip_allowlist(policy, "::1") is None

    def test_denial_message_includes_ip(self):
        policy = make_policy(ip_allowlist=["10.0.0.1"])
        result = check_ip_allowlist(policy, "8.8.4.4")
        assert "8.8.4.4" in result


# ═══════════════════════════════════════════════════════════════════════════════
# TIME WINDOW - HOURS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimeWindowHours:
    def test_empty_hours_allows_all(self):
        policy = make_policy(allowed_hours=[])
        now = datetime(2026, 3, 12, 14, 30, tzinfo=timezone.utc)
        assert check_time_window(policy, now) is None

    def test_allowed_hour_passes(self):
        policy = make_policy(allowed_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17])
        now = datetime(2026, 3, 12, 14, 30, tzinfo=timezone.utc)
        assert check_time_window(policy, now) is None

    def test_disallowed_hour_denied(self):
        policy = make_policy(allowed_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17])
        now = datetime(2026, 3, 12, 22, 30, tzinfo=timezone.utc)
        result = check_time_window(policy, now)
        assert result is not None
        assert "allowed hours" in result

    def test_midnight_hour_zero(self):
        policy = make_policy(allowed_hours=[0, 1, 2, 3])
        now = datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc)
        assert check_time_window(policy, now) is None

    def test_single_hour_window(self):
        policy = make_policy(allowed_hours=[12])
        now = datetime(2026, 3, 12, 12, 59, tzinfo=timezone.utc)
        assert check_time_window(policy, now) is None
        now2 = datetime(2026, 3, 12, 13, 0, tzinfo=timezone.utc)
        result = check_time_window(policy, now2)
        assert result is not None

    def test_last_hour_23(self):
        policy = make_policy(allowed_hours=[23])
        now = datetime(2026, 3, 12, 23, 59, tzinfo=timezone.utc)
        assert check_time_window(policy, now) is None

    def test_hour_boundary_start(self):
        policy = make_policy(allowed_hours=[9])
        now = datetime(2026, 3, 12, 9, 0, 0, tzinfo=timezone.utc)
        assert check_time_window(policy, now) is None

    def test_hour_boundary_end(self):
        policy = make_policy(allowed_hours=[9])
        now = datetime(2026, 3, 12, 9, 59, 59, tzinfo=timezone.utc)
        assert check_time_window(policy, now) is None

    def test_denial_message_includes_hour_range(self):
        policy = make_policy(allowed_hours=[9, 10, 17])
        now = datetime(2026, 3, 12, 22, 0, tzinfo=timezone.utc)
        result = check_time_window(policy, now)
        assert "09:00" in result
        assert "17:59" in result


# ═══════════════════════════════════════════════════════════════════════════════
# TIME WINDOW - DAYS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimeWindowDays:
    def test_empty_days_allows_all(self):
        policy = make_policy(allowed_days=[])
        now = datetime(2026, 3, 12, 14, 30, tzinfo=timezone.utc)  # Thursday
        assert check_time_window(policy, now) is None

    def test_weekday_policy_allows_weekday(self):
        policy = make_policy(allowed_days=[0, 1, 2, 3, 4])  # Mon-Fri
        now = datetime(2026, 3, 12, 14, 30, tzinfo=timezone.utc)  # Thursday = 3
        assert check_time_window(policy, now) is None

    def test_weekday_policy_denies_weekend(self):
        policy = make_policy(allowed_days=[0, 1, 2, 3, 4])  # Mon-Fri
        now = datetime(2026, 3, 14, 14, 30, tzinfo=timezone.utc)  # Saturday = 5
        result = check_time_window(policy, now)
        assert result is not None
        assert "allowed days" in result

    def test_weekend_only_policy(self):
        policy = make_policy(allowed_days=[5, 6])  # Sat-Sun
        now = datetime(2026, 3, 14, 14, 30, tzinfo=timezone.utc)  # Saturday = 5
        assert check_time_window(policy, now) is None

    def test_single_day_allowed(self):
        policy = make_policy(allowed_days=[3])  # Thursday only
        now = datetime(2026, 3, 12, 14, 30, tzinfo=timezone.utc)  # Thursday
        assert check_time_window(policy, now) is None

    def test_single_day_denied(self):
        policy = make_policy(allowed_days=[3])  # Thursday only
        now = datetime(2026, 3, 13, 14, 30, tzinfo=timezone.utc)  # Friday = 4
        result = check_time_window(policy, now)
        assert result is not None

    def test_denial_message_includes_day_names(self):
        policy = make_policy(allowed_days=[0, 2, 4])  # Mon, Wed, Fri
        now = datetime(2026, 3, 14, 14, 30, tzinfo=timezone.utc)  # Saturday
        result = check_time_window(policy, now)
        assert "Mon" in result
        assert "Wed" in result
        assert "Fri" in result


# ═══════════════════════════════════════════════════════════════════════════════
# TIME WINDOW - COMBINED HOURS + DAYS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimeWindowCombined:
    def test_right_day_right_hour_allowed(self):
        policy = make_policy(
            allowed_days=[0, 1, 2, 3, 4],  # Mon-Fri
            allowed_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],  # 9am-5pm
        )
        now = datetime(2026, 3, 12, 14, 30, tzinfo=timezone.utc)  # Thursday 2:30pm
        assert check_time_window(policy, now) is None

    def test_right_day_wrong_hour_denied(self):
        policy = make_policy(
            allowed_days=[0, 1, 2, 3, 4],
            allowed_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
        )
        now = datetime(2026, 3, 12, 22, 0, tzinfo=timezone.utc)  # Thursday 10pm
        result = check_time_window(policy, now)
        assert result is not None

    def test_wrong_day_right_hour_denied(self):
        policy = make_policy(
            allowed_days=[0, 1, 2, 3, 4],
            allowed_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
        )
        now = datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)  # Saturday noon
        result = check_time_window(policy, now)
        assert result is not None

    def test_wrong_day_wrong_hour_denied(self):
        policy = make_policy(
            allowed_days=[0, 1, 2, 3, 4],
            allowed_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
        )
        now = datetime(2026, 3, 14, 22, 0, tzinfo=timezone.utc)  # Saturday 10pm
        result = check_time_window(policy, now)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# PolicyDenied EXCEPTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestPolicyDeniedException:
    def test_basic_creation(self):
        exc = PolicyDenied("test reason")
        assert str(exc) == "test reason"
        assert exc.reason == "test reason"
        assert exc.agent_id == ""
        assert exc.service == ""

    def test_with_agent_and_service(self):
        exc = PolicyDenied("denied", agent_id="a1", service="github")
        assert exc.reason == "denied"
        assert exc.agent_id == "a1"
        assert exc.service == "github"

    def test_inherits_from_exception(self):
        exc = PolicyDenied("test")
        assert isinstance(exc, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(PolicyDenied) as exc_info:
            raise PolicyDenied("access denied", agent_id="x", service="s")
        assert exc_info.value.reason == "access denied"
        assert exc_info.value.agent_id == "x"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRequiresStepUp:
    def test_returns_true_for_listed_service(self):
        policy = make_policy(requires_step_up=["github", "slack"])
        assert requires_step_up(policy, "github") is True
        assert requires_step_up(policy, "slack") is True

    def test_returns_false_for_unlisted_service(self):
        policy = make_policy(requires_step_up=["github"])
        assert requires_step_up(policy, "slack") is False

    def test_returns_false_when_empty(self):
        policy = make_policy(requires_step_up=[])
        assert requires_step_up(policy, "github") is False


class TestGetEffectiveScopes:
    def test_returns_intersection(self):
        policy = make_policy(allowed_scopes={"github": ["read", "write", "admin"]})
        result = get_effective_scopes(policy, "github", ["read", "write"])
        assert set(result) == {"read", "write"}

    def test_filters_out_disallowed(self):
        policy = make_policy(allowed_scopes={"github": ["read"]})
        result = get_effective_scopes(policy, "github", ["read", "write", "admin"])
        assert result == ["read"]

    def test_empty_when_no_overlap(self):
        policy = make_policy(allowed_scopes={"github": ["read"]})
        result = get_effective_scopes(policy, "github", ["write", "admin"])
        assert result == []

    def test_empty_when_service_not_in_scopes(self):
        policy = make_policy(allowed_scopes={})
        result = get_effective_scopes(policy, "unknown", ["read"])
        assert result == []

    def test_empty_requested_returns_empty(self):
        policy = make_policy(allowed_scopes={"github": ["read", "write"]})
        result = get_effective_scopes(policy, "github", [])
        assert result == []

    def test_all_requested_allowed(self):
        policy = make_policy(allowed_scopes={"github": ["read", "write", "admin"]})
        result = get_effective_scopes(policy, "github", ["read", "write", "admin"])
        assert set(result) == {"read", "write", "admin"}


# ═══════════════════════════════════════════════════════════════════════════════
# AgentPolicy DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentPolicyDataclass:
    def test_default_values(self):
        policy = AgentPolicy(agent_id="a1", agent_name="test")
        assert policy.allowed_services == []
        assert policy.allowed_scopes == {}
        assert policy.rate_limit_per_minute == 60
        assert policy.requires_step_up == []
        assert policy.created_by == ""
        assert policy.is_active is True
        assert policy.allowed_hours == []
        assert policy.allowed_days == []
        assert policy.expires_at == 0.0
        assert policy.ip_allowlist == []

    def test_custom_values(self):
        policy = AgentPolicy(
            agent_id="a2",
            agent_name="Custom",
            allowed_services=["github"],
            rate_limit_per_minute=10,
            is_active=False,
            expires_at=9999.0,
        )
        assert policy.agent_id == "a2"
        assert policy.rate_limit_per_minute == 10
        assert policy.is_active is False
        assert policy.expires_at == 9999.0

    def test_mutable_default_lists_are_independent(self):
        p1 = AgentPolicy(agent_id="1", agent_name="1")
        p2 = AgentPolicy(agent_id="2", agent_name="2")
        p1.allowed_services.append("github")
        assert "github" not in p2.allowed_services
