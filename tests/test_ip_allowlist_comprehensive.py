"""Comprehensive IP allowlist tests — IPv4/IPv6 edge cases, CIDR boundaries,
malformed entries, mixed configurations, and integration with policy enforcement."""

import time

import pytest

from src.database import AgentPolicy, create_agent_policy
from src.policy import (
    PolicyDenied,
    _rate_counters,
    check_ip_allowlist,
    enforce_policy,
)


@pytest.fixture(autouse=True)
def clear_rate_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


def _policy_with_ips(ips):
    return AgentPolicy(agent_id="t", agent_name="t", ip_allowlist=ips)


class TestIPv4ExactMatch:
    """Exact IPv4 address matching."""

    def test_exact_match(self):
        p = _policy_with_ips(["192.168.1.1"])
        assert check_ip_allowlist(p, "192.168.1.1") is None

    def test_no_match(self):
        p = _policy_with_ips(["192.168.1.1"])
        result = check_ip_allowlist(p, "192.168.1.2")
        assert "not in allowlist" in result

    def test_multiple_exact_ips(self):
        p = _policy_with_ips(["10.0.0.1", "10.0.0.2", "10.0.0.3"])
        assert check_ip_allowlist(p, "10.0.0.2") is None

    def test_loopback(self):
        p = _policy_with_ips(["127.0.0.1"])
        assert check_ip_allowlist(p, "127.0.0.1") is None

    def test_loopback_denied_when_not_listed(self):
        p = _policy_with_ips(["10.0.0.1"])
        result = check_ip_allowlist(p, "127.0.0.1")
        assert "not in allowlist" in result

    def test_all_zeros(self):
        p = _policy_with_ips(["0.0.0.0"])
        assert check_ip_allowlist(p, "0.0.0.0") is None

    def test_all_255(self):
        p = _policy_with_ips(["255.255.255.255"])
        assert check_ip_allowlist(p, "255.255.255.255") is None


class TestIPv4CIDR:
    """IPv4 CIDR range matching."""

    def test_slash_24_first_address(self):
        p = _policy_with_ips(["192.168.1.0/24"])
        assert check_ip_allowlist(p, "192.168.1.0") is None

    def test_slash_24_last_address(self):
        p = _policy_with_ips(["192.168.1.0/24"])
        assert check_ip_allowlist(p, "192.168.1.255") is None

    def test_slash_24_mid_address(self):
        p = _policy_with_ips(["192.168.1.0/24"])
        assert check_ip_allowlist(p, "192.168.1.128") is None

    def test_slash_24_outside(self):
        p = _policy_with_ips(["192.168.1.0/24"])
        result = check_ip_allowlist(p, "192.168.2.1")
        assert "not in allowlist" in result

    def test_slash_16(self):
        p = _policy_with_ips(["172.16.0.0/16"])
        assert check_ip_allowlist(p, "172.16.255.255") is None

    def test_slash_16_outside(self):
        p = _policy_with_ips(["172.16.0.0/16"])
        result = check_ip_allowlist(p, "172.17.0.1")
        assert "not in allowlist" in result

    def test_slash_8(self):
        p = _policy_with_ips(["10.0.0.0/8"])
        assert check_ip_allowlist(p, "10.255.255.254") is None

    def test_slash_32_single_host(self):
        p = _policy_with_ips(["10.0.0.5/32"])
        assert check_ip_allowlist(p, "10.0.0.5") is None
        result = check_ip_allowlist(p, "10.0.0.6")
        assert "not in allowlist" in result

    def test_slash_0_matches_all(self):
        p = _policy_with_ips(["0.0.0.0/0"])
        assert check_ip_allowlist(p, "1.2.3.4") is None
        assert check_ip_allowlist(p, "255.255.255.255") is None


class TestIPv6:
    """IPv6 address matching."""

    def test_ipv6_exact_match(self):
        p = _policy_with_ips(["::1"])
        assert check_ip_allowlist(p, "::1") is None

    def test_ipv6_full_notation(self):
        p = _policy_with_ips(["2001:0db8:0000:0000:0000:0000:0000:0001"])
        assert check_ip_allowlist(p, "2001:db8::1") is None

    def test_ipv6_cidr(self):
        p = _policy_with_ips(["2001:db8::/32"])
        assert check_ip_allowlist(p, "2001:db8::ffff") is None

    def test_ipv6_cidr_outside(self):
        p = _policy_with_ips(["2001:db8::/32"])
        result = check_ip_allowlist(p, "2001:db9::1")
        assert "not in allowlist" in result

    def test_ipv6_loopback(self):
        p = _policy_with_ips(["::1"])
        assert check_ip_allowlist(p, "::1") is None

    def test_ipv6_slash_128_single_host(self):
        p = _policy_with_ips(["fe80::1/128"])
        assert check_ip_allowlist(p, "fe80::1") is None
        result = check_ip_allowlist(p, "fe80::2")
        assert "not in allowlist" in result


class TestMixedConfig:
    """Mixed IPv4/IPv6 and exact/CIDR configurations."""

    def test_ipv4_and_ipv6_mixed(self):
        p = _policy_with_ips(["10.0.0.1", "::1"])
        assert check_ip_allowlist(p, "10.0.0.1") is None
        assert check_ip_allowlist(p, "::1") is None

    def test_exact_and_cidr_mixed(self):
        p = _policy_with_ips(["192.168.1.100", "10.0.0.0/24"])
        assert check_ip_allowlist(p, "192.168.1.100") is None
        assert check_ip_allowlist(p, "10.0.0.55") is None

    def test_multiple_cidrs(self):
        p = _policy_with_ips(["10.0.0.0/24", "172.16.0.0/16"])
        assert check_ip_allowlist(p, "10.0.0.5") is None
        assert check_ip_allowlist(p, "172.16.5.5") is None

    def test_overlapping_cidrs(self):
        p = _policy_with_ips(["10.0.0.0/8", "10.0.0.0/24"])
        assert check_ip_allowlist(p, "10.0.0.5") is None
        assert check_ip_allowlist(p, "10.1.0.5") is None


class TestMalformedEntries:
    """Malformed IP entries in the allowlist."""

    def test_invalid_requesting_ip(self):
        p = _policy_with_ips(["10.0.0.1"])
        result = check_ip_allowlist(p, "not-an-ip")
        assert "invalid IP" in result

    def test_empty_requesting_ip(self):
        p = _policy_with_ips(["10.0.0.1"])
        result = check_ip_allowlist(p, "")
        assert result is not None

    def test_malformed_allowlist_entry_skipped(self):
        """Malformed entries in allowlist are skipped, valid ones still match."""
        p = _policy_with_ips(["garbage", "10.0.0.1"])
        assert check_ip_allowlist(p, "10.0.0.1") is None

    def test_all_malformed_entries_deny(self):
        """If all allowlist entries are malformed, IP is denied."""
        p = _policy_with_ips(["garbage1", "garbage2"])
        result = check_ip_allowlist(p, "10.0.0.1")
        assert "not in allowlist" in result

    def test_ipv4_requesting_against_ipv6_only(self):
        p = _policy_with_ips(["::1", "2001:db8::/32"])
        result = check_ip_allowlist(p, "10.0.0.1")
        assert "not in allowlist" in result


class TestEmptyAllowlist:
    """Empty allowlist behavior."""

    def test_empty_list_allows_all(self):
        p = _policy_with_ips([])
        assert check_ip_allowlist(p, "1.2.3.4") is None

    def test_empty_allows_ipv6(self):
        p = _policy_with_ips([])
        assert check_ip_allowlist(p, "::1") is None

    def test_empty_allows_any_address(self):
        p = _policy_with_ips([])
        assert check_ip_allowlist(p, "255.255.255.255") is None


class TestIPAllowlistWithEnforcePolicy:
    """IP allowlist integration with full enforce_policy flow."""

    @pytest.mark.asyncio
    async def test_empty_ip_passes_enforcement(self, db):
        """Empty IP string passes when allowlist is empty."""
        policy = AgentPolicy(
            agent_id="ip-test-1", agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            ip_allowlist=[],
            rate_limit_per_minute=60,
            created_by="u1", created_at=time.time(), is_active=True,
        )
        await create_agent_policy(policy)
        result = await enforce_policy("u1", "ip-test-1", "github", ["repo"], ip_address="")
        assert result.agent_id == "ip-test-1"

    @pytest.mark.asyncio
    async def test_cidr_match_passes_enforcement(self, db):
        policy = AgentPolicy(
            agent_id="ip-test-2", agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            ip_allowlist=["192.168.0.0/16"],
            rate_limit_per_minute=60,
            created_by="u1", created_at=time.time(), is_active=True,
        )
        await create_agent_policy(policy)
        result = await enforce_policy(
            "u1", "ip-test-2", "github", ["repo"], ip_address="192.168.50.10"
        )
        assert result.agent_id == "ip-test-2"

    @pytest.mark.asyncio
    async def test_ip_denied_before_service_check(self, db):
        """IP denial fires before service authorization check."""
        policy = AgentPolicy(
            agent_id="ip-test-3", agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            ip_allowlist=["10.0.0.1"],
            rate_limit_per_minute=60,
            created_by="u1", created_at=time.time(), is_active=True,
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="not in allowlist"):
            await enforce_policy(
                "u1", "ip-test-3", "nonexistent", ["repo"], ip_address="192.168.1.1"
            )
