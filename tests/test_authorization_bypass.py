"""Authorization bypass tests -- ensure agents cannot access resources outside
their scope, cross-user isolation, policy boundary enforcement, and various
attempts to circumvent authorization checks."""

import time
import pytest

from src.database import (
    AgentPolicy,
    create_agent_policy,
    create_api_key,
    delete_agent_policy,
    get_agent_policy,
    get_all_policies,
    log_audit,
    toggle_agent_policy,
    validate_api_key,
)
from src.policy import (
    PolicyDenied,
    _rate_counters,
    check_ip_allowlist,
    enforce_policy,
    get_effective_scopes,
    requires_step_up,
)


@pytest.fixture(autouse=True)
def clear_rate_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


def _policy(agent_id="agent-1", user="user-A", services=None, scopes=None, **kw):
    return AgentPolicy(
        agent_id=agent_id,
        agent_name=f"Bot {agent_id}",
        allowed_services=["github"] if services is None else services,
        allowed_scopes={"github": ["repo"]} if scopes is None else scopes,
        rate_limit_per_minute=kw.get("rate_limit", 60),
        created_by=user,
        created_at=time.time(),
        is_active=kw.get("is_active", True),
        expires_at=kw.get("expires_at", 0.0),
        ip_allowlist=kw.get("ip_allowlist", []),
        requires_step_up=kw.get("requires_step_up", []),
        allowed_hours=kw.get("allowed_hours", []),
        allowed_days=kw.get("allowed_days", []),
    )


class TestCrossUserIsolation:
    """Ensure one user cannot access another user's agents or policies."""

    @pytest.mark.asyncio
    async def test_user_b_cannot_enforce_user_a_agent(self, db):
        await create_agent_policy(_policy("priv-agent", user="user-A"))
        with pytest.raises(PolicyDenied, match="do not own"):
            await enforce_policy("user-B", "priv-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_user_b_cannot_toggle_user_a_policy(self, db):
        await create_agent_policy(_policy("tog-agent", user="user-A"))
        result = await toggle_agent_policy("tog-agent", "user-B")
        assert result is None

    @pytest.mark.asyncio
    async def test_user_b_cannot_delete_user_a_policy(self, db):
        await create_agent_policy(_policy("del-agent", user="user-A"))
        deleted = await delete_agent_policy("del-agent", "user-B")
        assert deleted is False
        # Policy still exists
        p = await get_agent_policy("del-agent")
        assert p is not None

    @pytest.mark.asyncio
    async def test_user_b_policies_list_excludes_user_a(self, db):
        await create_agent_policy(_policy("a-agent", user="user-A"))
        await create_agent_policy(_policy("b-agent", user="user-B"))
        a_policies = await get_all_policies("user-A")
        b_policies = await get_all_policies("user-B")
        assert all(p.agent_id == "a-agent" for p in a_policies)
        assert all(p.agent_id == "b-agent" for p in b_policies)

    @pytest.mark.asyncio
    async def test_user_b_api_key_cannot_auth_user_a_agent(self, db):
        await create_agent_policy(_policy("a-sec-agent", user="user-A"))
        _, raw_key = await create_api_key("user-B", "a-sec-agent", "evil-key")
        validated = await validate_api_key(raw_key)
        # Key validates (it's a real key), but enforce_policy rejects
        assert validated is not None
        with pytest.raises(PolicyDenied, match="do not own"):
            await enforce_policy(
                validated.user_id, "a-sec-agent", "github", ["repo"]
            )


class TestServiceBoundaryEnforcement:
    """Ensure agents cannot access services outside their allowed list."""

    @pytest.mark.asyncio
    async def test_single_service_rejects_other(self, db):
        await create_agent_policy(_policy("gh-only", services=["github"]))
        with pytest.raises(PolicyDenied, match="not authorized to access"):
            await enforce_policy("user-A", "gh-only", "slack", [])

    @pytest.mark.asyncio
    async def test_multi_service_rejects_unlisted(self, db):
        await create_agent_policy(
            _policy("multi", services=["github", "slack"],
                    scopes={"github": ["repo"], "slack": ["channels:read"]})
        )
        with pytest.raises(PolicyDenied, match="not authorized to access"):
            await enforce_policy("user-A", "multi", "google", [])

    @pytest.mark.asyncio
    async def test_empty_services_rejects_all(self, db):
        await create_agent_policy(_policy("empty-svc", services=[]))
        with pytest.raises(PolicyDenied, match="not authorized to access"):
            await enforce_policy("user-A", "empty-svc", "github", [])

    @pytest.mark.asyncio
    async def test_case_sensitive_service_name(self, db):
        await create_agent_policy(_policy("cs-agent", services=["github"]))
        with pytest.raises(PolicyDenied, match="not authorized to access"):
            await enforce_policy("user-A", "cs-agent", "GitHub", [])

    @pytest.mark.asyncio
    async def test_service_with_whitespace_mismatch(self, db):
        await create_agent_policy(_policy("ws-agent", services=["github"]))
        with pytest.raises(PolicyDenied, match="not authorized to access"):
            await enforce_policy("user-A", "ws-agent", " github", [])


class TestScopeBoundaryEnforcement:
    """Ensure agents cannot request scopes beyond what's allowed."""

    @pytest.mark.asyncio
    async def test_excess_scope_single(self, db):
        await create_agent_policy(
            _policy("scope-agent", scopes={"github": ["repo"]})
        )
        with pytest.raises(PolicyDenied, match="Scopes not permitted"):
            await enforce_policy("user-A", "scope-agent", "github", ["repo", "admin:org"])

    @pytest.mark.asyncio
    async def test_completely_wrong_scopes(self, db):
        await create_agent_policy(
            _policy("scope2", scopes={"github": ["repo"]})
        )
        with pytest.raises(PolicyDenied, match="Scopes not permitted"):
            await enforce_policy("user-A", "scope2", "github", ["read:org", "admin:org"])

    @pytest.mark.asyncio
    async def test_empty_allowed_rejects_any_scope(self, db):
        await create_agent_policy(
            _policy("no-scope", scopes={"github": []})
        )
        with pytest.raises(PolicyDenied, match="Scopes not permitted"):
            await enforce_policy("user-A", "no-scope", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_requesting_empty_scopes_allowed(self, db):
        await create_agent_policy(
            _policy("any-scope", scopes={"github": ["repo"]})
        )
        result = await enforce_policy("user-A", "any-scope", "github", [])
        assert result.agent_id == "any-scope"

    @pytest.mark.asyncio
    async def test_scope_for_wrong_service(self, db):
        """Scopes defined for github should not apply to slack."""
        await create_agent_policy(
            _policy("cross-scope",
                    services=["github", "slack"],
                    scopes={"github": ["repo"], "slack": ["channels:read"]})
        )
        with pytest.raises(PolicyDenied, match="Scopes not permitted"):
            await enforce_policy("user-A", "cross-scope", "slack", ["repo"])


class TestDisabledAndExpiredBypass:
    """Ensure disabled/expired agents cannot be used."""

    @pytest.mark.asyncio
    async def test_disabled_agent_blocked(self, db):
        await create_agent_policy(_policy("dis-agent", is_active=False))
        with pytest.raises(PolicyDenied, match="disabled"):
            await enforce_policy("user-A", "dis-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_expired_policy_blocked(self, db):
        await create_agent_policy(
            _policy("exp-agent", expires_at=time.time() - 3600)
        )
        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy("user-A", "exp-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_just_expired_policy_blocked(self, db):
        """Policy that expired 1 second ago is denied."""
        await create_agent_policy(
            _policy("just-exp", expires_at=time.time() - 1)
        )
        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy("user-A", "just-exp", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_not_yet_expired_policy_allowed(self, db):
        await create_agent_policy(
            _policy("future-exp", expires_at=time.time() + 3600)
        )
        result = await enforce_policy("user-A", "future-exp", "github", ["repo"])
        assert result.agent_id == "future-exp"

    @pytest.mark.asyncio
    async def test_disabled_then_reenabled_works(self, db):
        await create_agent_policy(_policy("re-enable"))
        await toggle_agent_policy("re-enable", "user-A")
        with pytest.raises(PolicyDenied, match="disabled"):
            await enforce_policy("user-A", "re-enable", "github", ["repo"])
        await toggle_agent_policy("re-enable", "user-A")
        result = await enforce_policy("user-A", "re-enable", "github", ["repo"])
        assert result.agent_id == "re-enable"


class TestNonexistentAgentBypass:
    """Ensure nonexistent agents are always rejected."""

    @pytest.mark.asyncio
    async def test_nonexistent_agent_denied(self, db):
        with pytest.raises(PolicyDenied, match="not registered"):
            await enforce_policy("user-A", "ghost-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_deleted_agent_denied(self, db):
        await create_agent_policy(_policy("temp-agent"))
        await delete_agent_policy("temp-agent", "user-A")
        with pytest.raises(PolicyDenied, match="not registered"):
            await enforce_policy("user-A", "temp-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_empty_agent_id(self, db):
        with pytest.raises(PolicyDenied, match="not registered"):
            await enforce_policy("user-A", "", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_whitespace_agent_id(self, db):
        with pytest.raises(PolicyDenied, match="not registered"):
            await enforce_policy("user-A", "   ", "github", ["repo"])


class TestIPAllowlistBypass:
    """Ensure IP allowlist cannot be circumvented."""

    @pytest.mark.asyncio
    async def test_wrong_ip_denied(self, db):
        await create_agent_policy(
            _policy("ip-agent", ip_allowlist=["10.0.0.1"])
        )
        with pytest.raises(PolicyDenied, match="not in allowlist"):
            await enforce_policy("user-A", "ip-agent", "github", ["repo"], ip_address="192.168.1.1")

    @pytest.mark.asyncio
    async def test_cidr_outside_range_denied(self, db):
        await create_agent_policy(
            _policy("cidr-agent", ip_allowlist=["10.0.0.0/24"])
        )
        with pytest.raises(PolicyDenied, match="not in allowlist"):
            await enforce_policy("user-A", "cidr-agent", "github", ["repo"], ip_address="10.0.1.1")

    @pytest.mark.asyncio
    async def test_ipv6_not_matching_ipv4_allowlist(self, db):
        await create_agent_policy(
            _policy("v4-only", ip_allowlist=["10.0.0.1"])
        )
        with pytest.raises(PolicyDenied, match="not in allowlist"):
            await enforce_policy("user-A", "v4-only", "github", ["repo"], ip_address="::1")

    @pytest.mark.asyncio
    async def test_spoofed_ip_format_denied(self, db):
        await create_agent_policy(
            _policy("format-agent", ip_allowlist=["10.0.0.1"])
        )
        with pytest.raises(PolicyDenied, match="invalid IP"):
            await enforce_policy("user-A", "format-agent", "github", ["repo"], ip_address="not-an-ip")

    @pytest.mark.asyncio
    async def test_empty_ip_with_allowlist_denied(self, db):
        await create_agent_policy(
            _policy("empty-ip", ip_allowlist=["10.0.0.1"])
        )
        with pytest.raises(PolicyDenied, match="invalid IP"):
            await enforce_policy("user-A", "empty-ip", "github", ["repo"], ip_address="")


class TestEnforcementOrdering:
    """Verify all policy checks run in correct order."""

    @pytest.mark.asyncio
    async def test_disabled_checked_before_ownership(self, db):
        """Disabled check (step 1) should fire before ownership (step 2)."""
        await create_agent_policy(_policy("ord-agent", user="user-A", is_active=False))
        with pytest.raises(PolicyDenied, match="disabled"):
            await enforce_policy("user-B", "ord-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_ownership_checked_before_service(self, db):
        """Ownership check fires before service check."""
        await create_agent_policy(_policy("own-agent", user="user-A", services=["slack"]))
        with pytest.raises(PolicyDenied, match="do not own"):
            await enforce_policy("user-B", "own-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_expiry_checked_before_ip(self, db):
        """Expiry check fires before IP check."""
        await create_agent_policy(
            _policy("exp-ip", expires_at=time.time() - 100, ip_allowlist=["10.0.0.1"])
        )
        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy("user-A", "exp-ip", "github", ["repo"], ip_address="192.168.1.1")

    @pytest.mark.asyncio
    async def test_all_checks_pass_happy_path(self, db):
        await create_agent_policy(
            _policy("happy",
                    services=["github"],
                    scopes={"github": ["repo"]},
                    ip_allowlist=["10.0.0.1"],
                    expires_at=time.time() + 3600)
        )
        result = await enforce_policy("user-A", "happy", "github", ["repo"], ip_address="10.0.0.1")
        assert result.agent_id == "happy"


class TestPolicyOverwriteBypass:
    """Ensure policy overwrite doesn't bypass controls."""

    @pytest.mark.asyncio
    async def test_overwrite_preserves_owner(self, db):
        """Overwriting a policy keeps the new owner."""
        await create_agent_policy(_policy("ow-agent", user="user-A"))
        p = await get_agent_policy("ow-agent")
        assert p.created_by == "user-A"
        # Overwrite with user-B
        await create_agent_policy(_policy("ow-agent", user="user-B"))
        p2 = await get_agent_policy("ow-agent")
        assert p2.created_by == "user-B"
        # user-A can no longer access
        with pytest.raises(PolicyDenied, match="do not own"):
            await enforce_policy("user-A", "ow-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_overwrite_with_expanded_services(self, db):
        """Overwriting a policy can expand services."""
        await create_agent_policy(_policy("expand", services=["github"]))
        await create_agent_policy(
            _policy("expand", services=["github", "slack"],
                    scopes={"github": ["repo"], "slack": ["channels:read"]})
        )
        result = await enforce_policy("user-A", "expand", "slack", ["channels:read"])
        assert result.agent_id == "expand"


class TestEffectiveScopesIntegrity:
    """Ensure effective scope intersection is correct."""

    def test_no_overlap(self):
        p = AgentPolicy(agent_id="t", agent_name="t",
                        allowed_scopes={"github": ["repo"]})
        assert get_effective_scopes(p, "github", ["admin:org"]) == []

    def test_partial_overlap(self):
        p = AgentPolicy(agent_id="t", agent_name="t",
                        allowed_scopes={"github": ["repo", "read:user"]})
        result = get_effective_scopes(p, "github", ["repo", "admin:org"])
        assert set(result) == {"repo"}

    def test_full_overlap(self):
        p = AgentPolicy(agent_id="t", agent_name="t",
                        allowed_scopes={"github": ["repo", "read:user"]})
        result = get_effective_scopes(p, "github", ["repo", "read:user"])
        assert set(result) == {"repo", "read:user"}

    def test_service_not_in_scopes(self):
        p = AgentPolicy(agent_id="t", agent_name="t",
                        allowed_scopes={"github": ["repo"]})
        assert get_effective_scopes(p, "slack", ["channels:read"]) == []

    def test_duplicate_scopes_deduplicated(self):
        p = AgentPolicy(agent_id="t", agent_name="t",
                        allowed_scopes={"github": ["repo", "repo"]})
        result = get_effective_scopes(p, "github", ["repo", "repo"])
        assert result == ["repo"]
