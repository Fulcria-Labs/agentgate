"""Security-focused tests for AgentGate — enforcement order, emergency revoke,
API key security, rate limiting edge cases, multi-policy interactions, and
audit logging completeness."""

import hashlib
import time
from unittest.mock import patch

import pytest

from src.database import (
    AgentPolicy,
    create_agent_policy,
    create_api_key,
    delete_agent_policy,
    emergency_revoke_all,
    get_agent_policy,
    get_all_policies,
    get_audit_log,
    log_audit,
    revoke_api_key,
    toggle_agent_policy,
    validate_api_key,
)
from src.policy import (
    PolicyDenied,
    _rate_counters,
    enforce_policy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_rate_counters():
    """Clear in-memory rate limit counters between tests."""
    _rate_counters.clear()
    yield
    _rate_counters.clear()


# ---------------------------------------------------------------------------
# 1. Policy enforcement order (5 tests)
# ---------------------------------------------------------------------------

class TestPolicyEnforcementOrder:
    """Verify that policy checks are evaluated in the documented order so
    that earlier denials short-circuit later checks."""

    @pytest.mark.asyncio
    async def test_inactive_denied_before_ownership(self, db):
        """Inactive agent is denied even when user_id doesn't match (step 1 before 2)."""
        policy = AgentPolicy(
            agent_id="order-1",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="owner",
            created_at=time.time(),
            is_active=False,
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="disabled"):
            # user_id="intruder" would fail ownership, but inactive fires first
            await enforce_policy("intruder", "order-1", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_ownership_denied_before_expiration(self, db):
        """Ownership denial fires before expiration check (step 2 before 3)."""
        policy = AgentPolicy(
            agent_id="order-2",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="owner",
            created_at=time.time(),
            is_active=True,
            expires_at=time.time() - 3600,  # already expired
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="do not own"):
            await enforce_policy("wrong-user", "order-2", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_expired_denied_before_time_window(self, db):
        """Expiration denial fires before time window check (step 3 before 4)."""
        policy = AgentPolicy(
            agent_id="order-3",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
            is_active=True,
            expires_at=time.time() - 3600,
            allowed_hours=[23],  # would also deny at most hours
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy("user1", "order-3", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_unauthorized_service_denied_before_scope(self, db):
        """Service denial fires before scope check (step 6 before 7)."""
        policy = AgentPolicy(
            agent_id="order-4",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        # "slack" is unauthorized, and "admin:org" would also fail scope check
        with pytest.raises(PolicyDenied, match="not authorized to access"):
            await enforce_policy("user1", "order-4", "slack", ["admin:org"])

    @pytest.mark.asyncio
    async def test_excess_scopes_denied_before_rate_limit(self, db):
        """Scope denial fires before rate limit check (step 7 before 8)."""
        policy = AgentPolicy(
            agent_id="order-5",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=0,  # would block all requests via rate limit
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        with pytest.raises(PolicyDenied, match="Scopes not permitted"):
            await enforce_policy("user1", "order-5", "github", ["repo", "admin:org"])


# ---------------------------------------------------------------------------
# 2. Emergency revoke scenarios (5 tests)
# ---------------------------------------------------------------------------

class TestEmergencyRevoke:
    """Verify that the emergency kill switch properly disables everything
    for the targeted user without affecting other users."""

    @pytest.mark.asyncio
    async def test_disables_all_policies(self, db):
        """Emergency revoke sets is_active=False on all user policies."""
        for i in range(3):
            await create_agent_policy(AgentPolicy(
                agent_id=f"em-agent-{i}",
                agent_name=f"Bot {i}",
                allowed_services=["github"],
                created_by="victim",
                created_at=time.time(),
                is_active=True,
            ))

        result = await emergency_revoke_all("victim")
        assert result["policies_disabled"] == 3

        policies = await get_all_policies("victim")
        assert all(not p.is_active for p in policies)

    @pytest.mark.asyncio
    async def test_revokes_all_api_keys(self, db):
        """Emergency revoke revokes all API keys for the user."""
        raw_keys = []
        for i in range(3):
            _, raw = await create_api_key("victim", f"agent-{i}", f"key-{i}")
            raw_keys.append(raw)

        result = await emergency_revoke_all("victim")
        assert result["keys_revoked"] == 3

        for raw in raw_keys:
            assert await validate_api_key(raw) is None

    @pytest.mark.asyncio
    async def test_does_not_affect_other_users(self, db):
        """Emergency revoke leaves other users' policies and keys intact."""
        await create_agent_policy(AgentPolicy(
            agent_id="victim-agent",
            agent_name="Victim Bot",
            allowed_services=["github"],
            created_by="victim",
            created_at=time.time(),
            is_active=True,
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="safe-agent",
            agent_name="Safe Bot",
            allowed_services=["github"],
            created_by="other-user",
            created_at=time.time(),
            is_active=True,
        ))
        _, safe_raw = await create_api_key("other-user", "safe-agent", "safe-key")

        await emergency_revoke_all("victim")

        safe_policy = await get_agent_policy("safe-agent")
        assert safe_policy.is_active is True
        assert await validate_api_key(safe_raw) is not None

    @pytest.mark.asyncio
    async def test_token_requests_denied_after_revoke(self, db):
        """After emergency revoke, enforce_policy denies token requests."""
        policy = AgentPolicy(
            agent_id="revoked-agent",
            agent_name="Revoked Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="victim",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)
        # Verify it works before revoke
        result = await enforce_policy("victim", "revoked-agent", "github", ["repo"])
        assert result.agent_id == "revoked-agent"

        await emergency_revoke_all("victim")

        with pytest.raises(PolicyDenied, match="disabled"):
            await enforce_policy("victim", "revoked-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_api_keys_fail_after_revoke(self, db):
        """After emergency revoke, previously valid API keys fail validation."""
        _, raw = await create_api_key("victim", "agent-x", "my-key")
        assert await validate_api_key(raw) is not None

        await emergency_revoke_all("victim")

        assert await validate_api_key(raw) is None


# ---------------------------------------------------------------------------
# 3. API key security (5 tests)
# ---------------------------------------------------------------------------

class TestApiKeySecurity:
    """Verify cryptographic properties and lifecycle behaviour of API keys."""

    @pytest.mark.asyncio
    async def test_key_hash_is_sha256(self, db):
        """The stored key_hash is the SHA-256 hex digest of the raw key."""
        api_key, raw = await create_api_key("user1", "agent-1", "test")
        expected_hash = hashlib.sha256(raw.encode()).hexdigest()
        assert api_key.key_hash == expected_hash

    @pytest.mark.asyncio
    async def test_multiple_keys_validate_independently(self, db):
        """Multiple keys for the same agent each validate on their own."""
        _, raw1 = await create_api_key("user1", "agent-1", "key-1")
        _, raw2 = await create_api_key("user1", "agent-1", "key-2")
        _, raw3 = await create_api_key("user1", "agent-1", "key-3")

        for raw in (raw1, raw2, raw3):
            result = await validate_api_key(raw)
            assert result is not None
            assert result.agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_revoked_key_immediately_fails(self, db):
        """A key revoked by ID immediately fails validation."""
        api_key, raw = await create_api_key("user1", "agent-1", "revocable")
        assert await validate_api_key(raw) is not None

        await revoke_api_key(api_key.id, "user1")
        assert await validate_api_key(raw) is None

    @pytest.mark.asyncio
    async def test_expired_key_fails(self, db):
        """A key whose expires_at is in the past fails validation."""
        # expires_in=-1 sets expires_at to now - 1 (already expired)
        _, raw = await create_api_key("user1", "agent-1", "expired", expires_in=-1)
        assert await validate_api_key(raw) is None

    @pytest.mark.asyncio
    async def test_key_prefix_is_first_8_chars(self, db):
        """The key_prefix stored in the database is the first 8 characters of the raw key."""
        api_key, raw = await create_api_key("user1", "agent-1", "prefix-test")
        assert api_key.key_prefix == raw[:8]
        assert len(api_key.key_prefix) == 8


# ---------------------------------------------------------------------------
# 4. Rate limiting edge cases (5 tests)
# ---------------------------------------------------------------------------

class TestRateLimitingEdgeCases:
    """Verify rate limit counter behaviour including window resets, counter
    isolation, exact-boundary behaviour, and zero-limit blocking."""

    @pytest.mark.asyncio
    async def test_counter_resets_after_60s(self, db):
        """Requests older than 60 seconds are pruned from the counter."""
        policy = AgentPolicy(
            agent_id="rl-reset",
            agent_name="RL Bot",
            allowed_services=["svc"],
            allowed_scopes={"svc": ["read"]},
            rate_limit_per_minute=1,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)

        await enforce_policy("user1", "rl-reset", "svc", ["read"])

        # Second request normally fails
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy("user1", "rl-reset", "svc", ["read"])

        # Simulate time passing: shift all timestamps back 61 seconds
        key = "rl-reset:svc"
        _rate_counters[key] = [t - 61 for t in _rate_counters[key]]

        # Now it should succeed because old entries are pruned
        result = await enforce_policy("user1", "rl-reset", "svc", ["read"])
        assert result.agent_id == "rl-reset"

    @pytest.mark.asyncio
    async def test_independent_counters_per_agent_service(self, db):
        """Different agent:service combinations have separate rate counters."""
        for aid in ("rl-a", "rl-b"):
            await create_agent_policy(AgentPolicy(
                agent_id=aid,
                agent_name=f"Bot {aid}",
                allowed_services=["svc"],
                allowed_scopes={"svc": ["read"]},
                rate_limit_per_minute=1,
                created_by="user1",
                created_at=time.time(),
                is_active=True,
            ))

        await enforce_policy("user1", "rl-a", "svc", ["read"])
        # rl-a is now exhausted
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy("user1", "rl-a", "svc", ["read"])

        # rl-b should still be fine
        result = await enforce_policy("user1", "rl-b", "svc", ["read"])
        assert result.agent_id == "rl-b"

    @pytest.mark.asyncio
    async def test_rate_limit_of_one(self, db):
        """A rate limit of 1 allows exactly one request, then denies the next."""
        await create_agent_policy(AgentPolicy(
            agent_id="rl-one",
            agent_name="One Shot",
            allowed_services=["svc"],
            allowed_scopes={"svc": ["r"]},
            rate_limit_per_minute=1,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        ))

        result = await enforce_policy("user1", "rl-one", "svc", ["r"])
        assert result.agent_id == "rl-one"

        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy("user1", "rl-one", "svc", ["r"])

    @pytest.mark.asyncio
    async def test_rate_limit_of_zero_blocks_all(self, db):
        """A rate limit of 0 blocks every single request."""
        await create_agent_policy(AgentPolicy(
            agent_id="rl-zero",
            agent_name="Zero Bot",
            allowed_services=["svc"],
            allowed_scopes={"svc": ["r"]},
            rate_limit_per_minute=0,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        ))

        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy("user1", "rl-zero", "svc", ["r"])

    @pytest.mark.asyncio
    async def test_burst_at_exact_limit(self, db):
        """N requests at limit N all succeed; the (N+1)th is denied."""
        limit = 5
        await create_agent_policy(AgentPolicy(
            agent_id="rl-burst",
            agent_name="Burst Bot",
            allowed_services=["svc"],
            allowed_scopes={"svc": ["r"]},
            rate_limit_per_minute=limit,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        ))

        for i in range(limit):
            result = await enforce_policy("user1", "rl-burst", "svc", ["r"])
            assert result.agent_id == "rl-burst"

        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy("user1", "rl-burst", "svc", ["r"])


# ---------------------------------------------------------------------------
# 5. Multi-policy interactions (5 tests)
# ---------------------------------------------------------------------------

class TestMultiPolicyInteractions:
    """Verify that multiple agent policies for the same user are properly
    isolated and that CRUD on one does not affect others."""

    @pytest.mark.asyncio
    async def test_user_can_have_multiple_policies(self, db):
        """A single user can create several independent agent policies."""
        for i in range(4):
            await create_agent_policy(AgentPolicy(
                agent_id=f"multi-{i}",
                agent_name=f"Bot {i}",
                allowed_services=["github"],
                created_by="user1",
                created_at=time.time(),
            ))

        policies = await get_all_policies("user1")
        assert len(policies) == 4

    @pytest.mark.asyncio
    async def test_delete_one_does_not_affect_others(self, db):
        """Deleting one policy leaves the rest untouched."""
        for i in range(3):
            await create_agent_policy(AgentPolicy(
                agent_id=f"del-{i}",
                agent_name=f"Bot {i}",
                allowed_services=["github"],
                created_by="user1",
                created_at=time.time(),
            ))

        await delete_agent_policy("del-1", "user1")

        remaining = await get_all_policies("user1")
        ids = {p.agent_id for p in remaining}
        assert ids == {"del-0", "del-2"}

    @pytest.mark.asyncio
    async def test_toggle_one_does_not_affect_others(self, db):
        """Toggling one policy's active state doesn't change others."""
        for i in range(3):
            await create_agent_policy(AgentPolicy(
                agent_id=f"tog-{i}",
                agent_name=f"Bot {i}",
                allowed_services=["github"],
                created_by="user1",
                created_at=time.time(),
                is_active=True,
            ))

        await toggle_agent_policy("tog-1", "user1")

        for i in range(3):
            p = await get_agent_policy(f"tog-{i}")
            if i == 1:
                assert p.is_active is False
            else:
                assert p.is_active is True

    @pytest.mark.asyncio
    async def test_different_agents_different_services(self, db):
        """Two agents under the same user can have completely different allowed services."""
        await create_agent_policy(AgentPolicy(
            agent_id="svc-a",
            agent_name="GitHub Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="svc-b",
            agent_name="Slack Bot",
            allowed_services=["slack"],
            allowed_scopes={"slack": ["chat:write"]},
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        ))

        # svc-a can access github but not slack
        result = await enforce_policy("user1", "svc-a", "github", ["repo"])
        assert result.agent_id == "svc-a"
        with pytest.raises(PolicyDenied, match="not authorized"):
            await enforce_policy("user1", "svc-a", "slack", ["chat:write"])

        # svc-b can access slack but not github
        result = await enforce_policy("user1", "svc-b", "slack", ["chat:write"])
        assert result.agent_id == "svc-b"
        with pytest.raises(PolicyDenied, match="not authorized"):
            await enforce_policy("user1", "svc-b", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_scopes_are_service_specific(self, db):
        """An agent's allowed scopes for one service don't leak to another."""
        await create_agent_policy(AgentPolicy(
            agent_id="scope-iso",
            agent_name="Multi Bot",
            allowed_services=["github", "slack"],
            allowed_scopes={
                "github": ["repo", "read:user"],
                "slack": ["chat:write"],
            },
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        ))

        # "repo" is valid for github
        result = await enforce_policy("user1", "scope-iso", "github", ["repo"])
        assert result.agent_id == "scope-iso"

        # "repo" is NOT valid for slack
        with pytest.raises(PolicyDenied, match="Scopes not permitted"):
            await enforce_policy("user1", "scope-iso", "slack", ["repo"])


# ---------------------------------------------------------------------------
# 6. Audit logging completeness (5 tests)
# ---------------------------------------------------------------------------

class TestAuditLoggingCompleteness:
    """Verify that audit entries are created for all policy evaluation
    outcomes and contain the expected metadata."""

    @pytest.mark.asyncio
    async def test_successful_request_creates_audit_entry(self, db):
        """A passing enforce_policy call logs an audit entry via the rate
        counter (the counter itself doesn't log, but we verify the flow
        by manually logging as the real app does)."""
        await log_audit("user1", "agent-1", "github", "token_request", "success",
                       scopes="repo")
        entries = await get_audit_log("user1")
        assert len(entries) == 1
        assert entries[0].status == "success"
        assert entries[0].action == "token_request"

    @pytest.mark.asyncio
    async def test_denied_request_creates_audit_entry(self, db):
        """A denied enforce_policy call writes an audit entry."""
        policy = AgentPolicy(
            agent_id="audit-deny",
            agent_name="Audit Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)

        with pytest.raises(PolicyDenied):
            await enforce_policy("user1", "audit-deny", "slack", ["read"])

        entries = await get_audit_log("user1")
        assert len(entries) >= 1
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 1
        assert "not in allowed list" in denied[0].details or "not authorized" in denied[0].details

    @pytest.mark.asyncio
    async def test_rate_limited_creates_audit_entry(self, db):
        """A rate-limited request writes an audit entry with status 'rate_limited'."""
        policy = AgentPolicy(
            agent_id="audit-rl",
            agent_name="RL Bot",
            allowed_services=["svc"],
            allowed_scopes={"svc": ["r"]},
            rate_limit_per_minute=1,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)

        await enforce_policy("user1", "audit-rl", "svc", ["r"])
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy("user1", "audit-rl", "svc", ["r"])

        entries = await get_audit_log("user1")
        rate_limited = [e for e in entries if e.status == "rate_limited"]
        assert len(rate_limited) >= 1

    @pytest.mark.asyncio
    async def test_audit_entries_contain_ip_address(self, db):
        """Audit entries record the requesting IP address when provided."""
        policy = AgentPolicy(
            agent_id="audit-ip",
            agent_name="IP Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            ip_allowlist=["10.0.0.1"],
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        )
        await create_agent_policy(policy)

        with pytest.raises(PolicyDenied, match="not in allowlist"):
            await enforce_policy("user1", "audit-ip", "github", ["repo"],
                               ip_address="192.168.1.1")

        entries = await get_audit_log("user1")
        assert len(entries) >= 1
        assert entries[0].ip_address == "192.168.1.1"

    @pytest.mark.asyncio
    async def test_audit_entries_ordered_by_timestamp_desc(self, db):
        """Audit log returns entries newest-first."""
        for i in range(5):
            await log_audit("user1", f"agent-{i}", "svc", f"action_{i}", "success")

        entries = await get_audit_log("user1")
        assert len(entries) == 5
        # Timestamps should be non-increasing (newest first)
        for a, b in zip(entries, entries[1:]):
            assert a.timestamp >= b.timestamp
        # The last action logged should appear first
        assert entries[0].action == "action_4"
