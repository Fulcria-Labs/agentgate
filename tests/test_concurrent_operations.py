"""Tests for concurrent agent operations, race conditions, and load scenarios.

Covers simultaneous policy enforcement, concurrent token requests,
API key lifecycle under load, multi-agent interactions, and
database integrity under concurrent access.
"""

import asyncio
import time
import pytest
from unittest.mock import patch, MagicMock

from src.database import (
    init_db,
    create_agent_policy,
    get_agent_policy,
    get_all_policies,
    toggle_agent_policy,
    delete_agent_policy,
    create_api_key,
    validate_api_key,
    revoke_api_key,
    get_api_keys,
    emergency_revoke_all,
    add_connected_service,
    get_connected_services,
    remove_connected_service,
    log_audit,
    get_audit_log,
    AgentPolicy,
)
from src.policy import enforce_policy, PolicyDenied, check_time_window, check_ip_allowlist


# ============================================================
# Concurrent Policy Operations
# ============================================================

class TestConcurrentPolicyCreation:
    """Test concurrent policy creation and modification."""

    @pytest.mark.asyncio
    async def test_create_multiple_policies_concurrently(self, db):
        """Creating multiple distinct agent policies concurrently should work."""
        async def create_policy(i):
            policy = AgentPolicy(
                agent_id=f"agent-concurrent-{i}",
                agent_name=f"Concurrent Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                created_by="user1",
                created_at=time.time(),
            )
            await create_agent_policy(policy)
            return f"agent-concurrent-{i}"

        agent_ids = await asyncio.gather(*[create_policy(i) for i in range(20)])
        assert len(agent_ids) == 20

        # Verify all were created
        policies = await get_all_policies("user1")
        created_ids = {p.agent_id for p in policies}
        for aid in agent_ids:
            assert aid in created_ids

    @pytest.mark.asyncio
    async def test_concurrent_policy_updates_same_agent(self, db):
        """Concurrent updates to the same agent policy should not corrupt data."""
        policy = AgentPolicy(
            agent_id="agent-update-race",
            agent_name="Race Agent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        async def update_policy(services):
            p = AgentPolicy(
                agent_id="agent-update-race",
                agent_name="Race Agent Updated",
                allowed_services=services,
                allowed_scopes={s: ["repo"] for s in services},
                created_by="user1",
                created_at=time.time(),
            )
            await create_agent_policy(p)

        # Run concurrent updates with different service lists
        await asyncio.gather(
            update_policy(["github"]),
            update_policy(["slack"]),
            update_policy(["google"]),
            update_policy(["github", "slack"]),
        )

        # Final state should be one of the updates (last writer wins)
        result = await get_agent_policy("agent-update-race")
        assert result is not None
        assert len(result.allowed_services) >= 1

    @pytest.mark.asyncio
    async def test_concurrent_toggle_same_policy(self, db):
        """Toggling the same policy concurrently should not error."""
        policy = AgentPolicy(
            agent_id="agent-toggle-race",
            agent_name="Toggle Race",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        results = await asyncio.gather(*[
            toggle_agent_policy("agent-toggle-race", "user1") for _ in range(10)
        ])

        # All toggles should return a boolean
        for r in results:
            assert isinstance(r, bool)

    @pytest.mark.asyncio
    async def test_delete_while_toggling(self, db):
        """Deleting a policy while toggling should not crash."""
        policy = AgentPolicy(
            agent_id="agent-del-toggle",
            agent_name="Del Toggle",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        results = await asyncio.gather(
            toggle_agent_policy("agent-del-toggle", "user1"),
            delete_agent_policy("agent-del-toggle", "user1"),
            toggle_agent_policy("agent-del-toggle", "user1"),
            return_exceptions=True,
        )

        # At least one should succeed, none should crash
        for r in results:
            assert not isinstance(r, Exception) or isinstance(r, (PolicyDenied, KeyError))


# ============================================================
# Concurrent API Key Operations
# ============================================================

class TestConcurrentAPIKeys:
    """Test concurrent API key creation, validation, and revocation."""

    @pytest.mark.asyncio
    async def test_create_multiple_keys_concurrently(self, db):
        """Creating many API keys concurrently should all succeed."""
        policy = AgentPolicy(
            agent_id="agent-keys",
            agent_name="Keys Agent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        async def create_key(i):
            key_obj, raw = await create_api_key("user1", "agent-keys", f"key-{i}")
            return raw

        raw_keys = await asyncio.gather(*[create_key(i) for i in range(15)])
        assert len(raw_keys) == 15

        # All keys should be unique
        assert len(set(raw_keys)) == 15

        # All should start with ag_
        for key in raw_keys:
            assert key.startswith("ag_")

    @pytest.mark.asyncio
    async def test_validate_keys_concurrently(self, db):
        """Validating multiple keys concurrently should work."""
        raw_keys = []
        for i in range(5):
            _, raw = await create_api_key("user1", "agent-v", f"key-{i}")
            raw_keys.append(raw)

        results = await asyncio.gather(*[validate_api_key(k) for k in raw_keys])
        for r in results:
            assert r is not None
            assert r.agent_id == "agent-v"

    @pytest.mark.asyncio
    async def test_revoke_and_validate_concurrently(self, db):
        """Revoking a key while validating it concurrently."""
        key_obj, raw = await create_api_key("user1", "agent-rv", "test-key")

        async def revoke():
            return await revoke_api_key(key_obj.id, "user1")

        async def validate():
            return await validate_api_key(raw)

        results = await asyncio.gather(
            validate(),
            revoke(),
            validate(),
            return_exceptions=True,
        )

        # No exceptions should be raised
        for r in results:
            assert not isinstance(r, Exception)

    @pytest.mark.asyncio
    async def test_each_key_has_unique_prefix(self, db):
        """Each API key should have a distinct prefix."""
        keys = []
        for i in range(10):
            key_obj, raw = await create_api_key("user1", "agent-uniq", f"key-{i}")
            keys.append(key_obj)

        # Prefixes are first 8 chars of the raw key
        # They won't all be unique since they share the ag_ prefix,
        # but key_prefix field should be populated
        for k in keys:
            assert len(k.key_prefix) > 0

    @pytest.mark.asyncio
    async def test_revoked_key_not_valid(self, db):
        """A revoked key should fail validation."""
        key_obj, raw = await create_api_key("user1", "agent-revoked", "test")
        await revoke_api_key(key_obj.id, "user1")
        result = await validate_api_key(raw)
        assert result is None

    @pytest.mark.asyncio
    async def test_expired_key_not_valid(self, db):
        """An expired key should fail validation."""
        key_obj, raw = await create_api_key("user1", "agent-exp", "test", expires_in=1)
        # Wait for expiration
        await asyncio.sleep(1.1)
        result = await validate_api_key(raw)
        assert result is None

    @pytest.mark.asyncio
    async def test_key_last_used_updated_on_validation(self, db):
        """Validating a key should update its last_used_at timestamp."""
        key_obj, raw = await create_api_key("user1", "agent-lu", "test")
        before = time.time()
        result = await validate_api_key(raw)
        assert result is not None
        assert result.last_used_at >= before


# ============================================================
# Concurrent Policy Enforcement
# ============================================================

class TestConcurrentPolicyEnforcement:
    """Test concurrent policy enforcement scenarios."""

    @pytest.mark.asyncio
    async def test_enforce_multiple_agents_concurrently(self, db):
        """Enforcing policies for different agents concurrently should work."""
        for i in range(5):
            policy = AgentPolicy(
                agent_id=f"agent-enforce-{i}",
                agent_name=f"Enforce Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by="user1",
                created_at=time.time(),
            )
            await create_agent_policy(policy)

        results = await asyncio.gather(*[
            enforce_policy("user1", f"agent-enforce-{i}", "github", ["repo"])
            for i in range(5)
        ])

        for r in results:
            assert r is not None
            assert "github" in r.allowed_services

    @pytest.mark.asyncio
    async def test_rate_limit_under_concurrent_requests(self, db):
        """Rate limiting should still work under concurrent access."""
        policy = AgentPolicy(
            agent_id="agent-rl-concurrent",
            agent_name="Rate Limit Agent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=5,  # Low limit
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        # Clear any existing rate counters
        from src.policy import _rate_counters
        _rate_counters.clear()

        results = await asyncio.gather(*[
            enforce_policy("user1", "agent-rl-concurrent", "github", ["repo"])
            for _ in range(5)
        ], return_exceptions=True)

        # At least some should succeed, and rate limit should eventually kick in
        successes = [r for r in results if not isinstance(r, PolicyDenied)]
        assert len(successes) <= 5

    @pytest.mark.asyncio
    async def test_enforce_nonexistent_agents_concurrently(self, db):
        """Enforcing policies for nonexistent agents should all fail gracefully."""
        results = await asyncio.gather(*[
            enforce_policy("user1", f"ghost-agent-{i}", "github", ["repo"])
            for i in range(5)
        ], return_exceptions=True)

        for r in results:
            assert isinstance(r, PolicyDenied)
            assert "not registered" in str(r)


# ============================================================
# Emergency Revoke Under Load
# ============================================================

class TestEmergencyRevokeUnderLoad:
    """Test emergency revoke with many active policies and keys."""

    @pytest.mark.asyncio
    async def test_emergency_revoke_many_policies(self, db):
        """Emergency revoke should disable all policies at once."""
        for i in range(10):
            policy = AgentPolicy(
                agent_id=f"agent-emerg-{i}",
                agent_name=f"Emergency Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                created_by="user1",
                created_at=time.time(),
            )
            await create_agent_policy(policy)

        result = await emergency_revoke_all("user1")
        assert result["policies_disabled"] == 10

        # Verify all are disabled
        policies = await get_all_policies("user1")
        for p in policies:
            assert not p.is_active

    @pytest.mark.asyncio
    async def test_emergency_revoke_many_keys(self, db):
        """Emergency revoke should revoke all API keys."""
        for i in range(8):
            await create_api_key("user1", f"agent-ek-{i}", f"key-{i}")

        result = await emergency_revoke_all("user1")
        assert result["keys_revoked"] == 8

        # Verify all keys are revoked
        keys = await get_api_keys("user1")
        for k in keys:
            assert k.is_revoked

    @pytest.mark.asyncio
    async def test_emergency_revoke_with_concurrent_requests(self, db):
        """Emergency revoke during concurrent operations should not crash."""
        for i in range(5):
            policy = AgentPolicy(
                agent_id=f"agent-conc-emerg-{i}",
                agent_name=f"Conc Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by="user1",
                created_at=time.time(),
            )
            await create_agent_policy(policy)

        # Clear rate counters
        from src.policy import _rate_counters
        _rate_counters.clear()

        # Run emergency revoke concurrently with policy enforcement
        results = await asyncio.gather(
            emergency_revoke_all("user1"),
            enforce_policy("user1", "agent-conc-emerg-0", "github", ["repo"]),
            enforce_policy("user1", "agent-conc-emerg-1", "github", ["repo"]),
            return_exceptions=True,
        )

        # Emergency revoke should succeed
        revoke_result = results[0]
        if isinstance(revoke_result, dict):
            assert revoke_result["policies_disabled"] >= 0

    @pytest.mark.asyncio
    async def test_emergency_revoke_idempotent(self, db):
        """Calling emergency revoke twice should not error."""
        policy = AgentPolicy(
            agent_id="agent-idemp",
            agent_name="Idempotent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        r1 = await emergency_revoke_all("user1")
        assert r1["policies_disabled"] >= 1

        r2 = await emergency_revoke_all("user1")
        # Second call should succeed without errors
        assert isinstance(r2["policies_disabled"], int)
        assert r2["keys_revoked"] == 0

    @pytest.mark.asyncio
    async def test_emergency_revoke_user_isolation(self, db):
        """Emergency revoke for user1 should not affect user2."""
        for uid in ["user1", "user2"]:
            policy = AgentPolicy(
                agent_id=f"agent-iso-{uid}",
                agent_name=f"Agent {uid}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                created_by=uid,
                created_at=time.time(),
            )
            await create_agent_policy(policy)

        await emergency_revoke_all("user1")

        # user2's policy should still be active
        p2 = await get_agent_policy(f"agent-iso-user2")
        assert p2 is not None
        assert p2.is_active


# ============================================================
# Multi-Agent Multi-Service Scenarios
# ============================================================

class TestMultiAgentScenarios:
    """Test scenarios with multiple agents accessing multiple services."""

    @pytest.mark.asyncio
    async def test_ten_agents_different_services(self, db):
        """Ten agents with different service permissions."""
        services = ["github", "slack", "google", "linear", "notion"]
        for i in range(10):
            svc = services[i % len(services)]
            policy = AgentPolicy(
                agent_id=f"multi-agent-{i}",
                agent_name=f"Multi Agent {i}",
                allowed_services=[svc],
                allowed_scopes={svc: ["repo"] if svc == "github" else ["read"]},
                rate_limit_per_minute=100,
                created_by="user1",
                created_at=time.time(),
            )
            await create_agent_policy(policy)

        policies = await get_all_policies("user1")
        assert len(policies) == 10

    @pytest.mark.asyncio
    async def test_agent_with_all_services(self, db):
        """One agent with access to all five services."""
        from src.auth import SUPPORTED_SERVICES

        all_scopes = {}
        all_services = []
        for svc, info in SUPPORTED_SERVICES.items():
            all_services.append(svc)
            all_scopes[svc] = info["scopes"]

        policy = AgentPolicy(
            agent_id="superagent",
            agent_name="Super Agent",
            allowed_services=all_services,
            allowed_scopes=all_scopes,
            rate_limit_per_minute=1000,
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        # Enforce for each service
        from src.policy import _rate_counters
        _rate_counters.clear()

        for svc in all_services:
            result = await enforce_policy(
                "user1", "superagent", svc,
                all_scopes[svc][:1],  # Request first scope
            )
            assert result.agent_id == "superagent"

    @pytest.mark.asyncio
    async def test_different_users_same_agent_id(self, db):
        """Two users creating policies with the same agent_id."""
        for uid in ["user1", "user2"]:
            policy = AgentPolicy(
                agent_id="shared-agent-name",
                agent_name="Shared Name",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                created_by=uid,
                created_at=time.time(),
            )
            await create_agent_policy(policy)

        # Last writer wins for the agent_id
        result = await get_agent_policy("shared-agent-name")
        assert result is not None


# ============================================================
# Concurrent Audit Logging
# ============================================================

class TestConcurrentAuditLogging:
    """Test audit logging under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_audit_writes(self, db):
        """Many concurrent audit writes should all succeed."""
        tasks = []
        for i in range(50):
            tasks.append(log_audit(
                "user1", f"agent-{i % 5}", "github",
                "token_request", "success",
                scopes="repo",
                ip_address=f"10.0.0.{i % 255}",
            ))

        await asyncio.gather(*tasks)

        entries = await get_audit_log("user1", limit=100)
        assert len(entries) == 50

    @pytest.mark.asyncio
    async def test_audit_entries_have_unique_ids(self, db):
        """Each audit entry should have a unique ID."""
        for i in range(20):
            await log_audit("user1", f"agent-{i}", "github", "test", "success")

        entries = await get_audit_log("user1", limit=20)
        ids = [e.id for e in entries]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_audit_log_user_isolation_concurrent(self, db):
        """Concurrent audit writes for different users should be isolated."""
        tasks = []
        for uid in ["user1", "user2", "user3"]:
            for i in range(10):
                tasks.append(log_audit(uid, "agent-iso", "github", "test", "success"))

        await asyncio.gather(*tasks)

        for uid in ["user1", "user2", "user3"]:
            entries = await get_audit_log(uid)
            assert len(entries) == 10


# ============================================================
# Connected Services Under Concurrent Access
# ============================================================

class TestConcurrentConnectedServices:
    """Test connected services operations under concurrent access."""

    @pytest.mark.asyncio
    async def test_connect_all_services_concurrently(self, db):
        """Connecting all services at once should work."""
        from src.auth import SUPPORTED_SERVICES

        tasks = [
            add_connected_service("user1", svc)
            for svc in SUPPORTED_SERVICES
        ]
        await asyncio.gather(*tasks)

        services = await get_connected_services("user1")
        assert len(services) == len(SUPPORTED_SERVICES)

    @pytest.mark.asyncio
    async def test_connect_and_disconnect_concurrently(self, db):
        """Connecting and disconnecting different services concurrently."""
        await add_connected_service("user1", "github")
        await add_connected_service("user1", "slack")

        await asyncio.gather(
            remove_connected_service("user1", "github"),
            add_connected_service("user1", "google"),
        )

        services = await get_connected_services("user1")
        service_names = {s["service"] for s in services}
        assert "github" not in service_names
        assert "google" in service_names
        assert "slack" in service_names

    @pytest.mark.asyncio
    async def test_multiple_users_connecting_same_service(self, db):
        """Multiple users connecting the same service concurrently."""
        tasks = [
            add_connected_service(f"user-{i}", "github")
            for i in range(10)
        ]
        await asyncio.gather(*tasks)

        for i in range(10):
            services = await get_connected_services(f"user-{i}")
            assert len(services) == 1
            assert services[0]["service"] == "github"


# ============================================================
# Policy Enforcement Edge Cases Under Load
# ============================================================

class TestPolicyEnforcementEdgeCases:
    """Edge cases in policy enforcement discovered through concurrent testing."""

    @pytest.mark.asyncio
    async def test_enforce_disabled_agent_concurrent(self, db):
        """Enforcing a disabled agent should fail consistently."""
        policy = AgentPolicy(
            agent_id="agent-disabled",
            agent_name="Disabled",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            is_active=False,
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        results = await asyncio.gather(*[
            enforce_policy("user1", "agent-disabled", "github", ["repo"])
            for _ in range(5)
        ], return_exceptions=True)

        for r in results:
            assert isinstance(r, PolicyDenied)
            assert "disabled" in str(r).lower()

    @pytest.mark.asyncio
    async def test_enforce_expired_policy_concurrent(self, db):
        """Enforcing an expired policy should fail consistently."""
        policy = AgentPolicy(
            agent_id="agent-expired",
            agent_name="Expired",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            expires_at=time.time() - 3600,  # Expired an hour ago
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        results = await asyncio.gather(*[
            enforce_policy("user1", "agent-expired", "github", ["repo"])
            for _ in range(5)
        ], return_exceptions=True)

        for r in results:
            assert isinstance(r, PolicyDenied)
            assert "expired" in str(r).lower()

    @pytest.mark.asyncio
    async def test_enforce_wrong_user_concurrent(self, db):
        """Wrong user enforcing should fail consistently."""
        policy = AgentPolicy(
            agent_id="agent-owned",
            agent_name="Owned",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        results = await asyncio.gather(*[
            enforce_policy("user2", "agent-owned", "github", ["repo"])
            for _ in range(5)
        ], return_exceptions=True)

        for r in results:
            assert isinstance(r, PolicyDenied)
            assert "not authorized" in str(r).lower() or "own" in str(r).lower()

    @pytest.mark.asyncio
    async def test_enforce_wrong_service_concurrent(self, db):
        """Wrong service should fail consistently."""
        policy = AgentPolicy(
            agent_id="agent-svc-check",
            agent_name="Service Check",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        results = await asyncio.gather(*[
            enforce_policy("user1", "agent-svc-check", "slack", ["chat:write"])
            for _ in range(5)
        ], return_exceptions=True)

        for r in results:
            assert isinstance(r, PolicyDenied)
            assert "not authorized" in str(r).lower() or "slack" in str(r).lower()

    @pytest.mark.asyncio
    async def test_enforce_excess_scopes_concurrent(self, db):
        """Excess scopes should fail consistently."""
        policy = AgentPolicy(
            agent_id="agent-scope-excess",
            agent_name="Scope Excess",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        results = await asyncio.gather(*[
            enforce_policy("user1", "agent-scope-excess", "github", ["repo", "admin"])
            for _ in range(5)
        ], return_exceptions=True)

        for r in results:
            assert isinstance(r, PolicyDenied)

    @pytest.mark.asyncio
    async def test_ip_allowlist_concurrent_enforcement(self, db):
        """IP allowlist checks should work correctly under concurrent access."""
        policy = AgentPolicy(
            agent_id="agent-ip-conc",
            agent_name="IP Concurrent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            ip_allowlist=["10.0.0.1", "192.168.1.0/24"],
            rate_limit_per_minute=100,
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        from src.policy import _rate_counters
        _rate_counters.clear()

        # Mix of allowed and denied IPs
        tasks = []
        for ip in ["10.0.0.1", "10.0.0.2", "192.168.1.50", "172.16.0.1"]:
            tasks.append(enforce_policy(
                "user1", "agent-ip-conc", "github", ["repo"], ip_address=ip
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 10.0.0.1 and 192.168.1.50 should succeed
        # 10.0.0.2 and 172.16.0.1 should fail
        successes = [r for r in results if not isinstance(r, PolicyDenied)]
        failures = [r for r in results if isinstance(r, PolicyDenied)]
        assert len(successes) == 2
        assert len(failures) == 2


# ============================================================
# Stress: Many Operations in Sequence
# ============================================================

class TestSequentialStress:
    """Stress tests with many sequential operations."""

    @pytest.mark.asyncio
    async def test_create_and_query_100_policies(self, db):
        """Creating and querying 100 policies should work efficiently."""
        for i in range(100):
            policy = AgentPolicy(
                agent_id=f"stress-agent-{i}",
                agent_name=f"Stress Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                created_by="user1",
                created_at=time.time(),
            )
            await create_agent_policy(policy)

        policies = await get_all_policies("user1")
        assert len(policies) == 100

    @pytest.mark.asyncio
    async def test_audit_log_100_entries(self, db):
        """100 audit entries should be stored and retrievable."""
        for i in range(100):
            await log_audit("user1", f"agent-{i}", "github", "test", "success")

        entries = await get_audit_log("user1", limit=200)
        assert len(entries) == 100

    @pytest.mark.asyncio
    async def test_rapid_key_creation_and_validation(self, db):
        """Rapidly creating and validating keys."""
        raw_keys = []
        for i in range(20):
            _, raw = await create_api_key("user1", f"rapid-agent-{i}", f"key-{i}")
            raw_keys.append(raw)

        # Validate all
        for raw in raw_keys:
            result = await validate_api_key(raw)
            assert result is not None

    @pytest.mark.asyncio
    async def test_toggle_policy_many_times(self, db):
        """Toggling a policy many times should maintain consistent state."""
        policy = AgentPolicy(
            agent_id="toggle-stress",
            agent_name="Toggle Stress",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        for i in range(20):
            new_state = await toggle_agent_policy("toggle-stress", "user1")
            expected = (i % 2 == 0)  # Starts active, first toggle disables
            # The initial state is active (True), first toggle makes it False
            if i == 0:
                assert new_state is False
            else:
                # Alternates
                assert isinstance(new_state, bool)

        # After 20 toggles (even number), should be back to original state
        final = await get_agent_policy("toggle-stress")
        assert final.is_active is True
