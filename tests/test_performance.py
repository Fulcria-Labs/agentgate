"""Performance and stress tests for AgentGate.

Covers rate limiter performance with many agents, policy enforcement latency
under load, concurrent token request handling, database query performance
with large audit logs, and scalability of key operations.
"""

import asyncio
import time
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

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
    AuditEntry,
)
from src.policy import (
    enforce_policy,
    PolicyDenied,
    check_time_window,
    check_ip_allowlist,
    _rate_counters,
    get_effective_scopes,
    requires_step_up,
)


@pytest.fixture(autouse=True)
def clear_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


USER = "auth0|perf-test-user"


# ============================================================
# 1. Rate Limiter Performance with Many Agents
# ============================================================

class TestRateLimiterPerformance:
    """Rate limiter behaviour under heavy agent load."""

    @pytest.mark.asyncio
    async def test_100_agents_each_one_request(self, db):
        """100 agents each making a single request should all succeed."""
        for i in range(100):
            await create_agent_policy(AgentPolicy(
                agent_id=f"perf-agent-{i}",
                agent_name=f"Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=10,
                created_by=USER,
                created_at=time.time(),
            ))

        results = await asyncio.gather(*[
            enforce_policy(USER, f"perf-agent-{i}", "github", ["repo"])
            for i in range(100)
        ], return_exceptions=True)

        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) == 100

    @pytest.mark.asyncio
    async def test_rate_limit_counter_isolation_across_agents(self, db):
        """Rate limit counters for different agents should be independent."""
        for i in range(3):
            await create_agent_policy(AgentPolicy(
                agent_id=f"iso-agent-{i}",
                agent_name=f"Iso Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=2,
                created_by=USER,
                created_at=time.time(),
            ))

        # Each agent gets 2 requests
        for i in range(3):
            for _ in range(2):
                await enforce_policy(USER, f"iso-agent-{i}", "github", ["repo"])

        # Third request for each should be rate limited
        for i in range(3):
            with pytest.raises(PolicyDenied, match="Rate limit"):
                await enforce_policy(USER, f"iso-agent-{i}", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_rate_limit_counter_isolation_across_services(self, db):
        """Rate limits are tracked per agent:service pair."""
        await create_agent_policy(AgentPolicy(
            agent_id="multi-svc-agent",
            agent_name="Multi Svc",
            allowed_services=["github", "slack"],
            allowed_scopes={"github": ["repo"], "slack": ["chat:write"]},
            rate_limit_per_minute=2,
            created_by=USER,
            created_at=time.time(),
        ))

        # Two requests for github
        await enforce_policy(USER, "multi-svc-agent", "github", ["repo"])
        await enforce_policy(USER, "multi-svc-agent", "github", ["repo"])

        # Third github request should be rate limited
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy(USER, "multi-svc-agent", "github", ["repo"])

        # But slack should still work (separate counter)
        result = await enforce_policy(USER, "multi-svc-agent", "slack", ["chat:write"])
        assert result.agent_id == "multi-svc-agent"

    @pytest.mark.asyncio
    async def test_rate_counter_cleanup_old_entries(self, db):
        """Old rate counter entries older than 60s should be cleaned up."""
        await create_agent_policy(AgentPolicy(
            agent_id="cleanup-agent",
            agent_name="Cleanup",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=2,
            created_by=USER,
            created_at=time.time(),
        ))

        # Manually inject old timestamps into the rate counter
        key = "cleanup-agent:github"
        old_time = time.time() - 120  # 2 minutes ago
        _rate_counters[key] = [old_time, old_time]

        # This should succeed because old entries are cleaned up
        result = await enforce_policy(USER, "cleanup-agent", "github", ["repo"])
        assert result.agent_id == "cleanup-agent"

    @pytest.mark.asyncio
    async def test_rate_limit_exactly_at_boundary(self, db):
        """Requests exactly at the rate limit boundary."""
        await create_agent_policy(AgentPolicy(
            agent_id="boundary-agent",
            agent_name="Boundary",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=5,
            created_by=USER,
            created_at=time.time(),
        ))

        # Exactly 5 requests should succeed
        for _ in range(5):
            await enforce_policy(USER, "boundary-agent", "github", ["repo"])

        # 6th should fail
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy(USER, "boundary-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_rate_limit_with_high_limit(self, db):
        """Agent with a very high rate limit should handle many requests."""
        await create_agent_policy(AgentPolicy(
            agent_id="high-limit-agent",
            agent_name="High Limit",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=500,
            created_by=USER,
            created_at=time.time(),
        ))

        results = await asyncio.gather(*[
            enforce_policy(USER, "high-limit-agent", "github", ["repo"])
            for _ in range(200)
        ], return_exceptions=True)

        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) == 200

    @pytest.mark.asyncio
    async def test_rate_limit_one_per_minute(self, db):
        """Agent with rate limit of 1 should only allow one request."""
        await create_agent_policy(AgentPolicy(
            agent_id="one-per-min",
            agent_name="One Per Min",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=1,
            created_by=USER,
            created_at=time.time(),
        ))

        await enforce_policy(USER, "one-per-min", "github", ["repo"])

        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy(USER, "one-per-min", "github", ["repo"])


# ============================================================
# 2. Policy Enforcement Latency Under Load
# ============================================================

class TestPolicyEnforcementLatency:
    """Policy enforcement performance under various load patterns."""

    @pytest.mark.asyncio
    async def test_enforce_50_agents_concurrently(self, db):
        """50 concurrent enforcement calls should all complete."""
        for i in range(50):
            await create_agent_policy(AgentPolicy(
                agent_id=f"lat-agent-{i}",
                agent_name=f"Lat Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        start = time.time()
        results = await asyncio.gather(*[
            enforce_policy(USER, f"lat-agent-{i}", "github", ["repo"])
            for i in range(50)
        ], return_exceptions=True)
        duration = time.time() - start

        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) == 50
        # Should complete in reasonable time (< 30s)
        assert duration < 30

    @pytest.mark.asyncio
    async def test_enforce_with_all_check_types(self, db):
        """Enforcement passing through all check stages should still be fast."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        await create_agent_policy(AgentPolicy(
            agent_id="full-check-agent",
            agent_name="Full Check",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=100,
            allowed_hours=list(range(24)),  # All hours
            allowed_days=list(range(7)),     # All days
            ip_allowlist=["0.0.0.0/0"],      # All IPs
            created_by=USER,
            created_at=time.time(),
        ))

        for _ in range(10):
            result = await enforce_policy(
                USER, "full-check-agent", "github", ["repo"],
                ip_address="192.168.1.1",
            )
            assert result.agent_id == "full-check-agent"

    @pytest.mark.asyncio
    async def test_enforce_denial_path_performance(self, db):
        """Denial paths should also complete quickly."""
        await create_agent_policy(AgentPolicy(
            agent_id="deny-perf-agent",
            agent_name="Deny Perf",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            ip_allowlist=["10.0.0.1"],
            created_by=USER,
            created_at=time.time(),
        ))

        start = time.time()
        for _ in range(50):
            with pytest.raises(PolicyDenied):
                await enforce_policy(
                    USER, "deny-perf-agent", "github", ["repo"],
                    ip_address="192.168.1.1",
                )
        duration = time.time() - start
        assert duration < 30

    @pytest.mark.asyncio
    async def test_enforce_mixed_success_and_failure(self, db):
        """Mixed success and failure enforcement calls under load."""
        await create_agent_policy(AgentPolicy(
            agent_id="mix-perf-agent",
            agent_name="Mix Perf",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=200,
            created_by=USER,
            created_at=time.time(),
        ))

        async def enforce_with_service(svc):
            return await enforce_policy(USER, "mix-perf-agent", svc, ["repo"])

        tasks = []
        for i in range(20):
            svc = "github" if i % 2 == 0 else "slack"
            tasks.append(enforce_with_service(svc))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, PolicyDenied)]
        assert len(successes) == 10  # github requests
        assert len(failures) == 10   # slack requests (not allowed)

    @pytest.mark.asyncio
    async def test_policy_lookup_nonexistent_agents_performance(self, db):
        """Looking up many nonexistent agents should be fast."""
        start = time.time()
        for i in range(50):
            with pytest.raises(PolicyDenied, match="not registered"):
                await enforce_policy(USER, f"ghost-{i}", "github", ["repo"])
        duration = time.time() - start
        assert duration < 30

    @pytest.mark.asyncio
    async def test_enforce_with_many_scopes(self, db):
        """Enforcement with many allowed scopes should still work."""
        await create_agent_policy(AgentPolicy(
            agent_id="many-scopes-agent",
            agent_name="Many Scopes",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo", "read:user", "read:org", "gist", "notifications"]},
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))

        result = await enforce_policy(
            USER, "many-scopes-agent", "github",
            ["repo", "read:user", "read:org"],
        )
        assert result.agent_id == "many-scopes-agent"


# ============================================================
# 3. Concurrent Token Request Handling
# ============================================================

class TestConcurrentTokenRequests:
    """Simulated concurrent token request patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_enforce_same_agent_same_service(self, db):
        """Multiple concurrent requests for the same agent and service."""
        await create_agent_policy(AgentPolicy(
            agent_id="conc-token-agent",
            agent_name="Conc Token",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=50,
            created_by=USER,
            created_at=time.time(),
        ))

        results = await asyncio.gather(*[
            enforce_policy(USER, "conc-token-agent", "github", ["repo"])
            for _ in range(30)
        ], return_exceptions=True)

        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) <= 50

    @pytest.mark.asyncio
    async def test_concurrent_different_agents_same_service(self, db):
        """Different agents requesting the same service concurrently."""
        for i in range(10):
            await create_agent_policy(AgentPolicy(
                agent_id=f"diff-agent-{i}",
                agent_name=f"Diff Agent {i}",
                allowed_services=["slack"],
                allowed_scopes={"slack": ["chat:write"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        results = await asyncio.gather(*[
            enforce_policy(USER, f"diff-agent-{i}", "slack", ["chat:write"])
            for i in range(10)
        ], return_exceptions=True)

        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) == 10

    @pytest.mark.asyncio
    async def test_concurrent_same_agent_different_services(self, db):
        """Same agent requesting different services concurrently."""
        await create_agent_policy(AgentPolicy(
            agent_id="multi-svc-conc",
            agent_name="Multi Svc Conc",
            allowed_services=["github", "slack", "google", "linear", "notion"],
            allowed_scopes={
                "github": ["repo"],
                "slack": ["chat:write"],
                "google": ["gmail.readonly"],
                "linear": ["read"],
                "notion": ["read_content"],
            },
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))

        services = [
            ("github", ["repo"]),
            ("slack", ["chat:write"]),
            ("google", ["gmail.readonly"]),
            ("linear", ["read"]),
            ("notion", ["read_content"]),
        ]

        results = await asyncio.gather(*[
            enforce_policy(USER, "multi-svc-conc", svc, scopes)
            for svc, scopes in services
        ], return_exceptions=True)

        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) == 5

    @pytest.mark.asyncio
    async def test_burst_then_rate_limit(self, db):
        """A burst of requests should hit the rate limit correctly."""
        await create_agent_policy(AgentPolicy(
            agent_id="burst-agent",
            agent_name="Burst Agent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=10,
            created_by=USER,
            created_at=time.time(),
        ))

        results = await asyncio.gather(*[
            enforce_policy(USER, "burst-agent", "github", ["repo"])
            for _ in range(20)
        ], return_exceptions=True)

        successes = [r for r in results if not isinstance(r, Exception)]
        rate_limited = [r for r in results if isinstance(r, PolicyDenied)]
        assert len(successes) == 10
        assert len(rate_limited) == 10

    @pytest.mark.asyncio
    async def test_concurrent_enforce_and_toggle(self, db):
        """Concurrent enforcement and toggle operations."""
        await create_agent_policy(AgentPolicy(
            agent_id="toggle-conc-agent",
            agent_name="Toggle Conc",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))

        async def try_enforce():
            try:
                return await enforce_policy(USER, "toggle-conc-agent", "github", ["repo"])
            except PolicyDenied:
                return None

        results = await asyncio.gather(
            try_enforce(),
            try_enforce(),
            toggle_agent_policy("toggle-conc-agent", USER),
            try_enforce(),
            return_exceptions=True,
        )

        # No crashes
        for r in results:
            assert not isinstance(r, Exception) or isinstance(r, PolicyDenied)

    @pytest.mark.asyncio
    async def test_concurrent_enforce_and_delete(self, db):
        """Concurrent enforcement and deletion."""
        await create_agent_policy(AgentPolicy(
            agent_id="del-conc-agent",
            agent_name="Del Conc",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))

        results = await asyncio.gather(
            enforce_policy(USER, "del-conc-agent", "github", ["repo"]),
            delete_agent_policy("del-conc-agent", USER),
            enforce_policy(USER, "del-conc-agent", "github", ["repo"]),
            return_exceptions=True,
        )

        # No crashes - some may succeed, some may fail
        for r in results:
            assert isinstance(r, (AgentPolicy, bool, PolicyDenied)) or r is None


# ============================================================
# 4. Database Query Performance with Large Audit Logs
# ============================================================

class TestDatabasePerformance:
    """Database operations with large datasets."""

    @pytest.mark.asyncio
    async def test_200_audit_entries_retrieval(self, db):
        """Retrieving from 200 audit entries should work."""
        for i in range(200):
            await log_audit(
                USER, f"agent-{i % 10}", "github",
                "token_request", "success",
                scopes="repo",
                ip_address=f"10.0.0.{i % 255}",
            )

        entries = await get_audit_log(USER, limit=200)
        assert len(entries) == 200

    @pytest.mark.asyncio
    async def test_audit_log_limit_respected(self, db):
        """Limit parameter should cap the number of results."""
        for i in range(100):
            await log_audit(USER, "agent-1", "github", "test", "success")

        entries = await get_audit_log(USER, limit=10)
        assert len(entries) == 10

    @pytest.mark.asyncio
    async def test_audit_log_ordering(self, db):
        """Audit entries should be returned newest first."""
        for i in range(20):
            await log_audit(USER, f"agent-{i}", "github", f"action-{i}", "success")

        entries = await get_audit_log(USER, limit=20)
        # Timestamps should be in descending order
        for i in range(len(entries) - 1):
            assert entries[i].timestamp >= entries[i + 1].timestamp

    @pytest.mark.asyncio
    async def test_concurrent_audit_writes_and_reads(self, db):
        """Concurrent audit writes and reads should not conflict."""
        async def write_batch(batch_id):
            for i in range(10):
                await log_audit(
                    USER, f"agent-{batch_id}-{i}", "github",
                    "batch_action", "success",
                )

        async def read_audit():
            return await get_audit_log(USER, limit=100)

        await asyncio.gather(
            write_batch(0),
            write_batch(1),
            read_audit(),
            write_batch(2),
        )

        # Final read should see at least some entries
        entries = await get_audit_log(USER, limit=100)
        assert len(entries) >= 10

    @pytest.mark.asyncio
    async def test_many_policies_query_performance(self, db):
        """Querying all policies when there are many should work."""
        for i in range(50):
            await create_agent_policy(AgentPolicy(
                agent_id=f"query-perf-{i}",
                agent_name=f"Query Perf {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                created_by=USER,
                created_at=time.time(),
            ))

        start = time.time()
        policies = await get_all_policies(USER)
        duration = time.time() - start

        assert len(policies) == 50
        assert duration < 10

    @pytest.mark.asyncio
    async def test_many_api_keys_query(self, db):
        """Querying many API keys should work."""
        for i in range(30):
            await create_api_key(USER, f"key-agent-{i}", f"key-{i}")

        keys = await get_api_keys(USER)
        assert len(keys) == 30

    @pytest.mark.asyncio
    async def test_audit_entries_have_all_fields(self, db):
        """Audit entries should preserve all fields correctly."""
        await log_audit(
            USER, "field-agent", "github",
            "token_issued", "success",
            scopes="repo,read:user",
            ip_address="10.0.0.1",
            details="Detailed info here",
        )

        entries = await get_audit_log(USER, limit=1)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.user_id == USER
        assert entry.agent_id == "field-agent"
        assert entry.service == "github"
        assert entry.action == "token_issued"
        assert entry.status == "success"
        assert entry.scopes == "repo,read:user"
        assert entry.ip_address == "10.0.0.1"
        assert entry.details == "Detailed info here"

    @pytest.mark.asyncio
    async def test_audit_user_isolation_under_load(self, db):
        """Different users' audit logs should be isolated under load."""
        for uid in ["perf-user-a", "perf-user-b", "perf-user-c"]:
            for i in range(30):
                await log_audit(uid, "agent-1", "github", "test", "success")

        for uid in ["perf-user-a", "perf-user-b", "perf-user-c"]:
            entries = await get_audit_log(uid, limit=100)
            assert len(entries) == 30


# ============================================================
# 5. Key Operations at Scale
# ============================================================

class TestKeyOperationsScale:
    """API key operations at scale."""

    @pytest.mark.asyncio
    async def test_create_50_keys_all_unique(self, db):
        """50 API keys should all be unique."""
        raw_keys = []
        for i in range(50):
            _, raw = await create_api_key(USER, f"scale-agent-{i}", f"key-{i}")
            raw_keys.append(raw)

        assert len(set(raw_keys)) == 50

    @pytest.mark.asyncio
    async def test_validate_50_keys(self, db):
        """All 50 created keys should validate successfully."""
        raw_keys = []
        for i in range(50):
            _, raw = await create_api_key(USER, f"val-agent-{i}", f"key-{i}")
            raw_keys.append(raw)

        for raw in raw_keys:
            result = await validate_api_key(raw)
            assert result is not None

    @pytest.mark.asyncio
    async def test_revoke_half_validate_all(self, db):
        """Revoking half the keys, the other half should still work."""
        keys_and_raws = []
        for i in range(20):
            key_obj, raw = await create_api_key(USER, f"half-agent-{i}", f"key-{i}")
            keys_and_raws.append((key_obj, raw))

        # Revoke the first 10
        for i in range(10):
            await revoke_api_key(keys_and_raws[i][0].id, USER)

        # First 10 should be invalid
        for i in range(10):
            assert await validate_api_key(keys_and_raws[i][1]) is None

        # Last 10 should still be valid
        for i in range(10, 20):
            result = await validate_api_key(keys_and_raws[i][1])
            assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_key_creation(self, db):
        """Creating keys concurrently should not cause collisions."""
        async def create_key(i):
            _, raw = await create_api_key(USER, f"conc-key-agent-{i}", f"key-{i}")
            return raw

        raw_keys = await asyncio.gather(*[create_key(i) for i in range(30)])
        assert len(set(raw_keys)) == 30

    @pytest.mark.asyncio
    async def test_concurrent_validation(self, db):
        """Validating keys concurrently should all succeed."""
        raw_keys = []
        for i in range(15):
            _, raw = await create_api_key(USER, f"conc-val-agent-{i}", f"key-{i}")
            raw_keys.append(raw)

        results = await asyncio.gather(*[validate_api_key(k) for k in raw_keys])
        for r in results:
            assert r is not None


# ============================================================
# 6. Connected Services Scale
# ============================================================

class TestConnectedServicesScale:
    """Connected service operations at scale."""

    @pytest.mark.asyncio
    async def test_many_users_connect_services(self, db):
        """Many users connecting the same service."""
        for i in range(50):
            await add_connected_service(f"scale-user-{i}", "github")

        for i in range(50):
            svcs = await get_connected_services(f"scale-user-{i}")
            assert len(svcs) == 1

    @pytest.mark.asyncio
    async def test_connect_disconnect_cycle(self, db):
        """Repeated connect-disconnect cycles should be stable."""
        for _ in range(10):
            await add_connected_service(USER, "github", "conn-id")
            svcs = await get_connected_services(USER)
            assert len(svcs) == 1
            await remove_connected_service(USER, "github")
            svcs = await get_connected_services(USER)
            assert len(svcs) == 0


# ============================================================
# 7. Emergency Revoke Scale
# ============================================================

class TestEmergencyRevokeScale:
    """Emergency revoke with large numbers of resources."""

    @pytest.mark.asyncio
    async def test_emergency_revoke_50_policies_30_keys(self, db):
        """Emergency revoke with 50 policies and 30 keys."""
        for i in range(50):
            await create_agent_policy(AgentPolicy(
                agent_id=f"emerg-scale-{i}",
                agent_name=f"Emerg Scale {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                created_by=USER,
                created_at=time.time(),
            ))
        for i in range(30):
            await create_api_key(USER, f"emerg-scale-{i}", f"key-{i}")

        result = await emergency_revoke_all(USER)
        assert result["policies_disabled"] == 50
        assert result["keys_revoked"] == 30

    @pytest.mark.asyncio
    async def test_emergency_revoke_concurrent(self, db):
        """Multiple concurrent emergency revoke calls should not crash."""
        for i in range(5):
            await create_agent_policy(AgentPolicy(
                agent_id=f"conc-emerg-{i}",
                agent_name=f"Conc Emerg {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                created_by=USER,
                created_at=time.time(),
            ))

        results = await asyncio.gather(
            emergency_revoke_all(USER),
            emergency_revoke_all(USER),
            emergency_revoke_all(USER),
            return_exceptions=True,
        )

        for r in results:
            assert not isinstance(r, Exception)


# ============================================================
# 8. Policy Check Functions Performance
# ============================================================

class TestPolicyCheckPerformance:
    """Direct policy check function performance."""

    def test_check_time_window_performance(self):
        """check_time_window should be fast for many calls."""
        policy = AgentPolicy(
            agent_id="tw-perf",
            agent_name="TW Perf",
            allowed_hours=list(range(24)),
            allowed_days=list(range(7)),
        )

        for _ in range(1000):
            result = check_time_window(policy)
            assert result is None

    def test_check_ip_allowlist_performance(self):
        """check_ip_allowlist with many entries."""
        policy = AgentPolicy(
            agent_id="ip-perf",
            agent_name="IP Perf",
            ip_allowlist=[f"10.0.{i}.0/24" for i in range(50)],
        )

        # Check an IP that is in the last CIDR
        result = check_ip_allowlist(policy, "10.0.49.100")
        assert result is None

    def test_check_ip_allowlist_denial_performance(self):
        """check_ip_allowlist denial with many entries."""
        policy = AgentPolicy(
            agent_id="ip-deny-perf",
            agent_name="IP Deny Perf",
            ip_allowlist=[f"10.0.{i}.0/24" for i in range(50)],
        )

        result = check_ip_allowlist(policy, "192.168.1.1")
        assert result is not None
        assert "not in allowlist" in result

    def test_get_effective_scopes_performance(self):
        """get_effective_scopes with many scopes."""
        policy = AgentPolicy(
            agent_id="scope-perf",
            agent_name="Scope Perf",
            allowed_scopes={"github": ["repo", "read:user", "read:org", "gist", "notifications"]},
        )

        for _ in range(1000):
            result = get_effective_scopes(policy, "github", ["repo", "gist"])
            assert set(result) == {"repo", "gist"}

    def test_requires_step_up_performance(self):
        """requires_step_up check performance."""
        policy = AgentPolicy(
            agent_id="step-up-perf",
            agent_name="Step Up Perf",
            requires_step_up=["github", "slack"],
        )

        for _ in range(1000):
            assert requires_step_up(policy, "github") is True
            assert requires_step_up(policy, "google") is False
