"""Advanced multi-agent orchestration tests for AgentGate.

Covers multiple agents requesting tokens simultaneously, agent dependency
chains, cross-agent policy conflicts, cascading emergency revoke across
agents, and agent group policies.
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
)
from src.policy import (
    enforce_policy,
    PolicyDenied,
    check_time_window,
    check_ip_allowlist,
    get_effective_scopes,
    requires_step_up,
    _rate_counters,
)
from src.auth import SUPPORTED_SERVICES


@pytest.fixture(autouse=True)
def clear_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


USER = "auth0|multi-agent-user"
USER2 = "auth0|multi-agent-user-2"


# ============================================================
# 1. Multiple Agents Requesting Tokens Simultaneously
# ============================================================

class TestSimultaneousTokenRequests:
    """Multiple agents requesting access concurrently."""

    @pytest.mark.asyncio
    async def test_five_agents_simultaneous_github(self, db):
        """Five agents requesting GitHub tokens simultaneously."""
        for i in range(5):
            await create_agent_policy(AgentPolicy(
                agent_id=f"sim-agent-{i}",
                agent_name=f"Sim Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        results = await asyncio.gather(*[
            enforce_policy(USER, f"sim-agent-{i}", "github", ["repo"])
            for i in range(5)
        ])

        assert all(r.agent_id.startswith("sim-agent-") for r in results)

    @pytest.mark.asyncio
    async def test_agents_requesting_different_services(self, db):
        """Each agent requests a different service simultaneously."""
        services = ["github", "slack", "google", "linear", "notion"]
        scopes = {
            "github": ["repo"],
            "slack": ["chat:write"],
            "google": ["gmail.readonly"],
            "linear": ["read"],
            "notion": ["read_content"],
        }

        for i, svc in enumerate(services):
            await create_agent_policy(AgentPolicy(
                agent_id=f"svc-agent-{i}",
                agent_name=f"Svc Agent {i}",
                allowed_services=[svc],
                allowed_scopes={svc: scopes[svc]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        results = await asyncio.gather(*[
            enforce_policy(USER, f"svc-agent-{i}", services[i], scopes[services[i]])
            for i in range(5)
        ])

        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.agent_id == f"svc-agent-{i}"

    @pytest.mark.asyncio
    async def test_twenty_agents_burst_request(self, db):
        """Twenty agents making requests in a burst."""
        for i in range(20):
            await create_agent_policy(AgentPolicy(
                agent_id=f"burst-agent-{i}",
                agent_name=f"Burst Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        results = await asyncio.gather(*[
            enforce_policy(USER, f"burst-agent-{i}", "github", ["repo"])
            for i in range(20)
        ], return_exceptions=True)

        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) == 20

    @pytest.mark.asyncio
    async def test_agents_with_different_rate_limits(self, db):
        """Agents with different rate limits operating simultaneously."""
        limits = [1, 5, 10, 50, 100]
        for i, limit in enumerate(limits):
            await create_agent_policy(AgentPolicy(
                agent_id=f"rl-varied-{i}",
                agent_name=f"RL Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=limit,
                created_by=USER,
                created_at=time.time(),
            ))

        # Each agent makes 3 requests
        async def multi_request(agent_id):
            results = []
            for _ in range(3):
                try:
                    r = await enforce_policy(USER, agent_id, "github", ["repo"])
                    results.append(True)
                except PolicyDenied:
                    results.append(False)
            return results

        all_results = await asyncio.gather(*[
            multi_request(f"rl-varied-{i}") for i in range(5)
        ])

        # Agent with limit=1 should fail on 2nd and 3rd
        assert all_results[0] == [True, False, False]
        # Agents with higher limits should succeed on all 3
        for i in range(1, 5):
            assert all_results[i] == [True, True, True]

    @pytest.mark.asyncio
    async def test_mixed_allowed_and_denied_agents(self, db):
        """Mix of agents: some allowed, some denied."""
        # Allowed agents
        for i in range(3):
            await create_agent_policy(AgentPolicy(
                agent_id=f"allowed-agent-{i}",
                agent_name=f"Allowed Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        # Disabled agents
        for i in range(3):
            await create_agent_policy(AgentPolicy(
                agent_id=f"denied-agent-{i}",
                agent_name=f"Denied Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                is_active=False,
                created_by=USER,
                created_at=time.time(),
            ))

        tasks = []
        for i in range(3):
            tasks.append(enforce_policy(USER, f"allowed-agent-{i}", "github", ["repo"]))
            tasks.append(enforce_policy(USER, f"denied-agent-{i}", "github", ["repo"]))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, PolicyDenied)]
        assert len(successes) == 3
        assert len(failures) == 3


# ============================================================
# 2. Agent Dependency Chains
# ============================================================

class TestAgentDependencyChains:
    """Agent A needs result from agent B before proceeding."""

    @pytest.mark.asyncio
    async def test_sequential_agent_chain(self, db):
        """Agent B depends on agent A completing first."""
        for agent_id in ["chain-a", "chain-b", "chain-c"]:
            await create_agent_policy(AgentPolicy(
                agent_id=agent_id,
                agent_name=f"Chain {agent_id}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        # Sequential chain: A -> B -> C
        result_a = await enforce_policy(USER, "chain-a", "github", ["repo"])
        assert result_a.agent_id == "chain-a"

        result_b = await enforce_policy(USER, "chain-b", "github", ["repo"])
        assert result_b.agent_id == "chain-b"

        result_c = await enforce_policy(USER, "chain-c", "github", ["repo"])
        assert result_c.agent_id == "chain-c"

    @pytest.mark.asyncio
    async def test_fan_out_agent_chain(self, db):
        """One primary agent triggers multiple downstream agents."""
        await create_agent_policy(AgentPolicy(
            agent_id="primary",
            agent_name="Primary",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))

        for i in range(5):
            await create_agent_policy(AgentPolicy(
                agent_id=f"downstream-{i}",
                agent_name=f"Downstream {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        # Primary runs first
        await enforce_policy(USER, "primary", "github", ["repo"])

        # Then fan-out to all downstream agents
        results = await asyncio.gather(*[
            enforce_policy(USER, f"downstream-{i}", "github", ["repo"])
            for i in range(5)
        ])

        assert all(r.agent_id.startswith("downstream-") for r in results)

    @pytest.mark.asyncio
    async def test_chain_breaks_when_agent_disabled(self, db):
        """Chain should fail if a middle agent is disabled."""
        for agent_id in ["chain-ok-a", "chain-disabled-b", "chain-ok-c"]:
            active = agent_id != "chain-disabled-b"
            await create_agent_policy(AgentPolicy(
                agent_id=agent_id,
                agent_name=f"Chain {agent_id}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                is_active=active,
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        await enforce_policy(USER, "chain-ok-a", "github", ["repo"])

        with pytest.raises(PolicyDenied, match="disabled"):
            await enforce_policy(USER, "chain-disabled-b", "github", ["repo"])

        # C can still run independently
        result = await enforce_policy(USER, "chain-ok-c", "github", ["repo"])
        assert result.agent_id == "chain-ok-c"

    @pytest.mark.asyncio
    async def test_chain_with_different_services(self, db):
        """Chain where each agent accesses a different service."""
        configs = [
            ("reader", "github", ["repo"]),
            ("messenger", "slack", ["chat:write"]),
            ("tracker", "linear", ["read"]),
        ]

        for agent_id, svc, scopes in configs:
            await create_agent_policy(AgentPolicy(
                agent_id=agent_id,
                agent_name=f"Chain {agent_id}",
                allowed_services=[svc],
                allowed_scopes={svc: scopes},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        for agent_id, svc, scopes in configs:
            result = await enforce_policy(USER, agent_id, svc, scopes)
            assert result.agent_id == agent_id

    @pytest.mark.asyncio
    async def test_deep_chain_10_agents(self, db):
        """A chain of 10 agents in sequence."""
        for i in range(10):
            await create_agent_policy(AgentPolicy(
                agent_id=f"deep-chain-{i}",
                agent_name=f"Deep Chain {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        for i in range(10):
            result = await enforce_policy(USER, f"deep-chain-{i}", "github", ["repo"])
            assert result.agent_id == f"deep-chain-{i}"


# ============================================================
# 3. Cross-Agent Policy Conflicts
# ============================================================

class TestCrossAgentPolicyConflicts:
    """Conflicting policies across agents."""

    @pytest.mark.asyncio
    async def test_same_service_different_scopes(self, db):
        """Two agents with different scopes for the same service."""
        await create_agent_policy(AgentPolicy(
            agent_id="scope-narrow",
            agent_name="Narrow Scope",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="scope-wide",
            agent_name="Wide Scope",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo", "read:user", "read:org"]},
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))

        # Narrow can only access repo
        result = await enforce_policy(USER, "scope-narrow", "github", ["repo"])
        assert result.agent_id == "scope-narrow"

        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "scope-narrow", "github", ["repo", "read:user"])

        # Wide can access all three
        result = await enforce_policy(USER, "scope-wide", "github", ["repo", "read:user"])
        assert result.agent_id == "scope-wide"

    @pytest.mark.asyncio
    async def test_same_service_different_rate_limits(self, db):
        """Two agents with different rate limits for the same service."""
        await create_agent_policy(AgentPolicy(
            agent_id="slow-agent",
            agent_name="Slow",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=2,
            created_by=USER,
            created_at=time.time(),
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="fast-agent",
            agent_name="Fast",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))

        # Slow agent hits limit after 2
        await enforce_policy(USER, "slow-agent", "github", ["repo"])
        await enforce_policy(USER, "slow-agent", "github", ["repo"])
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy(USER, "slow-agent", "github", ["repo"])

        # Fast agent can still go
        for _ in range(10):
            await enforce_policy(USER, "fast-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_conflicting_ip_allowlists(self, db):
        """Two agents with non-overlapping IP allowlists."""
        await create_agent_policy(AgentPolicy(
            agent_id="ip-agent-a",
            agent_name="IP Agent A",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            ip_allowlist=["10.0.0.0/24"],
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="ip-agent-b",
            agent_name="IP Agent B",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            ip_allowlist=["192.168.1.0/24"],
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))

        # Agent A allows 10.x but not 192.168.x
        await enforce_policy(USER, "ip-agent-a", "github", ["repo"], ip_address="10.0.0.5")
        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "ip-agent-a", "github", ["repo"], ip_address="192.168.1.5")

        # Agent B allows 192.168.x but not 10.x
        await enforce_policy(USER, "ip-agent-b", "github", ["repo"], ip_address="192.168.1.5")
        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "ip-agent-b", "github", ["repo"], ip_address="10.0.0.5")

    @pytest.mark.asyncio
    async def test_conflicting_time_windows(self, db):
        """Two agents with non-overlapping time windows."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        current_hour = now.hour
        other_hour = (current_hour + 12) % 24  # Opposite time

        await create_agent_policy(AgentPolicy(
            agent_id="day-agent",
            agent_name="Day Agent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            allowed_hours=[current_hour],
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="night-agent",
            agent_name="Night Agent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            allowed_hours=[other_hour],
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))

        # Day agent should work now
        result = await enforce_policy(USER, "day-agent", "github", ["repo"])
        assert result.agent_id == "day-agent"

        # Night agent should be denied now
        with pytest.raises(PolicyDenied, match="hours"):
            await enforce_policy(USER, "night-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_one_expired_one_active(self, db):
        """One agent expired, one still active."""
        await create_agent_policy(AgentPolicy(
            agent_id="expired-conflict",
            agent_name="Expired",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            expires_at=time.time() - 3600,
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="active-conflict",
            agent_name="Active",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))

        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy(USER, "expired-conflict", "github", ["repo"])

        result = await enforce_policy(USER, "active-conflict", "github", ["repo"])
        assert result.agent_id == "active-conflict"


# ============================================================
# 4. Cascading Emergency Revoke Across Agents
# ============================================================

class TestCascadingEmergencyRevoke:
    """Emergency revoke cascading effects across all agents."""

    @pytest.mark.asyncio
    async def test_revoke_disables_all_agents(self, db):
        """Emergency revoke disables every agent for the user."""
        for i in range(10):
            await create_agent_policy(AgentPolicy(
                agent_id=f"cascade-agent-{i}",
                agent_name=f"Cascade Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        result = await emergency_revoke_all(USER)
        assert result["policies_disabled"] == 10

        for i in range(10):
            with pytest.raises(PolicyDenied, match="disabled"):
                await enforce_policy(USER, f"cascade-agent-{i}", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_revoke_invalidates_all_keys(self, db):
        """Emergency revoke invalidates all API keys for all agents."""
        raw_keys = []
        for i in range(10):
            _, raw = await create_api_key(USER, f"cascade-key-{i}", f"key-{i}")
            raw_keys.append(raw)

        await emergency_revoke_all(USER)

        for raw in raw_keys:
            assert await validate_api_key(raw) is None

    @pytest.mark.asyncio
    async def test_revoke_does_not_affect_other_user(self, db):
        """Emergency revoke for user1 does not cascade to user2."""
        await create_agent_policy(AgentPolicy(
            agent_id="user1-cascade",
            agent_name="User1 Cascade",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="user2-safe",
            agent_name="User2 Safe",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=100,
            created_by=USER2,
            created_at=time.time(),
        ))

        await emergency_revoke_all(USER)

        # User1's agent is disabled
        with pytest.raises(PolicyDenied, match="disabled"):
            await enforce_policy(USER, "user1-cascade", "github", ["repo"])

        # User2's agent still works
        result = await enforce_policy(USER2, "user2-safe", "github", ["repo"])
        assert result.agent_id == "user2-safe"

    @pytest.mark.asyncio
    async def test_revoke_followed_by_selective_re_enable(self, db):
        """After emergency revoke, selectively re-enable specific agents."""
        for i in range(5):
            await create_agent_policy(AgentPolicy(
                agent_id=f"renable-{i}",
                agent_name=f"Re-enable {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        await emergency_revoke_all(USER)

        # Re-enable only agent 0 and 2
        await toggle_agent_policy("renable-0", USER)
        await toggle_agent_policy("renable-2", USER)

        # 0 and 2 should work
        result = await enforce_policy(USER, "renable-0", "github", ["repo"])
        assert result.agent_id == "renable-0"
        result = await enforce_policy(USER, "renable-2", "github", ["repo"])
        assert result.agent_id == "renable-2"

        # Others still disabled
        for i in [1, 3, 4]:
            with pytest.raises(PolicyDenied, match="disabled"):
                await enforce_policy(USER, f"renable-{i}", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_revoke_audit_trail(self, db):
        """Emergency revoke should leave audit entries for each disabled agent."""
        for i in range(3):
            await create_agent_policy(AgentPolicy(
                agent_id=f"audit-cascade-{i}",
                agent_name=f"Audit Cascade {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                created_by=USER,
                created_at=time.time(),
            ))

        # Log the emergency revoke action (as the app would)
        result = await emergency_revoke_all(USER)
        await log_audit(USER, "*", "", "emergency_revoke", "success",
                       details=f"Disabled {result['policies_disabled']} policies")

        entries = await get_audit_log(USER, limit=10)
        revoke_entries = [e for e in entries if e.action == "emergency_revoke"]
        assert len(revoke_entries) >= 1


# ============================================================
# 5. Agent Group Policies
# ============================================================

class TestAgentGroupPolicies:
    """Testing patterns for groups of agents with shared characteristics."""

    @pytest.mark.asyncio
    async def test_agent_group_same_services(self, db):
        """Group of agents all with the same service access."""
        for i in range(5):
            await create_agent_policy(AgentPolicy(
                agent_id=f"group-a-{i}",
                agent_name=f"Group A Agent {i}",
                allowed_services=["github", "slack"],
                allowed_scopes={"github": ["repo"], "slack": ["chat:write"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        # All should have access to both services
        for i in range(5):
            r1 = await enforce_policy(USER, f"group-a-{i}", "github", ["repo"])
            assert r1.allowed_services == ["github", "slack"]
            r2 = await enforce_policy(USER, f"group-a-{i}", "slack", ["chat:write"])
            assert r2.allowed_services == ["github", "slack"]

    @pytest.mark.asyncio
    async def test_agent_group_with_shared_rate_limit_pattern(self, db):
        """Agents in a group each have their own rate limit counter."""
        for i in range(3):
            await create_agent_policy(AgentPolicy(
                agent_id=f"shared-rl-{i}",
                agent_name=f"Shared RL {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=2,
                created_by=USER,
                created_at=time.time(),
            ))

        # Each agent can make 2 requests independently
        for i in range(3):
            await enforce_policy(USER, f"shared-rl-{i}", "github", ["repo"])
            await enforce_policy(USER, f"shared-rl-{i}", "github", ["repo"])

        # Each reaches their limit independently
        for i in range(3):
            with pytest.raises(PolicyDenied, match="Rate limit"):
                await enforce_policy(USER, f"shared-rl-{i}", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_agent_group_mixed_permissions(self, db):
        """Group with tiered permissions (read-only, read-write, admin)."""
        tiers = [
            ("reader", ["github"], {"github": ["read:user"]}),
            ("writer", ["github"], {"github": ["read:user", "repo"]}),
            ("admin", ["github", "slack"], {"github": ["read:user", "repo", "gist"], "slack": ["chat:write"]}),
        ]

        for agent_id, services, scopes in tiers:
            await create_agent_policy(AgentPolicy(
                agent_id=agent_id,
                agent_name=agent_id.title(),
                allowed_services=services,
                allowed_scopes=scopes,
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        # Reader can only read
        await enforce_policy(USER, "reader", "github", ["read:user"])
        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "reader", "github", ["read:user", "repo"])

        # Writer can read and write
        await enforce_policy(USER, "writer", "github", ["read:user", "repo"])

        # Admin can do everything
        await enforce_policy(USER, "admin", "github", ["read:user", "repo", "gist"])
        await enforce_policy(USER, "admin", "slack", ["chat:write"])

    @pytest.mark.asyncio
    async def test_disable_entire_group(self, db):
        """Disabling all agents in a group."""
        for i in range(5):
            await create_agent_policy(AgentPolicy(
                agent_id=f"disable-group-{i}",
                agent_name=f"Disable Group {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        for i in range(5):
            await toggle_agent_policy(f"disable-group-{i}", USER)

        for i in range(5):
            with pytest.raises(PolicyDenied, match="disabled"):
                await enforce_policy(USER, f"disable-group-{i}", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_agent_group_ip_subnets(self, db):
        """Group of agents each restricted to different IP subnets."""
        subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]

        for i, subnet in enumerate(subnets):
            await create_agent_policy(AgentPolicy(
                agent_id=f"subnet-agent-{i}",
                agent_name=f"Subnet Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                ip_allowlist=[subnet],
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        # Each agent only works from its subnet
        await enforce_policy(USER, "subnet-agent-0", "github", ["repo"], ip_address="10.0.1.50")
        await enforce_policy(USER, "subnet-agent-1", "github", ["repo"], ip_address="10.0.2.50")
        await enforce_policy(USER, "subnet-agent-2", "github", ["repo"], ip_address="10.0.3.50")

        # Cross-subnet should fail
        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "subnet-agent-0", "github", ["repo"], ip_address="10.0.2.50")

    @pytest.mark.asyncio
    async def test_agent_group_step_up_requirements(self, db):
        """Group with different step-up requirements."""
        await create_agent_policy(AgentPolicy(
            agent_id="no-stepup",
            agent_name="No Step Up",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            requires_step_up=[],
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="yes-stepup",
            agent_name="Yes Step Up",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            requires_step_up=["github"],
            rate_limit_per_minute=100,
            created_by=USER,
            created_at=time.time(),
        ))

        no_su_policy = await get_agent_policy("no-stepup")
        yes_su_policy = await get_agent_policy("yes-stepup")

        assert requires_step_up(no_su_policy, "github") is False
        assert requires_step_up(yes_su_policy, "github") is True

    @pytest.mark.asyncio
    async def test_agent_effective_scopes_across_group(self, db):
        """Effective scopes should be computed per-agent in a group."""
        configs = [
            ("agent-read", {"github": ["read:user"]}),
            ("agent-write", {"github": ["repo", "gist"]}),
            ("agent-full", {"github": ["repo", "read:user", "gist", "notifications"]}),
        ]

        for agent_id, scopes in configs:
            await create_agent_policy(AgentPolicy(
                agent_id=agent_id,
                agent_name=agent_id,
                allowed_services=["github"],
                allowed_scopes=scopes,
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        # All request ["repo", "read:user"]
        requested = ["repo", "read:user"]

        p1 = await get_agent_policy("agent-read")
        eff1 = get_effective_scopes(p1, "github", requested)
        assert set(eff1) == {"read:user"}

        p2 = await get_agent_policy("agent-write")
        eff2 = get_effective_scopes(p2, "github", requested)
        assert set(eff2) == {"repo"}

        p3 = await get_agent_policy("agent-full")
        eff3 = get_effective_scopes(p3, "github", requested)
        assert set(eff3) == {"repo", "read:user"}

    @pytest.mark.asyncio
    async def test_agent_group_query_all_for_user(self, db):
        """get_all_policies returns all agents for a user."""
        for i in range(8):
            await create_agent_policy(AgentPolicy(
                agent_id=f"query-group-{i}",
                agent_name=f"Query Group {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                created_by=USER,
                created_at=time.time(),
            ))

        policies = await get_all_policies(USER)
        assert len(policies) == 8
        agent_ids = {p.agent_id for p in policies}
        for i in range(8):
            assert f"query-group-{i}" in agent_ids

    @pytest.mark.asyncio
    async def test_agent_group_delete_subset(self, db):
        """Deleting a subset of group agents leaves others intact."""
        for i in range(6):
            await create_agent_policy(AgentPolicy(
                agent_id=f"del-subset-{i}",
                agent_name=f"Del Subset {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        # Delete first 3
        for i in range(3):
            await delete_agent_policy(f"del-subset-{i}", USER)

        # First 3 gone
        for i in range(3):
            assert await get_agent_policy(f"del-subset-{i}") is None

        # Last 3 still exist
        for i in range(3, 6):
            p = await get_agent_policy(f"del-subset-{i}")
            assert p is not None

    @pytest.mark.asyncio
    async def test_agent_group_keys_per_agent(self, db):
        """Each agent in a group can have its own API keys."""
        for i in range(4):
            await create_agent_policy(AgentPolicy(
                agent_id=f"keyed-agent-{i}",
                agent_name=f"Keyed Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                created_by=USER,
                created_at=time.time(),
            ))
            _, raw = await create_api_key(USER, f"keyed-agent-{i}", f"key-{i}")
            result = await validate_api_key(raw)
            assert result is not None
            assert result.agent_id == f"keyed-agent-{i}"

    @pytest.mark.asyncio
    async def test_agent_group_connected_services_shared(self, db):
        """Connected services are shared across all agents for a user."""
        await add_connected_service(USER, "github")
        await add_connected_service(USER, "slack")

        for i in range(3):
            await create_agent_policy(AgentPolicy(
                agent_id=f"shared-conn-{i}",
                agent_name=f"Shared Conn {i}",
                allowed_services=["github", "slack"],
                allowed_scopes={"github": ["repo"], "slack": ["chat:write"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        svcs = await get_connected_services(USER)
        svc_names = {s["service"] for s in svcs}
        assert "github" in svc_names
        assert "slack" in svc_names

        # All agents can access both services
        for i in range(3):
            await enforce_policy(USER, f"shared-conn-{i}", "github", ["repo"])
            await enforce_policy(USER, f"shared-conn-{i}", "slack", ["chat:write"])

    @pytest.mark.asyncio
    async def test_agent_group_expiry_staggered(self, db):
        """Agents with staggered expiry times."""
        now = time.time()

        # Agent 0: already expired, Agent 1: expires later, Agent 2: never expires
        configs = [
            ("stagger-0", now - 3600),
            ("stagger-1", now + 86400),
            ("stagger-2", 0),
        ]

        for agent_id, exp in configs:
            await create_agent_policy(AgentPolicy(
                agent_id=agent_id,
                agent_name=agent_id,
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                expires_at=exp,
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))

        # Expired agent
        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy(USER, "stagger-0", "github", ["repo"])

        # Future expiry agent
        result = await enforce_policy(USER, "stagger-1", "github", ["repo"])
        assert result.agent_id == "stagger-1"

        # Never-expiring agent
        result = await enforce_policy(USER, "stagger-2", "github", ["repo"])
        assert result.agent_id == "stagger-2"

    @pytest.mark.asyncio
    async def test_agent_concurrent_creation_and_enforcement(self, db):
        """Creating agents and enforcing them concurrently."""
        async def create_and_enforce(i):
            await create_agent_policy(AgentPolicy(
                agent_id=f"create-enf-{i}",
                agent_name=f"Create Enf {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by=USER,
                created_at=time.time(),
            ))
            return await enforce_policy(USER, f"create-enf-{i}", "github", ["repo"])

        results = await asyncio.gather(*[
            create_and_enforce(i) for i in range(10)
        ])

        assert all(r.agent_id.startswith("create-enf-") for r in results)

    @pytest.mark.asyncio
    async def test_multi_user_multi_agent_isolation(self, db):
        """Multiple users with multiple agents should be fully isolated."""
        users = [USER, USER2, "auth0|multi-agent-user-3"]

        for uid in users:
            for i in range(3):
                await create_agent_policy(AgentPolicy(
                    agent_id=f"iso-{uid[-1]}-{i}",
                    agent_name=f"Iso {uid[-1]} {i}",
                    allowed_services=["github"],
                    allowed_scopes={"github": ["repo"]},
                    rate_limit_per_minute=100,
                    created_by=uid,
                    created_at=time.time(),
                ))

        # Each user can only enforce their own agents
        for uid in users:
            for i in range(3):
                result = await enforce_policy(uid, f"iso-{uid[-1]}-{i}", "github", ["repo"])
                assert result.created_by == uid

        # Cross-user access should fail
        with pytest.raises(PolicyDenied, match="Not authorized"):
            await enforce_policy(USER, f"iso-{USER2[-1]}-0", "github", ["repo"])
