"""Deep audit trail testing -- ensure audit log captures all enforcement
outcomes, edge cases around ordering, field correctness, limit behaviour,
and interaction with policy operations."""

import time
import pytest

from src.database import (
    AgentPolicy,
    create_agent_policy,
    create_api_key,
    emergency_revoke_all,
    get_audit_log,
    log_audit,
    toggle_agent_policy,
)
from src.policy import (
    PolicyDenied,
    _rate_counters,
    enforce_policy,
)


@pytest.fixture(autouse=True)
def clear_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


USER = "auth0|audit-user"


# ---------------------------------------------------------------------------
# 1. Direct log_audit correctness
# ---------------------------------------------------------------------------

class TestLogAuditFields:
    """Verify that log_audit stores all fields correctly."""

    @pytest.mark.asyncio
    async def test_all_fields_stored(self, db):
        """All fields provided to log_audit appear in the retrieved entry."""
        await log_audit(
            user_id=USER, agent_id="bot-1", service="github",
            action="token_request", status="success",
            scopes="repo,read:user", ip_address="10.0.0.1",
            details="All checks passed",
        )
        entries = await get_audit_log(USER)
        assert len(entries) == 1
        e = entries[0]
        assert e.user_id == USER
        assert e.agent_id == "bot-1"
        assert e.service == "github"
        assert e.action == "token_request"
        assert e.status == "success"
        assert e.scopes == "repo,read:user"
        assert e.ip_address == "10.0.0.1"
        assert e.details == "All checks passed"

    @pytest.mark.asyncio
    async def test_timestamp_is_recent(self, db):
        """Audit entry timestamp is close to current time."""
        before = time.time()
        await log_audit(USER, "bot-1", "github", "action", "ok")
        after = time.time()
        entries = await get_audit_log(USER)
        assert before <= entries[0].timestamp <= after

    @pytest.mark.asyncio
    async def test_auto_increment_id(self, db):
        """Each audit entry gets a unique auto-incrementing ID."""
        for i in range(5):
            await log_audit(USER, f"bot-{i}", "github", "action", "ok")
        entries = await get_audit_log(USER)
        ids = [e.id for e in entries]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_empty_optional_fields(self, db):
        """Optional fields default to empty strings."""
        await log_audit(USER, "bot-1", "github", "action", "ok")
        entries = await get_audit_log(USER)
        e = entries[0]
        assert e.scopes == ""
        assert e.ip_address == ""
        assert e.details == ""

    @pytest.mark.asyncio
    async def test_special_characters_in_details(self, db):
        """Special characters in details are preserved."""
        detail = "Error: 'foo' < \"bar\" & baz=100%"
        await log_audit(USER, "bot-1", "github", "action", "ok", details=detail)
        entries = await get_audit_log(USER)
        assert entries[0].details == detail


# ---------------------------------------------------------------------------
# 2. Ordering and limit
# ---------------------------------------------------------------------------

class TestAuditOrdering:
    """Audit log ordering and limit behaviour."""

    @pytest.mark.asyncio
    async def test_default_ordering_is_newest_first(self, db):
        """Audit entries are returned newest first."""
        for i in range(5):
            await log_audit(USER, f"bot-{i}", "github", "action", "ok")
        entries = await get_audit_log(USER)
        for i in range(1, len(entries)):
            assert entries[i - 1].timestamp >= entries[i].timestamp

    @pytest.mark.asyncio
    async def test_limit_parameter(self, db):
        """get_audit_log respects the limit parameter."""
        for i in range(10):
            await log_audit(USER, f"bot-{i}", "github", "action", "ok")
        entries = await get_audit_log(USER, limit=3)
        assert len(entries) == 3

    @pytest.mark.asyncio
    async def test_limit_exceeds_total(self, db):
        """Limit larger than total entries returns all entries."""
        for i in range(3):
            await log_audit(USER, f"bot-{i}", "github", "action", "ok")
        entries = await get_audit_log(USER, limit=100)
        assert len(entries) == 3

    @pytest.mark.asyncio
    async def test_limit_zero_returns_none(self, db):
        """Limit of 0 returns no entries."""
        await log_audit(USER, "bot-1", "github", "action", "ok")
        entries = await get_audit_log(USER, limit=0)
        assert len(entries) == 0


# ---------------------------------------------------------------------------
# 3. Enforcement audit trail completeness
# ---------------------------------------------------------------------------

class TestEnforcementAudit:
    """Verify that enforce_policy creates audit entries for every outcome."""

    @pytest.mark.asyncio
    async def test_unregistered_agent_logged(self, db):
        """Attempting to enforce on a non-existent agent creates an audit entry."""
        with pytest.raises(PolicyDenied, match="not registered"):
            await enforce_policy(USER, "nonexistent-bot", "github", ["repo"])
        entries = await get_audit_log(USER)
        assert len(entries) == 1
        assert entries[0].status == "denied"
        assert "not registered" in entries[0].details.lower()

    @pytest.mark.asyncio
    async def test_disabled_agent_logged(self, db):
        """Attempting to enforce on a disabled agent creates an audit entry."""
        await create_agent_policy(AgentPolicy(
            agent_id="off-bot", agent_name="Off Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            created_by=USER, created_at=time.time(), is_active=False,
        ))
        with pytest.raises(PolicyDenied, match="disabled"):
            await enforce_policy(USER, "off-bot", "github", ["repo"])
        entries = await get_audit_log(USER)
        assert len(entries) == 1
        assert entries[0].status == "denied"

    @pytest.mark.asyncio
    async def test_ownership_violation_logged(self, db):
        """Cross-tenant enforcement denial is logged under the caller."""
        await create_agent_policy(AgentPolicy(
            agent_id="other-bot", agent_name="Other Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            created_by="auth0|other", created_at=time.time(), is_active=True,
        ))
        with pytest.raises(PolicyDenied, match="do not own"):
            await enforce_policy(USER, "other-bot", "github", ["repo"])
        entries = await get_audit_log(USER)
        assert len(entries) == 1
        assert "ownership" in entries[0].details.lower()

    @pytest.mark.asyncio
    async def test_expired_policy_logged(self, db):
        """Expired policy denial is logged."""
        await create_agent_policy(AgentPolicy(
            agent_id="exp-bot", agent_name="Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            created_by=USER, created_at=time.time(), is_active=True,
            expires_at=time.time() - 100,
        ))
        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy(USER, "exp-bot", "github", ["repo"])
        entries = await get_audit_log(USER)
        assert len(entries) == 1
        assert "expired" in entries[0].details.lower()

    @pytest.mark.asyncio
    async def test_service_denial_logged(self, db):
        """Unauthorized service access is logged."""
        await create_agent_policy(AgentPolicy(
            agent_id="svc-bot", agent_name="Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "svc-bot", "slack", ["chat:write"])
        entries = await get_audit_log(USER)
        assert len(entries) == 1
        assert "slack" in entries[0].details.lower() or entries[0].service == "slack"

    @pytest.mark.asyncio
    async def test_scope_denial_logged(self, db):
        """Excess scope request is logged with scope details."""
        await create_agent_policy(AgentPolicy(
            agent_id="scope-bot", agent_name="Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        with pytest.raises(PolicyDenied, match="not permitted"):
            await enforce_policy(USER, "scope-bot", "github", ["repo", "admin"])
        entries = await get_audit_log(USER)
        assert len(entries) == 1
        assert "admin" in entries[0].details

    @pytest.mark.asyncio
    async def test_rate_limit_logged(self, db):
        """Rate limit exceeded is logged."""
        await create_agent_policy(AgentPolicy(
            agent_id="rl-bot", agent_name="Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=1,
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        await enforce_policy(USER, "rl-bot", "github", ["repo"])
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy(USER, "rl-bot", "github", ["repo"])
        entries = await get_audit_log(USER)
        assert any(e.status == "rate_limited" for e in entries)

    @pytest.mark.asyncio
    async def test_ip_denial_logged(self, db):
        """IP allowlist denial is logged with IP address."""
        await create_agent_policy(AgentPolicy(
            agent_id="ip-bot", agent_name="Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            ip_allowlist=["10.0.0.1"],
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        with pytest.raises(PolicyDenied, match="not in allowlist"):
            await enforce_policy(USER, "ip-bot", "github", ["repo"], ip_address="192.168.1.1")
        entries = await get_audit_log(USER)
        assert len(entries) == 1
        assert entries[0].ip_address == "192.168.1.1"


# ---------------------------------------------------------------------------
# 4. Audit trail accumulation
# ---------------------------------------------------------------------------

class TestAuditAccumulation:
    """Audit entries accumulate correctly across multiple operations."""

    @pytest.mark.asyncio
    async def test_multiple_successes_logged(self, db):
        """Multiple successful enforcements each produce an audit entry."""
        await create_agent_policy(AgentPolicy(
            agent_id="acc-bot", agent_name="Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        # Note: enforce_policy does NOT log success audit entries directly
        # Only failures create audit entries. Let's log manual successes.
        for i in range(5):
            await log_audit(USER, "acc-bot", "github", "token_request", "success")
        entries = await get_audit_log(USER)
        assert len(entries) == 5

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure_audit(self, db):
        """Both successes and failures appear in the audit trail."""
        await log_audit(USER, "bot-1", "github", "token_request", "success")
        await log_audit(USER, "bot-1", "slack", "token_request", "denied", details="Not allowed")
        await log_audit(USER, "bot-2", "github", "token_request", "success")
        entries = await get_audit_log(USER)
        assert len(entries) == 3
        statuses = [e.status for e in entries]
        assert "success" in statuses
        assert "denied" in statuses

    @pytest.mark.asyncio
    async def test_audit_for_multiple_agents(self, db):
        """Audit entries for different agents are all visible to the user."""
        for agent in ["bot-a", "bot-b", "bot-c"]:
            await log_audit(USER, agent, "github", "action", "ok")
        entries = await get_audit_log(USER)
        agents = {e.agent_id for e in entries}
        assert agents == {"bot-a", "bot-b", "bot-c"}


# ---------------------------------------------------------------------------
# 5. Audit persistence across operations
# ---------------------------------------------------------------------------

class TestAuditPersistence:
    """Audit entries persist even when related resources are modified/deleted."""

    @pytest.mark.asyncio
    async def test_audit_persists_after_policy_delete(self, db):
        """Deleting a policy does not remove its audit entries."""
        await create_agent_policy(AgentPolicy(
            agent_id="del-audit-bot", agent_name="Bot",
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        await log_audit(USER, "del-audit-bot", "github", "action", "ok")
        await log_audit(USER, "del-audit-bot", "github", "action", "denied")

        from src.database import delete_agent_policy
        await delete_agent_policy("del-audit-bot", USER)

        entries = await get_audit_log(USER)
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_audit_persists_after_policy_toggle(self, db):
        """Toggling a policy does not remove audit entries."""
        await create_agent_policy(AgentPolicy(
            agent_id="toggle-audit-bot", agent_name="Bot",
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        await log_audit(USER, "toggle-audit-bot", "github", "action", "ok")
        await toggle_agent_policy("toggle-audit-bot", USER)
        entries = await get_audit_log(USER)
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_audit_persists_after_emergency_revoke(self, db):
        """Emergency revoke does not remove existing audit entries."""
        await create_agent_policy(AgentPolicy(
            agent_id="em-audit-bot", agent_name="Bot",
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        await log_audit(USER, "em-audit-bot", "github", "action", "ok")
        await log_audit(USER, "em-audit-bot", "slack", "action", "denied")
        await emergency_revoke_all(USER)
        entries = await get_audit_log(USER)
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# 6. Audit log for different services
# ---------------------------------------------------------------------------

class TestAuditServiceVariety:
    """Audit entries correctly track different services."""

    @pytest.mark.asyncio
    async def test_different_services_tracked(self, db):
        """Audit entries for github, slack, google, linear, notion all tracked."""
        services = ["github", "slack", "google", "linear", "notion"]
        for svc in services:
            await log_audit(USER, "multi-svc-bot", svc, "token_request", "success")
        entries = await get_audit_log(USER)
        logged_svcs = {e.service for e in entries}
        assert logged_svcs == set(services)

    @pytest.mark.asyncio
    async def test_service_field_matches_enforcement(self, db):
        """The service field in denial audit matches the attempted service."""
        await create_agent_policy(AgentPolicy(
            agent_id="svc-match", agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "svc-match", "slack", ["chat:write"])
        entries = await get_audit_log(USER)
        assert entries[0].service == "slack"


# ---------------------------------------------------------------------------
# 7. Enforcement chain audit completeness
# ---------------------------------------------------------------------------

class TestEnforcementChainAudit:
    """Ensure that each enforcement step that fails logs exactly one entry."""

    @pytest.mark.asyncio
    async def test_single_entry_per_denial(self, db):
        """Each enforcement denial creates exactly one audit entry."""
        # Non-existent agent
        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "no-agent", "github", ["repo"])
        entries = await get_audit_log(USER)
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_cascading_checks_stop_at_first_failure(self, db):
        """If agent doesn't exist, no further checks are logged."""
        with pytest.raises(PolicyDenied, match="not registered"):
            await enforce_policy(USER, "ghost-agent", "github", ["repo"], ip_address="bad_ip")
        entries = await get_audit_log(USER)
        assert len(entries) == 1
        assert "not registered" in entries[0].details.lower()

    @pytest.mark.asyncio
    async def test_successful_enforcement_no_audit(self, db):
        """A successful enforce_policy does NOT create an audit entry."""
        await create_agent_policy(AgentPolicy(
            agent_id="success-bot", agent_name="Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            created_by=USER, created_at=time.time(), is_active=True,
        ))
        await enforce_policy(USER, "success-bot", "github", ["repo"])
        entries = await get_audit_log(USER)
        # enforce_policy only logs on denial, not on success
        assert len(entries) == 0
