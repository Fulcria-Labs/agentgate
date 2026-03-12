"""Advanced scenario tests for AgentGate — multi-tenant isolation, API key
lifecycle, policy CRUD edge cases, IP/time boundary conditions, scope
intersection, audit log behaviour, connected services, and PolicyDenied
exception attributes."""

import hashlib
import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.auth import SUPPORTED_SERVICES
from src.database import (
    AgentPolicy,
    ApiKey,
    AuditEntry,
    add_connected_service,
    create_agent_policy,
    create_api_key,
    delete_agent_policy,
    emergency_revoke_all,
    get_agent_policy,
    get_all_policies,
    get_api_keys,
    get_audit_log,
    get_connected_services,
    init_db,
    log_audit,
    remove_connected_service,
    revoke_api_key,
    toggle_agent_policy,
    validate_api_key,
)
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
def clear_rate_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


# ---------------------------------------------------------------------------
# 1. Multi-tenant isolation (8 tests)
# ---------------------------------------------------------------------------

class TestMultiTenantIsolation:
    """Verify that users cannot access, modify, or view other users' resources."""

    @pytest.mark.asyncio
    async def test_user_cannot_enforce_policy_of_another_user(self, db):
        """enforce_policy denies when caller is not the policy owner."""
        await create_agent_policy(AgentPolicy(
            agent_id="tenant-agent",
            agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user-a",
            created_at=time.time(),
            is_active=True,
        ))
        with pytest.raises(PolicyDenied, match="do not own"):
            await enforce_policy("user-b", "tenant-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_get_all_policies_scoped_to_user(self, db):
        """get_all_policies only returns policies owned by the requesting user."""
        await create_agent_policy(AgentPolicy(
            agent_id="a-bot", agent_name="A", allowed_services=["github"],
            created_by="user-a", created_at=time.time(),
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="b-bot", agent_name="B", allowed_services=["slack"],
            created_by="user-b", created_at=time.time(),
        ))
        policies_a = await get_all_policies("user-a")
        policies_b = await get_all_policies("user-b")
        assert len(policies_a) == 1
        assert policies_a[0].agent_id == "a-bot"
        assert len(policies_b) == 1
        assert policies_b[0].agent_id == "b-bot"

    @pytest.mark.asyncio
    async def test_toggle_fails_for_wrong_user(self, db):
        """toggle_agent_policy returns None when the user doesn't own the policy."""
        await create_agent_policy(AgentPolicy(
            agent_id="own-bot", agent_name="Bot",
            allowed_services=["github"], created_by="owner",
            created_at=time.time(), is_active=True,
        ))
        result = await toggle_agent_policy("own-bot", "intruder")
        assert result is None
        # Original state unchanged
        p = await get_agent_policy("own-bot")
        assert p.is_active is True

    @pytest.mark.asyncio
    async def test_delete_fails_for_wrong_user(self, db):
        """delete_agent_policy returns False when the user doesn't own the policy."""
        await create_agent_policy(AgentPolicy(
            agent_id="del-bot", agent_name="Bot",
            allowed_services=["github"], created_by="owner",
            created_at=time.time(),
        ))
        deleted = await delete_agent_policy("del-bot", "intruder")
        assert deleted is False
        assert await get_agent_policy("del-bot") is not None

    @pytest.mark.asyncio
    async def test_revoke_key_fails_for_wrong_user(self, db):
        """revoke_api_key returns False when the key belongs to another user."""
        api_key, raw = await create_api_key("user-a", "agent-1", "key")
        revoked = await revoke_api_key(api_key.id, "user-b")
        assert revoked is False
        # Key still works
        assert await validate_api_key(raw) is not None

    @pytest.mark.asyncio
    async def test_get_api_keys_scoped_to_user(self, db):
        """get_api_keys only returns keys owned by the requesting user."""
        await create_api_key("user-a", "agent-a", "key-a")
        await create_api_key("user-b", "agent-b", "key-b")
        keys_a = await get_api_keys("user-a")
        keys_b = await get_api_keys("user-b")
        assert len(keys_a) == 1
        assert keys_a[0].agent_id == "agent-a"
        assert len(keys_b) == 1
        assert keys_b[0].agent_id == "agent-b"

    @pytest.mark.asyncio
    async def test_audit_log_scoped_to_user(self, db):
        """get_audit_log only returns entries for the requesting user."""
        await log_audit("user-a", "agent-a", "github", "action", "success")
        await log_audit("user-b", "agent-b", "slack", "action", "success")
        entries_a = await get_audit_log("user-a")
        entries_b = await get_audit_log("user-b")
        assert len(entries_a) == 1
        assert entries_a[0].agent_id == "agent-a"
        assert len(entries_b) == 1
        assert entries_b[0].agent_id == "agent-b"

    @pytest.mark.asyncio
    async def test_connected_services_scoped_to_user(self, db):
        """Connected services are isolated between users."""
        await add_connected_service("user-a", "github", "conn-a")
        await add_connected_service("user-b", "slack", "conn-b")
        svcs_a = await get_connected_services("user-a")
        svcs_b = await get_connected_services("user-b")
        assert len(svcs_a) == 1
        assert svcs_a[0]["service"] == "github"
        assert len(svcs_b) == 1
        assert svcs_b[0]["service"] == "slack"


# ---------------------------------------------------------------------------
# 2. API key lifecycle edge cases (8 tests)
# ---------------------------------------------------------------------------

class TestApiKeyLifecycle:
    """Deep API key lifecycle: last_used tracking, prefix format, multiple
    keys per agent, and interaction with policy deletion."""

    @pytest.mark.asyncio
    async def test_last_used_at_updates_on_validation(self, db):
        """validate_api_key updates last_used_at timestamp."""
        _, raw = await create_api_key("user1", "agent-1", "track")
        before = time.time()
        result = await validate_api_key(raw)
        after = time.time()
        assert result.last_used_at >= before
        assert result.last_used_at <= after

    @pytest.mark.asyncio
    async def test_key_starts_with_ag_prefix(self, db):
        """Raw API keys always start with 'ag_'."""
        _, raw = await create_api_key("user1", "agent-1", "prefix")
        assert raw.startswith("ag_")

    @pytest.mark.asyncio
    async def test_each_key_has_unique_hash(self, db):
        """Multiple keys have distinct hashes."""
        _, raw1 = await create_api_key("user1", "agent-1", "k1")
        _, raw2 = await create_api_key("user1", "agent-1", "k2")
        h1 = hashlib.sha256(raw1.encode()).hexdigest()
        h2 = hashlib.sha256(raw2.encode()).hexdigest()
        assert h1 != h2

    @pytest.mark.asyncio
    async def test_revoking_one_key_does_not_affect_others(self, db):
        """Revoking one key leaves other keys for the same agent valid."""
        k1, raw1 = await create_api_key("user1", "agent-1", "key-1")
        _, raw2 = await create_api_key("user1", "agent-1", "key-2")
        await revoke_api_key(k1.id, "user1")
        assert await validate_api_key(raw1) is None
        assert await validate_api_key(raw2) is not None

    @pytest.mark.asyncio
    async def test_key_with_zero_expiry_never_expires(self, db):
        """A key created with expires_in=0 never expires."""
        _, raw = await create_api_key("user1", "agent-1", "forever", expires_in=0)
        result = await validate_api_key(raw)
        assert result is not None
        assert result.expires_at == 0.0

    @pytest.mark.asyncio
    async def test_key_with_positive_expiry_has_future_timestamp(self, db):
        """A key with expires_in=3600 has expires_at ~1 hour in the future."""
        before = time.time()
        api_key, _ = await create_api_key("user1", "agent-1", "hourly", expires_in=3600)
        assert api_key.expires_at >= before + 3599
        assert api_key.expires_at <= before + 3601

    @pytest.mark.asyncio
    async def test_invalid_key_string_returns_none(self, db):
        """Validating a completely bogus key string returns None."""
        assert await validate_api_key("ag_totallyinvalidkey12345") is None

    @pytest.mark.asyncio
    async def test_empty_key_string_returns_none(self, db):
        """Validating an empty string returns None."""
        assert await validate_api_key("") is None


# ---------------------------------------------------------------------------
# 3. Policy CRUD edge cases (8 tests)
# ---------------------------------------------------------------------------

class TestPolicyCrud:
    """Policy create, update (upsert), toggle, and delete edge cases."""

    @pytest.mark.asyncio
    async def test_create_replaces_existing_policy(self, db):
        """Creating a policy with same agent_id overwrites the existing one."""
        await create_agent_policy(AgentPolicy(
            agent_id="upsert-bot", agent_name="Original",
            allowed_services=["github"], created_by="user1",
            created_at=time.time(),
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="upsert-bot", agent_name="Updated",
            allowed_services=["github", "slack"], created_by="user1",
            created_at=time.time(),
        ))
        p = await get_agent_policy("upsert-bot")
        assert p.agent_name == "Updated"
        assert set(p.allowed_services) == {"github", "slack"}

    @pytest.mark.asyncio
    async def test_double_toggle_restores_state(self, db):
        """Toggling twice returns to the original is_active state."""
        await create_agent_policy(AgentPolicy(
            agent_id="dbl-tog", agent_name="Bot",
            allowed_services=["github"], created_by="user1",
            created_at=time.time(), is_active=True,
        ))
        await toggle_agent_policy("dbl-tog", "user1")
        p = await get_agent_policy("dbl-tog")
        assert p.is_active is False
        await toggle_agent_policy("dbl-tog", "user1")
        p = await get_agent_policy("dbl-tog")
        assert p.is_active is True

    @pytest.mark.asyncio
    async def test_toggle_nonexistent_returns_none(self, db):
        """Toggling a nonexistent agent returns None."""
        result = await toggle_agent_policy("no-such-agent", "user1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, db):
        """Deleting a nonexistent agent returns False."""
        deleted = await delete_agent_policy("no-such-agent", "user1")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_get_nonexistent_policy_returns_none(self, db):
        """Fetching a nonexistent agent returns None."""
        assert await get_agent_policy("ghost-agent") is None

    @pytest.mark.asyncio
    async def test_policy_preserves_all_fields(self, db):
        """All policy fields survive a round-trip through the database."""
        original = AgentPolicy(
            agent_id="full-fields",
            agent_name="Full Bot",
            allowed_services=["github", "slack"],
            allowed_scopes={"github": ["repo", "read:user"], "slack": ["chat:write"]},
            rate_limit_per_minute=42,
            requires_step_up=["slack"],
            created_by="user1",
            created_at=1234567890.5,
            is_active=True,
            allowed_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
            allowed_days=[0, 1, 2, 3, 4],
            expires_at=9999999999.0,
            ip_allowlist=["10.0.0.0/8", "192.168.1.1"],
        )
        await create_agent_policy(original)
        loaded = await get_agent_policy("full-fields")
        assert loaded.agent_id == original.agent_id
        assert loaded.agent_name == original.agent_name
        assert loaded.allowed_services == original.allowed_services
        assert loaded.allowed_scopes == original.allowed_scopes
        assert loaded.rate_limit_per_minute == original.rate_limit_per_minute
        assert loaded.requires_step_up == original.requires_step_up
        assert loaded.created_by == original.created_by
        assert loaded.created_at == original.created_at
        assert loaded.is_active == original.is_active
        assert loaded.allowed_hours == original.allowed_hours
        assert loaded.allowed_days == original.allowed_days
        assert loaded.expires_at == original.expires_at
        assert loaded.ip_allowlist == original.ip_allowlist

    @pytest.mark.asyncio
    async def test_empty_lists_preserved(self, db):
        """Policies with empty list fields round-trip correctly."""
        await create_agent_policy(AgentPolicy(
            agent_id="empty-lists", agent_name="Empty",
            allowed_services=[], allowed_scopes={},
            requires_step_up=[], allowed_hours=[], allowed_days=[],
            ip_allowlist=[], created_by="user1", created_at=time.time(),
        ))
        loaded = await get_agent_policy("empty-lists")
        assert loaded.allowed_services == []
        assert loaded.allowed_scopes == {}
        assert loaded.requires_step_up == []
        assert loaded.allowed_hours == []
        assert loaded.allowed_days == []
        assert loaded.ip_allowlist == []

    @pytest.mark.asyncio
    async def test_delete_then_recreate(self, db):
        """Deleting then recreating a policy with the same agent_id works."""
        await create_agent_policy(AgentPolicy(
            agent_id="phoenix", agent_name="V1",
            allowed_services=["github"], created_by="user1",
            created_at=time.time(),
        ))
        await delete_agent_policy("phoenix", "user1")
        assert await get_agent_policy("phoenix") is None
        await create_agent_policy(AgentPolicy(
            agent_id="phoenix", agent_name="V2",
            allowed_services=["slack"], created_by="user1",
            created_at=time.time(),
        ))
        loaded = await get_agent_policy("phoenix")
        assert loaded.agent_name == "V2"
        assert loaded.allowed_services == ["slack"]


# ---------------------------------------------------------------------------
# 4. Time window boundary conditions (8 tests)
# ---------------------------------------------------------------------------

class TestTimeWindowBoundaries:
    """Edge cases for time-based access control (hours and days)."""

    def _policy_with_hours(self, hours, days=None):
        return AgentPolicy(
            agent_id="time-bot", agent_name="Time Bot",
            allowed_services=["github"],
            created_by="user1", created_at=time.time(),
            allowed_hours=hours,
            allowed_days=days or [],
        )

    def test_hour_0_allowed(self):
        """Hour 0 (midnight UTC) is accepted when in allowed_hours."""
        p = self._policy_with_hours([0, 1, 2])
        now = datetime(2026, 3, 12, 0, 30, tzinfo=timezone.utc)
        assert check_time_window(p, now) is None

    def test_hour_23_allowed(self):
        """Hour 23 is accepted when in allowed_hours."""
        p = self._policy_with_hours([22, 23])
        now = datetime(2026, 3, 12, 23, 59, tzinfo=timezone.utc)
        assert check_time_window(p, now) is None

    def test_hour_boundary_denied(self):
        """Hour 18 is denied when allowed hours are 9-17."""
        p = self._policy_with_hours(list(range(9, 18)))
        now = datetime(2026, 3, 12, 18, 0, tzinfo=timezone.utc)
        result = check_time_window(p, now)
        assert result is not None
        assert "not within allowed hours" in result

    def test_empty_hours_always_allowed(self):
        """Empty allowed_hours means all hours are permitted."""
        p = self._policy_with_hours([])
        now = datetime(2026, 3, 12, 15, 0, tzinfo=timezone.utc)
        assert check_time_window(p, now) is None

    def test_monday_is_day_0(self):
        """Monday (weekday 0) is allowed when 0 is in allowed_days."""
        p = self._policy_with_hours([], days=[0, 1, 2, 3, 4])
        # March 16, 2026 is a Monday
        now = datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc)
        assert check_time_window(p, now) is None

    def test_sunday_is_day_6(self):
        """Sunday (weekday 6) is denied when only weekdays are allowed."""
        p = self._policy_with_hours([], days=[0, 1, 2, 3, 4])
        # March 15, 2026 is a Sunday
        now = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)
        result = check_time_window(p, now)
        assert result is not None
        assert "not within allowed days" in result

    def test_empty_days_always_allowed(self):
        """Empty allowed_days means all days are permitted."""
        p = self._policy_with_hours([], days=[])
        now = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)  # Sunday
        assert check_time_window(p, now) is None

    def test_both_hours_and_days_must_pass(self):
        """When both hours and days are set, both must match."""
        p = self._policy_with_hours([9, 10], days=[0])  # Mon 9-10 only
        # Monday but at hour 15
        now = datetime(2026, 3, 16, 15, 0, tzinfo=timezone.utc)
        result = check_time_window(p, now)
        assert result is not None


# ---------------------------------------------------------------------------
# 5. IP allowlist edge cases (8 tests)
# ---------------------------------------------------------------------------

class TestIpAllowlistEdgeCases:
    """IP allowlist parsing, CIDR matching, IPv6, and malformed entries."""

    def _policy_with_ips(self, ips):
        return AgentPolicy(
            agent_id="ip-bot", agent_name="IP Bot",
            allowed_services=["github"],
            created_by="user1", created_at=time.time(),
            ip_allowlist=ips,
        )

    def test_exact_ipv4_match(self):
        p = self._policy_with_ips(["192.168.1.100"])
        assert check_ip_allowlist(p, "192.168.1.100") is None

    def test_exact_ipv4_mismatch(self):
        p = self._policy_with_ips(["192.168.1.100"])
        result = check_ip_allowlist(p, "192.168.1.101")
        assert result is not None

    def test_cidr_match(self):
        p = self._policy_with_ips(["10.0.0.0/8"])
        assert check_ip_allowlist(p, "10.255.255.255") is None

    def test_cidr_mismatch(self):
        p = self._policy_with_ips(["10.0.0.0/8"])
        result = check_ip_allowlist(p, "11.0.0.1")
        assert result is not None

    def test_ipv6_exact_match(self):
        p = self._policy_with_ips(["::1"])
        assert check_ip_allowlist(p, "::1") is None

    def test_multiple_entries_any_match(self):
        """If any entry matches, access is allowed."""
        p = self._policy_with_ips(["10.0.0.0/8", "192.168.0.0/16", "172.16.0.1"])
        assert check_ip_allowlist(p, "192.168.5.5") is None

    def test_malformed_entry_skipped(self):
        """Malformed allowlist entries are skipped, not crash-causing."""
        p = self._policy_with_ips(["not-an-ip", "10.0.0.1"])
        assert check_ip_allowlist(p, "10.0.0.1") is None

    def test_invalid_request_ip_denied(self):
        """An unparseable requesting IP is denied."""
        p = self._policy_with_ips(["10.0.0.1"])
        result = check_ip_allowlist(p, "garbage")
        assert result is not None
        assert "invalid IP" in result


# ---------------------------------------------------------------------------
# 6. get_effective_scopes (6 tests)
# ---------------------------------------------------------------------------

class TestGetEffectiveScopes:
    """Verify scope intersection logic."""

    def _policy(self, scopes_dict):
        return AgentPolicy(
            agent_id="scope-bot", agent_name="Scope Bot",
            allowed_services=list(scopes_dict.keys()),
            allowed_scopes=scopes_dict,
            created_by="user1", created_at=time.time(),
        )

    def test_exact_match(self):
        p = self._policy({"github": ["repo", "read:user"]})
        result = get_effective_scopes(p, "github", ["repo", "read:user"])
        assert set(result) == {"repo", "read:user"}

    def test_subset_request(self):
        p = self._policy({"github": ["repo", "read:user", "gist"]})
        result = get_effective_scopes(p, "github", ["repo"])
        assert result == ["repo"]

    def test_excess_filtered_out(self):
        p = self._policy({"github": ["repo"]})
        result = get_effective_scopes(p, "github", ["repo", "admin:org"])
        assert result == ["repo"]

    def test_no_overlap_returns_empty(self):
        p = self._policy({"github": ["repo"]})
        result = get_effective_scopes(p, "github", ["admin:org"])
        assert result == []

    def test_empty_request_returns_empty(self):
        p = self._policy({"github": ["repo", "read:user"]})
        result = get_effective_scopes(p, "github", [])
        assert result == []

    def test_service_not_in_scopes_returns_empty(self):
        p = self._policy({"github": ["repo"]})
        result = get_effective_scopes(p, "slack", ["chat:write"])
        assert result == []


# ---------------------------------------------------------------------------
# 7. requires_step_up (4 tests)
# ---------------------------------------------------------------------------

class TestRequiresStepUp:
    """Verify step-up auth detection."""

    def _policy(self, step_up_services):
        return AgentPolicy(
            agent_id="su-bot", agent_name="StepUp Bot",
            allowed_services=["github", "slack"],
            requires_step_up=step_up_services,
            created_by="user1", created_at=time.time(),
        )

    def test_step_up_required_when_listed(self):
        p = self._policy(["github"])
        assert requires_step_up(p, "github") is True

    def test_step_up_not_required_when_not_listed(self):
        p = self._policy(["github"])
        assert requires_step_up(p, "slack") is False

    def test_empty_step_up_list(self):
        p = self._policy([])
        assert requires_step_up(p, "github") is False

    def test_multiple_step_up_services(self):
        p = self._policy(["github", "slack"])
        assert requires_step_up(p, "github") is True
        assert requires_step_up(p, "slack") is True


# ---------------------------------------------------------------------------
# 8. PolicyDenied exception attributes (4 tests)
# ---------------------------------------------------------------------------

class TestPolicyDeniedException:
    """Verify PolicyDenied carries agent_id and service context."""

    def test_basic_attributes(self):
        e = PolicyDenied("test reason", "agent-x", "github")
        assert str(e) == "test reason"
        assert e.agent_id == "agent-x"
        assert e.service == "github"

    def test_default_attributes(self):
        e = PolicyDenied("no context")
        assert e.agent_id == ""
        assert e.service == ""

    @pytest.mark.asyncio
    async def test_enforce_sets_agent_id_on_service_denial(self, db):
        """enforce_policy sets agent_id on the PolicyDenied for unauthorized service."""
        await create_agent_policy(AgentPolicy(
            agent_id="pd-agent", agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1", created_at=time.time(), is_active=True,
        ))
        with pytest.raises(PolicyDenied) as exc_info:
            await enforce_policy("user1", "pd-agent", "slack", ["read"])
        assert exc_info.value.agent_id == "pd-agent"
        assert exc_info.value.service == "slack"

    @pytest.mark.asyncio
    async def test_enforce_sets_fields_on_scope_denial(self, db):
        """enforce_policy sets agent_id and service on scope denial."""
        await create_agent_policy(AgentPolicy(
            agent_id="pd-scope", agent_name="Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1", created_at=time.time(), is_active=True,
        ))
        with pytest.raises(PolicyDenied) as exc_info:
            await enforce_policy("user1", "pd-scope", "github", ["repo", "admin:org"])
        assert exc_info.value.agent_id == "pd-scope"
        assert exc_info.value.service == "github"


# ---------------------------------------------------------------------------
# 9. Connected services (6 tests)
# ---------------------------------------------------------------------------

class TestConnectedServices:
    """CRUD operations for connected services."""

    @pytest.mark.asyncio
    async def test_add_service(self, db):
        await add_connected_service("user1", "github", "conn-1")
        svcs = await get_connected_services("user1")
        assert len(svcs) == 1
        assert svcs[0]["service"] == "github"
        assert svcs[0]["connection_id"] == "conn-1"

    @pytest.mark.asyncio
    async def test_add_duplicate_replaces(self, db):
        """Adding the same service twice updates the existing entry."""
        await add_connected_service("user1", "github", "conn-1")
        await add_connected_service("user1", "github", "conn-2")
        svcs = await get_connected_services("user1")
        assert len(svcs) == 1
        assert svcs[0]["connection_id"] == "conn-2"

    @pytest.mark.asyncio
    async def test_remove_service(self, db):
        await add_connected_service("user1", "github", "conn-1")
        await remove_connected_service("user1", "github")
        svcs = await get_connected_services("user1")
        assert len(svcs) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_is_noop(self, db):
        """Removing a service that isn't connected doesn't raise."""
        await remove_connected_service("user1", "github")
        svcs = await get_connected_services("user1")
        assert len(svcs) == 0

    @pytest.mark.asyncio
    async def test_multiple_services(self, db):
        """A user can connect multiple different services."""
        for svc in ["github", "slack", "google"]:
            await add_connected_service("user1", svc, f"conn-{svc}")
        svcs = await get_connected_services("user1")
        assert len(svcs) == 3
        svc_names = {s["service"] for s in svcs}
        assert svc_names == {"github", "slack", "google"}

    @pytest.mark.asyncio
    async def test_connected_at_timestamp(self, db):
        """connected_at is a recent timestamp."""
        before = time.time()
        await add_connected_service("user1", "github", "c1")
        after = time.time()
        svcs = await get_connected_services("user1")
        assert svcs[0]["connected_at"] >= before
        assert svcs[0]["connected_at"] <= after


# ---------------------------------------------------------------------------
# 10. Audit log edge cases (6 tests)
# ---------------------------------------------------------------------------

class TestAuditLogEdgeCases:
    """Audit log limit handling and entry content validation."""

    @pytest.mark.asyncio
    async def test_limit_constrains_results(self, db):
        """get_audit_log with limit returns at most that many entries."""
        for i in range(10):
            await log_audit("user1", f"agent-{i}", "svc", "action", "ok")
        entries = await get_audit_log("user1", limit=3)
        assert len(entries) == 3

    @pytest.mark.asyncio
    async def test_limit_of_one(self, db):
        for i in range(5):
            await log_audit("user1", f"agent-{i}", "svc", "action", "ok")
        entries = await get_audit_log("user1", limit=1)
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_empty_log_returns_empty_list(self, db):
        entries = await get_audit_log("user-no-activity")
        assert entries == []

    @pytest.mark.asyncio
    async def test_audit_entry_has_all_fields(self, db):
        """A logged entry preserves all provided fields."""
        await log_audit("user1", "agent-1", "github", "token_issued", "success",
                       scopes="repo,read:user", ip_address="10.0.0.1",
                       details="test details")
        entries = await get_audit_log("user1")
        e = entries[0]
        assert e.user_id == "user1"
        assert e.agent_id == "agent-1"
        assert e.service == "github"
        assert e.action == "token_issued"
        assert e.status == "success"
        assert e.scopes == "repo,read:user"
        assert e.ip_address == "10.0.0.1"
        assert e.details == "test details"

    @pytest.mark.asyncio
    async def test_audit_entry_has_auto_timestamp(self, db):
        """Audit entries get an automatic timestamp."""
        before = time.time()
        await log_audit("user1", "agent-1", "svc", "action", "ok")
        after = time.time()
        entries = await get_audit_log("user1")
        assert entries[0].timestamp >= before
        assert entries[0].timestamp <= after

    @pytest.mark.asyncio
    async def test_audit_entry_has_auto_id(self, db):
        """Audit entries get an auto-incremented ID."""
        await log_audit("user1", "a1", "svc", "action", "ok")
        await log_audit("user1", "a2", "svc", "action", "ok")
        entries = await get_audit_log("user1")
        assert entries[0].id > entries[1].id


# ---------------------------------------------------------------------------
# 11. SUPPORTED_SERVICES structure (5 tests)
# ---------------------------------------------------------------------------

class TestSupportedServicesStructure:
    """Validate the SUPPORTED_SERVICES configuration."""

    def test_all_services_have_required_keys(self):
        for name, svc in SUPPORTED_SERVICES.items():
            assert "display_name" in svc, f"{name} missing display_name"
            assert "icon" in svc, f"{name} missing icon"
            assert "scopes" in svc, f"{name} missing scopes"
            assert "description" in svc, f"{name} missing description"

    def test_all_services_have_nonempty_scopes(self):
        for name, svc in SUPPORTED_SERVICES.items():
            assert len(svc["scopes"]) > 0, f"{name} has empty scopes"

    def test_expected_services_present(self):
        expected = {"github", "slack", "google", "linear", "notion"}
        assert set(SUPPORTED_SERVICES.keys()) == expected

    def test_display_names_are_strings(self):
        for name, svc in SUPPORTED_SERVICES.items():
            assert isinstance(svc["display_name"], str)
            assert len(svc["display_name"]) > 0

    def test_scopes_are_lists_of_strings(self):
        for name, svc in SUPPORTED_SERVICES.items():
            assert isinstance(svc["scopes"], list)
            for scope in svc["scopes"]:
                assert isinstance(scope, str)


# ---------------------------------------------------------------------------
# 12. Dataclass defaults (5 tests)
# ---------------------------------------------------------------------------

class TestDataclassDefaults:
    """Verify default values for database dataclasses."""

    def test_agent_policy_defaults(self):
        p = AgentPolicy(agent_id="test", agent_name="Test")
        assert p.allowed_services == []
        assert p.allowed_scopes == {}
        assert p.rate_limit_per_minute == 60
        assert p.requires_step_up == []
        assert p.is_active is True
        assert p.allowed_hours == []
        assert p.allowed_days == []
        assert p.expires_at == 0.0
        assert p.ip_allowlist == []

    def test_api_key_defaults(self):
        k = ApiKey()
        assert k.id == ""
        assert k.key_hash == ""
        assert k.key_prefix == ""
        assert k.is_revoked is False
        assert k.expires_at == 0.0
        assert k.last_used_at == 0.0

    def test_audit_entry_defaults(self):
        e = AuditEntry()
        assert e.id == 0
        assert e.timestamp == 0.0
        assert e.scopes == ""
        assert e.ip_address == ""
        assert e.details == ""

    def test_agent_policy_mutable_defaults_independent(self):
        """Each AgentPolicy instance gets its own list instances."""
        p1 = AgentPolicy(agent_id="a", agent_name="A")
        p2 = AgentPolicy(agent_id="b", agent_name="B")
        p1.allowed_services.append("github")
        assert "github" not in p2.allowed_services

    def test_agent_policy_mutable_scopes_independent(self):
        """Each AgentPolicy instance gets its own scopes dict."""
        p1 = AgentPolicy(agent_id="a", agent_name="A")
        p2 = AgentPolicy(agent_id="b", agent_name="B")
        p1.allowed_scopes["github"] = ["repo"]
        assert "github" not in p2.allowed_scopes


# ---------------------------------------------------------------------------
# 13. Complex enforcement scenarios (8 tests)
# ---------------------------------------------------------------------------

class TestComplexEnforcementScenarios:
    """End-to-end scenarios combining multiple policy constraints."""

    @pytest.mark.asyncio
    async def test_full_valid_request_with_all_constraints(self, db):
        """A request passing all constraints (time, IP, service, scope, rate) succeeds."""
        await create_agent_policy(AgentPolicy(
            agent_id="full-valid",
            agent_name="Full Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo", "read:user"]},
            rate_limit_per_minute=10,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
            ip_allowlist=["10.0.0.0/8"],
        ))
        result = await enforce_policy(
            "user1", "full-valid", "github", ["repo"],
            ip_address="10.0.0.5"
        )
        assert result.agent_id == "full-valid"

    @pytest.mark.asyncio
    async def test_ip_denied_even_with_valid_everything_else(self, db):
        """IP denial overrides valid service/scope/rate."""
        await create_agent_policy(AgentPolicy(
            agent_id="ip-block",
            agent_name="IP Block Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=100,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
            ip_allowlist=["10.0.0.1"],
        ))
        with pytest.raises(PolicyDenied, match="not in allowlist"):
            await enforce_policy("user1", "ip-block", "github", ["repo"],
                               ip_address="192.168.1.1")

    @pytest.mark.asyncio
    async def test_expired_policy_cannot_issue_tokens(self, db):
        """An expired policy denies even valid requests."""
        await create_agent_policy(AgentPolicy(
            agent_id="expired-bot",
            agent_name="Expired",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
            is_active=True,
            expires_at=time.time() - 1,
        ))
        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy("user1", "expired-bot", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_disabled_then_reenabled_allows_access(self, db):
        """An agent disabled via toggle, then re-enabled, can access services again."""
        await create_agent_policy(AgentPolicy(
            agent_id="re-enable",
            agent_name="Re-enable Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        ))
        await toggle_agent_policy("re-enable", "user1")
        with pytest.raises(PolicyDenied, match="disabled"):
            await enforce_policy("user1", "re-enable", "github", ["repo"])
        await toggle_agent_policy("re-enable", "user1")
        result = await enforce_policy("user1", "re-enable", "github", ["repo"])
        assert result.agent_id == "re-enable"

    @pytest.mark.asyncio
    async def test_empty_scope_request_passes(self, db):
        """Requesting zero scopes doesn't trigger scope denial."""
        await create_agent_policy(AgentPolicy(
            agent_id="no-scope",
            agent_name="No Scope Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        ))
        result = await enforce_policy("user1", "no-scope", "github", [])
        assert result.agent_id == "no-scope"

    @pytest.mark.asyncio
    async def test_unregistered_agent_denied(self, db):
        """Requesting a token for a non-existent agent is denied."""
        with pytest.raises(PolicyDenied, match="not registered"):
            await enforce_policy("user1", "nonexistent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_agent_with_no_services_denies_all(self, db):
        """An agent with empty allowed_services denies any service request."""
        await create_agent_policy(AgentPolicy(
            agent_id="no-svc",
            agent_name="No Service Bot",
            allowed_services=[],
            allowed_scopes={},
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        ))
        with pytest.raises(PolicyDenied, match="not authorized"):
            await enforce_policy("user1", "no-svc", "github", [])

    @pytest.mark.asyncio
    async def test_high_rate_limit_allows_many_requests(self, db):
        """A policy with rate_limit=100 allows 100 requests in succession."""
        await create_agent_policy(AgentPolicy(
            agent_id="high-rl",
            agent_name="High RL Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=100,
            created_by="user1",
            created_at=time.time(),
            is_active=True,
        ))
        for _ in range(100):
            result = await enforce_policy("user1", "high-rl", "github", ["repo"])
            assert result.agent_id == "high-rl"
        with pytest.raises(PolicyDenied, match="Rate limit"):
            await enforce_policy("user1", "high-rl", "github", ["repo"])
