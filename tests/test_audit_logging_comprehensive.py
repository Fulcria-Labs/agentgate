"""Comprehensive audit logging tests — field completeness, ordering,
limits, user isolation, IP tracking, and details preservation."""

import time

import pytest

from src.database import (
    AuditEntry,
    get_audit_log,
    init_db,
    log_audit,
)


class TestAuditLogFieldCompleteness:
    """Verify all fields are stored and retrieved correctly."""

    @pytest.mark.asyncio
    async def test_all_fields_stored(self, db):
        await log_audit(
            user_id="u1", agent_id="a1", service="github",
            action="token_request", status="success",
            scopes="repo,read:user", ip_address="10.0.0.1",
            details="Test details",
        )
        entries = await get_audit_log("u1")
        assert len(entries) == 1
        e = entries[0]
        assert e.user_id == "u1"
        assert e.agent_id == "a1"
        assert e.service == "github"
        assert e.action == "token_request"
        assert e.status == "success"
        assert e.scopes == "repo,read:user"
        assert e.ip_address == "10.0.0.1"
        assert e.details == "Test details"

    @pytest.mark.asyncio
    async def test_timestamp_auto_set(self, db):
        before = time.time()
        await log_audit("u1", "a1", "github", "action", "status")
        entries = await get_audit_log("u1")
        assert entries[0].timestamp >= before

    @pytest.mark.asyncio
    async def test_id_auto_incremented(self, db):
        await log_audit("u1", "a1", "github", "a1", "s1")
        await log_audit("u1", "a1", "github", "a2", "s2")
        entries = await get_audit_log("u1")
        ids = {e.id for e in entries}
        assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_default_empty_scopes(self, db):
        await log_audit("u1", "a1", "github", "action", "status")
        entries = await get_audit_log("u1")
        assert entries[0].scopes == ""

    @pytest.mark.asyncio
    async def test_default_empty_ip(self, db):
        await log_audit("u1", "a1", "github", "action", "status")
        entries = await get_audit_log("u1")
        assert entries[0].ip_address == ""

    @pytest.mark.asyncio
    async def test_default_empty_details(self, db):
        await log_audit("u1", "a1", "github", "action", "status")
        entries = await get_audit_log("u1")
        assert entries[0].details == ""


class TestAuditLogOrdering:
    """Verify entries are ordered by timestamp descending."""

    @pytest.mark.asyncio
    async def test_most_recent_first(self, db):
        for i in range(5):
            await log_audit("u1", "a1", "github", f"action_{i}", "success")
        entries = await get_audit_log("u1")
        for j in range(len(entries) - 1):
            assert entries[j].timestamp >= entries[j + 1].timestamp

    @pytest.mark.asyncio
    async def test_order_with_different_services(self, db):
        await log_audit("u1", "a1", "github", "first", "success")
        await log_audit("u1", "a1", "slack", "second", "success")
        await log_audit("u1", "a1", "google", "third", "success")
        entries = await get_audit_log("u1")
        assert entries[0].service == "google"
        assert entries[1].service == "slack"
        assert entries[2].service == "github"


class TestAuditLogLimits:
    """Verify limit parameter behavior."""

    @pytest.mark.asyncio
    async def test_default_limit_50(self, db):
        for i in range(60):
            await log_audit("u1", "a1", "github", f"action_{i}", "success")
        entries = await get_audit_log("u1")
        assert len(entries) == 50

    @pytest.mark.asyncio
    async def test_custom_limit(self, db):
        for i in range(20):
            await log_audit("u1", "a1", "github", f"action_{i}", "success")
        entries = await get_audit_log("u1", limit=5)
        assert len(entries) == 5

    @pytest.mark.asyncio
    async def test_limit_larger_than_entries(self, db):
        for i in range(3):
            await log_audit("u1", "a1", "github", f"action_{i}", "success")
        entries = await get_audit_log("u1", limit=100)
        assert len(entries) == 3

    @pytest.mark.asyncio
    async def test_limit_of_one(self, db):
        for i in range(5):
            await log_audit("u1", "a1", "github", f"action_{i}", "success")
        entries = await get_audit_log("u1", limit=1)
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_limit_returns_most_recent(self, db):
        for i in range(10):
            await log_audit("u1", "a1", "github", f"action_{i}", "success")
        entries = await get_audit_log("u1", limit=1)
        assert entries[0].action == "action_9"


class TestAuditLogUserIsolation:
    """Audit entries are scoped to the requesting user."""

    @pytest.mark.asyncio
    async def test_different_users_isolated(self, db):
        await log_audit("user-a", "a1", "github", "action_a", "success")
        await log_audit("user-b", "a2", "slack", "action_b", "denied")
        a_entries = await get_audit_log("user-a")
        b_entries = await get_audit_log("user-b")
        assert len(a_entries) == 1
        assert len(b_entries) == 1
        assert a_entries[0].agent_id == "a1"
        assert b_entries[0].agent_id == "a2"

    @pytest.mark.asyncio
    async def test_no_entries_for_nonexistent_user(self, db):
        await log_audit("u1", "a1", "github", "action", "success")
        entries = await get_audit_log("nonexistent")
        assert entries == []


class TestAuditLogIPTracking:
    """IP address tracking in audit entries."""

    @pytest.mark.asyncio
    async def test_ipv4_stored(self, db):
        await log_audit("u1", "a1", "github", "action", "success", ip_address="192.168.1.1")
        entries = await get_audit_log("u1")
        assert entries[0].ip_address == "192.168.1.1"

    @pytest.mark.asyncio
    async def test_ipv6_stored(self, db):
        await log_audit("u1", "a1", "github", "action", "success", ip_address="::1")
        entries = await get_audit_log("u1")
        assert entries[0].ip_address == "::1"

    @pytest.mark.asyncio
    async def test_empty_ip_stored(self, db):
        await log_audit("u1", "a1", "github", "action", "success", ip_address="")
        entries = await get_audit_log("u1")
        assert entries[0].ip_address == ""


class TestAuditLogDetails:
    """Details field preservation."""

    @pytest.mark.asyncio
    async def test_long_details_stored(self, db):
        long_detail = "x" * 5000
        await log_audit("u1", "a1", "github", "action", "success", details=long_detail)
        entries = await get_audit_log("u1")
        assert entries[0].details == long_detail

    @pytest.mark.asyncio
    async def test_unicode_details_stored(self, db):
        await log_audit("u1", "a1", "github", "action", "success", details="Details: \u2603 \u2764")
        entries = await get_audit_log("u1")
        assert "\u2603" in entries[0].details

    @pytest.mark.asyncio
    async def test_json_like_details_stored(self, db):
        import json
        details = json.dumps({"key": "value", "count": 42})
        await log_audit("u1", "a1", "github", "action", "success", details=details)
        entries = await get_audit_log("u1")
        parsed = json.loads(entries[0].details)
        assert parsed["key"] == "value"


class TestAuditLogMultipleEntries:
    """Multiple audit entries behavior."""

    @pytest.mark.asyncio
    async def test_multiple_agents_tracked(self, db):
        for agent in ["agent-a", "agent-b", "agent-c"]:
            await log_audit("u1", agent, "github", "request", "success")
        entries = await get_audit_log("u1")
        agents = {e.agent_id for e in entries}
        assert agents == {"agent-a", "agent-b", "agent-c"}

    @pytest.mark.asyncio
    async def test_multiple_services_tracked(self, db):
        for svc in ["github", "slack", "google"]:
            await log_audit("u1", "a1", svc, "request", "success")
        entries = await get_audit_log("u1")
        services = {e.service for e in entries}
        assert services == {"github", "slack", "google"}

    @pytest.mark.asyncio
    async def test_multiple_statuses_tracked(self, db):
        for status in ["success", "denied", "error", "rate_limited"]:
            await log_audit("u1", "a1", "github", "request", status)
        entries = await get_audit_log("u1")
        statuses = {e.status for e in entries}
        assert statuses == {"success", "denied", "error", "rate_limited"}
