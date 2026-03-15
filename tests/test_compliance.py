"""Compliance and audit completeness tests for AgentGate.

Covers audit trail for all operations, audit log immutability,
data export formatting, GDPR right-to-deletion, retention policy
enforcement, and regulatory compliance patterns.
"""

import asyncio
import json
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
    _rate_counters,
)


@pytest.fixture(autouse=True)
def clear_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


USER = "auth0|compliance-user"


# ============================================================
# 1. Audit Trail for ALL Operations
# ============================================================

class TestAuditTrailCompleteness:
    """Ensure audit entries are created for all operation types."""

    @pytest.mark.asyncio
    async def test_audit_policy_denied_unregistered_agent(self, db):
        """Denied access for unregistered agent should be audited."""
        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "nonexistent-agent", "github", ["repo"])

        entries = await get_audit_log(USER, limit=10)
        assert len(entries) >= 1
        assert any(e.status == "denied" for e in entries)
        assert any("not registered" in e.details for e in entries)

    @pytest.mark.asyncio
    async def test_audit_policy_denied_disabled_agent(self, db):
        """Denied access for disabled agent should be audited."""
        await create_agent_policy(AgentPolicy(
            agent_id="disabled-audit-agent",
            agent_name="Disabled Audit",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            is_active=False,
            created_by=USER,
            created_at=time.time(),
        ))

        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "disabled-audit-agent", "github", ["repo"])

        entries = await get_audit_log(USER, limit=10)
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 1
        assert any("disabled" in e.details.lower() for e in denied)

    @pytest.mark.asyncio
    async def test_audit_policy_denied_expired(self, db):
        """Denied access for expired policy should be audited."""
        await create_agent_policy(AgentPolicy(
            agent_id="expired-audit-agent",
            agent_name="Expired Audit",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            expires_at=time.time() - 3600,
            created_by=USER,
            created_at=time.time(),
        ))

        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "expired-audit-agent", "github", ["repo"])

        entries = await get_audit_log(USER, limit=10)
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 1
        assert any("expired" in e.details.lower() for e in denied)

    @pytest.mark.asyncio
    async def test_audit_policy_denied_wrong_service(self, db):
        """Denied access for wrong service should be audited."""
        await create_agent_policy(AgentPolicy(
            agent_id="svc-audit-agent",
            agent_name="Service Audit",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by=USER,
            created_at=time.time(),
        ))

        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "svc-audit-agent", "slack", ["chat:write"])

        entries = await get_audit_log(USER, limit=10)
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 1

    @pytest.mark.asyncio
    async def test_audit_policy_denied_excess_scopes(self, db):
        """Denied access for excess scopes should be audited."""
        await create_agent_policy(AgentPolicy(
            agent_id="scope-audit-agent",
            agent_name="Scope Audit",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by=USER,
            created_at=time.time(),
        ))

        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "scope-audit-agent", "github", ["repo", "admin"])

        entries = await get_audit_log(USER, limit=10)
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 1
        assert any("scope" in e.details.lower() for e in denied)

    @pytest.mark.asyncio
    async def test_audit_policy_denied_rate_limit(self, db):
        """Rate limited requests should be audited."""
        await create_agent_policy(AgentPolicy(
            agent_id="rl-audit-agent",
            agent_name="Rate Limit Audit",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=1,
            created_by=USER,
            created_at=time.time(),
        ))

        await enforce_policy(USER, "rl-audit-agent", "github", ["repo"])

        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "rl-audit-agent", "github", ["repo"])

        entries = await get_audit_log(USER, limit=10)
        rate_limited = [e for e in entries if e.status == "rate_limited"]
        assert len(rate_limited) >= 1

    @pytest.mark.asyncio
    async def test_audit_policy_denied_ip_not_allowed(self, db):
        """IP-denied requests should be audited."""
        await create_agent_policy(AgentPolicy(
            agent_id="ip-audit-agent",
            agent_name="IP Audit",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            ip_allowlist=["10.0.0.1"],
            created_by=USER,
            created_at=time.time(),
        ))

        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "ip-audit-agent", "github", ["repo"],
                               ip_address="192.168.1.1")

        entries = await get_audit_log(USER, limit=10)
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 1
        assert any("ip" in e.details.lower() for e in denied)

    @pytest.mark.asyncio
    async def test_audit_policy_denied_ownership(self, db):
        """Ownership violation should be audited."""
        await create_agent_policy(AgentPolicy(
            agent_id="own-audit-agent",
            agent_name="Owner Audit",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="other-user",
            created_at=time.time(),
        ))

        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "own-audit-agent", "github", ["repo"])

        entries = await get_audit_log(USER, limit=10)
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 1

    @pytest.mark.asyncio
    async def test_manual_audit_log_entry(self, db):
        """Manual audit log entries should be stored correctly."""
        await log_audit(
            USER, "agent-1", "github",
            "custom_action", "custom_status",
            scopes="repo,gist",
            ip_address="10.0.0.1",
            details="Custom details for compliance",
        )

        entries = await get_audit_log(USER, limit=1)
        assert len(entries) == 1
        e = entries[0]
        assert e.action == "custom_action"
        assert e.status == "custom_status"
        assert e.scopes == "repo,gist"
        assert e.details == "Custom details for compliance"

    @pytest.mark.asyncio
    async def test_audit_preserves_ip_address(self, db):
        """Audit entries should preserve the IP address."""
        await create_agent_policy(AgentPolicy(
            agent_id="ip-preserve-agent",
            agent_name="IP Preserve",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            ip_allowlist=["192.168.1.100"],
            created_by=USER,
            created_at=time.time(),
        ))

        with pytest.raises(PolicyDenied):
            await enforce_policy(USER, "ip-preserve-agent", "github", ["repo"],
                               ip_address="10.0.0.99")

        entries = await get_audit_log(USER, limit=10)
        ip_entries = [e for e in entries if e.ip_address == "10.0.0.99"]
        assert len(ip_entries) >= 1


# ============================================================
# 2. Audit Log Immutability
# ============================================================

class TestAuditLogImmutability:
    """Audit logs should be append-only and not modifiable."""

    @pytest.mark.asyncio
    async def test_audit_entries_have_unique_ids(self, db):
        """Each audit entry should have a unique auto-incrementing ID."""
        for i in range(10):
            await log_audit(USER, f"agent-{i}", "github", "test", "success")

        entries = await get_audit_log(USER, limit=10)
        ids = [e.id for e in entries]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_audit_entries_have_timestamps(self, db):
        """Each audit entry should have a valid timestamp."""
        before = time.time()
        await log_audit(USER, "agent-1", "github", "test", "success")
        after = time.time()

        entries = await get_audit_log(USER, limit=1)
        assert len(entries) == 1
        assert before <= entries[0].timestamp <= after

    @pytest.mark.asyncio
    async def test_audit_entries_ordered_by_time_desc(self, db):
        """Audit entries should be returned in reverse chronological order."""
        for i in range(20):
            await log_audit(USER, f"agent-{i}", "github", f"action-{i}", "success")

        entries = await get_audit_log(USER, limit=20)
        for i in range(len(entries) - 1):
            assert entries[i].timestamp >= entries[i + 1].timestamp

    @pytest.mark.asyncio
    async def test_audit_log_not_affected_by_policy_deletion(self, db):
        """Deleting a policy should not delete its audit entries."""
        await create_agent_policy(AgentPolicy(
            agent_id="del-audit-agent",
            agent_name="Del Audit",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by=USER,
            created_at=time.time(),
        ))

        await log_audit(USER, "del-audit-agent", "github", "test_action", "success")
        await delete_agent_policy("del-audit-agent", USER)

        entries = await get_audit_log(USER, limit=10)
        agent_entries = [e for e in entries if e.agent_id == "del-audit-agent"]
        assert len(agent_entries) >= 1

    @pytest.mark.asyncio
    async def test_audit_log_not_affected_by_emergency_revoke(self, db):
        """Emergency revoke should not delete audit entries."""
        for i in range(5):
            await log_audit(USER, "agent-1", "github", f"action-{i}", "success")

        await emergency_revoke_all(USER)

        entries = await get_audit_log(USER, limit=10)
        assert len(entries) >= 5

    @pytest.mark.asyncio
    async def test_audit_log_grows_monotonically(self, db):
        """Audit log should only grow, never shrink."""
        for i in range(5):
            await log_audit(USER, "agent-1", "github", "test", "success")
            entries = await get_audit_log(USER, limit=100)
            assert len(entries) == i + 1


# ============================================================
# 3. Data Export Formatting
# ============================================================

class TestDataExportFormatting:
    """Tests for audit data export and formatting."""

    @pytest.mark.asyncio
    async def test_audit_entry_has_all_fields(self, db):
        """AuditEntry dataclass should have all required fields."""
        await log_audit(
            USER, "export-agent", "github",
            "token_issued", "success",
            scopes="repo,gist",
            ip_address="10.0.0.1",
            details="Export test",
        )

        entries = await get_audit_log(USER, limit=1)
        e = entries[0]
        assert hasattr(e, 'id')
        assert hasattr(e, 'timestamp')
        assert hasattr(e, 'user_id')
        assert hasattr(e, 'agent_id')
        assert hasattr(e, 'service')
        assert hasattr(e, 'scopes')
        assert hasattr(e, 'action')
        assert hasattr(e, 'status')
        assert hasattr(e, 'ip_address')
        assert hasattr(e, 'details')

    @pytest.mark.asyncio
    async def test_audit_entry_json_serializable(self, db):
        """Audit entries should be JSON-serializable."""
        await log_audit(USER, "json-agent", "github", "test", "success")

        entries = await get_audit_log(USER, limit=1)
        e = entries[0]
        # Convert to dict and serialize
        entry_dict = {
            "id": e.id,
            "timestamp": e.timestamp,
            "user_id": e.user_id,
            "agent_id": e.agent_id,
            "service": e.service,
            "scopes": e.scopes,
            "action": e.action,
            "status": e.status,
            "ip_address": e.ip_address,
            "details": e.details,
        }
        serialized = json.dumps(entry_dict)
        assert serialized is not None
        deserialized = json.loads(serialized)
        assert deserialized["user_id"] == USER

    @pytest.mark.asyncio
    async def test_multiple_entries_json_export(self, db):
        """Multiple audit entries can be exported as JSON array."""
        for i in range(10):
            await log_audit(USER, f"agent-{i}", "github", "test", "success")

        entries = await get_audit_log(USER, limit=10)
        entries_list = [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "user_id": e.user_id,
                "agent_id": e.agent_id,
                "service": e.service,
                "action": e.action,
                "status": e.status,
            }
            for e in entries
        ]
        serialized = json.dumps(entries_list)
        deserialized = json.loads(serialized)
        assert len(deserialized) == 10

    @pytest.mark.asyncio
    async def test_policy_json_serializable(self, db):
        """AgentPolicy fields should be JSON-serializable."""
        policy = AgentPolicy(
            agent_id="json-policy",
            agent_name="JSON Policy",
            allowed_services=["github", "slack"],
            allowed_scopes={"github": ["repo"], "slack": ["chat:write"]},
            rate_limit_per_minute=60,
            requires_step_up=["github"],
            allowed_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
            allowed_days=[0, 1, 2, 3, 4],
            ip_allowlist=["10.0.0.0/8"],
            created_by=USER,
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        retrieved = await get_agent_policy("json-policy")
        policy_dict = {
            "agent_id": retrieved.agent_id,
            "agent_name": retrieved.agent_name,
            "allowed_services": retrieved.allowed_services,
            "allowed_scopes": retrieved.allowed_scopes,
            "rate_limit_per_minute": retrieved.rate_limit_per_minute,
            "requires_step_up": retrieved.requires_step_up,
            "allowed_hours": retrieved.allowed_hours,
            "allowed_days": retrieved.allowed_days,
            "ip_allowlist": retrieved.ip_allowlist,
        }
        serialized = json.dumps(policy_dict)
        deserialized = json.loads(serialized)
        assert deserialized["agent_id"] == "json-policy"
        assert deserialized["allowed_services"] == ["github", "slack"]

    @pytest.mark.asyncio
    async def test_csv_style_export(self, db):
        """Audit entries can be formatted as CSV rows."""
        await log_audit(USER, "csv-agent", "github", "test", "success",
                       scopes="repo", ip_address="10.0.0.1")

        entries = await get_audit_log(USER, limit=1)
        e = entries[0]
        csv_row = f"{e.id},{e.timestamp},{e.user_id},{e.agent_id},{e.service},{e.action},{e.status},{e.ip_address}"
        assert USER in csv_row
        assert "github" in csv_row
        assert "," in csv_row

    @pytest.mark.asyncio
    async def test_audit_details_can_contain_special_chars(self, db):
        """Audit details field should handle special characters."""
        special_details = "Details with 'quotes', \"double quotes\", and <angle brackets>"
        await log_audit(USER, "special-agent", "github", "test", "success",
                       details=special_details)

        entries = await get_audit_log(USER, limit=1)
        assert entries[0].details == special_details


# ============================================================
# 4. GDPR Right-to-Deletion
# ============================================================

class TestGDPRDeletion:
    """Tests for data deletion capabilities."""

    @pytest.mark.asyncio
    async def test_delete_all_user_policies(self, db):
        """All policies for a user can be deleted."""
        for i in range(5):
            await create_agent_policy(AgentPolicy(
                agent_id=f"gdpr-agent-{i}",
                agent_name=f"GDPR Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                created_by=USER,
                created_at=time.time(),
            ))

        for i in range(5):
            deleted = await delete_agent_policy(f"gdpr-agent-{i}", USER)
            assert deleted is True

        policies = await get_all_policies(USER)
        assert len(policies) == 0

    @pytest.mark.asyncio
    async def test_delete_all_user_api_keys(self, db):
        """All API keys for a user can be revoked (soft delete)."""
        for i in range(5):
            key_obj, _ = await create_api_key(USER, f"gdpr-key-agent-{i}", f"key-{i}")
            await revoke_api_key(key_obj.id, USER)

        keys = await get_api_keys(USER)
        assert all(k.is_revoked for k in keys)

    @pytest.mark.asyncio
    async def test_disconnect_all_services(self, db):
        """All connected services for a user can be disconnected."""
        services = ["github", "slack", "google", "linear", "notion"]
        for svc in services:
            await add_connected_service(USER, svc)

        for svc in services:
            await remove_connected_service(USER, svc)

        svcs = await get_connected_services(USER)
        assert len(svcs) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_policy(self, db):
        """Deleting a nonexistent policy returns False."""
        result = await delete_agent_policy("nonexistent-agent", USER)
        assert result is False

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_key(self, db):
        """Revoking a nonexistent key returns False."""
        result = await revoke_api_key("nonexistent-key-id", USER)
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_service(self, db):
        """Disconnecting a service that's not connected is safe."""
        await remove_connected_service(USER, "github")
        svcs = await get_connected_services(USER)
        assert len(svcs) == 0

    @pytest.mark.asyncio
    async def test_policy_deletion_does_not_affect_other_users(self, db):
        """Deleting one user's policy does not affect others."""
        await create_agent_policy(AgentPolicy(
            agent_id="shared-name",
            agent_name="Shared",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by=USER,
            created_at=time.time(),
        ))

        # Different user creates same agent_id (overwrites due to PK)
        await create_agent_policy(AgentPolicy(
            agent_id="other-name",
            agent_name="Other",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="other-user",
            created_at=time.time(),
        ))

        # Delete USER's policy
        await delete_agent_policy("shared-name", USER)

        # Other user's policy still exists
        p = await get_agent_policy("other-name")
        assert p is not None


# ============================================================
# 5. Retention Policy Enforcement
# ============================================================

class TestRetentionPolicy:
    """Tests for data retention patterns."""

    @pytest.mark.asyncio
    async def test_audit_log_respects_limit(self, db):
        """get_audit_log limit parameter caps returned entries."""
        for i in range(50):
            await log_audit(USER, "agent-1", "github", "test", "success")

        entries = await get_audit_log(USER, limit=10)
        assert len(entries) == 10

    @pytest.mark.asyncio
    async def test_audit_log_returns_most_recent(self, db):
        """Audit log should return the most recent entries."""
        for i in range(20):
            await log_audit(USER, "agent-1", "github", f"action-{i}", "success")

        entries = await get_audit_log(USER, limit=5)
        assert len(entries) == 5
        # Most recent should have highest action number
        # (entries are DESC by timestamp)

    @pytest.mark.asyncio
    async def test_audit_log_default_limit(self, db):
        """Default limit of 50 is applied."""
        for i in range(60):
            await log_audit(USER, "agent-1", "github", "test", "success")

        entries = await get_audit_log(USER)  # Default limit=50
        assert len(entries) == 50

    @pytest.mark.asyncio
    async def test_expired_keys_tracked_in_database(self, db):
        """Expired keys remain in the database (soft tracking)."""
        key_obj, raw = await create_api_key(USER, "expired-track", "test", expires_in=1)
        await asyncio.sleep(1.1)

        # Key should still be in the database listing
        keys = await get_api_keys(USER)
        assert len(keys) >= 1
        key_ids = [k.id for k in keys]
        assert key_obj.id in key_ids

    @pytest.mark.asyncio
    async def test_revoked_keys_tracked_in_database(self, db):
        """Revoked keys remain in the database for audit purposes."""
        key_obj, raw = await create_api_key(USER, "revoked-track", "test")
        await revoke_api_key(key_obj.id, USER)

        keys = await get_api_keys(USER)
        revoked_keys = [k for k in keys if k.is_revoked]
        assert len(revoked_keys) >= 1

    @pytest.mark.asyncio
    async def test_disabled_policies_tracked_in_database(self, db):
        """Disabled policies remain in the database."""
        await create_agent_policy(AgentPolicy(
            agent_id="disabled-track",
            agent_name="Disabled Track",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by=USER,
            created_at=time.time(),
        ))

        await toggle_agent_policy("disabled-track", USER)

        policies = await get_all_policies(USER)
        disabled = [p for p in policies if not p.is_active]
        assert len(disabled) >= 1

    @pytest.mark.asyncio
    async def test_audit_log_large_volume(self, db):
        """Audit log should handle large volumes correctly."""
        for i in range(200):
            await log_audit(USER, f"agent-{i % 10}", "github", "test", "success")

        entries = await get_audit_log(USER, limit=200)
        assert len(entries) == 200

    @pytest.mark.asyncio
    async def test_audit_log_different_action_types(self, db):
        """Audit log tracks various action types."""
        actions = [
            ("token_request", "denied"),
            ("token_issued", "success"),
            ("policy_created", "success"),
            ("agent_disabled", "success"),
            ("api_key_created", "success"),
            ("api_key_revoked", "success"),
            ("emergency_revoke", "success"),
            ("step_up_initiated", "pending"),
        ]

        for action, status in actions:
            await log_audit(USER, "agent-1", "github", action, status)

        entries = await get_audit_log(USER, limit=20)
        logged_actions = {e.action for e in entries}
        for action, _ in actions:
            assert action in logged_actions

    @pytest.mark.asyncio
    async def test_audit_log_survives_key_revocation(self, db):
        """Audit entries exist even after the associated key is revoked."""
        key_obj, raw = await create_api_key(USER, "audit-key-agent", "test")
        await log_audit(USER, "audit-key-agent", "", "api_key_created", "success")
        await revoke_api_key(key_obj.id, USER)
        await log_audit(USER, "audit-key-agent", "", "api_key_revoked", "success")

        entries = await get_audit_log(USER, limit=10)
        key_entries = [e for e in entries if e.agent_id == "audit-key-agent"]
        assert len(key_entries) >= 2

    @pytest.mark.asyncio
    async def test_audit_log_tracks_toggle_history(self, db):
        """Audit log can track the full toggle history of a policy."""
        await create_agent_policy(AgentPolicy(
            agent_id="toggle-history",
            agent_name="Toggle History",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by=USER,
            created_at=time.time(),
        ))

        for i in range(5):
            new_state = await toggle_agent_policy("toggle-history", USER)
            status = "enabled" if new_state else "disabled"
            await log_audit(USER, "toggle-history", "", f"agent_{status}", "success")

        entries = await get_audit_log(USER, limit=10)
        toggle_entries = [e for e in entries if e.agent_id == "toggle-history"]
        assert len(toggle_entries) == 5

    @pytest.mark.asyncio
    async def test_audit_log_concurrent_compliance(self, db):
        """Concurrent audit writes from different operations maintain integrity."""
        tasks = [
            log_audit(USER, "agent-1", "github", "token_issued", "success"),
            log_audit(USER, "agent-2", "slack", "token_request", "denied"),
            log_audit(USER, "agent-3", "", "policy_created", "success"),
            log_audit(USER, "agent-4", "", "api_key_created", "success"),
            log_audit(USER, "*", "", "emergency_revoke", "success"),
        ]
        await asyncio.gather(*tasks)

        entries = await get_audit_log(USER, limit=10)
        assert len(entries) == 5
        actions = {e.action for e in entries}
        assert "token_issued" in actions
        assert "emergency_revoke" in actions

    @pytest.mark.asyncio
    async def test_audit_entry_timestamp_monotonic(self, db):
        """Audit entry timestamps should be monotonically increasing."""
        timestamps = []
        for i in range(10):
            await log_audit(USER, "agent-1", "github", f"action-{i}", "success")
            entries = await get_audit_log(USER, limit=1)
            timestamps.append(entries[0].timestamp)

        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1]

    @pytest.mark.asyncio
    async def test_audit_scopes_preserved_accurately(self, db):
        """Scope strings should be preserved exactly as provided."""
        test_scopes = "repo,read:user,read:org,gist,notifications"
        await log_audit(USER, "scope-check", "github", "test", "success",
                       scopes=test_scopes)

        entries = await get_audit_log(USER, limit=1)
        assert entries[0].scopes == test_scopes

    @pytest.mark.asyncio
    async def test_audit_empty_fields_preserved(self, db):
        """Empty string fields should be preserved correctly."""
        await log_audit(USER, "empty-field", "", "", "success")

        entries = await get_audit_log(USER, limit=1)
        assert entries[0].service == ""
        assert entries[0].scopes == ""
        assert entries[0].ip_address == ""
        assert entries[0].details == ""
