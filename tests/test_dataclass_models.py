"""Tests for dataclass models — AgentPolicy, ApiKey, AuditEntry default values,
field types, and serialization behavior."""

import time

import pytest

from src.database import AgentPolicy, ApiKey, AuditEntry


class TestAgentPolicyDefaults:
    """AgentPolicy default field values."""

    def test_default_allowed_services_empty(self):
        p = AgentPolicy(agent_id="t", agent_name="t")
        assert p.allowed_services == []

    def test_default_allowed_scopes_empty(self):
        p = AgentPolicy(agent_id="t", agent_name="t")
        assert p.allowed_scopes == {}

    def test_default_rate_limit(self):
        p = AgentPolicy(agent_id="t", agent_name="t")
        assert p.rate_limit_per_minute == 60

    def test_default_requires_step_up_empty(self):
        p = AgentPolicy(agent_id="t", agent_name="t")
        assert p.requires_step_up == []

    def test_default_created_by_empty(self):
        p = AgentPolicy(agent_id="t", agent_name="t")
        assert p.created_by == ""

    def test_default_created_at_zero(self):
        p = AgentPolicy(agent_id="t", agent_name="t")
        assert p.created_at == 0.0

    def test_default_is_active_true(self):
        p = AgentPolicy(agent_id="t", agent_name="t")
        assert p.is_active is True

    def test_default_allowed_hours_empty(self):
        p = AgentPolicy(agent_id="t", agent_name="t")
        assert p.allowed_hours == []

    def test_default_allowed_days_empty(self):
        p = AgentPolicy(agent_id="t", agent_name="t")
        assert p.allowed_days == []

    def test_default_expires_at_zero(self):
        p = AgentPolicy(agent_id="t", agent_name="t")
        assert p.expires_at == 0.0

    def test_default_ip_allowlist_empty(self):
        p = AgentPolicy(agent_id="t", agent_name="t")
        assert p.ip_allowlist == []


class TestAgentPolicyCustomValues:
    """AgentPolicy with custom values."""

    def test_custom_services(self):
        p = AgentPolicy(agent_id="t", agent_name="t",
                         allowed_services=["github", "slack"])
        assert len(p.allowed_services) == 2

    def test_custom_scopes(self):
        scopes = {"github": ["repo", "read:user"]}
        p = AgentPolicy(agent_id="t", agent_name="t", allowed_scopes=scopes)
        assert p.allowed_scopes["github"] == ["repo", "read:user"]

    def test_custom_rate_limit(self):
        p = AgentPolicy(agent_id="t", agent_name="t", rate_limit_per_minute=100)
        assert p.rate_limit_per_minute == 100

    def test_custom_ip_allowlist(self):
        p = AgentPolicy(agent_id="t", agent_name="t",
                         ip_allowlist=["10.0.0.1", "192.168.0.0/16"])
        assert len(p.ip_allowlist) == 2

    def test_custom_time_windows(self):
        p = AgentPolicy(agent_id="t", agent_name="t",
                         allowed_hours=[9, 10, 11], allowed_days=[0, 1, 2])
        assert len(p.allowed_hours) == 3
        assert len(p.allowed_days) == 3


class TestAgentPolicyIsolation:
    """Dataclass field isolation (no shared mutable defaults)."""

    def test_services_not_shared(self):
        p1 = AgentPolicy(agent_id="a", agent_name="a")
        p2 = AgentPolicy(agent_id="b", agent_name="b")
        p1.allowed_services.append("github")
        assert p2.allowed_services == []

    def test_scopes_not_shared(self):
        p1 = AgentPolicy(agent_id="a", agent_name="a")
        p2 = AgentPolicy(agent_id="b", agent_name="b")
        p1.allowed_scopes["github"] = ["repo"]
        assert p2.allowed_scopes == {}

    def test_ip_allowlist_not_shared(self):
        p1 = AgentPolicy(agent_id="a", agent_name="a")
        p2 = AgentPolicy(agent_id="b", agent_name="b")
        p1.ip_allowlist.append("10.0.0.1")
        assert p2.ip_allowlist == []

    def test_hours_not_shared(self):
        p1 = AgentPolicy(agent_id="a", agent_name="a")
        p2 = AgentPolicy(agent_id="b", agent_name="b")
        p1.allowed_hours.append(9)
        assert p2.allowed_hours == []

    def test_days_not_shared(self):
        p1 = AgentPolicy(agent_id="a", agent_name="a")
        p2 = AgentPolicy(agent_id="b", agent_name="b")
        p1.allowed_days.append(0)
        assert p2.allowed_days == []

    def test_step_up_not_shared(self):
        p1 = AgentPolicy(agent_id="a", agent_name="a")
        p2 = AgentPolicy(agent_id="b", agent_name="b")
        p1.requires_step_up.append("slack")
        assert p2.requires_step_up == []


class TestApiKeyDefaults:
    """ApiKey default field values."""

    def test_default_id_empty(self):
        k = ApiKey()
        assert k.id == ""

    def test_default_key_hash_empty(self):
        k = ApiKey()
        assert k.key_hash == ""

    def test_default_key_prefix_empty(self):
        k = ApiKey()
        assert k.key_prefix == ""

    def test_default_user_id_empty(self):
        k = ApiKey()
        assert k.user_id == ""

    def test_default_agent_id_empty(self):
        k = ApiKey()
        assert k.agent_id == ""

    def test_default_name_empty(self):
        k = ApiKey()
        assert k.name == ""

    def test_default_created_at_zero(self):
        k = ApiKey()
        assert k.created_at == 0.0

    def test_default_expires_at_zero(self):
        k = ApiKey()
        assert k.expires_at == 0.0

    def test_default_is_revoked_false(self):
        k = ApiKey()
        assert k.is_revoked is False

    def test_default_last_used_at_zero(self):
        k = ApiKey()
        assert k.last_used_at == 0.0


class TestApiKeyCustomValues:
    """ApiKey with custom values."""

    def test_custom_values(self):
        k = ApiKey(
            id="key-1", key_hash="abc123", key_prefix="ag_abc",
            user_id="u1", agent_id="a1", name="my-key",
            created_at=1000.0, expires_at=2000.0,
            is_revoked=True, last_used_at=1500.0,
        )
        assert k.id == "key-1"
        assert k.key_hash == "abc123"
        assert k.user_id == "u1"
        assert k.is_revoked is True


class TestAuditEntryDefaults:
    """AuditEntry default field values."""

    def test_default_id_zero(self):
        e = AuditEntry()
        assert e.id == 0

    def test_default_timestamp_zero(self):
        e = AuditEntry()
        assert e.timestamp == 0.0

    def test_default_user_id_empty(self):
        e = AuditEntry()
        assert e.user_id == ""

    def test_default_agent_id_empty(self):
        e = AuditEntry()
        assert e.agent_id == ""

    def test_default_service_empty(self):
        e = AuditEntry()
        assert e.service == ""

    def test_default_scopes_empty(self):
        e = AuditEntry()
        assert e.scopes == ""

    def test_default_action_empty(self):
        e = AuditEntry()
        assert e.action == ""

    def test_default_status_empty(self):
        e = AuditEntry()
        assert e.status == ""

    def test_default_ip_address_empty(self):
        e = AuditEntry()
        assert e.ip_address == ""

    def test_default_details_empty(self):
        e = AuditEntry()
        assert e.details == ""


class TestAuditEntryCustomValues:
    """AuditEntry with custom values."""

    def test_custom_values(self):
        e = AuditEntry(
            id=42, timestamp=1000.0, user_id="u1", agent_id="a1",
            service="github", scopes="repo", action="token_request",
            status="success", ip_address="10.0.0.1", details="detail text",
        )
        assert e.id == 42
        assert e.service == "github"
        assert e.ip_address == "10.0.0.1"
