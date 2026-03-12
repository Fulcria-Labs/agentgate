"""Tests for connected services, API key lifecycle, and emergency revoke operations."""

import hashlib
import time

import pytest
import pytest_asyncio

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
    _hash_key,
)


# ── Connected Services ──────────────────────────────────────────────


class TestConnectedServices:
    """Tests for add/remove/get connected services."""

    @pytest.mark.asyncio
    async def test_add_and_get_service(self, db):
        await add_connected_service("user1", "github")
        services = await get_connected_services("user1")
        assert len(services) == 1
        assert services[0]["service"] == "github"

    @pytest.mark.asyncio
    async def test_add_multiple_services(self, db):
        await add_connected_service("user1", "github")
        await add_connected_service("user1", "slack")
        await add_connected_service("user1", "jira")
        services = await get_connected_services("user1")
        assert len(services) == 3
        service_names = {s["service"] for s in services}
        assert service_names == {"github", "slack", "jira"}

    @pytest.mark.asyncio
    async def test_services_ordered_by_connected_at_desc(self, db):
        await add_connected_service("user1", "first")
        await add_connected_service("user1", "second")
        await add_connected_service("user1", "third")
        services = await get_connected_services("user1")
        # Most recently connected first
        timestamps = [s["connected_at"] for s in services]
        assert timestamps == sorted(timestamps, reverse=True)

    @pytest.mark.asyncio
    async def test_service_with_connection_id(self, db):
        await add_connected_service("user1", "github", connection_id="conn_abc123")
        services = await get_connected_services("user1")
        assert services[0]["connection_id"] == "conn_abc123"

    @pytest.mark.asyncio
    async def test_service_default_empty_connection_id(self, db):
        await add_connected_service("user1", "github")
        services = await get_connected_services("user1")
        assert services[0]["connection_id"] == ""

    @pytest.mark.asyncio
    async def test_remove_service(self, db):
        await add_connected_service("user1", "github")
        await add_connected_service("user1", "slack")
        await remove_connected_service("user1", "github")
        services = await get_connected_services("user1")
        assert len(services) == 1
        assert services[0]["service"] == "slack"

    @pytest.mark.asyncio
    async def test_remove_nonexistent_service_no_error(self, db):
        """Removing a service that doesn't exist should not raise."""
        await remove_connected_service("user1", "nonexistent")

    @pytest.mark.asyncio
    async def test_services_isolated_by_user(self, db):
        await add_connected_service("user1", "github")
        await add_connected_service("user2", "slack")
        services1 = await get_connected_services("user1")
        services2 = await get_connected_services("user2")
        assert len(services1) == 1
        assert services1[0]["service"] == "github"
        assert len(services2) == 1
        assert services2[0]["service"] == "slack"

    @pytest.mark.asyncio
    async def test_upsert_same_service(self, db):
        """Adding the same service again should update, not duplicate."""
        await add_connected_service("user1", "github", connection_id="old")
        await add_connected_service("user1", "github", connection_id="new")
        services = await get_connected_services("user1")
        assert len(services) == 1
        assert services[0]["connection_id"] == "new"

    @pytest.mark.asyncio
    async def test_empty_services_returns_empty_list(self, db):
        services = await get_connected_services("user_with_no_services")
        assert services == []


# ── API Key Lifecycle ────────────────────────────────────────────────


class TestApiKeyCreation:
    """Tests for API key creation."""

    @pytest.mark.asyncio
    async def test_create_returns_key_and_raw(self, db):
        api_key, raw_key = await create_api_key("user1", "agent1")
        assert isinstance(api_key, ApiKey)
        assert isinstance(raw_key, str)

    @pytest.mark.asyncio
    async def test_raw_key_starts_with_prefix(self, db):
        _, raw_key = await create_api_key("user1", "agent1")
        assert raw_key.startswith("ag_")

    @pytest.mark.asyncio
    async def test_key_prefix_is_first_8_chars(self, db):
        api_key, raw_key = await create_api_key("user1", "agent1")
        assert api_key.key_prefix == raw_key[:8]

    @pytest.mark.asyncio
    async def test_key_hash_matches_raw(self, db):
        api_key, raw_key = await create_api_key("user1", "agent1")
        expected_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        assert api_key.key_hash == expected_hash

    @pytest.mark.asyncio
    async def test_key_has_user_and_agent(self, db):
        api_key, _ = await create_api_key("user1", "agent1")
        assert api_key.user_id == "user1"
        assert api_key.agent_id == "agent1"

    @pytest.mark.asyncio
    async def test_key_with_name(self, db):
        api_key, _ = await create_api_key("user1", "agent1", name="Production Key")
        assert api_key.name == "Production Key"

    @pytest.mark.asyncio
    async def test_key_default_no_expiration(self, db):
        api_key, _ = await create_api_key("user1", "agent1")
        assert api_key.expires_at == 0

    @pytest.mark.asyncio
    async def test_key_with_expiration(self, db):
        api_key, _ = await create_api_key("user1", "agent1", expires_in=3600)
        assert api_key.expires_at > 0
        # Should expire approximately 1 hour from now
        assert abs(api_key.expires_at - (time.time() + 3600)) < 5

    @pytest.mark.asyncio
    async def test_each_key_is_unique(self, db):
        _, raw1 = await create_api_key("user1", "agent1")
        _, raw2 = await create_api_key("user1", "agent1")
        assert raw1 != raw2

    @pytest.mark.asyncio
    async def test_key_not_revoked_on_creation(self, db):
        api_key, _ = await create_api_key("user1", "agent1")
        assert api_key.is_revoked is False


class TestApiKeyValidation:
    """Tests for API key validation."""

    @pytest.mark.asyncio
    async def test_valid_key_returns_api_key(self, db):
        _, raw_key = await create_api_key("user1", "agent1")
        result = await validate_api_key(raw_key)
        assert result is not None
        assert result.user_id == "user1"

    @pytest.mark.asyncio
    async def test_invalid_key_returns_none(self, db):
        result = await validate_api_key("ag_this_is_not_a_real_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_validation_updates_last_used(self, db):
        _, raw_key = await create_api_key("user1", "agent1")
        before = time.time()
        result = await validate_api_key(raw_key)
        assert result is not None
        assert result.last_used_at >= before

    @pytest.mark.asyncio
    async def test_revoked_key_returns_none(self, db):
        api_key, raw_key = await create_api_key("user1", "agent1")
        await revoke_api_key(api_key.id, "user1")
        result = await validate_api_key(raw_key)
        assert result is None

    @pytest.mark.asyncio
    async def test_expired_key_returns_none(self, db):
        # Create key that expires in 1 second
        api_key, raw_key = await create_api_key("user1", "agent1", expires_in=-1)
        # Key should already be expired
        result = await validate_api_key(raw_key)
        assert result is None

    @pytest.mark.asyncio
    async def test_non_expiring_key_always_valid(self, db):
        _, raw_key = await create_api_key("user1", "agent1", expires_in=0)
        result = await validate_api_key(raw_key)
        assert result is not None


class TestApiKeyRevocation:
    """Tests for API key revocation."""

    @pytest.mark.asyncio
    async def test_revoke_own_key_succeeds(self, db):
        api_key, _ = await create_api_key("user1", "agent1")
        result = await revoke_api_key(api_key.id, "user1")
        assert result is True

    @pytest.mark.asyncio
    async def test_revoke_other_user_key_fails(self, db):
        api_key, _ = await create_api_key("user1", "agent1")
        result = await revoke_api_key(api_key.id, "user2")
        assert result is False

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_key_fails(self, db):
        result = await revoke_api_key("fake_id", "user1")
        assert result is False

    @pytest.mark.asyncio
    async def test_double_revoke(self, db):
        api_key, _ = await create_api_key("user1", "agent1")
        await revoke_api_key(api_key.id, "user1")
        # Second revoke still "succeeds" (row exists, already revoked)
        result = await revoke_api_key(api_key.id, "user1")
        # is_revoked is already 1, UPDATE matches but doesn't change
        assert isinstance(result, bool)


class TestGetApiKeys:
    """Tests for listing API keys."""

    @pytest.mark.asyncio
    async def test_get_keys_for_user(self, db):
        await create_api_key("user1", "agent1", name="Key 1")
        await create_api_key("user1", "agent2", name="Key 2")
        keys = await get_api_keys("user1")
        assert len(keys) == 2

    @pytest.mark.asyncio
    async def test_keys_exclude_hash(self, db):
        await create_api_key("user1", "agent1")
        keys = await get_api_keys("user1")
        assert keys[0].key_hash == ""

    @pytest.mark.asyncio
    async def test_keys_ordered_by_created_desc(self, db):
        await create_api_key("user1", "agent1", name="First")
        await create_api_key("user1", "agent2", name="Second")
        keys = await get_api_keys("user1")
        assert keys[0].created_at >= keys[1].created_at

    @pytest.mark.asyncio
    async def test_keys_isolated_by_user(self, db):
        await create_api_key("user1", "agent1")
        await create_api_key("user2", "agent2")
        keys1 = await get_api_keys("user1")
        keys2 = await get_api_keys("user2")
        assert len(keys1) == 1
        assert len(keys2) == 1

    @pytest.mark.asyncio
    async def test_includes_revoked_keys(self, db):
        api_key, _ = await create_api_key("user1", "agent1")
        await revoke_api_key(api_key.id, "user1")
        keys = await get_api_keys("user1")
        assert len(keys) == 1
        assert keys[0].is_revoked is True

    @pytest.mark.asyncio
    async def test_no_keys_returns_empty(self, db):
        keys = await get_api_keys("user_no_keys")
        assert keys == []


# ── Hash Key Function ────────────────────────────────────────────────


class TestHashKey:
    """Tests for the _hash_key helper."""

    def test_returns_sha256_hex(self):
        result = _hash_key("test_key")
        expected = hashlib.sha256("test_key".encode()).hexdigest()
        assert result == expected

    def test_deterministic(self):
        assert _hash_key("same_key") == _hash_key("same_key")

    def test_different_keys_different_hashes(self):
        assert _hash_key("key1") != _hash_key("key2")

    def test_hash_length_is_64(self):
        # SHA-256 hex digest is always 64 chars
        assert len(_hash_key("any_key")) == 64

    def test_empty_string(self):
        result = _hash_key("")
        assert isinstance(result, str)
        assert len(result) == 64


# ── Emergency Revoke ─────────────────────────────────────────────────


class TestEmergencyRevoke:
    """Tests for the emergency kill switch."""

    @pytest.mark.asyncio
    async def test_disables_all_policies(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="a1", agent_name="Agent 1", created_by="user1",
            allowed_services=["github"], is_active=True,
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="a2", agent_name="Agent 2", created_by="user1",
            allowed_services=["slack"], is_active=True,
        ))
        result = await emergency_revoke_all("user1")
        assert result["policies_disabled"] == 2

        # Verify policies are disabled
        p1 = await get_agent_policy("a1")
        p2 = await get_agent_policy("a2")
        assert p1.is_active is False
        assert p2.is_active is False

    @pytest.mark.asyncio
    async def test_revokes_all_api_keys(self, db):
        await create_api_key("user1", "agent1")
        await create_api_key("user1", "agent2")
        result = await emergency_revoke_all("user1")
        assert result["keys_revoked"] == 2

    @pytest.mark.asyncio
    async def test_does_not_affect_other_users(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="a1", agent_name="Agent 1", created_by="user1",
            allowed_services=["github"], is_active=True,
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="a2", agent_name="Agent 2", created_by="user2",
            allowed_services=["slack"], is_active=True,
        ))
        await create_api_key("user1", "agent1")
        await create_api_key("user2", "agent2")

        await emergency_revoke_all("user1")

        # user2's policy and key should be unaffected
        p2 = await get_agent_policy("a2")
        assert p2.is_active is True
        keys2 = await get_api_keys("user2")
        assert keys2[0].is_revoked is False

    @pytest.mark.asyncio
    async def test_already_revoked_keys_not_counted(self, db):
        api_key, _ = await create_api_key("user1", "agent1")
        await revoke_api_key(api_key.id, "user1")
        await create_api_key("user1", "agent2")

        result = await emergency_revoke_all("user1")
        # Only the non-revoked key should be counted
        assert result["keys_revoked"] == 1

    @pytest.mark.asyncio
    async def test_no_resources_returns_zeros(self, db):
        result = await emergency_revoke_all("user_with_nothing")
        assert result["policies_disabled"] == 0
        assert result["keys_revoked"] == 0

    @pytest.mark.asyncio
    async def test_keys_invalid_after_emergency_revoke(self, db):
        _, raw_key = await create_api_key("user1", "agent1")
        # Key should work before
        assert await validate_api_key(raw_key) is not None

        await emergency_revoke_all("user1")
        # Key should be invalid after
        assert await validate_api_key(raw_key) is None


# ── Policy CRUD ──────────────────────────────────────────────────────


class TestPolicyCRUD:
    """Tests for policy create/read/update/delete operations."""

    @pytest.mark.asyncio
    async def test_create_and_get_policy(self, db):
        policy = AgentPolicy(
            agent_id="a1", agent_name="Test Agent", created_by="user1",
            allowed_services=["github", "slack"],
            allowed_scopes={"github": ["read", "write"]},
        )
        await create_agent_policy(policy)
        result = await get_agent_policy("a1")
        assert result is not None
        assert result.agent_name == "Test Agent"
        assert result.allowed_services == ["github", "slack"]

    @pytest.mark.asyncio
    async def test_get_nonexistent_policy(self, db):
        result = await get_agent_policy("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_policies_by_user(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="a1", agent_name="Agent 1", created_by="user1",
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="a2", agent_name="Agent 2", created_by="user1",
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="a3", agent_name="Agent 3", created_by="user2",
        ))
        policies = await get_all_policies("user1")
        assert len(policies) == 2

    @pytest.mark.asyncio
    async def test_toggle_policy(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="a1", agent_name="Agent 1", created_by="user1", is_active=True,
        ))
        new_state = await toggle_agent_policy("a1", "user1")
        assert new_state is False

        # Toggle back
        new_state = await toggle_agent_policy("a1", "user1")
        assert new_state is True

    @pytest.mark.asyncio
    async def test_toggle_nonexistent_policy(self, db):
        result = await toggle_agent_policy("nonexistent", "user1")
        assert result is None

    @pytest.mark.asyncio
    async def test_toggle_wrong_user(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="a1", agent_name="Agent 1", created_by="user1",
        ))
        result = await toggle_agent_policy("a1", "user2")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_policy(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="a1", agent_name="Agent 1", created_by="user1",
        ))
        result = await delete_agent_policy("a1", "user1")
        assert result is True
        assert await get_agent_policy("a1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_policy(self, db):
        result = await delete_agent_policy("nonexistent", "user1")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_wrong_user(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="a1", agent_name="Agent 1", created_by="user1",
        ))
        result = await delete_agent_policy("a1", "user2")
        assert result is False
        # Policy should still exist
        assert await get_agent_policy("a1") is not None

    @pytest.mark.asyncio
    async def test_policy_upsert(self, db):
        """Creating a policy with the same agent_id should replace it."""
        await create_agent_policy(AgentPolicy(
            agent_id="a1", agent_name="Original", created_by="user1",
            allowed_services=["github"],
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="a1", agent_name="Updated", created_by="user1",
            allowed_services=["github", "slack"],
        ))
        result = await get_agent_policy("a1")
        assert result.agent_name == "Updated"
        assert len(result.allowed_services) == 2

    @pytest.mark.asyncio
    async def test_policy_preserves_json_fields(self, db):
        """Verify JSON serialization/deserialization of complex fields."""
        policy = AgentPolicy(
            agent_id="a1", agent_name="Agent", created_by="user1",
            allowed_services=["github", "slack", "jira"],
            allowed_scopes={"github": ["read", "write"], "slack": ["chat:write"]},
            requires_step_up=["banking", "admin"],
            allowed_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
            allowed_days=[0, 1, 2, 3, 4],
            ip_allowlist=["10.0.0.0/8", "192.168.1.1"],
        )
        await create_agent_policy(policy)
        result = await get_agent_policy("a1")
        assert result.allowed_services == ["github", "slack", "jira"]
        assert result.allowed_scopes == {"github": ["read", "write"], "slack": ["chat:write"]}
        assert result.requires_step_up == ["banking", "admin"]
        assert result.allowed_hours == [9, 10, 11, 12, 13, 14, 15, 16, 17]
        assert result.allowed_days == [0, 1, 2, 3, 4]
        assert result.ip_allowlist == ["10.0.0.0/8", "192.168.1.1"]


# ── Audit Log ────────────────────────────────────────────────────────


class TestAuditLog:
    """Tests for audit log operations."""

    @pytest.mark.asyncio
    async def test_log_and_retrieve(self, db):
        await log_audit("user1", "agent1", "github", "token_request", "granted")
        entries = await get_audit_log("user1")
        assert len(entries) == 1
        assert entries[0].agent_id == "agent1"
        assert entries[0].action == "token_request"
        assert entries[0].status == "granted"

    @pytest.mark.asyncio
    async def test_log_with_all_fields(self, db):
        await log_audit(
            "user1", "agent1", "github", "token_request", "denied",
            scopes="read,write", ip_address="10.0.0.1",
            details="Rate limit exceeded",
        )
        entries = await get_audit_log("user1")
        assert entries[0].scopes == "read,write"
        assert entries[0].ip_address == "10.0.0.1"
        assert entries[0].details == "Rate limit exceeded"

    @pytest.mark.asyncio
    async def test_audit_log_ordered_by_timestamp_desc(self, db):
        await log_audit("user1", "agent1", "s1", "a1", "ok")
        await log_audit("user1", "agent1", "s2", "a2", "ok")
        await log_audit("user1", "agent1", "s3", "a3", "ok")
        entries = await get_audit_log("user1")
        timestamps = [e.timestamp for e in entries]
        assert timestamps == sorted(timestamps, reverse=True)

    @pytest.mark.asyncio
    async def test_audit_log_respects_limit(self, db):
        for i in range(10):
            await log_audit("user1", f"agent{i}", "svc", "act", "ok")
        entries = await get_audit_log("user1", limit=5)
        assert len(entries) == 5

    @pytest.mark.asyncio
    async def test_audit_log_default_limit_50(self, db):
        for i in range(60):
            await log_audit("user1", f"agent{i}", "svc", "act", "ok")
        entries = await get_audit_log("user1")
        assert len(entries) == 50

    @pytest.mark.asyncio
    async def test_audit_log_isolated_by_user(self, db):
        await log_audit("user1", "agent1", "svc", "act", "ok")
        await log_audit("user2", "agent2", "svc", "act", "ok")
        entries1 = await get_audit_log("user1")
        entries2 = await get_audit_log("user2")
        assert len(entries1) == 1
        assert len(entries2) == 1

    @pytest.mark.asyncio
    async def test_audit_entry_has_auto_timestamp(self, db):
        before = time.time()
        await log_audit("user1", "agent1", "svc", "act", "ok")
        after = time.time()
        entries = await get_audit_log("user1")
        assert before <= entries[0].timestamp <= after

    @pytest.mark.asyncio
    async def test_empty_audit_log(self, db):
        entries = await get_audit_log("user_no_logs")
        assert entries == []


# ── AuditEntry Dataclass ─────────────────────────────────────────────


class TestAuditEntryDataclass:
    """Tests for AuditEntry dataclass."""

    def test_default_values(self):
        entry = AuditEntry()
        assert entry.id == 0
        assert entry.timestamp == 0.0
        assert entry.user_id == ""
        assert entry.agent_id == ""
        assert entry.service == ""
        assert entry.scopes == ""
        assert entry.action == ""
        assert entry.status == ""
        assert entry.ip_address == ""
        assert entry.details == ""

    def test_custom_values(self):
        entry = AuditEntry(
            id=42, timestamp=1000.0, user_id="u1", agent_id="a1",
            service="github", scopes="read", action="token_request",
            status="granted", ip_address="10.0.0.1", details="OK",
        )
        assert entry.id == 42
        assert entry.service == "github"
        assert entry.ip_address == "10.0.0.1"


# ── ApiKey Dataclass ─────────────────────────────────────────────────


class TestApiKeyDataclass:
    """Tests for ApiKey dataclass."""

    def test_default_values(self):
        key = ApiKey()
        assert key.id == ""
        assert key.key_hash == ""
        assert key.key_prefix == ""
        assert key.user_id == ""
        assert key.agent_id == ""
        assert key.name == ""
        assert key.created_at == 0.0
        assert key.expires_at == 0.0
        assert key.is_revoked is False
        assert key.last_used_at == 0.0

    def test_custom_values(self):
        key = ApiKey(
            id="k1", key_hash="abc", key_prefix="ag_12345",
            user_id="u1", agent_id="a1", name="Test Key",
            created_at=1000.0, expires_at=2000.0,
            is_revoked=True, last_used_at=1500.0,
        )
        assert key.id == "k1"
        assert key.is_revoked is True
        assert key.last_used_at == 1500.0
