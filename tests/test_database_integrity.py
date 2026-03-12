"""Database integrity tests — hash function, concurrent operations,
large data volumes, special characters, and cross-table interactions."""

import hashlib
import time
import pytest

from src.database import (
    AgentPolicy,
    ApiKey,
    AuditEntry,
    _hash_key,
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


# ---------------------------------------------------------------------------
# 1. _hash_key function tests
# ---------------------------------------------------------------------------

class TestHashKey:
    """Tests for the SHA-256 key hashing function."""

    def test_hash_is_sha256_hex(self):
        result = _hash_key("test_key")
        expected = hashlib.sha256("test_key".encode()).hexdigest()
        assert result == expected

    def test_hash_length_is_64(self):
        result = _hash_key("anything")
        assert len(result) == 64

    def test_hash_is_deterministic(self):
        assert _hash_key("same") == _hash_key("same")

    def test_different_inputs_different_hashes(self):
        assert _hash_key("key1") != _hash_key("key2")

    def test_empty_string_hashes(self):
        result = _hash_key("")
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_unicode_key_hashes(self):
        result = _hash_key("unicode_\u00e9\u00e8\u00ea")
        assert len(result) == 64

    def test_long_key_hashes(self):
        long_key = "a" * 10000
        result = _hash_key(long_key)
        assert len(result) == 64

    def test_special_characters(self):
        result = _hash_key("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        assert len(result) == 64


# ---------------------------------------------------------------------------
# 2. Large volume operations
# ---------------------------------------------------------------------------

class TestLargeVolumeOperations:
    """Test database operations with larger datasets."""

    @pytest.mark.asyncio
    async def test_many_audit_entries(self, db):
        """Writing and reading many audit entries."""
        for i in range(100):
            await log_audit("user1", f"agent-{i}", "svc", f"action-{i}", "success")
        entries = await get_audit_log("user1", limit=100)
        assert len(entries) == 100

    @pytest.mark.asyncio
    async def test_many_policies_for_user(self, db):
        """A user can have many policies."""
        for i in range(50):
            await create_agent_policy(AgentPolicy(
                agent_id=f"agent-{i}",
                agent_name=f"Bot {i}",
                allowed_services=["github"],
                created_by="user1",
                created_at=time.time(),
            ))
        policies = await get_all_policies("user1")
        assert len(policies) == 50

    @pytest.mark.asyncio
    async def test_many_api_keys_for_user(self, db):
        """A user can have many API keys."""
        for i in range(30):
            await create_api_key("user1", f"agent-{i}", f"key-{i}")
        keys = await get_api_keys("user1")
        assert len(keys) == 30

    @pytest.mark.asyncio
    async def test_many_connected_services(self, db):
        """A user can have multiple connected services."""
        services = ["svc1", "svc2", "svc3", "svc4", "svc5"]
        for svc in services:
            await add_connected_service("user1", svc, f"conn-{svc}")
        result = await get_connected_services("user1")
        assert len(result) == 5


# ---------------------------------------------------------------------------
# 3. Special characters in data
# ---------------------------------------------------------------------------

class TestSpecialCharacters:
    """Test handling of special characters in database fields."""

    @pytest.mark.asyncio
    async def test_agent_name_with_special_chars(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="special-agent",
            agent_name="Bot's \"Special\" Agent <test> & more",
            allowed_services=["github"],
            created_by="user1",
            created_at=time.time(),
        ))
        p = await get_agent_policy("special-agent")
        assert p.agent_name == "Bot's \"Special\" Agent <test> & more"

    @pytest.mark.asyncio
    async def test_agent_id_with_hyphens_and_underscores(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="my-special_agent-v2.1",
            agent_name="Bot",
            created_by="user1",
        ))
        p = await get_agent_policy("my-special_agent-v2.1")
        assert p is not None

    @pytest.mark.asyncio
    async def test_audit_details_with_unicode(self, db):
        await log_audit("user1", "agent-1", "svc", "action", "ok",
                       details="Unicode test: \u00e9\u00e8\u00ea \u4e16\u754c")
        entries = await get_audit_log("user1")
        assert "\u00e9" in entries[0].details

    @pytest.mark.asyncio
    async def test_api_key_name_with_special_chars(self, db):
        _, raw = await create_api_key("user1", "agent-1", "key-with-special!@#")
        result = await validate_api_key(raw)
        assert result.name == "key-with-special!@#"

    @pytest.mark.asyncio
    async def test_connection_id_with_uuid_format(self, db):
        await add_connected_service("user1", "github", "550e8400-e29b-41d4-a716-446655440000")
        svcs = await get_connected_services("user1")
        assert svcs[0]["connection_id"] == "550e8400-e29b-41d4-a716-446655440000"


# ---------------------------------------------------------------------------
# 4. Cross-table interactions
# ---------------------------------------------------------------------------

class TestCrossTableInteractions:
    """Test interactions between different database tables."""

    @pytest.mark.asyncio
    async def test_emergency_revoke_affects_both_tables(self, db):
        """Emergency revoke disables policies and revokes keys."""
        await create_agent_policy(AgentPolicy(
            agent_id="cross-agent",
            agent_name="Bot",
            created_by="user1",
            is_active=True,
        ))
        _, raw = await create_api_key("user1", "cross-agent", "key")

        result = await emergency_revoke_all("user1")
        assert result["policies_disabled"] >= 1
        assert result["keys_revoked"] >= 1

        # Verify policy disabled
        p = await get_agent_policy("cross-agent")
        assert p.is_active is False

        # Verify key revoked
        assert await validate_api_key(raw) is None

    @pytest.mark.asyncio
    async def test_delete_policy_does_not_affect_api_keys(self, db):
        """Deleting a policy doesn't revoke its API keys (keys survive)."""
        await create_agent_policy(AgentPolicy(
            agent_id="del-key-agent",
            agent_name="Bot",
            created_by="user1",
        ))
        _, raw = await create_api_key("user1", "del-key-agent", "key")

        await delete_agent_policy("del-key-agent", "user1")

        # Key still validates (it's bound to agent_id, not policy existence)
        result = await validate_api_key(raw)
        assert result is not None
        assert result.agent_id == "del-key-agent"

    @pytest.mark.asyncio
    async def test_toggle_policy_does_not_affect_api_keys(self, db):
        """Toggling a policy off doesn't revoke its API keys."""
        await create_agent_policy(AgentPolicy(
            agent_id="tog-key-agent",
            agent_name="Bot",
            created_by="user1",
            is_active=True,
        ))
        _, raw = await create_api_key("user1", "tog-key-agent", "key")

        await toggle_agent_policy("tog-key-agent", "user1")

        result = await validate_api_key(raw)
        assert result is not None

    @pytest.mark.asyncio
    async def test_connected_service_not_affected_by_policy_delete(self, db):
        """Deleting a policy doesn't remove connected services."""
        await create_agent_policy(AgentPolicy(
            agent_id="svc-pol",
            agent_name="Bot",
            created_by="user1",
        ))
        await add_connected_service("user1", "github", "conn-1")
        await delete_agent_policy("svc-pol", "user1")

        svcs = await get_connected_services("user1")
        assert len(svcs) == 1

    @pytest.mark.asyncio
    async def test_emergency_revoke_does_not_affect_connected_services(self, db):
        """Emergency revoke doesn't disconnect services."""
        await add_connected_service("user1", "github", "conn-1")
        await create_agent_policy(AgentPolicy(
            agent_id="em-svc",
            agent_name="Bot",
            created_by="user1",
        ))

        await emergency_revoke_all("user1")

        svcs = await get_connected_services("user1")
        assert len(svcs) == 1


# ---------------------------------------------------------------------------
# 5. Multiple init_db calls
# ---------------------------------------------------------------------------

class TestDatabaseRobustness:
    """Test database robustness scenarios."""

    @pytest.mark.asyncio
    async def test_triple_init_is_safe(self, db):
        await init_db()
        await init_db()
        await init_db()

    @pytest.mark.asyncio
    async def test_data_persists_across_reinit(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="persist",
            agent_name="Persistent Bot",
            created_by="user1",
        ))
        await init_db()
        p = await get_agent_policy("persist")
        assert p is not None
        assert p.agent_name == "Persistent Bot"

    @pytest.mark.asyncio
    async def test_audit_log_default_limit(self, db):
        """Default limit is 50."""
        for i in range(60):
            await log_audit("user1", "a", "s", f"act{i}", "ok")
        entries = await get_audit_log("user1")
        assert len(entries) == 50

    @pytest.mark.asyncio
    async def test_api_key_ordering(self, db):
        """API keys are returned ordered by created_at DESC."""
        for i in range(5):
            await create_api_key("user1", f"agent-{i}", f"key-{i}")
        keys = await get_api_keys("user1")
        for a, b in zip(keys, keys[1:]):
            assert a.created_at >= b.created_at


# ---------------------------------------------------------------------------
# 6. Edge cases for validate_api_key
# ---------------------------------------------------------------------------

class TestValidateApiKeyEdgeCases:
    """Additional edge cases for API key validation."""

    @pytest.mark.asyncio
    async def test_validate_key_with_whitespace(self, db):
        """Key with leading/trailing whitespace fails."""
        _, raw = await create_api_key("user1", "agent-1", "key")
        # Whitespace-padded key should fail
        result = await validate_api_key(f" {raw} ")
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_key_partial_match(self, db):
        """Partial key string should not match."""
        _, raw = await create_api_key("user1", "agent-1", "key")
        # Use first half only
        partial = raw[:len(raw)//2]
        result = await validate_api_key(partial)
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_key_after_multiple_uses(self, db):
        """Key remains valid after multiple uses."""
        _, raw = await create_api_key("user1", "agent-1", "key")
        for _ in range(5):
            result = await validate_api_key(raw)
            assert result is not None
            assert result.agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_validate_key_case_sensitive(self, db):
        """Key validation is case-sensitive."""
        _, raw = await create_api_key("user1", "agent-1", "key")
        result_upper = await validate_api_key(raw.upper())
        result_lower = await validate_api_key(raw.lower())
        # The original key should work, but upper/lower may not
        result_original = await validate_api_key(raw)
        assert result_original is not None
        # upper and lower versions of the key should fail
        # (since SHA-256 is case-sensitive)
        if raw.upper() != raw:
            assert result_upper is None
        if raw.lower() != raw:
            assert result_lower is None
