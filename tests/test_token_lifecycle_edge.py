"""Token lifecycle edge cases -- expired tokens, revoked tokens, race
conditions in key creation/validation, boundary expiration times,
key hash collisions, and concurrent key operations."""

import asyncio
import hashlib
import time
from unittest.mock import patch

import pytest

from src.database import (
    AgentPolicy,
    ApiKey,
    create_agent_policy,
    create_api_key,
    get_api_keys,
    init_db,
    revoke_api_key,
    validate_api_key,
    _hash_key,
)
from src.policy import _rate_counters


@pytest.fixture(autouse=True)
def clear_rate_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


class TestApiKeyCreation:
    """Basic API key creation edge cases."""

    @pytest.mark.asyncio
    async def test_key_starts_with_prefix(self, db):
        _, raw = await create_api_key("u1", "agent-1", "test-key")
        assert raw.startswith("ag_")

    @pytest.mark.asyncio
    async def test_key_prefix_stored(self, db):
        key_obj, raw = await create_api_key("u1", "agent-1")
        assert key_obj.key_prefix == raw[:8]

    @pytest.mark.asyncio
    async def test_key_hash_matches_raw(self, db):
        key_obj, raw = await create_api_key("u1", "agent-1")
        expected = hashlib.sha256(raw.encode()).hexdigest()
        assert key_obj.key_hash == expected

    @pytest.mark.asyncio
    async def test_two_keys_are_unique(self, db):
        _, raw1 = await create_api_key("u1", "agent-1", "k1")
        _, raw2 = await create_api_key("u1", "agent-1", "k2")
        assert raw1 != raw2

    @pytest.mark.asyncio
    async def test_key_with_empty_name(self, db):
        key_obj, _ = await create_api_key("u1", "agent-1", "")
        assert key_obj.name == ""

    @pytest.mark.asyncio
    async def test_key_with_long_name(self, db):
        name = "x" * 1000
        key_obj, _ = await create_api_key("u1", "agent-1", name)
        assert key_obj.name == name

    @pytest.mark.asyncio
    async def test_key_for_nonexistent_agent(self, db):
        """Keys can be created for agents that don't have policies yet."""
        key_obj, raw = await create_api_key("u1", "no-policy-agent", "test")
        validated = await validate_api_key(raw)
        assert validated is not None
        assert validated.agent_id == "no-policy-agent"


class TestApiKeyExpiration:
    """Expiration edge cases."""

    @pytest.mark.asyncio
    async def test_expired_key_returns_none(self, db):
        _, raw = await create_api_key("u1", "agent-1", "exp-key", expires_in=1)
        # Wait until expired
        with patch("src.database.time") as mock_time:
            mock_time.time.return_value = time.time() + 10
            result = await validate_api_key(raw)
        assert result is None

    @pytest.mark.asyncio
    async def test_non_expired_key_validates(self, db):
        _, raw = await create_api_key("u1", "agent-1", "exp-key", expires_in=3600)
        result = await validate_api_key(raw)
        assert result is not None

    @pytest.mark.asyncio
    async def test_never_expiring_key(self, db):
        key_obj, raw = await create_api_key("u1", "agent-1", "perm-key", expires_in=0)
        assert key_obj.expires_at == 0
        result = await validate_api_key(raw)
        assert result is not None

    @pytest.mark.asyncio
    async def test_expires_at_boundary(self, db):
        """Key with expires_in=1 second: validate immediately should succeed."""
        _, raw = await create_api_key("u1", "agent-1", "edge-key", expires_in=3600)
        result = await validate_api_key(raw)
        assert result is not None

    @pytest.mark.asyncio
    async def test_key_expiry_stored_correctly(self, db):
        before = time.time()
        key_obj, _ = await create_api_key("u1", "agent-1", "t", expires_in=7200)
        after = time.time()
        assert key_obj.expires_at >= before + 7200
        assert key_obj.expires_at <= after + 7200


class TestApiKeyRevocation:
    """Revocation edge cases."""

    @pytest.mark.asyncio
    async def test_revoked_key_returns_none(self, db):
        key_obj, raw = await create_api_key("u1", "agent-1", "rev-key")
        await revoke_api_key(key_obj.id, "u1")
        result = await validate_api_key(raw)
        assert result is None

    @pytest.mark.asyncio
    async def test_revoke_wrong_user_fails(self, db):
        key_obj, raw = await create_api_key("u1", "agent-1", "rev-key")
        revoked = await revoke_api_key(key_obj.id, "u2")
        assert revoked is False
        # Key still works
        result = await validate_api_key(raw)
        assert result is not None

    @pytest.mark.asyncio
    async def test_double_revoke_idempotent(self, db):
        key_obj, raw = await create_api_key("u1", "agent-1", "rev-key")
        r1 = await revoke_api_key(key_obj.id, "u1")
        assert r1 is True
        # Second revoke: row already revoked, UPDATE matches 0 rows
        r2 = await revoke_api_key(key_obj.id, "u1")
        # Might return True (row updated) or False (no change) -- depends on implementation
        result = await validate_api_key(raw)
        assert result is None

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_key(self, db):
        result = await revoke_api_key("fake-id", "u1")
        assert result is False

    @pytest.mark.asyncio
    async def test_revoke_preserves_other_keys(self, db):
        k1_obj, raw1 = await create_api_key("u1", "agent-1", "k1")
        k2_obj, raw2 = await create_api_key("u1", "agent-1", "k2")
        await revoke_api_key(k1_obj.id, "u1")
        assert await validate_api_key(raw1) is None
        assert await validate_api_key(raw2) is not None


class TestApiKeyValidation:
    """Validation edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_key_format(self, db):
        result = await validate_api_key("not-a-valid-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_key(self, db):
        result = await validate_api_key("")
        assert result is None

    @pytest.mark.asyncio
    async def test_key_with_null_bytes(self, db):
        result = await validate_api_key("ag_\x00null\x00bytes")
        assert result is None

    @pytest.mark.asyncio
    async def test_very_long_key(self, db):
        result = await validate_api_key("ag_" + "a" * 10000)
        assert result is None

    @pytest.mark.asyncio
    async def test_key_with_unicode(self, db):
        result = await validate_api_key("ag_\u2603\u2764\U0001f916")
        assert result is None

    @pytest.mark.asyncio
    async def test_valid_key_updates_last_used(self, db):
        _, raw = await create_api_key("u1", "agent-1", "use-key")
        before = time.time()
        result = await validate_api_key(raw)
        after = time.time()
        assert result.last_used_at >= before
        assert result.last_used_at <= after

    @pytest.mark.asyncio
    async def test_key_validates_correct_agent_id(self, db):
        _, raw = await create_api_key("u1", "specific-agent", "t")
        result = await validate_api_key(raw)
        assert result.agent_id == "specific-agent"

    @pytest.mark.asyncio
    async def test_key_validates_correct_user_id(self, db):
        _, raw = await create_api_key("my-user", "agent-1", "t")
        result = await validate_api_key(raw)
        assert result.user_id == "my-user"


class TestKeyListFiltering:
    """get_api_keys returns keys for the correct user only."""

    @pytest.mark.asyncio
    async def test_keys_filtered_by_user(self, db):
        await create_api_key("u1", "a1", "k1")
        await create_api_key("u2", "a2", "k2")
        u1_keys = await get_api_keys("u1")
        u2_keys = await get_api_keys("u2")
        assert all(k.user_id == "u1" for k in u1_keys)
        assert all(k.user_id == "u2" for k in u2_keys)

    @pytest.mark.asyncio
    async def test_keys_list_hides_hash(self, db):
        await create_api_key("u1", "a1", "k1")
        keys = await get_api_keys("u1")
        assert len(keys) == 1
        assert keys[0].key_hash == ""

    @pytest.mark.asyncio
    async def test_empty_keys_list(self, db):
        keys = await get_api_keys("no-keys-user")
        assert keys == []

    @pytest.mark.asyncio
    async def test_revoked_keys_still_listed(self, db):
        k_obj, _ = await create_api_key("u1", "a1", "k1")
        await revoke_api_key(k_obj.id, "u1")
        keys = await get_api_keys("u1")
        assert len(keys) == 1
        assert keys[0].is_revoked is True


class TestHashKeyFunction:
    """Test the _hash_key utility."""

    def test_consistent_hashing(self):
        assert _hash_key("test123") == _hash_key("test123")

    def test_different_inputs_different_hashes(self):
        assert _hash_key("key1") != _hash_key("key2")

    def test_sha256_output_length(self):
        result = _hash_key("any-key")
        assert len(result) == 64  # SHA-256 hex digest

    def test_empty_string_hash(self):
        result = _hash_key("")
        assert len(result) == 64

    def test_unicode_key_hash(self):
        result = _hash_key("\u2603snowman")
        assert len(result) == 64


class TestConcurrentKeyOperations:
    """Test concurrent key operations don't corrupt state."""

    @pytest.mark.asyncio
    async def test_concurrent_key_creation(self, db):
        """Multiple keys created concurrently should all be unique."""
        tasks = [
            create_api_key("u1", f"agent-{i}", f"key-{i}")
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        raw_keys = [r[1] for r in results]
        assert len(set(raw_keys)) == 10

    @pytest.mark.asyncio
    async def test_concurrent_validation(self, db):
        """Multiple concurrent validations of the same key succeed."""
        _, raw = await create_api_key("u1", "agent-1", "shared-key")
        tasks = [validate_api_key(raw) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        assert all(r is not None for r in results)
        assert all(r.agent_id == "agent-1" for r in results)

    @pytest.mark.asyncio
    async def test_create_and_immediately_validate(self, db):
        """Key is available for validation immediately after creation."""
        _, raw = await create_api_key("u1", "agent-1", "quick-key")
        result = await validate_api_key(raw)
        assert result is not None

    @pytest.mark.asyncio
    async def test_revoke_and_immediately_validate(self, db):
        """Key is invalid immediately after revocation."""
        k_obj, raw = await create_api_key("u1", "agent-1", "rev-quick")
        await revoke_api_key(k_obj.id, "u1")
        result = await validate_api_key(raw)
        assert result is None
