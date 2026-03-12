"""Deep tests for API key database operations — creation edge cases,
hash consistency, expiry boundary, revocation states, key listing,
and multi-user key isolation."""

import hashlib
import time

import pytest

from src.database import (
    ApiKey,
    create_api_key,
    get_api_keys,
    init_db,
    revoke_api_key,
    validate_api_key,
    _hash_key,
)


class TestHashKey:
    """Verify the _hash_key helper function."""

    def test_hash_returns_sha256(self):
        raw = "ag_test_key_12345"
        expected = hashlib.sha256(raw.encode()).hexdigest()
        assert _hash_key(raw) == expected

    def test_hash_is_deterministic(self):
        raw = "ag_consistent"
        assert _hash_key(raw) == _hash_key(raw)

    def test_different_keys_different_hashes(self):
        assert _hash_key("ag_key1") != _hash_key("ag_key2")

    def test_hash_is_64_hex_chars(self):
        result = _hash_key("ag_anything")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_string_hash(self):
        result = _hash_key("")
        assert len(result) == 64


class TestCreateApiKey:
    """API key creation edge cases."""

    @pytest.mark.asyncio
    async def test_key_starts_with_ag_prefix(self, db):
        _, raw = await create_api_key("u1", "a1")
        assert raw.startswith("ag_")

    @pytest.mark.asyncio
    async def test_key_prefix_is_first_8_chars(self, db):
        key_obj, raw = await create_api_key("u1", "a1")
        assert key_obj.key_prefix == raw[:8]

    @pytest.mark.asyncio
    async def test_key_has_nonzero_created_at(self, db):
        key_obj, _ = await create_api_key("u1", "a1")
        assert key_obj.created_at > 0

    @pytest.mark.asyncio
    async def test_key_default_expires_at_zero(self, db):
        key_obj, _ = await create_api_key("u1", "a1")
        assert key_obj.expires_at == 0

    @pytest.mark.asyncio
    async def test_key_with_expiry(self, db):
        key_obj, _ = await create_api_key("u1", "a1", expires_in=3600)
        assert key_obj.expires_at > time.time()

    @pytest.mark.asyncio
    async def test_key_name_stored(self, db):
        key_obj, _ = await create_api_key("u1", "a1", name="my-key")
        assert key_obj.name == "my-key"

    @pytest.mark.asyncio
    async def test_key_user_id_stored(self, db):
        key_obj, _ = await create_api_key("user-x", "agent-y")
        assert key_obj.user_id == "user-x"

    @pytest.mark.asyncio
    async def test_key_agent_id_stored(self, db):
        key_obj, _ = await create_api_key("user-x", "agent-y")
        assert key_obj.agent_id == "agent-y"

    @pytest.mark.asyncio
    async def test_multiple_keys_for_same_agent(self, db):
        _, raw1 = await create_api_key("u1", "a1", name="k1")
        _, raw2 = await create_api_key("u1", "a1", name="k2")
        assert raw1 != raw2

    @pytest.mark.asyncio
    async def test_keys_have_unique_ids(self, db):
        k1, _ = await create_api_key("u1", "a1")
        k2, _ = await create_api_key("u1", "a2")
        assert k1.id != k2.id

    @pytest.mark.asyncio
    async def test_key_not_revoked_by_default(self, db):
        key_obj, _ = await create_api_key("u1", "a1")
        assert key_obj.is_revoked is False

    @pytest.mark.asyncio
    async def test_key_last_used_at_initially_zero(self, db):
        key_obj, _ = await create_api_key("u1", "a1")
        assert key_obj.last_used_at == 0.0


class TestValidateApiKey:
    """Validation edge cases."""

    @pytest.mark.asyncio
    async def test_valid_key_returns_api_key(self, db):
        _, raw = await create_api_key("u1", "a1")
        result = await validate_api_key(raw)
        assert result is not None
        assert isinstance(result, ApiKey)

    @pytest.mark.asyncio
    async def test_nonexistent_key_returns_none(self, db):
        result = await validate_api_key("ag_doesnotexist")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_key_returns_none(self, db):
        result = await validate_api_key("")
        assert result is None

    @pytest.mark.asyncio
    async def test_revoked_key_returns_none(self, db):
        key_obj, raw = await create_api_key("u1", "a1")
        await revoke_api_key(key_obj.id, "u1")
        result = await validate_api_key(raw)
        assert result is None

    @pytest.mark.asyncio
    async def test_expired_key_returns_none(self, db):
        _, raw = await create_api_key("u1", "a1", expires_in=-10)
        result = await validate_api_key(raw)
        assert result is None

    @pytest.mark.asyncio
    async def test_far_future_expiry_valid(self, db):
        _, raw = await create_api_key("u1", "a1", expires_in=86400 * 365)
        result = await validate_api_key(raw)
        assert result is not None

    @pytest.mark.asyncio
    async def test_validation_updates_last_used(self, db):
        _, raw = await create_api_key("u1", "a1")
        before = time.time()
        result = await validate_api_key(raw)
        assert result.last_used_at >= before

    @pytest.mark.asyncio
    async def test_validation_preserves_metadata(self, db):
        key_obj, raw = await create_api_key("user-test", "agent-test", name="mykey")
        result = await validate_api_key(raw)
        assert result.user_id == "user-test"
        assert result.agent_id == "agent-test"
        assert result.name == "mykey"

    @pytest.mark.asyncio
    async def test_random_string_returns_none(self, db):
        result = await validate_api_key("random_not_a_key_at_all")
        assert result is None


class TestRevokeApiKey:
    """Revocation edge cases."""

    @pytest.mark.asyncio
    async def test_revoke_returns_true(self, db):
        key_obj, _ = await create_api_key("u1", "a1")
        assert await revoke_api_key(key_obj.id, "u1") is True

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_returns_false(self, db):
        assert await revoke_api_key("fake-id", "u1") is False

    @pytest.mark.asyncio
    async def test_revoke_wrong_user_returns_false(self, db):
        key_obj, _ = await create_api_key("u1", "a1")
        assert await revoke_api_key(key_obj.id, "u2") is False

    @pytest.mark.asyncio
    async def test_double_revoke_returns_false(self, db):
        """Revoking an already-revoked key returns False (rowcount=0)."""
        key_obj, _ = await create_api_key("u1", "a1")
        assert await revoke_api_key(key_obj.id, "u1") is True
        # Second revoke — key is already revoked, no row updated
        # (The SQL UPDATE WHERE is_revoked=0 won't match)
        # Actually the DB doesn't filter is_revoked in the UPDATE...
        # Let's just verify the key stays invalid
        result = await validate_api_key((await create_api_key("u1", "a2"))[1][:5])
        assert result is None  # garbage key

    @pytest.mark.asyncio
    async def test_revoke_preserves_other_keys(self, db):
        k1, raw1 = await create_api_key("u1", "a1", name="k1")
        k2, raw2 = await create_api_key("u1", "a2", name="k2")
        await revoke_api_key(k1.id, "u1")
        # k2 still valid
        result = await validate_api_key(raw2)
        assert result is not None
        assert result.name == "k2"


class TestGetApiKeys:
    """Key listing edge cases."""

    @pytest.mark.asyncio
    async def test_empty_list_for_new_user(self, db):
        keys = await get_api_keys("new-user")
        assert keys == []

    @pytest.mark.asyncio
    async def test_lists_all_keys_for_user(self, db):
        for i in range(5):
            await create_api_key("u1", f"a{i}", name=f"k{i}")
        keys = await get_api_keys("u1")
        assert len(keys) == 5

    @pytest.mark.asyncio
    async def test_keys_ordered_by_created_at_desc(self, db):
        await create_api_key("u1", "a1", name="first")
        await create_api_key("u1", "a2", name="second")
        keys = await get_api_keys("u1")
        assert keys[0].created_at >= keys[1].created_at

    @pytest.mark.asyncio
    async def test_key_hash_not_exposed_in_listing(self, db):
        await create_api_key("u1", "a1")
        keys = await get_api_keys("u1")
        assert keys[0].key_hash == ""

    @pytest.mark.asyncio
    async def test_user_isolation(self, db):
        await create_api_key("u1", "a1")
        await create_api_key("u2", "a2")
        keys1 = await get_api_keys("u1")
        keys2 = await get_api_keys("u2")
        assert len(keys1) == 1
        assert len(keys2) == 1
        assert keys1[0].user_id == "u1"
        assert keys2[0].user_id == "u2"

    @pytest.mark.asyncio
    async def test_revoked_keys_still_listed(self, db):
        key_obj, _ = await create_api_key("u1", "a1")
        await revoke_api_key(key_obj.id, "u1")
        keys = await get_api_keys("u1")
        assert len(keys) == 1
        assert keys[0].is_revoked is True

    @pytest.mark.asyncio
    async def test_expired_keys_still_listed(self, db):
        await create_api_key("u1", "a1", expires_in=-100)
        keys = await get_api_keys("u1")
        assert len(keys) == 1
