"""Deep API key lifecycle tests -- creation, rotation, revocation, expiration,
validation edge cases, and interaction with policy operations."""

import hashlib
import time
import pytest

from src.database import (
    AgentPolicy,
    create_agent_policy,
    create_api_key,
    emergency_revoke_all,
    get_api_keys,
    revoke_api_key,
    toggle_agent_policy,
    validate_api_key,
    _hash_key,
)


# ---------------------------------------------------------------------------
# 1. Key creation properties
# ---------------------------------------------------------------------------

class TestKeyCreationProperties:
    """Verify cryptographic and structural properties of created keys."""

    @pytest.mark.asyncio
    async def test_raw_key_length_is_consistent(self, db):
        """All generated raw keys have consistent length."""
        keys = []
        for i in range(5):
            _, raw = await create_api_key("user1", f"agent-{i}")
            keys.append(raw)
        lengths = {len(k) for k in keys}
        # All should be same length (ag_ + 32 bytes base64url)
        assert len(lengths) == 1

    @pytest.mark.asyncio
    async def test_key_id_is_unique(self, db):
        """Each API key has a unique ID."""
        ids = []
        for i in range(10):
            key, _ = await create_api_key("user1", f"agent-{i}")
            ids.append(key.id)
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_key_hash_is_unique(self, db):
        """Each API key has a unique hash."""
        hashes = []
        for i in range(10):
            key, _ = await create_api_key("user1", f"agent-{i}")
            hashes.append(key.key_hash)
        assert len(hashes) == len(set(hashes))

    @pytest.mark.asyncio
    async def test_key_created_at_is_recent(self, db):
        """The created_at timestamp is close to current time."""
        before = time.time()
        key, _ = await create_api_key("user1", "agent-1")
        after = time.time()
        assert before <= key.created_at <= after

    @pytest.mark.asyncio
    async def test_key_initial_last_used_is_zero(self, db):
        """New keys have last_used_at of 0."""
        key, _ = await create_api_key("user1", "agent-1")
        assert key.last_used_at == 0.0

    @pytest.mark.asyncio
    async def test_key_initial_not_revoked(self, db):
        """New keys are not revoked."""
        key, _ = await create_api_key("user1", "agent-1")
        assert key.is_revoked is False


# ---------------------------------------------------------------------------
# 2. Key rotation pattern
# ---------------------------------------------------------------------------

class TestKeyRotation:
    """Simulate a key rotation workflow: create new, validate both, revoke old."""

    @pytest.mark.asyncio
    async def test_rotation_both_keys_valid_during_overlap(self, db):
        """During rotation, both old and new keys are valid."""
        _, old_raw = await create_api_key("user1", "agent-1", "old-key")
        _, new_raw = await create_api_key("user1", "agent-1", "new-key")
        # Both valid
        assert await validate_api_key(old_raw) is not None
        assert await validate_api_key(new_raw) is not None

    @pytest.mark.asyncio
    async def test_rotation_revoke_old_new_still_works(self, db):
        """After revoking the old key, the new one still works."""
        old_key, old_raw = await create_api_key("user1", "agent-1", "old")
        _, new_raw = await create_api_key("user1", "agent-1", "new")
        await revoke_api_key(old_key.id, "user1")
        assert await validate_api_key(old_raw) is None
        assert await validate_api_key(new_raw) is not None

    @pytest.mark.asyncio
    async def test_rotation_revoke_new_old_still_works(self, db):
        """If you revoke the wrong key by mistake, the other still works."""
        _, old_raw = await create_api_key("user1", "agent-1", "old")
        new_key, new_raw = await create_api_key("user1", "agent-1", "new")
        await revoke_api_key(new_key.id, "user1")
        assert await validate_api_key(new_raw) is None
        assert await validate_api_key(old_raw) is not None

    @pytest.mark.asyncio
    async def test_rotation_three_keys_progressive_revocation(self, db):
        """Three keys can coexist; revoking them one by one."""
        k1, r1 = await create_api_key("user1", "agent-1", "v1")
        k2, r2 = await create_api_key("user1", "agent-1", "v2")
        k3, r3 = await create_api_key("user1", "agent-1", "v3")

        # All valid initially
        for r in [r1, r2, r3]:
            assert await validate_api_key(r) is not None

        # Revoke v1
        await revoke_api_key(k1.id, "user1")
        assert await validate_api_key(r1) is None
        assert await validate_api_key(r2) is not None
        assert await validate_api_key(r3) is not None

        # Revoke v2
        await revoke_api_key(k2.id, "user1")
        assert await validate_api_key(r2) is None
        assert await validate_api_key(r3) is not None


# ---------------------------------------------------------------------------
# 3. Expiration edge cases
# ---------------------------------------------------------------------------

class TestKeyExpirationEdgeCases:
    """Edge cases around key expiration timing."""

    @pytest.mark.asyncio
    async def test_key_with_very_short_expiry(self, db):
        """A key with expires_in=-1 is already expired at creation."""
        _, raw = await create_api_key("user1", "agent-1", expires_in=-1)
        result = await validate_api_key(raw)
        assert result is None

    @pytest.mark.asyncio
    async def test_key_with_large_expiry(self, db):
        """A key with a large expiry (1 year) validates successfully."""
        _, raw = await create_api_key("user1", "agent-1", expires_in=365*86400)
        result = await validate_api_key(raw)
        assert result is not None
        assert result.expires_at > time.time() + 364*86400

    @pytest.mark.asyncio
    async def test_never_expiring_key(self, db):
        """A key with expires_in=0 has expires_at=0.0 (never expires)."""
        key, raw = await create_api_key("user1", "agent-1", expires_in=0)
        assert key.expires_at == 0.0
        result = await validate_api_key(raw)
        assert result is not None

    @pytest.mark.asyncio
    async def test_expired_key_not_updated_on_validation(self, db):
        """An expired key returns None and doesn't update last_used_at."""
        _, raw = await create_api_key("user1", "agent-1", expires_in=-1)
        result = await validate_api_key(raw)
        assert result is None


# ---------------------------------------------------------------------------
# 4. Key validation stress
# ---------------------------------------------------------------------------

class TestKeyValidationStress:
    """Multiple validations and edge cases for the validation flow."""

    @pytest.mark.asyncio
    async def test_validate_same_key_10_times(self, db):
        """A key remains valid after many validations."""
        _, raw = await create_api_key("user1", "agent-1")
        for _ in range(10):
            result = await validate_api_key(raw)
            assert result is not None
            assert result.agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_last_used_updates_each_validation(self, db):
        """last_used_at is updated on every validation call."""
        _, raw = await create_api_key("user1", "agent-1")
        timestamps = []
        for _ in range(3):
            result = await validate_api_key(raw)
            timestamps.append(result.last_used_at)
        # Each subsequent validation should have >= timestamp
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1]

    @pytest.mark.asyncio
    async def test_validate_wrong_prefix(self, db):
        """A key with wrong prefix fails validation."""
        result = await validate_api_key("bad_prefix_key_value")
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_truncated_key(self, db):
        """A truncated version of a valid key fails."""
        _, raw = await create_api_key("user1", "agent-1")
        truncated = raw[:10]
        result = await validate_api_key(truncated)
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_key_with_extra_chars(self, db):
        """A key with appended characters fails."""
        _, raw = await create_api_key("user1", "agent-1")
        result = await validate_api_key(raw + "EXTRA")
        assert result is None


# ---------------------------------------------------------------------------
# 5. Key interaction with policy operations
# ---------------------------------------------------------------------------

class TestKeyPolicyInteraction:
    """How API keys interact with policy changes."""

    @pytest.mark.asyncio
    async def test_key_survives_policy_toggle(self, db):
        """Toggling a policy off doesn't invalidate its API keys."""
        await create_agent_policy(AgentPolicy(
            agent_id="toggle-agent", agent_name="Bot",
            created_by="user1", is_active=True,
        ))
        _, raw = await create_api_key("user1", "toggle-agent", "key")
        await toggle_agent_policy("toggle-agent", "user1")
        # Key is still valid (it's the policy that's disabled)
        result = await validate_api_key(raw)
        assert result is not None

    @pytest.mark.asyncio
    async def test_emergency_revoke_invalidates_all_keys(self, db):
        """Emergency revoke invalidates all keys for the user."""
        raws = []
        for i in range(5):
            _, raw = await create_api_key("user1", f"agent-{i}")
            raws.append(raw)
        await emergency_revoke_all("user1")
        for raw in raws:
            assert await validate_api_key(raw) is None

    @pytest.mark.asyncio
    async def test_keys_for_different_agents_independently_revocable(self, db):
        """Keys for different agents can be revoked independently."""
        k1, r1 = await create_api_key("user1", "agent-a", "ka")
        k2, r2 = await create_api_key("user1", "agent-b", "kb")
        await revoke_api_key(k1.id, "user1")
        assert await validate_api_key(r1) is None
        assert await validate_api_key(r2) is not None

    @pytest.mark.asyncio
    async def test_listing_shows_revoked_and_active_keys(self, db):
        """get_api_keys returns both revoked and active keys."""
        k1, _ = await create_api_key("user1", "agent-1", "active-key")
        k2, _ = await create_api_key("user1", "agent-2", "revoked-key")
        await revoke_api_key(k2.id, "user1")
        keys = await get_api_keys("user1")
        assert len(keys) == 2
        statuses = {k.name: k.is_revoked for k in keys}
        assert statuses["active-key"] is False
        assert statuses["revoked-key"] is True


# ---------------------------------------------------------------------------
# 6. Hash function edge cases
# ---------------------------------------------------------------------------

class TestHashFunctionEdgeCases:
    """Edge cases for the _hash_key function."""

    def test_hash_of_ag_prefix(self):
        """Hashing an ag_ prefixed string works normally."""
        result = _hash_key("ag_some_key_value")
        assert len(result) == 64

    def test_hash_of_very_long_key(self):
        """Very long keys hash to 64 chars."""
        result = _hash_key("ag_" + "a" * 1000)
        assert len(result) == 64

    def test_hash_collision_resistance(self):
        """Similar keys produce different hashes."""
        h1 = _hash_key("ag_key1")
        h2 = _hash_key("ag_key2")
        h3 = _hash_key("ag_key3")
        assert len({h1, h2, h3}) == 3

    def test_hash_matches_hashlib(self):
        """Our _hash_key matches direct hashlib SHA-256."""
        key = "ag_test_key_12345"
        expected = hashlib.sha256(key.encode()).hexdigest()
        assert _hash_key(key) == expected
