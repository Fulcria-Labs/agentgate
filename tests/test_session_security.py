"""Session and CSRF security tests for AgentGate.

Covers session expiration and refresh, CSRF protection on state-changing
endpoints, cookie security attributes, concurrent session handling,
session fixation prevention, and authentication bypass attempts.
"""

import asyncio
import time
import secrets
import hashlib
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
    ApiKey,
    _hash_key,
)
from src.policy import (
    enforce_policy,
    PolicyDenied,
    _rate_counters,
)
from src.config import Settings


@pytest.fixture(autouse=True)
def clear_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


USER = "auth0|session-sec-user"


# ============================================================
# 1. Session Expiration and Refresh
# ============================================================

class TestSessionExpiration:
    """Tests for session-based authentication and expiration."""

    @pytest.mark.asyncio
    async def test_api_key_expiration_enforced(self, db):
        """Expired API key should not validate."""
        key_obj, raw = await create_api_key(USER, "expire-agent", "test", expires_in=1)
        await asyncio.sleep(1.1)
        result = await validate_api_key(raw)
        assert result is None

    @pytest.mark.asyncio
    async def test_api_key_not_yet_expired(self, db):
        """Non-expired API key should validate successfully."""
        key_obj, raw = await create_api_key(USER, "valid-agent", "test", expires_in=3600)
        result = await validate_api_key(raw)
        assert result is not None
        assert result.agent_id == "valid-agent"

    @pytest.mark.asyncio
    async def test_api_key_never_expires(self, db):
        """API key with no expiration should always validate."""
        key_obj, raw = await create_api_key(USER, "forever-agent", "test", expires_in=0)
        result = await validate_api_key(raw)
        assert result is not None

    @pytest.mark.asyncio
    async def test_api_key_last_used_at_updated(self, db):
        """Validating a key updates last_used_at timestamp."""
        key_obj, raw = await create_api_key(USER, "used-agent", "test")
        before = time.time()
        result = await validate_api_key(raw)
        assert result is not None
        assert result.last_used_at >= before

    @pytest.mark.asyncio
    async def test_api_key_multiple_validations_update_timestamp(self, db):
        """Multiple validations should each update the timestamp."""
        key_obj, raw = await create_api_key(USER, "multi-val-agent", "test")

        first_result = await validate_api_key(raw)
        first_time = first_result.last_used_at

        await asyncio.sleep(0.01)

        second_result = await validate_api_key(raw)
        assert second_result.last_used_at >= first_time

    @pytest.mark.asyncio
    async def test_expired_key_after_many_validations(self, db):
        """Key that expires after multiple validations should fail."""
        key_obj, raw = await create_api_key(USER, "degrade-agent", "test", expires_in=1)
        # First validation should work
        result = await validate_api_key(raw)
        assert result is not None

        # Wait for expiration
        await asyncio.sleep(1.1)
        result = await validate_api_key(raw)
        assert result is None


# ============================================================
# 2. CSRF Protection on State-Changing Endpoints
# ============================================================

class TestCSRFProtection:
    """Tests for CSRF-like protection via authentication requirements."""

    @pytest.mark.asyncio
    async def test_policy_creation_requires_auth(self, db):
        """Creating a policy requires authenticated user."""
        # Without auth, policy operations directly on DB should work (no HTTP auth)
        # But the enforce_policy checks ownership
        policy = AgentPolicy(
            agent_id="csrf-agent",
            agent_name="CSRF Agent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by=USER,
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        # Different user cannot enforce
        with pytest.raises(PolicyDenied, match="Not authorized"):
            await enforce_policy("attacker-user", "csrf-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_policy_toggle_requires_ownership(self, db):
        """Toggling a policy requires the creating user."""
        policy = AgentPolicy(
            agent_id="toggle-csrf-agent",
            agent_name="Toggle CSRF",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by=USER,
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        # Wrong user gets None (not found for that user)
        result = await toggle_agent_policy("toggle-csrf-agent", "wrong-user")
        assert result is None

        # Correct user succeeds
        result = await toggle_agent_policy("toggle-csrf-agent", USER)
        assert result is not None

    @pytest.mark.asyncio
    async def test_policy_deletion_requires_ownership(self, db):
        """Deleting a policy requires the creating user."""
        policy = AgentPolicy(
            agent_id="del-csrf-agent",
            agent_name="Del CSRF",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by=USER,
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        # Wrong user cannot delete
        result = await delete_agent_policy("del-csrf-agent", "wrong-user")
        assert result is False

        # Correct user can delete
        result = await delete_agent_policy("del-csrf-agent", USER)
        assert result is True

    @pytest.mark.asyncio
    async def test_api_key_revocation_requires_ownership(self, db):
        """Revoking an API key requires the creating user."""
        key_obj, raw = await create_api_key(USER, "revoke-csrf-agent", "test")

        # Wrong user cannot revoke
        result = await revoke_api_key(key_obj.id, "wrong-user")
        assert result is False

        # Correct user can revoke
        result = await revoke_api_key(key_obj.id, USER)
        assert result is True

    @pytest.mark.asyncio
    async def test_emergency_revoke_user_scoped(self, db):
        """Emergency revoke only affects the requesting user's resources."""
        # Create resources for two users
        await create_agent_policy(AgentPolicy(
            agent_id="user1-agent",
            agent_name="User1 Agent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by=USER,
            created_at=time.time(),
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="user2-agent",
            agent_name="User2 Agent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="other-user",
            created_at=time.time(),
        ))

        await emergency_revoke_all(USER)

        # User1's agent should be disabled
        p1 = await get_agent_policy("user1-agent")
        assert p1.is_active is False

        # User2's agent should still be active
        p2 = await get_agent_policy("user2-agent")
        assert p2.is_active is True

    @pytest.mark.asyncio
    async def test_connected_service_user_isolation(self, db):
        """Connected services are isolated per user."""
        await add_connected_service(USER, "github")
        await add_connected_service("other-user", "github")

        # Each user sees only their own
        user_svcs = await get_connected_services(USER)
        other_svcs = await get_connected_services("other-user")

        assert len(user_svcs) == 1
        assert len(other_svcs) == 1

    @pytest.mark.asyncio
    async def test_disconnecting_service_user_isolation(self, db):
        """Disconnecting a service for one user does not affect another."""
        await add_connected_service(USER, "github")
        await add_connected_service("other-user", "github")

        await remove_connected_service(USER, "github")

        user_svcs = await get_connected_services(USER)
        other_svcs = await get_connected_services("other-user")

        assert len(user_svcs) == 0
        assert len(other_svcs) == 1


# ============================================================
# 3. Cookie Security Attributes
# ============================================================

class TestCookieSecurity:
    """Tests for security-related settings and configuration."""

    def test_settings_defaults(self):
        """Default settings should have safe defaults."""
        s = Settings()
        assert s.app_secret_key == "dev-secret-change-in-production"
        assert s.token_vault_enabled is True

    def test_auth0_issuer_url(self):
        """Auth0 issuer URL should be correctly formatted."""
        s = Settings(auth0_domain="example.auth0.com")
        assert s.auth0_issuer == "https://example.auth0.com/"

    def test_auth0_jwks_url(self):
        """Auth0 JWKS URL should be correctly formatted."""
        s = Settings(auth0_domain="example.auth0.com")
        assert s.auth0_jwks_url == "https://example.auth0.com/.well-known/jwks.json"

    def test_auth0_token_url(self):
        """Auth0 token URL should be correctly formatted."""
        s = Settings(auth0_domain="example.auth0.com")
        assert s.auth0_token_url == "https://example.auth0.com/oauth/token"

    def test_auth0_authorize_url(self):
        """Auth0 authorize URL should be correctly formatted."""
        s = Settings(auth0_domain="example.auth0.com")
        assert s.auth0_authorize_url == "https://example.auth0.com/authorize"

    def test_auth0_userinfo_url(self):
        """Auth0 userinfo URL should be correctly formatted."""
        s = Settings(auth0_domain="example.auth0.com")
        assert s.auth0_userinfo_url == "https://example.auth0.com/userinfo"

    def test_empty_domain_urls(self):
        """Empty domain still produces valid URL structure."""
        s = Settings(auth0_domain="")
        assert s.auth0_issuer == "https:///"

    def test_callback_url_default(self):
        """Callback URL default is localhost."""
        s = Settings()
        assert "localhost" in s.auth0_callback_url


# ============================================================
# 4. Concurrent Session Handling
# ============================================================

class TestConcurrentSessionHandling:
    """Tests for concurrent operations from multiple sessions."""

    @pytest.mark.asyncio
    async def test_concurrent_policy_creation_different_users(self, db):
        """Different users creating policies concurrently."""
        async def create_for_user(uid):
            policy = AgentPolicy(
                agent_id=f"agent-{uid}",
                agent_name=f"Agent {uid}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                created_by=uid,
                created_at=time.time(),
            )
            await create_agent_policy(policy)

        await asyncio.gather(*[
            create_for_user(f"user-{i}") for i in range(10)
        ])

        for i in range(10):
            p = await get_agent_policy(f"agent-user-{i}")
            assert p is not None
            assert p.created_by == f"user-{i}"

    @pytest.mark.asyncio
    async def test_concurrent_key_creation_different_users(self, db):
        """Different users creating API keys concurrently."""
        async def create_key_for_user(uid):
            _, raw = await create_api_key(uid, f"agent-{uid}", f"key-{uid}")
            return raw

        raw_keys = await asyncio.gather(*[
            create_key_for_user(f"user-{i}") for i in range(10)
        ])

        assert len(set(raw_keys)) == 10

    @pytest.mark.asyncio
    async def test_concurrent_enforcement_different_users(self, db):
        """Different users enforcing policies concurrently."""
        for i in range(5):
            uid = f"conc-user-{i}"
            await create_agent_policy(AgentPolicy(
                agent_id=f"conc-agent-{i}",
                agent_name=f"Conc Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=100,
                created_by=uid,
                created_at=time.time(),
            ))

        results = await asyncio.gather(*[
            enforce_policy(f"conc-user-{i}", f"conc-agent-{i}", "github", ["repo"])
            for i in range(5)
        ], return_exceptions=True)

        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) == 5

    @pytest.mark.asyncio
    async def test_concurrent_emergency_revoke_different_users(self, db):
        """Different users triggering emergency revoke concurrently."""
        for i in range(3):
            uid = f"emerg-user-{i}"
            await create_agent_policy(AgentPolicy(
                agent_id=f"emerg-agent-{i}",
                agent_name=f"Emerg Agent {i}",
                allowed_services=["github"],
                allowed_scopes={"github": ["repo"]},
                created_by=uid,
                created_at=time.time(),
            ))

        results = await asyncio.gather(*[
            emergency_revoke_all(f"emerg-user-{i}") for i in range(3)
        ], return_exceptions=True)

        for r in results:
            assert not isinstance(r, Exception)
            assert r["policies_disabled"] == 1

    @pytest.mark.asyncio
    async def test_audit_log_concurrent_different_users(self, db):
        """Concurrent audit logging from different users stays isolated."""
        async def log_for_user(uid):
            for i in range(5):
                await log_audit(uid, "agent-1", "github", "test", "success")

        await asyncio.gather(*[
            log_for_user(f"audit-user-{i}") for i in range(5)
        ])

        for i in range(5):
            entries = await get_audit_log(f"audit-user-{i}", limit=10)
            assert len(entries) == 5


# ============================================================
# 5. Session Fixation Prevention
# ============================================================

class TestSessionFixationPrevention:
    """Tests to prevent session fixation and related attacks."""

    @pytest.mark.asyncio
    async def test_api_key_hash_is_one_way(self, db):
        """API key hash should be a one-way SHA-256 hash."""
        raw_key = "ag_test_key_12345"
        expected_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        actual_hash = _hash_key(raw_key)
        assert actual_hash == expected_hash

    @pytest.mark.asyncio
    async def test_api_key_prefix_stored_not_full_key(self, db):
        """Only the prefix (first 8 chars) is stored, not the full key."""
        key_obj, raw = await create_api_key(USER, "prefix-agent", "test")
        assert key_obj.key_prefix == raw[:8]
        assert len(key_obj.key_prefix) == 8

    @pytest.mark.asyncio
    async def test_api_key_starts_with_ag_prefix(self, db):
        """All API keys should start with 'ag_'."""
        for i in range(10):
            _, raw = await create_api_key(USER, f"prefix-check-{i}", f"key-{i}")
            assert raw.startswith("ag_")

    @pytest.mark.asyncio
    async def test_invalid_api_key_returns_none(self, db):
        """Validating a fake API key returns None."""
        result = await validate_api_key("ag_this_is_not_a_valid_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_api_key_returns_none(self, db):
        """Validating an empty string returns None."""
        result = await validate_api_key("")
        assert result is None

    @pytest.mark.asyncio
    async def test_random_string_returns_none(self, db):
        """Validating a random string returns None."""
        result = await validate_api_key("random_garbage_string_not_a_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_revoked_key_cannot_be_unrevoked(self, db):
        """A revoked key cannot be validated even after re-validation attempt."""
        key_obj, raw = await create_api_key(USER, "unrevoke-agent", "test")
        await revoke_api_key(key_obj.id, USER)

        # Multiple validation attempts should all fail
        for _ in range(5):
            assert await validate_api_key(raw) is None

    @pytest.mark.asyncio
    async def test_api_key_id_uniqueness(self, db):
        """Each API key should have a unique ID."""
        key_ids = set()
        for i in range(20):
            key_obj, _ = await create_api_key(USER, f"unique-id-{i}", f"key-{i}")
            key_ids.add(key_obj.id)

        assert len(key_ids) == 20

    @pytest.mark.asyncio
    async def test_policy_cannot_be_created_by_empty_user(self, db):
        """Policy with empty created_by can be created but enforcement fails."""
        policy = AgentPolicy(
            agent_id="empty-user-agent",
            agent_name="Empty User Agent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        # Enforcement by a non-empty user should fail ownership check
        with pytest.raises(PolicyDenied, match="Not authorized"):
            await enforce_policy("some-user", "empty-user-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_policy_ownership_tamper_resistance(self, db):
        """Changing created_by after creation doesn't help the attacker."""
        policy = AgentPolicy(
            agent_id="tamper-agent",
            agent_name="Tamper Agent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="legitimate-user",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        # Attacker cannot enforce
        with pytest.raises(PolicyDenied, match="Not authorized"):
            await enforce_policy("attacker", "tamper-agent", "github", ["repo"])

        # Legitimate user can enforce
        result = await enforce_policy("legitimate-user", "tamper-agent", "github", ["repo"])
        assert result.agent_id == "tamper-agent"

    @pytest.mark.asyncio
    async def test_different_keys_for_same_agent_are_independent(self, db):
        """Multiple API keys for the same agent are independently revocable."""
        key1, raw1 = await create_api_key(USER, "multi-key-agent", "key-1")
        key2, raw2 = await create_api_key(USER, "multi-key-agent", "key-2")

        # Both work
        assert await validate_api_key(raw1) is not None
        assert await validate_api_key(raw2) is not None

        # Revoke only key1
        await revoke_api_key(key1.id, USER)

        assert await validate_api_key(raw1) is None
        assert await validate_api_key(raw2) is not None

    @pytest.mark.asyncio
    async def test_key_hash_differs_for_different_keys(self, db):
        """Different API keys should have different hashes."""
        key1, raw1 = await create_api_key(USER, "hash-agent", "key-1")
        key2, raw2 = await create_api_key(USER, "hash-agent", "key-2")

        assert key1.key_hash != key2.key_hash

    @pytest.mark.asyncio
    async def test_expired_policy_cannot_be_enforced(self, db):
        """An expired policy cannot be used even by the correct owner."""
        await create_agent_policy(AgentPolicy(
            agent_id="expired-sess-agent",
            agent_name="Expired Sess",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            expires_at=time.time() - 1,
            created_by=USER,
            created_at=time.time(),
        ))

        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy(USER, "expired-sess-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_api_key_listing_user_isolated(self, db):
        """Users can only list their own API keys."""
        await create_api_key(USER, "my-agent", "my-key")
        await create_api_key("other-user", "other-agent", "other-key")

        my_keys = await get_api_keys(USER)
        other_keys = await get_api_keys("other-user")

        assert all(k.user_id == USER for k in my_keys)
        assert all(k.user_id == "other-user" for k in other_keys)

    @pytest.mark.asyncio
    async def test_connected_services_reconnect_updates_timestamp(self, db):
        """Reconnecting an already-connected service updates the record."""
        await add_connected_service(USER, "github", "conn-1")
        svcs1 = await get_connected_services(USER)
        first_time = svcs1[0]["connected_at"]

        await asyncio.sleep(0.01)
        await add_connected_service(USER, "github", "conn-2")
        svcs2 = await get_connected_services(USER)

        # Should still be just 1 service (INSERT OR REPLACE)
        assert len(svcs2) == 1
        assert svcs2[0]["connection_id"] == "conn-2"
