"""Advanced database tests — edge cases, concurrency, data integrity."""

import time
import pytest

from src.database import (
    AgentPolicy,
    AuditEntry,
    ApiKey,
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


# ---------------------------------------------------------------------------
# 1. Data model / dataclass tests
# ---------------------------------------------------------------------------

class TestDataModels:
    """Verify defaults and structure of dataclass models."""

    def test_agent_policy_defaults(self):
        p = AgentPolicy(agent_id="a", agent_name="A")
        assert p.allowed_services == []
        assert p.allowed_scopes == {}
        assert p.rate_limit_per_minute == 60
        assert p.requires_step_up == []
        assert p.created_by == ""
        assert p.created_at == 0.0
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
        assert k.user_id == ""
        assert k.agent_id == ""
        assert k.name == ""
        assert k.created_at == 0.0
        assert k.expires_at == 0.0
        assert k.is_revoked is False
        assert k.last_used_at == 0.0

    def test_audit_entry_defaults(self):
        e = AuditEntry()
        assert e.id == 0
        assert e.timestamp == 0.0
        assert e.user_id == ""
        assert e.agent_id == ""
        assert e.service == ""
        assert e.scopes == ""
        assert e.action == ""
        assert e.status == ""
        assert e.ip_address == ""
        assert e.details == ""

    def test_hash_key_deterministic(self):
        assert _hash_key("test") == _hash_key("test")

    def test_hash_key_different_inputs(self):
        assert _hash_key("key1") != _hash_key("key2")


# ---------------------------------------------------------------------------
# 2. Policy CRUD edge cases
# ---------------------------------------------------------------------------

class TestPolicyCRUDEdgeCases:
    """Edge cases for agent policy create/read/update/delete operations."""

    @pytest.mark.asyncio
    async def test_create_policy_preserves_all_fields(self, db):
        now = time.time()
        policy = AgentPolicy(
            agent_id="full",
            agent_name="Full Bot",
            allowed_services=["github", "slack"],
            allowed_scopes={"github": ["repo"], "slack": ["chat:write"]},
            rate_limit_per_minute=42,
            requires_step_up=["slack"],
            created_by="user-x",
            created_at=now,
            is_active=True,
            allowed_hours=[9, 10, 11],
            allowed_days=[0, 1, 2],
            expires_at=now + 3600,
            ip_allowlist=["10.0.0.0/8"],
        )
        await create_agent_policy(policy)
        p = await get_agent_policy("full")
        assert p.agent_name == "Full Bot"
        assert p.allowed_services == ["github", "slack"]
        assert p.allowed_scopes == {"github": ["repo"], "slack": ["chat:write"]}
        assert p.rate_limit_per_minute == 42
        assert p.requires_step_up == ["slack"]
        assert p.created_by == "user-x"
        assert p.is_active is True
        assert p.allowed_hours == [9, 10, 11]
        assert p.allowed_days == [0, 1, 2]
        assert p.expires_at == now + 3600
        assert p.ip_allowlist == ["10.0.0.0/8"]

    @pytest.mark.asyncio
    async def test_update_policy_overwrites_all_fields(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="up", agent_name="V1", allowed_services=["github"],
            created_by="u1", created_at=1.0, is_active=True,
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="up", agent_name="V2", allowed_services=["slack"],
            created_by="u1", created_at=2.0, is_active=False,
        ))
        p = await get_agent_policy("up")
        assert p.agent_name == "V2"
        assert p.allowed_services == ["slack"]
        assert p.is_active is False

    @pytest.mark.asyncio
    async def test_delete_returns_false_for_wrong_user(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="d", agent_name="Bot", created_by="owner",
        ))
        assert await delete_agent_policy("d", "attacker") is False
        # Still exists
        assert await get_agent_policy("d") is not None

    @pytest.mark.asyncio
    async def test_delete_returns_false_for_nonexistent(self, db):
        assert await delete_agent_policy("ghost", "user1") is False

    @pytest.mark.asyncio
    async def test_empty_services_and_scopes(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="empty", agent_name="Empty",
            allowed_services=[], allowed_scopes={},
            created_by="u", created_at=0,
        ))
        p = await get_agent_policy("empty")
        assert p.allowed_services == []
        assert p.allowed_scopes == {}

    @pytest.mark.asyncio
    async def test_get_all_policies_empty(self, db):
        policies = await get_all_policies("nobody")
        assert policies == []


# ---------------------------------------------------------------------------
# 3. API key edge cases
# ---------------------------------------------------------------------------

class TestApiKeyEdgeCases:
    """Edge cases for API key lifecycle."""

    @pytest.mark.asyncio
    async def test_key_starts_with_ag_prefix(self, db):
        _, raw = await create_api_key("u", "a")
        assert raw.startswith("ag_")

    @pytest.mark.asyncio
    async def test_two_keys_are_unique(self, db):
        _, raw1 = await create_api_key("u", "a")
        _, raw2 = await create_api_key("u", "a")
        assert raw1 != raw2

    @pytest.mark.asyncio
    async def test_validate_updates_last_used_timestamp(self, db):
        k, raw = await create_api_key("u", "a")
        before = time.time()
        result = await validate_api_key(raw)
        after = time.time()
        assert before <= result.last_used_at <= after

    @pytest.mark.asyncio
    async def test_revoke_already_revoked_key(self, db):
        k, raw = await create_api_key("u", "a")
        assert await revoke_api_key(k.id, "u") is True
        # Revoking again should still return True (rowcount update)
        # Actually it sets is_revoked=1 on an already revoked key, but the
        # WHERE clause doesn't filter on is_revoked, so rowcount=1
        result = await revoke_api_key(k.id, "u")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_keys_excludes_hash(self, db):
        await create_api_key("u", "a", "k")
        keys = await get_api_keys("u")
        assert len(keys) == 1
        assert keys[0].key_hash == ""

    @pytest.mark.asyncio
    async def test_get_keys_returns_empty_for_unknown_user(self, db):
        keys = await get_api_keys("nobody")
        assert keys == []

    @pytest.mark.asyncio
    async def test_key_with_zero_expires_never_expires(self, db):
        _, raw = await create_api_key("u", "a", expires_in=0)
        result = await validate_api_key(raw)
        assert result is not None
        assert result.expires_at == 0.0

    @pytest.mark.asyncio
    async def test_key_with_future_expiry_validates(self, db):
        _, raw = await create_api_key("u", "a", expires_in=3600)
        result = await validate_api_key(raw)
        assert result is not None

    @pytest.mark.asyncio
    async def test_key_name_preserved(self, db):
        _, raw = await create_api_key("u", "a", "my-special-key")
        result = await validate_api_key(raw)
        assert result.name == "my-special-key"


# ---------------------------------------------------------------------------
# 4. Connected services edge cases
# ---------------------------------------------------------------------------

class TestConnectedServicesEdgeCases:
    """Edge cases for connected service operations."""

    @pytest.mark.asyncio
    async def test_remove_nonexistent_service_is_noop(self, db):
        # Should not raise
        await remove_connected_service("u", "nonexistent")

    @pytest.mark.asyncio
    async def test_empty_connection_id(self, db):
        await add_connected_service("u", "github", "")
        services = await get_connected_services("u")
        assert len(services) == 1
        assert services[0]["connection_id"] == ""

    @pytest.mark.asyncio
    async def test_multiple_services_ordering(self, db):
        await add_connected_service("u", "github")
        await add_connected_service("u", "slack")
        await add_connected_service("u", "google")
        services = await get_connected_services("u")
        assert len(services) == 3
        # Most recently connected first
        assert services[0]["service"] == "google"

    @pytest.mark.asyncio
    async def test_services_isolated_between_users(self, db):
        await add_connected_service("u1", "github")
        await add_connected_service("u2", "slack")
        s1 = await get_connected_services("u1")
        s2 = await get_connected_services("u2")
        assert len(s1) == 1
        assert s1[0]["service"] == "github"
        assert len(s2) == 1
        assert s2[0]["service"] == "slack"

    @pytest.mark.asyncio
    async def test_get_connected_services_empty(self, db):
        services = await get_connected_services("nobody")
        assert services == []


# ---------------------------------------------------------------------------
# 5. Audit log edge cases
# ---------------------------------------------------------------------------

class TestAuditLogEdgeCases:
    """Edge cases for audit logging."""

    @pytest.mark.asyncio
    async def test_audit_with_all_optional_fields(self, db):
        await log_audit("u", "a", "github", "test", "ok",
                       scopes="repo,read:user", ip_address="1.2.3.4",
                       details="some detail")
        entries = await get_audit_log("u")
        assert len(entries) == 1
        assert entries[0].scopes == "repo,read:user"
        assert entries[0].ip_address == "1.2.3.4"
        assert entries[0].details == "some detail"

    @pytest.mark.asyncio
    async def test_audit_with_empty_optional_fields(self, db):
        await log_audit("u", "a", "svc", "act", "ok")
        entries = await get_audit_log("u")
        assert len(entries) == 1
        assert entries[0].scopes == ""
        assert entries[0].ip_address == ""
        assert entries[0].details == ""

    @pytest.mark.asyncio
    async def test_audit_log_timestamp_is_recent(self, db):
        before = time.time()
        await log_audit("u", "a", "s", "act", "ok")
        after = time.time()
        entries = await get_audit_log("u")
        assert before <= entries[0].timestamp <= after

    @pytest.mark.asyncio
    async def test_audit_log_has_auto_increment_id(self, db):
        await log_audit("u", "a", "s", "act1", "ok")
        await log_audit("u", "a", "s", "act2", "ok")
        entries = await get_audit_log("u")
        ids = [e.id for e in entries]
        assert len(set(ids)) == 2  # Unique IDs
        assert all(i > 0 for i in ids)

    @pytest.mark.asyncio
    async def test_audit_log_large_limit(self, db):
        for i in range(5):
            await log_audit("u", "a", "s", f"act{i}", "ok")
        entries = await get_audit_log("u", limit=1000)
        assert len(entries) == 5


# ---------------------------------------------------------------------------
# 6. Emergency revoke edge cases
# ---------------------------------------------------------------------------

class TestEmergencyRevokeEdgeCases:
    """Edge cases for the emergency revoke mechanism."""

    @pytest.mark.asyncio
    async def test_double_emergency_revoke(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="em", agent_name="Bot", created_by="u",
            is_active=True,
        ))
        _, raw = await create_api_key("u", "em")
        r1 = await emergency_revoke_all("u")
        assert r1["policies_disabled"] == 1
        assert r1["keys_revoked"] == 1

        # Second revoke should find nothing new to revoke
        r2 = await emergency_revoke_all("u")
        assert r2["policies_disabled"] == 1  # Updates count (already disabled)
        assert r2["keys_revoked"] == 0  # Already revoked keys not counted again

    @pytest.mark.asyncio
    async def test_emergency_revoke_with_no_resources(self, db):
        result = await emergency_revoke_all("ghost-user")
        assert result["policies_disabled"] == 0
        assert result["keys_revoked"] == 0

    @pytest.mark.asyncio
    async def test_emergency_revoke_mixed_active_inactive(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="em1", agent_name="Bot1", created_by="u", is_active=True,
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="em2", agent_name="Bot2", created_by="u", is_active=False,
        ))
        result = await emergency_revoke_all("u")
        # Both get set to inactive (one already was, but UPDATE still affects it)
        assert result["policies_disabled"] == 2


# ---------------------------------------------------------------------------
# 7. Toggle edge cases
# ---------------------------------------------------------------------------

class TestToggleEdgeCases:
    """Edge cases for policy toggling."""

    @pytest.mark.asyncio
    async def test_toggle_twice_returns_to_original(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="tog", agent_name="Bot", created_by="u", is_active=True,
        ))
        s1 = await toggle_agent_policy("tog", "u")
        assert s1 is False
        s2 = await toggle_agent_policy("tog", "u")
        assert s2 is True

    @pytest.mark.asyncio
    async def test_toggle_wrong_user_returns_none(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="tog", agent_name="Bot", created_by="u1",
        ))
        result = await toggle_agent_policy("tog", "u2")
        assert result is None

    @pytest.mark.asyncio
    async def test_toggle_nonexistent_returns_none(self, db):
        result = await toggle_agent_policy("ghost", "u")
        assert result is None


# ---------------------------------------------------------------------------
# 8. Init DB idempotency
# ---------------------------------------------------------------------------

class TestInitDB:
    """Verify database initialization is idempotent."""

    @pytest.mark.asyncio
    async def test_double_init_no_error(self, db):
        # db fixture already called init_db once; call again
        await init_db()

    @pytest.mark.asyncio
    async def test_data_survives_reinit(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="surv", agent_name="Survivor", created_by="u",
        ))
        await init_db()  # Re-init
        p = await get_agent_policy("surv")
        assert p is not None
        assert p.agent_name == "Survivor"
