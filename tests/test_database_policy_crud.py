"""Database policy CRUD tests — create, read, update, delete,
toggle, serialization/deserialization of JSON fields."""

import json
import time

import pytest

from src.database import (
    AgentPolicy,
    create_agent_policy,
    delete_agent_policy,
    get_agent_policy,
    get_all_policies,
    init_db,
    toggle_agent_policy,
)


class TestCreatePolicy:
    """Policy creation edge cases."""

    @pytest.mark.asyncio
    async def test_minimal_policy(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="min", agent_name="Min Bot", created_by="u1",
        ))
        p = await get_agent_policy("min")
        assert p is not None
        assert p.agent_id == "min"
        assert p.agent_name == "Min Bot"

    @pytest.mark.asyncio
    async def test_full_policy(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="full", agent_name="Full Bot",
            allowed_services=["github", "slack", "google"],
            allowed_scopes={"github": ["repo"], "slack": ["chat:write"]},
            rate_limit_per_minute=100,
            requires_step_up=["google"],
            created_by="u1", created_at=1000.0, is_active=True,
            allowed_hours=[9, 10, 11, 12],
            allowed_days=[0, 1, 2, 3, 4],
            expires_at=9999999999.0,
            ip_allowlist=["10.0.0.0/8"],
        ))
        p = await get_agent_policy("full")
        assert p.allowed_services == ["github", "slack", "google"]
        assert p.allowed_scopes["github"] == ["repo"]
        assert p.rate_limit_per_minute == 100
        assert p.requires_step_up == ["google"]
        assert p.allowed_hours == [9, 10, 11, 12]
        assert p.allowed_days == [0, 1, 2, 3, 4]
        assert p.expires_at == 9999999999.0
        assert p.ip_allowlist == ["10.0.0.0/8"]

    @pytest.mark.asyncio
    async def test_policy_preserves_scope_structure(self, db):
        scopes = {
            "github": ["repo", "read:user", "gist"],
            "slack": ["channels:read", "chat:write", "users:read"],
        }
        await create_agent_policy(AgentPolicy(
            agent_id="scope-test", agent_name="Scope Bot",
            allowed_scopes=scopes, created_by="u1",
        ))
        p = await get_agent_policy("scope-test")
        assert p.allowed_scopes == scopes

    @pytest.mark.asyncio
    async def test_empty_services_list(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="no-svc", agent_name="No Service Bot",
            allowed_services=[], created_by="u1",
        ))
        p = await get_agent_policy("no-svc")
        assert p.allowed_services == []

    @pytest.mark.asyncio
    async def test_empty_scopes_dict(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="no-scope", agent_name="No Scope Bot",
            allowed_scopes={}, created_by="u1",
        ))
        p = await get_agent_policy("no-scope")
        assert p.allowed_scopes == {}

    @pytest.mark.asyncio
    async def test_created_at_auto_set(self, db):
        """If created_at is 0 (falsy), it gets auto-set to current time."""
        await create_agent_policy(AgentPolicy(
            agent_id="auto-time", agent_name="Auto Time Bot",
            created_by="u1", created_at=0.0,
        ))
        p = await get_agent_policy("auto-time")
        assert p.created_at > 0


class TestUpdatePolicy:
    """Policy update (INSERT OR REPLACE) behavior."""

    @pytest.mark.asyncio
    async def test_update_changes_name(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="upd", agent_name="v1", created_by="u1",
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="upd", agent_name="v2", created_by="u1",
        ))
        p = await get_agent_policy("upd")
        assert p.agent_name == "v2"

    @pytest.mark.asyncio
    async def test_update_changes_services(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="upd-svc", agent_name="Bot",
            allowed_services=["github"], created_by="u1",
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="upd-svc", agent_name="Bot",
            allowed_services=["github", "slack"], created_by="u1",
        ))
        p = await get_agent_policy("upd-svc")
        assert p.allowed_services == ["github", "slack"]

    @pytest.mark.asyncio
    async def test_update_changes_rate_limit(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="upd-rl", agent_name="Bot",
            rate_limit_per_minute=10, created_by="u1",
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="upd-rl", agent_name="Bot",
            rate_limit_per_minute=100, created_by="u1",
        ))
        p = await get_agent_policy("upd-rl")
        assert p.rate_limit_per_minute == 100

    @pytest.mark.asyncio
    async def test_update_changes_ip_allowlist(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="upd-ip", agent_name="Bot",
            ip_allowlist=["10.0.0.1"], created_by="u1",
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="upd-ip", agent_name="Bot",
            ip_allowlist=["192.168.0.0/16"], created_by="u1",
        ))
        p = await get_agent_policy("upd-ip")
        assert p.ip_allowlist == ["192.168.0.0/16"]

    @pytest.mark.asyncio
    async def test_update_changes_time_windows(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="upd-tw", agent_name="Bot",
            allowed_hours=[9], allowed_days=[0], created_by="u1",
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="upd-tw", agent_name="Bot",
            allowed_hours=[10, 11], allowed_days=[1, 2], created_by="u1",
        ))
        p = await get_agent_policy("upd-tw")
        assert p.allowed_hours == [10, 11]
        assert p.allowed_days == [1, 2]


class TestDeletePolicy:
    """Policy deletion."""

    @pytest.mark.asyncio
    async def test_delete_existing_returns_true(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="del", agent_name="Bot", created_by="u1",
        ))
        assert await delete_agent_policy("del", "u1") is True

    @pytest.mark.asyncio
    async def test_delete_removes_policy(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="del-gone", agent_name="Bot", created_by="u1",
        ))
        await delete_agent_policy("del-gone", "u1")
        p = await get_agent_policy("del-gone")
        assert p is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, db):
        assert await delete_agent_policy("nope", "u1") is False

    @pytest.mark.asyncio
    async def test_delete_wrong_user_returns_false(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="del-wrong", agent_name="Bot", created_by="owner",
        ))
        assert await delete_agent_policy("del-wrong", "attacker") is False

    @pytest.mark.asyncio
    async def test_delete_doesnt_affect_other_policies(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="keep", agent_name="Keep Bot", created_by="u1",
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="remove", agent_name="Remove Bot", created_by="u1",
        ))
        await delete_agent_policy("remove", "u1")
        p = await get_agent_policy("keep")
        assert p is not None


class TestTogglePolicy:
    """Policy toggle behavior."""

    @pytest.mark.asyncio
    async def test_toggle_active_to_inactive(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="tog", agent_name="Bot", created_by="u1", is_active=True,
        ))
        result = await toggle_agent_policy("tog", "u1")
        assert result is False

    @pytest.mark.asyncio
    async def test_toggle_inactive_to_active(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="tog-i", agent_name="Bot", created_by="u1", is_active=False,
        ))
        result = await toggle_agent_policy("tog-i", "u1")
        assert result is True

    @pytest.mark.asyncio
    async def test_toggle_twice_returns_to_original(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="tog-2x", agent_name="Bot", created_by="u1", is_active=True,
        ))
        await toggle_agent_policy("tog-2x", "u1")  # True -> False
        result = await toggle_agent_policy("tog-2x", "u1")  # False -> True
        assert result is True

    @pytest.mark.asyncio
    async def test_toggle_nonexistent_returns_none(self, db):
        result = await toggle_agent_policy("fake", "u1")
        assert result is None

    @pytest.mark.asyncio
    async def test_toggle_wrong_user_returns_none(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="tog-wrong", agent_name="Bot", created_by="owner",
        ))
        result = await toggle_agent_policy("tog-wrong", "attacker")
        assert result is None

    @pytest.mark.asyncio
    async def test_toggle_persists_in_db(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="tog-persist", agent_name="Bot", created_by="u1", is_active=True,
        ))
        await toggle_agent_policy("tog-persist", "u1")
        p = await get_agent_policy("tog-persist")
        assert p.is_active is False


class TestGetAllPolicies:
    """Get all policies for a user."""

    @pytest.mark.asyncio
    async def test_empty_for_new_user(self, db):
        policies = await get_all_policies("new-user")
        assert policies == []

    @pytest.mark.asyncio
    async def test_returns_correct_count(self, db):
        for i in range(7):
            await create_agent_policy(AgentPolicy(
                agent_id=f"list-{i}", agent_name=f"Bot {i}", created_by="u1",
            ))
        policies = await get_all_policies("u1")
        assert len(policies) == 7

    @pytest.mark.asyncio
    async def test_only_user_policies(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="u1-pol", agent_name="U1 Bot", created_by="u1",
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="u2-pol", agent_name="U2 Bot", created_by="u2",
        ))
        policies = await get_all_policies("u1")
        assert len(policies) == 1
        assert policies[0].agent_id == "u1-pol"

    @pytest.mark.asyncio
    async def test_includes_inactive_policies(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="active-p", agent_name="Bot", created_by="u1", is_active=True,
        ))
        await create_agent_policy(AgentPolicy(
            agent_id="inactive-p", agent_name="Bot", created_by="u1", is_active=False,
        ))
        policies = await get_all_policies("u1")
        assert len(policies) == 2
