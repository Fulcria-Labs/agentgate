"""Security tests — SQL injection attempts, privilege escalation,
policy bypasses, header manipulation, special characters in inputs,
and policy tampering scenarios."""

import time

import pytest

from src.database import (
    AgentPolicy,
    create_agent_policy,
    create_api_key,
    get_agent_policy,
    get_all_policies,
    get_audit_log,
    log_audit,
    validate_api_key,
)
from src.policy import (
    PolicyDenied,
    _rate_counters,
    check_ip_allowlist,
    check_time_window,
    enforce_policy,
    get_effective_scopes,
    requires_step_up,
)


@pytest.fixture(autouse=True)
def clear_rate_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


class TestSQLInjectionInAgentId:
    """SQL injection attempts via agent_id parameter."""

    @pytest.mark.asyncio
    async def test_sql_injection_single_quote(self, db):
        result = await get_agent_policy("'; DROP TABLE agent_policies; --")
        assert result is None

    @pytest.mark.asyncio
    async def test_sql_injection_double_quote(self, db):
        result = await get_agent_policy('" OR 1=1 --')
        assert result is None

    @pytest.mark.asyncio
    async def test_sql_injection_union_select(self, db):
        result = await get_agent_policy("' UNION SELECT * FROM api_keys --")
        assert result is None

    @pytest.mark.asyncio
    async def test_sql_injection_semicolon(self, db):
        result = await get_agent_policy("agent; DELETE FROM agent_policies;")
        assert result is None

    @pytest.mark.asyncio
    async def test_sql_injection_in_policy_creation(self, db):
        """SQL injection in agent_id during creation doesn't break the DB."""
        policy = AgentPolicy(
            agent_id="'; DROP TABLE--", agent_name="Evil Bot",
            created_by="user1",
        )
        await create_agent_policy(policy)
        # Should be stored as literal string
        retrieved = await get_agent_policy("'; DROP TABLE--")
        assert retrieved is not None
        assert retrieved.agent_id == "'; DROP TABLE--"


class TestSQLInjectionInUserId:
    """SQL injection attempts via user_id parameter."""

    @pytest.mark.asyncio
    async def test_sql_injection_in_audit_log(self, db):
        await log_audit("' OR 1=1 --", "agent", "github", "test", "success")
        entries = await get_audit_log("' OR 1=1 --")
        assert len(entries) == 1
        assert entries[0].user_id == "' OR 1=1 --"

    @pytest.mark.asyncio
    async def test_sql_injection_in_policy_query(self, db):
        policies = await get_all_policies("' UNION SELECT * --")
        assert policies == []

    @pytest.mark.asyncio
    async def test_sql_injection_in_api_key_creation(self, db):
        key_obj, raw = await create_api_key("'; DROP TABLE api_keys;--", "agent1")
        result = await validate_api_key(raw)
        assert result is not None
        assert result.user_id == "'; DROP TABLE api_keys;--"


class TestSQLInjectionInServiceFields:
    """SQL injection via service/scopes fields."""

    @pytest.mark.asyncio
    async def test_sql_injection_in_service_name(self, db):
        await log_audit("u1", "a1", "'; DROP TABLE audit_log;--", "test", "success")
        entries = await get_audit_log("u1")
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_sql_injection_in_details(self, db):
        await log_audit("u1", "a1", "github", "test", "success",
                        details="'; UPDATE agent_policies SET is_active=1;--")
        entries = await get_audit_log("u1")
        assert entries[0].details == "'; UPDATE agent_policies SET is_active=1;--"


class TestPrivilegeEscalation:
    """Attempts to escalate privileges."""

    @pytest.mark.asyncio
    async def test_cannot_access_other_users_agent(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="victim-agent", agent_name="Victim Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            created_by="victim", created_at=time.time(), is_active=True,
        ))
        with pytest.raises(PolicyDenied, match="do not own"):
            await enforce_policy("attacker", "victim-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_cannot_escalate_scopes(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="limited-agent", agent_name="Limited Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["read:user"]},
            rate_limit_per_minute=60,
            created_by="u1", created_at=time.time(), is_active=True,
        ))
        with pytest.raises(PolicyDenied, match="Scopes not permitted"):
            await enforce_policy("u1", "limited-agent", "github", ["read:user", "admin:org"])

    @pytest.mark.asyncio
    async def test_cannot_access_unauthorized_service(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="github-only", agent_name="GH Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            created_by="u1", created_at=time.time(), is_active=True,
        ))
        with pytest.raises(PolicyDenied, match="not authorized to access"):
            await enforce_policy("u1", "github-only", "slack", ["channels:read"])

    @pytest.mark.asyncio
    async def test_disabled_agent_cannot_be_used(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="disabled-agent", agent_name="Disabled Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            created_by="u1", created_at=time.time(), is_active=False,
        ))
        with pytest.raises(PolicyDenied, match="disabled"):
            await enforce_policy("u1", "disabled-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_expired_agent_cannot_be_used(self, db):
        await create_agent_policy(AgentPolicy(
            agent_id="expired-agent", agent_name="Expired Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            created_by="u1", created_at=time.time(), is_active=True,
            expires_at=time.time() - 1,
        ))
        with pytest.raises(PolicyDenied, match="expired"):
            await enforce_policy("u1", "expired-agent", "github", ["repo"])


class TestSpecialCharacters:
    """Special characters in various fields."""

    @pytest.mark.asyncio
    async def test_unicode_agent_name(self, db):
        policy = AgentPolicy(
            agent_id="unicode-agent", agent_name="Bot \u2603 \u2764",
            created_by="u1",
        )
        await create_agent_policy(policy)
        retrieved = await get_agent_policy("unicode-agent")
        assert retrieved.agent_name == "Bot \u2603 \u2764"

    @pytest.mark.asyncio
    async def test_emoji_in_agent_name(self, db):
        policy = AgentPolicy(
            agent_id="emoji-agent", agent_name="\U0001f916 RoboBot",
            created_by="u1",
        )
        await create_agent_policy(policy)
        retrieved = await get_agent_policy("emoji-agent")
        assert "\U0001f916" in retrieved.agent_name

    @pytest.mark.asyncio
    async def test_null_bytes_in_agent_id(self, db):
        result = await get_agent_policy("agent\x00null")
        assert result is None

    @pytest.mark.asyncio
    async def test_very_long_agent_id(self, db):
        long_id = "a" * 10000
        policy = AgentPolicy(agent_id=long_id, agent_name="Long", created_by="u1")
        await create_agent_policy(policy)
        retrieved = await get_agent_policy(long_id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_empty_agent_id(self, db):
        policy = AgentPolicy(agent_id="", agent_name="Empty", created_by="u1")
        await create_agent_policy(policy)
        retrieved = await get_agent_policy("")
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_newlines_in_agent_name(self, db):
        policy = AgentPolicy(
            agent_id="newline-agent", agent_name="Bot\nWith\nNewlines",
            created_by="u1",
        )
        await create_agent_policy(policy)
        retrieved = await get_agent_policy("newline-agent")
        assert "\n" in retrieved.agent_name

    @pytest.mark.asyncio
    async def test_html_in_agent_name(self, db):
        policy = AgentPolicy(
            agent_id="html-agent", agent_name="<script>alert('xss')</script>",
            created_by="u1",
        )
        await create_agent_policy(policy)
        retrieved = await get_agent_policy("html-agent")
        assert "<script>" in retrieved.agent_name  # Stored as-is (no XSS sanitization at DB layer)


class TestPolicyDeniedAttributes:
    """Verify PolicyDenied exception attributes."""

    @pytest.mark.asyncio
    async def test_denied_has_agent_id(self, db):
        with pytest.raises(PolicyDenied) as exc_info:
            await enforce_policy("u1", "nonexistent", "github", ["repo"])
        assert exc_info.value.agent_id == "nonexistent"

    @pytest.mark.asyncio
    async def test_denied_has_service(self, db):
        with pytest.raises(PolicyDenied) as exc_info:
            await enforce_policy("u1", "nonexistent", "my-service", ["scope"])
        assert exc_info.value.service == "my-service"

    @pytest.mark.asyncio
    async def test_denied_has_reason(self, db):
        with pytest.raises(PolicyDenied) as exc_info:
            await enforce_policy("u1", "nonexistent", "github", ["repo"])
        assert "not registered" in exc_info.value.reason

    def test_denied_str_is_reason(self):
        exc = PolicyDenied("test reason", "agent-1", "github")
        assert str(exc) == "test reason"

    def test_denied_default_agent_id_empty(self):
        exc = PolicyDenied("reason")
        assert exc.agent_id == ""

    def test_denied_default_service_empty(self):
        exc = PolicyDenied("reason")
        assert exc.service == ""


class TestScopeManipulation:
    """Attempts to manipulate scopes."""

    def test_effective_scopes_empty_requested(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={"github": ["repo", "read:user"]},
        )
        result = get_effective_scopes(policy, "github", [])
        assert result == []

    def test_effective_scopes_empty_allowed(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={"github": []},
        )
        result = get_effective_scopes(policy, "github", ["repo"])
        assert result == []

    def test_effective_scopes_exact_match(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={"github": ["repo"]},
        )
        result = get_effective_scopes(policy, "github", ["repo"])
        assert result == ["repo"]

    def test_effective_scopes_subset(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={"github": ["repo", "read:user", "gist"]},
        )
        result = get_effective_scopes(policy, "github", ["repo", "gist"])
        assert set(result) == {"repo", "gist"}

    def test_effective_scopes_superset_filtered(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            allowed_scopes={"github": ["repo"]},
        )
        result = get_effective_scopes(policy, "github", ["repo", "admin:org", "delete_repo"])
        assert result == ["repo"]


class TestStepUpAuthCheck:
    """Step-up authentication requirement checks."""

    def test_step_up_required(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            requires_step_up=["slack", "google"],
        )
        assert requires_step_up(policy, "slack") is True
        assert requires_step_up(policy, "google") is True

    def test_step_up_not_required(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            requires_step_up=["slack"],
        )
        assert requires_step_up(policy, "github") is False

    def test_step_up_empty_list(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            requires_step_up=[],
        )
        assert requires_step_up(policy, "any") is False

    def test_step_up_nonexistent_service(self):
        policy = AgentPolicy(
            agent_id="t", agent_name="t",
            requires_step_up=["slack"],
        )
        assert requires_step_up(policy, "nonexistent") is False
