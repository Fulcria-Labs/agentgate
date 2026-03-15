"""Comprehensive tests for the consent management module."""

import time
import pytest
import pytest_asyncio

from src.database import init_db, create_agent_policy, AgentPolicy
from src.consent import (
    ConsentAuditEntry,
    ConsentError,
    ConsentGrant,
    ConsentRequest,
    ConsentScope,
    ConsentStatus,
    approve_consent,
    check_consent,
    create_consent_grant,
    create_consent_request,
    deny_consent,
    get_consent_audit_log,
    get_consent_grant,
    get_consent_summary,
    init_consent_tables,
    list_consent_grants,
    list_consent_requests,
    resolve_consent_request,
    revoke_all_agent_consent,
    revoke_consent,
    use_consent,
    _pattern_matches,
    _check_conditions,
)


USER_ID = "user|consent-test"
OTHER_USER = "user|other"
AGENT_ID = "agent-consent-1"
AGENT_ID_2 = "agent-consent-2"


@pytest_asyncio.fixture
async def consent_db(db, monkeypatch):
    """Set up consent tables alongside normal DB."""
    monkeypatch.setattr("src.consent.DB_PATH", db)
    await init_consent_tables()
    return db


@pytest_asyncio.fixture
async def agent_policy(consent_db):
    """Create a test agent policy."""
    policy = AgentPolicy(
        agent_id=AGENT_ID,
        agent_name="ConsentTestBot",
        allowed_services=["github", "slack"],
        allowed_scopes={
            "github": ["repo", "read:user"],
            "slack": ["chat:write", "channels:read"],
        },
        rate_limit_per_minute=60,
        created_by=USER_ID,
        created_at=time.time(),
    )
    await create_agent_policy(policy)
    return policy


# ============================================================
# ConsentGrant creation tests
# ============================================================

class TestCreateConsentGrant:
    @pytest.mark.asyncio
    async def test_basic_creation(self, consent_db):
        grant = await create_consent_grant(
            user_id=USER_ID,
            agent_id=AGENT_ID,
            service="github",
        )
        assert grant.id
        assert grant.user_id == USER_ID
        assert grant.agent_id == AGENT_ID
        assert grant.service == "github"
        assert grant.status == ConsentStatus.PENDING.value
        assert grant.scope_type == "service"

    @pytest.mark.asyncio
    async def test_auto_approve(self, consent_db):
        grant = await create_consent_grant(
            user_id=USER_ID,
            agent_id=AGENT_ID,
            service="github",
            auto_approve=True,
        )
        assert grant.status == ConsentStatus.APPROVED.value
        assert grant.approved_at > 0

    @pytest.mark.asyncio
    async def test_with_expiration(self, consent_db):
        future = time.time() + 3600
        grant = await create_consent_grant(
            user_id=USER_ID,
            agent_id=AGENT_ID,
            service="github",
            expires_at=future,
        )
        assert grant.expires_at == future

    @pytest.mark.asyncio
    async def test_with_max_uses(self, consent_db):
        grant = await create_consent_grant(
            user_id=USER_ID,
            agent_id=AGENT_ID,
            service="github",
            max_uses=5,
        )
        assert grant.max_uses == 5
        assert grant.current_uses == 0

    @pytest.mark.asyncio
    async def test_blanket_scope(self, consent_db):
        grant = await create_consent_grant(
            user_id=USER_ID,
            agent_id=AGENT_ID,
            service="github",
            scope_type="blanket",
        )
        assert grant.scope_type == "blanket"

    @pytest.mark.asyncio
    async def test_action_scope(self, consent_db):
        grant = await create_consent_grant(
            user_id=USER_ID,
            agent_id=AGENT_ID,
            service="github",
            scope_type="action",
            action_pattern="read*",
        )
        assert grant.scope_type == "action"
        assert grant.action_pattern == "read*"

    @pytest.mark.asyncio
    async def test_resource_scope(self, consent_db):
        grant = await create_consent_grant(
            user_id=USER_ID,
            agent_id=AGENT_ID,
            service="github",
            scope_type="resource",
            resource_pattern="/repos/myorg/*",
        )
        assert grant.scope_type == "resource"
        assert grant.resource_pattern == "/repos/myorg/*"

    @pytest.mark.asyncio
    async def test_one_time_scope(self, consent_db):
        grant = await create_consent_grant(
            user_id=USER_ID,
            agent_id=AGENT_ID,
            service="github",
            scope_type="one_time",
        )
        assert grant.scope_type == "one_time"

    @pytest.mark.asyncio
    async def test_with_conditions(self, consent_db):
        conditions = {"require_mfa": True, "allowed_hours": [9, 10, 11, 12]}
        grant = await create_consent_grant(
            user_id=USER_ID,
            agent_id=AGENT_ID,
            service="github",
            conditions=conditions,
        )
        assert grant.conditions == conditions

    @pytest.mark.asyncio
    async def test_with_metadata(self, consent_db):
        metadata = {"purpose": "CI/CD pipeline", "ticket": "JIRA-123"}
        grant = await create_consent_grant(
            user_id=USER_ID,
            agent_id=AGENT_ID,
            service="github",
            metadata=metadata,
        )
        assert grant.metadata == metadata

    @pytest.mark.asyncio
    async def test_missing_user_id(self, consent_db):
        with pytest.raises(ConsentError, match="user_id is required"):
            await create_consent_grant(
                user_id="",
                agent_id=AGENT_ID,
                service="github",
            )

    @pytest.mark.asyncio
    async def test_missing_agent_id(self, consent_db):
        with pytest.raises(ConsentError, match="agent_id is required"):
            await create_consent_grant(
                user_id=USER_ID,
                agent_id="",
                service="github",
            )

    @pytest.mark.asyncio
    async def test_missing_service(self, consent_db):
        with pytest.raises(ConsentError, match="service is required"):
            await create_consent_grant(
                user_id=USER_ID,
                agent_id=AGENT_ID,
                service="",
            )

    @pytest.mark.asyncio
    async def test_invalid_scope_type(self, consent_db):
        with pytest.raises(ConsentError, match="Invalid scope_type"):
            await create_consent_grant(
                user_id=USER_ID,
                agent_id=AGENT_ID,
                service="github",
                scope_type="invalid",
            )

    @pytest.mark.asyncio
    async def test_invalid_mfa_condition(self, consent_db):
        with pytest.raises(ConsentError, match="require_mfa"):
            await create_consent_grant(
                user_id=USER_ID,
                agent_id=AGENT_ID,
                service="github",
                conditions={"require_mfa": "yes"},  # Should be bool
            )

    @pytest.mark.asyncio
    async def test_multiple_grants_same_agent(self, consent_db):
        g1 = await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        g2 = await create_consent_grant(USER_ID, AGENT_ID, "slack", auto_approve=True)
        assert g1.id != g2.id
        assert g1.service == "github"
        assert g2.service == "slack"

    @pytest.mark.asyncio
    async def test_grants_different_users(self, consent_db):
        g1 = await create_consent_grant(USER_ID, AGENT_ID, "github")
        g2 = await create_consent_grant(OTHER_USER, AGENT_ID, "github")
        assert g1.user_id == USER_ID
        assert g2.user_id == OTHER_USER

    @pytest.mark.asyncio
    async def test_created_at_set(self, consent_db):
        before = time.time()
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github")
        after = time.time()
        assert before <= grant.created_at <= after


# ============================================================
# Consent approval/denial/revocation tests
# ============================================================

class TestConsentLifecycle:
    @pytest.mark.asyncio
    async def test_approve_pending_grant(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github")
        assert grant.status == "pending"

        approved = await approve_consent(grant.id, USER_ID)
        assert approved is not None
        assert approved.status == "approved"
        assert approved.approved_at > 0

    @pytest.mark.asyncio
    async def test_approve_wrong_user(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github")
        result = await approve_consent(grant.id, OTHER_USER)
        assert result is None

    @pytest.mark.asyncio
    async def test_approve_nonexistent(self, consent_db):
        result = await approve_consent("nonexistent", USER_ID)
        assert result is None

    @pytest.mark.asyncio
    async def test_cannot_approve_already_approved(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        with pytest.raises(ConsentError, match="Cannot approve"):
            await approve_consent(grant.id, USER_ID)

    @pytest.mark.asyncio
    async def test_deny_pending_grant(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github")
        denied = await deny_consent(grant.id, USER_ID, reason="Not needed")
        assert denied is not None
        assert denied.status == "denied"

    @pytest.mark.asyncio
    async def test_deny_wrong_user(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github")
        result = await deny_consent(grant.id, OTHER_USER)
        assert result is None

    @pytest.mark.asyncio
    async def test_deny_nonexistent(self, consent_db):
        result = await deny_consent("nonexistent", USER_ID)
        assert result is None

    @pytest.mark.asyncio
    async def test_cannot_deny_already_denied(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github")
        await deny_consent(grant.id, USER_ID)
        with pytest.raises(ConsentError, match="Cannot deny"):
            await deny_consent(grant.id, USER_ID)

    @pytest.mark.asyncio
    async def test_revoke_approved_grant(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        revoked = await revoke_consent(grant.id, USER_ID, reason="Security concern")
        assert revoked is not None
        assert revoked.status == "revoked"

    @pytest.mark.asyncio
    async def test_revoke_pending_grant(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github")
        revoked = await revoke_consent(grant.id, USER_ID)
        assert revoked is not None
        assert revoked.status == "revoked"

    @pytest.mark.asyncio
    async def test_cannot_revoke_denied(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github")
        await deny_consent(grant.id, USER_ID)
        with pytest.raises(ConsentError, match="Cannot revoke"):
            await revoke_consent(grant.id, USER_ID)

    @pytest.mark.asyncio
    async def test_cannot_revoke_already_revoked(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        await revoke_consent(grant.id, USER_ID)
        with pytest.raises(ConsentError, match="Cannot revoke"):
            await revoke_consent(grant.id, USER_ID)

    @pytest.mark.asyncio
    async def test_revoke_wrong_user(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        result = await revoke_consent(grant.id, OTHER_USER)
        assert result is None


# ============================================================
# check_consent tests
# ============================================================

class TestCheckConsent:
    @pytest.mark.asyncio
    async def test_approved_grant_matches(self, consent_db):
        await create_consent_grant(
            USER_ID, AGENT_ID, "github", auto_approve=True,
        )
        has_consent, grant = await check_consent(USER_ID, AGENT_ID, "github")
        assert has_consent is True
        assert grant is not None
        assert grant.service == "github"

    @pytest.mark.asyncio
    async def test_pending_grant_does_not_match(self, consent_db):
        await create_consent_grant(USER_ID, AGENT_ID, "github")
        has_consent, grant = await check_consent(USER_ID, AGENT_ID, "github")
        assert has_consent is False
        assert grant is None

    @pytest.mark.asyncio
    async def test_no_consent_returns_false(self, consent_db):
        has_consent, grant = await check_consent(USER_ID, AGENT_ID, "github")
        assert has_consent is False
        assert grant is None

    @pytest.mark.asyncio
    async def test_wrong_service_returns_false(self, consent_db):
        await create_consent_grant(
            USER_ID, AGENT_ID, "github", auto_approve=True,
        )
        has_consent, grant = await check_consent(USER_ID, AGENT_ID, "slack")
        assert has_consent is False

    @pytest.mark.asyncio
    async def test_wrong_agent_returns_false(self, consent_db):
        await create_consent_grant(
            USER_ID, AGENT_ID, "github", auto_approve=True,
        )
        has_consent, grant = await check_consent(USER_ID, AGENT_ID_2, "github")
        assert has_consent is False

    @pytest.mark.asyncio
    async def test_expired_grant_not_matched(self, consent_db):
        await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            auto_approve=True,
            expires_at=time.time() - 3600,  # Expired 1 hour ago
        )
        has_consent, grant = await check_consent(USER_ID, AGENT_ID, "github")
        assert has_consent is False

    @pytest.mark.asyncio
    async def test_exhausted_uses_not_matched(self, consent_db):
        grant = await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            auto_approve=True,
            max_uses=1,
        )
        await use_consent(grant.id)
        has_consent, _ = await check_consent(USER_ID, AGENT_ID, "github")
        assert has_consent is False

    @pytest.mark.asyncio
    async def test_action_pattern_matching(self, consent_db):
        await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            scope_type="action",
            action_pattern="read*",
            auto_approve=True,
        )
        # Matches
        has_consent, _ = await check_consent(
            USER_ID, AGENT_ID, "github", action="read_issues",
        )
        assert has_consent is True

        # Does not match
        has_consent, _ = await check_consent(
            USER_ID, AGENT_ID, "github", action="write_issues",
        )
        assert has_consent is False

    @pytest.mark.asyncio
    async def test_resource_pattern_matching(self, consent_db):
        await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            scope_type="resource",
            resource_pattern="/repos/org/*",
            auto_approve=True,
        )
        has_consent, _ = await check_consent(
            USER_ID, AGENT_ID, "github", resource="/repos/org/myrepo",
        )
        assert has_consent is True

        has_consent, _ = await check_consent(
            USER_ID, AGENT_ID, "github", resource="/repos/other/myrepo",
        )
        assert has_consent is False

    @pytest.mark.asyncio
    async def test_wildcard_action_matches_all(self, consent_db):
        await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            action_pattern="*",
            auto_approve=True,
        )
        has_consent, _ = await check_consent(
            USER_ID, AGENT_ID, "github", action="anything",
        )
        assert has_consent is True

    @pytest.mark.asyncio
    async def test_condition_time_after(self, consent_db):
        future = time.time() + 3600
        await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            conditions={"time_after": future},
            auto_approve=True,
        )
        has_consent, _ = await check_consent(USER_ID, AGENT_ID, "github")
        assert has_consent is False

    @pytest.mark.asyncio
    async def test_condition_time_before(self, consent_db):
        past = time.time() - 3600
        await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            conditions={"time_before": past},
            auto_approve=True,
        )
        has_consent, _ = await check_consent(USER_ID, AGENT_ID, "github")
        assert has_consent is False

    @pytest.mark.asyncio
    async def test_revoked_grant_not_matched(self, consent_db):
        grant = await create_consent_grant(
            USER_ID, AGENT_ID, "github", auto_approve=True,
        )
        await revoke_consent(grant.id, USER_ID)
        has_consent, _ = await check_consent(USER_ID, AGENT_ID, "github")
        assert has_consent is False

    @pytest.mark.asyncio
    async def test_most_specific_grant_wins(self, consent_db):
        # Create a blanket grant
        await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            scope_type="blanket",
            auto_approve=True,
        )
        # Create a resource-specific grant
        resource_grant = await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            scope_type="resource",
            resource_pattern="/repos/secure/*",
            auto_approve=True,
        )
        has_consent, grant = await check_consent(
            USER_ID, AGENT_ID, "github", resource="/repos/secure/myrepo",
        )
        assert has_consent is True
        assert grant.scope_type == "resource"


# ============================================================
# use_consent tests
# ============================================================

class TestUseConsent:
    @pytest.mark.asyncio
    async def test_basic_usage(self, consent_db):
        grant = await create_consent_grant(
            USER_ID, AGENT_ID, "github", auto_approve=True,
        )
        result = await use_consent(grant.id)
        assert result is True

        updated = await get_consent_grant(grant.id)
        assert updated.current_uses == 1

    @pytest.mark.asyncio
    async def test_multiple_uses(self, consent_db):
        grant = await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            auto_approve=True,
            max_uses=5,
        )
        for i in range(5):
            result = await use_consent(grant.id)
            assert result is True

        # 6th use should fail
        result = await use_consent(grant.id)
        assert result is False

    @pytest.mark.asyncio
    async def test_unlimited_uses(self, consent_db):
        grant = await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            auto_approve=True,
            max_uses=0,  # Unlimited
        )
        for _ in range(20):
            result = await use_consent(grant.id)
            assert result is True

    @pytest.mark.asyncio
    async def test_one_time_consent_expires_after_use(self, consent_db):
        grant = await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            scope_type="one_time",
            auto_approve=True,
        )
        result = await use_consent(grant.id)
        assert result is True

        # Second use should fail
        result = await use_consent(grant.id)
        assert result is False

        updated = await get_consent_grant(grant.id)
        assert updated.status == ConsentStatus.EXPIRED.value

    @pytest.mark.asyncio
    async def test_use_nonexistent_grant(self, consent_db):
        result = await use_consent("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_use_pending_grant(self, consent_db):
        grant = await create_consent_grant(
            USER_ID, AGENT_ID, "github",
        )
        result = await use_consent(grant.id)
        assert result is False

    @pytest.mark.asyncio
    async def test_max_uses_expiration(self, consent_db):
        grant = await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            auto_approve=True,
            max_uses=3,
        )
        for _ in range(3):
            await use_consent(grant.id)

        updated = await get_consent_grant(grant.id)
        assert updated.status == ConsentStatus.EXPIRED.value


# ============================================================
# get_consent_grant tests
# ============================================================

class TestGetConsentGrant:
    @pytest.mark.asyncio
    async def test_get_existing_grant(self, consent_db):
        created = await create_consent_grant(USER_ID, AGENT_ID, "github")
        fetched = await get_consent_grant(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.user_id == USER_ID

    @pytest.mark.asyncio
    async def test_get_nonexistent_grant(self, consent_db):
        fetched = await get_consent_grant("nonexistent")
        assert fetched is None

    @pytest.mark.asyncio
    async def test_get_preserves_all_fields(self, consent_db):
        created = await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            scope_type="action",
            action_pattern="read*",
            resource_pattern="/repos/*",
            conditions={"require_mfa": True},
            max_uses=10,
            expires_at=time.time() + 3600,
            metadata={"test": "value"},
        )
        fetched = await get_consent_grant(created.id)
        assert fetched.scope_type == "action"
        assert fetched.action_pattern == "read*"
        assert fetched.resource_pattern == "/repos/*"
        assert fetched.conditions == {"require_mfa": True}
        assert fetched.max_uses == 10
        assert fetched.metadata == {"test": "value"}


# ============================================================
# list_consent_grants tests
# ============================================================

class TestListConsentGrants:
    @pytest.mark.asyncio
    async def test_list_empty(self, consent_db):
        grants = await list_consent_grants(USER_ID)
        assert grants == []

    @pytest.mark.asyncio
    async def test_list_all_for_user(self, consent_db):
        await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        await create_consent_grant(USER_ID, AGENT_ID, "slack", auto_approve=True)
        grants = await list_consent_grants(USER_ID)
        assert len(grants) == 2

    @pytest.mark.asyncio
    async def test_list_filtered_by_agent(self, consent_db):
        await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        await create_consent_grant(USER_ID, AGENT_ID_2, "github", auto_approve=True)
        grants = await list_consent_grants(USER_ID, agent_id=AGENT_ID)
        assert len(grants) == 1
        assert grants[0].agent_id == AGENT_ID

    @pytest.mark.asyncio
    async def test_list_filtered_by_service(self, consent_db):
        await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        await create_consent_grant(USER_ID, AGENT_ID, "slack", auto_approve=True)
        grants = await list_consent_grants(USER_ID, service="github")
        assert len(grants) == 1
        assert grants[0].service == "github"

    @pytest.mark.asyncio
    async def test_list_filtered_by_status(self, consent_db):
        await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        await create_consent_grant(USER_ID, AGENT_ID, "slack")  # pending
        grants = await list_consent_grants(USER_ID, status="approved")
        assert len(grants) == 1
        assert grants[0].status == "approved"

    @pytest.mark.asyncio
    async def test_list_excludes_expired_by_default(self, consent_db):
        grant = await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            scope_type="one_time",
            auto_approve=True,
        )
        await use_consent(grant.id)
        grants = await list_consent_grants(USER_ID)
        assert len(grants) == 0

    @pytest.mark.asyncio
    async def test_list_includes_expired_when_requested(self, consent_db):
        grant = await create_consent_grant(
            USER_ID, AGENT_ID, "github",
            scope_type="one_time",
            auto_approve=True,
        )
        await use_consent(grant.id)
        grants = await list_consent_grants(USER_ID, include_expired=True)
        assert len(grants) == 1

    @pytest.mark.asyncio
    async def test_list_only_own_grants(self, consent_db):
        await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        await create_consent_grant(OTHER_USER, AGENT_ID, "github", auto_approve=True)
        grants = await list_consent_grants(USER_ID)
        assert len(grants) == 1

    @pytest.mark.asyncio
    async def test_list_order_by_created_desc(self, consent_db):
        g1 = await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        g2 = await create_consent_grant(USER_ID, AGENT_ID, "slack", auto_approve=True)
        grants = await list_consent_grants(USER_ID)
        # Most recent first
        assert grants[0].service == "slack"
        assert grants[1].service == "github"


# ============================================================
# Consent request tests
# ============================================================

class TestConsentRequests:
    @pytest.mark.asyncio
    async def test_create_consent_request(self, consent_db):
        req = await create_consent_request(
            user_id=USER_ID,
            agent_id=AGENT_ID,
            service="github",
            action="create_issue",
            reason="Need to file a bug report",
        )
        assert req.id
        assert req.user_id == USER_ID
        assert req.agent_id == AGENT_ID
        assert req.service == "github"
        assert req.action == "create_issue"
        assert req.status == "pending"

    @pytest.mark.asyncio
    async def test_create_request_with_urgency(self, consent_db):
        req = await create_consent_request(
            USER_ID, AGENT_ID, "github",
            urgency="critical",
            reason="Urgent security fix",
        )
        assert req.urgency == "critical"

    @pytest.mark.asyncio
    async def test_invalid_urgency(self, consent_db):
        with pytest.raises(ConsentError, match="Invalid urgency"):
            await create_consent_request(
                USER_ID, AGENT_ID, "github",
                urgency="mega",
            )

    @pytest.mark.asyncio
    async def test_create_request_missing_user(self, consent_db):
        with pytest.raises(ConsentError, match="user_id"):
            await create_consent_request("", AGENT_ID, "github")

    @pytest.mark.asyncio
    async def test_create_request_missing_agent(self, consent_db):
        with pytest.raises(ConsentError, match="agent_id"):
            await create_consent_request(USER_ID, "", "github")

    @pytest.mark.asyncio
    async def test_create_request_missing_service(self, consent_db):
        with pytest.raises(ConsentError, match="service"):
            await create_consent_request(USER_ID, AGENT_ID, "")

    @pytest.mark.asyncio
    async def test_create_request_with_resource(self, consent_db):
        req = await create_consent_request(
            USER_ID, AGENT_ID, "github",
            action="read",
            resource="/repos/org/private-repo",
        )
        assert req.resource == "/repos/org/private-repo"

    @pytest.mark.asyncio
    async def test_create_request_with_metadata(self, consent_db):
        req = await create_consent_request(
            USER_ID, AGENT_ID, "github",
            metadata={"context": "automated workflow"},
        )
        assert req.metadata == {"context": "automated workflow"}

    @pytest.mark.asyncio
    async def test_resolve_request_approved(self, consent_db):
        req = await create_consent_request(
            USER_ID, AGENT_ID, "github",
            action="read_issues",
        )
        resolved, grant = await resolve_consent_request(
            req.id, USER_ID, approved=True,
        )
        assert resolved is not None
        assert resolved.status == "approved"
        assert grant is not None
        assert grant.status == "approved"
        assert grant.agent_id == AGENT_ID

    @pytest.mark.asyncio
    async def test_resolve_request_denied(self, consent_db):
        req = await create_consent_request(
            USER_ID, AGENT_ID, "github",
        )
        resolved, grant = await resolve_consent_request(
            req.id, USER_ID, approved=False,
        )
        assert resolved is not None
        assert resolved.status == "denied"
        assert grant is None

    @pytest.mark.asyncio
    async def test_resolve_with_grant_options(self, consent_db):
        req = await create_consent_request(
            USER_ID, AGENT_ID, "github",
            action="read_issues",
        )
        resolved, grant = await resolve_consent_request(
            req.id, USER_ID, approved=True,
            grant_options={
                "scope_type": "action",
                "max_uses": 10,
                "expires_at": time.time() + 7200,
            },
        )
        assert grant.scope_type == "action"
        assert grant.max_uses == 10

    @pytest.mark.asyncio
    async def test_resolve_wrong_user(self, consent_db):
        req = await create_consent_request(USER_ID, AGENT_ID, "github")
        resolved, grant = await resolve_consent_request(
            req.id, OTHER_USER, approved=True,
        )
        assert resolved is None
        assert grant is None

    @pytest.mark.asyncio
    async def test_resolve_nonexistent(self, consent_db):
        resolved, grant = await resolve_consent_request(
            "nonexistent", USER_ID, approved=True,
        )
        assert resolved is None
        assert grant is None

    @pytest.mark.asyncio
    async def test_cannot_resolve_twice(self, consent_db):
        req = await create_consent_request(USER_ID, AGENT_ID, "github")
        await resolve_consent_request(req.id, USER_ID, approved=True)
        with pytest.raises(ConsentError, match="already resolved"):
            await resolve_consent_request(req.id, USER_ID, approved=True)

    @pytest.mark.asyncio
    async def test_list_consent_requests_empty(self, consent_db):
        requests = await list_consent_requests(USER_ID)
        assert requests == []

    @pytest.mark.asyncio
    async def test_list_consent_requests_all(self, consent_db):
        await create_consent_request(USER_ID, AGENT_ID, "github")
        await create_consent_request(USER_ID, AGENT_ID, "slack")
        requests = await list_consent_requests(USER_ID)
        assert len(requests) == 2

    @pytest.mark.asyncio
    async def test_list_consent_requests_by_status(self, consent_db):
        req1 = await create_consent_request(USER_ID, AGENT_ID, "github")
        await create_consent_request(USER_ID, AGENT_ID, "slack")
        await resolve_consent_request(req1.id, USER_ID, approved=True)

        pending = await list_consent_requests(USER_ID, status="pending")
        assert len(pending) == 1
        assert pending[0].service == "slack"

    @pytest.mark.asyncio
    async def test_list_consent_requests_by_agent(self, consent_db):
        await create_consent_request(USER_ID, AGENT_ID, "github")
        await create_consent_request(USER_ID, AGENT_ID_2, "slack")

        reqs = await list_consent_requests(USER_ID, agent_id=AGENT_ID)
        assert len(reqs) == 1
        assert reqs[0].agent_id == AGENT_ID


# ============================================================
# Bulk operations tests
# ============================================================

class TestBulkOperations:
    @pytest.mark.asyncio
    async def test_revoke_all_agent_consent(self, consent_db):
        await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        await create_consent_grant(USER_ID, AGENT_ID, "slack", auto_approve=True)
        await create_consent_grant(USER_ID, AGENT_ID, "google")  # pending

        count = await revoke_all_agent_consent(USER_ID, AGENT_ID)
        assert count == 3

        grants = await list_consent_grants(USER_ID, include_expired=True)
        for g in grants:
            assert g.status == "revoked"

    @pytest.mark.asyncio
    async def test_revoke_all_only_affects_specified_agent(self, consent_db):
        await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        await create_consent_grant(USER_ID, AGENT_ID_2, "github", auto_approve=True)

        count = await revoke_all_agent_consent(USER_ID, AGENT_ID)
        assert count == 1

        # Agent 2 should be unaffected
        grants = await list_consent_grants(USER_ID, agent_id=AGENT_ID_2)
        assert len(grants) == 1
        assert grants[0].status == "approved"

    @pytest.mark.asyncio
    async def test_revoke_all_returns_zero_when_none(self, consent_db):
        count = await revoke_all_agent_consent(USER_ID, AGENT_ID)
        assert count == 0

    @pytest.mark.asyncio
    async def test_revoke_all_skips_already_revoked(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        await revoke_consent(grant.id, USER_ID)

        count = await revoke_all_agent_consent(USER_ID, AGENT_ID)
        assert count == 0


# ============================================================
# Consent summary tests
# ============================================================

class TestConsentSummary:
    @pytest.mark.asyncio
    async def test_empty_summary(self, consent_db):
        summary = await get_consent_summary(USER_ID)
        assert summary["total_grants"] == 0
        assert summary["active_grants"] == 0
        assert summary["pending_grants"] == 0

    @pytest.mark.asyncio
    async def test_summary_with_grants(self, consent_db):
        await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        await create_consent_grant(USER_ID, AGENT_ID, "slack", auto_approve=True)
        await create_consent_grant(USER_ID, AGENT_ID_2, "github")  # pending

        summary = await get_consent_summary(USER_ID)
        assert summary["total_grants"] == 3
        assert summary["active_grants"] == 2
        assert summary["pending_grants"] == 1
        assert AGENT_ID in summary["agents_with_consent"]
        assert "github" in summary["services_with_consent"]

    @pytest.mark.asyncio
    async def test_summary_pending_requests(self, consent_db):
        await create_consent_request(USER_ID, AGENT_ID, "github")
        await create_consent_request(USER_ID, AGENT_ID, "slack")

        summary = await get_consent_summary(USER_ID)
        assert summary["pending_requests"] == 2


# ============================================================
# Consent audit log tests
# ============================================================

class TestConsentAuditLog:
    @pytest.mark.asyncio
    async def test_audit_log_on_create(self, consent_db):
        await create_consent_grant(USER_ID, AGENT_ID, "github")
        log = await get_consent_audit_log(USER_ID)
        assert len(log) >= 1
        assert any(e.action == "grant_created" for e in log)

    @pytest.mark.asyncio
    async def test_audit_log_on_approve(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github")
        await approve_consent(grant.id, USER_ID)
        log = await get_consent_audit_log(USER_ID)
        assert any(e.action == "grant_approved" for e in log)

    @pytest.mark.asyncio
    async def test_audit_log_on_deny(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github")
        await deny_consent(grant.id, USER_ID, reason="Not needed")
        log = await get_consent_audit_log(USER_ID)
        assert any(e.action == "grant_denied" for e in log)

    @pytest.mark.asyncio
    async def test_audit_log_on_revoke(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        await revoke_consent(grant.id, USER_ID, reason="Compromised")
        log = await get_consent_audit_log(USER_ID)
        assert any(e.action == "grant_revoked" for e in log)

    @pytest.mark.asyncio
    async def test_audit_log_on_use(self, consent_db):
        grant = await create_consent_grant(USER_ID, AGENT_ID, "github", auto_approve=True)
        await use_consent(grant.id)
        log = await get_consent_audit_log(USER_ID)
        assert any(e.action == "consent_used" for e in log)

    @pytest.mark.asyncio
    async def test_audit_log_on_request(self, consent_db):
        await create_consent_request(USER_ID, AGENT_ID, "github")
        log = await get_consent_audit_log(USER_ID)
        assert any(e.action == "consent_requested" for e in log)

    @pytest.mark.asyncio
    async def test_audit_log_limit(self, consent_db):
        for _ in range(10):
            await create_consent_grant(USER_ID, AGENT_ID, "github")
        log = await get_consent_audit_log(USER_ID, limit=5)
        assert len(log) == 5

    @pytest.mark.asyncio
    async def test_audit_log_only_own_entries(self, consent_db):
        await create_consent_grant(USER_ID, AGENT_ID, "github")
        await create_consent_grant(OTHER_USER, AGENT_ID, "github")
        log = await get_consent_audit_log(USER_ID)
        assert all(e.user_id == USER_ID for e in log)


# ============================================================
# Pattern matching tests
# ============================================================

class TestPatternMatching:
    def test_wildcard_matches_everything(self):
        assert _pattern_matches("*", "anything") is True
        assert _pattern_matches("*", "") is True

    def test_exact_match(self):
        assert _pattern_matches("read", "read") is True
        assert _pattern_matches("read", "write") is False

    def test_prefix_wildcard(self):
        assert _pattern_matches("read*", "read_issues") is True
        assert _pattern_matches("read*", "read") is True
        assert _pattern_matches("read*", "write_issues") is False

    def test_suffix_wildcard(self):
        assert _pattern_matches("*_issues", "read_issues") is True
        assert _pattern_matches("*_issues", "write_issues") is True
        assert _pattern_matches("*_issues", "read_repos") is False

    def test_contains_wildcard(self):
        assert _pattern_matches("*issue*", "read_issues_list") is True
        assert _pattern_matches("*issue*", "no_match") is False

    def test_comma_separated_alternatives(self):
        assert _pattern_matches("read,write", "read") is True
        assert _pattern_matches("read,write", "write") is True
        assert _pattern_matches("read,write", "delete") is False

    def test_comma_with_wildcards(self):
        assert _pattern_matches("read*, write*", "read_issues") is True
        assert _pattern_matches("read*, write*", "write_issues") is True
        assert _pattern_matches("read*, write*", "delete_issues") is False

    def test_empty_pattern_exact(self):
        assert _pattern_matches("", "") is True
        assert _pattern_matches("", "something") is False

    def test_single_char_prefix(self):
        assert _pattern_matches("r*", "read") is True
        assert _pattern_matches("r*", "write") is False

    def test_path_patterns(self):
        assert _pattern_matches("/repos/*", "/repos/myorg/myrepo") is True
        assert _pattern_matches("/repos/*", "/users/me") is False

    def test_multiple_alternatives_with_spaces(self):
        assert _pattern_matches("a, b, c", "b") is True
        assert _pattern_matches("a, b, c", "d") is False


# ============================================================
# Condition checking tests
# ============================================================

class TestCheckConditions:
    def test_empty_conditions(self):
        assert _check_conditions({}, time.time()) is True

    def test_time_after_in_future(self):
        future = time.time() + 3600
        assert _check_conditions({"time_after": future}, time.time()) is False

    def test_time_after_in_past(self):
        past = time.time() - 3600
        assert _check_conditions({"time_after": past}, time.time()) is True

    def test_time_before_in_future(self):
        future = time.time() + 3600
        assert _check_conditions({"time_before": future}, time.time()) is True

    def test_time_before_in_past(self):
        past = time.time() - 3600
        assert _check_conditions({"time_before": past}, time.time()) is False

    def test_time_window(self):
        now = time.time()
        conditions = {
            "time_after": now - 100,
            "time_before": now + 100,
        }
        assert _check_conditions(conditions, now) is True

    def test_outside_time_window(self):
        now = time.time()
        conditions = {
            "time_after": now + 100,
            "time_before": now + 200,
        }
        assert _check_conditions(conditions, now) is False

    def test_allowed_hours(self):
        from datetime import datetime, timezone
        # Create a timestamp at hour 10 UTC
        dt = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts = dt.timestamp()
        assert _check_conditions({"allowed_hours": [9, 10, 11]}, ts) is True
        assert _check_conditions({"allowed_hours": [14, 15, 16]}, ts) is False

    def test_allowed_days(self):
        from datetime import datetime, timezone
        # 2026-03-15 is a Sunday (weekday 6)
        dt = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts = dt.timestamp()
        assert _check_conditions({"allowed_days": [5, 6]}, ts) is True  # Sat, Sun
        assert _check_conditions({"allowed_days": [0, 1, 2, 3, 4]}, ts) is False  # Weekdays


# ============================================================
# ConsentScope and ConsentStatus enum tests
# ============================================================

class TestEnums:
    def test_consent_status_values(self):
        assert ConsentStatus.PENDING.value == "pending"
        assert ConsentStatus.APPROVED.value == "approved"
        assert ConsentStatus.DENIED.value == "denied"
        assert ConsentStatus.REVOKED.value == "revoked"
        assert ConsentStatus.EXPIRED.value == "expired"

    def test_consent_scope_values(self):
        assert ConsentScope.BLANKET.value == "blanket"
        assert ConsentScope.SERVICE.value == "service"
        assert ConsentScope.ACTION.value == "action"
        assert ConsentScope.RESOURCE.value == "resource"
        assert ConsentScope.ONE_TIME.value == "one_time"

    def test_all_scope_types_valid(self):
        valid = {s.value for s in ConsentScope}
        assert len(valid) == 5


# ============================================================
# Dataclass tests
# ============================================================

class TestDataclasses:
    def test_consent_grant_defaults(self):
        g = ConsentGrant()
        assert g.id == ""
        assert g.status == "pending"
        assert g.max_uses == 0
        assert g.current_uses == 0
        assert g.conditions == {}
        assert g.metadata == {}

    def test_consent_request_defaults(self):
        r = ConsentRequest()
        assert r.id == ""
        assert r.urgency == "normal"
        assert r.status == "pending"
        assert r.grant_id == ""
        assert r.metadata == {}

    def test_consent_audit_entry_defaults(self):
        e = ConsentAuditEntry()
        assert e.id == 0
        assert e.timestamp == 0.0
        assert e.details == ""
