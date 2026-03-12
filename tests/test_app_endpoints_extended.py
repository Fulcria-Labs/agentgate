"""Extended API endpoint tests — logout flow, dashboard variants, service
connection with mocked initiate, audit page, policy listing field coverage,
token endpoint with IP address, and edge cases in request validation."""

import asyncio
import time
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.auth import SUPPORTED_SERVICES
from src.database import (
    AgentPolicy,
    add_connected_service,
    create_agent_policy,
    create_api_key,
    get_audit_log,
    log_audit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(db, monkeypatch):
    import os
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(project_dir)
    return TestClient(app)


@pytest.fixture
def auth_client(client):
    with client:
        client.cookies.set("session", "test")
        with patch("src.app.get_user", return_value={
            "sub": "auth0|user123",
            "name": "Test User",
            "email": "test@example.com",
            "picture": "",
        }):
            yield client


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _create_policy(agent_id="bot-1", user_id="auth0|user123", services=None,
                   scopes=None, step_up=None, rate_limit=60, **kwargs):
    services = services or ["github"]
    scopes = scopes or {"github": ["repo"]}
    step_up = step_up or []
    policy = AgentPolicy(
        agent_id=agent_id,
        agent_name=f"Agent {agent_id}",
        allowed_services=services,
        allowed_scopes=scopes,
        rate_limit_per_minute=rate_limit,
        requires_step_up=step_up,
        created_by=user_id,
        created_at=0,
        **kwargs,
    )
    _run(create_agent_policy(policy))
    return policy


# ---------------------------------------------------------------------------
# 1. Logout endpoint
# ---------------------------------------------------------------------------

class TestLogoutEndpoint:
    """Tests for GET /logout."""

    def test_logout_redirects(self, client):
        """Logout returns a redirect response."""
        response = client.get("/logout", follow_redirects=False)
        assert response.status_code in (302, 307)

    def test_logout_redirects_to_auth0(self, client):
        """Logout redirect URL points to Auth0 logout."""
        response = client.get("/logout", follow_redirects=False)
        location = response.headers.get("location", "")
        assert "v2/logout" in location

    def test_logout_includes_client_id(self, client):
        """Logout URL includes the client_id parameter."""
        response = client.get("/logout", follow_redirects=False)
        location = response.headers.get("location", "")
        assert "client_id=" in location


# ---------------------------------------------------------------------------
# 2. Dashboard variants
# ---------------------------------------------------------------------------

class TestDashboardVariants:
    """Dashboard page rendering for different states."""

    def test_dashboard_unauthenticated_returns_200(self, client):
        """Unauthenticated dashboard returns 200 (login page)."""
        response = client.get("/")
        assert response.status_code == 200

    def test_dashboard_authenticated_with_services(self, auth_client, db):
        """Dashboard renders with connected services."""
        _run(add_connected_service("auth0|user123", "github", "conn-1"))
        response = auth_client.get("/")
        assert response.status_code == 200

    def test_dashboard_authenticated_with_policies(self, auth_client, db):
        """Dashboard renders with agent policies."""
        _create_policy(agent_id="dash-bot")
        response = auth_client.get("/")
        assert response.status_code == 200

    def test_dashboard_authenticated_with_audit_entries(self, auth_client, db):
        """Dashboard renders with audit entries."""
        _run(log_audit("auth0|user123", "agent-1", "github", "token_issued", "success"))
        response = auth_client.get("/")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 3. Audit page
# ---------------------------------------------------------------------------

class TestAuditPage:
    """Tests for GET /audit."""

    def test_audit_page_requires_auth(self, client):
        response = client.get("/audit")
        assert response.status_code == 401

    def test_audit_page_authenticated(self, auth_client, db):
        _run(log_audit("auth0|user123", "agent-1", "github", "token_issued", "success"))
        response = auth_client.get("/audit")
        assert response.status_code == 200

    def test_audit_page_empty_log(self, auth_client, db):
        """Audit page renders fine with no entries."""
        response = auth_client.get("/audit")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 4. Connect service with valid service
# ---------------------------------------------------------------------------

class TestConnectServiceFlow:
    """Tests for POST /connect/{service} with mocked initiate_connection."""

    def test_connect_github_redirects(self, auth_client):
        """Connecting github initiates the OAuth flow."""
        with patch("src.app.initiate_connection", new_callable=AsyncMock,
                   return_value="https://auth0.example.com/authorize?connection=github"):
            response = auth_client.post("/connect/github", follow_redirects=False)
        assert response.status_code == 303

    def test_connect_slack_redirects(self, auth_client):
        with patch("src.app.initiate_connection", new_callable=AsyncMock,
                   return_value="https://auth0.example.com/authorize?connection=slack"):
            response = auth_client.post("/connect/slack", follow_redirects=False)
        assert response.status_code == 303

    def test_connect_google_redirects(self, auth_client):
        with patch("src.app.initiate_connection", new_callable=AsyncMock,
                   return_value="https://auth0.example.com/authorize?connection=google"):
            response = auth_client.post("/connect/google", follow_redirects=False)
        assert response.status_code == 303

    def test_connect_linear_redirects(self, auth_client):
        with patch("src.app.initiate_connection", new_callable=AsyncMock,
                   return_value="https://auth0.example.com/authorize?connection=linear"):
            response = auth_client.post("/connect/linear", follow_redirects=False)
        assert response.status_code == 303

    def test_connect_notion_redirects(self, auth_client):
        with patch("src.app.initiate_connection", new_callable=AsyncMock,
                   return_value="https://auth0.example.com/authorize?connection=notion"):
            response = auth_client.post("/connect/notion", follow_redirects=False)
        assert response.status_code == 303


# ---------------------------------------------------------------------------
# 5. Policy listing field coverage
# ---------------------------------------------------------------------------

class TestPolicyListFieldCoverage:
    """Validate all fields in GET /api/v1/policies responses."""

    def test_policy_with_expiry_shows_expires_at(self, auth_client, db):
        future = time.time() + 86400
        _create_policy(agent_id="exp-bot", expires_at=future)
        response = auth_client.get("/api/v1/policies")
        policies = response.json()
        assert len(policies) == 1
        assert policies[0]["expires_at"] == future

    def test_policy_without_expiry_shows_none(self, auth_client, db):
        _create_policy(agent_id="noexp-bot", expires_at=0.0)
        response = auth_client.get("/api/v1/policies")
        policies = response.json()
        assert len(policies) == 1
        assert policies[0]["expires_at"] is None

    def test_policy_with_time_windows(self, auth_client, db):
        _create_policy(
            agent_id="tw-bot",
            allowed_hours=[9, 10, 11],
            allowed_days=[0, 1, 2],
        )
        response = auth_client.get("/api/v1/policies")
        policies = response.json()
        assert len(policies) == 1
        assert policies[0]["allowed_hours"] == [9, 10, 11]
        assert policies[0]["allowed_days"] == [0, 1, 2]

    def test_policy_with_empty_time_windows(self, auth_client, db):
        _create_policy(agent_id="no-tw-bot")
        response = auth_client.get("/api/v1/policies")
        policies = response.json()
        assert len(policies) == 1
        assert policies[0]["allowed_hours"] == []
        assert policies[0]["allowed_days"] == []

    def test_policy_with_ip_allowlist(self, auth_client, db):
        _create_policy(agent_id="ip-list-bot", ip_allowlist=["10.0.0.0/8", "192.168.1.1"])
        response = auth_client.get("/api/v1/policies")
        policies = response.json()
        assert len(policies) == 1
        assert policies[0]["ip_allowlist"] == ["10.0.0.0/8", "192.168.1.1"]


# ---------------------------------------------------------------------------
# 6. Token endpoint edge cases
# ---------------------------------------------------------------------------

class TestTokenEndpointEdgeCases:
    """Edge cases for the token endpoint."""

    def test_token_with_empty_scopes_list(self, auth_client, db):
        """Token request with empty scopes succeeds."""
        _create_policy(agent_id="empty-scope-bot")
        with patch("src.app.get_token_vault_token", new_callable=AsyncMock,
                   return_value={
                       "access_token": "tok",
                       "token_type": "Bearer",
                       "expires_in": 3600,
                       "scope": "",
                   }):
            response = auth_client.post("/api/v1/token", json={
                "agent_id": "empty-scope-bot",
                "service": "github",
                "scopes": [],
            })
        assert response.status_code == 200

    def test_token_for_nonexistent_agent(self, auth_client, db):
        """Token for agent that doesn't exist."""
        response = auth_client.post("/api/v1/token", json={
            "agent_id": "ghost-agent",
            "service": "github",
            "scopes": ["repo"],
        })
        assert response.status_code == 403
        assert "not registered" in response.json()["detail"]

    def test_token_for_expired_policy(self, auth_client, db):
        """Token for an expired policy is denied."""
        _create_policy(agent_id="exp-pol-bot", expires_at=time.time() - 3600)
        response = auth_client.post("/api/v1/token", json={
            "agent_id": "exp-pol-bot",
            "service": "github",
            "scopes": ["repo"],
        })
        assert response.status_code == 403
        assert "expired" in response.json()["detail"]

    def test_token_with_ip_denied(self, auth_client, db):
        """Token request from unauthorized IP is denied."""
        _create_policy(agent_id="ip-deny-bot", ip_allowlist=["10.0.0.1"])
        response = auth_client.post("/api/v1/token", json={
            "agent_id": "ip-deny-bot",
            "service": "github",
            "scopes": ["repo"],
        })
        # TestClient uses 'testclient' host which won't match 10.0.0.1
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# 7. Service listing edge cases
# ---------------------------------------------------------------------------

class TestServiceListingEdgeCases:
    """Edge cases for GET /api/v1/services."""

    def test_service_listing_with_step_up(self, auth_client, db):
        """Services that require step-up show requires_step_up=true."""
        _create_policy(
            agent_id="su-svc-bot",
            services=["github", "slack"],
            scopes={"github": ["repo"], "slack": ["chat:write"]},
            step_up=["slack"],
        )
        response = auth_client.get("/api/v1/services?agent_id=su-svc-bot")
        assert response.status_code == 200
        services = response.json()
        github_svc = [s for s in services if s["service"] == "github"][0]
        slack_svc = [s for s in services if s["service"] == "slack"][0]
        assert github_svc["requires_step_up"] is False
        assert slack_svc["requires_step_up"] is True

    def test_service_listing_shows_allowed_scopes(self, auth_client, db):
        """Service listing includes the allowed scopes per service."""
        _create_policy(
            agent_id="scope-svc-bot",
            services=["github"],
            scopes={"github": ["repo", "read:user"]},
        )
        response = auth_client.get("/api/v1/services?agent_id=scope-svc-bot")
        services = response.json()
        assert len(services) == 1
        assert set(services[0]["allowed_scopes"]) == {"repo", "read:user"}

    def test_service_listing_with_multiple_connected(self, auth_client, db):
        """Multiple connected services show correctly."""
        _create_policy(
            agent_id="multi-conn-bot",
            services=["github", "slack"],
            scopes={"github": ["repo"], "slack": ["chat:write"]},
        )
        _run(add_connected_service("auth0|user123", "github", "conn-1"))
        _run(add_connected_service("auth0|user123", "slack", "conn-2"))

        response = auth_client.get("/api/v1/services?agent_id=multi-conn-bot")
        services = response.json()
        assert len(services) == 2
        assert all(s["connected"] for s in services)


# ---------------------------------------------------------------------------
# 8. Toggle and delete endpoint additional tests
# ---------------------------------------------------------------------------

class TestToggleDeleteAdditional:
    """Additional toggle and delete endpoint tests."""

    def test_toggle_then_token_denied(self, auth_client, db):
        """After toggling a policy off, token requests are denied."""
        _create_policy(agent_id="tog-tok-bot")
        auth_client.post("/api/v1/policies/tog-tok-bot/toggle")
        response = auth_client.post("/api/v1/token", json={
            "agent_id": "tog-tok-bot",
            "service": "github",
            "scopes": ["repo"],
        })
        assert response.status_code == 403

    def test_toggle_back_on_allows_token(self, auth_client, db):
        """After toggling off and back on, token requests succeed."""
        _create_policy(agent_id="tog-back-bot")
        auth_client.post("/api/v1/policies/tog-back-bot/toggle")  # off
        auth_client.post("/api/v1/policies/tog-back-bot/toggle")  # on

        with patch("src.app.get_token_vault_token", new_callable=AsyncMock,
                   return_value={
                       "access_token": "tok",
                       "token_type": "Bearer",
                       "expires_in": 3600,
                       "scope": "repo",
                   }):
            response = auth_client.post("/api/v1/token", json={
                "agent_id": "tog-back-bot",
                "service": "github",
                "scopes": ["repo"],
            })
        assert response.status_code == 200

    def test_delete_policy_then_toggle_returns_404(self, auth_client, db):
        """After deleting a policy, toggle returns 404."""
        _create_policy(agent_id="del-tog-bot")
        auth_client.delete("/api/v1/policies/del-tog-bot")
        response = auth_client.post("/api/v1/policies/del-tog-bot/toggle")
        assert response.status_code == 404

    def test_delete_policy_then_token_returns_403(self, auth_client, db):
        """After deleting a policy, token request returns 403."""
        _create_policy(agent_id="del-tok-bot")
        auth_client.delete("/api/v1/policies/del-tok-bot")
        response = auth_client.post("/api/v1/token", json={
            "agent_id": "del-tok-bot",
            "service": "github",
            "scopes": ["repo"],
        })
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# 9. API key creation and listing edge cases
# ---------------------------------------------------------------------------

class TestApiKeyEdgeCasesEndpoint:
    """Edge cases for API key management endpoints."""

    def test_create_key_default_name(self, auth_client):
        """Key created without name gets 'default'."""
        response = auth_client.post("/api/v1/keys", json={
            "agent_id": "default-name-bot",
        })
        assert response.status_code == 200
        # The default name is "default" per the Pydantic model

    def test_list_keys_empty(self, auth_client, db):
        """List keys returns empty list when no keys exist."""
        response = auth_client.get("/api/v1/keys")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_keys_shows_created_at(self, auth_client, db):
        """List keys includes created_at timestamp."""
        auth_client.post("/api/v1/keys", json={"agent_id": "ts-bot"})
        response = auth_client.get("/api/v1/keys")
        keys = response.json()
        assert len(keys) >= 1
        assert "created_at" in keys[0]
        assert keys[0]["created_at"] > 0

    def test_list_keys_shows_last_used_at(self, auth_client, db):
        """List keys includes last_used_at."""
        auth_client.post("/api/v1/keys", json={"agent_id": "lu-bot"})
        response = auth_client.get("/api/v1/keys")
        keys = response.json()
        assert "last_used_at" in keys[0]
        assert keys[0]["last_used_at"] is None  # Never used


# ---------------------------------------------------------------------------
# 10. Emergency revoke audit logging
# ---------------------------------------------------------------------------

class TestEmergencyRevokeAudit:
    """Verify emergency revoke creates audit entries."""

    def test_emergency_revoke_creates_audit_entry(self, auth_client, db):
        """Emergency revoke action is logged in the audit trail."""
        _create_policy(agent_id="em-audit-bot")
        auth_client.post("/api/v1/emergency-revoke")
        entries = _run(get_audit_log("auth0|user123"))
        emergency_entries = [e for e in entries if e.action == "emergency_revoke"]
        assert len(emergency_entries) >= 1
        assert emergency_entries[0].status == "success"

    def test_emergency_revoke_audit_includes_counts(self, auth_client, db):
        """Emergency revoke audit details include counts."""
        _create_policy(agent_id="em-count-bot")
        _run(create_api_key("auth0|user123", "em-count-bot", "key"))
        auth_client.post("/api/v1/emergency-revoke")
        entries = _run(get_audit_log("auth0|user123"))
        emergency_entries = [e for e in entries if e.action == "emergency_revoke"]
        assert len(emergency_entries) >= 1
        assert "1 policies" in emergency_entries[0].details
        assert "1 API keys" in emergency_entries[0].details
