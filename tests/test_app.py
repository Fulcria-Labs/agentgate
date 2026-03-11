"""Tests for the FastAPI application endpoints."""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from src.app import app
from src.database import (
    init_db, create_agent_policy, create_api_key, AgentPolicy, add_connected_service,
)


@pytest.fixture
def client(db, monkeypatch):
    """Create a test client with a fresh database."""
    # Patch static/template dirs to use project paths
    import os
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(project_dir)
    return TestClient(app)


class TestHealth:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "agentgate"


class TestUnauthenticated:
    def test_dashboard_shows_login(self, client):
        """Unauthenticated users see login page."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 200

    def test_connect_requires_auth(self, client):
        """Service connection requires authentication."""
        response = client.post("/connect/github", follow_redirects=False)
        assert response.status_code == 401

    def test_audit_requires_auth(self, client):
        """Audit page requires authentication."""
        response = client.get("/audit")
        assert response.status_code == 401

    def test_token_api_requires_auth(self, client):
        """Token API requires authentication."""
        response = client.post("/api/v1/token", json={
            "agent_id": "test", "service": "github", "scopes": ["repo"]
        })
        assert response.status_code == 401

    def test_policies_api_requires_auth(self, client):
        """Policies API requires authentication."""
        response = client.get("/api/v1/policies")
        assert response.status_code == 401


class TestAuthenticated:
    """Tests with a mocked authenticated session."""

    @pytest.fixture
    def auth_client(self, client):
        """Client with an authenticated session."""
        # Inject session data directly
        with client:
            client.cookies.set("session", "test")
            # Patch the session middleware to return a user
            with patch("src.app.get_user", return_value={
                "sub": "auth0|user123",
                "name": "Test User",
                "email": "test@example.com",
                "picture": "",
            }):
                yield client

    def test_dashboard_authenticated(self, auth_client):
        """Authenticated users see the dashboard."""
        response = auth_client.get("/")
        assert response.status_code == 200

    def test_create_policy(self, auth_client):
        """Create an agent policy."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "bot-1",
            "agent_name": "Test Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "rate_limit_per_minute": 30,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert data["agent_id"] == "bot-1"

    def test_create_policy_invalid_service(self, auth_client):
        """Creating a policy with an invalid service fails."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "bot-2",
            "agent_name": "Bad Bot",
            "allowed_services": ["fakebook"],
            "allowed_scopes": {},
        })
        assert response.status_code == 400

    def test_create_policy_invalid_scopes(self, auth_client):
        """Creating a policy with invalid scopes fails."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "bot-3",
            "agent_name": "Bad Scope Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo", "fake_scope"]},
        })
        assert response.status_code == 400

    def test_list_policies(self, auth_client):
        """List policies returns empty list initially."""
        response = auth_client.get("/api/v1/policies")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_token_request_no_policy(self, auth_client):
        """Token request for unregistered agent fails."""
        response = auth_client.post("/api/v1/token", json={
            "agent_id": "unknown-agent",
            "service": "github",
            "scopes": ["repo"],
        })
        assert response.status_code == 403

    def test_connect_unknown_service(self, auth_client):
        """Connecting an unknown service fails."""
        response = auth_client.post("/connect/fakebook", follow_redirects=False)
        assert response.status_code == 400

    def test_disconnect_service(self, auth_client):
        """Disconnecting a service returns success."""
        response = auth_client.delete("/connect/github")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "disconnected"

    def test_list_agent_services_not_found(self, auth_client):
        """Listing services for unknown agent returns 404."""
        response = auth_client.get("/api/v1/services?agent_id=unknown")
        assert response.status_code == 404

    def test_create_api_key(self, auth_client):
        """Create an API key for an agent."""
        response = auth_client.post("/api/v1/keys", json={
            "agent_id": "bot-1",
            "name": "test-key",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["key"].startswith("ag_")
        assert data["agent_id"] == "bot-1"
        assert "warning" in data

    def test_list_api_keys(self, auth_client):
        """List API keys returns metadata without raw keys."""
        auth_client.post("/api/v1/keys", json={"agent_id": "bot-1", "name": "k1"})
        response = auth_client.get("/api/v1/keys")
        assert response.status_code == 200
        keys = response.json()
        assert len(keys) >= 1
        assert "key" not in keys[0]  # Raw key never in list response

    def test_revoke_api_key(self, auth_client):
        """Revoke an API key."""
        create_resp = auth_client.post("/api/v1/keys", json={"agent_id": "bot-1"})
        key_id = create_resp.json()["key_id"]
        response = auth_client.delete(f"/api/v1/keys/{key_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "revoked"

    def test_revoke_nonexistent_key(self, auth_client):
        """Revoking non-existent key returns 404."""
        response = auth_client.delete("/api/v1/keys/fake-id")
        assert response.status_code == 404

    def test_toggle_policy(self, auth_client):
        """Toggle agent policy active state."""
        # Create policy first
        auth_client.post("/api/v1/policies", json={
            "agent_id": "toggle-bot",
            "agent_name": "Toggle Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        # Toggle to disabled
        response = auth_client.post("/api/v1/policies/toggle-bot/toggle")
        assert response.status_code == 200
        assert response.json()["is_active"] is False
        assert response.json()["status"] == "disabled"

    def test_toggle_nonexistent_policy(self, auth_client):
        """Toggle non-existent policy returns 404."""
        response = auth_client.post("/api/v1/policies/fake-bot/toggle")
        assert response.status_code == 404


class TestApiKeyAuth:
    """Tests for API key authentication on the token endpoint."""

    @pytest.fixture
    def client(self, db, monkeypatch):
        import os
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        monkeypatch.chdir(project_dir)
        return TestClient(app)

    def test_token_with_api_key(self, client, db):
        """Agents can authenticate with API key Bearer token."""
        import asyncio
        loop = asyncio.new_event_loop()
        # Create policy and key
        policy = AgentPolicy(
            agent_id="api-bot", agent_name="API Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            created_by="auth0|user123", created_at=0,
        )
        loop.run_until_complete(create_agent_policy(policy))
        _, raw_key = loop.run_until_complete(
            create_api_key("auth0|user123", "api-bot", "test")
        )
        loop.close()

        # Use API key to request token (mock the token vault call)
        with patch("src.app.get_token_vault_token", new_callable=AsyncMock,
                   return_value={
                       "access_token": "gh_token_123",
                       "token_type": "Bearer",
                       "expires_in": 3600,
                       "scope": "repo",
                   }):
            response = client.post(
                "/api/v1/token",
                json={"agent_id": "api-bot", "service": "github", "scopes": ["repo"]},
                headers={"Authorization": f"Bearer {raw_key}"},
            )
        assert response.status_code == 200
        assert response.json()["access_token"] == "gh_token_123"

    def test_token_with_wrong_agent_id(self, client, db):
        """API key bound to one agent cannot request for another."""
        import asyncio
        loop = asyncio.new_event_loop()
        policy = AgentPolicy(
            agent_id="api-bot", agent_name="API Bot",
            allowed_services=["github"], allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            created_by="auth0|user123", created_at=0,
        )
        loop.run_until_complete(create_agent_policy(policy))
        _, raw_key = loop.run_until_complete(
            create_api_key("auth0|user123", "api-bot", "test")
        )
        loop.close()

        response = client.post(
            "/api/v1/token",
            json={"agent_id": "other-bot", "service": "github", "scopes": ["repo"]},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert response.status_code == 403
        assert "bound to agent" in response.json()["detail"]

    def test_token_with_invalid_key(self, client, db):
        """Invalid API key returns 401."""
        response = client.post(
            "/api/v1/token",
            json={"agent_id": "bot", "service": "github", "scopes": ["repo"]},
            headers={"Authorization": "Bearer ag_invalid_key_value"},
        )
        assert response.status_code == 401
