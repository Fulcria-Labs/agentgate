"""Tests for auth module and app endpoint edge cases."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from src.app import app
from src.auth import SUPPORTED_SERVICES
from src.database import (
    init_db,
    create_agent_policy,
    create_api_key,
    add_connected_service,
    AgentPolicy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(db, monkeypatch):
    """Create a test client with a fresh database."""
    import os
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(project_dir)
    return TestClient(app)


@pytest.fixture
def auth_client(client):
    """Client with an authenticated session (user auth0|user123)."""
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
    """Run an async coroutine in a fresh event loop (for sync test setup)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _create_policy(agent_id="bot-1", user_id="auth0|user123", services=None,
                   scopes=None, step_up=None):
    """Helper to synchronously insert a policy into the test DB."""
    services = services or ["github"]
    scopes = scopes or {"github": ["repo"]}
    step_up = step_up or []
    policy = AgentPolicy(
        agent_id=agent_id,
        agent_name=f"Agent {agent_id}",
        allowed_services=services,
        allowed_scopes=scopes,
        rate_limit_per_minute=60,
        requires_step_up=step_up,
        created_by=user_id,
        created_at=0,
    )
    _run(create_agent_policy(policy))
    return policy


# ===========================================================================
# 1. SUPPORTED_SERVICES validation
# ===========================================================================

class TestSupportedServices:
    """Validate the SUPPORTED_SERVICES registry in src/auth.py."""

    def test_all_services_have_required_keys(self):
        """Every service entry must have display_name, icon, scopes, description."""
        required = {"display_name", "icon", "scopes", "description"}
        for name, info in SUPPORTED_SERVICES.items():
            missing = required - set(info.keys())
            assert not missing, f"Service '{name}' missing keys: {missing}"

    def test_github_has_expected_scopes(self):
        """GitHub must include the core scopes we advertise."""
        github_scopes = set(SUPPORTED_SERVICES["github"]["scopes"])
        expected = {"repo", "read:user", "read:org", "gist", "notifications"}
        assert expected.issubset(github_scopes)

    def test_each_service_has_at_least_one_scope(self):
        """Each service must expose at least one OAuth scope."""
        for name, info in SUPPORTED_SERVICES.items():
            assert len(info["scopes"]) >= 1, f"Service '{name}' has no scopes"

    def test_no_duplicate_service_names(self):
        """Service keys must be unique (dict guarantees this, but verify count)."""
        names = list(SUPPORTED_SERVICES.keys())
        assert len(names) == len(set(names))
        assert len(names) == 5  # github, slack, google, linear, notion

    def test_all_service_icons_are_strings(self):
        """Icon values must be non-empty strings."""
        for name, info in SUPPORTED_SERVICES.items():
            assert isinstance(info["icon"], str), f"Service '{name}' icon is not a string"
            assert len(info["icon"]) > 0, f"Service '{name}' has empty icon"


# ===========================================================================
# 2. Policy validation at API level
# ===========================================================================

class TestPolicyValidation:
    """API-level validation of policy creation edge cases."""

    def test_create_policy_invalid_hours(self, auth_client):
        """Hours outside 0-23 are rejected."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "hour-bot",
            "agent_name": "Hour Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_hours": [9, 10, 25],
        })
        assert response.status_code == 400
        assert "Invalid hours" in response.json()["detail"]

    def test_create_policy_invalid_days(self, auth_client):
        """Days outside 0-6 are rejected."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "day-bot",
            "agent_name": "Day Bot",
            "allowed_services": ["slack"],
            "allowed_scopes": {"slack": ["channels:read"]},
            "allowed_days": [0, 1, 7],
        })
        assert response.status_code == 400
        assert "Invalid days" in response.json()["detail"]

    def test_create_policy_invalid_scope_for_service(self, auth_client):
        """A scope that does not belong to the service is rejected."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "scope-bot",
            "agent_name": "Scope Bot",
            "allowed_services": ["slack"],
            "allowed_scopes": {"slack": ["channels:read", "nonexistent_scope"]},
        })
        assert response.status_code == 400
        assert "Invalid scopes" in response.json()["detail"]

    def test_create_policy_valid_time_windows(self, auth_client):
        """Policy with valid hours and days succeeds."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "window-bot",
            "agent_name": "Window Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17],
            "allowed_days": [0, 1, 2, 3, 4],
        })
        assert response.status_code == 200
        assert response.json()["status"] == "created"

    def test_create_policy_with_ip_allowlist(self, auth_client):
        """Policy with IP allowlist succeeds."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "ip-bot",
            "agent_name": "IP Bot",
            "allowed_services": ["google"],
            "allowed_scopes": {"google": ["gmail.readonly"]},
            "ip_allowlist": ["10.0.0.1", "192.168.1.0/24"],
        })
        assert response.status_code == 200
        assert response.json()["status"] == "created"


# ===========================================================================
# 3. Token request with step-up auth
# ===========================================================================

class TestTokenStepUp:
    """Token endpoint interactions involving step-up authentication."""

    def test_token_request_step_up_returns_202(self, auth_client, db):
        """When a service requires step-up, the token endpoint returns 202."""
        _create_policy(
            agent_id="stepup-bot",
            services=["github"],
            scopes={"github": ["repo"]},
            step_up=["github"],
        )
        with patch("src.app.trigger_step_up_auth", new_callable=AsyncMock,
                   return_value={"status": "pending", "auth_req_id": "req_abc123",
                                 "expires_in": 120}):
            response = auth_client.post("/api/v1/token", json={
                "agent_id": "stepup-bot",
                "service": "github",
                "scopes": ["repo"],
            })
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "step_up_required"
        assert "auth_req_id" in data

    def test_token_request_no_step_up_proceeds(self, auth_client, db):
        """Service without step-up goes straight to token exchange."""
        _create_policy(agent_id="normal-bot", services=["github"],
                       scopes={"github": ["repo"]})
        with patch("src.app.get_token_vault_token", new_callable=AsyncMock,
                   return_value={
                       "access_token": "tok_abc",
                       "token_type": "Bearer",
                       "expires_in": 3600,
                       "scope": "repo",
                   }):
            response = auth_client.post("/api/v1/token", json={
                "agent_id": "normal-bot",
                "service": "github",
                "scopes": ["repo"],
            })
        assert response.status_code == 200
        assert response.json()["access_token"] == "tok_abc"

    def test_step_up_initiation_endpoint(self, auth_client):
        """POST /api/v1/step-up triggers CIBA and returns the result."""
        with patch("src.app.trigger_step_up_auth", new_callable=AsyncMock,
                   return_value={"auth_req_id": "req_xyz", "expires_in": 120,
                                 "status": "pending"}):
            response = auth_client.post("/api/v1/step-up", json={
                "agent_id": "any-bot",
                "action": "delete all issues",
            })
        assert response.status_code == 200
        assert response.json()["status"] == "pending"
        assert response.json()["auth_req_id"] == "req_xyz"

    def test_step_up_status_check(self, auth_client):
        """GET /api/v1/step-up/status/{id} returns the CIBA poll result."""
        with patch("src.app.check_step_up_status", new_callable=AsyncMock,
                   return_value={"status": "approved", "token": "tok_approved"}):
            response = auth_client.get("/api/v1/step-up/status/req_xyz")
        assert response.status_code == 200
        assert response.json()["status"] == "approved"

    def test_token_vault_error_returns_502(self, auth_client, db):
        """When the Token Vault fails, the token endpoint returns 502."""
        _create_policy(agent_id="vault-err-bot", services=["slack"],
                       scopes={"slack": ["channels:read"]})
        with patch("src.app.get_token_vault_token", new_callable=AsyncMock,
                   return_value={"error": "Token exchange failed"}):
            response = auth_client.post("/api/v1/token", json={
                "agent_id": "vault-err-bot",
                "service": "slack",
                "scopes": ["channels:read"],
            })
        assert response.status_code == 502
        assert "Token exchange failed" in response.json()["detail"]


# ===========================================================================
# 4. Emergency revoke endpoint
# ===========================================================================

class TestEmergencyRevoke:
    """Emergency kill-switch endpoint tests."""

    def test_emergency_revoke_returns_counts(self, auth_client, db):
        """Emergency revoke reports how many policies/keys were affected."""
        # Create two policies and one key
        _create_policy(agent_id="em-bot-1")
        _create_policy(agent_id="em-bot-2", services=["slack"],
                       scopes={"slack": ["channels:read"]})
        _run(create_api_key("auth0|user123", "em-bot-1", "k1"))

        response = auth_client.post("/api/v1/emergency-revoke")
        assert response.status_code == 200
        data = response.json()
        assert data["policies_disabled"] == 2
        assert data["keys_revoked"] == 1
        assert data["status"] == "all_access_revoked"

    def test_after_emergency_revoke_policies_inactive(self, auth_client, db):
        """After emergency revoke all policies should show is_active=False."""
        _create_policy(agent_id="em-pol-bot")
        auth_client.post("/api/v1/emergency-revoke")
        response = auth_client.get("/api/v1/policies")
        assert response.status_code == 200
        policies = response.json()
        assert len(policies) == 1
        assert policies[0]["is_active"] is False

    def test_after_emergency_revoke_keys_revoked(self, auth_client, db):
        """After emergency revoke all API keys should be flagged as revoked."""
        _run(create_api_key("auth0|user123", "em-key-bot", "k1"))
        auth_client.post("/api/v1/emergency-revoke")
        response = auth_client.get("/api/v1/keys")
        assert response.status_code == 200
        keys = response.json()
        assert len(keys) == 1
        assert keys[0]["is_revoked"] is True

    def test_emergency_revoke_is_user_isolated(self, auth_client, db):
        """Emergency revoke only affects the requesting user's resources."""
        from src.database import get_agent_policy

        # user123 creates a policy
        _create_policy(agent_id="iso-bot-u1", user_id="auth0|user123")
        # other999 creates a policy
        _create_policy(agent_id="iso-bot-u2", user_id="auth0|other999")

        # user123 triggers emergency revoke
        auth_client.post("/api/v1/emergency-revoke")

        # user123's policy is disabled
        resp1 = auth_client.get("/api/v1/policies")
        assert all(p["is_active"] is False for p in resp1.json())

        # other999's policy is still active (verify directly via DB)
        other_policy = _run(get_agent_policy("iso-bot-u2"))
        assert other_policy is not None
        assert other_policy.is_active is True

    def test_emergency_revoke_no_resources_returns_zero(self, auth_client, db):
        """Emergency revoke with no policies or keys returns zero counts."""
        response = auth_client.post("/api/v1/emergency-revoke")
        assert response.status_code == 200
        data = response.json()
        assert data["policies_disabled"] == 0
        assert data["keys_revoked"] == 0


# ===========================================================================
# 5. Delete policy endpoint
# ===========================================================================

class TestDeletePolicy:
    """DELETE /api/v1/policies/{agent_id} edge cases."""

    def test_delete_existing_policy(self, auth_client, db):
        """Deleting an existing policy returns success."""
        _create_policy(agent_id="del-bot")
        response = auth_client.delete("/api/v1/policies/del-bot")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

    def test_delete_nonexistent_policy(self, auth_client, db):
        """Deleting a policy that does not exist returns 404."""
        response = auth_client.delete("/api/v1/policies/ghost-bot")
        assert response.status_code == 404

    def test_delete_wrong_users_policy(self, auth_client, db):
        """A user cannot delete another user's policy (returns 404)."""
        _create_policy(agent_id="other-bot", user_id="auth0|other999")
        response = auth_client.delete("/api/v1/policies/other-bot")
        assert response.status_code == 404

    def test_after_delete_get_returns_404(self, auth_client, db):
        """After deletion, listing services for the deleted agent returns 404."""
        _create_policy(agent_id="vanish-bot")
        auth_client.delete("/api/v1/policies/vanish-bot")
        response = auth_client.get("/api/v1/services?agent_id=vanish-bot")
        assert response.status_code == 404

    def test_delete_and_recreate_same_agent_id(self, auth_client, db):
        """Deleting and re-creating a policy with the same agent_id works."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "phoenix-bot",
            "agent_name": "Phoenix",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        del_resp = auth_client.delete("/api/v1/policies/phoenix-bot")
        assert del_resp.status_code == 200

        # Re-create with different settings
        create_resp = auth_client.post("/api/v1/policies", json={
            "agent_id": "phoenix-bot",
            "agent_name": "Phoenix v2",
            "allowed_services": ["slack"],
            "allowed_scopes": {"slack": ["channels:read"]},
        })
        assert create_resp.status_code == 200
        assert create_resp.json()["agent_id"] == "phoenix-bot"


# ===========================================================================
# 6. Service listing and connection
# ===========================================================================

class TestServiceListing:
    """GET /api/v1/services and connection management."""

    def test_connected_service_shows_connected_true(self, auth_client, db):
        """Services the user has connected should show connected=true."""
        _create_policy(agent_id="svc-bot", services=["github"],
                       scopes={"github": ["repo"]})
        _run(add_connected_service("auth0|user123", "github", "conn_123"))

        response = auth_client.get("/api/v1/services?agent_id=svc-bot")
        assert response.status_code == 200
        services = response.json()
        github = [s for s in services if s["service"] == "github"]
        assert len(github) == 1
        assert github[0]["connected"] is True

    def test_no_connections_shows_connected_false(self, auth_client, db):
        """Without any connected services, connected should be false."""
        _create_policy(agent_id="noconn-bot", services=["slack"],
                       scopes={"slack": ["channels:read"]})

        response = auth_client.get("/api/v1/services?agent_id=noconn-bot")
        assert response.status_code == 200
        services = response.json()
        assert len(services) == 1
        assert services[0]["connected"] is False

    def test_connect_unknown_service_returns_400(self, auth_client):
        """Attempting to connect an unsupported service returns 400."""
        response = auth_client.post("/connect/dropbox", follow_redirects=False)
        assert response.status_code == 400

    def test_disconnect_service_returns_success(self, auth_client, db):
        """Disconnecting a service returns a success response."""
        _run(add_connected_service("auth0|user123", "slack", "conn_456"))
        response = auth_client.delete("/connect/slack")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "disconnected"
        assert data["service"] == "slack"

    def test_service_list_filters_to_supported(self, auth_client, db):
        """Only services in SUPPORTED_SERVICES appear in the listing."""
        _create_policy(
            agent_id="filter-bot",
            services=["github", "notion"],
            scopes={"github": ["repo"], "notion": ["read_content"]},
        )
        response = auth_client.get("/api/v1/services?agent_id=filter-bot")
        assert response.status_code == 200
        services = response.json()
        service_names = {s["service"] for s in services}
        assert service_names.issubset(set(SUPPORTED_SERVICES.keys()))
        assert service_names == {"github", "notion"}
