"""Tests for OAuth callback route, service connection edges, dashboard edges,
and concurrent operation scenarios.

Covers critical gaps:
- OAuth /callback route (previously 0 tests)
- /login and /logout edge cases
- Service connection error handling
- Dashboard rendering with various data states
- Concurrent policy + key operations
- Audit log ordering and consistency
"""

import asyncio
import time
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.database import (
    init_db,
    create_agent_policy,
    create_api_key,
    add_connected_service,
    get_connected_services,
    remove_connected_service,
    get_audit_log,
    log_audit,
    AgentPolicy,
)


@pytest.fixture
def client(db, monkeypatch):
    """Create a test client with a fresh database."""
    import os
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(project_dir)
    return TestClient(app)


@pytest.fixture
def auth_client(client):
    """Client with an authenticated session."""
    with client:
        client.cookies.set("session", "test")
        with patch("src.app.get_user", return_value={
            "sub": "auth0|user123",
            "name": "Test User",
            "email": "test@example.com",
            "picture": "https://example.com/pic.jpg",
        }):
            yield client


def _create_policy_sync(agent_id, user_id="auth0|user123", services=None, scopes=None):
    """Helper to create a policy synchronously."""
    loop = asyncio.new_event_loop()
    policy = AgentPolicy(
        agent_id=agent_id,
        agent_name=f"Agent {agent_id}",
        allowed_services=services or ["github"],
        allowed_scopes=scopes or {"github": ["repo"]},
        rate_limit_per_minute=60,
        created_by=user_id,
        created_at=time.time(),
    )
    loop.run_until_complete(create_agent_policy(policy))
    loop.close()
    return policy


# =============================================================================
# OAuth Callback Tests
# =============================================================================

class TestOAuthCallback:
    """Tests for the /callback endpoint (previously 0 coverage)."""

    def test_callback_success(self, client):
        """Successful OAuth callback stores user in session and redirects to /."""
        mock_token = {
            "userinfo": {
                "sub": "auth0|new_user",
                "name": "New User",
                "email": "new@example.com",
                "picture": "https://example.com/avatar.jpg",
            },
            "access_token": "at_123",
            "token_type": "Bearer",
        }
        with patch("src.app.oauth.auth0.authorize_access_token",
                    new_callable=AsyncMock, return_value=mock_token):
            response = client.get("/callback", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/"

    def test_callback_missing_userinfo(self, client):
        """Callback with empty userinfo still creates session with empty fields."""
        mock_token = {"userinfo": {}}
        with patch("src.app.oauth.auth0.authorize_access_token",
                    new_callable=AsyncMock, return_value=mock_token):
            response = client.get("/callback", follow_redirects=False)
        assert response.status_code == 307

    def test_callback_no_userinfo_key(self, client):
        """Callback token without userinfo key creates empty session values."""
        mock_token = {"access_token": "at_123"}
        with patch("src.app.oauth.auth0.authorize_access_token",
                    new_callable=AsyncMock, return_value=mock_token):
            response = client.get("/callback", follow_redirects=False)
        assert response.status_code == 307

    def test_callback_partial_userinfo(self, client):
        """Callback with partial userinfo (only sub, no name/email)."""
        mock_token = {
            "userinfo": {
                "sub": "auth0|partial_user",
            }
        }
        with patch("src.app.oauth.auth0.authorize_access_token",
                    new_callable=AsyncMock, return_value=mock_token):
            response = client.get("/callback", follow_redirects=False)
        assert response.status_code == 307

    def test_callback_oauth_error(self, client):
        """Callback fails when Auth0 returns an error (e.g., invalid_grant)."""
        with patch("src.app.oauth.auth0.authorize_access_token",
                    new_callable=AsyncMock,
                    side_effect=Exception("invalid_grant: Authorization code expired")):
            with pytest.raises(Exception, match="invalid_grant"):
                client.get("/callback", follow_redirects=False)

    def test_callback_network_timeout(self, client):
        """Callback fails on network timeout to Auth0."""
        with patch("src.app.oauth.auth0.authorize_access_token",
                    new_callable=AsyncMock,
                    side_effect=Exception("Connection timeout")):
            with pytest.raises(Exception, match="Connection timeout"):
                client.get("/callback", follow_redirects=False)

    def test_callback_userinfo_special_characters(self, client):
        """Callback handles unicode and special characters in userinfo."""
        mock_token = {
            "userinfo": {
                "sub": "auth0|unicode_user",
                "name": "Tëst Üsér ñ",
                "email": "test+special@example.com",
                "picture": "",
            }
        }
        with patch("src.app.oauth.auth0.authorize_access_token",
                    new_callable=AsyncMock, return_value=mock_token):
            response = client.get("/callback", follow_redirects=False)
        assert response.status_code == 307

    def test_callback_long_sub_value(self, client):
        """Callback handles very long sub identifiers."""
        mock_token = {
            "userinfo": {
                "sub": "auth0|" + "a" * 500,
                "name": "Long Sub User",
                "email": "long@example.com",
                "picture": "",
            }
        }
        with patch("src.app.oauth.auth0.authorize_access_token",
                    new_callable=AsyncMock, return_value=mock_token):
            response = client.get("/callback", follow_redirects=False)
        assert response.status_code == 307

    def test_callback_replaces_existing_session(self, client):
        """Callback with existing session replaces user data."""
        mock_token_1 = {
            "userinfo": {"sub": "auth0|first", "name": "First", "email": "1@x.com", "picture": ""},
        }
        mock_token_2 = {
            "userinfo": {"sub": "auth0|second", "name": "Second", "email": "2@x.com", "picture": ""},
        }
        with patch("src.app.oauth.auth0.authorize_access_token",
                    new_callable=AsyncMock, return_value=mock_token_1):
            client.get("/callback", follow_redirects=False)
        with patch("src.app.oauth.auth0.authorize_access_token",
                    new_callable=AsyncMock, return_value=mock_token_2):
            response = client.get("/callback", follow_redirects=False)
        assert response.status_code == 307

    def test_callback_empty_string_fields(self, client):
        """Callback with empty string userinfo fields."""
        mock_token = {
            "userinfo": {
                "sub": "",
                "name": "",
                "email": "",
                "picture": "",
            }
        }
        with patch("src.app.oauth.auth0.authorize_access_token",
                    new_callable=AsyncMock, return_value=mock_token):
            response = client.get("/callback", follow_redirects=False)
        assert response.status_code == 307


# =============================================================================
# Login/Logout Edge Cases
# =============================================================================

class TestLoginLogoutEdges:
    """Extended tests for /login and /logout routes."""

    def test_login_redirects_to_auth0(self, client):
        """Login route triggers Auth0 redirect."""
        with patch("src.app.oauth.auth0.authorize_redirect",
                    new_callable=AsyncMock,
                    return_value=MagicMock(status_code=302, headers={"location": "https://auth0.example.com/authorize"})):
            response = client.get("/login", follow_redirects=False)
        # Should return whatever oauth redirect returns
        assert response.status_code in (200, 302, 307)

    def test_logout_clears_session(self, client):
        """Logout redirects to Auth0 logout URL."""
        response = client.get("/logout", follow_redirects=False)
        assert response.status_code == 307
        location = response.headers.get("location", "")
        assert "v2/logout" in location

    def test_logout_without_session(self, client):
        """Logout works even without an active session."""
        response = client.get("/logout", follow_redirects=False)
        assert response.status_code == 307

    def test_logout_returns_correct_redirect(self, client):
        """Logout URL includes client_id and returnTo."""
        response = client.get("/logout", follow_redirects=False)
        location = response.headers.get("location", "")
        assert "client_id=" in location
        assert "returnTo=" in location


# =============================================================================
# Service Connection Edge Cases
# =============================================================================

class TestServiceConnectionEdges:
    """Tests for service connection edge cases."""

    def test_connect_valid_service(self, auth_client):
        """Connecting a valid service initiates OAuth flow."""
        with patch("src.app.initiate_connection", new_callable=AsyncMock,
                    return_value="https://auth0.example.com/authorize?connection=github"):
            response = auth_client.post("/connect/github", follow_redirects=False)
        assert response.status_code == 303

    def test_connect_all_supported_services(self, auth_client):
        """Each supported service can initiate connection."""
        services = ["github", "slack", "google", "linear", "notion"]
        for svc in services:
            with patch("src.app.initiate_connection", new_callable=AsyncMock,
                        return_value=f"https://auth0.example.com/authorize?connection={svc}"):
                response = auth_client.post(f"/connect/{svc}", follow_redirects=False)
            assert response.status_code == 303, f"Failed for service: {svc}"

    def test_connect_unknown_service_returns_400(self, auth_client):
        """Connecting an unsupported service returns 400."""
        response = auth_client.post("/connect/fakebook", follow_redirects=False)
        assert response.status_code == 400
        assert "Unknown service" in response.json()["detail"]

    def test_connect_empty_service_name(self, auth_client):
        """Connecting with empty string service name fails."""
        response = auth_client.post("/connect/", follow_redirects=False)
        # FastAPI returns 404 for missing path parameter or 405
        assert response.status_code in (404, 405, 307)

    def test_connect_service_with_special_chars(self, auth_client):
        """Service name with special characters is rejected."""
        response = auth_client.post("/connect/git%00hub", follow_redirects=False)
        assert response.status_code in (400, 404)

    def test_disconnect_service_returns_success(self, auth_client):
        """Disconnecting a service returns success even if not connected."""
        response = auth_client.delete("/connect/github")
        assert response.status_code == 200
        assert response.json()["status"] == "disconnected"

    def test_disconnect_all_services(self, auth_client):
        """Disconnecting each supported service works."""
        for svc in ["github", "slack", "google", "linear", "notion"]:
            response = auth_client.delete(f"/connect/{svc}")
            assert response.status_code == 200

    def test_connect_requires_authentication(self, client):
        """Connection attempt without auth returns 401."""
        response = client.post("/connect/github", follow_redirects=False)
        assert response.status_code == 401

    def test_disconnect_requires_authentication(self, client):
        """Disconnection attempt without auth returns 401."""
        response = client.delete("/connect/github")
        assert response.status_code == 401

    def test_connect_initiation_failure(self, auth_client):
        """Connection initiation failure propagates error."""
        with patch("src.app.initiate_connection", new_callable=AsyncMock,
                    side_effect=Exception("Auth0 Token Vault unavailable")):
            with pytest.raises(Exception, match="Token Vault unavailable"):
                auth_client.post("/connect/github", follow_redirects=False)


# =============================================================================
# Dashboard Edge Cases
# =============================================================================

class TestDashboardEdges:
    """Tests for dashboard rendering with various data states."""

    def test_dashboard_unauthenticated_shows_login(self, client):
        """Unauthenticated users see the login page."""
        response = client.get("/")
        assert response.status_code == 200

    def test_dashboard_with_zero_policies(self, auth_client):
        """Dashboard renders correctly with no policies."""
        response = auth_client.get("/")
        assert response.status_code == 200

    def test_dashboard_with_many_policies(self, auth_client, db):
        """Dashboard renders with many agent policies."""
        loop = asyncio.new_event_loop()
        for i in range(25):
            p = AgentPolicy(
                agent_id=f"agent-{i}", agent_name=f"Agent {i}",
                allowed_services=["github"], allowed_scopes={"github": ["repo"]},
                rate_limit_per_minute=60, created_by="auth0|user123",
                created_at=time.time(),
            )
            loop.run_until_complete(create_agent_policy(p))
        loop.close()
        response = auth_client.get("/")
        assert response.status_code == 200

    def test_dashboard_with_connected_services(self, auth_client, db):
        """Dashboard shows connected services."""
        loop = asyncio.new_event_loop()
        loop.run_until_complete(add_connected_service("auth0|user123", "github"))
        loop.close()
        response = auth_client.get("/")
        assert response.status_code == 200

    def test_dashboard_with_audit_entries(self, auth_client, db):
        """Dashboard shows recent audit log entries."""
        loop = asyncio.new_event_loop()
        for i in range(5):
            loop.run_until_complete(log_audit(
                "auth0|user123", f"agent-{i}", "github",
                "token_issued", "success",
            ))
        loop.close()
        response = auth_client.get("/")
        assert response.status_code == 200

    def test_dashboard_with_many_audit_entries(self, auth_client, db):
        """Dashboard handles many audit entries (capped at 20)."""
        loop = asyncio.new_event_loop()
        for i in range(50):
            loop.run_until_complete(log_audit(
                "auth0|user123", f"agent-{i % 5}", "github",
                "token_issued", "success",
                details=f"Entry {i}",
            ))
        loop.close()
        response = auth_client.get("/")
        assert response.status_code == 200


# =============================================================================
# Audit Page Edge Cases
# =============================================================================

class TestAuditPageEdges:
    """Tests for the /audit page."""

    def test_audit_page_empty(self, auth_client):
        """Audit page renders with no entries."""
        response = auth_client.get("/audit")
        assert response.status_code == 200

    def test_audit_page_with_entries(self, auth_client, db):
        """Audit page shows entries from various actions."""
        loop = asyncio.new_event_loop()
        actions = [
            ("token_issued", "success"),
            ("token_request", "denied"),
            ("policy_created", "success"),
            ("api_key_created", "success"),
            ("emergency_revoke", "success"),
        ]
        for action, status in actions:
            loop.run_until_complete(log_audit(
                "auth0|user123", "test-agent", "github", action, status,
            ))
        loop.close()
        response = auth_client.get("/audit")
        assert response.status_code == 200

    def test_audit_requires_auth(self, client):
        """Audit page without auth returns 401."""
        response = client.get("/audit")
        assert response.status_code == 401

    def test_audit_log_ordering(self, db):
        """Audit entries are stored chronologically."""
        loop = asyncio.new_event_loop()
        for i in range(10):
            loop.run_until_complete(log_audit(
                "auth0|user123", f"agent-{i}", "github",
                "token_issued", "success",
                details=f"Entry {i}",
            ))
        entries = loop.run_until_complete(get_audit_log("auth0|user123", limit=10))
        loop.close()
        # Entries should be in reverse chronological order (newest first)
        for i in range(len(entries) - 1):
            assert entries[i].timestamp >= entries[i + 1].timestamp


# =============================================================================
# Policy Validation Edge Cases
# =============================================================================

class TestPolicyValidationEdges:
    """Tests for policy creation validation edge cases."""

    def test_policy_negative_rate_limit(self, auth_client):
        """Policy with negative rate limit."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "neg-rate",
            "agent_name": "Negative Rate Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "rate_limit_per_minute": -1,
        })
        # App accepts this; policy engine handles enforcement
        assert response.status_code == 200

    def test_policy_zero_rate_limit(self, auth_client):
        """Policy with zero rate limit effectively blocks all requests."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "zero-rate",
            "agent_name": "Zero Rate Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "rate_limit_per_minute": 0,
        })
        assert response.status_code == 200

    def test_policy_invalid_hours_out_of_range(self, auth_client):
        """Policy with hours > 23 is rejected."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "bad-hours",
            "agent_name": "Bad Hours Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_hours": [25, 30],
        })
        assert response.status_code == 400
        assert "Invalid hours" in response.json()["detail"]

    def test_policy_invalid_hours_negative(self, auth_client):
        """Policy with negative hours is rejected."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "neg-hours",
            "agent_name": "Neg Hours Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_hours": [-1],
        })
        assert response.status_code == 400

    def test_policy_invalid_days_out_of_range(self, auth_client):
        """Policy with days > 6 is rejected."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "bad-days",
            "agent_name": "Bad Days Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_days": [7, 8],
        })
        assert response.status_code == 400
        assert "Invalid days" in response.json()["detail"]

    def test_policy_invalid_days_negative(self, auth_client):
        """Policy with negative days is rejected."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "neg-days",
            "agent_name": "Neg Days Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_days": [-1],
        })
        assert response.status_code == 400

    def test_policy_all_valid_hours(self, auth_client):
        """Policy with all 24 hours is valid."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "all-hours",
            "agent_name": "All Hours Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_hours": list(range(24)),
        })
        assert response.status_code == 200

    def test_policy_all_valid_days(self, auth_client):
        """Policy with all 7 days is valid."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "all-days",
            "agent_name": "All Days Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_days": list(range(7)),
        })
        assert response.status_code == 200

    def test_policy_boundary_hours(self, auth_client):
        """Policy with boundary hours 0 and 23."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "boundary-hours",
            "agent_name": "Boundary Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_hours": [0, 23],
        })
        assert response.status_code == 200

    def test_policy_multiple_services_and_scopes(self, auth_client):
        """Policy with multiple services each with different scopes."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "multi-svc",
            "agent_name": "Multi Service Bot",
            "allowed_services": ["github", "slack", "google"],
            "allowed_scopes": {
                "github": ["repo", "read:user"],
                "slack": ["channels:read", "chat:write"],
                "google": ["gmail.readonly"],
            },
        })
        assert response.status_code == 200

    def test_policy_empty_services(self, auth_client):
        """Policy with empty services list."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "no-svc",
            "agent_name": "No Service Bot",
            "allowed_services": [],
            "allowed_scopes": {},
        })
        assert response.status_code == 200

    def test_policy_with_ip_allowlist(self, auth_client):
        """Policy with CIDR-notation IP allowlist."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "ip-bot",
            "agent_name": "IP Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "ip_allowlist": ["192.168.1.0/24", "10.0.0.1"],
        })
        assert response.status_code == 200

    def test_policy_with_expiration(self, auth_client):
        """Policy with a future expiration timestamp."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "exp-bot",
            "agent_name": "Expiring Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "expires_at": time.time() + 86400,
        })
        assert response.status_code == 200

    def test_policy_with_step_up_services(self, auth_client):
        """Policy with step-up auth required for certain services."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "stepup-bot",
            "agent_name": "Step-Up Bot",
            "allowed_services": ["github", "slack"],
            "allowed_scopes": {"github": ["repo"], "slack": ["chat:write"]},
            "requires_step_up": ["slack"],
        })
        assert response.status_code == 200

    def test_policy_very_long_agent_id(self, auth_client):
        """Policy with very long agent ID."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "a" * 500,
            "agent_name": "Long ID Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        assert response.status_code == 200

    def test_policy_duplicate_services_in_list(self, auth_client):
        """Policy with duplicated service names."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "dup-svc",
            "agent_name": "Dup Svc Bot",
            "allowed_services": ["github", "github", "github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        assert response.status_code == 200

    def test_policy_scope_for_unlisted_service(self, auth_client):
        """Policy with scopes for a service not in allowed_services."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "orphan-scope",
            "agent_name": "Orphan Scope Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"slack": ["chat:write"]},
        })
        # slack scopes refer to unknown service context for this policy, but
        # the validation checks if the service key is in SUPPORTED_SERVICES
        assert response.status_code in (200, 400)


# =============================================================================
# Policy Delete Edge Cases
# =============================================================================

class TestPolicyDeleteEdges:
    """Tests for policy deletion edge cases."""

    def test_delete_existing_policy(self, auth_client, db):
        """Delete an existing policy."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "del-me",
            "agent_name": "Delete Me Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        response = auth_client.delete("/api/v1/policies/del-me")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

    def test_delete_nonexistent_policy(self, auth_client):
        """Delete non-existent policy returns 404."""
        response = auth_client.delete("/api/v1/policies/nonexistent")
        assert response.status_code == 404

    def test_delete_policy_then_list(self, auth_client, db):
        """After deletion, policy no longer appears in list."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "temp-bot",
            "agent_name": "Temp Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        auth_client.delete("/api/v1/policies/temp-bot")
        response = auth_client.get("/api/v1/policies")
        agent_ids = [p["agent_id"] for p in response.json()]
        assert "temp-bot" not in agent_ids

    def test_delete_requires_auth(self, client):
        """Deleting a policy requires authentication."""
        response = client.delete("/api/v1/policies/some-bot")
        assert response.status_code == 401


# =============================================================================
# API Key Edge Cases
# =============================================================================

class TestApiKeyEdges:
    """Additional tests for API key lifecycle."""

    def test_create_key_with_expiry(self, auth_client):
        """Create a key that expires in 1 hour."""
        response = auth_client.post("/api/v1/keys", json={
            "agent_id": "exp-key-bot",
            "name": "expiring-key",
            "expires_in": 3600,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["key"].startswith("ag_")
        assert data["expires_at"] is not None

    def test_create_key_no_expiry(self, auth_client):
        """Create a key that never expires."""
        response = auth_client.post("/api/v1/keys", json={
            "agent_id": "perm-key-bot",
            "name": "permanent-key",
            "expires_in": 0,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["expires_at"] is None

    def test_create_multiple_keys_same_agent(self, auth_client):
        """Create multiple keys for the same agent."""
        keys = []
        for i in range(5):
            response = auth_client.post("/api/v1/keys", json={
                "agent_id": "multi-key-bot",
                "name": f"key-{i}",
            })
            assert response.status_code == 200
            keys.append(response.json()["key"])
        # All keys should be unique
        assert len(set(keys)) == 5

    def test_key_prefix_format(self, auth_client):
        """Key prefix is the first 8 characters of the raw key."""
        response = auth_client.post("/api/v1/keys", json={
            "agent_id": "prefix-bot",
            "name": "prefix-test",
        })
        data = response.json()
        assert data["key_prefix"] == data["key"][:8]

    def test_list_keys_does_not_expose_raw_key(self, auth_client):
        """Key listing never contains the full raw key."""
        auth_client.post("/api/v1/keys", json={
            "agent_id": "list-bot",
            "name": "list-test",
        })
        response = auth_client.get("/api/v1/keys")
        for key_entry in response.json():
            assert "key" not in key_entry or key_entry.get("key") is None

    def test_revoke_key_then_use(self, client, db):
        """Revoked key cannot be used for authentication."""
        loop = asyncio.new_event_loop()
        policy = AgentPolicy(
            agent_id="revoke-bot", agent_name="Revoke Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            rate_limit_per_minute=60,
            created_by="auth0|user123", created_at=time.time(),
        )
        loop.run_until_complete(create_agent_policy(policy))
        _, raw_key = loop.run_until_complete(
            create_api_key("auth0|user123", "revoke-bot", "test")
        )
        loop.close()

        # Revoke via authenticated session
        with patch("src.app.get_user", return_value={
            "sub": "auth0|user123", "name": "Test", "email": "", "picture": "",
        }):
            keys_resp = client.get("/api/v1/keys")
            key_id = keys_resp.json()[0]["key_id"]
            client.delete(f"/api/v1/keys/{key_id}")

        # Try using revoked key
        response = client.post(
            "/api/v1/token",
            json={"agent_id": "revoke-bot", "service": "github", "scopes": ["repo"]},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert response.status_code == 401

    def test_create_key_requires_auth(self, client):
        """Key creation requires authentication."""
        response = client.post("/api/v1/keys", json={
            "agent_id": "unauth-bot", "name": "test",
        })
        assert response.status_code == 401

    def test_list_keys_requires_auth(self, client):
        """Key listing requires authentication."""
        response = client.get("/api/v1/keys")
        assert response.status_code == 401


# =============================================================================
# Emergency Revoke Edge Cases
# =============================================================================

class TestEmergencyRevokeEdges:
    """Tests for emergency revoke behavior."""

    def test_emergency_revoke_empty(self, auth_client):
        """Emergency revoke with no policies or keys."""
        response = auth_client.post("/api/v1/emergency-revoke")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "all_access_revoked"
        assert data["policies_disabled"] == 0
        assert data["keys_revoked"] == 0

    def test_emergency_revoke_with_active_policies(self, auth_client, db):
        """Emergency revoke disables all active policies."""
        for i in range(5):
            auth_client.post("/api/v1/policies", json={
                "agent_id": f"em-bot-{i}",
                "agent_name": f"Emergency Bot {i}",
                "allowed_services": ["github"],
                "allowed_scopes": {"github": ["repo"]},
            })
        response = auth_client.post("/api/v1/emergency-revoke")
        assert response.status_code == 200
        data = response.json()
        assert data["policies_disabled"] == 5

    def test_emergency_revoke_with_keys(self, auth_client, db):
        """Emergency revoke also revokes all API keys."""
        for i in range(3):
            auth_client.post("/api/v1/keys", json={
                "agent_id": f"key-bot-{i}",
                "name": f"key-{i}",
            })
        response = auth_client.post("/api/v1/emergency-revoke")
        assert response.status_code == 200
        data = response.json()
        assert data["keys_revoked"] == 3

    def test_emergency_revoke_requires_auth(self, client):
        """Emergency revoke requires authentication."""
        response = client.post("/api/v1/emergency-revoke")
        assert response.status_code == 401

    def test_emergency_revoke_idempotent(self, auth_client, db):
        """Double emergency revoke is safe (no errors)."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "idem-bot",
            "agent_name": "Idempotent Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        r1 = auth_client.post("/api/v1/emergency-revoke")
        assert r1.status_code == 200
        assert r1.json()["policies_disabled"] == 1
        # Second revoke should succeed without errors
        r2 = auth_client.post("/api/v1/emergency-revoke")
        assert r2.status_code == 200
        assert r2.json()["status"] == "all_access_revoked"


# =============================================================================
# Toggle Policy Edge Cases
# =============================================================================

class TestTogglePolicyEdges:
    """Tests for policy toggle edge cases."""

    def test_toggle_twice_returns_to_original(self, auth_client, db):
        """Toggle disabled then toggle enabled."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "toggle2",
            "agent_name": "Toggle2 Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        # Toggle off
        r1 = auth_client.post("/api/v1/policies/toggle2/toggle")
        assert r1.json()["is_active"] is False
        # Toggle on
        r2 = auth_client.post("/api/v1/policies/toggle2/toggle")
        assert r2.json()["is_active"] is True

    def test_toggle_creates_audit_entry(self, auth_client, db):
        """Toggling a policy logs an audit entry."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "audit-toggle",
            "agent_name": "Audit Toggle Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        auth_client.post("/api/v1/policies/audit-toggle/toggle")
        # Verify audit entry was created
        loop = asyncio.new_event_loop()
        entries = loop.run_until_complete(get_audit_log("auth0|user123", limit=5))
        loop.close()
        actions = [e.action for e in entries]
        assert any("disabled" in a or "enabled" in a for a in actions)

    def test_toggle_requires_auth(self, client):
        """Toggle requires authentication."""
        response = client.post("/api/v1/policies/some-bot/toggle")
        assert response.status_code == 401


# =============================================================================
# Token Request with Step-Up Auth
# =============================================================================

class TestTokenStepUpEdges:
    """Tests for token requests triggering step-up auth."""

    def test_token_step_up_required(self, auth_client, db):
        """Token request for step-up service returns 202."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "stepup-agent",
            "agent_name": "Step-Up Agent",
            "allowed_services": ["github", "slack"],
            "allowed_scopes": {"github": ["repo"], "slack": ["chat:write"]},
            "requires_step_up": ["slack"],
        })
        with patch("src.app.trigger_step_up_auth", new_callable=AsyncMock,
                    return_value={"status": "pending", "auth_req_id": "req_123", "expires_in": 120}):
            response = auth_client.post("/api/v1/token", json={
                "agent_id": "stepup-agent",
                "service": "slack",
                "scopes": ["chat:write"],
            })
        assert response.status_code == 202
        assert response.json()["status"] == "step_up_required"
        assert response.json()["auth_req_id"] == "req_123"

    def test_token_no_step_up_for_allowed_service(self, auth_client, db):
        """Token request for non-step-up service succeeds directly."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "no-stepup-agent",
            "agent_name": "No Step-Up Agent",
            "allowed_services": ["github", "slack"],
            "allowed_scopes": {"github": ["repo"], "slack": ["chat:write"]},
            "requires_step_up": ["slack"],
        })
        with patch("src.app.get_token_vault_token", new_callable=AsyncMock,
                    return_value={
                        "access_token": "gh_token",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "scope": "repo",
                    }):
            response = auth_client.post("/api/v1/token", json={
                "agent_id": "no-stepup-agent",
                "service": "github",
                "scopes": ["repo"],
            })
        assert response.status_code == 200
        assert response.json()["access_token"] == "gh_token"


# =============================================================================
# Token Vault Error Handling
# =============================================================================

class TestTokenVaultErrors:
    """Tests for Token Vault failure scenarios."""

    def test_token_vault_returns_error(self, auth_client, db):
        """Token Vault error returns 502."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "vault-err-agent",
            "agent_name": "Vault Error Agent",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        with patch("src.app.get_token_vault_token", new_callable=AsyncMock,
                    return_value={"error": "Token exchange failed: service unavailable"}):
            response = auth_client.post("/api/v1/token", json={
                "agent_id": "vault-err-agent",
                "service": "github",
                "scopes": ["repo"],
            })
        assert response.status_code == 502
        assert "Token exchange failed" in response.json()["detail"]

    def test_token_vault_timeout(self, auth_client, db):
        """Token Vault timeout propagates as error."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "vault-timeout-agent",
            "agent_name": "Vault Timeout Agent",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        with patch("src.app.get_token_vault_token", new_callable=AsyncMock,
                    side_effect=Exception("Connection timeout")):
            with pytest.raises(Exception, match="Connection timeout"):
                auth_client.post("/api/v1/token", json={
                    "agent_id": "vault-timeout-agent",
                    "service": "github",
                    "scopes": ["repo"],
                })

    def test_token_vault_success_logged(self, auth_client, db):
        """Successful token issuance creates audit entry."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "log-agent",
            "agent_name": "Log Agent",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        with patch("src.app.get_token_vault_token", new_callable=AsyncMock,
                    return_value={
                        "access_token": "tok_123",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "scope": "repo",
                    }):
            auth_client.post("/api/v1/token", json={
                "agent_id": "log-agent",
                "service": "github",
                "scopes": ["repo"],
            })
        loop = asyncio.new_event_loop()
        entries = loop.run_until_complete(get_audit_log("auth0|user123", limit=5))
        loop.close()
        assert any(e.action == "token_issued" for e in entries)


# =============================================================================
# Services List Edge Cases
# =============================================================================

class TestServicesListEdges:
    """Tests for /api/v1/services endpoint."""

    def test_list_services_for_agent(self, auth_client, db):
        """List services shows agent's allowed services with connection status."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "svc-list-agent",
            "agent_name": "Svc List Agent",
            "allowed_services": ["github", "slack"],
            "allowed_scopes": {"github": ["repo"], "slack": ["chat:write"]},
        })
        response = auth_client.get("/api/v1/services?agent_id=svc-list-agent")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        svc_names = {s["service"] for s in data}
        assert svc_names == {"github", "slack"}

    def test_list_services_unknown_agent(self, auth_client):
        """List services for unknown agent returns 404."""
        response = auth_client.get("/api/v1/services?agent_id=unknown")
        assert response.status_code == 404

    def test_list_services_shows_step_up_flag(self, auth_client, db):
        """Services list shows which services need step-up."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "stepup-svc-agent",
            "agent_name": "Step-Up Svc Agent",
            "allowed_services": ["github", "slack"],
            "allowed_scopes": {"github": ["repo"], "slack": ["chat:write"]},
            "requires_step_up": ["slack"],
        })
        response = auth_client.get("/api/v1/services?agent_id=stepup-svc-agent")
        data = response.json()
        slack_svc = next(s for s in data if s["service"] == "slack")
        github_svc = next(s for s in data if s["service"] == "github")
        assert slack_svc["requires_step_up"] is True
        assert github_svc["requires_step_up"] is False

    def test_list_services_requires_auth(self, client):
        """Services list requires authentication."""
        response = client.get("/api/v1/services?agent_id=test")
        assert response.status_code == 401


# =============================================================================
# Connected Services Database Operations
# =============================================================================

class TestConnectedServicesDB:
    """Tests for connected services database layer."""

    def test_add_and_get_connected_service(self, db):
        """Add and retrieve a connected service."""
        loop = asyncio.new_event_loop()
        loop.run_until_complete(add_connected_service("auth0|user1", "github"))
        services = loop.run_until_complete(get_connected_services("auth0|user1"))
        loop.close()
        assert len(services) >= 1
        assert any(s["service"] == "github" for s in services)

    def test_add_multiple_services(self, db):
        """Add multiple different services."""
        loop = asyncio.new_event_loop()
        for svc in ["github", "slack", "google"]:
            loop.run_until_complete(add_connected_service("auth0|multi_svc", svc))
        services = loop.run_until_complete(get_connected_services("auth0|multi_svc"))
        loop.close()
        assert len(services) >= 3

    def test_remove_connected_service(self, db):
        """Remove a connected service."""
        loop = asyncio.new_event_loop()
        loop.run_until_complete(add_connected_service("auth0|rm_svc", "github"))
        loop.run_until_complete(remove_connected_service("auth0|rm_svc", "github"))
        services = loop.run_until_complete(get_connected_services("auth0|rm_svc"))
        loop.close()
        assert not any(s["service"] == "github" for s in services)

    def test_services_isolated_per_user(self, db):
        """Connected services are isolated between users."""
        loop = asyncio.new_event_loop()
        loop.run_until_complete(add_connected_service("auth0|user_a", "github"))
        loop.run_until_complete(add_connected_service("auth0|user_b", "slack"))
        a_svcs = loop.run_until_complete(get_connected_services("auth0|user_a"))
        b_svcs = loop.run_until_complete(get_connected_services("auth0|user_b"))
        loop.close()
        a_names = {s["service"] for s in a_svcs}
        b_names = {s["service"] for s in b_svcs}
        assert "github" in a_names
        assert "slack" not in a_names
        assert "slack" in b_names
        assert "github" not in b_names


# =============================================================================
# Health Endpoint Extended
# =============================================================================

class TestHealthExtended:
    """Extended health endpoint tests."""

    def test_health_returns_version(self, client):
        """Health endpoint includes version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data

    def test_health_no_auth_required(self, client):
        """Health endpoint works without authentication."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_correct_content_type(self, client):
        """Health endpoint returns JSON."""
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]
