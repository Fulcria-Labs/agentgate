"""Comprehensive API endpoint tests — covering edge cases, error handling,
multi-step workflows, and response validation."""

import asyncio
import time
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.auth import SUPPORTED_SERVICES
from src.database import (
    AgentPolicy,
    add_connected_service,
    create_agent_policy,
    create_api_key,
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
# 1. Policy creation edge cases
# ---------------------------------------------------------------------------

class TestPolicyCreationEdgeCases:
    """Edge cases for POST /api/v1/policies."""

    def test_create_policy_with_all_services(self, auth_client):
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "all-svc-bot",
            "agent_name": "All Services Bot",
            "allowed_services": list(SUPPORTED_SERVICES.keys()),
            "allowed_scopes": {
                svc: info["scopes"][:1]
                for svc, info in SUPPORTED_SERVICES.items()
            },
        })
        assert response.status_code == 200
        assert response.json()["status"] == "created"

    def test_create_policy_with_multiple_scopes_per_service(self, auth_client):
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "multi-scope-bot",
            "agent_name": "Multi Scope Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo", "read:user", "gist"]},
        })
        assert response.status_code == 200

    def test_create_policy_with_all_constraints(self, auth_client):
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "constrained-bot",
            "agent_name": "Constrained Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "rate_limit_per_minute": 10,
            "requires_step_up": ["github"],
            "allowed_hours": [9, 10, 11, 12],
            "allowed_days": [0, 1, 2, 3, 4],
            "ip_allowlist": ["10.0.0.0/8"],
        })
        assert response.status_code == 200

    def test_create_policy_updates_existing(self, auth_client):
        auth_client.post("/api/v1/policies", json={
            "agent_id": "update-bot",
            "agent_name": "V1",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "update-bot",
            "agent_name": "V2",
            "allowed_services": ["slack"],
            "allowed_scopes": {"slack": ["chat:write"]},
        })
        assert response.status_code == 200

    def test_create_policy_scope_not_in_services(self, auth_client):
        """Scopes dict references a service not in SUPPORTED_SERVICES."""
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "scope-err",
            "agent_name": "Scope Error Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"dropbox": ["files"]},
        })
        assert response.status_code == 400

    def test_create_policy_negative_hours(self, auth_client):
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "neg-hour",
            "agent_name": "Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_hours": [-1, 5],
        })
        assert response.status_code == 400

    def test_create_policy_negative_days(self, auth_client):
        response = auth_client.post("/api/v1/policies", json={
            "agent_id": "neg-day",
            "agent_name": "Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_days": [-1],
        })
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# 2. Policy listing edge cases
# ---------------------------------------------------------------------------

class TestPolicyListingEdgeCases:
    """Edge cases for GET /api/v1/policies."""

    def test_list_policies_with_multiple(self, auth_client, db):
        for i in range(5):
            _create_policy(agent_id=f"list-bot-{i}")
        response = auth_client.get("/api/v1/policies")
        assert response.status_code == 200
        policies = response.json()
        assert len(policies) == 5

    def test_list_policies_shows_correct_fields(self, auth_client, db):
        _create_policy(
            agent_id="field-bot",
            services=["github"],
            scopes={"github": ["repo"]},
            rate_limit=30,
            ip_allowlist=["10.0.0.1"],
        )
        response = auth_client.get("/api/v1/policies")
        policies = response.json()
        assert len(policies) == 1
        p = policies[0]
        assert p["agent_id"] == "field-bot"
        assert p["agent_name"] == "Agent field-bot"
        assert p["allowed_services"] == ["github"]
        assert p["rate_limit_per_minute"] == 30
        assert p["is_active"] is True
        assert p["ip_allowlist"] == ["10.0.0.1"]


# ---------------------------------------------------------------------------
# 3. Token endpoint comprehensive
# ---------------------------------------------------------------------------

class TestTokenEndpointComprehensive:
    """Comprehensive tests for POST /api/v1/token."""

    def test_token_success_response_format(self, auth_client, db):
        _create_policy(agent_id="tok-bot")
        with patch("src.app.get_token_vault_token", new_callable=AsyncMock,
                   return_value={
                       "access_token": "tok_xyz",
                       "token_type": "Bearer",
                       "expires_in": 3600,
                       "scope": "repo",
                   }):
            response = auth_client.post("/api/v1/token", json={
                "agent_id": "tok-bot",
                "service": "github",
                "scopes": ["repo"],
            })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert "expires_in" in data
        assert "scope" in data
        assert data["agent_id"] == "tok-bot"
        assert data["service"] == "github"

    def test_token_for_disabled_agent(self, auth_client, db):
        _create_policy(agent_id="disabled-bot", is_active=False)
        response = auth_client.post("/api/v1/token", json={
            "agent_id": "disabled-bot",
            "service": "github",
            "scopes": ["repo"],
        })
        assert response.status_code == 403
        assert "disabled" in response.json()["detail"]

    def test_token_for_unauthorized_service(self, auth_client, db):
        _create_policy(agent_id="svc-bot", services=["github"],
                       scopes={"github": ["repo"]})
        response = auth_client.post("/api/v1/token", json={
            "agent_id": "svc-bot",
            "service": "slack",
            "scopes": ["channels:read"],
        })
        assert response.status_code == 403

    def test_token_for_excess_scopes(self, auth_client, db):
        _create_policy(agent_id="scope-bot", services=["github"],
                       scopes={"github": ["repo"]})
        response = auth_client.post("/api/v1/token", json={
            "agent_id": "scope-bot",
            "service": "github",
            "scopes": ["repo", "admin:org"],
        })
        assert response.status_code == 403

    def test_token_with_step_up_approved(self, auth_client, db):
        _create_policy(agent_id="stepup-ok", services=["github"],
                       scopes={"github": ["repo"]}, step_up=["github"])
        with patch("src.app.trigger_step_up_auth", new_callable=AsyncMock,
                   return_value={"status": "approved"}):
            with patch("src.app.get_token_vault_token", new_callable=AsyncMock,
                       return_value={
                           "access_token": "tok_approved",
                           "token_type": "Bearer",
                           "expires_in": 3600,
                           "scope": "repo",
                       }):
                response = auth_client.post("/api/v1/token", json={
                    "agent_id": "stepup-ok",
                    "service": "github",
                    "scopes": ["repo"],
                })
        assert response.status_code == 200
        assert response.json()["access_token"] == "tok_approved"

    def test_token_with_step_up_denied(self, auth_client, db):
        _create_policy(agent_id="stepup-deny", services=["github"],
                       scopes={"github": ["repo"]}, step_up=["github"])
        with patch("src.app.trigger_step_up_auth", new_callable=AsyncMock,
                   return_value={"status": "denied"}):
            response = auth_client.post("/api/v1/token", json={
                "agent_id": "stepup-deny",
                "service": "github",
                "scopes": ["repo"],
            })
        # Denied step-up should return 202 (step_up_required)
        assert response.status_code == 202

    def test_token_vault_error_502(self, auth_client, db):
        _create_policy(agent_id="vault-err")
        with patch("src.app.get_token_vault_token", new_callable=AsyncMock,
                   return_value={"error": "Service unavailable"}):
            response = auth_client.post("/api/v1/token", json={
                "agent_id": "vault-err",
                "service": "github",
                "scopes": ["repo"],
            })
        assert response.status_code == 502


# ---------------------------------------------------------------------------
# 4. API key management comprehensive
# ---------------------------------------------------------------------------

class TestApiKeyManagementComprehensive:
    """Comprehensive tests for API key endpoints."""

    def test_create_key_response_format(self, auth_client):
        response = auth_client.post("/api/v1/keys", json={
            "agent_id": "key-bot",
            "name": "my-key",
        })
        assert response.status_code == 200
        data = response.json()
        assert "key" in data
        assert "key_id" in data
        assert "key_prefix" in data
        assert "agent_id" in data
        assert "warning" in data
        assert data["key"].startswith("ag_")

    def test_create_key_with_expiry(self, auth_client):
        response = auth_client.post("/api/v1/keys", json={
            "agent_id": "exp-key-bot",
            "name": "expiring",
            "expires_in": 3600,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["expires_at"] is not None
        assert data["expires_at"] > 0

    def test_create_key_without_expiry(self, auth_client):
        response = auth_client.post("/api/v1/keys", json={
            "agent_id": "noexp-key-bot",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["expires_at"] is None

    def test_list_keys_metadata_only(self, auth_client):
        auth_client.post("/api/v1/keys", json={"agent_id": "meta-bot"})
        response = auth_client.get("/api/v1/keys")
        assert response.status_code == 200
        keys = response.json()
        for k in keys:
            assert "key" not in k  # Raw key never exposed
            assert "key_id" in k
            assert "key_prefix" in k
            assert "agent_id" in k
            assert "is_revoked" in k

    def test_revoke_key_then_verify_status(self, auth_client):
        create_resp = auth_client.post("/api/v1/keys", json={
            "agent_id": "rev-bot",
        })
        key_id = create_resp.json()["key_id"]
        auth_client.delete(f"/api/v1/keys/{key_id}")
        list_resp = auth_client.get("/api/v1/keys")
        keys = list_resp.json()
        revoked = [k for k in keys if k["key_id"] == key_id]
        assert len(revoked) == 1
        assert revoked[0]["is_revoked"] is True

    def test_multiple_keys_for_same_agent(self, auth_client):
        for i in range(3):
            auth_client.post("/api/v1/keys", json={
                "agent_id": "multi-key-bot",
                "name": f"key-{i}",
            })
        response = auth_client.get("/api/v1/keys")
        keys = [k for k in response.json() if k["agent_id"] == "multi-key-bot"]
        assert len(keys) == 3


# ---------------------------------------------------------------------------
# 5. API key auth on token endpoint
# ---------------------------------------------------------------------------

class TestApiKeyAuthComprehensive:
    """Comprehensive tests for API key authentication on token endpoint."""

    def test_api_key_with_matching_agent_id(self, client, db):
        _create_policy(agent_id="ak-bot")
        _, raw_key = _run(create_api_key("auth0|user123", "ak-bot", "test"))

        with patch("src.app.get_token_vault_token", new_callable=AsyncMock,
                   return_value={
                       "access_token": "tok",
                       "token_type": "Bearer",
                       "expires_in": 3600,
                       "scope": "repo",
                   }):
            response = client.post(
                "/api/v1/token",
                json={"agent_id": "ak-bot", "service": "github", "scopes": ["repo"]},
                headers={"Authorization": f"Bearer {raw_key}"},
            )
        assert response.status_code == 200

    def test_api_key_with_empty_agent_id_uses_key_agent(self, client, db):
        _create_policy(agent_id="ak-empty")
        _, raw_key = _run(create_api_key("auth0|user123", "ak-empty", "test"))

        with patch("src.app.get_token_vault_token", new_callable=AsyncMock,
                   return_value={
                       "access_token": "tok",
                       "token_type": "Bearer",
                       "expires_in": 3600,
                       "scope": "repo",
                   }):
            response = client.post(
                "/api/v1/token",
                json={"agent_id": "", "service": "github", "scopes": ["repo"]},
                headers={"Authorization": f"Bearer {raw_key}"},
            )
        assert response.status_code == 200

    def test_api_key_revoked_returns_401(self, client, db):
        _create_policy(agent_id="ak-rev")
        key_obj, raw_key = _run(create_api_key("auth0|user123", "ak-rev", "test"))
        # Revoke the key
        from src.database import revoke_api_key
        _run(revoke_api_key(key_obj.id, "auth0|user123"))

        response = client.post(
            "/api/v1/token",
            json={"agent_id": "ak-rev", "service": "github", "scopes": ["repo"]},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert response.status_code == 401

    def test_non_bearer_auth_header_ignored(self, client, db):
        """Authorization header without 'Bearer ag_' falls back to session."""
        response = client.post(
            "/api/v1/token",
            json={"agent_id": "bot", "service": "github", "scopes": ["repo"]},
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        # No session, so should get 401
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# 6. Emergency revoke endpoint
# ---------------------------------------------------------------------------

class TestEmergencyRevokeEndpoint:
    """Tests for POST /api/v1/emergency-revoke."""

    def test_emergency_revoke_response_format(self, auth_client, db):
        _create_policy(agent_id="em-bot")
        response = auth_client.post("/api/v1/emergency-revoke")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "policies_disabled" in data
        assert "keys_revoked" in data
        assert "message" in data
        assert data["status"] == "all_access_revoked"

    def test_emergency_revoke_then_token_denied(self, auth_client, db):
        _create_policy(agent_id="em-tok-bot")
        auth_client.post("/api/v1/emergency-revoke")
        response = auth_client.post("/api/v1/token", json={
            "agent_id": "em-tok-bot",
            "service": "github",
            "scopes": ["repo"],
        })
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# 7. Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_includes_version(self, client):
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_health_includes_service_name(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["service"] == "agentgate"


# ---------------------------------------------------------------------------
# 8. Step-up endpoint
# ---------------------------------------------------------------------------

class TestStepUpEndpoints:
    """Tests for step-up auth endpoints."""

    def test_step_up_requires_auth(self, client):
        response = client.post("/api/v1/step-up", json={
            "agent_id": "bot",
            "action": "delete",
        })
        assert response.status_code == 401

    def test_step_up_status_requires_auth(self, client):
        response = client.get("/api/v1/step-up/status/req123")
        assert response.status_code == 401

    def test_step_up_denied_response(self, auth_client):
        with patch("src.app.trigger_step_up_auth", new_callable=AsyncMock,
                   return_value={"status": "denied", "error": "CIBA failed"}):
            response = auth_client.post("/api/v1/step-up", json={
                "agent_id": "bot",
                "action": "risky action",
            })
        assert response.status_code == 200
        assert response.json()["status"] == "denied"

    def test_step_up_status_pending(self, auth_client):
        with patch("src.app.check_step_up_status", new_callable=AsyncMock,
                   return_value={"status": "pending"}):
            response = auth_client.get("/api/v1/step-up/status/req123")
        assert response.status_code == 200
        assert response.json()["status"] == "pending"

    def test_step_up_status_denied(self, auth_client):
        with patch("src.app.check_step_up_status", new_callable=AsyncMock,
                   return_value={"status": "denied"}):
            response = auth_client.get("/api/v1/step-up/status/req456")
        assert response.status_code == 200
        assert response.json()["status"] == "denied"
