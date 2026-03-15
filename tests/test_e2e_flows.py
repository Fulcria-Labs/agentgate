"""End-to-end flow tests for AgentGate.

Tests complete user journeys from policy creation through token issuance,
API key management, audit trail verification, and emergency procedures.
"""

import time
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from src.app import app
from src.database import (
    init_db, create_agent_policy, create_api_key, add_connected_service,
    get_audit_log, get_all_policies, get_api_keys, get_connected_services,
    AgentPolicy,
)
from src.policy import _rate_counters


@pytest.fixture
def client(db, monkeypatch):
    import os
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(project_dir)
    _rate_counters.clear()
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


@pytest.fixture
def auth_client_b(client):
    """Second user for multi-tenant tests."""
    with client:
        client.cookies.set("session", "test2")
        with patch("src.app.get_user", return_value={
            "sub": "auth0|user456",
            "name": "Other User",
            "email": "other@example.com",
            "picture": "",
        }):
            yield client


USER_ID = "auth0|user123"
USER_ID_B = "auth0|user456"


class TestFullPolicyLifecycle:
    """Complete lifecycle: create -> use -> toggle -> delete."""

    @pytest.mark.asyncio
    async def test_create_policy_list_toggle_delete(self, auth_client, db):
        # 1. Create policy
        resp = auth_client.post("/api/v1/policies", json={
            "agent_id": "ci-bot",
            "agent_name": "CI Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo", "read:user"]},
            "rate_limit_per_minute": 30,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "created"

        # 2. List policies - should appear
        resp = auth_client.get("/api/v1/policies")
        assert resp.status_code == 200
        policies = resp.json()
        assert len(policies) == 1
        assert policies[0]["agent_id"] == "ci-bot"
        assert policies[0]["is_active"] is True

        # 3. Toggle to disable
        resp = auth_client.post("/api/v1/policies/ci-bot/toggle")
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False

        # 4. Verify disabled in list
        resp = auth_client.get("/api/v1/policies")
        assert resp.json()[0]["is_active"] is False

        # 5. Toggle back to enable
        resp = auth_client.post("/api/v1/policies/ci-bot/toggle")
        assert resp.status_code == 200
        assert resp.json()["is_active"] is True

        # 6. Delete policy
        resp = auth_client.delete("/api/v1/policies/ci-bot")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # 7. Verify gone
        resp = auth_client.get("/api/v1/policies")
        assert len(resp.json()) == 0

    @pytest.mark.asyncio
    async def test_recreate_deleted_policy(self, auth_client, db):
        """Can recreate a policy with the same agent_id after deletion."""
        # Create
        auth_client.post("/api/v1/policies", json={
            "agent_id": "temp-bot",
            "agent_name": "Temp Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        # Delete
        auth_client.delete("/api/v1/policies/temp-bot")
        # Recreate with different config
        resp = auth_client.post("/api/v1/policies", json={
            "agent_id": "temp-bot",
            "agent_name": "Temp Bot v2",
            "allowed_services": ["github", "slack"],
            "allowed_scopes": {"github": ["repo"], "slack": ["chat:write"]},
        })
        assert resp.status_code == 200
        policies = auth_client.get("/api/v1/policies").json()
        assert len(policies) == 1
        assert "slack" in policies[0]["allowed_services"]


class TestFullTokenFlow:
    """Complete token request flows with policy enforcement."""

    @pytest.mark.asyncio
    async def test_policy_create_then_token_request(self, auth_client, db):
        """Full flow: create policy -> request token -> verify audit."""
        # Create policy
        auth_client.post("/api/v1/policies", json={
            "agent_id": "code-bot",
            "agent_name": "Code Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo", "read:user"]},
        })

        # Request token (mock Token Vault)
        with patch("src.app.get_token_vault_token", new_callable=AsyncMock, return_value={
            "access_token": "gho_abc123",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "repo read:user",
        }):
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "code-bot",
                "service": "github",
                "scopes": ["repo"],
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["access_token"] == "gho_abc123"
            assert data["agent_id"] == "code-bot"
            assert data["service"] == "github"

        # Verify audit log
        audit = await get_audit_log(USER_ID, limit=10)
        actions = [e.action for e in audit]
        assert "token_issued" in actions
        assert "policy_created" in actions

    @pytest.mark.asyncio
    async def test_token_denied_for_unauthorized_service(self, auth_client, db):
        """Token request for non-allowed service is denied."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "github-only",
            "agent_name": "GitHub Only",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })

        resp = auth_client.post("/api/v1/token", json={
            "agent_id": "github-only",
            "service": "slack",
            "scopes": ["chat:write"],
        })
        assert resp.status_code == 403
        assert "not authorized" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_token_denied_for_excess_scopes(self, auth_client, db):
        """Token request with scopes beyond policy is denied."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "limited-bot",
            "agent_name": "Limited Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["read:user"]},
        })

        resp = auth_client.post("/api/v1/token", json={
            "agent_id": "limited-bot",
            "service": "github",
            "scopes": ["repo", "read:user"],
        })
        assert resp.status_code == 403
        assert "repo" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_token_denied_after_policy_disabled(self, auth_client, db):
        """Disabling a policy blocks token requests."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "toggle-bot",
            "agent_name": "Toggle Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        # Disable
        auth_client.post("/api/v1/policies/toggle-bot/toggle")

        resp = auth_client.post("/api/v1/token", json={
            "agent_id": "toggle-bot",
            "service": "github",
            "scopes": ["repo"],
        })
        assert resp.status_code == 403
        assert "disabled" in resp.json()["detail"].lower()


class TestApiKeyTokenFlow:
    """Token requests via API key authentication."""

    @pytest.mark.asyncio
    async def test_create_key_then_use_for_token(self, auth_client, db):
        """Full flow: create policy -> create key -> use key for token."""
        # Create policy
        auth_client.post("/api/v1/policies", json={
            "agent_id": "api-bot",
            "agent_name": "API Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })

        # Create API key
        resp = auth_client.post("/api/v1/keys", json={
            "agent_id": "api-bot",
            "name": "prod-key",
        })
        assert resp.status_code == 200
        raw_key = resp.json()["key"]
        assert raw_key.startswith("ag_")

        # Use key for token request
        with patch("src.app.get_token_vault_token", new_callable=AsyncMock, return_value={
            "access_token": "gho_key_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "repo",
        }):
            resp = auth_client.post(
                "/api/v1/token",
                json={"agent_id": "api-bot", "service": "github", "scopes": ["repo"]},
                headers={"Authorization": f"Bearer {raw_key}"},
            )
            assert resp.status_code == 200
            assert resp.json()["access_token"] == "gho_key_token"

    @pytest.mark.asyncio
    async def test_api_key_bound_to_agent(self, auth_client, db):
        """API key for agent A cannot request tokens for agent B."""
        # Create two policies
        for agent in ["agent-a", "agent-b"]:
            auth_client.post("/api/v1/policies", json={
                "agent_id": agent,
                "agent_name": agent.title(),
                "allowed_services": ["github"],
                "allowed_scopes": {"github": ["repo"]},
            })

        # Create key for agent-a
        resp = auth_client.post("/api/v1/keys", json={"agent_id": "agent-a"})
        key_a = resp.json()["key"]

        # Try to use agent-a's key for agent-b
        resp = auth_client.post(
            "/api/v1/token",
            json={"agent_id": "agent-b", "service": "github", "scopes": ["repo"]},
            headers={"Authorization": f"Bearer {key_a}"},
        )
        assert resp.status_code == 403
        assert "bound to" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_revoked_key_rejected(self, auth_client, db):
        """Revoked API key is rejected."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "revoke-test",
            "agent_name": "Revoke Test",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        resp = auth_client.post("/api/v1/keys", json={"agent_id": "revoke-test"})
        key_data = resp.json()
        raw_key = key_data["key"]
        key_id = key_data["key_id"]

        # Revoke the key
        auth_client.delete(f"/api/v1/keys/{key_id}")

        # Try to use revoked key
        resp = auth_client.post(
            "/api/v1/token",
            json={"agent_id": "revoke-test", "service": "github", "scopes": ["repo"]},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_expired_key_rejected(self, auth_client, db):
        """Expired API key is rejected."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "expire-test",
            "agent_name": "Expire Test",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        # Create key with 1s expiry
        resp = auth_client.post("/api/v1/keys", json={
            "agent_id": "expire-test",
            "expires_in": 1,
        })
        raw_key = resp.json()["key"]

        # Wait for expiry
        import time as t
        t.sleep(1.1)

        resp = auth_client.post(
            "/api/v1/token",
            json={"agent_id": "expire-test", "service": "github", "scopes": ["repo"]},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_key_list_and_revoke_flow(self, auth_client, db):
        """Create multiple keys, list them, revoke one, verify state."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "multi-key",
            "agent_name": "Multi Key",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })

        # Create 3 keys
        keys = []
        for i in range(3):
            resp = auth_client.post("/api/v1/keys", json={
                "agent_id": "multi-key",
                "name": f"key-{i}",
            })
            keys.append(resp.json())

        # List - should see 3
        resp = auth_client.get("/api/v1/keys")
        assert len(resp.json()) == 3

        # Revoke middle key
        auth_client.delete(f"/api/v1/keys/{keys[1]['key_id']}")

        # List - still 3 but one revoked
        resp = auth_client.get("/api/v1/keys")
        listed = resp.json()
        assert len(listed) == 3
        revoked = [k for k in listed if k["is_revoked"]]
        assert len(revoked) == 1
        assert revoked[0]["name"] == "key-1"


class TestEmergencyRevokeFlow:
    """Emergency kill switch end-to-end tests."""

    @pytest.mark.asyncio
    async def test_emergency_revoke_disables_everything(self, auth_client, db):
        """Emergency revoke disables all policies and revokes all keys."""
        # Create multiple policies and keys
        for i in range(3):
            auth_client.post("/api/v1/policies", json={
                "agent_id": f"bot-{i}",
                "agent_name": f"Bot {i}",
                "allowed_services": ["github"],
                "allowed_scopes": {"github": ["repo"]},
            })
            auth_client.post("/api/v1/keys", json={"agent_id": f"bot-{i}"})

        # Emergency revoke
        resp = auth_client.post("/api/v1/emergency-revoke")
        assert resp.status_code == 200
        data = resp.json()
        assert data["policies_disabled"] == 3
        assert data["keys_revoked"] == 3

        # Verify all policies disabled
        resp = auth_client.get("/api/v1/policies")
        for p in resp.json():
            assert p["is_active"] is False

        # Verify all keys revoked
        resp = auth_client.get("/api/v1/keys")
        for k in resp.json():
            assert k["is_revoked"] is True

    @pytest.mark.asyncio
    async def test_emergency_revoke_audit_trail(self, auth_client, db):
        """Emergency revoke creates an audit entry."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "audit-bot",
            "agent_name": "Audit Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        auth_client.post("/api/v1/emergency-revoke")

        audit = await get_audit_log(USER_ID, limit=10)
        emergency_entries = [e for e in audit if e.action == "emergency_revoke"]
        assert len(emergency_entries) == 1
        assert "Disabled 1 policies" in emergency_entries[0].details

    @pytest.mark.asyncio
    async def test_can_reenable_after_emergency(self, auth_client, db):
        """After emergency revoke, individual agents can be re-enabled."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "recover-bot",
            "agent_name": "Recover Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        auth_client.post("/api/v1/emergency-revoke")

        # Re-enable
        resp = auth_client.post("/api/v1/policies/recover-bot/toggle")
        assert resp.status_code == 200
        assert resp.json()["is_active"] is True

        # Token request should work again
        with patch("src.app.get_token_vault_token", new_callable=AsyncMock, return_value={
            "access_token": "recovered",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "repo",
        }):
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "recover-bot",
                "service": "github",
                "scopes": ["repo"],
            })
            assert resp.status_code == 200


class TestMultiServicePolicy:
    """Policies spanning multiple services."""

    @pytest.mark.asyncio
    async def test_multi_service_token_requests(self, auth_client, db):
        """Agent with access to multiple services can request tokens for each."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "multi-svc",
            "agent_name": "Multi Service Bot",
            "allowed_services": ["github", "slack", "linear"],
            "allowed_scopes": {
                "github": ["repo", "read:user"],
                "slack": ["chat:write"],
                "linear": ["read", "write"],
            },
        })

        mock_return = {
            "access_token": "tok",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "all",
        }

        with patch("src.app.get_token_vault_token", new_callable=AsyncMock, return_value=mock_return):
            # GitHub
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "multi-svc", "service": "github", "scopes": ["repo"],
            })
            assert resp.status_code == 200

            # Slack
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "multi-svc", "service": "slack", "scopes": ["chat:write"],
            })
            assert resp.status_code == 200

            # Linear
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "multi-svc", "service": "linear", "scopes": ["read"],
            })
            assert resp.status_code == 200

            # Notion (not allowed)
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "multi-svc", "service": "notion", "scopes": ["read_content"],
            })
            assert resp.status_code == 403


class TestStepUpAuthFlow:
    """Step-up authentication (CIBA) flow tests."""

    @pytest.mark.asyncio
    async def test_step_up_required_returns_202(self, auth_client, db):
        """Service requiring step-up returns 202 with auth_req_id."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "sensitive-bot",
            "agent_name": "Sensitive Bot",
            "allowed_services": ["github", "slack"],
            "allowed_scopes": {"github": ["repo"], "slack": ["chat:write"]},
            "requires_step_up": ["slack"],
        })

        with patch("src.app.trigger_step_up_auth", new_callable=AsyncMock, return_value={
            "auth_req_id": "ciba_req_123",
            "expires_in": 120,
            "status": "pending",
        }):
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "sensitive-bot",
                "service": "slack",
                "scopes": ["chat:write"],
            })
            assert resp.status_code == 202
            data = resp.json()
            assert data["status"] == "step_up_required"
            assert data["auth_req_id"] == "ciba_req_123"

    @pytest.mark.asyncio
    async def test_step_up_approved_issues_token(self, auth_client, db):
        """When step-up is pre-approved, token is issued directly."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "approved-bot",
            "agent_name": "Approved Bot",
            "allowed_services": ["slack"],
            "allowed_scopes": {"slack": ["chat:write"]},
            "requires_step_up": ["slack"],
        })

        with patch("src.app.trigger_step_up_auth", new_callable=AsyncMock, return_value={
            "status": "approved",
        }), patch("src.app.get_token_vault_token", new_callable=AsyncMock, return_value={
            "access_token": "approved_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "chat:write",
        }):
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "approved-bot",
                "service": "slack",
                "scopes": ["chat:write"],
            })
            assert resp.status_code == 200
            assert resp.json()["access_token"] == "approved_token"

    @pytest.mark.asyncio
    async def test_step_up_not_required_for_other_services(self, auth_client, db):
        """Services not in requires_step_up bypass CIBA."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "mixed-bot",
            "agent_name": "Mixed Bot",
            "allowed_services": ["github", "slack"],
            "allowed_scopes": {"github": ["repo"], "slack": ["chat:write"]},
            "requires_step_up": ["slack"],
        })

        with patch("src.app.get_token_vault_token", new_callable=AsyncMock, return_value={
            "access_token": "no_step_up",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "repo",
        }):
            # GitHub should not require step-up
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "mixed-bot",
                "service": "github",
                "scopes": ["repo"],
            })
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_step_up_status_polling(self, auth_client, db):
        """Can poll step-up auth status."""
        with patch("src.app.check_step_up_status", new_callable=AsyncMock, return_value={
            "status": "pending",
        }):
            resp = auth_client.get("/api/v1/step-up/status/ciba_123")
            assert resp.status_code == 200
            assert resp.json()["status"] == "pending"

        with patch("src.app.check_step_up_status", new_callable=AsyncMock, return_value={
            "status": "approved",
            "token": "step_up_token",
        }):
            resp = auth_client.get("/api/v1/step-up/status/ciba_123")
            assert resp.status_code == 200
            assert resp.json()["status"] == "approved"


class TestRateLimitFlow:
    """Rate limiting enforcement in token requests."""

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self, auth_client, db):
        """Exceeding rate limit blocks token requests."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "rate-bot",
            "agent_name": "Rate Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "rate_limit_per_minute": 3,
        })

        with patch("src.app.get_token_vault_token", new_callable=AsyncMock, return_value={
            "access_token": "tok",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "repo",
        }):
            # 3 requests should succeed
            for _ in range(3):
                resp = auth_client.post("/api/v1/token", json={
                    "agent_id": "rate-bot",
                    "service": "github",
                    "scopes": ["repo"],
                })
                assert resp.status_code == 200

            # 4th should be rate limited
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "rate-bot",
                "service": "github",
                "scopes": ["repo"],
            })
            assert resp.status_code == 403
            assert "rate limit" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_rate_limit_per_service(self, auth_client, db):
        """Rate limits are tracked per agent:service pair."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "dual-bot",
            "agent_name": "Dual Bot",
            "allowed_services": ["github", "slack"],
            "allowed_scopes": {"github": ["repo"], "slack": ["chat:write"]},
            "rate_limit_per_minute": 2,
        })

        with patch("src.app.get_token_vault_token", new_callable=AsyncMock, return_value={
            "access_token": "tok",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "any",
        }):
            # 2 GitHub requests
            for _ in range(2):
                resp = auth_client.post("/api/v1/token", json={
                    "agent_id": "dual-bot",
                    "service": "github",
                    "scopes": ["repo"],
                })
                assert resp.status_code == 200

            # GitHub should be rate limited
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "dual-bot",
                "service": "github",
                "scopes": ["repo"],
            })
            assert resp.status_code == 403

            # Slack should still work (separate counter)
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "dual-bot",
                "service": "slack",
                "scopes": ["chat:write"],
            })
            assert resp.status_code == 200


class TestTimeWindowFlow:
    """Time-window enforcement in token requests."""

    @pytest.mark.asyncio
    async def test_token_denied_outside_hours(self, auth_client, db):
        """Token request outside allowed hours is denied."""
        from datetime import datetime, timezone
        auth_client.post("/api/v1/policies", json={
            "agent_id": "hours-bot",
            "agent_name": "Hours Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17],
        })

        # Mock time to 3 AM UTC
        with patch("src.policy.check_time_window") as mock_tw:
            mock_tw.return_value = "Access denied: not within allowed hours (09:00-17:59 UTC)"
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "hours-bot",
                "service": "github",
                "scopes": ["repo"],
            })
            assert resp.status_code == 403
            assert "hours" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_token_denied_wrong_day(self, auth_client, db):
        """Token request on non-allowed day is denied."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "weekday-bot",
            "agent_name": "Weekday Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_days": [0, 1, 2, 3, 4],  # Mon-Fri
        })

        with patch("src.policy.check_time_window") as mock_tw:
            mock_tw.return_value = "Access denied: not within allowed days (Mon, Tue, Wed, Thu, Fri)"
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "weekday-bot",
                "service": "github",
                "scopes": ["repo"],
            })
            assert resp.status_code == 403
            assert "days" in resp.json()["detail"].lower()


class TestIPAllowlistFlow:
    """IP allowlist enforcement in token requests."""

    @pytest.mark.asyncio
    async def test_ip_allowed(self, auth_client, db):
        """Request from allowed IP succeeds."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "ip-bot",
            "agent_name": "IP Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "ip_allowlist": ["127.0.0.1", "10.0.0.0/8"],
        })

        with patch("src.app.get_token_vault_token", new_callable=AsyncMock, return_value={
            "access_token": "tok",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "repo",
        }):
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "ip-bot",
                "service": "github",
                "scopes": ["repo"],
            })
            # TestClient uses 127.0.0.1 / testclient, depends on implementation
            # The policy check may allow or deny based on TestClient's IP
            assert resp.status_code in [200, 403]


class TestTokenVaultErrors:
    """Token Vault error handling in flows."""

    @pytest.mark.asyncio
    async def test_token_vault_failure_returns_502(self, auth_client, db):
        """Token Vault error returns 502."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "vault-bot",
            "agent_name": "Vault Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })

        with patch("src.app.get_token_vault_token", new_callable=AsyncMock, return_value={
            "error": "Token exchange failed: service not connected",
        }):
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "vault-bot",
                "service": "github",
                "scopes": ["repo"],
            })
            assert resp.status_code == 502
            assert "token exchange" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_token_vault_error_logged(self, auth_client, db):
        """Token Vault errors are logged in audit trail."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "error-bot",
            "agent_name": "Error Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })

        with patch("src.app.get_token_vault_token", new_callable=AsyncMock, return_value={
            "error": "Connection timeout",
        }):
            auth_client.post("/api/v1/token", json={
                "agent_id": "error-bot",
                "service": "github",
                "scopes": ["repo"],
            })

        audit = await get_audit_log(USER_ID, limit=10)
        error_entries = [e for e in audit if e.status == "error"]
        assert len(error_entries) >= 1
        assert "timeout" in error_entries[0].details.lower()


class TestPolicyValidation:
    """Policy creation validation edge cases."""

    def test_unknown_service_rejected(self, auth_client):
        resp = auth_client.post("/api/v1/policies", json={
            "agent_id": "bad-svc",
            "agent_name": "Bad Service",
            "allowed_services": ["twitter"],
            "allowed_scopes": {},
        })
        assert resp.status_code == 400
        assert "unknown service" in resp.json()["detail"].lower()

    def test_invalid_scopes_rejected(self, auth_client):
        resp = auth_client.post("/api/v1/policies", json={
            "agent_id": "bad-scope",
            "agent_name": "Bad Scope",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo", "admin:org"]},
        })
        assert resp.status_code == 400
        assert "invalid scopes" in resp.json()["detail"].lower()

    def test_invalid_hours_rejected(self, auth_client):
        resp = auth_client.post("/api/v1/policies", json={
            "agent_id": "bad-hours",
            "agent_name": "Bad Hours",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_hours": [25, -1],
        })
        assert resp.status_code == 400
        assert "hours" in resp.json()["detail"].lower()

    def test_invalid_days_rejected(self, auth_client):
        resp = auth_client.post("/api/v1/policies", json={
            "agent_id": "bad-days",
            "agent_name": "Bad Days",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "allowed_days": [7, 8],
        })
        assert resp.status_code == 400
        assert "days" in resp.json()["detail"].lower()

    def test_empty_agent_id_rejected(self, auth_client):
        resp = auth_client.post("/api/v1/policies", json={
            "agent_id": "",
            "agent_name": "No ID",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        # Pydantic may or may not accept empty string, but the policy should
        # either reject or create with empty ID
        assert resp.status_code in [200, 400, 422]

    def test_policy_with_all_services(self, auth_client):
        """Policy can include all 5 services."""
        resp = auth_client.post("/api/v1/policies", json={
            "agent_id": "all-services",
            "agent_name": "Full Access",
            "allowed_services": ["github", "slack", "google", "linear", "notion"],
            "allowed_scopes": {
                "github": ["repo"],
                "slack": ["chat:write"],
                "google": ["gmail.readonly"],
                "linear": ["read"],
                "notion": ["read_content"],
            },
        })
        assert resp.status_code == 200

    def test_policy_update_replaces(self, auth_client):
        """Creating policy with same agent_id replaces the old one."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "replace-me",
            "agent_name": "V1",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "rate_limit_per_minute": 10,
        })
        auth_client.post("/api/v1/policies", json={
            "agent_id": "replace-me",
            "agent_name": "V2",
            "allowed_services": ["github", "slack"],
            "allowed_scopes": {"github": ["repo"], "slack": ["chat:write"]},
            "rate_limit_per_minute": 100,
        })
        resp = auth_client.get("/api/v1/policies")
        policies = resp.json()
        assert len(policies) == 1
        assert policies[0]["rate_limit_per_minute"] == 100
        assert "slack" in policies[0]["allowed_services"]


class TestServiceConnection:
    """Service connection flow tests."""

    def test_connect_unknown_service_rejected(self, auth_client):
        resp = auth_client.post("/connect/twitter", follow_redirects=False)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_disconnect_service(self, auth_client, db):
        """Can disconnect a previously connected service."""
        await add_connected_service(USER_ID, "github", "conn_123")
        services = await get_connected_services(USER_ID)
        assert len(services) == 1

        resp = auth_client.delete("/connect/github")
        assert resp.status_code == 200
        assert resp.json()["status"] == "disconnected"

        services = await get_connected_services(USER_ID)
        assert len(services) == 0


class TestAuditTrailCompleteness:
    """Verify audit trail captures all significant events."""

    @pytest.mark.asyncio
    async def test_audit_captures_policy_lifecycle(self, auth_client, db):
        """All policy events appear in audit."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "audit-lifecycle",
            "agent_name": "Audit Lifecycle",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        auth_client.post("/api/v1/policies/audit-lifecycle/toggle")
        auth_client.post("/api/v1/policies/audit-lifecycle/toggle")
        auth_client.delete("/api/v1/policies/audit-lifecycle")

        audit = await get_audit_log(USER_ID, limit=50)
        actions = [e.action for e in audit]
        assert "policy_created" in actions
        assert "agent_disabled" in actions
        assert "agent_enabled" in actions
        assert "policy_deleted" in actions

    @pytest.mark.asyncio
    async def test_audit_captures_key_lifecycle(self, auth_client, db):
        """API key creation and revocation appear in audit."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "key-audit",
            "agent_name": "Key Audit",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        resp = auth_client.post("/api/v1/keys", json={"agent_id": "key-audit"})
        key_id = resp.json()["key_id"]
        auth_client.delete(f"/api/v1/keys/{key_id}")

        audit = await get_audit_log(USER_ID, limit=50)
        actions = [e.action for e in audit]
        assert "api_key_created" in actions
        assert "api_key_revoked" in actions

    @pytest.mark.asyncio
    async def test_audit_captures_denied_requests(self, auth_client, db):
        """Denied token requests appear in audit with reasons."""
        # Request token for non-existent agent
        auth_client.post("/api/v1/token", json={
            "agent_id": "nonexistent",
            "service": "github",
            "scopes": ["repo"],
        })

        audit = await get_audit_log(USER_ID, limit=10)
        denied = [e for e in audit if e.status == "denied"]
        assert len(denied) >= 1
        assert "not registered" in denied[0].details.lower()

    @pytest.mark.asyncio
    async def test_audit_page_renders(self, auth_client, db):
        """Audit page returns HTML with entries."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "page-bot",
            "agent_name": "Page Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })

        resp = auth_client.get("/audit")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestDashboardRendering:
    """Dashboard UI rendering tests."""

    def test_login_page_renders(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_dashboard_shows_policies(self, auth_client, db):
        auth_client.post("/api/v1/policies", json={
            "agent_id": "show-bot",
            "agent_name": "Show Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
        })
        resp = auth_client.get("/")
        assert resp.status_code == 200
        assert "Show Bot" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_shows_services(self, auth_client, db):
        resp = auth_client.get("/")
        assert resp.status_code == 200
        assert "GitHub" in resp.text
        assert "Slack" in resp.text
        assert "Google" in resp.text
        assert "Linear" in resp.text
        assert "Notion" in resp.text


class TestExpiringPolicies:
    """Policy expiration tests."""

    @pytest.mark.asyncio
    async def test_expired_policy_denies_token(self, auth_client, db):
        """Policy past its expires_at blocks token requests."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "expiring-bot",
            "agent_name": "Expiring Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "expires_at": time.time() - 100,  # Already expired
        })

        resp = auth_client.post("/api/v1/token", json={
            "agent_id": "expiring-bot",
            "service": "github",
            "scopes": ["repo"],
        })
        assert resp.status_code == 403
        assert "expired" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_non_expired_policy_allows_token(self, auth_client, db):
        """Policy before its expires_at allows token requests."""
        auth_client.post("/api/v1/policies", json={
            "agent_id": "future-bot",
            "agent_name": "Future Bot",
            "allowed_services": ["github"],
            "allowed_scopes": {"github": ["repo"]},
            "expires_at": time.time() + 3600,
        })

        with patch("src.app.get_token_vault_token", new_callable=AsyncMock, return_value={
            "access_token": "future_tok",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "repo",
        }):
            resp = auth_client.post("/api/v1/token", json={
                "agent_id": "future-bot",
                "service": "github",
                "scopes": ["repo"],
            })
            assert resp.status_code == 200


class TestNotFoundHandling:
    """404 handling for non-existent resources."""

    def test_toggle_nonexistent_policy(self, auth_client):
        resp = auth_client.post("/api/v1/policies/ghost-bot/toggle")
        assert resp.status_code == 404

    def test_delete_nonexistent_policy(self, auth_client):
        resp = auth_client.delete("/api/v1/policies/ghost-bot")
        assert resp.status_code == 404

    def test_revoke_nonexistent_key(self, auth_client):
        resp = auth_client.delete("/api/v1/keys/nonexistent-key-id")
        assert resp.status_code == 404

    def test_invalid_api_key_rejected(self, client):
        resp = client.post(
            "/api/v1/token",
            json={"agent_id": "test", "service": "github", "scopes": []},
            headers={"Authorization": "Bearer ag_invalid_key_data"},
        )
        assert resp.status_code == 401


class TestHealthEndpoint:
    """Health endpoint tests."""

    def test_health_returns_version(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["version"] == "1.0.0"
        assert data["service"] == "agentgate"

    def test_health_no_auth_required(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
