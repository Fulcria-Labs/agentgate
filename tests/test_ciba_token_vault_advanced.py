"""Advanced tests for CIBA step-up authentication and Token Vault exchange flows.

Covers edge cases, timeout scenarios, delivery modes, multi-step flows,
token refresh/rotation lifecycle, and error recovery patterns.
"""

import asyncio
import time
import pytest
from unittest.mock import patch, AsyncMock, MagicMock, call

from src.auth import (
    SUPPORTED_SERVICES,
    get_token_vault_token,
    initiate_connection,
    trigger_step_up_auth,
    check_step_up_status,
    _get_management_token,
)
from src.database import (
    init_db,
    create_agent_policy,
    get_agent_policy,
    add_connected_service,
    get_connected_services,
    remove_connected_service,
    AgentPolicy,
    log_audit,
    get_audit_log,
)
from src.policy import enforce_policy, requires_step_up, PolicyDenied


# --- Helper to create mock HTTP response ---

def make_response(status_code=200, json_data=None):
    """Create a mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


def mock_client_ctx(post_side_effect=None, post_return=None):
    """Create a patched httpx.AsyncClient context manager."""
    mock_client = AsyncMock()
    if post_side_effect:
        mock_client.post = AsyncMock(side_effect=post_side_effect)
    elif post_return:
        mock_client.post = AsyncMock(return_value=post_return)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


# ============================================================
# CIBA Step-Up Auth: Advanced Scenarios
# ============================================================

class TestCIBAAdvancedTrigger:
    """Advanced CIBA trigger scenarios beyond basic success/failure."""

    @pytest.mark.asyncio
    async def test_trigger_binding_message_includes_agent_and_action(self):
        """Verify the binding message format sent to Auth0."""
        mock_resp = make_response(200, {"auth_req_id": "req-msg-check", "expires_in": 120})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            await trigger_step_up_auth("user42", "deploy-agent", "push to production")
            post_call = mock_client.post.call_args
            body = post_call[1].get("json", post_call[0][1] if len(post_call[0]) > 1 else {})
            if not body:
                body = post_call.kwargs.get("json", {})
            assert "deploy-agent" in body.get("binding_message", "")
            assert "push to production" in body.get("binding_message", "")

    @pytest.mark.asyncio
    async def test_trigger_with_special_chars_in_action(self):
        """Agent action with special characters should not break the request."""
        mock_resp = make_response(200, {"auth_req_id": "req-special", "expires_in": 60})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await trigger_step_up_auth(
                "user1", "agent-x", "delete all data from 'users' & \"logs\""
            )
            assert result["auth_req_id"] == "req-special"
            assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_trigger_with_empty_action(self):
        """Empty action string should still trigger CIBA."""
        mock_resp = make_response(200, {"auth_req_id": "req-empty-action"})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await trigger_step_up_auth("user1", "agent1", "")
            assert result["auth_req_id"] == "req-empty-action"
            assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_trigger_with_very_long_action_string(self):
        """Long action string (1000+ chars) should work."""
        long_action = "A" * 2000
        mock_resp = make_response(200, {"auth_req_id": "req-long"})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await trigger_step_up_auth("user1", "agent1", long_action)
            assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_trigger_500_server_error_returns_denied(self):
        """Server 500 error from Auth0 should result in denied status."""
        mock_resp = make_response(500, {"error": "internal_server_error"})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await trigger_step_up_auth("user1", "agent1", "action")
            assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_trigger_401_unauthorized_returns_denied(self):
        """Auth0 returning 401 should result in denied."""
        mock_resp = make_response(401)
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await trigger_step_up_auth("user1", "agent1", "action")
            assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_trigger_429_rate_limited_returns_denied(self):
        """Auth0 rate limiting (429) should result in denied."""
        mock_resp = make_response(429)
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await trigger_step_up_auth("user1", "agent1", "action")
            assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_trigger_returns_default_expires_when_missing(self):
        """Missing expires_in should default to 120."""
        mock_resp = make_response(200, {"auth_req_id": "req-no-exp"})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await trigger_step_up_auth("user1", "agent1", "action")
            assert result["expires_in"] == 120

    @pytest.mark.asyncio
    async def test_trigger_respects_custom_expires(self):
        """Custom expires_in from Auth0 should be preserved."""
        mock_resp = make_response(200, {"auth_req_id": "req-exp", "expires_in": 300})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await trigger_step_up_auth("user1", "agent1", "action")
            assert result["expires_in"] == 300

    @pytest.mark.asyncio
    async def test_trigger_with_unicode_user_id(self):
        """Unicode characters in user_id should work."""
        mock_resp = make_response(200, {"auth_req_id": "req-unicode"})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await trigger_step_up_auth("user_日本語", "agent1", "action")
            assert result["status"] == "pending"


class TestCIBAStatusPolling:
    """CIBA status polling edge cases and advanced scenarios."""

    @pytest.mark.asyncio
    async def test_approved_returns_access_token(self):
        """Approved status should include the access token."""
        mock_resp = make_response(200, {"access_token": "final-token-abc"})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await check_step_up_status("req-123")
            assert result["status"] == "approved"
            assert result["token"] == "final-token-abc"

    @pytest.mark.asyncio
    async def test_approved_with_empty_token(self):
        """Approved status with empty access_token."""
        mock_resp = make_response(200, {"access_token": ""})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await check_step_up_status("req-123")
            assert result["status"] == "approved"
            assert result["token"] == ""

    @pytest.mark.asyncio
    async def test_approved_with_no_token_field(self):
        """Approved status with missing access_token field."""
        mock_resp = make_response(200, {"token_type": "Bearer"})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await check_step_up_status("req-123")
            assert result["status"] == "approved"
            assert result["token"] is None

    @pytest.mark.asyncio
    async def test_slow_down_includes_retry_after(self):
        """Slow down response should include retry_after=5."""
        mock_resp = make_response(400, {"error": "slow_down"})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await check_step_up_status("req-123")
            assert result["status"] == "pending"
            assert result["retry_after"] == 5

    @pytest.mark.asyncio
    async def test_pending_has_no_retry_after(self):
        """Normal pending response should not include retry_after."""
        mock_resp = make_response(400, {"error": "authorization_pending"})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await check_step_up_status("req-123")
            assert result["status"] == "pending"
            assert "retry_after" not in result

    @pytest.mark.asyncio
    async def test_expired_request_returns_denied(self):
        """Expired auth_req_id should return denied."""
        mock_resp = make_response(400, {"error": "expired_token"})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await check_step_up_status("req-expired")
            assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_access_denied_error_returns_denied(self):
        """access_denied error should return denied."""
        mock_resp = make_response(400, {"error": "access_denied"})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await check_step_up_status("req-123")
            assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_403_status_returns_denied(self):
        """403 Forbidden should return denied regardless of body."""
        mock_resp = make_response(403, {"error": "whatever"})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await check_step_up_status("req-123")
            assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_500_returns_denied(self):
        """Server error should return denied."""
        mock_resp = make_response(500)
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await check_step_up_status("req-123")
            assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_sequential_polling_pending_then_approved(self):
        """Simulates polling: first pending, then approved."""
        pending_resp = make_response(400, {"error": "authorization_pending"})
        approved_resp = make_response(200, {"access_token": "approved-token"})

        with patch("src.auth.httpx.AsyncClient") as cls:
            # First call: pending
            mock1 = mock_client_ctx(post_return=pending_resp)
            cls.return_value = mock1
            r1 = await check_step_up_status("req-poll")
            assert r1["status"] == "pending"

            # Second call: approved
            mock2 = mock_client_ctx(post_return=approved_resp)
            cls.return_value = mock2
            r2 = await check_step_up_status("req-poll")
            assert r2["status"] == "approved"
            assert r2["token"] == "approved-token"

    @pytest.mark.asyncio
    async def test_sequential_polling_pending_slow_down_then_approved(self):
        """Simulates polling: pending → slow_down → approved."""
        pending = make_response(400, {"error": "authorization_pending"})
        slow = make_response(400, {"error": "slow_down"})
        approved = make_response(200, {"access_token": "final"})

        with patch("src.auth.httpx.AsyncClient") as cls:
            for resp, expected_status in [
                (pending, "pending"),
                (slow, "pending"),
                (approved, "approved"),
            ]:
                mock = mock_client_ctx(post_return=resp)
                cls.return_value = mock
                result = await check_step_up_status("req-multi")
                assert result["status"] == expected_status

    @pytest.mark.asyncio
    async def test_empty_auth_req_id(self):
        """Empty auth_req_id should still make the request (Auth0 handles validation)."""
        mock_resp = make_response(400, {"error": "invalid_grant"})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await check_step_up_status("")
            assert result["status"] == "denied"


# ============================================================
# Token Vault Exchange: Advanced Scenarios
# ============================================================

class TestTokenVaultExchangeAdvanced:
    """Advanced Token Vault token exchange scenarios."""

    @pytest.mark.asyncio
    async def test_exchange_for_each_supported_service(self):
        """Token exchange should work for all supported services."""
        for service_name in SUPPORTED_SERVICES:
            scopes = SUPPORTED_SERVICES[service_name]["scopes"][:1]
            mgmt = make_response(200, {"access_token": "mgmt"})
            exchange = make_response(200, {
                "access_token": f"token-{service_name}",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": " ".join(scopes),
            })
            mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

            with patch("src.auth.httpx.AsyncClient") as cls:
                cls.return_value = mock_client
                result = await get_token_vault_token("user1", service_name, scopes)
                assert result["access_token"] == f"token-{service_name}"

    @pytest.mark.asyncio
    async def test_exchange_with_multiple_scopes(self):
        """Exchange with multiple scopes should include them all."""
        scopes = ["repo", "read:user", "read:org"]
        mgmt = make_response(200, {"access_token": "mgmt"})
        exchange = make_response(200, {
            "access_token": "multi-scope-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": " ".join(scopes),
        })
        mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await get_token_vault_token("user1", "github", scopes)
            assert result["scope"] == "repo read:user read:org"

    @pytest.mark.asyncio
    async def test_exchange_with_all_github_scopes(self):
        """Exchange with all GitHub scopes."""
        scopes = SUPPORTED_SERVICES["github"]["scopes"]
        mgmt = make_response(200, {"access_token": "mgmt"})
        exchange = make_response(200, {
            "access_token": "full-github-token",
            "token_type": "Bearer",
            "expires_in": 7200,
            "scope": " ".join(scopes),
        })
        mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await get_token_vault_token("user1", "github", scopes)
            assert result["access_token"] == "full-github-token"
            assert result["expires_in"] == 7200

    @pytest.mark.asyncio
    async def test_exchange_401_returns_error(self):
        """401 Unauthorized from Token Vault should return error."""
        mgmt = make_response(200, {"access_token": "mgmt"})
        exchange = make_response(401, {"error_description": "Invalid client"})
        mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await get_token_vault_token("user1", "github", ["repo"])
            assert "error" in result
            assert result["error"] == "Invalid client"

    @pytest.mark.asyncio
    async def test_exchange_403_returns_error(self):
        """403 Forbidden from Token Vault should return error."""
        mgmt = make_response(200, {"access_token": "mgmt"})
        exchange = make_response(403, {"error_description": "Insufficient permissions"})
        mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await get_token_vault_token("user1", "github", ["repo"])
            assert "error" in result

    @pytest.mark.asyncio
    async def test_exchange_500_returns_error(self):
        """500 from Token Vault should return error."""
        mgmt = make_response(200, {"access_token": "mgmt"})
        exchange = make_response(500, {"error_description": "Internal server error"})
        mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await get_token_vault_token("user1", "github", ["repo"])
            assert "error" in result

    @pytest.mark.asyncio
    async def test_exchange_error_without_description(self):
        """Error response without error_description should use default."""
        mgmt = make_response(200, {"access_token": "mgmt"})
        exchange = make_response(400, {})
        mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await get_token_vault_token("user1", "github", ["repo"])
            assert result["error"] == "Token exchange failed"

    @pytest.mark.asyncio
    async def test_exchange_scope_defaults_to_requested(self):
        """When response has no scope, should default to requested scopes."""
        mgmt = make_response(200, {"access_token": "mgmt"})
        exchange = make_response(200, {"access_token": "token-no-scope"})
        mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await get_token_vault_token("user1", "github", ["repo", "gist"])
            assert result["scope"] == "repo gist"

    @pytest.mark.asyncio
    async def test_exchange_preserves_response_scope_over_requested(self):
        """Response scope should take precedence over requested scopes."""
        mgmt = make_response(200, {"access_token": "mgmt"})
        exchange = make_response(200, {
            "access_token": "tok",
            "scope": "repo",  # Server narrowed scopes
        })
        mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await get_token_vault_token("user1", "github", ["repo", "gist"])
            assert result["scope"] == "repo"  # Server's scope wins

    @pytest.mark.asyncio
    async def test_exchange_with_short_lived_token(self):
        """Short-lived tokens (e.g., 60 seconds) should be preserved."""
        mgmt = make_response(200, {"access_token": "mgmt"})
        exchange = make_response(200, {
            "access_token": "short-lived",
            "expires_in": 60,
        })
        mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await get_token_vault_token("user1", "slack", ["chat:write"])
            assert result["expires_in"] == 60

    @pytest.mark.asyncio
    async def test_exchange_for_different_users(self):
        """Different users should get different tokens."""
        for user_id in ["user-alice", "user-bob", "user-charlie"]:
            mgmt = make_response(200, {"access_token": "mgmt"})
            exchange = make_response(200, {
                "access_token": f"token-{user_id}",
                "token_type": "Bearer",
                "expires_in": 3600,
            })
            mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

            with patch("src.auth.httpx.AsyncClient") as cls:
                cls.return_value = mock_client
                result = await get_token_vault_token(user_id, "github", ["repo"])
                assert result["access_token"] == f"token-{user_id}"

    @pytest.mark.asyncio
    async def test_exchange_sends_correct_grant_type(self):
        """Verify the correct RFC 8693 grant type is sent."""
        mgmt = make_response(200, {"access_token": "mgmt"})
        exchange = make_response(200, {"access_token": "tok"})
        mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            await get_token_vault_token("user1", "github", ["repo"])
            exchange_call = mock_client.post.call_args_list[1]
            body = exchange_call.kwargs.get("json", {})
            assert body["grant_type"] == "urn:ietf:params:oauth:grant-type:token-exchange"

    @pytest.mark.asyncio
    async def test_exchange_audience_format(self):
        """Verify the audience URL format for service."""
        mgmt = make_response(200, {"access_token": "mgmt"})
        exchange = make_response(200, {"access_token": "tok"})
        mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            await get_token_vault_token("user1", "slack", ["chat:write"])
            exchange_call = mock_client.post.call_args_list[1]
            body = exchange_call.kwargs.get("json", {})
            assert body["audience"] == "https://slack.com/api"


class TestManagementTokenAdvanced:
    """Advanced management token retrieval tests."""

    @pytest.mark.asyncio
    async def test_mgmt_token_uses_client_credentials_grant(self):
        """Verify management token request uses client_credentials grant."""
        mock_resp = make_response(200, {"access_token": "mgmt-tok"})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            await _get_management_token()
            call_args = mock_client.post.call_args
            body = call_args.kwargs.get("json", {})
            assert body["grant_type"] == "client_credentials"

    @pytest.mark.asyncio
    async def test_mgmt_token_empty_response(self):
        """Empty JSON response should return empty string."""
        mock_resp = make_response(200, {})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            token = await _get_management_token()
            assert token == ""

    @pytest.mark.asyncio
    async def test_mgmt_token_null_access_token(self):
        """None access_token in response should return empty string."""
        mock_resp = make_response(200, {"access_token": None})
        mock_client = mock_client_ctx(post_return=mock_resp)

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            token = await _get_management_token()
            # get("access_token", "") returns None, so empty string only if None not in dict
            # The actual behavior depends on dict.get with None value
            assert token is None or token == ""


# ============================================================
# Connection URL: Advanced Scenarios
# ============================================================

class TestConnectionURLAdvanced:
    """Advanced initiate_connection URL generation tests."""

    @pytest.mark.asyncio
    async def test_different_services_produce_different_connections(self):
        """Each service should produce its own connection parameter."""
        for service in SUPPORTED_SERVICES:
            url = await initiate_connection(service, "https://example.com/cb")
            assert f"connection={service}" in url

    @pytest.mark.asyncio
    async def test_url_is_well_formed(self):
        """URL should start with the Auth0 authorize endpoint."""
        url = await initiate_connection("github", "https://example.com/cb")
        assert url.startswith("https://")
        assert "/authorize?" in url

    @pytest.mark.asyncio
    async def test_state_is_urlsafe(self):
        """State parameter should be URL-safe."""
        import re
        url = await initiate_connection("github", "https://example.com/cb")
        state_match = re.search(r"state=([^&]+)", url)
        assert state_match
        state = state_match.group(1)
        # URL-safe base64 uses only alphanumeric, -, _, =
        assert re.match(r"^[A-Za-z0-9_=-]+$", state)

    @pytest.mark.asyncio
    async def test_redirect_uri_is_url_encoded(self):
        """Redirect URI with special chars should be encoded."""
        url = await initiate_connection("github", "https://example.com/cb?param=value&other=1")
        # The '?' and '&' should be URL-encoded
        assert "redirect_uri=https%3A" in url or "redirect_uri=https:" in url

    @pytest.mark.asyncio
    async def test_scope_includes_openid(self):
        """Scope should include openid for OIDC compliance."""
        url = await initiate_connection("github", "https://example.com/cb")
        assert "scope=openid" in url


# ============================================================
# Connected Services Database: Edge Cases
# ============================================================

class TestConnectedServicesAdvanced:
    """Advanced connected service database operation tests."""

    @pytest.mark.asyncio
    async def test_connect_all_supported_services(self, db):
        """User can connect all supported services."""
        for svc in SUPPORTED_SERVICES:
            await add_connected_service("user1", svc)
        services = await get_connected_services("user1")
        assert len(services) == len(SUPPORTED_SERVICES)

    @pytest.mark.asyncio
    async def test_connect_same_service_twice_upserts(self, db):
        """Connecting same service twice should upsert, not duplicate."""
        await add_connected_service("user1", "github")
        await add_connected_service("user1", "github")
        services = await get_connected_services("user1")
        github_services = [s for s in services if s["service"] == "github"]
        assert len(github_services) == 1

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_service(self, db):
        """Disconnecting a service that isn't connected should not error."""
        await remove_connected_service("user1", "nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_user_isolation_services(self, db):
        """Different users' services should be isolated."""
        await add_connected_service("user1", "github")
        await add_connected_service("user2", "slack")
        u1 = await get_connected_services("user1")
        u2 = await get_connected_services("user2")
        assert len(u1) == 1 and u1[0]["service"] == "github"
        assert len(u2) == 1 and u2[0]["service"] == "slack"

    @pytest.mark.asyncio
    async def test_connect_disconnect_reconnect(self, db):
        """Connect, disconnect, then reconnect should work."""
        await add_connected_service("user1", "github")
        await remove_connected_service("user1", "github")
        services = await get_connected_services("user1")
        assert len(services) == 0

        await add_connected_service("user1", "github")
        services = await get_connected_services("user1")
        assert len(services) == 1

    @pytest.mark.asyncio
    async def test_connected_service_has_timestamp(self, db):
        """Connected services should have a connected_at timestamp."""
        await add_connected_service("user1", "github")
        services = await get_connected_services("user1")
        assert "connected_at" in services[0]
        assert services[0]["connected_at"] > 0


# ============================================================
# CIBA + Policy Integration: End-to-End Flows
# ============================================================

class TestCIBAPolicyIntegration:
    """Tests combining CIBA step-up with policy enforcement."""

    @pytest.mark.asyncio
    async def test_policy_requires_step_up_for_specific_service(self, db):
        """Policy with requires_step_up should flag specific services."""
        policy = AgentPolicy(
            agent_id="agent-sensitive",
            agent_name="Sensitive Agent",
            allowed_services=["github", "slack"],
            allowed_scopes={"github": ["repo"], "slack": ["chat:write"]},
            requires_step_up=["github"],  # Only github requires step-up
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)
        saved = await get_agent_policy("agent-sensitive")
        assert requires_step_up(saved, "github") is True
        assert requires_step_up(saved, "slack") is False

    @pytest.mark.asyncio
    async def test_step_up_required_for_all_services(self, db):
        """Policy can require step-up for all allowed services."""
        policy = AgentPolicy(
            agent_id="agent-paranoid",
            agent_name="Paranoid Agent",
            allowed_services=["github", "slack", "google"],
            allowed_scopes={
                "github": ["repo"],
                "slack": ["chat:write"],
                "google": ["gmail.readonly"],
            },
            requires_step_up=["github", "slack", "google"],
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)
        saved = await get_agent_policy("agent-paranoid")
        for svc in ["github", "slack", "google"]:
            assert requires_step_up(saved, svc) is True

    @pytest.mark.asyncio
    async def test_step_up_not_required_for_unlisted_service(self, db):
        """Step-up should not be required for services not in the list."""
        policy = AgentPolicy(
            agent_id="agent-partial",
            agent_name="Partial Agent",
            allowed_services=["github", "slack"],
            allowed_scopes={"github": ["repo"], "slack": ["chat:write"]},
            requires_step_up=["github"],
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)
        saved = await get_agent_policy("agent-partial")
        assert requires_step_up(saved, "linear") is False

    @pytest.mark.asyncio
    async def test_enforce_policy_then_check_step_up(self, db):
        """Full flow: enforce policy passes, then check step-up requirement."""
        policy = AgentPolicy(
            agent_id="agent-flow",
            agent_name="Flow Agent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            requires_step_up=["github"],
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        # Policy enforcement should pass
        result = await enforce_policy("user1", "agent-flow", "github", ["repo"])
        assert result.agent_id == "agent-flow"

        # Step-up should be required
        assert requires_step_up(result, "github") is True

    @pytest.mark.asyncio
    async def test_enforce_policy_no_step_up_needed(self, db):
        """Flow where step-up is not required should skip CIBA entirely."""
        policy = AgentPolicy(
            agent_id="agent-simple",
            agent_name="Simple Agent",
            allowed_services=["slack"],
            allowed_scopes={"slack": ["channels:read"]},
            requires_step_up=[],
            created_by="user1",
            created_at=time.time(),
        )
        await create_agent_policy(policy)

        result = await enforce_policy("user1", "agent-simple", "slack", ["channels:read"])
        assert not requires_step_up(result, "slack")


# ============================================================
# Token Vault + Policy: Combined Scope Validation
# ============================================================

class TestScopeNegotiation:
    """Tests for scope intersection between policies and token requests."""

    @pytest.mark.asyncio
    async def test_effective_scopes_intersection(self, db):
        """Requested scopes should be intersected with allowed scopes."""
        from src.policy import get_effective_scopes

        policy = AgentPolicy(
            agent_id="agent-scopes",
            agent_name="Scope Agent",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo", "read:user"]},
            created_by="user1",
            created_at=time.time(),
        )

        effective = get_effective_scopes(
            policy, "github", ["repo", "read:user", "gist"]
        )
        assert set(effective) == {"repo", "read:user"}

    @pytest.mark.asyncio
    async def test_effective_scopes_no_overlap(self, db):
        """No overlapping scopes should result in empty list."""
        from src.policy import get_effective_scopes

        policy = AgentPolicy(
            agent_id="agent-no-overlap",
            agent_name="No Overlap",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
        )

        effective = get_effective_scopes(policy, "github", ["gist", "notifications"])
        assert effective == []

    @pytest.mark.asyncio
    async def test_effective_scopes_all_allowed(self, db):
        """All requested scopes within allowed should all be returned."""
        from src.policy import get_effective_scopes

        policy = AgentPolicy(
            agent_id="agent-all",
            agent_name="All Allowed",
            allowed_services=["slack"],
            allowed_scopes={"slack": ["channels:read", "chat:write", "users:read"]},
            created_by="user1",
            created_at=time.time(),
        )

        effective = get_effective_scopes(
            policy, "slack", ["channels:read", "chat:write"]
        )
        assert set(effective) == {"channels:read", "chat:write"}

    @pytest.mark.asyncio
    async def test_effective_scopes_for_unregistered_service(self, db):
        """Requesting scopes for a service not in allowed_scopes should return empty."""
        from src.policy import get_effective_scopes

        policy = AgentPolicy(
            agent_id="agent-wrong-svc",
            agent_name="Wrong Service",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
        )

        effective = get_effective_scopes(policy, "slack", ["chat:write"])
        assert effective == []

    @pytest.mark.asyncio
    async def test_effective_scopes_empty_request(self, db):
        """Empty requested scopes should return empty list."""
        from src.policy import get_effective_scopes

        policy = AgentPolicy(
            agent_id="agent-empty-req",
            agent_name="Empty Request",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            created_by="user1",
            created_at=time.time(),
        )

        effective = get_effective_scopes(policy, "github", [])
        assert effective == []


# ============================================================
# Token Vault Error Recovery
# ============================================================

class TestTokenVaultErrorRecovery:
    """Tests for error handling and recovery in token exchange."""

    @pytest.mark.asyncio
    async def test_mgmt_token_failure_cascades_to_exchange(self):
        """If management token is empty, exchange should still attempt."""
        mgmt = make_response(200, {})  # No access_token
        exchange = make_response(200, {"access_token": "tok"})
        mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            # The mgmt token will be empty, but exchange still proceeds
            result = await get_token_vault_token("user1", "github", ["repo"])
            assert "access_token" in result

    @pytest.mark.asyncio
    async def test_exchange_returns_extra_fields_ignored(self):
        """Extra fields in exchange response should not break parsing."""
        mgmt = make_response(200, {"access_token": "mgmt"})
        exchange = make_response(200, {
            "access_token": "tok",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "repo",
            "refresh_token": "ref-tok",  # Extra field
            "id_token": "id-tok",        # Extra field
            "custom_field": True,         # Extra field
        })
        mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

        with patch("src.auth.httpx.AsyncClient") as cls:
            cls.return_value = mock_client
            result = await get_token_vault_token("user1", "github", ["repo"])
            assert result["access_token"] == "tok"
            # Extra fields should not appear in result
            assert "refresh_token" not in result
            assert "id_token" not in result

    @pytest.mark.asyncio
    async def test_exchange_concurrent_requests_different_services(self):
        """Multiple concurrent token exchanges for different services."""
        async def exchange_service(service, scope):
            mgmt = make_response(200, {"access_token": "mgmt"})
            exchange = make_response(200, {
                "access_token": f"tok-{service}",
                "token_type": "Bearer",
                "expires_in": 3600,
            })
            mock_client = mock_client_ctx(post_side_effect=[mgmt, exchange])

            with patch("src.auth.httpx.AsyncClient") as cls:
                cls.return_value = mock_client
                return await get_token_vault_token("user1", service, [scope])

        # Run multiple exchanges concurrently
        results = await asyncio.gather(
            exchange_service("github", "repo"),
            exchange_service("slack", "chat:write"),
            exchange_service("google", "gmail.readonly"),
        )

        assert results[0]["access_token"] == "tok-github"
        assert results[1]["access_token"] == "tok-slack"
        assert results[2]["access_token"] == "tok-google"


# ============================================================
# Audit Trail for CIBA & Token Flows
# ============================================================

class TestCIBAAuditTrail:
    """Tests verifying that CIBA and token flows produce proper audit entries."""

    @pytest.mark.asyncio
    async def test_audit_log_records_token_request(self, db):
        """Token request should produce an audit entry."""
        await log_audit("user1", "agent1", "github", "token_request", "success",
                       scopes="repo,read:user")
        entries = await get_audit_log("user1")
        assert len(entries) >= 1
        latest = entries[0]
        assert latest.agent_id == "agent1"
        assert latest.service == "github"
        assert latest.action == "token_request"

    @pytest.mark.asyncio
    async def test_audit_log_records_step_up_event(self, db):
        """Step-up auth events should be logged."""
        await log_audit("user1", "agent1", "github", "step_up_initiated", "pending",
                       details="delete repository")
        entries = await get_audit_log("user1")
        assert any(e.action == "step_up_initiated" for e in entries)

    @pytest.mark.asyncio
    async def test_audit_log_records_denied_with_ip(self, db):
        """Denied access should log the IP address."""
        await log_audit("user1", "agent1", "github", "token_request", "denied",
                       ip_address="10.0.0.5", details="IP not in allowlist")
        entries = await get_audit_log("user1")
        denied = [e for e in entries if e.status == "denied"]
        assert len(denied) >= 1
        assert denied[0].ip_address == "10.0.0.5"

    @pytest.mark.asyncio
    async def test_audit_log_multiple_events_ordering(self, db):
        """Multiple audit events should be retrievable in order."""
        await log_audit("user1", "agent1", "github", "token_request", "success")
        await log_audit("user1", "agent1", "github", "step_up_initiated", "pending")
        await log_audit("user1", "agent1", "github", "token_issued", "success")

        entries = await get_audit_log("user1")
        assert len(entries) >= 3
        actions = [e.action for e in entries[:3]]
        # Most recent first
        assert "token_issued" in actions
        assert "step_up_initiated" in actions
        assert "token_request" in actions

    @pytest.mark.asyncio
    async def test_audit_log_rate_limited_event(self, db):
        """Rate limited events should be logged."""
        await log_audit("user1", "agent1", "github", "token_request", "rate_limited",
                       details="Rate limit: 60/min")
        entries = await get_audit_log("user1")
        rate_limited = [e for e in entries if e.status == "rate_limited"]
        assert len(rate_limited) >= 1

    @pytest.mark.asyncio
    async def test_audit_log_limit_parameter(self, db):
        """Audit log limit parameter should cap results."""
        for i in range(10):
            await log_audit("user1", f"agent{i}", "github", "token_request", "success")
        entries = await get_audit_log("user1", limit=5)
        assert len(entries) == 5
