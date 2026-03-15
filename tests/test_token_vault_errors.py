"""Token Vault error scenario tests for AgentGate.

Covers refresh token expiry, token rotation lifecycle, scope mismatch,
service offline/unreachable, Token Vault connection failures,
and various error recovery patterns.
"""

import asyncio
import time
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

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
    log_audit,
    get_audit_log,
    AgentPolicy,
)
from src.policy import (
    enforce_policy,
    PolicyDenied,
    get_effective_scopes,
    requires_step_up,
    _rate_counters,
)


@pytest.fixture(autouse=True)
def clear_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


USER = "auth0|token-vault-err-user"


def make_response(status_code=200, json_data=None):
    """Create a mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


def mock_client(post_side_effect=None, post_return=None):
    """Create a patched httpx.AsyncClient context manager."""
    client = AsyncMock()
    if post_side_effect:
        client.post = AsyncMock(side_effect=post_side_effect)
    elif post_return:
        client.post = AsyncMock(return_value=post_return)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


# ============================================================
# 1. Refresh Token Expiry Scenarios
# ============================================================

class TestRefreshTokenExpiry:
    """Token Vault errors when refresh tokens have expired."""

    @pytest.mark.asyncio
    async def test_refresh_token_expired_error(self):
        """Token exchange fails when refresh token has expired."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(400, {"error_description": "Refresh token has expired"}),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", ["repo"])
            assert "error" in result
            assert "expired" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_refresh_token_revoked_error(self):
        """Token exchange fails when refresh token has been revoked."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(403, {"error_description": "Refresh token has been revoked"}),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", ["repo"])
            assert "error" in result
            assert "revoked" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_refresh_token_invalid_grant(self):
        """Token exchange returns invalid_grant when refresh token is bad."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(400, {"error_description": "invalid_grant: token is invalid"}),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "slack", ["chat:write"])
            assert "error" in result

    @pytest.mark.asyncio
    async def test_refresh_token_missing_error(self):
        """Token exchange when no refresh token exists for the connection."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(404, {"error_description": "No refresh token available for this connection"}),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", ["repo"])
            assert "error" in result

    @pytest.mark.asyncio
    async def test_refresh_token_max_lifetime_exceeded(self):
        """Token exchange fails when refresh token lifetime is exceeded."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(400, {"error_description": "Refresh token max lifetime exceeded"}),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "google", ["gmail.readonly"])
            assert "error" in result


# ============================================================
# 2. Token Rotation Lifecycle
# ============================================================

class TestTokenRotationLifecycle:
    """Token rotation and renewal scenarios."""

    @pytest.mark.asyncio
    async def test_successful_token_exchange(self):
        """Successful token exchange returns expected fields."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(200, {
                "access_token": "new-access-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "repo",
            }),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", ["repo"])
            assert result["access_token"] == "new-access-token"
            assert result["token_type"] == "Bearer"
            assert result["expires_in"] == 3600
            assert result["scope"] == "repo"

    @pytest.mark.asyncio
    async def test_token_exchange_missing_token_type(self):
        """Token exchange response missing token_type defaults to Bearer."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(200, {
                "access_token": "new-token",
                "expires_in": 1800,
                "scope": "read",
            }),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "linear", ["read"])
            assert result["token_type"] == "Bearer"

    @pytest.mark.asyncio
    async def test_token_exchange_missing_expires_in(self):
        """Token exchange response missing expires_in defaults to 3600."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(200, {
                "access_token": "new-token",
                "token_type": "Bearer",
                "scope": "read",
            }),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "notion", ["read_content"])
            assert result["expires_in"] == 3600

    @pytest.mark.asyncio
    async def test_token_exchange_missing_scope_uses_requested(self):
        """Token exchange without scope in response uses requested scopes."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(200, {
                "access_token": "new-token",
                "token_type": "Bearer",
                "expires_in": 7200,
            }),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", ["repo", "gist"])
            assert result["scope"] == "repo gist"

    @pytest.mark.asyncio
    async def test_token_exchange_with_rotated_token(self):
        """Rotated tokens should work through the same flow."""
        responses = [
            make_response(200, {"access_token": "rotated-mgmt-token"}),
            make_response(200, {
                "access_token": "rotated-access-token",
                "token_type": "Bearer",
                "expires_in": 900,
                "scope": "repo",
            }),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", ["repo"])
            assert result["access_token"] == "rotated-access-token"
            assert result["expires_in"] == 900

    @pytest.mark.asyncio
    async def test_multiple_sequential_token_exchanges(self):
        """Multiple sequential exchanges should each succeed independently."""
        for i in range(5):
            responses = [
                make_response(200, {"access_token": f"mgmt-{i}"}),
                make_response(200, {
                    "access_token": f"token-{i}",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "repo",
                }),
            ]
            mc = mock_client(post_side_effect=responses)

            with patch("src.auth.httpx.AsyncClient", return_value=mc):
                result = await get_token_vault_token(USER, "github", ["repo"])
                assert result["access_token"] == f"token-{i}"


# ============================================================
# 3. Scope Mismatch Scenarios
# ============================================================

class TestScopeMismatch:
    """Scope-related error scenarios in token exchange."""

    @pytest.mark.asyncio
    async def test_scope_downgrade_by_provider(self):
        """Provider returns fewer scopes than requested."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(200, {
                "access_token": "limited-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "repo",  # Only repo, not gist
            }),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", ["repo", "gist"])
            assert "repo" in result["scope"]

    @pytest.mark.asyncio
    async def test_insufficient_scope_error(self):
        """Provider rejects with insufficient_scope."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(403, {"error_description": "insufficient_scope: admin required"}),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", ["admin"])
            assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_scopes_request(self):
        """Token exchange with empty scopes list."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(200, {
                "access_token": "no-scope-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "",
            }),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", [])
            assert "access_token" in result

    @pytest.mark.asyncio
    async def test_effective_scopes_intersection(self):
        """get_effective_scopes returns intersection of requested and allowed."""
        policy = AgentPolicy(
            agent_id="scope-test",
            agent_name="Scope Test",
            allowed_scopes={"github": ["repo", "read:user"]},
        )

        result = get_effective_scopes(policy, "github", ["repo", "admin", "gist"])
        assert set(result) == {"repo"}

    @pytest.mark.asyncio
    async def test_effective_scopes_no_overlap(self):
        """get_effective_scopes returns empty when no overlap."""
        policy = AgentPolicy(
            agent_id="no-overlap",
            agent_name="No Overlap",
            allowed_scopes={"github": ["repo"]},
        )

        result = get_effective_scopes(policy, "github", ["admin", "gist"])
        assert result == []

    @pytest.mark.asyncio
    async def test_effective_scopes_unknown_service(self):
        """get_effective_scopes for unknown service returns empty."""
        policy = AgentPolicy(
            agent_id="unknown-svc",
            agent_name="Unknown Svc",
            allowed_scopes={"github": ["repo"]},
        )

        result = get_effective_scopes(policy, "unknown_service", ["repo"])
        assert result == []

    @pytest.mark.asyncio
    async def test_effective_scopes_all_requested_allowed(self):
        """get_effective_scopes when all requested scopes are allowed."""
        policy = AgentPolicy(
            agent_id="all-allowed",
            agent_name="All Allowed",
            allowed_scopes={"github": ["repo", "read:user", "read:org", "gist"]},
        )

        result = get_effective_scopes(policy, "github", ["repo", "gist"])
        assert set(result) == {"repo", "gist"}

    @pytest.mark.asyncio
    async def test_scope_conflict_error_from_provider(self):
        """Provider returns scope conflict error."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(400, {"error_description": "Conflicting scopes requested"}),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", ["repo", "gist"])
            assert "error" in result


# ============================================================
# 4. Service Offline/Unreachable
# ============================================================

class TestServiceOffline:
    """Scenarios where the upstream service or Auth0 is unreachable."""

    @pytest.mark.asyncio
    async def test_auth0_server_error(self):
        """Auth0 returns 500 internal server error."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(500, {"error_description": "Internal server error"}),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", ["repo"])
            assert "error" in result

    @pytest.mark.asyncio
    async def test_auth0_service_unavailable(self):
        """Auth0 returns 503 service unavailable."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(503, {"error_description": "Service temporarily unavailable"}),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", ["repo"])
            assert "error" in result

    @pytest.mark.asyncio
    async def test_auth0_gateway_timeout(self):
        """Auth0 returns 504 gateway timeout."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(504, {"error_description": "Gateway timeout"}),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", ["repo"])
            assert "error" in result

    @pytest.mark.asyncio
    async def test_management_token_failure(self):
        """Management token request fails."""
        mc = mock_client(post_return=make_response(401, {"error": "unauthorized"}))

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            mgmt = await _get_management_token()
            assert mgmt == ""

    @pytest.mark.asyncio
    async def test_auth0_rate_limited(self):
        """Auth0 returns 429 too many requests."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(429, {"error_description": "Rate limit exceeded"}),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", ["repo"])
            assert "error" in result

    @pytest.mark.asyncio
    async def test_step_up_auth_server_error(self):
        """Step-up authentication returns server error."""
        mc = mock_client(post_return=make_response(500, {"error": "server_error"}))

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await trigger_step_up_auth(USER, "agent-1", "access github")
            assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_step_up_status_server_error(self):
        """Step-up status check returns server error."""
        mc = mock_client(post_return=make_response(500, {}))

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await check_step_up_status("req-123")
            assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_token_exchange_empty_response_body(self):
        """Token exchange returns 200 but with empty body raises KeyError."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(200, {}),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            # The code does data["access_token"] which will KeyError on empty body
            with pytest.raises(KeyError, match="access_token"):
                await get_token_vault_token(USER, "github", ["repo"])


# ============================================================
# 5. Token Vault Connection Failures
# ============================================================

class TestTokenVaultConnectionFailures:
    """Connection-level failures when talking to Token Vault."""

    @pytest.mark.asyncio
    async def test_connection_initiation_url_construction(self):
        """initiate_connection should construct a valid authorization URL."""
        url = await initiate_connection("github", "http://localhost:8000/callback")
        assert "github" in url
        assert "response_type=code" in url
        assert "redirect_uri" in url

    @pytest.mark.asyncio
    async def test_connection_initiation_different_services(self):
        """initiate_connection works for all supported services."""
        for svc in SUPPORTED_SERVICES:
            url = await initiate_connection(svc, f"http://localhost:8000/callback?connection={svc}")
            assert svc in url

    @pytest.mark.asyncio
    async def test_step_up_auth_success(self):
        """Successful step-up authentication returns pending status."""
        mc = mock_client(post_return=make_response(200, {
            "auth_req_id": "auth-req-123",
            "expires_in": 120,
        }))

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await trigger_step_up_auth(USER, "agent-1", "access github")
            assert result["status"] == "pending"
            assert result["auth_req_id"] == "auth-req-123"

    @pytest.mark.asyncio
    async def test_step_up_status_approved(self):
        """Step-up status check returns approved."""
        mc = mock_client(post_return=make_response(200, {"access_token": "step-up-token"}))

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await check_step_up_status("req-123")
            assert result["status"] == "approved"
            assert result["token"] == "step-up-token"

    @pytest.mark.asyncio
    async def test_step_up_status_pending(self):
        """Step-up status check returns authorization_pending."""
        mc = mock_client(post_return=make_response(400, {"error": "authorization_pending"}))

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await check_step_up_status("req-123")
            assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_step_up_status_slow_down(self):
        """Step-up status check returns slow_down."""
        mc = mock_client(post_return=make_response(400, {"error": "slow_down"}))

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await check_step_up_status("req-123")
            assert result["status"] == "pending"
            assert result.get("retry_after") == 5

    @pytest.mark.asyncio
    async def test_step_up_status_denied(self):
        """Step-up status check returns denied (unknown error)."""
        mc = mock_client(post_return=make_response(400, {"error": "access_denied"}))

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await check_step_up_status("req-123")
            assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_management_token_success(self):
        """Successful management token retrieval."""
        mc = mock_client(post_return=make_response(200, {"access_token": "mgmt-abc"}))

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            token = await _get_management_token()
            assert token == "mgmt-abc"

    @pytest.mark.asyncio
    async def test_management_token_empty_response(self):
        """Management token request returns empty access_token."""
        mc = mock_client(post_return=make_response(200, {}))

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            token = await _get_management_token()
            assert token == ""

    @pytest.mark.asyncio
    async def test_token_exchange_all_five_services(self):
        """Token exchange should work for all five supported services."""
        for svc in SUPPORTED_SERVICES:
            scopes = SUPPORTED_SERVICES[svc]["scopes"][:1]
            responses = [
                make_response(200, {"access_token": "mgmt-token"}),
                make_response(200, {
                    "access_token": f"token-{svc}",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": " ".join(scopes),
                }),
            ]
            mc = mock_client(post_side_effect=responses)

            with patch("src.auth.httpx.AsyncClient", return_value=mc):
                result = await get_token_vault_token(USER, svc, scopes)
                assert result["access_token"] == f"token-{svc}"

    @pytest.mark.asyncio
    async def test_token_exchange_unauthorized_client(self):
        """Token exchange fails due to unauthorized client."""
        responses = [
            make_response(200, {"access_token": "mgmt-token"}),
            make_response(401, {"error_description": "Unauthorized client"}),
        ]
        mc = mock_client(post_side_effect=responses)

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await get_token_vault_token(USER, "github", ["repo"])
            assert "error" in result

    @pytest.mark.asyncio
    async def test_step_up_auth_denied(self):
        """Step-up authentication request is denied."""
        mc = mock_client(post_return=make_response(403, {"error": "access_denied"}))

        with patch("src.auth.httpx.AsyncClient", return_value=mc):
            result = await trigger_step_up_auth(USER, "agent-1", "sensitive action")
            assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_requires_step_up_true(self):
        """requires_step_up returns True for listed services."""
        policy = AgentPolicy(
            agent_id="step-agent",
            agent_name="Step Agent",
            requires_step_up=["github", "slack"],
        )
        assert requires_step_up(policy, "github") is True
        assert requires_step_up(policy, "slack") is True

    @pytest.mark.asyncio
    async def test_requires_step_up_false(self):
        """requires_step_up returns False for unlisted services."""
        policy = AgentPolicy(
            agent_id="no-step-agent",
            agent_name="No Step Agent",
            requires_step_up=["github"],
        )
        assert requires_step_up(policy, "slack") is False
        assert requires_step_up(policy, "google") is False

    @pytest.mark.asyncio
    async def test_requires_step_up_empty_list(self):
        """requires_step_up returns False when list is empty."""
        policy = AgentPolicy(
            agent_id="empty-step",
            agent_name="Empty Step",
            requires_step_up=[],
        )
        for svc in SUPPORTED_SERVICES:
            assert requires_step_up(policy, svc) is False
