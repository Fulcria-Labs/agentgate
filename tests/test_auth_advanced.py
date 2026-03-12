"""Advanced auth module tests — management token, token vault edge cases,
initiate_connection variations, CIBA flow edge cases, and SUPPORTED_SERVICES
deep validation."""

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


# ---------------------------------------------------------------------------
# 1. _get_management_token tests
# ---------------------------------------------------------------------------

class TestGetManagementToken:
    """Test the internal management token retrieval."""

    @pytest.mark.asyncio
    async def test_returns_access_token_on_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "mgmt_tok_123"}

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await _get_management_token()

        assert result == "mgmt_tok_123"

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_missing_token(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "unauthorized"}

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await _get_management_token()

        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_on_empty_response(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {}

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await _get_management_token()

        assert result == ""


# ---------------------------------------------------------------------------
# 2. Token vault exchange edge cases
# ---------------------------------------------------------------------------

class TestTokenVaultEdgeCases:
    """Edge cases for the token vault exchange flow."""

    @pytest.mark.asyncio
    async def test_exchange_with_empty_scopes(self):
        """Token exchange with empty scopes list."""
        mock_mgmt = MagicMock()
        mock_mgmt.status_code = 200
        mock_mgmt.json.return_value = {"access_token": "mgmt"}

        mock_exchange = MagicMock()
        mock_exchange.status_code = 200
        mock_exchange.json.return_value = {
            "access_token": "tok_empty_scope",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "",
        }

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[mock_mgmt, mock_exchange])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await get_token_vault_token("user1", "github", [])

        assert result["access_token"] == "tok_empty_scope"

    @pytest.mark.asyncio
    async def test_exchange_missing_optional_fields(self):
        """Token exchange response missing optional fields uses defaults."""
        mock_mgmt = MagicMock()
        mock_mgmt.status_code = 200
        mock_mgmt.json.return_value = {"access_token": "mgmt"}

        mock_exchange = MagicMock()
        mock_exchange.status_code = 200
        mock_exchange.json.return_value = {
            "access_token": "tok_minimal",
        }

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[mock_mgmt, mock_exchange])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await get_token_vault_token("user1", "github", ["repo"])

        assert result["access_token"] == "tok_minimal"
        assert result["token_type"] == "Bearer"
        assert result["expires_in"] == 3600
        assert result["scope"] == "repo"

    @pytest.mark.asyncio
    async def test_exchange_error_without_description(self):
        """Token exchange error with no error_description."""
        mock_mgmt = MagicMock()
        mock_mgmt.status_code = 200
        mock_mgmt.json.return_value = {"access_token": "mgmt"}

        mock_error = MagicMock()
        mock_error.status_code = 500
        mock_error.json.return_value = {}

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[mock_mgmt, mock_error])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await get_token_vault_token("user1", "github", ["repo"])

        assert "error" in result
        assert result["error"] == "Token exchange failed"

    @pytest.mark.asyncio
    async def test_exchange_with_multiple_scopes(self):
        """Token exchange with multiple scopes joined."""
        mock_mgmt = MagicMock()
        mock_mgmt.status_code = 200
        mock_mgmt.json.return_value = {"access_token": "mgmt"}

        mock_exchange = MagicMock()
        mock_exchange.status_code = 200
        mock_exchange.json.return_value = {
            "access_token": "tok_multi",
            "scope": "repo read:user gist",
        }

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[mock_mgmt, mock_exchange])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await get_token_vault_token(
                "user1", "github", ["repo", "read:user", "gist"]
            )

        assert result["scope"] == "repo read:user gist"

    @pytest.mark.asyncio
    async def test_exchange_401_error(self):
        """Token exchange returns 401 unauthorized."""
        mock_mgmt = MagicMock()
        mock_mgmt.status_code = 200
        mock_mgmt.json.return_value = {"access_token": "mgmt"}

        mock_error = MagicMock()
        mock_error.status_code = 401
        mock_error.json.return_value = {
            "error_description": "Unauthorized client"
        }

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[mock_mgmt, mock_error])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await get_token_vault_token("user1", "github", ["repo"])

        assert "error" in result
        assert result["error"] == "Unauthorized client"

    @pytest.mark.asyncio
    async def test_exchange_403_error(self):
        """Token exchange returns 403 forbidden."""
        mock_mgmt = MagicMock()
        mock_mgmt.status_code = 200
        mock_mgmt.json.return_value = {"access_token": "mgmt"}

        mock_error = MagicMock()
        mock_error.status_code = 403
        mock_error.json.return_value = {
            "error_description": "Insufficient permissions"
        }

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[mock_mgmt, mock_error])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await get_token_vault_token("user1", "github", ["repo"])

        assert result["error"] == "Insufficient permissions"


# ---------------------------------------------------------------------------
# 3. initiate_connection edge cases
# ---------------------------------------------------------------------------

class TestInitiateConnectionEdgeCases:
    """Edge cases for service connection initiation."""

    @pytest.mark.asyncio
    async def test_url_contains_state_parameter(self):
        url = await initiate_connection("github", "http://localhost/callback")
        assert "state=" in url

    @pytest.mark.asyncio
    async def test_url_contains_scope(self):
        url = await initiate_connection("github", "http://localhost/callback")
        assert "scope=" in url

    @pytest.mark.asyncio
    async def test_different_redirect_uris(self):
        url1 = await initiate_connection("github", "http://localhost:8000/cb")
        url2 = await initiate_connection("github", "http://example.com/cb")
        # Both should work but contain different redirect URIs
        assert "localhost" in url1
        assert "example.com" in url2

    @pytest.mark.asyncio
    async def test_slack_connection(self):
        url = await initiate_connection("slack", "http://localhost/callback")
        assert "connection=slack" in url

    @pytest.mark.asyncio
    async def test_google_connection(self):
        url = await initiate_connection("google", "http://localhost/callback")
        assert "connection=google" in url

    @pytest.mark.asyncio
    async def test_linear_connection(self):
        url = await initiate_connection("linear", "http://localhost/callback")
        assert "connection=linear" in url

    @pytest.mark.asyncio
    async def test_notion_connection(self):
        url = await initiate_connection("notion", "http://localhost/callback")
        assert "connection=notion" in url

    @pytest.mark.asyncio
    async def test_state_is_unique_per_call(self):
        """Each call generates a unique state parameter."""
        url1 = await initiate_connection("github", "http://localhost/callback")
        url2 = await initiate_connection("github", "http://localhost/callback")
        # Extract state values
        import re
        states = []
        for url in [url1, url2]:
            match = re.search(r"state=([^&]+)", url)
            assert match is not None
            states.append(match.group(1))
        assert states[0] != states[1]


# ---------------------------------------------------------------------------
# 4. CIBA step-up edge cases
# ---------------------------------------------------------------------------

class TestCIBAEdgeCases:
    """Additional CIBA step-up authentication edge cases."""

    @pytest.mark.asyncio
    async def test_trigger_step_up_returns_expires_in(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "auth_req_id": "req_123",
            "expires_in": 300,
        }

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await trigger_step_up_auth("user1", "agent1", "action")

        assert result["expires_in"] == 300

    @pytest.mark.asyncio
    async def test_trigger_step_up_default_expires(self):
        """Missing expires_in defaults to 120."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "auth_req_id": "req_456",
        }

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await trigger_step_up_auth("user1", "agent1", "action")

        assert result["expires_in"] == 120

    @pytest.mark.asyncio
    async def test_check_status_approved_returns_token(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "approved_token_xyz"}

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await check_step_up_status("req_789")

        assert result["status"] == "approved"
        assert result["token"] == "approved_token_xyz"

    @pytest.mark.asyncio
    async def test_check_status_400_unknown_error_is_denied(self):
        """A 400 with an unknown error code is treated as denied."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "unknown_error_code"}

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await check_step_up_status("req_xyz")

        assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_check_status_500_is_denied(self):
        """A 500 response is treated as denied."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await check_step_up_status("req_xyz")

        assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_trigger_step_up_http_error_returns_denied(self):
        """Non-200 response from trigger returns denied status."""
        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await trigger_step_up_auth("user1", "agent1", "action")

        assert result["status"] == "denied"
        assert "error" in result


# ---------------------------------------------------------------------------
# 5. SUPPORTED_SERVICES deep validation
# ---------------------------------------------------------------------------

class TestSupportedServicesDeep:
    """Deep validation of SUPPORTED_SERVICES configuration."""

    def test_github_scopes_complete(self):
        scopes = set(SUPPORTED_SERVICES["github"]["scopes"])
        assert "repo" in scopes
        assert "read:user" in scopes
        assert "read:org" in scopes
        assert "gist" in scopes
        assert "notifications" in scopes

    def test_slack_scopes_complete(self):
        scopes = set(SUPPORTED_SERVICES["slack"]["scopes"])
        assert "channels:read" in scopes
        assert "chat:write" in scopes
        assert "users:read" in scopes
        assert "files:read" in scopes

    def test_google_scopes_complete(self):
        scopes = set(SUPPORTED_SERVICES["google"]["scopes"])
        assert "gmail.readonly" in scopes
        assert "calendar.readonly" in scopes
        assert "drive.readonly" in scopes

    def test_linear_scopes_complete(self):
        scopes = set(SUPPORTED_SERVICES["linear"]["scopes"])
        assert "read" in scopes
        assert "write" in scopes
        assert "issues:create" in scopes

    def test_notion_scopes_complete(self):
        scopes = set(SUPPORTED_SERVICES["notion"]["scopes"])
        assert "read_content" in scopes
        assert "update_content" in scopes

    def test_all_descriptions_non_empty(self):
        for name, info in SUPPORTED_SERVICES.items():
            assert len(info["description"]) > 10, f"{name} description too short"

    def test_display_names_are_capitalized(self):
        for name, info in SUPPORTED_SERVICES.items():
            assert info["display_name"][0].isupper(), \
                f"{name} display_name not capitalized"

    def test_no_scope_duplicates_per_service(self):
        for name, info in SUPPORTED_SERVICES.items():
            scopes = info["scopes"]
            assert len(scopes) == len(set(scopes)), \
                f"{name} has duplicate scopes"
