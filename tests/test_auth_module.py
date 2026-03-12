"""Tests for the auth module — SUPPORTED_SERVICES structure, initiate_connection,
token vault, and CIBA step-up auth flows."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.auth import (
    SUPPORTED_SERVICES,
    get_token_vault_token,
    initiate_connection,
    trigger_step_up_auth,
    check_step_up_status,
)


# ---------------------------------------------------------------------------
# 1. SUPPORTED_SERVICES structure validation
# ---------------------------------------------------------------------------

class TestSupportedServicesStructure:
    """Validate the structure and content of SUPPORTED_SERVICES."""

    def test_five_services_defined(self):
        assert len(SUPPORTED_SERVICES) == 5

    def test_github_present(self):
        assert "github" in SUPPORTED_SERVICES

    def test_slack_present(self):
        assert "slack" in SUPPORTED_SERVICES

    def test_google_present(self):
        assert "google" in SUPPORTED_SERVICES

    def test_linear_present(self):
        assert "linear" in SUPPORTED_SERVICES

    def test_notion_present(self):
        assert "notion" in SUPPORTED_SERVICES

    def test_each_has_display_name(self):
        for svc, info in SUPPORTED_SERVICES.items():
            assert "display_name" in info
            assert isinstance(info["display_name"], str)
            assert len(info["display_name"]) > 0

    def test_each_has_description(self):
        for svc, info in SUPPORTED_SERVICES.items():
            assert "description" in info
            assert isinstance(info["description"], str)
            assert len(info["description"]) > 0

    def test_each_has_scopes_list(self):
        for svc, info in SUPPORTED_SERVICES.items():
            assert "scopes" in info
            assert isinstance(info["scopes"], list)

    def test_slack_scopes(self):
        slack_scopes = set(SUPPORTED_SERVICES["slack"]["scopes"])
        assert "channels:read" in slack_scopes
        assert "chat:write" in slack_scopes

    def test_google_scopes(self):
        google_scopes = set(SUPPORTED_SERVICES["google"]["scopes"])
        assert "gmail.readonly" in google_scopes

    def test_linear_scopes(self):
        linear_scopes = set(SUPPORTED_SERVICES["linear"]["scopes"])
        assert "read" in linear_scopes
        assert "write" in linear_scopes

    def test_notion_scopes(self):
        notion_scopes = set(SUPPORTED_SERVICES["notion"]["scopes"])
        assert "read_content" in notion_scopes
        assert "update_content" in notion_scopes


# ---------------------------------------------------------------------------
# 2. initiate_connection URL generation
# ---------------------------------------------------------------------------

class TestInitiateConnection:
    """Test the URL generation for service connections."""

    @pytest.mark.asyncio
    async def test_returns_url_string(self):
        url = await initiate_connection("github", "http://localhost/callback")
        assert isinstance(url, str)
        assert url.startswith("https://")

    @pytest.mark.asyncio
    async def test_url_contains_service_as_connection(self):
        url = await initiate_connection("github", "http://localhost/callback")
        assert "connection=github" in url

    @pytest.mark.asyncio
    async def test_url_contains_client_id(self):
        url = await initiate_connection("github", "http://localhost/callback")
        assert "client_id=" in url

    @pytest.mark.asyncio
    async def test_url_contains_redirect_uri(self):
        redirect = "http://localhost:8000/callback?connection=github"
        url = await initiate_connection("github", redirect)
        assert "redirect_uri=" in url

    @pytest.mark.asyncio
    async def test_url_contains_response_type_code(self):
        url = await initiate_connection("github", "http://localhost/callback")
        assert "response_type=code" in url

    @pytest.mark.asyncio
    async def test_url_contains_access_type_offline(self):
        url = await initiate_connection("github", "http://localhost/callback")
        assert "access_type=offline" in url

    @pytest.mark.asyncio
    async def test_different_services_produce_different_urls(self):
        url1 = await initiate_connection("github", "http://localhost/callback")
        url2 = await initiate_connection("slack", "http://localhost/callback")
        assert url1 != url2
        assert "connection=github" in url1
        assert "connection=slack" in url2


# ---------------------------------------------------------------------------
# 3. Token vault token exchange (mocked HTTP)
# ---------------------------------------------------------------------------

class TestTokenVaultExchange:
    """Test token vault exchange with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_successful_exchange(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "mgmt_token_123",
        }
        mock_exchange_response = MagicMock()
        mock_exchange_response.status_code = 200
        mock_exchange_response.json.return_value = {
            "access_token": "gh_token_xyz",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "repo",
        }

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[
                mock_response,  # management token call
                mock_exchange_response,  # token exchange call
            ])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await get_token_vault_token("user1", "github", ["repo"])

        assert "access_token" in result
        assert result["access_token"] == "gh_token_xyz"
        assert result["token_type"] == "Bearer"

    @pytest.mark.asyncio
    async def test_exchange_failure_returns_error(self):
        mock_mgmt_response = MagicMock()
        mock_mgmt_response.status_code = 200
        mock_mgmt_response.json.return_value = {"access_token": "mgmt"}

        mock_error_response = MagicMock()
        mock_error_response.status_code = 400
        mock_error_response.json.return_value = {
            "error_description": "Invalid scope"
        }

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[
                mock_mgmt_response,
                mock_error_response,
            ])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await get_token_vault_token("user1", "github", ["bad_scope"])

        assert "error" in result
        assert result["error"] == "Invalid scope"


# ---------------------------------------------------------------------------
# 4. Step-up auth (CIBA) flows
# ---------------------------------------------------------------------------

class TestStepUpAuthFlows:
    """Test trigger_step_up_auth and check_step_up_status with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_trigger_step_up_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "auth_req_id": "req_abc",
            "expires_in": 120,
        }

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await trigger_step_up_auth("user1", "agent1", "delete files")

        assert result["status"] == "pending"
        assert result["auth_req_id"] == "req_abc"

    @pytest.mark.asyncio
    async def test_trigger_step_up_failure(self):
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await trigger_step_up_auth("user1", "agent1", "action")

        assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_check_status_approved(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "approved_tok"}

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await check_step_up_status("req_123")

        assert result["status"] == "approved"

    @pytest.mark.asyncio
    async def test_check_status_pending(self):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "authorization_pending"}

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await check_step_up_status("req_123")

        assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_check_status_slow_down(self):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "slow_down"}

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await check_step_up_status("req_123")

        assert result["status"] == "pending"
        assert result["retry_after"] == 5

    @pytest.mark.asyncio
    async def test_check_status_denied(self):
        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch("src.auth.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await check_step_up_status("req_123")

        assert result["status"] == "denied"
