"""Tests for Auth0 token exchange, CIBA step-up auth, and connection flows
using mocked HTTP responses."""

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


class TestGetTokenVaultToken:
    """Token vault token exchange tests."""

    @pytest.mark.asyncio
    async def test_successful_exchange(self):
        mock_resp_mgmt = MagicMock()
        mock_resp_mgmt.json.return_value = {"access_token": "mgmt-token-123"}

        mock_resp_exchange = MagicMock()
        mock_resp_exchange.status_code = 200
        mock_resp_exchange.json.return_value = {
            "access_token": "service-token-xyz",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "repo read:user",
        }

        with patch("src.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[mock_resp_mgmt, mock_resp_exchange])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await get_token_vault_token("user1", "github", ["repo", "read:user"])
            assert result["access_token"] == "service-token-xyz"
            assert result["token_type"] == "Bearer"
            assert result["expires_in"] == 3600

    @pytest.mark.asyncio
    async def test_failed_exchange_returns_error(self):
        mock_resp_mgmt = MagicMock()
        mock_resp_mgmt.json.return_value = {"access_token": "mgmt-token"}

        mock_resp_exchange = MagicMock()
        mock_resp_exchange.status_code = 400
        mock_resp_exchange.json.return_value = {
            "error_description": "Token exchange failed"
        }

        with patch("src.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[mock_resp_mgmt, mock_resp_exchange])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await get_token_vault_token("user1", "github", ["repo"])
            assert "error" in result

    @pytest.mark.asyncio
    async def test_exchange_with_empty_scopes(self):
        mock_resp_mgmt = MagicMock()
        mock_resp_mgmt.json.return_value = {"access_token": "mgmt"}

        mock_resp_exchange = MagicMock()
        mock_resp_exchange.status_code = 200
        mock_resp_exchange.json.return_value = {
            "access_token": "token",
            "token_type": "Bearer",
            "expires_in": 1800,
            "scope": "",
        }

        with patch("src.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[mock_resp_mgmt, mock_resp_exchange])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await get_token_vault_token("user1", "github", [])
            assert "access_token" in result

    @pytest.mark.asyncio
    async def test_exchange_missing_token_type_defaults(self):
        mock_resp_mgmt = MagicMock()
        mock_resp_mgmt.json.return_value = {"access_token": "mgmt"}

        mock_resp_exchange = MagicMock()
        mock_resp_exchange.status_code = 200
        mock_resp_exchange.json.return_value = {
            "access_token": "token-abc",
        }

        with patch("src.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[mock_resp_mgmt, mock_resp_exchange])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await get_token_vault_token("user1", "slack", ["channels:read"])
            assert result["token_type"] == "Bearer"
            assert result["expires_in"] == 3600


class TestInitiateConnection:
    """Connection URL generation tests."""

    @pytest.mark.asyncio
    async def test_url_contains_service(self):
        url = await initiate_connection("github", "https://example.com/callback")
        assert "connection=github" in url

    @pytest.mark.asyncio
    async def test_url_contains_redirect_uri(self):
        url = await initiate_connection("slack", "https://myapp.com/cb")
        assert "redirect_uri=https" in url

    @pytest.mark.asyncio
    async def test_url_contains_client_id(self):
        url = await initiate_connection("google", "https://example.com/cb")
        assert "client_id=" in url

    @pytest.mark.asyncio
    async def test_url_contains_response_type(self):
        url = await initiate_connection("linear", "https://example.com/cb")
        assert "response_type=code" in url

    @pytest.mark.asyncio
    async def test_url_contains_state(self):
        url = await initiate_connection("notion", "https://example.com/cb")
        assert "state=" in url

    @pytest.mark.asyncio
    async def test_url_contains_scope(self):
        url = await initiate_connection("github", "https://example.com/cb")
        assert "scope=" in url

    @pytest.mark.asyncio
    async def test_url_contains_access_type(self):
        url = await initiate_connection("github", "https://example.com/cb")
        assert "access_type=offline" in url

    @pytest.mark.asyncio
    async def test_different_calls_produce_different_states(self):
        url1 = await initiate_connection("github", "https://example.com/cb")
        url2 = await initiate_connection("github", "https://example.com/cb")
        # Extract state values
        import re
        state1 = re.search(r"state=([^&]+)", url1).group(1)
        state2 = re.search(r"state=([^&]+)", url2).group(1)
        assert state1 != state2


class TestTriggerStepUpAuth:
    """CIBA step-up authentication trigger tests."""

    @pytest.mark.asyncio
    async def test_successful_trigger(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "auth_req_id": "req-123",
            "expires_in": 120,
        }

        with patch("src.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await trigger_step_up_auth("user1", "agent1", "delete repo")
            assert result["auth_req_id"] == "req-123"
            assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_failed_trigger(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 400

        with patch("src.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await trigger_step_up_auth("user1", "agent1", "action")
            assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_trigger_with_missing_expires(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "auth_req_id": "req-456",
        }

        with patch("src.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await trigger_step_up_auth("user1", "agent1", "action")
            assert result["expires_in"] == 120  # default


class TestCheckStepUpStatus:
    """CIBA step-up status polling tests."""

    @pytest.mark.asyncio
    async def test_approved_status(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": "approved-token",
        }

        with patch("src.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await check_step_up_status("req-123")
            assert result["status"] == "approved"
            assert result["token"] == "approved-token"

    @pytest.mark.asyncio
    async def test_pending_status(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": "authorization_pending"}

        with patch("src.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await check_step_up_status("req-123")
            assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_slow_down_status(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": "slow_down"}

        with patch("src.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await check_step_up_status("req-123")
            assert result["status"] == "pending"
            assert result["retry_after"] == 5

    @pytest.mark.asyncio
    async def test_denied_status(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 403

        with patch("src.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await check_step_up_status("req-123")
            assert result["status"] == "denied"

    @pytest.mark.asyncio
    async def test_unknown_400_error_denied(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": "unknown_error"}

        with patch("src.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await check_step_up_status("req-123")
            assert result["status"] == "denied"


class TestGetManagementToken:
    """Management token retrieval tests."""

    @pytest.mark.asyncio
    async def test_successful_token_retrieval(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"access_token": "mgmt-token-abc"}

        with patch("src.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            token = await _get_management_token()
            assert token == "mgmt-token-abc"

    @pytest.mark.asyncio
    async def test_missing_token_returns_empty(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}

        with patch("src.auth.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            token = await _get_management_token()
            assert token == ""


class TestSupportedServicesDetailed:
    """Detailed validation of SUPPORTED_SERVICES."""

    def test_github_display_name(self):
        assert SUPPORTED_SERVICES["github"]["display_name"] == "GitHub"

    def test_slack_display_name(self):
        assert SUPPORTED_SERVICES["slack"]["display_name"] == "Slack"

    def test_google_display_name(self):
        assert SUPPORTED_SERVICES["google"]["display_name"] == "Google"

    def test_linear_display_name(self):
        assert SUPPORTED_SERVICES["linear"]["display_name"] == "Linear"

    def test_notion_display_name(self):
        assert SUPPORTED_SERVICES["notion"]["display_name"] == "Notion"

    def test_github_has_repo_scope(self):
        assert "repo" in SUPPORTED_SERVICES["github"]["scopes"]

    def test_github_has_read_user_scope(self):
        assert "read:user" in SUPPORTED_SERVICES["github"]["scopes"]

    def test_github_has_read_org_scope(self):
        assert "read:org" in SUPPORTED_SERVICES["github"]["scopes"]

    def test_github_has_gist_scope(self):
        assert "gist" in SUPPORTED_SERVICES["github"]["scopes"]

    def test_github_has_notifications_scope(self):
        assert "notifications" in SUPPORTED_SERVICES["github"]["scopes"]

    def test_slack_has_files_read(self):
        assert "files:read" in SUPPORTED_SERVICES["slack"]["scopes"]

    def test_google_has_calendar_readonly(self):
        assert "calendar.readonly" in SUPPORTED_SERVICES["google"]["scopes"]

    def test_google_has_drive_readonly(self):
        assert "drive.readonly" in SUPPORTED_SERVICES["google"]["scopes"]

    def test_linear_has_issues_create(self):
        assert "issues:create" in SUPPORTED_SERVICES["linear"]["scopes"]

    def test_each_service_has_icon(self):
        for svc, info in SUPPORTED_SERVICES.items():
            assert "icon" in info

    def test_no_duplicate_scopes_per_service(self):
        for svc, info in SUPPORTED_SERVICES.items():
            scopes = info["scopes"]
            assert len(scopes) == len(set(scopes)), f"Duplicate scopes in {svc}"
