"""Tests for config module — Settings properties, URL construction,
and environment variable handling."""

import pytest

from src.config import Settings


class TestSettingsDefaults:
    """Verify default values for all settings."""

    def test_default_auth0_domain_empty(self):
        s = Settings(auth0_domain="", auth0_client_id="", auth0_client_secret="")
        assert s.auth0_domain == ""

    def test_default_auth0_client_id_empty(self):
        s = Settings(auth0_domain="", auth0_client_id="", auth0_client_secret="")
        assert s.auth0_client_id == ""

    def test_default_auth0_client_secret_empty(self):
        s = Settings(auth0_domain="", auth0_client_id="", auth0_client_secret="")
        assert s.auth0_client_secret == ""

    def test_default_callback_url(self):
        s = Settings(auth0_domain="", auth0_client_id="", auth0_client_secret="")
        assert s.auth0_callback_url == "http://localhost:8000/callback"

    def test_default_audience_empty(self):
        s = Settings(auth0_domain="", auth0_client_id="", auth0_client_secret="")
        assert s.auth0_audience == ""

    def test_default_secret_key(self):
        s = Settings(auth0_domain="", auth0_client_id="", auth0_client_secret="")
        assert s.app_secret_key == "dev-secret-change-in-production"

    def test_default_database_url(self):
        s = Settings(auth0_domain="", auth0_client_id="", auth0_client_secret="")
        assert "sqlite" in s.database_url

    def test_default_token_vault_enabled(self):
        s = Settings(auth0_domain="", auth0_client_id="", auth0_client_secret="")
        assert s.token_vault_enabled is True


class TestSettingsProperties:
    """Verify computed properties with a sample domain."""

    @pytest.fixture
    def s(self):
        return Settings(
            auth0_domain="example.auth0.com",
            auth0_client_id="client123",
            auth0_client_secret="secret456",
        )

    def test_issuer_url(self, s):
        assert s.auth0_issuer == "https://example.auth0.com/"

    def test_jwks_url(self, s):
        assert s.auth0_jwks_url == "https://example.auth0.com/.well-known/jwks.json"

    def test_token_url(self, s):
        assert s.auth0_token_url == "https://example.auth0.com/oauth/token"

    def test_authorize_url(self, s):
        assert s.auth0_authorize_url == "https://example.auth0.com/authorize"

    def test_userinfo_url(self, s):
        assert s.auth0_userinfo_url == "https://example.auth0.com/userinfo"


class TestSettingsPropertiesEdgeCases:
    """Edge cases for settings properties."""

    def test_domain_with_trailing_slash(self):
        s = Settings(
            auth0_domain="example.auth0.com/",
            auth0_client_id="c", auth0_client_secret="s",
        )
        # Property just prepends https:// and appends /
        assert "https://example.auth0.com//" in s.auth0_issuer

    def test_empty_domain_urls(self):
        s = Settings(
            auth0_domain="",
            auth0_client_id="c", auth0_client_secret="s",
        )
        assert s.auth0_issuer == "https:///"
        assert s.auth0_jwks_url == "https:///.well-known/jwks.json"

    def test_domain_with_subdomain(self):
        s = Settings(
            auth0_domain="dev-abc123.us.auth0.com",
            auth0_client_id="c", auth0_client_secret="s",
        )
        assert s.auth0_issuer == "https://dev-abc123.us.auth0.com/"
        assert "dev-abc123.us.auth0.com" in s.auth0_jwks_url

    def test_custom_callback_url(self):
        s = Settings(
            auth0_domain="d",
            auth0_client_id="c", auth0_client_secret="s",
            auth0_callback_url="https://myapp.com/auth/callback",
        )
        assert s.auth0_callback_url == "https://myapp.com/auth/callback"

    def test_custom_audience(self):
        s = Settings(
            auth0_domain="d",
            auth0_client_id="c", auth0_client_secret="s",
            auth0_audience="https://api.example.com",
        )
        assert s.auth0_audience == "https://api.example.com"

    def test_token_vault_disabled(self):
        s = Settings(
            auth0_domain="d",
            auth0_client_id="c", auth0_client_secret="s",
            token_vault_enabled=False,
        )
        assert s.token_vault_enabled is False


class TestSettingsPropertyTypes:
    """Verify property return types."""

    def test_issuer_is_string(self):
        s = Settings(auth0_domain="d", auth0_client_id="c", auth0_client_secret="s")
        assert isinstance(s.auth0_issuer, str)

    def test_jwks_url_is_string(self):
        s = Settings(auth0_domain="d", auth0_client_id="c", auth0_client_secret="s")
        assert isinstance(s.auth0_jwks_url, str)

    def test_token_url_is_string(self):
        s = Settings(auth0_domain="d", auth0_client_id="c", auth0_client_secret="s")
        assert isinstance(s.auth0_token_url, str)

    def test_authorize_url_is_string(self):
        s = Settings(auth0_domain="d", auth0_client_id="c", auth0_client_secret="s")
        assert isinstance(s.auth0_authorize_url, str)

    def test_userinfo_url_is_string(self):
        s = Settings(auth0_domain="d", auth0_client_id="c", auth0_client_secret="s")
        assert isinstance(s.auth0_userinfo_url, str)


class TestSettingsURLSecurity:
    """Settings URLs use HTTPS."""

    @pytest.fixture
    def s(self):
        return Settings(
            auth0_domain="secure.auth0.com",
            auth0_client_id="c", auth0_client_secret="s",
        )

    def test_issuer_uses_https(self, s):
        assert s.auth0_issuer.startswith("https://")

    def test_jwks_uses_https(self, s):
        assert s.auth0_jwks_url.startswith("https://")

    def test_token_url_uses_https(self, s):
        assert s.auth0_token_url.startswith("https://")

    def test_authorize_url_uses_https(self, s):
        assert s.auth0_authorize_url.startswith("https://")

    def test_userinfo_url_uses_https(self, s):
        assert s.auth0_userinfo_url.startswith("https://")
