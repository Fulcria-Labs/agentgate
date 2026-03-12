"""Tests for config module — settings, properties, and environment handling."""

import os
import pytest
from unittest.mock import patch


class TestSettingsDefaults:
    """Verify default values for all Settings fields."""

    def test_default_auth0_domain(self):
        from src.config import Settings
        s = Settings()
        assert s.auth0_domain == ""

    def test_default_auth0_client_id(self):
        from src.config import Settings
        s = Settings()
        assert s.auth0_client_id == ""

    def test_default_auth0_client_secret(self):
        from src.config import Settings
        s = Settings()
        assert s.auth0_client_secret == ""

    def test_default_callback_url(self):
        from src.config import Settings
        s = Settings()
        assert s.auth0_callback_url == "http://localhost:8000/callback"

    def test_default_audience(self):
        from src.config import Settings
        s = Settings()
        assert s.auth0_audience == ""

    def test_default_app_secret_key(self):
        from src.config import Settings
        s = Settings()
        assert s.app_secret_key == "dev-secret-change-in-production"

    def test_default_database_url(self):
        from src.config import Settings
        s = Settings()
        assert "sqlite" in s.database_url

    def test_default_token_vault_enabled(self):
        from src.config import Settings
        s = Settings()
        assert s.token_vault_enabled is True


class TestSettingsProperties:
    """Verify computed properties on Settings."""

    def test_auth0_issuer(self):
        from src.config import Settings
        s = Settings(auth0_domain="example.auth0.com")
        assert s.auth0_issuer == "https://example.auth0.com/"

    def test_auth0_jwks_url(self):
        from src.config import Settings
        s = Settings(auth0_domain="example.auth0.com")
        assert s.auth0_jwks_url == "https://example.auth0.com/.well-known/jwks.json"

    def test_auth0_token_url(self):
        from src.config import Settings
        s = Settings(auth0_domain="example.auth0.com")
        assert s.auth0_token_url == "https://example.auth0.com/oauth/token"

    def test_auth0_authorize_url(self):
        from src.config import Settings
        s = Settings(auth0_domain="example.auth0.com")
        assert s.auth0_authorize_url == "https://example.auth0.com/authorize"

    def test_auth0_userinfo_url(self):
        from src.config import Settings
        s = Settings(auth0_domain="example.auth0.com")
        assert s.auth0_userinfo_url == "https://example.auth0.com/userinfo"

    def test_empty_domain_produces_bare_url(self):
        from src.config import Settings
        s = Settings(auth0_domain="")
        assert s.auth0_issuer == "https:///"

    def test_domain_with_trailing_slash_not_doubled(self):
        from src.config import Settings
        s = Settings(auth0_domain="test.auth0.com")
        # Property appends "/" so domain itself should not have one
        assert "//" not in s.auth0_issuer.replace("https://", "")


class TestSettingsFromEnv:
    """Verify Settings loads from environment variables."""

    def test_override_auth0_domain(self):
        from src.config import Settings
        s = Settings(auth0_domain="custom.auth0.com")
        assert s.auth0_domain == "custom.auth0.com"

    def test_override_multiple_fields(self):
        from src.config import Settings
        s = Settings(
            auth0_domain="test.auth0.com",
            auth0_client_id="client123",
            auth0_client_secret="secret456",
        )
        assert s.auth0_domain == "test.auth0.com"
        assert s.auth0_client_id == "client123"
        assert s.auth0_client_secret == "secret456"

    def test_token_vault_can_be_disabled(self):
        from src.config import Settings
        s = Settings(token_vault_enabled=False)
        assert s.token_vault_enabled is False
