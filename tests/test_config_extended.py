"""Extended configuration tests — property edge cases, domain formatting,
URL composition, and model_config verification."""

import pytest
from src.config import Settings


class TestSettingsURLComposition:
    """Verify URL composition with various domain formats."""

    def test_domain_with_subdomain(self):
        s = Settings(auth0_domain="dev-12345.us.auth0.com")
        assert s.auth0_issuer == "https://dev-12345.us.auth0.com/"
        assert s.auth0_jwks_url == "https://dev-12345.us.auth0.com/.well-known/jwks.json"

    def test_domain_produces_correct_token_url(self):
        s = Settings(auth0_domain="my-tenant.auth0.com")
        assert s.auth0_token_url == "https://my-tenant.auth0.com/oauth/token"

    def test_domain_produces_correct_authorize_url(self):
        s = Settings(auth0_domain="my-tenant.auth0.com")
        assert s.auth0_authorize_url == "https://my-tenant.auth0.com/authorize"

    def test_domain_produces_correct_userinfo_url(self):
        s = Settings(auth0_domain="my-tenant.auth0.com")
        assert s.auth0_userinfo_url == "https://my-tenant.auth0.com/userinfo"

    def test_domain_with_dashes(self):
        s = Settings(auth0_domain="my-long-domain-name.auth0.com")
        assert "my-long-domain-name" in s.auth0_issuer

    def test_domain_with_numbers(self):
        s = Settings(auth0_domain="tenant123.auth0.com")
        assert s.auth0_issuer == "https://tenant123.auth0.com/"


class TestSettingsFieldTypes:
    """Verify field type correctness."""

    def test_auth0_domain_is_string(self):
        s = Settings()
        assert isinstance(s.auth0_domain, str)

    def test_auth0_client_id_is_string(self):
        s = Settings()
        assert isinstance(s.auth0_client_id, str)

    def test_auth0_client_secret_is_string(self):
        s = Settings()
        assert isinstance(s.auth0_client_secret, str)

    def test_auth0_callback_url_is_string(self):
        s = Settings()
        assert isinstance(s.auth0_callback_url, str)

    def test_app_secret_key_is_string(self):
        s = Settings()
        assert isinstance(s.app_secret_key, str)

    def test_database_url_is_string(self):
        s = Settings()
        assert isinstance(s.database_url, str)

    def test_token_vault_enabled_is_bool(self):
        s = Settings()
        assert isinstance(s.token_vault_enabled, bool)


class TestSettingsCustomValues:
    """Verify settings accept custom values."""

    def test_custom_callback_url(self):
        s = Settings(auth0_callback_url="https://my-app.com/callback")
        assert s.auth0_callback_url == "https://my-app.com/callback"

    def test_custom_secret_key(self):
        s = Settings(app_secret_key="my-production-secret-key-123")
        assert s.app_secret_key == "my-production-secret-key-123"

    def test_custom_database_url(self):
        s = Settings(database_url="postgresql+asyncpg://localhost/agentgate")
        assert s.database_url == "postgresql+asyncpg://localhost/agentgate"

    def test_custom_audience(self):
        s = Settings(auth0_audience="https://api.example.com")
        assert s.auth0_audience == "https://api.example.com"

    def test_all_custom_values(self):
        s = Settings(
            auth0_domain="custom.auth0.com",
            auth0_client_id="cid",
            auth0_client_secret="csec",
            auth0_callback_url="https://custom.com/cb",
            auth0_audience="https://api.custom.com",
            app_secret_key="secret",
            database_url="sqlite:///custom.db",
            token_vault_enabled=False,
        )
        assert s.auth0_domain == "custom.auth0.com"
        assert s.auth0_client_id == "cid"
        assert s.auth0_client_secret == "csec"
        assert s.auth0_callback_url == "https://custom.com/cb"
        assert s.auth0_audience == "https://api.custom.com"
        assert s.app_secret_key == "secret"
        assert s.database_url == "sqlite:///custom.db"
        assert s.token_vault_enabled is False


class TestSettingsModelConfig:
    """Verify the model_config settings."""

    def test_model_config_env_file(self):
        assert Settings.model_config["env_file"] == ".env"

    def test_model_config_encoding(self):
        assert Settings.model_config["env_file_encoding"] == "utf-8"


class TestSettingsModuleLevelInstance:
    """Verify the module-level settings instance."""

    def test_module_settings_exists(self):
        from src.config import settings
        assert settings is not None

    def test_module_settings_is_settings_instance(self):
        from src.config import settings
        assert isinstance(settings, Settings)

    def test_module_settings_has_default_values(self):
        from src.config import settings
        # The module-level instance uses defaults (unless .env overrides)
        assert isinstance(settings.auth0_domain, str)
        assert isinstance(settings.token_vault_enabled, bool)
