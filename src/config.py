"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Auth0
    auth0_domain: str = ""
    auth0_client_id: str = ""
    auth0_client_secret: str = ""
    auth0_callback_url: str = "http://localhost:8000/callback"
    auth0_audience: str = ""

    # App
    app_secret_key: str = "dev-secret-change-in-production"
    database_url: str = "sqlite+aiosqlite:///./agentgate.db"

    # Token Vault
    token_vault_enabled: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def auth0_issuer(self) -> str:
        return f"https://{self.auth0_domain}/"

    @property
    def auth0_jwks_url(self) -> str:
        return f"https://{self.auth0_domain}/.well-known/jwks.json"

    @property
    def auth0_token_url(self) -> str:
        return f"https://{self.auth0_domain}/oauth/token"

    @property
    def auth0_authorize_url(self) -> str:
        return f"https://{self.auth0_domain}/authorize"

    @property
    def auth0_userinfo_url(self) -> str:
        return f"https://{self.auth0_domain}/userinfo"


settings = Settings()
