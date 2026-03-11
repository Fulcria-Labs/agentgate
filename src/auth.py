"""Auth0 authentication and Token Vault integration."""

import secrets
from urllib.parse import urlencode

import httpx
from authlib.integrations.starlette_client import OAuth

from .config import settings

oauth = OAuth()

# Register Auth0 as OAuth provider
oauth.register(
    name="auth0",
    client_id=settings.auth0_client_id,
    client_secret=settings.auth0_client_secret,
    client_kwargs={"scope": "openid profile email"},
    server_metadata_url=f"https://{settings.auth0_domain}/.well-known/openid-configuration",
)

# Supported services and their OAuth scopes
SUPPORTED_SERVICES = {
    "github": {
        "display_name": "GitHub",
        "icon": "github",
        "scopes": ["repo", "read:user", "read:org", "gist", "notifications"],
        "description": "Access repositories, issues, and pull requests",
    },
    "slack": {
        "display_name": "Slack",
        "icon": "slack",
        "scopes": ["channels:read", "chat:write", "users:read", "files:read"],
        "description": "Read channels, send messages, manage files",
    },
    "google": {
        "display_name": "Google",
        "icon": "google",
        "scopes": ["gmail.readonly", "calendar.readonly", "drive.readonly"],
        "description": "Read Gmail, Calendar, and Drive",
    },
    "linear": {
        "display_name": "Linear",
        "icon": "linear",
        "scopes": ["read", "write", "issues:create"],
        "description": "Manage issues and projects",
    },
    "notion": {
        "display_name": "Notion",
        "icon": "notion",
        "scopes": ["read_content", "update_content"],
        "description": "Read and update Notion pages",
    },
}


async def get_token_vault_token(user_id: str, service: str, scopes: list[str]) -> dict:
    """Exchange a user's connected account token via Auth0 Token Vault.

    Uses OAuth 2.0 Token Exchange (RFC 8693) to get a scoped token
    for a specific service on behalf of the user.
    """
    async with httpx.AsyncClient() as client:
        # Get management API token
        mgmt_token = await _get_management_token()

        # Request token exchange from Token Vault
        response = await client.post(
            settings.auth0_token_url,
            json={
                "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                "client_id": settings.auth0_client_id,
                "client_secret": settings.auth0_client_secret,
                "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "subject_token": mgmt_token,
                "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "audience": f"https://{service}.com/api",
                "scope": " ".join(scopes),
            },
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            return {"error": response.json().get("error_description", "Token exchange failed")}

        data = response.json()
        return {
            "access_token": data["access_token"],
            "token_type": data.get("token_type", "Bearer"),
            "expires_in": data.get("expires_in", 3600),
            "scope": data.get("scope", " ".join(scopes)),
        }


async def initiate_connection(service: str, redirect_uri: str) -> str:
    """Generate the Auth0 authorization URL to connect a service via Token Vault.

    This initiates the Connected Accounts flow where the user authorizes
    the service through Auth0's Token Vault.
    """
    state = secrets.token_urlsafe(32)
    params = {
        "response_type": "code",
        "client_id": settings.auth0_client_id,
        "redirect_uri": redirect_uri,
        "scope": "openid profile email",
        "state": state,
        "connection": service,
        "access_type": "offline",
    }
    url = f"{settings.auth0_authorize_url}?{urlencode(params)}"
    return url


async def trigger_step_up_auth(user_id: str, agent_id: str, action: str) -> dict:
    """Trigger CIBA step-up authentication for sensitive operations.

    Uses Client-Initiated Backchannel Authentication to push a consent
    request to the user before allowing the agent to proceed.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://{settings.auth0_domain}/bc-authorize",
            json={
                "client_id": settings.auth0_client_id,
                "client_secret": settings.auth0_client_secret,
                "scope": "openid",
                "login_hint": f"user:{user_id}",
                "binding_message": f"Agent '{agent_id}' requests: {action}",
            },
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            return {"error": "Step-up authentication failed", "status": "denied"}

        data = response.json()
        return {
            "auth_req_id": data.get("auth_req_id"),
            "expires_in": data.get("expires_in", 120),
            "status": "pending",
        }


async def check_step_up_status(auth_req_id: str) -> dict:
    """Poll the CIBA endpoint to check if the user approved the step-up."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            settings.auth0_token_url,
            json={
                "grant_type": "urn:openid:params:grant-type:ciba",
                "client_id": settings.auth0_client_id,
                "client_secret": settings.auth0_client_secret,
                "auth_req_id": auth_req_id,
            },
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            return {"status": "approved", "token": response.json().get("access_token")}
        elif response.status_code == 400:
            error = response.json().get("error", "")
            if error == "authorization_pending":
                return {"status": "pending"}
            elif error == "slow_down":
                return {"status": "pending", "retry_after": 5}
        return {"status": "denied"}


async def _get_management_token() -> str:
    """Get an Auth0 Management API token."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            settings.auth0_token_url,
            json={
                "grant_type": "client_credentials",
                "client_id": settings.auth0_client_id,
                "client_secret": settings.auth0_client_secret,
                "audience": settings.auth0_audience,
            },
            headers={"Content-Type": "application/json"},
        )
        data = response.json()
        return data.get("access_token", "")
