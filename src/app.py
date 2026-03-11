"""AgentGate - FastAPI application with Auth0 Token Vault integration."""

import secrets
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from .auth import (
    SUPPORTED_SERVICES,
    get_token_vault_token,
    initiate_connection,
    oauth,
    trigger_step_up_auth,
    check_step_up_status,
)
from .config import settings
from .database import (
    add_connected_service,
    create_agent_policy,
    create_api_key,
    delete_agent_policy,
    emergency_revoke_all,
    get_all_policies,
    get_api_keys,
    get_audit_log,
    get_connected_services,
    init_db,
    log_audit,
    remove_connected_service,
    revoke_api_key,
    toggle_agent_policy,
    validate_api_key,
    AgentPolicy,
)
from .policy import enforce_policy, requires_step_up, get_effective_scopes, PolicyDenied


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="AgentGate",
    description="Secure AI Agent Gateway with Auth0 Token Vault",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(SessionMiddleware, secret_key=settings.app_secret_key)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# --- Helpers ---

def get_user(request: Request) -> dict | None:
    """Get the current user from the session."""
    return request.session.get("user")


async def get_user_or_api_key(request: Request) -> tuple[dict, str | None]:
    """Authenticate via session OR API key. Returns (user_dict, agent_id_or_None).

    For session auth: returns (user, None)
    For API key auth: returns (synthetic_user, agent_id)
    """
    # Check Authorization header first (for agent API key auth)
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer ag_"):
        raw_key = auth_header[7:]  # Strip "Bearer "
        api_key = await validate_api_key(raw_key)
        if not api_key:
            raise HTTPException(status_code=401, detail="Invalid or expired API key")
        user = {"sub": api_key.user_id, "name": f"api_key:{api_key.name}", "email": ""}
        return user, api_key.agent_id

    # Fall back to session auth
    user = get_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user, None


def require_user(request: Request) -> dict:
    """Require an authenticated user, redirect to login otherwise."""
    user = get_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# --- Auth Routes ---

@app.get("/login")
async def login(request: Request):
    """Initiate Auth0 login flow."""
    redirect_uri = settings.auth0_callback_url
    return await oauth.auth0.authorize_redirect(request, redirect_uri)


@app.get("/callback")
async def callback(request: Request):
    """Handle Auth0 OAuth callback."""
    token = await oauth.auth0.authorize_access_token(request)
    userinfo = token.get("userinfo", {})
    request.session["user"] = {
        "sub": userinfo.get("sub", ""),
        "name": userinfo.get("name", ""),
        "email": userinfo.get("email", ""),
        "picture": userinfo.get("picture", ""),
    }
    return RedirectResponse(url="/")


@app.get("/logout")
async def logout(request: Request):
    """Clear session and redirect to Auth0 logout."""
    request.session.clear()
    return RedirectResponse(
        url=f"https://{settings.auth0_domain}/v2/logout?"
        f"client_id={settings.auth0_client_id}&returnTo=http://localhost:8000"
    )


# --- Dashboard Routes ---

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard showing connected services and agent activity."""
    user = get_user(request)
    if not user:
        return templates.TemplateResponse(request, "login.html")

    services = await get_connected_services(user["sub"])
    policies = await get_all_policies(user["sub"])
    audit = await get_audit_log(user["sub"], limit=20)

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": user,
            "connected_services": services,
            "available_services": SUPPORTED_SERVICES,
            "agent_policies": policies,
            "audit_entries": audit,
        },
    )


@app.get("/audit", response_class=HTMLResponse)
async def audit_page(request: Request):
    """Full audit trail view."""
    user = require_user(request)
    entries = await get_audit_log(user["sub"], limit=200)
    return templates.TemplateResponse(
        request,
        "audit.html",
        {"user": user, "audit_entries": entries},
    )


# --- Service Connection Routes ---

@app.post("/connect/{service}")
async def connect_service(service: str, request: Request):
    """Initiate connection to a service via Auth0 Token Vault."""
    user = require_user(request)
    if service not in SUPPORTED_SERVICES:
        raise HTTPException(status_code=400, detail=f"Unknown service: {service}")

    redirect_uri = f"{settings.auth0_callback_url}?connection={service}"
    url = await initiate_connection(service, redirect_uri)

    await log_audit(user["sub"], "user", service, "connect_initiated", "pending")
    return RedirectResponse(url=url, status_code=303)


@app.delete("/connect/{service}")
async def disconnect_service(service: str, request: Request):
    """Disconnect a service and revoke Token Vault tokens."""
    user = require_user(request)
    await remove_connected_service(user["sub"], service)
    await log_audit(user["sub"], "user", service, "disconnect", "success")
    return {"status": "disconnected", "service": service}


# --- Agent Policy Routes ---

class CreatePolicyRequest(BaseModel):
    agent_id: str
    agent_name: str
    allowed_services: list[str]
    allowed_scopes: dict[str, list[str]]
    rate_limit_per_minute: int = 60
    requires_step_up: list[str] = []
    allowed_hours: list[int] = []   # 0-23 UTC, empty = always allowed
    allowed_days: list[int] = []    # 0=Mon..6=Sun, empty = always allowed
    expires_at: float = 0.0         # Unix timestamp, 0 = never
    ip_allowlist: list[str] = []    # CIDR or IP, empty = allow all


@app.post("/api/v1/policies")
async def create_policy(body: CreatePolicyRequest, request: Request):
    """Create or update an agent access policy."""
    user = require_user(request)

    # Validate services
    for svc in body.allowed_services:
        if svc not in SUPPORTED_SERVICES:
            raise HTTPException(status_code=400, detail=f"Unknown service: {svc}")

    # Validate scopes
    for svc, scopes in body.allowed_scopes.items():
        if svc not in SUPPORTED_SERVICES:
            raise HTTPException(status_code=400, detail=f"Unknown service: {svc}")
        valid_scopes = set(SUPPORTED_SERVICES[svc]["scopes"])
        invalid = set(scopes) - valid_scopes
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid scopes for {svc}: {', '.join(invalid)}",
            )

    # Validate time window constraints
    if body.allowed_hours:
        invalid_hours = [h for h in body.allowed_hours if h < 0 or h > 23]
        if invalid_hours:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid hours (must be 0-23): {invalid_hours}",
            )

    if body.allowed_days:
        invalid_days = [d for d in body.allowed_days if d < 0 or d > 6]
        if invalid_days:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid days (must be 0=Mon..6=Sun): {invalid_days}",
            )

    policy = AgentPolicy(
        agent_id=body.agent_id,
        agent_name=body.agent_name,
        allowed_services=body.allowed_services,
        allowed_scopes=body.allowed_scopes,
        rate_limit_per_minute=body.rate_limit_per_minute,
        requires_step_up=body.requires_step_up,
        allowed_hours=body.allowed_hours,
        allowed_days=body.allowed_days,
        expires_at=body.expires_at,
        ip_allowlist=body.ip_allowlist,
        created_by=user["sub"],
        created_at=time.time(),
    )
    await create_agent_policy(policy)
    await log_audit(user["sub"], body.agent_id, "", "policy_created", "success",
                   details=f"Services: {body.allowed_services}")
    return {"status": "created", "agent_id": body.agent_id}


@app.get("/api/v1/policies")
async def list_policies(request: Request):
    """List all agent policies for the current user."""
    user = require_user(request)
    policies = await get_all_policies(user["sub"])
    return [
        {
            "agent_id": p.agent_id,
            "agent_name": p.agent_name,
            "allowed_services": p.allowed_services,
            "rate_limit_per_minute": p.rate_limit_per_minute,
            "is_active": p.is_active,
            "allowed_hours": p.allowed_hours,
            "allowed_days": p.allowed_days,
            "expires_at": p.expires_at or None,
            "ip_allowlist": p.ip_allowlist,
        }
        for p in policies
    ]


# --- Agent Token API ---

class TokenRequest(BaseModel):
    agent_id: str
    service: str
    scopes: list[str] = []


class StepUpRequest(BaseModel):
    agent_id: str
    action: str


@app.post("/api/v1/token")
async def request_token(body: TokenRequest, request: Request):
    """Request a scoped token for a service on behalf of a user.

    This is the primary endpoint agents call to get access tokens.
    Supports both session auth and API key auth (Bearer ag_xxx).
    Enforces policy checks, rate limits, and step-up auth requirements.
    """
    user, api_key_agent_id = await get_user_or_api_key(request)
    ip = request.client.host if request.client else ""

    # If authenticated via API key, enforce that the agent_id matches
    effective_agent_id = body.agent_id
    if api_key_agent_id:
        if body.agent_id and body.agent_id != api_key_agent_id:
            raise HTTPException(
                status_code=403,
                detail=f"API key is bound to agent '{api_key_agent_id}', "
                       f"cannot request tokens for '{body.agent_id}'",
            )
        effective_agent_id = api_key_agent_id

    try:
        policy = await enforce_policy(
            user_id=user["sub"],
            agent_id=effective_agent_id,
            service=body.service,
            requested_scopes=body.scopes,
            ip_address=ip,
        )
    except PolicyDenied as e:
        raise HTTPException(status_code=403, detail=str(e))

    # Check if step-up auth is required
    if requires_step_up(policy, body.service):
        step_up = await trigger_step_up_auth(
            user["sub"], body.agent_id, f"access {body.service}"
        )
        if step_up.get("status") != "approved":
            await log_audit(
                user["sub"], body.agent_id, body.service,
                "token_request", "step_up_required",
                scopes=",".join(body.scopes), ip_address=ip,
            )
            return JSONResponse(
                status_code=202,
                content={
                    "status": "step_up_required",
                    "auth_req_id": step_up.get("auth_req_id"),
                    "message": "User approval required. Poll /api/v1/step-up/status.",
                },
            )

    # Get effective scopes (intersection of requested and allowed)
    effective_scopes = get_effective_scopes(policy, body.service, body.scopes)

    # Exchange token via Token Vault
    result = await get_token_vault_token(user["sub"], body.service, effective_scopes)

    if "error" in result:
        await log_audit(
            user["sub"], body.agent_id, body.service,
            "token_request", "error",
            scopes=",".join(effective_scopes), ip_address=ip,
            details=result["error"],
        )
        raise HTTPException(status_code=502, detail=result["error"])

    # Log successful token issuance
    await log_audit(
        user["sub"], body.agent_id, body.service,
        "token_issued", "success",
        scopes=",".join(effective_scopes), ip_address=ip,
    )

    return {
        "access_token": result["access_token"],
        "token_type": result["token_type"],
        "expires_in": result["expires_in"],
        "scope": result["scope"],
        "agent_id": body.agent_id,
        "service": body.service,
    }


@app.get("/api/v1/services")
async def list_agent_services(agent_id: str, request: Request):
    """List services available to a specific agent."""
    user = require_user(request)
    from .database import get_agent_policy
    policy = await get_agent_policy(agent_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Agent not found")

    connected = await get_connected_services(user["sub"])
    connected_names = {s["service"] for s in connected}

    return [
        {
            "service": svc,
            "display_name": SUPPORTED_SERVICES[svc]["display_name"],
            "allowed_scopes": policy.allowed_scopes.get(svc, []),
            "connected": svc in connected_names,
            "requires_step_up": svc in policy.requires_step_up,
        }
        for svc in policy.allowed_services
        if svc in SUPPORTED_SERVICES
    ]


@app.post("/api/v1/step-up")
async def initiate_step_up(body: StepUpRequest, request: Request):
    """Trigger step-up authentication for a sensitive operation."""
    user = require_user(request)
    result = await trigger_step_up_auth(user["sub"], body.agent_id, body.action)
    await log_audit(
        user["sub"], body.agent_id, "",
        "step_up_initiated", result.get("status", "unknown"),
        details=body.action,
    )
    return result


@app.get("/api/v1/step-up/status/{auth_req_id}")
async def step_up_status(auth_req_id: str, request: Request):
    """Check the status of a pending step-up authentication."""
    require_user(request)
    return await check_step_up_status(auth_req_id)


# --- API Key Management ---

class CreateApiKeyRequest(BaseModel):
    agent_id: str
    name: str = "default"
    expires_in: int = 0  # seconds, 0 = never


@app.post("/api/v1/keys")
async def create_key(body: CreateApiKeyRequest, request: Request):
    """Generate an API key for an agent. The raw key is only shown once."""
    user = require_user(request)
    api_key, raw_key = await create_api_key(
        user_id=user["sub"],
        agent_id=body.agent_id,
        name=body.name,
        expires_in=body.expires_in,
    )
    await log_audit(user["sub"], body.agent_id, "", "api_key_created", "success",
                   details=f"Key: {api_key.key_prefix}...")
    return {
        "key": raw_key,
        "key_id": api_key.id,
        "key_prefix": api_key.key_prefix,
        "agent_id": body.agent_id,
        "expires_at": api_key.expires_at or None,
        "warning": "Save this key now. It cannot be retrieved again.",
    }


@app.get("/api/v1/keys")
async def list_keys(request: Request):
    """List all API keys (without the actual key values)."""
    user = require_user(request)
    keys = await get_api_keys(user["sub"])
    return [
        {
            "key_id": k.id,
            "key_prefix": k.key_prefix,
            "agent_id": k.agent_id,
            "name": k.name,
            "created_at": k.created_at,
            "expires_at": k.expires_at or None,
            "is_revoked": k.is_revoked,
            "last_used_at": k.last_used_at or None,
        }
        for k in keys
    ]


@app.delete("/api/v1/keys/{key_id}")
async def revoke_key(key_id: str, request: Request):
    """Revoke an API key."""
    user = require_user(request)
    revoked = await revoke_api_key(key_id, user["sub"])
    if not revoked:
        raise HTTPException(status_code=404, detail="Key not found")
    await log_audit(user["sub"], "", "", "api_key_revoked", "success",
                   details=f"Key ID: {key_id}")
    return {"status": "revoked", "key_id": key_id}


# --- Policy Toggle ---

@app.post("/api/v1/policies/{agent_id}/toggle")
async def toggle_policy(agent_id: str, request: Request):
    """Enable or disable an agent without deleting its policy."""
    user = require_user(request)
    new_state = await toggle_agent_policy(agent_id, user["sub"])
    if new_state is None:
        raise HTTPException(status_code=404, detail="Agent policy not found")
    status = "enabled" if new_state else "disabled"
    await log_audit(user["sub"], agent_id, "", f"agent_{status}", "success")
    return {"agent_id": agent_id, "is_active": new_state, "status": status}


# --- Policy Deletion ---

@app.delete("/api/v1/policies/{agent_id}")
async def delete_policy(agent_id: str, request: Request):
    """Permanently delete an agent policy and log the action."""
    user = require_user(request)
    deleted = await delete_agent_policy(agent_id, user["sub"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent policy not found")
    await log_audit(user["sub"], agent_id, "", "policy_deleted", "success")
    return {"status": "deleted", "agent_id": agent_id}


# --- Emergency Kill Switch ---

@app.post("/api/v1/emergency-revoke")
async def emergency_revoke(request: Request):
    """Emergency kill switch: instantly disable ALL agent policies and revoke ALL API keys.

    This is an irreversible safety mechanism. Use when you suspect an agent is
    compromised or behaving unexpectedly. All agents will immediately lose access
    and all API keys will be invalidated.
    """
    user = require_user(request)
    ip = request.client.host if request.client else ""
    result = await emergency_revoke_all(user["sub"])
    await log_audit(
        user["sub"], "*", "", "emergency_revoke", "success",
        ip_address=ip,
        details=f"Disabled {result['policies_disabled']} policies, "
                f"revoked {result['keys_revoked']} API keys",
    )
    return {
        "status": "all_access_revoked",
        "policies_disabled": result["policies_disabled"],
        "keys_revoked": result["keys_revoked"],
        "message": "All agent access has been immediately revoked. "
                   "Re-enable individual agents via policy toggle.",
    }


# --- Health ---

@app.get("/health")
async def health():
    return {"status": "ok", "service": "agentgate", "version": "1.0.0"}
