"""Tests for delegation and simulation API endpoints."""

import time
import pytest
import pytest_asyncio
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport

from src.app import app
from src.database import (
    init_db, create_agent_policy, AgentPolicy,
)
from src.delegation import init_delegation_tables


USER_ID = "user|api-test"
USER_SESSION = {"sub": USER_ID, "name": "TestUser", "email": "test@test.com"}


def make_policy(agent_id, services=None, scopes=None):
    return AgentPolicy(
        agent_id=agent_id,
        agent_name=f"Agent-{agent_id}",
        allowed_services=services or ["github", "slack"],
        allowed_scopes=scopes or {
            "github": ["repo", "read:user", "write:org"],
            "slack": ["chat:write", "channels:read"],
        },
        rate_limit_per_minute=60,
        created_by=USER_ID,
        created_at=time.time(),
        is_active=True,
    )


@pytest_asyncio.fixture
async def api_db(db, monkeypatch):
    monkeypatch.setattr("src.delegation.DB_PATH", db)
    await init_delegation_tables()
    return db


@pytest_asyncio.fixture
async def client(api_db):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("src.app.get_user", return_value=USER_SESSION):
            with patch("src.app.require_user", return_value=USER_SESSION):
                yield client


@pytest_asyncio.fixture
async def agents(api_db):
    parent = make_policy("api-parent")
    child = make_policy("api-child")
    await create_agent_policy(parent)
    await create_agent_policy(child)
    return parent, child


# ── Delegation API ──


@pytest.mark.asyncio
async def test_create_delegation(client, agents):
    resp = await client.post("/api/v1/delegate", json={
        "parent_agent_id": "api-parent",
        "child_agent_id": "api-child",
        "services": ["github"],
        "scopes": {"github": ["repo"]},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["parent_agent_id"] == "api-parent"
    assert data["child_agent_id"] == "api-child"
    assert "delegation_id" in data


@pytest.mark.asyncio
async def test_create_delegation_self(client, agents):
    resp = await client.post("/api/v1/delegate", json={
        "parent_agent_id": "api-parent",
        "child_agent_id": "api-parent",
        "services": ["github"],
        "scopes": {},
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_create_delegation_nonexistent_parent(client, agents):
    resp = await client.post("/api/v1/delegate", json={
        "parent_agent_id": "nonexistent",
        "child_agent_id": "api-child",
        "services": ["github"],
        "scopes": {},
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_create_delegation_excess_service(client, agents):
    resp = await client.post("/api/v1/delegate", json={
        "parent_agent_id": "api-parent",
        "child_agent_id": "api-child",
        "services": ["google"],
        "scopes": {},
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_create_delegation_excess_scopes(client, agents):
    resp = await client.post("/api/v1/delegate", json={
        "parent_agent_id": "api-parent",
        "child_agent_id": "api-child",
        "services": ["github"],
        "scopes": {"github": ["admin:org"]},
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_list_delegations_empty(client, api_db):
    resp = await client.get("/api/v1/delegations")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_delegations_after_create(client, agents):
    await client.post("/api/v1/delegate", json={
        "parent_agent_id": "api-parent",
        "child_agent_id": "api-child",
        "services": ["github"],
        "scopes": {},
    })
    resp = await client.get("/api/v1/delegations")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1


@pytest.mark.asyncio
async def test_get_delegation_permissions(client, agents):
    await client.post("/api/v1/delegate", json={
        "parent_agent_id": "api-parent",
        "child_agent_id": "api-child",
        "services": ["github"],
        "scopes": {"github": ["repo"]},
    })
    resp = await client.get("/api/v1/delegations/api-child/permissions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["parent_agent_id"] == "api-parent"
    assert "github" in data["effective_services"]


@pytest.mark.asyncio
async def test_get_permissions_no_delegation(client, api_db):
    resp = await client.get("/api/v1/delegations/random/permissions")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_revoke_delegation(client, agents):
    resp = await client.post("/api/v1/delegate", json={
        "parent_agent_id": "api-parent",
        "child_agent_id": "api-child",
        "services": ["github"],
        "scopes": {},
    })
    delegation_id = resp.json()["delegation_id"]

    resp = await client.delete(f"/api/v1/delegations/{delegation_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "revoked"


@pytest.mark.asyncio
async def test_revoke_nonexistent_delegation(client, api_db):
    resp = await client.delete("/api/v1/delegations/fake-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delegation_with_expiration(client, agents):
    resp = await client.post("/api/v1/delegate", json={
        "parent_agent_id": "api-parent",
        "child_agent_id": "api-child",
        "services": ["github"],
        "scopes": {},
        "expires_at": time.time() + 3600,
    })
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_delegation_multiple_services(client, agents):
    resp = await client.post("/api/v1/delegate", json={
        "parent_agent_id": "api-parent",
        "child_agent_id": "api-child",
        "services": ["github", "slack"],
        "scopes": {"github": ["repo"], "slack": ["chat:write"]},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["delegated_services"]) == {"github", "slack"}


# ── Simulation API ──


@pytest.mark.asyncio
async def test_simulate_success(client, agents):
    resp = await client.post("/api/v1/simulate", json={
        "agent_id": "api-parent",
        "service": "github",
        "scopes": ["repo"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["would_succeed"] is True
    assert data["checks_failed"] == 0


@pytest.mark.asyncio
async def test_simulate_nonexistent_agent(client, api_db):
    resp = await client.post("/api/v1/simulate", json={
        "agent_id": "ghost",
        "service": "github",
        "scopes": ["repo"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["would_succeed"] is False


@pytest.mark.asyncio
async def test_simulate_unauthorized_service(client, agents):
    resp = await client.post("/api/v1/simulate", json={
        "agent_id": "api-parent",
        "service": "google",
        "scopes": ["mail.read"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["would_succeed"] is False


@pytest.mark.asyncio
async def test_simulate_excess_scopes(client, agents):
    resp = await client.post("/api/v1/simulate", json={
        "agent_id": "api-parent",
        "service": "github",
        "scopes": ["repo", "admin:org"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["would_succeed"] is False


@pytest.mark.asyncio
async def test_simulate_with_ip(client, api_db):
    await create_agent_policy(AgentPolicy(
        agent_id="ip-sim-agent",
        agent_name="IP Agent",
        allowed_services=["github"],
        allowed_scopes={"github": ["repo"]},
        rate_limit_per_minute=60,
        created_by=USER_ID,
        created_at=time.time(),
        is_active=True,
        ip_allowlist=["10.0.0.0/8"],
    ))
    resp = await client.post("/api/v1/simulate", json={
        "agent_id": "ip-sim-agent",
        "service": "github",
        "scopes": ["repo"],
        "ip_address": "10.1.2.3",
    })
    assert resp.json()["would_succeed"] is True

    resp = await client.post("/api/v1/simulate", json={
        "agent_id": "ip-sim-agent",
        "service": "github",
        "scopes": ["repo"],
        "ip_address": "192.168.1.1",
    })
    assert resp.json()["would_succeed"] is False


@pytest.mark.asyncio
async def test_simulate_empty_scopes(client, agents):
    resp = await client.post("/api/v1/simulate", json={
        "agent_id": "api-parent",
        "service": "github",
        "scopes": [],
    })
    assert resp.status_code == 200
    assert resp.json()["would_succeed"] is True


@pytest.mark.asyncio
async def test_simulate_disabled_agent(client, api_db):
    await create_agent_policy(AgentPolicy(
        agent_id="disabled-sim",
        agent_name="Disabled",
        allowed_services=["github"],
        allowed_scopes={"github": ["repo"]},
        rate_limit_per_minute=60,
        created_by=USER_ID,
        created_at=time.time(),
        is_active=False,
    ))
    resp = await client.post("/api/v1/simulate", json={
        "agent_id": "disabled-sim",
        "service": "github",
        "scopes": [],
    })
    assert resp.json()["would_succeed"] is False


@pytest.mark.asyncio
async def test_simulate_slack_service(client, agents):
    resp = await client.post("/api/v1/simulate", json={
        "agent_id": "api-parent",
        "service": "slack",
        "scopes": ["chat:write"],
    })
    assert resp.status_code == 200
    assert resp.json()["would_succeed"] is True


@pytest.mark.asyncio
async def test_simulate_response_structure(client, agents):
    resp = await client.post("/api/v1/simulate", json={
        "agent_id": "api-parent",
        "service": "github",
        "scopes": ["repo"],
    })
    data = resp.json()
    assert "would_succeed" in data
    assert "agent_id" in data
    assert "service" in data
    assert "requested_scopes" in data
    assert "effective_scopes" in data
    assert "checks" in data
    assert "checks_passed" in data
    assert "checks_failed" in data
    assert "total_checks" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_simulate_check_names(client, agents):
    resp = await client.post("/api/v1/simulate", json={
        "agent_id": "api-parent",
        "service": "github",
        "scopes": ["repo"],
    })
    check_names = {c["name"] for c in resp.json()["checks"]}
    expected = {
        "agent_exists", "agent_active", "ownership", "expiration",
        "time_window", "ip_allowlist", "service_auth",
        "scope_validation", "rate_limit", "step_up_auth",
    }
    assert check_names == expected
