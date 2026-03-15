"""Tests for the analytics and compliance API endpoints."""

import time
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.database import init_db


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Create a test client with an authenticated session."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        # Simulate authentication by setting session cookie
        ac.cookies.set("session", "test")
        yield ac


@pytest.fixture(autouse=True)
def mock_auth():
    """Mock authentication to always return a test user."""
    with patch("src.app.get_user") as mock:
        mock.return_value = {
            "sub": "test-user-123",
            "name": "Test User",
            "email": "test@example.com",
        }
        yield mock


@pytest.fixture(autouse=True)
def mock_require_user():
    """Mock require_user to always return a test user."""
    with patch("src.app.require_user") as mock:
        mock.return_value = {
            "sub": "test-user-123",
            "name": "Test User",
            "email": "test@example.com",
        }
        yield mock


# ============================================================
# Analytics endpoint tests
# ============================================================

class TestAnalyticsEndpoint:
    @pytest.mark.anyio
    async def test_analytics_default_hours(self, client):
        with patch("src.app.generate_usage_analytics", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "period_hours": 24,
                "total_agents": 0,
                "total_requests": 0,
                "total_denied": 0,
                "anomaly_count": 0,
                "high_risk_agents": [],
                "agents": [],
                "anomalies": [],
            }
            resp = await client.get("/api/v1/analytics")
            assert resp.status_code == 200
            data = resp.json()
            assert data["period_hours"] == 24
            mock.assert_called_once_with("test-user-123", hours=24)

    @pytest.mark.anyio
    async def test_analytics_custom_hours(self, client):
        with patch("src.app.generate_usage_analytics", new_callable=AsyncMock) as mock:
            mock.return_value = {"period_hours": 48, "agents": [], "anomalies": []}
            resp = await client.get("/api/v1/analytics?hours=48")
            assert resp.status_code == 200
            mock.assert_called_once_with("test-user-123", hours=48)

    @pytest.mark.anyio
    async def test_analytics_min_hours_clamped(self, client):
        with patch("src.app.generate_usage_analytics", new_callable=AsyncMock) as mock:
            mock.return_value = {"period_hours": 1, "agents": [], "anomalies": []}
            resp = await client.get("/api/v1/analytics?hours=0")
            assert resp.status_code == 200
            mock.assert_called_once_with("test-user-123", hours=1)

    @pytest.mark.anyio
    async def test_analytics_max_hours_clamped(self, client):
        with patch("src.app.generate_usage_analytics", new_callable=AsyncMock) as mock:
            mock.return_value = {"period_hours": 8760, "agents": [], "anomalies": []}
            resp = await client.get("/api/v1/analytics?hours=99999")
            assert resp.status_code == 200
            mock.assert_called_once_with("test-user-123", hours=8760)

    @pytest.mark.anyio
    async def test_analytics_with_agents(self, client):
        with patch("src.app.generate_usage_analytics", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "period_hours": 24,
                "total_agents": 2,
                "total_requests": 150,
                "total_denied": 10,
                "anomaly_count": 1,
                "high_risk_agents": ["agent-bad"],
                "agents": [
                    {
                        "agent_id": "agent-bad",
                        "total_requests": 100,
                        "risk_score": 0.75,
                    },
                    {
                        "agent_id": "agent-good",
                        "total_requests": 50,
                        "risk_score": 0.0,
                    },
                ],
                "anomalies": [
                    {
                        "type": "burst",
                        "severity": "high",
                        "agent_id": "agent-bad",
                        "description": "25 requests in 60s",
                    }
                ],
            }
            resp = await client.get("/api/v1/analytics")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_agents"] == 2
            assert "agent-bad" in data["high_risk_agents"]
            assert len(data["anomalies"]) == 1

    @pytest.mark.anyio
    async def test_analytics_with_anomalies(self, client):
        with patch("src.app.generate_usage_analytics", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "period_hours": 24,
                "total_agents": 1,
                "total_requests": 50,
                "total_denied": 25,
                "anomaly_count": 3,
                "high_risk_agents": ["a1"],
                "agents": [{"agent_id": "a1", "risk_score": 0.8}],
                "anomalies": [
                    {"type": "burst", "severity": "high", "agent_id": "a1"},
                    {"type": "off_hours", "severity": "medium", "agent_id": "a1"},
                    {"type": "high_denial_rate", "severity": "critical", "agent_id": "a1"},
                ],
            }
            resp = await client.get("/api/v1/analytics")
            data = resp.json()
            assert data["anomaly_count"] == 3
            severities = {a["severity"] for a in data["anomalies"]}
            assert "critical" in severities


# ============================================================
# Compliance endpoint tests
# ============================================================

class TestComplianceEndpoint:
    @pytest.mark.anyio
    async def test_compliance_default_days(self, client):
        with patch("src.app.generate_compliance_report", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "report_type": "compliance",
                "period_days": 30,
                "summary": {},
            }
            resp = await client.get("/api/v1/compliance")
            assert resp.status_code == 200
            data = resp.json()
            assert data["report_type"] == "compliance"
            mock.assert_called_once_with("test-user-123", period_days=30)

    @pytest.mark.anyio
    async def test_compliance_custom_days(self, client):
        with patch("src.app.generate_compliance_report", new_callable=AsyncMock) as mock:
            mock.return_value = {"report_type": "compliance", "period_days": 7, "summary": {}}
            resp = await client.get("/api/v1/compliance?days=7")
            assert resp.status_code == 200
            mock.assert_called_once_with("test-user-123", period_days=7)

    @pytest.mark.anyio
    async def test_compliance_min_days_clamped(self, client):
        with patch("src.app.generate_compliance_report", new_callable=AsyncMock) as mock:
            mock.return_value = {"report_type": "compliance", "period_days": 1, "summary": {}}
            resp = await client.get("/api/v1/compliance?days=0")
            assert resp.status_code == 200
            mock.assert_called_once_with("test-user-123", period_days=1)

    @pytest.mark.anyio
    async def test_compliance_max_days_clamped(self, client):
        with patch("src.app.generate_compliance_report", new_callable=AsyncMock) as mock:
            mock.return_value = {"report_type": "compliance", "period_days": 365, "summary": {}}
            resp = await client.get("/api/v1/compliance?days=9999")
            assert resp.status_code == 200
            mock.assert_called_once_with("test-user-123", period_days=365)

    @pytest.mark.anyio
    async def test_compliance_full_report(self, client):
        with patch("src.app.generate_compliance_report", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "report_type": "compliance",
                "generated_at": time.time(),
                "period_days": 30,
                "summary": {
                    "total_agents": 5,
                    "active_agents": 3,
                    "total_access_events": 500,
                    "denied_events": 25,
                    "denial_rate": 0.05,
                    "emergency_revocations": 1,
                    "step_up_challenges": 10,
                    "policy_changes": 8,
                    "unique_source_ips": 12,
                    "anomalies_detected": 3,
                },
                "risk_overview": {
                    "high_risk_agents": [
                        {"agent_id": "a1", "risk_score": 0.7, "factors": ["burst"]}
                    ],
                    "critical_anomalies": [
                        {"type": "high_denial_rate", "agent_id": "a1", "description": "80% denied"}
                    ],
                },
                "policy_changes": [
                    {"timestamp": time.time(), "agent_id": "a2", "action": "policy_created"},
                ],
                "emergency_revocations": [
                    {"timestamp": time.time(), "ip_address": "10.0.0.1"},
                ],
            }
            resp = await client.get("/api/v1/compliance")
            assert resp.status_code == 200
            data = resp.json()
            assert data["summary"]["total_agents"] == 5
            assert data["summary"]["emergency_revocations"] == 1
            assert len(data["risk_overview"]["high_risk_agents"]) == 1
            assert len(data["policy_changes"]) == 1

    @pytest.mark.anyio
    async def test_compliance_empty_report(self, client):
        with patch("src.app.generate_compliance_report", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "report_type": "compliance",
                "period_days": 30,
                "summary": {
                    "total_agents": 0,
                    "total_access_events": 0,
                    "denied_events": 0,
                    "emergency_revocations": 0,
                },
                "risk_overview": {"high_risk_agents": [], "critical_anomalies": []},
                "policy_changes": [],
                "emergency_revocations": [],
            }
            resp = await client.get("/api/v1/compliance")
            assert resp.status_code == 200
            data = resp.json()
            assert data["summary"]["total_agents"] == 0


# ============================================================
# Auth requirement tests
# ============================================================

class TestAnalyticsAuthRequired:
    @pytest.mark.anyio
    async def test_analytics_requires_auth(self):
        with patch("src.app.require_user") as mock:
            from fastapi import HTTPException
            mock.side_effect = HTTPException(status_code=401, detail="Not authenticated")
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as ac:
                resp = await ac.get("/api/v1/analytics")
                assert resp.status_code == 401

    @pytest.mark.anyio
    async def test_compliance_requires_auth(self):
        with patch("src.app.require_user") as mock:
            from fastapi import HTTPException
            mock.side_effect = HTTPException(status_code=401, detail="Not authenticated")
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as ac:
                resp = await ac.get("/api/v1/compliance")
                assert resp.status_code == 401
