"""Tests for webhook, template, and quota API endpoints."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.app import app
from src.database import init_db
from src.webhooks import init_webhook_tables
from src.quotas import init_quota_tables


@pytest.fixture
def client(db, monkeypatch):
    """Create a test client with fresh database."""
    import os
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(project_dir)
    monkeypatch.setattr("src.webhooks.DB_PATH", db)
    monkeypatch.setattr("src.quotas.DB_PATH", db)
    return TestClient(app)


@pytest.fixture
def auth_client(client):
    """Client with an authenticated session."""
    with client:
        client.cookies.set("session", "test")
        with patch("src.app.get_user", return_value={
            "sub": "auth0|user123",
            "name": "Test User",
            "email": "test@example.com",
            "picture": "",
        }):
            yield client


# --- Webhook API Tests ---

class TestWebhookEndpoints:
    def test_create_webhook(self, auth_client):
        resp = auth_client.post("/api/v1/webhooks", json={
            "url": "https://example.com/hook",
            "events": ["policy.violated", "anomaly.detected"],
            "description": "Test webhook",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "webhook_id" in data
        assert "secret" in data
        assert data["secret"].startswith("whsec_")
        assert data["events"] == ["policy.violated", "anomaly.detected"]

    def test_create_webhook_invalid_event(self, auth_client):
        resp = auth_client.post("/api/v1/webhooks", json={
            "url": "https://example.com/hook",
            "events": ["not.real"],
        })
        assert resp.status_code == 400

    def test_create_webhook_empty_events(self, auth_client):
        resp = auth_client.post("/api/v1/webhooks", json={
            "url": "https://example.com/hook",
            "events": [],
        })
        assert resp.status_code == 400

    def test_list_webhooks_empty(self, auth_client):
        resp = auth_client.get("/api/v1/webhooks")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_webhooks_after_create(self, auth_client):
        auth_client.post("/api/v1/webhooks", json={
            "url": "https://a.com/h",
            "events": ["policy.violated"],
        })
        resp = auth_client.get("/api/v1/webhooks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert "secret" not in data[0]  # Secret not in list response

    def test_get_webhook_detail(self, auth_client):
        create_resp = auth_client.post("/api/v1/webhooks", json={
            "url": "https://a.com/h",
            "events": ["policy.violated"],
        })
        webhook_id = create_resp.json()["webhook_id"]
        resp = auth_client.get(f"/api/v1/webhooks/{webhook_id}")
        assert resp.status_code == 200
        assert resp.json()["webhook_id"] == webhook_id

    def test_get_webhook_not_found(self, auth_client):
        resp = auth_client.get("/api/v1/webhooks/fake123")
        assert resp.status_code == 404

    def test_update_webhook(self, auth_client):
        create_resp = auth_client.post("/api/v1/webhooks", json={
            "url": "https://old.com/h",
            "events": ["policy.violated"],
        })
        webhook_id = create_resp.json()["webhook_id"]
        resp = auth_client.patch(f"/api/v1/webhooks/{webhook_id}", json={
            "url": "https://new.com/h",
            "is_active": False,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

    def test_update_webhook_not_found(self, auth_client):
        resp = auth_client.patch("/api/v1/webhooks/fake", json={"url": "https://x.com"})
        assert resp.status_code == 404

    def test_delete_webhook(self, auth_client):
        create_resp = auth_client.post("/api/v1/webhooks", json={
            "url": "https://a.com/h",
            "events": ["policy.violated"],
        })
        webhook_id = create_resp.json()["webhook_id"]
        resp = auth_client.delete(f"/api/v1/webhooks/{webhook_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_webhook_not_found(self, auth_client):
        resp = auth_client.delete("/api/v1/webhooks/fake")
        assert resp.status_code == 404

    def test_rotate_secret(self, auth_client):
        create_resp = auth_client.post("/api/v1/webhooks", json={
            "url": "https://a.com/h",
            "events": ["policy.violated"],
        })
        webhook_id = create_resp.json()["webhook_id"]
        old_secret = create_resp.json()["secret"]
        resp = auth_client.post(f"/api/v1/webhooks/{webhook_id}/rotate-secret")
        assert resp.status_code == 200
        new_secret = resp.json()["secret"]
        assert new_secret != old_secret
        assert new_secret.startswith("whsec_")

    def test_rotate_secret_not_found(self, auth_client):
        resp = auth_client.post("/api/v1/webhooks/fake/rotate-secret")
        assert resp.status_code == 404

    def test_get_deliveries_empty(self, auth_client):
        create_resp = auth_client.post("/api/v1/webhooks", json={
            "url": "https://a.com/h",
            "events": ["policy.violated"],
        })
        webhook_id = create_resp.json()["webhook_id"]
        resp = auth_client.get(f"/api/v1/webhooks/{webhook_id}/deliveries")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_webhook_requires_auth(self, client):
        resp = client.post("/api/v1/webhooks", json={
            "url": "https://a.com", "events": ["policy.violated"],
        })
        assert resp.status_code == 401

    def test_list_webhook_events(self, auth_client):
        resp = auth_client.get("/api/v1/webhooks/events/list")
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) >= 17
        assert all("event" in e and "name" in e for e in events)


# --- Template API Tests ---

class TestTemplateEndpoints:
    def test_list_templates(self, auth_client):
        resp = auth_client.get("/api/v1/templates")
        assert resp.status_code == 200
        templates = resp.json()
        assert len(templates) >= 8

    def test_list_templates_filter_category(self, auth_client):
        resp = auth_client.get("/api/v1/templates?category=security")
        assert resp.status_code == 200
        templates = resp.json()
        assert all(t["category"] == "security" for t in templates)

    def test_list_templates_filter_risk(self, auth_client):
        resp = auth_client.get("/api/v1/templates?risk_level=minimal")
        assert resp.status_code == 200
        templates = resp.json()
        assert all(t["risk_level"] == "minimal" for t in templates)

    def test_list_templates_filter_tag(self, auth_client):
        resp = auth_client.get("/api/v1/templates?tag=ci-cd")
        assert resp.status_code == 200
        templates = resp.json()
        assert all("ci-cd" in t["tags"] for t in templates)

    def test_get_template(self, auth_client):
        resp = auth_client.get("/api/v1/templates/read-only")
        assert resp.status_code == 200
        t = resp.json()
        assert t["id"] == "read-only"
        assert t["risk_level"] == "minimal"

    def test_get_template_not_found(self, auth_client):
        resp = auth_client.get("/api/v1/templates/nonexistent")
        assert resp.status_code == 404

    def test_preview_template(self, auth_client):
        resp = auth_client.get("/api/v1/templates/read-only/preview")
        assert resp.status_code == 200
        data = resp.json()
        assert "template" in data
        assert "policy_preview" in data
        assert "risk_assessment" in data

    def test_preview_template_not_found(self, auth_client):
        resp = auth_client.get("/api/v1/templates/fake/preview")
        assert resp.status_code == 404

    def test_compare_templates(self, auth_client):
        resp = auth_client.get("/api/v1/templates/compare/read-only/admin-full")
        assert resp.status_code == 200
        data = resp.json()
        assert "differences" in data
        assert data["template_a"]["id"] == "read-only"
        assert data["template_b"]["id"] == "admin-full"

    def test_compare_templates_not_found(self, auth_client):
        resp = auth_client.get("/api/v1/templates/compare/fake/admin-full")
        assert resp.status_code == 404

    def test_apply_template(self, auth_client):
        resp = auth_client.post("/api/v1/templates/apply", json={
            "template_id": "read-only",
            "agent_id": "agent-test",
            "agent_name": "Test Agent",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"
        assert data["agent_id"] == "agent-test"
        assert data["template_id"] == "read-only"

    def test_apply_template_with_overrides(self, auth_client):
        resp = auth_client.post("/api/v1/templates/apply", json={
            "template_id": "dev-standard",
            "agent_id": "agent-custom",
            "agent_name": "Custom Agent",
            "overrides": {"rate_limit_per_minute": 99},
        })
        assert resp.status_code == 200

    def test_apply_template_not_found(self, auth_client):
        resp = auth_client.post("/api/v1/templates/apply", json={
            "template_id": "nonexistent",
            "agent_id": "a1",
            "agent_name": "Agent",
        })
        assert resp.status_code == 404

    def test_apply_template_creates_policy(self, auth_client):
        auth_client.post("/api/v1/templates/apply", json={
            "template_id": "slack-notifier",
            "agent_id": "slack-agent",
            "agent_name": "Slack Bot",
        })
        resp = auth_client.get("/api/v1/policies")
        policies = resp.json()
        agent_ids = [p["agent_id"] for p in policies]
        assert "slack-agent" in agent_ids

    def test_templates_require_auth(self, client):
        resp = client.get("/api/v1/templates")
        assert resp.status_code == 401


# --- Quota API Tests ---

class TestQuotaEndpoints:
    def test_create_quota(self, auth_client):
        resp = auth_client.post("/api/v1/quotas", json={
            "agent_id": "agent-1",
            "quota_type": "daily",
            "max_tokens": 100,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "quota_id" in data
        assert data["agent_id"] == "agent-1"
        assert data["quota_type"] == "daily"
        assert data["max_tokens"] == 100

    def test_create_quota_monthly(self, auth_client):
        resp = auth_client.post("/api/v1/quotas", json={
            "agent_id": "agent-1",
            "quota_type": "monthly",
            "max_tokens": 1000,
            "action_on_exceed": "warn",
        })
        assert resp.status_code == 200
        assert resp.json()["quota_type"] == "monthly"

    def test_create_quota_invalid_type(self, auth_client):
        resp = auth_client.post("/api/v1/quotas", json={
            "agent_id": "a1",
            "quota_type": "weekly",
            "max_tokens": 100,
        })
        assert resp.status_code == 400

    def test_create_quota_invalid_action(self, auth_client):
        resp = auth_client.post("/api/v1/quotas", json={
            "agent_id": "a1",
            "quota_type": "daily",
            "max_tokens": 100,
            "action_on_exceed": "explode",
        })
        assert resp.status_code == 400

    def test_list_quotas_empty(self, auth_client):
        resp = auth_client.get("/api/v1/quotas")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_quotas_after_create(self, auth_client):
        auth_client.post("/api/v1/quotas", json={
            "agent_id": "a1", "max_tokens": 100,
        })
        resp = auth_client.get("/api/v1/quotas")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_list_quotas_filter_agent(self, auth_client):
        auth_client.post("/api/v1/quotas", json={
            "agent_id": "a1", "max_tokens": 100,
        })
        auth_client.post("/api/v1/quotas", json={
            "agent_id": "a2", "max_tokens": 200,
        })
        resp = auth_client.get("/api/v1/quotas?agent_id=a1")
        assert len(resp.json()) == 1

    def test_check_quota(self, auth_client):
        auth_client.post("/api/v1/quotas", json={
            "agent_id": "a1", "max_tokens": 100,
        })
        resp = auth_client.get("/api/v1/quotas/check/a1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["allowed"] is True
        assert data[0]["current_usage"] == 0

    def test_check_quota_no_quotas(self, auth_client):
        resp = auth_client.get("/api/v1/quotas/check/nonexistent")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_delete_quota(self, auth_client):
        create_resp = auth_client.post("/api/v1/quotas", json={
            "agent_id": "a1", "max_tokens": 100,
        })
        quota_id = create_resp.json()["quota_id"]
        resp = auth_client.delete(f"/api/v1/quotas/{quota_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_quota_not_found(self, auth_client):
        resp = auth_client.delete("/api/v1/quotas/fake")
        assert resp.status_code == 404

    def test_reset_quota(self, auth_client):
        create_resp = auth_client.post("/api/v1/quotas", json={
            "agent_id": "a1", "max_tokens": 100,
        })
        quota_id = create_resp.json()["quota_id"]
        resp = auth_client.post(f"/api/v1/quotas/{quota_id}/reset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reset"

    def test_reset_quota_not_found(self, auth_client):
        resp = auth_client.post("/api/v1/quotas/fake/reset")
        assert resp.status_code == 404

    def test_quota_history_empty(self, auth_client):
        create_resp = auth_client.post("/api/v1/quotas", json={
            "agent_id": "a1", "max_tokens": 100,
        })
        quota_id = create_resp.json()["quota_id"]
        resp = auth_client.get(f"/api/v1/quotas/{quota_id}/history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_quotas_require_auth(self, client):
        resp = client.post("/api/v1/quotas", json={
            "agent_id": "a1", "max_tokens": 100,
        })
        assert resp.status_code == 401
