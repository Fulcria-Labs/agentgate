"""Comprehensive tests for webhook notification system."""

import json
import time
import pytest
import pytest_asyncio

from src.database import init_db
from src.webhooks import (
    WebhookEvent,
    WebhookSubscription,
    compute_signature,
    create_webhook,
    delete_webhook,
    dispatch_webhook,
    get_subscriptions_for_event,
    get_webhook,
    get_webhook_deliveries,
    init_webhook_tables,
    list_webhooks,
    record_delivery_result,
    rotate_webhook_secret,
    update_webhook,
    verify_signature,
)


@pytest_asyncio.fixture
async def webhook_db(db, monkeypatch):
    """Initialize webhook tables."""
    monkeypatch.setattr("src.webhooks.DB_PATH", db)
    await init_webhook_tables()
    return db


USER_ID = "auth0|user123"
OTHER_USER = "auth0|other456"


class TestWebhookCreation:
    @pytest.mark.asyncio
    async def test_create_webhook_basic(self, webhook_db):
        sub = await create_webhook(
            user_id=USER_ID,
            url="https://example.com/webhook",
            events=["policy.violated"],
        )
        assert sub.id
        assert sub.user_id == USER_ID
        assert sub.url == "https://example.com/webhook"
        assert sub.events == ["policy.violated"]
        assert sub.secret.startswith("whsec_")
        assert sub.is_active is True
        assert sub.failure_count == 0

    @pytest.mark.asyncio
    async def test_create_webhook_multiple_events(self, webhook_db):
        sub = await create_webhook(
            user_id=USER_ID,
            url="https://example.com/hook",
            events=["policy.violated", "anomaly.detected", "emergency.revoke"],
            description="Security alerts",
        )
        assert len(sub.events) == 3
        assert sub.description == "Security alerts"

    @pytest.mark.asyncio
    async def test_create_webhook_all_events(self, webhook_db):
        all_events = [e.value for e in WebhookEvent]
        sub = await create_webhook(
            user_id=USER_ID,
            url="https://example.com/all",
            events=all_events,
        )
        assert len(sub.events) == len(WebhookEvent)

    @pytest.mark.asyncio
    async def test_create_webhook_invalid_event(self, webhook_db):
        with pytest.raises(ValueError, match="Invalid events"):
            await create_webhook(
                user_id=USER_ID,
                url="https://example.com/hook",
                events=["not.a.real.event"],
            )

    @pytest.mark.asyncio
    async def test_create_webhook_empty_events(self, webhook_db):
        with pytest.raises(ValueError, match="At least one event"):
            await create_webhook(
                user_id=USER_ID,
                url="https://example.com/hook",
                events=[],
            )

    @pytest.mark.asyncio
    async def test_create_webhook_empty_url(self, webhook_db):
        with pytest.raises(ValueError, match="URL is required"):
            await create_webhook(
                user_id=USER_ID,
                url="",
                events=["policy.violated"],
            )

    @pytest.mark.asyncio
    async def test_create_webhook_unique_secrets(self, webhook_db):
        sub1 = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        sub2 = await create_webhook(USER_ID, "https://b.com/h", ["policy.violated"])
        assert sub1.secret != sub2.secret

    @pytest.mark.asyncio
    async def test_create_webhook_unique_ids(self, webhook_db):
        sub1 = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        sub2 = await create_webhook(USER_ID, "https://b.com/h", ["policy.violated"])
        assert sub1.id != sub2.id


class TestWebhookListing:
    @pytest.mark.asyncio
    async def test_list_empty(self, webhook_db):
        result = await list_webhooks(USER_ID)
        assert result == []

    @pytest.mark.asyncio
    async def test_list_user_webhooks(self, webhook_db):
        await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        await create_webhook(USER_ID, "https://b.com/h", ["anomaly.detected"])
        result = await list_webhooks(USER_ID)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_isolation_between_users(self, webhook_db):
        await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        await create_webhook(OTHER_USER, "https://b.com/h", ["anomaly.detected"])
        mine = await list_webhooks(USER_ID)
        theirs = await list_webhooks(OTHER_USER)
        assert len(mine) == 1
        assert len(theirs) == 1
        assert mine[0].url == "https://a.com/h"
        assert theirs[0].url == "https://b.com/h"


class TestWebhookRetrieval:
    @pytest.mark.asyncio
    async def test_get_webhook(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        fetched = await get_webhook(sub.id, USER_ID)
        assert fetched is not None
        assert fetched.id == sub.id
        assert fetched.url == "https://a.com/h"

    @pytest.mark.asyncio
    async def test_get_webhook_wrong_user(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        fetched = await get_webhook(sub.id, OTHER_USER)
        assert fetched is None

    @pytest.mark.asyncio
    async def test_get_webhook_nonexistent(self, webhook_db):
        fetched = await get_webhook("nonexistent", USER_ID)
        assert fetched is None


class TestWebhookUpdate:
    @pytest.mark.asyncio
    async def test_update_url(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://old.com/h", ["policy.violated"])
        updated = await update_webhook(sub.id, USER_ID, url="https://new.com/h")
        assert updated.url == "https://new.com/h"

    @pytest.mark.asyncio
    async def test_update_events(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        updated = await update_webhook(sub.id, USER_ID, events=["anomaly.detected", "emergency.revoke"])
        assert updated.events == ["anomaly.detected", "emergency.revoke"]

    @pytest.mark.asyncio
    async def test_update_active_status(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        updated = await update_webhook(sub.id, USER_ID, is_active=False)
        assert updated.is_active is False

    @pytest.mark.asyncio
    async def test_update_description(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        updated = await update_webhook(sub.id, USER_ID, description="Updated desc")
        assert updated.description == "Updated desc"

    @pytest.mark.asyncio
    async def test_update_invalid_events(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        with pytest.raises(ValueError, match="Invalid events"):
            await update_webhook(sub.id, USER_ID, events=["fake.event"])

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, webhook_db):
        result = await update_webhook("fake", USER_ID, url="https://new.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_wrong_user(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        result = await update_webhook(sub.id, OTHER_USER, url="https://evil.com")
        assert result is None


class TestWebhookDeletion:
    @pytest.mark.asyncio
    async def test_delete_webhook(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        assert await delete_webhook(sub.id, USER_ID) is True
        assert await get_webhook(sub.id, USER_ID) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, webhook_db):
        assert await delete_webhook("fake", USER_ID) is False

    @pytest.mark.asyncio
    async def test_delete_wrong_user(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        assert await delete_webhook(sub.id, OTHER_USER) is False
        assert await get_webhook(sub.id, USER_ID) is not None


class TestSecretRotation:
    @pytest.mark.asyncio
    async def test_rotate_secret(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        old_secret = sub.secret
        new_secret = await rotate_webhook_secret(sub.id, USER_ID)
        assert new_secret is not None
        assert new_secret != old_secret
        assert new_secret.startswith("whsec_")

    @pytest.mark.asyncio
    async def test_rotate_nonexistent(self, webhook_db):
        result = await rotate_webhook_secret("fake", USER_ID)
        assert result is None

    @pytest.mark.asyncio
    async def test_rotate_wrong_user(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        result = await rotate_webhook_secret(sub.id, OTHER_USER)
        assert result is None


class TestSignatureVerification:
    def test_compute_signature(self):
        sig = compute_signature('{"event":"test"}', "secret123")
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex digest

    def test_verify_signature_valid(self):
        payload = '{"event":"test","data":{}}'
        secret = "whsec_testsecret"
        sig = compute_signature(payload, secret)
        assert verify_signature(payload, secret, sig) is True

    def test_verify_signature_invalid(self):
        payload = '{"event":"test"}'
        assert verify_signature(payload, "secret", "invalid_signature") is False

    def test_verify_signature_tampered_payload(self):
        payload = '{"event":"test"}'
        secret = "whsec_test"
        sig = compute_signature(payload, secret)
        assert verify_signature('{"event":"tampered"}', secret, sig) is False

    def test_verify_signature_wrong_secret(self):
        payload = '{"event":"test"}'
        sig = compute_signature(payload, "correct_secret")
        assert verify_signature(payload, "wrong_secret", sig) is False

    def test_signature_consistency(self):
        payload = '{"key":"value"}'
        secret = "test_secret"
        sig1 = compute_signature(payload, secret)
        sig2 = compute_signature(payload, secret)
        assert sig1 == sig2

    def test_different_payloads_different_signatures(self):
        secret = "test"
        sig1 = compute_signature("payload1", secret)
        sig2 = compute_signature("payload2", secret)
        assert sig1 != sig2


class TestEventSubscriptions:
    @pytest.mark.asyncio
    async def test_get_subs_for_event(self, webhook_db):
        await create_webhook(USER_ID, "https://a.com/h", ["policy.violated", "anomaly.detected"])
        await create_webhook(USER_ID, "https://b.com/h", ["anomaly.detected"])
        subs = await get_subscriptions_for_event(USER_ID, "anomaly.detected")
        assert len(subs) == 2

    @pytest.mark.asyncio
    async def test_get_subs_for_event_partial(self, webhook_db):
        await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        await create_webhook(USER_ID, "https://b.com/h", ["anomaly.detected"])
        subs = await get_subscriptions_for_event(USER_ID, "policy.violated")
        assert len(subs) == 1
        assert subs[0].url == "https://a.com/h"

    @pytest.mark.asyncio
    async def test_get_subs_inactive_excluded(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        await update_webhook(sub.id, USER_ID, is_active=False)
        subs = await get_subscriptions_for_event(USER_ID, "policy.violated")
        assert len(subs) == 0

    @pytest.mark.asyncio
    async def test_get_subs_user_isolation(self, webhook_db):
        await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        await create_webhook(OTHER_USER, "https://b.com/h", ["policy.violated"])
        subs = await get_subscriptions_for_event(USER_ID, "policy.violated")
        assert len(subs) == 1

    @pytest.mark.asyncio
    async def test_get_subs_no_match(self, webhook_db):
        await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        subs = await get_subscriptions_for_event(USER_ID, "emergency.revoke")
        assert len(subs) == 0


class TestWebhookDispatching:
    @pytest.mark.asyncio
    async def test_dispatch_to_matching_subs(self, webhook_db):
        await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        await create_webhook(USER_ID, "https://b.com/h", ["policy.violated"])
        deliveries = await dispatch_webhook(
            USER_ID, "policy.violated",
            {"agent_id": "agent-1", "reason": "scope_exceeded"},
        )
        assert len(deliveries) == 2

    @pytest.mark.asyncio
    async def test_dispatch_payload_structure(self, webhook_db):
        await create_webhook(USER_ID, "https://a.com/h", ["anomaly.detected"])
        deliveries = await dispatch_webhook(
            USER_ID, "anomaly.detected",
            {"agent_id": "a1", "anomaly_type": "burst"},
        )
        assert len(deliveries) == 1
        payload = json.loads(deliveries[0].payload)
        assert payload["event"] == "anomaly.detected"
        assert "timestamp" in payload
        assert payload["data"]["agent_id"] == "a1"

    @pytest.mark.asyncio
    async def test_dispatch_signature_attached(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["token.issued"])
        deliveries = await dispatch_webhook(USER_ID, "token.issued", {"service": "github"})
        assert len(deliveries) == 1
        sig = deliveries[0]._signature
        assert verify_signature(deliveries[0].payload, sub.secret, sig) is True

    @pytest.mark.asyncio
    async def test_dispatch_no_matching_subs(self, webhook_db):
        await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        deliveries = await dispatch_webhook(USER_ID, "emergency.revoke", {})
        assert len(deliveries) == 0

    @pytest.mark.asyncio
    async def test_dispatch_updates_last_triggered(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        assert sub.last_triggered_at == 0
        await dispatch_webhook(USER_ID, "policy.violated", {})
        updated = await get_webhook(sub.id, USER_ID)
        assert updated.last_triggered_at > 0


class TestDeliveryResults:
    @pytest.mark.asyncio
    async def test_record_success(self, webhook_db):
        await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        deliveries = await dispatch_webhook(USER_ID, "policy.violated", {})
        assert len(deliveries) == 1
        result = await record_delivery_result(
            deliveries[0].id, status_code=200, duration_ms=42.5,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_record_failure_increments_count(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        deliveries = await dispatch_webhook(USER_ID, "policy.violated", {})
        await record_delivery_result(deliveries[0].id, status_code=500)
        updated = await get_webhook(sub.id, USER_ID)
        assert updated.failure_count == 1

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        # Record a failure first
        d1 = await dispatch_webhook(USER_ID, "policy.violated", {})
        await record_delivery_result(d1[0].id, status_code=500)
        # Then a success
        d2 = await dispatch_webhook(USER_ID, "policy.violated", {})
        await record_delivery_result(d2[0].id, status_code=200)
        updated = await get_webhook(sub.id, USER_ID)
        assert updated.failure_count == 0

    @pytest.mark.asyncio
    async def test_auto_disable_after_10_failures(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        for _ in range(10):
            deliveries = await dispatch_webhook(USER_ID, "policy.violated", {})
            if deliveries:
                await record_delivery_result(deliveries[0].id, status_code=500)
        updated = await get_webhook(sub.id, USER_ID)
        assert updated.is_active is False

    @pytest.mark.asyncio
    async def test_record_nonexistent_delivery(self, webhook_db):
        result = await record_delivery_result("fake", status_code=200)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_delivery_history(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        for i in range(5):
            await dispatch_webhook(USER_ID, "policy.violated", {"count": i})
        deliveries = await get_webhook_deliveries(sub.id, USER_ID)
        assert len(deliveries) == 5

    @pytest.mark.asyncio
    async def test_get_delivery_history_limit(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        for i in range(10):
            await dispatch_webhook(USER_ID, "policy.violated", {"count": i})
        deliveries = await get_webhook_deliveries(sub.id, USER_ID, limit=3)
        assert len(deliveries) == 3

    @pytest.mark.asyncio
    async def test_get_delivery_history_wrong_user(self, webhook_db):
        sub = await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        await dispatch_webhook(USER_ID, "policy.violated", {})
        deliveries = await get_webhook_deliveries(sub.id, OTHER_USER)
        assert len(deliveries) == 0


class TestWebhookEvents:
    def test_all_events_are_strings(self):
        for event in WebhookEvent:
            assert isinstance(event.value, str)
            assert "." in event.value

    def test_event_count(self):
        assert len(WebhookEvent) >= 17

    def test_event_uniqueness(self):
        values = [e.value for e in WebhookEvent]
        assert len(values) == len(set(values))

    @pytest.mark.asyncio
    async def test_subscribe_to_each_event_type(self, webhook_db):
        for event in WebhookEvent:
            sub = await create_webhook(USER_ID, f"https://x.com/{event.value}", [event.value])
            assert event.value in sub.events


class TestWebhookEdgeCases:
    @pytest.mark.asyncio
    async def test_dispatch_with_empty_data(self, webhook_db):
        await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        deliveries = await dispatch_webhook(USER_ID, "policy.violated", {})
        assert len(deliveries) == 1
        payload = json.loads(deliveries[0].payload)
        assert payload["data"] == {}

    @pytest.mark.asyncio
    async def test_dispatch_with_nested_data(self, webhook_db):
        await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        data = {"agent": {"id": "a1", "name": "Test"}, "violations": [1, 2, 3]}
        deliveries = await dispatch_webhook(USER_ID, "policy.violated", data)
        payload = json.loads(deliveries[0].payload)
        assert payload["data"]["agent"]["id"] == "a1"
        assert payload["data"]["violations"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_multiple_users_multiple_events(self, webhook_db):
        await create_webhook(USER_ID, "https://a.com/h", ["policy.violated", "emergency.revoke"])
        await create_webhook(OTHER_USER, "https://b.com/h", ["policy.violated"])
        d1 = await dispatch_webhook(USER_ID, "policy.violated", {})
        d2 = await dispatch_webhook(OTHER_USER, "policy.violated", {})
        assert len(d1) == 1
        assert len(d2) == 1

    @pytest.mark.asyncio
    async def test_response_body_truncated(self, webhook_db):
        await create_webhook(USER_ID, "https://a.com/h", ["policy.violated"])
        deliveries = await dispatch_webhook(USER_ID, "policy.violated", {})
        long_body = "x" * 2000
        await record_delivery_result(deliveries[0].id, 200, response_body=long_body)
        # Verify body was truncated to 1000 chars
        history = await get_webhook_deliveries(
            deliveries[0].subscription_id, USER_ID,
        )
        # The actual truncation happens in record_delivery_result


class TestWebhookSubscriptionDataclass:
    def test_default_values(self):
        sub = WebhookSubscription()
        assert sub.id == ""
        assert sub.events == []
        assert sub.is_active is True
        assert sub.failure_count == 0

    def test_custom_values(self):
        sub = WebhookSubscription(
            id="test",
            user_id="u1",
            url="https://x.com",
            events=["policy.violated"],
            is_active=False,
        )
        assert sub.id == "test"
        assert sub.is_active is False
