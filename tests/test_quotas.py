"""Comprehensive tests for token usage quotas system."""

import time
import pytest
import pytest_asyncio

from src.database import init_db
from src.quotas import (
    QuotaStatus,
    UsageQuota,
    _get_period_bounds,
    check_quota,
    create_quota,
    delete_quota,
    get_quota_usage_history,
    get_quotas,
    init_quota_tables,
    record_quota_usage,
    reset_quota,
)


@pytest_asyncio.fixture
async def quota_db(db, monkeypatch):
    """Initialize quota tables."""
    monkeypatch.setattr("src.quotas.DB_PATH", db)
    await init_quota_tables()
    return db


USER_ID = "auth0|user123"
OTHER_USER = "auth0|other456"
AGENT_ID = "agent-1"


class TestQuotaCreation:
    @pytest.mark.asyncio
    async def test_create_daily_quota(self, quota_db):
        q = await create_quota(USER_ID, AGENT_ID, "daily", 100)
        assert q.id != ""
        assert q.user_id == USER_ID
        assert q.agent_id == AGENT_ID
        assert q.quota_type == "daily"
        assert q.max_tokens == 100
        assert q.current_usage == 0
        assert q.is_active is True

    @pytest.mark.asyncio
    async def test_create_monthly_quota(self, quota_db):
        q = await create_quota(USER_ID, AGENT_ID, "monthly", 1000)
        assert q.quota_type == "monthly"
        assert q.max_tokens == 1000

    @pytest.mark.asyncio
    async def test_create_total_quota(self, quota_db):
        q = await create_quota(USER_ID, AGENT_ID, "total", 5000)
        assert q.quota_type == "total"
        assert q.max_tokens == 5000

    @pytest.mark.asyncio
    async def test_create_with_action_deny(self, quota_db):
        q = await create_quota(USER_ID, AGENT_ID, "daily", 100, action_on_exceed="deny")
        assert q.action_on_exceed == "deny"

    @pytest.mark.asyncio
    async def test_create_with_action_warn(self, quota_db):
        q = await create_quota(USER_ID, AGENT_ID, "daily", 100, action_on_exceed="warn")
        assert q.action_on_exceed == "warn"

    @pytest.mark.asyncio
    async def test_create_with_action_step_up(self, quota_db):
        q = await create_quota(USER_ID, AGENT_ID, "daily", 100, action_on_exceed="step_up")
        assert q.action_on_exceed == "step_up"

    @pytest.mark.asyncio
    async def test_create_with_notify_percent(self, quota_db):
        q = await create_quota(USER_ID, AGENT_ID, "daily", 100, notify_at_percent=90)
        assert q.notify_at_percent == 90

    @pytest.mark.asyncio
    async def test_create_invalid_type(self, quota_db):
        with pytest.raises(ValueError, match="Invalid quota type"):
            await create_quota(USER_ID, AGENT_ID, "weekly", 100)

    @pytest.mark.asyncio
    async def test_create_invalid_action(self, quota_db):
        with pytest.raises(ValueError, match="Invalid action"):
            await create_quota(USER_ID, AGENT_ID, "daily", 100, action_on_exceed="crash")

    @pytest.mark.asyncio
    async def test_create_zero_tokens(self, quota_db):
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            await create_quota(USER_ID, AGENT_ID, "daily", 0)

    @pytest.mark.asyncio
    async def test_create_negative_tokens(self, quota_db):
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            await create_quota(USER_ID, AGENT_ID, "daily", -10)

    @pytest.mark.asyncio
    async def test_create_invalid_notify_percent(self, quota_db):
        with pytest.raises(ValueError, match="notify_at_percent"):
            await create_quota(USER_ID, AGENT_ID, "daily", 100, notify_at_percent=150)

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self, quota_db):
        q1 = await create_quota(USER_ID, AGENT_ID, "daily", 100)
        q2 = await create_quota(USER_ID, AGENT_ID, "daily", 200)
        # Should be one quota, not two
        quotas = await get_quotas(USER_ID, AGENT_ID)
        daily_quotas = [q for q in quotas if q.quota_type == "daily"]
        assert len(daily_quotas) == 1
        assert daily_quotas[0].max_tokens == 200

    @pytest.mark.asyncio
    async def test_different_types_coexist(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 100)
        await create_quota(USER_ID, AGENT_ID, "monthly", 1000)
        quotas = await get_quotas(USER_ID, AGENT_ID)
        assert len(quotas) == 2
        types = {q.quota_type for q in quotas}
        assert types == {"daily", "monthly"}


class TestQuotaChecking:
    @pytest.mark.asyncio
    async def test_check_no_quotas(self, quota_db):
        statuses = await check_quota(USER_ID, AGENT_ID)
        assert statuses == []

    @pytest.mark.asyncio
    async def test_check_unused_quota(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 100)
        statuses = await check_quota(USER_ID, AGENT_ID)
        assert len(statuses) == 1
        s = statuses[0]
        assert s.allowed is True
        assert s.current_usage == 0
        assert s.max_tokens == 100
        assert s.remaining == 100
        assert s.usage_percent == 0
        assert s.threshold_reached is False

    @pytest.mark.asyncio
    async def test_check_after_usage(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 100)
        for _ in range(50):
            await record_quota_usage(USER_ID, AGENT_ID, "github")
        statuses = await check_quota(USER_ID, AGENT_ID)
        s = statuses[0]
        assert s.current_usage == 50
        assert s.remaining == 50
        assert s.usage_percent == 50.0

    @pytest.mark.asyncio
    async def test_check_threshold_reached(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 10, notify_at_percent=80)
        for _ in range(8):
            await record_quota_usage(USER_ID, AGENT_ID)
        statuses = await check_quota(USER_ID, AGENT_ID)
        assert statuses[0].threshold_reached is True

    @pytest.mark.asyncio
    async def test_check_threshold_not_reached(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 10, notify_at_percent=80)
        for _ in range(7):
            await record_quota_usage(USER_ID, AGENT_ID)
        statuses = await check_quota(USER_ID, AGENT_ID)
        assert statuses[0].threshold_reached is False

    @pytest.mark.asyncio
    async def test_check_exceeded_deny(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 5, action_on_exceed="deny")
        for _ in range(5):
            await record_quota_usage(USER_ID, AGENT_ID)
        statuses = await check_quota(USER_ID, AGENT_ID)
        s = statuses[0]
        assert s.allowed is False
        assert s.action == "deny"
        assert s.remaining == 0

    @pytest.mark.asyncio
    async def test_check_exceeded_warn_still_allowed(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 5, action_on_exceed="warn")
        for _ in range(5):
            await record_quota_usage(USER_ID, AGENT_ID)
        statuses = await check_quota(USER_ID, AGENT_ID)
        s = statuses[0]
        assert s.allowed is True  # warn mode still allows
        assert s.action == "warn"

    @pytest.mark.asyncio
    async def test_check_exceeded_step_up(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 5, action_on_exceed="step_up")
        for _ in range(5):
            await record_quota_usage(USER_ID, AGENT_ID)
        statuses = await check_quota(USER_ID, AGENT_ID)
        s = statuses[0]
        assert s.allowed is False
        assert s.action == "step_up"

    @pytest.mark.asyncio
    async def test_check_multiple_quotas(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 10)
        await create_quota(USER_ID, AGENT_ID, "monthly", 100)
        for _ in range(5):
            await record_quota_usage(USER_ID, AGENT_ID)
        statuses = await check_quota(USER_ID, AGENT_ID)
        assert len(statuses) == 2


class TestQuotaUsageRecording:
    @pytest.mark.asyncio
    async def test_record_increments_usage(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 100)
        statuses = await record_quota_usage(USER_ID, AGENT_ID, "github")
        assert statuses[0].current_usage == 1

    @pytest.mark.asyncio
    async def test_record_multiple_increments(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 100)
        for i in range(10):
            statuses = await record_quota_usage(USER_ID, AGENT_ID)
        assert statuses[0].current_usage == 10

    @pytest.mark.asyncio
    async def test_record_with_service(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 100)
        await record_quota_usage(USER_ID, AGENT_ID, "github")
        await record_quota_usage(USER_ID, AGENT_ID, "slack")
        statuses = await check_quota(USER_ID, AGENT_ID)
        assert statuses[0].current_usage == 2

    @pytest.mark.asyncio
    async def test_record_creates_log(self, quota_db):
        q = await create_quota(USER_ID, AGENT_ID, "daily", 100)
        await record_quota_usage(USER_ID, AGENT_ID, "github")
        history = await get_quota_usage_history(q.id)
        assert len(history) == 1
        assert history[0]["agent_id"] == AGENT_ID
        assert history[0]["service"] == "github"

    @pytest.mark.asyncio
    async def test_record_no_quota_no_error(self, quota_db):
        # Should not error even if no quotas exist
        statuses = await record_quota_usage(USER_ID, AGENT_ID)
        assert statuses == []


class TestQuotaListing:
    @pytest.mark.asyncio
    async def test_list_empty(self, quota_db):
        quotas = await get_quotas(USER_ID)
        assert quotas == []

    @pytest.mark.asyncio
    async def test_list_all_for_user(self, quota_db):
        await create_quota(USER_ID, "agent-1", "daily", 100)
        await create_quota(USER_ID, "agent-2", "daily", 200)
        quotas = await get_quotas(USER_ID)
        assert len(quotas) == 2

    @pytest.mark.asyncio
    async def test_list_filtered_by_agent(self, quota_db):
        await create_quota(USER_ID, "agent-1", "daily", 100)
        await create_quota(USER_ID, "agent-2", "daily", 200)
        quotas = await get_quotas(USER_ID, agent_id="agent-1")
        assert len(quotas) == 1
        assert quotas[0].agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_list_user_isolation(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 100)
        await create_quota(OTHER_USER, AGENT_ID, "daily", 200)
        mine = await get_quotas(USER_ID)
        theirs = await get_quotas(OTHER_USER)
        assert len(mine) == 1
        assert len(theirs) == 1


class TestQuotaDeletion:
    @pytest.mark.asyncio
    async def test_delete_quota(self, quota_db):
        q = await create_quota(USER_ID, AGENT_ID, "daily", 100)
        assert await delete_quota(q.id, USER_ID) is True
        quotas = await get_quotas(USER_ID, AGENT_ID)
        assert len(quotas) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, quota_db):
        assert await delete_quota("fake", USER_ID) is False

    @pytest.mark.asyncio
    async def test_delete_wrong_user(self, quota_db):
        q = await create_quota(USER_ID, AGENT_ID, "daily", 100)
        assert await delete_quota(q.id, OTHER_USER) is False
        quotas = await get_quotas(USER_ID, AGENT_ID)
        assert len(quotas) == 1


class TestQuotaReset:
    @pytest.mark.asyncio
    async def test_reset_quota(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 100)
        for _ in range(50):
            await record_quota_usage(USER_ID, AGENT_ID)
        quotas = await get_quotas(USER_ID, AGENT_ID)
        q_id = quotas[0].id
        assert await reset_quota(q_id, USER_ID) is True
        statuses = await check_quota(USER_ID, AGENT_ID)
        assert statuses[0].current_usage == 0

    @pytest.mark.asyncio
    async def test_reset_nonexistent(self, quota_db):
        assert await reset_quota("fake", USER_ID) is False

    @pytest.mark.asyncio
    async def test_reset_wrong_user(self, quota_db):
        q = await create_quota(USER_ID, AGENT_ID, "daily", 100)
        assert await reset_quota(q.id, OTHER_USER) is False


class TestQuotaUsageHistory:
    @pytest.mark.asyncio
    async def test_empty_history(self, quota_db):
        q = await create_quota(USER_ID, AGENT_ID, "daily", 100)
        history = await get_quota_usage_history(q.id)
        assert history == []

    @pytest.mark.asyncio
    async def test_history_records(self, quota_db):
        q = await create_quota(USER_ID, AGENT_ID, "daily", 100)
        await record_quota_usage(USER_ID, AGENT_ID, "github")
        await record_quota_usage(USER_ID, AGENT_ID, "slack")
        await record_quota_usage(USER_ID, AGENT_ID, "google")
        history = await get_quota_usage_history(q.id)
        assert len(history) == 3
        services = {h["service"] for h in history}
        assert services == {"github", "slack", "google"}

    @pytest.mark.asyncio
    async def test_history_limit(self, quota_db):
        q = await create_quota(USER_ID, AGENT_ID, "daily", 1000)
        for _ in range(20):
            await record_quota_usage(USER_ID, AGENT_ID)
        history = await get_quota_usage_history(q.id, limit=5)
        assert len(history) == 5


class TestPeriodBounds:
    def test_daily_period(self):
        now = time.time()
        start, end = _get_period_bounds("daily", now)
        assert end - start == 86400  # Exactly one day

    def test_monthly_period(self):
        now = time.time()
        start, end = _get_period_bounds("monthly", now)
        assert end > start

    def test_total_period(self):
        now = time.time()
        start, end = _get_period_bounds("total", now)
        assert start == 0
        assert end == 0

    def test_unknown_period(self):
        start, end = _get_period_bounds("weekly", time.time())
        assert start == 0
        assert end == 0


class TestQuotaStatusDataclass:
    def test_default_values(self):
        s = QuotaStatus()
        assert s.allowed is True
        assert s.current_usage == 0
        assert s.remaining == 0
        assert s.threshold_reached is False

    def test_to_dict(self):
        s = QuotaStatus(
            allowed=True,
            quota_id="q1",
            quota_type="daily",
            current_usage=50,
            max_tokens=100,
            remaining=50,
            usage_percent=50.0,
        )
        d = s.to_dict()
        assert d["allowed"] is True
        assert d["quota_id"] == "q1"
        assert d["usage_percent"] == 50.0
        assert d["remaining"] == 50

    def test_to_dict_null_reset(self):
        s = QuotaStatus(resets_at=0)
        d = s.to_dict()
        assert d["resets_at"] is None

    def test_to_dict_with_reset(self):
        reset_time = time.time() + 3600
        s = QuotaStatus(resets_at=reset_time)
        d = s.to_dict()
        assert d["resets_at"] == reset_time


class TestUsageQuotaDataclass:
    def test_default_values(self):
        q = UsageQuota()
        assert q.quota_type == "daily"
        assert q.max_tokens == 0
        assert q.is_active is True
        assert q.action_on_exceed == "deny"
        assert q.notify_at_percent == 80


class TestQuotaIsolation:
    @pytest.mark.asyncio
    async def test_different_agents_separate_quotas(self, quota_db):
        await create_quota(USER_ID, "agent-1", "daily", 10)
        await create_quota(USER_ID, "agent-2", "daily", 20)
        for _ in range(5):
            await record_quota_usage(USER_ID, "agent-1")
        s1 = await check_quota(USER_ID, "agent-1")
        s2 = await check_quota(USER_ID, "agent-2")
        assert s1[0].current_usage == 5
        assert s2[0].current_usage == 0

    @pytest.mark.asyncio
    async def test_different_users_separate_quotas(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 10)
        await create_quota(OTHER_USER, AGENT_ID, "daily", 10)
        for _ in range(5):
            await record_quota_usage(USER_ID, AGENT_ID)
        s1 = await check_quota(USER_ID, AGENT_ID)
        s2 = await check_quota(OTHER_USER, AGENT_ID)
        assert s1[0].current_usage == 5
        assert s2[0].current_usage == 0


class TestQuotaEdgeCases:
    @pytest.mark.asyncio
    async def test_exceed_then_reset_then_use(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 5)
        for _ in range(5):
            await record_quota_usage(USER_ID, AGENT_ID)
        statuses = await check_quota(USER_ID, AGENT_ID)
        assert statuses[0].allowed is False

        quotas = await get_quotas(USER_ID, AGENT_ID)
        await reset_quota(quotas[0].id, USER_ID)
        statuses = await check_quota(USER_ID, AGENT_ID)
        assert statuses[0].allowed is True
        assert statuses[0].current_usage == 0

    @pytest.mark.asyncio
    async def test_max_one_token(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 1)
        statuses = await check_quota(USER_ID, AGENT_ID)
        assert statuses[0].allowed is True
        await record_quota_usage(USER_ID, AGENT_ID)
        statuses = await check_quota(USER_ID, AGENT_ID)
        assert statuses[0].allowed is False

    @pytest.mark.asyncio
    async def test_large_quota(self, quota_db):
        await create_quota(USER_ID, AGENT_ID, "daily", 1_000_000)
        statuses = await check_quota(USER_ID, AGENT_ID)
        assert statuses[0].max_tokens == 1_000_000
        assert statuses[0].remaining == 1_000_000
