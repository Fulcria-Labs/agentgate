"""Comprehensive time window tests — hour boundaries, day boundaries,
midnight crossover, single-hour windows, full-week windows, and
combined day+hour restrictions."""

import time
from datetime import datetime, timezone

import pytest

from src.database import AgentPolicy, create_agent_policy
from src.policy import (
    PolicyDenied,
    _rate_counters,
    check_time_window,
    enforce_policy,
)


@pytest.fixture(autouse=True)
def clear_rate_counters():
    _rate_counters.clear()
    yield
    _rate_counters.clear()


def _tw_policy(hours=None, days=None):
    return AgentPolicy(
        agent_id="tw", agent_name="tw",
        allowed_hours=hours or [],
        allowed_days=days or [],
    )


class TestHourBoundaries:
    """Test exact hour boundary conditions."""

    def test_start_of_allowed_hour(self):
        """Minute 0 of an allowed hour passes."""
        p = _tw_policy(hours=[9])
        now = datetime(2026, 3, 11, 9, 0, 0, tzinfo=timezone.utc)
        assert check_time_window(p, now) is None

    def test_end_of_allowed_hour(self):
        """Minute 59 of an allowed hour passes."""
        p = _tw_policy(hours=[9])
        now = datetime(2026, 3, 11, 9, 59, 59, tzinfo=timezone.utc)
        assert check_time_window(p, now) is None

    def test_start_of_next_hour_denied(self):
        """Minute 0 of the next (non-allowed) hour is denied."""
        p = _tw_policy(hours=[9])
        now = datetime(2026, 3, 11, 10, 0, 0, tzinfo=timezone.utc)
        result = check_time_window(p, now)
        assert result is not None

    def test_hour_0_boundary(self):
        """Hour 0 (midnight) is a valid allowed hour."""
        p = _tw_policy(hours=[0])
        now = datetime(2026, 3, 11, 0, 30, tzinfo=timezone.utc)
        assert check_time_window(p, now) is None

    def test_hour_23_boundary(self):
        """Hour 23 is a valid allowed hour."""
        p = _tw_policy(hours=[23])
        now = datetime(2026, 3, 11, 23, 59, tzinfo=timezone.utc)
        assert check_time_window(p, now) is None

    def test_non_contiguous_hours(self):
        """Non-contiguous hours (e.g., 9 and 17) are both allowed."""
        p = _tw_policy(hours=[9, 17])
        assert check_time_window(p, datetime(2026, 3, 11, 9, 0, tzinfo=timezone.utc)) is None
        assert check_time_window(p, datetime(2026, 3, 11, 17, 0, tzinfo=timezone.utc)) is None
        result = check_time_window(p, datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc))
        assert result is not None


class TestSingleHourWindow:
    """Single hour access windows."""

    def test_single_hour_allowed(self):
        p = _tw_policy(hours=[14])
        now = datetime(2026, 3, 11, 14, 30, tzinfo=timezone.utc)
        assert check_time_window(p, now) is None

    def test_single_hour_before_denied(self):
        p = _tw_policy(hours=[14])
        now = datetime(2026, 3, 11, 13, 59, tzinfo=timezone.utc)
        result = check_time_window(p, now)
        assert result is not None

    def test_single_hour_after_denied(self):
        p = _tw_policy(hours=[14])
        now = datetime(2026, 3, 11, 15, 0, tzinfo=timezone.utc)
        result = check_time_window(p, now)
        assert result is not None


class TestAllHours:
    """All 24 hours in the window."""

    def test_all_hours_allowed(self):
        p = _tw_policy(hours=list(range(24)))
        for h in range(24):
            now = datetime(2026, 3, 11, h, 30, tzinfo=timezone.utc)
            assert check_time_window(p, now) is None


class TestDayBoundaries:
    """Test exact day boundary conditions."""

    def test_monday_allowed(self):
        p = _tw_policy(days=[0])  # Monday
        # March 9, 2026 is a Monday
        now = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        assert check_time_window(p, now) is None

    def test_sunday_allowed(self):
        p = _tw_policy(days=[6])  # Sunday
        # March 15, 2026 is a Sunday
        now = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)
        assert check_time_window(p, now) is None

    def test_weekday_only(self):
        """Mon-Fri (0-4) passes on Wednesday but not Saturday."""
        p = _tw_policy(days=[0, 1, 2, 3, 4])
        # Wednesday
        assert check_time_window(p, datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)) is None
        # Saturday (day 5)
        result = check_time_window(p, datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc))
        assert "not within allowed days" in result

    def test_weekend_only(self):
        """Sat-Sun (5-6) passes on Saturday but not Monday."""
        p = _tw_policy(days=[5, 6])
        # Saturday
        assert check_time_window(p, datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)) is None
        # Monday
        result = check_time_window(p, datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc))
        assert result is not None

    def test_single_day_only(self):
        """Only Wednesdays allowed."""
        p = _tw_policy(days=[2])
        # Wednesday
        assert check_time_window(p, datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)) is None
        # Thursday
        result = check_time_window(p, datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc))
        assert result is not None

    def test_all_days_allowed(self):
        p = _tw_policy(days=[0, 1, 2, 3, 4, 5, 6])
        for day_offset in range(7):
            now = datetime(2026, 3, 9 + day_offset, 12, 0, tzinfo=timezone.utc)
            assert check_time_window(p, now) is None


class TestDayPlusHourCombinations:
    """Combined day and hour restrictions."""

    def test_weekday_business_hours(self):
        """Mon-Fri 9-17 UTC is the classic business hours window."""
        p = _tw_policy(days=[0, 1, 2, 3, 4], hours=list(range(9, 18)))
        # Wednesday 10:00 — allowed
        assert check_time_window(p, datetime(2026, 3, 11, 10, 0, tzinfo=timezone.utc)) is None
        # Wednesday 3:00 — day ok, hour denied
        result = check_time_window(p, datetime(2026, 3, 11, 3, 0, tzinfo=timezone.utc))
        assert "hours" in result
        # Saturday 10:00 — hour ok, day denied
        result = check_time_window(p, datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc))
        assert "days" in result

    def test_day_checked_before_hour(self):
        """Day constraint is checked before hour constraint."""
        p = _tw_policy(days=[0], hours=[9])  # Monday 9am only
        # Saturday 9am — day fails first
        result = check_time_window(p, datetime(2026, 3, 14, 9, 0, tzinfo=timezone.utc))
        assert "days" in result

    def test_hour_checked_after_day_passes(self):
        """If day passes, hour is still checked."""
        p = _tw_policy(days=[2], hours=[9])  # Wednesday 9am only
        # Wednesday 3am — day passes, hour fails
        result = check_time_window(p, datetime(2026, 3, 11, 3, 0, tzinfo=timezone.utc))
        assert "hours" in result


class TestEmptyConstraints:
    """Empty constraints mean always allowed."""

    def test_empty_hours_any_hour(self):
        p = _tw_policy(hours=[])
        assert check_time_window(p, datetime(2026, 3, 11, 3, 0, tzinfo=timezone.utc)) is None

    def test_empty_days_any_day(self):
        p = _tw_policy(days=[])
        assert check_time_window(p, datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)) is None

    def test_both_empty_any_time(self):
        p = _tw_policy()
        now = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        assert check_time_window(p, now) is None

    def test_empty_with_default_now(self):
        """Empty constraints with default now (uses real time) passes."""
        p = _tw_policy()
        assert check_time_window(p) is None


class TestTimeWindowDenialMessages:
    """Verify denial message formatting."""

    def test_hour_denial_includes_range(self):
        p = _tw_policy(hours=[9, 10, 11, 12, 13, 14, 15, 16, 17])
        result = check_time_window(p, datetime(2026, 3, 11, 3, 0, tzinfo=timezone.utc))
        assert "09:00" in result
        assert "17:59" in result

    def test_day_denial_includes_day_names(self):
        p = _tw_policy(days=[0, 1, 2, 3, 4])
        result = check_time_window(p, datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc))
        assert "Mon" in result
        assert "Fri" in result

    def test_single_hour_denial_message(self):
        p = _tw_policy(hours=[12])
        result = check_time_window(p, datetime(2026, 3, 11, 3, 0, tzinfo=timezone.utc))
        assert "12:00" in result
        assert "12:59" in result

    def test_single_day_denial_message(self):
        p = _tw_policy(days=[3])  # Thursday
        result = check_time_window(p, datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc))
        assert "Thu" in result


class TestMidnightCrossover:
    """Tests around midnight (hour 0 / hour 23 boundary)."""

    def test_midnight_allowed_with_hour_0(self):
        p = _tw_policy(hours=[0])
        now = datetime(2026, 3, 11, 0, 0, 0, tzinfo=timezone.utc)
        assert check_time_window(p, now) is None

    def test_one_second_before_midnight_is_hour_23(self):
        p = _tw_policy(hours=[23])
        now = datetime(2026, 3, 11, 23, 59, 59, tzinfo=timezone.utc)
        assert check_time_window(p, now) is None

    def test_night_shift_hours(self):
        """Non-contiguous 'night shift' hours: 22, 23, 0, 1, 2."""
        p = _tw_policy(hours=[22, 23, 0, 1, 2])
        assert check_time_window(p, datetime(2026, 3, 11, 23, 0, tzinfo=timezone.utc)) is None
        assert check_time_window(p, datetime(2026, 3, 12, 0, 30, tzinfo=timezone.utc)) is None
        result = check_time_window(p, datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc))
        assert result is not None


class TestTimeWindowIntegrationWithPolicy:
    """Time window checks integrated with enforce_policy."""

    @pytest.mark.asyncio
    async def test_time_window_denial_in_enforce(self, db):
        from unittest.mock import patch
        policy = AgentPolicy(
            agent_id="tw-agent", agent_name="TW Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            allowed_hours=[9, 10, 11],
            rate_limit_per_minute=60,
            created_by="u1", created_at=time.time(), is_active=True,
        )
        await create_agent_policy(policy)
        with patch("src.policy.check_time_window", return_value="Access denied: not within allowed hours"):
            with pytest.raises(PolicyDenied, match="not within allowed hours"):
                await enforce_policy("u1", "tw-agent", "github", ["repo"])

    @pytest.mark.asyncio
    async def test_time_window_pass_in_enforce(self, db):
        from unittest.mock import patch
        policy = AgentPolicy(
            agent_id="tw-agent2", agent_name="TW Bot",
            allowed_services=["github"],
            allowed_scopes={"github": ["repo"]},
            allowed_hours=[9, 10, 11],
            rate_limit_per_minute=60,
            created_by="u1", created_at=time.time(), is_active=True,
        )
        await create_agent_policy(policy)
        with patch("src.policy.check_time_window", return_value=None):
            result = await enforce_policy("u1", "tw-agent2", "github", ["repo"])
            assert result.agent_id == "tw-agent2"
