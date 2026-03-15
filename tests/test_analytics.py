"""Tests for the analytics and anomaly detection module."""

import time
from datetime import datetime, timezone

import pytest

from src.analytics import (
    AgentUsageSummary,
    AnomalyAlert,
    compute_agent_summary,
    compute_risk_score,
    detect_anomalies,
)
from src.database import AuditEntry


# --- Helpers ---

def make_entry(
    agent_id="agent-1",
    service="github",
    action="token_request",
    status="success",
    timestamp=None,
    ip_address="10.0.0.1",
    scopes="repo",
    details="",
):
    return AuditEntry(
        id=0,
        timestamp=timestamp or time.time(),
        user_id="user-1",
        agent_id=agent_id,
        service=service,
        scopes=scopes,
        action=action,
        status=status,
        ip_address=ip_address,
        details=details,
    )


def make_entries_at_time(count, base_time, interval=1.0, **kwargs):
    """Create entries spaced by interval seconds starting at base_time."""
    return [
        make_entry(timestamp=base_time + i * interval, **kwargs)
        for i in range(count)
    ]


# ============================================================
# compute_agent_summary tests
# ============================================================

class TestComputeAgentSummary:
    def test_empty_entries(self):
        s = compute_agent_summary("agent-1", [])
        assert s.agent_id == "agent-1"
        assert s.total_requests == 0
        assert s.successful_requests == 0
        assert s.services_accessed == []
        assert s.unique_ips == []

    def test_single_success(self):
        e = make_entry(status="success", service="github")
        s = compute_agent_summary("agent-1", [e])
        assert s.total_requests == 1
        assert s.successful_requests == 1
        assert s.denied_requests == 0
        assert "github" in s.services_accessed

    def test_counts_by_status(self):
        entries = [
            make_entry(status="success"),
            make_entry(status="success"),
            make_entry(status="denied"),
            make_entry(status="rate_limited"),
        ]
        s = compute_agent_summary("a1", entries)
        assert s.total_requests == 4
        assert s.successful_requests == 2
        assert s.denied_requests == 1
        assert s.rate_limited_requests == 1

    def test_unique_services(self):
        entries = [
            make_entry(service="github"),
            make_entry(service="slack"),
            make_entry(service="github"),
        ]
        s = compute_agent_summary("a1", entries)
        assert s.services_accessed == ["github", "slack"]

    def test_unique_ips(self):
        entries = [
            make_entry(ip_address="10.0.0.1"),
            make_entry(ip_address="10.0.0.2"),
            make_entry(ip_address="10.0.0.1"),
        ]
        s = compute_agent_summary("a1", entries)
        assert len(s.unique_ips) == 2

    def test_first_and_last_seen(self):
        entries = [
            make_entry(timestamp=1000.0),
            make_entry(timestamp=2000.0),
            make_entry(timestamp=3000.0),
        ]
        s = compute_agent_summary("a1", entries)
        assert s.first_seen == 1000.0
        assert s.last_seen == 3000.0

    def test_requests_per_hour(self):
        # 10 requests over 2 hours = 5/hr
        base = time.time() - 7200
        entries = make_entries_at_time(10, base, interval=720)
        s = compute_agent_summary("a1", entries)
        assert 4.0 <= s.requests_per_hour <= 6.0

    def test_peak_hour(self):
        # All requests at hour 14 UTC
        dt = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone.utc)
        ts = dt.timestamp()
        entries = make_entries_at_time(5, ts, interval=60)
        s = compute_agent_summary("a1", entries)
        assert s.peak_hour == 14

    def test_empty_ip_not_tracked(self):
        entries = [make_entry(ip_address=""), make_entry(ip_address="10.0.0.1")]
        s = compute_agent_summary("a1", entries)
        assert s.unique_ips == ["10.0.0.1"]

    def test_empty_service_not_tracked(self):
        entries = [make_entry(service=""), make_entry(service="github")]
        s = compute_agent_summary("a1", entries)
        assert s.services_accessed == ["github"]

    def test_unknown_status_counted_as_total(self):
        entries = [make_entry(status="error")]
        s = compute_agent_summary("a1", entries)
        assert s.total_requests == 1
        assert s.successful_requests == 0
        assert s.denied_requests == 0

    def test_many_services(self):
        services = ["github", "slack", "google", "jira", "confluence"]
        entries = [make_entry(service=svc) for svc in services]
        s = compute_agent_summary("a1", entries)
        assert len(s.services_accessed) == 5

    def test_requests_per_hour_minimum_one_hour(self):
        # All requests at same timestamp - should use 1 hour minimum
        ts = time.time()
        entries = [make_entry(timestamp=ts) for _ in range(5)]
        s = compute_agent_summary("a1", entries)
        assert s.requests_per_hour == 5.0


# ============================================================
# detect_anomalies tests
# ============================================================

class TestDetectAnomalies:
    def test_no_entries_no_anomalies(self):
        alerts = detect_anomalies("a1", [])
        assert alerts == []

    def test_burst_detection(self):
        # 25 requests in 30 seconds
        base = time.time()
        entries = make_entries_at_time(25, base, interval=1.0)
        alerts = detect_anomalies("a1", entries, burst_threshold=20, burst_window_seconds=60)
        burst_alerts = [a for a in alerts if a.alert_type == "burst"]
        assert len(burst_alerts) == 1
        assert burst_alerts[0].severity == "high"
        assert "25" in burst_alerts[0].description

    def test_no_burst_below_threshold(self):
        base = time.time()
        entries = make_entries_at_time(10, base, interval=1.0)
        alerts = detect_anomalies("a1", entries, burst_threshold=20)
        burst_alerts = [a for a in alerts if a.alert_type == "burst"]
        assert len(burst_alerts) == 0

    def test_off_hours_detection(self):
        # Requests at 3 AM UTC
        dt = datetime(2026, 3, 15, 3, 0, 0, tzinfo=timezone.utc)
        ts = dt.timestamp()
        entries = make_entries_at_time(5, ts, interval=60)
        alerts = detect_anomalies("a1", entries, off_hours=(0, 6))
        off_alerts = [a for a in alerts if a.alert_type == "off_hours"]
        assert len(off_alerts) == 1
        assert off_alerts[0].severity == "medium"

    def test_no_off_hours_during_business(self):
        # Requests at 10 AM UTC
        dt = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts = dt.timestamp()
        entries = make_entries_at_time(5, ts, interval=60)
        alerts = detect_anomalies("a1", entries, off_hours=(0, 6))
        off_alerts = [a for a in alerts if a.alert_type == "off_hours"]
        assert len(off_alerts) == 0

    def test_high_denial_rate_critical(self):
        entries = [make_entry(status="denied") for _ in range(8)] + [
            make_entry(status="success") for _ in range(2)
        ]
        alerts = detect_anomalies("a1", entries, denial_rate_threshold=0.3)
        denial_alerts = [a for a in alerts if a.alert_type == "high_denial_rate"]
        assert len(denial_alerts) == 1
        assert denial_alerts[0].severity == "critical"  # 80% denial

    def test_high_denial_rate_high(self):
        # 50% denial rate
        entries = [make_entry(status="denied") for _ in range(5)] + [
            make_entry(status="success") for _ in range(5)
        ]
        alerts = detect_anomalies("a1", entries, denial_rate_threshold=0.3)
        denial_alerts = [a for a in alerts if a.alert_type == "high_denial_rate"]
        assert len(denial_alerts) == 1
        assert denial_alerts[0].severity == "high"

    def test_high_denial_rate_medium(self):
        # 35% denial rate
        entries = [make_entry(status="denied") for _ in range(7)] + [
            make_entry(status="success") for _ in range(13)
        ]
        alerts = detect_anomalies("a1", entries, denial_rate_threshold=0.3)
        denial_alerts = [a for a in alerts if a.alert_type == "high_denial_rate"]
        assert len(denial_alerts) == 1
        assert denial_alerts[0].severity == "medium"

    def test_no_denial_alert_below_threshold(self):
        entries = [make_entry(status="denied")] + [
            make_entry(status="success") for _ in range(9)
        ]
        alerts = detect_anomalies("a1", entries, denial_rate_threshold=0.3)
        denial_alerts = [a for a in alerts if a.alert_type == "high_denial_rate"]
        assert len(denial_alerts) == 0

    def test_no_denial_alert_too_few_entries(self):
        entries = [make_entry(status="denied") for _ in range(3)]
        alerts = detect_anomalies("a1", entries, denial_rate_threshold=0.3)
        denial_alerts = [a for a in alerts if a.alert_type == "high_denial_rate"]
        assert len(denial_alerts) == 0  # < 5 entries

    def test_new_ip_detection(self):
        base = time.time()
        entries = (
            make_entries_at_time(6, base, interval=1.0, ip_address="10.0.0.1")
            + make_entries_at_time(6, base + 10, interval=1.0, ip_address="10.0.0.99")
        )
        alerts = detect_anomalies("a1", entries)
        ip_alerts = [a for a in alerts if a.alert_type == "new_ip"]
        assert len(ip_alerts) == 1
        assert "10.0.0.99" in ip_alerts[0].description

    def test_no_new_ip_when_consistent(self):
        base = time.time()
        entries = make_entries_at_time(12, base, interval=1.0, ip_address="10.0.0.1")
        alerts = detect_anomalies("a1", entries)
        ip_alerts = [a for a in alerts if a.alert_type == "new_ip"]
        assert len(ip_alerts) == 0

    def test_scope_escalation_detection(self):
        entries = [
            make_entry(status="denied", details="Excess scopes: {'admin'}"),
            make_entry(status="denied", details="Scopes not permitted: admin"),
            make_entry(status="success"),
            make_entry(status="success"),
            make_entry(status="success"),
        ]
        alerts = detect_anomalies("a1", entries)
        esc_alerts = [a for a in alerts if a.alert_type == "scope_escalation"]
        assert len(esc_alerts) == 1
        # 2 attempts = medium severity (graduated: 2-4=medium, 5-9=high, 10+=critical)
        assert esc_alerts[0].severity == "medium"

    def test_no_scope_escalation_without_scope_in_details(self):
        entries = [
            make_entry(status="denied", details="Service not allowed"),
        ] + [make_entry(status="success") for _ in range(9)]
        alerts = detect_anomalies("a1", entries)
        esc_alerts = [a for a in alerts if a.alert_type == "scope_escalation"]
        assert len(esc_alerts) == 0

    def test_multiple_anomaly_types(self):
        # Burst + high denial + off-hours
        dt = datetime(2026, 3, 15, 2, 0, 0, tzinfo=timezone.utc)
        ts = dt.timestamp()
        entries = (
            make_entries_at_time(25, ts, interval=1.0, status="denied")
        )
        alerts = detect_anomalies("a1", entries, burst_threshold=20, off_hours=(0, 6))
        alert_types = {a.alert_type for a in alerts}
        assert "burst" in alert_types
        assert "off_hours" in alert_types
        assert "high_denial_rate" in alert_types

    def test_custom_burst_window(self):
        base = time.time()
        # 10 requests in 5 seconds
        entries = make_entries_at_time(10, base, interval=0.5)
        alerts = detect_anomalies(
            "a1", entries, burst_threshold=8, burst_window_seconds=5
        )
        burst_alerts = [a for a in alerts if a.alert_type == "burst"]
        assert len(burst_alerts) == 1

    def test_custom_off_hours_range(self):
        dt = datetime(2026, 3, 15, 20, 0, 0, tzinfo=timezone.utc)
        ts = dt.timestamp()
        entries = make_entries_at_time(5, ts, interval=60)
        alerts = detect_anomalies("a1", entries, off_hours=(18, 23))
        off_alerts = [a for a in alerts if a.alert_type == "off_hours"]
        assert len(off_alerts) == 1

    def test_burst_only_reports_first(self):
        base = time.time()
        # Multiple bursts at different times
        entries = (
            make_entries_at_time(25, base, interval=1.0)
            + make_entries_at_time(25, base + 300, interval=1.0)
        )
        alerts = detect_anomalies("a1", entries, burst_threshold=20)
        burst_alerts = [a for a in alerts if a.alert_type == "burst"]
        assert len(burst_alerts) == 1


# ============================================================
# compute_risk_score tests
# ============================================================

class TestComputeRiskScore:
    def test_zero_risk_clean_agent(self):
        s = AgentUsageSummary(
            agent_id="a1",
            total_requests=100,
            successful_requests=100,
            denied_requests=0,
        )
        score = compute_risk_score(s, [])
        assert score == 0.0
        assert s.risk_factors == []

    def test_denial_rate_factor(self):
        s = AgentUsageSummary(
            agent_id="a1",
            total_requests=100,
            successful_requests=50,
            denied_requests=50,
        )
        score = compute_risk_score(s, [])
        assert score > 0.0
        assert any("denial_rate" in f for f in s.risk_factors)

    def test_anomaly_severity_critical(self):
        s = AgentUsageSummary(agent_id="a1", total_requests=10, successful_requests=10)
        anomalies = [
            AnomalyAlert(
                alert_type="burst", severity="critical",
                agent_id="a1", description="test"
            )
        ]
        score = compute_risk_score(s, anomalies)
        assert score >= 0.4  # critical = 1.0 * 0.4 weight

    def test_anomaly_severity_low(self):
        s = AgentUsageSummary(agent_id="a1", total_requests=10, successful_requests=10)
        anomalies = [
            AnomalyAlert(
                alert_type="off_hours", severity="low",
                agent_id="a1", description="test"
            )
        ]
        score = compute_risk_score(s, anomalies)
        assert 0.0 < score <= 0.1

    def test_ip_diversity_factor(self):
        s = AgentUsageSummary(
            agent_id="a1",
            total_requests=100,
            successful_requests=100,
            unique_ips=[f"10.0.0.{i}" for i in range(15)],
        )
        score = compute_risk_score(s, [])
        assert score > 0.0
        assert any("ip_diversity" in f for f in s.risk_factors)

    def test_high_volume_factor(self):
        s = AgentUsageSummary(
            agent_id="a1",
            total_requests=1000,
            successful_requests=1000,
            requests_per_hour=300.0,
        )
        score = compute_risk_score(s, [])
        assert score > 0.0
        assert any("high_volume" in f for f in s.risk_factors)

    def test_score_capped_at_1(self):
        s = AgentUsageSummary(
            agent_id="a1",
            total_requests=100,
            denied_requests=100,
            unique_ips=[f"10.0.0.{i}" for i in range(30)],
            requests_per_hour=1000.0,
        )
        anomalies = [
            AnomalyAlert(
                alert_type="burst", severity="critical",
                agent_id="a1", description="test"
            )
        ]
        score = compute_risk_score(s, anomalies)
        assert score == 1.0

    def test_combined_factors(self):
        s = AgentUsageSummary(
            agent_id="a1",
            total_requests=100,
            successful_requests=60,
            denied_requests=40,
            unique_ips=[f"10.0.0.{i}" for i in range(10)],
            requests_per_hour=200.0,
        )
        anomalies = [
            AnomalyAlert(
                alert_type="burst", severity="high",
                agent_id="a1", description="test"
            )
        ]
        score = compute_risk_score(s, anomalies)
        assert score > 0.3
        assert len(s.risk_factors) >= 2

    def test_no_requests_no_denial_factor(self):
        s = AgentUsageSummary(agent_id="a1", total_requests=0)
        score = compute_risk_score(s, [])
        assert score == 0.0

    def test_low_denial_rate_no_factor(self):
        s = AgentUsageSummary(
            agent_id="a1",
            total_requests=100,
            successful_requests=95,
            denied_requests=5,
        )
        score = compute_risk_score(s, [])
        assert score == 0.0  # 5% < 10% threshold

    def test_ip_diversity_below_threshold(self):
        s = AgentUsageSummary(
            agent_id="a1",
            total_requests=100,
            successful_requests=100,
            unique_ips=["10.0.0.1", "10.0.0.2", "10.0.0.3"],
        )
        score = compute_risk_score(s, [])
        assert score == 0.0  # 3 IPs < 5 threshold

    def test_volume_below_threshold(self):
        s = AgentUsageSummary(
            agent_id="a1",
            total_requests=50,
            successful_requests=50,
            requests_per_hour=50.0,
        )
        score = compute_risk_score(s, [])
        assert score == 0.0  # 50/hr < 100/hr threshold

    def test_risk_score_stored_on_summary(self):
        s = AgentUsageSummary(
            agent_id="a1",
            total_requests=100,
            denied_requests=80,
        )
        compute_risk_score(s, [])
        assert s.risk_score > 0.0

    def test_multiple_anomalies_uses_max_severity(self):
        s = AgentUsageSummary(agent_id="a1", total_requests=10, successful_requests=10)
        anomalies = [
            AnomalyAlert(alert_type="off_hours", severity="low", agent_id="a1", description="t"),
            AnomalyAlert(alert_type="burst", severity="high", agent_id="a1", description="t"),
        ]
        score = compute_risk_score(s, anomalies)
        # Should use high severity (0.6 * 0.4 = 0.24), not low
        assert score >= 0.2


# ============================================================
# AnomalyAlert dataclass tests
# ============================================================

class TestAnomalyAlert:
    def test_default_values(self):
        a = AnomalyAlert(
            alert_type="burst", severity="high",
            agent_id="a1", description="test"
        )
        assert a.timestamp == 0.0
        assert a.details == {}

    def test_with_details(self):
        a = AnomalyAlert(
            alert_type="burst", severity="high",
            agent_id="a1", description="test",
            details={"count": 25},
        )
        assert a.details["count"] == 25


# ============================================================
# AgentUsageSummary dataclass tests
# ============================================================

class TestAgentUsageSummary:
    def test_default_values(self):
        s = AgentUsageSummary(agent_id="a1")
        assert s.total_requests == 0
        assert s.risk_score == 0.0
        assert s.peak_hour == -1
        assert s.services_accessed == []
        assert s.risk_factors == []


# ============================================================
# Edge cases and integration-like tests
# ============================================================

class TestAnalyticsEdgeCases:
    def test_single_entry_summary(self):
        e = make_entry(timestamp=1000.0, status="success", service="github", ip_address="10.0.0.1")
        s = compute_agent_summary("a1", [e])
        assert s.total_requests == 1
        assert s.first_seen == 1000.0
        assert s.last_seen == 1000.0
        assert s.requests_per_hour == 1.0  # 1 request, min 1 hour

    def test_all_denied_entries(self):
        entries = [make_entry(status="denied") for _ in range(10)]
        s = compute_agent_summary("a1", entries)
        assert s.denied_requests == 10
        assert s.successful_requests == 0

    def test_all_rate_limited(self):
        entries = [make_entry(status="rate_limited") for _ in range(10)]
        s = compute_agent_summary("a1", entries)
        assert s.rate_limited_requests == 10

    def test_anomaly_detection_with_mixed_agents(self):
        base = time.time()
        entries = [
            make_entry(agent_id="agent-1", timestamp=base + i, status="success")
            for i in range(5)
        ]
        # Even though agent-1 has only 5 entries, anomaly detection works
        alerts = detect_anomalies("agent-1", entries, burst_threshold=3)
        burst_alerts = [a for a in alerts if a.alert_type == "burst"]
        assert len(burst_alerts) == 1

    def test_risk_score_with_empty_anomalies(self):
        s = AgentUsageSummary(
            agent_id="a1",
            total_requests=50,
            successful_requests=50,
        )
        score = compute_risk_score(s, [])
        assert score == 0.0

    def test_burst_at_exact_threshold(self):
        base = time.time()
        entries = make_entries_at_time(20, base, interval=1.0)
        alerts = detect_anomalies("a1", entries, burst_threshold=20)
        burst_alerts = [a for a in alerts if a.alert_type == "burst"]
        assert len(burst_alerts) == 1

    def test_burst_below_threshold_by_one(self):
        base = time.time()
        entries = make_entries_at_time(19, base, interval=1.0)
        alerts = detect_anomalies("a1", entries, burst_threshold=20)
        burst_alerts = [a for a in alerts if a.alert_type == "burst"]
        assert len(burst_alerts) == 0

    def test_off_hours_boundary_start(self):
        dt = datetime(2026, 3, 15, 0, 0, 0, tzinfo=timezone.utc)
        ts = dt.timestamp()
        entries = [make_entry(timestamp=ts)]
        alerts = detect_anomalies("a1", entries, off_hours=(0, 6))
        off_alerts = [a for a in alerts if a.alert_type == "off_hours"]
        assert len(off_alerts) == 1

    def test_off_hours_boundary_end(self):
        dt = datetime(2026, 3, 15, 6, 0, 0, tzinfo=timezone.utc)
        ts = dt.timestamp()
        entries = [make_entry(timestamp=ts)]
        alerts = detect_anomalies("a1", entries, off_hours=(0, 6))
        off_alerts = [a for a in alerts if a.alert_type == "off_hours"]
        assert len(off_alerts) == 0  # hour 6 is NOT in range [0, 6)

    def test_new_ip_needs_minimum_entries(self):
        entries = [
            make_entry(ip_address="10.0.0.1"),
            make_entry(ip_address="10.0.0.99"),
        ]
        alerts = detect_anomalies("a1", entries)
        ip_alerts = [a for a in alerts if a.alert_type == "new_ip"]
        assert len(ip_alerts) == 0  # < 10 entries

    def test_denial_rate_exactly_at_threshold(self):
        # 30% denial = threshold
        entries = [make_entry(status="denied") for _ in range(3)] + [
            make_entry(status="success") for _ in range(7)
        ]
        alerts = detect_anomalies("a1", entries, denial_rate_threshold=0.3)
        denial_alerts = [a for a in alerts if a.alert_type == "high_denial_rate"]
        assert len(denial_alerts) == 0  # Not strictly greater

    def test_denial_rate_just_above_threshold(self):
        # 40% denial > 30% threshold
        entries = [make_entry(status="denied") for _ in range(4)] + [
            make_entry(status="success") for _ in range(6)
        ]
        alerts = detect_anomalies("a1", entries, denial_rate_threshold=0.3)
        denial_alerts = [a for a in alerts if a.alert_type == "high_denial_rate"]
        assert len(denial_alerts) == 1

    def test_scope_escalation_case_insensitive(self):
        entries = [
            make_entry(status="denied", details="EXCESS SCOPES: admin"),
        ] + [make_entry(status="success") for _ in range(9)]
        alerts = detect_anomalies("a1", entries)
        esc_alerts = [a for a in alerts if a.alert_type == "scope_escalation"]
        # "scope" in lower case of "EXCESS SCOPES" -> should detect
        assert len(esc_alerts) == 1

    def test_large_dataset_performance(self):
        # 1000 entries should still be fast
        base = time.time() - 3600
        entries = make_entries_at_time(1000, base, interval=3.6)
        s = compute_agent_summary("a1", entries)
        assert s.total_requests == 1000
        alerts = detect_anomalies("a1", entries)
        # Should complete without issue
        assert isinstance(alerts, list)

    def test_multiple_services_in_summary(self):
        entries = [
            make_entry(service="github"),
            make_entry(service="slack"),
            make_entry(service="google"),
            make_entry(service="jira"),
        ]
        s = compute_agent_summary("a1", entries)
        assert s.services_accessed == ["github", "google", "jira", "slack"]

    def test_ip_diversity_risk_scales(self):
        # 10 IPs -> moderate factor
        s1 = AgentUsageSummary(
            agent_id="a1", total_requests=100, successful_requests=100,
            unique_ips=[f"10.0.0.{i}" for i in range(10)],
        )
        score1 = compute_risk_score(s1, [])

        # 25 IPs -> higher factor (capped)
        s2 = AgentUsageSummary(
            agent_id="a2", total_requests=100, successful_requests=100,
            unique_ips=[f"10.0.0.{i}" for i in range(25)],
        )
        score2 = compute_risk_score(s2, [])
        assert score2 > score1

    def test_volume_risk_scales(self):
        s1 = AgentUsageSummary(
            agent_id="a1", total_requests=500, successful_requests=500,
            requests_per_hour=200.0,
        )
        score1 = compute_risk_score(s1, [])

        s2 = AgentUsageSummary(
            agent_id="a2", total_requests=5000, successful_requests=5000,
            requests_per_hour=500.0,
        )
        score2 = compute_risk_score(s2, [])
        assert score2 > score1

    def test_anomaly_types_in_risk_factors(self):
        s = AgentUsageSummary(agent_id="a1", total_requests=10, successful_requests=10)
        anomalies = [
            AnomalyAlert(alert_type="burst", severity="high", agent_id="a1", description="t"),
            AnomalyAlert(alert_type="off_hours", severity="medium", agent_id="a1", description="t"),
        ]
        compute_risk_score(s, anomalies)
        anomaly_factor = [f for f in s.risk_factors if f.startswith("anomalies=")]
        assert len(anomaly_factor) == 1
        assert "burst" in anomaly_factor[0]
        assert "off_hours" in anomaly_factor[0]
