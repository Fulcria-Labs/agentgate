"""Token usage analytics and anomaly detection for AgentGate."""

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import aiosqlite

from .database import DB_PATH, AuditEntry


@dataclass
class AgentUsageSummary:
    """Usage summary for a single agent."""
    agent_id: str
    total_requests: int = 0
    successful_requests: int = 0
    denied_requests: int = 0
    rate_limited_requests: int = 0
    services_accessed: list[str] = field(default_factory=list)
    unique_ips: list[str] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    requests_per_hour: float = 0.0
    peak_hour: int = -1  # 0-23 UTC
    risk_score: float = 0.0  # 0.0 (safe) to 1.0 (critical)
    risk_factors: list[str] = field(default_factory=list)


@dataclass
class AnomalyAlert:
    """A detected anomaly in agent behavior."""
    alert_type: str  # burst, off_hours, new_ip, scope_escalation, high_denial_rate
    severity: str  # low, medium, high, critical
    agent_id: str
    description: str
    timestamp: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Compliance summary for audit export."""
    generated_at: float = 0.0
    period_start: float = 0.0
    period_end: float = 0.0
    total_agents: int = 0
    active_agents: int = 0
    total_requests: int = 0
    denied_requests: int = 0
    emergency_revokes: int = 0
    step_up_challenges: int = 0
    policy_changes: int = 0
    unique_ips: int = 0
    agents: list[AgentUsageSummary] = field(default_factory=list)
    anomalies: list[AnomalyAlert] = field(default_factory=list)


async def get_audit_entries(
    user_id: str,
    since: float = 0.0,
    until: float = 0.0,
) -> list[AuditEntry]:
    """Fetch audit entries for a user within a time range."""
    if until == 0.0:
        until = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT * FROM audit_log
               WHERE user_id = ? AND timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp ASC""",
            (user_id, since, until),
        )
        rows = await cursor.fetchall()
        return [
            AuditEntry(
                id=row["id"],
                timestamp=row["timestamp"],
                user_id=row["user_id"],
                agent_id=row["agent_id"],
                service=row["service"],
                scopes=row["scopes"],
                action=row["action"],
                status=row["status"],
                ip_address=row["ip_address"],
                details=row["details"],
            )
            for row in rows
        ]


def compute_agent_summary(agent_id: str, entries: list[AuditEntry]) -> AgentUsageSummary:
    """Compute usage statistics for a single agent from audit entries."""
    summary = AgentUsageSummary(agent_id=agent_id)
    if not entries:
        return summary

    services = set()
    ips = set()
    hour_counts: dict[int, int] = defaultdict(int)
    timestamps: list[float] = []

    for e in entries:
        summary.total_requests += 1
        timestamps.append(e.timestamp)

        if e.status == "success":
            summary.successful_requests += 1
        elif e.status == "denied":
            summary.denied_requests += 1
        elif e.status == "rate_limited":
            summary.rate_limited_requests += 1

        if e.service:
            services.add(e.service)
        if e.ip_address:
            ips.add(e.ip_address)

        # Track hourly distribution
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(e.timestamp, tz=timezone.utc)
        hour_counts[dt.hour] += 1

    summary.services_accessed = sorted(services)
    summary.unique_ips = sorted(ips)
    summary.first_seen = min(timestamps) if timestamps else 0.0
    summary.last_seen = max(timestamps) if timestamps else 0.0

    # Compute requests per hour
    if summary.first_seen and summary.last_seen:
        duration_hours = max((summary.last_seen - summary.first_seen) / 3600, 1.0)
        summary.requests_per_hour = summary.total_requests / duration_hours

    # Peak hour
    if hour_counts:
        summary.peak_hour = max(hour_counts, key=hour_counts.get)

    return summary


def detect_anomalies(
    agent_id: str,
    entries: list[AuditEntry],
    baseline_rph: float = 10.0,
    burst_window_seconds: int = 60,
    burst_threshold: int = 20,
    denial_rate_threshold: float = 0.3,
    off_hours: tuple[int, int] = (0, 6),
) -> list[AnomalyAlert]:
    """Detect anomalous patterns in agent behavior.

    Checks for:
    - Burst activity (many requests in a short window)
    - Off-hours access (requests outside normal business hours)
    - High denial rate (many denied requests)
    - New/unusual IP addresses
    - Scope escalation attempts (requesting scopes that get denied)
    """
    alerts: list[AnomalyAlert] = []
    if not entries:
        return alerts

    from datetime import datetime, timezone

    # --- Burst detection ---
    timestamps = [e.timestamp for e in entries]
    for i, ts in enumerate(timestamps):
        window_end = ts + burst_window_seconds
        count = sum(1 for t in timestamps[i:] if t <= window_end)
        if count >= burst_threshold:
            alerts.append(AnomalyAlert(
                alert_type="burst",
                severity="high",
                agent_id=agent_id,
                description=f"{count} requests in {burst_window_seconds}s window",
                timestamp=ts,
                details={"count": count, "window_seconds": burst_window_seconds},
            ))
            break  # Only report first burst

    # --- Off-hours access ---
    off_start, off_end = off_hours
    off_hours_entries = []
    for e in entries:
        dt = datetime.fromtimestamp(e.timestamp, tz=timezone.utc)
        if off_start <= dt.hour < off_end:
            off_hours_entries.append(e)

    if off_hours_entries:
        alerts.append(AnomalyAlert(
            alert_type="off_hours",
            severity="medium",
            agent_id=agent_id,
            description=(
                f"{len(off_hours_entries)} requests during off-hours "
                f"({off_start:02d}:00-{off_end:02d}:00 UTC)"
            ),
            timestamp=off_hours_entries[0].timestamp,
            details={"count": len(off_hours_entries), "off_start": off_start, "off_end": off_end},
        ))

    # --- High denial rate ---
    total = len(entries)
    denied = sum(1 for e in entries if e.status == "denied")
    if total >= 5 and denied / total > denial_rate_threshold:
        rate = denied / total
        severity = "critical" if rate > 0.6 else "high" if rate > 0.4 else "medium"
        alerts.append(AnomalyAlert(
            alert_type="high_denial_rate",
            severity=severity,
            agent_id=agent_id,
            description=f"{denied}/{total} requests denied ({rate:.0%})",
            timestamp=entries[-1].timestamp,
            details={"denied": denied, "total": total, "rate": round(rate, 3)},
        ))

    # --- New IP detection ---
    # Track IPs seen in first half vs second half of the period
    if len(entries) >= 10:
        mid = len(entries) // 2
        early_ips = {e.ip_address for e in entries[:mid] if e.ip_address}
        late_ips = {e.ip_address for e in entries[mid:] if e.ip_address}
        new_ips = late_ips - early_ips
        if new_ips:
            alerts.append(AnomalyAlert(
                alert_type="new_ip",
                severity="medium",
                agent_id=agent_id,
                description=f"{len(new_ips)} new IP(s) detected: {', '.join(sorted(new_ips))}",
                timestamp=entries[mid].timestamp,
                details={"new_ips": sorted(new_ips)},
            ))

    # --- Scope escalation ---
    escalation_entries = [
        e for e in entries
        if e.status == "denied" and "scope" in e.details.lower()
    ]
    if escalation_entries:
        alerts.append(AnomalyAlert(
            alert_type="scope_escalation",
            severity="high",
            agent_id=agent_id,
            description=f"{len(escalation_entries)} scope escalation attempt(s)",
            timestamp=escalation_entries[0].timestamp,
            details={"count": len(escalation_entries)},
        ))

    return alerts


def compute_risk_score(summary: AgentUsageSummary, anomalies: list[AnomalyAlert]) -> float:
    """Compute a 0.0-1.0 risk score for an agent based on usage and anomalies.

    Factors:
    - Denial rate (weight: 0.3)
    - Anomaly severity (weight: 0.4)
    - IP diversity (weight: 0.15)
    - Request volume (weight: 0.15)
    """
    score = 0.0
    factors = []

    # Denial rate factor
    if summary.total_requests > 0:
        denial_rate = summary.denied_requests / summary.total_requests
        if denial_rate > 0.1:
            score += min(denial_rate, 1.0) * 0.3
            factors.append(f"denial_rate={denial_rate:.0%}")

    # Anomaly severity factor
    severity_weights = {"low": 0.1, "medium": 0.3, "high": 0.6, "critical": 1.0}
    if anomalies:
        max_severity = max(severity_weights.get(a.severity, 0) for a in anomalies)
        anomaly_score = max_severity * 0.4
        score += anomaly_score
        anomaly_types = sorted({a.alert_type for a in anomalies})
        factors.append(f"anomalies={','.join(anomaly_types)}")

    # IP diversity factor (many IPs = potentially shared/leaked credentials)
    ip_count = len(summary.unique_ips)
    if ip_count > 5:
        ip_factor = min((ip_count - 5) / 20, 1.0) * 0.15
        score += ip_factor
        factors.append(f"ip_diversity={ip_count}")

    # Volume factor (very high volume = potential abuse)
    if summary.requests_per_hour > 100:
        vol_factor = min((summary.requests_per_hour - 100) / 500, 1.0) * 0.15
        score += vol_factor
        factors.append(f"high_volume={summary.requests_per_hour:.0f}/hr")

    summary.risk_score = min(score, 1.0)
    summary.risk_factors = factors
    return summary.risk_score


async def generate_usage_analytics(
    user_id: str,
    hours: int = 24,
) -> dict:
    """Generate a complete analytics report for a user's agents.

    Returns a dict with agent summaries, anomalies, and top-level stats.
    """
    since = time.time() - (hours * 3600)
    entries = await get_audit_entries(user_id, since=since)

    # Group entries by agent
    by_agent: dict[str, list[AuditEntry]] = defaultdict(list)
    for e in entries:
        if e.agent_id and e.agent_id != "user" and e.agent_id != "*":
            by_agent[e.agent_id].append(e)

    agent_summaries = []
    all_anomalies = []

    for agent_id, agent_entries in by_agent.items():
        summary = compute_agent_summary(agent_id, agent_entries)
        anomalies = detect_anomalies(agent_id, agent_entries)
        compute_risk_score(summary, anomalies)
        agent_summaries.append(summary)
        all_anomalies.extend(anomalies)

    # Sort by risk score descending
    agent_summaries.sort(key=lambda s: s.risk_score, reverse=True)

    return {
        "period_hours": hours,
        "total_agents": len(agent_summaries),
        "total_requests": sum(s.total_requests for s in agent_summaries),
        "total_denied": sum(s.denied_requests for s in agent_summaries),
        "anomaly_count": len(all_anomalies),
        "high_risk_agents": [s.agent_id for s in agent_summaries if s.risk_score > 0.5],
        "agents": [
            {
                "agent_id": s.agent_id,
                "total_requests": s.total_requests,
                "successful": s.successful_requests,
                "denied": s.denied_requests,
                "rate_limited": s.rate_limited_requests,
                "services": s.services_accessed,
                "unique_ips": len(s.unique_ips),
                "requests_per_hour": round(s.requests_per_hour, 2),
                "peak_hour": s.peak_hour,
                "risk_score": round(s.risk_score, 3),
                "risk_factors": s.risk_factors,
            }
            for s in agent_summaries
        ],
        "anomalies": [
            {
                "type": a.alert_type,
                "severity": a.severity,
                "agent_id": a.agent_id,
                "description": a.description,
                "timestamp": a.timestamp,
                "details": a.details,
            }
            for a in all_anomalies
        ],
    }


async def generate_compliance_report(
    user_id: str,
    period_days: int = 30,
) -> dict:
    """Generate a compliance report suitable for SOC2/GDPR audit export.

    Includes:
    - Total access events and outcomes
    - Emergency revocations
    - Policy changes
    - Step-up authentication challenges
    - Anomaly summary
    """
    now = time.time()
    since = now - (period_days * 86400)
    entries = await get_audit_entries(user_id, since=since)

    # Categorize entries
    emergency_revokes = [e for e in entries if e.action == "emergency_revoke"]
    step_ups = [e for e in entries if "step_up" in e.action]
    policy_changes = [
        e for e in entries
        if e.action in ("policy_created", "policy_deleted", "agent_enabled", "agent_disabled")
    ]
    token_requests = [e for e in entries if e.action in ("token_request", "token_issued")]
    all_ips = {e.ip_address for e in entries if e.ip_address}

    # Per-agent summaries
    by_agent: dict[str, list[AuditEntry]] = defaultdict(list)
    for e in entries:
        if e.agent_id and e.agent_id != "user" and e.agent_id != "*":
            by_agent[e.agent_id].append(e)

    agent_summaries = []
    all_anomalies = []
    active_agents = set()
    for agent_id, agent_entries in by_agent.items():
        summary = compute_agent_summary(agent_id, agent_entries)
        anomalies = detect_anomalies(agent_id, agent_entries)
        compute_risk_score(summary, anomalies)
        agent_summaries.append(summary)
        all_anomalies.extend(anomalies)
        # Active = had a successful request in the period
        if summary.successful_requests > 0:
            active_agents.add(agent_id)

    denied_count = sum(1 for e in token_requests if e.status == "denied")

    return {
        "report_type": "compliance",
        "generated_at": now,
        "period_start": since,
        "period_end": now,
        "period_days": period_days,
        "summary": {
            "total_agents": len(by_agent),
            "active_agents": len(active_agents),
            "total_access_events": len(token_requests),
            "denied_events": denied_count,
            "denial_rate": round(denied_count / max(len(token_requests), 1), 3),
            "emergency_revocations": len(emergency_revokes),
            "step_up_challenges": len(step_ups),
            "policy_changes": len(policy_changes),
            "unique_source_ips": len(all_ips),
            "anomalies_detected": len(all_anomalies),
        },
        "risk_overview": {
            "high_risk_agents": [
                {"agent_id": s.agent_id, "risk_score": round(s.risk_score, 3), "factors": s.risk_factors}
                for s in agent_summaries if s.risk_score > 0.5
            ],
            "critical_anomalies": [
                {"type": a.alert_type, "agent_id": a.agent_id, "description": a.description}
                for a in all_anomalies if a.severity == "critical"
            ],
        },
        "policy_changes": [
            {
                "timestamp": e.timestamp,
                "agent_id": e.agent_id,
                "action": e.action,
                "details": e.details,
            }
            for e in policy_changes
        ],
        "emergency_revocations": [
            {
                "timestamp": e.timestamp,
                "ip_address": e.ip_address,
                "details": e.details,
            }
            for e in emergency_revokes
        ],
    }
