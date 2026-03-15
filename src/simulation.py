"""Policy simulation engine for dry-run authorization checks.

Allows testing whether a token request would succeed without actually
issuing a token. Returns a detailed report of which policy checks
pass or fail, useful for debugging agent configurations.
"""

import ipaddress
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .database import get_agent_policy, get_rate_limit_count, AgentPolicy
from .policy import check_time_window, check_ip_allowlist, _rate_counters


@dataclass
class CheckResult:
    """Result of a single policy check."""
    name: str
    passed: bool
    detail: str


@dataclass
class SimulationResult:
    """Full simulation result for a token request."""
    would_succeed: bool
    agent_id: str
    service: str
    requested_scopes: list[str]
    effective_scopes: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "would_succeed": self.would_succeed,
            "agent_id": self.agent_id,
            "service": self.service,
            "requested_scopes": self.requested_scopes,
            "effective_scopes": self.effective_scopes,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
            "timestamp": self.timestamp,
            "checks_passed": sum(1 for c in self.checks if c.passed),
            "checks_failed": sum(1 for c in self.checks if not c.passed),
            "total_checks": len(self.checks),
        }


async def simulate_token_request(
    user_id: str,
    agent_id: str,
    service: str,
    requested_scopes: list[str],
    ip_address: str = "",
    now: datetime | None = None,
) -> SimulationResult:
    """Simulate a token request without actually issuing a token.

    Runs all policy checks and returns a detailed report of which
    checks pass or fail.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    result = SimulationResult(
        would_succeed=True,
        agent_id=agent_id,
        service=service,
        requested_scopes=requested_scopes,
        timestamp=time.time(),
    )

    # 1. Agent existence
    policy = await get_agent_policy(agent_id)
    if not policy:
        result.checks.append(CheckResult(
            "agent_exists", False, f"Agent '{agent_id}' not registered",
        ))
        result.would_succeed = False
        return result
    result.checks.append(CheckResult(
        "agent_exists", True, f"Agent '{policy.agent_name}' found",
    ))

    # 2. Active status
    if not policy.is_active:
        result.checks.append(CheckResult(
            "agent_active", False, "Agent is disabled",
        ))
        result.would_succeed = False
    else:
        result.checks.append(CheckResult(
            "agent_active", True, "Agent is active",
        ))

    # 3. Ownership
    if policy.created_by != user_id:
        result.checks.append(CheckResult(
            "ownership", False,
            f"User '{user_id}' does not own this policy (owner: '{policy.created_by}')",
        ))
        result.would_succeed = False
    else:
        result.checks.append(CheckResult(
            "ownership", True, "User owns this agent's policy",
        ))

    # 4. Expiration
    current_time = time.time()
    if policy.expires_at > 0 and current_time > policy.expires_at:
        result.checks.append(CheckResult(
            "expiration", False,
            f"Policy expired at {policy.expires_at:.0f} (now: {current_time:.0f})",
        ))
        result.would_succeed = False
    elif policy.expires_at > 0:
        remaining = policy.expires_at - current_time
        result.checks.append(CheckResult(
            "expiration", True,
            f"Policy expires in {remaining:.0f} seconds",
        ))
    else:
        result.checks.append(CheckResult(
            "expiration", True, "No expiration set",
        ))

    # 5. Time window
    time_denial = check_time_window(policy, now)
    if time_denial:
        result.checks.append(CheckResult(
            "time_window", False, time_denial,
        ))
        result.would_succeed = False
    else:
        detail = "No time restrictions"
        if policy.allowed_hours:
            detail = f"Current hour {now.hour} is within allowed hours"
        if policy.allowed_days:
            detail += f"; day {now.weekday()} is allowed"
        result.checks.append(CheckResult("time_window", True, detail))

    # 6. IP allowlist
    if ip_address:
        ip_denial = check_ip_allowlist(policy, ip_address)
        if ip_denial:
            result.checks.append(CheckResult(
                "ip_allowlist", False, ip_denial,
            ))
            result.would_succeed = False
        else:
            if policy.ip_allowlist:
                result.checks.append(CheckResult(
                    "ip_allowlist", True,
                    f"IP {ip_address} is in allowlist ({len(policy.ip_allowlist)} entries)",
                ))
            else:
                result.checks.append(CheckResult(
                    "ip_allowlist", True, "No IP restrictions configured",
                ))
    else:
        result.checks.append(CheckResult(
            "ip_allowlist", True, "No IP provided (skipped)",
        ))

    # 7. Service authorization
    if service not in policy.allowed_services:
        result.checks.append(CheckResult(
            "service_auth", False,
            f"Service '{service}' not in allowed list: {policy.allowed_services}",
        ))
        result.would_succeed = False
    else:
        result.checks.append(CheckResult(
            "service_auth", True,
            f"Service '{service}' is authorized",
        ))

    # 8. Scope validation
    allowed = set(policy.allowed_scopes.get(service, []))
    requested = set(requested_scopes)
    excess = requested - allowed
    if excess:
        result.checks.append(CheckResult(
            "scope_validation", False,
            f"Scopes not permitted: {', '.join(excess)} (allowed: {', '.join(allowed)})",
        ))
        result.would_succeed = False
    else:
        effective = list(requested & allowed)
        result.effective_scopes = effective
        result.checks.append(CheckResult(
            "scope_validation", True,
            f"All requested scopes permitted; effective scopes: {effective}",
        ))

    # 9. Rate limit check
    key = f"{agent_id}:{service}"
    current_count = len([t for t in _rate_counters.get(key, []) if t > current_time - 60])
    if current_count >= policy.rate_limit_per_minute:
        result.checks.append(CheckResult(
            "rate_limit", False,
            f"Rate limit exceeded: {current_count}/{policy.rate_limit_per_minute} per minute",
        ))
        result.would_succeed = False
    else:
        result.checks.append(CheckResult(
            "rate_limit", True,
            f"Within rate limit: {current_count}/{policy.rate_limit_per_minute} per minute",
        ))

    # 10. Step-up auth check
    if service in policy.requires_step_up:
        result.checks.append(CheckResult(
            "step_up_auth", True,
            "Step-up auth required (would prompt user for CIBA approval)",
        ))
    else:
        result.checks.append(CheckResult(
            "step_up_auth", True, "No step-up auth required for this service",
        ))

    return result
