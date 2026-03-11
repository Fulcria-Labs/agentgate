"""Policy engine for enforcing agent access controls."""

import ipaddress
import time
from collections import defaultdict
from datetime import datetime, timezone

from .database import AgentPolicy, get_agent_policy, log_audit

# In-memory rate limit tracking (resets on restart, fine for demo)
_rate_counters: dict[str, list[float]] = defaultdict(list)


class PolicyDenied(Exception):
    """Raised when a policy check fails."""
    def __init__(self, reason: str, agent_id: str = "", service: str = ""):
        self.reason = reason
        self.agent_id = agent_id
        self.service = service
        super().__init__(reason)


def check_time_window(policy: AgentPolicy, now: datetime | None = None) -> str | None:
    """Check if the current time falls within the policy's allowed time window.

    Returns None if access is allowed, or a denial reason string if denied.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Check allowed days (0=Monday .. 6=Sunday)
    if policy.allowed_days:
        current_day = now.weekday()
        if current_day not in policy.allowed_days:
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            allowed = ", ".join(day_names[d] for d in sorted(policy.allowed_days))
            return f"Access denied: not within allowed days ({allowed})"

    # Check allowed hours (0-23 UTC)
    if policy.allowed_hours:
        current_hour = now.hour
        if current_hour not in policy.allowed_hours:
            return (
                f"Access denied: not within allowed hours "
                f"({min(policy.allowed_hours):02d}:00-{max(policy.allowed_hours):02d}:59 UTC)"
            )

    return None


def check_ip_allowlist(policy: AgentPolicy, ip_address: str) -> str | None:
    """Check if the requesting IP is in the policy's allowlist.

    Supports both exact IPs and CIDR notation (e.g., '192.168.1.0/24').
    Returns None if access is allowed, or a denial reason string if denied.
    """
    if not policy.ip_allowlist:
        return None  # Empty = allow all

    try:
        addr = ipaddress.ip_address(ip_address)
    except ValueError:
        return f"Access denied: invalid IP address '{ip_address}'"

    for entry in policy.ip_allowlist:
        try:
            if "/" in entry:
                if addr in ipaddress.ip_network(entry, strict=False):
                    return None
            else:
                if addr == ipaddress.ip_address(entry):
                    return None
        except ValueError:
            continue  # Skip malformed entries

    return f"Access denied: IP {ip_address} not in allowlist"


async def enforce_policy(
    user_id: str,
    agent_id: str,
    service: str,
    requested_scopes: list[str],
    ip_address: str = "",
) -> AgentPolicy:
    """Check all policy constraints before issuing a token.

    Returns the policy if all checks pass, raises PolicyDenied otherwise.

    Enforces (in order):
      1. Agent existence and active status
      2. Policy ownership (user must own the policy)
      3. Policy expiration
      4. Time-based access windows (allowed hours/days)
      5. IP allowlist
      6. Service authorization
      7. Scope restrictions
      8. Rate limiting
    """
    policy = await get_agent_policy(agent_id)

    # 1. Agent must exist and be active
    if not policy:
        await log_audit(user_id, agent_id, service, "token_request", "denied",
                       details="Agent not registered")
        raise PolicyDenied("Agent not registered", agent_id, service)

    if not policy.is_active:
        await log_audit(user_id, agent_id, service, "token_request", "denied",
                       details="Agent is disabled")
        raise PolicyDenied("Agent is disabled", agent_id, service)

    # 2. Policy ownership: requesting user must be the policy creator
    if policy.created_by != user_id:
        await log_audit(user_id, agent_id, service, "token_request", "denied",
                       details=f"Ownership violation: user '{user_id}' is not policy owner",
                       ip_address=ip_address)
        raise PolicyDenied(
            "Not authorized: you do not own this agent's policy",
            agent_id, service,
        )

    # 3. Policy expiration check
    now = time.time()
    if policy.expires_at > 0 and now > policy.expires_at:
        await log_audit(user_id, agent_id, service, "token_request", "denied",
                       details="Policy has expired")
        raise PolicyDenied("Agent policy has expired", agent_id, service)

    # 4. Time-based access window check
    time_denial = check_time_window(policy)
    if time_denial:
        await log_audit(user_id, agent_id, service, "token_request", "denied",
                       details=time_denial, ip_address=ip_address)
        raise PolicyDenied(time_denial, agent_id, service)

    # 5. IP allowlist check
    ip_denial = check_ip_allowlist(policy, ip_address)
    if ip_denial:
        await log_audit(user_id, agent_id, service, "token_request", "denied",
                       details=ip_denial, ip_address=ip_address)
        raise PolicyDenied(ip_denial, agent_id, service)

    # 6. Service must be in the allowed list
    if service not in policy.allowed_services:
        await log_audit(user_id, agent_id, service, "token_request", "denied",
                       details=f"Service '{service}' not in allowed list: {policy.allowed_services}")
        raise PolicyDenied(
            f"Agent '{policy.agent_name}' is not authorized to access '{service}'",
            agent_id, service,
        )

    # 7. Requested scopes must be within allowed scopes
    allowed = set(policy.allowed_scopes.get(service, []))
    requested = set(requested_scopes)
    excess = requested - allowed
    if excess:
        await log_audit(user_id, agent_id, service, "token_request", "denied",
                       scopes=",".join(requested_scopes),
                       details=f"Excess scopes: {excess}")
        raise PolicyDenied(
            f"Scopes not permitted: {', '.join(excess)}",
            agent_id, service,
        )

    # 8. Rate limit check
    window_start = now - 60
    key = f"{agent_id}:{service}"
    _rate_counters[key] = [t for t in _rate_counters[key] if t > window_start]
    if len(_rate_counters[key]) >= policy.rate_limit_per_minute:
        await log_audit(user_id, agent_id, service, "token_request", "rate_limited",
                       details=f"Rate limit: {policy.rate_limit_per_minute}/min")
        raise PolicyDenied(
            f"Rate limit exceeded ({policy.rate_limit_per_minute}/min)",
            agent_id, service,
        )
    _rate_counters[key].append(now)

    # 9. Step-up auth check (returns policy, caller handles CIBA flow)
    # This is checked but not enforced here - the caller must handle it
    # by calling trigger_step_up_auth if the service requires it

    return policy


def requires_step_up(policy: AgentPolicy, service: str) -> bool:
    """Check if a service requires step-up authentication for this agent."""
    return service in policy.requires_step_up


def get_effective_scopes(policy: AgentPolicy, service: str, requested: list[str]) -> list[str]:
    """Return the intersection of requested and allowed scopes."""
    allowed = set(policy.allowed_scopes.get(service, []))
    return list(set(requested) & allowed)
