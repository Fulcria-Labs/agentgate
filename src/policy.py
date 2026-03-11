"""Policy engine for enforcing agent access controls."""

import time
from collections import defaultdict

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


async def enforce_policy(
    user_id: str,
    agent_id: str,
    service: str,
    requested_scopes: list[str],
    ip_address: str = "",
) -> AgentPolicy:
    """Check all policy constraints before issuing a token.

    Returns the policy if all checks pass, raises PolicyDenied otherwise.
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

    # 2. Service must be in the allowed list
    if service not in policy.allowed_services:
        await log_audit(user_id, agent_id, service, "token_request", "denied",
                       details=f"Service '{service}' not in allowed list: {policy.allowed_services}")
        raise PolicyDenied(
            f"Agent '{policy.agent_name}' is not authorized to access '{service}'",
            agent_id, service,
        )

    # 3. Requested scopes must be within allowed scopes
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

    # 4. Rate limit check
    now = time.time()
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

    # 5. Step-up auth check (returns policy, caller handles CIBA flow)
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
