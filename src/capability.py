"""Agent capability discovery protocol for AgentGate.

Provides a structured way for agents to discover:
- What services they can access
- What scopes/permissions are available per service
- Current rate limits and usage
- Active constraints (time windows, IP restrictions)
- Delegation chain information
- Consent requirements

This implements a capability manifest that agents can query to
understand their authorization boundaries before making requests,
reducing denied requests and improving agent reliability.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .database import (
    AgentPolicy,
    get_agent_policy,
    get_rate_limit_count,
    get_connected_services,
)
from .policy import _rate_counters, check_time_window, check_ip_allowlist
from .delegation import get_delegated_permissions


@dataclass
class ServiceCapability:
    """Capability description for a single service."""
    service: str
    available: bool = True
    allowed_scopes: list[str] = field(default_factory=list)
    requires_step_up: bool = False
    connected: bool = False
    rate_limit_remaining: int = 0
    rate_limit_total: int = 0
    constraints: dict = field(default_factory=dict)


@dataclass
class CapabilityManifest:
    """Complete capability manifest for an agent."""
    agent_id: str
    agent_name: str = ""
    is_active: bool = True
    services: list[ServiceCapability] = field(default_factory=list)
    delegation_info: dict | None = None
    time_constraints: dict = field(default_factory=dict)
    ip_constraints: dict = field(default_factory=dict)
    policy_expires_at: float = 0.0
    current_time_allowed: bool = True
    generated_at: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "is_active": self.is_active,
            "services": [
                {
                    "service": s.service,
                    "available": s.available,
                    "allowed_scopes": s.allowed_scopes,
                    "requires_step_up": s.requires_step_up,
                    "connected": s.connected,
                    "rate_limit": {
                        "remaining": s.rate_limit_remaining,
                        "total": s.rate_limit_total,
                        "window_seconds": 60,
                    },
                    "constraints": s.constraints,
                }
                for s in self.services
            ],
            "delegation_info": self.delegation_info,
            "time_constraints": self.time_constraints,
            "ip_constraints": self.ip_constraints,
            "policy_expires_at": self.policy_expires_at or None,
            "policy_expires_in": (
                max(0, self.policy_expires_at - time.time())
                if self.policy_expires_at > 0
                else None
            ),
            "current_time_allowed": self.current_time_allowed,
            "generated_at": self.generated_at,
            "warnings": self.warnings,
            "service_count": len(self.services),
            "available_service_count": sum(1 for s in self.services if s.available),
        }


async def discover_capabilities(
    agent_id: str,
    user_id: str = "",
    ip_address: str = "",
    include_delegation: bool = True,
    include_connected: bool = True,
) -> CapabilityManifest:
    """Discover the full capability manifest for an agent.

    This is the primary API for agents to understand their authorization
    boundaries. Returns a structured manifest describing:
    - Available services and scopes
    - Rate limit status per service
    - Time/IP constraints
    - Delegation chain info
    - Warnings about approaching limits

    Args:
        agent_id: The agent to discover capabilities for
        user_id: Optional user context for connected services check
        ip_address: Optional IP to check against allowlist
        include_delegation: Whether to include delegation chain info
        include_connected: Whether to check connected service status
    """
    now = time.time()
    manifest = CapabilityManifest(
        agent_id=agent_id,
        generated_at=now,
    )

    # Fetch the agent's policy
    policy = await get_agent_policy(agent_id)
    if not policy:
        manifest.is_active = False
        manifest.warnings.append("Agent not registered")
        return manifest

    manifest.agent_name = policy.agent_name
    manifest.is_active = policy.is_active

    if not policy.is_active:
        manifest.warnings.append("Agent is currently disabled")
        return manifest

    # Check policy expiration
    manifest.policy_expires_at = policy.expires_at
    if policy.expires_at > 0:
        remaining = policy.expires_at - now
        if remaining <= 0:
            manifest.warnings.append("Policy has expired")
            manifest.is_active = False
            return manifest
        elif remaining < 3600:
            manifest.warnings.append(
                f"Policy expires in {remaining:.0f} seconds (less than 1 hour)"
            )
        elif remaining < 86400:
            hours = remaining / 3600
            manifest.warnings.append(
                f"Policy expires in {hours:.1f} hours (less than 1 day)"
            )

    # Time constraints
    if policy.allowed_hours or policy.allowed_days:
        manifest.time_constraints = {
            "allowed_hours": policy.allowed_hours,
            "allowed_days": policy.allowed_days,
        }
        time_denial = check_time_window(policy)
        if time_denial:
            manifest.current_time_allowed = False
            manifest.warnings.append(f"Currently outside allowed time window")

    # IP constraints
    if policy.ip_allowlist:
        manifest.ip_constraints = {
            "ip_allowlist": policy.ip_allowlist,
            "ip_count": len(policy.ip_allowlist),
        }
        if ip_address:
            ip_denial = check_ip_allowlist(policy, ip_address)
            if ip_denial:
                manifest.warnings.append(f"Current IP {ip_address} not in allowlist")

    # Connected services
    connected_services = set()
    if include_connected and user_id:
        try:
            connected = await get_connected_services(user_id)
            connected_services = {s["service"] for s in connected}
        except Exception:
            pass

    # Build per-service capabilities
    for service in policy.allowed_services:
        scopes = policy.allowed_scopes.get(service, [])
        requires_step_up = service in policy.requires_step_up
        is_connected = service in connected_services

        # Calculate rate limit remaining
        key = f"{agent_id}:{service}"
        window_start = now - 60
        current_count = len([t for t in _rate_counters.get(key, []) if t > window_start])
        remaining = max(0, policy.rate_limit_per_minute - current_count)

        constraints = {}
        if requires_step_up:
            constraints["requires_step_up"] = True
        if not is_connected and include_connected and user_id:
            constraints["not_connected"] = True

        cap = ServiceCapability(
            service=service,
            available=manifest.current_time_allowed and manifest.is_active,
            allowed_scopes=scopes,
            requires_step_up=requires_step_up,
            connected=is_connected,
            rate_limit_remaining=remaining,
            rate_limit_total=policy.rate_limit_per_minute,
            constraints=constraints,
        )
        manifest.services.append(cap)

        # Rate limit warnings
        if remaining == 0:
            manifest.warnings.append(f"Rate limit exhausted for {service}")
        elif remaining <= policy.rate_limit_per_minute * 0.1:
            manifest.warnings.append(
                f"Rate limit nearly exhausted for {service}: {remaining} remaining"
            )

    # Delegation info
    if include_delegation:
        try:
            delegation = await get_delegated_permissions(agent_id)
            if delegation:
                manifest.delegation_info = {
                    "is_delegated": True,
                    "parent_agent_id": delegation["parent_agent_id"],
                    "chain_path": delegation["chain_path"],
                    "effective_services": delegation["effective_services"],
                    "effective_scopes": delegation["effective_scopes"],
                    "delegation_expires_at": delegation.get("expires_at", 0),
                }
        except Exception:
            pass

    return manifest


async def check_capability(
    agent_id: str,
    service: str,
    scopes: list[str] | None = None,
    user_id: str = "",
    ip_address: str = "",
) -> dict:
    """Quick check if an agent can perform a specific action.

    Returns a simple pass/fail with details, without the full manifest overhead.
    Useful for pre-flight checks before making a token request.
    """
    now = time.time()
    result = {
        "agent_id": agent_id,
        "service": service,
        "can_access": False,
        "reasons": [],
        "checked_at": now,
    }

    policy = await get_agent_policy(agent_id)
    if not policy:
        result["reasons"].append("Agent not registered")
        return result

    if not policy.is_active:
        result["reasons"].append("Agent is disabled")
        return result

    # Ownership check
    if user_id and policy.created_by != user_id:
        result["reasons"].append("User does not own this agent's policy")
        return result

    # Expiration
    if policy.expires_at > 0 and now > policy.expires_at:
        result["reasons"].append("Policy has expired")
        return result

    # Time window
    time_denial = check_time_window(policy)
    if time_denial:
        result["reasons"].append(time_denial)
        return result

    # IP check
    if ip_address:
        ip_denial = check_ip_allowlist(policy, ip_address)
        if ip_denial:
            result["reasons"].append(ip_denial)
            return result

    # Service check
    if service not in policy.allowed_services:
        result["reasons"].append(f"Service '{service}' not authorized")
        return result

    # Scope check
    if scopes:
        allowed = set(policy.allowed_scopes.get(service, []))
        excess = set(scopes) - allowed
        if excess:
            result["reasons"].append(f"Scopes not permitted: {', '.join(excess)}")
            return result
        result["effective_scopes"] = list(set(scopes) & allowed)

    # Rate limit check
    key = f"{agent_id}:{service}"
    window_start = now - 60
    current_count = len([t for t in _rate_counters.get(key, []) if t > window_start])
    if current_count >= policy.rate_limit_per_minute:
        result["reasons"].append("Rate limit exceeded")
        return result

    result["can_access"] = True
    result["rate_limit_remaining"] = policy.rate_limit_per_minute - current_count
    result["requires_step_up"] = service in policy.requires_step_up
    return result


def compute_capability_diff(
    manifest_before: CapabilityManifest,
    manifest_after: CapabilityManifest,
) -> dict:
    """Compare two capability manifests and return the differences.

    Useful for tracking how an agent's capabilities change over time,
    e.g., after a policy update or delegation change.
    """
    diff = {
        "agent_id": manifest_after.agent_id,
        "changes": [],
    }

    # Active status change
    if manifest_before.is_active != manifest_after.is_active:
        diff["changes"].append({
            "type": "status_change",
            "field": "is_active",
            "before": manifest_before.is_active,
            "after": manifest_after.is_active,
        })

    # Service changes
    before_services = {s.service for s in manifest_before.services}
    after_services = {s.service for s in manifest_after.services}

    added = after_services - before_services
    removed = before_services - after_services

    for svc in added:
        diff["changes"].append({
            "type": "service_added",
            "service": svc,
        })

    for svc in removed:
        diff["changes"].append({
            "type": "service_removed",
            "service": svc,
        })

    # Scope changes for common services
    before_map = {s.service: s for s in manifest_before.services}
    after_map = {s.service: s for s in manifest_after.services}

    for svc in before_services & after_services:
        before_scopes = set(before_map[svc].allowed_scopes)
        after_scopes = set(after_map[svc].allowed_scopes)

        added_scopes = after_scopes - before_scopes
        removed_scopes = before_scopes - after_scopes

        if added_scopes:
            diff["changes"].append({
                "type": "scopes_added",
                "service": svc,
                "scopes": sorted(added_scopes),
            })

        if removed_scopes:
            diff["changes"].append({
                "type": "scopes_removed",
                "service": svc,
                "scopes": sorted(removed_scopes),
            })

        # Rate limit changes
        if before_map[svc].rate_limit_total != after_map[svc].rate_limit_total:
            diff["changes"].append({
                "type": "rate_limit_changed",
                "service": svc,
                "before": before_map[svc].rate_limit_total,
                "after": after_map[svc].rate_limit_total,
            })

    # Time constraint changes
    if manifest_before.time_constraints != manifest_after.time_constraints:
        diff["changes"].append({
            "type": "time_constraints_changed",
            "before": manifest_before.time_constraints,
            "after": manifest_after.time_constraints,
        })

    # IP constraint changes
    if manifest_before.ip_constraints != manifest_after.ip_constraints:
        diff["changes"].append({
            "type": "ip_constraints_changed",
            "before": manifest_before.ip_constraints,
            "after": manifest_after.ip_constraints,
        })

    # Warning changes
    new_warnings = set(manifest_after.warnings) - set(manifest_before.warnings)
    resolved_warnings = set(manifest_before.warnings) - set(manifest_after.warnings)

    if new_warnings:
        diff["changes"].append({
            "type": "new_warnings",
            "warnings": sorted(new_warnings),
        })

    if resolved_warnings:
        diff["changes"].append({
            "type": "resolved_warnings",
            "warnings": sorted(resolved_warnings),
        })

    diff["total_changes"] = len(diff["changes"])
    diff["has_breaking_changes"] = any(
        c["type"] in ("service_removed", "scopes_removed", "status_change")
        for c in diff["changes"]
    )

    return diff
