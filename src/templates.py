"""Policy templates for common agent authorization profiles.

Pre-built security profiles that encode best practices for common use cases.
Users can apply a template and customize it instead of building policies from scratch.
"""

import time
from dataclasses import dataclass, field

from .database import AgentPolicy


@dataclass
class PolicyTemplate:
    """A reusable policy template with default settings."""
    id: str = ""
    name: str = ""
    description: str = ""
    category: str = ""  # "security", "productivity", "integration", "custom"
    allowed_services: list[str] = field(default_factory=list)
    allowed_scopes: dict[str, list[str]] = field(default_factory=dict)
    rate_limit_per_minute: int = 60
    requires_step_up: list[str] = field(default_factory=list)
    allowed_hours: list[int] = field(default_factory=list)
    allowed_days: list[int] = field(default_factory=list)
    expires_in_seconds: int = 0  # 0 = never, positive = relative from creation
    ip_allowlist: list[str] = field(default_factory=list)
    risk_level: str = "medium"  # "minimal", "low", "medium", "high"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "allowed_services": self.allowed_services,
            "allowed_scopes": self.allowed_scopes,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "requires_step_up": self.requires_step_up,
            "allowed_hours": self.allowed_hours,
            "allowed_days": self.allowed_days,
            "expires_in_seconds": self.expires_in_seconds,
            "risk_level": self.risk_level,
            "tags": self.tags,
        }


# --- Built-in Templates ---

BUILTIN_TEMPLATES: dict[str, PolicyTemplate] = {}


def _register(template: PolicyTemplate) -> PolicyTemplate:
    BUILTIN_TEMPLATES[template.id] = template
    return template


# Minimal read-only access
_register(PolicyTemplate(
    id="read-only",
    name="Read-Only Observer",
    description="Minimal access for monitoring agents. Read-only scopes only, "
                "low rate limit, step-up required for all services.",
    category="security",
    allowed_services=["github", "slack", "google"],
    allowed_scopes={
        "github": ["repo:read"],
        "slack": ["channels:read", "users:read"],
        "google": ["gmail.readonly", "drive.readonly"],
    },
    rate_limit_per_minute=10,
    requires_step_up=["github", "slack", "google"],
    risk_level="minimal",
    tags=["read-only", "monitoring", "low-risk"],
))

# Standard development agent
_register(PolicyTemplate(
    id="dev-standard",
    name="Standard Developer Agent",
    description="Balanced access for code review and CI/CD agents. "
                "Read/write to GitHub, read-only Slack, business hours only.",
    category="productivity",
    allowed_services=["github", "slack"],
    allowed_scopes={
        "github": ["repo:read", "repo:write", "pull_request:read", "pull_request:write"],
        "slack": ["channels:read", "chat:write"],
    },
    rate_limit_per_minute=30,
    requires_step_up=["github"],
    allowed_hours=list(range(8, 20)),  # 8 AM - 7 PM UTC
    allowed_days=[0, 1, 2, 3, 4],  # Mon-Fri
    risk_level="medium",
    tags=["development", "ci-cd", "code-review"],
))

# Time-boxed contractor
_register(PolicyTemplate(
    id="contractor-limited",
    name="Time-Boxed Contractor",
    description="Temporary access that auto-expires after 30 days. "
                "Limited services, strict rate limits, business hours only.",
    category="security",
    allowed_services=["github"],
    allowed_scopes={
        "github": ["repo:read", "pull_request:read", "pull_request:write"],
    },
    rate_limit_per_minute=15,
    requires_step_up=["github"],
    allowed_hours=list(range(9, 18)),  # 9 AM - 5 PM UTC
    allowed_days=[0, 1, 2, 3, 4],  # Mon-Fri
    expires_in_seconds=30 * 24 * 3600,  # 30 days
    risk_level="low",
    tags=["temporary", "contractor", "time-limited"],
))

# Full-access admin agent
_register(PolicyTemplate(
    id="admin-full",
    name="Admin Full Access",
    description="Full access to all services with all scopes. "
                "Step-up required for write operations. Use sparingly.",
    category="integration",
    allowed_services=["github", "slack", "google"],
    allowed_scopes={
        "github": ["repo:read", "repo:write", "pull_request:read",
                    "pull_request:write", "admin:org"],
        "slack": ["channels:read", "channels:write", "chat:write",
                  "users:read", "files:write"],
        "google": ["gmail.readonly", "gmail.send", "drive.readonly",
                   "drive.file", "calendar.readonly", "calendar.events"],
    },
    rate_limit_per_minute=120,
    requires_step_up=["github", "slack", "google"],
    risk_level="high",
    tags=["admin", "full-access", "privileged"],
))

# Slack-only notification agent
_register(PolicyTemplate(
    id="slack-notifier",
    name="Slack Notification Agent",
    description="Write-only Slack access for sending notifications. "
                "Cannot read channels or user data.",
    category="integration",
    allowed_services=["slack"],
    allowed_scopes={
        "slack": ["chat:write"],
    },
    rate_limit_per_minute=20,
    requires_step_up=[],
    risk_level="low",
    tags=["slack", "notifications", "write-only"],
))

# GitHub CI/CD pipeline
_register(PolicyTemplate(
    id="ci-cd-pipeline",
    name="CI/CD Pipeline Agent",
    description="Automated pipeline agent with repo read/write. "
                "No step-up (automated), higher rate limit for batch operations.",
    category="productivity",
    allowed_services=["github"],
    allowed_scopes={
        "github": ["repo:read", "repo:write", "pull_request:read",
                    "pull_request:write"],
    },
    rate_limit_per_minute=100,
    requires_step_up=[],
    risk_level="medium",
    tags=["ci-cd", "automated", "pipeline"],
))

# Email assistant
_register(PolicyTemplate(
    id="email-assistant",
    name="Email Assistant",
    description="Read and draft emails, manage calendar. "
                "Step-up required for sending emails.",
    category="productivity",
    allowed_services=["google"],
    allowed_scopes={
        "google": ["gmail.readonly", "gmail.send", "calendar.readonly",
                   "calendar.events"],
    },
    rate_limit_per_minute=30,
    requires_step_up=["google"],
    risk_level="medium",
    tags=["email", "calendar", "assistant"],
))

# Paranoid mode - maximum restrictions
_register(PolicyTemplate(
    id="paranoid",
    name="Paranoid Mode",
    description="Maximum restrictions. Every operation requires step-up auth. "
                "Ultra-low rate limit. Business hours on weekdays only. "
                "Expires in 7 days.",
    category="security",
    allowed_services=["github"],
    allowed_scopes={
        "github": ["repo:read"],
    },
    rate_limit_per_minute=5,
    requires_step_up=["github"],
    allowed_hours=list(range(10, 16)),  # 10 AM - 3 PM UTC
    allowed_days=[0, 1, 2, 3, 4],  # Mon-Fri
    expires_in_seconds=7 * 24 * 3600,  # 7 days
    risk_level="minimal",
    tags=["paranoid", "maximum-security", "short-lived"],
))


def list_templates(
    category: str = "",
    risk_level: str = "",
    tag: str = "",
) -> list[PolicyTemplate]:
    """List available policy templates with optional filters."""
    templates = list(BUILTIN_TEMPLATES.values())

    if category:
        templates = [t for t in templates if t.category == category]
    if risk_level:
        templates = [t for t in templates if t.risk_level == risk_level]
    if tag:
        templates = [t for t in templates if tag in t.tags]

    return templates


def get_template(template_id: str) -> PolicyTemplate | None:
    """Get a specific template by ID."""
    return BUILTIN_TEMPLATES.get(template_id)


def apply_template(
    template_id: str,
    agent_id: str,
    agent_name: str,
    user_id: str,
    overrides: dict | None = None,
) -> AgentPolicy | None:
    """Create an AgentPolicy from a template with optional overrides.

    Overrides can include any AgentPolicy field to customize the template.
    Returns None if the template ID is not found.
    """
    template = get_template(template_id)
    if not template:
        return None

    now = time.time()
    expires_at = 0.0
    if template.expires_in_seconds > 0:
        expires_at = now + template.expires_in_seconds

    policy = AgentPolicy(
        agent_id=agent_id,
        agent_name=agent_name,
        allowed_services=list(template.allowed_services),
        allowed_scopes={k: list(v) for k, v in template.allowed_scopes.items()},
        rate_limit_per_minute=template.rate_limit_per_minute,
        requires_step_up=list(template.requires_step_up),
        allowed_hours=list(template.allowed_hours),
        allowed_days=list(template.allowed_days),
        expires_at=expires_at,
        ip_allowlist=list(template.ip_allowlist),
        created_by=user_id,
        created_at=now,
        is_active=True,
    )

    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            if hasattr(policy, key) and key not in ("agent_id", "created_by", "created_at"):
                setattr(policy, key, value)

    return policy


def preview_template(template_id: str) -> dict | None:
    """Preview what a policy would look like from a template.

    Returns a detailed breakdown without creating anything.
    """
    template = get_template(template_id)
    if not template:
        return None

    return {
        "template": template.to_dict(),
        "policy_preview": {
            "allowed_services": template.allowed_services,
            "allowed_scopes": template.allowed_scopes,
            "rate_limit_per_minute": template.rate_limit_per_minute,
            "requires_step_up": template.requires_step_up,
            "time_restrictions": {
                "hours": template.allowed_hours or "always",
                "days": template.allowed_days or "always",
            },
            "auto_expires": template.expires_in_seconds > 0,
            "expires_in_human": _format_duration(template.expires_in_seconds)
            if template.expires_in_seconds > 0 else "never",
        },
        "risk_assessment": {
            "level": template.risk_level,
            "has_step_up": bool(template.requires_step_up),
            "has_time_restrictions": bool(template.allowed_hours or template.allowed_days),
            "has_ip_restrictions": bool(template.ip_allowlist),
            "auto_expires": template.expires_in_seconds > 0,
            "total_services": len(template.allowed_services),
            "total_scopes": sum(len(v) for v in template.allowed_scopes.values()),
        },
    }


def compare_templates(template_id_a: str, template_id_b: str) -> dict | None:
    """Compare two templates side by side.

    Returns differences in permissions, restrictions, and risk levels.
    """
    a = get_template(template_id_a)
    b = get_template(template_id_b)
    if not a or not b:
        return None

    a_scopes = set()
    for svc_scopes in a.allowed_scopes.values():
        a_scopes.update(svc_scopes)
    b_scopes = set()
    for svc_scopes in b.allowed_scopes.values():
        b_scopes.update(svc_scopes)

    return {
        "template_a": {"id": a.id, "name": a.name},
        "template_b": {"id": b.id, "name": b.name},
        "differences": {
            "services": {
                "only_in_a": list(set(a.allowed_services) - set(b.allowed_services)),
                "only_in_b": list(set(b.allowed_services) - set(a.allowed_services)),
                "common": list(set(a.allowed_services) & set(b.allowed_services)),
            },
            "scopes": {
                "only_in_a": sorted(a_scopes - b_scopes),
                "only_in_b": sorted(b_scopes - a_scopes),
            },
            "rate_limit": {
                "a": a.rate_limit_per_minute,
                "b": b.rate_limit_per_minute,
            },
            "risk_level": {
                "a": a.risk_level,
                "b": b.risk_level,
            },
            "step_up_services": {
                "a": a.requires_step_up,
                "b": b.requires_step_up,
            },
            "time_restricted": {
                "a": bool(a.allowed_hours or a.allowed_days),
                "b": bool(b.allowed_hours or b.allowed_days),
            },
            "auto_expires": {
                "a": a.expires_in_seconds > 0,
                "b": b.expires_in_seconds > 0,
            },
        },
    }


def _format_duration(seconds: int) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds <= 0:
        return "never"
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    if days > 0:
        return f"{days}d {hours}h" if hours else f"{days}d"
    if hours > 0:
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    minutes = seconds // 60
    return f"{minutes}m" if minutes else f"{seconds}s"
