"""Comprehensive tests for the agent capability discovery protocol."""

import time
import pytest
import pytest_asyncio

from src.database import (
    init_db,
    create_agent_policy,
    add_connected_service,
    AgentPolicy,
)
from src.delegation import init_delegation_tables, create_delegation
from src.policy import _rate_counters
from src.capability import (
    CapabilityManifest,
    ServiceCapability,
    check_capability,
    compute_capability_diff,
    discover_capabilities,
)


USER_ID = "user|cap-test"
OTHER_USER = "user|cap-other"
AGENT_ID = "agent-cap-1"
AGENT_ID_2 = "agent-cap-2"


def make_policy(
    agent_id=AGENT_ID,
    name="CapBot",
    services=None,
    scopes=None,
    user_id=USER_ID,
    is_active=True,
    rate_limit=60,
    allowed_hours=None,
    allowed_days=None,
    expires_at=0.0,
    ip_allowlist=None,
    requires_step_up=None,
):
    return AgentPolicy(
        agent_id=agent_id,
        agent_name=name,
        allowed_services=services or ["github", "slack"],
        allowed_scopes=scopes or {
            "github": ["repo", "read:user"],
            "slack": ["chat:write", "channels:read"],
        },
        rate_limit_per_minute=rate_limit,
        requires_step_up=requires_step_up or [],
        created_by=user_id,
        created_at=time.time(),
        is_active=is_active,
        allowed_hours=allowed_hours or [],
        allowed_days=allowed_days or [],
        expires_at=expires_at,
        ip_allowlist=ip_allowlist or [],
    )


@pytest_asyncio.fixture
async def cap_db(db, monkeypatch):
    """Set up DB for capability tests."""
    monkeypatch.setattr("src.delegation.DB_PATH", db)
    await init_delegation_tables()
    # Clear rate counters between tests
    _rate_counters.clear()
    return db


@pytest_asyncio.fixture
async def basic_agent(cap_db):
    """Create a basic agent with github and slack access."""
    policy = make_policy()
    await create_agent_policy(policy)
    return policy


# ============================================================
# discover_capabilities tests
# ============================================================

class TestDiscoverCapabilities:
    @pytest.mark.asyncio
    async def test_basic_discovery(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID)
        assert manifest.agent_id == AGENT_ID
        assert manifest.agent_name == "CapBot"
        assert manifest.is_active is True
        assert len(manifest.services) == 2

    @pytest.mark.asyncio
    async def test_unregistered_agent(self, cap_db):
        manifest = await discover_capabilities("nonexistent")
        assert manifest.is_active is False
        assert "not registered" in manifest.warnings[0].lower()
        assert manifest.services == []

    @pytest.mark.asyncio
    async def test_disabled_agent(self, cap_db):
        await create_agent_policy(make_policy(is_active=False))
        manifest = await discover_capabilities(AGENT_ID)
        assert manifest.is_active is False
        assert any("disabled" in w.lower() for w in manifest.warnings)

    @pytest.mark.asyncio
    async def test_service_capabilities(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID)
        svc_names = [s.service for s in manifest.services]
        assert "github" in svc_names
        assert "slack" in svc_names

    @pytest.mark.asyncio
    async def test_scopes_per_service(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID)
        github_svc = next(s for s in manifest.services if s.service == "github")
        assert "repo" in github_svc.allowed_scopes
        assert "read:user" in github_svc.allowed_scopes

    @pytest.mark.asyncio
    async def test_rate_limit_info(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID)
        github_svc = next(s for s in manifest.services if s.service == "github")
        assert github_svc.rate_limit_total == 60
        assert github_svc.rate_limit_remaining == 60

    @pytest.mark.asyncio
    async def test_rate_limit_remaining_decreases(self, basic_agent):
        key = f"{AGENT_ID}:github"
        now = time.time()
        _rate_counters[key] = [now - i for i in range(10)]

        manifest = await discover_capabilities(AGENT_ID)
        github_svc = next(s for s in manifest.services if s.service == "github")
        assert github_svc.rate_limit_remaining == 50

    @pytest.mark.asyncio
    async def test_rate_limit_exhausted_warning(self, basic_agent):
        key = f"{AGENT_ID}:github"
        now = time.time()
        _rate_counters[key] = [now - i * 0.5 for i in range(60)]

        manifest = await discover_capabilities(AGENT_ID)
        assert any("exhausted" in w.lower() and "github" in w.lower()
                    for w in manifest.warnings)

    @pytest.mark.asyncio
    async def test_rate_limit_nearly_exhausted_warning(self, basic_agent):
        key = f"{AGENT_ID}:github"
        now = time.time()
        _rate_counters[key] = [now - i * 0.5 for i in range(56)]  # 56/60 used -> 4 remaining < 10%

        manifest = await discover_capabilities(AGENT_ID)
        assert any("nearly exhausted" in w.lower() for w in manifest.warnings)

    @pytest.mark.asyncio
    async def test_step_up_required_flag(self, cap_db):
        policy = make_policy(requires_step_up=["github"])
        await create_agent_policy(policy)

        manifest = await discover_capabilities(AGENT_ID)
        github_svc = next(s for s in manifest.services if s.service == "github")
        assert github_svc.requires_step_up is True
        slack_svc = next(s for s in manifest.services if s.service == "slack")
        assert slack_svc.requires_step_up is False

    @pytest.mark.asyncio
    async def test_step_up_in_constraints(self, cap_db):
        policy = make_policy(requires_step_up=["github"])
        await create_agent_policy(policy)

        manifest = await discover_capabilities(AGENT_ID)
        github_svc = next(s for s in manifest.services if s.service == "github")
        assert github_svc.constraints.get("requires_step_up") is True

    @pytest.mark.asyncio
    async def test_expired_policy(self, cap_db):
        policy = make_policy(expires_at=time.time() - 3600)
        await create_agent_policy(policy)

        manifest = await discover_capabilities(AGENT_ID)
        assert manifest.is_active is False
        assert any("expired" in w.lower() for w in manifest.warnings)

    @pytest.mark.asyncio
    async def test_expiring_soon_warning(self, cap_db):
        policy = make_policy(expires_at=time.time() + 1800)  # 30 min
        await create_agent_policy(policy)

        manifest = await discover_capabilities(AGENT_ID)
        assert manifest.is_active is True
        assert any("less than 1 hour" in w.lower() for w in manifest.warnings)

    @pytest.mark.asyncio
    async def test_expiring_within_day_warning(self, cap_db):
        policy = make_policy(expires_at=time.time() + 43200)  # 12 hours
        await create_agent_policy(policy)

        manifest = await discover_capabilities(AGENT_ID)
        assert any("less than 1 day" in w.lower() for w in manifest.warnings)

    @pytest.mark.asyncio
    async def test_no_expiration_no_warning(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID)
        assert not any("expir" in w.lower() for w in manifest.warnings)

    @pytest.mark.asyncio
    async def test_time_constraints_reported(self, cap_db):
        policy = make_policy(allowed_hours=[9, 10, 11, 12], allowed_days=[0, 1, 2, 3, 4])
        await create_agent_policy(policy)

        manifest = await discover_capabilities(AGENT_ID)
        assert manifest.time_constraints["allowed_hours"] == [9, 10, 11, 12]
        assert manifest.time_constraints["allowed_days"] == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_no_time_constraints(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID)
        assert manifest.time_constraints == {}

    @pytest.mark.asyncio
    async def test_ip_constraints_reported(self, cap_db):
        policy = make_policy(ip_allowlist=["10.0.0.0/24", "192.168.1.1"])
        await create_agent_policy(policy)

        manifest = await discover_capabilities(AGENT_ID)
        assert manifest.ip_constraints["ip_count"] == 2
        assert "10.0.0.0/24" in manifest.ip_constraints["ip_allowlist"]

    @pytest.mark.asyncio
    async def test_ip_not_in_allowlist_warning(self, cap_db):
        policy = make_policy(ip_allowlist=["10.0.0.1"])
        await create_agent_policy(policy)

        manifest = await discover_capabilities(AGENT_ID, ip_address="192.168.1.1")
        assert any("not in allowlist" in w.lower() for w in manifest.warnings)

    @pytest.mark.asyncio
    async def test_ip_in_allowlist_no_warning(self, cap_db):
        policy = make_policy(ip_allowlist=["10.0.0.1"])
        await create_agent_policy(policy)

        manifest = await discover_capabilities(AGENT_ID, ip_address="10.0.0.1")
        assert not any("allowlist" in w.lower() for w in manifest.warnings)

    @pytest.mark.asyncio
    async def test_connected_services_check(self, basic_agent):
        await add_connected_service(USER_ID, "github")

        manifest = await discover_capabilities(AGENT_ID, user_id=USER_ID)
        github_svc = next(s for s in manifest.services if s.service == "github")
        assert github_svc.connected is True

        slack_svc = next(s for s in manifest.services if s.service == "slack")
        assert slack_svc.connected is False
        assert slack_svc.constraints.get("not_connected") is True

    @pytest.mark.asyncio
    async def test_skip_connected_services_check(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID, include_connected=False)
        github_svc = next(s for s in manifest.services if s.service == "github")
        # Without user_id, connected should be false and no constraint added
        assert github_svc.connected is False

    @pytest.mark.asyncio
    async def test_delegation_info_included(self, cap_db):
        parent = make_policy("parent-agent", "Parent")
        await create_agent_policy(parent)
        child = make_policy("child-agent", "Child")
        await create_agent_policy(child)

        await create_delegation(
            USER_ID, "parent-agent", "child-agent",
            ["github"], {"github": ["repo"]},
        )

        manifest = await discover_capabilities("child-agent", include_delegation=True)
        assert manifest.delegation_info is not None
        assert manifest.delegation_info["is_delegated"] is True
        assert manifest.delegation_info["parent_agent_id"] == "parent-agent"

    @pytest.mark.asyncio
    async def test_no_delegation_info(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID, include_delegation=True)
        assert manifest.delegation_info is None

    @pytest.mark.asyncio
    async def test_skip_delegation_info(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID, include_delegation=False)
        assert manifest.delegation_info is None

    @pytest.mark.asyncio
    async def test_generated_at_timestamp(self, basic_agent):
        before = time.time()
        manifest = await discover_capabilities(AGENT_ID)
        after = time.time()
        assert before <= manifest.generated_at <= after

    @pytest.mark.asyncio
    async def test_single_service_agent(self, cap_db):
        policy = make_policy(
            services=["github"],
            scopes={"github": ["repo"]},
        )
        await create_agent_policy(policy)

        manifest = await discover_capabilities(AGENT_ID)
        assert len(manifest.services) == 1
        assert manifest.services[0].service == "github"

    @pytest.mark.asyncio
    async def test_many_services_agent(self, cap_db):
        services = ["github", "slack", "google", "linear", "notion"]
        scopes = {s: ["read"] for s in services}
        policy = make_policy(services=services, scopes=scopes)
        await create_agent_policy(policy)

        manifest = await discover_capabilities(AGENT_ID)
        assert len(manifest.services) == 5

    @pytest.mark.asyncio
    async def test_manifest_services_available_when_active(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID)
        for svc in manifest.services:
            assert svc.available is True


# ============================================================
# CapabilityManifest.to_dict() tests
# ============================================================

class TestManifestToDict:
    @pytest.mark.asyncio
    async def test_to_dict_structure(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID)
        d = manifest.to_dict()
        assert "agent_id" in d
        assert "agent_name" in d
        assert "services" in d
        assert "warnings" in d
        assert "service_count" in d
        assert "available_service_count" in d

    @pytest.mark.asyncio
    async def test_to_dict_service_structure(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID)
        d = manifest.to_dict()
        svc = d["services"][0]
        assert "service" in svc
        assert "available" in svc
        assert "allowed_scopes" in svc
        assert "rate_limit" in svc
        assert "remaining" in svc["rate_limit"]
        assert "total" in svc["rate_limit"]
        assert "window_seconds" in svc["rate_limit"]

    @pytest.mark.asyncio
    async def test_to_dict_counts(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID)
        d = manifest.to_dict()
        assert d["service_count"] == 2
        assert d["available_service_count"] == 2

    @pytest.mark.asyncio
    async def test_to_dict_policy_expires_none(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID)
        d = manifest.to_dict()
        assert d["policy_expires_at"] is None
        assert d["policy_expires_in"] is None

    @pytest.mark.asyncio
    async def test_to_dict_policy_expires_set(self, cap_db):
        expires = time.time() + 3600
        policy = make_policy(expires_at=expires)
        await create_agent_policy(policy)

        manifest = await discover_capabilities(AGENT_ID)
        d = manifest.to_dict()
        assert d["policy_expires_at"] == expires
        assert d["policy_expires_in"] is not None
        assert 3500 < d["policy_expires_in"] < 3700

    @pytest.mark.asyncio
    async def test_to_dict_delegation_info(self, cap_db):
        parent = make_policy("parent-d", "Parent")
        await create_agent_policy(parent)
        child = make_policy("child-d", "Child")
        await create_agent_policy(child)

        await create_delegation(
            USER_ID, "parent-d", "child-d",
            ["github"], {"github": ["repo"]},
        )

        manifest = await discover_capabilities("child-d")
        d = manifest.to_dict()
        assert d["delegation_info"] is not None
        assert d["delegation_info"]["is_delegated"] is True


# ============================================================
# check_capability tests
# ============================================================

class TestCheckCapability:
    @pytest.mark.asyncio
    async def test_basic_access_check(self, basic_agent):
        result = await check_capability(AGENT_ID, "github")
        assert result["can_access"] is True
        assert result["agent_id"] == AGENT_ID

    @pytest.mark.asyncio
    async def test_unregistered_agent(self, cap_db):
        result = await check_capability("nonexistent", "github")
        assert result["can_access"] is False
        assert "not registered" in result["reasons"][0].lower()

    @pytest.mark.asyncio
    async def test_disabled_agent(self, cap_db):
        await create_agent_policy(make_policy(is_active=False))
        result = await check_capability(AGENT_ID, "github")
        assert result["can_access"] is False
        assert "disabled" in result["reasons"][0].lower()

    @pytest.mark.asyncio
    async def test_unauthorized_service(self, basic_agent):
        result = await check_capability(AGENT_ID, "google")
        assert result["can_access"] is False
        assert "not authorized" in result["reasons"][0].lower()

    @pytest.mark.asyncio
    async def test_scope_check_passes(self, basic_agent):
        result = await check_capability(AGENT_ID, "github", scopes=["repo"])
        assert result["can_access"] is True
        assert "repo" in result["effective_scopes"]

    @pytest.mark.asyncio
    async def test_scope_check_fails(self, basic_agent):
        result = await check_capability(AGENT_ID, "github", scopes=["admin:org"])
        assert result["can_access"] is False
        assert "not permitted" in result["reasons"][0].lower()

    @pytest.mark.asyncio
    async def test_partial_scope_match(self, basic_agent):
        result = await check_capability(
            AGENT_ID, "github", scopes=["repo", "admin:org"],
        )
        assert result["can_access"] is False

    @pytest.mark.asyncio
    async def test_expired_policy(self, cap_db):
        await create_agent_policy(make_policy(expires_at=time.time() - 100))
        result = await check_capability(AGENT_ID, "github")
        assert result["can_access"] is False
        assert "expired" in result["reasons"][0].lower()

    @pytest.mark.asyncio
    async def test_wrong_owner(self, basic_agent):
        result = await check_capability(AGENT_ID, "github", user_id=OTHER_USER)
        assert result["can_access"] is False
        assert "does not own" in result["reasons"][0].lower()

    @pytest.mark.asyncio
    async def test_correct_owner(self, basic_agent):
        result = await check_capability(AGENT_ID, "github", user_id=USER_ID)
        assert result["can_access"] is True

    @pytest.mark.asyncio
    async def test_no_user_id_skips_ownership(self, basic_agent):
        result = await check_capability(AGENT_ID, "github")
        assert result["can_access"] is True

    @pytest.mark.asyncio
    async def test_ip_check_fails(self, cap_db):
        await create_agent_policy(make_policy(ip_allowlist=["10.0.0.1"]))
        result = await check_capability(AGENT_ID, "github", ip_address="192.168.1.1")
        assert result["can_access"] is False
        assert "not in allowlist" in result["reasons"][0].lower()

    @pytest.mark.asyncio
    async def test_ip_check_passes(self, cap_db):
        await create_agent_policy(make_policy(ip_allowlist=["10.0.0.1"]))
        result = await check_capability(AGENT_ID, "github", ip_address="10.0.0.1")
        assert result["can_access"] is True

    @pytest.mark.asyncio
    async def test_rate_limit_check(self, basic_agent):
        key = f"{AGENT_ID}:github"
        now = time.time()
        _rate_counters[key] = [now - i * 0.5 for i in range(60)]

        result = await check_capability(AGENT_ID, "github")
        assert result["can_access"] is False
        assert "rate limit" in result["reasons"][0].lower()

    @pytest.mark.asyncio
    async def test_rate_limit_remaining_reported(self, basic_agent):
        result = await check_capability(AGENT_ID, "github")
        assert result["rate_limit_remaining"] == 60

    @pytest.mark.asyncio
    async def test_step_up_reported(self, cap_db):
        await create_agent_policy(make_policy(requires_step_up=["github"]))
        result = await check_capability(AGENT_ID, "github")
        assert result["can_access"] is True
        assert result["requires_step_up"] is True

    @pytest.mark.asyncio
    async def test_no_step_up(self, basic_agent):
        result = await check_capability(AGENT_ID, "github")
        assert result["requires_step_up"] is False

    @pytest.mark.asyncio
    async def test_checked_at_timestamp(self, basic_agent):
        before = time.time()
        result = await check_capability(AGENT_ID, "github")
        after = time.time()
        assert before <= result["checked_at"] <= after


# ============================================================
# compute_capability_diff tests
# ============================================================

class TestComputeCapabilityDiff:
    def test_no_changes(self):
        m1 = CapabilityManifest(
            agent_id=AGENT_ID,
            is_active=True,
            services=[
                ServiceCapability(service="github", allowed_scopes=["repo"],
                                  rate_limit_total=60),
            ],
        )
        m2 = CapabilityManifest(
            agent_id=AGENT_ID,
            is_active=True,
            services=[
                ServiceCapability(service="github", allowed_scopes=["repo"],
                                  rate_limit_total=60),
            ],
        )
        diff = compute_capability_diff(m1, m2)
        assert diff["total_changes"] == 0
        assert diff["has_breaking_changes"] is False

    def test_status_change(self):
        m1 = CapabilityManifest(agent_id=AGENT_ID, is_active=True)
        m2 = CapabilityManifest(agent_id=AGENT_ID, is_active=False)
        diff = compute_capability_diff(m1, m2)
        assert diff["total_changes"] >= 1
        assert any(c["type"] == "status_change" for c in diff["changes"])
        assert diff["has_breaking_changes"] is True

    def test_service_added(self):
        m1 = CapabilityManifest(
            agent_id=AGENT_ID,
            services=[ServiceCapability(service="github")],
        )
        m2 = CapabilityManifest(
            agent_id=AGENT_ID,
            services=[
                ServiceCapability(service="github"),
                ServiceCapability(service="slack"),
            ],
        )
        diff = compute_capability_diff(m1, m2)
        added = [c for c in diff["changes"] if c["type"] == "service_added"]
        assert len(added) == 1
        assert added[0]["service"] == "slack"
        assert diff["has_breaking_changes"] is False

    def test_service_removed(self):
        m1 = CapabilityManifest(
            agent_id=AGENT_ID,
            services=[
                ServiceCapability(service="github"),
                ServiceCapability(service="slack"),
            ],
        )
        m2 = CapabilityManifest(
            agent_id=AGENT_ID,
            services=[ServiceCapability(service="github")],
        )
        diff = compute_capability_diff(m1, m2)
        removed = [c for c in diff["changes"] if c["type"] == "service_removed"]
        assert len(removed) == 1
        assert removed[0]["service"] == "slack"
        assert diff["has_breaking_changes"] is True

    def test_scopes_added(self):
        m1 = CapabilityManifest(
            agent_id=AGENT_ID,
            services=[ServiceCapability(service="github", allowed_scopes=["repo"])],
        )
        m2 = CapabilityManifest(
            agent_id=AGENT_ID,
            services=[ServiceCapability(service="github",
                                        allowed_scopes=["repo", "read:user"])],
        )
        diff = compute_capability_diff(m1, m2)
        added = [c for c in diff["changes"] if c["type"] == "scopes_added"]
        assert len(added) == 1
        assert "read:user" in added[0]["scopes"]

    def test_scopes_removed(self):
        m1 = CapabilityManifest(
            agent_id=AGENT_ID,
            services=[ServiceCapability(service="github",
                                        allowed_scopes=["repo", "read:user"])],
        )
        m2 = CapabilityManifest(
            agent_id=AGENT_ID,
            services=[ServiceCapability(service="github",
                                        allowed_scopes=["repo"])],
        )
        diff = compute_capability_diff(m1, m2)
        removed = [c for c in diff["changes"] if c["type"] == "scopes_removed"]
        assert len(removed) == 1
        assert "read:user" in removed[0]["scopes"]
        assert diff["has_breaking_changes"] is True

    def test_rate_limit_changed(self):
        m1 = CapabilityManifest(
            agent_id=AGENT_ID,
            services=[ServiceCapability(service="github", rate_limit_total=60)],
        )
        m2 = CapabilityManifest(
            agent_id=AGENT_ID,
            services=[ServiceCapability(service="github", rate_limit_total=120)],
        )
        diff = compute_capability_diff(m1, m2)
        rl = [c for c in diff["changes"] if c["type"] == "rate_limit_changed"]
        assert len(rl) == 1
        assert rl[0]["before"] == 60
        assert rl[0]["after"] == 120

    def test_time_constraints_changed(self):
        m1 = CapabilityManifest(
            agent_id=AGENT_ID,
            time_constraints={"allowed_hours": [9, 10, 11]},
        )
        m2 = CapabilityManifest(
            agent_id=AGENT_ID,
            time_constraints={"allowed_hours": [9, 10, 11, 12]},
        )
        diff = compute_capability_diff(m1, m2)
        tc = [c for c in diff["changes"] if c["type"] == "time_constraints_changed"]
        assert len(tc) == 1

    def test_ip_constraints_changed(self):
        m1 = CapabilityManifest(
            agent_id=AGENT_ID,
            ip_constraints={"ip_allowlist": ["10.0.0.1"]},
        )
        m2 = CapabilityManifest(
            agent_id=AGENT_ID,
            ip_constraints={"ip_allowlist": ["10.0.0.1", "10.0.0.2"]},
        )
        diff = compute_capability_diff(m1, m2)
        ic = [c for c in diff["changes"] if c["type"] == "ip_constraints_changed"]
        assert len(ic) == 1

    def test_new_warnings(self):
        m1 = CapabilityManifest(agent_id=AGENT_ID, warnings=[])
        m2 = CapabilityManifest(agent_id=AGENT_ID, warnings=["Rate limit nearly exhausted"])
        diff = compute_capability_diff(m1, m2)
        nw = [c for c in diff["changes"] if c["type"] == "new_warnings"]
        assert len(nw) == 1

    def test_resolved_warnings(self):
        m1 = CapabilityManifest(
            agent_id=AGENT_ID,
            warnings=["Rate limit nearly exhausted"],
        )
        m2 = CapabilityManifest(agent_id=AGENT_ID, warnings=[])
        diff = compute_capability_diff(m1, m2)
        rw = [c for c in diff["changes"] if c["type"] == "resolved_warnings"]
        assert len(rw) == 1

    def test_multiple_changes(self):
        m1 = CapabilityManifest(
            agent_id=AGENT_ID,
            is_active=True,
            services=[
                ServiceCapability(service="github", allowed_scopes=["repo"],
                                  rate_limit_total=60),
            ],
        )
        m2 = CapabilityManifest(
            agent_id=AGENT_ID,
            is_active=False,
            services=[
                ServiceCapability(service="github", allowed_scopes=["repo", "read:user"],
                                  rate_limit_total=120),
                ServiceCapability(service="slack"),
            ],
            warnings=["Agent disabled"],
        )
        diff = compute_capability_diff(m1, m2)
        assert diff["total_changes"] >= 4
        assert diff["has_breaking_changes"] is True

    def test_diff_agent_id(self):
        m1 = CapabilityManifest(agent_id=AGENT_ID)
        m2 = CapabilityManifest(agent_id=AGENT_ID)
        diff = compute_capability_diff(m1, m2)
        assert diff["agent_id"] == AGENT_ID

    def test_empty_manifests(self):
        m1 = CapabilityManifest(agent_id=AGENT_ID)
        m2 = CapabilityManifest(agent_id=AGENT_ID)
        diff = compute_capability_diff(m1, m2)
        assert diff["total_changes"] == 0

    def test_service_swap(self):
        m1 = CapabilityManifest(
            agent_id=AGENT_ID,
            services=[ServiceCapability(service="github")],
        )
        m2 = CapabilityManifest(
            agent_id=AGENT_ID,
            services=[ServiceCapability(service="slack")],
        )
        diff = compute_capability_diff(m1, m2)
        added = [c for c in diff["changes"] if c["type"] == "service_added"]
        removed = [c for c in diff["changes"] if c["type"] == "service_removed"]
        assert len(added) == 1
        assert added[0]["service"] == "slack"
        assert len(removed) == 1
        assert removed[0]["service"] == "github"


# ============================================================
# ServiceCapability dataclass tests
# ============================================================

class TestServiceCapability:
    def test_defaults(self):
        cap = ServiceCapability(service="github")
        assert cap.service == "github"
        assert cap.available is True
        assert cap.allowed_scopes == []
        assert cap.requires_step_up is False
        assert cap.connected is False
        assert cap.rate_limit_remaining == 0
        assert cap.rate_limit_total == 0
        assert cap.constraints == {}

    def test_with_all_fields(self):
        cap = ServiceCapability(
            service="github",
            available=True,
            allowed_scopes=["repo", "read:user"],
            requires_step_up=True,
            connected=True,
            rate_limit_remaining=50,
            rate_limit_total=60,
            constraints={"requires_step_up": True},
        )
        assert cap.allowed_scopes == ["repo", "read:user"]
        assert cap.rate_limit_remaining == 50


# ============================================================
# CapabilityManifest dataclass tests
# ============================================================

class TestCapabilityManifestDataclass:
    def test_defaults(self):
        m = CapabilityManifest(agent_id=AGENT_ID)
        assert m.agent_id == AGENT_ID
        assert m.agent_name == ""
        assert m.is_active is True
        assert m.services == []
        assert m.delegation_info is None
        assert m.time_constraints == {}
        assert m.ip_constraints == {}
        assert m.policy_expires_at == 0.0
        assert m.current_time_allowed is True
        assert m.generated_at == 0.0
        assert m.warnings == []

    def test_to_dict_empty(self):
        m = CapabilityManifest(agent_id=AGENT_ID)
        d = m.to_dict()
        assert d["agent_id"] == AGENT_ID
        assert d["service_count"] == 0
        assert d["available_service_count"] == 0
        assert d["warnings"] == []


# ============================================================
# Integration tests
# ============================================================

class TestCapabilityIntegration:
    @pytest.mark.asyncio
    async def test_discover_then_check(self, basic_agent):
        manifest = await discover_capabilities(AGENT_ID)
        assert manifest.is_active is True

        for svc in manifest.services:
            result = await check_capability(AGENT_ID, svc.service)
            assert result["can_access"] is True

    @pytest.mark.asyncio
    async def test_discover_disabled_then_check(self, cap_db):
        await create_agent_policy(make_policy(is_active=False))
        manifest = await discover_capabilities(AGENT_ID)
        assert manifest.is_active is False

        result = await check_capability(AGENT_ID, "github")
        assert result["can_access"] is False

    @pytest.mark.asyncio
    async def test_diff_after_policy_change(self, cap_db):
        policy1 = make_policy(services=["github"], scopes={"github": ["repo"]})
        await create_agent_policy(policy1)
        m1 = await discover_capabilities(AGENT_ID)

        policy2 = make_policy(
            services=["github", "slack"],
            scopes={"github": ["repo", "read:user"], "slack": ["chat:write"]},
        )
        await create_agent_policy(policy2)
        m2 = await discover_capabilities(AGENT_ID)

        diff = compute_capability_diff(m1, m2)
        assert diff["total_changes"] > 0
        assert any(c["type"] == "service_added" and c["service"] == "slack"
                    for c in diff["changes"])
        assert any(c["type"] == "scopes_added" for c in diff["changes"])

    @pytest.mark.asyncio
    async def test_multiple_agents_independent(self, cap_db):
        await create_agent_policy(make_policy(
            AGENT_ID, "Agent1", ["github"], {"github": ["repo"]},
        ))
        await create_agent_policy(make_policy(
            AGENT_ID_2, "Agent2", ["slack"], {"slack": ["chat:write"]},
        ))

        m1 = await discover_capabilities(AGENT_ID)
        m2 = await discover_capabilities(AGENT_ID_2)

        assert len(m1.services) == 1
        assert m1.services[0].service == "github"
        assert len(m2.services) == 1
        assert m2.services[0].service == "slack"

    @pytest.mark.asyncio
    async def test_capability_with_all_constraints(self, cap_db):
        policy = make_policy(
            expires_at=time.time() + 3600,
            allowed_hours=list(range(9, 18)),
            allowed_days=[0, 1, 2, 3, 4],
            ip_allowlist=["10.0.0.0/8"],
            requires_step_up=["github"],
            rate_limit=30,
        )
        await create_agent_policy(policy)

        manifest = await discover_capabilities(AGENT_ID, ip_address="10.0.0.1")
        assert manifest.time_constraints != {}
        assert manifest.ip_constraints != {}
        assert manifest.policy_expires_at > 0

        github_svc = next(s for s in manifest.services if s.service == "github")
        assert github_svc.requires_step_up is True
        assert github_svc.rate_limit_total == 30
