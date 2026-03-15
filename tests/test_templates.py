"""Comprehensive tests for policy templates system."""

import time
import pytest

from src.database import AgentPolicy
from src.templates import (
    BUILTIN_TEMPLATES,
    PolicyTemplate,
    apply_template,
    compare_templates,
    get_template,
    list_templates,
    preview_template,
    _format_duration,
)


USER_ID = "auth0|user123"


class TestTemplateRegistry:
    def test_builtin_templates_exist(self):
        assert len(BUILTIN_TEMPLATES) >= 8

    def test_all_templates_have_ids(self):
        for tid, t in BUILTIN_TEMPLATES.items():
            assert t.id == tid
            assert t.id != ""

    def test_all_templates_have_names(self):
        for t in BUILTIN_TEMPLATES.values():
            assert t.name != ""

    def test_all_templates_have_descriptions(self):
        for t in BUILTIN_TEMPLATES.values():
            assert t.description != ""

    def test_all_templates_have_categories(self):
        valid_categories = {"security", "productivity", "integration", "custom"}
        for t in BUILTIN_TEMPLATES.values():
            assert t.category in valid_categories

    def test_all_templates_have_risk_levels(self):
        valid_levels = {"minimal", "low", "medium", "high"}
        for t in BUILTIN_TEMPLATES.values():
            assert t.risk_level in valid_levels

    def test_all_templates_have_services(self):
        for t in BUILTIN_TEMPLATES.values():
            assert len(t.allowed_services) > 0

    def test_all_templates_have_matching_scopes(self):
        """Every service in allowed_services should have scopes defined."""
        for t in BUILTIN_TEMPLATES.values():
            for svc in t.allowed_services:
                assert svc in t.allowed_scopes, f"Template {t.id}: {svc} missing scopes"
                assert len(t.allowed_scopes[svc]) > 0

    def test_all_templates_have_tags(self):
        for t in BUILTIN_TEMPLATES.values():
            assert len(t.tags) > 0

    def test_unique_template_ids(self):
        ids = list(BUILTIN_TEMPLATES.keys())
        assert len(ids) == len(set(ids))


class TestSpecificTemplates:
    def test_read_only_template(self):
        t = get_template("read-only")
        assert t is not None
        assert t.risk_level == "minimal"
        assert t.rate_limit_per_minute == 10
        assert len(t.requires_step_up) == 3  # All services require step-up
        for scopes in t.allowed_scopes.values():
            for scope in scopes:
                assert "write" not in scope.lower() or "readonly" in scope.lower()

    def test_dev_standard_template(self):
        t = get_template("dev-standard")
        assert t is not None
        assert t.risk_level == "medium"
        assert len(t.allowed_hours) > 0  # Has time restrictions
        assert len(t.allowed_days) == 5  # Mon-Fri
        assert 5 not in t.allowed_days  # No Saturday
        assert 6 not in t.allowed_days  # No Sunday

    def test_contractor_limited_template(self):
        t = get_template("contractor-limited")
        assert t is not None
        assert t.expires_in_seconds > 0  # Auto-expires
        assert t.expires_in_seconds == 30 * 24 * 3600  # 30 days

    def test_admin_full_template(self):
        t = get_template("admin-full")
        assert t is not None
        assert t.risk_level == "high"
        assert t.rate_limit_per_minute >= 100
        assert len(t.allowed_services) >= 3

    def test_slack_notifier_template(self):
        t = get_template("slack-notifier")
        assert t is not None
        assert t.allowed_services == ["slack"]
        assert t.allowed_scopes["slack"] == ["chat:write"]
        assert t.requires_step_up == []

    def test_ci_cd_pipeline_template(self):
        t = get_template("ci-cd-pipeline")
        assert t is not None
        assert t.requires_step_up == []  # Automated, no step-up
        assert t.rate_limit_per_minute >= 100

    def test_email_assistant_template(self):
        t = get_template("email-assistant")
        assert t is not None
        assert "google" in t.allowed_services
        assert "gmail.send" in t.allowed_scopes["google"]

    def test_paranoid_template(self):
        t = get_template("paranoid")
        assert t is not None
        assert t.risk_level == "minimal"
        assert t.rate_limit_per_minute <= 5
        assert t.expires_in_seconds == 7 * 24 * 3600  # 7 days
        assert len(t.allowed_hours) < 12  # Restricted hours


class TestListTemplates:
    def test_list_all(self):
        result = list_templates()
        assert len(result) == len(BUILTIN_TEMPLATES)

    def test_filter_by_category_security(self):
        result = list_templates(category="security")
        assert all(t.category == "security" for t in result)
        assert len(result) >= 2

    def test_filter_by_category_productivity(self):
        result = list_templates(category="productivity")
        assert all(t.category == "productivity" for t in result)

    def test_filter_by_category_integration(self):
        result = list_templates(category="integration")
        assert all(t.category == "integration" for t in result)

    def test_filter_by_risk_minimal(self):
        result = list_templates(risk_level="minimal")
        assert all(t.risk_level == "minimal" for t in result)

    def test_filter_by_risk_high(self):
        result = list_templates(risk_level="high")
        assert all(t.risk_level == "high" for t in result)

    def test_filter_by_tag(self):
        result = list_templates(tag="ci-cd")
        assert all("ci-cd" in t.tags for t in result)

    def test_filter_no_match(self):
        result = list_templates(category="nonexistent")
        assert result == []

    def test_filter_combined(self):
        result = list_templates(category="security", risk_level="minimal")
        assert all(t.category == "security" and t.risk_level == "minimal" for t in result)


class TestGetTemplate:
    def test_get_existing(self):
        t = get_template("read-only")
        assert t is not None
        assert t.id == "read-only"

    def test_get_nonexistent(self):
        assert get_template("nonexistent") is None

    def test_get_empty_string(self):
        assert get_template("") is None


class TestApplyTemplate:
    def test_apply_basic(self):
        policy = apply_template("read-only", "agent-1", "My Agent", USER_ID)
        assert policy is not None
        assert isinstance(policy, AgentPolicy)
        assert policy.agent_id == "agent-1"
        assert policy.agent_name == "My Agent"
        assert policy.created_by == USER_ID
        assert policy.is_active is True
        assert policy.rate_limit_per_minute == 10

    def test_apply_sets_created_at(self):
        before = time.time()
        policy = apply_template("read-only", "a1", "Agent", USER_ID)
        after = time.time()
        assert before <= policy.created_at <= after

    def test_apply_with_expiration(self):
        policy = apply_template("contractor-limited", "a1", "Contractor", USER_ID)
        assert policy.expires_at > 0
        assert policy.expires_at > time.time()

    def test_apply_no_expiration(self):
        policy = apply_template("read-only", "a1", "Agent", USER_ID)
        assert policy.expires_at == 0

    def test_apply_copies_services(self):
        t = get_template("admin-full")
        policy = apply_template("admin-full", "a1", "Admin", USER_ID)
        assert policy.allowed_services == t.allowed_services
        # Ensure it's a copy, not a reference
        policy.allowed_services.append("extra")
        assert "extra" not in t.allowed_services

    def test_apply_copies_scopes(self):
        t = get_template("dev-standard")
        policy = apply_template("dev-standard", "a1", "Dev", USER_ID)
        policy.allowed_scopes["github"].append("extra")
        assert "extra" not in t.allowed_scopes["github"]

    def test_apply_with_override_rate_limit(self):
        policy = apply_template(
            "read-only", "a1", "Agent", USER_ID,
            overrides={"rate_limit_per_minute": 99},
        )
        assert policy.rate_limit_per_minute == 99

    def test_apply_with_override_services(self):
        policy = apply_template(
            "read-only", "a1", "Agent", USER_ID,
            overrides={"allowed_services": ["github"]},
        )
        assert policy.allowed_services == ["github"]

    def test_apply_with_override_ip_allowlist(self):
        policy = apply_template(
            "read-only", "a1", "Agent", USER_ID,
            overrides={"ip_allowlist": ["10.0.0.0/24"]},
        )
        assert policy.ip_allowlist == ["10.0.0.0/24"]

    def test_apply_override_cannot_change_agent_id(self):
        policy = apply_template(
            "read-only", "a1", "Agent", USER_ID,
            overrides={"agent_id": "overridden"},
        )
        assert policy.agent_id == "a1"  # Not overridden

    def test_apply_override_cannot_change_created_by(self):
        policy = apply_template(
            "read-only", "a1", "Agent", USER_ID,
            overrides={"created_by": "hacker"},
        )
        assert policy.created_by == USER_ID

    def test_apply_nonexistent_template(self):
        assert apply_template("fake", "a1", "Agent", USER_ID) is None

    def test_apply_with_empty_overrides(self):
        policy = apply_template("read-only", "a1", "Agent", USER_ID, overrides={})
        assert policy is not None

    def test_apply_with_none_overrides(self):
        policy = apply_template("read-only", "a1", "Agent", USER_ID, overrides=None)
        assert policy is not None


class TestPreviewTemplate:
    def test_preview_structure(self):
        result = preview_template("read-only")
        assert result is not None
        assert "template" in result
        assert "policy_preview" in result
        assert "risk_assessment" in result

    def test_preview_template_section(self):
        result = preview_template("read-only")
        t = result["template"]
        assert t["id"] == "read-only"
        assert "name" in t
        assert "description" in t

    def test_preview_policy_section(self):
        result = preview_template("dev-standard")
        p = result["policy_preview"]
        assert "allowed_services" in p
        assert "allowed_scopes" in p
        assert "rate_limit_per_minute" in p
        assert "time_restrictions" in p

    def test_preview_risk_assessment(self):
        result = preview_template("admin-full")
        r = result["risk_assessment"]
        assert r["level"] == "high"
        assert r["has_step_up"] is True
        assert r["total_services"] >= 3
        assert r["total_scopes"] > 0

    def test_preview_expiring_template(self):
        result = preview_template("contractor-limited")
        p = result["policy_preview"]
        assert p["auto_expires"] is True
        assert "30d" in p["expires_in_human"]

    def test_preview_non_expiring(self):
        result = preview_template("read-only")
        p = result["policy_preview"]
        assert p["auto_expires"] is False
        assert p["expires_in_human"] == "never"

    def test_preview_nonexistent(self):
        assert preview_template("fake") is None


class TestCompareTemplates:
    def test_compare_basic(self):
        result = compare_templates("read-only", "admin-full")
        assert result is not None
        assert result["template_a"]["id"] == "read-only"
        assert result["template_b"]["id"] == "admin-full"

    def test_compare_services_diff(self):
        result = compare_templates("slack-notifier", "admin-full")
        diff = result["differences"]["services"]
        assert "common" in diff
        assert "only_in_a" in diff
        assert "only_in_b" in diff

    def test_compare_rate_limits(self):
        result = compare_templates("paranoid", "admin-full")
        rates = result["differences"]["rate_limit"]
        assert rates["a"] < rates["b"]

    def test_compare_risk_levels(self):
        result = compare_templates("read-only", "admin-full")
        risk = result["differences"]["risk_level"]
        assert risk["a"] == "minimal"
        assert risk["b"] == "high"

    def test_compare_same_template(self):
        result = compare_templates("read-only", "read-only")
        assert result is not None
        diff = result["differences"]
        assert diff["services"]["only_in_a"] == []
        assert diff["services"]["only_in_b"] == []

    def test_compare_nonexistent_a(self):
        assert compare_templates("fake", "read-only") is None

    def test_compare_nonexistent_b(self):
        assert compare_templates("read-only", "fake") is None

    def test_compare_step_up(self):
        result = compare_templates("slack-notifier", "read-only")
        step_up = result["differences"]["step_up_services"]
        assert step_up["a"] == []  # slack-notifier has no step-up
        assert len(step_up["b"]) > 0  # read-only requires step-up

    def test_compare_time_restrictions(self):
        result = compare_templates("read-only", "dev-standard")
        time_r = result["differences"]["time_restricted"]
        assert time_r["a"] is False  # read-only has no time restrictions
        assert time_r["b"] is True  # dev-standard has business hours

    def test_compare_auto_expires(self):
        result = compare_templates("read-only", "contractor-limited")
        expires = result["differences"]["auto_expires"]
        assert expires["a"] is False
        assert expires["b"] is True


class TestFormatDuration:
    def test_zero(self):
        assert _format_duration(0) == "never"

    def test_negative(self):
        assert _format_duration(-1) == "never"

    def test_seconds(self):
        assert _format_duration(45) == "45s"

    def test_minutes(self):
        assert _format_duration(300) == "5m"

    def test_hours(self):
        assert _format_duration(3600) == "1h"

    def test_hours_and_minutes(self):
        assert _format_duration(5400) == "1h 30m"

    def test_days(self):
        assert _format_duration(86400) == "1d"

    def test_days_and_hours(self):
        assert _format_duration(90000) == "1d 1h"

    def test_30_days(self):
        assert _format_duration(30 * 86400) == "30d"

    def test_7_days(self):
        assert _format_duration(7 * 86400) == "7d"


class TestPolicyTemplateDataclass:
    def test_default_values(self):
        t = PolicyTemplate()
        assert t.id == ""
        assert t.allowed_services == []
        assert t.rate_limit_per_minute == 60
        assert t.risk_level == "medium"
        assert t.expires_in_seconds == 0

    def test_to_dict(self):
        t = PolicyTemplate(id="test", name="Test", category="security")
        d = t.to_dict()
        assert d["id"] == "test"
        assert d["name"] == "Test"
        assert "allowed_services" in d
        assert "risk_level" in d

    def test_to_dict_completeness(self):
        t = get_template("read-only")
        d = t.to_dict()
        expected_keys = {
            "id", "name", "description", "category", "allowed_services",
            "allowed_scopes", "rate_limit_per_minute", "requires_step_up",
            "allowed_hours", "allowed_days", "expires_in_seconds",
            "risk_level", "tags",
        }
        assert set(d.keys()) == expected_keys
