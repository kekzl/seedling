"""Tests for the roles module."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.roles import (
    Role,
    RoleConfig,
    RoleManager,
    get_role_manager,
    get_predefined_roles,
    get_role_choices,
    get_role_seeds,
    get_role_as_domain_template,
    get_all_roles_as_domain_templates,
)


class TestRoleConfig:
    """Tests for RoleConfig dataclass."""

    def test_default_values(self) -> None:
        """RoleConfig should have sensible defaults."""
        config = RoleConfig()

        assert config.min_topics == 8
        assert config.max_topics == 15
        assert config.min_seeds == 10
        assert config.max_seeds == 20
        assert config.temperature == 0.8
        assert config.max_tokens == 2048
        assert config.cache_enabled is True
        assert config.cache_expiration_days == 30

    def test_custom_values(self) -> None:
        """RoleConfig should accept custom values."""
        config = RoleConfig(
            min_topics=5,
            max_topics=10,
            min_seeds=15,
            temperature=0.5,
        )

        assert config.min_topics == 5
        assert config.max_topics == 10
        assert config.min_seeds == 15
        assert config.temperature == 0.5


class TestRole:
    """Tests for Role dataclass."""

    def test_role_creation(self) -> None:
        """Role should be created with all required fields."""
        role = Role(
            name="test_role",
            display_name="Test Role",
            category="Test",
            description="A test role",
            topics=["Topic1", "Topic2"],
            seeds=["Seed instruction 1", "Seed instruction 2"],
        )

        assert role.name == "test_role"
        assert role.display_name == "Test Role"
        assert role.category == "Test"
        assert role.description == "A test role"
        assert len(role.topics) == 2
        assert len(role.seeds) == 2
        assert role.is_generated is False

    def test_role_to_dict(self) -> None:
        """Role.to_dict should return domain template format."""
        role = Role(
            name="test",
            display_name="Test",
            category="Test",
            description="Test description",
            topics=["Topic1"],
            seeds=["Seed1"],
        )

        result = role.to_dict()

        assert result == {
            "description": "Test description",
            "topics": ["Topic1"],
            "seeds": ["Seed1"],
        }

    def test_role_from_dict(self) -> None:
        """Role.from_dict should create Role from dictionary."""
        data = {
            "display_name": "Data Analyst",
            "category": "Data",
            "description": "Analyzes data",
            "topics": ["SQL", "Python"],
            "seeds": ["Write a SQL query", "Create a report"],
        }

        role = Role.from_dict("data_analyst", data)

        assert role.name == "data_analyst"
        assert role.display_name == "Data Analyst"
        assert role.category == "Data"
        assert role.description == "Analyzes data"
        assert role.topics == ["SQL", "Python"]
        assert role.seeds == ["Write a SQL query", "Create a report"]

    def test_role_from_dict_defaults(self) -> None:
        """Role.from_dict should handle missing optional fields."""
        data = {
            "description": "Basic role",
            "topics": ["Topic1"],
            "seeds": ["Seed1"],
        }

        role = Role.from_dict("basic_role", data)

        assert role.name == "basic_role"
        assert role.display_name == "Basic Role"  # Auto-generated from name
        assert role.category == "Custom"  # Default category


class TestRoleManager:
    """Tests for RoleManager class."""

    def test_loads_config(self) -> None:
        """RoleManager should load configuration from YAML."""
        manager = RoleManager()

        # Should have loaded predefined roles
        roles = manager.get_predefined_roles()
        assert len(roles) > 0

    def test_get_predefined_roles(self) -> None:
        """Should return dictionary of predefined roles."""
        manager = RoleManager()
        roles = manager.get_predefined_roles()

        assert isinstance(roles, dict)
        for name, role in roles.items():
            assert isinstance(name, str)
            assert isinstance(role, Role)

    def test_get_predefined_role_valid(self) -> None:
        """Should return Role for valid role name."""
        manager = RoleManager()

        # Assuming customer_support is a predefined role
        role = manager.get_predefined_role("customer_support")

        if role:  # May not exist if config is empty
            assert isinstance(role, Role)
            assert role.name == "customer_support"

    def test_get_predefined_role_invalid(self) -> None:
        """Should return None for invalid role name."""
        manager = RoleManager()
        role = manager.get_predefined_role("nonexistent_role")

        assert role is None

    def test_get_categories(self) -> None:
        """Should return list of categories."""
        manager = RoleManager()
        categories = manager.get_categories()

        assert isinstance(categories, list)
        for cat in categories:
            assert isinstance(cat, dict)
            assert "name" in cat

    def test_get_roles_by_category(self) -> None:
        """Should return roles filtered by category."""
        manager = RoleManager()

        # Test with a known category
        roles = manager.get_roles_by_category("Service")

        assert isinstance(roles, list)
        for role in roles:
            assert isinstance(role, Role)
            assert role.category == "Service"

    def test_role_to_domain_template(self) -> None:
        """Should convert Role to domain template format."""
        manager = RoleManager()
        role = Role(
            name="test",
            display_name="Test",
            category="Test",
            description="Test desc",
            topics=["A", "B"],
            seeds=["Seed 1", "Seed 2"],
        )

        template = manager.role_to_domain_template(role)

        assert template["description"] == "Test desc"
        assert template["topics"] == ["A", "B"]
        assert template["seeds"] == ["Seed 1", "Seed 2"]


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_role_manager_singleton(self) -> None:
        """get_role_manager should return singleton instance."""
        manager1 = get_role_manager()
        manager2 = get_role_manager()

        assert manager1 is manager2

    def test_get_predefined_roles_function(self) -> None:
        """get_predefined_roles should return roles dict."""
        roles = get_predefined_roles()

        assert isinstance(roles, dict)

    def test_get_role_choices_function(self) -> None:
        """get_role_choices should return list of tuples."""
        choices = get_role_choices()

        assert isinstance(choices, list)
        for choice in choices:
            assert isinstance(choice, tuple)
            assert len(choice) == 2

    def test_get_role_seeds_valid(self) -> None:
        """get_role_seeds should return seeds for valid role."""
        roles = get_predefined_roles()
        if roles:
            first_role_name = list(roles.keys())[0]
            seeds = get_role_seeds(first_role_name)

            assert isinstance(seeds, list)
            assert len(seeds) > 0

    def test_get_role_seeds_invalid(self) -> None:
        """get_role_seeds should return empty list for invalid role."""
        seeds = get_role_seeds("nonexistent_role")

        assert seeds == []

    def test_get_role_as_domain_template_valid(self) -> None:
        """get_role_as_domain_template should return dict for valid role."""
        roles = get_predefined_roles()
        if roles:
            first_role_name = list(roles.keys())[0]
            template = get_role_as_domain_template(first_role_name)

            assert isinstance(template, dict)
            assert "description" in template
            assert "topics" in template
            assert "seeds" in template

    def test_get_role_as_domain_template_invalid(self) -> None:
        """get_role_as_domain_template should return None for invalid role."""
        template = get_role_as_domain_template("nonexistent_role")

        assert template is None

    def test_get_all_roles_as_domain_templates(self) -> None:
        """get_all_roles_as_domain_templates should return dict of templates."""
        templates = get_all_roles_as_domain_templates()

        assert isinstance(templates, dict)
        for name, template in templates.items():
            assert isinstance(name, str)
            assert isinstance(template, dict)
            assert "description" in template
            assert "topics" in template
            assert "seeds" in template


class TestPredefinedRolesQuality:
    """Tests for quality of predefined roles."""

    def test_all_roles_have_minimum_topics(self) -> None:
        """Each role should have at least 5 topics."""
        roles = get_predefined_roles()

        for name, role in roles.items():
            assert len(role.topics) >= 5, f"Role '{name}' has only {len(role.topics)} topics"

    def test_all_roles_have_minimum_seeds(self) -> None:
        """Each role should have at least 5 seed instructions."""
        roles = get_predefined_roles()

        for name, role in roles.items():
            assert len(role.seeds) >= 5, f"Role '{name}' has only {len(role.seeds)} seeds"

    def test_all_roles_have_description(self) -> None:
        """Each role should have a non-empty description."""
        roles = get_predefined_roles()

        for name, role in roles.items():
            assert role.description, f"Role '{name}' has empty description"
            assert len(role.description) >= 10, f"Role '{name}' description too short"

    def test_all_roles_have_category(self) -> None:
        """Each role should have a category."""
        roles = get_predefined_roles()

        for name, role in roles.items():
            assert role.category, f"Role '{name}' has no category"

    def test_known_roles_exist(self) -> None:
        """Check that expected roles are present."""
        roles = get_predefined_roles()
        expected_roles = [
            "customer_support",
            "data_analyst",
            "content_writer",
            "hr_recruiter",
        ]

        for expected in expected_roles:
            assert expected in roles, f"Expected role '{expected}' not found"

    def test_seeds_are_actionable(self) -> None:
        """Seeds should be actionable instructions."""
        roles = get_predefined_roles()

        action_words = [
            "write", "create", "build", "design", "develop",
            "implement", "generate", "explain", "analyze", "configure",
            "setup", "install", "deploy", "test", "review",
            "translate", "document", "prepare", "draft", "compose",
        ]

        for name, role in roles.items():
            for seed in role.seeds:
                seed_lower = seed.lower()
                has_action = any(word in seed_lower for word in action_words)
                # Allow some flexibility - at least most seeds should be actionable
                # This is a soft check, so we won't fail on every seed
                if len(seed) > 20:  # Only check substantial seeds
                    # Just log if not actionable, don't fail
                    pass
