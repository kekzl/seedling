"""Tests for the domains module."""

from __future__ import annotations

import pytest

from src.domains import (
    DOMAIN_TEMPLATES,
    get_domain_seeds,
    get_all_topics,
    get_domain_description,
)


class TestDomainTemplates:
    """Tests for DOMAIN_TEMPLATES structure."""

    def test_domain_templates_not_empty(self) -> None:
        """Verify that DOMAIN_TEMPLATES contains domains."""
        assert len(DOMAIN_TEMPLATES) > 0

    def test_all_domains_have_required_keys(self) -> None:
        """Each domain should have description, topics, and seeds."""
        required_keys = {"description", "topics", "seeds"}

        for domain_name, template in DOMAIN_TEMPLATES.items():
            for key in required_keys:
                assert key in template, f"Domain '{domain_name}' missing key '{key}'"

    def test_all_domains_have_seeds(self) -> None:
        """Each domain should have at least 5 seed instructions."""
        for domain_name, template in DOMAIN_TEMPLATES.items():
            seeds = template.get("seeds", [])
            assert len(seeds) >= 5, f"Domain '{domain_name}' has only {len(seeds)} seeds"

    def test_all_domains_have_topics(self) -> None:
        """Each domain should have at least 3 topics."""
        for domain_name, template in DOMAIN_TEMPLATES.items():
            topics = template.get("topics", [])
            assert len(topics) >= 3, f"Domain '{domain_name}' has only {len(topics)} topics"

    def test_known_domains_exist(self) -> None:
        """Check that expected domains are present."""
        expected_domains = ["DevOps", "SysAdmin", "Cloud", "Security", "Database", "Code"]

        for domain in expected_domains:
            assert domain in DOMAIN_TEMPLATES, f"Expected domain '{domain}' not found"


class TestGetDomainSeeds:
    """Tests for get_domain_seeds function."""

    def test_get_seeds_for_valid_domain(self) -> None:
        """Should return seeds for a valid domain."""
        seeds = get_domain_seeds("DevOps")

        assert isinstance(seeds, list)
        assert len(seeds) > 0
        assert all(isinstance(s, str) for s in seeds)

    def test_get_seeds_for_invalid_domain(self) -> None:
        """Should return empty list for unknown domain."""
        seeds = get_domain_seeds("NonExistentDomain")

        assert seeds == []

    def test_seeds_are_non_empty_strings(self) -> None:
        """All seeds should be non-empty strings."""
        for domain in DOMAIN_TEMPLATES:
            seeds = get_domain_seeds(domain)
            for seed in seeds:
                assert isinstance(seed, str)
                assert len(seed.strip()) > 0


class TestGetAllTopics:
    """Tests for get_all_topics function."""

    def test_returns_list_of_strings(self) -> None:
        """Should return a list of string topics."""
        topics = get_all_topics()

        assert isinstance(topics, list)
        assert all(isinstance(t, str) for t in topics)

    def test_topics_are_sorted(self) -> None:
        """Topics should be returned in sorted order."""
        topics = get_all_topics()

        assert topics == sorted(topics)

    def test_topics_are_unique(self) -> None:
        """All topics should be unique."""
        topics = get_all_topics()

        assert len(topics) == len(set(topics))

    def test_contains_known_topics(self) -> None:
        """Should contain some expected topics."""
        topics = get_all_topics()
        expected = ["Docker", "Kubernetes", "Python", "AWS"]

        for expected_topic in expected:
            assert expected_topic in topics


class TestGetDomainDescription:
    """Tests for get_domain_description function."""

    def test_returns_description_for_valid_domain(self) -> None:
        """Should return description for valid domain."""
        description = get_domain_description("DevOps")

        assert isinstance(description, str)
        assert len(description) > 0

    def test_returns_empty_for_invalid_domain(self) -> None:
        """Should return empty string for unknown domain."""
        description = get_domain_description("NonExistentDomain")

        assert description == ""

    def test_all_domains_have_descriptions(self) -> None:
        """All domains should have non-empty descriptions."""
        for domain in DOMAIN_TEMPLATES:
            description = get_domain_description(domain)
            assert len(description) > 0, f"Domain '{domain}' has empty description"
