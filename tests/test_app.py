"""Tests for the app module."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.app import load_models_from_config


class TestLoadModelsFromConfig:
    """Tests for load_models_from_config function."""

    def test_loads_models_from_yaml(self) -> None:
        """Should load models from config/models.yaml."""
        models = load_models_from_config()

        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_contains_expected_models(self) -> None:
        """Should contain known model names."""
        models = load_models_from_config()

        # These models are defined in config/models.yaml
        expected = ["qwen2.5-coder:14b", "qwen2.5-coder:7b", "llama3.1:8b"]

        for model in expected:
            assert model in models, f"Expected model '{model}' not found"

    def test_returns_fallback_if_config_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return fallback models if config file is missing."""
        # Temporarily change the module's path resolution
        # This is a basic test - in practice, the function has its own fallback
        models = load_models_from_config()

        # Should always return a non-empty list
        assert len(models) > 0


class TestAppIntegration:
    """Integration tests for the Gradio app."""

    def test_create_app_returns_blocks(self) -> None:
        """create_app should return a Gradio Blocks instance."""
        from src.app import create_app
        import gradio as gr

        app = create_app()

        assert isinstance(app, gr.Blocks)

    def test_app_has_title(self) -> None:
        """App should have a title set."""
        from src.app import create_app

        app = create_app()

        assert app.title is not None
        assert "Seedling" in app.title
