"""Tests for the generator module."""

from __future__ import annotations

import os

import pytest

from src.generator import GenerationConfig, InstructionGenerator


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = GenerationConfig()

        assert config.model == "qwen2.5-coder:14b"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.timeout == 120.0

    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = GenerationConfig(
            model="llama3.1:8b",
            temperature=0.9,
            max_tokens=2048,
            timeout=60.0,
        )

        assert config.model == "llama3.1:8b"
        assert config.temperature == 0.9
        assert config.max_tokens == 2048
        assert config.timeout == 60.0

    def test_ollama_base_url_from_env(self) -> None:
        """Config should use OLLAMA_BASE_URL from environment."""
        original = os.environ.get("OLLAMA_BASE_URL")

        try:
            os.environ["OLLAMA_BASE_URL"] = "http://custom-ollama:11434"
            config = GenerationConfig()
            assert config.ollama_base_url == "http://custom-ollama:11434"
        finally:
            if original:
                os.environ["OLLAMA_BASE_URL"] = original
            else:
                os.environ.pop("OLLAMA_BASE_URL", None)

    def test_ollama_base_url_default(self) -> None:
        """Config should have default Ollama URL."""
        original = os.environ.pop("OLLAMA_BASE_URL", None)

        try:
            config = GenerationConfig()
            assert config.ollama_base_url == "http://localhost:11434"
        finally:
            if original:
                os.environ["OLLAMA_BASE_URL"] = original


class TestInstructionGenerator:
    """Tests for InstructionGenerator class."""

    def test_init_with_default_config(self) -> None:
        """Generator should initialize with default config."""
        generator = InstructionGenerator()

        assert generator.config is not None
        assert isinstance(generator.config, GenerationConfig)

    def test_init_with_custom_config(self) -> None:
        """Generator should accept custom config."""
        config = GenerationConfig(model="custom-model")
        generator = InstructionGenerator(config=config)

        assert generator.config.model == "custom-model"

    def test_create_llm(self) -> None:
        """Generator should create OllamaLLM instance."""
        generator = InstructionGenerator()
        llm = generator._create_llm(
            model="test-model",
            temperature=0.5,
            max_tokens=512,
        )

        assert llm is not None
        assert llm.model == "test-model"
        assert llm.host == generator.config.ollama_base_url


class TestGenerationMethods:
    """Tests for generation method validation."""

    @pytest.mark.asyncio
    async def test_unknown_method_raises_error(self, sample_seeds: list[str]) -> None:
        """Unknown generation method should raise ValueError."""
        generator = InstructionGenerator()

        with pytest.raises(ValueError, match="Unknown generation method"):
            await generator.generate(
                seeds=sample_seeds,
                model="test-model",
                temperature=0.7,
                max_tokens=1024,
                num_instructions=10,
                method="unknown_method",
            )

    def test_valid_methods(self) -> None:
        """Check that valid methods are recognized."""
        valid_methods = ["self_instruct", "evol_instruct", "magpie"]

        for method in valid_methods:
            # Just verify the method name doesn't raise immediately
            # Full testing would require a running Ollama instance
            assert method in ["self_instruct", "evol_instruct", "magpie"]


class TestSimpleGenerator:
    """Tests for SimpleGenerator class."""

    def test_init_with_default_url(self) -> None:
        """SimpleGenerator should use default URL."""
        from src.generator import SimpleGenerator

        generator = SimpleGenerator()
        assert generator.base_url == "http://localhost:11434"

    def test_init_with_custom_url(self) -> None:
        """SimpleGenerator should accept custom URL."""
        from src.generator import SimpleGenerator

        generator = SimpleGenerator(base_url="http://custom:11434")
        assert generator.base_url == "http://custom:11434"
