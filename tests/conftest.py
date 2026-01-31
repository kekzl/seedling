"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest


@pytest.fixture
def sample_data() -> list[dict[str, Any]]:
    """Sample instruction-response data for testing."""
    return [
        {
            "instruction": "Schreibe ein Bash-Skript das alle Docker Container stoppt",
            "response": "#!/bin/bash\ndocker stop $(docker ps -q)",
            "method": "self_instruct",
            "model": "qwen2.5-coder:14b",
        },
        {
            "instruction": "Erstelle eine Terraform-Konfiguration für einen S3 Bucket",
            "response": 'resource "aws_s3_bucket" "example" {\n  bucket = "my-bucket"\n}',
            "method": "evol_instruct",
            "model": "qwen2.5-coder:14b",
        },
        {
            "instruction": "Write a Python function to validate email addresses",
            "response": "import re\n\ndef validate_email(email):\n    pattern = r'^[\\w.-]+@[\\w.-]+\\.\\w+$'\n    return bool(re.match(pattern, email))",
            "method": "magpie",
            "model": "llama3.1:8b",
        },
    ]


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_seeds() -> list[str]:
    """Sample seed instructions for testing."""
    return [
        "Schreibe ein Bash-Skript das alle Docker Container stoppt",
        "Erstelle eine Terraform-Konfiguration für einen S3 Bucket",
        "Schreibe ein Kubernetes Deployment für eine Node.js Anwendung",
        "Erstelle eine GitHub Actions Pipeline die Tests ausführt",
        "Schreibe ein Helm Chart für eine Redis Installation",
    ]


@pytest.fixture(autouse=True)
def set_test_env() -> Generator[None, None, None]:
    """Set test environment variables."""
    original_env = os.environ.copy()

    # Set test defaults
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    os.environ.setdefault("ARGILLA_API_URL", "http://localhost:6900")
    os.environ.setdefault("ARGILLA_API_KEY", "test-api-key")

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
