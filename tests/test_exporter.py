"""Tests for the exporter module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.exporter import DatasetExporter


class TestDatasetExporter:
    """Tests for DatasetExporter class."""

    def test_init_creates_output_dir(self, temp_output_dir: Path) -> None:
        """Exporter should create output directory if it doesn't exist."""
        new_dir = temp_output_dir / "new_subdir"
        exporter = DatasetExporter(output_dir=str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_init_uses_existing_dir(self, temp_output_dir: Path) -> None:
        """Exporter should work with existing directory."""
        exporter = DatasetExporter(output_dir=str(temp_output_dir))

        assert exporter.output_dir == temp_output_dir


class TestExportJsonl:
    """Tests for JSONL export functionality."""

    def test_export_creates_file(
        self, temp_output_dir: Path, sample_data: list[dict[str, Any]]
    ) -> None:
        """Should create a JSONL file."""
        exporter = DatasetExporter(output_dir=str(temp_output_dir))
        filepath = exporter.export(sample_data, "jsonl", "test_dataset")

        assert Path(filepath).exists()
        assert filepath.endswith(".jsonl")

    def test_export_jsonl_format(
        self, temp_output_dir: Path, sample_data: list[dict[str, Any]]
    ) -> None:
        """JSONL file should have correct format."""
        exporter = DatasetExporter(output_dir=str(temp_output_dir))
        filepath = exporter.export(sample_data, "jsonl", "test_dataset")

        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == len(sample_data)

        for line in lines:
            data = json.loads(line)
            assert "instruction" in data
            assert "response" in data

    def test_export_jsonl_preserves_content(
        self, temp_output_dir: Path, sample_data: list[dict[str, Any]]
    ) -> None:
        """JSONL export should preserve instruction and response content."""
        exporter = DatasetExporter(output_dir=str(temp_output_dir))
        filepath = exporter.export(sample_data, "jsonl", "test_dataset")

        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data["instruction"] == sample_data[i]["instruction"]
            assert data["response"] == sample_data[i]["response"]


class TestExportAlpaca:
    """Tests for Alpaca format export."""

    def test_export_creates_file(
        self, temp_output_dir: Path, sample_data: list[dict[str, Any]]
    ) -> None:
        """Should create an Alpaca format JSON file."""
        exporter = DatasetExporter(output_dir=str(temp_output_dir))
        filepath = exporter.export(sample_data, "alpaca_format", "test_dataset")

        assert Path(filepath).exists()
        assert "_alpaca.json" in filepath

    def test_export_alpaca_format(
        self, temp_output_dir: Path, sample_data: list[dict[str, Any]]
    ) -> None:
        """Alpaca file should have correct structure."""
        exporter = DatasetExporter(output_dir=str(temp_output_dir))
        filepath = exporter.export(sample_data, "alpaca_format", "test_dataset")

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == len(sample_data)

        for item in data:
            assert "instruction" in item
            assert "input" in item
            assert "output" in item

    def test_export_alpaca_maps_response_to_output(
        self, temp_output_dir: Path, sample_data: list[dict[str, Any]]
    ) -> None:
        """Alpaca format should map 'response' to 'output'."""
        exporter = DatasetExporter(output_dir=str(temp_output_dir))
        filepath = exporter.export(sample_data, "alpaca_format", "test_dataset")

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        for i, item in enumerate(data):
            assert item["output"] == sample_data[i]["response"]


class TestExportSharegpt:
    """Tests for ShareGPT format export."""

    def test_export_creates_file(
        self, temp_output_dir: Path, sample_data: list[dict[str, Any]]
    ) -> None:
        """Should create a ShareGPT format JSON file."""
        exporter = DatasetExporter(output_dir=str(temp_output_dir))
        filepath = exporter.export(sample_data, "sharegpt_format", "test_dataset")

        assert Path(filepath).exists()
        assert "_sharegpt.json" in filepath

    def test_export_sharegpt_format(
        self, temp_output_dir: Path, sample_data: list[dict[str, Any]]
    ) -> None:
        """ShareGPT file should have conversation structure."""
        exporter = DatasetExporter(output_dir=str(temp_output_dir))
        filepath = exporter.export(sample_data, "sharegpt_format", "test_dataset")

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == len(sample_data)

        for item in data:
            assert "conversations" in item
            conversations = item["conversations"]
            assert len(conversations) >= 2

            # Check human message
            assert conversations[0]["from"] == "human"
            assert "value" in conversations[0]

            # Check GPT response
            assert conversations[1]["from"] == "gpt"
            assert "value" in conversations[1]


class TestExportHfDataset:
    """Tests for Hugging Face Dataset export."""

    def test_export_creates_directory(
        self, temp_output_dir: Path, sample_data: list[dict[str, Any]]
    ) -> None:
        """Should create a dataset directory."""
        exporter = DatasetExporter(output_dir=str(temp_output_dir))
        filepath = exporter.export(sample_data, "hugging_face_dataset", "test_dataset")

        assert Path(filepath).exists()
        assert Path(filepath).is_dir()


class TestExportErrors:
    """Tests for export error handling."""

    def test_unknown_format_raises_error(
        self, temp_output_dir: Path, sample_data: list[dict[str, Any]]
    ) -> None:
        """Should raise ValueError for unknown format."""
        exporter = DatasetExporter(output_dir=str(temp_output_dir))

        with pytest.raises(ValueError, match="Unknown format"):
            exporter.export(sample_data, "unknown_format", "test_dataset")

    def test_export_empty_data(self, temp_output_dir: Path) -> None:
        """Should handle empty data gracefully."""
        exporter = DatasetExporter(output_dir=str(temp_output_dir))
        filepath = exporter.export([], "jsonl", "empty_dataset")

        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        assert content == ""
