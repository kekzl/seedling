# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Seedling is an open-source synthetic instruction dataset generator for supervised fine-tuning (SFT) with local LLMs. It provides a Gradio web UI for generating, curating, and exporting instruction-response pairs using local models via Ollama.

## Commands

```bash
# Install (development mode)
pip install -e .

# Run application
python -m src.app

# Run tests
pytest tests/ -v
pytest tests/ -v -k "test_name"              # single test
pytest tests/ -v -m "not slow"               # exclude slow tests
pytest tests/ -v --cov=src --cov-report=html # with coverage

# Code quality
ruff check src/ tests/
mypy src/

# Docker (full stack with Ollama, Argilla, Elasticsearch)
docker compose up -d
docker compose down
```

## Architecture

```
src/
├── app.py         # Gradio web UI (5 tabs: Templates, Seeds, Generation, Review, Export)
├── generator.py   # InstructionGenerator using Distilabel pipelines
├── domains.py     # 29+ pre-built domain templates with seed instructions
├── roles.py       # Dynamic role generation and management
├── exporter.py    # Export to JSONL, HF Datasets, Alpaca, ShareGPT
├── hardware.py    # GPU detection and automatic model selection

config/
├── models.yaml    # Model configurations with VRAM requirements
├── roles.yaml     # Role generation templates
├── generation.yaml# Generation parameters
```

**Key flows:**
- Generation uses async/await with Distilabel pipelines (Self-Instruct, Evol-Instruct, Magpie methods)
- Long-running generation spawns ThreadPoolExecutor; progress communicated via Queue to Gradio UI
- Hardware detection: OS-aware VRAM calculation (WSL2 has 3GB overhead per GPU, Windows 2GB for DWM)
- Roles can be predefined or AI-generated from profession names with caching in `.cache/roles/`

**Ports:**
- 7860: Seedling Web UI
- 11434: Ollama
- 6900: Argilla
- 9200: Elasticsearch

## Test Markers

- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Requires external services (Ollama, Argilla)

## Code Style

- Python 3.11+, strict type hints required (`mypy --disallow_untyped_defs`)
- 100-char line length
- Use `ruff` for linting/formatting
