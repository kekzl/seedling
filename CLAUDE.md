# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Seedling is an open-source synthetic instruction dataset generator for supervised fine-tuning (SFT) with local LLMs. It provides a Gradio web UI for generating and exporting instruction-response pairs using local models via Ollama.

## Commands

```bash
# Install (development mode)
pip install -e .

# Run application
python -m src.app

# Run tests (79 tests)
pytest tests/ -v
pytest tests/ -v -k "test_name"              # single test
pytest tests/ -v --cov=src --cov-report=html # with coverage

# Code quality
ruff check src/ tests/
ruff format src/
mypy src/

# Docker (Ollama + Seedling)
docker compose up -d
docker compose down
docker compose up -d --build seedling  # rebuild after code changes
```

## Architecture

```
src/
├── app.py         # Gradio web UI (Quick Start + Advanced modes)
├── generator.py   # InstructionGenerator (Distilabel) + SimpleGenerator (fallback)
├── domains.py     # 25+ pre-built domain templates with seed instructions
├── roles.py       # 24 professional roles with full instruction sets
├── exporter.py    # Export to JSONL, HF Datasets, Alpaca, ShareGPT
├── hardware.py    # GPU detection and automatic model selection

config/
├── models.yaml    # Model configurations with VRAM requirements
├── roles.yaml     # Role definitions (persona, system_prompt, constraints, examples)

assets/
├── icon.svg       # Seedling app icon
```

**Key flows:**
- Generation uses async/await with Distilabel pipelines (Self-Instruct, Evol-Instruct, Magpie)
- SimpleGenerator provides fallback using direct Ollama API when Distilabel fails
- Long-running generation spawns ThreadPoolExecutor; progress communicated via Queue to Gradio UI
- Hardware detection: OS-aware VRAM calculation (WSL2 has 3GB overhead per GPU, Windows 2GB for DWM)
- Roles can be predefined (24 built-in) or AI-generated from profession names

**Ports:**
- 7860: Seedling Web UI
- 11434: Ollama

## Key Components

### Generators
- `InstructionGenerator` - Uses Distilabel pipelines for sophisticated generation
- `SimpleGenerator` - Direct Ollama API fallback, more reliable but simpler

### Roles (config/roles.yaml)
Each role has:
- `persona` - tone, expertise_level, traits
- `system_prompt` - detailed instructions for the LLM
- `response_guidelines` - format, length, code examples
- `constraints` - forbidden actions, cautions, knowledge limits
- `examples` - sample instruction-response pairs

### Domains (src/domains.py)
25+ technical domains with:
- Description
- Topics list
- 15 seed instructions each

## Test Markers

- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Requires external services (Ollama)

## Code Style

- Python 3.11+, strict type hints required
- 100-char line length
- Use `ruff` for linting/formatting
- Use `ruff format` before committing

## Common Tasks

### Adding a new role
1. Edit `config/roles.yaml`
2. Add role under appropriate category
3. Include: name, display_name, description, topics, seeds, persona, system_prompt, response_guidelines, constraints, examples

### Adding a new domain
1. Edit `src/domains.py`
2. Add to `DOMAIN_TEMPLATES` dict
3. Include: description, topics, seeds (15 recommended)

### Debugging generation issues
1. Check Docker logs: `docker logs seedling-app --tail 50`
2. If Distilabel fails, SimpleGenerator fallback kicks in automatically
3. Test Ollama directly: `curl http://localhost:11434/api/generate -d '{"model":"qwen2.5:7b","prompt":"test"}'`
