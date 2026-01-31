# Seedling

**Open-Source Synthetic Instruction Dataset Generator**

A complete stack for creating instruction-response pairs for SFT (Supervised Fine-Tuning) with local LLMs.

## Features

- **Web UI** - Gradio-based interface for easy operation
- **Local LLMs** - Ollama integration, no API key required
- **Domain Templates** - Pre-built domains (DevOps, SysAdmin, Cloud, etc.)
- **Multiple Generation Methods** - Self-Instruct, Evol-Instruct, Magpie
- **Curation** - Review and filtering with Argilla
- **Export** - JSONL, Hugging Face Datasets, Alpaca, ShareGPT format

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Seedling Web UI                         │
│                      (Gradio)                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Generation Pipeline                        │
│                    (Distilabel)                             │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │ Ollama  │   │  vLLM   │   │  HF API │
        │ (local) │   │ (local) │   │ (remote)│
        └─────────┘   └─────────┘   └─────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Curation                            │
│                     (Argilla)                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Export (JSONL / HF Datasets)                   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### With Docker Compose (recommended)

```bash
# Clone repository
git clone https://github.com/kekzl/seedling.git
cd seedling

# Configure environment
cp .env.example .env

# Start with Docker Compose
docker compose up -d

# Open Web UI
open http://localhost:7860
```

### Local Installation

```bash
# Clone repository
git clone https://github.com/kekzl/seedling.git
cd seedling

# Install dependencies
pip install -e .

# Start app
python -m src.app
```

## Requirements

- **Docker & Docker Compose** (for container deployment)
- **Python 3.11+** (for local installation)
- **NVIDIA GPU with CUDA** (for local LLMs)
- Min. 16GB VRAM for 7B models, 32GB for 14B+

## Domain Templates

Seedling comes with pre-built templates for various domains:

| Domain | Description | Example Topics |
|--------|-------------|----------------|
| **DevOps** | Infrastructure, CI/CD, Automation | Terraform, Ansible, Kubernetes, Docker |
| **SysAdmin** | Windows/Linux Administration | PowerShell, Bash, Active Directory, Intune |
| **Cloud** | AWS, Azure, GCP | IAM, Networking, Serverless |
| **Security** | InfoSec, Compliance | ISMS, Hardening, Incident Response |
| **Database** | SQL, NoSQL, Data Engineering | PostgreSQL, MongoDB, ETL |
| **Code** | General Programming | Python, TypeScript, Rust, Go |

## Usage

### 1. Select Domain
Choose one or more domains from the templates or create your own.

### 2. Create Seed Instructions
Enter 10-50 example instructions as a starting point.

### 3. Start Generation
Generate hundreds/thousands of instructions with Self-Instruct, Evol-Instruct, or Magpie.

### 4. Curation (optional)
Review and filter the data with Argilla.

### 5. Export
Export as JSONL, Hugging Face Dataset, Alpaca, or ShareGPT format.

## Generation Methods

### Self-Instruct
Generates new instructions based on seed examples. Ideal for diverse, similar instructions.

### Evol-Instruct
Evolves instructions into more complex versions. Good for more challenging training data.

### Magpie
Uses LLM-specific templates for more natural prompts. Optimized for chat models.

## Configuration

See `config/` for configuration files:

- `config/models.yaml` - LLM configuration
- `config/generation.yaml` - Generation parameters

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama Server URL | `http://localhost:11434` |
| `ARGILLA_API_URL` | Argilla Server URL | `http://localhost:6900` |
| `ARGILLA_API_KEY` | Argilla API Key | `argilla.apikey` |
| `HF_TOKEN` | Hugging Face Token | - |

## Tests

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html
```

## Development

```bash
# Install dependencies (including dev tools)
pip install -e .

# Linting
ruff check src/ tests/

# Type checking
mypy src/
```

## Technology Stack

- **Gradio 4.x** - Web UI Framework
- **Distilabel 1.x** - Data Generation Pipeline
- **Ollama** - Local LLM Inference
- **Argilla 2.x** - Data Curation Platform
- **Hugging Face Datasets** - Dataset Format

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

---

Developed with the goal of making it easy to create high-quality training data for LLMs locally.
