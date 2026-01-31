# 🌱 Seedling

**Open-source synthetic instruction dataset generator for Supervised Fine-Tuning (SFT) with local LLMs.**

Generate high-quality instruction-response pairs using your own hardware with Ollama. No API keys required.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![Tests](https://img.shields.io/badge/tests-79%20passed-brightgreen.svg)

## Features

- **100% Local** - Uses Ollama for completely local, private data generation
- **Multiple Generation Methods** - Self-Instruct, Evol-Instruct, and Magpie techniques
- **25+ Domain Templates** - Pre-built templates for DevOps, Python, Cloud, Security, and more
- **24 Professional Roles** - Data Analyst, Software Engineer, AI Agents, and more with full instruction sets
- **Hardware-Aware** - Automatic GPU detection and model recommendations based on your VRAM
- **Export Formats** - JSONL, Alpaca, ShareGPT, and Hugging Face Datasets
- **Simple Web UI** - Gradio interface with Quick Start and Advanced modes

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/kekzl/seedling.git
cd seedling

# Start the stack (Ollama + Seedling)
docker compose up -d

# Open http://localhost:7860
```

### Manual Installation

```bash
# Clone and install
git clone https://github.com/kekzl/seedling.git
cd seedling
pip install -e .

# Start Ollama (in a separate terminal)
ollama serve

# Pull a model
ollama pull qwen2.5:7b

# Run Seedling
python -m src.app
```

## Usage

### Quick Start Mode

1. Open http://localhost:7860
2. Select a topic (e.g., "Python Development", "DevOps")
3. Choose quantity (10, 25, 50, or 100 pairs)
4. Click **Generate**
5. Download in your preferred format (JSONL, Alpaca, ShareGPT)

### Advanced Mode

- Select multiple domains
- Choose specific professional roles
- Customize seed instructions
- Configure generation parameters
- Upload directly to Hugging Face Hub

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Seedling UI   │────▶│     Ollama      │
│   (Gradio)      │     │   (Local LLM)   │
│   Port 7860     │     │   Port 11434    │
└─────────────────┘     └─────────────────┘
```

### Project Structure

```
seedling/
├── src/
│   ├── app.py          # Gradio web UI
│   ├── generator.py    # Instruction generation (Distilabel + SimpleGenerator)
│   ├── domains.py      # 25+ domain templates
│   ├── roles.py        # 24 professional roles
│   ├── exporter.py     # Export to JSONL, Alpaca, ShareGPT, HF
│   └── hardware.py     # GPU detection & model recommendations
├── config/
│   ├── models.yaml     # Model configurations with VRAM requirements
│   └── roles.yaml      # Role definitions with full instruction sets
├── assets/
│   └── icon.svg        # Seedling icon
└── tests/              # Test suite (79 tests)
```

## Supported Models

Seedling automatically recommends models based on your available VRAM:

| VRAM | Recommended Models |
|------|-------------------|
| 4GB  | qwen2.5:3b, phi3:mini |
| 8GB  | qwen2.5:7b, llama3.1:8b, gemma2:9b |
| 16GB | qwen2.5:14b, deepseek-coder:16b |
| 24GB+ | qwen2.5:32b, llama3.1:70b-q4 |

## Domain Templates

Pre-built templates with seed instructions for:

| Category | Domains |
|----------|---------|
| **Programming** | Python, JavaScript, TypeScript, Rust, Go |
| **Infrastructure** | DevOps, Kubernetes, Docker, Terraform, Ansible |
| **Cloud** | AWS, Azure, GCP |
| **Data** | Data Engineering, Machine Learning, Analytics |
| **Security** | InfoSec, Compliance, Hardening |
| **Professional Roles** | 24 roles including Data Analyst, Software Engineer, AI Agents |

## Generation Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Self-Instruct** | Generates new instructions from seed examples | Diverse, similar instructions |
| **Evol-Instruct** | Evolves instructions into more complex versions | Challenging training data |
| **Magpie** | Uses LLM-specific templates for natural prompts | Chat model training |

## Export Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| JSONL | One JSON object per line | General training |
| Alpaca | `{instruction, input, output}` | Stanford Alpaca format |
| ShareGPT | Conversation format | Chat model training |
| HF Dataset | Hugging Face Datasets | Direct Hub upload |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama Server URL | `http://localhost:11434` |
| `HF_TOKEN` | Hugging Face Token (for uploads) | - |
| `GRADIO_SERVER_PORT` | Web UI port | `7860` |

### Docker Compose

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  seedling:
    build: .
    ports:
      - "7860:7860"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
```

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v

# Run linter
ruff check src/

# Format code
ruff format src/

# Type checking
mypy src/
```

## API Example

```python
from src import InstructionGenerator, DatasetExporter

# Generate instructions
generator = InstructionGenerator()
results = await generator.generate(
    seeds=["Write a Python function to sort a list"],
    model="qwen2.5:7b",
    temperature=0.7,
    max_tokens=1024,
    num_instructions=10,
    method="self_instruct",
)

# Export to JSONL
exporter = DatasetExporter(output_dir="./outputs")
filepath = exporter.export(results, format_type="jsonl", name="my_dataset")
```

## Technology Stack

- **[Gradio](https://gradio.app/)** - Web UI Framework
- **[Distilabel](https://github.com/argilla-io/distilabel)** - Data Generation Pipeline
- **[Ollama](https://ollama.ai/)** - Local LLM Inference
- **[Hugging Face Datasets](https://huggingface.co/docs/datasets)** - Dataset Format & Hub

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with 🌱 for the open-source ML community
</p>
