# Seedling

**Open-Source Synthetic Instruction Dataset Generator**

Ein vollständiger Stack zum Erstellen von Instruction-Response-Paaren für SFT (Supervised Fine-Tuning) mit lokalen LLMs.

## Features

- **Web UI** - Gradio-basierte Oberfläche für einfache Bedienung
- **Lokale LLMs** - Ollama-Integration, kein API-Key nötig
- **Domain Templates** - Vorgefertigte Domänen (DevOps, SysAdmin, Cloud, etc.)
- **Mehrere Generierungsmethoden** - Self-Instruct, Evol-Instruct, Magpie
- **Curation** - Review und Filterung mit Argilla
- **Export** - JSONL, Hugging Face Datasets, Alpaca, ShareGPT Format

## Architektur

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

### Mit Docker Compose (empfohlen)

```bash
# Clone Repository
git clone https://github.com/kekzl/seedling.git
cd seedling

# Environment konfigurieren
cp .env.example .env

# Start mit Docker Compose
docker compose up -d

# Web UI öffnen
open http://localhost:7860
```

### Lokale Installation

```bash
# Clone Repository
git clone https://github.com/kekzl/seedling.git
cd seedling

# Dependencies installieren
pip install -e .

# App starten
python -m src.app
```

## Voraussetzungen

- **Docker & Docker Compose** (für Container-Deployment)
- **Python 3.11+** (für lokale Installation)
- **NVIDIA GPU mit CUDA** (für lokale LLMs)
- Min. 16GB VRAM für 7B Modelle, 32GB für 14B+

## Domain Templates

Seedling kommt mit vorgefertigten Templates für verschiedene Domänen:

| Domain | Beschreibung | Beispiel-Topics |
|--------|--------------|-----------------|
| **DevOps** | Infrastructure, CI/CD, Automation | Terraform, Ansible, Kubernetes, Docker |
| **SysAdmin** | Windows/Linux Administration | PowerShell, Bash, Active Directory, Intune |
| **Cloud** | AWS, Azure, GCP | IAM, Networking, Serverless |
| **Security** | InfoSec, Compliance | ISMS, Hardening, Incident Response |
| **Database** | SQL, NoSQL, Data Engineering | PostgreSQL, MongoDB, ETL |
| **Code** | General Programming | Python, TypeScript, Rust, Go |

## Nutzung

### 1. Domain auswählen
Wähle eine oder mehrere Domänen aus den Templates oder erstelle eigene.

### 2. Seed Instructions erstellen
Gib 10-50 Beispiel-Instructions als Ausgangspunkt ein.

### 3. Generation starten
Generiere hunderte/tausende Instructions mit Self-Instruct, Evol-Instruct oder Magpie.

### 4. Curation (optional)
Review und filtere die Daten mit Argilla.

### 5. Export
Exportiere als JSONL, Hugging Face Dataset, Alpaca oder ShareGPT Format.

## Generierungsmethoden

### Self-Instruct
Generiert neue Instructions basierend auf Seed-Beispielen. Ideal für diverse, ähnliche Instructions.

### Evol-Instruct
Entwickelt Instructions zu komplexeren Versionen. Gut für anspruchsvollere Trainingsdaten.

### Magpie
Nutzt LLM-spezifische Templates für natürlichere Prompts. Optimiert für Chat-Modelle.

## Konfiguration

Siehe `config/` für Konfigurationsdateien:

- `config/models.yaml` - LLM-Konfiguration
- `config/generation.yaml` - Generierungs-Parameter

### Umgebungsvariablen

| Variable | Beschreibung | Default |
|----------|--------------|---------|
| `OLLAMA_BASE_URL` | Ollama Server URL | `http://localhost:11434` |
| `ARGILLA_API_URL` | Argilla Server URL | `http://localhost:6900` |
| `ARGILLA_API_KEY` | Argilla API Key | `argilla.apikey` |
| `HF_TOKEN` | Hugging Face Token | - |

## Tests

```bash
# Tests ausführen
pytest tests/ -v

# Mit Coverage
pytest tests/ -v --cov=src --cov-report=html
```

## Entwicklung

```bash
# Dependencies installieren (inkl. Dev-Tools)
pip install -e .

# Linting
ruff check src/ tests/

# Type Checking
mypy src/
```

## Technologie-Stack

- **Gradio 4.x** - Web UI Framework
- **Distilabel 1.x** - Data Generation Pipeline
- **Ollama** - Lokale LLM-Inferenz
- **Argilla 2.x** - Data Curation Platform
- **Hugging Face Datasets** - Dataset-Format

## Lizenz

MIT

## Contributing

Beiträge sind willkommen! Bitte öffne ein Issue oder Pull Request auf GitHub.

---

Entwickelt mit dem Ziel, hochwertige Trainingsdaten für LLMs einfach und lokal zu erstellen.
