# 🌱 Seedling

**Open-Source Synthetic Instruction Dataset Generator**

Ein vollständiger Stack zum Erstellen von Instruction-Response-Paaren für SFT (Supervised Fine-Tuning) mit lokalen LLMs.

## Features

- 🖥️ **Web UI** - Gradio-basierte Oberfläche für einfache Bedienung
- 🤖 **Lokale LLMs** - Ollama-Integration, kein API-Key nötig
- 📚 **Domain Templates** - Vorgefertigte Domänen (DevOps, Code, etc.)
- 🔄 **Batch Generation** - Massenhaft Instructions generieren
- ✅ **Curation** - Review und Filterung mit Argilla
- 📤 **Export** - JSONL, Hugging Face Datasets Format

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

## Quick Start

```bash
# Clone repo
git clone https://github.com/yourusername/seedling.git
cd seedling

# Start mit Docker Compose
docker compose up -d

# Web UI öffnen
open http://localhost:7860
```

## Voraussetzungen

- Docker & Docker Compose
- NVIDIA GPU mit CUDA (für lokale LLMs)
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

### 3. Batch Generation
Generiere hunderte/tausende Instructions mit Self-Instruct oder Evol-Instruct.

### 4. Response Generation
Generiere Responses für alle Instructions.

### 5. Curation (optional)
Review und filtere die Daten mit Argilla.

### 6. Export
Exportiere als JSONL oder direkt zum Hugging Face Hub.

## Konfiguration

Siehe `config/` für Beispielkonfigurationen:

- `config/domains/` - Domain-Templates
- `config/models.yaml` - LLM-Konfiguration
- `config/generation.yaml` - Generierungs-Parameter

## Lizenz

MIT
