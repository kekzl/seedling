"""
Seedling - Synthetic Instruction Dataset Generator
Main Gradio Application
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import yaml

from .domains import DOMAIN_TEMPLATES, get_domain_seeds
from .generator import InstructionGenerator, GenerationConfig
from .exporter import DatasetExporter, ArgillaExporter


def load_models_from_config() -> list[str]:
    """Load available models from the YAML configuration file.

    Returns:
        List of model names available for generation.
    """
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return list(config.get("models", {}).keys())

    # Fallback to default models
    return [
        "qwen2.5-coder:14b",
        "qwen2.5-coder:7b",
        "qwen2.5:7b",
        "qwen2.5:14b",
        "llama3.1:8b",
        "codellama:13b",
        "deepseek-coder:6.7b",
    ]


def create_app() -> gr.Blocks:
    """Create the Gradio application.

    Returns:
        Configured Gradio Blocks application.
    """
    generator = InstructionGenerator()
    exporter = DatasetExporter()
    argilla_exporter = ArgillaExporter(
        api_url=os.getenv("ARGILLA_API_URL", "http://localhost:6900"),
        api_key=os.getenv("ARGILLA_API_KEY", "argilla.apikey"),
    )

    # Load models from config
    available_models = load_models_from_config()

    # State for generated data (using gr.State for multi-user safety)
    generated_data: list[dict[str, Any]] = []
    
    # Custom CSS for the app
    custom_css = """
    .domain-card { padding: 10px; border-radius: 8px; margin: 5px; }
    .stats-box { background: #f0f0f0; padding: 15px; border-radius: 8px; }
    """

    with gr.Blocks(title="Seedling") as app:
        
        gr.Markdown("""
        # 🌱 Seedling
        ### Synthetic Instruction Dataset Generator
        
        Erstelle hochwertige Instruction-Response-Paare für SFT mit lokalen LLMs.
        """)
        
        with gr.Tabs():
            # =================================================================
            # TAB 1: Domain Selection
            # =================================================================
            with gr.Tab("1️⃣ Domain auswählen"):
                gr.Markdown("### Wähle eine oder mehrere Domänen")
                
                domain_checkboxes = gr.CheckboxGroup(
                    choices=list(DOMAIN_TEMPLATES.keys()),
                    label="Domänen",
                    info="Wähle die Domänen für die du Daten generieren möchtest"
                )
                
                with gr.Accordion("Domain Details", open=False):
                    domain_info = gr.Markdown()
                
                def show_domain_info(domains: list[str] | None) -> str:
                    """Display detailed information about selected domains."""
                    if not domains:
                        return "Wähle Domänen um Details zu sehen."
                    info = ""
                    for d in domains:
                        template = DOMAIN_TEMPLATES.get(d, {})
                        info += f"### {d}\n"
                        info += f"**Beschreibung:** {template.get('description', 'N/A')}\n\n"
                        info += f"**Topics:** {', '.join(template.get('topics', []))}\n\n"
                        info += "---\n"
                    return info
                
                domain_checkboxes.change(
                    show_domain_info, 
                    inputs=[domain_checkboxes], 
                    outputs=[domain_info]
                )
            
            # =================================================================
            # TAB 2: Seed Instructions
            # =================================================================
            with gr.Tab("2️⃣ Seed Instructions"):
                gr.Markdown("""
                ### Seed Instructions
                
                Gib 10-50 Beispiel-Instructions als Ausgangspunkt ein.
                Diese werden verwendet um ähnliche Instructions zu generieren.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        seed_input = gr.Textbox(
                            label="Seed Instructions (eine pro Zeile)",
                            placeholder="Schreibe ein Bash-Skript das alle Docker Container stoppt\nErstelle eine Terraform-Konfiguration für einen S3 Bucket\n...",
                            lines=15
                        )
                    
                    with gr.Column(scale=1):
                        load_template_btn = gr.Button("📥 Template laden")
                        template_dropdown = gr.Dropdown(
                            choices=list(DOMAIN_TEMPLATES.keys()),
                            label="Domain Template"
                        )
                        
                        seed_count = gr.Number(
                            label="Anzahl Seeds",
                            value=0,
                            interactive=False
                        )
                
                def load_template_seeds(domain: str | None) -> tuple[str, int]:
                    """Load seed instructions from a domain template."""
                    if not domain:
                        return "", 0
                    seeds = get_domain_seeds(domain)
                    return "\n".join(seeds), len(seeds)

                def count_seeds(text: str) -> int:
                    """Count the number of non-empty seed instructions."""
                    if not text.strip():
                        return 0
                    return len([line for line in text.strip().split("\n") if line.strip()])
                
                load_template_btn.click(
                    load_template_seeds,
                    inputs=[template_dropdown],
                    outputs=[seed_input, seed_count]
                )
                
                seed_input.change(
                    count_seeds,
                    inputs=[seed_input],
                    outputs=[seed_count]
                )
            
            # =================================================================
            # TAB 3: Generation
            # =================================================================
            with gr.Tab("3️⃣ Generation"):
                gr.Markdown("### Instruction & Response Generation")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### LLM Konfiguration")
                        
                        model_dropdown = gr.Dropdown(
                            choices=available_models,
                            value=available_models[0] if available_models else "qwen2.5-coder:14b",
                            label="Model"
                        )
                        
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.5,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                        
                        max_tokens = gr.Slider(
                            minimum=256,
                            maximum=4096,
                            value=1024,
                            step=256,
                            label="Max Tokens"
                        )
                    
                    with gr.Column():
                        gr.Markdown("#### Generation Settings")
                        
                        num_instructions = gr.Slider(
                            minimum=10,
                            maximum=1000,
                            value=100,
                            step=10,
                            label="Anzahl zu generierende Instructions"
                        )
                        
                        generation_method = gr.Radio(
                            choices=["Self-Instruct", "Evol-Instruct", "Magpie"],
                            value="Self-Instruct",
                            label="Generierungsmethode"
                        )
                        
                        with gr.Accordion("Methodendetails", open=False):
                            gr.Markdown("""
                            **Self-Instruct:** Generiert neue Instructions basierend auf Seeds.
                            Gut für diverse, ähnliche Instructions.
                            
                            **Evol-Instruct:** Entwickelt Instructions zu komplexeren Versionen.
                            Gut für anspruchsvollere Trainingsdaten.
                            
                            **Magpie:** Nutzt LLM-spezifische Templates für natürlichere Prompts.
                            Gut für Chat-Modelle.
                            """)
                
                with gr.Row():
                    generate_btn = gr.Button("🚀 Generation starten", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop")
                
                progress = gr.Progress()
                generation_log = gr.Textbox(
                    label="Generation Log",
                    lines=10,
                    interactive=False
                )
                
                generation_stats = gr.JSON(
                    label="Statistiken",
                    value={}
                )
                
                async def run_generation(seeds_text, model, temp, max_tok, num_instr, method):
                    seeds = [l.strip() for l in seeds_text.strip().split("\n") if l.strip()]
                    
                    if len(seeds) < 5:
                        yield "❌ Mindestens 5 Seed Instructions benötigt!", {}
                        return
                    
                    log = f"🚀 Starte Generation mit {len(seeds)} Seeds...\n"
                    log += f"   Model: {model}\n"
                    log += f"   Method: {method}\n"
                    log += f"   Target: {num_instr} Instructions\n\n"
                    
                    yield log, {"status": "running", "generated": 0}
                    
                    try:
                        results = await generator.generate(
                            seeds=seeds,
                            model=model,
                            temperature=temp,
                            max_tokens=max_tok,
                            num_instructions=num_instr,
                            method=method.lower().replace("-", "_"),
                            on_progress=lambda msg: None
                        )
                        
                        generated_data.clear()
                        generated_data.extend(results)
                        
                        log += f"\n✅ Generation abgeschlossen!\n"
                        log += f"   Generiert: {len(results)} Instruction-Response-Paare\n"
                        
                        stats = {
                            "status": "completed",
                            "generated": len(results),
                            "avg_instruction_length": sum(len(r["instruction"]) for r in results) / len(results) if results else 0,
                            "avg_response_length": sum(len(r.get("response", "")) for r in results) / len(results) if results else 0,
                        }
                        
                        yield log, stats
                        
                    except Exception as e:
                        log += f"\n❌ Fehler: {str(e)}\n"
                        yield log, {"status": "error", "error": str(e)}
                
                generate_btn.click(
                    run_generation,
                    inputs=[seed_input, model_dropdown, temperature, max_tokens, num_instructions, generation_method],
                    outputs=[generation_log, generation_stats]
                )
            
            # =================================================================
            # TAB 4: Review & Curation
            # =================================================================
            with gr.Tab("4️⃣ Review"):
                gr.Markdown("### Daten Review")
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Daten laden")
                    push_argilla_btn = gr.Button("📤 Zu Argilla senden")
                
                data_preview = gr.Dataframe(
                    headers=["instruction", "response"],
                    label="Generierte Daten",
                    wrap=True
                )
                
                def refresh_data() -> list[list[str]]:
                    """Refresh the data preview table."""
                    if not generated_data:
                        return []
                    return [[d.get("instruction", ""), d.get("response", "")[:200] + "..."]
                            for d in generated_data[:100]]

                def push_to_argilla() -> str:
                    """Push generated data to Argilla for curation."""
                    if not generated_data:
                        return "❌ Keine Daten vorhanden. Generiere zuerst Daten."

                    try:
                        dataset_name = f"seedling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        url = argilla_exporter.push_to_argilla(
                            data=generated_data,
                            dataset_name=dataset_name,
                        )
                        return f"✅ Daten erfolgreich zu Argilla gesendet!\n📍 URL: {url}"
                    except ImportError:
                        return "❌ Argilla ist nicht installiert. Bitte `pip install argilla` ausführen."
                    except Exception as e:
                        return f"❌ Fehler beim Senden zu Argilla: {e}"

                argilla_status = gr.Textbox(label="Argilla Status", interactive=False)

                refresh_btn.click(refresh_data, outputs=[data_preview])
                push_argilla_btn.click(push_to_argilla, outputs=[argilla_status])

                gr.Markdown("""
                ---
                **Argilla Web UI:** [http://localhost:6900](http://localhost:6900)

                Nutze Argilla um die generierten Daten zu reviewen, annotieren und filtern.
                """)
            
            # =================================================================
            # TAB 5: Export
            # =================================================================
            with gr.Tab("5️⃣ Export"):
                gr.Markdown("### Daten exportieren")
                
                with gr.Row():
                    with gr.Column():
                        export_format = gr.Radio(
                            choices=["JSONL", "Hugging Face Dataset", "Alpaca Format", "ShareGPT Format"],
                            value="JSONL",
                            label="Export Format"
                        )
                        
                        export_name = gr.Textbox(
                            label="Dateiname / Dataset Name",
                            value=f"seedling_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        )
                        
                        with gr.Accordion("Hugging Face Upload", open=False):
                            hf_repo = gr.Textbox(
                                label="Repository ID",
                                placeholder="username/dataset-name"
                            )
                            hf_private = gr.Checkbox(label="Privates Repository")
                    
                    with gr.Column():
                        export_btn = gr.Button("📦 Exportieren", variant="primary")
                        export_status = gr.Textbox(label="Status", interactive=False)
                        export_file = gr.File(label="Download")
                
                def export_data(
                    format_type: str,
                    name: str,
                    hf_repo_id: str,
                    private: bool,
                ) -> tuple[str, str | None]:
                    """Export generated data to the specified format."""
                    if not generated_data:
                        return "❌ Keine Daten zum Exportieren!", None

                    if not name.strip():
                        return "❌ Bitte einen Dateinamen angeben!", None

                    try:
                        filepath = exporter.export(
                            data=generated_data,
                            format_type=format_type.lower().replace(" ", "_"),
                            name=name.strip(),
                            hf_repo=hf_repo_id.strip() if hf_repo_id else None,
                            private=private,
                        )
                        return f"✅ Export erfolgreich: {filepath}", filepath
                    except ValueError as e:
                        return f"❌ Ungültiges Format: {e}", None
                    except Exception as e:
                        return f"❌ Export fehlgeschlagen: {e}", None
                
                export_btn.click(
                    export_data,
                    inputs=[export_format, export_name, hf_repo, hf_private],
                    outputs=[export_status, export_file]
                )
        
        # Footer
        gr.Markdown("""
        ---
        <center>

        **Seedling** - Open Source Synthetic Data Generator

        [GitHub](https://github.com/kekzl/seedling) |
        [Documentation](https://github.com/kekzl/seedling/wiki)

        </center>
        """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
