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

from .domains import DOMAIN_TEMPLATES, get_domain_seeds, get_template_choices
from .roles import (
    get_role_manager,
    get_predefined_roles,
    get_role_seeds,
    generate_role_from_name,
)
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
        
        # Get role manager for predefined roles
        role_manager = get_role_manager()
        predefined_roles = get_predefined_roles()

        # Build role choices grouped by category
        role_choices_by_category = {}
        for category_info in role_manager.get_categories():
            cat_name = category_info.get("name", "Other")
            roles = role_manager.get_roles_by_category(cat_name)
            if roles:
                role_choices_by_category[cat_name] = [
                    (role.display_name, role.name) for role in roles
                ]

        with gr.Tabs():
            # =================================================================
            # TAB 1: Template Selection (Domains + Roles)
            # =================================================================
            with gr.Tab("1️⃣ Template Selection"):
                gr.Markdown("""
                ### Choose Templates for Training Data Generation

                Select from **Technical Domains** (DevOps, Cloud, etc.) or
                **Professional Roles** (typical roles that can be augmented by AI).

                You can also generate a custom role by entering any profession name.
                """)

                with gr.Tabs():
                    # ---------------------------------------------------------
                    # Sub-Tab: Technical Domains
                    # ---------------------------------------------------------
                    with gr.Tab("Technical Domains"):
                        domain_checkboxes = gr.CheckboxGroup(
                            choices=list(DOMAIN_TEMPLATES.keys()),
                            label="Domains",
                            info="Select domains for data generation"
                        )

                        with gr.Accordion("Domain Details", open=False):
                            domain_info = gr.Markdown()

                        def show_domain_info(domains: list[str] | None) -> str:
                            """Display detailed information about selected domains."""
                            if not domains:
                                return "Select domains to see details."
                            info = ""
                            for d in domains:
                                template = DOMAIN_TEMPLATES.get(d, {})
                                info += f"### {d}\n"
                                info += f"**Description:** {template.get('description', 'N/A')}\n\n"
                                info += f"**Topics:** {', '.join(template.get('topics', []))}\n\n"
                                info += f"**Seed Count:** {len(template.get('seeds', []))}\n\n"
                                info += "---\n"
                            return info

                        domain_checkboxes.change(
                            show_domain_info,
                            inputs=[domain_checkboxes],
                            outputs=[domain_info]
                        )

                    # ---------------------------------------------------------
                    # Sub-Tab: Predefined Roles
                    # ---------------------------------------------------------
                    with gr.Tab("Professional Roles"):
                        gr.Markdown("""
                        **Predefined roles** that may be augmented or replaced by AI.
                        Each role comes with relevant topics and seed instructions.
                        """)

                        # Create role selection by category
                        role_selections = {}
                        for cat_name, roles in role_choices_by_category.items():
                            with gr.Accordion(f"{cat_name}", open=False):
                                role_selections[cat_name] = gr.CheckboxGroup(
                                    choices=[r[0] for r in roles],
                                    label=f"{cat_name} Roles",
                                )

                        with gr.Accordion("Role Details", open=False):
                            role_info = gr.Markdown()

                        def show_role_info(*selected_roles_lists) -> str:
                            """Display detailed information about selected roles."""
                            all_selected = []
                            for roles_list in selected_roles_lists:
                                if roles_list:
                                    all_selected.extend(roles_list)

                            if not all_selected:
                                return "Select roles to see details."

                            info = ""
                            for display_name in all_selected:
                                # Find the role by display name
                                for role in predefined_roles.values():
                                    if role.display_name == display_name:
                                        info += f"### {role.display_name}\n"
                                        info += f"**Category:** {role.category}\n\n"
                                        info += f"**Description:** {role.description}\n\n"
                                        info += f"**Topics:** {', '.join(role.topics[:8])}{'...' if len(role.topics) > 8 else ''}\n\n"
                                        info += f"**Seed Count:** {len(role.seeds)}\n\n"
                                        info += "---\n"
                                        break
                            return info

                        # Connect all role checkboxes to the info display
                        for checkbox in role_selections.values():
                            checkbox.change(
                                show_role_info,
                                inputs=list(role_selections.values()),
                                outputs=[role_info]
                            )

                    # ---------------------------------------------------------
                    # Sub-Tab: Custom Role Generation
                    # ---------------------------------------------------------
                    with gr.Tab("Custom Role (AI Generated)"):
                        gr.Markdown("""
                        ### Generate a Custom Role

                        Enter any profession or role name and the system will automatically
                        generate relevant topics and seed instructions using the LLM.

                        **Examples:** Researcher, Marketing Manager, UX Designer, Journalist,
                        Pharmacist, Real Estate Agent, Supply Chain Manager, etc.
                        """)

                        with gr.Row():
                            with gr.Column(scale=2):
                                custom_role_input = gr.Textbox(
                                    label="Role/Profession Name",
                                    placeholder="e.g., Researcher, UX Designer, Journalist...",
                                    info="Enter the name of the role you want to generate"
                                )
                            with gr.Column(scale=1):
                                custom_role_model = gr.Dropdown(
                                    choices=available_models,
                                    value=available_models[0] if available_models else "qwen2.5-coder:14b",
                                    label="Model for Generation"
                                )

                        generate_role_btn = gr.Button("Generate Role Template", variant="primary")

                        with gr.Row():
                            custom_role_status = gr.Textbox(
                                label="Status",
                                interactive=False,
                                lines=2
                            )

                        with gr.Accordion("Generated Role Details", open=True):
                            generated_role_info = gr.Markdown()
                            generated_role_seeds = gr.Textbox(
                                label="Generated Seeds (editable)",
                                lines=10,
                                interactive=True
                            )

                        async def generate_custom_role(
                            role_name: str,
                            model: str,
                        ) -> tuple[str, str, str]:
                            """Generate a custom role from the given name."""
                            if not role_name or not role_name.strip():
                                return "Please enter a role name.", "", ""

                            try:
                                status_messages = []
                                def on_progress(msg: str):
                                    status_messages.append(msg)

                                role = await generate_role_from_name(
                                    role_name=role_name.strip(),
                                    model=model,
                                    on_progress=on_progress,
                                )

                                info = f"""
### {role.display_name}

**Description:** {role.description}

**Generated Topics ({len(role.topics)}):**
{', '.join(role.topics)}

**Generated Seeds:** {len(role.seeds)} instructions
                                """

                                seeds_text = "\n".join(role.seeds)

                                return (
                                    f"Successfully generated role: {role.display_name}",
                                    info,
                                    seeds_text,
                                )
                            except Exception as e:
                                return f"Error: {str(e)}", "", ""

                        generate_role_btn.click(
                            generate_custom_role,
                            inputs=[custom_role_input, custom_role_model],
                            outputs=[custom_role_status, generated_role_info, generated_role_seeds]
                        )
            
            # =================================================================
            # TAB 2: Seed Instructions
            # =================================================================
            with gr.Tab("2️⃣ Seed Instructions"):
                gr.Markdown("""
                ### Seed Instructions

                Enter 10-50 example instructions as a starting point.
                These will be used to generate similar instructions.

                You can load seeds from domains, predefined roles, or use
                your custom generated role seeds.
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        seed_input = gr.Textbox(
                            label="Seed Instructions (one per line)",
                            placeholder="Write a Bash script that stops all Docker containers\nCreate a Terraform configuration for an S3 bucket\n...",
                            lines=15
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### Load from Template")

                        template_source = gr.Radio(
                            choices=["Domain", "Role"],
                            value="Domain",
                            label="Template Source"
                        )

                        template_dropdown = gr.Dropdown(
                            choices=list(DOMAIN_TEMPLATES.keys()),
                            label="Select Template"
                        )

                        load_template_btn = gr.Button("Load Seeds")

                        seed_count = gr.Number(
                            label="Seed Count",
                            value=0,
                            interactive=False
                        )

                        # Add button to load from custom role
                        use_custom_role_btn = gr.Button("Use Custom Role Seeds")

                def update_template_choices(source: str) -> dict:
                    """Update template dropdown based on source selection."""
                    if source == "Domain":
                        choices = list(DOMAIN_TEMPLATES.keys())
                    else:
                        # Get role display names
                        choices = [role.display_name for role in predefined_roles.values()]
                    return gr.update(choices=choices, value=choices[0] if choices else None)

                def load_template_seeds(source: str, template_name: str | None) -> tuple[str, int]:
                    """Load seed instructions from a domain or role template."""
                    if not template_name:
                        return "", 0

                    if source == "Domain":
                        seeds = get_domain_seeds(template_name)
                    else:
                        # Find role by display name
                        seeds = []
                        for role in predefined_roles.values():
                            if role.display_name == template_name:
                                seeds = role.seeds
                                break
                    return "\n".join(seeds), len(seeds)

                def count_seeds(text: str) -> int:
                    """Count the number of non-empty seed instructions."""
                    if not text.strip():
                        return 0
                    return len([line for line in text.strip().split("\n") if line.strip()])

                def use_custom_role_seeds(generated_seeds: str) -> tuple[str, int]:
                    """Use seeds from the custom generated role."""
                    if not generated_seeds:
                        return "", 0
                    return generated_seeds, count_seeds(generated_seeds)

                template_source.change(
                    update_template_choices,
                    inputs=[template_source],
                    outputs=[template_dropdown]
                )

                load_template_btn.click(
                    load_template_seeds,
                    inputs=[template_source, template_dropdown],
                    outputs=[seed_input, seed_count]
                )

                seed_input.change(
                    count_seeds,
                    inputs=[seed_input],
                    outputs=[seed_count]
                )

                use_custom_role_btn.click(
                    use_custom_role_seeds,
                    inputs=[generated_role_seeds],
                    outputs=[seed_input, seed_count]
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
