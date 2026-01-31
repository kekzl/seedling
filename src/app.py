"""
Seedling - Synthetic Instruction Dataset Generator
Main Gradio Application
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
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
from .hardware import (
    get_system_info,
    get_hardware_summary,
    load_model_requirements,
    get_recommended_models,
    get_best_default_model,
)


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


def get_models_with_recommendations() -> tuple[list[str], str, str]:
    """Get models sorted by recommendation based on detected hardware.

    Returns:
        Tuple of (sorted_models, default_model, status_message)
    """
    all_models = load_models_from_config()
    system_info = get_system_info()
    model_requirements = load_model_requirements()

    # Get recommendations
    recommended, possible, incompatible = get_recommended_models(
        system_info, model_requirements
    )

    # Build sorted list: recommended first, then possible, then incompatible
    sorted_models = []

    # Add recommended models with indicator
    for model in recommended:
        sorted_models.append(f"{model}")

    # Add possible models
    for model in possible:
        if model not in sorted_models:
            sorted_models.append(f"{model}")

    # Add incompatible models (user might still want to try)
    for model in incompatible:
        if model not in sorted_models:
            sorted_models.append(f"{model}")

    # Add any models not in requirements (from config but not evaluated)
    for model in all_models:
        if model not in sorted_models:
            sorted_models.append(model)

    # Determine default model
    default_model = get_best_default_model(system_info, model_requirements)
    if not default_model and sorted_models:
        default_model = sorted_models[0]

    # Build status message
    effective_vram = system_info.effective_vram_gb
    if system_info.has_gpu:
        gpu_name = system_info.gpus[0].name.replace("NVIDIA ", "").replace("GeForce ", "")
        if system_info.is_wsl:
            status = f"{gpu_name} | {effective_vram}GB effective (WSL2)"
        elif system_info.os_type == "Windows":
            status = f"{gpu_name} | {effective_vram}GB effective (Windows)"
        else:
            status = f"{gpu_name} | {effective_vram}GB effective"

        if recommended:
            status += f" | {len(recommended)} recommended models"
        elif possible:
            status += f" | {len(possible)} possible models (tight fit)"
        else:
            status += " | No compatible models (consider smaller models)"
    else:
        status = "CPU only | No GPU detected"

    return sorted_models, default_model or all_models[0], status


def get_model_compatibility_info(model_name: str) -> str:
    """Get compatibility info for a specific model.

    Args:
        model_name: The model name to check

    Returns:
        Compatibility status string
    """
    system_info = get_system_info()
    model_requirements = load_model_requirements()

    if model_name not in model_requirements:
        return "Unknown VRAM requirements"

    req = model_requirements[model_name]
    available_vram = system_info.available_vram_mb

    vram_required_gb = round(req.vram_required_mb / 1024, 1)
    vram_recommended_gb = round(req.vram_recommended_mb / 1024, 1)
    available_gb = round(available_vram / 1024, 1)

    if req.vram_recommended_mb <= available_vram:
        return f"Recommended ({vram_required_gb}-{vram_recommended_gb}GB required, {available_gb}GB available)"
    elif req.vram_required_mb <= available_vram:
        return f"Possible but tight ({vram_required_gb}GB required, {available_gb}GB available)"
    else:
        return f"May not fit ({vram_required_gb}GB required, only {available_gb}GB available)"


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

    # Load models with hardware-based recommendations
    available_models, default_model, hardware_status = get_models_with_recommendations()
    system_info = get_system_info()

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

        # Hardware detection info display
        with gr.Accordion("System Info", open=False):
            with gr.Row():
                with gr.Column(scale=2):
                    hardware_info_display = gr.Textbox(
                        value=system_info.get_display_string(),
                        label="Detected Hardware",
                        lines=8,
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    gr.Markdown(f"""
**Status:** {hardware_status}

**Auto-selected Model:** `{default_model}`

*Models are sorted by compatibility with your hardware.
WSL2 and Windows reserve some VRAM for the display system.*
                    """)
                    refresh_hw_btn = gr.Button("Refresh Hardware Info")

            def refresh_hardware() -> tuple[str, str]:
                """Refresh hardware detection."""
                new_info = get_system_info(refresh=True)
                _, new_default, new_status = get_models_with_recommendations()
                return (
                    new_info.get_display_string(),
                    f"**Status:** {new_status}\n\n**Auto-selected Model:** `{new_default}`"
                )

            # Note: Gradio doesn't easily support updating Markdown, so we use a simpler approach
            refresh_hw_btn.click(
                lambda: get_system_info(refresh=True).get_display_string(),
                outputs=[hardware_info_display]
            )
        
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
                                    value=default_model,
                                    label="Model for Generation",
                                    info="Auto-selected based on your hardware"
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
                            value=default_model,
                            label="Model",
                            info="Auto-selected based on your hardware"
                        )

                        model_compatibility = gr.Textbox(
                            value=get_model_compatibility_info(default_model),
                            label="Compatibility",
                            interactive=False,
                            lines=1
                        )

                        def update_compatibility(model: str) -> str:
                            """Update compatibility info when model changes."""
                            return get_model_compatibility_info(model)

                        model_dropdown.change(
                            update_compatibility,
                            inputs=[model_dropdown],
                            outputs=[model_compatibility]
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

                    log_lines = [
                        f"🚀 Starte Generation mit {len(seeds)} Seeds...",
                        f"   Model: {model}",
                        f"   Method: {method}",
                        f"   Target: {num_instr} Instructions",
                        "",
                    ]

                    yield "\n".join(log_lines), {"status": "running", "generated": 0}

                    # Queue for real-time progress updates from generator thread
                    progress_queue: Queue[str | None] = Queue()
                    result_holder: dict[str, Any] = {"results": None, "error": None}

                    def on_progress(msg: str) -> None:
                        """Callback that puts progress messages into the queue."""
                        progress_queue.put(msg)

                    def run_generator() -> None:
                        """Run the generator in a separate thread."""
                        try:
                            # Create new event loop for this thread
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)

                            results = loop.run_until_complete(
                                generator.generate(
                                    seeds=seeds,
                                    model=model,
                                    temperature=temp,
                                    max_tokens=max_tok,
                                    num_instructions=num_instr,
                                    method=method.lower().replace("-", "_"),
                                    on_progress=on_progress,
                                )
                            )
                            result_holder["results"] = results
                        except Exception as e:
                            result_holder["error"] = str(e)
                        finally:
                            # Signal completion
                            progress_queue.put(None)

                    # Start generator in background thread
                    executor = ThreadPoolExecutor(max_workers=1)
                    future = executor.submit(run_generator)

                    # Poll for progress updates while generator runs
                    try:
                        while True:
                            try:
                                # Wait for progress message with timeout
                                msg = progress_queue.get(timeout=0.5)

                                if msg is None:
                                    # Generator finished
                                    break

                                # Add progress message to log
                                log_lines.append(f"📍 {msg}")
                                yield "\n".join(log_lines), {
                                    "status": "running",
                                    "generated": 0,
                                    "last_update": msg,
                                }

                            except Empty:
                                # No message yet, just continue polling
                                await asyncio.sleep(0.1)
                                continue

                        # Wait for thread to fully complete
                        future.result(timeout=5)

                    finally:
                        executor.shutdown(wait=False)

                    # Process results
                    if result_holder["error"]:
                        log_lines.append(f"\n❌ Fehler: {result_holder['error']}")
                        yield "\n".join(log_lines), {"status": "error", "error": result_holder["error"]}
                        return

                    results = result_holder["results"] or []
                    generated_data.clear()
                    generated_data.extend(results)

                    log_lines.append("")
                    log_lines.append("✅ Generation abgeschlossen!")
                    log_lines.append(f"   Generiert: {len(results)} Instruction-Response-Paare")

                    stats = {
                        "status": "completed",
                        "generated": len(results),
                        "avg_instruction_length": round(
                            sum(len(r["instruction"]) for r in results) / len(results), 1
                        ) if results else 0,
                        "avg_response_length": round(
                            sum(len(r.get("response", "")) for r in results) / len(results), 1
                        ) if results else 0,
                    }

                    yield "\n".join(log_lines), stats
                
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
    import os
    app = create_app()
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
    )
