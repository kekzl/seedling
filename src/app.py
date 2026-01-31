"""
Seedling - Synthetic Instruction Dataset Generator
Main Gradio Application - Simplified Quick Start UI
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import gradio as gr
import requests
import yaml

from .domains import DOMAIN_TEMPLATES, get_domain_seeds
from .exporter import DatasetExporter
from .generator import InstructionGenerator, SimpleGenerator
from .hardware import (
    get_best_default_model,
    get_recommended_models,
    get_system_info,
    load_model_requirements,
)
from .roles import (
    generate_role_from_name,
    get_predefined_roles,
    get_role_manager,
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


def get_ollama_base_url() -> str:
    """Get Ollama base URL from environment or default."""
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_gpu_stats() -> dict[str, Any]:
    """Get current GPU utilization stats.

    Returns:
        Dict with gpu_util, mem_used, mem_total, temperature
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                return {
                    "gpu_util": int(parts[0]),
                    "mem_used_mb": int(parts[1]),
                    "mem_total_mb": int(parts[2]),
                    "temp_c": int(parts[3]),
                }
    except Exception:
        pass
    return {}


def get_ollama_models() -> dict[str, dict]:
    """Get list of models available in Ollama.

    Returns:
        Dict mapping model name to model info (size, loaded status, etc.)
    """
    try:
        resp = requests.get(f"{get_ollama_base_url()}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = {}
            for m in data.get("models", []):
                name = m.get("name", "")
                models[name] = {
                    "size_gb": m.get("size", 0) / 1e9,
                    "parameter_size": m.get("details", {}).get("parameter_size", ""),
                    "quantization": m.get("details", {}).get("quantization_level", ""),
                }
            return models
    except Exception:
        pass
    return {}


def get_ollama_running_models() -> dict[str, dict]:
    """Get models currently loaded in Ollama memory.

    Returns:
        Dict mapping model name to runtime info (VRAM usage, etc.)
    """
    try:
        resp = requests.get(f"{get_ollama_base_url()}/api/ps", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = {}
            for m in data.get("models", []):
                name = m.get("name", "")
                models[name] = {
                    "size_vram_gb": m.get("size_vram", 0) / 1e9,
                    "expires_at": m.get("expires_at", ""),
                }
            return models
    except Exception:
        pass
    return {}


def pull_ollama_model(model_name: str) -> tuple[bool, str]:
    """Pull a model from Ollama registry.

    Args:
        model_name: Name of the model to pull

    Returns:
        Tuple of (success, message)
    """
    try:
        # Use streaming to track progress
        resp = requests.post(
            f"{get_ollama_base_url()}/api/pull",
            json={"name": model_name, "stream": False},
            timeout=600,  # 10 minutes timeout for large models
        )
        if resp.status_code == 200:
            return True, f"Model '{model_name}' downloaded successfully"
        else:
            return False, f"Failed to pull model: {resp.text}"
    except requests.Timeout:
        return False, "Download timed out - model may be too large"
    except Exception as e:
        return False, f"Error pulling model: {str(e)}"


def get_models_with_recommendations(
    hide_incompatible: bool = True,
) -> tuple[list[str], str, str, dict[str, str]]:
    """Get models sorted by recommendation based on detected hardware.

    Args:
        hide_incompatible: If True, don't include models that exceed VRAM

    Returns:
        Tuple of (sorted_models, default_model, status_message, model_status_map)
        model_status_map: Dict mapping model name to status emoji/text
    """
    all_models = load_models_from_config()
    system_info = get_system_info()
    model_requirements = load_model_requirements()

    # Get Ollama status
    ollama_models = get_ollama_models()
    running_models = get_ollama_running_models()

    # Get recommendations
    recommended, possible, incompatible = get_recommended_models(system_info, model_requirements)

    # Build model status map
    model_status: dict[str, str] = {}
    for model in all_models:
        if model in running_models:
            model_status[model] = "loaded"  # Currently in GPU memory
        elif model in ollama_models:
            model_status[model] = "ready"  # Downloaded, ready to use
        elif model in incompatible:
            model_status[model] = "too_large"  # Exceeds VRAM
        else:
            model_status[model] = "download"  # Needs to be pulled

    # Build sorted list with status indicators
    sorted_models = []

    # First: Running/loaded models (ready to use immediately)
    for model in running_models:
        if model in all_models or model in ollama_models:
            sorted_models.append(model)

    # Second: Downloaded models that are recommended
    for model in recommended:
        if model not in sorted_models and model in ollama_models:
            sorted_models.append(model)

    # Third: Downloaded models that are possible
    for model in possible:
        if model not in sorted_models and model in ollama_models:
            sorted_models.append(model)

    # Fourth: Other downloaded models from Ollama
    for model in ollama_models:
        if model not in sorted_models:
            sorted_models.append(model)

    # Fifth: Recommended models not yet downloaded
    for model in recommended:
        if model not in sorted_models:
            sorted_models.append(model)

    # Sixth: Possible models not yet downloaded
    for model in possible:
        if model not in sorted_models:
            sorted_models.append(model)

    # Seventh: Incompatible models (only if not hiding)
    if not hide_incompatible:
        for model in incompatible:
            if model not in sorted_models:
                sorted_models.append(model)

    # Determine default model - prefer loaded, then ready, then recommended
    default_model = None
    # First check running models
    for model in running_models:
        if model in recommended or model in possible:
            default_model = model
            break
    # Then check ready models
    if not default_model:
        for model in recommended:
            if model in ollama_models:
                default_model = model
                break
    # Fall back to hardware recommendation
    if not default_model:
        default_model = get_best_default_model(system_info, model_requirements)
    if not default_model and sorted_models:
        default_model = sorted_models[0]

    # Build status message
    effective_vram = system_info.effective_vram_gb
    loaded_count = len(running_models)
    ready_count = len([m for m in ollama_models if m not in running_models])

    if system_info.has_gpu:
        gpu_name = system_info.gpus[0].name.replace("NVIDIA ", "").replace("GeForce ", "")
        if system_info.is_wsl:
            status = f"{gpu_name} | {effective_vram:.0f}GB VRAM (WSL2)"
        elif system_info.os_type == "Windows":
            status = f"{gpu_name} | {effective_vram:.0f}GB VRAM (Windows)"
        else:
            status = f"{gpu_name} | {effective_vram:.0f}GB VRAM"

        status += f" | {loaded_count} loaded, {ready_count} ready"
    else:
        status = f"CPU only | {loaded_count} loaded, {ready_count} ready"

    return (
        sorted_models,
        default_model or (all_models[0] if all_models else ""),
        status,
        model_status,
    )


def format_model_choice(model: str, model_status: dict[str, str]) -> str:
    """Format model name with status indicator for dropdown.

    Args:
        model: Model name
        model_status: Dict mapping model names to status

    Returns:
        Formatted string with status emoji
    """
    status = model_status.get(model, "download")
    status_prefixes = {
        "loaded": "[GPU]",
        "ready": "[OK]",
        "too_large": "[!!]",
    }
    prefix = status_prefixes.get(status, "[DL]")
    return f"{prefix} {model}"


def get_model_choices_and_status() -> tuple[list[str], str, str, dict[str, str]]:
    """Get formatted model choices for dropdown with status indicators.

    Returns:
        Tuple of (choices, default, status_message, raw_model_status)
    """
    models, default, status, model_status = get_models_with_recommendations()

    # Format choices with status indicators
    choices = [format_model_choice(m, model_status) for m in models]
    default_choice = (
        format_model_choice(default, model_status) if default else choices[0] if choices else ""
    )

    return choices, default_choice, status, model_status


def extract_model_name(choice: str) -> str:
    """Extract raw model name from formatted dropdown choice.

    Args:
        choice: Formatted choice like "[GPU] qwen2.5:7b"

    Returns:
        Raw model name like "qwen2.5:7b"
    """
    # Remove status prefix
    for prefix in ["[GPU] ", "[OK] ", "[DL] ", "[!!] "]:
        if choice.startswith(prefix):
            return choice[len(prefix) :]
    return choice


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


def get_topic_choices() -> list[tuple[str, str]]:
    """Combine domains and roles into grouped dropdown choices.

    Returns:
        List of tuples (display_name, internal_value) for dropdown
    """
    choices = []

    # Add domains
    for domain in DOMAIN_TEMPLATES:
        choices.append((f"Domain: {domain}", f"domain:{domain}"))

    # Add roles grouped by category
    role_manager = get_role_manager()

    for category_info in role_manager.get_categories():
        cat_name = category_info.get("name", "Other")
        roles = role_manager.get_roles_by_category(cat_name)
        for role in roles:
            choices.append((f"Role: {role.display_name}", f"role:{role.name}"))

    return choices


def load_seeds_for_topic(topic_value: str) -> list[str]:
    """Load seed instructions for a given topic.

    Args:
        topic_value: Topic value like "domain:DevOps" or "role:customer_support"

    Returns:
        List of seed instruction strings
    """
    if not topic_value:
        return []

    if topic_value.startswith("domain:"):
        domain_name = topic_value[7:]
        return get_domain_seeds(domain_name)
    elif topic_value.startswith("role:"):
        role_name = topic_value[5:]
        predefined_roles = get_predefined_roles()
        role = predefined_roles.get(role_name)
        if role:
            return role.seeds
    return []


def parse_quantity(quantity_str: str) -> int:
    """Parse quantity string to integer.

    Args:
        quantity_str: String like "100 (Quick)" or "500 (Standard)"

    Returns:
        Integer quantity
    """
    if "100" in quantity_str:
        return 100
    elif "500" in quantity_str:
        return 500
    elif "1000" in quantity_str:
        return 1000
    return 500  # default


def create_app() -> gr.Blocks:
    """Create the Gradio application.

    Returns:
        Configured Gradio Blocks application.
    """
    generator = InstructionGenerator()
    exporter = DatasetExporter()

    # Load models with hardware-based recommendations
    available_models, default_model, hardware_status, model_status = (
        get_models_with_recommendations()
    )
    # Format choices with status indicators
    model_choices = [format_model_choice(m, model_status) for m in available_models]
    default_choice = format_model_choice(default_model, model_status) if default_model else ""

    # State for generated data (using gr.State for multi-user safety)
    generated_data: list[dict[str, Any]] = []

    # Build topic choices
    topic_choices = get_topic_choices()
    topic_choices_display = [t[0] for t in topic_choices]

    # Get role manager for predefined roles
    role_manager = get_role_manager()
    predefined_roles = get_predefined_roles()

    # Build role choices grouped by category
    role_choices_by_category = {}
    for category_info in role_manager.get_categories():
        cat_name = category_info.get("name", "Other")
        roles = role_manager.get_roles_by_category(cat_name)
        if roles:
            role_choices_by_category[cat_name] = [(role.display_name, role.name) for role in roles]

    # Get icon path
    icon_path = Path(__file__).parent.parent / "assets" / "icon.svg"

    with gr.Blocks(
        title="Seedling",
        head=f'<link rel="icon" type="image/svg+xml" href="/file={icon_path}">'
        if icon_path.exists()
        else None,
    ) as app:
        gr.Markdown("""
        # 🌱 Seedling
        ### Synthetic Instruction Dataset Generator

        Generate high-quality instruction-response pairs for SFT with local LLMs.
        """)

        # Mode toggle
        mode_toggle = gr.Radio(
            choices=["Quick Start", "Advanced"],
            value="Quick Start",
            label="Mode",
            container=False,
        )

        # Hidden state for seeds (not visible to user in Quick Start)
        seeds_state = gr.State([])

        # =====================================================================
        # QUICK START MODE
        # =====================================================================
        with gr.Column(visible=True) as quick_section:
            gr.Markdown("---")

            with gr.Row():
                with gr.Column(scale=2):
                    # Topic dropdown
                    topic_dropdown = gr.Dropdown(
                        choices=topic_choices_display,
                        value=topic_choices_display[0] if topic_choices_display else None,
                        label="Choose a topic",
                        info="Select a domain or professional role",
                    )

                    # Quantity radio
                    quantity_radio = gr.Radio(
                        choices=["100 (Quick)", "500 (Standard)", "1000 (Large)"],
                        value="500 (Standard)",
                        label="How many instructions?",
                    )

                with gr.Column(scale=1):
                    # Status display
                    gr.Markdown(f"**Model:** `{default_model}`\n\n**Hardware:** {hardware_status}")

            # Generate button
            generate_quick_btn = gr.Button(
                "Generate Instructions",
                variant="primary",
                size="lg",
            )

            # Progress section
            quick_progress = gr.Textbox(
                label="Progress",
                lines=5,
                interactive=False,
                visible=False,
            )

            # Export section (hidden until generation completes)
            with gr.Column(visible=False) as quick_export_section:
                export_summary = gr.Markdown()

                with gr.Row():
                    download_jsonl_btn = gr.Button("Download JSONL")
                    download_alpaca_btn = gr.Button("Download Alpaca")
                    download_sharegpt_btn = gr.Button("Download ShareGPT")

                quick_download_file = gr.File(label="Download", visible=False)

        # =====================================================================
        # ADVANCED MODE
        # =====================================================================
        with gr.Column(visible=False) as advanced_section:
            gr.Markdown("---")
            gr.Markdown("### Advanced Settings")

            with gr.Row():
                # ---------------------------------------------------------
                # Left Column: Template Selection
                # ---------------------------------------------------------
                with gr.Column(scale=1):  # noqa: SIM117
                    with gr.Tabs():
                        # Sub-Tab: Technical Domains
                        with gr.Tab("Domains"):
                            domain_checkboxes = gr.CheckboxGroup(
                                choices=list(DOMAIN_TEMPLATES.keys()),
                                label="Technical Domains",
                                info="Select domains for data generation",
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
                                    info += (
                                        f"**Description:** {template.get('description', 'N/A')}\n\n"
                                    )
                                    info += (
                                        f"**Topics:** {', '.join(template.get('topics', []))}\n\n"
                                    )
                                    info += f"**Seed Count:** {len(template.get('seeds', []))}\n\n"
                                    info += "---\n"
                                return info

                        # Sub-Tab: Professional Roles
                        with gr.Tab("Roles"):
                            gr.Markdown("""
                            **Predefined roles** with relevant topics and seed instructions.
                            """)

                            # Create role selection by category
                            role_selections: dict[str, gr.CheckboxGroup] = {}
                            for cat_name, role_tuples in role_choices_by_category.items():
                                with gr.Accordion(f"{cat_name}", open=False):
                                    role_selections[cat_name] = gr.CheckboxGroup(
                                        choices=[r[0] for r in role_tuples],
                                        label=f"{cat_name} Roles",
                                    )

                            with gr.Accordion("Role Details", open=False):
                                role_info = gr.Markdown()

                        # Sub-Tab: Custom Role Generation
                        with gr.Tab("Custom (AI)"):
                            gr.Markdown("""
                            Enter any profession name to generate topics and seeds using the LLM.
                            """)

                            with gr.Row():
                                custom_role_input = gr.Textbox(
                                    label="Role/Profession Name",
                                    placeholder="e.g., Researcher, UX Designer...",
                                    scale=2,
                                )
                                custom_role_model = gr.Dropdown(
                                    choices=model_choices,
                                    value=default_choice,
                                    label="Model",
                                    scale=1,
                                )

                            generate_role_btn = gr.Button(
                                "Generate Role Template", variant="primary"
                            )

                            custom_role_status = gr.Textbox(
                                label="Status", interactive=False, lines=1
                            )

                            with gr.Accordion("Generated Role Details", open=True):
                                generated_role_info = gr.Markdown()
                                generated_role_seeds = gr.Textbox(
                                    label="Generated Seeds (editable)", lines=8, interactive=True
                                )

                            async def generate_custom_role(
                                role_name: str,
                                model_choice: str,
                            ) -> tuple[str, str, str]:
                                """Generate a custom role from the given name."""
                                if not role_name or not role_name.strip():
                                    return "Please enter a role name.", "", ""

                                # Extract raw model name from formatted choice
                                model = extract_model_name(model_choice)

                                try:
                                    status_messages: list[str] = []

                                    def on_progress(msg: str) -> None:
                                        status_messages.append(msg)

                                    role = await generate_role_from_name(
                                        role_name=role_name.strip(),
                                        model=model,
                                        on_progress=on_progress,
                                    )

                                    info = f"""
### {role.display_name}

**Description:** {role.description}

**Topics ({len(role.topics)}):** {", ".join(role.topics)}

**Seeds:** {len(role.seeds)} instructions
                                    """

                                    seeds_text = "\n".join(role.seeds)

                                    return (
                                        f"Generated: {role.display_name}",
                                        info,
                                        seeds_text,
                                    )
                                except Exception as e:
                                    return f"Error: {str(e)}", "", ""

                            generate_role_btn.click(
                                generate_custom_role,
                                inputs=[custom_role_input, custom_role_model],
                                outputs=[
                                    custom_role_status,
                                    generated_role_info,
                                    generated_role_seeds,
                                ],
                            )

                # ---------------------------------------------------------
                # Right Column: Seeds & Settings
                # ---------------------------------------------------------
                with gr.Column(scale=1):
                    gr.Markdown("#### Seed Instructions")
                    seed_input = gr.Textbox(
                        label="Seeds (one per line)",
                        placeholder="Write a Bash script that stops all Docker containers\nCreate a Terraform configuration for an S3 bucket\n...",
                        lines=10,
                    )

                    with gr.Row():
                        seed_count = gr.Number(
                            label="Seed Count", value=0, interactive=False, scale=1
                        )
                        use_custom_role_btn = gr.Button("Use Custom Role Seeds", scale=2)

            # Model and generation settings
            gr.Markdown("---")
            gr.Markdown("#### LLM Configuration")

            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        "*[GPU] = loaded in VRAM | [OK] = downloaded | [DL] = needs download*",
                    )

                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=model_choices,
                            value=default_choice,
                            label="Model",
                            scale=3,
                        )
                        refresh_models_btn = gr.Button("Refresh", scale=1, min_width=50)

                    with gr.Row():
                        download_model_btn = gr.Button("Download Model", variant="secondary")
                        download_status = gr.Textbox(
                            label="",
                            interactive=False,
                            lines=1,
                            scale=2,
                        )

                    model_compatibility = gr.Textbox(
                        value=get_model_compatibility_info(default_model),
                        label="Compatibility",
                        interactive=False,
                        lines=1,
                    )

                with gr.Column():
                    temperature = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature"
                    )

                    max_tokens = gr.Slider(
                        minimum=256, maximum=4096, value=1024, step=256, label="Max Tokens"
                    )

            with gr.Row():
                with gr.Column():
                    num_instructions = gr.Slider(
                        minimum=10, maximum=1000, value=100, step=10, label="Number of Instructions"
                    )

                with gr.Column():
                    generation_method = gr.Radio(
                        choices=["Self-Instruct", "Evol-Instruct", "Magpie"],
                        value="Self-Instruct",
                        label="Generation Method",
                    )

            with gr.Accordion("Method Details", open=False):
                gr.Markdown("""
                **Self-Instruct:** Generates new instructions based on seeds.
                Good for diverse, similar instructions.

                **Evol-Instruct:** Evolves instructions to more complex versions.
                Good for challenging training data.

                **Magpie:** Uses LLM-specific templates for more natural prompts.
                Good for chat models.
                """)

            with gr.Row():
                generate_adv_btn = gr.Button("Generate", variant="primary", size="lg")
                gr.Button("Stop", variant="stop")  # TODO: Implement stop functionality

            generation_log = gr.Textbox(label="Generation Log", lines=8, interactive=False)

            generation_stats = gr.JSON(label="Statistics", value={})

            # Advanced export section
            gr.Markdown("---")
            gr.Markdown("#### Export & Review")

            with gr.Row():
                with gr.Column():
                    export_format = gr.Radio(
                        choices=[
                            "JSONL",
                            "Hugging Face Dataset",
                            "Alpaca Format",
                            "ShareGPT Format",
                        ],
                        value="JSONL",
                        label="Export Format",
                    )

                    export_name = gr.Textbox(
                        label="Filename / Dataset Name",
                        value=f"seedling_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    )

                    with gr.Accordion("Hugging Face Upload", open=False):
                        hf_repo = gr.Textbox(
                            label="Repository ID", placeholder="username/dataset-name"
                        )
                        hf_private = gr.Checkbox(label="Private Repository")

                with gr.Column():
                    export_adv_btn = gr.Button("Export", variant="primary")
                    export_status = gr.Textbox(label="Status", interactive=False)
                    export_file = gr.File(label="Download")

        # =====================================================================
        # EVENT HANDLERS
        # =====================================================================

        # Mode toggle handler
        def toggle_mode(mode: str) -> tuple[gr.update, gr.update]:
            """Toggle between Quick Start and Advanced modes."""
            is_quick = mode == "Quick Start"
            return gr.update(visible=is_quick), gr.update(visible=not is_quick)

        mode_toggle.change(
            toggle_mode, inputs=[mode_toggle], outputs=[quick_section, advanced_section]
        )

        # Topic selection handler (Quick Start)
        def on_topic_select(topic_display: str) -> list[str]:
            """Load seeds when topic is selected."""
            # Find the value for this display name
            for display, value in topic_choices:
                if display == topic_display:
                    return load_seeds_for_topic(value)
            return []

        topic_dropdown.change(on_topic_select, inputs=[topic_dropdown], outputs=[seeds_state])

        # Quick Start generation handler
        async def run_quick_generation(
            topic_display: str,
            quantity_str: str,
            seeds: list[str],
        ):
            """Run generation in Quick Start mode."""
            # Parse inputs
            num = parse_quantity(quantity_str)

            # Get default model
            _, default_model, _, _ = get_models_with_recommendations()

            # Load seeds if not already loaded
            if not seeds:
                for display, value in topic_choices:
                    if display == topic_display:
                        seeds = load_seeds_for_topic(value)
                        break

            if len(seeds) < 5:
                yield (
                    gr.update(
                        visible=True,
                        value="Need at least 5 seed instructions. Try a different topic.",
                    ),
                    gr.update(visible=False),
                    "",
                    None,
                    seeds,
                )
                return

            # Show progress
            yield (
                gr.update(visible=True, value=f"Starting generation with {len(seeds)} seeds..."),
                gr.update(visible=False),
                "",
                None,
                seeds,
            )

            # Queue for real-time progress updates from generator thread
            progress_queue: Queue[str | None] = Queue()
            result_holder: dict[str, Any] = {"results": None, "error": None}

            def on_progress(msg: str) -> None:
                """Callback that puts progress messages into the queue."""
                progress_queue.put(msg)

            def run_generator_sync() -> None:
                """Run the generator synchronously (blocking)."""
                import signal as sig_module  # noqa: I001

                import nest_asyncio

                # Patch signal at the start of this thread
                original_sig = sig_module.signal

                def safe_signal(signum, handler):
                    try:
                        return original_sig(signum, handler)
                    except ValueError:
                        return None

                sig_module.signal = safe_signal

                try:
                    # Create new event loop for this thread and allow nesting
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    nest_asyncio.apply(loop)

                    # Try distilabel-based InstructionGenerator first
                    results = loop.run_until_complete(
                        generator.generate(
                            seeds=seeds,
                            model=default_model,
                            temperature=0.7,
                            max_tokens=1024,
                            num_instructions=num,
                            method="self_instruct",
                            on_progress=on_progress,
                        )
                    )

                    # If distilabel pipeline failed, fallback to SimpleGenerator
                    if not results:
                        on_progress("Distilabel pipeline returned no results, using fallback...")
                        simple_gen = SimpleGenerator()
                        results = loop.run_until_complete(
                            simple_gen.generate(
                                seeds=seeds,
                                model=default_model,
                                temperature=0.7,
                                max_tokens=1024,
                                num_instructions=num,
                                method="self_instruct",
                                on_progress=on_progress,
                            )
                        )

                    result_holder["results"] = results
                except Exception as e:
                    import traceback

                    result_holder["error"] = f"{e}\n{traceback.format_exc()}"
                finally:
                    # Restore signal and signal completion
                    sig_module.signal = original_sig
                    progress_queue.put(None)

            # Start generator in background thread
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(run_generator_sync)

            # Track timing
            start_time = time.time()
            log_lines = [f"Generating {num} instructions..."]

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
                        log_lines.append(msg)
                        yield (
                            gr.update(visible=True, value="\n".join(log_lines[-10:])),
                            gr.update(visible=False),
                            "",
                            None,
                            seeds,
                        )

                    except Empty:
                        # No message from generator, show activity
                        elapsed = time.time() - start_time
                        elapsed_str = (
                            f"{int(elapsed)}s"
                            if elapsed < 60
                            else f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
                        )
                        status_line = f"Generating... [{elapsed_str}]"

                        yield (
                            gr.update(
                                visible=True, value="\n".join(log_lines[-9:] + [status_line])
                            ),
                            gr.update(visible=False),
                            "",
                            None,
                            seeds,
                        )

                        await asyncio.sleep(0.1)
                        continue

                # Wait for thread to fully complete
                future.result(timeout=5)

            finally:
                executor.shutdown(wait=False)

            # Process results
            if result_holder["error"]:
                yield (
                    gr.update(visible=True, value=f"Error: {result_holder['error']}"),
                    gr.update(visible=False),
                    "",
                    None,
                    seeds,
                )
                return

            results = result_holder["results"] or []
            generated_data.clear()
            generated_data.extend(results)

            # Show export section
            total_time = time.time() - start_time
            time_str = (
                f"{total_time:.1f}s"
                if total_time < 60
                else f"{int(total_time // 60)}m {int(total_time % 60)}s"
            )

            if len(results) == 0:
                # Generation failed - show error message
                summary = f"""
**Generation failed!**

- Generated: **0** instruction-response pairs
- Time: {time_str}
- **Error:** The LLM did not return any valid responses.

**Possible causes:**
- Model may be too slow or timing out
- Model may not be loaded in Ollama
- Try a smaller model (e.g., qwen2.5:7b or qwen3:8b)

Check the Docker logs for more details:
`docker logs seedling-app --tail 50`
                """
                yield (
                    gr.update(visible=True, value=summary),
                    gr.update(visible=False),
                    "",
                    None,
                    seeds,
                )
            else:
                summary = f"""
**Generation complete!**

- Generated: **{len(results)}** instruction-response pairs
- Time: {time_str}
- Ready to download
                """

                yield (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    summary,
                    None,
                    seeds,
                )

        generate_quick_btn.click(
            run_quick_generation,
            inputs=[topic_dropdown, quantity_radio, seeds_state],
            outputs=[
                quick_progress,
                quick_export_section,
                export_summary,
                quick_download_file,
                seeds_state,
            ],
        )

        # Quick export handlers
        def export_quick_jsonl() -> str | None:
            """Export as JSONL."""
            if not generated_data:
                return None
            name = f"seedling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            filepath = exporter.export(
                data=generated_data,
                format_type="jsonl",
                name=name,
            )
            return filepath

        def export_quick_alpaca() -> str | None:
            """Export as Alpaca format."""
            if not generated_data:
                return None
            name = f"seedling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            filepath = exporter.export(
                data=generated_data,
                format_type="alpaca_format",
                name=name,
            )
            return filepath

        def export_quick_sharegpt() -> str | None:
            """Export as ShareGPT format."""
            if not generated_data:
                return None
            name = f"seedling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            filepath = exporter.export(
                data=generated_data,
                format_type="sharegpt_format",
                name=name,
            )
            return filepath

        download_jsonl_btn.click(export_quick_jsonl, outputs=[quick_download_file])
        download_alpaca_btn.click(export_quick_alpaca, outputs=[quick_download_file])
        download_sharegpt_btn.click(export_quick_sharegpt, outputs=[quick_download_file])

        # -----------------------------------------------------------------
        # Advanced Mode Handlers
        # -----------------------------------------------------------------

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

        seed_input.change(count_seeds, inputs=[seed_input], outputs=[seed_count])

        use_custom_role_btn.click(
            use_custom_role_seeds, inputs=[generated_role_seeds], outputs=[seed_input, seed_count]
        )

        # Auto-load seeds when domain is selected
        def show_domain_info_and_load_seeds(
            domains: list[str] | None,
        ) -> tuple[str, str, int]:
            """Display domain info AND auto-load seeds from ALL selected domains."""
            info = show_domain_info(domains)
            if domains:
                # Load and combine seeds from ALL selected domains
                all_seeds = []
                for domain in domains:
                    all_seeds.extend(get_domain_seeds(domain))
                seeds_text = "\n".join(all_seeds)
                return info, seeds_text, len(all_seeds)
            return info, "", 0

        domain_checkboxes.change(
            show_domain_info_and_load_seeds,
            inputs=[domain_checkboxes],
            outputs=[domain_info, seed_input, seed_count],
        )

        # Auto-load seeds when roles are selected
        def show_role_info_and_load_seeds(
            *selected_roles_lists,
        ) -> tuple[str, str, int]:
            """Display role info AND auto-load seeds from ALL selected roles."""
            all_selected = []
            for roles_list in selected_roles_lists:
                if roles_list:
                    all_selected.extend(roles_list)

            if not all_selected:
                return "Select roles to see details.", "", 0

            info = ""
            all_seeds = []
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
                        # Collect seeds
                        all_seeds.extend(role.seeds)
                        break

            seeds_text = "\n".join(all_seeds)
            return info, seeds_text, len(all_seeds)

        # Connect all role checkboxes to info display AND seed loading
        for checkbox in role_selections.values():
            checkbox.change(
                show_role_info_and_load_seeds,
                inputs=list(role_selections.values()),
                outputs=[role_info, seed_input, seed_count],
            )

        # Model handlers
        def update_compatibility(model_choice: str) -> str:
            """Update compatibility info when model changes."""
            model = extract_model_name(model_choice)
            return get_model_compatibility_info(model)

        def refresh_model_list() -> tuple[gr.update, gr.update]:
            """Refresh the model dropdown with current Ollama status."""
            models, default, _, status = get_models_with_recommendations()
            choices = [format_model_choice(m, status) for m in models]
            default_fmt = format_model_choice(default, status) if default else ""
            return gr.update(choices=choices, value=default_fmt), ""

        def download_selected_model(model_choice: str):
            """Download the selected model from Ollama."""
            model = extract_model_name(model_choice)
            yield gr.update(), f"Downloading {model}..."
            success, msg = pull_ollama_model(model)
            # Refresh model list after download
            models, default, _, status = get_models_with_recommendations()
            choices = [format_model_choice(m, status) for m in models]
            # Keep current selection but update its status
            new_choice = format_model_choice(model, status)
            yield gr.update(choices=choices, value=new_choice), msg

        model_dropdown.change(
            update_compatibility, inputs=[model_dropdown], outputs=[model_compatibility]
        )

        refresh_models_btn.click(refresh_model_list, outputs=[model_dropdown, download_status])

        download_model_btn.click(
            download_selected_model,
            inputs=[model_dropdown],
            outputs=[model_dropdown, download_status],
        )

        # Advanced generation handler
        async def run_generation(seeds_text, model_choice, temp, max_tok, num_instr, method):
            seeds = [line.strip() for line in seeds_text.strip().split("\n") if line.strip()]

            if len(seeds) < 5:
                yield "Need at least 5 seed instructions!", {}
                return

            # Extract raw model name from formatted choice
            model = extract_model_name(model_choice)

            log_lines = [
                f"Starting generation with {len(seeds)} seeds...",
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

            def run_generator_sync() -> None:
                """Run the generator synchronously (blocking)."""
                import signal as sig_module  # noqa: I001

                import nest_asyncio

                # Patch signal at the start of this thread
                original_sig = sig_module.signal

                def safe_signal(signum, handler):
                    try:
                        return original_sig(signum, handler)
                    except ValueError:
                        return None

                sig_module.signal = safe_signal

                try:
                    # Create new event loop for this thread and allow nesting
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    nest_asyncio.apply(loop)

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
                    import traceback

                    result_holder["error"] = f"{e}\n{traceback.format_exc()}"
                finally:
                    # Restore signal and signal completion
                    sig_module.signal = original_sig
                    progress_queue.put(None)

            # Start generator in background thread
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(run_generator_sync)

            # Track timing and GPU stats
            start_time = time.time()
            last_gpu_update = 0
            gpu_update_interval = 2.0  # Update GPU stats every 2 seconds
            spinner_chars = [".", "..", "...", "...."]
            spinner_idx = 0

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
                        log_lines.append(f">> {msg}")
                        yield (
                            "\n".join(log_lines),
                            {
                                "status": "running",
                                "generated": 0,
                                "last_update": msg,
                            },
                        )

                    except Empty:
                        # No message from generator, show activity with GPU stats
                        elapsed = time.time() - start_time
                        current_time = time.time()

                        # Update GPU stats periodically
                        if current_time - last_gpu_update >= gpu_update_interval:
                            last_gpu_update = current_time
                            gpu = get_gpu_stats()

                            # Build status line
                            spinner = spinner_chars[spinner_idx % len(spinner_chars)]
                            spinner_idx += 1

                            elapsed_str = f"{int(elapsed)}s"
                            if elapsed >= 60:
                                elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

                            if gpu:
                                status_line = (
                                    f"Generating{spinner} "
                                    f"[{elapsed_str}] "
                                    f"GPU: {gpu.get('gpu_util', 0)}% | "
                                    f"VRAM: {gpu.get('mem_used_mb', 0) / 1024:.1f}GB | "
                                    f"Temp: {gpu.get('temp_c', 0)}C"
                                )
                            else:
                                status_line = f"Generating{spinner} [{elapsed_str}]"

                            # Update the last line if it's a status line, otherwise append
                            if log_lines and log_lines[-1].startswith("Generating"):
                                log_lines[-1] = status_line
                            else:
                                log_lines.append(status_line)

                            yield (
                                "\n".join(log_lines),
                                {
                                    "status": "running",
                                    "generated": 0,
                                    "elapsed_seconds": int(elapsed),
                                    "gpu_util": gpu.get("gpu_util", 0) if gpu else None,
                                },
                            )

                        await asyncio.sleep(0.1)
                        continue

                # Wait for thread to fully complete
                future.result(timeout=5)

                # Final timing
                total_time = time.time() - start_time
                if total_time >= 60:
                    time_str = f"{int(total_time // 60)}m {int(total_time % 60)}s"
                else:
                    time_str = f"{total_time:.1f}s"
                log_lines.append(f"Total time: {time_str}")

            finally:
                executor.shutdown(wait=False)

            # Process results
            if result_holder["error"]:
                log_lines.append(f"\nError: {result_holder['error']}")
                yield "\n".join(log_lines), {"status": "error", "error": result_holder["error"]}
                return

            results = result_holder["results"] or []
            generated_data.clear()
            generated_data.extend(results)

            log_lines.append("")
            log_lines.append("Generation complete!")
            log_lines.append(f"   Generated: {len(results)} instruction-response pairs")

            stats = {
                "status": "completed",
                "generated": len(results),
                "avg_instruction_length": round(
                    sum(len(r["instruction"]) for r in results) / len(results), 1
                )
                if results
                else 0,
                "avg_response_length": round(
                    sum(len(r.get("response") or "") for r in results) / len(results), 1
                )
                if results
                else 0,
            }

            yield "\n".join(log_lines), stats

        generate_adv_btn.click(
            run_generation,
            inputs=[
                seed_input,
                model_dropdown,
                temperature,
                max_tokens,
                num_instructions,
                generation_method,
            ],
            outputs=[generation_log, generation_stats],
        )

        # Export handler
        def export_data(
            format_type: str,
            name: str,
            hf_repo_id: str,
            private: bool,
        ) -> tuple[str, str | None]:
            """Export generated data to the specified format."""
            if not generated_data:
                return "No data to export!", None

            if not name.strip():
                return "Please enter a filename!", None

            try:
                filepath = exporter.export(
                    data=generated_data,
                    format_type=format_type.lower().replace(" ", "_"),
                    name=name.strip(),
                    hf_repo=hf_repo_id.strip() if hf_repo_id else None,
                    private=private,
                )
                return f"Export successful: {filepath}", filepath
            except ValueError as e:
                return f"Invalid format: {e}", None
            except Exception as e:
                return f"Export failed: {e}", None

        export_adv_btn.click(
            export_data,
            inputs=[export_format, export_name, hf_repo, hf_private],
            outputs=[export_status, export_file],
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
