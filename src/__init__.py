"""Seedling - Synthetic Instruction Dataset Generator.

A complete stack for generating Instruction-Response pairs for
Supervised Fine-Tuning (SFT) with local LLMs.
"""

__version__ = "0.2.0"

from .domains import (
    DOMAIN_TEMPLATES,
    get_all_templates,
    get_all_topics,
    get_domain_description,
    get_domain_seeds,
    get_template_choices,
    get_template_seeds,
)
from .exporter import DatasetExporter
from .generator import GenerationConfig, InstructionGenerator, SimpleGenerator
from .hardware import (
    GPUInfo,
    SystemInfo,
    detect_system,
    get_best_default_model,
    get_hardware_summary,
    get_recommended_models,
    get_system_info,
    load_model_requirements,
)
from .roles import (
    Role,
    RoleConfig,
    RoleManager,
    generate_role_from_name,
    get_all_roles_as_domain_templates,
    get_predefined_roles,
    get_role_as_domain_template,
    get_role_choices,
    get_role_manager,
    get_role_seeds,
)

__all__ = [
    "__version__",
    # Domains
    "DOMAIN_TEMPLATES",
    "get_domain_seeds",
    "get_all_topics",
    "get_domain_description",
    "get_all_templates",
    "get_template_seeds",
    "get_template_choices",
    # Roles
    "Role",
    "RoleManager",
    "RoleConfig",
    "get_role_manager",
    "get_predefined_roles",
    "get_role_choices",
    "get_role_seeds",
    "get_role_as_domain_template",
    "get_all_roles_as_domain_templates",
    "generate_role_from_name",
    # Generator
    "GenerationConfig",
    "InstructionGenerator",
    "SimpleGenerator",
    # Exporter
    "DatasetExporter",
    # Hardware Detection
    "GPUInfo",
    "SystemInfo",
    "detect_system",
    "get_system_info",
    "get_hardware_summary",
    "load_model_requirements",
    "get_recommended_models",
    "get_best_default_model",
]
