"""Seedling - Synthetic Instruction Dataset Generator.

A complete stack for generating Instruction-Response pairs for
Supervised Fine-Tuning (SFT) with local LLMs.
"""

__version__ = "0.2.0"

from .domains import (
    DOMAIN_TEMPLATES,
    get_domain_seeds,
    get_all_topics,
    get_domain_description,
    get_all_templates,
    get_template_seeds,
    get_template_choices,
)
from .roles import (
    Role,
    RoleManager,
    RoleConfig,
    get_role_manager,
    get_predefined_roles,
    get_role_choices,
    get_role_seeds,
    get_role_as_domain_template,
    get_all_roles_as_domain_templates,
    generate_role_from_name,
)
from .generator import GenerationConfig, InstructionGenerator, SimpleGenerator
from .exporter import DatasetExporter, ArgillaExporter
from .hardware import (
    GPUInfo,
    SystemInfo,
    detect_system,
    get_system_info,
    get_hardware_summary,
    load_model_requirements,
    get_recommended_models,
    get_best_default_model,
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
    "ArgillaExporter",
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
