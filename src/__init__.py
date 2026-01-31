"""Seedling - Synthetic Instruction Dataset Generator.

A complete stack for generating Instruction-Response pairs for
Supervised Fine-Tuning (SFT) with local LLMs.
"""

__version__ = "0.2.0"

from .domains import DOMAIN_TEMPLATES, get_domain_seeds, get_all_topics, get_domain_description
from .generator import GenerationConfig, InstructionGenerator, SimpleGenerator
from .exporter import DatasetExporter, ArgillaExporter

__all__ = [
    "__version__",
    # Domains
    "DOMAIN_TEMPLATES",
    "get_domain_seeds",
    "get_all_topics",
    "get_domain_description",
    # Generator
    "GenerationConfig",
    "InstructionGenerator",
    "SimpleGenerator",
    # Exporter
    "DatasetExporter",
    "ArgillaExporter",
]
