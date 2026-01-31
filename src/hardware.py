"""
Hardware and OS Detection Module for Seedling.

This module detects system hardware (GPU, VRAM) and operating system
to recommend appropriate models for generation tasks.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GPUInfo:
    """Information about a detected GPU.

    Attributes:
        name: GPU model name (e.g., "NVIDIA GeForce RTX 5090")
        vram_total_mb: Total VRAM in megabytes
        vram_free_mb: Currently free VRAM in megabytes
        vram_used_mb: Currently used VRAM in megabytes
        cuda_version: CUDA version if available
        driver_version: GPU driver version
        index: GPU index (for multi-GPU systems)
    """

    name: str
    vram_total_mb: int
    vram_free_mb: int
    vram_used_mb: int
    cuda_version: str | None = None
    driver_version: str | None = None
    index: int = 0


@dataclass
class SystemInfo:
    """Complete system information for model selection.

    Attributes:
        os_type: Operating system type (Linux, Windows, Darwin/macOS)
        os_version: Operating system version
        is_wsl: Whether running under WSL (Windows Subsystem for Linux)
        wsl_version: WSL version (1 or 2) if applicable
        gpus: List of detected GPUs
        cpu_name: CPU model name
        ram_total_mb: Total system RAM in megabytes
        ram_free_mb: Free system RAM in megabytes
    """

    os_type: str
    os_version: str
    is_wsl: bool = False
    wsl_version: int | None = None
    gpus: list[GPUInfo] = field(default_factory=list)
    cpu_name: str = "Unknown"
    ram_total_mb: int = 0
    ram_free_mb: int = 0

    @property
    def has_gpu(self) -> bool:
        """Check if any GPU is available."""
        return len(self.gpus) > 0

    @property
    def total_vram_mb(self) -> int:
        """Get total VRAM across all GPUs."""
        return sum(gpu.vram_total_mb for gpu in self.gpus)

    @property
    def available_vram_mb(self) -> int:
        """Get available VRAM for model selection considering OS overhead.

        Uses TOTAL VRAM (not free) because Ollama automatically manages model loading/unloading.
        Model filtering should be based on what the GPU can handle, not what's currently free.

        WSL2 and Windows reserve some VRAM for the display system.
        This returns the effectively usable VRAM for ML workloads.
        """
        if not self.gpus:
            return 0

        # Use TOTAL VRAM for model selection (Ollama manages loaded models)
        total_vram = sum(gpu.vram_total_mb for gpu in self.gpus)

        # OS-specific overhead adjustments
        if self.is_wsl:
            # WSL2: Windows reserves VRAM for display compositor
            # Typically 2-4GB depending on resolution and monitors
            # Conservative estimate: 3GB overhead per GPU
            overhead_per_gpu = 3072  # 3GB
            total_overhead = overhead_per_gpu * len(self.gpus)
            return max(0, total_vram - total_overhead)
        elif self.os_type == "Windows":
            # Native Windows: Similar overhead for DWM (Desktop Window Manager)
            overhead_per_gpu = 2048  # 2GB
            total_overhead = overhead_per_gpu * len(self.gpus)
            return max(0, total_vram - total_overhead)
        else:
            # Linux without desktop (headless): minimal overhead
            # Linux with desktop: ~1GB overhead
            overhead_per_gpu = 512  # 0.5GB conservative
            total_overhead = overhead_per_gpu * len(self.gpus)
            return max(0, total_vram - total_overhead)

    @property
    def current_free_vram_mb(self) -> int:
        """Get currently free VRAM across all GPUs.

        This reflects real-time availability and may be reduced by loaded models.
        Use this for monitoring, not for model selection.
        """
        return sum(gpu.vram_free_mb for gpu in self.gpus)

    @property
    def effective_vram_gb(self) -> float:
        """Get effective available VRAM in GB (rounded to 1 decimal)."""
        return round(self.available_vram_mb / 1024, 1)

    def get_display_string(self) -> str:
        """Get a human-readable system summary."""
        lines = []

        # OS info
        os_str = f"{self.os_type} {self.os_version}"
        if self.is_wsl:
            os_str += f" (WSL{self.wsl_version or ''})"
        lines.append(f"OS: {os_str}")

        # CPU info
        if self.cpu_name != "Unknown":
            lines.append(f"CPU: {self.cpu_name}")

        # RAM info
        if self.ram_total_mb > 0:
            ram_gb = round(self.ram_total_mb / 1024, 1)
            lines.append(f"RAM: {ram_gb} GB")

        # GPU info
        if self.gpus:
            for gpu in self.gpus:
                vram_total_gb = round(gpu.vram_total_mb / 1024, 1)
                vram_free_gb = round(gpu.vram_free_mb / 1024, 1)
                lines.append(f"GPU {gpu.index}: {gpu.name}")
                lines.append(f"  VRAM: {vram_free_gb} / {vram_total_gb} GB free")
                if gpu.cuda_version:
                    lines.append(f"  CUDA: {gpu.cuda_version}")

            # Show effective VRAM with overhead note
            effective_gb = self.effective_vram_gb
            if self.is_wsl:
                lines.append(f"Effective VRAM: ~{effective_gb} GB (WSL2 overhead deducted)")
            elif self.os_type == "Windows":
                lines.append(f"Effective VRAM: ~{effective_gb} GB (Windows overhead deducted)")
            else:
                lines.append(f"Effective VRAM: ~{effective_gb} GB")
        else:
            lines.append("GPU: None detected (CPU-only mode)")

        return "\n".join(lines)


def detect_wsl() -> tuple[bool, int | None]:
    """Detect if running under Windows Subsystem for Linux.

    Returns:
        Tuple of (is_wsl, wsl_version)
    """
    # Check for WSL-specific indicators
    is_wsl = False
    wsl_version = None

    # Method 1: Check /proc/version for Microsoft
    try:
        with open("/proc/version") as f:
            version_info = f.read().lower()
            if "microsoft" in version_info or "wsl" in version_info:
                is_wsl = True
    except (FileNotFoundError, PermissionError):
        pass

    # Method 2: Check for WSL-specific environment variables
    if os.environ.get("WSL_DISTRO_NAME"):
        is_wsl = True

    if os.environ.get("WSL_INTEROP"):
        is_wsl = True
        wsl_version = 2  # WSL_INTEROP only exists in WSL2

    # Method 3: Check for /run/WSL directory (WSL2 specific)
    if Path("/run/WSL").exists():
        is_wsl = True
        wsl_version = 2

    # Try to determine WSL version if we know it's WSL but don't know version
    if is_wsl and wsl_version is None:
        try:
            # Check /proc/sys/fs/binfmt_misc for WSL version hints
            # WSL2 uses a real Linux kernel, WSL1 uses translation layer
            kernel_release = platform.release()
            if "microsoft" in kernel_release.lower():
                # Contains "microsoft-standard" for WSL2
                wsl_version = 2 if "standard" in kernel_release.lower() else 1
        except Exception:
            pass

    return is_wsl, wsl_version


def detect_nvidia_gpus() -> list[GPUInfo]:
    """Detect NVIDIA GPUs using nvidia-smi.

    Returns:
        List of GPUInfo objects for each detected GPU
    """
    gpus = []

    try:
        # Query GPU info using nvidia-smi
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return gpus

        # Parse output
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                try:
                    gpu = GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        vram_total_mb=int(float(parts[2])),
                        vram_free_mb=int(float(parts[3])),
                        vram_used_mb=int(float(parts[4])),
                        driver_version=parts[5],
                    )
                    gpus.append(gpu)
                except (ValueError, IndexError):
                    continue

        # Try to get CUDA version
        cuda_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if cuda_result.returncode == 0:
            cuda_versions = cuda_result.stdout.strip().split("\n")
            for i, version in enumerate(cuda_versions):
                if i < len(gpus):
                    gpus[i].cuda_version = version.strip()

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return gpus


def detect_cpu() -> str:
    """Detect CPU model name.

    Returns:
        CPU model name string
    """
    cpu_name = "Unknown"

    # Try /proc/cpuinfo on Linux
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":")[1].strip()
                    break
    except (FileNotFoundError, PermissionError):
        pass

    # Fallback to platform module
    if cpu_name == "Unknown":
        cpu_name = platform.processor() or "Unknown"

    return cpu_name


def detect_ram() -> tuple[int, int]:
    """Detect system RAM.

    Returns:
        Tuple of (total_mb, free_mb)
    """
    total_mb = 0
    free_mb = 0

    # Try /proc/meminfo on Linux
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # Value is in kB
                    total_kb = int(line.split()[1])
                    total_mb = total_kb // 1024
                elif line.startswith("MemAvailable:"):
                    free_kb = int(line.split()[1])
                    free_mb = free_kb // 1024
    except (FileNotFoundError, PermissionError, ValueError):
        pass

    return total_mb, free_mb


def detect_system() -> SystemInfo:
    """Detect complete system information.

    Returns:
        SystemInfo object with all detected hardware info
    """
    # OS detection
    os_type = platform.system()
    os_version = platform.release()

    # WSL detection
    is_wsl, wsl_version = detect_wsl()

    # GPU detection
    gpus = detect_nvidia_gpus()

    # CPU detection
    cpu_name = detect_cpu()

    # RAM detection
    ram_total, ram_free = detect_ram()

    return SystemInfo(
        os_type=os_type,
        os_version=os_version,
        is_wsl=is_wsl,
        wsl_version=wsl_version,
        gpus=gpus,
        cpu_name=cpu_name,
        ram_total_mb=ram_total,
        ram_free_mb=ram_free,
    )


@dataclass
class ModelRequirements:
    """VRAM requirements for a model.

    Attributes:
        name: Model name
        vram_required_mb: Required VRAM in megabytes
        vram_recommended_mb: Recommended VRAM for good performance
        description: Human-readable description
        priority: Quality ranking for instruction generation (lower = better)
    """

    name: str
    vram_required_mb: int
    vram_recommended_mb: int
    description: str = ""
    priority: int = 50  # Default mid-range priority


def load_model_requirements(config_path: Path | None = None) -> dict[str, ModelRequirements]:
    """Load model VRAM requirements from configuration.

    Args:
        config_path: Path to models.yaml. Uses default if not provided.

    Returns:
        Dictionary mapping model names to their requirements
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "models.yaml"

    requirements = {}

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        models = config.get("models", {})
        for model_name, model_info in models.items():
            vram_str = model_info.get("vram_required", "0")
            vram_recommended_str = model_info.get("vram_recommended", vram_str)

            # Parse VRAM string (e.g., "~28GB" or "28" or 28000)
            vram_required = parse_vram_string(vram_str)
            vram_recommended = parse_vram_string(vram_recommended_str)

            requirements[model_name] = ModelRequirements(
                name=model_name,
                vram_required_mb=vram_required,
                vram_recommended_mb=vram_recommended,
                description=model_info.get("description", ""),
                priority=model_info.get("priority", 50),
            )

    except (FileNotFoundError, yaml.YAMLError):
        pass

    return requirements


def parse_vram_string(vram_str: str | int | float) -> int:
    """Parse a VRAM string into megabytes.

    Handles formats like:
    - "~28GB" -> 28672 MB
    - "28GB" -> 28672 MB
    - "14" -> 14336 MB (assumed GB)
    - 28000 -> 28000 MB

    Args:
        vram_str: VRAM specification string or number

    Returns:
        VRAM in megabytes
    """
    if isinstance(vram_str, int | float):
        # If it's a small number, assume GB
        if vram_str < 1000:
            return int(vram_str * 1024)
        return int(vram_str)

    vram_str = str(vram_str).strip().lower()

    # Remove common prefixes
    vram_str = vram_str.lstrip("~").lstrip("≈").strip()

    # Extract number
    match = re.search(r"(\d+(?:\.\d+)?)", vram_str)
    if not match:
        return 0

    value = float(match.group(1))

    # Check for unit
    if "gb" in vram_str or "g" in vram_str:
        return int(value * 1024)
    elif "mb" in vram_str or "m" in vram_str:
        return int(value)
    else:
        # Assume GB if no unit and value is small
        if value < 1000:
            return int(value * 1024)
        return int(value)


def get_recommended_models(
    system_info: SystemInfo,
    model_requirements: dict[str, ModelRequirements],
    safety_margin_mb: int = 1024,
) -> tuple[list[str], list[str], list[str]]:
    """Get model recommendations based on available VRAM.

    Args:
        system_info: Detected system information
        model_requirements: Dictionary of model requirements
        safety_margin_mb: Extra VRAM margin to keep free (default 1GB)

    Returns:
        Tuple of (recommended_models, possible_models, incompatible_models)
        - recommended: Models that fit comfortably with recommended VRAM
        - possible: Models that fit but may be tight
        - incompatible: Models that won't fit
    """
    available_vram = system_info.available_vram_mb - safety_margin_mb

    recommended = []
    possible = []
    incompatible = []

    for model_name, req in model_requirements.items():
        if req.vram_recommended_mb <= available_vram:
            recommended.append(model_name)
        elif req.vram_required_mb <= available_vram:
            possible.append(model_name)
        else:
            incompatible.append(model_name)

    # Sort by priority (lower = better for instruction generation)
    recommended.sort(key=lambda m: model_requirements[m].priority)
    possible.sort(key=lambda m: model_requirements[m].priority)
    incompatible.sort(key=lambda m: model_requirements[m].priority)

    return recommended, possible, incompatible


def get_best_default_model(
    system_info: SystemInfo,
    model_requirements: dict[str, ModelRequirements],
    preferred_models: list[str] | None = None,
) -> str | None:
    """Get the best default model for the detected hardware.

    Args:
        system_info: Detected system information
        model_requirements: Dictionary of model requirements
        preferred_models: Optional list of preferred model names (in priority order)

    Returns:
        Name of the recommended model, or None if no suitable model found
    """
    recommended, possible, _ = get_recommended_models(system_info, model_requirements)

    all_compatible = recommended + possible

    if not all_compatible:
        return None

    # If preferred models specified, try to find one that's compatible
    if preferred_models:
        for preferred in preferred_models:
            if preferred in all_compatible:
                return preferred

    # Default priority: prefer larger models from recommended list
    # Preference order for coding tasks
    default_preference = [
        "qwen2.5-coder:14b",  # Best for code if VRAM allows
        "qwen2.5:14b",  # Good general purpose
        "qwen2.5-coder:7b",  # Smaller but still good for code
        "llama3.1:8b",  # Good English performance
        "qwen2.5:7b",  # General purpose
        "deepseek-coder:6.7b",  # Efficient code model
        "codellama:13b",  # Specialized for code
    ]

    for model in default_preference:
        if model in recommended:
            return model

    for model in default_preference:
        if model in possible:
            return model

    # Fallback to first recommended or possible model
    return all_compatible[0] if all_compatible else None


# Cached system info to avoid repeated detection
_cached_system_info: SystemInfo | None = None


def get_system_info(refresh: bool = False) -> SystemInfo:
    """Get cached system information.

    Args:
        refresh: If True, force re-detection

    Returns:
        SystemInfo object
    """
    global _cached_system_info

    if _cached_system_info is None or refresh:
        _cached_system_info = detect_system()

    return _cached_system_info


def get_hardware_summary() -> str:
    """Get a short hardware summary for display.

    Returns:
        Short summary string like "RTX 5090 (28GB effective) | WSL2"
    """
    info = get_system_info()

    parts = []

    # GPU info
    if info.gpus:
        gpu = info.gpus[0]  # Primary GPU
        # Shorten GPU name
        name = gpu.name.replace("NVIDIA ", "").replace("GeForce ", "")
        effective_gb = info.effective_vram_gb
        parts.append(f"{name} ({effective_gb}GB effective)")
    else:
        parts.append("CPU only")

    # OS info
    if info.is_wsl:
        parts.append(f"WSL{info.wsl_version or ''}")
    else:
        parts.append(info.os_type)

    return " | ".join(parts)
