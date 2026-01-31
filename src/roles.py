"""
Role-based Instruction Set Generator for Seedling.

This module provides functionality to automatically generate instruction sets
(description, topics, seeds) from role/profession names. It supports both
predefined roles and dynamic LLM-based generation for custom roles.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import yaml

# Type alias for role template
RoleTemplate = dict[str, Any]


@dataclass
class RoleConfig:
    """Configuration for role generation.

    Attributes:
        min_topics: Minimum number of topics to generate
        max_topics: Maximum number of topics to generate
        min_seeds: Minimum number of seed instructions to generate
        max_seeds: Maximum number of seed instructions to generate
        temperature: LLM temperature for generation
        max_tokens: Maximum tokens for LLM generation
        cache_enabled: Whether to cache generated roles
        cache_dir: Directory for role cache
        cache_expiration_days: Days until cache expires (0 = never)
    """

    min_topics: int = 8
    max_topics: int = 15
    min_seeds: int = 10
    max_seeds: int = 20
    temperature: float = 0.8
    max_tokens: int = 2048
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".cache/roles"))
    cache_expiration_days: int = 30


@dataclass
class Persona:
    """Defines how the AI assistant should behave and communicate.

    Attributes:
        tone: Communication tone (formal, professional, casual, friendly, technical)
        expertise_level: Level of expertise (beginner, intermediate, senior, expert)
        language: Primary response language (ISO 639-1 code)
        secondary_languages: Additional supported languages
        identity_statement: How the assistant should introduce itself
        traits: Key personality traits (max 5)
    """

    tone: str = "professional"
    expertise_level: str = "senior"
    language: str = "en"
    secondary_languages: list[str] = field(default_factory=list)
    identity_statement: str = ""
    traits: list[str] = field(default_factory=list)


@dataclass
class ResponseGuidelines:
    """Defines how responses should be structured and formatted.

    Attributes:
        include_code_examples: Include code examples when relevant
        include_explanations: Include step-by-step explanations
        format: Output format (markdown, plain_text, structured)
        length: Response length (concise, moderate, detailed)
        max_tokens: Maximum response length in tokens
        use_headers: Use headers in responses
        use_bullet_points: Use bullet points
        use_code_blocks: Use code blocks for code
        use_tables: Use tables when appropriate
    """

    include_code_examples: bool = True
    include_explanations: bool = True
    format: str = "markdown"
    length: str = "moderate"
    max_tokens: int = 1500
    use_headers: bool = True
    use_bullet_points: bool = True
    use_code_blocks: bool = True
    use_tables: bool = False


@dataclass
class Constraints:
    """Defines boundaries and limitations for the role.

    Attributes:
        forbidden: Hard boundaries - things the role must never do
        cautions: Soft boundaries - areas requiring caution
        knowledge_limits: Areas where knowledge is limited
    """

    forbidden: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    knowledge_limits: list[str] = field(default_factory=list)


@dataclass
class Example:
    """Example interaction demonstrating expected behavior.

    Attributes:
        instruction: User question or request
        response: Assistant response demonstrating desired format
        tags: Optional categorization tags
    """

    instruction: str
    response: str
    tags: list[str] = field(default_factory=list)


@dataclass
class Role:
    """Represents a role with its complete instruction set.

    Attributes:
        name: Internal role identifier
        display_name: Human-readable role name
        category: Role category for grouping
        description: Brief description of the role
        topics: List of relevant topics/tools
        seeds: List of seed instructions
        is_generated: Whether this role was dynamically generated
        persona: Communication and behavior settings
        system_prompt: Complete system prompt template
        response_guidelines: Response formatting guidelines
        constraints: Boundaries and limitations
        examples: Few-shot example interactions
    """

    name: str
    display_name: str
    category: str
    description: str
    topics: list[str]
    seeds: list[str]
    is_generated: bool = False
    # New instruction set attributes
    persona: Persona = field(default_factory=Persona)
    system_prompt: str = ""
    response_guidelines: ResponseGuidelines = field(default_factory=ResponseGuidelines)
    constraints: Constraints = field(default_factory=Constraints)
    examples: list[Example] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert role to dictionary format compatible with DOMAIN_TEMPLATES."""
        return {
            "description": self.description,
            "topics": self.topics,
            "seeds": self.seeds,
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Convert role to complete dictionary with all instruction set attributes."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "category": self.category,
            "description": self.description,
            "topics": self.topics,
            "seeds": self.seeds,
            "persona": {
                "tone": self.persona.tone,
                "expertise_level": self.persona.expertise_level,
                "language": self.persona.language,
                "secondary_languages": self.persona.secondary_languages,
                "identity_statement": self.persona.identity_statement,
                "traits": self.persona.traits,
            },
            "system_prompt": self.system_prompt,
            "response_guidelines": {
                "include_code_examples": self.response_guidelines.include_code_examples,
                "include_explanations": self.response_guidelines.include_explanations,
                "format": self.response_guidelines.format,
                "length": self.response_guidelines.length,
                "max_tokens": self.response_guidelines.max_tokens,
                "use_headers": self.response_guidelines.use_headers,
                "use_bullet_points": self.response_guidelines.use_bullet_points,
                "use_code_blocks": self.response_guidelines.use_code_blocks,
                "use_tables": self.response_guidelines.use_tables,
            },
            "constraints": {
                "forbidden": self.constraints.forbidden,
                "cautions": self.constraints.cautions,
                "knowledge_limits": self.constraints.knowledge_limits,
            },
            "examples": [
                {"instruction": ex.instruction, "response": ex.response, "tags": ex.tags}
                for ex in self.examples
            ],
        }

    def get_effective_system_prompt(self) -> str:
        """Generate the effective system prompt with placeholders resolved."""
        if self.system_prompt:
            return self.system_prompt.format(
                role_name=self.display_name,
                description=self.description,
                topics=", ".join(self.topics[:10]),
                language=self.persona.language,
            )
        # Default system prompt if none specified
        return self._generate_default_system_prompt()

    def _generate_default_system_prompt(self) -> str:
        """Generate a default system prompt based on role attributes."""
        lang_instruction = {
            "en": "Respond in English.",
            "de": "Antworte auf Deutsch.",
            "fr": "Répondez en français.",
            "es": "Responde en español.",
        }.get(self.persona.language, f"Respond in {self.persona.language}.")

        tone_instruction = {
            "formal": "Maintain a formal, professional tone.",
            "professional": "Be professional and clear.",
            "casual": "Be friendly and approachable.",
            "friendly": "Be warm and helpful.",
            "technical": "Use precise technical language.",
        }.get(self.persona.tone, "Be professional and clear.")

        expertise_instruction = {
            "beginner": "Explain concepts thoroughly, assuming limited prior knowledge.",
            "intermediate": "Provide balanced explanations suitable for practitioners.",
            "senior": "Provide expert-level insights with practical depth.",
            "expert": "Engage at an advanced level, assuming deep domain expertise.",
        }.get(self.persona.expertise_level, "Provide expert-level insights.")

        prompt = f"""You are an experienced {self.display_name}.

{self.description}

Your expertise includes: {", ".join(self.topics[:10])}.

Guidelines:
- {tone_instruction}
- {expertise_instruction}
- Provide accurate, actionable advice
- Include practical examples when appropriate
- Acknowledge limitations when uncertain
- {lang_instruction}"""

        # Add constraints if defined
        if self.constraints.forbidden:
            prompt += "\n\nImportant boundaries:"
            for constraint in self.constraints.forbidden[:5]:
                prompt += f"\n- {constraint}"

        return prompt

    @classmethod
    def from_dict(
        cls,
        name: str,
        data: dict[str, Any],
        is_generated: bool = False,
    ) -> "Role":
        """Create a Role from a dictionary."""
        # Parse persona if present
        persona_data = data.get("persona", {})
        persona = Persona(
            tone=persona_data.get("tone", "professional"),
            expertise_level=persona_data.get("expertise_level", "senior"),
            language=persona_data.get("language", "en"),
            secondary_languages=persona_data.get("secondary_languages", []),
            identity_statement=persona_data.get("identity_statement", ""),
            traits=persona_data.get("traits", []),
        )

        # Parse response guidelines if present
        guidelines_data = data.get("response_guidelines", {})
        response_guidelines = ResponseGuidelines(
            include_code_examples=guidelines_data.get("include_code_examples", True),
            include_explanations=guidelines_data.get("include_explanations", True),
            format=guidelines_data.get("format", "markdown"),
            length=guidelines_data.get("length", "moderate"),
            max_tokens=guidelines_data.get("max_tokens", 1500),
            use_headers=guidelines_data.get("use_headers", True),
            use_bullet_points=guidelines_data.get("use_bullet_points", True),
            use_code_blocks=guidelines_data.get("use_code_blocks", True),
            use_tables=guidelines_data.get("use_tables", False),
        )

        # Parse constraints if present
        constraints_data = data.get("constraints", {})
        constraints = Constraints(
            forbidden=constraints_data.get("forbidden", []),
            cautions=constraints_data.get("cautions", []),
            knowledge_limits=constraints_data.get("knowledge_limits", []),
        )

        # Parse examples if present
        examples_data = data.get("examples", [])
        examples = [
            Example(
                instruction=ex.get("instruction", ""),
                response=ex.get("response", ""),
                tags=ex.get("tags", []),
            )
            for ex in examples_data
        ]

        return cls(
            name=name,
            display_name=data.get("display_name", name.replace("_", " ").title()),
            category=data.get("category", "Custom"),
            description=data.get("description", ""),
            topics=data.get("topics", []),
            seeds=data.get("seeds", []),
            is_generated=is_generated,
            persona=persona,
            system_prompt=data.get("system_prompt", ""),
            response_guidelines=response_guidelines,
            constraints=constraints,
            examples=examples,
        )


class RoleManager:
    """Manages predefined and dynamically generated roles.

    This class handles:
    - Loading predefined roles from configuration
    - Dynamic role generation using LLMs
    - Caching of generated roles
    - Converting roles to domain templates
    """

    def __init__(
        self,
        config_path: Path | None = None,
        role_config: RoleConfig | None = None,
    ) -> None:
        """Initialize the RoleManager.

        Args:
            config_path: Path to roles.yaml configuration file
            role_config: Optional RoleConfig for generation settings
        """
        self.config_path = config_path or Path(__file__).parent.parent / "config" / "roles.yaml"
        self.role_config = role_config or RoleConfig()
        self._config: dict[str, Any] = {}
        self._predefined_roles: dict[str, Role] = {}
        self._categories: list[dict[str, str]] = []
        self._prompts: dict[str, str] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}

            # Load role generation settings
            gen_config = self._config.get("role_generation", {})
            self.role_config.min_topics = gen_config.get("min_topics", self.role_config.min_topics)
            self.role_config.max_topics = gen_config.get("max_topics", self.role_config.max_topics)
            self.role_config.min_seeds = gen_config.get("min_seeds", self.role_config.min_seeds)
            self.role_config.max_seeds = gen_config.get("max_seeds", self.role_config.max_seeds)
            self.role_config.temperature = gen_config.get("temperature", self.role_config.temperature)
            self.role_config.max_tokens = gen_config.get("max_tokens", self.role_config.max_tokens)

            # Load cache settings
            cache_config = self._config.get("cache", {})
            self.role_config.cache_enabled = cache_config.get("enabled", True)
            cache_dir = cache_config.get("directory", ".cache/roles")
            self.role_config.cache_dir = Path(cache_dir)
            self.role_config.cache_expiration_days = cache_config.get("expiration_days", 30)

            # Load prompts
            self._prompts = self._config.get("prompts", {})

            # Load categories
            self._categories = self._config.get("categories", [])

            # Load predefined roles
            predefined = self._config.get("predefined_roles", {})
            for name, data in predefined.items():
                self._predefined_roles[name] = Role.from_dict(name, data)

    def get_predefined_roles(self) -> dict[str, Role]:
        """Get all predefined roles.

        Returns:
            Dictionary mapping role names to Role objects
        """
        return self._predefined_roles.copy()

    def get_predefined_role(self, name: str) -> Role | None:
        """Get a specific predefined role by name.

        Args:
            name: Role identifier

        Returns:
            Role object if found, None otherwise
        """
        return self._predefined_roles.get(name)

    def get_categories(self) -> list[dict[str, str]]:
        """Get role categories for UI organization.

        Returns:
            List of category dictionaries with name and description
        """
        return self._categories.copy()

    def get_roles_by_category(self, category: str) -> list[Role]:
        """Get all roles in a specific category.

        Args:
            category: Category name to filter by

        Returns:
            List of roles in the category
        """
        return [
            role for role in self._predefined_roles.values()
            if role.category == category
        ]

    def get_role_choices(self) -> list[tuple[str, str]]:
        """Get role choices for UI dropdown.

        Returns:
            List of tuples (display_name, internal_name) for dropdown
        """
        choices = []
        for name, role in sorted(
            self._predefined_roles.items(),
            key=lambda x: (x[1].category, x[1].display_name),
        ):
            choices.append((f"[{role.category}] {role.display_name}", name))
        return choices

    def role_to_domain_template(self, role: Role) -> dict[str, Any]:
        """Convert a Role to domain template format.

        Args:
            role: Role to convert

        Returns:
            Dictionary in DOMAIN_TEMPLATES format
        """
        return role.to_dict()

    def get_all_as_domain_templates(self) -> dict[str, dict[str, Any]]:
        """Get all predefined roles as domain templates.

        Returns:
            Dictionary compatible with DOMAIN_TEMPLATES
        """
        return {
            role.display_name: role.to_dict()
            for role in self._predefined_roles.values()
        }

    # =========================================================================
    # Dynamic Role Generation
    # =========================================================================

    def _get_cache_path(self, role_name: str) -> Path:
        """Get cache file path for a role.

        Args:
            role_name: Name of the role

        Returns:
            Path to the cache file
        """
        # Create a safe filename from the role name
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", role_name.lower())
        hash_suffix = hashlib.md5(role_name.encode()).hexdigest()[:8]
        return self.role_config.cache_dir / f"{safe_name}_{hash_suffix}.json"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if a cache file is still valid.

        Args:
            cache_path: Path to the cache file

        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False

        if self.role_config.cache_expiration_days == 0:
            return True  # Never expires

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiration = mtime + timedelta(days=self.role_config.cache_expiration_days)
        return datetime.now() < expiration

    def _load_from_cache(self, role_name: str) -> Role | None:
        """Load a generated role from cache.

        Args:
            role_name: Name of the role to load

        Returns:
            Role if found in cache and valid, None otherwise
        """
        if not self.role_config.cache_enabled:
            return None

        cache_path = self._get_cache_path(role_name)
        if not self._is_cache_valid(cache_path):
            return None

        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            return Role.from_dict(role_name, data, is_generated=True)
        except (OSError, json.JSONDecodeError):
            return None

    def _save_to_cache(self, role: Role) -> None:
        """Save a generated role to cache.

        Args:
            role: Role to cache
        """
        if not self.role_config.cache_enabled:
            return

        cache_path = self._get_cache_path(role.name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "display_name": role.display_name,
            "category": role.category,
            "description": role.description,
            "topics": role.topics,
            "seeds": role.seeds,
            "generated_at": datetime.now().isoformat(),
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    async def generate_role(
        self,
        role_name: str,
        model: str = "qwen2.5-coder:14b",
        ollama_base_url: str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> Role:
        """Generate a complete role template from a role name using LLM.

        This method uses the configured LLM to generate:
        1. A description of the role
        2. Relevant topics and tools
        3. Seed instructions for training data generation

        Args:
            role_name: Name of the role/profession to generate
            model: LLM model to use for generation
            ollama_base_url: Base URL for Ollama API
            on_progress: Optional callback for progress updates

        Returns:
            Generated Role object
        """
        # Check cache first
        cached = self._load_from_cache(role_name)
        if cached:
            if on_progress:
                on_progress(f"Loaded '{role_name}' from cache")
            return cached

        if on_progress:
            on_progress(f"Generating instruction set for '{role_name}'...")

        base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Import here to avoid circular imports
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Step 1: Generate description
            if on_progress:
                on_progress("Generating role description...")

            description = await self._generate_description(
                client, base_url, model, role_name
            )

            # Step 2: Generate topics
            if on_progress:
                on_progress("Generating relevant topics...")

            num_topics = (self.role_config.min_topics + self.role_config.max_topics) // 2
            topics = await self._generate_topics(
                client, base_url, model, role_name, description, num_topics
            )

            # Step 3: Generate seed instructions
            if on_progress:
                on_progress("Generating seed instructions...")

            num_seeds = (self.role_config.min_seeds + self.role_config.max_seeds) // 2
            seeds = await self._generate_seeds(
                client, base_url, model, role_name, description, topics, num_seeds
            )

        # Create role object
        role = Role(
            name=role_name.lower().replace(" ", "_"),
            display_name=role_name,
            category="Custom",
            description=description,
            topics=topics,
            seeds=seeds,
            is_generated=True,
        )

        # Cache the generated role
        self._save_to_cache(role)

        if on_progress:
            on_progress(f"Successfully generated role '{role_name}' with {len(topics)} topics and {len(seeds)} seeds")

        return role

    async def _generate_description(
        self,
        client: Any,
        base_url: str,
        model: str,
        role_name: str,
    ) -> str:
        """Generate a description for the role."""
        prompt_template = self._prompts.get("description", "")
        if not prompt_template:
            prompt_template = (
                f"Briefly describe the role of a {role_name} in 1-2 sentences. "
                "Focus on their main responsibilities and expertise areas."
            )
        else:
            prompt_template = prompt_template.format(role_name=role_name)

        response = await client.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt_template,
                "stream": False,
                "options": {
                    "temperature": self.role_config.temperature,
                    "num_predict": 256,
                },
            },
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()
        return ""

    async def _generate_topics(
        self,
        client: Any,
        base_url: str,
        model: str,
        role_name: str,
        description: str,
        num_topics: int,
    ) -> list[str]:
        """Generate relevant topics for the role."""
        prompt_template = self._prompts.get("topics", "")
        if not prompt_template:
            prompt_template = (
                f"List {num_topics} important tools, technologies, and concepts "
                f"that a {role_name} works with daily. "
                "Be specific (e.g., 'PostgreSQL' not 'databases'). "
                "One item per line, no numbering."
            )
        else:
            prompt_template = prompt_template.format(
                role_name=role_name,
                description=description,
                num_topics=num_topics,
            )

        response = await client.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt_template,
                "stream": False,
                "options": {
                    "temperature": self.role_config.temperature,
                    "num_predict": 512,
                },
            },
        )

        if response.status_code == 200:
            data = response.json()
            text = data.get("response", "")
            # Parse topics from response
            topics = []
            for line in text.strip().split("\n"):
                line = line.strip()
                # Remove common prefixes like "- ", "* ", "1. ", etc.
                line = re.sub(r"^[-*\d.)\]]+\s*", "", line)
                if line and len(line) < 100:  # Sanity check
                    topics.append(line)
            return topics[:num_topics]
        return []

    async def _generate_seeds(
        self,
        client: Any,
        base_url: str,
        model: str,
        role_name: str,
        description: str,
        topics: list[str],
        num_seeds: int,
    ) -> list[str]:
        """Generate seed instructions for the role."""
        topics_str = ", ".join(topics[:10])  # Limit topics in prompt

        prompt_template = self._prompts.get("seeds", "")
        if not prompt_template:
            prompt_template = (
                f"Generate {num_seeds} diverse, realistic instructions that a {role_name} "
                f"would ask an AI assistant. Consider their work with: {topics_str}. "
                "Include different task types: writing, creating, explaining, troubleshooting. "
                "One instruction per line, no numbering."
            )
        else:
            prompt_template = prompt_template.format(
                role_name=role_name,
                description=description,
                topics=topics_str,
                num_seeds=num_seeds,
            )

        response = await client.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt_template,
                "stream": False,
                "options": {
                    "temperature": self.role_config.temperature,
                    "num_predict": self.role_config.max_tokens,
                },
            },
        )

        if response.status_code == 200:
            data = response.json()
            text = data.get("response", "")
            # Parse seeds from response
            seeds = []
            for line in text.strip().split("\n"):
                line = line.strip()
                # Remove common prefixes
                line = re.sub(r"^[-*\d.)\]]+\s*", "", line)
                # Valid seeds should be reasonably long
                if line and len(line) >= 20 and len(line) < 500:
                    seeds.append(line)
            return seeds[:num_seeds]
        return []


# =============================================================================
# Module-level convenience functions
# =============================================================================

_role_manager: RoleManager | None = None


def get_role_manager() -> RoleManager:
    """Get the global RoleManager instance.

    Returns:
        Singleton RoleManager instance
    """
    global _role_manager
    if _role_manager is None:
        _role_manager = RoleManager()
    return _role_manager


def get_predefined_roles() -> dict[str, Role]:
    """Get all predefined roles.

    Returns:
        Dictionary mapping role names to Role objects
    """
    return get_role_manager().get_predefined_roles()


def get_role_choices() -> list[tuple[str, str]]:
    """Get role choices for UI dropdown.

    Returns:
        List of tuples (display_name, internal_name) for dropdown
    """
    return get_role_manager().get_role_choices()


def get_role_seeds(role_name: str) -> list[str]:
    """Get seed instructions for a specific role.

    Args:
        role_name: Internal role identifier

    Returns:
        List of seed instructions, empty list if not found
    """
    role = get_role_manager().get_predefined_role(role_name)
    if role:
        return role.seeds
    return []


def get_role_as_domain_template(role_name: str) -> dict[str, Any] | None:
    """Get a role in domain template format.

    Args:
        role_name: Internal role identifier

    Returns:
        Dictionary in DOMAIN_TEMPLATES format, None if not found
    """
    role = get_role_manager().get_predefined_role(role_name)
    if role:
        return role.to_dict()
    return None


def get_all_roles_as_domain_templates() -> dict[str, dict[str, Any]]:
    """Get all predefined roles as domain templates.

    Returns:
        Dictionary compatible with DOMAIN_TEMPLATES
    """
    return get_role_manager().get_all_as_domain_templates()


async def generate_role_from_name(
    role_name: str,
    model: str = "qwen2.5-coder:14b",
    on_progress: Callable[[str], None] | None = None,
) -> Role:
    """Generate a complete role from just a role name.

    Args:
        role_name: Name of the role/profession
        model: LLM model to use
        on_progress: Optional progress callback

    Returns:
        Generated Role object
    """
    return await get_role_manager().generate_role(
        role_name=role_name,
        model=model,
        on_progress=on_progress,
    )
