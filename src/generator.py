"""
Instruction Generator using Distilabel and Local LLMs.

This module provides classes for generating synthetic instruction-response pairs
using various methods like Self-Instruct, Evol-Instruct, and Magpie.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, Optional
from dataclasses import dataclass, field

from distilabel.models import OllamaLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration, SelfInstruct

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from .roles import Role


@dataclass
class GenerationConfig:
    """Configuration for instruction generation.

    Attributes:
        model: The model name to use (e.g., "qwen2.5-coder:14b")
        temperature: Sampling temperature (0.0-1.5)
        max_tokens: Maximum tokens to generate per response
        ollama_base_url: Base URL for the Ollama API
        timeout: Request timeout in seconds
    """
    model: str = "qwen2.5-coder:14b"
    temperature: float = 0.7
    max_tokens: int = 1024
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    timeout: float = 120.0


class InstructionGenerator:
    """Generate instruction-response pairs using local LLMs.

    Supports multiple generation methods:
    - Self-Instruct: Generate diverse instructions from seed examples
    - Evol-Instruct: Evolve instructions to be more complex
    - Magpie: Use model-specific templates for natural instructions

    Attributes:
        config: Generation configuration settings
    """

    def __init__(self, config: GenerationConfig | None = None) -> None:
        """Initialize the generator with optional configuration.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or GenerationConfig()
    
    def _create_llm(self, model: str, temperature: float, max_tokens: int) -> OllamaLLM:
        """Create an Ollama LLM instance.

        Args:
            model: The model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Configured OllamaLLM instance
        """
        return OllamaLLM(
            model=model,
            host=self.config.ollama_base_url,
            timeout=int(self.config.timeout),
            generation_kwargs={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
    
    async def generate(
        self,
        seeds: list[str],
        model: str,
        temperature: float,
        max_tokens: int,
        num_instructions: int,
        method: str = "self_instruct",
        on_progress: Optional[Callable[[str], None]] = None,
        role: Optional["Role"] = None,
    ) -> list[dict]:
        """
        Generate instruction-response pairs.

        Args:
            seeds: List of seed instructions
            model: Model name (e.g., "qwen2.5-coder:14b")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            num_instructions: Target number of instructions to generate
            method: Generation method (self_instruct, evol_instruct, magpie)
            on_progress: Callback for progress updates
            role: Optional Role object for role-specific generation

        Returns:
            List of instruction-response pairs
        """

        if method == "self_instruct":
            return await self._generate_self_instruct(
                seeds, model, temperature, max_tokens, num_instructions, on_progress, role
            )
        elif method == "evol_instruct":
            return await self._generate_evol_instruct(
                seeds, model, temperature, max_tokens, num_instructions, on_progress, role
            )
        elif method == "magpie":
            return await self._generate_magpie(
                seeds, model, temperature, max_tokens, num_instructions, on_progress, role
            )
        else:
            raise ValueError(f"Unknown generation method: {method}")

    def _get_system_prompt_for_role(self, role: Optional["Role"]) -> str:
        """Get the appropriate system prompt for a role.

        Args:
            role: Optional Role object

        Returns:
            System prompt string
        """
        if role:
            return role.get_effective_system_prompt()

        # Default fallback system prompt
        return """You are a helpful technical assistant.
Answer questions precisely and with concrete code examples when appropriate.
Respond in the same language as the question was asked."""
    
    async def _generate_self_instruct(
        self,
        seeds: list[str],
        model: str,
        temperature: float,
        max_tokens: int,
        num_instructions: int,
        on_progress: Optional[Callable[[str], None]] = None,
        role: Optional["Role"] = None,
    ) -> list[dict]:
        """Generate using Self-Instruct method."""

        llm = self._create_llm(model, temperature, max_tokens)

        # Prepare seed data
        seed_data = [{"instruction": s} for s in seeds]

        # Get role-specific system prompt
        system_prompt = self._get_system_prompt_for_role(role)

        # Build criteria for instruction generation based on role
        if role:
            criteria = f"Generate diverse, specific instructions for a {role.display_name}. "
            criteria += f"Focus on: {', '.join(role.topics[:5])}. "
            criteria += "Include different task types: writing, creating, explaining, troubleshooting."
        else:
            criteria = "Generate diverse, specific technical instructions"

        # Self-Instruct pipeline
        with Pipeline(name="self-instruct-pipeline") as pipeline:
            load_seeds = LoadDataFromDicts(data=seed_data)

            # Generate new instructions based on seeds
            self_instruct = SelfInstruct(
                llm=llm,
                num_instructions=min(5, num_instructions // len(seeds)),  # Per seed
                criteria_for_query_generation=criteria,
            )

            # Generate responses for instructions
            generate_response = TextGeneration(
                llm=llm,
                system_prompt=system_prompt,
            )

            load_seeds >> self_instruct >> generate_response

        # Run pipeline
        if on_progress:
            role_info = f" for {role.display_name}" if role else ""
            on_progress(f"Starting Self-Instruct pipeline{role_info}...")
            on_progress(f"Loading {len(seed_data)} seed instructions...")

        distiset = pipeline.run()

        if on_progress:
            on_progress("Pipeline completed, extracting results...")

        # Extract results
        results = []
        if distiset and "default" in distiset:
            for row in distiset["default"]["train"]:
                if "instruction" in row and "generation" in row:
                    result = {
                        "instruction": row["instruction"],
                        "response": row["generation"],
                        "method": "self_instruct",
                        "model": model,
                    }
                    if role:
                        result["role"] = role.name
                        result["role_display_name"] = role.display_name
                    results.append(result)

        if on_progress:
            on_progress(f"Generated {len(results)} instruction-response pairs")

        return results[:num_instructions]
    
    async def _generate_evol_instruct(
        self,
        seeds: list[str],
        model: str,
        temperature: float,
        max_tokens: int,
        num_instructions: int,
        on_progress: Optional[Callable[[str], None]] = None,
        role: Optional["Role"] = None,
    ) -> list[dict]:
        """Generate using Evol-Instruct method (evolve instructions to be more complex)."""

        llm = self._create_llm(model, temperature, max_tokens)

        # Build evolution prompt with role context if available
        if role:
            role_context = f"This instruction is for a {role.display_name} working with {', '.join(role.topics[:5])}."
            EVOLUTION_PROMPT = f"""Given this instruction for a {role.display_name}, create a more complex and challenging version.
{role_context}

The evolved instruction should:
1. Add constraints or requirements relevant to a {role.display_name}
2. Require multi-step reasoning
3. Include edge cases or error handling
4. Be more specific about expected output format

Original instruction:
{{instruction}}

Evolved instruction (only output the new instruction, nothing else):"""
        else:
            EVOLUTION_PROMPT = """Given this instruction, create a more complex and challenging version.
The evolved instruction should:
1. Add constraints or requirements
2. Require multi-step reasoning
3. Include edge cases or error handling
4. Be more specific about expected output format

Original instruction:
{instruction}

Evolved instruction (only output the new instruction, nothing else):"""

        # Get role-specific system prompt for response generation
        system_prompt = self._get_system_prompt_for_role(role)

        results = []

        if on_progress:
            role_info = f" for {role.display_name}" if role else ""
            on_progress(f"Starting Evol-Instruct{role_info}...")
            on_progress(f"Will evolve {len(seeds)} seed instructions")

        for i, seed in enumerate(seeds):
            if on_progress:
                role_info = f" ({role.display_name})" if role else ""
                on_progress(f"Evolving instruction {i+1}/{len(seeds)}{role_info}")

            # Evolve the instruction multiple times
            current = seed
            evolutions_per_seed = max(1, num_instructions // len(seeds))

            for j in range(evolutions_per_seed):
                # Generate evolved instruction
                evolved_response = llm.generate([[{
                    "role": "user",
                    "content": EVOLUTION_PROMPT.format(instruction=current)
                }]])

                if evolved_response and evolved_response[0]:
                    evolved = evolved_response[0][0].strip()

                    # Generate response for evolved instruction with role-specific system prompt
                    response = llm.generate([[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": evolved}
                    ]])

                    if response and response[0]:
                        result = {
                            "instruction": evolved,
                            "response": response[0][0],
                            "original_seed": seed,
                            "evolution_depth": j + 1,
                            "method": "evol_instruct",
                            "model": model,
                        }
                        if role:
                            result["role"] = role.name
                            result["role_display_name"] = role.display_name
                        results.append(result)

                        if on_progress:
                            on_progress(f"Generated {len(results)} pairs (depth {j+1})")

                    current = evolved

        if on_progress:
            on_progress(f"Evol-Instruct completed: {len(results)} instruction-response pairs")

        return results[:num_instructions]
    
    async def _generate_magpie(
        self,
        seeds: list[str],
        model: str,
        temperature: float,
        max_tokens: int,
        num_instructions: int,
        on_progress: Optional[Callable[[str], None]] = None,
        role: Optional["Role"] = None,
    ) -> list[dict]:
        """
        Generate using Magpie method.
        Uses model-specific templates to generate natural instructions.
        """

        llm = self._create_llm(model, temperature, max_tokens)

        # Build Magpie system prompt based on role
        if role:
            topics_list = "\n".join(f"- {topic}" for topic in role.topics[:10])
            MAGPIE_SYSTEM = f"""You are an experienced {role.display_name}.
{role.description}

You ask precise, practical questions about topics such as:
{topics_list}"""

            instruction_prompt_template = """Based on this topic: "{topic_hint}"

Formulate a concrete, practical question that a {role_name} would ask.
Only the question, no answer:"""
        else:
            MAGPIE_SYSTEM = """You are an experienced developer and system administrator.
You ask precise technical questions about topics such as:
- DevOps and Infrastructure as Code
- Cloud Services (AWS, Azure, GCP)
- System Administration (Linux, Windows)
- Scripting and Automation
- Security and Compliance
- Databases and Data Engineering"""

            instruction_prompt_template = """Based on this topic: "{topic_hint}"

Formulate a concrete, technical question that a developer or admin would ask.
Only the question, no answer:"""

        # Get role-specific system prompt for response generation
        response_system_prompt = self._get_system_prompt_for_role(role)

        results = []

        if on_progress:
            role_info = f" for {role.display_name}" if role else ""
            on_progress(f"Starting Magpie generation{role_info}...")
            on_progress(f"Target: {num_instructions} instruction-response pairs")

        # Use seeds as topic hints
        topics = seeds * (num_instructions // len(seeds) + 1)

        for i, topic_hint in enumerate(topics[:num_instructions]):
            if on_progress and i % 5 == 0:
                role_info = f" for {role.display_name}" if role else ""
                on_progress(f"Generating {i+1}/{num_instructions}{role_info}...")

            # Generate instruction using Magpie approach
            if role:
                instruction_prompt = instruction_prompt_template.format(
                    topic_hint=topic_hint,
                    role_name=role.display_name
                )
            else:
                instruction_prompt = instruction_prompt_template.format(topic_hint=topic_hint)

            instruction_response = llm.generate([[{
                "role": "system",
                "content": MAGPIE_SYSTEM
            }, {
                "role": "user",
                "content": instruction_prompt
            }]])

            if instruction_response and instruction_response[0]:
                instruction = instruction_response[0][0].strip()

                # Generate response with role-specific system prompt
                response = llm.generate([[
                    {"role": "system", "content": response_system_prompt},
                    {"role": "user", "content": instruction}
                ]])

                if response and response[0]:
                    result = {
                        "instruction": instruction,
                        "response": response[0][0],
                        "topic_hint": topic_hint,
                        "method": "magpie",
                        "model": model,
                    }
                    if role:
                        result["role"] = role.name
                        result["role_display_name"] = role.display_name
                    results.append(result)

        if on_progress:
            on_progress(f"Magpie completed: {len(results)} instruction-response pairs")

        return results


# Alternative: Direct Ollama client without distilabel for simpler cases
class SimpleGenerator:
    """Simplified generator using direct Ollama API calls."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    async def generate_batch(
        self,
        instructions: list[str],
        model: str = "qwen2.5-coder:14b",
        system_prompt: str = "Du bist ein hilfreicher technischer Assistent.",
    ) -> list[dict]:
        """Generate responses for a batch of instructions."""
        
        import httpx
        
        results = []
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            for instruction in instructions:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": instruction,
                        "system": system_prompt,
                        "stream": False,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        "instruction": instruction,
                        "response": data.get("response", ""),
                        "model": model,
                    })
        
        return results
