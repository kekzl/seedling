"""
Instruction Generator using Distilabel and Local LLMs.

This module provides classes for generating synthetic instruction-response pairs
using various methods like Self-Instruct, Evol-Instruct, and Magpie.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable
from dataclasses import dataclass, field

from distilabel.models import OllamaLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration, SelfInstruct

if TYPE_CHECKING:
    from collections.abc import Awaitable


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
        on_progress: Optional[Callable[[str], None]] = None
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
            
        Returns:
            List of instruction-response pairs
        """
        
        if method == "self_instruct":
            return await self._generate_self_instruct(
                seeds, model, temperature, max_tokens, num_instructions, on_progress
            )
        elif method == "evol_instruct":
            return await self._generate_evol_instruct(
                seeds, model, temperature, max_tokens, num_instructions, on_progress
            )
        elif method == "magpie":
            return await self._generate_magpie(
                seeds, model, temperature, max_tokens, num_instructions, on_progress
            )
        else:
            raise ValueError(f"Unknown generation method: {method}")
    
    async def _generate_self_instruct(
        self,
        seeds: list[str],
        model: str,
        temperature: float,
        max_tokens: int,
        num_instructions: int,
        on_progress: Optional[Callable[[str], None]] = None
    ) -> list[dict]:
        """Generate using Self-Instruct method."""
        
        llm = self._create_llm(model, temperature, max_tokens)
        
        # Prepare seed data
        seed_data = [{"instruction": s} for s in seeds]
        
        # Self-Instruct pipeline
        with Pipeline(name="self-instruct-pipeline") as pipeline:
            load_seeds = LoadDataFromDicts(data=seed_data)
            
            # Generate new instructions based on seeds
            self_instruct = SelfInstruct(
                llm=llm,
                num_instructions=min(5, num_instructions // len(seeds)),  # Per seed
                criteria_for_query_generation="Generate diverse, specific technical instructions",
            )
            
            # Generate responses for instructions
            generate_response = TextGeneration(
                llm=llm,
                system_prompt="""Du bist ein hilfreicher technischer Assistent. 
Beantworte die Frage präzise und mit konkreten Code-Beispielen wenn angemessen.
Antworte auf Deutsch wenn die Frage auf Deutsch gestellt wurde.""",
            )
            
            load_seeds >> self_instruct >> generate_response
        
        # Run pipeline
        if on_progress:
            on_progress("Starting Self-Instruct pipeline...")
        
        distiset = pipeline.run()
        
        # Extract results
        results = []
        if distiset and "default" in distiset:
            for row in distiset["default"]["train"]:
                if "instruction" in row and "generation" in row:
                    results.append({
                        "instruction": row["instruction"],
                        "response": row["generation"],
                        "method": "self_instruct",
                        "model": model,
                    })
        
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
        on_progress: Optional[Callable[[str], None]] = None
    ) -> list[dict]:
        """Generate using Evol-Instruct method (evolve instructions to be more complex)."""
        
        llm = self._create_llm(model, temperature, max_tokens)
        
        EVOLUTION_PROMPT = """Given this instruction, create a more complex and challenging version.
The evolved instruction should:
1. Add constraints or requirements
2. Require multi-step reasoning
3. Include edge cases or error handling
4. Be more specific about expected output format

Original instruction:
{instruction}

Evolved instruction (only output the new instruction, nothing else):"""

        results = []
        
        for i, seed in enumerate(seeds):
            if on_progress:
                on_progress(f"Evolving instruction {i+1}/{len(seeds)}")
            
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
                    
                    # Generate response for evolved instruction
                    response = llm.generate([[{
                        "role": "user",
                        "content": evolved
                    }]])
                    
                    if response and response[0]:
                        results.append({
                            "instruction": evolved,
                            "response": response[0][0],
                            "original_seed": seed,
                            "evolution_depth": j + 1,
                            "method": "evol_instruct",
                            "model": model,
                        })
                    
                    current = evolved
        
        return results[:num_instructions]
    
    async def _generate_magpie(
        self,
        seeds: list[str],
        model: str,
        temperature: float,
        max_tokens: int,
        num_instructions: int,
        on_progress: Optional[Callable[[str], None]] = None
    ) -> list[dict]:
        """
        Generate using Magpie method.
        Uses model-specific templates to generate natural instructions.
        """
        
        llm = self._create_llm(model, temperature, max_tokens)
        
        # Magpie-style: Let the model complete a user turn
        # This produces more natural, diverse instructions
        
        MAGPIE_SYSTEM = """Du bist ein erfahrener Entwickler und System-Administrator.
Du stellst präzise technische Fragen zu Themen wie:
- DevOps und Infrastructure as Code
- Cloud Services (AWS, Azure, GCP)
- System Administration (Linux, Windows)
- Scripting und Automation
- Security und Compliance
- Datenbanken und Data Engineering"""

        results = []
        
        # Use seeds as topic hints
        topics = seeds * (num_instructions // len(seeds) + 1)
        
        for i, topic_hint in enumerate(topics[:num_instructions]):
            if on_progress and i % 10 == 0:
                on_progress(f"Generating {i+1}/{num_instructions}...")
            
            # Generate instruction using Magpie approach
            # The model completes what a user would ask
            instruction_prompt = f"""Basierend auf diesem Thema: "{topic_hint}"

Formuliere eine konkrete, technische Frage die ein Entwickler oder Admin stellen würde.
Nur die Frage, keine Antwort:"""

            instruction_response = llm.generate([[{
                "role": "system",
                "content": MAGPIE_SYSTEM
            }, {
                "role": "user", 
                "content": instruction_prompt
            }]])
            
            if instruction_response and instruction_response[0]:
                instruction = instruction_response[0][0].strip()
                
                # Generate response
                response = llm.generate([[{
                    "role": "user",
                    "content": instruction
                }]])
                
                if response and response[0]:
                    results.append({
                        "instruction": instruction,
                        "response": response[0][0],
                        "topic_hint": topic_hint,
                        "method": "magpie",
                        "model": model,
                    })
        
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
