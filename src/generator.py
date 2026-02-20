"""
LLM generation module with chain-of-thought financial reasoning.

Supports OpenAI and Anthropic Claude models. Constructs structured prompts
that instruct the LLM to:
1. Reason step-by-step through numerical computations.
2. Cite specific source passages for every factual claim.
3. Clearly distinguish between information found in the sources and any
   inferences or qualifications the model adds.

Also provides streaming response support for interactive applications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from src.config import LLMProvider, Settings
from src.vector_store import SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior financial analyst specializing in SEC 10-K filing analysis.
You answer questions using ONLY the provided source documents. Follow these rules strictly:

1. GROUNDING: Every factual claim MUST be supported by a specific source passage. If the sources
   do not contain enough information, say so explicitly rather than guessing.

2. CITATIONS: After each claim, cite the source using [Source N] notation, where N corresponds
   to the source number provided below.

3. NUMERICAL REASONING: When the question involves numbers (revenue, growth rates, ratios),
   show your computation step-by-step. Extract the exact figures from the sources, state them
   clearly, and perform any arithmetic explicitly.

4. STRUCTURE: Organize your response with clear paragraphs. For comparative or multi-part
   questions, use subheadings.

5. UNCERTAINTY: If the sources provide partial information, state what is available and what
   is missing. Never fabricate financial data.

6. CURRENCY & UNITS: Always include currency symbols and units (millions, billions, etc.)
   as stated in the source documents."""

USER_PROMPT_TEMPLATE = """Sources:
{sources}

Question: {query}

Provide a thorough, well-cited answer based on the sources above. Show your reasoning
for any numerical computations."""

SOURCE_TEMPLATE = """[Source {index}]
Section: {section}
Company: {company}
Filing Date: {filing_date}
Content:
{content}
---"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Output from the generation step.

    Attributes:
        answer: The generated answer text.
        sources: The source chunks used for generation.
        model: The model name used.
        prompt_tokens: Estimated prompt token count.
        completion_tokens: Estimated completion token count.
        raw_response: The raw API response object (for debugging).
    """

    answer: str = ""
    sources: list[SearchResult] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: Any = None


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


class OpenAIGenerator:
    """Generate answers using the OpenAI Chat Completions API.

    Args:
        api_key: OpenAI API key.
        model: Model name (e.g., ``gpt-4o``).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the completion.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Call the OpenAI API and return structured output.

        Returns:
            A dict with keys: answer, prompt_tokens, completion_tokens, raw.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0]
        usage = response.usage
        return {
            "answer": choice.message.content or "",
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "raw": response,
        }

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Generator[str, None, None]:
        """Stream tokens from the OpenAI API.

        Yields:
            Individual text tokens as they arrive.
        """
        stream = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


class AnthropicGenerator:
    """Generate answers using the Anthropic Messages API.

    Args:
        api_key: Anthropic API key.
        model: Model name (e.g., ``claude-3-5-sonnet-20241022``).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the completion.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Call the Anthropic API and return structured output.

        Returns:
            A dict with keys: answer, prompt_tokens, completion_tokens, raw.
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        answer = ""
        for block in response.content:
            if hasattr(block, "text"):
                answer += block.text

        return {
            "answer": answer,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "raw": response,
        }

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Generator[str, None, None]:
        """Stream tokens from the Anthropic API.

        Yields:
            Individual text tokens as they arrive.
        """
        with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text


# ---------------------------------------------------------------------------
# Generation pipeline
# ---------------------------------------------------------------------------


class FinancialGenerator:
    """Orchestrates prompt construction and LLM generation for financial QA.

    Formats retrieved source chunks into a structured prompt, calls the
    configured LLM backend, and returns a ``GenerationResult``.

    Args:
        settings: Application configuration.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._backend = self._init_backend(settings)
        self._model_name = settings.llm_model_name

    @staticmethod
    def _init_backend(
        settings: Settings,
    ) -> OpenAIGenerator | AnthropicGenerator:
        """Instantiate the appropriate LLM backend."""
        if settings.llm_provider == LLMProvider.OPENAI:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY required for OpenAI generation.")
            return OpenAIGenerator(
                api_key=settings.openai_api_key,
                model=settings.llm_model_name,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )

        if settings.llm_provider == LLMProvider.ANTHROPIC:
            if not settings.anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY required for Anthropic generation."
                )
            return AnthropicGenerator(
                api_key=settings.anthropic_api_key,
                model=settings.llm_model_name,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )

        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    def format_sources(self, results: list[SearchResult]) -> str:
        """Format search results into the sources block for the prompt.

        Args:
            results: Retrieved search results.

        Returns:
            Formatted string ready for inclusion in the user prompt.
        """
        source_blocks: list[str] = []
        for idx, result in enumerate(results, start=1):
            block = SOURCE_TEMPLATE.format(
                index=idx,
                section=result.metadata.get("section", "Unknown"),
                company=result.metadata.get("company_name", "Unknown"),
                filing_date=result.metadata.get("filing_date", "Unknown"),
                content=result.text[:3000],  # Truncate very long chunks
            )
            source_blocks.append(block)
        return "\n".join(source_blocks)

    def generate(
        self,
        query: str,
        results: list[SearchResult],
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """Generate an answer for the query using retrieved sources.

        Args:
            query: The user's question.
            results: Retrieved source chunks.
            system_prompt: Override the default system prompt.

        Returns:
            A ``GenerationResult`` with the answer and metadata.
        """
        sys_prompt = system_prompt or SYSTEM_PROMPT
        sources_text = self.format_sources(results)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            sources=sources_text, query=query
        )

        logger.info(
            "Generating answer with %s (sources=%d)",
            self._model_name,
            len(results),
        )

        try:
            output = self._backend.generate(sys_prompt, user_prompt)
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            return GenerationResult(
                answer=f"Generation failed: {exc}",
                sources=results,
                model=self._model_name,
            )

        return GenerationResult(
            answer=output["answer"],
            sources=results,
            model=self._model_name,
            prompt_tokens=output.get("prompt_tokens", 0),
            completion_tokens=output.get("completion_tokens", 0),
            raw_response=output.get("raw"),
        )

    def generate_stream(
        self,
        query: str,
        results: list[SearchResult],
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream the generated answer token by token.

        Args:
            query: The user's question.
            results: Retrieved source chunks.
            system_prompt: Override the default system prompt.

        Yields:
            Individual text tokens.
        """
        sys_prompt = system_prompt or SYSTEM_PROMPT
        sources_text = self.format_sources(results)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            sources=sources_text, query=query
        )

        logger.info(
            "Streaming answer with %s (sources=%d)",
            self._model_name,
            len(results),
        )

        yield from self._backend.generate_stream(sys_prompt, user_prompt)

    def simple_generate(self, prompt: str) -> str:
        """Utility method for simple prompts (e.g., query expansion).

        Args:
            prompt: A standalone prompt string.

        Returns:
            The generated text.
        """
        try:
            output = self._backend.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt=prompt,
            )
            return output["answer"]
        except Exception as exc:
            logger.error("Simple generation failed: %s", exc)
            return ""
