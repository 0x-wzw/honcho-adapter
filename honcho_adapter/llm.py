"""
LLM provider abstraction — adapted from Honcho's src/llm/ backend layer.

Supports OpenAI, Anthropic, and Gemini with a unified interface.
Extensible via the LLMBackend protocol.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ── Types ──────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Configuration for an LLM call."""
    model: str = "gpt-4o-mini"
    max_tokens: int = 4096
    temperature: float = 0.0
    max_input_tokens: int | None = None
    retry_attempts: int = 3


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    hit_input_token_cap: bool = False


# ── Provider Protocol ──────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    config: LLMConfig

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        response_model: type | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        ...


# ── OpenAI Provider ────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """OpenAI / compatible API provider."""

    def __init__(
        self,
        api_key: str,
        config: LLMConfig | None = None,
        base_url: str | None = None,
    ):
        self.config = config or LLMConfig()
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        except ImportError:
            raise ImportError("Install openai: pip install honcho-adapter[openai]")

    async def generate(
        self,
        prompt: str,
        response_model: type | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        response_format = {"type": "json_object"} if json_mode else None
        resp = await self._client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            response_format=response_format,
        )
        choice = resp.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            model=resp.model,
        )


# ── Anthropic Provider ─────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        try:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install honcho-adapter[anthropic]")

    async def generate(
        self,
        prompt: str,
        response_model: type | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        resp = await self._client.messages.create(
            model=kwargs.get("model", self.config.model),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            messages=[{"role": "user", "content": prompt}],
        )
        content = ""
        for block in resp.content:
            if hasattr(block, "text"):
                content += block.text
        return LLMResponse(
            content=content,
            input_tokens=resp.usage.input_tokens if resp.usage else 0,
            output_tokens=resp.usage.output_tokens if resp.usage else 0,
            model=resp.model,
        )


# ── Gemini Provider ────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(self, api_key: str, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
        except ImportError:
            raise ImportError("Install google-genai: pip install honcho-adapter[gemini]")

    async def generate(
        self,
        prompt: str,
        response_model: type | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        resp = await self._client.aio.models.generate_content(
            model=kwargs.get("model", self.config.model),
            contents=prompt,
            config={
                "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
            },
        )
        return LLMResponse(
            content=resp.text or "",
            model=resp.model_version or self.config.model,
        )


# ── Provider Registry ──────────────────────────────────────────────────────

class LLMBackend:
    """Unified LLM backend with provider selection and retry logic."""

    def __init__(self, provider: LLMProvider):
        self._provider = provider

    async def call(
        self,
        prompt: str,
        response_model: type | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        last_error: Exception | None = None
        config = self._provider.config
        attempts = kwargs.pop("retry_attempts", config.retry_attempts)

        for attempt in range(attempts):
            try:
                return await self._provider.generate(
                    prompt=prompt,
                    response_model=response_model,
                    json_mode=json_mode,
                    **kwargs,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt + 1, attempts, e,
                )
                if attempt < attempts - 1:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"LLM call failed after {attempts} attempts") from last_error


def create_backend(
    provider: str = "openai",
    api_key: str | None = None,
    config: LLMConfig | None = None,
    **kwargs: Any,
) -> LLMBackend:
    """Factory: create an LLMBackend from a provider name."""
    if api_key is None:
        import os
        api_key = os.environ.get(f"{provider.upper()}_API_KEY", "")
    if not api_key:
        raise ValueError(f"No API key for {provider}. Set {provider.upper()}_API_KEY")

    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
    }
    cls = providers.get(provider.lower())
    if cls is None:
        raise ValueError(f"Unknown provider: {provider}. Choose: {list(providers.keys())}")
    return LLMBackend(cls(api_key=api_key, config=config, **kwargs))