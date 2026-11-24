"""OpenAI provider implementation."""

import time
from typing import List, Dict

from openai import AsyncOpenAI

from .base import BaseProvider, ProviderResponse
from ..pricing import calculate_cost


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str):
        """Initialize OpenAI provider."""
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> ProviderResponse:
        """Generate completion using OpenAI API."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        start_time = time.time()

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latency_ms = (time.time() - start_time) * 1000

        choice = response.choices[0]
        usage = response.usage

        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        cost = calculate_cost(input_tokens, output_tokens, f"openai/{model}")

        return ProviderResponse(
            content=choice.message.content or "",
            provider="openai",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            finish_reason=choice.finish_reason or "stop",
            raw_response=response.model_dump(),
        )

    def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        return bool(self.api_key and self.client)
