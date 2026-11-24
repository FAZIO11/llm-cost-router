"""Anthropic provider implementation."""

import time
from typing import List, Dict

from anthropic import AsyncAnthropic

from .base import BaseProvider, ProviderResponse
from ..pricing import calculate_cost


class AnthropicProvider(BaseProvider):
    """Anthropic API provider."""

    def __init__(self, api_key: str):
        """Initialize Anthropic provider."""
        self.api_key = api_key
        self.client = AsyncAnthropic(api_key=api_key) if api_key else None

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> ProviderResponse:
        """Generate completion using Anthropic API."""
        if not self.client:
            raise ValueError("Anthropic API key not configured")

        start_time = time.time()

        # Convert messages format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_message:
            kwargs["system"] = system_message

        response = await self.client.messages.create(**kwargs)

        latency_ms = (time.time() - start_time) * 1000

        content = "".join([block.text for block in response.content if hasattr(block, "text")])

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = calculate_cost(input_tokens, output_tokens, f"anthropic/{model}")

        return ProviderResponse(
            content=content,
            provider="anthropic",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            finish_reason=response.stop_reason or "stop",
            raw_response=response.model_dump(),
        )

    def is_available(self) -> bool:
        """Check if Anthropic provider is available."""
        return bool(self.api_key and self.client)
