"""Ollama local provider implementation."""

import time
from typing import List, Dict

import httpx

from .base import BaseProvider, ProviderResponse


class OllamaProvider(BaseProvider):
    """Ollama local LLM provider."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama provider."""
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> ProviderResponse:
        """Generate completion using Ollama API."""
        start_time = time.time()

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = await self.client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()

        data = response.json()
        latency_ms = (time.time() - start_time) * 1000

        content = data.get("message", {}).get("content", "")

        # Estimate token counts (Ollama doesn't always provide exact counts)
        input_tokens = data.get("prompt_eval_count", len(" ".join([m["content"] for m in messages])) // 4)
        output_tokens = data.get("eval_count", len(content) // 4)

        return ProviderResponse(
            content=content,
            provider="ollama",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=0.0,  # Local models are free
            finish_reason="stop",
            raw_response=data,
        )

    def is_available(self) -> bool:
        """Check if Ollama provider is available."""
        try:
            import httpx
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
