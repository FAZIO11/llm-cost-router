"""Base provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ProviderResponse:
    """Standardized response from an LLM provider."""

    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    finish_reason: str = "stop"
    raw_response: Optional[Dict[str, Any]] = None


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> ProviderResponse:
        """
        Generate a completion from the provider.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            ProviderResponse with completion details
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available."""
        pass
