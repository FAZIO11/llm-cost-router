"""Mock provider for demo mode without API keys."""

import asyncio
import random
from typing import List, Dict

from .base import BaseProvider, ProviderResponse
from ..pricing import calculate_cost, MODEL_CATALOG


class MockProvider(BaseProvider):
    """Mock provider simulating realistic responses without API calls."""

    def __init__(self):
        """Initialize mock provider."""
        self.tier_responses = {
            "local": [
                "Here's a quick response to your question: {}",
                "Based on your input, here's what I think: {}",
                "Let me help with that. {}",
            ],
            "budget": [
                "I'll provide a detailed response to your query.\n\n{}",
                "Here's a comprehensive answer:\n\n{}",
                "Let me break this down for you:\n\n{}",
            ],
            "balanced": [
                "I'll provide a thorough and well-structured response.\n\n{}\n\nLet me elaborate further on the key points...",
                "Here's a comprehensive analysis:\n\n{}\n\nThis approach considers multiple factors...",
                "I'll address your question systematically:\n\n{}\n\nIn summary, the optimal approach would be...",
            ],
            "frontier": [
                "I'll provide an expert-level, deeply researched response.\n\n{}\n\nTo fully understand this topic, we need to consider:\n- Advanced theoretical foundations\n- Practical implementation details\n- Edge cases and optimization strategies\n\nIn conclusion, the most sophisticated approach involves...",
                "Here's a comprehensive, research-backed analysis:\n\n{}\n\nThis multifaceted solution considers:\n1. Theoretical framework\n2. Empirical evidence\n3. Best practices\n4. Future considerations\n\nBased on this thorough examination...",
            ],
        }

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> ProviderResponse:
        """Generate mock completion with realistic characteristics."""
        # Extract full model key
        if "/" not in model:
            # Find full model key
            for key in MODEL_CATALOG:
                if model in key:
                    model_key = key
                    break
            else:
                model_key = list(MODEL_CATALOG.keys())[0]
        else:
            model_key = model

        model_info = MODEL_CATALOG[model_key]
        tier = model_info.tier

        # Get user message
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        user_input = user_messages[-1] if user_messages else "your question"

        # Generate tier-appropriate response
        response_template = random.choice(self.tier_responses[tier])
        
        # Create content based on tier
        if tier == "local":
            core_content = f"The answer involves considering {user_input[:50]}..."
        elif tier == "budget":
            core_content = f"To address '{user_input[:60]}...', we should consider several factors and provide a structured solution."
        elif tier == "balanced":
            core_content = f"Regarding '{user_input[:60]}...', a balanced approach requires analyzing multiple dimensions and providing well-reasoned recommendations."
        else:  # frontier
            core_content = f"Your question about '{user_input[:60]}...' requires deep analysis across theoretical, practical, and strategic dimensions."

        content = response_template.format(core_content)

        # Simulate realistic token counts
        input_tokens = sum(len(m["content"].split()) for m in messages) * 1.3
        output_tokens = len(content.split()) * 1.3

        # Simulate realistic latency based on tier
        latency_base = {
            "local": (200, 800),
            "budget": (300, 1200),
            "balanced": (500, 1800),
            "frontier": (800, 2500),
        }
        
        latency_range = latency_base[tier]
        latency_ms = random.uniform(*latency_range)
        
        # Simulate network delay
        await asyncio.sleep(latency_ms / 1000)

        # Calculate cost
        cost = calculate_cost(int(input_tokens), int(output_tokens), model_key)

        return ProviderResponse(
            content=content,
            provider=model_info.provider,
            model=model_info.model,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            latency_ms=latency_ms,
            cost_usd=cost,
            finish_reason="stop",
        )

    def is_available(self) -> bool:
        """Mock provider is always available."""
        return True
