"""Model pricing catalog and cost calculations."""

from dataclasses import dataclass
from typing import Literal

Tier = Literal["local", "budget", "balanced", "frontier"]


@dataclass
class ModelInfo:
    """Model information and pricing."""

    provider: str
    model: str
    tier: Tier
    input_price_per_1m: float  # USD per 1M input tokens
    output_price_per_1m: float  # USD per 1M output tokens
    context_window: int = 128000


# Model catalog with real pricing (as of 2024)
MODEL_CATALOG = {
    # OpenAI models
    "openai/gpt-4o-mini": ModelInfo(
        provider="openai",
        model="gpt-4o-mini",
        tier="budget",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
        context_window=128000,
    ),
    "openai/gpt-4o": ModelInfo(
        provider="openai",
        model="gpt-4o",
        tier="frontier",
        input_price_per_1m=2.50,
        output_price_per_1m=10.00,
        context_window=128000,
    ),
    # Anthropic models
    "anthropic/claude-3-5-haiku-latest": ModelInfo(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        tier="balanced",
        input_price_per_1m=0.80,
        output_price_per_1m=4.00,
        context_window=200000,
    ),
    "anthropic/claude-3-5-sonnet-latest": ModelInfo(
        provider="anthropic",
        model="claude-3-5-sonnet-latest",
        tier="frontier",
        input_price_per_1m=3.00,
        output_price_per_1m=15.00,
        context_window=200000,
    ),
    # Ollama local models (zero cost)
    "ollama/llama3.2:3b": ModelInfo(
        provider="ollama",
        model="llama3.2:3b",
        tier="local",
        input_price_per_1m=0.0,
        output_price_per_1m=0.0,
        context_window=128000,
    ),
    "ollama/qwen2.5:7b": ModelInfo(
        provider="ollama",
        model="qwen2.5:7b",
        tier="local",
        input_price_per_1m=0.0,
        output_price_per_1m=0.0,
        context_window=128000,
    ),
}


# Tier-ordered model preferences (cheapest to most expensive within tier)
TIER_MODELS = {
    "local": ["ollama/llama3.2:3b", "ollama/qwen2.5:7b"],
    "budget": ["openai/gpt-4o-mini"],
    "balanced": ["anthropic/claude-3-5-haiku-latest"],
    "frontier": ["openai/gpt-4o", "anthropic/claude-3-5-sonnet-latest"],
}


# Fallback chain for each model
FALLBACK_CHAINS = {
    "ollama/llama3.2:3b": ["ollama/qwen2.5:7b", "openai/gpt-4o-mini"],
    "ollama/qwen2.5:7b": ["ollama/llama3.2:3b", "openai/gpt-4o-mini"],
    "openai/gpt-4o-mini": ["anthropic/claude-3-5-haiku-latest", "openai/gpt-4o"],
    "anthropic/claude-3-5-haiku-latest": [
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet-latest",
    ],
    "openai/gpt-4o": ["anthropic/claude-3-5-sonnet-latest"],
    "anthropic/claude-3-5-sonnet-latest": ["openai/gpt-4o"],
}


def calculate_cost(input_tokens: int, output_tokens: int, model_key: str) -> float:
    """Calculate cost for a given token usage and model."""
    model_info = MODEL_CATALOG.get(model_key)
    if not model_info:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * model_info.input_price_per_1m
    output_cost = (output_tokens / 1_000_000) * model_info.output_price_per_1m
    return input_cost + output_cost


def calculate_frontier_baseline_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost using frontier baseline model (gpt-4o)."""
    return calculate_cost(input_tokens, output_tokens, "openai/gpt-4o")


def get_model_for_tier(tier: Tier) -> str:
    """Get the preferred model for a given tier."""
    return TIER_MODELS[tier][0]


def get_tier_for_complexity(
    complexity_score: float, simple_threshold: float = 0.35, moderate_threshold: float = 0.65
) -> Tier:
    """Map complexity score to tier based on thresholds."""
    if complexity_score < simple_threshold:
        return "local"
    elif complexity_score < moderate_threshold:
        return "budget"
    elif complexity_score < 0.85:
        return "balanced"
    else:
        return "frontier"
