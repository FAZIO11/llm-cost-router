"""Tests for pricing and cost calculations."""

import pytest
from llm_cost_router.pricing import (
    calculate_cost,
    calculate_frontier_baseline_cost,
    get_model_for_tier,
    get_tier_for_complexity,
    MODEL_CATALOG,
    TIER_MODELS,
)


def test_calculate_cost_gpt4o_mini():
    """Test cost calculation for GPT-4o mini."""
    cost = calculate_cost(1000, 500, "openai/gpt-4o-mini")
    
    # 1000 input tokens = 0.001M * $0.15 = $0.00015
    # 500 output tokens = 0.0005M * $0.60 = $0.0003
    # Total = $0.00045
    expected = (1000 / 1_000_000 * 0.15) + (500 / 1_000_000 * 0.60)
    assert abs(cost - expected) < 0.000001


def test_calculate_cost_gpt4o():
    """Test cost calculation for GPT-4o."""
    cost = calculate_cost(1000, 500, "openai/gpt-4o")
    
    expected = (1000 / 1_000_000 * 2.50) + (500 / 1_000_000 * 10.00)
    assert abs(cost - expected) < 0.000001


def test_calculate_cost_ollama_free():
    """Test that Ollama models have zero cost."""
    cost = calculate_cost(1000, 500, "ollama/llama3.2:3b")
    assert cost == 0.0


def test_frontier_baseline_cost():
    """Test frontier baseline cost calculation."""
    baseline = calculate_frontier_baseline_cost(1000, 500)
    
    # Should use gpt-4o pricing
    expected = (1000 / 1_000_000 * 2.50) + (500 / 1_000_000 * 10.00)
    assert abs(baseline - expected) < 0.000001


def test_get_model_for_tier():
    """Test model selection for each tier."""
    assert "ollama" in get_model_for_tier("local")
    assert "gpt-4o-mini" in get_model_for_tier("budget")
    assert "haiku" in get_model_for_tier("balanced")
    assert get_model_for_tier("frontier") in ["openai/gpt-4o", "anthropic/claude-3-5-sonnet-latest"]


def test_get_tier_for_complexity_simple():
    """Test tier selection for simple complexity."""
    tier = get_tier_for_complexity(0.2, simple_threshold=0.35, moderate_threshold=0.65)
    assert tier == "local"


def test_get_tier_for_complexity_moderate():
    """Test tier selection for moderate complexity."""
    tier = get_tier_for_complexity(0.5, simple_threshold=0.35, moderate_threshold=0.65)
    assert tier == "budget"


def test_get_tier_for_complexity_balanced():
    """Test tier selection for balanced complexity."""
    tier = get_tier_for_complexity(0.75, simple_threshold=0.35, moderate_threshold=0.65)
    assert tier == "balanced"


def test_get_tier_for_complexity_frontier():
    """Test tier selection for frontier complexity."""
    tier = get_tier_for_complexity(0.9, simple_threshold=0.35, moderate_threshold=0.65)
    assert tier == "frontier"


def test_adaptive_thresholds():
    """Test that adaptive thresholds affect tier selection."""
    score = 0.33
    
    # With standard thresholds
    tier1 = get_tier_for_complexity(score, simple_threshold=0.35, moderate_threshold=0.65)
    assert tier1 == "local"
    
    # With tightened thresholds (escalate more)
    tier2 = get_tier_for_complexity(score, simple_threshold=0.30, moderate_threshold=0.65)
    assert tier2 == "budget"


def test_model_catalog_completeness():
    """Test that model catalog has required fields."""
    for model_key, info in MODEL_CATALOG.items():
        assert "/" in model_key
        assert info.provider
        assert info.model
        assert info.tier in ["local", "budget", "balanced", "frontier"]
        assert info.input_price_per_1m >= 0
        assert info.output_price_per_1m >= 0
        assert info.context_window > 0


def test_tier_models_valid():
    """Test that all tier models exist in catalog."""
    for tier, models in TIER_MODELS.items():
        for model_key in models:
            assert model_key in MODEL_CATALOG


def test_cost_savings_calculation():
    """Test typical cost savings scenario."""
    # Using budget model instead of frontier
    input_tokens = 1000
    output_tokens = 500
    
    frontier_cost = calculate_frontier_baseline_cost(input_tokens, output_tokens)
    budget_cost = calculate_cost(input_tokens, output_tokens, "openai/gpt-4o-mini")
    
    savings = frontier_cost - budget_cost
    savings_percent = (savings / frontier_cost * 100) if frontier_cost > 0 else 0
    
    assert savings > 0
    assert savings_percent > 80  # Budget model should save >80%
