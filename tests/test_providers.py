"""Tests for LLM providers."""

import pytest
from llm_cost_router.providers import MockProvider, OpenAIProvider, AnthropicProvider


@pytest.mark.asyncio
async def test_mock_provider_simple():
    """Test mock provider with simple prompt."""
    provider = MockProvider()
    
    messages = [{"role": "user", "content": "What is 2+2?"}]
    response = await provider.complete(messages, "llama3.2:3b")
    
    assert response.content
    assert len(response.content) > 0
    assert response.provider == "ollama"
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms > 0
    assert response.cost_usd == 0.0  # Local models are free
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_mock_provider_budget():
    """Test mock provider with budget model."""
    provider = MockProvider()
    
    messages = [{"role": "user", "content": "Explain how binary search works"}]
    response = await provider.complete(messages, "gpt-4o-mini")
    
    assert response.content
    assert response.provider == "openai"
    assert response.cost_usd > 0  # Budget models have cost
    assert len(response.content) > len(messages[0]["content"])


@pytest.mark.asyncio
async def test_mock_provider_frontier():
    """Test mock provider with frontier model."""
    provider = MockProvider()
    
    messages = [{"role": "user", "content": "Design a distributed system"}]
    response = await provider.complete(messages, "gpt-4o")
    
    assert response.content
    assert response.provider == "openai"
    assert len(response.content) > 100  # Frontier models give longer responses


@pytest.mark.asyncio
async def test_mock_provider_latency_realistic():
    """Test that mock provider simulates realistic latency."""
    provider = MockProvider()
    
    messages = [{"role": "user", "content": "What is Python?"}]
    response = await provider.complete(messages, "llama3.2:3b")
    
    # Local models should be relatively fast
    assert 100 < response.latency_ms < 1500


@pytest.mark.asyncio
async def test_mock_provider_token_counts():
    """Test that mock provider returns reasonable token counts."""
    provider = MockProvider()
    
    messages = [{"role": "user", "content": "Explain recursion"}]
    response = await provider.complete(messages, "gpt-4o-mini")
    
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    # Output should be more than input for a typical response
    assert response.output_tokens >= response.input_tokens * 0.5


@pytest.mark.asyncio
async def test_mock_provider_tier_appropriate_responses():
    """Test that mock provider generates tier-appropriate response lengths."""
    provider = MockProvider()
    
    prompt = "Explain AI"
    messages = [{"role": "user", "content": prompt}]
    
    # Local tier
    local_response = await provider.complete(messages, "llama3.2:3b")
    
    # Frontier tier
    frontier_response = await provider.complete(messages, "gpt-4o")
    
    # Frontier should produce longer responses
    assert len(frontier_response.content) > len(local_response.content)


def test_openai_provider_initialization():
    """Test OpenAI provider initialization."""
    provider = OpenAIProvider(api_key="test-key")
    assert provider.api_key == "test-key"
    assert provider.client is not None


def test_openai_provider_no_key():
    """Test OpenAI provider without API key."""
    provider = OpenAIProvider(api_key="")
    assert not provider.is_available()


def test_anthropic_provider_initialization():
    """Test Anthropic provider initialization."""
    provider = AnthropicProvider(api_key="test-key")
    assert provider.api_key == "test-key"
    assert provider.client is not None


def test_anthropic_provider_no_key():
    """Test Anthropic provider without API key."""
    provider = AnthropicProvider(api_key="")
    assert not provider.is_available()


def test_mock_provider_always_available():
    """Test that mock provider is always available."""
    provider = MockProvider()
    assert provider.is_available()


@pytest.mark.asyncio
async def test_mock_provider_with_messages():
    """Test mock provider with multi-turn messages."""
    provider = MockProvider()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Tell me more."},
    ]
    
    response = await provider.complete(messages, "gpt-4o-mini")
    
    assert response.content
    assert response.input_tokens > 0
    # Input should account for all messages
    assert response.input_tokens > 10


@pytest.mark.asyncio
async def test_mock_provider_cost_calculation():
    """Test that mock provider calculates costs correctly."""
    provider = MockProvider()
    
    # Use a longer prompt to ensure meaningful token counts and cost differences
    messages = [{"role": "user", "content": "Explain how machine learning algorithms work in detail"}]
    
    # Budget model
    budget_response = await provider.complete(messages, "gpt-4o-mini")
    
    # Frontier model
    frontier_response = await provider.complete(messages, "gpt-4o")
    
    # Frontier should cost more (with longer prompts the difference is clearer)
    assert frontier_response.cost_usd >= budget_response.cost_usd
    
    # For longer responses, frontier typically costs significantly more
    if frontier_response.output_tokens > 50:
        assert frontier_response.cost_usd > budget_response.cost_usd
