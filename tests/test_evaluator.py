"""Tests for response evaluation."""

import pytest
from llm_cost_router.eval import Evaluator
from llm_cost_router.eval.evaluator import MockJudge
from llm_cost_router.providers.base import ProviderResponse


@pytest.fixture
def evaluator():
    """Create evaluator without judge."""
    return Evaluator(judge_provider=None, use_judge=False)


@pytest.mark.asyncio
async def test_heuristic_good_response(evaluator):
    """Test heuristic evaluation of good response."""
    prompt = "What is Python?"
    response = ProviderResponse(
        content="Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=10,
        output_tokens=50,
        latency_ms=500,
        cost_usd=0.0001,
    )
    
    result = await evaluator.evaluate(prompt, response, "simple")
    
    assert result.heuristic_score > 0.7
    assert result.quality_score > 0.7
    assert len(result.failure_reasons or []) == 0


@pytest.mark.asyncio
async def test_heuristic_empty_response(evaluator):
    """Test heuristic evaluation of empty response."""
    prompt = "What is Python?"
    response = ProviderResponse(
        content="",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=10,
        output_tokens=0,
        latency_ms=100,
        cost_usd=0.0,
    )
    
    result = await evaluator.evaluate(prompt, response, "simple")
    
    assert result.heuristic_score < 0.5
    assert "too short or empty" in " ".join(result.failure_reasons or []).lower()


@pytest.mark.asyncio
async def test_heuristic_code_requested(evaluator):
    """Test heuristic evaluation when code is requested."""
    prompt = "Write a function to reverse a string"
    
    # Response with code
    good_response = ProviderResponse(
        content="Here's a function to reverse a string:\n```python\ndef reverse_string(s):\n    return s[::-1]\n```",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=15,
        output_tokens=30,
        latency_ms=400,
        cost_usd=0.0001,
    )
    
    result = await evaluator.evaluate(prompt, good_response, "moderate")
    assert result.heuristic_score >= 0.8
    
    # Response without code
    bad_response = ProviderResponse(
        content="To reverse a string, you can use various methods.",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=15,
        output_tokens=20,
        latency_ms=300,
        cost_usd=0.0001,
    )
    
    result = await evaluator.evaluate(prompt, bad_response, "moderate")
    assert result.heuristic_score < 0.8
    assert "code requested but not provided" in " ".join(result.failure_reasons or []).lower()


@pytest.mark.asyncio
async def test_heuristic_refusal_pattern(evaluator):
    """Test heuristic evaluation of refusal."""
    prompt = "Explain AI"
    response = ProviderResponse(
        content="I cannot help with that request.",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=10,
        output_tokens=10,
        latency_ms=200,
        cost_usd=0.0001,
    )
    
    result = await evaluator.evaluate(prompt, response, "simple")
    
    assert result.heuristic_score < 0.7
    assert any("refusal" in (reason or "").lower() for reason in (result.failure_reasons or []))


@pytest.mark.asyncio
async def test_heuristic_length_vs_complexity(evaluator):
    """Test that response length is evaluated against complexity."""
    prompt = "Design a distributed system with fault tolerance, load balancing, and monitoring"
    
    # Short response for complex prompt
    short_response = ProviderResponse(
        content="Use microservices.",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=20,
        output_tokens=5,
        latency_ms=200,
        cost_usd=0.0001,
    )
    
    result = await evaluator.evaluate(prompt, short_response, "complex")
    
    assert result.heuristic_score < 0.9
    assert any("inadequate" in (reason or "").lower() for reason in (result.failure_reasons or []))


@pytest.mark.asyncio
async def test_mock_judge():
    """Test mock judge evaluation."""
    judge = MockJudge()
    
    # Simple tier
    score, rationale = await judge.evaluate_mock(
        "What is 2+2?",
        "The answer is 4.",
        "local",
    )
    
    assert 0.0 <= score <= 1.0
    assert rationale
    assert "local" in rationale.lower()
    
    # Frontier tier with long response
    long_response = "This is a very comprehensive response. " * 20
    score, rationale = await judge.evaluate_mock(
        "Explain AI",
        long_response,
        "frontier",
    )
    
    assert score > 0.85  # Frontier tier with good length should score high


@pytest.mark.asyncio
async def test_quality_pass_fail(evaluator):
    """Test quality pass/fail determination."""
    # Good response
    good_response = ProviderResponse(
        content="Python is a programming language known for its simplicity and powerful libraries.",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=10,
        output_tokens=20,
        latency_ms=400,
        cost_usd=0.0001,
    )
    
    result = await evaluator.evaluate("What is Python?", good_response, "simple")
    assert result.passed
    
    # Bad response (empty)
    bad_response = ProviderResponse(
        content="",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=10,
        output_tokens=0,
        latency_ms=100,
        cost_usd=0.0,
    )
    
    result = await evaluator.evaluate("What is Python?", bad_response, "simple")
    assert not result.passed


@pytest.mark.asyncio
async def test_json_validation(evaluator):
    """Test JSON validation in responses."""
    prompt = "Return a JSON object with name and age"
    
    # Valid JSON
    good_response = ProviderResponse(
        content='Here is the JSON:\n```json\n{"name": "John", "age": 30}\n```',
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=15,
        output_tokens=25,
        latency_ms=400,
        cost_usd=0.0001,
    )
    
    result = await evaluator.evaluate(prompt, good_response, "simple")
    assert result.heuristic_score > 0.8
    
    # Invalid JSON
    bad_response = ProviderResponse(
        content='Here is the JSON:\n```json\n{name: "John", age: 30}\n```',
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=15,
        output_tokens=25,
        latency_ms=400,
        cost_usd=0.0001,
    )
    
    result = await evaluator.evaluate(prompt, bad_response, "simple")
    assert result.heuristic_score < 0.9


@pytest.mark.asyncio
async def test_evaluation_score_range(evaluator):
    """Test that evaluation scores are always in [0, 1]."""
    test_cases = [
        ("", "empty"),
        ("Hi", "short"),
        ("This is a reasonable response.", "normal"),
        ("I cannot help with that.", "refusal"),
    ]
    
    for content, desc in test_cases:
        response = ProviderResponse(
            content=content,
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=10,
            output_tokens=10,
            latency_ms=300,
            cost_usd=0.0001,
        )
        
        result = await evaluator.evaluate("Test prompt", response, "simple")
        
        assert 0.0 <= result.quality_score <= 1.0, f"Failed for {desc}"
        assert 0.0 <= result.heuristic_score <= 1.0, f"Failed for {desc}"
