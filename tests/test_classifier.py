"""Tests for complexity classifier."""

import pytest
from llm_cost_router.classifier import ComplexityClassifier


@pytest.fixture
def trained_classifier():
    """Create a trained classifier."""
    classifier = ComplexityClassifier()
    
    # Simple training data
    prompts = [
        "What is 2+2?",
        "What color is the sky?",
        "Name three fruits",
        "Explain how binary search works with example code",
        "Compare Python and JavaScript for web development",
        "How do I implement pagination in an API?",
        "Design and implement a distributed rate limiting system that works across multiple servers with accurate enforcement, handles millions of requests per second, and provides detailed analytics",
        "Build a production-ready recommendation engine using collaborative filtering, content-based filtering, and hybrid approaches with cold-start handling",
        "Create a comprehensive observability platform that collects metrics, logs, and traces from distributed systems",
    ]
    
    labels = [
        "simple", "simple", "simple",
        "moderate", "moderate", "moderate",
        "complex", "complex", "complex",
    ]
    
    classifier.train(prompts, labels)
    return classifier


def test_classifier_train(trained_classifier):
    """Test classifier training."""
    assert trained_classifier.pipeline is not None
    assert trained_classifier.classes is not None
    assert len(trained_classifier.classes) == 3


def test_simple_prompt_classification(trained_classifier):
    """Test classification of simple prompt."""
    result = trained_classifier.predict("What is the capital of France?")
    
    assert result.tier in ["simple", "moderate"]
    assert 0.0 <= result.score <= 1.0
    assert result.score < 0.5  # Simple prompts should have lower scores
    assert "simple" in result.probabilities
    assert result.explanation


def test_moderate_prompt_classification(trained_classifier):
    """Test classification of moderate prompt."""
    result = trained_classifier.predict("Explain how JWT authentication works with example code")
    
    assert result.tier in ["moderate", "complex"]
    assert 0.0 <= result.score <= 1.0
    assert "moderate" in result.probabilities
    assert result.explanation


def test_complex_prompt_classification(trained_classifier):
    """Test classification of complex prompt."""
    result = trained_classifier.predict(
        "Design and implement a distributed caching system with consistent hashing, automatic sharding, "
        "replication, failover, cache warming, eviction policies, and monitoring"
    )
    
    assert result.tier in ["moderate", "complex"]
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0.4  # Complex prompts should have higher scores
    assert "complex" in result.probabilities
    assert result.explanation


def test_code_prompt_classification(trained_classifier):
    """Test classification of prompt with code."""
    result = trained_classifier.predict(
        "Write a function to implement LRU cache:\n```python\ndef lru_cache():\n    pass\n```"
    )
    
    assert result.tier in ["moderate", "complex"]
    assert 0.0 <= result.score <= 1.0
    assert "code" in result.explanation.lower() or "contains code" in result.explanation.lower()


def test_probability_sum(trained_classifier):
    """Test that probabilities sum to approximately 1.0."""
    result = trained_classifier.predict("Explain how DNS works")
    
    prob_sum = sum(result.probabilities.values())
    assert 0.99 <= prob_sum <= 1.01


def test_score_boundaries(trained_classifier):
    """Test that scores are always within [0.0, 1.0]."""
    test_prompts = [
        "Hi",
        "What is Python?",
        "Explain how microservices work",
        "Build a comprehensive production-grade distributed system with fault tolerance",
    ]
    
    for prompt in test_prompts:
        result = trained_classifier.predict(prompt)
        assert 0.0 <= result.score <= 1.0


def test_save_load(trained_classifier, tmp_path):
    """Test saving and loading classifier."""
    model_path = tmp_path / "test_model.joblib"
    
    # Save
    trained_classifier.save(model_path)
    assert model_path.exists()
    
    # Load
    new_classifier = ComplexityClassifier()
    success = new_classifier.load(model_path)
    assert success
    assert new_classifier.pipeline is not None
    
    # Test prediction
    result = new_classifier.predict("What is 2+2?")
    assert result.tier in ["simple", "moderate", "complex"]
