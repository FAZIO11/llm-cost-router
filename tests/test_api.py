"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import os
import tempfile

# Set test environment
os.environ["ROUTER_MODE"] = "mock"
os.environ["DATABASE_PATH"] = tempfile.mktemp(suffix=".db")

from llm_cost_router.api import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["router_mode"] == "mock"
    assert data["classifier_loaded"] is True
    assert "mock" in data["configured_providers"]


def test_route_endpoint_with_prompt(client):
    """Test routing with simple prompt."""
    response = client.post(
        "/v1/route",
        json={"prompt": "What is Python?", "run_eval": True},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "content" in data
    assert "routing_decision" in data
    assert "cost_usd" in data
    assert "savings_usd" in data
    
    # Check routing decision
    routing = data["routing_decision"]
    assert routing["complexity_tier"] in ["simple", "moderate", "complex"]
    assert routing["selected_model"]
    assert routing["selected_tier"] in ["local", "budget", "balanced", "frontier"]
    
    # Check evaluation
    if data.get("eval_scores"):
        eval_scores = data["eval_scores"]
        assert 0.0 <= eval_scores["quality_score"] <= 1.0


def test_route_endpoint_with_messages(client):
    """Test routing with message array."""
    response = client.post(
        "/v1/route",
        json={
            "messages": [
                {"role": "user", "content": "Explain binary search"}
            ],
            "run_eval": True,
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["content"]


def test_route_endpoint_force_tier(client):
    """Test forcing a specific tier."""
    response = client.post(
        "/v1/route",
        json={
            "prompt": "What is 2+2?",
            "force_tier": "frontier",
            "run_eval": False,
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    
    routing = data["routing_decision"]
    assert routing["selected_tier"] == "frontier"


def test_route_endpoint_local_only(client):
    """Test local-only constraint."""
    response = client.post(
        "/v1/route",
        json={
            "prompt": "Explain machine learning",
            "local_only": True,
            "run_eval": False,
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    
    routing = data["routing_decision"]
    assert routing["selected_tier"] == "local"


def test_route_endpoint_no_eval(client):
    """Test routing without evaluation."""
    response = client.post(
        "/v1/route",
        json={
            "prompt": "Hello",
            "run_eval": False,
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data.get("eval_scores") is None


def test_route_endpoint_missing_input(client):
    """Test error handling for missing input."""
    response = client.post("/v1/route", json={})
    assert response.status_code == 400


def test_chat_completions_endpoint(client):
    """Test OpenAI-compatible chat completions."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "What is Python?"}
            ],
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check OpenAI-compatible structure
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert len(data["choices"]) > 0
    
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"]
    
    # Check usage
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0
    
    # Check routing metadata
    assert "x_routing" in data
    assert "complexity_tier" in data["x_routing"]
    assert "savings_usd" in data["x_routing"]


def test_list_models_endpoint(client):
    """Test models listing endpoint."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0
    
    model = data["models"][0]
    assert "provider" in model
    assert "model" in model
    assert "tier" in model
    assert "input_price_per_1m" in model


def test_metrics_summary_endpoint(client):
    """Test metrics summary endpoint."""
    # First make some requests
    client.post(
        "/v1/route",
        json={"prompt": "Test 1", "run_eval": True},
    )
    client.post(
        "/v1/route",
        json={"prompt": "Test 2", "run_eval": True},
    )
    
    # Get metrics
    response = client.get("/v1/metrics/summary")
    assert response.status_code == 200
    
    data = response.json()
    assert data["total_requests"] >= 2
    assert "total_cost_usd" in data
    assert "baseline_cost_usd" in data
    assert "total_savings_usd" in data
    assert "avg_quality_score" in data


def test_logs_endpoint(client):
    """Test logs retrieval endpoint."""
    # Make a request first
    client.post(
        "/v1/route",
        json={"prompt": "Test log", "run_eval": False},
    )
    
    # Get logs
    response = client.get("/v1/logs?limit=10")
    assert response.status_code == 200
    
    data = response.json()
    assert "requests" in data
    assert len(data["requests"]) >= 1


def test_logs_endpoint_with_filter(client):
    """Test logs with tier filter."""
    # Make requests with different tiers
    client.post(
        "/v1/route",
        json={"prompt": "Simple", "force_tier": "local", "run_eval": False},
    )
    client.post(
        "/v1/route",
        json={"prompt": "Complex", "force_tier": "frontier", "run_eval": False},
    )
    
    # Filter by tier
    response = client.get("/v1/logs?tier=local&limit=10")
    assert response.status_code == 200
    
    data = response.json()
    for req in data["requests"]:
        assert req["tier"] == "local"


def test_cost_savings_calculation(client):
    """Test that cost savings are calculated correctly."""
    response = client.post(
        "/v1/route",
        json={"prompt": "What is Python?", "run_eval": False},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Savings should be positive (unless using frontier model)
    if data["routing_decision"]["selected_tier"] != "frontier":
        assert data["savings_usd"] > 0
        assert data["savings_percent"] > 0
    
    # Cost should be less than or equal to baseline
    assert data["cost_usd"] <= data["frontier_baseline_cost_usd"]


def test_complexity_classification(client):
    """Test that complexity is classified correctly."""
    test_cases = [
        ("What is 2+2?", ["simple", "moderate"]),
        ("Explain how binary search works", ["moderate", "complex"]),
        ("Design and implement a distributed system with fault tolerance", ["moderate", "complex"]),
    ]
    
    for prompt, expected_tiers in test_cases:
        response = client.post(
            "/v1/route",
            json={"prompt": prompt, "run_eval": False},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        routing = data["routing_decision"]
        assert routing["complexity_tier"] in expected_tiers


def test_concurrent_requests(client):
    """Test handling concurrent requests."""
    import concurrent.futures
    
    def make_request():
        return client.post(
            "/v1/route",
            json={"prompt": "Test", "run_eval": False},
        )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        results = [f.result() for f in futures]
    
    # All requests should succeed
    assert all(r.status_code == 200 for r in results)
