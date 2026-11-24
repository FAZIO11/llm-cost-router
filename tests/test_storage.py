"""Tests for storage and database."""

import pytest
from pathlib import Path
from datetime import datetime

from llm_cost_router.storage import Database, RequestLog, EvaluationLog, ThresholdAdjustment


@pytest.fixture
def test_db(tmp_path):
    """Create test database."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    yield db
    db.close()


def test_database_initialization(test_db):
    """Test database initialization."""
    # Tables should be created
    tables = test_db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    
    table_names = [row[0] for row in tables]
    assert "requests" in table_names
    assert "evaluations" in table_names
    assert "threshold_adjustments" in table_names


def test_log_request(test_db):
    """Test logging a request."""
    log = RequestLog(
        timestamp=datetime.utcnow().isoformat(),
        prompt="What is Python?",
        complexity_tier="simple",
        complexity_score=0.3,
        selected_model="openai/gpt-4o-mini",
        provider="openai",
        tier="budget",
        response_content="Python is a programming language.",
        input_tokens=10,
        output_tokens=20,
        latency_ms=400,
        cost_usd=0.0001,
        frontier_baseline_cost_usd=0.001,
        savings_usd=0.0009,
        savings_percent=90.0,
    )
    
    request_id = test_db.log_request(log)
    assert request_id > 0
    
    # Verify retrieval
    requests = test_db.get_requests(limit=1)
    assert len(requests) == 1
    assert requests[0]["prompt"] == "What is Python?"
    assert requests[0]["cost_usd"] == 0.0001


def test_log_evaluation(test_db):
    """Test logging an evaluation."""
    # First create a request
    request_log = RequestLog(
        prompt="Test",
        complexity_tier="simple",
        complexity_score=0.3,
        selected_model="openai/gpt-4o-mini",
        provider="openai",
        tier="budget",
        response_content="Response",
        input_tokens=10,
        output_tokens=20,
        latency_ms=400,
        cost_usd=0.0001,
        frontier_baseline_cost_usd=0.001,
        savings_usd=0.0009,
        savings_percent=90.0,
    )
    request_id = test_db.log_request(request_log)
    
    # Log evaluation
    eval_log = EvaluationLog(
        request_id=request_id,
        timestamp=datetime.utcnow().isoformat(),
        quality_score=0.85,
        passed=True,
        heuristic_score=0.80,
        judge_score=0.90,
        judge_rationale="Good response",
    )
    
    eval_id = test_db.log_evaluation(eval_log)
    assert eval_id > 0
    
    # Verify retrieval
    evals = test_db.get_evaluations(limit=1)
    assert len(evals) == 1
    assert evals[0]["quality_score"] == 0.85
    assert evals[0]["passed"] == 1


def test_log_threshold_adjustment(test_db):
    """Test logging a threshold adjustment."""
    log = ThresholdAdjustment(
        timestamp=datetime.utcnow().isoformat(),
        tier="local",
        old_threshold=0.35,
        new_threshold=0.30,
        reason="Quality below target",
        rolling_quality_score=0.72,
    )
    
    adj_id = test_db.log_threshold_adjustment(log)
    assert adj_id > 0


def test_get_metrics_summary(test_db):
    """Test metrics summary calculation."""
    # Log some requests
    for i in range(5):
        test_db.log_request(
            RequestLog(
                prompt=f"Test {i}",
                complexity_tier="simple",
                complexity_score=0.3,
                selected_model="openai/gpt-4o-mini",
                provider="openai",
                tier="budget",
                response_content="Response",
                input_tokens=10,
                output_tokens=20,
                latency_ms=400,
                cost_usd=0.0001,
                frontier_baseline_cost_usd=0.001,
                savings_usd=0.0009,
                savings_percent=90.0,
            )
        )
    
    metrics = test_db.get_metrics_summary()
    
    assert metrics["total_requests"] == 5
    assert metrics["total_cost_usd"] == pytest.approx(0.0005, abs=1e-6)
    assert metrics["baseline_cost_usd"] == pytest.approx(0.005, abs=1e-6)
    assert metrics["total_savings_usd"] == pytest.approx(0.0045, abs=1e-6)
    assert metrics["savings_percent"] == pytest.approx(90.0, abs=0.1)


def test_get_requests_with_filter(test_db):
    """Test filtering requests by tier."""
    # Log requests with different tiers
    test_db.log_request(
        RequestLog(
            prompt="Test 1",
            complexity_tier="simple",
            complexity_score=0.2,
            selected_model="ollama/llama3.2:3b",
            provider="ollama",
            tier="local",
            response_content="Response",
            input_tokens=10,
            output_tokens=20,
            latency_ms=300,
            cost_usd=0.0,
            frontier_baseline_cost_usd=0.001,
            savings_usd=0.001,
            savings_percent=100.0,
        )
    )
    
    test_db.log_request(
        RequestLog(
            prompt="Test 2",
            complexity_tier="moderate",
            complexity_score=0.5,
            selected_model="openai/gpt-4o-mini",
            provider="openai",
            tier="budget",
            response_content="Response",
            input_tokens=10,
            output_tokens=20,
            latency_ms=400,
            cost_usd=0.0001,
            frontier_baseline_cost_usd=0.001,
            savings_usd=0.0009,
            savings_percent=90.0,
        )
    )
    
    # Filter by tier
    local_requests = test_db.get_requests(tier="local")
    assert len(local_requests) == 1
    assert local_requests[0]["tier"] == "local"
    
    budget_requests = test_db.get_requests(tier="budget")
    assert len(budget_requests) == 1
    assert budget_requests[0]["tier"] == "budget"


def test_get_rolling_quality_by_tier(test_db):
    """Test rolling quality calculation."""
    # Create request
    request_id = test_db.log_request(
        RequestLog(
            prompt="Test",
            complexity_tier="simple",
            complexity_score=0.3,
            selected_model="openai/gpt-4o-mini",
            provider="openai",
            tier="budget",
            response_content="Response",
            input_tokens=10,
            output_tokens=20,
            latency_ms=400,
            cost_usd=0.0001,
            frontier_baseline_cost_usd=0.001,
            savings_usd=0.0009,
            savings_percent=90.0,
        )
    )
    
    # Log evaluations
    for score in [0.8, 0.85, 0.75, 0.9]:
        test_db.log_evaluation(
            EvaluationLog(
                request_id=request_id,
                quality_score=score,
                passed=True,
                heuristic_score=score,
            )
        )
    
    rolling_quality = test_db.get_rolling_quality_by_tier("budget", window=10)
    assert 0.75 <= rolling_quality <= 0.9


def test_tier_distribution(test_db):
    """Test tier distribution in metrics."""
    # Log requests across different tiers
    tiers = ["local", "local", "budget", "budget", "budget", "balanced"]
    
    for tier in tiers:
        test_db.log_request(
            RequestLog(
                prompt="Test",
                complexity_tier="simple",
                complexity_score=0.3,
                selected_model=f"test/{tier}",
                provider="test",
                tier=tier,
                response_content="Response",
                input_tokens=10,
                output_tokens=20,
                latency_ms=400,
                cost_usd=0.0001,
                frontier_baseline_cost_usd=0.001,
                savings_usd=0.0009,
                savings_percent=90.0,
            )
        )
    
    metrics = test_db.get_metrics_summary()
    dist = metrics["tier_distribution"]
    
    assert dist["local"] == 2
    assert dist["budget"] == 3
    assert dist["balanced"] == 1


def test_database_persistence(tmp_path):
    """Test that database persists across connections."""
    db_path = tmp_path / "persist_test.db"
    
    # Create and write
    db1 = Database(db_path)
    db1.log_request(
        RequestLog(
            prompt="Test",
            complexity_tier="simple",
            complexity_score=0.3,
            selected_model="test/model",
            provider="test",
            tier="budget",
            response_content="Response",
            input_tokens=10,
            output_tokens=20,
            latency_ms=400,
            cost_usd=0.0001,
            frontier_baseline_cost_usd=0.001,
            savings_usd=0.0009,
            savings_percent=90.0,
        )
    )
    db1.close()
    
    # Reopen and read
    db2 = Database(db_path)
    requests = db2.get_requests()
    assert len(requests) == 1
    db2.close()
