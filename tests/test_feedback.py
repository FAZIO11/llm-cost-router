"""Tests for adaptive feedback loop."""

import pytest
from llm_cost_router.eval import FeedbackLoop
from llm_cost_router.storage import Database, RequestLog, EvaluationLog


@pytest.fixture
def test_db(tmp_path):
    """Create test database."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    yield db
    db.close()


@pytest.fixture
def feedback_loop(test_db):
    """Create feedback loop."""
    return FeedbackLoop(
        database=test_db,
        target_quality=0.78,
        window_size=20,
        adjustment_step=0.05,
    )


def test_feedback_loop_initialization(feedback_loop):
    """Test feedback loop initialization."""
    assert feedback_loop.simple_threshold == 0.35
    assert feedback_loop.moderate_threshold == 0.65
    assert feedback_loop.target_quality == 0.78


def test_get_thresholds(feedback_loop):
    """Test getting current thresholds."""
    thresholds = feedback_loop.get_thresholds()
    
    assert "simple_threshold" in thresholds
    assert "moderate_threshold" in thresholds
    assert thresholds["simple_threshold"] == 0.35
    assert thresholds["moderate_threshold"] == 0.65


def test_threshold_adjustment_on_low_quality(test_db, feedback_loop):
    """Test that thresholds are adjusted when quality is low."""
    # Create requests with low quality
    for i in range(25):
        request_id = test_db.log_request(
            RequestLog(
                prompt=f"Test {i}",
                complexity_tier="simple",
                complexity_score=0.3,
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
        
        # Log low quality evaluation
        test_db.log_evaluation(
            EvaluationLog(
                request_id=request_id,
                quality_score=0.70,  # Below target of 0.78
                passed=False,
                heuristic_score=0.70,
            )
        )
    
    # Initial threshold
    initial_threshold = feedback_loop.simple_threshold
    
    # Check and adjust
    thresholds = feedback_loop.check_and_adjust()
    
    # Threshold should be tightened (lowered)
    assert thresholds["simple_threshold"] < initial_threshold


def test_threshold_no_adjustment_on_good_quality(test_db, feedback_loop):
    """Test that thresholds are not adjusted when quality is good."""
    # Create requests with good quality
    for i in range(25):
        request_id = test_db.log_request(
            RequestLog(
                prompt=f"Test {i}",
                complexity_tier="simple",
                complexity_score=0.3,
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
        
        # Log good quality evaluation
        test_db.log_evaluation(
            EvaluationLog(
                request_id=request_id,
                quality_score=0.85,  # Above target
                passed=True,
                heuristic_score=0.85,
            )
        )
    
    # Initial threshold
    initial_threshold = feedback_loop.simple_threshold
    
    # Check and adjust
    thresholds = feedback_loop.check_and_adjust()
    
    # Threshold should remain the same
    assert thresholds["simple_threshold"] == initial_threshold


def test_threshold_adjustment_logged(test_db, feedback_loop):
    """Test that threshold adjustments are logged to database."""
    # Create low quality requests
    for i in range(25):
        request_id = test_db.log_request(
            RequestLog(
                prompt=f"Test {i}",
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
        
        test_db.log_evaluation(
            EvaluationLog(
                request_id=request_id,
                quality_score=0.72,  # Below target
                passed=False,
                heuristic_score=0.72,
            )
        )
    
    # Check and adjust
    feedback_loop.check_and_adjust()
    
    # Check if adjustment was logged
    adjustments = test_db.conn.execute(
        "SELECT * FROM threshold_adjustments"
    ).fetchall()
    
    assert len(adjustments) > 0


def test_threshold_minimum_bounds(test_db, feedback_loop):
    """Test that thresholds don't go below minimum bounds."""
    # Repeatedly adjust with low quality
    for _ in range(10):
        # Create low quality requests
        for i in range(25):
            request_id = test_db.log_request(
                RequestLog(
                    prompt=f"Test {i}",
                    complexity_tier="simple",
                    complexity_score=0.3,
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
            
            test_db.log_evaluation(
                EvaluationLog(
                    request_id=request_id,
                    quality_score=0.65,
                    passed=False,
                    heuristic_score=0.65,
                )
            )
        
        feedback_loop.check_and_adjust()
    
    # Threshold should not go below 0.20
    thresholds = feedback_loop.get_thresholds()
    assert thresholds["simple_threshold"] >= 0.20
    assert thresholds["moderate_threshold"] >= 0.50


def test_multiple_tier_adjustments(test_db, feedback_loop):
    """Test adjustments across multiple tiers."""
    # Create low quality requests for local tier
    for i in range(25):
        request_id = test_db.log_request(
            RequestLog(
                prompt=f"Local {i}",
                complexity_tier="simple",
                complexity_score=0.3,
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
        
        test_db.log_evaluation(
            EvaluationLog(
                request_id=request_id,
                quality_score=0.70,
                passed=False,
                heuristic_score=0.70,
            )
        )
    
    # Create low quality requests for budget tier
    for i in range(25):
        request_id = test_db.log_request(
            RequestLog(
                prompt=f"Budget {i}",
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
        
        test_db.log_evaluation(
            EvaluationLog(
                request_id=request_id,
                quality_score=0.72,
                passed=False,
                heuristic_score=0.72,
            )
        )
    
    initial_simple = feedback_loop.simple_threshold
    initial_moderate = feedback_loop.moderate_threshold
    
    feedback_loop.check_and_adjust()
    
    # Both thresholds should be adjusted
    assert feedback_loop.simple_threshold < initial_simple
    assert feedback_loop.moderate_threshold < initial_moderate
