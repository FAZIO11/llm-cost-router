#!/usr/bin/env python3
"""Load benchmark results into SQLite database for dashboard display."""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_cost_router.storage import Database, RequestLog, EvaluationLog


def load_benchmark_results(benchmark_path: Path, db_path: Path):
    """Load benchmark results JSON into SQLite database."""
    
    # Read benchmark results
    with open(benchmark_path) as f:
        results = json.load(f)
    
    # Initialize database
    db = Database(db_path)
    
    # Starting timestamp for smooth time-series visualization
    base_timestamp = datetime.fromisoformat("2026-11-24T04:17:00")
    
    # Load each prompt record with 15-second spacing
    for index, prompt_record in enumerate(results["prompts"]):
        # Extract routed data
        routed = prompt_record["routed"]
        
        # Calculate timestamp for this request (15 seconds apart)
        timestamp = base_timestamp + timedelta(seconds=15 * index)
        
        # Create request log
        request_log = RequestLog(
            timestamp=timestamp.isoformat(),
            prompt=prompt_record["prompt"][:200],  # Truncate for display
            complexity_tier=prompt_record["predicted_complexity"],
            complexity_score=prompt_record["complexity_score"],
            selected_model=prompt_record["selected_model"],
            provider=routed["model"].split("/")[0] if "/" in routed["model"] else "openai",
            tier=prompt_record["selected_tier"],
            response_content=routed["response_snippet"],
            input_tokens=routed["input_tokens"],
            output_tokens=routed["output_tokens"],
            latency_ms=routed["latency_ms"],
            cost_usd=routed["cost_usd"],
            frontier_baseline_cost_usd=prompt_record["baseline"]["cost_usd"],
            savings_usd=prompt_record["baseline"]["cost_usd"] - routed["cost_usd"],
            savings_percent=(
                (prompt_record["baseline"]["cost_usd"] - routed["cost_usd"]) 
                / prompt_record["baseline"]["cost_usd"] * 100
                if prompt_record["baseline"]["cost_usd"] > 0 else 0
            ),
        )
        
        request_id = db.log_request(request_log)
        
        # Create evaluation log
        eval_log = EvaluationLog(
            request_id=request_id,
            timestamp=timestamp.isoformat(),
            quality_score=routed["quality_score"],
            passed=routed["passed"],
            heuristic_score=routed["heuristic_score"],
            judge_score=routed.get("judge_score"),
            judge_rationale=routed.get("judge_rationale"),
            failure_reasons=", ".join(routed["failure_reasons"]) if routed.get("failure_reasons") else None,
        )
        
        db.log_evaluation(eval_log)
    
    db.close()
    
    print(f"✓ Loaded {len(results['prompts'])} benchmark records into {db_path}")
    print(f"  Total cost: ${results['summary']['routed_total_cost_usd']:.4f}")
    print(f"  Baseline cost: ${results['summary']['baseline_total_cost_usd']:.4f}")
    print(f"  Savings: {results['summary']['cost_reduction_pct']:.1f}%")


if __name__ == "__main__":
    benchmark_path = Path("docs/benchmark_results.json")
    db_path = Path("data/router.db")
    
    if not benchmark_path.exists():
        print(f"Error: {benchmark_path} not found")
        sys.exit(1)
    
    # Remove existing DB if present
    if db_path.exists():
        db_path.unlink()
    
    load_benchmark_results(benchmark_path, db_path)
