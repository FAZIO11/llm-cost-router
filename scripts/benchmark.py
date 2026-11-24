#!/usr/bin/env python3
"""Live OpenAI benchmark for llm-cost-router.

Classifies a held-out prompt suite, routes each prompt with the library's
tier and fallback rules (OpenAI-only in this run), compares the routed
model against an always-frontier gpt-4o baseline, and scores both with
the package Evaluator (heuristic + gpt-4o judge).

Usage (from the repo root, after installing package dependencies):

    python scripts/benchmark.py

Or, when the package is not installed, pass the source tree:

    python scripts/benchmark.py --repo-root . --model-path models/complexity_classifier.joblib

Requires OPENAI_API_KEY in the environment. The key is never printed,
logged, or written to the results file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

# ---------------------------------------------------------------------------
# Path bootstrap (kept free of machine-specific defaults)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def _maybe_add_repo_root(repo_root: Optional[Path]) -> Path:
    """Add the package root to sys.path if llm_cost_router is not installed."""
    if repo_root is not None:
        resolved = repo_root.resolve()
        sys.path.insert(0, str(resolved))
        return resolved

    cwd = Path.cwd()
    candidates = [cwd, SCRIPT_DIR, SCRIPT_DIR.parent]
    for candidate in candidates:
        if (candidate / "llm_cost_router").is_dir():
            sys.path.insert(0, str(candidate.resolve()))
            return candidate.resolve()
    return cwd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a live OpenAI benchmark of llm-cost-router routing."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Source tree containing the llm_cost_router package (optional if installed).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/complexity_classifier.joblib"),
        help="Path to the trained complexity classifier (default: models/complexity_classifier.joblib).",
    )
    parser.add_argument(
        "--prompts-path",
        type=Path,
        default=Path("data/benchmark_prompts.json"),
        help="JSON file of benchmark prompts (default: data/benchmark_prompts.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmark_results.json"),
        help="Where to write results JSON (default: docs/benchmark_results.json).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for generation calls (default: 0.3).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Max tokens for generation calls (default: 500).",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.75,
        help="Pass threshold forwarded to Evaluator (default: 0.75).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and route the suite without making any API calls.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Retry + cost-tracking provider wrapper
# ---------------------------------------------------------------------------

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
SNIPPET_CHARS = 200
USER_TZ = ZoneInfo("Europe/Rome")
FRONTIER_MODEL = "openai/gpt-4o"
JUDGE_MODEL = "gpt-4o"
AVAILABLE_PROVIDERS = {"openai"}


class TrackingProvider:
    """Wrap a provider with retries, purpose tagging, and cost accounting."""

    def __init__(self, inner: Any, max_attempts: int = 7) -> None:
        self.inner = inner
        self.max_attempts = max_attempts
        self.calls: List[Dict[str, Any]] = []
        self._purpose = "unknown"

    def is_available(self) -> bool:
        return self.inner.is_available()

    @contextmanager
    def purpose(self, name: str):
        previous = self._purpose
        self._purpose = name
        try:
            yield
        finally:
            self._purpose = previous

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Any:
        async def _once() -> Any:
            return await self.inner.complete(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        response = await retry_async(_once, max_attempts=self.max_attempts)
        self.calls.append(
            {
                "purpose": self._purpose,
                "model": f"openai/{response.model}" if not str(response.model).startswith("openai/") else response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "latency_ms": response.latency_ms,
                "finish_reason": response.finish_reason,
            }
        )
        return response


async def retry_async(
    fn: Callable[[], Awaitable[Any]],
    max_attempts: int = 7,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> Any:
    """Retry transient OpenAI / network errors with exponential backoff."""
    delay = base_delay
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except APIStatusError as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None)
            if status not in RETRYABLE_STATUS or attempt == max_attempts:
                raise
        except (APIConnectionError, APITimeoutError, RateLimitError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt == max_attempts:
                raise
        await asyncio.sleep(delay)
        delay = min(delay * 2, max_delay)
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Routing (mirrors llm_cost_router.api.app with a declared provider set)
# ---------------------------------------------------------------------------

def resolve_model(tier: str) -> tuple[str, bool]:
    """Pick a model for a routing tier using TIER_MODELS and FALLBACK_CHAINS."""
    from llm_cost_router.pricing import FALLBACK_CHAINS, MODEL_CATALOG, TIER_MODELS

    model_key = TIER_MODELS[tier][0]
    provider_name = MODEL_CATALOG[model_key].provider
    if provider_name in AVAILABLE_PROVIDERS:
        return model_key, False

    for fallback_key in FALLBACK_CHAINS.get(model_key, []):
        fallback_provider = MODEL_CATALOG[fallback_key].provider
        if fallback_provider in AVAILABLE_PROVIDERS:
            return fallback_key, True

    raise RuntimeError(
        f"No available provider for tier '{tier}' (preferred {model_key})."
    )


def model_api_name(model_key: str) -> str:
    from llm_cost_router.pricing import MODEL_CATALOG

    return MODEL_CATALOG[model_key].model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def response_metrics(response: Any) -> Dict[str, Any]:
    return {
        "model": f"openai/{response.model}" if not str(response.model).startswith("openai/") else response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_ms": response.latency_ms,
        "cost_usd": response.cost_usd,
        "finish_reason": response.finish_reason,
        "response_chars": len(response.content or ""),
        "response_words": len((response.content or "").split()),
        "response_snippet": snippet(response.content or ""),
    }


def eval_metrics(result: Any) -> Dict[str, Any]:
    return {
        "quality_score": result.quality_score,
        "passed": result.passed,
        "heuristic_score": result.heuristic_score,
        "judge_score": result.judge_score,
        "judge_rationale": result.judge_rationale,
        "failure_reasons": result.failure_reasons,
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def pct_change(old: float, new: float) -> float:
    """Return (old - new) / old * 100, or 0 if old is 0."""
    if old == 0:
        return 0.0
    return (old - new) / old * 100.0


def retention_pct(routed: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return routed / baseline * 100.0


async def evaluate_with_retry(
    evaluator: Any,
    provider: TrackingProvider,
    prompt: str,
    response: Any,
    complexity_tier: str,
    max_attempts: int = 4,
) -> Any:
    """Run Evaluator.evaluate, retrying if the judge call fails to produce a score."""
    last_result = None
    for attempt in range(1, max_attempts + 1):
        async with _async_null():
            with provider.purpose("judge"):
                result = await evaluator.evaluate(prompt, response, complexity_tier)
        last_result = result
        judge_ok = (
            result.judge_score is not None
            and result.judge_rationale not in (None, "Judge evaluation failed")
        )
        if judge_ok:
            return result
        if attempt < max_attempts:
            await asyncio.sleep(min(2 ** attempt, 16))
    return last_result


class _async_null:
    """No-op async context manager for readability."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def load_prompts(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        prompts = json.load(handle)
    if not isinstance(prompts, list) or not prompts:
        raise SystemExit(f"No prompts found in {path}")
    return prompts


def load_classifier(model_path: Path):
    from llm_cost_router.classifier import ComplexityClassifier

    classifier = ComplexityClassifier()
    if not classifier.load(model_path):
        raise SystemExit(f"Failed to load classifier from {model_path}")
    return classifier


def classify_suite(classifier, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from llm_cost_router.pricing import get_tier_for_complexity

    rows = []
    for item in prompts:
        prediction = classifier.predict(item["prompt"])
        selected_tier = get_tier_for_complexity(prediction.score)
        selected_model, fallback_used = resolve_model(selected_tier)
        rows.append(
            {
                "id": item["id"],
                "category": item.get("category", item.get("expected_tier")),
                "expected_tier": item.get("expected_tier", item.get("category")),
                "prompt": item["prompt"],
                "predicted_complexity": str(prediction.tier),
                "complexity_score": prediction.score,
                "probabilities": {str(k): float(v) for k, v in prediction.probabilities.items()},
                "explanation": prediction.explanation,
                "selected_tier": selected_tier,
                "selected_model": selected_model,
                "fallback_used": fallback_used,
            }
        )
    return rows


def print_preflight(rows: List[Dict[str, Any]]) -> None:
    from collections import Counter

    print("Pre-flight classification (zero API calls)")
    print("-" * 72)
    for row in rows:
        print(
            f"{row['id']:14s} exp={row['expected_tier']:8s} "
            f"pred={row['predicted_complexity']:8s} "
            f"score={row['complexity_score']:.3f} "
            f"route={row['selected_tier']:9s} "
            f"model={row['selected_model']} "
            f"fallback={row['fallback_used']}"
        )
    pred = Counter(r["predicted_complexity"] for r in rows)
    tiers = Counter(r["selected_tier"] for r in rows)
    models = Counter(r["selected_model"] for r in rows)
    print("-" * 72)
    print(f"predicted complexity: {dict(pred)}")
    print(f"routing tiers:        {dict(tiers)}")
    print(f"routed models:        {dict(models)}")
    print()


async def run_live(
    rows: List[Dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
    min_quality_score: float,
) -> Dict[str, Any]:
    from llm_cost_router.eval.evaluator import Evaluator
    from llm_cost_router.providers.openai_provider import OpenAIProvider

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    provider = TrackingProvider(OpenAIProvider(api_key))
    evaluator = Evaluator(
        judge_provider=provider,
        judge_model=JUDGE_MODEL,
        min_quality_score=min_quality_score,
        use_judge=True,
    )

    prompt_records: List[Dict[str, Any]] = []
    reused_frontier_calls = 0
    judge_calls_succeeded = 0
    judge_calls_total = 0

    total = len(rows)
    for index, row in enumerate(rows, 1):
        prompt = row["prompt"]
        messages = [{"role": "user", "content": prompt}]
        routed_model_key = row["selected_model"]
        print(
            f"[{index:02d}/{total}] {row['id']} -> {routed_model_key} "
            f"(score={row['complexity_score']:.3f}, tier={row['selected_tier']})",
            flush=True,
        )

        with provider.purpose("routed"):
            routed_response = await provider.complete(
                messages=messages,
                model=model_api_name(routed_model_key),
                temperature=temperature,
                max_tokens=max_tokens,
            )

        reused = routed_model_key == FRONTIER_MODEL
        if reused:
            reused_frontier_calls += 1
            baseline_response = routed_response
        else:
            with provider.purpose("baseline"):
                baseline_response = await provider.complete(
                    messages=messages,
                    model=model_api_name(FRONTIER_MODEL),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

        routed_eval = await evaluate_with_retry(
            evaluator,
            provider,
            prompt,
            routed_response,
            row["predicted_complexity"],
        )
        judge_calls_total += 1
        if routed_eval.judge_score is not None and routed_eval.judge_rationale != "Judge evaluation failed":
            judge_calls_succeeded += 1

        if reused:
            baseline_eval = routed_eval
        else:
            baseline_eval = await evaluate_with_retry(
                evaluator,
                provider,
                prompt,
                baseline_response,
                row["predicted_complexity"],
            )
            judge_calls_total += 1
            if (
                baseline_eval.judge_score is not None
                and baseline_eval.judge_rationale != "Judge evaluation failed"
            ):
                judge_calls_succeeded += 1

        record = {
            "id": row["id"],
            "category": row["category"],
            "expected_tier": row["expected_tier"],
            "prompt": prompt,
            "predicted_complexity": row["predicted_complexity"],
            "complexity_score": row["complexity_score"],
            "probabilities": row["probabilities"],
            "classification_explanation": row["explanation"],
            "selected_tier": row["selected_tier"],
            "selected_model": row["selected_model"],
            "fallback_used": row["fallback_used"],
            "baseline_reused_routed_call": reused,
            "routed": {**response_metrics(routed_response), **eval_metrics(routed_eval)},
            "baseline": {**response_metrics(baseline_response), **eval_metrics(baseline_eval)},
        }
        prompt_records.append(record)

    return {
        "prompt_records": prompt_records,
        "reused_frontier_calls": reused_frontier_calls,
        "judge_calls_succeeded": judge_calls_succeeded,
        "judge_calls_total": judge_calls_total,
        "provider_calls": provider.calls,
    }


def summarize(
    prompt_records: List[Dict[str, Any]],
    *,
    reused_frontier_calls: int,
    judge_calls_succeeded: int,
    judge_calls_total: int,
    provider_calls: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    from collections import Counter

    routed_cost = sum(r["routed"]["cost_usd"] for r in prompt_records)
    baseline_cost = sum(r["baseline"]["cost_usd"] for r in prompt_records)
    judge_cost = sum(c["cost_usd"] for c in provider_calls if c["purpose"] == "judge")
    total_spend = sum(c["cost_usd"] for c in provider_calls)

    routed_quality = [r["routed"]["quality_score"] for r in prompt_records]
    baseline_quality = [r["baseline"]["quality_score"] for r in prompt_records]
    routed_heuristic = [r["routed"]["heuristic_score"] for r in prompt_records]
    baseline_heuristic = [r["baseline"]["heuristic_score"] for r in prompt_records]
    routed_judge = [r["routed"]["judge_score"] for r in prompt_records if r["routed"]["judge_score"] is not None]
    baseline_judge = [r["baseline"]["judge_score"] for r in prompt_records if r["baseline"]["judge_score"] is not None]
    routed_pass = sum(1 for r in prompt_records if r["routed"]["passed"])
    baseline_pass = sum(1 for r in prompt_records if r["baseline"]["passed"])
    n = len(prompt_records)

    routed_latency = [r["routed"]["latency_ms"] for r in prompt_records]
    baseline_latency = [r["baseline"]["latency_ms"] for r in prompt_records]

    routed_avg_q = mean(routed_quality)
    baseline_avg_q = mean(baseline_quality)
    routed_avg_j = mean(routed_judge)
    baseline_avg_j = mean(baseline_judge)
    routed_avg_lat = mean(routed_latency)
    baseline_avg_lat = mean(baseline_latency)

    model_counts = dict(Counter(r["selected_model"] for r in prompt_records))
    tier_counts = dict(Counter(r["selected_tier"] for r in prompt_records))

    by_tier: Dict[str, Any] = {}
    for label in ("simple", "moderate", "complex"):
        group = [r for r in prompt_records if r["predicted_complexity"] == label]
        if not group:
            continue
        r_cost = sum(x["routed"]["cost_usd"] for x in group)
        b_cost = sum(x["baseline"]["cost_usd"] for x in group)
        r_q = mean(x["routed"]["quality_score"] for x in group)
        b_q = mean(x["baseline"]["quality_score"] for x in group)
        r_lat = mean(x["routed"]["latency_ms"] for x in group)
        b_lat = mean(x["baseline"]["latency_ms"] for x in group)
        by_tier[label] = {
            "count": len(group),
            "routed_cost_usd": r_cost,
            "baseline_cost_usd": b_cost,
            "cost_reduction_pct": pct_change(b_cost, r_cost),
            "routed_avg_quality": r_q,
            "baseline_avg_quality": b_q,
            "quality_retention_pct": retention_pct(r_q, b_q),
            "routed_avg_latency_ms": r_lat,
            "baseline_avg_latency_ms": b_lat,
        }

    now = datetime.now(USER_TZ)
    metadata = {
        "date": now.isoformat(timespec="seconds"),
        "timezone": "Europe/Rome",
        "total_prompts": n,
        "providers_active": ["openai"],
        "routed_models": model_counts,
        "baseline_model": FRONTIER_MODEL,
        "judge_model": f"openai/{JUDGE_MODEL}",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reused_frontier_calls": reused_frontier_calls,
        "judge_calls_succeeded": judge_calls_succeeded,
        "judge_calls_total": judge_calls_total,
        "judge_cost_usd": judge_cost,
        "estimated_total_benchmark_spend_usd": total_spend,
    }

    summary = {
        "routed_total_cost_usd": routed_cost,
        "baseline_total_cost_usd": baseline_cost,
        "cost_savings_usd": baseline_cost - routed_cost,
        "cost_reduction_pct": pct_change(baseline_cost, routed_cost),
        "routed_avg_quality": routed_avg_q,
        "baseline_avg_quality": baseline_avg_q,
        "quality_retention_pct": retention_pct(routed_avg_q, baseline_avg_q),
        "routed_avg_heuristic": mean(routed_heuristic),
        "baseline_avg_heuristic": mean(baseline_heuristic),
        "routed_avg_judge": routed_avg_j,
        "baseline_avg_judge": baseline_avg_j,
        "judge_retention_pct": retention_pct(routed_avg_j, baseline_avg_j),
        "routed_pass_rate_pct": routed_pass / n * 100.0 if n else 0.0,
        "baseline_pass_rate_pct": baseline_pass / n * 100.0 if n else 0.0,
        "routed_avg_latency_ms": routed_avg_lat,
        "baseline_avg_latency_ms": baseline_avg_lat,
        "latency_reduction_pct": pct_change(baseline_avg_lat, routed_avg_lat),
        "model_counts": model_counts,
        "tier_counts": tier_counts,
    }

    return {
        "metadata": metadata,
        "summary": summary,
        "by_complexity_tier": by_tier,
        "prompts": prompt_records,
    }


def assert_no_secrets(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "sk-" in text or "sk_" in text:
        raise SystemExit(f"Refusing to keep {path}: results appear to contain a secret token prefix")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _maybe_add_repo_root(args.repo_root)

    if not bool(os.environ.get("OPENAI_API_KEY")) and not args.dry_run:
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 1

    model_path = args.model_path
    if not model_path.exists():
        print(f"Classifier not found: {model_path}", file=sys.stderr)
        return 1

    prompts = load_prompts(args.prompts_path)
    classifier = load_classifier(model_path)
    rows = classify_suite(classifier, prompts)
    print_preflight(rows)

    if args.dry_run:
        print("Dry run complete; no API calls were made.")
        return 0

    live = asyncio.run(
        run_live(
            rows,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            min_quality_score=args.min_quality_score,
        )
    )
    results = summarize(
        live["prompt_records"],
        reused_frontier_calls=live["reused_frontier_calls"],
        judge_calls_succeeded=live["judge_calls_succeeded"],
        judge_calls_total=live["judge_calls_total"],
        provider_calls=live["provider_calls"],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert_no_secrets(output_path)

    summary = results["summary"]
    meta = results["metadata"]
    print()
    print("Benchmark complete")
    print(f"  wrote: {output_path.resolve()}")
    print(f"  routed_total_cost_usd:     {summary['routed_total_cost_usd']:.6f}")
    print(f"  baseline_total_cost_usd:   {summary['baseline_total_cost_usd']:.6f}")
    print(f"  cost_reduction_pct:        {summary['cost_reduction_pct']:.2f}")
    print(f"  quality_retention_pct:     {summary['quality_retention_pct']:.2f}")
    print(f"  judge success:             {meta['judge_calls_succeeded']}/{meta['judge_calls_total']}")
    print(f"  estimated total spend usd: {meta['estimated_total_benchmark_spend_usd']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
