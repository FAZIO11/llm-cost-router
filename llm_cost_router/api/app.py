"""FastAPI application for LLM Cost Router."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import Settings
from ..pricing import (
    MODEL_CATALOG,
    TIER_MODELS,
    FALLBACK_CHAINS,
    calculate_frontier_baseline_cost,
    get_tier_for_complexity,
)
from ..classifier import ComplexityClassifier
from ..providers import (
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    MockProvider,
)
from ..eval import Evaluator, FeedbackLoop
from ..eval.evaluator import MockJudge
from ..storage import Database, RequestLog, EvaluationLog, JSONLLogger


# Request/Response models
class Message(BaseModel):
    role: str
    content: str


class RouteRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[Message]] = None
    max_cost_usd: Optional[float] = None
    min_quality: Optional[float] = None
    local_only: bool = False
    force_tier: Optional[Literal["local", "budget", "balanced", "frontier"]] = None
    run_eval: bool = True


class RoutingDecision(BaseModel):
    complexity_tier: str
    complexity_score: float
    complexity_explanation: str
    selected_model: str
    selected_tier: str
    provider: str
    fallback_used: bool = False


class EvalScores(BaseModel):
    quality_score: float
    passed: bool
    heuristic_score: float
    judge_score: Optional[float]
    judge_rationale: Optional[str]


class RouteResponse(BaseModel):
    content: str
    routing_decision: RoutingDecision
    cost_usd: float
    input_tokens: int
    output_tokens: int
    latency_ms: float
    frontier_baseline_cost_usd: float
    savings_usd: float
    savings_percent: float
    eval_scores: Optional[EvalScores] = None


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 2000
    temperature: Optional[float] = 0.7


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    x_routing: Dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    provider: str
    model: str
    tier: str
    input_price_per_1m: float
    output_price_per_1m: float


class HealthResponse(BaseModel):
    status: str
    router_mode: str
    classifier_loaded: bool
    configured_providers: List[str]


# Global state
app_state = {
    "settings": None,
    "classifier": None,
    "providers": {},
    "database": None,
    "jsonl_logger": None,
    "evaluator": None,
    "feedback_loop": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application state."""
    # Initialize
    settings = Settings()
    app_state["settings"] = settings

    # Load or train classifier
    classifier = ComplexityClassifier()
    model_path = Path("models/complexity_classifier.joblib")
    
    if model_path.exists():
        classifier.load(model_path)
    else:
        # Auto-train on startup
        import json

        dataset_path = Path("data/complexity_dataset.jsonl")
        if dataset_path.exists():
            prompts, labels = [], []
            with open(dataset_path) as f:
                for line in f:
                    data = json.loads(line)
                    prompts.append(data["prompt"])
                    labels.append(data["complexity"])
            
            classifier.train(prompts, labels)
            classifier.save(model_path)

    app_state["classifier"] = classifier

    # Initialize providers
    if settings.router_mode == "mock":
        app_state["providers"]["mock"] = MockProvider()
    else:
        if settings.openai_api_key:
            app_state["providers"]["openai"] = OpenAIProvider(settings.openai_api_key)
        if settings.anthropic_api_key:
            app_state["providers"]["anthropic"] = AnthropicProvider(settings.anthropic_api_key)
        
        # Ollama (optional)
        try:
            ollama = OllamaProvider(settings.ollama_base_url)
            if ollama.is_available():
                app_state["providers"]["ollama"] = ollama
        except Exception:
            pass

    # Initialize database and logging
    app_state["database"] = Database(settings.database_path)
    app_state["jsonl_logger"] = JSONLLogger(settings.jsonl_log_path)

    # Initialize evaluator
    judge_provider = None
    if settings.router_mode == "live" and settings.judge_provider in app_state["providers"]:
        judge_provider = app_state["providers"][settings.judge_provider]
    
    app_state["evaluator"] = Evaluator(
        judge_provider=judge_provider,
        judge_model=settings.judge_model,
        min_quality_score=settings.min_quality_score,
        use_judge=(settings.router_mode == "live"),
    )

    # Initialize feedback loop
    app_state["feedback_loop"] = FeedbackLoop(
        database=app_state["database"],
        target_quality=0.78,
        window_size=20,
    )

    yield

    # Cleanup
    if app_state["database"]:
        app_state["database"].close()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="LLM Cost Router",
        description="Intelligent routing layer for LLMs that optimizes cost by complexity",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        settings = app_state["settings"]
        
        return HealthResponse(
            status="healthy",
            router_mode=settings.router_mode,
            classifier_loaded=app_state["classifier"] is not None,
            configured_providers=list(app_state["providers"].keys()),
        )

    @app.post("/v1/route", response_model=RouteResponse)
    async def route(request: RouteRequest):
        """Route a request to the optimal model."""
        # Extract prompt
        if request.prompt:
            prompt = request.prompt
            messages = [{"role": "user", "content": prompt}]
        elif request.messages:
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            prompt = messages[-1]["content"] if messages else ""
        else:
            raise HTTPException(status_code=400, detail="Either 'prompt' or 'messages' required")

        # Classify complexity
        classifier = app_state["classifier"]
        classification = classifier.predict(prompt)

        # Get current thresholds from feedback loop
        feedback_loop = app_state["feedback_loop"]
        thresholds = feedback_loop.get_thresholds()

        # Determine tier (with override or constraints)
        if request.force_tier:
            tier = request.force_tier
        else:
            tier = get_tier_for_complexity(
                classification.score,
                thresholds["simple_threshold"],
                thresholds["moderate_threshold"],
            )

        # Apply local_only constraint
        if request.local_only and tier != "local":
            tier = "local"

        # Select model
        model_key = TIER_MODELS[tier][0]

        # Get provider
        provider_name = MODEL_CATALOG[model_key].provider
        
        # In mock mode, use mock provider
        settings = app_state["settings"]
        if settings.router_mode == "mock":
            provider = app_state["providers"]["mock"]
        else:
            provider = app_state["providers"].get(provider_name)

        if not provider or not provider.is_available():
            # Try fallback chain
            fallback_chain = FALLBACK_CHAINS.get(model_key, [])
            fallback_used = False
            
            for fallback_key in fallback_chain:
                fallback_provider_name = MODEL_CATALOG[fallback_key].provider
                fallback_provider = app_state["providers"].get(fallback_provider_name)
                
                if fallback_provider and fallback_provider.is_available():
                    model_key = fallback_key
                    provider = fallback_provider
                    fallback_used = True
                    break
            
            if not provider:
                raise HTTPException(
                    status_code=503,
                    detail=f"Provider '{provider_name}' not available and no fallback found",
                )
        else:
            fallback_used = False

        # Call provider
        model_name = MODEL_CATALOG[model_key].model
        
        try:
            response = await provider.complete(
                messages=messages,
                model=model_name,
                temperature=0.7,
                max_tokens=2000,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Provider call failed: {str(e)}")

        # Calculate savings
        frontier_baseline_cost = calculate_frontier_baseline_cost(
            response.input_tokens, response.output_tokens
        )
        savings_usd = frontier_baseline_cost - response.cost_usd
        savings_percent = (savings_usd / frontier_baseline_cost * 100) if frontier_baseline_cost > 0 else 0

        # Log request
        database = app_state["database"]
        request_id = database.log_request(
            RequestLog(
                timestamp=datetime.utcnow().isoformat(),
                prompt=prompt,
                complexity_tier=classification.tier,
                complexity_score=classification.score,
                selected_model=model_key,
                provider=response.provider,
                tier=tier,
                response_content=response.content,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
                cost_usd=response.cost_usd,
                frontier_baseline_cost_usd=frontier_baseline_cost,
                savings_usd=savings_usd,
                savings_percent=savings_percent,
            )
        )

        # JSONL logging
        jsonl_logger = app_state["jsonl_logger"]
        jsonl_logger.log(
            "request",
            {
                "request_id": request_id,
                "prompt": prompt[:100],
                "complexity": classification.tier,
                "tier": tier,
                "model": model_key,
                "cost_usd": response.cost_usd,
                "savings_usd": savings_usd,
            },
        )

        # Evaluate quality (if requested)
        eval_scores = None
        
        if request.run_eval:
            evaluator = app_state["evaluator"]
            
            # In mock mode, use mock judge
            if settings.router_mode == "mock":
                mock_judge = MockJudge()
                judge_score, judge_rationale = await mock_judge.evaluate_mock(
                    prompt, response.content, tier
                )
                
                # Still run heuristic
                heuristic_score, _ = evaluator._heuristic_score(
                    prompt, response, classification.tier
                )
                
                quality_score = 0.4 * heuristic_score + 0.6 * judge_score
                passed = quality_score >= settings.min_quality_score
                
                eval_result_data = {
                    "quality_score": quality_score,
                    "passed": passed,
                    "heuristic_score": heuristic_score,
                    "judge_score": judge_score,
                    "judge_rationale": judge_rationale,
                }
            else:
                eval_result = await evaluator.evaluate(prompt, response, classification.tier)
                eval_result_data = {
                    "quality_score": eval_result.quality_score,
                    "passed": eval_result.passed,
                    "heuristic_score": eval_result.heuristic_score,
                    "judge_score": eval_result.judge_score,
                    "judge_rationale": eval_result.judge_rationale,
                }

            # Log evaluation
            database.log_evaluation(
                EvaluationLog(
                    request_id=request_id,
                    timestamp=datetime.utcnow().isoformat(),
                    **eval_result_data,
                    failure_reasons=None,
                )
            )

            jsonl_logger.log(
                "evaluation",
                {"request_id": request_id, **eval_result_data},
            )

            eval_scores = EvalScores(**eval_result_data)

            # Check and adjust thresholds (adaptive feedback)
            if settings.adaptive_feedback_enabled:
                feedback_loop.check_and_adjust()

        # Build response
        return RouteResponse(
            content=response.content,
            routing_decision=RoutingDecision(
                complexity_tier=classification.tier,
                complexity_score=classification.score,
                complexity_explanation=classification.explanation,
                selected_model=model_key,
                selected_tier=tier,
                provider=response.provider,
                fallback_used=fallback_used,
            ),
            cost_usd=response.cost_usd,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            frontier_baseline_cost_usd=frontier_baseline_cost,
            savings_usd=savings_usd,
            savings_percent=savings_percent,
            eval_scores=eval_scores,
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(request: ChatCompletionRequest):
        """OpenAI-compatible chat completions endpoint."""
        # Route the request
        route_request = RouteRequest(
            messages=[Message(role=m.role, content=m.content) for m in request.messages],
            run_eval=False,  # Skip eval for compatibility endpoint
        )
        
        route_response = await route(route_request)

        # Build OpenAI-compatible response
        return ChatCompletionResponse(
            id=f"chatcmpl-{datetime.utcnow().timestamp()}",
            created=int(datetime.utcnow().timestamp()),
            model=route_response.routing_decision.selected_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=route_response.content),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=route_response.input_tokens,
                completion_tokens=route_response.output_tokens,
                total_tokens=route_response.input_tokens + route_response.output_tokens,
            ),
            x_routing={
                "complexity_tier": route_response.routing_decision.complexity_tier,
                "complexity_score": route_response.routing_decision.complexity_score,
                "selected_tier": route_response.routing_decision.selected_tier,
                "cost_usd": route_response.cost_usd,
                "savings_usd": route_response.savings_usd,
                "savings_percent": route_response.savings_percent,
            },
        )

    @app.get("/v1/models")
    async def list_models():
        """List configured models with pricing."""
        return {
            "models": [
                ModelInfo(
                    provider=info.provider,
                    model=info.model,
                    tier=info.tier,
                    input_price_per_1m=info.input_price_per_1m,
                    output_price_per_1m=info.output_price_per_1m,
                )
                for info in MODEL_CATALOG.values()
            ]
        }

    @app.get("/v1/metrics/summary")
    async def metrics_summary():
        """Get summary metrics."""
        database = app_state["database"]
        return database.get_metrics_summary()

    @app.get("/v1/logs")
    async def get_logs(
        limit: int = 100,
        offset: int = 0,
        tier: Optional[str] = None,
    ):
        """Get paginated request logs."""
        database = app_state["database"]
        return {
            "requests": database.get_requests(limit=limit, offset=offset, tier=tier),
        }

    return app
