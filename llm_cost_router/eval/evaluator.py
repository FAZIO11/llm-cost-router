"""Response quality evaluation with heuristic and LLM-as-judge scoring."""

import json
import re
from dataclasses import dataclass
from typing import Optional, List, Dict

from ..providers.base import BaseProvider, ProviderResponse


@dataclass
class EvaluationResult:
    """Result of response evaluation."""

    quality_score: float  # Combined score [0.0, 1.0]
    passed: bool
    heuristic_score: float
    judge_score: Optional[float] = None
    judge_rationale: Optional[str] = None
    failure_reasons: List[str] = None


class Evaluator:
    """Response quality evaluator with heuristic and LLM-as-judge scoring."""

    def __init__(
        self,
        judge_provider: Optional[BaseProvider] = None,
        judge_model: str = "gpt-4o-mini",
        min_quality_score: float = 0.75,
        use_judge: bool = True,
    ):
        """Initialize evaluator."""
        self.judge_provider = judge_provider
        self.judge_model = judge_model
        self.min_quality_score = min_quality_score
        self.use_judge = use_judge

    async def evaluate(
        self,
        prompt: str,
        response: ProviderResponse,
        complexity_tier: str,
    ) -> EvaluationResult:
        """
        Evaluate response quality.

        Args:
            prompt: Original prompt
            response: Provider response
            complexity_tier: Predicted complexity tier

        Returns:
            EvaluationResult with scores and pass/fail status
        """
        # Heuristic scoring
        heuristic_score, heuristic_reasons = self._heuristic_score(
            prompt, response, complexity_tier
        )

        # LLM-as-judge scoring (if enabled and available)
        judge_score = None
        judge_rationale = None

        if self.use_judge and self.judge_provider and self.judge_provider.is_available():
            try:
                judge_score, judge_rationale = await self._judge_score(prompt, response.content)
            except Exception:
                # Fall back to heuristic only
                pass

        # Combine scores (weighted average if judge is available)
        if judge_score is not None:
            quality_score = 0.4 * heuristic_score + 0.6 * judge_score
        else:
            quality_score = heuristic_score

        # Determine pass/fail
        passed = quality_score >= self.min_quality_score and len(heuristic_reasons) == 0

        return EvaluationResult(
            quality_score=quality_score,
            passed=passed,
            heuristic_score=heuristic_score,
            judge_score=judge_score,
            judge_rationale=judge_rationale,
            failure_reasons=heuristic_reasons if heuristic_reasons else None,
        )

    def _heuristic_score(
        self, prompt: str, response: ProviderResponse, complexity_tier: str
    ) -> tuple[float, List[str]]:
        """
        Heuristic quality scoring based on response characteristics.

        Returns:
            (score, failure_reasons)
        """
        score = 1.0
        reasons = []

        content = response.content

        # Non-empty check
        if not content or len(content.strip()) < 10:
            reasons.append("Response too short or empty")
            score -= 0.5

        # Length vs complexity
        word_count = len(content.split())
        expected_min_length = {"simple": 10, "moderate": 30, "complex": 100}
        
        if word_count < expected_min_length.get(complexity_tier, 10):
            reasons.append(f"Response length ({word_count} words) inadequate for {complexity_tier} tier")
            score -= 0.2

        # Code block presence if requested
        if re.search(r"\b(code|function|implement|algorithm)\b", prompt, re.IGNORECASE):
            if "```" not in content and "def " not in content and "function " not in content:
                reasons.append("Code requested but not provided")
                score -= 0.3

        # JSON validity if requested
        if re.search(r"\bjson\b", prompt, re.IGNORECASE):
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                try:
                    json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    reasons.append("Invalid JSON in response")
                    score -= 0.2

        # Refusal patterns (strong negative signal)
        refusal_patterns = [
            r"I cannot",
            r"I'm unable to",
            r"I don't have access",
            r"I apologize, but I can't",
            r"as an AI",
        ]
        
        for pattern in refusal_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                reasons.append("Response contains refusal pattern")
                score -= 0.4
                break

        # Error patterns
        error_patterns = [r"error", r"exception", r"failed"]
        error_count = sum(1 for p in error_patterns if re.search(p, content, re.IGNORECASE))
        
        if error_count >= 2:
            reasons.append("Response contains multiple error indicators")
            score -= 0.2

        # Latency sanity check (too fast might indicate error)
        if response.latency_ms < 100 and len(content) > 500:
            reasons.append("Suspiciously fast response for length")
            score -= 0.1

        # Ensure score is in [0, 1]
        score = max(0.0, min(1.0, score))

        return score, reasons

    async def _judge_score(self, prompt: str, response: str) -> tuple[float, str]:
        """
        LLM-as-judge quality scoring.

        Returns:
            (score, rationale)
        """
        judge_prompt = f"""You are evaluating the quality of an LLM response. Rate the response on a scale from 0.0 to 1.0 based on:

1. **Accuracy**: Does the response correctly address the prompt?
2. **Completeness**: Does it cover all aspects of the question?
3. **Instruction-following**: Does it follow any specific requirements or constraints?

Original Prompt:
{prompt}

Response to Evaluate:
{response}

Provide your evaluation in this exact JSON format:
{{
  "score": <float between 0.0 and 1.0>,
  "rationale": "<brief explanation of the score>"
}}"""

        messages = [{"role": "user", "content": judge_prompt}]

        try:
            judge_response = await self.judge_provider.complete(
                messages=messages,
                model=self.judge_model,
                temperature=0.0,
                max_tokens=200,
            )

            # Parse JSON response
            content = judge_response.content
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group(0))
                score = float(data.get("score", 0.5))
                rationale = data.get("rationale", "No rationale provided")
                
                # Normalize score to [0, 1]
                score = max(0.0, min(1.0, score))
                
                return score, rationale

        except Exception:
            pass

        # Fallback: return neutral score
        return 0.5, "Judge evaluation failed"


class MockJudge:
    """Deterministic mock judge for demo mode."""

    def __init__(self):
        pass

    async def evaluate_mock(self, prompt: str, response: str, tier: str) -> tuple[float, str]:
        """
        Mock judge evaluation based on tier and response characteristics.

        Returns:
            (score, rationale)
        """
        # Base score by tier (simulating quality expectations)
        base_scores = {"local": 0.70, "budget": 0.80, "balanced": 0.88, "frontier": 0.95}
        base_score = base_scores.get(tier, 0.75)

        # Adjust based on response length (longer = slightly better for complex tasks)
        word_count = len(response.split())
        
        if word_count < 20:
            base_score -= 0.15
            rationale = f"Response adequate but brief ({word_count} words) for a {tier}-tier model"
        elif word_count > 100:
            base_score += 0.05
            rationale = f"Comprehensive response ({word_count} words) appropriate for {tier} tier"
        else:
            rationale = f"Response length ({word_count} words) suitable for {tier}-tier model"

        # Clamp score
        score = max(0.0, min(1.0, base_score))

        return score, rationale
