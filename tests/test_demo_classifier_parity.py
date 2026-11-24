"""Test parity between Python and JavaScript classifiers."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_cost_router.classifier import ComplexityClassifier


@pytest.fixture(scope="module")
def python_classifier():
    """Load Python classifier."""
    classifier = ComplexityClassifier()
    model_path = Path("models/complexity_classifier.joblib")
    assert classifier.load(model_path), "Failed to load classifier"
    return classifier


@pytest.fixture(scope="module")
def benchmark_prompts():
    """Load benchmark prompts."""
    prompts_path = Path("data/benchmark_prompts.json")
    with open(prompts_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def js_predictions(benchmark_prompts):
    """Get JavaScript predictions via Node.js."""
    # Check if Node.js is available
    try:
        subprocess.run(
            ["node", "--version"],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("Node.js not available")

    # Create Node.js test script
    test_script = """
const fs = require('fs');
const { ComplexityClassifier } = require('./demo/classifier.js');

// Load weights and prompts
const weights = JSON.parse(fs.readFileSync('./demo/data/model_weights.json', 'utf8'));
const prompts = JSON.parse(fs.readFileSync('./data/benchmark_prompts.json', 'utf8'));

// Initialize classifier
const classifier = new ComplexityClassifier(weights);

// Predict all prompts
const results = prompts.map(item => {
  const prediction = classifier.predict(item.prompt);
  return {
    id: item.id,
    prompt: item.prompt,
    tier: prediction.tier,
    score: prediction.score,
    probabilities: prediction.probabilities,
  };
});

// Output JSON
console.log(JSON.stringify(results, null, 2));
"""

    # Write temp script
    script_path = Path("test_js_classifier.js")
    script_path.write_text(test_script)

    try:
        # Run Node.js script
        result = subprocess.run(
            ["node", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path.cwd(),
        )

        # Parse results
        js_results = json.loads(result.stdout)
        return {r["id"]: r for r in js_results}

    finally:
        # Cleanup
        if script_path.exists():
            script_path.unlink()


def test_classifier_parity_exact_tier_match(python_classifier, benchmark_prompts, js_predictions):
    """Test that Python and JS classifiers predict the same tier for all prompts."""
    mismatches = []

    for item in benchmark_prompts:
        prompt = item["prompt"]
        prompt_id = item["id"]

        # Python prediction
        py_pred = python_classifier.predict(prompt)

        # JS prediction
        js_pred = js_predictions[prompt_id]

        # Check tier match
        if py_pred.tier != js_pred["tier"]:
            mismatches.append({
                "id": prompt_id,
                "prompt": prompt[:50] + "...",
                "python_tier": py_pred.tier,
                "js_tier": js_pred["tier"],
            })

    # Assert no mismatches
    assert len(mismatches) == 0, f"Found {len(mismatches)} tier mismatches:\n{json.dumps(mismatches, indent=2)}"


def test_classifier_parity_score_precision(python_classifier, benchmark_prompts, js_predictions):
    """Test that complexity scores are within 1e-4 absolute difference."""
    max_delta = 0.0
    deltas = []

    for item in benchmark_prompts:
        prompt = item["prompt"]
        prompt_id = item["id"]

        # Python prediction
        py_pred = python_classifier.predict(prompt)

        # JS prediction
        js_pred = js_predictions[prompt_id]

        # Calculate delta
        delta = abs(py_pred.score - js_pred["score"])
        max_delta = max(max_delta, delta)

        deltas.append({
            "id": prompt_id,
            "python_score": py_pred.score,
            "js_score": js_pred["score"],
            "delta": delta,
        })

    # Print max delta for verification
    print(f"\nMax complexity score delta: {max_delta:.8f}")

    # Assert within tolerance
    assert max_delta < 1e-4, f"Max delta {max_delta:.8f} exceeds threshold 1e-4"


def test_classifier_parity_probability_distribution(python_classifier, benchmark_prompts, js_predictions):
    """Test that probability distributions are close."""
    max_prob_delta = 0.0

    for item in benchmark_prompts:
        prompt = item["prompt"]
        prompt_id = item["id"]

        # Python prediction
        py_pred = python_classifier.predict(prompt)

        # JS prediction
        js_pred = js_predictions[prompt_id]

        # Check each class probability
        for cls in ["simple", "moderate", "complex"]:
            py_prob = py_pred.probabilities[cls]
            js_prob = js_pred["probabilities"][cls]
            delta = abs(py_prob - js_prob)
            max_prob_delta = max(max_prob_delta, delta)

    print(f"\nMax probability delta: {max_prob_delta:.8f}")

    # Probabilities should be very close (within 1e-4)
    assert max_prob_delta < 1e-4, f"Max probability delta {max_prob_delta:.8f} exceeds threshold"


def test_classifier_parity_all_prompts_processed(benchmark_prompts, js_predictions):
    """Test that JS classifier processed all prompts."""
    expected_ids = {item["id"] for item in benchmark_prompts}
    actual_ids = set(js_predictions.keys())

    missing = expected_ids - actual_ids
    assert len(missing) == 0, f"JS classifier missing predictions for: {missing}"

    extra = actual_ids - expected_ids
    assert len(extra) == 0, f"JS classifier has extra predictions for: {extra}"
