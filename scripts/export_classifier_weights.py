#!/usr/bin/env python3
"""Export trained classifier weights to JSON for browser-based inference."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_cost_router.classifier import ComplexityClassifier
from llm_cost_router.pricing import MODEL_CATALOG, TIER_MODELS, FALLBACK_CHAINS


def export_classifier_weights(model_path: Path, output_path: Path):
    """Export classifier weights and metadata to JSON."""
    
    # Load classifier
    classifier = ComplexityClassifier()
    if not classifier.load(model_path):
        raise ValueError(f"Failed to load classifier from {model_path}")
    
    # Extract pipeline components
    pipeline = classifier.pipeline
    tfidf = pipeline.named_steps['features'].transformer_list[0][1]
    structural = pipeline.named_steps['features'].transformer_list[1][1]
    scaler = structural.named_steps['scale']
    logistic = pipeline.named_steps['classifier']
    
    # Extract TF-IDF parameters
    vocabulary = {word: int(idx) for word, idx in tfidf.vocabulary_.items()}
    idf = tfidf.idf_.tolist()
    
    # Extract logistic regression parameters
    classes = classifier.classes.tolist()
    coef = logistic.coef_.tolist()
    intercept = logistic.intercept_.tolist()
    
    # Extract scaler parameters
    scaler_mean = scaler.mean_.tolist()
    scaler_scale = scaler.scale_.tolist()
    
    # Build weights object
    weights = {
        "tfidf": {
            "vocabulary": vocabulary,
            "idf": idf,
            "max_features": tfidf.max_features,
            "ngram_range": list(tfidf.ngram_range),
            "norm": tfidf.norm,
            "use_idf": tfidf.use_idf,
            "smooth_idf": tfidf.smooth_idf,
            "sublinear_tf": tfidf.sublinear_tf,
        },
        "scaler": {
            "mean": scaler_mean,
            "scale": scaler_scale,
        },
        "logistic": {
            "classes": classes,
            "coef": coef,
            "intercept": intercept,
        },
        "thresholds": {
            "simple": 0.35,
            "moderate": 0.65,
        },
        "pricing": {
            "models": {
                key: {
                    "provider": info.provider,
                    "model": info.model,
                    "tier": info.tier,
                    "input_price_per_1m": info.input_price_per_1m,
                    "output_price_per_1m": info.output_price_per_1m,
                }
                for key, info in MODEL_CATALOG.items()
            },
            "tier_models": TIER_MODELS,
            "fallback_chains": FALLBACK_CHAINS,
        },
    }
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"✓ Exported classifier weights to {output_path}")
    print(f"  Vocabulary size: {len(vocabulary)}")
    print(f"  Classes: {classes}")
    print(f"  Structural features: {len(scaler_mean)}")


def main():
    model_path = Path("models/complexity_classifier.joblib")
    output_path = Path("demo/data/model_weights.json")
    
    if not model_path.exists():
        print(f"Error: {model_path} not found. Train classifier first.")
        return 1
    
    export_classifier_weights(model_path, output_path)
    
    # Also copy benchmark results
    benchmark_src = Path("docs/benchmark_results.json")
    benchmark_dst = Path("demo/data/benchmark_results.json")
    
    if benchmark_src.exists():
        import shutil
        shutil.copy(benchmark_src, benchmark_dst)
        print(f"✓ Copied benchmark results to {benchmark_dst}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
