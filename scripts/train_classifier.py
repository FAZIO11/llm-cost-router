#!/usr/bin/env python3
"""Train the complexity classifier and report metrics."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_cost_router.classifier import ComplexityClassifier


def load_dataset(dataset_path: Path):
    """Load JSONL dataset."""
    prompts = []
    labels = []
    
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
            labels.append(data["complexity"])
    
    return prompts, labels


def main():
    """Train and evaluate classifier."""
    print("=" * 80)
    print("LLM Cost Router - Complexity Classifier Training")
    print("=" * 80)
    print()

    # Paths
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "complexity_dataset.jsonl"
    model_path = project_root / "models" / "complexity_classifier.joblib"

    # Load data
    print(f"Loading dataset from {dataset_path}...")
    prompts, labels = load_dataset(dataset_path)
    print(f"Loaded {len(prompts)} examples")
    
    # Count by class
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(unique, counts):
        print(f"  {cls}: {count}")
    print()

    # Split data
    print("Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        prompts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print()

    # Train
    print("Training classifier...")
    classifier = ComplexityClassifier()
    train_metrics = classifier.train(X_train, y_train)
    print(f"Training accuracy: {train_metrics['accuracy']:.4f}")
    print()

    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred = []
    for prompt in X_test:
        result = classifier.predict(prompt)
        y_pred.append(result.tier)
    
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print()
    
    print("Per-class metrics:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save model
    print(f"\nSaving model to {model_path}...")
    classifier.save(model_path)
    print("Model saved successfully!")
    print()

    # Example predictions
    print("Example predictions:")
    print("-" * 80)
    
    examples = [
        "What is 2+2?",
        "Explain how binary search works with example code",
        "Design and implement a distributed rate limiting system that works across multiple servers",
    ]
    
    for example in examples:
        result = classifier.predict(example)
        print(f"\nPrompt: {example[:70]}...")
        print(f"Tier: {result.tier}")
        print(f"Score: {result.score:.4f}")
        print(f"Explanation: {result.explanation}")
    
    print()
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
