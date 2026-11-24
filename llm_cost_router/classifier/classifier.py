"""Complexity classifier using scikit-learn."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Dict, List

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


ComplexityTier = Literal["simple", "moderate", "complex"]


@dataclass
class ClassificationResult:
    """Result of complexity classification."""

    tier: ComplexityTier
    score: float  # Continuous score [0.0, 1.0]
    probabilities: Dict[str, float]
    explanation: str


class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract structural features from prompts."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Extract features from prompts."""
        features = []
        
        for prompt in X:
            features.append(self._extract_features(prompt))
        
        return np.array(features)

    def _extract_features(self, prompt: str) -> List[float]:
        """Extract structural features from a single prompt."""
        # Length features
        char_count = len(prompt)
        word_count = len(prompt.split())
        token_estimate = word_count * 1.3  # Rough token estimate
        avg_word_length = np.mean([len(w) for w in prompt.split()]) if word_count > 0 else 0

        # Code indicators
        has_code_fence = int("```" in prompt)
        code_fence_count = prompt.count("```") // 2
        has_inline_code = int("`" in prompt and "```" not in prompt)
        
        # Technical/syntax markers
        has_function_def = int(
            re.search(r"\b(def|function|fn|func|class)\b", prompt, re.IGNORECASE) is not None
        )
        has_import = int(re.search(r"\b(import|require|include|using)\b", prompt, re.IGNORECASE) is not None)
        has_data_structures = int(
            re.search(r"\b(array|list|dict|map|set|tree|graph|queue)\b", prompt, re.IGNORECASE) is not None
        )
        
        # Complexity markers
        has_algorithm = int(
            re.search(r"\b(algorithm|optimize|complexity|efficient|performance)\b", prompt, re.IGNORECASE) is not None
        )
        has_design_pattern = int(
            re.search(r"\b(design|architecture|pattern|system|distributed|scalable)\b", prompt, re.IGNORECASE) is not None
        )
        has_production = int(
            re.search(r"\b(production|deployment|scale|fault.tolerant|high.availability)\b", prompt, re.IGNORECASE) is not None
        )
        
        # Reasoning indicators
        question_marks = prompt.count("?")
        has_multiple_questions = int(question_marks > 1)
        has_comparison = int(
            re.search(r"\b(compare|difference|versus|vs|better|worse)\b", prompt, re.IGNORECASE) is not None
        )
        has_explanation = int(
            re.search(r"\b(explain|how|why|describe|elaborate)\b", prompt, re.IGNORECASE) is not None
        )
        
        # Structured output cues
        has_list_request = int(
            re.search(r"\b(list|enumerate|steps|points)\b", prompt, re.IGNORECASE) is not None
        )
        has_json_request = int(re.search(r"\bjson\b", prompt, re.IGNORECASE) is not None)
        has_format_spec = int(re.search(r"\b(format|structure|schema)\b", prompt, re.IGNORECASE) is not None)
        
        # Constraint indicators
        has_requirements = int(
            re.search(r"\b(must|should|require|need|ensure|guarantee)\b", prompt, re.IGNORECASE) is not None
        )
        constraint_count = len(re.findall(r"\b(with|including|support|handle|provide)\b", prompt, re.IGNORECASE))
        
        # Sentence complexity
        sentence_count = len(re.split(r"[.!?]+", prompt))
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return [
            # Length features (normalized)
            np.log1p(char_count) / 10,
            np.log1p(word_count) / 5,
            np.log1p(token_estimate) / 5,
            avg_word_length / 10,
            # Code features
            has_code_fence,
            code_fence_count,
            has_inline_code,
            has_function_def,
            has_import,
            has_data_structures,
            # Complexity markers
            has_algorithm,
            has_design_pattern,
            has_production,
            # Reasoning features
            question_marks,
            has_multiple_questions,
            has_comparison,
            has_explanation,
            # Output structure
            has_list_request,
            has_json_request,
            has_format_spec,
            # Constraints
            has_requirements,
            constraint_count / 10,
            # Sentence features
            sentence_count,
            avg_sentence_length / 20,
        ]


class ComplexityClassifier:
    """Classify prompt complexity for routing decisions."""

    def __init__(self, model_path: Path | None = None):
        """Initialize classifier."""
        self.model_path = model_path
        self.pipeline: Pipeline | None = None
        self.classes: np.ndarray | None = None

    def train(self, prompts: List[str], labels: List[str]) -> Dict[str, float]:
        """
        Train the classifier on labeled data.

        Returns:
            Dictionary with training metrics
        """
        # Create feature extraction pipeline
        self.pipeline = Pipeline([
            (
                "features",
                FeatureUnion([
                    ("tfidf", TfidfVectorizer(
                        max_features=500,
                        ngram_range=(1, 3),
                        min_df=2,
                        max_df=0.8,
                    )),
                    ("structural", Pipeline([
                        ("extract", StructuralFeatureExtractor()),
                        ("scale", StandardScaler()),
                    ])),
                ]),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ])

        # Train
        self.pipeline.fit(prompts, labels)
        self.classes = self.pipeline.classes_

        # Calculate training accuracy
        predictions = self.pipeline.predict(prompts)
        accuracy = np.mean(predictions == np.array(labels))

        return {"accuracy": accuracy}

    def predict(self, prompt: str) -> ClassificationResult:
        """Classify a single prompt."""
        if self.pipeline is None:
            raise ValueError("Classifier not trained or loaded")

        # Get prediction and probabilities
        tier = self.pipeline.predict([prompt])[0]
        proba = self.pipeline.predict_proba([prompt])[0]

        # Create probability dict
        proba_dict = {cls: float(prob) for cls, prob in zip(self.classes, proba)}

        # Calculate continuous complexity score [0.0, 1.0]
        # Weight: simple=0, moderate=0.5, complex=1.0
        weights = {"simple": 0.0, "moderate": 0.5, "complex": 1.0}
        score = sum(proba_dict.get(cls, 0) * weights[cls] for cls in weights)

        # Generate explanation
        explanation = self._generate_explanation(prompt, tier, proba_dict)

        return ClassificationResult(
            tier=tier,
            score=score,
            probabilities=proba_dict,
            explanation=explanation,
        )

    def _generate_explanation(
        self, prompt: str, tier: str, probabilities: Dict[str, float]
    ) -> str:
        """Generate brief explanation for classification."""
        signals = []

        # Length signals
        word_count = len(prompt.split())
        if word_count < 10:
            signals.append("very short")
        elif word_count > 50:
            signals.append("lengthy")

        # Content signals
        if "```" in prompt:
            signals.append("contains code")
        if re.search(r"\b(design|architecture|system|distributed)\b", prompt, re.IGNORECASE):
            signals.append("system design")
        if re.search(r"\b(algorithm|complexity|optimize)\b", prompt, re.IGNORECASE):
            signals.append("algorithmic")
        if re.search(r"\b(production|scale|fault.tolerant)\b", prompt, re.IGNORECASE):
            signals.append("production-focused")
        if prompt.count("?") > 1:
            signals.append("multi-part question")

        confidence = max(probabilities.values())
        confidence_str = "high" if confidence > 0.7 else "moderate" if confidence > 0.5 else "low"

        signal_str = ", ".join(signals) if signals else "general query"
        return f"{tier} ({confidence_str} confidence): {signal_str}"

    def save(self, path: Path):
        """Save trained model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "classes": self.classes}, path)

    def load(self, path: Path):
        """Load trained model."""
        if not path.exists():
            return False
        
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.classes = data["classes"]
        return True
