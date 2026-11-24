/**
 * LLM Cost Router - Browser-compatible complexity classifier
 * Exact parity with Python ComplexityClassifier implementation
 */

class ComplexityClassifier {
  constructor(weights) {
    this.weights = weights;
    this.vocabulary = weights.tfidf.vocabulary;
    this.idf = weights.tfidf.idf;
    this.ngram_range = weights.tfidf.ngram_range;
    this.scaler_mean = weights.scaler.mean;
    this.scaler_scale = weights.scaler.scale;
    this.classes = weights.logistic.classes;
    this.coef = weights.logistic.coef;
    this.intercept = weights.logistic.intercept;
    this.thresholds = weights.thresholds;
    this.pricing = weights.pricing;
  }

  /**
   * Extract the exact 24 structural features matching Python implementation
   */
  extractStructuralFeatures(prompt) {
    const charCount = prompt.length;
    const words = prompt.split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length;
    const tokenEstimate = wordCount * 1.3;
    const avgWordLength = wordCount > 0 ? words.reduce((sum, w) => sum + w.length, 0) / wordCount : 0;

    // Code indicators
    const hasCodeFence = prompt.includes('```') ? 1 : 0;
    const codeFenceCount = (prompt.match(/```/g) || []).length / 2;
    const hasInlineCode = (prompt.includes('`') && !prompt.includes('```')) ? 1 : 0;

    // Technical/syntax markers
    const hasFunctionDef = /\b(def|function|fn|func|class)\b/i.test(prompt) ? 1 : 0;
    const hasImport = /\b(import|require|include|using)\b/i.test(prompt) ? 1 : 0;
    const hasDataStructures = /\b(array|list|dict|map|set|tree|graph|queue)\b/i.test(prompt) ? 1 : 0;

    // Complexity markers
    const hasAlgorithm = /\b(algorithm|optimize|complexity|efficient|performance)\b/i.test(prompt) ? 1 : 0;
    const hasDesignPattern = /\b(design|architecture|pattern|system|distributed|scalable)\b/i.test(prompt) ? 1 : 0;
    const hasProduction = /\b(production|deployment|scale|fault.tolerant|high.availability)\b/i.test(prompt) ? 1 : 0;

    // Reasoning indicators
    const questionMarks = (prompt.match(/\?/g) || []).length;
    const hasMultipleQuestions = questionMarks > 1 ? 1 : 0;
    const hasComparison = /\b(compare|difference|versus|vs|better|worse)\b/i.test(prompt) ? 1 : 0;
    const hasExplanation = /\b(explain|how|why|describe|elaborate)\b/i.test(prompt) ? 1 : 0;

    // Structured output cues
    const hasListRequest = /\b(list|enumerate|steps|points)\b/i.test(prompt) ? 1 : 0;
    const hasJsonRequest = /\bjson\b/i.test(prompt) ? 1 : 0;
    const hasFormatSpec = /\b(format|structure|schema)\b/i.test(prompt) ? 1 : 0;

    // Constraint indicators
    const hasRequirements = /\b(must|should|require|need|ensure|guarantee)\b/i.test(prompt) ? 1 : 0;
    const constraintCount = (prompt.match(/\b(with|including|support|handle|provide)\b/gi) || []).length;

    // Sentence complexity (matches Python re.split behavior - includes empty strings)
    const sentenceCount = prompt.split(/[.!?]+/).length;
    const avgSentenceLength = sentenceCount > 0 ? wordCount / sentenceCount : 0;

    // Return normalized features matching Python implementation
    return [
      Math.log1p(charCount) / 10,
      Math.log1p(wordCount) / 5,
      Math.log1p(tokenEstimate) / 5,
      avgWordLength / 10,
      hasCodeFence,
      codeFenceCount,
      hasInlineCode,
      hasFunctionDef,
      hasImport,
      hasDataStructures,
      hasAlgorithm,
      hasDesignPattern,
      hasProduction,
      questionMarks,
      hasMultipleQuestions,
      hasComparison,
      hasExplanation,
      hasListRequest,
      hasJsonRequest,
      hasFormatSpec,
      hasRequirements,
      constraintCount / 10,
      sentenceCount,
      avgSentenceLength / 20,
    ];
  }

  /**
   * Tokenize text for TF-IDF (matching sklearn's tokenization)
   */
  tokenize(text) {
    text = text.toLowerCase();
    // Split on word boundaries (matching sklearn's default pattern)
    return text.match(/\b\w\w+\b/g) || [];
  }

  /**
   * Generate n-grams from token array
   */
  generateNgrams(tokens, n) {
    if (n === 1) return tokens;
    const ngrams = [];
    for (let i = 0; i <= tokens.length - n; i++) {
      ngrams.push(tokens.slice(i, i + n).join(' '));
    }
    return ngrams;
  }

  /**
   * Compute TF-IDF features matching sklearn implementation
   */
  computeTfidf(prompt) {
    const tokens = this.tokenize(prompt);
    const tfidf = new Array(this.idf.length).fill(0);

    // Generate all n-grams
    const allNgrams = [];
    for (let n = this.ngram_range[0]; n <= this.ngram_range[1]; n++) {
      allNgrams.push(...this.generateNgrams(tokens, n));
    }

    // Count term frequencies
    const termCounts = {};
    for (const ngram of allNgrams) {
      termCounts[ngram] = (termCounts[ngram] || 0) + 1;
    }

    // Compute TF-IDF
    for (const [term, count] of Object.entries(termCounts)) {
      const idx = this.vocabulary[term];
      if (idx !== undefined) {
        // sklearn uses raw TF (not sublinear) when sublinear_tf=False
        const tf = count;
        tfidf[idx] = tf * this.idf[idx];
      }
    }

    // L2 normalization
    const norm = Math.sqrt(tfidf.reduce((sum, val) => sum + val * val, 0));
    if (norm > 0) {
      for (let i = 0; i < tfidf.length; i++) {
        tfidf[i] /= norm;
      }
    }

    return tfidf;
  }

  /**
   * Scale structural features using fitted StandardScaler
   */
  scaleFeatures(features) {
    return features.map((val, i) => (val - this.scaler_mean[i]) / this.scaler_scale[i]);
  }

  /**
   * Predict class probabilities using logistic regression
   */
  predictProba(combinedFeatures) {
    const scores = [];
    for (let i = 0; i < this.classes.length; i++) {
      let score = this.intercept[i];
      for (let j = 0; j < combinedFeatures.length; j++) {
        score += this.coef[i][j] * combinedFeatures[j];
      }
      scores.push(score);
    }

    // Softmax
    const maxScore = Math.max(...scores);
    const expScores = scores.map(s => Math.exp(s - maxScore));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    return expScores.map(s => s / sumExp);
  }

  /**
   * Predict complexity tier and score for a prompt
   */
  predict(prompt) {
    // Extract features
    const tfidfFeatures = this.computeTfidf(prompt);
    const structuralFeatures = this.extractStructuralFeatures(prompt);
    const scaledStructural = this.scaleFeatures(structuralFeatures);

    // Combine features (TF-IDF + structural)
    const combinedFeatures = [...tfidfFeatures, ...scaledStructural];

    // Get probabilities
    const probabilities = this.predictProba(combinedFeatures);

    // Build probability dict
    const probabilityDict = {};
    for (let i = 0; i < this.classes.length; i++) {
      probabilityDict[this.classes[i]] = probabilities[i];
    }

    // Predict class (argmax)
    const maxIdx = probabilities.indexOf(Math.max(...probabilities));
    const tier = this.classes[maxIdx];

    // Calculate continuous score [0, 1]
    const weights = { simple: 0.0, moderate: 0.5, complex: 1.0 };
    const score = Object.entries(probabilityDict).reduce(
      (sum, [cls, prob]) => sum + prob * weights[cls],
      0
    );

    // Generate explanation
    const explanation = this.generateExplanation(prompt, tier, probabilityDict);

    return {
      tier,
      score,
      probabilities: probabilityDict,
      explanation,
    };
  }

  /**
   * Generate brief explanation for classification
   */
  generateExplanation(prompt, tier, probabilities) {
    const signals = [];
    const wordCount = prompt.split(/\s+/).filter(w => w.length > 0).length;

    if (wordCount < 10) signals.push('very short');
    else if (wordCount > 50) signals.push('lengthy');

    if (prompt.includes('```')) signals.push('contains code');
    if (/\b(design|architecture|system|distributed)\b/i.test(prompt)) signals.push('system design');
    if (/\b(algorithm|complexity|optimize)\b/i.test(prompt)) signals.push('algorithmic');
    if (/\b(production|scale|fault.tolerant)\b/i.test(prompt)) signals.push('production-focused');
    if ((prompt.match(/\?/g) || []).length > 1) signals.push('multi-part question');

    const confidence = Math.max(...Object.values(probabilities));
    const confidenceStr = confidence > 0.7 ? 'high' : confidence > 0.5 ? 'moderate' : 'low';

    const signalStr = signals.length > 0 ? signals.join(', ') : 'general query';
    return `${tier} (${confidenceStr} confidence): ${signalStr}`;
  }

  /**
   * Map complexity score to routing tier
   */
  getTierForComplexity(score) {
    if (score < this.thresholds.simple) return 'local';
    if (score < this.thresholds.moderate) return 'budget';
    if (score < 0.85) return 'balanced';
    return 'frontier';
  }

  /**
   * Get model and fallback for a tier
   */
  getModelForTier(tier, availableProviders = ['openai', 'anthropic', 'ollama']) {
    const preferredModel = this.pricing.tier_models[tier][0];
    const modelInfo = this.pricing.models[preferredModel];

    if (availableProviders.includes(modelInfo.provider)) {
      return { model: preferredModel, fallback: false, info: modelInfo };
    }

    // Try fallback chain
    const fallbackChain = this.pricing.fallback_chains[preferredModel] || [];
    for (const fallbackModel of fallbackChain) {
      const fallbackInfo = this.pricing.models[fallbackModel];
      if (availableProviders.includes(fallbackInfo.provider)) {
        return { model: fallbackModel, fallback: true, info: fallbackInfo };
      }
    }

    return null;
  }

  /**
   * Calculate cost for given token usage
   */
  calculateCost(inputTokens, outputTokens, modelKey) {
    const modelInfo = this.pricing.models[modelKey];
    if (!modelInfo) return 0;

    const inputCost = (inputTokens / 1_000_000) * modelInfo.input_price_per_1m;
    const outputCost = (outputTokens / 1_000_000) * modelInfo.output_price_per_1m;
    return inputCost + outputCost;
  }

  /**
   * Calculate frontier baseline cost (using gpt-4o)
   */
  calculateFrontierBaselineCost(inputTokens, outputTokens) {
    return this.calculateCost(inputTokens, outputTokens, 'openai/gpt-4o');
  }
}

// Export for both browser and Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ComplexityClassifier };
}

export { ComplexityClassifier };
