import { ComplexityClassifier } from './classifier.js';

let classifier, benchmarkData;

const examples = {
  simple: "What is a palindrome?",
  moderate: "Write a Python function to implement binary search",
  sql: "Write a SQL query to find products that have never been ordered",
  system: "Design a distributed rate limiting system that works across multiple servers"
};

async function init() {
  const [weights, benchmark] = await Promise.all([
    fetch('./data/model_weights.json').then(r => r.json()),
    fetch('./data/benchmark_results.json').then(r => r.json())
  ]);

  classifier = new ComplexityClassifier(weights);
  benchmarkData = benchmark;

  renderKPIs();
  renderAnalytics();
  renderAudit();
  setupEventListeners();
}

function renderKPIs() {
  const s = benchmarkData.summary;
  const kpis = [
    { value: benchmarkData.metadata.total_prompts, label: 'Held-out Prompts' },
    { value: `$${s.routed_total_cost_usd.toFixed(4)}`, sublabel: `vs $${s.baseline_total_cost_usd.toFixed(4)}`, label: 'Routed vs Baseline' },
    { value: `${s.cost_reduction_pct.toFixed(1)}%`, label: 'Cost Saved', class: 'savings' },
    { value: `${s.quality_retention_pct.toFixed(1)}%`, label: 'Baseline Quality', class: 'savings' },
    { value: '93.3%', label: 'Classifier Accuracy' }
  ];

  document.getElementById('kpi-grid').innerHTML = kpis.map(k =>
    `<div class="kpi">
      <div class="kpi-value ${k.class || ''}">${k.value}${k.sublabel ? `<div style="font-size:0.75rem;margin-top:0.25rem;font-weight:400">${k.sublabel}</div>` : ''}</div>
      <div class="kpi-label">${k.label}</div>
    </div>`
  ).join('');
}

function naturalBenchmarkSort(a, b) {
  const orderMap = { simple: 1, moderate: 2, complex: 3 };
  const [aTier, aNum] = a.id.split('-');
  const [bTier, bNum] = b.id.split('-');
  
  const tierDiff = orderMap[aTier] - orderMap[bTier];
  if (tierDiff !== 0) return tierDiff;
  
  return parseInt(aNum, 10) - parseInt(bNum, 10);
}

function renderAnalytics() {
  const prompts = benchmarkData.prompts;

  const sortedPrompts = [...prompts].sort(naturalBenchmarkSort);

  let cumulativeRouted = 0, cumulativeBaseline = 0;
  const routedCumulative = [], baselineCumulative = [], indices = [];

  sortedPrompts.forEach((p, i) => {
    cumulativeRouted += p.routed.cost_usd;
    cumulativeBaseline += p.baseline.cost_usd;
    routedCumulative.push(cumulativeRouted);
    baselineCumulative.push(cumulativeBaseline);
    indices.push(i + 1);
  });

  Plotly.newPlot('chart-cumulative', [
    { x: indices, y: baselineCumulative, name: 'Always GPT-4o', line: { dash: 'dash', color: '#ef4444' } },
    { x: indices, y: routedCumulative, name: 'Routed', fill: 'tonexty', line: { color: '#059669' } }
  ], {
    title: 'Cumulative Cost: Routed vs Always-GPT-4o',
    xaxis: { title: 'Request #' },
    yaxis: { title: 'Cost (USD)' },
    margin: { t: 50, b: 50, l: 60, r: 20 },
    autosize: true
  }, { responsive: true });

  const tierCounts = {};
  prompts.forEach(p => {
    tierCounts[p.selected_tier] = (tierCounts[p.selected_tier] || 0) + 1;
  });

  Plotly.newPlot('chart-tier-dist', [{
    x: Object.keys(tierCounts),
    y: Object.values(tierCounts),
    type: 'bar',
    marker: { color: ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6'] }
  }], {
    title: 'Requests by Tier',
    margin: { t: 50, b: 50, l: 50, r: 20 },
    autosize: true
  }, { responsive: true });

  Plotly.newPlot('chart-scatter', [{
    x: prompts.map(p => p.complexity_score),
    y: prompts.map(p => p.routed.quality_score),
    text: prompts.map(p => p.id),
    mode: 'markers',
    marker: { size: 8, color: prompts.map(p => p.routed.cost_usd), colorscale: 'Viridis', showscale: true }
  }], {
    title: 'Complexity vs Quality',
    xaxis: { title: 'Complexity Score' },
    yaxis: { title: 'Quality Score' },
    margin: { t: 50, b: 50, l: 50, r: 20 },
    autosize: true
  }, { responsive: true });

  renderTierBreakdown(prompts);
}

function renderTierBreakdown(prompts) {
  const tiers = ['simple', 'moderate', 'complex'];
  const tierData = tiers.map(tier => {
    const tierPrompts = prompts.filter(p => p.predicted_complexity === tier);
    const count = tierPrompts.length;
    if (count === 0) return null;

    const avgComplexity = tierPrompts.reduce((sum, p) => sum + p.complexity_score, 0) / count;
    const avgQuality = tierPrompts.reduce((sum, p) => sum + p.routed.quality_score, 0) / count;
    const routedCost = tierPrompts.reduce((sum, p) => sum + p.routed.cost_usd, 0);
    const baselineCost = tierPrompts.reduce((sum, p) => sum + p.baseline.cost_usd, 0);
    const savings = baselineCost > 0 ? ((baselineCost - routedCost) / baselineCost * 100) : 0;

    const models = [...new Set(tierPrompts.map(p => p.selected_model.split('/')[1]))];
    const routedModel = models.length === 1 ? models[0] : models.join(', ');

    return {
      tier,
      count,
      routedModel,
      avgComplexity,
      avgQuality,
      routedCost,
      baselineCost,
      savings
    };
  }).filter(Boolean);

  const html = `
    <h3 style="margin-top:2rem;margin-bottom:1rem">Per-Complexity-Tier Breakdown</h3>
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:0.875rem">
        <thead>
          <tr style="border-bottom:2px solid #e5e7eb;text-align:left">
            <th style="padding:0.75rem">Tier</th>
            <th style="padding:0.75rem">Count</th>
            <th style="padding:0.75rem">Routed Model</th>
            <th style="padding:0.75rem">Avg Complexity</th>
            <th style="padding:0.75rem">Avg Quality</th>
            <th style="padding:0.75rem">Routed Cost</th>
            <th style="padding:0.75rem">Baseline Cost</th>
            <th style="padding:0.75rem">% Saved</th>
          </tr>
        </thead>
        <tbody>
          ${tierData.map(t => `
            <tr style="border-bottom:1px solid #e5e7eb">
              <td style="padding:0.75rem;text-transform:capitalize;font-weight:600">${t.tier}</td>
              <td style="padding:0.75rem">${t.count}</td>
              <td style="padding:0.75rem;font-family:monospace;font-size:0.8125rem">${t.routedModel}</td>
              <td style="padding:0.75rem">${t.avgComplexity.toFixed(4)}</td>
              <td style="padding:0.75rem">${t.avgQuality.toFixed(3)}</td>
              <td style="padding:0.75rem">$${t.routedCost.toFixed(6)}</td>
              <td style="padding:0.75rem">$${t.baselineCost.toFixed(6)}</td>
              <td style="padding:0.75rem;color:#059669;font-weight:600">${t.savings.toFixed(1)}%</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
    <p style="margin-top:1rem;font-size:0.875rem;color:#64748b;line-height:1.6">
      <strong>Methodology:</strong> Each held-out prompt was classified by the trained model, routed to the cheapest capable tier, and evaluated by GPT-4o. 
      Complexity scores are softmax probabilities from the logistic regression classifier. 
      Quality scores combine heuristic checks (0.4 weight) and GPT-4o judge ratings (0.6 weight).
    </p>
  `;

  document.getElementById('tier-breakdown').innerHTML = html;
}

function renderAudit() {
  const prompts = benchmarkData.prompts.sort(naturalBenchmarkSort);
  const html = prompts.map((p, idx) => `
    <div class="audit-row" data-tier="${p.predicted_complexity}" data-prompt-id="${idx}" data-search="${p.prompt.toLowerCase()} ${p.id}">
      <div class="audit-row-header" onclick="toggleAuditRow(${idx})">
        <span style="cursor:pointer">
          <span class="chevron" id="chevron-${idx}">▶</span>
          <strong>${p.id}</strong>: ${p.prompt.substring(0, 80)}${p.prompt.length > 80 ? '...' : ''}
        </span>
        <div style="margin-top:0.5rem;font-size:0.875rem;color:#64748b">
          ${p.selected_model.split('/')[1]} · $${p.routed.cost_usd.toFixed(6)} · Quality: ${p.routed.quality_score.toFixed(3)}
        </div>
      </div>
      <div class="audit-row-details" id="details-${idx}" style="display:none;margin-top:1rem;padding-top:1rem;border-top:1px solid #e5e7eb">
        <div style="margin-bottom:1rem">
          <strong>Full Prompt:</strong>
          <div style="background:#f8fafc;padding:1rem;border-radius:0.5rem;margin-top:0.5rem;white-space:pre-wrap;font-size:0.875rem">${p.prompt}</div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-bottom:1rem">
          <div>
            <strong>Routing:</strong>
            <div style="margin-top:0.5rem;font-size:0.875rem">Complexity: ${p.complexity_score.toFixed(4)}<br>Predicted: ${p.predicted_complexity}<br>Tier: ${p.selected_tier}</div>
          </div>
          <div>
            <strong>Routed (${p.selected_model.split('/')[1]}):</strong>
            <div style="margin-top:0.5rem;font-size:0.875rem">Cost: $${p.routed.cost_usd.toFixed(6)}<br>Quality: ${p.routed.quality_score.toFixed(3)}</div>
          </div>
          <div>
            <strong>Baseline (gpt-4o):</strong>
            <div style="margin-top:0.5rem;font-size:0.875rem">Cost: $${p.baseline.cost_usd.toFixed(6)}<br>Quality: ${p.baseline.quality_score.toFixed(3)}</div>
          </div>
          <div>
            <strong>Savings:</strong>
            <div style="margin-top:0.5rem;font-size:0.875rem;color:#059669;font-weight:600">${((p.baseline.cost_usd - p.routed.cost_usd) / p.baseline.cost_usd * 100).toFixed(1)}%</div>
          </div>
        </div>
        ${p.routed.response_preview ? `
          <div style="margin-bottom:1rem">
            <strong>Response Preview:</strong>
            <div style="background:#f8fafc;padding:1rem;border-radius:0.5rem;margin-top:0.5rem;font-size:0.875rem">${p.routed.response_preview.substring(0, 300)}${p.routed.response_preview.length > 300 ? '...' : ''}</div>
          </div>
        ` : ''}
        <div>
          <strong>Evaluation Scores:</strong>
          <div style="margin-top:0.5rem;font-size:0.875rem">
            Heuristic: ${p.routed.heuristic_score.toFixed(3)} · 
            GPT-4o Judge: ${p.routed.judge_score.toFixed(3)} · 
            Combined: ${p.routed.quality_score.toFixed(3)}
          </div>
        </div>
        ${p.routed.judge_rationale ? `
          <div style="margin-top:1rem">
            <strong>GPT-4o Judge Rationale:</strong>
            <div style="background:#fef3c7;padding:1rem;border-radius:0.5rem;margin-top:0.5rem;font-size:0.875rem;line-height:1.6">${p.routed.judge_rationale}</div>
          </div>
        ` : ''}
      </div>
    </div>
  `).join('');
  document.getElementById('audit-table').innerHTML = html;
}

function setupEventListeners() {
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
      
      if (tab.dataset.tab === 'analytics') {
        setTimeout(() => {
          Plotly.Plots.resize('chart-cumulative');
          Plotly.Plots.resize('chart-tier-dist');
          Plotly.Plots.resize('chart-scatter');
        }, 100);
      }
    });
  });

  document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('prompt-input').value = examples[btn.dataset.example];
    });
  });

  document.getElementById('route-btn').addEventListener('click', routePrompt);

  document.querySelectorAll('.pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      filterAudit();
    });
  });

  document.getElementById('audit-search').addEventListener('input', filterAudit);
}

async function routePrompt() {
  const prompt = document.getElementById('prompt-input').value.trim();
  if (!prompt) return alert('Enter a prompt');

  const availableProviders = [];
  if (document.getElementById('provider-openai').checked) availableProviders.push('openai');
  if (document.getElementById('provider-anthropic').checked) availableProviders.push('anthropic');
  if (document.getElementById('provider-ollama').checked) availableProviders.push('ollama');

  const prediction = classifier.predict(prompt);
  const tier = classifier.getTierForComplexity(prediction.score);
  const modelSelection = classifier.getModelForTier(tier, availableProviders);

  if (!modelSelection) {
    return alert('No available providers for this tier. Enable at least one provider.');
  }

  const inputTokens = Math.floor(prompt.split(/\s+/).length * 1.3);
  const outputTokens = 150;
  const cost = classifier.calculateCost(inputTokens, outputTokens, modelSelection.model);
  const baseline = classifier.calculateFrontierBaselineCost(inputTokens, outputTokens);
  const savings = baseline - cost;
  const savingsPct = baseline > 0 ? (savings / baseline * 100) : 0;

  const words = prompt.split(/\s+/).length;
  const sentences = prompt.split(/[.!?]+/).filter(s => s.trim()).length;
  const signals = [];
  if (prompt.includes('```')) signals.push('code blocks');
  if (/\b(design|architecture|system)\b/i.test(prompt)) signals.push('system design');
  if (/\b(algorithm|complexity)\b/i.test(prompt)) signals.push('algorithmic');
  if (/\b(sql|query|select|database)\b/i.test(prompt)) signals.push('SQL/database');

  const openaiKey = document.getElementById('openai-key').value.trim();
  const isOpenAI = modelSelection.model.startsWith('openai/');

  let resultHtml = `
    <h3>Routing Decision</h3>
    <div class="result-grid">
      <div class="result-metric">
        <div class="result-metric-value">${prediction.score.toFixed(4)}</div>
        <div class="result-metric-label">Complexity Score</div>
      </div>
      <div class="result-metric">
        <div class="result-metric-value">${prediction.tier}</div>
        <div class="result-metric-label">Predicted Complexity</div>
      </div>
      <div class="result-metric">
        <div class="result-metric-value">${tier}</div>
        <div class="result-metric-label">Selected Tier</div>
      </div>
      <div class="result-metric">
        <div class="result-metric-value">${modelSelection.model.split('/')[1]}</div>
        <div class="result-metric-label">Selected Model${modelSelection.fallback ? ' (fallback)' : ''}</div>
      </div>
    </div>
    <div style="margin:1rem 0;padding:1rem;background:white;border-radius:0.5rem;border:1px solid #e5e7eb">
      <strong>Probabilities:</strong>
      ${Object.entries(prediction.probabilities).map(([k, v]) =>
        `<div style="margin:0.5rem 0"><span style="display:inline-block;width:80px">${k}:</span>
        <div style="display:inline-block;width:200px;height:20px;background:#e5e7eb;border-radius:4px;overflow:hidden">
          <div style="width:${(v * 100).toFixed(1)}%;height:100%;background:#059669"></div>
        </div> ${(v * 100).toFixed(1)}%</div>`
      ).join('')}
    </div>
    <div style="margin:1rem 0;padding:1rem;background:white;border-radius:0.5rem;border:1px solid #e5e7eb">
      <strong>Structural Signals:</strong>
      <div style="margin-top:0.5rem;font-size:0.875rem">
        ${words} words · ${sentences} sentences${signals.length > 0 ? ` · ${signals.join(', ')}` : ''}
      </div>
    </div>
    <div style="margin:1rem 0;padding:1rem;background:white;border-radius:0.5rem;border:1px solid #e5e7eb">
      <strong>Cost Analysis:</strong>
      <div style="margin-top:0.5rem">Estimated: $${cost.toFixed(6)} · Baseline: $${baseline.toFixed(6)} · <span style="color:#059669;font-weight:600">Savings: ${savingsPct.toFixed(1)}%</span></div>
    </div>
  `;

  if (openaiKey && isOpenAI) {
    resultHtml += `<div id="completion-status" style="margin:1rem 0;padding:1rem;background:#eff6ff;border:1px solid #3b82f6;border-radius:0.5rem">
      <strong>Fetching real OpenAI completion...</strong>
      <div style="margin-top:0.5rem;font-size:0.875rem">Model: ${modelSelection.model}</div>
    </div>`;
    
    document.getElementById('result').innerHTML = resultHtml;
    document.getElementById('result').classList.remove('hidden');

    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${openaiKey}`
        },
        body: JSON.stringify({
          model: modelSelection.model.replace('openai/', ''),
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.7,
          max_tokens: 500
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error?.message || `API error: ${response.status}`);
      }

      const data = await response.json();
      const completion = data.choices[0].message.content;
      const actualInputTokens = data.usage.prompt_tokens;
      const actualOutputTokens = data.usage.completion_tokens;
      const actualCost = classifier.calculateCost(actualInputTokens, actualOutputTokens, modelSelection.model);
      const actualBaseline = classifier.calculateFrontierBaselineCost(actualInputTokens, actualOutputTokens);
      const actualSavings = actualBaseline - actualCost;
      const actualSavingsPct = actualBaseline > 0 ? (actualSavings / actualBaseline * 100) : 0;

      document.getElementById('completion-status').innerHTML = `
        <strong>Real OpenAI Completion:</strong>
        <div style="margin-top:0.5rem;padding:1rem;background:white;border-radius:0.5rem;white-space:pre-wrap;font-size:0.875rem;line-height:1.6">${completion}</div>
        <div style="margin-top:1rem;font-size:0.875rem">
          <strong>Actual Usage:</strong> ${actualInputTokens} input + ${actualOutputTokens} output tokens<br>
          <strong>Actual Cost:</strong> $${actualCost.toFixed(6)} (baseline: $${actualBaseline.toFixed(6)}) · 
          <span style="color:#059669;font-weight:600">Saved ${actualSavingsPct.toFixed(1)}%</span>
        </div>
      `;
    } catch (error) {
      document.getElementById('completion-status').innerHTML = `
        <strong style="color:#dc2626">OpenAI API Error:</strong>
        <div style="margin-top:0.5rem;font-size:0.875rem">${error.message}</div>
        <div style="margin-top:0.5rem;font-size:0.875rem;color:#64748b">Check your API key and ensure you have available credits.</div>
      `;
    }
  } else if (openaiKey && !isOpenAI) {
    resultHtml += `<div style="margin:1rem 0;padding:1rem;background:#fef3c7;border:1px solid #fbbf24;border-radius:0.5rem">
      <strong>Note:</strong> OpenAI key provided but routed to ${modelSelection.model}. 
      To test with OpenAI, disable Anthropic/Ollama providers or use a prompt that routes to OpenAI tier.
    </div>`;
    
    document.getElementById('result').innerHTML = resultHtml;
    document.getElementById('result').classList.remove('hidden');
  } else {
    resultHtml += `<div style="margin:1rem 0;padding:1rem;background:#fef3c7;border:1px solid #fbbf24;border-radius:0.5rem">
      <strong>Mode:</strong> Mock preview (100% real classifier, estimated costs)
      <div style="margin-top:0.5rem;font-size:0.875rem">Paste your OpenAI key above to run real completions.</div>
    </div>`;
    
    document.getElementById('result').innerHTML = resultHtml;
    document.getElementById('result').classList.remove('hidden');
  }
}

function filterAudit() {
  const searchTerm = document.getElementById('audit-search').value.toLowerCase();
  const activePill = document.querySelector('.pill.active').dataset.filter;

  document.querySelectorAll('.audit-row').forEach(row => {
    const tier = row.dataset.tier;
    const searchText = row.dataset.search;
    
    const tierMatch = activePill === 'all' || tier === activePill;
    const searchMatch = !searchTerm || searchText.includes(searchTerm);
    
    row.style.display = tierMatch && searchMatch ? 'block' : 'none';
  });
}

window.toggleAuditRow = function(idx) {
  const details = document.getElementById(`details-${idx}`);
  const chevron = document.getElementById(`chevron-${idx}`);
  
  if (details.style.display === 'none') {
    details.style.display = 'block';
    chevron.textContent = '▼';
  } else {
    details.style.display = 'none';
    chevron.textContent = '▶';
  }
};

init();
