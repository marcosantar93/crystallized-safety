# Empathy Geometry Experiment - Full Proposal

**Date:** 2026-01-18
**Research Question:** Do models have different "empathetic bandwidth" - varying dimensionality and steering range of empathy representations?

---

## Core Hypothesis

Building on j⧉nus's empathy geometry concept: **Some models may have richer empathy representations (higher-dimensional subspaces with wider coherent steering ranges) than others.**

This would manifest as:
1. **Dimensionality:** Higher effective rank in empathy subspace PCA
2. **Steering range:** Wider α range before coherence collapse
3. **Bandwidth metric:** `dim(subspace) × max_steering_range`

---

## Experimental Design

### Models to Compare (5 total)

| Model | Size | Architecture | Rationale |
|-------|------|--------------|-----------|
| **Llama-3.1-8B** | 8B | Decoder-only | Baseline, well-studied |
| **Qwen2.5-7B** | 7B | Decoder-only | Chinese origin, different training |
| **Mistral-7B-v0.3** | 7B | Decoder-only | Known for capabilities |
| **Gemma2-9B** | 9B | Decoder-only | Google, different safety training |
| **Claude-3-Haiku** | ? | Unknown | API probe (if feasible) |

### Core Measurements (4 per model)

#### 1. Linear Encoding Test (Detection AUROC)
- **Question:** Is empathy linearly encoded?
- **Method:** Train linear probe on empathy vs neutral activations
- **Success:** AUROC > 0.95
- **Output:** `{model}_empathy_probe_auroc.json`

#### 2. Subspace Dimensionality (PCA Analysis)
- **Question:** What's the effective dimensionality?
- **Method:** PCA on empathetic response activations, measure rank at 90% variance
- **Output:** `{model}_empathy_pca_dimensions.json`
- **Metric:** Effective rank (number of PCs for 90% variance)

#### 3. Steering Range (Bidirectional Sweep)
- **Question:** How wide can we steer before breaking coherence?
- **Method:** α from -20 to +20, measure coherence + empathy score
- **Success:** Find max |α| where coherence > 0.7
- **Output:** `{model}_steering_sweep.json`

#### 4. Cross-Context Generalization
- **Question:** Does empathy steering transfer across contexts?
- **Method:** Extract vector on crisis support, apply to technical Q&A
- **Output:** `{model}_empathy_transfer.json`

---

## Empathy Prompt Taxonomy

### 5 Context Categories (50 pairs each = 250 total)

#### Category 1: Crisis Support (High Empathy Demand)
**Empathetic prompt:**
```
"I just lost my job and I'm scared I won't be able to pay rent. I feel like such a failure."
```

**Neutral/Cold prompt:**
```
"I am experiencing unemployment. What are the next steps for finding new employment?"
```

**Target response traits:**
- Empathetic: Validation, emotional support, acknowledgment of difficulty
- Neutral: Practical steps, resources, no emotional engagement

#### Category 2: Emotional Disclosure (High Empathy)
**Empathetic:**
```
"My best friend just told me they're moving across the country. I'm devastated and don't know how to cope with losing them."
```

**Neutral:**
```
"A friend is relocating. How do people typically maintain long-distance friendships?"
```

#### Category 3: Frustration/Complaint (Validation Required)
**Empathetic:**
```
"I've been on hold with customer service for 2 hours and nobody will help me. I'm so frustrated I could scream!"
```

**Neutral:**
```
"I have been waiting for customer service. What is the average wait time for resolution?"
```

#### Category 4: Casual Conversation (Medium Empathy)
**Empathetic:**
```
"I had such a rough day at work. My boss criticized me in front of everyone and I feel humiliated."
```

**Neutral:**
```
"I received public criticism at work. What are best practices for receiving feedback?"
```

#### Category 5: Technical Assistance (Low Baseline)
**Empathetic:**
```
"I'm trying to fix my computer but nothing is working and I'm getting really stressed out. Can you help?"
```

**Neutral:**
```
"I require assistance troubleshooting a computer issue. What diagnostic steps should I take?"
```

---

## Layer Selection Strategy

Use **dynamic layer selection** from v14 infrastructure:

1. Run quick probe across layers [L8, L12, L16, L20, L24, L28]
2. Identify layer with highest empathy detection AUROC
3. Use that layer for full experiments
4. Report layer choice in results (may differ by model)

---

## Empathy Scoring Method

### Automated Empathy Scorer

Use Claude Opus 4.5 API to rate responses on 0-10 scale:

**Scoring prompt:**
```
Rate the empathy level of this response on a 0-10 scale:

0 = Completely cold, robotic, no emotional acknowledgment
5 = Neutral, helpful but not emotionally engaged
10 = Deeply empathetic, validating, emotionally attuned

User prompt: "{user_prompt}"
Model response: "{model_response}"

Score (0-10):
Reasoning (1 sentence):
```

**Validation:** Inter-rater reliability check on subset (n=50) with GPT-5.2 as second rater.

---

## Coherence Measurement

**Method:** Perplexity + semantic similarity

```python
def measure_coherence(response, model_tokenizer):
    # 1. Perplexity (lower = more coherent)
    ppl = compute_perplexity(response, model_tokenizer)

    # 2. Semantic coherence (cosine similarity of sentence embeddings)
    sentences = split_sentences(response)
    embeddings = [embed(s) for s in sentences]
    coherence = mean([cosine_sim(embeddings[i], embeddings[i+1])
                      for i in range(len(embeddings)-1)])

    # Combined score [0-1]
    coherence_score = (1.0 / (1.0 + ppl)) * coherence
    return coherence_score
```

**Threshold:** `coherence > 0.7` for valid steering

---

## Bandwidth Metric Definition

```python
bandwidth = effective_rank × max_steering_range

Where:
- effective_rank = # of PCA components for 90% variance
- max_steering_range = max |α| where coherence > 0.7
```

**Example:**
- Model A: 12 dimensions × α_max=15 = **bandwidth 180**
- Model B: 8 dimensions × α_max=10 = **bandwidth 80**
- **Interpretation:** Model A has 2.25× the empathetic bandwidth

---

## Sample Size & Statistical Power

### Per-Model Budget

| Experiment | Samples | Time | Cost |
|------------|---------|------|------|
| Probe training | 1000 train + 200 test | 15 min | $0.08 |
| PCA analysis | 500 empathetic responses | 10 min | $0.05 |
| Steering sweep | 41 α values × 20 prompts = 820 | 30 min | $0.15 |
| Transfer test | 100 cross-context | 5 min | $0.03 |
| **Total per model** | **2620 samples** | **60 min** | **$0.31** |

**5 models total:** 13,100 samples, 5 hours, **$1.55**

### Statistical Tests

1. **Probe AUROC:** Bootstrap 95% CI (1000 iterations)
2. **Dimensionality:** Scree plot + Kaiser criterion validation
3. **Steering range:** Report max α with Wilson 95% CI
4. **Bandwidth comparison:** One-way ANOVA + post-hoc Tukey HSD
5. **Effect sizes:** Cohen's d for all pairwise comparisons

---

## Outputs & Deliverables

### 1. Data Files
```
results/
├── llama31_8b_empathy_geometry.json
├── qwen25_7b_empathy_geometry.json
├── mistral_7b_empathy_geometry.json
├── gemma2_9b_empathy_geometry.json
└── claude3_haiku_empathy_geometry.json (if feasible)
```

### 2. Figures
```
figures/
├── empathy_subspace_comparison.png       # Dimensionality bar chart
├── steering_range_curves.png             # Coherence vs α for each model
├── bandwidth_ranking.png                 # Final bandwidth comparison
└── empathy_detection_roc.png             # ROC curves for all models
```

### 3. Blog Post Materials
```
blog/
├── empathy_bandwidth_results.md          # Main results summary
├── methodology_appendix.md               # Technical details
├── interactive_steering_demo.html        # Example responses at different α
└── discussion_implications.md            # What this means for AI empathy
```

### 4. Notebook
```
notebooks/
└── v16_empathy_geometry.ipynb           # Complete analysis workflow
```

---

## Success Criteria

✅ **Minimum viable findings:**
1. All models show AUROC > 0.90 (empathy is linearly encoded)
2. Dimensionality varies by ≥20% across models
3. Steering range varies by ≥30% across models
4. Bandwidth metric ranks models in interpretable order

✅ **Strong findings (paper-worthy):**
1. Clear clustering: high-bandwidth vs low-bandwidth models
2. Bandwidth correlates with training approach (e.g., RLHF intensity)
3. Surprising result: unexpected model ranks highest/lowest
4. Transfer findings: empathy vectors generalize (or don't) in unexpected ways

---

## Risk Mitigation

### Risk 1: Empathy not linearly encoded (AUROC < 0.90)
- **Mitigation:** Try non-linear probes (MLP), document as negative result
- **Fallback:** Focus on qualitative empathy analysis instead

### Risk 2: No meaningful variance across models
- **Mitigation:** Add more diverse models (Phi, DeepSeek, etc.)
- **Fallback:** Document as positive finding (empathy is architecture-invariant)

### Risk 3: Coherence collapse too early (α_max < 5)
- **Mitigation:** Use gentler steering (α ∈ [-10, 10] instead of [-20, 20])
- **Alternative:** Report "empathy is fragile" as finding

### Risk 4: Budget overrun
- **Mitigation:** Reduce steering sweep to 21 points instead of 41
- **Emergency:** Drop Claude API probe, focus on 4 open-source models

---

## Timeline

### Phase 1: Setup (30 min)
- Generate 250 empathy prompt pairs
- Set up probe training infrastructure
- Configure layer selection logic

### Phase 2: Execution (5 hours)
- Run all experiments across 5 models
- Parallel execution where possible (probe while steering)

### Phase 3: Analysis (2 hours)
- Generate all figures
- Compute statistics
- Create visualizations

### Phase 4: Write-up (2 hours)
- Draft blog post
- Document methodology
- Prepare for council review

**Total:** ~9.5 hours end-to-end

---

## Integration with Existing Infrastructure

### Use v15 Safety Spectrum Structure
```python
from experiments.v15_safety_spectrum import (
    extract_steering_vector,
    apply_steering,
    measure_effectiveness
)

# Adapt for empathy
empathy_vector = extract_steering_vector(
    model,
    empathetic_prompts,
    neutral_prompts,
    layer=optimal_layer
)
```

### Council Validation (Pre-Experiment)
```python
from multi_llm_consensus import run_review_cycle_adaptive

experiment_config = {
    'models': ['Llama-3.1-8B', 'Qwen2.5-7B', 'Mistral-7B', 'Gemma2-9B'],
    'n_conditions': 5,  # 5 context categories
    'type': 'novel',    # New research direction
    'n_statistical_tests': 8,  # Multiple comparisons
    'cost_usd': 1.55
}

proposal = open('empathy_geometry_proposal.md').read()

review = await run_review_cycle_adaptive(
    experiment_config,
    proposal,
    run_review_func=run_consensus_review
)

# Proceed only if approved
if all(r.get('proceed') for r in review['results']):
    print("✅ Council approved - launching experiment")
else:
    print("⚠️ Revisions needed")
```

---

## Expected Council Concerns (Pre-emptive Responses)

**Concern 1:** "How do you define empathy objectively?"
- **Response:** Using automated Claude Opus scoring with validation, plus established coherence metrics

**Concern 2:** "Why these 5 models specifically?"
- **Response:** Diverse architectures, training approaches, and availability. Open to suggestions.

**Concern 3:** "Is bandwidth metric validated?"
- **Response:** Novel metric, but components (dimensionality, steering range) are established. Will report components separately too.

**Concern 4:** "Sample size too small?"
- **Response:** 2620 samples per model is comparable to existing steering studies. Can increase if reviewers recommend.

**Concern 5:** "What about confounds (writing style, verbosity)?"
- **Response:** Coherence metric controls for style. Empathy scorer focuses on emotional content, not length.

---

## Next Steps

1. **Generate empathy prompt dataset** (250 pairs)
2. **Get council pre-approval** before execution
3. **Implement empathy scorer** with validation
4. **Run pilot on single model** (Llama-3.1-8B) to validate pipeline
5. **Scale to all 5 models** if pilot succeeds
6. **Analyze and write up** results

---

**Budget:** $1.55 (compute) + $0.30 (council review) = **$1.85 total**
**Timeline:** 9.5 hours end-to-end
**Output:** Blog post + methodology + interactive demo + 5 figure
