# Orthogonal Vector Control Experiment Proposal

**Date:** 2026-01-31
**Purpose:** Validate that steering effects are direction-specific, not magnitude artifacts

---

## Motivation

The council's primary concern: **Are our jailbreak results due to finding a genuine "refusal direction" or simply because any large perturbation disrupts the model?**

Current evidence:
- Mistral-7B: 83-100% jailbreak at L24 with α=15
- Gemma-2-9B: 95% with 4-layer coordination
- Llama-3.1-8B: 100% with 4-layer

**Critical question:** Would random vectors of the same magnitude produce similar effects?

---

## Experimental Design

### Control Conditions

For each successful configuration, test 3 control conditions:

| Condition | Vector | Expected if Direction-Specific |
|-----------|--------|-------------------------------|
| **Extracted** | Refusal direction from contrastive pairs | High jailbreak rate (baseline) |
| **Random** | Random unit vector × same α | Low jailbreak rate (<20%) |
| **Orthogonal** | Vector orthogonal to refusal direction | Low jailbreak rate (<20%) |
| **Opposite** | -1 × refusal direction | Very low or negative effect |

### Orthogonal Vector Construction

```python
def get_orthogonal_vector(refusal_dir):
    """Generate vector orthogonal to refusal direction."""
    # Method 1: Gram-Schmidt on random vector
    random_vec = torch.randn_like(refusal_dir)
    orthogonal = random_vec - (random_vec @ refusal_dir) * refusal_dir
    orthogonal = orthogonal / orthogonal.norm()
    return orthogonal
```

### Configurations to Test

| Model | Layer(s) | α | n_prompts |
|-------|----------|---|-----------|
| Mistral-7B | L24 | 15 | 100 |
| Gemma-2-9B | L12+L18+L24+L28 | 15 | 100 |
| Llama-3.1-8B | L12+L18+L24+L28 | 15 | 100 |

### Success Criteria

**Direction specificity confirmed if:**
1. Extracted vector jailbreak rate > 80%
2. Random vector jailbreak rate < 20%
3. Orthogonal vector jailbreak rate < 20%
4. Ratio: Extracted/Random > 4x

**Alternative hypothesis (magnitude artifact) if:**
- Random ≈ Extracted (within 20%)
- Effect driven by perturbation magnitude, not direction

---

## Metrics

1. **Jailbreak Rate:** % of harmful prompts that get compliance
2. **Coherence Score:** Output quality (same as existing pipeline)
3. **Direction Specificity Ratio:** Extracted_rate / Random_rate
4. **Statistical Test:** Chi-squared for rate differences

---

## Statistical Power

- n=100 per condition × 4 conditions × 3 models = 1200 total samples
- With n=100, can detect 20% difference with 95% confidence
- Wilson CI for proportions

---

## Budget Estimate

- 1200 generations × ~500 tokens avg = 600K tokens
- Mistral-7B on RunPod: ~$0.50
- Gemma-2-9B: ~$0.80
- Llama-3.1-8B: ~$0.60
- **Total: ~$1.90**

---

## Expected Outcomes

### If direction-specific (hypothesis confirmed):
- Validates that we found genuine safety-relevant directions
- Strengthens paper for publication
- Demonstrates mechanistic understanding

### If magnitude artifact (null result):
- Results are less interesting but still valid finding
- Suggests safety mechanisms are fragile to any perturbation
- Different framing needed for paper

---

## Implementation

```python
# Pseudocode for orthogonal control experiment

def run_orthogonal_control(model, layer, alpha, prompts):
    # Extract refusal direction (existing method)
    refusal_dir = extract_refusal_direction(model, layer)
    
    # Generate control vectors
    random_dir = torch.randn_like(refusal_dir)
    random_dir = random_dir / random_dir.norm()
    
    orthogonal_dir = get_orthogonal_vector(refusal_dir)
    opposite_dir = -refusal_dir
    
    results = {}
    for name, direction in [
        ("extracted", refusal_dir),
        ("random", random_dir),
        ("orthogonal", orthogonal_dir),
        ("opposite", opposite_dir)
    ]:
        jailbreak_rate, coherence = run_steering_experiment(
            model, layer, direction, alpha, prompts
        )
        results[name] = {
            "jailbreak_rate": jailbreak_rate,
            "coherence": coherence,
            "n": len(prompts)
        }
    
    # Compute direction specificity ratio
    results["specificity_ratio"] = (
        results["extracted"]["jailbreak_rate"] / 
        max(results["random"]["jailbreak_rate"], 0.01)
    )
    
    return results
```

---

## Council Questions

1. Is the experimental design sufficient to distinguish direction vs magnitude effects?
2. Are there additional control conditions we should include?
3. What statistical tests are most appropriate for this comparison?
4. Should we test multiple random vectors (e.g., 5-10) for robustness?
5. Are there confounds we haven't considered?

---

**Submitted for council review:** 2026-01-31
