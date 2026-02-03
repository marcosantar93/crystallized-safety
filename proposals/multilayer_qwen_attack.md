# Multi-Layer Coordinated Attack on Qwen

## Background
- **Qwen2-7B** is resistant to single-layer steering (14% baseline, no change at any α)
- **Gemma-2-9B** showed similar resistance but was broken by multi-layer coordination (Cycle 3 results: 95%+ success)
- **Hypothesis**: Qwen's safety is distributed across layers, requiring coordinated perturbation

## Proposed Experiment

### Phase 1: Layer Vulnerability Mapping
Extract refusal direction at each of Qwen's 28 layers and measure per-layer jailbreak rate at α=15.

**Expected outcome**: Identify layers with partial effects (e.g., 20-30% jailbreak vs 14% baseline)

### Phase 2: Coordinated Multi-Layer Steering
Apply steering vectors simultaneously to 2, 3, and 4 layers:
- Test combinations of high-partial-effect layers
- Vary relative magnitudes (α_1, α_2, ...) 
- Search for synergistic combinations

### Phase 3: Gradient-Based Layer Selection
Use gradient of jailbreak rate w.r.t. layer activation to identify optimal intervention points.

## Experimental Parameters

```python
QWEN_CONFIG = {
    "model": "Qwen/Qwen2-7B-Instruct",
    "total_layers": 28,
    "n_samples": 30,
    "alpha_per_layer": 15,
    "layer_combinations": [
        (L1, L2) for L1, L2 in combinations if abs(L1-L2) > 3
    ],
    "max_layers": 4
}
```

## Controls Required
1. **Additivity check**: Does multi-layer effect = sum of single-layer effects?
2. **Coherence monitoring**: Track output quality at each combination
3. **Direction orthogonality**: Verify extracted directions across layers are related

## Resource Estimate
- Layer sweep: ~45 min (28 layers × 30 samples)
- 2-layer combinations: ~3h (C(10,2) × 30 samples for top-10 layers)
- 3-layer combinations: ~8h (C(10,3) × 30 samples)
- **Total: ~12h on single RTX 3090**

## Success Criteria
- **GREEN**: >70% jailbreak rate with coherent outputs
- **YELLOW**: 40-70% jailbreak rate 
- **RED**: <40% or majority incoherent outputs

## Questions for Council
1. Is the layer combination search space reasonable?
2. Should we use PCA on layer directions to find "master" steering direction?
3. Any precedent in literature for multi-layer coordinated attacks on Qwen specifically?
