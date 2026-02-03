# Crystallized Safety Research Log
**Project:** Mapping Safety Mechanism Architecture in LLMs  
**Researcher:** Marco Santarcangelo  
**Start Date:** 2026-01-31  
**Last Updated:** 2026-02-03

---

## Executive Summary

This research investigates how safety mechanisms are implemented across different LLM architectures. The core finding is that safety robustness varies dramatically by model family:

| Model | Single-Layer Vulnerability | Multi-Layer Required | Status |
|-------|---------------------------|---------------------|--------|
| Llama-3.1-8B | 🔴 100% bypass | No | Liquid Safety |
| Mistral-7B | 🟡 88.6% bypass | No | Liquid Safety |
| Qwen2.5-7B | 🟢 0% (immune) | **Yes** | Crystallized Safety |
| Gemma-2-9B | 🟢 0% (immune) | **Yes (95%+ per Cycle 3)** | Crystallized Safety |

---

## Key Findings

### 1. Sign Inversion Phenomenon
**Discovery:** Contrastive extraction (harmful - harmless) yields a direction that REINFORCES safety, not bypasses it.

| Model | Standard +α | Standard -α | Inverted +α |
|-------|------------|-------------|-------------|
| Mistral-7B | 6% (reinforces) | 96% (bypass) | 99% (bypass) |
| Qwen2.5-7B | 0% | 9% | 7% |

### 2. Direction Specificity (Orthogonal Control)
**Mistral-7B, Layer 24, α=-15:**
- Extracted direction: 88.6% flip rate
- Orthogonal direction: 16.7% flip rate
- **Difference: +71.9 pp** → Effect is direction-specific, not magnitude artifact

### 3. Layer-Specific Vulnerability

**Llama-3.1-8B:**
| Layer | α=-10 | α=-15 | α=-20 |
|-------|-------|-------|-------|
| 20 | 28.6% | 88.6% | 100% |
| 22 | 37.1% | 45.7% | 100% |
| 24 | 37.1% | 68.6% | 88.6% |
| 26 | 48.6% | 20.0% | 57.1% |

**Threshold Collapse (Llama):**
- Sharp transition at α=9→10 (32% → 66%, p=0.0007)
- Saturation at α≥14

### 4. Multi-Layer Breaking (Cycle 3 - PRIOR FINDING)
Per MEMORY.md:
- **Gemma-2-9B:** Single-layer immune, but **95%+ success with 4-layer coordinated steering**
- **Llama-3.1-8B:** 100% success with 4-layer steering

---

## Experimental Timeline

### 2026-01-31: Initial Experiments
- Direction extraction methodology established
- Orthogonal control validation on Mistral
- Council approved proceed with 78.8% confidence

### 2026-02-01: Sign Inversion Discovery
- Confirmed sign inversion on Mistral (6% vs 96-99%)
- Replicated on Qwen (0% vs 7-9%)
- Llama α sweep: threshold at α=9-10
- Paper draft v0.3 completed

### 2026-02-02: Orthogonal Control n=100
- Mistral: 88.6% extracted vs 16.7% orthogonal
- Validation complete: direction-specific effect confirmed

### 2026-02-03: Cross-Model Single-Layer Sweep
- Llama full sweep complete (n=35 per condition)
- Qwen single-layer: 0% all conditions (as expected)
- Gemma single-layer: 0% all conditions (as expected)
- **Next: Multi-layer experiments to break Qwen/Gemma immunity**

---

## Data Files

### Local Results
```
research/results/
├── llama_results.json      # 2026-02-03, single-layer sweep
├── qwen_results.json       # 2026-02-03, single-layer (0% all)
├── gemma_results.json      # 2026-02-03, single-layer (0% all)

results/
├── orthogonal_control_mistral_20260131_110910.json
```

### Papers & Drafts
```
paper_draft.md              # v0.3 with Wilson CIs and p-values
```

### Memory
```
memory/2026-02-01_session.md   # Sign inversion discovery session
MEMORY.md                      # Running experiment log
```

---

## Methodology

### Direction Extraction
```python
direction = mean(activations_harmful) - mean(activations_harmless)
direction = direction / direction.norm()  # Unit normalize
```

### Steering Application
```python
# In forward hook:
output[0] = output[0] + α * direction  # or - for bypass
```

### Three-Gate Control System
1. **Direction Specificity:** Compare vs orthogonal/random vectors
2. **Coherence Gate:** Ensure outputs are coherent (not just noise)
3. **Statistical Power:** n≥35, Wilson CIs, p-values

### Multi-Layer Steering (Cycle 3)
```python
# Apply steering to multiple layers simultaneously
for layer in target_layers:
    model.layers[layer].register_forward_hook(steering_hook)
```

---

## Infrastructure

### Vast.ai Instances (Current)
| ID | Label | Host | Status |
|----|-------|------|--------|
| 30878383 | qwen-multilayer | ssh5.vast.ai:38382 | running |
| 30878384 | gemma-validation | ssh6.vast.ai:38384 | running |
| 30883700 | random-vectors | ssh2.vast.ai:13700 | running (needs fix) |
| 30883701 | llama-sweep | ssh4.vast.ai:13700 | running |
| 30880147 | extra-experiments | ssh3.vast.ai:10146 | running |

### Cost Tracking
- Estimated spend to date: ~$5-10
- RTX 4090 instances: ~$0.23-0.28/hr

---

## Terminology

**Liquid Safety:** Safety mechanisms concentrated in specific layers; vulnerable to single-layer steering (Llama, Mistral)

**Crystallized Safety:** Safety distributed across network; requires multi-layer coordination to bypass (Qwen, Gemma)

**Refusal Direction:** The vector in activation space that represents "this is harmful content"

**Sign Inversion:** The phenomenon where contrastive extraction yields a safety-reinforcing direction, requiring negation to achieve bypass

---

## Next Steps (Priority Order)

1. ⏳ **Multi-layer steering on Qwen** - Test if coordinated 4-layer steering breaks immunity
2. ⏳ **Multi-layer steering on Gemma** - Replicate Cycle 3 finding (95%+)
3. 🔧 **Fix random-vectors instance** - libcudnn issue
4. 📊 **n=100 validation** - Statistical power for Llama findings
5. 📝 **Update paper** - Add cross-model comparison section

---

## References

- Arditi et al. 2024 - "Refusal in LLMs is mediated by a single direction"
- Zhao et al. 2025 - "How do LLMs Encode Harmfulness and Refusal Separately?"
- Rogue Scalpel 2025 - "Activation Steering Compromises LLM Safety"

---

*This log is the authoritative record of the Crystallized Safety research project.*
