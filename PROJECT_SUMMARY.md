# Crystallized-Safety Project Summary

## Overview

**Crystallized-Safety** is an AI safety research project investigating **layer-specific vulnerabilities in LLM safety mechanisms** through activation steering attacks. The core finding demonstrates that safety mechanisms in instruction-tuned language models are layer-localized and systematically bypassable.

**Repository:** https://github.com/marcosantar93/crystallized-safety
**Publication Status:** v4 paper council-approved (9/10 readiness)

---

## Core Finding: Mistral-7B Vulnerability

The project's major discovery is that **Mistral-7B-Instruct exhibits an 83% jailbreak success rate** through layer-specific activation steering.

### Best Configuration
| Parameter | Value |
|-----------|-------|
| Model | Mistral-7B-Instruct-v0.3 |
| Target Layer | 24 (deep layer) |
| Steering Coefficient (α) | 15 |
| Jailbreak Success Rate | 83.3% |
| Coherence Score | 4.2/5.0 |

### Validation (Three-Gate Control System)
1. **Control 1 - Direction Specificity:** Extracted steering direction outperforms random by 89.6x
2. **Control 2 - Coherence:** Outputs remain fluent (4.2/5.0)
3. **Control 3 - Statistical Power:** 83.3% flip rate

---

## Experiments Conducted

### 1. Mistral-7B Parameter Sweep (COMPLETED)

**Date:** January 16, 2026
**Results File:** `results/mistral_sweep_results.json`
**Configurations Tested:** 28

**Grid:**
- Layers: 15, 18, 21, 24, 27
- Alpha values: 5, 10, 15, 20, 25, 30

**Top Results (Passing All Controls):**

| Layer | Alpha | Flip Rate | Status |
|-------|-------|-----------|--------|
| 24 | 15 | 83% | Best |
| 21 | 15 | 67% | Good |
| 24 | 10 | 67% | Good |
| 27 | 15 | 67% | Good |
| 27 | 20 | 57% | Moderate |

**Key Finding:** Vulnerability concentrated in layers 21-27, with layer 24 optimal.

---

### 2. Gemma-2-9B Comparison (COMPLETED)

**Date:** January 16, 2026
**Results File:** `results/gemma_sweep_results.json`
**Configurations Tested:** 11

**Key Finding:** Gemma is **resistant** to single-layer activation steering
- Best result: Layer 18, α=15 → 11% flip rate
- 0 of 11 configurations passed Control 3
- Suggests more robust safety mechanism design

---

### 3. Llama-3.1-8B Results (PARTIAL)

**Tested:** Layer 21, α=20
**Result:** 0% flip rate
**Conclusion:** Fully resistant to steering attacks

---

### 4. Validation Cycles 1-3 (LAUNCHED - Status Unknown)

Three validation experiments were launched on RunPod:

| Cycle | Purpose | Pod ID | Duration | Cost |
|-------|---------|--------|----------|------|
| 1 | Probing Classifiers | 54vxopm2i7r1ab | ~1.5h | $0.26 |
| 2 | Activation Patching | 51jvkjwuc8ze0t | ~3h | $0.51 |
| 3 | Multilayered Attacks | lrl7nkvf4z1kdj | ~6h | $2.04 |

**Goals:**
- **Cycle 1:** Validate L24 projection accuracy >85% vs random baseline
- **Cycle 2:** Test necessity/sufficiency of L24 for safety
- **Cycle 3:** Break Gemma/Llama resistance via 4-layer coordinated steering

**Status:** Pods terminated, results not recovered.

---

### 5. Cycle 4: Temporal Dynamics (APPROVED, NOT LAUNCHED)

**Budget:** $2.25
**Council Status:** Approved 2/3

**Planned Experiments:**
- Token-by-token steering analysis (n=50)
- Mistral attention head ablation (3200 prompts)
- Gemma attention head ablation (2842 prompts)
- Progressive steering decay test (600 prompts)
- Cross-model attention pattern comparison (150 prompts)

---

## Secondary Project: Empathy Geometry

*Note: This project has been moved to a separate repository.*

### Completed Work
- Council-approved proposal
- Complete synthetic data validation with 5 models
- Infrastructure setup (Docker, RunPod, EC2)
- Code implementation

### Empathy Bandwidth Findings (Synthetic Data)

| Model | Bandwidth | Dimensionality | Steering Range |
|-------|-----------|----------------|----------------|
| Gemma2-9B | 136.6 | 16 | 8.5 |
| Llama-3.1-8B | 127.0 | 14 | 9.1 |
| DeepSeek-R1-7B | 92.0 | 11 | 8.4 |
| Qwen2.5-7B | 67.3 | 10 | 6.7 |
| Mistral-7B | 36.3 | 6 | 6.0 |

**Results Location:** `experiments/results/empathy/`

---

## Methodology

### Steering Vector Extraction
Uses Contrastive Activation Addition (CAA) style extraction:
```
steering_vector = mean(harmful_activations) - mean(harmless_activations)
```

### Activation Steering During Generation
Hook inserted at target layer, modification applied:
```
h' = h + α·v
```
Where α is the scaling factor, applied at last token position.

### Three-Gate Control System
1. **Gate 1:** Direction specificity (extracted direction vs random)
2. **Gate 2:** Coherence preservation (outputs remain fluent)
3. **Gate 3:** Statistical power (jailbreak effectiveness)

---

## Key Implications

### Research Questions Addressed
1. **Is safety layer-localized?** YES - concentrated in layers 21-27
2. **Can it be bypassed with steering?** YES - 83% in Mistral-7B
3. **Does it maintain coherence?** YES - 4.2/5.0 coherence score
4. **Is this universal?** NO - Gemma & Llama show resistance

### Defense Recommendations
- Implement refusal mechanisms across all layers (distributed safety)
- Include activation steering in adversarial training
- Add runtime monitoring for anomalous activations
- Use ensemble approaches with multiple independent safeguards

---

## Project Structure

```
crystallized-safety/
├── pipeline.py                 # Main experimental pipeline
├── sweep_experiment.py         # Grid search across configs
├── run_adaptive_attacks.py     # Advanced attack variants
├── run_orthogonal_control.py   # Control experiments
│
├── src/
│   ├── extraction.py           # Steering vector extraction
│   ├── steering.py             # Steering hook implementation
│   ├── evaluation.py           # Coherence judging & metrics
│   └── models.py               # Model loading utilities
│
├── results/
│   ├── mistral_sweep_results.json   # 28 configs tested
│   └── gemma_sweep_results.json     # 11 configs tested
│
├── papers/
│   ├── ActivationSteering_CouncilApproved_v4.pdf  # Current version
│   └── [earlier versions...]
│
├── experiments/
│   └── results/empathy/        # Empathy geometry results
│
├── configs/                    # Configuration files
├── data/                       # Prompt datasets
└── figures/                    # Generated visualizations
```

---

## Budget Summary

| Item | Cost |
|------|------|
| Vast.ai A100 (6 days) | ~$55 |
| Vast.ai RTX A5000 (6 days) | ~$18 |
| Validation Cycles 1-3 | ~$2.81 |
| EC2 misc | ~$0.50 |
| **Total Spent** | **~$76** |
| **Remaining** | **~$145** |

---

## Next Steps

1. **Investigate Validation Cycle Results** - Recover or re-run Cycles 1-3
2. **Update Paper to v5** - Reflect actual validation status
3. **Analyze Results** - Determine if Cycle 3 broke Gemma/Llama resistance
4. **Consider Cycle 4** - Temporal dynamics analysis (council-approved)
5. **Prepare for Publication** - arXiv submission

---

*Last Updated: January 27, 2026*
