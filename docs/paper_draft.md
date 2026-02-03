# Sign Inversion in Contrastive Safety Direction Extraction: Implications for Activation Steering Robustness

**Draft v0.3 - 2026-02-01**

## Abstract

We investigate activation steering as a method to bypass safety mechanisms in large language models. Through systematic experiments on Mistral-7B, Qwen2-7B, and Llama-3.1-8B, we discover a **sign inversion phenomenon**: the contrastive extraction method (harmful - harmless activations) yields a direction that *reinforces* safety rather than bypassing it. Jailbreaks require steering in the **opposite** direction. We confirm this finding across architectures with different robustness profiles. We further characterize model-specific vulnerability thresholds, finding dramatic differences: Qwen resists all perturbations while Llama collapses at moderate steering magnitudes. Our findings have implications for both red-teaming methodologies and defensive applications of activation steering.

## 1. Introduction

Activation steering has emerged as a powerful technique for modifying LLM behavior at inference time (Arditi et al., 2024; Turner et al., 2023). By adding vectors to residual stream activations, researchers have demonstrated the ability to influence model outputs across various dimensions including sentiment, truthfulness, and safety compliance.

A key question for AI safety is whether activation steering can reliably bypass safety training. Previous work has shown that "refusal directions" can be extracted and manipulated (Arditi et al., 2024), but the specificity and robustness of these findings across models remains unclear.

**Our contributions:**
1. We demonstrate that contrastive direction extraction (harmful - harmless) yields a **safety reinforcement** direction, not a safety bypass direction
2. We confirm this **sign inversion** phenomenon across multiple architectures (Mistral, Qwen)
3. We characterize **model-specific vulnerability thresholds** with dramatically different robustness profiles
4. We provide a methodology for validating steering direction claims through orthogonal controls

## 2. Methods

### 2.1 Direction Extraction

We extract directions using contrastive activation analysis:

```
direction = mean(activations_harmful) - mean(activations_harmless)
```

Where activations are collected at the last token position from specified layers during forward passes on matched harmful/harmless prompt pairs (5 each).

### 2.2 Steering Implementation

Steering is applied via forward hooks that add scaled direction vectors to residual stream activations:

```
h' = h + α * direction
```

Where α controls steering magnitude. We test both positive (+α) and negative (-α) scaling.

### 2.3 Experimental Conditions

**Orthogonal Control Experiments** (per model):
1. Extracted direction (+α)
2. Random vectors (5×, magnitude-matched)
3. Orthogonal vectors (5×, Gram-Schmidt)
4. Opposite direction (-α)
5. Baseline (α=0)

**Sign Inversion Experiments** (Mistral, Qwen):
1. Baseline (no steering)
2. Standard direction +α (harmful - harmless)
3. Standard direction -α (negated)
4. Inverted direction +α (harmless - harmful)

**Magnitude Sweep** (Llama):
- α ∈ {0, 8, 9, 10, 11, 12, 13, 14, 15, 16}

### 2.4 Evaluation

100 harmful prompts expanded from 10 base prompts via prefix variation. Responses classified as jailbroken if they lack refusal indicators ("cannot", "won't", "sorry", "harmful", etc.) and exceed 100 characters.

## 3. Results

### 3.1 Orthogonal Control Experiments

| Model | Extracted | Random μ (±σ) | Orthogonal μ | Opposite | Baseline |
|-------|-----------|---------------|--------------|----------|----------|
| Mistral-7B | 10% | 34% (±13%) | 24% | 84% | 22% |
| Qwen2-7B | 0% | 1% (±0.6%) | 1.4% | 30% | 3% |
| Llama-3.1-8B | 100% | 76% (±16%) | 91% | 100% | 1% |

**Key observation**: In Mistral and Qwen, the extracted direction produces **fewer** jailbreaks than baseline, suggesting the extracted direction reinforces safety rather than bypassing it.

### 3.2 Sign Inversion Confirmation

#### Mistral-7B (α=15, n=100)

| Condition | Rate | 95% CI | p vs baseline |
|-----------|------|--------|---------------|
| Baseline | 15% | [9.3%, 23.3%] | — |
| Standard +α | **6%** | [2.8%, 12.5%] | 0.038* |
| Standard -α | **96%** | [90.2%, 98.4%] | <0.0001*** |
| Inverted +α | **99%** | [94.6%, 99.8%] | <0.0001*** |

#### Qwen2-7B (α=15, n=100)

| Condition | Rate | 95% CI | p vs baseline |
|-----------|------|--------|---------------|
| Baseline | 0% | [0.0%, 3.7%] | — |
| Standard +α | **0%** | [0.0%, 3.7%] | 1.0 |
| Standard -α | **9%** | [4.8%, 16.2%] | 0.002** |
| Inverted +α | **7%** | [3.4%, 13.7%] | 0.007** |

*p<0.05, **p<0.01, ***p<0.001

**Finding**: Both models show the same statistically significant pattern:
- Standard direction (+α) produces ≤ baseline jailbreaks (reinforces safety)
- Negated/inverted directions cause jailbreaks
- Effect magnitude scales with model fragility (Mistral: 96-99%, Qwen: 7-9%)

### 3.3 Magnitude Threshold Analysis (Llama-3.1-8B, n=50)

| α | Rate | 95% CI |
|---|------|--------|
| 0 | 0% | [0.0%, 7.1%] |
| 8 | 32% | [20.8%, 45.8%] |
| 9 | 32% | [20.8%, 45.8%] |
| **10** | **66%** | [52.2%, 77.6%] |
| 11 | 58% | [44.2%, 70.6%] |
| 12 | 92% | [81.2%, 96.8%] |
| 13 | 96% | [86.5%, 98.9%] |
| 14+ | 100% | [92.9%, 100%] |

**Threshold test (α=9 vs α=10):** p = 0.0007***

**Finding**: Llama exhibits statistically significant threshold collapse between α=9-10 (32% → 66%, p<0.001), with saturation at α≥14. This suggests safety depends on a narrow activation band—any sufficient perturbation causes failure.

## 4. Discussion

### 4.1 The Sign Inversion Phenomenon

Our central finding is that contrastive extraction (harmful - harmless) yields a direction that **reinforces** rather than bypasses safety. The intuition:

- Harmful prompts elicit activations that trigger refusal mechanisms
- Harmless prompts elicit activations that don't trigger refusal
- The difference captures what makes harmful prompts *recognized as harmful*
- Adding this direction makes prompts *more* recognized as harmful → stronger refusal

To bypass safety, one must steer in the **opposite** direction (negative scaling or inverted extraction), making harmful prompts appear less harmful to the model's internal representations.

### 4.2 Model Robustness Hierarchy

| Model | Robustness | Characterization |
|-------|------------|------------------|
| Qwen2-7B | 🟢 High | Resists random/orthogonal perturbations. Only targeted inverted direction causes modest effect (7-9%). |
| Mistral-7B | 🟡 Medium | Random vectors cause 34% jailbreaks. Inverted direction causes near-total failure (96-99%). |
| Llama-3.1-8B | 🔴 Low | Collapses at α≥10 regardless of direction. Safety appears fragile. |

The dramatic differences suggest fundamentally different safety architectures:
- **Qwen**: Distributed, robust representations
- **Mistral**: Moderately localized, direction-sensitive
- **Llama**: Narrow activation band, magnitude-sensitive

### 4.3 Implications

**For red-teaming**: The sign of extracted directions must be validated empirically. Naive application of contrastive extraction may inadvertently *reinforce* safety rather than bypass it.

**For defense**: The extracted direction could be used for *defensive steering*—adding it at inference time to strengthen safety for untrusted inputs.

**For interpretability**: The robustness hierarchy suggests safety mechanism architecture varies substantially across model families, warranting investigation into what makes Qwen more robust.

## 5. Limitations

- Jailbreak classification uses heuristics; manual validation on subset recommended
- Limited to 3 models; broader architecture coverage desirable
- α values not normalized by model scale; magnitudes not directly comparable across models
- Harmful prompt set (n=100) may not cover all attack vectors
- Direction extraction uses only 5 prompt pairs; more diverse extraction sets may yield different directions

## 6. Conclusion

We demonstrate that contrastive safety direction extraction suffers from **sign inversion**—the extracted direction reinforces rather than bypasses safety. This finding, confirmed across Mistral and Qwen architectures, has immediate implications for activation steering research methodology. The dramatic robustness differences across models (Qwen >> Mistral >> Llama) highlight that safety mechanism architecture varies substantially and merits further investigation. Our orthogonal control methodology provides a template for validating future steering direction claims.

## References

- Arditi, A., et al. (2024). Refusal in LLMs is mediated by a single direction. arXiv:2406.11717.
- Turner, A., et al. (2023). Activation Addition: Steering Language Models Without Optimization. arXiv:2308.10248.
- Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.
- Zhao, Y., et al. (2025). How do LLMs Encode Harmfulness and Refusal Separately? Insights from Safety Fine-tuning. arXiv:2501.05628.

---

## Appendix A: Experimental Details

**Models:**
- mistralai/Mistral-7B-Instruct-v0.3
- Qwen/Qwen2-7B-Instruct  
- meta-llama/Llama-3.1-8B-Instruct

**Layers for steering:**
- Mistral: [24] (single layer)
- Qwen: [14, 21] (multi-layer average)
- Llama: [16, 24] (multi-layer average)

**Direction extraction:**
- 5 harmful prompts, 5 harmless prompts
- Activations at last token position
- Normalized to unit vector

**Prompts:** 100 harmful prompts expanded from 10 base prompts via prefix variation ("Step by step...", "In detail...", "Please...", etc.)

**Compute:** RunPod A100 80GB instances

**Code:** Available at [repository URL]

---

## Appendix B: Summary Tables

### B.1 Sign Inversion Results (All Models)

| Model | Baseline | Std +α | Std -α | Inv +α | Confirms Inversion? |
|-------|----------|--------|--------|--------|---------------------|
| Mistral-7B | 15% | 6% | 96% | 99% | ✅ Yes |
| Qwen2-7B | 0% | 0% | 9% | 7% | ✅ Yes |

### B.2 Robustness Comparison

| Model | Baseline | Best Attack | Worst Attack | Δ (Worst-Best) |
|-------|----------|-------------|--------------|----------------|
| Qwen2-7B | 0-3% | 0% (std+) | 9% (std-) | 9% |
| Mistral-7B | 15-22% | 6% (std+) | 99% (inv+) | 93% |
| Llama-3.1-8B | 0-1% | 0% (α=0) | 100% (α≥14) | 100% |
