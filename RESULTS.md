# Results Summary

> Consolidated findings as of 2026-02-04. For methodology details, see [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).

---

## Main Finding: Safety Architecture Spectrum

We discovered a **spectrum of safety robustness** across model families:

```
Vulnerability:  HIGH ◄─────────────────────────────────► LOW

               Llama-3.1   Mistral-7B   Qwen2.5-7B   Gemma-2-9B
               (100%)      (88.6%)      (40%*)       (0%**)

               ════════════════════════════════════════════════
               "LIQUID SAFETY"              "CRYSTALLIZED SAFETY"
               Single-layer bypass          Distributed, robust
```

\* Qwen: 40% with early-layer multi-layer steering  
\** Gemma: Resistant even to multi-layer (max tested: 4 layers)

---

## Detailed Results by Model

### 1. Mistral-7B-Instruct-v0.3

**Status:** ⚠️ Vulnerable (single-layer bypass)

| Experiment | Result | N | p-value |
|------------|--------|---|---------|
| Sign inversion (+α vs -α) | 6% vs 96% | 100 | <0.001 |
| Layer 24, α=-15 | **88.6%** jailbreak | 100 | <0.001 |
| Orthogonal control | 88.6% vs 16.7% (random) | 35 | <0.001 |

**Key layer:** L24 (middle-to-late transformer layers)

**Orthogonal Control Validation:**
- Extracted direction: 88.6% flip rate, 68.6% coherent
- Orthogonal random: 16.7% flip rate, 16.7% coherent
- **Difference: +71.9 pp** → Direction-specific, not magnitude artifact

---

### 2. Llama-3.1-8B-Instruct

**Status:** 🔴 Highly Vulnerable (threshold collapse)

| Layer | α=-10 | α=-15 | α=-20 |
|-------|-------|-------|-------|
| L20 | 28.6% | 88.6% | **100%** |
| L22 | 37.1% | 45.7% | **100%** |
| L24 | 37.1% | 68.6% | 88.6% |
| L26 | 48.6% | 20.0%* | 57.1% |

\* Non-monotonic behavior at L26 — possible interference with other mechanisms

**Threshold Collapse:**
```
α=8:  32% jailbreak
α=9:  32% jailbreak
α=10: 66% jailbreak  ← Sharp transition (p=0.0007)
α=12: 92% jailbreak
α=14: 100% jailbreak (saturation)
```

**Key insight:** Llama's safety depends on a narrow activation band. Small perturbations at the threshold cause complete collapse.

---

### 3. Qwen2.5-7B-Instruct

**Status:** 🟢 Robust (crystallized, requires multi-layer)

**Single-layer results:**
| Layer | α=-10 | α=-15 | α=-20 |
|-------|-------|-------|-------|
| All tested (L20-L26) | 0% | 0% | 0% |

**Multi-layer results (corrected sign):**
| Configuration | Layers | α | Flip Rate |
|---------------|--------|---|-----------|
| quad_early | [12,16,20,24] | -12 | **40%** |
| six_layer | [8,12,16,20,24,27] | -6 | 10% |
| single_L24_strong | [24] | -25 | 0% |
| dense_late | [20-27] | -5 | 0% |

**Key insight:** Early layers (12-24) are vulnerable to coordinated attack; late layers (20-27) remain resistant.

---

### 4. Gemma-2-9B-Instruct

**Status:** 🟢 Most Robust (crystallized safety architecture)

**Single-layer results:**
| Layer | α=-10 | α=-15 | α=-20 |
|-------|-------|-------|-------|
| All tested (L20-L32) | 0% | 0% | 0% |

**Multi-layer results:**
- Up to 4-layer combinations tested: **0%** jailbreak
- Aggressive configurations (L12-L28, high α): **0%** jailbreak

**Prior Cycle 3 finding (from different methodology):** 95%+ success with coordinated multi-layer steering — **requires re-validation with corrected sign**.

---

## Sign Inversion Discovery

**Critical finding:** Standard contrastive extraction produces a direction that *reinforces* safety.

```python
# Standard extraction (WRONG for bypass):
direction = mean(harmful_activations) - mean(harmless_activations)
steering = activations + α * direction  # with positive α → REINFORCES safety

# Correct for bypass:
steering = activations - α * direction  # subtract, or use negative α
```

**Evidence (Mistral-7B, n=100):**
| Condition | Jailbreak Rate | Interpretation |
|-----------|---------------|----------------|
| Baseline (no steering) | 15% | Natural refusal rate |
| Standard +α | 6% | **Reinforced** safety |
| Standard -α | 96% | Bypassed safety |
| Inverted +α | 99% | Bypassed safety |

---

## Methodological Validations

### Three-Gate Control System

All experiments pass:

1. **Direction Specificity:** Extracted direction outperforms random (Z-score >3)
2. **Coherence:** Outputs remain fluent (≥4.0/5.0 coherence rating)
3. **Statistical Power:** 95% CI excludes baseline

### Multi-LLM Council Validation

Experimental designs validated by 4-model council (Claude, GPT-4o, Gemini, Grok) before execution.

- Orthogonal control experiment: 🟢 PROCEED (78.8% confidence)
- Multi-layer Gemma: 🟢 PROCEED (recommended additional controls)

---

## Implications

### For Red-Teamers
- Validate steering direction empirically — naive extraction may reinforce safety
- Target liquid models (Llama, Mistral) with single-layer attacks
- Crystallized models (Qwen, Gemma) require coordinated multi-layer strategies

### For Defenders
- Extracted direction can be used for **defensive steering** (add to reinforce safety)
- Llama's threshold collapse suggests training instability — investigate regularization
- Gemma's architecture provides a template for robust safety implementation

### For Researchers
- The robustness hierarchy suggests architectural or training differences worth investigating
- "Liquid vs crystallized" provides a new framework for safety analysis
- Multi-layer coordination attacks open new research directions

---

## Data Availability

All raw results in JSON format:
```
results/
├── orthogonal_control_results.json     # Mistral orthogonal validation
├── mistral_sweep_results.json          # Full layer×α sweep
├── gemma_sweep_results.json            # Single-layer sweep
├── cycle3_multilayer_results.json      # Multi-layer coordination
└── 2026-02-03/                         # Latest experiments
    ├── exp02_mistral_sweep.json
    ├── qwen_aggressive_results.json
    ├── gemma_aggressive_results.json
    └── REPORT.md
```

---

## Known Limitations

1. **Sample size:** Most experiments n=35-100; some edge conditions need more power
2. **Gemma multi-layer:** Prior 95%+ result needs re-validation (sign bug discovered)
3. **Prompt diversity:** Currently using AdvBench-derived prompts; broader coverage planned
4. **Coherence metric:** Subjective; working on automated evaluation

---

## Next Steps (from ROADMAP.md)

- [ ] EXP-01: Expand dataset to 600+ stratified prompts
- [ ] EXP-06: Capability preservation benchmarks (MMLU, MT-Bench)
- [ ] EXP-10: Multi-layer coordination on Qwen/Gemma (re-run with corrected sign)
- [ ] EXP-14: Cross-model transfer experiments

---

*Last updated: 2026-02-04*
