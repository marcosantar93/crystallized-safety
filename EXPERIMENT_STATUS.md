# Experiment Status

> Auto-generated: 2026-02-04 12:58 GMT-3

## Progress Summary

| Experiment | Status | Completion | Notes |
|------------|--------|------------|-------|
| **EXP-01** Dataset | ⏸️ Pending | 0% | Using AdvBench subset (~50 prompts) |
| **EXP-02** Layer×α Sweep | 🟡 Partial | 70% | Mistral ✅, Llama ✅, Qwen ✅, Gemma ✅ (single-layer) |
| **EXP-03** Null Controls | 🟡 Partial | 40% | Orthogonal control ✅ Mistral only |
| **EXP-04** Activation Patching | ⏸️ Pending | 0% | Requires EXP-02 completion |
| **EXP-05** Multi-metric Validation | ⏸️ Pending | 0% | Need LLM judge setup |
| **EXP-06** Capability Preservation | ⏸️ Pending | 0% | MMLU, MT-Bench benchmarks |
| **EXP-07** Extractor Sensitivity | ⏸️ Pending | 0% | Compare extraction methods |
| **EXP-10** Multi-layer Coordination | 🟡 Partial | 50% | Qwen 40% ✅, Gemma 0% ✅ |

## Completed Experiments

### EXP-02: Cross-Model Single-Layer Sweep

| Model | Best Config | Jailbreak Rate | Status |
|-------|-------------|----------------|--------|
| Llama-3.1-8B | L20, α=-20 | **100%** | ✅ Complete |
| Mistral-7B | L24, α=-15 | **88.6%** | ✅ Complete |
| Qwen2.5-7B | All configs | **0%** | ✅ Complete (immune) |
| Gemma-2-9B | All configs | **0%** | ✅ Complete (immune) |

### EXP-03: Orthogonal Control (Direction Specificity)

| Model | Extracted Dir | Random Dir | Δ | Status |
|-------|--------------|------------|---|--------|
| Mistral-7B | 88.6% | 16.7% | +71.9pp | ✅ Z>3 |
| Llama-3.1-8B | — | — | — | ⏸️ Pending |
| Qwen2.5-7B | — | — | — | ⏸️ Pending |

### EXP-10: Multi-Layer Coordination

| Model | Config | Jailbreak Rate | Status |
|-------|--------|----------------|--------|
| Qwen2.5-7B | [12,16,20,24] α=-12 | **40%** | ✅ Partial bypass |
| Gemma-2-9B | 4-layer configs | **0%** | ✅ Resistant |

## Priority Queue (Next to Run)

### 🔴 Critical (Tier 1)

1. **EXP-03 Extended**: Orthogonal control for Llama (validate direction specificity)
2. **EXP-06**: Capability preservation on Mistral (MMLU 5-shot)
3. **EXP-02 Extended**: Yi-1.5-9B full sweep

### 🟡 Important (Tier 2)

4. **EXP-10 Extended**: More aggressive multi-layer on Gemma (6-8 layers)
5. **EXP-08**: Dimensionality analysis (PCA rank comparison)

## Infrastructure

- **Current**: vast.ai (offline), migrating to AWS EC2
- **Instance**: g5.2xlarge (A10G 24GB) - launching
- **Estimated cost**: ~$1.50/hr

## Scripts Ready to Run

```bash
# EXP-03: Orthogonal control on Llama
python run_orthogonal_control.py --model meta-llama/Llama-3.1-8B-Instruct --layer 20 --alpha -15 --n 100

# EXP-06: Capability preservation
python scripts/run_capability_preservation.py --model mistralai/Mistral-7B-Instruct-v0.3

# EXP-10: Aggressive Gemma multi-layer
python scripts/run_gemma_aggressive.py --layers 8,12,16,20,24,28 --alpha -15
```

---

*Last updated: 2026-02-04*
