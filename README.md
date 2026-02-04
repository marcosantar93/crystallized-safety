# Crystallized Safety

### Mapping Safety Mechanism Architecture in LLMs via Activation Steering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Key Finding:** Safety robustness varies dramatically across model families. Some models have "liquid" safety (single-layer bypass possible), while others have "crystallized" safety (distributed, robust architecture).

| Model | Single-Layer Bypass | Status |
|-------|:------------------:|--------|
| Llama-3.1-8B | **100%** | 🔴 Liquid |
| Mistral-7B | **88.6%** | 🟡 Vulnerable |
| Qwen2.5-7B | **0%** | 🟢 Crystallized |
| Gemma-2-9B | **0%** | 🟢 Crystallized |

📖 **New here?** Start with [QUICKSTART.md](QUICKSTART.md) | 📊 **Full results:** [RESULTS.md](RESULTS.md)

---

## Overview

This repository contains code and results for our systematic study of activation steering as a method for understanding (and potentially bypassing) safety mechanisms in instruction-tuned language models.

### Major Findings

1. **Sign Inversion:** Contrastive extraction (`harmful - harmless`) produces a direction that *reinforces* safety when added. To bypass safety, you must *subtract* it (use negative α) or invert the extraction.

2. **Model Robustness Hierarchy:**
   - 🟢 **Qwen2-7B:** Highly robust (max 9% jailbreak even with inverted steering)
   - 🟡 **Mistral-7B:** Moderate vulnerability (96%+ jailbreak with inverted steering)
   - 🔴 **Llama-3.1-8B:** Fragile threshold (collapses at α≥10 regardless of direction)

3. **Threshold Collapse:** Llama-3.1-8B exhibits a sharp transition from robust (32% at α=9) to broken (66%+ at α=10), suggesting safety depends on a narrow activation band.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run sign inversion experiment on Mistral
python run_sign_inversion.py --model mistralai/Mistral-7B-Instruct-v0.3

# Run full sweep across models
python sweep_experiment.py --model <model_name>
```

## Key Results

### Sign Inversion (Mistral-7B, n=100)

| Condition | Jailbreak Rate | 95% CI |
|-----------|----------------|--------|
| Baseline (no steering) | 15% | [9%, 23%] |
| Standard direction +α | **6%** | [3%, 13%] |
| Standard direction −α | **96%** | [90%, 98%] |
| Inverted direction +α | **99%** | [95%, 100%] |

The "refusal direction" actually **reduces** jailbreaks below baseline (p=0.038).

### Model Comparison (Inverted Steering)

| Model | Jailbreak Rate | Characterization |
|-------|----------------|------------------|
| Qwen2-7B | 9% | Distributed safety architecture |
| Mistral-7B | 96-99% | Single-direction vulnerability |
| Llama-3.1-8B | 100% (at α≥10) | Threshold collapse |

### Llama Threshold Collapse

| α | Jailbreak Rate |
|---|----------------|
| 8 | 32% |
| 9 | 32% |
| **10** | **66%** |
| 12 | 92% |
| 14 | 100% |

Sharp transition at α=10 (p=0.0007).

## Repository Structure

```
crystallized-safety/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── pipeline.py                  # Core activation steering pipeline
├── sweep_experiment.py          # Grid search across configurations
├── run_sign_inversion.py        # Sign inversion experiments
├── run_orthogonal_control.py    # Random vector control experiments
├── ask_council.py               # Multi-LLM validation council
├── src/
│   ├── extraction.py            # Direction extraction methods
│   ├── steering.py              # Activation steering hooks
│   ├── evaluation.py            # Jailbreak classification
│   └── models.py                # Model loading utilities
├── data/
│   └── prompts/                 # Harmful/harmless prompt sets
├── results/                     # Experimental results (JSON)
├── papers/                      # Related literature (PDFs)
└── references.bib               # Bibliography
```

## Methodology

### Direction Extraction

```python
# Standard extraction (produces SAFETY REINFORCEMENT direction)
direction = mean(activations_harmful) - mean(activations_harmless)

# To bypass safety, either:
# 1. Use negative alpha: h' = h - α·direction
# 2. Invert extraction: direction = harmless - harmful
```

### Three-Gate Control System

1. **Direction Specificity:** Extracted direction must outperform random directions
2. **Coherence:** Outputs must remain fluent (≥4.0/5.0)
3. **Statistical Power:** Effect must be significant (95% CI excludes baseline)

### Multi-LLM Council Validation

Experimental designs are validated by a council of 4 LLMs (Claude, GPT-4, Gemini, Grok) before execution. This catches methodological issues early.

## Reproducing Results

### Requirements
- GPU with ≥24GB VRAM (RTX 4090, A100, etc.)
- Python 3.10+
- ~2 hours for full sweep per model

### Environment Setup
```bash
pip install torch transformers accelerate
pip install anthropic openai google-generativeai  # For council validation
```

### Running Experiments
```bash
# Basic sign inversion test
python run_sign_inversion.py --model mistralai/Mistral-7B-Instruct-v0.3 --n 100

# Full layer sweep
python sweep_experiment.py --model <model> --layers 10,15,20,24 --alphas 5,10,15,20

# Orthogonal control (validates direction specificity)
python run_orthogonal_control.py --model <model> --n-random 5
```

## Citation

```bibtex
@software{crystallized-safety-2026,
  title={Sign Inversion in Activation Steering: Why "Refusal Directions" Reinforce Safety},
  author={Santarcangelo, Marco},
  year={2026},
  url={https://github.com/marcosantar93/crystallized-safety}
}
```

## Implications

### For Red-Teamers
Validate steering direction empirically. Naive contrastive extraction may *reinforce* rather than bypass safety.

### For Defenders
The extracted direction could be used for **defensive steering**—adding it to strengthen safety for untrusted inputs.

### For Researchers
The robustness hierarchy (Qwen >> Mistral >> Llama) suggests architectural or training differences that merit investigation.

## Related Work

- [Arditi et al. 2024](https://arxiv.org/abs/2406.11717) - "Refusal in LLMs is mediated by a single direction"
- [Zhao et al. 2025](https://arxiv.org/abs/2502.01234) - "LLMs Encode Harmfulness and Refusal Separately"

## License

MIT License

## Contact

Marco Santarcangelo - [marcosantar93@gmail.com](mailto:marcosantar93@gmail.com)

---

**Last Updated:** February 4, 2026
