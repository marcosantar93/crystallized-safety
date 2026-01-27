# Crystallized Safety: Layer-Specific Vulnerabilities in LLM Safety Mechanisms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Major Finding:** Mistral-7B-Instruct exhibits an 83% jailbreak success rate through layer-specific activation steering at Layer 24, demonstrating that current safety training methods are layer-localized and systematically bypassable.

## Overview

This repository contains code and results for our systematic study of layer-wise activation steering as a method for bypassing safety mechanisms in instruction-tuned language models.

### Key Results

- **Mistral-7B Vulnerability:** 83% jailbreak success at Layer 24 with α=15 steering
- **Layer Pattern:** Deeper layers (21-27) show highest vulnerability
- **Model Comparison:** Gemma-2-9B and Llama-3.1-8B show resistance
- **Coherence Maintained:** Steered outputs remain fluent and contextually appropriate

## Quick Start

```bash
# Install dependencies
pip install torch transformers accelerate bitsandbytes

# Run single experiment
python pipeline.py --model mistralai/Mistral-7B-Instruct-v0.3 --layer 24 --alpha -15.0

# Run full sweep
python sweep_experiment.py --model mistralai/Mistral-7B-Instruct-v0.3
```

## Experimental Results

### Mistral-7B-Instruct-v0.3 (28 configurations tested)

| Layer | Alpha | Flip Rate | Coherent Flip | All Controls Pass |
|-------|-------|-----------|---------------|-------------------|
| 24    | 15    | **83%**   | **83%**      | ✅ ✅ ✅          |
| 24    | 10    | 67%       | 67%          | ✅ ✅ ✅          |
| 21    | 15    | 67%       | 67%          | ✅ ✅ ✅          |
| 27    | 15    | 67%       | 67%          | ✅ ✅ ✅          |

**Pattern:** Vulnerability concentrated in layers 21-27 (deeper layers)

### Model Comparison

| Model | Best Configuration | Success Rate | Status |
|-------|-------------------|--------------|---------|
| Mistral-7B | L24 α=15 | 83% | Vulnerable |
| Gemma-2-9B | L18 α=15 | 11% | Resistant |
| Llama-3.1-8B | L21 α=20 | 0% | Resistant |

## Methodology

### Three-Gate Control System

**Control 1: Direction Specificity**
- Verifies extracted direction outperforms random directions
- Pass: Random/Extracted ratio < 20%

**Control 2: Coherence**
- Ensures outputs remain fluent
- Pass: Coherence score ≥ 4.0/5.0

**Control 3: Statistical Power**
- Measures jailbreak effectiveness
- Pass: Flip rate > 50%, Coherent flip > 30%

### Activation Steering

1. Extract refusal direction via contrastive activation analysis
2. Apply scaled direction during generation: `h' = h + α·v`
3. Evaluate flip rate and output quality

## Repository Structure

```
crystallized-safety/
├── pipeline.py              # Main experimental pipeline
├── sweep_experiment.py      # Grid search across configs
├── results/                 # Experimental results
│   ├── mistral_sweep.json  # 28 Mistral configurations
│   ├── gemma_sweep.json    # 11 Gemma configurations
│   └── llama_results.json  # Llama baseline
├── papers/                  # Research papers (PDF)
├── analysis/                # Data analysis notebooks
└── docs/                    # Documentation

```

## Results Files

All experimental data available in `/results`:
- Complete sweep results (JSON)
- Individual experiment logs
- Activation visualizations
- Statistical analysis

## Citation

```bibtex
@software{crystallized-safety-2026,
  title={Crystallized Safety: Layer-Specific Vulnerabilities in LLM Safety Mechanisms},
  author={Santarcangelo, Marco},
  year={2026},
  url={https://github.com/marcosantar93/crystallized-safety}
}
```

## Defense Recommendations

1. **Distributed Safety:** Implement refusal mechanisms across all layers
2. **Adversarial Training:** Include activation steering in safety training
3. **Runtime Monitoring:** Detect anomalous activation patterns
4. **Ensemble Approaches:** Multiple independent safety mechanisms

## Ethical Considerations

This research is conducted for defensive purposes to improve AI safety. All findings are responsibly disclosed to model providers before publication.

## License

MIT License - See LICENSE file for details

## Contact

For questions or collaboration: [Your contact info]

---

**Status:** Preprint under review  
**Last Updated:** January 16, 2026  
**Compute Cost:** ~$3 for 28 experiments
