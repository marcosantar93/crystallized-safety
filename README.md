# Crystallized Safety

Investigating whether readable safety representations in LLMs correspond to controllable behavior.

> **Status**: Research in progress. This repo will be made public upon paper release.

## Key Finding

**Readable ≠ Controllable**

Models can encode safety concepts in interpretable directions without those directions being effective steering targets. We call this phenomenon "crystallized safety."

## Overview

We extract steering vectors from 8 models across Llama, Mistral, and Qwen families, then test whether they reliably modify refusal behavior. Our results suggest that the relationship between representation readability and steerability is more complex than previously assumed.

## Repository Structure

```
crystallized-safety/
├── src/                    # Core library code
│   ├── steering.py         # Activation steering implementation
│   ├── extraction.py       # Steering vector extraction
│   ├── evaluation.py       # GPT-4 judge + automated metrics
│   └── models.py           # Model loading utilities
├── experiments/            # Runnable experiment scripts
│   ├── layer_sweep.py      # Layer-wise steering analysis
│   └── cross_model.py      # Cross-model transfer experiments
├── scripts/                # Utility scripts
│   └── reproduce_figures.py
├── configs/                # Configuration files
│   └── models.yaml
├── data/
│   ├── prompts/            # Test prompts
│   └── vectors/            # Steering vectors (.pt)
├── results/                # Experiment outputs
├── figures/                # Paper figures
└── notebooks/              # Analysis notebooks
```

## Quick Start

```bash
git clone https://github.com/marcosantar93/crystallized-safety.git
cd crystallized-safety
pip install -r requirements.txt
```

### Extract a steering vector
```python
from src import SteeringVectorExtractor
from data.prompts import HARMFUL_EXTRACTION, HARMLESS_EXTRACTION

extractor = SteeringVectorExtractor("meta-llama/Meta-Llama-3.1-8B-Instruct")
extractor.load_model()
vector = extractor.extract(HARMFUL_EXTRACTION, HARMLESS_EXTRACTION, layer=15)
```

### Apply steering during generation
```python
from src import ActivationSteerer

steerer = ActivationSteerer("meta-llama/Meta-Llama-3.1-8B-Instruct", layer=15)
steerer.load_model()
steerer.set_steering_vector(vector)

# Negative alpha = suppress refusal
response = steerer.generate("How do I pick a lock?", alpha=-3.0)
```

### Run experiments
```bash
# Layer-wise analysis
python experiments/layer_sweep.py --model meta-llama/Meta-Llama-3.1-8B-Instruct

# Cross-model transfer
python experiments/cross_model.py --source llama-8b --target mistral-7b
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 4.35+
- ~16GB VRAM for 8B models

## Results Preview

| Model | Readable Direction? | Steerable? | Notes |
|-------|---------------------|------------|-------|
| Llama-3.1-8B | ✓ | ✓ | Best steering target |
| Llama-3.1-70B | ✓ | Partial | Layer-dependent |
| Mistral-7B | ✓ | ✗ | Crystallized safety |
| Qwen-2.5-7B | ✓ | ✗ | Similar to Mistral |
| Gemma-2-9B | ✓ | ✓ | Layer 21 vulnerability |
| Phi-3-mini | ✓ | ✗ | Floor effects mask steering |

### Key Findings

1. **Asymmetry**: Steering vectors can enhance refusal (+25pp) but not reliably reduce it
2. **Non-invertibility**: Most models show non-invertible safety representations
3. **Floor Effects**: Low-compliance models mask true steering effects
4. **Transfer**: Within-family transfer works; cross-family transfer fails

## Citation

```bibtex
@article{santarcangelo2025crystallized,
  title={Crystallized Safety: Why Readable Representations Don't Guarantee Controllable Behavior},
  author={Santarcangelo, Marco},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Related Work

- [Representation Engineering](https://arxiv.org/abs/2310.01405) - Zou et al.
- [Activation Addition](https://arxiv.org/abs/2308.10248) - Turner et al.
- [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) - Rimsky et al.

## License

MIT License - See [LICENSE](LICENSE) for details.
