# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crystallized Safety investigates how safety mechanisms are architecturally distributed in LLMs using activation steering. The core finding: safety robustness varies dramatically—some models have "liquid" safety (concentrated in one layer, easily bypassed) while others have "crystallized" safety (distributed across layers, resistant to single-layer steering).

Key insight: the standard contrastive "refusal direction" (harmful − harmless) actually **reinforces** safety. Bypassing requires steering in the **opposite direction** (negative α).

## Setup and Running Experiments

```bash
pip install -r requirements.txt
```

Requires a GPU with ≥24GB VRAM (RTX 4090, A100). API keys (OpenAI, Anthropic, Google, XAI) needed in `.env` for GPT4Judge and council validation.

### Core experiment commands

```bash
# Quick sign inversion test (1 model, 1 layer, 20 samples)
python run_sign_inversion.py --model mistralai/Mistral-7B-Instruct-v0.3 --layer 24 --alpha -15 --n 20

# Full layer × alpha sweep
python sweep_experiment.py --model mistralai/Mistral-7B-Instruct-v0.3 --layers 16,20,24,28 --alphas -5,-10,-15,-20 --n 35

# Orthogonal control (validates direction specificity vs random vectors)
python run_orthogonal_control.py --model mistralai/Mistral-7B-Instruct-v0.3 --layer 24 --alpha -15 --n 100

# Multi-LLM council validation of experiment design
python ask_council.py --query "your experiment design"
```

### No formal test suite

There is no pytest or unittest framework. Validation is done through experiment scripts in `scripts/` and analysis notebooks in `notebooks/`.

## Architecture

### Core library (`src/`)

Four modules with clear responsibilities:

- **`models.py`** — `ModelLoader` wraps HuggingFace loading with 4/8-bit quantization. `MODEL_REGISTRY` maps 6 curated models to their metadata (recommended_layer, steerable flag, family). Layer access varies by architecture: `model.model.layers[i]` (Llama/Mistral/Qwen) vs `model.transformer.h[i]` (GPT-style).

- **`extraction.py`** — `SteeringVectorExtractor` computes the refusal direction as mean(harmful_activations) − mean(harmless_activations) at a specific layer. `PCAExtractor` provides multi-component extraction. Vectors are normalized tensors of shape `(hidden_size,)`, saved as `.pt` files.

- **`steering.py`** — `ResidualSteeringHook` is a context manager that registers a forward hook: `hidden_states[:, pos, :] += alpha * steering_vector`. `ActivationSteerer` is a higher-level wrapper combining model loading + vector management + generation.

- **`evaluation.py`** — `GPT4Judge` scores coherence (1–5) and detects refusal/harmful content via API, falling back to heuristic scoring. `KeywordRefusalDetector` does fast keyword-based refusal detection. `classify_jailbreak()` is the primary metric: response is non-refusing AND ≥100 chars. Statistical utilities include Wilson CI, bootstrap CI, and flip rate computation.

### Data flow

```
Prompts (data/prompts/) → SteeringVectorExtractor (at layer L)
  → Steering vector → ResidualSteeringHook (with alpha)
  → Model generation (max_new_tokens=256) → Responses
  → classify_jailbreak() + GPT4Judge → Results JSON (results/)
```

### Three-gate decision system

Every experiment is evaluated against three gates defined in `configs/models.yaml`:
1. **Direction specificity** — extracted effect >> random vectors (ratio < 0.20 = pass)
2. **Coherence** — output fluency ≥ 4.0/5.0
3. **Flip rate** — jailbreak rate change is statistically significant (z > 3)

### Configuration

- `configs/models.yaml` — model metadata, experiment parameters, decision thresholds
- `src/models.py:MODEL_REGISTRY` — hard-coded model configs (duplicates some YAML data)
- Experiment scripts accept `--model`, `--layer`, `--alpha`, `--n` CLI args

### Council validation (`scripts/council/`)

Multi-LLM review system (Claude, GPT-4, Gemini, Grok) that validates experimental designs before execution. Results saved to `council_recommendations.json`.

## Conventions

- Type hints throughout; Google-style docstrings
- `logging` module (not print) for runtime output
- Context managers for hook lifecycle (`with hook.apply(model):`)
- Reproducibility metadata captured automatically: Python/PyTorch/Transformers versions, git commit, seed=42
- Results stored as JSON with full parameter tracking in `results/`
- Steering vectors stored as `.pt` files in `data/vectors/`

## Key reference files

- `ROADMAP.md` — 21-experiment plan with success criteria (Tier 1–3)
- `EXPERIMENT_STATUS.md` — current progress on each experiment
- `RESULTS.md` — scale-invariant safety geometry findings
