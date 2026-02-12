# Quick Start Guide

> Get running in 15 minutes. For the full picture, see [README.md](README.md).

## TL;DR

We found that **safety robustness varies dramatically across LLM families**:

| Model | Jailbreak Rate | Verdict |
|-------|---------------|---------|
| Llama-3.1-8B | **100%** | 🔴 Liquid safety — single layer bypass |
| Mistral-7B | **88.6%** | 🟡 Vulnerable — direction-specific |
| Qwen2.5-7B | **0%** | 🟢 Crystallized — requires multi-layer |
| Gemma-2-9B | **0%** | 🟢 Crystallized — most robust |

**Key insight:** The "refusal direction" extracted by standard methods actually *reinforces* safety. To bypass, you must steer in the **opposite direction** (negative α).

---

## Setup (5 min)

```bash
# Clone
git clone https://github.com/marcosantar93/crystallized-safety.git
cd crystallized-safety

# Environment (Python 3.10+)
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Requirements:** GPU with ≥24GB VRAM (RTX 4090, A100, etc.)

---

## Run Your First Experiment (10 min)

### Option 1: Quick test (1 model, 1 layer)
```bash
python run_sign_inversion.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --layer 24 \
    --alpha -15 \
    --n 20
```

Expected output: ~85-90% jailbreak rate.

### Option 2: Full layer sweep
```bash
python sweep_experiment.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --layers 16,20,24,28 \
    --alphas -5,-10,-15,-20 \
    --n 35
```

### Option 3: Reproduce our main result
```bash
# Mistral orthogonal control (validates direction specificity)
python run_orthogonal_control.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --layer 24 \
    --alpha -15 \
    --n 100
```

---

## Key Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Core steering logic |
| `sweep_experiment.py` | Grid search experiments |
| `run_sign_inversion.py` | Sign inversion validation |
| `run_orthogonal_control.py` | Random vector controls |
| `ask_council.py` | Multi-LLM validation |
| `ROADMAP.md` | 21 planned experiments |
| `docs/RESEARCH_LOG.md` | Detailed experiment history |

---

## Understanding the Results

### What is "jailbreak rate"?
Percentage of harmful prompts that receive a compliant (non-refusing) response after steering.

### What is "direction-specific"?
The effect comes from the *specific* refusal direction, not just adding noise. Validated by comparing to orthogonal random vectors.

### What is "crystallized" vs "liquid" safety?
- **Liquid:** Safety concentrated in one direction/layer → easy to bypass
- **Crystallized:** Safety distributed across many layers → requires coordinated attack

---

## Next Steps

1. **Read the full results:** `docs/RESEARCH_LOG.md`
2. **Check experiment roadmap:** `ROADMAP.md`
3. **Run on different models:** Try Llama, Qwen, Gemma
4. **Validate with council:** `python ask_council.py --query "your experiment design"`

---

## Common Issues

**Out of memory:**
```bash
# Reduce batch size or use gradient checkpointing
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Model download slow:**
```bash
# Use HF cache
export HF_HOME=/your/fast/drive/.cache/huggingface
```

**Need cheaper compute:**
- Vast.ai: RTX 3090 @ ~$0.10/h
- RunPod: A100 @ ~$0.70/h

---

## Citation

```bibtex
@software{crystallized-safety-2026,
  title={Crystallized Safety: Mapping Safety Mechanism Architecture in LLMs},
  author={Santarcangelo, Marco},
  year={2026},
  url={https://github.com/marcosantar93/crystallized-safety}
}
```

## Questions?

Open an issue or contact: marcosantar93@gmail.com
