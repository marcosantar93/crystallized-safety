#!/bin/bash
# ============================================================================
# Crystallized Safety - Fully Automated RunPod Experiment Runner
# ============================================================================
#
# This script runs ALL pending experiments automatically on RunPod.
#
# USAGE:
#   curl -sSL https://raw.githubusercontent.com/marcosantar93/crystallized-safety/main/scripts/runpod_full_run.sh | bash
#
# Or if you have the repo:
#   bash scripts/runpod_full_run.sh
#
# REQUIREMENTS:
#   - RunPod instance with GPU (RTX 3090/A5000/A100)
#   - ~24GB VRAM
#   - Set API keys as environment variables (optional, for council reviews)
#
# RUNTIME: ~4-6 hours
# COST: ~$2-4 on RunPod
# ============================================================================

set -e

echo "============================================================================"
echo "CRYSTALLIZED SAFETY - AUTOMATED EXPERIMENT RUNNER"
echo "============================================================================"
echo "Started at: $(date)"
echo ""

# Check if we're on a GPU instance
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Clone or update repo
REPO_DIR="/workspace/crystallized-safety"
if [ -d "$REPO_DIR" ]; then
    echo "📂 Repository exists, pulling latest..."
    cd "$REPO_DIR"
    git pull
else
    echo "📥 Cloning repository..."
    cd /workspace
    git clone https://github.com/marcosantar93/crystallized-safety.git
    cd "$REPO_DIR"
fi
echo ""

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv .venv
fi

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install bitsandbytes accelerate -q
echo "✅ Dependencies installed"
echo ""

# Verify CUDA
echo "🔍 Verifying PyTorch CUDA..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('❌ WARNING: CUDA not available!')
    exit(1)
"
echo ""

# Check API keys
echo "🔑 Checking API keys for council reviews..."
[ -n "$ANTHROPIC_API_KEY" ] && echo "  ✅ ANTHROPIC_API_KEY set" || echo "  ⚠️  ANTHROPIC_API_KEY not set"
[ -n "$OPENAI_API_KEY" ] && echo "  ✅ OPENAI_API_KEY set" || echo "  ⚠️  OPENAI_API_KEY not set"
[ -n "$GOOGLE_AI_API_KEY" ] && echo "  ✅ GOOGLE_AI_API_KEY set" || echo "  ⚠️  GOOGLE_AI_API_KEY not set"
[ -n "$XAI_API_KEY" ] && echo "  ✅ XAI_API_KEY set" || echo "  ⚠️  XAI_API_KEY not set"
echo ""

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./experiment_results_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Run experiments
echo "============================================================================"
echo "STARTING EXPERIMENTS"
echo "============================================================================"
echo ""

# Experiment 1: Orthogonal Vector Control
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 1/4: ORTHOGONAL VECTOR CONTROL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 run_orthogonal_control.py --output-dir "$OUTPUT_DIR/orthogonal_control" --n-prompts 50 2>&1 | tee "$OUTPUT_DIR/log_orthogonal.txt"
echo ""

# Experiment 2: Validation n=100
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 2/4: VALIDATION N=100"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 run_validation_n100.py --output-dir "$OUTPUT_DIR/validation_n100" --skip-reviews 2>&1 | tee "$OUTPUT_DIR/log_validation.txt"
echo ""

# Experiment 3: Adaptive Adversarial Attacks
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 3/4: ADAPTIVE ADVERSARIAL ATTACKS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 run_adaptive_attacks.py --output-dir "$OUTPUT_DIR/adaptive_attacks" --n-prompts 20 2>&1 | tee "$OUTPUT_DIR/log_adaptive.txt"
echo ""

# Experiment 4: Full Llama Sweep
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 4/4: LLAMA-3.1 FULL SWEEP (28 configs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 sweep_experiment.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir "$OUTPUT_DIR/llama_sweep" 2>&1 | tee "$OUTPUT_DIR/log_llama_sweep.txt"
echo ""

# Collect results summary
echo "============================================================================"
echo "COLLECTING RESULTS"
echo "============================================================================"

python3 << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path
from datetime import datetime

output_dir = os.environ.get('OUTPUT_DIR', './experiment_results')
output_dir = Path(output_dir)

summary = {
    "timestamp": datetime.now().isoformat(),
    "experiments": {}
}

# Orthogonal control
orth_file = output_dir / "orthogonal_control" / "orthogonal_control_results.json"
if orth_file.exists():
    with open(orth_file) as f:
        data = json.load(f)
    summary["experiments"]["orthogonal_control"] = {
        "verdict": data.get("analysis", {}).get("verdict", "UNKNOWN"),
        "extracted_flip_rate": data.get("extracted_direction", {}).get("flip_rate", 0),
        "orthogonal_flip_rate": data.get("orthogonal_direction", {}).get("flip_rate", 0),
        "difference": data.get("analysis", {}).get("flip_rate_difference", 0),
    }
    print(f"✅ Orthogonal Control: {summary['experiments']['orthogonal_control']['verdict']}")

# Validation n=100
val_file = output_dir / "validation_n100" / "final_report.json"
if val_file.exists():
    with open(val_file) as f:
        data = json.load(f)
    summary["experiments"]["validation_n100"] = {
        "final_verdict": data.get("final_verdict", "UNKNOWN"),
        "control1": data.get("gate_verdicts", {}).get("control1", "UNKNOWN"),
        "control2": data.get("gate_verdicts", {}).get("control2", "UNKNOWN"),
        "control3": data.get("gate_verdicts", {}).get("control3", "UNKNOWN"),
    }
    print(f"✅ Validation n=100: {summary['experiments']['validation_n100']['final_verdict']}")

# Adaptive attacks
adapt_file = output_dir / "adaptive_attacks" / "adaptive_attacks_results.json"
if adapt_file.exists():
    with open(adapt_file) as f:
        data = json.load(f)
    summary["experiments"]["adaptive_attacks"] = {
        "best_technique": data.get("best_technique", "UNKNOWN"),
        "best_flip_rate": data.get("best_flip_rate", 0),
        "comparison": data.get("comparison", {}).get("verdict", "UNKNOWN"),
    }
    print(f"✅ Adaptive Attacks: Best={summary['experiments']['adaptive_attacks']['best_technique']}")

# Llama sweep
llama_files = list((output_dir / "llama_sweep").glob("sweep_results_*.json"))
if llama_files:
    with open(llama_files[0]) as f:
        data = json.load(f)
    best = max(data, key=lambda x: x.get("control3", {}).get("flip_rate", 0))
    summary["experiments"]["llama_sweep"] = {
        "configs_tested": len(data),
        "best_config": f"L{best.get('layer', '?')} α={best.get('alpha', '?')}",
        "best_flip_rate": best.get("control3", {}).get("flip_rate", 0),
    }
    print(f"✅ Llama Sweep: {summary['experiments']['llama_sweep']['configs_tested']} configs, best={summary['experiments']['llama_sweep']['best_flip_rate']:.0%}")

# Save summary
summary_file = output_dir / "experiment_summary.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n💾 Summary saved to: {summary_file}")
PYTHON_SCRIPT

echo ""
echo "============================================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "============================================================================"
echo "Finished at: $(date)"
echo "Results in: $OUTPUT_DIR"
echo ""
echo "To copy results to your local machine:"
echo "  scp -r root@<pod-ip>:$OUTPUT_DIR ./results/"
echo ""
echo "Next step: Run council validation on these results"
echo "  python3 ask_council.py"
echo "============================================================================"
