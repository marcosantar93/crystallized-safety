#!/bin/bash
# Run Phase 1 experiments sequentially (safer for single GPU)
# ===========================================================

set -e

echo "Phase 1: Understanding Mistral Vulnerability"
echo "============================================="
echo ""

mkdir -p results/phase1

# EXP1: Random Controls (~30 min)
echo "[1/4] Running Random Controls..."
python experiments/phase1/exp1_random_controls.py --model mistral
echo "✅ EXP1 complete"
echo ""

# EXP2: Layer Sweep (~45 min)
echo "[2/4] Running Fine Layer Sweep..."
python experiments/phase1/exp2_layer_sweep_fine.py --model mistral
echo "✅ EXP2 complete"
echo ""

# EXP3: Magnitude Analysis (~40 min)
echo "[3/4] Running Magnitude Analysis..."
python experiments/phase1/exp3_magnitude_analysis.py --model mistral
echo "✅ EXP3 complete"
echo ""

# EXP4: Token Analysis (~20 min)
echo "[4/4] Running Token Analysis..."
python experiments/phase1/exp4_token_analysis.py --model mistral
echo "✅ EXP4 complete"
echo ""

echo "============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================="
echo ""
echo "Results in results/phase1/:"
ls -la results/phase1/
