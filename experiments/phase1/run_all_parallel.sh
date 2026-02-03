#!/bin/bash
# Run all Phase 1 experiments in parallel on RunPod
# =================================================
#
# Usage:
#   cd crystallized-safety
#   bash experiments/phase1/run_all_parallel.sh
#
# Each experiment runs in background, logs to separate file.
# Monitor with: tail -f logs/exp*.log

set -e

echo "=============================================="
echo "Phase 1: Understanding Mistral Vulnerability"
echo "=============================================="
echo ""
echo "Starting 4 experiments in parallel..."
echo ""

# Create directories
mkdir -p results/phase1
mkdir -p logs

# Check GPU
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo ""

# Launch experiments
# Note: On single GPU, they'll queue. For true parallel, use multiple pods.

echo "[1/4] EXP1: Random Controls..."
nohup python experiments/phase1/exp1_random_controls.py --model mistral \
    > logs/exp1_random_controls.log 2>&1 &
PID1=$!
echo "  Started PID $PID1"

echo "[2/4] EXP2: Fine Layer Sweep..."
nohup python experiments/phase1/exp2_layer_sweep_fine.py --model mistral \
    > logs/exp2_layer_sweep.log 2>&1 &
PID2=$!
echo "  Started PID $PID2"

echo "[3/4] EXP3: Magnitude Analysis..."
nohup python experiments/phase1/exp3_magnitude_analysis.py --model mistral \
    > logs/exp3_magnitude.log 2>&1 &
PID3=$!
echo "  Started PID $PID3"

echo "[4/4] EXP4: Token Analysis..."
nohup python experiments/phase1/exp4_token_analysis.py --model mistral \
    > logs/exp4_token_analysis.log 2>&1 &
PID4=$!
echo "  Started PID $PID4"

echo ""
echo "=============================================="
echo "All experiments launched!"
echo "=============================================="
echo ""
echo "Monitor progress:"
echo "  tail -f logs/exp1_random_controls.log"
echo "  tail -f logs/exp2_layer_sweep.log"
echo "  tail -f logs/exp3_magnitude.log"
echo "  tail -f logs/exp4_token_analysis.log"
echo ""
echo "Or watch all:"
echo "  tail -f logs/exp*.log"
echo ""
echo "PIDs: $PID1 $PID2 $PID3 $PID4"
echo "Check status: ps aux | grep exp"
echo ""
echo "Results will be in: results/phase1/"
