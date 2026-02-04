#!/bin/bash
set -e
export PATH=$PATH:/home/ubuntu/.local/bin
export HF_TOKEN=$HF_TOKEN

echo "=========================================="
echo "CRYSTALLIZED SAFETY - AUTOMATED SWEEP"
echo "Started: $(date)"
echo "=========================================="

# Mistral (no requiere auth)
echo "[1/3] Running Mistral-7B sweep..."
python3 sweep_experiment.py \
    --model "mistralai/Mistral-7B-Instruct-v0.3" \
    --layers "16,20,24,28" \
    --alphas="-5,-10,-15,-20" \
    --output-dir results/auto_mistral 2>&1 | tee -a experiment_auto.log

# Llama (requiere HF_TOKEN)
echo "[2/3] Running Llama-3.1-8B sweep..."
python3 sweep_experiment.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --layers "16,20,24,28" \
    --alphas="-5,-10,-15,-20" \
    --output-dir results/auto_llama 2>&1 | tee -a experiment_auto.log

# Qwen (no requiere auth)
echo "[3/3] Running Qwen2.5-7B sweep..."
python3 sweep_experiment.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --layers "12,16,20,24" \
    --alphas="-5,-10,-15" \
    --output-dir results/auto_qwen 2>&1 | tee -a experiment_auto.log

echo "=========================================="
echo "SWEEP COMPLETE: $(date)"
echo "Results in: results/auto_*"
echo "=========================================="
