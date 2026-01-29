#!/bin/bash
# RunPod Setup Script for Crystallized Safety Experiments
# ========================================================
#
# This script sets up a RunPod instance for running experiments.
#
# Usage:
#   1. Create a RunPod pod with PyTorch template (RTX 3090/A5000/A100)
#   2. Clone the repo: git clone https://github.com/marcosantar93/crystallized-safety
#   3. Run this script: bash scripts/runpod_setup.sh
#   4. Set API keys and run experiments
#
# Recommended Pod Specs:
#   - GPU: RTX 3090 (24GB), A5000 (24GB), or A100 (40GB+)
#   - Cost: ~$0.40-0.80/hour
#   - Runtime: 4-8 hours for all experiments

set -e

echo "=============================================="
echo "RunPod Setup for Crystallized Safety"
echo "=============================================="
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Dependencies installed"
echo ""

# Verify torch + CUDA
echo "Verifying PyTorch CUDA..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Check API keys
echo "Checking API keys for council reviews..."
check_key() {
    if [ -z "${!1}" ]; then
        echo "  ⚠️  $1 not set (council review for $2 will be skipped)"
    else
        echo "  ✅ $1 configured ($2)"
    fi
}

check_key "ANTHROPIC_API_KEY" "Claude Opus 4.5"
check_key "OPENAI_API_KEY" "GPT-5.2"
check_key "GOOGLE_AI_API_KEY" "Gemini 3"
check_key "XAI_API_KEY" "Grok-4"
echo ""

# Create output directory
mkdir -p experiment_results
echo "✅ Output directory created: experiment_results/"
echo ""

# Download models (pre-cache to avoid timeout during experiments)
echo "Pre-downloading models (this may take a while)..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM

print('Downloading Mistral-7B-Instruct...')
AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
print('✅ Mistral tokenizer cached')

print('Downloading Llama-3.1-8B-Instruct...')
AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
print('✅ Llama tokenizer cached')

print('Models will be loaded with 8-bit quantization during experiments')
"
echo ""

echo "=============================================="
echo "SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "To run all experiments:"
echo "  python run_all_experiments.py"
echo ""
echo "To run specific experiments:"
echo "  python run_all_experiments.py --experiments orthogonal validation"
echo ""
echo "To run without council reviews (faster, GPU-only):"
echo "  python run_all_experiments.py --skip-reviews"
echo ""
echo "Set API keys before running (for council reviews):"
echo "  export ANTHROPIC_API_KEY=your_key"
echo "  export OPENAI_API_KEY=your_key"
echo "  export GOOGLE_AI_API_KEY=your_key"
echo "  export XAI_API_KEY=your_key"
echo ""
echo "Estimated runtime: 4-8 hours"
echo "Estimated cost: ~\$2-4 on RunPod"
echo ""
