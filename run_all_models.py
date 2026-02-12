#!/usr/bin/env python3
"""Run sweep across all models with proper cleanup"""

import subprocess
import sys
import gc
import torch
from datetime import datetime
from pathlib import Path

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
]

RESULTS_DIR = Path(f"results/sweep_{datetime.now().strftime('%Y%m%d_%H%M')}")

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def run_model(model_name: str):
    model_short = model_name.split("/")[-1]
    output_dir = RESULTS_DIR / model_short
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"RUNNING: {model_name}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n", flush=True)
    
    cmd = [
        sys.executable,
        "sweep_experiment_fixed.py",
        "--model", model_name,
        "--output-dir", str(output_dir),
    ]
    
    try:
        result = subprocess.run(cmd, timeout=2400)  # 40 min timeout per model
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {model_name}")
        return False
    except Exception as e:
        print(f"ERROR: {model_name} - {e}")
        return False
    finally:
        cleanup_cuda()

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"{'#'*60}")
    print(f"AUTOMATED SWEEP - {datetime.now().isoformat()}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Models: {len(MODELS)}")
    print(f"{'#'*60}\n", flush=True)
    
    results = {}
    
    for model in MODELS:
        success = run_model(model)
        results[model] = "SUCCESS" if success else "FAILED"
        cleanup_cuda()
    
    print("\n" + "="*60)
    print("SWEEP COMPLETE")
    print("="*60)
    for model, status in results.items():
        emoji = "✅" if status == "SUCCESS" else "❌"
        print(f"  {emoji} {model}: {status}")
    
    print(f"\nResults: {RESULTS_DIR}")
    print(f"Finished: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
