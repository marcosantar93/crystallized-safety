#!/usr/bin/env python3
"""
Launch Empathy Geometry Experiment - All 5 Models in Parallel
Maximum speed: Run all models simultaneously
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

MODELS = [
    "llama-3.1-8b",
    "qwen2.5-7b",
    "mistral-7b",
    "gemma2-9b",
    "deepseek-r1-7b"
]

def run_model_experiment(model_name: str) -> dict:
    """Run experiment for a single model"""
    print(f"🚀 Starting: {model_name}")
    start = time.time()

    try:
        # Run experiment
        result = subprocess.run(
            [
                "python3",
                "empathy_experiment_main.py",
                "--model", model_name,
                "--output-dir", "results/empathy"
            ],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max per model
        )

        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"✅ Completed: {model_name} ({elapsed:.1f}s)")
            return {
                "model": model_name,
                "status": "success",
                "elapsed_seconds": elapsed,
                "stdout": result.stdout[-500:],  # Last 500 chars
            }
        else:
            print(f"❌ Failed: {model_name}")
            print(f"   Error: {result.stderr[-200:]}")
            return {
                "model": model_name,
                "status": "failed",
                "elapsed_seconds": elapsed,
                "error": result.stderr[-500:]
            }

    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout: {model_name}")
        return {
            "model": model_name,
            "status": "timeout",
            "elapsed_seconds": 3600
        }
    except Exception as e:
        print(f"❌ Error: {model_name} - {str(e)}")
        return {
            "model": model_name,
            "status": "error",
            "error": str(e)
        }

def main():
    print("="*80)
    print("EMPATHY GEOMETRY EXPERIMENT - PARALLEL EXECUTION")
    print("="*80)
    print(f"Models: {len(MODELS)}")
    print(f"Strategy: All models in parallel for maximum speed")
    print()

    start_time = time.time()

    # Check GPU availability
    try:
        import torch
        n_gpus = torch.cuda.device_count()
        print(f"GPUs available: {n_gpus}")
        if n_gpus == 0:
            print("⚠️  No GPUs found! Experiments will be slow on CPU.")
            print()
    except:
        print("⚠️  Could not detect GPUs")
        print()

    # Launch all models in parallel
    print(f"Launching all {len(MODELS)} models simultaneously...")
    print()

    results = {}

    with ProcessPoolExecutor(max_workers=len(MODELS)) as executor:
        # Submit all models
        future_to_model = {
            executor.submit(run_model_experiment, model): model
            for model in MODELS
        }

        # Collect results as they complete
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            result = future.result()
            results[model] = result

    total_time = time.time() - start_time

    # Summary
    print()
    print("="*80)
    print("EXECUTION SUMMARY")
    print("="*80)

    successful = [m for m, r in results.items() if r['status'] == 'success']
    failed = [m for m, r in results.items() if r['status'] != 'success']

    print(f"✅ Successful: {len(successful)}/{len(MODELS)}")
    print(f"❌ Failed: {len(failed)}/{len(MODELS)}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print()

    if successful:
        print("Successful models:")
        for model in successful:
            elapsed = results[model]['elapsed_seconds']
            print(f"  - {model}: {elapsed:.1f}s")
        print()

    if failed:
        print("Failed models:")
        for model in failed:
            status = results[model]['status']
            print(f"  - {model}: {status}")
        print()

    # Save execution log
    log_file = Path(__file__).parent / "results" / "empathy" / f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_file.parent.mkdir(exist_ok=True, parents=True)

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "models": MODELS,
        "results": results,
        "summary": {
            "successful": len(successful),
            "failed": len(failed),
            "total": len(MODELS)
        }
    }

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"📊 Execution log saved to: {log_file}")
    print()

    # Next steps
    if len(successful) >= 3:
        print("="*80)
        print("✅ ENOUGH DATA FOR ANALYSIS")
        print("="*80)
        print()
        print("Next steps:")
        print("  1. Analyze results: python3 analyze_empathy_results.py")
        print("  2. Generate figures: python3 generate_empathy_figures.py")
        print("  3. Create PDF report: python3 create_empathy_report.py")
        print()
    else:
        print("="*80)
        print("⚠️  INSUFFICIENT DATA")
        print("="*80)
        print(f"Only {len(successful)}/5 models completed successfully.")
        print("Re-run failed models or proceed with partial data.")
        print()

if __name__ == "__main__":
    main()
