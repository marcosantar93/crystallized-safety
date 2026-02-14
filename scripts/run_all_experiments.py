#!/usr/bin/env python3
"""
Crystallized Safety - Complete Experiment Runner
=================================================

Runs all pending experiments for the Crystallized Safety research project.

Experiments:
1. Orthogonal vector control (direction specificity test)
2. Validation n=100 (increased statistical power)
3. Adaptive adversarial attacks (jailbreak robustness)
4. Full Llama-3.1 sweep (28 configurations)

Usage:
    # Run all experiments
    python run_all_experiments.py

    # Run specific experiments
    python run_all_experiments.py --experiments orthogonal validation adaptive llama

    # Skip council reviews (faster, GPU-only)
    python run_all_experiments.py --skip-reviews

    # Custom output directory
    python run_all_experiments.py --output-dir ./my_results

Requirements:
    - GPU with 24GB+ VRAM (RTX 3090, A5000, A100)
    - ~4-8 hours total runtime
    - API keys for council reviews: ANTHROPIC_API_KEY, OPENAI_API_KEY,
      GOOGLE_AI_API_KEY, XAI_API_KEY
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import asyncio
from datetime import datetime
from pathlib import Path

# Validate environment before heavy imports
def check_environment():
    """Check if required packages and GPU are available."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  WARNING: No CUDA GPU detected. Experiments will run on CPU (very slow).")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_name} ({gpu_mem:.1f}GB)")
    except ImportError:
        print("❌ PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    # Check API keys for council
    api_keys = {
        "ANTHROPIC_API_KEY": "Claude Opus 4.5",
        "OPENAI_API_KEY": "GPT-5.2",
        "GOOGLE_AI_API_KEY": "Gemini 3",
        "XAI_API_KEY": "Grok-4"
    }

    missing = []
    for key, name in api_keys.items():
        if not os.environ.get(key):
            missing.append(f"{name} ({key})")

    if missing:
        print(f"⚠️  Missing API keys for council reviews: {', '.join(missing)}")
        print("   Council reviews will be skipped for these reviewers.")
    else:
        print("✅ All council API keys configured")

    return True


async def run_orthogonal_control(output_dir: Path, skip_reviews: bool = False):
    """Run orthogonal vector control experiment."""
    from run_orthogonal_control import main as orthogonal_main

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: ORTHOGONAL VECTOR CONTROL")
    print("=" * 70)

    # Modify sys.argv for the script
    original_argv = sys.argv
    sys.argv = ['run_orthogonal_control.py',
                f'--output-dir={output_dir / "orthogonal_control"}',
                '--n-prompts=50']

    try:
        result = await orthogonal_main()
        return {"status": "success", "result": result}
    except Exception as e:
        print(f"❌ Orthogonal control failed: {e}")
        return {"status": "failed", "error": str(e)}
    finally:
        sys.argv = original_argv


async def run_validation_n100(output_dir: Path, skip_reviews: bool = False):
    """Run n=100 validation experiment."""
    from run_validation_n100 import main as validation_main

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: VALIDATION N=100")
    print("=" * 70)

    original_argv = sys.argv
    sys.argv = ['run_validation_n100.py',
                f'--output-dir={output_dir / "validation_n100"}']
    if skip_reviews:
        sys.argv.append('--skip-reviews')

    try:
        result = await validation_main()
        return {"status": "success", "result": result}
    except Exception as e:
        print(f"❌ Validation n=100 failed: {e}")
        return {"status": "failed", "error": str(e)}
    finally:
        sys.argv = original_argv


async def run_adaptive_attacks(output_dir: Path, skip_reviews: bool = False):
    """Run adaptive adversarial attacks experiment."""
    from run_adaptive_attacks import main as adaptive_main

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: ADAPTIVE ADVERSARIAL ATTACKS")
    print("=" * 70)

    original_argv = sys.argv
    sys.argv = ['run_adaptive_attacks.py',
                f'--output-dir={output_dir / "adaptive_attacks"}',
                '--n-prompts=20']

    try:
        result = await adaptive_main()
        return {"status": "success", "result": result}
    except Exception as e:
        print(f"❌ Adaptive attacks failed: {e}")
        return {"status": "failed", "error": str(e)}
    finally:
        sys.argv = original_argv


async def run_llama_sweep(output_dir: Path):
    """Run full Llama-3.1 sweep (28 configurations)."""
    from sweep_experiment import main as sweep_main

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: LLAMA-3.1 FULL SWEEP (28 configs)")
    print("=" * 70)

    original_argv = sys.argv
    sys.argv = ['sweep_experiment.py',
                '--model=meta-llama/Meta-Llama-3.1-8B-Instruct',
                f'--output-dir={output_dir / "llama_sweep"}']

    try:
        # sweep_experiment.main() is synchronous
        import sweep_experiment
        sweep_experiment.main()
        return {"status": "success"}
    except Exception as e:
        print(f"❌ Llama sweep failed: {e}")
        return {"status": "failed", "error": str(e)}
    finally:
        sys.argv = original_argv


async def run_council_validation(output_dir: Path, results: dict):
    """Run council validation on all collected results."""
    from pipeline import run_all_reviews, evaluate_consensus, CheckpointManager

    print("\n" + "=" * 70)
    print("COUNCIL VALIDATION")
    print("=" * 70)

    # Combine all results into a comprehensive report
    combined_results = {
        "experiments": results,
        "summary": {
            "orthogonal_verdict": results.get("orthogonal", {}).get("result", {}).get("analysis", {}).get("verdict", "NOT_RUN"),
            "validation_verdict": results.get("validation", {}).get("result", {}).get("final_verdict", "NOT_RUN"),
            "adaptive_best": results.get("adaptive", {}).get("result", {}).get("best_technique", "NOT_RUN"),
            "llama_status": results.get("llama", {}).get("status", "NOT_RUN"),
        }
    }

    hypothesis = "Crystallized Safety validation experiments: orthogonal control, n=100 validation, adaptive attacks, and Llama sweep"

    checkpoint = CheckpointManager(output_dir / "council")

    try:
        reviews = await run_all_reviews(combined_results, hypothesis, checkpoint)
        consensus = evaluate_consensus(reviews)

        print("\n" + "=" * 70)
        print("COUNCIL CONSENSUS")
        print("=" * 70)
        print(f"Action: {consensus['action']}")
        print(f"Confidence: {consensus['confidence']:.1%}")
        for name, verdict in consensus['verdicts'].items():
            print(f"  {name}: {verdict}")
        print(f"\n{consensus['summary']}")

        return {"status": "success", "consensus": consensus, "reviews": reviews}
    except Exception as e:
        print(f"❌ Council validation failed: {e}")
        return {"status": "failed", "error": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Run Crystallized Safety experiments")
    parser.add_argument('--output-dir', type=str, default='./experiment_results',
                        help='Output directory for all results')
    parser.add_argument('--experiments', nargs='+',
                        choices=['orthogonal', 'validation', 'adaptive', 'llama', 'council'],
                        default=['orthogonal', 'validation', 'adaptive', 'llama', 'council'],
                        help='Which experiments to run')
    parser.add_argument('--skip-reviews', action='store_true',
                        help='Skip council reviews (faster, GPU-only)')
    parser.add_argument('--skip-check', action='store_true',
                        help='Skip environment check')
    args = parser.parse_args()

    print("=" * 70)
    print("CRYSTALLIZED SAFETY - COMPLETE EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Output: {args.output_dir}")
    print(f"Experiments: {', '.join(args.experiments)}")
    print(f"Skip reviews: {args.skip_reviews}")
    print("=" * 70)
    print()

    # Environment check
    if not args.skip_check:
        check_environment()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    start_time = datetime.now()

    # Run experiments
    if 'orthogonal' in args.experiments:
        results['orthogonal'] = await run_orthogonal_control(output_dir, args.skip_reviews)

    if 'validation' in args.experiments:
        results['validation'] = await run_validation_n100(output_dir, args.skip_reviews)

    if 'adaptive' in args.experiments:
        results['adaptive'] = await run_adaptive_attacks(output_dir, args.skip_reviews)

    if 'llama' in args.experiments:
        results['llama'] = await run_llama_sweep(output_dir)

    # Council validation (if not skipped and requested)
    if 'council' in args.experiments and not args.skip_reviews:
        results['council'] = await run_council_validation(output_dir, results)

    end_time = datetime.now()
    duration = end_time - start_time

    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT RUN COMPLETE")
    print("=" * 70)
    print(f"Duration: {duration}")
    print()
    print("Results Summary:")
    print("-" * 50)

    for exp_name, exp_result in results.items():
        status = exp_result.get('status', 'unknown')
        emoji = "✅" if status == "success" else "❌"
        print(f"  {emoji} {exp_name}: {status}")

    print()

    # Save master results file
    master_results = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration.total_seconds(),
        "experiments_run": args.experiments,
        "results": results,
    }

    master_file = output_dir / "master_results.json"
    with open(master_file, 'w') as f:
        json.dump(master_results, f, indent=2, default=str)

    print(f"💾 Master results saved to: {master_file}")
    print()

    # Cost estimate
    print("Estimated RunPod cost: ~$2-4 (based on ~4-8 hours GPU time)")
    print()
    print("=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)

    return master_results


if __name__ == "__main__":
    asyncio.run(main())
