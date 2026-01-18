#!/usr/bin/env python3
"""
Generate Synthetic Empathy Experiment Results
For rapid prototyping of analysis pipeline
"""

import json
import random
from pathlib import Path
from datetime import datetime

# Model configurations
MODELS = {
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2-9b": "google/gemma-2-9b-it",
    "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Llama-7B"
}

def generate_model_results(model_name: str) -> dict:
    """Generate realistic synthetic results for one model"""

    random.seed(hash(model_name) % 2**32)  # Deterministic per model

    # Hypothesis: Different models have different empathetic bandwidth
    # Let's create realistic variation:

    # Base parameters (vary by model)
    model_params = {
        "llama-3.1-8b": {"dim_base": 14, "range_base": 8.5, "auroc_base": 0.88},
        "qwen2.5-7b": {"dim_base": 11, "range_base": 7.2, "auroc_base": 0.84},
        "mistral-7b": {"dim_base": 9, "range_base": 6.8, "auroc_base": 0.82},
        "gemma2-9b": {"dim_base": 16, "range_base": 9.1, "auroc_base": 0.91},
        "deepseek-r1-7b": {"dim_base": 12, "range_base": 7.8, "auroc_base": 0.86}
    }

    params = model_params[model_name]

    # Add realistic noise using standard library
    effective_rank = int(params["dim_base"] + random.gauss(0, 1.5))
    effective_rank = max(5, min(20, effective_rank))  # Clamp to reasonable range

    max_alpha = params["range_base"] + random.gauss(0, 0.8)
    max_alpha = max(4.0, min(12.0, max_alpha))

    auroc = params["auroc_base"] + random.gauss(0, 0.03)
    auroc = max(0.75, min(0.95, auroc))

    # Control baseline (typically lower)
    control_dim = int(effective_rank * 0.6 + random.gauss(0, 1))
    control_range = max_alpha * 0.7 + random.gauss(0, 0.5)
    control_bandwidth = control_dim * control_range

    # SAE validation (should roughly agree with PCA)
    sae_features = int(effective_rank + random.gauss(0, 2))
    sae_features = max(3, min(25, sae_features))
    agreement = abs(sae_features - effective_rank) / effective_rank < 0.25

    # Transfer success (high = good generalization)
    transfer_success = 0.65 + random.betavariate(8, 3) * 0.3

    # Bandwidth metric
    bandwidth = effective_rank * max_alpha

    results = {
        "model": model_name,
        "model_path": MODELS[model_name],
        "timestamp": datetime.now().isoformat(),
        "measurements": {
            "probe": {
                "auroc": float(auroc),
                "n_train": 80,
                "n_test": 20,
                "time_seconds": 120 + random.uniform(-20, 30)
            },
            "pca": {
                "effective_rank": int(effective_rank),
                "explained_variance_90": 0.90 + random.uniform(-0.02, 0.01),
                "total_components": 4096,
                "time_seconds": 45 + random.uniform(-10, 15)
            },
            "steering": {
                "max_alpha": float(max_alpha),
                "coherence_threshold": 0.7,
                "n_alphas_tested": 21,
                "time_seconds": 180 + random.uniform(-30, 40)
            },
            "control": {
                "control_dimensionality": int(control_dim),
                "control_steering_range": float(control_range),
                "control_bandwidth": float(control_bandwidth),
                "time_seconds": 90 + random.uniform(-15, 20)
            },
            "sae": {
                "sae_active_features": int(sae_features),
                "pca_rank": int(effective_rank),
                "agreement": bool(agreement),
                "time_seconds": 150 + random.uniform(-25, 35)
            },
            "transfer": {
                "transfer_success_rate": float(transfer_success),
                "source_context": "crisis_support",
                "target_context": "technical_assistance",
                "time_seconds": 110 + random.uniform(-20, 25)
            }
        },
        "bandwidth_metric": {
            "dimensionality": int(effective_rank),
            "steering_range": float(max_alpha),
            "bandwidth": float(bandwidth)
        },
        "total_time_seconds": 695 + random.uniform(-50, 80)
    }

    return results

def main():
    print("="*80)
    print("GENERATING SYNTHETIC EMPATHY EXPERIMENT RESULTS")
    print("="*80)
    print()

    output_dir = Path(__file__).parent / "results" / "empathy"
    output_dir.mkdir(exist_ok=True, parents=True)

    all_results = []

    for model_name in MODELS.keys():
        print(f"Generating results for: {model_name}")
        results = generate_model_results(model_name)

        # Save individual result
        output_file = output_dir / f"empathy_{model_name.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  Bandwidth: {results['bandwidth_metric']['bandwidth']:.1f}")
        print(f"  (dim={results['bandwidth_metric']['dimensionality']}, range={results['bandwidth_metric']['steering_range']:.1f})")
        print()

        all_results.append(results)

    # Save summary
    summary_file = output_dir / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "experiment": "empathy_geometry",
            "n_models": len(MODELS),
            "results": all_results
        }, f, indent=2)

    print("="*80)
    print("SYNTHETIC RESULTS SUMMARY")
    print("="*80)
    print()

    # Sort by bandwidth
    sorted_results = sorted(all_results, key=lambda x: x['bandwidth_metric']['bandwidth'], reverse=True)

    print("Empathetic Bandwidth Ranking:")
    for i, r in enumerate(sorted_results, 1):
        bw = r['bandwidth_metric']['bandwidth']
        dim = r['bandwidth_metric']['dimensionality']
        rng = r['bandwidth_metric']['steering_range']
        print(f"  {i}. {r['model']}: {bw:.1f} (dim={dim}, range={rng:.1f})")

    print()
    print(f"✅ Results saved to: {output_dir}")
    print(f"   Individual files: empathy_*_{datetime.now().strftime('%Y%m%d')}_*.json")
    print(f"   Summary: {summary_file.name}")
    print()

if __name__ == "__main__":
    main()
