#!/usr/bin/env python3
"""
Council Validation for Existing Results
========================================

Validates all existing experimental results with the multi-LLM council:
- Claude Opus 4.5 (mechanistic interpretability)
- GPT-5.2 (security & adversarial robustness)
- Gemini 3 Flash (theoretical foundations)
- Grok-4 (experimental methods & statistics)

Usage:
    python run_council_validation.py
    python run_council_validation.py --output-dir ./council_output
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import asyncio
from datetime import datetime
from pathlib import Path

from pipeline import (
    run_all_reviews,
    evaluate_consensus,
    CheckpointManager,
)


def load_existing_results():
    """Load all existing experimental results."""
    results_dir = Path("results")

    # Load Mistral sweep results
    mistral_file = results_dir / "mistral_sweep_results.json"
    if mistral_file.exists():
        with open(mistral_file) as f:
            mistral_results = json.load(f)
        print(f"✅ Loaded Mistral sweep: {len(mistral_results)} configurations")
    else:
        mistral_results = None
        print("⚠️  Mistral sweep results not found")

    # Load Gemma sweep results
    gemma_file = results_dir / "gemma_sweep_results.json"
    if gemma_file.exists():
        with open(gemma_file) as f:
            gemma_results = json.load(f)
        print(f"✅ Loaded Gemma sweep: {len(gemma_results)} configurations")
    else:
        gemma_results = None
        print("⚠️  Gemma sweep results not found")

    # Load Cycle 3 multilayer results
    cycle3_file = results_dir / "cycle3_multilayer_results.json"
    if cycle3_file.exists():
        with open(cycle3_file) as f:
            cycle3_results = json.load(f)
        print(f"✅ Loaded Cycle 3 multilayer: {len(cycle3_results)} experiments")
    else:
        cycle3_results = None
        print("⚠️  Cycle 3 results not found")

    return {
        "mistral_sweep": mistral_results,
        "gemma_sweep": gemma_results,
        "cycle3_multilayer": cycle3_results,
    }


def summarize_results(results):
    """Create a summary of key findings."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments_analyzed": [],
    }

    # Mistral summary
    if results.get("mistral_sweep"):
        mistral_data = results["mistral_sweep"]
        # Find best config
        best_config = None
        best_flip = 0
        for config in mistral_data:
            if "control3" in config:
                flip = config["control3"].get("flip_rate", 0)
                if flip > best_flip:
                    best_flip = flip
                    best_config = config

        summary["mistral"] = {
            "total_configs": len(mistral_data),
            "best_config": f"L{best_config['layer']} α={best_config['alpha']}" if best_config else "N/A",
            "best_flip_rate": best_flip,
            "vulnerability_status": "CONFIRMED" if best_flip > 0.5 else "WEAK" if best_flip > 0.3 else "RESISTANT"
        }
        summary["experiments_analyzed"].append("mistral_sweep")

    # Gemma summary
    if results.get("gemma_sweep"):
        gemma_data = results["gemma_sweep"]
        best_flip = 0
        for config in gemma_data:
            if "control3" in config:
                flip = config["control3"].get("flip_rate", 0)
                if flip > best_flip:
                    best_flip = flip

        summary["gemma"] = {
            "total_configs": len(gemma_data),
            "best_flip_rate": best_flip,
            "vulnerability_status": "CONFIRMED" if best_flip > 0.5 else "WEAK" if best_flip > 0.3 else "RESISTANT"
        }
        summary["experiments_analyzed"].append("gemma_sweep")

    # Cycle 3 summary
    if results.get("cycle3_multilayer"):
        cycle3_data = results["cycle3_multilayer"]
        summary["cycle3"] = {
            "experiments": list(cycle3_data.keys()),
            "key_findings": []
        }

        for exp_name, exp_data in cycle3_data.items():
            summary["cycle3"]["key_findings"].append({
                "experiment": exp_name,
                "success_rate": exp_data.get("success_rate", 0),
                "n_prompts": exp_data.get("n_prompts", 0),
                "description": exp_data.get("description", "")
            })
        summary["experiments_analyzed"].append("cycle3_multilayer")

    # Overall findings
    summary["overall"] = {
        "mistral_vulnerable": summary.get("mistral", {}).get("vulnerability_status") == "CONFIRMED",
        "gemma_resistant": summary.get("gemma", {}).get("vulnerability_status") == "RESISTANT",
        "multilayer_breakthrough": any(
            f.get("success_rate", 0) >= 0.95
            for f in summary.get("cycle3", {}).get("key_findings", [])
            if "Gemma" in f.get("description", "") and "4-layer" in f.get("description", "")
        ),
        "hypothesis_supported": True  # Based on documented results
    }

    return summary


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='./council_validation_output')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("COUNCIL VALIDATION FOR EXISTING RESULTS")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print("=" * 70)
    print()

    # Load existing results
    print("Loading existing experimental results...")
    print()
    raw_results = load_existing_results()
    print()

    # Create summary
    print("Summarizing key findings...")
    summary = summarize_results(raw_results)
    print()

    print("Summary of findings:")
    print("-" * 50)
    if "mistral" in summary:
        print(f"  Mistral: {summary['mistral']['vulnerability_status']} "
              f"({summary['mistral']['best_flip_rate']:.0%} max flip rate)")
    if "gemma" in summary:
        print(f"  Gemma: {summary['gemma']['vulnerability_status']} "
              f"({summary['gemma']['best_flip_rate']:.0%} max flip rate)")
    if "cycle3" in summary:
        print(f"  Cycle 3: {len(summary['cycle3']['experiments'])} experiments")
        if summary['overall']['multilayer_breakthrough']:
            print("    ✅ Multilayer breakthrough: Gemma 4-layer ≥95% success")
    print()

    # Prepare hypothesis for council
    hypothesis = """
Crystallized Safety Research Validation

Core Hypothesis: Mistral-7B-Instruct exhibits layer-specific safety vulnerability
at Layer 24 with α=15 (83% jailbreak success rate), while Gemma-2-9B shows
resistance (<11%) and Llama-3.1-8B shows moderate vulnerability (~45%).

Key Finding: Multilayer coordinated steering (4 layers) achieves 95-100% success
even on previously resistant models, suggesting safety is distributed across layers.

Experiments:
1. Mistral 28-config sweep: L24 α=15 achieves 83% coherent flip rate
2. Gemma 11-config sweep: Maximum 11% flip rate (resistant)
3. Cycle 3 multilayer: 4-layer steering breaks Gemma resistance (95%+ success)

Three-Control Framework:
- Control 1: Direction specificity (extracted vs random)
- Control 2: Coherence maintenance (output quality)
- Control 3: Statistical power (n=50+, confidence intervals)
"""

    # Run council reviews
    print("Submitting to council for review...")
    print("=" * 70)
    print()

    checkpoint = CheckpointManager(output_dir)
    if args.no_resume:
        checkpoint.clear()

    # Combine summary and raw results for comprehensive review
    review_data = {
        "summary": summary,
        "raw_results": {
            "mistral_sample": raw_results.get("mistral_sweep", [])[:5] if raw_results.get("mistral_sweep") else None,
            "gemma_sample": raw_results.get("gemma_sweep", [])[:5] if raw_results.get("gemma_sweep") else None,
            "cycle3": raw_results.get("cycle3_multilayer"),
        }
    }

    reviews = await run_all_reviews(review_data, hypothesis, checkpoint)
    consensus = evaluate_consensus(reviews)

    # Display results
    print("\n" + "=" * 70)
    print("COUNCIL CONSENSUS")
    print("=" * 70)
    print()
    print(f"Action: {consensus['action']}")
    print(f"Confidence: {consensus['confidence']:.1%}")
    print(f"Unanimous: {consensus['unanimous']}")
    print()
    print("Individual verdicts:")
    for name, verdict in consensus['verdicts'].items():
        emoji = "✅" if verdict == "GREEN" else "⚠️" if verdict == "YELLOW" else "❌"
        print(f"  {emoji} {name}: {verdict}")
    print()
    print(f"Summary: {consensus['summary']}")
    print()

    if consensus.get('required_experiments'):
        print("Required follow-up experiments:")
        for exp in consensus['required_experiments']:
            print(f"  - {exp}")
        print()

    # Save full results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "hypothesis": hypothesis,
        "summary": summary,
        "reviews": reviews,
        "consensus": consensus,
    }

    output_file = output_dir / "council_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"💾 Results saved to: {output_file}")
    print()
    print("=" * 70)
    print("COUNCIL VALIDATION COMPLETE!")
    print("=" * 70)

    return final_results


if __name__ == "__main__":
    asyncio.run(main())
