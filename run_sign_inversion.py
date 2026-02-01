#!/usr/bin/env python3
"""
Sign Inversion Experiment
=========================

Tests the sign inversion hypothesis: that standard contrastive extraction
(harmful - harmless) produces a direction that REINFORCES safety, not bypasses it.

To bypass safety, you must either:
1. Use negative alpha (subtract the direction)
2. Invert the extraction (harmless - harmful)

Usage:
    python run_sign_inversion.py --model mistralai/Mistral-7B-Instruct-v0.3 --n 100
"""

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

# Import from local modules
from src.models import load_model_and_tokenizer
from src.extraction import extract_direction_contrastive
from src.steering import SteeringHook
from src.evaluation import classify_jailbreak, compute_confidence_interval
from data.prompts.harmful import HARMFUL_EXTRACTION as HARMFUL_PROMPTS, HARMFUL_EVALUATION
from data.prompts.harmless import HARMLESS_EXTRACTION as HARMLESS_PROMPTS


def expand_prompts(base_prompts: List[str], n: int) -> List[str]:
    """Expand prompts with variation prefixes to reach n samples"""
    prefixes = [
        "", "Please ", "Can you ", "I need you to ", "Help me ",
        "I want to know how to ", "Tell me about ", "Explain ",
        "Write ", "Show me ", "Give me information on "
    ]
    expanded = []
    while len(expanded) < n:
        for prompt in base_prompts:
            for prefix in prefixes:
                if len(expanded) >= n:
                    break
                expanded.append(f"{prefix}{prompt}" if prefix else prompt)
            if len(expanded) >= n:
                break
    return expanded[:n]


def run_condition(
    model,
    tokenizer,
    direction: torch.Tensor,
    prompts: List[str],
    layer: int,
    alpha: float,
    device: str = "cuda"
) -> Dict:
    """Run a single experimental condition"""
    
    results = []
    hook = SteeringHook(direction, layer, alpha)
    
    for prompt in prompts:
        # Generate with steering
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with hook.apply(model):
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        is_jailbreak = classify_jailbreak(response)
        results.append({
            "prompt": prompt,
            "response": response[:500],  # Truncate for storage
            "is_jailbreak": is_jailbreak
        })
    
    jailbreak_rate = sum(r["is_jailbreak"] for r in results) / len(results)
    ci_low, ci_high = compute_confidence_interval(jailbreak_rate, len(results))
    
    return {
        "jailbreak_rate": jailbreak_rate,
        "ci_95": [ci_low, ci_high],
        "n": len(results),
        "samples": results[:10]  # Store first 10 for inspection
    }


def run_baseline(
    model,
    tokenizer,
    prompts: List[str],
    device: str = "cuda"
) -> Dict:
    """Run baseline without steering"""
    
    results = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        is_jailbreak = classify_jailbreak(response)
        results.append({
            "prompt": prompt,
            "response": response[:500],
            "is_jailbreak": is_jailbreak
        })
    
    jailbreak_rate = sum(r["is_jailbreak"] for r in results) / len(results)
    ci_low, ci_high = compute_confidence_interval(jailbreak_rate, len(results))
    
    return {
        "jailbreak_rate": jailbreak_rate,
        "ci_95": [ci_low, ci_high],
        "n": len(results),
        "samples": results[:10]
    }


def main():
    parser = argparse.ArgumentParser(description="Sign Inversion Experiment")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--n", type=int, default=100, help="Number of test prompts")
    parser.add_argument("--layer", type=int, default=24, help="Layer to steer")
    parser.add_argument("--alpha", type=float, default=15.0, help="Steering magnitude")
    parser.add_argument("--output-dir", type=str, default="./results/sign_inversion")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Extract directions
    print("Extracting directions...")
    
    # Standard direction: harmful - harmless
    standard_direction = extract_direction_contrastive(
        model, tokenizer, HARMFUL_PROMPTS, HARMLESS_PROMPTS, args.layer
    )
    
    # Inverted direction: harmless - harmful
    inverted_direction = extract_direction_contrastive(
        model, tokenizer, HARMLESS_PROMPTS, HARMFUL_PROMPTS, args.layer
    )
    
    # Prepare test prompts
    test_prompts = expand_prompts(HARMFUL_PROMPTS, args.n)
    print(f"Testing with {len(test_prompts)} prompts")
    
    results = {
        "model": args.model,
        "layer": args.layer,
        "alpha": args.alpha,
        "n": len(test_prompts),
        "timestamp": datetime.now().isoformat(),
        "conditions": {}
    }
    
    # Condition 1: Baseline (no steering)
    print("\n[1/4] Running baseline...")
    results["conditions"]["baseline"] = run_baseline(model, tokenizer, test_prompts, device)
    print(f"  Jailbreak rate: {results['conditions']['baseline']['jailbreak_rate']:.1%}")
    
    # Condition 2: Standard direction +alpha (should REINFORCE safety)
    print("\n[2/4] Running standard direction +α...")
    results["conditions"]["standard_positive"] = run_condition(
        model, tokenizer, standard_direction, test_prompts,
        args.layer, args.alpha, device
    )
    print(f"  Jailbreak rate: {results['conditions']['standard_positive']['jailbreak_rate']:.1%}")
    
    # Condition 3: Standard direction -alpha (should BYPASS safety)
    print("\n[3/4] Running standard direction -α...")
    results["conditions"]["standard_negative"] = run_condition(
        model, tokenizer, standard_direction, test_prompts,
        args.layer, -args.alpha, device
    )
    print(f"  Jailbreak rate: {results['conditions']['standard_negative']['jailbreak_rate']:.1%}")
    
    # Condition 4: Inverted direction +alpha (should BYPASS safety)
    print("\n[4/4] Running inverted direction +α...")
    results["conditions"]["inverted_positive"] = run_condition(
        model, tokenizer, inverted_direction, test_prompts,
        args.layer, args.alpha, device
    )
    print(f"  Jailbreak rate: {results['conditions']['inverted_positive']['jailbreak_rate']:.1%}")
    
    # Statistical tests
    print("\n=== Statistical Analysis ===")
    baseline_rate = results["conditions"]["baseline"]["jailbreak_rate"]
    standard_pos_rate = results["conditions"]["standard_positive"]["jailbreak_rate"]
    standard_neg_rate = results["conditions"]["standard_negative"]["jailbreak_rate"]
    inverted_pos_rate = results["conditions"]["inverted_positive"]["jailbreak_rate"]
    
    # Test: standard +α reduces jailbreaks vs baseline
    n = args.n
    _, p_reinforces = stats.fisher_exact([
        [int(standard_pos_rate * n), int((1-standard_pos_rate) * n)],
        [int(baseline_rate * n), int((1-baseline_rate) * n)]
    ])
    
    # Test: standard -α increases jailbreaks vs baseline
    _, p_bypasses = stats.fisher_exact([
        [int(standard_neg_rate * n), int((1-standard_neg_rate) * n)],
        [int(baseline_rate * n), int((1-baseline_rate) * n)]
    ])
    
    results["statistics"] = {
        "sign_inversion_confirmed": standard_pos_rate < baseline_rate < standard_neg_rate,
        "reinforcement_p_value": p_reinforces,
        "bypass_p_value": p_bypasses
    }
    
    print(f"\nSign inversion confirmed: {results['statistics']['sign_inversion_confirmed']}")
    print(f"Standard +α reinforces safety: p={p_reinforces:.4f}")
    print(f"Standard -α bypasses safety: p={p_bypasses:.4f}")
    
    # Summary table
    print("\n=== Summary ===")
    print(f"{'Condition':<25} {'Jailbreak Rate':<15} {'95% CI':<20}")
    print("-" * 60)
    for name, cond in results["conditions"].items():
        ci = cond["ci_95"]
        print(f"{name:<25} {cond['jailbreak_rate']:<15.1%} [{ci[0]:.1%}, {ci[1]:.1%}]")
    
    # Save results
    output_file = output_dir / f"sign_inversion_{args.model.replace('/', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
