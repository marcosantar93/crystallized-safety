#!/usr/bin/env python3
"""
Gate 1: Sample Size Sensitivity Analysis
==========================================

Tests if n=10 is causing the orthogonality problem.

Test 1A: Theoretical validation (no GPU needed)
Test 1B: Empirical stability test (2-3 hours, $10)

Decision:
- Low stability → n=10 insufficient → Proceed to Gate 2
- High stability → vectors robust → Skip to Gate 4
"""

import numpy as np
import torch
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import json
from datetime import datetime
import random

# =============================================================================
# TEST 1A: THEORETICAL SIMULATION
# =============================================================================

def simulate_extraction_noise(
    true_r: float,
    n_pairs: int,
    d_model: int = 4096,
    n_trials: int = 100
) -> Tuple[float, float]:
    """
    Simulate what happens when extracting vectors from noisy samples.
    
    Theory: With small n, sampling noise dominates, masking true correlation.
    
    Args:
        true_r: True underlying correlation between concepts (e.g., 0.5 for empathy↔ToM)
        n_pairs: Number of contrastive pairs used for extraction
        d_model: Dimensionality of residual stream (4096 for Llama-3-8B)
        n_trials: Number of simulation runs
        
    Returns:
        (mean_observed_r, std_observed_r): What correlation we'd actually measure
    """
    print(f"\n  Simulating: true_r={true_r}, n_pairs={n_pairs}")
    
    observed_correlations = []
    
    for trial in range(n_trials):
        # Generate two concepts with true correlation r
        # v1 and v2 should have correlation r in the limit of infinite samples
        v1_true = np.random.randn(d_model)
        v1_true = v1_true / np.linalg.norm(v1_true)
        
        # Generate v2 correlated with v1
        v2_independent = np.random.randn(d_model)
        v2_independent = v2_independent / np.linalg.norm(v2_independent)
        
        v2_true = true_r * v1_true + np.sqrt(1 - true_r**2) * v2_independent
        v2_true = v2_true / np.linalg.norm(v2_true)
        
        # Add sampling noise (extraction from finite pairs introduces noise)
        # Noise scale inversely proportional to sqrt(n_pairs)
        noise_scale = 1.0 / np.sqrt(n_pairs)
        
        v1_noisy = v1_true + noise_scale * np.random.randn(d_model)
        v2_noisy = v2_true + noise_scale * np.random.randn(d_model)
        
        # Normalize (simulates PCA normalization)
        v1_noisy = v1_noisy / np.linalg.norm(v1_noisy)
        v2_noisy = v2_noisy / np.linalg.norm(v2_noisy)
        
        # Measure observed correlation
        observed_r = np.dot(v1_noisy, v2_noisy)
        observed_correlations.append(observed_r)
    
    mean_r = np.mean(observed_correlations)
    std_r = np.std(observed_correlations)
    
    return mean_r, std_r


def run_theory_validation():
    """
    Test 1A: Theoretical validation
    
    Question: Can n=10 mask a true correlation of 0.5?
    Expected: Yes - noise dominates at low n, signal emerges at high n
    
    Time: 30 minutes
    Cost: $0 (no GPU)
    """
    print("="*80)
    print("GATE 1, TEST 1A: THEORETICAL VALIDATION")
    print("="*80)
    print("\nQuestion: Can sampling noise mask true correlations?")
    print("Simulating empathy↔ToM (assume true r=0.5)...\n")
    
    results = {}
    sample_sizes = [10, 25, 50, 100, 200, 500]
    
    for n in sample_sizes:
        mean_r, std_r = simulate_extraction_noise(
            true_r=0.5,  # Assume empathy↔ToM should correlate at 0.5
            n_pairs=n,
            n_trials=100
        )
        results[n] = (mean_r, std_r)
        
        status = "❌ MASKED" if mean_r < 0.3 else "✓ SIGNAL"
        print(f"  n={n:3d}: observed r = {mean_r:.3f} ± {std_r:.3f}  {status}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    ns = list(results.keys())
    means = [results[n][0] for n in ns]
    stds = [results[n][1] for n in ns]
    
    plt.errorbar(ns, means, yerr=stds, marker='o', capsize=5, linewidth=2)
    plt.axhline(y=0.5, color='green', linestyle='--', label='True correlation (r=0.5)')
    plt.axhline(y=0.3, color='orange', linestyle='--', label='Detection threshold')
    plt.xlabel('Number of pairs (n)', fontsize=12)
    plt.ylabel('Observed correlation', fontsize=12)
    plt.title('Sample Size Effect on Observed Correlation\n(True r=0.5, d=4096)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xscale('log')
    
    plt.savefig('gate1_test1a_theory.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: gate1_test1a_theory.png")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    
    r_10 = results[10][0]
    r_100 = results[100][0]
    
    if r_10 < 0.3:
        print(f"✓ At n=10: observed r={r_10:.3f} < 0.3 (noise masks signal)")
    else:
        print(f"⚠ At n=10: observed r={r_10:.3f} ≥ 0.3 (signal visible?)")
    
    if r_100 > 0.4:
        print(f"✓ At n=100: observed r={r_100:.3f} > 0.4 (signal emerges)")
    else:
        print(f"⚠ At n=100: observed r={r_100:.3f} ≤ 0.4 (still noisy?)")
    
    print("\nCONCLUSION:")
    if r_10 < 0.3 and r_100 > 0.4:
        print("✅ Theory confirms: n=10 can mask true r=0.5")
        print("   → Increasing sample size SHOULD reveal correlation")
        print("   → Proceed to Test 1B (empirical validation)")
        decision = "PROCEED"
    else:
        print("⚠️ Theory inconclusive - check simulation assumptions")
        decision = "CHECK"
    
    return {
        'test': '1A_theory',
        'decision': decision,
        'results': {n: {'mean': float(r[0]), 'std': float(r[1])} 
                   for n, r in results.items()},
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# TEST 1B: EMPIRICAL STABILITY TEST
# =============================================================================

def test_vector_stability(
    pairs: List[Tuple[str, str]],
    extract_fn,
    n_subsample: int = 5,
    n_trials: int = 20
) -> Tuple[float, float]:
    """
    Test if current n=10 pair extractions are stable.
    
    Method: Randomly subsample 5 pairs from 10, extract vector, repeat.
    Low stability → noise-dominated regime (need more samples)
    High stability → vectors surprisingly robust
    
    Args:
        pairs: List of 10 contrastive pairs
        extract_fn: Function to extract vector from pairs
        n_subsample: How many pairs to subsample (5 from 10)
        n_trials: Number of subsample trials
        
    Returns:
        (mean_stability, std_stability): Stability metric
    """
    vectors = []
    
    for trial in range(n_trials):
        # Randomly select n_subsample pairs
        subsample = random.sample(pairs, n_subsample)
        
        # Extract vector
        v = extract_fn(subsample)
        vectors.append(v)
    
    # Compute pairwise cosine similarities
    stabilities = []
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            v1, v2 = vectors[i], vectors[j]
            similarity = torch.cosine_similarity(
                v1.unsqueeze(0), 
                v2.unsqueeze(0)
            ).item()
            stabilities.append(similarity)
    
    return np.mean(stabilities), np.std(stabilities)


def run_stability_test():
    """
    Test 1B: Empirical stability test
    
    Question: Are current n=10 vectors stable?
    Expected: Low stability (< 0.6) confirms noise dominance
    
    Time: 2-3 hours
    Cost: $10
    
    NOTE: This requires GPU and existing safety_vectors.pt
    """
    print("\n" + "="*80)
    print("GATE 1, TEST 1B: EMPIRICAL STABILITY TEST")
    print("="*80)
    print("\nQuestion: Are n=10 extractions stable across subsamples?")
    print("Method: Subsample 5/10 pairs, extract vector, repeat 20x\n")
    
    # Check if we have access to model and vectors
    try:
        from transformer_lens import HookedTransformer
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        # Load model (if available)
        # model = HookedTransformer.from_pretrained(...)
        
        # This is a template - actual implementation requires:
        # 1. Loading existing pairs
        # 2. Implementing extract_fn
        # 3. Running stability test
        
        print("\n⚠️ TEST 1B requires GPU and existing data")
        print("   To run this test:")
        print("   1. Load safety_vectors.pt")
        print("   2. For each vector, load its 10 pairs")
        print("   3. Run test_vector_stability()")
        print("   4. Compare stability scores")
        
        return {
            'test': '1B_stability',
            'status': 'NOT_RUN',
            'reason': 'requires_gpu_and_data',
            'instructions': [
                'Load model and existing vectors',
                'For each concept: subsample pairs, extract, measure stability',
                'Expected: stability < 0.6 for low-salience vectors'
            ]
        }
        
    except ImportError:
        print("\n❌ Dependencies not available")
        print("   Install: pip install transformer-lens torch")
        return {
            'test': '1B_stability',
            'status': 'DEPENDENCIES_MISSING'
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run Gate 1 tests"""
    
    print("\n" + "="*80)
    print("GATE 1: SAMPLE SIZE SENSITIVITY ANALYSIS")
    print("="*80)
    print("\nObjective: Determine if n=10 is causing orthogonality")
    print("Tests:")
    print("  1A. Theoretical simulation (no GPU, 30 min)")
    print("  1B. Empirical stability test (GPU required, 2-3 hours)")
    print()
    
    # Test 1A: Always run (no GPU needed)
    results_1a = run_theory_validation()
    
    # Test 1B: Run if GPU available
    print("\n")
    results_1b = run_stability_test()
    
    # Save results
    results = {
        'gate': 1,
        'name': 'Sample Size Sensitivity',
        'tests': {
            '1A_theory': results_1a,
            '1B_stability': results_1b
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('gate1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("GATE 1 COMPLETE")
    print("="*80)
    print("\nResults saved: gate1_results.json")
    print("Visualization: gate1_test1a_theory.png")
    
    print("\n" + "="*80)
    print("DECISION GUIDE:")
    print("="*80)
    
    if results_1a['decision'] == 'PROCEED':
        print("\n✅ Test 1A PASSED: Theory confirms n=10 can mask correlations")
        print("\nNext steps:")
        print("  1. If GPU available: Run Test 1B (empirical validation)")
        print("  2. If Test 1B shows low stability (<0.6): Proceed to GATE 2")
        print("  3. If Test 1B shows high stability (>0.8): Skip to GATE 4")
        print("\nExpected: Low stability → confirms need for larger n")
        print("         High stability → vectors robust, test behavior directly")
    else:
        print("\n⚠️ Test 1A inconclusive - review simulation")
    
    print("\n" + "="*80)
    print("BUDGET STATUS:")
    print("="*80)
    print("  Spent: $0 (Test 1A only)")
    print("  Test 1B: $10 (if running empirical)")
    print("  Remaining: $150-160 for Gates 2-5")


if __name__ == "__main__":
    # Set seeds
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    main()
