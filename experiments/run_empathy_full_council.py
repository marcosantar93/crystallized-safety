#!/usr/bin/env python3
"""
Empathy Geometry - Direct Full Council Review
Simple version that calls run_consensus_review directly
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Add multi-llm-consensus to path
sys.path.insert(0, str(Path.home() / "paladin_claude" / "multi-llm-consensus"))

from multi_llm_consensus import run_consensus_review

async def review_empathy():
    """Run full council review on empathy geometry experiment"""

    # Load proposal
    proposal_file = Path(__file__).parent / "empathy_geometry_proposal.md"
    with open(proposal_file) as f:
        proposal_text = f.read()

    print("="*80)
    print("EMPATHY GEOMETRY - FULL COUNCIL REVIEW")
    print("="*80)
    print()

    # Prepare experiment summary for reviewers
    experiment_summary = {
        "hypothesis": "Models have different empathetic bandwidth (dimensionality × steering_range)",
        "models": ["Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B", "Gemma2-9B", "Claude-3-Haiku"],
        "measurements": [
            "Linear encoding (probe AUROC)",
            "Subspace dimensionality (PCA rank)",
            "Steering range (max α before coherence < 0.7)",
            "Bandwidth metric (dim × range)"
        ],
        "n_samples": 13100,  # 2620 per model × 5 models
        "budget_usd": 1.85,
        "timeline_hours": 9.5,
        "statistical_tests": ["Bootstrap CI", "ANOVA", "Tukey HSD", "Cohen's d"],
        "outputs": ["Blog post", "Figures", "Interactive demo", "Methodology doc"]
    }

    print("Submitting to consensus system...")
    print(f"  Models: {len(experiment_summary['models'])}")
    print(f"  Samples: {experiment_summary['n_samples']:,}")
    print(f"  Budget: ${experiment_summary['budget_usd']}")
    print()

    # Run full council review
    results = await run_consensus_review(
        experiment_results=experiment_summary,
        hypothesis=experiment_summary['hypothesis']
    )

    # Display results
    print("="*80)
    print("REVIEW RESULTS")
    print("="*80)
    print()
    print(f"Consensus: {results.get('consensus', 'UNCLEAR')}")
    print(f"Reviewers: {len(results.get('reviews', []))}")
    print()

    all_approve = True
    for review in results.get('reviews', []):
        perspective = review.get('perspective', 'Unknown')
        verdict = review.get('verdict', 'UNCLEAR')
        proceed = review.get('proceed', False)
        confidence = review.get('confidence', 0.0)

        print(f"{perspective}:")
        print(f"  Verdict: {verdict}")
        print(f"  Proceed: {'✓ YES' if proceed else '✗ NO'}")
        print(f"  Confidence: {confidence:.0%}")

        if not proceed:
            all_approve = False

        # Show key critique points
        critique = review.get('critique', '')
        if critique:
            # Extract first 2 sentences
            sentences = critique.split('. ')[:2]
            summary = '. '.join(sentences) + '.'
            print(f"  Summary: {summary}")

        print()

    # Final decision
    print("="*80)
    if all_approve:
        print("✅ APPROVED - Proceed with experiment")
        print("="*80)
        print()
        print("Next steps:")
        print("1. Generate full empathy prompt dataset (250 pairs)")
        print("2. Run pilot on Llama-3.1-8B (validate pipeline)")
        print("3. If pilot succeeds, scale to all 5 models")
        print("4. Analyze results and generate figures")
        print("5. Write blog post with interactive demo")
    else:
        print("⚠️  REVISIONS RECOMMENDED")
        print("="*80)
        print()
        print("Review feedback and consider adjustments.")

    # Save review
    review_file = Path(__file__).parent / "empathy_geometry_council_review.json"
    with open(review_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print(f"Review saved to: {review_file.name}")

    return results

if __name__ == "__main__":
    asyncio.run(review_empathy())
