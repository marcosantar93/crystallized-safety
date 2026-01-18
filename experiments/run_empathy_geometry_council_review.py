#!/usr/bin/env python3
"""
Empathy Geometry Experiment - Council Review Request
Submit proposal to multi-LLM consensus system before execution
"""

import asyncio
import sys
import os
from pathlib import Path

# Add multi-llm-consensus to path
sys.path.insert(0, str(Path.home() / "paladin_claude" / "multi-llm-consensus"))

from multi_llm_consensus import run_review_cycle_adaptive, run_consensus_review

async def review_empathy_geometry():
    """Submit empathy geometry experiment for council review"""

    # Experiment configuration
    experiment_config = {
        'models': ['Llama-3.1-8B', 'Qwen2.5-7B', 'Mistral-7B', 'Gemma2-9B', 'Claude-3-Haiku'],
        'n_conditions': 5,  # 5 context categories
        'type': 'novel',    # New research direction
        'n_statistical_tests': 8,  # ANOVA + post-hoc tests
        'cost_usd': 1.85    # $1.55 compute + $0.30 council
    }

    # Load proposal
    proposal_file = Path(__file__).parent / "empathy_geometry_proposal.md"
    with open(proposal_file) as f:
        proposal_text = f.read()

    print("="*80)
    print("EMPATHY GEOMETRY EXPERIMENT - COUNCIL REVIEW")
    print("="*80)
    print()
    print("Submitting to multi-LLM consensus system...")
    print()

    # Run adaptive review
    results = await run_review_cycle_adaptive(
        experiment_config,
        proposal_text,
        run_review_func=run_consensus_review,
        api_key=os.environ.get('ANTHROPIC_API_KEY')
    )

    # Display results
    print("="*80)
    print("REVIEW RESULTS")
    print("="*80)
    print()
    print(f"Tier: {results['tier']}")
    print(f"Reviewers: {', '.join(results['reviewers_used'])}")
    print(f"Complexity: {results['complexity']:.2f}")
    print(f"Doubt: {results['doubt']:.2f}")
    print(f"Review time: {results.get('review_time', 0):.1f}s")
    print()

    # Individual reviews
    all_approve = True
    for review in results['results']:
        perspective = review.get('perspective', 'Unknown')
        verdict = review.get('verdict', 'UNCLEAR')
        proceed = review.get('proceed', False)
        confidence = review.get('confidence', 0.0)

        print(f"{perspective}:")
        print(f"  Verdict: {verdict}")
        print(f"  Proceed: {'✓' if proceed else '✗'}")
        print(f"  Confidence: {confidence:.2f}")

        if not proceed:
            all_approve = False
            print(f"  Concerns:")
            critique = review.get('critique', 'No details provided')
            # Print first 300 chars of critique
            print(f"    {critique[:300]}...")

        print()

    # Decision
    print("="*80)
    if all_approve:
        print("✅ APPROVED - Proceed with experiment")
        print("="*80)
        print()
        print("Next steps:")
        print("1. Run pilot on single model (Llama-3.1-8B)")
        print("2. If successful, scale to all 5 models")
        print("3. Analyze results and generate figures")
        print("4. Write blog post")
    else:
        print("⚠️  REVISIONS NEEDED")
        print("="*80)
        print()
        print("Address reviewer concerns above before proceeding.")
        print("Consider:")
        print("- Adjusting sample sizes")
        print("- Clarifying definitions (empathy scoring)")
        print("- Adding controls or validation steps")

    # Save review
    review_file = Path(__file__).parent / f"empathy_geometry_council_review_{results['tier']}.json"
    import json
    with open(review_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print(f"Review saved to: {review_file}")

    return results

if __name__ == "__main__":
    # Check API keys
    required_keys = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY']
    missing = [k for k in required_keys if not os.environ.get(k)]

    if missing:
        print("⚠️  Missing API keys:")
        for key in missing:
            print(f"  - {key}")
        print()
        print("Load from api_keys.txt:")
        api_keys_file = Path.home() / "paladin_claude" / "research_automation" / "api_keys.txt"
        if api_keys_file.exists():
            print(f"  source {api_keys_file}")
        sys.exit(1)

    # Run review
    asyncio.run(review_empathy_geometry())
