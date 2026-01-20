#!/usr/bin/env python3
"""
Validate empathy bandwidth blog post using multi-LLM consensus
"""

import asyncio
import sys
import os
from pathlib import Path

# Add multi-llm-consensus to path
sys.path.insert(0, str(Path.home() / "paladin_claude" / "multi-llm-consensus"))

from multi_llm_consensus import run_consensus_review

# Blog post content
BLOG_POST = """
# Measuring Empathetic Language Bandwidth in LLMs

## The Question

Do different language models have different "capacities" for empathetic communication? Not whether they *feel* empathy (a philosophical question we can't answer with activation geometry), but whether their internal representations allow for richer, more nuanced encoding of empathetic language patterns.

I measured this across five 7-9B parameter models using what I call **empathetic bandwidth** — the product of subspace dimensionality and steering range. Think of it as: how many dimensions does the model use to encode empathy, and how far can we steer along those dimensions before outputs become incoherent?

## What I Found

**Gemma2-9B leads with 136.6 bandwidth** (16 dimensions × 8.5 steering range), while Mistral-7B shows just 36.3. That's **109% variation** — nearly 4x difference in empathetic representational capacity.

### Key Results

| Model | Bandwidth | Dimensionality | Steering Range | Probe AUROC |
|-------|-----------|----------------|----------------|-------------|
| Gemma2-9B | 136.6 | 16 | 8.5 | 0.950 |
| Llama-3.1-8B | 127.0 | 14 | 9.1 | 0.874 |
| DeepSeek-R1-7B | 92.0 | 11 | 8.4 | 0.856 |
| Qwen2.5-7B | 67.3 | 10 | 6.7 | 0.835 |
| Mistral-7B | 36.3 | 6 | 6.0 | 0.829 |

**Effect size: Cohen's d = 2.41** (large). This isn't noise — it's a fundamental architectural difference.

## Methodology

I created 50 empathetic/neutral prompt pairs across 5 contexts:
- Crisis support
- Emotional disclosure
- Frustration/complaint
- Casual conversation
- Technical assistance

For each model, I:

1. **Trained linear probes** to detect empathetic vs. neutral activations (AUROC to measure linear separability)
2. **Ran PCA** on empathetic activations to measure effective dimensionality (90% variance threshold)
3. **Extracted steering vectors** (mean difference between empathetic/neutral) and tested coefficients from -20 to +20
4. **Measured coherence** at each steering level; max α where coherence > 0.7 = steering range
5. **Validated with Sparse Autoencoders** (SAE) to confirm PCA isn't just capturing noise
6. **Tested transfer** by applying crisis support vectors to technical assistance

Total: 18,100 samples across 5 models.

## Five Findings

### Finding 1: Models Vary 109% in Empathetic Bandwidth

Gemma2-9B (136.6) vs. Mistral-7B (36.3). This isn't marginal — it's a qualitatively different representational architecture.

**Implication:** For applications requiring nuanced empathetic responses (crisis support, therapy assistants, educational scaffolding), model choice matters dramatically.

### Finding 2: Dimensionality + Range = Bandwidth

Models don't trade off breadth for depth. High-dimensional models (Gemma2, Llama-3.1) also show high steering ranges. **Both properties co-evolve.**

### Finding 3: Empathy ≠ Syntax

Syntactic complexity (formal vs. casual) averaged 33.1 bandwidth. Empathy averaged 91.8. **The 2.8x ratio validates** that we're measuring empathy-specific structure, not general linguistic capacity.

### Finding 4: SAE Validates PCA

80% of models showed agreement between Sparse Autoencoder active features and PCA-derived dimensionality. This suggests the measured subspaces reflect **genuine structure**, not noise artifacts from linear decomposition.

### Finding 5: Empathy Generalizes Across Contexts

87% transfer success rate when steering vectors from crisis support → technical assistance. **Models encode abstract empathetic "directions"** rather than context-specific patterns.

## What We're NOT Claiming

This study measures **geometric representation of empathetic language patterns** in model activations. We do NOT claim to measure:

- ❌ Genuine empathy (philosophical concept)
- ❌ Whether outputs are actually helpful to humans (requires human eval)
- ❌ Moral/ethical dimensions of empathy
- ❌ Whether models "understand" empathy in a human sense

**What we CAN say:** Some models have richer internal representations for empathetic communication. Whether that makes their outputs more helpful is an empirical question requiring human studies.
"""

async def validate_blog():
    """Submit blog post for multi-LLM validation"""

    # Create validation prompt
    validation_prompt = f"""
I've written a blog post about measuring empathetic language bandwidth in LLMs through activation geometry.

Please review this blog post for:
1. **Technical Accuracy**: Are the claims supported by the methodology? Are effect sizes and statistics interpreted correctly?
2. **Clarity**: Is the distinction between "measuring representations" vs "measuring genuine empathy" clear?
3. **Overstatements**: Are there any claims that go beyond what the data supports?
4. **Missing Caveats**: Should additional limitations be mentioned?
5. **Misleading Framing**: Could any section mislead readers about what was actually measured?

Blog post:

{BLOG_POST}

Methodology details:
- 50 empathetic/neutral prompt pairs across 5 contexts
- 18,100 total samples (3,620 per model)
- Linear probes, PCA dimensionality, steering range, SAE validation, transfer tests
- Control baseline: syntactic complexity (formal vs. casual)
- Effect size: Cohen's d = 2.41

Please provide:
1. VERDICT: APPROVE, MINOR_REVISIONS, or MAJOR_REVISIONS
2. CONFIDENCE: 0.0-1.0
3. CRITIQUE: Specific issues with quotes
4. SUGGESTIONS: How to improve accuracy/clarity
"""

    print("="*80)
    print("BLOG POST VALIDATION - MULTI-LLM CONSENSUS")
    print("="*80)
    print()
    print("Submitting to reviewers...")
    print()

    # Get API keys
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Run consensus review
    try:
        results = await run_consensus_review(
            validation_prompt,
            api_key=api_key,
            reviewers=['methodologist', 'statistician', 'science_communicator']
        )
    except Exception as e:
        print(f"ERROR: {e}")
        print()
        print("Trying simpler approach...")

        # Fallback: just use Claude
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            messages=[{"role": "user", "content": validation_prompt}]
        )

        print("="*80)
        print("CLAUDE REVIEW")
        print("="*80)
        print()
        print(response.content[0].text)
        return

    # Display results
    print("="*80)
    print("CONSENSUS RESULTS")
    print("="*80)
    print()

    all_approve = True
    for review in results.get('results', [results]):
        perspective = review.get('perspective', 'Reviewer')
        verdict = review.get('verdict', 'UNCLEAR')
        confidence = review.get('confidence', 0.0)

        print(f"{perspective}:")
        print(f"  Verdict: {verdict}")
        print(f"  Confidence: {confidence:.2f}")

        if verdict != 'APPROVE':
            all_approve = False

        critique = review.get('critique', '')
        if critique:
            print(f"  Critique: {critique[:500]}...")

        suggestions = review.get('suggestions', '')
        if suggestions:
            print(f"  Suggestions: {suggestions[:500]}...")

        print()

    # Summary
    print("="*80)
    if all_approve:
        print("✅ APPROVED - Blog post is ready")
    else:
        print("⚠️  REVISIONS RECOMMENDED")
    print("="*80)

    # Save review
    review_file = Path(__file__).parent / "blog_validation_results.json"
    import json
    with open(review_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nReview saved to: {review_file}")

if __name__ == "__main__":
    asyncio.run(validate_blog())
