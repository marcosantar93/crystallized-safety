# V15.17: Hook Position Verification — Experiment Summary

## One-Line Summary

Test whether methodology sensitivity persists across different hook positions (resid_pre, attn_out, mlp_out, resid_post) to rule out "wrong tap point" as an alternative explanation.

---

## The Alternative Hypothesis We're Testing Against

A sophisticated reviewer might argue that V15.4 and V15.12 found different similarities not because they access different functional subspaces, but because they implicitly tap into different parts of the transformer computation. Maybe V15.4's methodology causes attention to dominate while V15.12's causes MLP to dominate, and these components have different base-chat alignment.

This experiment tests that alternative by explicitly controlling hook position while varying methodology.

---

## The Four Hook Positions

Within each transformer layer, there are multiple points where we can extract activations:

**resid_pre**: The residual stream before the attention sublayer processes it. This captures the "input" to the current layer.

**attn_out**: The output of the attention mechanism before it's added to the residual stream. This captures what attention contributes.

**mlp_out**: The output of the MLP after attention, before it's added to the residual. This captures what the MLP contributes.

**resid_post**: The final residual stream after both attention and MLP have contributed. This is our default extraction point.

---

## The Experimental Design

We run two extraction methodologies at layer 12:

**V15.4-style**: Few-shot forced compliance (generation-accessing)

**V15.12-style**: Standard contrastive, neutral framing (discrimination-accessing)

For each methodology, we extract at all four hook positions and compute base-chat similarity.

---

## What We Measure

For each (methodology × hook position) combination:
- Base-chat cosine similarity
- Separation quality

The key analysis is whether the methodology difference (V15.4 vs V15.12) persists across hook positions.

---

## Predictions

If the generation-discrimination framework is correct:

**V15.4 should show lower similarity than V15.12 at every hook position**. The functional subspace difference should be visible regardless of where we tap into the computation.

**The absolute similarity values may vary by position**, but the relative ordering (V15.4 < V15.12) should be stable.

**The difference magnitude should be relatively consistent** (within 0.1-0.2) across positions.

---

## What Would Falsify the Framework

If **V15.4 and V15.12 show similar similarity at some hook positions** but diverge at others, the methodology effect may be position-dependent rather than reflecting distinct functional subspaces.

If **V15.4 shows higher similarity than V15.12 at certain positions**, something is fundamentally wrong with our understanding—the relationship reverses.

If **all positions show the same similarity regardless of methodology**, hook position may be the confound we missed, and the V15.4 vs V15.12 difference arose from implicit position differences.

---

## Why This Matters

The transformer architecture processes information in a specific sequence: residual → attention → add → residual → MLP → add → residual. Different methodologies could, in principle, bias which part of this computation dominates the final representation.

By explicitly controlling hook position, we isolate the methodology effect from any architectural confounds. If methodology sensitivity persists across all positions, we can confidently attribute it to functional subspace access rather than architectural tap-point effects.

---

## Runtime and Resources

Approximately 1.5 hours on A100 GPU. We run 2 methodologies × 4 positions × 2 models = 16 extraction conditions.

---

## Questions for Reviewers

1. Should we test at multiple layers (e.g., layer 12 and layer 24) to see if hook position effects change with depth?

2. Are there other hook positions we should consider? For example, post-layer-norm positions?

3. If the methodology effect is stronger at certain hook positions than others, how do we interpret that? Does it tell us something about where the generation-discrimination split is implemented?

4. Should we also run steering tests at each hook position, or is similarity sufficient for this verification?
