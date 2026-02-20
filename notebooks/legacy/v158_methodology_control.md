# V15.8: Extraction Methodology Control

## Purpose

V15.6 showed that the orthogonal component (96% of the chat direction, perpendicular to the base direction) controls both base and chat models, while the parallel component (4%, aligned with base) is inert. We interpreted this as "RLHF discovers better features that supersede base-extracted directions."

But there's a critical alternative explanation: **maybe our base extraction methodology is simply bad, and any direction extracted that way would be weak—regardless of model state.**

This experiment directly tests whether extraction methodology or model state is the key variable.

## The Test

We extract a direction from the **chat model** using the exact same methodology we used for base: few-shot examples forcing compliance, completion-style prompts without the chat template, and the same harmful/benign contrastive pairs.

This produces a "degraded chat vector"—a direction extracted from the aligned model but using the naive base methodology.

Then we compare this degraded vector to three reference directions:
1. The proper chat direction (extracted with chat template)
2. The V15.4 base direction
3. A random direction (control)

Finally, we test all directions' ability to steer the chat model.

## Predictions

**If supersession is correct (RLHF genuinely discovers better features):**

The degraded chat vector should behave like the base vector because the methodology, not the model state, determines what features get captured. Specifically:

1. Degraded↔Base similarity should be HIGH (>0.5) — they should look alike because they're extracted the same way
2. Degraded↔Proper similarity should be LOW (<0.5) — the proper chat direction captures different features
3. Degraded steering should FAIL — it's missing the causally relevant features
4. Proper steering should WORK — it captures the right features

This would prove that extraction methodology is the key factor. The base direction was weak because few-shot completion-style extraction misses the causally relevant features, not because base models lack them.

**If methodology is NOT the issue:**

The degraded chat vector will still work well, resembling the proper chat direction more than the base direction. This would mean the chat model genuinely has better/different representations that are robust to extraction methodology. The supersession interpretation would be strengthened—RLHF changes the underlying representations, not just how accessible they are.

## Scientific Value

This experiment addresses the most obvious methodological objection: "Your base extraction might just be bad." By applying the same "bad" methodology to the chat model and seeing what happens, we can distinguish between two very different stories:

**Story A (Methodology):** The base direction was weak because few-shot extraction is a poor methodology. The chat model's features are only accessible via proper prompting. RLHF doesn't change what features exist; it changes how accessible they are.

**Story B (Supersession):** RLHF genuinely transforms representations. The chat model has different/better features that exist regardless of how we extract them. Even "degraded" extraction from chat produces something meaningfully different from base.

Both stories are interesting, but they have different implications for interpretability and alignment.

## Interpretation Guide

**METHODOLOGY_CONFIRMED (4/4):** Extraction methodology is the key factor. The "supersession" finding is really about prompt engineering and feature accessibility, not fundamental representational changes. This is still valuable—it means interpretability tools must use appropriate prompting for each model state.

**METHODOLOGY_REJECTED (0-1/4):** The degraded chat vector still works despite bad methodology. RLHF genuinely creates different representations. The supersession interpretation is correct in its strong form.

**METHODOLOGY_PARTIAL (2-3/4):** Both factors matter. Model state AND methodology contribute to direction quality. The story is more nuanced than either extreme.

## Expected Output

The notebook produces cosine similarities between all direction pairs, steering effectiveness for each direction, and a hypothesis test scoring 0-4 predictions. The key deliverable is a clear answer to: "Is the base direction weak because of methodology or because of model state?"

## Implications for the Paper

If methodology is confirmed, the paper should emphasize that "interpretability tools require alignment-appropriate prompting" and the orthogonality reflects prompt-format sensitivity. 

If methodology is rejected, the paper can make stronger claims about RLHF fundamentally transforming representations, not just making them differently accessible.

Either way, we learn something important about the nature of the finding.

## Runtime

Approximately 1 hour on A100 (extraction + steering tests on chat model).
