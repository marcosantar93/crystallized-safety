# V15.9: Qwen-2.5-7B Replication

## Purpose

V15.4-V15.6 established three key findings on Llama-3-8B and Mistral-7B: geometric separation between base and chat directions (~0.17-0.21 cosine similarity), asymmetric cross-transfer (chat→base works, base→chat fails), and supersession (the orthogonal component drives effects on both models).

With N=2 model families, claims about "cross-architectural" phenomena are weak. A hostile reviewer could argue the findings are specific to Meta/Mistral training recipes, Llama-style architectures, or coincidental alignment between two similar models.

This experiment tests whether the findings generalize to **Qwen-2.5-7B**, a third model family with substantially different characteristics.

## Why Qwen?

Qwen-2.5-7B differs from Llama and Mistral in several important ways:

**Pre-training data:** Qwen was trained on Chinese + English data, unlike Llama/Mistral which are primarily English. This means different tokenization patterns and potentially different concept representations.

**Alignment recipe:** Qwen uses Alibaba's proprietary alignment method, which differs from Meta's RLHF and Mistral's approach. If the geometric separation is an artifact of a specific RLHF implementation, Qwen might not show it.

**Tokenizer:** Qwen uses a different BPE vocabulary optimized for Chinese+English, which affects how concepts are encoded at the token level.

**Architecture:** Qwen-2.5-7B uses dense attention like Llama (unlike Mistral's sliding window), but has 28 layers instead of 32, and different hidden dimensions.

If Qwen shows the same pattern as Llama and Mistral despite these differences, we can make much stronger claims about the findings being fundamental to how RLHF transforms representations, not artifacts of specific training recipes.

## The Test

The experiment runs the core V15.4-V15.5 protocol on Qwen:

1. **Direction extraction:** Extract harm/safety directions from Qwen-2.5-7B (base) and Qwen-2.5-7B-Instruct (chat) using forced-compliance methodology at layer 11 (40% of 28 layers).

2. **Geometric analysis:** Compute cosine similarity between base and chat directions. Does Qwen show the ~0.2 similarity seen in Llama/Mistral?

3. **Bidirectional transfer test:** Apply sequence logprob steering in all four conditions (chat+native, chat+cross, base+native, base+cross). Does Qwen show the same asymmetric pattern?

## Predictions

If the findings generalize to Qwen, we expect:

**Geometric separation:** Base↔Chat cosine similarity < 0.3, comparable to Llama (0.207) and Mistral (0.173).

**Asymmetric transfer:** Chat direction induces refusal in base model (cross works). Base direction fails to jailbreak chat model (cross fails). This is the signature asymmetry from V15.5.

**Native steering effectiveness:** At minimum, the chat model should respond to its native direction. (Base native often fails due to ceiling effects, so this is less diagnostic.)

## Interpretation Guide

**FULL_REPLICATION (3/3):** Qwen shows all three predicted patterns. The findings are genuinely cross-architectural. We can claim the phenomenon reflects something fundamental about how RLHF transforms representations, not specific to Llama/Mistral/Meta training.

**PARTIAL_REPLICATION (2/3):** Qwen shows some but not all patterns. The core phenomenon may generalize with variations. Need to investigate what differs and whether it's due to architecture, training recipe, or extraction quality.

**REPLICATION_FAILED (0-1/3):** Qwen does not show the same pattern. The findings may be architecture-specific or training-recipe-specific. Would need to restrict claims to Llama/Mistral or investigate what makes Qwen different.

## What Each Outcome Means

**Full replication** transforms the paper from "we observed this in two model families" to "this is a cross-architectural phenomenon observed across three distinct model families with different training recipes." That's a much stronger claim for publication.

**Partial replication** is still valuable. It identifies boundary conditions and suggests follow-up experiments. For example, if geometric separation holds but asymmetric transfer doesn't, that would suggest the separation is universal but the transfer dynamics are architecture-dependent.

**Failed replication** is also informative. It would mean the findings are more specific than we thought, which affects the scope of claims but doesn't invalidate the Llama/Mistral results. The paper would need to be more careful about generalization.

## Comparison to Llama/Mistral

The experiment is designed to produce directly comparable metrics:

| Metric | Llama-3-8B | Mistral-7B | Qwen-2.5-7B |
|--------|------------|------------|-------------|
| Base↔Chat similarity | 0.207 | 0.173 | ? |
| Chat→Base cross works | ✓ (+1.43) | ✓ (+1.49) | ? |
| Base→Chat cross works | ✗ (+0.40) | ✗ (+1.21) | ? |
| Chat native works | ✓ (-4.95) | ✗ (-0.45) | ? |

The notebook will fill in the Qwen column and produce a summary figure comparing all three families.

## Expected Output

The notebook produces extraction metrics (separation, direction norms), geometric similarity, steering results for all four conditions, and a replication score (0-3) with automated verdict. The key deliverable is a clear answer to: "Does the finding generalize beyond Llama/Mistral?"

## Implications for the Paper

**If full replication:** Change all instances of "in the two model families tested" to "across three architecturally distinct model families." Add Qwen to comparison figures. Strengthen claims about universality.

**If partial/failed:** Keep claims restricted to Llama/Mistral. Discuss Qwen as a boundary condition or contrasting case. May need to investigate further before making broader claims.

## Runtime

Approximately 2 hours on A100 (extraction from both models + bidirectional steering tests).
