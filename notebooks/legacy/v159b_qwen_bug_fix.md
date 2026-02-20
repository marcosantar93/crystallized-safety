# V15.9b: Qwen Bug Fix and Symmetric Transfer Verification

## The Problem

The original V15.9 experiment produced a critical geometric finding (0.94 cosine similarity between Qwen base and chat directions) but contained corrupted behavioral data. A reviewer identified the issue by examining the raw JSON output.

At α=0.0, where no steering is applied, the reported scores were chat_native at 10.69 and chat_cross at -3.22. These values must be identical because when steering strength is zero, the model is untouched regardless of which direction we nominally "apply." The 13.9-point difference indicates that the "cross" condition was accidentally evaluated on the base model rather than the chat model.

## What We're Fixing

The V15.9b notebook adds explicit verification at multiple levels to prevent this error.

First, we implement baseline verification before running full experiments. We compute the α=0 score independently for each condition and verify that all conditions on the same model produce consistent baselines (within 0.5 points).

Second, we add explicit model identification in all logging so that each output line clearly identifies which model (base or chat) is being evaluated.

Third, we compute baseline spread after all tests complete, checking that native, cross, and random conditions all started from the same point. If spread exceeds 0.5, we flag a warning.

## The Type II Prediction

The 0.94 cosine similarity suggests Qwen may be a "Type II: Linear Amplification" model where RLHF sharpens existing safety features rather than creating orthogonal new ones. If this interpretation is correct, we predict symmetric transfer will occur.

For Type I models like Llama (0.21 similarity), transfer is asymmetric. The chat direction works on both models, but the base direction only works on base, not on chat. This is because the chat model's safety circuits live in an orthogonal subspace that the base direction doesn't access.

For Type II models like Qwen (0.94 similarity), both directions should work on both models because they point in nearly the same direction. The chat direction is just a scaled version of the base direction. Steering with either should produce similar effects.

## Possible Outcomes

If baselines are consistent and symmetric transfer is observed, Qwen is confirmed as Type II. The base direction works on chat, and the chat direction works on base. This validates the two-phenotype taxonomy and suggests Qwen's RLHF process amplified existing safety features rather than creating new orthogonal ones.

If baselines are consistent but no transfer is observed, we have a puzzle. The high geometric similarity doesn't translate to behavioral equivalence. Possible explanations include that the directions are geometrically similar but not causally equivalent, that Qwen's safety mechanism operates through different circuits than Llama/Mistral, or that our extraction methodology fails to capture Qwen's actual control-relevant features.

If baselines are still inconsistent, there's a deeper bug than variable passing. We would need to investigate tokenizer differences, prompt formatting issues, or model loading problems.

## What This Means for the Paper

If Qwen confirms as Type II, the paper's core contribution becomes demonstrating that alignment has distinct geometric phenotypes. We propose base-chat cosine similarity as a simple diagnostic that predicts whether interpretability tools developed on one model will transfer to its aligned variant. Type I models require model-specific tools while Type II models may support more universal approaches.

If Qwen doesn't confirm cleanly, we need to investigate further before committing to the two-phenotype framing. The geometric finding (0.94 similarity) still stands and contrasts with Llama/Mistral, but the behavioral interpretation would be uncertain.

## Runtime

Approximately 2 hours on A100, covering direction loading/verification, chat model steering tests (3 conditions × 4 strengths × 10 prompts), base model steering tests (3 conditions × 4 strengths × 10 prompts), and analysis and visualization.
