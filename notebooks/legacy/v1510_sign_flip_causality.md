# V15.10: Sign-Flip Causality Test

## Why This Experiment Matters

We've established that the orthogonal component of the chat direction produces strong steering effects in Llama-3-8B (V15.6b verified -6.07 specific effect for jailbreaking). However, a sophisticated reviewer could raise a fundamental objection about the nature of this effect.

The objection goes like this: "You found a direction that, when subtracted from activations, reduces refusal probability. But how do you know this is a genuine causal control axis rather than a statistical artifact? Maybe you're just pushing the model into an out-of-distribution region that happens to produce compliance. The direction might not represent 'refusal' as a controllable concept—it might just be a perturbation that breaks the model in a particular way."

The sign-flip test is designed to definitively answer this objection.

## The Logic of Sign-Flip Testing

If the orthogonal component truly represents a causal axis for refusal/compliance behavior, then steering in opposite directions should produce opposite effects. Specifically, positive steering (adding the direction) should push the model toward increased refusal, while negative steering (subtracting the direction) should push the model toward decreased refusal (jailbreak).

Moreover, the magnitudes should be approximately symmetric. If steering at +3.0 produces a +2.0 change in refusal score, then steering at -3.0 should produce approximately a -2.0 change. Perfect symmetry isn't required, but the effects should be in opposite directions with comparable magnitudes.

The random control direction should show no systematic relationship between steering sign and behavioral change. Any drift should be small and not consistently directional.

## What Symmetry Proves

If sign-flip symmetry holds, several important conclusions follow.

First, the direction is causally relevant, not just statistically correlated. We can move behavior in both directions along this axis, which means we're controlling the underlying mechanism rather than just finding a perturbation that happens to work.

Second, we have genuine bidirectional control. This isn't a one-way trick where subtracting the vector breaks the model's refusal mechanism. We can both increase and decrease refusal by steering along this axis.

Third, the direction represents a meaningful dimension of the model's behavior space. The fact that opposite steering produces opposite effects suggests the direction captures something real about how the model represents refusal versus compliance.

## The Test Protocol

We test the orthogonal component at seven steering strengths: -3.0, -2.0, -1.0, 0.0, +1.0, +2.0, and +3.0. For each strength, we compute the contrastive score (refusal logprob minus compliance logprob) across all evaluation prompts.

The key analysis computes the delta from baseline (α=0) for each non-zero strength. We then check whether the sign of the delta flips when the steering sign flips. A direction passes the sign-flip test if at magnitudes 1, 2, and 3, positive α produces positive delta and negative α produces negative delta (or vice versa, consistently).

We also compute a symmetry ratio, which is the absolute value of (delta at +α) divided by (delta at -α). Perfect symmetry would give a ratio of 1.0. Ratios between 0.3 and 3.0 are considered reasonable, accounting for nonlinearities and noise.

## Possible Outcomes

The best outcome is CAUSAL_AXIS_CONFIRMED. Sign-flip holds at all magnitudes, symmetry ratios are reasonable, and effect magnitudes substantially exceed the random control. This provides strong evidence that the orthogonal component is a genuine bidirectional control axis for refusal behavior.

A moderate outcome is CAUSAL_AXIS_LIKELY. Sign-flip holds but effects may be weak or asymmetric. This suggests the direction is probably causal but with some caveats. We might see saturation at extreme values or nonlinear effects that break perfect symmetry.

The concerning outcome is CAUSAL_AXIS_NOT_CONFIRMED. Sign-flip does not hold consistently. Perhaps positive steering increases refusal but negative steering has no effect, or effects are in the same direction regardless of sign. This would suggest the direction works through a different mechanism than we thought—perhaps breaking the model rather than controlling a meaningful axis.

## Implications for the Paper

If CAUSAL_AXIS_CONFIRMED, we add a figure showing the symmetric trajectory and claim that "the orthogonal component exhibits sign-flip symmetry, confirming it as a causal control axis for refusal behavior." This directly addresses reviewer concerns about artifacts versus genuine causal structure.

If CAUSAL_AXIS_NOT_CONFIRMED, we need to be more careful in our claims. We might still say that the direction is effective for reducing refusal, but we couldn't claim it represents a bidirectional control axis. The paper would focus more on the decomposition finding (orthogonal component effectiveness) rather than claiming causal control.

## Relationship to Other Experiments

This experiment complements V15.6b (iso-norm verification), which ruled out magnitude artifacts. Together, V15.6b and V15.10 address the two main methodological objections to steering results. V15.6b says the effect isn't due to comparing weak and strong signals, while V15.10 says the effect isn't a one-way perturbation artifact.

The sign-flip test also provides evidence relevant to the two-phenotype framing. If the orthogonal component is a genuine causal axis in Llama (Type I), we would expect different behavior in Qwen (Type II) where base and chat directions are nearly parallel. In Type II models, both the base and chat directions should function as causal axes since they point in the same direction.

## Runtime

Approximately 1 hour on A100, covering loading directions and computing decomposition, three full steering sweeps (7 strengths × 10 prompts each) for orthogonal, full chat, and random directions, and symmetry analysis and visualization.
