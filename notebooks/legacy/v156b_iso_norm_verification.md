# V15.6b: Iso-Norm Verification (Magnitude Confound Audit)

## The Concern

A reviewer raised a critical methodological concern about V15.6:

> "The parallel component has only 4% variance (norm ~0.2) compared to the orthogonal component (norm ~0.98). When you steered with 'Parallel,' you were effectively steering with a 5x weaker signal. Of course it failed!"

This is a legitimate concern. If the V15.6 comparison was between unnormalized components, the results would be meaningless—we'd be comparing a whisper to a shout.

## The Mathematical Reality

Here's the key insight that makes this verification tractable:

By construction, the parallel component is the projection of the chat direction onto the base direction:

```
parallel = (chat · base_normalized) * base_normalized
```

After normalizing to unit norm, `parallel_normalized` points in **exactly the same direction** as `base_normalized`. They are mathematically identical vectors (differing only in sign).

This means:
- "Parallel fails to steer" is equivalent to "Base fails to steer"
- We already knew base fails from V15.4 and V15.5
- The decomposition finding is consistent, not surprising

## What This Experiment Verifies

The V15.6b notebook performs explicit verification of three claims:

**Verification 1: Parallel = Base (after normalization)**

We compute `cosine(parallel_normalized, base_normalized)` and assert it equals ±1.0. If this fails, there's a bug in the decomposition code. If it passes, steering with parallel is mathematically identical to steering with base.

**Verification 2: Components are orthogonal**

We compute `|dot(parallel_normalized, orthogonal_normalized)|` and assert it's < 1e-5. This verifies the decomposition is geometrically correct.

**Verification 3: All test directions have exactly unit norm**

We explicitly verify that every direction used in steering tests has norm = 1.0, ensuring the iso-norm condition is satisfied.

## The Steering Test

After verification, we run steering tests with:
- `base_normalized` (unit norm)
- `parallel_normalized` (unit norm, should equal base_normalized)
- `orthogonal_normalized` (unit norm)
- `full_chat_normalized` (unit norm)
- 5 random directions (unit norm, for null distribution)

All directions use the same fixed scale factor, ensuring true iso-norm comparison.

## Expected Results

**If the verification passes:**

We expect `base_normalized` and `parallel_normalized` to show identical (weak) effects, because they're the same direction. We expect `orthogonal_normalized` to show strong effects, consistent with V15.6.

This would confirm that V15.6's finding is NOT a magnitude artifact—the parallel component genuinely fails because it points in a non-causal direction (the base direction), not because it was too small.

**If base and parallel show different effects:**

This would indicate a bug in the decomposition or steering code that needs investigation.

## Implications

If verification passes with base ≈ parallel effects, the reviewer's magnitude confound concern is addressed. The finding stands: the chat direction is 96% orthogonal to base, and that orthogonal component is what actually controls both models.

The "parallel is inert" finding is not surprising—it's mathematically equivalent to "base is inert," which we already established in V15.4-V15.5. The decomposition experiment's value is in showing that the *orthogonal* component (the part that shares nothing with base) is what works, not the shared part.

## Runtime

Approximately 30 minutes on A100 (verification + steering tests on chat model only).
