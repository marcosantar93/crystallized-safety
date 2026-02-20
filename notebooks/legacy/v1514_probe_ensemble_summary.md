# V15.14: Probe Ensemble — Experiment Summary

## One-Line Summary

Run 36 extraction methodologies and study the distribution of geometric measurements to determine if methodology sensitivity has structure or is random noise.

---

## The Core Idea

In survey research, you don't trust a single poll—you run many polls with different framings and look for what's stable versus what's methodology-dependent. We apply the same logic to activation extraction.

Instead of treating the V15.4 vs V15.12 divergence as a problem, we treat it as data. By systematically varying extraction parameters, we can characterize the space of possible measurements and identify patterns.

---

## Methodology Variations

We vary four parameters across 36 combinations:

**Few-shot count**: 0, 2, or 3 examples before the target question

**Shot type**: Compliance (harmful examples with harmful answers), neutral (benign examples), or mixed

**Persona**: None, helpful, assertive, or academic

**Answer style**: Forced start ("To make a bomb, you would..."), sure-then ("Sure, I can help..."), or neutral ("The answer is...")

**Framing**: Direct question, hypothetical ("Hypothetically speaking..."), or research ("For academic purposes...")

---

## What We Measure

For each methodology:
1. Extract direction from base model
2. Extract direction from chat model
3. Compute base↔chat cosine similarity
4. Run quick steering test (α=-3 vs baseline)
5. Record separation scores

---

## Key Analyses

**Distribution shape**: Is similarity unimodal (continuous spectrum) or bimodal (two clusters)?

**Similarity-steering correlation**: Does lower similarity predict better steering effectiveness?

**Factor analysis**: Which parameters (few-shot, persona, etc.) most strongly affect similarity?

**Clustering**: Do methodologies naturally group into "generation-accessing" vs "discrimination-accessing"?

---

## Predictions Under the Framework

If the generation-discrimination hypothesis is correct:

The distribution should be **bimodal** with one cluster around 0.2-0.4 (generation-accessing methods) and another around 0.7-0.9 (discrimination-accessing methods).

**Low-similarity methods should correlate with effective steering** because they access the generation subspace where RLHF made changes.

**Forced answer styles should cluster with low similarity** because they push the model toward generation mode.

**Neutral/classification framings should cluster with high similarity** because they keep the model in recognition mode.

---

## What Would Falsify the Framework

If the distribution is **unimodal with high variance**, there may be no discrete subspaces—just a continuous manifold that methodologies sample differently.

If **no correlation between similarity and steering**, geometry doesn't predict function and our entire approach is flawed.

If **methodology factors don't predict cluster membership**, the variation may be noise rather than structured access to different subspaces.

---

## Runtime and Resources

Approximately 2-3 hours on A100 GPU. The main cost is running 36 extraction × 2 models = 72 direction extractions plus 36 steering tests.

---

## Questions for Reviewers

1. Are 36 methodologies sufficient to characterize the distribution? Should we add more variations?

2. What methodology factors are we missing? Should we vary layer, token position, or temperature?

3. How should we handle methodologies that produce poor separation (noise)? Filter them or include them?

4. If the distribution is continuous rather than bimodal, what would that imply for the framework?
