# V15.10 Review Request: The Two-Phenotype Pivot

## Executive Summary

The validation experiments (V15.6b-V15.9) have revealed that our original "universal rotation" hypothesis was too strong, but they've also uncovered something more interesting: **alignment appears to have distinct geometric phenotypes**. Llama and Mistral show ~0.2 cosine similarity (orthogonal), while Qwen shows ~0.94 (nearly parallel). Rather than treating Qwen as a "failed replication," we're proposing to reframe the paper around this **taxonomy of alignment geometries**.

We are now running two additional experiments to solidify this framing before proceeding.

---

## What We Changed and Why

### The Original Hypothesis (Falsified)

Our original claim was that RLHF universally induces a geometric rotation of safety representations, creating ~80° orthogonality between base and chat harm directions across all aligned models. We interpreted this as "representational supersession"—RLHF discovers more powerful features that supersede naive base-extracted directions.

### What the Validation Experiments Showed

**V15.6b (Iso-Norm Verification): PASSED ✓**

The magnitude confound concern was definitively ruled out. Under strict iso-norm conditions (all directions with unit norm, identical scale factors), the orthogonal component still shows -6.07 effect while the base-aligned component shows +1.06. The decomposition finding is real and not an artifact.

**V15.7 (Layer Trajectory): UNEXPECTED**

The same model (Llama-3-8B) produces 0.76 similarity at layer 12 under V15.7's extraction methodology, versus 0.21 under V15.4's methodology. This reveals that the orthogonality is methodology-dependent, not an intrinsic property of the model.

**V15.8 (Methodology Control): HYPOTHESIS REJECTED**

The "degraded" chat direction (extracted using base methodology) has 0.84 similarity to the proper chat direction, not to the base direction (0.12). The chat model produces similar directions regardless of extraction methodology. This contradicts the supersession prediction.

**V15.9 (Qwen Replication): COMPLETE FAILURE**

Qwen-2.5-7B shows 0.94 similarity between base and chat directions. However, a reviewer identified a likely bug in the behavioral data (inconsistent α=0 baselines), so we need to re-run the steering tests while preserving the (likely correct) geometric finding.

### The New Hypothesis: Two Alignment Phenotypes

Rather than treating Qwen as a failure, we're reframing around the observation that **alignment is not a single geometric transformation**. We propose two phenotypes:

**Type I: Orthogonal Supersession (Llama, Mistral)**

RLHF constructs a new safety manifold orthogonal to pre-training representations. Cosine similarity is low (~0.2). Cross-transfer is asymmetric: chat→base works, base→chat fails. Safety is an "add-on" governor that overrides the base model's naive concepts. Interpretability tools must be model-specific.

**Type II: Linear Amplification (Qwen)**

RLHF sharpens the existing safety manifold found in pre-training. Cosine similarity is high (~0.94). Cross-transfer should be symmetric (both directions work). Safety is "baked in" during pre-training, and RLHF amplifies it. Interpretability tools may be more universal.

This is arguably a more important finding than "universal rotation"—it explains why some steering papers report success while others report failure. They may be studying different alignment phenotypes without knowing it.

---

## The Two New Experiments

### V15.9b: Qwen Bug Fix and Symmetric Transfer Verification

**Purpose:** The original V15.9 data has inconsistent α=0 baselines (10.69 vs -3.22 for the same model without steering), suggesting the "cross" condition was accidentally evaluated on the wrong model. We need to fix this and verify the Type II prediction.

**The Test:** Re-run all steering conditions with explicit verification that each evaluation uses the correct model. Check that all α=0 baselines are consistent within each model.

**Type II Prediction:** If Qwen is a Linear Amplification model (0.94 similarity), then both native AND cross steering should work. Transfer should be symmetric, contrasting with Llama's asymmetric pattern.

**Possible Outcomes:**

If baselines are consistent AND symmetric transfer is observed, Qwen is confirmed as Type II. This validates the two-phenotype taxonomy.

If baselines are consistent AND no transfer is observed, the high similarity doesn't translate to behavioral control. This would require further investigation.

If baselines are inconsistent, there's still a bug. We need to debug further before drawing conclusions.

### V15.10: Sign-Flip Causality Test

**Purpose:** A skeptical reviewer could argue that our steering results show correlation rather than causation—maybe we found a direction that happens to reduce refusal when subtracted, but it's not a genuine "control axis."

**The Test:** Test the orthogonal component with both positive AND negative steering strengths (α = -3, -2, -1, 0, +1, +2, +3). If it's a genuine causal axis, the effects should be symmetric around zero.

**Prediction:** If positive α → increased refusal and negative α → decreased refusal (with symmetric magnitudes), the direction is a genuine bidirectional control axis, not a one-way perturbation trick.

**Why This Matters:** Sign-flip symmetry is the simplest, most decisive test of causal control. If it holds, we can claim "causal control axis" rather than just "direction that correlates with steering effect." This eliminates a major class of reviewer objections.

---

## Questions for Reviewers

### Q1: Is the Two-Phenotype Framing Appropriate?

We're proposing to reframe the paper from "RLHF universally rotates representations" to "RLHF induces distinct geometric phenotypes across model families, with implications for interpretability tool transferability."

This is a more nuanced claim but potentially more important—it provides a framework for understanding why steering results vary across models.

**Concerns:**

The two-phenotype taxonomy is based on N=3 model families (Llama, Mistral, Qwen). Is this sufficient to propose a taxonomy, or does it appear like post-hoc rationalization of a failed replication?

Should we present this as a definitive taxonomy or as a preliminary observation that motivates future work?

### Q2: Is the Qwen Bug Fix Sufficient?

The V15.9b experiment adds explicit baseline verification to ensure each condition is evaluated on the correct model. This should catch the variable-passing bug identified in the original data.

**Concerns:**

Are there other potential sources of the baseline discrepancy we should check for (tokenizer differences, prompt formatting issues, etc.)?

Should we re-run the entire extraction as well, or is verifying the behavioral evaluation sufficient?

### Q3: Is the Sign-Flip Test Convincing?

We're proposing sign-flip symmetry as the definitive test of causal control. If both +α and -α produce symmetric effects in opposite directions, the objection "this is just a statistical artifact" becomes untenable.

**Concerns:**

Are there failure modes where sign-flip could pass but the direction is still not causally meaningful?

Should we run sign-flip on additional directions (base, chat, random) as controls, or just the orthogonal component?

### Q4: What About Methodology Sensitivity?

V15.7 and V15.8 revealed that the ~0.2 orthogonality is methodology-dependent. Different extraction procedures on the same model produce different geometric relationships.

**Current Plan:** Present this as a finding rather than a bug. The paper should explicitly state that steering directions are methodology-dependent and that the V15.4 methodology happens to reveal actionable structure in Llama/Mistral.

**Concerns:**

Does methodology sensitivity undermine the two-phenotype claim? (Counter-argument: the Qwen 0.94 similarity should be robust to methodology since it's so extreme.)

Should we run V15.7-style layer trajectories on Qwen to see if the similarity varies across layers?

### Q5: What's the Publication Target?

With the two-phenotype framing, we're claiming:

First rigorous demonstration that alignment geometry varies across model families, with a concrete mechanistic account (orthogonal supersession vs. linear amplification) and boundary conditions.

First proposal of a simple diagnostic (base-chat cosine similarity) for predicting interpretability tool transferability.

Causal verification (sign-flip) that the orthogonal component represents a genuine control axis.

**Question:** Is this NeurIPS main track, workshop, or ICLR level? The contribution has shifted from "discovery of universal phenomenon" to "taxonomy of alignment geometries with methodology caveats."

---

## The Proposed Paper Structure

**Title:** "The Two Geometries of Alignment: Why Safety Steering Varies Across Model Families"

**Abstract (Draft):**

Mechanistic interpretability assumes that safety representations are stable and transferable across alignment states. We demonstrate that this assumption fails for a major class of models. By extracting and decomposing harm-relevant directions across multiple model families (Llama-3, Mistral, Qwen-2.5), we identify two distinct geometric regimes of alignment. In Type I (Orthogonal Supersession) models, RLHF induces a massive geometric rotation, creating safety manifolds nearly orthogonal to pre-training representations. In these models, safety probes extracted from base models fail completely (0% transfer), as the aligned model relies on a novel "governor" circuit. In Type II (Linear Amplification) models, safety geometry is conserved, suggesting RLHF amplifies existing features. These findings reveal that "alignment" is not a monolithic geometric transformation. We propose base-chat cosine similarity as a simple diagnostic for predicting whether safety tools will transfer or require retraining.

**Key Figures:**

Figure 1: The geometric bifurcation—bar chart showing Llama (0.21), Mistral (0.17), and Qwen (0.94) similarities.

Figure 2: V15.6b iso-norm verification—orthogonal works (-6.07), parallel fails (+1.06).

Figure 3: V15.10 sign-flip test—symmetric effects confirming causal control axis.

Figure 4: Qwen symmetric transfer (pending V15.9b)—if Type II prediction holds, both directions work.

---

## Timeline

**Immediate (this week):**

Run V15.9b (Qwen bug fix) — 2 hours.

Run V15.10 (sign-flip test) — 1 hour.

**If results support two-phenotype framing:**

Revise paper draft — 1 week.

Internal review and revision — 1 week.

Submit to target venue — 2-3 weeks from now.

**If results complicate the picture:**

Investigate further before committing to framing.

Consider whether additional model families (Gemma, Phi-3) would clarify the taxonomy.

---

## Summary: Are We on the Right Path?

We believe the two-phenotype framing is stronger than the original "universal rotation" claim because it honestly accounts for the Qwen result rather than hiding it, it provides a framework for understanding why steering results vary, the core finding (V15.6b decomposition) survives rigorous verification, and the sign-flip test (V15.10) will provide decisive causal evidence.

**We're asking reviewers to assess:**

Is the two-phenotype taxonomy scientifically appropriate, or does it appear like post-hoc rationalization?

Are V15.9b and V15.10 the right experiments to run before finalizing the paper?

What level of venue is appropriate for this contribution?

What additional experiments (if any) would strengthen the paper?

---

## Files for This Review

**Experimental Notebooks:**

V15.9b: `v159b_qwen_bug_fix.ipynb` — Fixes baseline inconsistency, tests Type II prediction.

V15.10: `v1510_sign_flip_causality.ipynb` — Tests bidirectional causal control via sign-flip.

**Previous Results:**

V15.6b: `iso_norm_verification_llama3-8b.json` — Magnitude confound ruled out.

V15.7: `layer_trajectory_llama3-8b.json` — Methodology sensitivity revealed.

V15.8: `methodology_control_llama3-8b.json` — Supersession hypothesis rejected.

V15.9: `qwen_replication.json` — Original (buggy) Qwen data.
