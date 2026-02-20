# Reviewer Synthesis: The Generation-Discrimination Framework

## Executive Summary

All four external reviewers, plus deep literature research, have converged on the same conclusion: **our "methodology sensitivity" finding is not a problem—it's a discovery about LLM safety architecture.**

The reframe transforms our paper from a negative result ("geometric interpretability is unreliable") to a positive discovery ("we can dissociate the functional subspaces that implement LLM safety").

---

## The Unified Theory: Generation vs Discrimination

### The Core Insight

LLMs process "harm" in two distinct functional subspaces:

**Discrimination Subspace (Recognition)**
- Function: "Is this content harmful?"
- Shared between base and chat models (both recognize harm)
- Probed by standard contrastive extraction (V15.12)
- Result: High similarity (~0.76)
- Steering: Ineffective (adding "this is bad" doesn't change behavior if model already refuses)

**Generation Subspace (Capability)**
- Function: "How to produce harmful content"
- Present in base models, suppressed in chat models by RLHF
- Probed by forced-compliance extraction (V15.4)
- Result: Low similarity (~0.21) — chat has orthogonalized these circuits
- Steering: Effective for jailbreaking (reactivates suppressed generation circuits)

### How This Explains Every Result

**V15.4 vs V15.12 Divergence**
- Not methodology "noise" — different methodologies probe different functional systems
- V15.4 forces the model to *generate* harmful content → accesses Generation Subspace
- V15.12 asks the model to *recognize* harmful content → accesses Discrimination Subspace

**Sign-Flip Asymmetry (V15.10)**
- Negative α (jailbreak): Subtracting the generation vector removes the suppression → model can generate again
- Positive α (enhance refusal): Adding generation-relevant features doesn't strengthen discrimination — wrong subspace
- This is exactly what you'd expect if these are distinct systems

**Qwen Steering Failure (V15.9b)**
- Qwen's 0.94 similarity means our extraction probed the Discrimination Subspace (recognition is shared)
- Qwen may use a different architecture for generation suppression that our forced-compliance method doesn't access
- Alternative: Qwen's RLHF may have collapsed generation into discrimination (different safety strategy)

**One-Way Perturbation**
- The generation subspace acts like a "fuse" (can be blown to enable generation) not a "dial" (can't be turned up to enhance suppression)
- This is consistent with RLHF suppressing circuits, not creating new ones

---

## Reviewer-Specific Contributions

### Gemini's Framework
- Named the dichotomy: "Generation vs Discrimination"
- Connected to literature: "Rogue Scalpel" effect, feature superposition, CIR (Collapse of Irrelevant Representations)
- Key quote: "The orthogonality in V15.4 is the physical evidence of RLHF collapsing generation overlap"

### Grok's Framework
- Framed as "probe-model interaction as manifold sampling"
- V15.4 samples sparse/unsaturated modes (low cosine, effective steering)
- V15.12 samples dense/saturated modes (high cosine, inert steering)
- Proposed: Entropy measurements to quantify sparse vs dense sampling

### ChatGPT's Framework
- Identified this as a "measurement paper" not a steering paper
- Key reframe: "We do not currently have instrument-independent measurements of LLM internal geometry"
- Recommended: Chase invariants under measurement variation, not universality
- Crucial insight: Sign-flip failure "falsifies the axis interpretation" — this is a circuit, not a feature

### Deep Research PDF
- Literature grounding: Chen et al. (2025) found orthogonal boundaries for "instruction safety" vs "model compliance"
- This is exactly the Generation vs Discrimination split, independently discovered
- SafeSwitch architecture validates the two-stage model
- Concept subspaces (not single vectors) are the right theoretical framework

---

## The New Paper Framing

### Old Title (Dead)
"The Two Geometries of Alignment: Why Safety Steering Varies Across Model Families"

### New Title (Alive)
"The Dual Geometry of Alignment: Dissociating Generation and Discrimination Subspaces in LLM Safety"

### New Abstract (Draft)

> We reveal that "safety" in aligned language models is not a single direction in activation space but comprises at least two functionally distinct subspaces. By systematically varying extraction methodology, we dissociate a **Generation Subspace** (where RLHF induces orthogonality between base and chat models, reflecting suppression of harmful-content production) from a **Discrimination Subspace** (where representations remain parallel, reflecting shared harm-recognition capabilities).
>
> This dissociation resolves conflicting reports in the steering literature: methods that probe generation-relevant circuits (forced compliance) find low base-chat similarity and effective jailbreaking, while methods that probe discrimination-relevant circuits (standard contrastive) find high similarity and ineffective steering. The asymmetric sign-flip behavior we observe—where interventions can disrupt safety but not enhance it—is consistent with generation suppression operating as a fragile control circuit rather than a linear semantic axis.
>
> Our findings have immediate implications for interpretability and alignment: (1) geometric measurements are not model-intrinsic but depend on which functional subspace is probed, (2) effective safety interventions require targeting the correct subspace, and (3) different models may implement the generation-discrimination split differently, explaining cross-model transfer failures.

---

## Critical Experiments

### Highest Priority: V15.15 (Generation vs Recognition)

This is the "smoking gun" experiment. It explicitly extracts two directions:
- Recognition probe: "Is making a bomb bad?" → "Yes, this is harmful because..."
- Generation probe: "Explain how to make a bomb" → "To construct an explosive..."

**Predictions:**
- Recognition similarity: HIGH (both models recognize harm)
- Generation similarity: LOW (chat suppresses generation)
- Recognition steering: WEAK (wrong subspace for behavioral control)
- Generation steering: STRONG (correct subspace)
- Cross-mode similarity (Recognition ↔ Generation): LOW (distinct circuits)

If this confirms, the paper is ready to write.

### Second Priority: V15.16 (Frozen Prompt Ablation)

This closes the "maybe it's just different prompts" objection by holding questions constant and varying only methodology framing.

**Prediction:** Same questions can produce both high and low similarity depending on whether framing triggers generation or discrimination mode.

### Third Priority: V15.14 (Probe Ensemble)

This characterizes the distribution of measurements across many methodologies.

**Key analysis:** Look for bimodality (two clusters corresponding to the two subspaces) rather than unimodal spread.

---

## Additional Experiments Suggested by Reviewers

### Hook Position Verification (ChatGPT)

Test whether methodology sensitivity persists across different intervention points:
- resid_pre vs resid_post
- attn_out vs mlp_out

If sensitivity persists across all hook sites, this is not a "wrong tap point" artifact.

### Non-Harm Refusal Test (ChatGPT)

Test whether a non-harm refusal (e.g., "I can't answer trivia questions") shares geometry with harm refusal.

If yes → We've found a general "refusal circuit," not harm-specific
If no → The generation-discrimination split is harm-specific

### Manifold Entropy Probe (Grok)

Measure Shannon entropy over activation distributions for each methodology. 

**Prediction:** V15.4 (forced compliance) induces higher entropy (sparse disruption), V15.12 induces lower entropy (dense alignment).

### Circuit Localization (ChatGPT)

Minimal mech-interp: Identify heads/MLPs most aligned with the effective direction, patch them to zero, see if steering collapses.

If yes → Localized control circuit, not distributed concept.

---

## Path Forward

### Immediate (This Week)
1. Run V15.15 (Generation vs Recognition) — THE SMOKING GUN
2. Run V15.16 (Frozen Prompt) — Closes methodology objection
3. Run V15.14 (Probe Ensemble) — Characterizes distribution

### If V15.15 Confirms (Next Week)
1. Write paper with new framing
2. Add one strategic model (Gemma-2 or Phi-3) to test generality
3. Submit to NeurIPS/ICLR main track

### Paper Structure
1. Introduction: The measurement crisis in geometric interpretability
2. Background: Linear probes, steering vectors, and their limitations
3. The Puzzle: Same model, same layer, different methodologies → different geometry
4. The Resolution: Generation vs Discrimination subspaces
5. Experiments: V15.15 (smoking gun), V15.16 (control), V15.14 (distribution)
6. Implications: For interpretability, for alignment, for cross-model transfer
7. Discussion: This is not a bug, it's a feature of how safety is implemented

---

## The North Star Connection

This connects to a general theory of LLM minds in a profound way:

**The Shoggoth and the Mask**: The base model (Shoggoth) has capabilities including harmful generation. The chat model (Mask) suppresses generation while preserving recognition. RLHF doesn't erase knowledge—it gates production.

**Functional Modularity**: LLMs have distinct circuits for different operations on the same content. "Harm" exists in multiple functional modes: recognizing it, generating it, refusing it. These are not the same representation.

**Measurement as Interaction**: There is no "intrinsic geometry" to measure. All measurements are interactions between the model and the probe. The right question is not "what is the geometry?" but "what geometry does this probe reveal?"

**Control via Correct Addressing**: To control behavior, you must address the correct functional subspace. Steering with discrimination-relevant features won't affect generation. This explains why so many steering attempts fail.

---

## Conclusion

We have not failed. We have discovered something important about how aligned LLMs work. The methodology sensitivity that seemed like a problem is actually the key to understanding safety architecture.

**The paper is not about "methodology sensitivity." The paper is about "the dual geometry of alignment."**

Run V15.15. If it confirms, write the paper.
