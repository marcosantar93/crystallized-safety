# V15.5 Phase 1: Reviewer Consensus and Execution Plan

## Reviewer Verdict: GREEN LIGHT (With Critical Fix)

All three reviewers approved Phase 1 with one critical condition: **fix the token ambiguity in Experiment 4 before executing.**

---

## Critical Fix Required

### The Problem

The original `v155_bidirectional_logprob.ipynb` uses single tokens ("I", "Step") as refusal/compliance markers. Both Gemini and ChatGPT flagged this as a fatal flaw:

**Gemini's example:**
- Chat Model: "I cannot help..." → "I" indicates REFUSAL
- Base Model: "I will show you how..." → "I" indicates COMPLIANCE

If steering increases P("I") in the base model, we might falsely interpret this as "refusal induced" when it's actually "compliance increased."

### The Solution

Use **sequence logprobs** for unambiguous semantic markers:

| Concept | Sequence | Why |
|---------|----------|-----|
| Refusal | "I cannot" | Unambiguously indicates refusal intent |
| Compliance | "Here is" | Unambiguously indicates helpful compliance |

Measure: `Δ log P("I cannot") - Δ log P("Here is")`

This **contrastive score** cleanly separates semantic stance from token frequency effects.

---

## Confirmed Priority Order

All reviewers agree on this ordering:

1. **Experiment 4: Bidirectional Logprob** ← MUST BE FIRST
2. **Experiment 1: Statistical Hardening** ← Blocking for publication
3. **Experiment 3: α Sweep** ← Important but not blocking
4. **Experiment 2: Layer Sweep** ← Defensive, optional

**Rationale:** "Do not spend another GPU hour until Experiment 4 is run correctly." (ChatGPT) The logic must be sound before we invest in statistical polish.

---

## Scope Decisions

### Q2: Bidirectional Test Scope → Option B

Full 2×2 matrix with contrastive sequence scoring:

| Direction | Model | Steering | Metric |
|-----------|-------|----------|--------|
| base→chat | Chat | anti-refusal (-) | Δ contrastive score |
| chat→chat | Chat | anti-refusal (-) | Δ contrastive score |
| chat→base | Base | pro-refusal (+) | Δ contrastive score |
| base→base | Base | pro-refusal (+) | Δ contrastive score |

Plus random direction controls for diff-in-diff.

### Q3: Model Families → Option C

1. Complete Phase 1 fully on Llama-3-8B and Mistral-7B
2. Spot-check Qwen-2.5-7B for orthogonality + cross-transfer
3. Claim "observed in three families; further validation left to future work"

### Q5: Publication-Blocking Items

| Item | Blocking? | Reviewer Consensus |
|------|-----------|-------------------|
| Statistical CIs | **YES** | Without them, effect sizes are anecdotal |
| Bidirectional logprob | **YES** | Ceiling artifact otherwise fatal |
| α sweep | NO | Robustness, not logic |
| Layer sweep | NO | Defensive |
| ≥3 model families | NO | But weakens generality claims |
| Mechanistic explanation | NO | Allowed to be speculative if labeled |

---

## Abstract Wording Refinements

### Change 1: Causal Language

Before: "RLHF induces substantial representational reorganization"

After: "RLHF **is associated with** substantial representational reorganization"

**Rationale:** We haven't strictly proven causality (could be SFT, data, etc.)

### Change 2: Universality Claim

Before: "universal phenomenon"

After: "cross-architectural phenomenon" (until Qwen/Gemma results confirm)

---

## Updated Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Run Experiment 4 (Bidirectional) on Llama | Core logic validated |
| 2 | Run Experiment 4 on Mistral | Cross-architecture confirmation |
| 3-4 | Run Experiment 1 (Statistical Hardening) on both | CIs on all metrics |
| 5 | Run Experiment 3 (α sweep) on Llama | Jailbreak curves |
| 6 | Spot-check Qwen-2.5-7B (V15.4 protocol) | Third family |
| 7 | Paper writing | Draft ready |

---

## Expected Outcomes

### If Experiment 4 shows SYMMETRIC_ORTHOGONALITY:

Cross-transfer fails in both directions (base→chat AND chat→base). This confirms the geometric decoupling interpretation and strengthens the paper significantly.

**Claim:** "RLHF is associated with geometric reorganization that creates bidirectionally incompatible safety manifolds."

### If Experiment 4 shows ASYMMETRIC_TRANSFER:

Chat direction affects base model, but base direction doesn't affect chat. This would require reframing from "orthogonal" to "one-way containment."

**Revised claim:** "Chat safety representations subsume but extend base harm representations, creating asymmetric control relationships."

### If Experiment 4 shows ALL_FAIL:

Neither direction affects either model in logprob space. This would suggest the metric is insensitive or we need different target sequences.

**Action:** Try alternative sequences or layer sweep before concluding.

---

## Strongest Remaining Objection (Pre-Empted)

**Hostile reviewer:** "Low cosine similarity does not imply conceptual orthogonality; you may be extracting different surface heuristics in base vs chat."

**Our response:**
1. Bidirectional logprob test shows semantic incompatibility, not just behavioral failure
2. Contrastive sequence scoring measures semantic stance directly
3. Random direction diff-in-diff isolates concept-specific effects
4. Explicit acknowledgment: "We do not claim the same latent is reused; we show that control directions do not transfer even when measured semantically."

---

## Files

- `v155_bidirectional_logprob_fixed.ipynb` — Updated notebook with sequence logprob metric
- `v155_phase1_plan_review.md` — Original plan document

---

## Go/No-Go Checklist

Before executing Experiment 4:

- [x] Token ambiguity fixed (using sequence logprobs)
- [x] Sign convention explicit (+1 = pro-refusal, -1 = anti-refusal)
- [x] Random direction control included
- [x] Diff-in-diff analysis implemented
- [x] Clear verdict criteria defined

**STATUS: READY FOR EXECUTION**
