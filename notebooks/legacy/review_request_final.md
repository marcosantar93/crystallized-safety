# Review Request: Experimental Validation of the Dual Geometry Framework

## For External Reviewers — Pre-Execution Review

**Date**: January 2026  
**Request**: Please review the experimental designs (notebooks + summaries) and provide feedback before we execute them.

---

## Quick Context (2 minutes)

We discovered that LLM safety operates through at least two functionally distinct subspaces: **Generation** (circuits for producing harmful content) and **Discrimination** (circuits for recognizing harmful content). Different extraction methodologies selectively access these subspaces, which explains why the same model at the same layer produces dramatically different geometric measurements (0.21 vs 0.76 cosine similarity) depending on how you probe it.

The key evidence:
- V15.4 (forced compliance extraction): 0.207 base-chat similarity, effective jailbreaking
- V15.12 (standard contrastive extraction): 0.761 base-chat similarity, ineffective steering
- V15.10 (sign-flip test): 4.1:1 asymmetry ratio—you can break safety but not enhance it
- Qwen anomaly: 0.94 similarity but zero steering effect—high geometry doesn't guarantee control

We have designed five experiments to validate this framework. We request review of the experimental designs before execution.

---

## Documents Included

### Master Overview
| Document | Purpose | Read Time |
|----------|---------|-----------|
| `master_review_document.md` | Complete context, all results, 12 specific questions | 15 min |
| `comprehensive_review_package_v15.md` | Detailed data tables, claim assessment | 10 min |

### Experiment Summaries (Read These for Design Review)
| Experiment | Summary File | Notebook | Priority |
|------------|--------------|----------|----------|
| V15.15 Generation vs Recognition | `v1515_gen_vs_rec_summary.md` | `v1515_gen_vs_rec.ipynb` | **CRITICAL** |
| V15.18 Qwen Diagnosis | `v1518_qwen_diagnosis_summary.md` | `v1518_qwen_diagnosis.ipynb` | HIGH |
| V15.16 Frozen Prompt | `v1516_frozen_prompt_summary.md` | `v1516_frozen_prompt.ipynb` | HIGH |
| V15.17 Hook Position | `v1517_hook_position_summary.md` | `v1517_hook_position.ipynb` | MEDIUM |
| V15.14 Probe Ensemble | `v1514_probe_ensemble_summary.md` | `v1514_probe_ensemble.ipynb` | MEDIUM |

### Supporting Context
| Document | Content |
|----------|---------|
| `reviewer_synthesis_generation_discrimination.md` | Theoretical framework synthesis |
| `qwen_analysis_field_contribution.md` | Deep analysis of Qwen anomaly |

---

## What We Need From Reviewers

### Priority 1: V15.15 Design Review

This is the smoking gun experiment. Please review `v1515_gen_vs_rec_summary.md` and the notebook, then answer:

1. Are our generation prompts sufficiently distinct from discrimination prompts? Is there risk they access the same circuits despite different surface framing?

2. Should we add a "hybrid" condition (e.g., "This is dangerous, but here's how...") to test cross-talk between modes?

3. The discrimination prompts involve explicit meta-discussion of harm ("Is this dangerous?"). Could this access different circuits than implicit recognition during standard harmful prompts? How do we control for this?

4. What results would falsify the framework? We predict generation similarity ~0.2, discrimination similarity ~0.7. What if we get 0.4 and 0.6? How do we interpret moderate rather than clean splits?

### Priority 2: V15.18 Design Review

This determines whether Qwen uses a different safety architecture. Please review `v1518_qwen_diagnosis_summary.md` and answer:

5. Given Qwen's multilingual training (18T tokens, heavy Chinese), should we run Chinese-language prompts alongside English? If generation suppression was primarily trained on Chinese content, English probes might fundamentally miss it.

6. The notebook tests four hypotheses (layer sweep, attention gating, position mismatch, generation-discrimination split). Is this the right diagnostic order? Should we start with the generation-discrimination test since it could immediately resolve the anomaly?

7. If Qwen uses attention gating (Architecture B), what intervention would verify it? Can we steer attention patterns directly, or would we need a different approach?

### Priority 3: Methodology Controls

Please review `v1516_frozen_prompt_summary.md` and `v1517_hook_position_summary.md`:

8. For V15.16 (frozen prompts): We hold questions constant and vary only methodology framing. Is 6 harmful + 6 benign questions sufficient, or should we expand to 12+12 for statistical power?

9. For V15.17 (hook positions): If methodology sensitivity is stronger at certain hook positions (e.g., mlp_out vs attn_out), how do we interpret that? Does it tell us where the generation-discrimination split is implemented?

### Priority 4: Ensemble and Statistics

Please review `v1514_probe_ensemble_summary.md`:

10. We run 36 extraction methodologies. Is this sufficient to characterize the distribution? Should we add temperature variation (0.0, 0.5, 1.0) to test deterministic vs stochastic sampling?

11. If the similarity distribution is continuous rather than bimodal, what does that imply for the framework? Would it mean the subspaces overlap rather than being discrete?

### Priority 5: Publication Strategy

12. Given the dual nature of our findings (discovery of functional subspaces + effective jailbreaking methodology), how should we handle responsible disclosure? Should certain extraction details be restricted?

---

## Predictions We're Testing

For V15.15 (Generation vs Recognition):

| Mode | Predicted Similarity | Predicted Steering | Rationale |
|------|---------------------|-------------------|-----------|
| Generation | LOW (~0.2-0.4) | EFFECTIVE | RLHF orthogonalized generation subspace |
| Discrimination | HIGH (~0.7-0.9) | WEAK | Both models recognize harm similarly |
| Cross-mode | LOW (~0.3-0.5) | N/A | Distinct functional circuits |

For V15.18 (Qwen Diagnosis):

| Architecture | Layer Sweep | Attention Diff | Gen-Disc Split |
|--------------|-------------|----------------|----------------|
| A (Same as Llama) | High throughout | Similar base/chat | Split visible |
| B (Attention Gating) | High throughout | Chat >> Base | No split |
| C (Late Suppression) | Drop in final layers | Similar | No split |

---

## Falsification Criteria

The framework would be falsified if:

1. **Both modes show similar similarity** (within 0.1 of each other). This would mean extraction doesn't access distinct subspaces.

2. **Steering effectiveness doesn't correlate with similarity**. If high-similarity directions work as well as low-similarity directions, geometry doesn't predict function.

3. **Methodology effect disappears with frozen prompts** (V15.16 range < 0.15). This would mean the V15.4 vs V15.12 difference was due to prompt content, not methodology.

4. **Hook position explains the variance** (V15.17 shows different similarity rankings at different positions). This would mean we're tapping different computations, not different subspaces.

---

## Resource Estimate

| Experiment | GPU Time (A100) | Priority |
|------------|-----------------|----------|
| V15.15 | ~1.5 hours | Run first |
| V15.16 | ~1.5 hours | Run second |
| V15.17 | ~1.5 hours | Run third |
| V15.18 | ~2.5 hours | Run fourth |
| V15.14 | ~3.0 hours | Run last |
| **Total** | **~10 hours** | |

---

## Suggested Review Process

**Step 1** (10 min): Read this request document for context

**Step 2** (15 min): Read `v1515_gen_vs_rec_summary.md` carefully—this is the critical experiment

**Step 3** (10 min): Read `v1518_qwen_diagnosis_summary.md`—this determines generality

**Step 4** (10 min): Skim remaining summaries (V15.14, V15.16, V15.17)

**Step 5** (15 min): Review one notebook of your choice for code quality

**Step 6**: Provide feedback on Questions 1-12

---

## What Success Looks Like

If V15.15 confirms predictions:
- Generation similarity LOW, discrimination similarity HIGH
- Generation steering EFFECTIVE, discrimination steering WEAK
- Cross-mode similarity LOW

We write a discovery paper: **"The Dual Geometry of Alignment: Dissociating Generation and Discrimination Subspaces in LLM Safety"**

If V15.18 diagnoses Qwen:
- We explain the anomaly as "different safety architecture"
- The paper becomes comparative anatomy with diagnostic framework

Combined contribution:
- Explains methodology sensitivity (not noise, but subspace access)
- Explains cross-model transfer failures (different architectures)
- Provides diagnostic protocol for safety research
- Advances mechanistic interpretability beyond "find the concept vector"

---

*We appreciate thorough review before execution. The experiments are expensive and we want to maximize information value from each run.*
