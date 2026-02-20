# V15.2 Reviewer Questions Template

## Prompt to Send with V15.2 Files

---

**Context:** You previously reviewed our V15/V15.1 experimental design for the "Safety Steering Spectrum" paper. A critical blocking issue was identified: the notebook files did not contain the fixes we described in documentation. We have now implemented all changes in executable code. The attached V15.2 notebooks contain verified implementations, not just documentation.

**Specific changes verified in code (with cell references):**

1. **Matched-length block swap:** `v152_anchor_window_test.ipynb` Cell 5 contains `assert len(tokens_preserved) == len(tokens_lost)` and uses position-swapped blocks from identical source text. Weather padding has been completely removed.

2. **Statistical calibration artifacts:** Each notebook saves `null_distributions.npz` containing random-direction effect distributions. Thresholds are computed as percentiles of these distributions via `np.percentile(null_dist, 95)`, not hardcoded values. Functions `permutation_test()`, `bootstrap_ci()`, `wilson_ci()`, and `cohens_d()` are present and executable.

3. **Perplexity as outcome, not filter:** Following Gemini's critique, we no longer reject high-perplexity runs. `v152_enhanced_steering.ipynb` Cell 12 plots Compliance vs Perplexity; the perplexity cost of successful steering is reported as a primary finding. The variable `REJECT_HIGH_PERPLEXITY` is set to `False`.

4. **Layer sweep with onset detection:** `v152_enhanced_steering.ipynb` Cell 10 computes both peak separation layer (max harmful/harmless distance) and onset layer (first layer exceeding 25% of max). Results reported at both, with FDR correction via `fdr_correction()`. Fixed 40% depth is pre-registered as primary analysis in comments.

5. **Strong baseline control:** `v152_enhanced_steering.ipynb` Cell 13 adds concept-steering control (French language direction extracted from "Translate to French" vs "Translate to English" pairs) alongside random-direction control.

6. **Base model coherence metric:** `v152_base_chat_comparison.ipynb` Cell 8 evaluates base model outputs via `evaluate_base_model_coherence()` which uses embedding similarity to detect topical shifts, not refusal regex matching.

7. **Anchor reliance assay:** New `v152_anchor_ablation.ipynb` tests baseline refusal with anchor present vs ablated (same length via swap), without any steering intervention. This mechanizes H1 independent of steering dynamics.

8. **Normalized entropy:** `v152_entropy_auditing.ipynb` Cell 6 computes `H / H_max` where `H_max = log2(vocab_size)`, plus relative entropy drop vs preceding tokens.

9. **Dual temperature:** Entropy auditing runs at both temp=0 (deterministic, for causal claims) and temp=1.0 (sampling, for true entropy measurement).

10. **BOS preservation:** `v152_anchor_window_test.ipynb` Cell 10 contains assertion that BOS token remains in final W tokens for anchor-preserved condition.

---

**Please verify the following:**

### 1. Code Inspection (Blocking)

Do the implementations match our claims? Specifically:

- Does Cell 5 of the anchor test actually implement matched-length block swap with equality assertion?
- Are the statistical functions (`permutation_test`, `bootstrap_ci`, `cohens_d`) correctly implemented?
- Is perplexity truly logged-but-not-filtered in the patching code?
- Does the layer sweep apply FDR correction before selecting layers?

If any implementation is missing or incorrect, please identify the exact cell and what's wrong.

### 2. Perplexity-as-Outcome Approach

Is plotting Compliance vs Perplexity the right visualization? Should we also report:

- The perplexity threshold at which compliance first appears?
- The "perplexity cost" as a scalar (mean perplexity of compliant runs minus mean of refusal runs)?
- Perplexity distributions (not just means) for crystallized vs liquid models?

### 3. Onset Layer Detection

Is 25% of max separation the right threshold for "onset"? Alternatives considered:

- First layer where separation is statistically significant (p < 0.05 vs baseline)
- First layer where separation exceeds 2σ of early-layer variance
- Inflection point of separation curve (second derivative)

Which criterion is most defensible, and why?

### 4. Anchor Ablation Assay

The new `v152_anchor_ablation.ipynb` tests whether baseline refusal depends on anchor visibility without involving steering. Is this design sufficient to mechanize H1, or are there confounds we've introduced? Specifically:

- Does removing anchor tokens change more than just "anchor visibility" (e.g., prompt length, position encodings)?
- Should we also measure attention-to-anchor as a continuous variable rather than binary present/absent?

### 5. Remaining Blockers

Given these verified implementations, are there any issues that would still prevent execution? Please categorize as:

- **Blocking:** Must fix before running any experiments
- **Critical:** Should fix before publication but can run pilots
- **Minor:** Note in limitations section

### 6. Claims Calibration

Given the V15.2 design, which of these claims are now supportable vs still overclaims?

- "RLHF causes thermodynamic crystallization in dense attention models"
- "SWA architecture prevents crystallization via anchor loss"
- "The Crystallized/Liquid spectrum reflects mechanistically distinct regimes"
- "Downstream safety gates restore refusal in crystallized models"
- "Output entropy distinguishes crystallized from liquid models"

For each, indicate: **Supportable**, **Supportable with caveats**, or **Still overclaim**.

---

**Please be specific.** If an implementation is inadequate, provide the exact fix. If a claim is still an overclaim, explain what additional evidence would be needed. We prefer detailed critique now over rejection later.

---

## Expected Reviewer Responses

### If Reviewers Approve (Green Light)

Proceed to execution with the following priority:

1. Run `v152_anchor_ablation.ipynb` on Mistral and Llama (establishes anchor reliance baseline)
2. Run `v152_entropy_auditing.ipynb` on both (validates thermodynamic hypothesis)
3. Run `v152_anchor_window_test.ipynb` on Mistral (tests SWA hypothesis)
4. Run `v152_base_chat_comparison.ipynb` on Llama pair (isolates RLHF)
5. Run `v152_enhanced_steering.ipynb` on all models (full spectrum characterization)

### If Reviewers Identify Issues

Create V15.3 addressing specific concerns before execution. Do not run experiments until code-prose alignment is verified by at least 2/3 reviewers.

### Reviewer-Specific Concerns to Watch

**Grok** may flag: Statistical power for permutation tests, topic clustering in extraction pairs, Bayesian alternatives to frequentist thresholds.

**Gemini** may flag: Practical compute failures (OOM, slow permutations), onset layer threshold justification, base model evaluation validity.

**ChatGPT** may flag: Code-prose mismatches (primary concern), anchor ablation confounds, diff-in-diff implementation correctness.

---

*Template version: 15.2*
*Created: January 12, 2026*
