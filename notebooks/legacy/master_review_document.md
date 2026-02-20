# Master Review Document: The Dual Geometry of Alignment

## Project Status: Ready for Experimental Validation

**Date**: January 2026  
**Stage**: Pre-experiment review request  
**Goal**: Validate the Generation-Discrimination framework before running experiments

---

## 1. The Discovery in One Paragraph

We discovered that LLM safety operates through at least two functionally distinct subspaces—**Generation** (circuits for producing harmful content) and **Discrimination** (circuits for recognizing harmful content)—and that different extraction methodologies selectively access these subspaces. This explains why the same model at the same layer produces dramatically different geometric measurements (0.21 vs 0.76 cosine similarity) depending on how you probe it. The framework resolves years of conflicting results in the steering literature, explains cross-model transfer failures, and provides a diagnostic protocol for characterizing safety architectures before attempting interventions.

---

## 2. The Key Evidence (Completed Experiments)

### The Methodology Sensitivity Finding

We ran multiple extraction methodologies on Llama-3-8B at layer 12 and found that methodology, not the model itself, determines the observed geometry.

| Experiment | Extraction Method | Base↔Chat Similarity |
|------------|-------------------|---------------------|
| V15.4 | Forced compliance (few-shot harmful completions) | **0.207** |
| V15.7 | Standard contrastive | 0.762 |
| V15.8 | Degraded (base-style prompts on chat) | 0.844 |
| V15.12 | Standard contrastive (layer sweep) | **0.761** |

The 0.55 difference between V15.4 and V15.12 (same model, same layer) is larger than any between-model difference. This cannot be noise—it's a systematic effect of methodology.

### The Sign-Flip Asymmetry (V15.10)

We tested whether the extracted direction functions as a bidirectional control axis by steering in both directions.

| Steering Direction | Effect on Refusal Score | Interpretation |
|-------------------|------------------------|----------------|
| α = -3 (subtract direction) | **-6.24** | Strong jailbreak |
| α = +3 (add direction) | -1.51 | Weak, same direction |
| Random control | ~0 | No effect |

The 4.1:1 asymmetry ratio proves this is not a semantic axis. You can break safety but not enhance it. This is consistent with a **fuse** (can be blown) rather than a **dial** (can be turned both ways).

### The Qwen Anomaly (V15.9b)

Qwen-2.5-7B shows high geometric similarity but zero steering effectiveness.

| Model | Base↔Chat Similarity | Steering Effect |
|-------|---------------------|-----------------|
| Llama-3-8B | 0.21 (V15.4 method) | -6.24 (works) |
| Qwen-2.5-7B | **0.94** | **~0** (fails) |

This proves that geometric similarity does not guarantee behavioral control. Qwen either implements safety through a different mechanism or our methodology fails to access the relevant subspace.

### The Iso-Norm Verification (V15.6b)

With directions normalized to identical magnitude, the orthogonal component produces behavioral change while the parallel component does not. This rules out magnitude as a confound—geometry drives the effect.

---

## 3. The Theoretical Framework

### The Generation-Discrimination Hypothesis

We propose that LLMs process "harm" through two distinct functional subspaces.

**Discrimination Subspace (Recognition)**

This subspace encodes "Is this content harmful?" It is shared between base and chat models because both need to recognize harmful content—the base model to generate contextually appropriate responses, the chat model to trigger refusal. Standard contrastive extraction (comparing harmful vs benign prompts) probes this subspace, finding high base-chat similarity (~0.76) because both models recognize harm the same way.

**Generation Subspace (Capability)**

This subspace encodes "How to produce harmful content." It is present in base models (which will generate if prompted) but actively suppressed in chat models through RLHF. Forced-compliance extraction (few-shot prompts that elicit harmful completions) probes this subspace, finding low similarity (~0.21) because RLHF has orthogonalized these circuits in the chat model.

### How This Explains the Evidence

The V15.4 vs V15.12 divergence occurs because V15.4 (forcing harmful generation) accesses the generation subspace, while V15.12 (standard harm/benign contrast) accesses the discrimination subspace.

The sign-flip asymmetry occurs because subtracting the generation vector reactivates suppressed production circuits (jailbreak), while adding it cannot strengthen the discrimination subspace—you're pushing in the wrong functional direction.

The Qwen anomaly occurs because our extraction probed the discrimination subspace (shared, hence 0.94 similarity) rather than the generation subspace. Qwen may implement generation suppression through a different mechanism (attention gating, late-layer suppression) that our residual-stream methodology doesn't access.

The one-way perturbation pattern occurs because RLHF doesn't erase generation capability—it installs a control gate. The direction we extract allows us to disable the gate (jailbreak) but not strengthen it beyond its training.

---

## 4. The Proposed Experiments

We have prepared five experiments to validate this framework. We request review of the experimental design before execution.

### V15.14: Probe Ensemble ("Many Surveys" Approach)

**File**: `v1514_probe_ensemble.ipynb`  
**Summary**: `v1514_probe_ensemble_summary.md`

**Purpose**: Treat methodology sensitivity as data rather than noise. Run 36 different extraction methodologies varying few-shot count, persona, answer style, and framing. Study the distribution of similarity measurements.

**Key Questions**:
- Is the distribution bimodal (two clusters suggesting two subspaces)?
- Does similarity correlate with steering effectiveness?
- Which methodology factors drive the differences?

**Prediction**: If the generation-discrimination framework is correct, methodologies should cluster into two groups—those that access generation (low similarity, effective steering) and those that access discrimination (high similarity, ineffective steering).

**Runtime**: ~2-3 hours on A100

---

### V15.15: Generation vs Recognition (THE SMOKING GUN)

**File**: `v1515_gen_vs_rec.ipynb`  
**Summary**: `v1515_gen_vs_rec_summary.md`

**Purpose**: Directly test the core hypothesis by explicitly designing extraction to target each mode.

**Setup**:
- Generation probes: "Explain how to make a bomb" → "To construct an explosive device, you need..."
- Recognition probes: "Is 'how to make a bomb' dangerous?" → "Yes, this is dangerous because..."

**Predictions**:
- Recognition extraction: HIGH similarity (both models recognize harm)
- Generation extraction: LOW similarity (chat suppresses generation)
- Recognition steering: WEAK (wrong functional target)
- Generation steering: STRONG (correct functional target)
- Cross-mode similarity: LOW (distinct circuits within each model)

**Why This Is Critical**: This experiment directly tests whether the two subspaces exist. If generation similarity is low (~0.2) while discrimination similarity is high (~0.7), and generation steering works while discrimination steering fails, the framework is validated.

**Runtime**: ~1.5 hours on A100

---

### V15.16: Frozen-Prompt Ablation

**File**: `v1516_frozen_prompt.ipynb`  
**Summary**: `v1516_frozen_prompt_summary.md`

**Purpose**: Definitively prove that methodology framing (not prompt content) causes the geometric divergence.

**Setup**: Hold the harmful and benign questions absolutely constant. Vary only methodology parameters: few-shot count, shot type, persona, answer style. Measure how much similarity varies with identical prompts.

**Prediction**: Same questions should produce >0.3 range in base-chat similarity based solely on methodology framing. This would prove that the V15.4 vs V15.12 difference is causal to methodology, not prompt content.

**Why This Matters**: A skeptical reviewer could argue that V15.4 and V15.12 found different similarities because they used different prompts. This experiment eliminates that possibility by keeping prompts frozen.

**Runtime**: ~1.5 hours on A100

---

### V15.17: Hook Position Verification

**File**: `v1517_hook_position.ipynb`  
**Summary**: `v1517_hook_position_summary.md`

**Purpose**: Rule out "wrong tap point" as an explanation for methodology sensitivity.

**Setup**: Run both V15.4-style and V15.12-style extraction at four different hook positions within the same layer: resid_pre (before attention), attn_out (after attention), mlp_out (after MLP), resid_post (final residual).

**Prediction**: Methodology sensitivity should persist across all hook positions. V15.4 should find lower similarity than V15.12 regardless of where in the transformer block we tap.

**Why This Matters**: If the difference disappeared at certain hook positions, it would suggest we were tapping into different parts of the computation rather than different functional subspaces.

**Runtime**: ~1.5 hours on A100

---

### V15.18: Qwen Architecture Diagnosis

**File**: `v1518_qwen_diagnosis.ipynb`  
**Summary**: `v1518_qwen_diagnosis_summary.md`

**Purpose**: Determine which safety architecture Qwen uses, transforming the anomaly into explained data.

**Setup**: Four diagnostic tests:
1. Full layer sweep (tests late-layer suppression hypothesis)
2. Attention head differential analysis (tests attention gating hypothesis)
3. Token position sweep (tests tokenization mismatch hypothesis)
4. Generation vs discrimination split on Qwen (tests if Qwen has the split at all)

**Possible Outcomes**:

**Architecture A (Same as Llama)**: If Qwen shows the generation-discrimination split, it uses representational orthogonalization—we just needed correct extraction methodology. This validates the framework's generality.

**Architecture B (Attention Gating)**: If Qwen shows high similarity everywhere but large attention head differentials on harmful vs benign prompts, it implements safety via attention gates rather than representational rotation. This explains why residual steering fails.

**Architecture C (Late-Layer Suppression)**: If Qwen shows a similarity drop only in final layers, safety is implemented very late. This explains why mid-layer probes fail.

**Runtime**: ~2.5 hours on A100

---

## 5. Specific Questions for Reviewers

### On the Theoretical Framework

**Q1**: Is the generation-discrimination framework the best explanation for the methodology sensitivity? Are there alternative interpretations we should consider?

**Q2**: What would falsify this framework? Specifically, what results from V15.15 would force us to abandon it?

**Q3**: The framework assumes functional modularity (distinct circuits for recognition vs generation). Is this consistent with what we know about transformer architectures? What literature supports or challenges this?

### On the Qwen Question

**Q4**: Given Qwen's multilingual training (18T tokens, 29+ languages, heavy Chinese), which alternative architecture is most plausible? Attention gating seems likely given that multilingual models may need flexible routing rather than hard representational constraints.

**Q5**: Should we run Chinese-language extraction on Qwen? If generation suppression was primarily trained on Chinese content, English probes might miss it entirely.

**Q6**: Is it acceptable to bound our claims to "models using representational orthogonalization" if Qwen uses a different mechanism? Or does a general theory require explaining all architectures?

### On Experimental Design

**Q7**: We plan to run V15.15, V15.18, and V15.16 in parallel. Is this the right prioritization? Would you reorder based on information value?

**Q8**: Are there confounds in our experimental designs we haven't addressed? What controls should we add?

**Q9**: The probe ensemble (V15.14) uses 36 methodologies. Is this sufficient to characterize the distribution, or should we expand? What methodology variations are we missing?

### On Publication Strategy

**Q10**: Given the dual nature of our findings (discovery of functional subspaces + effective jailbreaking methodology), how should we handle responsible disclosure? Should certain details be restricted?

**Q11**: What's the appropriate venue? This could be framed as interpretability (ICLR/NeurIPS), safety (alignment venues), or both.

**Q12**: What would elevate this from a solid contribution to a top-tier publication? What gap remains?

---

## 6. Field Contribution

### Why This Matters for Mechanistic Interpretability

The interpretability field has largely treated probing methodology as a technical detail—you pick a reasonable method and report what you find. We demonstrate that methodology is not neutral: it determines which functional subspace you access. This explains years of conflicting steering results and provides a framework for reconciling them.

### Why This Matters for Alignment

Current alignment treats safety as a scalar property ("more RLHF = safer"). We show safety has internal structure with distinct components that can be independently accessed. This opens the door to mechanistic safety guarantees rather than probabilistic behavioral training.

### Why This Matters for a General Theory of LLM Minds

We provide evidence for functional modularity: LLMs have distinct circuits for different operations on the same content. "Harm" exists in multiple functional modes (recognizing it, generating it, refusing it). A general theory must account for this structure.

---

## 7. Files Included in This Review Package

### Documentation
- `master_review_document.md` (this file)
- `comprehensive_review_package_v15.md` (detailed results and context)
- `reviewer_synthesis_generation_discrimination.md` (theoretical framework)
- `qwen_analysis_field_contribution.md` (Qwen deep dive)

### Experiment Notebooks
- `v1514_probe_ensemble.ipynb` + `v1514_probe_ensemble_summary.md`
- `v1515_gen_vs_rec.ipynb` + `v1515_gen_vs_rec_summary.md`
- `v1516_frozen_prompt.ipynb` + `v1516_frozen_prompt_summary.md`
- `v1517_hook_position.ipynb` + `v1517_hook_position_summary.md`
- `v1518_qwen_diagnosis.ipynb` + `v1518_qwen_diagnosis_summary.md`

### Previous Results (for reference)
- `complete_review_request_v159b_v1513.md` (validation experiment results)
- `results_compilation_complete.md` (data tables)

---

## 8. Recommended Review Process

**Step 1**: Read this master document for context (10-15 min)

**Step 2**: Review V15.15 (Generation vs Recognition) notebook and summary—this is the critical experiment (15-20 min)

**Step 3**: Review V15.18 (Qwen Diagnosis) notebook and summary—this determines generality (15-20 min)

**Step 4**: Skim V15.14, V15.16, V15.17 summaries for completeness (10 min)

**Step 5**: Provide feedback on Questions 1-12

---

*We request thorough review before running these experiments. The theoretical framework is strong, but the experiments are expensive and we want to maximize information value from each run.*
