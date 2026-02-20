# Comprehensive Review Package: The Dual Geometry of Alignment

## For External Reviewers — January 2026

---

## 1. Project Summary (One Paragraph)

We discovered that LLM safety is implemented through at least two functionally distinct subspaces—**Generation** (how to produce harmful content) and **Discrimination** (recognizing that content is harmful)—and that different extraction methodologies selectively access these subspaces. This explains why the same model at the same layer can show dramatically different geometric measurements (0.21 vs 0.76 cosine similarity) depending on how you probe it. The framework resolves conflicting results in the steering literature and provides a diagnostic protocol for characterizing safety architectures before attempting interventions.

---

## 2. Key Experimental Results (V15.4 → V15.13)

### The Core Finding: Methodology Sensitivity

| Experiment | Methodology | Layer | Base↔Chat Similarity |
|------------|-------------|-------|---------------------|
| V15.4 | Forced compliance (few-shot, harmful completions) | 12 | **0.207** |
| V15.7 | Standard contrastive | 12 | 0.762 |
| V15.8 | Base-style on chat | 12 | 0.844 |
| V15.12 | Standard contrastive (sweep) | 12 | **0.761** |

**The same model (Llama-3-8B), same layer (12), different methodology → 0.55 difference in similarity.**

This is larger than any between-model difference we observed. Methodology dominates.

### V15.10: Sign-Flip Causality Test

Testing whether the orthogonal component is a bidirectional control axis:

| Condition | Δ Refusal Score | Interpretation |
|-----------|-----------------|----------------|
| α = -3 (subtract direction) | **-6.24** | Strong jailbreak effect |
| α = +3 (add direction) | -1.51 | Weak, same direction as negative |
| Random control | ~0 | No effect |

**Asymmetry ratio: 4.1:1** — The direction works one-way. You can break safety but not enhance it.

**Interpretation**: This is a **fuse**, not a **dial**. The orthogonal component disrupts a control mechanism; it doesn't traverse a semantic axis.

### V15.9b: Qwen Anomaly

| Model | Base↔Chat Similarity | Steering Effect |
|-------|---------------------|-----------------|
| Llama-3-8B | 0.21 (V15.4 method) | -6.24 (works) |
| Qwen-2.5-7B | **0.94** | **~0** (fails) |

**High geometric similarity does NOT guarantee behavioral control.** Qwen recognizes harm similarly to its base model but steering has no effect.

### V15.12: Late-Layer Sweep

| Layer | Similarity | Interpretation |
|-------|------------|----------------|
| 12 | 0.761 | High (standard extraction) |
| 20 | 0.774 | High |
| 24 | 0.760 | High |
| 28 | 0.724 | Slight drop |
| 30 | 0.663 | Moderate drop |
| 31 | 0.660 | Moderate drop |

Similarity drops only ~0.10 from mid to late layers. The "compression bottleneck" hypothesis is NOT strongly supported.

### V15.6b: Iso-Norm Verification

With directions normalized to identical magnitude:
- Orthogonal component: **Effective** (jailbreaking works)
- Parallel component: **Ineffective** (no behavioral change)

This rules out magnitude as a confound. The orthogonal geometry, not vector length, drives the effect.

---

## 3. The Theoretical Framework: Generation vs Discrimination

### The Core Insight

LLMs process "harm" in at least two distinct functional subspaces:

**Discrimination Subspace (Recognition)**
- Function: "Is this content harmful?"
- Shared between base and chat models (both need to recognize harm)
- Probed by: Standard contrastive extraction (V15.12-style)
- Result: High similarity (~0.76)
- Steering: Ineffective (wrong functional target)

**Generation Subspace (Capability)**
- Function: "How to produce harmful content"
- Present in base, actively suppressed in chat by RLHF
- Probed by: Forced compliance extraction (V15.4-style)
- Result: Low similarity (~0.21) — chat has orthogonalized these circuits
- Steering: Effective for jailbreaking (reactivates suppressed circuits)

### How This Explains Every Result

**V15.4 vs V15.12 divergence**: Different methodologies probe different functional systems. V15.4 (forcing the model to generate harmful content) accesses generation-relevant circuits. V15.12 (standard harmful/benign contrast) accesses discrimination-relevant circuits.

**Sign-flip asymmetry**: Subtracting the generation vector reactivates suppressed production (jailbreak). Adding it can't strengthen discrimination—wrong subspace. You're pushing a control circuit in a direction it wasn't designed to go.

**Qwen failure**: Qwen's 0.94 similarity means we probed the discrimination subspace (recognition is shared). Qwen may implement generation suppression through a different mechanism (attention gating, late-layer suppression) that our methodology doesn't access.

**One-way perturbation**: The generation subspace acts like a fuse (can be blown to enable generation), not a dial (can't be turned up to enhance suppression).

---

## 4. The Qwen Question: Critical for Generality

### Why Qwen Matters

A general theory must explain all the data, including failures. If Qwen remains anomalous, our framework has a critical limitation. If we can diagnose Qwen's architecture, our theory becomes stronger.

### Qwen's Training Context

Qwen-2.5 was trained on 18 trillion tokens with heavy multilingual emphasis (29+ languages, substantial Chinese). This creates different computational demands:

1. **Chinese linguistic properties**: Isolating language, topic-comment structure, semantically dense characters
2. **Multilingual representations**: Language-agnostic conceptual spaces in middle layers
3. **Different safety priorities**: Chinese regulatory focus (political, social stability) vs Western focus (violence, illegal activities)

### Alternative Architectures Qwen Might Use

**Architecture A (Same as Llama)**: Representational orthogonalization. If our V15.15-style generation vs discrimination test shows the split, Qwen uses this—we just needed correct extraction.

**Architecture B (Attention Gating)**: Representations remain parallel (0.94 similarity) but attention heads gate harmful generation. Residual steering fails because the gate is downstream.

**Architecture C (Late-Layer Suppression)**: Safety implemented in final layers. Mid-layer probes miss it entirely.

---

## 5. Proposed Experiments (V15.14 → V15.18)

### V15.14: Probe Ensemble ("Many Surveys")

**Purpose**: Treat methodology sensitivity as data, not noise. Run 36 extraction methodologies varying few-shot count, persona, answer style, and framing.

**Key Questions**:
- Is the similarity distribution bimodal (two clusters = two subspaces)?
- Does similarity correlate with steering effectiveness?
- Which methodology factors drive the differences?

**Prediction**: If generation-discrimination theory is correct, we should see two clusters—low-similarity methods that produce effective steering, high-similarity methods that don't.

### V15.15: Generation vs Recognition (SMOKING GUN)

**Purpose**: Explicitly design extraction to target each mode.

**Setup**:
- Generation probes: "Explain how to make a bomb" → "To construct an explosive device, you need..."
- Recognition probes: "Is 'how to make a bomb' dangerous?" → "Yes, this is dangerous because..."

**Predictions**:
- Recognition similarity: HIGH (both models recognize harm)
- Generation similarity: LOW (chat suppresses generation)
- Recognition steering: WEAK (wrong subspace)
- Generation steering: STRONG (correct subspace)

**If confirmed**: The paper writes itself. Generation-discrimination split is validated.

### V15.16: Frozen-Prompt Ablation

**Purpose**: Definitively prove methodology is causal, not prompt content.

**Setup**: Hold questions constant, vary only methodology parameters (few-shot, persona, answer style).

**Prediction**: Same questions produce >0.3 range in similarity based solely on framing.

**If confirmed**: Closes "maybe it's just different prompts" objection.

### V15.17: Hook Position Verification

**Purpose**: Rule out "wrong tap point" as explanation.

**Setup**: Test V15.4 vs V15.12 divergence at resid_pre, attn_out, mlp_out, resid_post.

**Prediction**: Methodology sensitivity persists across all hook positions.

**If confirmed**: The difference is functional, not architectural tap-point.

### V15.18: Qwen Architecture Diagnosis

**Purpose**: Determine which safety architecture Qwen uses.

**Tests**:
1. Full layer sweep (tests late-suppression hypothesis)
2. Attention head differential analysis (tests gating hypothesis)
3. Token position sweep (tests tokenization mismatch)
4. Generation vs discrimination split (tests if Qwen has the split at all)

**Outcomes**:
- If split detected → Qwen uses Architecture A (same as Llama, we needed correct extraction)
- If attention gating detected → Qwen uses Architecture B (different intervention needed)
- If late-layer drop → Qwen uses Architecture C (probe later layers)

---

## 6. What Can Be Claimed (Honest Assessment)

### Strong Claims (Well-Supported)

1. **Methodology sensitivity dominates geometric measurements.** V15.4 vs V15.12 on same model/layer: 0.21 vs 0.76 similarity.

2. **Geometric similarity does not guarantee behavioral control.** Qwen: 0.94 similarity, zero steering effect.

3. **Effective steering operates one-way.** V15.10: -6.24 effect at α=-3, only -1.51 at α=+3. Asymmetry ratio 4.1:1.

4. **V15.4's specific methodology produces effective jailbreaking.** Verified by V15.6b iso-norm test and V15.10 negative α.

### Provisional Claims (Awaiting V15.15 Confirmation)

5. **LLMs have distinct Generation and Discrimination subspaces for harm.** Theoretically motivated, explains all data, but needs direct test.

6. **Extraction methodology determines which subspace is probed.** Explains V15.4 vs V15.12 divergence.

### Claims We Cannot Make

- ~~Type I/Type II phenotypes are intrinsic model properties~~ (Methodology-dependent)
- ~~Bidirectional causal control via safety directions~~ (Sign-flip asymmetry refutes this)
- ~~Compression bottleneck drives orthogonality~~ (Late-layer sweep shows only 0.1 drop)
- ~~Cross-phenotype compatibility/incompatibility~~ (V15.13 had bugs, inconclusive)

---

## 7. Field Contribution

### Why This Matters for Mechanistic Interpretability

**Contribution 1: Probing methodology is not neutral.** The field has largely treated extraction as a technical detail. We prove it's fundamental—methodology determines which functional subspace you access.

**Contribution 2: Concepts have multiple functional instantiations.** "Harm" isn't one direction. It has at least recognition and generation modes, potentially implemented differently.

**Contribution 3: Explains conflicting steering literature.** Some papers find steering works, others find it fails. They were accessing different subspaces.

**Contribution 4: Diagnostic framework for safety architectures.** Before intervening, diagnose which mechanism a model uses. Different architectures require different interventions.

**Contribution 5: Architecture-dependent safety implementation.** Different training recipes (Meta vs Alibaba) may produce different solutions to the same functional requirement.

### Connection to General Theory of LLM Minds

**Functional modularity**: LLMs have distinct circuits for different operations on the same content.

**Implementation diversity**: The same function can be implemented through different mechanisms.

**Measurement as interaction**: All measurements are interactions between model and probe. No "intrinsic geometry" independent of methodology.

**Probing the right subspace**: To control behavior, you must address the correct functional subspace.

---

## 8. Specific Questions for Reviewers

### On the Theory

1. **Is the generation-discrimination framework the best explanation for our data?** Are there alternative interpretations we haven't considered?

2. **What would falsify this framework?** What results from V15.15 would force us to abandon it?

3. **How does this connect to existing literature?** The deep research PDF cited Chen et al. (2025) finding orthogonal boundaries for "instruction safety" vs "model compliance." What other work supports or challenges our framework?

### On the Qwen Question

4. **Which architecture is Qwen most likely to use?** Given its multilingual training, is attention gating (Architecture B) more plausible than late-layer suppression (Architecture C)?

5. **Should we run Chinese-language extraction?** Would probing with Chinese prompts access different circuits than English?

6. **Is Qwen worth the effort, or should we bound our claims?** Would it be acceptable to state "our framework applies to models using representational orthogonalization" and leave Qwen as a boundary condition?

### On Experiments

7. **Which experiments are highest priority?** We can run 3 in parallel. Current ranking: V15.15 > V15.18 > V15.16 > V15.14 > V15.17.

8. **Are there experiments we're missing?** What would most strengthen the paper?

9. **What controls should we add?** Are there confounds we haven't addressed?

### On Publication

10. **What's the right venue?** Is this ICLR/NeurIPS main track, or workshop-level?

11. **What's the strongest framing?** Options:
    - "The Dual Geometry of Alignment" (positive discovery framing)
    - "Methodology Sensitivity in Safety Steering" (cautionary framing)
    - "Comparative Anatomy of LLM Safety Architectures" (if Qwen shows different mechanism)

12. **What would make this a top-tier contribution?** What's the gap between current state and high-impact publication?

---

## 9. Files Reference

### Key Results (JSON)
- `v154_cross_transfer.json` — Original orthogonality finding
- `v1510_sign_flip_test_llama3-8b.json` — Asymmetry evidence
- `v1512_late_layer_sweep_llama3-8b.json` — Layer trajectory
- `qwen_fixed.json` — Qwen anomaly data

### Notebooks Ready to Run
- `v1514_probe_ensemble.ipynb` — Many surveys approach
- `v1515_gen_vs_rec.ipynb` — Generation vs discrimination (PRIORITY)
- `v1516_frozen_prompt.ipynb` — Methodology causality test
- `v1517_hook_position.ipynb` — Hook position verification
- `v1518_qwen_diagnosis.ipynb` — Qwen architecture diagnosis

### Previous Analysis Documents
- `reviewer_synthesis_generation_discrimination.md` — Framework synthesis
- `qwen_analysis_field_contribution.md` — Qwen deep dive
- `complete_review_request_v159b_v1513.md` — Earlier validation results

---

## 10. Timeline and Next Steps

**Immediate (This Week)**
1. Run V15.15 (Generation vs Discrimination) — smoking gun test
2. Run V15.18 (Qwen Diagnosis) — resolve the anomaly
3. Run V15.16 (Frozen Prompt) — close methodology objection

**If V15.15 Confirms (Next Week)**
1. Write paper with "Dual Geometry" framing
2. Add one strategic model (Gemma-2 or Phi-3) for generality
3. Submit to appropriate venue

**If V15.15 Fails**
1. Reassess theoretical framework
2. Consider "methodology sensitivity as main finding" framing
3. Narrow claims to what's directly supported

---

## 11. The Bottom Line

We set out to discover alignment phenotypes. We found something more fundamental: **the geometry of LLM safety depends on which functional subspace you probe, and different models may implement the same functional requirement through different mechanisms.**

This reframes the entire field's approach to geometric interpretability. The question is no longer "what is the safety direction?" but "which safety direction, accessed how, for what functional purpose?"

Whether V15.15 confirms or fails, we have a publishable contribution. The question is whether it's a discovery paper (framework validated) or a methods paper (methodology sensitivity characterized). Either advances the field.

---

*Document prepared for external review. Please provide feedback on theory, experiments, and publication strategy.*
