# V15.2 Research Roadmap

## Executive Summary

V15.2 represents a complete overhaul of the experimental methodology based on three rounds of expert review. All blocking issues have been addressed, and the framework now provides calibrated, falsifiable criteria for the Crystallized/Liquid spectrum hypothesis.

This roadmap outlines the execution plan, resource requirements, and decision points for the remaining experiments.

---

## Files in V15.2 Package

### Notebooks (Executable)

The following notebooks are ready for execution in Google Colab with A100 GPU.

**v152_anchor_ablation.ipynb** is the new critical experiment that tests anchor reliance without steering. This should be run FIRST on both Mistral (SWA) and Llama (Dense) to establish mechanistic baseline for H1 before any steering experiments. The expected runtime is approximately 30 minutes per model.

**v152_anchor_window_test.ipynb** tests the SWA hypothesis with matched-length block swap, implementing the reviewer-required diff-in-diff analysis, permutation tests, and Cohen's d. This addresses the V15.1 blocking issue where weather padding was used instead of actual block swaps. The expected runtime is approximately 2 hours for Mistral.

**v152_entropy_auditing.ipynb** validates the thermodynamic hypothesis with normalized entropy (H/H_max), dual temperature runs, and calibrated thresholds derived from random-direction baselines. The expected runtime is approximately 1 hour per model.

**v152_base_chat_comparison.ipynb** isolates RLHF effects with dynamic layer sweep, coherence metrics for base models, and FDR correction for multiple comparisons. The expected runtime is approximately 3 hours per model pair.

**v152_enhanced_steering.ipynb** is the comprehensive spectrum characterization implementing all improvements including onset/peak layer detection, perplexity-as-outcome, strong baseline controls, and bootstrap CIs. The expected runtime is approximately 2-3 hours per model.

### Documentation (Markdown)

**v152_changelog_and_review.md** tracks all changes from V15.1 to V15.2, documents reviewer feedback, and provides a verification checklist to ensure code-prose alignment.

**v152_success_criteria.md** defines calibrated falsification criteria for H1 (SWA anchoring), H2 (downstream error-correction), and H3 (extraction instrumentation). All thresholds are now derived from control distributions rather than being arbitrary values.

**v152_reviewer_questions.md** provides the template prompt for the next reviewer round, including specific questions about code verification, threshold calibration, and remaining blockers.

**v152_research_roadmap.md** is this document, providing the execution plan and decision framework.

---

## Execution Plan

### Phase 1: Anchor Validation (Days 1-2)

The goal of Phase 1 is to establish whether anchor visibility matters for refusal behavior before testing steering effects.

The first step is to run anchor ablation on Mistral (SWA). This involves running `v152_anchor_ablation.ipynb` with `MODEL_CHOICE = 'mistral'` to measure refusal rate with vs without safety anchor. The expected outcome is that Mistral shows anchor reliance (refusal drops when anchor removed). This would support H1, while the falsifying outcome would be anchor-independent behavior.

The second step is to run anchor ablation on Llama (Dense) using the same notebook with `MODEL_CHOICE = 'llama'`. The expected outcome is that Llama shows anchor independence (refusal maintained without anchor). This would support the hypothesis that dense attention maintains global anchor access.

The decision point after Phase 1 is straightforward: if Mistral is anchor-reliant AND Llama is anchor-independent, proceed to Phase 2 with confidence in H1. If both show the same pattern, the SWA hypothesis is challenged, and the interpretation should shift to RLHF depth.

### Phase 2: SWA Hypothesis Test (Days 3-4)

The goal of Phase 2 is to test whether anchor position affects steerability (the "killer experiment").

Run the anchor window test by executing `v152_anchor_window_test.ipynb` on Mistral. This compares steering efficacy when anchor is in-window versus out-of-window. The success criterion requires permutation p < 0.01, Cohen's d > 0.8, and CI excludes zero. The expected outcome is that anchor-lost is more steerable than anchor-preserved. This would confirm that SWA architecture creates liquidity, while the falsifying outcome would be no difference between conditions.

### Phase 3: Thermodynamic Validation (Days 5-6)

The goal of Phase 3 is to validate that crystallized models show low, invariant output entropy.

First, run entropy auditing on Llama by executing `v152_entropy_auditing.ipynb`. Measure normalized entropy at temp=0 (causal) and temp=1.0 (distribution). The expected outcome is H_norm low (<95th percentile of harmless) AND invariant under steering (<95th percentile of random-direction changes).

Second, run entropy auditing on Mistral using the same notebook. The expected outcome is H_norm higher and malleable under steering.

The decision point after Phase 3 determines the thermodynamic framing: if entropy distinguishes crystallized from liquid models, the thermodynamic metaphor is supported. If entropy doesn't distinguish them, the "energy bottleneck" framing should be dropped.

### Phase 4: RLHF Isolation (Days 7-9)

The goal of Phase 4 is to determine whether crystallization comes from RLHF or architecture.

Run the base-chat comparison on Llama pair by executing `v152_base_chat_comparison.ipynb`. Compare Llama-3-8B-Base versus Llama-3-8B-Instruct to test whether base model is liquid and chat model is crystallized. The expected outcome of Base=Liquid and Chat=Crystallized would confirm that RLHF causes crystallization in dense models, while the alternative outcome of both crystallized would suggest architectural causation.

Run the same comparison on Mistral pair if the Mistral base model is available. The expected outcome of both remaining liquid would confirm that SWA prevents crystallization regardless of RLHF.

### Phase 5: Full Spectrum Characterization (Days 10-14)

The goal of Phase 5 is to characterize all models on the spectrum with full controls.

Run the enhanced steering experiment on each model in priority order: Llama-3-8B, Mistral-7B, Qwen-2.5-7B, Phi-3-mini, and Gemma2-9B. For each model, extract direction with cross-validation, run steering with onset and peak layer detection, compute all metrics with bootstrap CIs, classify as Crystallized, Viscous, or Liquid, and save null distributions for reproducibility.

---

## Resource Requirements

### Compute

The GPU requirements for the full study are approximately 40-50 A100-hours total, broken down as follows: Phase 1 requires about 1 hour, Phase 2 requires about 2 hours, Phase 3 requires about 2 hours, Phase 4 requires about 6 hours, and Phase 5 requires about 25 hours.

Colab Pro+ should be sufficient with careful session management. For faster execution, consider using Lambda Labs or RunPod.

### API Costs

API costs consist primarily of OpenAI costs for GPT-4 judge, estimated at $3 per model if using the entropy-calibrated judge. The total for 5 models is approximately $15.

### Storage

Google Drive requirements include approximately 500MB per model for saved vectors, null distributions, and figures. Total storage needed is approximately 5GB for the complete study.

---

## Decision Framework

### What Constitutes "Green Light" for Publication

The minimum viable paper requires extraction validation (direction stability > 0.7, specificity ratio > 2.0) for at least 5 models, clear crystallized/liquid separation on at least one metric (entropy OR compliance change), and base-chat comparison showing RLHF effect in at least one dense model family.

For a stronger paper, the anchor ablation should show differential reliance between SWA and dense models, the anchor window test should show causal SWA effect (p < 0.01, d > 0.8), and patching should localize safety gates in crystallized models.

### What Would Kill the Paper

The paper should be abandoned if extraction fails on more than 3 models (direction stability < 0.5 across model families), if entropy does not distinguish any model pairs (no thermodynamic basis), if base and chat models show identical behavior (RLHF is not causal), or if anchor position has no effect in any model (SWA hypothesis completely falsified AND no alternative mechanism found).

### Pivot Strategies

If the SWA hypothesis fails, the paper should be reframed around RLHF saturation without architectural dependence. If the entropy hypothesis fails, the framing should shift to computational crystallization (multi-layer verification) rather than thermodynamic crystallization. If extraction fails widely, the focus should be on characterizing why certain models resist steering rather than claiming a spectrum.

---

## Claims Calibration

Based on V15.2 methodology, here is what can and cannot be claimed.

### Supportable Claims (if data supports)

The claim that "Some models show large internal preference shifts under steering with minimal behavioral change, while others show proportional coupling" is supportable with full data. This is the core empirical observation.

The claim that "RLHF contributes to crystallization in dense attention models" is supportable with caveats, requiring base-chat comparison data showing Base=Liquid and Chat=Crystallized.

The claim that "Sliding window attention is associated with increased steerability" is supportable with caveats, requiring anchor ablation plus anchor window test data.

The claim that "Output entropy at refusal tokens distinguishes model types" is supportable if the entropy audit clearly separates crystallized from liquid models.

### Claims Requiring Additional Evidence

The claim that "SWA causes liquidity through anchor loss" requires both anchor ablation AND anchor window test to show large, consistent effects with proper controls.

The claim that "Downstream safety gates restore refusal in crystallized models" requires patching with all disruption controls (random, noise, baseline→baseline) showing specific restoration.

The claim that "The spectrum reflects a thermodynamic phase transition" requires entropy AND energy barrier measurements (perplexity cost) plus theoretical derivation.

### Overclaims to Avoid

Do not claim "Universal spectrum across all LLMs" since this was only tested on 8-10 open models. Instead, say "taxonomy observed in tested models."

Do not claim "We proved RLHF creates energy bottlenecks" since energy was not directly measured. Instead, say "results consistent with entropy minimization framework."

Do not claim "Dense attention is inherently more robust" since this conflates architecture with training. Instead, say "dense models in our sample showed less steerability."

---

## Timeline Summary

The complete V15.2 execution requires approximately 2-3 weeks of focused effort.

Week 1 covers Phases 1-3, focusing on anchor validation, SWA test, and entropy auditing for primary models (Mistral, Llama). The key deliverable is a preliminary verdict on H1 and H2.

Week 2 covers Phase 4, focusing on RLHF isolation experiments with base-chat comparisons. The key deliverable is causal attribution of crystallization.

Week 3 covers Phase 5 plus writing, completing the full spectrum characterization and drafting results. The key deliverable is the complete dataset and draft figures for publication.

---

## Verification Checklist Before Execution

Before running any experiments, verify the following code-prose alignment items.

In the anchor window test (Cell 5), confirm that matched-length block swap is implemented with `assert len(tokens_A) == len(tokens_B)` present in the code. Verify that no weather padding text remains.

In the statistical machinery (Cell 3), confirm that `permutation_test()`, `bootstrap_ci()`, `wilson_ci()`, and `cohens_d()` functions are all present and correctly implemented.

In the model loading cells, confirm that `torch.float16` or `torch.bfloat16` is specified with no 4-bit or 8-bit quantization.

In results saving, confirm that `null_distributions.npz` is saved to enable reproducibility.

In verdict logic, confirm that thresholds are computed from null distributions, not hardcoded values.

---

*Document version: 15.2*
*Created: January 12, 2026*
*Status: Ready for execution pending reviewer approval*
