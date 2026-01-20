# Future Empathy Geometry Experiments

**Status:** Planned for future research cycles
**Priority:** When looking for next experiments to run

---

## 1. Causal Intervention via Activation Patching

**Research Question:** Are empathy dimensions causally relevant to generating empathetic responses?

**Method:**
- Ablate (zero out) specific empathy dimensions identified by PCA
- Generate responses with ablated activations
- Measure degradation in empathetic quality (AUROC, human ratings)

**Expected Outcome:**
If empathy dimensions are causally relevant, ablating them should produce less empathetic responses. This would validate that the geometric structure we measure actually drives model behavior.

**Implementation Notes:**
- Use activation patching at layer 24
- Test ablating individual dimensions vs. full subspace
- Compare against control (ablating random dimensions)

---

## 2. Layer-wise Bandwidth Profiling

**Research Question:** Does empathy emerge gradually across layers, or concentrate in specific regions?

**Method:**
- Extract steering vectors at each transformer layer (0-32)
- Compute dimensionality and steering range per layer
- Plot bandwidth evolution across depth

**Expected Outcome:**
Two hypotheses:
1. **Gradual emergence:** Bandwidth increases monotonically with depth
2. **Critical layers:** Sharp increases at specific layers (e.g., middle layers where semantic features form)

**Implementation Notes:**
- Run full experiment pipeline for each layer
- Visualization: Heatmap of bandwidth by layer and model
- Could inform optimal steering layer selection

**Applications:**
- If empathy concentrates in layers 18-24, we can apply steering there for maximum effect
- Informs where to look for empathy-specific neurons/circuits

---

## 3. Scaling to Larger Models (70B+)

**Research Question:** Do larger models show higher empathetic bandwidth, or hit diminishing returns?

**Method:**
- Run identical experiment on 70B parameter models:
  - Llama-3.1-70B
  - Qwen2.5-72B
  - DeepSeek-R1-70B (when available)
- Compare bandwidth scaling trends

**Expected Outcomes:**
1. **Linear scaling:** Bandwidth ∝ parameter count
2. **Sublinear scaling:** Diminishing returns (plateaus around 30-40B)
3. **Phase transition:** Sharp jump at specific model sizes

**Implementation Notes:**
- Requires 8xA100 (80GB) or A100 nodes
- May need model parallelism (expensive!)
- Alternative: Use quantized versions (4-bit) on cheaper hardware

**Cost Estimate:**
- 70B models: ~$5-10/hour on A100 instances
- Total: $25-50 for all 5 models

---

## 4. Human Evaluation of Steered Outputs

**Research Question:** Does higher bandwidth correlate with more helpful empathetic responses?

**Method:**
1. Generate responses with different steering coefficients (α = 0, 5, 10, 15, 20)
2. Human raters judge helpfulness and empathy quality (1-5 scale)
3. Correlate bandwidth with human ratings

**Evaluation Protocol:**
- 50 prompts × 5 models × 5 steering levels = 1,250 responses
- 3 raters per response (majority vote)
- Metrics:
  - Empathy rating (1-5)
  - Helpfulness rating (1-5)
  - Coherence check (binary: coherent / incoherent)

**Expected Outcome:**
High-bandwidth models (Gemma2-9B) should produce more helpful responses at higher steering coefficients before coherence collapses.

**Implementation Notes:**
- Use platform like Scale AI or Mechanical Turk
- Cost: ~$0.10-0.20 per rating × 3,750 ratings = $375-750
- Time: 1-2 weeks for data collection

---

## 5. Cross-Context Transfer Validation (Extended)

**Research Question:** Do empathy vectors generalize beyond the tested contexts?

**Method:**
- Extract steering vectors from one domain (e.g., crisis support)
- Apply to 10 diverse new contexts:
  1. Medical diagnosis delivery
  2. Conflict resolution
  3. Bereavement support
  4. Career counseling
  5. Relationship advice
  6. Customer service complaints
  7. Educational tutoring
  8. Legal consultation
  9. Mental health support
  10. Team leadership coaching

**Evaluation:**
- Human ratings: Does steered output match context-appropriate empathy?
- Transfer success rate: % of contexts where steering improves empathy without degrading quality

**Implementation Notes:**
- Need diverse prompt set (10 prompts per context = 100 prompts)
- Generate: 100 prompts × 5 models × 2 conditions (steered/unsteered) = 1,000 responses
- Human evaluation: 1,000 responses × 2 raters = 2,000 ratings

---

## 6. Empathy vs. Toxicity Trade-off

**Research Question:** Does steering toward empathy reduce toxicity, or are they orthogonal?

**Method:**
- Apply empathy steering to adversarial prompts (jailbreaks, toxic requests)
- Measure:
  - Toxicity scores (Perspective API)
  - Refusal rates
  - Empathy in refusals ("I understand this is frustrating, but I can't...")

**Expected Outcome:**
High empathy steering might:
1. Increase polite refusals (safer)
2. Reduce blunt "I can't do that" responses

**Implementation Notes:**
- Use existing adversarial datasets (AdvBench, ToxicChat)
- Compare against baseline safety tuning

---

## 7. Fine-Tuning Empathy Bandwidth

**Research Question:** Can we increase empathetic bandwidth through targeted fine-tuning?

**Method:**
1. Fine-tune base models on empathetic dialogue datasets
2. Measure bandwidth before and after
3. Test if trained empathy generalizes (transfer test)

**Datasets:**
- EmpatheticDialogues (25k conversations)
- Counseling conversations (simulated or real, de-identified)
- Crisis Text Line data (if accessible)

**Expected Outcome:**
Fine-tuning should increase both dimensionality and steering range, validating that bandwidth reflects learned empathy capacity.

---

## 8. Multimodal Empathy (Vision + Language)

**Research Question:** Do vision-language models encode empathy differently when processing images?

**Method:**
- Test multimodal models (LLaVA, GPT-4V equivalent)
- Present images with emotional content (distressed faces, disasters, celebrations)
- Extract steering vectors from image+text activations
- Compare bandwidth: text-only vs. multimodal

**Expected Outcome:**
Multimodal models might show higher bandwidth when visual emotional cues are present.

---

## 9. Adversarial Empathy Attacks

**Research Question:** Can we "oversteer" models into problematic empathetic responses?

**Method:**
- Apply extreme steering coefficients (α > 30)
- Test for:
  - Over-apologizing
  - Inappropriate emotional matching (empathizing with harmful requests)
  - Manipulative language

**Safety Implications:**
Understanding failure modes helps design safer empathy steering systems.

---

## 10. Cross-Lingual Empathy Transfer

**Research Question:** Do empathy vectors transfer across languages?

**Method:**
- Extract vectors from English prompts
- Apply to prompts in Spanish, Chinese, Arabic, French
- Measure transfer success

**Expected Outcome:**
If empathy is a universal concept, vectors should transfer (with degradation). If language-specific, transfer fails.

**Implementation Notes:**
- Requires multilingual models (Llama-3 supports 8 languages)
- Native speaker evaluation for quality

---

## Priority Ranking (Recommended Order)

1. **Human Evaluation (Exp 4)** - Validates that bandwidth matters for real helpfulness
2. **Layer-wise Profiling (Exp 2)** - Low cost, high insight
3. **Causal Intervention (Exp 1)** - Validates causal relevance
4. **Scaling to 70B (Exp 3)** - Tests generalization to frontier models
5. **Extended Transfer (Exp 5)** - Strengthens generalization claims
6. **Toxicity Trade-off (Exp 6)** - Safety implications
7. **Fine-tuning (Exp 7)** - Longer-term, more complex
8. **Multimodal (Exp 8)** - Requires different infrastructure
9. **Adversarial (Exp 9)** - Safety red-teaming
10. **Cross-lingual (Exp 10)** - Specialized evaluation needs

---

**Next Steps:**
When ready for follow-up work, start with Human Evaluation (Exp 4) to validate that the geometric measurements correlate with real-world helpfulness.
