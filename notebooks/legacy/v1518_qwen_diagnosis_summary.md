# V15.18: Qwen Architecture Diagnosis — Experiment Summary

## One-Line Summary

Systematically diagnose which safety architecture Qwen uses to transform the anomaly (0.94 similarity, zero steering effect) from an unexplained failure into explained data that strengthens our theory.

---

## Why Qwen Matters

A general theory of LLM safety must explain all the data, including apparent failures. Qwen shows the opposite pattern from Llama: high geometric similarity (0.94) yet zero steering effectiveness. If we cannot explain Qwen, our framework has a critical limitation.

But if we can diagnose Qwen's architecture, our theory becomes stronger. Instead of "one mechanism that works on some models," we have "multiple mechanisms for the same functional requirement, with diagnostics to identify which one a model uses."

---

## The Alternative Architectures

We hypothesize that all aligned LLMs must solve the same functional problem: separate harm recognition from harm generation. But the implementation can vary.

**Architecture A (Representational Orthogonalization)**: This is what Llama uses. RLHF rotates the generation subspace away from the decision pathway. Base and chat models have orthogonal harm-generation representations. Our V15.4 methodology detects this.

**Architecture B (Attention Gating)**: Representations remain parallel between base and chat (explaining 0.94 similarity), but specific attention heads are trained to block harmful generation pathways. Residual stream steering fails because the gate is downstream of our intervention point.

**Architecture C (Late-Layer Suppression)**: Safety is implemented in the final layers of the network, possibly even in the unembedding matrix. Mid-layer representations are similar because the differentiation happens very late. Our layer-12 extraction misses it entirely.

**Architecture D (Position/Tokenization Mismatch)**: This isn't a different architecture but a methodological failure. Qwen's tokenizer may place semantic content at different positions than Llama's, causing us to extract from the wrong location.

---

## The Four Diagnostic Tests

**Test 1: Full Layer Sweep**

We extract base-chat similarity at layers spanning 25% to 100% of network depth.

If similarity drops dramatically in late layers (below 0.5), Architecture C (late suppression) is indicated.

If similarity stays high throughout, safety is not implemented via representational changes at any depth.

**Test 2: Attention Head Differential Analysis**

We measure attention patterns on harmful vs benign prompts for both base and chat Qwen.

If the chat model shows much larger attention differentials than the base model on the same prompts, specific heads are implementing safety gating (Architecture B).

If attention patterns are similar between base and chat, gating is not the mechanism.

**Test 3: Token Position Sweep**

We extract harmful-benign separation at every token position in the sequence.

If maximum separation occurs at a position other than the last token, our standard extraction has been missing the semantic content.

This tests the methodological failure hypothesis rather than an architectural difference.

**Test 4: Generation vs Discrimination Split on Qwen**

We run the V15.15-style explicit mode separation on Qwen.

If Qwen shows low generation similarity but high discrimination similarity, it actually uses Architecture A—we just needed the correct extraction methodology. This would be the strongest outcome because it validates the framework's universality.

If both modes show high similarity, Qwen genuinely does not use representational orthogonalization.

---

## Decision Tree

The combination of test results determines our diagnosis:

If Test 4 shows the generation-discrimination split → **Architecture A** (same as Llama, methodology was wrong)

If Test 2 shows large attention differentials in chat → **Architecture B** (attention gating)

If Test 1 shows late-layer similarity drop → **Architecture C** (late suppression)

If Test 3 shows position mismatch → **Methodological failure** (fix extraction, retest)

If no test shows clear signal → **Unknown mechanism** (bounds our claims)

---

## Qwen's Training Context

Qwen-2.5 was trained on 18 trillion tokens with heavy multilingual emphasis, including substantial Chinese. This may explain why it differs from Llama.

**Multilingual training creates different representational demands.** To efficiently process 29 languages, Qwen likely develops language-agnostic conceptual spaces. Harm recognition may live in this shared space (explaining high discrimination similarity across base/chat), while harm generation may be routed through language-specific pathways.

**Chinese linguistic properties are different from English.** Chinese is isolating, uses topic-comment structure, and packs semantic density into characters. This may create different compression patterns in the network.

**Alibaba's safety priorities may emphasize different harms.** Chinese regulatory requirements focus on political and social stability concerns rather than the Western focus on violence and illegal activities.

---

## Predictions

**Most likely outcome**: Architecture B (attention gating) or Architecture A (we needed correct extraction).

Multilingual models may prefer gating over rotation because gating allows flexible routing across languages while rotation is a hard geometric constraint.

However, if Test 4 (generation vs discrimination) shows the split on Qwen, the simpler explanation is that we just needed the right methodology—Qwen is fundamentally the same as Llama.

---

## What Each Outcome Means for the Theory

**Architecture A**: The generation-discrimination framework is universal. All models we tested use representational orthogonalization; we just need appropriate methodology to detect it.

**Architecture B**: The framework's functional requirement (separate recognition from generation) is universal, but implementations vary. This is a richer theory with a diagnostic component.

**Architecture C**: Same as B, but the mechanism is depth-based rather than attention-based.

**Unknown**: We bound our claims to "models using representational orthogonalization" and note Qwen as a case requiring future investigation.

Any of these outcomes is publishable. The worst case still advances the field by characterizing when our framework applies.

---

## Runtime and Resources

Approximately 2.5 hours on A100 GPU. This is the most expensive experiment because it includes four separate diagnostic tests across multiple layers/positions.

---

## Questions for Reviewers

1. Which diagnostic test is highest priority? Should we run Test 4 (generation vs discrimination) first since it could immediately resolve the anomaly?

2. If we find attention gating (Architecture B), what intervention would verify it? Can we steer attention patterns directly?

3. Given Qwen's multilingual training, should we also test with Chinese prompts? If generation suppression was primarily trained on Chinese content, English probes might fundamentally miss it.

4. Is it acceptable to bound our claims if Qwen uses an unexplained mechanism? Or does a "general theory" require explaining all architectures we test?

5. Should we also diagnose Gemma-2-9B, which showed similar extraction failure in earlier experiments? One anomaly is a limitation; two anomalies with the same diagnosis would strengthen the architecture-diversity claim.
