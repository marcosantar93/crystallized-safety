# V15.5 Results: The Hierarchical Containment Discovery

## Executive Summary

The V15.5 bidirectional transfer experiment has produced a **major unexpected finding** that fundamentally changes our theoretical framework. Both Llama-3-8B and Mistral-7B show the same pattern: **ASYMMETRIC_CHAT_TO_BASE**.

The relationship between base and chat safety representations is not orthogonal but **hierarchical**. Chat-derived directions successfully induce refusal in base models, but base-derived directions completely fail to jailbreak chat models. This asymmetry is consistent across both dense (Llama) and SWA (Mistral) architectures.

**The revised core claim:** RLHF extends pre-training harm representations by adding policy layers, creating a hierarchical relationship where chat representations *contain* base representations as a subcomponent. This explains why transfer is one-way: chat directions can activate the underlying harm-awareness circuits in base models, but base directions cannot penetrate the policy circuitry that RLHF builds on top.

---

## V15.5 Results Summary

### Llama-3-8B

| Condition | Direction Applied | Steering Goal | Specific Effect | Works? |
|-----------|-------------------|---------------|-----------------|--------|
| Chat + Native | chat→chat | Jailbreak (anti-refusal) | **-4.95** | ✓ YES |
| Chat + Cross | base→chat | Jailbreak (anti-refusal) | +0.40 | ✗ NO |
| Base + Native | base→base | Induce refusal | -0.30 | ✗ NO |
| Base + Cross | chat→base | Induce refusal | **+1.43** | ✓ YES |

**Verdict: ASYMMETRIC_CHAT_TO_BASE**

### Mistral-7B

| Condition | Direction Applied | Steering Goal | Specific Effect | Works? |
|-----------|-------------------|---------------|-----------------|--------|
| Chat + Native | chat→chat | Jailbreak (anti-refusal) | -0.45 | ✗ NO |
| Chat + Cross | base→chat | Jailbreak (anti-refusal) | +1.21 | ✗ NO |
| Base + Native | base→base | Induce refusal | +0.21 | ✗ NO |
| Base + Cross | chat→base | Induce refusal | **+1.49** | ✓ YES |

**Verdict: ASYMMETRIC_CHAT_TO_BASE**

### Key Observations

The pattern is remarkably consistent across architectures. In both model families, the chat direction successfully increases refusal probability in base models by approximately 1.4-1.5 nats (specific effect after diff-in-diff with random control). Meanwhile, the base direction has zero or negative effect on chat models—it actually pushes them slightly *more* toward refusal, the opposite of the intended jailbreak direction.

The native directions show an interesting asymmetry as well. Llama-Chat responds strongly to its native direction (-4.95 specific effect, successfully jailbreaking), but Llama-Base doesn't respond to its native direction (-0.30, essentially noise). Neither Mistral model responds strongly to native directions, consistent with Mistral's lower extraction quality from V15.4.

---

## Theoretical Reframing: From Orthogonality to Containment

### The Original Hypothesis (Falsified)

We initially hypothesized that RLHF creates safety representations that are geometrically orthogonal to pre-training harm representations. The low cosine similarity (~0.2) between base and chat directions seemed to support this, and V15.4's behavioral results (0% cross-transfer) appeared consistent with symmetric incompatibility.

### The New Hypothesis (Supported)

V15.5 reveals that the ~0.2 cosine similarity reflects **partial overlap**, not orthogonality. The relationship is hierarchical:

**Chat direction = Base-compatible component + Policy component**

Where:
- The **base-compatible component** represents harm-topic awareness (dangerous content patterns, risk indicators) that exists in pre-training
- The **policy component** represents the refusal circuitry that RLHF adds on top

This decomposition explains the asymmetric transfer:

| Transfer Direction | Why It Works/Fails |
|--------------------|-------------------|
| chat→base | Chat direction activates the base-compatible component, which exists in base models |
| base→chat | Base direction only contains the harm-topic signal, missing the policy component that controls chat behavior |

### Mechanistic Interpretation

Think of it this way: The base model has learned to recognize dangerous content as a *topic*—patterns of text associated with weapons, drugs, hacking, etc. This is a byproduct of pre-training on internet text. The model doesn't "refuse" because it has no policy; it's just a pattern matcher.

RLHF doesn't erase this harm-topic representation. Instead, it builds a *monitoring layer* that watches for activations in this region and triggers a refusal response. The chat model's "refusal direction" is actually a compound vector that says both "this is dangerous content" AND "I should refuse."

When we apply the chat direction to a base model, we're activating the "this is dangerous content" component. The base model doesn't have a refusal policy, but this activation makes refusal-like tokens (like "I cannot") more probable simply because they're associated with the harm-topic region of activation space.

When we apply the base direction to a chat model, we're only activating the harm-topic signal without the policy trigger. The chat model's refusal circuit is trained to respond to the *combined* signal, not the topic signal alone. It's like trying to unlock a two-factor authentication with only one factor.

---

## Reconciliation with Prior Results

### V15.4 Behavioral Results

V15.4 showed that base→chat cross-transfer produced 0% compliance change behaviorally. This is consistent with V15.5: the base direction doesn't activate the chat model's policy circuitry, so behavior doesn't change even though we're steering in activation space.

V15.4 also showed that both chat models (Llama and Mistral) could be jailbroken with their native directions (45% and 20% compliance respectively). V15.5 confirms this for Llama (-4.95 specific effect in logprob space) but shows weaker effects for Mistral, likely due to extraction quality issues.

### V15.2 Anchor Ablation

The anchor ablation finding (Llama is anchor-independent, Mistral is anchor-reliant) remains valid and orthogonal to the containment finding. Anchor dependence is about *where* safety information is stored (weights vs. context), while containment is about *how* safety representations relate to pre-training representations.

### The ~0.2 Cosine Similarity

This number now has a different interpretation. It's not measuring "orthogonality" but rather the fraction of the chat direction that overlaps with the base direction. The chat direction is ~80% novel (policy component) and ~20% shared (base-compatible component). That 20% is enough to enable one-way transfer.

---

## Implications

### For Interpretability

This finding has significant implications for probe validity. Probes trained on base models may partially capture harm-relevant concepts, but they will miss the policy circuitry that actually controls aligned model behavior. Conversely, probes trained on aligned models may work better than expected on base models because they contain the underlying harm-topic component.

The asymmetry suggests a **methodological prescription**: always extract safety probes from aligned models when the goal is to understand or control aligned model safety. Base-model probes are insufficient.

### For Safety

The hierarchical containment model suggests a potential vulnerability. If RLHF builds policy layers on top of pre-training representations rather than replacing them, then sophisticated attacks might be able to:
1. Identify the underlying harm-topic component
2. Construct adversarial inputs that activate this component without triggering the policy response
3. Bypass safety training while still eliciting harmful content

The fact that base directions don't work for this isn't reassuring—it just means the *obvious* attack fails. More sophisticated component-isolation attacks might succeed.

### For Alignment Theory

The containment model suggests that RLHF is not "rewriting" the model's understanding of harm but "annotating" it with policy responses. This has implications for alignment robustness:

**Pros**: The model retains its pre-training understanding of what's dangerous, which may be more robust and generalizable than policy-specific training.

**Cons**: The safety behavior is an "overlay" that might be more easily removed or bypassed than deeply integrated safety concepts.

---

## Questions for Reviewers

### Q1: Is the Containment Interpretation Correct?

The asymmetric transfer pattern is clear, but there might be alternative explanations:

1. **Containment** (our interpretation): Chat direction includes a base-compatible component
2. **Activation strength**: Chat direction simply has stronger activation magnitude
3. **Polysemantic overlap**: The two directions happen to share some polysemantic features unrelated to safety

How should we distinguish between these? The proposed component decomposition experiment (V15.6) directly tests interpretation 1.

### Q2: How Should We Revise the Paper Framing?

The paper was originally framed around "orthogonal safety manifolds" and "geometric decoupling." This framing is now incorrect. Options:

**Option A**: Reframe entirely around "hierarchical safety representations" and "one-way transfer"

**Option B**: Present the journey—original orthogonality hypothesis, V15.4 evidence, V15.5 revision—as a case study in scientific refinement

**Option C**: Frame around the practical finding (cross-alignment probing has asymmetric validity) without committing strongly to mechanism

### Q3: What Does This Mean for "Crystallization"?

The original liquid/crystallized spectrum was about steering resistance. Llama-Chat showed 0% compliance change under base direction steering—but V15.5 shows this isn't because the model is "crystallized." It's because the base direction doesn't speak the right language.

Should we abandon the crystallization terminology entirely? Or redefine it more precisely as "resistance to simple-component steering"?

### Q4: Is the Component Decomposition Experiment (V15.6) the Right Next Step?

The most direct test of the containment hypothesis is to decompose the chat direction into:
- **Parallel component**: Projection onto base direction
- **Orthogonal component**: Remainder after subtracting parallel component

Then test each component separately. Predictions under containment hypothesis:
- Parallel component transfers to base (it IS the base-compatible component)
- Parallel component fails on chat (it's missing the policy layer)
- Orthogonal component is what actually controls chat behavior
- Orthogonal component has minimal effect on base

Is this the right experiment, or is there a more informative test?

### Q5: How Does This Affect Publication Strategy?

The containment finding is arguably more interesting and novel than symmetric orthogonality would have been. But it requires more careful framing to avoid overclaiming mechanism from correlation.

Is this ready for submission with the containment interpretation as a "proposed explanation"? Or do we need V15.6 results before the story is complete?

---

## Proposed V15.6: Component Decomposition Experiment

To directly test the containment hypothesis, we propose decomposing the chat direction into base-aligned and base-orthogonal components, then testing each separately.

### Protocol

Given:
- `base_dir`: Direction extracted from base model (normalized)
- `chat_dir`: Direction extracted from chat model (normalized)

Compute:
- `parallel_component = (chat_dir · base_dir) * base_dir` (projection onto base subspace)
- `orthogonal_component = chat_dir - parallel_component` (remainder)

Then normalize both components and test all combinations:

| Component | Target Model | Expected Effect (if containment) |
|-----------|--------------|----------------------------------|
| parallel | base | Induces refusal (this IS the shared harm-topic signal) |
| parallel | chat | Fails to jailbreak (missing policy) |
| orthogonal | base | Minimal effect (base doesn't have this circuitry) |
| orthogonal | chat | Jailbreaks (this IS the policy-relevant signal) |

### Falsification Criteria

If containment is correct:
- parallel→base effect ≈ chat→base effect (same underlying mechanism)
- orthogonal→chat effect ≈ native→chat effect (same underlying mechanism)
- parallel→chat ≈ 0 (wrong component)
- orthogonal→base ≈ 0 (wrong component)

If containment is wrong (alternative: coincidental feature overlap):
- Decomposition won't cleanly separate effects
- Both components will have mixed effects on both models

---

## Summary Table: The Journey So Far

| Experiment | Finding | Implication |
|------------|---------|-------------|
| V15.2 | Llama anchor-independent, Mistral anchor-reliant | Safety implementation varies by architecture |
| V15.4 | ~0.2 cosine similarity, 0% behavioral cross-transfer | Base and chat have different safety representations |
| V15.5 | **Asymmetric transfer: chat→base works, base→chat fails** | Relationship is hierarchical, not orthogonal |
| V15.6 (proposed) | Component decomposition | Direct test of containment hypothesis |

---

## The Revised One-Sentence Summary

> **RLHF does not replace pre-training harm representations with orthogonal safety concepts; it extends them by adding policy layers, creating a hierarchical relationship where chat directions can induce refusal in base models but base directions cannot penetrate the policy circuitry of aligned models.**

This is a more nuanced and more accurate story than "orthogonal manifolds," and it has clearer implications for both interpretability methodology and safety vulnerability analysis.

---

## Attached Data

- `bidirectional_sequence_llama3-8b.json` — Full Llama V15.5 results
- `bidirectional_sequence_mistral-7b.json` — Full Mistral V15.5 results
- `bidirectional_sequence_llama3-8b.png` — Llama visualization
- `bidirectional_sequence_mistral-7b.png` — Mistral visualization
