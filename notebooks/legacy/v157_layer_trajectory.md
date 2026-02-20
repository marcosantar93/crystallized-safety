# V15.7: Layer-wise Orthogonality Trajectory

## Purpose

V15.4-V15.6 established that base and chat harm/safety directions are nearly orthogonal (~0.17-0.21 cosine similarity) at layer 12. But this measurement at a single layer leaves a critical question unanswered: **where does this orthogonality emerge?**

This experiment tracks cosine similarity between base and chat directions at every layer in the network, producing a "trajectory" that reveals when and how RLHF's geometric transformation occurs.

## Scientific Value

A single-layer measurement tells us *that* base and chat directions differ. A layer-wise trajectory tells us *where* and *how* that difference arises. This converts a descriptive geometric finding into a mechanistically grounded causal claim.

If the trajectory shows divergence concentrated in specific layers, we can identify where RLHF "acts" in the network. This has implications for targeted interpretability, safety interventions, and understanding the alignment process itself.

## Method

The experiment extracts harm/safety directions at each layer (1-32 for Llama-3-8B) using the same contrastive methodology as V15.4, then computes pairwise cosine similarity between base and chat directions at corresponding layers.

For each layer, we also compute extraction quality (separation score) to ensure the trajectory isn't confounded by poor direction quality at certain layers.

## Predictions

Under the supersession interpretation from V15.6, we expect a **gradual divergence** pattern:

**Early layers (1-8):** Higher similarity, perhaps 0.4-0.6. These layers process low-level features (token embeddings, basic syntax) that are likely shared between base and chat models regardless of alignment training.

**Mid layers (8-20):** Rapid divergence toward lower similarity. This is where we expect RLHF's transformation to occur, as mid-layers typically encode more abstract, task-relevant representations.

**Late layers (20-32):** Maximal orthogonality, reaching the ~0.2 similarity observed at layer 12. These layers encode task-specific behaviors, and alignment training should maximally affect them.

An alternative pattern would be a **phase transition**: similarity remains high until a specific layer, then drops sharply. This would indicate a more localized transformation, suggesting particular layers are critical for alignment.

A third possibility is **uniform low similarity**: base and chat directions are orthogonal from the earliest layers. This would suggest the divergence originates from differences in tokenization or embeddings rather than mid-network transformations.

## Interpretation Guide

**GRADUAL_DIVERGENCE:** RLHF transforms representations progressively through the network. The supersession interpretation is supported—RLHF "discovers" better features through cumulative modifications across layers, not by adding a single policy module at the output.

**PHASE_TRANSITION:** RLHF acts at specific layers, creating an abrupt geometric reorganization. This would localize where safety-relevant transformations occur and could inform targeted interventions.

**PERSISTENT_SIMILARITY:** If similarity remains above 0.3 even in late layers, the V15.4-V15.6 orthogonality may be specific to layer 12 rather than a global property. Would need to investigate why that layer shows divergence while others don't.

**UNIFORM_LOW:** If similarity is low from the earliest layers, the divergence may originate from tokenizer differences or embedding-layer changes, not mid-network transformations.

## Expected Output

The notebook produces a JSON file with similarity values at each layer, a figure showing the trajectory with phase annotations, and an automated pattern classification. The key deliverable is a curve showing how base↔chat similarity evolves from layer 1 to layer 32.

## Implications for the Paper

If the trajectory shows gradual divergence or a phase transition, we can add a figure demonstrating "RLHF-induced geometric transformation is localized to layers X-Y" or "emerges progressively through mid-layers." This strengthens the mechanistic story significantly.

If the pattern is unexpected (persistent similarity or uniform low), we learn something important about the limitations of the layer-12 analysis and may need to revise interpretations.

## Runtime

Approximately 1.5 hours on A100 (extracting directions at 32 layers for both models).
