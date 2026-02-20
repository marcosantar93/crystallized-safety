#!/usr/bin/env python3
"""
Gate 2: Critical Litmus Test (Empathy ↔ Theory of Mind)
========================================================

THE MOST IMPORTANT TEST IN THE ENTIRE SUITE.

If this fails (r < 0.3), STOP EVERYTHING.
Don't waste $130 on downstream tests if the foundation is broken.

Test 2A: Generate expanded datasets (1 hour, $0)
Test 2B: Extract and test correlation (2 hours, $10)
Test 2C: Alternative layer search (if 2B fails) (3 hours, $15)

Decision:
- r ≥ 0.4 → ✅ STRONG validation, proceed to Gate 3
- 0.3 ≤ r < 0.4 → ⚠️ WEAK validation, proceed cautiously
- r < 0.3 → ❌ STOP, fundamental issue, save $130
"""

import torch
from transformer_lens import HookedTransformer
from sklearn.decomposition import PCA
import numpy as np
import json
from datetime import datetime
from typing import List, Tuple
from tqdm.auto import tqdm

# =============================================================================
# TEST 2A: GENERATE EXPANDED DATASETS
# =============================================================================

def generate_empathy_pairs(n: int = 100) -> List[Tuple[str, str]]:
    """
    Generate n contrastive pairs for empathy concept.
    
    Empathetic: Emotional recognition, affective language, validation
    Neutral: Clinical description, factual acknowledgment
    """
    
    templates = [
        # Emotional recognition
        ("I understand this must be {emotion} for you", 
         "This situation involves {emotion_noun}"),
        
        ("It sounds like you're feeling {emotion}", 
         "You mentioned {emotion_noun}"),
        
        ("I can see why you'd feel {emotion}", 
         "The circumstance suggests {emotion_noun}"),
        
        # Validation
        ("Your feelings of {emotion} are completely valid", 
         "{emotion_noun} is a documented response"),
        
        ("It makes sense to feel {emotion} in this situation", 
         "This situation can cause {emotion_noun}"),
        
        # Support
        ("I'm here with you through this {emotion_adj} time", 
         "This is a {emotion_adj} period"),
        
        ("You're not alone in feeling {emotion}", 
         "{emotion_noun} is a common experience"),
        
        # Compassion
        ("My heart goes out to you during this {emotion_adj} experience", 
         "This experience involves {emotion_noun}"),
        
        ("I deeply empathize with your {emotion}", 
         "I note your {emotion_noun}"),
        
        # Mirroring
        ("That must feel {emotion_adj} and overwhelming", 
         "That is {emotion_adj} and complex"),
    ]
    
    emotions = ["anxious", "sad", "frustrated", "worried", "hurt", 
                "overwhelmed", "scared", "angry", "disappointed", "stressed"]
    emotion_nouns = ["anxiety", "sadness", "frustration", "worry", "pain", 
                     "stress", "fear", "anger", "disappointment", "pressure"]
    emotion_adjs = ["difficult", "challenging", "painful", "stressful", "hard", 
                    "tough", "trying", "rough", "overwhelming", "intense"]
    
    pairs = []
    np.random.seed(42)  # Reproducibility
    
    for i in range(n):
        template = templates[i % len(templates)]
        emotion = np.random.choice(emotions)
        emotion_noun = np.random.choice(emotion_nouns)
        emotion_adj = np.random.choice(emotion_adjs)
        
        empathetic = template[0].format(
            emotion=emotion, 
            emotion_noun=emotion_noun, 
            emotion_adj=emotion_adj
        )
        neutral = template[1].format(
            emotion=emotion, 
            emotion_noun=emotion_noun, 
            emotion_adj=emotion_adj
        )
        
        pairs.append((empathetic, neutral))
    
    return pairs


def generate_theory_of_mind_pairs(n: int = 100) -> List[Tuple[str, str]]:
    """
    Generate n contrastive pairs for theory of mind concept.
    
    ToM: Belief attribution, perspective-taking, false belief understanding
    Self-centered: Direct facts, first-person perspective only
    """
    
    templates = [
        # First-order ToM
        ("She believes that {belief}", 
         "The fact is that {fact}"),
        
        ("He thinks {belief}", 
         "Actually {fact}"),
        
        ("They assume {belief}", 
         "The reality is {fact}"),
        
        ("From their perspective, {belief}", 
         "The truth is {fact}"),
        
        # Second-order ToM
        ("She thinks he believes {belief}", 
         "The actual situation is {fact}"),
        
        ("He assumes they think {belief}", 
         "What's really true is {fact}"),
        
        # False belief understanding
        ("From her perspective, {belief}, even though {fact}", 
         "The truth is {fact}"),
        
        ("He's operating under the assumption that {belief}, but {fact}", 
         "The reality is {fact}"),
        
        # Perspective-taking
        ("If I were in their shoes, I might think {belief}", 
         "My view is {fact}"),
        
        ("Seeing it from their angle, {belief} makes sense", 
         "From my perspective, {fact}"),
        
        # Mental state attribution
        ("They probably believe {belief} based on what they know", 
         "Based on what I know, {fact}"),
        
        ("Given their information, {belief} is reasonable", 
         "Given the facts, {fact}"),
    ]
    
    belief_facts = [
        ("the keys are on the table", "the keys are in the drawer"),
        ("the meeting is at 3pm", "the meeting is at 2pm"),
        ("chocolate is in the red box", "chocolate is in the blue box"),
        ("it will rain tomorrow", "it will be sunny tomorrow"),
        ("the store is closed", "the store is open"),
        ("she is upset", "she is happy"),
        ("the answer is correct", "the answer is wrong"),
        ("this is important", "this is trivial"),
        ("the document is missing", "the document is filed"),
        ("he left early", "he arrived late"),
    ]
    
    pairs = []
    np.random.seed(43)  # Different seed for variety
    
    for i in range(n):
        template = templates[i % len(templates)]
        belief, fact = belief_facts[i % len(belief_facts)]
        
        tom_statement = template[0].format(belief=belief, fact=fact)
        self_centered = template[1].format(belief=belief, fact=fact)
        
        pairs.append((tom_statement, self_centered))
    
    return pairs


def run_dataset_generation():
    """
    Test 2A: Generate expanded datasets
    
    Time: 1 hour
    Cost: $0 (no GPU)
    """
    print("="*80)
    print("GATE 2, TEST 2A: GENERATE EXPANDED DATASETS")
    print("="*80)
    print("\nGenerating 100 pairs each for empathy and theory of mind...")
    
    empathy_pairs = generate_empathy_pairs(100)
    tom_pairs = generate_theory_of_mind_pairs(100)
    
    print(f"\n✓ Generated {len(empathy_pairs)} empathy pairs")
    print(f"✓ Generated {len(tom_pairs)} theory of mind pairs")
    
    # Quality check: Show samples
    print("\n" + "="*80)
    print("QUALITY CHECK - SAMPLE PAIRS:")
    print("="*80)
    
    print("\nEmpathy samples:")
    for i in [0, 25, 50, 75]:
        emp, neu = empathy_pairs[i]
        print(f"\n  [{i}] Empathetic: {emp}")
        print(f"      Neutral:     {neu}")
    
    print("\n" + "-"*80)
    print("\nTheory of Mind samples:")
    for i in [0, 25, 50, 75]:
        tom, self = tom_pairs[i]
        print(f"\n  [{i}] ToM:          {tom}")
        print(f"      Self-centered: {self}")
    
    # Save datasets
    data = {
        'empathy': empathy_pairs,
        'theory_of_mind': tom_pairs,
        'n_pairs': 100,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('gate2_datasets.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("\n" + "="*80)
    print("MANUAL REVIEW REQUIRED:")
    print("="*80)
    print("\nBefore proceeding to Test 2B, please verify:")
    print("  1. Pairs are clearly contrastive")
    print("  2. Cover multiple subtypes (not repetitive)")
    print("  3. Empathy pairs show affective language vs neutral")
    print("  4. ToM pairs show perspective-taking vs self-centered")
    print("\nIf quality looks good, proceed to Test 2B")
    
    return {
        'test': '2A_generation',
        'empathy_pairs': len(empathy_pairs),
        'tom_pairs': len(tom_pairs),
        'status': 'COMPLETE',
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# TEST 2B: EXTRACT VECTORS AND TEST CORRELATION
# =============================================================================

def extract_vector(
    pairs: List[Tuple[str, str]],
    model: HookedTransformer,
    layer: int = 12,
    position: int = -1
) -> torch.Tensor:
    """
    Extract contrastive vector from pairs using PCA.
    
    Args:
        pairs: List of (positive, negative) text pairs
        model: HookedTransformer model
        layer: Which layer to extract from
        position: Which token position (-1 = last)
        
    Returns:
        Normalized vector (d_model dimensions)
    """
    device = next(model.parameters()).device
    
    # Get activations for all pairs
    differences = []
    
    for pos_text, neg_text in tqdm(pairs, desc=f"Extracting L{layer}"):
        # Tokenize and get activations
        pos_tokens = model.to_tokens(pos_text).to(device)
        neg_tokens = model.to_tokens(neg_text).to(device)
        
        with torch.no_grad():
            _, cache_pos = model.run_with_cache(pos_tokens)
            _, cache_neg = model.run_with_cache(neg_tokens)
        
        # Get residual stream at specified layer and position
        pos_act = cache_pos[f"blocks.{layer}.hook_resid_post"][0, position, :].cpu()
        neg_act = cache_neg[f"blocks.{layer}.hook_resid_post"][0, position, :].cpu()
        
        # Compute difference
        diff = pos_act - neg_act
        differences.append(diff.numpy())
    
    # PCA on differences
    differences = np.array(differences)
    pca = PCA(n_components=1)
    pca.fit(differences)
    
    # Get primary component
    vector = torch.tensor(pca.components_[0], dtype=torch.float32)
    vector = vector / vector.norm()  # Normalize
    
    variance_explained = float(pca.explained_variance_ratio_[0])
    
    print(f"  Variance explained: {variance_explained:.3f}")
    
    return vector, variance_explained


def run_correlation_test():
    """
    Test 2B: Extract vectors and test correlation
    
    THE CRITICAL TEST.
    
    Time: 2 hours
    Cost: $10
    
    Success criteria:
    - r ≥ 0.4: ✅ STRONG validation
    - 0.3 ≤ r < 0.4: ⚠️ WEAK validation  
    - r < 0.3: ❌ STOP
    """
    print("\n" + "="*80)
    print("GATE 2, TEST 2B: CRITICAL CORRELATION TEST")
    print("="*80)
    print("\n⚠️  THIS IS THE GO/NO-GO DECISION POINT")
    print("    If r < 0.3, STOP and don't waste $130 on Gates 3-5")
    print()
    
    # Load datasets
    print("Loading datasets...")
    with open('gate2_datasets.json', 'r') as f:
        data = json.load(f)
    
    empathy_pairs = [tuple(p) for p in data['empathy']]
    tom_pairs = [tuple(p) for p in data['theory_of_mind']]
    
    print(f"✓ Loaded {len(empathy_pairs)} empathy pairs")
    print(f"✓ Loaded {len(tom_pairs)} ToM pairs")
    
    # Load model
    print("\nLoading Llama-3-8B-Instruct...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = HookedTransformer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        device=device,
        dtype=torch.bfloat16,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False
    )
    
    print("✓ Model loaded")
    
    # Extract vectors
    print("\nExtracting empathy vector (n=100)...")
    empathy_vector, empathy_var = extract_vector(empathy_pairs, model, layer=12)
    
    print("\nExtracting theory of mind vector (n=100)...")
    tom_vector, tom_var = extract_vector(tom_pairs, model, layer=12)
    
    # Compute correlation
    correlation = torch.cosine_similarity(
        empathy_vector.unsqueeze(0),
        tom_vector.unsqueeze(0)
    ).item()
    
    # CRITICAL DECISION POINT
    print("\n" + "="*80)
    print("CRITICAL RESULT:")
    print("="*80)
    print(f"\nEmpathy ↔ Theory of Mind correlation (n=100): r = {correlation:.3f}")
    print(f"Empathy variance explained: {empathy_var:.3f}")
    print(f"ToM variance explained: {tom_var:.3f}")
    
    print("\n" + "="*80)
    print("DECISION:")
    print("="*80)
    
    if correlation >= 0.4:
        decision = "STRONG_GO"
        print("\n🟢 r ≥ 0.4: STRONG VALIDATION")
        print("\n   ✅ Framework validated")
        print("   ✅ Sample size WAS the problem (n=10 insufficient)")
        print("   ✅ Conceptually related vectors DO correlate with adequate data")
        print("\n   → Proceed confidently to Gate 3")
        print("   → Expect similar improvements for other vectors")
        
    elif correlation >= 0.3:
        decision = "WEAK_GO"
        print("\n🟡 0.3 ≤ r < 0.4: WEAK VALIDATION")
        print("\n   ⚠️  Framework marginally validated")
        print("   ⚠️  Correlation weaker than expected (psych r~0.5-0.7)")
        print("   ⚠️  May need n>100 or alternative methods")
        print("\n   → Proceed cautiously to Gate 3")
        print("   → Flag uncertainty in interpretation")
        
    else:
        decision = "STOP"
        print("\n🔴 r < 0.3: VALIDATION FAILED")
        print("\n   ❌ Framework NOT validated")
        print("   ❌ Even with n=100, no correlation")
        print("   ❌ Fundamental methodological issue")
        print("\n   → STOP HERE - don't waste $130 on Gates 3-5")
        print("\n   Possible causes:")
        print("      1. Contrastive PCA inappropriate for these concepts")
        print("      2. LLMs represent empathy≠ToM orthogonally (ChatGPT theory)")
        print("      3. Layer 12 wrong for social concepts")
        print("      4. Dataset generation flawed")
        print("\n   Recommended next steps:")
        print("      - Run Test 2C (alternative layer search)")
        print("      - Try non-contrastive PCA (Gate 3)")
        print("      - Consider fundamentally different approach")
        print("      - Accept ChatGPT's reframing: 'control surface not ontology'")
    
    # Save results
    results = {
        'test': '2B_correlation',
        'correlation': float(correlation),
        'empathy_variance': float(empathy_var),
        'tom_variance': float(tom_var),
        'decision': decision,
        'n_pairs': 100,
        'layer': 12,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('gate2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save vectors
    torch.save({
        'empathy': empathy_vector,
        'theory_of_mind': tom_vector,
        'metadata': results
    }, 'gate2_vectors.pt')
    
    print("\n✓ Results saved: gate2_results.json")
    print("✓ Vectors saved: gate2_vectors.pt")
    
    return results


# =============================================================================
# TEST 2C: ALTERNATIVE LAYER SEARCH (IF 2B FAILS)
# =============================================================================

def run_layer_search():
    """
    Test 2C: Alternative layer search
    
    Only run if Test 2B failed (r < 0.3)
    
    Hypothesis: Maybe social concepts aren't at layer 12
    
    Time: 3 hours
    Cost: $15
    """
    print("\n" + "="*80)
    print("GATE 2, TEST 2C: ALTERNATIVE LAYER SEARCH")
    print("="*80)
    print("\nHypothesis: Social concepts may route through different layer")
    print("Testing layers 8-16 to find social concept bottleneck...\n")
    
    # Load datasets and model (assumes 2B already ran)
    with open('gate2_datasets.json', 'r') as f:
        data = json.load(f)
    
    empathy_pairs = [tuple(p) for p in data['empathy']]
    tom_pairs = [tuple(p) for p in data['theory_of_mind']]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        device=device,
        dtype=torch.bfloat16
    )
    
    # Test layers 8-16
    results = {}
    
    for layer in range(8, 17):
        print(f"\nTesting layer {layer}...")
        
        empathy_v, _ = extract_vector(empathy_pairs, model, layer=layer)
        tom_v, _ = extract_vector(tom_pairs, model, layer=layer)
        
        r = torch.cosine_similarity(
            empathy_v.unsqueeze(0),
            tom_v.unsqueeze(0)
        ).item()
        
        results[layer] = r
        print(f"  Layer {layer}: r = {r:.3f}")
    
    # Find best layer
    best_layer = max(results, key=results.get)
    best_r = results[best_layer]
    
    print("\n" + "="*80)
    print("LAYER SEARCH RESULTS:")
    print("="*80)
    print(f"\nBest layer: {best_layer}")
    print(f"Best correlation: r = {best_r:.3f}")
    
    if best_r >= 0.3:
        print(f"\n✓ Found better layer: {best_layer} (r={best_r:.3f})")
        print("  → Social concepts DO correlate at different layer")
        print("  → Re-extract all social vectors at this layer")
        decision = "FOUND_BETTER_LAYER"
    else:
        print(f"\n❌ No layer shows r ≥ 0.3")
        print("  → Social concepts genuinely orthogonal across all layers")
        print("  → Accept ChatGPT's interpretation:")
        print("     'LLMs factorize social cognition differently than humans'")
        decision = "NO_LAYER_WORKS"
    
    search_results = {
        'test': '2C_layer_search',
        'results_by_layer': {str(k): float(v) for k, v in results.items()},
        'best_layer': int(best_layer),
        'best_correlation': float(best_r),
        'decision': decision,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('gate2_layer_search.json', 'w') as f:
        json.dump(search_results, f, indent=2)
    
    return search_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run Gate 2 tests"""
    
    print("\n" + "="*80)
    print("GATE 2: CRITICAL LITMUS TEST")
    print("Empathy ↔ Theory of Mind Correlation")
    print("="*80)
    print("\n⚠️  THIS IS THE MOST IMPORTANT TEST")
    print("   If r < 0.3, STOP and save $130")
    print()
    
    import sys
    
    # Test 2A: Generate datasets
    if '--skip-2a' not in sys.argv:
        print("Running Test 2A (dataset generation)...\n")
        results_2a = run_dataset_generation()
        
        input("\nPress Enter after reviewing sample pairs to continue...")
    else:
        print("Skipping Test 2A (assuming datasets exist)\n")
    
    # Test 2B: Extract and correlate
    if '--skip-2b' not in sys.argv:
        print("\nRunning Test 2B (CRITICAL CORRELATION TEST)...\n")
        results_2b = run_correlation_test()
        
        # Decision gate
        if results_2b['decision'] == 'STOP':
            print("\n" + "="*80)
            print("STOP CONDITION MET")
            print("="*80)
            print("\nDo NOT proceed to Gates 3-5")
            print("Budget saved: $130")
            print("\nOptions:")
            print("  1. Run Test 2C (layer search) - $15")
            print("  2. Try alternative methods (Gate 3 only) - $35")
            print("  3. Accept negative result, write paper about what doesn't work")
            
            if input("\nRun Test 2C (alternative layer search)? [y/N]: ").lower() == 'y':
                results_2c = run_layer_search()
            
            return
        
        else:
            print(f"\n✅ Gate 2 PASSED (decision: {results_2b['decision']})")
            print("\nProceed to Gate 3 (method validation)")
    
    print("\n" + "="*80)
    print("GATE 2 COMPLETE")
    print("="*80)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()
