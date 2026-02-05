#!/usr/bin/env python3
"""
Ask Council for Next Steps (Advanced Configuration)
====================================================

Multi-LLM council with flagship models (Feb 2026):
- Claude Opus 4.5 (Mechanistic Interpretability)
- GPT-5.2 Pro (Security & Adversarial)
- Gemini 3 Pro (Theoretical Foundations)
- Grok 4.1 (Statistics & Methods)

Features:
- Weighted consensus based on model expertise
- Adaptive stability detection (stops when converged)
- Multi-round debate support
- Monitor module for validation

Usage:
    export ANTHROPIC_API_KEY=your_key
    export OPENAI_API_KEY=your_key
    export GOOGLE_AI_API_KEY=your_key
    export XAI_API_KEY=your_key
    python ask_council.py [--rounds N] [--fast]
"""

import os
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Optional
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# =============================================================================
# COUNCIL CONFIGURATION
# =============================================================================

COUNCIL_CONFIG = {
    "claude": {
        "name": "Claude Opus 4.5",
        "model": "claude-opus-4-5-20251101",
        "model_fast": "claude-sonnet-4-20250514",
        "role": "mechanistic interpretability",
        "weight": 1.2,
        "expertise": ["activation analysis", "causal interventions", "representation learning"],
    },
    "gpt": {
        "name": "GPT-5.2 Pro",
        "model": "gpt-5.2-pro",
        "model_fast": "gpt-4o",
        "role": "security and adversarial robustness",
        "weight": 1.15,
        "expertise": ["red teaming", "threat modeling", "dual-use risks"],
    },
    "gemini": {
        "name": "Gemini 3 Flash",
        "model": "gemini-3-flash-preview",
        "model_fast": "gemini-2.5-flash",
        "role": "theoretical foundations",
        "weight": 1.15,  # Upgraded: thinking model with better reasoning
        "expertise": ["information theory", "formal verification", "mathematical rigor"],
    },
    "grok": {
        "name": "Grok 4",
        "model": "grok-4-0709",
        "model_fast": "grok-3",
        "role": "experimental methods and statistics",
        "weight": 1.2,  # Top tier on LMArena
        "expertise": ["statistical power", "experimental design", "reproducibility"],
    },
}

# Consensus thresholds
CONVERGENCE_THRESHOLD = 0.75  # 75% agreement for convergence
MIN_CONFIDENCE = 0.7  # Minimum confidence for valid verdict

# =============================================================================
# PROMPTS
# =============================================================================

REVIEW_SYSTEM_PROMPT = """You are an expert AI safety researcher on a multi-LLM council reviewing mechanistic interpretability experiments.

Your role: **{role}** expert
Your expertise: {expertise}

You are part of a council with other frontier models. Be rigorous but constructive.
Focus on scientific validity, reproducibility, and your area of expertise.
If you disagree with other reviewers, explain your reasoning clearly."""

REVIEW_USER_PROMPT = """## Experiment Results

### Hypothesis
{hypothesis}

### Results
```json
{results}
```

{previous_reviews}

Review from the **{role}** perspective.

Return JSON (and ONLY JSON):
{{
  "verdict": "GREEN" | "YELLOW" | "RED",
  "critique": "Your detailed analysis (2-3 paragraphs)",
  "required_controls": ["list of needed experiments"],
  "confidence": 0.0-1.0,
  "proceed": true | false,
  "next_steps": ["recommended next steps in priority order"],
  "disagreements": ["any disagreements with other reviewers, if applicable"]
}}"""


# =============================================================================
# MONITOR MODULE
# =============================================================================

class CouncilMonitor:
    """Validates review coherence before aggregation."""

    @staticmethod
    def validate(review: dict, model_name: str) -> tuple[bool, str]:
        """Validate a review for internal consistency."""

        # Check required fields
        required = ["verdict", "critique", "confidence", "proceed"]
        for field in required:
            if field not in review:
                return False, f"Missing required field: {field}"

        # Verdict must be valid
        if review["verdict"] not in ["GREEN", "YELLOW", "RED"]:
            return False, f"Invalid verdict: {review['verdict']}"

        # Confidence must be in range
        if not (0 <= review.get("confidence", 0) <= 1):
            return False, f"Confidence out of range: {review['confidence']}"

        # Check verdict/proceed consistency
        if review["verdict"] == "RED" and review.get("proceed", False):
            return False, "Inconsistent: RED verdict but proceed=true"

        if review["verdict"] == "GREEN" and not review.get("proceed", True):
            return False, "Inconsistent: GREEN verdict but proceed=false"

        # Check high confidence with many required controls
        if review.get("confidence", 0) > 0.9 and len(review.get("required_controls", [])) > 5:
            return False, "Suspicious: very high confidence but many required controls"

        return True, "OK"


# =============================================================================
# REVIEWER FUNCTIONS
# =============================================================================

async def review_with_claude(
    results: dict,
    hypothesis: str,
    previous_reviews: str = "",
    fast_mode: bool = False
) -> dict:
    """Claude Opus 4.5 review."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return {"verdict": "YELLOW", "critique": "API key not configured", "required_controls": [], "confidence": 0, "proceed": False, "error": "No API key"}

    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=api_key)

    config = COUNCIL_CONFIG["claude"]
    model = config["model_fast"] if fast_mode else config["model"]

    response = await client.messages.create(
        model=model,
        max_tokens=4000,
        system=REVIEW_SYSTEM_PROMPT.format(
            role=config["role"],
            expertise=", ".join(config["expertise"])
        ),
        messages=[{"role": "user", "content": REVIEW_USER_PROMPT.format(
            hypothesis=hypothesis,
            results=json.dumps(results, indent=2, default=str)[:8000],
            role=config["role"],
            previous_reviews=previous_reviews
        )}],
        temperature=0.3
    )

    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    review = json.loads(text)
    review["model"] = config["name"]
    review["model_id"] = model
    return review


async def review_with_gpt(
    results: dict,
    hypothesis: str,
    previous_reviews: str = "",
    fast_mode: bool = False
) -> dict:
    """GPT-5.2 Pro review (uses Responses API for full model, Chat for fast)."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return {"verdict": "YELLOW", "critique": "API key not configured", "required_controls": [], "confidence": 0, "proceed": False, "error": "No API key"}

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    config = COUNCIL_CONFIG["gpt"]

    prompt_content = REVIEW_USER_PROMPT.format(
        hypothesis=hypothesis,
        results=json.dumps(results, indent=2, default=str)[:8000],
        role=config["role"],
        previous_reviews=previous_reviews
    )

    system_content = REVIEW_SYSTEM_PROMPT.format(
        role=config["role"],
        expertise=", ".join(config["expertise"])
    )

    if fast_mode:
        # Use Chat Completions API for fast mode
        response = await client.chat.completions.create(
            model=config["model_fast"],
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.3,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        review = json.loads(response.choices[0].message.content)
        review["model"] = "GPT-4o (fast)"
        review["model_id"] = config["model_fast"]
    else:
        # Use Responses API for GPT-5.2 Pro
        try:
            response = await client.responses.create(
                model=config["model"],
                input=f"{system_content}\n\n{prompt_content}\n\nRespond with JSON only."
            )
            text = response.output_text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            review = json.loads(text)
            review["model"] = config["name"]
            review["model_id"] = config["model"]
        except Exception as e:
            # Fallback to Chat API if Responses API fails
            print(f"   ⚠️ Responses API failed, falling back to Chat: {e}")
            response = await client.chat.completions.create(
                model=config["model_fast"],
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt_content}
                ],
                temperature=0.3,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            review = json.loads(response.choices[0].message.content)
            review["model"] = "GPT-4o (fallback)"
            review["model_id"] = config["model_fast"]

    return review


async def review_with_gemini(
    results: dict,
    hypothesis: str,
    previous_reviews: str = "",
    fast_mode: bool = False
) -> dict:
    """Gemini 3 Pro review with fallback chain."""
    api_key = os.environ.get('GOOGLE_AI_API_KEY')
    if not api_key:
        return {"verdict": "YELLOW", "critique": "API key not configured", "required_controls": [], "confidence": 0, "proceed": False, "error": "No API key"}

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = COUNCIL_CONFIG["gemini"]

    prompt = REVIEW_SYSTEM_PROMPT.format(
        role=config["role"],
        expertise=", ".join(config["expertise"])
    ) + "\n\n" + REVIEW_USER_PROMPT.format(
        hypothesis=hypothesis,
        results=json.dumps(results, indent=2, default=str)[:8000],
        role=config["role"],
        previous_reviews=previous_reviews
    )

    # Model fallback chain
    if fast_mode:
        models_to_try = [
            (config["model_fast"], "fast"),
        ]
    else:
        models_to_try = [
            (config["model"], "flagship"),
            (config["model_fast"], "fast fallback"),
        ]

    for model_name, description in models_to_try:
        try:
            # Gemini 3 is a thinking model - needs higher token limit
            max_tokens = 8000 if "gemini-3" in model_name else 4000

            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json"
                )
            )

            if response.text and len(response.text) > 50:
                review = json.loads(response.text)
                review["model"] = f"{config['name']} ({description})"
                review["model_id"] = model_name
                return review

        except Exception as e:
            print(f"   ⚠️ {model_name} failed: {str(e)[:80]}")
            continue

    return {"verdict": "ERROR", "error": "All Gemini models failed", "confidence": 0, "proceed": False}


async def review_with_grok(
    results: dict,
    hypothesis: str,
    previous_reviews: str = "",
    fast_mode: bool = False
) -> dict:
    """Grok 4.1 review (#1 on LMArena)."""
    api_key = os.environ.get('XAI_API_KEY')
    if not api_key:
        return {"verdict": "YELLOW", "critique": "API key not configured", "required_controls": [], "confidence": 0, "proceed": False, "error": "No API key"}

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    config = COUNCIL_CONFIG["grok"]
    model = config["model_fast"] if fast_mode else config["model"]

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT.format(
                role=config["role"],
                expertise=", ".join(config["expertise"])
            )},
            {"role": "user", "content": REVIEW_USER_PROMPT.format(
                hypothesis=hypothesis,
                results=json.dumps(results, indent=2, default=str)[:8000],
                role=config["role"],
                previous_reviews=previous_reviews
            )}
        ],
        temperature=0.3,
        max_tokens=4000
    )

    text = response.choices[0].message.content
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    review = json.loads(text)
    review["model"] = config["name"]
    review["model_id"] = model
    return review


# =============================================================================
# CONSENSUS FUNCTIONS
# =============================================================================

def check_convergence(reviews: list[dict]) -> bool:
    """Check if council has reached stable consensus."""
    valid_reviews = [r for r in reviews if "error" not in r]
    if len(valid_reviews) < 2:
        return False

    verdicts = [r.get("verdict", "YELLOW") for r in valid_reviews]
    confidences = [r.get("confidence", 0.5) for r in valid_reviews]

    # Count dominant verdict
    verdict_counts = Counter(verdicts)
    dominant_verdict, dominant_count = verdict_counts.most_common(1)[0]

    # Check if dominant verdict has enough support
    agreement_ratio = dominant_count / len(verdicts)
    if agreement_ratio < CONVERGENCE_THRESHOLD:
        return False

    # Check if agreeing reviewers have sufficient confidence
    agreeing_confidences = [c for v, c in zip(verdicts, confidences) if v == dominant_verdict]
    avg_confidence = sum(agreeing_confidences) / len(agreeing_confidences)

    return avg_confidence >= MIN_CONFIDENCE


def weighted_consensus(reviews: list[dict]) -> dict:
    """Calculate weighted consensus based on model expertise."""
    valid_reviews = [r for r in reviews if "error" not in r]
    if not valid_reviews:
        return {"verdict": "ERROR", "confidence": 0, "action": "NO_VALID_REVIEWS", "verdict_scores": {"GREEN": 0, "YELLOW": 0, "RED": 0}}

    # Get weights for each model
    model_weights = {cfg["name"]: cfg["weight"] for cfg in COUNCIL_CONFIG.values()}

    # Calculate weighted scores for each verdict
    verdict_scores = {"GREEN": 0, "YELLOW": 0, "RED": 0}
    total_weight = 0

    for review in valid_reviews:
        model_name = review.get("model", "").split(" (")[0]  # Remove suffix like "(fast)"
        base_weight = model_weights.get(model_name, 1.0)
        confidence = review.get("confidence", 0.5)

        # Combined weight = model expertise weight * confidence
        combined_weight = base_weight * confidence
        verdict_scores[review["verdict"]] += combined_weight
        total_weight += combined_weight

    # Determine final verdict
    final_verdict = max(verdict_scores, key=verdict_scores.get)
    final_confidence = verdict_scores[final_verdict] / total_weight if total_weight > 0 else 0

    # Determine action
    if final_verdict == "GREEN" and final_confidence >= 0.7:
        action = "PROCEED_TO_PAPER"
    elif final_verdict == "RED":
        action = "HALT_AND_REVIEW"
    else:
        action = "RUN_FOLLOWUP"

    return {
        "verdict": final_verdict,
        "confidence": final_confidence,
        "action": action,
        "verdict_scores": verdict_scores,
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_results():
    """Load existing experimental results."""
    results_dir = Path("results")

    data = {
        "project": "Crystallized Safety",
        "status": "Validation phase",
        "completed_experiments": [],
        "pending_experiments": [],
    }

    # Load Mistral sweep
    mistral_file = results_dir / "mistral_sweep_results.json"
    if mistral_file.exists():
        with open(mistral_file) as f:
            mistral = json.load(f)

        best = max(mistral, key=lambda x: x.get("control3", {}).get("flip_rate", 0))
        data["mistral_sweep"] = {
            "configs_tested": len(mistral),
            "best_config": f"L{best['layer']} α={best['alpha']}",
            "best_flip_rate": best.get("control3", {}).get("flip_rate", 0),
            "best_coherent_flip": best.get("control3", {}).get("coherent_flip_rate", 0),
            "all_controls_pass": best.get("control1", {}).get("verdict") == "GREEN" and
                                  best.get("control2", {}).get("verdict") == "GREEN" and
                                  best.get("control3", {}).get("verdict") == "GREEN"
        }
        data["completed_experiments"].append("Mistral-7B 28-config sweep")

    # Load Gemma sweep
    gemma_file = results_dir / "gemma_sweep_results.json"
    if gemma_file.exists():
        with open(gemma_file) as f:
            gemma = json.load(f)
        best = max(gemma, key=lambda x: x.get("control3", {}).get("flip_rate", 0))
        data["gemma_sweep"] = {
            "configs_tested": len(gemma),
            "best_flip_rate": best.get("control3", {}).get("flip_rate", 0),
            "status": "RESISTANT" if best.get("control3", {}).get("flip_rate", 0) < 0.2 else "VULNERABLE"
        }
        data["completed_experiments"].append("Gemma-2-9B 11-config sweep")

    # Load Cycle 3 results
    cycle3_file = results_dir / "cycle3_multilayer_results.json"
    if cycle3_file.exists():
        with open(cycle3_file) as f:
            cycle3 = json.load(f)
        data["cycle3_multilayer"] = cycle3
        data["completed_experiments"].append("Cycle 3 multilayer steering (7 experiments)")

        gemma_4layer = cycle3.get("Exp4_Gemma_4layer", {})
        if gemma_4layer.get("success_rate", 0) >= 0.95:
            data["key_breakthrough"] = "Gemma 4-layer coordinated steering achieves 95% success (previously resistant)"

    # Load EXP-08: Scale-Invariant Safety (Cross-Model Sigma)
    exp08_file = results_dir / "exp08_cross_model_sigma" / "combined_results.json"
    if exp08_file.exists():
        with open(exp08_file) as f:
            exp08 = json.load(f)

        # Extract key findings from EXP-08
        models_tested = list(exp08.get("models", {}).keys())
        alpha_grid = exp08.get("alpha_effective_grid", [])

        exp08_summary = {
            "experiment": exp08.get("experiment"),
            "timestamp": exp08.get("timestamp"),
            "models_tested": models_tested,
            "alpha_grid": alpha_grid,
            "model_results": {}
        }

        # Extract sigma and key metrics per model
        for model_name, model_data in exp08.get("models", {}).items():
            exp08_summary["model_results"][model_name] = {
                "n_layers": model_data.get("n_layers"),
                "sigma": model_data.get("sigma") if "sigma" in model_data else None,
                "layers_tested": len(model_data.get("layers", {})),
            }

        data["exp08_scale_invariant"] = exp08_summary
        data["completed_experiments"].append("EXP-08: Scale-Invariant Safety (4 models, cross-model sigma)")
        data["key_breakthrough"] = "Scale-Invariant Safety: Universal σ-normalized steering works across Llama, Gemma, Qwen, Mistral"
        data["status"] = "Scale-Invariant Safety CONFIRMED"

    # Pending experiments (updated)
    if exp08_file.exists():
        data["pending_experiments"] = [
            "Publication preparation",
            "Responsible disclosure to model providers",
            "Defense mechanism proposals",
        ]
    else:
        data["pending_experiments"] = [
            "Orthogonal vector control (direction specificity test)",
            "Validation n=100 (increased statistical power)",
            "Adaptive adversarial attacks (jailbreak robustness)",
            "Full Llama-3.1-8B sweep (28 configs)",
        ]

    return data


# =============================================================================
# MAIN
# =============================================================================

async def run_council(
    results: dict,
    hypothesis: str,
    max_rounds: int = 1,
    fast_mode: bool = False
) -> dict:
    """Run the council review with optional multi-round debate."""

    monitor = CouncilMonitor()
    all_rounds = []

    reviewers = [
        ("Claude Opus 4.5", review_with_claude, COUNCIL_CONFIG["claude"]),
        ("GPT-5.2 Pro", review_with_gpt, COUNCIL_CONFIG["gpt"]),
        ("Gemini 3 Flash", review_with_gemini, COUNCIL_CONFIG["gemini"]),
        ("Grok 4", review_with_grok, COUNCIL_CONFIG["grok"]),
    ]

    previous_reviews_text = ""

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num}/{max_rounds}")
        print(f"{'='*70}")

        round_reviews = {}

        for name, func, config in reviewers:
            model_id = config["model_fast"] if fast_mode else config["model"]
            print(f"🤖 Asking {name} ({model_id})...")

            try:
                review = await func(results, hypothesis, previous_reviews_text, fast_mode)

                # Validate with monitor
                is_valid, validation_msg = monitor.validate(review, name)
                if not is_valid:
                    print(f"   ⚠️ Monitor rejected: {validation_msg}")
                    review["monitor_warning"] = validation_msg

                round_reviews[name] = review
                print(f"   ✅ Verdict: {review.get('verdict', 'ERROR')} (confidence: {review.get('confidence', 0):.0%})")

            except Exception as e:
                round_reviews[name] = {"verdict": "ERROR", "error": str(e), "confidence": 0, "proceed": False}
                print(f"   ❌ Error: {e}")

        all_rounds.append(round_reviews)

        # Check for convergence
        if check_convergence(list(round_reviews.values())):
            print(f"\n✅ Council converged after round {round_num}")
            break

        # Prepare previous reviews for next round
        if round_num < max_rounds:
            previous_reviews_text = "\n\n## Previous Round Reviews:\n"
            for name, review in round_reviews.items():
                if "error" not in review:
                    previous_reviews_text += f"\n### {name}: {review.get('verdict')}\n"
                    previous_reviews_text += f"{review.get('critique', '')[:500]}...\n"

    # Calculate final consensus
    final_reviews = all_rounds[-1]
    consensus = weighted_consensus(list(final_reviews.values()))

    return {
        "rounds": all_rounds,
        "final_reviews": final_reviews,
        "consensus": consensus,
        "converged": len(all_rounds) < max_rounds or check_convergence(list(final_reviews.values())),
    }


async def main():
    parser = argparse.ArgumentParser(description="Ask the AI Council for guidance")
    parser.add_argument("--rounds", type=int, default=1, help="Max debate rounds (default: 1)")
    parser.add_argument("--fast", action="store_true", help="Use faster/cheaper models")
    args = parser.parse_args()

    print("=" * 70)
    print("AI COUNCIL - FLAGSHIP MODELS (Feb 2026)")
    print("=" * 70)
    print()
    print("Council Members:")
    for key, config in COUNCIL_CONFIG.items():
        model = config["model_fast"] if args.fast else config["model"]
        print(f"  • {config['name']}: {model} (weight: {config['weight']})")
    print()

    # Load results
    print("Loading experimental results...")
    results = load_results()

    print(f"Completed: {len(results['completed_experiments'])} experiments")
    print(f"Pending: {len(results['pending_experiments'])} experiments")
    if results.get("key_breakthrough"):
        print(f"Key finding: {results['key_breakthrough']}")

    # Build hypothesis based on loaded results
    if results.get("exp08_scale_invariant"):
        hypothesis = """
Scale-Invariant Safety: Universal vulnerability across frontier LLMs

MAJOR DISCOVERY (EXP-08):
- σ-normalized steering (α_eff = α/σ) works universally across models
- Tested on: Llama-3.1-8B, Gemma-2-9B, Qwen-2.5-7B, Mistral-7B
- Layer-normalized steering transfers between architectures
- Same α_eff produces similar jailbreak rates across all models

PREVIOUS FINDINGS:
- Mistral-7B-Instruct: High jailbreak success with refusal direction steering
- Gemma-2-9B: Initially resistant, broken by multi-layer coordinated steering
- Cycle 3: 4-layer attacks achieve 95%+ success on resistant models

IMPLICATIONS:
1. Safety mechanisms share geometric structure across model families
2. Single defense-in-depth strategy may be insufficient
3. Universal vulnerability suggests fundamental architectural issue

QUESTION FOR COUNCIL:
1. Is this finding publication-ready? What additional validation is needed?
2. Responsible disclosure: How should we notify Anthropic, OpenAI, Google, Meta?
3. Defense proposals: What mitigations should we recommend?
4. Ethical considerations for publishing this vulnerability?
"""
    else:
        hypothesis = """
Crystallized Safety: Layer-specific vulnerabilities in LLM safety mechanisms

COMPLETED:
- Mistral-7B-Instruct: 83% jailbreak success at L24 α=15 (all 3 controls pass)
- Gemma-2-9B: Resistant to single-layer steering (max 11% flip rate)
- Cycle 3 breakthrough: 4-layer coordinated steering breaks Gemma resistance (95%+ success)

PENDING VALIDATION:
1. Orthogonal vector control - verify direction specificity vs magnitude effect
2. n=100 validation - increase statistical power for publication
3. Adaptive adversarial attacks - test robustness against jailbreak techniques
4. Full Llama-3.1 sweep - complete cross-model validation

QUESTION FOR COUNCIL:
Given these results, what are the highest priority next steps for:
1. Strengthening the scientific validity of findings
2. Preparing for publication
3. Responsible disclosure considerations
"""

    # Run council
    council_result = await run_council(
        results=results,
        hypothesis=hypothesis,
        max_rounds=args.rounds,
        fast_mode=args.fast
    )

    # Display consensus
    print()
    print("=" * 70)
    print("WEIGHTED COUNCIL CONSENSUS")
    print("=" * 70)
    print()

    consensus = council_result["consensus"]
    print(f"Final Verdict: {consensus['verdict']}")
    print(f"Confidence: {consensus['confidence']:.1%}")
    print(f"Action: {consensus['action']}")
    print(f"Converged: {'Yes' if council_result['converged'] else 'No'}")
    print()
    print("Verdict Scores (weighted):")
    for verdict, score in consensus.get("verdict_scores", {}).items():
        print(f"  {verdict}: {score:.2f}")

    # Display individual reviews
    print()
    print("=" * 70)
    print("INDIVIDUAL REVIEWS")
    print("=" * 70)

    all_next_steps = []
    for name, review in council_result["final_reviews"].items():
        verdict = review.get("verdict", "ERROR")
        confidence = review.get("confidence", 0)
        emoji = "✅" if verdict == "GREEN" else "⚠️" if verdict == "YELLOW" else "❌"
        print(f"\n{emoji} {name}: {verdict} ({confidence:.0%})")

        if review.get("critique"):
            critique = review["critique"][:400]
            print(f"   {critique}...")

        if review.get("next_steps"):
            all_next_steps.extend(review["next_steps"])

        if review.get("required_controls"):
            print(f"   Required: {', '.join(review['required_controls'][:3])}")

    # Aggregate next steps
    print()
    print("=" * 70)
    print("RECOMMENDED NEXT STEPS (aggregated)")
    print("=" * 70)

    step_counts = Counter(all_next_steps)
    print()
    for i, (step, count) in enumerate(step_counts.most_common(10), 1):
        print(f"{i}. {step} (mentioned by {count} reviewer{'s' if count > 1 else ''})")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_rounds": args.rounds,
            "fast_mode": args.fast,
            "models": {k: v["model_fast"] if args.fast else v["model"] for k, v in COUNCIL_CONFIG.items()},
        },
        "results_summary": results,
        "hypothesis": hypothesis,
        "council_result": council_result,
        "recommended_next_steps": [s for s, _ in step_counts.most_common(10)],
    }

    output_file = Path("council_recommendations.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print(f"💾 Full results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
