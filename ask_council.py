#!/usr/bin/env python3
"""
Ask Council for Next Steps
==========================

Lightweight script to get council recommendations without needing GPU/PyTorch.
Uses existing experimental results to ask the 4-model council for guidance.

Usage:
    export ANTHROPIC_API_KEY=your_key
    export OPENAI_API_KEY=your_key
    export GOOGLE_AI_API_KEY=your_key
    export XAI_API_KEY=your_key
    python ask_council.py
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path

# Review prompts
REVIEW_SYSTEM_PROMPT = """You are an expert AI safety researcher reviewing mechanistic interpretability experiments.

Your role: {perspective} reviewer

Be rigorous but constructive. Focus on scientific validity and reproducibility."""

REVIEW_USER_PROMPT = """## Experiment Results

### Hypothesis
{hypothesis}

### Results
```json
{results}
```

Review from the **{perspective}** perspective.

Return JSON:
{{
  "verdict": "GREEN" | "YELLOW" | "RED",
  "critique": "Your analysis (2-3 paragraphs)",
  "required_controls": ["list of needed experiments"],
  "confidence": 0.0-1.0,
  "proceed": true | false,
  "next_steps": ["recommended next steps in priority order"]
}}

JSON only:"""


async def review_with_claude(results: dict, hypothesis: str, perspective: str) -> dict:
    """Claude Opus 4.5 review."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return {"verdict": "YELLOW", "critique": "API key not configured", "required_controls": [], "confidence": 0, "proceed": False, "error": "No API key"}

    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=api_key)

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=REVIEW_SYSTEM_PROMPT.format(perspective=perspective),
        messages=[{"role": "user", "content": REVIEW_USER_PROMPT.format(
            hypothesis=hypothesis,
            results=json.dumps(results, indent=2, default=str)[:8000],
            perspective=perspective
        )}],
        temperature=0.3
    )

    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text)


async def review_with_gpt(results: dict, hypothesis: str, perspective: str) -> dict:
    """GPT-4o review."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return {"verdict": "YELLOW", "critique": "API key not configured", "required_controls": [], "confidence": 0, "proceed": False, "error": "No API key"}

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT.format(perspective=perspective)},
            {"role": "user", "content": REVIEW_USER_PROMPT.format(
                hypothesis=hypothesis,
                results=json.dumps(results, indent=2, default=str)[:8000],
                perspective=perspective
            )}
        ],
        temperature=0.3,
        max_tokens=4000,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


async def review_with_gemini(results: dict, hypothesis: str, perspective: str) -> dict:
    """Gemini review."""
    api_key = os.environ.get('GOOGLE_AI_API_KEY')
    if not api_key:
        return {"verdict": "YELLOW", "critique": "API key not configured", "required_controls": [], "confidence": 0, "proceed": False, "error": "No API key"}

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    prompt = REVIEW_SYSTEM_PROMPT.format(perspective=perspective) + "\n\n" + REVIEW_USER_PROMPT.format(
        hypothesis=hypothesis,
        results=json.dumps(results, indent=2, default=str)[:8000],
        perspective=perspective
    )

    response = await client.aio.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=4000,
            response_mime_type="application/json"
        )
    )

    return json.loads(response.text)


async def review_with_grok(results: dict, hypothesis: str, perspective: str) -> dict:
    """Grok review."""
    api_key = os.environ.get('XAI_API_KEY')
    if not api_key:
        return {"verdict": "YELLOW", "critique": "API key not configured", "required_controls": [], "confidence": 0, "proceed": False, "error": "No API key"}

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    response = await client.chat.completions.create(
        model="grok-3-fast",
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT.format(perspective=perspective)},
            {"role": "user", "content": REVIEW_USER_PROMPT.format(
                hypothesis=hypothesis,
                results=json.dumps(results, indent=2, default=str)[:8000],
                perspective=perspective
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
    return json.loads(text)


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

        # Find best config
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

        # Key finding
        gemma_4layer = cycle3.get("Exp4_Gemma_4layer", {})
        if gemma_4layer.get("success_rate", 0) >= 0.95:
            data["key_breakthrough"] = "Gemma 4-layer coordinated steering achieves 95% success (previously resistant)"

    # Pending experiments
    data["pending_experiments"] = [
        "Orthogonal vector control (direction specificity test)",
        "Validation n=100 (increased statistical power)",
        "Adaptive adversarial attacks (jailbreak robustness)",
        "Full Llama-3.1-8B sweep (28 configs)",
    ]

    return data


async def main():
    print("=" * 70)
    print("ASKING COUNCIL FOR NEXT STEPS")
    print("=" * 70)
    print()

    # Load results
    print("Loading experimental results...")
    results = load_results()

    print(f"Completed: {len(results['completed_experiments'])} experiments")
    print(f"Pending: {len(results['pending_experiments'])} experiments")
    if results.get("key_breakthrough"):
        print(f"Key finding: {results['key_breakthrough']}")
    print()

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

    # Run reviews
    reviewers = [
        ("Claude (mechanistic)", review_with_claude, "mechanistic interpretability"),
        ("GPT-4o (security)", review_with_gpt, "security and adversarial robustness"),
        ("Gemini (theory)", review_with_gemini, "theoretical foundations"),
        ("Grok (methods)", review_with_grok, "experimental methods and statistics"),
    ]

    reviews = {}
    for name, func, perspective in reviewers:
        print(f"🤖 Asking {name}...")
        try:
            review = await func(results, hypothesis, perspective)
            reviews[name] = review
            print(f"   ✅ Verdict: {review.get('verdict', 'ERROR')}")
        except Exception as e:
            reviews[name] = {"verdict": "ERROR", "error": str(e)}
            print(f"   ❌ Error: {e}")

    # Consensus
    print()
    print("=" * 70)
    print("COUNCIL CONSENSUS")
    print("=" * 70)
    print()

    verdicts = [r.get("verdict", "YELLOW") for r in reviews.values() if "error" not in r]
    confidences = [r.get("confidence", 0.5) for r in reviews.values() if "error" not in r]

    if verdicts:
        avg_confidence = sum(confidences) / len(confidences)

        if all(v == "GREEN" for v in verdicts):
            action = "PROCEED_TO_PAPER"
        elif any(v == "RED" for v in verdicts):
            action = "HALT_AND_REVIEW"
        else:
            action = "RUN_FOLLOWUP"

        print(f"Action: {action}")
        print(f"Average confidence: {avg_confidence:.1%}")
        print()

    # Display individual reviews
    print("Individual Reviews:")
    print("-" * 70)

    all_next_steps = []
    for name, review in reviews.items():
        verdict = review.get("verdict", "ERROR")
        emoji = "✅" if verdict == "GREEN" else "⚠️" if verdict == "YELLOW" else "❌"
        print(f"\n{emoji} {name}: {verdict}")

        if review.get("critique"):
            # Print first 300 chars of critique
            critique = review["critique"][:500]
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

    # Count frequency of recommendations
    from collections import Counter
    step_counts = Counter(all_next_steps)

    print()
    for i, (step, count) in enumerate(step_counts.most_common(10), 1):
        print(f"{i}. {step} (mentioned by {count} reviewer{'s' if count > 1 else ''})")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "results_summary": results,
        "hypothesis": hypothesis,
        "reviews": reviews,
        "recommended_next_steps": [s for s, _ in step_counts.most_common(10)],
    }

    output_file = Path("council_recommendations.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print(f"💾 Full results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
