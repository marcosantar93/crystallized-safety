#!/usr/bin/env python3
"""
Multi-LLM Consensus Pilot Study - Baseline Implementations
===========================================================

Implements three baseline conditions:
1. Single-model: Each model evaluates independently
2. Homogeneous-4: 4 instances of Claude Opus 4.5
3. Heterogeneous-4: Claude + GPT + Gemini + Grok (current system)

For each of 20 research decisions, each baseline predicts the correct answer,
then we compare to ground truth.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Tuple
from datetime import datetime
from pathlib import Path


# =============================================================================
# MODEL API CLIENTS
# =============================================================================

async def call_claude(prompt: str, system_prompt: str = "") -> Dict[str, Any]:
    """Call Claude Opus 4.5."""
    print("[DEBUG] call_claude: Starting...")
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return {"error": "No API key", "response": None, "model": "claude-opus-4-5-20251101"}

    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=api_key)

    print("[DEBUG] call_claude: Making API call...")
    try:
        response = await client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        print("[DEBUG] call_claude: Got response, parsing...")
        text = response.content[0].text

        # Parse JSON if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        print("[DEBUG] call_claude: Success!")
        return {"response": text, "model": "claude-opus-4-5-20251101", "error": None}
    except Exception as e:
        print(f"[DEBUG] call_claude: Error - {e}")
        return {"error": str(e), "response": None, "model": "claude-opus-4-5-20251101"}


async def call_gpt(prompt: str, system_prompt: str = "") -> Dict[str, Any]:
    """Call GPT-4o."""
    print("[DEBUG] call_gpt: Starting...")
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return {"error": "No API key", "response": None, "model": "gpt-4o"}

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, timeout=60.0)

    try:
        print("[DEBUG] call_gpt: Making API call...")
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            ),
            timeout=60.0
        )
        print("[DEBUG] call_gpt: Got response, parsing...")
        text = response.choices[0].message.content

        # Parse JSON if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        print("[DEBUG] call_gpt: Success!")
        return {"response": text, "model": "gpt-4o", "error": None}
    except asyncio.TimeoutError:
        print("[DEBUG] call_gpt: Timeout!")
        return {"error": "Timeout after 60s", "response": None, "model": "gpt-4o"}
    except Exception as e:
        print(f"[DEBUG] call_gpt: Error - {e}")
        return {"error": str(e), "response": None, "model": "gpt-4o"}


async def call_gemini(prompt: str, system_prompt: str = "") -> Dict[str, Any]:
    """Call Gemini 2.5 Pro."""
    print("[DEBUG] call_gemini: Starting...")
    api_key = os.environ.get('GOOGLE_AI_API_KEY')
    if not api_key:
        return {"error": "No API key", "response": None, "model": "gemini-2.5-pro"}

    import google.generativeai as genai
    genai.configure(api_key=api_key)

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config={"temperature": 0.3, "max_output_tokens": 2000}
        )

        print("[DEBUG] call_gemini: Making API call...")
        full_prompt = system_prompt + "\n\n" + prompt if system_prompt else prompt
        response = await asyncio.wait_for(
            model.generate_content_async(full_prompt),
            timeout=60.0
        )
        print("[DEBUG] call_gemini: Got response, parsing...")
        text = response.text

        # Parse JSON if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        print("[DEBUG] call_gemini: Success!")
        return {"response": text, "model": "gemini-2.5-pro", "error": None}
    except asyncio.TimeoutError:
        print("[DEBUG] call_gemini: Timeout!")
        return {"error": "Timeout after 60s", "response": None, "model": "gemini-2.5-pro"}
    except Exception as e:
        print(f"[DEBUG] call_gemini: Error - {e}")
        return {"error": str(e), "response": None, "model": "gemini-2.5-pro"}


async def call_grok(prompt: str, system_prompt: str = "") -> Dict[str, Any]:
    """Call Grok-3."""
    print("[DEBUG] call_grok: Starting...")
    api_key = os.environ.get('XAI_API_KEY')
    if not api_key:
        return {"error": "No API key", "response": None, "model": "grok-3"}

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1", timeout=60.0)

    try:
        print("[DEBUG] call_grok: Making API call...")
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            ),
            timeout=60.0
        )
        print("[DEBUG] call_grok: Got response, parsing...")
        text = response.choices[0].message.content

        # Parse JSON if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        print("[DEBUG] call_grok: Success!")
        return {"response": text, "model": "grok-3", "error": None}
    except asyncio.TimeoutError:
        print("[DEBUG] call_grok: Timeout!")
        return {"error": "Timeout after 60s", "response": None, "model": "grok-3"}
    except Exception as e:
        print(f"[DEBUG] call_grok: Error - {e}")
        return {"error": str(e), "response": None, "model": "grok-3"}


# =============================================================================
# DECISION PROMPT CONSTRUCTION
# =============================================================================

DECISION_SYSTEM_PROMPT = """You are an expert AI safety researcher helping to make decisions about a mechanistic interpretability research project.

You will be presented with a research decision point and context. Your task is to recommend what decision should be made based on the available information.

Be rigorous, scientific, and practical. Consider:
- Experimental validity and statistical power
- Resource efficiency (GPU time, API costs)
- Scientific value of different approaches
- Risk of false positives/negatives

Respond in JSON format:
{
  "decision": "Your recommended decision (be specific)",
  "correct": true or false (based on your prediction),
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence justification"
}"""


def create_decision_prompt(decision: Dict) -> str:
    """Create prompt for a decision."""
    context_str = json.dumps(decision['context'], indent=2)

    prompt = f"""## Research Decision

**Decision Type**: {decision['decision_type']}

**Question**: {decision['question']}

**Available Context**:
```json
{context_str}
```

**What was actually decided**: {decision['decision_made']}

**Your task**: Was this the correct decision? Would you have made the same choice?

Respond with JSON only."""

    return prompt


# =============================================================================
# BASELINE CONDITIONS
# =============================================================================

async def evaluate_single_model(decision: Dict, model_name: str) -> Dict:
    """Baseline 1: Single model evaluation."""
    prompt = create_decision_prompt(decision)

    model_funcs = {
        "claude": call_claude,
        "gpt": call_gpt,
        "gemini": call_gemini,
        "grok": call_grok
    }

    result = await model_funcs[model_name](prompt, DECISION_SYSTEM_PROMPT)

    # Parse response
    try:
        if result['response']:
            parsed = json.loads(result['response'])
            predicted_correct = parsed.get('correct', None)
            confidence = parsed.get('confidence', 0.5)
            reasoning = parsed.get('reasoning', '')
        else:
            predicted_correct = None
            confidence = 0.0
            reasoning = f"Error: {result['error']}"
    except Exception as e:
        predicted_correct = None
        confidence = 0.0
        reasoning = f"Parse error: {str(e)}"

    return {
        "model": model_name,
        "predicted_correct": predicted_correct,
        "actual_correct": decision['ground_truth']['correct'],
        "confidence": confidence,
        "reasoning": reasoning,
        "error": result['error']
    }


async def evaluate_homogeneous_ensemble(decision: Dict) -> Dict:
    """Baseline 2: Homogeneous ensemble (4x Claude Opus 4.5)."""
    prompt = create_decision_prompt(decision)

    # Run 4 instances of Claude in parallel
    tasks = [call_claude(prompt, DECISION_SYSTEM_PROMPT) for _ in range(4)]
    results = await asyncio.gather(*tasks)

    predictions = []
    confidences = []
    reasonings = []

    for i, result in enumerate(results):
        try:
            if result['response']:
                parsed = json.loads(result['response'])
                predictions.append(parsed.get('correct', None))
                confidences.append(parsed.get('confidence', 0.5))
                reasonings.append(f"Claude-{i+1}: {parsed.get('reasoning', '')}")
            else:
                predictions.append(None)
                confidences.append(0.0)
                reasonings.append(f"Claude-{i+1}: Error - {result['error']}")
        except Exception as e:
            predictions.append(None)
            confidences.append(0.0)
            reasonings.append(f"Claude-{i+1}: Parse error - {str(e)}")

    # Consensus: Majority vote
    valid_predictions = [p for p in predictions if p is not None]
    if valid_predictions:
        consensus = sum(valid_predictions) / len(valid_predictions) >= 0.5
        avg_confidence = sum(confidences) / len(confidences)
    else:
        consensus = None
        avg_confidence = 0.0

    return {
        "condition": "homogeneous_4",
        "individual_predictions": predictions,
        "individual_confidences": confidences,
        "consensus_decision": consensus,
        "actual_correct": decision['ground_truth']['correct'],
        "avg_confidence": avg_confidence,
        "reasonings": reasonings
    }


async def evaluate_heterogeneous_consensus(decision: Dict) -> Dict:
    """Baseline 3: Heterogeneous consensus (Claude + GPT + Gemini + Grok)."""
    prompt = create_decision_prompt(decision)

    # Run all 4 models in parallel
    tasks = [
        call_claude(prompt, DECISION_SYSTEM_PROMPT),
        call_gpt(prompt, DECISION_SYSTEM_PROMPT),
        call_gemini(prompt, DECISION_SYSTEM_PROMPT),
        call_grok(prompt, DECISION_SYSTEM_PROMPT)
    ]
    results = await asyncio.gather(*tasks)

    model_names = ["claude", "gpt", "gemini", "grok"]
    predictions = []
    confidences = []
    reasonings = []

    for model_name, result in zip(model_names, results):
        try:
            if result['response']:
                parsed = json.loads(result['response'])
                predictions.append(parsed.get('correct', None))
                confidences.append(parsed.get('confidence', 0.5))
                reasonings.append(f"{model_name}: {parsed.get('reasoning', '')}")
            else:
                predictions.append(None)
                confidences.append(0.0)
                reasonings.append(f"{model_name}: Error - {result['error']}")
        except Exception as e:
            predictions.append(None)
            confidences.append(0.0)
            reasonings.append(f"{model_name}: Parse error - {str(e)}")

    # Consensus: Majority vote among valid predictions
    valid_predictions = [p for p in predictions if p is not None]
    if valid_predictions:
        consensus = sum(valid_predictions) / len(valid_predictions) >= 0.5
        avg_confidence = sum(confidences) / len(confidences)
    else:
        consensus = None
        avg_confidence = 0.0

    return {
        "condition": "heterogeneous_4",
        "models": model_names,
        "individual_predictions": predictions,
        "individual_confidences": confidences,
        "consensus_decision": consensus,
        "actual_correct": decision['ground_truth']['correct'],
        "avg_confidence": avg_confidence,
        "reasonings": reasonings
    }


# =============================================================================
# MAIN RUNNER
# =============================================================================

async def run_pilot_study(decisions_file: str = "pilot_decisions.json",
                         output_file: str = "pilot_results.json"):
    """Run the full pilot study across all baseline conditions."""

    print("="*60)
    print("MULTI-LLM CONSENSUS PILOT STUDY")
    print("="*60)
    print(f"Loading decisions from: {decisions_file}")

    with open(decisions_file, 'r') as f:
        data = json.load(f)

    decisions = data['decisions']
    print(f"Loaded {len(decisions)} decisions")

    results = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "decisions_file": decisions_file,
            "total_decisions": len(decisions)
        },
        "single_model": {},
        "homogeneous_4": [],
        "heterogeneous_4": []
    }

    # Run each decision through all baselines
    for i, decision in enumerate(decisions):
        decision_id = decision['decision_id']
        print(f"\n[{i+1}/{len(decisions)}] Processing {decision_id}: {decision['question'][:60]}...")

        # Single-model baselines
        print("  - Single-model baselines...")
        for model in ["claude", "gpt", "gemini", "grok"]:
            result = await evaluate_single_model(decision, model)
            if decision_id not in results["single_model"]:
                results["single_model"][decision_id] = {}
            results["single_model"][decision_id][model] = result
            status = "✓" if result['predicted_correct'] == result['actual_correct'] else "✗"
            print(f"    {model}: {status} (conf={result['confidence']:.2f})")

        # Homogeneous ensemble
        print("  - Homogeneous-4 (4x Claude)...")
        homo_result = await evaluate_homogeneous_ensemble(decision)
        homo_result['decision_id'] = decision_id
        results["homogeneous_4"].append(homo_result)
        status = "✓" if homo_result['consensus_decision'] == homo_result['actual_correct'] else "✗"
        print(f"    Consensus: {status} (conf={homo_result['avg_confidence']:.2f})")

        # Heterogeneous consensus
        print("  - Heterogeneous-4 (Claude+GPT+Gemini+Grok)...")
        hetero_result = await evaluate_heterogeneous_consensus(decision)
        hetero_result['decision_id'] = decision_id
        results["heterogeneous_4"].append(hetero_result)
        status = "✓" if hetero_result['consensus_decision'] == hetero_result['actual_correct'] else "✗"
        print(f"    Consensus: {status} (conf={hetero_result['avg_confidence']:.2f})")

        # Save after each decision (checkpoint)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("PILOT STUDY COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_file}")

    return results


async def run_single_decision(decision_id: str, decisions_file: str = "pilot_decisions.json"):
    """Run a single decision for testing."""
    print(f"[DEBUG] Loading decisions from {decisions_file}")
    with open(decisions_file, 'r') as f:
        data = json.load(f)

    decision = next(d for d in data['decisions'] if d['decision_id'] == decision_id)

    print(f"Testing {decision_id}: {decision['question']}")
    print("\n[DEBUG] Starting Single-model (Claude)...")
    result = await evaluate_single_model(decision, "claude")
    print(json.dumps(result, indent=2))

    print("\n[DEBUG] Starting Homogeneous-4...")
    result = await evaluate_homogeneous_ensemble(decision)
    print(json.dumps(result, indent=2))

    print("\n[DEBUG] Starting Heterogeneous-4...")
    result = await evaluate_heterogeneous_consensus(decision)
    print(json.dumps(result, indent=2))


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-LLM Consensus Pilot Study")
    parser.add_argument("--test", type=str, help="Test single decision by ID (e.g., D1)")
    parser.add_argument("--decisions", type=str, default="pilot_decisions.json",
                       help="Input decisions file")
    parser.add_argument("--output", type=str, default="pilot_results.json",
                       help="Output results file")
    args = parser.parse_args()

    if args.test:
        asyncio.run(run_single_decision(args.test, args.decisions))
    else:
        asyncio.run(run_pilot_study(args.decisions, args.output))


if __name__ == "__main__":
    main()
