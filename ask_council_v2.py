#!/usr/bin/env python3
"""
Ask Council for Next Steps (v2)
===============================

Multi-model council with latest models and optional deep research mode.

Models:
  - Claude Opus 4.5 (with extended thinking)
  - GPT-5.2 (with reasoning)
  - Gemini 3
  - Grok 3

Usage:
    # Standard council review
    python ask_council_v2.py "Your question or proposal"
    
    # Deep research mode (uses o3-deep-research, takes 10-30 min)
    python ask_council_v2.py --deep-research "Research topic"
    
    # With custom hypothesis file
    python ask_council_v2.py --file proposal.md

Environment:
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    GOOGLE_AI_API_KEY (or GEMINI_API_KEY)
    XAI_API_KEY
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter

# =============================================================================
# MODEL CONFIGURATION (Updated Feb 2026)
# =============================================================================

MODELS = {
    "claude": {
        "name": "Claude Opus 4.5",
        "model_id": "claude-opus-4-5-20250115",
        "perspective": "mechanistic interpretability",
        "supports_thinking": True,
    },
    "gpt": {
        "name": "GPT-5.2",
        "model_id": "gpt-5.2",
        "perspective": "security and adversarial robustness",
        "supports_reasoning": True,
    },
    "gemini": {
        "name": "Gemini 3",
        "model_id": "gemini-3-flash",
        "perspective": "theoretical foundations",
    },
    "grok": {
        "name": "Grok 3",
        "model_id": "grok-3-fast",
        "perspective": "experimental methods and statistics",
    },
}

DEEP_RESEARCH_MODEL = "o3-deep-research"
DEEP_RESEARCH_MODEL_FAST = "o4-mini-deep-research"

# =============================================================================
# PROMPTS
# =============================================================================

REVIEW_SYSTEM_PROMPT = """You are an expert AI safety researcher reviewing mechanistic interpretability experiments.

Your role: {perspective} reviewer

Be rigorous but constructive. Focus on scientific validity and reproducibility.
Provide specific, actionable feedback."""

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
  "critique": "Your detailed analysis (2-3 paragraphs)",
  "required_controls": ["list of needed experiments or controls"],
  "confidence": 0.0-1.0,
  "proceed": true | false,
  "next_steps": ["recommended next steps in priority order"],
  "publication_readiness": "ready" | "needs_work" | "not_ready",
  "key_concerns": ["main issues to address"]
}}

JSON only:"""

DEEP_RESEARCH_PROMPT = """You are conducting deep research for an AI safety project.

## Research Task
{query}

## Context
{context}

## Requirements
- Cite all sources with URLs
- Focus on peer-reviewed papers and reputable sources
- Include specific figures, statistics, and methodology details
- Analyze implications for AI safety research
- Identify gaps in current literature
- Suggest novel experimental approaches

Produce a comprehensive research report with inline citations."""


# =============================================================================
# COUNCIL REVIEWERS
# =============================================================================

async def review_with_claude(results: dict, hypothesis: str, perspective: str, use_thinking: bool = True) -> dict:
    """Claude Opus 4.5 review with optional extended thinking."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return {"verdict": "YELLOW", "critique": "API key not configured", "error": "No ANTHROPIC_API_KEY"}

    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=api_key)

    messages = [{"role": "user", "content": REVIEW_USER_PROMPT.format(
        hypothesis=hypothesis,
        results=json.dumps(results, indent=2, default=str)[:12000],
        perspective=perspective
    )}]

    kwargs = {
        "model": MODELS["claude"]["model_id"],
        "max_tokens": 16000,
        "system": REVIEW_SYSTEM_PROMPT.format(perspective=perspective),
        "messages": messages,
    }

    # Enable extended thinking for complex analysis
    if use_thinking:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": 8000
        }

    response = await client.messages.create(**kwargs)

    # Extract text from response
    text = ""
    for block in response.content:
        if hasattr(block, 'text'):
            text = block.text
            break

    # Parse JSON
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    return json.loads(text)


async def review_with_gpt(results: dict, hypothesis: str, perspective: str, use_reasoning: bool = True) -> dict:
    """GPT-5.2 review with optional reasoning."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return {"verdict": "YELLOW", "critique": "API key not configured", "error": "No OPENAI_API_KEY"}

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    kwargs = {
        "model": MODELS["gpt"]["model_id"],
        "messages": [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT.format(perspective=perspective)},
            {"role": "user", "content": REVIEW_USER_PROMPT.format(
                hypothesis=hypothesis,
                results=json.dumps(results, indent=2, default=str)[:12000],
                perspective=perspective
            )}
        ],
        "max_tokens": 8000,
        "response_format": {"type": "json_object"}
    }

    # Enable reasoning for complex analysis
    if use_reasoning:
        kwargs["reasoning"] = {"effort": "medium"}

    response = await client.chat.completions.create(**kwargs)
    return json.loads(response.choices[0].message.content)


async def review_with_gemini(results: dict, hypothesis: str, perspective: str) -> dict:
    """Gemini 3 review."""
    api_key = os.environ.get('GOOGLE_AI_API_KEY') or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return {"verdict": "YELLOW", "critique": "API key not configured", "error": "No GOOGLE_AI_API_KEY"}

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    prompt = REVIEW_SYSTEM_PROMPT.format(perspective=perspective) + "\n\n" + REVIEW_USER_PROMPT.format(
        hypothesis=hypothesis,
        results=json.dumps(results, indent=2, default=str)[:12000],
        perspective=perspective
    )

    response = await client.aio.models.generate_content(
        model=MODELS["gemini"]["model_id"],
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8000,
            response_mime_type="application/json"
        )
    )

    return json.loads(response.text)


async def review_with_grok(results: dict, hypothesis: str, perspective: str) -> dict:
    """Grok 3 review."""
    api_key = os.environ.get('XAI_API_KEY')
    if not api_key:
        return {"verdict": "YELLOW", "critique": "API key not configured", "error": "No XAI_API_KEY"}

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    response = await client.chat.completions.create(
        model=MODELS["grok"]["model_id"],
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT.format(perspective=perspective)},
            {"role": "user", "content": REVIEW_USER_PROMPT.format(
                hypothesis=hypothesis,
                results=json.dumps(results, indent=2, default=str)[:12000],
                perspective=perspective
            )}
        ],
        temperature=0.3,
        max_tokens=8000
    )

    text = response.choices[0].message.content
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text)


# =============================================================================
# DEEP RESEARCH
# =============================================================================

async def run_deep_research(query: str, context: str = "", fast: bool = False) -> dict:
    """Run deep research using o3-deep-research or o4-mini-deep-research."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return {"error": "No OPENAI_API_KEY", "report": None}

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, timeout=3600)  # 1 hour timeout

    model = DEEP_RESEARCH_MODEL_FAST if fast else DEEP_RESEARCH_MODEL
    
    print(f"\n🔬 Starting deep research with {model}...")
    print("   This may take 10-30 minutes. Running in background mode.")
    print()

    input_text = DEEP_RESEARCH_PROMPT.format(query=query, context=context)

    try:
        response = await client.responses.create(
            model=model,
            input=input_text,
            background=True,
            tools=[
                {"type": "web_search_preview"},
                {"type": "code_interpreter", "container": {"type": "auto"}}
            ],
        )

        # Poll for completion
        while response.status == "in_progress":
            print(f"   Status: {response.status}...")
            await asyncio.sleep(30)
            response = await client.responses.retrieve(response.id)

        # Extract report
        report_text = ""
        sources = []
        
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if hasattr(content, 'text'):
                        report_text = content.text
                    if hasattr(content, 'annotations'):
                        sources.extend([{
                            "url": a.url,
                            "title": getattr(a, 'title', 'Unknown')
                        } for a in content.annotations])

        return {
            "report": report_text,
            "sources": sources,
            "model": model,
            "response_id": response.id
        }

    except Exception as e:
        return {"error": str(e), "report": None}


# =============================================================================
# RESULTS LOADING
# =============================================================================

def load_results():
    """Load existing experimental results."""
    results_dir = Path("results")

    data = {
        "project": "Crystallized Safety",
        "status": "Validation phase",
        "completed_experiments": [],
        "model_results": {},
    }

    # Load all result files
    for result_file in results_dir.glob("**/*.json"):
        try:
            with open(result_file) as f:
                content = json.load(f)
            
            name = result_file.stem
            data["model_results"][name] = content
            data["completed_experiments"].append(name)
        except:
            pass

    # Summary stats
    data["summary"] = {
        "total_experiments": len(data["completed_experiments"]),
        "models_tested": ["Llama-3.1-8B", "Mistral-7B", "Qwen-2.5-7B", "Gemma-2-9B"],
    }

    return data


# =============================================================================
# MAIN
# =============================================================================

async def run_council(hypothesis: str, results: dict = None, use_thinking: bool = True):
    """Run the full council review."""
    
    if results is None:
        results = load_results()

    print("=" * 70)
    print("COUNCIL REVIEW (v2 - Latest Models)")
    print("=" * 70)
    print()
    print("Models:")
    for key, config in MODELS.items():
        extras = []
        if config.get("supports_thinking"):
            extras.append("extended thinking")
        if config.get("supports_reasoning"):
            extras.append("reasoning")
        extra_str = f" ({', '.join(extras)})" if extras else ""
        print(f"  • {config['name']}: {config['model_id']}{extra_str}")
    print()

    # Run reviews in parallel
    reviewers = [
        ("Claude", review_with_claude, MODELS["claude"]["perspective"], use_thinking),
        ("GPT-5.2", review_with_gpt, MODELS["gpt"]["perspective"], use_thinking),
        ("Gemini", review_with_gemini, MODELS["gemini"]["perspective"], False),
        ("Grok", review_with_grok, MODELS["grok"]["perspective"], False),
    ]

    reviews = {}
    tasks = []

    for name, func, perspective, needs_extra in reviewers:
        print(f"🤖 Asking {name} ({perspective})...")
        if needs_extra and name == "Claude":
            tasks.append((name, func(results, hypothesis, perspective, use_thinking=True)))
        elif needs_extra and name == "GPT-5.2":
            tasks.append((name, func(results, hypothesis, perspective, use_reasoning=True)))
        else:
            tasks.append((name, func(results, hypothesis, perspective)))

    # Gather results
    for name, task in tasks:
        try:
            review = await task
            reviews[name] = review
            verdict = review.get('verdict', 'ERROR')
            print(f"   ✅ {name}: {verdict}")
        except Exception as e:
            reviews[name] = {"verdict": "ERROR", "error": str(e)}
            print(f"   ❌ {name}: Error - {e}")

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
            action = "🟢 PROCEED_TO_PAPER"
            emoji = "✅"
        elif any(v == "RED" for v in verdicts):
            action = "🔴 HALT_AND_REVIEW"
            emoji = "🛑"
        else:
            action = "🟡 RUN_FOLLOWUP"
            emoji = "⚠️"

        print(f"{emoji} Action: {action}")
        print(f"📊 Average confidence: {avg_confidence:.1%}")
        print()

    # Display reviews
    print("Individual Reviews:")
    print("-" * 70)

    all_next_steps = []
    all_concerns = []

    for name, review in reviews.items():
        verdict = review.get("verdict", "ERROR")
        emoji = "✅" if verdict == "GREEN" else "⚠️" if verdict == "YELLOW" else "❌"
        print(f"\n{emoji} {name}: {verdict}")

        if review.get("critique"):
            print(f"   {review['critique'][:600]}...")

        if review.get("next_steps"):
            all_next_steps.extend(review["next_steps"])
        
        if review.get("key_concerns"):
            all_concerns.extend(review["key_concerns"])

        if review.get("required_controls"):
            print(f"   Required: {', '.join(review['required_controls'][:3])}")

    # Aggregate recommendations
    print()
    print("=" * 70)
    print("AGGREGATED RECOMMENDATIONS")
    print("=" * 70)

    step_counts = Counter(all_next_steps)
    print("\n📋 Next Steps (by priority):")
    for i, (step, count) in enumerate(step_counts.most_common(10), 1):
        print(f"   {i}. {step} (×{count})")

    if all_concerns:
        concern_counts = Counter(all_concerns)
        print("\n⚠️ Key Concerns:")
        for concern, count in concern_counts.most_common(5):
            print(f"   • {concern} (×{count})")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "models_used": {k: v["model_id"] for k, v in MODELS.items()},
        "hypothesis": hypothesis,
        "reviews": reviews,
        "consensus": {
            "action": action if verdicts else "INCOMPLETE",
            "avg_confidence": avg_confidence if confidences else 0,
            "verdicts": verdicts
        },
        "next_steps": [s for s, _ in step_counts.most_common(10)],
        "concerns": [c for c, _ in Counter(all_concerns).most_common(5)] if all_concerns else []
    }

    output_file = Path("council_recommendations_v2.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print(f"💾 Results saved to: {output_file}")

    return output


async def main():
    parser = argparse.ArgumentParser(description="Ask the AI Council for research guidance")
    parser.add_argument("query", nargs="?", help="Your question or hypothesis")
    parser.add_argument("--deep-research", "-d", action="store_true", 
                       help="Use deep research mode (o3-deep-research, 10-30 min)")
    parser.add_argument("--fast", "-f", action="store_true",
                       help="Use faster deep research model (o4-mini)")
    parser.add_argument("--file", "-F", type=str, help="Load hypothesis from file")
    parser.add_argument("--no-thinking", action="store_true",
                       help="Disable extended thinking/reasoning")
    
    args = parser.parse_args()

    # Get query
    if args.file:
        with open(args.file) as f:
            query = f.read()
    elif args.query:
        query = args.query
    else:
        # Default: load results and ask for review
        query = """
Review the current state of the Crystallized Safety project:

FINDINGS:
- Llama-3.1-8B: 100% jailbreak @ L20 α=-15
- Mistral-7B: 89% jailbreak @ L24 α=-15  
- Qwen-2.5-7B: 40% jailbreak @ multi-layer
- Gemma-2-9B: 0% jailbreak (all configs tested)

KEY RESULTS:
- Orthogonal control: Llama +74pp, Mistral +72pp (direction-specific)
- Capability preservation: 0% MMLU degradation under steering
- Gemma resistance: Tested 9 multi-layer configs, all 0%

QUESTIONS:
1. Is the Gemma resistance finding publishable as "crystallized safety"?
2. What additional controls are needed for scientific validity?
3. What is the recommended path to publication?
"""

    # Run deep research or council
    if args.deep_research:
        print("=" * 70)
        print("DEEP RESEARCH MODE")
        print("=" * 70)
        
        context = json.dumps(load_results(), indent=2, default=str)[:5000]
        result = await run_deep_research(query, context, fast=args.fast)
        
        if result.get("report"):
            print("\n📄 RESEARCH REPORT")
            print("=" * 70)
            print(result["report"][:10000])
            
            if result.get("sources"):
                print("\n📚 SOURCES")
                for src in result["sources"][:20]:
                    print(f"  • {src.get('title', 'Unknown')}: {src.get('url', '')}")
            
            # Save
            with open("deep_research_report.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Full report saved to: deep_research_report.json")
        else:
            print(f"❌ Error: {result.get('error')}")
    else:
        results = load_results()
        await run_council(query, results, use_thinking=not args.no_thinking)


if __name__ == "__main__":
    asyncio.run(main())
