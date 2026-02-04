#!/usr/bin/env python3
"""Quick council query for budget prioritization"""

import os
import json
import anthropic
import openai
import google.generativeai as genai

# Load env
from dotenv import load_dotenv
load_dotenv()

QUERY = """
## Current State (2026-02-03)

**Confirmed Results:**
- Mistral-7B: 88.6% jailbreak @ L24, α=-15 (n=50, orthogonal control Z=4.74)
- Gemma-2-9B: 20% max with aggressive multi-layer attacks (resistant)
- Yi-1.5-9B: 100% jailbreak with multi-layer
- Qwen-2.5-7B: 90% jailbreak @ [8,12,16,20], α=-15 (n=50)
- Llama-3.1-8B: Pending (disk space issue)

**Budget: $22 USD on Vast.ai**
- RTX 3090: ~$0.15/hr
- A100 40GB: ~$0.50/hr
- Estimated compute: 70-140 GPU hours

**Tier 1 Experiments (required for publication):**
1. EXP-01: Dataset expansion (≥600 prompts) - ~$2
2. EXP-02: Full layer×magnitude sweep (5 models) - ~$15-20
3. EXP-03: Null controls (100 random + 50 orthogonal) - ~$5
4. EXP-05: Triple metric validation (keyword + logprob + LLM judge) - ~$7
5. EXP-06: Capability preservation (MMLU, MT-Bench) - ~$10-15
6. EXP-07: Extraction method sensitivity - ~$5

**Question:** With $22, which experiments should we prioritize to maximize publication readiness? Consider:
- What we already have vs what's missing
- Reviewer requirements
- Scientific validity

Return JSON:
{
  "priority_order": ["EXP-XX", ...],
  "reasoning": "...",
  "estimated_spend": {"EXP-XX": "$Y", ...},
  "total_estimated": "$ZZ",
  "critical_gaps": ["..."],
  "can_skip_for_now": ["..."]
}
"""

def ask_claude():
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": QUERY}]
    )
    return "Claude", response.content[0].text

def ask_gpt():
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": QUERY}],
        max_tokens=1500
    )
    return "GPT-4o", response.choices[0].message.content

def ask_gemini():
    genai.configure(api_key=os.environ["GOOGLE_AI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(QUERY)
    return "Gemini", response.text

def ask_grok():
    client = openai.OpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1"
    )
    response = client.chat.completions.create(
        model="grok-3-fast",
        messages=[{"role": "user", "content": QUERY}],
        max_tokens=1500
    )
    return "Grok", response.choices[0].message.content

if __name__ == "__main__":
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    print("🎯 Querying council on $22 budget prioritization...\n")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(ask_claude),
            executor.submit(ask_gpt),
            executor.submit(ask_gemini),
            executor.submit(ask_grok)
        ]
        
        for future in futures:
            try:
                name, response = future.result(timeout=60)
                print(f"\n{'='*60}")
                print(f"## {name}")
                print(f"{'='*60}")
                print(response)
            except Exception as e:
                print(f"Error: {e}")
