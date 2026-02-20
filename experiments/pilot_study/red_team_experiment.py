#!/usr/bin/env python3
"""
Red Team Experiment: Adversarial Minority Protocol
===================================================

Tests whether adding a 5th "adversarial minority" agent can catch
the 3 unanimous failure cases (D1, D3, D16) where all 4 models were wrong.

Based on Gemini's recommendation from consensus query.
"""

import os
import json
import asyncio
from datetime import datetime
from baselines import call_claude, call_gpt, call_gemini, call_grok, DECISION_SYSTEM_PROMPT


ADVERSARIAL_SYSTEM_PROMPT = """You are the ADVERSARIAL MINORITY reviewer.

Your unique role: Challenge the emerging consensus. Question fundamental assumptions.
Propose alternative framings. Identify blind spots in conventional thinking.

The other reviewers will provide their consensus. Your job is to:
1. Identify what assumption they might ALL be missing
2. Question the premise of the question itself
3. Suggest non-incremental alternatives
4. Flag when "standard practice" might be wrong

Be contrarian but rigorous. Your goal is to catch groupthink, not to be difficult.
"""


async def red_team_decision(decision: dict) -> dict:
    """Run decision through 4-model consensus + adversarial 5th agent."""

    decision_id = decision['decision_id']
    question = decision['question']
    context_str = json.dumps(decision['context'], indent=2)

    prompt = f"""## Research Decision

**Decision Type**: {decision['decision_type']}

**Question**: {question}

**Available Context**:
```json
{context_str}
```

**What was actually decided**: {decision['decision_made']}

**Your task**: Was this the correct decision? Would you have made the same choice?

Respond with JSON only:
{{
  "decision": "Your recommended decision (be specific)",
  "correct": true or false (based on your prediction),
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence justification"
}}"""

    print(f"\n{'='*80}")
    print(f"RED TEAM: {decision_id}")
    print(f"{'='*80}")
    print(f"Question: {question[:100]}...")

    # Phase 1: Get standard 4-model consensus
    print("\n[Phase 1] Standard 4-model consensus...")

    responses = await asyncio.gather(
        call_claude(prompt, DECISION_SYSTEM_PROMPT),
        call_gpt(prompt, DECISION_SYSTEM_PROMPT),
        call_gemini(prompt, DECISION_SYSTEM_PROMPT),
        call_grok(prompt, DECISION_SYSTEM_PROMPT),
        return_exceptions=True
    )

    model_names = ["Claude", "GPT", "Gemini", "Grok"]
    predictions = []
    confidences = []
    reasonings = []

    for name, result in zip(model_names, responses):
        if isinstance(result, Exception) or result.get('error'):
            print(f"  ⚠️  {name}: Error")
            predictions.append(None)
            confidences.append(0.0)
            reasonings.append(f"{name}: Error")
        else:
            try:
                parsed = json.loads(result['response'])
                pred = parsed.get('correct', None)
                conf = parsed.get('confidence', 0.5)
                reason = parsed.get('reasoning', '')

                predictions.append(pred)
                confidences.append(conf)
                reasonings.append(f"{name}: {reason}")

                status = "✓" if pred == decision['ground_truth']['correct'] else "✗"
                print(f"  {name:10s}: {str(pred):5s} (conf={conf:.2f}) {status}")

            except Exception as e:
                print(f"  ⚠️  {name}: Parse error")
                predictions.append(None)
                confidences.append(0.0)
                reasonings.append(f"{name}: Parse error")

    # Calculate standard consensus
    valid_preds = [p for p in predictions if p is not None]
    if valid_preds:
        standard_consensus = sum(valid_preds) / len(valid_preds) >= 0.5
        unanimity = len(set(valid_preds)) == 1
        avg_confidence = sum(confidences) / len([c for c in confidences if c > 0])
    else:
        standard_consensus = None
        unanimity = False
        avg_confidence = 0.0

    print(f"\n  Standard Consensus: {standard_consensus} (conf={avg_confidence:.2f})")
    print(f"  Unanimous: {unanimity}")

    # Phase 2: Adversarial minority
    print(f"\n[Phase 2] Adversarial minority challenge...")

    consensus_summary = {
        "consensus_decision": standard_consensus,
        "unanimity": unanimity,
        "confidence": avg_confidence,
        "individual_reasoning": reasonings
    }

    adversarial_prompt = f"""{prompt}

## ADVERSARIAL REVIEW

The other 4 models reached the following consensus:
- **Consensus decision**: {standard_consensus}
- **Unanimity**: {'YES - all 4 agreed' if unanimity else 'NO - split vote'}
- **Confidence**: {avg_confidence:.0%}

Their reasoning:
{chr(10).join(['- ' + r for r in reasonings])}

## YOUR ADVERSARIAL TASK

Challenge this consensus. Specifically:
1. What assumption might all 4 models be missing?
2. Is there a fundamental premise that should be questioned?
3. Could "standard practice" be wrong here?
4. What non-incremental alternative exists?

Respond with JSON:
{{
  "agree_with_consensus": true/false,
  "your_decision": "What you think is correct",
  "correct": true/false,
  "confidence": 0.0-1.0,
  "adversarial_reasoning": "What are they missing? Why might consensus be wrong?",
  "red_flags": ["List of concerns with consensus reasoning"]
}}"""

    adversarial_result = await call_claude(adversarial_prompt, ADVERSARIAL_SYSTEM_PROMPT)

    adversarial_response = None
    if not adversarial_result.get('error') and adversarial_result.get('response'):
        try:
            adversarial_response = json.loads(adversarial_result['response'])

            adv_pred = adversarial_response.get('correct', None)
            adv_conf = adversarial_response.get('confidence', 0.5)
            agrees = adversarial_response.get('agree_with_consensus', None)

            status = "✓" if adv_pred == decision['ground_truth']['correct'] else "✗"

            print(f"  Adversarial: {str(adv_pred):5s} (conf={adv_conf:.2f}) {status}")
            print(f"  Agrees with consensus: {agrees}")

            if not agrees:
                print(f"  🚨 DISSENT! Adversarial reasoning:")
                print(f"     {adversarial_response.get('adversarial_reasoning', 'N/A')[:150]}...")

                if adversarial_response.get('red_flags'):
                    print(f"  🚩 Red flags:")
                    for flag in adversarial_response['red_flags'][:3]:
                        print(f"     - {flag}")

        except Exception as e:
            print(f"  ⚠️  Adversarial: Parse error - {e}")
    else:
        print(f"  ⚠️  Adversarial: Error - {adversarial_result.get('error', 'Unknown')}")

    # Evaluate outcomes
    ground_truth = decision['ground_truth']['correct']

    standard_correct = standard_consensus == ground_truth if standard_consensus is not None else False
    adversarial_correct = (adversarial_response.get('correct') == ground_truth
                          if adversarial_response else False)

    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"{'='*80}")
    print(f"Ground truth: {ground_truth}")
    print(f"Standard consensus: {standard_consensus} {'✓ CORRECT' if standard_correct else '✗ WRONG'}")
    print(f"Adversarial prediction: {adversarial_response.get('correct') if adversarial_response else 'N/A'} "
          f"{'✓ CORRECT' if adversarial_correct else '✗ WRONG'}")

    # Key question: Did adversarial catch a standard consensus failure?
    caught_failure = (not standard_correct) and adversarial_correct

    if caught_failure:
        print(f"\n🎯 SUCCESS: Adversarial agent caught the consensus failure!")
    elif not standard_correct:
        print(f"\n❌ FAILURE: Adversarial agent also got it wrong")
    else:
        print(f"\n✅ No failure to catch (standard consensus was correct)")

    return {
        "decision_id": decision_id,
        "ground_truth": ground_truth,
        "standard_consensus": {
            "prediction": standard_consensus,
            "correct": standard_correct,
            "unanimity": unanimity,
            "confidence": avg_confidence,
            "individual_predictions": predictions,
            "individual_confidences": confidences,
            "reasonings": reasonings
        },
        "adversarial": {
            "prediction": adversarial_response.get('correct') if adversarial_response else None,
            "correct": adversarial_correct,
            "agrees_with_consensus": adversarial_response.get('agree_with_consensus') if adversarial_response else None,
            "confidence": adversarial_response.get('confidence') if adversarial_response else 0.0,
            "reasoning": adversarial_response.get('adversarial_reasoning') if adversarial_response else None,
            "red_flags": adversarial_response.get('red_flags', []) if adversarial_response else [],
            "full_response": adversarial_response
        },
        "caught_failure": caught_failure
    }


async def run_red_team_experiment():
    """Run red team experiment on the 3 failure cases."""

    print("="*80)
    print("RED TEAM EXPERIMENT: Adversarial Minority Protocol")
    print("="*80)
    print("\nTesting whether adversarial 5th agent can catch unanimous failures\n")

    # Load decisions
    with open('pilot_decisions.json', 'r') as f:
        decisions_data = json.load(f)

    # The 3 failure cases
    failure_ids = ['D1', 'D3', 'D16']
    failure_decisions = [d for d in decisions_data['decisions'] if d['decision_id'] in failure_ids]

    print(f"Testing on {len(failure_decisions)} failure cases: {failure_ids}\n")

    results = []
    for decision in failure_decisions:
        result = await red_team_decision(decision)
        results.append(result)

        # Save after each
        with open('red_team_results.json', 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "n_decisions": len(results),
                "results": results
            }, f, indent=2)

    # Final analysis
    print("\n" + "="*80)
    print("RED TEAM EXPERIMENT: FINAL RESULTS")
    print("="*80)

    total = len(results)
    caught = sum(1 for r in results if r['caught_failure'])
    standard_wrong = sum(1 for r in results if not r['standard_consensus']['correct'])
    adversarial_right = sum(1 for r in results if r['adversarial']['correct'])

    print(f"\n📊 Performance:")
    print(f"   Total failure cases tested: {total}")
    print(f"   Standard consensus wrong: {standard_wrong}/{total}")
    print(f"   Adversarial agent correct: {adversarial_right}/{total}")
    print(f"   Adversarial CAUGHT failures: {caught}/{standard_wrong}")

    catch_rate = caught / standard_wrong if standard_wrong > 0 else 0

    print(f"\n🎯 Catch Rate: {catch_rate:.0%}")

    if catch_rate >= 0.33:  # Caught 1+ of 3
        print(f"\n✅ SUCCESS: Adversarial minority protocol shows promise!")
        print(f"   Recommendation: Include in paper as mitigation strategy")
    else:
        print(f"\n❌ FAILURE: Adversarial minority did not improve accuracy")
        print(f"   Recommendation: Document as limitation, proceed with n=100 scale-up")

    print(f"\n💾 Results saved to: red_team_results.json")

    return results


async def main():
    results = await run_red_team_experiment()

    print("\n" + "="*80)
    print("Next steps based on results:")
    print("="*80)

    catch_rate = sum(1 for r in results if r['caught_failure']) / len([r for r in results if not r['standard_consensus']['correct']])

    if catch_rate >= 0.33:
        print("""
✅ RED TEAM WORKED!

Next steps:
1. Write up protocol for paper (Methods §3.6)
2. Update results with adversarial catch rate
3. Submit to ArXiv with red team validation
4. Target NeurIPS workshop (complete story: problem + solution)
        """)
    else:
        print("""
❌ RED TEAM FAILED

Next steps:
1. Document as limitation in paper (Discussion §5.4)
2. Proceed with n=100 scale-up (focus on strong categories)
3. Frame weak categories as "boundary conditions"
4. Consider alternative mitigation strategies
        """)


if __name__ == "__main__":
    asyncio.run(main())
