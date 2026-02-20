# Multi-LLM Consensus for Research Decision Validation
## Empirical Study Design Document

### Executive Summary

This document formalizes our automated research pipeline into an empirically rigorous academic study. Based on literature review, we have identified a novel research gap: **no existing work combines heterogeneous multi-LLM consensus (different companies) with research decisions and ground truth validation from experiments.**

---

## 1. GAP ANALYSIS (Confirmed)

### What EXISTS in Literature:
- **X-MAS (2025)**: Heterogeneous LLM multi-agent systems for benchmarks (not research decisions)
- **ChemMAS (2025)**: Research decisions with validation, but single model internally
- **Aleks (2025)**: Multi-agent research automation, but no explicit consensus debate
- **AI-Scientist (2024-2025)**: Automated research, but single-model architecture

### What's MISSING (Our Niche):
A system that:
1. Combines **multiple different LLMs** (Claude/GPT-4/Gemini/Grok) in **true consensus**
2. Applied to **research decisions** (not just QA benchmarks)
3. With **ground truth validation** from actual experiment execution
4. Systematically comparing heterogeneous vs homogeneous consensus

### Novelty Claim:
"First empirical study of heterogeneous multi-LLM consensus for AI safety research decisions with experimental ground truth validation"

---

## 2. RESEARCH QUESTIONS

### Primary:
**RQ1**: Does multi-model consensus improve research decision validity compared to single-model judgment?

### Secondary:
**RQ2**: For which types of research decisions does consensus help most? (hypothesis, design, interpretation, novelty)

**RQ3**: Does model diversity (different companies) reduce shared biases more than homogeneous ensembles?

**RQ4**: What is the optimal consensus protocol (rounds, voting weights, quorum requirements)?

---

## 3. DECISION TYPE TAXONOMY

| Decision Type | Example from Our Project | Ground Truth Source |
|--------------|--------------------------|---------------------|
| **Hypothesis Validity** | "Does Layer 21 contain safety direction?" | Experiment results (flip rate) |
| **Experimental Design** | "Should we test positive alpha?" | Outcome success |
| **Result Interpretation** | "Is 0% flip rate a null result?" | Human expert consensus |
| **Parameter Selection** | "What layer to test next?" | Experiment efficiency |
| **Code Correctness** | "Is the pipeline bug-free?" | Execution success |
| **Novelty Assessment** | "Has this been done before?" | Literature search |

---

## 4. EXPERIMENTAL DESIGN

### 4.1 Baselines (following ReConcile methodology)
1. **Single-model**: Same decisions made by each model individually
2. **Homogeneous-4**: 4 instances of Claude Opus 4.5 (same model)
3. **Heterogeneous-4**: Claude + GPT-4 + Gemini + Grok (our system)
4. **Random**: For calibration

### 4.2 Metrics
- **Accuracy**: Ground truth match rate (where available)
- **Calibration**: Confidence vs correctness correlation
- **Consensus Speed**: Rounds to converge
- **Diversity Score**: Distribution of initial responses
- **Flip Detection**: Did consensus catch errors single-model missed?

### 4.3 Sample Size
- Pilot: 20 decisions (from existing refusal direction project)
- Main study: 100+ decisions (across multiple research topics)
- Power analysis: n=100 provides 80% power to detect 15% accuracy difference

---

## 5. PILOT STUDY: RETRODICTION ON EXISTING PROJECT

### Decisions from Refusal Direction Research

| ID | Decision Point | What We Decided | Actual Outcome | Correct? |
|----|---------------|-----------------|----------------|----------|
| D1 | "Test Gemma Layer 21 first?" | Yes | 0% flip (null) | ? |
| D2 | "Use negative alpha?" | Yes | All negative alpha → RED Control 1 | Wrong |
| D3 | "Try positive alpha?" | Eventually yes | GREEN Control 1 | Correct |
| D4 | "Is 0% flip rate methodological failure?" | TBD | Model is robust | ? |
| D5 | "Continue with more layers?" | Yes | Running L10-L27 | TBD |

### Retrospective Protocol:
1. Present each decision point to all 4 models independently
2. Run consensus protocol
3. Compare consensus decision to actual outcome
4. Score accuracy

---

## 6. CONSENSUS PROTOCOL SPECIFICATION

```python
class ConsensusProtocol:
    agents = [
        "claude-opus-4-5-20251101",  # Anthropic
        "gpt-4o",                     # OpenAI
        "gemini-2.5-pro",             # Google
        "grok-3",                     # xAI
    ]

    perspectives = {
        "claude": "mechanistic interpretability",
        "gpt": "security analysis",
        "gemini": "theoretical validity",
        "grok": "methodological rigor"
    }

    round_structure = {
        "round_1": "Independent parallel generation (no cross-contamination)",
        "round_2": "Share grouped answers + confidences, request revision",
        "round_3": "Final vote with justification"
    }

    consensus_rules = {
        "unanimous_green": "PROCEED",
        "any_red": "HALT_AND_REVIEW",
        "mixed_yellow": "INVESTIGATE",
        "quorum": 3  # Minimum reviewers needed
    }

    tiebreaker = "Flag for human review"
```

---

## 7. DEFENSES AGAINST KNOWN FAILURE MODES

### 7.1 Spurious Consensus (Du et al. concern)
- **Risk**: All models share web training data biases
- **Defense**: Include xAI's Grok (different training data), add literature verification

### 7.2 Cascade Effects (Liang et al. finding)
- **Risk**: Models anchor to first responses they see
- **Defense**: Round 1 is strictly parallel/independent

### 7.3 Unfair Judging (Liang et al.)
- **Risk**: If one model judges, it may favor similar models
- **Defense**: Voting mechanism, no single judge

### 7.4 Echo Chamber (Estornell & Liu)
- **Risk**: Convergence to shared errors
- **Defense**: Preserve and log dissenting opinions

---

## 8. DATA LOGGING SCHEMA

```json
{
  "decision_id": "uuid",
  "timestamp": "2026-01-15T15:00:00Z",
  "decision_type": "hypothesis_validity",

  "context": {
    "question": "Is Layer 21 α=+2.0 a safety vulnerability?",
    "supporting_data": ["control1_result", "control2_result", "control3_result"],
    "prior_decisions": ["D1", "D2"]
  },

  "round_1_responses": [
    {"agent": "claude", "response": "RED", "confidence": 0.85, "reasoning": "..."},
    {"agent": "gpt", "response": "YELLOW", "confidence": 0.0, "reasoning": "API error"},
    {"agent": "gemini", "response": "YELLOW", "confidence": 0.85, "reasoning": "..."},
    {"agent": "grok", "response": "YELLOW", "confidence": 0.6, "reasoning": "..."}
  ],

  "consensus": {
    "reached": true,
    "value": "HALT_AND_REVIEW",
    "rounds_needed": 1,
    "dissenting_opinions": ["Claude raised mechanistic concerns"]
  },

  "validation": {
    "ground_truth": "Gemma safety is robust (0% flip)",
    "ground_truth_source": "experiment",
    "consensus_was_correct": true
  }
}
```

---

## 9. RISK ASSESSMENT

### Top 3 Scientific Validity Risks:

**Risk 1: Ground Truth Ambiguity**
- Some research decisions don't have clear right/wrong answers
- **Mitigation**: Focus on decisions with objective outcomes (experiment results, literature existence)

**Risk 2: API Key Failures**
- GPT-5.2 currently returns YELLOW due to missing API key
- **Mitigation**: Fix all API keys, implement retry logic, require 3/4 quorum

**Risk 3: Confirmation Bias in Retrodiction**
- We know outcomes when designing retrodiction test
- **Mitigation**: Pre-register decision criteria, have external reviewer score

---

## 10. PAPER OPTIONS

### Option A: Methods Paper
**Title**: "Multi-Model Consensus for Research Decision Validation: An Empirical Study"
- Venue: NeurIPS/ICML Workshop, COLM, EMNLP
- Focus: Protocol design and empirical comparison

### Option B: Negative Result
**Title**: "When Does Multi-LLM Consensus Fail? Lessons from AI Safety Research Automation"
- Follow ICLR 2025 Blogpost framing
- Valuable if heterogeneous consensus doesn't outperform single-model

### Option C: Combined Finding + Methods
**Title**: "Robust Safety Representations in Gemma-2-9B: Discovery via Multi-Agent Research Automation"
- Substantive safety finding + methodology contribution
- Strongest option if both components are solid

---

## 11. TIMELINE

### Week 1-2: Pilot Study
- Fix all API keys
- Run retrodiction on 20 existing decisions
- Compute preliminary accuracy metrics

### Week 3-4: Protocol Refinement
- Analyze pilot results
- Refine consensus protocol based on failure modes observed
- Add any missing defenses

### Week 5-8: Main Study
- Collect 100+ decisions across multiple research projects
- Run baselines (single-model, homogeneous, random)
- Compute final metrics

### Week 9-10: Analysis & Writing
- Statistical analysis
- Paper draft

---

## 12. IMMEDIATE NEXT STEPS

1. **Fix GPT-4o API key** - Currently returning YELLOW errors
2. **Complete running experiments** (A5, Apos5) for more ground truth
3. **Create pilot decision dataset** from existing project
4. **Implement baseline conditions** (single-model, homogeneous)
5. **Set up logging schema** for systematic data collection

---

## References

- Du et al. (2023) "Improving Factuality and Reasoning through Multiagent Debate"
- Chen et al. (2024) "ReConcile: Multi-MODEL Consensus Framework"
- Liang et al. (2024) "Encouraging Divergent Thinking in LLM Debate"
- Sakana AI Scientist (2024, 2025)
- ICLR 2025 Blogpost "Multi-LLM-Agents Debate Limitations"
- Estornell & Liu (2024) "Interventions for Debate Failure Modes"
