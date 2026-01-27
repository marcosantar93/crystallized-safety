# Crystallized Safety - Project Status

**Date:** 2026-01-21
**Last Updated:** Session with Claude Sonnet 4.5
**Focus:** Activation steering safety vulnerabilities in LLMs

---

## 🎯 Project Scope

**This repository focuses exclusively on:**
- Layer-specific activation steering attacks on LLM safety mechanisms
- Mistral-7B-Instruct vulnerability (83% jailbreak success at Layer 24)
- Validation cycles (probing classifiers, activation patching, multilayered attacks)
- Cross-model comparison (Mistral vs Gemma vs Llama)
- Publication of findings

**NOT in scope for this repository:**
- ❌ Empathy geometry experiments (moved to separate repository)
- ❌ Other tangential research directions

---

## 📊 Current Project Status

### Paper Status
- **Version:** ActivationSteering_CouncilApproved_v4.pdf (12 pages)
- **Publication Readiness:** 9/10 (council-approved)
- **Location:** `papers/`

### Main Finding
**Mistral-7B-Instruct is vulnerable to layer-specific steering attacks:**
- 83% jailbreak success rate at Layer 24 (α=15)
- Maintains output coherence (83% coherent flip rate)
- Demonstrates layer-localized safety mechanisms

### Cross-Model Results
| Model | Best Config | Success Rate | Status |
|-------|-------------|--------------|--------|
| Mistral-7B | L24 α=15 | 83% | Vulnerable |
| Gemma-2-9B | L18 α=15 | 11% | Resistant |
| Llama-3.1-8B | L21 α=20 | 0% | Resistant |

### Validation Status (Cycles 1-3)
These experiments were launched on 2026-01-18 but status is unknown:

| Cycle | Focus | Pod ID | Status |
|-------|-------|--------|--------|
| 1 | Probing Classifiers | 54vxopm2i7r1ab | Unknown |
| 2 | Activation Patching | 51jvkjwuc8ze0t | Unknown |
| 3 | Multilayered Attacks | lrl7nkvf4z1kdj | Unknown |

**Next step:** Check if these completed and retrieve results.

### Cycle 4 (Approved, Not Launched)
- **Focus:** Temporal dynamics & attention head ablation
- **Status:** Council-approved (2/3 reviewers)
- **Budget:** $2.25, 11.5 hours
- **Proposal:** `~/runpod_experiments/cycle4_temporal_dynamics_proposal.md`

---

## 🚫 Empathy Experiments - MOVED TO SEPARATE REPOSITORY

**IMPORTANT:** Empathy geometry research has been branched off to a separate repository.

**Status:** Being worked on elsewhere by the user.

**Do NOT work on empathy experiments in this repository:**
- Empathy prompts
- Empathy Docker images
- Empathy analysis scripts
- Empathy bandwidth measurements

**Files related to empathy (for reference only):**
- `experiments/empathy_experiment_main.py`
- `experiments/analyze_empathy_results.py`
- `experiments/generate_empathy_figures.py`
- `data/empathy_prompts_v1.json`
- `EMPATHY_EXPERIMENT_STATUS.md`
- `experiments/FUTURE_EXPERIMENTS.md`

These files remain for historical context but are **NOT active work items**.

---

## 🎯 Priority Tasks for Next Session

### Focus Areas (Safety Research Only)

#### 1. Check Validation Cycle Results
Cycles 1-3 were launched on Jan 18. Status unknown.

```bash
cd ~/runpod_experiments
python3 -c "
from runpod_graphql_orchestrator import get_pod_status
for pid in ['54vxopm2i7r1ab', '51jvkjwuc8ze0t', 'lrl7nkvf4z1kdj']:
    status = get_pod_status(pid)
    print(f'{pid}: {status}')
"
```

If complete, download and analyze results:
- `cycle1_probing_results.json`
- `cycle2_patching_results.json`
- `cycle3_multilayer_results.json`

#### 2. Update Paper with Results
Once Cycles 1-3 results are available:
- Update paper sections with actual data
- Replace "Status: Running" with "Status: Completed"
- Add result tables
- Update conclusions if breakthrough achieved

#### 3. Launch Cycle 4 (Optional)
If Cycles 1-3 show promising results:
- Implement temporal dynamics experiment
- Launch attention head ablation studies
- Budget: $2.25, ~11.5 hours

#### 4. Prepare for Publication
- Finalize all validation results
- Update paper to v5 with complete validation
- Prepare supplementary materials
- Submit to conference/journal

---

## 💰 Budget Status

- **Spent to date:** ~$3.81
- **Remaining:** ~$221
- **Cycles 1-3:** Unknown if completed/charged
- **Cycle 4 (approved):** $2.25 (not launched)

---

## 📁 Project Structure

### Core Research Code
```
crystallized-safety/
├── pipeline.py                    # Main steering pipeline
├── sweep_experiment.py            # Grid search experiments
├── src/                          # Core modules
│   ├── steering.py
│   ├── extraction.py
│   ├── evaluation.py
│   └── models.py
├── papers/                        # Research papers
│   └── ActivationSteering_CouncilApproved_v4.pdf
├── results/                       # Experimental results
│   ├── mistral_sweep_results.json
│   └── gemma_sweep_results.json
└── experiments/                   # Validation experiments
    ├── (empathy files - NOT active)
    └── (validation cycle files)
```

### External Experiment Runner
```
~/runpod_experiments/              # Separate directory
├── cycle1-3 implementations
├── cycle4 proposal
└── Docker build scripts
```

---

## 🔧 Recent Infrastructure Work (2026-01-21)

### Docker EC2 Builder Integration
Created automated Docker build system using `~/docker_ec2_builder`:

**New files:**
- `build_docker_empathy.py` - Python script using docker_ec2_builder
- `build_docker_empathy.sh` - Shell wrapper
- `BUILD_GUIDE.md` - Comprehensive documentation
- `DOCKER_BUILD_INTEGRATION.md` - Integration summary

**Benefits:**
- Automatic instance selection
- Spot instance support (90% savings)
- Proper SSH/SCP file transfer
- Auto-termination
- Cost: ~$0.02-0.05 per build (vs $0.06-0.10 before)

**Note:** These Docker build tools were created for empathy experiments but can be adapted for safety validation Docker images if needed.

---

## 🎯 Session Focus Reminder

**FOR NEXT SESSIONS:**

✅ **DO focus on:**
- Activation steering vulnerabilities
- Validation cycle results (Cycles 1-4)
- Paper updates and publication
- Mistral/Gemma/Llama comparative analysis
- Safety mechanism bypass techniques

❌ **DO NOT focus on:**
- Empathy experiments (separate repository)
- Empathy Docker builds
- Empathy analysis
- Empathy paper writing

---

## 📚 Key Documentation

### Active (Safety Research)
- `CONTINUE_HERE.md` - Session continuation guide (needs update)
- `SESSION_SUMMARY_2026-01-18.md` - Last session summary
- `VALIDATION_SUMMARY.md` - Validation methodology
- `papers/ActivationSteering_CouncilApproved_v4.pdf` - Current paper

### Reference Only (Spin-off Project)
- `EMPATHY_EXPERIMENT_STATUS.md` - Empathy status (separate repo)
- `experiments/FUTURE_EXPERIMENTS.md` - Future empathy work (separate repo)
- `experiments/REVIEWER_SUMMARY.md` - Empathy council review (separate repo)

### Infrastructure
- `BUILD_GUIDE.md` - Docker build guide
- `DOCKER_BUILD_INTEGRATION.md` - docker_ec2_builder integration

---

## 🔑 Environment Setup

### Required Environment Variables
```bash
export DOCKERHUB_TOKEN='your-token'  # For Docker builds
# AWS credentials via ~/.aws/credentials
```

### Repository Location
```bash
cd ~/paladin_claude/crystallized-safety
```

### External Resources
- RunPod experiments: `~/runpod_experiments/`
- Docker builder: `~/docker_ec2_builder/`

---

## Summary

**This is a focused safety research project** investigating layer-specific vulnerabilities in LLM safety mechanisms through activation steering. The main finding (83% jailbreak success in Mistral-7B) is publication-ready pending validation cycle completion.

**Empathy research has been spun off** to a separate repository and should not be worked on here.

**Next immediate actions:**
1. Check Cycles 1-3 completion status
2. Download and analyze validation results
3. Update paper with findings
4. Decide on Cycle 4 launch
5. Prepare for publication
