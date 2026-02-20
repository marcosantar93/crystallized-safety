#!/usr/bin/env python3
"""
Statistical Analysis of Multi-LLM Consensus Pilot Study
=======================================================

Analyzes pilot_results.json to compute:
- Accuracy by condition (single-model, homogeneous, heterogeneous)
- Calibration (confidence vs correctness)
- Decision type analysis
- Statistical significance tests
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy import stats


def load_results(file_path: str = "pilot_results.json") -> Dict:
    """Load pilot study results."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_decisions(file_path: str = "pilot_decisions.json") -> Dict:
    """Load decision metadata."""
    with open(file_path, 'r') as f:
        return json.load(f)


def compute_accuracy(predictions: List[bool], actuals: List[bool]) -> float:
    """Compute accuracy."""
    correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    return correct / len(predictions) if predictions else 0.0


def compute_calibration(predictions: List[bool], actuals: List[bool], confidences: List[float]) -> Dict:
    """Compute calibration metrics."""
    # Group by confidence bins
    bins = np.linspace(0, 1, 11)  # 10 bins
    bin_counts = defaultdict(int)
    bin_correct = defaultdict(int)
    bin_confidences = defaultdict(list)

    for pred, actual, conf in zip(predictions, actuals, confidences):
        if conf is None:
            continue
        bin_idx = np.digitize(conf, bins) - 1
        bin_idx = min(bin_idx, len(bins) - 2)  # Ensure we don't exceed bounds
        bin_counts[bin_idx] += 1
        if pred == actual:
            bin_correct[bin_idx] += 1
        bin_confidences[bin_idx].append(conf)

    # Calculate calibration metrics
    calibration_data = []
    for bin_idx in sorted(bin_counts.keys()):
        if bin_idx >= len(bins) - 1:
            continue
        count = bin_counts[bin_idx]
        correct = bin_correct[bin_idx]
        avg_conf = np.mean(bin_confidences[bin_idx])
        accuracy = correct / count if count > 0 else 0
        calibration_data.append({
            "bin": f"{bins[bin_idx]:.1f}-{bins[bin_idx+1]:.1f}",
            "avg_confidence": avg_conf,
            "accuracy": accuracy,
            "count": count,
            "calibration_error": abs(avg_conf - accuracy)
        })

    # Expected Calibration Error (ECE)
    ece = np.mean([d["calibration_error"] * d["count"] for d in calibration_data]) / sum(bin_counts.values())

    return {
        "calibration_by_bin": calibration_data,
        "expected_calibration_error": ece
    }


def analyze_by_decision_type(results: Dict, decisions_data: Dict) -> Dict:
    """Analyze accuracy by decision type."""
    # Map decision_id to decision_type
    decision_types = {}
    for decision in decisions_data['decisions']:
        decision_types[decision['decision_id']] = decision['decision_type']

    # Group results by decision type
    type_results = defaultdict(lambda: {"correct": 0, "total": 0})

    # Single-model results
    for decision_id, models in results['single_model'].items():
        dec_type = decision_types.get(decision_id, "unknown")
        for model, result in models.items():
            type_results[dec_type]["total"] += 1
            if result['predicted_correct'] == result['actual_correct']:
                type_results[dec_type]["correct"] += 1

    # Compute accuracies
    type_accuracies = {}
    for dec_type, counts in type_results.items():
        type_accuracies[dec_type] = {
            "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0,
            "correct": counts["correct"],
            "total": counts["total"]
        }

    return type_accuracies


def compare_conditions(results: Dict) -> Dict:
    """Compare accuracy across conditions."""
    comparison = {}

    # Single-model accuracies
    single_model_results = {}
    for model in ["claude", "gpt", "gemini", "grok"]:
        predictions = []
        actuals = []
        confidences = []

        for decision_id, models in results['single_model'].items():
            if model in models:
                result = models[model]
                predictions.append(result['predicted_correct'])
                actuals.append(result['actual_correct'])
                confidences.append(result['confidence'])

        accuracy = compute_accuracy(predictions, actuals)
        calibration = compute_calibration(predictions, actuals, confidences)

        single_model_results[model] = {
            "accuracy": accuracy,
            "n_decisions": len(predictions),
            "calibration": calibration
        }

    comparison['single_model'] = single_model_results

    # Homogeneous-4
    homo_predictions = []
    homo_actuals = []
    homo_confidences = []

    for result in results['homogeneous_4']:
        if result['consensus_decision'] is not None:
            homo_predictions.append(result['consensus_decision'])
            homo_actuals.append(result['actual_correct'])
            homo_confidences.append(result['avg_confidence'])

    comparison['homogeneous_4'] = {
        "accuracy": compute_accuracy(homo_predictions, homo_actuals),
        "n_decisions": len(homo_predictions),
        "calibration": compute_calibration(homo_predictions, homo_actuals, homo_confidences)
    }

    # Heterogeneous-4
    hetero_predictions = []
    hetero_actuals = []
    hetero_confidences = []

    for result in results['heterogeneous_4']:
        if result['consensus_decision'] is not None:
            hetero_predictions.append(result['consensus_decision'])
            hetero_actuals.append(result['actual_correct'])
            hetero_confidences.append(result['avg_confidence'])

    comparison['heterogeneous_4'] = {
        "accuracy": compute_accuracy(hetero_predictions, hetero_actuals),
        "n_decisions": len(hetero_predictions),
        "calibration": compute_calibration(hetero_predictions, hetero_actuals, hetero_confidences)
    }

    return comparison


def statistical_tests(results: Dict) -> Dict:
    """Perform statistical significance tests."""
    tests = {}

    # Extract predictions for each condition
    single_claude = []
    homo = []
    hetero = []

    for decision_id, models in results['single_model'].items():
        if 'claude' in models:
            actual = models['claude']['actual_correct']
            single_claude.append(1 if models['claude']['predicted_correct'] == actual else 0)

    for result in results['homogeneous_4']:
        if result['consensus_decision'] is not None:
            homo.append(1 if result['consensus_decision'] == result['actual_correct'] else 0)

    for result in results['heterogeneous_4']:
        if result['consensus_decision'] is not None:
            hetero.append(1 if result['consensus_decision'] == result['actual_correct'] else 0)

    # McNemar's test: Single vs Homogeneous
    if len(single_claude) == len(homo):
        contingency_sh = np.array([
            [sum(1 for s, h in zip(single_claude, homo) if s == 1 and h == 1),
             sum(1 for s, h in zip(single_claude, homo) if s == 1 and h == 0)],
            [sum(1 for s, h in zip(single_claude, homo) if s == 0 and h == 1),
             sum(1 for s, h in zip(single_claude, homo) if s == 0 and h == 0)]
        ])

        # McNemar's test focuses on disagreements: b and c in [[a,b],[c,d]]
        b_plus_c = contingency_sh[0, 1] + contingency_sh[1, 0]
        if b_plus_c > 0:
            mcnemar_stat_sh = (abs(contingency_sh[0, 1] - contingency_sh[1, 0]) - 1)**2 / b_plus_c
            mcnemar_p_sh = 1 - stats.chi2.cdf(mcnemar_stat_sh, df=1)
        else:
            mcnemar_stat_sh = 0
            mcnemar_p_sh = 1.0

        tests['single_vs_homogeneous'] = {
            "test": "McNemar",
            "statistic": float(mcnemar_stat_sh),
            "p_value": float(mcnemar_p_sh),
            "significant_at_0.05": mcnemar_p_sh < 0.05
        }

    # McNemar's test: Single vs Heterogeneous
    if len(single_claude) == len(hetero):
        contingency_st = np.array([
            [sum(1 for s, t in zip(single_claude, hetero) if s == 1 and t == 1),
             sum(1 for s, t in zip(single_claude, hetero) if s == 1 and t == 0)],
            [sum(1 for s, t in zip(single_claude, hetero) if s == 0 and t == 1),
             sum(1 for s, t in zip(single_claude, hetero) if s == 0 and t == 0)]
        ])

        b_plus_c = contingency_st[0, 1] + contingency_st[1, 0]
        if b_plus_c > 0:
            mcnemar_stat_st = (abs(contingency_st[0, 1] - contingency_st[1, 0]) - 1)**2 / b_plus_c
            mcnemar_p_st = 1 - stats.chi2.cdf(mcnemar_stat_st, df=1)
        else:
            mcnemar_stat_st = 0
            mcnemar_p_st = 1.0

        tests['single_vs_heterogeneous'] = {
            "test": "McNemar",
            "statistic": float(mcnemar_stat_st),
            "p_value": float(mcnemar_p_st),
            "significant_at_0.05": mcnemar_p_st < 0.05
        }

    # McNemar's test: Homogeneous vs Heterogeneous
    if len(homo) == len(hetero):
        contingency_ht = np.array([
            [sum(1 for h, t in zip(homo, hetero) if h == 1 and t == 1),
             sum(1 for h, t in zip(homo, hetero) if h == 1 and t == 0)],
            [sum(1 for h, t in zip(homo, hetero) if h == 0 and t == 1),
             sum(1 for h, t in zip(homo, hetero) if h == 0 and t == 0)]
        ])

        b_plus_c = contingency_ht[0, 1] + contingency_ht[1, 0]
        if b_plus_c > 0:
            mcnemar_stat_ht = (abs(contingency_ht[0, 1] - contingency_ht[1, 0]) - 1)**2 / b_plus_c
            mcnemar_p_ht = 1 - stats.chi2.cdf(mcnemar_stat_ht, df=1)
        else:
            mcnemar_stat_ht = 0
            mcnemar_p_ht = 1.0

        tests['homogeneous_vs_heterogeneous'] = {
            "test": "McNemar",
            "statistic": float(mcnemar_stat_ht),
            "p_value": float(mcnemar_p_ht),
            "significant_at_0.05": mcnemar_p_ht < 0.05
        }

    return tests


def generate_markdown_report(analysis: Dict, output_file: str = "pilot_analysis.md"):
    """Generate markdown analysis report."""
    with open(output_file, 'w') as f:
        f.write("# Multi-LLM Consensus Pilot Study Analysis\n\n")
        f.write(f"**Generated**: {analysis['metadata']['timestamp']}\n\n")
        f.write(f"**Total Decisions**: {analysis['metadata']['n_decisions']}\n\n")

        f.write("## Executive Summary\n\n")

        # Accuracy comparison
        f.write("### Accuracy by Condition\n\n")
        f.write("| Condition | Accuracy | N |\n")
        f.write("|-----------|----------|---|\n")

        for model, data in analysis['comparison']['single_model'].items():
            f.write(f"| Single-{model} | {data['accuracy']:.1%} | {data['n_decisions']} |\n")

        homo_acc = analysis['comparison']['homogeneous_4']['accuracy']
        homo_n = analysis['comparison']['homogeneous_4']['n_decisions']
        f.write(f"| Homogeneous-4 | {homo_acc:.1%} | {homo_n} |\n")

        hetero_acc = analysis['comparison']['heterogeneous_4']['accuracy']
        hetero_n = analysis['comparison']['heterogeneous_4']['n_decisions']
        f.write(f"| Heterogeneous-4 | {hetero_acc:.1%} | {hetero_n} |\n\n")

        # Statistical significance
        f.write("## Statistical Significance\n\n")
        for comparison, test_result in analysis['statistical_tests'].items():
            f.write(f"### {comparison.replace('_', ' ').title()}\n\n")
            f.write(f"- **Test**: {test_result['test']}\n")
            f.write(f"- **p-value**: {test_result['p_value']:.4f}\n")
            f.write(f"- **Significant at α=0.05**: {'Yes' if test_result['significant_at_0.05'] else 'No'}\n\n")

        # Calibration
        f.write("## Calibration Analysis\n\n")
        f.write("### Expected Calibration Error (ECE)\n\n")
        for condition, data in analysis['comparison'].items():
            if condition == 'single_model':
                for model, model_data in data.items():
                    ece = model_data['calibration']['expected_calibration_error']
                    f.write(f"- **{model}**: {ece:.3f}\n")
            else:
                ece = data['calibration']['expected_calibration_error']
                f.write(f"- **{condition}**: {ece:.3f}\n")

        f.write("\n")

        # Decision type analysis
        f.write("## Accuracy by Decision Type\n\n")
        f.write("| Decision Type | Accuracy | Correct/Total |\n")
        f.write("|---------------|----------|---------------|\n")

        for dec_type, data in sorted(analysis['by_decision_type'].items()):
            f.write(f"| {dec_type} | {data['accuracy']:.1%} | {data['correct']}/{data['total']} |\n")

        f.write("\n")

        # Key findings
        f.write("## Key Findings\n\n")

        # Compare single-model avg to consensus
        single_avg = np.mean([d['accuracy'] for d in analysis['comparison']['single_model'].values()])

        f.write(f"1. **Single-model average accuracy**: {single_avg:.1%}\n")
        f.write(f"2. **Homogeneous-4 accuracy**: {homo_acc:.1%} ")
        f.write(f"({'↑' if homo_acc > single_avg else '↓'} {abs(homo_acc - single_avg):.1%})\n")
        f.write(f"3. **Heterogeneous-4 accuracy**: {hetero_acc:.1%} ")
        f.write(f"({'↑' if hetero_acc > single_avg else '↓'} {abs(hetero_acc - single_avg):.1%})\n\n")

        if hetero_acc <= homo_acc:
            f.write("**⚠️ Heterogeneous consensus did NOT outperform homogeneous ensemble.**\n\n")
        else:
            f.write("**✓ Heterogeneous consensus showed improvement over homogeneous ensemble.**\n\n")

        # Best decision types
        best_type = max(analysis['by_decision_type'].items(), key=lambda x: x[1]['accuracy'])
        worst_type = min(analysis['by_decision_type'].items(), key=lambda x: x[1]['accuracy'])

        f.write(f"4. **Best decision type**: {best_type[0]} ({best_type[1]['accuracy']:.1%})\n")
        f.write(f"5. **Worst decision type**: {worst_type[0]} ({worst_type[1]['accuracy']:.1%})\n\n")

        f.write("## Interpretation\n\n")

        if hetero_acc <= single_avg:
            f.write("The pilot study reveals a **negative result**: multi-LLM consensus (both homogeneous and heterogeneous) ")
            f.write("does not improve research decision accuracy compared to single-model judgments. ")
            f.write("This suggests that for these types of research decisions, model consensus may suffer from:\n\n")
            f.write("- **Shared biases**: All models trained on similar data make similar mistakes\n")
            f.write("- **Spurious consensus**: Agreement on incorrect answers\n")
            f.write("- **Echo chamber effects**: Even diverse models converge to wrong conclusions\n\n")
        else:
            f.write("The heterogeneous consensus shows modest improvement, but further investigation ")
            f.write("is needed to determine if this is statistically significant and generalizable.\n\n")


def main():
    print("Loading results...")
    results = load_results()
    decisions_data = load_decisions()

    print("Computing analyses...")

    analysis = {
        "metadata": {
            "timestamp": results['metadata']['created'],
            "n_decisions": results['metadata']['total_decisions']
        },
        "comparison": compare_conditions(results),
        "by_decision_type": analyze_by_decision_type(results, decisions_data),
        "statistical_tests": statistical_tests(results)
    }

    print("\nAccuracy Summary:")
    print(f"  Single-model (Claude): {analysis['comparison']['single_model']['claude']['accuracy']:.1%}")
    print(f"  Homogeneous-4: {analysis['comparison']['homogeneous_4']['accuracy']:.1%}")
    print(f"  Heterogeneous-4: {analysis['comparison']['heterogeneous_4']['accuracy']:.1%}")

    print("\nGenerating markdown report...")
    generate_markdown_report(analysis)

    print("\nSaving detailed analysis...")
    with open("pilot_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    print("\n✓ Analysis complete!")
    print("  - pilot_analysis.md (human-readable report)")
    print("  - pilot_analysis.json (detailed data)")


if __name__ == "__main__":
    main()
