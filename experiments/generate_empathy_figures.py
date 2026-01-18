#!/usr/bin/env python3
"""
Generate Empathy Geometry Figures
Creates figure specifications and ASCII visualizations
"""

import json
from pathlib import Path
from datetime import datetime

def load_analysis():
    """Load the latest analysis file"""
    results_dir = Path(__file__).parent / "results" / "empathy"
    analysis_files = sorted(results_dir.glob("analysis_*.json"))

    if not analysis_files:
        print("❌ No analysis file found!")
        return None

    with open(analysis_files[-1]) as f:
        return json.load(f)

def create_ascii_bar_chart(rankings, metric_key, metric_name, width=60):
    """Create ASCII bar chart"""
    values = [r[metric_key] for r in rankings]
    max_val = max(values)

    lines = []
    lines.append(f"\n{metric_name} by Model:")
    lines.append("=" * (width + 20))

    for r in rankings:
        val = r[metric_key]
        bar_len = int((val / max_val) * width)
        bar = "█" * bar_len
        lines.append(f"{r['model']:20} │{bar} {val:.1f}")

    lines.append("=" * (width + 20))

    return "\n".join(lines)

def create_ranking_table(rankings):
    """Create formatted ranking table"""
    lines = []
    lines.append("\nModel Rankings Table:")
    lines.append("="*100)
    lines.append(f"{'Rank':<6} {'Model':<20} {'Bandwidth':<12} {'Dim':<6} {'Range':<8} {'AUROC':<8} {'Transfer':<10} {'SAE':<5}")
    lines.append("-"*100)

    for r in rankings:
        lines.append(
            f"{r['rank']:<6} "
            f"{r['model']:<20} "
            f"{r['bandwidth']:<12.1f} "
            f"{r['dimensionality']:<6} "
            f"{r['steering_range']:<8.1f} "
            f"{r['probe_auroc']:<8.3f} "
            f"{r['transfer_success']*100:<10.1f}% "
            f"{'✓' if r['sae_agreement'] else '✗':<5}"
        )

    lines.append("="*100)

    return "\n".join(lines)

def create_scatter_plot_ascii(rankings):
    """Create ASCII scatter plot of dimensionality vs steering range"""
    dims = [r['dimensionality'] for r in rankings]
    ranges = [r['steering_range'] for r in rankings]

    min_dim, max_dim = min(dims), max(dims)
    min_range, max_range = min(ranges), max(ranges)

    # Create 20x50 grid
    height, width = 20, 50
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Plot points
    for i, r in enumerate(rankings):
        x = int(((r['dimensionality'] - min_dim) / (max_dim - min_dim)) * (width - 1))
        y = height - 1 - int(((r['steering_range'] - min_range) / (max_range - min_range)) * (height - 1))

        # Use model number as marker
        grid[y][x] = str(i + 1)

    lines = []
    lines.append("\nDimensionality vs Steering Range:")
    lines.append("="*60)
    lines.append(f"Range │")

    for row in grid:
        lines.append(f"{' '*6}│{''.join(row)}")

    lines.append(f"{' '*6}└{'─'*width}")
    lines.append(f"{' '*8}Dimensionality")
    lines.append("")

    for i, r in enumerate(rankings, 1):
        lines.append(f"  {i}. {r['model']} ({r['dimensionality']}, {r['steering_range']:.1f})")

    lines.append("="*60)

    return "\n".join(lines)

def create_comparison_chart(rankings, stats):
    """Create empathy vs control bandwidth comparison"""
    lines = []
    lines.append("\nEmpathy Bandwidth vs Control Baseline:")
    lines.append("="*80)

    max_bw = max([r['bandwidth'] for r in rankings] + [r['control_bandwidth'] for r in rankings])

    for r in rankings:
        emp_len = int((r['bandwidth'] / max_bw) * 50)
        ctrl_len = int((r['control_bandwidth'] / max_bw) * 50)

        lines.append(f"\n{r['model']}:")
        lines.append(f"  Empathy:  │{'█'*emp_len} {r['bandwidth']:.1f}")
        lines.append(f"  Control:  │{'░'*ctrl_len} {r['control_bandwidth']:.1f}")
        lines.append(f"  Ratio:    │{r['bandwidth']/r['control_bandwidth']:.2f}x")

    avg_emp = stats['bandwidth']['mean']
    avg_ctrl = stats['control_bandwidth']['mean']

    lines.append("\n" + "-"*80)
    lines.append(f"Average empathy bandwidth: {avg_emp:.1f}")
    lines.append(f"Average control bandwidth: {avg_ctrl:.1f}")
    lines.append(f"Overall ratio: {avg_emp/avg_ctrl:.2f}x")
    lines.append("="*80)

    return "\n".join(lines)

def create_figure_specs(rankings, stats, effect_sizes):
    """Create matplotlib figure specifications for later rendering"""

    specs = {
        "figure_1_bandwidth_ranking": {
            "type": "horizontal_bar",
            "title": "Empathetic Bandwidth by Model",
            "xlabel": "Bandwidth (dimensionality × steering range)",
            "ylabel": "Model",
            "data": [
                {
                    "label": r['model'],
                    "value": r['bandwidth']
                }
                for r in rankings
            ]
        },
        "figure_2_dimensionality": {
            "type": "bar",
            "title": "Subspace Dimensionality (PCA Rank)",
            "xlabel": "Model",
            "ylabel": "Effective Dimensionality",
            "data": [
                {
                    "label": r['model'],
                    "value": r['dimensionality']
                }
                for r in rankings
            ]
        },
        "figure_3_steering_range": {
            "type": "bar",
            "title": "Steering Range (Max α)",
            "xlabel": "Model",
            "ylabel": "Maximum Steering Coefficient",
            "data": [
                {
                    "label": r['model'],
                    "value": r['steering_range']
                }
                for r in rankings
            ]
        },
        "figure_4_scatter": {
            "type": "scatter",
            "title": "Dimensionality vs Steering Range",
            "xlabel": "Effective Dimensionality",
            "ylabel": "Steering Range",
            "data": [
                {
                    "label": r['model'],
                    "x": r['dimensionality'],
                    "y": r['steering_range'],
                    "size": r['bandwidth']
                }
                for r in rankings
            ]
        },
        "figure_5_control_comparison": {
            "type": "grouped_bar",
            "title": "Empathy vs Control Bandwidth",
            "xlabel": "Model",
            "ylabel": "Bandwidth",
            "data": [
                {
                    "model": r['model'],
                    "empathy": r['bandwidth'],
                    "control": r['control_bandwidth']
                }
                for r in rankings
            ]
        },
        "figure_6_probe_auroc": {
            "type": "bar",
            "title": "Linear Probe Performance (AUROC)",
            "xlabel": "Model",
            "ylabel": "AUROC",
            "data": [
                {
                    "label": r['model'],
                    "value": r['probe_auroc']
                }
                for r in rankings
            ]
        },
        "figure_7_transfer_success": {
            "type": "bar",
            "title": "Cross-Context Transfer Success",
            "xlabel": "Model",
            "ylabel": "Transfer Success Rate",
            "data": [
                {
                    "label": r['model'],
                    "value": r['transfer_success'] * 100
                }
                for r in rankings
            ]
        }
    }

    return specs

def main():
    print("="*80)
    print("EMPATHY GEOMETRY - FIGURE GENERATION")
    print("="*80)
    print()

    # Load analysis
    analysis = load_analysis()
    if not analysis:
        return

    rankings = analysis['rankings']
    stats = analysis['statistics']
    effect_sizes = analysis['effect_sizes']

    print(f"Loaded analysis with {len(rankings)} models")
    print()

    # Generate ASCII visualizations
    print("Generating ASCII visualizations...")
    print()

    # Figure 1: Bandwidth ranking
    print(create_ascii_bar_chart(rankings, 'bandwidth', 'Empathetic Bandwidth'))
    print()

    # Figure 2: Dimensionality
    print(create_ascii_bar_chart(rankings, 'dimensionality', 'Subspace Dimensionality'))
    print()

    # Figure 3: Steering range
    print(create_ascii_bar_chart(rankings, 'steering_range', 'Steering Range'))
    print()

    # Figure 4: Scatter plot
    print(create_scatter_plot_ascii(rankings))
    print()

    # Figure 5: Empathy vs control
    print(create_comparison_chart(rankings, stats))
    print()

    # Table
    print(create_ranking_table(rankings))
    print()

    # Generate figure specifications
    print("Generating matplotlib figure specifications...")
    specs = create_figure_specs(rankings, stats, effect_sizes)

    # Save specifications
    results_dir = Path(__file__).parent / "results" / "empathy"
    specs_file = results_dir / f"figure_specs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(specs_file, 'w') as f:
        json.dump(specs, f, indent=2)

    print(f"✅ Figure specifications saved to: {specs_file}")
    print()
    print("Next step:")
    print("  Create PDF report: python3 create_empathy_report.py")
    print()

if __name__ == "__main__":
    main()
