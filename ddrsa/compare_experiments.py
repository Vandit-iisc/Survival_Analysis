"""
Compare results across all experiments and generate comparison plots
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse


def load_experiment_results(summary_file):
    """Load experiment results from summary JSON"""
    with open(summary_file, 'r') as f:
        results = json.load(f)
    return results


def plot_metric_comparison(results, metric_name, save_path=None):
    """
    Plot bar chart comparing a specific metric across experiments

    Args:
        results: Dictionary of experiment results
        metric_name: Name of metric to compare
        save_path: Path to save figure
    """
    experiments = []
    values = []

    for exp_name, result in results.items():
        if result['status'] == 'completed' and 'metrics' in result:
            if metric_name in result['metrics']:
                experiments.append(exp_name.replace('hybrid_', '').upper())
                values.append(result['metrics'][metric_name])

    if not experiments:
        print(f"No data available for metric: {metric_name}")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(experiments, values, color=colors[:len(experiments)])

    # Customize plot
    ax.set_ylabel(metric_name.upper().replace('_', ' '), fontsize=12)
    ax.set_title(f'{metric_name.upper().replace("_", " ")} Comparison Across Models', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric_name} comparison to: {save_path}")

    return fig


def plot_all_metrics_comparison(results, save_dir='comparison_plots'):
    """
    Generate comparison plots for all metrics

    Args:
        results: Dictionary of experiment results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get list of all available metrics
    all_metrics = set()
    for result in results.values():
        if result['status'] == 'completed' and 'metrics' in result:
            all_metrics.update(result['metrics'].keys())

    # Important metrics to plot
    priority_metrics = ['nasa_score', 'mse', 'mae', 'rmse', 'mean_error']

    # Plot priority metrics
    for metric in priority_metrics:
        if metric in all_metrics:
            save_path = os.path.join(save_dir, f'{metric}_comparison.png')
            fig = plot_metric_comparison(results, metric, save_path)
            if fig:
                plt.close(fig)


def create_summary_table(results, save_path=None):
    """
    Create a summary table of all experiments

    Args:
        results: Dictionary of experiment results
        save_path: Path to save table as image
    """
    # Extract data
    experiments = []
    nasa_scores = []
    mses = []
    maes = []
    rmses = []
    statuses = []

    for exp_name, result in sorted(results.items()):
        experiments.append(exp_name.replace('hybrid_', '').upper())
        statuses.append(result['status'])

        if result['status'] == 'completed' and 'metrics' in result:
            metrics = result['metrics']
            nasa_scores.append(f"{metrics.get('nasa_score', 0):.2f}")
            mses.append(f"{metrics.get('mse', 0):.4f}")
            maes.append(f"{metrics.get('mae', 0):.4f}")
            rmses.append(f"{metrics.get('rmse', 0):.4f}")
        else:
            nasa_scores.append('N/A')
            mses.append('N/A')
            maes.append('N/A')
            rmses.append('N/A')

    # Create table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for i in range(len(experiments)):
        table_data.append([
            experiments[i],
            statuses[i],
            nasa_scores[i],
            mses[i],
            maes[i],
            rmses[i]
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Status', 'NASA Score', 'MSE', 'MAE', 'RMSE'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.15, 0.17, 0.17, 0.17, 0.17]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    for i in range(1, len(experiments) + 1):
        if statuses[i-1] == 'completed':
            color = '#E7E6E6' if i % 2 == 0 else 'white'
        else:
            color = '#FFB3B3'
        for j in range(6):
            table[(i, j)].set_facecolor(color)

    plt.title('Experiment Results Summary', fontsize=16, fontweight='bold', pad=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary table to: {save_path}")

    return fig


def plot_training_comparison(results, save_path=None):
    """
    Plot training time/epochs comparison if available

    Args:
        results: Dictionary of experiment results
        save_path: Path to save figure
    """
    # This would require storing training time in metrics
    # For now, just a placeholder
    pass


def generate_report(results, output_dir='comparison_plots'):
    """
    Generate comprehensive comparison report

    Args:
        results: Dictionary of experiment results
        output_dir: Directory to save all outputs
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS AND REPORT")
    print("="*80 + "\n")

    # 1. Summary table
    print("1. Creating summary table...")
    fig_table = create_summary_table(
        results,
        save_path=os.path.join(output_dir, 'summary_table.png')
    )
    plt.close(fig_table)

    # 2. Metric comparisons
    print("2. Creating metric comparison plots...")
    plot_all_metrics_comparison(results, save_dir=output_dir)

    # 3. Print text summary
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Status':<12} {'NASA Score':<15} {'MSE':<15} {'RMSE':<15}")
    print("-"*80)

    for exp_name, result in sorted(results.items()):
        model = exp_name.replace('hybrid_', '').upper()
        status = result['status']

        if status == 'completed' and 'metrics' in result:
            metrics = result['metrics']
            nasa = metrics.get('nasa_score', 0)
            mse = metrics.get('mse', 0)
            rmse = metrics.get('rmse', 0)
            print(f"{model:<20} {status:<12} {nasa:<15.2f} {mse:<15.4f} {rmse:<15.4f}")
        else:
            print(f"{model:<20} {status:<12} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

    print("="*80)

    # Find best model
    completed = {k: v for k, v in results.items() if v['status'] == 'completed'}
    if completed:
        best_exp = min(completed.items(),
                      key=lambda x: x[1]['metrics'].get('nasa_score', float('inf')))
        best_model = best_exp[0].replace('hybrid_', '').upper()
        best_score = best_exp[1]['metrics'].get('nasa_score', 0)

        print(f"\n🏆 BEST MODEL: {best_model} (NASA Score: {best_score:.2f})")

    print(f"\n✓ All comparison plots saved to: {output_dir}/")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare results across NASA loss experiments'
    )

    parser.add_argument('--summary-file', type=str,
                       default='nasa_full_experiments_summary.json',
                       help='Path to experiments summary JSON file')
    parser.add_argument('--output-dir', type=str,
                       default='comparison_plots',
                       help='Directory to save comparison plots')

    args = parser.parse_args()

    # Check if summary file exists
    if not os.path.exists(args.summary_file):
        print(f"Error: Summary file not found: {args.summary_file}")
        print("\nAvailable summary files:")
        for f in os.listdir('.'):
            if f.endswith('_summary.json'):
                print(f"  - {f}")
        return

    # Load results
    print(f"Loading results from: {args.summary_file}")
    results = load_experiment_results(args.summary_file)

    # Generate report
    generate_report(results, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
