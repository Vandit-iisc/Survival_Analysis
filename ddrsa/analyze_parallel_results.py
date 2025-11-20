"""
Analyze and visualize results from parallel hyperparameter experiments
Generates comparison plots and summary tables
"""

import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_results(output_dir):
    """Load all experiment results"""

    # Load manifest
    manifest_path = os.path.join(output_dir, 'experiment_manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Load results
    results_path = os.path.join(output_dir, 'all_results.json')
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Load individual experiment metrics
    data = []
    for result in results:
        if result['status'] != 'SUCCESS':
            continue

        exp_name = result['exp_name']
        config = result['config']

        # Try to load test metrics
        metrics_path = os.path.join(output_dir, 'logs', exp_name, 'test_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            data.append({
                'exp_name': exp_name,
                'dataset': config['dataset'],
                'model_variant': config['variant'],
                'batch_size': config['batch_size'],
                'learning_rate': config['learning_rate'],
                'lambda_param': config['lambda_param'],
                'nasa_weight': config['nasa_weight'],
                'dropout': config['dropout'],
                'duration_min': result['duration'] / 60,
                **metrics
            })

    return pd.DataFrame(data), manifest


def create_summary_table(df, output_dir):
    """Create summary statistics table"""

    summary = df.groupby(['dataset', 'model_variant']).agg({
        'rul_mae': ['mean', 'std', 'min', 'max'],
        'rul_rmse': ['mean', 'std', 'min', 'max'],
        'concordance_index': ['mean', 'std', 'min', 'max'],
        'duration_min': ['mean', 'std']
    }).round(4)

    # Save to CSV
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary.to_csv(summary_path)
    print(f"Summary statistics saved to: {summary_path}")

    return summary


def plot_hyperparameter_effects(df, output_dir):
    """Plot the effect of each hyperparameter on performance"""

    os.makedirs(os.path.join(output_dir, 'analysis_plots'), exist_ok=True)

    metrics = ['rul_mae', 'rul_rmse', 'concordance_index']
    hyperparams = ['batch_size', 'learning_rate', 'lambda_param', 'nasa_weight', 'dropout']

    for metric in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, hyperparam in enumerate(hyperparams):
            ax = axes[i]

            # Group by hyperparameter and compute mean/std
            grouped = df.groupby(hyperparam)[metric].agg(['mean', 'std', 'count'])

            ax.errorbar(
                grouped.index,
                grouped['mean'],
                yerr=grouped['std'],
                marker='o',
                capsize=5,
                capthick=2,
                linewidth=2,
                markersize=8
            )

            ax.set_xlabel(hyperparam.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').upper(), fontsize=12)
            ax.set_title(f'{metric.upper()} vs {hyperparam.replace("_", " ").title()}', fontsize=14)
            ax.grid(True, alpha=0.3)

            # Add count annotations
            for x, y, count in zip(grouped.index, grouped['mean'], grouped['count']):
                ax.annotate(f'n={count}', (x, y), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=8)

        # Remove extra subplot
        fig.delaxes(axes[5])

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'analysis_plots', f'{metric}_vs_hyperparameters.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")


def plot_model_comparison(df, output_dir):
    """Compare different model variants"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ['rul_mae', 'rul_rmse', 'concordance_index']

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Box plot for each model variant
        df.boxplot(column=metric, by='model_variant', ax=ax)
        ax.set_xlabel('Model Variant', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} by Model Variant', fontsize=14)
        ax.get_figure().suptitle('')  # Remove default title
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'analysis_plots', 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def plot_batch_size_vs_performance(df, output_dir):
    """Analyze batch size impact on performance and training time"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Batch size vs MAE
    ax = axes[0, 0]
    for model in df['model_variant'].unique():
        model_df = df[df['model_variant'] == model]
        grouped = model_df.groupby('batch_size')['rul_mae'].agg(['mean', 'std'])
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   marker='o', label=model, capsize=5, linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('RUL MAE', fontsize=12)
    ax.set_title('Batch Size vs RUL MAE', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Batch size vs RMSE
    ax = axes[0, 1]
    for model in df['model_variant'].unique():
        model_df = df[df['model_variant'] == model]
        grouped = model_df.groupby('batch_size')['rul_rmse'].agg(['mean', 'std'])
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   marker='o', label=model, capsize=5, linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('RUL RMSE', fontsize=12)
    ax.set_title('Batch Size vs RUL RMSE', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Batch size vs C-Index
    ax = axes[1, 0]
    for model in df['model_variant'].unique():
        model_df = df[df['model_variant'] == model]
        grouped = model_df.groupby('batch_size')['concordance_index'].agg(['mean', 'std'])
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   marker='o', label=model, capsize=5, linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Concordance Index', fontsize=12)
    ax.set_title('Batch Size vs Concordance Index', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Batch size vs Training Time
    ax = axes[1, 1]
    for model in df['model_variant'].unique():
        model_df = df[df['model_variant'] == model]
        grouped = model_df.groupby('batch_size')['duration_min'].agg(['mean', 'std'])
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   marker='o', label=model, capsize=5, linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Training Time (minutes)', fontsize=12)
    ax.set_title('Batch Size vs Training Time', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'analysis_plots', 'batch_size_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def plot_learning_rate_analysis(df, output_dir):
    """Analyze learning rate impact"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ['rul_mae', 'rul_rmse', 'concordance_index']

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for model in df['model_variant'].unique():
            model_df = df[df['model_variant'] == model]
            grouped = model_df.groupby('learning_rate')[metric].agg(['mean', 'std'])
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                       marker='o', label=model, capsize=5, linewidth=2)

        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').upper(), fontsize=12)
        ax.set_title(f'Learning Rate vs {metric.upper()}', fontsize=14)
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'analysis_plots', 'learning_rate_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def plot_nasa_loss_impact(df, output_dir):
    """Analyze NASA loss weight impact"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    metrics = ['rul_mae', 'rul_rmse', 'concordance_index', 'nasa_score']
    titles = ['RUL MAE', 'RUL RMSE', 'Concordance Index', 'NASA Score']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i // 2, i % 2]

        for model in df['model_variant'].unique():
            model_df = df[df['model_variant'] == model]
            grouped = model_df.groupby('nasa_weight')[metric].agg(['mean', 'std'])
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                       marker='o', label=model, capsize=5, linewidth=2)

        ax.set_xlabel('NASA Loss Weight', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'NASA Weight vs {title}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'analysis_plots', 'nasa_loss_impact.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def find_best_configurations(df, output_dir):
    """Find and save best hyperparameter configurations for each model"""

    best_configs = []

    for model in df['model_variant'].unique():
        model_df = df[df['model_variant'] == model]

        # Find best based on different criteria
        best_mae = model_df.loc[model_df['rul_mae'].idxmin()]
        best_rmse = model_df.loc[model_df['rul_rmse'].idxmin()]
        best_cindex = model_df.loc[model_df['concordance_index'].idxmax()]
        best_nasa = model_df.loc[model_df['nasa_score'].idxmin()]

        best_configs.append({
            'model': model,
            'criterion': 'MAE',
            'exp_name': best_mae['exp_name'],
            'batch_size': best_mae['batch_size'],
            'learning_rate': best_mae['learning_rate'],
            'lambda_param': best_mae['lambda_param'],
            'nasa_weight': best_mae['nasa_weight'],
            'dropout': best_mae['dropout'],
            'rul_mae': best_mae['rul_mae'],
            'rul_rmse': best_mae['rul_rmse'],
            'concordance_index': best_mae['concordance_index']
        })

        best_configs.append({
            'model': model,
            'criterion': 'C-Index',
            'exp_name': best_cindex['exp_name'],
            'batch_size': best_cindex['batch_size'],
            'learning_rate': best_cindex['learning_rate'],
            'lambda_param': best_cindex['lambda_param'],
            'nasa_weight': best_cindex['nasa_weight'],
            'dropout': best_cindex['dropout'],
            'rul_mae': best_cindex['rul_mae'],
            'rul_rmse': best_cindex['rul_rmse'],
            'concordance_index': best_cindex['concordance_index']
        })

    best_df = pd.DataFrame(best_configs)
    best_path = os.path.join(output_dir, 'best_configurations.csv')
    best_df.to_csv(best_path, index=False)
    print(f"Best configurations saved to: {best_path}")

    return best_df


def create_heatmap(df, output_dir):
    """Create heatmaps showing interaction between hyperparameters"""

    # Heatmap: Batch size vs Learning rate (averaged over other params)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # MAE heatmap
    pivot_mae = df.pivot_table(
        values='rul_mae',
        index='batch_size',
        columns='learning_rate',
        aggfunc='mean'
    )
    sns.heatmap(pivot_mae, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[0])
    axes[0].set_title('RUL MAE: Batch Size vs Learning Rate', fontsize=14)
    axes[0].set_xlabel('Learning Rate', fontsize=12)
    axes[0].set_ylabel('Batch Size', fontsize=12)

    # C-Index heatmap
    pivot_cindex = df.pivot_table(
        values='concordance_index',
        index='batch_size',
        columns='learning_rate',
        aggfunc='mean'
    )
    sns.heatmap(pivot_cindex, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1])
    axes[1].set_title('Concordance Index: Batch Size vs Learning Rate', fontsize=14)
    axes[1].set_xlabel('Learning Rate', fontsize=12)
    axes[1].set_ylabel('Batch Size', fontsize=12)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'analysis_plots', 'hyperparameter_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def plot_lambda_analysis(df, output_dir):
    """Analyze lambda parameter impact on performance"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    metrics = ['rul_mae', 'rul_rmse', 'concordance_index', 'duration_min']
    titles = ['RUL MAE', 'RUL RMSE', 'Concordance Index', 'Training Time (min)']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i // 2, i % 2]

        for model in df['model_variant'].unique():
            model_df = df[df['model_variant'] == model]
            grouped = model_df.groupby('lambda_param')[metric].agg(['mean', 'std'])
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                       marker='o', label=model, capsize=5, linewidth=2)

        ax.set_xlabel('Lambda Parameter', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'Lambda vs {title}', fontsize=14)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'analysis_plots', 'lambda_parameter_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def plot_dropout_analysis(df, output_dir):
    """Analyze dropout rate impact on performance"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    metrics = ['rul_mae', 'rul_rmse', 'concordance_index', 'duration_min']
    titles = ['RUL MAE', 'RUL RMSE', 'Concordance Index', 'Training Time (min)']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i // 2, i % 2]

        for model in df['model_variant'].unique():
            model_df = df[df['model_variant'] == model]
            grouped = model_df.groupby('dropout')[metric].agg(['mean', 'std'])
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                       marker='o', label=model, capsize=5, linewidth=2)

        ax.set_xlabel('Dropout Rate', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'Dropout vs {title}', fontsize=14)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'analysis_plots', 'dropout_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def plot_dataset_comparison(df, output_dir):
    """Compare model performance across different datasets"""

    datasets = df['dataset'].unique()
    if len(datasets) < 2:
        print("  Skipping dataset comparison (only one dataset)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ['rul_mae', 'rul_rmse', 'concordance_index']

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Create grouped bar plot
        model_variants = df['model_variant'].unique()
        x = np.arange(len(model_variants))
        width = 0.35

        for j, dataset in enumerate(datasets):
            dataset_df = df[df['dataset'] == dataset]
            means = [dataset_df[dataset_df['model_variant'] == model][metric].mean()
                    for model in model_variants]
            stds = [dataset_df[dataset_df['model_variant'] == model][metric].std()
                   for model in model_variants]

            ax.bar(x + j * width, means, width, yerr=stds,
                  label=dataset, capsize=5, alpha=0.8)

        ax.set_xlabel('Model Variant', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} by Dataset', fontsize=14)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(model_variants, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'analysis_plots', 'dataset_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def create_comprehensive_heatmaps(df, output_dir):
    """Create additional heatmaps for all hyperparameter pairs"""

    # Lambda vs NASA weight
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    pivot_mae = df.pivot_table(
        values='rul_mae',
        index='lambda_param',
        columns='nasa_weight',
        aggfunc='mean'
    )
    sns.heatmap(pivot_mae, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[0])
    axes[0].set_title('RUL MAE: Lambda vs NASA Weight', fontsize=14)
    axes[0].set_xlabel('NASA Weight', fontsize=12)
    axes[0].set_ylabel('Lambda Parameter', fontsize=12)

    pivot_cindex = df.pivot_table(
        values='concordance_index',
        index='lambda_param',
        columns='nasa_weight',
        aggfunc='mean'
    )
    sns.heatmap(pivot_cindex, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1])
    axes[1].set_title('Concordance Index: Lambda vs NASA Weight', fontsize=14)
    axes[1].set_xlabel('NASA Weight', fontsize=12)
    axes[1].set_ylabel('Lambda Parameter', fontsize=12)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'analysis_plots', 'lambda_nasa_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

    # Dropout vs Learning rate
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    pivot_mae = df.pivot_table(
        values='rul_mae',
        index='dropout',
        columns='learning_rate',
        aggfunc='mean'
    )
    sns.heatmap(pivot_mae, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[0])
    axes[0].set_title('RUL MAE: Dropout vs Learning Rate', fontsize=14)
    axes[0].set_xlabel('Learning Rate', fontsize=12)
    axes[0].set_ylabel('Dropout Rate', fontsize=12)

    pivot_cindex = df.pivot_table(
        values='concordance_index',
        index='dropout',
        columns='learning_rate',
        aggfunc='mean'
    )
    sns.heatmap(pivot_cindex, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1])
    axes[1].set_title('Concordance Index: Dropout vs Learning Rate', fontsize=14)
    axes[1].set_xlabel('Learning Rate', fontsize=12)
    axes[1].set_ylabel('Dropout Rate', fontsize=12)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'analysis_plots', 'dropout_lr_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def create_performance_ranking(df, output_dir):
    """Create ranking table showing top configurations overall"""

    # Overall best by MAE
    best_by_mae = df.nsmallest(20, 'rul_mae')[
        ['exp_name', 'dataset', 'model_variant', 'batch_size', 'learning_rate',
         'lambda_param', 'nasa_weight', 'dropout', 'rul_mae', 'rul_rmse', 'concordance_index']
    ]
    best_by_mae.to_csv(os.path.join(output_dir, 'top_20_by_mae.csv'), index=False)
    print(f"Saved: {os.path.join(output_dir, 'top_20_by_mae.csv')}")

    # Overall best by C-Index
    best_by_cindex = df.nlargest(20, 'concordance_index')[
        ['exp_name', 'dataset', 'model_variant', 'batch_size', 'learning_rate',
         'lambda_param', 'nasa_weight', 'dropout', 'rul_mae', 'rul_rmse', 'concordance_index']
    ]
    best_by_cindex.to_csv(os.path.join(output_dir, 'top_20_by_cindex.csv'), index=False)
    print(f"Saved: {os.path.join(output_dir, 'top_20_by_cindex.csv')}")

    # Overall best by NASA score
    best_by_nasa = df.nsmallest(20, 'nasa_score')[
        ['exp_name', 'dataset', 'model_variant', 'batch_size', 'learning_rate',
         'lambda_param', 'nasa_weight', 'dropout', 'rul_mae', 'rul_rmse', 'concordance_index', 'nasa_score']
    ]
    best_by_nasa.to_csv(os.path.join(output_dir, 'top_20_by_nasa_score.csv'), index=False)
    print(f"Saved: {os.path.join(output_dir, 'top_20_by_nasa_score.csv')}")


def plot_training_efficiency(df, output_dir):
    """Analyze training time vs performance tradeoff"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ['rul_mae', 'rul_rmse', 'concordance_index']
    titles = ['Training Time vs MAE', 'Training Time vs RMSE', 'Training Time vs C-Index']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]

        for model in df['model_variant'].unique():
            model_df = df[df['model_variant'] == model]
            ax.scatter(model_df['duration_min'], model_df[metric],
                      label=model, alpha=0.6, s=50)

        ax.set_xlabel('Training Time (minutes)', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').upper(), fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'analysis_plots', 'training_efficiency.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze parallel experiment results')
    parser.add_argument('--output-dir', type=str, default='parallel_experiments',
                       help='Directory containing experiment results')

    args = parser.parse_args()

    print("="*80)
    print("PARALLEL EXPERIMENT ANALYSIS")
    print("="*80)

    # Load results
    print("\nLoading results...")
    df, manifest = load_results(args.output_dir)
    print(f"Loaded {len(df)} successful experiments")

    # Create summary table
    print("\nCreating summary statistics...")
    summary = create_summary_table(df, args.output_dir)

    # Find best configurations
    print("\nFinding best configurations...")
    best_configs = find_best_configurations(df, args.output_dir)

    # Create performance rankings
    print("\nCreating performance rankings...")
    create_performance_ranking(df, args.output_dir)

    # Create visualizations
    print("\nGenerating comprehensive visualizations...")

    print("  1/13: Hyperparameter effects...")
    plot_hyperparameter_effects(df, args.output_dir)

    print("  2/13: Model comparison...")
    plot_model_comparison(df, args.output_dir)

    print("  3/13: Batch size analysis...")
    plot_batch_size_vs_performance(df, args.output_dir)

    print("  4/13: Learning rate analysis...")
    plot_learning_rate_analysis(df, args.output_dir)

    print("  5/13: Lambda parameter analysis...")
    plot_lambda_analysis(df, args.output_dir)

    print("  6/13: Dropout analysis...")
    plot_dropout_analysis(df, args.output_dir)

    print("  7/13: NASA loss impact...")
    plot_nasa_loss_impact(df, args.output_dir)

    print("  8/13: Dataset comparison...")
    plot_dataset_comparison(df, args.output_dir)

    print("  9/13: Training efficiency...")
    plot_training_efficiency(df, args.output_dir)

    print("  10/13: Batch size vs LR heatmap...")
    create_heatmap(df, args.output_dir)

    print("  11/13: Lambda vs NASA heatmap...")
    print("  12/13: Dropout vs LR heatmap...")
    create_comprehensive_heatmaps(df, args.output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nGenerated Outputs:")
    print(f"  - Summary statistics: {args.output_dir}/summary_statistics.csv")
    print(f"  - Best configurations: {args.output_dir}/best_configurations.csv")
    print(f"  - Top 20 by MAE: {args.output_dir}/top_20_by_mae.csv")
    print(f"  - Top 20 by C-Index: {args.output_dir}/top_20_by_cindex.csv")
    print(f"  - Top 20 by NASA Score: {args.output_dir}/top_20_by_nasa_score.csv")
    print(f"  - Visualizations: {args.output_dir}/analysis_plots/")
    print(f"\nTotal visualizations: 13 plots")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
