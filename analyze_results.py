#!/usr/bin/env python3
"""
Quick reference: How to use the evaluation scripts

This file shows common usage patterns and commands.
"""

# ============================================================================
# STEP 1: Pre-flight checks (on server)
# ============================================================================

# Command:
# $ python test_evaluation_setup.py

# This will check:
# - All imports work correctly
# - Dataset CSVs exist (data/raw/test.csv, data/raw/val.csv, etc.)
# - Model directory structure is correct
# - Number of discovered models


# ============================================================================
# STEP 2: Run full evaluation (on server)
# ============================================================================

# Command:
# $ python evaluate_all_models.py

# This will:
# 1. Discover all models in models/{architecture}/{fourier_mode}/weights/
# 2. For each model, load 3 test datasets
# 3. Run inference and collect metrics
# 4. Save unified results to models/all_models_metrics_unified.csv
# 5. Print summary statistics to console and evaluation.log


# ============================================================================
# STEP 3: Analyze results
# ============================================================================

import pandas as pd

# Load results
df = pd.read_csv("models/all_models_metrics_unified.csv")

# View first few rows
print(df.head())

# Get best models overall (by F1 score)
print("\n=== Top 10 Best Models ===")
print(df.nlargest(10, 'f1')[['model_variant', 'dataset', 'f1', 'auc', 'accuracy']])

# Get worst models (for comparison)
print("\n=== Bottom 5 Models ===")
print(df.nsmallest(5, 'f1')[['model_variant', 'dataset', 'f1', 'auc', 'accuracy']])

# Average metrics by architecture
print("\n=== Average Metrics by Architecture ===")
print(df.groupby('architecture')[['accuracy', 'f1', 'auc']].mean().round(4))

# Average metrics by Fourier mode
print("\n=== Average Metrics by Fourier Mode ===")
print(df.groupby('fourier_mode')[['accuracy', 'f1', 'auc']].mean().round(4))

# Best model per dataset
print("\n=== Best Model per Dataset ===")
for dataset in sorted(df['dataset'].unique()):
    subset = df[df['dataset'] == dataset]
    best_idx = subset['f1'].idxmax()
    best = subset.loc[best_idx]
    print(f"{dataset:15s}: {best['model_variant']:35s} "
          f"F1={best['f1']:.4f} AUC={best['auc']:.4f} Acc={best['accuracy']:.4f}")

# Best model per architecture
print("\n=== Best Model per Architecture ===")
for arch in sorted(df['architecture'].unique()):
    subset = df[df['architecture'] == arch]
    best_idx = subset['f1'].idxmax()
    best = subset.loc[best_idx]
    print(f"{arch:15s}: {best['model_variant']:35s} "
          f"F1={best['f1']:.4f} ({best['dataset']})")

# Filter by Fourier mode and see best architecture for each
print("\n=== Best Architecture per Fourier Mode ===")
for mode in sorted(df['fourier_mode'].unique()):
    subset = df[df['fourier_mode'] == mode]
    arch_avg = subset.groupby('architecture')['f1'].mean()
    best_arch = arch_avg.idxmax()
    print(f"{mode:20s}: {best_arch:15s} (avg F1={arch_avg[best_arch]:.4f})")

# Export to different formats
print("\n=== Exporting Results ===")

# To Excel (if openpyxl installed)
try:
    df.to_excel("models/all_models_metrics_unified.xlsx", index=False)
    print("✓ Exported to: models/all_models_metrics_unified.xlsx")
except:
    print("✗ openpyxl not installed, skipping Excel export")

# By architecture (separate CSVs)
for arch in df['architecture'].unique():
    arch_df = df[df['architecture'] == arch]
    arch_df.to_csv(f"models/metrics_{arch}.csv", index=False)
    print(f"✓ Exported to: models/metrics_{arch}.csv")

# By Fourier mode (separate CSVs)
for mode in df['fourier_mode'].unique():
    mode_df = df[df['fourier_mode'] == mode]
    mode_df.to_csv(f"models/metrics_{mode}.csv", index=False)
    print(f"✓ Exported to: models/metrics_{mode}.csv")

# By dataset (separate CSVs)
for dataset in df['dataset'].unique():
    dataset_df = df[df['dataset'] == dataset]
    dataset_df.to_csv(f"models/metrics_{dataset}.csv", index=False)
    print(f"✓ Exported to: models/metrics_{dataset}.csv")


# ============================================================================
# STEP 4: Create visualizations (optional)
# ============================================================================

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Setup
    sns.set_theme()
    
    # Plot 1: Accuracy by Architecture
    fig, ax = plt.subplots(figsize=(10, 6))
    df.groupby('architecture')['accuracy'].mean().plot(kind='bar', ax=ax)
    ax.set_title('Average Accuracy by Architecture')
    ax.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('models/plot_accuracy_by_arch.png')
    print("\n✓ Created plot: models/plot_accuracy_by_arch.png")
    
    # Plot 2: F1 by Architecture and Dataset
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df.pivot_table(values='f1', index='architecture', columns='dataset', aggfunc='mean')
    pivot.plot(kind='bar', ax=ax)
    ax.set_title('Average F1-Score by Architecture and Dataset')
    ax.set_ylabel('F1-Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('models/plot_f1_by_arch_dataset.png')
    print("✓ Created plot: models/plot_f1_by_arch_dataset.png")
    
    # Plot 3: Heatmap of metrics by model and dataset
    if len(df) < 50:  # Only if manageable
        pivot = df.pivot_table(values='f1', index='model_variant', columns='dataset', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(8, 12))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
        ax.set_title('F1-Score Heatmap: Models vs Datasets')
        plt.tight_layout()
        plt.savefig('models/plot_f1_heatmap.png')
        print("✓ Created plot: models/plot_f1_heatmap.png")
        
except ImportError:
    print("\n⚠ matplotlib/seaborn not installed, skipping plots")
except Exception as e:
    print(f"\n⚠ Error creating plots: {e}")


# ============================================================================
# STEP 5: Generate summary report
# ============================================================================

def generate_report(df, output_file="models/evaluation_report.txt"):
    """Generate a text summary report"""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Evaluations: {len(df)}\n")
        f.write(f"Architectures: {df['architecture'].nunique()}\n")
        f.write(f"Fourier Modes: {df['fourier_mode'].nunique()}\n")
        f.write(f"Test Datasets: {df['dataset'].nunique()}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("TOP 10 BEST MODELS\n")
        f.write("-"*80 + "\n")
        top10 = df.nlargest(10, 'f1')[['model_variant', 'dataset', 'accuracy', 'f1', 'auc']]
        f.write(top10.to_string(index=False))
        
        f.write("\n\n" + "-"*80 + "\n")
        f.write("AVERAGE METRICS BY ARCHITECTURE\n")
        f.write("-"*80 + "\n")
        arch_stats = df.groupby('architecture')[['accuracy', 'precision', 'recall', 'f1', 'auc']].mean()
        f.write(arch_stats.round(4).to_string())
        
        f.write("\n\n" + "-"*80 + "\n")
        f.write("AVERAGE METRICS BY FOURIER MODE\n")
        f.write("-"*80 + "\n")
        mode_stats = df.groupby('fourier_mode')[['accuracy', 'precision', 'recall', 'f1', 'auc']].mean()
        f.write(mode_stats.round(4).to_string())
        
        f.write("\n\n" + "-"*80 + "\n")
        f.write("AVERAGE METRICS BY DATASET\n")
        f.write("-"*80 + "\n")
        dataset_stats = df.groupby('dataset')[['accuracy', 'precision', 'recall', 'f1', 'auc']].mean()
        f.write(dataset_stats.round(4).to_string())
        
        f.write("\n\n" + "="*80 + "\n")
    
    print(f"✓ Report saved to: {output_file}")

generate_report(df)


# ============================================================================
# That's it!
# ============================================================================

print("\n" + "="*80)
print("Evaluation complete. See EVALUATION_GUIDE.md for more details.")
print("="*80)
