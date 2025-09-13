#!/usr/bin/env python3
"""
Visual Analysis of Fine-tune Training Runs
Creates plots to visualize the training curves and comparisons
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_normalize_data(csv_path, max_steps=None):
    """Load training data and optionally truncate to max_steps"""
    df = pd.read_csv(csv_path)
    if max_steps:
        df = df.head(max_steps)
    return df

def create_comparison_plots():
    """Create comparison plots for all training runs"""
    
    # Load data
    base_path = Path('/Users/ruicheng/Documents/GitHub/Prediction-Guided-Active-Experiments')
    
    files = {
        'Run 2 (lr=1.5)': base_path / 'fine_tune_training_2.csv',
        'Run 3 (lr=0.5)': base_path / 'fine_tune_training_3.csv', 
        'Run 4 (lr=0.75)': base_path / 'fine_tune_training_4.csv'
    }
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fine-tuning Training Analysis: Comparison of Configurations', fontsize=16, fontweight='bold')
    
    colors = {'Run 2 (lr=1.5)': 'red', 'Run 3 (lr=0.5)': 'green', 'Run 4 (lr=0.75)': 'blue'}
    
    # Plot 1: Training Loss Comparison (normalized steps)
    max_steps = 300  # Normalize to common length for comparison
    for run_name, file_path in files.items():
        if file_path.exists():
            df = load_and_normalize_data(file_path, max_steps)
            ax1.plot(df.index, df['train_loss'], label=run_name, color=colors[run_name], alpha=0.8)
    
    ax1.set_title('Training Loss Comparison (First 300 Steps)', fontweight='bold')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy Comparison
    for run_name, file_path in files.items():
        if file_path.exists():
            df = load_and_normalize_data(file_path, max_steps)
            ax2.plot(df.index, df['train_mean_token_accuracy'], label=run_name, color=colors[run_name], alpha=0.8)
    
    ax2.set_title('Training Accuracy Comparison (First 300 Steps)', fontweight='bold')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Run 4 Full Training Curve (2 epochs)
    run4_data = pd.read_csv(files['Run 4 (lr=0.75)'])
    
    # Mark epoch boundary
    epoch_boundary = len(run4_data) // 2
    
    ax3.plot(run4_data.index, run4_data['train_loss'], color='blue', linewidth=2, label='Training Loss')
    ax3.axvline(x=epoch_boundary, color='orange', linestyle='--', linewidth=2, label='Epoch 2 Start')
    ax3.set_title('Run 4: Full Training Curve (2 Epochs)', fontweight='bold')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Training Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add text annotations for epoch analysis
    ax3.text(epoch_boundary//2, 0.35, 'Epoch 1\n(Improving)', ha='center', va='center', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax3.text(epoch_boundary + epoch_boundary//2, 0.35, 'Epoch 2\n(Degrading)', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Plot 4: Oscillation Analysis - Loss Volatility
    window_size = 20
    oscillation_data = []
    
    for run_name, file_path in files.items():
        if file_path.exists():
            df = pd.read_csv(file_path)
            
            # Calculate rolling standard deviation as volatility measure
            rolling_std = df['train_loss'].rolling(window=window_size).std()
            
            # Skip NaN values
            valid_std = rolling_std.dropna()
            
            oscillation_data.append({
                'run_name': run_name,
                'mean_volatility': valid_std.mean(),
                'max_volatility': valid_std.max(),
                'steps': len(df)
            })
            
            # Plot volatility over time
            ax4.plot(rolling_std.index, rolling_std, label=f'{run_name}', color=colors[run_name], alpha=0.8)
    
    ax4.set_title(f'Training Volatility (Rolling Std Dev, window={window_size})', fontweight='bold')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Loss Standard Deviation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = base_path / 'training_analysis_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()
    
    # Print volatility summary
    print("\nTraining Volatility Analysis:")
    print("-" * 50)
    for data in oscillation_data:
        print(f"{data['run_name']:15}: Mean volatility={data['mean_volatility']:.4f}, "
              f"Max volatility={data['max_volatility']:.4f}, Steps={data['steps']}")

def create_summary_metrics_chart():
    """Create a summary comparison chart"""
    
    # Data from analysis
    metrics = {
        'Run 2\n(lr=1.5)': {
            'oscillations': 654,
            'final_train_acc': 80.95,
            'final_valid_acc': 84.38,
            'steps': 683
        },
        'Run 3\n(lr=0.5)': {
            'oscillations': 269,
            'final_train_acc': 84.44,
            'final_valid_acc': 86.46,
            'steps': 291
        },
        'Run 4\n(lr=0.75)': {
            'oscillations': 517,
            'final_train_acc': 83.33,
            'final_valid_acc': 87.50,
            'steps': 560
        }
    }
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Runs: Key Metrics Comparison', fontsize=16, fontweight='bold')
    
    runs = list(metrics.keys())
    colors = ['red', 'green', 'blue']
    
    # Plot 1: Oscillations (lower is better)
    oscillations = [metrics[run]['oscillations'] for run in runs]
    bars1 = ax1.bar(runs, oscillations, color=colors, alpha=0.7)
    ax1.set_title('Loss Oscillations (Lower = Better)', fontweight='bold')
    ax1.set_ylabel('Number of Oscillations')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars1, oscillations):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                str(value), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Final Training Accuracy (higher is better)
    train_accs = [metrics[run]['final_train_acc'] for run in runs]
    bars2 = ax2.bar(runs, train_accs, color=colors, alpha=0.7)
    ax2.set_title('Final Training Accuracy (Higher = Better)', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim([75, 90])
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars2, train_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Final Validation Accuracy (higher is better)
    valid_accs = [metrics[run]['final_valid_acc'] for run in runs]
    bars3 = ax3.bar(runs, valid_accs, color=colors, alpha=0.7)
    ax3.set_title('Final Validation Accuracy (Higher = Better)', fontweight='bold')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_ylim([80, 90])
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars3, valid_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Training Steps
    steps = [metrics[run]['steps'] for run in runs]
    bars4 = ax4.bar(runs, steps, color=colors, alpha=0.7)
    ax4.set_title('Training Steps to Completion', fontweight='bold')
    ax4.set_ylabel('Number of Steps')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars4, steps):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                str(value), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the summary chart
    base_path = Path('/Users/ruicheng/Documents/GitHub/Prediction-Guided-Active-Experiments')
    output_path = base_path / 'training_metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison saved to: {output_path}")
    
    plt.show()

def main():
    """Main visualization function"""
    print("Generating training analysis visualizations...")
    print()
    
    # Create training curve comparisons
    create_comparison_plots()
    print()
    
    # Create summary metrics comparison
    create_summary_metrics_chart()
    print()
    
    print("Visualization Summary:")
    print("=" * 50)
    print("1. Training curves show Run 4 had higher volatility than Run 3")
    print("2. Epoch 2 in Run 4 clearly shows performance degradation")
    print("3. Run 3 achieved best balance of stability and training performance")
    print("4. Run 4 achieved best validation performance despite training issues")
    print("5. Learning rate 0.75 appears too high for stable convergence")

if __name__ == "__main__":
    main()