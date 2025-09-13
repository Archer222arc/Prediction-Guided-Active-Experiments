#!/usr/bin/env python3
"""
Visualize training curves from fine_tune_training_3.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Read the training data
    df = pd.read_csv('/Users/ruicheng/Documents/GitHub/Prediction-Guided-Active-Experiments/fine_tune_training_3.csv')
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curve Analysis - fine_tune_training_3.csv', fontsize=16)
    
    # Plot 1: Training Loss
    ax1.plot(df['step'], df['train_loss'], 'b-', alpha=0.7, label='Training Loss')
    valid_steps = df[df['valid_loss'].notna()]
    ax1.plot(valid_steps['step'], valid_steps['valid_loss'], 'r-', alpha=0.8, label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy
    ax2.plot(df['step'], df['train_mean_token_accuracy'], 'b-', alpha=0.7, label='Training Accuracy')
    ax2.plot(valid_steps['step'], valid_steps['valid_mean_token_accuracy'], 'r-', alpha=0.8, label='Validation Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss Oscillation Analysis (moving standard deviation)
    window_size = 20
    train_loss_clean = df['train_loss'].dropna()
    rolling_std = train_loss_clean.rolling(window=window_size).std()
    steps_clean = df['step'][:len(train_loss_clean)]
    
    ax3.plot(steps_clean[window_size-1:], rolling_std[window_size-1:], 'g-', alpha=0.8)
    ax3.set_title(f'Training Loss Stability (Rolling Std, window={window_size})')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Loss Standard Deviation')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Train-Validation Gap
    # Interpolate validation values to match training steps for comparison
    train_acc_clean = df['train_mean_token_accuracy'].dropna()
    valid_interp = np.interp(df['step'][:len(train_acc_clean)], valid_steps['step'], valid_steps['valid_mean_token_accuracy'])
    gap = train_acc_clean - valid_interp
    
    ax4.plot(df['step'][:len(gap)], gap, 'purple', alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Train-Validation Accuracy Gap')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Training - Validation Accuracy')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/ruicheng/Documents/GitHub/Prediction-Guided-Active-Experiments/training_analysis.png', dpi=150, bbox_inches='tight')
    print("Training visualization saved as 'training_analysis.png'")
    
    # Additional detailed analysis
    print("\nDETAILED METRICS:")
    print(f"Final training accuracy: {train_acc_clean.iloc[-1]:.3f}")
    print(f"Final validation accuracy: {valid_steps['valid_mean_token_accuracy'].iloc[-1]:.3f}")
    print(f"Best training accuracy: {train_acc_clean.max():.3f}")
    print(f"Best validation accuracy: {valid_steps['valid_mean_token_accuracy'].max():.3f}")
    
    # Loss progression analysis
    first_50_loss = train_loss_clean[:50].mean()
    last_50_loss = train_loss_clean[-50:].mean()
    print(f"\nLoss progression:")
    print(f"First 50 steps average loss: {first_50_loss:.4f}")
    print(f"Last 50 steps average loss: {last_50_loss:.4f}")
    print(f"Loss change: {last_50_loss - first_50_loss:.4f}")

if __name__ == "__main__":
    main()