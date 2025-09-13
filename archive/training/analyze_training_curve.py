#!/usr/bin/env python3
"""
Analysis of fine_tune_training_3.csv training curve data
Focuses on stability, performance metrics, and validation data availability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_training_stability(train_loss):
    """Count loss oscillations (direction changes) in training loss"""
    if len(train_loss) < 2:
        return 0
    
    # Calculate direction changes
    diff = np.diff(train_loss)
    sign_changes = 0
    
    for i in range(len(diff) - 1):
        # Count when loss direction changes from increasing to decreasing or vice versa
        if diff[i] > 0 and diff[i+1] < 0:  # Peak
            sign_changes += 1
        elif diff[i] < 0 and diff[i+1] > 0:  # Valley
            sign_changes += 1
            
    return sign_changes

def main():
    # Read the training data
    df = pd.read_csv('/Users/ruicheng/Documents/GitHub/Prediction-Guided-Active-Experiments/fine_tune_training_3.csv')
    
    print("=" * 60)
    print("TRAINING CURVE ANALYSIS - fine_tune_training_3.csv")
    print("=" * 60)
    
    # Basic data info
    print(f"\nDATA OVERVIEW:")
    print(f"Total training steps: {len(df)}")
    print(f"Training completed: {'Yes' if len(df) > 280 else 'No (stopped early)'}")
    
    # Clean data - remove rows with NaN values for analysis
    train_loss = df['train_loss'].dropna()
    train_acc = df['train_mean_token_accuracy'].dropna()
    valid_loss = df['valid_loss'].dropna()
    valid_acc = df['valid_mean_token_accuracy'].dropna()
    
    print(f"\n1. STABILITY ANALYSIS:")
    print(f"   Training steps with loss data: {len(train_loss)}")
    
    # Count oscillations
    oscillations = analyze_training_stability(train_loss)
    print(f"   Loss oscillations (direction changes): {oscillations}")
    print(f"   Previous run oscillations: 464")
    print(f"   Stability improvement: {464 - oscillations} fewer oscillations ({((464 - oscillations)/464)*100:.1f}% reduction)")
    
    # Calculate loss statistics
    loss_std = train_loss.std()
    loss_mean = train_loss.mean()
    loss_cv = loss_std / loss_mean  # Coefficient of variation
    
    print(f"   Training loss std deviation: {loss_std:.4f}")
    print(f"   Training loss mean: {loss_mean:.4f}")
    print(f"   Coefficient of variation: {loss_cv:.4f}")
    
    print(f"\n2. PERFORMANCE METRICS:")
    
    # Training performance
    initial_train_acc = train_acc.iloc[0] if len(train_acc) > 0 else None
    final_train_acc = train_acc.iloc[-1] if len(train_acc) > 0 else None
    initial_train_loss = train_loss.iloc[0] if len(train_loss) > 0 else None
    final_train_loss = train_loss.iloc[-1] if len(train_loss) > 0 else None
    
    if initial_train_acc and final_train_acc:
        print(f"   Training accuracy: {initial_train_acc:.3f} → {final_train_acc:.3f}")
        print(f"   Training accuracy change: {final_train_acc - initial_train_acc:.3f}")
        print(f"   Previous run: 87.5% → 80.95% (dropped -6.55%)")
        
    if initial_train_loss and final_train_loss:
        print(f"   Training loss: {initial_train_loss:.4f} → {final_train_loss:.4f}")
        print(f"   Training loss change: {final_train_loss - initial_train_loss:.4f}")
    
    print(f"\n3. VALIDATION DATA ANALYSIS:")
    print(f"   Validation steps available: {len(valid_loss)}")
    print(f"   Validation frequency: Every {len(df) // len(valid_loss) if len(valid_loss) > 0 else 'N/A'} training steps")
    print(f"   Previous run had very limited validation data")
    
    if len(valid_loss) > 0 and len(valid_acc) > 0:
        initial_valid_acc = valid_acc.iloc[0]
        final_valid_acc = valid_acc.iloc[-1]
        initial_valid_loss = valid_loss.iloc[0]
        final_valid_loss = valid_loss.iloc[-1]
        
        print(f"   Validation accuracy: {initial_valid_acc:.3f} → {final_valid_acc:.3f}")
        print(f"   Validation accuracy change: {final_valid_acc - initial_valid_acc:.3f}")
        print(f"   Validation loss: {initial_valid_loss:.4f} → {final_valid_loss:.4f}")
        print(f"   Validation loss change: {final_valid_loss - initial_valid_loss:.4f}")
        
        # Check for overfitting
        if final_train_acc and final_valid_acc:
            acc_gap = final_train_acc - final_valid_acc
            print(f"   Train-Validation accuracy gap: {acc_gap:.3f}")
            print(f"   Overfitting indicator: {'High' if acc_gap > 0.05 else 'Moderate' if acc_gap > 0.02 else 'Low'}")
    
    print(f"\n4. PARAMETER CORRECTIONS APPLIED:")
    print(f"   ✓ Learning rate: 1.5 → 0.5 (reduced)")
    print(f"   ✓ Batch size: 32 → 64 (increased)")  
    print(f"   ✓ Validation set: Regenerated with 15% ratio")
    print(f"   ✓ Data balancing: 5x → 2x weights (reduced)")
    
    print(f"\n5. OVERALL ASSESSMENT:")
    
    # Stability assessment
    if oscillations < 100:
        stability = "EXCELLENT"
    elif oscillations < 200:
        stability = "GOOD"
    elif oscillations < 350:
        stability = "MODERATE"
    else:
        stability = "POOR"
    
    print(f"   Training stability: {stability}")
    print(f"   Oscillation reduction: {((464 - oscillations)/464)*100:.1f}%")
    
    # Performance assessment
    if final_train_acc and initial_train_acc:
        acc_change = final_train_acc - initial_train_acc
        if acc_change > 0.01:
            perf = "IMPROVED"
        elif acc_change > -0.01:
            perf = "STABLE"
        else:
            perf = "DEGRADED"
        print(f"   Training performance: {perf}")
    
    # Validation assessment
    val_quality = "EXCELLENT" if len(valid_loss) > 20 else "GOOD" if len(valid_loss) > 10 else "LIMITED"
    print(f"   Validation data quality: {val_quality}")
    
    print(f"\n6. RECOMMENDATIONS:")
    
    if oscillations < 200:
        print(f"   ✓ Training stability significantly improved")
    else:
        print(f"   ⚠ Consider further reducing learning rate")
        
    if len(valid_loss) > 15:
        print(f"   ✓ Validation monitoring is adequate")
    else:
        print(f"   ⚠ Consider more frequent validation")
        
    if final_train_acc and final_valid_acc and abs(final_train_acc - final_valid_acc) < 0.03:
        print(f"   ✓ No significant overfitting detected")
    elif final_train_acc and final_valid_acc:
        print(f"   ⚠ Monitor for overfitting (gap: {abs(final_train_acc - final_valid_acc):.3f})")
        
    print(f"\n   NEXT STEPS:")
    if oscillations < 200 and final_train_acc and final_train_acc > 0.85:
        print(f"   → Parameters appear well-tuned, consider production deployment")
        print(f"   → Monitor validation performance on larger test set")
    else:
        print(f"   → Fine-tune learning rate schedule")
        print(f"   → Consider early stopping based on validation loss")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()