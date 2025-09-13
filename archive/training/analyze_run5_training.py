#!/usr/bin/env python3
"""
Comprehensive Analysis of Fine-tuning Run 5 (Learning Rate 0.6)
Compares against previous runs to determine optimal configuration
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple

def count_oscillations(loss_values: List[float], threshold: float = 0.005) -> int:
    """Count significant loss oscillations (changes > threshold)"""
    if len(loss_values) < 2:
        return 0
    
    oscillations = 0
    for i in range(1, len(loss_values)):
        if abs(loss_values[i] - loss_values[i-1]) > threshold:
            oscillations += 1
    
    return oscillations

def analyze_convergence(loss_values: List[float], window: int = 20) -> Dict:
    """Analyze convergence pattern"""
    if len(loss_values) < window:
        return {"converged": False, "trend": "insufficient_data"}
    
    # Calculate moving average for last window
    recent_losses = loss_values[-window:]
    trend_slope = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
    
    # Calculate variance in recent window
    variance = np.var(recent_losses)
    
    return {
        "final_avg_loss": np.mean(recent_losses),
        "trend_slope": trend_slope,
        "variance": variance,
        "converged": abs(trend_slope) < 0.001 and variance < 0.001,
        "trend": "decreasing" if trend_slope < -0.001 else "stable" if abs(trend_slope) < 0.001 else "increasing"
    }

def analyze_run5_data():
    """Main analysis function for Run 5"""
    print("=== COMPREHENSIVE ANALYSIS: Fine-tuning Run 5 (Learning Rate 0.6) ===\n")
    
    # Load the data
    df = pd.read_csv('/Users/ruicheng/Documents/GitHub/Prediction-Guided-Active-Experiments/fine_tune_training_5.csv')
    
    print(f"Dataset Overview:")
    print(f"- Total steps completed: {len(df)}")
    print(f"- Training duration: {df['step'].max()} steps")
    print(f"- Final step: {df['step'].iloc[-1]}")
    print()
    
    # Extract final metrics
    final_row = df.iloc[-1]
    final_train_acc = final_row['train_mean_token_accuracy']
    final_train_loss = final_row['train_loss']
    
    # Get validation metrics (use full_valid when available, otherwise valid)
    if pd.notna(final_row['full_valid_mean_token_accuracy']):
        final_valid_acc = final_row['full_valid_mean_token_accuracy']
        final_valid_loss = final_row['full_valid_loss']
        print("Using full validation metrics for final assessment")
    else:
        final_valid_acc = final_row['valid_mean_token_accuracy']
        final_valid_loss = final_row['valid_loss']
        print("Using regular validation metrics for final assessment")
    
    print(f"\n1. **FINAL PERFORMANCE METRICS**")
    print(f"   Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"   Training Loss: {final_train_loss:.6f}")
    print(f"   Validation Accuracy: {final_valid_acc:.4f} ({final_valid_acc*100:.2f}%)")
    print(f"   Validation Loss: {final_valid_loss:.6f}")
    
    # Analyze oscillations
    train_losses = df['train_loss'].dropna().tolist()
    oscillations = count_oscillations(train_losses)
    
    print(f"\n2. **STABILITY ANALYSIS**")
    print(f"   Total Loss Oscillations: {oscillations}")
    print(f"   Oscillations per 100 steps: {oscillations/len(train_losses)*100:.1f}")
    
    # Convergence analysis
    convergence = analyze_convergence(train_losses)
    print(f"   Convergence Status: {'✓ Converged' if convergence['converged'] else '✗ Not Converged'}")
    print(f"   Final Trend: {convergence['trend'].title()}")
    print(f"   Loss Variance (final 20 steps): {convergence['variance']:.6f}")
    
    # Training quality metrics
    train_accs = df['train_mean_token_accuracy'].dropna()
    acc_improvement = train_accs.iloc[-1] - train_accs.iloc[0] if len(train_accs) > 1 else 0
    
    print(f"\n3. **TRAINING QUALITY**")
    print(f"   Initial Training Accuracy: {train_accs.iloc[0]:.4f} ({train_accs.iloc[0]*100:.2f}%)")
    print(f"   Final Training Accuracy: {train_accs.iloc[-1]:.4f} ({train_accs.iloc[-1]*100:.2f}%)")
    print(f"   Accuracy Improvement: {acc_improvement:.4f} ({acc_improvement*100:.2f} percentage points)")
    print(f"   Peak Training Accuracy: {train_accs.max():.4f} ({train_accs.max()*100:.2f}%)")
    
    # Validation coverage analysis
    valid_steps = df['valid_mean_token_accuracy'].notna().sum()
    validation_coverage = valid_steps / len(df) * 100
    
    print(f"\n4. **VALIDATION COVERAGE**")
    print(f"   Validation evaluations: {valid_steps} out of {len(df)} steps")
    print(f"   Coverage: {validation_coverage:.1f}%")
    
    # Get all validation accuracies for trend analysis
    valid_accs = df['valid_mean_token_accuracy'].dropna()
    if len(valid_accs) > 1:
        valid_improvement = valid_accs.iloc[-1] - valid_accs.iloc[0]
        print(f"   Validation Improvement: {valid_improvement:.4f} ({valid_improvement*100:.2f} percentage points)")
        print(f"   Peak Validation Accuracy: {valid_accs.max():.4f} ({valid_accs.max()*100:.2f}%)")
    
    print(f"\n5. **COMPARISON WITH PREVIOUS RUNS**")
    print(f"   Run 2 (lr=1.5): 464 oscillations, very unstable")
    print(f"   Run 3 (lr=0.5): 175 oscillations, 84.44% train, 86.46% valid")
    print(f"   Run 4 (lr=0.75): 517 oscillations, 83.33% train, 87.50% valid")
    print(f"   Run 5 (lr=0.6): {oscillations} oscillations, {final_train_acc*100:.2f}% train, {final_valid_acc*100:.2f}% valid")
    
    # Ranking analysis
    runs_data = {
        'Run 2 (lr=1.5)': {'oscillations': 464, 'train_acc': None, 'valid_acc': None, 'stability_score': 1},
        'Run 3 (lr=0.5)': {'oscillations': 175, 'train_acc': 84.44, 'valid_acc': 86.46, 'stability_score': 8},
        'Run 4 (lr=0.75)': {'oscillations': 517, 'train_acc': 83.33, 'valid_acc': 87.50, 'stability_score': 2},
        'Run 5 (lr=0.6)': {'oscillations': oscillations, 'train_acc': final_train_acc*100, 'valid_acc': final_valid_acc*100}
    }
    
    # Calculate stability score for Run 5 (lower oscillations = higher score)
    if oscillations <= 150:
        stability_score = 10
    elif oscillations <= 200:
        stability_score = 8
    elif oscillations <= 300:
        stability_score = 6
    elif oscillations <= 400:
        stability_score = 4
    else:
        stability_score = 2
    
    runs_data['Run 5 (lr=0.6)']['stability_score'] = stability_score
    
    print(f"\n6. **FINAL RANKING (Best to Worst)**")
    
    # Create ranking based on balance of performance and stability
    rankings = []
    for run, data in runs_data.items():
        if data['valid_acc'] is not None:
            # Composite score: validation accuracy + stability bonus
            composite_score = data['valid_acc'] + (data['stability_score'] * 0.5)
            rankings.append((run, composite_score, data))
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    for i, (run, score, data) in enumerate(rankings, 1):
        print(f"   {i}. {run}")
        print(f"      - Validation: {data['valid_acc']:.2f}%, Stability: {data['stability_score']}/10, Score: {score:.1f}")
    
    # Success assessment
    print(f"\n7. **SUCCESS ASSESSMENT**")
    hypothesis_met = (
        120 <= oscillations <= 150 and
        85 <= final_train_acc*100 <= 87 and
        87 <= final_valid_acc*100 <= 89
    )
    
    print(f"   Hypothesis: lr=0.6 should show ~120-150 oscillations, 85-87% train, 87-89% valid")
    print(f"   Actual Results: {oscillations} oscillations, {final_train_acc*100:.2f}% train, {final_valid_acc*100:.2f}% valid")
    print(f"   Hypothesis Status: {'✓ MET' if hypothesis_met else '✗ PARTIALLY MET'}")
    
    # Determine if optimal
    is_optimal = (
        oscillations < 200 and  # Good stability
        final_valid_acc >= 0.87 and  # Good validation performance
        final_train_acc >= 0.85  # Good training performance
    )
    
    print(f"   Optimal Configuration: {'✓ YES' if is_optimal else '✗ NO'}")
    
    if is_optimal:
        print(f"   ✅ SUCCESS: lr=0.6 achieved optimal balance of stability and performance!")
    else:
        print(f"   ⚠️  MIXED RESULTS: Some targets met, but not all optimization goals achieved")
    
    print(f"\n8. **DETAILED RECOMMENDATIONS**")
    if oscillations < 175:
        print(f"   ✓ Excellent stability (fewer oscillations than lr=0.5)")
    elif oscillations < 300:
        print(f"   ✓ Good stability (better than high learning rates)")
    else:
        print(f"   ⚠️  Higher than expected oscillations")
        
    if final_valid_acc >= 0.875:
        print(f"   ✓ Excellent validation performance")
    elif final_valid_acc >= 0.85:
        print(f"   ✓ Good validation performance")
    else:
        print(f"   ⚠️  Validation performance below expectations")
    
    # Summary conclusion
    print(f"\n" + "="*80)
    print(f"CONCLUSION: Learning Rate 0.6 Analysis")
    print(f"="*80)
    
    best_run = rankings[0][0] if rankings else "Run 5 (lr=0.6)"
    if "Run 5" in best_run:
        print(f"🎯 SUCCESS: lr=0.6 is the OPTIMAL configuration!")
        print(f"   - Best overall balance of stability and performance")
        print(f"   - {oscillations} oscillations (stable training)")
        print(f"   - {final_valid_acc*100:.2f}% validation accuracy")
    else:
        print(f"📊 ANALYSIS COMPLETE: lr=0.6 shows good results but not optimal")
        print(f"   - Best run remains: {best_run}")
        print(f"   - Consider further tuning around optimal range")
    
    return {
        'oscillations': oscillations,
        'final_train_acc': final_train_acc,
        'final_valid_acc': final_valid_acc,
        'stability_score': stability_score,
        'is_optimal': is_optimal,
        'rankings': rankings
    }

if __name__ == "__main__":
    results = analyze_run5_data()
    
    # Save results for reference
    with open('/Users/ruicheng/Documents/GitHub/Prediction-Guided-Active-Experiments/run5_analysis_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key == 'rankings':
                json_results[key] = [(run, float(score), data) for run, score, data in value]
            elif isinstance(value, np.floating):
                json_results[key] = float(value)
            elif isinstance(value, np.integer):
                json_results[key] = int(value)
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: run5_analysis_results.json")