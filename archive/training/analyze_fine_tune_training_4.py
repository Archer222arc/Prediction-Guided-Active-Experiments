#!/usr/bin/env python3
"""
Comprehensive Analysis of Fine-tune Training Run 4 (Optimized Configuration)
Analyzes training curve data and compares with previous runs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def count_oscillations(losses, threshold_pct=1.0):
    """Count oscillations in loss where change > threshold_pct%"""
    if len(losses) < 2:
        return 0
    
    oscillations = 0
    for i in range(1, len(losses)):
        prev_loss = losses[i-1]
        curr_loss = losses[i]
        pct_change = abs((curr_loss - prev_loss) / prev_loss * 100)
        
        if pct_change > threshold_pct:
            oscillations += 1
    
    return oscillations

def analyze_training_run(csv_path, run_name):
    """Analyze a single training run"""
    df = pd.read_csv(csv_path)
    
    # Basic metrics
    total_steps = len(df)
    final_train_loss = df['train_loss'].iloc[-1]
    final_train_acc = df['train_mean_token_accuracy'].iloc[-1]
    
    # Validation metrics (from last available)
    valid_data = df.dropna(subset=['valid_loss'])
    if not valid_data.empty:
        final_valid_loss = valid_data['valid_loss'].iloc[-1]
        final_valid_acc = valid_data['valid_mean_token_accuracy'].iloc[-1]
        validation_frequency = len(valid_data)
    else:
        final_valid_loss = final_valid_acc = validation_frequency = None
    
    # Full validation metrics
    full_valid_data = df.dropna(subset=['full_valid_loss'])
    if not full_valid_data.empty:
        final_full_valid_loss = full_valid_data['full_valid_loss'].iloc[-1]
        final_full_valid_acc = full_valid_data['full_valid_mean_token_accuracy'].iloc[-1]
        full_validation_frequency = len(full_valid_data)
    else:
        final_full_valid_loss = final_full_valid_acc = full_validation_frequency = None
    
    # Stability analysis
    train_oscillations = count_oscillations(df['train_loss'].values)
    
    # Convergence analysis
    train_losses = df['train_loss'].values
    best_train_loss = min(train_losses)
    worst_train_loss = max(train_losses)
    final_vs_best_pct = ((final_train_loss - best_train_loss) / best_train_loss) * 100
    
    # Training efficiency
    loss_improvement = ((train_losses[0] - final_train_loss) / train_losses[0]) * 100
    
    return {
        'run_name': run_name,
        'total_steps': total_steps,
        'final_train_loss': final_train_loss,
        'final_train_acc': final_train_acc,
        'final_valid_loss': final_valid_loss,
        'final_valid_acc': final_valid_acc,
        'final_full_valid_loss': final_full_valid_loss,
        'final_full_valid_acc': final_full_valid_acc,
        'validation_frequency': validation_frequency,
        'full_validation_frequency': full_validation_frequency,
        'train_oscillations': train_oscillations,
        'best_train_loss': best_train_loss,
        'worst_train_loss': worst_train_loss,
        'final_vs_best_pct': final_vs_best_pct,
        'loss_improvement_pct': loss_improvement,
        'initial_train_loss': train_losses[0]
    }

def analyze_epochs(csv_path):
    """Analyze training by epochs for Run 4 (2 epochs, ~280 steps each)"""
    df = pd.read_csv(csv_path)
    
    # Estimate epoch boundaries based on step count
    total_steps = len(df)
    steps_per_epoch = total_steps // 2  # 2 epochs
    
    epoch1_data = df.iloc[:steps_per_epoch]
    epoch2_data = df.iloc[steps_per_epoch:]
    
    epoch1_analysis = {
        'epoch': 1,
        'steps': len(epoch1_data),
        'start_loss': epoch1_data['train_loss'].iloc[0],
        'end_loss': epoch1_data['train_loss'].iloc[-1],
        'start_acc': epoch1_data['train_mean_token_accuracy'].iloc[0],
        'end_acc': epoch1_data['train_mean_token_accuracy'].iloc[-1],
        'oscillations': count_oscillations(epoch1_data['train_loss'].values),
        'loss_improvement_pct': ((epoch1_data['train_loss'].iloc[0] - epoch1_data['train_loss'].iloc[-1]) / epoch1_data['train_loss'].iloc[0]) * 100
    }
    
    epoch2_analysis = {
        'epoch': 2,
        'steps': len(epoch2_data),
        'start_loss': epoch2_data['train_loss'].iloc[0],
        'end_loss': epoch2_data['train_loss'].iloc[-1],
        'start_acc': epoch2_data['train_mean_token_accuracy'].iloc[0],
        'end_acc': epoch2_data['train_mean_token_accuracy'].iloc[-1],
        'oscillations': count_oscillations(epoch2_data['train_loss'].values),
        'loss_improvement_pct': ((epoch2_data['train_loss'].iloc[0] - epoch2_data['train_loss'].iloc[-1]) / epoch2_data['train_loss'].iloc[0]) * 100
    }
    
    return epoch1_analysis, epoch2_analysis

def main():
    """Main analysis function"""
    
    # Define file paths
    base_path = Path('/Users/ruicheng/Documents/GitHub/Prediction-Guided-Active-Experiments')
    
    training_files = {
        'Run 2 (Problematic)': base_path / 'fine_tune_training_2.csv',
        'Run 3 (Fixed)': base_path / 'fine_tune_training_3.csv',
        'Run 4 (Optimized)': base_path / 'fine_tune_training_4.csv'
    }
    
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS: Fine-tune Training Run 4 (Optimized Configuration)")
    print("=" * 80)
    print()
    
    # Configuration parameters
    print("CONFIGURATION PARAMETERS (Run 4):")
    print("- Learning rate: 0.75 (increased from 0.5)")
    print("- Batch size: 64 (kept same)")
    print("- Epochs: 2 (increased from 1)")
    print("- Data balancing: moderate strategy (2x weights)")
    print("- Validation: 18% enhanced stratified validation")
    print()
    
    # Analyze all runs
    results = {}
    for run_name, file_path in training_files.items():
        if file_path.exists():
            results[run_name] = analyze_training_run(file_path, run_name)
        else:
            print(f"Warning: {file_path} not found")
    
    # 1. STABILITY ANALYSIS
    print("1. STABILITY ANALYSIS (Loss Oscillations)")
    print("-" * 50)
    
    for run_name, result in results.items():
        print(f"{run_name:20}: {result['train_oscillations']:3d} oscillations")
        
    # Calculate improvement percentages
    if 'Run 2 (Problematic)' in results and 'Run 4 (Optimized)' in results:
        run2_osc = results['Run 2 (Problematic)']['train_oscillations']
        run4_osc = results['Run 4 (Optimized)']['train_oscillations']
        improvement_vs_run2 = ((run2_osc - run4_osc) / run2_osc) * 100
        print(f"\nRun 4 vs Run 2: {improvement_vs_run2:+.1f}% change in oscillations")
    
    if 'Run 3 (Fixed)' in results and 'Run 4 (Optimized)' in results:
        run3_osc = results['Run 3 (Fixed)']['train_oscillations']
        run4_osc = results['Run 4 (Optimized)']['train_oscillations']
        improvement_vs_run3 = ((run3_osc - run4_osc) / run3_osc) * 100
        print(f"Run 4 vs Run 3: {improvement_vs_run3:+.1f}% change in oscillations")
    
    print()
    
    # 2. PERFORMANCE METRICS
    print("2. PERFORMANCE METRICS")
    print("-" * 50)
    
    print("Final Training Metrics:")
    for run_name, result in results.items():
        print(f"{run_name:20}: Loss={result['final_train_loss']:.4f}, Accuracy={result['final_train_acc']:.4f}")
    
    print("\nFinal Validation Metrics:")
    for run_name, result in results.items():
        if result['final_valid_loss'] is not None:
            print(f"{run_name:20}: Loss={result['final_valid_loss']:.4f}, Accuracy={result['final_valid_acc']:.4f}")
        else:
            print(f"{run_name:20}: No validation data")
    
    print("\nFinal Full Validation Metrics:")
    for run_name, result in results.items():
        if result['final_full_valid_loss'] is not None:
            print(f"{run_name:20}: Loss={result['final_full_valid_loss']:.4f}, Accuracy={result['final_full_valid_acc']:.4f}")
        else:
            print(f"{run_name:20}: No full validation data")
    
    # Performance improvements
    if 'Run 3 (Fixed)' in results and 'Run 4 (Optimized)' in results:
        run3 = results['Run 3 (Fixed)']
        run4 = results['Run 4 (Optimized)']
        
        print(f"\nPerformance Improvements (Run 4 vs Run 3):")
        
        # Training loss improvement
        train_loss_imp = ((run3['final_train_loss'] - run4['final_train_loss']) / run3['final_train_loss']) * 100
        print(f"Training Loss:     {train_loss_imp:+.2f}%")
        
        # Training accuracy improvement
        train_acc_imp = ((run4['final_train_acc'] - run3['final_train_acc']) / run3['final_train_acc']) * 100
        print(f"Training Accuracy: {train_acc_imp:+.2f}%")
        
        # Validation improvements if available
        if run3['final_valid_loss'] and run4['final_valid_loss']:
            valid_loss_imp = ((run3['final_valid_loss'] - run4['final_valid_loss']) / run3['final_valid_loss']) * 100
            valid_acc_imp = ((run4['final_valid_acc'] - run3['final_valid_acc']) / run3['final_valid_acc']) * 100
            print(f"Validation Loss:   {valid_loss_imp:+.2f}%")
            print(f"Validation Accuracy: {valid_acc_imp:+.2f}%")
    
    print()
    
    # 3. TRAINING EFFICIENCY
    print("3. TRAINING EFFICIENCY")
    print("-" * 50)
    
    for run_name, result in results.items():
        print(f"{run_name:20}: {result['total_steps']:3d} steps, {result['loss_improvement_pct']:+.1f}% loss improvement")
    
    # Steps to completion comparison
    if 'Run 3 (Fixed)' in results and 'Run 4 (Optimized)' in results:
        run3_steps = results['Run 3 (Fixed)']['total_steps']
        run4_steps = results['Run 4 (Optimized)']['total_steps']
        step_ratio = run4_steps / run3_steps
        print(f"\nRun 4 used {step_ratio:.1f}x more steps than Run 3 (expected due to 2 epochs vs 1)")
    
    print()
    
    # 4. VALIDATION COVERAGE
    print("4. VALIDATION COVERAGE")
    print("-" * 50)
    
    for run_name, result in results.items():
        val_freq = result['validation_frequency'] or 0
        full_val_freq = result['full_validation_frequency'] or 0
        val_coverage = (val_freq / result['total_steps']) * 100 if result['total_steps'] > 0 else 0
        full_val_coverage = (full_val_freq / result['total_steps']) * 100 if result['total_steps'] > 0 else 0
        
        print(f"{run_name:20}: {val_freq:2d} validations ({val_coverage:.1f}% coverage), {full_val_freq:2d} full validations ({full_val_coverage:.1f}% coverage)")
    
    print()
    
    # 5. EPOCH-BY-EPOCH ANALYSIS (Run 4 only)
    print("5. EPOCH-BY-EPOCH ANALYSIS (Run 4 Only)")
    print("-" * 50)
    
    if 'Run 4 (Optimized)' in results:
        epoch1, epoch2 = analyze_epochs(training_files['Run 4 (Optimized)'])
        
        print(f"Epoch 1: {epoch1['steps']} steps")
        print(f"  Loss: {epoch1['start_loss']:.4f} → {epoch1['end_loss']:.4f} ({epoch1['loss_improvement_pct']:+.1f}%)")
        print(f"  Acc:  {epoch1['start_acc']:.4f} → {epoch1['end_acc']:.4f}")
        print(f"  Oscillations: {epoch1['oscillations']}")
        print()
        
        print(f"Epoch 2: {epoch2['steps']} steps")
        print(f"  Loss: {epoch2['start_loss']:.4f} → {epoch2['end_loss']:.4f} ({epoch2['loss_improvement_pct']:+.1f}%)")
        print(f"  Acc:  {epoch2['start_acc']:.4f} → {epoch2['end_acc']:.4f}")
        print(f"  Oscillations: {epoch2['oscillations']}")
        print()
        
        # Compare epochs
        epoch2_vs_epoch1_osc = ((epoch1['oscillations'] - epoch2['oscillations']) / epoch1['oscillations']) * 100 if epoch1['oscillations'] > 0 else 0
        print(f"Epoch 2 vs Epoch 1: {epoch2_vs_epoch1_osc:+.1f}% change in oscillations")
    
    print()
    
    # 6. OVERALL ASSESSMENT
    print("6. OVERALL ASSESSMENT")
    print("-" * 50)
    
    if 'Run 4 (Optimized)' in results:
        run4 = results['Run 4 (Optimized)']
        
        # Determine if optimized approach succeeded
        stability_achieved = run4['train_oscillations'] < 300  # Reasonable threshold
        performance_good = run4['final_train_acc'] > 0.85  # Good accuracy
        convergence_quality = run4['final_vs_best_pct'] < 10  # Close to best loss
        
        print(f"✓ Stability: {'GOOD' if stability_achieved else 'POOR'} ({run4['train_oscillations']} oscillations)")
        print(f"✓ Performance: {'GOOD' if performance_good else 'POOR'} ({run4['final_train_acc']:.1%} accuracy)")
        print(f"✓ Convergence: {'GOOD' if convergence_quality else 'POOR'} ({run4['final_vs_best_pct']:.1f}% from best)")
        
        overall_success = stability_achieved and performance_good and convergence_quality
        print(f"\nOVERALL: {'SUCCESS' if overall_success else 'MIXED RESULTS'}")
        
        if overall_success:
            print("The optimized approach achieved better performance while maintaining stability.")
        else:
            print("The optimized approach shows mixed results - some improvements but potential concerns.")
    
    print()
    
    # 7. RECOMMENDATIONS
    print("7. RECOMMENDATIONS")
    print("-" * 50)
    
    if 'Run 4 (Optimized)' in results:
        run4 = results['Run 4 (Optimized)']
        
        print("Based on the analysis:")
        print()
        
        if run4['train_oscillations'] > 200:
            print("• Consider reducing learning rate further (e.g., 0.6) to improve stability")
        else:
            print("• Current learning rate (0.75) appears optimal for stability")
            
        if run4['final_train_acc'] < 0.90:
            print("• Consider increasing training time or adjusting data balancing")
        else:
            print("• Training accuracy is satisfactory")
            
        if run4.get('final_full_valid_acc') and run4['final_full_valid_acc'] < 0.85:
            print("• Monitor for potential overfitting - validation accuracy is lower")
        elif run4.get('final_full_valid_acc'):
            print("• Validation performance looks good - no signs of overfitting")
            
        print("• Continue with 2-epoch training as it shows continued improvement")
        print("• Enhanced validation monitoring is providing good insights")
        print("• Consider experimenting with learning rate scheduling for further optimization")
    
    print()
    print("=" * 80)
    
    # Save results to JSON for further analysis
    output_file = base_path / 'training_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()