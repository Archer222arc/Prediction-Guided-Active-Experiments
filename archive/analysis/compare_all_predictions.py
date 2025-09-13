#!/usr/bin/env python3
"""
对比三种预测方法的准确性：基础模型、微调模型、微调+CoT模型
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_datasets():
    """加载四个预测结果数据集"""
    
    # 加载基础模型结果 (原始LLM prediction)
    baseline_df = pd.read_csv('data/NPORS_2024_for_public_release_basic_prompting.csv')
    
    # 加载微调模型结果 (Step 7)
    finetuned_df = pd.read_csv('NPORS_2024_LLM_finetuned_predictions_20250910_205048.csv')
    
    # 加载微调+CoT结果 (改进的CoT)  
    cot_df = pd.read_csv('NPORS_2024_cot_optimized_lr06_step560_20250911_232934.csv')
    
    # 加载最新优化微调模型结果 (lr=0.6, step 560)
    optimized_df = pd.read_csv('NPORS_2024_base_gpt41mini_lr06_step560_20250911_214943.csv')
    
    return baseline_df, finetuned_df, cot_df, optimized_df

def calculate_accuracy_metrics(df, true_col, pred_col, valid_responses):
    """计算预测准确性指标"""
    
    # 过滤有效数据
    valid_mask = (
        df[true_col].notna() & 
        df[pred_col].notna() & 
        (df[true_col] != 99.0) &
        df[true_col].isin(valid_responses) &
        df[pred_col].isin(valid_responses)
    )
    
    if valid_mask.sum() == 0:
        return None
    
    y_true = df.loc[valid_mask, true_col].astype(int)
    y_pred = df.loc[valid_mask, pred_col].astype(int)
    
    metrics = {
        'total_samples': len(df),
        'valid_predictions': valid_mask.sum(),
        'coverage': valid_mask.sum() / len(df),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics, y_true, y_pred

def compare_four_methods(baseline_df, finetuned_df, cot_df, optimized_df):
    """对比四种方法的预测结果"""
    
    questions = {
        'ECON1MOD': [1, 2, 3, 4],
        'UNITY': [1, 2],
        'GPT1': [1, 2, 3],
        'MOREGUNIMPACT': [1, 2, 3],
        'GAMBLERESTR': [1, 2, 3, 4, 5]
    }
    
    comparison_results = {}
    
    for question, valid_responses in questions.items():
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print('='*60)
        
        # 计算四种方法的指标
        baseline_metrics = calculate_accuracy_metrics(
            baseline_df, question, f'{question}_LLM', valid_responses
        )
        
        finetuned_metrics = calculate_accuracy_metrics(
            finetuned_df, question, f'{question}_LLM', valid_responses
        )
        
        cot_metrics = calculate_accuracy_metrics(
            cot_df, question, f'{question}_LLM', valid_responses
        )
        
        optimized_metrics = calculate_accuracy_metrics(
            optimized_df, question, f'{question}_LLM', valid_responses
        )
        
        if baseline_metrics is None or finetuned_metrics is None or cot_metrics is None or optimized_metrics is None:
            print(f"  跳过 {question}: 缺少有效数据")
            continue
        
        baseline_stats, _, _ = baseline_metrics
        finetuned_stats, _, _ = finetuned_metrics
        cot_stats, _, _ = cot_metrics
        optimized_stats, _, _ = optimized_metrics
        
        # 打印结果表格
        print(f"{'方法':<20} {'覆盖率':<10} {'准确率':<10} {'F1(macro)':<12} {'F1(weighted)':<12}")
        print("-" * 80)
        print(f"{'基础模型':<20} {baseline_stats['coverage']:.3f}     {baseline_stats['accuracy']:.3f}     {baseline_stats['f1_macro']:.3f}       {baseline_stats['f1_weighted']:.3f}")
        print(f"{'微调模型':<20} {finetuned_stats['coverage']:.3f}     {finetuned_stats['accuracy']:.3f}     {finetuned_stats['f1_macro']:.3f}       {finetuned_stats['f1_weighted']:.3f}")
        print(f"{'微调+CoT':<20} {cot_stats['coverage']:.3f}     {cot_stats['accuracy']:.3f}     {cot_stats['f1_macro']:.3f}       {cot_stats['f1_weighted']:.3f}")
        print(f"{'优化微调(lr0.6)':<20} {optimized_stats['coverage']:.3f}     {optimized_stats['accuracy']:.3f}     {optimized_stats['f1_macro']:.3f}       {optimized_stats['f1_weighted']:.3f}")
        
        print(f"\n相对于基础模型的改进:")
        print(f"  微调模型准确率: {finetuned_stats['accuracy'] - baseline_stats['accuracy']:+.3f}")
        print(f"  微调+CoT准确率: {cot_stats['accuracy'] - baseline_stats['accuracy']:+.3f}")
        print(f"  优化微调准确率: {optimized_stats['accuracy'] - baseline_stats['accuracy']:+.3f}")
        print(f"  微调模型F1: {finetuned_stats['f1_macro'] - baseline_stats['f1_macro']:+.3f}")
        print(f"  微调+CoT F1: {cot_stats['f1_macro'] - baseline_stats['f1_macro']:+.3f}")
        print(f"  优化微调F1: {optimized_stats['f1_macro'] - baseline_stats['f1_macro']:+.3f}")
        
        print(f"\n优化微调 vs 原微调:")
        print(f"  准确率改进: {optimized_stats['accuracy'] - finetuned_stats['accuracy']:+.3f}")
        print(f"  F1分数改进: {optimized_stats['f1_macro'] - finetuned_stats['f1_macro']:+.3f}")
        
        comparison_results[question] = {
            'baseline': baseline_stats,
            'finetuned': finetuned_stats,
            'cot': cot_stats,
            'optimized': optimized_stats,
            'improvements': {
                'finetuned_vs_baseline': {
                    'accuracy': finetuned_stats['accuracy'] - baseline_stats['accuracy'],
                    'f1_macro': finetuned_stats['f1_macro'] - baseline_stats['f1_macro'],
                    'coverage': finetuned_stats['coverage'] - baseline_stats['coverage']
                },
                'cot_vs_baseline': {
                    'accuracy': cot_stats['accuracy'] - baseline_stats['accuracy'],
                    'f1_macro': cot_stats['f1_macro'] - baseline_stats['f1_macro'],
                    'coverage': cot_stats['coverage'] - baseline_stats['coverage']
                },
                'cot_vs_finetuned': {
                    'accuracy': cot_stats['accuracy'] - finetuned_stats['accuracy'],
                    'f1_macro': cot_stats['f1_macro'] - finetuned_stats['f1_macro'],
                    'coverage': cot_stats['coverage'] - finetuned_stats['coverage']
                },
                'optimized_vs_baseline': {
                    'accuracy': optimized_stats['accuracy'] - baseline_stats['accuracy'],
                    'f1_macro': optimized_stats['f1_macro'] - baseline_stats['f1_macro'],
                    'coverage': optimized_stats['coverage'] - baseline_stats['coverage']
                },
                'optimized_vs_finetuned': {
                    'accuracy': optimized_stats['accuracy'] - finetuned_stats['accuracy'],
                    'f1_macro': optimized_stats['f1_macro'] - finetuned_stats['f1_macro'],
                    'coverage': optimized_stats['coverage'] - finetuned_stats['coverage']
                }
            }
        }
    
    return comparison_results

def generate_comprehensive_summary(comparison_results):
    """生成综合总结报告"""
    
    print("\n" + "="*80)
    print("综合总结报告")
    print("="*80)
    
    total_questions = len(comparison_results)
    
    # 统计各种改进情况
    finetuned_better_acc = sum(1 for q in comparison_results.values() 
                              if q['improvements']['finetuned_vs_baseline']['accuracy'] > 0)
    cot_better_acc = sum(1 for q in comparison_results.values() 
                        if q['improvements']['cot_vs_baseline']['accuracy'] > 0)
    cot_vs_finetuned_better = sum(1 for q in comparison_results.values() 
                                 if q['improvements']['cot_vs_finetuned']['accuracy'] > 0)
    
    finetuned_better_f1 = sum(1 for q in comparison_results.values() 
                             if q['improvements']['finetuned_vs_baseline']['f1_macro'] > 0)
    cot_better_f1 = sum(1 for q in comparison_results.values() 
                       if q['improvements']['cot_vs_baseline']['f1_macro'] > 0)
    cot_vs_finetuned_better_f1 = sum(1 for q in comparison_results.values() 
                                    if q['improvements']['cot_vs_finetuned']['f1_macro'] > 0)
    
    # 平均改进
    avg_finetuned_acc_improvement = np.mean([q['improvements']['finetuned_vs_baseline']['accuracy'] 
                                           for q in comparison_results.values()])
    avg_cot_acc_improvement = np.mean([q['improvements']['cot_vs_baseline']['accuracy'] 
                                     for q in comparison_results.values()])
    avg_cot_vs_finetuned_improvement = np.mean([q['improvements']['cot_vs_finetuned']['accuracy'] 
                                              for q in comparison_results.values()])
    
    avg_finetuned_f1_improvement = np.mean([q['improvements']['finetuned_vs_baseline']['f1_macro'] 
                                          for q in comparison_results.values()])
    avg_cot_f1_improvement = np.mean([q['improvements']['cot_vs_baseline']['f1_macro'] 
                                    for q in comparison_results.values()])
    avg_cot_vs_finetuned_f1_improvement = np.mean([q['improvements']['cot_vs_finetuned']['f1_macro'] 
                                                 for q in comparison_results.values()])
    
    print(f"测试问题总数: {total_questions}")
    print("\n准确率改进统计:")
    print(f"  微调模型 vs 基础模型: {finetuned_better_acc}/{total_questions} 问题改进 (平均: {avg_finetuned_acc_improvement:+.3f})")
    print(f"  微调+CoT vs 基础模型: {cot_better_acc}/{total_questions} 问题改进 (平均: {avg_cot_acc_improvement:+.3f})")
    print(f"  微调+CoT vs 微调模型: {cot_vs_finetuned_better}/{total_questions} 问题改进 (平均: {avg_cot_vs_finetuned_improvement:+.3f})")
    
    print("\nF1分数改进统计:")
    print(f"  微调模型 vs 基础模型: {finetuned_better_f1}/{total_questions} 问题改进 (平均: {avg_finetuned_f1_improvement:+.3f})")
    print(f"  微调+CoT vs 基础模型: {cot_better_f1}/{total_questions} 问题改进 (平均: {avg_cot_f1_improvement:+.3f})")
    print(f"  微调+CoT vs 微调模型: {cot_vs_finetuned_better_f1}/{total_questions} 问题改进 (平均: {avg_cot_vs_finetuned_f1_improvement:+.3f})")
    
    # 找出最佳和最差的改进
    best_overall_q = max(comparison_results.items(), 
                        key=lambda x: x[1]['improvements']['cot_vs_baseline']['accuracy'])
    worst_overall_q = min(comparison_results.items(), 
                         key=lambda x: x[1]['improvements']['cot_vs_baseline']['accuracy'])
    
    best_cot_additional_q = max(comparison_results.items(), 
                               key=lambda x: x[1]['improvements']['cot_vs_finetuned']['accuracy'])
    
    print(f"\n关键发现:")
    print(f"  最佳整体改进 (微调+CoT vs 基础): {best_overall_q[0]} ({best_overall_q[1]['improvements']['cot_vs_baseline']['accuracy']:+.3f})")
    print(f"  最差整体改进 (微调+CoT vs 基础): {worst_overall_q[0]} ({worst_overall_q[1]['improvements']['cot_vs_baseline']['accuracy']:+.3f})")
    print(f"  CoT最佳额外贡献: {best_cot_additional_q[0]} ({best_cot_additional_q[1]['improvements']['cot_vs_finetuned']['accuracy']:+.3f})")
    
    # 方法排名
    print(f"\n方法效果排名 (按平均准确率提升):")
    method_scores = {
        '基础模型': 0.0,
        '微调模型': avg_finetuned_acc_improvement,
        '微调+CoT': avg_cot_acc_improvement
    }
    
    ranked_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (method, score) in enumerate(ranked_methods, 1):
        print(f"  {i}. {method}: {score:+.3f}")

def create_visualization(comparison_results):
    """创建可视化图表"""
    
    questions = list(comparison_results.keys())
    
    # 准备数据
    baseline_acc = [comparison_results[q]['baseline']['accuracy'] for q in questions]
    finetuned_acc = [comparison_results[q]['finetuned']['accuracy'] for q in questions]
    cot_acc = [comparison_results[q]['cot']['accuracy'] for q in questions]
    optimized_acc = [comparison_results[q]['optimized']['accuracy'] for q in questions]
    
    baseline_f1 = [comparison_results[q]['baseline']['f1_macro'] for q in questions]
    finetuned_f1 = [comparison_results[q]['finetuned']['f1_macro'] for q in questions]
    cot_f1 = [comparison_results[q]['cot']['f1_macro'] for q in questions]
    optimized_f1 = [comparison_results[q]['optimized']['f1_macro'] for q in questions]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(questions))
    width = 0.2
    
    # 准确率对比
    ax1.bar(x - 1.5*width, baseline_acc, width, label='Baseline', alpha=0.8, color='skyblue')
    ax1.bar(x - 0.5*width, finetuned_acc, width, label='Finetuned', alpha=0.8, color='orange')
    ax1.bar(x + 0.5*width, cot_acc, width, label='CoT', alpha=0.8, color='green')
    ax1.bar(x + 1.5*width, optimized_acc, width, label='Optimized', alpha=0.8, color='red')
    
    ax1.set_xlabel('Questions')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison Across Methods')
    ax1.set_xticks(x)
    ax1.set_xticklabels(questions, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1分数对比
    ax2.bar(x - 1.5*width, baseline_f1, width, label='Baseline', alpha=0.8, color='skyblue')
    ax2.bar(x - 0.5*width, finetuned_f1, width, label='Finetuned', alpha=0.8, color='orange')
    ax2.bar(x + 0.5*width, cot_f1, width, label='CoT', alpha=0.8, color='green')
    ax2.bar(x + 1.5*width, optimized_f1, width, label='Optimized', alpha=0.8, color='red')
    
    ax2.set_xlabel('Questions')
    ax2.set_ylabel('F1 Score (macro)')
    ax2.set_title('F1 Score Comparison Across Methods')
    ax2.set_xticks(x)
    ax2.set_xticklabels(questions, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization chart saved as: prediction_methods_comparison.png")

def main():
    """主函数"""
    
    print("加载四个数据集...")
    baseline_df, finetuned_df, cot_df, optimized_df = load_datasets()
    
    print(f"基础模型数据: {len(baseline_df)} 行")
    print(f"微调模型数据: {len(finetuned_df)} 行")
    print(f"微调+CoT数据: {len(cot_df)} 行")
    print(f"优化微调模型数据: {len(optimized_df)} 行")
    
    # 对比分析
    comparison_results = compare_four_methods(baseline_df, finetuned_df, cot_df, optimized_df)
    
    # 生成综合报告
    generate_comprehensive_summary(comparison_results)
    
    # 创建可视化
    try:
        create_visualization(comparison_results)
    except Exception as e:
        print(f"Visualization creation failed: {e}")
    
    # 保存结果
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(comparison_results)
    with open('comprehensive_prediction_comparison.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nDetailed results saved to: comprehensive_prediction_comparison.json")

if __name__ == "__main__":
    main()