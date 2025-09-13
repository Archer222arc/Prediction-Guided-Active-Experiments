#!/usr/bin/env python3
"""
对比微调模型和基础模型的预测准确性
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_datasets():
    """加载两个预测结果数据集"""
    
    # 加载微调模型结果 (Step 7)
    finetuned_df = pd.read_csv('NPORS_2024_LLM_finetuned_predictions_20250910_205048.csv')
    
    # 加载基础模型结果 (原始LLM prediction)
    baseline_df = pd.read_csv('data/NPORS_2024_for_public_release_basic_prompting.csv')
    
    return finetuned_df, baseline_df

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

def compare_questions(finetuned_df, baseline_df):
    """对比各个问题的预测结果"""
    
    questions = {
        'ECON1MOD': [1, 2, 3, 4],
        'UNITY': [1, 2],
        'GPT1': [1, 2, 3],
        'MOREGUNIMPACT': [1, 2, 3],
        'GAMBLERESTR': [1, 2, 3, 4, 5]
    }
    
    comparison_results = {}
    
    for question, valid_responses in questions.items():
        print(f"\n=== {question} ===")
        
        # 计算微调模型指标
        finetuned_metrics = calculate_accuracy_metrics(
            finetuned_df, question, f'{question}_LLM', valid_responses
        )
        
        # 计算基础模型指标
        baseline_metrics = calculate_accuracy_metrics(
            baseline_df, question, f'{question}_LLM', valid_responses
        )
        
        if finetuned_metrics is None or baseline_metrics is None:
            print(f"  跳过 {question}: 缺少有效数据")
            continue
        
        finetuned_stats, ft_true, ft_pred = finetuned_metrics
        baseline_stats, bl_true, bl_pred = baseline_metrics
        
        print(f"  微调模型:")
        print(f"    覆盖率: {finetuned_stats['coverage']:.3f}")
        print(f"    准确率: {finetuned_stats['accuracy']:.3f}")
        print(f"    F1 (macro): {finetuned_stats['f1_macro']:.3f}")
        
        print(f"  基础模型:")
        print(f"    覆盖率: {baseline_stats['coverage']:.3f}")
        print(f"    准确率: {baseline_stats['accuracy']:.3f}")
        print(f"    F1 (macro): {baseline_stats['f1_macro']:.3f}")
        
        print(f"  改进:")
        print(f"    准确率提升: {finetuned_stats['accuracy'] - baseline_stats['accuracy']:.3f}")
        print(f"    F1提升: {finetuned_stats['f1_macro'] - baseline_stats['f1_macro']:.3f}")
        
        comparison_results[question] = {
            'finetuned': finetuned_stats,
            'baseline': baseline_stats,
            'improvement': {
                'accuracy': finetuned_stats['accuracy'] - baseline_stats['accuracy'],
                'f1_macro': finetuned_stats['f1_macro'] - baseline_stats['f1_macro'],
                'coverage': finetuned_stats['coverage'] - baseline_stats['coverage']
            }
        }
    
    return comparison_results

def generate_summary_report(comparison_results):
    """生成总结报告"""
    
    print("\n" + "="*60)
    print("总结报告")
    print("="*60)
    
    total_questions = len(comparison_results)
    improved_accuracy = sum(1 for q in comparison_results.values() 
                          if q['improvement']['accuracy'] > 0)
    improved_f1 = sum(1 for q in comparison_results.values() 
                     if q['improvement']['f1_macro'] > 0)
    
    avg_accuracy_improvement = np.mean([q['improvement']['accuracy'] 
                                      for q in comparison_results.values()])
    avg_f1_improvement = np.mean([q['improvement']['f1_macro'] 
                                for q in comparison_results.values()])
    
    print(f"测试问题总数: {total_questions}")
    print(f"准确率改进的问题数: {improved_accuracy}/{total_questions}")
    print(f"F1分数改进的问题数: {improved_f1}/{total_questions}")
    print(f"平均准确率提升: {avg_accuracy_improvement:.3f}")
    print(f"平均F1分数提升: {avg_f1_improvement:.3f}")
    
    # 找出改进最大和最小的问题
    best_accuracy_q = max(comparison_results.items(), 
                         key=lambda x: x[1]['improvement']['accuracy'])
    worst_accuracy_q = min(comparison_results.items(), 
                          key=lambda x: x[1]['improvement']['accuracy'])
    
    print(f"\n最佳改进问题: {best_accuracy_q[0]} (+{best_accuracy_q[1]['improvement']['accuracy']:.3f})")
    print(f"最差改进问题: {worst_accuracy_q[0]} ({worst_accuracy_q[1]['improvement']['accuracy']:.3f})")

def main():
    """主函数"""
    
    print("加载数据集...")
    finetuned_df, baseline_df = load_datasets()
    
    print(f"微调模型数据: {len(finetuned_df)} 行")
    print(f"基础模型数据: {len(baseline_df)} 行")
    
    # 对比分析
    comparison_results = compare_questions(finetuned_df, baseline_df)
    
    # 生成报告
    generate_summary_report(comparison_results)
    
    # 保存结果 (转换numpy类型为Python原生类型)
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
    with open('prediction_comparison_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n详细结果已保存至: prediction_comparison_results.json")

if __name__ == "__main__":
    main()