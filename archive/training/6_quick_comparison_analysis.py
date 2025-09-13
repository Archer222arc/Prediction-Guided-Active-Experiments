#!/usr/bin/env python3
"""
快速对比分析 - 将微调结果与基线结果合并分析
"""

import pandas as pd
import numpy as np
import json

def merge_and_analyze_results():
    """合并微调结果和基线结果并分析"""
    
    # 加载微调预测结果
    print("加载微调预测结果...")
    finetuned_df = pd.read_csv('comparison_results_sample_20.csv')
    
    # 加载基线预测结果
    print("加载基线预测结果...")
    baseline_df = pd.read_csv('data/NPORS_2024_for_public_release_basic_prompting.csv')
    
    # 合并数据 (基于RESPID)
    print("合并数据...")
    merged_df = finetuned_df.merge(
        baseline_df[['RESPID', 'ECON1MOD_LLM', 'UNITY_LLM', 'GPT1_LLM', 'MOREGUNIMPACT_LLM', 'GAMBLERESTR_LLM']], 
        on='RESPID', 
        how='inner'
    )
    
    print(f"合并后数据: {len(merged_df)} 条记录")
    
    # 问题定义
    questions = {
        'ECON1MOD': "How would you rate the economic conditions in your community today?",
        'UNITY': "Which statement comes closer to your own view about American values?",
        'GPT1': "Have you heard of ChatGPT?",
        'MOREGUNIMPACT': "If more Americans owned guns, do you think there would be...",
        'GAMBLERESTR': "How much government regulation of gambling do you favor?"
    }
    
    results = {}
    
    print("\n" + "="*80)
    print("微调模型 vs 基线模型对比分析")
    print("="*80)
    
    for question_id in questions.keys():
        baseline_col = f'{question_id}_LLM'
        finetuned_col = f'{question_id}_FINETUNED'
        actual_col = question_id
        
        # 过滤有效数据
        valid_mask = (
            merged_df[actual_col].notna() & 
            (merged_df[actual_col] != 99.0) &
            merged_df[baseline_col].notna() &
            merged_df[finetuned_col].notna()
        )
        
        if valid_mask.sum() == 0:
            print(f"\n❌ {question_id}: 没有有效数据")
            continue
        
        actual = merged_df.loc[valid_mask, actual_col].astype(int)
        baseline = merged_df.loc[valid_mask, baseline_col].astype(int) 
        finetuned = merged_df.loc[valid_mask, finetuned_col].astype(int)
        
        # 计算准确率
        baseline_accuracy = (actual == baseline).mean()
        finetuned_accuracy = (actual == finetuned).mean()
        accuracy_improvement = finetuned_accuracy - baseline_accuracy
        
        # 计算各选项的分布
        def get_distribution(values):
            return pd.Series(values).value_counts(normalize=True).sort_index()
        
        actual_dist = get_distribution(actual)
        baseline_dist = get_distribution(baseline)
        finetuned_dist = get_distribution(finetuned)
        
        # 计算MAE (平均绝对误差)
        baseline_mae = np.mean(np.abs(actual - baseline))
        finetuned_mae = np.mean(np.abs(actual - finetuned))
        
        print(f"\n📊 {question_id}")
        print(f"   问题: {questions[question_id]}")
        print(f"   样本数: {len(actual)}")
        print(f"   基线准确率: {baseline_accuracy:.3f}")
        print(f"   微调准确率: {finetuned_accuracy:.3f}")
        print(f"   准确率提升: {accuracy_improvement:+.3f} ({accuracy_improvement*100:+.1f}%)")
        print(f"   基线MAE: {baseline_mae:.3f}")
        print(f"   微调MAE: {finetuned_mae:.3f}")
        
        # 显示一些具体预测对比
        print(f"   具体对比示例 (实际|基线|微调):")
        sample_indices = actual.index[:5]  # 显示前5个
        for idx in sample_indices:
            a = actual.loc[idx]
            b = baseline.loc[idx] 
            f = finetuned.loc[idx]
            status_b = "✅" if a == b else "❌"
            status_f = "✅" if a == f else "❌"
            print(f"     {a}|{b}{status_b}|{f}{status_f}")
        
        if accuracy_improvement > 0.05:
            print("   🎉 微调显著提升!")
        elif accuracy_improvement > 0:
            print("   ✅ 微调有所提升")
        elif accuracy_improvement < -0.05:
            print("   ⚠️ 微调显著下降")
        else:
            print("   ➖ 变化不大")
        
        results[question_id] = {
            'sample_size': len(actual),
            'baseline_accuracy': baseline_accuracy,
            'finetuned_accuracy': finetuned_accuracy,
            'accuracy_improvement': accuracy_improvement,
            'baseline_mae': baseline_mae,
            'finetuned_mae': finetuned_mae
        }
    
    # 总体统计
    if results:
        avg_baseline_acc = np.mean([r['baseline_accuracy'] for r in results.values()])
        avg_finetuned_acc = np.mean([r['finetuned_accuracy'] for r in results.values()])
        avg_improvement = avg_finetuned_acc - avg_baseline_acc
        
        print(f"\n🎯 总体表现:")
        print(f"   平均基线准确率: {avg_baseline_acc:.3f}")
        print(f"   平均微调准确率: {avg_finetuned_acc:.3f}")
        print(f"   平均准确率提升: {avg_improvement:+.3f} ({avg_improvement*100:+.1f}%)")
        
        if avg_improvement > 0.03:
            print("   🎉 微调带来了显著提升!")
        elif avg_improvement > 0:
            print("   ✅ 微调有积极效果")
        else:
            print("   ❌ 微调效果不明显")
    
    # 保存分析结果
    with open('comparison_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n分析结果已保存到: comparison_analysis_results.json")
    
    return results

if __name__ == "__main__":
    merge_and_analyze_results()