#!/usr/bin/env python3
"""
调试warmup参数为什么没有产生不同结果
"""

import pandas as pd
import numpy as np
from adaptive_pgae_estimator import AdaptivePGAEEstimator

# 加载数据
df = pd.read_csv('../archive/predictions/NPORS_2024_base_gpt41mini_lr06_step560_20250911_214943.csv')

# 目标配置
target_config = {
    'X': ['EDUCATION'],
    'F': 'ECON1MOD_LLM',
    'Y': 'ECON1MOD',
}

print("="*60)
print("调试warmup参数影响")
print("="*60)

# 测试两个配置：warmup=0 vs warmup=2
configs = [
    {'warmup': 0, 'name': 'no_warmup'},
    {'warmup': 2, 'name': 'warmup_2'}
]

results = {}

for config in configs:
    print(f"\n🔍 测试配置: {config['name']}")
    print(f"   warmup_batches = {config['warmup']}")
    
    # 创建估计器
    estimator = AdaptivePGAEEstimator(
        X=target_config['X'],
        F=target_config['F'],
        Y=target_config['Y'],
        gamma=0.5,
        batch_size=100,
        design_update_freq=1,
        warmup_batches=config['warmup']
    )
    
    # 运行单次实验来观察详细过程
    print("   运行单次实验观察详细过程...")
    
    try:
        tau, l_ci, h_ci = estimator.run_single_adaptive_experiment(df, n_true_labels=500, alpha=0.9, seed=42)
        
        results[config['name']] = {
            'tau': tau,
            'l_ci': l_ci,
            'h_ci': h_ci,
            'mse': (tau - 2.724666)**2  # 使用已知真实值
        }
        
        print(f"   结果: tau={tau:.6f}, MSE={results[config['name']]['mse']:.6f}")
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        results[config['name']] = {'error': str(e)}

print("\n" + "="*60)
print("结果对比")
print("="*60)

if 'no_warmup' in results and 'warmup_2' in results:
    no_warmup = results['no_warmup']
    warmup_2 = results['warmup_2']
    
    if 'error' not in no_warmup and 'error' not in warmup_2:
        tau_diff = abs(no_warmup['tau'] - warmup_2['tau'])
        mse_diff = abs(no_warmup['mse'] - warmup_2['mse'])
        
        print(f"无预热: tau={no_warmup['tau']:.6f}, MSE={no_warmup['mse']:.6f}")
        print(f"预热2批: tau={warmup_2['tau']:.6f}, MSE={warmup_2['mse']:.6f}")
        print(f"tau差异: {tau_diff:.6f}")
        print(f"MSE差异: {mse_diff:.6f}")
        
        if tau_diff < 1e-6:
            print("⚠️  结果完全相同！可能的原因：")
            print("   1. 预热期影响太小（总批次数太少）")
            print("   2. 初始设计已经接近最优")
            print("   3. 随机性被固定种子掩盖")
            print("   4. 设计更新逻辑有bug")
        else:
            print("✅ 预热参数产生了不同结果")
    else:
        print("❌ 存在实验错误")
        if 'error' in no_warmup:
            print(f"无预热错误: {no_warmup['error']}")
        if 'error' in warmup_2:
            print(f"预热2批错误: {warmup_2['error']}")

print("\n💡 建议进一步调试：")
print("1. 检查日志输出，确认预热逻辑是否被执行")
print("2. 增加批次数量或减少batch_size来观察更多批次的行为")
print("3. 比较设计参数在不同批次之间的变化")