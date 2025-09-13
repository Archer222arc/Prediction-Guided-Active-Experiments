#!/usr/bin/env python3
"""
测试修复后的warmup和update_freq参数
"""

import pandas as pd
from adaptive_pgae_estimator import AdaptivePGAEEstimator

if __name__ == '__main__':

# 加载数据
df = pd.read_csv('../archive/predictions/NPORS_2024_base_gpt41mini_lr06_step560_20250911_214943.csv')

# 目标配置
target_config = {
    'X': ['EDUCATION'],
    'F': 'ECON1MOD_LLM', 
    'Y': 'ECON1MOD',
}

print("="*60)
print("测试修复后的Adaptive PGAE参数传递")
print("="*60)

# 测试极端不同的两个配置
configs = [
    {
        'warmup': 0, 'update_freq': 1, 'batch_size': 100,
        'name': '无预热_每批更新'
    },
    {
        'warmup': 5, 'update_freq': 3, 'batch_size': 100, 
        'name': '预热5批_每3批更新'
    }
]

results = {}

for config in configs:
    print(f"\n🧪 测试配置: {config['name']}")
    print(f"   warmup_batches = {config['warmup']}")
    print(f"   design_update_freq = {config['update_freq']}")
    print(f"   batch_size = {config['batch_size']}")
    
    # 创建估计器
    estimator = AdaptivePGAEEstimator(
        X=target_config['X'],
        F=target_config['F'],
        Y=target_config['Y'],
        gamma=0.5,
        batch_size=config['batch_size'],
        design_update_freq=config['update_freq'],
        warmup_batches=config['warmup']
    )
    
    # 运行少量串行实验来验证参数传递
    print("   运行3次串行实验...")
    experiment_results = estimator.run_experiments(
        df,
        n_experiments=3,  # 少量实验快速测试
        n_true_labels=500,
        alpha=0.9,
        seed=42,
        use_concurrent=False  # 使用串行模式避免多进程问题
    )
    
    results[config['name']] = {
        'mse': experiment_results['mse'],
        'tau_mean': experiment_results.get('tau_mean', 0),
        'coverage_rate': experiment_results['coverage_rate']
    }
    
    print(f"   ✅ MSE: {results[config['name']]['mse']:.6f}")
    print(f"   ✅ 覆盖率: {results[config['name']]['coverage_rate']:.3f}")

print("\n" + "="*60)
print("结果对比分析")
print("="*60)

config1_name = '无预热_每批更新'
config2_name = '预热5批_每3批更新'

mse1 = results[config1_name]['mse']
mse2 = results[config2_name]['mse']
mse_diff = abs(mse1 - mse2)

print(f"{config1_name}: MSE = {mse1:.6f}")
print(f"{config2_name}: MSE = {mse2:.6f}")
print(f"MSE差异: {mse_diff:.6f}")

if mse_diff > 1e-6:
    print("✅ 参数修复成功！不同配置产生了不同结果")
    relative_diff = mse_diff / min(mse1, mse2) * 100
    print(f"   相对差异: {relative_diff:.2f}%")
    
    if mse2 < mse1:
        print(f"   🎯 预热+频率控制策略表现更好")
    else:
        print(f"   🎯 无预热频繁更新策略表现更好")
else:
    print("⚠️  结果仍然相同，可能需要进一步调试")

print(f"\n💡 建议: 现在可以运行完整的调参脚本来寻找最优参数组合")