#!/usr/bin/env python3
"""
测试修复后的Adaptive PGAE CI计算
"""

import pandas as pd
import time
import numpy as np
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
    
    print("=" * 80)
    print("测试修复后的Adaptive PGAE CI计算")
    print("(adaptive采样 + regular PGAE CV计算CI)")
    print("=" * 80)
    
    # 使用最优参数配置
    estimator = AdaptivePGAEEstimator(
        X=target_config['X'],
        F=target_config['F'],
        Y=target_config['Y'],
        gamma=0.5,
        batch_size=250,
        design_update_freq=1,  # 最优参数
        warmup_batches=2       # 最优参数
    )
    
    print("\n🧪 运行10次实验测试CI性能...")
    print(f"   配置: batch_size=250, design_update_freq=1, warmup_batches=2")
    print(f"   CI计算: 使用regular PGAE的K-fold交叉验证 (K=3)")
    
    start_time = time.time()
    
    try:
        # 运行实验
        experiment_results = estimator.run_experiments(
            df,
            n_experiments=10,  # 快速测试
            n_true_labels=500,
            alpha=0.9,
            seed=42,
            use_concurrent=False  # 串行模式便于观察
        )
        
        end_time = time.time()
        
        print(f"\n✅ 实验完成! 耗时: {end_time - start_time:.1f}s")
        print(f"   MSE: {experiment_results['mse']:.6f}")
        print(f"   覆盖率: {experiment_results['coverage_rate']:.3f}")
        print(f"   平均CI长度: {experiment_results['avg_ci_length']:.4f}")
        print(f"   真实值: {experiment_results['true_value']:.6f}")
        
        # 检查CI性能改善
        coverage_rate = experiment_results['coverage_rate']
        target_coverage = 0.90
        
        print(f"\n📊 CI性能分析:")
        print(f"   目标覆盖率: {target_coverage:.1%}")
        print(f"   实际覆盖率: {coverage_rate:.1%}")
        
        if abs(coverage_rate - target_coverage) <= 0.05:
            print(f"   ✅ 覆盖率正常! (误差 ≤ 5%)")
        elif coverage_rate < target_coverage - 0.05:
            print(f"   ⚠️  覆盖率偏低 (差 {target_coverage - coverage_rate:.1%})")
        else:
            print(f"   ⚠️  覆盖率偏高 (超 {coverage_rate - target_coverage:.1%})")
        
        ci_length = experiment_results['avg_ci_length']
        print(f"   CI宽度: {ci_length:.4f}")
        
        if ci_length < 0.1:
            print(f"   ✅ CI宽度合理")
        else:
            print(f"   ⚠️  CI宽度偏大")
            
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n💡 修复方案总结:")
    print("1. ✅ 保持adaptive采样的优势（动态设计更新）")
    print("2. ✅ 采样完成后使用regular PGAE的K-fold CV计算CI")
    print("3. ✅ 应该获得更稳定和准确的置信区间")
    print("4. ✅ coverage rate应该接近目标90%")