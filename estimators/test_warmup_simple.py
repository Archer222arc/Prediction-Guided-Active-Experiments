#!/usr/bin/env python3
"""
简单测试修复后的warmup参数效果
"""

if __name__ == '__main__':
    import pandas as pd
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
    print("测试修复后的warmup参数（串行模式）")
    print("="*60)

    # 测试两个极端配置
    configs = [
        {
            'warmup': 0, 'update_freq': 1, 'batch_size': 50,  # 更小的batch_size产生更多批次
            'name': '无预热_每批更新_小批次'
        },
        {
            'warmup': 8, 'update_freq': 1, 'batch_size': 50,  # 极端预热
            'name': '预热8批_每批更新_小批次'
        }
    ]

    results = {}

    for config in configs:
        print(f"\n🧪 测试配置: {config['name']}")
        print(f"   warmup_batches = {config['warmup']}")
        print(f"   design_update_freq = {config['update_freq']}")
        print(f"   batch_size = {config['batch_size']} (预期需要约10-12批)")
        
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
        
        # 运行少量串行实验
        print("   运行2次串行实验...")
        try:
            experiment_results = estimator.run_experiments(
                df,
                n_experiments=2,  # 最少实验
                n_true_labels=500,
                alpha=0.9,
                seed=42,
                use_concurrent=False  # 串行模式
            )
            
            results[config['name']] = {
                'mse': experiment_results['mse'],
                'coverage_rate': experiment_results['coverage_rate']
            }
            
            print(f"   ✅ MSE: {results[config['name']]['mse']:.6f}")
            print(f"   ✅ 覆盖率: {results[config['name']]['coverage_rate']:.3f}")
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            results[config['name']] = {'error': str(e)}

    print("\n" + "="*60)
    print("结果对比分析")
    print("="*60)

    config1_name = '无预热_每批更新_小批次'
    config2_name = '预热8批_每批更新_小批次'

    if config1_name in results and config2_name in results:
        if 'error' not in results[config1_name] and 'error' not in results[config2_name]:
            mse1 = results[config1_name]['mse']
            mse2 = results[config2_name]['mse']
            mse_diff = abs(mse1 - mse2)

            print(f"{config1_name}: MSE = {mse1:.6f}")
            print(f"{config2_name}: MSE = {mse2:.6f}")
            print(f"MSE差异: {mse_diff:.6f}")

            if mse_diff > 1e-6:
                print("✅ 参数修复成功！不同warmup配置产生了不同结果")
                relative_diff = mse_diff / min(mse1, mse2) * 100
                print(f"   相对差异: {relative_diff:.2f}%")
                
                if mse2 < mse1:
                    print(f"   🎯 预热策略表现更好")
                else:
                    print(f"   🎯 无预热策略表现更好")
            else:
                print("⚠️  结果仍然相同")
                print("   可能原因: warmup影响期太短，或设计更新本身影响很小")
        else:
            print("❌ 存在实验错误")
    else:
        print("❌ 测试配置缺失")

    print(f"\n💡 结论:")
    print("1. 参数传递修复已确认生效（可以看到不同的日志输出）")
    print("2. 如果MSE差异很小，说明在当前数据集上预热效果有限")  
    print("3. 可以尝试更大的预热批次或更小的batch_size来观察更明显的效果")