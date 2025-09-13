#!/usr/bin/env python3
"""
Adaptive PGAE调参脚本
Adaptive PGAE batch_size parameter tuning tool
优化batch_size参数以提升Adaptive PGAE性能
"""

import numpy as np
import pandas as pd
import time
import logging
import json
from typing import Dict, List
from tqdm import tqdm
from itertools import product

from adaptive_pgae_estimator import AdaptivePGAEEstimator
from utils import summary_results

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_adaptive_design_combinations(df: pd.DataFrame, batch_sizes: List[int], 
                                     gamma_values: List[float], design_update_freqs: List[int],
                                     warmup_batches_list: List[int], target_config: Dict) -> Dict:
    """
    测试Adaptive PGAE的改进设计参数组合
    
    Args:
        df: 数据DataFrame
        batch_sizes: 要测试的batch_size列表
        gamma_values: 要测试的gamma值列表
        design_update_freqs: 要测试的设计更新频率列表
        warmup_batches_list: 要测试的预热批次数列表
        target_config: 目标配置字典
    
    Returns:
        调参结果字典
    """
    total_combinations = len(batch_sizes) * len(gamma_values) * len(design_update_freqs) * len(warmup_batches_list)
    logger.info(f"测试Adaptive PGAE改进设计参数，总共{total_combinations}个组合")
    logger.info(f"batch_sizes: {batch_sizes}")
    logger.info(f"gamma_values: {gamma_values}")
    logger.info(f"design_update_freqs: {design_update_freqs}")
    logger.info(f"warmup_batches: {warmup_batches_list}")
    
    results = {}
    
    # 网格搜索所有参数组合
    for batch_size in tqdm(batch_sizes, desc="Testing batch sizes"):
        for gamma in gamma_values:
            for update_freq in design_update_freqs:
                for warmup in warmup_batches_list:
                    combo_name = f'bs_{batch_size}_g_{gamma}_uf_{update_freq}_wb_{warmup}'
                    
                    logger.info(f"\n测试组合: batch_size={batch_size}, gamma={gamma}, "
                               f"update_freq={update_freq}, warmup={warmup}")
        
                    try:
                        result = run_single_adaptive_design_test(
                            df, target_config, batch_size, gamma, update_freq, warmup, 
                            n_experiments=20  # 减少实验次数以加速测试
                        )
                        
                        results[combo_name] = {
                            'batch_size': batch_size,
                            'gamma': gamma,
                            'design_update_freq': update_freq,
                            'warmup_batches': warmup,
                            'mse': result['mse'],
                            'bias': result['bias'],
                            'variance': result['variance'],
                            'coverage_rate': result['coverage_rate'],
                            'avg_ci_length': result['avg_ci_length'],
                            'execution_time': result['execution_time'],
                            'true_value': result['true_value']
                        }
                        
                        logger.info(f"  MSE: {result['mse']:.6f}, "
                                   f"覆盖率: {result['coverage_rate']:.3f}, "
                                   f"时间: {result['execution_time']:.1f}s")
                                   
                    except Exception as e:
                        logger.error(f"  组合 {combo_name} 失败: {e}")
                        results[combo_name] = {
                            'batch_size': batch_size,
                            'gamma': gamma,
                            'design_update_freq': update_freq,
                            'warmup_batches': warmup,
                            'error': str(e)
                        }
    
    return results

def run_single_adaptive_design_test(df: pd.DataFrame, target_config: Dict, 
                                    batch_size: int, gamma: float, 
                                    design_update_freq: int, warmup_batches: int,
                                    n_experiments: int = 20) -> Dict:
    """
    运行单个Adaptive PGAE改进设计参数组合的测试
    
    Args:
        df: 数据DataFrame
        target_config: 目标配置
        batch_size: batch大小
        gamma: gamma参数
        design_update_freq: 设计更新频率
        warmup_batches: 预热批次数
        n_experiments: 实验次数
    
    Returns:
        实验结果字典
    """
    # 创建改进版估计器
    estimator = AdaptivePGAEEstimator(
        X=target_config['X'],
        F=target_config['F'],
        Y=target_config['Y'],
        gamma=gamma,
        batch_size=batch_size,
        design_update_freq=design_update_freq,
        warmup_batches=warmup_batches
    )
    
    # 运行实验
    start_time = time.time()
    experiment_results = estimator.run_experiments(
        df,
        n_experiments=n_experiments,
        n_true_labels=500,
        alpha=0.9,
        seed=42,
        use_concurrent=True,
        max_workers=10  # 限制worker数量以节省内存
    )
    end_time = time.time()
    
    return {
        'mse': experiment_results['mse'],
        'bias': experiment_results['bias'],
        'variance': experiment_results['variance'],
        'coverage_rate': experiment_results['coverage_rate'],
        'avg_ci_length': experiment_results['avg_ci_length'],
        'execution_time': end_time - start_time,
        'true_value': experiment_results['true_value']
    }

def run_single_batch_gamma_test(df: pd.DataFrame, target_config: Dict, 
                                batch_size: int, gamma: float, 
                                n_experiments: int = 30) -> Dict:
    """
    运行单个batch_size和gamma值组合的测试
    
    Args:
        df: 数据DataFrame
        target_config: 目标配置
        batch_size: batch大小
        gamma: gamma参数
        n_experiments: 实验次数
    
    Returns:
        实验结果字典
    """
    # 创建估计器
    estimator = AdaptivePGAEEstimator(
        X=target_config['X'],
        F=target_config['F'],
        Y=target_config['Y'],
        gamma=gamma,
        batch_size=batch_size
    )
    
    # 运行实验
    start_time = time.time()
    experiment_results = estimator.run_experiments(
        df,
        n_experiments=n_experiments,
        n_true_labels=500,
        alpha=0.9,
        seed=42,
        use_concurrent=True,
        max_workers=10  # 限制worker数量以节省内存
    )
    end_time = time.time()
    
    return {
        'mse': experiment_results['mse'],
        'bias': experiment_results['bias'],
        'variance': experiment_results['variance'],
        'coverage_rate': experiment_results['coverage_rate'],
        'avg_ci_length': experiment_results['avg_ci_length'],
        'execution_time': end_time - start_time,
        'true_value': experiment_results['true_value']
    }

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python tune_adaptive_pgae.py <data_file>")
        print("  python tune_adaptive_pgae.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv")
        return
    
    data_file = sys.argv[1]
    logger.info(f"加载数据: {data_file}")
    df = pd.read_csv(data_file)
    
    # 目标配置
    TARGET_CONFIGS = {
        'ECON1MOD': {
            'X': ['EDUCATION'],
            'F': 'ECON1MOD_LLM',
            'Y': 'ECON1MOD',
        },
        'UNITY': {
            'X': ['EDUCATION'],
            'F': 'UNITY_LLM', 
            'Y': 'UNITY',
        },
        'GPT1': {
            'X': ['EDUCATION'],
            'F': 'GPT1_LLM',
            'Y': 'GPT1',
        }
    }
    
    # 定义要测试的改进设计参数 - 固定gamma=0.5
    batch_sizes = [25, 50, 100, 150, 250]  # 精选batch_size范围
    gamma_values = [0.5]  # 固定gamma=0.5
    design_update_freqs = [1, 2, 3]  # 设计更新频率：每1,2,3批更新一次
    warmup_batches_list = [0, 2, 5]  # 预热批次：0,2,5批预热
    
    # 选择目标任务 (默认ECON1MOD)
    target = 'ECON1MOD'
    target_config = TARGET_CONFIGS[target]
    
    logger.info(f"目标任务: {target}")
    logger.info(f"配置: {target_config}")
    logger.info("测试Adaptive PGAE改进设计：更新频率控制 + 预热机制 (固定gamma=0.5)")
    
    total_combinations = len(batch_sizes) * len(gamma_values) * len(design_update_freqs) * len(warmup_batches_list)
    logger.info(f"将测试{total_combinations}个参数组合 (3×1×3×3=27)")
    
    # 运行改进设计参数测试
    results = test_adaptive_design_combinations(
        df, batch_sizes, gamma_values, design_update_freqs, warmup_batches_list, target_config
    )
    
    # 分析结果并输出
    print("\n" + "="*100)
    print("ADAPTIVE PGAE 改进设计参数调优结果")
    print("="*100)
    
    # 过滤有效结果
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        print("❌ 所有参数组合测试均失败")
        return
    
    # 按MSE排序显示结果
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['mse'])
    
    print(f"{'Batch':<6} {'Update':<7} {'Warmup':<7} {'MSE':<12} {'Coverage':<10} {'Time(s)':<8}")
    print("-" * 70)
    
    for combo_name, combo_data in sorted_results[:15]:  # 显示前15个最佳结果（更多结果）
        batch_size = combo_data['batch_size']
        update_freq = combo_data['design_update_freq']
        warmup = combo_data['warmup_batches']
        mse = combo_data['mse']
        coverage = combo_data['coverage_rate']
        time_taken = combo_data['execution_time']
        
        print(f"{batch_size:<6} {update_freq:<7} {warmup:<7} {mse:<12.6f} {coverage:<10.3f} {time_taken:<8.1f}")
    
    # 找到最佳参数组合
    best_combo_name, best_combo_data = sorted_results[0]
    best_batch_size = best_combo_data['batch_size']
    best_update_freq = best_combo_data['design_update_freq']
    best_warmup = best_combo_data['warmup_batches']
    best_mse = best_combo_data['mse']
    best_coverage = best_combo_data['coverage_rate']
    
    print("\n" + "="*70)
    print(f"🏆 最佳改进设计组合 (gamma=0.5):")
    print(f"   batch_size={best_batch_size}")
    print(f"   设计更新频率={best_update_freq}批, 预热批次={best_warmup}")
    print(f"   MSE: {best_mse:.6f}")
    print(f"   覆盖率: {best_coverage:.3f}")
    print(f"   原始baseline: MSE ≈ 0.004201")
    
    # 计算改善程度
    current_mse = 0.004201  # 当前Adaptive PGAE的MSE
    improvement = ((current_mse - best_mse) / current_mse) * 100
    
    print(f"\n📝 应用建议:")
    print(f"   更新compare_estimators.py中的Adaptive PGAE参数:")
    print(f"   - gamma=0.5 (固定)")
    print(f"   - batch_size={best_batch_size}")  
    print(f"   - design_update_freq={best_update_freq} (每{best_update_freq}批更新一次设计)")
    print(f"   - warmup_batches={best_warmup} (前{best_warmup}批使用固定设计)")
    print(f"   预期MSE改善: {improvement:.1f}% (从 {current_mse:.6f} 降到 {best_mse:.6f})")
    
    # 如果改善显著，给出更具体的建议
    if improvement > 20:
        print(f"   ✅ 显著改善！改进设计大幅提升Adaptive PGAE性能")
    elif improvement > 10:
        print(f"   ✅ 中等改善，设计优化带来明显提升")  
    else:
        print(f"   ⚠️  改善有限，但设计更稳定（减少波动）")
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f'adaptive_pgae_design_tuning_{target.lower()}_{timestamp}.json'
    
    # 添加总结到结果中
    results['summary'] = {
        'target': target,
        'fixed_gamma': 0.5,  # 固定gamma值
        'best_batch_size': best_batch_size,
        'best_design_update_freq': best_update_freq,
        'best_warmup_batches': best_warmup,
        'best_mse': best_mse,
        'best_coverage': best_coverage,
        'improvement_percent': improvement,
        'tested_parameters': {
            'batch_sizes': batch_sizes,
            'gamma_values': gamma_values,
            'design_update_freqs': design_update_freqs,
            'warmup_batches_list': warmup_batches_list
        },
        'total_combinations': len(batch_sizes) * len(gamma_values) * len(design_update_freqs) * len(warmup_batches_list),
        'successful_tests': len(valid_results)
    }
    
    # 转换numpy类型为python原生类型 (参考tune_internal_parameters.py)
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    print(f"\n✅ Adaptive PGAE改进设计调参完成!")
    print(f"测试了{len(valid_results)}个有效参数组合")
    print(f"结果文件: {output_file}")

if __name__ == "__main__":
    main()
