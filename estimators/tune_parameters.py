#!/usr/bin/env python3
"""
PGAE参数调优工具
Parameter tuning tool for PGAE estimators
"""

import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import logging
from itertools import product
from compare_estimators import run_notebook_style_comparison

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def tune_gamma_parameter(df: pd.DataFrame, 
                        gamma_values: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
                        n_experiments: int = 50,
                        n_true_labels: int = 500,
                        alpha: float = 0.90) -> Dict:
    """
    调优gamma参数
    
    Args:
        df: 输入数据框
        gamma_values: 要测试的gamma值列表
        n_experiments: 每个gamma值的实验次数
        n_true_labels: 每次实验的真实标签数量
        alpha: 置信水平
        
    Returns:
        调优结果字典
    """
    logger.info(f"开始gamma参数调优: 测试值 {gamma_values}")
    
    results = {}
    
    for gamma in tqdm(gamma_values, desc="调优gamma参数"):
        logger.info(f"\n测试 gamma = {gamma}")
        
        # 运行对比实验
        comparison_results = run_notebook_style_comparison(
            df, 
            gamma=gamma,
            n_experiments=n_experiments,
            n_true_labels=n_true_labels,
            alpha=alpha,
            seed=42
        )
        
        # 提取PGAE结果
        if 'PGAE' in comparison_results:
            pgae_result = comparison_results['PGAE']
            results[gamma] = {
                'gamma': gamma,
                'mse': pgae_result['mse'],
                'bias': pgae_result['bias'],
                'variance': pgae_result['variance'],
                'coverage_rate': pgae_result['coverage_rate'],
                'avg_ci_length': pgae_result['avg_ci_length'],
                'execution_time': pgae_result['execution_time']
            }
            
            logger.info(f"Gamma {gamma}: MSE={pgae_result['mse']:.6f}, "
                       f"覆盖率={pgae_result['coverage_rate']:.3f}, "
                       f"CI长度={pgae_result['avg_ci_length']:.4f}")
    
    return results

def tune_alpha_parameter(df: pd.DataFrame,
                        alpha_values: List[float] = [0.85, 0.90, 0.95],
                        gamma: float = 0.5,
                        n_experiments: int = 50,
                        n_true_labels: int = 500) -> Dict:
    """
    调优置信水平alpha参数
    
    Args:
        df: 输入数据框
        alpha_values: 要测试的alpha值列表
        gamma: 固定的gamma值
        n_experiments: 每个alpha值的实验次数
        n_true_labels: 每次实验的真实标签数量
        
    Returns:
        调优结果字典
    """
    logger.info(f"开始alpha参数调优: 测试值 {alpha_values}")
    
    results = {}
    
    for alpha in tqdm(alpha_values, desc="调优alpha参数"):
        logger.info(f"\n测试 alpha = {alpha}")
        
        # 运行对比实验
        comparison_results = run_notebook_style_comparison(
            df, 
            gamma=gamma,
            n_experiments=n_experiments,
            n_true_labels=n_true_labels,
            alpha=alpha,
            seed=42
        )
        
        # 提取PGAE结果
        if 'PGAE' in comparison_results:
            pgae_result = comparison_results['PGAE']
            results[alpha] = {
                'alpha': alpha,
                'mse': pgae_result['mse'],
                'bias': pgae_result['bias'],
                'variance': pgae_result['variance'],
                'coverage_rate': pgae_result['coverage_rate'],
                'avg_ci_length': pgae_result['avg_ci_length'],
                'execution_time': pgae_result['execution_time']
            }
            
            logger.info(f"Alpha {alpha}: MSE={pgae_result['mse']:.6f}, "
                       f"覆盖率={pgae_result['coverage_rate']:.3f}, "
                       f"CI长度={pgae_result['avg_ci_length']:.4f}")
    
    return results

def tune_sample_size(df: pd.DataFrame,
                    n_true_labels_values: List[int] = [300, 500, 700, 1000],
                    gamma: float = 0.5,
                    n_experiments: int = 50,
                    alpha: float = 0.90) -> Dict:
    """
    调优样本数量参数
    
    Args:
        df: 输入数据框
        n_true_labels_values: 要测试的样本数量列表
        gamma: 固定的gamma值
        n_experiments: 实验次数
        alpha: 置信水平
        
    Returns:
        调优结果字典
    """
    logger.info(f"开始样本数量调优: 测试值 {n_true_labels_values}")
    
    results = {}
    
    for n_labels in tqdm(n_true_labels_values, desc="调优样本数量"):
        logger.info(f"\n测试 n_true_labels = {n_labels}")
        
        # 运行对比实验
        comparison_results = run_notebook_style_comparison(
            df, 
            gamma=gamma,
            n_experiments=n_experiments,
            n_true_labels=n_labels,
            alpha=alpha,
            seed=42
        )
        
        # 提取PGAE结果
        if 'PGAE' in comparison_results:
            pgae_result = comparison_results['PGAE']
            results[n_labels] = {
                'n_true_labels': n_labels,
                'mse': pgae_result['mse'],
                'bias': pgae_result['bias'],
                'variance': pgae_result['variance'],
                'coverage_rate': pgae_result['coverage_rate'],
                'avg_ci_length': pgae_result['avg_ci_length'],
                'execution_time': pgae_result['execution_time']
            }
            
            logger.info(f"样本数 {n_labels}: MSE={pgae_result['mse']:.6f}, "
                       f"覆盖率={pgae_result['coverage_rate']:.3f}, "
                       f"CI长度={pgae_result['avg_ci_length']:.4f}")
    
    return results

def comprehensive_parameter_search(df: pd.DataFrame,
                                 gamma_values: List[float] = [0.3, 0.5, 0.7],
                                 alpha_values: List[float] = [0.90, 0.95],
                                 n_experiments: int = 30) -> Dict:
    """
    综合参数搜索
    
    Args:
        df: 输入数据框
        gamma_values: gamma值列表
        alpha_values: alpha值列表
        n_experiments: 实验次数（较少以节省时间）
        
    Returns:
        所有参数组合的结果
    """
    logger.info("开始综合参数搜索...")
    
    results = {}
    param_combinations = list(product(gamma_values, alpha_values))
    
    for gamma, alpha in tqdm(param_combinations, desc="参数组合搜索"):
        logger.info(f"\n测试组合: gamma={gamma}, alpha={alpha}")
        
        # 运行对比实验
        comparison_results = run_notebook_style_comparison(
            df, 
            gamma=gamma,
            n_experiments=n_experiments,
            n_true_labels=500,
            alpha=alpha,
            seed=42
        )
        
        # 提取PGAE结果
        if 'PGAE' in comparison_results:
            pgae_result = comparison_results['PGAE']
            
            # 计算综合评分（权衡MSE和覆盖率）
            mse_score = pgae_result['mse']
            coverage_penalty = max(0, 0.90 - pgae_result['coverage_rate']) * 10  # 覆盖率低于90%的惩罚
            composite_score = mse_score + coverage_penalty
            
            results[f"gamma_{gamma}_alpha_{alpha}"] = {
                'gamma': gamma,
                'alpha': alpha,
                'mse': pgae_result['mse'],
                'bias': pgae_result['bias'],
                'variance': pgae_result['variance'],
                'coverage_rate': pgae_result['coverage_rate'],
                'avg_ci_length': pgae_result['avg_ci_length'],
                'execution_time': pgae_result['execution_time'],
                'composite_score': composite_score
            }
            
            logger.info(f"组合 ({gamma}, {alpha}): MSE={pgae_result['mse']:.6f}, "
                       f"覆盖率={pgae_result['coverage_rate']:.3f}, "
                       f"综合评分={composite_score:.6f}")
    
    return results

def visualize_tuning_results(results: Dict, parameter_name: str, output_file: str = None):
    """
    可视化调优结果
    
    Args:
        results: 调优结果字典
        parameter_name: 参数名称
        output_file: 输出文件名
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{parameter_name} Parameter Tuning Results', fontsize=16, fontweight='bold')
    
    param_values = []
    mse_values = []
    coverage_values = []
    ci_length_values = []
    
    for key, result in results.items():
        if parameter_name == 'gamma':
            param_values.append(result['gamma'])
        elif parameter_name == 'alpha':
            param_values.append(result['alpha'])
        elif parameter_name == 'n_true_labels':
            param_values.append(result['n_true_labels'])
        
        mse_values.append(result['mse'])
        coverage_values.append(result['coverage_rate'])
        ci_length_values.append(result['avg_ci_length'])
    
    # MSE vs 参数
    axes[0, 0].plot(param_values, mse_values, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_title(f'MSE vs {parameter_name}')
    axes[0, 0].set_xlabel(parameter_name)
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 覆盖率 vs 参数
    axes[0, 1].plot(param_values, coverage_values, 'o-', color='green', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Target (90%)')
    axes[0, 1].set_title(f'Coverage Rate vs {parameter_name}')
    axes[0, 1].set_xlabel(parameter_name)
    axes[0, 1].set_ylabel('Coverage Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # CI长度 vs 参数
    axes[1, 0].plot(param_values, ci_length_values, 'o-', color='orange', linewidth=2, markersize=8)
    axes[1, 0].set_title(f'CI Length vs {parameter_name}')
    axes[1, 0].set_xlabel(parameter_name)
    axes[1, 0].set_ylabel('CI Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    # MSE vs 覆盖率散点图
    axes[1, 1].scatter(coverage_values, mse_values, s=100, alpha=0.7)
    for i, param in enumerate(param_values):
        axes[1, 1].annotate(f'{param}', (coverage_values[i], mse_values[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 1].axvline(x=0.9, color='red', linestyle='--', alpha=0.7, label='Target Coverage')
    axes[1, 1].set_title('MSE vs Coverage Rate Trade-off')
    axes[1, 1].set_xlabel('Coverage Rate')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"调优结果图表已保存: {output_file}")
    
    plt.show()

def find_optimal_parameters(results: Dict) -> Tuple[str, Dict]:
    """
    找出最佳参数组合
    
    Args:
        results: 参数搜索结果
        
    Returns:
        最佳参数组合的键和详细结果
    """
    best_key = None
    best_score = float('inf')
    
    for key, result in results.items():
        # 优先考虑覆盖率达到90%的结果
        if result['coverage_rate'] >= 0.90:
            score = result['mse']  # 在满足覆盖率的前提下最小化MSE
        else:
            score = result['mse'] + (0.90 - result['coverage_rate']) * 0.01  # 覆盖率不足的惩罚
        
        if score < best_score:
            best_score = score
            best_key = key
    
    return best_key, results[best_key] if best_key else None

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python tune_parameters.py <data_file> [tuning_type]")
        print("  python tune_parameters.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv gamma")
        print("  python tune_parameters.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv alpha")
        print("  python tune_parameters.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv sample_size")
        print("  python tune_parameters.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv comprehensive")
        return
    
    data_file = sys.argv[1]
    tuning_type = sys.argv[2] if len(sys.argv) > 2 else 'gamma'
    
    logger.info(f"加载数据: {data_file}")
    df = pd.read_csv(data_file)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if tuning_type == 'gamma':
        results = tune_gamma_parameter(df, n_experiments=50)
        visualize_tuning_results(results, 'gamma', f'gamma_tuning_{timestamp}.png')
        
        # 保存结果
        with open(f'gamma_tuning_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    elif tuning_type == 'alpha':
        results = tune_alpha_parameter(df, n_experiments=50)
        visualize_tuning_results(results, 'alpha', f'alpha_tuning_{timestamp}.png')
        
        # 保存结果
        with open(f'alpha_tuning_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    elif tuning_type == 'sample_size':
        results = tune_sample_size(df, n_experiments=50)
        visualize_tuning_results(results, 'n_true_labels', f'sample_size_tuning_{timestamp}.png')
        
        # 保存结果
        with open(f'sample_size_tuning_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    elif tuning_type == 'comprehensive':
        results = comprehensive_parameter_search(df, n_experiments=30)
        
        # 找出最佳参数
        best_key, best_result = find_optimal_parameters(results)
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE PARAMETER TUNING RESULTS")
        print(f"{'='*80}")
        
        if best_result:
            print(f"最佳参数组合: gamma={best_result['gamma']}, alpha={best_result['alpha']}")
            print(f"MSE: {best_result['mse']:.6f}")
            print(f"Coverage Rate: {best_result['coverage_rate']:.4f}")
            print(f"CI Length: {best_result['avg_ci_length']:.4f}")
            print(f"Composite Score: {best_result['composite_score']:.6f}")
        
        print(f"\n所有结果:")
        for key, result in sorted(results.items(), key=lambda x: x[1]['composite_score']):
            print(f"{key}: MSE={result['mse']:.6f}, Coverage={result['coverage_rate']:.3f}, "
                  f"CI={result['avg_ci_length']:.4f}, Score={result['composite_score']:.6f}")
        
        # 保存结果
        with open(f'comprehensive_tuning_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"{'='*80}")
    
    print(f"\n✅ {tuning_type}参数调优完成!")

if __name__ == "__main__":
    main()