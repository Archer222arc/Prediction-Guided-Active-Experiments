#!/usr/bin/env python3
"""
统计估计器对比工具 - 参照notebook实现
Compare different statistical estimators - matching notebook implementation
"""

import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Optional
import logging
from scipy.stats import norm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import psutil

# 导入估计器和工具函数
from pgae_estimator import PGAEEstimator
from adaptive_pgae_estimator import AdaptivePGAEEstimator
from active_inference_estimator import ActiveInferenceEstimator
from naive_estimator import NaiveEstimator
from utils import (
    get_PGAE_design, rejection_sample, PGAE_est_ci, overwrite_merge,
    summary_results, validate_data
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置绘图风格
plt.style.use('default')
sns.set_palette("husl")

def run_notebook_style_comparison(df: pd.DataFrame, 
                                 target: str = 'ECON1MOD',
                                 X: List[str] = None,
                                 F: str = None, 
                                 Y: str = None,
                                 gamma: float = 0.5,
                                 n_experiments: int = 1000,
                                 n_true_labels: int = 500,
                                 alpha: float = 0.90,
                                 seed: Optional[int] = 42) -> Dict:
    """
    运行notebook风格的估计器对比 - 支持不同预测目标
    
    Args:
        df: 输入数据框
        target: 预测目标任务名称 ('ECON1MOD', 'UNITY', 'GPT1', 'MOREGUNIMPACT', 'GAMBLERESTR')
        X: 协变量列名列表 (如果为None，将使用默认配置)
        F: 预测列名 (如果为None，将使用默认配置) 
        Y: 真实标签列名 (如果为None，将使用默认配置)
        gamma: PGAE参数
        n_experiments: 实验次数
        n_true_labels: 每次实验的真实标签数量
        alpha: 置信水平
        seed: 随机种子
        
    Returns:
        包含所有估计器结果的字典
    """
    
    # 预测目标配置 - 基于notebook和预测质量分析
    TARGET_CONFIGS = {
        'ECON1MOD': {
            'X': ['EDUCATION'],
            'F': 'ECON1MOD_LLM',
            'Y': 'ECON1MOD',
            'description': 'Economic conditions rating (1-4)',
            'optimal_method': 'finetuned'
        },
        'UNITY': {
            'X': ['EDUCATION'],  
            'F': 'UNITY_LLM',
            'Y': 'UNITY',
            'description': 'American unity perception (1-2)',
            'optimal_method': 'baseline'
        },
        'GPT1': {
            'X': ['EDUCATION'],
            'F': 'GPT1_LLM', 
            'Y': 'GPT1',
            'description': 'ChatGPT awareness (1-3)',
            'optimal_method': 'optimized'
        },
        'MOREGUNIMPACT': {
            'X': ['EDUCATION'],
            'F': 'MOREGUNIMPACT_LLM',
            'Y': 'MOREGUNIMPACT', 
            'description': 'More guns crime impact (1-3)',
            'optimal_method': 'finetuned'
        },
        'GAMBLERESTR': {
            'X': ['EDUCATION'],
            'F': 'GAMBLERESTR_LLM',
            'Y': 'GAMBLERESTR',
            'description': 'Gambling restrictions view (1-3)', 
            'optimal_method': 'finetuned'
        }
    }
    
    # 使用目标配置或用户覆盖
    if target not in TARGET_CONFIGS:
        raise ValueError(f"Unsupported target: {target}. Available targets: {list(TARGET_CONFIGS.keys())}")
    
    config = TARGET_CONFIGS[target]
    X = X if X is not None else config['X']
    F = F if F is not None else config['F'] 
    Y = Y if Y is not None else config['Y']
    
    logger.info(f"运行 {target} 预测目标对比")
    logger.info(f"描述: {config['description']}")
    logger.info(f"推荐预测方法: {config['optimal_method']}")
    logger.info(f"X={X}, F={F}, Y={Y}")
    logger.info("开始notebook风格的统计估计器对比")
    
    # 数据预处理 - 完全按照notebook
    df_clean = df[X + [F] + [Y]].copy()
    df_clean = df_clean[df_clean[X + [F] + [Y]].lt(10).all(axis=1)]
    logger.info(f"数据清理后样本数: {len(df_clean)}")
    
    # 计算true_pmf - 按照notebook
    group_stats = df_clean.groupby(X).agg(
        cnt=(F, 'count'),
    ).reset_index()
    group_stats['true_pmf'] = group_stats['cnt'] / group_stats['cnt'].sum()
    df_clean = df_clean.merge(group_stats, on=X, how='left')
    
    # 计算真实值
    true_value = df_clean[Y].mean()
    logger.info(f"真实目标值: {true_value:.6f}")
    
    if seed is not None:
        np.random.seed(seed)
    
    # 初始化结果数组
    results = {}
    
    # 1. PGAE方法 - 使用并发优化的PGAEEstimator
    logger.info("运行PGAE估计器...")
    pgae_estimator = PGAEEstimator(X=X, F=F, Y=Y, gamma=gamma)
    
    # 使用并发执行PGAE实验
    pgae_results = pgae_estimator.run_experiments(
        df_clean, 
        n_experiments=n_experiments,
        n_true_labels=n_true_labels,
        alpha=alpha,
        seed=seed,
        use_concurrent=True,
        max_workers=10  # 限制worker数量以节省内存
    )
    
    results['PGAE'] = pgae_results
    
    # 2. Adaptive PGAE方法 - 使用改进的自适应设计更新策略（调参优化版本）
    logger.info("运行自适应PGAE估计器...")
    adaptive_pgae_estimator = AdaptivePGAEEstimator(
        X=X, F=F, Y=Y, gamma=gamma, batch_size=250,
        design_update_freq=1,  # 每批更新一次设计（最优参数）
        warmup_batches=2       # 前2批使用固定设计（最优参数）
    )
    
    # 使用并发执行自适应PGAE实验
    adaptive_results = adaptive_pgae_estimator.run_experiments(
        df_clean,
        n_experiments=n_experiments,
        n_true_labels=n_true_labels,
        alpha=alpha,
        seed=seed,
        use_concurrent=True,
        max_workers=10  # 限制worker数量以节省内存
    )
    
    results['Adaptive_PGAE'] = adaptive_results
    
    # 3. Active Statistical Inference方法 - 使用并发优化的ActiveInferenceEstimator  
    logger.info("运行主动统计推断估计器...")
    active_inference_estimator = ActiveInferenceEstimator(X=X, F=F, Y=Y, gamma=gamma)
    
    # 使用并发执行主动统计推断实验
    active_results = active_inference_estimator.run_experiments(
        df_clean,
        n_experiments=n_experiments,
        n_true_labels=n_true_labels,
        alpha=alpha,
        seed=seed,
        use_concurrent=True,
        max_workers=10  # 限制worker数量以节省内存
    )
    
    results['Active_Inference'] = active_results
    
    # 4. Naive方法 - 基线对比，使用最简单的统计方法
    logger.info("运行Naive估计器...")
    naive_estimator = NaiveEstimator(X=X, F=F, Y=Y, gamma=gamma)
    
    # 使用并发执行Naive实验
    naive_results = naive_estimator.run_experiments(
        df_clean,
        n_experiments=n_experiments,
        n_true_labels=n_true_labels,
        alpha=alpha,
        seed=seed,
        use_concurrent=True,
        max_workers=10  # 限制worker数量以节省内存
    )
    
    results['Naive'] = naive_results
    
    # 汇总结果
    total_time = (pgae_results.get('execution_time', 0) + 
                 adaptive_results.get('execution_time', 0) + 
                 active_results.get('execution_time', 0))
    logger.info(f"所有实验完成，总耗时: {total_time:.2f}秒")
    
    results['summary'] = {
        'true_value': true_value,
        'n_experiments': n_experiments,
        'n_true_labels': n_true_labels,
        'alpha': alpha,
        'gamma': gamma,
        'total_execution_time': total_time,
        'parameters': {
            'X': X, 'F': F, 'Y': Y,
            'batch_size': 100,
            'seed': seed
        }
    }
    
    return results

def create_comparison_plots(results: Dict, output_dir: str = "./"):
    """
    创建对比可视化图表 - 参照notebook结果
    
    Args:
        results: 对比结果字典
        output_dir: 输出目录
    """
    logger.info("生成对比可视化图表...")
    
    # 提取方法和指标
    methods = ['PGAE', 'Adaptive_PGAE', 'Active_Inference', 'Naive']
    method_labels = ['PGAE', 'Adaptive PGAE', 'Active Inference', 'Naive']
    
    metrics = {
        'MSE': [results[method]['mse'] for method in methods],
        'Coverage Rate': [results[method]['coverage_rate'] for method in methods],
        'Avg CI Length': [results[method]['avg_ci_length'] for method in methods],
        'Execution Time (s)': [results[method]['execution_time'] for method in methods]
    }
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # MSE比较
    ax1 = axes[0, 0]
    bars1 = ax1.bar(method_labels, metrics['MSE'], color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.6f}', ha='center', va='bottom', fontsize=10)
    
    # 覆盖率比较
    ax2 = axes[0, 1]
    bars2 = ax2.bar(method_labels, metrics['Coverage Rate'], color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title('Coverage Rate', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Coverage Rate', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Target (90%)')
    ax2.legend()
    
    # 添加数值标签
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 置信区间长度比较
    ax3 = axes[1, 0]
    bars3 = ax3.bar(method_labels, metrics['Avg CI Length'], color=colors, alpha=0.7, edgecolor='black')
    ax3.set_title('Average CI Length', fontsize=14, fontweight='bold')
    ax3.set_ylabel('CI Length', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 执行时间比较
    ax4 = axes[1, 1]
    bars4 = ax4.bar(method_labels, metrics['Execution Time (s)'], color=colors, alpha=0.7, edgecolor='black')
    ax4.set_title('Execution Time', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Time (seconds)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/estimator_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建性能总结表
    summary_df = pd.DataFrame({
        'Method': method_labels,
        'MSE': [f"{mse:.6f}" for mse in metrics['MSE']],
        'Coverage Rate': [f"{cr:.4f}" for cr in metrics['Coverage Rate']],
        'Avg CI Length': [f"{cl:.4f}" for cl in metrics['Avg CI Length']],
        'Time (s)': [f"{t:.1f}" for t in metrics['Execution Time (s)']]
    })
    
    print("\n" + "="*80)
    print("STATISTICAL ESTIMATOR COMPARISON RESULTS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    logger.info(f"可视化图表已保存: {output_dir}/estimator_comparison.png")

class EstimatorComparison:
    """估计器对比分析类 - 保持向后兼容"""
    
    def __init__(self, data_file: str, X: List[str] = ['EDUCATION'], 
                 F: str = 'ECON1MOD_LLM', Y: str = 'ECON1MOD', gamma: float = 0.5):
        """
        初始化对比分析
        
        Args:
            data_file: 数据文件路径
            X: 协变量列名列表
            F: 预测列名
            Y: 真实标签列名
            gamma: 参数gamma
        """
        self.data_file = data_file
        self.X = X
        self.F = F
        self.Y = Y
        self.gamma = gamma
        
        # 加载数据
        logger.info(f"加载数据: {data_file}")
        self.df = pd.read_csv(data_file)
        
        # 初始化估计器
        self.estimators = {
            'PGAE': PGAEEstimator(X=self.X, F=self.F, Y=self.Y, gamma=self.gamma),
            'Adaptive_PGAE': AdaptivePGAEEstimator(X=self.X, F=self.F, Y=self.Y, gamma=self.gamma),
            'Active_Inference': ActiveInferenceEstimator(X=self.X, F=self.F, Y=self.Y, gamma=self.gamma),
            'Naive': NaiveEstimator(X=self.X, F=self.F, Y=self.Y, gamma=self.gamma)
        }
        
        logger.info("估计器对比分析初始化完成")
    
    def run_comparison(self, n_experiments: int = 100, n_true_labels: int = 500, 
                      alpha: float = 0.9, seed: int = 42) -> Dict:
        """
        运行所有估计器的对比实验
        
        Args:
            n_experiments: 实验次数
            n_true_labels: 每次实验的真实标签数量
            alpha: 置信水平
            seed: 随机种子
            
        Returns:
            所有方法的实验结果
        """
        logger.info(f"开始对比实验: {n_experiments}次实验, 每次{n_true_labels}个标签")
        
        results = {}
        
        for method_name, estimator in self.estimators.items():
            logger.info(f"\n运行 {method_name} 估计器...")
            start_time = time.time()
            
            try:
                method_results = estimator.run_experiments(
                    self.df, n_experiments, n_true_labels, alpha, seed
                )
                results[method_name] = method_results
                
                end_time = time.time()
                logger.info(f"{method_name} 完成，耗时: {end_time - start_time:.2f}秒")
                
            except Exception as e:
                logger.error(f"{method_name} 运行失败: {e}")
                continue
        
        # 添加对比总结
        results['comparison_summary'] = self._generate_comparison_summary(results)
        
        return results
    
    def _generate_comparison_summary(self, results: Dict) -> Dict:
        """生成对比总结"""
        summary = {
            'methods_compared': list(results.keys()),
            'best_mse': {},
            'best_coverage': {},
            'best_ci_length': {},
            'execution_times': {}
        }
        
        # 找出各项指标最佳的方法
        mse_values = {method: res['mse'] for method, res in results.items() if 'mse' in res}
        if mse_values:
            best_mse_method = min(mse_values.keys(), key=lambda x: mse_values[x])
            summary['best_mse'] = {'method': best_mse_method, 'value': mse_values[best_mse_method]}
        
        coverage_values = {method: res['coverage_rate'] for method, res in results.items() if 'coverage_rate' in res}
        if coverage_values:
            best_coverage_method = max(coverage_values.keys(), key=lambda x: coverage_values[x])
            summary['best_coverage'] = {'method': best_coverage_method, 'value': coverage_values[best_coverage_method]}
        
        ci_length_values = {method: res['avg_ci_length'] for method, res in results.items() if 'avg_ci_length' in res}
        if ci_length_values:
            best_ci_method = min(ci_length_values.keys(), key=lambda x: ci_length_values[x])
            summary['best_ci_length'] = {'method': best_ci_method, 'value': ci_length_values[best_ci_method]}
        
        # 执行时间
        summary['execution_times'] = {method: res.get('execution_time', 0) for method, res in results.items() if 'execution_time' in res}
        
        return summary
    
    def create_visualization(self, results: Dict, save_path: str = "estimator_comparison.png"):
        """
        创建可视化对比图表
        
        Args:
            results: 实验结果
            save_path: 保存路径
        """
        # 过滤出有效结果
        valid_results = {k: v for k, v in results.items() 
                        if k != 'comparison_summary' and 'mse' in v}
        
        if not valid_results:
            logger.warning("没有有效的结果用于可视化")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Statistical Estimators Comparison', fontsize=16, fontweight='bold')
        
        methods = list(valid_results.keys())
        
        # 1. MSE对比
        mse_values = [valid_results[method]['mse'] for method in methods]
        axes[0, 0].bar(methods, mse_values, color=sns.color_palette("husl", len(methods)))
        axes[0, 0].set_title('Mean Squared Error (MSE)')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Coverage Rate对比
        coverage_values = [valid_results[method]['coverage_rate'] for method in methods]
        axes[0, 1].bar(methods, coverage_values, color=sns.color_palette("husl", len(methods)))
        axes[0, 1].set_title('Coverage Rate')
        axes[0, 1].set_ylabel('Coverage Rate')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. CI Length对比
        ci_length_values = [valid_results[method]['avg_ci_length'] for method in methods]
        axes[1, 0].bar(methods, ci_length_values, color=sns.color_palette("husl", len(methods)))
        axes[1, 0].set_title('Average Confidence Interval Length')
        axes[1, 0].set_ylabel('CI Length')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 执行时间对比
        execution_times = [valid_results[method].get('execution_time', 0) for method in methods]
        axes[1, 1].bar(methods, execution_times, color=sns.color_palette("husl", len(methods)))
        axes[1, 1].set_title('Execution Time')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化结果已保存: {save_path}")
        
        plt.show()
    
    def create_detailed_comparison_plot(self, results: Dict, save_path: str = "detailed_comparison.png"):
        """
        创建详细的估计值分布对比图
        
        Args:
            results: 实验结果
            save_path: 保存路径
        """
        valid_results = {k: v for k, v in results.items() 
                        if k != 'comparison_summary' and 'tau_estimates' in v}
        
        if not valid_results:
            logger.warning("没有详细结果用于可视化")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 估计值分布的箱线图
        tau_data = []
        method_labels = []
        
        for method, result in valid_results.items():
            tau_data.append(result['tau_estimates'])
            method_labels.append(method)
        
        ax1.boxplot(tau_data, labels=method_labels)
        ax1.set_title('Distribution of Estimates')
        ax1.set_ylabel('Estimate Value')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加真实值线
        if valid_results:
            true_value = list(valid_results.values())[0]['true_value']
            ax1.axhline(y=true_value, color='red', linestyle='--', 
                       label=f'True Value: {true_value:.4f}')
            ax1.legend()
        
        # CI宽度分布
        ci_widths = []
        for method, result in valid_results.items():
            ci_width = result['h_ci'] - result['l_ci']
            ci_widths.append(ci_width)
        
        ax2.boxplot(ci_widths, labels=method_labels)
        ax2.set_title('Distribution of CI Widths')
        ax2.set_ylabel('CI Width')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"详细对比图已保存: {save_path}")
        
        plt.show()
    
    def save_comparison_results(self, results: Dict, filename: str = "") -> str:
        """
        保存对比结果
        
        Args:
            results: 对比结果
            filename: 文件名
            
        Returns:
            保存的文件名
        """
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'estimators_comparison_{timestamp}.json'
        
        # 准备保存数据
        save_data = {}
        for method, result in results.items():
            if method == 'comparison_summary':
                save_data[method] = result
            else:
                save_data[method] = {
                    'method': result.get('method', method),
                    'mse': result.get('mse', 0),
                    'bias': result.get('bias', 0),
                    'variance': result.get('variance', 0),
                    'avg_ci_length': result.get('avg_ci_length', 0),
                    'coverage_rate': result.get('coverage_rate', 0),
                    'true_value': result.get('true_value', 0),
                    'execution_time': result.get('execution_time', 0),
                    'parameters': result.get('parameters', {}),
                    'n_experiments': result.get('n_experiments', 0)
                }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"对比结果已保存: {filename}")
        return filename

def main():
    """主函数"""
    import sys
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description='统计估计器对比工具')
    parser.add_argument('data_file', help='数据文件路径')
    parser.add_argument('--dataset-choice', choices=['base', 'cot'], default=None,
                       help='快捷选择内置数据集: base 或 cot (若提供，将覆盖 data_file)')
    parser.add_argument('--target', '-t', default='ECON1MOD', 
                       choices=['ECON1MOD', 'UNITY', 'GPT1', 'MOREGUNIMPACT', 'GAMBLERESTR'],
                       help='预测目标任务 (默认: ECON1MOD)')
    parser.add_argument('--experiments', '-e', type=int, default=100,
                       help='实验次数 (默认: 100)')
    parser.add_argument('--labels', '-l', type=int, default=500,
                       help='每次实验的真实标签数量 (默认: 500)')
    parser.add_argument('--gamma', '-g', type=float, default=0.5,
                       help='PGAE gamma参数 (默认: 0.5)')
    parser.add_argument('--alpha', '-a', type=float, default=0.90,
                       help='置信水平 (默认: 0.90)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出文件名前缀')
    # 固定CI宽度 → 最小标签数 模式参数（可选）
    parser.add_argument('--ci-width', type=float, default=None, help='目标CI宽度（例如 0.05）。提供该参数则运行“最小标签成本”模式')
    parser.add_argument('--ci-tolerance', type=float, default=0.005, help='CI宽度容忍度，例如0.005表示允许 avg_CI <= 目标+0.005 视为达标')
    parser.add_argument('--methods', nargs='*', default=['PGAE','Adaptive_PGAE'], help='参与方法，默认 [PGAE, Adaptive_PGAE]')
    parser.add_argument('--gamma-grid', nargs='*', type=float, default=[0.5], help='扫描的gamma取值，默认[0.5]')
    # CI模式下默认沿用MSE模式调优过的Adaptive参数：batch_size=250, design_update_freq=1, warmup_batches=2
    parser.add_argument('--batch-size', type=int, default=250, help='Adaptive: 批大小（默认: 250，来自MSE调优）')
    parser.add_argument('--design-update-freq', type=int, default=1, help='Adaptive: 设计更新频率（默认: 1，来自MSE调优）')
    parser.add_argument('--warmup-batches', type=int, default=2, help='Adaptive: 预热批次数（默认: 2，来自MSE调优）')
    parser.add_argument('--min-labels', type=int, default=100, help='搜索的最小标签数')
    parser.add_argument('--max-labels', type=int, default=2000, help='搜索的最大标签数')
    parser.add_argument('--label-step', type=int, default=100, help='n_labels 搜索步长（默认100）')
    parser.add_argument('--concurrent', action='store_true', help='在CI成本模式下启用并发执行（默认启用）')
    parser.add_argument('--max-workers', type=int, default=10, help='并发worker数量上限（默认10）')
    parser.add_argument('--results-csv', type=str, default='compare_runs_log.csv', help='比较结果汇总CSV（同设置将覆盖）')
    
    # 总是使用argparse解析参数
    args = parser.parse_args()
    # 处理数据集快捷选择
    data_file = args.data_file
    if args.dataset_choice is not None:
        preset_map = {
            'base': 'archive/predictions/NPORS_2024_base_gpt41mini_lr06_step560_20250911_214943.csv',
            'cot': 'archive/predictions/NPORS_2024_cot_optimized_lr06_step560_20250911_232934.csv',
        }
        chosen = preset_map.get(args.dataset_choice)
        if chosen is not None:
            data_file = chosen
            logger.info(f"使用内置数据集选择: {args.dataset_choice} -> {data_file}")
        else:
            logger.warning(f"未知的数据集选项: {args.dataset_choice}，继续使用提供的 data_file")
    target = args.target
    n_experiments = args.experiments
    n_true_labels = args.labels
    gamma = args.gamma
    alpha = args.alpha
    seed = args.seed
    output = args.output
    
    logger.info(f"加载数据: {data_file}")
    df = pd.read_csv(data_file)
    
    # 如果提供了 --ci-width，则运行“固定CI→最小标签成本”模式
    if args.ci_width is not None:
        # 部署基于CI模式的最佳默认参数
        # PGAE: 来自调参结果（固定gamma=0.5时的最佳）
        DEFAULT_PGAE_EST_CI_PARAMS = {
            'n_estimators_mu': 100,
            'n_estimators_tau': 200,
            'K': 5,
            'max_depth': 10,
        }
        DEFAULT_PGAE_DESIGN_PARAMS = {
            'min_var_threshold': 0.0001,
            'prob_clip_min': 0.1,
            'prob_clip_max': 0.9,
        }

        def run_once_and_get_ci_length(estimator, df_in: pd.DataFrame, n_exp: int, n_labels: int,
                                       a: float, sd: Optional[int], use_cc: bool, max_w: Optional[int]):
            kwargs = dict(n_experiments=n_exp, n_true_labels=n_labels, alpha=a, seed=sd, use_concurrent=use_cc)
            try:
                res = estimator.run_experiments(df_in, **kwargs, max_workers=max_w)
            except TypeError:
                raise ValueError("No workers argument supported in this estimator's run_experiments method.")
                res = estimator.run_experiments(df_in, **kwargs)
            return float(res['avg_ci_length']), res

        def search_min_labels_for_ci(factory, df_in: pd.DataFrame, target_ci: float,
                                      a: float, sd: Optional[int], n_exp: int,
                                      labels_lo: int, labels_hi: int, step: int,
                                      use_cc: bool, max_w: Optional[int], tol: float):
            audit = []
            # 对齐到步长（向上取整）
            def align_up(n: int, s: int) -> int:
                return int(((n + s - 1) // s) * s)

            def align_range(lo: int, hi: int, s: int) -> (int, int):
                alo = align_up(lo, s)
                ahi = (hi // s) * s if hi >= s else hi
                if ahi < alo:
                    ahi = alo
                return alo, ahi

            labels_lo, labels_hi = align_range(labels_lo, labels_hi, step)
            low = labels_lo
            high = labels_lo
            best = None
            # grow
            while True:
                est = factory()
                ci_len, res = run_once_and_get_ci_length(est, df_in, n_exp, high, a, sd, use_cc, max_w)
                audit.append({'labels': int(high), 'avg_ci_length': float(ci_len)})
                if ci_len <= target_ci + tol:
                    best = (high, ci_len, res)
                    break
                if high >= labels_hi:
                    break
                low = high
                high = min(align_up(high * 2, step), labels_hi)
            if best is None:
                return {'achieved': False, 'required_labels': None, 'avg_ci_length': audit[-1]['avg_ci_length'] if audit else None, 'audit': audit}
            # binary search
            lo = min(align_up(low + 1, step), labels_hi)
            hi = best[0]
            req_labels, req_ci, req_res = hi, best[1], best[2]
            while lo <= hi:
                # 取中点并对齐到步长（向上对齐保证不低于中点）
                mid_raw = (lo + hi) // 2
                mid = align_up(mid_raw, step)
                if mid > hi:
                    mid = hi
                est = factory()
                ci_len, res = run_once_and_get_ci_length(est, df_in, n_exp, mid, a, sd, use_cc, max_w)
                audit.append({'labels': int(mid), 'avg_ci_length': float(ci_len)})
                if ci_len <= target_ci + tol:
                    req_labels, req_ci, req_res = mid, ci_len, res
                    # 向下缩小一格步长
                    hi = mid - step
                else:
                    # 向上移动一格步长
                    lo = mid + step
            return {'achieved': True, 'required_labels': int(req_labels), 'avg_ci_length': float(req_ci), 'results_snapshot': {
                'mse': float(req_res['mse']), 'coverage_rate': float(req_res['coverage_rate']), 'variance': float(req_res['variance']), 'parameters': req_res.get('parameters', {})
            }, 'audit': sorted(audit, key=lambda x: x['labels']), 'tolerance': tol}

        # Build config
        TARGET_CONFIGS = {
            'ECON1MOD': {'X': ['EDUCATION'], 'F': 'ECON1MOD_LLM', 'Y': 'ECON1MOD', 'description': 'Economic conditions rating (1-4)'},
            'UNITY': {'X': ['EDUCATION'], 'F': 'UNITY_LLM', 'Y': 'UNITY', 'description': 'US unity perception (1-2)'},
            'GPT1': {'X': ['EDUCATION'], 'F': 'GPT1_LLM', 'Y': 'GPT1', 'description': 'ChatGPT familiarity (1-3)'},
            'MOREGUNIMPACT': {'X': ['EDUCATION'], 'F': 'MOREGUNIMPACT_LLM', 'Y': 'MOREGUNIMPACT', 'description': 'Gun control impact (1-3)'},
            'GAMBLERESTR': {'X': ['EDUCATION'], 'F': 'GAMBLERESTR_LLM', 'Y': 'GAMBLERESTR', 'description': 'Gambling restriction opinion (1-3)'}
        }
        cfg = TARGET_CONFIGS[target]
        X, F, Y = cfg['X'], cfg['F'], cfg['Y']

        logger.info(f"运行CI成本模式: 目标CI宽度={args.ci_width} (±{args.ci_tolerance}), alpha={alpha}")
        methods = args.methods
        gamma_grid = args.gamma_grid
        # 默认启用并发；即使未提供 --concurrent 也启用
        use_cc = True
        max_w = args.max_workers or 10
        results = {}
        for method in methods:
            logger.info(f"方法: {method}")
            best = None
            for g in gamma_grid:
                if method == 'PGAE':
                    def factory(gg=g):
                        return PGAEEstimator(
                            X=X, F=F, Y=Y, gamma=gg,
                            design_params=DEFAULT_PGAE_DESIGN_PARAMS,
                            est_ci_params=DEFAULT_PGAE_EST_CI_PARAMS
                        )
                elif method == 'Adaptive_PGAE':
                    def factory(gg=g, bs=args.batch_size, uf=args.design_update_freq, wu=args.warmup_batches):
                        return AdaptivePGAEEstimator(X=X, F=F, Y=Y, gamma=gg, batch_size=bs,
                                                     design_update_freq=uf, warmup_batches=wu)
                elif method == 'Active_Inference':
                    def factory(gg=g):
                        return ActiveInferenceEstimator(X=X, F=F, Y=Y, gamma=gg)
                elif method == 'Naive':
                    def factory(gg=g):
                        return NaiveEstimator(X=X, F=F, Y=Y, gamma=gg)
                else:
                    logger.warning(f"  不支持的方法: {method}, 跳过")
                    continue
                out = search_min_labels_for_ci(lambda: factory(), df, args.ci_width, alpha, seed, n_experiments,
                                               args.min_labels, args.max_labels, args.label_step,
                                               use_cc, max_w, args.ci_tolerance)
                out['gamma'] = g
                if best is None or (out.get('achieved') and out.get('required_labels') is not None and out['required_labels'] < best['required_labels']):
                    best = out
            results[method] = best if best is not None else {'achieved': False}

        # 打印总结
        print("\n" + "=" * 90)
        print("CI WIDTH COST COMPARISON (lower labels = lower cost)")
        print("=" * 90)
        for method, res in results.items():
            if not res or not res.get('achieved'):
                print(f"{method:<18} -> 未达到目标CI宽度 (<= {args.ci_width}+{args.ci_tolerance})，在 labels <= {args.max_labels} 范围内")
            else:
                mse_text = None
                snap = res.get('results_snapshot', {})
                if 'mse' in snap:
                    mse_text = f"MSE={snap['mse']:.6f}"
                extra = f" | {mse_text}" if mse_text else ""
                tol_txt = f"±{res.get('tolerance', args.ci_tolerance)}"
                print(f"{method:<18} -> 需要标签: {res['required_labels']}, gamma={res.get('gamma')} | avg_CI={res['avg_ci_length']:.4f} ({tol_txt}){extra}")
        print("=" * 90)

        # 保存结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_name = (args.output or 'ci_cost_comparison') + f"_{target.lower()}_{timestamp}.json"
        with open(out_name, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"CI成本对比结果已保存: {out_name}")

        # 保存到CSV（按设置去重覆盖）
        try:
            import pandas as pd
            rows = []
            settings_key = (
                f"mode=ci-cost|target={target}|dataset={data_file}|ci={args.ci_width}|tol={args.ci_tolerance}|"
                f"alpha={alpha}|experiments={n_experiments}|gamma_grid={','.join(map(str, args.gamma_grid))}|"
                f"adaptive_defaults={args.batch_size}-{args.design_update_freq}-{args.warmup_batches}"
            )
            for method, res in results.items():
                rows.append({
                    'timestamp': timestamp,
                    'mode': 'ci-cost',
                    'target': target,
                    'dataset': data_file,
                    'alpha': alpha,
                    'experiments': n_experiments,
                    'ci_width': args.ci_width,
                    'ci_tolerance': args.ci_tolerance,
                    'method': method,
                    'achieved': bool(res.get('achieved', False)),
                    'required_labels': res.get('required_labels'),
                    'avg_ci_length': res.get('avg_ci_length'),
                    'gamma': res.get('gamma'),
                    'mse_snapshot': (res.get('results_snapshot') or {}).get('mse'),
                    'coverage_snapshot': (res.get('results_snapshot') or {}).get('coverage_rate'),
                    'variance_snapshot': (res.get('results_snapshot') or {}).get('variance'),
                    'adaptive_batch_size': res.get('batch_size'),
                    'adaptive_update_freq': res.get('design_update_freq'),
                    'adaptive_warmup_batches': res.get('warmup_batches'),
                    'settings_key': settings_key,
                })
            df_new = pd.DataFrame(rows)
            try:
                df_old = pd.read_csv(args.results_csv)
                keep_mask = ~df_old.apply(lambda r: (str(r.get('settings_key')) == settings_key) and (r.get('method') in df_new['method'].values), axis=1)
                df_merged = pd.concat([df_old[keep_mask], df_new], ignore_index=True)
            except FileNotFoundError:
                df_merged = df_new
            df_merged.to_csv(args.results_csv, index=False)
            logger.info(f"CI成本对比结果已写入CSV: {args.results_csv}")
        except Exception as e:
            logger.warning(f"写入CSV失败: {e}")
        return

    # 否则运行notebook风格对比
    results = run_notebook_style_comparison(
        df,
        target=target,
        n_experiments=n_experiments,
        n_true_labels=n_true_labels,
        gamma=gamma,
        alpha=alpha,
        seed=seed
    )
    
    # 生成可视化
    create_comparison_plots(results)
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if output:
        output_file = f'{output}_{target.lower()}_{timestamp}.json'
    else:
        output_file = f'estimator_comparison_{target.lower()}_{timestamp}.json'
    
    # 准备保存的数据（移除numpy数组）
    save_data = {}
    for method in ['PGAE', 'Adaptive_PGAE', 'Active_Inference']:
        if method in results:
            save_data[method] = {
                'mse': float(results[method]['mse']),
                'bias': float(results[method]['bias']),
                'variance': float(results[method]['variance']),
                'avg_ci_length': float(results[method]['avg_ci_length']),
                'coverage_rate': float(results[method]['coverage_rate']),
                'execution_time': float(results[method]['execution_time']),
                'n_experiments': int(results[method]['n_experiments'])
            }
    
    save_data['summary'] = results['summary']
    
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n✅ {target} 统计估计器对比完成!")
    print(f"预测目标: {target}")
    print(f"实验设置: {n_experiments} experiments, {n_true_labels} labels per experiment")
    print(f"结果文件: {output_file}")
    print(f"可视化文件: estimator_comparison.png")

    # 保存到CSV（按设置去重覆盖）
    try:
        import pandas as pd
        rows = []
        settings_key = (
            f"mode=mse-compare|target={target}|dataset={data_file}|alpha={alpha}|experiments={n_experiments}|"
            f"labels={n_true_labels}|gamma={gamma}"
        )
        for method in ['PGAE', 'Adaptive_PGAE', 'Active_Inference', 'Naive']:
            if method in results and isinstance(results[method], dict) and 'mse' in results[method]:
                r = results[method]
                rows.append({
                    'timestamp': timestamp,
                    'mode': 'mse-compare',
                    'target': target,
                    'dataset': data_file,
                    'alpha': alpha,
                    'experiments': n_experiments,
                    'n_true_labels': n_true_labels,
                    'method': method,
                    'gamma': gamma,
                    'mse': r.get('mse'),
                    'avg_ci_length': r.get('avg_ci_length'),
                    'coverage_rate': r.get('coverage_rate'),
                    'execution_time': r.get('execution_time'),
                    'settings_key': settings_key,
                })
        if rows:
            df_new = pd.DataFrame(rows)
            try:
                df_old = pd.read_csv(args.results_csv)
                keep_mask = ~df_old.apply(lambda r: (str(r.get('settings_key')) == settings_key) and (r.get('method') in df_new['method'].values), axis=1)
                df_merged = pd.concat([df_old[keep_mask], df_new], ignore_index=True)
            except FileNotFoundError:
                df_merged = df_new
            df_merged.to_csv(args.results_csv, index=False)
            logger.info(f"MSE比较结果已写入CSV: {args.results_csv}")
    except Exception as e:
        logger.warning(f"写入CSV失败: {e}")
    
    # 显示最佳性能方法
    method_results = {k: v for k, v in results.items() if k not in ['summary'] and isinstance(v, dict) and 'mse' in v}
    if method_results:
        # 逐方法打印MSE/覆盖率/CI
        print("\n各方法指标摘要:")
        for m, r in method_results.items():
            try:
                print(f"  {m:<18} MSE={r['mse']:.6f} | Coverage={r['coverage_rate']:.4f} | CI={r['avg_ci_length']:.4f}")
            except Exception:
                pass
        best_method = min(method_results.keys(), key=lambda x: method_results[x]['mse'])
        best_mse = method_results[best_method]['mse']
        best_coverage = method_results[best_method]['coverage_rate']
        print(f"\n🏆 最佳方法: {best_method}")
        print(f"   MSE: {best_mse:.6f}")
        print(f"   覆盖率: {best_coverage:.4f}")

if __name__ == "__main__":
    main()
