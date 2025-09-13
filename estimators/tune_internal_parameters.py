#!/usr/bin/env python3
"""
PGAE内部参数调优工具
Internal parameter tuning tool for PGAE algorithm
"""

import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
import logging
from itertools import product

# 导入基础工具
from utils import summary_results

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_robust_PGAE_est_ci(n_estimators_mu=50, n_estimators_tau=200, 
                              max_depth=None, min_samples_split=5,
                              min_samples_leaf=2, K=5):
    """
    创建可调参的PGAE置信区间估计函数
    """
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats import norm
    
    def robust_PGAE_est_ci(PGAE_df: pd.DataFrame, X: List[str], F: str, Y: str, 
                          alpha: float = 0.95) -> Tuple[float, float, float]:
        """
        改进的PGAE置信区间计算，参数可调
        """
        PGAE_labeled = PGAE_df[PGAE_df['true_label'] == 1]
        unlabeled_indices = PGAE_df[PGAE_df['true_label'] == 0].index.to_list()
        n = len(PGAE_labeled)
        N = len(unlabeled_indices)
        labeled_indices = PGAE_labeled.index.to_list()

        if n < K:  # 如果样本不足，减少K值
            effective_K = max(2, n // 2)
        else:
            effective_K = K

        tau_PGAE = 0
        var_PGAE = 0
        
        for i in range(effective_K):
            fold1 = labeled_indices[i*n//effective_K: (i+1)*n//effective_K]
            fold2 = labeled_indices[: i*n//effective_K] + labeled_indices[(i+1)*n//effective_K:]

            if N > 0:
                unlabeled_fold1 = unlabeled_indices[i*N//effective_K: (i+1)*N//effective_K]
            else:
                unlabeled_fold1 = []

            # 使用调优后的RandomForest参数
            model_mu = RandomForestRegressor(
                n_estimators=n_estimators_mu,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            model_mu.fit(PGAE_df.loc[fold2, X], PGAE_df.loc[fold2, Y])
            PGAE_df.loc[fold1+unlabeled_fold1, 'mu'] = model_mu.predict(PGAE_df.loc[fold1+unlabeled_fold1, X])

            model_tau = RandomForestRegressor(
                n_estimators=n_estimators_tau,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            model_tau.fit(PGAE_df.loc[fold2, X+[F]], PGAE_df.loc[fold2, Y])
            PGAE_df.loc[fold1+unlabeled_fold1, 'tau'] = model_tau.predict(PGAE_df.loc[fold1+unlabeled_fold1, X+[F]])

        PGAE_df['psi'] = PGAE_df['true_pmf'] / PGAE_df['sample_pmf'] / PGAE_df['exp_prob'] * (
            PGAE_df['true_label'] * (PGAE_df[Y] - PGAE_df['tau']) - 
            PGAE_df['exp_prob'] * (PGAE_df['mu'] - PGAE_df['tau'])
        )

        # Aggregate the mean of true_pmf by group
        df_summary = PGAE_df.groupby(X).agg({'true_pmf': 'mean', 'mu': 'mean'}).reset_index()
        tau_PGAE = np.sum(df_summary['mu'].values * df_summary['true_pmf'].values) / np.sum(df_summary['true_pmf'].values)
        tau_PGAE += PGAE_df['psi'].mean()

        var_PGAE = PGAE_df['psi'].var()
        coef = norm.ppf(1 - (1 - alpha) / 2)
        tau_var = var_PGAE / len(PGAE_df)

        return tau_PGAE, tau_PGAE - np.sqrt(tau_var) * coef, tau_PGAE + np.sqrt(tau_var) * coef
    
    return robust_PGAE_est_ci

def create_robust_get_PGAE_design(min_var_threshold=1e-6, prob_clip_min=0.01, prob_clip_max=0.99):
    """
    创建更鲁棒的PGAE设计函数
    """
    def robust_get_PGAE_design(df: pd.DataFrame, X: List[str], F: str, Y: str, gamma: float) -> pd.DataFrame:
        """
        改进的PGAE设计参数计算，增加数值稳定性
        """
        # 按照notebook cell 9ab345bf的实现，但增加稳定性
        summary = df.groupby(X + [F])[Y].agg(
            cond_mean='mean',
            cond_var='var'
        ).reset_index()

        df = df.merge(summary, on=X + [F], how='left')

        # Group by X and compute required statistics with robustness
        group_stats = df.groupby(X).agg(
            var_of_cond_mean=('cond_mean', 'var'),
            mean_of_cond_var=('cond_var', 'mean'),
            cnt=('cond_mean', 'count'),
            true_pmf=('true_pmf', 'mean'),
        ).reset_index()

        # 添加数值稳定性 - 避免极小的方差
        group_stats['var_of_cond_mean'] = np.maximum(group_stats['var_of_cond_mean'].fillna(min_var_threshold), min_var_threshold)
        group_stats['mean_of_cond_var'] = np.maximum(group_stats['mean_of_cond_var'].fillna(min_var_threshold), min_var_threshold)

        group_stats['sample_pmf'] = group_stats['true_pmf'] * np.sqrt(group_stats['var_of_cond_mean'])
        total_sample_pmf = group_stats['sample_pmf'].sum()
        if total_sample_pmf > 0:
            group_stats['sample_pmf'] = group_stats['sample_pmf'] / total_sample_pmf
        else:
            group_stats['sample_pmf'] = group_stats['true_pmf']

        group_stats['exp_prob'] = np.sqrt(group_stats['mean_of_cond_var'] / group_stats['var_of_cond_mean'])
        exp_prob_weighted = np.sum(group_stats['exp_prob'] * group_stats['sample_pmf'])
        if exp_prob_weighted > 0:
            group_stats['exp_prob'] /= exp_prob_weighted / gamma
        
        # 更鲁棒的clipping
        group_stats['exp_prob'] = np.clip(group_stats['exp_prob'], prob_clip_min, prob_clip_max)
        
        max_var = group_stats['var_of_cond_mean'].max()
        if max_var > min_var_threshold:
            group_stats['accept_prob'] = np.sqrt(group_stats['var_of_cond_mean']) / np.sqrt(max_var)
        else:
            group_stats['accept_prob'] = 1.0
            
        group_stats = group_stats.drop(columns=['true_pmf', 'cnt'])

        return group_stats
    
    return robust_get_PGAE_design

def test_parameter_combinations(df: pd.DataFrame, gamma_values: List[float],
                               parameter_configs: List[Dict]) -> Dict:
    """
    测试不同参数组合在所有gamma值下的表现
    """
    from utils import rejection_sample, overwrite_merge
    
    logger.info(f"测试 {len(parameter_configs)} 个参数配置在 {len(gamma_values)} 个gamma值下")
    
    results = {}
    
    for config_idx, config in enumerate(tqdm(parameter_configs, desc="参数配置")):
        logger.info(f"\n测试配置 {config_idx + 1}: {config}")
        
        # 创建该配置的函数
        robust_est_ci = create_robust_PGAE_est_ci(**config['est_ci_params'])
        robust_design = create_robust_get_PGAE_design(**config['design_params'])
        
        config_results = {}
        
        for gamma in gamma_values:
            logger.info(f"  测试 gamma = {gamma}")
            
            # 运行单个实验来测试稳定性
            try:
                gamma_results = run_single_gamma_test(df, gamma, robust_est_ci, robust_design, 
                                                    n_experiments=30, n_true_labels=500)
                config_results[f'gamma_{gamma}'] = gamma_results
                
                logger.info(f"    MSE: {gamma_results['mse']:.6f}, "
                           f"覆盖率: {gamma_results['coverage_rate']:.3f}")
                           
            except Exception as e:
                logger.error(f"    配置 {config_idx} 在 gamma={gamma} 下失败: {e}")
                config_results[f'gamma_{gamma}'] = {
                    'mse': float('inf'),
                    'coverage_rate': 0.0,
                    'error': str(e)
                }
        
        results[f'config_{config_idx}'] = {
            'config': config,
            'results': config_results
        }
    
    return results

def run_single_gamma_test(df: pd.DataFrame, gamma: float, est_ci_func, design_func,
                         n_experiments: int = 30, n_true_labels: int = 500) -> Dict:
    """
    运行单个gamma值的测试
    """
    from utils import rejection_sample, overwrite_merge
    
    X = ['EDUCATION']
    F = 'ECON1MOD_LLM'
    Y = 'ECON1MOD'
    alpha = 0.90
    
    # 数据预处理
    df_clean = df[X + [F] + [Y]].copy()
    df_clean = df_clean[df_clean[X + [F] + [Y]].lt(10).all(axis=1)]
    
    # 计算true_pmf
    group_stats = df_clean.groupby(X).agg(cnt=(F, 'count')).reset_index()
    group_stats['true_pmf'] = group_stats['cnt'] / group_stats['cnt'].sum()
    df_clean = df_clean.merge(group_stats, on=X, how='left')
    
    true_value = df_clean[Y].mean()
    
    # 获取设计
    design = design_func(df_clean, X, F, Y, gamma)
    df_pgae = df_clean.merge(design, on=X, how='left')
    
    # 运行实验
    tau_results = np.zeros(n_experiments)
    l_ci_results = np.zeros(n_experiments)
    h_ci_results = np.zeros(n_experiments)
    
    for i in range(n_experiments):
        df_pgae['true_label'] = 0
        PGAE_df = []
        cnt_true = 0
        
        while cnt_true < n_true_labels:
            sampled_df = rejection_sample(df_pgae, df_pgae.columns.tolist(), 'accept_prob', n_samples=n_true_labels)
            u = np.random.uniform(0, 1, size=len(sampled_df))
            sampled_df['true_label'] = (u < sampled_df['exp_prob']).astype(int)
            cnt_true += sampled_df['true_label'].sum()
            PGAE_df.append(sampled_df)
            
        PGAE_df = pd.concat(PGAE_df, ignore_index=True)
        PGAE_df['cum_sum'] = PGAE_df['true_label'].cumsum()
        cutoff_index = PGAE_df[PGAE_df['cum_sum'] == n_true_labels].index[0]
        PGAE_df = PGAE_df.iloc[:cutoff_index + 1]
        
        tau, l_ci, h_ci = est_ci_func(PGAE_df, X, F, Y, alpha=alpha)
        tau_results[i], l_ci_results[i], h_ci_results[i] = tau, l_ci, h_ci
    
    # 计算指标
    mse = np.mean((tau_results - true_value) ** 2)
    coverage_rate = np.mean((l_ci_results <= true_value) & (true_value <= h_ci_results))
    avg_ci_length = np.mean(h_ci_results - l_ci_results)
    
    return {
        'mse': mse,
        'coverage_rate': coverage_rate,
        'avg_ci_length': avg_ci_length,
        'true_value': true_value
    }

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python tune_internal_parameters.py <data_file>")
        return
    
    data_file = sys.argv[1]
    logger.info(f"加载数据: {data_file}")
    df = pd.read_csv(data_file)
    
    # 固定 gamma 为 0.5（如需扫描可手动修改此列表或加CLI参数）
    gamma_values = [0.5]
    
    # 定义参数配置组合
    parameter_configs = [
        # 基础配置
        {
            'est_ci_params': {'n_estimators_mu': 20, 'n_estimators_tau': 100, 'K': 2},
            'design_params': {'min_var_threshold': 1e-6, 'prob_clip_min': 0.01, 'prob_clip_max': 0.99}
        },
        # 增强稳定性配置1
        {
            'est_ci_params': {'n_estimators_mu': 50, 'n_estimators_tau': 150, 'K': 3},
            'design_params': {'min_var_threshold': 1e-4, 'prob_clip_min': 0.05, 'prob_clip_max': 0.95}
        },
        # 增强稳定性配置2
        {
            'est_ci_params': {'n_estimators_mu': 100, 'n_estimators_tau': 200, 'K': 5, 'max_depth': 10},
            'design_params': {'min_var_threshold': 1e-4, 'prob_clip_min': 0.1, 'prob_clip_max': 0.9}
        },
        # 保守配置
        {
            'est_ci_params': {'n_estimators_mu': 80, 'n_estimators_tau': 150, 'K': 4, 'max_depth': 8, 'min_samples_split': 10},
            'design_params': {'min_var_threshold': 1e-3, 'prob_clip_min': 0.1, 'prob_clip_max': 0.8}
        }
    ]
    
    # 运行测试
    results = test_parameter_combinations(df, gamma_values, parameter_configs)
    
    # 分析结果
    print("\n" + "="*100)
    print("INTERNAL PARAMETER TUNING RESULTS")
    print("="*100)
    
    for config_name, config_data in results.items():
        print(f"\n{config_name.upper()}: {config_data['config']}")
        print("-" * 80)
        
        gamma_results = config_data['results']
        avg_mse = np.mean([r.get('mse', float('inf')) for r in gamma_results.values() if 'error' not in r])
        avg_coverage = np.mean([r.get('coverage_rate', 0) for r in gamma_results.values() if 'error' not in r])
        
        print(f"平均MSE: {avg_mse:.6f}, 平均覆盖率: {avg_coverage:.3f}")
        
        for gamma_key, gamma_result in gamma_results.items():
            gamma_val = gamma_key.split('_')[1]
            if 'error' in gamma_result:
                print(f"  gamma {gamma_val}: 失败 - {gamma_result['error']}")
            else:
                print(f"  gamma {gamma_val}: MSE={gamma_result['mse']:.6f}, 覆盖率={gamma_result['coverage_rate']:.3f}")
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(f'internal_parameter_tuning_results_{timestamp}.json', 'w') as f:
        # 转换numpy类型为python原生类型
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
        
        json.dump(convert_numpy(results), f, indent=2)
    
    print(f"\n✅ 内部参数调优完成!")
    print(f"结果文件: internal_parameter_tuning_results_{timestamp}.json")

if __name__ == "__main__":
    main()
