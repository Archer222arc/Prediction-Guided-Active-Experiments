#!/usr/bin/env python3
"""
Utility functions for statistical estimators
统计估计器的工具函数
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Optional, List

def overwrite_merge(df, merge_df, on, how='left'):
    """合并数据框，如果列名冲突则用新值覆盖旧值"""
    df_merged = df.merge(merge_df, on=on, how=how, suffixes=('', '_new'))
    for col in merge_df.columns:
        if col in df_merged and col + '_new' in df_merged:
            df_merged[col] = df_merged[col + '_new']
            df_merged.drop(columns=[col + '_new'], inplace=True)
    return df_merged

def rejection_sample(df: pd.DataFrame, columns: List[str], prob_col: str, n_samples: int) -> pd.DataFrame:
    """
    使用拒绝采样方法进行采样 - 原始notebook实现
    
    Args:
        df: 数据框
        columns: 要保留的列
        prob_col: 接受概率列名
        n_samples: 目标样本数
    
    Returns:
        采样后的数据框
    """
    accepted_rows = []
    while len(accepted_rows) < n_samples:
        sample = df.sample(n=n_samples, replace=True).reset_index(drop=True)
        u = np.random.uniform(0, 1, size=n_samples)
        accepted = sample[u < sample[prob_col].values]
        accepted_rows.append(accepted[columns])
    return pd.concat(accepted_rows, ignore_index=True).head(n_samples).reset_index(drop=True)

def PGAE_est_ci(PGAE_df: pd.DataFrame, X: List[str], F: str, Y: str, 
                alpha: float = 0.95, K: int = 3) -> Tuple[float, float, float]:
    """
    PGAE估计器的置信区间计算 - 原始notebook实现
    
    Parameters:
    - PGAE_df: DataFrame containing the data with columns for features, treatment, and outcome.
    - X: List of feature column names.
    - F: Name of the prediction column.
    - Y: Name of the outcome column.
    - alpha: Significance level for the confidence interval.
    - K: Number of cross-validation folds.
    
    Returns:
    - tau_PGAE: Estimated treatment effect.
    - lower_bound: Lower bound of the confidence interval.
    - upper_bound: Upper bound of the confidence interval.
    """
    PGAE_labeled = PGAE_df[PGAE_df['true_label'] == 1]
    unlabeled_indices = PGAE_df[PGAE_df['true_label'] == 0].index.to_list()
    n = len(PGAE_labeled)
    N = len(unlabeled_indices)
    labeled_indices = PGAE_labeled.index.to_list()

    tau_PGAE = 0
    var_PGAE = 0
    for i in range(K):
        fold1 = labeled_indices[i*n//K: (i+1)*n//K]
        fold2 = labeled_indices[: i*n//K] + labeled_indices[(i+1)*n//K:]

        unlabeled_fold1 = unlabeled_indices[i*N//K: (i+1)*N//K]

        model_mu = RandomForestRegressor(n_estimators=50, random_state=42)
        model_mu.fit(PGAE_df.loc[fold2, X], PGAE_df.loc[fold2, Y])
        PGAE_df.loc[fold1+unlabeled_fold1, 'mu'] = model_mu.predict(PGAE_df.loc[fold1+unlabeled_fold1, X])

        model_tau = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=5, random_state=42)
        model_tau.fit(PGAE_df.loc[fold2, X+[F]], PGAE_df.loc[fold2, Y])
        PGAE_df.loc[fold1+unlabeled_fold1, 'tau'] = model_tau.predict(PGAE_df.loc[fold1+unlabeled_fold1, X+[F]])

    PGAE_df['psi'] = PGAE_df['true_pmf'] / PGAE_df['sample_pmf'] / PGAE_df['exp_prob'] * (PGAE_df['true_label'] * (PGAE_df[Y] - PGAE_df['tau']) - PGAE_df['exp_prob'] * (PGAE_df['mu'] - PGAE_df['tau']))

    # Aggregate the mean of true_pmf by group
    df_summary = PGAE_df.groupby(X).agg({'true_pmf': 'mean', 'mu': 'mean'}).reset_index()
    tau_PGAE = np.sum(df_summary['mu'].values * df_summary['true_pmf'].values) / np.sum(df_summary['true_pmf'].values)
    tau_PGAE += PGAE_df['psi'].mean()

    var_PGAE = PGAE_df['psi'].var()
    coef = norm.ppf(1 - (1 - alpha) / 2)
    tau_var = float(var_PGAE) / len(PGAE_df)

    return tau_PGAE, tau_PGAE - np.sqrt(tau_var) * coef, tau_PGAE + np.sqrt(tau_var) * coef

def adaptive_PGAE_est_ci(PGAE_df: pd.DataFrame, X: List[str], F: str, Y: str, 
                        alpha: float = 0.95, batch_size: int = 100) -> Tuple[float, float, float]:
    """
    自适应PGAE估计器的置信区间计算 - 原始notebook实现
    """
    PGAE_df['mu'] = PGAE_df[Y].astype(float)    
    PGAE_df['tau'] = PGAE_df[Y].astype(float)
    PGAE_df['psi'] = 0.0
    
    # 初始化最终模型变量
    final_model_mu = None
    
    for i in range(batch_size, len(PGAE_df), batch_size):
        batch_start = i
        batch_end = min(i + batch_size, len(PGAE_df))
        
        if batch_end - batch_start < batch_size and batch_start + batch_size < len(PGAE_df):
            break
            
        # 使用整数索引而不是Index对象
        batch_slice = slice(batch_start, batch_end)
        training_slice = slice(0, i)
        
        training_data = PGAE_df.iloc[training_slice]
        batch_data = PGAE_df.iloc[batch_slice]
        
        model_mu = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=5, random_state=42)
        model_mu.fit(training_data[X], training_data[Y])
        PGAE_df.iloc[batch_slice, PGAE_df.columns.get_loc('mu')] = model_mu.predict(batch_data[X])

        model_tau = RandomForestRegressor(n_estimators=100, random_state=42)
        model_tau.fit(training_data[X+[F]], training_data[Y])
        PGAE_df.iloc[batch_slice, PGAE_df.columns.get_loc('tau')] = model_tau.predict(batch_data[X+[F]])
        
        # 保存最后训练的model_mu用于后续计算
        final_model_mu = model_mu

    PGAE_df['psi'] = PGAE_df['true_pmf'] / PGAE_df['sample_pmf'] / PGAE_df['exp_prob'] * (PGAE_df['true_label'] * (PGAE_df[Y] - PGAE_df['tau']) - PGAE_df['exp_prob'] * (PGAE_df['mu'] - PGAE_df['tau']))

    df_summary = PGAE_df.groupby(X).agg({'true_pmf': 'mean'}).reset_index()
    
    # 使用最后训练的模型进行预测，如果没有则重新训练
    if final_model_mu is None:
        final_model_mu = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=5, random_state=42)
        final_model_mu.fit(PGAE_df[X], PGAE_df[Y])
    
    predictions = final_model_mu.predict(df_summary[X])
    tau_PGAE = np.sum(predictions * df_summary['true_pmf'].values) / np.sum(df_summary['true_pmf'].values)
    tau_PGAE += PGAE_df['psi'].mean()
    var_PGAE = PGAE_df['psi'].var()

    coef = norm.ppf(1 - (1 - alpha) / 2)
    tau_var = float(var_PGAE) / len(PGAE_df)
    lower_bound = tau_PGAE - np.sqrt(tau_var) * coef
    upper_bound = tau_PGAE + np.sqrt(tau_var) * coef
    return tau_PGAE, lower_bound, upper_bound

def get_PGAE_design(df: pd.DataFrame, X: List[str], F: str, Y: str, gamma: float) -> pd.DataFrame:
    """
    获取PGAE设计参数 - 匹配notebook实现
    
    Args:
        df: 数据框
        X: 协变量列名列表
        F: 预测列名
        Y: 真实标签列名
        gamma: PGAE参数
    
    Returns:
        包含设计参数的数据框
    """
    # 按照notebook cell 9ab345bf的实现
    summary = df.groupby(X + [F])[Y].agg(
        cond_mean='mean',
        cond_var='var'
    ).reset_index()

    df = df.merge(summary, on=X + [F], how='left')

    # Group by X and compute required statistics
    group_stats = df.groupby(X).agg(
        var_of_cond_mean=('cond_mean', 'var'),  # sample variance
        mean_of_cond_var=('cond_var', 'mean'),
        cnt=('cond_mean', 'count'),
        true_pmf=('true_pmf', 'mean'),
    ).reset_index()

    # 添加数值稳定性改进 - CONFIG_1参数
    min_var_threshold = 0.0001
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
    
    # 更鲁棒的概率clipping - CONFIG_1参数
    group_stats['exp_prob'] = np.clip(group_stats['exp_prob'], 0.05, 0.95)
    group_stats['accept_prob'] = np.sqrt(group_stats['var_of_cond_mean']) / np.sqrt(group_stats['var_of_cond_mean']).max()
    group_stats = group_stats.drop(columns=['true_pmf', 'cnt'])

    return group_stats

def calculate_confidence_interval(estimates: np.ndarray, alpha: float = 0.9) -> Tuple[float, float, float]:
    """
    计算估计值的置信区间
    
    Args:
        estimates: 估计值数组
        alpha: 置信水平
    
    Returns:
        (均值, 置信区间下界, 置信区间上界)
    """
    mean_est = np.mean(estimates)
    std_est = np.std(estimates)
    n = len(estimates)
    
    # 使用t分布
    t_alpha = stats.t.ppf(1 - (1 - alpha) / 2, df=n-1)
    margin_of_error = t_alpha * std_est / np.sqrt(n)
    
    return mean_est, mean_est - margin_of_error, mean_est + margin_of_error

def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    验证数据框是否包含必需的列
    
    Args:
        df: 要验证的数据框
        required_columns: 必需的列名列表
    
    Returns:
        True如果数据有效，否则False
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return False
    
    # 检查数据类型
    for col in required_columns:
        if col != 'RACE_TEXT' and not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                print(f"Warning: Cannot convert column {col} to numeric")
    
    return True

def summary_results(tau_estimates: np.ndarray, l_ci: np.ndarray, h_ci: np.ndarray, 
                   true_value: float, method_name: str = "Method") -> dict:
    """
    总结统计估计结果
    
    Args:
        tau_estimates: 估计值数组
        l_ci: 置信区间下界数组
        h_ci: 置信区间上界数组
        true_value: 真实值
        method_name: 方法名称
    
    Returns:
        结果字典
    """
    mse = np.mean((tau_estimates - true_value) ** 2)
    bias = np.mean(tau_estimates) - true_value
    variance = np.var(tau_estimates)
    avg_ci_length = np.mean(h_ci - l_ci)
    coverage_rate = np.mean((l_ci <= true_value) & (true_value <= h_ci))
    
    results = {
        'method': method_name,
        'mse': mse,
        'bias': bias,
        'variance': variance,
        'avg_ci_length': avg_ci_length,
        'coverage_rate': coverage_rate,
        'n_experiments': len(tau_estimates)
    }
    
    print(f"\n{method_name} Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Bias: {bias:.6f}")
    print(f"  Variance: {variance:.6f}")
    print(f"  Avg CI length: {avg_ci_length:.6f}")
    print(f"  Coverage rate: {coverage_rate:.4f}")
    print(f"  Number of experiments: {len(tau_estimates)}")
    
    return results