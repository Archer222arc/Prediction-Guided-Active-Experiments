#!/usr/bin/env python3
"""
PGAE (Prediction-Guided Active Experiments) Estimator
预测引导主动实验估计器
"""

import numpy as np
import pandas as pd
import time
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import gc
from utils import (
    get_PGAE_design, rejection_sample, PGAE_est_ci, PGAE_est_ci_param,
    overwrite_merge, summary_results, validate_data
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PGAEEstimator:
    """PGAE估计器类"""
    
    def __init__(self, X: List[str], F: str, Y: str, gamma: float = 0.5,
                 design_params: Optional[Dict] = None,
                 est_ci_params: Optional[Dict] = None):
        """
        初始化PGAE估计器
        
        Args:
            X: 协变量列名列表
            F: 预测列名
            Y: 真实标签列名
            gamma: PGAE参数，控制实验概率的权重
        """
        self.X = X
        self.F = F
        self.Y = Y
        self.gamma = gamma
        self.design_params = design_params or {}
        self.est_ci_params = est_ci_params or {}

        logger.info(f"PGAE估计器初始化: X={X}, F={F}, Y={Y}, gamma={gamma}")
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备数据，添加必要的统计量
        
        Args:
            df: 输入数据框
            
        Returns:
            处理后的数据框
        """
        # 验证数据
        required_columns = self.X + [self.F, self.Y]
        if not validate_data(df, required_columns):
            raise ValueError("Data validation failed")
        
        # 过滤有效数据
        df_clean = df[required_columns].copy()
        df_clean = df_clean[df_clean[required_columns].lt(10).all(axis=1)]
        
        logger.info(f"数据清理后样本数: {len(df_clean)}")
        
        # 计算真实PMF
        group_stats = df_clean.groupby(self.X).agg(
            cnt=(self.F, 'count')
        ).reset_index()
        
        group_stats['true_pmf'] = group_stats['cnt'] / group_stats['cnt'].sum()
        df_clean = df_clean.merge(group_stats, on=self.X, how='left')
        
        # 获取PGAE设计
        # 允许覆盖设计稳健参数
        design = get_PGAE_design(
            df_clean, self.X, self.F, self.Y, self.gamma,
            min_var_threshold=self.design_params.get('min_var_threshold'),
            clip_low=self.design_params.get('prob_clip_min'),
            clip_high=self.design_params.get('prob_clip_max')
        )
        df_clean = df_clean.merge(design, on=self.X, how='left')
        
        return df_clean
    
    def run_single_experiment(self, df: pd.DataFrame, n_true_labels: int = 500, 
                             alpha: float = 0.9, seed: Optional[int] = None) -> Tuple[float, float, float]:
        """
        运行单次PGAE实验
        
        Args:
            df: 准备好的数据框
            n_true_labels: 目标真实标签数量
            alpha: 置信水平
            seed: 随机种子
            
        Returns:
            (估计值, 置信区间下界, 置信区间上界)
        """
        if seed is not None:
            np.random.seed(seed)
        
        df_work = df.copy()
        df_work['true_label'] = 0
        
        PGAE_df = []
        cnt_true = 0
        
        while cnt_true < n_true_labels:
            # 使用拒绝采样
            sampled_df = rejection_sample(df_work, df_work.columns, 'accept_prob', 
                                        n_samples=n_true_labels)
            
            # 根据实验概率决定是否标记
            u = np.random.uniform(0, 1, size=len(sampled_df))
            sampled_df['true_label'] = (u < sampled_df['exp_prob']).astype(int)
            
            cnt_true += sampled_df['true_label'].sum()
            PGAE_df.append(sampled_df)
        
        # 合并所有采样数据
        PGAE_df = pd.concat(PGAE_df, ignore_index=True)
        
        # 截断到目标标签数量
        PGAE_df['cum_sum'] = PGAE_df['true_label'].cumsum()
        cutoff_index = PGAE_df[PGAE_df['cum_sum'] == n_true_labels].index[0]
        PGAE_df = PGAE_df.iloc[:cutoff_index + 1]
        
        # 计算估计值和置信区间（支持可调RF与折数参数）
        if self.est_ci_params:
            tau, l_ci, h_ci = PGAE_est_ci_param(
                PGAE_df, self.X, self.F, self.Y,
                alpha=alpha,
                K=self.est_ci_params.get('K', 3),
                n_estimators_mu=self.est_ci_params.get('n_estimators_mu'),
                n_estimators_tau=self.est_ci_params.get('n_estimators_tau'),
                max_depth=self.est_ci_params.get('max_depth'),
                min_samples_split=self.est_ci_params.get('min_samples_split'),
                min_samples_leaf=self.est_ci_params.get('min_samples_leaf')
            )
        else:
            tau, l_ci, h_ci = PGAE_est_ci(PGAE_df, self.X, self.F, self.Y, alpha=alpha)
        
        return tau, l_ci, h_ci
    
    def run_experiments(self, df: pd.DataFrame, n_experiments: int = 1000, 
                       n_true_labels: int = 500, alpha: float = 0.9, 
                       seed: Optional[int] = None, use_concurrent: bool = True,
                       max_workers: Optional[int] = 10) -> Dict:
        """
        运行多次PGAE实验
        
        Args:
            df: 输入数据框
            n_experiments: 实验次数
            n_true_labels: 每次实验的真实标签数量
            alpha: 置信水平
            seed: 随机种子
            use_concurrent: 是否使用并发执行
            max_workers: 最大并发worker数量
            
        Returns:
            实验结果字典
        """
        logger.info(f"开始PGAE实验: {n_experiments}次实验, 每次{n_true_labels}个标签")
        logger.info(f"并发: {use_concurrent}, max_workers={max_workers}")
        
        # 准备数据
        df_prepared = self.prepare_data(df)
        true_value = df_prepared[self.Y].mean()
        
        logger.info(f"真实目标值: {true_value:.6f}")
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        # 运行实验
        start_time = time.time()
        
        if use_concurrent:
            # 并发执行
            tau_results, l_ci_results, h_ci_results = self._run_concurrent_experiments(
                df_prepared, n_experiments, n_true_labels, alpha, seed, max_workers
            )
        else:
            # 串行执行
            tau_results, l_ci_results, h_ci_results = self._run_sequential_experiments(
                df_prepared, n_experiments, n_true_labels, alpha, seed
            )
        
        end_time = time.time()
        
        # 汇总结果
        results = summary_results(tau_results, l_ci_results, h_ci_results, 
                                true_value, "PGAE")
        
        results.update({
            'tau_estimates': tau_results,
            'l_ci': l_ci_results,
            'h_ci': h_ci_results,
            'true_value': true_value,
            'execution_time': end_time - start_time,
            'parameters': {
                'X': self.X,
                'F': self.F,
                'Y': self.Y,
                'gamma': self.gamma,
                'n_experiments': n_experiments,
                'n_true_labels': n_true_labels,
                'alpha': alpha
            }
        })
        
        logger.info(f"PGAE实验完成，耗时: {results['execution_time']:.2f}秒")
        
        return results
    
    def _run_sequential_experiments(self, df_prepared: pd.DataFrame, n_experiments: int,
                                  n_true_labels: int, alpha: float, seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """串行执行实验"""
        tau_results = np.zeros(n_experiments)
        l_ci_results = np.zeros(n_experiments)
        h_ci_results = np.zeros(n_experiments)
        
        for i in tqdm(range(n_experiments), desc="PGAE实验进行中"):
            exp_seed = None if seed is None else seed + i
            tau, l_ci, h_ci = self.run_single_experiment(
                df_prepared, n_true_labels, alpha, exp_seed
            )
            
            tau_results[i] = tau
            l_ci_results[i] = l_ci
            h_ci_results[i] = h_ci
            
        return tau_results, l_ci_results, h_ci_results
    
    def _run_concurrent_experiments(self, df_prepared: pd.DataFrame, n_experiments: int,
                                  n_true_labels: int, alpha: float, seed: Optional[int], 
                                  max_workers: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """并发执行实验"""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), n_experiments, 10)  # 默认限制为10
        
        logger.info(f"使用 {max_workers} 个并发worker")
        
        # 准备任务参数
        task_args = []
        for i in range(n_experiments):
            exp_seed = None if seed is None else seed + i
            task_args.append((df_prepared, self.X, self.F, self.Y, n_true_labels, alpha, exp_seed, self.est_ci_params))
        
        # 初始化结果数组
        tau_results = np.zeros(n_experiments)
        l_ci_results = np.zeros(n_experiments)
        h_ci_results = np.zeros(n_experiments)
        
        # 并发执行
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_index = {executor.submit(_run_single_pgae_experiment, *args): i 
                             for i, args in enumerate(task_args)}
            
            # 收集结果
            for future in tqdm(as_completed(future_to_index), total=n_experiments, desc="PGAE并发实验"):
                index = future_to_index[future]
                try:
                    tau, l_ci, h_ci = future.result()
                    tau_results[index] = tau
                    l_ci_results[index] = l_ci
                    h_ci_results[index] = h_ci
                except Exception as exc:
                    logger.error(f'实验 {index} 生成异常: {exc}')
                    # 设置默认值，避免程序崩溃
                    tau_results[index] = 0.0
                    l_ci_results[index] = 0.0  
                    h_ci_results[index] = 0.0
                
                # 内存清理
                if index % 50 == 0:
                    gc.collect()
        
        return tau_results, l_ci_results, h_ci_results

    def save_results(self, results: Dict, filename: str = None) -> str:
        """
        保存实验结果
        
        Args:
            results: 实验结果字典
            filename: 输出文件名
            
        Returns:
            保存的文件名
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'pgae_results_{timestamp}.json'
        
        # 准备保存的数据（移除numpy数组）
        save_data = {
            'method': results['method'],
            'mse': results['mse'],
            'bias': results['bias'],
            'variance': results['variance'],
            'avg_ci_length': results['avg_ci_length'],
            'coverage_rate': results['coverage_rate'],
            'true_value': results['true_value'],
            'execution_time': results['execution_time'],
            'parameters': results['parameters'],
            'summary_statistics': {
                'mean_tau': float(np.mean(results['tau_estimates'])),
                'std_tau': float(np.std(results['tau_estimates'])),
                'mean_ci_length': float(np.mean(results['h_ci'] - results['l_ci'])),
                'min_tau': float(np.min(results['tau_estimates'])),
                'max_tau': float(np.max(results['tau_estimates']))
            }
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"PGAE结果已保存: {filename}")
        return filename

# 独立的worker函数，用于并发执行
def _run_single_pgae_experiment(df_prepared: pd.DataFrame, X: List[str], F: str, Y: str,
                               n_true_labels: int, alpha: float, seed: Optional[int],
                               ci_params: Optional[Dict] = None) -> Tuple[float, float, float]:
    """
    运行单个PGAE实验的worker函数
    
    Args:
        df_prepared: 准备好的数据框
        X: 协变量列名列表
        F: 预测列名
        Y: 真实标签列名
        n_true_labels: 目标真实标签数量
        alpha: 置信水平
        seed: 随机种子
        
    Returns:
        (估计值, 置信区间下界, 置信区间上界)
    """
    try:
        if seed is not None:
            np.random.seed(seed)
        
        df_work = df_prepared.copy()
        df_work['true_label'] = 0
        
        PGAE_df = []
        cnt_true = 0
        
        while cnt_true < n_true_labels:
            # 使用拒绝采样
            sampled_df = rejection_sample(df_work, df_work.columns, 'accept_prob', 
                                        n_samples=n_true_labels)
            
            # 根据实验概率决定是否标记
            u = np.random.uniform(0, 1, size=len(sampled_df))
            sampled_df['true_label'] = (u < sampled_df['exp_prob']).astype(int)
            
            cnt_true += sampled_df['true_label'].sum()
            PGAE_df.append(sampled_df)
        
        # 合并所有采样数据
        PGAE_df = pd.concat(PGAE_df, ignore_index=True)
        
        # 截断到目标标签数量
        PGAE_df['cum_sum'] = PGAE_df['true_label'].cumsum()
        cutoff_index = PGAE_df[PGAE_df['cum_sum'] == n_true_labels].index[0]
        PGAE_df = PGAE_df.iloc[:cutoff_index + 1]
        
        # 计算估计值和置信区间（支持可调RF与折数参数）
        if ci_params:
            tau, l_ci, h_ci = PGAE_est_ci_param(
                PGAE_df, X, F, Y,
                alpha=alpha,
                K=ci_params.get('K', 3),
                n_estimators_mu=ci_params.get('n_estimators_mu'),
                n_estimators_tau=ci_params.get('n_estimators_tau'),
                max_depth=ci_params.get('max_depth'),
                min_samples_split=ci_params.get('min_samples_split'),
                min_samples_leaf=ci_params.get('min_samples_leaf')
            )
        else:
            tau, l_ci, h_ci = PGAE_est_ci(PGAE_df, X, F, Y, alpha=alpha)
        
        # 内存清理
        del PGAE_df, df_work
        gc.collect()
        
        return tau, l_ci, h_ci
        
    except Exception as e:
        # 记录异常但返回默认值以保持稳定性
        print(f"Worker异常: {e}")
        return 0.0, 0.0, 0.0

def main():
    """主函数"""
    import argparse
    import time
    
    # 目标配置
    TARGET_CONFIGS = {
        'ECON1MOD': {
            'X': ['EDUCATION'],
            'F': 'ECON1MOD_LLM',
            'Y': 'ECON1MOD',
            'description': 'Economic conditions rating (1-4)'
        },
        'UNITY': {
            'X': ['EDUCATION'],
            'F': 'UNITY_LLM',
            'Y': 'UNITY',
            'description': 'US unity perception (1-2)'
        },
        'GPT1': {
            'X': ['EDUCATION'],
            'F': 'GPT1_LLM',
            'Y': 'GPT1',
            'description': 'ChatGPT familiarity (1-3)'
        },
        'MOREGUNIMPACT': {
            'X': ['EDUCATION'],
            'F': 'MOREGUNIMPACT_LLM',
            'Y': 'MOREGUNIMPACT',
            'description': 'Gun control impact (1-3)'
        },
        'GAMBLERESTR': {
            'X': ['EDUCATION'],
            'F': 'GAMBLERESTR_LLM',
            'Y': 'GAMBLERESTR',
            'description': 'Gambling restriction opinion (1-3)'
        }
    }
    
    parser = argparse.ArgumentParser(description='PGAE估计器')
    parser.add_argument('data_file', help='数据文件路径')
    parser.add_argument('--target', '-t', default='ECON1MOD', 
                       choices=list(TARGET_CONFIGS.keys()),
                       help='预测目标 (默认: ECON1MOD)')
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
    parser.add_argument('--output', '-o', help='输出文件名')
    parser.add_argument('--concurrent', action='store_true',
                       help='使用并发执行')
    
    args = parser.parse_args()
    
    # 获取目标配置
    config = TARGET_CONFIGS[args.target]
    
    # 加载数据
    logger.info(f"加载数据: {args.data_file}")
    df = pd.read_csv(args.data_file)
    
    logger.info(f"目标: {args.target} - {config['description']}")
    
    # 初始化估计器
    estimator = PGAEEstimator(
        X=config['X'],
        F=config['F'],
        Y=config['Y'],
        gamma=args.gamma
    )
    
    # 运行实验
    results = estimator.run_experiments(
        df, 
        n_experiments=args.experiments,
        n_true_labels=args.labels,
        alpha=args.alpha,
        seed=args.seed,
        use_concurrent=args.concurrent
    )
    
    # 生成输出文件名
    output_file = args.output
    if output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f'pgae_{args.target.lower()}_{timestamp}.json'
    
    # 保存结果
    saved_file = estimator.save_results(results, output_file)
    
    print(f"\n✅ PGAE实验完成!")
    print(f"目标: {args.target}")
    print(f"实验设置: {args.experiments} experiments, {args.labels} labels per experiment")
    print(f"结果文件: {saved_file}")

if __name__ == "__main__":
    main()
