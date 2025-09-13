#!/usr/bin/env python3
"""
Naive Estimator - 基线对比估计器
使用最简单的方法：随机采样 + 样本均值 + 正态分布置信区间
"""

import numpy as np
import pandas as pd
import time
import logging
from tqdm import tqdm
from scipy.stats import norm, t
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import gc
from utils import summary_results, validate_data

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NaiveEstimator:
    """
    Naive估计器 - 使用最简单的统计方法
    
    方法：
    1. 简单随机采样 (不考虑预测信息)
    2. 样本均值作为点估计
    3. 样本标准差 + t分布置信区间
    """
    
    def __init__(self, X: List[str], F: str, Y: str, gamma: float = 0.5):
        """
        初始化Naive估计器
        
        Args:
            X: 协变量列名列表 (在naive方法中不使用，但保持接口一致)
            F: 预测列名 (在naive方法中不使用，但保持接口一致)
            Y: 真实标签列名
            gamma: 参数 (在naive方法中不使用，但保持接口一致)
        """
        self.X = X
        self.F = F
        self.Y = Y
        self.gamma = gamma
        
        logger.info(f"Naive估计器初始化: Y={Y}, gamma={gamma} (忽略X={X}, F={F})")
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备数据 - Naive方法只需要清理数据
        
        Args:
            df: 输入数据框
            
        Returns:
            处理后的数据框
        """
        # 验证数据
        required_columns = [self.Y]
        if not validate_data(df, required_columns):
            raise ValueError("Data validation failed")
        
        # 过滤有效数据 - 只考虑Y列
        df_clean = df[[self.Y]].copy()
        df_clean = df_clean[df_clean[self.Y] < 10]  # 过滤异常值
        df_clean = df_clean.dropna()
        
        logger.info(f"数据清理后样本数: {len(df_clean)}")
        
        return df_clean
    
    def run_single_experiment(self, df: pd.DataFrame, n_true_labels: int = 500, 
                             alpha: float = 0.9, seed: Optional[int] = None) -> Tuple[float, float, float]:
        """
        运行单次Naive实验
        
        Args:
            df: 准备好的数据框
            n_true_labels: 目标样本数量
            alpha: 置信水平
            seed: 随机种子
            
        Returns:
            (估计值, 置信区间下界, 置信区间上界)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 使用gamma控制实际采样数量
        actual_sample_size = max(1, int(n_true_labels * self.gamma))
        
        # 简单随机采样
        if len(df) < actual_sample_size:
            # 如果数据不够，有放回采样
            sampled_df = df.sample(n=actual_sample_size, replace=True)
        else:
            # 无放回采样
            sampled_df = df.sample(n=actual_sample_size, replace=False)
        
        # 计算样本均值和标准差
        sample_values = sampled_df[self.Y].values
        tau_naive = np.mean(sample_values)
        sample_std = np.std(sample_values, ddof=1)  # 使用无偏估计
        
        # 使用t分布计算置信区间 (更准确的小样本方法)
        n = len(sample_values)
        t_alpha = t.ppf(1 - (1 - alpha) / 2, df=n-1)
        margin_of_error = t_alpha * sample_std / np.sqrt(n)
        
        l_ci = tau_naive - margin_of_error
        h_ci = tau_naive + margin_of_error
        
        return tau_naive, l_ci, h_ci
    
    def run_experiments(self, df: pd.DataFrame, n_experiments: int = 1000, 
                       n_true_labels: int = 500, alpha: float = 0.9, 
                       seed: Optional[int] = None, use_concurrent: bool = True,
                       max_workers: Optional[int] = 10) -> Dict:
        """
        运行多次Naive实验
        
        Args:
            df: 输入数据框
            n_experiments: 实验次数
            n_true_labels: 每次实验的样本数量
            alpha: 置信水平
            seed: 随机种子
            use_concurrent: 是否使用并发执行
            max_workers: 最大并发worker数量
            
        Returns:
            实验结果字典
        """
        logger.info(f"开始Naive实验: {n_experiments}次实验, 每次{n_true_labels}个样本")
        if use_concurrent:
            logger.info("使用并发执行模式")
        
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
                                true_value, "Naive")
        
        results.update({
            'tau_estimates': tau_results,
            'l_ci': l_ci_results,
            'h_ci': h_ci_results,
            'true_value': true_value,
            'execution_time': end_time - start_time,
            'parameters': {
                'Y': self.Y,
                'n_experiments': n_experiments,
                'n_true_labels': n_true_labels,
                'alpha': alpha
            }
        })
        
        logger.info(f"Naive实验完成，耗时: {results['execution_time']:.2f}秒")
        
        return results
    
    def _run_sequential_experiments(self, df_prepared: pd.DataFrame, n_experiments: int,
                                  n_true_labels: int, alpha: float, seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """串行执行实验"""
        tau_results = np.zeros(n_experiments)
        l_ci_results = np.zeros(n_experiments)
        h_ci_results = np.zeros(n_experiments)
        
        for i in tqdm(range(n_experiments), desc="Naive实验进行中"):
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
            max_workers = min(mp.cpu_count(), n_experiments, 10)  # 限制最大worker数量
        
        logger.info(f"使用 {max_workers} 个并发worker")
        
        # 准备任务参数
        task_args = []
        for i in range(n_experiments):
            exp_seed = None if seed is None else seed + i
            task_args.append((df_prepared, self.Y, self.gamma, n_true_labels, alpha, exp_seed))
        
        # 初始化结果数组
        tau_results = np.zeros(n_experiments)
        l_ci_results = np.zeros(n_experiments)
        h_ci_results = np.zeros(n_experiments)
        
        # 并发执行
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_index = {executor.submit(_run_single_naive_experiment, *args): i 
                             for i, args in enumerate(task_args)}
            
            # 收集结果
            for future in tqdm(as_completed(future_to_index), total=n_experiments, desc="Naive并发实验"):
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
            filename = f'naive_results_{timestamp}.json'
        
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
        
        logger.info(f"Naive结果已保存: {filename}")
        return filename

# 独立的worker函数，用于并发执行
def _run_single_naive_experiment(df_prepared: pd.DataFrame, Y: str, gamma: float, n_true_labels: int, 
                                 alpha: float, seed: Optional[int]) -> Tuple[float, float, float]:
    """
    运行单个Naive实验的worker函数
    
    Args:
        df_prepared: 准备好的数据框
        Y: 真实标签列名
        gamma: 控制采样数量的参数
        n_true_labels: 目标样本数量
        alpha: 置信水平
        seed: 随机种子
        
    Returns:
        (估计值, 置信区间下界, 置信区间上界)
    """
    try:
        if seed is not None:
            np.random.seed(seed)
        
        # 使用gamma控制实际采样数量
        actual_sample_size = max(1, int(n_true_labels * gamma))
        
        # 简单随机采样
        if len(df_prepared) < actual_sample_size:
            # 如果数据不够，有放回采样
            sampled_df = df_prepared.sample(n=actual_sample_size, replace=True)
        else:
            # 无放回采样
            sampled_df = df_prepared.sample(n=actual_sample_size, replace=False)
        
        # 计算样本均值和标准差
        sample_values = sampled_df[Y].values
        tau_naive = np.mean(sample_values)
        sample_std = np.std(sample_values, ddof=1)  # 使用无偏估计
        
        # 使用t分布计算置信区间
        n = len(sample_values)
        t_alpha = t.ppf(1 - (1 - alpha) / 2, df=n-1)
        margin_of_error = t_alpha * sample_std / np.sqrt(n)
        
        l_ci = tau_naive - margin_of_error
        h_ci = tau_naive + margin_of_error
        
        # 内存清理
        del sampled_df
        gc.collect()
        
        return tau_naive, l_ci, h_ci
        
    except Exception as e:
        # 记录异常但返回默认值以保持稳定性
        print(f"Naive Worker异常: {e}")
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
    
    parser = argparse.ArgumentParser(description='Naive基线估计器')
    parser.add_argument('data_file', help='数据文件路径')
    parser.add_argument('--target', '-t', default='ECON1MOD', 
                       choices=list(TARGET_CONFIGS.keys()),
                       help='预测目标 (默认: ECON1MOD)')
    parser.add_argument('--experiments', '-e', type=int, default=100,
                       help='实验次数 (默认: 100)')
    parser.add_argument('--labels', '-l', type=int, default=500,
                       help='每次实验的真实标签数量 (默认: 500)')
    parser.add_argument('--gamma', '-g', type=float, default=0.5,
                       help='gamma参数 (控制实际采样数量=n_true_labels*gamma) (默认: 0.5)')
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
    
    # 初始化估计器 (Naive只使用Y列，其他参数忽略)
    estimator = NaiveEstimator(
        X=config['X'],  # 不使用但保持接口一致
        F=config['F'],  # 不使用但保持接口一致
        Y=config['Y'],
        gamma=args.gamma  # 传递但不使用，保持接口一致
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
        output_file = f'naive_{args.target.lower()}_{timestamp}.json'
    
    # 保存结果
    saved_file = estimator.save_results(results, output_file)
    
    print(f"\n✅ Naive实验完成!")
    print(f"目标: {args.target}")
    print(f"实验设置: {args.experiments} experiments, {args.labels} labels per experiment")
    print(f"结果文件: {saved_file}")

if __name__ == "__main__":
    main()
