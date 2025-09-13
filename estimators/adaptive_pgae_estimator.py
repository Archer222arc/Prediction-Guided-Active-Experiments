#!/usr/bin/env python3
"""
Adaptive PGAE (Prediction-Guided Active Experiments) Estimator
自适应预测引导主动实验估计器
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
    get_PGAE_design, rejection_sample, PGAE_est_ci,
    overwrite_merge, summary_results, validate_data
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptivePGAEEstimator:
    """自适应PGAE估计器类"""
    
    def __init__(self, X: List[str], F: str, Y: str, gamma: float = 0.5, batch_size: int = 100,
                 design_update_freq: int = 1, warmup_batches: int = 0):
        """
        初始化自适应PGAE估计器
        
        Args:
            X: 协变量列名列表
            F: 预测列名
            Y: 真实标签列名
            gamma: PGAE参数，控制实验概率的权重
            batch_size: 每批次的样本数量
            design_update_freq: 设计更新频率（每几个batch更新一次设计，默认1为每批都更新）
            warmup_batches: 预热批次数（前几批固定使用初始设计，不进行自适应更新）
        """
        self.X = X
        self.F = F
        self.Y = Y
        self.gamma = gamma
        self.batch_size = batch_size
        self.design_update_freq = design_update_freq
        self.warmup_batches = warmup_batches
        
        logger.info(f"自适应PGAE估计器初始化: X={X}, F={F}, Y={Y}, gamma={gamma}, batch_size={batch_size}")
        if design_update_freq > 1:
            logger.info(f"  设计更新频率: 每{design_update_freq}批更新一次（减少波动与开销）")
        if warmup_batches > 0:
            logger.info(f"  预热批次数: 前{warmup_batches}批使用初始设计（提升初期稳定性）")
    
    def prepare_initial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备初始数据
        
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
        
        return df_clean
    
    def run_single_adaptive_experiment(self, df: pd.DataFrame, n_true_labels: int = 500, 
                                     alpha: float = 0.9, seed: Optional[int] = None) -> Tuple[float, float, float]:
        """
        运行单次自适应PGAE实验
        
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
        
        cnt_true = 0
        PGAE_df = pd.DataFrame()
        
        # 初始化设计参数
        initial_design = pd.DataFrame()
        for x_val in df_work[self.X].drop_duplicates().values:
            if len(self.X) == 1:
                x_val = [x_val]
            row = dict(zip(self.X, x_val))
            
            # 安全获取sample_pmf
            if len(self.X) == 1:
                mask = df_work[self.X[0]] == x_val[0]
            else:
                mask = df_work[self.X].eq(x_val).all(axis=1)
                
            matched_rows = df_work.loc[mask, 'true_pmf']
            sample_pmf_val = matched_rows.iloc[0] if len(matched_rows) > 0 else 1.0
            
            row.update({
                'accept_prob': 1.0,
                'exp_prob': 1.0,
                'sample_pmf': sample_pmf_val
            })
            initial_design = pd.concat([initial_design, pd.DataFrame([row])], ignore_index=True)
        
        df_work = overwrite_merge(df_work, initial_design, on=self.X, how='left')
        
        # 自适应采样循环
        batch_count = 0
        logger.info(f"开始自适应采样：目标{n_true_labels}个标签，批次大小{self.batch_size}")
        logger.info(f"预热设置：前{self.warmup_batches}批固定设计，每{self.design_update_freq}批更新一次")
        
        while cnt_true < n_true_labels:
            batch_count += 1
            
            # 采样一个批次
            sampled_df = rejection_sample(df_work, df_work.columns, 'accept_prob', 
                                        n_samples=self.batch_size)
            
            # 根据实验概率决定是否标记
            u = np.random.uniform(0, 1, size=len(sampled_df))
            sampled_df['true_label'] = (u < sampled_df['exp_prob']).astype(int)
            
            batch_true_labels = sampled_df['true_label'].sum()
            cnt_true += batch_true_labels
            PGAE_df = pd.concat([PGAE_df, sampled_df], ignore_index=True)
            
            logger.info(f"第{batch_count}批：获得{batch_true_labels}个标签，总计{cnt_true}/{n_true_labels}")
            
            # 设计更新逻辑：考虑预热期和更新频率
            should_update_design = (
                len(PGAE_df[PGAE_df['true_label'] == 1]) > 0 and  # 有标记数据
                batch_count > self.warmup_batches and  # 过了预热期
                batch_count % self.design_update_freq == 0  # 到了更新频率
            )
            
            if should_update_design:
                try:
                    updated_design = get_PGAE_design(PGAE_df, self.X, self.F, self.Y, self.gamma)
                    df_work = overwrite_merge(df_work, updated_design, on=self.X, how='left')
                    logger.info(f"第{batch_count}批：✅更新设计参数")
                except Exception as e:
                    # 如果设计更新失败，继续使用当前设计
                    logger.warning(f"第{batch_count}批设计更新失败: {e}")
            elif batch_count <= self.warmup_batches:
                logger.info(f"第{batch_count}批：🔥预热期，使用固定设计")
            else:
                logger.info(f"第{batch_count}批：⏭️跳过设计更新（频率控制）")
        
        # 截断到目标标签数量
        if cnt_true > n_true_labels:
            PGAE_df['cum_sum'] = PGAE_df['true_label'].cumsum()
            cutoff_mask = PGAE_df['cum_sum'] <= n_true_labels
            PGAE_df = PGAE_df[cutoff_mask]
        
        # 计算估计值和置信区间 - 采样完成后使用regular PGAE的CV方法
        tau, l_ci, h_ci = PGAE_est_ci(PGAE_df, self.X, self.F, self.Y, alpha=alpha, K=3)
        
        return tau, l_ci, h_ci
    
    def run_experiments(self, df: pd.DataFrame, n_experiments: int = 1000, 
                       n_true_labels: int = 500, alpha: float = 0.9, 
                       seed: Optional[int] = None, use_concurrent: bool = True,
                       max_workers: Optional[int] = 10) -> Dict:
        """
        运行多次自适应PGAE实验
        
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
        logger.info(f"开始自适应PGAE实验: {n_experiments}次实验, 每次{n_true_labels}个标签")
        if use_concurrent:
            logger.info("使用并发执行模式")
        
        # 准备数据
        df_prepared = self.prepare_initial_data(df)
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
                                true_value, "Adaptive PGAE")
        
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
                'batch_size': self.batch_size,
                'n_experiments': n_experiments,
                'n_true_labels': n_true_labels,
                'alpha': alpha
            }
        })
        
        logger.info(f"自适应PGAE实验完成，耗时: {results['execution_time']:.2f}秒")
        
        return results
    
    def _run_sequential_experiments(self, df_prepared: pd.DataFrame, n_experiments: int,
                                  n_true_labels: int, alpha: float, seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """串行执行实验"""
        tau_results = np.zeros(n_experiments)
        l_ci_results = np.zeros(n_experiments)
        h_ci_results = np.zeros(n_experiments)
        
        for i in tqdm(range(n_experiments), desc="自适应PGAE实验进行中"):
            exp_seed = None if seed is None else seed + i
            tau, l_ci, h_ci = self.run_single_adaptive_experiment(
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
            task_args.append((df_prepared, self.X, self.F, self.Y, self.gamma, self.batch_size,
                              self.design_update_freq, self.warmup_batches,
                              n_true_labels, alpha, exp_seed))
        
        # 初始化结果数组
        tau_results = np.zeros(n_experiments)
        l_ci_results = np.zeros(n_experiments)
        h_ci_results = np.zeros(n_experiments)
        
        # 并发执行
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_index = {executor.submit(_run_single_adaptive_experiment, *args): i 
                             for i, args in enumerate(task_args)}
            
            # 收集结果
            for future in tqdm(as_completed(future_to_index), total=n_experiments, desc="自适应PGAE并发实验"):
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
            filename = f'adaptive_pgae_results_{timestamp}.json'
        
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
        
        logger.info(f"自适应PGAE结果已保存: {filename}")
        return filename

# 独立的worker函数，用于并发执行
def _run_single_adaptive_experiment(df_prepared: pd.DataFrame, X: List[str], F: str, Y: str,
                                   gamma: float, batch_size: int,
                                   design_update_freq: int, warmup_batches: int,
                                   n_true_labels: int, 
                                   alpha: float, seed: Optional[int]) -> Tuple[float, float, float]:
    """
    运行单个自适应PGAE实验的worker函数
    
    Args:
        df_prepared: 准备好的数据框
        X: 协变量列名列表
        F: 预测列名
        Y: 真实标签列名
        gamma: PGAE参数
        batch_size: 批次大小
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
        
        cnt_true = 0
        PGAE_df = pd.DataFrame()
        
        # 初始化设计
        initial_design = get_PGAE_design(df_work, X, F, Y, gamma)
        initial_design['accept_prob'] = 1
        initial_design['exp_prob'] = 1
        df_work = overwrite_merge(df_work, initial_design, on=X, how='left')
        df_work['sample_pmf'] = df_work['true_pmf']
        
        batch_count = 0
        while cnt_true < n_true_labels:
            # 使用拒绝采样
            sampled_df = rejection_sample(df_work, df_work.columns.tolist(), 'accept_prob', 
                                        n_samples=batch_size)
            
            # 根据实验概率决定是否标记
            u = np.random.uniform(0, 1, size=len(sampled_df))
            sampled_df['true_label'] = (u < sampled_df['exp_prob']).astype(int)
            
            batch_true = sampled_df['true_label'].sum()
            cnt_true += batch_true
            PGAE_df = pd.concat([PGAE_df, sampled_df], ignore_index=True)
            
            batch_count += 1
            # 如果有足够的标记数据，按预热与频率策略更新设计
            if len(PGAE_df[PGAE_df['true_label'] == 1]) > 0:
                try:
                    if batch_count > max(0, int(warmup_batches)):
                        if batch_count % max(1, int(design_update_freq)) == 0:
                            updated_design = get_PGAE_design(PGAE_df, X, F, Y, gamma)
                            df_work = overwrite_merge(df_work, updated_design, on=X, how='left')
                except Exception:
                    # 如果设计更新失败，继续使用当前设计
                    pass
        
        # 截断到目标标签数量
        if cnt_true > n_true_labels:
            PGAE_df['cum_sum'] = PGAE_df['true_label'].cumsum()
            cutoff_mask = PGAE_df['cum_sum'] <= n_true_labels
            PGAE_df = PGAE_df[cutoff_mask]
        
        # 计算估计值和置信区间 - 采样完成后使用regular PGAE的CV方法
        tau, l_ci, h_ci = PGAE_est_ci(PGAE_df, X, F, Y, alpha=alpha, K=3)
        
        # 内存清理
        del PGAE_df, df_work
        gc.collect()
        
        return tau, l_ci, h_ci
        
    except Exception as e:
        # 记录异常但返回默认值以保持稳定性
        print(f"自适应PGAE Worker异常: {e}")
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
    
    parser = argparse.ArgumentParser(description='自适应PGAE估计器')
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
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                       help='批大小 (默认: 100)')
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
    estimator = AdaptivePGAEEstimator(
        X=config['X'],
        F=config['F'],
        Y=config['Y'],
        gamma=args.gamma,
        batch_size=args.batch_size
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
        output_file = f'adaptive_pgae_{args.target.lower()}_{timestamp}.json'
    
    # 保存结果
    saved_file = estimator.save_results(results, output_file)
    
    print(f"\n✅ 自适应PGAE实验完成!")
    print(f"目标: {args.target}")
    print(f"实验设置: {args.experiments} experiments, {args.labels} labels per experiment")
    print(f"结果文件: {saved_file}")

if __name__ == "__main__":
    main()
