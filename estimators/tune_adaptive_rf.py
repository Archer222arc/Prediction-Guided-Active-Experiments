#!/usr/bin/env python3
"""
PGAE和Adaptive PGAE RandomForest参数调优
不改变算法本质，只调优RandomForest参数
为两个估计器分别寻找最优参数组合
"""

import numpy as np
import pandas as pd
import time
import json
import shutil
import importlib
import os
import sys
from typing import Dict, List, Tuple
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(SCRIPT_DIR, 'utils.py')
UTILS_BACKUP_PATH = os.path.join(SCRIPT_DIR, 'utils_backup.py')

def backup_utils():
    """备份estimators/utils.py文件（按脚本所在目录计算）"""
    shutil.copy(UTILS_PATH, UTILS_BACKUP_PATH)
    print("已备份estimators/utils.py为estimators/utils_backup.py")

def restore_utils():
    """恢复estimators/utils.py文件"""
    if os.path.exists(UTILS_BACKUP_PATH):
        shutil.copy(UTILS_BACKUP_PATH, UTILS_PATH)
        print("已恢复estimators/utils.py")
    else:
        print("未找到estimators/utils_backup.py，跳过恢复")

def modify_utils_rf_params(pgae_params: Dict, adaptive_params: Dict):
    """
    修改utils.py中的RandomForest参数
    
    Args:
        pgae_params: PGAE的RandomForest参数
        adaptive_params: Adaptive PGAE的RandomForest参数
    """
    with open(UTILS_PATH, 'r') as f:
        content = f.read()
    
    # 构建参数字符串
    def params_to_str(params):
        return ', '.join([f'{k}={v}' for k, v in params.items()])
    
    pgae_params_str = params_to_str(pgae_params)
    adaptive_params_str = params_to_str(adaptive_params)
    
    lines = content.split('\n')
    in_pgae_function = False
    in_adaptive_function = False
    
    for i, line in enumerate(lines):
        # 检测函数范围
        if 'def PGAE_est_ci(' in line:
            in_pgae_function = True
            in_adaptive_function = False
        elif 'def adaptive_PGAE_est_ci(' in line:
            in_adaptive_function = True
            in_pgae_function = False
        elif line.startswith('def ') and not line.startswith('    '):
            in_pgae_function = False
            in_adaptive_function = False
            
        # 替换参数
        if 'model_mu = RandomForestRegressor(' in line:
            if in_pgae_function:
                lines[i] = f"        model_mu = RandomForestRegressor({pgae_params_str})"
                print(f"替换PGAE model_mu: {lines[i].strip()}")
            elif in_adaptive_function:
                lines[i] = f"        model_mu = RandomForestRegressor({adaptive_params_str})"
                print(f"替换Adaptive model_mu: {lines[i].strip()}")
        elif 'model_tau = RandomForestRegressor(' in line:
            if in_pgae_function:
                lines[i] = f"        model_tau = RandomForestRegressor({pgae_params_str})"
                print(f"替换PGAE model_tau: {lines[i].strip()}")
            elif in_adaptive_function:
                lines[i] = f"        model_tau = RandomForestRegressor({adaptive_params_str})"
                print(f"替换Adaptive model_tau: {lines[i].strip()}")
        elif 'final_model_mu = RandomForestRegressor(' in line and in_adaptive_function:
            lines[i] = f"        final_model_mu = RandomForestRegressor({adaptive_params_str})"
            print(f"替换Adaptive final_model_mu: {lines[i].strip()}")
    
    with open(UTILS_PATH, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"参数修改完成：PGAE({pgae_params_str}), Adaptive({adaptive_params_str})")

def test_single_rf_combination(df: pd.DataFrame, target_config: Dict, 
                              pgae_params: Dict, adaptive_params: Dict,
                              n_experiments: int = 15) -> Dict:
    """
    测试单个RandomForest参数组合
    
    Args:
        df: 数据DataFrame
        target_config: 目标配置 
        pgae_params: PGAE的RandomForest参数
        adaptive_params: Adaptive PGAE的RandomForest参数
        n_experiments: 实验次数
    
    Returns:
        测试结果
    """
    # 修改utils.py中的参数
    modify_utils_rf_params(pgae_params, adaptive_params)
    
    # 使模块缓存失效并强制重新导入，确保新参数生效
    importlib.invalidate_caches()
    for name in ['utils', 'adaptive_pgae_estimator', 'pgae_estimator']:
        if name in sys.modules:
            del sys.modules[name]

    # 确保estimators目录在搜索路径前列，以加载本目录下的utils.py
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    # 同时通过环境变量影响子进程的导入搜索路径（并发执行时生效）
    current_pp = os.environ.get('PYTHONPATH', '')
    if SCRIPT_DIR not in current_pp.split(os.pathsep):
        os.environ['PYTHONPATH'] = (SCRIPT_DIR + (os.pathsep + current_pp if current_pp else ''))

    adaptive_pgae_estimator = importlib.import_module('adaptive_pgae_estimator')
    pgae_estimator = importlib.import_module('pgae_estimator')
    
    results = {}
    
    try:
        # 测试PGAE
        print(f"    测试PGAE...")
        pgae_estimator_class = pgae_estimator.PGAEEstimator
        pgae_est = pgae_estimator_class(
            X=target_config['X'],
            F=target_config['F'],
            Y=target_config['Y'],
            gamma=0.5
        )
        
        start_time = time.time()
        pgae_results = pgae_est.run_experiments(
            df, n_experiments=n_experiments, n_true_labels=500, alpha=0.9, 
            seed=42, use_concurrent=True, max_workers=6
        )
        pgae_time = time.time() - start_time
        
        results['pgae'] = {
            'mse': pgae_results['mse'],
            'bias': pgae_results['bias'],
            'variance': pgae_results['variance'],
            'coverage_rate': pgae_results['coverage_rate'],
            'time': pgae_time
        }
        
        # 测试Adaptive PGAE
        print(f"    测试Adaptive PGAE...")
        adaptive_estimator_class = adaptive_pgae_estimator.AdaptivePGAEEstimator
        adaptive_est = adaptive_estimator_class(
            X=target_config['X'],
            F=target_config['F'],
            Y=target_config['Y'],
            gamma=0.5,
            batch_size=250
        )
        
        start_time = time.time()
        adaptive_results = adaptive_est.run_experiments(
            df, n_experiments=n_experiments, n_true_labels=500, alpha=0.9,
            seed=42, use_concurrent=True, max_workers=6
        )
        adaptive_time = time.time() - start_time
        
        results['adaptive'] = {
            'mse': adaptive_results['mse'],
            'bias': adaptive_results['bias'],
            'variance': adaptive_results['variance'],
            'coverage_rate': adaptive_results['coverage_rate'],
            'time': adaptive_time
        }
        
    except Exception as e:
        print(f"    测试失败: {e}")
        results['error'] = str(e)
    
    return results

def test_rf_parameters(data_file: str, target: str = 'ECON1MOD') -> Dict:
    """
    测试不同RandomForest参数组合，为PGAE和Adaptive PGAE分别寻找最优参数
    """
    # 加载数据
    df = pd.read_csv(data_file)
    
    # 目标配置
    TARGET_CONFIGS = {
        'ECON1MOD': {
            'X': ['EDUCATION'],
            'F': 'ECON1MOD_LLM',
            'Y': 'ECON1MOD',
        }
    }
    config = TARGET_CONFIGS[target]
    
    # 定义要测试的RF参数组合
    rf_configs = [
        # 当前默认配置
        {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42, 'name': 'default'},
        
        # 保守配置（防止过拟合）
        {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42, 'name': 'conservative'},
        
        # 增强配置（更多树）
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42, 'name': 'enhanced'},
        
        # 稳定配置（中等参数）
        {'n_estimators': 100, 'max_depth': 12, 'min_samples_split': 3, 'min_samples_leaf': 2, 'random_state': 42, 'name': 'stable'},
        
        # 轻量配置（更快训练）
        {'n_estimators': 30, 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 5, 'random_state': 42, 'name': 'lightweight'},
        
        # 平衡配置
        {'n_estimators': 75, 'max_depth': 12, 'min_samples_split': 5, 'min_samples_leaf': 1, 'random_state': 42, 'name': 'balanced'}
    ]
    
    print(f"测试 {len(rf_configs)} 个RandomForest参数配置...")
    print("将为PGAE和Adaptive PGAE分别寻找最优参数组合")
    
    # 备份原始utils.py
    backup_utils()
    
    all_results = {}
    
    try:
        for i, rf_config in enumerate(tqdm(rf_configs, desc="Testing RF configs")):
            config_name = rf_config['name']
            
            print(f"\n配置 {i+1}/{len(rf_configs)}: {config_name}")
            print(f"  参数: n_est={rf_config['n_estimators']}, max_depth={rf_config.get('max_depth', 'None')}")
            print(f"        min_split={rf_config['min_samples_split']}, min_leaf={rf_config['min_samples_leaf']}")
            
            # 移除name字段用于参数传递
            params = {k: v for k, v in rf_config.items() if k != 'name'}
            
            # 测试该参数组合 (PGAE和Adaptive PGAE使用相同参数)
            results = test_single_rf_combination(df, config, params, params, n_experiments=12)
            
            all_results[config_name] = {
                'params': rf_config,
                'results': results
            }
            
            if 'error' not in results:
                print(f"  PGAE - MSE: {results['pgae']['mse']:.6f}, 覆盖率: {results['pgae']['coverage_rate']:.3f}")
                print(f"  Adaptive - MSE: {results['adaptive']['mse']:.6f}, 覆盖率: {results['adaptive']['coverage_rate']:.3f}")
            else:
                print(f"  失败: {results['error']}")
    
    finally:
        # 恢复原始utils.py
        restore_utils()
    
    return all_results

def analyze_results(all_results: Dict):
    """分析调参结果并给出建议"""
    print("\n" + "="*100)
    print("RANDOMFOREST PARAMETER TUNING RESULTS")
    print("="*100)
    
    # 过滤有效结果
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v['results']}
    
    if not valid_results:
        print("❌ 所有参数组合测试均失败")
        return
    
    # 按PGAE MSE排序
    pgae_sorted = sorted(valid_results.items(), key=lambda x: x[1]['results']['pgae']['mse'])
    
    # 按Adaptive PGAE MSE排序  
    adaptive_sorted = sorted(valid_results.items(), key=lambda x: x[1]['results']['adaptive']['mse'])
    
    print(f"\n🏆 PGAE 最佳参数排名:")
    print(f"{'Rank':<5} {'Config':<12} {'MSE':<12} {'Coverage':<10} {'n_est':<6} {'max_depth':<10} {'min_split':<10}")
    print("-" * 85)
    
    for rank, (config_name, config_data) in enumerate(pgae_sorted[:3], 1):
        params = config_data['params']
        results = config_data['results']['pgae']
        print(f"{rank:<5} {config_name:<12} {results['mse']:<12.6f} {results['coverage_rate']:<10.3f} "
              f"{params['n_estimators']:<6} {str(params['max_depth']):<10} {params['min_samples_split']:<10}")
    
    print(f"\n🏆 Adaptive PGAE 最佳参数排名:")
    print(f"{'Rank':<5} {'Config':<12} {'MSE':<12} {'Coverage':<10} {'n_est':<6} {'max_depth':<10} {'min_split':<10}")
    print("-" * 85)
    
    for rank, (config_name, config_data) in enumerate(adaptive_sorted[:3], 1):
        params = config_data['params']
        results = config_data['results']['adaptive']
        print(f"{rank:<5} {config_name:<12} {results['mse']:<12.6f} {results['coverage_rate']:<10.3f} "
              f"{params['n_estimators']:<6} {str(params['max_depth']):<10} {params['min_samples_split']:<10}")
    
    # 推荐最佳参数
    best_pgae = pgae_sorted[0]
    best_adaptive = adaptive_sorted[0]
    
    print(f"\n✅ 最终推荐:")
    print(f"PGAE 最佳配置: {best_pgae[0]} (MSE: {best_pgae[1]['results']['pgae']['mse']:.6f})")
    pgae_params = best_pgae[1]['params']
    print(f"  参数: {', '.join([f'{k}={v}' for k, v in pgae_params.items() if k != 'name'])}")
    
    print(f"\nAdaptive PGAE 最佳配置: {best_adaptive[0]} (MSE: {best_adaptive[1]['results']['adaptive']['mse']:.6f})")
    adaptive_params = best_adaptive[1]['params']  
    print(f"  参数: {', '.join([f'{k}={v}' for k, v in adaptive_params.items() if k != 'name'])}")
    
    print(f"\n📝 应用建议:")
    print(f"在estimators/utils.py中分别设置不同的RandomForest参数:")
    print(f"1. PGAE_est_ci 中的 model_mu 与 model_tau:")
    pgae_str = ', '.join([f'{k}={v}' for k, v in pgae_params.items() if k != 'name'])
    print(f"   RandomForestRegressor({pgae_str})")
    
    print(f"\n2. adaptive_PGAE_est_ci 中的 model_mu 与 model_tau:")
    adaptive_str = ', '.join([f'{k}={v}' for k, v in adaptive_params.items() if k != 'name'])
    print(f"   RandomForestRegressor({adaptive_str})")

def main():
    """主函数 - RandomForest参数调优"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python tune_adaptive_rf.py <data_file>")
        print("  python tune_adaptive_rf.py data/NPORS_2024_for_public_release_with_LLM_prediction.csv")
        return
    
    data_file = sys.argv[1]
    print("="*100)
    print("PGAE & ADAPTIVE PGAE RANDOMFOREST PARAMETER TUNING")
    print("="*100)
    print(f"数据文件: {data_file}")
    print("目标: 为PGAE和Adaptive PGAE分别寻找最优RandomForest参数")
    
    # 运行调参实验
    all_results = test_rf_parameters(data_file)
    
    # 分析结果
    analyze_results(all_results)
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f'rf_parameter_tuning_results_{timestamp}.json'
    
    # 转换numpy类型
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
        json.dump(convert_numpy(all_results), f, indent=2)
    
    print(f"\n✅ 调参完成！结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
