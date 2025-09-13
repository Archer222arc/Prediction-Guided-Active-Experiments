#!/usr/bin/env python3
"""
CI-mode parameter tuning for PGAE and Adaptive PGAE
在固定CI宽度目标下，搜索使所需标签数最小的参数组合，并报告MSE等指标。

- 支持方法：PGAE、Adaptive_PGAE
- 可调参数：
  - 通用：gamma 网格
  - Adaptive：batch_size、design_update_freq、warmup_batches
"""

import json
import time
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pgae_estimator import PGAEEstimator
from adaptive_pgae_estimator import AdaptivePGAEEstimator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


TARGET_CONFIGS = {
    'ECON1MOD': {'X': ['EDUCATION'], 'F': 'ECON1MOD_LLM', 'Y': 'ECON1MOD', 'description': 'Economic conditions rating (1-4)'},
    'UNITY': {'X': ['EDUCATION'], 'F': 'UNITY_LLM', 'Y': 'UNITY', 'description': 'US unity perception (1-2)'},
    'GPT1': {'X': ['EDUCATION'], 'F': 'GPT1_LLM', 'Y': 'GPT1', 'description': 'ChatGPT familiarity (1-3)'},
    'MOREGUNIMPACT': {'X': ['EDUCATION'], 'F': 'MOREGUNIMPACT_LLM', 'Y': 'MOREGUNIMPACT', 'description': 'Gun control impact (1-3)'},
    'GAMBLERESTR': {'X': ['EDUCATION'], 'F': 'GAMBLERESTR_LLM', 'Y': 'GAMBLERESTR', 'description': 'Gambling restriction opinion (1-3)'}
}


def run_once_and_get_ci_length(estimator, df_in: pd.DataFrame, n_exp: int, n_labels: int,
                               a: float, sd: Optional[int], use_cc: bool, max_w: Optional[int]) -> Tuple[float, Dict]:
    """Run experiments once and return (avg_ci_length, results_dict)."""
    kwargs = dict(n_experiments=n_exp, n_true_labels=n_labels, alpha=a, seed=sd, use_concurrent=use_cc)
    try:
        res = estimator.run_experiments(df_in, **kwargs, max_workers=max_w)
    except TypeError:
        # Fallback for estimators without max_workers
        res = estimator.run_experiments(df_in, **kwargs)
    return float(res['avg_ci_length']), res


def run_once_and_get_metrics(estimator, df_in: pd.DataFrame, n_exp: int, n_labels: int,
                             a: float, sd: Optional[int], use_cc: bool, max_w: Optional[int]) -> Dict:
    """Run experiments once and return metrics dict."""
    kwargs = dict(n_experiments=n_exp, n_true_labels=n_labels, alpha=a, seed=sd, use_concurrent=use_cc)
    try:
        res = estimator.run_experiments(df_in, **kwargs, max_workers=max_w)
    except TypeError:
        res = estimator.run_experiments(df_in, **kwargs)
    # Return shallow metrics to keep JSON light
    return {
        'avg_ci_length': float(res.get('avg_ci_length')),
        'mse': float(res.get('mse')),
        'coverage_rate': float(res.get('coverage_rate')),
        'variance': float(res.get('variance')),
        'parameters': res.get('parameters', {})
    }


def _align_up(n: int, s: int) -> int:
    return int(((n + s - 1) // s) * s)


def _align_range(lo: int, hi: int, s: int) -> Tuple[int, int]:
    alo = _align_up(lo, s)
    ahi = (hi // s) * s if hi >= s else hi
    if ahi < alo:
        ahi = alo
    return alo, ahi


def search_min_labels_for_ci(factory, df_in: pd.DataFrame, target_ci: float,
                             a: float, sd: Optional[int], n_exp: int,
                             labels_lo: int, labels_hi: int, step: int,
                             use_cc: bool, max_w: Optional[int]) -> Dict:
    """Exponential grow + binary search to hit target CI width with minimal labels."""
    audit: List[Dict] = []
    labels_lo, labels_hi = _align_range(labels_lo, labels_hi, step)
    low = labels_lo
    high = labels_lo
    best = None

    # Grow
    while True:
        est = factory()
        ci_len, res = run_once_and_get_ci_length(est, df_in, n_exp, high, a, sd, use_cc, max_w)
        audit.append({'labels': int(high), 'avg_ci_length': float(ci_len)})
        if ci_len <= target_ci:
            best = (high, ci_len, res)
            break
        if high >= labels_hi:
            break
        low = high
        high = min(_align_up(high * 2, step), labels_hi)

    if best is None:
        return {
            'achieved': False,
            'required_labels': None,
            'avg_ci_length': audit[-1]['avg_ci_length'] if audit else None,
            'audit': audit
        }

    # Binary search
    lo = min(_align_up(low + 1, step), labels_hi)
    hi = best[0]
    req_labels, req_ci, req_res = hi, best[1], best[2]
    while lo <= hi:
        mid_raw = (lo + hi) // 2
        mid = _align_up(mid_raw, step)
        if mid > hi:
            mid = hi
        est = factory()
        ci_len, res = run_once_and_get_ci_length(est, df_in, n_exp, mid, a, sd, use_cc, max_w)
        audit.append({'labels': int(mid), 'avg_ci_length': float(ci_len)})
        if ci_len <= target_ci:
            req_labels, req_ci, req_res = mid, ci_len, res
            hi = mid - step
        else:
            lo = mid + step

    return {
        'achieved': True,
        'required_labels': int(req_labels),
        'avg_ci_length': float(req_ci),
        'results_snapshot': {
            'mse': float(req_res['mse']),
            'coverage_rate': float(req_res['coverage_rate']),
            'variance': float(req_res['variance']),
            'parameters': req_res.get('parameters', {})
        },
        'audit': sorted(audit, key=lambda x: x['labels'])
    }


def tune_ci_for_methods(df: pd.DataFrame, target: str,
                        methods: List[str],
                        ci_width: float, alpha: float, seed: Optional[int], n_experiments: int,
                        gamma_grid: List[float],
                        batch_sizes: List[int], design_update_freqs: List[int], warmup_batches_list: List[int],
                        min_labels: int, max_labels: int, label_step: int,
                        use_concurrent: bool, max_workers: Optional[int],
                        enable_pgae_internal: bool = False,
                        pgae_internal_presets: Optional[List[Dict]] = None,
                        mode: str = 'fixed-labels', fixed_labels: int = 1000) -> Dict:
    cfg = TARGET_CONFIGS[target]
    X, F, Y = cfg['X'], cfg['F'], cfg['Y']

    results: Dict[str, Dict] = {}
    for method in methods:
        logger.info(f"方法: {method}")
        best: Optional[Dict] = None
        trials: List[Dict] = []

        def is_better(candidate: Dict, incumbent: Optional[Dict]) -> bool:
            if incumbent is None:
                return True
            if mode == 'min-labels':
                # 优先最小标签数，其次更小的 avg_CI，再次更低 MSE
                cr = candidate.get('required_labels')
                ir = incumbent.get('required_labels')
                if cr != ir:
                    return cr < ir
                ca = candidate.get('avg_ci_length', float('inf'))
                ia = incumbent.get('avg_ci_length', float('inf'))
                if abs(ca - ia) > 1e-9:
                    return ca < ia
                cm = candidate.get('results_snapshot', {}).get('mse', float('inf'))
                im = incumbent.get('results_snapshot', {}).get('mse', float('inf'))
                return cm < im
            else:
                # fixed-labels: 优先更小 avg_CI，其次更低 MSE
                ca = candidate.get('avg_ci_length', float('inf'))
                ia = incumbent.get('avg_ci_length', float('inf'))
                if abs(ca - ia) > 1e-9:
                    return ca < ia
                cm = candidate.get('results_snapshot', {}).get('mse', float('inf'))
                im = incumbent.get('results_snapshot', {}).get('mse', float('inf'))
                return cm < im

        for g in gamma_grid:
            if method == 'PGAE':
                if enable_pgae_internal and pgae_internal_presets:
                    # 使用结构化参数传递，支持并发
                    for idx, preset in enumerate(pgae_internal_presets):
                        logger.info(f"PGAE preset[{idx}]: est_ci={preset.get('est_ci_params', {})}, design={preset.get('design_params', {})}")
                        design_params = preset.get('design_params', {})
                        est_ci_params = preset.get('est_ci_params', {})
                        if mode == 'min-labels':
                            def factory(gg=g, dp=design_params, cp=est_ci_params):
                                return PGAEEstimator(X=X, F=F, Y=Y, gamma=gg, design_params=dp, est_ci_params=cp)
                            out = search_min_labels_for_ci(lambda: factory(), df, ci_width, alpha, seed, n_experiments,
                                                           min_labels, max_labels, label_step, use_concurrent, max_workers)
                            out.update({'gamma': g, 'method': method, 'internal_preset': preset})
                        else:
                            metrics = run_once_and_get_metrics(
                                PGAEEstimator(X=X, F=F, Y=Y, gamma=g, design_params=design_params, est_ci_params=est_ci_params),
                                df, n_experiments, fixed_labels, alpha, seed, use_concurrent, max_workers
                            )
                            out = {
                                'achieved': True,
                                'required_labels': fixed_labels,
                                'avg_ci_length': metrics['avg_ci_length'],
                                'results_snapshot': metrics,
                                'gamma': g,
                                'method': method,
                                'internal_preset': preset
                            }
                        trials.append(out)
                        if is_better(out, best):
                            best = out
                else:
                    if mode == 'min-labels':
                        def factory(gg=g):
                            return PGAEEstimator(X=X, F=F, Y=Y, gamma=gg)
                        out = search_min_labels_for_ci(lambda: factory(), df, ci_width, alpha, seed, n_experiments,
                                                       min_labels, max_labels, label_step, use_concurrent, max_workers)
                        out['gamma'] = g
                        out['method'] = method
                        trials.append(out)
                    else:
                        metrics = run_once_and_get_metrics(
                            PGAEEstimator(X=X, F=F, Y=Y, gamma=g), df, n_experiments, fixed_labels,
                            alpha, seed, use_concurrent, max_workers
                        )
                        out = {
                            'achieved': True,
                            'required_labels': fixed_labels,
                            'avg_ci_length': metrics['avg_ci_length'],
                            'results_snapshot': metrics,
                            'gamma': g,
                            'method': method
                        }
                    trials.append(out)
                    if is_better(out, best):
                        best = out

            elif method == 'Adaptive_PGAE':
                for bs in batch_sizes:
                    for uf in design_update_freqs:
                        for wb in warmup_batches_list:
                            if mode == 'min-labels':
                                def factory(gg=g, bsz=bs, upf=uf, wup=wb):
                                    return AdaptivePGAEEstimator(X=X, F=F, Y=Y, gamma=gg,
                                                                 batch_size=bsz, design_update_freq=upf, warmup_batches=wup)
                                out = search_min_labels_for_ci(lambda: factory(), df, ci_width, alpha, seed, n_experiments,
                                                               min_labels, max_labels, label_step, use_concurrent, max_workers)
                                out.update({'gamma': g, 'batch_size': bs, 'design_update_freq': uf, 'warmup_batches': wb, 'method': method})
                            else:
                                metrics = run_once_and_get_metrics(
                                    AdaptivePGAEEstimator(X=X, F=F, Y=Y, gamma=g,
                                                         batch_size=bs, design_update_freq=uf, warmup_batches=wb),
                                    df, n_experiments, fixed_labels, alpha, seed, use_concurrent, max_workers
                                )
                                out = {
                                    'achieved': True,
                                    'required_labels': fixed_labels,
                                    'avg_ci_length': metrics['avg_ci_length'],
                                    'results_snapshot': metrics,
                                    'gamma': g,
                                    'batch_size': bs,
                                    'design_update_freq': uf,
                                    'warmup_batches': wb,
                                    'method': method
                                }
                            trials.append(out)
                            if is_better(out, best):
                                best = out
            else:
                logger.warning(f"不支持的方法: {method}，跳过")

        if best is None:
            results[method] = {'achieved': False}
        else:
            # 排序 trials 便于后续打印
            if mode == 'min-labels':
                trials_sorted = sorted(trials, key=lambda r: (r.get('required_labels', 10**9), r.get('avg_ci_length', 1e9), r.get('results_snapshot', {}).get('mse', 1e9)))
            else:
                trials_sorted = sorted(trials, key=lambda r: (r.get('avg_ci_length', 1e9), r.get('results_snapshot', {}).get('mse', 1e9)))
            best['trials'] = trials_sorted
            results[method] = best

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CI模式参数调优：PGAE 与 Adaptive PGAE')
    parser.add_argument('data_file', help='数据文件路径')
    parser.add_argument('--target', '-t', default='ECON1MOD', choices=list(TARGET_CONFIGS.keys()), help='预测目标 (默认: ECON1MOD)')
    parser.add_argument('--ci-width', type=float, default=0.1, help='目标CI宽度（默认：0.1；仅在 min-labels 模式下使用）')
    parser.add_argument('--experiments', '-e', type=int, default=100, help='每次评估的实验次数 (默认: 100)')
    parser.add_argument('--alpha', '-a', type=float, default=0.95, help='置信水平 (默认: 0.95)')
    parser.add_argument('--seed', '-s', type=int, default=42, help='随机种子 (默认: 42)')

    parser.add_argument('--methods', nargs='*', default=['PGAE', 'Adaptive_PGAE'], help='参与方法（默认：PGAE Adaptive_PGAE）')
    # 固定 gamma=0.5 为默认（可通过CLI覆盖）
    parser.add_argument('--gamma-grid', nargs='*', type=float, default=[0.5], help='gamma取值（默认：0.5，固定）')

    # Adaptive 参数网格（默认沿用你在MSE模式的经验：250/1/2，同时加入100作为对比）
    parser.add_argument('--batch-sizes', nargs='*', type=int, default=[100, 250], help='Adaptive: 批大小列表（默认：100 250）')
    parser.add_argument('--design-update-freqs', nargs='*', type=int, default=[1], help='Adaptive: 设计更新频率列表（默认：1）')
    parser.add_argument('--warmup-batches-list', nargs='*', type=int, default=[0, 2], help='Adaptive: 预热批次数列表（默认：0 2）')

    parser.add_argument('--min-labels', type=int, default=100, help='搜索的最小标签数 (默认: 100)')
    parser.add_argument('--max-labels', type=int, default=2500, help='搜索的最大标签数 (默认: 2500)')
    parser.add_argument('--label-step', type=int, default=100, help='标签搜索步长 (默认: 100)')
    parser.add_argument('--max-workers', type=int, default=10, help='并发worker数量上限（默认10）')
    # 默认开启并发；如需关闭使用 --no-concurrent
    parser.add_argument('--no-concurrent', dest='concurrent', action='store_false', help='禁用并发执行（默认开启）')
    # 调参模式：min-labels（找到达到CI目标的最小标签数）或 fixed-labels（固定标签数比较CI宽度）
    parser.add_argument('--mode', choices=['min-labels', 'fixed-labels'], default='fixed-labels', help='调参模式（默认：fixed-labels）')
    parser.add_argument('--labels', type=int, default=1000, help='固定标签数（默认：1000，仅在 fixed-labels 模式下使用）')

    # PGAE 内部参数调优（参考 tune_internal_parameters）
    parser.add_argument('--enable-pgae-internal', action='store_true', help='对PGAE启用内部参数调优（RF与设计稳健参数）')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出文件名前缀')
    parser.add_argument('--report-all', action='store_true', help='同时打印所有测试组合（按指标排序）')

    args = parser.parse_args()

    logger.info(f"加载数据: {args.data_file}")
    df = pd.read_csv(args.data_file)

    logger.info(f"目标: {args.target} | 目标CI宽度: {args.ci_width} | alpha: {args.alpha}")

    # 预设的内部参数配置（与 tune_internal_parameters 保持一致）
    pgae_presets = [
        {'est_ci_params': {'n_estimators_mu': 20, 'n_estimators_tau': 100, 'K': 2},
         'design_params': {'min_var_threshold': 1e-6, 'prob_clip_min': 0.01, 'prob_clip_max': 0.99}},
        {'est_ci_params': {'n_estimators_mu': 50, 'n_estimators_tau': 150, 'K': 3},
         'design_params': {'min_var_threshold': 1e-4, 'prob_clip_min': 0.05, 'prob_clip_max': 0.95}},
        {'est_ci_params': {'n_estimators_mu': 100, 'n_estimators_tau': 200, 'K': 5, 'max_depth': 10},
         'design_params': {'min_var_threshold': 1e-4, 'prob_clip_min': 0.1, 'prob_clip_max': 0.9}},
        {'est_ci_params': {'n_estimators_mu': 80, 'n_estimators_tau': 150, 'K': 4, 'max_depth': 8, 'min_samples_split': 10},
         'design_params': {'min_var_threshold': 1e-3, 'prob_clip_min': 0.1, 'prob_clip_max': 0.8}},
    ]

    results = tune_ci_for_methods(
        df=df,
        target=args.target,
        methods=args.methods,
        ci_width=args.ci_width,
        alpha=args.alpha,
        seed=args.seed,
        n_experiments=args.experiments,
        gamma_grid=args.gamma_grid,
        batch_sizes=args.batch_sizes,
        design_update_freqs=args.design_update_freqs,
        warmup_batches_list=args.warmup_batches_list,
        min_labels=args.min_labels,
        max_labels=args.max_labels,
        label_step=args.label_step,
        use_concurrent=args.concurrent,
        max_workers=args.max_workers,
        enable_pgae_internal=args.enable_pgae_internal,
        pgae_internal_presets=pgae_presets,
        mode=args.mode,
        fixed_labels=args.labels
    )

    # 打印总结
    print("\n" + "=" * 90)
    if args.mode == 'min-labels':
        print("CI WIDTH PARAMETER TUNING (min labels to hit target CI)")
    else:
        print(f"CI WIDTH PARAMETER TUNING (fixed labels = {args.labels})")
    print("=" * 90)
    for method, res in results.items():
        if not res or not res.get('achieved'):
            print(f"{method:<18} -> 未达到目标CI宽度 (<= {args.ci_width})，在 labels <= {args.max_labels} 范围内")
        else:
            snap = res.get('results_snapshot', {})
            mse_text = f"MSE={snap['mse']:.6f}" if 'mse' in snap else ""
            extra = f" | {mse_text}" if mse_text else ""
            if args.mode == 'min-labels':
                if method == 'Adaptive_PGAE':
                    print(f"{method:<18} -> 需要标签: {res['required_labels']}, gamma={res.get('gamma')}, bs={res.get('batch_size')}, uf={res.get('design_update_freq')}, wb={res.get('warmup_batches')} | avg_CI={res['avg_ci_length']:.4f}{extra}")
                else:
                    print(f"{method:<18} -> 需要标签: {res['required_labels']}, gamma={res.get('gamma')} | avg_CI={res['avg_ci_length']:.4f}{extra}")
            else:  # fixed-labels
                if method == 'Adaptive_PGAE':
                    print(f"{method:<18} -> labels={res['required_labels']}, gamma={res.get('gamma')}, bs={res.get('batch_size')}, uf={res.get('design_update_freq')}, wb={res.get('warmup_batches')} | avg_CI={res['avg_ci_length']:.4f}{extra}")
                else:
                    print(f"{method:<18} -> labels={res['required_labels']}, gamma={res.get('gamma')} | avg_CI={res['avg_ci_length']:.4f}{extra}")
    print("=" * 90)

    # 可选打印所有组合
    if args.report_all:
        for method, res in results.items():
            trials = res.get('trials', [])
            if not trials:
                continue
            print(f"\n-- All trials for {method} --")
            for t in trials:
                snap = t.get('results_snapshot', {})
                mse_text = f"MSE={snap.get('mse', float('nan')):.6f}" if 'mse' in snap else ""
                extra = f" | {mse_text}" if mse_text else ""
                if method == 'Adaptive_PGAE':
                    print(f"  gamma={t.get('gamma')}, bs={t.get('batch_size')}, uf={t.get('design_update_freq')}, wb={t.get('warmup_batches')} | avg_CI={t.get('avg_ci_length', float('nan')):.4f}{extra}")
                else:
                    preset = t.get('internal_preset')
                    if preset:
                        tag = f"preset={preset}"
                    else:
                        tag = ""
                    print(f"  gamma={t.get('gamma')} {tag} | avg_CI={t.get('avg_ci_length', float('nan')):.4f}{extra}")

    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_name = (args.output or 'ci_mode_tuning') + f"_{args.target.lower()}_{timestamp}.json"

    # 转换numpy为原生类型，保证可序列化
    def convert_numpy(obj, depth=0, max_depth=50):
        if depth > max_depth:
            return str(obj)  # Convert to string if too deep
            
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert_numpy(v, depth+1, max_depth) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_numpy(v, depth+1, max_depth) for v in obj]
        # Handle other numpy types by converting to Python native types
        if hasattr(obj, 'item'):
            try:
                return obj.item()
            except:
                return str(obj)
        return obj

    with open(out_name, 'w') as f:
        payload = {'target': args.target, 'mode': args.mode, 'alpha': args.alpha,
                   'experiments': args.experiments, 'results': results}
        if args.mode == 'min-labels':
            payload['ci_width'] = args.ci_width
        else:
            payload['labels'] = args.labels
        json.dump(convert_numpy(payload), f, indent=2)
    logger.info(f"CI模式参数调优结果已保存: {out_name}")


if __name__ == "__main__":
    main()
