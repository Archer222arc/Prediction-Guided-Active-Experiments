#!/usr/bin/env python3
"""
Twin Dataset Estimator Comparison Runner

Purpose: Run MSE-style comparisons on the Tianyi digital twin dataset without
reusing the NPORS compare script. Uses EDUCATION (or encoded variant) as X and
multiple Likert questions as targets Y.

Outputs two CSVs by default:
- Summary per-method metrics per question
- Per-repeat detailed runs (estimate, ci_lower, ci_higher, etc.)
"""

import argparse
import time
import re
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from pgae_estimator import PGAEEstimator
from adaptive_pgae_estimator import AdaptivePGAEEstimator
from active_inference_estimator import ActiveInferenceEstimator
from naive_estimator import NaiveEstimator

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 使用NPORS标准列名
DEFAULT_QUESTIONS = [
    "CARBONTAX",
    "CLEANENERGY",
    "CLEANELEC",
    "MEDICAREALL",
    "PUBLICOPTION",
    "IMMIGRATION",
    "FAMILYLEAVE",
    "WEALTHTAX",
    "DEPORTATIONS",
    "MEDICVOUCHER",
]


def extract_first_1to5(value):
    if pd.isna(value):
        return np.nan
    s = str(value)
    m = re.search(r"\b([1-5])\b", s)
    return int(m.group(1)) if m else np.nan


def encode_small_category(series: pd.Series, max_levels: int = 9) -> pd.Series:
    if series.dtype == 'O' or str(series.dtype).startswith('category'):
        cats = series.astype('category')
        if len(cats.cat.categories) > (max_levels + 1):
            top = series.value_counts().nlargest(max_levels).index
            series = series.where(series.isin(top), 'Other')
            cats = series.astype('category')
        codes = cats.cat.codes  # 0..k-1
        return codes
    # numeric: ensure non-negative small ints if possible
    s = pd.to_numeric(series, errors='coerce')
    return s


def resolve_x_column(df: pd.DataFrame, x_col: str = 'EDUCATION') -> (pd.DataFrame, List[str], str):
    if x_col in df.columns:
        col = x_col
    else:
        # try case-insensitive
        lower_map = {c.lower(): c for c in df.columns}
        if x_col.lower() in lower_map:
            col = lower_map[x_col.lower()]
        else:
            raise ValueError(f"X column '{x_col}' not found in data.")
    enc_col = f"{col}_enc"
    dfx = df.copy()
    dfx[enc_col] = encode_small_category(dfx[col])
    return dfx, [enc_col], enc_col


def find_column(df: pd.DataFrame, base: str, preferred_suffixes: List[str]) -> Optional[str]:
    # try base+suffix order; also accept exact base
    for suf in preferred_suffixes:
        if suf == '':
            name = base  # exact match
        else:
            name = f"{base}{suf}"
        if name in df.columns:
            return name
    if base in df.columns:
        return base
    return None


def ensure_numeric_scale(df: pd.DataFrame, col: str) -> str:
    """Ensure a numeric 1..5 scale column exists; create <col>_num if needed."""
    if col is None:
        return None
    if pd.api.types.is_numeric_dtype(df[col]):
        return col
    num_col = f"{col}_num"
    if num_col not in df.columns:
        df[num_col] = df[col].apply(extract_first_1to5)
    return num_col


def run_for_question(df: pd.DataFrame, qid: str, X: List[str], F: str, Y: str,
                     n_exp: int, n_labels: int, alpha: float, gamma: float,
                     seed: Optional[int], use_concurrent: bool, max_workers: int) -> Dict:
    required = X + [F, Y]
    d = df.dropna(subset=required).copy()
    # Filter ranges per estimator expectations
    for x in X:
        d = d[(d[x] >= 0) & (d[x] < 10)]
    d = d[(d[F] >= 1) & (d[F] <= 5) & (d[Y] >= 1) & (d[Y] <= 5)]
    if len(d) == 0:
        raise ValueError(f"No valid rows after filtering for question {qid}")

    ests = {
        'PGAE': PGAEEstimator(X=X, F=F, Y=Y, gamma=gamma),
        'Adaptive_PGAE': AdaptivePGAEEstimator(X=X, F=F, Y=Y, gamma=gamma, batch_size=100, design_update_freq=1, warmup_batches=1),
        'Active_Inference': ActiveInferenceEstimator(X=X, F=F, Y=Y, gamma=gamma),
        'Naive': NaiveEstimator(X=X, F=F, Y=Y, gamma=gamma),
    }
    out = {}
    for name, est in ests.items():
        res = est.run_experiments(
            d,
            n_experiments=n_exp,
            n_true_labels=n_labels,
            alpha=alpha,
            seed=seed,
            use_concurrent=use_concurrent,
            max_workers=max_workers,
        )
        out[name] = res
    return out


def run_once_and_get_ci_length(estimator, df: pd.DataFrame, n_experiments: int, n_labels: int, alpha: float, seed: Optional[int], use_concurrent: bool, max_workers: int) -> (float, Dict):
    """运行一次实验并返回平均CI长度"""
    results = estimator.run_experiments(
        df,
        n_experiments=n_experiments,
        n_true_labels=n_labels,
        alpha=alpha,
        seed=seed,
        use_concurrent=use_concurrent,
        max_workers=max_workers
    )
    return results['avg_ci_length'], results


def search_min_labels_for_ci(estimator_factory, df: pd.DataFrame, target_ci: float, alpha: float, seed: Optional[int], n_experiments: int, labels_lo: int, labels_hi: int, step: int, use_concurrent: bool, max_workers: int, tolerance: float) -> Dict:
    """二分搜索找到达到目标CI宽度的最小标签数"""

    def align_up(n: int, s: int) -> int:
        return int(((n + s - 1) // s) * s)

    def align_range(lo: int, hi: int, s: int) -> (int, int):
        alo = align_up(lo, s)
        ahi = (hi // s) * s if hi >= s else hi
        if ahi < alo:
            ahi = alo
        return alo, ahi

    labels_lo, labels_hi = align_range(labels_lo, labels_hi, step)
    audit = []

    # 指数搜索找上界
    low = labels_lo
    high = labels_lo
    best = None
    best_result_at_max = None  # 记录在最大labels时的最佳结果

    while high <= labels_hi:
        estimator = estimator_factory()
        ci_len, res = run_once_and_get_ci_length(estimator, df, n_experiments, high, alpha, seed, use_concurrent, max_workers)
        audit.append({'labels': int(high), 'avg_ci_length': float(ci_len)})

        # 总是记录最新的结果（用于在max_labels时保存最佳尝试）
        best_result_at_max = (high, ci_len, res)

        if ci_len <= target_ci + tolerance:
            best = (high, ci_len, res)
            break

        # 防止死循环：如果已经达到最大值，退出
        if high >= labels_hi:
            break

        low = high
        next_high = min(align_up(high * 2, step), labels_hi)

        # 再次检查：如果下一个值和当前值相同，说明已经到达上限
        if next_high == high:
            break

        high = next_high

    # 如果没找到满足条件的解，返回在最大labels时的结果
    if best is None:
        if best_result_at_max:
            return {
                'achieved': False,
                'required_labels': int(best_result_at_max[0]),  # 使用最大测试过的labels
                'avg_ci_length': float(best_result_at_max[1]),
                'results_snapshot': {
                    'mse': float(best_result_at_max[2]['mse']),
                    'coverage_rate': float(best_result_at_max[2]['coverage_rate']),
                    'variance': float(best_result_at_max[2]['variance']),
                    'execution_time': float(best_result_at_max[2]['execution_time'])
                },
                'audit': audit,
                'note': f'Target CI {target_ci} not achieved within max_labels {labels_hi}. Best result at {best_result_at_max[0]} labels: CI={best_result_at_max[1]:.4f}'
            }
        else:
            return {'achieved': False, 'required_labels': None, 'avg_ci_length': None, 'audit': audit}

    # 二分搜索精确解
    lo = min(align_up(low + 1, step), labels_hi)
    hi = best[0]
    req_labels, req_ci, req_res = hi, best[1], best[2]

    while lo <= hi:
        mid_raw = (lo + hi) // 2
        mid = align_up(mid_raw, step)
        if mid > hi:
            mid = hi

        estimator = estimator_factory()
        ci_len, res = run_once_and_get_ci_length(estimator, df, n_experiments, mid, alpha, seed, use_concurrent, max_workers)
        audit.append({'labels': int(mid), 'avg_ci_length': float(ci_len)})

        if ci_len <= target_ci + tolerance:
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
            'execution_time': float(req_res['execution_time'])
        },
        'audit': audit
    }


def run_ci_width_mode(df: pd.DataFrame, qid: str, X: List[str], F: str, Y: str, target_ci: float, alpha: float, gamma: float, seed: Optional[int], n_experiments: int, min_labels: int, max_labels: int, label_step: int, use_concurrent: bool, max_workers: int, tolerance: float) -> Dict:
    """运行CI宽度模式，寻找最小标签数成本"""

    required = X + [F, Y]
    d = df.dropna(subset=required).copy()

    # Filter ranges per estimator expectations
    for x in X:
        d = d[(d[x] >= 0) & (d[x] < 10)]
    d = d[(d[F] >= 1) & (d[F] <= 5) & (d[Y] >= 1) & (d[Y] <= 5)]

    if len(d) == 0:
        raise ValueError(f"No valid rows after filtering for question {qid}")

    # 定义估计器工厂函数
    factories = {
        'PGAE': lambda: PGAEEstimator(X=X, F=F, Y=Y, gamma=gamma),
        'Adaptive_PGAE': lambda: AdaptivePGAEEstimator(X=X, F=F, Y=Y, gamma=gamma, batch_size=100, design_update_freq=1, warmup_batches=1),
        'Active_Inference': lambda: ActiveInferenceEstimator(X=X, F=F, Y=Y, gamma=gamma),
        'Naive': lambda: NaiveEstimator(X=X, F=F, Y=Y, gamma=gamma),
    }

    results = {}
    for method_name, factory in factories.items():
        logger.info(f"Searching minimum labels for {method_name} to achieve CI width <= {target_ci}")
        try:
            result = search_min_labels_for_ci(
                factory, d, target_ci, alpha, seed, n_experiments,
                min_labels, max_labels, label_step, use_concurrent, max_workers, tolerance
            )
            results[method_name] = result
        except Exception as e:
            logger.error(f"{method_name} failed: {e}")
            results[method_name] = {'achieved': False, 'error': str(e)}

    return results


def print_ci_width_summary(results: Dict, questions: List[str], target_ci: float, tolerance: float, data_file: str, timestamp: str):
    """打印CI宽度模式的总结"""

    print("\n" + "=" * 100)
    print("DIGITAL TWIN DATASET CI WIDTH COST COMPARISON")
    print("=" * 100)
    print(f"Dataset: {data_file}")
    print(f"Timestamp: {timestamp}")
    print(f"Target CI Width: {target_ci} (±{tolerance})")
    print(f"Questions analyzed: {len(questions)}")
    print("=" * 100)

    for qid in questions:
        if qid not in results:
            continue

        q_results = results[qid]
        print(f"\n📊 {qid.upper()} - CI WIDTH COST ANALYSIS:")
        print("-" * 80)

        for method, res in q_results.items():
            if not res:
                print(f"{method:<18} -> 执行失败")
            elif not res.get('achieved'):
                # 检查是否有在max_labels时的结果
                if res.get('required_labels') is not None and res.get('avg_ci_length') is not None:
                    req_labels = res['required_labels']
                    avg_ci = res['avg_ci_length']
                    snap = res.get('results_snapshot', {})
                    mse_text = f"MSE={snap.get('mse', 0):.6f}" if 'mse' in snap else ""
                    time_text = f"Time={snap.get('execution_time', 0):.1f}s" if 'execution_time' in snap else ""
                    extra = f" | {mse_text} | {time_text}" if mse_text or time_text else ""
                    print(f"{method:<18} -> 未达标(目标≤{target_ci:.3f}): 最大{req_labels}标签时 CI={avg_ci:.4f}{extra}")
                else:
                    print(f"{method:<18} -> 未达到目标CI宽度 (<= {target_ci}+{tolerance})，在最大标签范围内")
            else:
                req_labels = res['required_labels']
                avg_ci = res['avg_ci_length']
                snap = res.get('results_snapshot', {})
                mse_text = f"MSE={snap.get('mse', 0):.6f}" if 'mse' in snap else ""
                time_text = f"Time={snap.get('execution_time', 0):.1f}s" if 'execution_time' in snap else ""
                extra = f" | {mse_text} | {time_text}" if mse_text or time_text else ""
                print(f"{method:<18} -> ✅ 需要标签: {req_labels}, avg_CI={avg_ci:.4f} (±{tolerance}){extra}")

    # 全局总结
    print("\n" + "=" * 100)
    print("OVERALL CI COST SUMMARY")
    print("=" * 100)

    # 计算每个方法的成本和表现
    method_costs = {}        # 成功达标的成本
    method_attempts = {}     # 所有尝试（包括未达标的最大尝试）

    for qid in questions:
        if qid not in results:
            continue
        for method, res in results[qid].items():
            if res and res.get('required_labels') is not None:
                # 记录所有有效尝试
                if method not in method_attempts:
                    method_attempts[method] = {'labels': [], 'achieved': [], 'ci_lengths': []}

                method_attempts[method]['labels'].append(res['required_labels'])
                method_attempts[method]['achieved'].append(res.get('achieved', False))
                method_attempts[method]['ci_lengths'].append(res.get('avg_ci_length', float('inf')))

                # 只记录成功达标的成本
                if res.get('achieved'):
                    if method not in method_costs:
                        method_costs[method] = []
                    method_costs[method].append(res['required_labels'])

    if method_attempts:
        print(f"\n📊 Method Performance Summary (across {len(questions)} questions):")
        print("-" * 80)

        for method, data in method_attempts.items():
            total_attempts = len(data['labels'])
            successful = sum(data['achieved'])
            success_rate = (successful / total_attempts) * 100 if total_attempts > 0 else 0

            if successful > 0:
                # 成功案例的平均成本
                successful_costs = [data['labels'][i] for i in range(len(data['labels'])) if data['achieved'][i]]
                avg_successful_cost = np.mean(successful_costs)
                std_successful_cost = np.std(successful_costs) if len(successful_costs) > 1 else 0
                cost_text = f"Successful avg: {avg_successful_cost:.0f} ± {std_successful_cost:.0f} labels"
            else:
                cost_text = "No successful cases"

            # 所有尝试的平均CI长度
            avg_ci = np.mean([ci for ci in data['ci_lengths'] if ci != float('inf')])

            print(f"{method:<18} | Success: {successful}/{total_attempts} ({success_rate:.0f}%) | {cost_text} | Avg CI: {avg_ci:.4f}")

        # 最优方法（基于成功率和成本）
        if method_costs:
            best_by_cost = min(method_costs.items(), key=lambda x: np.mean(x[1]))
            print(f"\n🏆 Most Cost-Effective (among successful): {best_by_cost[0]} (avg {np.mean(best_by_cost[1]):.0f} labels)")

        best_by_success = max(method_attempts.items(), key=lambda x: sum(x[1]['achieved']))
        success_count = sum(best_by_success[1]['achieved'])
        print(f"🎯 Highest Success Rate: {best_by_success[0]} ({success_count}/{len(questions)} successful)")
    else:
        print("\n⚠️  No valid results to summarize")

    print("=" * 100)


def save_ci_results_to_csv(results: Dict, questions: List[str], target_ci: float, tolerance: float, data_file: str, timestamp: str, args):
    """保存CI宽度分析结果到CSV文件"""

    ci_summary_rows = []
    ci_audit_rows = []

    for qid in questions:
        if qid not in results:
            continue

        q_results = results[qid]
        for method, res in q_results.items():
            if not res:
                continue

            # 汇总结果行
            summary_row = {
                'timestamp': timestamp,
                'dataset': 'twin',
                'file': data_file,
                'question': qid,
                'method': method,
                'target_ci_width': target_ci,
                'ci_tolerance': tolerance,
                'achieved': res.get('achieved', False),
                'required_labels': res.get('required_labels'),
                'actual_ci_width': res.get('avg_ci_length'),
                'alpha': args.alpha,
                'experiments': args.experiments,
                'min_labels': args.min_labels,
                'max_labels': args.max_labels,
                'label_step': args.label_step,
                'gamma': args.gamma
            }

            # 添加性能快照
            snap = res.get('results_snapshot', {})
            summary_row.update({
                'mse': snap.get('mse'),
                'coverage_rate': snap.get('coverage_rate'),
                'variance': snap.get('variance'),
                'execution_time': snap.get('execution_time')
            })

            # 添加备注
            if 'note' in res:
                summary_row['note'] = res['note']

            ci_summary_rows.append(summary_row)

            # 详细搜索轨迹行
            audit = res.get('audit', [])
            for i, audit_entry in enumerate(audit):
                audit_row = {
                    'timestamp': timestamp,
                    'dataset': 'twin',
                    'file': data_file,
                    'question': qid,
                    'method': method,
                    'target_ci_width': target_ci,
                    'search_step': i + 1,
                    'labels_tried': audit_entry.get('labels'),
                    'ci_length_achieved': audit_entry.get('avg_ci_length'),
                    'achieved_target': audit_entry.get('avg_ci_length', float('inf')) <= target_ci + tolerance,
                    'alpha': args.alpha,
                    'experiments': args.experiments
                }
                ci_audit_rows.append(audit_row)

    # 保存汇总CSV
    if ci_summary_rows:
        summary_df = pd.DataFrame(ci_summary_rows)
        summary_file = f"{timestamp}_twin_ci_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"CI汇总结果已保存: {summary_file} ({len(ci_summary_rows)} 行)")

    # 保存搜索轨迹CSV
    if ci_audit_rows:
        audit_df = pd.DataFrame(ci_audit_rows)
        audit_file = f"{timestamp}_twin_ci_audit.csv"
        audit_df.to_csv(audit_file, index=False)
        logger.info(f"CI搜索轨迹已保存: {audit_file} ({len(ci_audit_rows)} 行)")

    return summary_file if ci_summary_rows else None, audit_file if ci_audit_rows else None


def print_comparison_summary(results_rows: List[Dict], data_file: str, timestamp: str):
    """打印对比总结表格 - 参照compare_estimators风格"""

    df_results = pd.DataFrame(results_rows)

    # 按问题分组显示结果
    questions = df_results['question'].unique()

    print("\n" + "=" * 100)
    print("DIGITAL TWIN DATASET ESTIMATOR COMPARISON RESULTS")
    print("=" * 100)
    print(f"Dataset: {data_file}")
    print(f"Timestamp: {timestamp}")
    print(f"Questions analyzed: {len(questions)}")
    print(f"Methods compared: {', '.join(df_results['method'].unique())}")
    print("=" * 100)

    for question in questions:
        q_data = df_results[df_results['question'] == question].copy()

        print(f"\n📊 {question.upper()} RESULTS:")
        print("-" * 60)

        # 创建汇总表
        summary_table = []
        for _, row in q_data.iterrows():
            summary_table.append({
                'Method': row['method'],
                'MSE': f"{row['mse']:.6f}",
                'Coverage': f"{row['coverage_rate']:.3f}",
                'CI Length': f"{row['avg_ci_length']:.4f}",
                'Time (s)': f"{row['execution_time']:.1f}"
            })

        summary_df = pd.DataFrame(summary_table)
        print(summary_df.to_string(index=False))

        # 找出最佳方法
        best_mse_idx = q_data['mse'].idxmin()
        best_coverage_idx = q_data['coverage_rate'].idxmax()
        best_ci_idx = q_data['avg_ci_length'].idxmin()
        best_time_idx = q_data['execution_time'].idxmin()

        print(f"\n🏆 Best Performance:")
        print(f"   • Lowest MSE: {q_data.loc[best_mse_idx, 'method']} ({q_data.loc[best_mse_idx, 'mse']:.6f})")
        print(f"   • Best Coverage: {q_data.loc[best_coverage_idx, 'method']} ({q_data.loc[best_coverage_idx, 'coverage_rate']:.3f})")
        print(f"   • Shortest CI: {q_data.loc[best_ci_idx, 'method']} ({q_data.loc[best_ci_idx, 'avg_ci_length']:.4f})")
        print(f"   • Fastest: {q_data.loc[best_time_idx, 'method']} ({q_data.loc[best_time_idx, 'execution_time']:.1f}s)")

    # 全局总结
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)

    # 计算每个方法的平均性能
    method_summary = df_results.groupby('method').agg({
        'mse': 'mean',
        'coverage_rate': 'mean',
        'avg_ci_length': 'mean',
        'execution_time': 'mean'
    }).reset_index()

    print("\n📈 Average Performance Across All Questions:")
    print("-" * 50)
    for _, row in method_summary.iterrows():
        print(f"{row['method']:<18} | MSE: {row['mse']:.6f} | Coverage: {row['coverage_rate']:.3f} | CI: {row['avg_ci_length']:.4f} | Time: {row['execution_time']:.1f}s")

    # 方法排名
    print(f"\n🥇 Method Rankings:")
    print(f"   • Best MSE: {method_summary.loc[method_summary['mse'].idxmin(), 'method']}")
    print(f"   • Best Coverage: {method_summary.loc[method_summary['coverage_rate'].idxmax(), 'method']}")
    print(f"   • Shortest CI: {method_summary.loc[method_summary['avg_ci_length'].idxmin(), 'method']}")
    print(f"   • Fastest: {method_summary.loc[method_summary['execution_time'].idxmin(), 'method']}")

    print("=" * 100)

    # 实验参数总结
    sample_row = df_results.iloc[0]
    print(f"\n⚙️  Experiment Parameters:")
    print(f"   • Experiments per question: {sample_row['experiments']}")
    print(f"   • True labels per experiment: {sample_row['n_true_labels']}")
    print(f"   • Confidence level: {sample_row['alpha']}")
    print(f"   • Gamma parameter: {sample_row['gamma']}")
    print(f"   • X variables: {sample_row['X']}")

    total_experiments = len(questions) * sample_row['experiments'] * len(df_results['method'].unique())
    print(f"   • Total experiments run: {total_experiments}")

    print("\n" + "=" * 100)


def main():
    ap = argparse.ArgumentParser(description='Twin dataset estimator comparison (MSE-mode)')
    ap.add_argument('pred_csv', help='Predictions CSV path (must include RESPID and question prediction columns)')
    ap.add_argument('--x-col', default='EDUCATION', help='Column to use as X (default: EDUCATION)')
    ap.add_argument('--questions', nargs='*', default=DEFAULT_QUESTIONS, help='Question IDs to evaluate')
    ap.add_argument('--pred-suffixes', nargs='*', default=['_LLM'], help='Tried suffixes for prediction columns in order')
    ap.add_argument('--true-suffixes', nargs='*', default=[''], help='Tried suffixes for true columns in order (empty means exact match)')
    ap.add_argument('--experiments', type=int, default=50, help='Number of repeats (reduced for faster testing)')
    ap.add_argument('--labels', type=int, default=300, help='True labels per repeat (reduced for digital twin dataset)')
    ap.add_argument('--alpha', type=float, default=0.95, help='Confidence level')
    ap.add_argument('--gamma', type=float, default=0.5, help='Gamma parameter')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    ap.add_argument('--no-concurrent', action='store_true', help='Disable concurrency')
    ap.add_argument('--max-workers', type=int, default=10, help='Max concurrent workers')
    ap.add_argument('--summary-csv', default='twin_compare_mse.csv', help='Summary CSV output')
    ap.add_argument('--runs-csv', default='twin_compare_runs.csv', help='Per-repeat runs CSV output')

    # CI宽度功能
    ap.add_argument('--ci-width', type=float, default=None, help='Target CI width (e.g. 0.05). If provided, runs "minimum labels cost" mode')
    ap.add_argument('--ci-tolerance', type=float, default=0.005, help='CI width tolerance (default: 0.005)')
    ap.add_argument('--min-labels', type=int, default=50, help='Minimum labels to search (CI mode)')
    ap.add_argument('--max-labels', type=int, default=1000, help='Maximum labels to search (CI mode)')
    ap.add_argument('--label-step', type=int, default=50, help='Labels search step (CI mode)')

    args = ap.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    df = pd.read_csv(args.pred_csv)
    logger.info(f"Loaded predictions: {args.pred_csv} ({len(df)} rows)")
    logger.info(f"Available columns: {list(df.columns)}")

    # 检查RESPID列是否存在
    if 'RESPID' not in df.columns:
        logger.error("RESPID column not found. Please ensure the input CSV has RESPID column.")
        return

    # Resolve X
    df, X, used_x = resolve_x_column(df, args.x_col)
    logger.info(f"Using X={X}")

    use_concurrent = not args.no_concurrent

    # 检查是否运行CI宽度模式
    if args.ci_width is not None:
        logger.info(f"运行CI宽度模式: 目标CI={args.ci_width} (±{args.ci_tolerance})")

        ci_results = {}
        for q in args.questions:
            # Resolve F and Y columns
            F_col = find_column(df, q, args.pred_suffixes)
            Y_col = find_column(df, q, args.true_suffixes)

            if F_col is None:
                logger.warning(f"Skip {q}: prediction column not found. Available columns: {[c for c in df.columns if q in c]}")
                continue

            if Y_col is None:
                logger.warning(f"Skip {q}: true column not found. Available columns: {[c for c in df.columns if q in c]}")
                continue

            # 对于数字twin数据集，直接使用列名，不需要转换
            F_num = F_col
            Y_num = Y_col

            # 验证列是否为数值类型且在正确范围内
            if not pd.api.types.is_numeric_dtype(df[F_num]):
                logger.warning(f"Skip {q}: prediction column {F_num} is not numeric")
                continue
            if not pd.api.types.is_numeric_dtype(df[Y_num]):
                logger.warning(f"Skip {q}: true column {Y_num} is not numeric")
                continue

            logger.info(f"CI Width Analysis for {q}: F={F_num}, Y={Y_num}")
            try:
                q_results = run_ci_width_mode(
                    df, q, X, F_num, Y_num,
                    target_ci=args.ci_width,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    seed=args.seed,
                    n_experiments=args.experiments,
                    min_labels=args.min_labels,
                    max_labels=args.max_labels,
                    label_step=args.label_step,
                    use_concurrent=use_concurrent,
                    max_workers=args.max_workers,
                    tolerance=args.ci_tolerance
                )
                ci_results[q] = q_results
            except Exception as e:
                logger.error(f"CI Width Analysis for {q} failed: {e}")
                continue

        # 打印CI宽度总结
        print_ci_width_summary(ci_results, args.questions, args.ci_width, args.ci_tolerance, args.pred_csv, timestamp)

        # 保存CI模式的CSV结果
        summary_file, audit_file = save_ci_results_to_csv(ci_results, args.questions, args.ci_width, args.ci_tolerance, args.pred_csv, timestamp, args)

        # 显示输出文件信息
        print(f"\n📄 CI分析结果文件:")
        if summary_file:
            print(f"   • 汇总结果: {summary_file}")
        if audit_file:
            print(f"   • 搜索轨迹: {audit_file}")
        print("=" * 100)

        return

    # 正常MSE模式
    results_rows = []
    runs_rows = []

    for q in args.questions:
        # Resolve F and Y columns
        F_col = find_column(df, q, args.pred_suffixes)
        Y_col = find_column(df, q, args.true_suffixes)

        if F_col is None:
            logger.warning(f"Skip {q}: prediction column not found. Available columns: {[c for c in df.columns if q in c]}")
            continue

        if Y_col is None:
            logger.warning(f"Skip {q}: true column not found. Available columns: {[c for c in df.columns if q in c]}")
            continue

        # 对于数字twin数据集，直接使用列名，不需要转换
        F_num = F_col
        Y_num = Y_col

        # 验证列是否为数值类型且在正确范围内
        if not pd.api.types.is_numeric_dtype(df[F_num]):
            logger.warning(f"Skip {q}: prediction column {F_num} is not numeric")
            continue
        if not pd.api.types.is_numeric_dtype(df[Y_num]):
            logger.warning(f"Skip {q}: true column {Y_num} is not numeric")
            continue

        logger.info(f"Question {q}: F={F_num}, Y={Y_num}")
        try:
            res = run_for_question(
                df, q, X, F_num, Y_num,
                n_exp=args.experiments,
                n_labels=args.labels,
                alpha=args.alpha,
                gamma=args.gamma,
                seed=args.seed,
                use_concurrent=use_concurrent,
                max_workers=args.max_workers,
            )
        except Exception as e:
            logger.error(f"Question {q} failed: {e}")
            continue

        # Summary rows
        settings_key = (
            f"dataset=twin|file={args.pred_csv}|question={q}|x={used_x}|alpha={args.alpha}|"
            f"experiments={args.experiments}|labels={args.labels}|gamma={args.gamma}"
        )
        for method, r in res.items():
            results_rows.append({
                'timestamp': timestamp,
                'dataset': 'twin',
                'file': args.pred_csv,
                'question': q,
                'X': used_x,
                'alpha': args.alpha,
                'experiments': args.experiments,
                'n_true_labels': args.labels,
                'method': method,
                'gamma': args.gamma,
                'mse': r.get('mse'),
                'avg_ci_length': r.get('avg_ci_length'),
                'coverage_rate': r.get('coverage_rate'),
                'execution_time': r.get('execution_time'),
                'settings_key': settings_key,
            })

        # Per-run rows
        for method, r in res.items():
            if not all(k in r for k in ['tau_estimates', 'l_ci', 'h_ci']):
                continue
            taus = r['tau_estimates']
            lcis = r['l_ci']
            hcis = r['h_ci']
            true_val = r.get('true_value')
            for i in range(len(taus)):
                run_seed = (args.seed + i) if (args.seed is not None) else None
                runs_rows.append({
                    'timestamp': timestamp,
                    'dataset': 'twin',
                    'file': args.pred_csv,
                    'question': q,
                    'X': used_x,
                    'alpha': args.alpha,
                    'experiment_index': i,
                    'n_true_labels': args.labels,
                    'n_label': args.labels,
                    'method': method,
                    'gamma': args.gamma,
                    'run_seed': run_seed,
                    'estimate': float(taus[i]),
                    'ci_lower': float(lcis[i]),
                    'ci_higher': float(hcis[i]),
                    'true_value': float(true_val) if true_val is not None else None,
                    'settings_key': settings_key,
                })

    # Write CSVs with timestamp to avoid conflicts
    if results_rows:
        df_sum_new = pd.DataFrame(results_rows)
        summary_file = f"{timestamp}_{args.summary_csv}"
        df_sum_new.to_csv(summary_file, index=False)
        logger.info(f"Summary written to {summary_file} ({len(results_rows)} rows)")
    else:
        logger.warning("No summary rows produced.")

    if runs_rows:
        df_runs_new = pd.DataFrame(runs_rows)
        runs_file = f"{timestamp}_{args.runs_csv}"
        df_runs_new.to_csv(runs_file, index=False)
        logger.info(f"Runs written to {runs_file} ({len(runs_rows)} rows)")
    else:
        logger.warning("No per-run rows produced.")

    # 打印总结表格
    if results_rows:
        print_comparison_summary(results_rows, args.pred_csv, timestamp)


if __name__ == '__main__':
    main()
