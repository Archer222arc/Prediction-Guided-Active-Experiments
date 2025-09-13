#!/usr/bin/env python3
"""
合并预测结果与ground truth，输出NPORS格式
"""

import pandas as pd
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def standardize_prediction_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    """标准化预测数据的列名为NPORS格式"""

    # 预测列名映射（enhanced版本到NPORS标准）
    pred_column_mapping = {
        'persona_id': 'RESPID',
        'carbon_tax_enhanced': 'CARBONTAX_LLM',
        'clean_energy_enhanced': 'CLEANENERGY_LLM',
        'clean_electricity_enhanced': 'CLEANELEC_LLM',
        'medicare_for_all_enhanced': 'MEDICAREALL_LLM',
        'public_option_enhanced': 'PUBLICOPTION_LLM',
        'immigration_reform_enhanced': 'IMMIGRATION_LLM',
        'paid_family_leave_enhanced': 'FAMILYLEAVE_LLM',
        'wealth_tax_enhanced': 'WEALTHTAX_LLM',
        'deportations_enhanced': 'DEPORTATIONS_LLM',
        'medicare_vouchers_enhanced': 'MEDICVOUCHER_LLM'
    }

    # 重命名列
    renamed_columns = {}
    for old_col, new_col in pred_column_mapping.items():
        if old_col in pred_df.columns:
            renamed_columns[old_col] = new_col

    if renamed_columns:
        pred_df = pred_df.rename(columns=renamed_columns)
        logger.info(f"预测列名重命名映射: {renamed_columns}")

    return pred_df

def merge_predictions_with_gt(gt_file: str, pred_file: str, output_file: str):
    """合并预测结果与ground truth"""

    logger.info(f"加载Ground Truth: {gt_file}")
    df_gt = pd.read_csv(gt_file)
    logger.info(f"Ground Truth形状: {df_gt.shape}")

    logger.info(f"加载预测数据: {pred_file}")
    df_pred = pd.read_csv(pred_file)
    logger.info(f"预测数据形状: {df_pred.shape}")

    # 标准化预测数据列名
    df_pred = standardize_prediction_columns(df_pred)

    logger.info(f"Ground Truth列名: {list(df_gt.columns)}")
    logger.info(f"标准化后预测数据列名: {list(df_pred.columns)}")

    # 合并数据
    merged_df = df_gt.merge(
        df_pred,
        on='RESPID',
        how='left'  # 保留所有ground truth记录
    )

    logger.info(f"合并后数据形状: {merged_df.shape}")

    # 保存结果
    merged_df.to_csv(output_file, index=False)
    logger.info(f"已保存合并结果: {output_file}")

    # 统计信息
    logger.info("=== 合并统计 ===")
    logger.info(f"Ground Truth记录: {len(df_gt)}")
    logger.info(f"预测记录: {len(df_pred)}")
    logger.info(f"成功合并记录: {len(merged_df)}")

    # 检查预测数据覆盖率
    pred_columns = [col for col in merged_df.columns if col.endswith('_LLM')]
    for col in pred_columns:
        non_null_count = merged_df[col].notna().sum()
        logger.info(f"{col}: {non_null_count}/{len(merged_df)} ({non_null_count/len(merged_df)*100:.1f}%)")

def main():
    if len(sys.argv) != 4:
        print("用法: python merge_predictions_npors.py <ground_truth.csv> <predictions.csv> <output.csv>")
        sys.exit(1)

    gt_file = sys.argv[1]
    pred_file = sys.argv[2]
    output_file = sys.argv[3]

    merge_predictions_with_gt(gt_file, pred_file, output_file)

if __name__ == "__main__":
    main()