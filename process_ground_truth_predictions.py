#!/usr/bin/env python3
"""
处理Ground Truth和Prediction数据的脚本
Process Ground Truth and Prediction Data Script

根据用户方案生成标准化的CSV文件用于估计器实验
"""

import json
import re
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import sys

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GroundTruthProcessor:
    """Ground Truth和预测数据处理器"""

    def __init__(self):
        # 10个政策题的统一ID和对应的标准列名
        self.question_mapping = {
            "carbon_tax": "CARBONTAX",
            "clean_energy": "CLEANENERGY",
            "clean_electricity": "CLEANELEC",
            "medicare_for_all": "MEDICAREALL",
            "public_option": "PUBLICOPTION",
            "immigration_reform": "IMMIGRATION",
            "paid_family_leave": "FAMILYLEAVE",
            "wealth_tax": "WEALTHTAX",
            "deportations": "DEPORTATIONS",
            "medicare_vouchers": "MEDICVOUCHER"
        }

        # 保持原有的question_ids用于内部处理
        self.question_ids = list(self.question_mapping.keys())

        # 教育水平映射（数值化，取值 < 10）
        self.education_mapping = {
            "less than high school": 1,
            "high school": 2,
            "some college": 3,
            "associate degree": 3,
            "associate": 3,
            "bachelor": 4,
            "bachelor's": 4,
            "graduate": 5,
            "professional": 5,
            "master": 5,
            "doctorate": 5,
            "phd": 5
        }

        logger.info("Ground Truth处理器初始化完成")

    def extract_number_1to5(self, value) -> Optional[int]:
        """从文本中提取1-5的数字"""
        if pd.isna(value):
            return pd.NA

        # 尝试直接转换数字
        try:
            num = int(float(value))
            if 1 <= num <= 5:
                return num
        except (ValueError, TypeError):
            pass

        # 从文本中用正则提取
        match = re.search(r'\b([1-5])\b', str(value))
        return int(match.group(1)) if match else pd.NA

    def map_education_value(self, edu_text) -> Optional[int]:
        """将教育文本映射为数值"""
        if pd.isna(edu_text):
            return pd.NA

        text_lower = str(edu_text).lower().strip()

        # 精确匹配
        for key, value in self.education_mapping.items():
            if key in text_lower:
                return value

        # 如果没有匹配到，尝试从数字提取
        match = re.search(r'\b([1-9])\b', text_lower)
        if match:
            num = int(match.group(1))
            return num if num < 10 else pd.NA

        logger.warning(f"无法映射教育文本: '{edu_text}'")
        return pd.NA

    def load_and_process_ground_truth(self, gt_json_path: str) -> pd.DataFrame:
        """
        加载并处理Ground Truth数据 - 专门处理Questions嵌套结构

        Args:
            gt_json_path: Ground truth JSON文件路径

        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        logger.info(f"加载Ground Truth数据: {gt_json_path}")

        try:
            with open(gt_json_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)

            logger.info(f"原始数据: {len(gt_data)} 条记录")

            # 处理特定的嵌套结构：每个persona是一个list，第一个元素包含Questions
            processed_records = []

            for i, persona_data in enumerate(gt_data):
                if isinstance(persona_data, list) and len(persona_data) > 0:
                    # 获取第一个块（包含Questions的块）
                    first_block = persona_data[0]

                    if isinstance(first_block, dict) and 'Questions' in first_block:
                        questions = first_block['Questions']

                        # 初始化记录
                        record = {
                            'persona_id': f"pid_{i+1}",  # 使用与parquet一致的格式
                            'original_index': i
                        }

                        # 处理Questions数据
                        for question in questions:
                            if isinstance(question, dict) and 'Answers' in question:
                                answers = question['Answers']
                                rows = question.get('Rows', [])

                                if 'SelectedByPosition' in answers and isinstance(answers['SelectedByPosition'], list):
                                    # 获取每个问题的答案（直接使用原始值，应该已经是1-5）
                                    selected_positions = answers['SelectedByPosition']

                                    # 映射每个Rows到我们的question_id
                                    for j, row_text in enumerate(rows):
                                        if j < len(selected_positions):
                                            # 直接使用原始值（应该已经是1-5）
                                            answer_value = selected_positions[j]

                                            # 根据问题文本映射到question_id
                                            question_id = self.map_row_to_question_id(row_text)
                                            if question_id:
                                                record[f"{question_id}_true_num"] = answer_value

                        processed_records.append(record)

            df_gt = pd.DataFrame(processed_records)
            logger.info(f"处理后数据形状: {df_gt.shape}")

            if not df_gt.empty:
                logger.info(f"提取的问题列: {[col for col in df_gt.columns if col.endswith('_true_num')]}")

            return df_gt

        except Exception as e:
            logger.error(f"加载Ground Truth失败: {e}")
            return pd.DataFrame()

    def map_row_to_question_id(self, row_text: str) -> Optional[str]:
        """将问题行文本映射到question_id"""

        if not row_text:
            return None

        row_lower = row_text.lower().strip()

        # 定义映射规则
        mapping_rules = {
            'carbon_tax': ['carbon emissions', 'carbon tax', 'tax on carbon'],
            'clean_energy': ['clean energy infrastructure', '40%', 'low-income communities'],
            'clean_electricity': ['carbon-pollution free electricity', 'electricity sector', '2035'],
            'medicare_for_all': ['medicare for all', 'government-run plan', 'all americans would get healthcare'],
            'public_option': ['public option', 'buy into a government-run healthcare'],
            'immigration_reform': ['immigration reforms', 'path to u.s. citizenship', 'undocumented immigrants'],
            'paid_family_leave': ['paid family leave', 'companies to provide paid family'],
            'wealth_tax': ['2% tax', 'assets of individuals', '$50 million', 'net worth'],
            'deportations': ['increasing deportations', 'us illegally'],
            'medicare_vouchers': ['healthcare vouchers', 'private healthcare plans', 'traditional medicare']
        }

        # 查找匹配
        for question_id, keywords in mapping_rules.items():
            if any(keyword in row_lower for keyword in keywords):
                return question_id

        logger.warning(f"无法映射问题文本到question_id: '{row_text}'")
        return None

    def process_ground_truth_questions(self, df_gt: pd.DataFrame) -> pd.DataFrame:
        """处理Ground Truth中的问题数据 - 现在数据已经在load阶段处理完成"""

        logger.info("验证Ground Truth问题数据...")

        # 统计每个问题的有效值数量
        for question_id in self.question_ids:
            col_name = f"{question_id}_true_num"
            if col_name in df_gt.columns:
                valid_count = df_gt[col_name].notna().sum()
                if valid_count > 0:
                    value_counts = df_gt[col_name].value_counts().sort_index()
                    logger.info(f"{question_id}: {valid_count} 个有效值, 分布: {dict(value_counts)}")
                else:
                    logger.warning(f"{question_id}: 没有有效值")
            else:
                logger.warning(f"缺少列: {col_name}")

        return df_gt

    def process_education(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理教育水平数据 - 从persona parquet文件获取教育信息"""

        logger.info("处理教育水平数据...")

        try:
            # 尝试从persona parquet文件加载教育信息
            parquet_file = Path("data/personas_for_prediction.parquet")
            if parquet_file.exists():
                logger.info("从persona parquet文件获取教育信息...")
                persona_df = pd.read_parquet(parquet_file)

                # 创建从persona_id到pid的映射
                education_map = {}

                for _, row in persona_df.iterrows():
                    persona_id = row['persona_id']  # 如pid_123
                    persona_text = row['persona_text']

                    # 从persona文本中提取教育信息
                    education_text = self.extract_education_from_persona_text(persona_text)
                    education_map[persona_id] = education_text

                # 为df中的每个persona匹配教育信息
                df["EDUCATION_TEXT"] = df["persona_id"].map(education_map).fillna("Unknown")
                df["EDUCATION"] = df["EDUCATION_TEXT"].apply(self.map_education_value)

                # 统计成功匹配的数量
                valid_edu_count = df["EDUCATION"].notna().sum()
                matched_count = df["EDUCATION_TEXT"].ne("Unknown").sum()
                logger.info(f"成功匹配 {matched_count} 个persona的教育信息")
                logger.info(f"成功映射 {valid_edu_count} 个有效教育值")

                return df

            else:
                logger.warning("persona parquet文件不存在，使用默认教育值")

        except Exception as e:
            logger.error(f"从persona文件获取教育信息失败: {e}")

        # 回退方案：使用默认值
        logger.warning("使用默认教育值")
        df["EDUCATION_TEXT"] = "Some college"
        df["EDUCATION"] = 3  # 默认为Some college

        return df

    def extract_education_from_persona_text(self, persona_text: str) -> str:
        """从persona文本中提取教育信息"""

        if pd.isna(persona_text):
            return "Unknown"

        # 查找"Education level:"行
        lines = persona_text.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.lower().startswith('education level:'):
                # 提取冒号后的内容
                edu_text = line_stripped.split(':', 1)[1].strip()
                return edu_text

        # 备用方案：查找其他教育相关行
        for line in lines:
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in ['education:', 'degree:', 'schooling:']):
                if ':' in line:
                    edu_text = line.split(':', 1)[1].strip()
                    return edu_text

        # 最后方案：在整个文本中查找关键词
        text_lower = str(persona_text).lower()
        education_keywords = {
            'less than high school': 'Less than high school',
            'high school graduate': 'High school graduate',
            'high school diploma': 'High school graduate',
            'some college': 'Some college',
            'associate degree': 'Associate degree',
            'bachelor': "Bachelor's degree",
            'master': "Master's degree",
            'graduate degree': 'Graduate degree',
            'professional degree': 'Professional degree',
            'doctorate': 'Doctorate',
            'phd': 'PhD'
        }

        for keyword, standard_form in education_keywords.items():
            if keyword in text_lower:
                return standard_form

        return "Unknown"

    def clean_ground_truth_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗Ground Truth数据"""

        logger.info("开始数据清洗...")
        original_count = len(df)

        # 确保persona_id存在
        if "persona_id" not in df.columns:
            df["persona_id"] = [f"gt_persona_{i+1}" for i in range(len(df))]

        # 选择需要保留的列
        keep_cols = ["persona_id", "EDUCATION", "EDUCATION_TEXT"]
        question_cols = [f"{q}_true_num" for q in self.question_ids]
        keep_cols.extend(question_cols)

        # 只保留存在的列
        existing_cols = [col for col in keep_cols if col in df.columns]
        df_clean = df[existing_cols].copy()

        # 清洗规则
        # 1. 移除persona_id为空的行
        df_clean = df_clean.dropna(subset=['persona_id'])

        # 2. 清洗EDUCATION值（必须在1-9之间）
        if 'EDUCATION' in df_clean.columns:
            df_clean = df_clean[(df_clean['EDUCATION'] >= 1) & (df_clean['EDUCATION'] < 10)]

        # 3. 清洗问题答案（必须在1-5之间）
        for question_id in self.question_ids:
            col_name = f"{question_id}_true_num"
            if col_name in df_clean.columns:
                df_clean = df_clean[
                    (df_clean[col_name] >= 1) & (df_clean[col_name] <= 5)
                ]

        # 4. 移除关键字段全为空的行
        essential_cols = question_cols if any(col in df_clean.columns for col in question_cols) else []
        if essential_cols:
            # 至少要有一个问题有有效答案
            df_clean = df_clean.dropna(subset=essential_cols, how='all')

        final_count = len(df_clean)
        logger.info(f"数据清洗完成: {original_count} -> {final_count} 行")

        return df_clean

    def merge_with_predictions(self, df_gt: pd.DataFrame, prediction_csv: str) -> pd.DataFrame:
        """与预测数据合并，按NPORS格式"""

        logger.info(f"加载预测数据: {prediction_csv}")

        try:
            df_pred = pd.read_csv(prediction_csv)
            logger.info(f"预测数据形状: {df_pred.shape}")

            # 标准化预测数据的列名
            df_pred = self.standardize_prediction_columns(df_pred)

            logger.info(f"Ground truth列名: {list(df_gt.columns)}")
            logger.info(f"预测数据列名: {list(df_pred.columns)}")

            # 基于RESPID合并
            merged_df = df_gt.merge(
                df_pred,
                left_on='RESPID',
                right_on='RESPID',
                how='left'  # 使用left join保留所有ground truth记录
            )

            logger.info(f"合并后数据形状: {merged_df.shape}")
            return merged_df

        except Exception as e:
            logger.error(f"合并预测数据失败: {e}")
            return df_gt

    def standardize_prediction_columns(self, pred_df: pd.DataFrame) -> pd.DataFrame:
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

    def legacy_merge_with_predictions(self, df_gt: pd.DataFrame, prediction_csv: str) -> pd.DataFrame:
        """旧版本的预测数据合并方法（保留备用）"""

        logger.info(f"加载预测数据: {prediction_csv}")

        try:
            df_pred = pd.read_csv(prediction_csv)
            logger.info(f"预测数据形状: {df_pred.shape}")

            # 处理预测数据中的问题列
            for question_id in self.question_ids:
                # 查找预测列
                pred_candidates = [
                    f"{question_id}_enhanced",
                    f"{question_id}_llm_pred",
                    f"{question_id}_LLM",
                    f"{question_id}_pred"
                ]

                found_pred_col = None
                for candidate in pred_candidates:
                    if candidate in df_pred.columns:
                        found_pred_col = candidate
                        break

                if found_pred_col:
                    # 确保预测值是数值型
                    if found_pred_col.endswith('_num'):
                        df_pred[f"{question_id}_llm_pred_num"] = df_pred[found_pred_col]
                    else:
                        df_pred[f"{question_id}_llm_pred_num"] = df_pred[found_pred_col].apply(self.extract_number_1to5)

                    logger.info(f"处理预测列: {found_pred_col} -> {question_id}_llm_pred_num")
                else:
                    logger.warning(f"未找到 {question_id} 的预测列")
                    df_pred[f"{question_id}_llm_pred_num"] = pd.NA

            # 执行inner join
            df_merged = df_pred.merge(df_gt, on='persona_id', how='inner')
            logger.info(f"合并后数据形状: {df_merged.shape}")

            return df_merged

        except Exception as e:
            logger.error(f"合并预测数据失败: {e}")
            return pd.DataFrame()

    def standardize_column_names(self, df: pd.DataFrame, data_type: str = "ground_truth") -> pd.DataFrame:
        """
        规范化列名以符合NPORS标准

        Args:
            df: 要规范化的DataFrame
            data_type: 数据类型 ("ground_truth" 或 "merged")
        """
        logger.info(f"规范化{data_type}的列名...")

        df_std = df.copy()

        # 构建重命名映射
        rename_mapping = {}

        # 1. 规范化ID列
        if 'persona_id' in df_std.columns:
            rename_mapping['persona_id'] = 'RESPID'

        # 2. 规范化问题列
        for question_id, standard_name in self.question_mapping.items():
            # Ground truth列
            old_gt_col = f"{question_id}_true_num"
            if old_gt_col in df_std.columns:
                rename_mapping[old_gt_col] = standard_name  # 直接使用标准名作为真值

            # LLM预测列 (如果存在)
            old_pred_col = f"{question_id}_llm_pred_num"
            if old_pred_col in df_std.columns:
                rename_mapping[old_pred_col] = f"{standard_name}_LLM"

            # Enhanced列 (如果存在)
            old_enhanced_col = f"{question_id}_enhanced"
            if old_enhanced_col in df_std.columns:
                rename_mapping[old_enhanced_col] = f"{standard_name}_RAW"

        # 3. 保持其他重要列名不变但规范化
        if 'EDUCATION' in df_std.columns:
            # EDUCATION已经符合标准
            pass
        if 'EDUCATION_TEXT' in df_std.columns:
            # 保持不变，作为辅助列
            pass

        # 应用重命名
        df_std = df_std.rename(columns=rename_mapping)

        # 记录重命名信息
        if rename_mapping:
            logger.info("列名重命名映射:")
            for old_name, new_name in rename_mapping.items():
                logger.info(f"  {old_name} -> {new_name}")

        return df_std

    def save_results(self, df_gt: pd.DataFrame, df_merged: pd.DataFrame = None):
        """保存处理结果"""

        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        # 规范化并保存Ground Truth
        df_gt_std = self.standardize_column_names(df_gt, "ground_truth")
        gt_output_path = data_dir / "twin_ground_truth_clean.csv"
        df_gt_std.to_csv(gt_output_path, index=False)
        logger.info(f"已保存Ground Truth清洗数据: {gt_output_path}")

        # 如果有合并数据，也规范化并保存
        if df_merged is not None and not df_merged.empty:
            df_merged_std = self.standardize_column_names(df_merged, "merged")
            merged_output_path = data_dir / "twin_merged_gt_pred.csv"
            df_merged_std.to_csv(merged_output_path, index=False)
            logger.info(f"已保存合并数据: {merged_output_path}")

        return str(gt_output_path)

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='处理Ground Truth和预测数据')
    parser.add_argument('--gt-json', default='data/ground_truth_output.json',
                       help='Ground truth JSON文件路径')
    parser.add_argument('--pred-csv', default=None,
                       help='预测CSV文件路径（可选）')
    parser.add_argument('--auto-find-pred', action='store_true',
                       help='自动查找最新的enhanced_digital_twin_predictions文件')

    args = parser.parse_args()

    processor = GroundTruthProcessor()

    # 检查Ground Truth文件是否存在
    if not Path(args.gt_json).exists():
        logger.error(f"Ground Truth文件不存在: {args.gt_json}")
        sys.exit(1)

    try:
        # 1. 处理Ground Truth
        logger.info("=== 第1步: 处理Ground Truth数据 ===")
        df_gt_raw = processor.load_and_process_ground_truth(args.gt_json)

        if df_gt_raw.empty:
            logger.error("无法加载Ground Truth数据")
            sys.exit(1)

        # 2. 处理问题数据
        logger.info("=== 第2步: 处理问题数据 ===")
        df_gt_with_questions = processor.process_ground_truth_questions(df_gt_raw)

        # 3. 处理教育数据
        logger.info("=== 第3步: 处理教育数据 ===")
        df_gt_with_edu = processor.process_education(df_gt_with_questions)

        # 4. 数据清洗
        logger.info("=== 第4步: 数据清洗 ===")
        df_gt_clean = processor.clean_ground_truth_data(df_gt_with_edu)

        if df_gt_clean.empty:
            logger.error("清洗后的数据为空")
            sys.exit(1)

        # 5. 查找并合并预测数据（可选）
        df_merged = None
        pred_file = args.pred_csv

        if args.auto_find_pred:
            # 自动查找最新的预测文件
            pred_files = list(Path('.').glob('enhanced_digital_twin_predictions_*.csv'))
            if pred_files:
                pred_file = str(sorted(pred_files)[-1])  # 取最新的
                logger.info(f"自动找到预测文件: {pred_file}")

        if pred_file and Path(pred_file).exists():
            logger.info("=== 第5步: 合并预测数据 ===")
            df_merged = processor.merge_with_predictions(df_gt_clean, pred_file)
        else:
            logger.info("未提供预测文件或文件不存在，跳过合并步骤")

        # 6. 保存结果
        logger.info("=== 第6步: 保存结果 ===")
        output_path = processor.save_results(df_gt_clean, df_merged)

        # 7. 输出统计信息
        logger.info("=== 处理完成统计 ===")
        logger.info(f"Ground Truth清洗数据: {len(df_gt_clean)} 行")
        if df_merged is not None:
            logger.info(f"合并数据: {len(df_merged)} 行")

        # 验证数据质量
        logger.info("=== 数据质量验证 ===")

        # 验证EDUCATION字段
        if 'EDUCATION' in df_gt_clean.columns:
            edu_range = df_gt_clean['EDUCATION'].value_counts().sort_index()
            logger.info(f"EDUCATION分布: \n{edu_range}")

        # 验证问题字段
        for question_id in processor.question_ids:
            col_name = f"{question_id}_true_num"
            if col_name in df_gt_clean.columns:
                valid_count = df_gt_clean[col_name].notna().sum()
                value_range = df_gt_clean[col_name].value_counts().sort_index()
                logger.info(f"{question_id}: {valid_count} 个有效值, 分布: {dict(value_range)}")

        print(f"\n✅ 数据处理完成!")
        print(f"输出文件: {output_path}")
        if df_merged is not None:
            print(f"合并文件: data/twin_merged_gt_pred.csv")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()