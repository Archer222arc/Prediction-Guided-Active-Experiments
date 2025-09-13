#!/usr/bin/env python3
"""
完整数据集预测 - 使用微调模型对所有NPORS数据进行预测
采用原始LLM_prediction.ipynb中的三阶段对话方法，使用并发处理加速
"""

import json
import pandas as pd
import numpy as np
from openai import AzureOpenAI
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FullDatasetPredictor:
    """完整数据集预测器"""
    
    def __init__(self):
        # 读取配置
        with open("config/azure_models_config.json", 'r') as f:
            config = json.load(f)
        
        # 微调模型配置
        finetuned_config = config["azure_endpoints"]["deployed_finetuned"]
        self.client = AzureOpenAI(
            api_version=finetuned_config["api_version"],
            azure_endpoint=finetuned_config["endpoint"],
            api_key=finetuned_config["api_key"]
        )
        # 使用最新优化模型
        self.deployment_name = "1-mini-2025-04-14-opt-final_optimal-step560"
        
        # 问题定义 (与原始LLM_prediction保持一致)
        self.questions = {
            'ECON1MOD': "How would you rate the economic conditions in your community today? 1. Excellent, 2. Good, 3. Only fair, 4. Poor.",
            'UNITY': "Which statement comes closer to your own view, even if neither is exactly right? 1. Americans are united when it comes to the most important values. 2. Americans are divided when it comes to the most important values.",
            'GPT1': "Have you heard of ChatGPT? 1. Yes, 2. No, 3. Not sure.",
            'MOREGUNIMPACT': "If more Americans owned guns, do you think there would be... 1. More crime, 2. Less crime, 3. No difference.",
            'GAMBLERESTR': "How much government regulation of gambling do you favor? 1. A lot more than now, 2. A little more than now, 3. About the same as now, 4. A little less than now, 5. A lot less than now."
        }
        
        # 响应范围
        self.response_ranges = {
            'ECON1MOD': [1, 2, 3, 4],
            'UNITY': [1, 2],
            'GPT1': [1, 2, 3],
            'MOREGUNIMPACT': [1, 2, 3],
            'GAMBLERESTR': [1, 2, 3, 4, 5]
        }
        
        logger.info(f"预测器初始化完成，使用模型: {self.deployment_name}")
    
    def build_demographic_prompt(self, row):
        """构建人口统计背景提示 (与原始LLM_prediction完全一致)"""
        
        # 处理性别
        gender_map = {1: "Man", 2: "Woman", 3: "Some other way"}
        gender = gender_map.get(row['GENDER'], "Unknown")
        
        # 处理出生地
        birthplace_map = {
            1: "50 U.S. states or D.C.",
            2: "Puerto Rico", 
            3: "A U.S. territory",
            4: "Another country other than U.S."
        }
        birthplace = birthplace_map.get(row['BIRTHPLACE'], "Unknown")
        
        # 处理婚姻状况
        marital_map = {
            1: "Married", 2: "Living with a partner", 3: "Divorced",
            4: "Separated", 5: "Widowed", 6: "Never married"
        }
        marital = marital_map.get(row['MARITAL'], "Unknown")
        
        # 处理教育水平
        education_map = {
            1: "No formal education", 2: "1st-8th grade", 3: "Some high school",
            4: "High school graduate", 5: "Some college", 6: "Bachelor's degree",
            7: "Postgraduate degree"
        }
        education = education_map.get(row['EDUCATION'], "Unknown")
        
        # 处理收入
        income_map = {
            1: "Less than $30,000", 2: "$30,000–39,999", 3: "$40,000–49,999",
            4: "$50,000–59,999", 5: "$60,000–69,999", 6: "$70,000–79,999",
            7: "$80,000–89,999", 8: "$90,000–99,999", 9: "$100,000+"
        }
        income = income_map.get(row['INC_SDT1'], "Unknown")
        
        # 处理地区
        division_map = {
            1: "New England (CT, ME, MA, NH, RI, VT)",
            2: "Middle Atlantic (NJ, NY, PA)", 
            3: "East North Central (IL, IN, MI, OH, WI)",
            4: "West North Central (IA, KS, MN, MO, NE, ND, SD)",
            5: "South Atlantic (DE, DC, FL, GA, MD, NC, SC, VA, WV)",
            6: "East South Central (AL, KY, MS, TN)",
            7: "West South Central (AR, LA, OK, TX)",
            8: "Mountain (AZ, CO, ID, MT, NV, NM, UT, WY)",
            9: "Pacific (AK, CA, HI, OR, WA)"
        }
        region = division_map.get(row['DIVISION'], "Unknown")
        
        # 处理都市区域
        metro_map = {1: "Non-metropolitan area", 2: "Metropolitan area"}
        area_type = metro_map.get(row['METRO'], "Unknown")
        
        # 构建系统提示 (与原始方法完全一致)
        system_prompt = f"""You are a respondent in a survey at the time of May 1st, 2024. You are a {row['AGE']}-year-old {gender} who is {row['RACE_TEXT']}. You were born in {birthplace}, and are currently {marital}. You have an education level of {education}. Your annual household income is {income}. You live in the {region} region and are located in a {area_type} area. Answer survey questions based on your demographic profile and personal circumstances. Be realistic and consistent with your background."""
        
        return system_prompt
    
    def predict_single_original_method(self, row, question_id: str) -> Optional[int]:
        """使用原始LLM_prediction.ipynb的三阶段对话方法进行单个预测"""
        
        # 检查是否有有效的真实回答
        if pd.isna(row[question_id]) or row[question_id] == 99.0:
            return None
        
        try:
            # 阶段1: 建立个人背景上下文
            personal_context = self.build_demographic_prompt(row)
            
            messages = [
                {"role": "system", "content": "You are participating in a survey. I will provide your demographic background and then ask you questions."},
                {"role": "user", "content": personal_context},
                {"role": "assistant", "content": "I understand my demographic background. I'm ready to answer survey questions from this perspective."}
            ]
            
            # 阶段2: 添加时间背景
            temporal_context = "The survey is taking place on May 1st, 2024. Please keep this timeframe in mind when answering."
            messages.extend([
                {"role": "user", "content": temporal_context},
                {"role": "assistant", "content": "Understood, I'll answer from the perspective of May 1st, 2024."}
            ])
            
            # 阶段3: 提出具体问题
            question_text = self.questions[question_id]
            final_question = f"Question: {question_text}\n\nPlease provide only the number of your answer (just the digit, nothing else)."
            messages.append({"role": "user", "content": final_question})
            
            # 发送完整对话
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_completion_tokens=5,
                temperature=0.0
            )
            
            prediction = response.choices[0].message.content.strip()
            
            # 验证预测结果
            try:
                pred_num = int(prediction)
                if pred_num in self.response_ranges[question_id]:
                    return pred_num
                else:
                    logger.warning(f"Invalid prediction {pred_num} for {question_id}, RESPID {row['RESPID']}")
                    return None
            except ValueError:
                logger.warning(f"Non-numeric prediction: {prediction} for {question_id}, RESPID {row['RESPID']}")
                return None
                
        except Exception as e:
            logger.error(f"Prediction failed for RESPID {row['RESPID']}, {question_id}: {e}")
            return None
    
    def predict_batch_threaded(self, df_batch, question_id: str, max_workers: int = 10):
        """使用线程池进行批量预测"""
        
        predictions = [None] * len(df_batch)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务 (使用原始方法)
            future_to_idx = {
                executor.submit(self.predict_single_original_method, row, question_id): idx 
                for idx, (_, row) in enumerate(df_batch.iterrows())
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    predictions[idx] = result
                    completed += 1
                    
                    if completed % 50 == 0:
                        logger.info(f"已完成 {completed}/{len(df_batch)} 个预测")
                        
                except Exception as e:
                    logger.error(f"Thread execution failed: {e}")
                    predictions[idx] = None
        
        return predictions
    
    def process_full_dataset(self, batch_size: int = 200, max_workers: int = 10):
        """处理完整数据集"""
        
        logger.info("加载NPORS数据集...")
        df = pd.read_csv('data/NPORS_2024_for_public_release_updated.csv')
        
        logger.info(f"数据集大小: {len(df)} 条记录")
        logger.info(f"将使用批处理，批大小: {batch_size}, 线程数: {max_workers}")
        
        # 初始化预测结果列
        for question_id in self.questions.keys():
            df[f'{question_id}_LLM'] = None
        
        # 计算每个问题的有效样本数
        logger.info("\n问题有效样本数统计:")
        for question_id in self.questions.keys():
            valid_count = ((df[question_id].notna()) & (df[question_id] != 99.0)).sum()
            logger.info(f"  {question_id}: {valid_count} 个有效样本")
        
        # 分批处理每个问题
        for question_id in self.questions.keys():
            logger.info(f"\n开始预测 {question_id}...")
            
            # 只处理有有效答案的记录
            valid_mask = (df[question_id].notna()) & (df[question_id] != 99.0)
            valid_indices = df[valid_mask].index.tolist()
            
            if len(valid_indices) == 0:
                logger.info(f"  跳过 {question_id}: 没有有效数据")
                continue
            
            logger.info(f"  处理 {len(valid_indices)} 个有效样本")
            
            # 分批处理
            for i in range(0, len(valid_indices), batch_size):
                batch_indices = valid_indices[i:i+batch_size]
                batch_df = df.loc[batch_indices]
                
                logger.info(f"  处理批次 {i//batch_size + 1}/{(len(valid_indices)-1)//batch_size + 1} "
                           f"(索引 {i} - {min(i+batch_size, len(valid_indices))})")
                
                # 预测批次
                batch_predictions = self.predict_batch_threaded(
                    batch_df, question_id, max_workers
                )
                
                # 保存结果
                for j, pred in enumerate(batch_predictions):
                    df.at[batch_indices[j], f'{question_id}_LLM'] = pred
                
                # 中间保存 (每批次后)
                if (i // batch_size + 1) % 5 == 0:  # 每5个批次保存一次
                    temp_filename = f'temp_predictions_{question_id}_batch_{i//batch_size + 1}.csv'
                    df.to_csv(temp_filename, index=False)
                    logger.info(f"  中间结果已保存: {temp_filename}")
                
                # 防止API限制
                time.sleep(1)
        
        return df
    
    def save_results(self, df, filename: str = None):
        """保存预测结果"""
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'NPORS_2024_base_gpt41mini_lr06_step560_{timestamp}.csv'
        
        df.to_csv(filename, index=False)
        logger.info(f"完整预测结果已保存: {filename}")
        
        # 保存统计信息
        stats = {}
        for question_id in self.questions.keys():
            col_name = f'{question_id}_LLM'
            if col_name in df.columns:
                total_valid = ((df[question_id].notna()) & (df[question_id] != 99.0)).sum()
                predicted = df[col_name].notna().sum()
                success_rate = predicted / total_valid if total_valid > 0 else 0
                
                stats[question_id] = {
                    'total_valid_samples': int(total_valid),
                    'successful_predictions': int(predicted),
                    'success_rate': success_rate
                }
        
        stats_filename = filename.replace('.csv', '_stats.json')
        with open(stats_filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"预测统计信息已保存: {stats_filename}")
        
        return filename

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python 7_full_dataset_prediction.py run [batch_size] [max_workers]")
        print("  python 7_full_dataset_prediction.py resume [filename]")
        print("")
        print("参数:")
        print("  batch_size: 批处理大小 (默认: 200)")
        print("  max_workers: 线程数 (默认: 10)")
        return
    
    action = sys.argv[1]
    
    predictor = FullDatasetPredictor()
    
    if action == "run":
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        
        logger.info(f"开始完整数据集预测 (批大小: {batch_size}, 线程数: {max_workers})")
        
        start_time = time.time()
        df_results = predictor.process_full_dataset(batch_size, max_workers)
        end_time = time.time()
        
        logger.info(f"预测完成，耗时: {(end_time - start_time)/60:.1f} 分钟")
        
        # 保存结果
        output_file = predictor.save_results(df_results)
        
        print(f"\n✅ 完整数据集预测完成!")
        print(f"结果文件: {output_file}")
        print(f"耗时: {(end_time - start_time)/60:.1f} 分钟")
        
    elif action == "resume":
        if len(sys.argv) < 3:
            print("请提供要恢复的文件名")
            return
        
        filename = sys.argv[2]
        logger.info(f"从文件恢复: {filename}")
        
        try:
            df = pd.read_csv(filename)
            output_file = predictor.save_results(df, filename.replace('temp_', 'final_'))
            print(f"结果已保存: {output_file}")
        except Exception as e:
            logger.error(f"恢复失败: {e}")
    
    else:
        print(f"未知操作: {action}")

if __name__ == "__main__":
    main()