#!/usr/bin/env python3
"""
CoT (Chain of Thought) 预测方法 - 使用推理链提高预测准确性
基于原始LLM_prediction.ipynb中的CoT方法，进行优化和扩展
"""

import json
import pandas as pd
import numpy as np
from openai import AzureOpenAI
from openai import RateLimitError, APITimeoutError, APIConnectionError
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import os
import random

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoTPredictor:
    """Chain of Thought 预测器"""
    
    def __init__(self):
        # 读取配置
        with open("config/azure_models_config.json", 'r') as f:
            config = json.load(f)
        
        # 使用微调模型配置 (与Step 7保持一致)
        finetuned_config = config["azure_endpoints"]["deployed_finetuned"]
        self.client = AzureOpenAI(
            api_version=finetuned_config["api_version"],
            azure_endpoint=finetuned_config["endpoint"],
            api_key=finetuned_config["api_key"]
        )
        self.deployment_name = finetuned_config["deployment_name"]
        
        # 问题定义
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
        
        # 优化的推理框架 - 简化为关键因素
        self.reasoning_frameworks = {
            'ECON1MOD': {
                'key_factors': [
                    "My income level and financial security",
                    "Employment opportunities in my area and education level"
                ],
                'guidance': "Think about: Am I financially stable? Are there good job opportunities for people like me?"
            },
            'UNITY': {
                'key_factors': [
                    "What I see in my community and social circles",
                    "Political and cultural divisions I observe"
                ],
                'guidance': "Think about: Do people around me generally agree on important issues, or do I see a lot of disagreement?"
            },
            'GPT1': {
                'key_factors': [
                    "My age and tech exposure",
                    "News and media I consume"
                ],
                'guidance': "Think about: Am I likely to follow tech news? Have I encountered AI/ChatGPT in news or social media since late 2022?"
            },
            'MOREGUNIMPACT': {
                'key_factors': [
                    "Gun culture and safety in my area",
                    "My personal safety concerns"
                ],
                'guidance': "Think about: Are guns common where I live? Do I see them as protection or as increasing danger?"
            },
            'GAMBLERESTR': {
                'key_factors': [
                    "My moral/religious values",
                    "Whether I see gambling as harmful or acceptable"
                ],
                'guidance': "Think about: Do my values favor more government regulation of moral issues, or less government involvement?"
            }
        }
        
        logger.info(f"CoT预测器初始化完成，使用模型: {self.deployment_name}")
    
    def build_demographic_prompt(self, row):
        """构建人口统计背景提示"""
        
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
        
        # 构建系统提示
        system_prompt = f"""Imagine you are a {row['AGE']}-year-old {gender} who is {row['RACE_TEXT']}. You were born in {birthplace}, and are currently {marital}. You have an education level of {education}. Your annual household income is {income}. You live in the {region} region and are located in a {area_type} area.

Now, imagine you are a respondent in the survey at the time of May 1st, 2024. Answer the following question honestly and accurately based on your demographic profile and personal circumstances."""
        
        return system_prompt
    
    def predict_single_cot(self, row, question_id: str) -> Optional[int]:
        """使用CoT方法进行单个预测"""
        
        # 检查是否有有效的真实回答
        if pd.isna(row[question_id]) or row[question_id] == 99.0:
            return None
        
        try:
            # 构建系统提示
            system_prompt = self.build_demographic_prompt(row)
            
            # 获取优化的推理框架
            framework = self.reasoning_frameworks[question_id]
            
            # 构建问题文本
            question_text = self.questions[question_id]
            
            # 优化的CoT prompt
            cot_prompt = f"""Question: {question_text}

Let me think through this step by step based on my background:

Step 1 - Consider the key factors that matter for this question:
• {framework['key_factors'][0]}
• {framework['key_factors'][1]}

Step 2 - Apply my personal circumstances:
{framework['guidance']}

Step 3 - Reasoning process:
Based on my profile, let me work through my likely perspective...

[Please think through this carefully, considering your demographic background and personal circumstances]

Step 4 - Final answer:
After thinking through this, my answer is: [Your final answer - just the number]"""
            
            # 发送请求 - 带重试机制
            max_retries = 3
            base_delay = 1
            
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": cot_prompt}
                        ],
                        # 取消max_completion_tokens限制
                        temperature=0.1  # 降低温度保持一致性
                    )
                    break  # 成功则退出重试循环
                    
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        # 指数退避 + 随机抖动
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Rate limit hit for RESPID {row['RESPID']}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} attempts for RESPID {row['RESPID']}")
                        return None
                        
                except (APITimeoutError, APIConnectionError) as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"API error for RESPID {row['RESPID']}, retrying in {delay:.1f}s: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"API error after {max_retries} attempts for RESPID {row['RESPID']}: {e}")
                        return None
                        
                except Exception as e:
                    logger.error(f"Unexpected error for RESPID {row['RESPID']}: {e}")
                    return None
            
            # 提取最终答案 - 改进的解析逻辑
            full_response = response.choices[0].message.content.strip()
            
            # 多种解析策略
            final_answer = None
            
            # 策略1: 寻找"answer is: X"或"my answer is: X"模式
            import re
            answer_patterns = [
                r"answer is:?\s*(\d+)",
                r"my answer is:?\s*(\d+)",
                r"final answer:?\s*(\d+)",
                r"answer:\s*(\d+)"
            ]
            
            for pattern in answer_patterns:
                match = re.search(pattern, full_response, re.IGNORECASE)
                if match:
                    final_answer = int(match.group(1))
                    break
            
            # 策略2: 如果没找到明确模式，找最后一个有效数字
            if final_answer is None:
                valid_numbers = []
                for char in full_response:
                    if char.isdigit():
                        num = int(char)
                        if num in self.response_ranges[question_id]:
                            valid_numbers.append(num)
                
                if valid_numbers:
                    final_answer = valid_numbers[-1]  # 取最后一个有效数字
            
            # 验证答案
            if final_answer is not None and final_answer in self.response_ranges[question_id]:
                return final_answer
            else:
                logger.warning(f"Invalid CoT prediction {final_answer} for {question_id}, RESPID {row['RESPID']}")
                return None
                
        except Exception as e:
            logger.error(f"CoT prediction failed for RESPID {row['RESPID']}, {question_id}: {e}")
            return None
    
    def predict_batch_threaded(self, df_batch, question_id: str, max_workers: int = 8):
        """使用线程池进行批量CoT预测"""
        
        predictions = [None] * len(df_batch)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(self.predict_single_cot, row, question_id): idx 
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
                    
                    if completed % 25 == 0:
                        logger.info(f"CoT已完成 {completed}/{len(df_batch)} 个预测")
                        
                except Exception as e:
                    logger.error(f"CoT thread execution failed: {e}")
                    predictions[idx] = None
        
        return predictions
    
    def process_full_dataset(self, batch_size: int = 100, max_workers: int = 8):
        """处理完整数据集"""
        
        logger.info("加载NPORS数据集...")
        df = pd.read_csv('data/NPORS_2024_for_public_release_updated.csv')
        
        logger.info(f"数据集大小: {len(df)} 条记录")
        logger.info(f"将使用CoT方法，批大小: {batch_size}, 线程数: {max_workers}")
        
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
            logger.info(f"\n开始CoT预测 {question_id}...")
            
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
                
                # 中间保存 (每5个批次后)
                if (i // batch_size + 1) % 5 == 0:
                    temp_filename = f'temp_cot_predictions_{question_id}_batch_{i//batch_size + 1}.csv'
                    df.to_csv(temp_filename, index=False)
                    logger.info(f"  CoT中间结果已保存: {temp_filename}")
                
                # 防止API限制 - 增加延迟
                time.sleep(3)  # 增加到3秒
        
        return df
    
    def save_results(self, df, filename: str = None):
        """保存预测结果"""
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'NPORS_2024_LLM_finetuned_optimized_cot_predictions_{timestamp}.csv'
        
        df.to_csv(filename, index=False)
        logger.info(f"CoT完整预测结果已保存: {filename}")
        
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
        
        logger.info(f"CoT预测统计信息已保存: {stats_filename}")
        
        return filename

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python 8_cot_prediction.py run [batch_size] [max_workers]")
        print("  python 8_cot_prediction.py resume [filename]")
        print("")
        print("参数:")
        print("  batch_size: 批处理大小 (默认: 100)")
        print("  max_workers: 线程数 (默认: 8)")
        return
    
    action = sys.argv[1]
    
    predictor = CoTPredictor()
    
    if action == "run":
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        
        logger.info(f"开始CoT完整数据集预测 (批大小: {batch_size}, 线程数: {max_workers})")
        
        start_time = time.time()
        df_results = predictor.process_full_dataset(batch_size, max_workers)
        end_time = time.time()
        
        logger.info(f"CoT预测完成，耗时: {(end_time - start_time)/60:.1f} 分钟")
        
        # 保存结果
        output_file = predictor.save_results(df_results)
        
        print(f"\n✅ CoT完整数据集预测完成!")
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