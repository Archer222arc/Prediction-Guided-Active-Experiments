#!/usr/bin/env python3
"""
改进的CoT (Chain of Thought) 预测方法
使用最新优化模型和改进的推理链方法
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

class ImprovedCoTPredictor:
    """改进的Chain of Thought预测器"""
    
    def __init__(self):
        # 读取配置
        with open("config/azure_models_config.json", 'r') as f:
            config = json.load(f)
        
        # 使用最新优化模型
        finetuned_config = config["azure_endpoints"]["deployed_finetuned"]
        self.client = AzureOpenAI(
            api_version=finetuned_config["api_version"],
            azure_endpoint=finetuned_config["endpoint"],
            api_key=finetuned_config["api_key"]
        )
        # 使用最新优化模型
        self.deployment_name = "1-mini-2025-04-14-opt-final_optimal-step560"
        
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
        
        # 改进的推理框架 - 更具体和实用的推理指导
        self.reasoning_frameworks = {
            'ECON1MOD': {
                'steps': [
                    "Consider my personal financial situation: income stability, employment status, and recent changes",
                    "Think about local economic indicators: job availability, business activity, housing market in my area",
                    "Reflect on how my demographic background (age, education, income) typically correlates with economic optimism",
                    "Weigh current economic pressures (inflation, cost of living) against personal financial security"
                ]
            },
            'UNITY': {
                'steps': [
                    "Observe political discourse in my community and social networks",
                    "Consider major national issues and whether people around me generally agree or disagree",
                    "Think about my own political views and whether I see compromise or division as more prevalent",
                    "Assess whether my demographic group tends toward seeing unity or division"
                ]
            },
            'GPT1': {
                'steps': [
                    "Consider my age and technology adoption patterns - am I likely to follow tech news?",
                    "Think about my media consumption: do I read tech news, use social media where AI is discussed?",
                    "Assess my education level and whether that correlates with awareness of emerging technologies",
                    "Consider the timeframe: ChatGPT became widely known in late 2022/early 2023"
                ]
            },
            'MOREGUNIMPACT': {
                'steps': [
                    "Think about gun culture in my region and community - are guns seen as normal/protective or dangerous?",
                    "Consider my personal safety concerns and whether I see guns as protection or risk",
                    "Reflect on my political views and whether they tend toward gun rights or gun control",
                    "Assess local crime patterns and whether more guns would help or hurt in my specific area"
                ]
            },
            'GAMBLERESTR': {
                'steps': [
                    "Consider my religious/moral values and stance on government regulation of personal choices",
                    "Think about whether I see gambling as harmful to society or as acceptable personal entertainment",
                    "Assess my general political philosophy: more government regulation vs. personal freedom",
                    "Consider my demographic group's typical stance on moral regulation"
                ]
            }
        }
        
        logger.info(f"改进CoT预测器初始化完成，使用模型: {self.deployment_name}")
    
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
        
        # 构建详细的背景描述
        background = f"""I am a {row['AGE']}-year-old {gender} who is {row['RACE_TEXT']}. I was born in {birthplace}, and am currently {marital}. I have an education level of {education}. My annual household income is {income}. I live in the {region} region and am located in a {area_type} area. The survey is taking place on May 1st, 2024."""
        
        return background
    
    def create_cot_prompt(self, background: str, question_id: str, question_text: str) -> str:
        """创建改进的CoT提示"""
        
        steps = self.reasoning_frameworks[question_id]['steps']
        step_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
        
        cot_prompt = f"""Given my background: {background}

Question: {question_text}

I need to think through this step by step:

{step_text}

Based on this reasoning, considering my specific demographic background and circumstances, my answer is: """

        return cot_prompt
    
    def predict_single_cot(self, row, question_id: str, max_retries: int = 3) -> Optional[int]:
        """使用改进的CoT方法进行单个预测"""
        
        # 检查是否有有效的真实回答
        if pd.isna(row[question_id]) or row[question_id] == 99.0:
            return None
        
        for attempt in range(max_retries):
            try:
                # 构建背景和CoT提示
                background = self.build_demographic_prompt(row)
                question_text = self.questions[question_id]
                cot_prompt = self.create_cot_prompt(background, question_id, question_text)
                
                # 使用改进的系统提示
                messages = [
                    {
                        "role": "system", 
                        "content": "You are a survey respondent answering based on your demographic background. Think through each question step by step, considering how your specific circumstances would influence your views. Provide your final answer as just the number (1, 2, 3, etc.)."
                    },
                    {
                        "role": "user", 
                        "content": cot_prompt
                    }
                ]
                
                # API调用
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_completion_tokens=200,  # 给CoT更多token空间
                    temperature=0.1,  # 略微降低温度以提高一致性
                    top_p=0.9
                )
                
                full_response = response.choices[0].message.content.strip()
                
                # 提取最终答案 - 寻找最后出现的数字
                import re
                numbers = re.findall(r'\b([1-9])\b', full_response)
                
                if numbers:
                    # 取最后一个数字作为答案
                    pred_num = int(numbers[-1])
                    if pred_num in self.response_ranges[question_id]:
                        return pred_num
                    else:
                        logger.warning(f"Invalid prediction {pred_num} for {question_id}, RESPID {row['RESPID']}, attempt {attempt+1}")
                else:
                    logger.warning(f"No number found in response for {question_id}, RESPID {row['RESPID']}, attempt {attempt+1}")
                
                # 如果第一次尝试失败，稍微等待再重试
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                logger.warning(f"API error for RESPID {row['RESPID']}, {question_id}, attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # 指数退避
                    logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"达到最大重试次数，放弃 RESPID {row['RESPID']}, {question_id}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error for RESPID {row['RESPID']}, {question_id}, attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return None
        
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
                        logger.info(f"已完成 {completed}/{len(df_batch)} 个CoT预测")
                        
                except Exception as e:
                    logger.error(f"Thread execution failed: {e}")
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
                
                logger.info(f"  CoT处理批次 {i//batch_size + 1}/{(len(valid_indices)-1)//batch_size + 1} "
                           f"(索引 {i} - {min(i+batch_size, len(valid_indices))})")
                
                # CoT预测批次
                batch_predictions = self.predict_batch_threaded(
                    batch_df, question_id, max_workers
                )
                
                # 保存结果
                for j, pred in enumerate(batch_predictions):
                    df.at[batch_indices[j], f'{question_id}_LLM'] = pred
                
                # 中间保存 (每3个批次后)
                if (i // batch_size + 1) % 3 == 0:
                    temp_filename = f'temp_cot_predictions_{question_id}_batch_{i//batch_size + 1}.csv'
                    df.to_csv(temp_filename, index=False)
                    logger.info(f"  中间结果已保存: {temp_filename}")
                
                # 防止API限制
                time.sleep(2)
        
        return df
    
    def save_results(self, df, filename: str = None):
        """保存CoT预测结果"""
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'NPORS_2024_cot_optimized_lr06_step560_{timestamp}.csv'
        
        df.to_csv(filename, index=False)
        logger.info(f"完整CoT预测结果已保存: {filename}")
        
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
        print("  python 9_improved_cot_prediction.py run [batch_size] [max_workers]")
        print("  python 9_improved_cot_prediction.py resume [filename]")
        print("")
        print("参数:")
        print("  batch_size: 批处理大小 (默认: 100, CoT需要更多token)")
        print("  max_workers: 线程数 (默认: 8, CoT响应时间较长)")
        return
    
    action = sys.argv[1]
    
    predictor = ImprovedCoTPredictor()
    
    if action == "run":
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        
        logger.info(f"开始改进CoT预测 (批大小: {batch_size}, 线程数: {max_workers})")
        
        start_time = time.time()
        df_results = predictor.process_full_dataset(batch_size, max_workers)
        end_time = time.time()
        
        logger.info(f"CoT预测完成，耗时: {(end_time - start_time)/60:.1f} 分钟")
        
        # 保存结果
        output_file = predictor.save_results(df_results)
        
        print(f"\n✅ 改进CoT预测完成!")
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