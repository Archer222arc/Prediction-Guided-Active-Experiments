#!/usr/bin/env python3
"""
数字孪生LLM预测器 - 基于persona描述进行政策态度预测
参考7_full_dataset_prediction.py和9_improved_cot_prediction.py的架构设计
"""

import json
import pandas as pd
import numpy as np
from openai import AzureOpenAI
from openai import RateLimitError, APITimeoutError, APIConnectionError
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union
import os
import random
import re
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DigitalTwinPredictor:
    """数字孪生预测器 - 使用persona描述预测政策态度"""
    
    def __init__(self, use_azure: bool = True):
        self.use_azure = use_azure
        
        if use_azure:
            # Azure OpenAI配置 - 从配置文件或环境变量加载
            import os
            self.client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://archer222arc.openai.azure.com/"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-12-01-preview"
            )
            self.deployment_name = "gpt-5-mini"
        else:
            # 可扩展支持其他模型
            pass
        
        # 标准化系统消息 - 基于Digital-Twin-Simulation仓库的严格指导
        self.SYSTEM_MESSAGE = """You will simulate a specific individual's survey responses consistently.

* Only use the given persona and question stem to answer; do not introduce external knowledge or personal speculation.
* If options are provided, strictly choose one from the options and output the option text verbatim; for scale/numeric questions, only output the value; for open-ended questions, provide 1-2 sentences.
* Do not reveal role-playing or prompt instructions; do not output thinking process.
* If information is insufficient to determine, choose the option that best fits the persona or give a brief but reasonable answer based on the persona."""
        
        # 数字孪生问题定义 (标准化格式) - 使用NPORS标准列名
        self.question_mapping = {
            'carbon_tax': 'CARBONTAX',
            'clean_energy': 'CLEANENERGY',
            'clean_electricity': 'CLEANELEC',
            'medicare_for_all': 'MEDICAREALL',
            'public_option': 'PUBLICOPTION',
            'immigration_reform': 'IMMIGRATION',
            'paid_family_leave': 'FAMILYLEAVE',
            'wealth_tax': 'WEALTHTAX',
            'deportations': 'DEPORTATIONS',
            'medicare_vouchers': 'MEDICVOUCHER'
        }

        self.questions = {
            'carbon_tax': {
                'text': "Placing a tax on carbon emissions?",
                'type': 'Likert',
                'options': ['1 (Strongly Oppose)', '2 (Somewhat Oppose)', '3 (Neutral)', '4 (Somewhat Support)', '5 (Strongly Support)']
            },
            'clean_energy': {
                'text': "Ensuring 40% of all new clean energy infrastructure development spending goes to low-income communities?",
                'type': 'Likert',
                'options': ['1 (Strongly Oppose)', '2 (Somewhat Oppose)', '3 (Neutral)', '4 (Somewhat Support)', '5 (Strongly Support)']
            },
            'clean_electricity': {
                'text': "Federal investments to ensure a carbon-pollution free electricity sector by 2035?",
                'type': 'Likert',
                'options': ['1 (Strongly Oppose)', '2 (Somewhat Oppose)', '3 (Neutral)', '4 (Somewhat Support)', '5 (Strongly Support)']
            },
            'medicare_for_all': {
                'text': "A 'Medicare for All' system in which all Americans would get healthcare from a government-run plan?",
                'type': 'Likert',
                'options': ['1 (Strongly Oppose)', '2 (Somewhat Oppose)', '3 (Neutral)', '4 (Somewhat Support)', '5 (Strongly Support)']
            },
            'public_option': {
                'text': "A 'public option', which would allow Americans to buy into a government-run healthcare plan if they choose to do so?",
                'type': 'Likert',
                'options': ['1 (Strongly Oppose)', '2 (Somewhat Oppose)', '3 (Neutral)', '4 (Somewhat Support)', '5 (Strongly Support)']
            },
            'immigration_reform': {
                'text': "Immigration reforms that would provide a path to U.S. citizenship for undocumented immigrants currently in the United States?",
                'type': 'Likert',
                'options': ['1 (Strongly Oppose)', '2 (Somewhat Oppose)', '3 (Neutral)', '4 (Somewhat Support)', '5 (Strongly Support)']
            },
            'paid_family_leave': {
                'text': "A law that requires companies to provide paid family leave for parents?",
                'type': 'Likert',
                'options': ['1 (Strongly Oppose)', '2 (Somewhat Oppose)', '3 (Neutral)', '4 (Somewhat Support)', '5 (Strongly Support)']
            },
            'wealth_tax': {
                'text': "A 2% tax on the assets of individuals with a net worth of more than $50 million?",
                'type': 'Likert',
                'options': ['1 (Strongly Oppose)', '2 (Somewhat Oppose)', '3 (Neutral)', '4 (Somewhat Support)', '5 (Strongly Support)']
            },
            'deportations': {
                'text': "Increasing deportations for those in the US illegally?",
                'type': 'Likert',
                'options': ['1 (Strongly Oppose)', '2 (Somewhat Oppose)', '3 (Neutral)', '4 (Somewhat Support)', '5 (Strongly Support)']
            },
            'medicare_vouchers': {
                'text': "Offering seniors healthcare vouchers to purchase private healthcare plans in place of traditional medicare coverage?",
                'type': 'Likert',
                'options': ['1 (Strongly Oppose)', '2 (Somewhat Oppose)', '3 (Neutral)', '4 (Somewhat Support)', '5 (Strongly Support)']
            }
        }
        
        # 有效响应值
        self.valid_responses = [1, 2, 3, 4, 5]
        
        # 改进的推理框架 - 针对政策议题设计
        self.reasoning_frameworks = {
            'carbon_tax': {
                'aspects': [
                    "Economic impact: How would a carbon tax affect my personal finances and job prospects?",
                    "Environmental values: Do I prioritize environmental protection over economic concerns?",
                    "Political ideology: Does my political leaning support government intervention for climate goals?",
                    "Regional context: How would this policy affect my local economy and community?"
                ]
            },
            'clean_energy': {
                'aspects': [
                    "Social equity: Do I support policies that specifically benefit low-income communities?",
                    "Government spending: Am I comfortable with federal investment in infrastructure?",
                    "Environmental priorities: How important is clean energy transition to me?",
                    "Economic fairness: Should government ensure equitable distribution of benefits?"
                ]
            },
            'clean_electricity': {
                'aspects': [
                    "Timeline feasibility: Is 2035 a realistic goal for carbon-free electricity?",
                    "Federal role: Should the federal government lead energy transformation?",
                    "Economic costs: Am I willing to accept higher energy costs for environmental benefits?",
                    "Technology confidence: Do I believe clean electricity technology is ready?"
                ]
            },
            'medicare_for_all': {
                'aspects': [
                    "Healthcare access: Would this improve healthcare access for people like me?",
                    "Government role: Am I comfortable with government-run healthcare?",
                    "Current insurance: How would this affect my current healthcare situation?",
                    "Cost concerns: Would this save or cost me money in healthcare expenses?"
                ]
            },
            'public_option': {
                'aspects': [
                    "Healthcare choice: Do I value having multiple healthcare options?",
                    "Government competition: Should government compete with private insurance?",
                    "Personal flexibility: Would a public option benefit my specific situation?",
                    "Market dynamics: How would this affect healthcare competition and quality?"
                ]
            },
            'immigration_reform': {
                'aspects': [
                    "Humanitarian values: Do I believe in providing pathways for undocumented immigrants?",
                    "Economic impact: How would immigration reform affect job opportunities in my area?",
                    "Cultural perspective: How do I view cultural diversity and integration?",
                    "Rule of law: How do I balance compassion with following immigration laws?"
                ]
            },
            'paid_family_leave': {
                'aspects': [
                    "Family priorities: How important is family time versus economic productivity?",
                    "Business impact: How would mandatory leave affect employers and the economy?",
                    "Personal relevance: Would this policy benefit me or my family directly?",
                    "Government mandates: Am I comfortable with government requiring business policies?"
                ]
            },
            'wealth_tax': {
                'aspects': [
                    "Income inequality: Do I think wealth inequality is a serious problem?",
                    "Tax philosophy: Should the wealthy pay higher tax rates?",
                    "Economic effects: Would this tax help or harm economic growth?",
                    "Fairness perception: Is this tax fair or would it discourage success?"
                ]
            },
            'deportations': {
                'aspects': [
                    "Immigration enforcement: How strictly should immigration laws be enforced?",
                    "Community safety: Do I see undocumented immigrants as a safety concern?",
                    "Humanitarian considerations: What about families and long-term residents?",
                    "Economic factors: How do undocumented immigrants affect local job markets?"
                ]
            },
            'medicare_vouchers': {
                'aspects': [
                    "Healthcare privatization: Do I prefer private or government healthcare management?",
                    "Senior care: What healthcare approach is best for elderly Americans?",
                    "Cost efficiency: Would vouchers provide better value than traditional Medicare?",
                    "Healthcare access: Would this improve or reduce healthcare access for seniors?"
                ]
            }
        }
        
        logger.info(f"数字孪生预测器初始化完成，使用{'Azure' if use_azure else 'Other'} OpenAI")
        if use_azure:
            logger.info(f"Azure配置: {self.deployment_name} @ archer222arc")

    def load_personas_from_parquet(self, num_personas: int = None) -> Dict[str, str]:
        """从parquet文件加载persona数据"""

        parquet_file = Path("data/personas_for_prediction.parquet")

        if not parquet_file.exists():
            logger.error(f"Parquet文件不存在: {parquet_file}")
            logger.info("请先运行 extract_personas.py 生成parquet文件")
            return {}

        try:
            logger.info(f"从parquet文件加载persona数据: {parquet_file}")
            df = pd.read_parquet(parquet_file)

            # 如果指定了数量，则随机选择
            if num_personas and num_personas < len(df):
                df = df.sample(n=num_personas, random_state=42).reset_index(drop=True)
                logger.info(f"随机选择了 {num_personas} 个personas")

            # 转换为字典格式
            personas = {}
            for _, row in df.iterrows():
                personas[row['persona_id']] = row['persona_text']

            logger.info(f"成功加载 {len(personas)} 个personas")
            return personas

        except Exception as e:
            logger.error(f"加载parquet文件失败: {e}")
            return {}

    def build_standardized_messages(self, persona_text: str, question_id: str) -> List[Dict[str, str]]:
        """
        构建标准化的消息格式，基于Digital-Twin-Simulation的prompt结构
        """
        question_config = self.questions[question_id]
        question_text = question_config['text']
        question_type = question_config['type']
        options = question_config['options']

        # 构建用户消息，严格按照参考格式
        user_parts = []

        # [Persona] 部分
        user_parts.append(f"[Persona]\n{persona_text.strip()}")

        # [Question] 部分
        question_part = f"[Question]\nType: {question_type}\nStem: {question_text.strip()}"

        # 添加选项
        if options:
            options_text = "\nOptions:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            question_part += options_text

        user_parts.append(question_part)

        # [Output format] 部分 - 严格指定输出格式
        output_format = """[Output format]
For choice questions: output only the chosen option text (e.g., "4 (Somewhat Support)").
For scales: output only the numeric/label value.
For numeric: output only the number.
For open-ended: 1-2 sentences only.

Please output only your answer, no other text."""

        user_parts.append(output_format)

        # 组合完整用户消息
        user_message = "\n\n".join(user_parts)

        return [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": user_message}
        ]

    def extract_response_value(self, response_text: str, question_id: str) -> Optional[int]:
        """从响应文本中提取数值"""
        response_text = response_text.strip()

        # 方法1: 直接匹配单个数字
        if response_text in ['1', '2', '3', '4', '5']:
            return int(response_text)

        # 方法2: 匹配带括号的格式，如 "4 (Somewhat Support)"
        pattern = r'^([1-5])\s*\('
        match = re.match(pattern, response_text)
        if match:
            return int(match.group(1))

        # 方法3: 从文本中提取数字
        numbers = re.findall(r'\b([1-5])\b', response_text)
        if numbers:
            return int(numbers[0])

        # 方法4: 关键词匹配
        response_lower = response_text.lower()
        if 'strongly oppose' in response_lower:
            return 1
        elif 'somewhat oppose' in response_lower:
            return 2
        elif 'neutral' in response_lower:
            return 3
        elif 'somewhat support' in response_lower:
            return 4
        elif 'strongly support' in response_lower:
            return 5

        logger.warning(f"无法从响应中提取有效值: '{response_text}' for {question_id}")
        return None
    
    def extract_persona_demographics(self, persona_text: str) -> Dict[str, str]:
        """从persona文本中提取关键人口统计学信息"""
        
        demographics = {}
        
        # 按行分析persona文本
        lines = persona_text.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # 提取关键字段
                key_lower = key.lower()
                if 'geographic' in key_lower or 'region' in key_lower:
                    demographics['region'] = value
                elif 'gender' in key_lower:
                    demographics['gender'] = value
                elif 'age' in key_lower:
                    demographics['age'] = value
                elif 'education' in key_lower:
                    demographics['education'] = value
                elif 'race' in key_lower:
                    demographics['race'] = value
                elif 'citizen' in key_lower:
                    demographics['citizenship'] = value
                elif 'marital' in key_lower:
                    demographics['marital'] = value
                elif 'religion' in key_lower and 'attendance' not in key_lower:
                    demographics['religion'] = value
                elif 'political affiliation' in key_lower:
                    demographics['political_party'] = value
                elif 'income' in key_lower:
                    demographics['income'] = value
                elif 'political views' in key_lower:
                    demographics['political_ideology'] = value
                elif 'household' in key_lower:
                    demographics['household_size'] = value
                elif 'employment' in key_lower:
                    demographics['employment'] = value
        
        return demographics
    
    def build_persona_prompt(self, persona_text: str, method: str = "basic") -> str:
        """构建基于persona的提示"""
        
        if method == "basic":
            return f"""You are role-playing as a person with the following characteristics:

{persona_text}

You will be asked about your opinion on various political policies. Answer as this person would, based on their demographic background, political views, and personal circumstances. The survey is taking place in May 2024."""
        
        elif method == "structured":
            demographics = self.extract_persona_demographics(persona_text)
            
            # 构建结构化描述
            structured_desc = "I am a person with the following characteristics:\n"
            for key, value in demographics.items():
                structured_desc += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            return f"""{structured_desc}

Based on my demographic background and personal circumstances, I will answer policy questions as this person would in May 2024."""
        
        else:  # detailed method
            return f"""Persona Profile:
{persona_text}

As this person, I will consider how my specific demographic background, political views, economic situation, and personal values would influence my opinions on policy questions. The survey takes place in May 2024, and I will answer authentically from this person's perspective."""
    
    def create_cot_prompt(self, persona_text: str, question_id: str, question_text: str) -> str:
        """创建针对数字孪生的CoT提示"""
        
        aspects = self.reasoning_frameworks[question_id]['aspects']
        aspect_text = "\n".join([f"{i+1}. {aspect}" for i, aspect in enumerate(aspects)])
        
        persona_prompt = self.build_persona_prompt(persona_text, "detailed")
        
        cot_prompt = f"""{persona_prompt}

Question: Would you support or oppose {question_text}

I need to think through this step by step, considering how my background influences my views:

{aspect_text}

Based on this reasoning and my specific demographic profile, my level of support is: """
        
        return cot_prompt
    
    def predict_single_basic(self, persona_text: str, question_id: str, max_retries: int = 3) -> Optional[int]:
        """基础预测方法 - 使用原始notebook的prompt格式"""
        
        for attempt in range(max_retries):
            try:
                question_text = self.questions[question_id]
                
                # 使用原始notebook中的USER_PROMPT_TEMPLATE格式
                user_prompt = f"""
{persona_text}

Answer the following question as the persona would answer it. There are five possible answers: 1 (Strongly Oppose), 2 (Somewhat Oppose), 3 (Neutral), 4 (Somewhat Support), 5 (Strongly Support). Return only the number (1, 2, 3, 4, or 5) as your answer.

"{question_text}"

FORMAT INSTRUCTIONS: Only return the number, no other text.
"""
                
                messages = [
                    {"role": "system", "content": self.SYSTEM_MESSAGE},
                    {"role": "user", "content": user_prompt}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_completion_tokens=10000  # 使用原始notebook的设置，移除temperature参数
                )
                
                prediction = response.choices[0].message.content.strip()
                
                # 验证预测结果
                try:
                    pred_num = int(prediction)
                    if pred_num in self.valid_responses:
                        return pred_num
                    else:
                        logger.warning(f"Invalid prediction {pred_num} for {question_id}, attempt {attempt+1}")
                except ValueError:
                    logger.warning(f"Non-numeric prediction: {prediction} for {question_id}, attempt {attempt+1}")
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # 指数退避
                    logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                    
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                logger.warning(f"API error for {question_id}, attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # 指数退避
                    logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"达到最大重试次数，放弃 {question_id}")
                    
            except Exception as e:
                logger.error(f"Basic prediction failed for {question_id}, attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = min(5, 1 + attempt)  # 渐进式延迟，最大5秒
                    time.sleep(wait_time)
        
        return None
    
    def predict_single_cot(self, persona_text: str, question_id: str, max_retries: int = 3) -> Optional[int]:
        """CoT预测方法 - 使用原始SYSTEM_MESSAGE + CoT推理"""
        
        for attempt in range(max_retries):
            try:
                question_text = self.questions[question_id]
                aspects = self.reasoning_frameworks[question_id]['aspects']
                aspect_text = "\n".join([f"{i+1}. {aspect}" for i, aspect in enumerate(aspects)])
                
                # 使用原始SYSTEM_MESSAGE，但在user prompt中添加CoT推理
                user_prompt = f"""
{persona_text}

Answer the following question as the persona would answer it. There are five possible answers: 1 (Strongly Oppose), 2 (Somewhat Oppose), 3 (Neutral), 4 (Somewhat Support), 5 (Strongly Support).

Question: "{question_text}"

Before answering, think through the following aspects based on your persona:
{aspect_text}

After considering these aspects from your persona's perspective, return only the number (1, 2, 3, 4, or 5) as your answer.

FORMAT INSTRUCTIONS: Only return the number, no other text.
"""
                
                messages = [
                    {"role": "system", "content": self.SYSTEM_MESSAGE},
                    {"role": "user", "content": user_prompt}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_completion_tokens=10000  # 保持与base版本一致，移除temperature和top_p参数
                )
                
                full_response = response.choices[0].message.content.strip()
                
                # 提取最终答案 - 寻找最后出现的数字
                numbers = re.findall(r'\b([1-5])\b', full_response)
                
                if numbers:
                    pred_num = int(numbers[-1])
                    if pred_num in self.valid_responses:
                        return pred_num
                    else:
                        logger.warning(f"Invalid CoT prediction {pred_num} for {question_id}, attempt {attempt+1}")
                else:
                    logger.warning(f"No valid number found in CoT response for {question_id}, attempt {attempt+1}")
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # 指数退避
                    logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                    
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                logger.warning(f"API error for {question_id}, attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # 指数退避
                    logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"达到最大重试次数，放弃 {question_id}")
                    
            except Exception as e:
                logger.error(f"CoT prediction failed for {question_id}, attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = min(5, 1 + attempt)  # 渐进式延迟，最大5秒
                    time.sleep(wait_time)
        
        return None
    
    def predict_persona_all_questions(self, persona_id: str, persona_text: str, method: str = "base", max_retries: int = 5) -> Dict[str, Optional[int]]:
        """对单个persona预测所有问题"""
        
        results = {}
        
        for question_id in self.questions.keys():
            if method == "base":
                prediction = self.predict_single_basic(persona_text, question_id, max_retries)
            elif method == "cot":
                prediction = self.predict_single_cot(persona_text, question_id, max_retries)
            else:
                logger.error(f"Unknown prediction method: {method}. Use 'base' or 'cot'")
                prediction = None
            
            results[question_id] = prediction
            
            # 移除固定延迟，依赖重试机制处理API限制
        
        logger.info(f"完成persona {persona_id} 的所有预测 (方法: {method})")
        return results
    
    def predict_batch_threaded(self, personas: Dict[str, str], method: str = "base", max_workers: int = 5, max_retries: int = 5):
        """使用线程池进行批量预测"""
        
        all_results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_persona = {
                executor.submit(self.predict_persona_all_questions, persona_id, persona_text, method, max_retries): persona_id
                for persona_id, persona_text in personas.items()
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_persona):
                persona_id = future_to_persona[future]
                try:
                    result = future.result()
                    all_results[persona_id] = result
                    completed += 1
                    
                    logger.info(f"已完成 {completed}/{len(personas)} 个persona预测")
                    
                except Exception as e:
                    logger.error(f"Persona {persona_id} 预测失败: {e}")
                    all_results[persona_id] = {q: None for q in self.questions.keys()}
        
        return all_results
    
    def process_personas_dataset(self, personas: Dict[str, str], method: str = "base", max_workers: int = 5, max_retries: int = 5):
        """处理完整的personas数据集"""
        
        logger.info(f"开始处理 {len(personas)} 个personas，使用方法: {method}")
        
        start_time = time.time()
        
        # 批量预测
        all_results = self.predict_batch_threaded(personas, method, max_workers, max_retries)
        
        end_time = time.time()
        
        logger.info(f"预测完成，耗时: {(end_time - start_time)/60:.1f} 分钟")
        
        # 转换为DataFrame格式
        df_results = []
        for persona_id, predictions in all_results.items():
            row = {'persona_id': persona_id}
            for question_id, prediction in predictions.items():
                row[f'{question_id}_enhanced'] = prediction
            df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        return df
    
    def save_results(self, df: pd.DataFrame, personas: Dict[str, str], method: str, filename: str = None):
        """保存预测结果"""
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'enhanced_digital_twin_predictions_{method}_{timestamp}.csv'
        
        df.to_csv(filename, index=False)
        logger.info(f"预测结果已保存: {filename}")
        
        # 保存统计信息
        stats = {
            'method': method,
            'total_personas': len(personas),
            'total_questions': len(self.questions),
            'questions': list(self.questions.keys())
        }
        
        # 计算每个问题的预测成功率
        for question_id in self.questions.keys():
            col_name = f'{question_id}_enhanced'
            if col_name in df.columns:
                predicted = df[col_name].notna().sum()
                success_rate = predicted / len(df) if len(df) > 0 else 0

                stats[f'{question_id}_success_rate'] = success_rate

        # 总体成功率
        enhanced_cols = [col for col in df.columns if col.endswith('_enhanced')]
        if enhanced_cols:
            overall_success = df[enhanced_cols].notna().values.mean()
            stats['overall_success_rate'] = overall_success
        
        stats_filename = filename.replace('.csv', '_stats.json')
        with open(stats_filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"预测统计信息已保存: {stats_filename}")
        
        return filename

def main():
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python digital_twin_prediction.py base [num_personas] [max_workers] [max_retries]")
        print("  python digital_twin_prediction.py cot [num_personas] [max_workers] [max_retries]")
        print("")
        print("参数:")
        print("  method: base 或 cot")
        print("    base: 基础方法，使用标准化Digital-Twin-Simulation prompt结构")
        print("    cot: Chain-of-Thought方法，在标准化prompt基础上添加推理步骤")
        print("  num_personas: 处理的persona数量 (-1 表示全部, 默认: 30)")
        print("  max_workers: 线程数 (默认: 8)")
        print("  max_retries: 每个API调用的最大重试次数 (默认: 5)")
        print("")
        print("新特性:")
        print("  - 使用标准化的Digital-Twin-Simulation prompt结构")
        print("  - 从parquet文件高效加载persona数据")
        print("  - 严格的输出格式控制和数值提取")
        print("  - 增强的错误处理和重试机制")
        print("")
        print("示例:")
        print("  python digital_twin_prediction.py base 30")
        print("  python digital_twin_prediction.py cot -1  # 全部personas")
        return
    
    method = sys.argv[1]
    if method not in ['base', 'cot']:
        print("错误: method 必须是 'base' 或 'cot'")
        return
    
    num_personas = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8  # 提高默认并发数
    max_retries = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    
    predictor = DigitalTwinPredictor()
    
    logger.info(f"开始增强数字孪生预测 (方法: {method}, persona数: {num_personas}, 线程数: {max_workers}, 重试次数: {max_retries})")
    logger.info("使用标准化Digital-Twin-Simulation prompt结构和parquet数据加载")
    
    try:
        # 加载personas数据
        logger.info("从parquet文件加载personas数据...")

        # 处理全量标记
        load_num = None if num_personas == -1 else num_personas
        personas = predictor.load_personas_from_parquet(load_num)

        if not personas:
            logger.error("未能加载personas数据")
            logger.info("请确保已运行 extract_personas.py 生成parquet数据文件")
            return
        
        # 执行预测
        df_results = predictor.process_personas_dataset(personas, method, max_workers, max_retries)
        
        # 保存结果
        output_file = predictor.save_results(df_results, personas, method)
        
        print(f"\n✅ 增强数字孪生预测完成!")
        print(f"方法: {method} (标准化prompt)")
        print(f"处理的personas: {len(personas)}")
        print(f"并发线程数: {max_workers}")
        print(f"重试次数: {max_retries}")
        print(f"结果文件: {output_file}")

        # 显示成功率摘要
        enhanced_cols = [col for col in df_results.columns if col.endswith('_enhanced')]
        if enhanced_cols:
            overall_success = df_results[enhanced_cols].notna().values.mean()
            print(f"总体成功率: {overall_success:.1%}")
        
    except Exception as e:
        logger.error(f"预测失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()