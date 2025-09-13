#!/usr/bin/env python3
"""
对比微调模型与基线模型的预测效果
使用基础提示方法测试同样的 NPORS 数据
"""

import json
import pandas as pd
import numpy as np
from openai import AzureOpenAI
import time
from typing import Dict, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NPORSComparison:
    """NPORS 预测对比分析器"""
    
    def __init__(self):
        # 读取配置
        with open("config/azure_models_config.json", 'r') as f:
            config = json.load(f)
        
        # 微调模型客户端
        finetuned_config = config["azure_endpoints"]["deployed_finetuned"]
        self.finetuned_client = AzureOpenAI(
            api_version=finetuned_config["api_version"],
            azure_endpoint=finetuned_config["endpoint"],
            api_key=finetuned_config["api_key"]
        )
        self.finetuned_deployment = finetuned_config["deployment_name"]
        
        print(f"✅ 微调模型已连接: {self.finetuned_deployment}")
        
        # 问题映射 (与原始LLM_prediction保持一致)
        self.questions = {
            'ECON1MOD': "How would you rate the economic conditions in your community today? 1. Excellent, 2. Good, 3. Only fair, 4. Poor.",
            'UNITY': "Which statement comes closer to your own view, even if neither is exactly right? 1. Americans are united when it comes to the most important values. 2. Americans are divided when it comes to the most important values.",
            'GPT1': "Have you heard of ChatGPT? 1. Yes, 2. No, 3. Not sure.",
            'MOREGUNIMPACT': "If more Americans owned guns, do you think there would be... 1. More crime, 2. Less crime, 3. No difference.",
            'GAMBLERESTR': "How much government regulation of gambling do you favor? 1. A lot more than now, 2. A little more than now, 3. About the same as now, 4. A little less than now, 5. A lot less than now."
        }
        
        # 响应范围映射
        self.response_ranges = {
            'ECON1MOD': [1, 2, 3, 4],
            'UNITY': [1, 2],
            'GPT1': [1, 2, 3],
            'MOREGUNIMPACT': [1, 2, 3],
            'GAMBLERESTR': [1, 2, 3, 4, 5]
        }
    
    def build_demographic_prompt(self, row):
        """构建人口统计背景提示 (与原始LLM_prediction保持一致)"""
        
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
    
    def predict_with_finetuned(self, row, question_id: str) -> str:
        """使用微调模型预测"""
        
        system_prompt = self.build_demographic_prompt(row)
        question_text = self.questions[question_id]
        user_prompt = f"Question: {question_text}\nPlease output the number only."
        
        try:
            response = self.finetuned_client.chat.completions.create(
                model=self.finetuned_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
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
                    logger.warning(f"Invalid prediction {pred_num} for {question_id}")
                    return None
            except ValueError:
                logger.warning(f"Non-numeric prediction: {prediction}")
                return None
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def run_comparison_sample(self, sample_size: int = 100):
        """在样本数据上运行对比测试"""
        
        # 加载原始数据
        logger.info("Loading NPORS data...")
        df = pd.read_csv('data/NPORS_2024_for_public_release_updated.csv')
        
        # 加载基线预测结果
        logger.info("Loading baseline predictions...")
        baseline_df = pd.read_csv('data/NPORS_2024_for_public_release_basic_prompting.csv')
        
        # 确保有基线结果的数据
        baseline_respids = set(baseline_df['RESPID'].values)
        df_with_baseline = df[df['RESPID'].isin(baseline_respids)].copy()
        
        # 随机采样
        if len(df_with_baseline) > sample_size:
            df_sample = df_with_baseline.sample(n=sample_size, random_state=42)
        else:
            df_sample = df_with_baseline.copy()
        
        logger.info(f"Testing on {len(df_sample)} samples")
        
        # 为每个问题运行预测
        for question_id in self.questions.keys():
            logger.info(f"Predicting {question_id}...")
            
            finetuned_predictions = []
            
            for idx, row in df_sample.iterrows():
                if pd.isna(row[question_id]) or row[question_id] == 99.0:
                    finetuned_predictions.append(None)
                    continue
                
                # 使用微调模型预测
                pred = self.predict_with_finetuned(row, question_id)
                finetuned_predictions.append(pred)
                
                # 避免API限制
                time.sleep(0.2)
            
            # 添加到结果中
            df_sample[f'{question_id}_FINETUNED'] = finetuned_predictions
        
        return df_sample
    
    def analyze_comparison(self, df_results):
        """分析对比结果"""
        
        results = {}
        
        for question_id in self.questions.keys():
            baseline_col = f'{question_id}_LLM'
            finetuned_col = f'{question_id}_FINETUNED'
            actual_col = question_id
            
            # 过滤有效数据
            valid_mask = (
                df_results[actual_col].notna() & 
                (df_results[actual_col] != 99.0) &
                df_results[baseline_col].notna() &
                df_results[finetuned_col].notna()
            )
            
            if valid_mask.sum() == 0:
                continue
            
            actual = df_results.loc[valid_mask, actual_col].astype(int)
            baseline = df_results.loc[valid_mask, baseline_col].astype(int) 
            finetuned = df_results.loc[valid_mask, finetuned_col].astype(int)
            
            # 计算准确率
            baseline_accuracy = (actual == baseline).mean()
            finetuned_accuracy = (actual == finetuned).mean()
            
            # 计算分布相似性 (KL散度)
            def calculate_distribution_similarity(actual, predicted):
                actual_dist = pd.Series(actual).value_counts(normalize=True).sort_index()
                pred_dist = pd.Series(predicted).value_counts(normalize=True).reindex(actual_dist.index, fill_value=0.001)
                
                # KL散度 (越小越好)
                kl_div = np.sum(actual_dist * np.log(actual_dist / pred_dist))
                return kl_div
            
            baseline_kl = calculate_distribution_similarity(actual, baseline)
            finetuned_kl = calculate_distribution_similarity(actual, finetuned)
            
            results[question_id] = {
                'sample_size': len(actual),
                'baseline_accuracy': baseline_accuracy,
                'finetuned_accuracy': finetuned_accuracy,
                'accuracy_improvement': finetuned_accuracy - baseline_accuracy,
                'baseline_kl_divergence': baseline_kl,
                'finetuned_kl_divergence': finetuned_kl,
                'kl_improvement': baseline_kl - finetuned_kl
            }
        
        return results
    
    def print_comparison_report(self, results):
        """打印对比报告"""
        
        print("\n" + "="*80)
        print("微调模型 vs 基线模型对比报告")
        print("="*80)
        
        for question_id, metrics in results.items():
            print(f"\n📊 {question_id} - {self.questions[question_id][:50]}...")
            print(f"   样本数量: {metrics['sample_size']}")
            print(f"   基线准确率: {metrics['baseline_accuracy']:.3f}")
            print(f"   微调准确率: {metrics['finetuned_accuracy']:.3f}")
            print(f"   准确率提升: {metrics['accuracy_improvement']:+.3f} ({metrics['accuracy_improvement']*100:+.1f}%)")
            print(f"   基线KL散度: {metrics['baseline_kl_divergence']:.3f}")
            print(f"   微调KL散度: {metrics['finetuned_kl_divergence']:.3f}")
            print(f"   分布相似性提升: {metrics['kl_improvement']:+.3f}")
            
            if metrics['accuracy_improvement'] > 0:
                print("   ✅ 微调模型更准确")
            elif metrics['accuracy_improvement'] < 0:
                print("   ⚠️ 基线模型更准确")
            else:
                print("   ➖ 准确率相同")
        
        # 总体统计
        total_baseline_acc = np.mean([m['baseline_accuracy'] for m in results.values()])
        total_finetuned_acc = np.mean([m['finetuned_accuracy'] for m in results.values()])
        total_improvement = total_finetuned_acc - total_baseline_acc
        
        print(f"\n🎯 总体表现:")
        print(f"   平均基线准确率: {total_baseline_acc:.3f}")
        print(f"   平均微调准确率: {total_finetuned_acc:.3f}")
        print(f"   平均准确率提升: {total_improvement:+.3f} ({total_improvement*100:+.1f}%)")
        
        if total_improvement > 0.01:
            print("   🎉 微调显著提升了预测效果!")
        elif total_improvement > 0:
            print("   ✅ 微调略微提升了预测效果")
        else:
            print("   ❌ 微调没有提升预测效果")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python 5_compare_finetuned_vs_baseline.py run [sample_size]    # 运行对比测试")
        print("  python 5_compare_finetuned_vs_baseline.py analyze              # 分析已有结果")
        return
    
    action = sys.argv[1]
    
    comparator = NPORSComparison()
    
    if action == "run":
        sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        
        print(f"运行对比测试 (样本数: {sample_size})")
        
        # 运行对比
        df_results = comparator.run_comparison_sample(sample_size)
        
        # 保存结果
        output_file = f"comparison_results_sample_{sample_size}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"结果已保存到: {output_file}")
        
        # 分析结果
        results = comparator.analyze_comparison(df_results)
        
        # 打印报告
        comparator.print_comparison_report(results)
        
        # 保存分析结果
        with open(f"comparison_analysis_sample_{sample_size}.json", "w") as f:
            json.dump(results, f, indent=2)
    
    elif action == "analyze":
        # 分析现有结果文件
        try:
            df_results = pd.read_csv("comparison_results_sample_50.csv")
            results = comparator.analyze_comparison(df_results)
            comparator.print_comparison_report(results)
        except FileNotFoundError:
            print("未找到结果文件，请先运行 'run' 命令")

if __name__ == "__main__":
    main()