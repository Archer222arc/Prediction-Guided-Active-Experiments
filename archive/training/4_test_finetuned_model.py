#!/usr/bin/env python3
"""
测试微调后的 NPORS 调查预测模型
"""

import json
import time
from openai import AzureOpenAI
import pandas as pd

class NPORSPredictor:
    """NPORS 调查回答预测器"""
    
    def __init__(self):
        # 读取配置
        with open("config/azure_models_config.json", 'r') as f:
            config = json.load(f)
        
        azure_config = config["azure_endpoints"]["deployed_finetuned"]
        
        self.client = AzureOpenAI(
            api_version=azure_config["api_version"],
            azure_endpoint=azure_config["endpoint"],
            api_key=azure_config["api_key"]
        )
        
        # 部署的模型名称
        self.deployment_name = azure_config["deployment_name"]
        
        print(f"✅ NPORS预测器已初始化")
        print(f"使用部署: {self.deployment_name}")
        print(f"端点: {azure_config['endpoint']}")
    
    def build_demographic_prompt(self, demographics):
        """构建人口统计背景提示"""
        return f"""You are a respondent in a survey at the time of May 1st, 2024. You are a {demographics['age']}-year-old {demographics['gender']} who is {demographics['race']}. You were born in {demographics['birth_place']}, and are currently {demographics['marital_status']}. You have an education level of {demographics['education']}. Your annual household income is {demographics['income']}. You live in the {demographics['region']} region and are located in a {demographics['area_type']} area. Answer survey questions based on your demographic profile and personal circumstances. Be realistic and consistent with your background."""
    
    def predict_response(self, demographics, question):
        """预测特定人口统计背景下的调查回答"""
        
        system_prompt = self.build_demographic_prompt(demographics)
        user_prompt = f"Question: {question}\nPlease output the number only."
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=10,
                temperature=0.1
            )
            
            prediction = response.choices[0].message.content.strip()
            return prediction
            
        except Exception as e:
            print(f"预测失败: {e}")
            return None
    
    def batch_predict(self, demographics_list, question):
        """批量预测多个人群的回答"""
        predictions = []
        
        for i, demographics in enumerate(demographics_list):
            print(f"预测第 {i+1}/{len(demographics_list)} 个人群...")
            prediction = self.predict_response(demographics, question)
            
            result = {
                "demographics": demographics,
                "question": question,
                "prediction": prediction
            }
            predictions.append(result)
            
            # 避免API限制
            time.sleep(0.5)
        
        return predictions

def create_test_demographics():
    """创建测试用的人口统计样本"""
    test_cases = [
        {
            "name": "年轻自由派",
            "age": 25,
            "gender": "Woman",
            "race": "Not Hispanic, White", 
            "birth_place": "50 U.S. states or D.C.",
            "marital_status": "Single",
            "education": "Bachelor's degree",
            "income": "$40,000–49,999",
            "region": "Pacific (AK, CA, HI, OR, WA)",
            "area_type": "Metropolitan area"
        },
        {
            "name": "中年保守派",
            "age": 55,
            "gender": "Man",
            "race": "Not Hispanic, White",
            "birth_place": "50 U.S. states or D.C.", 
            "marital_status": "Married",
            "education": "High school graduate",
            "income": "$75,000–99,999",
            "region": "West South Central (AR, LA, OK, TX)",
            "area_type": "Non-metropolitan area"
        },
        {
            "name": "高学历老年人",
            "age": 72,
            "gender": "Woman",
            "race": "Not Hispanic, White",
            "birth_place": "50 U.S. states or D.C.",
            "marital_status": "Widowed", 
            "education": "Postgraduate degree",
            "income": "$100,000+",
            "region": "New England (CT, ME, MA, NH, RI, VT)",
            "area_type": "Metropolitan area"
        },
        {
            "name": "年轻少数族裔",
            "age": 30,
            "gender": "Man",
            "race": "Hispanic",
            "birth_place": "Another country other than U.S.",
            "marital_status": "Married",
            "education": "Some college",
            "income": "$30,000–39,999", 
            "region": "Mountain (AZ, CO, ID, MT, NV, NM, UT, WY)",
            "area_type": "Metropolitan area"
        }
    ]
    
    return test_cases

def test_gun_ownership_question():
    """测试枪支拥有权问题"""
    predictor = NPORSPredictor()
    test_demographics = create_test_demographics()
    
    question = "If more Americans owned guns, do you think there would be... 1. More crime, 2. Less crime, 3. No difference."
    
    print(f"\n=== 测试问题: 枪支拥有权态度 ===")
    print(f"问题: {question}")
    print("1 = 更多犯罪, 2 = 更少犯罪, 3 = 没有区别")
    
    predictions = predictor.batch_predict(test_demographics, question)
    
    print(f"\n=== 预测结果 ===")
    for pred in predictions:
        demo = pred["demographics"]
        print(f"\n👤 {demo['name']}:")
        print(f"   {demo['age']}岁 {demo['gender']}, {demo['education']}, {demo['income']}")
        print(f"   地区: {demo['region']}")
        print(f"   预测回答: {pred['prediction']}")
    
    return predictions

def test_american_values_question():
    """测试美国价值观问题"""
    predictor = NPORSPredictor()
    test_demographics = create_test_demographics()
    
    question = "Which statement comes closer to your own view, even if neither is exactly right? 1. Americans are united when it comes to the most important values. 2. Americans are divided when it comes to the most important values."
    
    print(f"\n=== 测试问题: 美国价值观统一性 ===")
    print(f"问题: {question}")
    print("1 = 美国人在重要价值观上是统一的, 2 = 美国人在重要价值观上是分裂的")
    
    predictions = predictor.batch_predict(test_demographics, question)
    
    print(f"\n=== 预测结果 ===")
    for pred in predictions:
        demo = pred["demographics"]
        print(f"\n👤 {demo['name']}:")
        print(f"   {demo['age']}岁 {demo['gender']}, {demo['education']}, {demo['income']}")
        print(f"   地区: {demo['region']}")
        print(f"   预测回答: {pred['prediction']}")
    
    return predictions

def analyze_training_results():
    """分析训练结果"""
    try:
        df = pd.read_csv('npors_training_results_file-6fdf449e305a4a3c97d2fe3c1d83d32b.csv')
        
        print("=== 训练结果分析 ===")
        print(f"总训练步数: {df['step'].max():,}")
        print(f"最终训练损失: {df['train_loss'].iloc[-1]:.4f}")
        
        if 'valid_loss' in df.columns:
            valid_losses = df['valid_loss'].dropna()
            if len(valid_losses) > 0:
                print(f"最终验证损失: {valid_losses.iloc[-1]:.4f}")
        
        if 'train_mean_token_accuracy' in df.columns:
            accuracies = df['train_mean_token_accuracy'].dropna()
            if len(accuracies) > 0:
                print(f"最终训练准确率: {accuracies.iloc[-1]:.4f}")
        
        # 检查损失趋势
        recent_losses = df['train_loss'].tail(100).mean()
        print(f"最近100步平均损失: {recent_losses:.4f}")
        
    except Exception as e:
        print(f"分析训练结果失败: {e}")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python 4_test_finetuned_model.py gun        # 测试枪支问题")
        print("  python 4_test_finetuned_model.py values     # 测试价值观问题")
        print("  python 4_test_finetuned_model.py analysis   # 分析训练结果")
        print("  python 4_test_finetuned_model.py all        # 运行所有测试")
        return
    
    action = sys.argv[1]
    
    try:
        if action == "gun":
            test_gun_ownership_question()
        elif action == "values":
            test_american_values_question()
        elif action == "analysis":
            analyze_training_results()
        elif action == "all":
            analyze_training_results()
            test_gun_ownership_question() 
            test_american_values_question()
        else:
            print(f"未知操作: {action}")
    
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()