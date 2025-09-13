#!/usr/bin/env python3
"""
简单测试微调模型连接
"""

import json
from openai import AzureOpenAI

def test_simple_connection():
    """测试基本连接"""
    
    # 读取配置
    with open("config/azure_models_config.json", 'r') as f:
        config = json.load(f)
    
    azure_config = config["azure_endpoints"]["deployed_finetuned"]
    
    client = AzureOpenAI(
        api_version=azure_config["api_version"],
        azure_endpoint=azure_config["endpoint"],
        api_key=azure_config["api_key"]
    )
    
    print(f"测试连接到: {azure_config['endpoint']}")
    print(f"部署名称: {azure_config['deployment_name']}")
    
    try:
        # 简单的测试请求
        response = client.chat.completions.create(
            model=azure_config["deployment_name"],
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            max_completion_tokens=5,
            temperature=0.0
        )
        
        print("✅ 连接成功!")
        print(f"响应: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

def test_npors_prediction():
    """测试 NPORS 预测"""
    
    # 读取配置
    with open("config/azure_models_config.json", 'r') as f:
        config = json.load(f)
    
    azure_config = config["azure_endpoints"]["deployed_finetuned"]
    
    client = AzureOpenAI(
        api_version=azure_config["api_version"],
        azure_endpoint=azure_config["endpoint"],
        api_key=azure_config["api_key"]
    )
    
    system_prompt = """You are a respondent in a survey at the time of May 1st, 2024. You are a 35-year-old Woman who is Not Hispanic, White. You were born in 50 U.S. states or D.C., and are currently Married. You have an education level of Bachelor's degree. Your annual household income is $50,000–59,999. You live in the Pacific (AK, CA, HI, OR, WA) region and are located in a Metropolitan area. Answer survey questions based on your demographic profile and personal circumstances. Be realistic and consistent with your background."""
    
    user_prompt = "Question: If more Americans owned guns, do you think there would be... 1. More crime, 2. Less crime, 3. No difference.\nPlease output the number only."
    
    try:
        print("\n测试 NPORS 预测...")
        response = client.chat.completions.create(
            model=azure_config["deployment_name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=5,
            temperature=0.0
        )
        
        prediction = response.choices[0].message.content.strip()
        print(f"✅ NPORS 预测成功!")
        print(f"预测回答: {prediction}")
        
        # 解释预测
        if prediction == "1":
            print("解释: 预测认为更多美国人拥有枪支会导致更多犯罪")
        elif prediction == "2":
            print("解释: 预测认为更多美国人拥有枪支会导致更少犯罪")
        elif prediction == "3":
            print("解释: 预测认为更多美国人拥有枪支不会有什么区别")
        
        return True
        
    except Exception as e:
        print(f"❌ NPORS 预测失败: {e}")
        return False

if __name__ == "__main__":
    print("=== 测试微调模型部署 ===")
    
    # 先测试基本连接
    if test_simple_connection():
        # 再测试 NPORS 预测
        test_npors_prediction()
    else:
        print("基本连接失败，请检查配置")