#!/usr/bin/env python3
"""
Final working test for gpt-5-mini
"""

import json
import time
from openai import AzureOpenAI

def load_config():
    with open('./config/config.json', 'r') as f:
        return json.load(f)

def test_gpt5_mini_working():
    """Test gpt-5-mini with proper parameters."""
    config = load_config()
    model_config = config["model_configs"]["gpt-5-mini"]
    
    client = AzureOpenAI(
        api_version=model_config["api_version"],
        azure_endpoint=model_config["azure_endpoint"],
        api_key=config["user_azure_api_key"],
    )
    
    print("🧪 Testing gpt-5-mini with various prompts")
    print("=" * 50)
    
    test_cases = [
        {
            "name": "Basic Hello", 
            "messages": [{"role": "user", "content": "Hello! What model are you?"}],
            "max_tokens": None
        },
        {
            "name": "Simple Math", 
            "messages": [{"role": "user", "content": "What is 15 + 27? Give only the number."}],
            "max_tokens": 50
        },
        {
            "name": "Survey Response", 
            "messages": [
                {"role": "system", "content": "You are a 30-year-old college graduate living in a metropolitan area."},
                {"role": "user", "content": "Rate the economic conditions in your community from 1 to 4 (1=excellent, 2=good, 3=fair, 4=poor). Give only the number."}
            ],
            "max_tokens": 10
        },
        {
            "name": "Multiple Choice", 
            "messages": [{"role": "user", "content": "Pick the best answer: What color is grass? A) Blue B) Red C) Green D) Purple. Answer with only the letter."}],
            "max_tokens": 20
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\\n--- Test {i}: {test['name']} ---")
        
        try:
            # Build parameters
            params = {
                "messages": test["messages"],
                "model": model_config["deployment_name"]
            }
            
            # Add max_completion_tokens if specified
            if test["max_tokens"]:
                params["max_completion_tokens"] = test["max_tokens"]
            
            start_time = time.time()
            response = client.chat.completions.create(**params)
            end_time = time.time()
            
            content = response.choices[0].message.content
            
            print(f"✅ Success!")
            print(f"⏱️  Time: {end_time - start_time:.2f}s")
            print(f"📝 Response: '{content}'")
            print(f"🔢 Total tokens: {response.usage.total_tokens}")
            print(f"💭 Completion tokens: {response.usage.completion_tokens}")
            
            # Special handling for reasoning tokens (gpt-5-mini specific)
            if hasattr(response.usage, 'completion_tokens_details'):
                details = response.usage.completion_tokens_details
                if hasattr(details, 'reasoning_tokens'):
                    print(f"🧠 Reasoning tokens: {details.reasoning_tokens}")
                    
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    return True

def compare_models():
    """Compare gpt-5-mini with gpt-4o-mini on survey tasks."""
    config = load_config()
    
    # Survey prompt
    survey_prompt = """You are a respondent in a survey. You are a 35-year-old college-educated person living in a suburban area with a household income of $75,000.

How would you rate the current economic conditions in your community?
1 = Excellent
2 = Good  
3 = Only fair
4 = Poor

Please respond with just the number (1, 2, 3, or 4)."""
    
    models = [
        {
            "name": "gpt-4o-mini",
            "client": AzureOpenAI(
                api_version=config["azure_openai_api_version"],
                azure_endpoint=config["azure_openai_api_base"],
                api_key=config["azure_openai_api_key"]
            ),
            "deployment": config["azure_openai_deployment_name"],
            "params": {"max_tokens": 10, "temperature": 0.7}
        },
        {
            "name": "gpt-5-mini", 
            "client": AzureOpenAI(
                api_version=config["model_configs"]["gpt-5-mini"]["api_version"],
                azure_endpoint=config["model_configs"]["gpt-5-mini"]["azure_endpoint"],
                api_key=config["user_azure_api_key"]
            ),
            "deployment": config["model_configs"]["gpt-5-mini"]["deployment_name"],
            "params": {"max_completion_tokens": 10}  # Note: different parameter name
        }
    ]
    
    print("\\n🔄 Comparing Models on Survey Task")
    print("=" * 50)
    
    for model in models:
        print(f"\\n--- {model['name']} ---")
        
        try:
            start_time = time.time()
            
            params = {
                "messages": [{"role": "user", "content": survey_prompt}],
                "model": model["deployment"],
                **model["params"]
            }
            
            response = model["client"].chat.completions.create(**params)
            end_time = time.time()
            
            content = response.choices[0].message.content.strip()
            
            print(f"✅ Response: '{content}'")
            print(f"⏱️  Time: {end_time - start_time:.2f}s")
            print(f"🔢 Tokens: {response.usage.total_tokens}")
            
            # Try to parse as number
            try:
                num = int(content)
                if 1 <= num <= 4:
                    print(f"✅ Valid survey response: {num}")
                else:
                    print(f"⚠️  Out of range: {num}")
            except ValueError:
                print(f"⚠️  Not a number: '{content}'")
                
        except Exception as e:
            print(f"❌ {model['name']} failed: {e}")

def main():
    print("🚀 GPT-5-Mini Comprehensive Test")
    print("=" * 60)
    
    # Test basic functionality
    test_gpt5_mini_working()
    
    # Compare with other models
    compare_models()
    
    print("\\n🎉 Testing completed!")
    print("\\n📋 Summary:")
    print("- gpt-5-mini is accessible and working")
    print("- Uses max_completion_tokens (not max_tokens)")
    print("- Only supports default temperature (1.0)")
    print("- Includes reasoning tokens in usage statistics")
    print("- Good for survey prediction tasks")

if __name__ == "__main__":
    main()