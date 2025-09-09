#!/usr/bin/env python3
"""
Test script for gpt-5-mini model

This script tests the gpt-5-mini model using your existing configuration.
"""

import json
import time
from openai import AzureOpenAI

def load_config():
    """Load configuration from config.json."""
    try:
        with open('./config/config.json', 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def test_gpt5_mini():
    """Test gpt-5-mini model."""
    config = load_config()
    if not config:
        print("Failed to load config")
        return False
    
    # Get gpt-5-mini configuration
    if "gpt-5-mini" not in config["model_configs"]:
        print("gpt-5-mini not found in model_configs")
        return False
    
    model_config = config["model_configs"]["gpt-5-mini"]
    print(f"Testing gpt-5-mini with config: {model_config}")
    
    # Create Azure client using the model's specific configuration
    try:
        client = AzureOpenAI(
            api_version=model_config["api_version"],
            azure_endpoint=model_config["azure_endpoint"],
            api_key=config["user_azure_api_key"],  # Use the user_azure_api_key
        )
    except Exception as e:
        print(f"Error creating Azure client: {e}")
        return False
    
    # Test basic completion
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you tell me what model you are and say a brief hello?"}
    ]
    
    print("Testing basic chat completion...")
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            messages=test_messages,
            max_completion_tokens=100,  # Use max_completion_tokens for gpt-5-mini
            # temperature=1,  # gpt-5-mini only supports default temperature
            model=model_config["deployment_name"]
        )
        end_time = time.time()
        
        print(f"✅ gpt-5-mini test successful!")
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"📝 Response: {response.choices[0].message.content}")
        print(f"🔢 Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
        return True
        
    except Exception as e:
        print(f"❌ gpt-5-mini test failed: {e}")
        return False

def test_survey_prediction():
    """Test gpt-5-mini with survey prediction task."""
    config = load_config()
    model_config = config["model_configs"]["gpt-5-mini"]
    
    try:
        client = AzureOpenAI(
            api_version=model_config["api_version"],
            azure_endpoint=model_config["azure_endpoint"],
            api_key=config["user_azure_api_key"],
        )
        
        # Test with a demographic prompt similar to your LLM prediction task
        demographic_prompt = """You are a respondent in a survey at the time of May 1st, 2024. You are a 35-year-old Woman who is Not Hispanic, White. You were born in 50 U.S. states or D.C., and are currently Married, and have an education level of Bachelor's degree. Your annual household income is $70,000–79,999. You live in the Middle Atlantic (NJ, NY, PA) region and are located in a Metropolitan area.

For each question, pay attention to the instruction especially the output format. Think step by step. Answer in a concise way."""
        
        question = """Question (ECON1MOD): How would you rate the economic conditions in your community today? Answer on a scale of 1 to 4, where 1 is excellent, 2 is good, 3 is only fair, and 4 is poor.
Please output the number only."""
        
        print("\\nTesting survey prediction task...")
        start_time = time.time()
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": demographic_prompt},
                {"role": "user", "content": question}
            ],
            max_completion_tokens=50,  # Use max_completion_tokens for gpt-5-mini
            # temperature defaults to 1 for gpt-5-mini
            model=model_config["deployment_name"]
        )
        end_time = time.time()
        
        answer = response.choices[0].message.content.strip()
        print(f"✅ Survey prediction test successful!")
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"📝 Answer: {answer}")
        
        # Try to parse the numeric answer
        try:
            numeric_answer = int(answer)
            if 1 <= numeric_answer <= 4:
                print(f"✅ Valid numeric answer: {numeric_answer}")
            else:
                print(f"⚠️  Answer out of range: {numeric_answer}")
        except ValueError:
            print(f"⚠️  Non-numeric answer: {answer}")
        
        return True
        
    except Exception as e:
        print(f"❌ Survey prediction test failed: {e}")
        return False

def compare_with_gpt4o_mini():
    """Compare gpt-5-mini with gpt-4o-mini on the same task."""
    config = load_config()
    
    # Test prompt
    test_prompt = "Explain the concept of machine learning in exactly 50 words."
    
    models_to_test = [
        ("gpt-4o-mini", config["azure_openai_deployment_name"], config["azure_openai_api_key"], config["azure_openai_api_base"], config["azure_openai_api_version"]),
        ("gpt-5-mini", config["model_configs"]["gpt-5-mini"]["deployment_name"], config["user_azure_api_key"], config["model_configs"]["gpt-5-mini"]["azure_endpoint"], config["model_configs"]["gpt-5-mini"]["api_version"])
    ]
    
    print("\\n=== Model Comparison ===")
    
    for model_name, deployment, api_key, endpoint, api_version in models_to_test:
        print(f"\\nTesting {model_name}...")
        try:
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=api_key,
            )
            
            start_time = time.time()
            # Use appropriate parameters based on model
            params = {
                "messages": [{"role": "user", "content": test_prompt}],
                "model": deployment
            }
            
            if "gpt-5" in model_name:
                params["max_completion_tokens"] = 100
                # gpt-5-mini only supports default temperature (1)
            else:
                params["max_tokens"] = 100
                params["temperature"] = 0.7
            
            response = client.chat.completions.create(**params)
            end_time = time.time()
            
            print(f"✅ {model_name} - Time: {end_time - start_time:.2f}s")
            print(f"📝 Response: {response.choices[0].message.content}")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")

def main():
    """Run all tests."""
    print("🧪 Testing gpt-5-mini Model")
    print("=" * 40)
    
    # Basic test
    success = test_gpt5_mini()
    if not success:
        print("Basic test failed, skipping other tests")
        return
    
    # Survey prediction test
    test_survey_prediction()
    
    # Comparison test
    compare_with_gpt4o_mini()
    
    print("\\n🎉 All tests completed!")

if __name__ == "__main__":
    main()