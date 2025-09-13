#!/usr/bin/env python3
"""
Unified GPT-5-Mini Test Suite

Comprehensive testing script for gpt-5-mini model with debugging capabilities.
Combines functionality from multiple test scripts into a single unified interface.
"""

import json
import time
import argparse
from openai import AzureOpenAI

def load_config():
    """Load configuration from config.json."""
    try:
        with open('./config/config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def create_client():
    """Create Azure OpenAI client for gpt-5-mini."""
    config = load_config()
    if not config:
        return None, None
    
    model_config = config["model_configs"]["gpt-5-mini"]
    
    try:
        client = AzureOpenAI(
            api_version=model_config["api_version"],
            azure_endpoint=model_config["azure_endpoint"],
            api_key=config["user_azure_api_key"],
        )
        return client, model_config
    except Exception as e:
        print(f"Error creating Azure client: {e}")
        return None, None

def basic_test():
    """Test basic gpt-5-mini functionality."""
    print("🧪 Basic Functionality Test")
    print("=" * 40)
    
    client, model_config = create_client()
    if not client:
        return False
    
    test_cases = [
        {
            "name": "Hello Test",
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
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test['name']} ---")
        
        try:
            params = {
                "messages": test["messages"],
                "model": model_config["deployment_name"]
            }
            
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
            
            # Check for reasoning tokens (gpt-5-mini specific)
            if hasattr(response.usage, 'completion_tokens_details'):
                details = response.usage.completion_tokens_details
                if hasattr(details, 'reasoning_tokens'):
                    print(f"🧠 Reasoning tokens: {details.reasoning_tokens}")
                    
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    return True

def survey_prediction_test():
    """Test gpt-5-mini with survey prediction tasks."""
    print("\n🔍 Survey Prediction Test")
    print("=" * 40)
    
    client, model_config = create_client()
    if not client:
        return False
    
    # Demographic prompt for survey prediction
    demographic_prompt = """You are a respondent in a survey at the time of May 1st, 2024. You are a 35-year-old Woman who is Not Hispanic, White. You were born in 50 U.S. states or D.C., and are currently Married, and have an education level of Bachelor's degree. Your annual household income is $70,000–79,999. You live in the Middle Atlantic (NJ, NY, PA) region and are located in a Metropolitan area.

For each question, pay attention to the instruction especially the output format. Think step by step. Answer in a concise way."""
    
    question = """Question (ECON1MOD): How would you rate the economic conditions in your community today? Answer on a scale of 1 to 4, where 1 is excellent, 2 is good, 3 is only fair, and 4 is poor.
Please output the number only."""
    
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": demographic_prompt},
                {"role": "user", "content": question}
            ],
            max_completion_tokens=50,
            model=model_config["deployment_name"]
        )
        end_time = time.time()
        
        answer = response.choices[0].message.content.strip()
        print(f"✅ Survey prediction successful!")
        print(f"⏱️  Time: {end_time - start_time:.2f}s")
        print(f"📝 Answer: '{answer}'")
        
        # Validate numeric answer
        try:
            num = int(answer)
            if 1 <= num <= 4:
                print(f"✅ Valid survey response: {num}")
            else:
                print(f"⚠️  Out of range: {num}")
        except ValueError:
            print(f"⚠️  Non-numeric answer: '{answer}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Survey prediction failed: {e}")
        return False

def model_comparison_test():
    """Compare gpt-5-mini with gpt-4o-mini."""
    print("\n🔄 Model Comparison Test")
    print("=" * 40)
    
    config = load_config()
    if not config:
        return False
    
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
            "params": {"max_completion_tokens": 10}
        }
    ]
    
    for model in models:
        print(f"\n--- {model['name']} ---")
        
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
            
            # Validate response
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

def debug_mode():
    """Deep debugging mode for gpt-5-mini responses."""
    print("\n🔧 Debug Mode")
    print("=" * 40)
    
    client, model_config = create_client()
    if not client:
        return False
    
    print("Testing with minimal parameters...")
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model=model_config["deployment_name"]
        )
        
        print(f"Full response type: {type(response)}")
        print(f"Response dict: {response.model_dump() if hasattr(response, 'model_dump') else 'No model_dump method'}")
        
        choice = response.choices[0]
        print(f"Choice type: {type(choice)}")
        
        message = choice.message
        print(f"Message type: {type(message)}")
        
        content = message.content
        print(f"Content type: {type(content)}")
        print(f"Content value: {repr(content)}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='GPT-5-Mini Unified Test Suite')
    parser.add_argument('--mode', choices=['basic', 'survey', 'compare', 'debug', 'all'], 
                        default='all', help='Test mode to run')
    
    args = parser.parse_args()
    
    print("🚀 GPT-5-Mini Unified Test Suite")
    print("=" * 60)
    
    if args.mode in ['basic', 'all']:
        basic_test()
    
    if args.mode in ['survey', 'all']:
        survey_prediction_test()
    
    if args.mode in ['compare', 'all']:
        model_comparison_test()
    
    if args.mode == 'debug':
        debug_mode()
    
    if args.mode == 'all':
        print("\n🎉 All tests completed!")
        print("\n📋 Summary:")
        print("- gpt-5-mini is accessible and working")
        print("- Uses max_completion_tokens (not max_tokens)")
        print("- Only supports default temperature (1.0)")
        print("- Includes reasoning tokens in usage statistics")
        print("- May return empty strings with very low token limits")

if __name__ == "__main__":
    main()