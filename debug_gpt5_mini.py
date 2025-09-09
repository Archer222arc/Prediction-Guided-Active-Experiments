#!/usr/bin/env python3
"""
Debug script for gpt-5-mini to understand its response behavior
"""

import json
from openai import AzureOpenAI

def load_config():
    with open('./config/config.json', 'r') as f:
        return json.load(f)

def debug_gpt5_mini():
    config = load_config()
    model_config = config["model_configs"]["gpt-5-mini"]
    
    client = AzureOpenAI(
        api_version=model_config["api_version"],
        azure_endpoint=model_config["azure_endpoint"],
        api_key=config["user_azure_api_key"],
    )
    
    test_cases = [
        {"name": "Simple Math", "prompt": "What is 2 + 2?"},
        {"name": "Simple Question", "prompt": "What color is the sky?"},
        {"name": "Number Only", "prompt": "Pick a number between 1 and 4. Output only the number."},
        {"name": "Survey Question", "prompt": "Rate your satisfaction from 1-4, where 1=poor, 4=excellent. Just give the number."}
    ]
    
    for test in test_cases:
        print(f"\\n--- {test['name']} ---")
        print(f"Prompt: {test['prompt']}")
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": test['prompt']}],
                max_completion_tokens=20,
                model=model_config["deployment_name"]
            )
            
            content = response.choices[0].message.content
            print(f"Raw response: '{content}'")
            print(f"Response length: {len(content) if content else 'None'}")
            print(f"Response type: {type(content)}")
            
            if response.usage:
                print(f"Tokens: {response.usage.total_tokens}")
                print(f"Completion tokens: {response.usage.completion_tokens}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    debug_gpt5_mini()