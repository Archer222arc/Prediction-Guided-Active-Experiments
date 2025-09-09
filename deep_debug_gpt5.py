#!/usr/bin/env python3
"""
Deep debug for gpt-5-mini
"""

import json
from openai import AzureOpenAI

def deep_debug():
    with open('./config/config.json', 'r') as f:
        config = json.load(f)
    
    model_config = config["model_configs"]["gpt-5-mini"]
    
    client = AzureOpenAI(
        api_version=model_config["api_version"],
        azure_endpoint=model_config["azure_endpoint"],
        api_key=config["user_azure_api_key"],
    )
    
    print("Testing with minimal parameters...")
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model=model_config["deployment_name"]
        )
        
        print(f"Full response object type: {type(response)}")
        print(f"Response dict: {response.model_dump() if hasattr(response, 'model_dump') else 'No model_dump method'}")
        
        choice = response.choices[0]
        print(f"Choice type: {type(choice)}")
        print(f"Choice dict: {choice.model_dump() if hasattr(choice, 'model_dump') else 'No model_dump method'}")
        
        message = choice.message
        print(f"Message type: {type(message)}")
        print(f"Message dict: {message.model_dump() if hasattr(message, 'model_dump') else 'No model_dump method'}")
        
        content = message.content
        print(f"Content type: {type(content)}")
        print(f"Content value: {repr(content)}")
        print(f"Content bytes: {content.encode('utf-8') if content else None}")
        
        # Try different ways to access content
        print(f"Dir of message: {[attr for attr in dir(message) if not attr.startswith('_')]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    deep_debug()