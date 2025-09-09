#!/usr/bin/env python3
"""
Configuration manager for Prediction-Guided Active Experiments

This module handles all API configuration and client creation.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

class ConfigManager:
    """Manages API configurations and client creation."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "config.json"
    
    def create_config_template(self) -> bool:
        """Create a template config file if it doesn't exist."""
        self.config_dir.mkdir(exist_ok=True)
        
        if not self.config_file.exists():
            template_config = {
                "use_azure_openai": True,
                "azure_openai_api_key": "YOUR_API_KEY_HERE",
                "azure_openai_api_base": "https://your-endpoint.openai.azure.com/",
                "azure_openai_api_version": "2024-12-01-preview",
                "azure_openai_deployment_name": "gpt-4o-mini",
                "azure_openai_model": "gpt-4o-mini",
                "openai_api_key": "YOUR_OPENAI_API_KEY_HERE",
                "model": "gpt-4o-mini"
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(template_config, f, indent=2)
            
            print(f"Created template config at {self.config_file}")
            print("Please edit this file with your actual API credentials.")
            return False
        
        return True
    
    def load_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from config.json."""
        if not self.config_file.exists():
            print("Config file not found. Creating template...")
            self.create_config_template()
            return None
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return None
    
    def get_azure_client(self):
        """Get configured Azure OpenAI client."""
        config = self.load_config()
        if not config:
            return None, None
        
        try:
            from openai import AzureOpenAI
            
            client = AzureOpenAI(
                api_version=config["azure_openai_api_version"],
                azure_endpoint=config["azure_openai_api_base"],
                api_key=config["azure_openai_api_key"],
            )
            return client, config
        except Exception as e:
            print(f"Error creating Azure client: {e}")
            return None, None
    
    def test_azure_connection(self) -> bool:
        """Test the Azure API connection."""
        client, config = self.get_azure_client()
        if not client:
            print("Failed to create Azure client")
            return False
        
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello!"}
                ],
                max_tokens=10,
                model=config["azure_openai_deployment_name"]
            )
            print("Azure connection successful!")
            print(f"Response: {response.choices[0].message.content}")
            return True
        except Exception as e:
            print(f"Azure connection failed: {e}")
            return False

def main():
    """CLI for config management."""
    import sys
    
    config_manager = ConfigManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "test-azure":
            config_manager.test_azure_connection()
        elif command == "init":
            config_manager.create_config_template()
        else:
            print("Usage: python config_manager.py [init|test-azure]")
    else:
        config_manager.create_config_template()
        print("Use 'python config_manager.py test-azure' to test your API connection.")

if __name__ == "__main__":
    main()