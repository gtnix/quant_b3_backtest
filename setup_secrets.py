#!/usr/bin/env python3
"""
Setup script for configuring API secrets safely.

This script helps you create your secrets.yaml file without exposing
API keys in your code or accidentally committing them to Git.
"""

import os
import yaml
from pathlib import Path
import getpass

def setup_secrets():
    """Interactive setup for API secrets."""
    
    secrets_path = Path("config/secrets.yaml")
    example_path = Path("config/secrets.yaml.example")
    
    print("🔐 Brazilian Stock Market Backtesting Engine - Secrets Setup")
    print("=" * 60)
    
    # Check if secrets file already exists
    if secrets_path.exists():
        print(f"⚠️  Warning: {secrets_path} already exists!")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    # Ensure config directory exists
    secrets_path.parent.mkdir(exist_ok=True)
    
    # Load example template
    if example_path.exists():
        with open(example_path, 'r') as f:
            template = yaml.safe_load(f)
    else:
        template = {
            'alpha_vantage': {
                'api_key': 'YOUR_ALPHA_VANTAGE_API_KEY_HERE',
                'base_url': 'https://www.alphavantage.co/query'
            }
        }
    
    print("\n📋 Alpha Vantage API Configuration")
    print("-" * 40)
    
    # Get API key securely
    print("Enter your Alpha Vantage API key:")
    print("(Get one for free at: https://www.alphavantage.co/support/#api-key)")
    
    api_key = getpass.getpass("API Key: ").strip()
    
    if not api_key:
        print("❌ API key is required!")
        return
    
    if api_key == "YOUR_ALPHA_VANTAGE_API_KEY_HERE":
        print("❌ Please enter your actual API key, not the placeholder!")
        return
    
    # Update template with real values
    template['alpha_vantage']['api_key'] = api_key
    
    # Save secrets file
    try:
        with open(secrets_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
        
        print(f"\n✅ Successfully created {secrets_path}")
        print("🔒 This file is protected by .gitignore and will NOT be uploaded to GitHub")
        
    except Exception as e:
        print(f"❌ Error creating secrets file: {e}")
        return
    
    # Test the configuration
    print("\n🧪 Testing configuration...")
    try:
        from scripts.download_data import B3DataDownloader
        downloader = B3DataDownloader()
        print("✅ Configuration test successful!")
    except Exception as e:
        print(f"⚠️  Configuration test failed: {e}")
        print("You may need to check your API key or internet connection.")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run: python scripts/download_data.py")
    print("2. Check the data/raw/ directory for downloaded files")
    print("3. Start building your trading strategies!")

if __name__ == "__main__":
    setup_secrets() 