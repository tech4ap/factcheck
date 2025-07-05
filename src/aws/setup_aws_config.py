#!/usr/bin/env python3
"""
AWS Configuration Setup Script

This script helps users set up their AWS credentials securely for the deepfake detection project.
"""

import json
import os
from pathlib import Path

def setup_aws_config():
    """Interactive setup for AWS configuration."""
    print("🔧 AWS Configuration Setup for Deepfake Detection")
    print("=" * 50)
    
    # Check if aws_config.json already exists
    config_file = Path("aws_config.json")
    if config_file.exists():
        overwrite = input("⚠️  aws_config.json already exists. Overwrite? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("❌ Setup cancelled.")
            return
    
    # Get AWS credentials from user
    print("\n📋 Please enter your AWS credentials:")
    print("💡 Tip: You can get these from AWS Console > IAM > Users > Security Credentials")
    
    access_key_id = input("🔑 AWS Access Key ID: ").strip()
    if not access_key_id:
        print("❌ Access Key ID is required!")
        return
    
    secret_access_key = input("🔐 AWS Secret Access Key: ").strip()
    if not secret_access_key:
        print("❌ Secret Access Key is required!")
        return
    
    region = input("📍 AWS Region (default: us-east-1): ").strip() or "us-east-1"
    
    # Create configuration
    config = {
        "aws": {
            "access_key_id": access_key_id,
            "secret_access_key": secret_access_key,
            "region": region,
            "session_token": None
        },
        "data": {
            "temp_dir": "/tmp/deepfake_cache",
            "s3_cache_dir": "/tmp/deepfake_s3_cache",
            "cleanup_temp_files": True
        },
        "logging": {
            "level": "INFO",
            "log_to_console": True
        }
    }
    
    # Save configuration
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Configuration saved to {config_file}")
        print("🔒 This file is automatically excluded from git commits")
        print("\n🚀 You can now use the deepfake detection script:")
        print("   python src/aws/predict_s3_deepfake.py s3://bucket/your-file.jpg")
        
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")

def check_current_config():
    """Check and display current configuration status."""
    print("📊 Current Configuration Status")
    print("=" * 35)
    
    # Check aws_config.json
    config_file = Path("aws_config.json")
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            aws_config = config.get('aws', {})
            has_access_key = bool(aws_config.get('access_key_id'))
            has_secret_key = bool(aws_config.get('secret_access_key'))
            region = aws_config.get('region', 'Not set')
            
            print(f"📁 Config file: ✅ Found")
            print(f"🔑 Access Key: {'✅ Set' if has_access_key else '❌ Missing'}")
            print(f"🔐 Secret Key: {'✅ Set' if has_secret_key else '❌ Missing'}")
            print(f"📍 Region: {region}")
            
            if has_access_key and has_secret_key:
                print("\n🎉 Configuration is complete!")
            else:
                print("\n⚠️  Configuration is incomplete!")
                
        except Exception as e:
            print(f"📁 Config file: ❌ Invalid ({e})")
    else:
        print("📁 Config file: ❌ Not found")
    
    # Check environment variables
    print("\n🌍 Environment Variables:")
    env_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    env_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    env_region = os.getenv('AWS_REGION')
    
    print(f"🔑 AWS_ACCESS_KEY_ID: {'✅ Set' if env_access_key else '❌ Not set'}")
    print(f"🔐 AWS_SECRET_ACCESS_KEY: {'✅ Set' if env_secret_key else '❌ Not set'}")
    print(f"📍 AWS_REGION: {env_region if env_region else '❌ Not set'}")

def main():
    """Main function with menu."""
    while True:
        print("\n🔧 AWS Configuration Manager")
        print("=" * 30)
        print("1. 📋 Setup new configuration")
        print("2. 📊 Check current configuration")
        print("3. 📖 Show usage examples")
        print("4. 🚪 Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            setup_aws_config()
        elif choice == '2':
            check_current_config()
        elif choice == '3':
            show_usage_examples()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")

def show_usage_examples():
    """Show usage examples."""
    print("\n📖 Usage Examples")
    print("=" * 20)
    print("\n🔧 Setup (first time):")
    print("   python src/aws/setup_aws_config.py")
    print("\n🚀 Basic usage:")
    print("   python src/aws/predict_s3_deepfake.py s3://bucket/image.jpg")
    print("\n⚙️  Advanced usage:")
    print("   python src/aws/predict_s3_deepfake.py s3://bucket/video.mp4 --threshold 0.8")
    print("   python src/aws/predict_s3_deepfake.py s3://bucket/audio.wav --output results.json")
    print("   python src/aws/predict_s3_deepfake.py s3://bucket/file.jpg --config custom.json")
    print("\n📚 See AWS_INTEGRATION_GUIDE.md for detailed documentation")

if __name__ == "__main__":
    main() 