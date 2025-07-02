#!/usr/bin/env python3
"""
Example S3 Deepfake Detection Usage

This script demonstrates how to use the deepfake detection system
with files stored on Amazon S3.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

def demonstrate_s3_utilities():
    """Demonstrate S3 utility functions."""
    print("🔧 S3 UTILITIES DEMONSTRATION")
    print("="*50)
    
    from utils.s3_utils import is_s3_url, get_file_extension_from_s3_url, setup_aws_credentials_from_env
    
    # Test URLs
    test_urls = [
        "s3://capstone-development/02f9d020-7883-4b62-99a5-935b44702e62.jpg",
        "https://capstone-development.s3.us-east-1.amazonaws.com/video.mp4",
        "local_file.jpg",
        "https://example.com/file.jpg"
    ]
    
    print("\n🔍 URL Type Detection:")
    for url in test_urls:
        is_s3 = is_s3_url(url)
        print(f"  {url}")
        print(f"    Is S3 URL: {'✅ Yes' if is_s3 else '❌ No'}")
        if is_s3:
            ext = get_file_extension_from_s3_url(url)
            print(f"    File extension: {ext}")
        print()
    
    print("\n🔐 AWS Credentials Check:")
    setup_aws_credentials_from_env()

def demonstrate_s3_file_handler():
    """Demonstrate S3FileHandler functionality."""
    print("\n🗂️  S3 FILE HANDLER DEMONSTRATION")
    print("="*50)
    
    from utils.s3_utils import S3FileHandler
    
    # Initialize handler
    print("Initializing S3 handler...")
    handler = S3FileHandler()
    
    if handler.s3_client is None:
        print("❌ S3 client not initialized (no AWS credentials)")
        print("💡 This is expected if you haven't configured AWS credentials")
        return
    
    # Example S3 URL
    s3_url = "s3://capstone-development/02f9d020-7883-4b62-99a5-935b44702e62.jpg"
    
    try:
        # Get file info
        print(f"\n📊 Getting file info for: {s3_url}")
        file_info = handler.get_file_info(s3_url)
        print(f"  Bucket: {file_info['bucket']}")
        print(f"  Key: {file_info['key']}")
        print(f"  Size: {file_info['size']:,} bytes")
        print(f"  Content Type: {file_info['content_type']}")
        
        # Download file
        print(f"\n📥 Downloading file...")
        local_path = handler.download_file(s3_url)
        print(f"  Downloaded to: {local_path}")
        
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        handler.cleanup_temp_files(max_age_hours=0)  # Clean all files
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demonstrate_deepfake_detection():
    """Demonstrate deepfake detection with S3 files."""
    print("\n🔍 DEEPFAKE DETECTION DEMONSTRATION")
    print("="*50)
    
    # Check if models exist
    models_dir = Path("models")
    if not models_dir.exists() or not any(models_dir.glob("*.h5")):
        print("❌ No trained models found!")
        print("💡 Run training first: python train_and_save_models.py --demo")
        return
    
    from inference.predict_deepfake import EnhancedDeepfakePredictor
    
    print("Initializing deepfake predictor...")
    predictor = EnhancedDeepfakePredictor()
    
    # Load models
    loaded = predictor.load_models()
    if not any(loaded.values()):
        print("❌ No models loaded successfully!")
        return
    
    print(f"✅ Loaded models: {[k for k, v in loaded.items() if v]}")
    
    # Example S3 URLs for different media types
    example_urls = {
        "image": "s3://capstone-development/02f9d020-7883-4b62-99a5-935b44702e62.jpg",
        "video": "s3://your-bucket/sample-video.mp4",
        "audio": "s3://your-bucket/sample-audio.wav"
    }
    
    print("\n📝 Example S3 URLs you can test:")
    for media_type, url in example_urls.items():
        print(f"  {media_type.capitalize()}: {url}")
    
    print("\n💡 To test with real S3 files, use:")
    print("  python predict_s3_deepfake.py 's3://your-bucket/your-file.jpg'")
    print("  python src/inference/predict_deepfake.py --input 's3://your-bucket/your-file.mp4'")

def show_command_examples():
    """Show command-line usage examples."""
    print("\n📚 COMMAND-LINE EXAMPLES")
    print("="*50)
    
    examples = [
        {
            "title": "Check AWS Credentials Setup",
            "command": "python predict_s3_deepfake.py --setup-aws"
        },
        {
            "title": "Detect Deepfake in S3 Image",
            "command": "python predict_s3_deepfake.py 's3://capstone-development/02f9d020-7883-4b62-99a5-935b44702e62.jpg'"
        },
        {
            "title": "Use Custom AWS Credentials",
            "command": "python predict_s3_deepfake.py 's3://bucket/file.jpg' --aws-access-key-id YOUR_KEY --aws-secret-access-key YOUR_SECRET"
        },
        {
            "title": "Use General Inference Script with S3",
            "command": "python src/inference/predict_deepfake.py --input 's3://bucket/file.mp4'"
        },
        {
            "title": "Save Results to File",
            "command": "python predict_s3_deepfake.py 's3://bucket/file.wav' --output results.json"
        },
        {
            "title": "Use Custom Confidence Threshold",
            "command": "python predict_s3_deepfake.py 's3://bucket/file.jpg' --threshold 0.8"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}️⃣  {example['title']}:")
        print(f"   {example['command']}")

def show_programmatic_examples():
    """Show programmatic usage examples."""
    print("\n💻 PROGRAMMATIC USAGE EXAMPLES")
    print("="*50)
    
    print("""
# Example 1: Basic S3 file detection
from src.inference.predict_deepfake import EnhancedDeepfakePredictor

predictor = EnhancedDeepfakePredictor(
    aws_access_key_id='your_access_key',
    aws_secret_access_key='your_secret_key'
)
predictor.load_models()

result = predictor.predict_auto('s3://bucket/file.jpg')
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.1%}")

# Example 2: Using S3 utilities directly
from src.utils.s3_utils import S3FileHandler

handler = S3FileHandler()
local_path = handler.download_file('s3://bucket/file.mp4')
print(f"Downloaded to: {local_path}")

# Example 3: Check if URL is S3
from src.utils.s3_utils import is_s3_url

if is_s3_url('s3://bucket/file.wav'):
    print("This is an S3 URL!")

# Example 4: Environment variables setup
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'your_access_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret_key'

predictor = EnhancedDeepfakePredictor()  # Will use env vars
result = predictor.predict_auto('s3://bucket/file.jpg')
""")

def main():
    """Main demonstration function."""
    print("🌟 DEEPFAKE DETECTION S3 INTEGRATION DEMO")
    print("="*60)
    print("This demo shows how to use the deepfake detection system")
    print("with files stored on Amazon S3.")
    print()
    
    # Check if we have the required modules
    try:
        import boto3
        import botocore
        print("✅ AWS SDK (boto3) is installed")
    except ImportError:
        print("❌ AWS SDK not installed. Run: pip install boto3 botocore")
        return
    
    try:
        # Demonstrate utilities
        demonstrate_s3_utilities()
        
        # Demonstrate S3 file handler
        demonstrate_s3_file_handler()
        
        # Demonstrate deepfake detection
        demonstrate_deepfake_detection()
        
        # Show command examples
        show_command_examples()
        
        # Show programmatic examples
        show_programmatic_examples()
        
        print("\n🎉 DEMO COMPLETE!")
        print("="*60)
        print("For more information, see S3_INTEGRATION_GUIDE.md")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 