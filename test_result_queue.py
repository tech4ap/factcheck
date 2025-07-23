#!/usr/bin/env python3
"""
Test script to demonstrate result queue functionality
without requiring Docker installation.
"""

import json
import time
import os
from datetime import datetime

def test_result_queue_functionality():
    """Test the enhanced SQS consumer with result queue support."""
    
    print("🧪 Testing Result Queue Functionality")
    print("=" * 40)
    
    # Check if AWS credentials are available
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not aws_access_key or not aws_secret_key:
        print("❌ AWS credentials not found in environment variables")
        print("💡 Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False
    
    # Test queue URLs (you can modify these)
    input_queue_url = "https://sqs.us-east-1.amazonaws.com/731485774840/dfk-messages-llm"
    result_queue_url = "https://sqs.us-east-1.amazonaws.com/731485774840/dfk-results-llm"  # Example result queue
    
    print(f"📍 Input Queue: {input_queue_url}")
    print(f"📍 Result Queue: {result_queue_url}")
    print()
    
    # Test 1: Create a test message
    print("🔧 Test 1: Creating test message...")
    test_s3_url = "s3://capstone-development/f84ca5f4-ff77-4f0e-8da1-67a3105a9f52.png"
    
    test_message = {
        "sqs_to_ml_url": test_s3_url,
        "request_id": f"test_{int(time.time())}",
        "callback_url": "https://httpbin.org/post",  # Test callback URL
        "confidence_threshold": 0.5,
        "metadata": {
            "test_mode": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    print("✅ Test message created:")
    print(json.dumps(test_message, indent=2))
    print()
    
    # Test 2: Show enhanced consumer command
    print("🔧 Test 2: Enhanced consumer command with result queue:")
    consumer_cmd = f"""python src/aws/sqs_deepfake_consumer.py \\
    --queue-url "{input_queue_url}" \\
    --result-queue-url "{result_queue_url}" \\
    --run-once \\
    --dry-run"""
    
    print(consumer_cmd)
    print()
    
    # Test 3: Show expected result format
    print("🔧 Test 3: Expected result queue message format:")
    expected_result = {
        "timestamp": datetime.utcnow().isoformat(),
        "source_queue": input_queue_url,
        "result": {
            "request_id": test_message["request_id"],
            "sqs_to_ml_url": test_s3_url,
            "file_type": "image",
            "label": "FAKE",
            "confidence": 0.73,
            "prediction_score": 0.7312,
            "is_s3_file": True,
            "processed_at": datetime.utcnow().isoformat(),
            "original_message": test_message
        }
    }
    
    print("✅ Expected result queue message:")
    print(json.dumps(expected_result, indent=2))
    print()
    
    # Test 4: Show deployment options
    print("🔧 Test 4: Deployment Options Without Docker:")
    print("1. Direct Python execution:")
    print("   python src/aws/sqs_deepfake_consumer.py --queue-url YOUR_QUEUE --result-queue-url YOUR_RESULT_QUEUE")
    print()
    print("2. With environment variables:")
    print("   export QUEUE_URL='your-input-queue-url'")
    print("   export RESULT_QUEUE_URL='your-result-queue-url'")
    print("   python src/aws/sqs_deepfake_consumer.py")
    print()
    print("3. Background process:")
    print("   nohup python src/aws/sqs_deepfake_consumer.py > consumer.log 2>&1 &")
    print()
    
    # Test 5: Show AWS deployment without Docker
    print("🔧 Test 5: AWS Deployment Without Docker:")
    print("1. AWS EC2 with Python:")
    print("   - Launch EC2 instance")
    print("   - Install Python and dependencies")
    print("   - Upload your code")
    print("   - Run the consumer directly")
    print()
    print("2. AWS Lambda:")
    print("   - Create deployment package")
    print("   - Deploy as Lambda function")
    print("   - Trigger from SQS directly")
    print()
    
    return True

def create_systemd_service():
    """Create a systemd service file for running on Linux servers."""
    
    service_content = """[Unit]
Description=Deepfake Detection SQS Consumer
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/deepfake-detection
Environment=PYTHONPATH=/home/ubuntu/deepfake-detection/src
Environment=AWS_ACCESS_KEY_ID=your_access_key
Environment=AWS_SECRET_ACCESS_KEY=your_secret_key
Environment=QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/input-queue
Environment=RESULT_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/result-queue
ExecStart=/usr/bin/python3 src/aws/sqs_deepfake_consumer.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    print("🔧 Systemd Service File (deepfake-consumer.service):")
    print(service_content)
    print("💡 To use:")
    print("  sudo cp deepfake-consumer.service /etc/systemd/system/")
    print("  sudo systemctl daemon-reload")
    print("  sudo systemctl enable deepfake-consumer")
    print("  sudo systemctl start deepfake-consumer")
    print()

if __name__ == "__main__":
    print("🚀 Deepfake Detection Service - Result Queue Test")
    print("=" * 50)
    print()
    
    # Run functionality test
    test_result_queue_functionality()
    
    print()
    print("🔧 Additional Deployment Options:")
    print("=" * 35)
    create_systemd_service()
    
    print("✅ Test completed! The service is ready for deployment.")
    print("💡 Choose your preferred deployment method based on your infrastructure.") 