#!/usr/bin/env python3
"""
Test SQS Message Creation and Processing

This script demonstrates how to create test messages for the SQS deepfake consumer
and optionally send them to an actual SQS queue for testing.
"""

import json
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

from src.utils.sqs_consumer import create_test_message

def create_capstone_test_message():
    """Create a test message for the capstone S3 URL."""
    sqs_to_ml_url = "s3://capstone-development/f84ca5f4-ff77-4f0e-8da1-67a3105a9f52.png"
    
    message = create_test_message(
        sqs_to_ml_url=sqs_to_ml_url,
        callback_url="https://api.example.com/deepfake-result",
        confidence_threshold=0.5,
        metadata={
            "user_id": "capstone_user",
            "upload_timestamp": datetime.utcnow().isoformat(),
            "source": "capstone_test",
            "file_type": "image",
            "original_filename": "f84ca5f4-ff77-4f0e-8da1-67a3105a9f52.png"
        }
    )
    
    return message

def send_test_message_to_sqs(queue_url: str, message: str, aws_credentials: dict = None):
    """Send a test message to SQS queue."""
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        # Initialize SQS client
        session_kwargs = {}
        if aws_credentials:
            session_kwargs.update(aws_credentials)
        
        sqs = boto3.client('sqs', **session_kwargs)
        
        # Send message
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=message
        )
        
        print(f"✅ Message sent successfully!")
        print(f"📨 Message ID: {response['MessageId']}")
        print(f"📍 Queue: {queue_url}")
        
        return True
        
    except NoCredentialsError:
        print("❌ AWS credentials not found!")
        print("💡 Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        return False
    except ClientError as e:
        print(f"❌ Failed to send message: {e}")
        return False
    except ImportError:
        print("❌ boto3 library not installed. Install with: pip install boto3")
        return False

def main():
    """Main function."""
    print("🧪 SQS Deepfake Detection Test Message Generator")
    print("=" * 60)
    
    # Create test message
    test_message = create_capstone_test_message()
    
    print("📝 Generated Test Message:")
    print(test_message)
    print("\n" + "=" * 60)
    
    # Ask if user wants to send to actual SQS queue
    send_to_queue = input("\n🤔 Do you want to send this message to an SQS queue? (y/N): ").lower().strip()
    
    if send_to_queue == 'y':
        queue_url = input("📍 Enter SQS Queue URL: ").strip()
        
        if not queue_url:
            print("❌ No queue URL provided, skipping send.")
            return
        
        # Get AWS credentials
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID') or input("🔑 AWS Access Key ID (or press Enter to use default): ").strip()
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY') or input("🔐 AWS Secret Access Key (or press Enter to use default): ").strip()
        aws_region = os.getenv('AWS_REGION') or input("🌍 AWS Region (default: us-east-1): ").strip() or 'us-east-1'
        
        aws_credentials = None
        if aws_access_key_id and aws_secret_access_key:
            aws_credentials = {
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key,
                'region_name': aws_region
            }
        
        # Send message
        success = send_test_message_to_sqs(queue_url, test_message, aws_credentials)
        
        if success:
            print("\n🎉 Test message sent! You can now run the SQS consumer to process it:")
            print(f"   python sqs_deepfake_consumer.py --queue-url '{queue_url}'")
    else:
        print("\n💡 To send this message manually:")
        print("1. Copy the JSON message above")
        print("2. Send it to your SQS queue using AWS CLI or console:")
        print(f"   aws sqs send-message --queue-url YOUR_QUEUE_URL --message-body '{test_message.replace(chr(10), '')}'")
        print("\n🔄 To test the consumer with this message:")
        print("   python sqs_deepfake_consumer.py --queue-url YOUR_QUEUE_URL")

if __name__ == "__main__":
    main() 