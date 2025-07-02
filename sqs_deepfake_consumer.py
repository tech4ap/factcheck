#!/usr/bin/env python3
"""
SQS Deepfake Consumer Script

This script listens for S3 URLs from AWS SQS and processes them
through the deepfake detection pipeline. It integrates with the
existing modular prediction system.

Expected SQS message format:
{
    "s3_url": "s3://bucket/path/to/file.ext",
    "callback_url": "https://api.example.com/callback",
    "request_id": "unique-request-id",
    "confidence_threshold": 0.5,
    "metadata": {
        "user_id": "12345",
        "upload_timestamp": "2025-01-01T00:00:00Z"
    }
}
"""

import os
import sys
import argparse
import json
import signal
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from src.utils.sqs_consumer import SQSDeepfakeConsumer, create_test_message
from src.inference.common import setup_logging, add_aws_arguments

# Configure logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    logger.info("🛑 Received shutdown signal, stopping consumer...")
    sys.exit(0)

def main():
    """Main function for SQS consumer."""
    parser = argparse.ArgumentParser(
        description='AWS SQS Consumer for Deepfake Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start consumer with environment variables
  export QUEUE_URL="https://sqs.us-east-1.amazonaws.com/123456789012/deepfake-queue"
  export AWS_ACCESS_KEY_ID="your_key"
  export AWS_SECRET_ACCESS_KEY="your_secret"
  python sqs_deepfake_consumer.py
  
  # Start consumer with command line arguments
  python sqs_deepfake_consumer.py \\
    --queue-url "https://sqs.us-east-1.amazonaws.com/123456789012/deepfake-queue" \\
    --aws-access-key-id YOUR_KEY \\
    --aws-secret-access-key YOUR_SECRET
  
  # Test message creation
  python sqs_deepfake_consumer.py --create-test-message \\
    "s3://capstone-development/f84ca5f4-ff77-4f0e-8da1-67a3105a9f52.png"
  
  # Run once for testing
  python sqs_deepfake_consumer.py --run-once
        """)
    
    # SQS Configuration
    parser.add_argument('--queue-url', type=str,
                       help='AWS SQS queue URL (or use QUEUE_URL env var)')
    parser.add_argument('--poll-interval', type=int, default=5,
                       help='Polling interval in seconds (default: 5)')
    parser.add_argument('--max-messages', type=int, default=10,
                       help='Maximum messages to receive per poll (default: 10)')
    parser.add_argument('--wait-time', type=int, default=20,
                       help='Long polling wait time in seconds (default: 20)')
    parser.add_argument('--visibility-timeout', type=int, default=30,
                       help='Message visibility timeout in seconds (default: 30)')
    
    # AWS Configuration
    add_aws_arguments(parser, 'AWS Configuration')
    
    # Model Configuration
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing trained models (default: models)')
    
    # Operational Options
    parser.add_argument('--run-once', action='store_true',
                       help='Poll once and exit (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Parse messages but do not process them')
    parser.add_argument('--stats-interval', type=int, default=10,
                       help='Print stats every N processed messages (default: 10)')
    
    # Testing & Utilities
    parser.add_argument('--create-test-message', type=str, metavar='S3_URL',
                       help='Create a test message for the given S3 URL and exit')
    parser.add_argument('--validate-queue', action='store_true',
                       help='Validate SQS queue connection and exit')
    
    args = parser.parse_args()
    
    # Handle test message creation
    if args.create_test_message:
        test_message = create_test_message(
            args.create_test_message,
            callback_url="https://api.example.com/callback",
            confidence_threshold=0.5,
            metadata={
                "user_id": "test_user",
                "source": "test_script"
            }
        )
        print("📝 Test SQS Message:")
        print(test_message)
        print("\n💡 You can send this message to your SQS queue for testing.")
        return
    
    # Get queue URL
    queue_url = args.queue_url or os.getenv('QUEUE_URL')
    if not queue_url:
        logger.error("❌ SQS queue URL is required!")
        logger.info("💡 Use --queue-url argument or set QUEUE_URL environment variable")
        logger.info("💡 Example: https://sqs.us-east-1.amazonaws.com/123456789012/deepfake-queue")
        return
    
    # Validate AWS credentials
    aws_access_key_id = args.aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = args.aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not aws_access_key_id or not aws_secret_access_key:
        logger.warning("⚠️  AWS credentials not found in arguments or environment!")
        logger.info("💡 Trying to use default AWS credential chain (IAM roles, AWS CLI config, etc.)")
    
    try:
        # Initialize consumer
        consumer = SQSDeepfakeConsumer(
            queue_url=queue_url,
            aws_region=args.aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            models_dir=args.models_dir,
            max_messages=args.max_messages,
            wait_time_seconds=args.wait_time,
            visibility_timeout=args.visibility_timeout,
            poll_interval=args.poll_interval
        )
        
        # Handle queue validation
        if args.validate_queue:
            logger.info("✅ SQS queue connection validated successfully!")
            return
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start consuming messages
        logger.info("🚀 Starting SQS deepfake consumer...")
        logger.info(f"📍 Queue: {queue_url}")
        logger.info(f"🎯 Models directory: {args.models_dir}")
        logger.info(f"⚙️  Configuration: max_messages={args.max_messages}, "
                   f"poll_interval={args.poll_interval}s, wait_time={args.wait_time}s")
        
        if args.dry_run:
            logger.info("🧪 DRY RUN MODE: Messages will be parsed but not processed")
        
        consumer.start_consuming(run_once=args.run_once)
        
    except KeyboardInterrupt:
        logger.info("🛑 Consumer stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start consumer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 