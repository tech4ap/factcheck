"""
AWS SQS Consumer for Deepfake Detection

This module listens for S3 URLs from AWS SQS and processes them
through the deepfake detection pipeline.
"""

import json
import logging
import time
import os
import sys
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.inference.common import (
    setup_logging, validate_s3_url_and_credentials, 
    handle_prediction_error, PredictionContext
)

# Configure logging
logger = logging.getLogger(__name__)

class SQSDeepfakeConsumer:
    """
    AWS SQS Consumer for processing deepfake detection requests.
    
    Expected message format:
    {
        "sqs_to_ml_url": "s3://bucket/path/to/file.ext",
        "callback_url": "https://api.example.com/callback",  # optional
        "request_id": "unique-request-id",  # optional
        "metadata": {  # optional
            "user_id": "12345",
            "upload_timestamp": "2025-01-01T00:00:00Z"
        }
    }
    """
    
    def __init__(self, 
                 queue_url: str,
                 aws_region: str = 'us-east-1',
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 models_dir: str = 'models',
                 max_messages: int = 10,
                 wait_time_seconds: int = 20,
                 visibility_timeout: int = 30,
                 poll_interval: int = 5,
                 result_queue_url: Optional[str] = None):
        """
        Initialize SQS consumer.
        
        Args:
            queue_url: AWS SQS queue URL for receiving messages
            aws_region: AWS region
            aws_access_key_id: AWS access key ID (optional, uses env vars if not provided)
            aws_secret_access_key: AWS secret access key (optional)
            models_dir: Directory containing trained models
            max_messages: Maximum messages to receive per poll
            wait_time_seconds: Long polling wait time
            visibility_timeout: Message visibility timeout
            poll_interval: Polling interval in seconds
            result_queue_url: Optional SQS queue URL for sending results
        """
        self.queue_url = queue_url
        self.result_queue_url = result_queue_url
        self.aws_region = aws_region
        self.models_dir = models_dir
        self.max_messages = max_messages
        self.wait_time_seconds = wait_time_seconds
        self.visibility_timeout = visibility_timeout
        self.poll_interval = poll_interval
        
        # Initialize AWS credentials
        self.aws_access_key_id = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        
        # Initialize SQS client
        self.sqs_client = self._initialize_sqs_client()
        
        # Initialize predictor (lazy loading)
        self._predictor = None
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'messages_failed': 0,
            'results_sent': 0,
            'callbacks_sent': 0,
            'start_time': None
        }
    
    def _initialize_sqs_client(self):
        """Initialize AWS SQS client with proper error handling."""
        try:
            session_kwargs = {'region_name': self.aws_region}
            
            if self.aws_access_key_id and self.aws_secret_access_key:
                session_kwargs.update({
                    'aws_access_key_id': self.aws_access_key_id,
                    'aws_secret_access_key': self.aws_secret_access_key
                })
            
            sqs = boto3.client('sqs', **session_kwargs)
            
            # Test connection
            sqs.get_queue_attributes(QueueUrl=self.queue_url, AttributeNames=['QueueArn'])
            logger.info(f"✅ Successfully connected to SQS queue: {self.queue_url}")
            
            return sqs
            
        except NoCredentialsError:
            logger.error("❌ AWS credentials not found!")
            logger.info("💡 Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
            raise
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AWS.SimpleQueueService.NonExistentQueue':
                logger.error(f"❌ SQS queue does not exist: {self.queue_url}")
            elif error_code == 'AccessDenied':
                logger.error(f"❌ Access denied to SQS queue: {self.queue_url}")
            else:
                logger.error(f"❌ Failed to connect to SQS: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error initializing SQS client: {e}")
            raise
    
    def _get_predictor(self):
        """Lazy load the deepfake predictor."""
        if self._predictor is None:
            try:
                from src.inference.predict_deepfake import EnhancedDeepfakePredictor
                from src.inference.common import validate_and_load_models
                
                self._predictor = EnhancedDeepfakePredictor(
                    models_dir=self.models_dir,
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    aws_region=self.aws_region
                )
                
                # Load models
                validate_and_load_models(self._predictor)
                logger.info("🤖 Deepfake predictor initialized and models loaded")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize predictor: {e}")
                raise
        
        return self._predictor
    
    def parse_message(self, message_body: str) -> Dict[str, Any]:
        """
        Parse SQS message body.
        
        Args:
            message_body: Raw message body string
            
        Returns:
            Parsed message dictionary
            
        Raises:
            ValueError: If message format is invalid
        """
        try:
            # Try to parse as JSON
            message_data = json.loads(message_body)
            
            # Validate required fields
            if 'sqs_to_ml_url' not in message_data:
                raise ValueError("Message missing required field 'sqs_to_ml_url'")
            
            # Validate S3 URL format
            sqs_to_ml_url = message_data['sqs_to_ml_url']
            validate_s3_url_and_credentials(sqs_to_ml_url, self.aws_access_key_id)
            
            return message_data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Message validation failed: {e}")
    
    def process_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single deepfake detection message.
        
        Args:
            message_data: Parsed message data
            
        Returns:
            Processing result dictionary
        """
        sqs_to_ml_url = message_data['sqs_to_ml_url']
        request_id = message_data.get('request_id', f"req_{int(time.time())}")
        
        logger.info(f"🔍 Processing deepfake detection for: {sqs_to_ml_url} (ID: {request_id})")
        
        try:
            predictor = self._get_predictor()
            
            # Run prediction with context management for cleanup
            with PredictionContext(predictor, cleanup_files=True) as ctx:
                result = ctx.predict(sqs_to_ml_url, confidence_threshold=0.5)
                
                # Enhance result with message metadata
                result.update({
                    'request_id': request_id,
                    'processed_at': datetime.utcnow().isoformat(),
                    'queue_url': self.queue_url,
                    'original_message': message_data
                })
                
                logger.info(f"✅ Detection complete - {result['label']} "
                           f"(confidence: {result['confidence']:.1%}) for {request_id}")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Failed to process {request_id}: {e}")
            
            # Return error result
            return {
                'request_id': request_id,
                'sqs_to_ml_url': sqs_to_ml_url,
                'error': str(e),
                'processed_at': datetime.utcnow().isoformat(),
                'status': 'failed'
            }
    
    def send_result_to_queue(self, result: Dict[str, Any]) -> bool:
        """
        Send result to result queue.
        
        Args:
            result: Processing result
            
        Returns:
            True if successful, False otherwise
        """
        if not self.result_queue_url:
            logger.debug("🔄 No result queue configured, skipping result queue send")
            return False
            
        try:
            # Prepare result message
            result_message = {
                'timestamp': datetime.utcnow().isoformat(),
                'source_queue': self.queue_url,
                'result': result
            }
            
            # Send to result queue
            response = self.sqs_client.send_message(
                QueueUrl=self.result_queue_url,
                MessageBody=json.dumps(result_message)
            )
            
            logger.info(f"✅ Result sent to queue: {self.result_queue_url} (MessageId: {response['MessageId']})")
            self.stats['results_sent'] += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send result to queue {self.result_queue_url}: {e}")
            return False
    
    def send_callback(self, result: Dict[str, Any], callback_url: str) -> bool:
        """
        Send result to callback URL.
        
        Args:
            result: Processing result
            callback_url: URL to send result to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import requests
            
            response = requests.post(
                callback_url,
                json=result,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Callback sent successfully to {callback_url}")
                self.stats['callbacks_sent'] += 1
                return True
            else:
                logger.warning(f"⚠️ Callback failed with status {response.status_code}: {callback_url}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to send callback to {callback_url}: {e}")
            return False
    
    def poll_messages(self) -> None:
        """Poll SQS for messages and process them."""
        try:
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=self.max_messages,
                WaitTimeSeconds=self.wait_time_seconds,
                VisibilityTimeout=self.visibility_timeout
            )
            
            messages = response.get('Messages', [])
            
            if not messages:
                logger.debug("📭 No messages in queue")
                return
            
            logger.info(f"📨 Received {len(messages)} message(s)")
            
            for message in messages:
                self._process_single_message(message)
                
        except Exception as e:
            logger.error(f"❌ Error polling messages: {e}")
    
    def _process_single_message(self, message: Dict[str, Any]) -> None:
        """Process a single SQS message."""
        message_id = message['MessageId']
        receipt_handle = message['ReceiptHandle']
        body = message['Body']
        
        try:
            # Parse message
            message_data = self.parse_message(body)
            
            # Process deepfake detection
            result = self.process_message(message_data)
            
            # Send result to result queue if configured
            if self.result_queue_url:
                self.send_result_to_queue(result)
            
            # Send callback if provided
            callback_url = message_data.get('callback_url')
            if callback_url:
                self.send_callback(result, callback_url)
            
            # Delete message from queue
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            
            logger.info(f"✅ Message {message_id} processed and deleted")
            self.stats['messages_processed'] += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to process message {message_id}: {e}")
            self.stats['messages_failed'] += 1
            
            # Optionally: send message to DLQ or handle retry logic
            # For now, we'll let the message return to queue after visibility timeout
    
    def start_consuming(self, 
                       message_handler: Optional[Callable] = None,
                       run_once: bool = False) -> None:
        """
        Start consuming messages from SQS.
        
        Args:
            message_handler: Custom message handler function (optional)
            run_once: If True, poll once and exit (for testing)
        """
        self.stats['start_time'] = datetime.utcnow()
        
        logger.info(f"🚀 Starting SQS consumer for queue: {self.queue_url}")
        logger.info(f"📊 Poll interval: {self.poll_interval}s, Max messages: {self.max_messages}")
        
        try:
            while True:
                self.poll_messages()
                
                if run_once:
                    break
                
                # Print stats periodically
                if self.stats['messages_processed'] % 100 == 0 and self.stats['messages_processed'] > 0:
                    self._print_stats()
                
                time.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Consumer stopped by user")
        except Exception as e:
            logger.error(f"❌ Consumer crashed: {e}")
            raise
        finally:
            self._print_stats()
    
    def _print_stats(self) -> None:
        """Print consumer statistics."""
        if self.stats['start_time']:
            runtime = datetime.utcnow() - self.stats['start_time']
            logger.info(f"📊 Stats - Processed: {self.stats['messages_processed']}, "
                       f"Failed: {self.stats['messages_failed']}, "
                       f"Results sent: {self.stats['results_sent']}, "
                       f"Callbacks sent: {self.stats['callbacks_sent']}, "
                       f"Runtime: {runtime}")

def create_test_message(sqs_to_ml_url: str, **kwargs) -> str:
    """
    Create a test SQS message for the given S3 URL.
    
    Args:
        sqs_to_ml_url: S3 URL to include in message
        **kwargs: Additional message fields
        
    Returns:
        JSON string of the message
    """
    message = {
        'sqs_to_ml_url': sqs_to_ml_url,
        'request_id': f"test_{int(time.time())}",
        **kwargs
    }
    return json.dumps(message, indent=2)

if __name__ == "__main__":
    # Example usage
    setup_logging()
    
    # Configuration from environment variables
    queue_url = os.getenv('QUEUE_URL', 'https://sqs.us-east-1.amazonaws.com/123456789012/deepfake-queue')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    
    try:
        consumer = SQSDeepfakeConsumer(
            queue_url=queue_url,
            aws_region=aws_region,
            models_dir='models'
        )
        
        # Start consuming (use run_once=True for testing)
        consumer.start_consuming(run_once=False)
        
    except Exception as e:
        logger.error(f"❌ Failed to start consumer: {e}")
        sys.exit(1) 