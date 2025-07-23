# AWS Integration Guide for Deepfake Detection

This comprehensive guide covers both Amazon S3 and SQS integration for the deepfake detection system, enabling cloud-based file processing and message queue automation.

## Table of Contents

1. [Overview](#overview)
2. [S3 Integration](#s3-integration)
3. [SQS Integration](#sqs-integration)
4. [Combined S3 + SQS Workflow](#combined-s3--sqs-workflow)
5. [AWS Credentials Configuration](#aws-credentials-configuration)
6. [Configuration & Security](#configuration--security)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)

## Overview

The deepfake detection system provides comprehensive AWS integration supporting:

- **S3 Integration**: Direct processing of files stored in Amazon S3 buckets
- **SQS Integration**: Automated processing via message queue system
- **Combined Workflow**: SQS messages triggering S3 file processing
- **Multiple Authentication**: Environment variables, CLI configuration, IAM roles
- **Production Ready**: Scalable, secure, and fully automated

### **Architecture Overview**

```
S3 Bucket → SQS Queue → Deepfake Detection → Results → Callback URL
    ↓           ↓              ↓              ↓           ↓
 Media Files → JSON Message → ML Models → JSON Response → POST
```

## S3 Integration

### **Core Components**

#### **S3 Utilities Module** (`src/utils/s3_utils.py`)
- S3FileHandler class for file operations
- URL validation and parsing
- File download with temporary storage
- Automatic cleanup functionality
- Multiple authentication methods support

#### **Enhanced Inference System** (`src/inference/predict_deepfake.py`)
- Automatic S3 URL detection
- Seamless integration with existing prediction pipeline
- AWS credentials support via command line arguments
- Enhanced result output with S3 metadata

#### **Dedicated S3 Script** (`src/aws/predict_s3_deepfake.py`)
- Simplified S3-focused interface
- Comprehensive error handling and user guidance
- Built-in AWS credentials setup instructions

### **Supported S3 URL Formats**

```bash
# Standard S3 URL format
s3://bucket-name/path/to/file.jpg

# HTTPS S3 URL format (also supported)
https://bucket-name.s3.region.amazonaws.com/path/to/file.mp4
```

### **Supported Media Types**

All existing media types are supported via S3:

- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- **Videos**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`
- **Audio**: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`

### **S3 Usage Examples**

#### **Basic S3 Detection**

```bash
# Simple S3 file detection
python src/aws/predict_s3_deepfake.py s3://capstone-development/file.jpg

# Using the general inference script
python src/inference/predict_deepfake.py --input s3://bucket/file.mp4
```

#### **Advanced S3 Usage**

```bash
# With custom AWS credentials
python src/aws/predict_s3_deepfake.py s3://bucket/file.jpg \
  --aws-access-key-id YOUR_KEY \
  --aws-secret-access-key YOUR_SECRET

# With custom confidence threshold
python src/aws/predict_s3_deepfake.py s3://bucket/file.wav --threshold 0.8

# Save results to file
python src/aws/predict_s3_deepfake.py s3://bucket/file.mp4 --output results.json

# Different AWS region
python src/aws/predict_s3_deepfake.py s3://bucket/file.jpg --aws-region eu-west-1
```

#### **S3 Setup and Help**

```bash
# Check AWS credentials setup
python src/aws/predict_s3_deepfake.py --setup-aws

# View help for S3-specific script
python src/aws/predict_s3_deepfake.py --help
```

### **S3 Programmatic Usage**

```python
from src.inference.predict_deepfake import EnhancedDeepfakePredictor

# Initialize with AWS credentials
predictor = EnhancedDeepfakePredictor(
    aws_access_key_id='your_access_key',
    aws_secret_access_key='your_secret_key',
    aws_region='us-east-1'
)

# Load models
predictor.load_models()

# Predict from S3 URL
result = predictor.predict_auto('s3://bucket/file.jpg')

print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"S3 File: {result['is_s3_file']}")
print(f"Downloaded to: {result['local_path']}")
```

## SQS Integration

### **SQS Architecture**

The SQS integration enables automated processing of deepfake detection requests through a message queue system:

```
SQS Queue → SQS Consumer → Deepfake Detection → Results → Callback URL
     ↓              ↓              ↓              ↓           ↓
JSON Message  →  File Download  →  ML Models   →  JSON     →  POST
```

### **Message Format**

The SQS consumer expects JSON messages with the following structure:

```json
{
  "sqs_to_ml_url": "s3://bucket/path/to/file.ext",
  "callback_url": "https://api.example.com/callback",
  "request_id": "unique-request-id",
  "confidence_threshold": 0.5,
  "metadata": {
    "user_id": "12345",
    "upload_timestamp": "2025-01-01T00:00:00Z",
    "original_filename": "file.ext"
  }
}
```

#### **Required Fields**
- `sqs_to_ml_url`: S3 URL of the file to analyze (supports images, videos, audio)

#### **Optional Fields**
- `callback_url`: URL to POST results to after processing
- `request_id`: Unique identifier for tracking
- `confidence_threshold`: Custom threshold for classification (default: 0.5)
- `metadata`: Additional data to include in results

### **SQS Quick Start**

#### **1. Install Dependencies**

```bash
# Core dependencies (already in pyproject.toml)
pip install boto3 requests

# Or install all project dependencies
pip install -e .
```

#### **2. Set Environment Variables**

```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"
export QUEUE_URL="https://sqs.us-east-1.amazonaws.com/123456789012/deepfake-queue"
```

#### **3. Start the Consumer**

```bash
# Start with environment variables
python src/aws/sqs_deepfake_consumer.py

# Or with command line arguments
python src/aws/sqs_deepfake_consumer.py \
  --queue-url "https://sqs.us-east-1.amazonaws.com/123456789012/deepfake-queue" \
  --aws-access-key-id YOUR_KEY \
  --aws-secret-access-key YOUR_SECRET
```

### **SQS Configuration Options**

#### **SQS Configuration**
- `--queue-url`: AWS SQS queue URL
- `--poll-interval`: Polling interval in seconds (default: 5)
- `--max-messages`: Max messages per poll (default: 10)
- `--wait-time`: Long polling wait time (default: 20)
- `--visibility-timeout`: Message visibility timeout (default: 30)

#### **Model Configuration**
- `--models-dir`: Directory containing trained models (default: models)

#### **Operational Options**
- `--run-once`: Poll once and exit
- `--dry-run`: Parse messages but don't process
- `--stats-interval`: Print stats every N messages

### **SQS Result Format**

After processing, the system returns a comprehensive result:

```json
{
  "request_id": "unique-request-id",
  "sqs_to_ml_url": "s3://bucket/file.ext",
  "file_path": "s3://bucket/file.ext",
  "local_path": "/tmp/deepfake_s3_cache/file.ext",
  "file_type": "image",
  "label": "FAKE",
  "confidence": 0.73,
  "prediction_score": 0.7312,
  "is_s3_file": true,
  "processed_at": "2025-01-01T12:00:00.000Z",
  "queue_url": "https://sqs.us-east-1.amazonaws.com/123456789012/deepfake-queue",
  "s3_file_info": {
    "size": 2048576,
    "content_type": "image/png",
    "last_modified": "2025-01-01T10:00:00.000Z"
  },
  "original_message": {
    "sqs_to_ml_url": "s3://bucket/file.ext",
    "callback_url": "https://api.example.com/callback",
    "request_id": "unique-request-id"
  }
}
```

### **SQS Development & Testing**

#### **Test Message Creation**

```bash
# Generate test message for your S3 URL
python src/aws/sqs_deepfake_consumer.py --create-test-message \
  "s3://capstone-development/f84ca5f4-ff77-4f0e-8da1-67a3105a9f52.png"

# Interactive test script
python src/aws/test_sqs_message.py

# Validate queue connection
python src/aws/sqs_deepfake_consumer.py --validate-queue \
  --queue-url "https://sqs.us-east-1.amazonaws.com/123456789012/deepfake-queue"

# Run once for testing
python src/aws/sqs_deepfake_consumer.py --run-once
```

#### **Programmatic SQS Usage**

```python
from src.utils.sqs_consumer import SQSDeepfakeConsumer, create_test_message

# Create test message
message = create_test_message(
    sqs_to_ml_url="s3://capstone-development/file.png",
    callback_url="https://api.example.com/callback",
    confidence_threshold=0.7
)

# Initialize consumer
consumer = SQSDeepfakeConsumer(
    queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/deepfake-queue",
    aws_region="us-east-1"
)

# Process messages once
consumer.start_consuming(run_once=True)

# Custom message handler
def custom_handler(result):
    print(f"Processed {result['request_id']}: {result['label']}")
    
consumer.start_consuming(message_handler=custom_handler)
```

## Combined S3 + SQS Workflow

### **End-to-End Processing**

The most powerful integration combines both S3 and SQS for automated cloud processing:

1. **Files uploaded to S3 bucket**
2. **S3 event triggers message to SQS queue**
3. **SQS consumer polls for messages**
4. **Consumer downloads file from S3**
5. **Deepfake detection runs on downloaded file**
6. **Results sent to callback URL**
7. **Temporary files cleaned up**

### **Example Workflow**

```json
{
  "sqs_to_ml_url": "s3://capstone-development/video.mp4",
  "callback_url": "https://myapi.com/deepfake-results",
  "request_id": "req_001",
  "confidence_threshold": 0.6,
  "metadata": {
    "user_id": "user123",
    "upload_timestamp": "2025-01-01T12:00:00Z"
  }
}
```

**Processing Flow:**
1. SQS message received with S3 URL
2. File downloaded from S3 (video.mp4)
3. Video deepfake detection executed
4. Results posted to callback URL
5. Message deleted from SQS queue
6. Temporary file cleaned up

### **Integration with Existing System**

The SQS consumer leverages the modular architecture:

- **`src/inference/common.py`**: Common functions for validation, error handling
- **`src/inference/predict_deepfake.py`**: Core prediction logic
- **`src/utils/s3_utils.py`**: S3 file handling
- **`PredictionContext`**: Automatic cleanup of temporary files

## AWS Credentials Configuration

### **Quick Setup**

1. **Copy the template configuration:**
   ```bash
   cp aws_config_template.json aws_config.json
   ```

2. **Edit `aws_config.json` with your AWS credentials:**
   ```json
   {
     "aws": {
       "access_key_id": "YOUR_ACTUAL_ACCESS_KEY_ID",
       "secret_access_key": "YOUR_ACTUAL_SECRET_ACCESS_KEY", 
       "region": "us-east-1",
       "session_token": null
     }
   }
   ```

3. **The `aws_config.json` file is automatically ignored by git** (see `.gitignore`), so your credentials won't be committed.

### **Configuration Options**

#### **1. Configuration File (Recommended)**
- Create `aws_config.json` from the template
- Secure and easy to manage
- Automatically excluded from version control

#### **2. Environment Variables**
Set these environment variables and they'll be loaded automatically:
```bash
export AWS_ACCESS_KEY_ID="your_access_key_id"
export AWS_SECRET_ACCESS_KEY="your_secret_access_key"
export AWS_REGION="us-east-1"
```

#### **3. Custom Configuration File**
```bash
python src/aws/predict_s3_deepfake.py s3://bucket/file.jpg --config my_custom_aws_config.json
```

### **Configuration Usage Examples**

#### **Basic Usage**
```bash
# Uses aws_config.json by default
python src/aws/predict_s3_deepfake.py s3://bucket/image.jpg
```

#### **With Custom Config**
```bash
# Uses specific config file
python src/aws/predict_s3_deepfake.py s3://bucket/video.mp4 --config prod_aws_config.json
```

#### **With Threshold and Output**
```bash
python src/aws/predict_s3_deepfake.py s3://bucket/audio.wav --threshold 0.8 --output results.json
```

### **Configuration Schema**

The complete configuration file supports these options:

```json
{
  "aws": {
    "access_key_id": "string",
    "secret_access_key": "string",
    "region": "string (default: us-east-1)",
    "session_token": "string (optional)"
  },
  "data": {
    "temp_dir": "string (default: /tmp/deepfake_cache)",
    "s3_cache_dir": "string (default: /tmp/deepfake_s3_cache)",
    "cleanup_temp_files": "boolean (default: true)"
  },
  "logging": {
    "level": "string (default: INFO)",
    "log_to_console": "boolean (default: true)"
  }
}
```

Only the AWS credentials are required - other settings have sensible defaults.

### **Interactive Setup Script**

Use the interactive setup script for easy configuration:

```bash
python src/aws/setup_aws_config.py
```

This script provides:
- Interactive credential input
- Configuration validation
- Status checking
- Usage examples

## Configuration & Security

### **Authentication Methods**

#### **1. Environment Variables (Recommended)**
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```

#### **2. AWS CLI Configuration**
```bash
aws configure
```

#### **3. Command Line Arguments**
```bash
--aws-access-key-id YOUR_KEY --aws-secret-access-key YOUR_SECRET
```

#### **4. IAM Roles (for EC2 instances)**
- Automatic detection and usage

### **Required AWS Permissions**

#### **SQS Permissions**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:GetQueueAttributes"
      ],
      "Resource": "arn:aws:sqs:region:account:queue-name"
    }
  ]
}
```

#### **S3 Permissions**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:GetObjectMetadata",
        "s3:GetObjectVersion",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::bucket-name",
        "arn:aws:s3:::bucket-name/*"
      ]
    }
  ]
}
```

### **Security Best Practices**

1. **Never commit `aws_config.json`** - It's automatically excluded by `.gitignore`
2. **Use IAM roles** when possible instead of access keys
3. **Rotate credentials** regularly
4. **Use least privilege** - only grant necessary S3 permissions
5. **Consider using AWS profiles** instead of hardcoded credentials
6. **Use IAM roles when running on EC2**
7. **Enable CloudTrail for audit logging**
8. **Use VPC endpoints for private communication**

## Usage Examples

### **Basic S3 Processing**

```bash
# Direct S3 file processing
python src/aws/predict_s3_deepfake.py s3://bucket/image.jpg

# With custom threshold
python src/aws/predict_s3_deepfake.py s3://bucket/video.mp4 --threshold 0.8

# Save results to JSON
python src/aws/predict_s3_deepfake.py s3://bucket/audio.wav --output results.json
```

### **SQS Message Processing**

```bash
# Send message to SQS (via AWS CLI)
aws sqs send-message \
  --queue-url "YOUR_QUEUE_URL" \
  --message-body '{
    "sqs_to_ml_url": "s3://capstone-development/file.png",
    "request_id": "test_001",
    "callback_url": "https://api.example.com/callback"
  }'

# Start SQS consumer
python src/aws/sqs_deepfake_consumer.py --queue-url "YOUR_QUEUE_URL"
```

### **Production Deployment**

#### **Docker Setup**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -e .

# Environment variables
ENV QUEUE_URL=""
ENV AWS_REGION="us-east-1"
ENV MODELS_DIR="/app/models"
ENV LOG_LEVEL="INFO"

CMD ["python", "src/aws/sqs_deepfake_consumer.py"]
```

#### **Environment Variables for Production**
```bash
QUEUE_URL=https://sqs.us-east-1.amazonaws.com/account/queue
AWS_REGION=us-east-1
MODELS_DIR=/app/models
LOG_LEVEL=INFO
POLL_INTERVAL=5
MAX_MESSAGES=10
```

### **Monitoring & Logging**

#### **Built-in Statistics**
The consumer tracks:
- Messages processed
- Messages failed
- Runtime duration
- Processing rates

#### **Example Log Output**
```
2025-01-01 12:00:00,000 - INFO - Starting SQS consumer for queue: https://sqs.us-east-1.amazonaws.com/123456789012/deepfake-queue
2025-01-01 12:00:01,000 - INFO - Loaded models: ['image', 'video', 'audio']
2025-01-01 12:00:02,000 - INFO - Received 1 message(s)
2025-01-01 12:00:03,000 - INFO - Processing deepfake detection for: s3://bucket/file.jpg (ID: req_12345)
2025-01-01 12:00:05,000 - INFO - Detection complete - FAKE (confidence: 73.1%) for req_12345
2025-01-01 12:00:06,000 - INFO - Message processed and deleted
```

## Troubleshooting

### **Common Issues and Solutions**

#### **S3 Access Issues**

**S3 Access Denied:**
```
Failed to download S3 file: Access Denied
Check S3 bucket permissions and AWS credentials
```

**S3 File Not Found:**
```
S3 file not found: s3://bucket/file.jpg
Verify the S3 URL and file exists
```

#### **SQS Connection Issues**

**SQS Connection Errors:**
```
AWS credentials not found!
Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables
```

**Queue Not Found:**
```
SQS queue does not exist: https://sqs.us-east-1.amazonaws.com/123456789012/wrong-queue
Verify the queue URL and region
```

#### **Authentication Issues**

**Invalid Credentials:**
```bash
# Test AWS credentials
aws sts get-caller-identity

# Setup AWS credentials
python src/aws/predict_s3_deepfake.py --setup-aws
```

#### **Configuration Issues**

**Missing Configuration:**
```
Error: AWS credentials not found
```
**Solution:** Ensure you have either:
- A `aws_config.json` file with valid credentials
- Environment variables set
- Or specify a custom config file with `--config`

**Invalid Credentials:**
```
Error: Access denied
```
**Solution:** Verify your AWS credentials are correct and have S3 access permissions.

**Config File Not Found:**
```
Error: Configuration file not found
```
**Solution:** Create `aws_config.json` from `aws_config_template.json` or specify the correct path with `--config`.

#### **Model Loading Issues**

**No Models Found:**
```
No models could be loaded!
Train models using: python src/training/train_and_save_models.py --demo
```

**Model Directory Issues:**
```bash
# Check model directory
ls -la models/

# Verify models exist
python src/inference/predict_deepfake.py --list-models
```

### **Debug Mode**

```bash
# Enable debug logging for S3
export LOG_LEVEL=DEBUG
python src/aws/predict_s3_deepfake.py s3://bucket/file.jpg --verbose

# Enable debug logging for SQS
export LOG_LEVEL=DEBUG
python src/aws/sqs_deepfake_consumer.py --run-once
```

### **Scaling Considerations**

#### **Multiple Consumers**
- Run multiple instances for higher throughput
- Configure Dead Letter Queue for failed messages
- Monitor queue depth and processing metrics
- Use Auto Scaling based on queue depth

#### **Performance Optimization**
- Use larger EC2 instances for video processing
- Implement batch processing for multiple files
- Cache models to reduce startup time
- Use CloudWatch for monitoring and alerting

## Use Cases

### **Real-World Applications**

1. **Mobile App Backend**: Process user-uploaded content stored in S3
2. **Batch Processing**: Analyze large datasets stored in S3 buckets
3. **API Integration**: Provide deepfake detection as a service for S3 files
4. **Research Workflows**: Analyze research datasets stored in cloud storage
5. **Content Moderation**: Real-time analysis of uploaded media files
6. **Automated Pipelines**: Process files as they're uploaded to S3

### **Integration Examples**

#### **Web Application Integration**
```python
# Upload file to S3, then send SQS message
import boto3

s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')

# Upload file to S3
s3_client.upload_file('local_file.jpg', 'bucket', 'file.jpg')

# Send SQS message for processing
message = {
    "sqs_to_ml_url": "s3://bucket/file.jpg",
    "callback_url": "https://myapp.com/deepfake-result",
    "request_id": "req_12345"
}

sqs_client.send_message(
    QueueUrl='YOUR_QUEUE_URL',
    MessageBody=json.dumps(message)
)
```

#### **Lambda Function Integration**
```python
import json
import boto3

def lambda_handler(event, context):
    """Lambda function triggered by S3 upload"""
    
    # Extract S3 details from event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Send SQS message for deepfake detection
    sqs = boto3.client('sqs')
    message = {
        "sqs_to_ml_url": f"s3://{bucket}/{key}",
        "request_id": context.aws_request_id,
        "callback_url": "https://myapi.com/deepfake-result"
    }
    
    sqs.send_message(
        QueueUrl=os.environ['QUEUE_URL'],
        MessageBody=json.dumps(message)
    )
    
    return {'statusCode': 200}
```

## Production Readiness

### **Features Implemented**

#### **S3 File Operations**
- [x] Parse S3 URLs (s3:// and https:// formats)
- [x] Download files to temporary storage
- [x] Get file metadata (size, content type, etc.)
- [x] Automatic cleanup of temporary files
- [x] Error handling for missing files/permissions

#### **SQS Message Processing**
- [x] JSON message parsing and validation
- [x] Long polling for efficient message retrieval
- [x] Automatic message deletion after processing
- [x] Dead letter queue support
- [x] Comprehensive error handling and logging

#### **Integration Features**
- [x] Seamless integration with existing inference pipeline
- [x] Auto-detection of S3 URLs vs local files
- [x] Support for all media types (image, video, audio)
- [x] Enhanced result output with S3 and SQS metadata
- [x] Callback URL support for result delivery

#### **Security & Authentication**
- [x] Multiple authentication methods
- [x] IAM role support for EC2
- [x] Least-privilege permissions
- [x] Secure credential handling

### **System Requirements**

- Python 3.9+
- boto3 and botocore packages
- Valid AWS account and credentials
- Appropriate S3 and SQS permissions

## Getting Started

### **Quick Setup**

1. **Install dependencies**: `pip install -e .`
2. **Setup AWS credentials**: `python src/aws/predict_s3_deepfake.py --setup-aws`
3. **Test S3 integration**: `python src/aws/predict_s3_deepfake.py s3://your-bucket/file.jpg`
4. **Test SQS integration**: `python src/aws/test_sqs_message.py`
5. **Start SQS consumer**: `python src/aws/sqs_deepfake_consumer.py --queue-url YOUR_QUEUE_URL`

### **Next Steps**

1. **Configure your S3 buckets** with appropriate permissions
2. **Create SQS queues** for message processing
3. **Set up monitoring** with CloudWatch
4. **Deploy consumer instances** for production scaling
5. **Integrate with your applications** using the provided APIs

---

**The deepfake detection system now provides comprehensive AWS integration, enabling scalable cloud-based processing with both S3 file storage and SQS message queuing!** 