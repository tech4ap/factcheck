# Docker Deployment Guide for Deepfake Detection Service

This guide covers deploying the deepfake detection service as a Docker container that processes SQS messages and returns results to another SQS queue.

## 🏗️ Architecture Overview

```
Input SQS Queue → Docker Container → ML Processing → Result SQS Queue
                      ↓
                 S3 File Download
                      ↓
                 Deepfake Detection
                      ↓
                 Result Processing
```

## 🚀 Quick Start

### 1. Local Development with Docker Compose

```bash
# 1. Copy environment template
cp docker.env.template .env

# 2. Edit .env file with your AWS credentials and queue URLs
nano .env

# 3. Build and run the service
docker-compose up --build

# 4. For local testing with LocalStack
docker-compose --profile local-testing up
```

### 2. Build and Run Manually

```bash
# Build the Docker image
docker build -t deepfake-detection .

# Run the container
docker run -d \
  --name deepfake-sqs-consumer \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_REGION=us-east-1 \
  -e QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/input-queue \
  -e RESULT_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/result-queue \
  -v ./models:/app/models:ro \
  -v ./logs:/app/logs \
  deepfake-detection
```

## 📋 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key ID | Yes | - |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key | Yes | - |
| `AWS_REGION` | AWS region | No | `us-east-1` |
| `QUEUE_URL` | Input SQS queue URL | Yes | - |
| `RESULT_QUEUE_URL` | Result SQS queue URL | No | - |
| `LOG_LEVEL` | Logging level | No | `INFO` |
| `MODELS_DIR` | Models directory path | No | `/app/models` |

### SQS Message Format

#### Input Message (Expected)
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

#### Result Message (Sent to Result Queue)
```json
{
  "timestamp": "2025-01-01T12:00:00.000Z",
  "source_queue": "https://sqs.us-east-1.amazonaws.com/123456789012/input-queue",
  "result": {
    "request_id": "unique-request-id",
    "sqs_to_ml_url": "s3://bucket/file.ext",
    "file_type": "image",
    "label": "FAKE",
    "confidence": 0.73,
    "prediction_score": 0.7312,
    "is_s3_file": true,
    "processed_at": "2025-01-01T12:00:00.000Z",
    "original_message": { ... }
  }
}
```

## 🔧 Local Testing

### Using LocalStack for SQS

```bash
# Start LocalStack
docker-compose --profile local-testing up localstack

# Create test queues
aws --endpoint-url=http://localhost:4566 sqs create-queue --queue-name deepfake-input-queue
aws --endpoint-url=http://localhost:4566 sqs create-queue --queue-name deepfake-result-queue

# Send test message
aws --endpoint-url=http://localhost:4566 sqs send-message \
  --queue-url http://localhost:4566/000000000000/deepfake-input-queue \
  --message-body '{"sqs_to_ml_url": "s3://test-bucket/test-file.jpg", "request_id": "test-123"}'

# Start the service with LocalStack configuration
docker-compose up deepfake-detector
```

### Testing with AWS SQS

```bash
# Send test message using the test script
python src/aws/send_test_message.py

# Or manually send a message
aws sqs send-message \
  --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/deepfake-input-queue \
  --message-body '{"sqs_to_ml_url": "s3://your-bucket/test-file.jpg", "request_id": "test-123"}'
```

## 🌐 AWS Deployment

### Option 1: AWS ECS/Fargate (Recommended)

```bash
# Use the provided deployment script
./deploy/aws-deploy.sh

# Or manually deploy using AWS CLI
aws ecs create-cluster --cluster-name deepfake-detection-cluster
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs create-service --cluster deepfake-detection-cluster --service-name deepfake-sqs-consumer --task-definition deepfake-detection-task
```

### Option 2: AWS EC2

```bash
# On EC2 instance
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker

# Pull and run the image
docker pull your-ecr-repo/deepfake-detection:latest
docker run -d \
  --name deepfake-sqs-consumer \
  --restart unless-stopped \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e QUEUE_URL=your_queue_url \
  -e RESULT_QUEUE_URL=your_result_queue_url \
  your-ecr-repo/deepfake-detection:latest
```

### Option 3: AWS Lambda (For lightweight processing)

```bash
# Create Lambda deployment package
zip -r deepfake-lambda.zip src/ models/

# Deploy using AWS CLI
aws lambda create-function \
  --function-name deepfake-sqs-processor \
  --runtime python3.9 \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --handler src.aws.lambda_handler.handler \
  --zip-file fileb://deepfake-lambda.zip \
  --timeout 900 \
  --memory-size 3008
```

## 📊 Monitoring and Logging

### Docker Logs

```bash
# View container logs
docker logs deepfake-sqs-consumer

# Follow logs in real-time
docker logs -f deepfake-sqs-consumer

# View logs with timestamps
docker logs -t deepfake-sqs-consumer
```

### AWS CloudWatch (for ECS deployment)

```bash
# View ECS service logs
aws logs tail /ecs/deepfake-detection-task --follow

# Create CloudWatch dashboard
aws cloudwatch put-dashboard --dashboard-name deepfake-detection --dashboard-body file://cloudwatch-dashboard.json
```

### Health Checks

```bash
# Check container health
docker inspect deepfake-sqs-consumer | grep -A 5 "Health"

# Manual health check
docker exec deepfake-sqs-consumer python -c "import sys; sys.exit(0)"
```

## 🔐 Security Best Practices

### 1. AWS Credentials Management

```bash
# Use AWS Secrets Manager (recommended for production)
aws secretsmanager create-secret \
  --name deepfake-detection/aws-credentials \
  --secret-string '{"AWS_ACCESS_KEY_ID":"your_key","AWS_SECRET_ACCESS_KEY":"your_secret"}'

# Use IAM roles for ECS tasks (preferred)
aws iam create-role --role-name ecsTaskRole --assume-role-policy-document file://trust-policy.json
aws iam attach-role-policy --role-name ecsTaskRole --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
```

### 2. Network Security

```bash
# Use VPC endpoints for SQS access
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-12345678 \
  --service-name com.amazonaws.us-east-1.sqs \
  --route-table-ids rtb-12345678
```

### 3. Container Security

```bash
# Run container as non-root user
docker run --user 1000:1000 deepfake-detection

# Use read-only filesystem
docker run --read-only deepfake-detection

# Limit resources
docker run --memory=2g --cpus=1.0 deepfake-detection
```

## 🛠️ Troubleshooting

### Common Issues

1. **Container fails to start**
   ```bash
   # Check logs
   docker logs deepfake-sqs-consumer
   
   # Check resource usage
   docker stats deepfake-sqs-consumer
   ```

2. **AWS credentials not found**
   ```bash
   # Verify environment variables
   docker exec deepfake-sqs-consumer env | grep AWS
   
   # Test AWS connectivity
   docker exec deepfake-sqs-consumer aws sts get-caller-identity
   ```

3. **SQS connection issues**
   ```bash
   # Test SQS connectivity
   docker exec deepfake-sqs-consumer python -c "
   import boto3
   sqs = boto3.client('sqs')
   print(sqs.get_queue_attributes(QueueUrl='YOUR_QUEUE_URL'))
   "
   ```

4. **Model loading failures**
   ```bash
   # Check models directory
   docker exec deepfake-sqs-consumer ls -la /app/models/
   
   # Verify model files
   docker exec deepfake-sqs-consumer python -c "
   import tensorflow as tf
   model = tf.keras.models.load_model('/app/models/image_model_final.h5')
   print('Model loaded successfully')
   "
   ```

### Performance Tuning

```bash
# Adjust memory limits
docker run --memory=4g deepfake-detection

# Adjust CPU limits
docker run --cpus=2.0 deepfake-detection

# Use GPU support (if available)
docker run --gpus all deepfake-detection
```

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale ECS service
aws ecs update-service \
  --cluster deepfake-detection-cluster \
  --service deepfake-sqs-consumer \
  --desired-count 3

# Auto-scaling based on SQS queue depth
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/deepfake-detection-cluster/deepfake-sqs-consumer \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 1 \
  --max-capacity 10
```

### Vertical Scaling

```bash
# Update task definition with more resources
aws ecs register-task-definition \
  --family deepfake-detection-task \
  --cpu 2048 \
  --memory 4096 \
  --container-definitions file://container-definitions.json
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy Deepfake Detection Service

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Build and push Docker image
        run: |
          docker build -t deepfake-detection .
          docker tag deepfake-detection:latest $ECR_REGISTRY/deepfake-detection:latest
          docker push $ECR_REGISTRY/deepfake-detection:latest
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster deepfake-detection-cluster \
            --service deepfake-sqs-consumer \
            --force-new-deployment
```

## 📚 Additional Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [AWS SQS Documentation](https://docs.aws.amazon.com/sqs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [AWS Security Best Practices](https://aws.amazon.com/security/security-resources/)

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review container logs: `docker logs deepfake-sqs-consumer`
3. Test individual components using the provided scripts
4. Refer to the main project documentation in `README.md` 