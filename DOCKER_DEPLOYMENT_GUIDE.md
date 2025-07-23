# Docker Deployment Guide for Deepfake Detection Service

This guide covers deploying the deepfake detection service as a Docker container that processes SQS messages and returns results to another SQS queue.

## Architecture Overview

```
Input SQS Queue → Docker Container → ML Processing → Result SQS Queue
                      ↓
                 S3 File Download
                      ↓
                 Deepfake Detection
                      ↓
                 Result Processing
```

## Container Specifications

**Current Docker Image:**
- **Name:** `deepfake-detection:latest`
- **Size:** 5.17GB (includes TensorFlow 2.19.0 + ML models)
- **Base:** Python 3.12.11 on Debian Bookworm
- **Architecture:** Multi-stage optimized build
- **Status:** Tested and verified working

**Verified Components:**
- Python 3.12.11 runtime
- TensorFlow 2.19.0 with CPU support
- Core application modules
- Configuration system
- Logging system with rotation
- SQS consumer functionality

## Quick Start

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
# Build the Docker image (verified working)
docker build -t deepfake-detection:latest .

# Run the container
docker run -d \
  --name deepfake-sqs-consumer \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_REGION=us-east-1 \
  -e QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/input-queue \
  -e RESULT_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/result-queue \
  -e LOG_LEVEL=INFO \
  -v ./models:/app/models:ro \
  -v ./logs:/app/logs \
  deepfake-detection:latest
```

### 3. Verify Container Functionality

```bash
# Quick functionality test (verified working)
docker run --rm -it deepfake-detection:latest python -c "
import sys
print('Python version:', sys.version)
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
from src.core import get_config
print('Core modules: OK')
config = get_config()
print('Configuration: OK')
print('Container test: PASSED')
"
```

## Configuration

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
| `PYTHONPATH` | Python module path | No | `/app` |
| `PYTHONUNBUFFERED` | Disable Python buffering | No | `1` |

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

## Local Testing

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

### Interactive Container Testing

```bash
# Run container interactively for debugging
docker run -it \
  --name deepfake-test \
  -e PYTHONPATH=/app \
  -v ./models:/app/models:ro \
  deepfake-detection:latest bash

# Inside container, test components:
python -c "from src.aws.sqs_deepfake_consumer import main; print('SQS consumer: OK')"
python -c "from src.core import get_config; print('Config system: OK')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

## AWS Deployment

### Option 1: AWS ECS/Fargate (Recommended)

```bash
# Tag image for ECR
docker tag deepfake-detection:latest your-account.dkr.ecr.us-east-1.amazonaws.com/deepfake-detection:latest

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/deepfake-detection:latest

# Deploy using ECS
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

# Add user to docker group
sudo usermod -a -G docker ec2-user

# Pull and run the image
docker pull your-account.dkr.ecr.us-east-1.amazonaws.com/deepfake-detection:latest
docker run -d \
  --name deepfake-sqs-consumer \
  --restart unless-stopped \
  --memory=4g \
  --cpus=2.0 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e QUEUE_URL=your_queue_url \
  -e RESULT_QUEUE_URL=your_result_queue_url \
  -v /opt/deepfake/models:/app/models:ro \
  -v /opt/deepfake/logs:/app/logs \
  your-account.dkr.ecr.us-east-1.amazonaws.com/deepfake-detection:latest
```

### Option 3: Docker Swarm / Kubernetes

```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepfake-detection
spec:
  replicas: 2
  selector:
    matchLabels:
      app: deepfake-detection
  template:
    metadata:
      labels:
        app: deepfake-detection
    spec:
      containers:
      - name: deepfake-sqs-consumer
        image: your-account.dkr.ecr.us-east-1.amazonaws.com/deepfake-detection:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: access-key-id
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: secret-access-key
        - name: QUEUE_URL
          value: "https://sqs.us-east-1.amazonaws.com/123456789012/input-queue"
        - name: RESULT_QUEUE_URL
          value: "https://sqs.us-east-1.amazonaws.com/123456789012/result-queue"
```

## Monitoring and Logging

### Docker Logs

```bash
# View container logs
docker logs deepfake-sqs-consumer

# Follow logs in real-time
docker logs -f deepfake-sqs-consumer

# View logs with timestamps
docker logs -t deepfake-sqs-consumer

# View last 100 lines
docker logs --tail 100 deepfake-sqs-consumer
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

# Manual health check (verified working)
docker exec deepfake-sqs-consumer python -c "
import sys
print('Health check: PASSED')
from src.core import get_config
config = get_config()
print('Configuration: LOADED')
sys.exit(0)
"

# Check resource usage
docker stats deepfake-sqs-consumer
```

## Performance and Resource Management

### Recommended Resource Allocation

| Deployment Type | Memory | CPU | Storage | Notes |
|----------------|--------|-----|---------|-------|
| Development | 2GB | 1 CPU | 10GB | Local testing |
| Staging | 4GB | 2 CPU | 20GB | Performance testing |
| Production | 6-8GB | 2-4 CPU | 50GB | High availability |

### Memory Management

```bash
# Monitor memory usage
docker exec deepfake-sqs-consumer python -c "
import psutil
import os
process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f'Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB')
"

# Set memory limits
docker run --memory=4g --memory-swap=4g deepfake-detection:latest
```

### CPU Optimization

```bash
# Set CPU limits
docker run --cpus=2.0 deepfake-detection:latest

# Check CPU usage
docker exec deepfake-sqs-consumer python -c "
import psutil
print(f'CPU usage: {psutil.cpu_percent()}%')
print(f'CPU count: {psutil.cpu_count()}')
"
```

## Security Best Practices

### 1. AWS Credentials Management

```bash
# Use AWS Secrets Manager (recommended for production)
aws secretsmanager create-secret \
  --name deepfake-detection/aws-credentials \
  --secret-string '{"AWS_ACCESS_KEY_ID":"your_key","AWS_SECRET_ACCESS_KEY":"your_secret"}'

# Use IAM roles for ECS tasks (preferred)
aws iam create-role --role-name ecsTaskRole --assume-role-policy-document file://trust-policy.json
aws iam attach-role-policy --role-name ecsTaskRole --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
aws iam attach-role-policy --role-name ecsTaskRole --policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess
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
# Run container as non-root user (add to Dockerfile)
RUN useradd -r -u 1001 deepfake-user
USER deepfake-user

# Use read-only filesystem where possible
docker run --read-only --tmpfs /tmp deepfake-detection:latest

# Limit resources
docker run --memory=4g --cpus=2.0 --pids-limit=100 deepfake-detection:latest
```

## Troubleshooting

### Container Build Issues

1. **UV sync fails with lock file error**
   ```bash
   # Solution: Ensure uv.lock is present and up to date
   uv lock
   docker build -t deepfake-detection:latest .
   ```

2. **System dependencies missing**
   ```bash
   # Check if ffmpeg and other dependencies are installed
   docker run --rm -it deepfake-detection:latest which ffmpeg
   docker run --rm -it deepfake-detection:latest python -c "import cv2; print('OpenCV: OK')"
   ```

### Runtime Issues

1. **Module import errors (RESOLVED)**
   
   **This issue has been fixed in the current version.** The container now properly:
   - Uses `PYTHONPATH=/app` environment variable
   - Executes modules with `python -m src.aws.sqs_deepfake_consumer`
   - Has correct virtual environment path in `PATH`
   
   If you still encounter import issues:
   ```bash
   # Verify container structure
   docker exec deepfake-sqs-consumer ls -la /app/src/
   
   # Test imports directly
   docker exec deepfake-sqs-consumer python -c "from src.aws.sqs_deepfake_consumer import main; print('Import successful')"
   
   # Check Python path
   docker exec deepfake-sqs-consumer python -c "import sys; print('\\n'.join(sys.path))"
   ```

2. **Container fails to start**
   ```bash
   # Check detailed logs
   docker logs deepfake-sqs-consumer
   
   # Check resource usage and limits
   docker stats deepfake-sqs-consumer
   
   # Check if ports are available
   docker run --rm -it deepfake-detection:latest netstat -tlnp
   ```

3. **AWS credentials not found**
   ```bash
   # Verify environment variables are set
   docker exec deepfake-sqs-consumer env | grep AWS
   
   # Test AWS connectivity
   docker exec deepfake-sqs-consumer python -c "
   import boto3
   try:
       client = boto3.client('sts')
       identity = client.get_caller_identity()
       print(f'AWS Identity: {identity}')
   except Exception as e:
       print(f'AWS Error: {e}')
   "
   ```

4. **SQS connection issues**
   ```bash
   # Test SQS connectivity
   docker exec deepfake-sqs-consumer python -c "
   import boto3
   import os
   queue_url = os.environ.get('QUEUE_URL')
   if queue_url:
       sqs = boto3.client('sqs')
       attrs = sqs.get_queue_attributes(QueueUrl=queue_url)
       print(f'Queue attributes: {attrs}')
   else:
       print('QUEUE_URL not set')
   "
   ```

5. **Model loading failures**
   ```bash
   # Check models directory and files
   docker exec deepfake-sqs-consumer ls -la /app/models/
   
   # Test model loading
   docker exec deepfake-sqs-consumer python -c "
   import os
   import tensorflow as tf
   models_dir = '/app/models'
   model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
   print(f'Available models: {model_files}')
   
   if model_files:
       try:
           model_path = os.path.join(models_dir, model_files[0])
           model = tf.keras.models.load_model(model_path)
           print(f'Successfully loaded: {model_files[0]}')
       except Exception as e:
           print(f'Model loading error: {e}')
   "
   ```

### Performance Issues

```bash
# Monitor resource usage
docker exec deepfake-sqs-consumer python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
print(f'CPU: {process.cpu_percent()}%')
print(f'Threads: {process.num_threads()}')
"

# Check disk usage
docker exec deepfake-sqs-consumer df -h

# Monitor network connections
docker exec deepfake-sqs-consumer netstat -an | grep ESTABLISHED
```

## Scaling and High Availability

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

# Create scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --resource-id service/deepfake-detection-cluster/deepfake-sqs-consumer \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name deepfake-scaling-policy \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

### Load Balancing

```bash
# For HTTP-based health checks, add ALB
aws elbv2 create-load-balancer \
  --name deepfake-detection-alb \
  --subnets subnet-12345678 subnet-87654321 \
  --security-groups sg-12345678
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy Deepfake Detection Service

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build, tag, and push image to Amazon ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: deepfake-detection
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
      
      - name: Deploy to ECS
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: deepfake-detection
          IMAGE_TAG: ${{ github.sha }}
        run: |
          aws ecs update-service \
            --cluster deepfake-detection-cluster \
            --service deepfake-sqs-consumer \
            --force-new-deployment
```

### Docker Compose for CI/CD

```yaml
# docker-compose.ci.yml
version: '3.8'

services:
  deepfake-detector:
    build: .
    image: deepfake-detection:${IMAGE_TAG:-latest}
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - QUEUE_URL=${QUEUE_URL}
      - RESULT_QUEUE_URL=${RESULT_QUEUE_URL}
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

## Backup and Recovery

### Model Backup

```bash
# Backup models to S3
aws s3 sync ./models/ s3://your-backup-bucket/deepfake-models/$(date +%Y%m%d)/

# Restore models from S3
aws s3 sync s3://your-backup-bucket/deepfake-models/20250101/ ./models/
```

### Configuration Backup

```bash
# Backup configuration
docker run --rm \
  -v ./backup:/backup \
  deepfake-detection:latest \
  tar czf /backup/config-$(date +%Y%m%d).tar.gz /app/src/core/
```

## Additional Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [AWS SQS Documentation](https://docs.aws.amazon.com/sqs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [AWS Security Best Practices](https://aws.amazon.com/security/security-resources/)
- [TensorFlow Docker Guide](https://www.tensorflow.org/install/docker)
- [Container Security Guide](https://docs.docker.com/engine/security/)

## Support and Maintenance

For issues and questions:
1. **Check this troubleshooting guide** for common solutions
2. **Review container logs** using `docker logs deepfake-sqs-consumer`
3. **Test individual components** using the provided verification scripts
4. **Monitor resource usage** with `docker stats` and CloudWatch
5. **Refer to the main project documentation** in `README.md`
6. **Check the testing guide** in `TESTING_README.md`

### Regular Maintenance Tasks

```bash
# Weekly: Clean up old containers and images
docker system prune -a

# Monthly: Update base images and rebuild
docker pull python:3.12-slim-bookworm
docker build -t deepfake-detection:latest .

# Monitor: Check logs for errors
docker logs deepfake-sqs-consumer | grep ERROR

# Health: Verify model performance
docker exec deepfake-sqs-consumer python -c "
from src.core import get_config
config = get_config()
print(f'System status: HEALTHY')
print(f'Environment: {config.environment}')
"
```

---

**Last Updated:** July 2025  
**Docker Image Version:** deepfake-detection:latest (5.17GB)  
**Verified Status:** Tested and working ✓ 