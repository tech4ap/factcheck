#!/bin/bash

# AWS ECS Deployment Script for Deepfake Detection Service
# This script helps deploy the Docker container to AWS ECS

set -e

# Configuration
CLUSTER_NAME="deepfake-detection-cluster"
SERVICE_NAME="deepfake-sqs-consumer"
TASK_DEFINITION_NAME="deepfake-detection-task"
ECR_REPOSITORY="deepfake-detection"
IMAGE_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 AWS ECS Deployment Script for Deepfake Detection${NC}"
echo "================================================"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install it first.${NC}"
    exit 1
fi

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region)

if [ -z "$AWS_REGION" ]; then
    AWS_REGION="us-east-1"
    echo -e "${YELLOW}⚠️  No AWS region configured, using default: $AWS_REGION${NC}"
fi

ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY"

echo -e "${GREEN}📋 Deployment Configuration:${NC}"
echo "  AWS Account ID: $AWS_ACCOUNT_ID"
echo "  AWS Region: $AWS_REGION"
echo "  ECR Repository: $ECR_URI"
echo "  ECS Cluster: $CLUSTER_NAME"
echo "  ECS Service: $SERVICE_NAME"
echo ""

# Function to create ECR repository if it doesn't exist
create_ecr_repository() {
    echo -e "${GREEN}🔍 Checking ECR repository...${NC}"
    
    if aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION &> /dev/null; then
        echo -e "${GREEN}✅ ECR repository exists${NC}"
    else
        echo -e "${YELLOW}📦 Creating ECR repository...${NC}"
        aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION
        echo -e "${GREEN}✅ ECR repository created${NC}"
    fi
}

# Function to build and push Docker image
build_and_push_image() {
    echo -e "${GREEN}🔨 Building Docker image...${NC}"
    
    # Build the image
    docker build -t $ECR_REPOSITORY:$IMAGE_TAG .
    
    # Tag for ECR
    docker tag $ECR_REPOSITORY:$IMAGE_TAG $ECR_URI:$IMAGE_TAG
    
    # Login to ECR
    echo -e "${GREEN}🔐 Logging in to ECR...${NC}"
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI
    
    # Push the image
    echo -e "${GREEN}📤 Pushing image to ECR...${NC}"
    docker push $ECR_URI:$IMAGE_TAG
    
    echo -e "${GREEN}✅ Image pushed successfully${NC}"
}

# Function to create ECS task definition
create_task_definition() {
    echo -e "${GREEN}📝 Creating ECS task definition...${NC}"
    
    cat > task-definition.json << EOF
{
    "family": "$TASK_DEFINITION_NAME",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048",
    "executionRoleArn": "arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsTaskRole",
    "containerDefinitions": [
        {
            "name": "$SERVICE_NAME",
            "image": "$ECR_URI:$IMAGE_TAG",
            "essential": true,
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/$TASK_DEFINITION_NAME",
                    "awslogs-region": "$AWS_REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "environment": [
                {
                    "name": "AWS_REGION",
                    "value": "$AWS_REGION"
                },
                {
                    "name": "LOG_LEVEL",
                    "value": "INFO"
                },
                {
                    "name": "PYTHONUNBUFFERED",
                    "value": "1"
                }
            ],
            "secrets": [
                {
                    "name": "AWS_ACCESS_KEY_ID",
                    "valueFrom": "arn:aws:secretsmanager:$AWS_REGION:$AWS_ACCOUNT_ID:secret:deepfake-detection/aws-credentials:AWS_ACCESS_KEY_ID::"
                },
                {
                    "name": "AWS_SECRET_ACCESS_KEY",
                    "valueFrom": "arn:aws:secretsmanager:$AWS_REGION:$AWS_ACCOUNT_ID:secret:deepfake-detection/aws-credentials:AWS_SECRET_ACCESS_KEY::"
                },
                {
                    "name": "QUEUE_URL",
                    "valueFrom": "arn:aws:secretsmanager:$AWS_REGION:$AWS_ACCOUNT_ID:secret:deepfake-detection/sqs-queues:QUEUE_URL::"
                },
                {
                    "name": "RESULT_QUEUE_URL",
                    "valueFrom": "arn:aws:secretsmanager:$AWS_REGION:$AWS_ACCOUNT_ID:secret:deepfake-detection/sqs-queues:RESULT_QUEUE_URL::"
                }
            ]
        }
    ]
}
EOF

    # Register task definition
    aws ecs register-task-definition --cli-input-json file://task-definition.json --region $AWS_REGION
    
    echo -e "${GREEN}✅ Task definition created${NC}"
}

# Function to create ECS cluster if it doesn't exist
create_cluster() {
    echo -e "${GREEN}🔍 Checking ECS cluster...${NC}"
    
    if aws ecs describe-clusters --clusters $CLUSTER_NAME --region $AWS_REGION --query 'clusters[0].status' --output text | grep -q "ACTIVE"; then
        echo -e "${GREEN}✅ ECS cluster exists and is active${NC}"
    else
        echo -e "${YELLOW}🏗️  Creating ECS cluster...${NC}"
        aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $AWS_REGION
        echo -e "${GREEN}✅ ECS cluster created${NC}"
    fi
}

# Function to create or update ECS service
create_or_update_service() {
    echo -e "${GREEN}🔍 Checking ECS service...${NC}"
    
    if aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION --query 'services[0].status' --output text 2>/dev/null | grep -q "ACTIVE"; then
        echo -e "${YELLOW}🔄 Updating existing ECS service...${NC}"
        aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --task-definition $TASK_DEFINITION_NAME --region $AWS_REGION
    else
        echo -e "${YELLOW}🆕 Creating new ECS service...${NC}"
        
        # Note: You'll need to replace these subnet and security group IDs with your actual values
        echo -e "${RED}⚠️  Please update the subnet and security group IDs in the script before running${NC}"
        
        # aws ecs create-service \
        #     --cluster $CLUSTER_NAME \
        #     --service-name $SERVICE_NAME \
        #     --task-definition $TASK_DEFINITION_NAME \
        #     --desired-count 1 \
        #     --launch-type FARGATE \
        #     --network-configuration "awsvpcConfiguration={subnets=[subnet-12345678],securityGroups=[sg-12345678],assignPublicIp=ENABLED}" \
        #     --region $AWS_REGION
    fi
    
    echo -e "${GREEN}✅ ECS service configured${NC}"
}

# Main deployment flow
echo -e "${GREEN}🚀 Starting deployment...${NC}"

create_ecr_repository
build_and_push_image
create_task_definition
create_cluster
create_or_update_service

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo ""
echo -e "${YELLOW}📋 Next steps:${NC}"
echo "1. Create AWS Secrets Manager secrets for credentials"
echo "2. Update subnet and security group IDs in the script"
echo "3. Create CloudWatch log group: /ecs/$TASK_DEFINITION_NAME"
echo "4. Ensure IAM roles (ecsTaskExecutionRole, ecsTaskRole) exist with proper permissions"
echo ""
echo -e "${GREEN}🔗 Useful commands:${NC}"
echo "  Monitor service: aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME"
echo "  View logs: aws logs tail /ecs/$TASK_DEFINITION_NAME --follow"
echo "  Stop service: aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --desired-count 0" 