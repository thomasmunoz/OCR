#!/bin/bash
# Deployment script for Dual-AI OCR Lambda Function

set -e

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
PROJECT_NAME="dual-ai-ocr"
ENVIRONMENT="${ENVIRONMENT:-prod}"

echo "=========================================="
echo "Deploying Dual-AI OCR Lambda to AWS"
echo "=========================================="
echo "Region: $AWS_REGION"
echo "Account: $AWS_ACCOUNT_ID"
echo "Environment: $ENVIRONMENT"
echo ""

# Step 1: Build Docker image
echo "Step 1: Building Docker image..."
cd lambda
docker build -t ${PROJECT_NAME}:latest .

# Step 2: Create ECR repository (if not exists)
echo "Step 2: Setting up ECR repository..."
aws ecr describe-repositories --repository-names ${PROJECT_NAME}-lambda --region $AWS_REGION 2>/dev/null || \
  aws ecr create-repository --repository-name ${PROJECT_NAME}-lambda --region $AWS_REGION

# Step 3: Login to ECR
echo "Step 3: Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Step 4: Tag and push image
echo "Step 4: Pushing Docker image to ECR..."
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-lambda:latest"
docker tag ${PROJECT_NAME}:latest $ECR_URI
docker push $ECR_URI

echo "Docker image pushed to: $ECR_URI"

# Step 5: Deploy infrastructure with Terraform
echo "Step 5: Deploying infrastructure with Terraform..."
cd ../infrastructure

terraform init
terraform plan -var="aws_region=$AWS_REGION" -var="environment=$ENVIRONMENT"
terraform apply -var="aws_region=$AWS_REGION" -var="environment=$ENVIRONMENT" -auto-approve

# Get outputs
API_ENDPOINT=$(terraform output -raw api_endpoint)
DOCUMENTS_BUCKET=$(terraform output -raw documents_bucket)
RESULTS_BUCKET=$(terraform output -raw results_bucket)

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "API Endpoint: $API_ENDPOINT"
echo "Documents Bucket: $DOCUMENTS_BUCKET"
echo "Results Bucket: $RESULTS_BUCKET"
echo ""
echo "Test the deployment:"
echo "  curl ${API_ENDPOINT}/health"
echo ""
echo "Use the client SDK:"
echo "  from ocr_client import OCRClient"
echo "  client = OCRClient(api_endpoint='${API_ENDPOINT}')"
echo "  result = client.process_document('invoice.pdf')"
echo ""
