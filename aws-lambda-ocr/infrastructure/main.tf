# Terraform configuration for Dual-AI OCR Lambda deployment
# Deploys complete serverless OCR system with multi-project support

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ============================================================================
# VARIABLES
# ============================================================================

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name prefix"
  type        = string
  default     = "dual-ai-ocr"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "prod"
}

variable "lambda_memory" {
  description = "Lambda memory in MB"
  type        = number
  default     = 10240  # 10 GB for ML models
}

variable "lambda_timeout" {
  description = "Lambda timeout in seconds"
  type        = number
  default     = 900  # 15 minutes max
}

variable "default_ocr_model" {
  description = "Default OCR model"
  type        = string
  default     = "qwen2-vl-2b-ocr"
}

variable "default_org_model" {
  description = "Default organization model"
  type        = string
  default     = "qwen-32b"
}

# ============================================================================
# S3 BUCKETS
# ============================================================================

# Input documents bucket
resource "aws_s3_bucket" "documents" {
  bucket = "${var.project_name}-documents-${var.environment}"
  
  tags = {
    Name        = "OCR Documents"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Results bucket
resource "aws_s3_bucket" "results" {
  bucket = "${var.project_name}-results-${var.environment}"
  
  tags = {
    Name        = "OCR Results"
    Environment = var.environment
    Project     = var.project_name
  }
}

# CORS configuration for results bucket
resource "aws_s3_bucket_cors_configuration" "results" {
  bucket = aws_s3_bucket.results.id
  
  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

# ============================================================================
# DYNAMODB TABLE (Job Tracking)
# ============================================================================

resource "aws_dynamodb_table" "jobs" {
  name           = "${var.project_name}-jobs-${var.environment}"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "jobId"
  range_key      = "projectId"
  
  attribute {
    name = "jobId"
    type = "S"
  }
  
  attribute {
    name = "projectId"
    type = "S"
  }
  
  attribute {
    name = "status"
    type = "S"
  }
  
  attribute {
    name = "createdAt"
    type = "N"
  }
  
  global_secondary_index {
    name            = "ProjectStatusIndex"
    hash_key        = "projectId"
    range_key       = "status"
    projection_type = "ALL"
  }
  
  global_secondary_index {
    name            = "StatusCreatedAtIndex"
    hash_key        = "status"
    range_key       = "createdAt"
    projection_type = "ALL"
  }
  
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
  
  tags = {
    Name        = "OCR Jobs"
    Environment = var.environment
    Project     = var.project_name
  }
}

# ============================================================================
# SQS QUEUE (Async Processing)
# ============================================================================

resource "aws_sqs_queue" "ocr_jobs" {
  name                       = "${var.project_name}-jobs-${var.environment}"
  delay_seconds              = 0
  max_message_size           = 262144
  message_retention_seconds  = 1209600  # 14 days
  receive_wait_time_seconds  = 10
  visibility_timeout_seconds = var.lambda_timeout + 30
  
  tags = {
    Name        = "OCR Jobs Queue"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Dead letter queue
resource "aws_sqs_queue" "ocr_jobs_dlq" {
  name = "${var.project_name}-jobs-dlq-${var.environment}"
  
  tags = {
    Name        = "OCR Jobs DLQ"
    Environment = var.environment
    Project     = var.project_name
  }
}

# ============================================================================
# ECR REPOSITORY (Docker Images)
# ============================================================================

resource "aws_ecr_repository" "lambda" {
  name = "${var.project_name}-lambda"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  tags = {
    Name        = "OCR Lambda Container"
    Environment = var.environment
    Project     = var.project_name
  }
}

# ============================================================================
# IAM ROLES & POLICIES
# ============================================================================

# Lambda execution role
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
  
  tags = {
    Name        = "OCR Lambda Role"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Lambda policy
resource "aws_iam_role_policy" "lambda_policy" {
  name = "${var.project_name}-lambda-policy-${var.environment}"
  role = aws_iam_role.lambda_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # CloudWatch Logs
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      # S3 Access
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.documents.arn}/*",
          "${aws_s3_bucket.results.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.documents.arn,
          aws_s3_bucket.results.arn
        ]
      },
      # DynamoDB Access
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = [
          aws_dynamodb_table.jobs.arn,
          "${aws_dynamodb_table.jobs.arn}/index/*"
        ]
      },
      # SQS Access
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes",
          "sqs:SendMessage"
        ]
        Resource = aws_sqs_queue.ocr_jobs.arn
      }
    ]
  })
}

# ============================================================================
# LAMBDA FUNCTION
# ============================================================================

resource "aws_lambda_function" "ocr_processor" {
  function_name = "${var.project_name}-processor-${var.environment}"
  role          = aws_iam_role.lambda_role.arn
  
  # Using container image
  package_type = "Image"
  image_uri    = "${aws_ecr_repository.lambda.repository_url}:latest"
  
  memory_size = var.lambda_memory
  timeout     = var.lambda_timeout
  
  environment {
    variables = {
      RESULTS_BUCKET      = aws_s3_bucket.results.id
      JOBS_TABLE          = aws_dynamodb_table.jobs.name
      DEFAULT_OCR_MODEL   = var.default_ocr_model
      DEFAULT_ORG_MODEL   = var.default_org_model
      ENVIRONMENT         = var.environment
    }
  }
  
  ephemeral_storage {
    size = 10240  # 10 GB for temp files
  }
  
  tags = {
    Name        = "OCR Processor"
    Environment = var.environment
    Project     = var.project_name
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${aws_lambda_function.ocr_processor.function_name}"
  retention_in_days = 30
  
  tags = {
    Name        = "OCR Lambda Logs"
    Environment = var.environment
    Project     = var.project_name
  }
}

# ============================================================================
# API GATEWAY
# ============================================================================

resource "aws_apigatewayv2_api" "ocr_api" {
  name          = "${var.project_name}-api-${var.environment}"
  protocol_type = "HTTP"
  
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["POST", "GET", "OPTIONS"]
    allow_headers = ["content-type", "x-api-key"]
    max_age       = 300
  }
  
  tags = {
    Name        = "OCR API"
    Environment = var.environment
    Project     = var.project_name
  }
}

# API Gateway Integration
resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id                 = aws_apigatewayv2_api.ocr_api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.ocr_processor.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

# Routes
resource "aws_apigatewayv2_route" "process_document" {
  api_id    = aws_apigatewayv2_api.ocr_api.id
  route_key = "POST /process"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_route" "health_check" {
  api_id    = aws_apigatewayv2_api.ocr_api.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# Stage
resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.ocr_api.id
  name        = "$default"
  auto_deploy = true
  
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_logs.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }
  
  tags = {
    Name        = "OCR API Stage"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_cloudwatch_log_group" "api_logs" {
  name              = "/aws/apigateway/${var.project_name}-${var.environment}"
  retention_in_days = 30
  
  tags = {
    Name        = "OCR API Logs"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Lambda permission for API Gateway
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ocr_processor.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.ocr_api.execution_arn}/*/*"
}

# ============================================================================
# OUTPUTS
# ============================================================================

output "api_endpoint" {
  description = "API Gateway endpoint URL"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "documents_bucket" {
  description = "S3 bucket for input documents"
  value       = aws_s3_bucket.documents.id
}

output "results_bucket" {
  description = "S3 bucket for results"
  value       = aws_s3_bucket.results.id
}

output "jobs_table" {
  description = "DynamoDB table for job tracking"
  value       = aws_dynamodb_table.jobs.name
}

output "ecr_repository" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.lambda.repository_url
}

output "lambda_function_arn" {
  description = "Lambda function ARN"
  value       = aws_lambda_function.ocr_processor.arn
}
