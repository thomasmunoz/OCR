# Dockerfile for AWS Lambda - Dual-AI OCR Extraction System
# Uses Lambda container image for large dependencies (ML models)

FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies
RUN yum install -y \
    gcc \
    gcc-c++ \
    make \
    wget \
    && yum clean all

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements
COPY requirements_lambda.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_lambda.txt

# Copy dual-AI OCR system
COPY model_config.py .
COPY format_converters.py .
COPY ai_organizer.py .
COPY qwen_extract_ai.py .

# Copy Lambda handler
COPY handler.py .

# Set Lambda handler
CMD ["handler.lambda_handler"]
