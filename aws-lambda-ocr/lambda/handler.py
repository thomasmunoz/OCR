"""
AWS Lambda Handler for Dual-AI OCR Extraction System

Serverless OCR processing with Vision AI + Language AI pipeline
Supports 11 file formats, 11 AI models, multi-project deployment
"""

import json
import os
import boto3
import base64
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import logging

# Import dual-AI system (must be in Lambda layer or container)
from model_config import PipelineConfig, ModelRegistry
from qwen_extract_ai import DualAIExtractor
from ai_organizer import CompanyOrganizer

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Environment variables
RESULTS_BUCKET = os.environ.get('RESULTS_BUCKET')
JOBS_TABLE = os.environ.get('JOBS_TABLE')
DEFAULT_OCR_MODEL = os.environ.get('DEFAULT_OCR_MODEL', 'qwen2-vl-2b-ocr')
DEFAULT_ORG_MODEL = os.environ.get('DEFAULT_ORG_MODEL', 'qwen-32b')

class LambdaOCRHandler:
    """Handles OCR extraction requests in AWS Lambda environment"""
    
    def __init__(self):
        self.pipeline_config = None
        self.extractor = None
        
    def initialize_extractor(self, ocr_model: str, org_model: str, api_keys: Optional[Dict] = None):
        """Initialize the dual-AI extractor (lazy loading)"""
        if self.extractor is None:
            logger.info(f"Initializing extractor: OCR={ocr_model}, Org={org_model}")
            self.pipeline_config = PipelineConfig(
                ocr_model=ocr_model,
                organization_model=org_model,
                device='cpu',  # Lambda uses CPU
                api_keys=api_keys or {}
            )
            self.extractor = DualAIExtractor(self.pipeline_config, verbose=True)
            logger.info("Extractor initialized successfully")
        return self.extractor
    
    def download_from_s3(self, bucket: str, key: str, local_path: str):
        """Download file from S3 to local temp storage"""
        logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        return local_path
    
    def upload_to_s3(self, local_path: str, bucket: str, key: str):
        """Upload result to S3"""
        logger.info(f"Uploading {local_path} to s3://{bucket}/{key}")
        s3_client.upload_file(local_path, bucket, key)
        return f"s3://{bucket}/{key}"
    
    def update_job_status(self, job_id: str, status: str, result: Optional[Dict] = None):
        """Update job status in DynamoDB"""
        if not JOBS_TABLE:
            return
        
        table = dynamodb.Table(JOBS_TABLE)
        update_expr = "SET #status = :status, updatedAt = :timestamp"
        expr_values = {
            ':status': status,
            ':timestamp': str(int(time.time()))
        }
        expr_names = {'#status': 'status'}
        
        if result:
            update_expr += ", result = :result"
            expr_values[':result'] = result
        
        table.update_item(
            Key={'jobId': job_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values
        )
    
    def process_document(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing function"""
        try:
            # Extract parameters from event
            body = event.get('body', {})
            if isinstance(body, str):
                body = json.loads(body)
            
            # Document source
            source_type = body.get('sourceType', 's3')  # 's3' or 'base64'
            
            # Models
            ocr_model = body.get('ocrModel', DEFAULT_OCR_MODEL)
            org_model = body.get('orgModel', DEFAULT_ORG_MODEL)
            
            # API keys (if using API models)
            api_keys = body.get('apiKeys', {})
            
            # Extraction options
            extract_company = body.get('extractCompany', False)
            raw_only = body.get('rawOnly', False)
            
            # Job tracking
            job_id = body.get('jobId', f"job-{int(time.time())}")
            project_id = body.get('projectId', 'default')
            
            logger.info(f"Processing job {job_id} for project {project_id}")
            
            # Update job status
            self.update_job_status(job_id, 'processing')
            
            # Get document
            with tempfile.TemporaryDirectory() as tmpdir:
                if source_type == 's3':
                    # Download from S3
                    s3_bucket = body['s3Bucket']
                    s3_key = body['s3Key']
                    file_ext = Path(s3_key).suffix
                    local_path = os.path.join(tmpdir, f"document{file_ext}")
                    self.download_from_s3(s3_bucket, s3_key, local_path)
                
                elif source_type == 'base64':
                    # Decode base64
                    file_content = base64.b64decode(body['fileContent'])
                    file_ext = body.get('fileExtension', '.pdf')
                    local_path = os.path.join(tmpdir, f"document{file_ext}")
                    with open(local_path, 'wb') as f:
                        f.write(file_content)
                
                else:
                    raise ValueError(f"Invalid sourceType: {source_type}")
                
                # Initialize extractor
                extractor = self.initialize_extractor(ocr_model, org_model, api_keys)
                
                # Process document
                logger.info(f"Extracting from {local_path}")
                result = extractor.extract(
                    Path(local_path),
                    organize=not raw_only,
                    schema=None if not extract_company else None,
                    instructions="Extract company information" if extract_company else None
                )
                
                # Save result to S3 if bucket specified
                if RESULTS_BUCKET:
                    result_key = f"{project_id}/results/{job_id}.json"
                    result_path = os.path.join(tmpdir, "result.json")
                    with open(result_path, 'w') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    result_s3_uri = self.upload_to_s3(result_path, RESULTS_BUCKET, result_key)
                    result['resultS3Uri'] = result_s3_uri
                
                # Update job status
                self.update_job_status(job_id, 'completed', {
                    'dataPoints': len(str(result)),
                    's3Uri': result.get('resultS3Uri')
                })
                
                logger.info(f"Job {job_id} completed successfully")
                
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'success': True,
                        'jobId': job_id,
                        'projectId': project_id,
                        'data': result,
                        'pipeline': {
                            'ocrModel': ocr_model,
                            'orgModel': org_model,
                            'stage1Time': result.get('pipeline', {}).get('stage1_time'),
                            'stage2Time': result.get('pipeline', {}).get('stage2_time')
                        }
                    }, ensure_ascii=False)
                }
        
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            if 'job_id' in locals():
                self.update_job_status(job_id, 'failed', {'error': str(e)})
            
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'success': False,
                    'error': str(e)
                })
            }

# Global handler instance (reused across invocations)
handler_instance = LambdaOCRHandler()

def lambda_handler(event, context):
    """
    AWS Lambda entry point
    
    Event structure:
    {
        "body": {
            "sourceType": "s3" | "base64",
            "s3Bucket": "my-bucket",      # if sourceType=s3
            "s3Key": "documents/file.pdf", # if sourceType=s3
            "fileContent": "base64...",    # if sourceType=base64
            "fileExtension": ".pdf",       # if sourceType=base64
            "ocrModel": "qwen2-vl-2b-ocr",
            "orgModel": "qwen-32b",
            "apiKeys": {
                "openai": "sk-...",
                "anthropic": "sk-ant-..."
            },
            "extractCompany": false,
            "rawOnly": false,
            "jobId": "job-123",
            "projectId": "my-project"
        }
    }
    """
    logger.info(f"Lambda invoked: {json.dumps(event)}")
    return handler_instance.process_document(event)

# Health check handler
def health_check_handler(event, context):
    """Simple health check endpoint"""
    return {
        'statusCode': 200,
        'body': json.dumps({
            'status': 'healthy',
            'version': '1.0.0',
            'models': {
                'ocr': list(ModelRegistry.OCR_MODELS.keys()),
                'organization': list(ModelRegistry.ORGANIZATION_MODELS.keys())
            }
        })
    }

import time
