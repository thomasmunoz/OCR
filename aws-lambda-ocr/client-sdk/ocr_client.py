"""
Client SDK for Dual-AI OCR Lambda Service
Easy integration for any Python project

Usage:
    from ocr_client import OCRClient
    
    client = OCRClient(api_endpoint="https://api.example.com")
    result = client.process_document("invoice.pdf", project_id="my-project")
"""

import requests
import json
import base64
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import boto3
from dataclasses import dataclass, asdict

@dataclass
class OCRConfig:
    """Configuration for OCR processing"""
    ocr_model: str = "qwen2-vl-2b-ocr"
    org_model: str = "qwen-32b"
    extract_company: bool = False
    raw_only: bool = False
    api_keys: Optional[Dict[str, str]] = None

class OCRClient:
    """
    Client for AWS Lambda Dual-AI OCR Service
    
    Supports multiple projects and flexible deployment
    """
    
    def __init__(
        self,
        api_endpoint: str,
        project_id: str = "default",
        api_key: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        region: str = "us-east-1"
    ):
        """
        Initialize OCR client
        
        Args:
            api_endpoint: API Gateway endpoint URL
            project_id: Project identifier for organization
            api_key: Optional API key for authentication
            s3_bucket: Optional S3 bucket for document uploads
            region: AWS region
        """
        self.api_endpoint = api_endpoint.rstrip('/')
        self.project_id = project_id
        self.api_key = api_key
        self.s3_bucket = s3_bucket
        self.region = region
        
        # Initialize S3 client if bucket specified
        self.s3_client = None
        if s3_bucket:
            self.s3_client = boto3.client('s3', region_name=region)
    
    def process_document(
        self,
        file_path: str,
        config: Optional[OCRConfig] = None,
        job_id: Optional[str] = None,
        use_s3: bool = False
    ) -> Dict[str, Any]:
        """
        Process a document through OCR
        
        Args:
            file_path: Path to document (11 formats supported)
            config: OCR configuration
            job_id: Optional job ID for tracking
            use_s3: Upload to S3 first (recommended for large files)
        
        Returns:
            Extraction results with structured data
        """
        if config is None:
            config = OCRConfig()
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate job ID
        if job_id is None:
            job_id = f"job-{self.project_id}-{int(time.time())}"
        
        # Prepare request
        if use_s3 and self.s3_client:
            # Upload to S3 first
            s3_key = f"{self.project_id}/uploads/{job_id}/{file_path.name}"
            self.s3_client.upload_file(str(file_path), self.s3_bucket, s3_key)
            
            request_body = {
                "sourceType": "s3",
                "s3Bucket": self.s3_bucket,
                "s3Key": s3_key,
                "ocrModel": config.ocr_model,
                "orgModel": config.org_model,
                "extractCompany": config.extract_company,
                "rawOnly": config.raw_only,
                "jobId": job_id,
                "projectId": self.project_id
            }
            
            if config.api_keys:
                request_body["apiKeys"] = config.api_keys
        
        else:
            # Send as base64
            with open(file_path, 'rb') as f:
                file_content = base64.b64encode(f.read()).decode('utf-8')
            
            request_body = {
                "sourceType": "base64",
                "fileContent": file_content,
                "fileExtension": file_path.suffix,
                "ocrModel": config.ocr_model,
                "orgModel": config.org_model,
                "extractCompany": config.extract_company,
                "rawOnly": config.raw_only,
                "jobId": job_id,
                "projectId": self.project_id
            }
            
            if config.api_keys:
                request_body["apiKeys"] = config.api_keys
        
        # Make API request
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['x-api-key'] = self.api_key
        
        response = requests.post(
            f"{self.api_endpoint}/process",
            json={"body": request_body},
            headers=headers,
            timeout=900  # 15 minutes
        )
        
        response.raise_for_status()
        result = response.json()
        
        if not result.get('success'):
            raise Exception(f"OCR processing failed: {result.get('error')}")
        
        return result
    
    def process_documents_batch(
        self,
        file_paths: List[str],
        config: Optional[OCRConfig] = None,
        use_s3: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents
        
        Args:
            file_paths: List of document paths
            config: OCR configuration
            use_s3: Upload to S3 first
        
        Returns:
            List of extraction results
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.process_document(file_path, config, use_s3=use_s3)
                results.append(result)
            except Exception as e:
                results.append({
                    'success': False,
                    'file': file_path,
                    'error': str(e)
                })
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        response = requests.get(
            f"{self.api_endpoint}/health",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, List[str]]:
        """List available models"""
        health = self.health_check()
        return health.get('models', {})

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_extract(
    file_path: str,
    api_endpoint: str,
    project_id: str = "default",
    extract_company: bool = False
) -> Dict[str, Any]:
    """
    Quick document extraction (one-liner)
    
    Example:
        result = quick_extract("invoice.pdf", "https://api.example.com")
    """
    client = OCRClient(api_endpoint=api_endpoint, project_id=project_id)
    config = OCRConfig(extract_company=extract_company)
    return client.process_document(file_path, config)

def batch_extract(
    file_paths: List[str],
    api_endpoint: str,
    project_id: str = "default"
) -> List[Dict[str, Any]]:
    """
    Batch document extraction
    
    Example:
        results = batch_extract(["invoice1.pdf", "invoice2.pdf"], "https://api.example.com")
    """
    client = OCRClient(api_endpoint=api_endpoint, project_id=project_id)
    return client.process_documents_batch(file_paths)
