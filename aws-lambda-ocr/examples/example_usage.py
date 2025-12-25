"""
Example Usage of Dual-AI OCR Lambda Service
Demonstrates multi-project integration
"""

from ocr_client import OCRClient, OCRConfig, quick_extract, batch_extract

# ============================================================================
# EXAMPLE 1: Simple Invoice Processing (Project A)
# ============================================================================

def example_invoice_processing():
    """Process invoices for accounting system"""
    
    # Initialize client for Project A
    client = OCRClient(
        api_endpoint="https://abc123.execute-api.us-east-1.amazonaws.com",
        project_id="accounting-system",
        s3_bucket="dual-ai-ocr-documents-prod"
    )
    
    # Process invoice with company extraction
    config = OCRConfig(
        ocr_model="qwen2-vl-2b-ocr",
        org_model="qwen-32b",
        extract_company=True
    )
    
    result = client.process_document(
        "invoices/invoice_001.pdf",
        config=config,
        use_s3=True
    )
    
    # Extract structured data
    company_data = result['data']['data']['company']
    financial_data = result['data']['data']['financial']
    
    print(f"Company: {company_data.get('name')}")
    print(f"Amount: {financial_data.get('total_amount')}")
    print(f"Invoice #: {financial_data.get('invoice_number')}")

# ============================================================================
# EXAMPLE 2: Contract Analysis (Project B)
# ============================================================================

def example_contract_analysis():
    """Analyze legal contracts using GPT-4"""
    
    client = OCRClient(
        api_endpoint="https://abc123.execute-api.us-east-1.amazonaws.com",
        project_id="legal-department"
    )
    
    # Use GPT-4 for complex legal documents
    config = OCRConfig(
        ocr_model="qwen2-vl-2b-ocr",
        org_model="gpt-4",  # Use GPT-4 for organization
        api_keys={
            "openai": "sk-proj-xxx..."  # Your OpenAI API key
        }
    )
    
    result = client.process_document(
        "contracts/service_agreement.pdf",
        config=config
    )
    
    print("Contract analyzed with GPT-4:")
    print(result['data']['data'])

# ============================================================================
# EXAMPLE 3: Batch Document Processing (Project C)
# ============================================================================

def example_batch_processing():
    """Process multiple documents from HR system"""
    
    client = OCRClient(
        api_endpoint="https://abc123.execute-api.us-east-1.amazonaws.com",
        project_id="hr-system",
        s3_bucket="dual-ai-ocr-documents-prod"
    )
    
    # Process batch of documents
    documents = [
        "hr/employee_001.pdf",
        "hr/employee_002.pdf",
        "hr/employee_003.pdf",
        "hr/contract_001.docx",
        "hr/contract_002.docx"
    ]
    
    config = OCRConfig(
        ocr_model="qwen2-vl-2b-ocr",
        org_model="qwen-14b"  # Faster model for batch
    )
    
    results = client.process_documents_batch(
        documents,
        config=config,
        use_s3=True
    )
    
    # Process results
    for i, result in enumerate(results):
        if result.get('success'):
            print(f"✅ Document {i+1}: Processed successfully")
        else:
            print(f"❌ Document {i+1}: {result.get('error')}")

# ============================================================================
# EXAMPLE 4: Quick One-Liner Extraction (Any Project)
# ============================================================================

def example_quick_extraction():
    """Ultra-simple one-liner extraction"""
    
    result = quick_extract(
        "document.pdf",
        api_endpoint="https://abc123.execute-api.us-east-1.amazonaws.com",
        project_id="my-app",
        extract_company=True
    )
    
    print("Quick extraction result:")
    print(result['data']['data'])

# ============================================================================
# EXAMPLE 5: Multi-Format Processing (Project D)
# ============================================================================

def example_multi_format():
    """Process different file formats for document management system"""
    
    client = OCRClient(
        api_endpoint="https://abc123.execute-api.us-east-1.amazonaws.com",
        project_id="document-management"
    )
    
    # Support all 11 formats
    documents = {
        "PDF": "documents/report.pdf",
        "DOCX": "documents/proposal.docx",
        "XLSX": "documents/spreadsheet.xlsx",
        "Image": "scans/scan001.jpg",
        "HTML": "web/page.html"
    }
    
    for doc_type, doc_path in documents.items():
        print(f"\nProcessing {doc_type}...")
        result = client.process_document(doc_path)
        print(f"✅ Extracted {len(result['data']['data'])} fields")

# ============================================================================
# EXAMPLE 6: Using Claude for Organization (Project E)
# ============================================================================

def example_claude_organization():
    """Use Claude Sonnet for cost-effective organization"""
    
    client = OCRClient(
        api_endpoint="https://abc123.execute-api.us-east-1.amazonaws.com",
        project_id="data-extraction"
    )
    
    config = OCRConfig(
        ocr_model="qwen2-vl-2b-ocr",  # Local OCR
        org_model="claude-sonnet",     # API organization
        api_keys={
            "anthropic": "sk-ant-xxx..."
        }
    )
    
    result = client.process_document(
        "complex_document.pdf",
        config=config
    )
    
    print("Organized by Claude Sonnet:")
    print(result['data']['data'])

# ============================================================================
# EXAMPLE 7: Health Check & Model Listing
# ============================================================================

def example_health_check():
    """Check service status and available models"""
    
    client = OCRClient(
        api_endpoint="https://abc123.execute-api.us-east-1.amazonaws.com"
    )
    
    # Health check
    health = client.health_check()
    print(f"Service Status: {health['status']}")
    print(f"Version: {health['version']}")
    
    # List models
    models = client.list_models()
    print(f"\nOCR Models: {models['ocr']}")
    print(f"Organization Models: {models['organization']}")

# ============================================================================
# EXAMPLE 8: Integration with Django/Flask (Project F)
# ============================================================================

def example_web_integration():
    """Example Flask endpoint using OCR service"""
    
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    ocr_client = OCRClient(
        api_endpoint="https://abc123.execute-api.us-east-1.amazonaws.com",
        project_id="web-app",
        s3_bucket="dual-ai-ocr-documents-prod"
    )
    
    @app.route('/api/process', methods=['POST'])
    def process_upload():
        """Process uploaded document"""
        
        file = request.files['document']
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)
        
        try:
            result = ocr_client.process_document(
                temp_path,
                config=OCRConfig(extract_company=True),
                use_s3=True
            )
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

# ============================================================================
# RUN EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("Dual-AI OCR Lambda Service - Usage Examples\n")
    
    # Run examples (uncomment to test)
    # example_invoice_processing()
    # example_contract_analysis()
    # example_batch_processing()
    # example_quick_extraction()
    # example_multi_format()
    # example_claude_organization()
    # example_health_check()
    
    print("\n✅ See examples above for integration patterns")
