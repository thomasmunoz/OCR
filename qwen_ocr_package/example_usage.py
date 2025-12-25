#!/usr/bin/env python3
"""
Quick Start Examples for Dual-AI OCR System

This file demonstrates common use cases and best practices.
"""

from pathlib import Path
import json
from typing import Dict, Any

# Import the dual-AI system
from model_config import PipelineConfig, ModelRegistry
from qwen_extract_ai import DualAIExtractor


def example_1_basic_extraction():
    """Example 1: Basic document extraction"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Document Extraction")
    print("="*80)

    # Configure with default models
    config = PipelineConfig(
        ocr_model="qwen2-vl-2b-ocr",
        organization_model="qwen-32b",
        device="auto"
    )

    # Create extractor
    extractor = DualAIExtractor(config)

    # Note: Replace with actual file path
    # result = extractor.extract(Path("your_document.pdf"))
    # print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\nCode:")
    print("""
    config = PipelineConfig(
        ocr_model="qwen2-vl-2b-ocr",
        organization_model="qwen-32b"
    )
    extractor = DualAIExtractor(config)
    result = extractor.extract(Path("document.pdf"))
    """)


def example_2_company_extraction():
    """Example 2: Extract company information"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Company Information Extraction")
    print("="*80)

    config = PipelineConfig()
    extractor = DualAIExtractor(config)

    # Note: Replace with actual invoice file
    # result = extractor.extract_company(Path("invoice.pdf"))
    # company = result["company"]
    # print(f"Company: {company.get('companyName')}")
    # print(f"SIREN: {company.get('registrationNumber')}")

    print("\nCode:")
    print("""
    config = PipelineConfig()
    extractor = DualAIExtractor(config)
    result = extractor.extract_company(Path("invoice.pdf"))

    # Access company data
    company = result["company"]
    print(company["companyName"])
    print(company["registrationNumber"])
    print(company["address"])
    """)


def example_3_custom_schema():
    """Example 3: Extract with custom schema"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Schema Extraction")
    print("="*80)

    # Define custom schema
    invoice_schema = {
        "invoice_number": "string",
        "date": "string",
        "due_date": "string",
        "supplier": {
            "name": "string",
            "address": "string",
            "vat": "string",
            "email": "string"
        },
        "client": {
            "name": "string",
            "address": "string"
        },
        "items": [
            {
                "description": "string",
                "quantity": "number",
                "unit_price": "number",
                "total": "number"
            }
        ],
        "totals": {
            "subtotal": "number",
            "tax_rate": "number",
            "tax_amount": "number",
            "total": "number",
            "currency": "string"
        }
    }

    print("\nSchema:")
    print(json.dumps(invoice_schema, indent=2))

    print("\nCode:")
    print("""
    result = extractor.extract(
        Path("invoice.pdf"),
        schema=invoice_schema,
        instructions="Extract complete invoice information"
    )
    """)


def example_4_api_models():
    """Example 4: Using API models"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Using API Models (GPT-4, Claude)")
    print("="*80)

    print("\nUsing GPT-4:")
    print("""
    config = PipelineConfig(
        ocr_model="qwen2-vl-2b-ocr",
        organization_model="gpt-4",
        api_keys={"openai": "sk-..."}
    )
    extractor = DualAIExtractor(config)
    result = extractor.extract(Path("document.pdf"))
    """)

    print("\nUsing Claude Sonnet:")
    print("""
    config = PipelineConfig(
        ocr_model="qwen2-vl-2b-ocr",
        organization_model="claude-sonnet",
        api_keys={"anthropic": "sk-ant-..."}
    )
    extractor = DualAIExtractor(config)
    result = extractor.extract(Path("document.pdf"))
    """)


def example_5_batch_processing():
    """Example 5: Batch process multiple files"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Batch Processing")
    print("="*80)

    print("\nCode:")
    print("""
    from pathlib import Path
    import json

    # Configure once
    config = PipelineConfig()
    extractor = DualAIExtractor(config)

    # Process all PDFs
    input_dir = Path("documents/")
    output_dir = Path("results/")
    output_dir.mkdir(exist_ok=True)

    for pdf_file in input_dir.glob("*.pdf"):
        print(f"Processing {pdf_file.name}...")

        result = extractor.extract_company(pdf_file)

        # Save result
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved to {output_file}")
    """)


def example_6_error_handling():
    """Example 6: Error handling and validation"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Error Handling")
    print("="*80)

    print("\nCode:")
    print("""
    from pathlib import Path
    import json

    def extract_with_validation(file_path: Path) -> Dict[str, Any]:
        try:
            config = PipelineConfig()
            extractor = DualAIExtractor(config)

            result = extractor.extract_company(file_path)

            # Validate required fields
            company = result["company"]
            required = ["companyName", "registrationNumber"]
            missing = [f for f in required if not company.get(f)]

            if missing:
                print(f"Warning: Missing fields: {missing}")

            return result

        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return None

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    # Use it
    result = extract_with_validation(Path("invoice.pdf"))
    if result:
        print("✓ Extraction successful")
    """)


def example_7_model_comparison():
    """Example 7: Compare different models"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Model Comparison")
    print("="*80)

    print("\nCode:")
    print("""
    from pathlib import Path
    import time

    document = Path("test_document.pdf")
    models = ["qwen-7b", "qwen-14b", "qwen-32b"]

    results = {}

    for model_name in models:
        print(f"\\nTesting {model_name}...")

        config = PipelineConfig(
            ocr_model="qwen2-vl-2b-ocr",
            organization_model=model_name
        )

        extractor = DualAIExtractor(config)

        start = time.time()
        result = extractor.extract(document)
        elapsed = time.time() - start

        results[model_name] = {
            "time": elapsed,
            "fields": len(result["data"])
        }

    # Print comparison
    print("\\n" + "="*60)
    print("Model Comparison Results")
    print("="*60)

    for model, stats in results.items():
        print(f"{model:15} - Time: {stats['time']:.2f}s, Fields: {stats['fields']}")
    """)


def example_8_different_formats():
    """Example 8: Process different file formats"""
    print("\n" + "="*80)
    print("EXAMPLE 8: Different File Formats")
    print("="*80)

    print("\nSupported formats:")
    print("""
    - PDF files (.pdf)
    - Images (.png, .jpg, .jpeg, .tiff)
    - Word documents (.docx, .doc)
    - Excel spreadsheets (.xlsx, .xls)
    - HTML files (.html, .htm)
    - Text files (.txt, .csv, .rtf)
    """)

    print("\nCode:")
    print("""
    config = PipelineConfig()
    extractor = DualAIExtractor(config)

    # PDF
    result = extractor.extract(Path("document.pdf"))

    # Image
    result = extractor.extract(Path("scan.png"))

    # Word document
    result = extractor.extract(Path("report.docx"))

    # Excel spreadsheet
    result = extractor.extract(Path("data.xlsx"))

    # HTML file
    result = extractor.extract(Path("webpage.html"))

    # All use the same API!
    """)


def example_9_save_configuration():
    """Example 9: Save and load configuration"""
    print("\n" + "="*80)
    print("EXAMPLE 9: Save & Load Configuration")
    print("="*80)

    print("\nCode:")
    print("""
    from pathlib import Path

    # Create configuration
    config = PipelineConfig(
        ocr_model="qwen2-vl-2b-ocr",
        organization_model="qwen-32b",
        device="cuda"
    )

    # Save configuration
    config.save(Path("my_config.json"))
    print("✓ Configuration saved")

    # Load configuration
    loaded_config = PipelineConfig.load(Path("my_config.json"))
    print("✓ Configuration loaded")

    # Use loaded config
    extractor = DualAIExtractor(loaded_config)
    result = extractor.extract(Path("document.pdf"))
    """)


def example_10_raw_ocr_only():
    """Example 10: Raw OCR extraction only (skip Stage 2)"""
    print("\n" + "="*80)
    print("EXAMPLE 10: Raw OCR Only (Skip Organization)")
    print("="*80)

    print("\nCode:")
    print("""
    config = PipelineConfig()
    extractor = DualAIExtractor(config)

    # Extract with organize=False (Stage 1 only)
    result = extractor.extract(
        Path("document.pdf"),
        organize=False
    )

    # Access raw OCR text
    raw_text = result["raw_text"]
    print(raw_text)

    # Metadata includes processing time
    print(f"Processing time: {result['pipeline']['processing_time']}s")
    """)


def list_available_models():
    """List all available models"""
    print("\n" + "="*80)
    print("AVAILABLE MODELS")
    print("="*80)

    ModelRegistry.print_available_models()


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("DUAL-AI OCR SYSTEM - USAGE EXAMPLES")
    print("="*80)

    examples = [
        ("Basic Extraction", example_1_basic_extraction),
        ("Company Extraction", example_2_company_extraction),
        ("Custom Schema", example_3_custom_schema),
        ("API Models", example_4_api_models),
        ("Batch Processing", example_5_batch_processing),
        ("Error Handling", example_6_error_handling),
        ("Model Comparison", example_7_model_comparison),
        ("Different Formats", example_8_different_formats),
        ("Save Configuration", example_9_save_configuration),
        ("Raw OCR Only", example_10_raw_ocr_only),
    ]

    print("\nAvailable Examples:\n")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i:2}. {name}")

    print("\nRunning all examples...\n")

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {name}: {e}")

    # List available models
    list_available_models()

    print("\n" + "="*80)
    print("For more information, see:")
    print("  - README_AI.md (complete documentation)")
    print("  - USAGE_GUIDE.md (detailed usage guide)")
    print("  - PROJECT_SUMMARY.md (project overview)")
    print("="*80)


if __name__ == "__main__":
    main()
