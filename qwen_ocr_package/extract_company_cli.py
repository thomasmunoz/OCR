#!/usr/bin/env python3
"""
Company information extraction CLI
Usage: python3 extract_company_cli.py <file> [--format json|json-ai|xml]
"""
import sys
import json
from pathlib import Path

def extract_company(file_path, output_format='json'):
    """Extract company information from document"""
    try:
        # Try importing from parent directory
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from qwen_ocr import QwenOCRReader
        from company_extractor import CompanyInfoExtractor, OutputFormatter

        # Extract text
        reader = QwenOCRReader()
        result = reader.extract(file_path)
        text = result.get('text', '')

        # Extract company info
        extractor = CompanyInfoExtractor()
        company_info = extractor.extract_company_info(text)

        # Format output
        formatter = OutputFormatter()
        if output_format == 'json':
            output = formatter.to_json(company_info, pretty=True)
        elif output_format == 'json-ai':
            output = formatter.to_json_ai_optimized(company_info)
        elif output_format == 'xml':
            output = formatter.to_xml(company_info, pretty=True)
        else:
            output = formatter.to_json(company_info, pretty=True)

        print(output)

    except ImportError as e:
        print(f"Error: Required modules not found - {e}")
        print("Please ensure qwen_ocr.py and company_extractor.py are in the parent directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_company_cli.py <file> [--format json|json-ai|xml]")
        print("")
        print("Examples:")
        print("  python3 extract_company_cli.py invoice.pdf")
        print("  python3 extract_company_cli.py document.pdf --format json")
        print("  python3 extract_company_cli.py kbis.pdf --format json-ai")
        print("  python3 extract_company_cli.py report.pdf --format xml")
        sys.exit(1)

    file_path = sys.argv[1]
    output_format = 'json'

    if '--format' in sys.argv:
        idx = sys.argv.index('--format')
        if idx + 1 < len(sys.argv):
            output_format = sys.argv[idx + 1]

    extract_company(file_path, output_format)

if __name__ == '__main__':
    main()
