#!/usr/bin/env python3
"""
Simple OCR extraction CLI
Usage: python3 extract_cli.py <file> [--format json|text]
"""
import sys
import json
from pathlib import Path

def extract_text(file_path, output_format='text'):
    """Extract text from document using Qwen OCR"""
    try:
        # Try importing from parent directory first
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from qwen_ocr import QwenOCRReader

        reader = QwenOCRReader()
        result = reader.extract(file_path)

        if output_format == 'json':
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result['text'])

    except ImportError:
        print("Error: qwen_ocr module not found")
        print("Please ensure qwen_ocr.py is in the parent directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_cli.py <file> [--format json|text]")
        print("")
        print("Examples:")
        print("  python3 extract_cli.py document.pdf")
        print("  python3 extract_cli.py invoice.pdf --format json")
        sys.exit(1)

    file_path = sys.argv[1]
    output_format = 'text'

    if '--format' in sys.argv:
        idx = sys.argv.index('--format')
        if idx + 1 < len(sys.argv):
            output_format = sys.argv[idx + 1]

    extract_text(file_path, output_format)

if __name__ == '__main__':
    main()
