#!/usr/bin/env python3
"""
Dual-AI OCR Extraction System

Two-Stage AI Pipeline:
- Stage 1: OCR extraction using vision models (Qwen2-VL, GPT-4V, Claude Vision)
- Stage 2: AI organization using language models (Qwen, GPT-4, Claude)

Supports multiple file formats: PDF, images, DOCX, XLSX, HTML, TXT, and more.

Usage:
    # Use default models
    python3 qwen_extract_ai.py document.pdf

    # Choose specific models
    python3 qwen_extract_ai.py document.pdf --ocr-model qwen2-vl-2b-ocr --org-model qwen-32b

    # Use API models
    python3 qwen_extract_ai.py document.pdf --org-model gpt-4 --api-key-openai sk-xxx

    # Show available models
    python3 qwen_extract_ai.py --list-models

    # Extract company information
    python3 qwen_extract_ai.py invoice.pdf --company
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
from datetime import datetime

# Import our modules
from model_config import (
    ModelRegistry,
    PipelineConfig,
    ModelType,
    ModelProvider,
    get_default_config
)
from format_converters import ConverterRegistry, convert_file
from ai_organizer import create_organizer, AIOrganizer


class Stage1OCRExtractor:
    """
    Stage 1: OCR Extraction using Vision Models

    Extracts raw text from images and PDFs using vision models.
    """

    def __init__(self, model_config, device="auto"):
        """
        Initialize OCR extractor

        Args:
            model_config: OCR model configuration
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        self.model_config = model_config
        self.device = self._resolve_device(device)
        self.model = None
        self.processor = None

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device"""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device

    def extract(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from document

        Args:
            file_path: Path to document

        Returns:
            Dict with 'text' and 'metadata'
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()

        print(f"[Stage 1] Processing {file_path.name}...", file=sys.stderr)

        # Check if file needs conversion first
        if ext in ['.txt', '.docx', '.doc', '.xlsx', '.xls', '.csv', '.rtf']:
            # These formats don't need OCR, just text extraction
            print(f"[Stage 1] Converting {ext} to text...", file=sys.stderr)
            return convert_file(file_path)

        # Handle OCR-required formats (PDF, images, HTML)
        if ext == '.pdf':
            return self._process_pdf(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp']:
            return self._process_image(file_path)
        elif ext in ['.html', '.htm', '.xhtml']:
            # HTML can be converted directly
            from format_converters import HTMLConverter
            text = HTMLConverter.convert(file_path)
            return {
                "text": text,
                "metadata": {
                    "file": file_path.name,
                    "type": "html",
                    "stage": "conversion"
                }
            }
        else:
            raise ValueError(
                f"Unsupported file type: {ext}\n"
                f"Supported: .pdf, .png, .jpg, .jpeg, .tiff, .docx, .doc, .xlsx, .xls, .txt, .html, .csv"
            )

    def _load_model(self):
        """Lazy load OCR model"""
        if self.model is not None:
            return

        print(f"[Stage 1] Loading {self.model_config.name}...", file=sys.stderr)

        if self.model_config.type == ModelType.API:
            # API models don't need loading
            print(f"[Stage 1] Using API model: {self.model_config.name}", file=sys.stderr)
            return

        # Load local model
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers torch"
            )

        model_id = self.model_config.parameters["model_id"]
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        print(f"[Stage 1] Model loaded successfully!", file=sys.stderr)

    def _process_image(self, image_path: Path) -> Dict[str, Any]:
        """Process image file with OCR"""
        self._load_model()

        from PIL import Image
        image = Image.open(image_path).convert('RGB')

        text = self._extract_text_from_image(image)

        return {
            "text": text,
            "metadata": {
                "file": image_path.name,
                "type": "image",
                "format": image_path.suffix.lower(),
                "stage": "ocr"
            }
        }

    def _process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process PDF file with OCR"""
        self._load_model()

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF not installed. Install with: pip install PyMuPDF"
            )

        from PIL import Image
        import io

        doc = fitz.open(pdf_path)
        all_text = []

        print(f"[Stage 1] Processing {len(doc)} pages...", file=sys.stderr)

        for page_num in range(len(doc)):
            print(f"[Stage 1] Page {page_num + 1}/{len(doc)}...", file=sys.stderr)
            page = doc[page_num]

            # Try text extraction first
            text = page.get_text()
            if text.strip():
                all_text.append(text)
            else:
                # Fallback to OCR
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                ocr_text = self._extract_text_from_image(image)
                all_text.append(ocr_text)

        combined_text = "\n\n".join(all_text)

        return {
            "text": combined_text,
            "metadata": {
                "file": pdf_path.name,
                "type": "pdf",
                "pages": len(doc),
                "stage": "ocr"
            }
        }

    def _extract_text_from_image(self, image) -> str:
        """Extract text from PIL Image using OCR model"""
        if self.model_config.type == ModelType.API:
            # Use API for OCR
            raise NotImplementedError("API-based OCR not yet implemented")

        # Use local model
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all text from this image. Preserve the exact layout and formatting."}
            ]
        }]

        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.model_config.parameters.get("max_tokens", 2048)
        )

        output_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        return output_text.strip()


class DualAIExtractor:
    """
    Dual-AI Extraction System

    Combines Stage 1 (OCR) and Stage 2 (Organization) for complete document processing.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        verbose: bool = True
    ):
        """
        Initialize dual-AI extractor

        Args:
            pipeline_config: Pipeline configuration
            verbose: Enable verbose output
        """
        self.config = pipeline_config
        self.verbose = verbose

        # Initialize Stage 1 (OCR)
        self.stage1 = Stage1OCRExtractor(
            self.config.ocr_config,
            self.config.device
        )

        # Stage 2 will be initialized on demand
        self.stage2 = None

    def extract(
        self,
        file_path: Path,
        organize: bool = True,
        schema: Optional[Dict[str, Any]] = None,
        instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract and organize document

        Args:
            file_path: Path to document
            organize: Whether to run Stage 2 (organization)
            schema: Optional JSON schema for organization
            instructions: Optional custom instructions

        Returns:
            Extraction results with metadata
        """
        start_time = datetime.now()

        # Stage 1: OCR Extraction
        stage1_result = self.stage1.extract(file_path)
        raw_text = stage1_result["text"]

        if not organize:
            # Return raw OCR result
            return {
                "raw_text": raw_text,
                "metadata": stage1_result["metadata"],
                "pipeline": {
                    "stage1_model": self.config.ocr_model_name,
                    "stage2_model": None,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            }

        # Stage 2: AI Organization
        if self.stage2 is None:
            self._initialize_stage2()

        print(f"[Stage 2] Organizing with {self.config.organization_model_name}...", file=sys.stderr)

        organized_data = self.stage2.organize(raw_text, schema, instructions)

        # Combine results
        return {
            "data": organized_data,
            "raw_text": raw_text,
            "metadata": stage1_result["metadata"],
            "pipeline": {
                "stage1_model": self.config.ocr_model_name,
                "stage2_model": self.config.organization_model_name,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
        }

    def extract_company(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract company information from document

        Specialized extraction for invoices, KBIS, company documents.
        """
        start_time = datetime.now()

        # Stage 1: OCR
        stage1_result = self.stage1.extract(file_path)
        raw_text = stage1_result["text"]

        # Stage 2: Company organization
        if self.stage2 is None:
            self._initialize_stage2(organizer_type="company")

        print(f"[Stage 2] Extracting company information...", file=sys.stderr)

        if hasattr(self.stage2, 'organize_company_document'):
            organized_data = self.stage2.organize_company_document(raw_text)
        else:
            # Fallback to general organization with company schema
            schema = self._get_company_schema()
            instructions = "Extract all company and document information."
            organized_data = self.stage2.organize(raw_text, schema, instructions)

        return {
            "company": organized_data,
            "raw_text": raw_text,
            "metadata": stage1_result["metadata"],
            "pipeline": {
                "stage1_model": self.config.ocr_model_name,
                "stage2_model": self.config.organization_model_name,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
        }

    def _initialize_stage2(self, organizer_type: str = "general"):
        """Initialize Stage 2 organizer"""
        api_key = None
        if self.config.org_config.type == ModelType.API:
            provider = self.config.org_config.provider.value
            api_key = self.config.api_keys.get(provider)

        self.stage2 = create_organizer(
            self.config.org_config,
            api_key,
            organizer_type
        )

    def _get_company_schema(self) -> Dict[str, Any]:
        """Get company extraction schema"""
        return {
            "companyName": "string",
            "legalForm": "string",
            "registrationNumber": "string",
            "vatNumber": "string",
            "capital": "string",
            "address": {
                "street": "string",
                "city": "string",
                "postalCode": "string",
                "country": "string"
            },
            "contact": {
                "email": "string",
                "phone": "string",
                "website": "string"
            }
        }


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Dual-AI OCR Extraction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default models
  python3 qwen_extract_ai.py document.pdf

  # Choose specific models
  python3 qwen_extract_ai.py document.pdf --ocr-model qwen2-vl-2b-ocr --org-model qwen-32b

  # Use API models
  python3 qwen_extract_ai.py document.pdf --org-model gpt-4 --api-key-openai sk-xxx

  # Extract company information
  python3 qwen_extract_ai.py invoice.pdf --company

  # Raw OCR only (no organization)
  python3 qwen_extract_ai.py document.pdf --raw

  # Show available models
  python3 qwen_extract_ai.py --list-models
        """
    )

    parser.add_argument(
        'file',
        nargs='?',
        help='Document to process'
    )

    parser.add_argument(
        '--ocr-model',
        default='qwen2-vl-2b-ocr',
        help='OCR model to use (Stage 1)'
    )

    parser.add_argument(
        '--org-model',
        default='qwen-32b',
        help='Organization model to use (Stage 2)'
    )

    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use for local models'
    )

    parser.add_argument(
        '--api-key-openai',
        help='OpenAI API key'
    )

    parser.add_argument(
        '--api-key-anthropic',
        help='Anthropic API key'
    )

    parser.add_argument(
        '--company',
        action='store_true',
        help='Extract company information'
    )

    parser.add_argument(
        '--raw',
        action='store_true',
        help='Output raw OCR text only (skip Stage 2)'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        default=True,
        help='Output as JSON (default)'
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process multiple files'
    )

    parser.add_argument(
        '--output',
        '-o',
        help='Output file (default: stdout)'
    )

    args = parser.parse_args()

    # List models
    if args.list_models:
        ModelRegistry.print_available_models()
        return

    # Validate input file
    if not args.file:
        parser.print_help()
        sys.exit(1)

    # Build API keys dict
    api_keys = {}
    if args.api_key_openai:
        api_keys['openai'] = args.api_key_openai
    elif os.getenv('OPENAI_API_KEY'):
        api_keys['openai'] = os.getenv('OPENAI_API_KEY')

    if args.api_key_anthropic:
        api_keys['anthropic'] = args.api_key_anthropic
    elif os.getenv('ANTHROPIC_API_KEY'):
        api_keys['anthropic'] = os.getenv('ANTHROPIC_API_KEY')

    try:
        # Create pipeline configuration
        config = PipelineConfig(
            ocr_model=args.ocr_model,
            organization_model=args.org_model,
            device=args.device,
            api_keys=api_keys
        )

        # Create extractor
        extractor = DualAIExtractor(config)

        # Process file
        file_path = Path(args.file)

        if args.company:
            result = extractor.extract_company(file_path)
        else:
            result = extractor.extract(
                file_path,
                organize=not args.raw
            )

        # Output results
        output_text = json.dumps(result, indent=2, ensure_ascii=False)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_text)
            print(f"Results saved to: {args.output}", file=sys.stderr)
        else:
            print(output_text)

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
