#!/usr/bin/env python3
"""
Self-contained Qwen OCR Extraction CLI
No external dependencies on other Python modules needed.

Usage:
    python3 qwen_extract.py <file>                    # Extract text
    python3 qwen_extract.py <file> --json             # Extract as JSON
    python3 qwen_extract.py <file> --company          # Extract company info
    python3 qwen_extract.py <file> --company --xml    # Company info as XML
"""
import sys
import json
import re
from pathlib import Path
from typing import Dict, Any

class QwenOCRExtractor:
    """Self-contained OCR extraction using Qwen2-VL-2B-OCR"""

    def __init__(self, device="cpu"):
        self.device = device
        self.model = None
        self.processor = None

    def _load_model(self):
        """Lazy load the model"""
        if self.model is not None:
            return

        print("Loading Qwen2-VL-2B-OCR model...", file=sys.stderr)
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch

            model_name = "JackChew/Qwen2-VL-2B-OCR"
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=self.device
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            print("Model loaded successfully!", file=sys.stderr)
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            sys.exit(1)

    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract text from document"""
        self._load_model()

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"Processing: {file_path.name}...", file=sys.stderr)

        # Handle different file types
        ext = file_path.suffix.lower()

        if ext == '.pdf':
            return self._process_pdf(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
            return self._process_image(file_path)
        elif ext in ['.html', '.xhtml', '.htm']:
            return self._process_html(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _process_image(self, image_path: Path) -> Dict[str, Any]:
        """Process image file"""
        from PIL import Image

        image = Image.open(image_path).convert('RGB')

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all text from this image."}
            ]
        }]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        output_ids = self.model.generate(**inputs, max_new_tokens=2048)
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        return {
            "text": output_text.strip(),
            "metadata": {
                "file": str(image_path.name),
                "type": "image"
            }
        }

    def _process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process PDF file"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("Error: PyMuPDF not installed. Run: pip install PyMuPDF", file=sys.stderr)
            sys.exit(1)

        doc = fitz.open(pdf_path)
        all_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap()

            # Convert to PIL Image
            from PIL import Image
            import io
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))

            # Extract text using OCR
            result = self._process_image_direct(image)
            all_text.append(result)

        combined_text = "\n\n".join(all_text)

        return {
            "text": combined_text,
            "metadata": {
                "file": str(pdf_path.name),
                "type": "pdf",
                "pages": len(doc)
            }
        }

    def _process_image_direct(self, image) -> str:
        """Process PIL Image object directly"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all text from this image."}
            ]
        }]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        output_ids = self.model.generate(**inputs, max_new_tokens=2048)
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        return output_text.strip()

    def _process_html(self, html_path: Path) -> Dict[str, Any]:
        """Process HTML/XHTML file"""
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Simple HTML text extraction
        import re
        text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return {
            "text": text.strip(),
            "metadata": {
                "file": str(html_path.name),
                "type": "html"
            }
        }

class CompanyExtractor:
    """Extract company information from text"""

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract company information"""
        return {
            "companyName": self._extract_company_name(text),
            "registrationNumber": self._extract_registration(text),
            "legalForm": self._extract_legal_form(text),
            "capital": self._extract_capital(text),
            "address": self._extract_address(text),
            "email": self._extract_email(text),
            "phone": self._extract_phone(text),
            "website": self._extract_website(text)
        }

    def _extract_company_name(self, text: str) -> str:
        patterns = [
            r"(?:Raison sociale|Company|Société|Entreprise)\s*:?\s*([A-Z][A-Za-z\s&'-]+(?:S\.?A\.?S?|SARL|SAS|SA|Ltd|Inc|Corp)?)",
            r"([A-Z][A-Za-z\s&'-]+(?:S\.?A\.?S?|SARL|SAS|SA))"
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_registration(self, text: str) -> str:
        patterns = [
            r"(?:SIREN|SIRET|RCS)\s*:?\s*([\d\s]{9,14})",
            r"\b(\d{3}\s?\d{3}\s?\d{3}(?:\s?\d{5})?)\b"
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_legal_form(self, text: str) -> str:
        forms = ["SAS", "SARL", "S.A.S", "S.A.R.L", "SA", "S.A", "EURL", "SCI"]
        for form in forms:
            if form in text:
                return form
        return ""

    def _extract_capital(self, text: str) -> str:
        pattern = r"(?:Capital|capital)\s*:?\s*([\d\s]+(?:€|EUR|euros?))"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_address(self, text: str) -> str:
        pattern = r"\d+[,\s]+(?:rue|avenue|boulevard|place|chemin)[^,\n]+"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(0).strip() if match else ""

    def _extract_email(self, text: str) -> str:
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(pattern, text)
        return match.group(0) if match else ""

    def _extract_phone(self, text: str) -> str:
        pattern = r'(?:\+33|0)[1-9](?:[\s.-]?\d{2}){4}'
        match = re.search(pattern, text)
        return match.group(0) if match else ""

    def _extract_website(self, text: str) -> str:
        pattern = r'(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}'
        match = re.search(pattern, text)
        return match.group(0) if match else ""

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 qwen_extract.py <file>                    # Extract text")
        print("  python3 qwen_extract.py <file> --json             # Extract as JSON")
        print("  python3 qwen_extract.py <file> --company          # Extract company info")
        print("  python3 qwen_extract.py <file> --company --xml    # Company info as XML")
        sys.exit(1)

    file_path = sys.argv[1]
    use_json = '--json' in sys.argv
    extract_company = '--company' in sys.argv
    use_xml = '--xml' in sys.argv

    try:
        # Extract text
        extractor = QwenOCRExtractor()
        result = extractor.extract(file_path)

        if extract_company:
            # Extract company information
            company_extractor = CompanyExtractor()
            company_info = company_extractor.extract(result['text'])

            if use_xml:
                # Output as XML
                print('<?xml version="1.0" encoding="UTF-8"?>')
                print('<CompanyInfo>')
                for key, value in company_info.items():
                    print(f'  <{key}>{value}</{key}>')
                print('</CompanyInfo>')
            else:
                # Output as JSON
                print(json.dumps(company_info, indent=2, ensure_ascii=False))
        else:
            # Output text
            if use_json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(result['text'])

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
