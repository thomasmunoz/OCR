#!/usr/bin/env python3
"""
Maximum Information Extraction from Documents
Extracts ALL possible information: text, tables, company info, metadata, layout, etc.

Usage:
    python3 qwen_extract_max.py <file>                  # Extract everything
    python3 qwen_extract_max.py <file> --json           # JSON output
    python3 qwen_extract_max.py <file> --categories     # Show what can be extracted
"""
import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

class MaximumExtractor:
    """Extract maximum possible information from documents"""

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

    def extract_maximum(self, file_path: str) -> Dict[str, Any]:
        """Extract MAXIMUM information from document"""
        self._load_model()

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"Extracting maximum information from: {file_path.name}...", file=sys.stderr)

        # Initialize result structure
        result = {
            "file": {
                "name": file_path.name,
                "path": str(file_path.absolute()),
                "size": file_path.stat().st_size,
                "extension": file_path.suffix,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            },
            "content": {},
            "structure": {},
            "entities": {},
            "metadata": {},
            "analysis": {}
        }

        # Extract based on file type
        ext = file_path.suffix.lower()

        if ext == '.pdf':
            self._extract_from_pdf(file_path, result)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
            self._extract_from_image(file_path, result)
        elif ext in ['.html', '.xhtml', '.htm']:
            self._extract_from_html(file_path, result)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # Extract entities from text
        if 'text' in result['content']:
            self._extract_entities(result['content']['text'], result)

        # Analyze content
        self._analyze_content(result)

        return result

    def _extract_from_image(self, image_path: Path, result: Dict):
        """Extract from image file"""
        from PIL import Image

        image = Image.open(image_path).convert('RGB')

        # Image metadata
        result['file']['image'] = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode
        }

        # Extract text
        text = self._ocr_image(image)
        result['content']['text'] = text
        result['content']['raw_text'] = text

        # Extract tables
        tables = self._extract_tables_from_text(text)
        if tables:
            result['structure']['tables'] = tables

        # Detect layout
        result['structure']['layout'] = self._detect_layout(text)

    def _extract_from_pdf(self, pdf_path: Path, result: Dict):
        """Extract from PDF file"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("Error: PyMuPDF not installed. Run: pip install PyMuPDF", file=sys.stderr)
            sys.exit(1)

        doc = fitz.open(pdf_path)

        # PDF metadata
        result['file']['pdf'] = {
            "pages": len(doc),
            "encrypted": doc.is_encrypted,
            "metadata": {
                "title": doc.metadata.get('title', ''),
                "author": doc.metadata.get('author', ''),
                "subject": doc.metadata.get('subject', ''),
                "keywords": doc.metadata.get('keywords', ''),
                "creator": doc.metadata.get('creator', ''),
                "producer": doc.metadata.get('producer', ''),
                "creationDate": doc.metadata.get('creationDate', ''),
                "modDate": doc.metadata.get('modDate', '')
            }
        }

        # Extract from each page
        all_text = []
        pages_info = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            page_info = {
                "page_number": page_num + 1,
                "width": page.rect.width,
                "height": page.rect.height,
                "rotation": page.rotation
            }

            # Extract text via OCR
            pix = page.get_pixmap()
            from PIL import Image
            import io
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))

            page_text = self._ocr_image(image)
            all_text.append(page_text)
            page_info['text'] = page_text
            page_info['text_length'] = len(page_text)

            # Extract native text (if available)
            native_text = page.get_text()
            if native_text.strip():
                page_info['native_text'] = native_text
                page_info['has_selectable_text'] = True
            else:
                page_info['has_selectable_text'] = False

            # Extract links
            links = page.get_links()
            if links:
                page_info['links'] = [
                    {
                        'type': link.get('kind', 0),
                        'uri': link.get('uri', ''),
                        'page': link.get('page', -1)
                    }
                    for link in links
                ]

            # Extract images
            image_list = page.get_images()
            if image_list:
                page_info['images_count'] = len(image_list)
                page_info['images'] = [
                    {
                        'xref': img[0],
                        'width': img[2],
                        'height': img[3]
                    }
                    for img in image_list[:5]  # Limit to first 5
                ]

            pages_info.append(page_info)

        result['content']['text'] = "\n\n".join(all_text)
        result['content']['raw_text'] = "\n\n".join(all_text)
        result['structure']['pages'] = pages_info

        # Extract tables from combined text
        tables = self._extract_tables_from_text(result['content']['text'])
        if tables:
            result['structure']['tables'] = tables

        doc.close()

    def _extract_from_html(self, html_path: Path, result: Dict):
        """Extract from HTML file"""
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Extract text
        import re
        text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        clean_text = re.sub(r'<[^>]+>', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text)

        result['content']['text'] = clean_text.strip()
        result['content']['raw_html'] = html_content
        result['content']['html_length'] = len(html_content)

        # Extract HTML structure
        result['structure']['html'] = {
            'title': self._extract_html_tag(html_content, 'title'),
            'meta_description': self._extract_meta(html_content, 'description'),
            'meta_keywords': self._extract_meta(html_content, 'keywords'),
            'headings': self._extract_headings(html_content),
            'links': self._extract_links(html_content),
            'images': self._extract_html_images(html_content)
        }

        # Extract tables
        tables = self._extract_html_tables(html_content)
        if tables:
            result['structure']['tables'] = tables

    def _ocr_image(self, image) -> str:
        """OCR extraction from PIL Image"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all text from this image, including tables, headers, footers, and any other visible text. Preserve structure."}
            ]
        }]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        output_ids = self.model.generate(**inputs, max_new_tokens=4096)
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        return output_text.strip()

    def _extract_entities(self, text: str, result: Dict):
        """Extract all possible entities from text"""
        result['entities'] = {
            'company': self._extract_company_info(text),
            'people': self._extract_people(text),
            'dates': self._extract_dates(text),
            'amounts': self._extract_amounts(text),
            'emails': self._extract_emails(text),
            'phones': self._extract_phones(text),
            'urls': self._extract_urls(text),
            'addresses': self._extract_addresses(text),
            'numbers': self._extract_numbers(text),
            'identifiers': self._extract_identifiers(text)
        }

    def _extract_company_info(self, text: str) -> Dict[str, Any]:
        """Extract company information"""
        return {
            "name": self._find_pattern(text, [
                r"(?:Raison sociale|Company|Société|Entreprise)\s*:?\s*([A-Z][A-Za-z\s&'-]+(?:S\.?A\.?S?|SARL|SAS|SA|Ltd|Inc|Corp)?)",
                r"([A-Z][A-Za-z\s&'-]+(?:S\.?A\.?S?|SARL|SAS|SA))"
            ]),
            "registration_number": self._find_pattern(text, [
                r"(?:SIREN|SIRET|RCS)\s*:?\s*([\d\s]{9,14})",
                r"\b(\d{3}\s?\d{3}\s?\d{3}(?:\s?\d{5})?)\b"
            ]),
            "legal_form": self._find_legal_form(text),
            "capital": self._find_pattern(text, [
                r"(?:Capital|capital)\s*:?\s*([\d\s]+(?:€|EUR|euros?))"
            ]),
            "vat_number": self._find_pattern(text, [
                r"(?:TVA|VAT)\s*:?\s*([A-Z]{2}\s?[\d\s]+)"
            ]),
            "trade_register": self._find_pattern(text, [
                r"(?:RCS|Registre du commerce)\s*:?\s*([A-Z][a-z]+\s+[\dA-Z\s]+)"
            ])
        }

    def _extract_people(self, text: str) -> List[Dict[str, str]]:
        """Extract people names and titles"""
        people = []

        # Directors/executives
        patterns = [
            r"(?:Directeur|Président|PDG|CEO|CFO|CTO|Manager)\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+),\s*(?:Directeur|Président|Manager)"
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                people.append({
                    'name': match.group(1).strip(),
                    'context': match.group(0)
                })

        return people[:10]  # Limit to first 10

    def _extract_dates(self, text: str) -> List[str]:
        """Extract all dates"""
        patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b'
        ]

        dates = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))

        return list(set(dates))

    def _extract_amounts(self, text: str) -> List[Dict[str, str]]:
        """Extract monetary amounts"""
        patterns = [
            r'([\d\s]+(?:,\d{2})?\s*(?:€|EUR|euros?))',
            r'([\d\s]+(?:\.\d{2})?\s*(?:\$|USD|dollars?))'
        ]

        amounts = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amounts.append({
                    'amount': match.group(1).strip(),
                    'position': match.start()
                })

        return amounts[:20]  # Limit to first 20

    def _extract_emails(self, text: str) -> List[str]:
        """Extract all email addresses"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return list(set(re.findall(pattern, text)))

    def _extract_phones(self, text: str) -> List[str]:
        """Extract phone numbers"""
        patterns = [
            r'(?:\+33|0)[1-9](?:[\s.-]?\d{2}){4}',
            r'\+\d{1,3}[\s.-]?\d{1,4}[\s.-]?\d{1,4}[\s.-]?\d{1,4}'
        ]

        phones = []
        for pattern in patterns:
            phones.extend(re.findall(pattern, text))

        return list(set(phones))

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs"""
        pattern = r'https?://(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        return list(set(re.findall(pattern, text)))

    def _extract_addresses(self, text: str) -> List[str]:
        """Extract physical addresses"""
        pattern = r'\d+[,\s]+(?:rue|avenue|boulevard|place|chemin|allée)[^,\n]{5,100}'
        return list(set(re.findall(pattern, text, re.IGNORECASE)))

    def _extract_numbers(self, text: str) -> Dict[str, List]:
        """Extract various number types"""
        return {
            'integers': re.findall(r'\b\d{3,}\b', text)[:20],
            'decimals': re.findall(r'\b\d+[.,]\d+\b', text)[:20],
            'percentages': re.findall(r'\b\d+(?:[.,]\d+)?%', text)
        }

    def _extract_identifiers(self, text: str) -> Dict[str, List]:
        """Extract various identifier types"""
        return {
            'invoice_numbers': re.findall(r'(?:Invoice|Facture)\s*#?\s*:?\s*([\dA-Z-]+)', text, re.IGNORECASE),
            'order_numbers': re.findall(r'(?:Order|Commande)\s*#?\s*:?\s*([\dA-Z-]+)', text, re.IGNORECASE),
            'reference_codes': re.findall(r'(?:Ref|Reference|Réf)\s*:?\s*([\dA-Z-]+)', text, re.IGNORECASE)
        }

    def _extract_tables_from_text(self, text: str) -> List[Dict]:
        """Detect and extract tables from text"""
        tables = []

        # Simple heuristic: lines with multiple separators
        lines = text.split('\n')
        table_lines = []

        for i, line in enumerate(lines):
            # Count separators
            separators = line.count('|') + line.count('\t')
            if separators >= 2:
                table_lines.append((i, line))

        if table_lines:
            tables.append({
                'detected': True,
                'line_count': len(table_lines),
                'first_line': table_lines[0][1] if table_lines else '',
                'sample': [line for _, line in table_lines[:5]]
            })

        return tables

    def _extract_html_tables(self, html: str) -> List[Dict]:
        """Extract HTML tables"""
        import re

        table_pattern = r'<table[^>]*>(.*?)</table>'
        tables = []

        for match in re.finditer(table_pattern, html, re.DOTALL | re.IGNORECASE):
            table_html = match.group(0)

            # Count rows
            rows = len(re.findall(r'<tr[^>]*>', table_html, re.IGNORECASE))

            tables.append({
                'rows': rows,
                'html_length': len(table_html),
                'position': match.start()
            })

        return tables

    def _detect_layout(self, text: str) -> Dict[str, Any]:
        """Detect document layout structure"""
        lines = text.split('\n')

        return {
            'total_lines': len(lines),
            'non_empty_lines': len([l for l in lines if l.strip()]),
            'avg_line_length': sum(len(l) for l in lines) / len(lines) if lines else 0,
            'has_headers': self._has_headers(text),
            'has_footer': self._has_footer(text),
            'has_columns': self._has_columns(text),
            'sections': self._detect_sections(text)
        }

    def _analyze_content(self, result: Dict):
        """Analyze content characteristics"""
        text = result['content'].get('text', '')

        result['analysis'] = {
            'statistics': {
                'total_characters': len(text),
                'total_words': len(text.split()),
                'total_lines': len(text.split('\n')),
                'unique_words': len(set(text.lower().split())),
                'avg_word_length': sum(len(w) for w in text.split()) / len(text.split()) if text.split() else 0
            },
            'language': self._detect_language(text),
            'document_type': self._detect_document_type(text),
            'has_tables': len(result['structure'].get('tables', [])) > 0,
            'has_company_info': bool(result['entities'].get('company', {}).get('name')),
            'has_amounts': len(result['entities'].get('amounts', [])) > 0,
            'has_dates': len(result['entities'].get('dates', [])) > 0
        }

    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        french_words = ['le', 'la', 'de', 'et', 'à', 'pour', 'société', 'entreprise']
        english_words = ['the', 'and', 'of', 'to', 'for', 'company', 'business']

        text_lower = text.lower()

        french_count = sum(1 for w in french_words if w in text_lower)
        english_count = sum(1 for w in english_words if w in text_lower)

        if french_count > english_count:
            return 'French'
        elif english_count > french_count:
            return 'English'
        else:
            return 'Unknown'

    def _detect_document_type(self, text: str) -> str:
        """Detect document type"""
        text_lower = text.lower()

        if 'facture' in text_lower or 'invoice' in text_lower:
            return 'Invoice'
        elif 'devis' in text_lower or 'quote' in text_lower:
            return 'Quote'
        elif 'contrat' in text_lower or 'contract' in text_lower:
            return 'Contract'
        elif 'rapport' in text_lower or 'report' in text_lower:
            return 'Report'
        elif 'kbis' in text_lower:
            return 'Company Registration'
        else:
            return 'Unknown'

    # Helper methods
    def _find_pattern(self, text: str, patterns: List[str]) -> str:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _find_legal_form(self, text: str) -> str:
        forms = ["SAS", "SARL", "S.A.S", "S.A.R.L", "SA", "S.A", "EURL", "SCI", "Ltd", "Inc", "Corp"]
        for form in forms:
            if form in text:
                return form
        return ""

    def _extract_html_tag(self, html: str, tag: str) -> str:
        match = re.search(f'<{tag}[^>]*>(.*?)</{tag}>', html, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_meta(self, html: str, name: str) -> str:
        match = re.search(f'<meta[^>]*name=["\']?{name}["\']?[^>]*content=["\']([^"\']*)["\']', html, re.IGNORECASE)
        return match.group(1) if match else ""

    def _extract_headings(self, html: str) -> Dict[str, List[str]]:
        headings = {}
        for level in range(1, 7):
            pattern = f'<h{level}[^>]*>(.*?)</h{level}>'
            matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
            headings[f'h{level}'] = [re.sub(r'<[^>]+>', '', m).strip() for m in matches]
        return headings

    def _extract_links(self, html: str) -> List[str]:
        pattern = r'<a[^>]*href=["\']([^"\']*)["\']'
        return list(set(re.findall(pattern, html, re.IGNORECASE)))[:20]

    def _extract_html_images(self, html: str) -> List[str]:
        pattern = r'<img[^>]*src=["\']([^"\']*)["\']'
        return list(set(re.findall(pattern, html, re.IGNORECASE)))[:20]

    def _has_headers(self, text: str) -> bool:
        first_lines = '\n'.join(text.split('\n')[:3])
        return bool(re.search(r'(?:header|en-tête|titre)', first_lines, re.IGNORECASE))

    def _has_footer(self, text: str) -> bool:
        last_lines = '\n'.join(text.split('\n')[-3:])
        return bool(re.search(r'(?:footer|pied de page|page \d+)', last_lines, re.IGNORECASE))

    def _has_columns(self, text: str) -> bool:
        lines = text.split('\n')
        tab_count = sum(1 for line in lines if '\t' in line or '  ' in line)
        return tab_count > len(lines) * 0.3

    def _detect_sections(self, text: str) -> int:
        section_markers = [
            r'^[A-Z\s]{10,}$',  # All caps headers
            r'^\d+\.\s+[A-Z]',  # Numbered sections
            r'^[IVX]+\.\s+',    # Roman numerals
        ]

        sections = 0
        for line in text.split('\n'):
            for pattern in section_markers:
                if re.match(pattern, line.strip()):
                    sections += 1
                    break

        return sections


def show_categories():
    """Show what information can be extracted"""
    categories = {
        "FILE METADATA": [
            "• File name, path, size, extension",
            "• Last modified date",
            "• File type (PDF, image, HTML)"
        ],
        "PDF-SPECIFIC": [
            "• Number of pages",
            "• PDF metadata (title, author, creation date)",
            "• Encryption status",
            "• Page dimensions and rotation",
            "• Embedded images count",
            "• Hyperlinks",
            "• Selectable text detection"
        ],
        "IMAGE-SPECIFIC": [
            "• Width and height",
            "• Image format and mode",
            "• Color space"
        ],
        "HTML-SPECIFIC": [
            "• Title and meta tags",
            "• Headings hierarchy (H1-H6)",
            "• Links and images",
            "• HTML tables structure"
        ],
        "CONTENT": [
            "• Full text extraction (OCR)",
            "• Raw text with structure preserved",
            "• Tables detection and extraction",
            "• Layout structure (headers, footers, columns)"
        ],
        "COMPANY INFORMATION": [
            "• Company name",
            "• Registration number (SIREN/SIRET)",
            "• Legal form (SAS, SARL, SA, etc.)",
            "• Capital amount",
            "• VAT number",
            "• Trade register (RCS)"
        ],
        "PEOPLE & CONTACTS": [
            "• Names and titles (directors, executives)",
            "• Email addresses",
            "• Phone numbers",
            "• Physical addresses"
        ],
        "DATES & AMOUNTS": [
            "• All dates (multiple formats)",
            "• Monetary amounts (€, $, etc.)",
            "• Percentages",
            "• Numbers (integers, decimals)"
        ],
        "IDENTIFIERS": [
            "• Invoice/facture numbers",
            "• Order/commande numbers",
            "• Reference codes",
            "• URLs and websites"
        ],
        "DOCUMENT ANALYSIS": [
            "• Total characters, words, lines",
            "• Unique words count",
            "• Average word length",
            "• Language detection (French/English)",
            "• Document type (invoice, contract, report, etc.)",
            "• Structural features (tables, sections)"
        ],
        "STRUCTURE": [
            "• Page-by-page breakdown",
            "• Table detection and structure",
            "• Sections and headings",
            "• Layout patterns (columns, headers, footers)"
        ]
    }

    print("\n" + "="*70)
    print("MAXIMUM INFORMATION EXTRACTION - CATEGORIES")
    print("="*70 + "\n")

    for category, items in categories.items():
        print(f"📊 {category}:")
        for item in items:
            print(f"   {item}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 qwen_extract_max.py <file>            # Extract everything")
        print("  python3 qwen_extract_max.py <file> --json     # JSON output")
        print("  python3 qwen_extract_max.py --categories      # Show extraction categories")
        sys.exit(1)

    if '--categories' in sys.argv:
        show_categories()
        sys.exit(0)

    file_path = sys.argv[1]
    use_json = '--json' in sys.argv or file_path.endswith('.json')

    try:
        extractor = MaximumExtractor()
        result = extractor.extract_maximum(file_path)

        if use_json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # Human-readable output
            print_human_readable(result)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_human_readable(result: Dict):
    """Print results in human-readable format"""
    print("\n" + "="*70)
    print("MAXIMUM INFORMATION EXTRACTION RESULTS")
    print("="*70 + "\n")

    # File info
    print("📁 FILE INFORMATION")
    print(f"   Name: {result['file']['name']}")
    print(f"   Size: {result['file']['size']:,} bytes")
    print(f"   Modified: {result['file']['modified']}")

    if 'pdf' in result['file']:
        print(f"   Pages: {result['file']['pdf']['pages']}")
    if 'image' in result['file']:
        print(f"   Dimensions: {result['file']['image']['width']}x{result['file']['image']['height']}")

    print()

    # Analysis
    if 'analysis' in result:
        print("📊 ANALYSIS")
        stats = result['analysis']['statistics']
        print(f"   Characters: {stats['total_characters']:,}")
        print(f"   Words: {stats['total_words']:,}")
        print(f"   Lines: {stats['total_lines']:,}")
        print(f"   Language: {result['analysis']['language']}")
        print(f"   Document Type: {result['analysis']['document_type']}")
        print()

    # Company info
    if result['entities'].get('company', {}).get('name'):
        print("🏢 COMPANY INFORMATION")
        company = result['entities']['company']
        if company.get('name'):
            print(f"   Name: {company['name']}")
        if company.get('registration_number'):
            print(f"   SIREN/SIRET: {company['registration_number']}")
        if company.get('legal_form'):
            print(f"   Legal Form: {company['legal_form']}")
        if company.get('capital'):
            print(f"   Capital: {company['capital']}")
        print()

    # Contacts
    emails = result['entities'].get('emails', [])
    phones = result['entities'].get('phones', [])
    if emails or phones:
        print("📞 CONTACT INFORMATION")
        if emails:
            print(f"   Emails: {', '.join(emails[:3])}")
        if phones:
            print(f"   Phones: {', '.join(phones[:3])}")
        print()

    # Dates
    dates = result['entities'].get('dates', [])
    if dates:
        print(f"📅 DATES FOUND: {len(dates)} dates")
        print(f"   Sample: {', '.join(dates[:5])}")
        print()

    # Amounts
    amounts = result['entities'].get('amounts', [])
    if amounts:
        print(f"💰 MONETARY AMOUNTS: {len(amounts)} amounts")
        print(f"   Sample: {', '.join(a['amount'] for a in amounts[:5])}")
        print()

    # Tables
    tables = result['structure'].get('tables', [])
    if tables:
        print(f"📋 TABLES: {len(tables)} table(s) detected")
        print()

    # People
    people = result['entities'].get('people', [])
    if people:
        print(f"👤 PEOPLE: {len(people)} person(s) found")
        for person in people[:5]:
            print(f"   • {person['name']}")
        print()

    # Text preview
    text = result['content'].get('text', '')
    if text:
        print("📄 TEXT PREVIEW (first 500 characters)")
        print("   " + "-" * 66)
        preview = text[:500].replace('\n', '\n   ')
        print(f"   {preview}")
        if len(text) > 500:
            print(f"   ... (and {len(text) - 500} more characters)")
        print()

    print("="*70)
    print(f"For full JSON output, run with --json flag")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
