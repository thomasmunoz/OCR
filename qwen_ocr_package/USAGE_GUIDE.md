# 📘 Dual-AI OCR System - Complete Usage Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Understanding the Two-Stage Pipeline](#understanding-the-two-stage-pipeline)
4. [Model Selection Guide](#model-selection-guide)
5. [File Format Support](#file-format-support)
6. [Command Line Usage](#command-line-usage)
7. [Python API Usage](#python-api-usage)
8. [Advanced Examples](#advanced-examples)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

---

## Installation

### Minimal Installation (Local Models Only)

```bash
pip install transformers torch pillow PyMuPDF python-docx openpyxl
```

### Full Installation (with API Support)

```bash
pip install -r requirements_ai.txt
```

### Platform-Specific Notes

**macOS (Apple Silicon)**:
```bash
# MPS (Metal Performance Shaders) support is automatic
pip install transformers torch pillow PyMuPDF python-docx openpyxl
```

**Linux (NVIDIA GPU)**:
```bash
# For CUDA support
pip install transformers torch --index-url https://download.pytorch.org/whl/cu118
pip install pillow PyMuPDF python-docx openpyxl
```

**Windows**:
```bash
# Standard installation
pip install transformers torch pillow PyMuPDF python-docx openpyxl
```

---

## Quick Start

### 1. Extract Text from Any Document

```bash
# PDF document
python3 qwen_extract_ai.py document.pdf

# Image (PNG, JPG, etc.)
python3 qwen_extract_ai.py scan.png

# Word document
python3 qwen_extract_ai.py report.docx

# Excel spreadsheet
python3 qwen_extract_ai.py data.xlsx
```

### 2. Extract Company Information

```bash
python3 qwen_extract_ai.py invoice.pdf --company
python3 qwen_extract_ai.py kbis.pdf --company
```

### 3. Save Results to File

```bash
python3 qwen_extract_ai.py document.pdf -o results.json
```

### 4. Use Different Models

```bash
# Lighter, faster model
python3 qwen_extract_ai.py document.pdf --org-model qwen-7b

# More powerful model
python3 qwen_extract_ai.py document.pdf --org-model qwen-32b
```

---

## Understanding the Two-Stage Pipeline

### Stage 1: OCR Extraction (Vision)

**What it does**:
- Reads scanned documents, PDFs, images
- Extracts raw text using vision models
- Handles layout, formatting, handwriting

**Models**:
- `qwen2-vl-2b-ocr` - Fast, optimized for OCR
- `qwen2-vl-7b` - More accurate, slower
- `gpt-4-vision` - API-based (requires API key)

**When to use different models**:
- Simple documents → `qwen2-vl-2b-ocr`
- Complex layouts → `qwen2-vl-7b`
- Maximum accuracy → `gpt-4-vision` (API)

### Stage 2: AI Organization (Language Model)

**What it does**:
- Analyzes raw text from Stage 1
- Extracts structured entities
- Organizes into clean JSON
- Corrects OCR errors
- Infers relationships

**Models**:
- `qwen-7b` - Fast, good for simple documents
- `qwen-14b` - Balanced performance
- `qwen-32b` - Most accurate (default)
- `gpt-4` - API-based, very accurate
- `claude-sonnet` - API-based, fast and accurate

**When to use different models**:
- Simple extraction → `qwen-7b`
- Standard documents → `qwen-32b`
- Complex reasoning → `gpt-4` or `claude-opus`

### Pipeline Flow

```
Document (PDF/Image/DOCX/etc.)
    ↓
[Stage 1: OCR Model]
    ↓
Raw Text
    ↓
[Stage 2: Organization Model]
    ↓
Structured JSON
```

---

## Model Selection Guide

### Choosing OCR Model (Stage 1)

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| Standard documents | `qwen2-vl-2b-ocr` | Fast, efficient, good accuracy |
| Complex layouts | `qwen2-vl-7b` | Better understanding of structure |
| Multilingual | `qwen2-vl-7b` | Better language support |
| Maximum accuracy | `gpt-4-vision` | Best accuracy (requires API) |
| No GPU available | `qwen2-vl-2b-ocr` + `--device cpu` | Works on CPU |

### Choosing Organization Model (Stage 2)

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| Simple forms | `qwen-7b` | Fast, sufficient for basic extraction |
| Invoices | `qwen-14b` or `qwen-32b` | Good balance |
| Contracts | `qwen-32b` | Better reasoning |
| Complex analysis | `gpt-4` or `claude-opus` | Superior reasoning |
| Limited GPU | `qwen-7b` | Lower memory requirements |
| No local GPU | `gpt-4o` or `claude-sonnet` | API-based, no local compute |

### Cost Comparison

| Model | Type | Cost | Speed | Accuracy |
|-------|------|------|-------|----------|
| qwen2-vl-2b-ocr | Local | Free* | Fast | Good |
| qwen-7b | Local | Free* | Fast | Good |
| qwen-32b | Local | Free* | Medium | Excellent |
| gpt-4 | API | $$ | Fast | Excellent |
| claude-opus | API | $$$ | Medium | Excellent |

*Free after one-time model download

---

## File Format Support

### Directly Supported (No OCR Needed)

| Format | Extension | Converter | Notes |
|--------|-----------|-----------|-------|
| Plain Text | `.txt` | TextConverter | Multiple encodings supported |
| Word 2007+ | `.docx` | DOCXConverter | Full support including tables |
| Excel 2007+ | `.xlsx` | XLSXConverter | All sheets processed |
| HTML/Web | `.html`, `.htm` | HTMLConverter | Scripts/styles removed |
| CSV | `.csv` | CSVConverter | Automatic delimiter detection |
| RTF | `.rtf` | RTFConverter | Rich text format |

### OCR Required

| Format | Extension | Process | Notes |
|--------|-----------|---------|-------|
| PDF | `.pdf` | Text extraction first, OCR if needed | Hybrid approach |
| PNG | `.png` | OCR | All image formats supported |
| JPEG | `.jpg`, `.jpeg` | OCR | Color and grayscale |
| TIFF | `.tiff`, `.tif` | OCR | Multi-page support |
| BMP | `.bmp` | OCR | Basic format |
| WebP | `.webp` | OCR | Modern format |

### Legacy Formats (Require Extra Tools)

| Format | Extension | Requirements | Notes |
|--------|-----------|--------------|-------|
| Word 97-2003 | `.doc` | antiword (Linux), textutil (macOS), or LibreOffice | Auto-detects available converter |
| Excel 97-2003 | `.xls` | xlrd | Install: `pip install xlrd` |

---

## Command Line Usage

### Basic Commands

```bash
# Extract from document
python3 qwen_extract_ai.py document.pdf

# Extract company information
python3 qwen_extract_ai.py invoice.pdf --company

# Raw OCR only (skip organization)
python3 qwen_extract_ai.py scan.png --raw

# Save to file
python3 qwen_extract_ai.py document.pdf -o output.json

# List available models
python3 qwen_extract_ai.py --list-models
```

### Model Selection

```bash
# Use smaller organization model (faster)
python3 qwen_extract_ai.py document.pdf --org-model qwen-7b

# Use larger OCR model (more accurate)
python3 qwen_extract_ai.py scan.png --ocr-model qwen2-vl-7b

# Use API models
python3 qwen_extract_ai.py document.pdf --org-model gpt-4 --api-key-openai sk-xxx
python3 qwen_extract_ai.py document.pdf --org-model claude-sonnet --api-key-anthropic sk-ant-xxx
```

### Device Selection

```bash
# Force CPU usage
python3 qwen_extract_ai.py document.pdf --device cpu

# Use CUDA GPU
python3 qwen_extract_ai.py document.pdf --device cuda

# Use Apple Silicon GPU
python3 qwen_extract_ai.py document.pdf --device mps

# Auto-detect (default)
python3 qwen_extract_ai.py document.pdf --device auto
```

### Environment Variables

```bash
# Set API keys as environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Use without --api-key flags
python3 qwen_extract_ai.py document.pdf --org-model gpt-4
```

---

## Python API Usage

### Basic Extraction

```python
from pathlib import Path
from model_config import PipelineConfig
from qwen_extract_ai import DualAIExtractor

# Configure pipeline
config = PipelineConfig(
    ocr_model="qwen2-vl-2b-ocr",
    organization_model="qwen-32b",
    device="auto"
)

# Create extractor
extractor = DualAIExtractor(config)

# Extract document
result = extractor.extract(Path("document.pdf"))

# Access results
print(result["data"])          # Organized data
print(result["raw_text"])      # Raw OCR text
print(result["metadata"])      # Document metadata
print(result["pipeline"])      # Processing info
```

### Company Extraction

```python
# Extract company information
result = extractor.extract_company(Path("invoice.pdf"))

company = result["company"]
print(f"Company: {company.get('companyName')}")
print(f"SIREN: {company.get('registrationNumber')}")
print(f"Address: {company.get('address')}")
```

### Custom Schema

```python
# Define custom extraction schema
schema = {
    "invoice_number": "string",
    "date": "string",
    "supplier": {
        "name": "string",
        "address": "string",
        "vat": "string"
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
        "tax": "number",
        "total": "number",
        "currency": "string"
    }
}

# Extract with custom schema
result = extractor.extract(
    Path("invoice.pdf"),
    schema=schema,
    instructions="Extract complete invoice information including all line items"
)

print(result["data"]["invoice_number"])
print(result["data"]["items"])
```

### Using API Models

```python
# Configure with API model
config = PipelineConfig(
    ocr_model="qwen2-vl-2b-ocr",
    organization_model="gpt-4",
    api_keys={
        "openai": "sk-..."
    }
)

extractor = DualAIExtractor(config)
result = extractor.extract(Path("document.pdf"))
```

### Batch Processing

```python
from pathlib import Path
import json

# Process all PDFs in directory
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

    print(f"Saved to {output_file}")
```

---

## Advanced Examples

### Example 1: Invoice Processing Pipeline

```python
from pathlib import Path
from model_config import PipelineConfig
from qwen_extract_ai import DualAIExtractor
import json

# Configure for invoice processing
config = PipelineConfig(
    ocr_model="qwen2-vl-2b-ocr",
    organization_model="qwen-32b"
)

extractor = DualAIExtractor(config)

# Process invoice
result = extractor.extract_company(Path("invoice.pdf"))

# Validate required fields
required_fields = [
    "companyName",
    "registrationNumber",
    "address"
]

company = result["company"]
missing = [field for field in required_fields if not company.get(field)]

if missing:
    print(f"Warning: Missing fields: {missing}")
else:
    print("✓ All required fields extracted")

# Export to JSON
with open("invoice_data.json", 'w') as f:
    json.dump(company, f, indent=2, ensure_ascii=False)
```

### Example 2: Multi-Format Document Processing

```python
from pathlib import Path
from qwen_extract_ai import DualAIExtractor
from model_config import PipelineConfig

# Configure extractor
config = PipelineConfig()
extractor = DualAIExtractor(config)

# Process different formats
files = [
    "report.pdf",
    "data.xlsx",
    "contract.docx",
    "scan.png"
]

results = {}

for filename in files:
    file_path = Path(filename)
    if file_path.exists():
        print(f"Processing {filename}...")
        result = extractor.extract(file_path)
        results[filename] = result["data"]

# Save combined results
with open("all_results.json", 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
```

### Example 3: Using Multiple Models for Comparison

```python
from pathlib import Path
from model_config import PipelineConfig
from qwen_extract_ai import DualAIExtractor
import json

document = Path("complex_invoice.pdf")

# Test with different organization models
models = ["qwen-7b", "qwen-14b", "qwen-32b"]

results = {}

for model_name in models:
    print(f"\nTesting with {model_name}...")

    config = PipelineConfig(
        ocr_model="qwen2-vl-2b-ocr",
        organization_model=model_name
    )

    extractor = DualAIExtractor(config)
    result = extractor.extract(document)

    results[model_name] = {
        "data": result["data"],
        "processing_time": result["pipeline"]["processing_time"]
    }

# Compare results
for model_name, result in results.items():
    print(f"\n{model_name}:")
    print(f"  Time: {result['processing_time']:.2f}s")
    print(f"  Fields extracted: {len(result['data'])}")

# Save comparison
with open("model_comparison.json", 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
```

### Example 4: Error Handling and Retry

```python
from pathlib import Path
from qwen_extract_ai import DualAIExtractor
from model_config import PipelineConfig
import time

def extract_with_retry(file_path, max_retries=3):
    """Extract with retry logic"""

    config = PipelineConfig()
    extractor = DualAIExtractor(config)

    for attempt in range(max_retries):
        try:
            result = extractor.extract(file_path)
            return result

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("Max retries reached")
                raise

    return None

# Use it
try:
    result = extract_with_retry(Path("document.pdf"))
    print("Success!")
except Exception as e:
    print(f"Failed after retries: {e}")
```

---

## Performance Optimization

### 1. Choose the Right Model

```python
# For speed: Use smaller models
config = PipelineConfig(
    ocr_model="qwen2-vl-2b-ocr",
    organization_model="qwen-7b"
)

# For accuracy: Use larger models
config = PipelineConfig(
    ocr_model="qwen2-vl-7b",
    organization_model="qwen-32b"
)

# For balance: Mix and match
config = PipelineConfig(
    ocr_model="qwen2-vl-2b-ocr",  # Fast OCR
    organization_model="qwen-32b"  # Accurate organization
)
```

### 2. Use GPU Acceleration

```python
# Automatic GPU detection
config = PipelineConfig(device="auto")

# Force specific GPU
config = PipelineConfig(device="cuda")  # NVIDIA
config = PipelineConfig(device="mps")   # Apple Silicon
```

### 3. Batch Processing Optimization

```python
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def process_file(file_path):
    """Process single file"""
    # Each thread gets its own extractor
    config = PipelineConfig()
    extractor = DualAIExtractor(config)
    return extractor.extract(file_path)

# Process files in parallel (for API models)
files = list(Path("documents/").glob("*.pdf"))

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_file, files))
```

### 4. Cache Raw OCR Results

```python
import json
from pathlib import Path

cache_dir = Path("ocr_cache/")
cache_dir.mkdir(exist_ok=True)

def extract_with_cache(file_path):
    """Extract with OCR result caching"""

    cache_file = cache_dir / f"{file_path.stem}_ocr.json"

    # Check cache
    if cache_file.exists():
        print(f"Using cached OCR for {file_path.name}")
        with open(cache_file, 'r') as f:
            raw_text = json.load(f)["text"]

        # Only run Stage 2
        # ... (implement Stage 2 only processing)

    else:
        # Full extraction
        result = extractor.extract(file_path)

        # Cache raw OCR
        with open(cache_file, 'w') as f:
            json.dump({"text": result["raw_text"]}, f)

        return result
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
```bash
# Use smaller model
python3 qwen_extract_ai.py doc.pdf --org-model qwen-7b

# Use CPU
python3 qwen_extract_ai.py doc.pdf --device cpu

# Use API model (no local memory)
python3 qwen_extract_ai.py doc.pdf --org-model gpt-4o
```

#### 2. Model Download Fails

**Symptoms**:
```
Error: Connection timeout
```

**Solutions**:
```bash
# Check internet connection
# Models are 10-20GB, takes 5-15 minutes

# Use proxy if needed
export HF_ENDPOINT="https://huggingface.co"

# Manual download
huggingface-cli download Qwen/Qwen2.5-32B-Instruct
```

#### 3. File Format Not Supported

**Symptoms**:
```
ValueError: Unsupported file type: .doc
```

**Solutions**:
```bash
# Install converters
# macOS: textutil is built-in
# Linux:
sudo apt-get install antiword libreoffice

# Or convert manually to .docx
```

#### 4. API Key Errors

**Symptoms**:
```
ValueError: API key required for openai
```

**Solutions**:
```bash
# Set environment variable
export OPENAI_API_KEY="sk-..."

# Or use command line
python3 qwen_extract_ai.py doc.pdf --org-model gpt-4 --api-key-openai sk-...
```

#### 5. Slow Processing

**Solutions**:
- Use GPU instead of CPU
- Use smaller models
- Use API models for parallel processing
- Process fewer pages at once
- Close other applications

### Getting Help

1. Run test suite: `python3 test_system.py`
2. Check model list: `python3 qwen_extract_ai.py --list-models`
3. Verify dependencies: `pip list | grep -E "transformers|torch|pillow"`
4. Check GPU: `python3 -c "import torch; print(torch.cuda.is_available())"`

---

**Last Updated**: December 2025
**Version**: 1.0.0
