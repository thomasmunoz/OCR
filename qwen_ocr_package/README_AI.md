# 🚀 Dual-AI OCR Extraction System

**Production-ready two-stage AI pipeline for document extraction and organization**

## 📋 Overview

This system combines two AI models in sequence:

1. **Stage 1 (OCR)**: Vision model extracts raw text from documents
2. **Stage 2 (Organization)**: Language model organizes text into structured JSON

### Why Two Stages?

- **Stage 1** handles vision tasks (reading scanned documents, PDFs, images)
- **Stage 2** handles reasoning tasks (understanding context, extracting entities, organizing data)
- **Result**: Better accuracy, more structured output, easier to customize

---

## ✨ Features

### Multi-Format Support
- **PDF** (scanned or native text)
- **Images**: PNG, JPG, JPEG, TIFF, BMP, WEBP
- **Documents**: DOCX, DOC (legacy Word)
- **Spreadsheets**: XLSX, XLS (legacy Excel)
- **Web**: HTML, XHTML
- **Text**: TXT, CSV, RTF

### Flexible Model Selection

**OCR Models (Stage 1 - Vision)**:
- `qwen2-vl-2b-ocr` - Default, optimized for text extraction (local)
- `qwen2-vl-7b` - Larger, more accurate (local)
- `gpt-4-vision` - OpenAI's vision model (API)
- `claude-3-opus` - Anthropic's vision model (API)

**Organization Models (Stage 2 - LLM)**:
- `qwen-32b` - Default, powerful local model
- `qwen-14b` - Faster, smaller alternative
- `qwen-7b` - Lightweight for resource-constrained systems
- `gpt-4` / `gpt-4o` - OpenAI's language models (API)
- `claude-opus` / `claude-sonnet` - Anthropic's models (API)

### Intelligent Organization

Stage 2 automatically extracts:
- Company information (name, registration, VAT, etc.)
- Addresses and contact details
- Financial data (amounts, currencies, line items)
- Dates and periods
- People and roles
- Document metadata
- Custom entities based on schema

---

## 🚀 Quick Start

### Installation

```bash
# Minimum install (local models only)
pip install transformers torch pillow PyMuPDF python-docx openpyxl

# Full install (with API support)
pip install -r requirements_ai.txt
```

### Basic Usage

```bash
# Use default models (Qwen2-VL-2B-OCR + Qwen-32B)
python3 qwen_extract_ai.py document.pdf

# Extract company information
python3 qwen_extract_ai.py invoice.pdf --company

# Process DOCX, XLSX, or other formats
python3 qwen_extract_ai.py report.docx
python3 qwen_extract_ai.py spreadsheet.xlsx

# Raw OCR only (skip Stage 2)
python3 qwen_extract_ai.py document.pdf --raw
```

### Choose Models

```bash
# Use smaller local model (faster)
python3 qwen_extract_ai.py document.pdf --org-model qwen-7b

# Use GPT-4 for organization
python3 qwen_extract_ai.py document.pdf --org-model gpt-4 --api-key-openai sk-xxx

# Use Claude for organization
python3 qwen_extract_ai.py document.pdf --org-model claude-sonnet --api-key-anthropic sk-xxx

# Mix local OCR + API organization
python3 qwen_extract_ai.py scan.pdf --ocr-model qwen2-vl-2b-ocr --org-model gpt-4o
```

### List Available Models

```bash
python3 qwen_extract_ai.py --list-models
```

---

## 💻 Advanced Usage

### Custom Organization Schema

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

# Define custom schema
schema = {
    "invoice_number": "string",
    "date": "string",
    "supplier": {
        "name": "string",
        "address": "string"
    },
    "items": [
        {
            "description": "string",
            "quantity": "number",
            "price": "number"
        }
    ],
    "total": "number"
}

# Extract with schema
result = extractor.extract(
    Path("invoice.pdf"),
    schema=schema,
    instructions="Extract complete invoice information"
)

print(result["data"])
```

### Batch Processing

```bash
# Process all PDFs in directory
for file in *.pdf; do
    echo "Processing: $file"
    python3 qwen_extract_ai.py "$file" --company -o "${file%.pdf}.json"
done

# Process mixed formats
for file in documents/*; do
    python3 qwen_extract_ai.py "$file" -o "results/$(basename $file).json"
done
```

### Environment Variables for API Keys

```bash
# Set API keys as environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Use without --api-key flags
python3 qwen_extract_ai.py document.pdf --org-model gpt-4
```

### Save Configuration

```python
from model_config import PipelineConfig

# Create and save configuration
config = PipelineConfig(
    ocr_model="qwen2-vl-2b-ocr",
    organization_model="qwen-32b",
    device="cuda"
)

config.save(Path("my_config.json"))

# Load configuration
config = PipelineConfig.load(Path("my_config.json"))
```

---

## 📊 Output Examples

### Default Extraction

```bash
$ python3 qwen_extract_ai.py invoice.pdf
```

```json
{
  "data": {
    "company": {
      "name": "Acme Corporation",
      "registrationNumber": "123 456 789",
      "address": {
        "street": "123 Main Street",
        "city": "Paris",
        "postalCode": "75001",
        "country": "France"
      }
    },
    "invoice": {
      "number": "INV-2025-001",
      "date": "2025-01-15",
      "dueDate": "2025-02-15"
    },
    "items": [
      {
        "description": "Software License",
        "quantity": 10,
        "unitPrice": 99.99,
        "total": 999.90
      }
    ],
    "totals": {
      "subtotal": 999.90,
      "tax": 199.98,
      "total": 1199.88,
      "currency": "EUR"
    }
  },
  "raw_text": "Acme Corporation\n123 456 789\n...",
  "metadata": {
    "file": "invoice.pdf",
    "type": "pdf",
    "pages": 2
  },
  "pipeline": {
    "stage1_model": "qwen2-vl-2b-ocr",
    "stage2_model": "qwen-32b",
    "processing_time": 12.45
  }
}
```

### Company Extraction

```bash
$ python3 qwen_extract_ai.py kbis.pdf --company
```

```json
{
  "company": {
    "companyName": "Société Exemple SAS",
    "legalForm": "SAS",
    "registrationNumber": "123 456 789",
    "vatNumber": "FR12345678900",
    "capital": "100,000 EUR",
    "address": {
      "street": "10 Rue de la Paix",
      "city": "Paris",
      "postalCode": "75002",
      "country": "France"
    },
    "contact": {
      "email": "contact@exemple.fr",
      "phone": "+33 1 23 45 67 89",
      "website": "www.exemple.fr"
    }
  },
  "raw_text": "...",
  "metadata": {...},
  "pipeline": {...}
}
```

### Raw OCR (Stage 1 Only)

```bash
$ python3 qwen_extract_ai.py scan.png --raw
```

```json
{
  "raw_text": "Invoice\nDate: 2025-01-15\nAmount: €1,199.88\n...",
  "metadata": {
    "file": "scan.png",
    "type": "image",
    "format": ".png"
  },
  "pipeline": {
    "stage1_model": "qwen2-vl-2b-ocr",
    "stage2_model": null,
    "processing_time": 2.34
  }
}
```

---

## 🎯 Use Cases

### 1. Invoice Processing

```bash
python3 qwen_extract_ai.py invoice.pdf --company
```

Extracts:
- Supplier and client details
- Invoice number and dates
- Line items with prices
- Totals and tax
- Payment terms

### 2. Contract Analysis

```bash
python3 qwen_extract_ai.py contract.docx
```

Extracts:
- Parties involved
- Contract dates and terms
- Financial clauses
- Obligations and rights

### 3. Company Registration Documents

```bash
python3 qwen_extract_ai.py kbis.pdf --company
```

Extracts:
- Company identification
- Legal structure
- Registration numbers
- Address and contact

### 4. Multi-Language Documents

```python
# Handles French, English, Spanish, etc. automatically
result = extractor.extract(
    Path("multilingual_doc.pdf"),
    instructions="Extract all information in original language"
)
```

### 5. Spreadsheet Data

```bash
python3 qwen_extract_ai.py financial_report.xlsx
```

Extracts structured data from Excel files with intelligent organization.

---

## 🔧 Configuration Reference

### Pipeline Configuration

```python
from model_config import PipelineConfig

config = PipelineConfig(
    ocr_model="qwen2-vl-2b-ocr",      # OCR model name
    organization_model="qwen-32b",     # Organization model name
    device="auto",                     # 'auto', 'cpu', 'cuda', 'mps'
    api_keys={                         # API keys for API models
        "openai": "sk-...",
        "anthropic": "sk-ant-..."
    }
)
```

### Model Parameters

Each model has configurable parameters in `model_config.py`:

```python
# Example: Qwen-32B parameters
{
    "model_id": "Qwen/Qwen2.5-32B-Instruct",
    "max_tokens": 8192,
    "torch_dtype": "bfloat16",
    "temperature": 0.1
}
```

### Device Selection

- `auto`: Automatically selects best available device
- `cpu`: Force CPU usage
- `cuda`: Use NVIDIA GPU
- `mps`: Use Apple Silicon GPU

---

## 📁 File Structure

```
qwen_ocr_package/
├── qwen_extract_ai.py      # Main dual-AI extractor
├── model_config.py          # Model configuration system
├── ai_organizer.py          # Stage 2 AI organization
├── format_converters.py     # Document format converters
├── requirements_ai.txt      # Dependencies
├── README_AI.md            # This file
│
├── qwen_extract.py         # Legacy single-stage extractor
├── extract_cli.py          # Simple CLI wrapper
└── README.md               # Original documentation
```

---

## ⚡ Performance

### Processing Times (Approximate)

| Document Type | Stage 1 (OCR) | Stage 2 (Org) | Total |
|--------------|---------------|---------------|-------|
| Single image | 2-5s (GPU)    | 3-8s         | 5-13s |
| PDF (10 pages)| 15-30s (GPU)  | 5-10s        | 20-40s|
| DOCX (no OCR)| 0.1s          | 3-8s         | 3-8s  |
| XLSX (no OCR)| 0.5s          | 3-8s         | 3-8s  |

**Note**: First run takes longer due to model download (~10-20GB for local models)

### Hardware Requirements

**Minimum (CPU only)**:
- RAM: 16GB
- Storage: 30GB free space
- Processing: Slow but functional

**Recommended (GPU)**:
- GPU: 8GB+ VRAM (NVIDIA/Apple Silicon)
- RAM: 32GB
- Storage: 50GB free space
- Processing: Fast and efficient

**API Models**:
- No local GPU required
- Fast processing
- Pay per use

---

## 🐛 Troubleshooting

### Model Download Issues

```bash
# Check internet connection
# Models download from HuggingFace (10-20GB)
# Takes 5-15 minutes on first run

# Force CPU if GPU issues
python3 qwen_extract_ai.py document.pdf --device cpu
```

### Out of Memory

```bash
# Use smaller model
python3 qwen_extract_ai.py document.pdf --org-model qwen-7b

# Use CPU instead of GPU
python3 qwen_extract_ai.py document.pdf --device cpu

# Use API model (no local memory needed)
python3 qwen_extract_ai.py document.pdf --org-model gpt-4o
```

### API Key Errors

```bash
# Set environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use command line
python3 qwen_extract_ai.py doc.pdf --org-model gpt-4 --api-key-openai sk-...
```

### File Format Issues

```bash
# Check supported formats
python3 format_converters.py

# For DOC files, install converters:
# macOS: Built-in textutil works
# Linux: sudo apt-get install antiword
# Or: Use LibreOffice: sudo apt-get install libreoffice
```

---

## 🔒 Security & Privacy

### Local Models
- ✅ All processing happens on your machine
- ✅ No data sent to external servers
- ✅ Complete privacy

### API Models
- ⚠️ Data sent to API provider (OpenAI/Anthropic)
- ✅ Data encrypted in transit
- ⚠️ Check provider's privacy policy
- 💡 Use local models for sensitive documents

---

## 📚 API Reference

### DualAIExtractor

```python
class DualAIExtractor:
    def __init__(self, pipeline_config: PipelineConfig, verbose: bool = True)

    def extract(
        self,
        file_path: Path,
        organize: bool = True,
        schema: Optional[Dict[str, Any]] = None,
        instructions: Optional[str] = None
    ) -> Dict[str, Any]

    def extract_company(self, file_path: Path) -> Dict[str, Any]
```

### PipelineConfig

```python
class PipelineConfig:
    def __init__(
        self,
        ocr_model: str = "qwen2-vl-2b-ocr",
        organization_model: str = "qwen-32b",
        device: str = "auto",
        api_keys: Optional[Dict[str, str]] = None
    )

    def save(self, path: Path) -> None

    @classmethod
    def load(cls, path: Path) -> 'PipelineConfig'
```

### ModelRegistry

```python
class ModelRegistry:
    @classmethod
    def get_ocr_model(cls, model_name: str) -> Optional[ModelConfig]

    @classmethod
    def get_organization_model(cls, model_name: str) -> Optional[ModelConfig]

    @classmethod
    def list_ocr_models(cls) -> List[str]

    @classmethod
    def list_organization_models(cls) -> List[str]

    @classmethod
    def print_available_models(cls) -> None
```

---

## 🎓 Examples

See the `examples/` directory for:
- Invoice processing
- Contract analysis
- Batch processing scripts
- Custom schema examples
- Integration examples

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional format converters
- New model integrations
- Performance optimizations
- Documentation improvements

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Qwen Team**: Qwen2-VL and Qwen2.5 models
- **HuggingFace**: Transformers library
- **OpenAI**: GPT-4 Vision and language models
- **Anthropic**: Claude 3 models

---

## 📞 Support

For issues, questions, or contributions:
- GitHub Issues: [Create an issue]
- Documentation: This README
- Model docs: See `model_config.py`

---

**Status**: ✅ Production Ready
**Version**: 1.0.0
**Last Updated**: December 2025

🚀 **Ready to extract!**
