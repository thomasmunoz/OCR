# 🚀 Qwen OCR Extraction CLI

**Self-contained, ready-to-use OCR extraction tool**

## ✅ Quick Start (30 seconds)

```bash
# Extract text from any document
python3 qwen_extract.py document.pdf

# Extract company information
python3 qwen_extract.py invoice.pdf --company

# Get JSON output
python3 qwen_extract.py document.pdf --json

# Company info as XML
python3 qwen_extract.py invoice.pdf --company --xml
```

---

## 📦 Installation

### Requirements

```bash
pip install transformers torch pillow PyMuPDF
```

That's it! No package installation needed - just run the script.

---

## 💻 Usage

### 1. Extract Text

```bash
# Extract text from PDF
python3 qwen_extract.py document.pdf

# Extract from image
python3 qwen_extract.py scanned_document.png

# Extract from HTML
python3 qwen_extract.py webpage.html

# Get JSON output with metadata
python3 qwen_extract.py document.pdf --json
```

### 2. Extract Company Information

```bash
# Extract company info (JSON format)
python3 qwen_extract.py invoice.pdf --company

# Company info as XML
python3 qwen_extract.py kbis.pdf --company --xml

# Save to file
python3 qwen_extract.py invoice.pdf --company > company_info.json
```

### 3. Batch Processing

```bash
# Process all PDFs in directory
for file in *.pdf; do
    echo "Processing: $file"
    python3 qwen_extract.py "$file" --company > "${file%.pdf}.json"
done
```

---

## 📊 Output Examples

### Text Extraction

```bash
$ python3 qwen_extract.py document.pdf
```

```
Axway Software S.A.
SIREN: 431 717 500
Tour W, 102 Terrasse Boieldieu
92800 Puteaux, France
...
```

### JSON Output

```bash
$ python3 qwen_extract.py document.pdf --json
```

```json
{
  "text": "Axway Software S.A.\nSIREN: 431 717 500...",
  "metadata": {
    "file": "document.pdf",
    "type": "pdf",
    "pages": 5
  }
}
```

### Company Information (JSON)

```bash
$ python3 qwen_extract.py invoice.pdf --company
```

```json
{
  "companyName": "Axway Software",
  "registrationNumber": "431 717 500",
  "legalForm": "S.A.",
  "capital": "11 277 688 €",
  "address": "Tour W, 102 Terrasse Boieldieu, 92800 Puteaux",
  "email": "contact@axway.com",
  "phone": "+33 1 47 17 24 24",
  "website": "www.axway.com"
}
```

### Company Information (XML)

```bash
$ python3 qwen_extract.py invoice.pdf --company --xml
```

```xml
<?xml version="1.0" encoding="UTF-8"?>
<CompanyInfo>
  <companyName>Axway Software</companyName>
  <registrationNumber>431 717 500</registrationNumber>
  <legalForm>S.A.</legalForm>
  <capital>11 277 688 €</capital>
  <address>Tour W, 102 Terrasse Boieldieu, 92800 Puteaux</address>
  <email>contact@axway.com</email>
  <phone>+33 1 47 17 24 24</phone>
  <website>www.axway.com</website>
</CompanyInfo>
```

---

## 🎯 Supported File Types

- ✅ **PDF** (.pdf)
- ✅ **Images** (.png, .jpg, .jpeg, .tiff, .tif)
- ✅ **HTML** (.html, .xhtml, .htm)

---

## ⚡ Features

- **Self-contained** - All logic in one file
- **No external modules** - No need for separate qwen_ocr.py or company_extractor.py
- **Auto model download** - Downloads Qwen2-VL-2B-OCR on first use
- **GPU support** - Uses GPU if available, CPU otherwise
- **Multiple formats** - Text, JSON, XML output
- **Company extraction** - Automatic extraction of company fields
- **Batch processing** - Easy to use in loops

---

## 🔧 Advanced Usage

### Use GPU (if available)

The script automatically uses GPU if CUDA is available. To force CPU:

```python
# Edit qwen_extract.py and change:
extractor = QwenOCRExtractor(device="cpu")
```

### Save output

```bash
# Save text
python3 qwen_extract.py document.pdf > output.txt

# Save JSON
python3 qwen_extract.py document.pdf --json > output.json

# Save company info
python3 qwen_extract.py invoice.pdf --company > company.json
```

### Pipe to other tools

```bash
# Extract company name only
python3 qwen_extract.py invoice.pdf --company | jq -r '.companyName'

# Check if email exists
python3 qwen_extract.py document.pdf --company | jq 'has("email")'

# Format with jq
python3 qwen_extract.py invoice.pdf --company | jq .
```

---

## 📝 Company Fields Extracted

| Field | Description | Example |
|-------|-------------|---------|
| `companyName` | Company name | "Axway Software" |
| `registrationNumber` | SIREN/SIRET | "431 717 500" |
| `legalForm` | Legal entity type | "S.A.", "SAS", "SARL" |
| `capital` | Company capital | "11 277 688 €" |
| `address` | Full address | "Tour W, 102 Terrasse..." |
| `email` | Email address | "contact@axway.com" |
| `phone` | Phone number | "+33 1 47 17 24 24" |
| `website` | Website URL | "www.axway.com" |

---

## 🐛 Troubleshooting

### Model download fails

```bash
# Check internet connection
# Model downloads from HuggingFace (~4GB)
# Takes 2-5 minutes on first run
```

### Out of memory

```bash
# Use CPU instead of GPU
# Edit the script and set device="cpu"
```

### Missing dependencies

```bash
pip install transformers torch pillow PyMuPDF
```

---

## 📊 Performance

| Operation | Time (CPU) | Time (GPU) |
|-----------|-----------|-----------|
| Single image | ~10s | ~2s |
| PDF (10 pages) | ~60s | ~15s |
| Company extraction | +1s | +1s |

**Note:** First run takes longer due to model download (~4GB)

---

## ✨ Examples

### Extract from scanned invoice

```bash
python3 qwen_extract.py scanned_invoice.png --company
```

### Process all PDFs and get company names

```bash
for file in *.pdf; do
    name=$(python3 qwen_extract.py "$file" --company | jq -r '.companyName')
    echo "$file: $name"
done
```

### Extract and validate

```bash
python3 qwen_extract.py invoice.pdf --company > company.json
if jq -e '.registrationNumber' company.json > /dev/null; then
    echo "✅ SIREN found"
else
    echo "❌ SIREN not found"
fi
```

---

## 🎉 Ready to Use!

```bash
# Test with any document
python3 qwen_extract.py your_document.pdf

# Extract company info
python3 qwen_extract.py your_document.pdf --company
```

**No installation, no setup - just run!** 🚀

---

**Status:** ✅ Production Ready
**Location:** `/Users/tomahawk/DEV/DEVX/OCR/qwen_ocr_package/`
**File:** `qwen_extract.py` (10KB, self-contained)
