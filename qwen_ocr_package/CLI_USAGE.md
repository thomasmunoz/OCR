# 🚀 OCR CLI - Quick Usage Guide

## ✅ Available Commands

You now have 2 CLI tools ready to use:

### 1. **Text Extraction** - `extract_cli.py`

Extract raw text from any document.

```bash
# Basic extraction (prints text to console)
python3 extract_cli.py document.pdf

# Extract as JSON (includes metadata)
python3 extract_cli.py document.pdf --format json

# Save to file
python3 extract_cli.py document.pdf > output.txt
python3 extract_cli.py document.pdf --format json > output.json
```

**Supported formats:** PDF, HTML, XHTML, PNG, JPG, JPEG, TIFF

---

### 2. **Company Information Extraction** - `extract_company_cli.py`

Extract structured company information (name, SIREN, address, etc.)

```bash
# Extract company info as JSON
python3 extract_company_cli.py invoice.pdf

# Compact JSON (AI-optimized, 30-50% fewer tokens)
python3 extract_company_cli.py invoice.pdf --format json-ai

# XML format
python3 extract_company_cli.py kbis.pdf --format xml

# Save to file
python3 extract_company_cli.py invoice.pdf > company_info.json
python3 extract_company_cli.py invoice.pdf --format xml > company_info.xml
```

**Extracted fields:**
- Company name
- SIREN/SIRET
- Legal form
- Capital
- Address (full + components)
- Email
- Phone
- Website
- Executives
- Fiscal year
- Confidence score

---

## 📊 Output Formats

### JSON Standard (Default)
```json
{
  "companyName": "Axway Software",
  "registrationNumber": "431 717 500",
  "legalForm": "S.A.",
  "capital": "11 277 688 €",
  "address": {
    "full": "Tour W, 102 Terrasse Boieldieu, 92800 Puteaux",
    "street": "102 Terrasse Boieldieu",
    "city": "Puteaux",
    "postalCode": "92800"
  },
  "email": "contact@axway.com",
  "phone": "+33 1 47 17 24 24",
  "metadata": {
    "confidence": 83.33
  }
}
```

### JSON-AI (Compact)
```json
{"name":"Axway Software","reg":"431 717 500","legal":"S.A.","cap":"11 277 688 €","addr":"Tour W, 102 Terrasse Boieldieu, 92800 Puteaux","email":"contact@axway.com","phone":"+33 1 47 17 24 24","conf":83.33}
```

### XML
```xml
<?xml version="1.0" encoding="UTF-8"?>
<CompanyInfo>
  <CompanyName>Axway Software</CompanyName>
  <RegistrationNumber>431 717 500</RegistrationNumber>
  <LegalForm>S.A.</LegalForm>
  <Capital>11 277 688 €</Capital>
  ...
</CompanyInfo>
```

---

## 💡 Common Use Cases

### Extract text from PDF and save
```bash
python3 extract_cli.py report.pdf > extracted_text.txt
```

### Extract company info and pipe to another tool
```bash
python3 extract_company_cli.py invoice.pdf | jq '.companyName'
```

### Process multiple files
```bash
for file in *.pdf; do
    echo "Processing: $file"
    python3 extract_company_cli.py "$file" > "${file%.pdf}.json"
done
```

### Get just the company name
```bash
python3 extract_company_cli.py invoice.pdf | jq -r '.companyName'
```

### Check confidence score
```bash
python3 extract_company_cli.py document.pdf | jq '.metadata.confidence'
```

---

## 🔧 Requirements

These CLI tools require:
- `qwen_ocr.py` in parent directory (for text extraction)
- `company_extractor.py` in parent directory (for company extraction)
- Python 3.8+
- Dependencies installed (transformers, torch, etc.)

---

## ⚡ Quick Test

Try with a test file:

```bash
# Extract text
python3 extract_cli.py ../test-company-doc.html

# Extract company info
python3 extract_company_cli.py ../test-company-doc.html
```

---

## 📝 Notes

- First run will download the AI model (~4GB) - this is automatic
- GPU is used automatically if available, otherwise CPU
- Processing time: 5-30 seconds depending on document size and hardware

---

**Status:** ✅ Ready to use!
**Location:** `/Users/tomahawk/DEV/DEVX/OCR/qwen_ocr_package/`
