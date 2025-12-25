# 🚀 Maximum Information Extraction Guide

## 📊 **What Can Be Extracted?**

The system can extract **11 major categories** of information from any document:

---

## 1️⃣ **FILE METADATA**

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Filename | "invoice_2024.pdf" |
| `path` | Full file path | "/Users/docs/invoice_2024.pdf" |
| `size` | File size in bytes | 2,458,624 |
| `extension` | File type | ".pdf" |
| `modified` | Last modification date | "2024-12-21T10:30:00" |

---

## 2️⃣ **PDF-SPECIFIC METADATA** (for PDF files)

| Field | Description | Example |
|-------|-------------|---------|
| `pages` | Total page count | 15 |
| `encrypted` | Encryption status | false |
| `title` | PDF document title | "Annual Report 2024" |
| `author` | Document author | "John Doe" |
| `creator` | Application used | "Microsoft Word" |
| `creationDate` | PDF creation date | "D:20240115120000" |
| `width/height` | Page dimensions | 595.2 x 841.8 (A4) |
| `rotation` | Page rotation | 0, 90, 180, 270 |
| `images_count` | Embedded images | 12 |
| `links` | Hyperlinks | ["https://example.com"] |
| `has_selectable_text` | Native text present | true/false |

---

## 3️⃣ **IMAGE-SPECIFIC METADATA** (for images)

| Field | Description | Example |
|-------|-------------|---------|
| `width` | Image width in pixels | 2480 |
| `height` | Image height in pixels | 3508 |
| `format` | Image format | "JPEG" |
| `mode` | Color mode | "RGB" |

---

## 4️⃣ **HTML-SPECIFIC METADATA** (for HTML files)

| Field | Description | Example |
|-------|-------------|---------|
| `title` | Page title | "Company Profile" |
| `meta_description` | Meta description | "Leading software company..." |
| `meta_keywords` | Meta keywords | "software, technology, AI" |
| `headings` | All headings (H1-H6) | {"h1": ["Main Title"], "h2": [...]} |
| `links` | All hyperlinks | ["https://...", "mailto:..."] |
| `images` | Image sources | ["logo.png", "banner.jpg"] |
| `tables` | HTML table count | 3 |

---

## 5️⃣ **CONTENT EXTRACTION**

| Field | Description |
|-------|-------------|
| `text` | Full extracted text (OCR) |
| `raw_text` | Unprocessed text with original structure |
| `raw_html` | Original HTML content (for HTML files) |
| `pages` | Per-page breakdown with individual text |

**Per-page information:**
- Page number
- Text content per page
- Text length
- Native text (if available)
- Links on page
- Images on page

---

## 6️⃣ **COMPANY INFORMATION**

| Field | Patterns Detected | Example |
|-------|-------------------|---------|
| `name` | Company name | "Axway Software S.A." |
| `registration_number` | SIREN, SIRET, RCS | "431 717 500" |
| `legal_form` | SAS, SARL, SA, etc. | "S.A." |
| `capital` | Capital social | "11 277 688 €" |
| `vat_number` | TVA intracommunautaire | "FR 12 345 678 901" |
| `trade_register` | RCS registration | "RCS Paris 431 717 500" |

---

## 7️⃣ **PEOPLE & CONTACTS**

### People (with titles)
- Names: "Jean Dupont"
- Titles: "Directeur Général", "PDG", "CEO", "CFO", "CTO"
- Context: Full phrase where found

### Contact Information
| Type | Patterns | Example |
|------|----------|---------|
| **Emails** | All email addresses | contact@company.com |
| **Phones** | French & international formats | +33 1 47 17 24 24 |
| **Addresses** | Street addresses | "102 Rue de la Paix, 75002 Paris" |
| **URLs** | Websites | https://www.company.com |

---

## 8️⃣ **DATES & AMOUNTS**

### Dates (multiple formats)
- DD/MM/YYYY: "21/12/2024"
- DD-MM-YYYY: "21-12-2024"
- French long: "21 décembre 2024"
- ISO: "2024-12-21"

### Monetary Amounts
- Euros: "1 234,56 €"
- USD: "$1,234.56"
- With context (position in document)
- All amounts extracted with their positions

### Numbers
| Type | Example |
|------|---------|
| Integers | 12345, 1000, 500 |
| Decimals | 123.45, 1,234.56 |
| Percentages | 15%, 99.9% |

---

## 9️⃣ **IDENTIFIERS**

| Type | Pattern | Example |
|------|---------|---------|
| Invoice numbers | "Facture #", "Invoice #" | "INV-2024-001" |
| Order numbers | "Commande #", "Order #" | "ORD-12345" |
| Reference codes | "Ref:", "Reference:" | "REF-ABC-123" |

---

## 🔟 **DOCUMENT ANALYSIS**

### Statistics
| Metric | Description |
|--------|-------------|
| `total_characters` | Total character count |
| `total_words` | Total word count |
| `total_lines` | Total line count |
| `unique_words` | Unique vocabulary size |
| `avg_word_length` | Average word length |

### Classification
| Field | Values |
|-------|--------|
| `language` | French, English, Unknown |
| `document_type` | Invoice, Quote, Contract, Report, Company Registration, Unknown |
| `has_tables` | true/false |
| `has_company_info` | true/false |
| `has_amounts` | true/false |
| `has_dates` | true/false |

---

## 1️⃣1️⃣ **STRUCTURE DETECTION**

### Layout
| Feature | Description |
|---------|-------------|
| `total_lines` | Total line count |
| `non_empty_lines` | Lines with content |
| `avg_line_length` | Average line length |
| `has_headers` | Header detected |
| `has_footer` | Footer detected |
| `has_columns` | Multi-column layout |
| `sections` | Section count |

### Tables
- Table detection
- Row/column count
- Table position in document
- Table structure (for HTML)

---

## 💻 **Usage Examples**

### Extract Everything (Human-Readable)

```bash
python3 qwen_extract_max.py document.pdf
```

**Output:**
```
======================================================================
MAXIMUM INFORMATION EXTRACTION RESULTS
======================================================================

📁 FILE INFORMATION
   Name: document.pdf
   Size: 2,458,624 bytes
   Modified: 2024-12-21T10:30:00
   Pages: 15

📊 ANALYSIS
   Characters: 45,234
   Words: 7,856
   Lines: 1,234
   Language: French
   Document Type: Invoice

🏢 COMPANY INFORMATION
   Name: Axway Software S.A.
   SIREN/SIRET: 431 717 500
   Legal Form: S.A.
   Capital: 11 277 688 €

📞 CONTACT INFORMATION
   Emails: contact@axway.com, support@axway.com
   Phones: +33 1 47 17 24 24

📅 DATES FOUND: 12 dates
   Sample: 21/12/2024, 15/01/2024, 2024-12-01

💰 MONETARY AMOUNTS: 8 amounts
   Sample: 1 234,56 €, 5 678,90 €, 999,00 €

📋 TABLES: 2 table(s) detected

👤 PEOPLE: 3 person(s) found
   • Jean Dupont
   • Marie Martin
   • Pierre Lefebvre

📄 TEXT PREVIEW (first 500 characters)
   ------------------------------------------------------------------
   Axway Software S.A.
   Société Anonyme au capital de 11 277 688 €
   ...
```

---

### Extract as JSON (Full Data)

```bash
python3 qwen_extract_max.py document.pdf --json > full_data.json
```

**JSON Structure:**
```json
{
  "file": {
    "name": "document.pdf",
    "path": "/Users/docs/document.pdf",
    "size": 2458624,
    "extension": ".pdf",
    "modified": "2024-12-21T10:30:00",
    "pdf": {
      "pages": 15,
      "encrypted": false,
      "metadata": {
        "title": "Annual Report",
        "author": "John Doe",
        "creator": "Microsoft Word",
        "creationDate": "D:20240115120000"
      }
    }
  },
  "content": {
    "text": "Full extracted text...",
    "raw_text": "Raw text with structure..."
  },
  "structure": {
    "pages": [
      {
        "page_number": 1,
        "width": 595.2,
        "height": 841.8,
        "text": "Page 1 content...",
        "has_selectable_text": true,
        "images_count": 2
      }
    ],
    "tables": [
      {
        "detected": true,
        "line_count": 25,
        "sample": ["Header | Column1 | Column2"]
      }
    ],
    "layout": {
      "total_lines": 1234,
      "has_headers": true,
      "has_footer": true,
      "sections": 5
    }
  },
  "entities": {
    "company": {
      "name": "Axway Software S.A.",
      "registration_number": "431 717 500",
      "legal_form": "S.A.",
      "capital": "11 277 688 €",
      "vat_number": "",
      "trade_register": "RCS Paris 431 717 500"
    },
    "people": [
      {
        "name": "Jean Dupont",
        "context": "Directeur Général: Jean Dupont"
      }
    ],
    "dates": [
      "21/12/2024",
      "15/01/2024"
    ],
    "amounts": [
      {
        "amount": "1 234,56 €",
        "position": 1234
      }
    ],
    "emails": [
      "contact@axway.com"
    ],
    "phones": [
      "+33 1 47 17 24 24"
    ],
    "urls": [
      "https://www.axway.com"
    ],
    "addresses": [
      "102 Terrasse Boieldieu, 92800 Puteaux"
    ],
    "numbers": {
      "integers": ["1234", "5678"],
      "decimals": ["123.45"],
      "percentages": ["15%"]
    },
    "identifiers": {
      "invoice_numbers": ["INV-2024-001"],
      "order_numbers": [],
      "reference_codes": ["REF-ABC"]
    }
  },
  "analysis": {
    "statistics": {
      "total_characters": 45234,
      "total_words": 7856,
      "total_lines": 1234,
      "unique_words": 2345,
      "avg_word_length": 5.75
    },
    "language": "French",
    "document_type": "Invoice",
    "has_tables": true,
    "has_company_info": true,
    "has_amounts": true,
    "has_dates": true
  }
}
```

---

## 🎯 **Practical Use Cases**

### 1. Complete Document Analysis
```bash
# Get everything in JSON
python3 qwen_extract_max.py contract.pdf --json > analysis.json

# Extract specific field with jq
cat analysis.json | jq '.entities.company.name'
```

### 2. Batch Processing with Full Extraction
```bash
for file in *.pdf; do
    echo "Processing: $file"
    python3 qwen_extract_max.py "$file" --json > "${file%.pdf}_full.json"
done
```

### 3. Extract Company Info from All Invoices
```bash
for invoice in invoices/*.pdf; do
    name=$(python3 qwen_extract_max.py "$invoice" --json | jq -r '.entities.company.name')
    amount=$(python3 qwen_extract_max.py "$invoice" --json | jq '.entities.amounts[0].amount')
    echo "$name: $amount"
done
```

### 4. Verify Document Quality
```bash
# Check if document has selectable text
python3 qwen_extract_max.py document.pdf --json | jq '.structure.pages[0].has_selectable_text'

# Count total images
python3 qwen_extract_max.py document.pdf --json | jq '[.structure.pages[].images_count] | add'
```

### 5. Extract All Contact Information
```bash
python3 qwen_extract_max.py business_card.jpg --json | jq '{
  name: .entities.company.name,
  emails: .entities.emails,
  phones: .entities.phones,
  address: .entities.addresses[0]
}'
```

---

## 📊 **Output Size Comparison**

| Mode | Output Size | Information Depth |
|------|-------------|-------------------|
| Basic text | ~5 KB | Text only |
| Company info | ~1 KB | Structured company data |
| **Maximum extraction** | ~50-200 KB | **Everything possible** |

---

## ⚡ **Performance**

| Document Type | Time (CPU) | Time (GPU) | JSON Size |
|--------------|-----------|-----------|-----------|
| Single image | ~15s | ~5s | ~20 KB |
| PDF 10 pages | ~90s | ~30s | ~150 KB |
| HTML page | ~5s | ~2s | ~50 KB |

---

## 🔍 **What Gets Extracted?**

Run this to see all categories:

```bash
python3 qwen_extract_max.py --categories
```

---

## 📝 **Summary**

### **Total Information Categories: 11**

1. ✅ File metadata
2. ✅ PDF-specific data
3. ✅ Image metadata
4. ✅ HTML structure
5. ✅ Full content (text, tables)
6. ✅ Company information
7. ✅ People & contacts
8. ✅ Dates & amounts
9. ✅ Identifiers
10. ✅ Document analysis
11. ✅ Structure detection

### **Total Data Points: 100+**

- **File level:** 20+ fields
- **Content level:** 30+ fields
- **Entity level:** 50+ extracted patterns
- **Analysis level:** 15+ metrics

---

## 🎉 **Ready to Use!**

```bash
# See what can be extracted
python3 qwen_extract_max.py --categories

# Extract everything (human-readable)
python3 qwen_extract_max.py your_document.pdf

# Extract everything (JSON)
python3 qwen_extract_max.py your_document.pdf --json
```

**This is the MAXIMUM extraction possible!** 🚀

---

**File:** `qwen_extract_max.py` (28KB)
**Status:** ✅ Production Ready
**Location:** `/Users/tomahawk/DEV/DEVX/OCR/qwen_ocr_package/`
