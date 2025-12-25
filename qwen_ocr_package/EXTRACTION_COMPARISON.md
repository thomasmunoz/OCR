# 📊 Extraction Tools Comparison

## 🎯 **Three Extraction Levels Available**

| Tool | What It Extracts | Use Case | Output Size |
|------|------------------|----------|-------------|
| **`qwen_extract.py`** | Text + basic company info | Quick extraction | ~5 KB |
| **`qwen_extract_max.py`** | **EVERYTHING** (100+ data points) | **Complete analysis** | **~50-200 KB** |

---

## 🔍 **Detailed Comparison**

### **qwen_extract.py** - Basic Extraction

**What you get:**
- ✅ Full text (OCR)
- ✅ Basic company info (8 fields)
- ✅ Simple metadata

**Commands:**
```bash
python3 qwen_extract.py document.pdf
python3 qwen_extract.py invoice.pdf --company
python3 qwen_extract.py document.pdf --json
```

**Output example:**
```json
{
  "companyName": "Axway Software",
  "registrationNumber": "431 717 500",
  "legalForm": "S.A.",
  "capital": "11 277 688 €",
  "address": "Tour W, 92800 Puteaux",
  "email": "contact@axway.com",
  "phone": "+33 1 47 17 24 24",
  "website": "www.axway.com"
}
```

---

### **qwen_extract_max.py** - MAXIMUM Extraction ⭐

**What you get (100+ data points):**

#### File Information
- ✅ Name, path, size, extension, modified date
- ✅ PDF: pages, metadata, encryption, dimensions, images, links
- ✅ Image: width, height, format, color mode
- ✅ HTML: title, meta tags, headings, links

#### Content
- ✅ Full OCR text
- ✅ Raw text with structure
- ✅ Per-page breakdown
- ✅ Tables detection and extraction
- ✅ Layout structure (headers, footers, columns)

#### Company Information
- ✅ Company name
- ✅ SIREN/SIRET
- ✅ Legal form
- ✅ Capital
- ✅ VAT number
- ✅ Trade register (RCS)

#### People & Contacts
- ✅ Names with titles (CEO, CFO, directors)
- ✅ All email addresses
- ✅ All phone numbers
- ✅ Physical addresses
- ✅ Websites/URLs

#### Dates & Amounts
- ✅ All dates (multiple formats)
- ✅ All monetary amounts (€, $)
- ✅ Percentages
- ✅ All numbers (integers, decimals)

#### Identifiers
- ✅ Invoice numbers
- ✅ Order numbers
- ✅ Reference codes

#### Document Analysis
- ✅ Character/word/line counts
- ✅ Unique vocabulary
- ✅ Average word length
- ✅ Language detection
- ✅ Document type classification
- ✅ Structural analysis

**Commands:**
```bash
# Human-readable output
python3 qwen_extract_max.py document.pdf

# Full JSON output
python3 qwen_extract_max.py document.pdf --json

# See all categories
python3 qwen_extract_max.py --categories
```

**Output structure:**
```json
{
  "file": { /* 20+ fields */ },
  "content": { /* Full text + structure */ },
  "structure": {
    "pages": [ /* Per-page info */ ],
    "tables": [ /* Detected tables */ ],
    "layout": { /* Layout analysis */ }
  },
  "entities": {
    "company": { /* 6 fields */ },
    "people": [ /* Names + titles */ ],
    "dates": [ /* All dates */ ],
    "amounts": [ /* All amounts */ ],
    "emails": [ /* All emails */ ],
    "phones": [ /* All phones */ ],
    "urls": [ /* All URLs */ ],
    "addresses": [ /* All addresses */ ],
    "numbers": { /* Integers, decimals, percentages */ },
    "identifiers": { /* Invoice #, Order #, etc. */ }
  },
  "analysis": {
    "statistics": { /* 5 metrics */ },
    "language": "French",
    "document_type": "Invoice",
    "has_tables": true,
    "has_company_info": true
  }
}
```

---

## 📊 **Side-by-Side Comparison**

| Feature | Basic | **Maximum** |
|---------|-------|-------------|
| **Text Extraction** | ✅ | ✅ |
| **Company Name** | ✅ | ✅ |
| **SIREN/SIRET** | ✅ | ✅ |
| **Legal Form** | ✅ | ✅ |
| **Capital** | ✅ | ✅ |
| **Email** | ✅ | ✅ (all) |
| **Phone** | ✅ | ✅ (all) |
| **Address** | ✅ | ✅ (all) |
| **Website** | ✅ | ✅ (all URLs) |
| **VAT Number** | ❌ | ✅ |
| **Trade Register** | ❌ | ✅ |
| **People (names + titles)** | ❌ | ✅ |
| **All Dates** | ❌ | ✅ |
| **All Amounts** | ❌ | ✅ |
| **Invoice/Order Numbers** | ❌ | ✅ |
| **File Metadata** | Basic | ✅ Complete |
| **PDF Metadata** | ❌ | ✅ |
| **Per-Page Analysis** | ❌ | ✅ |
| **Table Detection** | ❌ | ✅ |
| **Layout Analysis** | ❌ | ✅ |
| **Language Detection** | ❌ | ✅ |
| **Document Type** | ❌ | ✅ |
| **Statistics** | ❌ | ✅ |
| **Links (PDF/HTML)** | ❌ | ✅ |
| **Embedded Images** | ❌ | ✅ |
| **HTML Structure** | ❌ | ✅ |
| **Number Extraction** | ❌ | ✅ |
| **Percentages** | ❌ | ✅ |
| **Section Detection** | ❌ | ✅ |

**Total Data Points:**
- Basic: ~10 fields
- **Maximum: 100+ fields** ⭐

---

## 🎯 **When to Use Each Tool**

### Use **`qwen_extract.py`** when:
- ✅ You just need text
- ✅ You want basic company info (8 fields)
- ✅ You need fast results
- ✅ Small output size is important

### Use **`qwen_extract_max.py`** when: ⭐
- ✅ You need **EVERYTHING**
- ✅ Complete document analysis required
- ✅ Building a database with rich metadata
- ✅ Document classification
- ✅ Compliance/audit requirements
- ✅ Research and data mining
- ✅ Quality assurance checks
- ✅ Finding ALL instances (dates, amounts, contacts)

---

## 💡 **Real-World Examples**

### Scenario 1: Quick Invoice Processing
**Need:** Just company name and amount
**Tool:** `qwen_extract.py --company`
**Time:** ~10s
**Output:** 8 fields

### Scenario 2: Complete Due Diligence
**Need:** Every piece of information
**Tool:** `qwen_extract_max.py --json` ⭐
**Time:** ~15s
**Output:** 100+ fields with metadata

### Scenario 3: Building a Database
**Need:** All contacts from 1000 documents
**Tool:** `qwen_extract_max.py` in batch mode ⭐
**Why:** Extracts ALL emails, phones, addresses (not just first)

### Scenario 4: Compliance Audit
**Need:** Verify document completeness
**Tool:** `qwen_extract_max.py` ⭐
**Why:** Analyzes structure, checks for required fields, counts pages/images

---

## 📈 **Performance Comparison**

| Tool | Single Image | PDF (10 pages) | JSON Size |
|------|--------------|----------------|-----------|
| Basic | ~10s | ~60s | ~5 KB |
| **Maximum** | ~15s | ~90s | **~150 KB** |

**Extra time for maximum extraction: +5-30 seconds**
**Extra information gained: 10x more data points** ⭐

---

## 🚀 **Quick Start Commands**

### Basic Extraction
```bash
# Text only
python3 qwen_extract.py document.pdf

# Company info (8 fields)
python3 qwen_extract.py invoice.pdf --company
```

### Maximum Extraction ⭐
```bash
# Human-readable summary
python3 qwen_extract_max.py document.pdf

# Complete JSON (100+ fields)
python3 qwen_extract_max.py document.pdf --json > full_data.json

# See all extraction categories
python3 qwen_extract_max.py --categories
```

---

## 💰 **Value Proposition**

### Basic Tool
- **Input:** 1 document
- **Output:** 10 fields
- **Use case:** Quick checks

### Maximum Tool ⭐
- **Input:** 1 document
- **Output:** 100+ fields across 11 categories
- **Use case:** Complete intelligence

**10x more information for just 50% more processing time!**

---

## 🎯 **Recommendation**

### For Production Use:
**Start with Maximum Extraction** (`qwen_extract_max.py`) because:

1. ✅ Extract once, use forever
2. ✅ No need to re-process when you need more data
3. ✅ Complete audit trail
4. ✅ Filter down in post-processing
5. ✅ Only 50% more time for 10x more data

### For Quick Tests:
Use Basic Extraction (`qwen_extract.py`)

---

## 📊 **Summary Table**

|  | **Basic** | **Maximum** ⭐ |
|---|-----------|----------------|
| **Data Points** | 10 | 100+ |
| **Categories** | 2 | 11 |
| **Processing Time** | Fast | +50% |
| **Output Size** | ~5 KB | ~150 KB |
| **File Metadata** | Minimal | Complete |
| **Entities** | Company only | All (company, people, dates, amounts) |
| **Analysis** | None | Full statistics |
| **Structure** | None | Complete layout |
| **Best For** | Quick checks | Complete intelligence |

---

## 🎉 **Both Tools Ready!**

Choose based on your needs:
- **Quick & Simple:** `qwen_extract.py`
- **Complete & Comprehensive:** `qwen_extract_max.py` ⭐

**Location:** `/Users/tomahawk/DEV/DEVX/OCR/qwen_ocr_package/`

---

**Recommendation: Use Maximum Extraction for production!** 🚀
