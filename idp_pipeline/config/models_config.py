"""
ORACLE-DESIGNED MODEL CONFIGURATION
===================================
Optimal HuggingFace models for each use case.
All models run 100% on-premise.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class DocumentType(Enum):
    """Document type classification"""
    PRINTED_TEXT = "printed_text"
    HANDWRITTEN = "handwritten"
    TABLE = "table"
    FORM = "form"
    INVOICE = "invoice"
    MIXED = "mixed"
    SCANNED = "scanned"
    PHOTO = "photo"

class Language(Enum):
    """Supported languages with ISO codes"""
    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    RUSSIAN = "ru"
    HINDI = "hi"
    MULTI = "multi"  # Multi-language document

@dataclass
class OCRModel:
    """OCR Model specification"""
    name: str
    hf_repo: str
    supported_formats: List[str]
    supported_languages: List[Language]
    supported_doc_types: List[DocumentType]
    vram_required_gb: float
    quality_score: int  # 1-100
    speed_score: int    # 1-100
    description: str

@dataclass
class OrganizationModel:
    """Organization/Structuring Model specification"""
    name: str
    hf_repo: str
    vram_required_gb: float
    context_length: int
    quality_score: int
    speed_score: int
    description: str

# =============================================================================
# STAGE 1: OCR MODELS (HuggingFace)
# =============================================================================

OCR_MODELS: Dict[str, OCRModel] = {
    # ---------------------------------------------------------------------
    # TOP TIER: Best Quality OCR Models
    # ---------------------------------------------------------------------
    "got_ocr2": OCRModel(
        name="GOT-OCR 2.0",
        hf_repo="ucaslcl/GOT-OCR2_0",
        supported_formats=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        supported_languages=[Language.ENGLISH, Language.CHINESE, Language.MULTI],
        supported_doc_types=[DocumentType.PRINTED_TEXT, DocumentType.TABLE, DocumentType.MIXED, DocumentType.SCANNED],
        vram_required_gb=8.0,
        quality_score=98,
        speed_score=60,
        description="State-of-the-art OCR. Best for complex documents, tables, mixed content."
    ),

    "qwen2_vl_7b": OCRModel(
        name="Qwen2-VL-7B",
        hf_repo="Qwen/Qwen2-VL-7B-Instruct",
        supported_formats=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        supported_languages=[l for l in Language],  # All languages
        supported_doc_types=[dt for dt in DocumentType],  # All types
        vram_required_gb=16.0,
        quality_score=95,
        speed_score=50,
        description="Best multi-language support. Excellent for any document type."
    ),

    "florence2_large": OCRModel(
        name="Florence-2-Large",
        hf_repo="microsoft/Florence-2-large",
        supported_formats=["png", "jpg", "jpeg", "tiff", "bmp"],
        supported_languages=[Language.ENGLISH, Language.MULTI],
        supported_doc_types=[DocumentType.PRINTED_TEXT, DocumentType.PHOTO, DocumentType.MIXED],
        vram_required_gb=6.0,
        quality_score=92,
        speed_score=70,
        description="Microsoft's vision model. Excellent for document understanding."
    ),

    # ---------------------------------------------------------------------
    # MID TIER: Good Balance
    # ---------------------------------------------------------------------
    "qwen2_vl_2b": OCRModel(
        name="Qwen2-VL-2B",
        hf_repo="Qwen/Qwen2-VL-2B-Instruct",
        supported_formats=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        supported_languages=[l for l in Language],
        supported_doc_types=[dt for dt in DocumentType],
        vram_required_gb=6.0,
        quality_score=88,
        speed_score=80,
        description="Best ROI. Good quality, fast, multi-language."
    ),

    "donut_base": OCRModel(
        name="Donut-Base",
        hf_repo="naver-clova-ix/donut-base",
        supported_formats=["png", "jpg", "jpeg"],
        supported_languages=[Language.ENGLISH, Language.KOREAN, Language.JAPANESE, Language.CHINESE],
        supported_doc_types=[DocumentType.PRINTED_TEXT, DocumentType.FORM, DocumentType.INVOICE],
        vram_required_gb=4.0,
        quality_score=85,
        speed_score=85,
        description="Fast document understanding. Great for forms and invoices."
    ),

    "trocr_large_printed": OCRModel(
        name="TrOCR-Large-Printed",
        hf_repo="microsoft/trocr-large-printed",
        supported_formats=["png", "jpg", "jpeg"],
        supported_languages=[Language.ENGLISH],
        supported_doc_types=[DocumentType.PRINTED_TEXT, DocumentType.SCANNED],
        vram_required_gb=2.0,
        quality_score=90,
        speed_score=75,
        description="Best for English printed text. Very accurate."
    ),

    "trocr_large_handwritten": OCRModel(
        name="TrOCR-Large-Handwritten",
        hf_repo="microsoft/trocr-large-handwritten",
        supported_formats=["png", "jpg", "jpeg"],
        supported_languages=[Language.ENGLISH],
        supported_doc_types=[DocumentType.HANDWRITTEN],
        vram_required_gb=2.0,
        quality_score=88,
        speed_score=75,
        description="Specialized for handwritten English text."
    ),

    # ---------------------------------------------------------------------
    # SPECIALIZED: Tables and Structure
    # ---------------------------------------------------------------------
    "table_transformer": OCRModel(
        name="Table-Transformer",
        hf_repo="microsoft/table-transformer-structure-recognition",
        supported_formats=["png", "jpg", "jpeg"],
        supported_languages=[l for l in Language],
        supported_doc_types=[DocumentType.TABLE],
        vram_required_gb=2.0,
        quality_score=95,
        speed_score=80,
        description="Specialized for table extraction. Use with other OCR."
    ),

    # ---------------------------------------------------------------------
    # LIGHTWEIGHT: CPU-friendly
    # ---------------------------------------------------------------------
    "surya_ocr": OCRModel(
        name="Surya-OCR",
        hf_repo="vikp/surya_rec",
        supported_formats=["png", "jpg", "jpeg", "pdf"],
        supported_languages=[l for l in Language],
        supported_doc_types=[DocumentType.PRINTED_TEXT, DocumentType.SCANNED],
        vram_required_gb=2.0,
        quality_score=82,
        speed_score=90,
        description="Fast multi-language OCR. CPU-friendly."
    ),
}

# =============================================================================
# STAGE 2: ORGANIZATION MODELS (HuggingFace)
# =============================================================================

ORGANIZATION_MODELS: Dict[str, OrganizationModel] = {
    # ---------------------------------------------------------------------
    # TOP TIER: Best Quality
    # ---------------------------------------------------------------------
    "qwen2_5_7b": OrganizationModel(
        name="Qwen2.5-7B-Instruct",
        hf_repo="Qwen/Qwen2.5-7B-Instruct",
        vram_required_gb=16.0,
        context_length=131072,
        quality_score=95,
        speed_score=60,
        description="Best local model for JSON structuring. 128K context."
    ),

    "llama3_2_3b": OrganizationModel(
        name="Llama-3.2-3B-Instruct",
        hf_repo="meta-llama/Llama-3.2-3B-Instruct",
        vram_required_gb=8.0,
        context_length=131072,
        quality_score=88,
        speed_score=80,
        description="Good balance. Fast and accurate for JSON extraction."
    ),

    "mistral_7b": OrganizationModel(
        name="Mistral-7B-Instruct-v0.3",
        hf_repo="mistralai/Mistral-7B-Instruct-v0.3",
        vram_required_gb=16.0,
        context_length=32768,
        quality_score=90,
        speed_score=70,
        description="Strong performer for structured data extraction."
    ),

    # ---------------------------------------------------------------------
    # MID TIER: Good Balance
    # ---------------------------------------------------------------------
    "phi3_5_mini": OrganizationModel(
        name="Phi-3.5-mini-instruct",
        hf_repo="microsoft/Phi-3.5-mini-instruct",
        vram_required_gb=8.0,
        context_length=131072,
        quality_score=85,
        speed_score=85,
        description="Microsoft's compact model. 128K context, fast."
    ),

    "gemma2_2b": OrganizationModel(
        name="Gemma-2-2B-IT",
        hf_repo="google/gemma-2-2b-it",
        vram_required_gb=6.0,
        context_length=8192,
        quality_score=82,
        speed_score=90,
        description="Google's compact model. Very fast."
    ),

    # ---------------------------------------------------------------------
    # LIGHTWEIGHT: CPU-friendly
    # ---------------------------------------------------------------------
    "qwen2_5_1_5b": OrganizationModel(
        name="Qwen2.5-1.5B-Instruct",
        hf_repo="Qwen/Qwen2.5-1.5B-Instruct",
        vram_required_gb=4.0,
        context_length=131072,
        quality_score=78,
        speed_score=95,
        description="Ultra-fast. Good for simple documents."
    ),

    "tinyllama": OrganizationModel(
        name="TinyLlama-1.1B-Chat",
        hf_repo="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        vram_required_gb=3.0,
        context_length=2048,
        quality_score=70,
        speed_score=98,
        description="Fastest option. CPU-only capable."
    ),
}

# =============================================================================
# MODEL SELECTION RULES
# =============================================================================

MODEL_SELECTION_RULES = {
    # Format-based rules
    "pdf_complex": ["got_ocr2", "qwen2_vl_7b", "qwen2_vl_2b"],
    "pdf_simple": ["qwen2_vl_2b", "florence2_large", "trocr_large_printed"],
    "image_printed": ["trocr_large_printed", "qwen2_vl_2b", "florence2_large"],
    "image_handwritten": ["trocr_large_handwritten", "qwen2_vl_7b"],
    "image_table": ["table_transformer", "got_ocr2", "florence2_large"],
    "image_photo": ["florence2_large", "qwen2_vl_2b"],

    # Language-based rules
    "latin_scripts": ["trocr_large_printed", "qwen2_vl_2b", "florence2_large"],
    "cjk": ["qwen2_vl_7b", "got_ocr2", "donut_base"],  # Chinese, Japanese, Korean
    "arabic_rtl": ["qwen2_vl_7b", "qwen2_vl_2b"],
    "multi_language": ["qwen2_vl_7b", "qwen2_vl_2b", "florence2_large"],

    # Quality-based rules
    "max_quality": ["got_ocr2", "qwen2_vl_7b"],
    "balanced": ["qwen2_vl_2b", "florence2_large"],
    "max_speed": ["florence2_large", "trocr_large_printed", "qwen2_vl_2b"],

    # Organization rules
    "org_max_quality": ["qwen2_5_7b", "mistral_7b"],
    "org_balanced": ["llama3_2_3b", "phi3_5_mini"],
    "org_max_speed": ["gemma2_2b", "qwen2_5_1_5b", "tinyllama"],
}

# =============================================================================
# RECOMMENDED DEFAULTS
# =============================================================================

RECOMMENDED_CONFIG = {
    "ocr_default": "trocr_large_printed",  # Changed to lighter model for Docker
    "ocr_quality": "got_ocr2",
    "ocr_speed": "florence2_large",  # Changed from surya_ocr (no engine impl)
    "org_default": "llama3_2_3b",
    "org_quality": "qwen2_5_7b",
    "org_speed": "gemma2_2b",
}

def get_all_hf_repos() -> Dict[str, str]:
    """Get all HuggingFace repository URLs for download"""
    repos = {}
    for key, model in OCR_MODELS.items():
        repos[f"ocr_{key}"] = model.hf_repo
    for key, model in ORGANIZATION_MODELS.items():
        repos[f"org_{key}"] = model.hf_repo
    return repos

def print_download_commands():
    """Print commands to download all models"""
    print("# OCR Models")
    for key, model in OCR_MODELS.items():
        print(f"# {model.name} ({model.vram_required_gb}GB VRAM)")
        print(f"huggingface-cli download {model.hf_repo}")
        print()

    print("\n# Organization Models")
    for key, model in ORGANIZATION_MODELS.items():
        print(f"# {model.name} ({model.vram_required_gb}GB VRAM)")
        print(f"huggingface-cli download {model.hf_repo}")
        print()
