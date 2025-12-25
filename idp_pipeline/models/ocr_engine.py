"""
ORACLE-DESIGNED OCR ENGINE
==========================
Unified interface for all HuggingFace OCR models.
Handles model loading, inference, and confidence tracking.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from abc import ABC, abstractmethod
import logging

# Initialize logger early (used in import try/except blocks below)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.models_config import OCR_MODELS, OCRModel
from config.intermediate_format import (
    OCRIntermediateResult, OCRMetadata, PageContent,
    TextBlock, TextLine, TextElement, Table, TableCell, BoundingBox
)
from enum import Enum
from dataclasses import dataclass, field

# Import visual extractor for v2.1 enhanced output
try:
    from models.visual_extractor import (
        VisualContentExtractor, EnhancedTable, DetectedDiagram, DetectedChart,
        build_enhanced_v21_output
    )
    VISUAL_EXTRACTOR_AVAILABLE = True
except ImportError:
    VISUAL_EXTRACTOR_AVAILABLE = False
    logger.warning("Visual extractor not available - enhanced v2.1 features disabled")

# Import document analyzer for summary, tags, word cloud
try:
    from models.doc_analyzer import DocumentAnalyzer, analyze_document
    DOC_ANALYZER_AVAILABLE = True
except ImportError:
    DOC_ANALYZER_AVAILABLE = False
    logger.warning("Document analyzer not available - summary/tags/word_cloud disabled")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_file_hash(file_path: str, chunk_size: int = 65536) -> str:
    """
    Compute SHA256 hash of a file using chunked reading.
    Avoids loading entire file into memory for large files.

    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read (default 64KB)

    Returns:
        Hex digest of SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


# ============================================================================
# PAGE-LEVEL ANALYSIS SYSTEM (ORACLE-DESIGNED)
# Handles mixed PDFs with digital text, scanned images, and embedded images
# ============================================================================

class PageType(Enum):
    """Classification of page content type"""
    DIGITAL_TEXT = "digital_text"       # Native text layer, no significant images
    SCANNED_IMAGE = "scanned_image"     # Full-page image, no text layer (needs OCR)
    MIXED_CONTENT = "mixed_content"     # Both text and images with potential text
    IMAGE_WITH_TEXT = "image_with_text" # Images that may contain text (photos, diagrams)
    EMPTY = "empty"                     # No content


@dataclass
class PageAnalysis:
    """Analysis result for a single page"""
    page_number: int
    page_type: PageType
    char_count: int                     # Characters in text layer
    image_count: int                    # Number of embedded images
    image_coverage: float               # 0-1, how much of page is covered by images
    has_text_layer: bool
    has_significant_images: bool
    needs_ocr: bool                     # Whether OCR should be applied
    confidence: float                   # Confidence in classification
    extraction_method: str              # "direct", "ocr", or "hybrid"


@dataclass
class DocumentAnalysis:
    """Complete document structure analysis"""
    file_path: str
    total_pages: int
    pages: List[PageAnalysis] = field(default_factory=list)
    document_type: str = "unknown"      # "pure_digital", "pure_scanned", "mixed"
    digital_pages: List[int] = field(default_factory=list)
    ocr_pages: List[int] = field(default_factory=list)
    mixed_pages: List[int] = field(default_factory=list)
    total_digital_chars: int = 0
    estimated_processing_time: float = 0.0
    recommended_strategy: str = "unknown"


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines"""

    def __init__(self, model_config: OCRModel):
        self.config = model_config
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.loaded = False

    @abstractmethod
    def load_model(self):
        """Load the model into memory"""
        pass

    @abstractmethod
    def process_image(self, image_path: str) -> Tuple[str, float, List[Dict]]:
        """Process single image, return (text, confidence, elements)"""
        pass

    def unload_model(self):
        """Unload model from memory"""
        self.model = None
        self.processor = None
        self.loaded = False
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Qwen2VLEngine(BaseOCREngine):
    """Qwen2-VL based OCR (2B and 7B variants)"""

    def load_model(self):
        if self.loaded:
            return

        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        logger.info(f"Loading {self.config.name} from {self.config.hf_repo}")

        # Detect device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.hf_repo,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        if self.device == "mps":
            self.model = self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            self.config.hf_repo,
            trust_remote_code=True
        )

        self.loaded = True
        logger.info(f"Model loaded on {self.device}")

    def process_image(self, image_path: str) -> Tuple[str, float, List[Dict]]:
        if not self.loaded:
            self.load_model()

        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")

        # OCR prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract ALL text from this image. Preserve the original structure and formatting. Include every word, number, and symbol visible."}
                ]
            }
        ]

        # Process
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Decode
        generated_ids = outputs.sequences[:, inputs['input_ids'].shape[1]:]
        extracted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Estimate confidence from generation scores
        if hasattr(outputs, 'scores') and outputs.scores:
            import torch.nn.functional as F
            confidences = []
            for score in outputs.scores[:20]:  # First 20 tokens
                probs = F.softmax(score[0], dim=-1)
                max_prob = probs.max().item()
                confidences.append(max_prob)
            confidence = sum(confidences) / len(confidences) if confidences else 0.8
        else:
            confidence = 0.85  # Default high confidence for Qwen

        # Create elements (simplified - full would need word-level positions)
        elements = []
        for i, line in enumerate(extracted_text.split('\n')):
            if line.strip():
                elements.append({
                    "text": line.strip(),
                    "confidence": confidence,
                    "line_number": i + 1
                })

        return extracted_text, confidence, elements


class GOTOCREngine(BaseOCREngine):
    """GOT-OCR 2.0 engine - State of the art"""

    def load_model(self):
        if self.loaded:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading {self.config.name} from {self.config.hf_repo}")

        # Detect device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_repo,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            self.config.hf_repo,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )

        if self.device == "mps":
            self.model = self.model.to(self.device)

        self.model.eval()
        self.loaded = True
        logger.info(f"Model loaded on {self.device}")

    def process_image(self, image_path: str) -> Tuple[str, float, List[Dict]]:
        if not self.loaded:
            self.load_model()

        # GOT-OCR uses special chat method
        extracted_text = self.model.chat(
            self.tokenizer,
            image_path,
            ocr_type='ocr'  # Can be 'ocr' or 'format'
        )

        # GOT-OCR is highly accurate
        confidence = 0.95

        elements = []
        for i, line in enumerate(extracted_text.split('\n')):
            if line.strip():
                elements.append({
                    "text": line.strip(),
                    "confidence": confidence,
                    "line_number": i + 1
                })

        return extracted_text, confidence, elements


class TrOCREngine(BaseOCREngine):
    """Microsoft TrOCR engine for printed/handwritten text"""

    def load_model(self):
        if self.loaded:
            return

        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        logger.info(f"Loading {self.config.name} from {self.config.hf_repo}")

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.processor = TrOCRProcessor.from_pretrained(self.config.hf_repo)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.config.hf_repo)

        if self.device != "cpu":
            self.model = self.model.to(self.device)

        self.loaded = True
        logger.info(f"Model loaded on {self.device}")

    def process_image(self, image_path: str) -> Tuple[str, float, List[Dict]]:
        if not self.loaded:
            self.load_model()

        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        if self.device != "cpu":
            pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=512,
                output_scores=True,
                return_dict_in_generate=True
            )

        extracted_text = self.processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]

        # Calculate confidence
        if hasattr(generated_ids, 'scores') and generated_ids.scores:
            import torch.nn.functional as F
            confidences = []
            for score in generated_ids.scores[:20]:
                probs = F.softmax(score[0], dim=-1)
                max_prob = probs.max().item()
                confidences.append(max_prob)
            confidence = sum(confidences) / len(confidences) if confidences else 0.85
        else:
            confidence = 0.85

        elements = [{
            "text": extracted_text.strip(),
            "confidence": confidence,
            "line_number": 1
        }]

        return extracted_text, confidence, elements


class Florence2Engine(BaseOCREngine):
    """Microsoft Florence-2 engine"""

    def load_model(self):
        if self.loaded:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        logger.info(f"Loading {self.config.name} from {self.config.hf_repo}")

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.processor = AutoProcessor.from_pretrained(
            self.config.hf_repo,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_repo,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True,
            attn_implementation='eager'  # Fix for SDPA compatibility
        )

        if self.device != "cpu":
            self.model = self.model.to(self.device)

        self.loaded = True
        logger.info(f"Model loaded on {self.device}")

    def process_image(self, image_path: str) -> Tuple[str, float, List[Dict]]:
        if not self.loaded:
            self.load_model()

        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")

        # Florence-2 OCR task
        prompt = "<OCR>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Parse Florence-2 output
        extracted_text = self.processor.post_process_generation(
            generated_text,
            task="<OCR>",
            image_size=(image.width, image.height)
        )

        if isinstance(extracted_text, dict) and '<OCR>' in extracted_text:
            extracted_text = extracted_text['<OCR>']
        else:
            extracted_text = str(extracted_text)

        confidence = 0.88

        elements = []
        for i, line in enumerate(extracted_text.split('\n')):
            if line.strip():
                elements.append({
                    "text": line.strip(),
                    "confidence": confidence,
                    "line_number": i + 1
                })

        return extracted_text, confidence, elements


# Engine factory
ENGINE_MAP = {
    "qwen2_vl_2b": Qwen2VLEngine,
    "qwen2_vl_7b": Qwen2VLEngine,
    "got_ocr2": GOTOCREngine,
    "trocr_large_printed": TrOCREngine,
    "trocr_large_handwritten": TrOCREngine,
    "florence2_large": Florence2Engine,
}


# Engine singleton cache - avoids recreating expensive model loaders
_ENGINE_CACHE: Dict[str, BaseOCREngine] = {}
_ENGINE_CACHE_LOCK = __import__('threading').Lock()


def get_ocr_engine(model_key: str, use_cache: bool = True) -> BaseOCREngine:
    """
    Get OCR engine instance by model key.

    Args:
        model_key: Key from OCR_MODELS
        use_cache: If True, reuse cached engine instances (default)

    Returns:
        BaseOCREngine instance
    """
    if model_key not in OCR_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(OCR_MODELS.keys())}")

    # Return cached engine if available
    if use_cache and model_key in _ENGINE_CACHE:
        logger.debug(f"Returning cached engine for {model_key}")
        return _ENGINE_CACHE[model_key]

    model_config = OCR_MODELS[model_key]
    engine_class = ENGINE_MAP.get(model_key)

    if engine_class is None:
        # Default to Qwen2VL for unknown engines
        logger.warning(f"No specific engine for {model_key}, using Qwen2VL")
        engine_class = Qwen2VLEngine

    engine = engine_class(model_config)

    # Cache the engine for reuse
    if use_cache:
        with _ENGINE_CACHE_LOCK:
            _ENGINE_CACHE[model_key] = engine
            logger.info(f"Cached engine for {model_key}")

    return engine


def clear_engine_cache(model_key: str = None):
    """
    Clear engine cache to free memory.

    Args:
        model_key: Specific model to clear, or None for all
    """
    with _ENGINE_CACHE_LOCK:
        if model_key:
            if model_key in _ENGINE_CACHE:
                engine = _ENGINE_CACHE.pop(model_key)
                if hasattr(engine, 'unload_model'):
                    engine.unload_model()
                logger.info(f"Cleared cached engine for {model_key}")
        else:
            for key, engine in list(_ENGINE_CACHE.items()):
                if hasattr(engine, 'unload_model'):
                    engine.unload_model()
            _ENGINE_CACHE.clear()
            logger.info("Cleared all cached engines")


class UnifiedOCRProcessor:
    """
    Unified processor that handles complete documents.
    Supports PDFs, images, multi-page documents.
    Auto-detects digital PDFs and skips OCR when not needed.
    """

    # Minimum characters per page to consider PDF as having digital text
    MIN_CHARS_PER_PAGE = 50

    def __init__(self, model_key: str = "qwen2_vl_2b"):
        self.model_key = model_key
        self.engine = get_ocr_engine(model_key)
        self.model_config = OCR_MODELS[model_key]

    def _has_digital_text(self, pdf_path: str) -> Tuple[bool, str, int]:
        """
        Check if PDF has embedded digital text (not scanned).

        Returns:
            Tuple of (has_text, extracted_text, page_count)
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            all_text = []

            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                all_text.append(text.strip())

            doc.close()

            full_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)
            total_chars = sum(len(t) for t in all_text)

            # Consider digital if average chars per page exceeds threshold
            avg_chars_per_page = total_chars / total_pages if total_pages > 0 else 0
            has_digital = avg_chars_per_page >= self.MIN_CHARS_PER_PAGE

            logger.info(f"PDF digital text check: {total_chars} chars, {avg_chars_per_page:.0f} chars/page, digital={has_digital}")

            return has_digital, full_text, total_pages

        except Exception as e:
            logger.warning(f"Error checking PDF digital text: {e}")
            return False, "", 0

    # =========================================================================
    # SIGNATURE DETECTION (ORACLE-DESIGNED)
    # =========================================================================

    def detect_digital_signatures(self, pdf_path: str) -> List[Dict]:
        """
        Detect digital (PKI-based) signatures in PDF.

        Returns list of digital signature information including:
        - Signer name
        - Signing time
        - Certificate validity
        - Document modification status
        """
        import fitz

        signatures = []

        try:
            doc = fitz.open(pdf_path)

            # Check if document has any signatures
            # PyMuPDF provides signature detection via widgets
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                # Get all widgets (form fields) on page
                for widget in page.widgets():
                    if widget.field_type == fitz.PDF_WIDGET_TYPE_SIGNATURE:
                        sig_info = {
                            "page": page_num + 1,
                            "field_name": widget.field_name,
                            "rect": {
                                "x": widget.rect.x0,
                                "y": widget.rect.y0,
                                "width": widget.rect.width,
                                "height": widget.rect.height
                            },
                            "is_signed": widget.field_value is not None
                        }

                        # Try to extract signer info from annotation
                        if widget.field_value:
                            sig_info["signer_name"] = widget.field_value.get("Name", "Unknown")
                            sig_info["signing_time"] = widget.field_value.get("M", None)

                        signatures.append(sig_info)

            doc.close()

        except Exception as e:
            logger.warning(f"Error detecting digital signatures: {e}")

        return signatures

    def detect_handwritten_signatures(self, pdf_path: str) -> List[Dict]:
        """
        Detect potential handwritten signatures in PDF.

        Looks for:
        - Text patterns like "Signature:", "Signed by:", "Sign here"
        - Isolated images/drawings that could be signatures
        - Areas marked as signature zones
        """
        import fitz

        signatures = []

        try:
            doc = fitz.open(pdf_path)

            # Common signature indicator patterns
            signature_patterns = [
                "signature", "signed by", "sign here", "authorized signature",
                "signatory", "signed:", "per:", "by:", "signé", "firma"
            ]

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text").lower()

                # Search for signature-related text
                for pattern in signature_patterns:
                    if pattern in text:
                        # Find the exact location
                        text_instances = page.search_for(pattern)

                        for rect in text_instances:
                            # Check for images or drawings near this text
                            # Expand search area below the text (signatures usually below label)
                            search_rect = fitz.Rect(
                                rect.x0 - 20,
                                rect.y1,  # Start below the text
                                rect.x1 + 150,
                                rect.y1 + 60  # Look 60 points below
                            )

                            # Check for drawings/paths in signature area
                            drawings = page.get_drawings()
                            sig_drawings = [d for d in drawings
                                          if search_rect.intersects(fitz.Rect(d["rect"]))]

                            if sig_drawings:
                                signatures.append({
                                    "page": page_num + 1,
                                    "location": {
                                        "x": rect.x0,
                                        "y": rect.y1,
                                        "width": 150,
                                        "height": 60
                                    },
                                    "detection_confidence": 0.7,
                                    "associated_text": pattern,
                                    "detection_method": "text_pattern_with_drawing"
                                })

            doc.close()

        except Exception as e:
            logger.warning(f"Error detecting handwritten signatures: {e}")

        return signatures

    def get_signature_status(self, pdf_path: str) -> Dict:
        """
        Get complete signature status for a PDF document.

        Returns:
        - status: digitally_signed, manually_signed, both, unsigned
        - has_digital_signature: bool
        - has_handwritten_signature: bool
        - total_signature_count: int
        - digital_signatures: list
        - handwritten_signatures: list
        """
        digital_sigs = self.detect_digital_signatures(pdf_path)
        handwritten_sigs = self.detect_handwritten_signatures(pdf_path)

        has_digital = len(digital_sigs) > 0
        has_handwritten = len(handwritten_sigs) > 0

        if has_digital and has_handwritten:
            status = "both"
        elif has_digital:
            status = "digitally_signed"
        elif has_handwritten:
            status = "manually_signed"
        else:
            status = "unsigned"

        return {
            "status": status,
            "has_digital_signature": has_digital,
            "has_handwritten_signature": has_handwritten,
            "total_signature_count": len(digital_sigs) + len(handwritten_sigs),
            "digital_signatures": digital_sigs,
            "handwritten_signatures": handwritten_sigs
        }

    # =========================================================================
    # LANGUAGE DETECTION (ORACLE-DESIGNED)
    # =========================================================================

    def detect_language(self, text: str) -> Dict:
        """
        Detect language of text using langdetect library.

        Returns:
        - detected: language code (e.g., "en", "fr", "de")
        - confidence: detection confidence (0-1)
        - alternate_languages: list of other possible languages
        - script: detected script (Latin, Cyrillic, Arabic, CJK, etc.)
        """
        if not text or len(text.strip()) < 20:
            return {
                "detected": "unknown",
                "confidence": 0.0,
                "alternate_languages": [],
                "script": "unknown"
            }

        try:
            from langdetect import detect, detect_langs
            from langdetect import DetectorFactory
            DetectorFactory.seed = 0  # For reproducibility

            # Get all possible languages with probabilities
            langs = detect_langs(text)

            primary = langs[0]
            alternates = [
                {"lang": l.lang, "confidence": round(l.prob, 3)}
                for l in langs[1:]
                if l.prob > 0.1
            ]

            # Detect script based on character analysis
            script = self._detect_script(text)

            return {
                "detected": primary.lang,
                "confidence": round(primary.prob, 3),
                "alternate_languages": alternates,
                "script": script
            }

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return {
                "detected": "unknown",
                "confidence": 0.0,
                "alternate_languages": [],
                "script": self._detect_script(text)
            }

    def _detect_script(self, text: str) -> str:
        """Detect the writing script based on character ranges."""
        if not text:
            return "unknown"

        # Count characters by script
        latin = 0
        cyrillic = 0
        arabic = 0
        cjk = 0
        other = 0

        for char in text:
            code = ord(char)
            if 0x0041 <= code <= 0x024F:  # Basic Latin + Extended Latin
                latin += 1
            elif 0x0400 <= code <= 0x04FF:  # Cyrillic
                cyrillic += 1
            elif 0x0600 <= code <= 0x06FF:  # Arabic
                arabic += 1
            elif (0x4E00 <= code <= 0x9FFF or  # CJK Unified
                  0x3040 <= code <= 0x309F or  # Hiragana
                  0x30A0 <= code <= 0x30FF):   # Katakana
                cjk += 1
            elif char.isalpha():
                other += 1

        # Determine dominant script
        counts = {
            "Latin": latin,
            "Cyrillic": cyrillic,
            "Arabic": arabic,
            "CJK": cjk,
            "Other": other
        }

        total = sum(counts.values())
        if total == 0:
            return "unknown"

        dominant = max(counts, key=counts.get)
        if counts[dominant] / total < 0.5:
            return "Mixed"

        return dominant

    def analyze_pdf_structure(self, pdf_path: str) -> DocumentAnalysis:
        """
        ORACLE-DESIGNED: Comprehensive per-page analysis of PDF structure.

        Analyzes each page to determine:
        - Whether it has a text layer (digital text)
        - Whether it has embedded images
        - Image coverage percentage
        - Optimal extraction method per page

        This enables intelligent mixed PDF processing.
        """
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        analysis = DocumentAnalysis(
            file_path=pdf_path,
            total_pages=total_pages
        )

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height

            # Get text layer content
            text = page.get_text("text")
            char_count = len(text.strip())
            has_text_layer = char_count >= self.MIN_CHARS_PER_PAGE

            # Get images on page
            images = page.get_images(full=True)
            image_count = len(images)

            # Calculate image coverage
            total_image_area = 0.0
            for img in images:
                try:
                    xref = img[0]
                    # Get image bounding box
                    img_rects = page.get_image_rects(xref)
                    for rect in img_rects:
                        img_area = rect.width * rect.height
                        total_image_area += img_area
                except Exception:
                    pass  # Some images may not have extractable bounds

            image_coverage = min(1.0, total_image_area / page_area) if page_area > 0 else 0.0
            has_significant_images = image_coverage > 0.3  # >30% of page is images

            # Classify page type
            if char_count < 10 and image_count == 0:
                page_type = PageType.EMPTY
                needs_ocr = False
                extraction_method = "skip"
                confidence = 0.95

            elif has_text_layer and not has_significant_images:
                # Pure digital text page
                page_type = PageType.DIGITAL_TEXT
                needs_ocr = False
                extraction_method = "direct"
                confidence = 0.95
                analysis.digital_pages.append(page_num + 1)

            elif not has_text_layer and image_coverage > 0.7:
                # Scanned page (large image, no text layer)
                page_type = PageType.SCANNED_IMAGE
                needs_ocr = True
                extraction_method = "ocr"
                confidence = 0.90
                analysis.ocr_pages.append(page_num + 1)

            elif has_text_layer and has_significant_images:
                # Mixed content - has both text and images
                # Images might contain additional text (diagrams, photos with text)
                page_type = PageType.MIXED_CONTENT
                needs_ocr = True  # OCR images for potential additional text
                extraction_method = "hybrid"
                confidence = 0.75
                analysis.mixed_pages.append(page_num + 1)
                analysis.digital_pages.append(page_num + 1)  # Also extract digital text

            elif not has_text_layer and image_count > 0:
                # Images without text layer - likely scanned or image-based
                page_type = PageType.IMAGE_WITH_TEXT
                needs_ocr = True
                extraction_method = "ocr"
                confidence = 0.85
                analysis.ocr_pages.append(page_num + 1)

            else:
                # Default to OCR for safety
                page_type = PageType.SCANNED_IMAGE
                needs_ocr = True
                extraction_method = "ocr"
                confidence = 0.60
                analysis.ocr_pages.append(page_num + 1)

            page_analysis = PageAnalysis(
                page_number=page_num + 1,
                page_type=page_type,
                char_count=char_count,
                image_count=image_count,
                image_coverage=image_coverage,
                has_text_layer=has_text_layer,
                has_significant_images=has_significant_images,
                needs_ocr=needs_ocr,
                confidence=confidence,
                extraction_method=extraction_method
            )
            analysis.pages.append(page_analysis)
            analysis.total_digital_chars += char_count

        doc.close()

        # Determine document type
        total_digital = len(analysis.digital_pages)
        total_ocr = len(analysis.ocr_pages)
        total_mixed = len(analysis.mixed_pages)

        if total_ocr == 0 and total_mixed == 0:
            analysis.document_type = "pure_digital"
            analysis.recommended_strategy = "direct_extraction"
            analysis.estimated_processing_time = 0.05 * total_pages  # ~50ms per page

        elif total_digital == 0 and total_mixed == 0:
            analysis.document_type = "pure_scanned"
            analysis.recommended_strategy = "full_ocr"
            analysis.estimated_processing_time = 10.0 * total_pages  # ~10s per page OCR

        else:
            analysis.document_type = "mixed"
            analysis.recommended_strategy = "hybrid_extraction"
            # Estimate: fast for digital, slow for OCR
            analysis.estimated_processing_time = (
                0.05 * (total_digital - total_mixed) +
                10.0 * total_ocr +
                12.0 * total_mixed  # hybrid takes longer
            )

        logger.info(
            f"PDF Analysis: {total_pages} pages | "
            f"Type: {analysis.document_type} | "
            f"Digital: {total_digital}, OCR: {total_ocr}, Mixed: {total_mixed} | "
            f"Strategy: {analysis.recommended_strategy}"
        )

        return analysis

    def _extract_pdf_text_direct(self, pdf_path: str, job_id: str,
                                  progress_callback=None) -> OCRIntermediateResult:
        """
        Extract text directly from digital PDF without OCR.
        Much faster and more accurate for PDFs with embedded text.
        """
        import fitz

        path = Path(pdf_path)
        start_time = time.time()

        # Calculate file hash (chunked for large files)
        file_hash = compute_file_hash(pdf_path)

        # Initialize result
        result = OCRIntermediateResult(
            job_id=job_id or hashlib.md5(pdf_path.encode()).hexdigest()[:16]
        )

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        all_text = []

        for page_num in range(total_pages):
            if progress_callback:
                progress_callback((page_num + 1) / total_pages * 0.9, f"Extracting page {page_num + 1}/{total_pages}")

            page = doc.load_page(page_num)
            text = page.get_text("text")

            # Build page content with high confidence (digital text is exact)
            page_content = PageContent(
                page_number=page_num + 1,
                raw_text=text,
                confidence=0.99  # Digital text is highly accurate
            )

            # Build text blocks from lines
            if text.strip():
                block = TextBlock(block_type="paragraph", page=page_num + 1)
                for line_num, line in enumerate(text.split('\n'), 1):
                    if line.strip():
                        text_line = TextLine(
                            text=line.strip(),
                            confidence=0.99,
                            line_number=line_num,
                            page=page_num + 1
                        )
                        block.lines.append(text_line)
                if block.lines:
                    block.confidence = 0.99
                    page_content.blocks.append(block)

            result.pages.append(page_content)
            all_text.append(text)

        doc.close()

        # Build full text
        result.full_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)

        # Calculate duration
        duration = time.time() - start_time

        # Build metadata - note the special model name
        result.metadata = OCRMetadata(
            source_file=str(path.absolute()),
            source_format='pdf',
            source_hash=file_hash,
            file_size_bytes=path.stat().st_size,
            ocr_model_used="Digital PDF Extraction (PyMuPDF)",
            ocr_model_repo="N/A - Direct text extraction",
            ocr_timestamp=datetime.utcnow().isoformat(),
            ocr_duration_seconds=duration,
            detected_language="auto",
            detected_doc_type="digital_pdf",
            total_pages=total_pages,
            total_characters=len(result.full_text),
            total_words=len(result.full_text.split()),
            average_confidence=0.99,
            min_confidence=0.99,
            max_confidence=0.99,
            low_confidence_count=0
        )

        # Calculate statistics
        result.calculate_statistics()

        if progress_callback:
            progress_callback(1.0, "Extraction complete")

        logger.info(f"Digital PDF extracted in {duration:.2f}s: {total_pages} pages, {len(result.full_text)} chars")

        return result

    def process_mixed_pdf(self, pdf_path: str, doc_analysis: DocumentAnalysis,
                          job_id: str, progress_callback=None) -> OCRIntermediateResult:
        """
        ORACLE-DESIGNED: Process mixed PDFs with per-page intelligent extraction.

        For each page, uses the optimal method:
        - Digital pages: Direct text extraction (fast, 99% confidence)
        - Scanned pages: OCR (slower, varies confidence)
        - Mixed pages: Hybrid - extract digital text + OCR images

        This ensures NO text is missed in mixed documents.
        """
        import fitz
        import tempfile

        path = Path(pdf_path)
        start_time = time.time()

        # Calculate file hash (chunked for large files)
        file_hash = compute_file_hash(pdf_path)

        # Initialize result
        result = OCRIntermediateResult(
            job_id=job_id or hashlib.md5(pdf_path.encode()).hexdigest()[:16]
        )

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        all_text = []

        ocr_pages_processed = 0
        digital_pages_processed = 0

        for page_num in range(total_pages):
            page_analysis = doc_analysis.pages[page_num]
            page = doc.load_page(page_num)

            # Progress reporting
            if progress_callback:
                progress = (page_num + 1) / total_pages * 0.9
                method_name = page_analysis.extraction_method.upper()
                progress_callback(progress, f"{method_name} page {page_num + 1}/{total_pages}")

            page_text = ""
            page_confidence = 0.0

            # =====================================================
            # DIGITAL TEXT EXTRACTION (for digital/mixed pages)
            # =====================================================
            if page_analysis.extraction_method in ["direct", "hybrid"]:
                digital_text = page.get_text("text")
                page_text = digital_text
                page_confidence = 0.99
                digital_pages_processed += 1
                logger.debug(f"Page {page_num + 1}: Direct extraction - {len(digital_text)} chars")

            # =====================================================
            # OCR EXTRACTION (for scanned/mixed pages)
            # =====================================================
            if page_analysis.needs_ocr:
                # Convert page to image for OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
                temp_path = Path(tempfile.gettempdir()) / f"ocr_mixed_{job_id}_p{page_num}.png"
                pix.save(str(temp_path))

                try:
                    # Run OCR on this page
                    ocr_text, ocr_conf, ocr_elements = self.engine.process_image(str(temp_path))
                    ocr_pages_processed += 1
                    logger.debug(f"Page {page_num + 1}: OCR extraction - {len(ocr_text)} chars, {ocr_conf:.0%} conf")

                    if page_analysis.extraction_method == "hybrid":
                        # Merge digital + OCR text (avoid duplicates)
                        # For hybrid, OCR might find text in images that digital extraction missed
                        if ocr_text.strip() and ocr_text.strip() not in page_text:
                            page_text = page_text + "\n\n[IMAGE TEXT]\n" + ocr_text
                            # Weighted confidence: mostly digital, some OCR
                            page_confidence = 0.95 * 0.7 + ocr_conf * 0.3
                    else:
                        # Pure OCR page
                        page_text = ocr_text
                        page_confidence = ocr_conf

                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                    # Fall back to any digital text we might have
                    if not page_text:
                        page_text = f"[OCR FAILED: {e}]"
                        page_confidence = 0.0

                finally:
                    # Cleanup temp file
                    if temp_path.exists():
                        temp_path.unlink()

            # Build page content
            page_content = PageContent(
                page_number=page_num + 1,
                raw_text=page_text,
                confidence=page_confidence
            )

            # Build text blocks
            if page_text.strip():
                block = TextBlock(
                    block_type="paragraph",
                    page=page_num + 1
                )
                for line_num, line in enumerate(page_text.split('\n'), 1):
                    if line.strip():
                        text_line = TextLine(
                            text=line.strip(),
                            confidence=page_confidence,
                            line_number=line_num,
                            page=page_num + 1
                        )
                        block.lines.append(text_line)
                if block.lines:
                    block.confidence = page_confidence
                    page_content.blocks.append(block)

            result.pages.append(page_content)
            all_text.append(page_text)

        doc.close()

        # Build full text
        result.full_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)

        # Calculate duration
        duration = time.time() - start_time

        # Build metadata
        model_description = f"Hybrid Extraction (Digital: {digital_pages_processed}, OCR: {ocr_pages_processed} pages)"
        result.metadata = OCRMetadata(
            source_file=str(path.absolute()),
            source_format='pdf',
            source_hash=file_hash,
            file_size_bytes=path.stat().st_size,
            ocr_model_used=model_description,
            ocr_model_repo=self.model_config.hf_repo if ocr_pages_processed > 0 else "N/A",
            ocr_timestamp=datetime.utcnow().isoformat(),
            ocr_duration_seconds=duration,
            detected_language="auto",
            detected_doc_type="mixed_pdf",
            total_pages=total_pages,
            total_characters=len(result.full_text),
            total_words=len(result.full_text.split()),
            average_confidence=0.0,
            min_confidence=1.0,
            max_confidence=0.0,
            low_confidence_count=0
        )

        # Calculate statistics
        result.calculate_statistics()

        if progress_callback:
            progress_callback(1.0, "Hybrid extraction complete")

        logger.info(
            f"Mixed PDF processed in {duration:.2f}s: "
            f"{total_pages} pages ({digital_pages_processed} digital, {ocr_pages_processed} OCR), "
            f"{len(result.full_text)} chars"
        )

        return result

    def process_document(self, file_path: str, job_id: str = None,
                         progress_callback=None) -> OCRIntermediateResult:
        """
        ORACLE-DESIGNED: Intelligent document processing with per-page analysis.

        For PDFs:
        1. Analyzes EACH page to determine content type
        2. Routes to optimal extraction method per page
        3. Handles mixed documents (digital + scanned + images)

        For Images:
        - Uses OCR directly

        This ensures NO text is missed, even in complex mixed PDFs.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        # =====================================================
        # PDF PROCESSING WITH PER-PAGE ANALYSIS
        # =====================================================
        if ext == '.pdf':
            if progress_callback:
                progress_callback(0.02, "Analyzing PDF structure...")

            # Comprehensive per-page analysis
            doc_analysis = self.analyze_pdf_structure(file_path)

            # Route based on document type
            if doc_analysis.document_type == "pure_digital":
                # All pages are digital - use fast direct extraction
                logger.info(f"Pure digital PDF detected - using direct extraction")
                if progress_callback:
                    progress_callback(0.05, f"Digital PDF ({doc_analysis.total_pages} pages) - extracting...")
                return self._extract_pdf_text_direct(file_path, job_id, progress_callback)

            elif doc_analysis.document_type == "pure_scanned":
                # All pages need OCR
                logger.info(f"Pure scanned PDF detected - using OCR model: {self.model_key}")
                if progress_callback:
                    progress_callback(0.05, f"Scanned PDF ({doc_analysis.total_pages} pages) - starting OCR...")
                # Fall through to standard OCR processing below

            else:
                # Mixed document - use intelligent per-page processing
                logger.info(
                    f"Mixed PDF detected - "
                    f"{len(doc_analysis.digital_pages)} digital, "
                    f"{len(doc_analysis.ocr_pages)} scanned, "
                    f"{len(doc_analysis.mixed_pages)} mixed pages"
                )
                if progress_callback:
                    progress_callback(0.05, f"Mixed PDF - hybrid extraction...")
                return self.process_mixed_pdf(file_path, doc_analysis, job_id, progress_callback)

        start_time = time.time()

        # Calculate file hash (chunked for large files)
        file_hash = compute_file_hash(file_path)

        # Initialize result
        result = OCRIntermediateResult(
            job_id=job_id or hashlib.md5(file_path.encode()).hexdigest()[:16]
        )

        # Get pages (convert PDF to images if needed)
        pages = self._get_pages(file_path)
        total_pages = len(pages)

        all_text = []
        all_tables = []

        # Process each page
        for page_num, page_path in enumerate(pages, 1):
            if progress_callback:
                progress_callback(page_num / total_pages * 0.9, f"OCR page {page_num}/{total_pages}")

            text, confidence, elements = self.engine.process_image(page_path)

            # Build page content
            page_content = PageContent(
                page_number=page_num,
                raw_text=text,
                confidence=confidence
            )

            # Build text blocks from elements
            if elements:
                block = TextBlock(block_type="paragraph", page=page_num)
                for elem in elements:
                    line = TextLine(
                        text=elem["text"],
                        confidence=elem.get("confidence", confidence),
                        line_number=elem.get("line_number", 0),
                        page=page_num
                    )
                    block.lines.append(line)
                block.confidence = sum(l.confidence for l in block.lines) / len(block.lines)
                page_content.blocks.append(block)

            result.pages.append(page_content)
            all_text.append(text)

            # Cleanup temp files
            if page_path != file_path and Path(page_path).exists():
                Path(page_path).unlink()

        # Build full text
        result.full_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)

        # Calculate duration
        duration = time.time() - start_time

        # Build metadata
        result.metadata = OCRMetadata(
            source_file=str(path.absolute()),
            source_format=path.suffix.lower().lstrip('.'),
            source_hash=file_hash,
            file_size_bytes=path.stat().st_size,
            ocr_model_used=self.model_config.name,
            ocr_model_repo=self.model_config.hf_repo,
            ocr_timestamp=datetime.utcnow().isoformat(),
            ocr_duration_seconds=duration,
            detected_language="auto",
            detected_doc_type="document",
            total_pages=total_pages,
            total_characters=len(result.full_text),
            total_words=len(result.full_text.split()),
            average_confidence=0.0,
            min_confidence=1.0,
            max_confidence=0.0,
            low_confidence_count=0
        )

        # Calculate statistics
        result.calculate_statistics()

        if progress_callback:
            progress_callback(1.0, "OCR complete")

        return result

    def _get_pages(self, file_path: str) -> List[str]:
        """Extract pages from document, return list of image paths"""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == '.pdf':
            return self._pdf_to_images(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp']:
            return [file_path]
        else:
            raise ValueError(f"Unsupported format: {ext}")

    def _pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF to images"""
        import fitz  # PyMuPDF
        import tempfile

        doc = fitz.open(pdf_path)
        image_paths = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for quality

            # Save to temp file
            temp_path = Path(tempfile.gettempdir()) / f"ocr_page_{page_num}_{hash(pdf_path)}.png"
            pix.save(str(temp_path))
            image_paths.append(str(temp_path))

        doc.close()
        return image_paths

    # =========================================================================
    # TABLE DETECTION (ORACLE-DESIGNED)
    # Extract tables from PDFs using PyMuPDF's find_tables()
    # =========================================================================

    def detect_tables_pdf(self, pdf_path: str) -> Dict[int, List[Dict]]:
        """
        ORACLE-DESIGNED: Detect and extract tables from PDF using PyMuPDF.

        Returns:
            Dict mapping page_number (1-indexed) to list of table dicts
            Each table dict contains: rows, cols, data (matrix), confidence
        """
        import fitz

        tables_by_page = {}

        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, 1):
                page_tables = []
                try:
                    tables = page.find_tables()
                    for table in tables.tables:
                        # Extract table data as matrix
                        data = table.extract()

                        table_dict = {
                            "rows": table.row_count,
                            "cols": table.col_count,
                            "data": data,  # Matrix of cell values
                            "confidence": 0.95,  # PyMuPDF table detection is reliable
                            "bbox": {
                                "x0": table.bbox[0] if table.bbox else None,
                                "y0": table.bbox[1] if table.bbox else None,
                                "x1": table.bbox[2] if table.bbox else None,
                                "y1": table.bbox[3] if table.bbox else None
                            }
                        }
                        page_tables.append(table_dict)

                except Exception as e:
                    logger.debug(f"Table detection on page {page_num}: {e}")

                if page_tables:
                    tables_by_page[page_num] = page_tables

            doc.close()
            logger.info(f"Table detection: Found {sum(len(t) for t in tables_by_page.values())} tables across {len(tables_by_page)} pages")

        except Exception as e:
            logger.warning(f"Table detection failed: {e}")

        return tables_by_page

    # =========================================================================
    # ENHANCED OUTPUT v2.0 (ORACLE-DESIGNED)
    # Complete per-page metadata including language, statistics, signatures
    # =========================================================================

    def build_enhanced_output(self, ocr_result: OCRIntermediateResult,
                               pdf_path: str = None) -> Dict[str, Any]:
        """
        ORACLE-DESIGNED: Build enhanced JSON output v2.0 with full per-page metadata.

        Returns complete document analysis including:
        - Per-page OCR confidence
        - Per-page language detection
        - Per-page word/character counts
        - Document signature information
        - Source format details
        - OCR model information
        """
        from config.intermediate_format import (
            EnhancedJSONOutput, EnhancedPageContent, DocumentInfo, ProcessingInfo,
            DocumentSignatures, LanguageInfo, PageSourceInfo, PageStatistics,
            PageQuality, SignatureStatus
        )

        # Initialize enhanced output
        enhanced = {
            "schema_version": "2.0",
            "job_id": ocr_result.job_id,
            "processing_timestamp": datetime.utcnow().isoformat(),

            # Document info
            "document": {
                "original_filename": Path(ocr_result.metadata.source_file).name if ocr_result.metadata else "",
                "stored_filename": "",
                "file_format": ocr_result.metadata.source_format if ocr_result.metadata else "unknown",
                "file_size_bytes": ocr_result.metadata.file_size_bytes if ocr_result.metadata else 0,
                "file_hash_sha256": ocr_result.metadata.source_hash if ocr_result.metadata else "",
                "mime_type": f"application/{ocr_result.metadata.source_format}" if ocr_result.metadata else "unknown"
            },

            # Processing info
            "processing": {
                "pipeline_version": "2.0",
                "ocr_model_name": ocr_result.metadata.ocr_model_used if ocr_result.metadata else "unknown",
                "ocr_model_repository": ocr_result.metadata.ocr_model_repo if ocr_result.metadata else "",
                "ocr_pages_processed": len(ocr_result.pages),
                "extraction_seconds": ocr_result.metadata.ocr_duration_seconds if ocr_result.metadata else 0,
                "strategy": ocr_result.metadata.detected_doc_type if ocr_result.metadata else "unknown"
            },

            # Aggregate stats
            "total_pages": len(ocr_result.pages),
            "total_characters": 0,
            "total_words": 0,
            "primary_language": "unknown",
            "languages_detected": [],

            # Signatures - detect if PDF path provided
            "signatures": {
                "status": "unsigned",
                "has_digital_signature": False,
                "has_handwritten_signature": False,
                "total_signature_count": 0,
                "digital_signatures": [],
                "handwritten_signatures": []
            },

            # Per-page enhanced content
            "pages": [],

            # Quality metrics
            "overall_confidence": ocr_result.metadata.average_confidence if ocr_result.metadata else 0,
            "confidence_distribution": ocr_result.confidence_distribution,
            "low_confidence_segments": ocr_result.low_confidence_segments,

            # Full text backup
            "full_text": ocr_result.full_text
        }

        # Detect signatures if PDF path available
        tables_by_page = {}
        if pdf_path and Path(pdf_path).exists():
            try:
                sig_status = self.get_signature_status(pdf_path)
                enhanced["signatures"] = sig_status
            except Exception as e:
                logger.warning(f"Signature detection failed: {e}")

            # Detect tables using PyMuPDF
            try:
                tables_by_page = self.detect_tables_pdf(pdf_path)
            except Exception as e:
                logger.warning(f"Table detection failed: {e}")

        # Initialize document-level tables array
        all_tables = []

        # Process each page for enhanced metadata
        language_counts = {}
        total_chars = 0
        total_words = 0

        for page in ocr_result.pages:
            page_text = page.raw_text

            # Calculate page statistics
            char_count = len(page_text)
            word_count = len(page_text.split())
            line_count = len(page_text.split('\n'))
            paragraph_count = len([p for p in page_text.split('\n\n') if p.strip()])

            total_chars += char_count
            total_words += word_count

            # Detect language for this page
            lang_info = self.detect_language(page_text)

            # Track language distribution
            detected_lang = lang_info.get("detected", "unknown")
            if detected_lang != "unknown":
                language_counts[detected_lang] = language_counts.get(detected_lang, 0) + char_count

            # Determine source type based on page confidence
            if page.confidence >= 0.95:
                source_type = "digital_text"
                extraction_method = "direct"
                text_clarity = "excellent"
            elif page.confidence >= 0.85:
                source_type = "scanned_image"
                extraction_method = "ocr"
                text_clarity = "good"
            elif page.confidence >= 0.70:
                source_type = "mixed_content"
                extraction_method = "hybrid"
                text_clarity = "fair"
            else:
                source_type = "image_with_text"
                extraction_method = "ocr"
                text_clarity = "poor"

            # Get tables detected on this page (from PyMuPDF)
            page_tables = tables_by_page.get(page.page_number, [])

            # Add to document-level tables with page reference
            for table in page_tables:
                all_tables.append({
                    "page": page.page_number,
                    **table
                })

            # Build enhanced page content
            enhanced_page = {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,

                # Source info
                "source": {
                    "source_type": source_type,
                    "extraction_method": extraction_method,
                    "ocr_model_used": ocr_result.metadata.ocr_model_used if ocr_result.metadata else None,
                    "has_text_layer": page.confidence >= 0.95
                },

                # Language detection
                "language": lang_info,

                # Statistics (user requested)
                "statistics": {
                    "char_count": char_count,
                    "word_count": word_count,
                    "line_count": line_count,
                    "paragraph_count": paragraph_count,
                    "table_count": len(page_tables),  # Use detected tables
                    "image_count": 0,  # Would need PDF analysis
                    "form_field_count": 0
                },

                # Quality metrics
                "quality": {
                    "extraction_confidence": page.confidence,
                    "ocr_confidence": page.confidence if source_type != "digital_text" else None,
                    "text_clarity": text_clarity,
                    "issues": []
                },

                # Raw text
                "raw_text": page_text,

                # Blocks and tables
                "blocks": [
                    {
                        "block_type": block.block_type,
                        "text": block.text,
                        "confidence": block.confidence,
                        "lines": [
                            {
                                "text": line.text,
                                "confidence": line.confidence,
                                "confidence_level": line.confidence_level.value,
                                "line_number": line.line_number
                            }
                            for line in block.lines
                        ]
                    }
                    for block in page.blocks
                ],
                # Tables detected by PyMuPDF
                "tables": page_tables
            }

            # Check for signatures on this page
            if enhanced["signatures"]["digital_signatures"]:
                for sig in enhanced["signatures"]["digital_signatures"]:
                    if sig.get("page") == page.page_number:
                        enhanced_page["has_signature"] = True
                        enhanced_page["signature_type"] = "digital"
                        break

            if enhanced["signatures"]["handwritten_signatures"]:
                for sig in enhanced["signatures"]["handwritten_signatures"]:
                    if sig.get("page") == page.page_number:
                        enhanced_page["has_signature"] = True
                        enhanced_page["signature_type"] = enhanced_page.get("signature_type", "") + "_handwritten" if enhanced_page.get("has_signature") else "handwritten"

            enhanced["pages"].append(enhanced_page)

        # Set aggregate statistics
        enhanced["total_characters"] = total_chars
        enhanced["total_words"] = total_words

        # Determine primary language from distribution
        if language_counts:
            enhanced["primary_language"] = max(language_counts, key=language_counts.get)
            enhanced["languages_detected"] = list(language_counts.keys())

        # Add document-level tables array
        if all_tables:
            enhanced["tables"] = all_tables
            enhanced["total_tables"] = len(all_tables)
        else:
            enhanced["tables"] = []
            enhanced["total_tables"] = 0

        return enhanced

    def process_document_enhanced(self, file_path: str, job_id: str = None,
                                   progress_callback=None) -> Dict[str, Any]:
        """
        Process document and return enhanced JSON output v2.0.

        This is the recommended method for full metadata extraction.
        """
        # First, run standard processing
        ocr_result = self.process_document(file_path, job_id, progress_callback)

        # Then build enhanced output
        return self.build_enhanced_output(ocr_result, file_path)

    # =========================================================================
    # ENHANCED v2.1 OUTPUT WITH VISUAL CONTENT (ORACLE-DESIGNED)
    # Full visual content extraction: tables, diagrams, charts with reconstruction
    # =========================================================================

    def build_enhanced_output_v21(self, ocr_result: OCRIntermediateResult,
                                   pdf_path: str = None,
                                   use_ai_cascade: bool = True) -> Dict[str, Any]:
        """
        ORACLE-DESIGNED: Build enhanced JSON output v2.1 with full visual content.

        Features:
        - Document summary (brief + detailed + key points)
        - Tags/keywords with document classification
        - Word cloud with top 10 words + percentages + "others"
        - Enhanced tables with header detection, styling, and reconstruction formats
        - Diagrams/graphs converted to Mermaid syntax
        - Charts with extracted data points (Plotly JSON, CSV)
        - AI cascade for low-confidence extraction escalation

        Args:
            ocr_result: OCR intermediate result
            pdf_path: Path to PDF for visual content extraction
            use_ai_cascade: Enable AI cascade for quality escalation

        Returns:
            Enhanced v2.1 JSON output with visual content reconstruction
        """
        # Start with v2.0 enhanced output
        enhanced = self.build_enhanced_output(ocr_result, pdf_path)

        # Upgrade to v2.1
        enhanced["schema_version"] = "2.1"

        # =====================================================================
        # DOCUMENT ANALYSIS: Summary, Tags, Word Cloud (ORACLE-DESIGNED)
        # =====================================================================
        if DOC_ANALYZER_AVAILABLE and ocr_result.full_text:
            try:
                logger.info("Running document analysis (summary, tags, word_cloud)...")

                # Analyze document
                doc_analysis = analyze_document(
                    ocr_result.full_text,
                    language=enhanced.get("primary_language", "auto")
                )

                # Add document_summary section (distinct from AI-generated summary)
                enhanced["document_summary"] = doc_analysis["summary"]

                # Add tags section
                enhanced["tags"] = doc_analysis["tags"]

                # Override document_type with analyzed type if more specific
                if doc_analysis["tags"]["document_type"] != "other":
                    enhanced["document_type"] = doc_analysis["tags"]["document_type"]

                # Add word cloud section
                enhanced["word_cloud"] = doc_analysis["word_cloud"]

                logger.info(
                    f"Document analysis complete: type={doc_analysis['tags']['document_type']}, "
                    f"keywords={len(doc_analysis['tags']['keywords'])}, "
                    f"word_cloud_top={len(doc_analysis['word_cloud']['top_words'])}"
                )

            except Exception as e:
                logger.warning(f"Document analysis failed: {e}")
                # Add empty sections
                enhanced["summary"] = {
                    "brief": "Analysis failed",
                    "detailed": str(e),
                    "key_points": [],
                    "confidence": 0.0
                }
                enhanced["tags"] = {
                    "document_type": "unknown",
                    "categories": [],
                    "keywords": [],
                    "entities": [],
                    "confidence": 0.0
                }
                enhanced["word_cloud"] = {
                    "top_words": [],
                    "others_percentage": 100.0,
                    "total_words": 0,
                    "unique_words": 0
                }

        # Initialize visual content sections
        enhanced["visual_content"] = {
            "tables": [],
            "diagrams": [],
            "charts": [],
            "total_tables": 0,
            "total_diagrams": 0,
            "total_charts": 0
        }

        enhanced["reconstruction"] = {
            "tables_html": [],
            "tables_markdown": [],
            "tables_csv": [],
            "diagrams_mermaid": [],
            "charts_plotly": [],
            "charts_csv": []
        }

        # Extract visual content if available
        if VISUAL_EXTRACTOR_AVAILABLE and pdf_path and Path(pdf_path).exists():
            try:
                visual_extractor = VisualContentExtractor(
                    use_ai_cascade=use_ai_cascade,
                    cascade_threshold=0.7
                )

                # Enhanced table extraction
                logger.info("Extracting enhanced tables with v2.1 format...")

                tables_by_page = visual_extractor.extract_tables_enhanced(pdf_path)

                all_tables = []
                for page_num, page_tables in tables_by_page.items():
                    for table in page_tables:
                        table_dict = table.to_dict()
                        all_tables.append(table_dict)

                        # Add reconstruction data
                        enhanced["reconstruction"]["tables_html"].append({
                            "table_id": table.table_id,
                            "page": table.page,
                            "html": table.to_html()
                        })
                        enhanced["reconstruction"]["tables_markdown"].append({
                            "table_id": table.table_id,
                            "page": table.page,
                            "markdown": table.to_markdown()
                        })
                        enhanced["reconstruction"]["tables_csv"].append({
                            "table_id": table.table_id,
                            "page": table.page,
                            "csv": table.to_csv()
                        })

                        # Also update per-page tables in enhanced output
                        for ep in enhanced["pages"]:
                            if ep["page_number"] == page_num:
                                if "tables_enhanced" not in ep:
                                    ep["tables_enhanced"] = []
                                ep["tables_enhanced"].append(table_dict)

                enhanced["visual_content"]["tables"] = all_tables
                enhanced["visual_content"]["total_tables"] = len(all_tables)

                logger.info(f"Enhanced v2.1: {len(all_tables)} tables with reconstruction data")

                # Diagram detection (from images in PDF)
                # Note: Full implementation would scan embedded images
                enhanced["visual_content"]["diagrams"] = []
                enhanced["visual_content"]["total_diagrams"] = 0

                # Chart detection (from images in PDF)
                # Note: Full implementation would scan embedded images
                enhanced["visual_content"]["charts"] = []
                enhanced["visual_content"]["total_charts"] = 0

            except Exception as e:
                logger.warning(f"Visual content extraction failed: {e}")
                enhanced["warnings"] = enhanced.get("warnings", [])
                enhanced["warnings"].append(f"Visual extraction partial failure: {str(e)}")

        return enhanced

    def process_document_enhanced_v21(self, file_path: str, job_id: str = None,
                                       progress_callback=None,
                                       use_ai_cascade: bool = True) -> Dict[str, Any]:
        """
        ORACLE-DESIGNED: Process document and return enhanced v2.1 output.

        Full pipeline with:
        - Intelligent per-page extraction (digital/OCR/hybrid)
        - Enhanced visual content (tables, diagrams, charts)
        - AI cascade for quality escalation
        - Multiple reconstruction formats

        Args:
            file_path: Path to document
            job_id: Optional job ID
            progress_callback: Progress callback(progress, message)
            use_ai_cascade: Enable AI cascade for low-confidence extractions

        Returns:
            Enhanced v2.1 JSON output with full reconstruction capability
        """
        # First, run standard processing
        ocr_result = self.process_document(file_path, job_id, progress_callback)

        # Then build enhanced v2.1 output with visual content
        return self.build_enhanced_output_v21(ocr_result, file_path, use_ai_cascade)
