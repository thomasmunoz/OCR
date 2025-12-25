"""
ORACLE-DESIGNED INTERMEDIATE FORMAT
====================================
Optimized format between OCR and Organization stages.
Preserves all data with confidence scores.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import json
import hashlib

class ConfidenceLevel(Enum):
    """Confidence classification"""
    CERTAIN = "certain"      # 95-100%
    HIGH = "high"            # 80-94%
    MEDIUM = "medium"        # 60-79%
    LOW = "low"              # 40-59%
    UNCERTAIN = "uncertain"  # <40%

@dataclass
class BoundingBox:
    """Position of text element"""
    x: float
    y: float
    width: float
    height: float
    page: int = 1

@dataclass
class TextElement:
    """Individual text element with confidence"""
    text: str
    confidence: float  # 0.0 - 1.0
    confidence_level: ConfidenceLevel = field(default=ConfidenceLevel.MEDIUM)
    bbox: Optional[BoundingBox] = None
    font_size: Optional[float] = None
    is_bold: bool = False
    is_italic: bool = False
    language_detected: Optional[str] = None

    def __post_init__(self):
        # Auto-classify confidence level
        if self.confidence >= 0.95:
            self.confidence_level = ConfidenceLevel.CERTAIN
        elif self.confidence >= 0.80:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence >= 0.60:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.40:
            self.confidence_level = ConfidenceLevel.LOW
        else:
            self.confidence_level = ConfidenceLevel.UNCERTAIN

@dataclass
class TextLine:
    """Line of text with aggregated confidence"""
    text: str
    elements: List[TextElement] = field(default_factory=list)
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = field(default=ConfidenceLevel.MEDIUM)
    bbox: Optional[BoundingBox] = None
    line_number: int = 0
    page: int = 1

    def __post_init__(self):
        if self.elements and self.confidence == 0.0:
            # Calculate average confidence from elements
            self.confidence = sum(e.confidence for e in self.elements) / len(self.elements)

        # Classify
        if self.confidence >= 0.95:
            self.confidence_level = ConfidenceLevel.CERTAIN
        elif self.confidence >= 0.80:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence >= 0.60:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.40:
            self.confidence_level = ConfidenceLevel.LOW
        else:
            self.confidence_level = ConfidenceLevel.UNCERTAIN

@dataclass
class TextBlock:
    """Block/paragraph of text"""
    lines: List[TextLine] = field(default_factory=list)
    block_type: str = "paragraph"  # paragraph, heading, list, table_cell, etc.
    confidence: float = 0.0
    bbox: Optional[BoundingBox] = None
    page: int = 1

    def __post_init__(self):
        if self.lines and self.confidence == 0.0:
            self.confidence = sum(l.confidence for l in self.lines) / len(self.lines)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

@dataclass
class TableCell:
    """Table cell with position"""
    text: str
    confidence: float
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False

@dataclass
class Table:
    """Extracted table structure"""
    cells: List[TableCell] = field(default_factory=list)
    rows: int = 0
    cols: int = 0
    confidence: float = 0.0
    bbox: Optional[BoundingBox] = None
    page: int = 1

    def to_matrix(self) -> List[List[str]]:
        """Convert to 2D matrix"""
        if not self.cells:
            return []
        matrix = [["" for _ in range(self.cols)] for _ in range(self.rows)]
        for cell in self.cells:
            if 0 <= cell.row < self.rows and 0 <= cell.col < self.cols:
                matrix[cell.row][cell.col] = cell.text
        return matrix

@dataclass
class PageContent:
    """Content of a single page"""
    page_number: int
    blocks: List[TextBlock] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 0.0
    width: Optional[float] = None
    height: Optional[float] = None

    def __post_init__(self):
        all_confidences = []
        if self.blocks:
            all_confidences.extend(b.confidence for b in self.blocks)
        if self.tables:
            all_confidences.extend(t.confidence for t in self.tables)
        if all_confidences:
            self.confidence = sum(all_confidences) / len(all_confidences)

@dataclass
class OCRMetadata:
    """Metadata about the OCR process"""
    source_file: str
    source_format: str  # pdf, png, jpg, etc.
    source_hash: str    # SHA256 of original file
    file_size_bytes: int

    ocr_model_used: str
    ocr_model_repo: str
    ocr_timestamp: str
    ocr_duration_seconds: float

    detected_language: str
    detected_doc_type: str
    total_pages: int
    total_characters: int
    total_words: int

    average_confidence: float
    min_confidence: float
    max_confidence: float
    low_confidence_count: int  # Elements with <60% confidence

@dataclass
class OCRIntermediateResult:
    """
    Complete OCR result in intermediate format.
    Optimized for second-pass AI processing.
    """
    version: str = "1.0"
    job_id: str = ""
    metadata: Optional[OCRMetadata] = None
    pages: List[PageContent] = field(default_factory=list)

    # Aggregated data for quick access
    full_text: str = ""
    all_tables: List[Table] = field(default_factory=list)

    # Confidence statistics
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    low_confidence_segments: List[Dict] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON"""
        return json.dumps(self._to_dict(), indent=indent, default=str)

    def _to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = {
            "version": self.version,
            "job_id": self.job_id,
            "metadata": asdict(self.metadata) if self.metadata else None,
            "pages": [],
            "full_text": self.full_text,
            "all_tables": [],
            "confidence_distribution": self.confidence_distribution,
            "low_confidence_segments": self.low_confidence_segments
        }

        for page in self.pages:
            page_dict = {
                "page_number": page.page_number,
                "raw_text": page.raw_text,
                "confidence": page.confidence,
                "width": page.width,
                "height": page.height,
                "blocks": [],
                "tables": []
            }
            for block in page.blocks:
                block_dict = {
                    "block_type": block.block_type,
                    "text": block.text,
                    "confidence": block.confidence,
                    "lines": []
                }
                for line in block.lines:
                    line_dict = {
                        "text": line.text,
                        "confidence": line.confidence,
                        "confidence_level": line.confidence_level.value,
                        "line_number": line.line_number
                    }
                    block_dict["lines"].append(line_dict)
                page_dict["blocks"].append(block_dict)

            for table in page.tables:
                table_dict = {
                    "rows": table.rows,
                    "cols": table.cols,
                    "confidence": table.confidence,
                    "matrix": table.to_matrix(),
                    "cells": [asdict(c) for c in table.cells]
                }
                page_dict["tables"].append(table_dict)

            result["pages"].append(page_dict)

        for table in self.all_tables:
            result["all_tables"].append({
                "page": table.page,
                "rows": table.rows,
                "cols": table.cols,
                "confidence": table.confidence,
                "matrix": table.to_matrix()
            })

        return result

    @classmethod
    def from_json(cls, json_str: str) -> 'OCRIntermediateResult':
        """Deserialize from JSON"""
        data = json.loads(json_str)
        # Reconstruction logic would go here
        result = cls()
        result.version = data.get("version", "1.0")
        result.job_id = data.get("job_id", "")
        result.full_text = data.get("full_text", "")
        result.confidence_distribution = data.get("confidence_distribution", {})
        result.low_confidence_segments = data.get("low_confidence_segments", [])
        return result

    def calculate_statistics(self):
        """Calculate confidence statistics"""
        all_confidences = []

        for page in self.pages:
            for block in page.blocks:
                for line in block.lines:
                    all_confidences.append(line.confidence)
                    if line.confidence < 0.6:
                        self.low_confidence_segments.append({
                            "page": page.page_number,
                            "text": line.text[:100],
                            "confidence": line.confidence
                        })

        if all_confidences:
            # Distribution
            self.confidence_distribution = {
                "certain_95_100": sum(1 for c in all_confidences if c >= 0.95),
                "high_80_94": sum(1 for c in all_confidences if 0.80 <= c < 0.95),
                "medium_60_79": sum(1 for c in all_confidences if 0.60 <= c < 0.80),
                "low_40_59": sum(1 for c in all_confidences if 0.40 <= c < 0.60),
                "uncertain_below_40": sum(1 for c in all_confidences if c < 0.40)
            }

            if self.metadata:
                self.metadata.average_confidence = sum(all_confidences) / len(all_confidences)
                self.metadata.min_confidence = min(all_confidences)
                self.metadata.max_confidence = max(all_confidences)
                self.metadata.low_confidence_count = self.confidence_distribution["low_40_59"] + \
                                                     self.confidence_distribution["uncertain_below_40"]


# =============================================================================
# ENHANCED OUTPUT FORMAT v2.0 (ORACLE-DESIGNED)
# Per-page metadata, signature detection, language detection
# =============================================================================

class SignatureStatus(Enum):
    """Document signature status"""
    DIGITALLY_SIGNED = "digitally_signed"
    MANUALLY_SIGNED = "manually_signed"
    BOTH = "both"
    UNSIGNED = "unsigned"


@dataclass
class DigitalSignature:
    """Digital (PKI-based) signature information"""
    signer_name: str
    signer_email: Optional[str] = None
    signing_time: Optional[str] = None
    certificate_issuer: Optional[str] = None
    certificate_valid_from: Optional[str] = None
    certificate_valid_to: Optional[str] = None
    signature_valid: bool = True
    document_modified_after_signing: bool = False
    page: int = 0
    field_name: Optional[str] = None


@dataclass
class HandwrittenSignature:
    """Detected handwritten signature"""
    page: int
    location_x: float
    location_y: float
    location_width: float
    location_height: float
    detection_confidence: float
    associated_text: Optional[str] = None  # "Signature of...", "Signed by..."
    signer_name_nearby: Optional[str] = None


@dataclass
class DocumentSignatures:
    """Complete signature information for document"""
    status: SignatureStatus = SignatureStatus.UNSIGNED
    has_digital_signature: bool = False
    has_handwritten_signature: bool = False
    total_signature_count: int = 0
    digital_signatures: List[DigitalSignature] = field(default_factory=list)
    handwritten_signatures: List[HandwrittenSignature] = field(default_factory=list)


@dataclass
class LanguageInfo:
    """Language detection result for a page"""
    detected: str = "unknown"
    confidence: float = 0.0
    alternate_languages: List[Dict[str, Any]] = field(default_factory=list)
    script: str = "unknown"  # Latin, Cyrillic, Arabic, CJK, etc.


@dataclass
class PageSourceInfo:
    """Source type and extraction method for a page"""
    source_type: str = "unknown"  # digital_text, scanned_image, mixed_content, image_with_text
    extraction_method: str = "unknown"  # direct, ocr, hybrid
    ocr_model_used: Optional[str] = None
    has_text_layer: bool = False
    has_images: bool = False
    image_coverage_percent: float = 0.0


@dataclass
class PageStatistics:
    """Content statistics for a page"""
    char_count: int = 0
    word_count: int = 0
    line_count: int = 0
    paragraph_count: int = 0
    table_count: int = 0
    image_count: int = 0
    form_field_count: int = 0


@dataclass
class PageQuality:
    """Quality metrics for a page"""
    extraction_confidence: float = 0.0
    ocr_confidence: Optional[float] = None
    text_clarity: str = "unknown"  # excellent, good, fair, poor
    issues: List[str] = field(default_factory=list)


@dataclass
class EnhancedPageContent:
    """Enhanced page content with full metadata"""
    page_number: int
    width: Optional[float] = None
    height: Optional[float] = None

    # Source & Extraction info
    source: PageSourceInfo = field(default_factory=PageSourceInfo)

    # Language detection
    language: LanguageInfo = field(default_factory=LanguageInfo)

    # Statistics
    statistics: PageStatistics = field(default_factory=PageStatistics)

    # Quality
    quality: PageQuality = field(default_factory=PageQuality)

    # Signature info for this page
    has_signature: bool = False
    signature_type: Optional[str] = None  # digital, handwritten, None

    # Content
    raw_text: str = ""
    blocks: List[TextBlock] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)


# =============================================================================
# FINAL JSON OUTPUT FORMAT (for Stage 2)
# =============================================================================

@dataclass
class ExtractedEntity:
    """Single extracted entity with metadata"""
    field_name: str
    value: Any
    data_type: str  # string, number, date, boolean, array, object
    confidence: float
    source_page: int
    source_text: str  # Original OCR text this was extracted from

@dataclass
class FinalJSONOutput:
    """
    Final structured JSON output after organization.
    Ready for knowledge queries and statistics.
    """
    version: str = "1.0"
    job_id: str = ""

    # Source tracking
    source_file: str = ""
    source_hash: str = ""
    processing_timestamp: str = ""

    # Pipeline info
    ocr_model_used: str = ""
    organization_model_used: str = ""
    total_processing_seconds: float = 0.0

    # Extracted data
    document_type: str = ""  # invoice, contract, report, etc.
    entities: List[ExtractedEntity] = field(default_factory=list)
    structured_data: Dict[str, Any] = field(default_factory=dict)

    # Tables preserved
    tables: List[Dict] = field(default_factory=list)

    # Full text backup
    raw_text: str = ""

    # Quality metrics
    overall_confidence: float = 0.0
    extraction_completeness: float = 0.0  # % of text that was structured
    low_confidence_fields: List[str] = field(default_factory=list)

    # Audit trail
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON"""
        return json.dumps(asdict(self), indent=indent, default=str)

    def to_compact_json(self) -> str:
        """Minimal JSON for storage"""
        compact = {
            "job_id": self.job_id,
            "source": self.source_file,
            "type": self.document_type,
            "data": self.structured_data,
            "tables": self.tables,
            "confidence": self.overall_confidence,
            "timestamp": self.processing_timestamp
        }
        return json.dumps(compact)


# =============================================================================
# ENHANCED JSON OUTPUT v2.0 (ORACLE-DESIGNED)
# Complete output with per-page metadata, signatures, language detection
# =============================================================================

@dataclass
class DocumentInfo:
    """Document-level information"""
    original_filename: str = ""
    stored_filename: str = ""
    file_format: str = ""
    file_size_bytes: int = 0
    file_hash_sha256: str = ""
    mime_type: str = ""


@dataclass
class ProcessingInfo:
    """Processing pipeline information"""
    pipeline_version: str = "2.0"
    ocr_model_name: Optional[str] = None
    ocr_model_repository: Optional[str] = None
    ocr_pages_processed: int = 0
    organization_model_name: Optional[str] = None
    organization_model_repository: Optional[str] = None
    extraction_seconds: float = 0.0
    organization_seconds: float = 0.0
    total_seconds: float = 0.0
    strategy: str = "unknown"  # direct_extraction, full_ocr, hybrid_per_page


@dataclass
class EnhancedJSONOutput:
    """
    Enhanced JSON output v2.0 with per-page metadata, signatures, language detection.
    Oracle-designed comprehensive document processing output.
    """
    schema_version: str = "2.0"
    job_id: str = ""
    processing_timestamp: str = ""

    # Document metadata
    document: DocumentInfo = field(default_factory=DocumentInfo)
    document_type: str = ""
    document_structure: str = ""  # pure_digital, pure_scanned, mixed
    total_pages: int = 0
    total_characters: int = 0
    total_words: int = 0
    primary_language: str = "unknown"
    languages_detected: List[str] = field(default_factory=list)

    # Signature information
    signatures: DocumentSignatures = field(default_factory=DocumentSignatures)

    # Per-page content with full metadata
    pages: List[EnhancedPageContent] = field(default_factory=list)

    # Processing info
    processing: ProcessingInfo = field(default_factory=ProcessingInfo)

    # Extracted structured data
    summary: str = ""
    entities: Dict[str, Any] = field(default_factory=dict)
    key_terms: Dict[str, Any] = field(default_factory=dict)
    tables: List[Dict] = field(default_factory=list)

    # Quality metrics
    overall_confidence: float = 0.0
    extraction_completeness: float = 0.0
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    low_confidence_segments: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Full text backup for backwards compatibility
    full_text: str = ""

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON with proper enum handling"""
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            if hasattr(obj, '__dataclass_fields__'):
                return {k: serialize(v) for k, v in asdict(obj).items()}
            if isinstance(obj, list):
                return [serialize(i) for i in obj]
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return obj

        return json.dumps(serialize(self), indent=indent, default=str)

    def to_compact_json(self) -> str:
        """Minimal JSON for API responses"""
        return json.dumps({
            "job_id": self.job_id,
            "document_type": self.document_type,
            "pages": self.total_pages,
            "language": self.primary_language,
            "signature_status": self.signatures.status.value if self.signatures else "unknown",
            "confidence": self.overall_confidence,
            "entities": self.entities,
            "full_text": self.full_text[:1000] + "..." if len(self.full_text) > 1000 else self.full_text
        }, indent=2)
