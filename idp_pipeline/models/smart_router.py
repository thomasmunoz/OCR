"""
ORACLE-DESIGNED SMART MODEL ROUTER
==================================
Automatically selects the best AI model based on:
- Input file format
- Detected language
- Document type
- Available VRAM
- User preferences (speed vs quality)

Includes automatic model downloading with progress bar.
"""

import os
import sys
import magic
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Callable
from dataclasses import dataclass
from enum import Enum
import subprocess
import re
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.models_config import (
    OCR_MODELS, ORGANIZATION_MODELS,
    DocumentType, Language, OCRModel, OrganizationModel,
    MODEL_SELECTION_RULES, RECOMMENDED_CONFIG
)

# Import ModelManager for download handling
try:
    from model_store import ModelManager, get_model_manager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    logger.warning("ModelManager not available - models must be pre-downloaded")

# Models that have working engine implementations
MODELS_WITH_ENGINES = {
    "qwen2_vl_2b",
    "qwen2_vl_7b",
    "got_ocr2",
    "trocr_large_printed",
    "trocr_large_handwritten",
    "florence2_large",
}

class Priority(Enum):
    """Processing priority"""
    QUALITY = "quality"    # Best quality, slower
    BALANCED = "balanced"  # Good balance
    SPEED = "speed"        # Fast, acceptable quality

@dataclass
class FileAnalysis:
    """Analysis result for input file"""
    file_path: str
    file_format: str
    file_size_bytes: int
    file_hash: str
    mime_type: str
    is_scanned: bool = False
    has_tables: bool = False
    is_handwritten: bool = False
    detected_language: Language = Language.ENGLISH
    page_count: int = 1
    complexity_score: int = 50  # 0-100

@dataclass
class ModelSelection:
    """Selected models with reasoning"""
    ocr_model: OCRModel
    organization_model: OrganizationModel
    reasoning: List[str]
    confidence: float
    estimated_vram_gb: float
    warnings: List[str]

class SmartModelRouter:
    """
    Intelligent model selection based on input characteristics.
    """

    def __init__(self, available_vram_gb: float = 16.0, priority: Priority = Priority.BALANCED):
        self.available_vram = available_vram_gb
        self.priority = priority
        self._detect_system_capabilities()

    def _detect_system_capabilities(self):
        """Detect available GPU and VRAM"""
        self.gpu_available = False
        self.detected_vram = 0.0

        # Try NVIDIA
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                vram_mb = int(result.stdout.strip().split('\n')[0])
                self.detected_vram = vram_mb / 1024
                self.gpu_available = True
        except:
            pass

        # Try Apple Silicon (MPS)
        if not self.gpu_available:
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Apple Silicon shares RAM with GPU
                    total_ram = int(result.stdout.strip()) / (1024**3)
                    self.detected_vram = total_ram * 0.6  # ~60% available for MPS
                    self.gpu_available = True
            except Exception:
                pass

        if self.detected_vram > 0:
            self.available_vram = min(self.available_vram, self.detected_vram)

    def analyze_file(self, file_path: str) -> FileAnalysis:
        """Analyze input file to determine characteristics"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Basic info
        file_size = path.stat().st_size

        # File hash
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Detect format and mime type
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(file_path)

        # Determine format
        extension = path.suffix.lower().lstrip('.')
        format_map = {
            'pdf': 'pdf',
            'png': 'png',
            'jpg': 'jpg',
            'jpeg': 'jpg',
            'tiff': 'tiff',
            'tif': 'tiff',
            'bmp': 'bmp',
            'webp': 'webp',
            'gif': 'gif',
        }
        file_format = format_map.get(extension, extension)

        # Create analysis
        analysis = FileAnalysis(
            file_path=str(path.absolute()),
            file_format=file_format,
            file_size_bytes=file_size,
            file_hash=file_hash,
            mime_type=mime_type
        )

        # Detect page count for PDFs
        if file_format == 'pdf':
            analysis.page_count = self._get_pdf_page_count(file_path)

        # Detect language (heuristic based on filename for now)
        analysis.detected_language = self._detect_language_hint(file_path)

        # Estimate complexity
        analysis.complexity_score = self._estimate_complexity(analysis)

        return analysis

    def _get_pdf_page_count(self, file_path: str) -> int:
        """Get page count from PDF"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            count = len(doc)
            doc.close()
            return count
        except:
            return 1

    def _detect_language_hint(self, file_path: str) -> Language:
        """Detect language from filename hints"""
        filename = Path(file_path).name.lower()

        language_hints = {
            Language.FRENCH: ['_fr', '_french', 'fr_', 'french_', 'français'],
            Language.GERMAN: ['_de', '_german', 'de_', 'german_', 'deutsch'],
            Language.SPANISH: ['_es', '_spanish', 'es_', 'spanish_', 'español'],
            Language.CHINESE: ['_zh', '_chinese', 'zh_', 'chinese_', '中文'],
            Language.JAPANESE: ['_ja', '_japanese', 'ja_', 'japanese_', '日本語'],
            Language.KOREAN: ['_ko', '_korean', 'ko_', 'korean_', '한국어'],
            Language.ARABIC: ['_ar', '_arabic', 'ar_', 'arabic_', 'عربي'],
        }

        for lang, hints in language_hints.items():
            if any(hint in filename for hint in hints):
                return lang

        return Language.ENGLISH  # Default

    def _estimate_complexity(self, analysis: FileAnalysis) -> int:
        """Estimate document complexity 0-100"""
        score = 50  # Base score

        # Size factor
        if analysis.file_size_bytes > 10 * 1024 * 1024:  # >10MB
            score += 20
        elif analysis.file_size_bytes > 1 * 1024 * 1024:  # >1MB
            score += 10

        # Page count factor
        if analysis.page_count > 50:
            score += 20
        elif analysis.page_count > 10:
            score += 10

        # Format factor
        if analysis.file_format == 'pdf':
            score += 10  # PDFs can have complex layouts

        # Non-Latin language factor
        if analysis.detected_language in [Language.CHINESE, Language.JAPANESE, Language.KOREAN, Language.ARABIC]:
            score += 15

        return min(100, score)

    def select_models(self, analysis: FileAnalysis) -> ModelSelection:
        """Select optimal models based on file analysis"""
        reasoning = []
        warnings = []

        # =========================================
        # STAGE 1: SELECT OCR MODEL
        # =========================================

        ocr_candidates = []

        # Filter by format support AND working engine implementation
        # Note: PDFs are converted to images before OCR, so image-only models can handle PDFs
        for name, model in OCR_MODELS.items():
            if name not in MODELS_WITH_ENGINES:
                continue  # Skip models without engine implementations
            # PDFs are converted to images, so any model that supports images can handle PDF
            if analysis.file_format in model.supported_formats:
                ocr_candidates.append((name, model))
            elif analysis.file_format == 'pdf' and any(fmt in model.supported_formats for fmt in ['png', 'jpg', 'jpeg']):
                ocr_candidates.append((name, model))  # Model can handle PDF after image conversion

        reasoning.append(f"Format '{analysis.file_format}': {len(ocr_candidates)} OCR models with engines support it")

        # Filter by language
        lang_filtered = []
        for name, model in ocr_candidates:
            if analysis.detected_language in model.supported_languages or Language.MULTI in model.supported_languages:
                lang_filtered.append((name, model))

        if lang_filtered:
            ocr_candidates = lang_filtered
            reasoning.append(f"Language '{analysis.detected_language.value}': {len(ocr_candidates)} models support it")

        # Filter by VRAM
        vram_filtered = []
        for name, model in ocr_candidates:
            if model.vram_required_gb <= self.available_vram:
                vram_filtered.append((name, model))

        if vram_filtered:
            ocr_candidates = vram_filtered
        else:
            warnings.append(f"No OCR model fits in {self.available_vram}GB VRAM. Using smallest.")
            ocr_candidates = sorted(ocr_candidates, key=lambda x: x[1].vram_required_gb)[:3]

        reasoning.append(f"VRAM filter ({self.available_vram}GB): {len(ocr_candidates)} models fit")

        # Score and rank by priority
        def ocr_score(item):
            name, model = item
            if self.priority == Priority.QUALITY:
                return model.quality_score * 2 + model.speed_score
            elif self.priority == Priority.SPEED:
                return model.quality_score + model.speed_score * 2
            else:  # BALANCED
                return model.quality_score + model.speed_score

        ocr_candidates.sort(key=ocr_score, reverse=True)

        selected_ocr_name, selected_ocr = ocr_candidates[0]
        reasoning.append(f"Selected OCR: {selected_ocr.name} (quality={selected_ocr.quality_score}, speed={selected_ocr.speed_score})")

        # =========================================
        # STAGE 2: SELECT ORGANIZATION MODEL
        # =========================================

        org_candidates = []

        # Filter by VRAM (considering OCR model too)
        remaining_vram = self.available_vram - selected_ocr.vram_required_gb * 0.5  # Overlap possible

        for name, model in ORGANIZATION_MODELS.items():
            if model.vram_required_gb <= remaining_vram:
                org_candidates.append((name, model))

        if not org_candidates:
            warnings.append("Limited VRAM for organization model. Using smallest.")
            org_candidates = sorted(ORGANIZATION_MODELS.items(), key=lambda x: x[1].vram_required_gb)[:3]

        # Score and rank
        def org_score(item):
            name, model = item
            if self.priority == Priority.QUALITY:
                return model.quality_score * 2 + model.speed_score
            elif self.priority == Priority.SPEED:
                return model.quality_score + model.speed_score * 2
            else:
                return model.quality_score + model.speed_score

        org_candidates.sort(key=org_score, reverse=True)

        selected_org_name, selected_org = org_candidates[0]
        reasoning.append(f"Selected Organization: {selected_org.name} (quality={selected_org.quality_score}, speed={selected_org.speed_score})")

        # =========================================
        # CALCULATE CONFIDENCE
        # =========================================

        # Confidence based on how well models match the input
        confidence = 0.7  # Base

        # Boost for explicit language support
        if analysis.detected_language in selected_ocr.supported_languages:
            confidence += 0.1

        # Boost for quality models
        if selected_ocr.quality_score >= 90:
            confidence += 0.1

        # Reduce for warnings
        confidence -= len(warnings) * 0.05

        confidence = max(0.5, min(1.0, confidence))

        # =========================================
        # RETURN SELECTION
        # =========================================

        return ModelSelection(
            ocr_model=selected_ocr,
            organization_model=selected_org,
            reasoning=reasoning,
            confidence=confidence,
            estimated_vram_gb=selected_ocr.vram_required_gb + selected_org.vram_required_gb * 0.5,
            warnings=warnings
        )

    def get_download_commands(self, selection: ModelSelection) -> List[str]:
        """Get commands to download selected models"""
        commands = []
        commands.append(f"# Download OCR model: {selection.ocr_model.name}")
        commands.append(f"huggingface-cli download {selection.ocr_model.hf_repo}")
        commands.append("")
        commands.append(f"# Download Organization model: {selection.organization_model.name}")
        commands.append(f"huggingface-cli download {selection.organization_model.hf_repo}")
        return commands

    def ensure_models_ready(
        self,
        selection: ModelSelection,
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None
    ) -> bool:
        """
        Ensure selected models are downloaded and ready.
        Downloads with progress bar if not present.

        Args:
            selection: ModelSelection from select_models()
            progress_callback: Optional callback(current, total, model_name, status)
                              status is "downloading", "ready", or "error"

        Returns:
            True if all models are ready, False otherwise
        """
        if not MODEL_MANAGER_AVAILABLE:
            logger.warning("ModelManager not available. Assuming models are pre-downloaded.")
            return True

        manager = get_model_manager()

        # Find OCR model key
        ocr_key = None
        for key, model in OCR_MODELS.items():
            if model.hf_repo == selection.ocr_model.hf_repo:
                ocr_key = key
                break

        # Find organization model key
        org_key = None
        for key, model in ORGANIZATION_MODELS.items():
            if model.hf_repo == selection.organization_model.hf_repo:
                org_key = key
                break

        success = True

        # Check and download OCR model
        if ocr_key:
            if not manager.is_model_ready(ocr_key):
                logger.info(f"OCR model {selection.ocr_model.name} not found. Downloading...")
                if progress_callback:
                    progress_callback(0, 0, selection.ocr_model.name, "downloading")

                def ocr_progress(current, total, name):
                    if progress_callback:
                        progress_callback(current, total, name, "downloading")

                if manager.download_model(ocr_key, ocr_progress):
                    if progress_callback:
                        progress_callback(100, 100, selection.ocr_model.name, "ready")
                    logger.info(f"OCR model {selection.ocr_model.name} ready")
                else:
                    if progress_callback:
                        progress_callback(0, 0, selection.ocr_model.name, "error")
                    logger.error(f"Failed to download OCR model {selection.ocr_model.name}")
                    success = False
            else:
                logger.info(f"OCR model {selection.ocr_model.name} already available")
                if progress_callback:
                    progress_callback(100, 100, selection.ocr_model.name, "ready")

        # Check and download organization model
        if org_key:
            if not manager.is_model_ready(org_key):
                logger.info(f"Organization model {selection.organization_model.name} not found. Downloading...")
                if progress_callback:
                    progress_callback(0, 0, selection.organization_model.name, "downloading")

                def org_progress(current, total, name):
                    if progress_callback:
                        progress_callback(current, total, name, "downloading")

                if manager.download_model(org_key, org_progress):
                    if progress_callback:
                        progress_callback(100, 100, selection.organization_model.name, "ready")
                    logger.info(f"Organization model {selection.organization_model.name} ready")
                else:
                    if progress_callback:
                        progress_callback(0, 0, selection.organization_model.name, "error")
                    logger.error(f"Failed to download organization model {selection.organization_model.name}")
                    success = False
            else:
                logger.info(f"Organization model {selection.organization_model.name} already available")
                if progress_callback:
                    progress_callback(100, 100, selection.organization_model.name, "ready")

        return success

    def get_model_paths(self, selection: ModelSelection) -> Dict[str, Optional[str]]:
        """
        Get local paths for selected models.

        Returns:
            Dict with 'ocr' and 'organization' paths (None if not downloaded)
        """
        if not MODEL_MANAGER_AVAILABLE:
            return {"ocr": None, "organization": None}

        manager = get_model_manager()
        paths = {"ocr": None, "organization": None}

        # Find OCR model key and path
        for key, model in OCR_MODELS.items():
            if model.hf_repo == selection.ocr_model.hf_repo:
                paths["ocr"] = manager.get_model_path(key)
                break

        # Find organization model key and path
        for key, model in ORGANIZATION_MODELS.items():
            if model.hf_repo == selection.organization_model.hf_repo:
                paths["organization"] = manager.get_model_path(key)
                break

        return paths


def create_router(vram_gb: float = 16.0, priority: str = "balanced") -> SmartModelRouter:
    """Factory function to create a router"""
    priority_map = {
        "quality": Priority.QUALITY,
        "balanced": Priority.BALANCED,
        "speed": Priority.SPEED
    }
    return SmartModelRouter(
        available_vram_gb=vram_gb,
        priority=priority_map.get(priority, Priority.BALANCED)
    )


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smart Model Router")
    parser.add_argument("file", help="File to analyze")
    parser.add_argument("--vram", type=float, default=16.0, help="Available VRAM in GB")
    parser.add_argument("--priority", choices=["quality", "balanced", "speed"], default="balanced")

    args = parser.parse_args()

    router = create_router(args.vram, args.priority)
    analysis = router.analyze_file(args.file)
    selection = router.select_models(analysis)

    print(f"\n{'='*60}")
    print("FILE ANALYSIS")
    print(f"{'='*60}")
    print(f"File: {analysis.file_path}")
    print(f"Format: {analysis.file_format}")
    print(f"Size: {analysis.file_size_bytes / 1024:.1f} KB")
    print(f"Language: {analysis.detected_language.value}")
    print(f"Pages: {analysis.page_count}")
    print(f"Complexity: {analysis.complexity_score}/100")

    print(f"\n{'='*60}")
    print("MODEL SELECTION")
    print(f"{'='*60}")
    print(f"OCR Model: {selection.ocr_model.name}")
    print(f"  - Repo: {selection.ocr_model.hf_repo}")
    print(f"  - VRAM: {selection.ocr_model.vram_required_gb} GB")
    print(f"Organization Model: {selection.organization_model.name}")
    print(f"  - Repo: {selection.organization_model.hf_repo}")
    print(f"  - VRAM: {selection.organization_model.vram_required_gb} GB")
    print(f"\nSelection Confidence: {selection.confidence*100:.0f}%")
    print(f"Total VRAM Needed: ~{selection.estimated_vram_gb:.1f} GB")

    print(f"\n{'='*60}")
    print("REASONING")
    print(f"{'='*60}")
    for reason in selection.reasoning:
        print(f"  - {reason}")

    if selection.warnings:
        print(f"\nWARNINGS:")
        for warn in selection.warnings:
            print(f"  ⚠️  {warn}")

    print(f"\n{'='*60}")
    print("DOWNLOAD COMMANDS")
    print(f"{'='*60}")
    for cmd in router.get_download_commands(selection):
        print(cmd)
