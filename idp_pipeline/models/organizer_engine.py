"""
ORACLE-DESIGNED ORGANIZATION ENGINE
====================================
Transforms OCR intermediate format into structured JSON.
Uses local HuggingFace LLMs for 100% on-premise processing.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.models_config import ORGANIZATION_MODELS, OrganizationModel
from config.intermediate_format import (
    OCRIntermediateResult, FinalJSONOutput, ExtractedEntity
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseOrganizerEngine(ABC):
    """Abstract base class for organization engines"""

    def __init__(self, model_config: OrganizationModel):
        self.config = model_config
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.loaded = False

    @abstractmethod
    def load_model(self):
        """Load model into memory"""
        pass

    @abstractmethod
    def organize(self, text: str, doc_type: str = None) -> Dict[str, Any]:
        """Organize text into structured JSON"""
        pass

    def unload_model(self):
        """Unload model from memory"""
        self.model = None
        self.tokenizer = None
        self.loaded = False
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class QwenOrganizerEngine(BaseOrganizerEngine):
    """Qwen2.5 based organization engine"""

    def load_model(self):
        if self.loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading {self.config.name} from {self.config.hf_repo}")

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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_repo,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        if self.device == "mps":
            self.model = self.model.to(self.device)

        self.loaded = True
        logger.info(f"Model loaded on {self.device}")

    def organize(self, text: str, doc_type: str = None) -> Dict[str, Any]:
        if not self.loaded:
            self.load_model()

        import torch

        # Build prompt
        prompt = self._build_prompt(text, doc_type)

        messages = [
            {"role": "system", "content": "You are a document analysis expert. Extract ALL information from the text and organize it into a structured JSON format. Never omit any data."},
            {"role": "user", "content": prompt}
        ]

        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text_input], return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        # Parse JSON from response
        return self._extract_json(response)

    def _build_prompt(self, text: str, doc_type: str = None) -> str:
        type_hint = f"This is a {doc_type}. " if doc_type else ""

        return f"""{type_hint}Analyze the following OCR-extracted text and convert it into a comprehensive JSON structure.

REQUIREMENTS:
1. Extract ALL information - do not omit anything
2. Identify the document type (invoice, contract, report, letter, form, etc.)
3. Extract all entities: names, dates, numbers, addresses, amounts, etc.
4. Preserve any tabular data as arrays
5. Include confidence notes for unclear extractions

OCR TEXT:
```
{text[:8000]}  # Truncate if needed
```

Return ONLY valid JSON in this format:
{{
    "document_type": "detected type",
    "summary": "brief summary",
    "entities": {{
        "dates": [...],
        "amounts": [...],
        "names": [...],
        "addresses": [...],
        "other": [...]
    }},
    "structured_data": {{
        // Document-specific structured fields
    }},
    "tables": [
        // Any tabular data found
    ],
    "metadata": {{
        "language": "detected language",
        "page_count": estimated,
        "confidence_notes": [...]
    }}
}}"""

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from model response"""
        # Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # Try direct parse
        try:
            # Find first { and last }
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                return json.loads(response[start:end+1])
        except:
            pass

        # Return raw if parsing fails
        return {
            "raw_response": response,
            "parse_error": True
        }


class LlamaOrganizerEngine(BaseOrganizerEngine):
    """Llama 3.2 based organization engine"""

    def load_model(self):
        if self.loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading {self.config.name} from {self.config.hf_repo}")

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.hf_repo)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_repo,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.device == "mps":
            self.model = self.model.to(self.device)

        self.loaded = True
        logger.info(f"Model loaded on {self.device}")

    def organize(self, text: str, doc_type: str = None) -> Dict[str, Any]:
        if not self.loaded:
            self.load_model()

        import torch

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a document analysis expert. Extract and organize information into JSON.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Extract ALL data from this OCR text into structured JSON:

{text[:6000]}

Return ONLY valid JSON with document_type, entities, structured_data, and tables.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        return self._extract_json(response)

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from response"""
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                return json.loads(response[start:end+1])
        except:
            pass
        return {"raw_response": response, "parse_error": True}


class MistralOrganizerEngine(BaseOrganizerEngine):
    """Mistral based organization engine"""

    def load_model(self):
        if self.loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading {self.config.name} from {self.config.hf_repo}")

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.hf_repo)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_repo,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        self.loaded = True
        logger.info(f"Model loaded on {self.device}")

    def organize(self, text: str, doc_type: str = None) -> Dict[str, Any]:
        if not self.loaded:
            self.load_model()

        import torch

        messages = [
            {"role": "user", "content": f"Extract ALL information from this OCR text into a structured JSON with document_type, entities, structured_data, and tables:\n\n{text[:6000]}"}
        ]

        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        if self.device != "cpu":
            inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=4096,
                do_sample=False
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            start = response.rfind('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                return json.loads(response[start:end+1])
        except:
            pass
        return {"raw_response": response, "parse_error": True}


# Engine factory
ORG_ENGINE_MAP = {
    "qwen2_5_7b": QwenOrganizerEngine,
    "qwen2_5_1_5b": QwenOrganizerEngine,
    "llama3_2_3b": LlamaOrganizerEngine,
    "mistral_7b": MistralOrganizerEngine,
    "phi3_5_mini": QwenOrganizerEngine,  # Similar interface
    "gemma2_2b": QwenOrganizerEngine,
    "tinyllama": LlamaOrganizerEngine,
}


def get_organizer_engine(model_key: str) -> BaseOrganizerEngine:
    """Get organizer engine by model key"""
    if model_key not in ORGANIZATION_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(ORGANIZATION_MODELS.keys())}")

    model_config = ORGANIZATION_MODELS[model_key]
    engine_class = ORG_ENGINE_MAP.get(model_key, QwenOrganizerEngine)

    return engine_class(model_config)


class UnifiedOrganizer:
    """
    Unified organizer that processes OCR results into final JSON.
    Supports both v1.0 (FinalJSONOutput) and v2.0 (EnhancedJSONOutput) formats.
    """

    def __init__(self, model_key: str = "llama3_2_3b"):
        self.model_key = model_key
        self.engine = get_organizer_engine(model_key)
        self.model_config = ORGANIZATION_MODELS[model_key]

    def process(self, ocr_result: OCRIntermediateResult,
                progress_callback=None, enhanced_ocr_data: dict = None) -> FinalJSONOutput:
        """Process OCR result into final structured JSON"""
        start_time = time.time()

        if progress_callback:
            progress_callback(0.1, "Starting organization")

        # Extract text for processing
        full_text = ocr_result.full_text

        # Detect document type from content
        doc_type = self._detect_doc_type(full_text)

        if progress_callback:
            progress_callback(0.3, f"Detected type: {doc_type}")

        # Run organization
        organized = self.engine.organize(full_text, doc_type)

        if progress_callback:
            progress_callback(0.8, "Structuring output")

        # Build final output
        output = FinalJSONOutput(
            job_id=ocr_result.job_id,
            source_file=ocr_result.metadata.source_file if ocr_result.metadata else "",
            source_hash=ocr_result.metadata.source_hash if ocr_result.metadata else "",
            processing_timestamp=datetime.utcnow().isoformat(),
            ocr_model_used=ocr_result.metadata.ocr_model_used if ocr_result.metadata else "",
            organization_model_used=self.model_config.name,
            document_type=organized.get("document_type", doc_type),
            structured_data=organized.get("structured_data", {}),
            tables=organized.get("tables", []),
            raw_text=full_text,
            overall_confidence=ocr_result.metadata.average_confidence if ocr_result.metadata else 0.8
        )

        # Extract entities
        entities = organized.get("entities", {})
        for entity_type, values in entities.items():
            if isinstance(values, list):
                for val in values:
                    output.entities.append(ExtractedEntity(
                        field_name=entity_type,
                        value=val,
                        data_type="string",
                        confidence=0.8,
                        source_page=1,
                        source_text=""
                    ))

        # Calculate processing time
        output.total_processing_seconds = time.time() - start_time

        # Check for warnings
        if organized.get("parse_error"):
            output.warnings.append("JSON parsing had issues - check raw_response")
        if ocr_result.metadata and ocr_result.metadata.low_confidence_count > 0:
            output.warnings.append(f"{ocr_result.metadata.low_confidence_count} low-confidence segments detected")

        if progress_callback:
            progress_callback(1.0, "Organization complete")

        return output

    def _detect_doc_type(self, text: str) -> str:
        """Simple document type detection"""
        text_lower = text.lower()

        indicators = {
            "invoice": ["invoice", "facture", "bill to", "due date", "total amount", "subtotal"],
            "contract": ["agreement", "contract", "parties", "whereas", "terms and conditions"],
            "report": ["report", "summary", "findings", "conclusion", "analysis"],
            "letter": ["dear", "sincerely", "regards", "yours truly"],
            "form": ["form", "please fill", "applicant", "signature"],
            "receipt": ["receipt", "transaction", "paid", "change"],
            "resume": ["resume", "cv", "curriculum", "experience", "education", "skills"],
        }

        scores = {}
        for doc_type, keywords in indicators.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[doc_type] = score

        if scores:
            return max(scores, key=scores.get)
        return "document"

    def process_enhanced(self, ocr_result: OCRIntermediateResult,
                          enhanced_ocr_data: dict,
                          progress_callback=None) -> dict:
        """
        ORACLE-DESIGNED: Process OCR result into enhanced JSON v2.0 format.

        This method produces the complete v2.0 output with:
        - Per-page OCR confidence
        - Per-page language detection
        - Per-page word/character counts
        - Signature information
        - Source format details
        - AI-extracted entities and structured data
        """
        start_time = time.time()

        if progress_callback:
            progress_callback(0.1, "Starting enhanced organization")

        # Extract text for processing
        full_text = ocr_result.full_text

        # Detect document type from content
        doc_type = self._detect_doc_type(full_text)

        if progress_callback:
            progress_callback(0.3, f"Detected type: {doc_type}")

        # Run AI organization to extract entities
        organized = self.engine.organize(full_text, doc_type)

        if progress_callback:
            progress_callback(0.8, "Structuring enhanced output")

        # Merge enhanced OCR data with AI-extracted data
        # Start with the enhanced OCR data as base
        output = enhanced_ocr_data.copy()

        # Add organization results
        output["document_type"] = organized.get("document_type", doc_type)
        output["summary"] = organized.get("summary", "")
        output["entities"] = organized.get("entities", {})
        output["key_terms"] = organized.get("metadata", {})
        output["structured_data"] = organized.get("structured_data", {})

        # Add tables from AI extraction if not already present
        ai_tables = organized.get("tables", [])
        if ai_tables and "tables" not in output:
            output["tables"] = ai_tables

        # Add processing info for organization stage
        org_time = time.time() - start_time
        output["processing"]["organization_model_name"] = self.model_config.name
        output["processing"]["organization_model_repository"] = self.model_config.hf_repo
        output["processing"]["organization_seconds"] = org_time
        output["processing"]["total_seconds"] = (
            output["processing"].get("extraction_seconds", 0) + org_time
        )

        # Add warnings if any
        warnings = output.get("warnings", [])
        if organized.get("parse_error"):
            warnings.append("JSON parsing had issues - check raw_response")
        output["warnings"] = warnings
        output["errors"] = output.get("errors", [])

        if progress_callback:
            progress_callback(1.0, "Enhanced organization complete")

        logger.info(f"Enhanced organization completed in {org_time:.2f}s")

        return output
