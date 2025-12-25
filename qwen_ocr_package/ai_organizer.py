#!/usr/bin/env python3
"""
AI Organization Module (Stage 2)

Takes raw OCR text and uses LLM to organize it into structured JSON.
Supports local models (Transformers) and API models (OpenAI, Anthropic).
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import sys
import os

from model_config import ModelConfig, ModelType, ModelProvider


class AIOrganizer:
    """
    Stage 2 AI Processor - Organizes raw OCR text using LLM

    Takes unstructured text and creates clean, organized JSON with:
    - Entity extraction
    - Relationship inference
    - OCR error correction
    - Multilingual support
    - Structured output
    """

    def __init__(self, model_config: ModelConfig, api_key: Optional[str] = None):
        """
        Initialize AI organizer

        Args:
            model_config: Configuration for organization model
            api_key: API key if using API model
        """
        self.model_config = model_config
        self.api_key = api_key
        self.model = None
        self.tokenizer = None

        if model_config.type == ModelType.API and not api_key:
            raise ValueError(f"API key required for {model_config.provider.value}")

    def organize(
        self,
        raw_text: str,
        schema: Optional[Dict[str, Any]] = None,
        instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Organize raw OCR text into structured JSON

        Args:
            raw_text: Raw text from OCR (Stage 1)
            schema: Optional JSON schema to enforce
            instructions: Optional custom instructions

        Returns:
            Organized data as dictionary
        """
        if self.model_config.type == ModelType.LOCAL:
            return self._organize_local(raw_text, schema, instructions)
        else:
            return self._organize_api(raw_text, schema, instructions)

    def _organize_local(
        self,
        raw_text: str,
        schema: Optional[Dict[str, Any]],
        instructions: Optional[str]
    ) -> Dict[str, Any]:
        """Organize using local LLM"""
        if self.model is None:
            self._load_local_model()

        prompt = self._build_organization_prompt(raw_text, schema, instructions)

        # Generate response
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.model_config.parameters.get("max_tokens", 8192),
            temperature=self.model_config.parameters.get("temperature", 0.1),
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        return self._parse_json_response(response)

    def _organize_api(
        self,
        raw_text: str,
        schema: Optional[Dict[str, Any]],
        instructions: Optional[str]
    ) -> Dict[str, Any]:
        """Organize using API LLM"""
        prompt = self._build_organization_prompt(raw_text, schema, instructions)

        if self.model_config.provider == ModelProvider.OPENAI:
            return self._organize_openai(prompt)
        elif self.model_config.provider == ModelProvider.ANTHROPIC:
            return self._organize_anthropic(prompt)
        else:
            raise ValueError(f"Unsupported API provider: {self.model_config.provider}")

    def _organize_openai(self, prompt: str) -> Dict[str, Any]:
        """Organize using OpenAI API"""
        try:
            import openai
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

        client = openai.OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model_config.parameters["model"],
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=self.model_config.parameters.get("temperature", 0.1),
            max_tokens=self.model_config.parameters.get("max_tokens", 8192),
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def _organize_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Organize using Anthropic API"""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Install with: pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)

        response = client.messages.create(
            model=self.model_config.parameters["model"],
            max_tokens=self.model_config.parameters.get("max_tokens", 8192),
            temperature=self.model_config.parameters.get("temperature", 0.1),
            system=self._get_system_prompt(),
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return self._parse_json_response(response.content[0].text)

    def _load_local_model(self) -> None:
        """Load local LLM model"""
        print(f"Loading {self.model_config.name}...", file=sys.stderr)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers torch"
            )

        model_id = self.model_config.parameters["model_id"]
        torch_dtype = self.model_config.parameters.get("torch_dtype", "bfloat16")

        # Map string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype_map.get(torch_dtype, torch.bfloat16),
            device_map="auto"
        )

        print(f"Model {self.model_config.name} loaded successfully!", file=sys.stderr)

    def _get_system_prompt(self) -> str:
        """Get system prompt for organization"""
        return """You are an expert document analyzer and data organizer.

Your task is to:
1. Analyze raw OCR text carefully
2. Extract ALL relevant entities and information
3. Organize data into clean, structured JSON
4. Correct OCR errors intelligently
5. Infer relationships between entities
6. Handle multilingual content
7. Extract implicit information
8. Ensure 100% accuracy

Output ONLY valid JSON. No explanations, no markdown, just pure JSON."""

    def _build_organization_prompt(
        self,
        raw_text: str,
        schema: Optional[Dict[str, Any]],
        instructions: Optional[str]
    ) -> str:
        """Build organization prompt"""
        prompt_parts = []

        if instructions:
            prompt_parts.append(f"INSTRUCTIONS:\n{instructions}\n")

        prompt_parts.append("RAW OCR TEXT:")
        prompt_parts.append("=" * 80)
        prompt_parts.append(raw_text)
        prompt_parts.append("=" * 80)

        if schema:
            prompt_parts.append("\nREQUIRED JSON SCHEMA:")
            prompt_parts.append(json.dumps(schema, indent=2))
        else:
            prompt_parts.append(self._get_default_extraction_instructions())

        prompt_parts.append("\nOutput the organized data as valid JSON:")

        return "\n".join(prompt_parts)

    def _get_default_extraction_instructions(self) -> str:
        """Get default extraction instructions"""
        return """
EXTRACT AND ORGANIZE:

1. **Company Information**:
   - Name, legal form, registration numbers
   - Address, contact details
   - Capital, VAT numbers
   - Parent company, subsidiaries

2. **People & Roles**:
   - Names, titles, positions
   - Contact information
   - Signatures, dates

3. **Financial Data**:
   - Amounts, currencies
   - Dates, periods
   - Line items, totals
   - Tax information

4. **Dates & Periods**:
   - Document date
   - Due dates
   - Fiscal periods
   - Payment terms

5. **Products & Services**:
   - Item descriptions
   - Quantities, units
   - Prices, discounts

6. **Document Metadata**:
   - Type (invoice, contract, etc.)
   - Reference numbers
   - Status, version

7. **Additional Context**:
   - Notes, comments
   - Terms and conditions
   - Legal clauses

Output comprehensive JSON with ALL extracted information.
"""

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        # Try to find JSON in response
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]

        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

            raise ValueError(f"Failed to parse JSON from response: {e}\n\nResponse:\n{response}")


class CompanyOrganizer(AIOrganizer):
    """Specialized organizer for company/invoice documents"""

    def organize_company_document(self, raw_text: str) -> Dict[str, Any]:
        """
        Organize company/invoice document

        Returns comprehensive company information
        """
        schema = {
            "type": "object",
            "properties": {
                "company": {
                    "name": "string",
                    "legalForm": "string",
                    "registrationNumber": "string",
                    "vatNumber": "string",
                    "capital": "string",
                    "address": "object",
                    "contact": "object",
                    "bankDetails": "object"
                },
                "document": {
                    "type": "string",
                    "number": "string",
                    "date": "string",
                    "dueDate": "string"
                },
                "amounts": {
                    "subtotal": "number",
                    "tax": "number",
                    "total": "number",
                    "currency": "string"
                },
                "lineItems": "array",
                "parties": "object",
                "metadata": "object"
            }
        }

        instructions = """
Extract complete company and document information.

Focus on:
- Company identification (SIREN, SIRET, VAT)
- Full address and contact details
- Financial information
- Document details (invoice number, dates)
- All line items with prices
- Client/supplier information
- Payment terms and bank details
"""

        return self.organize(raw_text, schema, instructions)


def create_organizer(
    model_config: ModelConfig,
    api_key: Optional[str] = None,
    organizer_type: str = "general"
) -> AIOrganizer:
    """
    Factory function to create organizer

    Args:
        model_config: Model configuration
        api_key: API key for API models
        organizer_type: Type of organizer ('general' or 'company')

    Returns:
        AIOrganizer instance
    """
    if organizer_type == "company":
        return CompanyOrganizer(model_config, api_key)
    else:
        return AIOrganizer(model_config, api_key)


if __name__ == "__main__":
    # Test the organizer
    from model_config import ModelRegistry

    if len(sys.argv) < 2:
        print("AI Organizer Test")
        print("\nUsage:")
        print("  python3 ai_organizer.py <text_file>")
        sys.exit(1)

    # Load test text
    test_file = Path(sys.argv[1])
    with open(test_file, 'r') as f:
        raw_text = f.read()

    # Use default model
    model_config = ModelRegistry.get_organization_model("qwen-7b")
    organizer = AIOrganizer(model_config)

    print("Organizing text...", file=sys.stderr)
    result = organizer.organize(raw_text)

    print(json.dumps(result, indent=2, ensure_ascii=False))
