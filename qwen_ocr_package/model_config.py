#!/usr/bin/env python3
"""
Model Configuration System for Dual-AI OCR Pipeline

Manages OCR (vision) and organization (LLM) model configurations.
Supports local models (Transformers) and API models (OpenAI, Anthropic, etc.)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path


class ModelType(Enum):
    """Model deployment type"""
    LOCAL = "local"
    API = "api"


class ModelProvider(Enum):
    """Model provider"""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Configuration for a single model"""
    name: str
    type: ModelType
    provider: ModelProvider
    description: str
    capabilities: List[str]
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "type": self.type.value,
            "provider": self.provider.value,
            "description": self.description,
            "capabilities": self.capabilities,
            "parameters": self.parameters
        }


class ModelRegistry:
    """Registry of available models for OCR and organization"""

    # OCR Models (Vision Models)
    OCR_MODELS = {
        "qwen2-vl-2b-ocr": ModelConfig(
            name="qwen2-vl-2b-ocr",
            type=ModelType.LOCAL,
            provider=ModelProvider.HUGGINGFACE,
            description="Qwen2-VL-2B-OCR - Optimized for text extraction",
            capabilities=["ocr", "vision", "multilingual"],
            parameters={
                "model_id": "JackChew/Qwen2-VL-2B-OCR",
                "max_tokens": 2048,
                "torch_dtype": "float32"
            }
        ),
        "qwen2-vl-7b": ModelConfig(
            name="qwen2-vl-7b",
            type=ModelType.LOCAL,
            provider=ModelProvider.HUGGINGFACE,
            description="Qwen2-VL-7B - Larger vision model for complex documents",
            capabilities=["ocr", "vision", "multilingual", "analysis"],
            parameters={
                "model_id": "Qwen/Qwen2-VL-7B-Instruct",
                "max_tokens": 4096,
                "torch_dtype": "float16"
            }
        ),
        "gpt-4-vision": ModelConfig(
            name="gpt-4-vision",
            type=ModelType.API,
            provider=ModelProvider.OPENAI,
            description="GPT-4 Vision - OpenAI's vision model",
            capabilities=["ocr", "vision", "multilingual", "analysis"],
            parameters={
                "model": "gpt-4-vision-preview",
                "max_tokens": 4096
            }
        ),
        "claude-3-opus": ModelConfig(
            name="claude-3-opus",
            type=ModelType.API,
            provider=ModelProvider.ANTHROPIC,
            description="Claude 3 Opus - Anthropic's vision model",
            capabilities=["ocr", "vision", "multilingual", "analysis"],
            parameters={
                "model": "claude-3-opus-20240229",
                "max_tokens": 4096
            }
        )
    }

    # Organization Models (Language Models)
    ORGANIZATION_MODELS = {
        "qwen-32b": ModelConfig(
            name="qwen-32b",
            type=ModelType.LOCAL,
            provider=ModelProvider.HUGGINGFACE,
            description="Qwen 32B - Default local LLM for text organization",
            capabilities=["text", "json", "multilingual", "reasoning"],
            parameters={
                "model_id": "Qwen/Qwen2.5-32B-Instruct",
                "max_tokens": 8192,
                "torch_dtype": "bfloat16",
                "temperature": 0.1
            }
        ),
        "qwen-14b": ModelConfig(
            name="qwen-14b",
            type=ModelType.LOCAL,
            provider=ModelProvider.HUGGINGFACE,
            description="Qwen 14B - Smaller, faster alternative",
            capabilities=["text", "json", "multilingual"],
            parameters={
                "model_id": "Qwen/Qwen2.5-14B-Instruct",
                "max_tokens": 4096,
                "torch_dtype": "bfloat16",
                "temperature": 0.1
            }
        ),
        "qwen-7b": ModelConfig(
            name="qwen-7b",
            type=ModelType.LOCAL,
            provider=ModelProvider.HUGGINGFACE,
            description="Qwen 7B - Lightweight option for resource-constrained systems",
            capabilities=["text", "json", "multilingual"],
            parameters={
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "max_tokens": 4096,
                "torch_dtype": "float16",
                "temperature": 0.1
            }
        ),
        "gpt-4": ModelConfig(
            name="gpt-4",
            type=ModelType.API,
            provider=ModelProvider.OPENAI,
            description="GPT-4 - OpenAI's most capable model",
            capabilities=["text", "json", "multilingual", "reasoning"],
            parameters={
                "model": "gpt-4-turbo-preview",
                "max_tokens": 8192,
                "temperature": 0.1
            }
        ),
        "gpt-4o": ModelConfig(
            name="gpt-4o",
            type=ModelType.API,
            provider=ModelProvider.OPENAI,
            description="GPT-4o - OpenAI's optimized model",
            capabilities=["text", "json", "multilingual", "reasoning"],
            parameters={
                "model": "gpt-4o",
                "max_tokens": 8192,
                "temperature": 0.1
            }
        ),
        "claude-opus": ModelConfig(
            name="claude-opus",
            type=ModelType.API,
            provider=ModelProvider.ANTHROPIC,
            description="Claude 3 Opus - Anthropic's most capable model",
            capabilities=["text", "json", "multilingual", "reasoning"],
            parameters={
                "model": "claude-3-opus-20240229",
                "max_tokens": 8192,
                "temperature": 0.1
            }
        ),
        "claude-sonnet": ModelConfig(
            name="claude-sonnet",
            type=ModelType.API,
            provider=ModelProvider.ANTHROPIC,
            description="Claude 3.5 Sonnet - Balanced performance and cost",
            capabilities=["text", "json", "multilingual", "reasoning"],
            parameters={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 8192,
                "temperature": 0.1
            }
        )
    }

    @classmethod
    def get_ocr_model(cls, model_name: str) -> Optional[ModelConfig]:
        """Get OCR model configuration"""
        return cls.OCR_MODELS.get(model_name)

    @classmethod
    def get_organization_model(cls, model_name: str) -> Optional[ModelConfig]:
        """Get organization model configuration"""
        return cls.ORGANIZATION_MODELS.get(model_name)

    @classmethod
    def list_ocr_models(cls) -> List[str]:
        """List available OCR models"""
        return list(cls.OCR_MODELS.keys())

    @classmethod
    def list_organization_models(cls) -> List[str]:
        """List available organization models"""
        return list(cls.ORGANIZATION_MODELS.keys())

    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed model information"""
        model = cls.get_ocr_model(model_name) or cls.get_organization_model(model_name)
        return model.to_dict() if model else None

    @classmethod
    def print_available_models(cls) -> None:
        """Print all available models in a formatted table"""
        print("\n" + "="*80)
        print("AVAILABLE OCR MODELS (Stage 1 - Vision)")
        print("="*80)

        for name, config in cls.OCR_MODELS.items():
            print(f"\n📷 {name}")
            print(f"   Type: {config.type.value.upper()}")
            print(f"   Provider: {config.provider.value}")
            print(f"   Description: {config.description}")
            print(f"   Capabilities: {', '.join(config.capabilities)}")

        print("\n" + "="*80)
        print("AVAILABLE ORGANIZATION MODELS (Stage 2 - LLM)")
        print("="*80)

        for name, config in cls.ORGANIZATION_MODELS.items():
            print(f"\n🧠 {name}")
            print(f"   Type: {config.type.value.upper()}")
            print(f"   Provider: {config.provider.value}")
            print(f"   Description: {config.description}")
            print(f"   Capabilities: {', '.join(config.capabilities)}")

        print("\n" + "="*80)


class PipelineConfig:
    """Configuration for the dual-AI extraction pipeline"""

    def __init__(
        self,
        ocr_model: str = "qwen2-vl-2b-ocr",
        organization_model: str = "qwen-32b",
        device: str = "auto",
        api_keys: Optional[Dict[str, str]] = None
    ):
        """
        Initialize pipeline configuration

        Args:
            ocr_model: Name of OCR model to use
            organization_model: Name of organization model to use
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            api_keys: Dictionary of API keys {'openai': 'key', 'anthropic': 'key'}
        """
        self.ocr_model_name = ocr_model
        self.organization_model_name = organization_model
        self.device = device
        self.api_keys = api_keys or {}

        # Validate models
        self.ocr_config = ModelRegistry.get_ocr_model(ocr_model)
        if not self.ocr_config:
            raise ValueError(f"Unknown OCR model: {ocr_model}")

        self.org_config = ModelRegistry.get_organization_model(organization_model)
        if not self.org_config:
            raise ValueError(f"Unknown organization model: {organization_model}")

        # Check API keys for API models
        if self.org_config.type == ModelType.API:
            provider = self.org_config.provider.value
            if provider not in self.api_keys:
                raise ValueError(
                    f"API key required for {provider}. "
                    f"Set via --api-key-{provider} or environment variable"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "ocr_model": {
                "name": self.ocr_model_name,
                "config": self.ocr_config.to_dict()
            },
            "organization_model": {
                "name": self.organization_model_name,
                "config": self.org_config.to_dict()
            },
            "device": self.device,
            "has_api_keys": bool(self.api_keys)
        }

    def save(self, path: Path) -> None:
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'PipelineConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            ocr_model=data["ocr_model"]["name"],
            organization_model=data["organization_model"]["name"],
            device=data.get("device", "auto")
        )


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration"""
    return PipelineConfig(
        ocr_model="qwen2-vl-2b-ocr",
        organization_model="qwen-32b",
        device="auto"
    )


if __name__ == "__main__":
    # Print available models
    ModelRegistry.print_available_models()

    # Show default configuration
    print("\n" + "="*80)
    print("DEFAULT CONFIGURATION")
    print("="*80)
    config = get_default_config()
    print(json.dumps(config.to_dict(), indent=2))
