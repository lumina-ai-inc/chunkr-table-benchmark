"""
Processor modules for table extraction from various APIs.

This package contains individual processor implementations for:
- UnstructuredProcessor: Unstructured.io API
- AzureDocumentAnalysisProcessor: Azure Document Intelligence
- OpenRouterTableProcessor: OpenRouter API with various LLM models
- MathpixProcessor: Mathpix OCR with LaTeX support
- MistralProcessor: Mistral OCR API
- TextractProcessor: Amazon Textract
- ChunkrDefaultProcessor: Chunkr API (default variant)

Each processor implements a common interface:
- __init__(model): Initialize with model/configuration
- process_document(file_path): Process document and return HTML table
"""

from .azure_processor import AzureDocumentAnalysisProcessor
from .chunkr_processor import ChunkrDefaultProcessor
from .mathpix_processor import MathpixProcessor
from .mistral_processor import MistralProcessor
from .openrouter_processor import OpenRouterTableProcessor
from .textract_processor import TextractProcessor
from .unstructured_processor import UnstructuredProcessor

__all__ = [
    'UnstructuredProcessor',
    'AzureDocumentAnalysisProcessor', 
    'OpenRouterTableProcessor',
    'MathpixProcessor',
    'MistralProcessor',
    'TextractProcessor',
    'ChunkrDefaultProcessor',
]

# Model configuration examples
PROCESSOR_CONFIGS = {
    "openrouter": {
        "models": [
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "openai/gpt-4o-mini"
        ],
        "processor_class": OpenRouterTableProcessor
    },
    "azure": {
        "models": ["prebuilt-layout"],
        "processor_class": AzureDocumentAnalysisProcessor
    },
    "mathpix": {
        "models": ["mathpix"],
        "processor_class": MathpixProcessor
    },
    "mistral": {
        "models": ["mistral-ocr-latest"],
        "processor_class": MistralProcessor
    },
    "textract": {
        "models": ["default"],
        "processor_class": TextractProcessor
    },
    "unstructured": {
        "models": ["hi_res"],
        "processor_class": UnstructuredProcessor
    },
    "chunkr": {
        "models": ["table", "default", "layout"],
        "processor_classes": {
            "default": ChunkrDefaultProcessor,
        }
    }
}