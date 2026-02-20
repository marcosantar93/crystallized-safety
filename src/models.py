"""
Model loading utilities for crystallized-safety experiments.

Handles loading and configuration of various LLM families with optional
quantization for memory efficiency.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    name: str
    num_layers: int
    hidden_size: int
    recommended_layer: int  # Best layer for steering based on experiments
    family: str  # llama, mistral, qwen, gemma, phi
    steerable: bool = True
    notes: str = ""


# Model registry with experimental findings
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ModelConfig(
        name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        num_layers=32,
        hidden_size=4096,
        recommended_layer=15,
        family="llama",
        steerable=True,
        notes="Best steering target in our experiments"
    ),
    "meta-llama/Meta-Llama-3.1-70B-Instruct": ModelConfig(
        name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        num_layers=80,
        hidden_size=8192,
        recommended_layer=40,
        family="llama",
        steerable=True,
        notes="Layer-dependent steering; partial effectiveness"
    ),
    "mistralai/Mistral-7B-Instruct-v0.2": ModelConfig(
        name="mistralai/Mistral-7B-Instruct-v0.2",
        num_layers=32,
        hidden_size=4096,
        recommended_layer=16,
        family="mistral",
        steerable=False,
        notes="Crystallized safety - readable but not steerable"
    ),
    "Qwen/Qwen2.5-7B-Instruct": ModelConfig(
        name="Qwen/Qwen2.5-7B-Instruct",
        num_layers=28,
        hidden_size=3584,
        recommended_layer=14,
        family="qwen",
        steerable=False,
        notes="Similar crystallization pattern to Mistral"
    ),
    "google/gemma-2-9b-it": ModelConfig(
        name="google/gemma-2-9b-it",
        num_layers=42,
        hidden_size=3584,
        recommended_layer=21,
        family="gemma",
        steerable=True,
        notes="Layer 21 shows specific vulnerability (glass jaw)"
    ),
    "microsoft/Phi-3-mini-4k-instruct": ModelConfig(
        name="microsoft/Phi-3-mini-4k-instruct",
        num_layers=32,
        hidden_size=3072,
        recommended_layer=16,
        family="phi",
        steerable=False,
        notes="Shows floor effects masking steering"
    ),
}


def lazy_import_transformers():
    """Lazy import to avoid loading transformers until needed."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    return AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ModelLoader:
    """Handles loading models with various configurations."""

    def __init__(
        self,
        model_name: str,
        use_8bit: bool = True,
        use_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16
    ):
        self.model_name = model_name
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.device_map = device_map
        self.torch_dtype = torch_dtype

        self.model = None
        self.tokenizer = None
        self.config = MODEL_REGISTRY.get(model_name)

    def load(self) -> tuple:
        """Load model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = lazy_import_transformers()

        logger.info(f"Loading {self.model_name}...")

        # Setup quantization
        quantization_config = None
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )

        # Load model
        load_kwargs = {
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
        }
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        num_layers = self.model.config.num_hidden_layers
        logger.info(f"Model loaded. Layers: {num_layers}, Device: {self.model.device}")

        return self.model, self.tokenizer

    def get_layer_module(self, layer_idx: int):
        """Get the module for a specific layer.

        Works across different model architectures.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Handle different architectures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError(f"Unknown model architecture for {self.model_name}")

    @property
    def num_layers(self) -> int:
        """Get number of layers in the model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        """Get hidden size of the model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.config.hidden_size


def get_model_info(model_name: str) -> Optional[ModelConfig]:
    """Get information about a model from the registry."""
    return MODEL_REGISTRY.get(model_name)


def list_supported_models() -> list:
    """List all supported models."""
    return list(MODEL_REGISTRY.keys())


def get_steerable_models() -> list:
    """Get list of models that show steering effects."""
    return [name for name, config in MODEL_REGISTRY.items() if config.steerable]


def get_crystallized_models() -> list:
    """Get list of models showing crystallized safety (readable but not steerable)."""
    return [name for name, config in MODEL_REGISTRY.items() if not config.steerable]


def load_model_and_tokenizer(
    model_name: str,
    *,
    use_8bit: bool = True,
    use_4bit: bool = False,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
):
    """Backward-compatible helper used by older scripts.

    Prefer using `ModelLoader` directly in new code.
    """
    loader = ModelLoader(
        model_name=model_name,
        use_8bit=use_8bit,
        use_4bit=use_4bit,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return loader.load()
