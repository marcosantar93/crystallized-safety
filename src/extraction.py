"""
Steering vector extraction for crystallized-safety experiments.

This module implements methods for extracting steering vectors from model
activations using contrastive pairs (harmful vs harmless prompts).
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Callable

import torch
from tqdm import tqdm

from .models import ModelLoader

logger = logging.getLogger(__name__)


class SteeringVectorExtractor:
    """Extract steering vectors using the mean difference method.

    This implements Contrastive Activation Addition (CAA) style extraction:
    steering_vector = mean(harmful_activations) - mean(harmless_activations)

    The resulting vector represents the "direction" in activation space
    associated with the safety/refusal concept.

    Example:
        >>> extractor = SteeringVectorExtractor("meta-llama/Meta-Llama-3.1-8B-Instruct")
        >>> extractor.load_model()
        >>> vector = extractor.extract(harmful_prompts, harmless_prompts, layer=15)
        >>> torch.save(vector, "refusal_direction.pt")
    """

    def __init__(
        self,
        model_name: str,
        use_8bit: bool = True
    ):
        self.model_name = model_name
        self.use_8bit = use_8bit
        self.loader = None
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer."""
        self.loader = ModelLoader(
            self.model_name,
            use_8bit=self.use_8bit
        )
        self.model, self.tokenizer = self.loader.load()

    def _get_activation(
        self,
        prompt: str,
        layer: int,
        position: int = -1
    ) -> torch.Tensor:
        """Get activation at specific layer and position for a prompt.

        Args:
            prompt: Input prompt
            layer: Layer index to extract from
            position: Token position (-1 = last token)

        Returns:
            Activation tensor of shape (hidden_size,)
        """
        # Format prompt using chat template
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        activations = []

        def hook(module, input, output):
            if isinstance(output, tuple):
                act = output[0][:, position, :].detach().cpu()
            else:
                act = output[:, position, :].detach().cpu()
            activations.append(act)

        # Get layer module
        layer_module = self.loader.get_layer_module(layer)
        handle = layer_module.register_forward_hook(hook)

        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            handle.remove()

        return activations[0].squeeze(0)

    def extract(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layer: int,
        normalize: bool = True,
        show_progress: bool = True
    ) -> torch.Tensor:
        """Extract steering vector using mean difference method.

        Args:
            harmful_prompts: List of prompts that should trigger refusal
            harmless_prompts: List of benign prompts
            layer: Layer to extract from
            normalize: Whether to normalize the resulting vector
            show_progress: Whether to show progress bar

        Returns:
            Steering vector tensor of shape (hidden_size,)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(f"Extracting steering vector at layer {layer}...")
        logger.info(f"Using {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts")

        # Extract harmful activations
        harmful_acts = []
        iterator = tqdm(harmful_prompts, desc="Harmful") if show_progress else harmful_prompts
        for prompt in iterator:
            act = self._get_activation(prompt, layer)
            harmful_acts.append(act)

        # Extract harmless activations
        harmless_acts = []
        iterator = tqdm(harmless_prompts, desc="Harmless") if show_progress else harmless_prompts
        for prompt in iterator:
            act = self._get_activation(prompt, layer)
            harmless_acts.append(act)

        # Compute mean difference
        harmful_mean = torch.stack(harmful_acts).mean(dim=0)
        harmless_mean = torch.stack(harmless_acts).mean(dim=0)
        steering_vector = harmful_mean - harmless_mean

        if normalize:
            steering_vector = steering_vector / steering_vector.norm()

        logger.info(f"Steering vector extracted. Shape: {steering_vector.shape}")
        logger.info(f"Norm: {steering_vector.norm():.4f}")

        return steering_vector

    def extract_all_layers(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layers: Optional[List[int]] = None,
        normalize: bool = True,
        show_progress: bool = True
    ) -> Dict[int, torch.Tensor]:
        """Extract steering vectors from multiple layers.

        Args:
            harmful_prompts: List of prompts that should trigger refusal
            harmless_prompts: List of benign prompts
            layers: List of layers to extract from (default: all layers)
            normalize: Whether to normalize vectors
            show_progress: Whether to show progress bar

        Returns:
            Dict mapping layer index to steering vector
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if layers is None:
            layers = list(range(self.loader.num_layers))

        vectors = {}
        for layer in tqdm(layers, desc="Layers", disable=not show_progress):
            vectors[layer] = self.extract(
                harmful_prompts,
                harmless_prompts,
                layer,
                normalize=normalize,
                show_progress=False
            )

        return vectors


class PCAExtractor:
    """Extract steering vectors using PCA on activation differences.

    An alternative to mean difference that can capture multiple relevant
    directions in the activation space.
    """

    def __init__(
        self,
        model_name: str,
        use_8bit: bool = True
    ):
        self.model_name = model_name
        self.use_8bit = use_8bit
        self.loader = None
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer."""
        self.loader = ModelLoader(
            self.model_name,
            use_8bit=self.use_8bit
        )
        self.model, self.tokenizer = self.loader.load()

    def extract(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layer: int,
        n_components: int = 5,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract top PCA components of activation differences.

        Args:
            harmful_prompts: List of harmful prompts
            harmless_prompts: List of harmless prompts
            layer: Layer to extract from
            n_components: Number of PCA components to return
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (components, explained_variance_ratio)
            - components: Shape (n_components, hidden_size)
            - explained_variance: Shape (n_components,)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Use the mean difference extractor for getting activations
        extractor = SteeringVectorExtractor.__new__(SteeringVectorExtractor)
        extractor.model = self.model
        extractor.tokenizer = self.tokenizer
        extractor.loader = self.loader

        # Collect paired differences
        n_pairs = min(len(harmful_prompts), len(harmless_prompts))
        differences = []

        iterator = range(n_pairs)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting pairs")

        for i in iterator:
            harmful_act = extractor._get_activation(harmful_prompts[i], layer)
            harmless_act = extractor._get_activation(harmless_prompts[i], layer)
            differences.append(harmful_act - harmless_act)

        # Stack and compute PCA
        diff_matrix = torch.stack(differences)  # (n_pairs, hidden_size)

        # Center the data
        diff_centered = diff_matrix - diff_matrix.mean(dim=0)

        # SVD for PCA
        U, S, Vh = torch.linalg.svd(diff_centered, full_matrices=False)

        # Get top components
        components = Vh[:n_components]  # (n_components, hidden_size)
        total_var = (S ** 2).sum()
        explained_variance = (S[:n_components] ** 2) / total_var

        logger.info(f"PCA extraction complete. Top {n_components} components explain "
                    f"{explained_variance.sum():.1%} of variance")

        return components, explained_variance


def save_steering_vector(
    vector: torch.Tensor,
    path: str,
    metadata: Optional[Dict] = None
):
    """Save steering vector with optional metadata.

    Args:
        vector: Steering vector to save
        path: Output path (.pt file)
        metadata: Optional metadata dict to save alongside
    """
    save_dict = {"vector": vector}
    if metadata:
        save_dict["metadata"] = metadata

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, path)
    logger.info(f"Saved steering vector to {path}")


def load_steering_vector(path: str) -> Tuple[torch.Tensor, Optional[Dict]]:
    """Load steering vector and optional metadata.

    Args:
        path: Path to .pt file

    Returns:
        Tuple of (vector, metadata) where metadata may be None
    """
    data = torch.load(path, map_location='cpu')

    if isinstance(data, dict) and "vector" in data:
        return data["vector"], data.get("metadata")
    else:
        # Legacy format: just the tensor
        return data, None
