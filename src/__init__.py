"""
Crystallized Safety - Core Library

Tools for investigating the relationship between readable safety
representations and controllable behavior in LLMs.
"""

from .models import (
    ModelLoader,
    ModelConfig,
    MODEL_REGISTRY,
    get_model_info,
    list_supported_models,
    get_steerable_models,
    get_crystallized_models,
)

from .steering import (
    ResidualSteeringHook,
    ActivationSteerer,
    generate_random_directions,
    generate_orthogonal_directions,
    rotate_vector,
)

from .extraction import (
    SteeringVectorExtractor,
    PCAExtractor,
    save_steering_vector,
    load_steering_vector,
)

from .evaluation import (
    GPT4Judge,
    KeywordRefusalDetector,
    Verdict,
    FinalVerdict,
    GateResult,
    compute_flip_rate,
    bootstrap_confidence_interval,
    evaluate_direction_specificity,
    evaluate_coherence,
    compute_final_verdict,
)

__version__ = "0.1.0"
__all__ = [
    # Models
    "ModelLoader",
    "ModelConfig",
    "MODEL_REGISTRY",
    "get_model_info",
    "list_supported_models",
    "get_steerable_models",
    "get_crystallized_models",
    # Steering
    "ResidualSteeringHook",
    "ActivationSteerer",
    "generate_random_directions",
    "generate_orthogonal_directions",
    "rotate_vector",
    # Extraction
    "SteeringVectorExtractor",
    "PCAExtractor",
    "save_steering_vector",
    "load_steering_vector",
    # Evaluation
    "GPT4Judge",
    "KeywordRefusalDetector",
    "Verdict",
    "FinalVerdict",
    "GateResult",
    "compute_flip_rate",
    "bootstrap_confidence_interval",
    "evaluate_direction_specificity",
    "evaluate_coherence",
    "compute_final_verdict",
]
