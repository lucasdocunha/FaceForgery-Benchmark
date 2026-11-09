"""Experiments A-D: genuine reconstruction and spatial/latent detection."""

from .losses import BetaSchedule, CompositeReconstructionLoss, OfflineLPIPS, structural_similarity
from .models import (
    AutoencoderConfig, FaceAutoencoder, LatentDetector, MeanReconstructionEnsemble,
    ReconstructionAnomaly, ResidualDetector, ResidualFusionBlock, analytical_kl,
)
from .training import build_model, combine, describe_run, fit, load_model, normalize_config, predict, run_contract

__all__ = [
    "AutoencoderConfig", "FaceAutoencoder", "LatentDetector", "MeanReconstructionEnsemble",
    "ReconstructionAnomaly", "ResidualDetector", "ResidualFusionBlock", "analytical_kl",
    "BetaSchedule", "CompositeReconstructionLoss", "OfflineLPIPS", "structural_similarity",
    "build_model", "combine", "describe_run", "fit", "load_model", "normalize_config", "predict", "run_contract",
]
