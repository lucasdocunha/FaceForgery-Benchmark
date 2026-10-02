"""Compact VLM SFT and audited continuous fake-probability inference."""

from .inference import load_predictor
from .training import fit, input_contract

__all__ = ["fit", "load_predictor", "input_contract"]
