"""Módulos forenses avançados: SRM (Spatial Rich Models) e DTCWT (Dual-Tree Complex Wavelets)."""
from .srm import SRMConv2d, extract_srm_residuals

try:
    from .dtcwt_module import DTCWTExtractor, extract_dtcwt_features
except (ImportError, ModuleNotFoundError):
    DTCWTExtractor = None  # type: ignore
    extract_dtcwt_features = None  # type: ignore

__all__ = ["SRMConv2d", "extract_srm_residuals", "DTCWTExtractor", "extract_dtcwt_features"]
