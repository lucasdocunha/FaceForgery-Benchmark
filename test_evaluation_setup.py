#!/usr/bin/env python3
"""
Quick test to verify model discovery and dataset loading works correctly.
Run this first before full evaluation.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_discovery():
    """Test if we can discover models in the structure."""
    logger.info("Testing model discovery...")
    
    models_dir = Path("models")
    architectures = {
        "resnet": "resnet",
        "mobilenet": "mobilenet/mobilenetv3_large",
        "xception": "xception",
        "vit": "vit/vit_scratch",
    }
    
    fourier_modes = [
        "none", "magnitude", "phase", "complex", 
        "concat", "frequency_3", "concat_frequency"
    ]
    
    discovered = {}
    for arch_name, arch_path in architectures.items():
        arch_dir = models_dir / arch_path
        logger.info(f"Checking: {arch_dir}")
        
        if not arch_dir.exists():
            logger.warning(f"  NOT FOUND")
            continue
        
        discovered[arch_name] = []
        for mode in fourier_modes:
            mode_dir = arch_dir / mode
            weights_dir = mode_dir / "weights"
            
            if weights_dir.exists():
                weights = list(weights_dir.glob("*.pth"))
                if weights:
                    discovered[arch_name].append((mode, weights[0]))
                    logger.info(f"  ✓ {arch_name:15s} {mode:20s} → {weights[0].name}")
    
    logger.info(f"\nDiscovered {sum(len(v) for v in discovered.values())} model variants")
    return discovered

def test_dataset_loading():
    """Test if we can load datasets."""
    logger.info("\nTesting dataset loading...")
    
    datasets = {
        "test_raw": "data/raw/test.csv",
        "test_raw_min": "data/raw_min/test.csv",
        "val_raw": "data/raw/val.csv",
    }
    
    for name, csv_path in datasets.items():
        p = Path(csv_path)
        if p.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            logger.info(f"  ✓ {name:15s} → {len(df):5d} images")
        else:
            logger.warning(f"  ✗ {name:15s} NOT FOUND at {csv_path}")

def test_imports():
    """Test if all imports work."""
    logger.info("\nTesting imports...")
    
    try:
        from data.data import ImageDataset, FourierMode
        logger.info("  ✓ ImageDataset, FourierMode")
    except Exception as e:
        logger.error(f"  ✗ Failed to import from data.data: {e}")
        return False
    
    try:
        from data.paths import phase1_split_root
        logger.info("  ✓ phase1_split_root")
    except Exception as e:
        logger.error(f"  ✗ Failed to import phase1_split_root: {e}")
        return False
    
    try:
        from pipelines.evaluation import evaluate_classifier
        logger.info("  ✓ evaluate_classifier")
    except Exception as e:
        logger.error(f"  ✗ Failed to import evaluate_classifier: {e}")
        return False
    
    try:
        from models.resnet import create_resnet_model
        logger.info("  ✓ create_resnet_model")
    except Exception as e:
        logger.error(f"  ✗ Failed to import model creation functions: {e}")
        return False
    
    return True

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("PRE-EVALUATION CHECKS")
    logger.info("="*80)
    
    if not test_imports():
        logger.error("\nImports failed. Fix issues before running full evaluation.")
        sys.exit(1)
    
    test_dataset_loading()
    test_model_discovery()
    
    logger.info("\n" + "="*80)
    logger.info("Pre-checks complete. Ready to run full evaluation.")
    logger.info("Run: python evaluate_all_models.py")
    logger.info("="*80)
