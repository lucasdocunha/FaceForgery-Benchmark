"""
Comprehensive model evaluation script.
Loads all trained models from models/{architecture}/{fourier_mode}/weights/
Evaluates on 3 test datasets and generates unified metrics CSV.
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.data import ImageDataset, FourierMode
from data.paths import phase1_split_root
from pipelines.evaluation import evaluate_classifier
from models.resnet import create_resnet_model
from models.mobilenet import create_mobilenet_model
from models.xception import create_xception_model
from models.vit import create_vit_model
from models.clip import create_clip_model
from torch.utils.data import DataLoader
from torchvision import transforms


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Orchestrates model discovery, loading, evaluation, and metrics aggregation."""
    
    def __init__(self, models_dir: str = "models", device: str = "cuda"):
        self.models_dir = Path(models_dir)
        self.device = device
        self.results = []
        
        # Define model architectures and their factories
        self.model_factories = {
            "resnet": self._create_resnet,
            "mobilenet": self._create_mobilenet,
            "xception": self._create_xception,
            "vit": self._create_vit,
            "clip": self._create_clip,
        }
        
        # Define Fourier modes and their input channels
        self.fourier_modes = {
            "none": 3,          # RGB only
            "magnitude": 1,     # FFT magnitude
            "phase": 1,         # FFT phase
            "complex": 2,       # Real + imaginary
            "concat": 4,        # RGB + magnitude
            "frequency_3": 1,   # Magnitude with high-pass
            "concat_frequency": 6,  # RGB + magnitude + phase + high-pass
        }
        
        # Test datasets configuration
        self.test_datasets = {
            "test_raw": "data/raw/test.csv",
            "test_raw_min": "data/raw_min/test.csv",
            "val_raw": "data/raw/val.csv",
        }
    
    def discover_models(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Discover all available trained models.
        Returns: {architecture: [(fourier_mode, weights_path, model_variant), ...]}
        """
        discovered = {}
        
        for arch in self.model_factories.keys():
            arch_dir = self.models_dir / arch
            
            # Handle special cases for nested architectures
            if arch == "mobilenet":
                arch_dir = arch_dir / "mobilenetv3_large"
            elif arch == "vit":
                arch_dir = arch_dir / "vit_scratch"
            
            if not arch_dir.exists():
                logger.warning(f"Architecture directory not found: {arch_dir}")
                continue
            
            discovered[arch] = []
            
            # Search for Fourier mode subdirectories
            for mode_dir in arch_dir.iterdir():
                if not mode_dir.is_dir():
                    continue
                
                fourier_mode = mode_dir.name
                
                # Skip if not a known Fourier mode
                if fourier_mode not in self.fourier_modes:
                    continue
                
                weights_dir = mode_dir / "weights"
                if not weights_dir.exists():
                    logger.warning(f"Weights directory not found: {weights_dir}")
                    continue
                
                # Look for best model weights
                best_weights = list(weights_dir.glob("best_*.pth"))
                if not best_weights:
                    best_weights = list(weights_dir.glob("*.pth"))
                
                if not best_weights:
                    logger.warning(f"No model weights found in: {weights_dir}")
                    continue
                
                weights_path = str(best_weights[0])
                model_variant = f"{arch}_{fourier_mode}"
                discovered[arch].append((fourier_mode, weights_path, model_variant))
                logger.info(f"Discovered: {model_variant} at {weights_path}")
        
        return discovered
    
    def load_test_datasets_for_fourier(self, fourier_mode: FourierMode) -> Dict[str, DataLoader]:
        """
        Load all three test datasets as DataLoaders for a specific Fourier mode.
        Returns: {dataset_name: DataLoader}
        """
        datasets = {}
        
        # Determine split mapping and data directory based on fourier mode
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        spatial_size = (224, 224) if fourier_mode != "none" else None
        
        dataset_configs = {
            "test_raw": {
                "csv_path": Path("data/raw/test.csv"),
                "split": "test",
                "raw_min": False,
            },
            "test_raw_min": {
                "csv_path": Path("data/raw_min/test.csv"),
                "split": "test",
                "raw_min": True,
            },
            "val_raw": {
                "csv_path": Path("data/raw/val.csv"),
                "split": "val",
                "raw_min": False,
            },
        }
        
        for dataset_name, config in dataset_configs.items():
            csv_path = config["csv_path"]
            
            if not csv_path.exists():
                logger.warning(f"Dataset CSV not found: {csv_path}")
                continue
            
            try:
                # Determine images directory based on raw_min flag
                if config["raw_min"]:
                    if config["split"] == "test":
                        images_dir = Path("min_dataset/test") if Path("min_dataset/test").exists() else phase1_split_root("test")
                    elif config["split"] == "val":
                        images_dir = Path("min_dataset/val") if Path("min_dataset/val").exists() else phase1_split_root("val")
                    else:
                        images_dir = phase1_split_root(config["split"])
                else:
                    images_dir = phase1_split_root(config["split"])
                
                dataset = ImageDataset(
                    file_csv=csv_path,
                    images_dir=images_dir,
                    transform=eval_transform,
                    fourier=fourier_mode,
                    spatial_size=spatial_size,
                )
                
                loader = DataLoader(
                    dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                )
                
                datasets[dataset_name] = loader
                logger.info(f"Loaded dataset: {dataset_name} ({len(dataset)} images) for Fourier mode: {fourier_mode}")
                
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
                continue
        
        return datasets
    
    def _create_resnet(self, fourier_mode: str, weights_path: str) -> Optional[torch.nn.Module]:
        """Create and load ResNet model."""
        try:
            in_channels = self.fourier_modes.get(fourier_mode, 3)
            model = create_resnet_model(
                arch="resnet18",
                in_channels=in_channels,
                num_classes=2,
                pretrained=False
            )
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            return model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to create ResNet model: {e}")
            return None
    
    def _create_mobilenet(self, fourier_mode: str, weights_path: str) -> Optional[torch.nn.Module]:
        """Create and load MobileNet model."""
        try:
            in_channels = self.fourier_modes.get(fourier_mode, 3)
            model = create_mobilenet_model(
                model_name="mobilenetv3_large",
                in_channels=in_channels,
                num_classes=2,
                pretrained=False
            )
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            return model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to create MobileNet model: {e}")
            return None
    
    def _create_xception(self, fourier_mode: str, weights_path: str) -> Optional[torch.nn.Module]:
        """Create and load Xception model."""
        try:
            in_channels = self.fourier_modes.get(fourier_mode, 3)
            model = create_xception_model(
                in_channels=in_channels,
                num_classes=2
            )
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            return model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to create Xception model: {e}")
            return None
    
    def _create_vit(self, fourier_mode: str, weights_path: str) -> Optional[torch.nn.Module]:
        """Create and load ViT model."""
        try:
            in_channels = self.fourier_modes.get(fourier_mode, 3)
            model = create_vit_model(
                image_size=224,
                patch_size=16,
                in_channels=in_channels,
                num_classes=2,
                num_layers=12,
                hidden_size=768,
                num_attention_heads=12
            )
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            return model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to create ViT model: {e}")
            return None
    
    def _create_clip(self, fourier_mode: str, weights_path: str) -> Optional[torch.nn.Module]:
        """Create and load CLIP model."""
        try:
            in_channels = self.fourier_modes.get(fourier_mode, 3)
            model = create_clip_model(
                image_size=224,
                patch_size=16,
                in_channels=in_channels,
                num_classes=2,
                hidden_size=768,
                projection_dim=512,
                num_layers=12,
                num_attention_heads=12
            )
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            return model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to create CLIP model: {e}")
            return None
    
    def evaluate_model_on_dataset(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        dataset_name: str,
        model_variant: str,
        fourier_mode: str,
        architecture: str
    ) -> Optional[Dict]:
        """
        Evaluate a single model on a dataset.
        Returns dictionary with metrics.
        """
        try:
            criterion = nn.BCEWithLogitsLoss()
            
            metrics = evaluate_classifier(
                model=model,
                loader=dataloader,
                criterion=criterion,
                device=self.device,
                threshold=0.5,
                use_amp=True,
                desc=f"{model_variant} on {dataset_name}"
            )
            
            # Extract confusion matrix values
            tn = metrics.get("tn", 0)
            fp = metrics.get("fp", 0)
            fn = metrics.get("fn", 0)
            tp = metrics.get("tp", 0)
            
            result = {
                "architecture": architecture,
                "fourier_mode": fourier_mode,
                "model_variant": model_variant,
                "dataset": dataset_name,
                "accuracy": metrics.get("acc", np.nan),
                "precision": metrics.get("precision", np.nan),
                "recall": metrics.get("recall", np.nan),
                "f1": metrics.get("f1", np.nan),
                "auc": metrics.get("auc", np.nan),
                "specificity": metrics.get("specificity", np.nan),
                "sensitivity": metrics.get("recall", np.nan),
                "loss": metrics.get("loss", np.nan),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "optimal_threshold": 0.5,  # Using 0.5 for consistency
            }
            
            logger.info(
                f"✓ {model_variant:40s} on {dataset_name:15s}: "
                f"Acc={result['accuracy']:.4f}, F1={result['f1']:.4f}, AUC={result['auc']:.4f}"
            )
            return result
            
        except Exception as e:
            logger.error(f"✗ Failed to evaluate {model_variant} on {dataset_name}: {e}", exc_info=True)
            return None
    
    def run_evaluation(self) -> pd.DataFrame:
        """
        Main evaluation loop.
        Discover models -> Load datasets -> Evaluate each model on each dataset -> Aggregate results.
        """
        logger.info("="*80)
        logger.info("Starting comprehensive model evaluation")
        logger.info("="*80)
        
        # Phase 1: Discover models
        logger.info("\n[Phase 1] Discovering models...")
        discovered_models = self.discover_models()
        total_models = sum(len(models) for models in discovered_models.values())
        logger.info(f"Discovered {total_models} trained models across {len(discovered_models)} architectures")
        
        if total_models == 0:
            logger.error("No models discovered. Check models directory structure.")
            return pd.DataFrame()
        
        # Phase 2 & 3: Evaluate each model on each dataset
        logger.info("\n[Phase 2-3] Loading datasets and evaluating models...")
        total_datasets = len(self.test_datasets)
        total_evaluations = total_models * total_datasets
        
        with tqdm(total=total_evaluations, desc="Total evaluations") as pbar:
            for architecture, models in discovered_models.items():
                for fourier_mode, weights_path, model_variant in models:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"Processing: {model_variant}")
                    logger.info(f"{'='*80}")
                    
                    # Load datasets for this Fourier mode
                    logger.info(f"  Loading datasets for Fourier mode: {fourier_mode}")
                    test_datasets = self.load_test_datasets_for_fourier(fourier_mode)
                    
                    if len(test_datasets) == 0:
                        logger.warning(f"  No test datasets loaded for {fourier_mode}")
                        pbar.update(total_datasets)
                        continue
                    
                    # Load model once for all datasets
                    logger.info(f"  Loading model weights from: {weights_path}")
                    factory = self.model_factories.get(architecture)
                    if not factory:
                        logger.warning(f"  Unknown architecture: {architecture}")
                        pbar.update(len(test_datasets))
                        continue
                    
                    model = factory(fourier_mode, weights_path)
                    if model is None:
                        logger.warning(f"  Failed to load model: {model_variant}")
                        pbar.update(len(test_datasets))
                        continue
                    
                    model.eval()
                    
                    # Evaluate on each dataset
                    for dataset_name, dataloader in test_datasets.items():
                        result = self.evaluate_model_on_dataset(
                            model=model,
                            dataloader=dataloader,
                            dataset_name=dataset_name,
                            model_variant=model_variant,
                            fourier_mode=fourier_mode,
                            architecture=architecture
                        )
                        
                        if result:
                            self.results.append(result)
                        
                        pbar.update(1)
                    
                    # Clean up model to free GPU memory
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Phase 4: Aggregate results into DataFrame
        logger.info("\n[Phase 4] Aggregating results...")
        results_df = pd.DataFrame(self.results)
        
        if len(results_df) == 0:
            logger.error("No evaluation results collected.")
            return pd.DataFrame()
        
        # Sort for better readability
        results_df = results_df.sort_values(
            by=["architecture", "fourier_mode", "dataset"]
        ).reset_index(drop=True)
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame, output_path: str = "models/all_models_metrics_unified.csv"):
        """Save unified metrics to CSV."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Results saved to: {output_path}")
        logger.info(f"Total evaluations: {len(results_df)}")
        
        # Print summary statistics
        self.print_summary(results_df)
    
    def print_summary(self, results_df: pd.DataFrame):
        """Print summary statistics."""
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\nTotal results: {len(results_df)}")
        logger.info(f"Architectures: {results_df['architecture'].nunique()}")
        logger.info(f"Fourier modes: {results_df['fourier_mode'].nunique()}")
        logger.info(f"Test datasets: {results_df['dataset'].nunique()}")
        
        # Best models per dataset
        logger.info("\nBest models (highest F1) per dataset:")
        for dataset in results_df['dataset'].unique():
            subset = results_df[results_df['dataset'] == dataset]
            best_idx = subset['f1'].idxmax()
            best_model = subset.loc[best_idx]
            logger.info(
                f"  {dataset}: {best_model['model_variant']} "
                f"(F1={best_model['f1']:.4f}, AUC={best_model['auc']:.4f})"
            )
        
        # Statistics by architecture
        logger.info("\nAverage metrics by architecture:")
        arch_stats = results_df.groupby('architecture')[['accuracy', 'f1', 'auc']].mean()
        logger.info(arch_stats.to_string())
        
        logger.info("\n" + "="*80)


def main():
    """Main execution function."""
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Run evaluation
    evaluator = ModelEvaluator(models_dir="models", device=device)
    results_df = evaluator.run_evaluation()
    
    if len(results_df) > 0:
        evaluator.save_results(results_df)
        return 0
    else:
        logger.error("Evaluation failed - no results generated.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
