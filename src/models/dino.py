from __future__ import annotations
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class DINOVisionClassifier(nn.Module):
    """SOTA DINO (v2 / v3) vision foundation model classifier.
    Supports ViT and ConvNeXt variants of DINOv2 and DINOv3.
    """
    def __init__(
        self,
        num_classes: int = 2,
        dino_version: str = "v3",
        model_size: str = "base",  # "tiny", "small", "base", "large"
        dropout: float = 0.2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.dino_version = dino_version.lower()
        self.model_size = model_size.lower()
        self.pretrained = pretrained
        
        # Load backbone
        self.backbone = self._load_backbone()
        
        # Determine hidden size
        self.hidden_size = self._get_hidden_size()
        
        # Supervised classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes)
        )
        
        if freeze_backbone:
            self.freeze_backbone()

    def _load_backbone(self) -> nn.Module:
        if self.dino_version == "v3":
            # State-of-the-Art DINOv3
            try:
                import timm
                size_map = {
                    "tiny": "convnext_tiny.dinov3_lvd1689m",
                    "small": "convnext_small.dinov3_lvd1689m",
                    "base": "convnext_base.dinov3_lvd1689m",
                    "large": "convnext_large.dinov3_lvd1689m",
                }
                model_name = size_map.get(self.model_size, "convnext_base.dinov3_lvd1689m")
                logger.info(f"Loading DINOv3 via timm: {model_name}")
                backbone = timm.create_model(model_name, pretrained=self.pretrained, num_classes=0)
                return backbone
            except Exception as e:
                logger.warning(
                    f"Could not load DINOv3 via timm ({e}). "
                    "Trying to load DINOv3 via Hugging Face..."
                )
                try:
                    from transformers import AutoModel
                    hf_map = {
                        "tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
                        "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
                        "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                        "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                    }
                    model_name = hf_map.get(self.model_size, "facebook/dinov3-vitb16-pretrain-lvd1689m")
                    logger.info(f"Loading DINOv3 via Hugging Face: {model_name}")
                    backbone = AutoModel.from_pretrained(model_name)
                    return backbone
                except Exception as e2:
                    logger.error(
                        f"Failed to load DINOv3 via Hugging Face ({e2}). "
                        "Because DINOv3 requires license acceptance (gated repo), "
                        "falling back to the extremely robust, open-source and ungated DINOv2..."
                    )
                    self.dino_version = "v2"
                    
        if self.dino_version == "v2":
            # DINOv2 from PyTorch Hub (extremely robust, requires no tokens)
            hub_map = {
                "tiny": "dinov2_vits14",
                "small": "dinov2_vits14",
                "base": "dinov2_vitb14",
                "large": "dinov2_vitl14",
            }
            model_name = hub_map.get(self.model_size, "dinov2_vitb14")
            logger.info(f"Loading DINOv2 via PyTorch Hub: {model_name}")
            try:
                backbone = torch.hub.load('facebookresearch/dinov2', model_name)
                return backbone
            except Exception as e:
                logger.error(f"Error loading DINOv2 via PyTorch Hub ({e}). Trying to load via timm...")
                try:
                    import timm
                    timm_map = {
                        "tiny": "vit_tiny_patch14_dinov2",
                        "small": "vit_small_patch14_dinov2",
                        "base": "vit_base_patch14_dinov2",
                        "large": "vit_large_patch14_dinov2",
                    }
                    model_name = timm_map.get(self.model_size, "vit_base_patch14_dinov2")
                    backbone = timm.create_model(model_name, pretrained=self.pretrained, num_classes=0)
                    return backbone
                except Exception as e3:
                    raise RuntimeError(
                        f"Could not load DINOv2 or DINOv3. Check your internet connection. Error: {e3}"
                    ) from e3

    def _get_hidden_size(self) -> int:
        if hasattr(self.backbone, "num_features"):
            return self.backbone.num_features
        elif hasattr(self.backbone, "embed_dim"):
            return self.backbone.embed_dim
        elif hasattr(self.backbone, "config") and hasattr(self.backbone.config, "hidden_size"):
            return self.backbone.config.hidden_size
        else:
            size_map = {"tiny": 384, "small": 384, "base": 768, "large": 1024}
            return size_map.get(self.model_size, 768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check channels: DINO expects 3 channels
        if x.shape[1] != 3:
            raise ValueError(f"DINO expects 3 input channels (RGB), but got {x.shape[1]} channels.")
            
        # Extract features
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
            if len(features.shape) == 4:  # [B, C, H, W] for ConvNeXt
                features = features.mean(dim=[2, 3])
            elif len(features.shape) == 3:  # [B, N, C] for ViT
                features = features[:, 0]
        else:
            # AutoModel output
            out = self.backbone(x)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                features = out.pooler_output
            else:
                features = out.last_hidden_state[:, 0]
                
        return self.classifier(features)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n: int = 2) -> None:
        # Allow partial fine-tuning of the last n layers
        if hasattr(self.backbone, "stages"):  # ConvNeXt stage unfreezing
            stages = self.backbone.stages
            n = max(0, min(n, len(stages)))
            for stage in stages[-n:]:
                for param in stage.parameters():
                    param.requires_grad = True
        elif hasattr(self.backbone, "blocks"):  # ViT block unfreezing
            blocks = self.backbone.blocks
            n = max(0, min(n, len(blocks)))
            for block in blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
        else:
            # Standard generic unfreeze fallback
            logger.info("Unfreezing entire backbone as layer-specific unfreezing is not supported for this architecture.")
            for param in self.backbone.parameters():
                param.requires_grad = True
