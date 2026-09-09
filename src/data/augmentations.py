"""Módulo de pré-processamento e aumentações para imagens.

Inclui pipeline dinâmico e aleatorizado de augmentação robusta (RandomizedRobustAugment)
para avaliação e treino de robustez contra perturbações severas (OOD).
"""

from __future__ import annotations

import io
import random
from typing import Tuple

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms


class RandomizedRobustAugment:
    """Pipeline dinâmico de pré-processamento com parâmetros aleatorizados por imagem.

    Simula perturbações comuns encontradas no mundo real e no split test_d:
    - Rotação aleatória (-30° a +30°)
    - Ajuste de contraste dinâmico (fator 0.4 a 1.8)
    - Superexposição / brilho dinâmico (fator 0.5 a 1.8)
    - Nitidez dinâmica (fator 0.2 a 2.0)
    - Blur Gaussiano estocástico (raio 0.5 a 2.0)
    - Compressão JPEG com fator de qualidade aleatório (Q 25 a 90)
    - Ruído Gaussiano aditivo (sigma 0.01 a 0.08)
    - Corte redimensionado e flip horizontal
    """

    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.image_size = image_size
        self.crop = transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))
        self.hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.normalize = transforms.Normalize(mean=list(mean), std=list(std))

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1. Transformações espaciais base
        img = self.crop(img)
        img = self.hflip(img)

        # 2. Rotação aleatória (-30 a +30 graus)
        if random.random() < 0.5:
            angle = random.uniform(-30.0, 30.0)
            img = img.rotate(angle, resample=Image.BILINEAR)

        # 3. Contraste dinâmico
        if random.random() < 0.5:
            c = random.uniform(0.4, 1.8)
            img = ImageEnhance.Contrast(img).enhance(c)

        # 4. Superexposição / Brilho dinâmico
        if random.random() < 0.5:
            b = random.uniform(0.5, 1.8)
            img = ImageEnhance.Brightness(img).enhance(b)

        # 5. Nitidez
        if random.random() < 0.3:
            s = random.uniform(0.2, 2.0)
            img = ImageEnhance.Sharpness(img).enhance(s)

        # 6. Desfoque Gaussiano estocástico
        if random.random() < 0.3:
            r = random.uniform(0.5, 2.0)
            img = img.filter(ImageFilter.GaussianBlur(r))

        # 7. Compressão JPEG em memória com fator de qualidade variável
        if random.random() < 0.5:
            q = random.randint(25, 90)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        # Conversão para Tensor
        tensor = transforms.ToTensor()(img)

        # 8. Ruído Gaussiano aditivo aleatório
        if random.random() < 0.5:
            sigma = random.uniform(0.01, 0.08)
            tensor = (tensor + torch.randn_like(tensor) * sigma).clamp(0.0, 1.0)

        # 9. Normalização ImageNet
        return self.normalize(tensor)


def clean_transform(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Transformação padrão sem aumentações estocásticas (usada para Val, Test e Test_d)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])
