from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_mha_modules(model: nn.Module) -> list[nn.MultiheadAttention]:
    return [m for m in model.modules() if isinstance(m, nn.MultiheadAttention)]


def _patch_mha(mha: nn.MultiheadAttention, collector: list) -> object:
    """Replace mha.forward with a version that forces need_weights=True."""
    orig = mha.forward

    def patched(query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None, **kwargs):
        kwargs.pop("average_attn_weights", None)
        try:
            out, weights = orig(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                attn_mask=attn_mask,
                average_attn_weights=True,
            )
        except TypeError:
            out, weights = orig(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                attn_mask=attn_mask,
            )
        if weights is not None:
            collector.append(weights.detach().cpu())
        return out, weights

    mha.forward = patched
    return orig


def attention_rollout_heatmap(
    model: nn.Module,
    image_tensor: torch.Tensor,
) -> torch.Tensor:
    """Compute an Attention Rollout heatmap for ViT / CLIP models.

    Returns a normalized [H, W] tensor in [0, 1], same spatial size as the
    input image. Raises ValueError for models without MultiheadAttention.
    """
    base = model.module if isinstance(model, nn.DataParallel) else model
    mha_modules = _find_mha_modules(base)
    if not mha_modules:
        raise ValueError(
            "No MultiheadAttention found — model does not support Attention Rollout."
        )

    collector: list[torch.Tensor] = []
    originals = [(mha, _patch_mha(mha, collector)) for mha in mha_modules]

    # PyTorch ≥ 2.0 TransformerEncoderLayer has a C++ fast path that completely
    # bypasses Python-level MHA calls. Disabling it forces the slow Python path
    # so our patched forward is actually invoked.
    try:
        fastpath_was_enabled = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        _fastpath_patched = True
    except AttributeError:
        _fastpath_patched = False

    try:
        device = next(base.parameters()).device
        img = image_tensor.detach().clone()
        if img.ndim == 3:
            img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        was_training = base.training
        base.eval()
        with torch.no_grad():
            base(img)
        if was_training:
            base.train()

        if not collector:
            raise RuntimeError(
                "No attention maps were captured. "
                "The model may not have accessible MultiheadAttention layers."
            )

        num_tokens = collector[0].shape[-1]
        rollout = torch.eye(num_tokens)

        for attn in collector:
            a = attn[0]  # [num_tokens, num_tokens]
            a = 0.5 * a + 0.5 * torch.eye(num_tokens)
            a = a / a.sum(dim=-1, keepdim=True)
            rollout = a @ rollout

        # Row 0 = CLS token, skip column 0 (CLS attending to itself)
        mask = rollout[0, 1:]  # [num_patches]
        h = w = int(round(mask.shape[0] ** 0.5))
        mask = mask.reshape(h, w)

        mask = mask - mask.min()
        if mask.max() > 1e-8:
            mask = mask / mask.max()

        height, width = image_tensor.shape[-2:]
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        return mask.detach().cpu()

    finally:
        for mha, orig in originals:
            mha.forward = orig
        if _fastpath_patched:
            torch.backends.mha.set_fastpath_enabled(fastpath_was_enabled)
