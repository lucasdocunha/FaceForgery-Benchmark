from __future__ import annotations
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
CNN_FAMILIES={"resnet","xception","mobilenet"}; TRANSFORMER_FAMILIES={"vit","clip","dino"}
def _attention_rollout(model, image):
    with torch.no_grad(): model(image)
    attentions=getattr(model,"last_attentions",None)
    if not attentions: return None
    rollout=None
    for attention in attentions:
        matrix=attention.mean(1); eye=torch.eye(matrix.shape[-1],device=matrix.device); matrix=(matrix+eye); matrix/=matrix.sum(-1,keepdim=True)
        rollout=matrix if rollout is None else matrix@rollout
    tokens=rollout[:,0,1:]; side=int(tokens.shape[-1]**.5)
    if side*side!=tokens.shape[-1]: return None
    return F.interpolate(tokens.reshape(-1,1,side,side),size=image.shape[-2:],mode="bilinear",align_corners=False)
def generate(model,family,image,method="auto"):
    if method=="auto": method="gradcam" if family in CNN_FAMILIES else "attention"
    if method=="attention":
        result=_attention_rollout(model,image)
        if result is not None:
            result-=result.amin((2,3),keepdim=True);result/=result.amax((2,3),keepdim=True).clamp_min(1e-8);return result
    x=image.clone().detach().requires_grad_(True); model.zero_grad(set_to_none=True); logits=model(x); logits[:,logits.argmax(1)].sum().backward()
    heat=x.grad.detach().abs().mean(1,keepdim=True); heat=F.interpolate(heat,size=x.shape[-2:],mode="bilinear",align_corners=False); heat-=heat.amin((2,3),keepdim=True); heat/=heat.amax((2,3),keepdim=True).clamp_min(1e-8); return heat
def overlay(image,heatmap):
    x=image.detach().cpu()[0,:3];x=(x-x.min())/(x.max()-x.min()+1e-8);h=heatmap.detach().cpu()[0,0];rgb=torch.stack((h,torch.zeros_like(h),1-h));return (0.6*x+0.4*rgb).clamp(0,1)
