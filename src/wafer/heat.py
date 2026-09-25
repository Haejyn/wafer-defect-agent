"""이상 히트맵 두 가지 — 패치 kNN(정상 패치와의 거리) · Grad-CAM(판정한 클래스 기준). 둘 다 64×64, 0~1."""
import numpy as np
import torch
import torch.nn.functional as F

from .model import one_hot
from .ood import feature_maps, patch_scores


def _norm(h):
    return (h - h.min()) / (h.max() - h.min() + 1e-9)


def knn_heat(model, x: np.ndarray, bank, device="cuda") -> np.ndarray:
    maps, _ = feature_maps(model, x[None], device)
    h, _ = patch_scores(maps, bank, device)
    up = F.interpolate(torch.from_numpy(h[None]).float(), size=(64, 64), mode="bilinear", align_corners=False)[0, 0].numpy()
    return _norm(up)


def gradcam_heat(model, x: np.ndarray, device="cuda", layer: str = "b3") -> np.ndarray:
    """판정한 클래스 로짓의 기울기로 가중한 특징맵. layer='b3' 은 16×16, 'b4' 는 8×8."""
    model.eval()
    xb = one_hot(torch.from_numpy(x[None]).to(device))
    feats = {}
    h1 = getattr(model, layer).register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    logits = model(xb)
    h1.remove()
    a = feats["a"]
    cls = int(logits.argmax(1))
    g = torch.autograd.grad(logits[0, cls], a)[0]
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    up = F.interpolate(cam, size=(64, 64), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
    return _norm(up)
