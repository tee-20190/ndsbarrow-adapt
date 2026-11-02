"""DENSE-TERRAIN-1 experiment-owned model adapter (FREEZE.md Sections 2, 3.3, 7). Frozen loaders are imported, never edited.

  A        match_train.fresh_model("BARROW_L"): the official AiTLAS HRNet on the public ADAF barrow tar, 3-channel input
           [l, l, l] (D1's F1 model object).
  B, G, D  the same network with backbone.conv1 (3 -> 64, 3x3, s2, no bias) replaced by a 4 -> 64 convolution:
           weight[:, 0] = sum of the three original input-channel weights, weight[:, 1:] = 0. Input [l, a2, a32, v].
           G and D start F1 from an SSL teacher encoder (load_encoder); the head is the ADAF head in every arm.
Inference: load_checkpoint(path, channels) rebuilds the official module (adaf_configs.load_models' construction) with the
arm's stem and loads the state dict strictly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in ("work/adapt_match_2026-10-05", "work/campaign_2026-09-22", "work/adaf_head_to_head_2026-09-19"):
    if str(ROOT / p) not in sys.path:
        sys.path.insert(0, str(ROOT / p))

ARMS = ("A", "B", "G", "D")
CHANNELS = {"A": 3, "B": 4, "G": 4, "D": 4}
HRNET_SOURCE_SHA256 = "4fd8706232e8e73ea8f193f057e1401eddc6e0bcf2c810198dcc8a5c8fc175ce"


def four_channel_stem(conv: nn.Conv2d) -> nn.Conv2d:
    """The summed-weight 4-channel replacement of a 3-channel stem convolution (FREEZE Section 3.3)."""
    if conv.in_channels != 3 or conv.bias is not None:
        raise ValueError("expected the 3-channel, bias-free HRNet stem")
    new = nn.Conv2d(4, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups,
                    bias=False, padding_mode=conv.padding_mode).to(conv.weight.device, conv.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, 0] = conv.weight.sum(dim=1)
    return new


def official_module() -> nn.Module:
    """adaf_configs.load_models' construction without weights (official AiTLAS source, hash-checked)."""
    import adapter
    ns, _, source_sha = adapter.official_nodes()
    if source_sha != HRNET_SOURCE_SHA256:
        raise RuntimeError("official HRNet source changed")
    return ns["HRNetModule"](ns["HRNetSegHead"](2), pretrained=False)


def to_four(model: nn.Module) -> nn.Module:
    model.backbone.conv1 = four_channel_stem(model.backbone.conv1)
    return model


def build(arm: str, device: str = "cuda", encoder: Path | None = None):
    """(model, optimizer) for F1, D1's construction: rt.load_model on the pinned ADAF barrow tar, frozen-BN train mode,
    a fresh AdamW (plans.LR_MAX, r2_train.WD) over every parameter. G/D: `encoder` = the SSL teacher's stem + backbone."""
    import match_train as MT
    if arm not in ARMS:
        raise ValueError(arm)
    if arm == "A":
        if encoder is not None:
            raise ValueError("arm A has no SSL encoder")
        return MT.fresh_model("BARROW_L", device)
    if (encoder is None) != (arm == "B"):
        raise ValueError(f"arm {arm}: encoder {'required' if arm != 'B' else 'not allowed'}")
    model = MT.rt.load_model(MT.start_weights("BARROW_L"), device=device)
    to_four(model)
    if encoder is not None:
        load_encoder(model, encoder)
    model = model.to(device)
    MT.rt.set_train_mode(model)
    return model, torch.optim.AdamW(model.parameters(), lr=MT.plans.LR_MAX, weight_decay=MT.rt.WD)


def load_encoder(model: nn.Module, path: Path) -> dict:
    """Load an SSL teacher encoder (keys 'backbone.*', 4-channel conv1) strictly into the backbone; the head is kept."""
    state = torch.load(path, map_location="cpu", weights_only=True)["encoder"]
    want = {k for k in model.state_dict() if k.startswith("backbone.")}
    if set(state) != want:
        raise SystemExit(f"encoder keys differ: {sorted(set(state) ^ want)[:5]}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(k.startswith("backbone.") for k in missing):
        raise SystemExit("encoder load incomplete")
    return {"tensors": len(state)}


def load_checkpoint(path: Path, channels: int, device: str = "cuda") -> nn.Module:
    """Inference loader for a DT1 final.pt (state_dict) or an ADAF tar; eval mode."""
    model = official_module()
    if channels == 4:
        to_four(model)
    state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    del state
    return model.eval().to(device)
