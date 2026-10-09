"""Counterfactual relief supervision, v1. No file access, geodata, or model code.

All geometry and elevations are in metres. Apply to native 1 m TRAIN DTMs BEFORE
calling the pinned source_view / SLRM / normalize_image pipeline. The caller owns
split checks, CRS checks, source hashing, crop alignment and budget enforcement.
This is a procedural-shape experiment, not a simulation of LiDAR acquisition.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from scipy.ndimage import binary_dilation
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Earthwork:
    # x/y are offsets from the top-left OUTER corner, in image-axis metres.
    x_m: float
    y_m: float
    a_m: float
    b_m: float
    width_m: float
    relief_m: float
    angle_rad: float = 0.0
    kind: Literal['mound', 'bank_ditch', 'ditch_bank'] = 'mound'
    exponent: float = 2.0
    gap_rad: float = 0.0
    gap_angle_rad: float = 0.0
    profile: Literal['bump', 'cosine'] = 'bump'

    def validate(self) -> None:
        numeric = (self.x_m, self.y_m, self.a_m, self.b_m, self.width_m,
                   self.relief_m, self.angle_rad, self.exponent, self.gap_rad,
                   self.gap_angle_rad)
        if not all(math.isfinite(x) for x in numeric):
            raise ValueError('all geometry must be finite')
        if min(self.a_m, self.b_m, self.width_m) <= 0 or self.relief_m < 0:
            raise ValueError('positive lengths and non-negative relief required')
        if self.width_m > min(self.a_m, self.b_m) / 3:
            raise ValueError('width must be <= one third of the minor radius')
        if not 2 <= self.exponent <= 8 or not 0 <= self.gap_rad < math.pi:
            raise ValueError('invalid exponent or gap width')
        if self.kind not in ('mound', 'bank_ditch', 'ditch_bank'):
            raise ValueError('unsupported earthwork kind')
        if self.profile not in ('bump', 'cosine'):
            raise ValueError('unsupported profile')


def _bump(t: np.ndarray, profile: str) -> np.ndarray:
    """Compact C1 radial profiles with support [-1, 1]."""
    inside = np.abs(t) < 1
    out = np.zeros_like(t)
    if profile == 'bump':
        out[inside] = np.exp(-4.5 * t[inside]**2) * (1-t[inside]**2)**2
    else:
        out[inside] = 0.5 * (1 + np.cos(np.pi * t[inside]))
    return out


def render(shape: tuple[int, int], pixel_m: float, spec: Earthwork,
           margin_m: float = 12.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return delta-z, positive footprint, and preprocessing influence mask.

    Positive footprint is a filled planform, NOT a claim that its interior was
    raised. Influence includes the planform and the height field plus margin_m.
    margin_m covers 10 m SLRM support and resampling in the intended 1 m path;
    it is NOT the neural network receptive field. Require a full context margin.
    """
    spec.validate()
    if len(shape) != 2 or min(shape) < 2 or not math.isfinite(pixel_m) or pixel_m <= 0:
        raise ValueError('invalid raster shape or pixel size')
    if not math.isfinite(margin_m) or margin_m < 0:
        raise ValueError('invalid context margin')
    yy, xx = np.indices(shape, dtype=np.float64)
    xx = (xx + .5) * pixel_m - spec.x_m
    yy = (yy + .5) * pixel_m - spec.y_m
    c, s = math.cos(spec.angle_rad), math.sin(spec.angle_rad)
    u, v = c*xx+s*yy, -s*xx+c*yy
    rho = ((np.abs(u)/spec.a_m)**spec.exponent +
           (np.abs(v)/spec.b_m)**spec.exponent)**(1/spec.exponent)
    footprint = rho <= 1.0
    if not footprint.any():
        raise ValueError('empty footprint at this resolution')
    # d is a radial surrogate, not exact signed distance to a superellipse.
    d = (rho-1.0) * min(spec.a_m, spec.b_m)
    if spec.kind == 'mound':
        if spec.profile == 'bump':
            dz = np.maximum(1-rho*rho, 0)**2
        else:
            dz = np.where(rho < 1, .5*(1+np.cos(np.pi*np.minimum(rho, 1))), 0)
    else:
        ring = _bump(d/(3*spec.width_m), spec.profile)
        outer = _bump((d-2*spec.width_m)/(3*spec.width_m), spec.profile)
        dz = ring - .65*outer
        if spec.kind == 'ditch_bank':
            dz = -dz
        if spec.gap_rad:
            theta = np.arctan2(v/spec.b_m, u/spec.a_m)
            angular_distance = np.abs(np.angle(np.exp(1j*(theta-spec.gap_angle_rad))))
            # Smooth causeway transition, not a hard image edge.
            transition = np.clip((angular_distance-spec.gap_rad/2)/.12, 0, 1)
            dz *= transition*transition*(3-2*transition)
    # Get support before multiplying by relief, so zero-height shams use the
    # same masks as their nonzero counterparts.
    support = (np.abs(dz) > 0) | footprint
    support_y, support_x = np.nonzero(support)
    nmargin = int(math.ceil(margin_m/pixel_m))
    if (support_y.min() < nmargin or support_x.min() < nmargin or
        support_y.max() >= shape[0]-nmargin or support_x.max() >= shape[1]-nmargin):
        raise ValueError('earthwork lacks a complete preprocessing context margin')
    influence = (binary_dilation(support, structure=np.ones((3, 3), bool),
                                 iterations=nmargin) if nmargin else support.copy())
    return (dz*spec.relief_m).astype(np.float32), footprint, influence


def implant(dtm: np.ndarray, pixel_m: float, spec: Earthwork,
            forbidden: np.ndarray | None = None,
            margin_m: float = 12.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an independent augmented array; never modify the caller's terrain.

    forbidden must include catalogue/ignore regions plus the caller's protected
    buffers. Unknown background is NOT thereby asserted to be non-archaeological.
    """
    dtm = np.asarray(dtm)
    if dtm.ndim != 2 or not np.issubdtype(dtm.dtype, np.floating):
        raise ValueError('DTM must be a two-dimensional floating array')
    dz, target, influence = render(dtm.shape, pixel_m, spec, margin_m)
    if not np.isfinite(dtm[influence]).all():
        raise ValueError('non-finite terrain within the preprocessing footprint')
    if forbidden is not None:
        forbidden = np.asarray(forbidden)
        if forbidden.shape != dtm.shape or forbidden.dtype != bool:
            raise ValueError('forbidden must be a matching boolean mask')
        if np.any(forbidden & influence):
            raise ValueError('implant intersects a forbidden region')
    result = dtm.copy()
    result += dz.astype(dtm.dtype)
    return result, target, influence


def auxiliary_loss(student: torch.Tensor, teacher_clean: torch.Tensor,
                   target: torch.Tensor, outside: torch.Tensor,
                   enabled: bool = True, margin: float = 2.0,
                   logit_cap: float = math.log(9.0)) -> torch.Tensor:
    """Positive-only synthetic evidence + bounded uplift + teacher consistency.

    Pass ONLY channel-1 logits. Teacher is the frozen CONT e41 on untouched
    terrain, in eval/no_grad mode. target/outside must be disjoint, aligned masks.
    No pixel of the untreated terrain is assigned a synthetic negative label.
    For the matched control use enabled=False but execute the same forwards.
    """
    if student.shape != teacher_clean.shape or student.shape != target.shape or student.shape != outside.shape:
        raise ValueError('all tensors must have identical shapes')
    if target.dtype != torch.bool or outside.dtype != torch.bool:
        raise ValueError('masks must be boolean')
    if target.device != student.device or outside.device != student.device or teacher_clean.device != student.device:
        raise ValueError('all tensors must be on the same device')
    if bool((target & outside).any()) or not bool(target.any()) or not bool(outside.any()):
        raise ValueError('disjoint, nonempty target and outside masks required')
    if not (math.isfinite(margin) and margin > 0 and math.isfinite(logit_cap)):
        raise ValueError('invalid margin or cap')
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher_clean).all()):
        raise ValueError('non-finite logits')
    if not enabled:
        return student.sum() * 0.0
    teacher_clean = teacher_clean.detach()
    positive = F.softplus(-student[target]).mean()
    desired = torch.clamp(teacher_clean[target] + margin, max=logit_cap)
    uplift = F.relu(desired - student[target]).square().mean()
    preserve = F.smooth_l1_loss(student[outside], teacher_clean[outside])
    return .5*positive + .5*uplift + .1*preserve


def matched_update(model: torch.nn.Module, teacher: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   real_x: torch.Tensor, real_label: torch.Tensor,
                   masked_loss_fn, *, clean_x: torch.Tensor | None = None,
                   implanted_x: torch.Tensor | None = None,
                   target: torch.Tensor | None = None,
                   outside: torch.Tensor | None = None,
                   treatment: bool = False, aux_weight: float = .25) -> dict:
    """One optimizer update; sequential backwards avoid retaining two graphs.

    Inputs use the existing batch-one ADAF {'out': [1,2,H,W]} interface.
    masked_loss_fn is r2_train.masked_loss, unchanged. For scheduled auxiliary
    steps BOTH arms receive the same arrays and execute the same forwards;
    the control's auxiliary gradient is exactly zero. No extra optimizer step.
    """
    if teacher is model or aux_weight < 0 or not math.isfinite(aux_weight):
        raise ValueError('need a separate teacher and nonnegative finite aux weight')
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training:
            raise ValueError('student BatchNorm statistics must be frozen')
    if real_x.ndim != 4 or real_x.shape[0] != 1:
        raise ValueError('this integration uses the existing batch-one interface')
    aux_args = (clean_x, implanted_x, target, outside)
    if any(x is not None for x in aux_args) and not all(x is not None for x in aux_args):
        raise ValueError('supply all auxiliary inputs or none')
    optimizer.zero_grad(set_to_none=True)
    real_z = model(real_x)['out'][0, 1]
    real_result = masked_loss_fn(real_z, real_label)
    if real_result is None:
        return {'skipped_all_ignore': True}
    base = real_result[0]
    base_value = float(base.detach())
    base.backward()
    del base, real_z, real_result
    aux_value = 0.0
    if clean_x is not None:
        teacher.eval()
        with torch.no_grad():
            clean_z = teacher(clean_x)['out'][0, 1]
        synthetic_z = model(implanted_x)['out'][0, 1]
        aux = auxiliary_loss(synthetic_z, clean_z, target, outside,
                             enabled=treatment)
        aux_value = float(aux.detach())
        (aux_weight*aux).backward()
    optimizer.step()
    return {'skipped_all_ignore': False, 'real_loss': base_value, 'aux_loss': aux_value}
