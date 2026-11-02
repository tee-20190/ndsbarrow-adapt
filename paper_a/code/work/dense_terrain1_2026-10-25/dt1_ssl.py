"""DENSE-TERRAIN-1 SSL for arms G and D (FREEZE.md Section 5), VM side.

Student and teacher: the official HRNet with the four-channel summed-weight stem on the ADAF barrow start; only the
backbone is used (layer1 + stage-4 branches sampled at points, 976-d, dt1_coords). BatchNorm running statistics stay at
ADAF values (BN modules in eval mode); student BN affine parameters learn, the teacher follows by EMA. The segmentation
head is frozen and unused.

Data per optimiser update u (seed s): 8 distinct FIT windows (a seeded permutation of the 1,704 cycling without
replacement), each with two 256 m views: centre = window centre + independent integer-metre offsets in [-32, 32] m per
axis, independent dihedral k; legacy channel = adapt_train.relief_v + apply_v (training chain), added channels from the
window's native products. Points: the same 256-point multiset serves G and D: 128 uniform + 128 balanced over five
FIT-fixed quintiles of e = sqrt((r2/.1)^2 + (r32/1)^2) (26, 26, 26, 25, 25), drawn from window pixel centres with v = 1 in
both views and >= 64 px (32 m) from both crop boundaries; every draw comes from numpy default_rng keyed by
(seed, purpose, update, slot), so it does not depend on the arm, the worker or the device.

Objective: G[i, v] = proj(mean_p f[i, v, p]); D[i, v, p] = proj(f[i, v, p]). Projector 976 -> 2048 -> 2048 -> 256 (GELU,
no BN), L2-normalised, prototype layer 256 -> 4,096 (unit-norm directions, gain 1), directions frozen for 128 updates.
Symmetric cross-view cross-entropy H(softmax((t - c) / tau_t), log_softmax(s / tau_s)), D averaged over points within an
image, then over the 8 images; float32. Centre EMA 0.9 over the per-image mean teacher logits (G: two pooled views; D:
points and views). tau_s 0.10; tau_t 0.04 -> 0.07 over 512 updates. Teacher = exact student copy at start; EMA momentum
cosine 0.996 -> 1.0 over U. AdamW wd 0.04 (no decay on biases / norm parameters), lr warm-up 128 updates to 1e-4 then
cosine to 1e-6 at U, gradient-norm clip 3; four microbatches of two windows accumulated before each step.

  python -B dt1_ssl.py edges PRIV_EDGES.json --ls-table T --ls-win W --ls-mask M --prod P        (CPU, once)
  python -B dt1_ssl.py pool OUT.json --ls-table T --ls-win W --ls-mask M --prod P                 (Erratum E1, once)
  python -B dt1_ssl.py train ARM SEED U OUT_DIR --edges E --pool POOL --ls-table T --ls-win W --ls-mask M --prod P [--stop-at N]
        ARM in {G, D}; --stop-at N stops after N updates (the disposable 200-update benchmark) with U's schedule
Outputs: OUT_DIR/{curves.jsonl, ckpt.pt (every 256 updates, atomic), ENCODER_FINAL.pt, SSL_RESULT.json}
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "work/d1_collapse_2026-10-06"))
import d1_train as T1  # noqa: E402
from d1_train import MT, N, at, rt  # noqa: E402
import dt1_channels as CH  # noqa: E402
import dt1_coords as CO  # noqa: E402
import dt1_model as DM  # noqa: E402

EXPERIMENT = "DENSE-TERRAIN-1"
SEEDS = (101, 202, 303)
WIN_PX, VIEW_PX, MARGIN_PX, OFF_M = 846, 512, 64, 32
N_UNIF, STRATA_ALLOC = 128, (26, 26, 26, 25, 25)
N_WIN, MICRO = 8, 2
PROJ_HID, PROJ_OUT, N_PROTO = 2048, 256, 4096
TAU_S, TAU_T0, TAU_T1, TAU_T_WARM = 0.10, 0.04, 0.07, 512
M0, LR_MAX, LR_MIN, WARM, WD, CLIP, CENTRE_M, PROTO_FREEZE = 0.996, 1e-4, 1e-6, 128, 0.04, 3.0, 0.9, 128
CKPT_EVERY = 256
MAX_REDRAW, POOL_MIN = 32, 256          # Erratum E1
P_UPDATE, P_VIEW, P_PTS = 7, 11, 13


# ---------------------------------------------------------------- data
def window_inputs(row: dict, prod_dir: Path) -> np.ndarray:
    """(4, 846, 846): the training-chain legacy channel (policy V) and the three added channels of a whole window."""
    img, _ = at.apply_v(at.relief_v(Path(row["tif"])), np.zeros((WIN_PX, WIN_PX), dtype=np.uint8))
    with np.load(prod_dir / f"{row['id']}.npz") as z:
        ex = CH.view_channels({k: z[k] for k in ("r2", "r32", "support")})[:, 2:-2, 2:-2]
    if img.shape != (WIN_PX, WIN_PX) or ex.shape != (3, WIN_PX, WIN_PX):
        raise SystemExit(f"{row['id']}: unexpected window size {img.shape} {ex.shape}")
    return np.concatenate([img[None], ex]).astype(np.float32)


def e_from_channels(x: np.ndarray) -> np.ndarray:
    return CH.e_value(0.1 * np.sinh(x[1].astype(np.float64)), np.sinh(x[2].astype(np.float64)))


def window_order(seed: int, u: int, n: int) -> list[int]:
    """Windows 8u .. 8u + 7 of the stream of seeded permutations of range(n), one permutation per cycle."""
    out = []
    for g in range(N_WIN * u, N_WIN * (u + 1)):
        cycle, pos = divmod(g, n)
        out.append(int(np.random.default_rng([seed, P_UPDATE, cycle]).permutation(n)[pos]))
    return out


def view_params(seed: int, u: int, slot: int, attempt: int = 0) -> list[tuple[int, int, int]]:
    """Two views: (dy, dx, k) with dy, dx = 2 x integer metres in [-32, 32] (rows south-positive, as r2_train.crop).
    Erratum E1: attempt > 0 redraws a view pair that has no common valid point (key extended by the attempt)."""
    rng = np.random.default_rng([seed, P_VIEW, u, slot] + ([attempt] if attempt else []))
    out = []
    for _ in range(2):
        a, b = (int(v) for v in rng.integers(-OFF_M, OFF_M + 1, 2))
        out.append((2 * a, 2 * b, int(rng.integers(0, 8))))
    return out


def crop_box(dy: int, dx: int) -> tuple[int, int]:
    return CO.crop_offset(WIN_PX, WIN_PX, dy, dx)


def candidates(x: np.ndarray, views: list[tuple[int, int, int]]) -> np.ndarray:
    """Window pixel indices (row, col) with v = 1 and >= 64 px from both crop boundaries."""
    lo_r = max(crop_box(dy, dx)[0] for dy, dx, _ in views) + MARGIN_PX
    hi_r = min(crop_box(dy, dx)[0] for dy, dx, _ in views) + VIEW_PX - MARGIN_PX
    lo_c = max(crop_box(dy, dx)[1] for dy, dx, _ in views) + MARGIN_PX
    hi_c = min(crop_box(dy, dx)[1] for dy, dx, _ in views) + VIEW_PX - MARGIN_PX
    sub = x[3, lo_r:hi_r, lo_c:hi_c] == 1
    rr, cc = np.nonzero(sub)
    return np.stack([rr + lo_r, cc + lo_c], 1)


def choose_points(x: np.ndarray, cand: np.ndarray, edges: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(256, 2) window pixel indices: 128 uniform + 128 quintile-balanced, with replacement only when needed."""
    if len(cand) == 0:
        raise SystemExit("no valid common point in a view pair")
    uni = cand[rng.choice(len(cand), N_UNIF, replace=len(cand) < N_UNIF)]
    e = e_from_channels(x)[cand[:, 0], cand[:, 1]]
    q = np.searchsorted(edges, e, side="right")
    alloc = list(STRATA_ALLOC)
    empty = [s for s in range(5) if not (q == s).any()]
    if len(empty) == 5:
        raise SystemExit("no stratum populated")
    full = [s for s in range(5) if s not in empty]
    for k, s in enumerate(empty):
        alloc[full[k % len(full)]] += alloc[s]
        alloc[s] = 0
    parts = [uni]
    for s in range(5):
        if alloc[s]:
            idx = np.flatnonzero(q == s)
            parts.append(cand[idx[rng.choice(len(idx), alloc[s], replace=len(idx) < alloc[s])]])
    return np.concatenate(parts)


def view_coords(pts: np.ndarray, dy: int, dx: int, k: int) -> np.ndarray:
    r0, c0 = crop_box(dy, dx)
    r, c = CO.dihedral_coords(pts[:, 0].astype(np.float64) - r0, pts[:, 1].astype(np.float64) - c0, k)
    return np.stack([r, c], 1)


class UpdateData:
    """Item u -> 8 windows: views (8, 2, 4, 512, 512), view coordinates (8, 2, 256, 2), point digest."""

    def __init__(self, rows: list[dict], prod_dir: Path, edges: np.ndarray, seed: int, start: int, stop: int,
                 pool: list[int] | None = None) -> None:
        self.rows, self.prod, self.edges, self.seed, self.start, self.stop = rows, prod_dir, edges, seed, start, stop
        self.pool = list(range(len(rows))) if pool is None else pool

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, i: int):
        import torch
        u = self.start + i
        views, coords, h = [], [], hashlib.sha256()
        for slot, pi in enumerate(window_order(self.seed, u, len(self.pool))):
            wi = self.pool[pi]
            x = window_inputs(self.rows[wi], self.prod)
            for attempt in range(MAX_REDRAW):
                vp = view_params(self.seed, u, slot, attempt)
                cand = candidates(x, vp)
                if len(cand):
                    break
            else:
                raise SystemExit(f"window {self.rows[wi]['id']}: no common valid point in {MAX_REDRAW} view pairs")
            pts = choose_points(x, cand, self.edges, np.random.default_rng([self.seed, P_PTS, u, slot]))
            h.update(np.int64([wi, *[v for t in vp for v in t]]).tobytes())
            h.update(pts.astype(np.int64).tobytes())
            vv, cc = [], []
            for dy, dx, k in vp:
                vv.append(np.stack([rt.dihedral(rt.crop(ch, dy, dx), k) for ch in x]))
                cc.append(view_coords(pts, dy, dx, k))
            views.append(np.stack(vv))
            coords.append(np.stack(cc))
        return u, torch.from_numpy(np.stack(views)), torch.from_numpy(np.stack(coords)), h.hexdigest()


# ---------------------------------------------------------------- model
def projector():
    import torch.nn as nn
    return nn.Sequential(nn.Linear(CO.FEATURE_DIM, PROJ_HID), nn.GELU(), nn.Linear(PROJ_HID, PROJ_HID), nn.GELU(),
                         nn.Linear(PROJ_HID, PROJ_OUT))


class Net:
    """Encoder (official module, backbone used) + projector + prototype directions."""

    def __init__(self, device: str, seed: int) -> None:
        import torch
        self.enc = DM.to_four(MT.rt.load_model(MT.start_weights("BARROW_L"), device="cpu")).to(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.proj = projector().to(device)
            self.proto = torch.nn.Parameter(torch.randn(N_PROTO, PROJ_OUT, device=device))
        for p in self.enc.head.parameters():
            p.requires_grad_(False)

    def modules(self):
        return [self.enc, self.proj]

    def params(self):
        return [p for m in self.modules() for p in m.parameters() if p.requires_grad] + [self.proto]

    def train_mode(self) -> None:
        import torch
        self.enc.train()
        for m in self.enc.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
        self.proj.train()

    def logits(self, x, coords, arm: str):
        """x (B, 4, 512, 512), coords (B, P, 2) -> G: (B, K); D: (B, P, K)."""
        import torch
        import torch.nn.functional as F
        maps = CO.branch_maps(self.enc, x)
        f = CO.features_at(maps, coords[..., 0], coords[..., 1])               # (B, P, 976)
        if arm == "G":
            f = f.mean(1)
        z = F.normalize(self.proj(f), dim=-1)
        w = F.normalize(self.proto, dim=-1)
        return torch.matmul(z, w.t())


def schedules(u: int, U: int) -> dict:
    lr = LR_MAX * (u + 1) / WARM if u < WARM else (
        LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * (u - WARM) / max(1, U - WARM - 1))))
    tau_t = TAU_T0 + (TAU_T1 - TAU_T0) * min(1.0, u / TAU_T_WARM)
    m = 1 - (1 - M0) * (math.cos(math.pi * u / U) + 1) / 2
    return {"lr": lr, "tau_t": tau_t, "momentum": m}


def param_groups(net: Net) -> list[dict]:
    decay, no = [], []
    for m in net.modules():
        for n, p in m.named_parameters():
            if p.requires_grad:
                (no if p.ndim <= 1 else decay).append(p)
    decay.append(net.proto)
    return [{"params": decay, "weight_decay": WD}, {"params": no, "weight_decay": 0.0}]


def loss_and_centre(s_log, t_log, centre, tau_t: float, arm: str):
    """s_log, t_log: (2W, ...) logits ordered [w0v0, w0v1, w1v0, ...]; returns per-window losses and teacher means."""
    import torch
    W = s_log.shape[0] // 2
    sv = s_log.float().view(W, 2, *s_log.shape[1:])
    tv = t_log.float().view(W, 2, *t_log.shape[1:])
    tp = torch.softmax((tv - centre) / tau_t, dim=-1)
    ls = torch.log_softmax(sv / TAU_S, dim=-1)
    ce = lambda t, s: -(t * s).sum(-1)  # noqa: E731
    l12, l21 = ce(tp[:, 0], ls[:, 1]), ce(tp[:, 1], ls[:, 0])
    if arm == "D":
        l12, l21 = l12.mean(-1), l21.mean(-1)
        tmean = tv.mean(dim=(1, 2))
    else:
        tmean = tv.mean(dim=1)
    return 0.5 * (l12 + l21), tmean, tp


def ema(teacher: Net, student: Net, m: float) -> None:
    import torch
    with torch.no_grad():
        for ms, mt in zip(student.modules(), teacher.modules()):
            for ps, pt_ in zip(ms.parameters(), mt.parameters()):
                pt_.mul_(m).add_(ps.detach(), alpha=1 - m)
        teacher.proto.mul_(m).add_(student.proto.detach(), alpha=1 - m)


def edges_cmd(out: Path, paths: dict) -> dict:
    ctx = MT.ls_context(paths)
    vals = []
    for r in ctx["rows"]:
        x = window_inputs(r, paths["prod"])
        r0, c0 = crop_box(0, 0)
        sub = x[:, r0:r0 + VIEW_PX:8, c0:c0 + VIEW_PX:8]
        v = sub[3] == 1
        vals.append(e_from_channels(sub)[v])
    e = np.concatenate(vals)
    edges = np.quantile(e, [0.2, 0.4, 0.6, 0.8])
    res = {"experiment": EXPERIMENT, "windows": len(ctx["rows"]), "pixels": int(e.size), "edges": edges.tolist(),
           "rule": "quintile edges of e over v = 1 pixels at stride 8 px of the central unshifted 256 m view (k = 0)",
           "channels_sha256": N.sha256_file(Path(CH.__file__)), "utc": N.utc()}
    N.write_json(out, res)
    return res


def _pool_one(args: tuple[dict, str]) -> int:
    row, prod = args
    x = window_inputs(row, Path(prod))
    return int(len(candidates(x, [(0, 0, 0), (0, 0, 0)])))


def pool_cmd(out: Path, paths: dict) -> dict:
    """Erratum E1: the SSL pool = FIT windows whose central unshifted view has >= POOL_MIN valid candidate points
    (v = 1, >= 64 px from the view boundary)."""
    from concurrent.futures import ProcessPoolExecutor
    ctx = MT.ls_context(paths)
    with ProcessPoolExecutor(max_workers=32) as ex:
        counts = list(ex.map(_pool_one, [(r, str(paths["prod"])) for r in ctx["rows"]], chunksize=8))
    pool = [i for i, n in enumerate(counts) if n >= POOL_MIN]
    res = {"experiment": EXPERIMENT, "rule": f"central view candidates >= {POOL_MIN}", "windows": len(counts),
           "pool": pool, "excluded_ids": [ctx["rows"][i]["id"] for i, n in enumerate(counts) if n < POOL_MIN],
           "excluded_roles": {r: sum(1 for i, n in enumerate(counts) if n < POOL_MIN and ctx["rows"][i]["role"] == r)
                              for r in ("positive", "background")},
           "candidate_count_min": min(counts), "utc": N.utc()}
    N.write_json(out, res)
    return {k: v for k, v in res.items() if k not in ("pool", "excluded_ids")} | {"pool_size": len(pool)}


def train(arm: str, seed: int, U: int, out: Path, paths: dict, stop_at: int | None) -> dict:
    import torch
    from torch.utils.data import DataLoader
    if arm not in ("G", "D") or seed not in SEEDS:
        raise SystemExit(f"SSL run outside DT1: {arm} {seed}")
    T1.JT1.J.torch_flags()
    dev = "cuda"
    edges = np.asarray(N.read_json(paths["edges"])["edges"])
    ctx = MT.ls_context(paths)
    out.mkdir(parents=True, exist_ok=True)
    student = Net(dev, seed)
    teacher = Net(dev, seed)
    teacher.enc.load_state_dict(student.enc.state_dict())
    teacher.proj.load_state_dict(student.proj.state_dict())
    with torch.no_grad():
        teacher.proto.copy_(student.proto)
    for p in teacher.params():
        p.requires_grad_(False)
    student.train_mode()
    teacher.enc.eval()
    teacher.proj.eval()
    opt = torch.optim.AdamW(param_groups(student), lr=LR_MAX)
    centre = torch.zeros(N_PROTO, device=dev)
    start, ck = 0, out / "ckpt.pt"
    curves = out / "curves.jsonl"
    if ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=False)
        for k, net in (("student", student), ("teacher", teacher)):
            net.enc.load_state_dict(s[k]["enc"])
            net.proj.load_state_dict(s[k]["proj"])
            with torch.no_grad():
                net.proto.copy_(s[k]["proto"])
        opt.load_state_dict(s["opt"])
        centre = s["centre"]
        start = s["u"]
        keep = [ln for ln in curves.read_text(encoding="utf-8").splitlines() if json.loads(ln)["u"] < start]
        curves.write_text("".join(x + "\n" for x in keep), encoding="utf-8")
    elif curves.exists():
        curves.unlink()
    stop = U if stop_at is None else min(U, stop_at)
    pool = N.read_json(paths["pool"])["pool"] if paths.get("pool") else None
    data = UpdateData(ctx["rows"], paths["prod"], edges, seed, start, stop, pool)
    dl = DataLoader(data, batch_size=None, shuffle=False, num_workers=12, prefetch_factor=2, persistent_workers=False)
    t0, low_ent, hi_ent, warn = time.monotonic(), 0, 0, None
    torch.cuda.reset_peak_memory_stats()
    with curves.open("a", encoding="utf-8") as cf:
        for u, views, coords, digest in dl:
            tu = time.monotonic()
            sch = schedules(u, U)
            for g in opt.param_groups:
                g["lr"] = sch["lr"]
            opt.zero_grad(set_to_none=True)
            losses, tmeans, ents = [], [], []
            for mb in range(0, N_WIN, MICRO):
                x = views[mb:mb + MICRO].reshape(-1, 4, VIEW_PX, VIEW_PX).to(dev, non_blocking=True)
                c = coords[mb:mb + MICRO].reshape(-1, coords.shape[2], 2).to(dev)
                s_log = student.logits(x, c, arm)
                with torch.no_grad():
                    t_log = teacher.logits(x, c, arm)
                lw, tm, tp = loss_and_centre(s_log, t_log, centre, sch["tau_t"], arm)
                (lw.sum() / N_WIN).backward()
                losses += lw.detach().tolist()
                tmeans.append(tm.detach())
                pm = tp.reshape(-1, N_PROTO).mean(0)
                ents.append(float(-(pm * torch.log(pm + 1e-12)).sum()))
            if u < PROTO_FREEZE and student.proto.grad is not None:
                student.proto.grad.zero_()
            gn = float(torch.nn.utils.clip_grad_norm_(student.params(), CLIP))
            opt.step()
            ema(teacher, student, sch["momentum"])
            with torch.no_grad():
                centre.mul_(CENTRE_M).add_(torch.cat(tmeans).mean(0), alpha=1 - CENTRE_M)
            loss = float(np.mean(losses))
            ent = float(np.mean(ents))
            if not math.isfinite(loss) or not math.isfinite(gn):
                raise SystemExit(f"non-finite SSL update {u}: loss {loss} grad {gn}")
            low_ent = low_ent + 1 if ent < 0.05 * math.log(N_PROTO) else 0
            hi_ent = hi_ent + 1 if ent > 0.995 * math.log(N_PROTO) else 0
            if (low_ent >= 256 or hi_ent >= 256) and warn is None:
                warn = {"u": int(u), "kind": "entropy_min" if low_ent >= 256 else "entropy_max"}
            rec = {"u": int(u), "loss": loss, "grad_norm": gn, "teacher_mean_entropy": ent, "centre_norm":
                   float(centre.norm()), **sch, "points_digest": digest, "s_update": round(time.monotonic() - tu, 3),
                   "elapsed_s": round(time.monotonic() - t0, 1)}
            cf.write(json.dumps(rec) + "\n")
            cf.flush()
            if int(u) % 25 == 0:
                print(json.dumps(rec), flush=True)
            if (int(u) + 1) % CKPT_EVERY == 0 and int(u) + 1 < stop:
                st = {k: {"enc": n.enc.state_dict(), "proj": n.proj.state_dict(), "proto": n.proto.detach()}
                      for k, n in (("student", student), ("teacher", teacher))}
                tmp = out / "ckpt.pt.tmp"
                torch.save({**st, "opt": opt.state_dict(), "centre": centre, "u": int(u) + 1}, tmp)
                tmp.replace(ck)
    wall = time.monotonic() - t0
    rows = [json.loads(x) for x in curves.read_text(encoding="utf-8").splitlines()]
    enc = {k: v for k, v in teacher.enc.state_dict().items() if k.startswith("backbone.")}
    final = out / ("ENCODER_FINAL.pt" if stop == U else f"ENCODER_BENCH_{stop}.pt")
    if stop == U:
        tmp = final.with_name(final.name + ".tmp")
        torch.save({"encoder": enc, "arm": arm, "seed": seed, "updates": U, "experiment": EXPERIMENT}, tmp)
        tmp.replace(final)
    timed = [r["s_update"] for r in rows if r["u"] >= start + 10]
    res = {"experiment": EXPERIMENT, "arm": arm, "seed": seed, "U": U, "updates_done": len(rows), "stop": stop,
           "status": "COMPLETE" if stop == U else "BENCHMARK", "wall_s": wall,
           "s_per_update_median": float(np.median(timed)) if timed else None,
           "s_per_update_mean": float(np.mean(timed)) if timed else None,
           "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
           "loss_first": rows[0]["loss"] if rows else None, "loss_last": rows[-1]["loss"] if rows else None,
           "entropy_last": rows[-1]["teacher_mean_entropy"] if rows else None, "collapse_warning": warn,
           "points_digest_of_digests": hashlib.sha256("".join(r["points_digest"] for r in rows).encode()).hexdigest(),
           "encoder_final_sha256": N.sha256_file(final) if stop == U else None,
           "edges_sha256": N.sha256_file(paths["edges"]),
           "pool_sha256": N.sha256_file(paths["pool"]) if paths.get("pool") else None, "code_sha256": {
               "dt1_ssl.py": N.sha256_file(Path(__file__)), "dt1_coords.py": N.sha256_file(Path(CO.__file__)),
               "dt1_channels.py": N.sha256_file(Path(CH.__file__)), "dt1_model.py": N.sha256_file(Path(DM.__file__))},
           "utc": N.utc()}
    if stop == U:
        ck.unlink(missing_ok=True)
    N.write_json(out / ("SSL_RESULT.json" if stop == U else f"SSL_BENCH_{stop}.json"), res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("edges", "pool", "train"))
    ap.add_argument("args", nargs="*")
    for k in ("ls-table", "ls-win", "ls-mask", "prod", "edges", "pool"):
        ap.add_argument(f"--{k}", type=Path)
    ap.add_argument("--stop-at", type=int)
    a = ap.parse_args()
    paths = {k: getattr(a, k) for k in ("ls_table", "ls_win", "ls_mask", "prod", "edges", "pool")
             if getattr(a, k) is not None}
    if a.cmd == "pool":
        print(json.dumps(pool_cmd(Path(a.args[0]), paths)))
        return
    if a.cmd == "edges":
        r = edges_cmd(Path(a.args[0]), paths)
        print(json.dumps({k: r[k] for k in ("windows", "pixels", "edges")}))
    else:
        arm, seed, U, out = a.args[0], int(a.args[1]), int(a.args[2]), Path(a.args[3])
        r = train(arm, seed, U, out, paths, a.stop_at)
        print(json.dumps({k: r[k] for k in ("arm", "seed", "status", "updates_done", "wall_s", "s_per_update_median",
                                            "collapse_warning")}))


if __name__ == "__main__":
    main()
