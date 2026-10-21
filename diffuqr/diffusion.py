"""Module-space conditional diffusion for QR restoration (Phase 2).

What is diffused: the QR MODULE GRID as analog bits (+1 dark, -1 light;
Chen, Zhang & Hinton, "Analog Bits", ICLR 2023), on a fixed CANVAS x CANVAS
cell canvas so one model serves versions 10-18.

Conditioning (all concatenated as input channels, SR3/Palette style):
  - blurry crop, rectified at PPM px/module (diffuqr.modspace format),
    encoded by a stride-8 conv encoder -> exactly one feature vector per
    module cell (the rectification makes pixels and modules line up);
  - structure: values + mask of modules fixed by the QR VERSION (finders,
    timing, alignment, version info) and the quiet-zone margin;
  - validity mask of the symbol;
  - self-conditioning: the model's previous x0 estimate (Analog Bits).

Backbone: diffusers `UNet2DModel` (standard DDPM U-Net), unmodified.
Parameterisation: the network outputs one logit per module; x0_hat =
tanh(logit / 2) is fed to the scheduler with prediction_type="sample".
Loss: binary cross-entropy on module bits over symbol modules that are not
fixed by the version. Sampling: DDIM; version-fixed modules are replaced by
their known values at every step (RePaint-style replacement).

`mode="direct"` trains the same network as a one-shot predictor (no noise,
t = 0): the "why diffusion?" ablation.
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, UNet2DModel

from diffuqr.qrcodec import decode_grid, decode_soft
from diffuqr.qrgrid import modules_for_version, read_format, version_fixed

CANVAS = 96          # grid cells; symbol + 2*MARGIN must fit (v18: 89 + 4 = 93)
PPM = 8              # input px per module (diffuqr.modspace default)
MARGIN = 2           # modules of context on each side (diffuqr.modspace default)
T_TRAIN = 1000


# -- Examples ----------------------------------------------------------------

def make_example(inp_u8: np.ndarray, grid: Optional[np.ndarray], version: int) -> Dict[str, np.ndarray]:
    """Place one modspace sample on the canvas.

    inp_u8: (S, S) uint8 with S = (N + 2*MARGIN) * PPM.  grid: (N, N) 0/1 or
    None (inference). Returns float32 arrays shaped (1, H, W)."""
    n = modules_for_version(version)
    cells = n + 2 * MARGIN
    assert inp_u8.shape == (cells * PPM, cells * PPM), (inp_u8.shape, n)
    assert cells <= CANVAS, f"version {version} does not fit the {CANVAS}-cell canvas"
    img = np.zeros((CANVAS * PPM, CANVAS * PPM), np.float32)
    img[: cells * PPM, : cells * PPM] = inp_u8.astype(np.float32) / 127.5 - 1.0

    vmask, vvals = version_fixed(version)
    known_mask = np.ones((CANVAS, CANVAS), np.float32)        # margin + padding are known light
    known_val = -np.ones((CANVAS, CANVAS), np.float32)
    sym = (slice(MARGIN, MARGIN + n), slice(MARGIN, MARGIN + n))
    known_mask[sym] = vmask.astype(np.float32)
    known_val[sym] = np.where(vmask, vvals.astype(np.float32) * 2 - 1, 0.0)
    valid = np.zeros((CANVAS, CANVAS), np.float32)
    valid[sym] = 1.0

    x0 = -np.ones((CANVAS, CANVAS), np.float32)
    if grid is not None:
        x0[sym] = grid.astype(np.float32) * 2 - 1
    return {"img": img[None], "x0": x0[None], "known_mask": known_mask[None],
            "known_val": known_val[None], "valid": valid[None]}


def collate(examples: Sequence[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.stack([e[k] for e in examples])) for k in examples[0]}


def symbol_from_canvas(canvas: np.ndarray, version: int) -> np.ndarray:
    n = modules_for_version(version)
    return canvas[..., MARGIN:MARGIN + n, MARGIN:MARGIN + n]


# -- Model -------------------------------------------------------------------

def _block(cin: int, cout: int, stride: int) -> nn.Sequential:
    k, p = (4, 1) if stride == 2 else (3, 1)
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, p), nn.GroupNorm(8, cout), nn.SiLU())


class CondEncoder(nn.Module):
    """Rectified crop (PPM px per module) -> one feature vector per module
    cell: three stride-2 stages (PPM = 8) plus refinement convs."""

    def __init__(self, ch: int = 32, out: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            _block(1, ch, 2), _block(ch, ch, 1),
            _block(ch, 2 * ch, 2), _block(2 * ch, 2 * ch, 1),
            _block(2 * ch, 2 * ch, 2), _block(2 * ch, 2 * ch, 1),
            nn.Conv2d(2 * ch, out, 1),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(img)


@dataclass
class ModelConfig:
    cond_ch: int = 32
    enc_ch: int = 32
    block_out: Tuple[int, ...] = (64, 128, 256, 256)
    layers_per_block: int = 2
    attn_level: int = 2  # index of the down block with attention (2 -> 24x24 on a 96 canvas)


class _UNet(UNet2DModel):
    """UNet2DModel whose `dtype` comes from a concrete weight. diffusers'
    default walks `self.parameters()`, which is empty on nn.DataParallel
    replicas and then fails (UnboundLocalError on Python 3.13)."""

    @property
    def dtype(self) -> torch.dtype:
        return self.conv_in.weight.dtype


class ModuleDiffusion(nn.Module):
    N_STRUCT = 5  # x_t, self-cond, known_val, known_mask, valid

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.enc = CondEncoder(cfg.enc_ch, cfg.cond_ch)
        n = len(cfg.block_out)
        down = tuple("AttnDownBlock2D" if i == cfg.attn_level else "DownBlock2D" for i in range(n))
        up = tuple("AttnUpBlock2D" if (n - 1 - i) == cfg.attn_level else "UpBlock2D" for i in range(n))
        self.unet = _UNet(
            sample_size=CANVAS, in_channels=self.N_STRUCT + cfg.cond_ch, out_channels=1,
            layers_per_block=cfg.layers_per_block, block_out_channels=tuple(cfg.block_out),
            down_block_types=down, up_block_types=up, norm_num_groups=min(32, cfg.block_out[0]),
        )

    def forward(self, x_t, t, known_val, known_mask, valid, sc, img=None, cond=None):
        """Returns (logits, cond). Pass `cond` to reuse encoder features."""
        if cond is None:
            cond = self.enc(img)
        h = torch.cat([x_t, sc, known_val, known_mask, valid, cond.to(x_t.dtype)], dim=1)
        return self.unet(h, t).sample, cond


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in unwrap(model).parameters())


def make_train_scheduler() -> DDIMScheduler:
    return DDIMScheduler(num_train_timesteps=T_TRAIN, beta_schedule="squaredcos_cap_v2",
                         prediction_type="sample", clip_sample=True, clip_sample_range=1.0)


# -- Training ----------------------------------------------------------------

def _clamp_known(x0: torch.Tensor, batch) -> torch.Tensor:
    return torch.where(batch["known_mask"] > 0, batch["known_val"], x0)


def training_loss(model, batch: Dict[str, torch.Tensor], sched: DDIMScheduler, mode: str = "diffusion",
                  self_cond_p: float = 0.5) -> torch.Tensor:
    x0 = batch["x0"]
    B = x0.shape[0]
    if mode == "direct":
        t = torch.zeros(B, dtype=torch.long, device=x0.device)
        x_t = torch.zeros_like(x0)
    else:
        t = torch.randint(0, T_TRAIN, (B,), device=x0.device)
        x_t = sched.add_noise(x0, torch.randn_like(x0), t)
    sc = torch.zeros_like(x0)
    common = dict(known_val=batch["known_val"], known_mask=batch["known_mask"], valid=batch["valid"])
    if mode == "diffusion" and random.random() < self_cond_p:
        with torch.no_grad():
            logit0, _ = model(x_t, t, sc=sc, img=batch["img"], **common)
            sc = _clamp_known(torch.tanh(logit0.float() / 2), batch).to(x0.dtype)
    logit, _ = model(x_t, t, sc=sc, img=batch["img"], **common)
    w = batch["valid"] * (1 - batch["known_mask"])
    bce = F.binary_cross_entropy_with_logits(logit.float(), (x0 > 0).float(), reduction="none")
    return (bce * w).sum() / w.sum().clamp_min(1.0)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(unwrap(model)).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        src = unwrap(model)
        for s, p in zip(self.shadow.parameters(), src.parameters()):
            s.lerp_(p.detach(), 1 - self.decay)
        for s, b in zip(self.shadow.buffers(), src.buffers()):
            s.copy_(b)


# -- Sampling ----------------------------------------------------------------

@torch.no_grad()
def sample(model, batch: Dict[str, torch.Tensor], K: int = 8, steps: int = 20, eta: float = 0.0,
           mode: str = "diffusion", seed: int = 0) -> torch.Tensor:
    """K samples per item. Returns P(dark) on the canvas, shape (B, K, CANVAS, CANVAS).
    Version-fixed modules are clamped to their known values at every step."""
    m = unwrap(model)
    dev = batch["img"].device
    B = batch["img"].shape[0]
    rep = lambda x: x.repeat_interleave(K, dim=0)
    cond = rep(m.enc(batch["img"]))
    kb = {k: rep(batch[k]) for k in ("known_val", "known_mask", "valid")}
    if mode == "direct":
        zeros = torch.zeros_like(kb["valid"])
        t = torch.zeros(B * K, dtype=torch.long, device=dev)
        logit, _ = m(zeros, t, sc=zeros, cond=cond, **kb)
        p = torch.sigmoid(logit.float())
    else:
        sched = make_train_scheduler()
        sched.set_timesteps(steps)
        g = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn(kb["valid"].shape, generator=g, device=dev)
        sc = torch.zeros_like(x)
        logit = None
        for t in sched.timesteps:
            tt = torch.full((B * K,), int(t), dtype=torch.long, device=dev)
            logit, _ = m(x, tt, sc=sc, cond=cond, **kb)
            x0 = _clamp_known(torch.tanh(logit.float() / 2), kb)
            sc = x0
            x = sched.step(x0, t, x, eta=eta, generator=g).prev_sample
        p = torch.sigmoid(logit.float())
    p = torch.where(kb["known_mask"] > 0, (kb["known_val"] > 0).float(), p)
    return p.view(B, K, CANVAS, CANVAS)


# -- Decoding (QR-aware, numpy) -----------------------------------------------

def decode_k(probs_k: np.ndarray, version: int, ks: Sequence[int], payload: Optional[str] = None) -> Dict[int, dict]:
    """probs_k: (K, n, n) P(dark) of K samples for one frame. For each k in
    `ks` (using the first k samples):
      verifier   any single sample decodes (RS checksum accepts it)
      posterior  majority vote over k samples + erasures where they disagree
    EC level and mask are READ from the grid's format info (not oracle).
    If `payload` is given, a decode is 'match' or 'wrong'; else 'ok'."""
    def status(res):
        if not res.ok:
            return "none"
        if payload is None:
            return "ok"
        return "match" if res.payload == payload else "wrong"

    bits_all = (probs_k > 0.5).astype(np.uint8)
    single = []
    for b in bits_all[: max(ks)]:
        e, m, _ = read_format(b, version)
        single.append(status(decode_grid(b, version, e, m)))
    out = {}
    for k in ks:
        p = probs_k[:k].mean(axis=0) if k > 1 else probs_k[0]
        bits = (p > 0.5).astype(np.uint8)
        e, m, _ = read_format(bits, version)
        res, used = decode_soft(bits, np.abs(p - 0.5), version, e, m)
        sk = single[:k]
        ver = "match" if "match" in sk else ("ok" if "ok" in sk else ("wrong" if "wrong" in sk else "none"))
        out[k] = {"verifier": ver, "posterior": status(res), "posterior_erasures": used if res.ok else None}
    return out


@torch.no_grad()
def run_eval(model, items: List[dict], k_list: Sequence[int], steps: int, device, mode: str = "diffusion",
             frames_per_batch: int = 8, seed: int = 0) -> List[dict]:
    """Sample max(k_list) grids per item and score them.

    items: dicts with `input` (u8 modspace crop), `version`, `payload`
    (or None), `grid` (GT or None) and any extra tags (copied to output).
    Module error is over symbol modules NOT fixed by the version (what the
    model actually has to infer), for the first sample and for the
    K-sample posterior mean."""
    import time
    model_eval = unwrap(model).eval()
    K = max(k_list)
    rows: List[dict] = []
    for s in range(0, len(items), frames_per_batch):
        chunk = items[s:s + frames_per_batch]
        batch = collate([make_example(it["input"], it.get("grid"), it["version"]) for it in chunk])
        batch = {k: v.to(device) for k, v in batch.items()}
        t0 = time.time()
        probs = sample(model_eval, batch, K=K, steps=steps, mode=mode, seed=seed + s).cpu().numpy()
        ms = 1000 * (time.time() - t0) / len(chunk)
        for it, pk in zip(chunk, probs):
            v = it["version"]
            sym = symbol_from_canvas(pk, v)
            row = {k: val for k, val in it.items() if k not in ("input", "grid")}
            row["ms_per_item_allK"] = ms
            if it.get("grid") is not None:
                vmask, _ = version_fixed(v)
                free = ~vmask
                gt = it["grid"]
                row["module_err_k1"] = float(((sym[0] > 0.5) != gt)[free].mean())
                row["module_err_post"] = float(((sym.mean(0) > 0.5) != gt)[free].mean())
            for k, r in decode_k(sym, v, k_list, it.get("payload")).items():
                row[f"verifier_k{k}"] = r["verifier"]
                row[f"posterior_k{k}"] = r["posterior"]
            rows.append(row)
    return rows


# -- Data --------------------------------------------------------------------

class PoolDataset(torch.utils.data.Dataset):
    """Synthetic training examples: drawn from an in-RAM pool of
    pre-generated samples, with probability `p_fresh` replaced by a freshly
    generated one (keeps adding new codes during training).
    Pool entries: (input_u8, grid_u8, version)."""

    def __init__(self, pool: List[tuple], length: int, p_fresh: float = 0.0, bank_path: Optional[str] = None,
                 seed: int = 0, sample_cfg=None):
        self.pool, self.length, self.p_fresh = pool, length, p_fresh
        self.bank_path, self.seed = bank_path, seed
        self.sample_cfg = sample_cfg   # diffuqr.modspace.ModuleSampleConfig for fresh samples (None = default)
        self._bank = None
        self._rng = None

    def __len__(self) -> int:
        return self.length

    def _worker_rng(self) -> random.Random:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = random.Random(self.seed * 1000 + wid + random.randrange(1 << 30))
        return self._rng

    def __getitem__(self, i: int):
        rng = self._worker_rng()
        if self.p_fresh > 0 and rng.random() < self.p_fresh:
            from diffuqr.modspace import KernelBank, ModuleSampleConfig, make_synthetic_sample
            if self._bank is None:
                self._bank = KernelBank.load(self.bank_path)
            s = make_synthetic_sample(rng, self._bank, self.sample_cfg or ModuleSampleConfig())
            return make_example(s["input"], s["target"], s["version"])
        inp, grid, version = self.pool[rng.randrange(len(self.pool))]
        return make_example(inp, grid, version)


_GEN_BANK = None
_GEN_CFG = None


def _gen_init(bank_path: str, cfg=None) -> None:
    global _GEN_BANK, _GEN_CFG
    from diffuqr.modspace import KernelBank, ModuleSampleConfig
    _GEN_BANK = KernelBank.load(bank_path)
    _GEN_CFG = cfg or ModuleSampleConfig()


def _gen_one(seed: int):
    from diffuqr.modspace import make_synthetic_sample
    s = make_synthetic_sample(random.Random(seed), _GEN_BANK, _GEN_CFG)
    return s["input"], s["target"], s["version"], s["payload"], float(s["kernel_len_modules"])


def generate_pool(n: int, bank_path: str, seed0: int, workers: int = 4, cfg=None) -> List[tuple]:
    """n synthetic samples in parallel. Returns [(input, grid, version, payload, kernel_len_modules)].
    Seeds seed0 .. seed0+n-1, so train/val pools with disjoint seed ranges never overlap.
    cfg: diffuqr.modspace.ModuleSampleConfig (None = default)."""
    import multiprocessing as mp
    seeds = range(seed0, seed0 + n)
    if workers <= 1:
        _gen_init(bank_path, cfg)
        return [_gen_one(s) for s in seeds]
    with mp.get_context("fork" if hasattr(__import__("os"), "fork") else "spawn").Pool(
            workers, initializer=_gen_init, initargs=(bank_path, cfg)) as pool:
        return list(pool.imap(_gen_one, seeds, chunksize=16))
