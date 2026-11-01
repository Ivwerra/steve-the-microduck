"""
brain.py -- a spiking duck brain with a fixed, Dale-signed connectome.

Topology is frozen. Only synaptic magnitudes, input/readout projections and
per-neuron bias train. Excitatory neurons stay excitatory, inhibitory stay
inhibitory (weight = sign * softplus(raw)).

Connectome file format (safetensors):
    pre, post   int64   [E]   synapse endpoints (global neuron ids)
    exc         uint8   [N]   1 = excitatory neuron
    offsets     int64   [R+1] region boundaries in neuron-id space
    metadata["regions"] = json list of region names (same order as offsets)
Any measured or inferred wiring that fits this layout can replace the default.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

# name: (fraction of neurons, excitatory fraction)
REGIONS = {
    "PrV":    (0.04, 0.85),  # principal trigeminal nucleus: bill-tip Herbst corpuscle relay
    "TeO":    (0.10, 0.80),  # optic tectum: orienting, looming
    "Rt":     (0.04, 0.85),  # nucleus rotundus
    "Ento":   (0.06, 0.80),  # entopallium: tectofugal visual pallium
    "Wulst":  (0.07, 0.80),  # thalamofugal visual pallium
    "L2":     (0.04, 0.85),  # field L2: primary auditory pallium
    "NCL":    (0.12, 0.80),  # nidopallium caudolaterale: executive / working memory
    "Hp":     (0.05, 0.80),  # hippocampus: place, food memory
    "Str":    (0.08, 0.15),  # striatum: GABAergic action selection
    "Amy":    (0.04, 0.75),  # nucleus taeniae / arcopallial amygdala: threat, social
    "Hyp":    (0.03, 0.80),  # hypothalamus: hunger / arousal drive
    "Arco":   (0.08, 0.85),  # arcopallium: pallial motor output
    "AI":     (0.03, 0.85),  # nucleus intermedius arcopallialis: vocal pallial
    "DM":     (0.03, 0.85),  # dorsomedial nucleus of the intercollicular complex
    "nXIIts": (0.03, 0.90),  # tracheosyringeal hypoglossal motor nucleus -> syrinx
    "RF":     (0.06, 0.80),  # reticular formation: locomotor / gait pattern
    "Cb":     (0.10, 0.90),  # cerebellum: timing, balance
}

# (src, dst): relative projection strength (sets share of a target region's in-degree)
PROJ = {
    ("PrV", "NCL"): .5, ("PrV", "RF"): .6, ("PrV", "TeO"): .2,
    ("TeO", "Rt"): .9, ("TeO", "RF"): .6, ("TeO", "Amy"): .3, ("TeO", "Cb"): .2,
    ("Rt", "Ento"): .9, ("Ento", "NCL"): .6, ("Ento", "Arco"): .4,
    ("Wulst", "NCL"): .5, ("Wulst", "Arco"): .4, ("Wulst", "Str"): .3,
    ("L2", "NCL"): .5, ("L2", "Amy"): .4, ("L2", "AI"): .3,
    ("NCL", "Arco"): .7, ("NCL", "Str"): .7, ("NCL", "Hp"): .4,
    ("Hp", "NCL"): .5, ("Hp", "Hyp"): .2,
    ("Amy", "Hyp"): .6, ("Amy", "Arco"): .5, ("Amy", "DM"): .3,
    ("Hyp", "Amy"): .4, ("Hyp", "NCL"): .3, ("Hyp", "DM"): .3,
    ("Str", "RF"): .4, ("Str", "NCL"): .5,
    ("Arco", "RF"): .7, ("Arco", "AI"): .5, ("Arco", "Cb"): .4,
    ("AI", "DM"): .9, ("DM", "nXIIts"): .9, ("DM", "RF"): .3,
    ("RF", "Cb"): .4, ("Cb", "RF"): .6, ("Cb", "NCL"): .2,
}
LOCAL = 0.8  # intra-region recurrence strength

SENSORY = {"PrV": "touch", "TeO": "vision", "L2": "audio", "Hyp": "intero"}
LOCO_OUT = ["RF", "Cb", "Arco"]
VOICE_OUT = ["nXIIts", "DM"]
N_LOCO, N_VOICE = 6, 4  # [speed, turn, dabble, flap, neck_yaw, neck_pitch], [gate, f0, formant, rough]


def build_connectome(n: int = 4096, k_in: int = 48, seed: int = 0) -> dict:
    """Sample a neuron-resolution connectome from the region-level projection prior.
    Every neuron draws k_in synapses; source region ~ Categorical(strengths of
    that target's incoming projections), source neuron ~ Uniform(region)."""
    rng = np.random.default_rng(seed)
    names = list(REGIONS)
    idx = {r: i for i, r in enumerate(names)}
    counts = np.array([max(8, round(REGIONS[r][0] * n)) for r in names])
    offs = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    N = int(offs[-1])

    exc = np.zeros(N, np.uint8)
    for i, r in enumerate(names):
        exc[offs[i]: offs[i] + int(round(REGIONS[r][1] * counts[i]))] = 1

    incoming = {r: [(idx[r], LOCAL)] for r in names}
    for (s, d), w in PROJ.items():
        incoming[d].append((idx[s], w))

    pre, post = [], []
    for d in names:
        di = idx[d]
        srcs, w = map(np.array, zip(*incoming[d]))
        w = w / w.sum()
        cnt = counts[di]
        reg = srcs[rng.choice(len(srcs), size=(cnt, k_in), p=w)]
        pre.append(offs[reg] + (rng.random((cnt, k_in)) * counts[reg]).astype(np.int64))
        post.append(np.repeat(np.arange(offs[di], offs[di + 1]), k_in))
    return {
        "pre": torch.from_numpy(np.concatenate([p.ravel() for p in pre])),
        "post": torch.from_numpy(np.concatenate(post).astype(np.int64)),
        "exc": torch.from_numpy(exc),
        "offsets": torch.from_numpy(offs),
        "regions": names,
    }


def save_connectome(conn: dict, path: str) -> None:
    t = {k: conn[k].contiguous() for k in ("pre", "post", "exc", "offsets")}
    save_file(t, path, metadata={"regions": json.dumps(conn["regions"])})


def load_connectome(path: str) -> dict:
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        names = json.loads(f.metadata()["regions"])
    conn = load_file(path)
    conn["regions"] = names
    return conn


class SpikeFn(torch.autograd.Function):
    """Heaviside forward, SuperSpike-style fast-sigmoid surrogate backward."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1.0 + 10.0 * x.abs()) ** 2


spike = SpikeFn.apply


class DuckBrain(nn.Module):
    def __init__(self, conn: dict, in_dims: dict | None = None, dt_ms: float = 5.0,
                 substeps: int = 10, tau_m: float = 20.0, tau_s: float = 5.0,
                 tau_out: float = 40.0, v_th: float = 1.0):
        super().__init__()
        in_dims = in_dims or {"touch": 5, "vision": 48, "audio": 10, "intero": 2}
        self.names, self.offs = conn["regions"], conn["offsets"].tolist()
        self.N, self.substeps, self.v_th = self.offs[-1], substeps, v_th
        self.register_buffer("pre", conn["pre"].long())
        self.register_buffer("post", conn["post"].long())
        exc = conn["exc"].float()
        self.register_buffer("exc", exc)
        self.register_buffer("sign", (2 * exc - 1)[self.pre])  # sign follows presynaptic neuron

        k_in = self.pre.numel() / self.N
        mag = (torch.randn(self.pre.numel()).abs() + 0.2) * 1.6 / math.sqrt(k_in)
        self.w_raw = nn.Parameter(torch.log(torch.expm1(mag)))
        self.bias = nn.Parameter(torch.zeros(self.N))

        self.beta_m = math.exp(-dt_ms / tau_m)
        self.beta_s = math.exp(-dt_ms / tau_s)
        self.beta_o = math.exp(-dt_ms * substeps / tau_out)

        self.inp = nn.ModuleDict()
        for reg, grp in SENSORY.items():
            r = self.names.index(reg)
            self.inp[reg] = nn.Linear(in_dims[grp], self.offs[r + 1] - self.offs[r])
        self.group_of = SENSORY

        def ids(regs):
            return torch.cat([torch.arange(self.offs[self.names.index(r)],
                                           self.offs[self.names.index(r) + 1]) for r in regs])
        self.register_buffer("loco_idx", ids(LOCO_OUT))
        self.register_buffer("voice_idx", ids(VOICE_OUT))
        self.head_loco = nn.Linear(len(self.loco_idx), N_LOCO)
        self.head_voice = nn.Linear(len(self.voice_idx), N_VOICE)

    def init_state(self, B: int, device=None) -> dict:
        z = lambda: torch.zeros(B, self.N, device=device or self.pre.device)
        return {"v": z(), "i": z(), "s": z(), "rate": z()}

    def _external(self, obs: dict) -> torch.Tensor:
        parts = []
        for r, name in enumerate(self.names):
            n = self.offs[r + 1] - self.offs[r]
            if name in self.inp:
                parts.append(self.inp[name](obs[self.group_of[name]]))
            else:
                parts.append(obs["vision"].new_zeros(obs["vision"].shape[0], n))
        return torch.cat(parts, 1) + self.bias

    def forward(self, obs: dict, st: dict):
        ext = self._external(obs)
        w = self.sign * F.softplus(self.w_raw)
        v, i, s, rate = st["v"], st["i"], st["s"], st["rate"]
        acc = torch.zeros_like(v)
        for _ in range(self.substeps):
            syn = torch.zeros_like(v).index_add_(1, self.post, s[:, self.pre] * w)
            i = self.beta_s * i + syn
            v = self.beta_m * v + (1 - self.beta_m) * (i + ext) * 4.0
            s = spike(v - self.v_th)
            v = v - s.detach() * self.v_th  # soft reset
            acc = acc + s
        rate = self.beta_o * rate + (1 - self.beta_o) * acc / self.substeps
        scale = 20.0
        out = {
            "loco": self.head_loco(rate[:, self.loco_idx] * scale),
            "voice": self.head_voice(rate[:, self.voice_idx] * scale),
            "mean_rate": acc.mean() / self.substeps,
        }
        return out, {"v": v, "i": i, "s": s, "rate": rate}

    @staticmethod
    def detach(st: dict) -> dict:
        return {k: x.detach() for k, x in st.items()}
