"""
voice.py -- differentiable syrinx and the spectral prior that shapes quacks.

nXIIts/DM population rates -> 4 control signals per 50 ms brain step
    gate     0..1      expiratory drive / syringeal opening
    f0       250-700 Hz  fundamental (hen mallard quacks sit ~300-600 Hz)
    formant  800-2800 Hz vocal-tract resonance (tracheal length / bill gape)
    rough    0..1      harshness: AM at ~90 Hz + pitch jitter, the "rasp" of a quack

Controls are upsampled to audio rate and drive a harmonic source through a
Gaussian formant filter. Everything is differentiable, so the vocal loss
backpropagates into the spiking network.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

SR = 16000
HOP_S = 0.05  # one brain/control step


def decode_voice(raw: torch.Tensor) -> torch.Tensor:
    """raw [...,4] head output -> physical controls [gate, f0_hz, formant_hz, rough]."""
    g = torch.sigmoid(raw[..., 0])
    f0 = 250.0 + 450.0 * torch.sigmoid(raw[..., 1])
    fm = 800.0 + 2000.0 * torch.sigmoid(raw[..., 2])
    ro = torch.sigmoid(raw[..., 3])
    return torch.stack([g, f0, fm, ro], -1)


class Syrinx(nn.Module):
    def __init__(self, sr: int = SR, n_harm: int = 24, am_hz: float = 90.0):
        super().__init__()
        self.sr, self.H, self.am_hz = sr, n_harm, am_hz
        self.register_buffer("k", torch.arange(1, n_harm + 1).float())

    def forward(self, ctrl: torch.Tensor) -> torch.Tensor:
        """ctrl [B,T,4] (decoded) -> waveform [B, T*sr*HOP_S]"""
        B, T, _ = ctrl.shape
        n = int(self.sr * HOP_S) * T
        up = torch.nn.functional.interpolate(ctrl.transpose(1, 2), size=n, mode="linear",
                                             align_corners=False)  # [B,4,n]
        g, f0, fm, ro = up[:, 0], up[:, 1], up[:, 2], up[:, 3]
        t = torch.arange(n, device=ctrl.device) / self.sr

        f0 = f0 * (1.0 + 0.03 * ro * torch.sin(2 * math.pi * 37.0 * t))  # jitter
        phase = 2 * math.pi * torch.cumsum(f0, 1) / self.sr  # [B,n]
        freqs = f0[..., None] * self.k  # [B,n,H]
        env = torch.exp(-0.5 * ((freqs - fm[..., None]) / 450.0) ** 2) + 0.15 / self.k
        env = env * (freqs < self.sr / 2 * 0.95)
        harm = (env * torch.sin(phase[..., None] * self.k)).sum(-1) / 4.0
        am = 1.0 - ro * 0.5 * (1.0 - torch.sin(2 * math.pi * self.am_hz * t))  # rasp
        amp = g ** 2  # gate -> pressure -> amplitude
        return amp * am * harm


def _logspec(wave: torch.Tensor, n_fft: int = 512, hop: int = 128):
    win = torch.hann_window(n_fft, device=wave.device)
    S = torch.stft(wave, n_fft, hop, window=win, return_complex=True).abs()
    return torch.log(S + 1e-5), S  # [B,F,T]


def spectral_prior_loss(wave: torch.Tensor, ref_logspec: torch.Tensor, thr: float = 0.5):
    """Mean spectral shape of voiced frames should match the reference quack shape.
    Level-invariant: both sides are mean-centered across frequency."""
    L, S = _logspec(wave)
    rms = S.pow(2).mean(1).sqrt()  # [B,T]
    w = (rms > thr).float().detach()
    n = w.sum(1)
    if (n > 0).sum() == 0:
        return wave.sum() * 0.0
    shape = (L * w[:, None]).sum(2) / n.clamp_min(1)[:, None]
    shape = shape - shape.mean(1, keepdim=True)
    ref = ref_logspec - ref_logspec.mean()
    per = (shape - ref).abs().mean(1)
    return (per * (n > 0)).sum() / (n > 0).sum()


def template_quack(syrinx: Syrinx, dur: float = 0.22) -> torch.Tensor:
    """Canonical hen quack: pitch falls ~520 -> 380 Hz, rasp-heavy, short, decrescendo."""
    T = int(dur / HOP_S) + 1
    x = torch.linspace(0, 1, T)
    ctrl = torch.stack([torch.sigmoid(6 * (1 - x) - 1.0),
                        520 - 140 * x,
                        1300 + 400 * (1 - x),
                        torch.full_like(x, 0.65)], -1)[None]
    return syrinx(ctrl)


@torch.no_grad()
def reference_logspec(syrinx: Syrinx, dataset_id: str | None = None, split: str = "train",
                      n: int = 64, device="cpu") -> torch.Tensor:
    """Mean voiced log-spectrum of real duck recordings (HF audio dataset) or the
    procedural template if no dataset is given."""
    waves = []
    if dataset_id:
        from datasets import Audio, load_dataset
        ds = load_dataset(dataset_id, split=split).cast_column("audio", Audio(sampling_rate=SR))
        for row in ds.select(range(min(n, len(ds)))):
            a = torch.tensor(row["audio"]["array"], dtype=torch.float32)
            if a.numel() > 512:
                waves.append(a / (a.abs().max() + 1e-6) * 0.5)
    else:
        waves.append(template_quack(Syrinx())[0])
    shapes = []
    for a in waves:
        L, S = _logspec(a[None].to(device))
        rms = S.pow(2).mean(1).sqrt()[0]
        m = rms > 0.25 * rms.max()
        if m.any():
            shapes.append(L[0][:, m].mean(1))
    ref = torch.stack(shapes).mean(0)
    return ref - ref.mean()
