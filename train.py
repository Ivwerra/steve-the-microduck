"""
train.py -- train the duck brain through the differentiable pond, export
servo + audio rollouts, and push everything to the Hugging Face Hub.

    python train.py --iters 2000 --repo_id you/microduck-brain
    python train.py --connectome my_duck.safetensors --quack_ds you/mallard-calls
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from safetensors.torch import save_file

from brain import DuckBrain, build_connectome, load_connectome, save_connectome
from voice import SR, Syrinx, reference_logspec, spectral_prior_loss
from world import JOINTS, DuckWorld


@dataclass
class Cfg:
    n_neurons: int = 2048
    k_in: int = 48
    connectome: Optional[str] = None     # load measured wiring instead of sampling the prior
    batch: int = 16
    episode: int = 200                   # control steps (50 ms) per episode = 10 s
    tbptt: int = 20
    iters: int = 2000
    lr: float = 3e-3
    target_rate: float = 0.03            # homeostatic spikes / neuron / substep
    w_rate: float = 200.0
    w_voice: float = 0.5
    voice_batch: int = 8                 # batch elements synthesized for the vocal loss
    quack_ds: Optional[str] = None       # HF audio dataset of real duck calls (column "audio")
    seed: int = 0
    out: str = "runs/microduck-v0"
    repo_id: Optional[str] = None
    private: bool = False


def write_wav(path: str, x: np.ndarray, sr: int = SR):
    x = x / (np.abs(x).max() + 1e-6) * 0.9
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes((x * 32767).astype(np.int16).tobytes())


@torch.no_grad()
def export_rollout(brain, syrinx, cfg, dev, steps=400):
    """Greedy rollout of one duck: servos.npz (20 Hz, rad) + quacks.wav."""
    w = DuckWorld(1, dev); obs = w.reset(); st = brain.init_state(1, dev)
    servos, ctrl, score = [], [], 0.0
    for _ in range(steps):
        out, st = brain(obs, st)
        obs, info = w.step(out)
        servos.append(info["servo"][0].cpu()); ctrl.append(info["ctrl"][0].cpu())
        score += info["reward"].item()
    wave_ = syrinx(torch.stack(ctrl)[None].to(dev))[0].cpu().numpy()
    np.savez(f"{cfg.out}/servos.npz", servos=torch.stack(servos).numpy(), joints=np.array(JOINTS), hz=20)
    write_wav(f"{cfg.out}/quacks.wav", wave_)
    return score


def model_card(cfg, brain, hist):
    return f"""---
library_name: pytorch
tags: [spiking-neural-network, connectome, neuromorphic, robotics, microduck, bioacoustics]
---
# microduck-brain

Spiking duck brain ({brain.N} neurons, {brain.pre.numel()} synapses, {len(brain.names)} regions)
trained in a differentiable pond to forage, flee a raptor, and answer conspecific calls.

- `connectome.safetensors` frozen Dale-signed wiring (the brain data loaded onto the microduck)
- `brain_weights.safetensors` trained synaptic magnitudes, sensory projections, readout heads
- `servos.npz` 12-servo joint targets @ 20 Hz from a greedy rollout
- `quacks.wav` synthesized vocal output of that rollout

final: reward/step {hist['reward']:.3f}, reply {hist['reply']:.3f}, hits/ep {hist['hit']:.2f}, rate {hist['rate']:.4f}
"""


def main(cfg: Cfg):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.out, exist_ok=True)

    conn = load_connectome(cfg.connectome) if cfg.connectome else \
        build_connectome(cfg.n_neurons, cfg.k_in, cfg.seed)
    save_connectome(conn, f"{cfg.out}/connectome.safetensors")
    brain = DuckBrain(conn).to(dev)
    syrinx = Syrinx().to(dev)
    ref = reference_logspec(syrinx, cfg.quack_ds, device=dev).to(dev)
    world = DuckWorld(cfg.batch, dev)
    opt = torch.optim.AdamW(brain.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.iters)
    print(f"{brain.N} neurons | {brain.pre.numel()} synapses | {sum(p.numel() for p in brain.parameters())} params | {dev}")

    hist, t0 = {}, time.time()
    for it in range(1, cfg.iters + 1):
        obs, st = world.reset(), brain.init_state(cfg.batch, dev)
        agg = {k: 0.0 for k in ("reward", "reply", "hit", "rate", "vloss")}
        for c in range(0, cfg.episode, cfg.tbptt):
            R, ctrl, rates = [], [], []
            for _ in range(cfg.tbptt):
                out, st = brain(obs, st)
                obs, info = world.step(out)
                R.append(info["reward"]); ctrl.append(info["ctrl"]); rates.append(out["mean_rate"])
                agg["reply"] += info["reply"].mean().item(); agg["hit"] += info["hit"].sum().item() / cfg.batch
            reward = torch.stack(R).mean()
            ctrl_t = torch.stack(ctrl, 1)                                   # [B,T,4]
            wav = syrinx(ctrl_t[: cfg.voice_batch])
            vloss = spectral_prior_loss(wav, ref) + torch.relu(ctrl_t[..., 0].mean() - 0.15)
            rate = torch.stack(rates).mean()
            loss = -reward + cfg.w_voice * vloss + cfg.w_rate * (rate - cfg.target_rate) ** 2

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            opt.step()
            st = brain.detach(st); world.detach()
            agg["reward"] += reward.item(); agg["rate"] += rate.item(); agg["vloss"] += vloss.item()
        sched.step()

        n_chunks = cfg.episode // cfg.tbptt
        hist = {"reward": agg["reward"] / n_chunks, "reply": agg["reply"] / cfg.episode,
                "hit": agg["hit"], "rate": agg["rate"] / n_chunks, "vloss": agg["vloss"] / n_chunks}
        if it % 10 == 0 or it == 1:
            print(f"it {it:5d} | " + " | ".join(f"{k} {v:.4f}" for k, v in hist.items()) +
                  f" | {time.time() - t0:.0f}s")
        if it % 250 == 0:
            save_file({k: v.detach().cpu().contiguous() for k, v in brain.named_parameters()},
                      f"{cfg.out}/brain_weights.safetensors")

    save_file({k: v.detach().cpu().contiguous() for k, v in brain.named_parameters()},
              f"{cfg.out}/brain_weights.safetensors")
    score = export_rollout(brain, syrinx, cfg, dev)
    json.dump({**dataclasses.asdict(cfg), "eval_return": score, "joints": JOINTS},
              open(f"{cfg.out}/config.json", "w"), indent=2)
    open(f"{cfg.out}/README.md", "w").write(model_card(cfg, brain, hist))
    print(f"eval return {score:.2f} -> {cfg.out}")

    if cfg.repo_id:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(cfg.repo_id, private=cfg.private, exist_ok=True)
        api.upload_folder(folder_path=cfg.out, repo_id=cfg.repo_id)
        print(f"pushed https://huggingface.co/{cfg.repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(Cfg):
        t = {"int": int, "float": float, "str": str, "bool": bool, "Optional[str]": str}[str(f.type)]
        if t is bool:
            ap.add_argument(f"--{f.name}", action="store_true")
        else:
            ap.add_argument(f"--{f.name}", type=t, default=f.default)
    main(Cfg(**vars(ap.parse_args())))
