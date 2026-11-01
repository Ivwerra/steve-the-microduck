"""
world.py -- a differentiable pond.

Fully batched torch dynamics, so reward gradients flow through the world into
the spiking brain (BPTT, no policy-gradient variance). Contents: dabbling
food patches, a stalking raptor, a calling conspecific, a shoreline.

Sensors (encoded the way the corresponding duck pathway would see them):
  vision  16 bearing bins over a 300 deg field x {food, predator, conspecific},
          amplitude ~ 1/(1+d) so approach looks like looming          -> TeO
  touch   bill-tip food contact, shore proximity, efference copy      -> PrV
  audio   8 cochlear bins around the caller's f0 + interaural diff    -> L2
  intero  satiety, hunger                                             -> Hyp

Effectors: 6 locomotor commands -> 12 microduck servo targets (waddle CPG +
neck), 4 vocal controls -> syrinx (voice.py).
"""
from __future__ import annotations

import math

import torch

from voice import decode_voice

N_BINS, FOV = 16, math.radians(300)
JOINTS = [f"{s}_{j}" for s in ("L", "R") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")] \
         + ["neck_yaw", "neck_pitch"]  # 12 servos, order = servos.npz columns


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


class DuckWorld:
    def __init__(self, B: int, device="cpu", n_food: int = 6, R: float = 5.0, dt: float = 0.05):
        self.B, self.dev, self.F, self.R, self.dt = B, device, n_food, R, dt
        self.vmax, self.wmax = 1.2, 3.0
        self.centers = torch.linspace(-FOV / 2, FOV / 2, N_BINS, device=device)
        self.sigma = FOV / N_BINS
        self.f_bins = torch.linspace(250, 700, 8, device=device)

    # ------------------------------------------------------------------ state
    def _rand_in_pond(self, *shape, r=0.85):
        rad = self.R * r * torch.sqrt(torch.rand(*shape, device=self.dev))
        th = 2 * math.pi * torch.rand(*shape, device=self.dev)
        return torch.stack([rad * torch.cos(th), rad * torch.sin(th)], -1)

    def reset(self):
        B, d = self.B, self.dev
        self.pos = self._rand_in_pond(B, r=0.5)
        self.heading = 2 * math.pi * torch.rand(B, device=d)
        self.food = self._rand_in_pond(B, self.F)
        self.amount = torch.ones(B, self.F, device=d)
        self.energy = 0.5 * torch.ones(B, device=d)
        self.phase = torch.zeros(B, device=d)
        self.pred_phi = 2 * math.pi * torch.rand(B, device=d)
        self.pred = 1.6 * self.R * torch.stack([torch.cos(self.pred_phi), torch.sin(self.pred_phi)], -1)
        self.pred_active = torch.zeros(B, device=d)
        self.mate = self._rand_in_pond(B)
        self.call_left = torch.zeros(B, device=d)  # steps of call remaining
        self.silence = 20 * torch.rand(B, device=d)
        self.mate_f0 = 350 + 250 * torch.rand(B, device=d)
        self.since_call = 99.0 * torch.ones(B, device=d)
        self.prev = torch.zeros(B, 3, device=d)  # speed, turn, dabble efference
        return self.observe()

    def detach(self):
        for k in ("pos", "heading", "food", "amount", "energy", "phase", "pred", "prev"):
            setattr(self, k, getattr(self, k).detach())

    # ---------------------------------------------------------------- sensing
    def _bearing(self, tgt, w=None):
        """tgt [B,K,2] -> [B,N_BINS] retinal activation summed over K."""
        d = tgt - self.pos[:, None]
        dist = d.norm(dim=-1).clamp_min(1e-2)
        ang = wrap(torch.atan2(d[..., 1], d[..., 0]) - self.heading[:, None])
        a = torch.exp(-0.5 * ((ang[..., None] - self.centers) / self.sigma) ** 2)
        a = a / (1 + dist[..., None])
        if w is not None:
            a = a * w[..., None]
        return a.sum(1)

    def observe(self):
        fwd = torch.stack([torch.cos(self.heading), torch.sin(self.heading)], -1)
        bill = self.pos + 0.15 * fwd
        contact = (torch.exp(-((self.food - bill[:, None]) ** 2).sum(-1) / (2 * 0.15 ** 2)) * self.amount).sum(1)
        shore = (self.pos.norm(dim=-1) / self.R).clamp(0, 1.5) ** 4

        calling = (self.call_left > 0).float()
        dm = self.mate - self.pos
        ang = wrap(torch.atan2(dm[:, 1], dm[:, 0]) - self.heading)
        loud = calling / (1 + dm.norm(dim=-1))
        coch = torch.exp(-0.5 * ((self.mate_f0[:, None] - self.f_bins) / 60.0) ** 2) * loud[:, None]
        ild = torch.stack([loud * torch.sin(ang).clamp(min=0), loud * (-torch.sin(ang)).clamp(min=0)], -1)

        vision = torch.cat([self._bearing(self.food, self.amount),
                            self._bearing(self.pred[:, None], 1 + self.pred_active[:, None]),
                            self._bearing(self.mate[:, None])], -1)
        return {
            "vision": vision * 3.0,
            "touch": torch.cat([contact[:, None], shore[:, None], self.prev], -1),
            "audio": torch.cat([coch, ild], -1) * 3.0,
            "intero": torch.stack([self.energy, 1 - self.energy], -1),
        }

    # --------------------------------------------------------------- dynamics
    def step(self, out: dict):
        B, dt = self.B, self.dt
        a, vc = out["loco"], decode_voice(out["voice"])
        speed_c, turn = torch.sigmoid(a[:, 0]), torch.tanh(a[:, 1])
        dab, flap = torch.sigmoid(a[:, 2]), torch.sigmoid(a[:, 3])
        neck = torch.tanh(a[:, 4:6])

        v = self.vmax * speed_c * (1 - 0.7 * dab) * (1 + 1.2 * flap)
        self.heading = self.heading + self.wmax * turn * dt
        fwd = torch.stack([torch.cos(self.heading), torch.sin(self.heading)], -1)
        pos = self.pos + v[:, None] * fwd * dt
        r = pos.norm(dim=-1, keepdim=True)
        over = torch.relu(r.squeeze(-1) - self.R)
        self.pos = pos * (self.R / torch.clamp(r, min=self.R))  # project back into pond
        self.phase = self.phase + 2 * math.pi * (0.5 + 2.5 * speed_c.detach()) * dt

        # dabbling: bill-reach gaussian x head-down command
        bill = self.pos + 0.15 * fwd
        reach = torch.exp(-((self.food - bill[:, None]) ** 2).sum(-1) / (2 * 0.15 ** 2))
        eaten = (dab[:, None] * reach * self.amount * 0.4).clamp(max=0.4)
        self.amount = self.amount - eaten.detach()
        gone = (self.amount < 0.1)
        with torch.no_grad():
            self.food = torch.where(gone[..., None], self._rand_in_pond(B, self.F), self.food)
            self.amount = torch.where(gone, torch.ones_like(self.amount), self.amount)
        ate = eaten.sum(1)
        hunger = 1.5 - self.energy
        r_food = ate * hunger
        self.energy = (self.energy * 0.998 + 0.25 * ate.detach() - 0.002 * flap.detach()).clamp(0, 1)

        # raptor: orbits, bursts into stalks, strikes at < 0.4
        with torch.no_grad():
            toggle = (torch.rand(B, device=self.dev) < 0.01).float()
            self.pred_active = (self.pred_active + toggle) % 2
            self.pred_phi = self.pred_phi + 0.2 * dt
            orbit = 1.6 * self.R * torch.stack([torch.cos(self.pred_phi), torch.sin(self.pred_phi)], -1)
            tgt = torch.where(self.pred_active[:, None] > 0, self.pos.detach(), orbit)
            dv = tgt - self.pred
            dist = dv.norm(dim=-1, keepdim=True).clamp_min(1e-3)
            self.pred = self.pred + dv / dist * torch.minimum(dist, torch.full_like(dist, 2.4 * dt))
            hit = ((self.pred - self.pos).norm(dim=-1) < 0.4).float() * self.pred_active
            self.pred = torch.where(hit[:, None] > 0, orbit, self.pred)
            self.pred_active = self.pred_active * (1 - hit)
            self.energy = (self.energy - 0.2 * hit).clamp(0, 1)
        danger = torch.exp(-((self.pred - self.pos) ** 2).sum(-1) / (2 * 1.5 ** 2)) * self.pred_active

        # conspecific: periodic contact calls; reward = vocal reply with matched pitch
        with torch.no_grad():
            self.call_left = (self.call_left - 1).clamp(min=0)
            self.silence = self.silence - 1
            start = (self.silence <= 0) & (self.call_left <= 0)
            self.call_left = torch.where(start, torch.full_like(self.call_left, 6), self.call_left)
            self.mate_f0 = torch.where(start, 350 + 250 * torch.rand(B, device=self.dev), self.mate_f0)
            self.silence = torch.where(start, 30 + 60 * torch.rand(B, device=self.dev), self.silence)
            self.since_call = torch.where(self.call_left > 0, torch.zeros_like(self.since_call), self.since_call + 1)
            self.mate = self.mate + 0.05 * dt * torch.randn_like(self.mate)
        window = ((self.call_left > 0) | (self.since_call < 10)).float()
        gate, f0 = vc[:, 0], vc[:, 1]
        match = torch.exp(-0.5 * ((f0 - self.mate_f0) / 60.0) ** 2)
        r_reply = window * gate * match
        r_social = 0.05 * torch.exp(-((self.mate - self.pos) ** 2).sum(-1) / (2 * 2.0 ** 2))

        r_cost = 0.02 * v ** 2 + 0.05 * flap + 0.3 * over
        r_pred = 1.0 * danger + 5.0 * hit + 0.5 * gate * danger  # hush near the hawk
        reward = r_food + 0.8 * r_reply + r_social - r_cost - r_pred
        self.prev = torch.stack([speed_c, turn, dab], -1)

        info = {"reward": reward, "food": r_food.detach(), "reply": r_reply.detach(),
                "danger": danger.detach(), "hit": hit, "gate": gate, "ctrl": vc,
                "servo": self.joint_targets(speed_c, turn, dab, flap, neck)}
        return self.observe(), info

    # ------------------------------------------------------- microduck output
    def joint_targets(self, speed, turn, dab, flap, neck):
        """Waddle CPG: alternating legs, exaggerated hip-roll sway (the duck gait),
        stride scaled by speed, turn via hip-yaw asymmetry. Radians, [B,12]."""
        p = self.phase.detach()
        amp = (0.15 + 0.45 * speed).detach()
        cols = []
        for side, off in ((+1, 0.0), (-1, math.pi)):
            ph = p + off
            cols += [0.25 * turn.detach() * side + 0.05 * torch.sin(ph),   # hip_yaw
                     0.22 * torch.sin(ph + math.pi / 2) * (0.4 + speed.detach()),  # hip_roll sway
                     amp * torch.sin(ph),                                    # hip_pitch
                     amp * 1.4 * torch.relu(torch.sin(ph + 0.6)),            # knee (swing flexion)
                     -0.5 * amp * torch.sin(ph + 0.3)]                       # ankle
        cols += [neck[:, 0], -1.2 * dab + 0.3 * neck[:, 1]]  # neck_yaw, neck_pitch (head-down dabble)
        return torch.stack(cols, -1)
