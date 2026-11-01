# microduck - connectome prototype

<img src="https://i.ibb.co/KjV7Qs50/ec9d6623-385a-400d-92d2-9183375fa2c9.png" alt="Chucho" width="20" />

A 17-region spiking duck brain with a connectome, trained end-to-end through a differentiable pond until a microduck robot forages, flees raptors, and answers other ducks in their own voice.

```
 pond ──► sensors ──► [ PrV  TeO  L2  Hyp ] ──► pallium / striatum / limbic ──► [ RF  Cb  Arco | AI→DM→nXIIts ]
 (world.py)            bill  eye  ear  gut        NCL Hp Str Amy Ento Wulst        legs+neck │      syrinx
    ▲                                              (brain.py, 2048 LIF neurons)             │        │
    └──── 12 servo targets @ 20 Hz  ◄── waddle CPG ◄─────────────────────────────────────────┘        ▼
                                                             differentiable syrinx (voice.py) ──► quacks.wav
```

## Files

| file | role |
|---|---|
| `brain.py` | region table, projection prior, neuron-level connectome sampler, Dale-signed LIF network with surrogate gradients |
| `world.py` | batched differentiable pond: food patches, raptor, calling mate, sensory encoders, waddle gait → 12 servo targets |
| `voice.py` | differentiable harmonic syrinx (f0, formant, rasp), spectral-shape prior from real or template quacks |
| `train.py` | BPTT training loop, rate homeostasis, rollout export, Hub push |
| `README.md` | this |

## The brain

Topology is frozen; learning only moves synaptic magnitudes, sensory projections, per-neuron bias and two linear readouts. Signs follow the presynaptic neuron (Dale), so a striatal neuron can never learn to excite.

- **Connectome**: each neuron draws `k_in` synapses. Source region ~ Categorical over that target's incoming projection strengths (`PROJ` in `brain.py`), source neuron ~ Uniform within the region. Region sizes and E/I ratios are in `REGIONS`.
- **Sensory entry points**: bill-tip touch → `PrV`, vision → `TeO`, audition → `L2`, satiety/hunger → `Hyp`.
- **Motor exits**: locomotion read from `RF + Cb + Arco`; voice read from `nXIIts + DM`. The vocal path `Arco → AI → DM → nXIIts` mirrors the descending pallio-bulbar route to the syrinx.
- **Dynamics**: 5 ms LIF substeps, 10 per 50 ms control step, exponential synaptic current, soft reset, fast-sigmoid surrogate. Readouts see low-passed population rates (τ = 40 ms).
- **Homeostasis**: a quadratic penalty holds mean rate near `--target_rate` so the network doesn't saturate or go silent during early training.

### Bring your own wiring

`connectome.safetensors` is the brain data the microduck loads. Layout:

```
pre, post   int64 [E]      synapse endpoints (global neuron ids)
exc         uint8 [N]      1 = excitatory neuron
offsets     int64 [R+1]    region boundaries
metadata["regions"]        JSON list of region names, same order as offsets
```

Any inferred or measured duck/avian wiring in this layout drops in with `--connectome path.safetensors`. Input and output region names (`PrV TeO L2 Hyp`, `RF Cb Arco`, `nXIIts DM`) must exist.

## The pond

| sense | encoding | enters |
|---|---|---|
| vision | 16 bearing bins × 300° field × {food, raptor, mate}, amplitude 1/(1+d) | `TeO` |
| touch | bill-tip food contact, shore proximity, efference copy | `PrV` |
| audio | 8 cochlear bins over the caller's f0, interaural level difference | `L2` |
| intero | satiety, hunger | `Hyp` |

Everything is torch, batched, and differentiable (soft dabbling reach, gaussian danger field, radial pond projection), so reward gradients flow through the world into spikes with BPTT instead of high-variance policy gradients. The raptor orbits, bursts into stalks, and strikes inside 0.4 m.

Reward per step:

```
+ hunger-weighted food eaten (dabble command × bill reach × patch)
+ 0.8 × reply        (call window × vocal gate × pitch match to the mate's f0)
+ 0.05 × cohesion    (stay near the mate)
− 0.02 v² − 0.05 flap − shore penalty
− danger − 5 × strike − 0.5 × gate × danger     (hush near the hawk)
```

## Motor and voice

Six locomotor channels (`speed, turn, dabble, flap, neck_yaw, neck_pitch`) feed a waddle CPG: legs in antiphase, hip-roll sway phase-shifted 90° from hip pitch, knee flexion on swing, turn through hip-yaw asymmetry, head-down on dabble. Output is 12 joint targets in radians (`world.JOINTS`) at 20 Hz.

Four vocal channels drive `Syrinx`: a 24-harmonic source with jitter and 90 Hz amplitude-modulated rasp, shaped by a Gaussian formant, amplitude ∝ gate². The vocal loss compares the mean voiced log-spectral shape against a reference quack spectrum, from the procedural hen-quack template by default or from a real recordings dataset with `--quack_ds user/dataset` (needs an `audio` column; resampled to 16 kHz). A duty-cycle penalty keeps calls bursty.

## Run

```bash
pip install torch safetensors huggingface_hub datasets numpy
python train.py                                   # ~2000 iters, writes runs/microduck-v0/
python train.py --repo_id you/microduck-brain     # also pushes to the Hub (huggingface-cli login first)
python train.py --quack_ds you/mallard-calls --connectome my_duck.safetensors
```

Output folder (also the Hub repo layout):

```
connectome.safetensors     frozen wiring
brain_weights.safetensors  trained parameters
servos.npz                 servos [T,12] rad, joints [12], hz=20
quacks.wav                 16 kHz synthesized rollout audio
config.json  README.md     run config + model card
```

Memory is dominated by BPTT through `tbptt × 10` substeps over `N·k_in·batch` synaptic activations. On a 24 GB GPU, `--n_neurons 4096 --batch 32` fits at `--tbptt 10`; the default is sized for 8 GB.

## Deploy to the microduck

Stream `servos.npz["servos"]` row by row at 20 Hz to the servo bus with the column order in `servos.npz["joints"]`. For closed loop, run `DuckBrain` on the robot's compute with the same `world.observe()` encoders fed from real bill, camera, and microphone features, and send `info["ctrl"]` to the speaker through `Syrinx`.
