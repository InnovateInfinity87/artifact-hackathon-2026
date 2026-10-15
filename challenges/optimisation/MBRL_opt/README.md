# MBRL_opt — model-based RL tuning of CLARA

A tuning agent for the `RL_framework` `SimEnv`. It does what an operator does: steers
the beam down the line on the BPMs with the correctors, then shapes and places the spot
on the FEBE hutch screen with the final quadrupoles, the sextupoles, the RF cavity
phases, the VBC chicane and the last correctors.

The "model" in model-based RL here is **a differentiable Cheetah twin of CLARA**, not a
learned network. The lattice is known — `RL_framework/FEBE_full.json`, which is the same
file SimEnv tracks — so nothing about the machine's *structure* has to be learned. What
is unknown is the particular machine `reset()` dealt: how the beam was launched, how big
it is, and where the magnets and diagnostics actually sit. Those are fitted from the
shots the agent takes, by gradient descent straight through Cheetah.

```
SimEnv  ── bpm_x, bpm_y, screen moments ─────────────────┐
  ▲                                                      ▼
  │                                   ┌──────────────────────────────────┐
  │                                   │ system_id   fit the latent       │
  │                                   │   beam offset, size, mismatch    │
  │                                   │   (survey error, optional)       │
  │                                   └──────────────┬───────────────────┘
  │                                                  ▼
  │                                   ┌──────────────────────────────────┐
  │                                   │ twin  differentiable Cheetah     │
  │                                   │   373 elements, batched, ~2 ms   │
  │                                   └──────────────┬───────────────────┘
  │                                                  ▼
  │                                   ┌──────────────────────────────────┐
  └── magnet settings ────────────────│ planner  SVD steer, CEM + Adam   │
                                      │   anchored on the last reading   │
                                      └──────────────────────────────────┘
```

## Why a physics twin rather than a learned one

Three things fall out of it.

**It is exact.** With every variation at zero the twin reproduces SimEnv's BPM orbit and
screen moments to 0.0002 µm — float32 rounding. `smoke_test.py` asserts this, and it is
the test the whole approach rests on.

**It is fast, and batched.** Cheetah fuses linear transfer maps, so a full track of the
373-element stretch costs a couple of milliseconds whether it carries 2,000 particles or
10,000, and element parameters take a leading batch dimension — 128 candidate tunes
evaluate in ~190 ms. CEM over a large population is affordable every single shot.

**It is differentiable.** `d sigma_x / d k1` at the readout comes back from
`backward()`, and so does the derivative with respect to a misalignment nobody measured.
That is what makes both the system-identification fit and the final plan polish possible.

The cost is one real shot per MPC step and no training at all. A policy-gradient method
would need many thousands of episodes to learn physics that is already written down.

## Running it

```bash
cd challenges/optimisation
# one-off: the venv needs gymnasium, stable-baselines3, scikit-image, the xsuite
# stack, and cheetah 0.8.4 (0.8.1 cannot parse the lattice JSON at all)
uv pip install --python .venv/bin/python \
  -r RL_framework/requirements.txt -r nominal_model/requirements.txt

.venv/bin/python -I MBRL_opt/smoke_test.py                 # ~95 s, 20 checks
.venv/bin/python -I MBRL_opt/evaluate.py --episodes 20
.venv/bin/python -I MBRL_opt/evaluate.py --episodes 20 --fit-misalignments
```

`evaluate.py` runs every method over the same seeds — the seed fixes the beam, the
survey error and the request — so a difference between two rows is a difference between
two methods, not two machines. The headline is the mean of `info["benchmark_score"]`,
which is `score.Report.calc_score` for one episode: the mismatch in requested spot sizes
plus `EVAL_WEIGHT` per doubling of `EVAL_BUDGET` shots. Lower is better.

## The files

| File | What it is |
|---|---|
| `twin.py` | `CheetahTwin` and `LatentState`. The lattice, the beam, and a `readings()` that reproduces `SimEnv._track` and `_expose` exactly. The core of the whole thing. |
| `system_id.py` | Fits `LatentState` to the episode's shots. The whole history goes through in one batched track. |
| `planner.py` | Truncated-SVD orbit correction for the steering, CEM plus Adam for the focus. |
| `agent.py` | `MBRLAgent`: the two stages, the shot budget, and the MPC loop. |
| `knobs.py` | Which numbers a stage may turn, and the translation to and from SimEnv's names and ranges. |
| `config.py` | `MBRLConfig`: the knob groups, budgets and solver settings. |
| `baselines.py` | do-nothing, SVD-only steering, random search, on the same budget. |
| `evaluate.py` | Runs everything over a set of episodes and prints the table. |
| `smoke_test.py` | Twin fidelity, batching, gradients, and two real episodes. |

## What it turns

A deliberately small proof-of-concept set, 31 knobs of the 111 SimEnv offers. Every
group is a list in `config.py`; widening is a one-line change.

**Steering, 16 knobs** — both planes of eight correctors spread along the line
(`cla_s02_mag_hvcor_01`/`_03`, `cla_s04_mag_hvcor_02`/`_05`, `cla_s07_mag_hvcor_03`/`_06`,
`cla_fea_mag_hvcor_03`, `cla_feh_mag_hvcor_01`), against all 31 BPMs.

**Focus, 17 knobs** — `cla_fea_mag_quad_13..16` and `cla_feh_mag_quad_01/02` (k1), both
FEA sextupoles (k2), all four powered cavity phases, `cla_vbc_chicane`, and
`cla_feh_mag_hvcor_01/02` both planes.

The two correctors shared between the stages are one knob, not two.

## Three things worth knowing

**Anchor the planner on the measurement.** The planner never asks the twin where the
beam is, only what *changes* if a knob moves:

```
predicted(delta) = measured + [twin(knobs + delta) − twin(knobs)]
```

A model wrong by a constant still predicts the effect of moving a quadrupole perfectly,
and anchoring cancels that constant exactly.

**Keep the trust region small.** That anchoring is exact for a small move and drifts for
a large one. Measured over one episode: a trust region of 0.25 of each knob's span
leaves the tune stuck at a mismatch of 13 and books a failed 10.2, while 0.08 walks it
down to 4.1 and books 5.5. This is the single most important number in `config.py`.

**Do not clip the planner's cost at the benchmark's scale.** The benchmark stops counting
at ten requested spot sizes. An untuned machine sits at seventy or more — a 1.5 mm spot
against an 85 µm request is eighteen spot sizes in size alone. A cost clipped at ten is
perfectly flat across the entire region where the tuning happens, and both CEM and
gradient descent see a plateau. `SimEnv._compute_reward` makes the same point about its
own `MAX_MISMATCH`.

A fourth, smaller one: whether a particle is inside the beam pipe is a step function, and
half the beam is scraped in the FEA arc at the nominal tune, so the objective is a
staircase with no slope. Blurring the wall outright is badly biased — at the true latent
it moved the predicted loss from 0.00 to 114.7, and the fit walked away from the right
answer chasing it. `twin._aperture` instead takes the *value* from the hard edge and the
*gradient* from a sigmoid, which costs nothing and keeps both honest.

## Results

Measured on this machine, CPU only, 10 episodes, seeds 1234–1243, at SimEnv's default
difficulty (`beam_variation` 1.0, `misalignment_variation` 1.0, shot jitter on). Score is
the mean `benchmark_score`; lower is better.

<!-- RESULTS -->

## Limits, and what to try next

**Reachability, not optimisation, is the binding constraint.** Given a twin that knows
the machine exactly, the planner drives the mismatch from ~70 down to 1.3–6.4 and its
prediction matches the delivered beam to within 1 % — so the planner is doing its job and
17 focus knobs simply cannot hit every request. Matching `sigma_x` and `sigma_y`
simultaneously is the hard part. **Widening `focus_quads` to more of the FEA arc is the
first thing to try**, and it is one line in `config.py`.

**Turn on the misalignment fit.** It is off by default so the aligned twin is the
baseline, but on one episode it took the score from 5.5 to 3.1: a survey error does
things no launch offset can imitate, since a misaligned quadrupole kicks and a misaligned
sextupole focuses. `--fit-misalignments`.

**Steering is nearly free and nearly useless here.** The nominal optics already deliver
beam to the hutch with every BPM reading, so there is no beam to recover, and eight
correctors cannot flatten an orbit driven by 46 misaligned quadrupoles read by BPMs with
30 µm centring errors of their own — it bottoms out around 230 µm. The stage earns its
place on a machine that has lost the beam, which is `score.py`'s setup rather than
SimEnv's; `steering_patience` stops it wasting shots in the meantime.

**The noise floor is real.** Every step redraws the beam at `beam_jitter` of
`beam_variation`, which is about 8 % rms on a spot size and a few microns on a centroid.
No amount of planning gets below it.

**Not yet built: the Dyna layer.** The twin is a free simulator, so the natural next step
is to randomise the latent, roll out inside the twin, and train a policy that proposes
the first move — turning the ~37-shot MPC tune into a handful of shots plus a refinement.
The MPC agent here is the thing that layer would warm-start, and it works standalone.
