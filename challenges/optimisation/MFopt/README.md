# MFopt

Getting through a `SimEnv` episode (a new day on CLARA) to the requested spot at
`cla_feh_dia_scr_01` in at most 50 steps: steering first, then multi-fidelity
Bayesian optimisation between the model and the machine. The plan is in the
project notes (`mfopt/plan.md`).

## Target optics: `optics_env.py`

```python
from MFopt.optics_env import OpticsEnv

env = OpticsEnv()                        # a SimEnv; episodes are unchanged
optics = env.optics_for_target(150e-6, 250e-6)
k1, quad_sv, corrector_sv = optics.vectors

optics.quadrupoles.ranked(4)             # quadrupoles by their weight in the modes
optics.correctors.ranked(6)              # corrector planes, `_h` and `_v`
env.step(env.optics_action(optics))      # load the optics (one step)
```

- **Optics.** Starts from the matched family in `optics_matching/` (`MatchedOptics`),
  whose 42-quadrupole prefix carries the beam through the apertures. That family was
  matched in xsuite with the VBC and L4H off, so the four final-focus quadrupoles
  (FEA Q15/Q16, FEH Q1/Q2) are re-matched to the request in an error-free copy of the
  environment (VBC on, L4H on, apertures on, Cheetah), within the magnet limits.
  `optics.achieved`, `optics.transmission` and `optics.clipped` say how well that went.
- **Quadrupole modes.** SVD of d ln(rms size) / d k1 at every screen and the readout,
  for all 46 quadrupoles. Rows and weights follow `SizeModes` (the readout carries
  half of the squared weight), but it is the beam-size response in this environment's
  model, so dispersion and the VBC are included.
- **Corrector modes.** The same for the centroid against all 58 corrector planes.
- **Scores.** An SVD has one singular value per mode, so each magnet also gets
  `scores[name] = sqrt(sum_i s_i^2 V_ij^2)` over the modes holding 99 % of the squared
  response. `ranked()` sorts by it.

Everything runs on the error-free model with the nominal beam: no environment steps.
One target takes about a minute (re-match plus 208 finite-difference tracks), and
results are cached per target.
