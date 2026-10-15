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

- **Optics.** Starts from the 144 nodes rematched with the VBC on at 131.7 mrad
  (`optics_matching_vbc_on/`), interpolated in log size. Those were fitted in
  second-order Cheetah, and in this environment's first-order model their spot is off
  by up to about 30 %, so the four final-focus quadrupoles (FEA Q15/Q16, FEH Q1/Q2)
  are re-matched to the request in an error-free copy of the environment (VBC on,
  L4H on, apertures on), within the magnet limits.
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
One target takes 1.5 to 2.5 minutes (re-match plus 208 finite-difference tracks), and
results are cached per target.

## Steering responses and MF-BO knobs per spot: `responses.py`

```python
from MFopt.responses import DOF, ResponseTable

table = ResponseTable()                  # MFopt/data/responses_vbc_on.npz
at = table.at(150e-6, 250e-6)            # bilinear in log size between nodes
at["R"]      # centroid response: BPM x, BPM y, screen x, y  x  all 58 corrector planes
at["orbit"]  # the model's centroids at the same rows: what steering aims for
at["J"]      # readout (mu_x, mu_y, sigma_x, sigma_y, T)  x  DOF
```

- **MF-BO knobs (`DOF`).** The five quadrupoles closest to the IP (FEA Q14, Q15,
  Q16, FEH Q1, Q2), the five correctors per plane closest to it (FEA HVCOR 04, 05,
  06, FEH HVCOR 01, 02) and the two FEA sextupoles.
- **Table.** Built at every node of the VBC-on grid after the final-focus re-match,
  in the error-free first-order model: `k1`, `readout`, `clipped`, `feasible`
  (T >= 95 % and both sizes within 5 %), `relative_error`.
- **Twin.** The error-free twin tracks 5,000 particles (`OpticsEnv(twin_particles=...)`),
  half of the environment's 10,000, which keeps the rms sizes to about 1 %.
- **Rebuild.** `python -m MFopt.responses --workers 16` from `challenges/optimisation`,
  about 3 minutes on 16 cores. 118 of the 144 nodes are feasible, including every
  node in the benchmark's 80 to 400 um range except 0.1 x 0.1 mm.

Stage A steering (`steering.py`) reads the orbit and spot responses from this table
instead of tracking them per episode.

## Running episodes: `run_baselines.py`

```bash
python -m MFopt.run_baselines --seeds 30 --workers 16
python -m MFopt.run_baselines --misalignment 5 --seeds 12 --tag lost_beam --policies optics_and_steering mfbo
```

One process per episode. Policies: `do_nothing`, `optics_only`, `optics_and_steering`
(stage A) and `mfbo` (stage A, then MF-BO in `mfbo.py`). `--misalignment 5` gives
days on which the beam is lost part way along the line; stage A then threads it
(`steering.thread`). Results go to `MFopt/results/<policy>[_<tag>].json`. MF-BO needs
`botorch`.

## Results (15 Oct, VBC 131.7 mrad, 5k-particle twin table)

| Days | Policy | Mean score | Median | Within 10 % | Within 25 % | Steps |
|---|---|---|---|---|---|---|
| 30 default (seeds 1000-1029) | do nothing | 9.37 | 10.00 | 0 % | 0 % | 0 |
| | optics only | 6.50 | 6.22 | 0 % | 0 % | 1 |
| | stage A | 0.49 | 0.38 | 30 % | 73 % | 4.6 |
| | stage A + MF-BO | 0.32 | 0.25 | 77 % | 97 % | 10.2 |
| 12 lost-beam (misalignment x5, seeds 1000-1011) | stage A | 0.48 | 0.43 | 33 % | 83 % | 7.2 |
| | stage A + MF-BO | 0.37 | 0.30 | 75 % | 92 % | 13.2 |

Every episode ends within 50 steps (longest 32) with the beam on the screen.
