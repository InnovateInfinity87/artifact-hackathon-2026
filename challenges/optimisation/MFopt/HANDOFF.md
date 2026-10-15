# MFopt handoff to the Mac (15 Oct 2026, 14:50)

Branch: `claude/mfopt-harness` on the fork InnovateInfinity87/artifact-hackathon-2026 (pushed, includes fork master with VBC 131.7 mrad and the rematched optics). Code in `challenges/optimisation/MFopt/`; plan in `/mnt/project-files/mfopt/plan.md`. Commits as Francesco, no AI attribution.

## Done
- `optics_env.py`: `optics_for_target` starts from `optics_matching_vbc_on/results/rematched_grid_vbc_on.npz` (bilinear in log size) and re-matches FEA Q15/Q16, FEH Q1/Q2 in the first-order env twin (the second-order rematch is up to ~30 % off in the scored first-order env).
- `responses.py`: per node (12x12) -> `k1`, model `orbit` (BPM x, BPM y, screen x,y), `R` (those rows x 58 corrector planes), `J` (readout mu_x, mu_y, sigma_x, sigma_y, T x 17 DOF), `feasible`. DOF (Francesco): FEA Q14/15/16, FEH Q1/Q2; FEA hvcor 04/05/06 + FEH hvcor 01/02 both planes; FEA sext 01/02. `ResponseTable.at(sx, sy)`. Sextupoles barely act in the first-order env (~2 % in sigma_y at full k2).
- `steering.py`: `thread()` crops R to BPMs with a reading and correctors upstream of them, iterates, undoes a shot that loses ground and lowers rcond; stage A (`optics_and_steering`) reads the table by default (`table=False` tracks instead).
- `mfbo.py` (draft, untested): low fidelity = GP on 256 Sobol model points over 5 quads + 2 sextupoles; high = model GP + discrepancy (constant offset, GP from 3 points); FEH correctors cancel the predicted centroid each shot; EI on the benchmark mismatch from posterior samples; stops when EI < cost of one more step, ends on the best reading.

## In flight in the cloud (finishing there)
- Response table build: `python -m MFopt.responses --workers 4` -> `MFopt/data/responses_vbc_on.npz` (about 95/144 at 14:48; it only writes at the end). The cloud thread keeps it running and pushes the npz to this branch when it finishes (~15 min): `git pull` for it. To rebuild on the Mac instead: `python -m MFopt.responses --workers <cores>`.
- Threading test on lost-beam days (stopped in the cloud after 1 day; rerun on the Mac) (`misalignment_variation=5`, seeds 1000-1011; 6/12 resets lose the beam after BPM 9). Seed 1000: T 3 % -> 99 % in 3 shots, score 0.38, 8 steps.

## Next
1. Once the table is on the branch: commit, open a PR on the fork, merge, then upstream PR on amylizzle/artifact-hackathon-2026 (Francesco: merge on ours first, then PR on Amy's).
2. Test `mfbo.mfbo` vs `steering.optics_and_steering` on 30 default days (seeds 1000-1029) with `harness.run_episode`; add an `mfbo` entry to `run_baselines.POLICIES`. Then the lost-beam days.
3. Stage A at VBC 118.5 mrad scored mean 0.30, 70 % within 10 %, 4.4 steps; misses were spot size, which is what MF-BO should fix.

Run from `challenges/optimisation`; BoTorch needs `pip install botorch`.
