# 6D phase space reconstruction on CLARA: observability and demo

Scope: can the 6D second-moment matrix (21 numbers) at `cla_s07_sim_aper_01` (247.6 MeV, exit of L04) be reconstructed with the existing lattice, instruments and the xsuite model? Baseline measurement set, as requested: every YAG screen downstream of the reconstruction point (5 non-dispersive S07 screens, 4 dispersive FEA screens), each used as a separate destructive shot, with the TDC off / +V / -V. Quadrupole changes are added only where the screens alone fall short.

## Verdict

- Algebraically yes. With the screens alone at nominal optics and the TDC on, the 25 shots give a rank 21 operator, and 8 well chosen shots already reach rank 21. TDC off cannot do it: rank 15 (nothing involving tau). The TDC streak (vertical) and the horizontal dispersion of the FEA dogleg are both needed.
- Practically, the screens alone are not enough. At nominal optics <x delta> and <px delta> are unobservable (1 sigma of 140 and 74 in scaled units, condition number 8e5). One extra quad setting (S07 quad_01 and quad_02 scaled by 0.7, 2 magnets) or any S07 scan makes all 21 elements observable to better than 2 % of scale (condition ~1e2).
- What is recovered well: all rms sizes, sigma_z (+0.9 %), sigma_delta (+0.8 %), the chirp (-2 %), the x-y correlation, and the small x, y vs tau, delta correlations (errors of a few 1e-2 in correlation).
- What is not: emittances. The beam at S07 is extremely correlated (corr(x,px)=0.99994, corr(tau,delta)=0.99997), so the emittances are tiny differences of large second moments. With 1 % noise on each moment plus 3 % model error, the demo returns eps_x within -38 % and eps_y within a factor 4.2 (true 25.6 and 165 nm geometric). The Fisher estimate (pure 1 % noise, optics scan of 2 settings) gives about 4 % on eps_x and eps_y, so the demo is limited by model error, not by noise. The longitudinal emittance (slice energy spread) is out of reach for moments (Fisher relative error above 20, i.e. no information). That needs screen images with TDC plus dispersion (slice widths), not moments.
- Upstream (S02 to S06, 35.8 to 247 MeV) cannot give longitudinal information: linear rank 10 of 21 (the 4D transverse moments only), and chromatic effects give only a nominal sensitivity (1 sigma of 4 to 35 in scaled units on every tau and delta moment). Reason: no TDC, no dispersion (the VBC dipoles are at angle 0 in the model; the real chicane would add dispersion at `cla_vbc_dia_scr_01`, not modelled), so tau and delta never reach the transverse position at first order.
- Full 6D density (not just moments) is plausible only with images and a differentiable tracker (GPSR style); see the last section.

## Assumptions

- CLARA TDC parameters were not found in a source I could open. Assumed: S-band 2998.5 MHz, vertical deflection, up to 5 MV (the VELA TDC design value, a 9 cell S-band cavity, was quoted as 5 MV). It sits at s 36.6 m, 0.45 m long, 247.6 MeV.
- The lattice file has the TDC at voltage 0, so the xsuite conversion makes it a drift. `tdc.py` replaces it by drift(L/2), thin kick, drift(L/2).
- Instruments follow `clara_virtual.VirtualMachine.screen_moments`: 1 % noise on xx, yy, xy and 20 um resolution; no screen aperture or clipping is applied. With the nominal optics the beam is huge (sigma_y 26 mm at `cla_s07_dia_scr_02`), so a real YAG would clip. This is an open issue for the baseline set.
- True beam: the nominal starting beam (`initial_distribution.npz`) tracked through xsuite to `cla_s07_sim_aper_01` (`make_s07_beam.py`). Hidden errors: `HiddenErrors.random(seed=0)` (launch offset, quad misalignments). The demo uses the 20000 particle sample as the truth, for scoring only.
- Coordinates are the Cheetah ones (x, px, y, py, tau, delta), tau = -zeta.

## TDC model (task 1)

`tdc.py`: `VerticalDeflectingKick`, a custom xsuite element (C kernel), thin, with the Hamiltonian H = -a y sin(phi + k zeta):

    py    += a sin(phi + k zeta)
    delta += a k y cos(phi + k zeta)      (Panofsky-Wenzel, same Hamiltonian, so symplectic)

with a = eV/(p0 c), k = 2 pi f / c = 62.84 1/m. `install_tdc(converted, V, phi)` and `set_tdc(line, V, phi)`.

Checks (`verify_tdc.py`, results in `verify_tdc.json`):

- Kick at 5 MV, tau = 1 mm: py = -1.2681e-3 (expected -1.2681e-3), delta kick for y = 1 mm: 1.2664e-3 (expected 1.2664e-3).
- Streak slope at `cla_s07_dia_scr_02` from the TDC exit map: dy/dtau = -0.4493 per MV, identical to -R34 a k (R34 = 1.770 m). So 1 um on the screen is 2.2 um in tau, that is 7.4 fs per um at 1 MV and 1.5 fs per um at 5 MV.
- sigma_y vs V at `cla_s07_dia_scr_02` (tracked vs linear map): 26.33 (26.31) mm at 0 MV, 26.37 (26.36) mm at 5 MV. The streak is a 0.2 % effect on the vertical size here, because the nominal optics has sigma_y = 26 mm. The streak is invisible in the y size but is clearly seen in the xy moment on the dispersive screens.
- Phase: at phi = 0 the kick is a pure streak; phi = pi flips its sign (equivalent to -V), phi = pi/2 gives a constant deflection with no streak. In the linear operator only g = a k cos(phi) matters, so a phase scan is equivalent to a voltage scan (and sign flip).

## Observability (task 2)

Method (`observability.py`, `run_observability.py`): the 21 unknowns are the covariance elements at the reconstruction point, scaled by sqrt(sigma_ii sigma_jj). A screen with 6x6 map M (from xsuite, finite differences, TDC kick included) gives <xx>, <yy>, <xy> = sum_ij M_ui M_vj sigma_ij. Rows are weighted by the 1 % noise of the predicted moment. Maps are exact xsuite maps at delta = 0; chromatic effects are not in the rank (they matter in the demo).

Rank and condition number for several screen sets (nominal optics, 5 MV):

| Measurement set | Shots | Rank | Condition | Observable |
|---|---|---|---|---|
| 5 non-dispersive S07 screens, TDC off | 5 | 10 | inf | the 10 transverse 4D moments only |
| all 9 screens, TDC off | 9 | 15 | inf | 4D plus x,px,y,py with delta and <delta delta>; nothing with tau |
| 5 non-dispersive screens, TDC off/+/- | 13 | 15 | 4e24 | 4D plus <y tau>, <py tau>, <tau tau>, <x tau>, <px tau>; no delta moment |
| 4 FEA screens only, TDC off/+/- | 12 | 19 | 4e16 | cannot separate <x x>, <x px>, <px px> from <x delta>, <px delta> |
| all 9 screens, TDC off/+ only | 17 | 20 | 7e11 | all but one combination of <y tau>, <py tau>, <tau tau> (needs a second TDC setting: -V or V/2) |
| all 9 screens, TDC off/+/- (baseline) | 25 | 21 | 8e5 | all, but <x delta> and <px delta> have 1 sigma 140 and 74 |
| baseline plus 2 quad settings (scan) | 75 | 21 | ~1e2 to 1e3 | all, 1 sigma 0.1 to 0.7 % of scale |

Which physical channel gives which moment (this is what the table follows):

| Moments (21) | Measured through | Needs |
|---|---|---|
| xx, x px, px px, yy, y py, py py, xy, x py, px y, px py (10) | beam size and xy tilt on non-dispersive screens at different phase advance | screens only, nominal optics |
| <y tau>, <py tau>, <tau tau> | y size of the streaked beam: y_f contains R34 a k tau | TDC on, at least two nonzero TDC settings (+V and -V) |
| <x tau>, <px tau> | <x y_f> cross moment of the streaked beam | TDC on, non-dispersive screens are enough |
| <x delta>, <px delta>, <delta delta> | x size on dispersive screens (x_f contains D delta) | FEA screens and enough phase advance variation between them |
| <y delta>, <py delta> | <x_f y_f> on a dispersive screen: x_f has D delta, y_f has y, py | FEA screens |
| <tau delta> (chirp) | <x_f y_f>: x_f has D delta, streaked y_f has tau | TDC on and FEA screens together |

Why the baseline is ill conditioned for <x delta>, <px delta>: only the 4 FEA screens see delta, they sit within one dogleg cell, so D R11, D R12 and R11^2 are nearly collinear. Remedy, the minimal quad change: scale S07 quad_01 and quad_02 by 0.7 (currents 9.0 to 6.3 A and -2.9 to -2.0 A). That is one extra setting, 9 screens times TDC off/+/- = 25 more shots, and all 21 elements reach 1 sigma below 2 % of scale (condition 1e2). Alternatives: scaling q03 to q10 together by 0.5 to 1.5 also works (condition 3e2 to 8e3, worse), changing only q07 to q10 or q09 or q10 helps much less.

Minimal subset at nominal optics that reaches rank 21 (greedy, `minimal_set.py`): 8 shots, `cla_s07_dia_scr_04` (-5 MV), `cla_fea_dia_scr_02` (+5 MV), `cla_s07_dia_scr_01` (off), `cla_fea_dia_scr_03` (off), `cla_fea_dia_scr_01` (-5 MV and +5 MV), `cla_fea_dia_scr_04` (off), `cla_s07_dia_scr_04` (+5 MV). Condition 1e6, so this is only an existence statement.

Derived quantities (Fisher, 1 % noise, `optics_scan.py`, greedy choice among 80 random S07 quad settings, 25 shots per setting). Relative 1 sigma:

| Settings | eps_x | eps_y | eps_z | sigma_tau | sigma_delta | chirp |
|---|---|---|---|---|---|---|
| nominal only | 75 % | 74000 % | 24000 % | 1.7 % | 0.5 % | 5 % |
| + 1 setting | 52 % | 3.3 % | 5100 % | 0.2 % | 0.3 % | 0.9 % |
| + 2 settings | 3.7 % | 3.2 % | 3900 % | 0.1 % | 0.3 % | 0.7 % |
| + 6 settings | 0.6 % | 3.2 % | 2500 % | 0.1 % | 0.2 % | 0.5 % |

The longitudinal emittance (the width of the tau-delta line, 1.9e-7 against sigma_tau sigma_delta = 1.4e-5) cannot be reached with second moments of a 1 % measurement.

Figures: `figures/observability.png` (singular values; 1 sigma per element, screens only vs with 2 optics settings), `figures/optics_scan.png` (precision vs number of added settings).

Upstream (S02 to S06, `upstream.py`, 13 screens, 35.8 MeV start): linear rank 10 of 21. A sample based nonlinear Jacobian (chromatic focusing and RF defocusing included) gives 1 sigma of 4 to 35 scaled units on all tau and delta moments (unobservable in practice). Knobs that could help but are not in this analysis: the VBC chicane at a real angle (dispersion in the VBC and in S06), and RF phases of L02/L03/L04 (they shear the longitudinal plane by a known amount, a longitudinal tomography, but the delta axis is still only visible with a dispersive observation).

## Demo (task 3)

Code: `reconstruct.py`, `run_demo.py`. The machine is the virtual CLARA (true beam, hidden launch and quad errors, 1 % moment noise, 20 um resolution); the model is the error free xsuite line.

1. Linear weighted least squares with the model's 6x6 maps (generic prior scales, tiny ridge), then projection to a positive semidefinite matrix.
2. Nonlinear refinement: Sigma = D L L^T D (L lower triangular, 21 parameters), a fixed whitened Gaussian sample of 4000 particles tracked through the xsuite model for every (quad setting, TDC voltage), screen moments matched with a soft L1 loss. Weights use 1 % noise plus a 3 % model error allowance (the true beam is not Gaussian and the chromatic terms are large; with true Sigma the Gaussian prediction misses the machine by 1 to 10 %, `diag_model_error.py`; the linear maps miss by 10 to 85 % on some dispersive screens, which is why step 2 is needed).

Results (nonlinear refinement, 40 function evaluations, not fully converged):

| Quantity | True | Screens only (25 shots) | + q01,q02 x0.7 (50 shots) | + 2 optics settings (75 shots) | 75 shots, no hidden errors |
|---|---|---|---|---|---|
| eps_x geometric (nm) | 25.6 | 10.5 | 6.8 | 15.9 | 41.7 |
| eps_y geometric (nm) | 165 | 10800 | 2040 | 699 | 629 |
| sigma_z (mm) | 0.673 | 0.700 | 0.690 | 0.679 | 0.686 |
| sigma_delta (%) | 2.098 | 2.101 | 2.113 | 2.114 | 2.168 |
| chirp <tau delta>/<tau tau> (1/m) | 31.2 | 24.1 | 27.0 | 30.5 | 30.9 |
| corr(tau,delta) | 0.99994 | 0.80 | 0.88 | 0.98 | 0.98 |
| corr(x,y) | -0.173 | -0.179 | -0.174 | -0.177 | -0.178 |
| corr(x,tau) | 0.009 | 0.047 | 0.026 | 0.021 | 0.007 |
| corr(y,tau) | -0.001 | -0.039 | -0.022 | -0.008 | 0.006 |
| corr(x,delta) | 0.013 | 0.036 | 0.035 | 0.045 | 0.034 |
| corr(y,delta) | -0.001 | 0.021 | 0.005 | 0.001 | -0.024 |
| corr(px,delta) | 0.013 | 0.030 | 0.031 | 0.064 | 0.040 |

The linear step alone, for the 75 shot set, already gives sigma_z 0.709, sigma_delta 2.084 %, chirp 29.4. Reading the table: rms sizes, chirp and all correlations between planes are recovered, with errors of a few 1e-2 in correlation coefficient; the emittances are not (eps_y wrong by a factor 4 to 65). More optics settings improve both steps monotonically here; hidden errors are not the limiting factor (the no-error column is no better). Figure `figures/demo_ratio.png`.

Limits:

- Second moments are not the density: a Gaussian is assumed in step 2, and the beam has non-Gaussian tails and a nearly singular covariance.
- Centroids were not fitted (second moments about the mean only).
- Screens clip at a few mm in reality; the nominal optics gives 26 mm. Measuring the baseline set on real screens needs a matched optics, or the quad settings above.
- The reconstruction point is after L04, so the starting S02 beam is not inferred; going back to S02 needs the longitudinal tracking through the linacs (Cheetah achromatic cavities differ from xsuite by 45 to 55 % in chromatic emittance growth, per the model README).

## Density reconstruction (GPSR style)

Needed on top of the moments:

- Images, not moments: 2D screen histograms (x, y) for each setting; with the TDC on a dispersive screen gives the (tau, delta) projection directly, from which the slice energy spread (the missing longitudinal emittance) is read. The virtual machine has `screen_image`.
- A differentiable beam model: Cheetah is differentiable, but its cavities are achromatic and its drifts linear (per earlier findings), and it has no TDC with finite voltage in the CLARA JSON (voltage 0, frequency 0), so a TDC element (thin kick plus Panofsky-Wenzel, as in `tdc.py`) has to be written in Cheetah. xsuite is exact but not differentiable, so use it as the forward model for validation, or fit a surrogate/ finite difference scheme (the 21 parameter moment fit above is such a case).
- A generative beam parametrisation (neural network or Gaussian mixture on 6D) and a loss on image pixels, as in the AWA work (below).

## Literature

- R. Roussel et al., Efficient 6-dimensional phase space reconstruction from experimental measurements using generative machine learning, [arXiv:2404.10853](https://arxiv.org/pdf/2404.10853) (AWA: quads, a TDC and a dipole; as few as 20 images, differentiable simulation, generative model). I could confirm the arXiv and the conference versions (IPAC'24 TUPS73, LINAC 2024 THPB065), not a PRL record for the 6D paper; the PRL is the 2023 GPSR paper.
- S. Jaster-Merz et al., 5D tomographic phase-space reconstruction of particle bunches, [arXiv:2305.03538](https://arxiv.org/pdf/2305.03538) (Phys. Rev. Accel. Beams 27, 072801, 2024): quadrupole tomography plus the streaking angle of a polarizable X-band TDS (DESY ARES/SINBAD); experimental demonstration: [arXiv:2505.13724](https://arxiv.org/pdf/2505.13724). I did not check a FLASH specific reference.
- Hock and Wolski, 4D transverse tomography (K. Hock, A. Wolski, NIM A 726, 2013): cited from memory, not looked up.
- CLARA TDC: the layout (TDC for 6D characterisation after the bunch compressor, with a spectrometer line) is described in the [FEBE specification](https://arxiv.org/pdf/2309.13125); voltage and frequency were not found.

## Files

- `tdc.py`, `verify_tdc.py`, `verify_tdc.json`: TDC element and its checks.
- `common.py`: S07 segment builder with the TDC (`make_pair`), index helpers.
- `make_s07_beam.py`, `s07_beam.npz`: true beam at `cla_s07_sim_aper_01`.
- `observability.py`, `run_observability.py`, `minimal_set.py`, `optics_scan.py` (`optics_scan.json`), `upstream.py`: observability studies.
- `reconstruct.py`, `run_demo.py`, `diag_model_error.py`, `demo_results_*.pkl`: demo.
- `make_figures.py`, `figures/`.
- Run with `python`; the first run of anything with the TDC compiles a kernel (a few seconds). `run_demo.py <baseline|q01q02|scan> [--noerr]` takes 4 to 11 minutes.
