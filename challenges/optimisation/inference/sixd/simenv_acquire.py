"""Run a measurement plan on Amy's SimEnv (RL_framework, xsuite backend) and store the
images in the images.npz layout, with each screen's own chip geometry.

The episode's machine is SimEnv's: its own beam (size, shape, offset), magnet set-point
errors, 50 um quad/sextupole misalignments, 30 um diagnostic offsets and shot jitter.
Only `env.step` is used to set magnets, pick the screen and read BPMs and images, with
two exceptions, both said where they happen: the linac phase (SimEnv has no RF knob; the
phase is written into its xsuite line the way full6d.set_rf_phases does for the model)
and the true beam, read for scoring only.
"""

from __future__ import annotations

import sys

import common
import full6d
import numpy as np

sys.path.insert(0, str(common.HERE.parents[1] / "RL_framework"))
sys.path.insert(0, str(common.HERE.parent / "steering"))

SHAPE = (1024, 1024)  # camera grid; the reconstruction rebins it by REBIN
REBIN = 4
L02 = "cla_l02_lin_cav_01"


def make_env(num_particles=20_000, seed=0, **kwargs):
    from rl_framework.config import RLConfig
    from rl_framework.envs.sim_env import SimEnv

    env = SimEnv(RLConfig(backend="xsuite", num_particles=num_particles, normalise_image=False,
                          screen_shape=SHAPE, max_episode_steps=10**9, **kwargs))
    env.reset(seed=seed)
    return env


def nominal_settings(env):
    """Every magnet at the strength the lattice file carries. In SimEnv a commanded
    setting is exact, so this removes the episode's set-point errors (survey errors and
    the beam stay)."""
    return dict(env._nominal_strengths)


def shot(env, magnets, screen):
    obs, *_ = env.step({"magnets": magnets, "screen": screen})
    return obs


def svd_steer(env, model, base, iterations=3, cutoff=1e-3, screen="cla_feh_dia_scr_01"):
    """One-shot SVD orbit correction (steering/svd_correction.py): corrector response
    from the design model, one BPM shot per iteration. Returns the corrector settings
    {knob: angle} on top of `base` and the kicks {corrector: (h, v)} for the model."""
    import svd_correction

    bpms = [b for b in env.bpm_names if b in model.line.element_names]
    correctors = [c for c in model.inventory.correctors
                  if c in model.line.element_names and f"{c}_h" in env.magnet_names]
    response = svd_correction.corrector_response(model, correctors, bpms)
    inverse, _, _ = svd_correction.svd_pseudo_inverse(response, cutoff)
    kicks = np.zeros(response.shape[1])
    settings = dict(base)
    index = [env.bpm_names.index(b) for b in bpms]
    for _ in range(iterations):
        obs = shot(env, settings, screen)
        valid = obs["bpm_valid"][index].astype(bool)
        orbit = np.ravel(np.column_stack([obs["bpm_x"][index], obs["bpm_y"][index]]) * valid[:, None])
        kicks = kicks - inverse @ orbit
        for i, c in enumerate(correctors):
            settings[f"{c}_h"] = base[f"{c}_h"] + kicks[2 * i]
            settings[f"{c}_v"] = base[f"{c}_v"] + kicks[2 * i + 1]
    applied = {c: (settings[f"{c}_h"], settings[f"{c}_v"]) for c in correctors}
    return settings, applied


def config_magnets(base, config):
    out = dict(base)
    out.update(config.k1)
    return out


def acquire(env, configs, keys, base, shots=10, out=common.HERE / "images_simenv.npz"):
    """Images for (config, screen) keys: `shots` steps each, summed. Also the mean BPM
    readings of every shot (valid ones), for orbit work."""
    by_name = {c.name: c for c in configs}
    images, bpm, valid = [], [], []
    for cname, screen in keys:
        config = by_name[cname]
        # SimEnv has no RF knob: the phase goes straight into its xsuite line
        full6d.set_rf_phases(env.transport._line, config.rf_phase_deg)
        magnets = config_magnets(base, config)
        img, xs, ys, vs = 0, [], [], []
        for _ in range(shots):
            obs = shot(env, magnets, screen)
            img = img + obs["screen_image"].astype(float)
            xs.append(obs["bpm_x"]); ys.append(obs["bpm_y"]); vs.append(obs["bpm_valid"])
        images.append(img)
        bpm.append(np.stack([np.mean(xs, 0), np.mean(ys, 0)], 1))
        valid.append(np.min(vs, 0))
    full6d.set_rf_phases(env.transport._line, {})
    screens = sorted({s for _, s in keys})
    geometry = [(*env.screen_extent(s), SHAPE[1] // REBIN, SHAPE[0] // REBIN, 0.0) for s in screens]
    beam = env._coordinates  # the episode's true beam: scoring only
    np.savez_compressed(out, keys=np.array([f"{c}|{s}" for c, s in keys]), images=np.array(images),
                        beam=beam, n_particles=len(beam) * shots, geometry_screens=np.array(screens),
                        geometry=np.array(geometry), bpm=np.array(bpm), bpm_valid=np.array(valid),
                        bpm_names=np.array(env.bpm_names))
    return out


def no_tdc_plan(line, phases=(-30.0, -23.0, -15.0, -8.0, 0.0), rescale=True):
    """Transverse block of the 21-image plan (S02-03, S03-01, S04-03 under nominal, s02a,
    s02b, s02c) and the FEA-01 spectrometer setting at several L02 phases, TDC off."""
    plan = {c.name: c for c in full6d.make_plan(line)}
    screens = ["cla_s02_dia_scr_03", "cla_s03_dia_scr_01", "cla_s04_dia_scr_03"]
    configs = [full6d.Config(n, plan[n].k1, 0.0, screens) for n in ("nominal_off", "s02a", "s02b", "s02c")]
    fea = plan["fea_01_off"]
    for p in phases:
        configs.append(full6d.Config(f"fea_01_ph{p:+.0f}", fea.k1, 0.0, ["cla_fea_dia_scr_01"], {L02: p}, rescale))
    keys = [(c.name, s) for c in configs for s in c.screens]
    return configs, keys
