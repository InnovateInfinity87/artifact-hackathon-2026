"""Image acquisition with shot-to-shot launch jitter.

Each image is one screen insertion: `shots` bunches at 10 Hz, each with its own launch
jitter on top of the fixed hidden launch error, summed into one image. The bunch is
`particles / shots` macroparticles per shot, so the counting statistics of the summed image
match the jitter-free images. Two images are kept per insertion: the plain sum, and the sum
after shifting every shot onto the mean centroid (the centroid is computed from the shot's own
image, as it would be online). Each screen is its own insertion, so jitter is independent
between screens.
"""

from __future__ import annotations

import multiprocessing as mp

import common
import full6d
import numpy as np

cv = common.cv
# project convention (steering/README.md): 20 um, 2 urad, 20 um, 2 urad, 0.02 %; plus an
# arrival-time jitter of 100 fs (30 um in tau), which matters for the TDC images
JITTER = np.array([20e-6, 2e-6, 20e-6, 2e-6, 30e-6, 2e-4])

_STATE = {}


def _centroid(image):
    edges = np.linspace(-full6d.FIELD_OF_VIEW_M, full6d.FIELD_OF_VIEW_M, full6d.PIXELS + 1)
    c = 0.5 * (edges[1:] + edges[:-1])
    s = image.sum()
    return np.array([image.sum(0) @ c, image.sum(1) @ c]) / max(s, 1)


def _insertion(args):
    config, screen, seed = args
    line, beam, launch, jitter, shots = (_STATE[k] for k in ("line", "beam", "launch", "jitter", "shots"))
    rng = np.random.default_rng(seed)
    full6d.apply_config(line, config)
    per_shot = np.array_split(rng.permutation(len(beam)), shots)
    coords, cents = [], []
    for idx in per_shot:
        shot = beam[idx] + launch + rng.normal(0, 1, 6) * jitter
        c = line.track_through(shot, [screen], allow_losses=True)[screen]
        img = full6d.blur_and_histogram(c, rng)
        coords.append(c)
        cents.append(_centroid(img))
    line.restore_nominal_quads()
    full6d.set_tdc_voltage(line, 0.0)
    plain = sum(full6d.blur_and_histogram(c, rng) for c in coords)
    mean = np.mean(cents, axis=0)
    aligned = 0
    for c, ce in zip(coords, cents):
        shifted = c.copy()
        shifted[:, 0] += mean[0] - ce[0]
        shifted[:, 2] += mean[1] - ce[1]
        aligned = aligned + full6d.blur_and_histogram(shifted, rng)
    return (config.name, screen), plain, aligned, np.array(cents)


def acquire(line, beam, configs, keys, jitter=JITTER, shots=10, errors=None, seed=7, workers=4,
            out_prefix="images_jitter"):
    """Images for the (config, screen) `keys`; writes <prefix>_plain.npz and _aligned.npz."""
    errors = errors or cv.HiddenErrors()
    for q, (dx, dy) in errors.quad_offsets.items():
        line.line[q].shift_x, line.line[q].shift_y = dx, dy
    _STATE.update(line=line, beam=beam, launch=errors.launch, jitter=np.asarray(jitter), shots=shots)
    by_name = {c.name: c for c in configs}
    jobs = [(by_name[c], s, seed + i) for i, (c, s) in enumerate(keys)]
    try:
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(_insertion, jobs)
    finally:
        for q in errors.quad_offsets:
            line.line[q].shift_x, line.line[q].shift_y = 0.0, 0.0
    names = np.array([f"{c}|{s}" for (c, s), *_ in results])
    order = np.argsort(names)
    for label, col in (("plain", 1), ("aligned", 2)):
        np.savez_compressed(common.HERE / f"{out_prefix}_{label}.npz", keys=names[order],
                            images=np.array([results[i][col] for i in order]), beam=beam,
                            shots=len(results) * shots, jitter=np.asarray(jitter))
    return {names[i]: results[i][3] for i in order}
