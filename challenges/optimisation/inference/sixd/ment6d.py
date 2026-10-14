"""Image-based 6D density reconstruction at the S02 start by sample reweighting (MENT).

The density at the start is written as rho(x) = g(x) prod_k h_k(P_k x), where g is a
Gaussian prior, P_k maps a start coordinate to its pixel on image k (xsuite tracking through
the whole line for the configuration of image k, plus the 20 um screen resolution), and h_k
is a per-pixel factor. This is the maximum-entropy density relative to g that reproduces
every image (MENT). With a sample drawn from g, rho is a set of per-particle weights
w_i = prod_k h_k(pixel_k(i)), so each configuration is tracked once per prior and the fit
needs no differentiable model. The h_k are found by multiplicative Gauss-Seidel sweeps over
the images.

Every image is normalised by the bunch charge (assumed known), and the part of the beam
outside the +-5 mm window is one extra bin, so clipped images are used as they are.

Prior: a Gaussian fitted to the image second moments with the model's linear maps,
inflated, then refined by refitting a Gaussian to the weighted sample and redrawing.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time

import common
import full6d
import numpy as np

cv = common.cv
xsuite_model = common.xsuite_model

IMAGES = common.HERE / "images.npz"
REBIN = 2  # 200 px of 50 um -> 100 px of 100 um
PIXELS = full6d.PIXELS // REBIN
PIXEL_M = 2 * full6d.FIELD_OF_VIEW_M / PIXELS
OUTSIDE = PIXELS * PIXELS  # index of the out-of-window bin
LABELS = common.LABELS

# set before forking the tracking workers
_LINE = None
_SAMPLE = None


# ---------------------------------------------------------------------------- data
def load_images(path=IMAGES):
    data = np.load(path)
    keys = [tuple(str(k).split("|")) for k in data["keys"]]
    images = data["images"].astype(float)
    n_particles = len(data["beam"])
    return keys, images, n_particles, data["beam"]


def measured_fractions(image: np.ndarray, n_particles: int) -> np.ndarray:
    """Rebinned pixel fractions plus the out-of-window fraction (sums to 1)."""
    coarse = image.reshape(PIXELS, REBIN, PIXELS, REBIN).sum((1, 3)).ravel()
    return np.append(coarse, max(n_particles - coarse.sum(), 0.0)) / n_particles


def image_moments(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Centroid-removed second moments (xx, yy, xy) of a full-resolution image, resolution
    removed, and the fraction of the image within 1 pixel of the edge (clipping flag)."""
    centres = (np.arange(full6d.PIXELS) + 0.5) * (2 * full6d.FIELD_OF_VIEW_M / full6d.PIXELS) - full6d.FIELD_OF_VIEW_M
    total = image.sum()
    py, px = image.sum(1) / total, image.sum(0) / total
    mx, my = px @ centres, py @ centres
    xx = px @ (centres - mx) ** 2 - full6d.SCREEN_RESOLUTION_M**2
    yy = py @ (centres - my) ** 2 - full6d.SCREEN_RESOLUTION_M**2
    xy = ((centres - my)[:, None] * (centres - mx)[None, :] * image).sum() / total
    edge = (image[[0, -1], :].sum() + image[:, [0, -1]].sum()) / total
    return np.array([xx, yy, xy]), edge


# ---------------------------------------------------------------------------- tracking
def pixel_indices(coords: np.ndarray, alive: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    blur = rng.normal(0, full6d.SCREEN_RESOLUTION_M, (len(coords), 2))
    ix = np.floor((coords[:, 0] + blur[:, 0] + full6d.FIELD_OF_VIEW_M) / PIXEL_M)
    iy = np.floor((coords[:, 2] + blur[:, 1] + full6d.FIELD_OF_VIEW_M) / PIXEL_M)
    inside = alive & (ix >= 0) & (ix < PIXELS) & (iy >= 0) & (iy < PIXELS)
    index = np.full(len(coords), OUTSIDE, dtype=np.int32)
    index[inside] = (iy[inside] * PIXELS + ix[inside]).astype(np.int32)
    return index


def track_config(args):
    """Track the global sample through one configuration; pixel index per screen."""
    config, seed = args
    line = _LINE
    full6d.apply_config(line, config)
    rng = np.random.default_rng(seed)
    order = sorted(config.screens, key=line.element_index.__getitem__)
    particles = xsuite_model._particles(_SAMPLE, line.converted.initial_energy_ev)
    out, start = {}, 0
    for screen in order:
        stop = line.element_index[screen] + 1
        line.line.track(particles, ele_start=start, ele_stop=stop)
        start = stop
        ids = particles.particle_id
        coords = np.zeros((len(_SAMPLE), 6))
        alive = np.zeros(len(_SAMPLE), bool)
        coords[ids] = _raw_coordinates(particles)
        alive[ids] = particles.state > 0
        out[screen] = pixel_indices(coords, alive, rng)
    line.restore_nominal_quads()
    full6d.set_tdc_voltage(line, 0.0)
    return config.name, out


def _raw_coordinates(particles) -> np.ndarray:
    coords = np.column_stack([np.asarray(getattr(particles, k)) for k in xsuite_model.COORDINATES])
    coords[:, 4] *= -1
    return coords


def track_all(line, sample, configs, seed=0, workers=4):
    global _LINE, _SAMPLE
    _LINE, _SAMPLE = line, sample
    jobs = [(c, seed + i) for i, c in enumerate(configs)]
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(track_config, jobs)
    else:
        results = [track_config(j) for j in jobs]
    return {(name, s): idx for name, out in results for s, idx in out.items()}


# ---------------------------------------------------------------------------- prior
def linear_gaussian_fit(line, configs, keys, images, max_edge=0.01):
    """Weighted least squares for the 21 covariance elements from the moments of images that
    are not clipped, with the model's linear maps; projected to positive definite."""
    import observability as obs

    by_name = {c.name: c for c in configs}
    rows, values, sig = [], [], []
    for (name, screen), image in zip(keys, images):
        moments, edge = image_moments(image)
        if edge > max_edge:
            continue
        config = by_name[name]
        full6d.apply_config(line, config)
        transfer = line.linear_map([screen])[screen]
        rows.append(obs.moment_rows(transfer))
        values.append(moments)
        sig.append([0.02 * moments[0], 0.02 * moments[1], 0.02 * np.sqrt(moments[0] * moments[1])])
    line.restore_nominal_quads()
    full6d.set_tdc_voltage(line, 0.0)
    a = np.vstack(rows)
    b = np.concatenate(values)
    s = np.concatenate(sig)
    a, b = a / s[:, None], b / s
    # column scaling for conditioning
    scale = np.linalg.norm(a, axis=0)
    scale[scale == 0] = 1
    solution, *_ = np.linalg.lstsq(a / scale, b, rcond=1e-10)
    sigma = common.unique_to_matrix(solution / scale)
    return nearest_positive(sigma), len(values)


def nearest_positive(sigma: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    d = np.sqrt(np.abs(np.diag(sigma)))
    corr = sigma / np.outer(d, d)
    values, vectors = np.linalg.eigh(corr)
    values = np.maximum(values, floor)
    corr = (vectors * values) @ vectors.T
    dd = np.sqrt(np.diag(corr))
    corr = corr / np.outer(dd, dd)
    return corr * np.outer(d, d)


def draw_gaussian(sigma, n, rng, mean=None):
    mean = np.zeros(6) if mean is None else mean
    return mean + rng.multivariate_normal(np.zeros(6), sigma, size=n, method="eigh")


# ---------------------------------------------------------------------------- MENT
def ment(indices: dict, data: dict, keys_used, sweeps=30, omega=0.6, eps=None, n_meas=1e5,
         log=None):
    """Gauss-Seidel MENT on per-particle weights. indices[key]: (N,) pixel index per image;
    data[key]: measured fractions (PIXELS^2 + 1). Returns normalised weights and history."""
    n = len(next(iter(indices.values())))
    eps = 0.5 / n_meas if eps is None else eps
    logw = np.zeros(n)
    history = []
    for sweep in range(sweeps):
        chi2, dof = 0.0, 0
        for key in keys_used:
            idx, meas = indices[key], data[key]
            w = np.exp(logw - logw.max())
            w /= w.sum()
            sim = np.bincount(idx, weights=w, minlength=OUTSIDE + 1)
            populated = sim > 0
            update = np.zeros_like(sim)
            update[populated] = omega * np.log((meas[populated] + eps) / (sim[populated] + eps))
            logw += update[idx]
            var = np.maximum(meas, sim) / n_meas + 1e-12
            used = (meas > 0) | (sim > 0)
            chi2 += float((((meas - sim) ** 2) / var)[used].sum())
            dof += int(used.sum())
        w = np.exp(logw - logw.max())
        w /= w.sum()
        ess = 1.0 / np.sum(w**2)
        history.append((sweep, chi2 / max(dof, 1), ess))
        if log:
            log(f"   sweep {sweep:2d} chi2/pixel {chi2 / max(dof, 1):.3f} ESS {ess:.0f}")
    return w, history


def predicted(indices, w, key):
    return np.bincount(indices[key], weights=w, minlength=OUTSIDE + 1)


# ---------------------------------------------------------------------------- summaries
def weighted_cov(sample, w):
    mean = w @ sample
    d = sample - mean
    return (d * w[:, None]).T @ d, mean


def summary(sample, w=None) -> dict:
    w = np.full(len(sample), 1 / len(sample)) if w is None else w
    s, mean = weighted_cov(sample, w)
    sd = np.sqrt(np.diag(s))
    corr = s / np.outer(sd, sd)

    def eps(i):
        return float(np.sqrt(np.linalg.det(s[i : i + 2, i : i + 2])))

    tau, delta = sample[:, 4] - mean[4], sample[:, 5] - mean[5]
    chirp = s[4, 5] / s[4, 4]
    resid = delta - chirp * tau
    # slice emittance x in the central slice |tau| < 0.25 sigma_tau
    core = np.abs(tau) < 0.25 * sd[4]
    wc = w * core
    wc = wc / wc.sum()
    sc, _ = weighted_cov(sample, wc)
    return {
        "eps_x": eps(0),
        "eps_y": eps(2),
        "eps_z": eps(4),
        "eps_4d_ratio": float(np.sqrt(np.linalg.det(s[:4, :4])) / (eps(0) * eps(2))),
        "sigma_x": float(sd[0]),
        "sigma_y": float(sd[2]),
        "sigma_z": float(sd[4]),
        "sigma_delta": float(sd[5]),
        "chirp": float(chirp),
        "slice_sigma_delta": float(np.sqrt(w @ resid**2)),
        "corr_tau_delta": float(corr[4, 5]),
        "corr_x_y": float(corr[0, 2]),
        "corr_x_py": float(corr[0, 3]),
        "corr_px_y": float(corr[1, 2]),
        "corr_x_delta": float(corr[0, 5]),
        "corr_y_tau": float(corr[2, 4]),
        "slice_eps_x": float(np.sqrt(np.linalg.det(sc[0:2, 0:2]))),
        "slice_eps_y": float(np.sqrt(np.linalg.det(sc[2:4, 2:4]))),
        "cov": s.tolist(),
    }


# ---------------------------------------------------------------------------- pipeline
def reconstruct(line, configs, keys, images, n_meas, keys_used=None, n_sample=300_000,
                inflation=(2.0, 1.4, 1.2), sweeps=(15, 20, 30), seed=0, log=print,
                workers=4):
    keys_used = keys if keys_used is None else keys_used
    data = {k: measured_fractions(img, n_meas) for k, img in zip(keys, images)}
    used_set = set(keys_used)
    used_images = [(k, img) for k, img in zip(keys, images) if k in used_set]
    names_used = {k[0] for k in keys_used}
    cfgs = [c for c in configs if c.name in names_used]
    sigma, n_fit = linear_gaussian_fit(
        line, cfgs, [k for k, _ in used_images], [img for _, img in used_images]
    )
    log(f"linear Gaussian fit from {n_fit} unclipped images: sd {np.sqrt(np.diag(sigma))}")
    rng = np.random.default_rng(seed)
    mean = np.zeros(6)
    stages = []
    for stage, (factor, n_sweeps) in enumerate(zip(inflation, sweeps)):
        prior = sigma * factor**2
        sample = draw_gaussian(prior, n_sample, rng, mean)
        t0 = time.time()
        indices = track_all(line, sample, cfgs, seed=seed + 100 * stage, workers=workers)
        log(f"stage {stage}: prior x{factor}, tracked {len(cfgs)} configs in {time.time() - t0:.0f} s")
        w, history = ment(indices, data, keys_used, sweeps=n_sweeps, n_meas=n_meas, log=log)
        sigma, mean = weighted_cov(sample, w)
        stages.append({"factor": factor, "history": history, "summary": summary(sample, w)})
    return {"sample": sample, "weights": w, "indices": indices, "data": data,
            "stages": stages, "keys_used": keys_used}


def main() -> None:
    line = full6d.build_line()
    configs = full6d.make_plan(line)
    keys, images, n_meas, beam = load_images()
    result = reconstruct(line, configs, keys, images, n_meas)
    truth = summary(beam)
    reco = summary(result["sample"], result["weights"])
    for k in truth:
        if k != "cov":
            print(f"{k:18s} true {truth[k]: .4e} reco {reco[k]: .4e}")
    np.savez_compressed(common.HERE / "ment_result.npz", sample=result["sample"], weights=result["weights"])


if __name__ == "__main__":
    sys.exit(main())
