"""Polynomial transfer maps from the S02 start to every (config, screen) of the plan.

The forward model of each measurement configuration is fixed, so tracking the reconstruction
sample through xsuite again at every stage is wasted work. Here every (config, screen) map
x(z0), y(z0) is fitted once as a polynomial of total order <= ORDER in the six start
coordinates, from one tracking of a training sample (plus an aperture-loss flag). After that
a "tracking" of N particles through all 68 screens is one matrix product,
monomials(N x terms) @ coefficients(terms x 136), about 1 s for 150k particles instead of
about 3 min on 4 cores.
"""

from __future__ import annotations

import itertools
import time

import common
import full6d
import ment6d
import numpy as np

ORDER = 4
# input scaling (also the training distribution: Gaussian with these rms, which covers the
# broad MENT prior)
SCALE = np.array(ment6d.BROAD_PRIOR_RMS)
OUT = common.HERE / "surrogate_maps.npz"


def exponents(order=ORDER, dim=6):
    return np.array([e for n in range(order + 1) for e in itertools.product(range(n + 1), repeat=dim)
                     if sum(e) == n], dtype=np.int8)


def monomials(z, exps, scale=SCALE):
    u = z / scale
    powers = [np.ones_like(u)]
    for _ in range(int(exps.max())):
        powers.append(powers[-1] * u)
    out = np.empty((len(z), len(exps)))
    for j, e in enumerate(exps):
        col = np.ones(len(z))
        for d in range(z.shape[1]):
            if e[d]:
                col = col * powers[e[d]][:, d]
        out[:, j] = col
    return out


def _track_coords(args):
    """Track the global sample through one configuration; (x, y, alive) per screen."""
    config = args
    line = ment6d._LINE
    sample = ment6d._SAMPLE
    full6d.apply_config(line, config)
    order = sorted(config.screens, key=line.element_index.__getitem__)
    particles = ment6d.xsuite_model._particles(sample, line.converted.initial_energy_ev)
    out, start = {}, 0
    for screen in order:
        stop = line.element_index[screen] + 1
        line.line.track(particles, ele_start=start, ele_stop=stop)
        start = stop
        ids = particles.particle_id
        xy = np.zeros((len(sample), 2))
        alive = np.zeros(len(sample), bool)
        xy[ids, 0], xy[ids, 1] = np.asarray(particles.x), np.asarray(particles.y)
        alive[ids] = particles.state > 0
        out[screen] = (xy, alive)
    line.restore_nominal_quads()
    full6d.set_tdc_voltage(line, 0.0)
    return config.name, out


def track_coords(line, sample, configs, workers=4):
    import multiprocessing as mp

    ment6d._LINE, ment6d._SAMPLE = line, sample
    with mp.get_context("fork").Pool(workers) as pool:
        results = pool.map(_track_coords, configs)
    return {(name, s): v for name, out in results for s, v in out.items()}


class Maps:
    """Fitted polynomial maps for a list of (config, screen) keys."""

    def __init__(self, keys, coef, exps, loss_radius):
        self.keys = [tuple(k) for k in keys]
        self.coef = coef  # terms x (2 * n_keys): x then y per key
        self.exps = exps
        self.loss_radius = loss_radius  # per key: start radius (in SCALE units) beyond which losses occur

    def coords(self, z):
        """{key: (xy, alive)} for start coordinates z (N x 6)."""
        xy = monomials(z, self.exps) @ self.coef
        r = np.sqrt(((z / SCALE) ** 2).sum(1))
        return {k: (xy[:, 2 * i: 2 * i + 2], r < self.loss_radius[i]) for i, k in enumerate(self.keys)}

    def indices(self, z, rng, keys=None):
        """Pixel index per key (as ment6d.track_all)."""
        keys = self.keys if keys is None else keys
        out = {}
        for k, (xy, alive) in self.coords(z).items():
            if k not in keys:
                continue
            c = np.zeros((len(z), 4))
            c[:, 0], c[:, 2] = xy[:, 0], xy[:, 1]
            out[k] = ment6d.pixel_indices(c, alive, rng)
        return out

    def save(self, path=OUT):
        np.savez_compressed(path, keys=np.array([f"{c}|{s}" for c, s in self.keys]), coef=self.coef,
                            exps=self.exps, loss_radius=self.loss_radius)

    @classmethod
    def load(cls, path=OUT):
        d = np.load(path)
        return cls([tuple(k.split("|")) for k in d["keys"]], d["coef"], d["exps"], d["loss_radius"])


def fit(line, configs, n_train=60_000, order=ORDER, seed=11, workers=4, log=print, cov=None,
        scale=1.0):
    """Fit on a Gaussian training sample: diagonal SCALE * scale, or covariance `cov` (e.g.
    1.5^2 times the current reconstruction's) when given; the maps are accurate where the
    training sample is, which matters on the dispersive FEA screens."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    if cov is None:
        z = rng.normal(0, 1, (n_train, 6)) * SCALE * scale
    else:
        z = rng.multivariate_normal(np.zeros(6), cov, n_train)
    tracked = track_coords(line, z, configs, workers)
    log(f"tracked {n_train} training particles in {time.time() - t0:.0f} s")
    exps = exponents(order)
    phi = monomials(z, exps)
    r = np.sqrt(((z / SCALE) ** 2).sum(1))
    keys = sorted(tracked)
    coef = np.zeros((len(exps), 2 * len(keys)))
    loss_radius = np.zeros(len(keys))
    for i, k in enumerate(keys):
        xy, alive = tracked[k]
        loss_radius[i] = r[~alive].min() if (~alive).any() else np.inf
        good = alive & (r < loss_radius[i])
        coef[:, 2 * i: 2 * i + 2] = np.linalg.lstsq(phi[good], xy[good], rcond=None)[0]
    log(f"fitted {len(keys)} maps, order {order}, {len(exps)} terms, {time.time() - t0:.0f} s")
    return Maps(keys, coef, exps, loss_radius), tracked, z


def validate(maps, line, configs, beam, log=print):
    """Compare surrogate and tracked coordinates on a test beam (inside-window particles)."""
    tracked = track_coords(line, beam, configs)
    pred = maps.coords(beam)
    rows = []
    for k in maps.keys:
        xy, alive = tracked[k]
        pxy, palive = pred[k]
        inside = alive & (np.abs(xy) < full6d.FIELD_OF_VIEW_M).all(1)
        err = (pxy - xy)[inside]
        rows.append(dict(key="|".join(k), inside=float(inside.mean()),
                         rms_err_um=float(np.sqrt((err**2).mean()) * 1e6) if inside.any() else 0.0,
                         p99_err_um=float(np.percentile(np.abs(err), 99) * 1e6) if inside.any() else 0.0,
                         alive_mismatch=float((alive != palive).mean())))
    worst = sorted(rows, key=lambda r: -r["rms_err_um"])[:8]
    for r in worst:
        log(r)
    return rows, tracked


class CombinedMaps:
    """Narrow, high-order maps inside their training domain (radius < `radius` in units of
    their training rms), broad maps outside it (early MENT stages sample the broad prior)."""

    def __init__(self, narrow, broad, narrow_scale, radius=4.0):
        self.narrow, self.broad = narrow, broad
        self.keys = narrow.keys
        self.narrow_scale, self.radius = narrow_scale, radius

    def indices(self, z, rng, keys=None):
        keys = self.keys if keys is None else keys
        inner = np.sqrt(((z / (SCALE * self.narrow_scale)) ** 2).sum(1)) < self.radius
        a = self.narrow.indices(z[inner], rng, keys)
        b = self.broad.indices(z[~inner], rng, keys) if (~inner).any() else {}
        out = {}
        for k in keys:
            idx = np.empty(len(z), dtype=np.int32)
            idx[inner] = a[k]
            if (~inner).any():
                idx[~inner] = b[k]
            out[k] = idx
        return out


NARROW = dict(order=5, scale=0.5, path=common.HERE / "surrogate_maps_narrow.npz")


def load_combined():
    return CombinedMaps(Maps.load(NARROW["path"]), Maps.load(OUT), NARROW["scale"])
