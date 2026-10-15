"""The SimEnv with the target optics of the SVD analysis attached.

`OpticsEnv.optics_for_target(sigma_x, sigma_y)` answers, for a requested rms spot
at the readout screen, three questions an optimiser needs before it touches the
machine:

1. **Which optics?** The quadrupole k1s that give that spot. They start from the
   matched family in `optics_matching/` (`MatchedOptics`), whose 42 upstream
   strengths are a retuned prefix that carries the beam through the apertures. The
   family was matched in xsuite with the VBC off and L4H off, so in this
   environment's model (VBC on, L4H on, Cheetah) its four final-focus strengths
   miss the requested spot by up to a factor of six. They are re-matched here, in
   an error-free copy of the environment, within the environment's magnet limits.
2. **Which quadrupoles matter?** The SVD of the weighted response of ln(beam size)
   at every screen and the readout to every quadrupole, at that optics. Rows and
   weights follow `SizeModes` (the readout carries half of the squared weight), but
   the response is of the beam size in this environment's model rather than of the
   beta function in the VBC-off xsuite family, so dispersion and the VBC are in it.
3. **Which correctors matter?** The same SVD for the beam centroid at every
   screen and the readout against every corrector plane.

Each SVD gives one singular value per mode. Ranking magnets needs one number per
magnet, so each also carries `scores`: a magnet's weight in the modes that make up
`fraction` of the squared response, sqrt(sum_i s_i^2 V_ij^2).

Everything is computed on the error-free model with the nominal beam, so it costs
no steps of the environment, and results are cached per target.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

CHALLENGE_DIR = Path(__file__).resolve().parents[1]
for path in (CHALLENGE_DIR, CHALLENGE_DIR / "RL_framework"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from optics_matching import MatchedOptics, SizeModes  # noqa: E402
from rl_framework.config import RLConfig  # noqa: E402
from rl_framework.envs.sim_env import BPM_TAG, SCREEN_TAG, SimEnv  # noqa: E402

# The magnets the matched family varies with the requested spot; the other 42 are
# a fixed prefix.
FINAL_FOCUS = ("cla_fea_mag_quad_15", "cla_fea_mag_quad_16",
               "cla_feh_mag_quad_01", "cla_feh_mag_quad_02")
# The readout carries half of the squared weight, as in `SizeModes`.
READOUT_WEIGHT = 0.5
# Finite-difference steps: k1 in m^-2, corrector kicks in rad.
QUAD_STEP = 0.05
CORRECTOR_STEP = 2e-5


@dataclass
class Modes:
    """An SVD of a weighted response, and what it says about each magnet."""

    names: list[str]  # the magnets, in column order
    rows: list[str]  # what each row of the response is
    response: np.ndarray  # weighted, rows x magnets
    singular_values: np.ndarray
    Vt: np.ndarray  # rows are directions in magnet space
    scores: dict[str, float]  # each magnet's weight in the leading modes
    n_modes: int  # how many modes `scores` is taken over

    def ranked(self, count: int | None = None) -> list[str]:
        """The magnets with the largest scores, best first."""
        order = sorted(self.names, key=lambda name: -self.scores[name])
        return order[:count]


@dataclass
class TargetOptics:
    """What `optics_for_target` returns."""

    target: tuple[float, float]  # requested (sigma_x, sigma_y), metres
    k1: dict[str, float]  # all 46 quadrupoles
    quadrupoles: Modes | None  # None when asked for without modes
    correctors: Modes | None  # both planes, `<corrector>_h` and `<corrector>_v`
    achieved: tuple[float, float]  # (sigma_x, sigma_y) in the model, metres
    transmission: float  # in the model, to the readout
    clipped: list[str] = field(default_factory=list)  # pinned at a limit
    # `SizeModes().evaluate` at the same request (clamped to its 0.1 to 20 mm
    # domain): the SVD analysis's own quadrupole modes, for the VBC-off family.
    size_modes: dict | None = None

    @property
    def vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The three vectors: k1 in quadrupole order, the quadrupole singular
        values and the corrector singular values."""
        return (np.array(list(self.k1.values())),
                self.quadrupoles.singular_values,
                self.correctors.singular_values)

    @property
    def relative_error(self) -> float:
        """The worse of the two planes, |achieved / requested - 1|."""
        return float(max(abs(a / t - 1) for a, t in zip(self.achieved, self.target)))


class OpticsEnv(SimEnv):
    """`SimEnv` with `optics_for_target`. Episodes behave exactly as in `SimEnv`."""

    def __init__(self, config: RLConfig | None = None):
        super().__init__(config or RLConfig())
        self._matched = MatchedOptics()
        self._size_modes = SizeModes()
        self._twin: SimEnv | None = None
        self._optics_cache: dict[tuple[float, float, float], TargetOptics] = {}

    # ---------- the error-free model ----------

    @property
    def twin(self) -> SimEnv:
        """An error-free copy of this environment: the nominal beam, no survey
        error and no shot jitter, with the same lattice, backend and apertures."""
        if self._twin is None:
            config = RLConfig(**{**self.config.__dict__,
                                 "beam_variation": 0.0,
                                 "misalignment_variation": 0.0,
                                 "power_supply_ripple": 0.0,
                                 "bpm_noise": 0.0})
            self._twin = SimEnv(config)
            self._twin.reset(seed=0)
        return self._twin

    def _probe(self, settings: dict[str, float]) -> dict[str, np.ndarray]:
        """Track the nominal beam through the twin with these magnets changed
        from the lattice file, and read every screen and BPM on the way.

        Returns the centroid and rms size at every diagnostic in tracking order,
        with NaN where too little beam arrives, and the transmission to the end.
        """
        twin = self.twin
        twin._restore_nominal()
        for name, value in settings.items():
            twin._knobs[name].set(value, clip=False)
        coordinates = twin._coordinates
        count = len(coordinates)
        radius = twin.config.beam_pipe_radius
        minimum = twin.config.min_screen_fraction * count
        live = np.ones(count, dtype=bool)
        names, stats = [], []
        for name, x, y, alive in twin.transport.run(
            coordinates, [stop.name for stop in twin._stops]
        ):
            live &= alive
            if radius is not None:
                live &= x**2 + y**2 <= radius**2
            if BPM_TAG not in name and SCREEN_TAG not in name:
                continue
            names.append(name)
            if live.sum() < minimum:
                stats.append([np.nan] * 4)
            else:
                stats.append([x[live].mean(), y[live].mean(),
                              x[live].std(), y[live].std()])
        return {"names": names, "stats": np.array(stats),
                "transmission": float(live.sum()) / count}

    def _rows(self, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Which diagnostics the responses use, and the weight of each: every
        screen, with the readout carrying `READOUT_WEIGHT` of the total."""
        screens = np.array([SCREEN_TAG in name for name in names])
        readout = np.array([name == self.config.target_screen for name in names])
        weights = np.where(readout, READOUT_WEIGHT, 0.0)
        upstream = screens & ~readout
        weights[upstream] = (1 - READOUT_WEIGHT) / upstream.sum()
        return screens, weights

    # ---------- the optics ----------

    def _match(self, target, start: dict[str, float]):
        """Re-match the final focus to `target` in the twin, within limits."""
        names = list(FINAL_FOCUS)
        low = np.array([self.twin._knobs[n].low for n in names])
        high = np.array([self.twin._knobs[n].high for n in names])
        x0 = np.clip([start[n] for n in names], low, high)

        def residual(k):
            readings = self._probe({**start, **dict(zip(names, k))})
            final = readings["stats"][readings["names"].index(self.config.target_screen)]
            if not np.all(np.isfinite(final)):
                return np.array([10.0, 10.0, 10.0])
            return np.array([np.log(final[2] / target[0]),
                             np.log(final[3] / target[1]),
                             2.0 * (1.0 - readings["transmission"])])

        fit = least_squares(residual, x0, bounds=(low, high), diff_step=1e-3,
                            x_scale=np.maximum(np.abs(x0), 1.0), xtol=1e-6,
                            ftol=1e-6, max_nfev=60)
        k = {**start, **dict(zip(names, fit.x))}
        clipped = [n for n, v, lo, hi in zip(names, fit.x, low, high)
                   if min(v - lo, hi - v) < 1e-6 * (hi - lo)]
        return k, clipped

    def _modes(self, k1, knob_names, step, columns_of, fraction) -> Modes:
        """Weighted central-difference response of the readings to each knob."""
        base = self._probe(k1)
        screens, weights = self._rows(base["names"])
        rows = [f"{name}:{plane}" for name, w in zip(base["names"], weights) if w
                for plane in ("x", "y")]
        columns = []
        for name in knob_names:
            centre = self.twin._knobs[name].value if name not in k1 else k1[name]
            plus = self._probe({**k1, name: centre + step})
            minus = self._probe({**k1, name: centre - step})
            columns.append(columns_of(plus, minus, base, step)[weights > 0].ravel())
        response = np.nan_to_num(np.array(columns).T)
        response *= np.repeat(np.sqrt(weights[weights > 0] / 2), 2)[:, None]
        _, s, vt = np.linalg.svd(response, full_matrices=False)
        cumulative = np.cumsum(s**2) / np.sum(s**2)
        n = int(np.searchsorted(cumulative, fraction)) + 1
        scores = np.sqrt(((s[:n, None] * vt[:n]) ** 2).sum(axis=0))
        return Modes(names=list(knob_names), rows=rows, response=response,
                     singular_values=s, Vt=vt, n_modes=n,
                     scores=dict(zip(knob_names, scores.tolist())))

    def optics_for_target(self, sigma_x: float, sigma_y: float,
                          fraction: float = 0.99, modes: bool = True) -> TargetOptics:
        """The optics for a requested rms spot (metres) at the readout screen, and
        the quadrupole and corrector modes about it. See the module docstring.

        `modes=False` skips the two SVDs (the 208 finite-difference tracks) and
        returns the optics alone, in about a quarter of the time."""
        key = (float(sigma_x), float(sigma_y), float(fraction), modes)
        if not modes:
            full = (float(sigma_x), float(sigma_y), float(fraction), True)
            if full in self._optics_cache:
                return self._optics_cache[full]
        if key in self._optics_cache:
            return self._optics_cache[key]
        lo, hi = self._matched.axis[0], self._matched.axis[-1]
        inside = (float(np.clip(sigma_x, lo, hi)), float(np.clip(sigma_y, lo, hi)))
        start = self._matched.evaluate(*inside)["settings"]
        k1, clipped = self._match((sigma_x, sigma_y), start)

        def log_size(plus, minus, base, step):
            return (np.log(plus["stats"][:, 2:]) - np.log(minus["stats"][:, 2:])) / (2 * step)

        def centroid(plus, minus, base, step):
            return (plus["stats"][:, :2] - minus["stats"][:, :2]) / (2 * step)

        quads = kicks = None
        if modes:
            quads = self._modes(k1, list(self._matched.quadrupoles), QUAD_STEP,
                                log_size, fraction)
            kicks = self._modes(k1, self.corrector_names, CORRECTOR_STEP,
                                centroid, fraction)
        final = self._probe(k1)
        readout = final["stats"][final["names"].index(self.config.target_screen)]
        result = TargetOptics(
            target=(float(sigma_x), float(sigma_y)), k1=k1,
            quadrupoles=quads, correctors=kicks,
            achieved=(float(readout[2]), float(readout[3])),
            transmission=final["transmission"], clipped=clipped,
            size_modes=self._size_modes.evaluate(*inside))
        self._optics_cache[key] = result
        return result

    @property
    def corrector_names(self) -> list[str]:
        """Every corrector plane the agent may set, `_h` and `_v`."""
        return [n for n in self.magnet_names
                if self._knobs[n].prop in ("horizontal_angle", "vertical_angle")]

    def optics_action(self, optics: TargetOptics) -> dict:
        """An action for `step()` that loads the optics' quadrupoles."""
        return {"magnets": dict(optics.k1)}
