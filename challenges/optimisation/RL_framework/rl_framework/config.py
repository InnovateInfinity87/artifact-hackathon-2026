from dataclasses import dataclass, field
from typing import Any


@dataclass
class RLConfig:
    # Path to the cheetah lattice JSON
    lattice_path: str = "FEBE_full.json"

    # Beam factory parameters (Twiss + energy + charge)
    beam: dict[str, Any] = field(default_factory=lambda: {
        "num_particles": 10000,
        "alpha_x": -0.949, "beta_x": 14.236, "emittance_x": 3.29e-9,
        "alpha_y": 72.003, "beta_y": 925.297, "emittance_y": 4.20e-9,
        "sigma_p": 0.01, "energy": 250e6, "total_charge": 10e-12,
    })

    # Target diagnostic screen — reward minimises beam size here
    target_screen: str = "cla_s04_dia_scr_03"

    # Magnets the RL agent can control.
    # Each entry: [name, min_native_value, max_native_value]
    # Native units: k1 (1/m²) for quadrupoles, angle (mrad) for correctors.
    magnet_config: list[list] = field(default_factory=lambda: [
        ["cla_s04_mag_hvcor_01_h",  -5.0e-3, 5.0e-3],
        ["cla_s04_mag_hvcor_01_v",  -5.0e-3, 5.0e-3],
        ["cla_s04_mag_quad_01",  -15.0, 15.0],
        ["cla_s04_mag_quad_02",  -15.0, 15.0],
        ["cla_s04_mag_quad_03",  -15.0, 15.0],
    ])

    # Episode length before truncation
    max_episode_steps: int = 100

    # ---------------- nominal xsuite model (SimEnv only) ---------------- #
    # Everything below describes the model in `nominal_model/`, which SimEnv tracks
    # through. `lattice_path`, `beam` and `magnet_config` above are for RealEnv.

    # Which model tracks the beam. "cheetah" is first-order magnets and achromatic
    # cavities, the physics `score.py` is scored on, and starts in a second;
    # "xsuite" is thick elements and cavities that focus at each particle's own
    # energy, closer to the machine off momentum, and spends a minute compiling a
    # tracking kernel. Same lattice, same beam, same interface either way.
    backend: str = "cheetah"

    # Where `xsuite_model.py` lives, and what it is given. None takes the defaults
    # of the nominal model: the challenge lattice file and its starting beam.
    nominal_model_dir: str | None = None
    nominal_lattice_json: str | None = None
    distribution_path: str | None = None

    # The stretch of machine that is modelled. The default is the benchmark's:
    # the start of S02 to the FEBE hutch screen scored by `score.py`.
    lattice_start: str = "cla_s02_sim_aper_01"
    lattice_end: str = "cla_feh_dia_scr_01"

    # Particles tracked per step, taken from the front of the 200,000 in the
    # distribution file. None uses all of them (slow).
    num_particles: int | None = 10_000

    # Screen images are binned onto this (rows, columns) grid, covering the whole
    # chip whatever its native resolution is. `env.screen_extent(name)` gives the
    # physical size the grid spans.
    screen_shape: tuple[int, int] = (128, 128)
    # Divide each image by its brightest pixel, so observations are in [0, 1].
    normalise_image: bool = True

    # Beam pipe the particles have to stay inside; a particle that leaves it stops
    # counting towards every reading further down the line. None removes the pipe.
    beam_pipe_radius: float | None = 50e-3
    # Fraction of the launched beam a diagnostic needs for its reading to count.
    min_screen_fraction: float = 0.10
    # Gaussian noise added to each BPM reading, in metres.
    bpm_noise: float = 0.0

    # How much the machine varies from one episode to the next. Each is a multiplier
    # on the spreads at the top of `sim_env.py`: 0 is the nominal model exactly, 1 is
    # the machine as it realistically differs shot to shot and day to day, and larger
    # values stay a harder version of the same problem rather than a different one.
    # `reset()` draws a new beam, a new set point and a new survey error from them.
    beam_variation: float = 1.0
    magnet_variation: float = 1.0
    misalignment_variation: float = 1.0

    # Shot-to-shot jitter, as a fraction of `beam_variation`: within an episode the
    # machine holds still, but every step draws its own beam around the episode's,
    # so two identical actions do not give identical readings. Scales with
    # `beam_variation`, so setting that to 0 leaves the environment deterministic.
    beam_jitter: float = 0.1

    # Where each magnet's operating range comes from. "current" is what
    # +-`magnet_current_limit` amps reaches through the calibration in
    # `data/clara/lattice_data/utils.py` — what the hardware can physically do.
    # "archiver" is where the magnets happened to run over six months, from
    # `magnet_ranges.json`; far narrower, and it has the FEA arc quadrupoles at
    # [0, 0]. Magnets with no calibration (the sextupoles) fall back to the
    # archiver either way.
    magnet_range_source: str = "current"
    magnet_current_limit: float = 100.0  # A

    # Offer the VBC chicane as a knob: four dipoles on one setting, up to 120 mrad.
    # It is the only handle in this line on the energy spread at the readout.
    # Careful: a dipole angle is not in `score.TUNABLE`, so a tune that moves it
    # fails the benchmark's validation. Set False to keep the action space scorable.
    include_vbc_chicane: bool = True

    # Magnets the agent may set, by knob name (`<quadrupole>`, `<sextupole>`,
    # `<corrector>_h` / `_v`, or `cla_vbc_chicane`). None gives it every magnet in
    # the modelled stretch.
    controllable_magnets: list[str] | None = None
    # {knob name: [low, high]} overriding the ranges worked out above.
    magnet_limits: dict[str, list[float]] | None = None
