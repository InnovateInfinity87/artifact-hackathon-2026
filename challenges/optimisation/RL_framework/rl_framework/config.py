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
