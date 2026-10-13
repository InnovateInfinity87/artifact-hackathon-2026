import cheetah
import torch
import numpy as np

from rl_framework.config import RLConfig
from rl_framework.envs.base_env import BaseAcceleratorEnv






class SimEnv(BaseAcceleratorEnv):
    """
    Simulation environment using cheetah-accelerator.
    No CATAP dependency — runs entirely in PyTorch.
    """

    def __init__(self, config: RLConfig):
        super().__init__(config)
        self._lattice: cheetah.Segment = cheetah.Segment.from_lattice_json(self.config.lattice_path)

    def _hard_reset(self) -> np.ndarray:
        # Activate the target screen (screens default to inactive in the lattice JSON)
        idx = self._lattice.element_index(self.config.target_screen)
        self._lattice.elements[idx].is_active = True

        # Zero all controllable magnets
        self._set_magnets_to_zero()

        return self._get_obs()

    def _set_magnets_to_zero(self):
        """Set every controllable magnet to its native zero."""
        native = np.zeros(len(self.config.magnet_config), dtype=np.float32)
        self._apply_action(native)

    def _apply_action(self, native_values: np.ndarray):
        for i, name in enumerate(self._magnet_names):
            if name[-2:] == "_h" or name[-2:] == "_v": # handle combined corrector naming convention
                idx = self._lattice.element_index(name[:-2])
            else:
                idx = self._lattice.element_index(name)
            el = self._lattice.elements[idx]
            val = float(native_values[i])

            if isinstance(el, cheetah.Quadrupole):
                el.k1 = torch.tensor(val)
            elif isinstance(el, cheetah.CombinedCorrector):
                if name[-2:] == "_h":
                    el.horizontal_angle = torch.tensor(val)
                elif name[-2:] == "_v":
                    el.vertical_angle = torch.tensor(val)
                else:
                    raise ValueError(f"Combined corrector {name} does not follow naming convention with _h or _v suffix.")
            elif isinstance(el, cheetah.Dipole):
                el.k1 = torch.tensor(val)
            elif isinstance(el, cheetah.HorizontalCorrector) or isinstance(el, cheetah.VerticalCorrector):
                el.angle = torch.tensor(val)
            else:
                raise ValueError(f"Element {name} is not a supported magnet type.")
            
        # Create beam
        beam = self._beam_factory()
        self._lattice.track(beam)

    def _get_screen_statistics(self, screen_name: str) -> tuple[float, float, float, float, bool]:
        idx = self._lattice.element_index(screen_name)
        screen: cheetah.Screen = self._lattice.elements[idx]
        return self._read_screen_stats(screen.reading.detach().numpy())

    def _beam_factory(self) -> cheetah.ParticleBeam:
        """Create a ParticleBeam from the Twiss parameters stored in config."""
        b = self.config.beam
        return cheetah.ParticleBeam.from_twiss(
            num_particles=b["num_particles"],
            alpha_x=torch.tensor(b["alpha_x"]),
            beta_x=torch.tensor(b["beta_x"]),
            emittance_x=torch.tensor(b["emittance_x"]),
            alpha_y=torch.tensor(b["alpha_y"]),
            beta_y=torch.tensor(b["beta_y"]),
            emittance_y=torch.tensor(b["emittance_y"]),
            sigma_p=torch.tensor(b["sigma_p"]),
            energy=torch.tensor(b["energy"]),
            total_charge=torch.tensor(b["total_charge"]),
        )