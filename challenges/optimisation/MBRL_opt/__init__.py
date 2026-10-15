"""Model-based RL tuning of CLARA against the `RL_framework` `SimEnv`.

The model is a differentiable Cheetah twin of the machine rather than a learned
network: the lattice is known, and only which particular machine this episode dealt
has to be worked out, from the shots the agent takes. See `README.md`.

    from MBRL_opt.agent import MBRLAgent, controllable_magnets
    from MBRL_opt.config import MBRLConfig
"""

from config import MBRLConfig
from agent import EpisodeResult, MBRLAgent, controllable_magnets
from twin import CheetahTwin, LatentState, Readings, tune_threads

__all__ = [
    "CheetahTwin",
    "EpisodeResult",
    "LatentState",
    "MBRLAgent",
    "MBRLConfig",
    "Readings",
    "controllable_magnets",
    "tune_threads",
]
