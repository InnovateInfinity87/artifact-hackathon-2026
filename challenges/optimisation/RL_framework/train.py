import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from rl_framework.config import RLConfig
from rl_framework.envs.sim_env import SimEnv


class ProgressCallback(BaseCallback):
    """Print rolling reward stats every `report_every` steps."""

    def __init__(self, report_every: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.report_every = report_every
        self._last_report = 0
        self._start_time = time.monotonic()
        self._rewards = []

    def _on_step(self) -> bool:
        steps = self.num_timesteps
        self._rewards.append(self.locals.get("rewards", [float("nan")])[0])
        if steps - self._last_report >= self.report_every:
            self._last_report = steps
            elapsed = time.monotonic() - self._start_time
            recent = self._rewards[-100:]
            if recent:
                print(
                    f"  [{steps:>6d} steps | {elapsed:.1f}s] "
                    f"rew100_mean={np.mean(recent):+.3f}  "
                    f"rew100_std={np.std(recent):.3f}"
                )
            else:
                print(f"  [{steps:>6d} steps | {elapsed:.1f}s]")
        return True


def train(config: RLConfig, total_timesteps: int = 100, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)

    # Single environment with normalisation wrappers
    env = SimEnv(config)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env)
    env = Monitor(env, filename=os.path.join(model_dir, "monitor.csv"))

    # Checkpoint every 5000 steps
    checkpoint_cb = CheckpointCallback(
        save_freq=10,
        save_path=model_dir,
        name_prefix="ppo_accelerator",
    )

    # Custom progress callback
    progress_cb = ProgressCallback(report_every=1)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=10
    )
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, progress_cb])
    model.save(os.path.join(model_dir, "ppo_accelerator_final"))
    print(f"Training complete. Model saved to {model_dir}/")
    return model


if __name__ == "__main__":
    cfg = RLConfig()
    train(cfg)
