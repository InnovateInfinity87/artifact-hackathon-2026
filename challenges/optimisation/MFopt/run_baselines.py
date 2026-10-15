"""Run the baselines and stage A on seeded episodes.

    python -m MFopt.run_baselines --seeds 30 --out MFopt/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .harness import run_episode, save, summarise
from .optics_env import OpticsEnv
from .steering import optics_and_steering, optics_only

POLICIES = {
    "do_nothing": lambda session: None,
    "optics_only": optics_only,
    "optics_and_steering": optics_and_steering,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--out", default="MFopt/results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = OpticsEnv()
    for name in args.policies:
        results = []
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            result = run_episode(env, POLICIES[name], seed)
            results.append(result)
            print(f"{name} seed {seed}: score {result.score:.2f} "
                  f"rel.err {result.relative_error:.2f} T {result.transmission:.2f} "
                  f"steps {result.steps} ({result.seconds:.0f} s)", flush=True)
        save(results, out / f"{name}.json")
        print(name, summarise(results), flush=True)


if __name__ == "__main__":
    main()
