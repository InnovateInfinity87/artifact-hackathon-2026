"""Run the agent over a set of SimEnv episodes and report it against the baselines.

    python -I evaluate.py --episodes 20
    python -I evaluate.py --episodes 20 --fit-misalignments
    python -I evaluate.py --episodes 5 --methods mbrl,do-nothing --verbose

Every method sees the same episodes: the seed fixes the beam, the survey error and
the request, so a difference between two rows is a difference between two methods and
not between two machines.

The headline is `score`, the mean of `info["benchmark_score"]`, which is
`score.Report.calc_score` evaluated for each episode as a single case — the mismatch
in requested spot sizes plus `EVAL_WEIGHT` for every doubling of `EVAL_BUDGET` shots.
Lower is better. The hit rates alongside it are `score.HIT_TOLERANCES`, the fraction
of episodes whose mean component error came inside 50 %, 25 % and 10 % of the spot
that was asked for.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RL_framework"))

from rl_framework.config import RLConfig  # noqa: E402

from MFopt.optics_env import OpticsEnv  # noqa: E402

from agent import MBRLAgent  # noqa: E402
from baselines import BASELINES  # noqa: E402
from config import MBRLConfig  # noqa: E402
from twin import CheetahTwin, tune_threads  # noqa: E402

tune_threads()

HIT_TOLERANCES = (0.50, 0.25, 0.10)  # `score.HIT_TOLERANCES`


_WORKER: dict = {}


def _worker_init(config, particles, variation, misalignment) -> None:
    """Give this process its own environment, twin and agent.

    None of the three can cross a process boundary — the twin holds torch tensors
    wired into a Cheetah segment — so each worker builds its own once and reuses it
    for every episode it is handed. Building costs about a second; an episode costs
    minutes.
    """
    tune_threads()
    env, twin = build(config, particles, variation, misalignment)
    _WORKER.update(
        config=config, env=env, twin=twin, agent=MBRLAgent(config, twin)
    )


def _worker_run(task):
    """One (method, seed) episode, as a plain dictionary the parent can collect."""
    name, seed = task
    if name == "mbrl":
        result = _WORKER["agent"].run(_WORKER["env"], seed=seed)
    else:
        result = BASELINES[name](_WORKER["env"], _WORKER["twin"], _WORKER["config"], seed=seed)
    return name, seed, result


def build(config: MBRLConfig, particles: int, variation: float, misalignment: float):
    """The environment and the twin.

    An `OpticsEnv`, because the pipeline starts by asking it for the optics of the
    requested spot. Every magnet stays settable: the optics load all 46 quadrupoles
    and the SVD chooses the rest, so there is no fixed list to configure.

    The screen grid is finer than SimEnv's default. The agent reads its beam off the
    camera frame rather than from the particle moments, and at 128 by 128 a pixel is
    175 by 109 microns — wider than the smallest spot the benchmark asks for, and
    enough to mismeasure an rms by 70 microns. At 512 the measure tracks the true
    moments to a tenth of a micron.
    """
    twin = CheetahTwin(num_particles=config.twin_particles)
    rl = RLConfig()
    rl.num_particles = particles
    rl.beam_variation = variation
    rl.misalignment_variation = misalignment
    rl.screen_shape = config.screen_shape
    return OpticsEnv(rl), twin


def summarise(name: str, results: list, seconds: float) -> dict:
    """One row of the table: how the method did over the episodes it was given."""
    scores = np.array([r.benchmark_score for r in results])
    errors = np.array([r.relative_error for r in results])
    return {
        "method": name,
        "score": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "mismatch": float(np.mean([r.mismatch for r in results])),
        "shots": float(np.mean([r.shots for r in results])),
        "delivered": float(np.mean([r.on_screen for r in results])),
        "transmission": float(np.mean([r.transmission for r in results])),
        "orbit_um": float(np.mean([r.orbit_rms for r in results]) * 1e6),
        "optics_mismatch": float(np.mean([r.optics_score for r in results])),
        **{f"hit_{int(t * 100)}": float((errors <= t).mean()) for t in HIT_TOLERANCES},
        "seconds": seconds,
        "episodes": len(results),
    }


def table(rows: list[dict]) -> str:
    header = (
        f"{'method':<14}{'score':>8}{'median':>8}{'mismatch':>10}{'shots':>7}"
        f"{'on scr':>8}{'<50%':>7}{'<25%':>7}{'<10%':>7}{'s/ep':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['method']:<14}{row['score']:>8.3f}{row['score_median']:>8.3f}"
            f"{row['mismatch']:>10.3f}{row['shots']:>7.1f}"
            f"{row['delivered']:>7.0%}{row['hit_50']:>7.0%}{row['hit_25']:>7.0%}"
            f"{row['hit_10']:>7.0%}{row['seconds'] / row['episodes']:>8.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234, help="first episode seed")
    parser.add_argument("--particles", type=int, default=10_000)
    parser.add_argument("--twin-particles", type=int, default=4_000)
    parser.add_argument("--beam-variation", type=float, default=1.0)
    parser.add_argument("--misalignment-variation", type=float, default=1.0)
    parser.add_argument(
        "--fit-misalignments",
        action="store_true",
        help="fit the survey error as well as the beam (an ablation; see config.py)",
    )
    parser.add_argument("--steering-shots", type=int, default=None)
    parser.add_argument("--focus-shots", type=int, default=None)
    parser.add_argument(
        "--methods",
        default="mbrl," + ",".join(BASELINES),
        help="comma-separated: mbrl and any of " + ", ".join(BASELINES),
    )
    parser.add_argument("--out", type=Path, default=None, help="write results as JSON")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="episodes to run at once; each pins itself to one thread (1 = serial)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = MBRLConfig()
    config.twin_particles = args.twin_particles
    config.fit_misalignments = args.fit_misalignments
    if args.steering_shots is not None:
        config.steering_shots = args.steering_shots
    if args.focus_shots is not None:
        config.focus_shots = args.focus_shots

    seeds = [args.seed + i for i in range(args.episodes)]
    methods = [name.strip() for name in args.methods.split(",")]
    unknown = [n for n in methods if n != "mbrl" and n not in BASELINES]
    if unknown:
        raise SystemExit(f"{unknown}: not methods. mbrl, {', '.join(BASELINES)}")

    settings = (config, args.particles, args.beam_variation, args.misalignment_variation)
    tasks = [(name, seed) for name in methods for seed in seeds]
    collected: dict[str, list] = {name: [] for name in methods}
    start_times = {name: time.time() for name in methods}
    done = 0

    if args.workers > 1 and not args.verbose:
        # Episodes are independent and each one pins itself to a single thread, so
        # the only way to use the rest of the machine is to run several at once.
        # A twenty-episode comparison goes from most of an hour to a few minutes.
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(
            args.workers, initializer=_worker_init, initargs=settings
        ) as pool:
            for name, seed, result in pool.imap_unordered(_worker_run, tasks):
                collected[name].append(result)
                done += 1
                print(
                    f"  {name:<14} {done:>3}/{len(tasks)}  seed {seed}  "
                    f"score {result.benchmark_score:7.3f}  shots {result.shots:>3}",
                    flush=True,
                )
    else:
        env, twin = build(*settings)
        agent = MBRLAgent(config, twin)
        for name, seed in tasks:
            if name == "mbrl":
                result = agent.run(env, seed=seed, verbose=args.verbose)
            else:
                result = BASELINES[name](env, twin, config, seed=seed)
            collected[name].append(result)
            done += 1
            print(
                f"  {name:<14} {done:>3}/{len(tasks)}  seed {seed}  "
                f"score {result.benchmark_score:7.3f}  shots {result.shots:>3}",
                flush=True,
            )

    elapsed = time.time() - min(start_times.values())
    rows = [
        summarise(name, collected[name], elapsed * len(collected[name]) / max(done, 1))
        for name in methods
    ]
    print()

    print(table(rows))
    print(
        f"\n{args.episodes} episodes, seeds {seeds[0]}-{seeds[-1]}, "
        f"beam variation {args.beam_variation}, misalignment "
        f"{args.misalignment_variation}, "
        f"misalignment fit {'on' if config.fit_misalignments else 'off'}."
    )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
