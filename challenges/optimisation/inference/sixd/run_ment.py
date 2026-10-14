"""Run the MENT reconstruction at the S02 start for nested image sets and score it against
the true beam (used for scoring only). Writes ment_results.json and ment_full.npz."""

from __future__ import annotations

import sys
import time

import common
import full6d
import ment6d
import numpy as np

OUT_JSON = common.HERE / "ment_results.json"
OUT_NPZ = common.HERE / "ment_full.npz"


def image_sets(keys):
    """Nested measurement sets: name -> list of (config, screen) keys."""
    def pick(predicate):
        return [k for k in keys if predicate(k[0])]

    nominal = pick(lambda c: c == "nominal_off")
    s02 = pick(lambda c: c.startswith("s02"))
    streak = pick(lambda c: c.startswith("s07"))
    spectrometer = pick(lambda c: c.startswith("fea"))
    return {
        "nominal": nominal,
        "+S02 quads": nominal + s02,
        "+streak": nominal + s02 + streak,
        "full": nominal + s02 + streak + spectrometer,
    }


def main() -> None:
    images_path = common.HERE / (sys.argv[1] if len(sys.argv) > 1 else "images.npz")
    only = sys.argv[2:] or None
    line = full6d.build_line()
    configs = full6d.make_plan(line)
    keys, images, n_meas, beam = ment6d.load_images(images_path)
    truth = ment6d.summary(beam)
    results = {"truth": truth, "sets": {}}
    if OUT_JSON.exists():
        results["sets"] = common.read_json(OUT_JSON).get("sets", {})
    for name, used in image_sets(keys).items():
        if only and name not in only:
            continue
        t0 = time.time()
        print(f"== {name}: {len(used)} images", flush=True)
        out = ment6d.reconstruct(line, configs, keys, images, n_meas, keys_used=used)
        reco = ment6d.summary(out["sample"], out["weights"])
        results["sets"][name] = {
            "n_images": len(used),
            "summary": reco,
            "stages": [{"factor": s["factor"], "history": s["history"]} for s in out["stages"]],
            "seconds": time.time() - t0,
        }
        for k in truth:
            if k != "cov":
                print(f"   {k:18s} true {truth[k]: .4e} reco {reco[k]: .4e} ratio {reco[k] / truth[k]: .3f}")
        common.write_json(OUT_JSON, results, indent=1)
        if name == "full":
            keep = [k for k in used]
            np.savez_compressed(
                OUT_NPZ,
                sample=out["sample"],
                weights=out["weights"],
                keys=np.array([f"{c}|{s}" for c, s in keep]),
                predicted=np.array([ment6d.predicted(out["indices"], out["weights"], k) for k in keep]),
                measured=np.array([out["data"][k] for k in keep]),
            )


if __name__ == "__main__":
    main()
