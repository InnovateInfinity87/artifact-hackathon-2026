"""The run_ment.py reconstruction with every xsuite re-tracking replaced by the fitted
polynomial maps of surrogate.py (one matrix product per stage instead of about 3 min of
tracking). Same steps and outputs, written with a suffix."""

from __future__ import annotations

import sys
import time

import common
import full6d
import ment6d
import numpy as np
import run_ment
import surrogate


def use_maps(maps):
    """Route ment6d.track_all through the maps."""
    def track_all(line, sample, configs, seed=0, workers=4):
        names = {c.name for c in configs}
        keys = [k for k in maps.keys if k[0] in names]
        return maps.indices(sample, np.random.default_rng(seed), keys)

    ment6d.track_all = track_all


UPSTREAM_SCREENS = ("s02_", "s03_", "s04_", "s05_")
FEW_SCREENS = ("cla_s02_dia_scr_01", "cla_s02_dia_scr_03", "cla_s03_dia_scr_01", "cla_s04_dia_scr_03",
               "cla_s05_dia_scr_01")


def lean_sets(keys):
    """Nested image sets, from the full plan down. Name -> list of (config, screen)."""
    up = lambda s: any(t in s for t in UPSTREAM_SCREENS)
    full = list(keys)
    s2 = [k for k in full if k[0] != "nominal_off" or up(k[1])]
    s3 = [k for k in s2 if k[0] != "s02b"]
    s4 = [k for k in s3 if not k[0].startswith(("s07_03", "s07_04"))]
    s5 = [k for k in s4 if not up(k[1]) or k[1] in FEW_SCREENS]
    s6 = [k for k in s5 if not k[0].startswith("s07_02")]
    s7 = [k for k in s6 if k[0] != "s02c"]
    return {"full": full, "upstream nominal only": s2, "two S02 settings": s3,
            "two streak screens": s4, "five transverse screens": s5, "one streak screen": s6,
            "one S02 setting": s7}


def run(line, images_path=common.HERE / "images.npz", set_name="full", suffix="_fast", log=print):
    t0 = time.time()
    use_maps(surrogate.load_combined())
    run_ment.OUT_JSON = common.HERE / f"ment_results{suffix}.json"
    run_ment.OUT_NPZ = common.HERE / f"ment_full{suffix}.npz"
    if set_name in run_ment.image_sets([]):
        results = run_ment.run(line, images_path, set_name, log=log)
    else:
        original = run_ment.image_sets
        run_ment.image_sets = lambda keys: {set_name: lean_sets(keys)[set_name]}
        try:
            results = run_ment.run(line, images_path, set_name, log=log)
        finally:
            run_ment.image_sets = original
    results["seconds"] = time.time() - t0
    common.write_json(run_ment.OUT_JSON, results, indent=1)
    log(f"reconstruction took {results['seconds']:.0f} s")
    return results


if __name__ == "__main__":
    run(full6d.build_line(), set_name=sys.argv[1] if len(sys.argv) > 1 else "full")
