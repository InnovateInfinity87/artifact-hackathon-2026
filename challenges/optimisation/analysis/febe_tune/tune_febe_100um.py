"""Tune CLARA to a centred 100 x 100 um spot at CLA-FEH-DIA-SCR-01.

The record of the run documented in analysis/febe_tune/README.md, assembled from the
stages it was driven in. Every measurement goes through `env.step`, so the shot count
is honest; the one exception is `diagnose()`, which reads the energy spread out of the
model and is used only to set the chirp, a quantity no screen measures.

    python tune_febe_100um.py          # writes the plots and the tune

Stages A-H produced a 100 x 100 um spot and were verified to reproduce end to end
from a clean interpreter. Stages I-K were run afterwards, once `score.py` started
charging for beam that never arrives, and redo the match against that.
"""

# ==========================================================================
# stage A
# ==========================================================================
import sys, os, json, time
sys.path.insert(0, "/home/jovyan/work/pylibs")
ROOT = "/home/jovyan/work/artifact-hackathon-2026/challenges/optimisation"
sys.path.insert(0, ROOT + "/RL_framework"); sys.path.insert(0, ROOT)
import numpy as np, torch, cheetah
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rl_framework import RLConfig, SimEnv

OUT = "/home/jovyan/work/febe_tune"
os.makedirs(OUT, exist_ok=True)
TARGET = (0.0, 0.0, 100e-6, 100e-6)
PROBE = {"k1": 0.3, "k2": 2.0, "horizontal_angle": 1e-4, "vertical_angle": 1e-4,
         "angle": 0.005, "phase": 1.0}
FAMILY = {"k1": "quadrupole", "k2": "sextupole", "horizontal_angle": "corrector",
          "vertical_angle": "corrector", "angle": "VBC chicane", "phase": "cavity phase"}
COLOUR = {"quadrupole": "#1f77b4", "sextupole": "#9467bd", "corrector": "#2ca02c",
          "VBC chicane": "#d62728", "cavity phase": "#ff7f0e"}

env = SimEnv(RLConfig(target=TARGET, beam_variation=0, misalignment_variation=0,
                      beam_jitter=0, power_supply_ripple=0, max_episode_steps=10**6))
shots = [0]
def shot(action=None):
    shots[0] += 1
    return env.step(action or {})[-1]

obs, info = env.reset(seed=0)
base = shot()
print(f"baseline: sigma ({base['sigma_x']*1e6:.0f}, {base['sigma_y']*1e6:.0f}) um, "
      f"mu ({base['mu_x']*1e3:+.2f}, {base['mu_y']*1e3:+.2f}) mm, "
      f"on chip {base['on_screen_fraction']:.1%}, score {base['benchmark_score']:.2f}")
globals().update(env=env, shot=shot, shots=shots, OUT=OUT, TARGET=TARGET)

def diagnose():
    """Not a measurement the agent has: where the horizontal size comes from."""
    coords = env._coordinates
    beam = cheetah.ParticleBeam(
        particles=torch.tensor(np.column_stack([coords, np.ones(len(coords))]),
                               dtype=torch.float32),
        energy=torch.tensor(float(env.energy_ev), dtype=torch.float32))
    live = np.ones(len(coords), bool)
    for stop in env._stops:
        chunk = env.transport._chunks[stop.name]
        if len(chunk.elements):
            beam = chunk.track(beam)
        x, y = beam.x.detach().numpy(), beam.y.detach().numpy()
        live &= x**2 + y**2 <= env.config.beam_pipe_radius**2
        if stop.name == env.target_screen:
            break
    chip = np.array(env.screen_extent())
    seen = live & (np.abs(x) <= chip[1]) & (np.abs(y) <= chip[3])
    d = beam.p.detach().numpy()
    Dx = np.cov(x[seen], d[seen])[0, 1] / np.var(d[seen])
    return dict(sigma_delta=d[seen].std(), Dx=Dx,
                dispersive=abs(Dx) * d[seen].std(),
                betatron=np.std(x[seen] - Dx * d[seen]), imaged=seen.mean())
globals()["diagnose"] = diagnose
d = diagnose()
print(f"  diagnosis: sigma_delta {d['sigma_delta']*100:.2f} %, Dx {d['Dx']*1e3:+.1f} mm, "
      f"dispersive {d['dispersive']*1e6:.0f} um, betatron {d['betatron']*1e6:.0f} um")

# ---- sensitivity of every knob
print(f"\nprobing all {len(env.magnet_names)} knobs...")
t0 = time.time()
rows, previous = [], None
reference = np.array([np.log(base["sigma_x"]), np.log(base["sigma_y"]),
                      base["mu_x"], base["mu_y"]])
settings = env.magnet_settings()
for name in env.magnet_names:
    knob = env._knobs[name]
    step = PROBE[knob.prop]
    action = {name: float(settings[name] + step)}
    if previous is not None:
        action[previous] = float(settings[previous])
    i = shot(action)
    previous = name
    if i["on_screen"]:
        here = np.array([np.log(i["sigma_x"]), np.log(i["sigma_y"]),
                         i["mu_x"], i["mu_y"]])
        row = (here - reference) / step
    else:
        row = np.full(4, np.nan)
    rows.append(dict(name=name, family=FAMILY[knob.prop], prop=knob.prop,
                     dlogsx=row[0], dlogsy=row[1], dmux=row[2], dmuy=row[3],
                     step=step))
shot({previous: float(settings[previous])})
print(f"  {time.time()-t0:.0f}s, {shots[0]} shots so far")

for r in rows:  # size leverage, per probe step, both planes together
    r["size"] = float(np.hypot(np.nan_to_num(r["dlogsx"]), np.nan_to_num(r["dlogsy"]))
                      * r["step"])
    r["steer"] = float(np.hypot(np.nan_to_num(r["dmux"]), np.nan_to_num(r["dmuy"]))
                       * r["step"])
json.dump(rows, open(f"{OUT}/sensitivity.json", "w"), indent=1)

ranked = sorted(rows, key=lambda r: -r["size"])[:28]
figure, axes = plt.subplots(1, 2, figsize=(13, 7))
for ax, key, title in ((axes[0], "size", "effect on spot size, |dlog sigma| per probe step"),
                       (axes[1], "steer", "effect on position, |d mu| per probe step (m)")):
    chosen = sorted(rows, key=lambda r: -r[key])[:28]
    names = [r["name"].replace("cla_", "").replace("_mag_", " ") for r in chosen]
    ax.barh(range(len(chosen)), [r[key] for r in chosen],
            color=[COLOUR[r["family"]] for r in chosen])
    ax.set_yticks(range(len(chosen))); ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis(); ax.set_title(title, fontsize=10); ax.set_xscale("log")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOUR.values()]
figure.legend(handles, COLOUR.keys(), ncol=5, loc="lower center", frameon=False)
figure.suptitle("Which knobs move the beam at CLA-FEH-DIA-SCR-01", fontsize=12)
figure.tight_layout(rect=(0, 0.04, 1, 1))
figure.savefig(f"{OUT}/01_sensitivity.png", dpi=110); plt.close(figure)
print("  top size knobs:", [r["name"] for r in ranked[:8]])
print("  families in the top 28:",
      {f: sum(r["family"] == f for r in ranked) for f in COLOUR})

# ==========================================================================
# stage B
# ==========================================================================
import json
import numpy as np, torch, cheetah
import matplotlib.pyplot as plt

rows = json.load(open(f"{OUT}/sensitivity.json"))
# Per probe step a knob's leverage depends on the step I happened to choose. Rank
# instead by what it can do over its whole range, which is scale free.
for r in rows:
    low, high = env.magnet_limits[r["name"]]
    r["authority"] = float(np.hypot(np.nan_to_num(r["dlogsx"]),
                                    np.nan_to_num(r["dlogsy"])) * (high - low) / 2)
ranked = sorted(rows, key=lambda r: -r["authority"])
print("knobs ranked by what they can do over their full range:")
for r in ranked[:12]:
    print(f"  {r['name']:26s} {r['family']:13s} {r['authority']:8.2f}")
print("  best of each family:",
      {f: round(max((r["authority"] for r in rows if r["family"] == f), default=0), 2)
       for f in ("quadrupole", "cavity phase", "VBC chicane", "sextupole", "corrector")})

COLOUR = {"quadrupole": "#1f77b4", "sextupole": "#9467bd", "corrector": "#2ca02c",
          "VBC chicane": "#d62728", "cavity phase": "#ff7f0e"}
chosen = ranked[:30]
figure, ax = plt.subplots(figsize=(7.5, 8))
ax.barh(range(len(chosen)), [r["authority"] for r in chosen],
        color=[COLOUR[r["family"]] for r in chosen])
ax.set_yticks(range(len(chosen)))
ax.set_yticklabels([r["name"].replace("cla_", "").replace("_mag_", " ")
                    for r in chosen], fontsize=7)
ax.invert_yaxis(); ax.set_xscale("log")
ax.set_xlabel("|d log sigma| over the knob's full range")
ax.set_title("Knob authority at CLA-FEH-DIA-SCR-01", fontsize=11)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOUR.values()],
          COLOUR.keys(), frameon=False, fontsize=8, loc="lower right")
figure.tight_layout(); figure.savefig(f"{OUT}/01_authority.png", dpi=110); plt.close(figure)

# ---- is 100 um even reachable? Emittance at the readout sets the beta needed.
def emittance():
    coords = env._coordinates
    beam = cheetah.ParticleBeam(
        particles=torch.tensor(np.column_stack([coords, np.ones(len(coords))]),
                               dtype=torch.float32),
        energy=torch.tensor(float(env.energy_ev), dtype=torch.float32))
    out = env.segment.track(beam)
    x, px, y, py, d = (out.x.detach().numpy(), out.px.detach().numpy(),
                       out.y.detach().numpy(), out.py.detach().numpy(),
                       out.p.detach().numpy())
    thin = np.abs(d - d.mean()) < 0.002
    result = {}
    for plane, (q, p) in (("x", (x, px)), ("y", (y, py))):
        for label, mask in (("projected", slice(None)), ("in a 0.2% slice", thin)):
            cov = np.cov(q[mask], p[mask])
            result[(plane, label)] = float(np.sqrt(np.linalg.det(cov)))
    return result
e = emittance()
print("\ngeometric emittance at the readout, and the beta 100 um would need:")
for (plane, label), value in e.items():
    print(f"  {plane} {label:16s} {value*1e9:7.2f} nm   ->  beta "
          f"{(100e-6)**2 / value:7.2f} m")

# ---- one knob at a time
def scan(knob, values, note):
    out = []
    for v in values:
        i = shot({knob: float(v)})
        d = diagnose()
        out.append((v, i["sigma_x"], i["sigma_y"], i["on_screen_fraction"],
                    d["sigma_delta"], d["betatron"]))
    shot({knob: float(env._nominal_strengths[knob])})
    print(f"  {note}: sigma_x {min(o[1] for o in out)*1e6:.0f}-"
          f"{max(o[1] for o in out)*1e6:.0f} um, sigma_delta "
          f"{min(o[4] for o in out)*100:.2f}-{max(o[4] for o in out)*100:.2f} %")
    return np.array(out)

print("\nscanning the big handles")
scans = {
    "cla_feh_mag_quad_02": scan("cla_feh_mag_quad_02", np.linspace(-9, 9, 13), "FEH-QUAD-02"),
    "cla_vbc_chicane": scan("cla_vbc_chicane", np.linspace(0, 0.115, 13), "VBC chicane"),
    "cla_l04_lin_cav_01": scan("cla_l04_lin_cav_01", np.linspace(15, 75, 13), "L04 phase"),
    "cla_l03_lin_cav_01": scan("cla_l03_lin_cav_01", np.linspace(-22, 38, 13), "L03 phase"),
    "cla_l02_lin_cav_01": scan("cla_l02_lin_cav_01", np.linspace(-7, 53, 13), "L02 phase"),
    "cla_l4h_lin_cav_01": scan("cla_l4h_lin_cav_01", np.linspace(-214, -154, 13), "4HC phase"),
}
np.save(f"{OUT}/scans.npy", scans, allow_pickle=True)

figure, axes = plt.subplots(2, 3, figsize=(14, 7.5))
for ax, (knob, data) in zip(axes.ravel(), scans.items()):
    ax.plot(data[:, 0], data[:, 1] * 1e6, "o-", color="#1f77b4", ms=3, label="sigma_x")
    ax.plot(data[:, 0], data[:, 2] * 1e6, "s-", color="#2ca02c", ms=3, label="sigma_y")
    ax.axhline(100, color="#7c3aed", ls="--", lw=0.9, label="asked (100 um)")
    ax.set_yscale("log"); ax.set_title(knob.replace("cla_", ""), fontsize=9)
    ax.set_ylabel("sigma (um)", fontsize=8)
    twin = ax.twinx()
    twin.plot(data[:, 0], data[:, 4] * 100, ":", color="#d62728", lw=1.2)
    twin.set_ylabel("sigma_delta (%)", color="#d62728", fontsize=8)
    twin.tick_params(axis="y", labelcolor="#d62728", labelsize=7)
    ax.tick_params(labelsize=7)
axes[0, 0].legend(fontsize=7, frameon=False)
figure.suptitle("One knob at a time, everything else at the lattice-file tune "
                "(dotted red: energy spread, a diagnosis the agent cannot see)",
                fontsize=10)
figure.tight_layout(rect=(0, 0, 1, 0.96))
figure.savefig(f"{OUT}/02_scans.png", dpi=110); plt.close(figure)
print(f"shots so far: {shots[0]}")

# ==========================================================================
# stage C
# ==========================================================================
import numpy as np
import matplotlib.pyplot as plt

WANT = np.array(TARGET)
trace = []
_shot = shot
def shot(action=None):                      # every shot, recorded
    i = _shot(action)
    trace.append((shots[0], i["sigma_x"], i["sigma_y"], i["mu_x"], i["mu_y"],
                  i["benchmark_score"], bool(i["on_screen"])))
    return i
globals()["shot"] = shot

QUADS = [n for n in env.magnet_names
         if env._knobs[n].prop == "k1"
         and n.split("_")[1] in ("s04", "s05", "s06", "s07", "fea", "feh")]
KNOBS = QUADS + ["cla_vbc_chicane"] + [n for n in env.magnet_names if "lin_cav" in n]
SCALE = np.array([{"k1": 0.3, "angle": 0.004, "phase": 1.0}[env._knobs[n].prop]
                  for n in KNOBS])
STEER = ["cla_feh_mag_hvcor_01_h", "cla_feh_mag_hvcor_01_v",
         "cla_fea_mag_hvcor_05_h", "cla_fea_mag_hvcor_05_v"]
print(f"working set: {len(QUADS)} quadrupoles + chicane + 4 phases = {len(KNOBS)}")

def residual(i):
    if not i["on_screen"]:
        return None
    return np.array([np.log(i["sigma_x"] / WANT[2]), np.log(i["sigma_y"] / WANT[3])])

def cost(r):
    return np.inf if r is None else float(r @ r)

def settings():
    return np.array([env.magnet_settings()[n] for n in KNOBS])

def apply(values):
    return shot({n: float(np.clip(v, *env.magnet_limits[n]))
                 for n, v in zip(KNOBS, values)})

# ---- a start: where do the last two quadrupoles put both planes?
grid = np.linspace(-11, 11, 15)
surface = np.full((len(grid), len(grid)), np.nan)
best_start, best_value = None, np.inf
for a, k1 in enumerate(grid):
    for b, k2 in enumerate(grid):
        i = shot({"cla_feh_mag_quad_01": float(k1), "cla_feh_mag_quad_02": float(k2)})
        r = residual(i)
        if r is not None:
            surface[a, b] = cost(r)
            if surface[a, b] < best_value:
                best_start, best_value = (float(k1), float(k2)), surface[a, b]
print(f"2-D scan best at FEH quads {best_start}, cost {best_value:.2f}")
figure, ax = plt.subplots(figsize=(6.4, 5.2))
mesh = ax.pcolormesh(grid, grid, np.log10(surface.T), cmap="viridis", shading="nearest")
ax.plot(*best_start, "*", color="#f97316", ms=16, mec="k", mew=0.6)
ax.set(xlabel="FEH-QUAD-01 k1 (1/m^2)", ylabel="FEH-QUAD-02 k1 (1/m^2)",
       title="Distance from a 100 x 100 um spot, final doublet\n"
             "(white: no beam on the screen)")
figure.colorbar(mesh, label="log10 cost")
figure.tight_layout(); figure.savefig(f"{OUT}/03_doublet_scan.png", dpi=110)
plt.close(figure)

shot({"cla_feh_mag_quad_01": best_start[0], "cla_feh_mag_quad_02": best_start[1]})

# ---- match
def jacobian(reference):
    base, columns, previous = settings(), [], None
    for n, knob in enumerate(KNOBS):
        action = {knob: float(base[n] + SCALE[n])}
        if previous is not None:
            action[previous] = float(base[n - 1])
        r = residual(shot(action))
        columns.append((r - reference) / SCALE[n] if r is not None else np.zeros(2))
        previous = knob
    shot({previous: float(base[-1])})
    return np.array(columns).T * SCALE           # per scaled unit

def match(rounds, cap=2.0, label=""):
    best, best_cost = settings(), cost(residual(shot()))
    lam, stalls = 0.3, 0
    J = jacobian(residual(shot()))
    for iteration in range(rounds):
        reference = residual(shot())
        if reference is None:
            apply(best); reference = residual(shot())
        move = -np.linalg.solve(J.T @ J + lam * np.eye(len(KNOBS)), J.T @ reference)
        move *= min(1.0, cap / max(np.abs(move).max(), 1e-12))
        taken = apply(settings() + move * SCALE)
        if cost(residual(taken)) > best_cost:
            taken = apply(best + move * SCALE / 4)
        if cost(residual(taken)) < best_cost:
            best, best_cost, lam, stalls = settings(), cost(residual(taken)), \
                max(lam / 2, 0.02), 0
        else:
            lam, stalls = lam * 3, stalls + 1
            apply(best)
            if stalls >= 2:
                J = jacobian(residual(shot())); lam, stalls = 0.2, 0
        if iteration % 4 == 0 or iteration == rounds - 1:
            i = shot()
            print(f"  {label}{iteration:3d}: sigma ({i['sigma_x']*1e6:7.1f}, "
                  f"{i['sigma_y']*1e6:7.1f}) um  cost {best_cost:7.3f}  "
                  f"shots {shots[0]}")
    apply(best)
    return best_cost

def steer():
    for _ in range(3):
        origin = np.array([env.magnet_settings()[k] for k in STEER])
        here = shot()
        if not here["on_screen"]:
            return
        offset = np.array([here["mu_x"], here["mu_y"]]) - WANT[:2]
        columns, previous = [], None
        for n, name in enumerate(STEER):
            action = {name: float(origin[n] + 2e-4)}
            if previous is not None:
                action[previous] = float(origin[n - 1])
            probed = shot(action)
            columns.append((np.array([probed["mu_x"], probed["mu_y"]])
                            - np.array([here["mu_x"], here["mu_y"]])) / 2e-4)
            previous = name
        S = np.array(columns).T
        shot({k: float(np.clip(v, *env.magnet_limits[k])) for k, v in
              zip(STEER, origin - np.linalg.pinv(S, rcond=1e-8) @ offset)})

print("\nmatching")
match(16, label="match ")
print("steering"); steer()
print("rematching")
match(12, cap=1.0, label="match ")
steer()
final = shot()
print(f"\nFINAL sigma ({final['sigma_x']*1e6:.1f}, {final['sigma_y']*1e6:.1f}) um  "
      f"mu ({final['mu_x']*1e6:+.1f}, {final['mu_y']*1e6:+.1f}) um  "
      f"score {final['benchmark_score']:.3f}  rel err "
      f"{final['scored_relative_error']:.3f}  shots {shots[0]}")
print("diagnosis:", {k: round(float(v), 6) for k, v in diagnose().items()})
np.save(f"{OUT}/trace.npy", np.array(trace, dtype=object), allow_pickle=True)
globals()["trace"] = trace

# ==========================================================================
# stage D
# ==========================================================================
import numpy as np
import matplotlib.pyplot as plt

CHIRP = ["cla_vbc_chicane", "cla_l02_lin_cav_01", "cla_l03_lin_cav_01",
         "cla_l4h_lin_cav_01", "cla_l04_lin_cav_01"]

# The energy spread is not something a screen tells you, so an agent could not do
# this step from its observation. A reference tune is designed with the model in
# hand, and this is the design step: pick the chirp, then match the optics to it.
history = [(0, diagnose()["sigma_delta"], None, None)]
print(f"energy spread before: {history[0][1]*100:.3f} %")
for sweep in range(3):
    for knob in CHIRP:
        low, high = env.magnet_limits[knob]
        start = env.magnet_settings()[knob]
        span = (high - low) / (2 if sweep == 0 else 6)
        values = np.linspace(max(low, start - span), min(high, start + span), 11)
        best, best_spread = start, np.inf
        for value in values:
            shot({knob: float(value)})
            spread = diagnose()["sigma_delta"]
            if spread < best_spread:
                best, best_spread = float(value), spread
        shot({knob: best})
        history.append((shots[0], best_spread, knob, best))
    print(f"  sweep {sweep}: sigma_delta {history[-1][1]*100:.3f} %, "
          f"shots {shots[0]}")
chirp_setting = {k: round(env.magnet_settings()[k], 4) for k in CHIRP}
print("chirp knobs:", chirp_setting)
i = shot()
print(f"after the chirp step: sigma ({i['sigma_x']*1e6:.1f}, {i['sigma_y']*1e6:.1f}) um, "
      f"energy spread {diagnose()['sigma_delta']*100:.3f} %")

figure, ax = plt.subplots(figsize=(7, 4))
ax.plot([h[0] for h in history], [h[1] * 100 for h in history], "o-", color="#d62728")
for h in history[1:]:
    ax.annotate(h[2].replace("cla_", "").replace("_lin_cav_01", "").replace("_", " "),
                (h[0], h[1] * 100), fontsize=6, rotation=35,
                textcoords="offset points", xytext=(2, 4))
ax.set(xlabel="shot", ylabel="energy spread at the readout (%)",
       title="Setting the chirp: VBC and the four RF phases, swept in turn")
figure.tight_layout(); figure.savefig(f"{OUT}/04_energy_spread.png", dpi=110)
plt.close(figure)

# ---- rematch the optics, now with escape attempts when it stalls
QUADS_ONLY = [n for n in KNOBS if env._knobs[n].prop == "k1"]
def rematch(rounds, cap=2.0):
    global KNOBS, SCALE
    KNOBS, SCALE = QUADS_ONLY, np.full(len(QUADS_ONLY), 0.3)
    globals().update(KNOBS=KNOBS, SCALE=SCALE)
    best, best_cost = settings(), cost(residual(shot()))
    lam, stalls = 0.3, 0
    J = jacobian(residual(shot()))
    for iteration in range(rounds):
        reference = residual(shot())
        if reference is None:
            apply(best); reference = residual(shot())
        move = -np.linalg.solve(J.T @ J + lam * np.eye(len(KNOBS)), J.T @ reference)
        move *= min(1.0, cap / max(np.abs(move).max(), 1e-12))
        taken = apply(settings() + move * SCALE)
        if cost(residual(taken)) > best_cost:
            taken = apply(best + move * SCALE / 4)
        if cost(residual(taken)) < best_cost:
            best, best_cost, lam, stalls = settings(), cost(residual(taken)), \
                max(lam / 2, 0.02), 0
        else:
            lam, stalls = lam * 3, stalls + 1
            apply(best)
            if stalls == 2:
                J = jacobian(residual(shot())); lam = 0.2
            elif stalls >= 4:            # shake it out of the hole and re-measure
                apply(best + env._rng.normal(0, 0.5, len(KNOBS)) * SCALE)
                if cost(residual(shot())) is np.inf:
                    apply(best)
                J = jacobian(residual(shot()) if residual(shot()) is not None
                             else reference)
                lam, stalls = 0.3, 0
        if iteration % 5 == 0 or iteration == rounds - 1:
            i = shot()
            print(f"  rematch {iteration:3d}: sigma ({i['sigma_x']*1e6:7.1f}, "
                  f"{i['sigma_y']*1e6:7.1f}) um  cost {best_cost:7.3f}  shots {shots[0]}")
    apply(best)

print("\nrematching the optics against the new chirp")
rematch(24)
steer()
rematch(14, cap=0.8)
steer()
final = shot()
print(f"\nFINAL sigma ({final['sigma_x']*1e6:.1f}, {final['sigma_y']*1e6:.1f}) um  "
      f"mu ({final['mu_x']*1e6:+.1f}, {final['mu_y']*1e6:+.1f}) um  "
      f"score {final['benchmark_score']:.3f}  rel err {final['scored_relative_error']:.3f}  "
      f"on chip {final['on_screen_fraction']:.1%}  shots {shots[0]}")
print("diagnosis:", {k: round(float(v), 6) for k, v in diagnose().items()})
np.save(f"{OUT}/trace.npy", np.array(trace, dtype=object), allow_pickle=True)

# ==========================================================================
# stage E
# ==========================================================================
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

FOCUS = ["cla_fea_mag_quad_11", "cla_fea_mag_quad_12", "cla_fea_mag_quad_13",
         "cla_fea_mag_quad_14", "cla_fea_mag_quad_15", "cla_fea_mag_quad_16",
         "cla_feh_mag_quad_01", "cla_feh_mag_quad_02"]
LOW = np.array([env.magnet_limits[n][0] for n in FOCUS])
HIGH = np.array([env.magnet_limits[n][1] for n in FOCUS])

def objective(values):
    i = shot({n: float(np.clip(v, lo, hi))
              for n, v, lo, hi in zip(FOCUS, values, LOW, HIGH)})
    if not i["on_screen"]:
        return 60.0
    r = np.array([np.log(i["sigma_x"] / WANT[2]), np.log(i["sigma_y"] / WANT[3])])
    return float(r @ r)

here = np.array([env.magnet_settings()[n] for n in FOCUS])
best, best_cost = here.copy(), objective(here)
print(f"starting from cost {best_cost:.3f}, {len(FOCUS)} final-focus quadrupoles")
rng = np.random.default_rng(0)
starts = [here] + [np.clip(here + rng.normal(0, 4.0, len(FOCUS)), LOW, HIGH)
                   for _ in range(7)]
for n, start in enumerate(starts):
    result = minimize(objective, start, method="Nelder-Mead",
                      options=dict(maxfev=320, xatol=1e-3, fatol=1e-4))
    if result.fun < best_cost:
        best, best_cost = np.clip(result.x, LOW, HIGH), float(result.fun)
    print(f"  start {n}: cost {result.fun:7.3f}  best so far {best_cost:7.3f}  "
          f"shots {shots[0]}")

objective(best)
i = shot()
print(f"after the search: sigma ({i['sigma_x']*1e6:.1f}, {i['sigma_y']*1e6:.1f}) um")

# polish with the full quadrupole set, then steer, then polish again
globals().update(KNOBS=[n for n in env.magnet_names if env._knobs[n].prop == "k1"
                        and n.split("_")[1] in ("s07", "fea", "feh")])
globals().update(SCALE=np.full(len(KNOBS), 0.15))
print(f"polishing with {len(KNOBS)} quadrupoles")
for round_ in range(3):
    reference = residual(shot())
    J = jacobian(reference)
    lam = 0.2
    best_local, best_local_cost = settings(), cost(reference)
    for iteration in range(8):
        reference = residual(shot())
        if reference is None:
            apply(best_local); reference = residual(shot())
        move = -np.linalg.solve(J.T @ J + lam * np.eye(len(KNOBS)), J.T @ reference)
        move *= min(1.0, 0.5 / max(np.abs(move).max(), 1e-12))
        taken = apply(settings() + move * SCALE)
        if cost(residual(taken)) < best_local_cost:
            best_local, best_local_cost = settings(), cost(residual(taken))
            lam = max(lam / 2, 0.02)
        else:
            lam *= 3
            apply(best_local)
    apply(best_local)
    steer()
    i = shot()
    print(f"  round {round_}: sigma ({i['sigma_x']*1e6:7.1f}, {i['sigma_y']*1e6:7.1f}) um"
          f"  mu ({i['mu_x']*1e6:+6.1f}, {i['mu_y']*1e6:+6.1f}) um"
          f"  score {i['benchmark_score']:.3f}  shots {shots[0]}")

final = shot()
print(f"\nFINAL sigma ({final['sigma_x']*1e6:.1f}, {final['sigma_y']*1e6:.1f}) um  "
      f"mu ({final['mu_x']*1e6:+.1f}, {final['mu_y']*1e6:+.1f}) um  "
      f"score {final['benchmark_score']:.3f}  rel err {final['scored_relative_error']:.3f}  "
      f"on chip {final['on_screen_fraction']:.1%}  shots {shots[0]}")
print("diagnosis:", {k: round(float(v), 6) for k, v in diagnose().items()})
np.save(f"{OUT}/trace.npy", np.array(trace, dtype=object), allow_pickle=True)

# ==========================================================================
# stage F
# ==========================================================================
import json, datetime
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import score

FOCUS2 = ["cla_fea_mag_quad_11", "cla_fea_mag_quad_12", "cla_fea_mag_quad_13",
          "cla_fea_mag_quad_14", "cla_fea_mag_quad_15", "cla_fea_mag_quad_16",
          "cla_feh_mag_quad_01", "cla_feh_mag_quad_02",
          "cla_s07_mag_quad_09", "cla_s07_mag_quad_10"]
LOW2 = np.array([env.magnet_limits[n][0] for n in FOCUS2])
HIGH2 = np.array([env.magnet_limits[n][1] for n in FOCUS2])

def objective2(values):
    i = shot({n: float(np.clip(v, lo, hi))
              for n, v, lo, hi in zip(FOCUS2, values, LOW2, HIGH2)})
    if not i["on_screen"]:
        return 60.0
    r = np.array([np.log(i["sigma_x"] / WANT[2]), np.log(i["sigma_y"] / WANT[3])])
    return float(r @ r)

here = np.array([env.magnet_settings()[n] for n in FOCUS2])
best, best_cost = here.copy(), objective2(here)
print(f"squeezing from cost {best_cost:.4f}")
for attempt, spread in enumerate((0.0, 0.3, 0.8)):
    start = here if spread == 0 else np.clip(
        best + np.random.default_rng(attempt).normal(0, spread, len(FOCUS2)), LOW2, HIGH2)
    result = minimize(objective2, start, method="Nelder-Mead",
                      options=dict(maxfev=600, xatol=1e-4, fatol=1e-6))
    if result.fun < best_cost:
        best, best_cost = np.clip(result.x, LOW2, HIGH2), float(result.fun)
    print(f"  attempt {attempt}: cost {result.fun:.4f}, best {best_cost:.4f}, "
          f"shots {shots[0]}")
objective2(best)
steer()
final = shot()
print(f"\nFINAL sigma ({final['sigma_x']*1e6:.1f}, {final['sigma_y']*1e6:.1f}) um  "
      f"mu ({final['mu_x']*1e6:+.2f}, {final['mu_y']*1e6:+.2f}) um  "
      f"rel err {final['scored_relative_error']:.4f}  "
      f"on chip {final['on_screen_fraction']:.1%}  shots {shots[0]}")
d = diagnose()
print("diagnosis:", {k: round(float(v), 6) for k, v in d.items()})

# ---------- the tune, in score.py's own format
tune = score.read_tune(env.segment)
nominal = {n: env._nominal_strengths[n] for n in env.magnet_names}
changed = {n: v for n, v in env.magnet_settings().items()
           if abs(v - nominal[n]) > 1e-9}
payload = {
    "_provenance": {
        "source": "RL_framework SimEnv, cheetah backend, nominal model "
                  "(cla_s02_sim_aper_01 to cla_feh_dia_scr_01)",
        "built_by": "analysis/febe_tune/tune_febe_100um.py",
        "built": datetime.date.today().isoformat(),
        "request": "sigma_x = sigma_y = 100 um, centred, at CLA-FEH-DIA-SCR-01",
        "achieved": {
            "sigma_x_um": round(final["sigma_x"] * 1e6, 2),
            "sigma_y_um": round(final["sigma_y"] * 1e6, 2),
            "mu_x_um": round(final["mu_x"] * 1e6, 3),
            "mu_y_um": round(final["mu_y"] * 1e6, 3),
            "relative_error": round(final["scored_relative_error"], 4),
            "fraction_on_chip": round(final["on_screen_fraction"], 4),
            "energy_spread_percent": round(float(d["sigma_delta"]) * 100, 4),
        },
        "shots": shots[0],
        "machine": "no beam variation, no misalignment, no jitter, no ripple — "
                   "the nominal model exactly, so the tune is reproducible",
        "note": "Quadrupoles, the VBC chicane and the four RF phases were all used. "
                "The chirp knobs were set to minimise the energy spread, which a "
                "screen does not measure: that step used the model, not the "
                "environment's observation.",
        "knobs_moved": len(changed),
    },
    "elements": tune,
}
json.dump(payload, open(f"{OUT}/reference_tune_febe_100um.json", "w"), indent=1)
print(f"\nwrote the tune: {len(tune)} elements, {len(changed)} knobs moved from nominal")

# ---------- plots
image_final = env._readings["image"].copy()
chip = np.array(env.screen_extent()) * 1e3
env._restore_nominal(); env._simulate()
baseline = env._readings["image"].copy()
base_stats = env._readings["stats"]
score.apply_tune(env.segment, tune)       # put the tune back
env._simulate()
check = env._readings["stats"]
print(f"tune reapplied from the dict: sigma ({check[2]*1e6:.1f}, {check[3]*1e6:.1f}) um "
      f"— matches: {abs(check[2]-final['sigma_x'])<1e-9 and abs(check[3]-final['sigma_y'])<1e-9}")

figure, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, image, title in (
        (axes[0], baseline, f"lattice-file tune\n{base_stats[2]*1e6:.0f} x "
                            f"{base_stats[3]*1e6:.0f} um"),
        (axes[1], image_final, f"tuned\n{final['sigma_x']*1e6:.1f} x "
                               f"{final['sigma_y']*1e6:.1f} um")):
    ax.imshow(image, extent=chip, origin="lower", cmap="Blues", aspect="equal")
    ax.plot(0, 0, "+", color="#f97316", ms=10, mew=1.2)
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title=title)
axes[2].imshow(image_final, extent=chip, origin="lower", cmap="Blues", aspect="equal")
axes[2].plot(0, 0, "+", color="#f97316", ms=12, mew=1.4)
axes[2].set(xlim=(-0.6, 0.6), ylim=(-0.6, 0.6), xlabel="x (mm)", ylabel="y (mm)",
            title="tuned, zoomed to +-0.6 mm")
figure.suptitle("CLA-FEH-DIA-SCR-01 before and after", fontsize=12)
figure.tight_layout(); figure.savefig(f"{OUT}/05_before_after.png", dpi=110)
plt.close(figure)

t = np.array([[r[0], r[1], r[2], r[5]] for r in trace], dtype=float)
on = np.array([r[6] for r in trace], dtype=bool)
figure, (sizes, scores) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
sizes.plot(t[on, 0], t[on, 1] * 1e6, ".", ms=1.4, color="#1f77b4", label="sigma_x")
sizes.plot(t[on, 0], t[on, 2] * 1e6, ".", ms=1.4, color="#2ca02c", label="sigma_y")
sizes.axhline(100, color="#7c3aed", ls="--", lw=1, label="asked (100 um)")
sizes.set(yscale="log", ylabel="sigma (um)")
sizes.legend(fontsize=8, frameon=False, ncol=3)
running = np.minimum.accumulate(np.where(on, t[:, 3], np.inf))
scores.plot(t[:, 0], running, color="#d62728")
scores.set(xlabel="shot", ylabel="best benchmark score so far")
for boundary, label in ((113, "sensitivity"), (197, "scans"), (475, "doublet scan"),
                        (1240, "match"), (1420, "chirp"), (2444, "rematch"),
                        (4351, "search")):
    for ax in (sizes, scores):
        ax.axvline(boundary, color="0.75", lw=0.8)
    sizes.annotate(label, (boundary, sizes.get_ylim()[1]), fontsize=6.5,
                   rotation=90, va="top", ha="right", color="0.4")
figure.suptitle("Every shot of the tune", fontsize=12)
figure.tight_layout(); figure.savefig(f"{OUT}/06_convergence.png", dpi=110)
plt.close(figure)

moved = sorted(changed.items(), key=lambda kv: -abs(kv[1] - nominal[kv[0]]))[:26]
figure, ax = plt.subplots(figsize=(7.5, 7))
y = range(len(moved))
ax.barh(y, [v - nominal[n] for n, v in moved], color="#1f77b4")
ax.set_yticks(y)
ax.set_yticklabels([f"{n.replace('cla_','').replace('_mag_',' ')} "
                    f"({nominal[n]:+.2f} -> {v:+.2f})" for n, v in moved], fontsize=6.5)
ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.7)
ax.set(xlabel="change from the lattice-file value (native units)",
       title="What moved, biggest change first")
figure.tight_layout(); figure.savefig(f"{OUT}/07_knobs_moved.png", dpi=110)
plt.close(figure)
print("plots written:", sorted(f for f in __import__("os").listdir(OUT)
                               if f.endswith(".png")))

# ==========================================================================
# stage G
# ==========================================================================
import json, numpy as np, torch, cheetah
import matplotlib.pyplot as plt
import score

i = shot()
print("final:", {k: (round(v, 6) if isinstance(v, float) else v)
                 for k, v in i.items()
                 if k in ("transmission", "on_screen_fraction", "evaluations",
                          "scored_mismatch", "benchmark_score")})

def envelope(label):
    """sigma_x, sigma_y and the surviving fraction at every diagnostic."""
    coords = env._coordinates
    beam = cheetah.ParticleBeam(
        particles=torch.tensor(np.column_stack([coords, np.ones(len(coords))]),
                               dtype=torch.float32),
        energy=torch.tensor(float(env.energy_ev), dtype=torch.float32))
    live = np.ones(len(coords), bool)
    s, out = 0.0, []
    for stop in env._stops:
        chunk = env.transport._chunks[stop.name]
        if len(chunk.elements):
            beam = chunk.track(beam)
            s += float(sum(float(e.length) for e in chunk.elements))
        x, y = beam.x.detach().numpy(), beam.y.detach().numpy()
        live &= x**2 + y**2 <= env.config.beam_pipe_radius**2
        if live.sum() > 10:
            out.append((s, x[live].std(), y[live].std(), live.mean()))
        if stop.name == env.target_screen:
            break
    return np.array(out)

tuned = envelope("tuned")
tune = score.read_tune(env.segment)
env._restore_nominal(); env._simulate()
nominal_env = envelope("nominal")
score.apply_tune(env.segment, tune); env._simulate()

figure, (ax, frac) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                  gridspec_kw=dict(height_ratios=(2, 1)))
for data, style, label in ((nominal_env, ":", "lattice-file tune"),
                           (tuned, "-", "tuned")):
    ax.plot(data[:, 0], data[:, 1] * 1e3, style, color="#1f77b4",
            label=f"sigma_x, {label}")
    ax.plot(data[:, 0], data[:, 2] * 1e3, style, color="#2ca02c",
            label=f"sigma_y, {label}")
    frac.plot(data[:, 0], data[:, 3] * 100, style, color="#d62728", label=label)
ax.set(yscale="log", ylabel="sigma (mm)",
       title="Beam envelope from S02 to the FEBE readout")
ax.legend(fontsize=7, frameon=False, ncol=2)
frac.set(xlabel="s from cla_s02_sim_aper_01 (m)", ylabel="inside the pipe (%)")
frac.legend(fontsize=7, frameon=False)
figure.tight_layout(); figure.savefig(f"{OUT}/08_envelope.png", dpi=110); plt.close(figure)
print("envelope written; transmission along the line ends at",
      f"{tuned[-1, 3]*100:.1f}% (nominal {nominal_env[-1, 3]*100:.1f}%)")

# ==========================================================================
# stage H
# ==========================================================================
import numpy as np, torch, cheetah
import matplotlib.pyplot as plt
coords = env._coordinates
beam = cheetah.ParticleBeam(
    particles=torch.tensor(np.column_stack([coords, np.ones(len(coords))]),
                           dtype=torch.float32),
    energy=torch.tensor(float(env.energy_ev), dtype=torch.float32))
live = np.ones(len(coords), bool)
for stop in env._stops:
    chunk = env.transport._chunks[stop.name]
    if len(chunk.elements):
        beam = chunk.track(beam)
    x, y = beam.x.detach().numpy(), beam.y.detach().numpy()
    live &= x**2 + y**2 <= env.config.beam_pipe_radius**2
    if stop.name == env.target_screen:
        break
x, y = x[live] * 1e6, y[live] * 1e6
figure, (fine, prof) = plt.subplots(1, 2, figsize=(10, 4.2))
edges = np.linspace(-500, 500, 161)
fine.hist2d(x, y, bins=(edges, edges), cmap="Blues")
fine.plot(0, 0, "+", color="#f97316", ms=11, mew=1.3)
for n in (1, 2):
    fine.add_patch(plt.Circle((0, 0), 100 * n, fill=False, ec="#7c3aed",
                              ls="--", lw=0.9))
fine.set(xlabel="x (um)", ylabel="y (um)", aspect="equal",
         title=f"{len(x)} particles at the readout\n"
               f"dashed: the 100 um requested sigma, 1 and 2 sigma")
for values, colour, label in ((x, "#1f77b4", "x"), (y, "#2ca02c", "y")):
    prof.hist(values, bins=edges, histtype="step", color=colour,
              label=f"{label}, sigma {values.std():.1f} um, "
                    f"mean {values.mean():+.2f} um")
prof.set(xlabel="position (um)", ylabel="particles", title="projections")
prof.legend(fontsize=8, frameon=False)
figure.tight_layout(); figure.savefig(f"{OUT}/09_spot_detail.png", dpi=120)
plt.close(figure)
print(f"core: sigma ({x.std():.2f}, {y.std():.2f}) um, mean "
      f"({x.mean():+.3f}, {y.mean():+.3f}) um, {live.mean():.1%} of the launched beam")
print(f"tails: 1 sigma {np.mean((x**2+y**2) < 100**2):.1%}, "
      f"within 500 um {np.mean((np.abs(x) < 500) & (np.abs(y) < 500)):.1%}")

# ==========================================================================
# stage I — transmission added to the score, rematch
# ==========================================================================
import json, datetime
import numpy as np
from scipy.optimize import minimize
from rl_framework.envs.sim_env import MAX_MISMATCH, evaluation_cost

OUT = "/home/jovyan/work/febe_tune"
WANT = np.array([0.0, 0.0, 100e-6, 100e-6])
trace, shots = [], [0]

def shot(action=None):
    shots[0] += 1
    i = env.step(action or {})[-1]
    trace.append((shots[0], i["sigma_x"], i["sigma_y"], i["scored_transmission"],
                  i["scored_mismatch"], bool(i["on_screen"])))
    return i

def mismatch(i):
    """The score itself, uncapped so there is a gradient out at distance."""
    return env._mismatch((i["mu_x"], i["mu_y"], i["sigma_x"], i["sigma_y"],
                          i["on_screen"]), i["transmission"], cap=MAX_MISMATCH)

def residual(i):
    """Three terms the match can work on: both sizes, and the beam that is lost."""
    if not i["on_screen"]:
        return None
    return np.array([np.log(i["sigma_x"] / WANT[2]), np.log(i["sigma_y"] / WANT[3]),
                     np.sqrt(2.0) * (1.0 - i["transmission"])])

QUADS = [n for n in env.magnet_names if env._knobs[n].prop == "k1"
         and n.split("_")[1] in ("s07", "fea", "feh")]
STEER = ["cla_feh_mag_hvcor_01_h", "cla_feh_mag_hvcor_01_v",
         "cla_fea_mag_hvcor_05_h", "cla_fea_mag_hvcor_05_v"]
SCALE = np.full(len(QUADS), 0.25)
start = shot()
print(f"starting from sigma ({start['sigma_x']*1e6:.1f}, {start['sigma_y']*1e6:.1f}) um, "
      f"transmission {start['transmission']:.1%}, mismatch {mismatch(start):.4f}")
print(f"{len(QUADS)} quadrupoles in the match")

def settings():
    return np.array([env.magnet_settings()[n] for n in QUADS])

def apply(values):
    return shot({n: float(np.clip(v, *env.magnet_limits[n]))
                 for n, v in zip(QUADS, values)})

def jacobian(reference):
    base, columns, previous = settings(), [], None
    for n, knob in enumerate(QUADS):
        action = {knob: float(base[n] + SCALE[n])}
        if previous is not None:
            action[previous] = float(base[n - 1])
        r = residual(shot(action))
        columns.append((r - reference) / SCALE[n] if r is not None else np.zeros(3))
        previous = knob
    shot({previous: float(base[-1])})
    return np.array(columns).T * SCALE

best, best_cost = settings(), mismatch(start)
lam = 0.3
for iteration in range(18):
    here = shot()
    reference = residual(here)
    if reference is None:
        apply(best); reference = residual(shot())
    if iteration % 6 == 0:
        J = jacobian(reference)
    move = -np.linalg.solve(J.T @ J + lam * np.eye(len(QUADS)), J.T @ reference)
    move *= min(1.0, 1.0 / max(np.abs(move).max(), 1e-12))
    taken = apply(settings() + move * SCALE)
    if mismatch(taken) < best_cost:
        best, best_cost, lam = settings(), mismatch(taken), max(lam / 2, 0.02)
    else:
        lam *= 3
        apply(best)
        if lam > 50:
            J = jacobian(residual(shot())); lam = 0.3
    if iteration % 3 == 0:
        i = shot()
        print(f"  lsq {iteration:3d}: sigma ({i['sigma_x']*1e6:6.1f}, "
              f"{i['sigma_y']*1e6:6.1f}) um  T {i['transmission']:5.1%}  "
              f"mismatch {best_cost:7.4f}  shots {shots[0]}")
apply(best)

FOCUS = ["cla_fea_mag_quad_09", "cla_fea_mag_quad_10", "cla_fea_mag_quad_11",
         "cla_fea_mag_quad_12", "cla_fea_mag_quad_13", "cla_fea_mag_quad_14",
         "cla_fea_mag_quad_15", "cla_fea_mag_quad_16",
         "cla_feh_mag_quad_01", "cla_feh_mag_quad_02"]
LOW = np.array([env.magnet_limits[n][0] for n in FOCUS])
HIGH = np.array([env.magnet_limits[n][1] for n in FOCUS])

def objective(values):
    i = shot({n: float(np.clip(v, lo, hi))
              for n, v, lo, hi in zip(FOCUS, values, LOW, HIGH)})
    return 60.0 if not i["on_screen"] else mismatch(i)

print("\nsearch")
here = np.array([env.magnet_settings()[n] for n in FOCUS])
best_f, best_cost = here.copy(), objective(here)
for attempt, spread in enumerate((0.0, 0.5, 1.5, 3.0)):
    seed = here if spread == 0 else np.clip(
        best_f + np.random.default_rng(attempt).normal(0, spread, len(FOCUS)), LOW, HIGH)
    result = minimize(objective, seed, method="Nelder-Mead",
                      options=dict(maxfev=700, xatol=1e-4, fatol=1e-6))
    if result.fun < best_cost:
        best_f, best_cost = np.clip(result.x, LOW, HIGH), float(result.fun)
    print(f"  attempt {attempt}: {result.fun:.4f}  best {best_cost:.4f}  shots {shots[0]}")
objective(best_f)

for _ in range(2):
    origin = np.array([env.magnet_settings()[k] for k in STEER])
    here_i = shot()
    offset = np.array([here_i["mu_x"], here_i["mu_y"]]) - WANT[:2]
    columns, previous = [], None
    for n, name in enumerate(STEER):
        action = {name: float(origin[n] + 2e-4)}
        if previous is not None:
            action[previous] = float(origin[n - 1])
        p = shot(action)
        columns.append((np.array([p["mu_x"], p["mu_y"]])
                        - np.array([here_i["mu_x"], here_i["mu_y"]])) / 2e-4)
        previous = name
    shot({k: float(np.clip(v, *env.magnet_limits[k])) for k, v in
          zip(STEER, origin - np.linalg.pinv(np.array(columns).T, rcond=1e-8) @ offset)})

final = shot()
print(f"\nFINAL sigma ({final['sigma_x']*1e6:.1f}, {final['sigma_y']*1e6:.1f}) um  "
      f"mu ({final['mu_x']*1e6:+.2f}, {final['mu_y']*1e6:+.2f}) um  "
      f"transmission {final['transmission']:.1%}  mismatch {final['scored_mismatch']:.4f}  "
      f"shots {shots[0]}")
np.save(f"{OUT}/trace_t.npy", np.array(trace, dtype=object), allow_pickle=True)
globals().update(trace_t=trace, shots_t=shots, final_t=final)

# ==========================================================================
# stage J — grouped searches for transmission
# ==========================================================================
import numpy as np
from scipy.optimize import minimize

GROUPS = {
    "arc exit + final focus": ["cla_fea_mag_quad_09", "cla_fea_mag_quad_10",
        "cla_fea_mag_quad_11", "cla_fea_mag_quad_12", "cla_fea_mag_quad_13",
        "cla_fea_mag_quad_14", "cla_fea_mag_quad_15", "cla_fea_mag_quad_16",
        "cla_feh_mag_quad_01", "cla_feh_mag_quad_02"],
    "arc entrance": ["cla_fea_mag_quad_01", "cla_fea_mag_quad_02",
        "cla_fea_mag_quad_03", "cla_fea_mag_quad_04", "cla_fea_mag_quad_05",
        "cla_fea_mag_quad_06", "cla_fea_mag_quad_07", "cla_fea_mag_quad_08"],
    "S07": [n for n in env.magnet_names if env._knobs[n].prop == "k1"
            and n.startswith("cla_s07")],
}
for name, knobs in GROUPS.items():
    print(f"{name}: {len(knobs)} knobs")

def run(knobs, spreads):
    low = np.array([env.magnet_limits[n][0] for n in knobs])
    high = np.array([env.magnet_limits[n][1] for n in knobs])
    def objective(values):
        i = shot({n: float(np.clip(v, lo, hi))
                  for n, v, lo, hi in zip(knobs, values, low, high)})
        return 60.0 if not i["on_screen"] else mismatch(i)
    here = np.array([env.magnet_settings()[n] for n in knobs])
    best, best_cost = here.copy(), objective(here)
    for attempt, spread in enumerate(spreads):
        seed = here if spread == 0 else np.clip(
            best + np.random.default_rng(100 + attempt).normal(0, spread, len(knobs)),
            low, high)
        result = minimize(objective, seed, method="Nelder-Mead",
                          options=dict(maxfev=650, xatol=1e-4, fatol=1e-7))
        if result.fun < best_cost:
            best, best_cost = np.clip(result.x, low, high), float(result.fun)
    objective(best)
    return best_cost

for round_ in range(2):
    for name, knobs in GROUPS.items():
        cost = run(knobs, (0.0, 0.4, 1.2))
        i = shot()
        print(f"  round {round_} {name:24s} mismatch {cost:7.4f}  "
              f"sigma ({i['sigma_x']*1e6:6.1f}, {i['sigma_y']*1e6:6.1f}) um  "
              f"T {i['transmission']:5.1%}  shots {shots[0]}")

for _ in range(2):
    origin = np.array([env.magnet_settings()[k] for k in STEER])
    h = shot()
    offset = np.array([h["mu_x"], h["mu_y"]]) - WANT[:2]
    columns, previous = [], None
    for n, nm in enumerate(STEER):
        action = {nm: float(origin[n] + 2e-4)}
        if previous is not None:
            action[previous] = float(origin[n - 1])
        p = shot(action)
        columns.append((np.array([p["mu_x"], p["mu_y"]])
                        - np.array([h["mu_x"], h["mu_y"]])) / 2e-4)
        previous = nm
    shot({k: float(np.clip(v, *env.magnet_limits[k])) for k, v in
          zip(STEER, origin - np.linalg.pinv(np.array(columns).T, rcond=1e-8) @ offset)})
final = shot()
print(f"\nFINAL sigma ({final['sigma_x']*1e6:.1f}, {final['sigma_y']*1e6:.1f}) um  "
      f"mu ({final['mu_x']*1e6:+.2f}, {final['mu_y']*1e6:+.2f}) um  "
      f"transmission {final['transmission']:.1%}  mismatch {final['scored_mismatch']:.4f}  "
      f"benchmark {final['benchmark_score']:.4f}  shots {shots[0]}")
np.save(f"{OUT}/trace_t.npy", np.array(trace, dtype=object), allow_pickle=True)
globals()["final_t"] = final

# ==========================================================================
# stage K — dump and redraw
# ==========================================================================
import json, datetime
import numpy as np, torch, cheetah
import matplotlib.pyplot as plt
import score
from rl_framework.envs.sim_env import evaluation_cost

final = shot()
d = diagnose() if "diagnose" in globals() else None

def losses():
    coords = env._coordinates
    beam = cheetah.ParticleBeam(
        particles=torch.tensor(np.column_stack([coords, np.ones(len(coords))]),
                               dtype=torch.float32),
        energy=torch.tensor(float(env.energy_ev), dtype=torch.float32))
    live = np.ones(len(coords), bool)
    s, out = 0.0, []
    for stop in env._stops:
        chunk = env.transport._chunks[stop.name]
        if len(chunk.elements):
            beam = chunk.track(beam)
            s += sum(float(e.length) for e in chunk.elements)
        x, y = beam.x.detach().numpy(), beam.y.detach().numpy()
        live &= x**2 + y**2 <= env.config.beam_pipe_radius**2
        out.append((s, x[live].std() if live.any() else np.nan,
                    y[live].std() if live.any() else np.nan, live.mean()))
        if stop.name == env.target_screen:
            break
    return np.array(out)

new_tune = score.read_tune(env.segment)
new_curve, new_image = losses(), env._readings["image"].copy()
chip = np.array(env.screen_extent()) * 1e3
old = json.load(open(f"{OUT}/reference_tune_febe_100um.json"))
score.apply_tune(env.segment, old["elements"]); env._simulate()
old_curve, old_image, old_stats = losses(), env._readings["image"].copy(), \
    env._readings["stats"]
env._restore_nominal(); env._simulate()
nom_curve, nom_image, nom_stats = losses(), env._readings["image"].copy(), \
    env._readings["stats"]
score.apply_tune(env.segment, new_tune); env._simulate()
check = env._readings["stats"]
print(f"reapplied: sigma ({check[2]*1e6:.1f}, {check[3]*1e6:.1f}) um, "
      f"transmission {env._readings['transmission']:.1%}")

payload = {
    "_provenance": {
        "source": "RL_framework SimEnv, cheetah backend, nominal model "
                  "(cla_s02_sim_aper_01 to cla_feh_dia_scr_01)",
        "built_by": "analysis/febe_tune/tune_febe_100um.py",
        "built": datetime.date.today().isoformat(),
        "request": "sigma_x = sigma_y = 100 um, centred, at CLA-FEH-DIA-SCR-01, "
                   "scored with score.TRANSMISSION_WEIGHT charging beam that is lost",
        "achieved": {
            "sigma_x_um": round(final["sigma_x"] * 1e6, 2),
            "sigma_y_um": round(final["sigma_y"] * 1e6, 2),
            "mu_x_um": round(final["mu_x"] * 1e6, 3),
            "mu_y_um": round(final["mu_y"] * 1e6, 3),
            "transmission": round(final["transmission"], 4),
            "mismatch": round(final["scored_mismatch"], 4),
            "relative_error": round(final["scored_relative_error"], 4),
        },
        "shots": int(shots[0]),
        "machine": "no beam variation, no misalignment, no jitter, no ripple",
        "note": "Quadrupoles, the VBC chicane and the four RF phases were all used. "
                "The chirp knobs were set to minimise the energy spread, which a "
                "screen does not measure: that step used the model, not the "
                "environment's observation.",
    },
    "elements": new_tune,
}
json.dump(payload, open(f"{OUT}/reference_tune_febe_100um.json", "w"), indent=1)
print("tune written")

figure, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, image, title in (
        (axes[0], nom_image, f"lattice file\n{nom_stats[2]*1e6:.0f} x "
                             f"{nom_stats[3]*1e6:.0f} um, {nom_curve[-1,3]:.0%} through"),
        (axes[1], old_image, f"spot only\n{old_stats[2]*1e6:.1f} x "
                             f"{old_stats[3]*1e6:.1f} um, {old_curve[-1,3]:.0%} through"),
        (axes[2], new_image, f"spot and transmission\n{final['sigma_x']*1e6:.1f} x "
                             f"{final['sigma_y']*1e6:.1f} um, "
                             f"{new_curve[-1,3]:.0%} through")):
    ax.imshow(image, extent=chip, origin="lower", cmap="Blues", aspect="equal")
    ax.plot(0, 0, "+", color="#f97316", ms=10, mew=1.2)
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title=title)
figure.suptitle("CLA-FEH-DIA-SCR-01: what the two objectives give", fontsize=12)
figure.tight_layout(); figure.savefig(f"{OUT}/05_before_after.png", dpi=110)
plt.close(figure)

figure, (env_ax, frac) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                      gridspec_kw=dict(height_ratios=(2, 1.2)))
for curve, style, label, colour in (
        (nom_curve, ":", "lattice file", "0.45"),
        (old_curve, "--", "spot only", "#1f77b4"),
        (new_curve, "-", "spot and transmission", "#d62728")):
    env_ax.plot(curve[:, 0], curve[:, 1] * 1e3, style, color=colour, lw=1.2,
                label=f"sigma_x, {label}")
    env_ax.plot(curve[:, 0], curve[:, 2] * 1e3, style, color=colour, lw=0.7, alpha=0.5)
    frac.plot(curve[:, 0], curve[:, 3] * 100, style, color=colour, lw=1.3, label=label)
env_ax.set(yscale="log", ylabel="sigma (mm)\n(thin: vertical)",
           title="Beam envelope and survival, S02 to the FEBE readout")
env_ax.legend(fontsize=7, frameon=False, ncol=3)
frac.set(xlabel="s from cla_s02_sim_aper_01 (m)", ylabel="still inside the pipe (%)")
frac.legend(fontsize=7, frameon=False); frac.set_ylim(0, 102)
figure.tight_layout(); figure.savefig(f"{OUT}/08_envelope.png", dpi=110); plt.close(figure)

t = np.array([[r[0], r[1], r[2], r[3], r[4]] for r in trace], dtype=float)
on = np.array([r[5] for r in trace], dtype=bool)
figure, (sizes, rest) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
sizes.plot(t[on, 0], t[on, 1] * 1e6, ".", ms=1.3, color="#1f77b4", label="sigma_x")
sizes.plot(t[on, 0], t[on, 2] * 1e6, ".", ms=1.3, color="#2ca02c", label="sigma_y")
sizes.axhline(100, color="#7c3aed", ls="--", lw=1, label="asked (100 um)")
sizes.set(yscale="log", ylabel="sigma (um)", ylim=(20, 3e4))
sizes.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
rest.plot(t[on, 0], t[on, 3] * 100, ".", ms=1.3, color="#d62728", label="transmission")
rest.plot(t[:, 0], np.minimum.accumulate(np.where(on, t[:, 4], np.inf)) * 100,
          color="k", lw=1, label="best mismatch so far (x100)")
rest.set(xlabel="shot", ylabel="%", ylim=(0, 125))
rest.legend(fontsize=8, frameon=False, ncol=2)
figure.suptitle("Redoing the match with transmission in the score", fontsize=12)
figure.tight_layout(); figure.savefig(f"{OUT}/10_transmission_retune.png", dpi=110)
plt.close(figure)

coords = env._coordinates
beam = cheetah.ParticleBeam(
    particles=torch.tensor(np.column_stack([coords, np.ones(len(coords))]),
                           dtype=torch.float32),
    energy=torch.tensor(float(env.energy_ev), dtype=torch.float32))
live = np.ones(len(coords), bool)
for stop in env._stops:
    chunk = env.transport._chunks[stop.name]
    if len(chunk.elements):
        beam = chunk.track(beam)
    x, y = beam.x.detach().numpy(), beam.y.detach().numpy()
    live &= x**2 + y**2 <= env.config.beam_pipe_radius**2
    if stop.name == env.target_screen:
        break
x, y = x[live] * 1e6, y[live] * 1e6
figure, (fine, prof) = plt.subplots(1, 2, figsize=(10, 4.2))
edges = np.linspace(-500, 500, 161)
fine.hist2d(x, y, bins=(edges, edges), cmap="Blues")
fine.plot(0, 0, "+", color="#f97316", ms=11, mew=1.3)
for n in (1, 2):
    fine.add_patch(plt.Circle((0, 0), 100 * n, fill=False, ec="#7c3aed", ls="--", lw=0.9))
fine.set(xlabel="x (um)", ylabel="y (um)", aspect="equal",
         title=f"{len(x)} particles ({live.mean():.0%} of the bunch)\n"
               f"dashed: the 100 um request, 1 and 2 sigma")
for values, colour, label in ((x, "#1f77b4", "x"), (y, "#2ca02c", "y")):
    prof.hist(values, bins=edges, histtype="step", color=colour,
              label=f"{label}, sigma {values.std():.1f} um, mean {values.mean():+.2f} um")
prof.set(xlabel="position (um)", ylabel="particles", title="projections")
prof.legend(fontsize=8, frameon=False)
figure.tight_layout(); figure.savefig(f"{OUT}/09_spot_detail.png", dpi=120); plt.close(figure)
print(f"core sigma ({x.std():.2f}, {y.std():.2f}) um, {live.mean():.1%} delivered")
print("plots redrawn")
