"""Figures for the image-based 6D reconstruction at the S02 start (run_ment.py outputs)."""

from __future__ import annotations

import sys
from pathlib import Path

import common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ment6d
import numpy as np

FIG = common.HERE / "figures"
UNITS = {"x": ("x", 1e3, "mm"), "px": ("px", 1e3, "mrad"), "y": ("y", 1e3, "mm"),
         "py": ("py", 1e3, "mrad"), "tau": ("tau", 1e3, "mm"), "delta": ("delta", 1e2, "%")}
QUANTITIES = [
    ("eps_x", "eps x"), ("eps_y", "eps y"), ("eps_z", "eps z"), ("eps_4d_ratio", "eps4d/(ex ey)"),
    ("slice_eps_x", "slice eps x"), ("slice_eps_y", "slice eps y"), ("sigma_z", "sigma z"),
    ("sigma_delta", "sigma delta"), ("slice_sigma_delta", "slice sigma delta"), ("chirp", "chirp"),
]
CORRELATIONS = [("corr_tau_delta", "tau,delta"), ("corr_x_y", "x,y"), ("corr_x_py", "x,py"),
                ("corr_px_y", "px,y"), ("corr_x_delta", "x,delta"), ("corr_y_tau", "y,tau")]


def projections(sample, weights, beam, out, extent_sd=3.5, bins=60):
    labels = common.LABELS
    sd = beam.std(0)
    mean = beam.mean(0)
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    for row in range(5):
        for col in range(5):
            ax = axes[row, col]
            i, j = col, row + 1
            if col > row:
                ax.axis("off")
                continue
            ri = (mean[i] - extent_sd * sd[i], mean[i] + extent_sd * sd[i])
            rj = (mean[j] - extent_sd * sd[j], mean[j] + extent_sd * sd[j])
            h, xe, ye = np.histogram2d(sample[:, i], sample[:, j], bins=bins, range=[ri, rj], weights=weights)
            t, _, _ = np.histogram2d(beam[:, i], beam[:, j], bins=bins, range=[ri, rj])
            si, sj = UNITS[labels[i]][1], UNITS[labels[j]][1]
            ax.imshow(h.T, origin="lower", aspect="auto", cmap="inferno",
                      extent=[xe[0] * si, xe[-1] * si, ye[0] * sj, ye[-1] * sj])
            xc = 0.5 * (xe[1:] + xe[:-1]) * si
            yc = 0.5 * (ye[1:] + ye[:-1]) * sj
            ax.contour(xc, yc, t.T / t.max(), levels=[0.1, 0.5], colors="white", linewidths=0.6)
            if row == 4:
                ax.set_xlabel(f"{labels[i]} ({UNITS[labels[i]][2]})", fontsize=7)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(f"{labels[j]} ({UNITS[labels[j]][2]})", fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)
    fig.suptitle("Reconstructed density at S02 (colour) and true beam (white contours, 10 and 50 %)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def quantity_ratios(results, out):
    truth = results["truth"]
    sets = results["sets"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2), gridspec_kw={"width_ratios": [1.7, 1]})
    names = list(sets)
    width = 0.8 / max(len(names), 1)
    for n, name in enumerate(names):
        reco = sets[name]["summary"]
        ratios = [reco[k] / truth[k] for k, _ in QUANTITIES]
        ax1.bar(np.arange(len(QUANTITIES)) + n * width, ratios, width, label=f"{name} ({sets[name]['n_images']})")
        diffs = [reco[k] - truth[k] for k, _ in CORRELATIONS]
        ax2.bar(np.arange(len(CORRELATIONS)) + n * width, diffs, width)
    ax1.axhline(1, color="k", lw=0.6)
    ax1.set_yscale("log")
    ax1.set_xticks(np.arange(len(QUANTITIES)) + 0.4 - width / 2, [lab for _, lab in QUANTITIES], rotation=60, ha="right", fontsize=7)
    ax1.set_ylabel("reconstructed / true")
    ax1.legend(fontsize=6, title="image set (images)", title_fontsize=6)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xticks(np.arange(len(CORRELATIONS)) + 0.4 - width / 2, [lab for _, lab in CORRELATIONS], rotation=60, ha="right", fontsize=7)
    ax2.set_ylabel("correlation error")
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def image_fits(full, out, picks):
    keys = [str(k) for k in full["keys"]]
    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(min(8, 2 * n), 4))
    for c, key in enumerate(picks):
        k = keys.index(key)
        for r, arr in enumerate((full["measured"][k], full["predicted"][k])):
            img = arr[:-1].reshape(ment6d.PIXELS, ment6d.PIXELS)
            ax = axes[r, c]
            e = ment6d.full6d.FIELD_OF_VIEW_M * 1e3
            ax.imshow(img, origin="lower", cmap="inferno", extent=[-e, e, -e, e])
            ax.tick_params(labelsize=6)
            if r == 0:
                cfg, scr = key.split("|")
                ax.set_title(f"{cfg}\n{scr.replace('cla_', '')}", fontsize=6)
            if c == 0:
                ax.set_ylabel(("measured" if r == 0 else "reconstruction") + "\ny (mm)", fontsize=7)
            if r == 1:
                ax.set_xlabel("x (mm)", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def convergence(results, out):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    for name, entry in results["sets"].items():
        chi = [h[1] for st in entry["stages"] for h in st["history"]]
        ax.plot(chi, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("MENT sweep (all stages)")
    ax.set_ylabel("chi2 per pixel")
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    results = common.read_json(common.HERE / "ment_results.json")
    full = np.load(common.HERE / "ment_full.npz")
    beam = np.load(common.HERE / "images.npz")["beam"]
    projections(full["sample"], full["weights"], beam, FIG / "start_projections.png")
    quantity_ratios(results, FIG / "start_quantities.png")
    convergence(results, FIG / "start_convergence.png")
    keys = [str(k) for k in full["keys"]]
    wanted = ["nominal_off|cla_s02_dia_scr_02", "s02a|cla_s04_dia_scr_01", "nominal_off|cla_s07_dia_scr_02"]
    wanted += [k for k in keys if k.startswith("s07_dcp")][:2] + [k for k in keys if k.startswith("fea")][:3]
    image_fits(full, FIG / "start_images.png", [k for k in wanted if k in keys][:8])
    if len(sys.argv) > 1:  # copy to a destination folder
        dest = Path(sys.argv[1])
        dest.mkdir(parents=True, exist_ok=True)
        for f in FIG.glob("start_*.png"):
            (dest / f.name).write_bytes(f.read_bytes())


if __name__ == "__main__":
    main()
