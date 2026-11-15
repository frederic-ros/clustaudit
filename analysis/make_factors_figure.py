#!/usr/bin/env python
"""
Figure 2 of the v2 paper: the controllable factors, one 2-D panel each, same
seed, each panel varying one factor from the medium operating point. Cores
filled, boundary hollow, internal noise 'x', distant noise '^' (as identified
by the STATE measurement, not by construction).

    python analysis/make_factors_figure.py            # -> analysis/figures/factors_2d.pdf
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clustaudit import generate_dataset, StructuralParams, NoiseParams, AmbiguityParams, DensityParams

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
SEED, N, K = 7, 600, 4


def base():
    return dict(structural=StructuralParams(hypercube_connectivity=0.5, enable_anisotropy=False, enable_rotation=True,
                                            cluster_std_range=(0.25, 0.25), cluster_size_imbalance=1.0,
                                            density_params=DensityParams(0.0, intra_density_strength=0.0)),
                noise=NoiseParams(noise_ratio=0.0), ambiguity=AmbiguityParams())


def panels():
    p = base(); yield "reference", p
    p = base(); p["structural"].cluster_std_range = (0.6, 0.6); yield r"overlap $\sigma=0.6$", p
    p = base(); p["structural"].enable_anisotropy = True; p["structural"].anisotropy_factor_range = (6, 6); yield r"anisotropy $\alpha=6$", p
    p = base(); p["structural"].density_params = DensityParams(1.0, intra_density_strength=0.0); yield r"inter-density $\delta_{inter}=1$", p
    p = base(); p["structural"].density_params = DensityParams(0.0, intra_density_mode="core_dense", intra_density_strength=2.0); yield "intra-density core-dense", p
    p = base(); p["structural"].cluster_size_imbalance = 4.0; yield r"imbalance $\rho_{imb}=4$", p
    p = base(); p["noise"] = NoiseParams(noise_ratio=0.15, internal_noise_ratio=0.0, distant_noise_factor=1.5); yield r"noise $\eta=0.15,\ \eta_{int}=0$ ($\lambda_{dist}=1.5$ in 2-D)", p
    p = base(); p["noise"] = NoiseParams(noise_ratio=0.15, internal_noise_ratio=1.0); yield r"noise $\eta=0.15,\ \eta_{int}=1$", p


def main():
    items = list(panels())
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    for ax, (title, p) in zip(axes.ravel(), items):
        st = generate_dataset(N, 2, K, random_seed=SEED, **p)
        X, y, m = st.Z, st.y, st.masks
        cmap = plt.get_cmap("tab10")
        for k in np.unique(y[y >= 0]):
            c = m["core"] & (y == k); b = m["boundary"] & (y == k)
            ax.scatter(X[c, 0], X[c, 1], s=9, color=cmap(k), alpha=.8, linewidths=0)
            ax.scatter(X[b, 0], X[b, 1], s=16, facecolors="none", edgecolors=cmap(k), linewidths=.7)
        ax.scatter(X[m["internal_noise"], 0], X[m["internal_noise"], 1], s=18, marker="x", color="k", linewidths=.8)
        ax.scatter(X[m["distant_noise"], 0], X[m["distant_noise"], 1], s=18, marker="^", color="k", linewidths=0)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        s = st.summary()
        ax.text(.02, .02, f"bnd {s['n_boundary']}  int {s['n_internal_noise']}  dist {s['n_distant_noise']}",
                transform=ax.transAxes, fontsize=7, va="bottom", zorder=5,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"factors_2d.{ext}"), dpi=200)
    print(f"-> {OUT}/factors_2d.pdf")


if __name__ == "__main__":
    main()
