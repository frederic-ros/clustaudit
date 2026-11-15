#!/usr/bin/env python
"""
One generated dataset, stage by stage (Figure "pipeline" of the paper).

    python analysis/make_pipeline_figure.py [--out analysis/figures_paper] [--sigma 0.5] [--seed 7]

Four panels, the same 2-D dataset at the four stages of the pipeline:
  (a) clusters generated: labels as sampled                (stages 1-2)
  (b) pre-noise refinement: labels made self-consistent    (stage 3)
  (c) noise injected: bridge and distant populations       (stage 4)
  (d) STATE measured: cores, boundary, internal/distant    (stage 5)
Every number printed on the figure is computed here from the trace of the run.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, os.pardir)); sys.path.insert(0, ROOT)
from clustaudit.config.params import StructuralParams, NoiseParams, AmbiguityParams, DensityParams
from clustaudit.core.pipeline import generate_dataset

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 11.5, "legend.fontsize": 10})
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]


def refinement_trajectory(X, y0, n_iter_max=10, thr=0.999):
    """Replay the pre-noise refinement loop to recover its accuracy trajectory."""
    k = max(1, min(int(np.sqrt(len(y0))), len(y0) - 1)); y = y0.copy(); acc = []
    for _ in range(n_iter_max):
        clf = KNeighborsClassifier(n_neighbors=k).fit(X, y); p = clf.predict(X); a = np.mean(p == y); acc.append(a)
        if a >= thr or len(np.unique(p)) < len(np.unique(y0)): break
        y = p
    return acc, y, k


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "figures_paper"))
    ap.add_argument("--sigma", type=float, default=0.5); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--K", type=int, default=4)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    structural = StructuralParams(hypercube_connectivity=0.5, enable_anisotropy=False, enable_rotation=True,
                                  cluster_std_range=(a.sigma, a.sigma), cluster_size_imbalance=1.0,
                                  density_params=DensityParams(0.0, intra_density_strength=0.0))
    noise = NoiseParams(noise_ratio=0.15, internal_noise_ratio=0.6, distant_noise_factor=1.5)   # lambda_dist=1.5 in 2-D (see Fig. 2)
    amb = AmbiguityParams()
    state, trace = generate_dataset(a.n, 2, a.K, structural, noise, amb, random_seed=a.seed, record_trace=True)
    st = {s.name: s.state for s in trace.steps}
    S0, S1, S2, S3 = st["clusters_generated"], st["pre_refinement"], st["noise_injected"], st["post_refinement"]
    acc, y_replay, k = refinement_trajectory(S0.Z, S0.y, thr=amb.pre_refinement_threshold)
    changed = S1.y != S0.y
    assert np.array_equal(y_replay, S1.y), "replayed refinement differs from the pipeline's"
    lab = S3.y >= 0; core = S3.masks["core"] & lab; bnd = S3.masks["boundary"] & lab
    internal = S3.masks["internal_noise"]; distant = S3.masks["distant_noise"]
    # out-of-sample k-NN error of the final labelled points (what the measurement stage reports)
    kk = max(1, min(int(np.sqrt(lab.sum())), lab.sum() - 1))
    cv = cross_val_predict(KNeighborsClassifier(n_neighbors=kk), S3.Z[lab], S3.y[lab], cv=StratifiedKFold(5, shuffle=True, random_state=0))
    cv_err = np.mean(cv != S3.y[lab])

    fig, axes = plt.subplots(2, 2, figsize=(11, 10)); axes = axes.ravel()
    lo, hi = S0.Z.min(0), S0.Z.max(0); span = (hi - lo).max(); c0 = (lo + hi) / 2; half = 0.62 * span
    for ax in axes:
        ax.set_xlim(c0[0] - half, c0[0] + half); ax.set_ylim(c0[1] - half, c0[1] + half); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    inside = lambda Z: (np.abs(Z - c0) <= half).all(1)

    ax = axes[0]
    for c in range(a.K): m = S0.y == c; ax.scatter(*S0.Z[m].T, s=9, color=COL[c], alpha=.8, lw=0)
    ax.set_title(f"(a) stages 1–2: {a.K} clusters sampled ($\\sigma = {a.sigma}$), labels as sampled")

    ax = axes[1]
    # decision regions of the k-NN *before* refinement (fitted to the sampled labels): where they disagree
    # with the sampled labels is where the labels are not supported by the geometry
    clf0 = KNeighborsClassifier(n_neighbors=k).fit(S0.Z, S0.y)
    gx, gy = np.meshgrid(np.linspace(c0[0] - half, c0[0] + half, 300), np.linspace(c0[1] - half, c0[1] + half, 300))
    Zg = clf0.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    from matplotlib.colors import ListedColormap
    ax.contourf(gx, gy, Zg, levels=np.arange(-0.5, a.K + 0.5, 1), colors=COL[:a.K], alpha=.12, antialiased=True)
    ax.contour(gx, gy, Zg, levels=np.arange(0.5, a.K - 0.5, 1), colors="k", linewidths=.5, alpha=.5)
    for c in range(a.K): m = (S0.y == c) & ~changed; ax.scatter(*S0.Z[m].T, s=9, color=COL[c], alpha=.35, lw=0)
    for i in np.where(changed)[0]:   # face = new label, edge = old label
        ax.scatter(S0.Z[i, 0], S0.Z[i, 1], s=70, color=COL[S1.y[i]], edgecolor=COL[S0.y[i]], lw=2.2, zorder=3)
    traj = " → ".join(f"{x:.3f}" for x in acc)
    ax.set_title(f"(b) stage 3: pre-noise refinement ($k$-NN, $k = {k}$)\nshaded: $k$-NN regions fitted to the sampled labels\nin-sample accuracy {traj}\n{changed.mean()*100:.1f}% of labels changed (face = new, ring = sampled)")

    ax = axes[2]
    lab2 = S2.y >= 0
    for c in range(a.K): m = S2.y == c; ax.scatter(*S2.Z[m].T, s=9, color=COL[c], alpha=.35, lw=0)
    ni = S2.masks["internal_noise"] if S2.masks["internal_noise"].any() else None
    # at this stage the placed populations are known from the generator: show them as placed
    n_noise = (~lab2).sum(); n_int = int(round(noise.internal_noise_ratio * n_noise))
    idx_noise = np.where(~lab2)[0]
    ax.scatter(*S2.Z[idx_noise[:n_int]].T, s=26, marker="x", color="k", lw=1, label=f"bridge noise ({n_int})")
    ax.scatter(*S2.Z[idx_noise[n_int:]].T, s=30, marker="+", color="#c0392b", lw=1.2, label=f"distant noise ({n_noise - n_int})")
    n_out = (~inside(S2.Z[idx_noise[n_int:]])).sum()
    ax.legend(loc="lower left", frameon=True, title=f"{n_out} distant points outside the frame", title_fontsize=10)
    ax.set_title(f"(c) stage 4: noise injected, $\\eta = {noise.noise_ratio}$, $\\eta_{{\\mathrm{{int}}}} = {noise.internal_noise_ratio}$\nthe two populations as placed by the generator")

    ax = axes[3]
    for c in range(a.K):
        m = (S3.y == c) & core; ax.scatter(*S3.Z[m].T, s=9, color=COL[c], alpha=.8, lw=0)
        m = (S3.y == c) & bnd; ax.scatter(*S3.Z[m].T, s=26, facecolor="none", edgecolor=COL[c], lw=1.0)
    ax.scatter(*S3.Z[internal].T, s=26, marker="x", color="k", lw=1)
    ax.scatter(*S3.Z[distant].T, s=30, marker="+", color="#c0392b", lw=1.2)
    ax.scatter([], [], s=9, color="grey", label=f"core, $c_i \\geq {amb.core_threshold}$ ({core.sum()})")
    ax.scatter([], [], s=26, facecolor="none", edgecolor="grey", label=f"boundary ({bnd.sum()})")
    ax.scatter([], [], s=26, marker="x", color="k", label=f"internal noise ({internal.sum()})")
    ax.scatter([], [], s=30, marker="+", color="#c0392b", label=f"distant noise ({distant.sum()})")
    ax.legend(loc="lower left", frameon=True)
    ax.set_title(f"(d) stage 5: STATE measured by one fixed instrument\ncategories from the out-of-sample confidence and the gap scale\n5-fold $k$-NN error of the reference labels: {cv_err*100:.1f}%")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"pipeline_stages.{ext}"), dpi=200, bbox_inches="tight")
    print(f"-> {a.out}/pipeline_stages.pdf | acc {traj} | changed {changed.mean():.3f} | cv err {cv_err:.3f} | core {core.sum()} bnd {bnd.sum()} int {internal.sum()} dist {distant.sum()}")


if __name__ == "__main__":
    main()
