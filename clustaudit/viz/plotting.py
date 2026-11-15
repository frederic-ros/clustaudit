"""
Visualization: Plot synthetic clustering datasets with t-SNE/UMAP
==================================================================

This module provides visualization functions for synthetic clustering datasets.

Part of clustaudit.
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal
from ..core.state import ClusteringState

# Optional dependencies
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def plot_dataset(
    state: ClusteringState,
    method: Literal["auto", "2d", "tsne", "umap", "shared"] = "auto",
    figsize: tuple = (12, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot clustering dataset with automatic 2D handling and optional projections.

    Parameters
    ----------
    state : ClusteringState
        Dataset to visualize
    method : str, default="auto"
        Projection method: "auto", "2d", "tsne", "umap", or "shared"
    figsize : tuple, default=(12, 8)
        Figure size
    title : str, optional
        Figure title
    save_path : str, optional
        Path to save figure
    """
    dimension = state.dimension()
    n_samples = state.n_samples()

    # ========================================================================
    # Projection logic
    # ========================================================================
    def project_to_2d():
        # Shared projection (trace mode)
        if method == "shared":
            if not hasattr(state, "_cached_projection"):
                raise ValueError("Shared projection requested but none cached in state.")
            return state._cached_projection, "X", "Y", "shared"

        # Native 2D → no projection
        if dimension == 2:
            return state.Z, "X", "Y", "native"

        # Auto / UMAP
        if method in ("auto", "umap"):
            if HAS_UMAP:
                reducer = umap.UMAP(
                    n_components=2,
                    random_state=state.random_seed or 42,
                    n_neighbors=min(15, max(2, n_samples // 10)),
                )
                return reducer.fit_transform(state.Z), "UMAP-1", "UMAP-2", "umap"

        # Auto / t-SNE
        if method in ("auto", "tsne"):
            if not HAS_TSNE:
                raise ImportError("t-SNE requires scikit-learn")
            tsne = TSNE(
                n_components=2,
                random_state=state.random_seed or 42,
                perplexity=min(30, max(5, n_samples // 4)),
            )
            return tsne.fit_transform(state.Z), "t-SNE-1", "t-SNE-2", "tsne"

        raise ValueError(f"Cannot project {dimension}D data with method='{method}'.")

    # ========================================================================
    # Projection
    # ========================================================================
    X_2d, xlabel, ylabel, used_method = project_to_2d()

    # ========================================================================
    # Figure
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ========================================================================
    # Left: clusters
    # ========================================================================
    ax1 = axes[0]

    core_idx = state.core_points()
    if len(core_idx):
        ax1.scatter(
            X_2d[core_idx, 0],
            X_2d[core_idx, 1],
            c=state.y[core_idx],
            cmap="tab10",
            s=50,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
            label="Core",
        )

    boundary_idx = state.boundary_points()
    if len(boundary_idx):
        ax1.scatter(
            X_2d[boundary_idx, 0],
            X_2d[boundary_idx, 1],
            c=state.y[boundary_idx],
            cmap="tab10",
            s=40,
            alpha=0.5,
            marker="s",
            edgecolors="orange",
            linewidths=1,
            label="Boundary",
        )

    noise_idx = state.noise_points()
    if len(noise_idx):
        ax1.scatter(
            X_2d[noise_idx, 0],
            X_2d[noise_idx, 1],
            c="gray",
            s=30,
            alpha=0.3,
            marker="x",
            label="Noise",
        )

    ax1.set_title(
        f"Clusters (n={n_samples}, K={state.n_clusters()}, d={dimension}, proj={used_method})"
    )
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ========================================================================
    # Right: entropy
    # ========================================================================
    ax2 = axes[1]
    non_noise = state.y >= 0

    if np.any(non_noise):
        scatter = ax2.scatter(
            X_2d[non_noise, 0],
            X_2d[non_noise, 1],
            c=state.entropy[non_noise],
            cmap="YlOrRd",
            s=50,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.3,
        )
        plt.colorbar(scatter, ax=ax2, label="Entropy")

    if len(noise_idx):
        ax2.scatter(
            X_2d[noise_idx, 0],
            X_2d[noise_idx, 1],
            c="black",
            s=30,
            alpha=0.3,
            marker="x",
        )

    ax2.set_title(f"Entropy (mean={state.summary()['mean_entropy']:.3f})")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {save_path}")

    plt.show()


def plot_dataset_comparison(
    states: list,
    titles: list,
    method: Literal["2d", "tsne", "umap"] = "tsne",
    figsize: tuple = (18, 5),
    save_path: Optional[str] = None
):
    """
    Plot multiple datasets side-by-side.

    Parameters
    ----------
    states : list of ClusteringState
        Datasets to compare
    titles : list of str
        Titles for each dataset
    method : str, default="tsne"
        Projection method
    figsize : tuple, default=(18, 5)
        Figure size
    save_path : str, optional
        Path to save figure
    """
    n = len(states)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for idx, (state, title) in enumerate(zip(states, titles)):
        ax = axes[idx]

        # Project to 2D
        if method == "2d":
            X_2d = state.Z
        elif method == "tsne":
            if not HAS_TSNE:
                raise ImportError("t-SNE requires scikit-learn")
            tsne = TSNE(
                n_components=2,
                random_state=state.random_seed or 42,
                perplexity=min(30, state.n_samples() // 4)
            )
            X_2d = tsne.fit_transform(state.Z)
        else:
            if not HAS_UMAP:
                raise ImportError("UMAP requires umap-learn")
            reducer = umap.UMAP(
                n_components=2,
                random_state=state.random_seed or 42,
                n_neighbors=min(15, state.n_samples() // 10)
            )
            X_2d = reducer.fit_transform(state.Z)

        # Plot core points
        core_idx = state.core_points()
        if len(core_idx) > 0:
            ax.scatter(
                X_2d[core_idx, 0],
                X_2d[core_idx, 1],
                c=state.y[core_idx],
                cmap='tab10',
                s=40,
                alpha=0.7,
                edgecolors='k',
                linewidths=0.3,
                label='Core'
            )

        # Plot boundary points
        boundary_idx = state.boundary_points()
        if len(boundary_idx) > 0:
            ax.scatter(
                X_2d[boundary_idx, 0],
                X_2d[boundary_idx, 1],
                c=state.y[boundary_idx],
                cmap='tab10',
                s=30,
                alpha=0.5,
                marker='s',
                edgecolors='orange',
                linewidths=0.8,
                label='Boundary'
            )

        # Plot noise points
        noise_idx = state.noise_points()
        if len(noise_idx) > 0:
            ax.scatter(
                X_2d[noise_idx, 0],
                X_2d[noise_idx, 1],
                c='gray',
                s=20,
                alpha=0.3,
                marker='x',
                label='Noise'
            )

        ax.set_title(f"{title}\n(n={state.n_samples()}, d={state.dimension()})")
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()
