"""
Config I/O for the clustering framework
=======================================

Translates a plain dict (loaded from YAML or JSON) into the framework's
parameter dataclasses (StructuralParams / NoiseParams / AmbiguityParams /
DensityParams), and back. This is what lets a user drive the generator from an
editable config file instead of writing Python.

A config file has the shape::

    n_samples: 1000
    dimension: 20
    n_clusters: 5
    random_seed: 42
    structural:      { ... Theta_struct fields ... }
    noise:           { ... Theta_noise fields ... }
    ambiguity:       { ... Theta_amb fields ... }

Any field left out falls back to the dataclass default, so configs can be as
terse or as complete as you like. Nested `density_params` (inside `structural`)
is also supported.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

from clustaudit.config.params import (  # noqa: E402
    StructuralParams,
    NoiseParams,
    AmbiguityParams,
    DensityParams,
    ClusterParams,
    ManifoldParams,
    DataGenerationConfig,
)


# ---------------------------------------------------------------------------
# Loading raw config files
# ---------------------------------------------------------------------------
def load_raw(path: str | Path) -> Dict[str, Any]:
    """Load a YAML or JSON config file into a plain dict."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # optional dependency
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "PyYAML is required to read .yaml configs "
                "(`pip install pyyaml`), or use a .json config instead."
            ) from e
        return yaml.safe_load(text) or {}
    if path.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported config extension: {path.suffix} (use .yaml/.yml/.json)")


# ---------------------------------------------------------------------------
# dict -> dataclasses
# ---------------------------------------------------------------------------
def _tuplify(d: Dict[str, Any], keys) -> None:
    """YAML/JSON have no tuples; convert list-valued range fields to tuples in place."""
    for k in keys:
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])


def build_structural(d: Dict[str, Any]) -> StructuralParams:
    d = dict(d or {})
    density = d.pop("density_params", None)
    cluster = d.pop("cluster_params", None)
    manifold = d.pop("manifold_params", None)
    _tuplify(d, ["cluster_std_range", "anisotropy_factor_range"])
    kwargs: Dict[str, Any] = dict(d)
    if density is not None:
        dd = dict(density)
        _tuplify(dd, ["inter_density_clip"])
        kwargs["density_params"] = DensityParams(**dd)
    if cluster is not None:
        cc = dict(cluster)
        _tuplify(cc, ["n_components_range", "std_range", "anisotropy_clip",
                      "cross_strength_range", "cross_clip"])
        kwargs["cluster_params"] = ClusterParams(**cc)
    if manifold is not None:
        kwargs["manifold_params"] = ManifoldParams(**dict(manifold))
    return StructuralParams(**kwargs)


def build_noise(d: Dict[str, Any]) -> NoiseParams:
    return NoiseParams(**dict(d or {}))


def build_ambiguity(d: Dict[str, Any]) -> AmbiguityParams:
    return AmbiguityParams(**dict(d or {}))


def config_from_dict(cfg: Dict[str, Any]) -> DataGenerationConfig:
    """Turn a plain dict (from YAML/JSON) into a validated DataGenerationConfig."""
    return DataGenerationConfig(
        n_samples=int(cfg["n_samples"]),
        dimension=int(cfg["dimension"]),
        n_clusters=int(cfg["n_clusters"]),
        random_seed=cfg.get("random_seed"),
        structural=build_structural(cfg.get("structural", {})),
        noise=build_noise(cfg.get("noise", {})),
        ambiguity=build_ambiguity(cfg.get("ambiguity", {})),
        pipeline_steps=cfg.get("pipeline_steps") or {
            "hypercube": True,
            "generate_clusters": True,
            "pre_refine": True,
            "inject_noise": True,
            "post_refine": True,
        },
    )


def load_config(path: str | Path) -> DataGenerationConfig:
    """Convenience: file path -> validated DataGenerationConfig."""
    return config_from_dict(load_raw(path))


# ---------------------------------------------------------------------------
# dataclasses -> dict (for round-tripping / saving the effective config)
# ---------------------------------------------------------------------------
def config_to_dict(config: DataGenerationConfig) -> Dict[str, Any]:
    """Serialize a DataGenerationConfig back to a plain, JSON-friendly dict."""
    return {
        "n_samples": config.n_samples,
        "dimension": config.dimension,
        "n_clusters": config.n_clusters,
        "random_seed": config.random_seed,
        "structural": asdict(config.structural),
        "noise": asdict(config.noise),
        "ambiguity": asdict(config.ambiguity),
        "pipeline_steps": config.pipeline_steps,
    }


def save_effective_config(config: DataGenerationConfig, path: str | Path) -> None:
    """Write the fully-resolved config (defaults included) next to the outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(config), indent=2, default=str),
                    encoding="utf-8")
