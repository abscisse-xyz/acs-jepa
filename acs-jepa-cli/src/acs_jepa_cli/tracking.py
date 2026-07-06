"""MLflow tracking helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf

from acs_jepa_cli.config import flatten_config


def configure_mlflow(config: DictConfig) -> None:
    """Apply tracking URI and experiment config."""

    tracking_uri = config.tracking.mlflow_tracking_uri
    if tracking_uri is not None:
        mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(str(config.tracking.experiment_name))


def start_run(config: DictConfig, *, fallback_name: str, extra_tags: dict[str, Any] | None = None):
    """Start an MLflow run with configured name and tags."""

    run_name = config.tracking.run_name
    tags = OmegaConf.to_container(config.tracking.tags, resolve=True)
    if not isinstance(tags, dict):
        tags = {}
    if extra_tags:
        tags.update(extra_tags)
    return mlflow.start_run(run_name=str(run_name or fallback_name), tags={str(k): str(v) for k, v in tags.items()})


def log_config_params(config: DictConfig, *, extra: dict[str, Any] | None = None) -> None:
    """Log config scalars and optional extra params."""

    params = flatten_config(config)
    if extra:
        for key, value in extra.items():
            params[key] = str(value)[:240]
    for start in range(0, len(params), 100):
        items = list(params.items())[start : start + 100]
        mlflow.log_params(dict(items))


def log_metrics(prefix: str, metrics: dict[str, float], *, step: int | None = None) -> None:
    """Log numeric metrics with a prefix."""

    mlflow.log_metrics({f"{prefix}/{key}": float(value) for key, value in metrics.items()}, step=step)


def log_json_artifact(payload: Any, path: Path, *, artifact_path: str | None = None) -> None:
    """Write and log a JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def config_hash(config: DictConfig) -> str:
    """Return a short stable hash for a resolved config."""

    payload = json.dumps(OmegaConf.to_container(config, resolve=True), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
