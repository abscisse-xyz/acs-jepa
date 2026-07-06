"""OmegaConf configuration helpers for the ACS-JEPA CLI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
ConfigPathInput = str | Path | Sequence[str | Path] | None


def load_config(path: ConfigPathInput = None) -> DictConfig:
    """Load YAML config overlays over the default YAML config."""

    base = OmegaConf.load(DEFAULT_CONFIG_PATH)
    return OmegaConf.merge(base, load_config_overrides(path))


def load_config_overrides(path: ConfigPathInput = None) -> DictConfig:
    """Load one or more YAML config overlays without the default config."""

    configs = [OmegaConf.load(config_path) for config_path in _config_paths(path)]
    if not configs:
        return OmegaConf.create({})
    return OmegaConf.merge(*configs)


def config_paths_text(path: ConfigPathInput = None) -> str:
    """Return config paths in merge order for logging."""

    return ",".join(str(config_path) for config_path in _config_paths(path))


def tuning_overlay_tags(path: ConfigPathInput = None) -> dict[str, str]:
    """Return MLflow tags summarizing tuning overlays in merge order."""

    stage_variants: dict[str, list[str]] = {}
    stack = []
    for config_path in _config_paths(path):
        parsed = Path(config_path)
        stage = _tuning_stage(parsed)
        if stage is None:
            continue
        variant = parsed.stem
        stage_variants.setdefault(stage, []).append(variant)
        stack.append(f"{stage}:{variant}")

    tags = {f"tuning.{stage}": "+".join(variants) for stage, variants in stage_variants.items()}
    if stack:
        tags["tuning.stack"] = "|".join(stack)
    return tags


def _config_paths(path: ConfigPathInput) -> tuple[str | Path, ...]:
    if path is None:
        return ()
    if isinstance(path, (str, Path)):
        return (path,)
    return tuple(path)


def _tuning_stage(path: Path) -> str | None:
    parts = path.parts
    if "tuning" in parts:
        tuning_idx = parts.index("tuning")
        if tuning_idx + 1 >= len(parts) - 1:
            return None
        return _stage_name(parts[tuning_idx + 1])
    parent = path.parent.name
    if parent[:2].isdigit() and len(parent) > 3 and parent[2] == "_":
        return _stage_name(parent)
    return None


def _stage_name(value: str) -> str:
    if value[:2].isdigit() and len(value) > 3 and value[2] == "_":
        return value[3:]
    return value


def save_resolved_config(config: DictConfig, path: str | Path) -> None:
    """Write a fully resolved YAML config."""

    OmegaConf.save(config=config, f=Path(path), resolve=True)


def config_to_container(config: DictConfig) -> dict[str, Any]:
    """Return a plain Python dictionary with interpolations resolved."""

    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("Expected a mapping config")
    return value


def flatten_config(config: DictConfig) -> dict[str, str | int | float | bool]:
    """Flatten scalar config values for MLflow parameters."""

    flat: dict[str, str | int | float | bool] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, dict):
            for key, item in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                visit(name, item)
            return
        if value is None:
            flat[prefix] = "null"
            return
        if isinstance(value, (str, int, float, bool)):
            text = value if not isinstance(value, str) else value[:240]
            flat[prefix] = text
            return
        flat[prefix] = str(value)[:240]

    visit("", config)
    return flat
