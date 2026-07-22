"""Action-latent distribution statistics for diagnostic scripts."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Any, Sequence

import torch


def latent_distribution_stats(latents: torch.Tensor) -> dict[str, Any]:
    """Return JSON-safe global distribution statistics for `[N, D]` latents."""

    values = _as_2d_float(latents)
    count, dim = values.shape
    if count == 0:
        return {
            "count": 0,
            "dim": int(dim),
            "std_mean": 0.0,
            "std_min": 0.0,
            "std_max": 0.0,
            "std_values": [],
            "cov_offdiag_mean_sq": 0.0,
            "effective_rank": 0.0,
            "eigenvalues": [],
        }

    std = values.std(dim=0, unbiased=False)
    covariance = _covariance(values)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0).sort(descending=True).values
    return {
        "count": int(count),
        "dim": int(dim),
        "std_mean": _float(std.mean()),
        "std_min": _float(std.min()),
        "std_max": _float(std.max()),
        "std_values": [_float(value) for value in std],
        "cov_offdiag_mean_sq": _float(_offdiag_mean_sq(covariance)),
        "effective_rank": _effective_rank(eigenvalues),
        "eigenvalues": [_float(value) for value in eigenvalues],
    }


def schema_group_stats(latents: torch.Tensor, schema_ids: Sequence[str], *, min_count: int = 2) -> dict[str, Any]:
    """Return distribution statistics per action schema with enough samples."""

    values = _as_2d_float(latents)
    _check_metadata(values, schema_ids)
    groups = _schema_indices(schema_ids)
    schemas = {}
    skipped = 0
    for schema, indices in sorted(groups.items()):
        if len(indices) < min_count:
            skipped += 1
            continue
        schemas[schema] = latent_distribution_stats(values[indices])
    return {
        "min_count": int(min_count),
        "schema_count": len(schemas),
        "skipped_schema_count": skipped,
        "schemas": schemas,
    }


def same_schema_nearest_wrong_margins(
    latents: torch.Tensor,
    schema_ids: Sequence[str],
    action_keys: Sequence[Any],
    *,
    group_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Measure nearest different action under the same schema for each latent."""

    values = _as_2d_float(latents)
    _check_metadata(values, schema_ids)
    if len(action_keys) != values.size(0):
        raise ValueError("action_keys length must match number of latents")
    if group_ids is not None and len(group_ids) != values.size(0):
        raise ValueError("group_ids length must match number of latents")

    groups = _margin_group_indices(schema_ids, group_ids)
    distances: list[float] = []
    details: list[dict[str, Any]] = []
    skipped_singletons = 0
    for group_key, indices in sorted(groups.items()):
        schema = group_key[0]
        if len(indices) < 2:
            skipped_singletons += len(indices)
            continue
        group = values[indices]
        squared = torch.cdist(group, group, p=2).pow(2)
        squared.fill_diagonal_(float("inf"))
        nearest_distances, nearest_positions = squared.min(dim=1)
        for local_idx, squared_distance in enumerate(nearest_distances):
            source_idx = indices[local_idx]
            nearest_idx = indices[int(nearest_positions[local_idx].item())]
            distance = math.sqrt(float(squared_distance.item()))
            distances.append(distance)
            details.append(
                {
                    "schema": schema,
                    "group": _json_scalar(group_key[1]),
                    "action": _json_action_key(action_keys[source_idx]),
                    "nearest_wrong_action": _json_action_key(action_keys[nearest_idx]),
                    "nearest_wrong_distance": distance,
                }
            )

    return {
        "count": len(distances),
        "skipped_singleton_count": skipped_singletons,
        "nearest_wrong_distance_min": min(distances) if distances else None,
        "nearest_wrong_distance_median": median(distances) if distances else None,
        "nearest_wrong_distance_mean": sum(distances) / len(distances) if distances else None,
        "details": details,
    }


def reference_same_schema_margins(
    latents: torch.Tensor,
    schema_ids: Sequence[str],
    action_keys: Sequence[Any],
    reference_mask: Sequence[bool],
    group_ids: Sequence[Any],
) -> dict[str, Any]:
    """Measure reference action to nearest wrong same-schema candidate per group."""

    values = _as_2d_float(latents)
    _check_metadata(values, schema_ids)
    if len(action_keys) != values.size(0):
        raise ValueError("action_keys length must match number of latents")
    if len(reference_mask) != values.size(0):
        raise ValueError("reference_mask length must match number of latents")
    if len(group_ids) != values.size(0):
        raise ValueError("group_ids length must match number of latents")

    distances: list[float] = []
    details: list[dict[str, Any]] = []
    skipped_no_wrong = 0
    reference_indices = [idx for idx, is_reference in enumerate(reference_mask) if is_reference]
    for ref_idx in reference_indices:
        ref_schema = str(schema_ids[ref_idx])
        ref_group = group_ids[ref_idx]
        candidate_indices = [
            idx
            for idx, schema in enumerate(schema_ids)
            if idx != ref_idx and str(schema) == ref_schema and group_ids[idx] == ref_group
        ]
        if not candidate_indices:
            skipped_no_wrong += 1
            continue
        deltas = values[candidate_indices] - values[ref_idx].unsqueeze(0)
        candidate_distances = deltas.pow(2).sum(dim=1).sqrt()
        distance, nearest_position = candidate_distances.min(dim=0)
        nearest_idx = candidate_indices[int(nearest_position.item())]
        distance_float = _float(distance)
        distances.append(distance_float)
        details.append(
            {
                "schema": ref_schema,
                "group": _json_scalar(ref_group),
                "reference_action": _json_action_key(action_keys[ref_idx]),
                "nearest_wrong_action": _json_action_key(action_keys[nearest_idx]),
                "nearest_wrong_distance": distance_float,
            }
        )

    return {
        "count": len(distances),
        "reference_count": len(reference_indices),
        "skipped_no_wrong_count": skipped_no_wrong,
        "nearest_wrong_distance_min": min(distances) if distances else None,
        "nearest_wrong_distance_median": median(distances) if distances else None,
        "nearest_wrong_distance_mean": sum(distances) / len(distances) if distances else None,
        "details": details,
    }


def schema_argument_variance_decomposition(latents: torch.Tensor, schema_ids: Sequence[str]) -> dict[str, Any]:
    """Decompose total variance into between-schema and within-schema parts."""

    values = _as_2d_float(latents)
    _check_metadata(values, schema_ids)
    count = values.size(0)
    groups = _schema_indices(schema_ids)
    if count == 0:
        return {
            "count": 0,
            "schema_count": 0,
            "total_variance": 0.0,
            "between_schema_variance": 0.0,
            "within_schema_variance": 0.0,
            "between_fraction": 0.0,
            "within_fraction": 0.0,
            "note": "within_schema_variance includes argument and source-state/context variation",
        }

    global_mean = values.mean(dim=0, keepdim=True)
    total = ((values - global_mean) ** 2).sum(dim=1).mean()
    between = values.new_tensor(0.0)
    within = values.new_tensor(0.0)
    for indices in groups.values():
        group = values[indices]
        weight = group.size(0) / count
        group_mean = group.mean(dim=0, keepdim=True)
        between = between + weight * ((group_mean - global_mean) ** 2).sum()
        within = within + weight * ((group - group_mean) ** 2).sum(dim=1).mean()
    total_float = _float(total)
    return {
        "count": int(count),
        "schema_count": len(groups),
        "total_variance": total_float,
        "between_schema_variance": _float(between),
        "within_schema_variance": _float(within),
        "between_fraction": 0.0 if total_float == 0.0 else _float(between) / total_float,
        "within_fraction": 0.0 if total_float == 0.0 else _float(within) / total_float,
        "note": "within_schema_variance includes argument and source-state/context variation",
    }


def schema_residuals(
    latents: torch.Tensor,
    schema_ids: Sequence[str],
    group_ids: Sequence[Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global-schema and source-state/schema residuals in float64."""

    values = _as_2d_float64(latents)
    _check_metadata(values, schema_ids)
    if len(group_ids) != values.size(0):
        raise ValueError("group_ids length must match number of latents")
    global_residual = torch.empty_like(values)
    state_residual = torch.empty_like(values)
    schema_groups = _schema_indices(schema_ids)
    state_groups = _margin_group_indices(schema_ids, group_ids)
    for indices in schema_groups.values():
        global_residual[indices] = values[indices] - values[indices].mean(dim=0)
    for bucket, indices in state_groups.items():
        if len(indices) < 2:
            raise ValueError(f"state/schema bucket {bucket!r} is a singleton")
        state_residual[indices] = values[indices] - values[indices].mean(dim=0)
    return global_residual, state_residual


def raw_variance_decomposition(latents: torch.Tensor, schema_ids: Sequence[str]) -> dict[str, float]:
    """Compute the exact float64 population schema variance decomposition."""

    values = _as_2d_float64(latents)
    _check_metadata(values, schema_ids)
    if values.size(0) == 0:
        raise ValueError("raw variance decomposition requires a nonempty population")
    grand_mean = values.mean(dim=0)
    total = (values - grand_mean).square().sum(dim=1).mean()
    if float(total) <= torch.finfo(torch.float64).eps:
        raise ValueError("total variance is at or below float64 epsilon")
    between = values.new_zeros(())
    within = values.new_zeros(())
    count = values.size(0)
    for indices in _schema_indices(schema_ids).values():
        group = values[indices]
        mean = group.mean(dim=0)
        between += (len(indices) / count) * (mean - grand_mean).square().sum()
        within += (group - mean).square().sum(dim=1).sum() / count
    error = (total - between - within).abs()
    tolerance = 1e-10 * max(1.0, float(total))
    if float(error) > tolerance:
        raise ValueError("raw variance decomposition does not reconstruct total variance")
    return {
        "total_variance": float(total),
        "between_schema_variance": float(between),
        "within_schema_variance": float(within),
        "between_schema_fraction": float(between / total),
        "within_schema_fraction": float(within / total),
        "reconstruction_absolute_error": float(error),
    }


def residual_statistics(residuals: torch.Tensor) -> dict[str, Any]:
    """Report exact uncentered float64 population residual statistics."""

    values = _as_2d_float64(residuals)
    count, dimension = values.shape
    if count == 0 or dimension == 0:
        raise ValueError("residual statistics require a nonempty matrix")
    std = values.std(dim=0, correction=0)
    covariance = values.T @ values / count
    eigenvalues = torch.linalg.eigvalsh(covariance).sort(descending=True).values
    if bool((eigenvalues < -1e-12).any()):
        raise ValueError("residual covariance has eigenvalue below -1e-12")
    eigenvalues = torch.where(eigenvalues < 0, torch.zeros_like(eigenvalues), eigenvalues)
    eigenvalue_sum = eigenvalues.sum()
    if float(eigenvalue_sum) == 0.0:
        raise ValueError("residual covariance has zero eigenvalue sum")
    spectrum = eigenvalues / eigenvalue_sum
    positive = spectrum[spectrum > 0]
    effective_rank = torch.exp(-(positive * positive.log()).sum())
    max_eigenvalue = eigenvalues.max()
    numerical_rank = int((eigenvalues > 1e-6 * max_eigenvalue).sum())
    return {
        "count": int(count),
        "dimension": int(dimension),
        "std_min": float(std.min()),
        "std_mean": float(std.mean()),
        "std_max": float(std.max()),
        "std_values": [float(value) for value in std],
        "covariance_eigenvalues": [float(value) for value in eigenvalues],
        "normalized_eigenvalue_spectrum": [float(value) for value in spectrum],
        "effective_rank": float(effective_rank),
        "numerical_rank": numerical_rank,
        "zero_norm_count": int((torch.linalg.vector_norm(values, dim=1) == 0).sum()),
    }


def unit_normalize_zero(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-normalize rows, mapping exact zero norms to exact zero vectors."""

    matrix = _as_2d_float64(values)
    norms = torch.linalg.vector_norm(matrix, dim=1, keepdim=True)
    zero = norms.squeeze(1) == 0
    normalized = torch.where(zero.unsqueeze(1), torch.zeros_like(matrix), matrix / norms.clamp_min(1.0e-300))
    return normalized, zero


def nearest_candidate(
    values: torch.Tensor,
    reference_index: int,
    candidate_indices: Sequence[int],
    action_keys: Sequence[tuple[str, tuple[str, ...]]],
) -> tuple[int, float]:
    """Select nearest explicit candidate, breaking exact distance ties by action key."""

    matrix = _as_2d_float64(values)
    if len(action_keys) != matrix.size(0):
        raise ValueError("action_keys length must match number of rows")
    if not candidate_indices:
        raise ValueError("at least one nearest-neighbor candidate is required")
    ranked = []
    for index in candidate_indices:
        distance = float(torch.linalg.vector_norm(matrix[index] - matrix[reference_index]))
        ranked.append((distance, action_keys[index], index))
    distance, _, index = min(ranked)
    return index, distance


def _as_2d_float(latents: torch.Tensor) -> torch.Tensor:
    values = latents.detach().to(dtype=torch.float32, device="cpu")
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D latent tensor, got shape {tuple(values.shape)}")
    return values


def _as_2d_float64(latents: torch.Tensor) -> torch.Tensor:
    values = latents.detach().to(dtype=torch.float64, device="cpu")
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D latent tensor, got shape {tuple(values.shape)}")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("latent matrix contains non-finite values")
    return values


def _covariance(values: torch.Tensor) -> torch.Tensor:
    if values.size(0) == 0:
        return torch.zeros((values.size(1), values.size(1)), dtype=values.dtype)
    centered = values - values.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(values.size(0), 1)


def _offdiag_mean_sq(matrix: torch.Tensor) -> torch.Tensor:
    dim = matrix.size(0)
    if dim <= 1:
        return matrix.new_tensor(0.0)
    mask = ~torch.eye(dim, dtype=torch.bool)
    return matrix[mask].pow(2).mean()


def _effective_rank(eigenvalues: torch.Tensor) -> float:
    total = eigenvalues.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    probabilities = eigenvalues / total
    probabilities = probabilities[probabilities > 0]
    entropy = -(probabilities * probabilities.log()).sum()
    return _float(entropy.exp())


def _schema_indices(schema_ids: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, schema in enumerate(schema_ids):
        groups[str(schema)].append(idx)
    return groups


def _margin_group_indices(
    schema_ids: Sequence[str],
    group_ids: Sequence[Any] | None,
) -> dict[tuple[str, Any], list[int]]:
    groups: dict[tuple[str, Any], list[int]] = defaultdict(list)
    for idx, schema in enumerate(schema_ids):
        group_id = None if group_ids is None else group_ids[idx]
        groups[(str(schema), group_id)].append(idx)
    return groups


def _check_metadata(values: torch.Tensor, schema_ids: Sequence[str]) -> None:
    if len(schema_ids) != values.size(0):
        raise ValueError("schema_ids length must match number of latents")


def _json_action_key(action_key: Any) -> Any:
    if isinstance(action_key, tuple) and len(action_key) == 2:
        name, arguments = action_key
        return [name, list(arguments)]
    return action_key


def _json_scalar(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
