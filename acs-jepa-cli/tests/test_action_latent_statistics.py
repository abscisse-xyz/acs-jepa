from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = ROOT / "script" / "action_latent_statistics.py"
DIAG_PATH = ROOT / "script" / "diagnose_action_latent_statistics.py"


def _load_module(path: Path, name: str):
    import sys

    script_dir = str(ROOT / "script")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stats_module():
    return _load_module(STATS_PATH, "action_latent_statistics")


def test_latent_distribution_stats_reports_std_covariance_and_effective_rank() -> None:
    stats = _stats_module()
    latents = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    result = stats.latent_distribution_stats(latents)

    assert result["count"] == 4
    assert result["dim"] == 3
    assert result["std_mean"] > 0.0
    assert result["std_min"] > 0.0
    assert len(result["std_values"]) == 3
    assert result["cov_offdiag_mean_sq"] >= 0.0
    assert 1.0 <= result["effective_rank"] <= 3.0
    assert len(result["eigenvalues"]) == 3


def test_latent_distribution_stats_handles_constant_collapse() -> None:
    stats = _stats_module()

    result = stats.latent_distribution_stats(torch.ones(5, 4))

    assert result["count"] == 5
    assert result["dim"] == 4
    assert result["std_mean"] == 0.0
    assert result["std_min"] == 0.0
    assert result["std_values"] == [0.0, 0.0, 0.0, 0.0]
    assert result["cov_offdiag_mean_sq"] == 0.0
    assert result["effective_rank"] == 0.0
    assert result["eigenvalues"] == [0.0, 0.0, 0.0, 0.0]


def test_schema_group_stats_skips_under_supported_schemas() -> None:
    stats = _stats_module()
    latents = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
        ]
    )
    schema_ids = ["move", "move", "move", "build"]

    result = stats.schema_group_stats(latents, schema_ids, min_count=2)

    assert result["min_count"] == 2
    assert result["schema_count"] == 1
    assert result["skipped_schema_count"] == 1
    assert set(result["schemas"]) == {"move"}
    assert result["schemas"]["move"]["count"] == 3


def test_same_schema_nearest_wrong_margins_reports_distances_and_skips_singletons() -> None:
    stats = _stats_module()
    latents = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [10.0, 10.0],
        ]
    )
    schema_ids = ["move", "move", "build"]
    action_keys = [
        ("move", ("car0", "j0", "j1")),
        ("move", ("car0", "j0", "j2")),
        ("build", ("j0", "j1")),
    ]

    result = stats.same_schema_nearest_wrong_margins(latents, schema_ids, action_keys)

    assert result["count"] == 2
    assert result["skipped_singleton_count"] == 1
    assert result["nearest_wrong_distance_min"] == pytest.approx(5.0)
    assert result["nearest_wrong_distance_median"] == pytest.approx(5.0)
    assert result["details"][0]["nearest_wrong_action"] == ["move", ["car0", "j0", "j2"]]


def test_same_schema_nearest_wrong_margins_can_be_restricted_to_transition_groups() -> None:
    stats = _stats_module()
    latents = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.0],
            [3.0, 4.0],
        ]
    )
    schema_ids = ["move", "move", "move", "move"]
    group_ids = ["t0", "t1", "t1", "t1"]
    action_keys = [
        ("move", ("car0", "j0")),
        ("move", ("car0", "j1")),
        ("move", ("car0", "j2")),
        ("move", ("car0", "j3")),
    ]

    result = stats.same_schema_nearest_wrong_margins(latents, schema_ids, action_keys, group_ids=group_ids)

    assert result["count"] == 3
    assert result["skipped_singleton_count"] == 1
    assert result["nearest_wrong_distance_min"] == pytest.approx(0.1)


def test_reference_same_schema_margins_report_true_action_to_nearest_wrong() -> None:
    stats = _stats_module()
    latents = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [10.0, 10.0],
            [10.0, 12.0],
            [0.1, 0.0],
        ]
    )
    schema_ids = ["move", "move", "build", "build", "move"]
    group_ids = ["t0", "t0", "t1", "t1", "t0"]
    reference_mask = [True, False, True, False, False]
    action_keys = [
        ("move", ("car0", "j0")),
        ("move", ("car0", "j1")),
        ("build", ("j0", "j1")),
        ("build", ("j0", "j2")),
        ("move", ("car0", "j2")),
    ]

    result = stats.reference_same_schema_margins(
        latents,
        schema_ids,
        action_keys,
        reference_mask,
        group_ids,
    )

    assert result["count"] == 2
    assert result["skipped_no_wrong_count"] == 0
    assert result["nearest_wrong_distance_min"] == pytest.approx(0.1)
    assert result["nearest_wrong_distance_median"] == pytest.approx(1.05)
    assert result["details"][0]["reference_action"] == ["move", ["car0", "j0"]]


def test_diagnostic_reference_margins_add_scale_robust_unit_l2_without_changing_raw() -> None:
    stats = _stats_module()
    diagnostic = _load_module(DIAG_PATH, "diagnose_action_latent_statistics_unit")
    latents = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    schema_ids = ["move", "move"]
    action_keys = [("move", ("a",)), ("move", ("b",))]
    reference_mask = [True, False]
    group_ids = ["t0", "t0"]

    expected_raw = stats.reference_same_schema_margins(
        latents, schema_ids, action_keys, reference_mask, group_ids
    )
    result = diagnostic._reference_same_schema_margin_metrics(
        latents, schema_ids, action_keys, reference_mask, group_ids
    )

    assert {key: result[key] for key in expected_raw} == expected_raw
    assert result["unit_l2"]["nearest_wrong_distance_min"] == pytest.approx(2**0.5)


def test_diagnostic_reference_unit_l2_is_finite_for_zero_norm_latents() -> None:
    diagnostic = _load_module(DIAG_PATH, "diagnose_action_latent_statistics_zero")

    result = diagnostic._reference_same_schema_margin_metrics(
        torch.tensor([[0.0, 0.0], [3.0, 4.0]]),
        ["move", "move"],
        [("move", ("a",)), ("move", ("b",))],
        [True, False],
        ["t0", "t0"],
    )

    assert result["unit_l2"]["nearest_wrong_distance_min"] == pytest.approx(1.0)


def test_compact_summary_recursively_omits_detail_arrays_without_mutating_input() -> None:
    diagnostic = _load_module(DIAG_PATH, "diagnose_action_latent_statistics_compact")
    payload = {
        "metrics": {"count": 2},
        "raw": {"details": [{"row": 1}], "unit_l2": {"details": [{"row": 2}], "median": 1.0}},
    }

    compact = diagnostic._without_details(payload)

    assert compact == {"metrics": {"count": 2}, "raw": {"unit_l2": {"median": 1.0}}}
    assert payload["raw"]["details"] == [{"row": 1}]


def test_schema_argument_variance_decomposition_reports_between_and_within_variance() -> None:
    stats = _stats_module()
    latents = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [10.0, 0.0],
            [10.0, 2.0],
        ]
    )
    schema_ids = ["move", "move", "build", "build"]

    result = stats.schema_argument_variance_decomposition(latents, schema_ids)

    assert result["schema_count"] == 2
    assert result["between_schema_variance"] > 0.0
    assert result["within_schema_variance"] > 0.0
    assert result["between_fraction"] > result["within_fraction"]
    assert "within_schema_variance includes argument and source-state/context variation" in result["note"]


def test_diagnostic_script_imports_without_running_main() -> None:
    module = _load_module(DIAG_PATH, "diagnose_action_latent_statistics")

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "data",
            "--checkpoint",
            "checkpoint.pt",
            "--output",
            "out",
            "--max-transitions",
            "2",
            "--max-candidates-per-state",
            "128",
            "--same-schema-only",
            "--omit-details",
        ]
    )

    assert args.dataset_dir == Path("data")
    assert args.checkpoint == Path("checkpoint.pt")
    assert args.output == Path("out")
    assert args.max_transitions == 2
    assert args.max_candidates_per_state == 128
    assert args.same_schema_only is True
    assert args.omit_details is True


def test_diagnostic_argument_validation_rejects_non_positive_chunk_size() -> None:
    module = _load_module(DIAG_PATH, "diagnose_action_latent_statistics")
    args = module.build_parser().parse_args(
        [
            "data",
            "--checkpoint",
            "checkpoint.pt",
            "--output",
            "out",
            "--chunk-size",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="--chunk-size must be positive"):
        module.validate_args(args)
