from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = ROOT / "script" / "action_latent_statistics.py"
DIAG_PATH = ROOT / "script" / "diagnose_action_schema_residuals.py"


def _load(path: Path, name: str):
    script = str(ROOT / "script")
    if script not in sys.path:
        sys.path.insert(0, script)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_global_and_state_schema_residuals_use_distinct_exact_buckets() -> None:
    stats = _load(STATS_PATH, "action_latent_statistics_phase0_centering")
    latents = torch.tensor([[0.0, 0.0], [2.0, 0.0], [10.0, 2.0], [14.0, 2.0]])
    schemas = ["move"] * 4
    groups = ["g0", "g0", "g1", "g1"]

    global_residual, state_residual = stats.schema_residuals(latents, schemas, groups)

    assert global_residual.dtype == torch.float64
    assert state_residual.dtype == torch.float64
    assert torch.allclose(global_residual.mean(0), torch.zeros(2, dtype=torch.float64), atol=1e-7)
    assert torch.allclose(state_residual[:2].mean(0), torch.zeros(2, dtype=torch.float64), atol=1e-7)
    assert torch.allclose(state_residual[2:].mean(0), torch.zeros(2, dtype=torch.float64), atol=1e-7)
    assert not torch.equal(global_residual, state_residual)


def test_state_schema_centering_rejects_singletons_and_does_not_mix_schemas() -> None:
    stats = _load(STATS_PATH, "action_latent_statistics_phase0_buckets")
    with pytest.raises(ValueError, match="singleton"):
        stats.schema_residuals(torch.tensor([[0.0], [1.0]]), ["a", "b"], ["g", "g"])

    latents = torch.tensor([[0.0], [2.0], [10.0], [14.0]])
    _, residual = stats.schema_residuals(latents, ["a", "a", "b", "b"], ["g", "g", "g", "g"])
    assert residual[:, 0].tolist() == pytest.approx([-1.0, 1.0, -2.0, 2.0])


def test_float64_raw_variance_decomposition_matches_oracle_and_reconstructs() -> None:
    stats = _load(STATS_PATH, "action_latent_statistics_phase0_variance")
    latents = torch.tensor([[0.0], [2.0], [10.0], [12.0]], dtype=torch.float32)

    result = stats.raw_variance_decomposition(latents, ["a", "a", "b", "b"])

    assert result == pytest.approx(
        {
            "total_variance": 26.0,
            "between_schema_variance": 25.0,
            "within_schema_variance": 1.0,
            "between_schema_fraction": 25.0 / 26.0,
            "within_schema_fraction": 1.0 / 26.0,
            "reconstruction_absolute_error": 0.0,
        }
    )
    with pytest.raises(ValueError, match="total variance"):
        stats.raw_variance_decomposition(torch.ones(3, 2), ["a"] * 3)


def test_residual_statistics_uses_uncentered_population_covariance_and_rank_oracle() -> None:
    stats = _load(STATS_PATH, "action_latent_statistics_phase0_stats")
    residuals = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]])

    result = stats.residual_statistics(residuals)

    assert set(result) == {
        "count",
        "dimension",
        "std_min",
        "std_mean",
        "std_max",
        "std_values",
        "covariance_eigenvalues",
        "normalized_eigenvalue_spectrum",
        "effective_rank",
        "numerical_rank",
        "zero_norm_count",
    }
    assert result["covariance_eigenvalues"] == pytest.approx([2.0, 0.5])
    assert result["normalized_eigenvalue_spectrum"] == pytest.approx([0.8, 0.2])
    assert result["effective_rank"] == pytest.approx(math.exp(-(0.8 * math.log(0.8) + 0.2 * math.log(0.2))))
    assert result["numerical_rank"] == 2
    assert result["zero_norm_count"] == 0
    with pytest.raises(ValueError, match="zero eigenvalue sum"):
        stats.residual_statistics(torch.zeros(2, 2))


def test_zero_norm_unit_normalization_is_finite_zero_vector() -> None:
    stats = _load(STATS_PATH, "action_latent_statistics_phase0_normalize")
    normalized, zero = stats.unit_normalize_zero(torch.tensor([[0.0, 0.0], [3.0, 4.0]]))

    assert normalized.dtype == torch.float64
    assert torch.allclose(normalized, torch.tensor([[0.0, 0.0], [0.6, 0.8]], dtype=torch.float64))
    assert zero.tolist() == [True, False]
    assert torch.isfinite(normalized).all()


def test_nearest_candidate_ties_break_canonical_action_key_and_stay_in_explicit_bucket() -> None:
    stats = _load(STATS_PATH, "action_latent_statistics_phase0_nearest")
    values = torch.tensor([[0.0], [1.0], [-1.0], [0.1]])
    keys = [("move", ("trace",)), ("move", ("b",)), ("move", ("a",)), ("move", ("other-state",))]

    index, distance = stats.nearest_candidate(values, 0, [1, 2], keys)

    assert index == 2
    assert distance == pytest.approx(1.0)


def test_schema_residual_extraction_has_no_simulator_or_oracle_dependency() -> None:
    for path in (DIAG_PATH, ROOT / "script" / "action_phase0_common.py"):
        tree = ast.parse(path.read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        imported = {ast.unparse(node) for node in imports}
        assert all("simulator" not in statement for statement in imported)
        assert all("action_diag_common" not in statement for statement in imported)
        source = path.read_text()
        assert "applicable_actions(" not in source
        assert "replay_trajectory(" not in source
        assert "_build_simulator_engine" not in source


def test_schema_residual_parser_defaults_and_nonpositive_chunk_rejection() -> None:
    diagnostic = _load(DIAG_PATH, "diagnose_action_schema_residuals_parser")
    args = diagnostic.build_parser().parse_args(
        ["data", "--checkpoint", "checkpoint.pt", "--candidate-manifest", "manifest.json", "--output", "out"]
    )

    assert args.dataset_dir == Path("data")
    assert args.checkpoint == Path("checkpoint.pt")
    assert args.candidate_manifest == Path("manifest.json")
    assert args.output == Path("out")
    assert args.device == "cuda"
    assert args.split == "val"
    assert args.chunk_size == 2048
    assert args.seed == 20260717

    args.chunk_size = 0
    with pytest.raises(ValueError, match="--chunk-size must be positive"):
        diagnostic.validate_args(args)


def test_distance_details_use_same_state_only_offline_invalids_and_canonical_ties() -> None:
    diagnostic = _load(DIAG_PATH, "diagnose_action_schema_residuals_details")
    raw = torch.tensor([[0.0], [1.0], [-1.0], [0.1], [0.2], [10.0]])
    state_residual = torch.tensor([[0.0], [1.0], [-1.0], [-0.05], [0.05], [9.0]])
    action_keys = [
        ("move", ("trace0",)),
        ("move", ("b",)),
        ("move", ("a",)),
        ("move", ("trace1",)),
        ("move", ("invalid1",)),
        ("move", ("far1",)),
    ]
    groups = ["g0", "g0", "g0", "g1", "g1", "g1"]
    sources = [
        {"group": "g0", "problem": "p", "step": 0, "trace": 0, "invalid": [1]},
        {"group": "g1", "problem": "p", "step": 1, "trace": 3, "invalid": [4]},
    ]

    details = diagnostic.distance_diagnostics(raw, state_residual, action_keys, groups, sources)

    assert details[0]["nearest_wrong_action"] == {"name": "move", "arguments": ["a"]}
    assert details[0]["nearest_wrong_raw_l2"] == pytest.approx(1.0)
    assert details[0]["nearest_invalid_action"] == {"name": "move", "arguments": ["b"]}
    assert details[0]["trace_zero_residual_norm"] is True
    assert details[1]["nearest_wrong_raw_l2"] == pytest.approx(0.1)
    assert details[1]["nearest_invalid_unit_residual_l2"] == pytest.approx(2.0)


def test_extraction_determinism_is_enabled_from_seed() -> None:
    diagnostic = _load(DIAG_PATH, "diagnose_action_schema_residuals_determinism")
    torch.use_deterministic_algorithms(False)

    diagnostic.configure_determinism(123)
    first = torch.rand(3)
    diagnostic.configure_determinism(123)
    second = torch.rand(3)

    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.equal(first, second)


def test_checkpoint_is_strictly_bound_to_baseline_or_phase2_output_slot() -> None:
    diagnostic = _load(DIAG_PATH, "diagnose_action_schema_residuals_binding")
    root = Path("/tmp/local/updated_phase0")

    diagnostic.validate_run_binding(
        diagnostic.BASELINE_CHECKPOINT,
        root / "schema_residual" / "baseline" / "run1",
    )
    diagnostic.validate_run_binding(
        diagnostic.PHASE2_CHECKPOINT,
        root / "schema_residual" / "phase2" / "run2",
    )
    with pytest.raises(ValueError, match="checkpoint/output binding"):
        diagnostic.validate_run_binding(
            diagnostic.PHASE2_CHECKPOINT,
            root / "schema_residual" / "baseline" / "run1",
        )
    with pytest.raises(ValueError, match="run1 or run2"):
        diagnostic.validate_run_binding(
            diagnostic.BASELINE_CHECKPOINT,
            root / "schema_residual" / "baseline" / "other",
        )
