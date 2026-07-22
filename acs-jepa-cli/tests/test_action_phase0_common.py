from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "script" / "action_phase0_common.py"
FIXED_MANIFEST = Path(
    "/opt/data/workspace/acs-jepa-runs/smoke/action_auxiliary_seed0/phase2g/"
    "baseline/probe_run1/example_manifest.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("action_phase0_common", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def test_fixed_candidate_manifest_validates_exact_identity_without_simulator_import() -> None:
    common = _module()

    records, identity = common.load_and_validate_candidate_manifest(FIXED_MANIFEST)

    assert len(records) == 604
    assert identity == {
        "path": str(FIXED_MANIFEST),
        "bytes": 117385,
        "sha256": "bf6d11149cadf7a34c6c1520e28e9fe389c09c13ce53f3bd3f988f827e936ce9",
        "count": 604,
    }
    tree = ast.parse(MODULE_PATH.read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("simulator" not in ast.unparse(node) for node in imports)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda rows: rows.append(rows[0]), "count"),
        (lambda rows: rows[0].__setitem__("extra", 1), "keys"),
        (lambda rows: rows[0]["action"].__setitem__("extra", 1), "action keys"),
        (lambda rows: rows[0].__setitem__("group", "p166:99"), "group"),
        (lambda rows: rows[0].__setitem__("step", 99), "group/step"),
        (lambda rows: rows[0].__setitem__("category", "trace"), "trace"),
        (lambda rows: rows[0].__setitem__("applicability_label", True), "label"),
        (lambda rows: rows.__setitem__(1, rows[0]), "duplicate record"),
    ],
)
def test_manifest_validator_rejects_structural_and_count_drift(tmp_path: Path, mutation, message: str) -> None:
    common = _module()
    rows = json.loads(FIXED_MANIFEST.read_bytes())
    mutation(rows)
    path = tmp_path / "manifest.json"
    path.write_bytes(_canonical(rows))

    with pytest.raises(ValueError, match=message):
        common.load_and_validate_candidate_manifest(path, require_fixed_identity=False)


def test_manifest_validator_rejects_duplicate_keys_and_noncanonical_bytes(tmp_path: Path) -> None:
    common = _module()
    path = tmp_path / "manifest.json"
    path.write_text('[{"action":{},"action":{}}]\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        common.load_and_validate_candidate_manifest(path, require_fixed_identity=False)

    path.write_bytes(json.dumps(json.loads(FIXED_MANIFEST.read_bytes()), indent=2).encode())
    with pytest.raises(ValueError, match="canonical"):
        common.load_and_validate_candidate_manifest(path, require_fixed_identity=False)


def test_manifest_validator_rejects_changed_fixed_hash_even_when_structurally_valid(tmp_path: Path) -> None:
    common = _module()
    rows = json.loads(FIXED_MANIFEST.read_bytes())
    rows[0], rows[1] = rows[1], rows[0]
    path = tmp_path / "manifest.json"
    path.write_bytes(_canonical(rows))

    with pytest.raises(ValueError, match="identity"):
        common.load_and_validate_candidate_manifest(path)


def test_reconcile_source_states_checks_group_step_and_trace_identity() -> None:
    common = _module()
    trace = {"name": "move", "arguments": ["car0", "j0"]}
    records = [
        {
            "group": "p1:0",
            "problem": "p1",
            "step": 0,
            "category": "trace",
            "applicability_label": True,
            "action": trace,
        },
        {
            "group": "p1:0",
            "problem": "p1",
            "step": 0,
            "category": "random_same_schema",
            "applicability_label": False,
            "action": {"name": "move", "arguments": ["car0", "j1"]},
        },
    ]
    action = SimpleNamespace(name="move", arguments=("car0", "j0"))
    trajectory = SimpleNamespace(problem_index=0, actions=(action,), states=(("s0",), ("s1",)))
    corpus = SimpleNamespace(
        records=(SimpleNamespace(problem_name="p1", start_step_index=7),),
        trajectories=(trajectory,),
        parsed_problems=("parsed",),
    )

    sources = common.reconcile_manifest_source_states(records, corpus)

    assert len(sources) == 1
    assert sources[0].group == "p1:0"
    assert sources[0].source_atoms == ("s0",)
    assert sources[0].trace_action is action

    bad = [dict(row) for row in records]
    bad[0] = {**bad[0], "action": {"name": "move", "arguments": ["wrong", "j0"]}}
    with pytest.raises(ValueError, match="trace action"):
        common.reconcile_manifest_source_states(bad, corpus)


def test_population_identity_preserves_order_and_repeated_action_keys() -> None:
    common = _module()
    rows = [
        {"group": "p:0", "problem": "p", "step": 0, "action": {"name": "move", "arguments": ["a"]}},
        {"group": "p:1", "problem": "p", "step": 1, "action": {"name": "move", "arguments": ["a"]}},
    ]
    expected = b"".join(_canonical(row) for row in rows)

    assert common.population_identity(rows, expected_count=2) == {
        "count": 2,
        "bytes": len(expected),
        "sha256": hashlib.sha256(expected).hexdigest(),
    }
    with pytest.raises(ValueError, match="population count"):
        common.population_identity(rows, expected_count=3)


def test_actual_fixed_sources_enumerate_exact_population_identity() -> None:
    common = _module()
    from acs_jepa_cli.config import load_config
    from acs_jepa_cli.data import load_corpus, split_corpus

    records, _ = common.load_and_validate_candidate_manifest(FIXED_MANIFEST)
    config = load_config(Path("/opt/data/workspace/acs-jepa-runs/smoke/default_seed0/config.yaml"))
    corpus = load_corpus([Path("/opt/data/workspace/acs-jepa-tuning-data/smoke")], strict=True)
    selected = split_corpus(
        corpus,
        val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction,
        seed=20260717,
    ).val
    sources = common.reconcile_manifest_source_states(records, selected)

    buckets = common.enumerate_same_schema_population(sources)
    rows = [row for bucket in buckets for row in bucket.identity_rows]

    assert len(sources) == 44
    assert len(buckets) == 44
    assert common.population_identity(rows) == {
        "count": 174780,
        "bytes": 24455400,
        "sha256": "c6b056d8a976a77c994338aeceedcc519a7506d770e6921d08d00da4753e97d2",
    }
    assert all(len(bucket.actions) >= 2 for bucket in buckets)


def test_root_protocol_creates_marker_first_then_refuses_changed_or_existing_destinations(tmp_path: Path) -> None:
    common = _module()
    root = tmp_path / "phase0"
    first_output = root / "schema_residual" / "baseline" / "run1"
    marker = {
        "schema_version": "action_latent_updated_phase0.root_identity.v1",
        "updated_spec": {"path": "/spec", "bytes": 1, "sha256": "a" * 64},
        "baseline_checkpoint": {"path": "/b.pt", "bytes": 1, "sha256": "b" * 64},
        "baseline_config": {"path": "/b.yaml", "bytes": 1, "sha256": "c" * 64},
        "phase2_checkpoint": {"path": "/p.pt", "bytes": 1, "sha256": "d" * 64},
        "phase2_config": {"path": "/p.yaml", "bytes": 1, "sha256": "e" * 64},
        "corpus_manifest": {"path": "/corpus", "bytes": 1, "sha256": "f" * 64, "count": 12},
        "candidate_manifest": {"path": "/candidates", "bytes": 1, "sha256": "0" * 64, "count": 604},
        "split_sha256": common.SPLIT_SHA256,
        "created_by": "schema_residual/baseline/run1",
    }

    common.prepare_output_directory(root, first_output, marker, first_command=True)

    assert (root / "root_identity.json").read_bytes() == _canonical(marker)
    assert first_output.is_dir()
    with pytest.raises(FileExistsError, match="destination"):
        common.prepare_output_directory(root, first_output, marker, first_command=False)

    second = root / "schema_residual" / "baseline" / "run2"
    changed = {**marker, "split_sha256": "1" * 64}
    with pytest.raises(ValueError, match="root identity"):
        common.prepare_output_directory(root, second, changed, first_command=False)
    assert not second.exists()


def test_root_protocol_requires_absent_root_for_first_and_marker_for_later(tmp_path: Path) -> None:
    common = _module()
    root = tmp_path / "phase0"
    root.mkdir()
    with pytest.raises(FileExistsError, match="root"):
        common.prepare_output_directory(root, root / "out", {}, first_command=True)
    with pytest.raises(ValueError, match="root identity"):
        common.prepare_output_directory(root, root / "out", {}, first_command=False)
