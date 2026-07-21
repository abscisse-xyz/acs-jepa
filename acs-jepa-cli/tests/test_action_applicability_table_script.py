from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest
from acs_jepa.graph.schemas import GroundAction
from acs_jepa_cli.data import load_action_applicability_table, load_corpus

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "script" / "build_action_applicability_table.py"

DOMAIN = """
(define (domain tiny)
  (:requirements :typing)
  (:types item place)
  (:predicates (at ?x - item ?p - place) (clear ?p - place))
  (:action move
    :parameters (?x - item ?from - place ?to - place)
    :precondition (and (at ?x ?from) (clear ?to))
    :effect (and
      (not (at ?x ?from)) (at ?x ?to)
      (clear ?from) (not (clear ?to))))
)
"""

PROBLEM = """
(define (problem internal-problem-name)
  (:domain tiny)
  (:objects box-a box-b - item p0 p1 - place)
  (:init (at box-a p0) (at box-b p0) (clear p1))
  (:goal (at box-a p1)))
"""

INITIAL_FACTS = [
    {"name": "at", "arguments": ["box-a", "p0"]},
    {"name": "at", "arguments": ["box-b", "p0"]},
    {"name": "clear", "arguments": ["p1"]},
]
TERMINAL_FACTS = [
    {"name": "at", "arguments": ["box-a", "p1"]},
    {"name": "at", "arguments": ["box-b", "p0"]},
    {"name": "clear", "arguments": ["p0"]},
]


def test_standalone_producer_is_deterministic_strict_and_complete(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_run = _run(dataset, first)
    second_run = _run(dataset, second)

    assert first_run.returncode == 0, first_run.stderr
    assert second_run.returncode == 0, second_run.stderr
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["semantics"] == "positive_ground_atoms_closed_world_v1"
    assert len(payload["entries"]) == 1  # Equal states from two trajectories agree and deduplicate.
    entry = payload["entries"][0]
    assert entry["problem_index"] == 1  # Keep the sparse filename alias index from load_corpus.
    assert entry["problem_name"] == "internal-problem-name"
    assert entry["applicable_actions"] == [
        {"arguments": ["box-a", "p0", "p1"], "name": "move"},
        {"arguments": ["box-b", "p0", "p1"], "name": "move"},
    ]
    assert entry["state_atoms"] == [
        {"arguments": ["box-a", "p0"], "predicate": "at"},
        {"arguments": ["box-b", "p0"], "predicate": "at"},
        {"arguments": ["p1"], "predicate": "clear"},
    ]

    corpus = load_corpus([dataset], strict=True)
    table = load_action_applicability_table(first, corpus.parsed_problems)
    assert len(table) == 1
    actions = next(iter(table.values()))
    assert actions == frozenset(
        {
            GroundAction("move", ("box-a", "p0", "p1")),
            GroundAction("move", ("box-b", "p0", "p1")),
        }
    )


def test_standalone_producer_refuses_relative_output_before_loading_data(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path / "missing-dataset", "relative.json")

    assert result.returncode != 0
    assert "--output must be an absolute path" in result.stderr
    assert not (ROOT / "relative.json").exists()


@pytest.mark.parametrize(
    ("column", "facts", "expected"),
    [
        ("state_facts", TERMINAL_FACTS, "source state mismatch"),
        ("next_state_facts", INITIAL_FACTS, "terminal state mismatch"),
    ],
)
def test_standalone_producer_rejects_state_mismatches_with_transition_context(
    tmp_path: Path,
    column: str,
    facts: list[dict[str, object]],
    expected: str,
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    with duckdb.connect(str(dataset / "simulation" / "simulation.duckdb")) as con:
        con.execute(
            f"UPDATE state_action_transitions SET {column} = ?",  # noqa: S608 - fixed test parametrization.
            [facts],
        )

    result = _run(dataset, tmp_path / "labels.json")

    assert result.returncode != 0
    assert expected in result.stderr
    _assert_transition_context(result.stderr, dataset)


def test_standalone_producer_rejects_recorded_action_outside_full_set(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    with duckdb.connect(str(dataset / "simulation" / "simulation.duckdb")) as con:
        con.execute(
            "UPDATE state_action_transitions SET arguments = "
            "['box-a', 'p1', 'p0']"
        )

    result = _run(dataset, tmp_path / "labels.json")

    assert result.returncode != 0
    assert "recorded action" in result.stderr
    assert "complete applicable-action set" in result.stderr
    _assert_transition_context(result.stderr, dataset)


def test_replay_failure_is_contextual_and_uses_finish_true(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    corpus = load_corpus([dataset], strict=True)
    module = _load_script_module()
    calls: list[tuple[str, tuple[str, ...], bool]] = []

    class FailingReplayEngine:
        def current_facts(self):
            return tuple(
                (fact["name"], *fact["arguments"])
                for fact in INITIAL_FACTS
            )

        def applicable_actions(self):
            return (
                SimpleNamespace(
                    name="move",
                    arguments=("box-a", "p0", "p1"),
                ),
            )

        def apply_action(self, name, arguments, *, finish):
            calls.append((name, arguments, finish))
            raise RuntimeError("synthetic replay failure")

    with pytest.raises(ValueError, match="replay failed") as error:
        module.build_payload(
            dataset,
            corpus=corpus,
            engine_factory=lambda _domain, _problem: FailingReplayEngine(),
        )

    assert calls == [("move", ("box-a", "p0", "p1"), True)]
    _assert_transition_context(str(error.value), dataset)


def test_duplicate_state_contradiction_is_rejected_at_unit_seam() -> None:
    module = _load_script_module()
    entries = {}
    state = (("at", "box-a", "p0"),)
    module._merge_entry(
        entries,
        problem_index=7,
        problem_name="internal-name",
        state_facts=state,
        applicable_actions=(("move", ("box-a", "p0", "p1")),),
        context="first observation",
    )

    with pytest.raises(ValueError, match="contradictory applicable-action sets") as error:
        module._merge_entry(
            entries,
            problem_index=7,
            problem_name="internal-name",
            state_facts=state,
            applicable_actions=(("wait", ()),),
            context="dataset=d problem=p trajectory=2 step=4",
        )

    assert "trajectory=2 step=4" in str(error.value)


def test_production_cli_and_core_do_not_import_offline_producer() -> None:
    production_roots = (
        ROOT / "acs-jepa-cli" / "src",
        ROOT / "packages" / "acs-jepa-core" / "src",
    )
    for production_root in production_roots:
        for path in production_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "build_action_applicability_table" not in source, path


def _run(dataset: Path, output: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(dataset), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "build_action_applicability_table_for_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_transition_context(message: str, dataset: Path) -> None:
    assert f"dataset={dataset}" in message
    assert "problem='alias-file'" in message
    assert "trajectory=0" in message
    assert "run=11" in message
    assert "step=" in message


def _dataset(root: Path) -> Path:
    problem_dir = root / "problem"
    simulation_dir = root / "simulation"
    problem_dir.mkdir(parents=True)
    simulation_dir.mkdir(parents=True)
    (problem_dir / "domain.pddl").write_text(DOMAIN, encoding="utf-8")
    (problem_dir / "alias-file.pddl").write_text(PROBLEM, encoding="utf-8")
    with duckdb.connect(str(simulation_dir / "simulation.duckdb")) as con:
        con.execute(
            "CREATE TABLE simulation_runs(id BIGINT, domain_name VARCHAR, "
            "problem_name VARCHAR, created_at TIMESTAMP)"
        )
        con.execute(
            "CREATE TABLE planner_attempts(run_id BIGINT, status VARCHAR, "
            "failure_stage VARCHAR, plan_topology VARCHAR)"
        )
        con.execute(
            """
            CREATE TABLE state_action_transitions(
                run_id BIGINT,
                step_index INTEGER,
                sim_time VARCHAR,
                phase VARCHAR,
                action_name VARCHAR,
                action_kind VARCHAR,
                arguments VARCHAR[],
                duration VARCHAR,
                state_facts STRUCT(name VARCHAR, arguments VARCHAR[])[],
                state_numeric_values STRUCT(name VARCHAR, arguments VARCHAR[], value VARCHAR)[],
                state_running_actions STRUCT(name VARCHAR, arguments VARCHAR[], started_at VARCHAR, ends_at VARCHAR)[],
                next_state_facts STRUCT(name VARCHAR, arguments VARCHAR[])[],
                next_state_numeric_values STRUCT(name VARCHAR, arguments VARCHAR[], value VARCHAR)[],
                next_state_running_actions STRUCT(
                    name VARCHAR,
                    arguments VARCHAR[],
                    started_at VARCHAR,
                    ends_at VARCHAR
                )[]
            )
            """
        )
        con.execute(
            "INSERT INTO simulation_runs VALUES "
            "(11, 'tiny', 'alias-file', CURRENT_TIMESTAMP), "
            "(12, 'tiny', 'alias-file', CURRENT_TIMESTAMP)"
        )
        con.execute(
            "INSERT INTO planner_attempts VALUES "
            "(11, 'SOLVED_SATISFICING', 'completed', 'SequentialPlan'), "
            "(12, 'SOLVED_OPTIMALLY', 'completed', 'SequentialPlan')"
        )
        for run_id in (11, 12):
            con.execute(
                """
                INSERT INTO state_action_transitions VALUES (
                    ?, 3, '0/1', 'apply', 'move', 'instantaneous',
                    ['box-a', 'p0', 'p1'], '0/1', ?, [], [], ?, [], []
                )
                """,
                [run_id, INITIAL_FACTS, TERMINAL_FACTS],
            )
    return root
