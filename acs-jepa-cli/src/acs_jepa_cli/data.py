"""Dataset ingestion from PDDL simulator DuckDB outputs."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import duckdb
from acs_jepa.graph import (
    ATOM_STATE_APPLICABILITY_SEMANTICS,
    ActionApplicabilityTable,
    ActionSupervisionConfig,
    PDDLAtomTrajectoryDataset,
    PDDLTrajectoryDataset,
    TrajectorySample,
    action_applicability_state_key,
    parse_domain_problem,
)
from acs_jepa.graph.schemas import GroundAction, GroundAtom, ParsedProblem

RELEVANT_TRANSITIONS_SQL = """
SELECT
    r.domain_name,
    r.problem_name,
    t.run_id,
    t.step_index,
    t.sim_time,
    t.phase,
    t.action_name,
    t.action_kind,
    t.arguments,
    t.duration,
    t.state_facts,
    t.state_numeric_values,
    t.state_running_actions,
    t.next_state_facts,
    t.next_state_numeric_values,
    t.next_state_running_actions
FROM state_action_transitions t
JOIN simulation_runs r
  ON r.id = t.run_id
WHERE EXISTS (
    SELECT 1
    FROM planner_attempts p
    WHERE p.run_id = r.id
      AND p.status IN ('SOLVED_SATISFICING', 'SOLVED_OPTIMALLY')
      AND p.plan_topology = 'SequentialPlan'
)
ORDER BY t.run_id, t.step_index
"""


REQUIRED_TABLES = {"simulation_runs", "planner_attempts", "state_action_transitions"}


@dataclass(frozen=True)
class DatasetSource:
    dataset_id: int
    root: Path
    domain_path: Path
    problem_dir: Path
    database_path: Path


@dataclass(frozen=True)
class TrajectoryRecord:
    dataset_id: int
    dataset_root: str
    run_id: int
    problem_name: str
    start_step_index: int
    num_actions: int


@dataclass(frozen=True)
class LoadedCorpus:
    parsed_problems: tuple[ParsedProblem, ...]
    trajectories: tuple[TrajectorySample, ...]
    records: tuple[TrajectoryRecord, ...]
    dataset_summaries: tuple[dict[str, Any], ...]
    malformed_rows: tuple[str, ...]


@dataclass(frozen=True)
class SplitCorpus:
    train: LoadedCorpus
    val: LoadedCorpus
    test: LoadedCorpus
    manifest: dict[str, list[dict[str, Any]]]


def discover_dataset_sources(dataset_dirs: Sequence[str | Path]) -> tuple[DatasetSource, ...]:
    """Validate dataset roots and return normalized source descriptors."""

    sources: list[DatasetSource] = []
    for dataset_id, raw_root in enumerate(dataset_dirs):
        root = Path(raw_root)
        problem_dir = root / "problem"
        domain_path = problem_dir / "domain.pddl"
        database_path = root / "simulation" / "simulation.duckdb"
        missing = [
            str(path)
            for path in (problem_dir, domain_path, database_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Dataset {root} is missing required paths: {missing}")
        sources.append(
            DatasetSource(
                dataset_id=dataset_id,
                root=root,
                domain_path=domain_path,
                problem_dir=problem_dir,
                database_path=database_path,
            )
        )
    return tuple(sources)


def load_corpus(dataset_dirs: Sequence[str | Path], *, strict: bool = True) -> LoadedCorpus:
    """Load relevant simulator trajectories from one or more dataset directories."""

    sources = discover_dataset_sources(dataset_dirs)
    parsed_problems: list[ParsedProblem] = []
    problem_index_by_source_name: dict[tuple[int, str], int] = {}
    trajectories: list[TrajectorySample] = []
    records: list[TrajectoryRecord] = []
    summaries: list[dict[str, Any]] = []
    malformed: list[str] = []

    for source in sources:
        _validate_duckdb_schema(source.database_path)
        source_problem_map = _parse_source_problems(source)
        for problem_name, parsed in source_problem_map.items():
            key = (source.dataset_id, problem_name)
            if key in problem_index_by_source_name:
                continue
            problem_index_by_source_name[key] = len(parsed_problems)
            parsed_problems.append(parsed)

        rows = _query_transition_rows(source.database_path)
        relevant_transition_count = 0
        relevant_trajectory_count = 0
        action_counts: dict[str, int] = {}
        problem_names: set[str] = set()
        grouped_rows: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for row in rows:
            try:
                problem_name = str(row["problem_name"])
                grouped_rows.setdefault((int(row["run_id"]), problem_name), []).append(row)
                problem_names.add(problem_name)
            except Exception as exc:  # noqa: BLE001 - report row-level ingestion failures.
                message = f"{source.root}: run={row.get('run_id')} step={row.get('step_index')}: {exc}"
                malformed.append(message)
                if strict:
                    raise ValueError(message) from exc

        for (run_id, problem_name), run_rows in sorted(grouped_rows.items()):
            run_rows = sorted(run_rows, key=lambda item: int(item["step_index"]))
            try:
                parsed_index = problem_index_by_source_name[(source.dataset_id, problem_name)]
                states: list[tuple[GroundAtom, ...]] = []
                actions: list[GroundAction] = []
                previous_next_atoms: tuple[GroundAtom, ...] | None = None
                for row in run_rows:
                    state_atoms = _facts_to_atoms(row["state_facts"])
                    next_atoms = _facts_to_atoms(row["next_state_facts"])
                    if previous_next_atoms is not None and state_atoms != previous_next_atoms:
                        raise ValueError(
                            "State continuity mismatch at "
                            f"step {int(row['step_index'])}: state_facts != previous next_state_facts"
                        )
                    if not states:
                        states.append(state_atoms)
                    action = GroundAction(
                        name=str(row["action_name"]),
                        arguments=tuple(str(arg) for arg in (row["arguments"] or ())),
                    )
                    actions.append(action)
                    states.append(next_atoms)
                    previous_next_atoms = next_atoms
                    relevant_transition_count += 1
                    action_counts[action.name] = action_counts.get(action.name, 0) + 1
                if actions:
                    trajectories.append(
                        TrajectorySample(
                            problem_index=parsed_index,
                            states=tuple(states),
                            actions=tuple(actions),
                            terminal_atoms=states[-1],
                        )
                    )
                    records.append(
                        TrajectoryRecord(
                            dataset_id=source.dataset_id,
                            dataset_root=str(source.root),
                            run_id=run_id,
                            problem_name=problem_name,
                            start_step_index=int(run_rows[0]["step_index"]),
                            num_actions=len(actions),
                        )
                    )
                    relevant_trajectory_count += 1
            except Exception as exc:  # noqa: BLE001 - report run-level ingestion failures.
                message = f"{source.root}: run={run_id} problem={problem_name}: {exc}"
                malformed.append(message)
                if strict:
                    raise ValueError(message) from exc

        summaries.append(
            {
                "dataset_id": source.dataset_id,
                "root": str(source.root),
                "runs": len(grouped_rows),
                "problems": len(problem_names),
                "relevant_trajectories": relevant_trajectory_count,
                "relevant_transitions": relevant_transition_count,
                "action_counts": dict(sorted(action_counts.items())),
            }
        )

    validate_compatible_problems(parsed_problems)
    return LoadedCorpus(
        parsed_problems=tuple(parsed_problems),
        trajectories=tuple(trajectories),
        records=tuple(records),
        dataset_summaries=tuple(summaries),
        malformed_rows=tuple(malformed),
    )


def split_corpus(corpus: LoadedCorpus, *, val_fraction: float, test_fraction: float, seed: int) -> SplitCorpus:
    """Split by problem identity while keeping all trajectories for a problem together."""

    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to less than 1")

    keys = sorted({(record.dataset_id, record.problem_name) for record in corpus.records})
    rng = random.Random(seed)
    rng.shuffle(keys)
    total = len(keys)
    test_count = int(round(total * test_fraction))
    val_count = int(round(total * val_fraction))
    test_keys = set(keys[:test_count])
    val_keys = set(keys[test_count : test_count + val_count])
    train_keys = set(keys[test_count + val_count :])

    manifest = {
        "train": _manifest_entries(train_keys),
        "val": _manifest_entries(val_keys),
        "test": _manifest_entries(test_keys),
    }
    return SplitCorpus(
        train=_subset_corpus(corpus, train_keys),
        val=_subset_corpus(corpus, val_keys),
        test=_subset_corpus(corpus, test_keys),
        manifest=manifest,
    )


def load_action_applicability_table(
    path: str | Path,
    parsed_problems: Sequence[ParsedProblem],
) -> ActionApplicabilityTable:
    """Load a strict, duplicate-safe offline atom-state applicability artifact."""

    artifact_path = Path(path)
    if not artifact_path.is_absolute():
        raise ValueError("action_applicability_table_path must be absolute")

    try:
        payload = json.loads(
            artifact_path.read_text(encoding="utf-8"),
            object_pairs_hook=_JsonObjectPairs,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid applicability JSON in {artifact_path}: {exc}") from exc
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read UTF-8 applicability JSON at {artifact_path}: {exc}") from exc
    artifact_context = f"{artifact_path}: artifact"
    payload = _require_exact_object(
        payload, {"semantics", "entries"}, context=artifact_context
    )
    if payload["semantics"] != ATOM_STATE_APPLICABILITY_SEMANTICS:
        raise ValueError(
            f"{artifact_context} semantics must be "
            f"{ATOM_STATE_APPLICABILITY_SEMANTICS!r}"
        )
    entries_payload = payload["entries"]
    if type(entries_payload) is not list:
        raise ValueError(f"{artifact_context}.entries must be a list")

    mapping = {}
    for entry_index, entry in enumerate(entries_payload):
        context = f"{artifact_path}: artifact.entries[{entry_index}]"
        entry = _require_exact_object(
            entry,
            {
                "problem_index",
                "problem_name",
                "state_atoms",
                "applicable_actions",
            },
            context=context,
        )
        problem_index = entry["problem_index"]
        if type(problem_index) is not int or not 0 <= problem_index < len(parsed_problems):
            raise ValueError(f"{context}.problem_index is out of range")
        problem_name = entry["problem_name"]
        if type(problem_name) is not str or problem_name != parsed_problems[problem_index].name:
            raise ValueError(f"{context}.problem_name does not match parsed problem identity")
        atoms = _parse_ground_values(
            entry["state_atoms"],
            name_key="predicate",
            value_type=GroundAtom,
            context=f"{context}.state_atoms",
        )
        if len(set(atoms)) != len(atoms):
            raise ValueError(f"{context} contains a duplicate state atom")
        actions = _parse_ground_values(
            entry["applicable_actions"],
            name_key="name",
            value_type=GroundAction,
            context=f"{context}.applicable_actions",
        )
        if len(set(actions)) != len(actions):
            raise ValueError(f"{context} contains a duplicate applicable action")
        key = action_applicability_state_key(problem_index, atoms)
        if key in mapping:
            raise ValueError(f"{context} duplicates applicability state key {key!r}")
        mapping[key] = set(actions)
        try:
            _validate_action_applicability_table(
                ActionApplicabilityTable.from_mapping({key: set(actions)}),
                parsed_problems,
            )
        except ValueError as exc:
            raise ValueError(f"{context}: {exc}") from exc

    table = ActionApplicabilityTable.from_mapping(mapping)
    _validate_action_applicability_table(table, parsed_problems)
    return table


class _JsonObjectPairs(list[tuple[str, Any]]):
    """Pair-preserving JSON object used for context-aware duplicate checks."""


def _validate_action_applicability_table(
    table: ActionApplicabilityTable,
    parsed_problems: Sequence[ParsedProblem],
) -> None:
    validation_config = ActionSupervisionConfig(
        num_negatives=0,
        applicable_actions_by_state=table,
        applicability_state_semantics=ATOM_STATE_APPLICABILITY_SEMANTICS,
    )
    # Construction performs Stage 2D2 problem-indexed symbolic validation without
    # simulator calls; an empty trajectory collection is an approved boundary.
    PDDLTrajectoryDataset(
        parsed_problems,
        (),
        include_static=True,
        action_supervision=validation_config,
    )


def _require_exact_object(
    value: Any, expected_keys: set[str], *, context: str
) -> dict[str, Any]:
    if not isinstance(value, _JsonObjectPairs):
        raise ValueError(f"{context} must be a JSON object")
    result: dict[str, Any] = {}
    for key, member_value in value:
        if key in result:
            raise ValueError(f"{context} contains duplicate JSON member {key!r}")
        result[key] = member_value
    actual = set(result)
    if actual != expected_keys:
        raise ValueError(
            f"{context} must contain exactly {sorted(expected_keys)!r}; got {sorted(actual)!r}"
        )
    return result


def _parse_ground_values(
    value: Any,
    *,
    name_key: str,
    value_type: type[GroundAtom] | type[GroundAction],
    context: str,
) -> tuple[GroundAtom, ...] | tuple[GroundAction, ...]:
    if type(value) is not list:
        raise ValueError(f"{context} must be a list")
    parsed = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        item = _require_exact_object(item, {name_key, "arguments"}, context=item_context)
        name = item[name_key]
        arguments = item["arguments"]
        if type(name) is not str or type(arguments) is not list or any(
            type(argument) is not str for argument in arguments
        ):
            raise ValueError(f"{item_context} name and arguments must be strings")
        parsed.append(value_type(name, tuple(arguments)))
    return tuple(parsed)


def make_torch_dataset(corpus: LoadedCorpus, config: Any):
    """Create the core PyG dataset matching the configured goal-head mode."""

    goal_kind = str(config.model.goal_head.kind)
    windows = trajectory_windows(corpus.trajectories, rollout_steps=int(config.data.rollout_steps))
    loss_cfg = config.model.loss
    action_supervision = None
    needs_action_supervision = any(
        float(value) > 0.0
        for value in (
            loss_cfg.action_contrastive_coeff,
            loss_cfg.argument_reconstruction_coeff,
            loss_cfg.applicability_coeff,
        )
    )
    if needs_action_supervision:
        table_path = config.data.action_applicability_table_path
        table = (
            None
            if table_path is None
            else load_action_applicability_table(str(table_path), corpus.parsed_problems)
        )
        action_supervision = ActionSupervisionConfig(
            num_negatives=config.model.loss.action_hard_negatives_per_positive,
            seed=config.data.action_supervision_seed,
            max_random_attempts_per_category=(
                config.data.action_negative_max_attempts_per_category
            ),
            applicable_actions_by_state=table,
            applicability_state_semantics=(
                None if table is None else ATOM_STATE_APPLICABILITY_SEMANTICS
            ),
        )
    if goal_kind == "none":
        return PDDLTrajectoryDataset(
            corpus.parsed_problems,
            windows,
            include_static=True,
            action_supervision=action_supervision,
        )
    return PDDLAtomTrajectoryDataset(
        corpus.parsed_problems,
        windows,
        num_positive_atoms=int(config.data.num_positive_atoms),
        num_negative_atoms=int(config.data.num_negative_atoms),
        goal_positive_fraction=float(config.data.goal_positive_fraction),
        include_static=True,
        include_goal=goal_kind in {"gaussian", "gmm", "conditional_sampler"},
        include_terminal_state=goal_kind in {"gaussian", "gmm", "conditional_sampler"},
        max_goal_atoms=None if config.data.max_goal_atoms is None else int(config.data.max_goal_atoms),
        seed=0,
        action_supervision=action_supervision,
    )


def trajectory_windows(
    trajectories: Sequence[TrajectorySample],
    *,
    rollout_steps: int,
) -> tuple[TrajectorySample, ...]:
    """Return fixed-length sliding windows from full trajectories."""

    if rollout_steps < 1:
        raise ValueError("rollout_steps must be at least 1")
    windows: list[TrajectorySample] = []
    for trajectory in trajectories:
        if len(trajectory.actions) < rollout_steps:
            continue
        for start_idx in range(0, len(trajectory.actions) - rollout_steps + 1):
            end_idx = start_idx + rollout_steps
            windows.append(
                TrajectorySample(
                    problem_index=trajectory.problem_index,
                    states=trajectory.states[start_idx : end_idx + 1],
                    actions=trajectory.actions[start_idx:end_idx],
                    terminal_atoms=trajectory.terminal_atoms,
                )
            )
    return tuple(windows)


def validate_compatible_problems(parsed_problems: Sequence[ParsedProblem]) -> None:
    """Ensure all parsed problems share model-relevant domain schema."""

    if not parsed_problems:
        raise ValueError("No parsed problems were loaded")
    first = parsed_problems[0]
    signature = _domain_signature(first)
    for parsed in parsed_problems[1:]:
        if _domain_signature(parsed) != signature:
            raise ValueError("All dataset problems must share the same domain schema")


def corpus_summary(corpus: LoadedCorpus) -> dict[str, Any]:
    """Return a JSON-serializable summary."""

    action_counts: dict[str, int] = {}
    for trajectory in corpus.trajectories:
        for action in trajectory.actions:
            action_counts[action.name] = action_counts.get(action.name, 0) + 1
    return {
        "num_parsed_problems": len(corpus.parsed_problems),
        "num_trajectories": len(corpus.trajectories),
        "num_transitions": sum(len(trajectory.actions) for trajectory in corpus.trajectories),
        "num_malformed_rows": len(corpus.malformed_rows),
        "dataset_summaries": list(corpus.dataset_summaries),
        "action_counts": dict(sorted(action_counts.items())),
    }


def _validate_duckdb_schema(database_path: Path) -> None:
    with duckdb.connect(str(database_path)) as con:
        rows = con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            """
        ).fetchall()
    names = {str(row[0]) for row in rows}
    missing = sorted(REQUIRED_TABLES - names)
    if missing:
        raise ValueError(f"{database_path} is missing required tables/views: {missing}")


def _parse_source_problems(source: DatasetSource) -> dict[str, ParsedProblem]:
    problem_paths = sorted(path for path in source.problem_dir.glob("*.pddl") if path.name != "domain.pddl")
    if not problem_paths:
        raise FileNotFoundError(f"No problem PDDL files found in {source.problem_dir}")
    parsed_by_name: dict[str, ParsedProblem] = {}
    for problem_path in problem_paths:
        parsed = parse_domain_problem(source.domain_path, problem_path)
        parsed_by_name[parsed.name] = parsed
        parsed_by_name.setdefault(problem_path.stem, parsed)
    return parsed_by_name


def _query_transition_rows(database_path: Path) -> list[dict[str, Any]]:
    with duckdb.connect(str(database_path)) as con:
        result = con.execute(RELEVANT_TRANSITIONS_SQL)
        columns = [item[0] for item in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]


def _facts_to_atoms(payload: Any) -> tuple[GroundAtom, ...]:
    if payload is None:
        return ()
    atoms = []
    for item in payload:
        name = _field(item, "name", 0)
        arguments = _field(item, "arguments", 1)
        atoms.append(GroundAtom(predicate=str(name), arguments=tuple(str(arg) for arg in (arguments or ()))))
    return tuple(sorted(atoms))


def _field(item: Any, name: str, index: int) -> Any:
    if isinstance(item, dict):
        return item[name]
    if hasattr(item, name):
        return getattr(item, name)
    return item[index]


def _domain_signature(parsed: ParsedProblem) -> tuple[Any, ...]:
    return (
        parsed.types,
        tuple(sorted((name, schema.arg_types) for name, schema in parsed.predicates.items())),
        tuple(sorted((name, schema.parameter_types) for name, schema in parsed.actions.items())),
    )


def _manifest_entries(keys: set[tuple[int, str]]) -> list[dict[str, Any]]:
    return [
        {"dataset_id": dataset_id, "problem_name": problem_name}
        for dataset_id, problem_name in sorted(keys)
    ]


def _subset_corpus(corpus: LoadedCorpus, keys: set[tuple[int, str]]) -> LoadedCorpus:
    trajectories: list[TrajectorySample] = []
    records: list[TrajectoryRecord] = []
    for trajectory, record in zip(corpus.trajectories, corpus.records):
        if (record.dataset_id, record.problem_name) in keys:
            trajectories.append(trajectory)
            records.append(record)
    return LoadedCorpus(
        parsed_problems=corpus.parsed_problems,
        trajectories=tuple(trajectories),
        records=tuple(records),
        dataset_summaries=corpus.dataset_summaries,
        malformed_rows=corpus.malformed_rows,
    )
