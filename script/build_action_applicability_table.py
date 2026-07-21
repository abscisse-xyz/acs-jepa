"""Produce strict offline state-to-applicable-action labels from recorded trajectories."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from acs_jepa.graph import ATOM_STATE_APPLICABILITY_SEMANTICS
from acs_jepa_cli.data import LoadedCorpus, load_corpus

FactKey = tuple[str, ...]
ActionKey = tuple[str, tuple[str, ...]]
EntryKey = tuple[int, tuple[FactKey, ...]]
EntryValue = tuple[str, tuple[ActionKey, ...]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline action-applicability table from a simulator corpus."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def build_payload(
    dataset_dir: Path,
    *,
    corpus: LoadedCorpus | None = None,
    engine_factory: Callable[[Path, Path], Any] | None = None,
) -> dict[str, Any]:
    """Replay the corpus and return a deterministic Stage 2E JSON payload."""

    corpus = load_corpus([dataset_dir], strict=True) if corpus is None else corpus
    engine_factory = _build_engine if engine_factory is None else engine_factory
    entries: dict[EntryKey, EntryValue] = {}

    if len(corpus.trajectories) != len(corpus.records):
        raise ValueError(
            f"dataset={dataset_dir}: trajectory/record count mismatch: "
            f"{len(corpus.trajectories)} != {len(corpus.records)}"
        )

    domain_path = dataset_dir / "problem" / "domain.pddl"
    for trajectory_index, (trajectory, record) in enumerate(
        zip(corpus.trajectories, corpus.records, strict=True)
    ):
        context = _context(dataset_dir, record, trajectory_index)
        problem_index = trajectory.problem_index
        if not 0 <= problem_index < len(corpus.parsed_problems):
            raise ValueError(f"{context}: problem_index={problem_index} is out of range")
        problem_name = corpus.parsed_problems[problem_index].name
        problem_path = _problem_path(dataset_dir, record.problem_name, context=context)
        try:
            engine = engine_factory(domain_path, problem_path)
        except Exception as exc:  # noqa: BLE001 - add corpus identity to oracle failures.
            raise ValueError(f"{context}: cannot create offline simulator: {exc}") from exc

        for step, action in enumerate(trajectory.actions):
            step_context = f"{context} step={step}"
            recorded_facts = _recorded_facts(trajectory.states[step])
            simulator_facts = _simulator_facts(engine)
            if simulator_facts != recorded_facts:
                raise ValueError(
                    f"{step_context}: source state mismatch: "
                    f"simulator={simulator_facts!r} recorded={recorded_facts!r}"
                )

            try:
                applicable_actions = _applicable_actions(engine)
            except Exception as exc:  # noqa: BLE001 - add transition identity to oracle failures.
                raise ValueError(f"{step_context}: applicable-action query failed: {exc}") from exc
            recorded_action = (action.name, tuple(action.arguments))
            if recorded_action not in applicable_actions:
                raise ValueError(
                    f"{step_context}: recorded action {recorded_action!r} is not in "
                    f"the complete applicable-action set {applicable_actions!r}"
                )

            _merge_entry(
                entries,
                problem_index=problem_index,
                problem_name=problem_name,
                state_facts=recorded_facts,
                applicable_actions=applicable_actions,
                context=step_context,
            )
            try:
                engine.apply_action(action.name, action.arguments, finish=True)
            except Exception as exc:  # noqa: BLE001 - contextualize replay failures.
                raise ValueError(
                    f"{step_context}: replay failed for recorded action {recorded_action!r}: {exc}"
                ) from exc

        terminal_facts = _recorded_facts(trajectory.terminal_atoms)
        simulator_terminal = _simulator_facts(engine)
        if simulator_terminal != terminal_facts:
            raise ValueError(
                f"{context} step={len(trajectory.actions)}: terminal state mismatch: "
                f"simulator={simulator_terminal!r} recorded={terminal_facts!r}"
            )

    return {
        "semantics": ATOM_STATE_APPLICABILITY_SEMANTICS,
        "entries": [_entry_payload(key, entries[key]) for key in sorted(entries)],
    }


def _merge_entry(
    entries: dict[EntryKey, EntryValue],
    *,
    problem_index: int,
    problem_name: str,
    state_facts: Sequence[FactKey],
    applicable_actions: Sequence[ActionKey],
    context: str,
) -> None:
    """Deduplicate an equal state or reject conflicting oracle labels."""

    key = (problem_index, tuple(sorted(state_facts)))
    value = (problem_name, tuple(sorted(set(applicable_actions))))
    previous = entries.get(key)
    if previous is not None and previous != value:
        raise ValueError(
            f"{context}: contradictory applicable-action sets for state key {key!r}: "
            f"previous={previous[1]!r} current={value[1]!r}"
        )
    entries[key] = value


def _entry_payload(key: EntryKey, value: EntryValue) -> dict[str, Any]:
    problem_index, state_facts = key
    problem_name, actions = value
    return {
        "problem_index": problem_index,
        "problem_name": problem_name,
        "state_atoms": [
            {"predicate": fact[0], "arguments": list(fact[1:])} for fact in state_facts
        ],
        "applicable_actions": [
            {"name": name, "arguments": list(arguments)} for name, arguments in actions
        ],
    }


def _recorded_facts(atoms: Sequence[Any]) -> tuple[FactKey, ...]:
    return tuple(sorted((atom.predicate, *atom.arguments) for atom in atoms))


def _simulator_facts(engine: Any) -> tuple[FactKey, ...]:
    return tuple(sorted(tuple(str(part) for part in fact) for fact in engine.current_facts()))


def _applicable_actions(engine: Any) -> tuple[ActionKey, ...]:
    return tuple(
        sorted({(action.name, tuple(action.arguments)) for action in engine.applicable_actions()})
    )


def _problem_path(dataset_dir: Path, problem_name: str, *, context: str) -> Path:
    problem_dir = dataset_dir / "problem"
    direct = problem_dir / problem_name
    suffixed = problem_dir / f"{problem_name}.pddl"
    for candidate in (direct, suffixed):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{context}: no PDDL file for record.problem_name={problem_name!r} in {problem_dir}"
    )


def _context(dataset_dir: Path, record: Any, trajectory_index: int) -> str:
    return (
        f"dataset={dataset_dir} problem={record.problem_name!r} "
        f"trajectory={trajectory_index} run={record.run_id}"
    )


def _build_engine(domain_path: Path, problem_path: Path) -> Any:
    from simulator import SimulatorEngine

    return SimulatorEngine.from_pddl(domain_path, problem_path)


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.output.is_absolute():
        raise ValueError("--output must be an absolute path")
    payload = build_payload(args.dataset_dir)
    _write_atomic_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
