"""Measure how sparse applicable actions are inside the typed action space."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Any

from acs_jepa.architectures import ActionDecodingSpace

from action_diag_common import (
    action_payload,
    applicable_keys,
    iter_transitions,
    replay_to_engine,
    select_split,
    write_json,
)
from acs_jepa_cli.config import load_config
from acs_jepa_cli.data import load_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--split", default="val", choices=("train", "val", "test", "all"))
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    corpus = load_corpus([args.dataset_dir], strict=True)
    if args.split == "all":
        selected = corpus
    else:
        if args.checkpoint is None:
            config = load_config(None)
        else:
            from action_diag_common import load_checkpoint_bundle

            config, _corpus_from_checkpoint, _bundle, _device = load_checkpoint_bundle(
                args.dataset_dir,
                args.checkpoint,
                device_name=args.device,
            )
        selected = select_split(corpus, config, args.split, seed=args.seed)

    details: list[dict[str, Any]] = []
    totals = Counter()
    schema_totals: dict[str, Counter] = {}
    space_cache: dict[int, tuple[ActionDecodingSpace, tuple[Any, ...], Counter]] = {}

    for trajectory, record, step_idx, true_action in iter_transitions(selected, max_transitions=args.max_transitions):
        parsed = selected.parsed_problems[trajectory.problem_index]
        if trajectory.problem_index not in space_cache:
            space = ActionDecodingSpace.from_parsed_problem(parsed)
            typed_actions = space.enumerate_ground_actions()
            typed_by_schema = Counter(action.name for action in typed_actions)
            space_cache[trajectory.problem_index] = (space, typed_actions, typed_by_schema)
        _space, typed_actions, typed_by_schema = space_cache[trajectory.problem_index]
        engine = replay_to_engine(args.dataset_dir, record.problem_name, trajectory.actions[:step_idx])
        applicable = applicable_keys(engine)
        applicable_by_schema = Counter(name for name, _args in applicable)
        true_is_applicable = (true_action.name, tuple(true_action.arguments)) in applicable

        typed_count = len(typed_actions)
        applicable_count = len(applicable)
        totals["transitions"] += 1
        totals["type_valid_actions"] += typed_count
        totals["applicable_actions"] += applicable_count
        totals["true_action_applicable"] += int(true_is_applicable)

        for schema, count in typed_by_schema.items():
            schema_totals.setdefault(schema, Counter())["type_valid"] += count
        for schema, count in applicable_by_schema.items():
            schema_totals.setdefault(schema, Counter())["applicable"] += count

        details.append(
            {
                "problem": record.problem_name,
                "step_index": step_idx,
                "true_action": action_payload(true_action),
                "true_action_applicable": true_is_applicable,
                "type_valid_actions": typed_count,
                "applicable_actions": applicable_count,
                "applicable_fraction": 0.0 if typed_count == 0 else applicable_count / typed_count,
                "type_valid_by_schema": dict(sorted(typed_by_schema.items())),
                "applicable_by_schema": dict(sorted(applicable_by_schema.items())),
            }
        )

    transition_count = totals["transitions"]
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset_dir),
        "split": args.split,
        "max_transitions": args.max_transitions,
        "metrics": {
            "transitions": transition_count,
            "avg_type_valid_actions": 0.0
            if transition_count == 0
            else totals["type_valid_actions"] / transition_count,
            "avg_applicable_actions": 0.0
            if transition_count == 0
            else totals["applicable_actions"] / transition_count,
            "avg_applicable_fraction": 0.0
            if totals["type_valid_actions"] == 0
            else totals["applicable_actions"] / totals["type_valid_actions"],
            "true_action_applicable_rate": 0.0
            if transition_count == 0
            else totals["true_action_applicable"] / transition_count,
            "runtime_seconds": time.perf_counter() - started,
        },
        "schema_totals": {
            schema: {
                "type_valid": int(counter["type_valid"]),
                "applicable": int(counter["applicable"]),
                "applicable_fraction": 0.0
                if counter["type_valid"] == 0
                else counter["applicable"] / counter["type_valid"],
            }
            for schema, counter in sorted(schema_totals.items())
        },
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "details.json", details)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
