"""Diagnose offline applicability labels for trace actions and hard negatives."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Any

from acs_jepa.architectures import ActionDecodingSpace
from acs_jepa_cli.config import load_config
from acs_jepa_cli.data import load_corpus
from action_applicability_labels import build_applicability_examples
from action_diag_common import (
    action_payload,
    applicable_keys,
    iter_transitions,
    replay_to_engine,
    select_split,
    write_json,
)
from action_negative_sampling import sample_action_negatives


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "test", "all"))
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--no-oracle-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def parse_args() -> argparse.Namespace:
    args = build_parser().parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.per_category <= 0:
        raise ValueError("--per-category must be positive")
    if args.max_transitions is not None and args.max_transitions < 0:
        raise ValueError("--max-transitions must be non-negative")


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    corpus = load_corpus([args.dataset_dir], strict=True)
    selected = select_split(corpus, load_config(None), args.split, seed=args.seed)
    space_cache: dict[int, ActionDecodingSpace] = {}
    details: list[dict[str, Any]] = []
    totals = Counter()
    kind_counts = Counter()
    category_counts = Counter()
    applicability_counts = Counter({"applicable": 0, "inapplicable": 0, "unknown": 0})
    true_action_applicable_known = 0
    true_action_applicable_count = 0

    for transition_index, (trajectory, record, step_idx, true_action) in enumerate(
        iter_transitions(selected, max_transitions=args.max_transitions)
    ):
        parsed = selected.parsed_problems[trajectory.problem_index]
        if trajectory.problem_index not in space_cache:
            space_cache[trajectory.problem_index] = ActionDecodingSpace.from_parsed_problem(parsed)
        negatives = sample_action_negatives(
            space_cache[trajectory.problem_index],
            true_action,
            per_category=args.per_category,
            seed=args.seed + transition_index,
        )
        applicable = None
        if not args.no_oracle_labels:
            engine = replay_to_engine(args.dataset_dir, record.problem_name, trajectory.actions[:step_idx])
            applicable = applicable_keys(engine)
        batch = build_applicability_examples(true_action, negatives, applicable_action_keys=applicable)
        totals["transitions"] += 1
        totals["examples"] += batch.summary["examples"]
        kind_counts.update(batch.summary["kind_counts"])
        category_counts.update(batch.summary["category_counts"])
        applicability_counts.update(batch.summary["applicability_counts"])
        if batch.summary["true_action_applicable"] is not None:
            true_action_applicable_known += 1
            true_action_applicable_count += int(batch.summary["true_action_applicable"])
        details.append(
            {
                "problem": record.problem_name,
                "step_index": step_idx,
                "true_action": action_payload(true_action),
                "summary": batch.summary,
                "examples": list(batch.examples),
            }
        )

    summary = {
        "dataset": str(args.dataset_dir),
        "split": args.split,
        "max_transitions": args.max_transitions,
        "per_category": args.per_category,
        "oracle_labels": not args.no_oracle_labels,
        "seed": args.seed,
        "metrics": {
            "transitions": int(totals["transitions"]),
            "examples": int(totals["examples"]),
            "runtime_seconds": time.perf_counter() - started,
            "true_action_applicable_rate": None
            if true_action_applicable_known == 0
            else true_action_applicable_count / true_action_applicable_known,
        },
        "kind_counts": dict(sorted(kind_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "applicability_counts": dict(sorted(applicability_counts.items())),
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "details.json", details)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
