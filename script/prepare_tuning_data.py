"""Create deterministic ACS-JEPA tuning datasets from the CityCar corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb


SPLIT_SEED = 20260717
TEST_SIZE = 20
DEVELOPMENT_SIZE = 48
SMOKE_SIZE = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem_source = args.source / "citycar-topology-200-problem"
    simulation_source = args.source / "citycar-topology-200-simulation"
    problem_manifest_path = problem_source / "manifest.json"
    simulation_manifest_path = simulation_source / "manifest.json"
    database_path = simulation_source / "simulation.duckdb"

    problem_manifest = json.loads(problem_manifest_path.read_text())
    simulation_manifest = json.loads(simulation_manifest_path.read_text())
    instances = {Path(item["file"]).stem: item for item in problem_manifest["instances"]}
    solved_results = {
        item["problem"]: item
        for item in simulation_manifest["results"]
        if item["goals_satisfied"]
        and item["planner_status"] in {"SOLVED_SATISFICING", "SOLVED_OPTIMALLY"}
        and item["plan_topology"] == "SequentialPlan"
    }
    missing = sorted(set(solved_results) - set(instances))
    if missing:
        raise ValueError(f"Solved problems missing generation metadata: {missing}")

    records = [_record(problem, instances[problem], result) for problem, result in solved_results.items()]
    _assign_length_bins(records)

    final_test = _stratified_sample(records, TEST_SIZE, seed=SPLIT_SEED)
    final_test_names = {item["problem"] for item in final_test}
    full_development = [item for item in records if item["problem"] not in final_test_names]
    development = _stratified_sample(full_development, DEVELOPMENT_SIZE, seed=SPLIT_SEED + 1)
    smoke = _stratified_sample(development, SMOKE_SIZE, seed=SPLIT_SEED + 2)

    source_fingerprint = _fingerprint((problem_manifest_path, simulation_manifest_path, database_path))
    datasets = {
        "smoke": smoke,
        "development": development,
        "full-dev": full_development,
        "final-test": final_test,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, selected in datasets.items():
        _materialize_dataset(
            name=name,
            records=selected,
            destination=args.output / name,
            problem_source=problem_source,
            database_source=database_path,
            source_fingerprint=source_fingerprint,
        )

    campaign_manifest = {
        "schema_version": 1,
        "split_seed": SPLIT_SEED,
        "stratification": ["difficulty", "reference_plan_length_tercile"],
        "source": str(args.source.resolve()),
        "source_fingerprint": source_fingerprint,
        "counts": {name: len(selected) for name, selected in datasets.items()},
        "datasets": {
            name: [item["problem"] for item in sorted(selected, key=lambda item: item["problem"])]
            for name, selected in datasets.items()
        },
    }
    (args.output / "campaign_manifest.json").write_text(json.dumps(campaign_manifest, indent=2, sort_keys=True))
    print(json.dumps(campaign_manifest, indent=2, sort_keys=True))


def _record(problem: str, instance: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    family = str(instance["family"])
    difficulty = next((value for value in ("easy", "medium", "hard") if f"_{value}_" in family), None)
    if difficulty is None:
        raise ValueError(f"Cannot infer difficulty from family {family}")
    return {
        "problem": problem,
        "file": instance["file"],
        "family": family,
        "difficulty": difficulty,
        "topology_family": instance["params"]["topology_family"],
        "rows": int(instance["params"]["rows"]),
        "columns": int(instance["params"]["columns"]),
        "cars": int(instance["params"]["cars"]),
        "garages": int(instance["params"]["garages"]),
        "roads": int(instance["params"]["roads"]),
        "reference_plan_length": int(result["step_count"]),
    }


def _assign_length_bins(records: list[dict[str, Any]]) -> None:
    ordered = sorted(records, key=lambda item: (item["reference_plan_length"], item["problem"]))
    count = len(ordered)
    for index, item in enumerate(ordered):
        item["reference_plan_length_tercile"] = min(2, 3 * index // count)


def _stratified_sample(records: list[dict[str, Any]], count: int, *, seed: int) -> list[dict[str, Any]]:
    if count > len(records):
        raise ValueError(f"Cannot sample {count} records from {len(records)}")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[(item["difficulty"], item["reference_plan_length_tercile"])].append(item)

    exact = {key: count * len(items) / len(records) for key, items in grouped.items()}
    allocation = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(allocation.values())
    remainder_order = sorted(grouped, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in remainder_order[:remaining]:
        allocation[key] += 1

    rng = random.Random(seed)
    selected = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda item: item["problem"])
        rng.shuffle(candidates)
        selected.extend(candidates[: allocation[key]])
    return sorted(selected, key=lambda item: item["problem"])


def _materialize_dataset(
    *,
    name: str,
    records: list[dict[str, Any]],
    destination: Path,
    problem_source: Path,
    database_source: Path,
    source_fingerprint: str,
) -> None:
    problem_destination = destination / "problem"
    simulation_destination = destination / "simulation"
    problem_destination.mkdir(parents=True, exist_ok=True)
    simulation_destination.mkdir(parents=True, exist_ok=True)

    shutil.copy2(problem_source / "domain.pddl", problem_destination / "domain.pddl")
    for item in records:
        shutil.copy2(problem_source / item["file"], problem_destination / item["file"])

    database_destination = simulation_destination / "simulation.duckdb"
    shutil.copy2(database_source, database_destination)
    selected_names = [item["problem"] for item in records]
    with duckdb.connect(str(database_destination)) as connection:
        connection.execute("CREATE TEMP TABLE selected_problems(name VARCHAR PRIMARY KEY)")
        connection.executemany("INSERT INTO selected_problems VALUES (?)", [(value,) for value in selected_names])
        connection.execute(
            "CREATE TEMP TABLE selected_runs AS "
            "SELECT id FROM simulation_runs WHERE problem_name IN (SELECT name FROM selected_problems)"
        )
        for table in (
            "action_events",
            "action_schemas",
            "fluent_symbols",
            "planner_attempts",
            "predicate_symbols",
            "problem_goals",
            "problem_objects",
            "problem_timed_effects",
            "state_snapshots",
            "trace_entries",
        ):
            connection.execute(f"DELETE FROM {table} WHERE run_id NOT IN (SELECT id FROM selected_runs)")
        connection.execute("DELETE FROM simulation_runs WHERE id NOT IN (SELECT id FROM selected_runs)")
        connection.execute("CHECKPOINT")

    manifest = {
        "schema_version": 1,
        "name": name,
        "source_fingerprint": source_fingerprint,
        "split_seed": SPLIT_SEED,
        "stratification": ["difficulty", "reference_plan_length_tercile"],
        "num_problems": len(records),
        "problems": records,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
