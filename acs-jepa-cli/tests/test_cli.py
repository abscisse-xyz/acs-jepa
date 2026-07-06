from __future__ import annotations

import json
from pathlib import Path

import duckdb
import torch
from acs_jepa_cli.cli import main
from acs_jepa_cli.config import load_config, tuning_overlay_tags
from acs_jepa_cli.data import load_corpus, split_corpus
from acs_jepa_cli.scheduling import NoOpScheduler, WarmupCosineScheduler, build_scheduler

DOMAIN = """
(define (domain tiny-city)
  (:requirements :typing :negative-preconditions)
  (:types car junction road)
  (:predicates
    (same_line ?a - junction ?b - junction)
    (clear ?j - junction)
    (at_car_jun ?c - car ?j - junction)
    (road_connect ?r - road ?from - junction ?to - junction)
  )

  (:action move
    :parameters (?c - car ?from - junction ?to - junction ?r - road)
    :precondition (and
      (at_car_jun ?c ?from)
      (road_connect ?r ?from ?to)
      (clear ?to)
    )
    :effect (and
      (not (at_car_jun ?c ?from))
      (at_car_jun ?c ?to)
      (clear ?from)
      (not (clear ?to))
    )
  )
)
"""


PROBLEM = """
(define (problem tiny-city-1)
  (:domain tiny-city)
  (:objects
    car0 - car
    road0 - road
    j0 j1 - junction
  )
  (:init
    (same_line j0 j1)
    (clear j1)
    (at_car_jun car0 j0)
    (road_connect road0 j0 j1)
  )
  (:goal (and
    (at_car_jun car0 j1)
  ))
)
"""


def test_load_corpus_filters_relevant_rows_and_keeps_dataset_ids(tmp_path: Path) -> None:
    first = _dataset(tmp_path / "a")
    second = _dataset(tmp_path / "b")

    corpus = load_corpus([first, second])
    splits = split_corpus(corpus, val_fraction=0.5, test_fraction=0.0, seed=0)

    assert len(corpus.trajectories) == 2
    assert sum(len(trajectory.actions) for trajectory in corpus.trajectories) == 4
    assert {record.dataset_id for record in corpus.records} == {0, 1}
    assert {record.run_id for record in corpus.records} == {1}
    assert all(record.num_actions == 2 for record in corpus.records)
    split_keys = {
        (entry["dataset_id"], entry["problem_name"])
        for entries in splits.manifest.values()
        for entry in entries
    }
    assert split_keys == {(0, "tiny-city-1"), (1, "tiny-city-1")}


def test_inspect_data_command_reports_summary(tmp_path: Path, capsys) -> None:
    dataset = _dataset(tmp_path / "dataset")

    exit_code = main(["inspect-data", str(dataset)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"relevant_trajectories": 1' in output
    assert '"relevant_transitions": 2' in output
    assert '"move": 2' in output


def test_load_config_uses_default_yaml_before_user_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  batch_size: 3
planning:
  horizon: 2
"""
    )

    config = load_config(config_path)

    assert config.data.batch_size == 3
    assert config.planning.horizon == 2
    assert config.model.goal_head.kind == "gmm"
    assert config.planning.action_pool_size == 512


def test_load_config_merges_multiple_overlays_in_order(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(
        """
data:
  batch_size: 16
model:
  action_encoder:
    kind: rnn
"""
    )
    second.write_text(
        """
data:
  batch_size: 24
model:
  predictor:
    kind: gru
"""
    )

    config = load_config([first, second])

    assert config.data.batch_size == 24
    assert config.model.action_encoder.kind == "rnn"
    assert config.model.predictor.kind == "gru"


def test_tuning_overlay_tags_summarize_stacked_configs() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "tuning"
    tags = tuning_overlay_tags(
        [
            root / "01_composition" / "backbone_rnn_gru.yaml",
            root / "01_composition" / "gmm_detach_false.yaml",
            root / "02_capacity" / "size96_layers2.yaml",
            root / "03_rollout" / "rollout6.yaml",
        ]
    )

    assert tags["tuning.composition"] == "backbone_rnn_gru+gmm_detach_false"
    assert tags["tuning.capacity"] == "size96_layers2"
    assert tags["tuning.rollout"] == "rollout6"
    assert tags["tuning.stack"] == (
        "composition:backbone_rnn_gru|composition:gmm_detach_false|capacity:size96_layers2|rollout:rollout6"
    )


def test_tuning_configs_load_and_keep_required_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    config_paths = sorted((root / "configs" / "tuning").glob("*/*.yaml"))

    assert config_paths
    for path in config_paths:
        config = load_config(path)
        assert float(config.model.loss.similarity_coeff) > 0.0
        assert float(config.model.loss.inverse_dynamics_coeff) > 0.0
        assert 4 <= int(config.data.rollout_steps) <= 8
        if "05_planning" in path.parts:
            assert int(config.planning.max_iters) >= 60
            assert config.planning.action_decoder.method != "exact"


def test_train_and_eval_commands_log_with_mlflow(tmp_path: Path, monkeypatch) -> None:
    dataset = _dataset(tmp_path / "dataset")
    config = tmp_path / "config.yaml"
    train_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    plan_dir = tmp_path / "plan"
    mlruns = tmp_path / "mlruns"
    config.write_text(
        f"""
model:
  graph_hidden_dim: 8
  graph_embed_dim: 8
  latent_dim: 6
  action_dim: 6
  action_encoder:
    kind: pooled
    hidden_dim: 10
  predictor:
    kind: mlp
    hidden_dim: 10
  loss:
    regularization_coeff: 0.0
  goal_head:
    kind: predicate
    hidden_dim: 10
data:
  batch_size: 1
  rollout_steps: 2
  val_fraction: 0.0
  test_fraction: 0.0
training:
  epochs: 1
  eval_every_steps: 0
  checkpoint_every_steps: 0
planning:
  horizon: 1
  num_samples: 4
  max_iters: 1
  apply_steps: 1
  max_total_actions: 1
  max_decode_attempts: 1
tracking:
  mlflow_tracking_uri: file://{mlruns}
  experiment_name: cli-test
  log_artifacts: true
"""
    )
    monkeypatch.setattr("acs_jepa_cli.cli._build_simulator_engine", lambda _domain, _problem: _AcceptAnyEngine())

    train_code = main(
        [
            "train",
            str(dataset),
            "--output",
            str(train_dir),
            "--config",
            str(config),
            "--device",
            "cpu",
            "--seed",
            "0",
        ]
    )
    eval_code = main(
        [
            "eval",
            str(dataset),
            "--checkpoint",
            str(train_dir / "checkpoints" / "latest.pt"),
            "--output",
            str(eval_dir),
            "--device",
            "cpu",
        ]
    )
    plan_code = main(
        [
            "plan",
            "--checkpoint",
            str(train_dir / "checkpoints" / "latest.pt"),
            "--domain",
            str(dataset / "problem" / "domain.pddl"),
            "--problem",
            str(dataset / "problem" / "p01.pddl"),
            "--output",
            str(plan_dir),
            "--device",
            "cpu",
            "--seed",
            "0",
        ]
    )

    assert train_code == 0
    assert eval_code == 0
    assert plan_code == 0
    assert (train_dir / "checkpoints" / "latest.pt").exists()
    assert (train_dir / "metrics" / "train.jsonl").exists()
    assert (eval_dir / "eval_summary.json").exists()
    assert (plan_dir / "plan_summary.json").exists()
    assert any(mlruns.rglob("meta.yaml"))
    checkpoint = torch.load(train_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 1
    assert checkpoint["scheduler_state_dict"] is not None
    train_metrics = [
        json.loads(line)
        for line in (train_dir / "metrics" / "train.jsonl").read_text().splitlines()
        if line
    ]
    assert train_metrics[0]["optim/lr"] == 0.001
    plan_summary = json.loads((plan_dir / "plan_summary.json").read_text())
    assert plan_summary["planning"]["horizon"] == 1
    assert "applied_actions" in plan_summary


def test_warmup_cosine_scheduler_steps_and_handles_noop_cases() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    config = load_config(None)

    scheduler = build_scheduler(optimizer, config, total_steps=4)

    assert isinstance(scheduler, WarmupCosineScheduler)
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] >= float(config.optimizer.scheduler.min_lr)

    tiny_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    tiny_scheduler = build_scheduler(tiny_optimizer, config, total_steps=1)
    assert isinstance(tiny_scheduler, NoOpScheduler)

    config.optimizer.scheduler.kind = "none"
    none_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    none_scheduler = build_scheduler(none_optimizer, config, total_steps=4)
    assert isinstance(none_scheduler, NoOpScheduler)


def _dataset(root: Path) -> Path:
    problem_dir = root / "problem"
    simulation_dir = root / "simulation"
    problem_dir.mkdir(parents=True)
    simulation_dir.mkdir(parents=True)
    (problem_dir / "domain.pddl").write_text(DOMAIN)
    (problem_dir / "p01.pddl").write_text(PROBLEM)
    _simulation_db(simulation_dir / "simulation.duckdb")
    return root


def _simulation_db(path: Path) -> None:
    with duckdb.connect(str(path)) as con:
        con.execute(
            """
            CREATE TABLE simulation_runs(
                id BIGINT,
                domain_name VARCHAR,
                problem_name VARCHAR,
                created_at TIMESTAMP
            )
            """
        )
        con.execute(
            """
            CREATE TABLE planner_attempts(
                run_id BIGINT,
                status VARCHAR,
                failure_stage VARCHAR,
                plan_topology VARCHAR
            )
            """
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
                next_state_running_actions
                    STRUCT(name VARCHAR, arguments VARCHAR[], started_at VARCHAR, ends_at VARCHAR)[]
            )
            """
        )
        con.execute(
            "INSERT INTO simulation_runs VALUES "
            "(1, 'tiny-city', 'tiny-city-1', CURRENT_TIMESTAMP),"
            "(2, 'tiny-city', 'tiny-city-1', CURRENT_TIMESTAMP),"
            "(3, 'tiny-city', 'tiny-city-1', CURRENT_TIMESTAMP)"
        )
        con.execute(
            "INSERT INTO planner_attempts VALUES "
            "(1, 'SOLVED_SATISFICING', 'completed', 'SequentialPlan'),"
            "(2, 'UNSOLVABLE_PROVEN', 'plan', NULL),"
            "(3, 'SOLVED_OPTIMALLY', 'completed', 'TimeTriggeredPlan')"
        )
        for run_id in (1, 2, 3):
            con.execute(
                """
                INSERT INTO state_action_transitions VALUES (
                    ?, 1, '1/1', 'end', 'move', 'instantaneous',
                    ['car0', 'j0', 'j1', 'road0'], '1/1',
                    ?, [], [],
                    ?, [], []
                )
                """,
                [run_id, _state_facts(), _next_state_facts()],
            )
            con.execute(
                """
                INSERT INTO state_action_transitions VALUES (
                    ?, 2, '2/1', 'end', 'move', 'instantaneous',
                    ['car0', 'j1', 'j0', 'road0'], '1/1',
                    ?, [], [],
                    ?, [], []
                )
                """,
                [run_id, _next_state_facts(), _final_state_facts()],
            )


class _AcceptAnyEngine:
    def __init__(self) -> None:
        self._satisfied = False

    def goals_satisfied(self) -> bool:
        return self._satisfied

    def current_facts(self) -> tuple[tuple[str, ...], ...]:
        return tuple((item["name"], *item["arguments"]) for item in _state_facts())

    def apply_action(self, _name: str, _arguments: tuple[str, ...], *, finish: bool = True) -> None:
        self._satisfied = True


def _state_facts() -> list[dict[str, object]]:
    return [
        {"name": "same_line", "arguments": ["j0", "j1"]},
        {"name": "clear", "arguments": ["j1"]},
        {"name": "at_car_jun", "arguments": ["car0", "j0"]},
        {"name": "road_connect", "arguments": ["road0", "j0", "j1"]},
    ]


def _next_state_facts() -> list[dict[str, object]]:
    return [
        {"name": "same_line", "arguments": ["j0", "j1"]},
        {"name": "clear", "arguments": ["j0"]},
        {"name": "at_car_jun", "arguments": ["car0", "j1"]},
        {"name": "road_connect", "arguments": ["road0", "j0", "j1"]},
    ]


def _final_state_facts() -> list[dict[str, object]]:
    return [
        {"name": "same_line", "arguments": ["j0", "j1"]},
        {"name": "clear", "arguments": ["j1"]},
        {"name": "at_car_jun", "arguments": ["car0", "j0"]},
        {"name": "road_connect", "arguments": ["road0", "j0", "j1"]},
    ]
