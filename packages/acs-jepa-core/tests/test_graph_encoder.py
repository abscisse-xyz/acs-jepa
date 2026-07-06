from __future__ import annotations

from pathlib import Path

from acs_jepa.graph import GraphEncoder, build_state_graph, parse_domain_problem
from torch_geometric.loader import DataLoader

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

  (:action build
    :parameters (?r - road ?from - junction ?to - junction)
    :precondition (same_line ?from ?to)
    :effect (road_connect ?r ?from ?to)
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


def test_graph_encoder_returns_single_graph_embeddings(tmp_path: Path) -> None:
    parsed, graph = _parsed_and_graph(tmp_path)
    encoder = GraphEncoder.from_parsed_problem(parsed, hidden_dim=16, embed_dim=8, num_layers=2)

    output = encoder(graph)

    assert output.graph_embedding.shape == (1, 8)
    assert output.object_embeddings.shape == (graph.num_objects, 8)
    assert output.object_ids.tolist() == [0, 1, 2, 3]
    assert output.object_batch.tolist() == [0, 0, 0, 0]


def test_graph_encoder_handles_batched_graphs(tmp_path: Path) -> None:
    parsed, graph = _parsed_and_graph(tmp_path)
    batch = next(iter(DataLoader([graph, graph], batch_size=2)))
    encoder = GraphEncoder.from_parsed_problem(parsed, hidden_dim=16, embed_dim=8, num_layers=2)

    output = encoder(batch)

    assert output.graph_embedding.shape == (2, 8)
    assert output.object_embeddings.shape == (graph.num_objects * 2, 8)
    assert output.object_batch.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_graph_encoder_handles_static_filtered_graphs(tmp_path: Path) -> None:
    parsed, _ = _parsed_and_graph(tmp_path)
    graph = build_state_graph(parsed, parsed.initial_atoms, include_static=False)
    encoder = GraphEncoder.from_parsed_problem(parsed, hidden_dim=16, embed_dim=8, num_layers=2)

    output = encoder(graph)

    assert output.graph_embedding.shape == (1, 8)
    assert output.object_embeddings.shape == (graph.num_objects, 8)


def test_graph_encoder_backpropagates_through_directed_edge_features(tmp_path: Path) -> None:
    parsed, graph = _parsed_and_graph(tmp_path)
    encoder = GraphEncoder.from_parsed_problem(parsed, hidden_dim=16, embed_dim=8, num_layers=2)

    output = encoder(graph)
    output.graph_embedding.sum().backward()

    direction_grad = encoder.edge_direction_embedding.embedding.weight.grad
    assert direction_grad is not None
    assert direction_grad.abs().sum().item() > 0


def _parsed_and_graph(tmp_path: Path):
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(DOMAIN)
    problem_path.write_text(PROBLEM)
    parsed = parse_domain_problem(domain_path, problem_path)
    return parsed, build_state_graph(parsed, parsed.initial_atoms)
