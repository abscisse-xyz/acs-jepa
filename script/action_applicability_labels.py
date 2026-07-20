"""Build JSON-safe offline applicability examples for action diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import AbstractSet, Any

from acs_jepa.graph.schemas import GroundAction
from action_diag_common import action_payload
from action_negative_sampling import NegativeActionExample

ActionKey = tuple[str, tuple[str, ...]]


@dataclass(frozen=True)
class ApplicabilityExampleBatch:
    """Labeled positive/negative examples and deterministic summary counters."""

    examples: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def build_applicability_examples(
    true_action: GroundAction,
    negatives: Sequence[NegativeActionExample],
    *,
    applicable_action_keys: AbstractSet[ActionKey] | None,
) -> ApplicabilityExampleBatch:
    """Create one trace positive plus deduplicated negative applicability examples."""

    examples: list[dict[str, Any]] = []
    seen: set[ActionKey] = set()

    def label(action: GroundAction) -> bool | None:
        if applicable_action_keys is None:
            return None
        return _action_key(action) in applicable_action_keys

    true_key = _action_key(true_action)
    seen.add(true_key)
    examples.append(
        {
            "kind": "positive_trace",
            "category": "trace",
            "action": action_payload(true_action),
            "changed_roles": [],
            "applicable": label(true_action),
        }
    )
    for negative in negatives:
        key = _action_key(negative.action)
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            {
                "kind": "negative",
                "category": negative.category,
                "action": action_payload(negative.action),
                "changed_roles": list(negative.changed_roles),
                "applicable": label(negative.action),
            }
        )

    return ApplicabilityExampleBatch(
        examples=tuple(examples),
        summary=_summary(true_action, examples, applicable_action_keys),
    )


def _summary(
    true_action: GroundAction,
    examples: Sequence[dict[str, Any]],
    applicable_action_keys: AbstractSet[ActionKey] | None,
) -> dict[str, Any]:
    kind_counts = Counter(str(example["kind"]) for example in examples)
    category_counts = Counter(str(example["category"]) for example in examples)
    applicability_counts = Counter({"applicable": 0, "inapplicable": 0, "unknown": 0})
    for example in examples:
        if example["applicable"] is None:
            applicability_counts["unknown"] += 1
        elif example["applicable"]:
            applicability_counts["applicable"] += 1
        else:
            applicability_counts["inapplicable"] += 1
    return {
        "examples": len(examples),
        "kind_counts": dict(sorted(kind_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "applicability_counts": dict(sorted(applicability_counts.items())),
        "true_action_applicable": None
        if applicable_action_keys is None
        else _action_key(true_action) in applicable_action_keys,
    }


def _action_key(action: GroundAction) -> ActionKey:
    return action.name, tuple(action.arguments)
