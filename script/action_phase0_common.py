"""Strict offline adapters shared by Updated Phase 0 diagnostics.

This module intentionally has no simulator dependency. Candidate applicability is
read only from the fixed, byte-identified offline manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from acs_jepa.architectures import ActionDecodingSpace
from acs_jepa.graph import build_state_graph
from acs_jepa_cli.config import load_config
from acs_jepa_cli.data import LoadedCorpus, load_corpus, split_corpus
from acs_jepa_cli.modeling import ModelBundle, build_model_bundle, vocab_sizes_from_dict
from omegaconf import OmegaConf

CANDIDATE_MANIFEST_BYTES = 117385
CANDIDATE_MANIFEST_COUNT = 604
CANDIDATE_MANIFEST_SHA256 = "bf6d11149cadf7a34c6c1520e28e9fe389c09c13ce53f3bd3f988f827e936ce9"
POPULATION_COUNT = 174780
SPLIT_SHA256 = "5397fc5e7820c9fdee3eb38c05278a3b680fb5ca8460d0bbe588ffa7ff22815c"
SCHEMAS = (
    "build_diagonal_oneway",
    "build_straight_oneway",
    "car_arrived",
    "car_start",
    "destroy_road",
    "move_car_in_road",
    "move_car_out_road",
)
GROUPS = tuple([f"p166:{index}" for index in range(23)] + [f"p192:{index}" for index in range(21)])
CATEGORY_COUNTS = {
    "trace": 44,
    "one_arg_substitution": 176,
    "random_same_schema": 176,
    "role_swap": 32,
    "random_other_schema": 176,
}
RECORD_KEYS = frozenset({"action", "applicability_label", "category", "group", "problem", "step"})
ACTION_KEYS = frozenset({"name", "arguments"})


class SourceState:
    """One manifest group reconciled to its strict recorded trajectory source."""

    def __init__(
        self,
        *,
        group: str,
        problem: str,
        step: int,
        parsed: Any,
        source_atoms: Any,
        trace_action: Any,
        manifest_records: tuple[Mapping[str, Any], ...],
    ) -> None:
        self.group = group
        self.problem = problem
        self.step = step
        self.parsed = parsed
        self.source_atoms = source_atoms
        self.trace_action = trace_action
        self.manifest_records = manifest_records


class SameSchemaPopulationBucket:
    """Canonical same-schema actions and identity rows for one source state."""

    def __init__(
        self,
        *,
        source: SourceState,
        space: ActionDecodingSpace,
        actions: tuple[Any, ...],
        identity_rows: tuple[dict[str, Any], ...],
    ) -> None:
        self.source = source
        self.space = space
        self.actions = actions
        self.identity_rows = identity_rows


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one canonical JSON document with one trailing newline."""

    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_and_validate_candidate_manifest(
    path: Path,
    *,
    require_fixed_identity: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and recursively validate the immutable 604-record manifest."""

    raw = path.read_bytes()
    try:
        records = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"candidate manifest is invalid UTF-8 JSON: {exc}") from exc
    if canonical_json_bytes(records) != raw:
        raise ValueError("candidate manifest is not canonical JSON with one trailing newline")
    if not isinstance(records, list):
        raise ValueError("candidate manifest must be a JSON list")
    _validate_manifest_records(records)
    digest = hashlib.sha256(raw).hexdigest()
    if require_fixed_identity and (
        len(raw) != CANDIDATE_MANIFEST_BYTES
        or len(records) != CANDIDATE_MANIFEST_COUNT
        or digest != CANDIDATE_MANIFEST_SHA256
    ):
        raise ValueError("candidate manifest identity does not match the fixed evidence")
    return records, {"path": str(path), "bytes": len(raw), "sha256": digest, "count": len(records)}


def _validate_manifest_records(records: Sequence[Any]) -> None:
    if len(records) != CANDIDATE_MANIFEST_COUNT:
        raise ValueError(f"candidate manifest count must be {CANDIDATE_MANIFEST_COUNT}")
    seen: set[bytes] = set()
    category_counts: Counter[str] = Counter()
    label_counts: Counter[bool] = Counter()
    group_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise ValueError(f"record {index} has unknown or missing keys")
        action = record["action"]
        if not isinstance(action, dict) or set(action) != ACTION_KEYS:
            raise ValueError(f"record {index} has invalid action keys")
        if not isinstance(action["name"], str) or action["name"] not in SCHEMAS:
            raise ValueError(f"record {index} has invalid action schema")
        if not isinstance(action["arguments"], list) or not all(isinstance(arg, str) for arg in action["arguments"]):
            raise ValueError(f"record {index} has invalid action arguments")
        if record["group"] not in GROUPS:
            raise ValueError(f"record {index} has invalid group")
        if (
            not isinstance(record["problem"], str)
            or not isinstance(record["step"], int)
            or isinstance(record["step"], bool)
        ):
            raise ValueError(f"record {index} has invalid problem/step")
        if record["group"] != f"{record['problem']}:{record['step']}":
            raise ValueError(f"record {index} group/step mapping is inconsistent")
        if record["category"] not in CATEGORY_COUNTS:
            raise ValueError(f"record {index} has invalid category")
        if type(record["applicability_label"]) is not bool:
            raise ValueError(f"record {index} has invalid label")
        encoded = canonical_json_bytes(record)
        if encoded in seen:
            raise ValueError(f"duplicate record at index {index}")
        seen.add(encoded)
        category_counts[record["category"]] += 1
        label_counts[record["applicability_label"]] += 1
        group_rows[record["group"]].append(record)
    if set(group_rows) != set(GROUPS):
        raise ValueError("candidate manifest group set drift")
    if dict(category_counts) != CATEGORY_COUNTS:
        raise ValueError(f"candidate category/trace count drift: {dict(category_counts)}")
    if label_counts != Counter({False: 542, True: 62}):
        raise ValueError(f"candidate label count drift: {dict(label_counts)}")
    for group in GROUPS:
        traces = [row for row in group_rows[group] if row["category"] == "trace"]
        if len(traces) != 1 or traces[0]["applicability_label"] is not True:
            raise ValueError(f"group {group} must have one applicable trace")


def reconcile_manifest_source_states(
    records: Sequence[Mapping[str, Any]],
    corpus: Any,
) -> list[SourceState]:
    """Recover and validate each manifest source directly from strict corpus states."""

    by_problem: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
    for trajectory, record in zip(corpus.trajectories, corpus.records, strict=True):
        by_problem[record.problem_name].append((trajectory, record))
    rows_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for row in records:
        if row["group"] not in rows_by_group:
            order.append(str(row["group"]))
        rows_by_group[str(row["group"])].append(row)

    sources: list[SourceState] = []
    for group in order:
        group_rows = rows_by_group[group]
        problem = str(group_rows[0]["problem"])
        step = int(group_rows[0]["step"])
        matches = []
        for trajectory, _record in by_problem.get(problem, []):
            local_step = step
            if 0 <= local_step < len(trajectory.actions):
                matches.append((trajectory, local_step))
        if len(matches) != 1:
            raise ValueError(f"group {group} does not map to exactly one strict corpus transition")
        trajectory, local_step = matches[0]
        trace_rows = [row for row in group_rows if row["category"] == "trace"]
        if len(trace_rows) != 1:
            raise ValueError(f"group {group} must have exactly one trace")
        trace_action = trajectory.actions[local_step]
        expected = (trace_rows[0]["action"]["name"], tuple(trace_rows[0]["action"]["arguments"]))
        actual = (trace_action.name, tuple(trace_action.arguments))
        if actual != expected:
            raise ValueError(f"group {group} trace action does not match strict corpus trajectory")
        sources.append(
            SourceState(
                group=group,
                problem=problem,
                step=step,
                parsed=corpus.parsed_problems[trajectory.problem_index],
                source_atoms=trajectory.states[local_step],
                trace_action=trace_action,
                manifest_records=tuple(group_rows),
            )
        )
    return sources


def enumerate_same_schema_population(
    sources: Sequence[SourceState],
) -> tuple[SameSchemaPopulationBucket, ...]:
    """Enumerate the canonical full same-schema population for fixed sources."""

    buckets = []
    for source in sources:
        space = ActionDecodingSpace.from_parsed_problem(source.parsed)
        actions = tuple(
            action for action in space.enumerate_ground_actions() if action.name == source.trace_action.name
        )
        if len(actions) < 2:
            raise ValueError(f"state/schema bucket {source.group} has fewer than two candidates")
        trace_key = (source.trace_action.name, tuple(source.trace_action.arguments))
        action_keys = [(action.name, tuple(action.arguments)) for action in actions]
        if action_keys.count(trace_key) != 1:
            raise ValueError(f"trace action for {source.group} does not occur exactly once in population")
        identity_rows = tuple(
            {
                "group": source.group,
                "problem": source.problem,
                "step": source.step,
                "action": {"name": action.name, "arguments": list(action.arguments)},
            }
            for action in actions
        )
        buckets.append(
            SameSchemaPopulationBucket(
                source=source,
                space=space,
                actions=actions,
                identity_rows=identity_rows,
            )
        )
    return tuple(buckets)


def population_identity(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_count: int = POPULATION_COUNT,
) -> dict[str, Any]:
    """Hash an ordered canonical JSON-lines population without writing it."""

    digest = hashlib.sha256()
    count = 0
    byte_count = 0
    for record in records:
        if set(record) != {"group", "problem", "step", "action"}:
            raise ValueError("population record has unknown or missing keys")
        raw = canonical_json_bytes(record)
        digest.update(raw)
        byte_count += len(raw)
        count += 1
    if count != expected_count:
        raise ValueError(f"population count is {count}, expected {expected_count}")
    return {"count": count, "bytes": byte_count, "sha256": digest.hexdigest()}


def file_identity(path: Path, *, manifest_count: int | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    identity: dict[str, Any] = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if manifest_count is not None:
        identity["count"] = manifest_count
    return identity


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(name)


def restore_diagnostic_checkpoint_modules(
    bundle: ModelBundle,
    checkpoint: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Strictly restore every configured module without simulator dependencies."""

    module_states = (
        ("jepa", "model_state_dict"),
        ("goal_head", "goal_head_state_dict"),
        ("action_contrastive_anchor", "action_contrastive_anchor_state_dict"),
        ("argument_reconstruction_head", "argument_reconstruction_head_state_dict"),
        ("applicability_head", "applicability_head_state_dict"),
    )
    restoration = {}
    for module_name, state_key in module_states:
        module = getattr(bundle, module_name)
        if module is None:
            restoration[module_name] = {"status": "disabled", "state_key": state_key}
            continue
        if state_key not in checkpoint:
            raise ValueError(f"{state_key} is missing for configured diagnostic module {module_name}")
        state = checkpoint[state_key]
        if state is None:
            raise ValueError(f"{state_key} is null for configured diagnostic module {module_name}")
        try:
            module.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            message = f"{state_key} is incompatible with configured diagnostic module {module_name}: {exc}"
            raise ValueError(message) from exc
        module.eval()
        restoration[module_name] = {"status": "restored", "state_key": state_key}
    return restoration


def load_checkpoint_bundle(
    dataset_dir: Path,
    checkpoint_path: Path,
    *,
    device_name: str,
    include_restoration_metadata: bool = False,
) -> Any:
    """Load fixed diagnostic state without importing simulator helpers."""

    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = OmegaConf.merge(load_config(None), OmegaConf.create(checkpoint["config"]))
    corpus = load_corpus([dataset_dir], strict=True)
    vocab_sizes = vocab_sizes_from_dict(checkpoint["vocab_sizes"])
    bundle = build_model_bundle(corpus.parsed_problems, config, device=device, vocab_sizes=vocab_sizes)
    restoration = restore_diagnostic_checkpoint_modules(bundle, checkpoint)
    result = (config, corpus, bundle, device)
    return (*result, restoration) if include_restoration_metadata else result


def select_split(corpus: LoadedCorpus, config: Any, split: str, *, seed: int) -> LoadedCorpus:
    if split == "all":
        return corpus
    split_seed = seed if config.data.split_seed is None else int(config.data.split_seed)
    splits = split_corpus(
        corpus,
        val_fraction=float(config.data.val_fraction),
        test_fraction=float(config.data.test_fraction),
        seed=split_seed,
    )
    return getattr(splits, split)


def encode_state(bundle: ModelBundle, parsed: Any, atoms: Any, *, device: torch.device) -> Any:
    state_graph = build_state_graph(parsed, atoms, include_static=True).to(device)
    return bundle.jepa.encode(state_graph)


def prepare_output_directory(
    root: Path,
    output: Path,
    root_identity: Mapping[str, Any],
    *,
    first_command: bool,
) -> None:
    """Enforce clean-root ownership and refuse only the command's destination."""

    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError("output destination must be inside the fixed phase root") from exc
    marker_path = root / "root_identity.json"
    expected = canonical_json_bytes(root_identity)
    if first_command:
        try:
            root.mkdir(parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(f"fixed output root already exists: {root}") from exc
        temporary = root / ".root_identity.json.tmp"
        temporary.write_bytes(expected)
        temporary.replace(marker_path)
    else:
        if not marker_path.is_file():
            raise ValueError(f"root identity marker is missing: {marker_path}")
        if marker_path.read_bytes() != expected:
            raise ValueError("root identity marker does not match immutable inputs")
    if output.exists():
        raise FileExistsError(f"destination already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
