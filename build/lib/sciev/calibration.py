import argparse
import json
import math
import os
import re
from collections.abc import Mapping
from contextlib import contextmanager
from numbers import Integral, Real
from pathlib import Path

import torch

from . import decisions, metrics, protocol


__all__ = ["fit_calibration", "save_calibration", "load_calibration", "resolve_inference_settings"]

_INFERENCE_FIELDS = {"head_kind", "mode", "layers", "canonical_order", "seq_len",
                     "encoding", "strict_inputs", "decision_type"}
_KINDS = ("choice", "noul", "score")
_LIMITATIONS = [
    "Calibration uses supplied labels, which may be weak or synthetic; their scientific validity is not verified.",
    "Acceptance controls empirical dev risk only, not statistical deployment risk; freeze before independent test evaluation.",
    "Split checks cover exact inputs and declared identifiers, not unreported document overlap or paraphrase leakage.",
    "The checkpoint digest does not independently verify the supplied runtime model or full external backbone/revision/adapter provenance.",
    "Scientific performance claims remain pending GPU reevaluation with independent data and frozen calibration.",
]


def _object(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return value


def _positive_temperature(value):
    value = _real(value, "temperature")
    if value <= 0:
        raise ValueError("temperature must be positive")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _choice(value, name, choices):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {choices}")
    return value


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _layers(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("layers must be a nonempty list or tuple of integer indices")
    if any(isinstance(index, bool) or not isinstance(index, Integral) for index in value):
        raise ValueError("layers must contain integer non-boolean indices")
    return [int(index) for index in value]


def _mode(value):
    value = _text(value, "mode").removeprefix("r2_")
    return _choice(value, "mode", ("marker", "spanpool"))


def _sources(checkpoint_data):
    checkpoint_data = _object(checkpoint_data, "checkpoint")
    inference = _object(checkpoint_data.get("inference", {}), "checkpoint inference")
    meta = _object(checkpoint_data.get("meta", {}), "checkpoint meta")
    return (inference, checkpoint_data, meta)


def _stored(sources, *names):
    for source in sources:
        for name in names:
            if source.get(name) is not None:
                return source[name]
    return None


def _setting(name, supplied, stored, validate, default=None):
    previous = validate(stored) if stored is not None else None
    if supplied is None:
        return previous if previous is not None else default
    selected = validate(supplied)
    if previous is not None and selected != previous:
        raise ValueError(f"{name} conflicts with stored inference settings")
    return selected


def resolve_inference_settings(checkpoint_data, *, mode=None, layers=None, canonical=None,
                               strict=None, decision_type=None):
    sources = _sources(checkpoint_data)
    model_config = _object(checkpoint_data.get("model_config", {}), "checkpoint model_config")
    training_data = checkpoint_data.get("training_data")
    training_data = {} if training_data is None else _object(training_data, "checkpoint training_data")
    head_kind = _stored(sources, "head_kind")
    head_kind = "mlp" if head_kind is None else _choice(head_kind, "head_kind", ("mlp", "attnpool"))
    count = _stored(sources, "n_layers")
    count = None if count is None else _integer(count, "n_layers", 1)
    stored_layers = _stored(sources, "layers", "layers_list", "r2_layers")
    if isinstance(stored_layers, str):
        stored_layers = _parse_layers(stored_layers)
    stored_layers = None if stored_layers is None else _layers(stored_layers)
    requested_layers = None if layers is None else _layers(layers)
    if head_kind == "mlp":
        effective_layers = [-1]
    else:
        if stored_layers is not None and requested_layers is not None and stored_layers != requested_layers:
            raise ValueError("layers conflict with stored inference settings")
        effective_layers = stored_layers if requested_layers is None else requested_layers
        if effective_layers is None:
            if count is not None and count > 1:
                raise ValueError("legacy multi-layer attention requires explicit --r2-layers indices")
            effective_layers = [-1]
        if count is not None and len(effective_layers) != count:
            raise ValueError("layers count does not match checkpoint n_layers")
    resolved_mode = _setting("mode", mode, _stored(sources, "mode", "r2_mode"), _mode, "marker")
    if head_kind == "attnpool" and resolved_mode != "spanpool":
        raise ValueError("attnpool requires spanpool mode")
    resolved_canonical = _setting(
        "canonical_order", canonical, _stored(sources, "canonical_order", "canonical"),
        lambda value: _boolean(value, "canonical_order"), False)
    resolved_strict = _setting(
        "strict_inputs", strict, _stored(sources, "strict_inputs", "strict"),
        lambda value: _boolean(value, "strict_inputs"))
    resolved_type = _setting(
        "decision_type", decision_type, _stored(sources, "decision_type"),
        lambda value: _choice(value, "decision_type", _KINDS))
    encoding = _stored(sources, "encoding")
    if encoding is None:
        encoding = training_data.get("encoding")
    if encoding is not None:
        encoding = _text(encoding, "encoding")
    seq_len = _stored(sources, "seq_len")
    if seq_len is None:
        seq_len = model_config.get("seq_len")
    if seq_len is not None:
        seq_len = _integer(seq_len, "seq_len", 1)
    return {"head_kind": head_kind, "mode": resolved_mode, "layers": effective_layers,
            "canonical_order": resolved_canonical, "seq_len": seq_len, "encoding": encoding,
            "strict_inputs": resolved_strict, "decision_type": resolved_type}


def _validated_rows(values, role):
    if isinstance(values, (str, bytes, Mapping, set, frozenset)):
        raise ValueError(f"{role} must be a nonempty ordered iterable of decision rows")
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(f"{role} must be a nonempty ordered iterable of decision rows") from exc
    if not values:
        raise ValueError(f"{role} is empty; at least one decision is required")
    return [decisions.validate_decision_row(row) for row in values]


def _policy_parameters(error_budget, min_accepted):
    error_budget = _real(error_budget, "error_budget")
    if not 0 <= error_budget <= 1:
        raise ValueError("error_budget must be in [0, 1]")
    return error_budget, _integer(min_accepted, "min_accepted", 1)


def _check_fingerprint(expected, path):
    actual = protocol.file_fingerprint(path)
    if (actual["sha256"].lower() != expected["sha256"].lower()
            or actual["bytes"] != expected["bytes"]):
        raise ValueError("checkpoint fingerprint mismatch: checkpoint digest or bytes changed")


def _match_contract(contract, inference, role):
    if len(contract["kinds"]) > 1:
        raise ValueError(f"{role} cannot mix decision types")
    if contract["encoding"] != inference["encoding"]:
        raise ValueError(f"{role} encoding conflicts with calibration inference protocol")
    if contract["kinds"] and contract["kinds"] != [inference["decision_type"]]:
        raise ValueError(f"{role} decision_type conflicts with calibration inference protocol")
    if inference["decision_type"] == "noul" and contract["option_counts"] != [2]:
        raise ValueError("noul requires exactly two options")


def _prepare_calibration(checkpoint_path, temperature_rows, acceptance_rows, train_rows, *,
                         mode, layers, canonical, strict, decision_type, error_budget, min_accepted):
    error_budget, min_accepted = _policy_parameters(error_budget, min_accepted)
    named_rows = {"temperature_dev": _validated_rows(temperature_rows, "temperature_dev")}
    if acceptance_rows is not None:
        named_rows["acceptance_dev"] = _validated_rows(acceptance_rows, "acceptance_dev")
    if train_rows is not None:
        named_rows["train_reference"] = _validated_rows(train_rows, "train_reference")
    checkpoint = protocol.file_fingerprint(checkpoint_path)
    checkpoint_data = _object(torch.load(checkpoint_path, map_location="cpu", weights_only=True), "checkpoint")
    _check_fingerprint(checkpoint, checkpoint_path)
    inference = resolve_inference_settings(checkpoint_data, mode=mode, layers=layers,
                                           canonical=canonical, strict=strict, decision_type=decision_type)
    split_checks = protocol.assert_disjoint_splits(named_rows)
    contracts = split_checks["splits"]
    if any(len(contract["kinds"]) > 1 for contract in contracts.values()):
        raise ValueError("calibration and reference datasets cannot mix decision types")
    calibration_data = {name: contract for name, contract in contracts.items() if name != "train_reference"}
    kinds = {kind for contract in calibration_data.values() for kind in contract["kinds"]}
    if len(kinds) > 1:
        raise ValueError("temperature_dev and acceptance_dev cannot mix decision types")
    if inference["decision_type"] is None and kinds:
        inference["decision_type"] = next(iter(kinds))
    original_encoding = inference["encoding"]
    inference["encoding"] = contracts["temperature_dev"]["encoding"]
    if inference["strict_inputs"] is None:
        inference["strict_inputs"] = inference["encoding"] == decisions.ENCODING_VERSION
    for name, contract in calibration_data.items():
        _match_contract(contract, inference, name)
    combined = [row for name in calibration_data for row in named_rows[name]]
    training_check = protocol.assert_checkpoint_disjoint(checkpoint_data, combined, role="calibration")
    encoding_status = ("unverified" if original_encoding is None else
                       "matched" if original_encoding == inference["encoding"] else "changed")
    sources = _sources(checkpoint_data)
    provenance = {
        "training_overlap_check": training_check,
        "split_checks": split_checks,
        "encoding_check": {"status": encoding_status, "checkpoint_encoding": original_encoding,
                           "calibration_encoding": inference["encoding"],
                           "limitation": "Legacy encoding changes or undeclared training encoding do not validate historical results."},
        "inference_resolution": {
            "undeclared_checkpoint_fields": sorted(field for field in _INFERENCE_FIELDS
                                                   if _stored(sources, field) is None),
            "mlp_layers": "effective last hidden layer [-1]; legacy requested multi-layer indices never affected MLP",
            "default_mode": "marker when neither caller nor checkpoint declares mode",
            "default_canonical_order": "false when neither caller nor checkpoint declares canonical ordering",
            "strict_default": "true for systemone-v2, false for legacy encodings when undeclared",
        },
        "model_provenance": {
            "checkpoint_contains_backbone": isinstance(checkpoint_data.get("model"), Mapping),
            "runtime_weights_verified": False,
            "full_backbone_provenance": "unverified",
            "saved_model_config_available": isinstance(checkpoint_data.get("model_config"), Mapping),
            "saved_model_config_fields": sorted(key for key in checkpoint_data.get("model_config", {})
                                                if isinstance(key, str)),
        },
        "label_quality": "unverified; supplied gold labels may be weak or synthetic",
    }
    for key in ("hf_backbone", "hf_revision", "revision", "lora_adapter"):
        value = _stored(sources, key)
        if value is not None:
            provenance["model_provenance"][key] = _text(value, key)
    if train_rows is not None:
        provenance["train_reference_check"] = {
            "status": "verified", "reason": "no_known_overlap_with_supplied_reference",
            "scope": split_checks["scope"],
            "limitation": "A supplied reference is not proof of the checkpoint's actual training identity.",
        }
    return {"checkpoint": checkpoint, "inference": inference, "named_rows": named_rows,
            "calibration_data": calibration_data, "provenance": provenance,
            "error_budget": error_budget, "min_accepted": min_accepted}


@contextmanager
def _evaluation_modes(*objects):
    modules, adapters, roots, seen = [], [], [], set()
    for obj in objects:
        if isinstance(obj, torch.nn.Module):
            roots.append(obj)
            for module in obj.modules():
                if id(module) not in seen:
                    seen.add(id(module))
                    modules.append((module, module.training))
        elif (callable(getattr(obj, "eval", None)) and callable(getattr(obj, "train", None))
              and isinstance(getattr(obj, "training", None), bool) and id(obj) not in seen):
            seen.add(id(obj))
            adapters.append((obj, obj.training))
            roots.append(obj)
    try:
        for obj in roots:
            obj.eval()
        yield
    finally:
        for obj, previous in adapters:
            obj.train(previous)
        for module, previous in modules:
            module.training = previous


def _fit_prepared(model, head, prepared, checkpoint_path, device):
    from . import eval as rj_eval
    from .model import AttnPoolHead, DecisionHead

    _check_fingerprint(prepared["checkpoint"], checkpoint_path)
    inference = dict(prepared["inference"])
    seq_len = _integer(getattr(model, "seq_len", None), "model.seq_len", 1)
    if inference["seq_len"] is not None and inference["seq_len"] != seq_len:
        raise ValueError("actual model seq_len conflicts with stored inference settings")
    inference["seq_len"] = seq_len
    if isinstance(head, (AttnPoolHead, DecisionHead)):
        actual_kind = "attnpool" if isinstance(head, AttnPoolHead) else "mlp"
        if inference["head_kind"] != actual_kind:
            raise ValueError("actual head_kind conflicts with checkpoint inference")
        if actual_kind == "attnpool" and head.n_layers != len(inference["layers"]):
            raise ValueError("actual attention head layer count conflicts with inference layers")
    truncation = {}
    for role in prepared["calibration_data"]:
        counts = {"context_rows": 0, "option_rows": 0}
        for row in prepared["named_rows"][role]:
            layout = decisions.prepare_decision(
                model, row["ctx"], row["opts"], mode=inference["mode"],
                canonical=inference["canonical_order"], strict=inference["strict_inputs"])
            counts["context_rows"] += int(layout.context_truncated > 0)
            counts["option_rows"] += int(any(layout.option_truncated))
        truncation[role] = counts
    acceptance_policy = None
    with _evaluation_modes(model, head), torch.no_grad():
        temperature, _ = rj_eval.fit_r2_temperature_decisions(
            model, head, prepared["named_rows"]["temperature_dev"], device,
            mode=inference["mode"], layers=tuple(inference["layers"]),
            canonical=inference["canonical_order"], strict=inference["strict_inputs"], return_logits=True)
        temperature = _positive_temperature(temperature)
        if "acceptance_dev" in prepared["named_rows"]:
            confidences, corrects = [], []
            for row in prepared["named_rows"]["acceptance_dev"]:
                logits, layout = decisions.decision_logits(
                    model, head, row["ctx"], row["opts"], device, mode=inference["mode"],
                    layers=tuple(inference["layers"]), canonical=inference["canonical_order"],
                    strict=inference["strict_inputs"])
                probabilities = decisions.decision_probabilities(logits, temperature)
                prediction = decisions.decision_prediction(logits, layout)
                confidences.append(max(probabilities))
                corrects.append(int(prediction == row["gold"]))
            acceptance_policy = metrics.fit_acceptance_policy(
                confidences, corrects, error_budget=prepared["error_budget"], min_accepted=prepared["min_accepted"])
    _check_fingerprint(prepared["checkpoint"], checkpoint_path)
    provenance = dict(prepared["provenance"], truncation=truncation)
    artifact = {"schema_version": 1, "kind": "r2_calibration", "checkpoint": prepared["checkpoint"],
                "inference": inference, "temperature": temperature, "acceptance_policy": acceptance_policy,
                "calibration_data": prepared["calibration_data"], "provenance": provenance,
                "limitations": list(_LIMITATIONS)}
    if any(sum(counts.values()) for counts in truncation.values()):
        artifact["limitations"].append(
            "Authorized truncation occurred; identity checks use declared original inputs, not all truncated-input collisions.")
    _validate_artifact(artifact)
    return artifact


def fit_calibration(model, head, temperature_rows, acceptance_rows=None, *, checkpoint_path,
                    device="cpu", mode="spanpool", layers=(-1,), canonical=True, decision_type=None,
                    strict=None, train_rows=None, error_budget=0.05, min_accepted=20):
    prepared = _prepare_calibration(
        checkpoint_path, temperature_rows, acceptance_rows, train_rows, mode=mode, layers=layers,
        canonical=canonical, strict=strict, decision_type=decision_type,
        error_budget=error_budget, min_accepted=min_accepted)
    return _fit_prepared(model, head, prepared, checkpoint_path, device)


def _hashes(value, name):
    if (not isinstance(value, list) or not value
            or any(not isinstance(item, str) or re.fullmatch(r"[0-9a-fA-F]{64}", item) is None
                   for item in value)):
        raise ValueError(f"{name} must be a nonempty list of SHA256 identity hashes")
    normalized = {item.lower() for item in value}
    if len(normalized) != len(value):
        raise ValueError(f"{name} must not contain duplicate identity hashes")
    return normalized


def _validate_contract(contract, name):
    contract = _object(contract, name)
    if _integer(contract.get("contract_version"), f"{name}.contract_version", 1) != 1:
        raise ValueError(f"unsupported {name} contract_version")
    count = _integer(contract.get("n_rows"), f"{name}.n_rows", 1)
    _text(contract.get("encoding"), f"{name}.encoding")
    kinds = contract.get("kinds")
    if not isinstance(kinds, list) or len(kinds) > 1 or any(kind not in _KINDS for kind in kinds):
        raise ValueError(f"{name} kinds must declare at most one decision type")
    option_counts = contract.get("option_counts")
    if not isinstance(option_counts, list) or not option_counts:
        raise ValueError(f"{name} requires option_counts")
    for value in option_counts:
        _integer(value, f"{name}.option_counts", 2)
    identities = _hashes(contract.get("identity_hashes"), f"{name}.identity_hashes")
    for key in ("input_hashes", "effective_input_hashes"):
        inputs = _hashes(contract.get(key), f"{name}.{key}")
        if not inputs <= identities or len(inputs) > count:
            raise ValueError(f"{name}.{key} is inconsistent with identity_hashes or n_rows")
    return identities


def _validate_inference(inference):
    inference = _object(inference, "inference")
    if set(inference) != _INFERENCE_FIELDS:
        raise ValueError("inference requires the complete supported protocol fields and no temperature")
    kind = _choice(inference["head_kind"], "inference head_kind", ("mlp", "attnpool"))
    mode = _choice(inference["mode"], "inference mode", ("marker", "spanpool"))
    layers = _layers(inference["layers"])
    if kind == "mlp" and layers != [-1]:
        raise ValueError("inference MLP layers must be effective [-1]")
    if kind == "attnpool" and mode != "spanpool":
        raise ValueError("inference attnpool requires spanpool mode")
    _boolean(inference["canonical_order"], "inference canonical_order")
    _boolean(inference["strict_inputs"], "inference strict_inputs")
    _integer(inference["seq_len"], "inference seq_len", 1)
    _text(inference["encoding"], "inference encoding")
    if inference["decision_type"] is not None:
        _choice(inference["decision_type"], "inference decision_type", _KINDS)


def _validate_policy_audit(policy, contract):
    metrics.evaluate_acceptance_policy([], [], policy)
    count = _integer(policy.get("dev_n"), "policy dev_n", 1)
    accepted = _integer(policy.get("dev_accepted"), "policy dev_accepted")
    errors = _integer(policy.get("dev_errors"), "policy dev_errors")
    budget, minimum = _policy_parameters(policy.get("error_budget"), policy.get("min_accepted"))
    if count != contract["n_rows"] or not errors <= accepted <= count:
        raise ValueError("acceptance policy counts conflict with acceptance_dev")
    expected_risk = errors / accepted if accepted else None
    if policy.get("dev_coverage") != accepted / count or policy.get("dev_risk") != expected_risk:
        raise ValueError("acceptance policy dev coverage/risk is inconsistent with its counts")
    if (policy["threshold"] is None) != (accepted == 0):
        raise ValueError("acceptance policy threshold conflicts with dev_accepted")
    if accepted and (accepted < minimum or expected_risk > budget):
        raise ValueError("acceptance policy violates its empirical dev support or error budget")
    if policy.get("guarantee") is not False:
        raise ValueError("acceptance policy cannot claim a deployment guarantee")


def _validate_artifact(artifact):
    artifact = _object(artifact, "calibration artifact")
    required = {"schema_version", "kind", "checkpoint", "inference", "temperature", "acceptance_policy",
                "calibration_data", "provenance", "limitations"}
    if not required <= artifact.keys():
        raise ValueError("calibration artifact is missing required schema fields")
    if _integer(artifact["schema_version"], "schema_version", 1) != 1 or artifact["kind"] != "r2_calibration":
        raise ValueError("unsupported calibration schema_version or kind")
    _positive_temperature(artifact["temperature"])
    _validate_inference(artifact["inference"])
    fingerprint = _object(artifact["checkpoint"], "checkpoint fingerprint")
    _text(fingerprint.get("path"), "checkpoint path")
    _hashes([fingerprint.get("sha256")], "checkpoint sha256")
    _integer(fingerprint.get("bytes"), "checkpoint bytes", 1)
    contracts = _object(artifact["calibration_data"], "calibration_data")
    if "temperature_dev" not in contracts or set(contracts) - {"temperature_dev", "acceptance_dev"}:
        raise ValueError("calibration_data requires temperature_dev and optional acceptance_dev")
    previous = set()
    for role, contract in contracts.items():
        identities = _validate_contract(contract, role)
        _match_contract(contract, artifact["inference"], role)
        if identities & previous:
            raise ValueError("calibration dataset identity overlap between calibration roles")
        previous.update(identities)
    policy = artifact["acceptance_policy"]
    if policy is None:
        if "acceptance_dev" in contracts:
            raise ValueError("acceptance_dev requires a fitted acceptance_policy, including accept-none policies")
    else:
        _object(policy, "acceptance_policy")
        if "acceptance_dev" not in contracts:
            raise ValueError("acceptance_policy requires acceptance_dev provenance")
        _validate_policy_audit(policy, contracts["acceptance_dev"])
    provenance = _object(artifact["provenance"], "provenance")
    training = _object(provenance.get("training_overlap_check"), "training_overlap_check")
    _choice(training.get("status"), "training_overlap_check status", ("verified", "unverified"))
    _object(training.get("scope"), "training_overlap_check scope")
    splits = _object(provenance.get("split_checks"), "split_checks")
    if splits.get("status") != "verified":
        raise ValueError("split_checks must record a scoped verified check")
    _object(splits.get("scope"), "split_checks scope")
    limitations = artifact["limitations"]
    if not isinstance(limitations, list) or not limitations:
        raise ValueError("calibration requires explicit limitations")
    for limitation in limitations:
        _text(limitation, "limitation")
    try:
        json.dumps(artifact, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("calibration artifact must contain finite JSON-safe values") from exc


def save_calibration(path, artifact):
    _validate_artifact(artifact)
    payload = json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload)


def _invalid_json_constant(value):
    raise ValueError(f"calibration JSON cannot contain nonfinite constant {value}")


def load_calibration(path, checkpoint_path, *, evaluation_rows=None, expected_inference=None):
    with Path(path).open(encoding="utf-8") as stream:
        artifact = json.load(stream, parse_constant=_invalid_json_constant)
    _validate_artifact(artifact)
    _check_fingerprint(artifact["checkpoint"], checkpoint_path)
    inference = artifact["inference"]
    if expected_inference is not None:
        expected = dict(_object(expected_inference, "expected_inference"))
        if set(expected) - _INFERENCE_FIELDS:
            raise ValueError("expected_inference contains unsupported fields; temperature is fitted separately")
        if "layers" in expected:
            expected["layers"] = _layers(expected["layers"])
        _validate_inference(dict(inference, **expected))
        for key, value in expected.items():
            if value != inference[key]:
                raise ValueError(f"expected inference {key} conflicts with frozen calibration protocol")
    if evaluation_rows is not None:
        evaluation = protocol.dataset_contract(_validated_rows(evaluation_rows, "evaluation"))
        _match_contract(evaluation, inference, "evaluation")
        identities = set(evaluation["identity_hashes"])
        for role, contract in artifact["calibration_data"].items():
            if identities & {identity.lower() for identity in contract["identity_hashes"]}:
                raise ValueError(f"evaluation/{role} identity overlap; calibration and test must be disjoint")
    return artifact


def _parse_layers(value):
    try:
        parsed = [int(part.strip()) for part in value.split(",")]
    except (AttributeError, ValueError) as exc:
        raise ValueError("--r2-layers must be comma-separated integer indices") from exc
    return _layers(parsed)


def _output_preflight(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite calibration output: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"calibration output parent must already exist: {path.parent}")
    if not os.access(path.parent, os.W_OK):
        raise PermissionError(f"calibration output parent is not writable: {path.parent}")


def _native_config(path):
    if path is None:
        return None
    import yaml
    from .model import DEFAULT_CONFIG

    document = _object(yaml.safe_load(Path(path).read_text(encoding="utf-8")), "native YAML config")
    config = dict(_object(document.get("model"), "native YAML model config"))
    if set(config) - DEFAULT_CONFIG.keys():
        raise ValueError("native YAML model config contains unsupported architecture fields")
    for key, value in config.items():
        if isinstance(DEFAULT_CONFIG[key], bool):
            _boolean(value, f"native model config {key}")
        else:
            _integer(value, f"native model config {key}", 1)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit checkpoint-bound R2 calibration on source-disjoint dev data.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--temperature-dev", required=True)
    parser.add_argument("--acceptance-dev")
    parser.add_argument("--train-reference")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config")
    parser.add_argument("--r2-mode", choices=("marker", "spanpool"))
    parser.add_argument("--r2-layers", type=_parse_layers)
    canonical = parser.add_mutually_exclusive_group()
    canonical.add_argument("--canonical-order", dest="canonical", action="store_true")
    canonical.add_argument("--no-canonical-order", dest="canonical", action="store_false")
    strict = parser.add_mutually_exclusive_group()
    strict.add_argument("--strict-inputs", dest="strict", action="store_true")
    strict.add_argument("--allow-truncation", dest="strict", action="store_false")
    parser.set_defaults(canonical=None, strict=None)
    parser.add_argument("--decision-type", choices=_KINDS)
    parser.add_argument("--error-budget", type=float, default=0.05)
    parser.add_argument("--min-accepted", type=int, default=20)
    args = parser.parse_args(argv)
    _policy_parameters(args.error_budget, args.min_accepted)
    _output_preflight(args.out)
    torch.device(args.device)
    config = _native_config(args.config)
    from .data import load_decisions_ids

    paths = {"temperature_dev": args.temperature_dev}
    if args.acceptance_dev is not None:
        paths["acceptance_dev"] = args.acceptance_dev
    if args.train_reference is not None:
        paths["train_reference"] = args.train_reference
    dataset_files = {role: protocol.file_fingerprint(path) for role, path in paths.items()}
    datasets = {role: load_decisions_ids(path) for role, path in paths.items()}
    prepared = _prepare_calibration(
        args.ckpt, datasets["temperature_dev"], datasets.get("acceptance_dev"), datasets.get("train_reference"),
        mode=args.r2_mode, layers=args.r2_layers, canonical=args.canonical, strict=args.strict,
        decision_type=args.decision_type, error_budget=args.error_budget, min_accepted=args.min_accepted)
    for role, path in paths.items():
        if protocol.file_fingerprint(path) != dataset_files[role]:
            raise ValueError(f"{role} dataset changed during calibration preflight")
    model_provenance = prepared["provenance"]["model_provenance"]
    if config is not None and model_provenance.get("hf_backbone") is not None:
        raise ValueError("--config is native YAML and cannot silently override an HF backbone")
    declared_seq_len = prepared["inference"]["seq_len"]
    if config and "seq_len" in config and declared_seq_len is not None and config["seq_len"] != declared_seq_len:
        raise ValueError("native config seq_len conflicts with checkpoint inference")
    from .model import DEFAULT_CONFIG, load_decision

    configured_fields = set(model_provenance["saved_model_config_fields"]) | set(config or {})
    potential_defaults = None if model_provenance.get("hf_backbone") is not None else {
        key: DEFAULT_CONFIG[key] for key in sorted(DEFAULT_CONFIG) if key not in configured_fields}
    prepared["provenance"]["dataset_files"] = dataset_files
    prepared["provenance"]["model_loading"] = {
        "loader": "load_decision", "config_file": protocol.file_fingerprint(args.config) if args.config else None,
        "explicit_model_config": config,
        "potential_native_defaults": potential_defaults,
        "legacy_defaults_possible": None if potential_defaults is None else bool(potential_defaults),
        "limitation": "Undeclared native architecture fields may use listed defaults or be inferred from weights; external HF defaults are unverified. seq_len is recorded from the actual model.",
    }

    model, head = load_decision(args.ckpt, config=config, device=args.device)
    artifact = _fit_prepared(model, head, prepared, args.ckpt, args.device)
    save_calibration(args.out, artifact)
    print(json.dumps({"calibration": str(args.out), "temperature": artifact["temperature"],
                      "training_overlap_check": artifact["provenance"]["training_overlap_check"]["status"],
                      "limitations": artifact["limitations"]}, allow_nan=False))
    return artifact


if __name__ == "__main__":
    main()
