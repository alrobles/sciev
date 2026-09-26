import copy
import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sciev.calibration import (
    fit_calibration, load_calibration, main, resolve_inference_settings, save_calibration,
)
from sciev.decisions import ENCODING_VERSION
from sciev.eval import eval_decisions_ids
from sciev.protocol import dataset_contract, file_fingerprint


def model():
    return SimpleNamespace(seq_len=64, mask_id=63,
                           tok_emb=SimpleNamespace(num_embeddings=64))


def rows(prefix, start, gold=1):
    return [{"ctx": [start + i], "opts": [[1], [2]], "gold": gold,
             "qid": f"{prefix}_{i}", "split_group": f"{prefix}_group_{i}",
             "kind": "choice", "encoding": ENCODING_VERSION}
            for i in range(4)]


def scores(backbone, head, ids, mode, bounds, layers):
    return torch.stack([ids[start].float() for start, _ in bounds])


def checkpoint(tmp_path):
    path = tmp_path / "head.pt"
    torch.save({"head_kind": "mlp", "n_layers": 1}, path)
    return path


def test_calibration_is_frozen_before_worse_test_risk(tmp_path):
    path = checkpoint(tmp_path)
    with patch("sciev.model.forward_feats", side_effect=scores):
        fitted = fit_calibration(model(), None, rows("temperature", 10), rows("policy", 20),
                                 checkpoint_path=path, device="cpu", canonical=True,
                                 decision_type="choice", min_accepted=2)
        assert fitted["acceptance_policy"]["dev_risk"] == 0.0
        output = tmp_path / "calibration.json"
        save_calibration(output, fitted)
        frozen = load_calibration(output, path, evaluation_rows=rows("test", 30, gold=0))
        before = copy.deepcopy(frozen)
        report = eval_decisions_ids(
            model(), None, rows("test", 30, gold=0), "cpu", canonical=True,
            temperature=frozen["temperature"], acceptance_policy=frozen["acceptance_policy"])
    assert report["selective"]["risk"] == 1.0
    assert frozen == before
    assert frozen["inference"]["encoding"] == ENCODING_VERSION
    assert frozen["provenance"]["training_overlap_check"]["status"] == "unverified"


def test_calibration_roles_cannot_reuse_source_groups(tmp_path):
    path = checkpoint(tmp_path)
    dev = rows("dev", 10)
    with pytest.raises(ValueError, match="overlap|disjoint|shared"):
        fit_calibration(model(), None, dev, [{**row, "qid": f"new_{i}"}
                                             for i, row in enumerate(dev)], checkpoint_path=path)


def test_saved_calibration_rejects_test_overlap_and_changed_weights(tmp_path):
    path = checkpoint(tmp_path)
    with patch("sciev.model.forward_feats", side_effect=scores):
        fitted = fit_calibration(model(), None, rows("dev", 10),
                                 checkpoint_path=path, canonical=True)
    output = tmp_path / "calibration.json"
    save_calibration(output, fitted)
    with pytest.raises(ValueError, match="overlap|disjoint|shared"):
        load_calibration(output, path, evaluation_rows=rows("dev", 10))
    torch.save({"changed": True}, path)
    with pytest.raises(ValueError, match="checkpoint|hash|fingerprint"):
        load_calibration(output, path)


def test_calibration_artifact_is_not_silently_overwritten(tmp_path):
    path = checkpoint(tmp_path)
    with patch("sciev.model.forward_feats", side_effect=scores):
        fitted = fit_calibration(model(), None, rows("dev", 10),
                                 checkpoint_path=path, canonical=True)
    output = tmp_path / "calibration.json"
    save_calibration(output, fitted)
    with pytest.raises(FileExistsError):
        save_calibration(output, fitted)


def test_frozen_calibration_checks_input_protocol(tmp_path):
    path = checkpoint(tmp_path)
    with patch("sciev.model.forward_feats", side_effect=scores):
        fitted = fit_calibration(model(), None, rows("dev", 10),
                                 checkpoint_path=path, canonical=True)
    output = tmp_path / "calibration.json"
    save_calibration(output, fitted)
    with pytest.raises(ValueError, match="inference|canonical|protocol"):
        load_calibration(output, path, expected_inference={"canonical_order": False})


@pytest.fixture
def fitted_artifact(tmp_path):
    path = checkpoint(tmp_path)
    with patch("sciev.eval.fit_r2_temperature_decisions", return_value=(1.5, [])), \
            patch("sciev.model.forward_feats", side_effect=scores):
        artifact = fit_calibration(
            model(), None, rows("temperature", 10), rows("acceptance", 20),
            checkpoint_path=path, min_accepted=2)
    return path, artifact


def write_jsonl(path, examples):
    path.write_text("".join(json.dumps(row) + "\n" for row in examples), encoding="utf-8")
    return path


def test_artifact_contract_and_reference_check_do_not_invent_training_provenance(tmp_path):
    path = checkpoint(tmp_path)
    dev = rows("temperature", 10)
    dev[0].update(raw_text="private scientific text", source_document_id="private-document")
    with patch("sciev.eval.fit_r2_temperature_decisions", return_value=(1.5, [])) as fitter:
        artifact = fit_calibration(model(), None, dev, checkpoint_path=path,
                                   train_rows=rows("reference", 30))
    assert artifact["schema_version"] == 1
    assert artifact["kind"] == "r2_calibration"
    assert artifact["checkpoint"] == file_fingerprint(path)
    assert artifact["temperature"] == 1.5
    assert math.isfinite(artifact["temperature"])
    assert artifact["acceptance_policy"] is None
    undeclared = artifact["provenance"]["inference_resolution"]["undeclared_checkpoint_fields"]
    assert undeclared == sorted(undeclared)
    assert artifact["calibration_data"] == {"temperature_dev": dataset_contract(dev)}
    assert artifact["provenance"]["training_overlap_check"]["status"] == "unverified"
    assert artifact["provenance"]["train_reference_check"]["status"] == "verified"
    assert artifact["provenance"]["split_checks"]["scope"]["document_disjointness"] == "not_proven"
    assert fitter.call_args.kwargs["return_logits"] is True
    assert fitter.call_args.kwargs["strict"] is True
    serialized = json.dumps(artifact, allow_nan=False)
    assert "private scientific text" not in serialized
    assert "private-document" not in serialized
    assert "temperature_0" not in serialized
    limitations = " ".join(artifact["limitations"]).lower()
    assert "weak" in limitations and "empirical" in limitations and "backbone" in limitations


@pytest.mark.parametrize("role", ["temperature", "acceptance"])
def test_checkpoint_training_overlap_fails_before_any_forward(tmp_path, role):
    path = checkpoint(tmp_path)
    training = rows(role, 10 if role == "temperature" else 20)
    torch.save({"head_kind": "mlp", "training_data": dataset_contract(training)}, path)
    with patch("sciev.eval.fit_r2_temperature_decisions") as fitter, \
            patch("sciev.model.forward_feats") as forward:
        with pytest.raises(ValueError, match="overlap"):
            fit_calibration(model(), None, rows("temperature", 10), rows("acceptance", 20),
                            checkpoint_path=path)
    fitter.assert_not_called()
    forward.assert_not_called()


def test_known_training_identity_is_scoped_verified(tmp_path):
    path = checkpoint(tmp_path)
    torch.save({"head_kind": "mlp", "training_data": dataset_contract(rows("train", 30))}, path)
    with patch("sciev.eval.fit_r2_temperature_decisions", return_value=(1.0, [])):
        artifact = fit_calibration(model(), None, rows("dev", 10), checkpoint_path=path)
    report = artifact["provenance"]["training_overlap_check"]
    assert report["status"] == "verified"
    assert report["scope"]["paraphrase_leakage"] == "not_checked"


@pytest.mark.parametrize("field,value", [
    ("gold", True), ("gold", 2), ("ctx", [64]),
    ("ctx", [1] * 40), ("opts", [[1], [1]]),
    ("kind", "score"), ("encoding", "legacy_ids"),
])
@pytest.mark.parametrize("role", ["temperature", "acceptance"])
def test_all_calibration_rows_are_validated_before_first_forward(tmp_path, field, value, role):
    path = checkpoint(tmp_path)
    temperature, acceptance = rows("temperature", 10), rows("acceptance", 20)
    target = temperature if role == "temperature" else acceptance
    target[-1][field] = value
    with patch("sciev.eval.fit_r2_temperature_decisions") as fitter, \
            patch("sciev.model.forward_feats") as forward:
        with pytest.raises(ValueError):
            fit_calibration(model(), None, temperature, acceptance, checkpoint_path=path)
    fitter.assert_not_called()
    forward.assert_not_called()


@pytest.mark.parametrize("role", ["temperature", "acceptance", "train"])
def test_explicit_empty_calibration_or_reference_splits_fail(tmp_path, role):
    path = checkpoint(tmp_path)
    temperature = [] if role == "temperature" else rows("temperature", 10)
    acceptance = [] if role == "acceptance" else None
    training = [] if role == "train" else None
    with patch("sciev.eval.fit_r2_temperature_decisions") as fitter:
        with pytest.raises(ValueError, match="empty|at least one|nonempty"):
            fit_calibration(model(), None, temperature, acceptance,
                            checkpoint_path=path, train_rows=training)
    fitter.assert_not_called()


def test_supplied_training_reference_cannot_overlap_dev(tmp_path):
    path = checkpoint(tmp_path)
    dev = rows("dev", 10)
    reference = rows("reference", 30)
    reference[0]["source_group"] = dev[0]["split_group"]
    with patch("sciev.eval.fit_r2_temperature_decisions") as fitter:
        with pytest.raises(ValueError, match="overlap"):
            fit_calibration(model(), None, dev, checkpoint_path=path, train_rows=reference)
    fitter.assert_not_called()


def test_acceptance_uses_canonical_original_tie_identity_and_max_probability(tmp_path):
    path = checkpoint(tmp_path)
    acceptance = rows("acceptance", 20, gold=0)
    for index, row in enumerate(acceptance):
        if index % 2:
            row.update(opts=[[2], [1]], gold=1)
    with patch("sciev.eval.fit_r2_temperature_decisions", return_value=(3.0, [])), \
            patch("sciev.model.forward_feats", side_effect=lambda *args: torch.zeros(2)):
        artifact = fit_calibration(model(), None, rows("temperature", 10), acceptance,
                                   checkpoint_path=path, min_accepted=2)
    policy = artifact["acceptance_policy"]
    assert policy["score"] == "max_probability"
    assert policy["threshold"] == 0.5
    assert policy["dev_accepted"] == 4
    assert policy["dev_errors"] == 0
    assert policy["dev_risk"] == 0.0


@pytest.mark.parametrize("fail", [False, True])
def test_calibration_restores_module_and_nested_training_modes(tmp_path, fail):
    backbone = torch.nn.Module()
    backbone.seq_len, backbone.mask_id = 64, 63
    backbone.tok_emb = torch.nn.Embedding(64, 2)
    backbone.dropout = torch.nn.Dropout()
    backbone.train()
    backbone.dropout.eval()
    head = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Linear(2, 1)).eval()
    before = [(module, module.training) for root in (backbone, head) for module in root.modules()]

    def checking_scores(*args):
        assert not torch.is_grad_enabled()
        assert all(not module.training for module, _ in before)
        if fail:
            raise RuntimeError("forward failed")
        return scores(*args)

    with patch("sciev.model.forward_feats", side_effect=checking_scores):
        if fail:
            with pytest.raises(RuntimeError, match="forward failed"):
                fit_calibration(backbone, head, rows("dev", 10), checkpoint_path=checkpoint(tmp_path))
        else:
            fit_calibration(backbone, head, rows("dev", 10), checkpoint_path=checkpoint(tmp_path))
    assert all(module.training == previous for module, previous in before)
    assert torch.is_grad_enabled()


def test_resolver_legacy_defaults_are_metadata_only():
    resolved = resolve_inference_settings({})
    assert resolved == {
        "head_kind": "mlp", "mode": "marker", "layers": [-1],
        "canonical_order": False, "strict_inputs": None, "decision_type": None,
        "encoding": None, "seq_len": None,
    }
    assert "temperature" not in resolved


def test_resolver_prefers_stored_inference_and_normalizes_layer_sequences():
    stored = {"head_kind": "attnpool", "mode": "spanpool", "layers": [-1, -3],
              "canonical_order": True, "strict_inputs": True, "decision_type": "choice",
              "encoding": ENCODING_VERSION, "seq_len": 64}
    metadata = {"head_kind": "attnpool", "n_layers": 2, "meta": {"mode": "r2_marker"},
                "inference": dict(stored, temperature=12.0)}
    assert resolve_inference_settings(metadata, layers=(-1, -3)) == stored


def test_resolver_reports_effective_mlp_layers_not_legacy_requested_layers():
    metadata = {"head_kind": "mlp", "n_layers": 4, "meta": {"mode": "r2_spanpool"}}
    resolved = resolve_inference_settings(metadata, layers=(-1, -5, -9, -13))
    assert resolved["layers"] == [-1]
    assert resolved["mode"] == "spanpool"


def test_legacy_multilayer_attention_requires_explicit_indices():
    metadata = {"head_kind": "attnpool", "n_layers": 4, "meta": {"mode": "r2_spanpool"}}
    with pytest.raises(ValueError, match="layers|indices"):
        resolve_inference_settings(metadata)
    resolved = resolve_inference_settings(metadata, layers=(-2, -4, -6, -8))
    assert resolved["layers"] == [-2, -4, -6, -8]
    with pytest.raises(ValueError, match="layer"):
        resolve_inference_settings(metadata, layers=(-1,))


@pytest.mark.parametrize("overrides", [
    {"mode": "marker"}, {"layers": [-1, -2]}, {"canonical": False},
    {"decision_type": "score"}, {"strict": False},
])
def test_resolver_rejects_conflicting_settings(overrides):
    metadata = {"head_kind": "attnpool", "n_layers": 2, "inference": {
        "mode": "spanpool", "layers": [-1, -3], "canonical_order": True,
        "decision_type": "choice", "strict_inputs": True}}
    with pytest.raises(ValueError, match="conflict|match"):
        resolve_inference_settings(metadata, **overrides)


@pytest.mark.parametrize("metadata", [
    [], {"inference": []}, {"meta": "legacy"}, {"head_kind": "unknown"},
    {"n_layers": True}, {"n_layers": 0},
    {"inference": {"canonical_order": 1}}, {"inference": {"strict_inputs": "true"}},
    {"inference": {"layers": [True]}}, {"inference": {"layers": []}},
    {"inference": {"seq_len": 0}}, {"inference": {"encoding": ""}},
    {"inference": {"decision_type": "other"}},
    {"head_kind": "attnpool", "meta": {"mode": "r2_marker"}},
])
def test_resolver_rejects_malformed_metadata(metadata):
    with pytest.raises(ValueError):
        resolve_inference_settings(metadata)


def test_fit_uses_actual_sequence_length_and_records_legacy_encoding_shift(tmp_path):
    path = checkpoint(tmp_path)
    torch.save({"head_kind": "mlp", "inference": {"encoding": "legacy_ids"}}, path)
    backbone = model()
    backbone.seq_len = 80
    with patch("sciev.eval.fit_r2_temperature_decisions", return_value=(1.0, [])):
        artifact = fit_calibration(backbone, None, rows("dev", 10), checkpoint_path=path)
    assert artifact["inference"]["seq_len"] == 80
    assert artifact["inference"]["encoding"] == ENCODING_VERSION
    assert artifact["provenance"]["encoding_check"]["status"] == "changed"


@pytest.mark.parametrize("field,value", [
    (("schema_version",), 2), (("schema_version",), True), (("kind",), "temperature"),
    (("temperature",), 0), (("temperature",), -1), (("temperature",), True),
    (("temperature",), "1.5"), (("temperature",), [1.5]),
    (("temperature",), float("nan")), (("temperature",), float("inf")),
    (("inference", "canonical_order"), 1), (("inference", "strict_inputs"), None),
    (("inference", "seq_len"), 0), (("inference", "layers"), []),
    (("inference", "layers"), [0]), (("inference", "head_kind"), "other"),
    (("inference", "encoding"), ""), (("inference", "decision_type"), "other"),
    (("checkpoint", "sha256"), "bad"), (("checkpoint", "bytes"), True),
    (("calibration_data", "temperature_dev", "identity_hashes"), []),
    (("calibration_data", "temperature_dev", "identity_hashes"), ["bad"]),
    (("calibration_data", "temperature_dev", "input_hashes"), ["a" * 64]),
    (("calibration_data", "temperature_dev", "contract_version"), True),
    (("acceptance_policy", "score"), "concentration"),
    (("acceptance_policy", "threshold"), 2.0),
])
def test_load_rejects_malformed_artifact_fields(tmp_path, fitted_artifact, field, value):
    checkpoint_path, artifact = fitted_artifact
    malformed = copy.deepcopy(artifact)
    target = malformed
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    output = tmp_path / "malformed.json"
    output.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError):
        load_calibration(output, checkpoint_path)


@pytest.mark.parametrize("missing", ["checkpoint", "temperature", "inference", "calibration_data",
                                     "acceptance_policy", "provenance", "limitations"])
def test_load_rejects_missing_artifact_fields(tmp_path, fitted_artifact, missing):
    checkpoint_path, artifact = fitted_artifact
    malformed = copy.deepcopy(artifact)
    del malformed[missing]
    output = tmp_path / "malformed.json"
    output.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError):
        load_calibration(output, checkpoint_path)


def test_load_checks_checkpoint_bytes_even_with_matching_digest(tmp_path, fitted_artifact):
    checkpoint_path, artifact = fitted_artifact
    artifact["checkpoint"]["bytes"] += 1
    output = tmp_path / "wrong_size.json"
    save_calibration(output, artifact)
    with pytest.raises(ValueError, match="checkpoint|fingerprint"):
        load_calibration(output, checkpoint_path)


def test_load_accepts_relocated_identical_checkpoint_without_refitting(tmp_path, fitted_artifact):
    checkpoint_path, artifact = fitted_artifact
    relocated = tmp_path / "relocated.pt"
    relocated.write_bytes(checkpoint_path.read_bytes())
    output = tmp_path / "calibration.json"
    save_calibration(output, artifact)
    with patch("sciev.eval.fit_r2_temperature_decisions") as fitter, \
            patch("sciev.metrics.fit_acceptance_policy") as policy_fitter:
        loaded = load_calibration(
            output, relocated, evaluation_rows=rows("test", 30, gold=0),
            expected_inference={"layers": (-1,), "canonical_order": True})
    assert loaded == artifact
    fitter.assert_not_called()
    policy_fitter.assert_not_called()


@pytest.mark.parametrize("identity", ["qid", "split_group", "source_id", "content_hash", "exact_input"])
def test_load_rejects_all_known_calibration_identity_overlaps(tmp_path, identity):
    path = checkpoint(tmp_path)
    dev = rows("dev", 10)
    dev[0].update(source_id="shared-source", content_hash="shared-content")
    evaluation = rows("test", 30)
    if identity == "exact_input":
        evaluation[0].update(ctx=dev[0]["ctx"], opts=list(reversed(dev[0]["opts"])), gold=0)
    else:
        evaluation[0][identity] = dev[0][identity]
    with patch("sciev.eval.fit_r2_temperature_decisions", return_value=(1.0, [])):
        artifact = fit_calibration(model(), None, dev, checkpoint_path=path)
    output = tmp_path / "calibration.json"
    save_calibration(output, artifact)
    with pytest.raises(ValueError, match="overlap"):
        load_calibration(output, path, evaluation_rows=evaluation)


def test_load_validates_evaluation_encoding_and_requested_settings(tmp_path, fitted_artifact):
    checkpoint_path, artifact = fitted_artifact
    output = tmp_path / "calibration.json"
    save_calibration(output, artifact)
    evaluation = [{**row, "encoding": "legacy_ids"} for row in rows("test", 30)]
    with pytest.raises(ValueError, match="encoding|inference|protocol"):
        load_calibration(output, checkpoint_path, evaluation_rows=evaluation)
    with pytest.raises(ValueError, match="inference|temperature"):
        load_calibration(output, checkpoint_path, expected_inference={"temperature": 1.0})


def test_save_validates_before_exclusive_creation(tmp_path, fitted_artifact):
    _, artifact = fitted_artifact
    artifact["temperature"] = float("nan")
    output = tmp_path / "not_created.json"
    with pytest.raises(ValueError):
        save_calibration(output, artifact)
    assert not output.exists()


def test_save_does_not_follow_existing_dangling_symlink(tmp_path, fitted_artifact):
    _, artifact = fitted_artifact
    target, output = tmp_path / "target.json", tmp_path / "link.json"
    output.symlink_to(target)
    with pytest.raises(FileExistsError):
        save_calibration(output, artifact)
    assert not target.exists()
    assert output.is_symlink()


@pytest.mark.parametrize("failure", ["existing_output", "empty_data", "overlap", "canonical_conflict",
                                     "ambiguous_layers", "invalid_budget", "invalid_config"])
def test_cli_preflight_fails_before_loading_a_model(tmp_path, failure):
    path = checkpoint(tmp_path)
    temperature_path = write_jsonl(tmp_path / "temperature.jsonl", rows("temperature", 10))
    output = tmp_path / "calibration.json"
    args = ["--ckpt", str(path), "--temperature-dev", str(temperature_path), "--out", str(output)]
    if failure == "existing_output":
        output.write_text("preserve", encoding="utf-8")
    elif failure == "empty_data":
        temperature_path.write_text("", encoding="utf-8")
    elif failure == "overlap":
        args += ["--acceptance-dev", str(temperature_path)]
    elif failure == "canonical_conflict":
        torch.save({"inference": {"canonical_order": False}}, path)
        args += ["--canonical-order"]
    elif failure == "ambiguous_layers":
        torch.save({"head_kind": "attnpool", "n_layers": 4,
                    "meta": {"mode": "r2_spanpool"}}, path)
    elif failure == "invalid_budget":
        args += ["--error-budget", "nan"]
    else:
        config = tmp_path / "bad.yaml"
        config.write_text("model: 7\n", encoding="utf-8")
        args += ["--config", str(config)]
    with patch("sciev.model.load_decision") as loader:
        with pytest.raises((ValueError, FileExistsError, SystemExit)):
            main(args)
    loader.assert_not_called()
    if failure == "existing_output":
        assert output.read_text(encoding="utf-8") == "preserve"
    else:
        assert not output.exists()


def test_cli_uses_metadata_native_config_and_records_dataset_fingerprints(tmp_path):
    path = checkpoint(tmp_path)
    torch.save({"head_kind": "mlp", "n_layers": 4, "meta": {"mode": "r2_spanpool"},
                "model_config": {"seq_len": 64}}, path)
    temperature = write_jsonl(tmp_path / "temperature.jsonl", rows("temperature", 10))
    acceptance = write_jsonl(tmp_path / "acceptance.jsonl", rows("acceptance", 20))
    reference = write_jsonl(tmp_path / "reference.jsonl", rows("reference", 30))
    config = tmp_path / "native.yaml"
    config.write_text("model:\n  seq_len: 64\n", encoding="utf-8")
    output = tmp_path / "calibration.json"
    with patch("sciev.model.load_decision", return_value=(model(), None)) as loader, \
            patch("sciev.eval.fit_r2_temperature_decisions", return_value=(1.0, [])), \
            patch("sciev.model.forward_feats", side_effect=scores):
        main(["--ckpt", str(path), "--temperature-dev", str(temperature),
              "--acceptance-dev", str(acceptance), "--train-reference", str(reference),
              "--out", str(output), "--device", "cpu", "--config", str(config),
              "--decision-type", "choice", "--no-canonical-order", "--min-accepted", "2"])
    loader.assert_called_once_with(str(path), config={"seq_len": 64}, device="cpu")
    artifact = load_calibration(output, path)
    assert artifact["inference"]["layers"] == [-1]
    assert artifact["inference"]["mode"] == "spanpool"
    assert artifact["inference"]["canonical_order"] is False
    assert artifact["inference"]["strict_inputs"] is True
    assert artifact["provenance"]["training_overlap_check"]["status"] == "unverified"
    assert artifact["provenance"]["dataset_files"]["temperature_dev"] == file_fingerprint(temperature)
    assert artifact["provenance"]["dataset_files"]["acceptance_dev"] == file_fingerprint(acceptance)
    assert artifact["provenance"]["dataset_files"]["train_reference"] == file_fingerprint(reference)
    loading = artifact["provenance"]["model_loading"]
    assert loading["legacy_defaults_possible"] is True
    assert "heads" in loading["potential_native_defaults"]
    assert "seq_len" not in loading["potential_native_defaults"]


def test_accept_none_policy_round_trips_without_missing_policy_ambiguity(tmp_path):
    path = checkpoint(tmp_path)
    with patch("sciev.eval.fit_r2_temperature_decisions", return_value=(1.0, [])), \
            patch("sciev.model.forward_feats", side_effect=scores):
        artifact = fit_calibration(model(), None, rows("temperature", 10), rows("acceptance", 20),
                                   checkpoint_path=path)
    assert artifact["acceptance_policy"] is not None
    assert artifact["acceptance_policy"]["threshold"] is None
    assert artifact["acceptance_policy"]["dev_accepted"] == 0
    assert artifact["acceptance_policy"]["dev_risk"] is None
    output = tmp_path / "accept_none.json"
    save_calibration(output, artifact)
    assert load_calibration(output, path) == artifact
