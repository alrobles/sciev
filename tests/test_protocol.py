import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from sciev import decisions
from sciev.protocol import (
    assert_checkpoint_disjoint,
    assert_disjoint_splits,
    dataset_contract,
    ensure_fresh_output,
    file_fingerprint,
    scientific_recipe,
)


def decision(offset=0, **changes):
    row = {"ctx": [101 + offset, 102 + offset],
           "opts": [[201 + offset, 202 + offset], [301 + offset]],
           "gold": 0, "kind": "choice", "option_keys": ["a", "b"],
           "encoding": "systemone-v2", "schema_version": 2}
    row.update(changes)
    return row


def permuted(row, order):
    result = deepcopy(row)
    result["opts"] = [row["opts"][index] for index in order]
    for field in ("option_keys", "soft"):
        if row.get(field) is not None:
            result[field] = [row[field][index] for index in order]
    result["gold"] = order.index(row["gold"])
    return result


@pytest.mark.parametrize("kind,head,steps,ordinal", [
    ("choice", "attnpool", 2000, 0.0),
    ("noul", "mlp", 3000, 0.0),
    ("score", "mlp", 3000, 1.0),
])
def test_scientific_recipe_is_matched_and_independent_of_controlled_axes(kind, head, steps, ordinal):
    expected = {"freeze": True, "r2_mode": "spanpool", "canonical_order": True,
                "orders": 1, "head_kind": head, "steps": steps, "head_lr": 3e-4,
                "warmup": 200, "accum": 1, "ordinal": ordinal}
    assert scientific_recipe(kind) == expected
    frozen = {**scientific_recipe(kind), "hf_backbone": "backbone-a", "seed": 1,
              "r2_layers": "-1", "lora_adapter": None, "out": "fresh-a",
              "decisions_train": "train-a.jsonl"}
    adapted = {**scientific_recipe(kind), "hf_backbone": "backbone-b", "seed": 2,
               "r2_layers": "-1,-3", "lora_adapter": "adapter", "out": "fresh-b",
               "decisions_train": "train-b.jsonl"}
    assert {key: frozen[key] for key in expected} == {key: adapted[key] for key in expected}
    changed = scientific_recipe(kind)
    changed["steps"] = 1
    assert scientific_recipe(kind) == expected


@pytest.mark.parametrize("kind", [None, "", "Choice", "boolean", "scientific-v1", 1, True, [], {}])
def test_recipe_rejects_invalid_decision_types(kind):
    with pytest.raises(ValueError, match="decision_type"):
        scientific_recipe(kind)


@pytest.mark.parametrize("payload", [b"", bytes(range(256)) * 8193])
def test_file_fingerprint_streams_exact_bytes(tmp_path, monkeypatch, payload):
    path = tmp_path / "input.jsonl"
    path.write_bytes(payload)

    def no_bulk_read(*args, **kwargs):
        raise AssertionError("fingerprinting must stream, not bulk-read a file")

    monkeypatch.setattr(Path, "read_bytes", no_bulk_read)
    monkeypatch.setattr(Path, "read_text", no_bulk_read)
    expected = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    assert file_fingerprint(path) == expected
    assert file_fingerprint(str(path)) == expected
    assert json.loads(json.dumps(expected)) == expected


def test_file_fingerprint_does_not_create_missing_input(tmp_path):
    path = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        file_fingerprint(path)
    assert not path.exists()


def test_dataset_contract_is_deterministic_json_safe_opaque_and_nonmutating():
    rows = [decision(qid="private-question", split_group="private-source", document_id=47),
            decision(10, kind="score", opts=[[3], [4], [5]], option_keys=["0", "1", "2"],
                     qid="other-private-question", split_group="other-private-source", doc_id=48)]
    snapshot = deepcopy(rows)
    contract = dataset_contract(iter(rows))
    assert rows == snapshot
    assert contract == dataset_contract(list(reversed(rows)))
    assert contract["encoding"] == "systemone-v2"
    assert contract["kinds"] == ["choice", "score"]
    assert contract["n_rows"] == 2
    assert contract["option_counts"] == [2, 3]
    assert contract["provenance_status"] == "available"
    assert contract["provenance"]["document_status"] == "available"
    for key in ("identity_hashes", "input_hashes", "effective_input_hashes"):
        assert contract[key] == sorted(set(contract[key]))
        assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in contract[key])
    serialized = json.dumps(contract, sort_keys=True, allow_nan=False)
    assert json.loads(serialized) == contract
    for private in ("private-question", "private-source", '"ctx"', '"opts"', "[101, 102]"):
        assert private not in serialized


def test_legacy_contract_does_not_claim_document_or_paraphrase_isolation():
    contract = dataset_contract([{"ctx": [], "opts": [[1], [2]], "gold": 0}])
    assert contract["encoding"] == "legacy_ids"
    assert contract["kinds"] == []
    assert contract["provenance_status"] == "not_available"
    assert contract["provenance"]["document_status"] == "not_available"
    assert contract["scope"]["document_disjointness"] == "not_proven"
    assert contract["scope"]["paraphrase_leakage"] == "not_checked"


def test_partial_provenance_and_qid_only_rows_are_reported_honestly():
    report = dataset_contract([decision(qid="identified", doc_id=1), decision(10, qid="qid-only")])
    assert report["provenance_status"] == "partial"
    assert report["provenance"]["document_status"] == "partial"
    assert report["provenance"]["rows_with_identifiers"] == 2
    assert report["provenance"]["rows_with_source_provenance"] == 1
    assert report["provenance"]["rows_with_document_ids"] == 1


def test_every_row_uses_shared_decision_validation(monkeypatch):
    validate = decisions.validate_decision_row
    seen = []

    def tracked(row, require_gold=True):
        seen.append(row)
        return validate(row, require_gold=require_gold)

    monkeypatch.setattr(decisions, "validate_decision_row", tracked)
    dataset_contract([decision(), decision(10)])
    assert len(seen) == 2


@pytest.mark.parametrize("rows", [[], (), None, {}, "not rows"])
def test_contract_rejects_empty_or_non_row_collections(rows):
    with pytest.raises(ValueError):
        dataset_contract(rows)


@pytest.mark.parametrize("patch", [
    {"gold": True}, {"gold": 2}, {"gold": -1}, {"gold": "0"},
    {"ctx": [True]}, {"ctx": [-1]}, {"opts": [[1], []]}, {"opts": [[1], [1]]},
    {"opts": [[1.5], [2]]}, {"soft": [0.5]}, {"soft": [float("nan"), 0.5]},
    {"kind": "verdict"}, {"qid": ""},
])
def test_contract_rejects_malformed_decisions(patch):
    with pytest.raises(ValueError):
        dataset_contract([decision(**patch)])


def test_contract_requires_gold():
    row = decision()
    row.pop("gold")
    with pytest.raises(ValueError, match="gold"):
        dataset_contract([row])


@pytest.mark.parametrize("encoding", [None, "", " ", 2, [], {}])
def test_contract_rejects_invalid_declared_encodings(encoding):
    with pytest.raises(ValueError, match="encoding"):
        dataset_contract([decision(encoding=encoding)])


@pytest.mark.parametrize("other", ["legacy_ids", "systemone-v3", None])
def test_contract_rejects_mixed_encodings(other):
    row = decision(10)
    if other is None:
        row.pop("encoding")
    else:
        row["encoding"] = other
    with pytest.raises(ValueError, match="encoding"):
        dataset_contract([decision(), row])


def test_input_identity_ignores_presentation_gold_and_question_id():
    row = decision(qid="original")
    other = permuted(row, [1, 0])
    other["qid"] = "renamed"
    first, second = dataset_contract([row]), dataset_contract([other])
    assert first["input_hashes"] == second["input_hashes"]
    assert first["effective_input_hashes"] == second["effective_input_hashes"]
    assert dataset_contract([row, other])["n_rows"] == 2
    other["gold"] = 0
    assert first["input_hashes"] == dataset_contract([other])["input_hashes"]
    with pytest.raises(ValueError, match="conflicting gold"):
        dataset_contract([row, other])


@pytest.mark.parametrize("kind,keys", [("noul", ["true", "false"]), ("score", ["0", "1"])])
def test_fixed_semantic_keys_survive_permutation_but_not_reassignment(kind, keys):
    row = decision(kind=kind, option_keys=keys)
    other = permuted(row, [1, 0])
    first = dataset_contract([row])
    assert first["effective_input_hashes"] == dataset_contract([other])["effective_input_hashes"]
    other["option_keys"] = keys
    other["gold"] = 0
    changed = dataset_contract([other])
    assert first["input_hashes"] == changed["input_hashes"]
    assert first["effective_input_hashes"] != changed["effective_input_hashes"]
    assert dataset_contract([row, other])["n_rows"] == 2
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_splits({"temperature_dev": [row], "test": [other]})


def test_ordinal_order_is_retained_when_semantic_keys_are_absent():
    row = decision(kind="score")
    row.pop("option_keys")
    other = permuted(row, [1, 0])
    first, second = dataset_contract([row]), dataset_contract([other])
    assert first["input_hashes"] == second["input_hashes"]
    assert first["effective_input_hashes"] != second["effective_input_hashes"]


def test_fixed_label_space_order_and_association_are_retained():
    row = decision(label_space=["a", "b"])
    other = permuted(row, [1, 0])
    first = dataset_contract([row])
    assert first["effective_input_hashes"] == dataset_contract([other])["effective_input_hashes"]
    other["label_space"] = ["b", "a"]
    assert first["effective_input_hashes"] != dataset_contract([other])["effective_input_hashes"]


@pytest.mark.parametrize("labels", ["fixed", [], ["a"], ["a", "a"], ["a", "c"], [1, 2]])
def test_fixed_label_space_is_validated(labels):
    with pytest.raises(ValueError, match="label_space"):
        dataset_contract([decision(label_space=labels)])


@pytest.mark.parametrize("first,second", [
    ({"qid": "shared"}, {"qid": "shared"}),
    ({"qid": 7}, {"qid": 7}),
    ({"sample_id": 7}, {"source_sample_id": 7}),
    ({"decision_id": "shared"}, {"reference_decision_id": "shared"}),
    ({"split_group": "shared"}, {"source_group": "shared"}),
    ({"group_id": 7}, {"group": 7}),
    ({"split_group": "shared", "source": "a"}, {"group_id": "shared", "source": "b"}),
    ({"content_hash": "shared"}, {"reference_content_hash": "shared"}),
    ({"content_id": 7}, {"content_id": 7}),
    ({"document_id": 7}, {"doc_id": 7}),
    ({"paper_id": "shared"}, {"corpus_id": "shared"}),
    ({"doi": "10.1234/example"}, {"doi": "10.1234/example"}),
    ({"pmid": 7}, {"pmid": 7}),
    ({"pmcid": "PMC7"}, {"pmcid": "PMC7"}),
    ({"source": "dataset", "pid": 7}, {"source": "dataset", "source_pid": 7}),
    ({"source": "dataset", "source_id": 7}, {"source": "dataset", "source_id": 7}),
    ({"source": "dataset", "claim_id": 7}, {"source": "dataset", "claim_id": 7}),
    ({"source_file": "input.jsonl", "source_line": 7}, {"source_file": "input.jsonl", "source_line": 7}),
])
def test_split_overlap_uses_known_identifier_aliases(first, second):
    with pytest.raises(ValueError, match="temperature_dev.*acceptance_dev.*overlap|overlap.*temperature_dev.*acceptance_dev"):
        assert_disjoint_splits({"temperature_dev": [decision(**first)],
                                "acceptance_dev": [decision(10, **second)]})


@pytest.mark.parametrize("container", ["meta", "metadata", "provenance", "source_provenance"])
def test_nested_source_metadata_detects_derived_variants(container):
    derived = decision(10, **{container: {"doc_id": 17, "source_group": "derived"}})
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_splits({"train": [decision(document_id=17)], "test": [derived]})


def test_duplicate_source_aliases_are_checked():
    derived = decision(10, duplicate_sources=[{"source_group": "original", "doc_id": 7}])
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_splits({"train": [decision(split_group="original")], "test": [derived]})


@pytest.mark.parametrize("metadata", [
    {"negative_provenance": {"source_group": "donor", "strategy": "cross_passage"}},
    {"negative_provenance": [{"source_group": "own"}, {"source_group": "donor"}]},
    {"negative_sources": [{"source_group": "donor"}]},
    {"negative_sources": ["donor"]},
    {"negative_source_group": "donor"},
    {"negative_source_groups": ["donor"]},
])
def test_negative_source_groups_overlap_their_original_source(metadata):
    derived = decision(10, split_group="derived", **metadata)
    contract = dataset_contract([derived])
    assert contract["provenance"]["rows_with_negative_provenance"] == 1
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_splits({"temperature_dev": [decision(split_group="donor")], "test": [derived]})


@pytest.mark.parametrize("field,target", [("source_sample_id", "sample_id"), ("source_pid", "pid"),
                                          ("document_id", "doc_id"), ("content_hash", "content_hash")])
def test_negative_source_identifiers_use_the_original_namespace(field, target):
    original = decision(source="dataset", **{target: 19})
    derived = decision(10, source="different-dataset", negative_provenance={"source": "dataset", field: 19})
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_splits({"train": [original], "test": [derived]})


def test_shared_negative_source_is_not_mistaken_for_disjoint_primary_groups():
    first = decision(split_group="a", negative_provenance={"source_group": "shared-donor"})
    second = decision(10, split_group="b", negative_provenance={"source_group": "shared-donor"})
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_splits({"temperature_dev": [first], "acceptance_dev": [second]})


def test_evidence_control_donor_group_is_checked():
    derived = decision(10, evidence_control={"source_group": "own", "donor_group": "donor"})
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_splits({"train": [decision(split_group="donor")], "test": [derived]})


def test_identical_input_is_caught_despite_different_ids_labels_and_order():
    first = decision(qid="a", sample_id="a", split_group="a")
    second = permuted(first, [1, 0])
    second.update(qid="b", sample_id="b", split_group="b", gold=0)
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_splits({"temperature_dev": [first], "acceptance_dev": [second]})


def test_identifier_namespaces_do_not_conflate_unrelated_identifiers():
    rows = {"temperature_dev": [decision(qid="same", group_id=7)],
            "acceptance_dev": [decision(10, group_id="same", doc_id=7)],
            "test": [decision(20, source="dataset", pid=7)]}
    report = assert_disjoint_splits(rows)
    assert report["status"] == "verified"
    assert report["pairs_checked"] == 3
    assert report["overlap_count"] == 0
    assert set(report["splits"]) == set(rows)
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_local_source_ids_are_namespaced_and_source_names_are_not_identities():
    rows = {"temperature_dev": [decision(source="a", pid=7, source_id=1, claim_id=2)],
            "acceptance_dev": [decision(10, source="b", pid=7, source_id=1, claim_id=2)],
            "test": [decision(20, source="a", pid=8, source_id=3, claim_id=4)]}
    assert assert_disjoint_splits(rows)["status"] == "verified"


@pytest.mark.parametrize("first,second", [("1e-3", "1e3"), ("CO", "Co"), ("a  b", "a b"), ("µ", "μ")])
def test_scientific_content_and_identifier_values_are_not_lossily_normalized(first, second):
    rows = {"temperature_dev": [decision(ctx=list(map(ord, first)), content_id=first)],
            "test": [decision(ctx=list(map(ord, second)), content_id=second)]}
    assert assert_disjoint_splits(rows)["status"] == "verified"


@pytest.mark.parametrize("field", ["qid", "sample_id", "source_group", "content_hash", "doc_id", "pid"])
@pytest.mark.parametrize("value", [True, 1.5, [], {}])
def test_known_identifiers_reject_invalid_types(field, value):
    with pytest.raises(ValueError):
        dataset_contract([decision(**{field: value})])


@pytest.mark.parametrize("named_rows", [{}, [], {"": [decision()]}, {1: [decision()]}, {"test": []}])
def test_split_checker_rejects_invalid_roles_and_empty_splits(named_rows):
    with pytest.raises(ValueError):
        assert_disjoint_splits(named_rows)


@pytest.mark.parametrize("role", ["evaluation", "temperature_dev", "acceptance_dev"])
def test_checkpoint_overlap_rejects_relabelled_permuted_evaluation(role):
    original = decision(qid="train")
    other = permuted(original, [1, 0])
    other.update(qid="eval", gold=0)
    checkpoint = {"training_data": dataset_contract([original])}
    with pytest.raises(ValueError, match=f"{role}.*overlap|overlap.*{role}"):
        assert_checkpoint_disjoint(checkpoint, [other], role=role)


def test_checkpoint_overlap_includes_negative_source_provenance():
    checkpoint = {"training_data": dataset_contract([decision(split_group="training-source")])}
    rows = [decision(10, negative_provenance={"source_group": "training-source"})]
    with pytest.raises(ValueError, match="overlap"):
        assert_checkpoint_disjoint(checkpoint, rows)


def test_disjoint_checkpoint_report_is_json_safe_and_nonmutating():
    checkpoint = {"training_data": dataset_contract([decision(document_id="train")])}
    snapshot = deepcopy(checkpoint)
    report = assert_checkpoint_disjoint(checkpoint, [decision(10, doc_id="test")], role="test")
    assert report["status"] == "verified"
    assert report["training_identity_status"] == "available"
    assert report["role"] == "test"
    assert report["overlap_count"] == 0
    assert report["data"]["n_rows"] == 1
    assert checkpoint == snapshot
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize("checkpoint", [{}, {"training_data": None}, {"training_data": {"n_rows": 5}}])
def test_legacy_checkpoint_is_explicitly_unverified(checkpoint):
    report = assert_checkpoint_disjoint(checkpoint, [decision()])
    assert report["status"] == "unverified"
    assert report["training_identity_status"] == "not_available"
    assert report["reason"] == "training_identities_not_available"
    assert report["overlap_count"] is None
    assert report["role"] == "evaluation"
    assert report["scope"]["document_disjointness"] == "not_proven"


@pytest.mark.parametrize("hashes", [None, [], "not-a-list", ["not-a-sha256"], [1]])
def test_malformed_saved_identity_hashes_do_not_produce_a_false_pass(hashes):
    with pytest.raises(ValueError, match="identity_hashes"):
        assert_checkpoint_disjoint({"training_data": {"identity_hashes": hashes}}, [decision()])


def test_legacy_checkpoint_does_not_bypass_dataset_validation():
    with pytest.raises(ValueError):
        assert_checkpoint_disjoint({}, [])


def test_fresh_output_check_never_creates_a_directory(tmp_path):
    path = tmp_path / "missing" / "run"
    assert ensure_fresh_output(path) is None
    assert not path.parent.exists()


def test_fresh_output_allows_empty_or_unrelated_files_without_mutation(tmp_path):
    path = tmp_path / "run"
    path.mkdir()
    assert ensure_fresh_output(path) is None
    log = path / "launcher.log"
    log.write_text("existing log\n")
    assert ensure_fresh_output(path) is None
    assert list(path.iterdir()) == [log]
    assert log.read_text() == "existing log\n"


@pytest.mark.parametrize("artifact", ["decision.pt", "model.pt", "training_config.json", "temperature.json"])
def test_fresh_output_rejects_every_reserved_artifact_without_overwriting(tmp_path, artifact):
    path = tmp_path / "run"
    path.mkdir()
    saved = path / artifact
    saved.write_bytes(b"historical artifact")
    with pytest.raises(FileExistsError, match="overwrite|existing|already"):
        ensure_fresh_output(path)
    assert saved.read_bytes() == b"historical artifact"
    assert list(path.iterdir()) == [saved]


def test_fresh_output_rejects_non_directory_paths(tmp_path):
    path = tmp_path / "file"
    path.write_text("original")
    with pytest.raises(FileExistsError):
        ensure_fresh_output(path)
    assert path.read_text() == "original"


@pytest.mark.parametrize("at_root", [False, True])
def test_fresh_output_rejects_dangling_artifact_symlinks(tmp_path, at_root):
    path = tmp_path / "run"
    if at_root:
        path.symlink_to(tmp_path / "nonexistent")
    else:
        path.mkdir()
        (path / "decision.pt").symlink_to(tmp_path / "nonexistent")
    with pytest.raises(FileExistsError):
        ensure_fresh_output(path)


def test_fresh_output_rejects_reserved_directories(tmp_path):
    path = tmp_path / "run"
    (path / "decision.pt").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        ensure_fresh_output(path)
    assert (path / "decision.pt").is_dir()
