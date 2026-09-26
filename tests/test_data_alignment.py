import json

import pytest

from reverse_jev.data import load_decisions_ids, split_dev_test


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_ids_loader_preserves_provenance(tmp_path):
    row = {"ctx": [1, 2], "opts": [[3], [4]], "gold": 1,
           "qid": "q", "soft": [0.25, 0.75], "pid": "p",
           "group_id": "document:d", "source": "synthetic",
           "reasoning_type": "numerical", "kind": "noul",
           "encoding": "systemone-v2", "schema_version": 2,
           "negative_provenance": {"strategy": "number_perturb",
                                   "verified": False}}
    assert load_decisions_ids(write_jsonl(tmp_path / "rows.jsonl", [row])) == [row]


@pytest.mark.parametrize("patch", [
    {"gold": "1"}, {"gold": 0.5}, {"gold": True}, {"gold": -1},
    {"gold": 2}, {"ctx": None}, {"ctx": [True]}, {"ctx": [-1]},
    {"opts": [[2], []]}, {"opts": [[2], [2.5]]},
    {"soft": [0.5]}, {"soft": [0.1, 0.1]}, {"soft": [float("nan"), 0.5]},
])
def test_ids_loader_rejects_malformed_rows_with_location(tmp_path, patch):
    row = {"ctx": [1], "opts": [[2], [3]], "gold": 0}
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(row) + "\n\n" + json.dumps(dict(row, **patch)) + "\n")
    with pytest.raises(ValueError, match=r"broken\.jsonl:3:"):
        load_decisions_ids(path)


@pytest.mark.parametrize("line", ["{broken", "[]", '{"ctx": [1]}'])
def test_ids_loader_reports_invalid_json_and_shape(tmp_path, line):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n" + line + "\n")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2:"):
        load_decisions_ids(path)


def request(state, prefix, **metadata):
    return dict(metadata, state=state, questions={
        f"{prefix}_{i}": {"type": "noul", "instructions": "Supported?",
                          "label": bool(i % 2)} for i in range(6)})


@pytest.mark.parametrize("field", ["group_id", "source_group", "pid"])
def test_dev_test_split_keeps_source_groups_together(tmp_path, field):
    rows = [request(f"state {i}", str(i), **{field: str(i // 2)})
            for i in range(8)]
    path = write_jsonl(tmp_path / "requests.jsonl", rows)
    dev, test = split_dev_test(path, dev_frac=0.5, seed=4)
    owners = {qid: row[field] for row in rows for qid in row["questions"]}
    assert dev and test
    assert {owners[r[1]] for r in dev}.isdisjoint({owners[r[1]] for r in test})
    assert len(dev) + len(test) == 48
    assert (dev, test) == split_dev_test(path, dev_frac=0.5, seed=4)


def test_dev_test_split_groups_questions_and_structured_state(tmp_path):
    rows = [request({"b": 2, "a": 1}, "a"), request({"a": 1, "b": 2}, "b"),
            request("different state", "c"), request("last state", "d")]
    dev, test = split_dev_test(write_jsonl(tmp_path / "states.jsonl", rows), seed=7)
    dev_states = {r[0] for r in dev}
    test_states = {r[0] for r in test}
    assert dev_states.isdisjoint(test_states)
    for row in rows:
        qids = set(row["questions"])
        assert qids <= {r[1] for r in dev} or qids <= {r[1] for r in test}


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf")])
def test_dev_test_split_validates_fraction(tmp_path, fraction):
    path = write_jsonl(tmp_path / "requests.jsonl", [request("state", "q")])
    with pytest.raises(ValueError, match="frac"):
        split_dev_test(path, dev_frac=fraction)


def test_empty_context_control_is_preserved_not_silently_dropped(tmp_path):
    row = {"ctx": [], "opts": [[1], [2]], "gold": 0, "control": "no_evidence"}
    loaded = load_decisions_ids(write_jsonl(tmp_path / "controls.jsonl", [row]))
    assert len(loaded) == 1 and loaded[0]["control"] == "no_evidence"
    assert loaded[0]["ctx"] == []


def test_structured_state_roundtrips_through_text_loader(tmp_path):
    from reverse_jev.data import iter_decisions
    from reverse_jev.decisions import encode_question

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return list(text.encode("utf-8"))

    state = {"z": ["evidence", {"d": 2, "a": 1}], "a": "claim"}
    q = {"type": "noul", "instructions": {"task": "Check claim"}, "label": True}
    path = write_jsonl(tmp_path / "structured.jsonl", [{"state": state, "questions": {"q": q}}])
    loaded_state, _, question, _ = next(iter_decisions(path))
    assert encode_question(Tok(), loaded_state, question) == encode_question(Tok(), state, q)


def evidence_request(group, evidence, prefix=None, split="eval"):
    prefix = prefix or group
    before = "Préface\nEvidence: "
    after = f"\nQuestion: What follows for {prefix}?\nProposed answer: original answer."
    span = [len(before), len(before) + len(evidence)]
    return {"state": before + evidence + after, "evidence_span": span,
            "split_group": group, "group_id": f"alias-{group}", "split": split,
            "source": "synthetic", "encoding": "systemone-v2", "schema_version": 2,
            "decision_id": f"request-fingerprint-{prefix}", "content_hash": f"evidence-hash-{prefix}",
            "label_status": "heuristic", "questions": {
                f"verify-{prefix}": {"type": "noul", "kind": "noul", "label": True,
                                     "instructions": {"task": "Check the answer", "focus": ["question", "passage"]},
                                     "criteria": {"true": {"meaning": "correct"}, "false": "incorrect"},
                                     "evidence_span": list(span), "label_status": "heuristic",
                                     "decision_id": f"question-fingerprint-{prefix}"},
                f"rate-{prefix}": {"type": "score", "kind": "score", "label": 1,
                                   "instructions": "Rate the proposed answer.", "criteria": ["wrong", "correct"],
                                   "evidence_span": list(span), "label_status": "heuristic_proxy"}}}


def assert_control_reference(source, controlled, mode, source_index):
    marker = controlled["evidence_control"]
    assert marker["mode"] == mode
    assert marker["source_group"] == source.get("split_group", source.get("group_id"))
    assert marker["source_record_index"] == source_index
    assert marker["label_semantics"] == "original_reference_not_relabelled"
    assert controlled["label_status"] == "evidence_control"
    assert controlled["reference_label_status"] == source["label_status"]
    assert "decision_id" not in controlled and "content_hash" not in controlled
    assert controlled["reference_decision_id"] == source["decision_id"]
    assert controlled["reference_content_hash"] == source["content_hash"]
    assert set(controlled["questions"]).isdisjoint(source["questions"])
    assert {q["reference_qid"] for q in controlled["questions"].values()} == set(source["questions"])
    for qid, question in controlled["questions"].items():
        original = source["questions"][question["reference_qid"]]
        assert question["qid"] == qid
        assert "label" not in question and question["reference_label"] == original["label"]
        assert question["label_status"] == "evidence_control"
        assert question["reference_label_status"] == original["label_status"]
        assert question["evidence_span"] == controlled["evidence_span"]
        assert question["evidence_control"] == marker
        assert "decision_id" not in question
        if "decision_id" in original:
            assert question["reference_decision_id"] == original["decision_id"]
        for key in ("type", "kind", "instructions", "criteria"):
            assert question[key] == original[key]


def test_empty_evidence_control_replaces_only_explicit_span_without_mutation():
    from copy import deepcopy
    import random
    from reverse_jev.data import make_evidence_controls

    records = [evidence_request("a", "α evidence. Question: a decoy inside evidence."),
               evidence_request("b", "Another passage with a different length.")]
    snapshot = deepcopy(records)
    rng_state = random.getstate()
    controls = make_evidence_controls(records)
    assert records == snapshot and random.getstate() == rng_state
    assert controls == make_evidence_controls(records, mode="empty", seed=7331)
    for i, (source, controlled) in enumerate(zip(records, controls)):
        start, end = source["evidence_span"]
        assert controlled["state"] == source["state"][:start] + source["state"][end:]
        assert controlled["evidence_span"] == [start, start]
        assert "Question: What follows" in controlled["state"]
        assert "Proposed answer: original answer." in controlled["state"]
        assert controlled["evidence_control"]["donor_group"] is None
        assert controlled["evidence_control"]["donor_record_index"] is None
        assert_control_reference(source, controlled, "empty", i)
    question = controls[0]["questions"][next(iter(controls[0]["questions"]))]
    question["instructions"]["focus"].append("changed only in output")
    question["criteria"]["true"]["meaning"] = "changed only in output"
    assert records == snapshot


def test_shuffle_evidence_controls_use_cross_group_donors_reproducibly():
    from copy import deepcopy
    import random
    from reverse_jev.data import make_evidence_controls

    records = [evidence_request("a", "α first evidence"),
               evidence_request("a", "α first evidence", prefix="another-question"),
               evidence_request("b", "A longer second passage."),
               evidence_request("c", "Third evidence.")]
    snapshot = deepcopy(records)
    rng_state = random.getstate()
    controls = make_evidence_controls(records, mode="shuffle", seed=7)
    assert controls == make_evidence_controls(records, mode="shuffle", seed=7)
    assert records == snapshot and random.getstate() == rng_state
    for i, (source, controlled) in enumerate(zip(records, controls)):
        marker = controlled["evidence_control"]
        donor = records[marker["donor_record_index"]]
        assert marker["seed"] == 7
        assert marker["donor_group"] == donor["split_group"] != source["split_group"]
        start, end = source["evidence_span"]
        donor_start, donor_end = donor["evidence_span"]
        replacement = donor["state"][donor_start:donor_end]
        assert controlled["state"] == source["state"][:start] + replacement + source["state"][end:]
        assert controlled["state"] != source["state"]
        assert controlled["evidence_span"] == [start, start + len(replacement)]
        assert_control_reference(source, controlled, "shuffle", i)
    empty_qids = {qid for rec in make_evidence_controls(records) for qid in rec["questions"]}
    assert empty_qids.isdisjoint(qid for rec in controls for qid in rec["questions"])


@pytest.mark.parametrize("span", [None, [], [0], [0, 1, 2], "0:5", [-1, 3], [5, 1],
                                   [0, 0], [0, 100000], [True, 5], [1.0, 5], ["1", 5]])
def test_evidence_controls_reject_invalid_explicit_offsets(span):
    from reverse_jev.data import make_evidence_controls

    row = evidence_request("a", "source evidence")
    row["evidence_span"] = span
    with pytest.raises(ValueError, match="evidence_span"):
        make_evidence_controls([row])


@pytest.mark.parametrize("field", ["evidence_span", "group"])
def test_evidence_controls_do_not_guess_missing_provenance(field):
    from reverse_jev.data import make_evidence_controls

    row = evidence_request("a", "source evidence")
    if field == "evidence_span":
        row.pop("evidence_span")
    else:
        row.pop("split_group")
        row.pop("group_id")
        row["pid"] = "pid-is-not-an-explicit-control-group"
    with pytest.raises(ValueError, match="evidence_span|group"):
        make_evidence_controls([row])


@pytest.mark.parametrize("group", [None, "", "  ", True, [], {}])
def test_evidence_controls_reject_invalid_groups(group):
    from reverse_jev.data import make_evidence_controls

    row = evidence_request("a", "source evidence")
    row["split_group"] = group
    with pytest.raises(ValueError, match="group"):
        make_evidence_controls([row])


def test_evidence_controls_accept_explicit_group_id_without_split_group():
    from reverse_jev.data import make_evidence_controls

    rows = [evidence_request("a", "first evidence"), evidence_request("b", "second evidence")]
    for i, row in enumerate(rows):
        row.pop("split_group")
        row["group_id"] = i
    controls = make_evidence_controls(rows, mode="shuffle")
    assert controls[0]["evidence_control"]["source_group"] == 0
    assert controls[0]["evidence_control"]["donor_group"] == 1


@pytest.mark.parametrize("records", [[], [evidence_request("a", "source evidence")],
    [evidence_request("a", "first evidence"), evidence_request("a", "different evidence", prefix="second")],
    [evidence_request("a", "identical evidence"), evidence_request("b", "identical evidence")]])
def test_shuffle_evidence_controls_fail_without_a_changed_cross_group_donor(records):
    from reverse_jev.data import make_evidence_controls

    with pytest.raises(ValueError, match="(?i)(empty|donor|records)"):
        make_evidence_controls(records, mode="shuffle")


def test_shuffle_evidence_controls_do_not_cross_source_splits():
    from reverse_jev.data import make_evidence_controls

    rows = [evidence_request("a", "train evidence", split="train"),
            evidence_request("b", "heldout evidence", split="eval")]
    with pytest.raises(ValueError, match="donor"):
        make_evidence_controls(rows, mode="shuffle")


@pytest.mark.parametrize("mode", ["delete_question", "random", None])
def test_evidence_controls_reject_unknown_modes(mode):
    from reverse_jev.data import make_evidence_controls

    with pytest.raises(ValueError, match="mode"):
        make_evidence_controls([evidence_request("a", "evidence")], mode=mode)


def test_evidence_controls_require_string_states_and_matching_question_spans():
    from copy import deepcopy
    from reverse_jev.data import make_evidence_controls

    row = evidence_request("a", "evidence")
    bad_state = dict(row, state={"passage": "evidence", "question": "Question"})
    with pytest.raises(ValueError, match="state.*string"):
        make_evidence_controls([bad_state])
    bad_span = deepcopy(row)
    bad_span["questions"]["verify-a"]["evidence_span"][0] += 1
    with pytest.raises(ValueError, match="evidence_span"):
        make_evidence_controls([bad_span])
    with pytest.raises(ValueError, match="questions"):
        make_evidence_controls([dict(row, questions={})])


def test_evidence_controls_do_not_invent_labels_or_reuse_encoded_targets():
    from reverse_jev.data import make_evidence_controls

    row = evidence_request("a", "evidence")
    labelled = row["questions"]["verify-a"]
    labelled.update(gold=0, soft=[0.8, 0.2])
    row["questions"]["rate-a"].pop("label")
    control = make_evidence_controls([row])[0]
    questions = {q["reference_qid"]: q for q in control["questions"].values()}
    assert questions["verify-a"]["reference_gold"] == 0
    assert questions["verify-a"]["reference_soft"] == [0.8, 0.2]
    assert "gold" not in questions["verify-a"] and "soft" not in questions["verify-a"]
    assert "reference_label" not in questions["rate-a"] and "label" not in questions["rate-a"]


@pytest.mark.parametrize("left,right", [
    ("The solution is 1 mM.", "The solution is 1 mm."),
    ("Variable X equals 5.", "Variable x equals 5."),
    ("Coordinate x₂ equals 5.", "Coordinate x2 equals 5."),
    ("The area is 4 m².", "The area is 4 m2."),
])
def test_scientific_identity_preserves_case_and_compatibility_symbols(left, right):
    from reverse_jev.data import content_fingerprint, group_records, normalized_content

    assert normalized_content(left) == left
    assert normalized_content(right) == right
    assert content_fingerprint(left) != content_fingerprint(right)
    rows = group_records([{"pid": "left", "passage": left},
                          {"pid": "right", "passage": right}], "passage")
    assert rows[0]["content_hash"] != rows[1]["content_hash"]
    assert rows[0]["split_group"] != rows[1]["split_group"]


def test_scientific_identity_groups_canonical_unicode_and_whitespace_duplicates():
    from reverse_jev.data import content_fingerprint, group_records, normalized_content

    original = "Café evidence uses X and 1 mM."
    equivalent = "\tCafe\u0301  evidence\nuses X  and 1 mM.\r\n"
    assert normalized_content(equivalent) == normalized_content(original) == original
    assert content_fingerprint(original) == content_fingerprint(equivalent)
    rows = group_records([{"pid": "original", "passage": original},
                          {"pid": "equivalent", "passage": equivalent}], "passage")
    assert rows[0]["content_hash"] == rows[1]["content_hash"]
    assert rows[0]["split_group"] == rows[1]["split_group"]
