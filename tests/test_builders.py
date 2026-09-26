import hashlib
import json
import random

import pandas as pd
import pytest

from data import build_sci_decisions as sci
from data import build_toolcall_decisions as tools
from data import convert_benchmarks as benchmarks
from data.teacher_label import render_prompt


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]


def rows_from(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def qa(pid, answer, passage=None, **metadata):
    return dict(metadata, pid=pid, passage=passage or f"Evidence {pid}: {answer}",
                q=f"What is the result for {pid}?", a=answer, type="numerical")


def scientific_rows():
    return [qa(str(i), f"The result is {10 * (i + 1)} units.") for i in range(6)]


def assert_paired(ids, texts):
    from reverse_jev.decisions import encode_question
    questions = {qid: (row, question) for row in texts
                 for qid, question in row["questions"].items()}
    assert len(questions) == len(ids)
    for row in ids:
        text, question = questions[row["qid"]]
        encoded = encode_question(CharTokenizer(), text["state"], question)
        for key in ("ctx", "opts", "kind", "option_keys", "encoding", "schema_version"):
            assert row[key] == encoded[key]
        for key in ("source", "pid", "group_id", "group_scope", "reasoning_type"):
            assert row[key] == question[key] == text[key]
        assert question["kind"] == row["kind"]
        assert question["encoding"] == text["encoding"] == "systemone-v2"


@pytest.mark.parametrize("left,right,equal", [
    ("10", "100", False), ("10", "1", False), ("1.0", "1", True),
    ("-10", "10", False), ("1e2", "100", True),
    ("1,000.00", "1000", True), ("0.010", "0.01", True),
])
def test_numeric_normalization_preserves_values(left, right, equal):
    assert (sci._norm_num(left) == sci._norm_num(right)) is equal


@pytest.mark.parametrize("answer,passage,expected", [
    ("The value is 10 units.", "The value is 100 units.", False),
    ("The value is -10 units.", "The value is 10 units.", False),
    ("The value is 1e2 units.", "The value is 100 units.", True),
    ("The value is 1.0 units.", "The value is 1 unit.", True),
])
def test_grounded_numeric_gate_is_value_preserving(answer, passage, expected):
    assert sci.grounded("What is the value?", answer, passage) is expected


def test_elite_and_raw_have_explicit_same_filter_policy(tmp_path):
    rec = qa("p", "The value is 10 units.", passage="The value is 100 units.")
    elite = tmp_path / "elite.jsonl"
    elite.write_text(json.dumps(rec) + "\n")
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"pid": "p", "document_id": "d",
                               "passage": rec["passage"],
                               "qa": [{"q": rec["q"], "a": rec["a"],
                                       "type": "numerical"}]}) + "\n")
    assert sci.load_rows(elite) == []
    assert sci.load_qa_raw(str(raw), random.Random(3)) == []
    kept = sci.load_rows(elite, filter_policy="schema")
    raw_kept = sci.load_qa_raw(str(raw), random.Random(3), filter_policy="schema")
    assert kept[0]["filter_policy"] == raw_kept[0]["filter_policy"] == "schema"
    assert raw_kept[0]["document_id"] == "d"
    assert raw_kept[0]["type"] == "numerical"


def test_scientific_split_groups_whitespace_duplicate_passages_and_documents():
    rows = scientific_rows()
    rows[1]["passage"] = "  " + rows[0]["passage"].replace(" ", " \t ") + "\n"
    rows[2]["document_id"] = rows[3]["document_id"] = "document-2"
    splits = sci.split_records(rows, random.Random(3), dev_frac=0.25, eval_frac=0.25)
    owners = {r["pid"]: tag for tag, group in splits.items() for r in group}
    assert owners["0"] == owners["1"]
    assert owners["2"] == owners["3"]
    groups = {tag: {r["group_id"] for r in group} for tag, group in splits.items()}
    assert groups["train"].isdisjoint(groups["dev"] | groups["eval"])
    assert groups["dev"].isdisjoint(groups["eval"])
    for group in splits.values():
        for row in group:
            assert row["group_scope"] == ("document" if row["pid"] in {"2", "3"}
                                            else "passage")


@pytest.mark.parametrize("kwargs", [
    {"dev_frac": -0.1}, {"eval_frac": float("nan")},
    {"dev_frac": 0.6, "eval_frac": 0.6},
    {"counts": {"train": -1, "dev": 1, "eval": 1}},
    {"counts": {"train": 10, "dev": 1, "eval": 1}},
])
def test_scientific_split_rejects_invalid_sizes(kwargs):
    with pytest.raises(ValueError):
        sci.split_records(scientific_rows(), random.Random(0), **kwargs)


def test_scientific_build_is_bounded_local_and_paired():
    records = scientific_rows()
    ids, texts = sci.build(records, CharTokenizer(), random.Random(11), "train")
    assert_paired(ids, texts)
    assert (ids, texts) == sci.build(records, CharTokenizer(), random.Random(11), "train")
    assert {row["kind"] for row in ids} == {"choice", "noul", "score"}
    local_pids = {r["pid"] for r in records}
    for row in ids:
        provenance = row["negative_provenance"]
        for entry in provenance if isinstance(provenance, list) else [provenance]:
            if "source_pid" in entry:
                assert entry["source_pid"] in local_pids
            assert entry["verified"] is False
    exclusions = []
    small_ids, _ = sci.build([qa("only", "A stable qualitative answer.")],
                            CharTokenizer(), random.Random(1), "dev",
                            exclusions=exclusions)
    assert not any(r["kind"] == "choice" for r in small_ids)
    assert "insufficient_unique_distractors" in {e["reason"] for e in exclusions}
    assert not any(r["kind"] == "score" and r["gold"] == 1 for r in small_ids)


def test_cross_passage_fallback_is_not_labelled_related():
    records = [qa(str(i), f"Qualitative answer {letter}.")
               for i, letter in enumerate("abcd")]
    ids, _ = sci.build(records, CharTokenizer(), random.Random(9), "train")
    assert not any(r["kind"] == "score" and r["gold"] == 1 for r in ids)


def test_builders_reject_overflow_instead_of_mismatching_remote_text():
    records = scientific_rows()
    records[0]["passage"] = "long evidence " * 100
    with pytest.raises(ValueError, match="(?i)(budget|exceed|long|overflow|token)"):
        sci.build(records, CharTokenizer(), random.Random(0), "train")


def tool_record(pid, value, prompt=None):
    return {"pid": pid, "prompt": prompt or f"Find observations for {pid}.",
            "gold": [{"tool": "gbif_occurrence", "args": {"taxon": value}}]}


def test_toolcall_pools_are_local_typed_and_paired(monkeypatch):
    records = [tool_record("a", 10), tool_record("b", 20)]
    monkeypatch.setattr(tools, "arg_pool_cache", {"taxon": ["HELDOUT_ONLY"]}, raising=False)
    ids, texts = tools.build(records, CharTokenizer(), random.Random(1), "train")
    assert "HELDOUT_ONLY" not in json.dumps(texts)
    assert_paired(ids, texts)
    assert tools.arg_pool(records) == {"taxon": [10, 20]}
    changed = tools.corrupt_args({"taxon": 10}, tools.arg_pool(records), random.Random(2))
    assert changed == {"taxon": 20}
    exclusions = []
    solo, _ = tools.build(records[:1], CharTokenizer(), random.Random(2), "dev",
                          exclusions=exclusions)
    assert not any(r["kind"] == "score" and r["gold"] == 1 for r in solo)
    assert "insufficient_argument_pool" in {e["reason"] for e in exclusions}


def test_toolcall_split_excludes_whitespace_eval_content_overlap():
    train = [tool_record("a", "a", "Shared request"),
             tool_record("b", "b", "unique request b"),
             tool_record("c", "c", "unique request c")]
    heldout = [tool_record("heldout", "z", "  Shared   request ")]
    exclusions = []
    splits = tools.split_records(train, heldout, random.Random(3), dev_frac=0.5,
                                 exclusions=exclusions)
    assert "a" not in {r["pid"] for tag in ("train", "dev") for r in splits[tag]}
    assert "eval_group_overlap" in {e["reason"] for e in exclusions}
    assert {r["group_id"] for r in splits["eval"]}.isdisjoint(
        {r["group_id"] for tag in ("train", "dev") for r in splits[tag]})


def scifact_source(tmp_path, verdicts=("SUPPORT", "CONTRADICT", "NEI")):
    src = tmp_path / "validation.parquet"
    pd.DataFrame([{"claim": f"Claim {i}", "title": "Evidence", "document_id": "doc",
                   "abstract": ["The observations give evidence."], "verdict": verdict}
                  for i, verdict in enumerate(verdicts)]).to_parquet(src)
    return src


def test_scifact_primary_tasks_are_native_and_versioned(tmp_path):
    src = scifact_source(tmp_path)
    benchmarks.conv_scifact(str(src), str(tmp_path / "sf"), CharTokenizer(), random.Random(1))
    output = tmp_path / "systemone-v2"
    noul = rows_from(output / "sf_noul_eval.jsonl")
    choice = rows_from(output / "sf_choice_eval.jsonl")
    assert [r["gold"] for r in noul] == [0, 1, 1]
    assert [r["gold"] for r in choice] == [0, 1, 2]
    assert all(r["option_keys"] == ["SUPPORT", "CONTRADICT", "NEI"] for r in choice)
    assert not list(output.glob("*score*"))
    for kind, rows in (("noul", noul), ("choice", choice)):
        assert_paired(rows, rows_from(output / f"sf_{kind}_eval_text.jsonl"))
    manifest = json.loads((output / "sf_manifest.json").read_text())
    assert manifest["encoding"] == "systemone-v2"
    assert manifest["inputs"][0]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert manifest["counts"]["input_records"] == 3
    assert manifest["exclusions"] == []
    with pytest.raises(FileExistsError):
        benchmarks.conv_scifact(str(src), str(tmp_path / "sf"), CharTokenizer(), random.Random(1))


def test_scifact_legacy_score_requires_opt_in(tmp_path):
    src = scifact_source(tmp_path)
    benchmarks.conv_scifact(str(src), str(tmp_path / "sf"), CharTokenizer(), random.Random(1),
                             legacy_score=True)
    output = tmp_path / "systemone-v2"
    rows = rows_from(output / "sf_score_legacy_proxy_eval.jsonl")
    assert [r["gold"] for r in rows] == [2, 1, 0]
    assert all(r["legacy_proxy"] is True and r["label_status"] == "legacy_proxy" for r in rows)
    assert_paired(rows, rows_from(output / "sf_score_legacy_proxy_eval_text.jsonl"))


def test_scifact_unknown_verdict_fails_clearly_without_outputs(tmp_path):
    src = scifact_source(tmp_path, ("MAYBE",))
    with pytest.raises(ValueError, match="(?i)verdict.*MAYBE"):
        benchmarks.conv_scifact(str(src), str(tmp_path / "sf"), CharTokenizer(), random.Random(1))
    assert not (tmp_path / "systemone-v2").exists()


def test_versioned_writer_does_not_overwrite_release(tmp_path):
    release = tmp_path / "existing.jsonl"
    release.write_text("immutable release\n")
    benchmarks.write_rows(release, [], [])
    assert release.read_text() == "immutable release\n"
    assert (tmp_path / "systemone-v2" / "existing.jsonl").exists()
    with pytest.raises(FileExistsError):
        benchmarks.write_rows(release, [], [])


def test_teacher_noul_uses_scientific_instruction_and_criteria():
    body, opts, gold = render_prompt({"passage": "Scientific evidence"}, {
        "type": "noul", "instructions": "Is the claim supported?",
        "criteria": {"true": "Evidence supports the claim", "false": "Not supported"},
        "label": False})
    assert "Evidence supports the claim" in body and "Not supported" in body
    assert "tool" not in body and "argument" not in body
    assert gold == 1 and len(opts) == 2
    assert '"passage":' in body


def test_teacher_choice_includes_descriptions():
    body, _, gold = render_prompt("Evidence", {
        "type": "choice", "instructions": "Choose the verdict",
        "criteria": {"S": "Supports", "C": "Contradicts"}, "label": "C"})
    assert "S: Supports" in body and "C: Contradicts" in body
    assert gold == 1


def fake_cli_tokenizer(monkeypatch, argv):
    import sys
    from types import SimpleNamespace

    def tokenizer(name, local_files_only):
        assert local_files_only is True
        return CharTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer)))
    monkeypatch.setattr(sys, "argv", argv)


def test_scientific_cli_manifest_and_split_local_provenance(tmp_path, monkeypatch):
    source = tmp_path / "qa.jsonl"
    records = scientific_rows()
    records += [dict(records[0], pid="alias"),
                qa("bad", "The result is 10 units.", passage="The result is 100 units.")]
    source.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    output = tmp_path / "battery"
    fake_cli_tokenizer(monkeypatch, ["build_sci_decisions", "--qa", str(source), "--out", str(output),
                                     "--train-pids", "3", "--dev-pids", "2", "--eval-pids", "1"])
    sci.main()
    versioned = output / "systemone-v2"
    manifest = json.loads((versioned / "sci_manifest.json").read_text())
    assert manifest["counts"]["input_records"] == 8
    assert manifest["counts"]["split_records"] == {"train": 3, "dev": 2, "eval": 1}
    assert manifest["counts"]["exclusions_by_reason"]["duplicate_qa"] == 1
    assert manifest["counts"]["exclusions_by_reason"]["lexical_consistency_filter"] == 1
    assert manifest["inputs"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    groups = {}
    for tag in ("train", "dev", "eval"):
        ids = [row for kind in ("choice", "noul", "score")
               for row in rows_from(versioned / f"sci_{kind}_{tag}.jsonl")]
        texts = rows_from(versioned / f"sci_decisions_{tag}_text.jsonl")
        assert_paired(ids, texts)
        local = {row["pid"] for row in ids}
        groups[tag] = {row["split_group"] for row in ids}
        for row in ids:
            entries = row["negative_provenance"]
            for entry in entries if isinstance(entries, list) else [entries]:
                assert entry["source_pid"] in local and entry["split"] == tag
    assert groups["train"].isdisjoint(groups["dev"] | groups["eval"])
    assert groups["dev"].isdisjoint(groups["eval"])
    with pytest.raises(FileExistsError):
        sci.main()


def test_toolcall_cli_manifest_and_split_local_provenance(tmp_path, monkeypatch):
    fase = [tool_record(str(i), f"training_{i}") for i in range(6)]
    for name, records in (("toolcalls_fase3_500", fase),
                          ("toolcalls_lit_gold", [tool_record("e1", "EVAL_ONLY_1")]),
                          ("toolcalls_lit_evolucion", [tool_record("e2", "EVAL_ONLY_2")]),
                          ("toolcalls_lit_pilot4", [tool_record("e3", "EVAL_ONLY_3")])):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
    output = tmp_path / "battery"
    fake_cli_tokenizer(monkeypatch, ["build_toolcall_decisions", "--l1-dir", str(tmp_path),
                                     "--out", str(output), "--dev-frac", "0.5"])
    tools.main()
    versioned = output / "systemone-v2"
    manifest = json.loads((versioned / "toolcall_manifest.json").read_text())
    assert manifest["counts"]["input_records"] == 9
    assert len(manifest["inputs"]) == 4
    for tag in ("train", "dev", "eval"):
        ids = [row for kind in ("choice", "noul", "score")
               for row in rows_from(versioned / f"toolcall_{kind}_{tag}.jsonl")]
        texts = rows_from(versioned / f"toolcall_decisions_{tag}_text.jsonl")
        assert_paired(ids, texts)
        local = {row["pid"] for row in ids}
        assert all(row["negative_provenance"]["source_pid"] in local for row in ids)
        if tag != "eval":
            assert "EVAL_ONLY" not in json.dumps(texts)
    with pytest.raises(FileExistsError):
        tools.main()


def test_classification_duplicate_records_are_explicitly_excluded(tmp_path):
    src = tmp_path / "classification.jsonl"
    source_row = {"text": "A scientific article", "label": 3}
    src.write_text("\n".join(json.dumps(row) for row in (source_row, source_row,
                                                        {"text": "", "label": 0})) + "\n")
    cfg = dict(benchmarks.CLASSIF_CFGS["ag_news"], tag="ag_news")
    benchmarks.conv_classification(src, tmp_path / "classified.jsonl", CharTokenizer(), random.Random(1), cfg)
    output = tmp_path / "systemone-v2"
    rows = rows_from(output / "classified.jsonl")
    assert len(rows) == 1
    assert_paired(rows, rows_from(output / "classified_text.jsonl"))
    manifest = json.loads((output / "classified.manifest.json").read_text())
    assert manifest["counts"]["input_records"] == 3
    assert manifest["counts"]["exclusions_by_reason"] == {
        "duplicate_record": 1, "invalid_classification_text": 1}


def test_gpqa_pairs_and_exclusion_manifest(tmp_path):
    import csv

    src = tmp_path / "gpqa.csv"
    rows = [{"Question": "What follows?", "Correct Answer": "Reference answer",
             "Incorrect Answer 1": "Alternative a", "Incorrect Answer 2": "Alternative b",
             "Incorrect Answer 3": "Alternative c", "Subdomain": "Synthetic"}]
    rows += [dict(rows[0]), dict(rows[0], **{"Incorrect Answer 1": "Reference answer"})]
    with src.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    benchmarks.conv_gpqa(src, tmp_path / "gpqa.jsonl", CharTokenizer(), random.Random(2))
    output = tmp_path / "systemone-v2"
    ids = rows_from(output / "gpqa.jsonl")
    assert len(ids) == 1
    assert_paired(ids, rows_from(output / "gpqa_text.jsonl"))
    manifest = json.loads((output / "gpqa.manifest.json").read_text())
    assert manifest["counts"]["exclusions_by_reason"] == {"duplicate_options": 1, "duplicate_record": 1}


def test_scifact_duplicate_records_do_not_overwrite_text_questions(tmp_path):
    src = scifact_source(tmp_path, ("SUPPORT",))
    frame = pd.read_parquet(src)
    pd.concat([frame, frame]).to_parquet(src)
    benchmarks.conv_scifact(src, tmp_path / "sf", CharTokenizer(), random.Random(1))
    output = tmp_path / "systemone-v2"
    ids = rows_from(output / "sf_choice_eval.jsonl")
    assert len(ids) == 1
    assert_paired(ids, rows_from(output / "sf_choice_eval_text.jsonl"))
    manifest = json.loads((output / "sf_manifest.json").read_text())
    assert manifest["counts"]["exclusions_by_reason"] == {"duplicate_record": 1}


@pytest.mark.parametrize("builder,records", [
    (sci.build, scientific_rows()),
    (tools.build, [tool_record(str(i), str(i)) for i in range(6)]),
])
def test_qids_bind_to_actual_candidates_and_option_order(builder, records):
    first, _ = builder(records, CharTokenizer(), random.Random(3), "train")
    second, _ = builder(records, CharTokenizer(), random.Random(4), "train")
    features = {r["qid"]: (r["ctx"], r["opts"], r["kind"]) for r in first}
    for row in second:
        if row["qid"] in features:
            assert features[row["qid"]] == (row["ctx"], row["opts"], row["kind"])


@pytest.mark.parametrize("instruction", [sci.INSTR_CHOICE, sci.INSTR_NOUL])
def test_scientific_instructions_require_answering_the_actual_question(instruction):
    assert "question" in instruction.lower()
    assert "correct" in instruction.lower()
    assert "supplied passage" in instruction.lower()


def test_scientific_evidence_spans_preserve_question_and_proposed_answer():
    records = [qa(str(i), f"The result is {i + 1} units.",
                  passage=f"The α result is {i + 1} units. Question: decoy text.") for i in range(5)]
    ids, texts = sci.build(records, CharTokenizer(), random.Random(2), "eval")
    sources = {row["pid"]: row for row in records}
    questions = {qid: (text, q) for text in texts for qid, q in text["questions"].items()}
    for row in ids:
        text, question = questions[row["qid"]]
        source = sources[row["pid"]]
        start, end = row["evidence_span"]
        assert row["evidence_span"] == question["evidence_span"] == text["evidence_span"]
        assert text["state"][:start] == "Passage: "
        assert text["state"][start:end] == source["passage"]
        assert text["state"][end:].startswith(f"\nQuestion: {source['q']}")
        assert len(source["passage"].encode("utf-8")) > end - start
        if row["kind"] in ("noul", "score"):
            assert "\nProposed answer: " in text["state"][end:]
        if row["kind"] == "score":
            assert question["instructions"] == "Rate the proposed answer."
            assert question["criteria"] == sci.SCORE_LEGEND
    from reverse_jev.data import make_evidence_controls
    for source, controlled in zip(texts, make_evidence_controls(texts)):
        start, end = source["evidence_span"]
        assert controlled["state"] == source["state"][:start] + source["state"][end:]
    assert_paired(ids, texts)


def test_scifact_evidence_spans_cover_title_and_abstract_but_not_claim(tmp_path):
    source_rows = [{"claim": f"The actual claim {i}", "title": f"Étude {i}", "document_id": f"doc-{i}",
                    "abstract": [f"Observation {i}.", "Claim: a decoy inside the evidence."],
                    "verdict": "SUPPORT"} for i in range(2)]
    src = tmp_path / "validation.parquet"
    pd.DataFrame(source_rows).to_parquet(src)
    benchmarks.conv_scifact(src, tmp_path / "sf", CharTokenizer(), random.Random(1), legacy_score=True)
    output = tmp_path / "systemone-v2"
    from reverse_jev.data import make_evidence_controls
    for name in ("noul", "choice", "score_legacy_proxy"):
        ids = rows_from(output / f"sf_{name}_eval.jsonl")
        texts = rows_from(output / f"sf_{name}_eval_text.jsonl")
        questions = {qid: (text, q) for text in texts for qid, q in text["questions"].items()}
        for row in ids:
            text, question = questions[row["qid"]]
            source = source_rows[row["source_row"] - 1]
            expected = f"Title: {source['title']}\nAbstract: {' '.join(source['abstract'])}"
            assert row["evidence_span"] == text["evidence_span"] == question["evidence_span"] == [0, len(expected)]
            assert text["state"][:len(expected)] == expected
            assert text["state"][len(expected):] == f"\nClaim: {source['claim']}"
        for source, controlled in zip(texts, make_evidence_controls(texts, mode="shuffle", seed=2)):
            marker = controlled["evidence_control"]
            donor = texts[marker["donor_record_index"]]
            start, end = donor["evidence_span"]
            assert controlled["state"] == donor["state"][start:end] + source["state"][source["evidence_span"][1]:]
            assert marker["donor_group"] != marker["source_group"]
        assert_paired(ids, texts)


@pytest.mark.parametrize("field,left,right", [
    ("a", "The value is 1 mM.", "The value is 1 mm."),
    ("a", "Variable X increases.", "Variable x increases."),
    ("a", "Coordinate x₂ increases.", "Coordinate x2 increases."),
    ("q", "What is the value of X?", "What is the value of x?"),
])
def test_scientific_dedup_preserves_case_sensitive_and_compatibility_assertions(field, left, right):
    original = qa("shared", "The reference value is 10 units.",
                  passage="The record distinguishes mM, mm, X, x, x₂ and x2.")
    exclusions = []
    rows = sci.prepare_records([dict(original, **{field: left}), dict(original, **{field: right})], exclusions)
    assert len(rows) == 2
    assert {row[field] for row in rows} == {left, right}
    assert len({row["sample_id"] for row in rows}) == 2
    assert exclusions == []


def test_scientific_dedup_retains_canonical_unicode_and_whitespace_equivalence():
    original = qa("original", "The café value is 10 mM.", passage="Café evidence uses X and 1 mM.")
    equivalent = dict(original, pid="equivalent", passage="\tCafe\u0301 evidence uses X and 1 mM.\n",
                      q=original["q"].replace(" ", " \t "), a="The cafe\u0301  value is 10 mM.")
    exclusions = []
    rows = sci.prepare_records([original, equivalent], exclusions)
    assert len(rows) == 1
    assert [entry["reason"] for entry in exclusions] == ["duplicate_qa"]
    assert rows[0]["duplicate_sources"][0]["pid"] == "equivalent"


def test_lexical_word_overlap_remains_deliberately_case_insensitive():
    assert sci.grounded("What is the value?", "THE VALUE IS 10 UNITS.", "the value is 10 units.")
    assert sci._norm_num("1e2") == sci._norm_num("1E2") == sci._norm_num("100")
