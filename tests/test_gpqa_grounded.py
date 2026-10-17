import csv
import json
import sys

import pytest

from data import build_gpqa_grounded as grounded


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]


def decode(ids):
    return "".join(chr(i - 1) for i in ids)


def write_csv(path, rows):
    fields = ["Question", "Correct Answer", "Incorrect Answer 1",
              "Incorrect Answer 2", "Incorrect Answer 3", "Explanation",
              "Explanation_NEV_1", "Explanation_NEV_2", "Explanation_NEV_3",
              "Subdomain"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def csv_row(i, explanation="expert expl", nevs=("", "", "")):
    return {"Question": f"q{i}?", "Correct Answer": f"gold{i}",
            "Incorrect Answer 1": f"bad{i}a", "Incorrect Answer 2": f"bad{i}b",
            "Incorrect Answer 3": f"bad{i}c", "Explanation": explanation,
            "Explanation_NEV_1": nevs[0], "Explanation_NEV_2": nevs[1],
            "Explanation_NEV_3": nevs[2], "Subdomain": "Bio"}


def text_record(i):
    opts = [f"gold{i}", f"bad{i}a", f"bad{i}b", f"bad{i}c"]
    meta = {"split_group": f"question:g{i}", "group_id": f"question:g{i}",
            "pid": f"question:p{i}", "source": "gpqa", "source_row": i,
            "group_scope": "question", "content_hash": f"p{i}",
            "sample_id": f"s{i}", "reasoning_type": "parametric_scientific_qa",
            "meta": "Bio", "label_status": "source_reference",
            "evidence_regime": "no_passage_parametric", "task": "gpqa_choice"}
    question = {"type": "choice", "instructions": "Which answer is correct?",
                "criteria": {option: None for option in opts},
                "label": f"gold{i}"}
    return {**meta, "state": f"Question: q{i}?", "questions": {
        f"gpqa_s{i}_x": {**meta, **question, "kind": "choice",
                         "option_keys": opts, "decision_id": f"d{i}"}}}


def test_oracle_appends_explanation_and_preserves_pairing(tmp_path):
    csv_path = tmp_path / "gpqa.csv"
    write_csv(csv_path, [csv_row(1), csv_row(2)])
    records = [text_record(1), text_record(2)]
    tok = CharTokenizer()
    ids, texts, exclusions = grounded.build_grounded(
        records, [csv_row(1), csv_row(2)], tok, "oracle")
    assert not exclusions and len(ids) == 2
    for rec, row in zip(records, ids):
        assert row["pid"] == rec["pid"]  # pairs with the closed-book row
        assert row["opts"] == [tok.encode(k) for k in
                               next(iter(rec["questions"].values()))["criteria"]]
        assert row["gold"] == 0 and row["evidence_regime"] == "oracle_explanation"
        assert row["qid"].endswith("_gr-oracle")
        assert "Evidence: expert expl" in decode(row["ctx"])
        assert row["task"] == "gpqa_choice_grounded"


def test_nev_uses_first_nonempty_validator_trace(tmp_path):
    row_data = csv_row(1, nevs=("", "validator two saw it", "third"))
    ids, _, exclusions = grounded.build_grounded(
        [text_record(1)], [row_data], CharTokenizer(), "nev")
    assert not exclusions and len(ids) == 1
    assert "Evidence: validator two saw it" in decode(ids[0]["ctx"])
    assert ids[0]["evidence_field"] == "Explanation_NEV_2"


def test_nev_excludes_rows_without_any_trace(tmp_path):
    ids, _, exclusions = grounded.build_grounded(
        [text_record(1)], [csv_row(1, nevs=("", "", ""))],
        CharTokenizer(), "nev")
    assert ids == [] and exclusions[0]["reason"] == "no_nev_evidence"


def test_cli_writes_id_and_text_files(tmp_path, monkeypatch):
    csv_path = tmp_path / "gpqa.csv"
    write_csv(csv_path, [csv_row(1)])
    text_path = tmp_path / "clean_text.jsonl"
    text_path.write_text(json.dumps(text_record(1)) + "\n")
    out = tmp_path / "grounded.jsonl"
    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer",
                        type("T", (), {"from_pretrained": staticmethod(
                            lambda *a, **k: CharTokenizer())}))
    monkeypatch.setattr(sys, "argv", ["build_gpqa_grounded", "--csv", str(csv_path),
                                    "--text-file", str(text_path), "--arm", "oracle",
                                    "--out", str(out)])
    grounded.main()
    from sciev.data import versioned_path
    out = versioned_path(out)  # write_dataset routes through the encoding dir
    assert out.exists() and out.with_name("grounded_text.jsonl").exists()
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["evidence_arm"] == "oracle"
    assert manifest["counts"]["rows"] == 1
