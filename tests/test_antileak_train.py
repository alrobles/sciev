import json
import sys

import pytest

from data import build_antileak_train as antileak


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]


def decode(ids):
    return "".join(chr(i - 1) for i in ids)


def record(i, kind="noul", label="true", evidence=None, criteria="default"):
    evidence = evidence or f"Evidence passage for record {i}."
    state = f"Passage: {evidence} Claim: claim {i}?"
    start = state.index(evidence)
    span = [start, start + len(evidence)]
    if criteria == "default":
        criteria = ({"true": "supported by the passage",
                     "false": "not supported by the passage"} if kind == "noul"
                    else ["unrelated", "partial", "correct"])
    return {"split_group": f"g{i}", "source": "sci", "sample_id": f"s{i}",
            "split": "train", "state": state, "evidence_span": span,
            "questions": {f"q{i}": {"type": kind,
                                    "instructions": "check it",
                                    "criteria": criteria,
                                    "label": label}}}


def test_noul_empty_controls_remap_gold_to_reject():
    rows = [record(1, label="true"), record(2, label="false")]
    ids, texts, per_mode = antileak.build_antileak(
        rows, CharTokenizer(), kind="noul", modes=["empty"])
    assert per_mode == {"empty": 2} and len(ids) == 2
    for row in ids:
        assert row["kind"] == "noul" and row["gold"] == 1  # "no"
        assert row["gold_semantics"] == "reject_on_unsupported"
        assert row["label_status"] == "antileak_train"
        assert "Evidence passage" not in decode(row["ctx"])
        assert row["evidence_control"]["mode"] == "empty"


def test_score_shuffle_uses_lowest_level_and_different_donor():
    rows = [record(1, kind="score", evidence="Alpha passage."),
            record(2, kind="score", evidence="Beta passage.")]
    ids, _, per_mode = antileak.build_antileak(
        rows, CharTokenizer(), kind="score", modes=["shuffle"], seed=7)
    assert per_mode == {"shuffle": 2}
    donors = {row["evidence_control"]["donor_group"] for row in ids}
    groups = {row["split_group"] for row in ids}
    assert donors == groups  # shuffled evidence must come from another group
    for row in ids:
        assert row["gold"] == 0  # lowest rubric level
        assert row["gold_semantics"] == "reject_lowest_rubric"


def test_choice_questions_are_not_emitted():
    # a choice question must not yield an anti-leak row (no reject gold)
    rec = {"split_group": "gc", "source": "sci", "sample_id": "sc",
           "split": "train", "state": "Passage: x Claim: y?",
           "evidence_span": [9, 10],
           "questions": {"qc": {"type": "choice",
                                "instructions": "pick",
                                "criteria": {"a": None, "b": None},
                                "label": "a"}}}
    rows = [rec, record(2)]
    ids, _, per_mode = antileak.build_antileak(
        rows, CharTokenizer(), kind="noul", modes=["empty"])
    # only the noul question of record 2 yields a row
    assert per_mode == {"empty": 1} and len(ids) == 1
    assert ids[0]["sample_id"] == "s2"


def test_noul_accepts_none_criteria():
    # real sci rows carry criteria=None (default yes/no encoding) + bool label
    rows = [record(1, criteria=None), record(2, criteria=None)]
    ids, _, per_mode = antileak.build_antileak(
        rows, CharTokenizer(), kind="noul", modes=["empty"])
    assert per_mode == {"empty": 2} and len(ids) == 2
    assert all(row["gold"] == 1 for row in ids)


def test_rejects_unknown_kind_and_bad_criteria():
    with pytest.raises(ValueError, match="kind"):
        antileak.build_antileak([record(1)], CharTokenizer(),
                                kind="choice", modes=["empty"])
    bad = record(3)
    bad["questions"]["q3"]["criteria"] = {"maybe": "x", "true": "y"}
    ids, _, per_mode = antileak.build_antileak(
        [bad, record(4)], CharTokenizer(), kind="noul", modes=["empty"])
    assert per_mode == {"empty": 1}  # bad criteria row excluded
