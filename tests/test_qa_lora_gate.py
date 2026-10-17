from copy import deepcopy

import pytest

from scripts.qa_lora_gate import decide


def report(acc=0.32):
    correct = round(acc * 440)
    return {"protocol_version": 2, "role": "evaluation", "input_file": {"sha256": "clean"},
            "training_overlap": {"status": "verified"},
            "decisions_eval": {"n": 440, "acc": correct / 440,
                               "predictions": [{"prediction": 0 if index < correct else 1, "gold": 0}
                                               for index in range(440)]}}


def test_lora_gate_waits_for_all_predefined_frozen_head_results():
    assert decide([report(.32), report(.34), report(.31)], "clean")["launch"]
    assert not decide([report(.32), report(.39), report(.31)], "clean")["launch"]


@pytest.mark.parametrize("field,value", [("protocol_version", 1), ("role", "acceptance_development"),
                                         ("training_overlap", {"status": "unverified"}),
                                         ("input_file", {"sha256": "old_441_cohort"})])
def test_lora_gate_rejects_incomparable_reports(field, value):
    altered = deepcopy(report())
    altered[field] = value
    with pytest.raises(ValueError, match="verified evaluations"):
        decide([altered], "clean")


def test_lora_gate_rejects_partial_results():
    altered = report()
    altered["decisions_eval"]["predictions"].pop()
    with pytest.raises(ValueError, match="complete"):
        decide([altered], "clean")
