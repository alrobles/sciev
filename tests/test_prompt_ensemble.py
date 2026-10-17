import json

import pytest

from scripts import ensemble_instr_variants as ensemble


def prediction(group="q1", keys=None, probs=None, gold=0):
    keys = keys or ["red", "blue", "green", "white"]
    probs = probs or [0.7, 0.1, 0.1, 0.1]
    pred = max(range(len(keys)), key=lambda i: probs[i])
    return {"split_group": group, "option_keys": keys, "gold": gold,
            "prediction": pred, "probabilities": probs, "correct": int(pred == gold),
            "max_probability": max(probs)}


def report(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(json.dumps({"decisions_eval": {"n": len(rows), "predictions": rows}}))
    return str(path)


def test_ensemble_rejects_duplicate_item_identity(tmp_path):
    path = report(tmp_path, "duplicate.json", [prediction(), prediction()])
    with pytest.raises(ValueError, match="duplicate"):
        ensemble.load_preds(path)


def test_ensemble_aligns_option_content_not_position(tmp_path):
    a = report(tmp_path, "a.json", [prediction()])
    b = report(tmp_path, "b.json", [prediction(keys=["white", "red", "blue", "green"],
                                               probs=[0.1, 0.7, 0.1, 0.1], gold=1)])
    output = ensemble.aggregate([a, b])
    assert output["ensemble"]["acc"] == 1
    assert output["ensemble"]["nll"] == pytest.approx(-__import__("math").log(0.7))
    assert output["ensemble"]["full_agreement_rate"] == 1
    assert output["ensemble"]["predictions"][0]["probabilities"] == pytest.approx([0.7, 0.1, 0.1, 0.1])


def test_ensemble_rejects_silent_cohort_changes(tmp_path):
    a = report(tmp_path, "a.json", [prediction(), prediction("q2")])
    b = report(tmp_path, "b.json", [prediction()])
    with pytest.raises(ValueError, match="cohort"):
        ensemble.aggregate([a, b])
    output = ensemble.aggregate([a, b], allow_intersection=True)
    assert output["n_common_items"] == 1
    assert output["per_variant"][0]["excluded_from_common"] == 1


def test_ensemble_rejects_empty_intersection_and_conflicting_gold(tmp_path):
    a = report(tmp_path, "a.json", [prediction()])
    b = report(tmp_path, "b.json", [prediction("q2")])
    with pytest.raises(ValueError, match="common"):
        ensemble.aggregate([a, b], allow_intersection=True)
    c = report(tmp_path, "c.json", [prediction(gold=1)])
    with pytest.raises(ValueError, match="gold"):
        ensemble.aggregate([a, c])
