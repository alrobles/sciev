import copy
import json
import math
from fractions import Fraction
from itertools import permutations

import numpy as np
import pytest

from sciev.metrics import (
    classification_metrics,
    evaluate_acceptance_policy,
    fit_acceptance_policy,
    oracle_automation_rate,
)


def test_ragged_classification_metrics_do_not_invent_semantic_classes():
    result = classification_metrics([[0.8, 0.2], [0.1, 0.4, 0.5]], [0, 1])
    assert result["n"] == 2
    assert result["acc"] == 0.5
    assert result["nll"] == pytest.approx(-(math.log(0.8) + math.log(0.4)) / 2)
    assert result["brier"] == pytest.approx(0.35)
    assert result["ece"] == pytest.approx(0.35)
    assert result["oracle_automation_5pct"] == 0.5
    assert result["nll_zero_gold_count"] == 0
    for key in ("class_support", "majority_baseline_acc", "balanced_acc",
                "macro_f1", "expected_level_mae", "argmax_mae"):
        assert key not in result
    json.dumps(result, allow_nan=False)


def test_equal_option_counts_do_not_imply_semantic_labels():
    result = classification_metrics([[0.1, 0.9]] * 3, [0, 1, 1])
    assert "majority_baseline_acc" not in result
    assert "class_support" not in result


def test_ece_uses_weighted_ten_bin_calibration_gaps():
    result = classification_metrics(
        [[0.65, 0.35], [0.69, 0.31], [0.8, 0.2]], [0, 1, 0])
    assert result["ece"] == pytest.approx(0.18)


def test_imbalanced_fixed_label_metrics_include_majority_baseline():
    result = classification_metrics(
        [[0.1, 0.9]] * 6, [0, 1, 1, 0, 1, 1], fixed_labels=True)
    assert result["class_support"] == [2, 4]
    assert result["majority_baseline_acc"] == pytest.approx(2 / 3)
    assert result["acc"] == pytest.approx(2 / 3)
    assert result["balanced_acc"] == 0.5
    assert result["macro_f1"] == pytest.approx(0.4)


def test_fixed_label_metrics_define_absent_true_class_handling():
    result = classification_metrics(
        [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]], [0, 1], fixed_labels=True)
    assert result["class_support"] == [1, 1, 0]
    assert result["balanced_acc"] == 1.0
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["balanced_acc_label_scope"] == "positive_true_support"
    assert result["macro_f1_label_scope"] == "all_fixed_labels"
    assert result["macro_f1_zero_division"] == 0.0
    json.dumps(result, allow_nan=False)


def test_ordinal_metrics_use_expected_level_and_argmax_distance():
    result = classification_metrics(
        [[0.6, 0.0, 0.4], [0.0, 0.2, 0.8]], [1, 0],
        fixed_labels=True, ordinal=True)
    assert result["ordinal_levels"] == [0, 1, 2]
    assert result["expected_level_mae"] == pytest.approx(1.0)
    assert result["argmax_mae"] == 1.5
    assert result["acc"] == 0.0
    json.dumps(result, allow_nan=False)


def test_explicit_tie_predictions_drive_selection_metrics_only():
    rows = [[0.4, 0.4, 0.2]] * 3
    golds = [1, 1, 0]
    predictions = np.array([1, 1, 0], dtype=np.int64)
    original_rows = copy.deepcopy(rows)
    original_predictions = predictions.copy()
    default = classification_metrics(rows, golds, fixed_labels=True, ordinal=True)
    selected = classification_metrics(
        rows, golds, fixed_labels=True, ordinal=True, predictions=predictions)
    assert default["acc"] == pytest.approx(1 / 3)
    assert selected["acc"] == 1.0
    assert default["ece"] == pytest.approx(1 / 15)
    assert selected["ece"] == pytest.approx(0.6)
    assert default["balanced_acc"] == 0.5
    assert selected["balanced_acc"] == 1.0
    assert default["macro_f1"] == pytest.approx(1 / 6)
    assert selected["macro_f1"] == pytest.approx(2 / 3)
    assert default["argmax_mae"] == pytest.approx(2 / 3)
    assert selected["argmax_mae"] == 0.0
    assert default["oracle_automation_5pct"] == 0.0
    assert selected["oracle_automation_5pct"] == 1.0
    for key in ("brier", "nll", "expected_level_mae", "class_support",
                "majority_baseline_acc"):
        assert selected[key] == default[key]
    assert rows == original_rows
    assert np.array_equal(predictions, original_predictions)
    json.dumps(selected, allow_nan=False)


def test_supplied_predictions_preserve_canonical_tie_identity_across_option_orders():
    default_accuracies = set()
    for option_order in permutations(["alpha", "beta", "gamma"]):
        prediction = option_order.index("alpha")
        gold = option_order.index("alpha")
        rows = [[1 / 3] * 3]
        result = classification_metrics(rows, [gold], predictions=[prediction])
        assert result["acc"] == 1.0
        assert result["ece"] == pytest.approx(2 / 3)
        assert result["oracle_automation_5pct"] == 1.0
        default_accuracies.add(classification_metrics(rows, [gold])["acc"])
    assert default_accuracies == {0.0, 1.0}


def test_explicit_predictions_support_ragged_rows():
    result = classification_metrics(
        [[0.5, 0.5], [0.2, 0.4, 0.4]], [1, 2], predictions=(1, 2))
    assert result["acc"] == 1.0
    assert result["ece"] == pytest.approx(0.55)
    assert result["oracle_automation_5pct"] == 1.0


def test_none_or_matching_predictions_preserve_default_argmax():
    rows, golds = [[0.5, 0.5], [0.2, 0.8]], [1, 0]
    default = classification_metrics(rows, golds)
    assert default["acc"] == 0.0
    assert classification_metrics(rows, golds, predictions=None) == default
    assert classification_metrics(rows, golds, predictions=[0, 1]) == default


@pytest.mark.parametrize("predictions", [
    [], [0, 1], [True], [np.bool_(False)], [0.0], ["0"], [None],
    [float("nan")], [float("inf")], [-1], [2], [2 ** 70], [1j],
    [[0]], np.array([[0]]), np.array(0), 0, "0", {0}, frozenset({0}), {0: 0},
])
def test_classification_rejects_invalid_predictions(predictions):
    with pytest.raises(ValueError, match="predictions"):
        classification_metrics([[0.5, 0.5]], [0], predictions=predictions)


@pytest.mark.parametrize("row,prediction", [
    ([0.9, 0.1], 1),
    ([0.45, 0.45, 0.1], 2),
    ([np.nextafter(0.5, 1.0), np.nextafter(0.5, 0.0)], 1),
])
def test_supplied_predictions_must_be_exact_probability_maxima(row, prediction):
    with pytest.raises(ValueError, match="maximum"):
        classification_metrics([row], [0], predictions=[prediction])


def test_ordinal_metrics_require_explicit_fixed_label_semantics():
    with pytest.raises(ValueError, match="fixed_labels"):
        classification_metrics([[0.5, 0.5]], [0], ordinal=True)


@pytest.mark.parametrize("ordinal", [False, True])
def test_fixed_labels_reject_ragged_label_spaces(ordinal):
    with pytest.raises(ValueError, match="fixed|consistent|same"):
        classification_metrics(
            [[0.8, 0.2], [0.1, 0.4, 0.5]], [0, 1],
            fixed_labels=True, ordinal=ordinal)


@pytest.mark.parametrize("flags", [
    {"ordinal": "true"}, {"ordinal": 1}, {"fixed_labels": "false"},
    {"fixed_labels": None},
])
def test_classification_rejects_nonboolean_flags(flags):
    with pytest.raises(ValueError):
        classification_metrics([[0.5, 0.5]], [0], **flags)


@pytest.mark.parametrize("rows,golds", [
    ([], []),
    ([[0.5, 0.5]], []),
    ([[0.5, 0.5]], [0, 1]),
    ([[1.0]], [0]),
    ([[]], [0]),
    ([[0.2, 0.2]], [0]),
    ([[0.6, 0.5]], [0]),
    ([[-0.1, 1.1]], [0]),
    ([[float("nan"), 0.5]], [0]),
    ([[float("inf"), 0.0]], [0]),
    ([[None, 1.0]], [0]),
    ([["0.5", "0.5"]], [0]),
    ([[0.5 + 0j, 0.5]], [0]),
    ([[True, False]], [0]),
    ([[[0.5, 0.5]]], [0]),
    ([0.5, 0.5], [0, 1]),
    (None, [0]),
    ({"row": [0.5, 0.5]}, [0]),
    ([[0.5, 0.5]], None),
    ([[0.5, 0.5]], [True]),
    ([[0.5, 0.5]], [np.bool_(False)]),
    ([[0.5, 0.5]], [0.0]),
    ([[0.5, 0.5]], ["0"]),
    ([[0.5, 0.5]], [2]),
    ([[0.5, 0.5]], [-1]),
    ([[0.5, 0.5]], [float("nan")]),
    ([[0.5, 0.5]], [None]),
    ([[0.5, 0.5]], [[0]]),
])
def test_classification_rejects_invalid_distributions_and_labels(rows, golds):
    with pytest.raises(ValueError):
        classification_metrics(rows, golds)


def test_near_normalized_probabilities_are_not_silently_changed():
    rows = [[0.8, 0.2000004]]
    original = copy.deepcopy(rows)
    result = classification_metrics(rows, np.array([1], dtype=np.int64))
    assert result["nll"] == pytest.approx(-math.log(0.2000004), abs=1e-12)
    assert result["brier"] == pytest.approx(0.8 ** 2 + 0.7999996 ** 2, abs=1e-12)
    assert rows == original


def test_zero_gold_probability_has_json_safe_explicit_infinite_nll():
    result = classification_metrics([[1.0, 0.0]], [1], fixed_labels=True)
    assert result["nll"] is None
    assert result["nll_zero_gold_count"] == 1
    assert result["brier"] == 2.0
    assert result["ece"] == 1.0
    assert result["oracle_automation_5pct"] == 0.0
    json.dumps(result, allow_nan=False)


def test_positive_tiny_gold_probability_is_not_clipped():
    tiny = np.nextafter(0.0, 1.0)
    result = classification_metrics([[tiny, 1.0]], [0])
    assert result["nll"] == pytest.approx(-math.log(tiny))
    assert result["nll_zero_gold_count"] == 0


def test_tied_confidences_cannot_cherry_pick_a_correct_prefix():
    confs, corrects = [0.9] * 4, [1, 1, 1, 0]
    prefix_risks = np.cumsum(1 - np.asarray(corrects)) / np.arange(1, 5)
    old_prefix_coverage = (np.flatnonzero(prefix_risks <= 0.05)[-1] + 1) / 4
    assert old_prefix_coverage == 0.75
    assert oracle_automation_rate(confs, corrects) == 0.0
    policy = fit_acceptance_policy(confs, corrects, min_accepted=1)
    assert policy["threshold"] is None
    result = classification_metrics([[0.9, 0.1]] * 4, [0, 0, 0, 1])
    assert result["oracle_automation_5pct"] == 0.0


@pytest.mark.parametrize("budget,accepted,threshold", [
    (0.05, 1, 0.95), (0.25, 4, 0.9),
])
def test_policies_are_permutation_invariant_and_include_whole_ties(
        budget, accepted, threshold):
    reference = None
    for tied_corrects in sorted(set(permutations([1, 1, 0]))):
        confs = [0.95, 0.9, 0.9, 0.9, 0.8]
        corrects = [1, *tied_corrects, 0]
        policy = fit_acceptance_policy(
            confs, corrects, error_budget=budget, min_accepted=1)
        assert policy["threshold"] == threshold
        assert policy["dev_accepted"] == accepted
        assert policy["dev_coverage"] == accepted / 5
        assert oracle_automation_rate(confs, corrects, budget) == accepted / 5
        evaluated = evaluate_acceptance_policy(confs, corrects, policy)
        assert evaluated["accepted"] == accepted
        assert evaluated["risk"] == policy["dev_risk"]
        if reference is not None:
            assert policy == reference
        reference = policy


def test_maximum_coverage_can_be_feasible_after_an_infeasible_prefix():
    confs = [0.99, 0.98] + [0.8] * 18 + [0.6]
    corrects = [0] + [1] * 19 + [0]
    policy = fit_acceptance_policy(confs, corrects)
    assert policy["threshold"] == 0.8
    assert policy["dev_accepted"] == 20
    assert policy["dev_errors"] == 1
    assert policy["dev_risk"] == 0.05
    assert policy["dev_coverage"] == 20 / 21


def test_default_minimum_support_can_prevent_otherwise_feasible_acceptance():
    confs, corrects = [0.95] * 19 + [0.8] * 2, [1] * 19 + [0] * 2
    policy = fit_acceptance_policy(confs, corrects)
    assert policy["min_accepted"] == 20
    assert policy["threshold"] is None
    assert policy["dev_n"] == 21
    assert policy["dev_accepted"] == policy["dev_errors"] == 0
    assert policy["dev_coverage"] == 0.0
    assert policy["dev_risk"] is None
    supported = fit_acceptance_policy(confs, corrects, min_accepted=19)
    assert supported["threshold"] == 0.95
    assert supported["dev_accepted"] == 19


def test_no_empirically_feasible_policy_accepts_none():
    policy = fit_acceptance_policy([0.99, 0.8], [0, 1], min_accepted=1)
    assert policy["threshold"] is None
    assert policy["dev_risk"] is None
    assert oracle_automation_rate([0.99, 0.8], [0, 1]) == 0.0
    json.dumps(policy, allow_nan=False)


def test_policy_metadata_is_serializable_and_disclaims_a_guarantee():
    policy = fit_acceptance_policy(
        np.array([1.0, 0.8]), np.array([True, False]),
        error_budget=np.float64(0.5), min_accepted=np.int64(2))
    assert policy["method"] == "empirical_dev_threshold"
    assert policy["version"] == 1
    assert policy["score"] == "max_probability"
    assert policy["comparison"] == ">="
    assert policy["guarantee"] is False
    assert "not a statistical deployment" in policy["caveat"]
    assert policy["threshold"] == 0.8
    assert policy["dev_n"] == policy["dev_accepted"] == 2
    assert policy["dev_errors"] == 1
    assert policy["dev_risk"] == policy["error_budget"] == 0.5
    assert json.loads(json.dumps(policy, allow_nan=False)) == policy


@pytest.mark.parametrize("budget,threshold,accepted", [
    (0.0, 1.0, 1), (1.0, 0.0, 2),
])
def test_budget_and_threshold_endpoints_are_inclusive(budget, threshold, accepted):
    policy = fit_acceptance_policy(
        [1.0, 0.0], [1.0, 0.0], error_budget=budget, min_accepted=1)
    assert policy["threshold"] == threshold
    result = evaluate_acceptance_policy([1.0, 0.0], [1, 0], policy)
    assert result["accepted"] == accepted


@pytest.fixture
def fitted_policy():
    return fit_acceptance_policy([0.8] * 20, [1] * 20)


@pytest.mark.parametrize("confs,corrects", [
    ([0.9], [1, 0]),
    ([float("nan")], [1]),
    ([float("inf")], [1]),
    ([-0.1], [1]),
    ([1.1], [1]),
    (["0.9"], [1]),
    ([None], [1]),
    ([[0.9]], [1]),
    (0.9, [1]),
    ([0.9], [0.5]),
    ([0.9], [Fraction(1, 10 ** 400)]),
    ([0.9], [Fraction(10 ** 40 - 1, 10 ** 40)]),
    ([0.9], [float("nan")]),
    ([0.9], [float("inf")]),
    ([0.9], [-1]),
    ([0.9], [2]),
    ([0.9], ["1"]),
    ([0.9], [1j]),
    ([0.9], [[1]]),
    ([0.9], None),
])
def test_acceptance_functions_validate_vectors(confs, corrects, fitted_policy):
    with pytest.raises(ValueError):
        fit_acceptance_policy(confs, corrects)
    with pytest.raises(ValueError):
        oracle_automation_rate(confs, corrects)
    with pytest.raises(ValueError):
        evaluate_acceptance_policy(confs, corrects, fitted_policy)


def test_fitting_and_oracle_require_nonempty_data():
    with pytest.raises(ValueError):
        fit_acceptance_policy([], [])
    with pytest.raises(ValueError):
        oracle_automation_rate([], [])


@pytest.mark.parametrize("budget", [-0.1, 1.1, float("nan"), float("inf"),
                                    "0.05", True, None])
def test_acceptance_functions_validate_error_budget(budget):
    with pytest.raises(ValueError):
        fit_acceptance_policy([0.9], [1], error_budget=budget)
    with pytest.raises(ValueError):
        oracle_automation_rate([0.9], [1], budget)


@pytest.mark.parametrize("minimum", [0, -1, 1.5, True, "20", None,
                                     float("nan"), np.bool_(True)])
def test_fit_validates_minimum_support(minimum):
    with pytest.raises(ValueError):
        fit_acceptance_policy([0.9], [1], min_accepted=minimum)


@pytest.mark.parametrize("updates", [
    {"threshold": float("nan")}, {"threshold": float("inf")},
    {"threshold": -0.1}, {"threshold": 1.1}, {"threshold": "0.8"},
    {"threshold": []}, {"threshold": True}, {"threshold": np.array(0.8)},
    {"score": "confidence"}, {"score": None},
    {"method": "oracle"}, {"version": 2}, {"version": True},
    {"comparison": ">"},
])
def test_evaluation_rejects_malformed_policy_values(updates, fitted_policy):
    fitted_policy.update(updates)
    with pytest.raises(ValueError):
        evaluate_acceptance_policy([0.9], [1], fitted_policy)


@pytest.mark.parametrize("missing", ["threshold", "score", "method", "version",
                                     "comparison"])
def test_evaluation_rejects_missing_policy_contract_fields(missing, fitted_policy):
    del fitted_policy[missing]
    with pytest.raises(ValueError):
        evaluate_acceptance_policy([0.9], [1], fitted_policy)


@pytest.mark.parametrize("policy", [None, [], 0, "0.8"])
def test_evaluation_rejects_nonmapping_policies(policy):
    with pytest.raises(ValueError):
        evaluate_acceptance_policy([0.9], [1], policy)


def test_test_risk_can_worsen_without_refitting_or_mutating_policy(fitted_policy):
    original = copy.deepcopy(fitted_policy)
    confs = [0.9] * 20 + [0.7] * 5
    result = evaluate_acceptance_policy(confs, [0] * 25, fitted_policy)
    alternative = evaluate_acceptance_policy(confs, [1] * 25, fitted_policy)
    assert fitted_policy["dev_risk"] <= fitted_policy["error_budget"]
    assert result["n"] == 25
    assert result["accepted"] == alternative["accepted"] == 20
    assert result["coverage"] == alternative["coverage"] == 0.8
    assert result["errors"] == 20
    assert result["risk"] == 1.0
    assert alternative["risk"] == 0.0
    assert fitted_policy == original


@pytest.mark.parametrize("errors,n,lower,upper", [
    (0, 20, 0.0, 0.16112515805281935),
    (5, 100, 0.02154367915436796, 0.11175046923191913),
    (20, 20, 0.8388748419471806, 1.0),
])
def test_wilson_interval_describes_accepted_binomial_error_rate(
        errors, n, lower, upper, fitted_policy):
    result = evaluate_acceptance_policy(
        [0.8] * n, [0] * errors + [1] * (n - errors), fitted_policy)
    interval = result["risk_ci95"]
    assert result["risk"] == errors / n
    assert interval["method"] == "wilson"
    assert interval["confidence_level"] == 0.95
    assert interval["lower"] == pytest.approx(lower)
    assert interval["upper"] == pytest.approx(upper)
    assert "Descriptive" in interval["description"]
    assert "independent" in interval["description"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("accept_none_policy", [True, False])
def test_empty_accepted_set_has_no_risk_or_interval(accept_none_policy):
    policy = fit_acceptance_policy(
        [0.95], [0 if accept_none_policy else 1], min_accepted=1)
    result = evaluate_acceptance_policy([0.9, 0.8], [0, 1], policy)
    assert result["n"] == 2
    assert result["accepted"] == result["errors"] == 0
    assert result["coverage"] == 0.0
    assert result["risk"] is None
    assert result["risk_ci95"]["lower"] is None
    assert result["risk_ci95"]["upper"] is None
    json.dumps(result, allow_nan=False)


def test_empty_evaluation_has_explicit_undefined_coverage(fitted_policy):
    result = evaluate_acceptance_policy([], [], fitted_policy)
    assert result["n"] == result["accepted"] == result["errors"] == 0
    assert result["coverage"] is None
    assert result["risk"] is None
    assert result["risk_ci95"]["lower"] is None
    assert result["risk_ci95"]["upper"] is None
    json.dumps(result, allow_nan=False)
